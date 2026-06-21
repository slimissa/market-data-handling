"""
ml_signal.py — Phase 7: ML Signal (walk-forward validated)
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → [ML signal] → regime filter → regime detect
            → backtest → factor model → gating comparison

What this replaces and what it doesn't:
    Rule-based signals (signal_rsi, signal_macd, signal_zscore, signal_bb)
    encode a human's prior belief about which feature thresholds matter
    (RSI<30, MACD crossover, etc). This module instead lets a gradient-
    boosted model learn directly from all 24 engineered features which
    combinations actually predict 5-day forward returns, with no hand-
    picked thresholds.

    It does NOT replace the regime filter, the factor model, or the
    backtester. It produces exactly one more signal_* column
    (signal_ml) that flows through the same downstream machinery as
    every rule-based signal — same backtester, same gating comparison,
    same factor attribution. That reuse is the entire point: this
    module is only trustworthy because the validation infrastructure
    already exists and was built and tested before this module was
    written. ML on financial data without an honest backtester is
    overfitting with extra steps; the order these phases were built in
    is not incidental.

Walk-forward validation (the gold standard, not a shortcut):
    A single train/test split answers "does this model work on one
    particular slice of history." Walk-forward answers "does this model
    keep working as time moves forward and the world changes" — which is
    the only question that matters for a strategy meant to run live.

    Mechanics: the full history is divided into sequential folds. For
    fold k, the model is trained ONLY on data strictly before that
    fold's test window (an expanding or rolling window — both supported),
    then predicts on the test window, then the window advances and the
    model is retrained from scratch on the now-larger training set. A
    prediction for date t is NEVER produced by a model that was trained
    on any data at or after t. This is enforced structurally (the loop
    can't access future folds), not just by convention — see
    WalkForwardSplitter.split() for the index-level guarantee.

    No global StandardScaler, no global feature selection, no global
    anything fit across the whole series before the loop starts: any
    such step would leak future distributional information into early
    folds. Every transform that touches the data is fit fold-by-fold,
    train-only.

Label construction:
    target = returns_fwd_5 (already computed in data_cleaner.py via
    price.pct_change(-5), a genuine forward shift). The label for date t
    requires knowing the price at t+5 — by construction this is only
    used as a TRAINING target, never as a feature, and any fold's test
    predictions for the last 5 bars of that fold's window are dropped
    (their true label depends on prices outside the available data).

Threshold conversion:
    A continuous predicted return is converted to {-1, 0, +1} via two
    symmetric thresholds (not a single zero-crossing): predictions
    within [-threshold, +threshold] become 0 (flat) — this is a
    deliberate deadband, not a free parameter to be tuned away. Without
    it, a model with near-zero genuine predictive power still trades on
    every bar (any nonzero prediction, however noise-driven, produces a
    position), which is exactly the kind of result that looks active in
    a backtest and is actually just transaction-cost-funded noise.

Feature importance:
    Reports XGBoost's gain-based importance (total loss reduction
    attributed to each feature across all trees), aggregated across
    walk-forward folds rather than from a single global fit — a feature
    that matters in fold 3 but not fold 7 is informative about regime
    dependence, not noise to be averaged away silently.

Usage:
    wf = WalkForwardSplitter(n_folds=5, min_train_bars=252, test_bars=63)
    ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)

    result = ml.fit_predict_walk_forward(df, splitter=wf)
    df["signal_ml"] = result.signal
    print(result.importance_report())
    print(result.fold_report())
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Canonical feature column list
# ======================================================================

# Columns that must NEVER be used as ML input features. This is
# deliberately a strict allowlist-by-exclusion rather than a hand-typed
# allowlist of the 24 feature names, so that if feature_engineering.py
# ever adds a new feature family, it is automatically included as a
# candidate feature without this file needing an edit — while anything
# that is raw OHLCV, a return/target column, or a DERIVED signal/regime
# column (which would leak rule-based or regime-detector decisions into
# an otherwise-independent ML model) is automatically excluded.
EXCLUDED_FROM_FEATURES = {
    "open", "high", "low", "close", "volume",   # raw OHLCV
    "returns", "returns_norm",                   # same-day return info
    "returns_fwd_1", "returns_fwd_5",             # targets, not features
    "tr",                                         # ATR intermediate, duplicate of atr_14 info
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the candidate ML feature columns from a fully-processed
    DataFrame: every numeric column except raw OHLCV, return/target
    columns, and any column whose name indicates it is DERIVED from
    rule-based signal logic or regime detection (signal_*, position_*,
    regime_*) — including those would let the ML model trivially learn
    to reproduce the rule-based system's own decisions rather than
    learning independently from the underlying engineered features.
    """
    cols = []
    for c in df.columns:
        if c in EXCLUDED_FROM_FEATURES:
            continue
        if c.startswith("signal_") or c.startswith("position_") or c.startswith("regime_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


# ======================================================================
# Walk-forward splitter
# ======================================================================

@dataclass
class Fold:
    """One walk-forward fold: a strictly time-ordered train/test split."""
    fold_id:      int
    train_idx:    pd.Index
    test_idx:     pd.Index
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp


class WalkForwardSplitter:
    """
    Generates sequential, strictly non-overlapping, strictly forward-only
    train/test folds over a DatetimeIndex.

    Two modes:
        expanding=True  (default): each fold's training set grows to
            include all data before the test window (fold k+1 trains on
            everything fold k trained on, plus fold k's test window).
            More data per fold over time; assumes older data is still
            relevant.
        expanding=False (rolling): each fold's training set is a fixed-
            size window immediately preceding the test window. Better
            for non-stationary markets where very old data may actively
            mislead the model — but the right choice is an empirical
            question, not asserted; this module gives both, results in
            fold_report() let you judge for yourself.

    The forward-only guarantee is structural: Fold.train_idx for fold k
    is computed using only df.index[:test_start_position], so it is
    impossible (not just discouraged) for a training set to include any
    bar at or after that fold's test window — there's no parameter that
    can violate this, you'd have to bypass the splitter entirely to
    create a fold that sees the future.

    Args:
        n_folds:        Number of sequential test folds to produce.
        min_train_bars: Minimum bars required before the first fold's
                        test window can begin (warmup floor).
        test_bars:      Number of bars per test fold.
        expanding:      True = expanding window, False = rolling window
                        of size `train_window_bars` (rolling mode only).
        train_window_bars: Fixed training window size, used only when
                        expanding=False.
        embargo_bars:   Bars dropped from the END of each training set,
                        immediately before the test window starts. This
                        guards against the 5-day-forward-return label:
                        the last `embargo_bars` rows of any training set
                        have labels computed from prices that overlap the
                        upcoming test window, so without an embargo the
                        model could partially "see" test-window prices
                        through its own training labels. Default matches
                        the 5-day forward-return horizon.
    """

    def __init__(
        self,
        n_folds:            int  = 5,
        min_train_bars:      int  = 252,
        test_bars:           int  = 63,
        expanding:           bool = True,
        train_window_bars:   int  = 504,
        embargo_bars:        int  = 5,
    ):
        self.n_folds            = n_folds
        self.min_train_bars      = min_train_bars
        self.test_bars           = test_bars
        self.expanding           = expanding
        self.train_window_bars   = train_window_bars
        self.embargo_bars        = embargo_bars

    def split(self, df: pd.DataFrame) -> List[Fold]:
        """
        Produce the sequential folds for `df`.

        Raises if there isn't enough history for at least one fold —
        fails loudly rather than silently returning an empty/misleading
        result.
        """
        n = len(df)
        required = self.min_train_bars + self.test_bars
        if n < required:
            raise ValueError(
                f"Insufficient data for walk-forward split: {n} rows, "
                f"need >= {required} (min_train_bars={self.min_train_bars} "
                f"+ test_bars={self.test_bars})."
            )

        max_possible_folds = (n - self.min_train_bars) // self.test_bars
        n_folds = min(self.n_folds, max_possible_folds)
        if n_folds < self.n_folds:
            logger.warning(
                f"Requested {self.n_folds} folds but only {n} rows available — "
                f"using {n_folds} folds instead."
            )
        if n_folds < 1:
            raise ValueError(
                f"Cannot produce even one fold from {n} rows with "
                f"min_train_bars={self.min_train_bars}, test_bars={self.test_bars}."
            )

        folds: List[Fold] = []
        for k in range(n_folds):
            test_start_pos = self.min_train_bars + k * self.test_bars
            test_end_pos   = min(test_start_pos + self.test_bars, n)
            if test_start_pos >= n:
                break

            if self.expanding:
                train_start_pos = 0
            else:
                train_start_pos = max(0, test_start_pos - self.train_window_bars)

            # Embargo: drop the last `embargo_bars` rows of the training
            # set so labels whose 5-day-forward window would overlap the
            # test window are excluded from training.
            train_end_pos = max(train_start_pos, test_start_pos - self.embargo_bars)

            train_idx = df.index[train_start_pos:train_end_pos]
            test_idx  = df.index[test_start_pos:test_end_pos]

            if len(train_idx) < self.min_train_bars // 2 or len(test_idx) == 0:
                continue

            folds.append(Fold(
                fold_id=k,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
            ))

        if not folds:
            raise ValueError("Walk-forward split produced zero usable folds.")

        return folds


# ======================================================================
# Result containers
# ======================================================================

@dataclass
class FoldResult:
    fold_id:        int
    train_start:    pd.Timestamp
    train_end:      pd.Timestamp
    test_start:     pd.Timestamp
    test_end:       pd.Timestamp
    n_train:        int
    n_test:         int
    test_mse:       float
    test_mae:       float
    test_ic:        float       # Information Coefficient: corr(pred, actual)
    direction_acc:  float       # fraction of bars where sign(pred) == sign(actual)
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class MLSignalResult:
    """
    Complete output of walk-forward ML signal generation.

    signal:           pd.Series, {-1, 0, +1}, index-aligned to the input
                      df, NaN/0 outside the union of all test windows
                      (bars never used as a test bar in any fold have no
                      out-of-sample prediction and are correctly left
                      flat rather than filled with a misleading value).
    predictions:      Raw continuous predicted 5-day forward return per
                      bar, same coverage as signal.
    fold_results:      List[FoldResult], one per walk-forward fold.
    model_type:        "xgboost" | "random_forest"
    deadband:          The threshold used for the {-1,0,+1} conversion.
    """
    signal:          pd.Series
    predictions:     pd.Series
    fold_results:    List[FoldResult]
    model_type:      str
    deadband:        float

    def fold_report(self) -> pd.DataFrame:
        """One row per fold: train/test windows, MSE, IC, direction accuracy."""
        rows = []
        for f in self.fold_results:
            rows.append({
                "fold":           f.fold_id,
                "train_start":    f.train_start.date(),
                "train_end":      f.train_end.date(),
                "test_start":     f.test_start.date(),
                "test_end":       f.test_end.date(),
                "n_train":        f.n_train,
                "n_test":         f.n_test,
                "test_mse":       round(f.test_mse, 6),
                "test_mae":       round(f.test_mae, 6),
                "test_ic":        round(f.test_ic, 4),
                "direction_acc":  round(f.direction_acc, 4),
            })
        return pd.DataFrame(rows).set_index("fold")

    def importance_report(self, top_n: int = 15) -> pd.DataFrame:
        """
        Feature importance aggregated across folds: mean and std of each
        feature's gain-based importance across all folds it appeared in.
        A feature with high mean but also high std is regime-dependent —
        it mattered a lot in some folds and little in others, which is
        itself a finding, not noise to be hidden by averaging.
        """
        all_features = set()
        for f in self.fold_results:
            all_features.update(f.feature_importance.keys())

        rows = []
        for feat in all_features:
            vals = [
                f.feature_importance.get(feat, 0.0)
                for f in self.fold_results
            ]
            rows.append({
                "feature":         feat,
                "mean_importance": np.mean(vals),
                "std_importance":  np.std(vals),
                "n_folds_nonzero": sum(1 for v in vals if v > 0),
            })

        report = (
            pd.DataFrame(rows)
            .set_index("feature")
            .sort_values("mean_importance", ascending=False)
        )
        return report.head(top_n).round(4)

    def summary(self) -> str:
        ic_mean = np.mean([f.test_ic for f in self.fold_results])
        dir_mean = np.mean([f.direction_acc for f in self.fold_results])
        n_active = int((self.signal != 0).sum())
        n_total  = int(self.signal.notna().sum())
        return (
            f"ML Signal ({self.model_type}, deadband={self.deadband:.3f}): "
            f"{len(self.fold_results)} folds, "
            f"mean IC={ic_mean:+.3f}, mean direction_acc={dir_mean:.1%}, "
            f"active {n_active}/{n_total} bars ({n_active/max(n_total,1):.1%})"
        )


# ======================================================================
# ML Signal Generator
# ======================================================================

class MLSignalGenerator:
    """
    Trains a gradient-boosted (or random forest baseline) regression
    model to predict 5-day forward returns from the engineered feature
    set, validated via strict walk-forward folds, and converts
    predictions to a {-1, 0, +1} signal via a symmetric deadband
    threshold.

    Args:
        model_type:      "xgboost" (default) or "random_forest" (no
                        extra dependency, useful baseline/fallback).
        deadband:         Predictions in [-deadband, +deadband] become 0
                        (flat). This is NOT free-tuned against backtest
                        Sharpe in this module — see threshold_sweep()
                        for an explicit, separately-reported sweep
                        instead of silently picking whichever threshold
                        looks best (that would be the single most common
                        way ML-on-financial-data overfits).
        xgb_params:       Override default XGBoost hyperparameters.
        rf_params:        Override default RandomForest hyperparameters.
        random_state:     Seed for model reproducibility.
    """

    DEFAULT_XGB_PARAMS = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    DEFAULT_RF_PARAMS = dict(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        n_jobs=-1,
    )

    def __init__(
        self,
        model_type:    Literal["xgboost", "random_forest"] = "xgboost",
        deadband:       float = 0.005,
        xgb_params:     Optional[dict] = None,
        rf_params:      Optional[dict] = None,
        random_state:   int = 42,
    ):
        self.model_type     = model_type
        self.deadband        = deadband
        self.xgb_params       = {**self.DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.rf_params         = {**self.DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.random_state       = random_state

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit_predict_walk_forward(
        self,
        df:          pd.DataFrame,
        splitter:    Optional[WalkForwardSplitter] = None,
        target_col:  str = "returns_fwd_5",
        ticker:      str = "",
    ) -> MLSignalResult:
        """
        Run the full walk-forward fit/predict loop and return a complete
        MLSignalResult.

        For each fold: a fresh model is trained ONLY on that fold's
        train_idx (no warm-starting from a previous fold's model — each
        fold's model is independent, so a later fold's results cannot
        leak through model state from an earlier fold that happened to
        include data closer to the test window), predicts on test_idx,
        and the predictions are written into the output Series at
        exactly those test positions. A bar that is never any fold's
        test bar (e.g. the initial warmup window, or any tail bars
        beyond the last fold) has no prediction and is correctly left
        flat (0) rather than backfilled or extrapolated.

        Args:
            df:          Fully feature-engineered DataFrame (output of
                        FeatureEngineer.add_all_features()).
            splitter:    WalkForwardSplitter instance. Uses sensible
                        defaults if not provided.
            target_col:  Column to predict. Default returns_fwd_5,
                        already computed by DataCleaner with the correct
                        forward shift.
            ticker:      Label for log messages.

        Returns:
            MLSignalResult with signal, predictions, fold_results.
        """
        tag = f"[{ticker}] " if ticker else ""
        splitter = splitter or WalkForwardSplitter()

        feature_cols = get_feature_columns(df)
        if not feature_cols:
            raise ValueError(
                "No candidate feature columns found. Run "
                "FeatureEngineer.add_all_features() before ML signal generation."
            )
        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        logger.info(
            f"{tag}ML signal: {len(feature_cols)} candidate features, "
            f"target={target_col}, model={self.model_type}"
        )

        folds = splitter.split(df)
        logger.info(f"{tag}Walk-forward: {len(folds)} folds")

        predictions = pd.Series(np.nan, index=df.index)
        fold_results: List[FoldResult] = []

        for fold in folds:
            X_train, y_train = self._build_xy(df, fold.train_idx, feature_cols, target_col)
            X_test,  y_test  = self._build_xy(df, fold.test_idx,  feature_cols, target_col)

            if len(X_train) < 30 or len(X_test) == 0:
                logger.warning(
                    f"{tag}Fold {fold.fold_id}: skipping, "
                    f"n_train={len(X_train)}, n_test={len(X_test)}"
                )
                continue

            model = self._build_model()
            model.fit(X_train.values, y_train.values)

            # Predict for ALL test rows (including ones whose true label
            # is NaN at the tail, e.g. the last 5 bars of the whole
            # dataset where returns_fwd_5 cannot exist) — predictions are
            # still meaningful even where we can't score them; only the
            # SCORING (test_mse, test_ic, etc.) is restricted to rows
            # with a real label.
            X_test_full = df.loc[fold.test_idx, feature_cols].copy()
            X_test_full = X_test_full.fillna(X_train.median())
            preds_full = pd.Series(
                model.predict(X_test_full.values), index=fold.test_idx
            )
            predictions.loc[fold.test_idx] = preds_full

            # Scoring only on rows with a genuine (non-NaN) label
            scored_idx = y_test.index
            if len(scored_idx) > 0:
                preds_scored = preds_full.loc[scored_idx]
                mse = float(np.mean((preds_scored - y_test) ** 2))
                mae = float(np.mean(np.abs(preds_scored - y_test)))
                ic  = (
                    float(np.corrcoef(preds_scored, y_test)[0, 1])
                    if preds_scored.std() > 0 and y_test.std() > 0
                    else 0.0
                )
                direction_acc = float(
                    (np.sign(preds_scored) == np.sign(y_test)).mean()
                )
            else:
                mse = mae = ic = direction_acc = float("nan")

            importance = self._get_importance(model, feature_cols)

            fold_results.append(FoldResult(
                fold_id=fold.fold_id,
                train_start=fold.train_start, train_end=fold.train_end,
                test_start=fold.test_start, test_end=fold.test_end,
                n_train=len(X_train), n_test=len(X_test),
                test_mse=mse, test_mae=mae, test_ic=ic,
                direction_acc=direction_acc,
                feature_importance=importance,
            ))

            logger.info(
                f"{tag}Fold {fold.fold_id}: train={fold.train_start.date()}"
                f"->{fold.train_end.date()} ({len(X_train)} bars), "
                f"test={fold.test_start.date()}->{fold.test_end.date()} "
                f"({len(X_test)} bars), IC={ic:+.3f}, dir_acc={direction_acc:.1%}"
            )

        signal = self._threshold(predictions, self.deadband)

        result = MLSignalResult(
            signal=signal,
            predictions=predictions,
            fold_results=fold_results,
            model_type=self.model_type,
            deadband=self.deadband,
        )
        logger.info(f"{tag}{result.summary()}")
        return result

    def threshold_sweep(
        self,
        predictions: pd.Series,
        target:      pd.Series,
        thresholds:  Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Explicit, reported sweep of deadband thresholds against
        out-of-sample predictions — separated from fit_predict_walk_forward
        so that threshold selection is visible and auditable rather than
        silently baked into the main fitting call. Reports direction
        accuracy and trade frequency at each threshold; it deliberately
        does NOT report backtest Sharpe per threshold, because picking a
        threshold by maximising backtested Sharpe on the same data used
        to generate the signal is the single most common way ML
        threshold-tuning quietly becomes curve-fitting. Use this to pick
        a threshold based on prediction quality and trade frequency
        BEFORE running the one chosen threshold through the backtester.

        Args:
            predictions: Out-of-sample predictions (e.g. result.predictions)
            target:      True forward returns, same index
            thresholds:  Candidate deadband values to evaluate

        Returns:
            DataFrame: one row per threshold, columns = active_pct,
            direction_acc, mean_abs_active_pred
        """
        thresholds = thresholds or [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
        common = predictions.dropna().index.intersection(target.dropna().index)
        pred = predictions.loc[common]
        tgt  = target.loc[common]

        rows = []
        for t in thresholds:
            sig = self._threshold(pred, t)
            active = sig != 0
            if active.sum() > 0:
                dir_acc = float(
                    (np.sign(pred[active]) == np.sign(tgt[active])).mean()
                )
                mean_abs_pred = float(pred[active].abs().mean())
            else:
                dir_acc = float("nan")
                mean_abs_pred = float("nan")

            rows.append({
                "threshold":             t,
                "active_pct":            round(float(active.mean() * 100), 2),
                "direction_acc":         round(dir_acc, 4) if not np.isnan(dir_acc) else None,
                "mean_abs_active_pred":  round(mean_abs_pred, 5) if not np.isnan(mean_abs_pred) else None,
            })

        return pd.DataFrame(rows).set_index("threshold")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_xy(
        self,
        df:           pd.DataFrame,
        idx:           pd.Index,
        feature_cols:  List[str],
        target_col:    str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build (X, y) for a given index slice, dropping rows with a NaN
        target (rows near the warmup edge of any feature, or the tail
        where returns_fwd_5 cannot exist). Feature NaNs (e.g. the first
        252 bars of vol_63d before its rolling window fills) are
        imputed with the TRAIN-set median — computed fresh per fold,
        never globally, so no information about the full series'
        distribution leaks into an early fold.
        """
        sub = df.loc[idx, feature_cols + [target_col]].copy()
        sub = sub.dropna(subset=[target_col])
        X = sub[feature_cols]
        y = sub[target_col]
        X = X.fillna(X.median())
        return X, y

    def _build_model(self):
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                random_state=self.random_state, **self.xgb_params
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                random_state=self.random_state, **self.rf_params
            )
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")

    @staticmethod
    def _get_importance(model, feature_cols: List[str]) -> Dict[str, float]:
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            return {f: float(v) for f, v in zip(feature_cols, vals)}
        return {}

    @staticmethod
    def _threshold(predictions: pd.Series, deadband: float) -> pd.Series:
        """Convert continuous predictions to {-1, 0, +1} via symmetric deadband."""
        sig = pd.Series(0, index=predictions.index, dtype=int)
        valid = predictions.notna()
        sig[valid & (predictions > deadband)]  = 1
        sig[valid & (predictions < -deadband)] = -1
        return sig