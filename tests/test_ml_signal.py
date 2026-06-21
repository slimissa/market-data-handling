"""
tests/test_ml_signal.py — Test suite for Phase 7: ML Signal
QuantOS Market Data Pipeline

Run:
    cd src && python -m pytest ../tests/test_ml_signal.py -v

Test philosophy — weighted heavily toward leak prevention, because that
is the one category of bug in this module that would silently produce
an impressive-looking but worthless result:
    - Every fold's train set must end strictly before its test set starts
    - The embargo gap must actually exceed the forward-return horizon
    - No timestamp from any later fold's test window can appear in an
      earlier fold's training data (checked by direct index intersection,
      not just by comparing boundary dates)
    - Feature columns must exclude every signal_*/position_*/regime_*
      column, not just the obvious OHLCV/return columns
    - Threshold conversion produces only {-1,0,+1}, with deadband
      respected at the boundary
    - A model with genuinely zero predictive power (pure noise target)
      should not show a stable, exploitable IC across folds
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_signal import (
    MLSignalGenerator,
    WalkForwardSplitter,
    Fold,
    FoldResult,
    MLSignalResult,
    get_feature_columns,
    EXCLUDED_FROM_FEATURES,
)
from feature_engineering import FeatureEngineer
from signal_generator import SignalGenerator
from regime_filter import RegimeFilteredEnsemble


# ======================================================================
# Fixtures
# ======================================================================

def make_full_df(n=700, drift=0.0003, vol=0.012, seed=42) -> pd.DataFrame:
    """Fully feature + signal + regime-filter enriched DataFrame."""
    rng = np.random.default_rng(seed)
    log_ret = drift + vol * rng.standard_normal(n)
    closes  = 100.0 * np.exp(np.cumsum(log_ret))
    highs   = closes * (1 + rng.uniform(0, 0.015, n))
    lows    = closes * (1 - rng.uniform(0, 0.015, n))
    volumes = rng.integers(500_000, 2_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": closes * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": highs, "low": lows, "close": closes, "volume": volumes,
        "returns": log_ret, "returns_norm": np.zeros(n),
        "returns_fwd_1": np.append(log_ret[1:], np.nan),
        "returns_fwd_5": np.append(log_ret[5:], [np.nan] * 5),
    }, index=idx)

    eng = FeatureEngineer()
    df  = eng.add_all_features(df)
    sg  = SignalGenerator()
    df  = sg.generate_all(df)
    rfe = RegimeFilteredEnsemble()
    df  = rfe.apply(df)
    return df


def make_pure_noise_df(n=700, seed=7) -> pd.DataFrame:
    """
    A DataFrame where the forward-return target is independent random
    noise, uncorrelated with any feature by construction. Used to verify
    the module does not report a spuriously stable IC on data with
    genuinely zero predictive structure.
    """
    df = make_full_df(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # Overwrite the target with pure noise, independent of all features
    df = df.copy()
    df["returns_fwd_5"] = rng.standard_normal(len(df)) * 0.02
    df.iloc[-5:, df.columns.get_loc("returns_fwd_5")] = np.nan  # keep tail-NaN convention
    return df


@pytest.fixture
def df():
    return make_full_df()


@pytest.fixture
def splitter():
    return WalkForwardSplitter(n_folds=4, min_train_bars=200, test_bars=50, embargo_bars=5)


# ======================================================================
# 1. get_feature_columns — the leak-prevention allowlist
# ======================================================================

class TestGetFeatureColumns:

    def test_excludes_raw_ohlcv(self, df):
        feats = get_feature_columns(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in feats, f"{col} must not be a candidate feature"

    def test_excludes_return_and_target_columns(self, df):
        feats = get_feature_columns(df)
        for col in ["returns", "returns_norm", "returns_fwd_1", "returns_fwd_5"]:
            assert col not in feats, f"{col} must not be a candidate feature"

    def test_excludes_all_signal_columns(self, df):
        """
        Critical leak check: every signal_* column (rule-based signals,
        their strength columns, and gated/pool/adaptive derivatives) must
        be excluded — these are outputs of human-designed rule logic and
        including them would let the ML model trivially learn to copy
        the rule-based system rather than learn independently.
        """
        feats = get_feature_columns(df)
        signal_cols_in_df = [c for c in df.columns if c.startswith("signal_")]
        assert len(signal_cols_in_df) > 0, "test fixture should contain signal_ columns"
        for col in signal_cols_in_df:
            assert col not in feats, f"{col} leaked into ML features"

    def test_excludes_position_scale_columns(self, df):
        feats = get_feature_columns(df)
        position_cols = [c for c in df.columns if c.startswith("position_")]
        for col in position_cols:
            assert col not in feats, f"{col} leaked into ML features"

    def test_excludes_regime_label_column(self, df):
        feats = get_feature_columns(df)
        regime_cols = [c for c in df.columns if c.startswith("regime_")]
        for col in regime_cols:
            assert col not in feats, f"{col} leaked into ML features"

    def test_includes_genuine_engineered_features(self, df):
        feats = get_feature_columns(df)
        # Spot-check a representative sample of genuine features
        for col in ["vol_21d", "rsi_14", "macd_line", "bb_width", "z_price_60d", "atr_14"]:
            assert col in feats, f"{col} should be a candidate feature, was excluded"

    def test_feature_count_matches_expected_24_family_engineering(self, df):
        """
        feature_engineering.py documents '24 columns added' (7 feature
        families). get_feature_columns excludes 'tr' (an ATR
        intermediate column, duplicate information vs atr_14), so the
        candidate ML feature count should be 23, not 24 — and should NOT
        include any of the 24 columns Stage 4/4c add on top of features.
        """
        feats = get_feature_columns(df)
        assert 20 <= len(feats) <= 24, (
            f"Expected ~23 candidate features (24 engineered - 'tr' "
            f"intermediate), got {len(feats)}: {feats}"
        )

    def test_excludes_non_numeric_columns(self, df):
        feats = get_feature_columns(df)
        for col in feats:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"{col} is non-numeric and should have been excluded"
            )

    def test_excluded_constant_matches_actual_exclusions(self):
        """EXCLUDED_FROM_FEATURES should contain exactly the non-derived exclusions."""
        expected_min = {"open", "high", "low", "close", "volume",
                         "returns", "returns_fwd_1", "returns_fwd_5"}
        assert expected_min.issubset(EXCLUDED_FROM_FEATURES)


# ======================================================================
# 2. WalkForwardSplitter — leak-prevention structural guarantees
# ======================================================================

class TestWalkForwardSplitter:

    def test_every_fold_train_strictly_before_test(self, df, splitter):
        """The core temporal guarantee: train_end < test_start, always."""
        folds = splitter.split(df)
        assert len(folds) > 0
        for f in folds:
            assert f.train_end < f.test_start, (
                f"Fold {f.fold_id}: train_end {f.train_end} is not strictly "
                f"before test_start {f.test_start}"
            )

    def test_embargo_gap_exceeds_forward_return_horizon(self, df):
        """
        With embargo_bars=5 (matching returns_fwd_5's horizon), the gap
        between train_end and test_start must be at least 5 calendar
        positions — otherwise the last training labels would read prices
        inside the test window.
        """
        splitter = WalkForwardSplitter(
            n_folds=3, min_train_bars=200, test_bars=50, embargo_bars=5
        )
        folds = splitter.split(df)
        for f in folds:
            train_end_pos = df.index.get_loc(f.train_end)
            test_start_pos = df.index.get_loc(f.test_start)
            gap = test_start_pos - train_end_pos
            assert gap >= 5, (
                f"Fold {f.fold_id}: embargo gap {gap} positions is less than "
                f"the 5-bar forward-return horizon — label leak is possible"
            )

    def test_no_test_timestamp_appears_in_any_earlier_training_set(self, df, splitter):
        """
        Direct index-intersection check (not just boundary-date
        comparison): no fold's test_idx may overlap any EARLIER fold's
        train_idx. This is the property that actually matters for
        leak-freedom, checked at the data level.
        """
        folds = splitter.split(df)
        for i, later_fold in enumerate(folds):
            for earlier_fold in folds[:i]:
                overlap = set(earlier_fold.train_idx) & set(later_fold.test_idx)
                assert not overlap, (
                    f"Fold {earlier_fold.fold_id}'s training set contains "
                    f"{len(overlap)} timestamps from fold {later_fold.fold_id}'s "
                    f"test window"
                )

    def test_folds_are_sequential_non_overlapping_test_windows(self, df, splitter):
        """Test windows themselves should not overlap each other."""
        folds = splitter.split(df)
        for i in range(len(folds) - 1):
            overlap = set(folds[i].test_idx) & set(folds[i + 1].test_idx)
            assert not overlap, f"Test windows of fold {i} and {i+1} overlap"

    def test_expanding_window_grows(self, df):
        splitter = WalkForwardSplitter(
            n_folds=4, min_train_bars=200, test_bars=50,
            expanding=True, embargo_bars=5,
        )
        folds = splitter.split(df)
        sizes = [len(f.train_idx) for f in folds]
        assert sizes == sorted(sizes), "Expanding window training sets should grow monotonically"

    def test_rolling_window_bounded(self, df):
        splitter = WalkForwardSplitter(
            n_folds=4, min_train_bars=200, test_bars=50,
            expanding=False, train_window_bars=150, embargo_bars=5,
        )
        folds = splitter.split(df)
        for f in folds:
            assert len(f.train_idx) <= 150, (
                f"Rolling window fold {f.fold_id} has {len(f.train_idx)} "
                f"training rows, exceeding train_window_bars=150"
            )

    def test_insufficient_data_raises(self):
        small_df = pd.DataFrame(
            {"x": np.arange(50)},
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        splitter = WalkForwardSplitter(n_folds=3, min_train_bars=200, test_bars=50)
        with pytest.raises(ValueError, match="Insufficient data"):
            splitter.split(small_df)

    def test_fewer_folds_than_requested_logs_warning_not_error(self, df):
        """Requesting more folds than the data supports should degrade gracefully."""
        splitter = WalkForwardSplitter(n_folds=100, min_train_bars=200, test_bars=50)
        folds = splitter.split(df)  # should not raise
        assert 0 < len(folds) < 100

    def test_zero_folds_raises(self):
        df_tiny = pd.DataFrame(
            {"x": np.arange(210)},
            index=pd.date_range("2020-01-01", periods=210, freq="D"),
        )
        splitter = WalkForwardSplitter(n_folds=5, min_train_bars=200, test_bars=50)
        with pytest.raises(ValueError):
            splitter.split(df_tiny)


# ======================================================================
# 3. MLSignalGenerator — fitting, prediction, no-lookahead in practice
# ======================================================================

class TestMLSignalGenerator:

    def test_fit_predict_returns_result(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        assert isinstance(result, MLSignalResult)
        assert len(result.fold_results) > 0

    def test_random_forest_backend_works(self, df, splitter):
        """RandomForest baseline should work without xgboost as a dependency."""
        ml = MLSignalGenerator(model_type="random_forest", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        assert len(result.fold_results) > 0

    def test_signal_values_valid(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        valid = {-1, 0, 1}
        actual = set(result.signal.dropna().unique())
        assert actual.issubset(valid)

    def test_bars_never_in_any_test_fold_are_flat(self, df, splitter):
        """
        The initial warmup window (before the first fold's test start)
        has no out-of-sample prediction and must be flat (0), not
        backfilled or extrapolated from later folds.
        """
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        folds = splitter.split(df)
        first_test_start = folds[0].test_start
        warmup_mask = df.index < first_test_start
        warmup_signal = result.signal.loc[warmup_mask]
        assert (warmup_signal == 0).all(), (
            "Warmup-period bars (before any fold's test window) must be "
            "flat, found non-zero values"
        )

    def test_deadband_produces_flat_for_small_predictions(self):
        """Predictions within [-deadband, +deadband] must threshold to 0."""
        ml = MLSignalGenerator(deadband=0.01)
        preds = pd.Series([0.005, -0.005, 0.0099, -0.0099, 0.0, np.nan])
        sig = ml._threshold(preds, 0.01)
        assert list(sig.values[:5]) == [0, 0, 0, 0, 0]

    def test_deadband_boundary_is_strict_inequality(self):
        """Exactly at the deadband should NOT trigger (strict > / <)."""
        ml = MLSignalGenerator(deadband=0.01)
        preds = pd.Series([0.01, -0.01, 0.0101, -0.0101])
        sig = ml._threshold(preds, 0.01)
        assert list(sig.values) == [0, 0, 1, -1]

    def test_predictions_beyond_deadband_get_correct_sign(self):
        ml = MLSignalGenerator(deadband=0.005)
        preds = pd.Series([0.02, -0.02, 0.001])
        sig = ml._threshold(preds, 0.005)
        assert list(sig.values) == [1, -1, 0]

    def test_missing_target_column_raises(self, df, splitter):
        df_bad = df.drop(columns=["returns_fwd_5"])
        ml = MLSignalGenerator()
        with pytest.raises(KeyError, match="returns_fwd_5"):
            ml.fit_predict_walk_forward(df_bad, splitter=splitter)

    def test_unknown_model_type_raises(self, df, splitter):
        ml = MLSignalGenerator(model_type="not_a_real_model")
        with pytest.raises(ValueError, match="Unknown model_type"):
            ml.fit_predict_walk_forward(df, splitter=splitter)

    def test_fold_count_matches_splitter(self, df, splitter):
        ml = MLSignalGenerator()
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        expected_folds = len(splitter.split(df))
        assert len(result.fold_results) == expected_folds

    def test_each_fold_train_size_matches_actual_fold(self, df, splitter):
        ml = MLSignalGenerator()
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        folds = splitter.split(df)
        for fr, f in zip(result.fold_results, folds):
            # n_train can be slightly less than len(f.train_idx) due to
            # dropping NaN-target rows, but should be close
            assert fr.n_train <= len(f.train_idx)
            assert fr.n_train > 0

    def test_pure_noise_target_does_not_show_stable_strong_ic(self):
        """
        Sanity check on the validation methodology itself: with a target
        that is genuinely independent random noise, the model should not
        be able to find a strong, CONSISTENT predictive signal across
        folds. Some folds may show a moderately large |IC| by chance
        (5-7 folds, finite sample), but the across-fold mean should stay
        small and folds should not all agree in sign — a model finding
        strong, same-signed IC in every fold on pure noise would indicate
        a leak somewhere in the pipeline, not genuine skill.
        """
        df_noise = make_pure_noise_df(n=700)
        splitter = WalkForwardSplitter(n_folds=5, min_train_bars=200, test_bars=50, embargo_bars=5)
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.0)
        result = ml.fit_predict_walk_forward(df_noise, splitter=splitter)

        ics = [f.test_ic for f in result.fold_results if not np.isnan(f.test_ic)]
        assert len(ics) > 0
        mean_ic = np.mean(ics)
        assert abs(mean_ic) < 0.25, (
            f"Mean IC on pure-noise target is {mean_ic:.3f} — suspiciously "
            f"strong for a target with no genuine relationship to the "
            f"features, suggests a possible data leak"
        )
        # Folds should not be unanimously the same sign on pure noise
        signs = [np.sign(ic) for ic in ics if abs(ic) > 1e-6]
        if len(signs) >= 4:
            assert not all(s == signs[0] for s in signs), (
                "All folds agree in IC sign on a pure-noise target — "
                "suspicious, suggests a leak rather than genuine consistency"
            )


# ======================================================================
# 4. Threshold sweep
# ======================================================================

class TestThresholdSweep:

    def test_sweep_returns_dataframe(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        sweep = ml.threshold_sweep(result.predictions, df["returns_fwd_5"])
        assert isinstance(sweep, pd.DataFrame)
        assert "active_pct" in sweep.columns

    def test_higher_threshold_reduces_active_pct(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        sweep = ml.threshold_sweep(
            result.predictions, df["returns_fwd_5"],
            thresholds=[0.0, 0.005, 0.02],
        )
        active_pcts = sweep["active_pct"].values
        assert active_pcts[0] >= active_pcts[1] >= active_pcts[2], (
            "active_pct should be monotonically non-increasing as "
            "threshold increases"
        )

    def test_sweep_does_not_report_backtest_sharpe(self, df, splitter):
        """
        Explicit design check: threshold_sweep() must not expose a
        Sharpe-like column, since selecting a threshold by maximising
        backtested Sharpe on the same data used to fit the signal is a
        direct path to overfitting the threshold.
        """
        ml = MLSignalGenerator(model_type="xgboost", deadband=0.005)
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        sweep = ml.threshold_sweep(result.predictions, df["returns_fwd_5"])
        forbidden = {"sharpe", "sortino", "calmar", "total_return"}
        assert not (forbidden & set(sweep.columns)), (
            f"threshold_sweep() exposes backtest-derived columns "
            f"{forbidden & set(sweep.columns)} — risk of threshold "
            f"overfitting to backtest performance"
        )


# ======================================================================
# 5. MLSignalResult reporting
# ======================================================================

class TestMLSignalResult:

    def test_fold_report_one_row_per_fold(self, df, splitter):
        ml = MLSignalGenerator()
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        report = result.fold_report()
        assert len(report) == len(result.fold_results)

    def test_importance_report_sorted_descending(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost")
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        report = result.importance_report(top_n=10)
        vals = report["mean_importance"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_importance_report_respects_top_n(self, df, splitter):
        ml = MLSignalGenerator(model_type="xgboost")
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        report = result.importance_report(top_n=5)
        assert len(report) <= 5

    def test_summary_returns_string(self, df, splitter):
        ml = MLSignalGenerator()
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        s = result.summary()
        assert isinstance(s, str)
        assert "ML Signal" in s

    def test_random_forest_importance_populated(self, df, splitter):
        ml = MLSignalGenerator(model_type="random_forest")
        result = ml.fit_predict_walk_forward(df, splitter=splitter)
        # At least one fold should have non-empty importances
        assert any(len(f.feature_importance) > 0 for f in result.fold_results)