"""
pipeline.py — Orchestrates fetch → clean → engineer → signal → backtest → factor attribution
QuantOS Market Data Pipeline

Usage:
    python pipeline.py                          # uses config.yaml
    python pipeline.py --config my_config.yaml  # custom config
    python pipeline.py --symbols AAPL SPY --start 2020-01-01 --end 2023-12-31
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yaml

from data_fetcher import MarketDataFetcher
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from signal_generator import SignalGenerator
from backtester import VectorisedBacktester, TransactionCostModel
from factor_model import FactorModel, RegimeAnalyser
from regime_filter import RegimeFilteredEnsemble, RegimeFilterPresets
from ml_signal import MLSignalGenerator, WalkForwardSplitter
from presets import apply_preset, list_presets, get_preset
from cli_utils import resolve_dates, OutputWriter
from market_search import MarketSearch
from regime_detector import (
    RuleBasedClassifier,
    HMMRegimeDetector,
    AdaptiveSignalSwitch,
    REGIME_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """
    End-to-end market data pipeline — Phases 1-7.

    Stages:
        1. Fetch          — yfinance API or local CSV cache
        2. Clean          — timestamps, alignment, missing data, returns
        3. Feature Eng.   — vol, RSI, ATR, volume, Bollinger, MACD, z-score
        4. Signal Gen.    — RSI, MACD, z-score, Bollinger, vol scale, ensemble
        4b. Regime Detect — rule-based or HMM regime classification + adaptive switching
        5. Backtest       — vectorised per-signal comparison + metrics
        6. Factor Model   — CAPM / FF3 / Carhart4 alpha attribution
        7. Regime Analysis— Sharpe per market regime (crisis/trending/range)
        8. Persist        — CSVs + JSON reports + backtest + factor results
    """

    DEFAULTS = {
        "data_dir":        "./data",
        "interval":        "1d",
        "timezone":        "UTC",
        "target_freq":     "D",
        "resample_method": "ffill",
        "fill_method":     "ffill",
        "max_gap":         5,
        "return_method":   "log",
        "use_cache":       True,
        "watchlist":       [],
        "features": {
            "volatility":   {"windows": [5, 21, 63]},
            "rsi":          {"window": 14},
            "atr":          {"window": 14},
            "volume":       {"window": 20},
            "bollinger":    {"window": 20, "num_std": 2.0},
            "macd":         {"fast": 12, "slow": 26, "signal": 9},
            "price_zscore": {"windows": [20, 60]},
        },
        "signals": {
            "rsi":       {"oversold": 30, "overbought": 70, "exit": 50, "smoothing": 1},
            "macd":      {"require_zero_cross": False},
            "zscore":    {"entry_threshold": 2.0, "exit_threshold": 0.0, "window": 60},
            "bollinger": {"squeeze_percentile": 20.0, "max_holding_bars": 10},
            "vol_scale": {"window": 21, "lookback": 252, "floor": 0.0, "ceiling": 2.0},
            "holding":   {"min_bars": 2, "max_bars": 20},
            "ensemble":  {"method": "majority_vote"},
        },
        "backtest": {
            "initial_capital": 100_000,
            "position_sizing": "fixed_notional",
            "target_notional": 100_000,
            "cost_model":      "liquid_equity",
            "max_drawdown_exit": None,
        },
        "factor": {
            "enabled":        True,
            "rf_annual":      0.05,
            "rolling_window": 126,
            "factors":        ["MKT", "SMB", "HML", "MOM"],
            "cache_dir":      "./data/factors",
        },
        "regime_filter": {
            "enabled":         True,
            "vol_percentile":  30.0,
            "max_trend_annual": 0.10,
            "bb_percentile":   40.0,
        },
        "ml_signal": {
            "enabled":          False,   # opt-in: walk-forward fitting is the
                                          # slowest stage in the pipeline and
                                          # the newest, so it defaults off
                                          # rather than silently lengthening
                                          # every run.
            "model_type":       "xgboost",   # "xgboost" | "random_forest"
            "deadband":         0.005,
            "target_col":       "returns_fwd_5",
            "n_folds":          5,
            "min_train_bars":   252,
            "test_bars":        63,
            "expanding":        True,
            "train_window_bars": 504,    # only used when expanding=False
            "embargo_bars":     5,       # must be >= the forward-return horizon
        },
        "regime": {
            "enabled":          True,
            "method":           "rule_based",
            "n_states":         4,
            "feature_cols":     ["returns", "vol_21d", "macd_line"],
            "adaptive_enabled": True,
        },
    }

    # ------------------------------------------------------------------ #
    # Init                                                                 #
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path} — using defaults.")
            self.config = dict(self.DEFAULTS)
        else:
            with open(config_path) as fh:
                loaded = yaml.safe_load(fh) or {}
            merged = dict(self.DEFAULTS)
            for k, v in loaded.items():
                if k in ("features", "signals", "backtest", "factor", "regime_filter", "ml_signal", "regime"):
                    merged[k] = {**self.DEFAULTS.get(k, {}), **v}
                else:
                    merged[k] = v
            self.config = merged

        self._data_dir = Path(self.config["data_dir"])
        self._ensure_dirs()

        # ---- Stage components ----
        self.fetcher    = MarketDataFetcher(self.config["data_dir"])
        self.cleaner    = DataCleaner()
        self.engineer   = FeatureEngineer()
        self.signal_gen = SignalGenerator()

        # Backtester
        bt_cfg = self.config["backtest"]
        cost_map = {
            "zero":          TransactionCostModel.zero,
            "liquid_equity": TransactionCostModel.liquid_equity,
            "small_cap":     TransactionCostModel.small_cap,
        }
        cost_fn = cost_map.get(bt_cfg["cost_model"], TransactionCostModel.liquid_equity)
        self.backtester = VectorisedBacktester(
            initial_capital=bt_cfg["initial_capital"],
            position_sizing=bt_cfg["position_sizing"],
            target_notional=bt_cfg["target_notional"],
            cost_model=cost_fn(),
            max_drawdown_exit=bt_cfg.get("max_drawdown_exit", None),
        )

        # Factor model
        f_cfg = self.config["factor"]
        self.factor_model = FactorModel(
            rf_annual=f_cfg["rf_annual"],
            rolling_window=f_cfg["rolling_window"],
            cache_dir=f_cfg["cache_dir"],
        )
        self.regime_analyser = RegimeAnalyser()
        rf_init = self.config["regime_filter"]
        self.regime_filter_ensemble = RegimeFilteredEnsemble(
            vol_percentile_mr=rf_init.get("vol_percentile", 20.0),
            max_trend_annual=rf_init.get("max_trend_annual", 0.06),
            bb_percentile=rf_init.get("bb_percentile", 25.0),
        )

        ml_cfg = self.config["ml_signal"]
        self.ml_generator = MLSignalGenerator(
            model_type=ml_cfg["model_type"],
            deadband=ml_cfg["deadband"],
        )
        self.ml_splitter = WalkForwardSplitter(
            n_folds=ml_cfg["n_folds"],
            min_train_bars=ml_cfg["min_train_bars"],
            test_bars=ml_cfg["test_bars"],
            expanding=ml_cfg["expanding"],
            train_window_bars=ml_cfg["train_window_bars"],
            embargo_bars=ml_cfg["embargo_bars"],
        )

        # Regime detection (Phase 7)
        self.rule_detector = RuleBasedClassifier()
        self.hmm_detector = HMMRegimeDetector(
            n_states=self.config.get("regime", {}).get("n_states", 4),
            feature_cols=self.config.get("regime", {}).get(
                "feature_cols", ["returns", "vol_21d", "macd_line"]
            ),
        )
        self.adaptive_switch = AdaptiveSignalSwitch()
        self.adaptive_switch.register("signal_rsi",   favourable=["range_bound"])
        self.adaptive_switch.register("signal_zscore", favourable=["range_bound"])
        self.adaptive_switch.register("signal_macd",  favourable=["trending_up", "trending_down"])
        self.adaptive_switch.register("signal_bb",    favourable=["trending_up", "trending_down"])
        # crisis: intentionally unregistered → zero weight for all signals

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        symbols:    Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date:   str = "2023-12-31",
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute the full 8-stage pipeline for all symbols.

        Returns a dict of fully-processed DataFrames (signals + features).
        All results are also saved to disk under data_dir/.
        """
        if symbols is None:
            symbols = self.config["watchlist"]
        if not symbols:
            raise ValueError("No symbols provided and watchlist is empty.")

        logger.info(f"Pipeline start — {len(symbols)} symbol(s): {symbols}")

        # ---- Stage 1: Fetch ----
        raw_data = self.fetcher.fetch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=self.config["interval"],
            use_cache=self.config["use_cache"],
        )

        # ---- Stages 2-7 per symbol ----
        processed: Dict[str, pd.DataFrame] = {}
        backtest_results: Dict[str, Dict] = {}
        daily_returns_all: Dict[str, Dict[str, pd.Series]] = {}

        feat_cfg = self.config["features"]
        sig_cfg  = self.config["signals"]

        for symbol, df in raw_data.items():
            try:
                # Stage 2: Clean
                df_clean = self.cleaner.clean(
                    df, ticker=symbol,
                    timezone=self.config["timezone"],
                    target_freq=self.config["target_freq"],
                    resample_method=self.config["resample_method"],
                    max_gap=self.config["max_gap"],
                    fill_method=self.config["fill_method"],
                    return_method=self.config["return_method"],
                )

                # Stage 3: Feature engineering
                df_feat = self.engineer.add_all_features(
                    df_clean, ticker=symbol,
                    vol_windows=feat_cfg["volatility"]["windows"],
                    rsi_window=feat_cfg["rsi"]["window"],
                    atr_window=feat_cfg["atr"]["window"],
                    volume_window=feat_cfg["volume"]["window"],
                    bb_window=feat_cfg["bollinger"]["window"],
                    bb_std=feat_cfg["bollinger"]["num_std"],
                    macd_fast=feat_cfg["macd"]["fast"],
                    macd_slow=feat_cfg["macd"]["slow"],
                    macd_signal=feat_cfg["macd"]["signal"],
                    zscore_windows=feat_cfg["price_zscore"]["windows"],
                )

                # Stage 4: Signal generation
                df_signals = self.signal_gen.generate_all(
                    df_feat, ticker=symbol,
                    rsi_oversold=sig_cfg["rsi"]["oversold"],
                    rsi_overbought=sig_cfg["rsi"]["overbought"],
                    rsi_exit=sig_cfg["rsi"]["exit"],
                    rsi_smoothing=sig_cfg["rsi"]["smoothing"],
                    macd_require_zero_cross=sig_cfg["macd"]["require_zero_cross"],
                    zscore_entry=sig_cfg["zscore"]["entry_threshold"],
                    zscore_exit=sig_cfg["zscore"]["exit_threshold"],
                    zscore_window=sig_cfg["zscore"]["window"],
                    bb_squeeze_percentile=sig_cfg["bollinger"]["squeeze_percentile"],
                    bb_max_holding_bars=sig_cfg["bollinger"]["max_holding_bars"],
                    vol_window=sig_cfg["vol_scale"]["window"],
                    vol_lookback=sig_cfg["vol_scale"]["lookback"],
                    vol_scale_floor=sig_cfg["vol_scale"]["floor"],
                    vol_scale_ceiling=sig_cfg["vol_scale"]["ceiling"],
                    min_holding_bars=sig_cfg["holding"]["min_bars"],
                    max_holding_bars=sig_cfg["holding"]["max_bars"],
                    ensemble_method=sig_cfg["ensemble"]["method"],
                )

                # Stage 4c: Regime-gated signal filtering (Phase 6a)
                #
                # PREVIOUSLY MISSING: self.regime_filter_ensemble was
                # instantiated in __init__ but never called anywhere in
                # run(). Every signal — including signal_ensemble — reached
                # the backtester completely unfiltered, taking full-size
                # fixed_notional positions even in regimes where that exact
                # signal family is known (from the Phase 5 regime analysis)
                # to lose money. This is the direct cause of the -62% to
                # -89% drawdowns: not a position-sizing bug, but the total
                # absence of any regime gate upstream of position sizing.
                #
                # Wiring this in adds the regime-gated columns
                # (signal_rsi_vol_trend_gated, signal_zscore_vol_bb_gated,
                # signal_macd_trend_gated, signal_bb_breakout_gated,
                # signal_mr_pool, signal_trend_pool, signal_regime_adaptive)
                # so they are backtested alongside the raw, ungated signals
                # — giving an honest side-by-side comparison instead of
                # silently discarding the Phase 6a work.
                rf_cfg = self.config["regime_filter"]
                if rf_cfg.get("enabled", True):
                    df_signals = self.regime_filter_ensemble.apply(df_signals)
                    logger.info(
                        f"[{symbol}] Regime-gated signals added "
                        f"(signal_regime_adaptive, *_gated columns)."
                    )

                # Stage 4d: ML signal (Phase 7) — walk-forward validated
                #
                # Disabled by default (ml_signal.enabled=False) since this
                # is the slowest stage in the pipeline (re-fits a fresh
                # model per fold) and the newest, least battle-tested one.
                # When enabled, produces signal_ml alongside every
                # rule-based signal, flowing through the SAME backtester,
                # gating comparison, and factor attribution as everything
                # else — that reuse is the entire point: this signal is
                # only trustworthy because that validation machinery
                # already existed and was tested before this stage was
                # written.
                ml_cfg = self.config["ml_signal"]
                if ml_cfg.get("enabled", False):
                    try:
                        ml_result = self.ml_generator.fit_predict_walk_forward(
                            df_signals,
                            splitter=self.ml_splitter,
                            target_col=ml_cfg["target_col"],
                            ticker=symbol,
                        )
                        df_signals["signal_ml"] = ml_result.signal
                        self._save_ml_reports(
                            ml_result, symbol,
                            target=df_signals[ml_cfg["target_col"]],
                        )
                        logger.info(f"[{symbol}] {ml_result.summary()}")
                    except Exception as exc:
                        logger.error(
                            f"[{symbol}] ML signal generation failed: {exc}",
                            exc_info=True,
                        )

                # Stage 4b: Regime detection (Phase 7)
                if self.config.get("regime", {}).get("enabled", False):
                    if self.config["regime"]["method"] == "hmm":
                        detector = self.hmm_detector
                        if not detector._is_fitted:
                            detector.fit(df_signals)
                        result = detector.predict(df_signals, online=True)
                    else:
                        detector = self.rule_detector
                        result = detector.classify(df_signals)

                    df_signals["regime_label"] = result.labels
                    for col in result.probabilities.columns:
                        df_signals[f"regime_prob_{col}"] = result.probabilities[col]

                    # Adaptive switching
                    #
                    # PREVIOUSLY: apply_discrete(threshold=0.0) collapsed the
                    # continuous regime-confidence-weighted blend to a hard
                    # {-1,0,+1} via sign(), discarding exactly the
                    # calibration information HMM probabilities exist to
                    # provide. A blend of +0.51 (barely-favoured long) and
                    # +0.99 (overwhelmingly-favoured long) both produced an
                    # identical full-size +1 position.
                    #
                    # NOW: apply() returns the continuous weighted value in
                    # [-1, +1] directly. Its sign gives the discrete
                    # direction (still required by the signal_* scanning
                    # logic in Stage 5 below, and by Trade-extraction logic
                    # in the backtester which expects {-1,0,+1}). Its
                    # absolute magnitude becomes position_scale_adaptive,
                    # consumed by VectorisedBacktester's existing scale_col
                    # mechanism — so a 51/49 regime split now produces a
                    # small position, and a 99/1 split produces a full-size
                    # one, instead of both being identical.
                    if self.config["regime"].get("adaptive_enabled", True):
                        continuous = self.adaptive_switch.apply(
                            df_signals, result.probabilities, normalise=True
                        ).fillna(0.0)
                        # fillna(0.0) above is required: result.probabilities
                        # legitimately contains NaN during the regime
                        # detector's warmup period (insufficient history for
                        # HMM/rule-based classification). np.sign(NaN) is
                        # NaN, and .astype(int) on a NaN-containing float
                        # Series raises IntCastingNaNError — without this
                        # fillna, the pipeline would crash on every run that
                        # includes a warmup period (i.e. every run).
                        df_signals["signal_adaptive"] = np.sign(continuous).astype(int)
                        df_signals["position_scale_adaptive"] = continuous.abs().clip(0.0, 1.0)

                    logger.info(
                        f"[{symbol}] Regime detection complete "
                        f"({result.method})."
                    )

                # Stage 5: Backtest — compare all signals
                bt_results_map = {}
                signal_cols = [
                    c for c in df_signals.columns
                    if c.startswith("signal_") and not c.endswith("_strength")
                ]
                for sig_col in signal_cols:
                    try:
                        # signal_adaptive must use its OWN confidence-derived
                        # scale column, not the generic position_scale (which
                        # is computed from vol percentile and has no
                        # knowledge of regime confidence). Every other
                        # signal continues to use the default scale_col.
                        scale_col = (
                            "position_scale_adaptive"
                            if sig_col == "signal_adaptive"
                            and "position_scale_adaptive" in df_signals.columns
                            else "position_scale"
                        )
                        bt = self.backtester.run(
                            df_signals, signal_col=sig_col, scale_col=scale_col,
                            ticker=symbol,
                        )
                        bt_results_map[sig_col] = bt
                    except Exception as e:
                        logger.warning(f"[{symbol}] Backtest failed for {sig_col}: {e}")

                # Save comparison table
                if bt_results_map:
                    comparison = pd.DataFrame(
                        {sig: res.metrics for sig, res in bt_results_map.items()}
                    ).T.sort_values("sharpe", ascending=False)
                    self._save_backtest(comparison, symbol)
                    backtest_results[symbol] = bt_results_map

                    # Gated-vs-naive comparison — the number this phase exists
                    # to produce. Pairs each base signal with its regime-gated
                    # counterpart and computes the delta directly, instead of
                    # leaving it as an exercise to diff two rows of the
                    # standard comparison table by hand.
                    self._save_gating_comparison(comparison, symbol)
                    # Filter to active trading days for factor attribution
                    filtered_returns = {}
                    for sig, res in bt_results_map.items():
                        active_mask = res.positions != 0
                        if active_mask.sum() > 30:
                            filtered_returns[sig] = res.daily_returns[active_mask]
                        else:
                            filtered_returns[sig] = res.daily_returns
                    daily_returns_all[symbol] = filtered_returns

                # Stage 8: Persist processed data + quality reports
                self._save(df_signals, symbol)
                self._save_reports(df_clean, df_feat, df_signals, symbol)
                processed[symbol] = df_signals

            except Exception as exc:
                logger.error(f"[{symbol}] Processing failed: {exc}", exc_info=True)

        # ---- Stage 6: Factor attribution (per symbol) ----
        if self.config["factor"]["enabled"] and daily_returns_all:
            self._run_factor_attribution(
                daily_returns_all, start_date, end_date,
                processed, backtest_results
            )

        # ---- Cross-watchlist summary ----
        if len(processed) >= 2:
            self._save_watchlist_comparison(backtest_results)

        logger.info(
            f"Pipeline complete — "
            f"{len(processed)}/{len(raw_data)} symbols processed."
        )
        return processed

    # ------------------------------------------------------------------ #
    # Stage 6 + 7: Factor attribution + regime analysis                   #
    # ------------------------------------------------------------------ #

    def _run_factor_attribution(
        self,
        daily_returns_all: Dict[str, Dict[str, pd.Series]],
        start_date: str,
        end_date: str,
        processed: Dict[str, pd.DataFrame],
        backtest_results: Dict[str, Dict],
    ) -> None:
        """
        Run factor attribution and regime analysis for every symbol.

        Saves per-symbol:
            {symbol}_factor_report.json    — alpha/beta/R²/IC per signal × model
            {symbol}_regime_report.json    — Sharpe per signal per regime
            {symbol}_rolling_alpha.csv     — rolling alpha time series (ensemble only)
        """
        f_cfg = self.config["factor"]

        for symbol, returns_map in daily_returns_all.items():
            logger.info(f"[{symbol}] Running factor attribution...")
            try:
                # ---- Factor regression ----
                factor_results = self.factor_model.run(
                    daily_returns=returns_map,
                    start_date=start_date,
                    end_date=end_date,
                    ticker=symbol,
                    factors=f_cfg["factors"],
                )

                # Save attribution table
                factor_json = self._factor_results_to_json(factor_results)
                out = self._data_dir / "results" / f"{symbol}_factor_report.json"
                with open(out, "w") as fh:
                    json.dump(factor_json, fh, indent=2, default=str)
                logger.info(f"[{symbol}] Factor report → {out}")

                # Save rolling alpha for ensemble signal
                if "signal_ensemble" in factor_results.rolling:
                    roll = factor_results.rolling["signal_ensemble"]
                    roll_df = pd.DataFrame({
                        "alpha_annual": roll.alpha_series,
                        "beta_mkt":     roll.beta_series,
                        "r2":           roll.r2_series,
                        "regime":       roll.regime_labels,
                    })
                    roll_out = self._data_dir / "results" / f"{symbol}_rolling_alpha.csv"
                    roll_df.to_csv(roll_out)
                    logger.info(f"[{symbol}] Rolling alpha → {roll_out}")

                # ---- Stage 7: Regime analysis ----
                if symbol in processed:
                    df = processed[symbol]
                    if "returns" in df.columns:
                        regime = self.regime_analyser.classify(
                            df["returns"].dropna()
                        )
                        regime_table = self.regime_analyser.performance_by_regime(
                            returns_map, regime
                        )
                        regime_out = self._data_dir / "results" / f"{symbol}_regime_report.json"
                        with open(regime_out, "w") as fh:
                            json.dump(
                                regime_table.round(4).to_dict(),
                                fh, indent=2, default=str
                            )
                        logger.info(f"[{symbol}] Regime report → {regime_out}")

                # Print summary to console
                factor_results.print_summary()

            except Exception as exc:
                logger.error(
                    f"[{symbol}] Factor attribution failed: {exc}", exc_info=True
                )

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def _save(self, df: pd.DataFrame, symbol: str) -> None:
        out = self._data_dir / "processed" / f"{symbol}_processed.csv"
        df.to_csv(out)
        logger.info(f"[{symbol}] Saved → {out}")

    def _save_reports(
        self,
        df_clean: pd.DataFrame,
        df_feat: pd.DataFrame,
        df_signals: pd.DataFrame,
        symbol: str,
    ) -> None:
        # Data quality
        clean_report = self.cleaner.quality_report(df_clean, ticker=symbol)
        with open(self._data_dir / "processed" / f"{symbol}_data_report.json", "w") as fh:
            json.dump(clean_report, fh, indent=2)

        # Feature quality
        feat_report = self.engineer.feature_report(df_feat, ticker=symbol)
        with open(self._data_dir / "processed" / f"{symbol}_feature_report.json", "w") as fh:
            json.dump(feat_report, fh, indent=2, default=str)

        # Signal quality
        sig_report = self.signal_gen.signal_report(df_signals, ticker=symbol)
        with open(self._data_dir / "processed" / f"{symbol}_signal_report.json", "w") as fh:
            json.dump(sig_report, fh, indent=2, default=str)

        logger.info(f"[{symbol}] Quality reports saved.")

    def _save_ml_reports(self, ml_result, symbol: str, target: pd.Series) -> None:
        """
        Persist the walk-forward fold report, aggregated feature
        importance, and an explicit deadband threshold sweep for the ML
        signal — these are the artefacts that make the ML signal
        auditable rather than a black box: which folds actually had
        predictive power (test_ic, direction_acc), which features the
        model relied on, and how trade frequency / direction accuracy
        trade off against the deadband threshold (reported separately
        from backtest Sharpe by design — see threshold_sweep()'s
        docstring for why).
        """
        fold_out = self._data_dir / "results" / f"{symbol}_ml_fold_report.csv"
        ml_result.fold_report().to_csv(fold_out)

        importance_out = self._data_dir / "results" / f"{symbol}_ml_importance.csv"
        ml_result.importance_report(top_n=30).to_csv(importance_out)

        sweep_out = self._data_dir / "results" / f"{symbol}_ml_threshold_sweep.csv"
        sweep = self.ml_generator.threshold_sweep(ml_result.predictions, target)
        sweep.to_csv(sweep_out)

        logger.info(
            f"[{symbol}] ML reports → {fold_out.name}, {importance_out.name}, "
            f"{sweep_out.name}"
        )

    def _save_backtest(self, comparison: pd.DataFrame, symbol: str) -> None:
        out = self._data_dir / "results" / f"{symbol}_backtest.csv"
        comparison.round(4).to_csv(out)
        logger.info(f"[{symbol}] Backtest results → {out}")

    # Base signal -> its regime-gated counterpart column name. Centralised
    # here so the mapping has one source of truth; RegimeFilterPresets
    # determines the actual *_gated/*_pool column names produced.
    _GATING_PAIRS = {
        "signal_rsi":    "signal_rsi_vol_trend_gated",
        "signal_zscore": "signal_zscore_vol_bb_gated",
        "signal_macd":   "signal_macd_trend_gated",
        "signal_bb":     "signal_bb_breakout_gated",
        "signal_ensemble": "signal_regime_adaptive",
    }

    def _save_gating_comparison(
        self,
        comparison: pd.DataFrame,
        symbol: str,
    ) -> Optional[pd.DataFrame]:
        """
        Build and save the gated-vs-naive comparison: for every base signal
        that has a known regime-gated counterpart, report the Sharpe and
        max-drawdown delta directly, plus an explicit verdict.

        This is the single number Phase 8 exists to produce. Without it,
        "did regime-gating help" required manually finding two rows in
        {symbol}_backtest.csv and subtracting them by eye.

        Verdict logic:
            "gating_hurt"                       — gated drawdown got WORSE
                                                   than naive by >2pp. The
                                                   gate is actively
                                                   counterproductive here.
            "gating_helped"                     — gated drawdown improved
                                                   by >2pp AND Sharpe did
                                                   not get meaningfully
                                                   worse. Only evaluated
                                                   when gated_n_trades is
                                                   high enough for Sharpe
                                                   to be a trustworthy
                                                   statistic.
            "gating_helped_but_sharpe_cost"     — drawdown improved but
                                                   Sharpe dropped more than
                                                   the allowed tolerance,
                                                   at a trade count where
                                                   that Sharpe number is
                                                   still meaningful.
            "gating_helped_low_sample"          — drawdown AND total
                                                   return both improved,
                                                   but gated_n_trades is
                                                   below the threshold
                                                   where Sharpe (computed
                                                   on daily returns,
                                                   annualised) is a
                                                   trustworthy statistic.
                                                   A 2-trade signal can
                                                   show Sharpe=-5.4 purely
                                                   from sample-size noise
                                                   even while its absolute
                                                   loss shrank 10x — this
                                                   label says "trust the
                                                   drawdown/return numbers
                                                   here, not the Sharpe
                                                   delta."
            "gating_helped_dd_only_low_sample"  — drawdown improved but
                                                   total return did not,
                                                   at low sample size.
            "inconclusive" / "inconclusive_low_sample" — neither condition
                                                   clearly met.

        Returns:
            The comparison DataFrame, or None if no pairs were available
            (e.g. regime filtering was disabled for this run).
        """
        rows = []
        for base_col, gated_col in self._GATING_PAIRS.items():
            if base_col not in comparison.index or gated_col not in comparison.index:
                continue

            base  = comparison.loc[base_col]
            gated = comparison.loc[gated_col]

            sharpe_delta = float(gated["sharpe"] - base["sharpe"])
            dd_delta     = float(gated["max_drawdown"] - base["max_drawdown"])  # both negative; less negative = improvement
            return_delta = float(gated["total_return"] - base["total_return"])
            gated_trades = int(gated["n_trades"])
            trade_reduction_pct = (
                float(1 - gated["n_trades"] / base["n_trades"]) * 100
                if base["n_trades"] > 0 else 0.0
            )

            dd_improved = dd_delta > 0.02          # gated DD at least 2pp shallower
            dd_worsened = dd_delta < -0.02          # gated DD at least 2pp deeper

            # Sharpe is computed on daily returns and annualised — at very
            # low trade counts (a near-empty return stream with only a
            # handful of non-zero days) it is dominated by noise and is
            # NOT a trustworthy statistic, regardless of its sign or
            # magnitude. A gated signal can show Sharpe=-5.4 on 2 trades
            # while its absolute loss shrank 10x versus the ungated
            # version — the Sharpe number in that case is an artifact of
            # sample size, not a real risk-adjusted-return tradeoff.
            # Below this threshold, judge the gate on drawdown and total
            # return alone, and say so explicitly rather than reporting a
            # misleading Sharpe comparison as if it were meaningful.
            min_trades_for_sharpe = 15
            low_sample_size = gated_trades < min_trades_for_sharpe

            if dd_worsened:
                verdict = "gating_hurt"
            elif low_sample_size:
                if dd_improved and return_delta > -0.02:
                    verdict = "gating_helped_low_sample"
                elif dd_improved:
                    verdict = "gating_helped_dd_only_low_sample"
                else:
                    verdict = "inconclusive_low_sample"
            else:
                sharpe_acceptable = sharpe_delta > -0.5  # allow modest Sharpe cost for safety
                if dd_improved and sharpe_acceptable:
                    verdict = "gating_helped"
                elif dd_improved:
                    verdict = "gating_helped_but_sharpe_cost"
                else:
                    verdict = "inconclusive"

            rows.append({
                "base_signal":           base_col,
                "gated_signal":          gated_col,
                "base_sharpe":           round(float(base["sharpe"]), 4),
                "gated_sharpe":          round(float(gated["sharpe"]), 4),
                "sharpe_delta":          round(sharpe_delta, 4),
                "base_max_drawdown":     round(float(base["max_drawdown"]), 4),
                "gated_max_drawdown":    round(float(gated["max_drawdown"]), 4),
                "max_drawdown_delta":    round(dd_delta, 4),
                "base_total_return":     round(float(base["total_return"]), 4),
                "gated_total_return":    round(float(gated["total_return"]), 4),
                "base_n_trades":         int(base["n_trades"]),
                "gated_n_trades":        gated_trades,
                "trade_reduction_pct":   round(trade_reduction_pct, 1),
                "low_sample_size":       low_sample_size,
                "verdict":               verdict,
            })

        if not rows:
            logger.info(
                f"[{symbol}] No gated/base signal pairs available — "
                f"skipping gating comparison report."
            )
            return None

        result = pd.DataFrame(rows).set_index("base_signal")
        out = self._data_dir / "results" / f"{symbol}_gating_comparison.csv"
        result.to_csv(out)
        logger.info(f"[{symbol}] Gating comparison → {out}")

        for base_col, row in result.iterrows():
            logger.info(
                f"[{symbol}] {base_col:18s} -> {row['gated_signal']:28s} "
                f"DD {row['base_max_drawdown']:+.1%} -> {row['gated_max_drawdown']:+.1%}  "
                f"Sharpe {row['base_sharpe']:+.2f} -> {row['gated_sharpe']:+.2f}  "
                f"[{row['verdict']}]"
            )

        return result

    def _save_watchlist_comparison(
        self,
        backtest_results: Dict[str, Dict],
    ) -> None:
        """Cross-ticker comparison for the ensemble signal."""
        rows = []
        for symbol, bt_map in backtest_results.items():
            if "signal_ensemble" in bt_map:
                row = {"ticker": symbol, **bt_map["signal_ensemble"].metrics}
                rows.append(row)

        if not rows:
            return

        comparison = (
            pd.DataFrame(rows)
            .set_index("ticker")
            .sort_values("sharpe", ascending=False)
        )
        out = self._data_dir / "results" / "watchlist_comparison.csv"
        comparison.round(4).to_csv(out)
        logger.info(f"Watchlist comparison (ensemble) → {out}")

    # ------------------------------------------------------------------ #
    # Utility helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        for sub in ("processed", "results", "raw", "factors"):
            (self._data_dir / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _factor_results_to_json(results) -> dict:
        """Serialise FactorModelResults to a plain JSON-safe dict."""
        out: dict = {"ticker": results.ticker, "regressions": {}}

        for sig, regs in results.regressions.items():
            out["regressions"][sig] = []
            for reg in regs:
                out["regressions"][sig].append({
                    "model":        reg.model_name,
                    "alpha_annual": round(reg.alpha_annual, 6),
                    "alpha_daily":  round(reg.alpha_daily, 8),
                    "t_stat":       round(reg.t_stat, 4),
                    "p_value":      round(reg.p_value, 6),
                    "significant":  bool(reg.is_significant),
                    "r2":           round(reg.r2, 4),
                    "adj_r2":       round(reg.adj_r2, 4),
                    "ic":           round(reg.ic, 4),
                    "ir":           round(reg.information_ratio, 4),
                    "n_obs":        reg.n_obs,
                    "betas":        {k: round(v, 4) for k, v in reg.betas.items()},
                })

        if not results.attribution_table.empty:
            out["attribution_table"] = (
                results.attribution_table
                .reset_index()
                .to_dict(orient="records")
            )

        return out



# ======================================================================
# Dry-run helper
# ======================================================================

def _run_dry_run(symbols: List[str], start_date: str, end_date: str) -> None:
    """
    Validate tickers and estimate data availability without downloading
    or running the pipeline. Prints a summary table and exits.
    """
    import yfinance as yf
    from datetime import date as date_cls
    from market_search import _safe_fast_info_read

    print(f"\n── Dry Run ─ {len(symbols)} symbol(s)  {start_date} → {end_date}\n")
    print(f"  {'Symbol':<12} {'Status':<8} {'Avail From':<14} {'Est. Bars':<12} {'Currency'}")
    print("  " + "─" * 65)

    valid = 0
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            last_price = _safe_fast_info_read(info, "last_price")
            currency = _safe_fast_info_read(info, "currency") or "?"

            if last_price is None:
                print(f"  {sym:<12} {'✗ INVALID':<8} "
                      f"{'no price data — bad symbol or API issue':<14}")
                continue

            # Quick history probe: fetch just 2 bars to find earliest date
            hist = ticker.history(period="max", interval="1d", auto_adjust=True)
            if hist.empty:
                print(f"  {sym:<12} {'✗ NO DATA':<8}")
                continue

            avail_from = hist.index[0].strftime("%Y-%m-%d")
            # Estimate bars in requested range
            req_start = date_cls.fromisoformat(start_date)
            req_end   = date_cls.fromisoformat(end_date)
            hist_in_range = hist.loc[start_date:end_date]
            est_bars = len(hist_in_range)

            print(f"  {sym:<12} {'✓ OK':<8} {avail_from:<14} {est_bars:<12} {currency}")
            valid += 1

        except Exception as exc:
            print(f"  {sym:<12} {'✗ ERROR':<8} {str(exc)[:40]}")

    print(f"\n  {valid}/{len(symbols)} symbols valid\n")
    sys.exit(0)





# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QuantOS Market Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --symbols AAPL MSFT --period 5y
  python pipeline.py --preset crypto --symbols BTC-USD --period max
  python pipeline.py --period ytd --interval 1h --tz US/Eastern
  python pipeline.py --search semiconductor --period 3y
  python pipeline.py --add TSLA NVDA --config config.yaml
  python pipeline.py --symbols AAPL --dry-run --period max
  python pipeline.py --period 5y --format excel
  python pipeline.py --preset equities --period 3y --export-config saved.yaml
        """,
    )

    # ── Data selection ──────────────────────────────────────────────────
    parser.add_argument("--config",   default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--symbols",  nargs="*",
                        help="Override watchlist with specific tickers")
    parser.add_argument("--start",    default=None,
                        help="Start date YYYY-MM-DD (overrides --period)")
    parser.add_argument("--end",      default=None,
                        help="End date YYYY-MM-DD (overrides --period)")
    parser.add_argument(
        "--period",
        default=None,
        metavar="PERIOD",
        help=(
            "Date range shorthand: 1y, 2y, 3y, 5y, 10y, ytd, max. "
            "Explicit --start/--end override this when both are provided."
        ),
    )

    # ── Market/instrument settings ──────────────────────────────────────
    parser.add_argument(
        "--interval",
        default=None,
        choices=["1m","5m","15m","30m","1h","2h","4h","1d","1wk","1mo"],
        help="Data interval (default: from config, usually 1d)",
    )
    parser.add_argument(
        "--tz",
        default=None,
        metavar="TIMEZONE",
        help="Timezone for data (e.g. US/Eastern, Europe/London, UTC, Asia/Tokyo)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["equities", "crypto", "forex"],
        help="Apply a signal preset for a specific market type",
    )

    # ── Pipeline control ────────────────────────────────────────────────
    parser.add_argument(
        "--no-factor", action="store_true",
        help="Skip factor attribution (faster runs during development)",
    )
    parser.add_argument(
        "--ml", action="store_true",
        help="Enable ML signal generation (slow, off by default)",
    )

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet", "excel"],
        help="Output format for result files (default: csv)",
    )
    parser.add_argument(
        "--export-config",
        default=None,
        metavar="PATH",
        help="Save effective config (defaults + overrides) to a new YAML and exit",
    )

    # ── Discovery ───────────────────────────────────────────────────────
    parser.add_argument(
        "--search",
        default=None,
        metavar="QUERY",
        help="Search the ticker database by name/sector/industry and print results",
    )
    parser.add_argument(
        "--search-field",
        default="all",
        choices=["all", "sector", "industry", "name", "symbol"],
        help="Field to search in (default: all)",
    )
    parser.add_argument(
        "--add",
        nargs="*",
        metavar="TICKER",
        help="Add ticker(s) to the watchlist in config.yaml",
    )
    parser.add_argument(
        "--list-sectors",
        action="store_true",
        help="List all sectors in the ticker database and exit",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available signal presets and exit",
    )

    # ── Utility ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate symbols and estimate bar counts without running "
            "the pipeline. Useful before committing to a large run."
        ),
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Handle pure-information commands (no pipeline run) ───────────────

    if args.list_presets:
        print("\nAvailable presets:\n")
        for name, desc in list_presets().items():
            print(f"  {name:<12} {desc}")
        print()
        sys.exit(0)

    if args.list_sectors:
        ms = MarketSearch()
        print("\nSectors in ticker database:\n")
        for sector in ms.sectors():
            tickers = [r["symbol"] for r in ms.search(sector, field="sector")]
            print(f"  {sector:<30} ({len(tickers)} tickers)")
        print()
        sys.exit(0)

    if args.search:
        ms = MarketSearch()
        results = ms.search(args.search, field=args.search_field)
        print(f"\nSearch results for '{args.search}' (field={args.search_field}):\n")
        if not results:
            print("  No results found.\n")
        else:
            print(f"  {'Symbol':<10} {'Sector':<25} {'Industry':<30} Name")
            print("  " + "─" * 80)
            for r in results:
                print(f"  {r['symbol']:<10} {r['sector']:<25} {r['industry']:<30} {r['name']}")
            print(f"\n  {len(results)} result(s). "
                  f"Add to watchlist: --add {' '.join(r['symbol'] for r in results[:5])}\n")
        sys.exit(0)

    if args.add:
        ms = MarketSearch()
        updated = ms.add_to_watchlist(args.add, config_path=args.config)
        print(f"\nWatchlist updated ({len(updated)} total): {updated}\n")
        sys.exit(0)

    # ── Build the pipeline ───────────────────────────────────────────────

    pipeline = MarketDataPipeline(args.config)

    # Apply preset first (lowest priority — config and CLI flags override)
    if args.preset:
        apply_preset(pipeline.config, args.preset)
        logging.getLogger(__name__).info(f"Applied preset: {args.preset}")

    # CLI flags override config (highest priority)
    if args.no_factor:
        pipeline.config["factor"]["enabled"] = False
    if args.ml:
        pipeline.config["ml_signal"]["enabled"] = True
    if args.interval:
        pipeline.config["interval"] = args.interval
    if args.tz:
        pipeline.config["timezone"] = args.tz

    # Resolve date range
    start_date, end_date = resolve_dates(
        period=args.period,
        start=args.start,
        end=args.end,
        default_start="2020-01-01",
    )

    # ── Export config and exit ───────────────────────────────────────────
    if args.export_config:
        # Bake resolved dates into the exported config
        export_cfg = dict(pipeline.config)
        export_cfg["_resolved_start"] = start_date
        export_cfg["_resolved_end"]   = end_date
        export_cfg["_preset"]          = args.preset or "none"
        with open(args.export_config, "w") as f:
            yaml.dump(export_cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"\nEffective config exported to: {args.export_config}\n")
        sys.exit(0)

    symbols = args.symbols if args.symbols else None

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        sym_list = symbols or list(pipeline.config.get("watchlist", []))
        if not sym_list:
            print("No symbols specified. Use --symbols or check watchlist in config.")
            sys.exit(1)
        _run_dry_run(sym_list, start_date, end_date)
        # _run_dry_run calls sys.exit()

    # ── Full pipeline run ────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    # Wire the output format into the pipeline's save methods
    output_fmt = args.format.lower()
    pipeline._output_format = output_fmt

    results = pipeline.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Excel post-processing: flush per-ticker workbooks if format=excel
    if output_fmt == "excel":
        results_dir = pipeline._data_dir / "results"
        try:
            import openpyxl
            for ticker, df in results.items():
                writer_xl = pd.ExcelWriter(
                    results_dir / f"{ticker}_results.xlsx",
                    engine="openpyxl",
                )
                # Load all CSVs for this ticker and write as sheets
                sheet_map = {
                    "backtest":        f"{ticker}_backtest.csv",
                    "gating":          f"{ticker}_gating_comparison.csv",
                    "factor":          None,   # JSON, skip for Excel
                    "regime":          None,
                    "ml_fold":         f"{ticker}_ml_fold_report.csv",
                    "ml_importance":   f"{ticker}_ml_importance.csv",
                    "ml_sweep":        f"{ticker}_ml_threshold_sweep.csv",
                }
                any_sheet = False
                for sheet, filename in sheet_map.items():
                    if filename is None:
                        continue
                    fpath = results_dir / filename
                    if fpath.exists():
                        pd.read_csv(fpath, index_col=0).reset_index().to_excel(
                            writer_xl, sheet_name=sheet[:31], index=False
                        )
                        any_sheet = True
                if any_sheet:
                    writer_xl.close()
                    print(f"  Excel workbook: {results_dir / (ticker + '_results.xlsx')}")
        except ImportError:
            print("  Note: openpyxl not installed — Excel output skipped. "
                  "Install with: pip install openpyxl")

    if args.preset:
        print(f"  preset={args.preset}  period={args.period or 'explicit'}  "
              f"interval={pipeline.config.get('interval','1d')}  "
              f"tz={pipeline.config.get('timezone','US/Eastern')}")