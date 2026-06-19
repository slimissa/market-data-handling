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
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from data_fetcher import MarketDataFetcher
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from signal_generator import SignalGenerator
from backtester import VectorisedBacktester, TransactionCostModel
from factor_model import FactorModel, RegimeAnalyser
from regime_filter import RegimeFilteredEnsemble, RegimeFilterPresets
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
                if k in ("features", "signals", "backtest", "factor", "regime_filter", "regime"):
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
        self.regime_filter_ensemble = RegimeFilteredEnsemble()

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
                    if self.config["regime"].get("adaptive_enabled", True):
                        df_signals["signal_adaptive"] = self.adaptive_switch.apply_discrete(
                            df_signals, result.probabilities, threshold=0.0
                        )
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
                        bt = self.backtester.run(
                            df_signals, signal_col=sig_col, ticker=symbol
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
                    daily_returns_all[symbol] = {
                        sig: res.daily_returns for sig, res in bt_results_map.items()
                    }

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

    def _save_backtest(self, comparison: pd.DataFrame, symbol: str) -> None:
        out = self._data_dir / "results" / f"{symbol}_backtest.csv"
        comparison.round(4).to_csv(out)
        logger.info(f"[{symbol}] Backtest results → {out}")

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
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantOS Market Data Pipeline")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default="2023-12-31")
    parser.add_argument(
        "--no-factor", action="store_true",
        help="Skip factor attribution (faster runs during development)"
    )
    args = parser.parse_args()

    pipeline = MarketDataPipeline(args.config)

    if args.no_factor:
        pipeline.config["factor"]["enabled"] = False

    results = pipeline.run(
        symbols=args.symbols if args.symbols else None,
        start_date=args.start,
        end_date=args.end,
    )

    # ── CLI summary ──────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    base_cols = {
        "open", "high", "low", "close", "volume", "returns",
        "returns_norm", "returns_fwd_1", "returns_fwd_5", "tr",
    }
    for ticker, df in results.items():
        feat_cols = [
            c for c in df.columns
            if c not in base_cols
            and not c.startswith("signal_")
            and c != "position_scale"
        ]
        sig_cols = [c for c in df.columns if c.startswith("signal_")]

        sharpe_str = ""
        bt_path = pipeline._data_dir / "results" / f"{ticker}_backtest.csv"
        if bt_path.exists():
            try:
                bt_df = pd.read_csv(bt_path, index_col=0)
                if "signal_ensemble" in bt_df.index and "sharpe" in bt_df.columns:
                    s = bt_df.loc["signal_ensemble", "sharpe"]
                    sharpe_str = f"  ensemble_Sharpe={s:+.2f}"
            except Exception:
                pass
        
        adaptive_str = ""
        if bt_path.exists():
            try:
                bt_df = pd.read_csv(bt_path, index_col=0)
                if "signal_adaptive" in bt_df.index and "sharpe" in bt_df.columns:
                    a = bt_df.loc["signal_adaptive", "sharpe"]
                    adaptive_str = f"  adaptive_Sharpe={a:+.2f}"
            except Exception:
                pass

        alpha_str = ""
        if pipeline.config["factor"]["enabled"]:
            f_path = pipeline._data_dir / "results" / f"{ticker}_factor_report.json"
            if f_path.exists():
                try:
                    with open(f_path) as fh:
                        f_data = json.load(fh)
                    ens_regs = f_data.get("regressions", {}).get("signal_ensemble", [])
                    capm = next((r for r in ens_regs if r["model"] == "CAPM"), None)
                    if capm:
                        sig_flag = "✓" if capm["significant"] else "✗"
                        alpha_str = (
                            f"  CAPM_alpha={capm['alpha_annual']:+.3f}"
                            f"(t={capm['t_stat']:+.2f}{sig_flag})"
                        )
                except Exception:
                    pass
        else:
            alpha_str = "  CAPM_alpha=N/A"

        print(
            f"  {ticker:6s}  rows={len(df):5d}  "
            f"features={len(feat_cols):2d}  "
            f"signals={len(sig_cols):2d}"
            f"{sharpe_str}{adaptive_str}{alpha_str}"
        )

    print(f"\nOutputs in: {pipeline._data_dir}/")
    print(f"  processed/   — CSVs + quality reports")
    print(f"  results/     — backtest CSVs + factor reports + rolling alpha")