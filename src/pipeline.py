"""
pipeline.py — Orchestrates fetch → clean → feature engineering → save
QuantOS Market Data Pipeline

Usage:
    python pipeline.py                          # uses config.yaml
    python pipeline.py --config my_config.yaml  # custom config
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """
    End-to-end market data pipeline.

    Stages:
        1. Fetch      — yfinance API or local CSV cache
        2. Clean      — timestamps, alignment, missing data, returns
        3. Engineer   — volatility, RSI, ATR, volume, Bollinger, MACD, z-score
        4. Persist    — processed CSV + JSON quality reports

    Config keys (config.yaml):
        data_dir        str   "./data"
        interval        str   "1d"
        timezone        str   "UTC"
        target_freq     str   "D"
        resample_method str   "ffill"
        fill_method     str   "ffill"
        max_gap         int   5
        return_method   str   "log"
        use_cache       bool  true
        watchlist       list  ["AAPL", ...]
        features        dict  (see DEFAULTS below)
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
            "volatility":  {"windows": [5, 21, 63]},
            "rsi":         {"window": 14},
            "atr":         {"window": 14},
            "volume":      {"window": 20},
            "bollinger":   {"window": 20, "num_std": 2.0},
            "macd":        {"fast": 12, "slow": 26, "signal": 9},
            "price_zscore":{"windows": [20, 60]},
        },
    }

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path} — using defaults.")
            self.config = dict(self.DEFAULTS)
        else:
            with open(config_path) as fh:
                loaded = yaml.safe_load(fh) or {}
            # Deep merge features sub-dict
            merged = dict(self.DEFAULTS)
            merged.update({k: v for k, v in loaded.items() if k != "features"})
            if "features" in loaded:
                merged["features"] = {**self.DEFAULTS["features"], **loaded["features"]}
            self.config = merged

        self.fetcher  = MarketDataFetcher(self.config["data_dir"])
        self.cleaner  = DataCleaner()
        self.engineer = FeatureEngineer()
        self._data_dir = Path(self.config["data_dir"])

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute the full pipeline for `symbols`.

        Falls back to config['watchlist'] when symbols is None.
        Returns a dict of fully-processed, feature-enriched DataFrames.
        """
        symbols = symbols or self.config["watchlist"]
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

        # ---- Stages 2-3: Clean + Engineer ----
        processed: Dict[str, pd.DataFrame] = {}
        feat_cfg = self.config["features"]

        for symbol, df in raw_data.items():
            try:
                # Stage 2: Clean
                df_clean = self.cleaner.clean(
                    df,
                    ticker=symbol,
                    timezone=self.config["timezone"],
                    target_freq=self.config["target_freq"],
                    resample_method=self.config["resample_method"],
                    max_gap=self.config["max_gap"],
                    fill_method=self.config["fill_method"],
                    return_method=self.config["return_method"],
                )

                # Stage 3: Feature engineering
                df_feat = self.engineer.add_all_features(
                    df_clean,
                    ticker=symbol,
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

                # Stage 4: Persist
                self._save(df_feat, symbol)
                self._save_reports(df_clean, df_feat, symbol)
                processed[symbol] = df_feat

            except Exception as exc:
                logger.error(f"[{symbol}] Processing failed: {exc}", exc_info=True)

        logger.info(
            f"Pipeline complete — "
            f"{len(processed)}/{len(raw_data)} symbols processed."
        )
        return processed

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
        symbol: str,
    ) -> None:
        # Data quality report (from cleaner)
        clean_report = self.cleaner.quality_report(df_clean, ticker=symbol)
        clean_out = self._data_dir / "processed" / f"{symbol}_data_report.json"
        with open(clean_out, "w") as fh:
            json.dump(clean_report, fh, indent=2)

        # Feature quality report
        feat_report = self.engineer.feature_report(df_feat, ticker=symbol)
        feat_out = self._data_dir / "processed" / f"{symbol}_feature_report.json"
        with open(feat_out, "w") as fh:
            json.dump(feat_report, fh, indent=2, default=str)

        logger.info(f"[{symbol}] Reports saved → data_report.json, feature_report.json")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantOS Market Data Pipeline")
    parser.add_argument("--config",  default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--symbols", nargs="*",             help="Override watchlist")
    parser.add_argument("--start",   default="2020-01-01",  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2023-12-31",  help="End date YYYY-MM-DD")
    args = parser.parse_args()

    pipeline = MarketDataPipeline(args.config)
    results  = pipeline.run(
        symbols=args.symbols or None,
        start_date=args.start,
        end_date=args.end,
    )

    print("\n── Summary ─────────────────────────────────────────────────────")
    for ticker, df in results.items():
        feat_cols = [c for c in df.columns
                     if c not in {"open","high","low","close","volume",
                                  "returns","returns_norm","returns_fwd_1",
                                  "returns_fwd_5","tr"}]
        nan_pct = df[feat_cols].isnull().mean().mean() * 100
        print(
            f"  {ticker:6s}  rows={len(df):5d}  "
            f"features={len(feat_cols):2d}  "
            f"feat_NaN={nan_pct:.1f}%"
        )