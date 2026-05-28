"""
pipeline.py — Orchestrates fetch → clean → save for a symbol list.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from data_fetcher import MarketDataFetcher
from data_cleaner import DataCleaner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """
    End-to-end market data pipeline: fetch → clean → persist → report.

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
    """

    DEFAULTS = {
        "data_dir": "./data",
        "interval": "1d",
        "timezone": "UTC",
        "target_freq": "D",
        "resample_method": "ffill",
        "fill_method": "ffill",
        "max_gap": 5,
        "return_method": "log",
        "use_cache": True,
        "watchlist": [],
    }

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path} — using defaults.")
            self.config = dict(self.DEFAULTS)
        else:
            with open(config_path) as fh:
                loaded = yaml.safe_load(fh) or {}
            self.config = {**self.DEFAULTS, **loaded}

        self.fetcher = MarketDataFetcher(self.config["data_dir"])
        self.cleaner = DataCleaner()
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
        Returns a dict of fully-processed DataFrames.
        """
        symbols = symbols or self.config["watchlist"]
        if not symbols:
            raise ValueError("No symbols provided and watchlist is empty.")

        logger.info(f"Pipeline start — {len(symbols)} symbol(s): {symbols}")

        # Step 1: fetch
        raw_data = self.fetcher.fetch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=self.config["interval"],
            use_cache=self.config["use_cache"],
        )

        # Step 2: clean + persist
        processed: Dict[str, pd.DataFrame] = {}
        for symbol, df in raw_data.items():
            try:
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
                self._save(df_clean, symbol)
                self._save_report(df_clean, symbol)
                processed[symbol] = df_clean
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

    def _save_report(self, df: pd.DataFrame, symbol: str) -> None:
        report = self.cleaner.quality_report(df, ticker=symbol)
        out = self._data_dir / "processed" / f"{symbol}_report.json"
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info(f"[{symbol}] Quality report → {out}")


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    pipeline = MarketDataPipeline("config.yaml")
    results = pipeline.run(
        symbols=["AAPL", "GOOGL", "MSFT", "SPY"],
        start_date="2020-01-01",
        end_date="2023-12-31",
    )

    # Quick sanity check
    for ticker, df in results.items():
        print(
            f"{ticker:6s}  rows={len(df):4d}  "
            f"NaN={df.isnull().sum().sum():3d}  "
            f"cols={list(df.columns)}"
        )