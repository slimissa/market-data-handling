"""
data_fetcher.py — Market data ingestion layer
Supports yfinance (live) and CSV (offline/backtest).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetch historical OHLCV data from Yahoo Finance or local CSVs.

    Usage:
        fetcher = MarketDataFetcher(data_dir="./data")
        data = fetcher.fetch(["AAPL", "MSFT"], "2020-01-01", "2023-12-31")
    """

    VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m",
        "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
    }

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self._ensure_directories()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for a list of symbols.

        Args:
            symbols:    Ticker list, e.g. ["AAPL", "SPY"]
            start_date: ISO date string "YYYY-MM-DD"
            end_date:   ISO date string "YYYY-MM-DD"
            interval:   Any yfinance-supported interval
            use_cache:  If True, return cached CSV when available

        Returns:
            Dict mapping symbol → raw DataFrame (DatetimeIndex, UTC)
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Choose from: {sorted(self.VALID_INTERVALS)}"
            )

        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            cache_path = self.data_dir / "raw" / f"{symbol}_{interval}.csv"

            if use_cache and cache_path.exists():
                logger.info(f"[{symbol}] Loading from cache: {cache_path}")
                df = self._load_csv(cache_path)
            else:
                df = self._fetch_yfinance(symbol, start_date, end_date, interval)
                if df is not None:
                    self._save_csv(df, cache_path)

            if df is not None and not df.empty:
                results[symbol] = df

        logger.info(f"Fetched {len(results)}/{len(symbols)} symbols successfully.")
        return results

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load an arbitrary OHLCV CSV with datetime index."""
        return self._load_csv(Path(filepath))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        try:
            logger.info(f"[{symbol}] Fetching from Yahoo Finance ...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,   # adjusts for splits/dividends
                actions=False,      # drop Dividends/Stock Splits cols
            )
            if df.empty:
                logger.warning(f"[{symbol}] No data returned — check symbol/dates.")
                return None

            df.index.name = "datetime"
            # Normalise column names immediately
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            logger.info(f"[{symbol}] Fetched {len(df)} rows ({interval}).")
            return df

        except Exception as exc:
            logger.error(f"[{symbol}] Fetch failed: {exc}")
            return None

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "datetime"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        logger.info(f"Loaded {len(df)} rows from {path.name}.")
        return df

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path)
        logger.info(f"Cached raw data → {path}")

    def _ensure_directories(self) -> None:
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)