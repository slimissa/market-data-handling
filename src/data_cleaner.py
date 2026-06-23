"""
data_cleaner.py — Market data cleaning and normalisation layer

Pipeline order (enforced by clean()):
    1. standardise_columns
    2. clean_timestamps
    3. align_frequency
    4. handle_missing_data
    5. calculate_returns
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and standardise OHLCV market data.

    Usage (low-level):
        cleaner = DataCleaner()
        df = cleaner.clean(raw_df, ticker="AAPL")

    Or call individual steps directly for custom pipelines.
    """

    # ------------------------------------------------------------------ #
    # Public high-level entry point                                        #
    # ------------------------------------------------------------------ #

    def clean(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        *,
        timezone: str = "UTC",
        target_freq: str = "D",
        resample_method: str = "ffill",
        max_gap: Optional[int] = 5,
        fill_method: str = "ffill",
        return_method: str = "log",
    ) -> pd.DataFrame:
        """
        Run the full cleaning pipeline in the correct order.

        Args:
            df:              Raw OHLCV DataFrame
            ticker:          Label used in log messages
            timezone:        Target timezone for DatetimeIndex
            target_freq:     Pandas offset alias (D, H, T, …)
            resample_method: Gap-filling strategy after reindex
            max_gap:         Max consecutive NaN periods to fill; None = unlimited
            fill_method:     ffill | bfill | interpolate
            return_method:   simple | log

        Returns:
            Cleaned DataFrame with returns columns appended.
        """
        tag = f"[{ticker}] " if ticker else ""
        logger.info(f"{tag}Starting cleaning pipeline.")

        df = self.standardise_columns(df)
        df = self.clean_timestamps(df, timezone=timezone)
        df = self.align_frequency(df, target_freq=target_freq, method=resample_method)
        df = self.handle_missing_data(df, max_gap=max_gap, fill_method=fill_method)
        df = self.calculate_returns(df, method=return_method)

        logger.info(f"{tag}Cleaning complete — {len(df)} rows, {df.shape[1]} cols.")
        return df

    # ------------------------------------------------------------------ #
    # Step 1 — column names                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase all column names; replace spaces with underscores."""
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Step 2 — timestamps                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def clean_timestamps(
        df: pd.DataFrame,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """
        Ensure a clean, tz-aware, sorted DatetimeIndex without duplicates.

        Handles:
        - Non-datetime index → coerced to datetime
        - Tz-naive index     → localised to `timezone`
        - Tz-aware index     → converted to `timezone`
        - Duplicate timestamps → first occurrence kept
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception as exc:
                raise ValueError(f"Cannot convert index to DatetimeIndex: {exc}") from exc

        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        else:
            df.index = df.index.tz_convert(timezone)

        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            logger.warning(f"Dropping {n_dupes} duplicate timestamps (keeping first).")
            df = df[~df.index.duplicated(keep="first")]

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------ #
    # Step 3 — frequency alignment                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def align_frequency(
        df: pd.DataFrame,
        target_freq: str = "D",
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Reindex to a uniform calendar grid at `target_freq`.

        Args:
            df:          Input DataFrame (must have DatetimeIndex)
            target_freq: Pandas offset alias — 'D', 'H', 'T', etc.
            method:      How to fill newly introduced NaNs:
                         'ffill' | 'bfill' | 'interpolate'

        Note:
            This does NOT drop weekends/holidays.  For equities use
            handle_missing_data() with an appropriate max_gap to avoid
            forward-filling across non-trading periods.
        """
        df = df.copy()
        before = len(df)

        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=target_freq,
            tz=df.index.tz,
        )
        df = df.reindex(full_range)
        df.index.name = "datetime"

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill()
        elif method == "bfill":
            df[numeric_cols] = df[numeric_cols].bfill()
        elif method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        else:
            raise ValueError(f"Unknown align method: '{method}'")

        logger.info(
            f"align_frequency: {before} → {len(df)} rows "
            f"(+{len(df) - before} added at {target_freq})."
        )
        return df

    # ------------------------------------------------------------------ #
    # Step 4 — missing data                                                #
    # ------------------------------------------------------------------ #

    def handle_missing_data(
        self,
        df: pd.DataFrame,
        max_gap: Optional[int] = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Fill remaining NaNs, optionally capping fill length.

        Args:
            df:          Input DataFrame
            max_gap:     Max consecutive NaN periods to fill.
                         None means fill everything.
            fill_method: 'ffill' | 'bfill' | 'interpolate'

        Design note:
            We use pandas' built-in `limit=` parameter instead of the
            custom mask approach in the original — simpler and faster.
        """
        df = df.copy()
        n_missing_before = df.isnull().sum().sum()

        if n_missing_before == 0:
            return df

        logger.info(f"handle_missing_data: {n_missing_before} NaN cells found.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if fill_method == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill(limit=max_gap)
        elif fill_method == "bfill":
            df[numeric_cols] = df[numeric_cols].bfill(limit=max_gap)
        elif fill_method == "interpolate":
            # interpolate doesn't support `limit` the same way; apply manually
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="time", limit=max_gap, limit_direction="forward"
            )
        else:
            raise ValueError(f"Unknown fill method: '{fill_method}'")

        n_missing_after = df.isnull().sum().sum()
        n_filled = n_missing_before - n_missing_after
        logger.info(
            f"handle_missing_data: filled {n_filled} cells "
            f"({n_missing_after} remain unfilled)."
        )
        if n_missing_after > 0:
            logger.warning(
                f"handle_missing_data: {n_missing_after} NaN cells remain after fill "
                f"(gaps longer than max_gap={max_gap} or at series start/end). "
                f"Downstream feature computation will propagate these NaNs."
            )
        return df

    # ------------------------------------------------------------------ #
    # Step 5 — returns                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        price_col: str = "close",
        method: str = "log",
        periods: int = 1,
    ) -> pd.DataFrame:
        """
        Append return columns to the DataFrame.

        Columns added:
            returns          — simple or log return (period=`periods`)
            returns_norm     — z-score of returns (rolling 252-day window)
            returns_fwd_1    — next-period return (for target labelling)
            returns_fwd_5    — 5-period forward return

        Args:
            df:         Input DataFrame
            price_col:  Column with price series (default: 'close')
            method:     'simple' (pct_change) or 'log' (ln ratio) — log
                        is preferred for quant work: additive across time,
                        symmetric, and approximately normally distributed.
            periods:    Lookback for the primary return calculation
        """
        if price_col not in df.columns:
            raise KeyError(
                f"Price column '{price_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        df = df.copy()
        price = df[price_col]

        # ---- Primary return ----
        if method == "simple":
            df["returns"] = price.pct_change(periods)
        elif method == "log":
            df["returns"] = np.log(price / price.shift(periods))
        else:
            raise ValueError(f"Unknown return method: '{method}'")

        # ---- Rolling z-score (252-day window = ~1 trading year) ----
        # Avoids lookahead bias — uses only past observations.
        roll = df["returns"].rolling(window=252, min_periods=30)
        df["returns_norm"] = (df["returns"] - roll.mean()) / roll.std()

        # ---- Forward returns (target labels for ML) ----
        # Negative shift = look forward; NaN at tail is expected.
        # Uses the same method as the primary return so the ML target
        # is on the same scale and distribution as the feature returns.
        if method == "simple":
            df["returns_fwd_1"] = price.pct_change(-1)
            df["returns_fwd_5"] = price.pct_change(-5)
        else:  # log
            df["returns_fwd_1"] = np.log(price.shift(-1) / price)
            df["returns_fwd_5"] = np.log(price.shift(-5) / price)

        return df

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def quality_report(df: pd.DataFrame, ticker: str = "") -> dict:
        """
        Return a dict of data quality metrics (no file I/O).

        Keeps reporting logic out of the pipeline — callers decide
        whether to print, log, or save.
        """
        tag = ticker or "unknown"
        numeric = df.select_dtypes(include=[np.number])

        return {
            "ticker": tag,
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "rows": len(df),
            "columns": list(df.columns),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "returns_skew": round(df["returns"].skew(), 4) if "returns" in df else None,
            "returns_kurt": round(df["returns"].kurtosis(), 4) if "returns" in df else None,
            "describe": numeric.describe().round(4).to_dict(),
        }