"""
feature_engineering.py — Phase 2: Predictive feature computation
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → [engineer features] → signal → backtest

All methods are pure functions: DataFrame in, DataFrame out.
No lookahead bias: every computation uses only past observations.

Feature families:
    1. Realised Volatility  — risk measurement, position sizing
    2. RSI-14               — momentum oscillator
    3. ATR-14               — gap-aware volatility, stop-loss sizing
    4. Volume Ratio         — signal confirmation
    5. Bollinger Bands      — volatility envelopes, breakout detection
    6. MACD                 — trend-following momentum
    7. Price Z-Score        — mean-reversion signal

Usage:
    engineer = FeatureEngineer()
    df = engineer.add_all_features(df)

    # Or selectively:
    df = engineer.add_realised_volatility(df)
    df = engineer.add_rsi(df)
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Compute predictive features from cleaned OHLCV DataFrames.

    Expects the output of DataCleaner.clean() as input:
    columns: open, high, low, close, volume, returns, returns_norm,
             returns_fwd_1, returns_fwd_5
    index:   DatetimeIndex (tz-aware, uniform frequency)
    """

    # ------------------------------------------------------------------ #
    # High-level entry point                                               #
    # ------------------------------------------------------------------ #

    def add_all_features(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        *,
        vol_windows: List[int] = (5, 21, 63),
        rsi_window: int = 14,
        atr_window: int = 14,
        volume_window: int = 20,
        bb_window: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        zscore_windows: List[int] = (20, 60),
    ) -> pd.DataFrame:
        """
        Run all feature families in dependency order.

        Args:
            df:             Cleaned OHLCV DataFrame from DataCleaner
            ticker:         Label for log messages
            vol_windows:    Rolling windows for realised volatility (days)
            rsi_window:     RSI lookback period
            atr_window:     ATR lookback period
            volume_window:  Window for volume ratio computation
            bb_window:      Bollinger Band SMA window
            bb_std:         Number of standard deviations for bands
            macd_fast:      MACD fast EMA period
            macd_slow:      MACD slow EMA period
            macd_signal:    MACD signal line EMA period
            zscore_windows: Rolling windows for price z-score

        Returns:
            DataFrame with all feature columns appended.
        """
        tag = f"[{ticker}] " if ticker else ""
        logger.info(f"{tag}Starting feature engineering.")
        n_cols_before = df.shape[1]

        df = self.add_realised_volatility(df, windows=vol_windows)
        df = self.add_rsi(df, window=rsi_window)
        df = self.add_atr(df, window=atr_window)
        df = self.add_volume_features(df, window=volume_window)
        df = self.add_bollinger_bands(df, window=bb_window, num_std=bb_std)
        df = self.add_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df = self.add_price_zscore(df, windows=zscore_windows)

        n_added = df.shape[1] - n_cols_before
        logger.info(
            f"{tag}Feature engineering complete — "
            f"{n_added} columns added, {df.shape[1]} total."
        )
        return df

    # ------------------------------------------------------------------ #
    # 1. Realised Volatility                                               #
    # ------------------------------------------------------------------ #

    def add_realised_volatility(
        self,
        df: pd.DataFrame,
        windows: List[int] = (5, 21, 63),
        price_col: str = "returns",
        min_periods_factor: float = 0.5,
    ) -> pd.DataFrame:
        """
        Rolling realised volatility — annualised.

        Formula:
            sigma_daily(t, N) = std(r_{t-N+1}, ..., r_t)
            sigma_annual(t, N) = sigma_daily(t, N) * sqrt(252)

        The sqrt(252) scaling derives from variance additivity:
            Var(r_1 + ... + r_252) = 252 * Var(r_i)  [i.i.d. assumption]
        Taking square root: sigma_annual = sigma_daily * sqrt(252).

        Columns added:
            vol_{N}d        — N-day rolling std of log returns
            vol_{N}d_annual — Annualised version (daily * sqrt(252))

        Args:
            df:                   Input DataFrame (must contain `price_col`)
            windows:              List of rolling window sizes in trading days
            price_col:            Column to compute vol from (default: 'returns')
            min_periods_factor:   Minimum periods = window * factor (avoids
                                  unstable early estimates)
        """
        self._require_columns(df, [price_col], "add_realised_volatility")
        df = df.copy()

        for w in windows:
            min_p = max(2, int(w * min_periods_factor))
            daily_vol = df[price_col].rolling(window=w, min_periods=min_p).std()
            df[f"vol_{w}d"] = daily_vol
            df[f"vol_{w}d_annual"] = daily_vol * np.sqrt(252)
            logger.debug(f"  vol_{w}d computed (min_periods={min_p}).")

        return df

    # ------------------------------------------------------------------ #
    # 2. RSI — Relative Strength Index                                     #
    # ------------------------------------------------------------------ #

    def add_rsi(
        self,
        df: pd.DataFrame,
        window: int = 14,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Wilder's RSI — momentum oscillator on [0, 100].

        Formula:
            delta_t  = P_t - P_{t-1}
            gain_t   = max(delta_t, 0)
            loss_t   = max(-delta_t, 0)

            avg_gain = EMA(gain, alpha=1/N, adjust=False)
            avg_loss = EMA(loss, alpha=1/N, adjust=False)

            RS  = avg_gain / avg_loss
            RSI = 100 - 100 / (1 + RS)

        Edge cases:
            avg_loss = 0  → RSI = 100  (all gains, no losses)
            avg_gain = 0  → RSI = 0    (all losses, no gains)

        Wilder's smoothing: alpha = 1/N gives a longer effective memory
        than standard EMA(span=N). This is the industry-standard RSI
        implementation (matches TradingView, Bloomberg).

        Columns added:
            rsi_{N}   — N-period RSI

        Interpretation:
            RSI > 70: overbought → potential mean-reversion short
            RSI < 30: oversold  → potential mean-reversion long
            RSI = 50: neutral momentum
        """
        self._require_columns(df, [price_col], "add_rsi")
        df = df.copy()

        delta = df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's exponential smoothing: alpha = 1/N
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Restore edge cases: avg_loss=0 → RSI=100
        rsi = rsi.where(avg_loss != 0, 100.0)

        df[f"rsi_{window}"] = rsi.clip(0, 100)
        logger.debug(f"  rsi_{window} computed (Wilder's smoothing, alpha={1/window:.4f}).")
        return df

    # ------------------------------------------------------------------ #
    # 3. ATR — Average True Range                                          #
    # ------------------------------------------------------------------ #

    def add_atr(
        self,
        df: pd.DataFrame,
        window: int = 14,
    ) -> pd.DataFrame:
        """
        Average True Range — gap-aware volatility measure.

        True Range accounts for overnight gaps (common in equities),
        which high-low alone misses:

            TR_t = max(
                high_t - low_t,           (intraday range)
                |high_t - close_{t-1}|,   (gap-up + intraday)
                |low_t  - close_{t-1}|    (gap-down + intraday)
            )

        ATR = EMA(TR, alpha=1/N)  [Wilder's smoothing]

        TR >= high - low always, because the max() can only increase it.

        Columns added:
            tr        — True Range
            atr_{N}   — N-period Average True Range

        Use cases:
            - Position sizing: risk 1% of capital per ATR unit
            - Stop-loss: place stop at entry ± k*ATR (typically k=1.5-2)
            - Volatility filter: only trade when ATR is above threshold
        """
        self._require_columns(df, ["high", "low", "close"], "add_atr")
        df = df.copy()

        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["tr"] = tr
        df[f"atr_{window}"] = tr.ewm(
            alpha=1.0 / window, adjust=False, min_periods=window
        ).mean()

        logger.debug(f"  atr_{window} computed (True Range with gap accounting).")
        return df

    # ------------------------------------------------------------------ #
    # 4. Volume Features                                                   #
    # ------------------------------------------------------------------ #

    def add_volume_features(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Volume-based signal confirmation features.

        A price move on high volume is more significant than the same
        move on low volume. Volume features help filter false signals.

        vol_ratio_{N}:
            Current volume / N-day rolling average volume.
            > 1.5: above-average activity — breakout candidate
            < 0.5: thin market — unreliable price action

        vwap_{N}d:
            Volume-Weighted Average Price over N days.
            VWAP = sum(P_t * V_t) / sum(V_t)
            Used as fair-value benchmark. Price above VWAP = bullish.

        vwap_dev:
            Percentage deviation of close from VWAP.
            Normalises VWAP across tickers with different price levels.

        Columns added:
            vol_ratio_{N}  — current / rolling avg volume
            vwap_{N}d      — N-day VWAP
            vwap_dev       — (close - vwap) / close
        """
        self._require_columns(df, ["close", "volume"], "add_volume_features")
        df = df.copy()

        # Volume ratio
        avg_vol = df["volume"].rolling(window=window, min_periods=max(2, window // 2)).mean()
        df[f"vol_ratio_{window}"] = df["volume"] / avg_vol.replace(0, np.nan)

        # Rolling VWAP
        pv = df["close"] * df["volume"]
        rolling_pv  = pv.rolling(window=window, min_periods=max(2, window // 2)).sum()
        rolling_vol = df["volume"].rolling(window=window, min_periods=max(2, window // 2)).sum()
        vwap = rolling_pv / rolling_vol.replace(0, np.nan)
        df[f"vwap_{window}d"] = vwap

        # VWAP deviation (percentage)
        df["vwap_dev"] = (df["close"] - vwap) / df["close"].replace(0, np.nan)

        logger.debug(f"  volume_features computed (window={window}).")
        return df

    # ------------------------------------------------------------------ #
    # 5. Bollinger Bands                                                   #
    # ------------------------------------------------------------------ #

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Bollinger Bands — volatility envelopes around a moving average.

        Formulas:
            middle = SMA(close, N)
            upper  = middle + k * rolling_std(close, N)
            lower  = middle - k * rolling_std(close, N)

            bb_width = (upper - lower) / middle   [normalised bandwidth]
            bb_pct   = (close - lower) / (upper - lower)  [% within bands]

        bb_pct interpretation:
            bb_pct > 1: price broke above upper band (momentum signal)
            bb_pct < 0: price broke below lower band (oversold signal)
            bb_pct = 0.5: price at middle band

        bb_width interpretation:
            Low width ("squeeze"): low volatility → often precedes breakout
            High width: high volatility, trending market

        upper > middle > lower is guaranteed by construction (std >= 0).

        Columns added:
            bb_middle  — N-period SMA
            bb_upper   — upper band (middle + k*sigma)
            bb_lower   — lower band (middle - k*sigma)
            bb_width   — normalised bandwidth
            bb_pct     — price position within bands [0,1], can exceed
        """
        self._require_columns(df, [price_col], "add_bollinger_bands")
        df = df.copy()

        roll = df[price_col].rolling(window=window, min_periods=max(2, window // 2))
        middle = roll.mean()
        std    = roll.std()

        upper = middle + num_std * std
        lower = middle - num_std * std

        df["bb_middle"] = middle
        df["bb_upper"]  = upper
        df["bb_lower"]  = lower
        df["bb_width"]  = (upper - lower) / middle.replace(0, np.nan)
        df["bb_pct"]    = (df[price_col] - lower) / (upper - lower).replace(0, np.nan)

        logger.debug(f"  bollinger_bands computed (window={window}, std={num_std}).")
        return df

    # ------------------------------------------------------------------ #
    # 6. MACD                                                              #
    # ------------------------------------------------------------------ #

    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        MACD — Moving Average Convergence Divergence.

        Trend-following momentum indicator derived from EMA crossovers.

        Formulas:
            ema_fast     = EMA(close, fast)
            ema_slow     = EMA(close, slow)
            macd_line    = ema_fast - ema_slow
            signal_line  = EMA(macd_line, signal)
            histogram    = macd_line - signal_line

        Interpretation:
            macd_line > 0:   fast EMA above slow EMA — uptrend
            macd_line < 0:   fast EMA below slow EMA — downtrend
            histogram > 0:   MACD above signal — bullish momentum building
            Sign change in histogram → crossover signal

        Note: MACD and RSI are both momentum signals. RSI is preferred
        for mean-reversion strategies; MACD for trend-following.

        Columns added:
            ema_{fast}      — fast EMA
            ema_{slow}      — slow EMA
            macd_line       — fast - slow EMA
            macd_signal     — signal line (EMA of MACD line)
            macd_histogram  — histogram (line - signal)
        """
        self._require_columns(df, [price_col], "add_macd")
        df = df.copy()

        ema_fast = df[price_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False, min_periods=slow).mean()

        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

        df[f"ema_{fast}"]    = ema_fast
        df[f"ema_{slow}"]    = ema_slow
        df["macd_line"]      = macd_line
        df["macd_signal"]    = signal_line
        df["macd_histogram"] = macd_line - signal_line

        logger.debug(f"  macd computed (fast={fast}, slow={slow}, signal={signal}).")
        return df

    # ------------------------------------------------------------------ #
    # 7. Price Z-Score                                                     #
    # ------------------------------------------------------------------ #

    def add_price_zscore(
        self,
        df: pd.DataFrame,
        windows: List[int] = (20, 60),
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Rolling price z-score — mean-reversion signal.

        Formula:
            z_price(t, N) = (P_t - mu_{t-N,t}) / sigma_{t-N,t}

        Measures how many standard deviations the current price is above
        or below its N-day rolling mean. Uses only past observations:
        no lookahead bias.

        Interpretation:
            z > +2: price is 2 std above recent mean → possible short
            z < -2: price is 2 std below recent mean → possible long
            Persistent positive z → trending up (not mean-reverting)

        Note: equity prices trend over long horizons. A 60-day z-score
        during a bull market will be persistently positive. This is
        correct — use shorter windows for faster mean-reversion signals.

        Columns added:
            z_price_{N}d  — N-day rolling price z-score
        """
        self._require_columns(df, [price_col], "add_price_zscore")
        df = df.copy()

        for w in windows:
            roll = df[price_col].rolling(window=w, min_periods=max(2, w // 2))
            df[f"z_price_{w}d"] = (
                (df[price_col] - roll.mean()) / roll.std().replace(0, np.nan)
            )
            logger.debug(f"  z_price_{w}d computed.")

        return df

    # ------------------------------------------------------------------ #
    # Quality report                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def feature_report(df: pd.DataFrame, ticker: str = "") -> dict:
        """
        Return feature quality metrics as a plain dict.

        Includes: NaN percentages, basic stats, and correlation matrix
        for all feature columns (excludes OHLCV and return columns).
        """
        base_cols = {"open", "high", "low", "close", "volume",
                     "returns", "returns_norm", "returns_fwd_1", "returns_fwd_5", "tr"}
        feature_cols = [c for c in df.columns if c not in base_cols]

        if not feature_cols:
            return {"ticker": ticker, "warning": "No feature columns found."}

        feat_df = df[feature_cols].select_dtypes(include=[np.number])

        return {
            "ticker": ticker or "unknown",
            "n_features": len(feat_df.columns),
            "feature_columns": list(feat_df.columns),
            "missing_pct": (feat_df.isnull().mean() * 100).round(2).to_dict(),
            "describe": feat_df.describe().round(4).to_dict(),
            "correlation": feat_df.corr().round(3).to_dict(),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: List[str], method: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"{method}() requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )