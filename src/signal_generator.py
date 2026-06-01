"""
signal_generator.py — Phase 3: Rule-based signal generation
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → engineer features → [generate signals] → backtest

Signal convention:
    +1  long  (expect price to rise)
    -1  short (expect price to fall)
     0  flat  (no position / no opinion)

Design principles:
    - Pure functions: DataFrame in, DataFrame out. No hidden state.
    - No lookahead bias: signal at t uses only data available at t.
    - Explicit entry AND exit rules for every signal.
    - Crossover detection via shift(1) comparison — never future data.
    - Signal strength ∈ [0, 1] alongside each binary signal.
    - Turnover and correlation diagnostics built into quality report.

Signal families:
    1. RSI mean-reversion      — oversold/overbought threshold crossings
    2. MACD trend-following    — histogram sign-change crossovers
    3. Z-score mean-reversion  — price deviation from rolling mean
    4. Bollinger breakout      — band breach with width/squeeze filter
    5. Vol scale overlay       — position multiplier from volatility regime
    6. Ensemble                — majority vote, weighted, or regime-switch

Usage:
    sg = SignalGenerator()
    df_signals = sg.generate_all(df_featured)

    # Or selectively:
    df["signal_rsi"] = sg.generate_rsi_signal(df)
    df["signal_macd"] = sg.generate_macd_signal(df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generate trading signals from feature-enriched DataFrames.

    Expects the output of FeatureEngineer.add_all_features() as input.
    All signal columns contain only values in {-1, 0, +1}.
    Strength columns contain values in [0, 1].
    """

    # ------------------------------------------------------------------ #
    # High-level entry point                                               #
    # ------------------------------------------------------------------ #

    def generate_all(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        *,
        # RSI params
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_exit: float = 50.0,
        rsi_smoothing: int = 1,
        # MACD params
        macd_require_zero_cross: bool = False,
        # Z-score params
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.0,
        zscore_window: int = 60,
        # Bollinger params
        bb_squeeze_percentile: float = 20.0,
        bb_max_holding_bars: int = 10,
        # Vol scale params
        vol_window: int = 21,
        vol_lookback: int = 252,
        vol_scale_floor: float = 0.0,
        vol_scale_ceiling: float = 2.0,
        # Holding period constraints
        min_holding_bars: int = 2,
        max_holding_bars: int = 20,
        # Ensemble
        ensemble_method: Literal[
            "majority_vote", "weighted", "regime_switch"
        ] = "majority_vote",
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Run all signal generators in order, return DataFrame with
        signal and strength columns appended.

        Args:
            df:                     Feature-enriched DataFrame
            ticker:                 Label for log messages
            rsi_oversold:           RSI entry threshold for long (default 30)
            rsi_overbought:         RSI entry threshold for short (default 70)
            rsi_exit:               RSI neutral line for exit (default 50)
            rsi_smoothing:          Bars RSI must stay beyond threshold (1=off)
            macd_require_zero_cross: Only trade when MACD line crosses zero
            zscore_entry:           |z| threshold for entry (default 2.0)
            zscore_exit:            z threshold for exit — reversion to mean
            zscore_window:          Which z_price_{N}d column to use (20 or 60)
            bb_squeeze_percentile:  bb_width percentile defining a "squeeze"
            bb_max_holding_bars:    Force exit after N bars (breakout trades)
            vol_window:             Vol column to use for regime detection
            vol_lookback:           Lookback for percentile calculation
            vol_scale_floor:        Minimum position multiplier
            vol_scale_ceiling:      Maximum position multiplier
            min_holding_bars:       Ignore exits within N bars of entry
            max_holding_bars:       Force exit after N bars regardless
            ensemble_method:        Combination method for ensemble signal
            ensemble_weights:       Dict of signal_name → weight (weighted only)

        Returns:
            DataFrame with all signal and strength columns appended.
        """
        tag = f"[{ticker}] " if ticker else ""
        logger.info(f"{tag}Starting signal generation.")
        n_cols_before = df.shape[1]

        df = df.copy()

        # ---- Individual signals ----
        rsi_sig, rsi_str = self.generate_rsi_signal(
            df,
            oversold=rsi_oversold,
            overbought=rsi_overbought,
            exit_threshold=rsi_exit,
            smoothing=rsi_smoothing,
            min_holding=min_holding_bars,
            max_holding=max_holding_bars,
        )
        df["signal_rsi"] = rsi_sig
        df["signal_rsi_strength"] = rsi_str

        macd_sig, macd_str = self.generate_macd_signal(
            df,
            require_zero_cross=macd_require_zero_cross,
            min_holding=min_holding_bars,
            max_holding=max_holding_bars,
        )
        df["signal_macd"] = macd_sig
        df["signal_macd_strength"] = macd_str

        z_sig, z_str = self.generate_zscore_signal(
            df,
            entry_threshold=zscore_entry,
            exit_threshold=zscore_exit,
            window=zscore_window,
            min_holding=min_holding_bars,
            max_holding=max_holding_bars,
        )
        df["signal_zscore"] = z_sig
        df["signal_zscore_strength"] = z_str

        bb_sig, bb_str = self.generate_bb_signal(
            df,
            squeeze_percentile=bb_squeeze_percentile,
            max_holding=bb_max_holding_bars,
            min_holding=min_holding_bars,
        )
        df["signal_bb"] = bb_sig
        df["signal_bb_strength"] = bb_str

        # ---- Vol scale overlay ----
        df["position_scale"] = self.generate_vol_scale(
            df,
            vol_window=vol_window,
            lookback=vol_lookback,
            floor=vol_scale_floor,
            ceiling=vol_scale_ceiling,
        )

        # ---- Ensemble ----
        df["signal_ensemble"] = self.generate_ensemble(
            df,
            method=ensemble_method,
            weights=ensemble_weights,
            vol_window=vol_window,
        )

        n_added = df.shape[1] - n_cols_before
        logger.info(
            f"{tag}Signal generation complete — "
            f"{n_added} columns added, {df.shape[1]} total."
        )
        return df

    # ------------------------------------------------------------------ #
    # Signal 1: RSI Mean-Reversion                                         #
    # ------------------------------------------------------------------ #

    def generate_rsi_signal(
        self,
        df: pd.DataFrame,
        rsi_col: str = "rsi_14",
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_threshold: float = 50.0,
        smoothing: int = 1,
        min_holding: int = 2,
        max_holding: int = 20,
    ) -> tuple[pd.Series, pd.Series]:
        """
        RSI mean-reversion signal.

        Entry logic:
            RSI crosses BELOW oversold (30)  → long  (+1): expect bounce
            RSI crosses ABOVE overbought (70) → short (-1): expect pullback

        Exit logic:
            Long position + RSI crosses ABOVE exit (50) → flat (0)
            Short position + RSI crosses BELOW exit (50) → flat (0)

        Exit at neutral line (50), not the opposite extreme, because:
            - You are trading mean-reversion, not momentum
            - Captures the bounce-to-neutral move (high probability)
            - Holding until opposite extreme is a lower-probability bet

        Strength:
            Long:  (oversold - RSI) / oversold  → higher = stronger signal
            Short: (RSI - overbought) / (100 - overbought)

        Args:
            smoothing: require RSI to stay beyond threshold for N bars
                       before confirming entry (reduces whipsaws)
            min_holding: ignore exit signals within N bars of entry
            max_holding: force exit after N bars regardless

        Returns:
            (signal Series ∈ {-1,0,+1}, strength Series ∈ [0,1])
        """
        self._require_columns(df, [rsi_col], "generate_rsi_signal")

        rsi = df[rsi_col]
        n = len(rsi)
        signal = pd.Series(0, index=df.index, dtype=float)
        strength = pd.Series(0.0, index=df.index)

        # Apply smoothing: rolling minimum/maximum for confirmation
        if smoothing > 1:
            rsi_min = rsi.rolling(smoothing).min()
            rsi_max = rsi.rolling(smoothing).max()
        else:
            rsi_min = rsi
            rsi_max = rsi

        position = 0       # current position: -1, 0, +1
        bars_held = 0      # bars since last entry

        for i in range(1, n):
            if pd.isna(rsi.iloc[i]):
                continue

            prev_pos = position

            # Increment first so bars_held reflects current bar's count
            if position != 0:
                bars_held += 1

            # ---- Force exit at max_holding ----
            # bars_held already incremented. Exit when bars_held EXCEEDS max_holding
            # so the position is held for exactly max_holding bars (not max_holding+1).
            force_exited = False
            if position != 0 and bars_held > max_holding:
                position = 0
                bars_held = 0
                force_exited = True  # prevent re-entry on the same bar

            # ---- Exit rules (only after min_holding) ----
            elif bars_held >= min_holding or bars_held == 0:
                if position == 1 and rsi.iloc[i] > exit_threshold:
                    # Long exits when RSI recovers to neutral
                    position = 0
                    bars_held = 0
                elif position == -1 and rsi.iloc[i] < exit_threshold:
                    # Short exits when RSI recovers to neutral
                    position = 0
                    bars_held = 0

            # ---- Entry rules (only when flat AND not force-exited this bar) ----
            if position == 0 and not force_exited:
                if rsi_min.iloc[i] < oversold:
                    # RSI crossed below oversold → long
                    position = 1
                    bars_held = 1
                elif rsi_max.iloc[i] > overbought:
                    # RSI crossed above overbought → short
                    position = -1
                    bars_held = 1

            signal.iloc[i] = position

            # Compute strength
            if position == 1:
                # How far below oversold? Deeper = stronger
                strength.iloc[i] = float(
                    np.clip((oversold - rsi.iloc[i]) / oversold, 0, 1)
                )
            elif position == -1:
                strength.iloc[i] = float(
                    np.clip(
                        (rsi.iloc[i] - overbought) / (100 - overbought), 0, 1
                    )
                )

        return signal.astype(int), strength

    # ------------------------------------------------------------------ #
    # Signal 2: MACD Trend-Following                                       #
    # ------------------------------------------------------------------ #

    def generate_macd_signal(
        self,
        df: pd.DataFrame,
        histogram_col: str = "macd_histogram",
        macd_line_col: str = "macd_line",
        require_zero_cross: bool = False,
        min_holding: int = 2,
        max_holding: int = 20,
    ) -> tuple[pd.Series, pd.Series]:
        """
        MACD trend-following signal via histogram crossovers.

        Entry/Exit logic (always-in, no flat state by default):
            Histogram crosses positive (neg → pos) → long  (+1)
            Histogram crosses negative (pos → neg) → short (-1)

        The histogram sign change IS both the entry and the exit:
            - Going long closes any existing short
            - Going short closes any existing long

        Optional filter (require_zero_cross=True):
            Only trade crossovers where MACD line also crosses zero.
            This filters out small corrections within a trend —
            stronger signal, fewer trades, higher win rate.

        Strength:
            |histogram| / rolling_max(|histogram|, 63d)
            Larger histogram → stronger momentum → higher strength

        Args:
            require_zero_cross: MACD line must cross zero to confirm signal
            min_holding:        ignore reversals within N bars of entry
            max_holding:        force exit after N bars

        Returns:
            (signal Series ∈ {-1,0,+1}, strength Series ∈ [0,1])
        """
        self._require_columns(
            df, [histogram_col, macd_line_col], "generate_macd_signal"
        )

        hist = df[histogram_col]
        macd = df[macd_line_col]
        n = len(hist)

        signal = pd.Series(0, index=df.index, dtype=float)
        strength = pd.Series(0.0, index=df.index)

        # Rolling max of |histogram| for strength normalisation
        hist_abs_max = hist.abs().rolling(63, min_periods=10).max().replace(0, np.nan)

        position = 0
        bars_held = 0

        for i in range(1, n):
            if pd.isna(hist.iloc[i]) or pd.isna(hist.iloc[i - 1]):
                continue

            if position != 0:
                bars_held += 1

            # ---- Force exit ----
            if position != 0 and bars_held > max_holding:
                position = 0
                bars_held = 0
                signal.iloc[i] = position
                continue

            # ---- Crossover detection ----
            bullish_cross = hist.iloc[i - 1] <= 0 and hist.iloc[i] > 0
            bearish_cross = hist.iloc[i - 1] >= 0 and hist.iloc[i] < 0

            # Optional: require MACD line to also cross zero
            if require_zero_cross:
                bullish_cross = bullish_cross and (
                    macd.iloc[i - 1] <= 0 and macd.iloc[i] > 0
                )
                bearish_cross = bearish_cross and (
                    macd.iloc[i - 1] >= 0 and macd.iloc[i] < 0
                )

            # ---- Apply crossover (respect min_holding) ----
            if bullish_cross and (bars_held == 0 or bars_held >= min_holding):
                position = 1
                bars_held = 1
            elif bearish_cross and (bars_held == 0 or bars_held >= min_holding):
                position = -1
                bars_held = 1

            signal.iloc[i] = position

            # Strength: normalised |histogram|
            if not pd.isna(hist_abs_max.iloc[i]) and hist_abs_max.iloc[i] > 0:
                strength.iloc[i] = float(
                    np.clip(abs(hist.iloc[i]) / hist_abs_max.iloc[i], 0, 1)
                )

        return signal.astype(int), strength

    # ------------------------------------------------------------------ #
    # Signal 3: Price Z-Score Mean-Reversion                               #
    # ------------------------------------------------------------------ #

    def generate_zscore_signal(
        self,
        df: pd.DataFrame,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        window: int = 60,
        min_holding: int = 2,
        max_holding: int = 20,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Price z-score mean-reversion signal.

        Entry logic:
            z < -entry_threshold  → long  (+1): price far below mean, expect up
            z > +entry_threshold  → short (-1): price far above mean, expect down

        Exit logic:
            Long  + z crosses above exit_threshold (0) → flat: reverted to mean
            Short + z crosses below -exit_threshold (0) → flat: reverted to mean

        Critical caveat:
            This assumes mean-reversion. Equities trend over long horizons.
            A stock in a bull market will have persistently positive z-score
            and this signal will repeatedly try to short it. Mitigated by:
            - Using a shorter window (20d instead of 60d) for faster mean
            - Adding a vol filter: only trade when vol_21d is below median
              (low-vol regimes are more range-bound)

        Strength:
            (|z| - entry_threshold) / entry_threshold
            Deeper z-score → stronger expected reversion → higher strength

        Args:
            window: which z_price_{N}d column to use (20 or 60)

        Returns:
            (signal Series ∈ {-1,0,+1}, strength Series ∈ [0,1])
        """
        z_col = f"z_price_{window}d"
        self._require_columns(df, [z_col], "generate_zscore_signal")

        z = df[z_col]
        n = len(z)
        signal = pd.Series(0, index=df.index, dtype=float)
        strength = pd.Series(0.0, index=df.index)

        position = 0
        bars_held = 0

        for i in range(1, n):
            if pd.isna(z.iloc[i]):
                continue

            if position != 0:
                bars_held += 1

            # ---- Force exit ----
            force_exited = False
            if position != 0 and bars_held > max_holding:
                position = 0
                bars_held = 0
                force_exited = True

            # ---- Exit rules ----
            elif bars_held >= min_holding or bars_held == 0:
                if position == 1 and z.iloc[i] >= exit_threshold:
                    position = 0
                    bars_held = 0
                elif position == -1 and z.iloc[i] <= -exit_threshold:
                    position = 0
                    bars_held = 0

            # ---- Entry rules (only when flat AND not force-exited this bar) ----
            if position == 0 and not force_exited:
                if z.iloc[i] < -entry_threshold:
                    position = 1
                    bars_held = 1
                elif z.iloc[i] > entry_threshold:
                    position = -1
                    bars_held = 1

            signal.iloc[i] = position

            # Strength: excess beyond threshold, normalised
            if position != 0:
                excess = max(abs(z.iloc[i]) - entry_threshold, 0)
                strength.iloc[i] = float(np.clip(excess / entry_threshold, 0, 1))

        return signal.astype(int), strength

    # ------------------------------------------------------------------ #
    # Signal 4: Bollinger Band Breakout/Reversion                          #
    # ------------------------------------------------------------------ #

    def generate_bb_signal(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        mode: Literal["breakout", "reversion"] = "breakout",
        squeeze_percentile: float = 20.0,
        max_holding: int = 10,
        min_holding: int = 2,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Bollinger Band signal — supports both breakout and reversion modes.

        BREAKOUT mode (momentum):
            Price crosses ABOVE upper band with squeeze filter → long
            Price crosses BELOW lower band with squeeze filter → short
            Exit: price crosses back through middle band
            Best in: trending markets after a volatility squeeze

        REVERSION mode (mean-reversion, Bollinger's original intent):
            Price crosses ABOVE upper band → short (overbought)
            Price crosses BELOW lower band → long  (oversold)
            Exit: price returns to middle band
            Best in: range-bound markets

        Squeeze filter (breakout mode only):
            Only trade breakouts when bb_width is below the
            `squeeze_percentile`-th percentile of its 252-day history.
            A squeeze (narrow bands) preceding a breakout is a stronger
            signal than a breakout from already-wide bands.

        Strength:
            Distance of price from breached band, normalised by band width.
            Deeper breakout/reversion → stronger signal.

        Args:
            mode:               'breakout' or 'reversion'
            squeeze_percentile: width percentile threshold for squeeze filter
            max_holding:        breakout signals have short shelf-life; force exit

        Returns:
            (signal Series ∈ {-1,0,+1}, strength Series ∈ [0,1])
        """
        required = [price_col, "bb_upper", "bb_lower", "bb_middle", "bb_width"]
        if mode not in ("breakout", "reversion"):
            raise ValueError(
                f"Invalid mode '{mode}'. Choose: 'breakout' or 'reversion'."
            )
        self._require_columns(df, required, "generate_bb_signal")

        price  = df[price_col]
        upper  = df["bb_upper"]
        lower  = df["bb_lower"]
        middle = df["bb_middle"]
        width  = df["bb_width"]
        n = len(price)

        signal   = pd.Series(0, index=df.index, dtype=float)
        strength = pd.Series(0.0, index=df.index)

        # Rolling squeeze threshold (252-day percentile of bb_width)
        squeeze_threshold = width.rolling(252, min_periods=60).quantile(
            squeeze_percentile / 100
        )

        position  = 0
        bars_held = 0

        for i in range(1, n):
            if any(pd.isna(x.iloc[i]) for x in [price, upper, lower, middle]):
                continue

            if position != 0:
                bars_held += 1

            # ---- Force exit (breakout trades are time-sensitive) ----
            if position != 0 and bars_held >= max_holding:
                position = 0
                bars_held = 0

            # ---- Exit: price returns to middle band ----
            elif bars_held >= min_holding or bars_held == 0:
                if position == 1 and price.iloc[i] >= middle.iloc[i]:
                    position = 0
                    bars_held = 0
                elif position == -1 and price.iloc[i] <= middle.iloc[i]:
                    position = 0
                    bars_held = 0

            # ---- Entry ----
            if position == 0:
                prev_p = price.iloc[i - 1]
                curr_p = price.iloc[i]
                curr_u = upper.iloc[i]
                curr_l = lower.iloc[i]

                # Squeeze condition: width currently below historical percentile
                in_squeeze = (
                    pd.isna(squeeze_threshold.iloc[i])
                    or width.iloc[i] <= squeeze_threshold.iloc[i]
                )

                if mode == "breakout":
                    # Cross above upper band after a squeeze
                    if prev_p <= curr_u and curr_p > curr_u and in_squeeze:
                        position = 1
                        bars_held = 1
                    # Cross below lower band after a squeeze
                    elif prev_p >= curr_l and curr_p < curr_l and in_squeeze:
                        position = -1
                        bars_held = 1

                else:  # reversion
                    # Touch/breach upper band → short (overbought)
                    if prev_p <= curr_u and curr_p >= curr_u:
                        position = -1
                        bars_held = 1
                    # Touch/breach lower band → long (oversold)
                    elif prev_p >= curr_l and curr_p <= curr_l:
                        position = 1
                        bars_held = 1

            signal.iloc[i] = position

            # Strength: price distance from breached band / band width
            bw = upper.iloc[i] - lower.iloc[i]
            if position == 1 and bw > 0:
                dist = max(middle.iloc[i] - price.iloc[i], 0)
                strength.iloc[i] = float(np.clip(dist / bw, 0, 1))
            elif position == -1 and bw > 0:
                dist = max(price.iloc[i] - middle.iloc[i], 0)
                strength.iloc[i] = float(np.clip(dist / bw, 0, 1))

        return signal.astype(int), strength

    # ------------------------------------------------------------------ #
    # Signal 5: Volatility-Adjusted Position Scale                         #
    # ------------------------------------------------------------------ #

    def generate_vol_scale(
        self,
        df: pd.DataFrame,
        vol_window: int = 21,
        lookback: int = 252,
        floor: float = 0.0,
        ceiling: float = 2.0,
    ) -> pd.Series:
        """
        Volatility-adjusted position size multiplier.

        This is NOT a directional signal — it is a risk management overlay
        that modulates position size based on the current volatility regime.

        Logic:
            vol > 90th percentile of past `lookback` days → scale = floor
            vol < 10th percentile of past `lookback` days → scale = ceiling
            Otherwise → linearly interpolate between ceiling and floor

        The linear interpolation gives a smooth, continuous multiplier
        rather than step-function jumps at percentile boundaries.

        Effect on trading:
            High-vol regime (e.g. March 2020): reduce or eliminate positions
            Low-vol regime  (e.g. 2017):       increase positions
            Normal regime:                      full size

        Formula:
            vol_pct = percentile rank of current vol in past N days
            scale   = ceiling - (ceiling - floor) * vol_pct
            clipped to [floor, ceiling]

        Returns:
            Series of floats ∈ [floor, ceiling]
        """
        vol_col = f"vol_{vol_window}d"
        self._require_columns(df, [vol_col], "generate_vol_scale")

        vol = df[vol_col]
        scale = pd.Series(1.0, index=df.index)

        # Rolling percentile rank: what fraction of past N days had lower vol?
        def percentile_rank(series: pd.Series, window: int) -> pd.Series:
            ranks = pd.Series(np.nan, index=series.index)
            for i in range(window, len(series)):
                window_vals = series.iloc[i - window: i].dropna()
                if len(window_vals) < 10:
                    continue
                current = series.iloc[i]
                if pd.isna(current):
                    continue
                ranks.iloc[i] = float((window_vals < current).mean())
            return ranks

        vol_pct_rank = percentile_rank(vol, lookback)

        # Linear interpolation: high vol → scale toward floor
        raw_scale = ceiling - (ceiling - floor) * vol_pct_rank
        scale = raw_scale.clip(lower=floor, upper=ceiling).fillna(1.0)

        logger.debug(
            f"vol_scale: mean={scale.mean():.3f}, "
            f"min={scale.min():.3f}, max={scale.max():.3f}"
        )
        return scale

    # ------------------------------------------------------------------ #
    # Signal 6: Ensemble                                                   #
    # ------------------------------------------------------------------ #

    def generate_ensemble(
        self,
        df: pd.DataFrame,
        method: Literal[
            "majority_vote", "weighted", "regime_switch"
        ] = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        vol_window: int = 21,
    ) -> pd.Series:
        """
        Combine individual signals into a single ensemble signal.

        Three combination methods:

        majority_vote:
            ensemble = sign(signal_rsi + signal_macd + signal_zscore)
            Requires ≥2 signals to agree. No agreement → flat (0).
            Simple, interpretable, no parameter fitting risk.

        weighted:
            ensemble = sign(w1*signal_rsi + w2*signal_macd + w3*signal_zscore)
            Weights should come from individual signal Sharpe ratios
            measured in backtesting — not set arbitrarily here.
            Default weights: equal (1/3 each).

        regime_switch:
            IF current vol > trailing vol (risk-on/trending regime):
                use signal_macd (trend-following works in trends)
            ELSE (low vol / range-bound regime):
                use signal_rsi (mean-reversion works in range markets)
            Most sophisticated — regime detection must be reliable.

        Note on combining signals:
            Check signal correlation before weighting.
            If signal_rsi and signal_zscore are 0.85 correlated,
            combining them is double-weighting the same bet, not
            diversification. Aim for inter-signal correlation < 0.4.

        Returns:
            Series ∈ {-1, 0, +1}
        """
        required_signals = ["signal_rsi", "signal_macd", "signal_zscore"]
        self._require_columns(df, required_signals, "generate_ensemble")

        s_rsi   = df["signal_rsi"]
        s_macd  = df["signal_macd"]
        s_z     = df["signal_zscore"]

        if method == "majority_vote":
            raw = s_rsi + s_macd + s_z
            ensemble = np.sign(raw).astype(int)

        elif method == "weighted":
            if weights is None:
                weights = {
                    "signal_rsi":    1 / 3,
                    "signal_macd":   1 / 3,
                    "signal_zscore": 1 / 3,
                }
            total_w = sum(weights.values())
            raw = (
                weights.get("signal_rsi",    0) * s_rsi  +
                weights.get("signal_macd",   0) * s_macd +
                weights.get("signal_zscore", 0) * s_z
            ) / total_w
            ensemble = np.sign(raw).astype(int)

        elif method == "regime_switch":
            vol_col = f"vol_{vol_window}d"
            self._require_columns(df, [vol_col], "generate_ensemble (regime_switch)")
            vol      = df[vol_col]
            vol_trail = vol.rolling(63, min_periods=20).mean()  # 3-month trailing avg

            # High vol (current > trailing) → trending regime → use MACD
            # Low vol  (current ≤ trailing) → range-bound   → use RSI
            is_trending = (vol > vol_trail).fillna(False)
            ensemble = pd.Series(
                np.where(is_trending, s_macd, s_rsi),
                index=df.index,
            ).astype(int)

        else:
            raise ValueError(
                f"Unknown ensemble method '{method}'. "
                "Choose: majority_vote | weighted | regime_switch"
            )

        return ensemble

    # ------------------------------------------------------------------ #
    # Quality Report                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def signal_report(df: pd.DataFrame, ticker: str = "") -> dict:
        """
        Return signal quality diagnostics as a plain dict.

        Metrics:
            turnover:    fraction of bars where signal changes (high = whipsaw)
            long_pct:    fraction of time long
            short_pct:   fraction of time short
            flat_pct:    fraction of time flat
            correlation: inter-signal correlation matrix
            strength:    mean strength per signal (how decisive are entries?)
        """
        signal_cols = [c for c in df.columns if c.startswith("signal_")]
        strength_cols = [c for c in df.columns if c.endswith("_strength")]

        report: dict = {"ticker": ticker or "unknown"}

        for col in signal_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            report[col] = {
                "turnover":   round(float((s.diff() != 0).mean()), 4),
                "long_pct":   round(float((s == 1).mean()), 4),
                "short_pct":  round(float((s == -1).mean()), 4),
                "flat_pct":   round(float((s == 0).mean()), 4),
                "unique_vals": sorted(s.unique().tolist()),
            }

        # Strength averages
        for col in strength_cols:
            if col not in df.columns:
                continue
            base = col.replace("_strength", "")
            if base in report:
                report[base]["mean_strength"] = round(
                    float(df[col].dropna().mean()), 4
                )

        # Inter-signal correlation
        if len(signal_cols) >= 2:
            corr = df[signal_cols].dropna().corr().round(3)
            report["signal_correlation"] = corr.to_dict()

        return report

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_columns(
        df: pd.DataFrame, cols: List[str], method: str
    ) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"{method}() requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )