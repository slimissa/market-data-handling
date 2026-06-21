"""
regime_filter.py — Phase 6: Regime-Gated Signal Filtering
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → signals → [regime filter] → backtest → factor model

The core insight from Phase 5:
    RSI and z-score have Sharpe +3 to +5 in range-bound regimes,
    but -0.75 in trending markets. They don't fail — they work perfectly
    in the right regime and fail in the wrong one.

    The fix is not a better signal. It is a gate:
        IF regime is unfavourable for this signal → return 0 (flat)
        IF regime is favourable                  → pass signal through unchanged

This module provides:

    1. RegimeFilter
       Wraps any signal column. Returns the original signal in favourable
       regimes, 0 (flat) otherwise. Can stack multiple conditions.

    2. Regime conditions (composable):
       - VolPercentileCondition     vol_21d < Nth percentile of trailing history
       - TrendCondition             rolling return > / < threshold
       - MACDCondition              MACD line sign (trending direction)
       - BBWidthCondition           bb_width < Nth percentile (squeeze = range-bound)
       - RSIRangeCondition          RSI within a band (avoid extremes)
       - CompositeCondition         AND / OR of multiple conditions

    3. RegimeFilteredEnsemble
       Builds regime-appropriate sub-portfolios:
           mean_reversion_signals → active only in range-bound regime
           trend_signals          → active only in trending regime
       Then combines them via weighted vote.

Mathematical justification:
    A signal has conditional expectation:
        E[r | regime=R] ≠ E[r]
    If Cov(signal, regime_indicator) ≠ 0, regime-conditioning improves
    the signal's information ratio.

    For RSI (mean-reversion):
        E[r | low_vol, range_bound] ≫ E[r | high_vol, trending]
    Regime filtering removes the bad-regime observations from the
    strategy's return stream — reducing drawdown without reducing alpha.

Usage:
    from regime_filter import RegimeFilter, VolPercentileCondition, \
                              TrendCondition, RegimeFilteredEnsemble

    # Single filter: RSI only when vol is low
    vol_gate = VolPercentileCondition(col="vol_21d", lookback=252, percentile=30)
    rf = RegimeFilter(signal_col="signal_rsi", conditions=[vol_gate])
    df["signal_rsi_filtered"] = rf.apply(df)

    # Composite: low vol AND range-bound (no strong trend)
    trend_gate = TrendCondition(return_col="returns", window=63, max_trend=0.10)
    rf2 = RegimeFilter(
        signal_col="signal_rsi",
        conditions=[vol_gate, trend_gate],
        logic="AND",
    )
    df["signal_rsi_filtered_v2"] = rf2.apply(df)

    # Full regime-adaptive ensemble
    rfe = RegimeFilteredEnsemble()
    df = rfe.apply(df)
    # Adds: signal_mr_gated, signal_trend_gated, signal_regime_adaptive
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract condition interface
# ======================================================================

class RegimeCondition(ABC):
    """
    A binary gate: returns True (favourable) or False (unfavourable) per bar.

    All conditions are:
        - Point-in-time: use only data available at bar t (no lookahead)
        - Rolling: computed over a backward-looking window
        - NaN-safe: NaN → False (no signal during warmup)
    """

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the condition for every bar.

        Returns:
            Boolean Series: True = favourable regime, False = unfavourable
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


# ======================================================================
# Concrete conditions
# ======================================================================

class VolPercentileCondition(RegimeCondition):
    """
    Gate: current volatility is below the Nth percentile of recent history.

    Rationale:
        Mean-reversion signals (RSI, z-score) work in low-vol, range-bound
        regimes. High vol indicates a trending or crisis market where
        mean-reversion bets get run over.

        vol_21d < 30th percentile of trailing 252 days
        ≡ current vol is in the lowest 30% of recent history
        ≡ the market is relatively calm → mean-reversion likely

    Args:
        col:        Volatility column to gate on (e.g. "vol_21d")
        lookback:   Trailing window for percentile calculation (bars)
        percentile: Threshold (0-100). Signal passes when col < this percentile.
        mode:       "below" = active when vol is LOW (mean-reversion)
                    "above" = active when vol is HIGH (vol-selling strategies)
    """

    def __init__(
        self,
        col:        str   = "vol_21d",
        lookback:   int   = 252,
        percentile: float = 30.0,
        mode:       Literal["below", "above"] = "below",
    ):
        self.col        = col
        self.lookback   = lookback
        self.percentile = percentile
        self.mode       = mode

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            logger.warning(f"VolPercentileCondition: '{self.col}' not in DataFrame.")
            return pd.Series(False, index=df.index)

        vol = df[self.col]
        # Rolling percentile threshold — point-in-time, no lookahead
        threshold = vol.rolling(self.lookback, min_periods=self.lookback // 2).quantile(
            self.percentile / 100
        )

        if self.mode == "below":
            active = vol < threshold
        else:
            active = vol > threshold

        return active.fillna(False)

    def __repr__(self) -> str:
        return (
            f"VolPercentile({self.col} {self.mode} "
            f"P{self.percentile:.0f}, lookback={self.lookback})"
        )


class TrendCondition(RegimeCondition):
    """
    Gate: rolling return is below a threshold (market is NOT strongly trending).

    Rationale:
        Mean-reversion signals need a range-bound market.
        A strong uptrend makes shorts from RSI>70 systematically lose.
        A strong downtrend makes longs from RSI<30 keep falling.

        |rolling_return_63d| < max_trend (e.g. 10% annualised)
        ≡ the market has been roughly flat over the past 3 months
        ≡ no strong directional momentum → mean-reversion can work

    Args:
        return_col:  Column with daily returns (e.g. "returns")
        window:      Rolling window in bars (63 = ~3 months)
        max_trend:   Maximum allowable annualised rolling return (absolute).
                     0.10 = 10% annual = 0.04% daily over 63 days
        directional: If set to "up" or "down", only gates in that direction.
                     "up"   = only active when market is NOT in uptrend
                     "down" = only active when market is NOT in downtrend
                     None   = active when abs(trend) < max_trend (both directions)
    """

    def __init__(
        self,
        return_col:  str   = "returns",
        window:      int   = 63,
        max_trend:   float = 0.10,
        directional: Optional[Literal["up", "down"]] = None,
    ):
        self.return_col  = return_col
        self.window      = window
        self.max_trend   = max_trend
        self.directional = directional
        # Convert annual threshold to daily rolling mean equivalent
        self._daily_threshold = max_trend / 252

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.return_col not in df.columns:
            logger.warning(f"TrendCondition: '{self.return_col}' not in DataFrame.")
            return pd.Series(False, index=df.index)

        ret = df[self.return_col]
        rolling_mean = ret.rolling(self.window, min_periods=self.window // 2).mean()

        if self.directional == "up":
            # Active when NOT in a strong uptrend
            active = rolling_mean < self._daily_threshold
        elif self.directional == "down":
            # Active when NOT in a strong downtrend
            active = rolling_mean > -self._daily_threshold
        else:
            # Active when trend is weak in either direction
            active = rolling_mean.abs() < self._daily_threshold

        return active.fillna(False)

    def __repr__(self) -> str:
        return (
            f"Trend(|{self.return_col}_{self.window}d| < "
            f"{self.max_trend:.0%}/yr, dir={self.directional})"
        )


class MACDCondition(RegimeCondition):
    """
    Gate: use MACD line sign to determine trend direction.

    Rationale:
        macd_line > 0 → fast EMA > slow EMA → uptrend
        macd_line < 0 → fast EMA < slow EMA → downtrend

        For mean-reversion signals: active when MACD is near zero
        (weak trend). For trend-following signals: active when MACD
        confirms the direction.

    Args:
        macd_col:   Column with MACD line values
        direction:  "positive" = active when macd_line > 0 (uptrend)
                    "negative" = active when macd_line < 0 (downtrend)
                    "weak"     = active when |macd_line| < threshold (range-bound)
        threshold:  Used only in "weak" mode. Defines "near zero."
                    If None, uses the 25th percentile of |macd_line|.
    """

    def __init__(
        self,
        macd_col:  str = "macd_line",
        direction: Literal["positive", "negative", "weak"] = "weak",
        threshold: Optional[float] = None,
    ):
        self.macd_col  = macd_col
        self.direction = direction
        self.threshold = threshold

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.macd_col not in df.columns:
            logger.warning(f"MACDCondition: '{self.macd_col}' not in DataFrame.")
            return pd.Series(False, index=df.index)

        macd = df[self.macd_col]

        if self.direction == "positive":
            active = macd > 0
        elif self.direction == "negative":
            active = macd < 0
        else:  # "weak"
            if self.threshold is not None:
                thresh = self.threshold
            else:
                # Rolling 25th percentile of |macd| as "near zero" threshold
                thresh_series = macd.abs().rolling(252, min_periods=60).quantile(0.25)
                active = macd.abs() < thresh_series.fillna(macd.abs().median())
                return active.fillna(False)
            active = macd.abs() < thresh

        return active.fillna(False)

    def __repr__(self) -> str:
        return f"MACD({self.macd_col} is {self.direction})"


class BBWidthCondition(RegimeCondition):
    """
    Gate: Bollinger Band width is below the Nth percentile (squeeze).

    Rationale:
        Low bb_width = low recent volatility = range-bound market.
        A Bollinger squeeze (narrow bands) precedes a breakout, but
        while the squeeze holds, mean-reversion signals are effective.

        bb_width < 20th percentile of trailing 252d
        ≡ bands are unusually narrow ≡ low-vol range-bound regime

    Args:
        bb_col:     Column with Bollinger Band width (default: "bb_width")
        lookback:   Trailing window for percentile
        percentile: Threshold — active when bb_width is below this percentile
    """

    def __init__(
        self,
        bb_col:     str   = "bb_width",
        lookback:   int   = 252,
        percentile: float = 40.0,
    ):
        self.bb_col     = bb_col
        self.lookback   = lookback
        self.percentile = percentile

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.bb_col not in df.columns:
            logger.warning(f"BBWidthCondition: '{self.bb_col}' not in DataFrame.")
            return pd.Series(False, index=df.index)

        width = df[self.bb_col]
        threshold = width.rolling(
            self.lookback, min_periods=self.lookback // 4
        ).quantile(self.percentile / 100)

        return (width < threshold).fillna(False)

    def __repr__(self) -> str:
        return f"BBWidth(< P{self.percentile:.0f}, lookback={self.lookback})"


class RSIRangeCondition(RegimeCondition):
    """
    Gate: RSI is within a 'neutral' band (avoid extreme momentum).

    Rationale:
        Some signals perform poorly when RSI is very high (strong uptrend)
        or very low (strong downtrend). This condition restricts activity
        to periods when momentum is not extreme.

    Args:
        rsi_col:  Column with RSI values
        low:      Lower bound — inactive below this (oversold momentum)
        high:     Upper bound — inactive above this (overbought momentum)

    Inverted use: pass low=70, high=30 to get a condition that is
    active only during EXTREME RSI (useful for mean-reversion entry gates).
    """

    def __init__(
        self,
        rsi_col: str   = "rsi_14",
        low:     float = 35.0,
        high:    float = 65.0,
    ):
        self.rsi_col = rsi_col
        self.low     = low
        self.high    = high

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.rsi_col not in df.columns:
            logger.warning(f"RSIRangeCondition: '{self.rsi_col}' not in DataFrame.")
            return pd.Series(False, index=df.index)

        rsi = df[self.rsi_col]
        return ((rsi >= self.low) & (rsi <= self.high)).fillna(False)

    def __repr__(self) -> str:
        return f"RSIRange({self.low} ≤ {self.rsi_col} ≤ {self.high})"


class CompositeCondition(RegimeCondition):
    """
    Combine multiple conditions with AND or OR logic.

    AND: all conditions must be True (strict gate — fewer trades, higher quality)
    OR:  any condition must be True (permissive gate — more trades)

    Example:
        gate = CompositeCondition([
            VolPercentileCondition(percentile=30),
            TrendCondition(max_trend=0.10),
        ], logic="AND")
        # Active only when BOTH vol is low AND trend is weak
    """

    def __init__(
        self,
        conditions: List[RegimeCondition],
        logic: Literal["AND", "OR"] = "AND",
    ):
        if not conditions:
            raise ValueError("CompositeCondition requires at least one condition.")
        self.conditions = conditions
        self.logic      = logic

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        results = [c.evaluate(df) for c in self.conditions]
        if self.logic == "AND":
            combined = results[0]
            for r in results[1:]:
                combined = combined & r
        else:  # OR
            combined = results[0]
            for r in results[1:]:
                combined = combined | r
        return combined.fillna(False)

    def __repr__(self) -> str:
        inner = f" {self.logic} ".join(repr(c) for c in self.conditions)
        return f"Composite({inner})"


# ======================================================================
# RegimeFilter — the main wrapper
# ======================================================================

class TrendConfirmationCondition(RegimeCondition):
    """
    Gate: the signal's own current direction agrees with MACD-line sign.

    Rationale (closes a structural gap vs the mean-reversion gates):
        rsi_filter() and zscore_filter() are both two-condition AND gates
        (vol percentile AND trend/BB-width). macd_filter() and
        bb_breakout_filter() were single-condition gates — vol-above-
        percentile only, with no check that the trend the signal is
        betting on is still actually confirmed. A trend-following signal
        could stay long deep into a reversal as long as vol stayed
        elevated, because elevated vol alone doesn't mean the ORIGINAL
        trend direction is still intact — it could be elevated because
        the trend just reversed violently.

        signal == +1  AND macd_line > 0   → confirmed uptrend, stay long
        signal == -1  AND macd_line < 0   → confirmed downtrend, stay short
        signal != 0   AND macd disagrees  → trend has turned, suppress

    This is direction-aware (unlike MACDCondition's fixed direction
    parameter): it checks agreement with whatever the wrapped signal is
    currently saying, not a hardcoded "always long" or "always short"
    bias — appropriate for signals that trade both directions.

    Args:
        signal_col: The signal column being gated (its own sign is read
                   here, separately from the value RegimeFilter eventually
                   passes through — this lets the gate disagree with a
                   signal that has gone stale).
        macd_col:   MACD line column used as the confirmation reference.
        min_macd_magnitude: Minimum |macd_line| required to count as a
                   genuine confirmation, not noise near zero. Expressed
                   as a rolling percentile of |macd_line| history so it
                   adapts to the ticker's typical MACD scale.
        lookback:   Rolling window for the magnitude percentile.
    """

    def __init__(
        self,
        signal_col:          str   = "signal_macd",
        macd_col:             str   = "macd_line",
        min_macd_magnitude:   float = 25.0,   # percentile, 0 = no minimum
        lookback:             int   = 252,
    ):
        self.signal_col         = signal_col
        self.macd_col            = macd_col
        self.min_macd_magnitude  = min_macd_magnitude
        self.lookback            = lookback

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in (self.signal_col, self.macd_col) if c not in df.columns]
        if missing:
            logger.warning(f"TrendConfirmationCondition: missing columns {missing}.")
            return pd.Series(False, index=df.index)

        signal = df[self.signal_col].fillna(0)
        macd   = df[self.macd_col]

        agrees = (
            ((signal > 0) & (macd > 0)) |
            ((signal < 0) & (macd < 0))
        )

        if self.min_macd_magnitude > 0:
            mag_threshold = macd.abs().rolling(
                self.lookback, min_periods=self.lookback // 4
            ).quantile(self.min_macd_magnitude / 100)
            strong_enough = macd.abs() >= mag_threshold.fillna(macd.abs().median())
            agrees = agrees & strong_enough

        return agrees.fillna(False)

    def __repr__(self) -> str:
        return (
            f"TrendConfirmation({self.signal_col} agrees with "
            f"sign({self.macd_col}), min_mag=P{self.min_macd_magnitude:.0f})"
        )


class SignalDrawdownCondition(RegimeCondition):
    """
    Gate: per-signal running drawdown breaker — a resettable backstop
    independent of the global portfolio-level circuit breaker in
    VectorisedBacktester.

    Rationale:
        VectorisedBacktester.max_drawdown_exit is a ONE-SHOT, PERMANENT
        breaker: once triggered, it zeros the position for the rest of
        the entire backtest (the position is `shares_series.loc[
        first_breach_idx:] = 0`, applied once, never re-evaluated). That
        is appropriate behaviour for "the whole portfolio blew up, stop
        trading" — but wrong as a per-signal mechanism: a single bad
        regime call early in a multi-year backtest should not
        permanently disable a signal that might recover and perform
        correctly in a later, genuinely favourable regime.

        This condition instead computes a lightweight simulated P&L for
        the SIGNAL IN ISOLATION (sign(signal) * daily return, no
        position sizing, no transaction costs — a fast proxy, not a
        full backtest) and tracks its own running drawdown from its own
        running peak. When that drawdown breaches `max_dd`, the gate
        goes False (suppressing the signal) until the simulated equity
        recovers to within `recovery_threshold` of its prior peak — at
        which point the gate re-opens automatically. This is what makes
        it a genuine circuit BREAKER rather than a one-time kill switch:
        it can fire, suppress, recover, and fire again across the same
        backtest.

        This directly targets the -89.3% / -73.9% MSFT class of
        drawdown: those occurred because nothing was watching the
        signal's own equity curve before it ever reached portfolio-level
        risk management. A signal that is "regime-favourable" by the
        other gates can still be wrong in that regime; this condition
        is the backstop for exactly that case.

    Args:
        signal_col: Signal column to simulate (reads its own sign,
                   independent of what RegimeFilter ultimately passes
                   through — same design as TrendConfirmationCondition).
        return_col: Daily return column used for the P&L proxy.
        max_dd:     Maximum allowable drawdown (positive fraction, e.g.
                   0.15 = 15%) of the signal's own simulated equity
                   before the gate suppresses it.
        recovery_threshold: Fraction of the prior peak the simulated
                   equity must recover to before the gate re-opens.
                   1.0 = must reach a new equity high. 0.9 = must
                   recover to within 10% of the prior peak. Lower values
                   re-enable the signal sooner after a drawdown.
    """

    def __init__(
        self,
        signal_col:          str   = "signal_macd",
        return_col:           str   = "returns",
        max_dd:                float = 0.15,
        recovery_threshold:    float = 0.95,
    ):
        self.signal_col          = signal_col
        self.return_col           = return_col
        self.max_dd                = max_dd
        self.recovery_threshold    = recovery_threshold

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in (self.signal_col, self.return_col) if c not in df.columns]
        if missing:
            logger.warning(f"SignalDrawdownCondition: missing columns {missing}.")
            return pd.Series(True, index=df.index)  # fail open: don't block on missing data

        signal = df[self.signal_col].fillna(0)
        ret    = df[self.return_col].fillna(0)

        # Lagged signal: today's P&L reflects yesterday's position,
        # consistent with the execution-lag convention used everywhere
        # else in this pipeline (signal at t -> fill at t+1).
        sim_pnl = signal.shift(1).fillna(0) * ret
        sim_equity = (1.0 + sim_pnl).cumprod()

        n = len(sim_equity)
        gate = pd.Series(True, index=df.index)
        peak = 1.0
        breaker_active = False

        equity_vals = sim_equity.values
        for i in range(n):
            e = equity_vals[i]
            if not breaker_active:
                peak = max(peak, e)
                dd = (e - peak) / peak if peak > 0 else 0.0
                if dd < -self.max_dd:
                    breaker_active = True
                    gate.iloc[i] = False
                # else: gate stays True (default), peak tracked
            else:
                # Breaker is active: stay suppressed until simulated
                # equity recovers to recovery_threshold * peak.
                gate.iloc[i] = False
                if peak > 0 and e >= self.recovery_threshold * peak:
                    breaker_active = False
                    peak = e  # reset tracking from the recovery point

        return gate

    def __repr__(self) -> str:
        return (
            f"SignalDrawdown({self.signal_col}: max_dd={self.max_dd:.0%}, "
            f"recovery={self.recovery_threshold:.0%})"
        )


class RegimeFilter:
    """
    Wraps any signal column, zeroing it out in unfavourable regimes.

    The filtered signal is identical to the original signal when the
    regime is favourable, and 0 (flat) otherwise.

    This is equivalent to multiplying the signal by a binary regime mask:
        filtered_signal[t] = signal[t] * regime_gate[t]

    No lookahead bias: all conditions use only data available at bar t.

    Args:
        signal_col:     Column name of the signal to filter
        conditions:     List of RegimeCondition objects
        logic:          How to combine multiple conditions: "AND" | "OR"
        output_col:     Name for the new filtered signal column.
                        Defaults to f"{signal_col}_filtered".
        invert:         If True, gate is active when conditions are FALSE
                        (useful for crisis/high-vol strategies).
    """

    def __init__(
        self,
        signal_col:  str,
        conditions:  List[RegimeCondition],
        logic:       Literal["AND", "OR"] = "AND",
        output_col:  Optional[str] = None,
        invert:      bool = False,
    ):
        if not conditions:
            raise ValueError("RegimeFilter requires at least one condition.")

        self.signal_col = signal_col
        self.conditions = conditions
        self.logic      = logic
        self.output_col = output_col or f"{signal_col}_filtered"
        self.invert     = invert

        # Bundle conditions into a CompositeCondition for evaluation
        if len(conditions) == 1:
            self._gate = conditions[0]
        else:
            self._gate = CompositeCondition(conditions, logic=logic)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the regime filter to the DataFrame.

        Args:
            df: DataFrame containing signal_col and all condition columns.

        Returns:
            Filtered signal Series ∈ {-1, 0, +1}.
            Original signal is passed through when regime is favourable.
            0 (flat) when regime is unfavourable.
        """
        if self.signal_col not in df.columns:
            raise KeyError(
                f"RegimeFilter: signal column '{self.signal_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        gate = self._gate.evaluate(df)

        if self.invert:
            gate = ~gate

        original = df[self.signal_col].fillna(0).astype(int)
        filtered = original.where(gate, other=0)

        n_total   = len(gate)
        n_active  = int(gate.sum())
        n_filtered = n_total - n_active
        filter_pct = n_filtered / n_total * 100

        logger.info(
            f"RegimeFilter({self.signal_col}): "
            f"{n_active}/{n_total} bars active ({100-filter_pct:.1f}%), "
            f"{n_filtered} bars zeroed ({filter_pct:.1f}%) | "
            f"gate: {self._gate!r}"
        )

        return filtered.rename(self.output_col)

    def apply_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filter and add the result as a new column to df."""
        df = df.copy()
        df[self.output_col] = self.apply(df)
        return df

    def regime_stats(self, df: pd.DataFrame) -> dict:
        """
        Return statistics about when the regime gate is active.

        Useful for understanding how much of the time each filter
        suppresses trading, and whether the gate is too restrictive.
        """
        gate = self._gate.evaluate(df)
        if self.invert:
            gate = ~gate

        original = df[self.signal_col].fillna(0).astype(int)
        filtered = original.where(gate, other=0)

        return {
            "signal_col":     self.signal_col,
            "output_col":     self.output_col,
            "gate":           repr(self._gate),
            "n_bars_total":   len(gate),
            "n_bars_active":  int(gate.sum()),
            "active_pct":     round(float(gate.mean() * 100), 2),
            "n_trades_orig":  int((original.diff() != 0).sum()),
            "n_trades_filt":  int((filtered.diff() != 0).sum()),
            "trade_reduction_pct": round(
                float(1 - (filtered.diff() != 0).sum() / max((original.diff() != 0).sum(), 1)) * 100,
                2
            ),
        }


# ======================================================================
# Preset filters for common signal types
# ======================================================================

class RegimeFilterPresets:
    """
    Ready-made RegimeFilter configurations for the six signal families.

    Each preset reflects the regime analysis findings from Phase 5:
        - RSI, z-score: mean-reversion → needs low-vol, range-bound market
        - MACD, BB breakout: trend-following → needs trending market
        - Ensemble: needs regime-switching

    Usage:
        presets = RegimeFilterPresets()
        df = presets.apply_all(df)
        # Adds filtered columns for every signal
    """

    @staticmethod
    @staticmethod
    def rsi_filter(
        signal_col: str = "signal_rsi",
        vol_percentile: float = 20.0,
        max_trend_annual: float = 0.06,
        max_dd: float = 0.15,
    ) -> RegimeFilter:
        """
        RSI mean-reversion: active only in low-vol, weak-trend regime,
        with a resettable per-signal drawdown backstop.

        Gate: vol_21d < P20 of trailing year
              AND |63d return| < 6%/yr
              AND signal's own simulated drawdown < 15% (resettable)

        Tightened from the original P30/10%-per-year thresholds: those
        let through enough borderline-regime trades that the gated
        signal's drawdown was still meaningfully larger than necessary.
        P20/6%-per-year is stricter — fewer trades, but each one is in a
        regime more confidently range-bound.

        Rationale from Phase 5:
            RSI Sharpe = +5.0 in range_bound, -0.75 in trending_up.
            Zeroing RSI trades during uptrends removes the -0.75 drag
            while preserving the +5.0 alpha in the right regime.
        """
        return RegimeFilter(
            signal_col=signal_col,
            conditions=[
                VolPercentileCondition(col="vol_21d", lookback=252, percentile=vol_percentile),
                TrendCondition(return_col="returns", window=63, max_trend=max_trend_annual),
                SignalDrawdownCondition(signal_col=signal_col, return_col="returns", max_dd=max_dd),
            ],
            logic="AND",
            output_col=f"{signal_col}_vol_trend_gated",
        )

    @staticmethod
    def zscore_filter(
        signal_col: str = "signal_zscore",
        vol_percentile: float = 25.0,
        bb_percentile: float = 25.0,
        max_dd: float = 0.15,
    ) -> RegimeFilter:
        """
        Z-score mean-reversion: active in low-vol + Bollinger squeeze
        regime, with a resettable per-signal drawdown backstop.

        Gate: vol_21d < P25 AND bb_width < P25 (narrow bands = range-bound)
              AND signal's own simulated drawdown < 15% (resettable)

        Tightened from P35/P40 — narrower squeeze requirement means the
        signal only fires when the market is more confidently range-bound.
        """
        return RegimeFilter(
            signal_col=signal_col,
            conditions=[
                VolPercentileCondition(col="vol_21d", lookback=252, percentile=vol_percentile),
                BBWidthCondition(bb_col="bb_width", lookback=252, percentile=bb_percentile),
                SignalDrawdownCondition(signal_col=signal_col, return_col="returns", max_dd=max_dd),
            ],
            logic="AND",
            output_col=f"{signal_col}_vol_bb_gated",
        )

    @staticmethod
    def macd_filter(
        signal_col: str = "signal_macd",
        vol_percentile: float = 50.0,
        max_dd: float = 0.20,
    ) -> RegimeFilter:
        """
        MACD trend-following: active when vol is elevated AND the
        signal's own current direction is confirmed by MACD sign, with a
        resettable per-signal drawdown backstop.

        Gate: vol_21d > P50 (elevated vol = directional move)
              AND signal direction agrees with MACD-line sign (confirmed,
                  not stale — see TrendConfirmationCondition)
              AND signal's own simulated drawdown < 20% (resettable)

        PREVIOUSLY: this was a single-condition gate (vol-above-percentile
        only), structurally weaker than rsi_filter/zscore_filter's
        two-condition AND gates. Elevated vol alone doesn't confirm the
        trend the signal is betting on is still intact — vol can be
        elevated because a trend just reversed violently. Adding
        TrendConfirmationCondition closes that gap: the gate now requires
        genuine, currently-confirmed trend agreement, not just "the
        market is moving."

        max_dd is wider than the mean-reversion gates (20% vs 15%)
        because trend-following inherently tolerates deeper pullbacks
        before a trend is invalidated — too tight a stop here would cut
        genuine trends on normal volatility, not just bad regime calls.

        Rationale: MACD Sharpe = +1.18 in trending_up, -1.30 in range_bound.
        """
        return RegimeFilter(
            signal_col=signal_col,
            conditions=[
                VolPercentileCondition(
                    col="vol_21d", lookback=252,
                    percentile=100 - vol_percentile,
                    mode="above"
                ),
                TrendConfirmationCondition(
                    signal_col=signal_col, macd_col="macd_line",
                    min_macd_magnitude=25.0,
                ),
                SignalDrawdownCondition(signal_col=signal_col, return_col="returns", max_dd=max_dd),
            ],
            logic="AND",
            output_col=f"{signal_col}_trend_gated",
        )

    @staticmethod
    def bb_breakout_filter(
        signal_col: str = "signal_bb",
        vol_percentile: float = 60.0,
        max_dd: float = 0.20,
    ) -> RegimeFilter:
        """
        Bollinger breakout: active when vol is rising AND the signal's
        own direction is confirmed by MACD sign, with a resettable
        per-signal drawdown backstop.

        Gate: vol_21d > P60 (above-median vol = expanding from squeeze)
              AND signal direction agrees with MACD-line sign
              AND signal's own simulated drawdown < 20% (resettable)

        PREVIOUSLY: single-condition gate, same structural gap as
        macd_filter before this fix. Now requires trend confirmation in
        addition to elevated vol, closing the asymmetry with the
        mean-reversion gates.
        """
        return RegimeFilter(
            signal_col=signal_col,
            conditions=[
                VolPercentileCondition(
                    col="vol_21d", lookback=252,
                    percentile=100 - vol_percentile,
                    mode="above"
                ),
                TrendConfirmationCondition(
                    signal_col=signal_col, macd_col="macd_line",
                    min_macd_magnitude=25.0,
                ),
                SignalDrawdownCondition(signal_col=signal_col, return_col="returns", max_dd=max_dd),
            ],
            logic="AND",
            output_col=f"{signal_col}_breakout_gated",
        )

    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preset filters and add filtered columns to df.

        New columns added:
            signal_rsi_vol_trend_gated
            signal_zscore_vol_bb_gated
            signal_macd_trend_gated
            signal_bb_breakout_gated
        """
        df = df.copy()
        filters = [
            self.rsi_filter(),
            self.zscore_filter(),
            self.macd_filter(),
            self.bb_breakout_filter(),
        ]
        for f in filters:
            if f.signal_col in df.columns:
                df[f.output_col] = f.apply(df)
                logger.debug(f"Applied preset filter: {f.output_col}")
        return df


# ======================================================================
# RegimeFilteredEnsemble
# ======================================================================

class RegimeFilteredEnsemble:
    """
    Regime-adaptive ensemble: combines regime-appropriate signals.

    Architecture:
        mean_reversion_pool = [signal_rsi_filtered, signal_zscore_filtered]
        trend_pool          = [signal_macd_filtered, signal_bb_filtered]

        In range-bound regime (low vol, weak trend):
            → use mean_reversion_pool only
        In trending regime (higher vol, strong direction):
            → use trend_pool only
        In transition / uncertain:
            → weighted average of both pools

    This avoids the "combine bad signals" problem in the naive ensemble.
    Each sub-pool is already filtered to its favourable regime, so the
    combination doesn't just average good and bad signals.

    Args:
        mr_signals:   Mean-reversion signal columns to combine
        trend_signals: Trend-following signal columns to combine
        vol_col:      Volatility column for regime detection
        regime_window: Window for regime rolling average
        vol_threshold: Percentile separating range-bound from trending
    """

    def __init__(
        self,
        mr_signals:    Optional[List[str]] = None,
        trend_signals: Optional[List[str]] = None,
        vol_col:       str   = "vol_21d",
        regime_window: int   = 63,
        vol_threshold: float = 40.0,   # percentile
    ):
        self.mr_signals    = mr_signals    or [
            "signal_rsi_vol_trend_gated",
            "signal_zscore_vol_bb_gated",
        ]
        self.trend_signals = trend_signals or [
            "signal_macd_trend_gated",
            "signal_bb_breakout_gated",
        ]
        self.vol_col       = vol_col
        self.regime_window = regime_window
        self.vol_threshold = vol_threshold

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preset filters then build the regime-adaptive ensemble.

        Modifies df in-place (copy returned) with new columns:
            signal_rsi_vol_trend_gated
            signal_zscore_vol_bb_gated
            signal_macd_trend_gated
            signal_bb_breakout_gated
            regime_label             — "range_bound" | "trending"
            signal_mr_pool           — vote of mean-reversion signals
            signal_trend_pool        — vote of trend signals
            signal_regime_adaptive   — regime-switched ensemble
        """
        df = df.copy()

        # Apply all preset filters
        presets = RegimeFilterPresets()
        df = presets.apply_all(df)

        # ---- Regime label ----
        if self.vol_col in df.columns:
            vol = df[self.vol_col]
            vol_threshold_series = vol.rolling(
                252, min_periods=60
            ).quantile(self.vol_threshold / 100)

            # range_bound = low vol; trending = high vol
            df["regime_label"] = "trending"
            df.loc[vol < vol_threshold_series.fillna(vol.median()), "regime_label"] = "range_bound"
        else:
            df["regime_label"] = "unknown"

        # ---- Mean-reversion pool ----
        mr_cols = [c for c in self.mr_signals if c in df.columns]
        if mr_cols:
            mr_vote = df[mr_cols].fillna(0).mean(axis=1)
            df["signal_mr_pool"] = np.sign(mr_vote).astype(int)
        else:
            df["signal_mr_pool"] = 0

        # ---- Trend pool ----
        trend_cols = [c for c in self.trend_signals if c in df.columns]
        if trend_cols:
            trend_vote = df[trend_cols].fillna(0).mean(axis=1)
            df["signal_trend_pool"] = np.sign(trend_vote).astype(int)
        else:
            df["signal_trend_pool"] = 0

        # ---- Regime-adaptive ensemble ----
        # In range-bound: use MR pool. In trending: use trend pool.
        is_range = df["regime_label"] == "range_bound"
        df["signal_regime_adaptive"] = np.where(
            is_range,
            df["signal_mr_pool"],
            df["signal_trend_pool"],
        ).astype(int)

        logger.info(
            f"RegimeFilteredEnsemble: "
            f"range_bound={is_range.mean():.1%}, "
            f"trending={(~is_range).mean():.1%}"
        )
        logger.info(
            f"  MR pool signals:    {mr_cols or 'none'}"
        )
        logger.info(
            f"  Trend pool signals: {trend_cols or 'none'}"
        )

        return df

    def filter_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary statistics for all filtered signals.

        Returns a DataFrame showing:
            - Active % (fraction of bars with non-zero signal)
            - Trade count (number of position changes)
            - Turnover vs original signal
        """
        filtered_cols = [
            c for c in df.columns
            if c.endswith("_filtered") or c.endswith("_gated")
            or c in ("signal_mr_pool", "signal_trend_pool", "signal_regime_adaptive")
        ]

        rows = []
        for col in filtered_cols:
            if col not in df.columns:
                continue
            s = df[col].fillna(0)
            # Find corresponding original signal
            base = col.split("_vol")[0].split("_trend")[0].split("_bb")[0].split("_breakout")[0]
            orig_col = base if base in df.columns else None
            orig = df[orig_col].fillna(0) if orig_col else None

            row = {
                "filtered_signal": col,
                "original_signal": orig_col or "—",
                "active_pct": round(float((s != 0).mean() * 100), 2),
                "n_trades":   int((s.diff() != 0).sum()),
                "turnover":   round(float((s.diff() != 0).mean() * 252), 2),
            }
            if orig is not None:
                orig_trades = (orig.diff() != 0).sum()
                filt_trades = (s.diff() != 0).sum()
                row["trade_reduction_pct"] = round(
                    float((1 - filt_trades / max(orig_trades, 1)) * 100), 2
                )
            rows.append(row)

        return pd.DataFrame(rows).set_index("filtered_signal") if rows else pd.DataFrame()