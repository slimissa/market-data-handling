"""
regime_detector.py — Phase 7: Regime Detection & Adaptive Signal Switching
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → signals → regime filter → [regime detection]
            → backtest → factor model

This module goes beyond the binary vol-percentile gate in regime_filter.py.
Two detectors, same interface, swappable:

    RuleBasedClassifier
        Deterministic multi-indicator classification:
            vol percentile + rolling return sign + MACD sign → regime label
        Fast, interpretable, no fitting required. Improves on the single-
        variable gate in Phase 6 by combining 3 independent signals.

    HMMRegimeDetector
        Gaussian Hidden Markov Model over [return, volatility, macd_line].
        Learns K latent states from data via Baum-Welch (EM algorithm).
        Produces a PROBABILITY of being in each regime at every bar, not
        just a label — P(trending)=0.73 carries more information than
        trending=True, and lets position size scale continuously with
        confidence rather than snapping on/off.

        Transition matrix A[i,j] = P(state_t=j | state_{t-1}=i) is learned,
        not assumed — it tells you empirically how persistent each regime
        is, which signal family should dominate, and how fast to react to
        a suspected regime change.

    AdaptiveSignalSwitch
        Combines individual signals weighted by regime probability:
            signal_adaptive = P(range_bound)*signal_rsi
                             + P(trending)   *signal_macd
                             + P(crisis)     *0
        Continuous version of the Phase 6 binary ensemble. No discontinuity
        at regime boundaries — the position scales smoothly as confidence
        shifts from one regime to another.

Mathematical core (HMM):
    Emission:   P(x_t | state=k) = N(x_t | mu_k, Sigma_k)
    Transition: P(state_t=j | state_{t-1}=i) = A[i,j]
    Fitting:    Baum-Welch (EM) maximises P(x_1:T | theta) over (mu, Sigma, A)
    Inference:  Forward algorithm gives P(state_t=k | x_1:t)  [filtered, online]
                Viterbi gives the single most likely state path [smoothed, offline]

    This module uses the FORWARD algorithm conceptually (via expanding-window
    predict_proba calls), which is the online/real-time-appropriate choice:
    P(state_t | x_1:t) uses only data up to t, with no lookahead. The
    smoothed predict_proba() on the full series uses future observations
    and is exposed separately, explicitly labelled as analysis-only.

Usage:
    # Rule-based (fast baseline)
    clf = RuleBasedClassifier()
    result = clf.classify(df)
    df["regime_label"] = result.labels

    # HMM (probabilistic, fitted)
    hmm = HMMRegimeDetector(n_states=4)
    hmm.fit(df, train_end="2022-01-01")          # fit on in-sample data only
    result = hmm.predict(df, online=True)         # P(state) per bar, no lookahead

    # Adaptive switch
    switch = AdaptiveSignalSwitch()
    switch.register("signal_rsi",  favourable=["range_bound"])
    switch.register("signal_macd", favourable=["trending_up", "trending_down"])
    df["signal_adaptive"] = switch.apply(df, result.probabilities)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Shared regime vocabulary
# ======================================================================

REGIME_NAMES = ["range_bound", "trending_up", "trending_down", "crisis"]


@dataclass
class RegimeDetectionResult:
    """
    Output of any regime detector — common interface for both
    RuleBasedClassifier and HMMRegimeDetector.

    labels:        Most likely regime per bar (argmax of probs, or rule output)
    probabilities: DataFrame, one column per regime, rows sum to 1.0
                   (rule-based: one-hot; HMM: continuous probabilities)
    transition_matrix: K x K matrix, only populated for HMM. None for rule-based.
    method:        "rule_based" | "hmm"
    """
    labels:             pd.Series
    probabilities:      pd.DataFrame
    transition_matrix:  Optional[pd.DataFrame] = None
    method:              str = ""
    state_names:         List[str] = field(default_factory=list)

    def regime_durations(self) -> pd.Series:
        """
        Average number of consecutive bars spent in each regime.
        Short durations → choppy/noisy classification. Long durations →
        persistent, tradeable regimes.
        """
        labels = self.labels.dropna()
        if len(labels) == 0:
            return pd.Series(dtype=float)

        change = labels != labels.shift(1)
        block_id = change.cumsum()
        block_sizes = labels.groupby(block_id).size()
        block_labels = labels.groupby(block_id).first()

        return block_sizes.groupby(block_labels).mean()

    def regime_frequency(self) -> pd.Series:
        """Fraction of bars spent in each regime."""
        return self.labels.value_counts(normalize=True)


# ======================================================================
# Rule-based classifier
# ======================================================================

class RuleBasedClassifier:
    """
    Deterministic multi-indicator regime classifier.

    Combines three independent signals into a single regime label:
        1. Volatility percentile  (calm vs turbulent)
        2. Rolling return sign    (up vs down vs flat)
        3. MACD line sign         (confirms trend direction)

    Decision logic (evaluated in order — first match wins):
        vol_pct > crisis_threshold                        → "crisis"
        |rolling_return| < flat_threshold                  → "range_bound"
        rolling_return > 0  AND macd_line > 0               → "trending_up"
        rolling_return < 0  AND macd_line < 0               → "trending_down"
        otherwise (return and MACD disagree)                → "range_bound"
                                                                (conflicting signals
                                                                 = no clear trend)

    This requires no fitting and runs instantly. Use as a baseline before
    comparing against the HMM, and as a fallback when there isn't enough
    history to fit an HMM reliably (HMM needs ~252+ bars to be stable).

    Args:
        vol_col:          Volatility column (e.g. "vol_21d")
        return_col:       Daily returns column
        macd_col:         MACD line column
        vol_lookback:      Window for volatility percentile ranking
        crisis_percentile: Vol percentile above which regime = "crisis"
        trend_window:      Window for rolling return (trend direction)
        flat_threshold:    |rolling return| below this (annualised) = flat/range-bound
    """

    def __init__(
        self,
        vol_col:           str   = "vol_21d",
        return_col:        str   = "returns",
        macd_col:           str   = "macd_line",
        vol_lookback:       int   = 252,
        crisis_percentile:  float = 90.0,
        trend_window:       int   = 63,
        flat_threshold:     float = 0.05,
    ):
        self.vol_col           = vol_col
        self.return_col        = return_col
        self.macd_col           = macd_col
        self.vol_lookback       = vol_lookback
        self.crisis_percentile  = crisis_percentile
        self.trend_window       = trend_window
        self.flat_threshold     = flat_threshold

    def classify(self, df: pd.DataFrame) -> RegimeDetectionResult:
        """
        Classify every bar into one of REGIME_NAMES.

        Returns:
            RegimeDetectionResult with one-hot probabilities (label is
            certain — rule-based has no uncertainty quantification).
        """
        self._require_columns(df)

        vol = df[self.vol_col]
        ret = df[self.return_col]
        macd = df[self.macd_col]

        # ---- Volatility percentile (rolling, point-in-time) ----
        vol_pct = vol.rolling(
            self.vol_lookback, min_periods=self.vol_lookback // 4
        ).rank(pct=True) * 100

        # ---- Rolling trend direction ----
        rolling_ret_annual = (
            ret.rolling(self.trend_window, min_periods=self.trend_window // 2).mean()
            * 252
        )

        # ---- Decision logic ----
        labels = pd.Series("range_bound", index=df.index)

        is_crisis    = vol_pct > self.crisis_percentile
        is_flat      = rolling_ret_annual.abs() < self.flat_threshold
        is_up        = (rolling_ret_annual > 0) & (macd > 0)
        is_down      = (rolling_ret_annual < 0) & (macd < 0)

        # Apply in priority order: crisis overrides everything
        labels[is_up & ~is_crisis]                  = "trending_up"
        labels[is_down & ~is_crisis]                 = "trending_down"
        labels[is_flat & ~is_crisis]                  = "range_bound"
        labels[is_crisis]                              = "crisis"
        # Where return/MACD disagree and not flat/crisis → leave as range_bound default

        # Warmup period: insufficient data for any classification
        warmup = vol_pct.isna() | rolling_ret_annual.isna()
        labels[warmup] = np.nan

        # ---- One-hot probabilities (rule-based = certain, no uncertainty) ----
        probs = pd.DataFrame(0.0, index=df.index, columns=REGIME_NAMES)
        for regime in REGIME_NAMES:
            probs.loc[labels == regime, regime] = 1.0
        probs.loc[warmup, :] = np.nan

        logger.info(
            f"RuleBasedClassifier: "
            f"{labels.value_counts(normalize=True).round(3).to_dict()}"
        )

        return RegimeDetectionResult(
            labels=labels,
            probabilities=probs,
            transition_matrix=None,
            method="rule_based",
            state_names=REGIME_NAMES,
        )

    def _require_columns(self, df: pd.DataFrame) -> None:
        missing = [
            c for c in [self.vol_col, self.return_col, self.macd_col]
            if c not in df.columns
        ]
        if missing:
            raise KeyError(
                f"RuleBasedClassifier requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )


# ======================================================================
# HMM regime detector
# ======================================================================

class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for probabilistic regime detection.

    Models the market as transitioning between K hidden states, each with
    its own Gaussian distribution over observed features (return, vol,
    MACD). Learns both the state-specific distributions and the
    transition matrix from historical data via Baum-Welch (EM).

    Critical design choice — online vs smoothed probabilities:
        predict_proba() computes the smoothed posterior using the full
        sequence (forward-backward algorithm) — this technically uses
        future observations relative to any given bar t. It is exposed
        for post-hoc analysis and visualisation only.

        predict_proba_online() instead re-applies the model on an
        EXPANDING window ending at t, so the probability assigned to bar t
        uses only x_1:t. This is the lookahead-free version and is the one
        that must be used when generating signals for backtesting.

    State labelling:
        HMM states are unordered by default (state 0, 1, 2 have no inherent
        meaning). After fitting, this class automatically labels each state
        by its empirical mean return and volatility:
            highest mean return, low-moderate vol  → "trending_up"
            lowest mean return  (most negative)     → "trending_down"
            lowest volatility                        → "range_bound"
            highest volatility                       → "crisis"
        This re-labelling is necessary because hmmlearn returns arbitrary
        state indices, not semantic labels.

    Args:
        n_states:       Number of hidden states (regimes) to fit.
                        4 maps to REGIME_NAMES; other values get generic
                        "state_0", "state_1", ... labels.
        feature_cols:   Columns used as the HMM observation vector.
        n_iter:         Max EM iterations for Baum-Welch.
        random_state:   Seed for reproducible fitting (EM is sensitive to init).
        covariance_type: "diag" (faster, assumes independent features) or
                        "full" (captures feature correlations, slower).
    """

    def __init__(
        self,
        n_states:        int  = 4,
        feature_cols:     List[str] = ("returns", "vol_21d", "macd_line"),
        n_iter:           int  = 100,
        random_state:     int  = 42,
        covariance_type:  Literal["diag", "full"] = "diag",
    ):
        self.n_states        = n_states
        self.feature_cols     = list(feature_cols)
        self.n_iter           = n_iter
        self.random_state     = random_state
        self.covariance_type  = covariance_type

        self._model         = None
        self._scaler_mean    = None
        self._scaler_std     = None
        self._state_labels   = None   # maps hmm state index → semantic name
        self._is_fitted      = False

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        df:        pd.DataFrame,
        train_end: Optional[str] = None,
    ) -> "HMMRegimeDetector":
        """
        Fit the HMM on historical data.

        Args:
            df:        Feature-enriched DataFrame
            train_end: ISO date string. If provided, fits only on data up
                       to this date (out-of-sample discipline). If None,
                       fits on the entire df (use only for exploratory work,
                       not for backtesting — this would be lookahead bias).

        Returns:
            self (fitted)
        """
        from hmmlearn.hmm import GaussianHMM

        self._require_columns(df)

        train_df = df if train_end is None else df.loc[:train_end]
        X = train_df[self.feature_cols].dropna()

        if len(X) < self.n_states * 30:
            raise ValueError(
                f"Insufficient data to fit HMM: {len(X)} rows for "
                f"{self.n_states} states (need ≥{self.n_states * 30})."
            )

        # Standardise features — HMM Gaussian emissions are scale-sensitive
        self._scaler_mean = X.mean()
        self._scaler_std  = X.std().replace(0, 1.0)
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X_scaled.values)

        self._model = model
        self._state_labels = self._label_states(model, X_scaled)
        self._is_fitted = True

        logger.info(
            f"HMM fitted on {len(X)} obs, {self.n_states} states, "
            f"log-likelihood={model.score(X_scaled.values):.1f}, "
            f"converged={model.monitor_.converged}"
        )
        logger.info(f"State labels: {self._state_labels}")

        return self

    def _label_states(
        self, model, X_scaled: pd.DataFrame
    ) -> Dict[int, str]:
        """
        Map arbitrary HMM state indices to semantic regime names based on
        each state's empirical mean return and volatility (un-scaled).
        """
        # Decode most likely state per training observation
        states = model.predict(X_scaled.values)

        ret_idx = self.feature_cols.index("returns") if "returns" in self.feature_cols else 0
        vol_idx = self.feature_cols.index("vol_21d") if "vol_21d" in self.feature_cols else (
            1 if len(self.feature_cols) > 1 else 0
        )

        # Compute per-state mean return and mean vol (in original units)
        ret_col = X_scaled.iloc[:, ret_idx] * self._scaler_std.iloc[ret_idx] + self._scaler_mean.iloc[ret_idx]
        vol_col = X_scaled.iloc[:, vol_idx] * self._scaler_std.iloc[vol_idx] + self._scaler_mean.iloc[vol_idx]

        state_stats = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                state_stats[s] = {"mean_ret": 0.0, "mean_vol": 0.0}
                continue
            state_stats[s] = {
                "mean_ret": float(ret_col[mask].mean()),
                "mean_vol": float(vol_col[mask].mean()),
            }

        if self.n_states == 4:
            # Standard 4-regime labelling
            by_vol = sorted(state_stats.items(), key=lambda kv: kv[1]["mean_vol"])
            crisis_state = by_vol[-1][0]            # highest vol
            remaining = [s for s, _ in by_vol if s != crisis_state]

            by_ret = sorted(
                [(s, state_stats[s]) for s in remaining],
                key=lambda kv: kv[1]["mean_ret"],
            )
            down_state  = by_ret[0][0]               # most negative return
            up_state    = by_ret[-1][0]               # most positive return
            range_state = [s for s, _ in by_ret if s not in (down_state, up_state)]
            range_state = range_state[0] if range_state else by_ret[len(by_ret)//2][0]

            labels = {
                crisis_state: "crisis",
                up_state:     "trending_up",
                down_state:   "trending_down",
                range_state:  "range_bound",
            }
            # Ensure all states got a label even with ties
            for s in range(self.n_states):
                labels.setdefault(s, f"state_{s}")
            return labels
        else:
            # Generic labelling for non-4 state counts
            by_vol = sorted(state_stats.items(), key=lambda kv: kv[1]["mean_vol"])
            return {s: f"state_{i}_vol_rank" for i, (s, _) in enumerate(by_vol)}

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_proba_online(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtered state probabilities: P(state_t | x_1:t).

        Uses ONLY past and current observations — no lookahead. This is
        the correct method to use when generating signals for backtesting.

        Implementation: re-applies the forward-backward pass on an
        expanding window ending at each bar t; only the LAST row of each
        window's output is kept, since that row reflects P(state_t | x_1:t)
        given the window contains only x_1:t. This is O(n^2) and slow for
        very long series; for production use, replace with an incremental
        forward-pass implementation.

        For typical backtest lengths (a few thousand bars), this is fast
        enough — the per-step cost is the HMM forward pass cost, not refitting.

        Returns:
            DataFrame with one column per regime name, rows sum to ~1.0,
            NaN during warmup (insufficient history for first window).
        """
        self._check_fitted()
        X = self._prepare_features(df)

        min_window = max(30, self.n_states * 10)
        unique_names = list(dict.fromkeys(self._state_labels.values()))
        probs = pd.DataFrame(
            np.nan, index=df.index, columns=unique_names
        )

        # Use hmmlearn's built-in forward-backward pass via predict_proba on
        # expanding windows — the LAST row of each window's output reflects
        # P(state_t | x_1:t), since the window itself only contains x_1:t.
        valid_idx = X.dropna().index

        for end_pos in range(min_window, len(valid_idx) + 1):
            window_idx = valid_idx[:end_pos]
            X_window = X.loc[window_idx].values

            try:
                state_probs = self._model.predict_proba(X_window)
                last_idx = window_idx[-1]
                last_probs = state_probs[-1]
                # Accumulate into the named column (sum probabilities of
                # any hmm states that share the same semantic label)
                row = {name: 0.0 for name in unique_names}
                for state_num, name in self._state_labels.items():
                    row[name] += float(last_probs[state_num])
                for name in unique_names:
                    probs.loc[last_idx, name] = row[name]
            except Exception as exc:
                logger.debug(f"HMM predict_proba failed at step {end_pos}: {exc}")
                continue

        return probs

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smoothed state probabilities: P(state_t | x_1:T), using the FULL
        sequence (forward-backward algorithm).

        WARNING: this uses future observations relative to t. Suitable for
        post-hoc regime analysis and visualisation, NOT for backtesting —
        using this to generate trading signals introduces lookahead bias.
        Use predict_proba_online() for any signal generation feeding a
        backtest.

        Returns:
            DataFrame with one column per regime name, rows sum to 1.0.
        """
        self._check_fitted()
        X = self._prepare_features(df)
        valid = X.dropna()

        state_probs = self._model.predict_proba(valid.values)

        # Use a list (not set) to preserve deterministic column order;
        # multiple HMM states can share a semantic name (e.g. two states
        # both labelled "range_bound" in a degenerate fit), so we sum
        # their probabilities into the same named column.
        unique_names = list(dict.fromkeys(self._state_labels.values()))
        probs = pd.DataFrame(0.0, index=df.index, columns=unique_names)
        for state_num, name in self._state_labels.items():
            probs.loc[valid.index, name] = (
                probs.loc[valid.index, name] + state_probs[:, state_num]
            )

        probs.loc[X.isna().any(axis=1), :] = np.nan
        return probs

    def predict(self, df: pd.DataFrame, online: bool = True) -> RegimeDetectionResult:
        """
        Full regime detection result: labels + probabilities + transition matrix.

        Args:
            online: If True, uses predict_proba_online() (lookahead-free,
                   correct for backtesting). If False, uses the smoothed
                   predict_proba() (for analysis/visualisation only).
        """
        probs = self.predict_proba_online(df) if online else self.predict_proba(df)

        all_nan_rows = probs.isna().all(axis=1)
        # idxmax cannot handle all-NaN rows; temporarily fill with 0 to get
        # a placeholder argmax, then mask those rows back to NaN afterward.
        labels = probs.fillna(0.0).idxmax(axis=1)
        labels = labels.astype(object)
        labels[all_nan_rows] = np.nan

        trans_df = pd.DataFrame(
            self._model.transmat_,
            index=[self._state_labels[i] for i in range(self.n_states)],
            columns=[self._state_labels[i] for i in range(self.n_states)],
        )
        # Collapse duplicate rows/cols if multiple states share a name
        trans_df = trans_df.groupby(trans_df.index).mean()
        trans_df = trans_df.T.groupby(trans_df.columns).mean().T

        return RegimeDetectionResult(
            labels=labels,
            probabilities=probs,
            transition_matrix=trans_df,
            method="hmm",
            state_names=list(probs.columns),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df)
        X = df[self.feature_cols].copy()
        X = (X - self._scaler_mean) / self._scaler_std
        return X

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HMMRegimeDetector is not fitted. Call .fit(df) first."
            )

    def _require_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"HMMRegimeDetector requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )


# ======================================================================
# Adaptive signal switch
# ======================================================================

class AdaptiveSignalSwitch:
    """
    Combines multiple signals weighted by their regime favourability.

    Continuous generalisation of the Phase 6 binary RegimeFilter:
        effective_signal[t] = sum_k( P(regime=k | t) * signal[t] for
                                      every signal registered to regime k )

    When a regime is uncertain (probabilities spread across states), the
    output signal is a probability-weighted blend rather than a hard
    on/off switch — avoiding discontinuous jumps at regime boundaries.

    Usage:
        switch = AdaptiveSignalSwitch()
        switch.register("signal_rsi",   favourable=["range_bound"])
        switch.register("signal_zscore", favourable=["range_bound"])
        switch.register("signal_macd",  favourable=["trending_up", "trending_down"])
        switch.register("signal_bb",    favourable=["trending_up", "trending_down"])
        # Crisis: no registration → implicitly zero in crisis for all signals

        df["signal_adaptive"] = switch.apply(df, regime_probs)
    """

    def __init__(self):
        self._registry: List[Tuple[str, List[str]]] = []

    def register(self, signal_col: str, favourable: List[str]) -> "AdaptiveSignalSwitch":
        """
        Register a signal column and the regimes it should be active in.

        Args:
            signal_col: Column name of the signal (values in {-1,0,+1})
            favourable: List of regime names where this signal should be
                       weighted. Unlisted regimes get zero weight for
                       this signal (e.g. crisis is implicitly excluded
                       unless explicitly registered).
        """
        self._registry.append((signal_col, favourable))
        return self

    def apply(
        self,
        df:           pd.DataFrame,
        regime_probs: pd.DataFrame,
        normalise:    bool = True,
    ) -> pd.Series:
        """
        Compute the regime-weighted adaptive signal.

        Args:
            df:           DataFrame containing the registered signal columns
            regime_probs: DataFrame from RegimeDetectionResult.probabilities,
                          one column per regime, rows summing to ~1.0
            normalise:    If True, divide by the sum of weights actually
                          used (handles partial regime coverage gracefully).
                          If False, raw weighted sum (can be < 1 in magnitude
                          if not all regime mass is captured by registrations).

        Returns:
            Series with continuous-weighted signal. NOTE: this is a
            continuous value, not strictly in {-1,0,+1} — round or
            threshold downstream if a discrete signal is required by
            the backtester's position-sizing logic. Most backtesters in
            this pipeline accept continuous position_scale separately,
            so np.sign() can be applied if a discrete signal is needed.
        """
        if not self._registry:
            raise ValueError("No signals registered. Call .register() first.")

        weighted_sum = pd.Series(0.0, index=df.index)
        weight_total = pd.Series(0.0, index=df.index)

        for signal_col, favourable_regimes in self._registry:
            if signal_col not in df.columns:
                logger.warning(f"AdaptiveSignalSwitch: '{signal_col}' not in df, skipping.")
                continue

            available = [r for r in favourable_regimes if r in regime_probs.columns]
            if not available:
                logger.warning(
                    f"AdaptiveSignalSwitch: none of {favourable_regimes} found in "
                    f"regime_probs columns {list(regime_probs.columns)}."
                )
                continue

            weight = regime_probs[available].sum(axis=1).fillna(0)
            sig = df[signal_col].fillna(0)

            weighted_sum += weight * sig
            weight_total += weight

        if normalise:
            result = weighted_sum / weight_total.replace(0, np.nan)
            result = result.fillna(0.0)
        else:
            result = weighted_sum

        result.name = "signal_adaptive"
        return result

    def apply_discrete(
        self,
        df:           pd.DataFrame,
        regime_probs: pd.DataFrame,
        threshold:    float = 0.0,
    ) -> pd.Series:
        """
        Convenience wrapper: applies the continuous switch, then discretises
        to {-1, 0, +1} via sign() with a deadband around zero.

        Args:
            threshold: |continuous signal| must exceed this to register as
                      non-zero. Filters out weak/uncertain blended signals.
        """
        continuous = self.apply(df, regime_probs, normalise=True)
        discrete = pd.Series(0, index=df.index)
        discrete[continuous > threshold]  = 1
        discrete[continuous < -threshold] = -1
        discrete.name = "signal_adaptive_discrete"
        return discrete

    def coverage_report(self, regime_probs: pd.DataFrame) -> dict:
        """
        Diagnostic: which regimes have registered signals, and which are
        implicitly zero-weighted (e.g. crisis with no registration).
        """
        all_regimes = set(regime_probs.columns)
        covered = set()
        for _, favourable in self._registry:
            covered.update(favourable)

        return {
            "all_regimes":       sorted(all_regimes),
            "covered_regimes":   sorted(covered & all_regimes),
            "uncovered_regimes": sorted(all_regimes - covered),
            "n_signals_registered": len(self._registry),
        }