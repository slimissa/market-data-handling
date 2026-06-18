"""
regime_detector.py — Phase 6.b: Probabilistic Regime Detection
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → signals → [regime detection] → regime filter → backtest

Two detectors, same interface:

    RuleBasedRegimeClassifier
        Fast, deterministic, no fitting required. Combines vol percentile,
        rolling return direction, and MACD sign into a discrete label.
        This is what RegimeFilteredEnsemble (Phase 6) used implicitly.

    HMMRegimeDetector
        Statistical model. Learns K hidden states from historical data via
        Gaussian HMM (Baum-Welch / EM algorithm). Produces a PROBABILITY of
        being in each regime at every bar, not just a binary label.

        P(trending) = 0.73 is more useful than trending=True — it lets you
        scale position size continuously rather than snap on/off.

Both detectors expose:
    .fit(df)                          — learn parameters (HMM only; rule-based is no-op)
    .predict(df)   → regime_label Series
    .predict_proba(df) → DataFrame of per-regime probabilities

AdaptiveSignalSwitch:
    Combines multiple signals using regime PROBABILITIES as continuous weights:
        effective_signal = Sum_k  P(regime=k) * signal_for_regime_k

    This generalises the Phase 6 binary gate into a continuous, confidence-
    weighted ensemble. In a transition period (P(trending)=0.5, P(range)=0.5),
    the signal blends rather than flipping abruptly — reducing whipsaw risk
    at regime boundaries.

Mathematical core (Gaussian HMM, K states):
    Emission:    P(x_t | state=k) = N(x_t | mu_k, Sigma_k)
    Transition:  A[i,j] = P(state_t=j | state_{t-1}=i)
    Inference:   Forward algorithm -> P(state_t=k | x_{1:t})   [filtered probability]
                 Viterbi algorithm -> most likely state sequence (smoothed)

Usage:
    # Rule-based (fast baseline)
    clf = RuleBasedRegimeClassifier()
    df["regime_label"] = clf.predict(df)

    # HMM (probabilistic, learns from data)
    hmm = HMMRegimeDetector(n_states=3)
    hmm.fit(df)
    probs = hmm.predict_proba(df)        # columns: trending_up, range_bound, ...
    labels = hmm.predict(df)             # most likely regime per bar

    # Adaptive switching
    switch = AdaptiveSignalSwitch()
    switch.register("trending_up",   "signal_macd")
    switch.register("range_bound",   "signal_rsi")
    switch.register("crisis",        None)   # flat in crisis
    df["signal_adaptive"] = switch.apply(df, probs)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class RegimeDetectionResult:
    """
    Output of any regime detector.

    labels:        Most likely regime per bar (string)
    probabilities: DataFrame, one column per regime, rows sum to 1.0
    model_name:    "rule_based" | "hmm"
    n_regimes:     Number of distinct regimes
    transition_matrix: For HMM only — P(state_j | state_i). None for rule-based.
    regime_stats:  Per-regime descriptive stats (mean return, vol, frequency)
    """
    labels:             pd.Series
    probabilities:       pd.DataFrame
    model_name:          str
    n_regimes:           int
    transition_matrix:   Optional[pd.DataFrame] = None
    regime_stats:         Optional[pd.DataFrame] = None

    def summary(self) -> str:
        freq = self.labels.value_counts(normalize=True).round(3)
        lines = [f"Regime detection ({self.model_name}, {self.n_regimes} states):"]
        for label, pct in freq.items():
            lines.append(f"  {label:<15s} {pct:.1%}")
        return "\n".join(lines)


# ======================================================================
# Rule-Based Classifier
# ======================================================================

class RuleBasedRegimeClassifier:
    """
    Fast, deterministic regime classification from indicator thresholds.

    Combines three signals into a discrete regime label:
        1. Volatility percentile (vol_21d vs trailing history)
        2. Rolling return direction (63d momentum)
        3. MACD line sign (trend confirmation)

    Decision logic (in priority order):
        vol > crisis_percentile                       → "crisis"
        |rolling_return| > trend_threshold AND
            macd agrees with direction                 → "trending_up" / "trending_down"
        otherwise                                       → "range_bound"

    No fitting required — this is a fixed-rule baseline.
    Use this when you need something working immediately, or as a sanity
    check against the HMM's learned regimes.

    Args:
        vol_col:           Volatility column (e.g. "vol_21d")
        return_col:        Daily returns column
        macd_col:          MACD line column
        crisis_percentile: vol above this percentile (trailing 252d) = crisis
        trend_window:      Window for rolling return (default 63 = ~3mo)
        trend_threshold:   Annualised return magnitude defining a "trend"
        lookback:          Trailing window for vol percentile calculation
    """

    REGIMES = ["trending_up", "trending_down", "range_bound", "crisis"]

    def __init__(
        self,
        vol_col:           str   = "vol_21d",
        return_col:        str   = "returns",
        macd_col:           str   = "macd_line",
        crisis_percentile: float = 90.0,
        trend_window:       int   = 63,
        trend_threshold:    float = 0.10,
        lookback:           int   = 252,
    ):
        self.vol_col           = vol_col
        self.return_col        = return_col
        self.macd_col          = macd_col
        self.crisis_percentile = crisis_percentile
        self.trend_window      = trend_window
        self.trend_threshold   = trend_threshold
        self.lookback          = lookback

    def fit(self, df: pd.DataFrame) -> "RuleBasedRegimeClassifier":
        """No-op — rule-based classifier requires no fitting."""
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the most likely regime label per bar."""
        return self._classify(df)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return one-hot probabilities (1.0 for the assigned regime, 0.0 else).

        Rule-based classification is deterministic, so probabilities are
        degenerate (always 0 or 1). Provided for interface compatibility
        with HMMRegimeDetector.
        """
        labels = self._classify(df)
        proba = pd.DataFrame(0.0, index=df.index, columns=self.REGIMES)
        for regime in self.REGIMES:
            proba.loc[labels == regime, regime] = 1.0
        return proba

    def detect(self, df: pd.DataFrame) -> RegimeDetectionResult:
        """Full detection result with stats."""
        labels = self._classify(df)
        proba  = self.predict_proba(df)
        stats  = self._regime_stats(df, labels)
        return RegimeDetectionResult(
            labels=labels,
            probabilities=proba,
            model_name="rule_based",
            n_regimes=len(self.REGIMES),
            transition_matrix=None,
            regime_stats=stats,
        )

    def _classify(self, df: pd.DataFrame) -> pd.Series:
        self._check_columns(df)

        vol    = df[self.vol_col]
        ret    = df[self.return_col]
        macd   = df[self.macd_col]

        # 1. Crisis: vol above the Nth percentile of trailing history
        vol_threshold = vol.rolling(
            self.lookback, min_periods=self.lookback // 4
        ).quantile(self.crisis_percentile / 100)
        is_crisis = (vol > vol_threshold).fillna(False)

        # 2. Trend: rolling annualised return beyond threshold
        rolling_ret_annual = ret.rolling(
            self.trend_window, min_periods=self.trend_window // 2
        ).mean() * 252

        is_trending_up   = (rolling_ret_annual >  self.trend_threshold) & (macd > 0)
        is_trending_down = (rolling_ret_annual < -self.trend_threshold) & (macd < 0)

        # ---- Assign labels with priority: crisis > trend > range_bound ----
        labels = pd.Series("range_bound", index=df.index)
        labels[is_trending_up]   = "trending_up"
        labels[is_trending_down] = "trending_down"
        labels[is_crisis]        = "crisis"   # crisis overrides trend

        return labels

    def _regime_stats(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Descriptive statistics per regime: mean return, vol, frequency."""
        rows = []
        for regime in self.REGIMES:
            mask = labels == regime
            if mask.sum() < 2:
                rows.append({
                    "regime": regime, "frequency": 0.0,
                    "mean_return_annual": np.nan, "vol_annual": np.nan,
                    "sharpe": np.nan, "n_bars": int(mask.sum()),
                })
                continue
            r = df.loc[mask, self.return_col].dropna()
            mean_ann = float(r.mean() * 252)
            vol_ann  = float(r.std() * np.sqrt(252))
            sharpe   = float(mean_ann / vol_ann) if vol_ann > 0 else 0.0
            rows.append({
                "regime": regime,
                "frequency": round(float(mask.mean()), 4),
                "mean_return_annual": round(mean_ann, 4),
                "vol_annual": round(vol_ann, 4),
                "sharpe": round(sharpe, 4),
                "n_bars": int(mask.sum()),
            })
        return pd.DataFrame(rows).set_index("regime")

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in [self.vol_col, self.return_col, self.macd_col]
                   if c not in df.columns]
        if missing:
            raise KeyError(
                f"RuleBasedRegimeClassifier requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )


# ======================================================================
# HMM Regime Detector
# ======================================================================

class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for probabilistic regime detection.

    Learns K hidden market states from a multivariate observation vector
    (typically returns and volatility). Unlike the rule-based classifier,
    this:
        - Learns regime parameters (mean, covariance) directly from data
        - Learns transition probabilities between regimes
        - Produces continuous probabilities, not just hard labels

    The model does NOT know in advance which state is "trending" vs
    "range_bound" — states are unlabelled until you inspect their
    learned mean return and volatility and assign semantic labels.

    Mathematical model:
        Emission:    x_t | state=k  ~  N(mu_k, Sigma_k)
        Transition:  P(state_t=j | state_{t-1}=i) = A[i,j]
        Estimation:  Baum-Welch (EM algorithm) via hmmlearn

    Args:
        n_states:     Number of hidden regimes to learn (typically 2-4)
        feature_cols: Observation columns used to fit the HMM
                      Default: returns + vol_21d (captures direction + magnitude)
        n_iter:       Max EM iterations for Baum-Welch
        random_state: Seed for reproducibility (EM has local optima)
        covariance_type: "diag" (independent features) or "full" (correlated)
    """

    def __init__(
        self,
        n_states:         int = 3,
        feature_cols:      Optional[List[str]] = None,
        n_iter:            int = 200,
        random_state:      int = 42,
        covariance_type:   Literal["diag", "full", "spherical"] = "diag",
    ):
        self.n_states        = n_states
        self.feature_cols    = feature_cols or ["returns", "vol_21d"]
        self.n_iter          = n_iter
        self.random_state    = random_state
        self.covariance_type = covariance_type

        self._model        = None
        self._scaler_mean   = None
        self._scaler_std    = None
        self._state_labels  = None   # learned mapping: state_idx -> semantic label
        self._fitted        = False

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the Gaussian HMM on historical data.

        Standardises features (zero mean, unit variance) before fitting —
        HMM emission covariances are sensitive to feature scale, and
        returns (~1%) vs volatility (~15% annualised) live on very
        different scales without standardisation.
        """
        try:
            from hmmlearn import hmm
        except ImportError as exc:
            raise ImportError(
                "HMMRegimeDetector requires hmmlearn: pip install hmmlearn"
            ) from exc

        X, valid_idx = self._prepare_features(df)
        if len(X) < 30:
            raise ValueError(
                f"Insufficient data to fit HMM: {len(X)} valid rows "
                f"(need at least 30)."
            )

        # Standardise
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std  = X.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0  # avoid div-by-zero
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X_scaled)
        self._model = model

        # Assign semantic labels based on learned state characteristics
        self._state_labels = self._infer_semantic_labels(X_scaled, model)
        self._fitted = True

        logger.info(
            f"HMM fitted: {self.n_states} states, "
            f"{len(X)} observations, "
            f"converged={model.monitor_.converged}"
        )
        logger.info(f"State -> regime mapping: {self._state_labels}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Return the most likely regime label per bar (Viterbi decoding).

        Uses the full sequence to find the globally most likely state path —
        this is "smoothed" in the sense that it considers the entire fitted
        window. For strict point-in-time inference use predict_proba()
        with method="filter".
        """
        self._check_fitted()
        X, valid_idx = self._prepare_features(df)
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        state_seq = self._model.predict(X_scaled)
        labels = pd.Series("unknown", index=df.index)
        semantic = [self._state_labels[s] for s in state_seq]
        labels.loc[valid_idx] = semantic

        return labels

    def predict_proba(
        self,
        df: pd.DataFrame,
        method: Literal["filter", "smooth"] = "filter",
    ) -> pd.DataFrame:
        """
        Return per-regime probability at each bar.

        Args:
            method: "filter" — forward algorithm, P(state_t | x_{1:t}).
                              Point-in-time, no lookahead. USE THIS for backtesting.
                    "smooth" — forward-backward, P(state_t | x_{1:T}).
                              Uses future data too — for diagnostics only,
                              NEVER for trading signals (lookahead bias).

        Returns:
            DataFrame with one column per semantic regime label, rows sum to 1.0.
        """
        self._check_fitted()
        X, valid_idx = self._prepare_features(df)
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        if method == "filter":
            state_probs = self._forward_only(X_scaled)
        else:
            _, state_probs = self._model.score_samples(X_scaled)

        unique_labels = sorted(set(self._state_labels.values()))
        proba = pd.DataFrame(
            0.0, index=df.index, columns=unique_labels,
        )

        # Aggregate raw state probabilities into semantic labels
        # (multiple HMM states might map to the same semantic regime)
        agg = np.zeros((len(X_scaled), len(unique_labels)))
        label_to_col = {label: i for i, label in enumerate(unique_labels)}
        for state_idx, label in self._state_labels.items():
            agg[:, label_to_col[label]] += state_probs[:, state_idx]

        proba.loc[valid_idx, :] = agg

        return proba

    def detect(self, df: pd.DataFrame) -> RegimeDetectionResult:
        """Full detection result including transition matrix and stats."""
        self._check_fitted()
        labels = self.predict(df)
        proba  = self.predict_proba(df)

        trans_df = pd.DataFrame(
            self._model.transmat_,
            index=[f"state_{i}({self._state_labels[i]})" for i in range(self.n_states)],
            columns=[f"state_{i}({self._state_labels[i]})" for i in range(self.n_states)],
        )

        stats = self._compute_regime_stats(df, labels)

        return RegimeDetectionResult(
            labels=labels,
            probabilities=proba,
            model_name="hmm",
            n_regimes=len(set(self._state_labels.values())),
            transition_matrix=trans_df.round(4),
            regime_stats=stats,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Index]:
        """Extract and clean the feature matrix for HMM fitting/inference."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"HMMRegimeDetector requires columns {missing}. "
                f"Available: {list(df.columns)}"
            )
        sub = df[self.feature_cols].dropna()
        return sub.values, sub.index

    def _forward_only(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute forward-algorithm (filtered) state probabilities manually.

        hmmlearn's score_samples() uses forward-backward (smoothed, has
        lookahead). For point-in-time trading signals we need the forward
        pass only: P(state_t | x_{1:t}), not P(state_t | x_{1:T}).
        """
        n_obs = len(X_scaled)
        n_states = self.n_states

        log_startprob = np.log(self._model.startprob_ + 1e-300)
        log_transmat  = np.log(self._model.transmat_ + 1e-300)

        framelogprob = self._model._compute_log_likelihood(X_scaled)

        log_alpha = np.zeros((n_obs, n_states))
        log_alpha[0] = log_startprob + framelogprob[0]

        for t in range(1, n_obs):
            for j in range(n_states):
                log_alpha[t, j] = (
                    self._logsumexp(log_alpha[t - 1] + log_transmat[:, j])
                    + framelogprob[t, j]
                )

        max_per_row = log_alpha.max(axis=1, keepdims=True)
        probs = np.exp(log_alpha - max_per_row)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs /= row_sums

        return probs

    @staticmethod
    def _logsumexp(arr: np.ndarray) -> float:
        m = np.max(arr)
        return m + np.log(np.sum(np.exp(arr - m)))

    def _infer_semantic_labels(
        self,
        X_scaled: np.ndarray,
        model,
    ) -> Dict[int, str]:
        """
        Map unlabelled HMM states to semantic regime names based on
        learned characteristics: mean return and mean volatility
        (in standardised feature space).

        Heuristic:
            highest vol state                          -> crisis (if n_states >= 3)
            among remaining: highest mean return        -> trending_up
                              lowest mean return         -> trending_down
                              middle (if any)            -> range_bound
        """
        state_seq = model.predict(X_scaled)
        ret_idx = (
            self.feature_cols.index("returns")
            if "returns" in self.feature_cols else 0
        )
        vol_idx = (
            self.feature_cols.index("vol_21d")
            if "vol_21d" in self.feature_cols
            else (1 if len(self.feature_cols) > 1 else 0)
        )

        state_chars = {}
        for s in range(self.n_states):
            mask = state_seq == s
            if mask.sum() == 0:
                state_chars[s] = {"mean_ret": 0.0, "mean_vol": 0.0}
                continue
            state_chars[s] = {
                "mean_ret": float(X_scaled[mask, ret_idx].mean()),
                "mean_vol": (
                    float(X_scaled[mask, vol_idx].mean())
                    if len(self.feature_cols) > 1 else 0.0
                ),
            }

        vol_ranked = sorted(
            state_chars.items(), key=lambda kv: kv[1]["mean_vol"], reverse=True
        )
        crisis_state = vol_ranked[0][0] if (len(vol_ranked) > 0 and self.n_states >= 3) else None

        remaining = {s: c for s, c in state_chars.items() if s != crisis_state}
        ret_ranked = sorted(remaining.items(), key=lambda kv: kv[1]["mean_ret"], reverse=True)

        labels: Dict[int, str] = {}
        if crisis_state is not None:
            labels[crisis_state] = "crisis"

        if len(ret_ranked) >= 2:
            labels[ret_ranked[0][0]]  = "trending_up"
            labels[ret_ranked[-1][0]] = "trending_down"
            for s, _ in ret_ranked[1:-1]:
                labels[s] = "range_bound"
        elif len(ret_ranked) == 1:
            labels[ret_ranked[0][0]] = "range_bound"

        for s in range(self.n_states):
            if s not in labels:
                labels[s] = "range_bound"

        return labels

    def _compute_regime_stats(
        self, df: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        rows = []
        ret_col = "returns" if "returns" in df.columns else self.feature_cols[0]
        for regime in sorted(labels.dropna().unique()):
            mask = labels == regime
            r = df.loc[mask, ret_col].dropna() if ret_col in df.columns else pd.Series(dtype=float)
            if len(r) < 2:
                continue
            mean_ann = float(r.mean() * 252)
            vol_ann  = float(r.std() * np.sqrt(252))
            sharpe   = float(mean_ann / vol_ann) if vol_ann > 0 else 0.0
            rows.append({
                "regime": regime,
                "frequency": round(float(mask.mean()), 4),
                "mean_return_annual": round(mean_ann, 4),
                "vol_annual": round(vol_ann, 4),
                "sharpe": round(sharpe, 4),
                "n_bars": int(mask.sum()),
            })
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "HMMRegimeDetector must be fit() before predict()/predict_proba()."
            )


# ======================================================================
# Adaptive Signal Switch
# ======================================================================

class AdaptiveSignalSwitch:
    """
    Combine multiple signals using regime probabilities as continuous weights.

    Generalises the Phase 6 binary RegimeFilter into a smooth blend:

        effective_signal(t) = Sum_k  P(regime=k, t) * signal_for_regime(k)(t)

    At a regime boundary where P(trending)=0.5, P(range_bound)=0.5, the
    output blends both signals rather than snapping discretely — this
    reduces whipsaw trades exactly at the moments when regime classification
    is least confident.

    Usage:
        switch = AdaptiveSignalSwitch()
        switch.register("trending_up",   "signal_macd")
        switch.register("trending_down", "signal_macd")
        switch.register("range_bound",   "signal_rsi")
        switch.register("crisis",        None)   # flat — no signal in crisis

        df["signal_adaptive"] = switch.apply(df, regime_probs)
    """

    def __init__(self):
        self._registry: Dict[str, Optional[str]] = {}

    def register(self, regime: str, signal_col: Optional[str]) -> "AdaptiveSignalSwitch":
        """
        Map a regime to the signal column that should be active in it.

        Args:
            regime:     Regime label (must match a column in regime_probs)
            signal_col: Signal column to use when this regime is active.
                       None means "flat" (always 0) in this regime.
        """
        self._registry[regime] = signal_col
        return self

    def apply(
        self,
        df: pd.DataFrame,
        regime_probs: pd.DataFrame,
        clip_output: bool = True,
    ) -> pd.Series:
        """
        Compute the regime-weighted adaptive signal.

        Args:
            df:           DataFrame containing the registered signal columns
            regime_probs: DataFrame of per-regime probabilities (rows sum to 1.0)
            clip_output:  If True, round the continuous blend to the nearest
                         of {-1, 0, +1} using sign(); if False, return the
                         raw continuous weighted value (useful for position
                         sizing rather than discrete signals).

        Returns:
            Series — discrete {-1,0,+1} if clip_output, else continuous float.
        """
        if not self._registry:
            result = pd.Series(0, index=df.index, name="signal_adaptive")
            return result.astype(int) if clip_output else result

        common_idx = df.index.intersection(regime_probs.index)

        weighted = pd.Series(0.0, index=common_idx)

        for regime, signal_col in self._registry.items():
            if regime not in regime_probs.columns:
                logger.warning(
                    f"Regime '{regime}' not found in regime_probs columns "
                    f"{list(regime_probs.columns)} — skipping."
                )
                continue

            prob = regime_probs.loc[common_idx, regime].fillna(0.0)

            if signal_col is None:
                continue  # flat in this regime — contributes 0
            if signal_col not in df.columns:
                logger.warning(
                    f"Signal column '{signal_col}' not found for regime "
                    f"'{regime}' — treating as flat."
                )
                continue

            sig = df.loc[common_idx, signal_col].fillna(0)
            weighted = weighted.add(prob * sig, fill_value=0.0)

        result = weighted.reindex(df.index, fill_value=0.0)

        if clip_output:
            return np.sign(result).astype(int).rename("signal_adaptive")
        return result.rename("signal_adaptive_continuous")

    def registry_summary(self) -> pd.DataFrame:
        """Return the current regime->signal mapping as a DataFrame."""
        return pd.DataFrame([
            {"regime": k, "signal": v or "flat"}
            for k, v in self._registry.items()
        ]).set_index("regime")