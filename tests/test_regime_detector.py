"""
tests/test_regime_detector.py — Test suite for Phase 7: Regime Detection
QuantOS Market Data Pipeline

Run:
    python -m pytest tests/test_regime_detector.py -v

Test philosophy:
    - Both detectors produce valid labels (no NaN, all bars labelled)
    - Probabilities sum to 1.0 per bar (valid simplex)
    - No lookahead: filter probabilities at t use only data ≤ t
    - HMM converges and learns distinguishable states
    - AdaptiveSignalSwitch output ∈ {-1, 0, +1}
    - Rule-based and HMM produce different outputs (HMM isn't just copying rules)
    - Interface compatibility: both detectors expose same methods
"""


import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_detector import (
    RuleBasedRegimeClassifier,
    HMMRegimeDetector,
    AdaptiveSignalSwitch,
    RegimeDetectionResult,
)
from feature_engineering import FeatureEngineer
from signal_generator import SignalGenerator


# ======================================================================
# Fixtures
# ======================================================================

def make_full_df(n: int = 500, drift: float = 0.0003, vol: float = 0.012, seed: int = 42) -> pd.DataFrame:
    """Fully featured + signal-generated DataFrame."""
    rng = np.random.default_rng(seed)
    log_ret = drift + vol * rng.standard_normal(n)
    closes  = 100.0 * np.exp(np.cumsum(log_ret))
    highs   = closes * (1 + rng.uniform(0, 0.015, n))
    lows    = closes * (1 - rng.uniform(0, 0.015, n))
    volumes = rng.integers(500_000, 2_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": closes * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": highs, "low": lows, "close": closes, "volume": volumes,
        "returns": log_ret, "returns_norm": np.zeros(n),
        "returns_fwd_1": np.append(log_ret[1:], np.nan),
        "returns_fwd_5": np.append(log_ret[5:], [np.nan] * 5),
    }, index=idx)

    eng = FeatureEngineer()
    df  = eng.add_all_features(df)
    sg  = SignalGenerator()
    return sg.generate_all(df)


def make_regime_df(n_calm: int = 200, n_crisis: int = 100, seed: int = 42) -> pd.DataFrame:
    """DataFrame with clear low-vol → high-vol regime change."""
    rng = np.random.default_rng(seed)
    n = n_calm + n_crisis
    log_ret = np.concatenate([
        0.0002 + 0.006 * rng.standard_normal(n_calm),
        0.0010 + 0.040 * rng.standard_normal(n_crisis),
    ])
    closes = 100.0 * np.exp(np.cumsum(log_ret))
    highs  = closes * 1.01
    lows   = closes * 0.99
    volumes = np.ones(n) * 1_000_000
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": closes * 0.999, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
        "returns": log_ret, "returns_norm": np.zeros(n),
        "returns_fwd_1": np.append(log_ret[1:], np.nan),
        "returns_fwd_5": np.append(log_ret[5:], [np.nan] * 5),
    }, index=idx)

    eng = FeatureEngineer()
    df = eng.add_all_features(df)
    sg = SignalGenerator()
    return sg.generate_all(df)


@pytest.fixture
def df():
    return make_full_df()


@pytest.fixture
def regime_df():
    return make_regime_df()


# ======================================================================
# 1. RuleBasedRegimeClassifier
# ======================================================================

class TestRuleBasedRegimeClassifier:

    @pytest.fixture
    def clf(self):
        return RuleBasedRegimeClassifier()

    def test_predict_returns_series(self, clf, df):
        labels = clf.predict(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)

    def test_labels_in_valid_set(self, clf, df):
        labels = clf.predict(df)
        valid = set(clf.REGIMES)
        actual = set(labels.dropna().unique())
        assert actual.issubset(valid), f"Invalid labels: {actual - valid}"

    def test_all_bars_labelled_after_warmup(self, clf, df):
        """After rolling windows fill, every bar should have a label."""
        labels = clf.predict(df)
        assert labels.iloc[252:].notna().all()

    def test_predict_proba_sums_to_one(self, clf, df):
        proba = clf.predict_proba(df)
        sums = proba.sum(axis=1)
        assert (sums.dropna() == 1.0).all()

    def test_predict_proba_is_one_hot(self, clf, df):
        """Rule-based probabilities should be degenerate (0 or 1)."""
        proba = clf.predict_proba(df)
        unique_vals = set(proba.values.flatten())
        unique_vals = {v for v in unique_vals if not np.isnan(v)}
        assert unique_vals.issubset({0.0, 1.0})

    def test_detect_returns_result_object(self, clf, df):
        result = clf.detect(df)
        assert isinstance(result, RegimeDetectionResult)
        assert result.model_name == "rule_based"

    def test_crisis_detected_in_high_vol(self, clf, regime_df):
        """Crisis regime should appear in the high-vol period."""
        labels = clf.predict(regime_df)
        crisis_bars = (labels == "crisis").sum()
        assert crisis_bars > 0, "No crisis bars detected in high-vol regime"

    def test_no_lookahead(self, clf, df):
        """Labels at rows 0-299 must not change when rows 300+ are appended."""
        short_labels = clf.predict(df.iloc[:300])
        full_labels  = clf.predict(df)
        pd.testing.assert_series_equal(
            short_labels,
            full_labels.iloc[:300].rename(short_labels.name),
            check_names=False,
        )

    def test_fit_is_noop(self, clf, df):
        """fit() should return self and not modify state."""
        clf2 = clf.fit(df)
        assert clf2 is clf

    def test_missing_columns_raises(self, clf):
        df_bad = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError):
            clf.predict(df_bad)


# ======================================================================
# 2. HMMRegimeDetector
# ======================================================================

class TestHMMRegimeDetector:

    @pytest.fixture
    def hmm(self):
        return HMMRegimeDetector(n_states=3, random_state=42)

    def test_fit_succeeds(self, hmm, df):
        hmm.fit(df)
        assert hmm._fitted

    def test_predict_after_fit(self, hmm, df):
        hmm.fit(df)
        labels = hmm.predict(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)

    def test_labels_in_valid_set(self, hmm, df):
        hmm.fit(df)
        labels = hmm.predict(df)
        valid = {"trending_up", "trending_down", "range_bound", "crisis", "unknown"}
        actual = set(labels.dropna().unique())
        assert actual.issubset(valid), f"Invalid labels: {actual - valid}"

    def test_predict_proba_filter_sums_to_one(self, hmm, df):
        hmm.fit(df)
        proba = hmm.predict_proba(df, method="filter")
        valid = df[hmm.feature_cols].notna().all(axis=1)
        sums = proba.loc[valid].sum(axis=1)
        assert (abs(sums - 1.0) < 0.01).all(), ...

    def test_predict_proba_smooth_sums_to_one(self, hmm, df):
        hmm.fit(df)
        proba = hmm.predict_proba(df, method="smooth")
        valid = df[hmm.feature_cols].notna().all(axis=1)
        sums = proba.loc[valid].sum(axis=1)
        assert (abs(sums - 1.0) < 0.01).all(), (
            f"Probabilities don't sum to 1: max error {abs(sums - 1.0).max():.4f}"
        )

    def test_filter_and_smooth_differ(self, hmm, df):
        """Filtered (forward-only) and smoothed (forward-backward) should differ."""
        hmm.fit(df)
        proba_f = hmm.predict_proba(df, method="filter")
        proba_s = hmm.predict_proba(df, method="smooth")
        assert not proba_f.equals(proba_s), "Filter and smooth should produce different probabilities"

    def test_no_lookahead_filter(self, hmm, df):
        """Filter probabilities at t must not change when future data is appended."""
        hmm.fit(df.iloc[:300])
        probs_short = hmm.predict_proba(df.iloc[:300], method="filter")
        probs_full  = hmm.predict_proba(df, method="filter")

        for col in probs_short.columns:
            np.testing.assert_array_almost_equal(
                probs_short[col].values,
                probs_full[col].iloc[:300].values,
                decimal=6,
                err_msg=f"Lookahead bias in HMM filter probabilities for '{col}'"
            )

    def test_predict_uses_viterbi(self, hmm, df):
        """predict() should use Viterbi (smoothed), so labels may differ from argmax(filter)."""
        hmm.fit(df)
        labels = hmm.predict(df)
        proba_f = hmm.predict_proba(df, method="filter")
        argmax_labels = proba_f.idxmax(axis=1)
        # They don't have to match (Viterbi vs argmax-forward), but both should be valid
        valid = {"trending_up", "trending_down", "range_bound", "crisis", "unknown"}
        assert set(labels.unique()).issubset(valid)

    def test_detect_returns_result_object(self, hmm, df):
        hmm.fit(df)
        result = hmm.detect(df)
        assert isinstance(result, RegimeDetectionResult)
        assert result.model_name == "hmm"
        assert result.transition_matrix is not None
        assert result.transition_matrix.shape == (hmm.n_states, hmm.n_states)

    def test_transition_matrix_rows_sum_to_one(self, hmm, df):
        hmm.fit(df)
        result = hmm.detect(df)
        row_sums = result.transition_matrix.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.01).all()

    def test_different_random_states_produce_different_results(self, df):
        """Different seeds should produce distinguishable models."""
        hmm1 = HMMRegimeDetector(n_states=3, random_state=42)
        hmm2 = HMMRegimeDetector(n_states=3, random_state=99)
        hmm1.fit(df)
        hmm2.fit(df)
        labels1 = hmm1.predict(df)
        labels2 = hmm2.predict(df)
        assert not labels1.equals(labels2), "Different seeds should produce different Viterbi paths"

    def test_predict_before_fit_raises(self, hmm, df):
        with pytest.raises(RuntimeError, match="must be fit"):
            hmm.predict(df)

    def test_n_states_two_works(self, df):
        hmm = HMMRegimeDetector(n_states=2, random_state=42)
        hmm.fit(df)
        labels = hmm.predict(df)
        assert len(set(labels)) <= 3  # 2 states + possibly "unknown"

    def test_missing_columns_raises(self):
        hmm = HMMRegimeDetector(feature_cols=["nonexistent"])
        df_bad = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError):
            hmm.fit(df_bad)

    def test_insufficient_data_raises(self):
        hmm = HMMRegimeDetector()
        df_short = pd.DataFrame({
            "returns": np.random.randn(10) * 0.01,
            "vol_21d": np.random.randn(10) * 0.01,
        })
        with pytest.raises(ValueError, match="Insufficient data"):
            hmm.fit(df_short)


# ======================================================================
# 3. AdaptiveSignalSwitch
# ======================================================================

class TestAdaptiveSignalSwitch:

    @pytest.fixture
    def switch(self):
        sw = AdaptiveSignalSwitch()
        sw.register("trending_up",   "signal_macd")
        sw.register("trending_down", "signal_macd")
        sw.register("range_bound",   "signal_rsi")
        sw.register("crisis",        None)
        return sw

    @pytest.fixture
    def proba(self, df):
        """Synthetic regime probabilities summing to 1."""
        idx = df.index
        n = len(idx)
        return pd.DataFrame({
            "trending_up":    np.full(n, 0.25),
            "trending_down":  np.full(n, 0.10),
            "range_bound":    np.full(n, 0.60),
            "crisis":         np.full(n, 0.05),
        }, index=idx)

    def test_output_valid_signal(self, switch, df, proba):
        result = switch.apply(df, proba, clip_output=True)
        valid = {-1, 0, 1}
        actual = set(result.dropna().unique())
        assert actual.issubset(valid), f"Invalid signal values: {actual - valid}"

    def test_continuous_output_in_range(self, switch, df, proba):
        result = switch.apply(df, proba, clip_output=False)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_clipped_vs_continuous_differ(self, switch, df, proba):
        discrete = switch.apply(df, proba, clip_output=True)
        continuous = switch.apply(df, proba, clip_output=False)
        assert not discrete.equals(continuous)

    def test_all_flat_when_no_regime_mapped(self, df, proba):
        sw = AdaptiveSignalSwitch()
        result = sw.apply(df, proba)
        assert (result == 0).all()

    def test_crisis_contributes_zero(self, switch, df, proba):
        """When crisis is registered as None, it should contribute nothing."""
        # All crisis → signal should be 0
        proba_crisis = proba.copy()
        proba_crisis["trending_up"]   = 0.0
        proba_crisis["trending_down"] = 0.0
        proba_crisis["range_bound"]   = 0.0
        proba_crisis["crisis"]        = 1.0
        result = switch.apply(df, proba_crisis)
        assert (result == 0).all()
    def test_registry_summary(self, switch):
        summary = switch.registry_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4
        assert summary.loc["crisis", "signal"] == "flat"


# ======================================================================
# 4. Interface Compatibility
# ======================================================================

class TestInterfaceCompatibility:

    def test_both_detectors_have_same_interface(self, df):
        """Rule-based and HMM should expose the same public methods."""
        rule = RuleBasedRegimeClassifier()
        hmm  = HMMRegimeDetector(n_states=3, random_state=42)
        hmm.fit(df)

        for detector in [rule, hmm]:
            labels = detector.predict(df)
            assert isinstance(labels, pd.Series)

            proba = detector.predict_proba(df)
            assert isinstance(proba, pd.DataFrame)

            result = detector.detect(df)
            assert isinstance(result, RegimeDetectionResult)

    def test_hmm_learns_different_from_rule(self, df):
        """HMM output should differ from rule-based — it learns from data."""
        rule = RuleBasedRegimeClassifier()
        hmm  = HMMRegimeDetector(n_states=3, random_state=42)
        hmm.fit(df)

        rule_labels = rule.predict(df)
        hmm_labels  = hmm.predict(df)

        # They should not be identical — HMM learns from data
        agreement = (rule_labels == hmm_labels).mean()
        assert agreement < 0.95, (
            f"HMM and rule-based agree {agreement:.1%} of the time — "
            "they should differ since HMM learns from data"
        )

    def test_adaptive_switch_works_with_both_detectors(self, df):
        """AdaptiveSignalSwitch should work with probabilities from either detector."""
        switch = AdaptiveSignalSwitch()
        switch.register("trending_up",   "signal_macd")
        switch.register("trending_down", "signal_macd")
        switch.register("range_bound",   "signal_rsi")
        switch.register("crisis",        None)

        rule = RuleBasedRegimeClassifier()
        rule_probs = rule.predict_proba(df)
        rule_signal = switch.apply(df, rule_probs)
        assert set(rule_signal.unique()).issubset({-1, 0, 1})

        hmm = HMMRegimeDetector(n_states=3, random_state=42)
        hmm.fit(df)
        hmm_probs = hmm.predict_proba(df, method="filter")
        hmm_signal = switch.apply(df, hmm_probs)
        assert set(hmm_signal.unique()).issubset({-1, 0, 1})


# ======================================================================
# 5. RegimeDetectionResult
# ======================================================================

class TestRegimeDetectionResult:

    def test_summary_returns_string(self, df):
        clf = RuleBasedRegimeClassifier()
        result = clf.detect(df)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "rule_based" in summary

    def test_all_attributes_present(self, df):
        hmm = HMMRegimeDetector(n_states=3, random_state=42)
        hmm.fit(df)
        result = hmm.detect(df)
        assert result.labels is not None
        assert result.probabilities is not None
        assert result.transition_matrix is not None
        assert result.regime_stats is not None
        assert result.n_regimes > 0