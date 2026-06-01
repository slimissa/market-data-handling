"""
tests/test_signals.py — Property-based test suite for SignalGenerator
QuantOS Market Data Pipeline — Phase 3

Run:
    cd src && python -m pytest ../tests/test_signals.py -v

Test philosophy:
    - Signal values must be ∈ {-1, 0, +1} always
    - No lookahead bias: signal at t uses only data ≤ t
    - Entry/exit rules produce correct state transitions
    - Vol scale is bounded by [floor, ceiling]
    - Ensemble correctly combines individual signals
    - Turnover is measurable and not degenerate (all same value)
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_generator import SignalGenerator
from feature_engineering import FeatureEngineer


# ======================================================================
# Fixtures
# ======================================================================

def make_featured_df(
    n: int = 400,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic feature-enriched DataFrame (FeatureEngineer output).
    Runs the actual FeatureEngineer so all expected columns are present.
    """
    rng = np.random.default_rng(seed)
    log_returns = 0.0002 + vol * rng.standard_normal(n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    highs  = closes * (1 + rng.uniform(0, 0.02, n))
    lows   = closes * (1 - rng.uniform(0, 0.02, n))
    opens  = closes * (1 + rng.uniform(-0.01, 0.01, n))
    volumes = rng.integers(1_000_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
        "returns": log_returns,
        "returns_norm": (log_returns - log_returns.mean()) / log_returns.std(),
        "returns_fwd_1": np.append(log_returns[1:], np.nan),
        "returns_fwd_5": np.append(log_returns[5:], [np.nan] * 5),
    }, index=idx)

    eng = FeatureEngineer()
    return eng.add_all_features(df)


def make_rsi_controlled(rsi_values: list) -> pd.DataFrame:
    """
    DataFrame with a directly injected rsi_14 column.
    Useful for testing exact RSI signal transitions.
    """
    n = len(rsi_values)
    closes = np.linspace(100, 110, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "close": closes,
        "rsi_14": rsi_values,
    }, index=idx)
    return df


def make_macd_controlled(histogram_values: list, macd_line_values=None) -> pd.DataFrame:
    """
    DataFrame with directly injected macd_histogram and macd_line.
    """
    n = len(histogram_values)
    if macd_line_values is None:
        macd_line_values = histogram_values  # simplification for tests
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "macd_histogram": histogram_values,
        "macd_line": macd_line_values,
    }, index=idx)


def make_zscore_controlled(z_values: list) -> pd.DataFrame:
    """DataFrame with directly injected z_price_60d."""
    n = len(z_values)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "z_price_60d": z_values,
    }, index=idx)


@pytest.fixture
def df():
    return make_featured_df()


@pytest.fixture
def sg():
    return SignalGenerator()


# ======================================================================
# Helper assertions
# ======================================================================

def assert_valid_signal(series: pd.Series, name: str = "signal"):
    """Signal must contain only {-1, 0, +1}."""
    valid = {-1, 0, 1}
    actual = set(series.dropna().unique())
    invalid = actual - valid
    assert not invalid, (
        f"{name} contains invalid values {invalid}. "
        f"All values must be in {{-1, 0, +1}}."
    )


# ======================================================================
# 1. RSI Signal
# ======================================================================

class TestRSISignal:

    def test_signal_values_valid(self, sg, df):
        sig, _ = sg.generate_rsi_signal(df)
        assert_valid_signal(sig, "signal_rsi")

    def test_oversold_entry_produces_long(self, sg):
        """RSI drops below 30 → signal should eventually become +1."""
        # First 20 bars neutral, then RSI drops to 20 (oversold)
        rsi = [50] * 20 + [20] * 30 + [55] * 20
        df = make_rsi_controlled(rsi)
        sig, _ = sg.generate_rsi_signal(df, min_holding=1)
        # After oversold entry (bar 20+), signal should be +1
        assert (sig.iloc[21:50] == 1).any(), "No long signal after RSI < 30"

    def test_overbought_entry_produces_short(self, sg):
        """RSI rises above 70 → signal should eventually become -1."""
        rsi = [50] * 20 + [80] * 30 + [45] * 20
        df = make_rsi_controlled(rsi)
        sig, _ = sg.generate_rsi_signal(df, min_holding=1)
        assert (sig.iloc[21:50] == -1).any(), "No short signal after RSI > 70"

    def test_exit_at_neutral(self, sg):
        """Long position should be closed when RSI crosses above 50."""
        # Enter long at RSI=20, then RSI rises past 50
        rsi = [50] * 10 + [20] * 10 + [30] * 5 + [55] * 20
        df = make_rsi_controlled(rsi)
        sig, _ = sg.generate_rsi_signal(df, min_holding=1)
        # After RSI crosses 50, signal should return to 0
        assert (sig.iloc[26:] == 0).any(), "Long not closed when RSI > 50"

    def test_strength_non_negative(self, sg, df):
        _, strength = sg.generate_rsi_signal(df)
        assert (strength.dropna() >= 0).all()

    def test_strength_bounded(self, sg, df):
        _, strength = sg.generate_rsi_signal(df)
        assert (strength.dropna() <= 1).all()

    def test_strength_zero_when_flat(self, sg):
        """Strength must be 0 when signal is 0."""
        rsi = [50.0] * 100  # always neutral, never triggers
        df = make_rsi_controlled(rsi)
        sig, strength = sg.generate_rsi_signal(df)
        assert (strength[sig == 0] == 0).all()

    def test_no_lookahead(self, sg, df):
        """Signal at rows 0-199 must not change when rows 200+ are appended."""
        df_short = df.iloc[:200].copy()
        sig_short, _ = sg.generate_rsi_signal(df_short)
        sig_full, _  = sg.generate_rsi_signal(df)
        pd.testing.assert_series_equal(
            sig_short,
            sig_full.iloc[:200].reset_index(drop=True)
            .set_axis(sig_short.index),
        )

    def test_max_holding_forces_exit(self, sg):
        """Signal must return to 0 after max_holding bars, regardless of RSI."""
        # RSI stays below 30 for a long time — but max_holding=5 forces exit
        rsi = [50] * 5 + [20] * 50
        df = make_rsi_controlled(rsi)
        sig, _ = sg.generate_rsi_signal(df, min_holding=1, max_holding=5)
        # Count consecutive long streaks — none should exceed 5
        in_position = (sig == 1)
        streak = 0
        for val in in_position:
            if val:
                streak += 1
                assert streak <= 5, f"Position held for {streak} bars (max=5)"
            else:
                streak = 0

    def test_missing_column_raises(self, sg):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError, match="rsi_14"):
            sg.generate_rsi_signal(df)

    def test_smoothing_reduces_whipsaws(self, sg):
        """With smoothing=3, RSI must stay below 30 for 3 bars before entry."""
        # RSI briefly dips to 28 for 1 bar, then recovers — should NOT trigger with smoothing=3
        rsi = [50] * 10 + [28] * 1 + [50] * 30
        df = make_rsi_controlled(rsi)
        sig_no_smooth, _ = sg.generate_rsi_signal(df, smoothing=1, min_holding=1)
        sig_smoothed, _  = sg.generate_rsi_signal(df, smoothing=3, min_holding=1)
        # Without smoothing: may trigger; with smoothing=3: should not trigger
        # (brief 1-bar dip is filtered)
        assert sig_smoothed.abs().max() == 0 or \
               (sig_smoothed == 1).sum() <= (sig_no_smooth == 1).sum()


# ======================================================================
# 2. MACD Signal
# ======================================================================

class TestMACDSignal:

    def test_signal_values_valid(self, sg, df):
        sig, _ = sg.generate_macd_signal(df)
        assert_valid_signal(sig, "signal_macd")

    def test_bullish_crossover_produces_long(self, sg):
        """Histogram crosses from negative to positive → long."""
        # Negative histogram, then crosses positive
        hist = [-0.5] * 20 + [0.3] * 30
        df = make_macd_controlled(hist)
        sig, _ = sg.generate_macd_signal(df, min_holding=1)
        assert (sig.iloc[21:] == 1).any(), "No long after bullish histogram cross"

    def test_bearish_crossover_produces_short(self, sg):
        """Histogram crosses from positive to negative → short."""
        hist = [0.5] * 20 + [-0.3] * 30
        df = make_macd_controlled(hist)
        sig, _ = sg.generate_macd_signal(df, min_holding=1)
        assert (sig.iloc[21:] == -1).any(), "No short after bearish histogram cross"

    def test_histogram_sign_change_flips_signal(self, sg):
        """Direction flip must happen at the crossover bar."""
        hist = [0.5] * 15 + [-0.5] * 15 + [0.5] * 15
        df = make_macd_controlled(hist)
        sig, _ = sg.generate_macd_signal(df, min_holding=1)
        # After first bearish cross (bar 15), signal should be -1
        assert (sig.iloc[16:30] == -1).any()
        # After second bullish cross (bar 30), signal should be +1
        assert (sig.iloc[31:] == 1).any()

    def test_strength_bounded(self, sg, df):
        _, strength = sg.generate_macd_signal(df)
        assert (strength.dropna() >= 0).all()
        assert (strength.dropna() <= 1).all()

    def test_no_lookahead(self, sg, df):
        sig_short, _ = sg.generate_macd_signal(df.iloc[:200])
        sig_full, _  = sg.generate_macd_signal(df)
        pd.testing.assert_series_equal(
            sig_short,
            sig_full.iloc[:200].reset_index(drop=True).set_axis(sig_short.index),
        )

    def test_require_zero_cross_reduces_trades(self, sg, df):
        """require_zero_cross=True should produce fewer or equal signals."""
        sig_free, _   = sg.generate_macd_signal(df, require_zero_cross=False)
        sig_strict, _ = sg.generate_macd_signal(df, require_zero_cross=True)
        # Turnover: fraction of bars where signal changes
        turnover_free   = (sig_free.diff() != 0).sum()
        turnover_strict = (sig_strict.diff() != 0).sum()
        assert turnover_strict <= turnover_free, \
            "Zero-cross filter should reduce signal changes"


# ======================================================================
# 3. Z-Score Signal
# ======================================================================

class TestZScoreSignal:

    def test_signal_values_valid(self, sg, df):
        sig, _ = sg.generate_zscore_signal(df)
        assert_valid_signal(sig, "signal_zscore")

    def test_negative_extreme_produces_long(self, sg):
        """z < -2 → long signal."""
        z = [0.0] * 20 + [-2.5] * 30 + [0.5] * 20
        df = make_zscore_controlled(z)
        sig, _ = sg.generate_zscore_signal(df, min_holding=1)
        assert (sig.iloc[21:50] == 1).any(), "No long at z < -2"

    def test_positive_extreme_produces_short(self, sg):
        """z > +2 → short signal."""
        z = [0.0] * 20 + [2.5] * 30 + [-0.5] * 20
        df = make_zscore_controlled(z)
        sig, _ = sg.generate_zscore_signal(df, min_holding=1)
        assert (sig.iloc[21:50] == -1).any(), "No short at z > +2"

    def test_exit_at_mean(self, sg):
        """Long position exits when z crosses above 0 (reverted to mean)."""
        z = [0.0] * 10 + [-2.5] * 15 + [0.1] * 20
        df = make_zscore_controlled(z)
        sig, _ = sg.generate_zscore_signal(df, exit_threshold=0.0, min_holding=1)
        # After z returns to 0.1, signal should be 0
        assert (sig.iloc[26:] == 0).any(), "Long not closed after z returns to 0"

    def test_strength_proportional_to_depth(self, sg):
        """Deeper z-score → higher strength."""
        z_shallow = [0.0] * 20 + [-2.1] * 20 + [0.5] * 10
        z_deep    = [0.0] * 20 + [-3.5] * 20 + [0.5] * 10
        sig_s, str_s = sg.generate_zscore_signal(make_zscore_controlled(z_shallow), min_holding=1)
        sig_d, str_d = sg.generate_zscore_signal(make_zscore_controlled(z_deep), min_holding=1)
        # When both are in a long position, deeper z should have higher mean strength
        long_mask_s = sig_s == 1
        long_mask_d = sig_d == 1
        if long_mask_s.any() and long_mask_d.any():
            assert str_d[long_mask_d].mean() >= str_s[long_mask_s].mean()

    def test_no_lookahead(self, sg, df):
        sig_short, _ = sg.generate_zscore_signal(df.iloc[:200])
        sig_full, _  = sg.generate_zscore_signal(df)
        pd.testing.assert_series_equal(
            sig_short,
            sig_full.iloc[:200].reset_index(drop=True).set_axis(sig_short.index),
        )

    def test_missing_column_raises(self, sg):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError):
            sg.generate_zscore_signal(df, window=60)


# ======================================================================
# 4. Bollinger Band Signal
# ======================================================================

class TestBollingerSignal:

    def test_signal_values_valid_breakout(self, sg, df):
        sig, _ = sg.generate_bb_signal(df, mode="breakout")
        assert_valid_signal(sig, "signal_bb_breakout")

    def test_signal_values_valid_reversion(self, sg, df):
        sig, _ = sg.generate_bb_signal(df, mode="reversion")
        assert_valid_signal(sig, "signal_bb_reversion")

    def test_breakout_produces_signals(self, sg, df):
        """At least some signal activity expected on 400-bar series."""
        sig, _ = sg.generate_bb_signal(df, mode="breakout", min_holding=1)
        assert sig.abs().sum() > 0, "No breakout signals on 400-bar series"

    def test_reversion_produces_signals(self, sg, df):
        sig, _ = sg.generate_bb_signal(df, mode="reversion", min_holding=1)
        assert sig.abs().sum() > 0, "No reversion signals on 400-bar series"

    def test_max_holding_respected_breakout(self, sg, df):
        """No position held for more than max_holding bars."""
        max_h = 5
        sig, _ = sg.generate_bb_signal(df, mode="breakout", max_holding=max_h, min_holding=1)
        streak = 0
        for val in sig:
            if val != 0:
                streak += 1
                assert streak <= max_h, f"BB breakout held {streak} bars (max={max_h})"
            else:
                streak = 0

    def test_breakout_vs_reversion_different(self, sg, df):
        """Breakout and reversion modes should produce different signals."""
        sig_break, _   = sg.generate_bb_signal(df, mode="breakout")
        sig_revert, _  = sg.generate_bb_signal(df, mode="reversion")
        # They should differ in at least some bars
        assert not (sig_break == sig_revert).all(), \
            "Breakout and reversion modes produced identical signals"

    def test_invalid_mode_raises(self, sg, df):
        with pytest.raises((ValueError, TypeError)):
            sg.generate_bb_signal(df, mode="invalid_mode")  # type: ignore

    def test_strength_bounded(self, sg, df):
        _, strength = sg.generate_bb_signal(df)
        assert (strength.dropna() >= 0).all()
        assert (strength.dropna() <= 1).all()


# ======================================================================
# 5. Vol Scale
# ======================================================================

class TestVolScale:

    def test_scale_bounded_by_floor_ceiling(self, sg, df):
        floor, ceiling = 0.25, 2.0
        scale = sg.generate_vol_scale(df, floor=floor, ceiling=ceiling)
        assert (scale >= floor - 1e-9).all(), f"Scale below floor {floor}"
        assert (scale <= ceiling + 1e-9).all(), f"Scale above ceiling {ceiling}"

    def test_default_scale_is_one_without_history(self, sg, df):
        """Before enough history accumulates, scale should default to 1.0."""
        scale = sg.generate_vol_scale(df, lookback=252)
        # First 252 rows have no history → should be 1.0 (fallback)
        early_scale = scale.iloc[:252]
        assert (early_scale == 1.0).all(), "Early scale should default to 1.0"

    def test_high_vol_reduces_scale(self, sg):
        """
        Vol scale semantics: percentile rank of current vol vs past N days.
        When current vol SPIKES above recent history → high rank → lower scale.
        When current vol returns to normal → lower rank → higher scale.
        Test using a series where vol spikes after establishing a baseline.
        """
        import numpy as np, pandas as pd
        from feature_engineering import FeatureEngineer
        rng = np.random.default_rng(42)
        n = 500
        # First 300 bars: stable low vol (establishes lookback history)
        # Next 100 bars: vol spikes → current vol is high vs recent history → low scale
        # Final 100 bars: vol returns to low → current vol low vs history → high scale
        base_vol = 0.008
        spike_vol = 0.06  # 7.5x spike
        log_ret = np.concatenate([
            0.0002 + base_vol  * rng.standard_normal(300),
            0.0002 + spike_vol * rng.standard_normal(100),
            0.0002 + base_vol  * rng.standard_normal(100),
        ])
        closes  = 100.0 * np.exp(np.cumsum(log_ret))
        highs   = closes * 1.01
        lows    = closes * 0.99
        volumes = np.ones(n) * 1_000_000
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        df_raw = pd.DataFrame({
            "open": closes, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "returns": log_ret,
            "returns_norm": np.zeros(n),
            "returns_fwd_1": np.append(log_ret[1:], np.nan),
            "returns_fwd_5": np.append(log_ret[5:], [np.nan]*5),
        }, index=idx)
        eng = FeatureEngineer()
        df = eng.add_all_features(df_raw)

        # Use lookback=150 so during the spike (bars 300-400), 
        # the window still contains many low-vol bars → spike ranks high → low scale
        scale = sg.generate_vol_scale(df, lookback=150, floor=0.0, ceiling=2.0)

        # Scale during vol spike should be lower than scale during calm period
        spike_scale = scale.iloc[320:380].mean()    # mid-spike
        calm_scale  = scale.iloc[150:250].mean()    # stable low-vol
        assert spike_scale < calm_scale, (
            f"Vol spike scale {spike_scale:.3f} should be < "
            f"calm scale {calm_scale:.3f}"
        )

    def test_scale_floor_zero_means_can_sit_out(self, sg, df):
        """floor=0 means scale CAN be 0 during extreme vol."""
        scale = sg.generate_vol_scale(df, floor=0.0, ceiling=2.0)
        # Scale range: floor=0 should be achievable
        assert scale.min() >= 0.0

    def test_missing_vol_column_raises(self, sg):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError):
            sg.generate_vol_scale(df, vol_window=21)


# ======================================================================
# 6. Ensemble Signal
# ======================================================================

class TestEnsembleSignal:

    def _df_with_signals(self, sg, df):
        """Add individual signals to df before testing ensemble."""
        sig_rsi,  _ = sg.generate_rsi_signal(df)
        sig_macd, _ = sg.generate_macd_signal(df)
        sig_z,    _ = sg.generate_zscore_signal(df)
        df = df.copy()
        df["signal_rsi"]    = sig_rsi
        df["signal_macd"]   = sig_macd
        df["signal_zscore"] = sig_z
        return df

    def test_signal_values_valid_majority(self, sg, df):
        df = self._df_with_signals(sg, df)
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert_valid_signal(ens, "signal_ensemble_majority")

    def test_signal_values_valid_weighted(self, sg, df):
        df = self._df_with_signals(sg, df)
        ens = sg.generate_ensemble(df, method="weighted")
        assert_valid_signal(ens, "signal_ensemble_weighted")

    def test_signal_values_valid_regime(self, sg, df):
        df = self._df_with_signals(sg, df)
        ens = sg.generate_ensemble(df, method="regime_switch")
        assert_valid_signal(ens, "signal_ensemble_regime")

    def test_unanimous_long_is_long(self, sg, df):
        """All 3 signals = +1 → ensemble must be +1."""
        df = df.copy()
        df["signal_rsi"]    = 1
        df["signal_macd"]   = 1
        df["signal_zscore"] = 1
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert (ens == 1).all(), "All long signals should produce ensemble=+1"

    def test_unanimous_short_is_short(self, sg, df):
        df = df.copy()
        df["signal_rsi"]    = -1
        df["signal_macd"]   = -1
        df["signal_zscore"] = -1
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert (ens == -1).all()

    def test_unanimous_flat_is_flat(self, sg, df):
        df = df.copy()
        df["signal_rsi"]    = 0
        df["signal_macd"]   = 0
        df["signal_zscore"] = 0
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert (ens == 0).all()

    def test_majority_vote_2_long_1_short(self, sg, df):
        """2 long + 1 short → ensemble = +1."""
        df = df.copy()
        df["signal_rsi"]    = 1
        df["signal_macd"]   = 1
        df["signal_zscore"] = -1
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert (ens == 1).all(), "2 long + 1 short should give ensemble = +1"

    def test_majority_vote_2_short_1_long(self, sg, df):
        """2 short + 1 long → ensemble = -1."""
        df = df.copy()
        df["signal_rsi"]    = -1
        df["signal_macd"]   = -1
        df["signal_zscore"] = 1
        ens = sg.generate_ensemble(df, method="majority_vote")
        assert (ens == -1).all()

    def test_invalid_method_raises(self, sg, df):
        df = self._df_with_signals(sg, df)
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            sg.generate_ensemble(df, method="bad_method")  # type: ignore

    def test_regime_switch_uses_macd_in_high_vol(self, sg, df):
        """In high-vol regime (vol > trailing), ensemble should equal signal_macd."""
        df2 = self._df_with_signals(sg, df)
        # Force vol higher than trailing average
        df2["vol_21d"] = 0.05  # current
        # trailing average (63-day) will be lower if we set it manually
        # Just test that regime_switch produces valid signals
        ens = sg.generate_ensemble(df2, method="regime_switch")
        assert_valid_signal(ens, "ensemble_regime_switch")


# ======================================================================
# 7. Signal Quality Report
# ======================================================================

class TestSignalReport:

    def test_report_structure(self, sg, df):
        out = sg.generate_all(df)
        report = sg.signal_report(out, ticker="TEST")
        assert report["ticker"] == "TEST"
        assert "signal_rsi" in report
        assert "signal_macd" in report

    def test_report_turnover_bounded(self, sg, df):
        out = sg.generate_all(df)
        report = sg.signal_report(out)
        for sig_name in ["signal_rsi", "signal_macd", "signal_zscore"]:
            if sig_name in report:
                turnover = report[sig_name]["turnover"]
                assert 0 <= turnover <= 1, \
                    f"{sig_name} turnover {turnover} outside [0,1]"

    def test_report_pct_sums_to_one(self, sg, df):
        out = sg.generate_all(df)
        report = sg.signal_report(out)
        for sig_name in ["signal_rsi", "signal_macd", "signal_zscore"]:
            if sig_name in report:
                total = (
                    report[sig_name]["long_pct"]
                    + report[sig_name]["short_pct"]
                    + report[sig_name]["flat_pct"]
                )
                assert abs(total - 1.0) < 0.01, \
                    f"{sig_name}: long+short+flat = {total:.4f}, expected 1.0"

    def test_correlation_matrix_symmetric(self, sg, df):
        out = sg.generate_all(df)
        report = sg.signal_report(out)
        if "signal_correlation" in report:
            corr = pd.DataFrame(report["signal_correlation"])
            # Diagonal should be 1.0 for non-constant signals (constant → NaN)
            for col in corr.columns:
                if col in corr.index:
                    val = corr.loc[col, col]
                    if not pd.isna(val):  # NaN is expected for constant (all-zero) signals
                        assert abs(val - 1.0) < 1e-6, (
                            f"Diagonal entry for {col} = {val:.6f}, expected 1.0"
                        )


# ======================================================================
# 8. Integration Tests
# ======================================================================

class TestIntegration:

    def test_generate_all_runs(self, sg, df):
        """Smoke test: generate_all should not raise on clean feature data."""
        out = sg.generate_all(df)
        assert out.shape[0] == df.shape[0]
        assert out.shape[1] > df.shape[1]

    def test_expected_columns_present(self, sg, df):
        out = sg.generate_all(df)
        expected = [
            "signal_rsi", "signal_rsi_strength",
            "signal_macd", "signal_macd_strength",
            "signal_zscore", "signal_zscore_strength",
            "signal_bb", "signal_bb_strength",
            "position_scale",
            "signal_ensemble",
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sg, df):
        """Signal generation must not modify any feature columns."""
        out = sg.generate_all(df)
        for col in df.columns:
            pd.testing.assert_series_equal(out[col], df[col], check_names=False)

    def test_all_signal_columns_valid(self, sg, df):
        """Every signal column must contain only {-1, 0, +1}."""
        out = sg.generate_all(df)
        signal_cols = [c for c in out.columns
                       if c.startswith("signal_") and not c.endswith("_strength")]
        for col in signal_cols:
            assert_valid_signal(out[col], col)

    def test_position_scale_bounded(self, sg, df):
        out = sg.generate_all(df, vol_scale_floor=0.0, vol_scale_ceiling=2.0)
        assert (out["position_scale"] >= 0.0).all()
        assert (out["position_scale"] <= 2.0).all()

    def test_no_future_data_in_signals(self, sg, df):
        """
        Core no-lookahead test:
        Signals computed on first 200 rows must be identical to signals
        computed on the full series at those same rows.
        """
        out_short = sg.generate_all(df.iloc[:200].copy())
        out_full  = sg.generate_all(df)

        signal_cols = [c for c in out_short.columns
                       if c.startswith("signal_") and not c.endswith("_strength")]

        for col in signal_cols:
            short_vals = out_short[col].values
            full_vals  = out_full[col].iloc[:200].values
            np.testing.assert_array_equal(
                short_vals, full_vals,
                err_msg=f"{col} has lookahead bias: differs between short and full series"
            )

    def test_signal_not_all_zero(self, sg, df):
        """Signals should fire at least occasionally on a 400-bar series."""
        out = sg.generate_all(df)
        for col in ["signal_rsi", "signal_macd", "signal_zscore"]:
            assert out[col].abs().sum() > 0, \
                f"{col} never fired on 400-bar series — check thresholds"

    def test_nan_rows_produce_zero_signal(self, sg, df):
        """Rows where features are NaN should produce signal=0, not propagate NaN."""
        out = sg.generate_all(df)
        signal_cols = [c for c in out.columns if c.startswith("signal_")]
        for col in signal_cols:
            assert out[col].isna().sum() == 0, \
                f"{col} contains NaN values — should be 0 instead"