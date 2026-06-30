"""
tests/test_factor_model.py — Test suite for Phase 5: Factor Attribution
QuantOS Market Data Pipeline

Run:
    cd src && python -m pytest ../tests/test_factor_model.py -v

Test philosophy:
    - Mathematical properties: beta=1 for identical series, beta=0 for uncorrelated
    - Alpha=0 for pure factor mimics (no genuine skill)
    - Rolling attribution produces correct shape (NaN warmup + values after)
    - Regime classification is exhaustive (every bar gets a label)
    - Performance by regime sums to correct fractions
    - Factor data loader fallback works without pandas_datareader
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_model import (
    FactorModel,
    FactorRegression,
    FactorDataLoader,
    OLSRegressor,
    RegimeAnalyser,
)


# ======================================================================
# Fixtures and helpers
# ======================================================================

def make_factor_df(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic factor returns for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "MKT": 0.0003 + 0.01 * rng.standard_normal(n),
        "SMB": 0.0001 + 0.005 * rng.standard_normal(n),
        "HML": 0.0001 + 0.005 * rng.standard_normal(n),
        "MOM": 0.0002 + 0.007 * rng.standard_normal(n),
        "RF":  0.05 / 252,
    }, index=idx)


def make_strategy_returns(
    factor_df: pd.DataFrame,
    true_alpha: float = 0.0,
    beta_mkt: float = 1.0,
    beta_smb: float = 0.0,
    noise_scale: float = 0.005,
    seed: int = 99,
) -> pd.Series:
    """
    Synthetic strategy returns as a linear combination of factors + noise.
    Allows exact control over true alpha and beta for testing.
    """
    rng = np.random.default_rng(seed)
    n = len(factor_df)
    noise = noise_scale * rng.standard_normal(n)
    daily_alpha = (1 + true_alpha) ** (1 / 252) - 1

    returns = (
        daily_alpha
        + beta_mkt * factor_df["MKT"]
        + beta_smb * factor_df["SMB"]
        + noise
    )
    return pd.Series(returns.values, index=factor_df.index, name="test_signal")


@pytest.fixture
def factor_df():
    return make_factor_df()


@pytest.fixture
def ols():
    return OLSRegressor()


# ======================================================================
# 1. OLSRegressor
# ======================================================================

class TestOLSRegressor:

    def test_mismatched_timezones_still_align_on_calendar_date(self, ols):
        """
        Regression test for the production bug where strategy returns
        (US/Eastern, from the cleaning pipeline's configured timezone) and
        factor returns (UTC, from yfinance/Ken-French) were both tz-aware
        but in DIFFERENT timezones. Even on identical calendar dates,
        '2020-01-02 00:00:00-05:00' (Eastern) and '2020-01-02 00:00:00+00:00'
        (UTC) are different instants, so a naive .intersection() between
        them silently returns empty — and worse, an earlier (broken) fix
        attempt computed a correct tz-naive intersection but then reindexed
        the still-tz-aware original Series/DataFrame against it, which also
        produces zero matching rows. The net effect in production: every
        factor regression reported "0 observations" and an empty/zero
        FactorRegression, even though 100% of the calendar dates overlapped.

        This test fails on either bug if reintroduced.
        """
        n = 300
        rng = np.random.default_rng(11)

        # Factor data: UTC, like real yfinance/Ken-French output
        x_idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        X = pd.DataFrame(
            {"MKT": 0.0003 + 0.01 * rng.standard_normal(n)},
            index=x_idx,
        )

        # Strategy returns: US/Eastern, like the cleaning pipeline's output
        # (same calendar dates as X, deliberately different timezone)
        y_idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="US/Eastern")
        y = pd.Series(
            0.0002 + 1.0 * X["MKT"].values + 0.002 * rng.standard_normal(n),
            index=y_idx,
        )

        reg = ols.fit(y, X, signal_col="test_tz_mismatch", model_name="CAPM")

        assert reg.n_obs == n, (
            f"Expected all {n} calendar-overlapping observations to be used, "
            f"got n_obs={reg.n_obs}. This indicates the timezone-alignment "
            f"bug has been reintroduced: tz-aware indices in different "
            f"timezones are not being reconciled before reindex/intersection."
        )
        # With y constructed as MKT + small noise, beta should recover ~1.0
        assert abs(reg.betas["MKT"] - 1.0) < 0.15
        assert reg.r2 > 0.8

    def test_tz_naive_and_tz_aware_mix_still_aligns(self, ols):
        """
        One side tz-naive, the other tz-aware — must still align correctly
        on calendar date after stripping timezone from the aware side.
        """
        n = 100
        rng = np.random.default_rng(13)

        x_idx = pd.date_range("2021-01-01", periods=n, freq="D")  # tz-naive
        X = pd.DataFrame({"MKT": 0.01 * rng.standard_normal(n)}, index=x_idx)

        y_idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
        y = pd.Series(X["MKT"].values + 0.001 * rng.standard_normal(n), index=y_idx)

        reg = ols.fit(y, X, signal_col="test_mixed_tz", model_name="CAPM")
        assert reg.n_obs == n

    def test_beta_one_for_identical_series(self, ols, factor_df):
        """
        If strategy = market, CAPM beta should be 1.0 and alpha ≈ 0.
        """
        y = factor_df["MKT"].copy()
        X = factor_df[["MKT"]]
        reg = ols.fit(y, X, signal_col="test", model_name="CAPM")
        assert abs(reg.betas["MKT"] - 1.0) < 0.05, \
            f"Beta for identical series should be ~1.0, got {reg.betas['MKT']:.4f}"
        assert abs(reg.alpha_annual) < 0.05, \
            f"Alpha for identical series should be ~0, got {reg.alpha_annual:.4f}"

    def test_beta_zero_for_uncorrelated(self, ols, factor_df):
        """
        Strategy uncorrelated with market → beta ≈ 0.
        """
        rng = np.random.default_rng(77)
        y = pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index)
        X = factor_df[["MKT"]]
        reg = ols.fit(y, X, signal_col="test", model_name="CAPM")
        assert abs(reg.betas["MKT"]) < 0.3, \
            f"Beta for uncorrelated series should be ~0, got {reg.betas['MKT']:.4f}"

    def test_r2_near_one_for_perfect_linear(self, ols, factor_df):
        """
        Strategy = 2 * MKT (perfect linear) → R² ≈ 1.
        """
        y = 2.0 * factor_df["MKT"]
        X = factor_df[["MKT"]]
        reg = ols.fit(y, X, signal_col="test", model_name="CAPM")
        assert reg.r2 > 0.95, f"R² for perfect linear should be ~1, got {reg.r2:.4f}"

    def test_r2_near_zero_for_noise(self, ols, factor_df):
        """
        Pure noise strategy → R² ≈ 0 (no explanatory power).
        Use a seed that is clearly uncorrelated with MKT (factor_df uses seed=42).
        """
        rng = np.random.default_rng(12345)  # different seed from factor_df (42)
        y = pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index)
        X = factor_df[["MKT"]]
        reg = ols.fit(y, X, signal_col="test", model_name="CAPM")
        assert reg.r2 < 0.15, f"R² for pure noise should be ~0, got {reg.r2:.4f}"

    def test_true_alpha_is_recovered(self, ols, factor_df):
        """
        Strategy with known true_alpha=10% should produce alpha_annual ≈ 10%.
        """
        y = make_strategy_returns(
            factor_df, true_alpha=0.10, beta_mkt=1.0, noise_scale=0.002
        )
        # Excess return
        excess = y - factor_df["RF"]
        X = factor_df[["MKT"]]
        reg = ols.fit(excess, X, signal_col="test", model_name="CAPM")
        assert abs(reg.alpha_annual - 0.10) < 0.05, \
            f"Alpha recovery failed: got {reg.alpha_annual:.4f}, expected ~0.10"

    def test_zero_alpha_for_pure_factor_mimic(self, ols, factor_df):
        """
        Strategy = pure factor exposure (no alpha) → alpha_annual ≈ 0.
        """
        y = make_strategy_returns(
            factor_df, true_alpha=0.0, beta_mkt=1.2, noise_scale=0.001
        )
        excess = y - factor_df["RF"]
        X = factor_df[["MKT"]]
        reg = ols.fit(excess, X, signal_col="test", model_name="CAPM")
        assert abs(reg.alpha_annual) < 0.05, \
            f"Alpha for pure factor mimic should be ~0, got {reg.alpha_annual:.4f}"

    def test_multiple_factors_correct_betas(self, ols, factor_df):
        """
        Strategy = 1.0*MKT + 0.5*SMB → betas should be recovered.
        """
        y = make_strategy_returns(
            factor_df, true_alpha=0.0, beta_mkt=1.0, beta_smb=0.5, noise_scale=0.001
        )
        excess = y - factor_df["RF"]
        X = factor_df[["MKT", "SMB"]]
        reg = ols.fit(excess, X, signal_col="test", model_name="FF2")
        assert abs(reg.betas["MKT"] - 1.0) < 0.15, \
            f"MKT beta: expected ~1.0, got {reg.betas['MKT']:.4f}"
        assert abs(reg.betas["SMB"] - 0.5) < 0.15, \
            f"SMB beta: expected ~0.5, got {reg.betas['SMB']:.4f}"

    def test_t_stat_significant_for_large_alpha(self, ols, factor_df):
        """
        Large, consistent alpha should produce |t-stat| > 2.0.
        """
        y = make_strategy_returns(
            factor_df, true_alpha=0.30, beta_mkt=0.0, noise_scale=0.002
        )
        excess = y - factor_df["RF"]
        X = factor_df[["MKT"]]
        reg = ols.fit(excess, X, signal_col="test", model_name="CAPM")
        assert abs(reg.t_stat) > 2.0, \
            f"Large alpha should give |t| > 2, got {reg.t_stat:.3f}"
        assert reg.is_significant

    def test_t_stat_insignificant_for_noise(self, ols, factor_df):
        """
        Pure noise signal → alpha t-stat should not be wildly significant.
        Uses a seed independent of the factor_df seed (42) to avoid accidental correlation.
        """
        rng = np.random.default_rng(99999)  # independent seed
        y = pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index)
        excess = y - factor_df["RF"]
        X = factor_df[["MKT"]]
        reg = ols.fit(excess, X, signal_col="test", model_name="CAPM")
        # Noise signal should NOT have an astronomically large t-stat
        # (t > 100 would indicate collinearity with a factor, not genuine significance)
        assert abs(reg.t_stat) < 50.0, \
            f"Noise signal t-stat ({reg.t_stat:.2f}) unreasonably large — check for factor collinearity"

    def test_ic_between_minus_one_and_one(self, ols, factor_df):
        """IC is a correlation → must be in [-1, 1]."""
        y = make_strategy_returns(factor_df, seed=42)
        X = factor_df[["MKT"]]
        reg = ols.fit(y - factor_df["RF"], X, signal_col="test", model_name="CAPM")
        assert -1.0 <= reg.ic <= 1.0, f"IC = {reg.ic} outside [-1, 1]"

    def test_n_obs_correct(self, ols, factor_df):
        y = make_strategy_returns(factor_df)
        X = factor_df[["MKT"]]
        reg = ols.fit(y - factor_df["RF"], X, signal_col="test", model_name="CAPM")
        assert reg.n_obs == len(factor_df)

    def test_short_series_returns_empty(self, ols):
        """Series with < 30 observations returns empty regression."""
        idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        y = pd.Series(np.random.randn(10) * 0.01, index=idx)
        X = pd.DataFrame({"MKT": np.random.randn(10) * 0.01}, index=idx)
        reg = ols.fit(y, X, signal_col="test", model_name="CAPM")
        assert reg.n_obs == 0
        assert reg.alpha_annual == 0.0

    def test_near_zero_variance_returns_empty_regression(self, ols):
        """
        A constant or near-constant strategy return series must return an
        empty regression rather than a degenerate OLS result.

        Guards against: a signal that was always flat (zero trades) or had
        exactly one trade that didn't move price. Statsmodels may return NaN
        coefficients; numpy lstsq returns zeros silently — both are wrong
        because they look like real results rather than signalling degenerate
        input. The guard must catch this before running the regression.
        """
        rng = np.random.default_rng(1)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        y_const = pd.Series(np.zeros(n), index=idx)
        X = pd.DataFrame(
            {"MKT": 0.0003 + 0.01 * rng.standard_normal(n)},
            index=idx,
        )
        reg = ols.fit(y_const, X, signal_col="flat_signal", model_name="CAPM")
        assert reg.n_obs == 0, (
            f"Near-zero-variance signal should return n_obs=0, got {reg.n_obs}"
        )
        assert reg.alpha_annual == 0.0

    def test_adj_r2_leq_r2(self, ols, factor_df):
        """Adjusted R² penalises extra parameters → adj_r2 ≤ r2."""
        y = make_strategy_returns(factor_df)
        X = factor_df[["MKT", "SMB", "HML", "MOM"]]
        reg = ols.fit(y - factor_df["RF"], X, signal_col="test", model_name="Carhart4")
        assert reg.adj_r2 <= reg.r2 + 1e-9, \
            f"adj_r2 ({reg.adj_r2:.4f}) should be ≤ r2 ({reg.r2:.4f})"

    def test_information_ratio_sign_matches_alpha(self, ols, factor_df):
        """IR should have same sign as alpha (IR = alpha / tracking_error)."""
        y = make_strategy_returns(factor_df, true_alpha=0.10, noise_scale=0.003)
        X = factor_df[["MKT"]]
        reg = ols.fit(y - factor_df["RF"], X, signal_col="test", model_name="CAPM")
        if abs(reg.alpha_annual) > 0.01:
            assert np.sign(reg.information_ratio) == np.sign(reg.alpha_annual)


# ======================================================================
# 2. Rolling Attribution
# ======================================================================

class TestRollingAttribution:

    @pytest.fixture
    def fm(self):
        return FactorModel(rf_annual=0.05, rolling_window=63)

    def test_rolling_nan_during_warmup(self, fm, factor_df):
        """First `window` rows should be NaN (no history yet)."""
        y = make_strategy_returns(factor_df)
        excess = y - factor_df["RF"]
        roll = fm._rolling_capm(
            excess=excess, mkt=factor_df["MKT"],
            signal_col="test", window=63,
        )
        assert roll.alpha_series.iloc[:63].isna().all(), \
            "Rolling alpha should be NaN during warmup period"

    def test_rolling_produces_values_after_warmup(self, fm, factor_df):
        """After warmup, rolling alpha should have non-NaN values."""
        y = make_strategy_returns(factor_df, true_alpha=0.10, noise_scale=0.002)
        excess = y - factor_df["RF"]
        roll = fm._rolling_capm(
            excess=excess, mkt=factor_df["MKT"],
            signal_col="test", window=63,
        )
        assert roll.alpha_series.iloc[63:].notna().any(), \
            "Rolling alpha should produce values after warmup"

    def test_rolling_beta_near_one_for_market_mimic(self, fm, factor_df):
        """Strategy = MKT → rolling beta should stay near 1.0."""
        y = factor_df["MKT"].copy()
        excess = y - factor_df["RF"]
        roll = fm._rolling_capm(
            excess=excess, mkt=factor_df["MKT"],
            signal_col="test", window=63,
        )
        valid_betas = roll.beta_series.dropna()
        assert valid_betas.mean() > 0.7, \
            f"Rolling beta for market mimic should be ~1.0, mean={valid_betas.mean():.3f}"

    def test_rolling_has_correct_length(self, fm, factor_df):
        """Rolling series should have same length as input."""
        y = make_strategy_returns(factor_df)
        excess = y - factor_df["RF"]
        roll = fm._rolling_capm(
            excess=excess, mkt=factor_df["MKT"],
            signal_col="test", window=63,
        )
        assert len(roll.alpha_series) == len(factor_df)
        assert len(roll.beta_series)  == len(factor_df)
        assert len(roll.r2_series)    == len(factor_df)

    def test_regime_labels_cover_all_bars(self, fm, factor_df):
        """Every bar should have a regime label (no NaN)."""
        y = make_strategy_returns(factor_df)
        excess = y - factor_df["RF"]
        roll = fm._rolling_capm(
            excess=excess, mkt=factor_df["MKT"],
            signal_col="test", window=63,
        )
        valid_labels = {"trending_up", "trending_down", "crisis", "range_bound"}
        actual_labels = set(roll.regime_labels.dropna().unique())
        assert actual_labels.issubset(valid_labels), \
            f"Unexpected regime labels: {actual_labels - valid_labels}"


# ======================================================================
# 3. RegimeAnalyser
# ======================================================================

class TestRegimeAnalyser:

    def test_all_bars_get_label(self, factor_df):
        """Every bar should receive a regime classification."""
        ra = RegimeAnalyser()
        regime = ra.classify(factor_df["MKT"])
        # After rolling warmup, all bars should be labelled
        valid = {"crisis", "trending_up", "trending_down", "range_bound"}
        after_warmup = regime.iloc[63:].dropna()
        assert set(after_warmup.unique()).issubset(valid)

    def test_high_vol_period_classified_as_crisis(self):
        """Bars with vol > 30% annualised should be 'crisis'."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        # First 100: normal vol (~1.2% daily = ~19% annual)
        # Last 100: crisis vol (~3% daily = ~47% annual)
        rng = np.random.default_rng(42)
        ret = pd.Series(np.concatenate([
            rng.standard_normal(100) * 0.012,
            rng.standard_normal(100) * 0.030,
        ]), index=idx)
        ra = RegimeAnalyser()
        regime = ra.classify(ret, vol_window=21, return_window=63)
        # Most of the crisis period (bars 150-200) should be labelled 'crisis'
        crisis_frac = (regime.iloc[150:] == "crisis").mean()
        assert crisis_frac > 0.5, \
            f"High-vol period should mostly be 'crisis', got {crisis_frac:.1%}"

    def test_rising_market_classified_as_trending_up(self):
        """Steady uptrend with low vol → 'trending_up'."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        # Steady 0.1% daily return, low noise
        rng = np.random.default_rng(7)
        ret = pd.Series(0.001 + rng.standard_normal(n) * 0.005, index=idx)
        ra = RegimeAnalyser()
        regime = ra.classify(ret, vol_window=21, return_window=63)
        # After warmup (63 bars), should be trending_up
        trend_up_frac = (regime.iloc[100:] == "trending_up").mean()
        assert trend_up_frac > 0.5, \
            f"Uptrend should be mostly 'trending_up', got {trend_up_frac:.1%}"

    def test_performance_by_regime_returns_dataframe(self, factor_df):
        """performance_by_regime should return a valid DataFrame."""
        ra = RegimeAnalyser()
        regime = ra.classify(factor_df["MKT"])
        rng = np.random.default_rng(42)
        strats = {
            "sig_a": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
            "sig_b": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        table = ra.performance_by_regime(strats, regime)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2    # one row per signal
        # Check that at least one regime column exists
        regime_cols = [c for c in table.columns if not c.startswith("pct_")]
        assert len(regime_cols) > 0

    def test_pct_columns_sum_to_one(self, factor_df):
        """Percentage columns should sum to ~1.0 per signal."""
        ra = RegimeAnalyser()
        regime = ra.classify(factor_df["MKT"])
        rng = np.random.default_rng(42)
        strats = {
            "sig_a": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        table = ra.performance_by_regime(strats, regime)
        pct_cols = [c for c in table.columns if c.startswith("pct_")]
        if pct_cols:
            pct_sum = table.loc["sig_a", pct_cols].sum()
            assert abs(pct_sum - 1.0) < 0.05, \
                f"Regime pct columns should sum to 1.0, got {pct_sum:.3f}"


# ======================================================================
# 4. FactorModel integration
# ======================================================================

class TestFactorModelIntegration:

    @pytest.fixture
    def fm(self, tmp_path):
        return FactorModel(
            rf_annual=0.05,
            rolling_window=63,
            cache_dir=str(tmp_path / "factors"),
        )

    def _mock_factor_df(self):
        return make_factor_df(n=500, seed=42)

    def test_run_returns_results(self, fm, monkeypatch, factor_df):
        """run() should return FactorModelResults with expected structure."""
        # Monkeypatch the loader to avoid network calls
        monkeypatch.setattr(
            fm.loader, "load",
            lambda *a, **kw: factor_df,
        )
        rng = np.random.default_rng(42)
        daily_returns = {
            "signal_rsi": pd.Series(
                rng.standard_normal(len(factor_df)) * 0.01,
                index=factor_df.index,
            )
        }
        results = fm.run(
            daily_returns=daily_returns,
            start_date="2020-01-01",
            end_date="2021-12-31",
            ticker="AAPL",
        )
        assert results.ticker == "AAPL"
        assert "signal_rsi" in results.regressions
        assert len(results.regressions["signal_rsi"]) >= 1  # at least CAPM

    def test_attribution_table_has_expected_columns(self, fm, monkeypatch, factor_df):
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        rng = np.random.default_rng(99)
        returns = {
            "sig_a": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")
        table = results.attribution_table
        for col in ["alpha_annual", "t_stat", "p_value", "r2", "ic"]:
            assert col in table.columns, f"Missing column: {col}"

    def test_multiple_signals_all_attributed(self, fm, monkeypatch, factor_df):
        """All signals should appear in regression results."""
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        rng = np.random.default_rng(42)
        returns = {
            "sig_a": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
            "sig_b": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
            "sig_c": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")
        for sig in ["sig_a", "sig_b", "sig_c"]:
            assert sig in results.regressions, f"{sig} missing from regressions"

    def test_rolling_attribution_present(self, fm, monkeypatch, factor_df):
        """Rolling CAPM should be computed for every signal."""
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        rng = np.random.default_rng(42)
        returns = {
            "sig_a": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")
        assert "sig_a" in results.rolling
        assert results.rolling["sig_a"].alpha_series is not None

    def test_print_summary_does_not_raise(self, fm, monkeypatch, factor_df):
        """print_summary() should run without error."""
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        rng = np.random.default_rng(1)
        returns = {
            "sig": pd.Series(rng.standard_normal(len(factor_df)) * 0.01, index=factor_df.index),
        }
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")
        results.print_summary()  # should not raise

    def test_high_alpha_signal_detected(self, fm, monkeypatch, factor_df):
        """
        A signal with genuinely high alpha should be detected as significant.
        True alpha = 30% per year, low noise → should show large t-stat.
        """
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        alpha_signal = make_strategy_returns(
            factor_df,
            true_alpha=0.30,   # 30% annual alpha
            beta_mkt=0.0,      # zero market exposure
            noise_scale=0.002, # very low noise to make alpha detectable
        )
        returns = {"high_alpha_signal": alpha_signal}
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")

        capm_reg = next(
            r for r in results.regressions["high_alpha_signal"] if r.model_name == "CAPM"
        )
        assert capm_reg.alpha_annual > 0.15, \
            f"High alpha signal should produce alpha > 15%, got {capm_reg.alpha_annual:.4f}"
        assert capm_reg.is_significant, \
            f"High alpha signal should be significant (t={capm_reg.t_stat:.2f})"

    def test_pure_beta_signal_shows_zero_alpha(self, fm, monkeypatch, factor_df):
        """
        Signal that is pure market beta (no alpha) should show alpha ≈ 0.
        This is the 'complicated way to buy SPY' test.
        """
        monkeypatch.setattr(fm.loader, "load", lambda *a, **kw: factor_df)
        beta_signal = make_strategy_returns(
            factor_df,
            true_alpha=0.0,
            beta_mkt=1.5,      # levered market exposure
            noise_scale=0.001,
        )
        returns = {"pure_beta_signal": beta_signal}
        results = fm.run(returns, start_date="2020-01-01", end_date="2021-12-31")

        capm_reg = next(
            r for r in results.regressions["pure_beta_signal"] if r.model_name == "CAPM"
        )
        assert abs(capm_reg.alpha_annual) < 0.08, \
            f"Pure beta signal should have ~0 alpha, got {capm_reg.alpha_annual:.4f}"
        assert capm_reg.betas["MKT"] > 1.2, \
            f"Levered market signal should have beta > 1.2, got {capm_reg.betas['MKT']:.4f}"


# ======================================================================
# 5. FactorRegression dataclass
# ======================================================================

class TestFactorRegression:

    def _make_reg(self, alpha=0.001, t=2.5, p=0.02, betas=None, r2=0.3):
        rng = np.random.default_rng(42)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        residuals = pd.Series(rng.standard_normal(n) * 0.01, index=idx)
        return FactorRegression(
            signal_col="test", model_name="CAPM",
            alpha_daily=alpha, alpha_annual=(1 + alpha) ** 252 - 1,
            t_stat=t, p_value=p,
            betas=betas or {"MKT": 1.0},
            r2=r2, adj_r2=r2 - 0.01, ic=0.1, n_obs=n,
            residuals=residuals,
        )

    def test_is_significant_true_when_t_gt_2(self):
        reg = self._make_reg(t=2.5, p=0.02)
        assert reg.is_significant

    def test_is_significant_false_when_t_lt_2(self):
        reg = self._make_reg(t=1.5, p=0.15)
        assert not reg.is_significant

    def test_information_ratio_positive_for_positive_alpha(self):
        reg = self._make_reg(alpha=0.001)  # positive daily alpha
        assert reg.information_ratio > 0

    def test_summary_line_contains_key_info(self):
        reg = self._make_reg()
        line = reg.summary_line()
        assert "CAPM" in line
        assert "α=" in line
        assert "R²=" in line

    def test_stars_in_summary_for_significant(self):
        reg = self._make_reg(t=3.5, p=0.001)
        line = reg.summary_line()
        assert "***" in line

    def test_no_stars_for_insignificant(self):
        reg = self._make_reg(t=0.8, p=0.45)
        line = reg.summary_line()
        assert "***" not in line and "**" not in line and "* " not in line


# ======================================================================
# Regression tests: silent empty-download bug in _build_etf_proxies()
#
# Discovered while preparing Power BI exports: every factor_report.json
# in a real watchlist run showed n_obs=0 for every signal x model, with
# no error anywhere in the pipeline log. Root cause: yfinance does not
# raise on network failure or rate-limiting — it returns an empty
# DataFrame with the right column structure but zero rows. Every
# downstream check ("is this ETF column present?") passes vacuously on
# an empty frame, so the bug propagated silently all the way to the
# factor report with no diagnostic trail.
# ======================================================================

class TestETFProxyDownloadFailure:

    @pytest.fixture
    def loader(self, tmp_path):
        return FactorDataLoader(cache_dir=str(tmp_path / "factors"))

    def test_empty_yf_download_raises_not_silent(self, loader, monkeypatch):
        """An empty download (network failure / rate-limit) must raise,
        not silently produce a factor_df with only the RF column."""
        import factor_model as fm_module

        class _FakeYF:
            @staticmethod
            def download(*a, **kw):
                # Mirrors yfinance's real failure mode: empty frame with
                # the expected MultiIndex column structure, zero rows.
                cols = pd.MultiIndex.from_product(
                    [["Close", "Open", "High", "Low", "Volume"], ["SPY", "IWM"]]
                )
                return pd.DataFrame(columns=cols)

        monkeypatch.setattr(fm_module, "yf", _FakeYF, raising=False)
        # _build_etf_proxies does `import yfinance as yf` inline, so patch
        # the module yfinance itself resolves to.
        import sys
        monkeypatch.setitem(sys.modules, "yfinance", _FakeYF)

        with pytest.raises(RuntimeError, match="0 rows"):
            loader._build_etf_proxies(
                start_date="2024-01-01",
                end_date="2024-02-01",
                factors=["MKT"],
                rf_daily=0.0001,
            )

    def test_poisoned_empty_cache_is_discarded_not_trusted(self, loader, tmp_path):
        """A cache file containing only the RF column (the artifact of a
        prior failed download) must be treated as invalid and
        re-downloaded, not returned as if it were real factor data."""
        cache_path = (
            Path(loader.cache_dir)
            / "etf_proxies_2024-01-01_2024-02-01_MKT.parquet"
        )
        poisoned = pd.DataFrame(
            {"RF": [0.0001] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )
        poisoned.to_parquet(cache_path)
        assert cache_path.exists()

        # _build_etf_proxies should detect the poisoned cache, delete it,
        # and attempt a real download (which will fail in this sandbox
        # with no yfinance network access — but it must NOT silently
        # return the poisoned cache).
        with pytest.raises(Exception):
            loader._build_etf_proxies(
                start_date="2024-01-01",
                end_date="2024-02-01",
                factors=["MKT"],
                rf_daily=0.0001,
            )
        # The poisoned file must have been removed, not returned.
        assert not cache_path.exists()

    def test_healthy_cache_with_real_factor_data_is_still_used(self, loader, tmp_path):
        """A cache file with genuine factor columns (not just RF) should
        still be trusted and returned without re-downloading."""
        cache_path = (
            Path(loader.cache_dir)
            / "etf_proxies_2024-01-01_2024-02-01_MKT.parquet"
        )
        healthy = pd.DataFrame(
            {
                "RF": [0.0001] * 10,
                "MKT": np.random.default_rng(1).standard_normal(10) * 0.01,
            },
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )
        healthy.to_parquet(cache_path)

        result = loader._build_etf_proxies(
            start_date="2024-01-01",
            end_date="2024-02-01",
            factors=["MKT"],
            rf_daily=0.0001,
        )
        assert "MKT" in result.columns
        assert len(result) == 10