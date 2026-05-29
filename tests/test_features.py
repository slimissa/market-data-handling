"""
tests/test_features.py — Property-based test suite for FeatureEngineer
QuantOS Market Data Pipeline — Phase 2

Run:
    cd src && python -m pytest ../tests/test_features.py -v

Test philosophy:
    - Test mathematical properties (RSI ∈ [0,100]), not specific values
    - Test edge cases: constant series, all-zero volume, single row
    - Test NaN structure: where NaNs appear and where they must not
    - Test relationships: ATR >= high-low always, bb_upper > bb_lower always
    - One integration smoke test that chains DataCleaner → FeatureEngineer
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Allow running from tests/ or from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import FeatureEngineer


# ======================================================================
# Fixtures
# ======================================================================

def make_ohlcv(
    n: int = 300,
    start_price: float = 100.0,
    drift: float = 0.0002,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame that mimics DataCleaner output.

    Prices follow a log-normal random walk:
        P_t = P_{t-1} * exp(drift + vol * Z),  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    log_returns = drift + vol * rng.standard_normal(n)
    closes = start_price * np.exp(np.cumsum(log_returns))
    highs  = closes * (1 + rng.uniform(0, 0.02, n))
    lows   = closes * (1 - rng.uniform(0, 0.02, n))
    opens  = closes * (1 + rng.uniform(-0.01, 0.01, n))
    volumes = rng.integers(1_000_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
        "returns": log_returns,
        "returns_norm": (log_returns - log_returns.mean()) / log_returns.std(),
        "returns_fwd_1": np.append(log_returns[1:], np.nan),
        "returns_fwd_5": np.append(log_returns[5:], [np.nan] * 5),
    }, index=idx)

    return df


def make_constant_up(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """Every close is higher than the previous: RSI should → 100."""
    closes = np.linspace(price, price * 2, n)
    return _wrap_closes(closes, n)


def make_constant_down(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """Every close is lower than the previous: RSI should → 0."""
    closes = np.linspace(price * 2, price, n)
    return _wrap_closes(closes, n)


def _wrap_closes(closes: np.ndarray, n: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    df = pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": np.ones(n) * 1_000_000,
        "returns": log_ret,
        "returns_norm": np.zeros(n),
        "returns_fwd_1": np.append(log_ret[1:], np.nan),
        "returns_fwd_5": np.append(log_ret[5:], [np.nan] * 5),
    }, index=idx)
    return df


@pytest.fixture
def df():
    return make_ohlcv()


@pytest.fixture
def eng():
    return FeatureEngineer()


# ======================================================================
# 1. Realised Volatility
# ======================================================================

class TestRealisedVolatility:

    def test_vol_non_negative(self, eng, df):
        out = eng.add_realised_volatility(df)
        for col in [c for c in out.columns if c.startswith("vol_")]:
            assert (out[col].dropna() >= 0).all(), f"{col} has negative values"

    def test_annual_scaling(self, eng, df):
        """vol_Xd_annual should equal vol_Xd * sqrt(252) everywhere non-NaN."""
        out = eng.add_realised_volatility(df, windows=[21])
        mask = out["vol_21d"].notna() & out["vol_21d_annual"].notna()
        ratio = out.loc[mask, "vol_21d_annual"] / out.loc[mask, "vol_21d"]
        np.testing.assert_allclose(ratio, np.sqrt(252), rtol=1e-6)

    def test_nan_at_start(self, eng, df):
        """First (window//2 - 1) rows should be NaN due to min_periods."""
        out = eng.add_realised_volatility(df, windows=[21])
        # With min_periods = 10 (21 * 0.5), first 9 rows must be NaN
        assert out["vol_21d"].iloc[:9].isna().all()

    def test_multiple_windows(self, eng, df):
        out = eng.add_realised_volatility(df, windows=[5, 21, 63])
        for w in [5, 21, 63]:
            assert f"vol_{w}d" in out.columns
            assert f"vol_{w}d_annual" in out.columns

    def test_longer_window_smoother(self, eng, df):
        """63-day vol should have lower variance than 5-day vol (smoother)."""
        out = eng.add_realised_volatility(df, windows=[5, 63])
        mask = out["vol_5d"].notna() & out["vol_63d"].notna()
        assert out.loc[mask, "vol_63d"].std() < out.loc[mask, "vol_5d"].std()

    def test_missing_returns_column_raises(self, eng):
        df_bad = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError, match="returns"):
            eng.add_realised_volatility(df_bad)

    def test_vol_spikes_on_high_vol_period(self, eng):
        """Inject a volatile period and verify vol spikes."""
        df_calm = make_ohlcv(n=300, vol=0.005, seed=1)
        df_crisis = make_ohlcv(n=300, vol=0.05, seed=2)  # 10x volatility
        calm_out  = eng.add_realised_volatility(df_calm, windows=[21])
        crisis_out = eng.add_realised_volatility(df_crisis, windows=[21])
        calm_mean  = calm_out["vol_21d"].dropna().mean()
        crisis_mean = crisis_out["vol_21d"].dropna().mean()
        assert crisis_mean > calm_mean * 3, "High-vol period should produce much larger vol_21d"


# ======================================================================
# 2. RSI
# ======================================================================

class TestRSI:

    def test_rsi_bounds(self, eng, df):
        """RSI must always be in [0, 100]."""
        out = eng.add_rsi(df)
        rsi = out["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_constant_up(self, eng):
        """All-up series → RSI should converge toward 100."""
        out = eng.add_rsi(make_constant_up(n=100), window=14)
        # After convergence (well past window), RSI should be > 90
        assert out["rsi_14"].dropna().iloc[-10:].mean() > 90

    def test_rsi_constant_down(self, eng):
        """All-down series → RSI should converge toward 0."""
        out = eng.add_rsi(make_constant_down(n=100), window=14)
        assert out["rsi_14"].dropna().iloc[-10:].mean() < 10

    def test_rsi_nan_at_start(self, eng, df):
        """First `window` rows should be NaN (min_periods=window)."""
        out = eng.add_rsi(df, window=14)
        assert out["rsi_14"].iloc[:14].isna().all()

    def test_rsi_non_nan_after_warmup(self, eng, df):
        """After warmup, RSI should be fully populated."""
        out = eng.add_rsi(df, window=14)
        assert out["rsi_14"].iloc[14:].notna().all()

    def test_rsi_custom_window(self, eng, df):
        out = eng.add_rsi(df, window=7)
        assert "rsi_7" in out.columns

    def test_rsi_wilder_vs_simple_ema_different(self, eng, df):
        """
        Wilder's alpha=1/N gives different results from standard span=N EMA.
        This tests that we're using the correct Wilder smoothing.
        Wilder's EMA has a longer memory → smoother, slower RSI.
        """
        # Compute RSI with Wilder's method (our implementation)
        out = eng.add_rsi(df, window=14)
        rsi_wilder = out["rsi_14"].dropna()

        # Compute RSI with standard EMA span=14 (incorrect but common)
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain_simple = gain.ewm(span=14, adjust=False, min_periods=14).mean()
        avg_loss_simple = loss.ewm(span=14, adjust=False, min_periods=14).mean()
        rs_simple = avg_gain_simple / avg_loss_simple.replace(0, np.nan)
        rsi_simple = (100 - 100 / (1 + rs_simple)).dropna()

        # They should differ — if they're identical, the implementation is wrong
        common_idx = rsi_wilder.index.intersection(rsi_simple.index)
        assert not np.allclose(
            rsi_wilder[common_idx].values,
            rsi_simple[common_idx].values,
            rtol=1e-3,
        ), "RSI should differ between Wilder's alpha=1/N and standard span=N EMA"


# ======================================================================
# 3. ATR
# ======================================================================

class TestATR:

    def test_tr_gte_high_minus_low(self, eng, df):
        """True Range always >= high - low (gap accounting can only increase it)."""
        out = eng.add_atr(df)
        intraday = (df["high"] - df["low"]).dropna()
        tr = out["tr"].dropna()
        common = intraday.index.intersection(tr.index)
        assert (tr[common] >= intraday[common] - 1e-10).all()

    def test_atr_positive(self, eng, df):
        """ATR must be positive for any series that moves."""
        out = eng.add_atr(df)
        assert (out["atr_14"].dropna() > 0).all()

    def test_atr_nan_at_start(self, eng, df):
        # EWM with min_periods=14 produces first value at index 13 (0-based)
        # so rows 0..12 are NaN (13 rows)
        out = eng.add_atr(df, window=14)
        assert out["atr_14"].iloc[:13].isna().all()
        assert pd.notna(out["atr_14"].iloc[13])

    def test_tr_accounts_for_gaps(self, eng, df):
        """
        Inject a large overnight gap and verify TR > high - low for that row.
        """
        df_gap = df.copy()
        idx = 50
        # Simulate a gap: previous close was 100, today opens at 130
        df_gap.iloc[idx, df_gap.columns.get_loc("close")] = 100.0
        df_gap.iloc[idx + 1, df_gap.columns.get_loc("high")] = 135.0
        df_gap.iloc[idx + 1, df_gap.columns.get_loc("low")]  = 128.0
        df_gap.iloc[idx + 1, df_gap.columns.get_loc("close")] = 132.0

        out = eng.add_atr(df_gap)
        tr_gap_row = out["tr"].iloc[idx + 1]
        hl_gap_row = df_gap["high"].iloc[idx + 1] - df_gap["low"].iloc[idx + 1]
        # TR should be |high - prev_close| = |135 - 100| = 35 > HL = 7
        assert tr_gap_row > hl_gap_row

    def test_atr_custom_window(self, eng, df):
        out = eng.add_atr(df, window=7)
        assert "atr_7" in out.columns

    def test_missing_ohlc_columns_raises(self, eng):
        df_bad = pd.DataFrame({"close": [1, 2, 3], "high": [2, 3, 4]})
        with pytest.raises(KeyError):
            eng.add_atr(df_bad)


# ======================================================================
# 4. Volume Features
# ======================================================================

class TestVolumeFeatures:

    def test_vol_ratio_non_negative(self, eng, df):
        out = eng.add_volume_features(df)
        assert (out["vol_ratio_20"].dropna() >= 0).all()

    def test_vol_ratio_mean_near_one(self, eng, df):
        """
        Over a long series, average vol_ratio should be close to 1.0
        (current vol / rolling avg ≈ 1 when averaged).
        """
        out = eng.add_volume_features(df, window=20)
        mean_ratio = out["vol_ratio_20"].dropna().mean()
        assert 0.8 < mean_ratio < 1.2, f"vol_ratio mean {mean_ratio:.3f} far from 1.0"

    def test_vwap_positive(self, eng, df):
        out = eng.add_volume_features(df)
        assert (out["vwap_20d"].dropna() > 0).all()

    def test_vwap_between_high_low(self, eng, df):
        """
        VWAP (rolling) should generally be in a reasonable price range.
        Not guaranteed to be within any single day's H/L, but shouldn't
        be wildly outside the price range.
        """
        out = eng.add_volume_features(df, window=20)
        vwap = out["vwap_20d"].dropna()
        close = df["close"]
        rolling_min = close.rolling(20).min().dropna()
        rolling_max = close.rolling(20).max().dropna()
        common = vwap.index.intersection(rolling_min.index)
        assert (vwap[common] >= rolling_min[common] * 0.95).all()
        assert (vwap[common] <= rolling_max[common] * 1.05).all()

    def test_zero_volume_handled(self, eng, df):
        """Zero volume rows should produce NaN, not division errors."""
        df_zero = df.copy()
        df_zero.iloc[50:55, df_zero.columns.get_loc("volume")] = 0
        out = eng.add_volume_features(df_zero, window=5)
        # Should not raise; some NaN is acceptable
        assert "vol_ratio_5" in out.columns


# ======================================================================
# 5. Bollinger Bands
# ======================================================================

class TestBollingerBands:

    def test_band_ordering(self, eng, df):
        """upper > middle > lower must hold everywhere non-NaN."""
        out = eng.add_bollinger_bands(df)
        mask = out["bb_upper"].notna()
        assert (out.loc[mask, "bb_upper"] > out.loc[mask, "bb_middle"]).all()
        assert (out.loc[mask, "bb_middle"] > out.loc[mask, "bb_lower"]).all()

    def test_width_positive(self, eng, df):
        out = eng.add_bollinger_bands(df)
        assert (out["bb_width"].dropna() > 0).all()

    def test_pct_range(self, eng, df):
        """
        bb_pct is usually in [0, 1], but CAN exceed [0, 1] during breakouts.
        Test that the typical (median) value is within [0, 1].
        """
        out = eng.add_bollinger_bands(df)
        median_pct = out["bb_pct"].dropna().median()
        assert 0 < median_pct < 1, f"Median bb_pct {median_pct:.3f} outside [0,1]"

    def test_width_spikes_on_high_vol(self, eng):
        """High-vol series should produce wider bands than low-vol series."""
        out_low  = eng.add_bollinger_bands(make_ohlcv(vol=0.005, seed=1))
        out_high = eng.add_bollinger_bands(make_ohlcv(vol=0.05, seed=2))
        assert out_high["bb_width"].dropna().mean() > out_low["bb_width"].dropna().mean()

    def test_constant_price_zero_width(self, eng):
        """Constant price → zero std → zero width."""
        closes = np.ones(100) * 100.0
        df_flat = _wrap_closes(closes, 100)
        out = eng.add_bollinger_bands(df_flat, window=20)
        # Width should be zero (or very close due to floating point)
        width = out["bb_width"].dropna()
        assert (width.abs() < 1e-10).all()

    def test_nan_at_start(self, eng, df):
        out = eng.add_bollinger_bands(df, window=20)
        # min_periods = max(2, 10) = 10; first 9 rows NaN
        assert out["bb_middle"].iloc[:9].isna().all()


# ======================================================================
# 6. MACD
# ======================================================================

class TestMACD:

    def test_macd_zero_when_emas_equal(self, eng):
        """If fast EMA == slow EMA, MACD line == 0."""
        # Constant price → both EMAs converge to same value
        closes = np.ones(200) * 100.0
        df_flat = _wrap_closes(closes, 200)
        out = eng.add_macd(df_flat)
        # After warmup, MACD line should be 0 (EMAs identical)
        macd_tail = out["macd_line"].iloc[50:].dropna()
        np.testing.assert_allclose(macd_tail, 0, atol=1e-8)

    def test_histogram_equals_line_minus_signal(self, eng, df):
        """histogram = macd_line - macd_signal exactly."""
        out = eng.add_macd(df)
        mask = out["macd_histogram"].notna()
        diff = (out.loc[mask, "macd_line"]
                - out.loc[mask, "macd_signal"]
                - out.loc[mask, "macd_histogram"])
        np.testing.assert_allclose(diff, 0, atol=1e-10)

    def test_signal_smoother_than_line(self, eng, df):
        """Signal line is an EMA of MACD line → lower variance."""
        out = eng.add_macd(df)
        mask = out["macd_signal"].notna()
        assert out.loc[mask, "macd_signal"].std() < out.loc[mask, "macd_line"].std()

    def test_crossover_detectable(self, eng, df):
        """Sign changes in histogram correspond to crossover events."""
        out = eng.add_macd(df)
        hist = out["macd_histogram"].dropna()
        sign_changes = (hist.shift(1) * hist < 0).sum()
        # A real price series should have at least a few crossovers
        assert sign_changes > 0

    def test_ema_columns_present(self, eng, df):
        out = eng.add_macd(df, fast=12, slow=26)
        assert "ema_12" in out.columns
        assert "ema_26" in out.columns


# ======================================================================
# 7. Price Z-Score
# ======================================================================

class TestPriceZScore:

    def test_nan_at_start(self, eng, df):
        out = eng.add_price_zscore(df, windows=[60])
        assert out["z_price_60d"].iloc[:29].isna().all()

    def test_mean_near_zero(self, eng, df):
        """
        Over a long stationary period, rolling z-score mean ≈ 0.
        This is a weak test — equity prices trend — but the mean should
        not be wildly far from 0.
        """
        out = eng.add_price_zscore(df, windows=[20])
        z = out["z_price_20d"].dropna()
        assert abs(z.mean()) < 2.0, f"z_price mean {z.mean():.3f} unexpectedly large"

    def test_spike_detection(self, eng, df):
        """Injecting a large price spike should produce a large z-score."""
        df_spike = df.copy()
        df_spike.iloc[150, df_spike.columns.get_loc("close")] *= 3.0  # 3x price
        out = eng.add_price_zscore(df_spike, windows=[20])
        z_at_spike = out["z_price_20d"].iloc[150]
        assert z_at_spike > 4.0, f"Expected z > 4 at spike, got {z_at_spike:.2f}"

    def test_multiple_windows(self, eng, df):
        out = eng.add_price_zscore(df, windows=[20, 60])
        assert "z_price_20d" in out.columns
        assert "z_price_60d" in out.columns

    def test_no_lookahead(self, eng, df):
        """
        Verify no lookahead bias: z-score at row T should not change when
        we append new data after row T.
        """
        out_short = eng.add_price_zscore(df.iloc[:200], windows=[20])
        out_full  = eng.add_price_zscore(df, windows=[20])
        z_short = out_short["z_price_20d"].iloc[100:190].values
        z_full  = out_full["z_price_20d"].iloc[100:190].values
        np.testing.assert_allclose(z_short, z_full, rtol=1e-6)


# ======================================================================
# 8. Integration Tests
# ======================================================================

class TestIntegration:

    def test_add_all_features_runs(self, eng, df):
        """Smoke test: add_all_features should not raise on clean data."""
        out = eng.add_all_features(df)
        assert out.shape[0] == df.shape[0]
        assert out.shape[1] > df.shape[1]

    def test_add_all_features_columns(self, eng, df):
        """All expected feature column families should be present."""
        out = eng.add_all_features(df)
        expected_prefixes = ["vol_", "rsi_", "atr_", "vol_ratio_",
                             "bb_", "macd_", "z_price_", "ema_", "vwap_"]
        for prefix in expected_prefixes:
            cols = [c for c in out.columns if c.startswith(prefix)]
            assert cols, f"No columns found with prefix '{prefix}'"

    def test_original_columns_preserved(self, eng, df):
        """Feature engineering must not modify original OHLCV columns."""
        out = eng.add_all_features(df)
        for col in ["open", "high", "low", "close", "volume", "returns"]:
            pd.testing.assert_series_equal(out[col], df[col])

    def test_feature_report_structure(self, eng, df):
        out = eng.add_all_features(df, ticker="TEST")
        report = eng.feature_report(out, ticker="TEST")
        assert report["ticker"] == "TEST"
        assert report["n_features"] > 0
        assert "missing_pct" in report
        assert "correlation" in report

    def test_nan_count_reasonable(self, eng, df):
        """
        Total NaN count should be bounded: only warmup and tail NaNs.
        Should be well under 50% of all cells.
        """
        out = eng.add_all_features(df)
        feature_cols = [c for c in out.columns
                        if c not in {"open", "high", "low", "close", "volume",
                                     "returns", "returns_norm", "returns_fwd_1",
                                     "returns_fwd_5"}]
        nan_pct = out[feature_cols].isnull().mean().mean()
        assert nan_pct < 0.30, f"Too many NaNs in features: {nan_pct:.1%}"

    def test_idempotent(self, eng, df):
        """Calling add_all_features twice should not duplicate columns."""
        out1 = eng.add_all_features(df)
        out2 = eng.add_all_features(df)
        assert list(out1.columns) == list(out2.columns)

    def test_short_series_does_not_crash(self, eng):
        """Very short series (30 rows) should not raise — just produce NaNs."""
        df_short = make_ohlcv(n=30)
        out = eng.add_all_features(df_short)
        assert out.shape[0] == 30

    def test_single_ticker_pipeline(self):
        """
        End-to-end: DataCleaner → FeatureEngineer on synthetic data.
        (Does not require network access.)
        """
        try:
            from data_cleaner import DataCleaner
        except ImportError:
            pytest.skip("DataCleaner not available in test path")

        # Build a minimal raw DataFrame that DataCleaner would receive
        n = 300
        closes = np.linspace(100, 150, n)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")

        raw = pd.DataFrame({
            "Open":   closes * 0.999,
            "High":   closes * 1.01,
            "Low":    closes * 0.99,
            "Close":  closes,
            "Volume": np.ones(n) * 1_000_000,
        }, index=idx)

        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw, ticker="SYNTHETIC")

        eng = FeatureEngineer()
        featured = eng.add_all_features(cleaned, ticker="SYNTHETIC")

        assert featured.shape[0] == cleaned.shape[0]
        assert "rsi_14" in featured.columns
        assert "vol_21d" in featured.columns