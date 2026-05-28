"""
tests/test_pipeline.py — Basic validation tests for the market data pipeline.
"""
import sys
from pathlib import Path
from unittest import result
import pandas as pd
import numpy as np
import pytest

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaner import DataCleaner


class TestDataCleaner:
    """Tests for the DataCleaner class — the core transformation engine."""

    @pytest.fixture
    def sample_df(self):
        """Create a minimal OHLCV DataFrame for testing."""
        dates = pd.date_range("2020-01-02", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [98 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                "volume": [1_000_000 + i * 1000 for i in range(10)],
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def cleaner(self):
        return DataCleaner()

    # ---- Step 1: Column standardisation ----

    def test_standardise_columns_lowercases(self, cleaner, sample_df):
        df = sample_df.copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        result = cleaner.standardise_columns(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_standardise_columns_replaces_spaces(self, cleaner, sample_df):
        df = sample_df.copy()
        # Give it realistic column names with spaces and capitals
        df.columns = ["Adj Close", "Some Col", "Another Col", "Last Col", "VOL"]
        result = cleaner.standardise_columns(df)
        assert "adj_close" in result.columns
        assert "some_col" in result.columns
        assert "another_col" in result.columns
        assert "last_col" in result.columns
        assert "vol" in result.columns

    # ---- Step 2: Timestamp cleaning ----

    def test_clean_timestamps_sorts_index(self, cleaner, sample_df):
        df = sample_df.copy()
        # Shuffle rows
        df = df.sample(frac=1.0)
        assert not df.index.is_monotonic_increasing
        result = cleaner.clean_timestamps(df, timezone="UTC")
        assert result.index.is_monotonic_increasing

    def test_clean_timestamps_drops_duplicates(self, cleaner, sample_df):
        df = pd.concat([sample_df, sample_df.iloc[:2]])
        assert df.index.duplicated().sum() == 2
        result = cleaner.clean_timestamps(df, timezone="UTC")
        assert result.index.duplicated().sum() == 0
        assert len(result) == len(sample_df)

    def test_clean_timestamps_localizes_naive(self, cleaner, sample_df):
        df = sample_df.copy()
        df.index = df.index.tz_localize(None)  # strip timezone
        result = cleaner.clean_timestamps(df, timezone="US/Eastern")
        assert result.index.tz is not None
        assert str(result.index.tz) == "US/Eastern"

    # ---- Step 3: Frequency alignment ----

    def test_align_frequency_expands_calendar(self, cleaner, sample_df):
        # 10 consecutive days includes a weekend gap
        result = cleaner.align_frequency(sample_df, target_freq="D", method="ffill")
        # Should now be a continuous daily grid
        expected_days = (sample_df.index.max() - sample_df.index.min()).days + 1
        assert len(result) == expected_days

    def test_align_frequency_preserves_tz(self, cleaner, sample_df):
        result = cleaner.align_frequency(sample_df, target_freq="D")
        assert result.index.tz == sample_df.index.tz

    # ---- Step 4: Missing data ----

    def test_handle_missing_data_ffill(self, cleaner, sample_df):
        df = sample_df.copy()
        df.iloc[5:7, df.columns.get_loc("close")] = np.nan
        result = cleaner.handle_missing_data(df, max_gap=3, fill_method="ffill")
        assert not result["close"].isnull().any()

    def test_handle_missing_data_respects_max_gap(self, cleaner, sample_df):
        df = sample_df.copy()
        df.iloc[3:8, df.columns.get_loc("close")] = np.nan  # 5 consecutive NaNs
        result = cleaner.handle_missing_data(df, max_gap=2, fill_method="ffill")
        # Some NaNs should remain beyond max_gap
        assert result["close"].isnull().any()

    # ---- Step 5: Returns ----

    def test_calculate_returns_adds_columns(self, cleaner, sample_df):
        result = cleaner.calculate_returns(sample_df, method="log")
        assert "returns" in result.columns
        assert "returns_norm" in result.columns
        assert "returns_fwd_1" in result.columns
        assert "returns_fwd_5" in result.columns

    def test_calculate_returns_log_additive(self, cleaner, sample_df):
        """Log returns over two periods should equal sum of single-period returns."""
        result = cleaner.calculate_returns(sample_df, method="log")
        two_period = np.log(
            sample_df["close"] / sample_df["close"].shift(2)
        ).dropna()
        summed = (
            result["returns"] + result["returns"].shift(1)
        ).dropna()
        # Align indices and compare
        common_idx = two_period.index.intersection(summed.index)
        pd.testing.assert_series_equal(
            two_period[common_idx].round(10),
            summed[common_idx].round(10),
            check_names=False,
        )

    def test_calculate_returns_norm_no_lookahead(self, cleaner, sample_df):
        """Z-score at time t must only use data up to time t."""
        result = cleaner.calculate_returns(sample_df, method="log")
        # returns_norm uses rolling window — first 30 are NaN, rest should exist
        assert result["returns_norm"].iloc[:29].isnull().all()
        # After min_periods, values should be finite
        valid = result["returns_norm"].dropna()
        assert not valid.isnull().any()

    def test_forward_returns_tail_is_nan(self, cleaner, sample_df):
        result = cleaner.calculate_returns(sample_df, method="log")
        # Last row has no forward price
        assert pd.isna(result["returns_fwd_1"].iloc[-1])
        # Last 5 rows have no 5-day forward price
        assert result["returns_fwd_5"].iloc[-5:].isnull().all()

    def test_calculate_returns_raises_on_bad_column(self, cleaner, sample_df):
        with pytest.raises(KeyError, match="nonexistent_col"):
            cleaner.calculate_returns(sample_df, price_col="nonexistent_col")

    # ---- Full pipeline integration ----

    def test_full_clean_pipeline_runs(self, cleaner, sample_df):
        result = cleaner.clean(sample_df, ticker="TEST")
        assert len(result) > 0
        assert "returns" in result.columns
        assert "returns_norm" in result.columns

    def test_quality_report_returns_dict(self, cleaner, sample_df):
        df = cleaner.clean(sample_df, ticker="TEST")
        report = cleaner.quality_report(df, ticker="TEST")
        assert isinstance(report, dict)
        assert report["ticker"] == "TEST"
        assert "returns_skew" in report
        assert "missing_pct" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])