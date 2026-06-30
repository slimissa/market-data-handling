"""
tests/test_market_search.py — Regression tests for market_search.py
QuantOS Market Data Pipeline

Covers two bugs found while auditing the CLI flexibility phase:

1. _safe_fast_info_read() / validate_ticker():
   yfinance's FastInfo computes fields via lazy properties that hit the
   network on first access. When Yahoo returns bad/empty data those
   properties raise TypeError/KeyError directly — never AttributeError —
   so the original `getattr(info, attr, default)` calls did not catch
   them. A single bad field made every ticker (valid or not) report a
   confusing raw exception string instead of a clear status.

2. _rewrite_watchlist_block() / add_to_watchlist() / remove_from_watchlist():
   The original implementation did a full yaml.safe_load -> mutate ->
   yaml.dump round trip on the entire config file. PyYAML's loader does
   not preserve comments, and yaml.dump() always re-serializes keys in
   its own (alphabetical) order — so a single --add or --remove call
   silently destroyed every comment and reordered every top-level key
   in a hand-maintained config.yaml. The fix rewrites only the
   watchlist: block in the raw file text, leaving every other line
   byte-for-byte untouched.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_search import MarketSearch, _safe_fast_info_read


# ======================================================================
# Bug 1 — fast_info attribute access swallowing the wrong exception type
# ======================================================================

class TestSafeFastInfoRead:

    def test_returns_value_when_attribute_is_healthy(self):
        class GoodInfo:
            last_price = 123.45
            currency = "USD"

        assert _safe_fast_info_read(GoodInfo(), "last_price") == 123.45
        assert _safe_fast_info_read(GoodInfo(), "currency") == "USD"

    def test_returns_none_when_property_raises_typeerror(self):
        """This is the exact failure mode seen with a bad/empty Yahoo
        response: FastInfo's property getter raises TypeError, which a
        plain getattr(obj, attr, default) does NOT catch."""
        class RaisingInfo:
            @property
            def last_price(self):
                raise TypeError("argument of type 'NoneType' is not iterable")

        assert _safe_fast_info_read(RaisingInfo(), "last_price") is None

    def test_returns_none_when_property_raises_keyerror(self):
        class RaisingInfo:
            @property
            def currency(self):
                raise KeyError("currency")

        assert _safe_fast_info_read(RaisingInfo(), "currency") is None

    def test_returns_none_for_missing_attribute(self):
        class EmptyInfo:
            pass

        assert _safe_fast_info_read(EmptyInfo(), "nonexistent") is None

    def test_plain_getattr_would_not_catch_this(self):
        """Sanity check that the bug this fixes is real: a bare getattr
        with a default does NOT protect against a property that raises
        something other than AttributeError."""
        class RaisingInfo:
            @property
            def bar(self):
                raise TypeError("boom")

        with pytest.raises(TypeError):
            getattr(RaisingInfo(), "bar", "default")


class TestValidateTicker:

    @pytest.fixture
    def ms(self, tmp_path):
        return MarketSearch()

    def test_validate_ticker_handles_raising_fast_info_gracefully(
        self, ms, monkeypatch
    ):
        """validate_ticker must report valid=False with a clear message
        when fast_info's fields raise, not crash or surface a raw
        TypeError string as the only diagnostic."""
        import market_search as ms_module

        class RaisingFastInfo:
            @property
            def currency(self):
                raise TypeError("argument of type 'NoneType' is not iterable")

            @property
            def last_price(self):
                raise TypeError("argument of type 'NoneType' is not iterable")

        class FakeTicker:
            def __init__(self, symbol):
                pass

            @property
            def fast_info(self):
                return RaisingFastInfo()

        class FakeYF:
            Ticker = FakeTicker

        monkeypatch.setitem(sys.modules, "yfinance", FakeYF)

        result = ms.validate_ticker("AAPL")
        assert result["valid"] is False
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        # Must not just dump the raw exception text with no context.
        assert "TypeError" not in result["error"]

    def test_validate_ticker_reports_valid_for_healthy_response(
        self, ms, monkeypatch
    ):
        class GoodFastInfo:
            currency = "USD"
            last_price = 234.56

        class FakeTicker:
            def __init__(self, symbol):
                pass

            @property
            def fast_info(self):
                return GoodFastInfo()

        class FakeYF:
            Ticker = FakeTicker

        monkeypatch.setitem(sys.modules, "yfinance", FakeYF)

        result = ms.validate_ticker("AAPL")
        assert result["valid"] is True
        assert result["currency"] == "USD"
        assert result["last_price"] == 234.56


# ======================================================================
# Bug 2 — watchlist YAML rewrite destroying comments and key order
# ======================================================================

SAMPLE_CONFIG = (
    "# config.yaml — sample\r\n"
    "# A top-of-file comment that must survive.\r\n"
    "\r\n"
    "data_dir: \"./data\"\r\n"
    "interval: \"1d\"        # inline comment that must survive\r\n"
    "\r\n"
    "features:\r\n"
    "  rsi:\r\n"
    "    window: 14          # Wilder's standard, must survive\r\n"
    "\r\n"
    "# ── Watchlist ──────────────────────────────────────\r\n"
    "watchlist:\r\n"
    "  - AAPL\r\n"
    "  - GOOGL\r\n"
    "  - MSFT\r\n"
    "  - SPY\r\n"
    "  - \"^GSPC\"               # S&P 500 index\r\n"
    "\r\n"
    "# ── Next section ──────────────────────────────────\r\n"
    "regime_filter:\r\n"
    "  enabled: true\r\n"
)


class TestRewriteWatchlistBlock:

    @pytest.fixture
    def ms(self):
        return MarketSearch()

    @pytest.fixture
    def config_path(self, tmp_path):
        path = tmp_path / "config.yaml"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(SAMPLE_CONFIG)
        return path

    def test_add_preserves_every_comment_outside_watchlist_block(
        self, ms, config_path
    ):
        ms.add_to_watchlist(["TSLA", "NVDA"], config_path=str(config_path))
        with open(config_path, encoding="utf-8", newline="") as f:
            result = f.read()

        assert "# A top-of-file comment that must survive." in result
        assert "# inline comment that must survive" in result
        assert "# Wilder's standard, must survive" in result
        assert "# ── Next section" in result

    def test_add_preserves_crlf_line_endings(self, ms, config_path):
        ms.add_to_watchlist(["TSLA"], config_path=str(config_path))
        with open(config_path, "rb") as f:
            raw = f.read()
        assert b"\r\n" in raw
        # No bare LF that isn't part of a CRLF pair.
        bare_lf = any(
            raw[i] == 10 and (i == 0 or raw[i - 1] != 13)
            for i in range(len(raw))
        )
        assert not bare_lf

    def test_add_preserves_key_order_elsewhere_in_file(self, ms, config_path):
        """The bug this fixes: a full yaml.safe_load/yaml.dump round
        trip re-serializes ALL top-level keys in PyYAML's own order
        (alphabetical), not the order the user wrote them in. After the
        fix, only the watchlist: block's internal content should
        change — every other line's position is untouched."""
        with open(config_path, encoding="utf-8", newline="") as f:
            before_lines = f.read().splitlines()

        ms.add_to_watchlist(["TSLA"], config_path=str(config_path))

        with open(config_path, encoding="utf-8", newline="") as f:
            after_lines = f.read().splitlines()

        # Every line before the watchlist: key must be byte-identical
        # and in the same position.
        watchlist_idx = before_lines.index("watchlist:")
        assert before_lines[:watchlist_idx] == after_lines[:watchlist_idx]

    def test_add_actually_adds_tickers(self, ms, config_path):
        result = ms.add_to_watchlist(["TSLA", "NVDA"], config_path=str(config_path))
        assert result == ["AAPL", "GOOGL", "MSFT", "SPY", "^GSPC", "TSLA", "NVDA"]

        with open(config_path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed["watchlist"] == ["AAPL", "GOOGL", "MSFT", "SPY", "^GSPC", "TSLA", "NVDA"]

    def test_add_deduplicates_by_default(self, ms, config_path):
        result = ms.add_to_watchlist(["AAPL", "TSLA"], config_path=str(config_path))
        assert result.count("AAPL") == 1
        assert "TSLA" in result

    def test_remove_preserves_comments_and_removes_correct_tickers(
        self, ms, config_path
    ):
        result = ms.remove_from_watchlist(["GOOGL", "SPY"], config_path=str(config_path))
        assert result == ["AAPL", "MSFT", "^GSPC"]

        with open(config_path, encoding="utf-8", newline="") as f:
            content = f.read()
        assert "# A top-of-file comment that must survive." in content
        assert "GOOGL" not in content

    def test_remove_all_produces_valid_empty_list_yaml(self, ms, config_path):
        """Regression test for a bug found during manual testing: when
        the new watchlist is empty, the old code emitted BOTH
        'watchlist: []' AND a stray indented '  []' line beneath it,
        producing invalid YAML that failed to parse."""
        result = ms.remove_from_watchlist(
            ["AAPL", "GOOGL", "MSFT", "SPY", "^GSPC"], config_path=str(config_path)
        )
        assert result == []

        with open(config_path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed["watchlist"] == []

        with open(config_path, encoding="utf-8", newline="") as f:
            content = f.read()
        # Must be the single inline form, not "watchlist: []" followed
        # by a leftover "  []" item line.
        assert "watchlist: []" in content
        assert "  []\r\n" not in content.replace("watchlist: []\r\n", "")

    def test_add_to_previously_emptied_watchlist(self, ms, config_path):
        """Round-trip: empty it, then add back to it. Exercises the
        'watchlist: []' -> 'watchlist:\\n  - X' transition."""
        ms.remove_from_watchlist(
            ["AAPL", "GOOGL", "MSFT", "SPY", "^GSPC"], config_path=str(config_path)
        )
        result = ms.add_to_watchlist(["AAPL", "MSFT"], config_path=str(config_path))
        assert result == ["AAPL", "MSFT"]

        with open(config_path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed["watchlist"] == ["AAPL", "MSFT"]

    def test_ticker_requiring_quoting_round_trips_correctly(self, ms, config_path):
        """Tickers like ^VIX must be quoted in the written YAML or they
        parse incorrectly (or fail to parse). Verify the round trip
        through a real YAML parser, not just string presence."""
        ms.add_to_watchlist(["^VIX"], config_path=str(config_path))

        with open(config_path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert "^VIX" in parsed["watchlist"]

    def test_no_existing_watchlist_key_falls_back_to_append(self, ms, tmp_path):
        path = tmp_path / "no_watchlist.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write("data_dir: \"./data\"\ninterval: \"1d\"\n")

        result = ms.add_to_watchlist(["AAPL", "MSFT"], config_path=str(path))
        assert result == ["AAPL", "MSFT"]

        with open(path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed["watchlist"] == ["AAPL", "MSFT"]
        assert parsed["data_dir"] == "./data"

    def test_missing_config_file_raises_filenotfounderror(self, ms, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            ms.add_to_watchlist(["AAPL"], config_path=str(missing))