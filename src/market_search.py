"""
market_search.py — Ticker search and watchlist management.
QuantOS Market Data Pipeline

Two modes:
    1. Local CSV search — searches a bundled S&P 500 constituent CSV by
       sector, industry, or name keyword. Fast, offline, reliable.
    2. yfinance validation — verifies a ticker exists on Yahoo Finance
       and returns basic metadata. Used by --dry-run and --add.

Design decision on live name search:
    yfinance's search API (yfinance.Search) is undocumented, changes
    without warning, has been rate-limited or broken repeatedly across
    versions, and returns inconsistent results. Building a "search EV
    makers" feature on top of it would create something that appears to
    work in demos and fails silently in production. This module does NOT
    implement live name search. If you need broad market discovery, use
    a proper financial data API (Polygon.io, Quandl, Bloomberg).
    What this module DOES provide:
        - Sector/industry search against the bundled S&P 500 CSV
        - Ticker validation via yfinance
        - Watchlist YAML management

Usage:
    from market_search import MarketSearch

    ms = MarketSearch()
    results = ms.search("semiconductor", field="sector")
    tickers = [r["symbol"] for r in results[:5]]
    ms.add_to_watchlist(tickers, "config.yaml")
"""

import csv
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _safe_fast_info_read(info, attr: str):
    """
    Read a single attribute off yfinance's FastInfo object, treating any
    exception as "unavailable" rather than letting it propagate.

    FastInfo's fields are lazy properties computed on first access. When
    Yahoo returns bad, empty, or unexpected data, those properties raise
    TypeError/KeyError/etc. directly — never AttributeError — so a plain
    `getattr(info, attr, default)` does not catch them; the exception
    passes straight through and a single bad field reads as a hard
    failure for the whole ticker instead of just "field unavailable".
    """
    try:
        return getattr(info, attr, None)
    except Exception:
        return None


# ======================================================================
# Bundled S&P 500 sector data
# ======================================================================

# Minimal bundled dataset: symbol, name, sector, industry.
# This is NOT the full S&P 500 — it's a representative sample of major
# liquid names across all sectors. Add your own rows or replace this
# with a downloaded constituents CSV (see load_csv() below).
_BUNDLED_CSV = """\
symbol,name,sector,industry
AAPL,Apple Inc.,Technology,Consumer Electronics
MSFT,Microsoft Corporation,Technology,Software-Application
NVDA,NVIDIA Corporation,Technology,Semiconductors
AMD,Advanced Micro Devices,Technology,Semiconductors
INTC,Intel Corporation,Technology,Semiconductors
TSM,Taiwan Semiconductor,Technology,Semiconductors
AVGO,Broadcom Inc.,Technology,Semiconductors
QCOM,QUALCOMM Incorporated,Technology,Semiconductors
AMAT,Applied Materials,Technology,Semiconductor Equipment
ASML,ASML Holding,Technology,Semiconductor Equipment
GOOGL,Alphabet Inc.,Technology,Internet Content
META,Meta Platforms,Technology,Internet Content
AMZN,Amazon.com Inc.,Consumer Cyclical,Internet Retail
NFLX,Netflix Inc.,Communication Services,Entertainment
SPOT,Spotify Technology,Communication Services,Entertainment
DIS,Walt Disney Co.,Communication Services,Entertainment
TSLA,Tesla Inc.,Consumer Cyclical,Auto Manufacturers
GM,General Motors,Consumer Cyclical,Auto Manufacturers
F,Ford Motor Company,Consumer Cyclical,Auto Manufacturers
RIVN,Rivian Automotive,Consumer Cyclical,Auto Manufacturers
NIO,NIO Inc.,Consumer Cyclical,Auto Manufacturers
JPM,JPMorgan Chase,Financial Services,Banks
BAC,Bank of America,Financial Services,Banks
WFC,Wells Fargo,Financial Services,Banks
GS,Goldman Sachs,Financial Services,Capital Markets
MS,Morgan Stanley,Financial Services,Capital Markets
BLK,BlackRock Inc.,Financial Services,Asset Management
V,Visa Inc.,Financial Services,Credit Services
MA,Mastercard Inc.,Financial Services,Credit Services
PYPL,PayPal Holdings,Financial Services,Credit Services
JNJ,Johnson & Johnson,Healthcare,Drug Manufacturers
PFE,Pfizer Inc.,Healthcare,Drug Manufacturers
MRK,Merck & Co.,Healthcare,Drug Manufacturers
ABBV,AbbVie Inc.,Healthcare,Drug Manufacturers
LLY,Eli Lilly and Company,Healthcare,Drug Manufacturers
UNH,UnitedHealth Group,Healthcare,Healthcare Plans
CVS,CVS Health,Healthcare,Healthcare Plans
XOM,Exxon Mobil,Energy,Oil & Gas Integrated
CVX,Chevron Corporation,Energy,Oil & Gas Integrated
COP,ConocoPhillips,Energy,Oil & Gas E&P
SLB,Schlumberger,Energy,Oil & Gas Equipment
NEE,NextEra Energy,Utilities,Utilities-Renewable
ENPH,Enphase Energy,Technology,Solar
SEDG,SolarEdge Technologies,Technology,Solar
BEP,Brookfield Renewable,Utilities,Utilities-Renewable
PLUG,Plug Power,Industrials,Electrical Equipment
PG,Procter & Gamble,Consumer Defensive,Household Products
KO,Coca-Cola Company,Consumer Defensive,Beverages
PEP,PepsiCo Inc.,Consumer Defensive,Beverages
WMT,Walmart Inc.,Consumer Defensive,Discount Stores
COST,Costco Wholesale,Consumer Defensive,Discount Stores
HD,Home Depot,Consumer Cyclical,Home Improvement Retail
LOW,Lowe's Companies,Consumer Cyclical,Home Improvement Retail
CAT,Caterpillar Inc.,Industrials,Farm & Heavy Construction
DE,Deere & Company,Industrials,Farm & Heavy Construction
BA,Boeing Company,Industrials,Aerospace & Defense
LMT,Lockheed Martin,Industrials,Aerospace & Defense
RTX,RTX Corporation,Industrials,Aerospace & Defense
GE,GE Aerospace,Industrials,Aerospace & Defense
SPY,SPDR S&P 500 ETF,ETF,Broad Market
QQQ,Invesco QQQ Trust,ETF,Technology
IWM,iShares Russell 2000,ETF,Small Cap
GLD,SPDR Gold Shares,ETF,Commodities
TLT,iShares 20+ Year Treasury,ETF,Bonds
BTC-USD,Bitcoin USD,Crypto,Digital Currency
ETH-USD,Ethereum USD,Crypto,Digital Currency
BNB-USD,Binance Coin USD,Crypto,Digital Currency
SOL-USD,Solana USD,Crypto,Digital Currency
EURUSD=X,EUR/USD,Forex,Currency Pair
GBPUSD=X,GBP/USD,Forex,Currency Pair
USDJPY=X,USD/JPY,Forex,Currency Pair
AUDUSD=X,AUD/USD,Forex,Currency Pair
CADUSD=X,CAD/USD,Forex,Currency Pair
"""


# ======================================================================
# MarketSearch
# ======================================================================

class MarketSearch:
    """
    Search for tickers and manage the watchlist.

    Args:
        csv_path: Path to a custom constituents CSV (columns: symbol,
                  name, sector, industry). If None, uses the bundled
                  dataset. Supply your own for broader coverage.
    """

    def __init__(self, csv_path: Optional[str] = None):
        self._data = self._load(csv_path)

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        field: str = "all",
        max_results: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Search the local constituents dataset.

        Args:
            query:       Case-insensitive substring search.
            field:       'sector', 'industry', 'name', 'symbol', or
                         'all' (searches all fields simultaneously).
            max_results: Cap on returned results.

        Returns:
            List of dicts with keys: symbol, name, sector, industry.

        Examples:
            ms.search("semiconductor")           → all semiconductor names
            ms.search("renewable", "industry")   → renewable energy names
            ms.search("AAPL", "symbol")           → exact AAPL row
        """
        query = query.lower().strip()
        results = []

        for row in self._data:
            if field == "all":
                haystack = " ".join(row.values()).lower()
            elif field in row:
                haystack = row[field].lower()
            else:
                raise ValueError(
                    f"Unknown field '{field}'. "
                    f"Valid: sector, industry, name, symbol, all"
                )

            if query in haystack:
                results.append(row)

        return results[:max_results]

    def search_by_sector(self, sector: str) -> List[Dict[str, str]]:
        """Convenience wrapper for sector search."""
        return self.search(sector, field="sector")

    def sectors(self) -> List[str]:
        """Return all unique sectors in the dataset, sorted."""
        return sorted({row["sector"] for row in self._data})

    def industries(self) -> List[str]:
        """Return all unique industries in the dataset, sorted."""
        return sorted({row["industry"] for row in self._data})

    # ------------------------------------------------------------------ #
    # Ticker validation                                                    #
    # ------------------------------------------------------------------ #

    def validate_ticker(self, symbol: str) -> Dict:
        """
        Check if a ticker exists on Yahoo Finance and return basic info.

        Returns dict with keys:
            symbol:    the ticker
            valid:     True if Yahoo Finance returned data
            name:      company name (if found)
            currency:  reported currency (if found)
            error:     error message (if invalid)
        """
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).fast_info
            # fast_info's fields are lazy properties that hit the network
            # on first access. When Yahoo returns bad/empty data, those
            # properties raise TypeError/KeyError directly rather than
            # AttributeError — getattr(info, attr, default) does NOT
            # catch that, so each field must be read defensively or a
            # transient API hiccup on one field looks identical to a
            # genuinely invalid ticker.
            currency = _safe_fast_info_read(info, "currency")
            last_price = _safe_fast_info_read(info, "last_price")
            if last_price is None and currency is None:
                return {
                    "symbol": symbol,
                    "valid": False,
                    "error": (
                        "No data returned — either an invalid symbol, or "
                        "a transient Yahoo Finance API issue. Retry to "
                        "rule out the latter."
                    ),
                }
            return {
                "symbol": symbol,
                "valid": True,
                "currency": currency or "unknown",
                "last_price": last_price,
            }
        except Exception as exc:
            return {"symbol": symbol, "valid": False, "error": str(exc)}

    def validate_tickers(
        self,
        symbols: List[str],
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        Validate multiple tickers. Used by --dry-run mode.

        Returns:
            Dict mapping symbol → validation result.
        """
        results = {}
        for sym in symbols:
            result = self.validate_ticker(sym)
            results[sym] = result
            if verbose:
                status = "✓" if result["valid"] else "✗"
                detail = (
                    f"  last={result.get('last_price', 'N/A')}  "
                    f"ccy={result.get('currency', 'N/A')}"
                    if result["valid"]
                    else f"  error: {result.get('error', '')}"
                )
                logger.info(f"  {status} {sym}{detail}")
        return results

    # ------------------------------------------------------------------ #
    # Watchlist management                                                 #
    # ------------------------------------------------------------------ #

    def add_to_watchlist(
        self,
        tickers: List[str],
        config_path: str = "config.yaml",
        deduplicate: bool = True,
    ) -> List[str]:
        """
        Append tickers to the watchlist in a YAML config file.

        Args:
            tickers:      List of ticker symbols to add.
            config_path:  Path to the config YAML file.
            deduplicate:  If True, skip tickers already in the watchlist.

        Returns:
            The updated watchlist (all tickers, including pre-existing).

        Note:
            This rewrites only the `watchlist:` block in place, preserving
            every other line in the file byte-for-byte — comments,
            section headers, key ordering, and blank lines all survive.
            (See _rewrite_watchlist_block for why a full
            yaml.safe_load → yaml.dump round trip is unsafe here.)
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        current = list(config.get("watchlist", []))
        current_set = {str(t).upper() for t in current}

        added = []
        for ticker in tickers:
            t = ticker.upper().strip()
            if deduplicate and t in current_set:
                logger.info(f"  {t} already in watchlist, skipping")
                continue
            current.append(t)
            current_set.add(t)
            added.append(t)
            logger.info(f"  + {t} added to watchlist")

        self._rewrite_watchlist_block(path, current)

        logger.info(f"Watchlist updated: {len(added)} added, {len(current)} total")
        return current

    def remove_from_watchlist(
        self,
        tickers: List[str],
        config_path: str = "config.yaml",
    ) -> List[str]:
        """
        Remove tickers from the watchlist in a YAML config file.

        Note:
            Same comment-preserving rewrite as add_to_watchlist — see
            its docstring.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        to_remove = {t.upper().strip() for t in tickers}
        current = list(config.get("watchlist", []))
        updated = [t for t in current if str(t).upper() not in to_remove]

        self._rewrite_watchlist_block(path, updated)

        removed_count = len(current) - len(updated)
        logger.info(f"Watchlist updated: {removed_count} removed, {len(updated)} remain")
        return updated

    def _rewrite_watchlist_block(
        self,
        path: Path,
        new_watchlist: List[str],
    ) -> None:
        """
        Replace only the `watchlist:` key's value in a YAML file's raw
        text, leaving every other line untouched.

        WHY THIS EXISTS:
            The previous implementation did
                config = yaml.safe_load(f)
                config["watchlist"] = new_list
                yaml.dump(config, f, ...)
            which round-trips the ENTIRE file through PyYAML. PyYAML's
            loader does not preserve comments (they are simply not part
            of the data model it builds), and yaml.dump() always
            re-serializes keys in its own order. On a hand-maintained
            config.yaml with section-header comments and per-line
            explanations, a single --add or --remove call silently
            strips every comment and alphabetizes every top-level key —
            a destructive, undocumented side effect of what looks like
            a small, additive operation.

            This method instead finds the existing `watchlist:` block by
            scanning raw lines, replaces just that block with freshly
            formatted entries, and writes everything else back exactly
            as it was — including line-ending style.

        Limitations (acceptable for this CLI's scope):
            - Assumes `watchlist:` is a top-level key (column 0), with
              its items as `  - VALUE` lines beneath it. This matches
              every config.yaml shipped with this project. A watchlist
              nested under another key would not be found and falls
              back to appending a new top-level block at the end of the
              file — still safe, just not in-place.
            - Inline comments on existing watchlist entries (e.g.
              `- "^GSPC"  # S&P 500 index`) are not preserved on
              individual tickers, since there is no reliable way to
              know which comment belongs to which ticker after the set
              has changed. Comments on every OTHER line in the file —
              including the lines immediately before and after the
              watchlist block — are fully preserved.
        """
        # newline="" disables universal-newline translation so we see the
        # file's actual line endings (CRLF vs LF) instead of having them
        # silently normalized to \n on read — which would make the
        # CRLF-detection below always fail and flip every config.yaml
        # from CRLF to LF on its first --add/--remove call. Using the
        # open() builtin directly rather than Path.read_text(newline=...)
        # since that parameter only exists from Python 3.13 onward.
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
        newline = "\r\n" if "\r\n" in raw else "\n"
        lines = raw.splitlines()

        # Find the watchlist: header line (top-level key, no leading space)
        start_idx = None
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if stripped == "watchlist:" or stripped.startswith("watchlist:"):
                # Must be a top-level key: no leading whitespace.
                if line[:1] not in (" ", "\t"):
                    start_idx = i
                    break

        if not new_watchlist:
            # Inline empty-list form is the whole replacement — no
            # separate indented item lines follow it.
            header_lines = ["watchlist: []"]
        else:
            header_lines = ["watchlist:"] + [
                f"  - {self._format_watchlist_item(t)}" for t in new_watchlist
            ]

        if start_idx is None:
            # No existing watchlist: key — append a new block at the end
            # rather than guessing where one should go.
            logger.warning(
                "No top-level 'watchlist:' key found in "
                f"{path} — appending a new watchlist block at the end "
                f"of the file instead of editing in place."
            )
            out_lines = lines + [""] + header_lines
        else:
            # Find the extent of the existing block: every following
            # line that is either blank or indented (a list item or its
            # continuation) belongs to this key. The block ends at the
            # first subsequent line that is non-blank and not indented.
            end_idx = start_idx + 1
            while end_idx < len(lines):
                line = lines[end_idx]
                if line.strip() == "":
                    # Blank lines inside the list are uncommon but
                    # treated as part of the block; trailing blank lines
                    # before the next section are handled by stopping at
                    # the next non-indented, non-blank line below.
                    end_idx += 1
                    continue
                if line[:1] in (" ", "\t"):
                    end_idx += 1
                    continue
                break

            # Trim back any blank lines we swallowed at the tail of the
            # block so we don't duplicate the spacing before whatever
            # comes next (e.g. a section-header comment).
            while end_idx > start_idx + 1 and lines[end_idx - 1].strip() == "":
                end_idx -= 1

            out_lines = lines[:start_idx] + header_lines + lines[end_idx:]

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(newline.join(out_lines) + newline)

    @staticmethod
    def _format_watchlist_item(ticker: str) -> str:
        """
        Format a single ticker for a YAML block-list entry, quoting it
        when required so the file stays valid YAML.

        Tickers like ^GSPC must be quoted — a bare leading `^` is not
        special in YAML, but tickers containing `:` , starting with
        punctuation YAML treats as indicators (`-`, `?`, `:`, `[`, `{`,
        `#`, `&`, `*`, `!`, `|`, `>`, `'`, `"`, `%`, `@`, `` ` ``), or
        matching YAML's boolean/null keywords would otherwise be parsed
        incorrectly or fail to parse at all.
        """
        ticker = str(ticker)
        needs_quoting = (
            ticker == ""
            or ticker[0] in "^?:-[]{}#&*!|>'\"%@`"
            or ticker.upper() in {"TRUE", "FALSE", "NULL", "YES", "NO", "ON", "OFF", "~"}
            or ":" in ticker
            or "#" in ticker
        )
        if needs_quoting:
            escaped = ticker.replace('"', '\\"')
            return f'"{escaped}"'
        return ticker

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _load(self, csv_path: Optional[str]) -> List[Dict[str, str]]:
        if csv_path is not None:
            with open(csv_path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        else:
            reader = csv.DictReader(io.StringIO(_BUNDLED_CSV))
            return list(reader)