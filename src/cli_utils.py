"""
cli_utils.py — CLI utility functions for the QuantOS pipeline.
QuantOS Market Data Pipeline

Covers:
    - period_to_dates(): convert --period flag to (start, end) strings
    - resolve_dates(): merge --period with explicit --start/--end
    - OutputWriter: format-agnostic save layer (csv, parquet, excel)
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Period → date resolution
# ======================================================================

_VALID_PERIODS = {"1y", "2y", "3y", "5y", "10y", "ytd", "max"}


def period_to_dates(period: str) -> Tuple[str, str]:
    """
    Convert a period string to (start_date, end_date) ISO strings.

    Args:
        period: One of '1y', '2y', '3y', '5y', '10y', 'ytd', 'max'.

    Returns:
        (start_date, end_date) as 'YYYY-MM-DD' strings.
        end_date is always today.

    Notes on 'max':
        Returns 1970-01-01 as the start date — yfinance will return
        whatever is available, which for most equities is ~20 years.
        For some indices (e.g. ^GSPC) it goes back to 1927.

    Examples:
        period_to_dates('5y')    → ('2020-01-07', '2025-01-07')
        period_to_dates('ytd')   → ('2025-01-01', '2025-01-07')
        period_to_dates('max')   → ('1970-01-01', '2025-01-07')
    """
    period = period.lower().strip()
    today = date.today()
    end = today.isoformat()

    if period == "ytd":
        start = date(today.year, 1, 1).isoformat()
    elif period == "max":
        start = "1970-01-01"
    elif period.endswith("y"):
        try:
            years = int(period[:-1])
        except ValueError:
            raise ValueError(
                f"Invalid period '{period}'. "
                f"Use a number followed by 'y' (e.g. '5y'), "
                f"'ytd', or 'max'."
            )
        start = (today - timedelta(days=years * 365)).isoformat()
    else:
        raise ValueError(
            f"Unknown period '{period}'. Valid options: "
            + ", ".join(sorted(_VALID_PERIODS))
        )

    logger.info(f"Period '{period}' resolved to {start} → {end}")
    return start, end


def resolve_dates(
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    default_start: str = "2020-01-01",
    default_end: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve date range from a combination of --period, --start, --end.

    Priority:
        1. Explicit --start/--end always win over --period.
        2. If only --period is given, convert it to dates.
        3. If nothing is given, use defaults.

    Args:
        period:         Period string or None.
        start:          Explicit start date or None.
        end:            Explicit end date or None.
        default_start:  Fallback start if nothing else is provided.
        default_end:    Fallback end. Defaults to today if None.

    Returns:
        (start_date, end_date) as ISO strings.
    """
    if default_end is None:
        default_end = date.today().isoformat()

    if period is not None:
        period_start, period_end = period_to_dates(period)
        # Explicit dates override period where provided
        final_start = start if start is not None else period_start
        final_end   = end   if end   is not None else period_end
    else:
        final_start = start if start is not None else default_start
        final_end   = end   if end   is not None else default_end

    return final_start, final_end


# ======================================================================
# Output writer
# ======================================================================

class OutputWriter:
    """
    Format-agnostic output writer for pipeline results.

    Supports:
        csv     — standard CSV (default, always works)
        parquet — columnar binary format (fast, compact; requires pyarrow)
        excel   — single .xlsx workbook with one sheet per result type
                  (requires openpyxl)

    Usage:
        writer = OutputWriter(fmt="parquet", output_dir="./data/results")
        writer.write_dataframe(df, "AAPL_backtest")
        writer.write_excel_workbook({"backtest": bt_df, "gating": gc_df}, "AAPL")
    """

    SUPPORTED_FORMATS = {"csv", "parquet", "excel"}

    def __init__(self, fmt: str = "csv", output_dir: str = "./data/results"):
        fmt = fmt.lower().strip()
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unknown format '{fmt}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )
        self.fmt = fmt
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._excel_buffers: dict = {}   # ticker → {sheet_name: DataFrame}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def write_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        ticker: Optional[str] = None,
    ) -> Path:
        """
        Write a single DataFrame to the output directory.

        Args:
            df:     DataFrame to write.
            name:   Base filename without extension (e.g. 'AAPL_backtest').
            ticker: If provided and format is 'excel', buffers this sheet
                   for later workbook assembly via flush_excel().

        Returns:
            Path to the written file (or None for excel, which buffers).
        """
        if self.fmt == "csv":
            path = self.output_dir / f"{name}.csv"
            df.to_csv(path)
            return path

        elif self.fmt == "parquet":
            path = self.output_dir / f"{name}.parquet"
            df.to_parquet(path, engine="pyarrow")
            return path

        elif self.fmt == "excel":
            # Buffer for workbook — actual writing happens in flush_excel()
            if ticker not in self._excel_buffers:
                self._excel_buffers[ticker] = {}
            # Strip ticker prefix from sheet name to keep sheets short
            sheet = name.replace(f"{ticker}_", "") if ticker else name
            self._excel_buffers[ticker][sheet] = df
            return None

    def flush_excel(self, ticker: Optional[str] = None) -> Optional[Path]:
        """
        Write buffered DataFrames to a single Excel workbook.

        Args:
            ticker: Write only this ticker's workbook. If None, writes all.

        Returns:
            Path to the workbook (or None if nothing was buffered).
        """
        if self.fmt != "excel":
            return None

        tickers = [ticker] if ticker else list(self._excel_buffers.keys())

        for t in tickers:
            sheets = self._excel_buffers.get(t, {})
            if not sheets:
                continue

            path = self.output_dir / f"{t}_results.xlsx"
            try:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    for sheet_name, df in sheets.items():
                        # Excel sheet names max 31 chars
                        safe_name = sheet_name[:31]
                        df.reset_index().to_excel(
                            writer, sheet_name=safe_name, index=False
                        )
                logger.info(
                    f"Excel workbook written: {path.name} "
                    f"({len(sheets)} sheets)"
                )
            except ImportError:
                raise ImportError(
                    "openpyxl is required for --format excel. "
                    "Install with: pip install openpyxl"
                )

        return path if tickers else None

    def write_json(self, data: dict, name: str) -> Path:
        """Write a dict to JSON (format-independent — always writes JSON)."""
        import json
        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def extension(self) -> str:
        """Return the file extension for the current format."""
        return {"csv": ".csv", "parquet": ".parquet", "excel": ".xlsx"}[self.fmt]