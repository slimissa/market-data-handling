# Market Data Handling — QuantOS Phase 1

**Fetch · Clean · Align · Normalise**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, and computes log returns with lookahead-free normalisation.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and reproducible processing.

---

## Project structure

```
Market_data_handling/
├── config.yaml              # Runtime configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # Orchestrator (fetch → clean → persist → report)
│   ├── data_fetcher.py      # yfinance API + CSV cache layer
│   ├── data_cleaner.py      # 5-step cleaning & normalisation engine
│   └── feature_engineering.py  # Phase 2 (upcoming)
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   └── processed/           # Cleaned CSVs + JSON quality reports
├── tests/
│   └── test_pipeline.py     # 16 unit tests (all passing)
└── notebooks/
    └── analysis.ipynb       # Exploratory analysis interface
```

---

## Quick start

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd Market_data_handling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python src/pipeline.py
```

This fetches AAPL, GOOGL, MSFT, SPY (and optionally ^GSPC) from the config watchlist, cleans the data, and writes outputs to `data/processed/`.

To specify custom symbols and dates:

```python
from src.pipeline import MarketDataPipeline

pipeline = MarketDataPipeline("config.yaml")
results = pipeline.run(
    symbols=["AAPL", "TSLA", "NVDA"],
    start_date="2021-01-01",
    end_date="2024-12-31",
)
```

### 3. Run the tests

```bash
python -m pytest tests/ -v
```

All 16 tests should pass in under a second.

---

## What the pipeline does

| Step | Method | Description |
|------|--------|-------------|
| 1 | `standardise_columns()` | Lowercase all column names, replace spaces with underscores |
| 2 | `clean_timestamps()` | Enforce tz-aware, deduplicated, chronologically sorted DatetimeIndex |
| 3 | `align_frequency()` | Reindex onto a complete calendar grid for multi-asset compatibility |
| 4 | `handle_missing_data()` | Forward-fill with configurable `max_gap` to prevent gap fabrication |
| 5 | `calculate_returns()` | Log returns, rolling z-scores (lookahead-free), forward return targets |

### Output columns

| Column | Description |
|--------|-------------|
| `open`, `high`, `low`, `close` | Adjusted OHLC prices |
| `volume` | Shares traded |
| `returns` | Log return: ln(P_t / P_{t-1}) |
| `returns_norm` | Rolling 252-day z-score (no lookahead bias) |
| `returns_fwd_1` | Next-period simple return (ML target label) |
| `returns_fwd_5` | 5-period forward simple return (ML target label) |

---

## Configuration

All runtime parameters live in `config.yaml`. Defaults are defined in `pipeline.py` and overridden by whatever the config file specifies.

```yaml
data_dir: "./data"
interval: "1d"
timezone: "US/Eastern"
target_freq: "D"
return_method: "log"
max_gap: 5
watchlist:
  - AAPL
  - GOOGL
  - MSFT
  - SPY
```

---

## Design principles

- **Lookahead-bias-free.** Rolling z-scores use only past observations. Forward returns are correctly NaN at the tail.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Pure transformations.** Every cleaning method is a pure function: DataFrame in, DataFrame out, no side effects.
- **Separation of concerns.** Fetching, cleaning, orchestrating, and reporting are independent, testable modules.
- **Honest about missing data.** NaN cells beyond `max_gap` remain NaN — the pipeline refuses to fabricate prices across extended closures.

---

## Roadmap

| Phase | Focus |
|-------|-------|
| **Phase 1** ✅ | Data ingestion, cleaning, frequency alignment, return normalisation |
| **Phase 2** | Feature engineering — realised volatility, RSI, price z-score, VWAP |
| **Phase 3** | Signal generation — cross-sectional momentum, pairs trading |
| **Phase 4** | Backtesting — event-driven engine, Sharpe/MaxDD/Calmar metrics |
| **Phase 5** | QuantOS integration — CLI commands via `lass_shell` |

---

## Author

**Slim Isaa** — QuantOS Project  
Mai 2026
```
