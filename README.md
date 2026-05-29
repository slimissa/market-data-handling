# Market Data Handling — QuantOS Phases 1 & 2

**Fetch · Clean · Engineer · Normalise**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, and generates 7 families of predictive features.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and reproducible feature engineering.

---

## Project structure

```
Market_data_handling/
├── config.yaml              # Runtime configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # 4-stage orchestrator + CLI
│   ├── data_fetcher.py      # yfinance API + CSV cache layer
│   ├── data_cleaner.py      # 5-step cleaning & normalisation engine
│   └── feature_engineering.py  # 7 feature families
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   └── processed/           # Cleaned CSVs + JSON quality reports
├── tests/
│   ├── test_pipeline.py     # 16 unit tests (Phase 1)
│   └── test_features.py     # 49 unit tests (Phase 2)
└── notebooks/
    └── analysis.ipynb       # Exploratory analysis interface
```

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/slimissa/market-data-handling.git
cd market-data-handling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python src/pipeline.py
```

This fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, and writes outputs to `data/processed/`.

**CLI options:**

```bash
python src/pipeline.py --symbols AAPL TSLA NVDA
python src/pipeline.py --start 2022-01-01 --end 2024-12-31
python src/pipeline.py --config custom_config.yaml
```

**As a library:**

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

**65 tests. All passing. Under 0.5 seconds.**

---

## Phase 1 — Data Cleaning

| Step | Method | Description |
|------|--------|-------------|
| 1 | `standardise_columns()` | Lowercase all column names, replace spaces with underscores |
| 2 | `clean_timestamps()` | Enforce tz-aware, deduplicated, chronologically sorted DatetimeIndex |
| 3 | `align_frequency()` | Reindex onto a complete calendar grid for multi-asset compatibility |
| 4 | `handle_missing_data()` | Forward-fill with configurable `max_gap` to prevent gap fabrication |
| 5 | `calculate_returns()` | Log returns, rolling z-scores (lookahead-free), forward return targets |

### Output columns (Phase 1)

| Column | Description |
|--------|-------------|
| `open`, `high`, `low`, `close` | Adjusted OHLC prices |
| `volume` | Shares traded |
| `returns` | Log return: ln(P_t / P_{t-1}) |
| `returns_norm` | Rolling 252-day z-score (no lookahead bias) |
| `returns_fwd_1` | Next-period simple return (ML target label) |
| `returns_fwd_5` | 5-period forward simple return (ML target label) |

---

## Phase 2 — Feature Engineering

| Family | Features | Signal Type |
|--------|----------|-------------|
| Realised Volatility | `vol_5d`, `vol_21d`, `vol_63d` (daily & annualised) | Risk measurement |
| RSI | `rsi_14` (Wilder's smoothing) | Momentum oscillator |
| ATR | `tr`, `atr_14` (gap-aware) | Stop-loss sizing |
| Volume | `vol_ratio_20`, `vwap_20d`, `vwap_dev` | Signal confirmation |
| Bollinger Bands | `bb_middle`, `bb_upper`, `bb_lower`, `bb_width`, `bb_pct` | Volatility envelopes |
| MACD | `ema_12`, `ema_26`, `macd_line`, `macd_signal`, `macd_histogram` | Trend following |
| Price Z-Score | `z_price_20d`, `z_price_60d` | Mean reversion |

**Total: 24 feature columns added (33 columns per ticker after processing).**

### Output per ticker

- `{TICKER}_processed.csv` — Full DataFrame (OHLCV + returns + features)
- `{TICKER}_data_report.json` — Data quality metrics (skew, kurtosis, missing %)
- `{TICKER}_feature_report.json` — Feature quality metrics (correlation matrix, NaN %, statistics)

---

## Configuration

All runtime parameters live in `config.yaml`. Defaults are defined in `pipeline.py` and overridden by the config file.

```yaml
# Phase 1 — Cleaning
data_dir: "./data"
interval: "1d"
timezone: "US/Eastern"
target_freq: "D"
return_method: "log"
max_gap: 5

# Phase 2 — Feature engineering
features:
  volatility:
    windows: [5, 21, 63]
  rsi:
    window: 14
  atr:
    window: 14
  volume:
    window: 20
  bollinger:
    window: 20
    num_std: 2.0
  macd:
    fast: 12
    slow: 26
    signal: 9
  price_zscore:
    windows: [20, 60]

watchlist:
  - AAPL
  - GOOGL
  - MSFT
  - SPY
  - "^GSPC"
```

---

## Design principles

- **Lookahead-bias-free.** Rolling z-scores and all features use only past observations. Forward returns are correctly NaN at the tail.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Pure transformations.** Every method is a pure function: DataFrame in, DataFrame out, no side effects.
- **Separation of concerns.** Fetching, cleaning, feature engineering, and reporting are independent, testable modules.
- **Honest about missing data.** NaN cells beyond `max_gap` remain NaN — the pipeline refuses to fabricate prices across extended closures.
- **Wilder's smoothing (RSI/ATR).** Industry-standard implementation matches Bloomberg/TradingView values.

---

## Test suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_pipeline.py` | 16 | Data cleaning: timestamps, alignment, gaps, returns, edge cases |
| `test_features.py` | 49 | Feature engineering: mathematical properties, edge cases, NaN structure, integration |
| **Total** | **65** | **All passing in < 0.5s** |

Key tests include:
- RSI bounds [0, 100] and Wilder vs. simple EMA differentiation
- ATR gap accounting (TR > high-low during overnight gaps)
- Lookahead bias prevention (z-score invariance to future data)
- Log return additivity property
- Bollinger Band ordering (upper > middle > lower)
- MACD histogram = line − signal identity
- End-to-end DataCleaner → FeatureEngineer integration

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Data ingestion, cleaning, frequency alignment, return normalisation |
| **Phase 2** | ✅ Complete | Feature engineering — 7 families, 24 columns, 49 tests |
| **Phase 3** | 🔜 Next | Signal generation — cross-sectional momentum, pairs trading |
| **Phase 4** | Planned | Backtesting — event-driven engine, Sharpe/MaxDD/Calmar metrics |
| **Phase 5** | Planned | QuantOS integration — CLI commands via `lass_sh` |

---

## Author

**Slim Issa** — QuantOS Project  
May 2026
