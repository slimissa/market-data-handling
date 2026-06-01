# Market Data Handling — QuantOS Phases 1–3

**Fetch · Clean · Engineer · Signal · Normalise**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, generates 7 families of predictive features, and produces tradeable trading signals with explicit entry/exit rules.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and a complete chain from raw API data to executable trading signals.

---

## Project structure

```
Market_data_handling/
├── config.yaml              # Runtime configuration (all phases)
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # 5-stage orchestrator + CLI
│   ├── data_fetcher.py      # yfinance API + CSV cache layer
│   ├── data_cleaner.py      # 5-step cleaning & normalisation engine
│   ├── feature_engineering.py  # 7 feature families
│   └── signal_generator.py     # 6 signal families + ensemble
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   └── processed/           # Cleaned CSVs + JSON quality reports
└── tests/
    ├── test_pipeline.py     # 16 unit tests (Phase 1)
    ├── test_features.py     # 49 unit tests (Phase 2)
    └── test_signals.py      # 58 unit tests (Phase 3)
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

This fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, generates signals, and writes outputs to `data/processed/`.

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

**125 tests. All passing. Under 3 seconds.**

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Fetch | `data_fetcher.py` | yfinance API or local CSV cache |
| 2. Clean | `data_cleaner.py` | Timestamps, alignment, missing data, returns |
| 3. Engineer | `feature_engineering.py` | 7 feature families (24 columns) |
| 4. Generate | `signal_generator.py` | 6 signal families + ensemble (10 columns) |
| 5. Persist | `pipeline.py` | CSV + 3 JSON quality reports per ticker |

---

## Phase 1 — Data Cleaning

| Step | Method | Description |
|------|--------|-------------|
| 1 | `standardise_columns()` | Lowercase all column names, replace spaces with underscores |
| 2 | `clean_timestamps()` | Enforce tz-aware, deduplicated, chronologically sorted DatetimeIndex |
| 3 | `align_frequency()` | Reindex onto a complete calendar grid for multi-asset compatibility |
| 4 | `handle_missing_data()` | Forward-fill with configurable `max_gap` to prevent gap fabrication |
| 5 | `calculate_returns()` | Log returns, rolling z-scores (lookahead-free), forward return targets |

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

---

## Phase 3 — Signal Generation

| Signal | Logic | Type |
|--------|-------|------|
| `signal_rsi` | RSI crosses below 30 → long (+1), above 70 → short (-1) | Mean-reversion |
| `signal_macd` | Histogram sign change → direction flip | Trend-following |
| `signal_zscore` | z < -2 → long, z > +2 → short | Mean-reversion |
| `signal_bb` | Band breach with squeeze filter (breakout or reversion mode) | Dual-mode |
| `position_scale` | Volatility percentile → continuous multiplier [floor, ceiling] | Risk overlay |
| `signal_ensemble` | Majority vote, weighted, or regime-switch combination | Meta-signal |

**Signal design features:**
- Explicit entry AND exit rules for every signal
- Signal strength ∈ [0, 1] alongside each binary signal
- Min/max holding period enforcement (anti-whipsaw)
- Signal smoothing for entry confirmation
- Turnover and correlation diagnostics in quality report

---

### Output per ticker

- `{TICKER}_processed.csv` — Full DataFrame (OHLCV + returns + features + signals, 43 columns)
- `{TICKER}_data_report.json` — Data quality metrics (skew, kurtosis, missing %)
- `{TICKER}_feature_report.json` — Feature quality metrics (correlation matrix, NaN %, statistics)
- `{TICKER}_signal_report.json` — Signal quality metrics (turnover, long/short/flat %, correlation)

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
  bollinger:
    window: 20
    num_std: 2.0
  macd:
    fast: 12
    slow: 26
    signal: 9
  price_zscore:
    windows: [20, 60]

# Phase 3 — Signal generation
signals:
  rsi:
    oversold: 30
    overbought: 70
    exit: 50
    smoothing: 1
  macd:
    require_zero_cross: false
  zscore:
    entry_threshold: 2.0
    exit_threshold: 0.0
    window: 60
  bollinger:
    squeeze_percentile: 20.0
    max_holding_bars: 10
  vol_scale:
    window: 21
    lookback: 252
    floor: 0.0
    ceiling: 2.0
  holding:
    min_bars: 2
    max_bars: 20
  ensemble:
    method: "majority_vote"

watchlist:
  - AAPL
  - GOOGL
  - MSFT
  - SPY
  - "^GSPC"
```

---

## Design principles

- **Lookahead-bias-free.** All signals, features, and normalisations use only past observations. Verified by dedicated tests.
- **Explicit entry/exit rules.** Every signal has documented entry and exit conditions. No ambiguous "buy when RSI is low."
- **Stateful signal tracking.** Positions persist across bars with min/max holding period enforcement — realistic trade simulation.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Pure transformations.** Every method is a pure function: DataFrame in, DataFrame out, no side effects.
- **Separation of concerns.** Fetching, cleaning, feature engineering, signal generation, and reporting are independent, testable modules.
- **Honest about missing data.** NaN cells beyond `max_gap` remain NaN — the pipeline refuses to fabricate data.
- **Wilder's smoothing (RSI/ATR).** Industry-standard implementation matches Bloomberg/TradingView values.

---

## Test suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_pipeline.py` | 16 | Data cleaning: timestamps, alignment, gaps, returns, edge cases |
| `test_features.py` | 49 | Feature engineering: mathematical properties, edge cases, NaN structure |
| `test_signals.py` | 58 | Signal generation: values, entry/exit, strength, lookahead, ensemble |
| **Total** | **125** | **All passing in < 3s** |

Key tests include:
- RSI bounds [0, 100] and Wilder vs. simple EMA differentiation
- ATR gap accounting (TR > high-low during overnight gaps)
- Lookahead bias prevention (signal invariance to future data — per-signal and integration)
- Log return additivity property
- Bollinger Band ordering (upper > middle > lower)
- MACD histogram = line − signal identity
- Ensemble majority vote: 2 long + 1 short → long
- Max holding period enforcement (no streak exceeds limit)
- Signal smoothing reduces whipsaws
- Vol scale bounded by [floor, ceiling]
- End-to-end DataCleaner → FeatureEngineer → SignalGenerator integration

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Data ingestion, cleaning, frequency alignment, return normalisation |
| **Phase 2** | ✅ Complete | Feature engineering — 7 families, 24 columns, 49 tests |
| **Phase 3** | ✅ Complete | Signal generation — 6 families, ensemble, 58 tests, 125 total |
| **Phase 4** | 🔜 Next | Backtesting — P&L simulation, Sharpe ratio, max drawdown, hit rate |
| **Phase 5** | Planned | QuantOS integration — CLI commands via `lass_sh` |

---

## Author

**Slim Issa** — QuantOS Project  
June 2026

---
