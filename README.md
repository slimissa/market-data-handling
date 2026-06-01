# Market Data Handling — QuantOS Phases 1–4

**Fetch · Clean · Engineer · Signal · Backtest**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, generates 7 families of predictive features, produces tradeable trading signals with explicit entry/exit rules, and backtests every signal on every ticker with a full performance metrics suite.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and a complete chain from raw API data to ranked backtest results with honest performance attribution.

---

## Project structure

```
Market_data_handling/
├── config.yaml              # Runtime configuration (all phases)
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # 6-stage orchestrator + CLI
│   ├── data_fetcher.py      # yfinance API + CSV cache layer
│   ├── data_cleaner.py      # 5-step cleaning & normalisation engine
│   ├── feature_engineering.py  # 7 feature families
│   ├── signal_generator.py     # 6 signal families + ensemble
│   └── backtester.py           # Vectorised + event-driven engines, full metrics
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   ├── processed/           # Cleaned CSVs + JSON quality reports
│   └── results/             # Per-ticker backtest CSVs + cross-watchlist comparison
└── tests/
    ├── test_pipeline.py     # 16 unit tests (Phase 1)
    ├── test_features.py     # 49 unit tests (Phase 2)
    ├── test_signals.py      # 58 unit tests (Phase 3)
    └── test_backtester.py   # 48 unit tests (Phase 4)
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

This fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, generates signals, backtests every signal, and writes outputs to `data/processed/` and `data/results/`.

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

**171 tests. All passing. Under 6 seconds.**

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Fetch | `data_fetcher.py` | yfinance API or local CSV cache |
| 2. Clean | `data_cleaner.py` | Timestamps, alignment, missing data, returns |
| 3. Engineer | `feature_engineering.py` | 7 feature families (24 columns) |
| 4. Signal | `signal_generator.py` | 6 signal families + ensemble (10 columns) |
| 5. Backtest | `backtester.py` | Per-signal comparison + cross-ticker ranking |
| 6. Persist | `pipeline.py` | CSV + 3 JSON reports + backtest CSVs per ticker |

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

## Phase 4 — Backtesting

| Component | Description |
|-----------|-------------|
| `VectorisedBacktester` | Fast pandas/numpy engine — ideal for research and parameter sweeps |
| `EventDrivenBacktester` | Bar-by-bar simulation with explicit order lifecycle |
| `TransactionCostModel` | Almgren-Chriss square-root market impact + spread + commission |
| `PerformanceEngine` | Full metrics suite: Sharpe, Sortino, Calmar, max DD, VaR, CVaR, alpha, beta |

**Metrics computed per signal:**
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown (magnitude and duration)
- Hit rate, profit factor, avg win/loss, avg holding days
- VaR 95%, CVaR 95% (Expected Shortfall)
- Annualised turnover
- CAPM alpha, beta, R² (when benchmark provided)

**Output per ticker:**
- `{TICKER}_backtest.csv` — All signals ranked by Sharpe ratio
- `watchlist_comparison.csv` — Cross-ticker ensemble comparison

---

### Output per ticker (full)

- `{TICKER}_processed.csv` — Full DataFrame (OHLCV + returns + features + signals, 43 columns)
- `{TICKER}_data_report.json` — Data quality metrics (skew, kurtosis, missing %)
- `{TICKER}_feature_report.json` — Feature quality metrics (correlation matrix, NaN %, statistics)
- `{TICKER}_signal_report.json` — Signal quality metrics (turnover, long/short/flat %, correlation)
- `{TICKER}_backtest.csv` — Backtest results (all signals ranked by Sharpe)
- `watchlist_comparison.csv` — Cross-ticker ensemble comparison

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
  zscore:
    entry_threshold: 2.0
    exit_threshold: 0.0
    window: 60
  ensemble:
    method: "majority_vote"

# Phase 4 — Backtesting
backtest:
  initial_capital: 100_000
  position_sizing: "fixed_notional"
  target_notional: 100_000
  cost_model: "liquid_equity"

watchlist:
  - AAPL
  - GOOGL
  - MSFT
  - SPY
  - "^GSPC"
```

---

## Design principles

- **Lookahead-bias-free.** All signals, features, normalisations, and backtest fills use only past observations. Verified by dedicated tests at every phase.
- **Explicit entry/exit rules.** Every signal has documented entry and exit conditions. No ambiguous "buy when RSI is low."
- **Stateful signal tracking.** Positions persist across bars with min/max holding period enforcement — realistic trade simulation.
- **Dual backtester architecture.** Vectorised for fast research, event-driven for realistic pre-deployment validation.
- **Realistic transaction costs.** Almgren-Chriss square-root market impact model + spread + commission.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Pure transformations.** Every method is a pure function: DataFrame in, DataFrame out, no side effects.
- **Separation of concerns.** Fetching, cleaning, feature engineering, signal generation, backtesting, and reporting are independent, testable modules.
- **Honest about missing data.** NaN cells beyond `max_gap` remain NaN — the pipeline refuses to fabricate data.
- **Wilder's smoothing (RSI/ATR).** Industry-standard implementation matches Bloomberg/TradingView values.

---

## Test suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_pipeline.py` | 16 | Data cleaning: timestamps, alignment, gaps, returns, edge cases |
| `test_features.py` | 49 | Feature engineering: mathematical properties, edge cases, NaN structure |
| `test_signals.py` | 58 | Signal generation: values, entry/exit, strength, lookahead, ensemble |
| `test_backtester.py` | 48 | Backtesting: costs, metrics, engines, trades, lookahead, consistency |
| **Total** | **171** | **All passing in < 6s** |

Key tests include:
- RSI bounds [0, 100] and Wilder vs. simple EMA differentiation
- ATR gap accounting (TR > high-low during overnight gaps)
- Almgren-Chriss square-root slippage sublinearity
- Lookahead bias prevention — per-signal, per-engine, and integration
- Log return additivity property
- CAPM beta ≈ 1 for identical strategy/benchmark, ≈ 0 for uncorrelated
- Both backtester engines agree on return direction and Sharpe within tolerance
- Trade P&L correctness across all four quadrants (long/short × win/loss)
- End-to-end DataCleaner → FeatureEngineer → SignalGenerator → Backtester integration

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Data ingestion, cleaning, frequency alignment, return normalisation |
| **Phase 2** | ✅ Complete | Feature engineering — 7 families, 24 columns, 49 tests |
| **Phase 3** | ✅ Complete | Signal generation — 6 families, ensemble, 58 tests |
| **Phase 4** | ✅ Complete | Backtesting — dual-engine, full metrics, cross-ticker comparison, 48 tests |
| **Phase 5** | 🔜 Next | Factor attribution — CAPM, Fama-French, regime analysis |
| **Phase 6** | Planned | Regime detection — trending vs range-bound market classification |
| **Phase 7** | Planned | Portfolio layer — multi-asset simultaneous backtest, risk management |
| **Phase 8** | Planned | Research report — automated tear sheet PDF/DOCX from backtest results |
| **Phase 9** | Planned | QuantOS integration — CLI commands via `lass_sh` |

---

## Author

**Slim Issa** — QuantOS Project  
June 2026
