# Market Data Handling — QuantOS Phases 1–6

**Fetch · Clean · Engineer · Signal · Backtest · Attribute · Filter**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, generates 7 families of predictive features, produces tradeable trading signals with explicit entry/exit rules, backtests every signal on every ticker with a full performance metrics suite, attributes returns to market factors (CAPM/FF3/Carhart4), and applies regime-gated signal filtering with a regime-adaptive ensemble.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and a complete chain from raw API data to regime-aware backtested signals with honest factor attribution.

---

## Project structure

```
Market_data_handling/
├── config.yaml              # Runtime configuration (all phases)
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # 8-stage orchestrator + CLI
│   ├── data_fetcher.py      # yfinance API + CSV cache layer
│   ├── data_cleaner.py      # 5-step cleaning & normalisation engine
│   ├── feature_engineering.py  # 7 feature families
│   ├── signal_generator.py     # 6 signal families + ensemble
│   ├── backtester.py           # Vectorised + event-driven engines, full metrics
│   ├── factor_model.py         # CAPM / FF3 / Carhart4 attribution
│   └── regime_filter.py        # Regime-gated signals + adaptive ensemble
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   ├── processed/           # Cleaned CSVs + JSON quality reports
│   └── results/             # Backtest CSVs + factor reports + regime reports
└── tests/
    ├── test_pipeline.py     # 16 unit tests (Phase 1)
    ├── test_features.py     # 49 unit tests (Phase 2)
    ├── test_signals.py      # 58 unit tests (Phase 3)
    ├── test_backtester.py   # 48 unit tests (Phase 4)
    ├── test_factor_model.py # 37 unit tests (Phase 5)
    └── test_regime_filter.py # 55 unit tests (Phase 6)
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

This fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, generates signals, applies regime filters, backtests every signal, runs factor attribution, and writes outputs to `data/processed/` and `data/results/`.

**CLI options:**

```bash
# Full pipeline (all 6 phases)
python src/pipeline.py --config config.yaml

# Fast mode (skip factor attribution — no network calls)
python src/pipeline.py --config config.yaml --no-factor

# Custom tickers and dates
python src/pipeline.py --symbols AAPL TSLA NVDA --start 2022-01-01 --end 2024-12-31

# Single ticker, fast
python src/pipeline.py --config config.yaml --symbols AAPL --no-factor
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

**268 tests. All passing. Under 13 seconds.**

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Fetch | `data_fetcher.py` | yfinance API or local CSV cache |
| 2. Clean | `data_cleaner.py` | Timestamps, alignment, missing data, returns |
| 3. Engineer | `feature_engineering.py` | 7 feature families (24 columns) |
| 4. Signal | `signal_generator.py` | 5 signals + ensemble (10 columns) |
| 5. Filter | `regime_filter.py` | Regime-gated signals + adaptive ensemble |
| 6. Backtest | `backtester.py` | Per-signal comparison + cross-ticker ranking |
| 7. Attribute | `factor_model.py` | CAPM / FF3 / Carhart4 + rolling alpha + regime analysis |
| 8. Persist | `pipeline.py` | CSV + 4 JSON reports + backtest/factor/regime CSVs |

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

---

## Phase 5 — Factor Attribution

| Component | Description |
|-----------|-------------|
| `FactorModel` | CAPM, Fama-French 3-Factor, Carhart 4-Factor regressions |
| `RegimeAnalyser` | Sharpe ratio per signal per market regime (crisis/trending/range-bound) |
| `OLSRegressor` | Newey-West HAC standard errors for financial time series |
| `FactorDataLoader` | Ken French data (via pandas_datareader) or ETF proxy fallback |

**Outputs per ticker:**
- `{TICKER}_factor_report.json` — Alpha, beta, t-stat, R², IC per signal × model
- `{TICKER}_regime_report.json` — Sharpe per signal per regime
- `{TICKER}_rolling_alpha.csv` — Rolling CAPM alpha time series

---

## Phase 6 — Regime Filtering

| Component | Description |
|-----------|-------------|
| `RegimeCondition` | Abstract gate: True = favourable regime, False = suppress |
| `VolPercentileCondition` | Gate: vol below Nth percentile → range-bound → mean-reversion |
| `TrendCondition` | Gate: \|rolling return\| < threshold → no strong trend |
| `MACDCondition` | Gate: MACD near zero → weak trend |
| `BBWidthCondition` | Gate: Bollinger squeeze → range-bound |
| `RSIRangeCondition` | Gate: RSI in neutral band → no extreme momentum |
| `CompositeCondition` | AND/OR combination of multiple gates |
| `RegimeFilter` | Wraps any signal, zeroes it in unfavourable regimes |
| `RegimeFilterPresets` | Pre-configured filters for RSI, z-score, MACD, BB signals |
| `RegimeFilteredEnsemble` | Regime-adaptive: MR pool in range-bound, trend pool in trending |

**New signal columns added:**
- `signal_rsi_vol_trend_gated` — RSI only in low-vol, weak-trend regime
- `signal_zscore_vol_bb_gated` — z-score only in low-vol, BB squeeze regime
- `signal_macd_trend_gated` — MACD only in elevated-vol (trending) regime
- `signal_bb_breakout_gated` — BB only above-median vol regime
- `signal_mr_pool` — Vote of gated mean-reversion signals
- `signal_trend_pool` — Vote of gated trend-following signals
- `signal_regime_adaptive` — Regime-switched ensemble

---

### Output per ticker (full)

- `{TICKER}_processed.csv` — Full DataFrame (OHLCV + returns + features + signals, 50+ columns)
- `{TICKER}_data_report.json` — Data quality metrics (skew, kurtosis, missing %)
- `{TICKER}_feature_report.json` — Feature quality metrics (correlation matrix, NaN %, statistics)
- `{TICKER}_signal_report.json` — Signal quality metrics (turnover, long/short/flat %, correlation)
- `{TICKER}_backtest.csv` — Backtest results (all signals ranked by Sharpe)
- `{TICKER}_factor_report.json` — Factor attribution (alpha/beta per signal × model)
- `{TICKER}_regime_report.json` — Sharpe per signal per regime
- `{TICKER}_rolling_alpha.csv` — Rolling CAPM alpha time series
- `watchlist_comparison.csv` — Cross-ticker ensemble comparison

---

## Configuration

All runtime parameters live in `config.yaml`. Defaults are defined in `pipeline.py`.

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
  volatility: {windows: [5, 21, 63]}
  rsi: {window: 14}
  atr: {window: 14}
  bollinger: {window: 20, num_std: 2.0}
  macd: {fast: 12, slow: 26, signal: 9}
  price_zscore: {windows: [20, 60]}

# Phase 3 — Signal generation
signals:
  rsi: {oversold: 30, overbought: 70, exit: 50, smoothing: 1}
  zscore: {entry_threshold: 2.0, exit_threshold: 0.0, window: 60}
  ensemble: {method: "majority_vote"}

# Phase 4 — Backtesting
backtest:
  initial_capital: 100_000
  position_sizing: "fixed_notional"
  target_notional: 100_000
  cost_model: "liquid_equity"

# Phase 5 — Factor attribution
factor:
  enabled: true
  rf_annual: 0.05
  rolling_window: 126
  factors: [MKT, SMB, HML, MOM]

# Phase 6 — Regime filtering
regime_filter:
  enabled: true
  vol_percentile: 30.0
  max_trend_annual: 0.10
  bb_percentile: 40.0

watchlist:
  - AAPL
  - GOOGL
  - MSFT
  - SPY
  - "^GSPC"
```

---

## Design principles

- **Lookahead-bias-free.** All signals, features, normalisations, backtest fills, and regime gates use only past observations. Verified by dedicated tests at every phase.
- **Explicit entry/exit rules.** Every signal has documented entry and exit conditions. No ambiguous "buy when RSI is low."
- **Stateful signal tracking.** Positions persist across bars with min/max holding period enforcement — realistic trade simulation.
- **Dual backtester architecture.** Vectorised for fast research, event-driven for realistic pre-deployment validation.
- **Realistic transaction costs.** Almgren-Chriss square-root market impact model + spread + commission.
- **Factor-aware evaluation.** CAPM/FF3/Carhart4 regression tells you if returns are alpha or beta.
- **Regime-adaptive deployment.** Signals are gated by market conditions — mean-reversion only in range-bound, trend-following only in trending.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Pure transformations.** Every method is a pure function: DataFrame in, DataFrame out, no side effects.
- **Separation of concerns.** All modules are independent, testable, and swappable.
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
| `test_factor_model.py` | 37 | Factor attribution: alpha/beta recovery, t-stats, rolling, regime analysis |
| `test_regime_filter.py` | 55 | Regime filtering: gate logic, filter invariants, presets, adaptive ensemble |
| **Total** | **268** | **All passing in < 13s** |

Key tests include:
- RSI bounds [0, 100] and Wilder vs. simple EMA differentiation
- ATR gap accounting (TR > high-low during overnight gaps)
- Almgren-Chriss square-root slippage sublinearity
- Lookahead bias prevention — per-signal, per-engine, per-gate, and integration
- Known alpha/beta recovery in factor regressions (ground-truth validation)
- Regime filter invariant: filtered ≤ original in absolute value (never amplifies)
- Composite AND/OR gate correctness against element-wise boolean operations
- Regime-adaptive ensemble: MR pool used in range-bound, trend pool in trending
- End-to-end DataCleaner → FeatureEngineer → SignalGenerator → RegimeFilter → Backtester

---

## Research Directions

### Option A: Improve the signals (highest ROI)

Use the vectorised backtester to iterate on signal parameters and logic.

- **Parameter sweeps** — Grid search RSI thresholds (20/80, 25/75, 30/70, 35/65)
- **Regime filter tuning** — Adjust vol percentiles and trend thresholds per ticker
- **Multi-timeframe confirmation** — Require RSI-14 AND RSI-5 to both signal
- **Benchmark-relative signals** — Cross-sectional z-scores across tickers

### Option B: Regime detection with HMM (higher sophistication)

Replace the binary vol-threshold heuristic with a Hidden Markov Model.

- **HMMRegimeDetector** — K=4 states, learns from returns/vol/MACD jointly
- **Continuous probabilities** — P(trending)=0.73 instead of trending=True
- **AdaptiveSignalSwitch** — Weight signals by regime probability, not binary gates
- **Transition matrix** — How likely is trend→range-bound? Drives signal selection

### Option C: ML signal (highest ceiling, highest risk)

Replace rule-based signals with a model that learns from 24 features.

- Train XGBoost/Random Forest on 2020-2021, test on 2022-2023
- Walk-forward validation to prevent overfitting
- Only safe because the backtester validates honestly

### Option D: Multi-asset portfolio construction

Trade all tickers simultaneously with proper risk management.

- Portfolio equity curve from simultaneous backtest
- Equal-weight, volatility-weighted, and risk-parity allocation
- Portfolio-level metrics: Sharpe, max DD, diversification ratio

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Data ingestion, cleaning, frequency alignment, return normalisation |
| **Phase 2** | ✅ Complete | Feature engineering — 7 families, 24 columns, 49 tests |
| **Phase 3** | ✅ Complete | Signal generation — 5 signals, ensemble, 58 tests |
| **Phase 4** | ✅ Complete | Backtesting — dual-engine, full metrics, cross-ticker, 48 tests |
| **Phase 5** | ✅ Complete | Factor attribution — CAPM/FF3/Carhart4, rolling alpha, regime analysis, 37 tests |
| **Phase 6a** | ✅ Complete | Regime filtering — binary gates, presets, adaptive ensemble, 55 tests |
| **Phase 6b** | 🔜 Next | Regime detection — HMM, continuous probabilities, adaptive switching |
| **Phase 7** | Planned | ML signal — XGBoost/Random Forest on 24 features, walk-forward validation |
| **Phase 8** | Planned | Portfolio layer — multi-asset simultaneous backtest, risk management, allocation |
| **Phase 9** | Planned | Research report — automated tear sheet PDF/DOCX from backtest results |
| **Phase 10** | Planned | QuantOS integration — CLI commands via `lass_sh` |

---

## Author

**Slim Issa** — QuantOS Project  
June 2026

