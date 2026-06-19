# Market Data Handling — QuantOS Phases 1–6.b

**Fetch · Clean · Engineer · Signal · Backtest · Attribute · Filter · Detect**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, generates 7 families of predictive features, produces tradeable trading signals with explicit entry/exit rules, backtests every signal on every ticker with a full performance metrics suite and drawdown circuit breaker, attributes returns to market factors (CAPM/FF3/Carhart4) with per-regime Sharpe analysis, applies regime-gated signal filtering, and features a Hidden Markov Model regime detector with continuous probability-weighted adaptive signal switching.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates junior quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and a complete chain from raw API data to regime-aware, probability-weighted backtested signals with honest factor attribution and drawdown risk management.

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
│   ├── signal_generator.py     # 5 signal families + ensemble
│   ├── backtester.py           # Vectorised + event-driven engines, drawdown circuit breaker
│   ├── factor_model.py         # CAPM / FF3 / Carhart4 attribution + regime analysis
│   ├── regime_filter.py        # Binary regime-gated signals + adaptive ensemble
│   └── regime_detector.py      # Rule-based + HMM regime detection + adaptive switching
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   ├── processed/           # Cleaned CSVs + JSON quality reports
│   └── results/             # Backtest CSVs + factor reports + regime reports + rolling alpha
└── tests/
    ├── test_pipeline.py     # 16 unit tests (Phase 1)
    ├── test_features.py     # 49 unit tests (Phase 2)
    ├── test_signals.py      # 58 unit tests (Phase 3)
    ├── test_backtester.py   # 48 unit tests (Phase 4)
    ├── test_factor_model.py # 37 unit tests (Phase 5)
    ├── test_regime_filter.py # 55 unit tests (Phase 6a)
    └── test_regime_detector.py # 36 unit tests (Phase 6b)
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

This fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, generates signals, detects regimes, applies adaptive switching, backtests every signal, runs factor attribution with regime analysis, and writes outputs to `data/processed/` and `data/results/`.

**CLI options:**

```bash
# Full pipeline (all 7 phases)
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

**300+ tests. All passing. Under 18 seconds.**

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Fetch | `data_fetcher.py` | yfinance API or local CSV cache |
| 2. Clean | `data_cleaner.py` | Timestamps, alignment, missing data, returns |
| 3. Engineer | `feature_engineering.py` | 7 feature families (24 columns) |
| 4. Signal | `signal_generator.py` | 5 signals + ensemble (10 columns) |
| 4b. Detect | `regime_detector.py` | Rule-based or HMM regime classification + adaptive switching |
| 5. Backtest | `backtester.py` | Per-signal comparison + drawdown circuit breaker |
| 6. Attribute | `factor_model.py` | CAPM / FF3 / Carhart4 + rolling alpha + per-regime Sharpe analysis |
| 7. Persist | `pipeline.py` | CSV + 4 JSON reports + backtest/factor/regime CSVs |

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

---

## Phase 4 — Backtesting

| Component | Description |
|-----------|-------------|
| `VectorisedBacktester` | Fast pandas/numpy engine with drawdown circuit breaker |
| `EventDrivenBacktester` | Bar-by-bar simulation with explicit order lifecycle |
| `TransactionCostModel` | Almgren-Chriss square-root market impact + spread + commission |
| `PerformanceEngine` | Full metrics: Sharpe, Sortino, Calmar, max DD, VaR, CVaR, alpha, beta |

---

## Phase 5 — Factor Attribution & Regime Analysis

| Component | Description |
|-----------|-------------|
| `FactorModel` | CAPM, Fama-French 3-Factor, Carhart 4-Factor regressions |
| `RegimeAnalyser` | Sharpe ratio per signal per market regime (crisis/trending/range-bound) |
| `OLSRegressor` | Newey-West HAC standard errors for financial time series |

**Key finding from regime analysis (AAPL 2020-2023):**

| Regime | % of time | Best Signal | Sharpe |
|--------|-----------|-------------|--------|
| Trending Up | 48.3% | — | No signal profitable |
| Trending Down | 19.5% | MACD | +1.88 |
| Range-Bound | 9.8% | MACD | +2.34 |
| Crisis | 22.4% | RSI | +0.68 |

The dominant regime (trending-up, 48% of time) is hostile to all simple rule-based signals — explaining why none generate positive alpha on large-cap US equities in this period.

---

## Phase 6a — Regime Filtering (Binary Gates)

| Component | Description |
|-----------|-------------|
| `RegimeCondition` | Abstract gate: True = favourable regime, False = suppress |
| `VolPercentileCondition` | Gate: vol below Nth percentile → range-bound |
| `TrendCondition` | Gate: \|rolling return\| < threshold → no strong trend |
| `MACDCondition` | Gate: MACD near zero → weak trend |
| `BBWidthCondition` | Gate: Bollinger squeeze → range-bound |
| `RSIRangeCondition` | Gate: RSI in neutral band → no extreme momentum |
| `RegimeFilter` | Wraps any signal, zeroes it in unfavourable regimes |
| `RegimeFilteredEnsemble` | Regime-adaptive: MR pool vs trend pool |

---

## Phase 6b — Regime Detection (Probabilistic)

| Component | Description |
|-----------|-------------|
| `RuleBasedClassifier` | Multi-indicator deterministic classification (vol + trend + MACD) |
| `HMMRegimeDetector` | Gaussian Hidden Markov Model — learns K states from data |
| `AdaptiveSignalSwitch` | Continuous probability-weighted signal blending |
| `predict_proba_online()` | Forward-algorithm probabilities — no lookahead, backtest-safe |
| `RegimeDetectionResult` | Labels + probabilities + transition matrix + regime stats |

---

### Output per ticker (full)

- `{TICKER}_processed.csv` — Full DataFrame (50+ columns)
- `{TICKER}_data_report.json` — Data quality metrics
- `{TICKER}_feature_report.json` — Feature quality metrics
- `{TICKER}_signal_report.json` — Signal quality metrics
- `{TICKER}_backtest.csv` — Backtest results (all signals ranked by Sharpe)
- `{TICKER}_factor_report.json` — Factor attribution (alpha/beta per signal × model)
- `{TICKER}_regime_report.json` — **Sharpe per signal per regime** (key output)
- `{TICKER}_rolling_alpha.csv` — Rolling CAPM alpha time series
- `watchlist_comparison.csv` — Cross-ticker ensemble comparison

---

## Design principles

- **Lookahead-bias-free.** All signals, features, normalisations, backtest fills, regime gates, and HMM probabilities use only past observations. Verified by dedicated tests at every phase.
- **Drawdown circuit breaker.** Positions force-flat when equity drawdown exceeds configurable threshold (default 50%) — prevents catastrophic -89% scenarios.
- **Tightened regime gates.** Mean-reversion signals only active in low-vol, weak-trend regimes (P20 vol, <5%/yr trend). Trend-following signals only in elevated vol.
- **Probability-weighted adaptive switching.** `signal_adaptive` blends signals by regime confidence, not binary on/off — reducing whipsaw at regime boundaries.
- **Per-regime Sharpe analysis.** Every signal's performance is decomposed by market regime, revealing *when* it works — not just whether it works on average.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Separation of concerns.** All 8 modules are independent, testable, and swappable.
- **Wilder's smoothing (RSI/ATR).** Industry-standard implementation matches Bloomberg/TradingView values.

---

## Test suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_pipeline.py` | 16 | Data cleaning |
| `test_features.py` | 49 | Feature engineering |
| `test_signals.py` | 58 | Signal generation |
| `test_backtester.py` | 48 | Backtesting + circuit breaker |
| `test_factor_model.py` | 37 | Factor attribution + regime analysis |
| `test_regime_filter.py` | 55 | Binary regime gates |
| `test_regime_detector.py` | 36 | HMM + rule-based detection + adaptive switch |
| **Total** | **300+** | **All passing** |

---

## Research Directions

### Option A: Signal parameter optimisation
Grid search RSI/MACD/z-score thresholds. The vectorised backtester runs 100 combinations in seconds.

### Option B: ML signal (Phase 7)
XGBoost/Random Forest on 24 features with walk-forward validation. The regime analysis above tells you which features matter in which regimes.

### Option C: Multi-asset portfolio construction
Trade all tickers simultaneously with volatility-weighted allocation and risk management.

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ | Data ingestion, cleaning, alignment, returns |
| **Phase 2** | ✅ | Feature engineering — 7 families, 24 columns |
| **Phase 3** | ✅ | Signal generation — 5 signals + ensemble |
| **Phase 4** | ✅ | Backtesting — dual-engine, circuit breaker, full metrics |
| **Phase 5** | ✅ | Factor attribution + per-regime Sharpe analysis |
| **Phase 6a** | ✅ | Regime filtering — binary gates, adaptive ensemble |
| **Phase 6b** | ✅ | Regime detection — HMM, rule-based, probability-weighted switching |
| **Phase 7** | 🔜 | ML signal — XGBoost/Random Forest, walk-forward validation |
| **Phase 8** | Planned | Portfolio layer — multi-asset backtest, risk management |
| **Phase 9** | Planned | Research report — automated tear sheet PDF/DOCX |
| **Phase 10** | Planned | QuantOS integration — CLI commands via `lass_sh` |


---

## Author

**Slim Issa** — QuantOS Project  
June 2026

