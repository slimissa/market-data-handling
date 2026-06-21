# Market Data Handling — QuantOS Phases 1–7

**Fetch · Clean · Engineer · Signal · Backtest · Attribute · Filter · Detect · ML**

A production-quality market data pipeline for quantitative finance. Pulls historical OHLCV data from Yahoo Finance (with CSV caching), enforces timestamp hygiene, aligns frequencies across assets, handles missing data with configurable gap limits, computes log returns with lookahead-free normalisation, generates 7 families of predictive features, produces tradeable trading signals with explicit entry/exit rules, backtests every signal on every ticker with a full performance metrics suite and drawdown circuit breaker, attributes returns to market factors (CAPM/FF3/Carhart4) with per-regime Sharpe analysis, applies regime-gated signal filtering with per-signal drawdown circuit breakers, features a Hidden Markov Model regime detector with continuous probability-weighted adaptive signal switching, and trains a walk-forward validated XGBoost model on 24 engineered features with leak-proof temporal splits.

---

## Why this exists

In real quant workflows, 70-80% of time is spent on data infrastructure — not modelling. Most candidates jump straight to backtesting on dirty data. This pipeline demonstrates the data engineering discipline that differentiates quant candidates: structured datasets, timestamp integrity, lookahead-bias awareness, and a complete chain from raw API data to regime-aware, probability-weighted backtested signals with honest factor attribution, drawdown risk management, and ML validation.

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
│   ├── feature_engineering.py  # 7 feature families (24 columns)
│   ├── signal_generator.py     # 5 signal families + ensemble
│   ├── backtester.py           # Vectorised + event-driven engines, drawdown circuit breaker
│   ├── factor_model.py         # CAPM / FF3 / Carhart4 attribution + regime analysis
│   ├── regime_filter.py        # Binary regime-gated signals + per-signal drawdown breakers
│   ├── regime_detector.py      # Rule-based + HMM regime detection + adaptive switching
│   └── ml_signal.py            # Walk-forward ML signal with leak prevention
├── data/
│   ├── raw/                 # Cached downloads from Yahoo Finance
│   ├── processed/           # Cleaned CSVs + JSON quality reports
│   └── results/             # Backtest CSVs + factor reports + gating comparisons + ML reports
└── tests/
    ├── test_pipeline.py     # 16 unit tests (Phase 1)
    ├── test_features.py     # 49 unit tests (Phase 2)
    ├── test_signals.py      # 58 unit tests (Phase 3)
    ├── test_backtester.py   # 48 unit tests (Phase 4)
    ├── test_factor_model.py # 37 unit tests (Phase 5)
    ├── test_regime_filter.py # 55 unit tests (Phase 6a)
    ├── test_regime_detector.py # 36 unit tests (Phase 6b)
    └── test_ml_signal.py    # 42 unit tests (Phase 7)
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

### 2. Quickest possible run

```bash
python src/pipeline.py --symbols AAPL --no-factor
```

Runs the full pipeline on AAPL in ~30 seconds. No network calls needed after the first fetch.

### 3. Full pipeline

```bash
python src/pipeline.py --config config.yaml
```

Fetches AAPL, GOOGL, MSFT, SPY, and ^GSPC from the config watchlist, cleans, engineers features, generates signals, applies regime gates, detects regimes with adaptive switching, backtests every signal, runs factor attribution with regime analysis, and writes outputs to `data/processed/` and `data/results/`.

**CLI options:**

```bash
# Fast mode (skip factor attribution — no network calls)
python src/pipeline.py --config config.yaml --no-factor

# Custom tickers and dates
python src/pipeline.py --symbols AAPL TSLA NVDA --start 2022-01-01 --end 2024-12-31

# Enable ML signal (walk-forward XGBoost — slower but auditable)
python src/pipeline.py --symbols AAPL --no-factor
# Then set ml_signal.enabled: true in config.yaml
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

### 4. Run the tests

```bash
python -m pytest tests/ -v
```

**348 tests. All passing.**

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Fetch | `data_fetcher.py` | yfinance API or local CSV cache |
| 2. Clean | `data_cleaner.py` | Timestamps, alignment, missing data, returns |
| 3. Engineer | `feature_engineering.py` | 7 feature families (24 columns) |
| 4. Signal | `signal_generator.py` | 5 signals + ensemble (10 columns) |
| 4c. Gate | `regime_filter.py` | Per-signal regime gates + drawdown breakers |
| 4d. ML | `ml_signal.py` | Walk-forward XGBoost signal (opt-in) |
| 4b. Detect | `regime_detector.py` | Rule-based or HMM regime classification + adaptive switching |
| 5. Backtest | `backtester.py` | Per-signal comparison + drawdown circuit breaker |
| 6. Attribute | `factor_model.py` | CAPM / FF3 / Carhart4 + rolling alpha + per-regime Sharpe analysis |
| 7. Persist | `pipeline.py` | CSV + JSON reports + backtest/factor/gating/ML results |

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

---

## Phase 6a — Regime Filtering (Binary Gates)

| Component | Description |
|-----------|-------------|
| `RegimeCondition` | Abstract gate: True = favourable regime, False = suppress |
| `VolPercentileCondition` | Gate: vol below Nth percentile → range-bound |
| `TrendCondition` | Gate: \|rolling return\| < threshold → no strong trend |
| `TrendConfirmationCondition` | Gate: signal direction must agree with MACD sign |
| `MACDCondition` | Gate: MACD near zero → weak trend |
| `BBWidthCondition` | Gate: Bollinger squeeze → range-bound |
| `RSIRangeCondition` | Gate: RSI in neutral band → no extreme momentum |
| `SignalDrawdownCondition` | Per-signal resettable drawdown circuit breaker |
| `RegimeFilter` | Wraps any signal, zeroes it in unfavourable regimes |
| `RegimeFilterPresets` | Pre-configured gates for RSI, z-score, MACD, BB breakout |
| `RegimeFilteredEnsemble` | Regime-adaptive: MR pool vs trend pool with gating comparison |

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

## Phase 7 — ML Signal (Walk-Forward Validated)

| Component | Description |
|-----------|-------------|
| `WalkForwardSplitter` | Strictly forward-only temporal folds with embargo gap |
| `MLSignalGenerator` | XGBoost/Random Forest on 24 features, fit per fold |
| `MLSignalResult` | Signal + predictions + fold report + feature importance |
| `threshold_sweep()` | Deadband analysis without backtest Sharpe — prevents overfitting |
| `get_feature_columns()` | Blocklist-based feature selection — no signal/regime leakage |

**Leak prevention guarantees:**
- Every fold's training set ends strictly before its test set starts
- Embargo gap matches the 5-day forward-return horizon
- No test timestamp appears in any earlier fold's training data (verified by direct index intersection)
- Feature columns exclude all `signal_*`, `position_*`, and `regime_*` columns
- Pure-noise sanity check: the model does not find stable IC on random targets

---

## Gating comparison

The pipeline automatically pairs each base signal with its regime-gated counterpart and reports the delta:

```
signal_rsi:      DD -38.2% → -1.8%   [gating_helped_low_sample]
signal_zscore:   DD -49.0% → +0.0%   [gating_helped_low_sample]
signal_macd:     DD -40.6% → -3.8%   [gating_helped_low_sample]
signal_bb:       DD -9.7%  → -1.4%   [gating_helped_low_sample]
signal_ensemble: DD -62.5% → -3.8%   [gating_helped_low_sample]
```

Verdicts account for trade count — at very low sample sizes, Sharpe is statistically unreliable and the verdict is based on drawdown and total return improvement rather than Sharpe delta.

---

## Example CLI output

```
── Summary ─────────────────────────────────────────────────────────
  AAPL    rows=1458  features=24  signals=17
  ensemble_Sharpe=-0.96  adaptive_Sharpe=-0.99
  CAPM_alpha=-0.214(t=-1.85✗)
  gating[gating_helped_low_sample] DD=-62.5%→-3.8%

Outputs in: data/
  processed/   — CSVs + quality reports
  results/     — backtest CSVs + factor reports + gating comparisons
```

---

### Output per ticker (full)

- `{TICKER}_processed.csv` — Full DataFrame (50+ columns)
- `{TICKER}_data_report.json` — Data quality metrics
- `{TICKER}_feature_report.json` — Feature quality metrics
- `{TICKER}_signal_report.json` — Signal quality metrics
- `{TICKER}_backtest.csv` — Backtest results (all signals ranked by Sharpe)
- `{TICKER}_gating_comparison.csv` — Gated vs naive comparison with verdict
- `{TICKER}_factor_report.json` — Factor attribution (alpha/beta per signal × model)
- `{TICKER}_regime_report.json` — Sharpe per signal per regime
- `{TICKER}_rolling_alpha.csv` — Rolling CAPM alpha time series
- `{TICKER}_ml_fold_report.csv` — Walk-forward fold metrics (ML only)
- `{TICKER}_ml_importance.csv` — Feature importance across folds (ML only)
- `{TICKER}_ml_threshold_sweep.csv` — Deadband sweep analysis (ML only)
- `watchlist_comparison.csv` — Cross-ticker ensemble comparison

---

## Design principles

- **Lookahead-bias-free.** All signals, features, normalisations, backtest fills, regime gates, HMM probabilities, and ML walk-forward folds use only past observations. Verified by dedicated tests at every phase.
- **Drawdown circuit breakers at two levels.** Portfolio-level one-shot breaker stops trading on catastrophic loss. Per-signal resettable breakers pause individual signals on drawdown and re-open on recovery.
- **Tightened regime gates.** Mean-reversion signals only active in low-vol, weak-trend regimes. Trend-following signals require MACD confirmation. Structural AND-gate symmetry between MR and trend sides.
- **Probability-weighted adaptive switching.** `signal_adaptive` blends signals by regime confidence, not binary on/off — reducing whipsaw at regime boundaries. Position size scales continuously with confidence via `position_scale_adaptive`.
- **Honest gating comparison.** Automated verdicts distinguish genuine improvement from statistical noise at low trade counts. Sharpe deltas are flagged as unreliable when trades < 15.
- **Walk-forward ML validation.** No global scaling, no global feature selection, no threshold optimisation on backtest Sharpe. Every fold is trained independently with train-set-only imputation.
- **Per-symbol error isolation.** One bad ticker doesn't abort the entire batch.
- **Separation of concerns.** All modules are independent, testable, and swappable.
- **Wilder's smoothing (RSI/ATR).** Industry-standard implementation matches Bloomberg/TradingView values.
- **Newey-West HAC standard errors.** Factor regression t-statistics account for heteroskedasticity and autocorrelation in financial returns.

---

## Test suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_pipeline.py` | 16 | Data cleaning |
| `test_features.py` | 49 | Feature engineering |
| `test_signals.py` | 58 | Signal generation |
| `test_backtester.py` | 48 | Backtesting + circuit breaker |
| `test_factor_model.py` | 37 | Factor attribution + regime analysis |
| `test_regime_filter.py` | 55 | Binary regime gates + drawdown breakers |
| `test_regime_detector.py` | 36 | HMM + rule-based detection + adaptive switch |
| `test_ml_signal.py` | 42 | ML signal — leak prevention, walk-forward, threshold sweep |
| **Total** | **341** | **All passing** |

---

## Research Directions

### Option A: Signal parameter optimisation
Grid search RSI/MACD/z-score thresholds. The vectorised backtester runs 100 combinations in seconds.

### Option B: Multi-asset portfolio construction
Trade all tickers simultaneously with volatility-weighted allocation and risk management.

### Option C: Regime-specific ML
Train separate models per regime using the regime labels already produced by the pipeline. The regime analysis tells you which features matter in which market conditions.

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ | Data ingestion, cleaning, alignment, returns |
| **Phase 2** | ✅ | Feature engineering — 7 families, 24 columns |
| **Phase 3** | ✅ | Signal generation — 5 signals + ensemble |
| **Phase 4** | ✅ | Backtesting — dual-engine, circuit breaker, full metrics |
| **Phase 5** | ✅ | Factor attribution + per-regime Sharpe analysis |
| **Phase 6a** | ✅ | Regime filtering — binary gates, per-signal breakers, gating comparison |
| **Phase 6b** | ✅ | Regime detection — HMM, rule-based, probability-weighted switching |
| **Phase 7** | ✅ | ML signal — XGBoost/Random Forest, walk-forward validation, leak-proof |
| **Phase 8** | Planned | Portfolio layer — multi-asset backtest, risk management |
| **Phase 9** | Planned | Research report — automated tear sheet PDF/DOCX |
| **Phase 10** | Planned | QuantOS integration — CLI commands via `lass_sh` |

---

## Author

**Slim Issa** — QuantOS Project  
June 2026
