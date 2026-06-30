"""
powerbi_prep/prepare_powerbi_data.py — Phase 1 of the Power BI roadmap.

Converts every pipeline output in data/results/ and data/processed/ into
flat, Power-BI-ready CSVs in powerbi/, and consolidates per-ticker files
into single watchlist-wide tables (one row per ticker x signal x model,
instead of one file per ticker) so Power BI only needs five imports
total, not five per ticker.

WHY THIS SCRIPT IS DEFENSIVE, NOT OPTIMISTIC
----------------------------------------------
A first audit of data/results/ found three problems that would silently
corrupt a dashboard if this script just trusted the files were correct:

  1. *_factor_report.json had n_obs=0 for every signal x model — caused
     by a silent yf.download() failure with no error surfaced anywhere
     (now fixed in src/factor_model.py, raises RuntimeError instead).
  2. *_rolling_alpha.csv was header-only (zero data rows) for every
     ticker — same root cause as (1), since rolling attribution needs
     the same factor data.
  3. *_gating_comparison.csv did not exist in the data/results/ folder
     at all, despite being core to the README's documented output and
     to Page 4 of the dashboard plan.

Rather than crash or — worse — produce a dashboard that quietly shows
zeros as if they were real, this script:
  - converts what's healthy now (regime reports, backtest results)
  - explicitly QUARANTINES anything that looks like the n_obs=0 /
    empty-rolling-alpha failure pattern, with a clear console warning
  - reports exactly what's missing so you know what a pipeline rerun
    needs to produce before Pages 2/3 and half of Page 4 will work

Run this again after rerunning the pipeline with more tickers and the
factor_model.py fix in place — it will pick up the new files
automatically with no code changes needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum total |alpha_annual| + |beta| across ALL regressions for a
# factor report to be considered "real". A report where every single
# regression came back n_obs=0 (alpha=0, beta exactly 1.0 or 0.0, r2=0)
# is the signature of the silent-empty-download bug, not a genuine
# finding of zero alpha. Genuine zero-alpha results exist too — but
# they don't ALSO have beta pinned exactly at the model's neutral
# default. We flag on n_obs, not on the substantive results, to avoid
# accidentally discarding a real "no significant alpha" finding.
MIN_REQUIRED_NOBS = 1


def discover_tickers() -> list[str]:
    """Find every ticker that has at least a backtest.csv — the one
    file type we know is always populated."""
    tickers = sorted(
        {p.stem.replace("_backtest", "") for p in RESULTS_DIR.glob("*_backtest.csv")}
    )
    return tickers


def load_backtest(tickers: list[str]) -> pd.DataFrame:
    """Consolidate {TICKER}_backtest.csv → one wide table with a ticker column."""
    frames = []
    for t in tickers:
        f = RESULTS_DIR / f"{t}_backtest.csv"
        if not f.exists():
            print(f"  [MISSING] {f.name}")
            continue
        df = pd.read_csv(f)
        df = df.rename(columns={df.columns[0]: "signal"})
        df.insert(0, "ticker", t)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def load_regime_reports(tickers: list[str]) -> pd.DataFrame:
    """Flatten {TICKER}_regime_report.json (nested dict of regime ->
    signal -> Sharpe) into a long table: ticker, regime, signal, sharpe,
    pct_time_in_regime. This is the file the Page 4 heatmap depends on,
    and it is currently the healthiest output in the pipeline."""
    rows = []
    for t in tickers:
        f = RESULTS_DIR / f"{t}_regime_report.json"
        if not f.exists():
            print(f"  [MISSING] {f.name}")
            continue
        data = json.loads(f.read_text())
        pct_keys = {k: v for k, v in data.items() if k.startswith("pct_")}
        sharpe_keys = {k: v for k, v in data.items() if not k.startswith("pct_")}

        for regime, signal_sharpes in sharpe_keys.items():
            for signal, sharpe in signal_sharpes.items():
                pct_key = f"pct_{regime}"
                pct_val = pct_keys.get(pct_key, {}).get(signal)
                rows.append(
                    {
                        "ticker": t,
                        "regime": regime,
                        "signal": signal,
                        "sharpe": sharpe,
                        "pct_time_in_regime": pct_val,
                    }
                )
    return pd.DataFrame(rows)


def load_factor_reports(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Flatten {TICKER}_factor_report.json into a long table: ticker,
    signal, model, alpha_annual, t_stat, p_value, r2, n_obs, betas...

    Returns (dataframe_of_healthy_rows, list_of_quarantined_tickers).
    A ticker is quarantined entirely if EVERY regression in its report
    has n_obs below MIN_REQUIRED_NOBS — that is the empty-download
    failure signature, not a real finding.

    Beyond that ticker-level check, every row also gets a
    `low_sample_size` flag. This is a SEPARATE, finer-grained signal:
    a ticker can be healthy overall (most signals trade often enough
    for a reliable regression) while a specific low-frequency signal —
    e.g. signal_regime_adaptive, which may only trigger ~30 times in a
    1250-day window — still falls below the model's own n<30
    reliability floor (see factor_model.py's OLS fit, which returns an
    empty regression below that threshold). Without this flag, a
    legitimate "not enough trades to say anything statistically" result
    is visually indistinguishable in the data from a genuine, well-
    powered "this signal truly has zero alpha" result — both show
    alpha=0, t=0, r2=0. The dashboard should not plot these as if they
    were comparable.
    """
    rows = []
    quarantined = []

    # Same threshold factor_model.py's OLS fit uses to decide whether a
    # regression is reliable enough to report (see the n < 30 check
    # immediately before the near-zero-variance guard in _fit()).
    RELIABLE_N_THRESHOLD = 30

    for t in tickers:
        f = RESULTS_DIR / f"{t}_factor_report.json"
        if not f.exists():
            print(f"  [MISSING] {f.name}")
            continue
        data = json.loads(f.read_text())
        regressions = data.get("regressions", {})

        ticker_rows = []
        max_nobs_seen = 0
        for signal, reglist in regressions.items():
            for reg in reglist:
                n_obs = reg.get("n_obs", 0)
                max_nobs_seen = max(max_nobs_seen, n_obs)
                row = {
                    "ticker": t,
                    "signal": signal,
                    "model": reg.get("model"),
                    "alpha_annual": reg.get("alpha_annual"),
                    "alpha_daily": reg.get("alpha_daily"),
                    "t_stat": reg.get("t_stat"),
                    "p_value": reg.get("p_value"),
                    "significant": reg.get("significant"),
                    "r2": reg.get("r2"),
                    "adj_r2": reg.get("adj_r2"),
                    "ic": reg.get("ic"),
                    "ir": reg.get("ir"),
                    "n_obs": n_obs,
                    "low_sample_size": n_obs < RELIABLE_N_THRESHOLD,
                }
                betas = reg.get("betas", {})
                for fname, fval in betas.items():
                    row[f"beta_{fname}"] = fval
                ticker_rows.append(row)

        if max_nobs_seen < MIN_REQUIRED_NOBS:
            quarantined.append(t)
            print(
                f"  [QUARANTINED] {f.name} — every regression has "
                f"n_obs=0. This is the empty-ETF-download failure "
                f"signature, not a genuine zero-alpha finding. "
                f"Excluded from factor_attribution.csv."
            )
        else:
            n_low_sample = sum(1 for r in ticker_rows if r["low_sample_size"])
            if n_low_sample:
                print(
                    f"  [NOTE] {f.name} — {n_low_sample}/{len(ticker_rows)} "
                    f"rows have n_obs < {RELIABLE_N_THRESHOLD} (signal "
                    f"trades too rarely for a reliable regression, not a "
                    f"data error). Flagged via low_sample_size, not "
                    f"excluded."
                )
            rows.extend(ticker_rows)

    return pd.DataFrame(rows), quarantined


def load_rolling_alpha(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Consolidate {TICKER}_rolling_alpha.csv. Flags tickers whose file
    is header-only (zero data rows) as quarantined for the same reason
    as the factor reports — same root cause, same fix."""
    frames = []
    quarantined = []
    for t in tickers:
        f = RESULTS_DIR / f"{t}_rolling_alpha.csv"
        if not f.exists():
            print(f"  [MISSING] {f.name}")
            continue
        df = pd.read_csv(f)
        if len(df) == 0:
            quarantined.append(t)
            print(
                f"  [QUARANTINED] {f.name} — header only, zero data rows. "
                f"Same root cause as the factor report failure. "
                f"Excluded from rolling_alpha.csv."
            )
            continue
        df = df.rename(columns={df.columns[0]: "date"})
        df.insert(0, "ticker", t)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), quarantined
    return pd.concat(frames, ignore_index=True), quarantined


def load_gating_comparison(tickers: list[str]) -> pd.DataFrame:
    """Consolidate {TICKER}_gating_comparison.csv — this file did not
    exist anywhere in the supplied data/results/ at audit time, despite
    being central to Page 4 and listed in the README's output spec.
    Returns whatever is found; an empty result here just means none
    exist yet, which the summary report will call out explicitly."""
    frames = []
    for t in tickers:
        f = RESULTS_DIR / f"{t}_gating_comparison.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df.insert(0, "ticker", t)
        frames.append(df)
    if not frames:
        print(f"  [MISSING] *_gating_comparison.csv for ALL {len(tickers)} ticker(s)")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ml_reports(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Consolidate the three optional ML output types, if ml_signal is
    enabled in config.yaml and has been run. Returns a dict of
    {output_name: dataframe}; missing types are simply absent."""
    out: dict[str, pd.DataFrame] = {}
    for kind in ("ml_fold_report", "ml_importance", "ml_threshold_sweep"):
        frames = []
        for t in tickers:
            f = RESULTS_DIR / f"{t}_{kind}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df.insert(0, "ticker", t)
            frames.append(df)
        if frames:
            out[kind] = pd.concat(frames, ignore_index=True)
        else:
            print(f"  [MISSING] *_{kind}.csv for ALL tickers (ml_signal likely disabled)")
    return out


def main() -> None:
    tickers = discover_tickers()
    if not tickers:
        print(f"No *_backtest.csv files found in {RESULTS_DIR}. Nothing to do.")
        sys.exit(1)

    print(f"Discovered {len(tickers)} ticker(s): {tickers}\n")

    print("── Backtest results ──────────────────────────────────────")
    backtest_df = load_backtest(tickers)
    if not backtest_df.empty:
        backtest_df.to_csv(OUT_DIR / "backtest_results.csv", index=False)
        print(f"  -> {OUT_DIR/'backtest_results.csv'}  ({len(backtest_df)} rows)")

    print("\n── Regime reports (Page 4 heatmap) ───────────────────────")
    regime_df = load_regime_reports(tickers)
    if not regime_df.empty:
        regime_df.to_csv(OUT_DIR / "regime_sharpe.csv", index=False)
        print(f"  -> {OUT_DIR/'regime_sharpe.csv'}  ({len(regime_df)} rows)")

    print("\n── Factor attribution (Page 2) ───────────────────────────")
    factor_df, factor_quarantined = load_factor_reports(tickers)
    if not factor_df.empty:
        factor_df.to_csv(OUT_DIR / "factor_attribution.csv", index=False)
        print(f"  -> {OUT_DIR/'factor_attribution.csv'}  ({len(factor_df)} rows)")
    else:
        print("  -> factor_attribution.csv NOT written — no healthy data for any ticker.")

    print("\n── Rolling alpha (Page 3) ────────────────────────────────")
    rolling_df, rolling_quarantined = load_rolling_alpha(tickers)
    if not rolling_df.empty:
        rolling_df.to_csv(OUT_DIR / "rolling_alpha.csv", index=False)
        print(f"  -> {OUT_DIR/'rolling_alpha.csv'}  ({len(rolling_df)} rows)")
    else:
        print("  -> rolling_alpha.csv NOT written — no healthy data for any ticker.")

    print("\n── Gating comparison (Page 4 before/after) ───────────────")
    gating_df = load_gating_comparison(tickers)
    if not gating_df.empty:
        gating_df.to_csv(OUT_DIR / "gating_comparison.csv", index=False)
        print(f"  -> {OUT_DIR/'gating_comparison.csv'}  ({len(gating_df)} rows)")

    print("\n── ML signal reports (optional) ──────────────────────────")
    ml_outputs = load_ml_reports(tickers)
    for name, df in ml_outputs.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  -> {OUT_DIR/(name + '.csv')}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Summary: exactly what's ready, what's quarantined, what's missing
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — what's ready for Power BI right now")
    print("=" * 70)
    print(f"Tickers found:              {len(tickers)} — {tickers}")
    print(f"backtest_results.csv:       {'OK' if not backtest_df.empty else 'MISSING'}")
    print(f"regime_sharpe.csv:          {'OK' if not regime_df.empty else 'MISSING'}")
    print(
        f"factor_attribution.csv:     "
        f"{'OK' if not factor_df.empty else 'EMPTY/MISSING'}"
        + (f"  (quarantined: {factor_quarantined})" if factor_quarantined else "")
    )
    print(
        f"rolling_alpha.csv:          "
        f"{'OK' if not rolling_df.empty else 'EMPTY/MISSING'}"
        + (f"  (quarantined: {rolling_quarantined})" if rolling_quarantined else "")
    )
    print(f"gating_comparison.csv:      {'OK' if not gating_df.empty else 'MISSING for all tickers'}")
    print(f"ML reports:                 {list(ml_outputs.keys()) if ml_outputs else 'none found'}")

    if factor_quarantined or rolling_quarantined or gating_df.empty:
        print("\nNEXT STEP: rerun the pipeline with:")
        print("  - the factor_model.py fix in place (raises loudly on empty ETF download)")
        print("  - factor.enabled: true in config.yaml")
        print("  - the expanded watchlist (NVDA, TSLA, AMZN, META, JPM, V, WMT, XOM, JNJ, BA)")
        print("Then re-run this script — it will pick up the new files with no changes needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()