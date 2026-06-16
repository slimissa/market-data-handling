"""
factor_model.py — Phase 5: Factor Attribution
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → signals → backtest → [factor attribution]

Answers the core question every quant must face:
    "Is your signal generating alpha, or just levered market beta?"

Three levels of analysis:

    1. CAPM Attribution
       r_strategy = alpha + beta * r_market + epsilon
       Decomposes returns into market exposure (beta) and genuine skill (alpha).
       A strategy with Sharpe=1.2 but alpha=-0.05 is just a complicated
       way to buy the index — you're being paid for risk, not skill.

    2. Fama-French 3-Factor Attribution
       r = alpha + b1*MKT + b2*SMB + b3*HML + epsilon
       Controls for size (SMB: small minus big) and value (HML: high minus low)
       premiums. A strategy that looks like alpha might just be a value tilt.
       FF3 is the minimum credible factor model in academic finance.

    3. Carhart 4-Factor Attribution (FF3 + Momentum)
       r = alpha + b1*MKT + b2*SMB + b3*HML + b4*MOM + epsilon
       Adds momentum factor (WML: winners minus losers).
       A momentum signal that shows alpha on FF3 might disappear under Carhart.

    4. Rolling Attribution
       Runs CAPM regression in a rolling window (63 or 126 days).
       Shows WHEN a signal had alpha vs. when it was purely beta.
       Regime analysis: was the signal working in 2020 crash? 2021 recovery?
       This is what you show when an interviewer asks "is it stable?"

    5. Signal Decomposition
       Quantifies what fraction of each signal's variance is explained by:
       - Market direction (beta)
       - Volatility regime (vol factor)
       - Residual (unexplained / potential alpha)

    6. Factor Exposure Report
       Full attribution table for every signal:
       alpha | t-stat | MKT_beta | SMB_beta | HML_beta | MOM_beta | R2 | IC

Factor data sources:
    - MKT (market): SPY daily returns (fetched via yfinance)
    - SMB, HML, MOM: proxied from ETF spreads when Ken French data unavailable
      IWM-SPY spread ≈ SMB (small minus large)
      IWD-IWF spread ≈ HML (value minus growth)
      MTUM-SPY spread ≈ MOM (momentum vs market)

Usage:
    fm = FactorModel()
    results = fm.run(backtest_results, benchmark_ticker="SPY")
    print(results.attribution_table)
    print(results.rolling_alpha["signal_rsi"])
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class FactorRegression:
    """
    Result of a single OLS factor regression.

    r_strategy = alpha + sum(beta_i * factor_i) + epsilon

    Statistics:
        t_stat:   alpha / se(alpha). |t| > 2 is conventional significance.
        p_value:  probability of observing this alpha by chance if true alpha=0.
                  p < 0.05 is the standard threshold but quant research
                  typically requires p < 0.01 to avoid overfitting.
        r2:       fraction of strategy variance explained by factors.
                  R2 near 1 → strategy IS the factors. R2 near 0 → independent.
        ic:       Information Coefficient — correlation between factor-predicted
                  returns and actual returns. IC > 0.05 is considered useful.
    """
    signal_col:   str
    model_name:   str           # "CAPM", "FF3", "Carhart4"
    alpha_daily:  float         # raw daily alpha
    alpha_annual: float         # annualised alpha
    t_stat:       float
    p_value:      float
    betas:        Dict[str, float]   # factor_name → beta coefficient
    r2:           float
    adj_r2:       float
    ic:           float
    n_obs:        int
    residuals:    pd.Series = field(default_factory=pd.Series)

    @property
    def is_significant(self) -> bool:
        """Alpha is statistically significant at 5% level."""
        return abs(self.t_stat) > 2.0 and self.p_value < 0.05

    @property
    def information_ratio(self) -> float:
        """IR = alpha / tracking_error. Annualised."""
        if len(self.residuals) < 2:
            return 0.0
        te = float(self.residuals.std() * np.sqrt(252))
        return self.alpha_annual / te if te > 0 else 0.0

    def summary_line(self) -> str:
        stars = "***" if self.p_value < 0.01 else ("**" if self.p_value < 0.05 else
                ("*" if self.p_value < 0.10 else ""))
        betas_str = "  ".join(
            f"{k}={v:+.3f}" for k, v in self.betas.items()
        )
        return (
            f"{self.signal_col:22s} [{self.model_name:8s}] "
            f"α={self.alpha_annual:+.4f}{stars:3s}  "
            f"t={self.t_stat:+.2f}  "
            f"R²={self.r2:.3f}  "
            f"IC={self.ic:.3f}  "
            f"IR={self.information_ratio:.3f}  "
            f"{betas_str}"
        )


@dataclass
class RollingAttribution:
    """
    Rolling-window CAPM regression for a single signal.

    Shows temporal stability: was the signal consistently alpha-generating,
    or did it work only in specific regimes?
    """
    signal_col:    str
    window:        int
    alpha_series:  pd.Series    # rolling alpha (annualised)
    beta_series:   pd.Series    # rolling market beta
    r2_series:     pd.Series    # rolling R²
    regime_labels: pd.Series    # "trending" | "range_bound" | "crisis"


@dataclass
class FactorModelResults:
    """
    Complete Phase 5 output: all regressions + rolling analysis + report.
    """
    regressions:      Dict[str, List[FactorRegression]]  # signal → [CAPM, FF3, C4]
    rolling:          Dict[str, RollingAttribution]       # signal → rolling
    attribution_table: pd.DataFrame                        # summary table
    factor_returns:   pd.DataFrame                         # the factor time series used
    ticker:           str = ""

    def print_summary(self) -> None:
        print(f"\n{'─'*100}")
        print(f"FACTOR ATTRIBUTION SUMMARY — {self.ticker or 'unknown'}")
        print(f"{'─'*100}")
        print(f"{'Signal':<22}  {'Model':<8}  {'Alpha':>8}  {'t-stat':>7}  "
              f"{'R²':>6}  {'IC':>6}  {'IR':>6}  Betas")
        print(f"{'─'*100}")
        for sig, regs in self.regressions.items():
            for reg in regs:
                print(f"  {reg.summary_line()}")
        print(f"{'─'*100}\n")


# ======================================================================
# Factor data loader
# ======================================================================

class FactorDataLoader:
    """
    Load or construct factor return series.

    Primary source: Ken French Data Library (if available via pandas_datareader)
    Fallback: ETF-spread proxies constructed from yfinance data

    Factor proxies (ETF spreads):
        MKT  = SPY returns - rf
        SMB  = IWM returns - SPY returns  (small minus large)
        HML  = IWD returns - IWF returns  (value minus growth)
        MOM  = MTUM returns - SPY returns (momentum minus market)
        VOL  = VXX or SVXY proxy          (vol factor — optional)
    """

    # ETF proxies for each factor
    FACTOR_ETFS = {
        "MKT":  ("SPY",  None),    # SPY vs rf
        "SMB":  ("IWM",  "SPY"),   # small (IWM) - large (SPY)
        "HML":  ("IWD",  "IWF"),   # value (IWD) - growth (IWF)
        "MOM":  ("MTUM", "SPY"),   # momentum (MTUM) - market (SPY)
    }

    def __init__(self, cache_dir: str = "./data/factors"):
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load(
        self,
        start_date: str,
        end_date: str,
        factors: List[str] = ("MKT", "SMB", "HML", "MOM"),
        rf_annual: float = 0.05,
    ) -> pd.DataFrame:
        """
        Load factor returns for the given date range.

        Tries Ken French data first, falls back to ETF proxies.

        Returns:
            DataFrame with columns = factor names, index = DatetimeIndex (UTC)
        """
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1

        # Try Ken French (requires pandas_datareader)
        ff_data = self._try_french_library(start_date, end_date)
        if ff_data is not None:
            logger.info("Loaded Fama-French factors from Ken French library.")
            return ff_data

        # Fallback: ETF proxies
        logger.info("Ken French library unavailable — building ETF-proxy factors.")
        return self._build_etf_proxies(start_date, end_date, list(factors), rf_daily)

    def _try_french_library(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Attempt to fetch FF3+MOM from pandas_datareader."""
        try:
            import pandas_datareader.data as web
            # Fama-French 3-Factor
            ff3 = web.DataReader(
                "F-F_Research_Data_Factors_daily", "famafrench",
                start=start_date, end=end_date
            )[0] / 100  # convert from percent to decimal

            # Momentum factor
            mom = web.DataReader(
                "F-F_Momentum_Factor_daily", "famafrench",
                start=start_date, end=end_date
            )[0] / 100

            df = ff3.join(mom, how="inner")
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
            df.columns = [c.strip() for c in df.columns]
            # Rename to standard names
            rename = {
                "Mkt-RF": "MKT", "SMB": "SMB", "HML": "HML",
                "RF": "RF", "Mom   ": "MOM", "Mom": "MOM",
            }
            df = df.rename(columns=rename)
            return df

        except Exception as exc:
            logger.debug(f"pandas_datareader unavailable: {exc}")
            return None

    def _build_etf_proxies(
        self,
        start_date: str,
        end_date: str,
        factors: List[str],
        rf_daily: float,
    ) -> pd.DataFrame:
        """Build factor proxies from ETF return spreads."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required for ETF factor proxies.")

        # Collect all unique tickers needed
        tickers_needed = set()
        for f in factors:
            if f in self.FACTOR_ETFS:
                long_etf, short_etf = self.FACTOR_ETFS[f]
                tickers_needed.add(long_etf)
                if short_etf:
                    tickers_needed.add(short_etf)

        # Download all at once
        logger.info(f"Downloading ETF proxies: {sorted(tickers_needed)}")
        raw = yf.download(
            list(tickers_needed),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        # Extract close prices
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": list(tickers_needed)[0]})

        returns = prices.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index).tz_localize("UTC")

        # Build factor series
        factor_df = pd.DataFrame(index=returns.index)
        factor_df["RF"] = rf_daily  # constant risk-free rate

        for f in factors:
            if f not in self.FACTOR_ETFS:
                continue
            long_etf, short_etf = self.FACTOR_ETFS[f]

            if long_etf not in returns.columns:
                logger.warning(f"ETF {long_etf} not in downloaded data, skipping {f}")
                continue

            if f == "MKT":
                # Excess market return = SPY - rf
                factor_df["MKT"] = returns[long_etf] - rf_daily
            elif short_etf and short_etf in returns.columns:
                # Long-short spread
                factor_df[f] = returns[long_etf] - returns[short_etf]
            else:
                logger.warning(f"Cannot build {f} factor — {short_etf} missing")

        factor_df = factor_df.dropna()
        logger.info(
            f"Built {len(factor_df.columns)-1} factor proxies: "
            f"{[c for c in factor_df.columns if c != 'RF']}"
        )
        return factor_df


# ======================================================================
# OLS Regression engine
# ======================================================================

class OLSRegressor:
    """
    Ordinary Least Squares regression with full statistics.

    r_excess = alpha + B @ factors + epsilon

    Computes:
        - Coefficient estimates (Newey-West HAC standard errors)
        - t-statistics and p-values
        - R², adjusted R²
        - Information Coefficient (IC)
        - Residuals

    Uses statsmodels for robust standard errors.
    Falls back to numpy if statsmodels unavailable.
    """

    def fit(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        signal_col: str = "",
        model_name: str = "",
        rf: Optional[pd.Series] = None,
    ) -> FactorRegression:
        """
        Fit OLS regression: y = alpha + X @ beta + epsilon

        Args:
            y:          Strategy excess returns (strategy - rf)
            X:          Factor returns DataFrame (columns = factor names)
            signal_col: Label for the signal being attributed
            model_name: Label for the model (CAPM, FF3, Carhart4)
            rf:         Risk-free rate series (for excess return computation)

        Returns:
            FactorRegression with all statistics
        """
        # Align indices
        common = y.index.intersection(X.index)
        if len(common) < 30:
            logger.warning(
                f"Only {len(common)} observations for {signal_col} — "
                "regression may be unreliable."
            )
        y_aligned = y.loc[common].dropna()
        X_aligned = X.loc[y_aligned.index].dropna()
        y_aligned = y_aligned.loc[X_aligned.index]

        n = len(y_aligned)
        k = X_aligned.shape[1]

        if n < 30:
            return self._empty_regression(signal_col, model_name, X_aligned.columns)

        # Add constant (alpha)
        X_const = X_aligned.copy()
        X_const.insert(0, "const", 1.0)

        try:
            result = self._fit_statsmodels(y_aligned, X_const, model_name)
        except Exception:
            result = self._fit_numpy(y_aligned, X_const.values)

        alpha_daily = float(result["params"][0])
        alpha_annual = float((1 + alpha_daily) ** 252 - 1)
        betas = {
            col: float(result["params"][i + 1])
            for i, col in enumerate(X_aligned.columns)
        }

        # IC: correlation between factor-predicted returns and actual returns
        y_pred = X_const.values @ result["params"]
        ic = float(np.corrcoef(y_aligned.values, y_pred)[0, 1])

        # Residuals
        residuals = pd.Series(
            y_aligned.values - y_pred, index=y_aligned.index
        )

        # R²
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_aligned - y_aligned.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else 0.0

        return FactorRegression(
            signal_col=signal_col,
            model_name=model_name,
            alpha_daily=alpha_daily,
            alpha_annual=alpha_annual,
            t_stat=float(result["tvalues"][0]),
            p_value=float(result["pvalues"][0]),
            betas=betas,
            r2=r2,
            adj_r2=adj_r2,
            ic=ic,
            n_obs=n,
            residuals=residuals,
        )

    def _fit_statsmodels(
        self, y: pd.Series, X_const: pd.DataFrame, model_name: str
    ) -> dict:
        """
        Fit with Newey-West HAC standard errors.

        HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
        are essential for financial time series where:
            - Returns exhibit autocorrelation (momentum effects)
            - Variance is time-varying (GARCH effects)
        Using OLS standard errors on financial returns understates uncertainty.
        """
        import statsmodels.api as sm
        lags = min(int(np.sqrt(len(y))), 12)   # Newey-West lag selection
        model = sm.OLS(y, X_const)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return {
            "params":  res.params.values,
            "tvalues": res.tvalues.values,
            "pvalues": res.pvalues.values,
        }

    @staticmethod
    def _fit_numpy(y: pd.Series, X: np.ndarray) -> dict:
        """
        Pure numpy OLS fallback.
        Uses standard OLS standard errors (no HAC correction).
        """
        n, k = X.shape
        # OLS: beta = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y.values

        try:
            params = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(X, y.values, rcond=None)[0]

        residuals = y.values - X @ params
        sigma2 = np.sum(residuals ** 2) / (n - k)
        var_beta = sigma2 * np.linalg.pinv(XtX)
        se = np.sqrt(np.diag(var_beta))
        tvalues = params / np.where(se > 0, se, np.nan)
        from scipy import stats
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df=n - k))

        return {
            "params":  params,
            "tvalues": tvalues,
            "pvalues": pvalues,
        }

    @staticmethod
    def _empty_regression(
        signal_col: str, model_name: str, factor_names
    ) -> FactorRegression:
        return FactorRegression(
            signal_col=signal_col, model_name=model_name,
            alpha_daily=0.0, alpha_annual=0.0,
            t_stat=0.0, p_value=1.0,
            betas={f: 0.0 for f in factor_names},
            r2=0.0, adj_r2=0.0, ic=0.0, n_obs=0,
        )


# ======================================================================
# Factor Model — main class
# ======================================================================

class FactorModel:
    """
    Factor attribution for backtest results.

    Usage:
        fm = FactorModel()
        results = fm.run(
            daily_returns_dict={"signal_rsi": rsi_returns, ...},
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        results.print_summary()
        print(results.attribution_table)
    """

    def __init__(
        self,
        rf_annual: float = 0.05,
        rolling_window: int = 126,   # 6-month rolling window
        cache_dir: str = "./data/factors",
    ):
        self.rf_annual     = rf_annual
        self.rf_daily      = (1 + rf_annual) ** (1 / 252) - 1
        self.rolling_window = rolling_window
        self.loader        = FactorDataLoader(cache_dir=cache_dir)
        self.ols           = OLSRegressor()

    # ------------------------------------------------------------------ #
    # Primary entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        daily_returns: Dict[str, pd.Series],
        start_date: str,
        end_date: str,
        ticker: str = "",
        factors: List[str] = ("MKT", "SMB", "HML", "MOM"),
    ) -> FactorModelResults:
        """
        Run full factor attribution for all signals.

        Args:
            daily_returns:  Dict mapping signal_col → daily return Series
                            (from BacktestResults.daily_returns)
            start_date:     ISO date string
            end_date:       ISO date string
            ticker:         Label for results
            factors:        Which factors to include

        Returns:
            FactorModelResults with all regressions, rolling analysis, table
        """
        tag = f"[{ticker}] " if ticker else ""
        logger.info(f"{tag}Loading factor data ({start_date} → {end_date})...")

        # Load factor returns
        factor_df = self.loader.load(
            start_date=start_date,
            end_date=end_date,
            factors=list(factors),
            rf_annual=self.rf_annual,
        )

        logger.info(
            f"{tag}Factor data loaded: {len(factor_df)} observations, "
            f"factors: {[c for c in factor_df.columns if c != 'RF']}"
        )

        # Determine which models to run based on available factors
        available = [c for c in factor_df.columns if c != "RF"]
        models = self._select_models(available)

        # Run regressions for each signal
        all_regressions: Dict[str, List[FactorRegression]] = {}
        all_rolling:     Dict[str, RollingAttribution]     = {}

        for sig_col, ret_series in daily_returns.items():
            logger.info(f"{tag}Attributing {sig_col}...")

            # Excess returns: strategy - rf
            rf_aligned = factor_df["RF"].reindex(ret_series.index).fillna(self.rf_daily)
            excess = ret_series - rf_aligned

            regs = []
            for model_name, factor_cols in models.items():
                X = factor_df[factor_cols].copy()
                reg = self.ols.fit(
                    y=excess, X=X,
                    signal_col=sig_col, model_name=model_name,
                )
                regs.append(reg)

            all_regressions[sig_col] = regs

            # Rolling CAPM
            if "MKT" in factor_df.columns:
                roll = self._rolling_capm(
                    excess=excess,
                    mkt=factor_df["MKT"],
                    signal_col=sig_col,
                    window=self.rolling_window,
                )
                all_rolling[sig_col] = roll

        # Build attribution table
        table = self._build_table(all_regressions)

        logger.info(f"{tag}Factor attribution complete.")

        return FactorModelResults(
            regressions=all_regressions,
            rolling=all_rolling,
            attribution_table=table,
            factor_returns=factor_df,
            ticker=ticker,
        )

    # ------------------------------------------------------------------ #
    # Convenience: run directly from BacktestResults                       #
    # ------------------------------------------------------------------ #

    def from_backtest(
        self,
        backtest_results: Dict[str, "BacktestResults"],  # signal → results
        start_date: str,
        end_date: str,
        ticker: str = "",
    ) -> FactorModelResults:
        """
        Run factor attribution directly from a dict of BacktestResults.

        Args:
            backtest_results: Dict from VectorisedBacktester.compare_signals()
                              output, or manually constructed.
                              Key = signal_col, Value = BacktestResults

        Example:
            vbt = VectorisedBacktester()
            results = {}
            for sig in signal_cols:
                results[sig] = vbt.run(df, signal_col=sig)
            factor_results = fm.from_backtest(results, "2020-01-01", "2023-12-31")
        """
        daily_returns = {
            sig: res.daily_returns
            for sig, res in backtest_results.items()
        }
        return self.run(daily_returns, start_date, end_date, ticker=ticker)

    # ------------------------------------------------------------------ #
    # Rolling CAPM                                                         #
    # ------------------------------------------------------------------ #

    def _rolling_capm(
        self,
        excess: pd.Series,
        mkt: pd.Series,
        signal_col: str,
        window: int,
    ) -> RollingAttribution:
        """
        Compute rolling alpha and beta over a sliding window.

        A stable signal has:
            - alpha consistently positive (or near zero)
            - beta stable over time (not regime-dependent)
            - R² relatively constant (not explained by market in some periods only)

        An unstable signal has:
            - alpha positive only in certain regimes
            - beta varying between 0 and 2 as market conditions change
            → Not deployable: you can't know in advance which regime you're in
        """
        common = excess.index.intersection(mkt.index)
        exc = excess.loc[common].dropna()
        mkt_aligned = mkt.loc[exc.index]

        n = len(exc)
        alpha_vals = pd.Series(np.nan, index=exc.index)
        beta_vals  = pd.Series(np.nan, index=exc.index)
        r2_vals    = pd.Series(np.nan, index=exc.index)

        for i in range(window, n):
            y_w = exc.iloc[i - window: i].values
            x_w = mkt_aligned.iloc[i - window: i].values

            if np.any(np.isnan(y_w)) or np.any(np.isnan(x_w)):
                continue

            # OLS in 2-var case: analytical solution
            x_dm = x_w - x_w.mean()
            beta = float(np.dot(x_dm, y_w - y_w.mean()) / (np.dot(x_dm, x_dm) + 1e-12))
            alpha_daily = float(y_w.mean() - beta * x_w.mean())

            residuals = y_w - (alpha_daily + beta * x_w)
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y_w - y_w.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            alpha_annual = float((1 + alpha_daily) ** 252 - 1)
            alpha_vals.iloc[i] = alpha_annual
            beta_vals.iloc[i]  = beta
            r2_vals.iloc[i]    = r2

        # Regime labels based on rolling market returns
        mkt_roll = mkt_aligned.rolling(63).mean()
        mkt_vol  = mkt_aligned.rolling(63).std() * np.sqrt(252)

        regime = pd.Series("range_bound", index=exc.index)
        regime[mkt_roll > 0.001] = "trending_up"
        regime[mkt_roll < -0.001] = "trending_down"
        regime[mkt_vol > 0.25] = "crisis"

        return RollingAttribution(
            signal_col=signal_col,
            window=window,
            alpha_series=alpha_vals,
            beta_series=beta_vals,
            r2_series=r2_vals,
            regime_labels=regime,
        )

    # ------------------------------------------------------------------ #
    # Attribution table builder                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_table(
        regressions: Dict[str, List[FactorRegression]],
    ) -> pd.DataFrame:
        """
        Build a flat summary DataFrame from all regressions.

        One row per (signal, model) combination.
        Columns: alpha_annual, t_stat, p_value, significant,
                 r2, adj_r2, ic, ir, n_obs, + one column per factor beta.
        """
        rows = []
        for sig, regs in regressions.items():
            for reg in regs:
                row = {
                    "signal":       sig,
                    "model":        reg.model_name,
                    "alpha_annual": round(reg.alpha_annual, 4),
                    "t_stat":       round(reg.t_stat, 3),
                    "p_value":      round(reg.p_value, 4),
                    "significant":  reg.is_significant,
                    "r2":           round(reg.r2, 4),
                    "adj_r2":       round(reg.adj_r2, 4),
                    "ic":           round(reg.ic, 4),
                    "ir":           round(reg.information_ratio, 4),
                    "n_obs":        reg.n_obs,
                }
                for factor, beta in reg.betas.items():
                    row[f"beta_{factor}"] = round(beta, 4)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index(["signal", "model"])
        return df

    @staticmethod
    def _select_models(
        available_factors: List[str],
    ) -> Dict[str, List[str]]:
        """
        Determine which models to run based on available factor data.

        Returns:
            Dict mapping model_name → list of factor columns to use
        """
        models = {}

        # CAPM: just market factor
        if "MKT" in available_factors:
            models["CAPM"] = ["MKT"]

        # FF3: market + size + value
        if all(f in available_factors for f in ["MKT", "SMB", "HML"]):
            models["FF3"] = ["MKT", "SMB", "HML"]

        # Carhart 4-factor: FF3 + momentum
        if all(f in available_factors for f in ["MKT", "SMB", "HML", "MOM"]):
            models["Carhart4"] = ["MKT", "SMB", "HML", "MOM"]

        # Fallback: single-factor with whatever is available
        if not models and available_factors:
            models["Single"] = available_factors[:1]

        return models


# ======================================================================
# Regime analysis helper
# ======================================================================

class RegimeAnalyser:
    """
    Classify market regimes and show signal performance per regime.

    Regimes:
        crisis        — annualised vol > 30%  (e.g. March 2020)
        trending_up   — rolling 63d return > 0 AND vol < 20%
        trending_down — rolling 63d return < 0 AND vol < 20%
        range_bound   — |rolling return| small AND low vol

    For each regime × signal combination, computes:
        - Mean daily return
        - Sharpe (annualised)
        - Fraction of time in each regime
    """

    @staticmethod
    def classify(
        mkt_returns: pd.Series,
        vol_window: int = 21,
        return_window: int = 63,
    ) -> pd.Series:
        """
        Classify each bar into a market regime.

        Returns:
            Series of strings: 'crisis' | 'trending_up' | 'trending_down' | 'range_bound'
        """
        vol = mkt_returns.rolling(vol_window).std() * np.sqrt(252)
        ret = mkt_returns.rolling(return_window).mean() * 252  # annualised

        regime = pd.Series("range_bound", index=mkt_returns.index)
        regime[vol > 0.30]                          = "crisis"
        regime[(ret > 0.05)  & (vol <= 0.30)]       = "trending_up"
        regime[(ret < -0.05) & (vol <= 0.30)]       = "trending_down"

        return regime

    @staticmethod
    def performance_by_regime(
        strategy_returns: Dict[str, pd.Series],
        regime: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute Sharpe ratio per signal per regime.

        Args:
            strategy_returns: Dict signal_col → daily returns
            regime:           Regime labels from classify()

        Returns:
            DataFrame with signals as rows, regimes as columns, Sharpe as values.
        """
        regimes = sorted(regime.dropna().unique())
        rows = []

        for sig, ret in strategy_returns.items():
            row = {"signal": sig}
            common = ret.index.intersection(regime.index)
            ret_aligned = ret.loc[common]
            reg_aligned = regime.loc[common]

            for r in regimes:
                mask = reg_aligned == r
                r_sub = ret_aligned[mask].dropna()
                if len(r_sub) < 5:
                    row[r] = np.nan
                else:
                    std = r_sub.std()
                    sharpe = float(r_sub.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0
                    row[r] = round(sharpe, 3)

            # Add regime frequencies
            for r in regimes:
                row[f"pct_{r}"] = round(float((reg_aligned == r).mean()), 3)

            rows.append(row)

        return pd.DataFrame(rows).set_index("signal")