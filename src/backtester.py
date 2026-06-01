"""
backtester.py — Phase 4: Strategy simulation and performance attribution
QuantOS Market Data Pipeline

Pipeline position:
    fetch → clean → features → signals → [backtest] → report

Two execution engines:

    VectorisedBacktester
        Operates on entire DataFrame at once using pandas/numpy.
        Fast (~1ms for 1000 bars). Ideal for research, parameter sweeps,
        and initial signal validation. Assumes fills at next-bar open.

    EventDrivenBacktester
        Simulates the market bar by bar. Processes each event in order:
        signal → order → fill → position update → P&L.
        Slower but realistic: enforces execution lag, slippage, and
        partial fills. Use for final validation before live deployment.

Transaction cost model (realistic):
    total_cost = spread_cost + slippage_cost + commission
    spread_cost  = 0.5 * spread_bps * price   (half-spread on entry/exit)
    slippage     = slippage_bps * price * sqrt(trade_size / adv)
    commission   = max(flat_fee, pct_fee * notional)

    Square-root market impact: slippage scales with sqrt(participation rate).
    This is the Almgren-Chriss model simplified — used by every major HF.

Performance metrics (full suite):
    Sharpe ratio     — (mean_return - rf) / std_return * sqrt(252)
    Sortino ratio    — (mean_return - rf) / downside_std * sqrt(252)
    Calmar ratio     — annualised_return / max_drawdown
    Max drawdown     — peak-to-trough equity decline (%)
    Hit rate         — fraction of trades that were profitable
    Profit factor    — gross_profit / gross_loss
    Turnover         — average daily position change (annualised)
    Alpha / Beta     — CAPM regression vs benchmark (SPY by default)
    VaR 95%          — 5th percentile of daily P&L distribution
    CVaR 95%         — mean of worst 5% of daily P&L (Expected Shortfall)

Usage:
    # Vectorised (fast, research)
    vbt = VectorisedBacktester(initial_capital=100_000)
    results = vbt.run(df_signals, signal_col="signal_ensemble")
    print(results.metrics)

    # Event-driven (realistic, pre-deployment)
    edb = EventDrivenBacktester(initial_capital=100_000)
    results = edb.run(df_signals, signal_col="signal_ensemble")
    print(results.metrics)

    # Compare all signals on one ticker
    bt = VectorisedBacktester()
    comparison = bt.compare_signals(df_signals)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class TransactionCostModel:
    """
    Realistic transaction cost model.

    Components:
        spread_bps:    Half-spread in basis points (1 bp = 0.01%).
                       Liquid large-caps: 1-2 bps. Small-caps: 5-20 bps.
        slippage_bps:  Market impact per unit of participation rate.
                       Models Almgren-Chriss square-root impact:
                       impact = slippage_bps * sqrt(trade_size / adv)
        commission:    Flat fee per trade in dollars.
        pct_fee:       Percentage fee on notional (e.g. 0.001 = 0.1%).
                       Final commission = max(flat_fee, pct_fee * notional).
        adv_shares:    Average daily volume (shares). Used for slippage.
                       Default 1M shares — reasonable for S&P 500 stocks.
    """
    spread_bps:   float = 2.0
    slippage_bps: float = 1.0
    commission:   float = 1.0    # dollars per trade
    pct_fee:      float = 0.0005 # 5 bps on notional
    adv_shares:   float = 1_000_000.0

    def total_cost(
        self,
        price: float,
        shares: float,
        volume: Optional[float] = None,
    ) -> float:
        """
        Compute total one-way transaction cost in dollars.

        Args:
            price:  Execution price
            shares: Number of shares traded (absolute value)
            volume: Bar volume (shares). Falls back to adv_shares if None.

        Returns:
            Total cost in dollars (always positive).
        """
        notional = abs(shares) * price

        # Half-spread: paid on every entry and exit
        spread_cost = (self.spread_bps / 10_000) * notional

        # Square-root market impact (Almgren-Chriss)
        adv = volume if (volume and volume > 0) else self.adv_shares
        participation = abs(shares) / adv
        slippage_cost = (self.slippage_bps / 10_000) * price * abs(shares) * np.sqrt(participation)

        # Commission: max of flat fee and percentage fee
        commission_cost = max(self.commission, self.pct_fee * notional)

        return spread_cost + slippage_cost + commission_cost

    @classmethod
    def zero(cls) -> "TransactionCostModel":
        """No transaction costs — for pure signal testing."""
        return cls(spread_bps=0, slippage_bps=0, commission=0, pct_fee=0)

    @classmethod
    def liquid_equity(cls) -> "TransactionCostModel":
        """Typical costs for S&P 500 stock at retail broker."""
        return cls(spread_bps=2.0, slippage_bps=1.0, commission=1.0, pct_fee=0.0005)

    @classmethod
    def small_cap(cls) -> "TransactionCostModel":
        """Higher costs for less liquid small-cap stocks."""
        return cls(spread_bps=10.0, slippage_bps=5.0, commission=1.0, pct_fee=0.001)


@dataclass
class Trade:
    """A completed round-trip trade."""
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    direction:   int           # +1 long, -1 short
    entry_price: float
    exit_price:  float
    shares:      float
    entry_cost:  float         # transaction cost at entry
    exit_cost:   float         # transaction cost at exit
    signal_col:  str = ""

    @property
    def pnl(self) -> float:
        """Net P&L after all transaction costs."""
        gross = self.direction * (self.exit_price - self.entry_price) * self.shares
        return gross - self.entry_cost - self.exit_cost

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of entry notional."""
        notional = self.entry_price * self.shares
        return self.pnl / notional if notional > 0 else 0.0

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResults:
    """
    Complete backtest output: equity curve, trades, and performance metrics.
    """
    equity_curve:  pd.Series           # daily portfolio value
    daily_returns: pd.Series           # daily P&L as fraction of equity
    positions:     pd.Series           # daily position: +1, -1, 0
    trades:        List[Trade]
    metrics:       Dict[str, float]
    signal_col:    str = ""
    ticker:        str = ""

    def summary(self) -> str:
        """One-line summary for quick comparison."""
        m = self.metrics
        return (
            f"{self.ticker or 'unknown':6s} | {self.signal_col:20s} | "
            f"Sharpe={m.get('sharpe',0):+.2f}  "
            f"Sortino={m.get('sortino',0):+.2f}  "
            f"Calmar={m.get('calmar',0):+.2f}  "
            f"MaxDD={m.get('max_drawdown',0):.1%}  "
            f"Hit={m.get('hit_rate',0):.1%}  "
            f"Alpha={m.get('alpha',0):+.3f}  "
            f"Trades={m.get('n_trades',0):.0f}"
        )


# ======================================================================
# Performance metrics engine
# ======================================================================

class PerformanceEngine:
    """
    Compute the full performance metrics suite from an equity curve and trade list.

    All metrics are annualised where applicable (252 trading days).
    """

    def compute(
        self,
        equity_curve:  pd.Series,
        daily_returns: pd.Series,
        trades:        List[Trade],
        benchmark:     Optional[pd.Series] = None,
        rf_annual:     float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            equity_curve:  Portfolio value over time
            daily_returns: Daily P&L / equity (fraction)
            trades:        List of completed Trade objects
            benchmark:     Benchmark daily returns (e.g. SPY) for alpha/beta
            rf_annual:     Annual risk-free rate (default 5% — 2024 T-bill rate)

        Returns:
            Dict of metric_name → float value
        """
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        ret = daily_returns.dropna()

        metrics: Dict[str, float] = {}

        # ---- Return metrics ----
        metrics["total_return"]    = self._total_return(equity_curve)
        metrics["annual_return"]   = self._annualise_return(ret)
        metrics["annual_vol"]      = float(ret.std() * np.sqrt(252))

        # ---- Risk-adjusted ----
        metrics["sharpe"]  = self._sharpe(ret, rf_daily)
        metrics["sortino"] = self._sortino(ret, rf_daily)
        metrics["calmar"]  = self._calmar(ret, equity_curve)

        # ---- Drawdown ----
        dd_series = self._drawdown_series(equity_curve)
        metrics["max_drawdown"]       = float(dd_series.min())
        metrics["avg_drawdown"]       = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0
        metrics["max_drawdown_days"]  = self._max_drawdown_duration(equity_curve)

        # ---- Trade statistics ----
        if trades:
            pnls = [t.pnl for t in trades]
            pnl_pcts = [t.pnl_pct for t in trades]
            winners = [p for p in pnls if p > 0]
            losers  = [p for p in pnls if p <= 0]
            metrics["n_trades"]     = float(len(trades))
            metrics["hit_rate"]     = float(len(winners) / len(trades))
            metrics["avg_win"]      = float(np.mean(winners)) if winners else 0.0
            metrics["avg_loss"]     = float(np.mean(losers))  if losers  else 0.0
            metrics["profit_factor"]= float(
                sum(winners) / abs(sum(losers))
            ) if losers and sum(losers) != 0 else np.inf
            metrics["avg_hold_days"]= float(np.mean([t.holding_days for t in trades]))
            metrics["avg_pnl_pct"]  = float(np.mean(pnl_pcts))
            metrics["total_costs"]  = float(sum(t.entry_cost + t.exit_cost for t in trades))
        else:
            for k in ["n_trades","hit_rate","avg_win","avg_loss",
                      "profit_factor","avg_hold_days","avg_pnl_pct","total_costs"]:
                metrics[k] = 0.0

        # ---- Tail risk ----
        metrics["var_95"]  = float(ret.quantile(0.05))
        metrics["cvar_95"] = float(ret[ret <= ret.quantile(0.05)].mean()) if len(ret) > 20 else 0.0

        # ---- Turnover ----
        metrics["turnover"] = self._turnover(daily_returns.index, trades)

        # ---- Alpha / Beta (CAPM) ----
        if benchmark is not None and len(benchmark.dropna()) > 30:
            alpha, beta, r2 = self._capm(ret, benchmark.reindex(ret.index).dropna(), rf_daily)
            metrics["alpha"] = float(alpha)
            metrics["beta"]  = float(beta)
            metrics["r2"]    = float(r2)
        else:
            metrics["alpha"] = 0.0
            metrics["beta"]  = 1.0
            metrics["r2"]    = 0.0

        return metrics

    # ------------------------------------------------------------------ #
    # Individual metric implementations                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _total_return(equity: pd.Series) -> float:
        """(final - initial) / initial"""
        if len(equity) < 2 or equity.iloc[0] == 0:
            return 0.0
        return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])

    @staticmethod
    def _annualise_return(ret: pd.Series) -> float:
        """Compound annual growth rate from daily returns."""
        if len(ret) == 0:
            return 0.0
        total = (1 + ret).prod()
        n_years = len(ret) / 252
        if n_years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    @staticmethod
    def _sharpe(ret: pd.Series, rf_daily: float) -> float:
        """
        Sharpe ratio = (mean_excess_return / std) * sqrt(252)

        Uses daily excess returns. Annualises by sqrt(252).
        A Sharpe > 1.0 is considered good. > 2.0 is exceptional.
        """
        excess = ret - rf_daily
        std = excess.std()
        if std < 1e-10:   # guard against floating-point near-zero for constant series
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    @staticmethod
    def _sortino(ret: pd.Series, rf_daily: float) -> float:
        """
        Sortino ratio = (mean_excess_return / downside_std) * sqrt(252)

        Uses only negative returns for the denominator (downside deviation).
        Penalises harmful volatility only — preferred over Sharpe for
        strategies with skewed return distributions.
        """
        excess = ret - rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(252))

    @staticmethod
    def _calmar(ret: pd.Series, equity: pd.Series) -> float:
        """
        Calmar ratio = annualised_return / |max_drawdown|

        Measures return per unit of worst-case drawdown risk.
        Preferred by CTA funds for trend-following strategies.
        Calmar > 1.0 is good; > 3.0 is exceptional.
        """
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        if max_dd == 0:
            return np.inf
        total = (1 + ret).prod()
        ann_ret = float(total ** (252 / len(ret)) - 1) if total > 0 else 0.0
        return float(ann_ret / max_dd)

    @staticmethod
    def _drawdown_series(equity: pd.Series) -> pd.Series:
        """Rolling drawdown from all-time high."""
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max.replace(0, np.nan)

    @staticmethod
    def _max_drawdown_duration(equity: pd.Series) -> float:
        """Longest period (days) from peak to recovery."""
        rolling_max = equity.cummax()
        underwater = equity < rolling_max
        if not underwater.any():
            return 0.0
        # Count consecutive underwater periods
        max_dur = 0
        cur_dur = 0
        for val in underwater:
            if val:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return float(max_dur)

    @staticmethod
    def _capm(
        strategy_ret: pd.Series,
        benchmark_ret: pd.Series,
        rf_daily: float,
    ) -> Tuple[float, float, float]:
        """
        CAPM regression: r_strategy - rf = alpha + beta * (r_benchmark - rf) + epsilon

        Returns: (alpha_annualised, beta, r_squared)

        Alpha > 0: strategy outperforms the market after adjusting for beta exposure.
        Beta > 1: strategy amplifies market moves.
        Beta < 0: strategy is market-neutral or inverse.
        R² close to 1: strategy returns are largely explained by market beta.
        """
        common = strategy_ret.index.intersection(benchmark_ret.index)
        if len(common) < 30:
            return 0.0, 1.0, 0.0

        y = strategy_ret.loc[common] - rf_daily
        x = benchmark_ret.loc[common] - rf_daily

        # OLS: beta = cov(x,y) / var(x), alpha = mean(y) - beta * mean(x)
        beta  = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 0 else 0.0
        alpha_daily = float(y.mean() - beta * x.mean())
        alpha_annual = float((1 + alpha_daily) ** 252 - 1)  # annualise

        # R²
        y_pred = alpha_daily + beta * x
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return alpha_annual, beta, r2

    @staticmethod
    def _turnover(index: pd.Index, trades: List[Trade]) -> float:
        """
        Annualised turnover: average number of complete portfolio turns per year.
        turnover = (total_shares_traded / 2) / avg_position_size / 252 * n_bars
        Simplified: n_trades / n_bars * 252 (trades per year).
        """
        if len(index) == 0:
            return 0.0
        n_bars = len(index)
        n_trades = len(trades)
        return float(n_trades / n_bars * 252)


# ======================================================================
# Vectorised Backtester
# ======================================================================

class VectorisedBacktester:
    """
    Fast pandas/numpy backtester. Entire simulation in O(n) vector ops.

    Execution assumptions:
        - Signal at bar t → order placed at close of bar t
        - Fill at OPEN of bar t+1 (execution lag = 1 bar)
        - No partial fills, no market impact on position size
        - Position size = (capital * position_scale) / price

    Transaction costs applied at each position change (entry/exit/flip).

    Best for:
        - Initial signal validation
        - Parameter sweeps (hundreds of runs)
        - Comparing multiple signals quickly
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_sizing: Literal["fixed_shares", "fixed_notional", "vol_target"] = "fixed_notional",
        target_notional: float = 100_000.0,
        vol_target: float = 0.10,    # 10% annual vol target
        cost_model: Optional[TransactionCostModel] = None,
        rf_annual: float = 0.05,
    ):
        """
        Args:
            initial_capital:  Starting portfolio value
            position_sizing:  How to size positions:
                              'fixed_notional': always invest target_notional
                              'fixed_shares':   always trade 100 shares
                              'vol_target':     size to hit target annual vol
            target_notional:  Notional per position (fixed_notional mode)
            vol_target:       Target annual portfolio volatility (vol_target mode)
            cost_model:       TransactionCostModel instance (default: liquid_equity)
            rf_annual:        Annual risk-free rate for Sharpe/Sortino
        """
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.target_notional = target_notional
        self.vol_target      = vol_target
        self.cost_model      = cost_model or TransactionCostModel.liquid_equity()
        self.rf_annual       = rf_annual
        self.perf            = PerformanceEngine()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal_ensemble",
        price_col: str = "close",
        volume_col: str = "volume",
        scale_col: Optional[str] = "position_scale",
        benchmark_returns: Optional[pd.Series] = None,
        ticker: str = "",
    ) -> BacktestResults:
        """
        Run vectorised backtest on a single ticker.

        Args:
            df:                 Feature + signal DataFrame
            signal_col:         Which signal column to trade
            price_col:          Price column for fills and P&L
            volume_col:         Volume column for slippage model
            scale_col:          Position scale column (None = ignore)
            benchmark_returns:  Daily returns of benchmark for alpha/beta
            ticker:             Label for results

        Returns:
            BacktestResults with equity curve, trades, and metrics
        """
        self._validate_inputs(df, signal_col, price_col)

        # ---- Execution lag: signal at t → fill at t+1 open ----
        # Approximate next-bar open as current close (common in daily backtests)
        signal  = df[signal_col].fillna(0).astype(int)
        price   = df[price_col]
        volume  = df[volume_col] if volume_col in df.columns else pd.Series(
            self.cost_model.adv_shares, index=df.index
        )
        scale   = df[scale_col].fillna(1.0) if (scale_col and scale_col in df.columns) else pd.Series(1.0, index=df.index)

        # Shift signal by 1: trade at next bar's price
        signal_lagged = signal.shift(1).fillna(0).astype(int)
        scale_lagged  = scale.shift(1).fillna(1.0)

        # ---- Position sizing ----
        shares_series = self._compute_shares(price, signal_lagged, scale_lagged, df)

        # ---- Position changes (where trades happen) ----
        prev_shares = shares_series.shift(1).fillna(0)
        delta_shares = shares_series - prev_shares  # non-zero = trade

        # ---- Transaction costs ----
        cost_series = pd.Series(0.0, index=df.index)
        for i in df.index[delta_shares != 0]:
            cost_series[i] = self.cost_model.total_cost(
                price=float(price[i]),
                shares=float(abs(delta_shares[i])),
                volume=float(volume[i]),
            )

        # ---- P&L calculation ----
        # Daily P&L = shares * (price_t - price_{t-1}) - costs_t
        price_change  = price.diff().fillna(0)
        daily_pnl     = shares_series * price_change - cost_series

        # ---- Equity curve ----
        equity = pd.Series(self.initial_capital, index=df.index)
        equity = self.initial_capital + daily_pnl.cumsum()

        daily_returns = daily_pnl / equity.shift(1).fillna(self.initial_capital)

        # ---- Extract trades ----
        trades = self._extract_trades(
            signal_lagged, price, shares_series,
            cost_series, delta_shares, ticker, signal_col
        )

        # ---- Metrics ----
        metrics = self.perf.compute(
            equity_curve=equity,
            daily_returns=daily_returns,
            trades=trades,
            benchmark=benchmark_returns,
            rf_annual=self.rf_annual,
        )

        logger.info(
            f"[{ticker}] Vectorised backtest complete: "
            f"Sharpe={metrics.get('sharpe', 0):.2f}  "
            f"MaxDD={metrics.get('max_drawdown', 0):.1%}  "
            f"Trades={metrics.get('n_trades', 0):.0f}"
        )

        return BacktestResults(
            equity_curve=equity,
            daily_returns=daily_returns,
            positions=signal_lagged,
            trades=trades,
            metrics=metrics,
            signal_col=signal_col,
            ticker=ticker,
        )

    def compare_signals(
        self,
        df: pd.DataFrame,
        signal_cols: Optional[List[str]] = None,
        price_col: str = "close",
        benchmark_returns: Optional[pd.Series] = None,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Run backtest for every signal column and return a comparison table.

        Args:
            df:               Feature + signal DataFrame
            signal_cols:      Columns to test (auto-detects signal_* if None)
            price_col:        Price column
            benchmark_returns: Benchmark for alpha/beta

        Returns:
            DataFrame with one row per signal and all metrics as columns
        """
        if signal_cols is None:
            signal_cols = [
                c for c in df.columns
                if c.startswith("signal_") and not c.endswith("_strength")
            ]

        rows = []
        for col in signal_cols:
            try:
                result = self.run(
                    df, signal_col=col, price_col=price_col,
                    benchmark_returns=benchmark_returns, ticker=ticker,
                )
                row = {"signal": col, **result.metrics}
                rows.append(row)
                logger.info(result.summary())
            except Exception as exc:
                logger.warning(f"Signal '{col}' failed: {exc}")

        if not rows:
            return pd.DataFrame()

        comparison = pd.DataFrame(rows).set_index("signal")

        # Sort by Sharpe descending
        if "sharpe" in comparison.columns:
            comparison = comparison.sort_values("sharpe", ascending=False)

        return comparison.round(4)

    # ------------------------------------------------------------------ #
    # Position sizing                                                      #
    # ------------------------------------------------------------------ #

    def _compute_shares(
        self,
        price: pd.Series,
        signal: pd.Series,
        scale: pd.Series,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Compute target share count at each bar."""
        if self.position_sizing == "fixed_shares":
            return signal * scale * 100  # 100 shares per signal unit

        elif self.position_sizing == "fixed_notional":
            safe_price = price.replace(0, np.nan)
            return signal * scale * (self.target_notional / safe_price)

        elif self.position_sizing == "vol_target":
            # Size inversely proportional to realised volatility
            # Target: portfolio vol = vol_target
            # shares = (capital * vol_target) / (price * vol_daily * sqrt(252))
            vol_col = "vol_21d" if "vol_21d" in df.columns else None
            if vol_col:
                vol = df[vol_col].replace(0, np.nan).fillna(0.02)  # default 2% daily vol
                safe_price = price.replace(0, np.nan)
                annual_vol = vol * np.sqrt(252)
                notional = self.initial_capital * self.vol_target / annual_vol.clip(lower=0.01)
                return signal * scale * (notional / safe_price)
            else:
                # Fallback to fixed notional
                safe_price = price.replace(0, np.nan)
                return signal * scale * (self.target_notional / safe_price)

        raise ValueError(f"Unknown position_sizing: '{self.position_sizing}'")

    # ------------------------------------------------------------------ #
    # Trade extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_trades(
        self,
        signal: pd.Series,
        price: pd.Series,
        shares: pd.Series,
        costs: pd.Series,
        delta: pd.Series,
        ticker: str,
        signal_col: str,
    ) -> List[Trade]:
        """
        Reconstruct completed round-trip trades from position series.
        A trade opens when position goes from 0 → non-zero.
        A trade closes when position goes to 0 or reverses direction.
        """
        trades: List[Trade] = []
        in_trade = False
        entry_date  = None
        entry_price = 0.0
        entry_dir   = 0
        entry_cost  = 0.0
        entry_shares = 0.0

        dates = signal.index
        for i, date in enumerate(dates):
            pos      = int(signal.iloc[i])
            prev_pos = int(signal.iloc[i - 1]) if i > 0 else 0
            p        = float(price.iloc[i])
            c        = float(costs.iloc[i])
            sh       = abs(float(shares.iloc[i]))

            if not in_trade and pos != 0:
                # Entry
                in_trade    = True
                entry_date  = date
                entry_price = p
                entry_dir   = pos
                entry_cost  = c
                entry_shares = sh

            elif in_trade and (pos == 0 or pos != entry_dir):
                # Exit or reversal
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    direction=entry_dir,
                    entry_price=entry_price,
                    exit_price=p,
                    shares=entry_shares,
                    entry_cost=entry_cost,
                    exit_cost=c,
                    signal_col=signal_col,
                ))

                # If reversal, open new trade immediately
                if pos != 0:
                    in_trade    = True
                    entry_date  = date
                    entry_price = p
                    entry_dir   = pos
                    entry_cost  = c
                    entry_shares = abs(float(shares.iloc[i]))
                else:
                    in_trade = False

        # Close any open trade at end of series
        if in_trade and entry_date is not None:
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=dates[-1],
                direction=entry_dir,
                entry_price=entry_price,
                exit_price=float(price.iloc[-1]),
                shares=entry_shares,
                entry_cost=entry_cost,
                exit_cost=0.0,  # open position, no exit cost yet
                signal_col=signal_col,
            ))

        return trades

    @staticmethod
    def _validate_inputs(
        df: pd.DataFrame, signal_col: str, price_col: str
    ) -> None:
        missing = [c for c in [signal_col, price_col] if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns: {missing}. Available: {list(df.columns)}"
            )


# ======================================================================
# Event-Driven Backtester
# ======================================================================

class EventDrivenBacktester:
    """
    Bar-by-bar simulation with explicit order lifecycle.

    Event sequence per bar:
        1. Market open  → fill any pending orders from yesterday
        2. Data arrives → update portfolio mark-to-market
        3. Signal fires → generate order for next bar
        4. Market close → record equity, P&L

    More realistic than vectorised because:
        - Enforces 1-bar execution lag (no forward-looking fills)
        - Applies slippage on the actual fill price, not close
        - Tracks unrealised P&L separately from realised P&L
        - Supports partial fill simulation (future extension)

    Use after vectorised backtester confirms a signal is worth testing.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_sizing: Literal["fixed_shares", "fixed_notional", "vol_target"] = "fixed_notional",
        target_notional: float = 100_000.0,
        vol_target: float = 0.10,
        cost_model: Optional[TransactionCostModel] = None,
        rf_annual: float = 0.05,
        execution_price: Literal["open", "close", "vwap"] = "open",
    ):
        """
        Args:
            execution_price: Which price to use for fills:
                             'open'  — next bar open (realistic, default)
                             'close' — same bar close (optimistic)
                             'vwap'  — midpoint of high+low (approximation)
        """
        self.initial_capital  = initial_capital
        self.position_sizing  = position_sizing
        self.target_notional  = target_notional
        self.vol_target       = vol_target
        self.cost_model       = cost_model or TransactionCostModel.liquid_equity()
        self.rf_annual        = rf_annual
        self.execution_price  = execution_price
        self.perf             = PerformanceEngine()

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal_ensemble",
        price_col: str = "close",
        volume_col: str = "volume",
        scale_col:  Optional[str] = "position_scale",
        benchmark_returns: Optional[pd.Series] = None,
        ticker: str = "",
    ) -> BacktestResults:
        """
        Run event-driven backtest on a single ticker.

        Returns identical interface as VectorisedBacktester.run()
        for easy comparison.
        """
        self._validate_inputs(df, signal_col)

        # ---- Portfolio state ----
        # We use a cash-position model:
        #   equity[t] = cash[t] + position[t] * close_price[t]
        # Daily P&L = position[t-1] * (close[t] - close[t-1]) - transaction_costs[t]
        # This correctly captures overnight moves and avoids the cost_basis reset bug.

        cash          = self.initial_capital  # cash not invested in positions
        position      = 0.0      # shares held (signed, float)
        pending_sig   = 0        # signal from yesterday → order today
        pending_scale = 1.0
        prev_close    = 0.0      # previous bar close (for overnight P&L)

        equity_list     = []
        daily_ret_list  = []
        position_list   = []
        trades: List[Trade] = []

        entry_date    = None
        entry_price   = 0.0
        entry_dir     = 0
        entry_cost_   = 0.0
        entry_shares_ = 0.0

        for i, (date, row) in enumerate(df.iterrows()):
            close_price = float(row.get(price_col, 0))
            fill_price  = self._get_fill_price(row, df, i)
            vol = float(row.get(volume_col, self.cost_model.adv_shares))

            # ---- Step 1: Fill pending order at today's fill price ----
            trade_cost = 0.0
            if pending_sig != 0 or (pending_sig == 0 and position != 0):
                target_shares = self._target_shares(
                    pending_sig, pending_scale, fill_price, cash, row
                )
                delta = target_shares - position

                if abs(delta) > 0.01:
                    trade_cost = self.cost_model.total_cost(fill_price, delta, vol)

                    # Direction change or close: record completed trade
                    if position != 0 and (
                        target_shares == 0 or
                        np.sign(target_shares) != np.sign(position)
                    ):
                        if entry_date is not None:
                            exit_cost = self.cost_model.total_cost(
                                fill_price, abs(position), vol
                            )
                            trades.append(Trade(
                                entry_date=entry_date,
                                exit_date=date,
                                direction=entry_dir,
                                entry_price=entry_price,
                                exit_price=fill_price,
                                shares=abs(entry_shares_),
                                entry_cost=entry_cost_,
                                exit_cost=exit_cost,
                                signal_col=signal_col,
                            ))
                            entry_date = None

                    # Open new entry
                    if target_shares != 0 and (
                        position == 0 or np.sign(target_shares) != np.sign(position)
                    ):
                        entry_date    = date
                        entry_price   = fill_price
                        entry_dir     = int(np.sign(target_shares))
                        entry_cost_   = trade_cost
                        entry_shares_ = target_shares

                    # Update cash: buy increases cash outflow; sell brings cash in
                    # cash change = -delta * fill_price (negative delta = sell = cash in)
                    cash    -= delta * fill_price + trade_cost
                    position = target_shares

            # ---- Step 2: Mark-to-market at close ----
            # equity = cash + current market value of position
            current_equity = cash + position * close_price

            if i == 0:
                daily_ret = 0.0
                prev_eq   = current_equity
            else:
                prev_eq = equity_list[-1]
                daily_ret = (current_equity - prev_eq) / prev_eq if prev_eq > 0 else 0.0

            equity_list.append(current_equity)
            daily_ret_list.append(daily_ret)
            position_list.append(int(np.sign(position)))
            prev_close = close_price

            # ---- Step 3: Capture tomorrow's order ----
            sig = int(row.get(signal_col, 0)) if not pd.isna(row.get(signal_col, 0)) else 0
            sc  = float(row.get(scale_col, 1.0)) if (scale_col and scale_col in df.columns) else 1.0
            pending_sig   = sig
            pending_scale = sc

        # Close any open position at end of series
        if entry_date is not None and position != 0:
            last_price = float(df[price_col].iloc[-1])
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=df.index[-1],
                direction=entry_dir,
                entry_price=entry_price,
                exit_price=last_price,
                shares=abs(entry_shares_),
                entry_cost=entry_cost_,
                exit_cost=0.0,
                signal_col=signal_col,
            ))

        # ---- Build result Series ----
        equity_curve  = pd.Series(equity_list,    index=df.index)
        daily_returns = pd.Series(daily_ret_list, index=df.index)
        positions     = pd.Series(position_list,  index=df.index)

        metrics = self.perf.compute(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=trades,
            benchmark=benchmark_returns,
            rf_annual=self.rf_annual,
        )

        logger.info(
            f"[{ticker}] Event-driven backtest complete: "
            f"Sharpe={metrics.get('sharpe',0):.2f}  "
            f"MaxDD={metrics.get('max_drawdown',0):.1%}  "
            f"Trades={metrics.get('n_trades',0):.0f}"
        )

        return BacktestResults(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            signal_col=signal_col,
            ticker=ticker,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_fill_price(
        self, row: pd.Series, df: pd.DataFrame, i: int
    ) -> float:
        """Return execution price based on execution_price setting."""
        if self.execution_price == "open" and "open" in row.index:
            return float(row["open"])
        elif self.execution_price == "vwap" and "high" in row.index and "low" in row.index:
            return float((row["high"] + row["low"]) / 2)
        return float(row.get("close", 0))

    def _target_shares(
        self,
        signal: int,
        scale: float,
        price: float,
        capital: float,
        row: pd.Series,
    ) -> float:
        """Compute target position size in shares."""
        if signal == 0 or price == 0:
            return 0.0
        if self.position_sizing == "fixed_shares":
            return float(signal * scale * 100)
        elif self.position_sizing == "fixed_notional":
            return float(signal * scale * self.target_notional / price)
        elif self.position_sizing == "vol_target":
            vol = float(row.get("vol_21d", 0.02))
            annual_vol = max(vol * np.sqrt(252), 0.01)
            notional = capital * self.vol_target / annual_vol
            return float(signal * scale * notional / price)
        return 0.0

    @staticmethod
    def _validate_inputs(df: pd.DataFrame, signal_col: str) -> None:
        if signal_col not in df.columns:
            raise KeyError(
                f"Signal column '{signal_col}' not found. "
                f"Available: {list(df.columns)}"
            )


# ======================================================================
# Convenience function: run both and compare
# ======================================================================

def run_both(
    df: pd.DataFrame,
    signal_col: str = "signal_ensemble",
    initial_capital: float = 100_000.0,
    cost_model: Optional[TransactionCostModel] = None,
    benchmark_returns: Optional[pd.Series] = None,
    ticker: str = "",
) -> Dict[str, BacktestResults]:
    """
    Run both vectorised and event-driven backtests and return both results.

    The difference in metrics between the two engines is a measure of
    implementation realism:
        - Sharpe higher in vectorised than event-driven → slippage matters
        - Large difference in n_trades → order lag is significant
        - Same metrics → signal is robust to execution assumptions

    Returns:
        Dict with keys 'vectorised' and 'event_driven'
    """
    cost = cost_model or TransactionCostModel.liquid_equity()

    vbt = VectorisedBacktester(
        initial_capital=initial_capital,
        cost_model=cost,
    )
    edb = EventDrivenBacktester(
        initial_capital=initial_capital,
        cost_model=cost,
    )

    v_result = vbt.run(df, signal_col=signal_col,
                       benchmark_returns=benchmark_returns, ticker=ticker)
    e_result = edb.run(df, signal_col=signal_col,
                       benchmark_returns=benchmark_returns, ticker=ticker)

    logger.info("── Comparison: Vectorised vs Event-Driven ──────────────────")
    logger.info(f"  Vectorised:    {v_result.summary()}")
    logger.info(f"  Event-Driven:  {e_result.summary()}")

    return {"vectorised": v_result, "event_driven": e_result}