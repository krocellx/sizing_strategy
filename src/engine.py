"""
Backtest engine. Applies a StopRule to a matrix of simulated returns
and produces equity curves + summary statistics.

Vectorized over paths where possible; per-day loop is unavoidable because
stop rules are path-dependent (state depends on history).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .stop_rules import StopRule, NoStop, VolScaledTrailingStop, RatioVolScaledTrailingStop


@dataclass
class BacktestResult:
    strategy_name: str
    rule_name: str
    equity_curves: np.ndarray   # (n_paths, path_length + 1), starts at initial_capital
    position_sizes: np.ndarray  # (n_paths, path_length), size used each day
    initial_capital: float
    vol_mult_log: np.ndarray | None = None   # (n_paths, path_length), vol-scaled rules
    transaction_cost_bps: float = 0.0        # one-way cost applied, for reference
    cash_flows: np.ndarray | None = None     # (n_paths, n_quarters), NaN if no reset
    quarterly_reset: bool = False            # whether quarterly reset was applied
    reset_every_days: int = 63

    @property
    def n_days(self) -> int:
        return self.equity_curves.shape[1] - 1

    @property
    def years(self) -> float:
        return self.n_days / 252

    @property
    def total_cash_flows(self) -> np.ndarray:
        """Sum all cash flows per path. Zero when no cash-flow reset is used."""
        if self.cash_flows is None:
            return np.zeros(len(self.equity_curves))
        return np.nansum(self.cash_flows, axis=1)

    @property
    def cumulative_cash_flow_curves(self) -> np.ndarray:
        """Cumulative extracted cash through each day, shape (n_paths, n_days+1)."""
        n_paths, n_days_plus1 = self.equity_curves.shape
        cumcash = np.zeros((n_paths, n_days_plus1))
        if self.cash_flows is None or not self.quarterly_reset:
            return cumcash

        n_quarters = self.cash_flows.shape[1]
        for q_idx in range(n_quarters):
            t = (q_idx + 1) * self.reset_every_days
            if t >= n_days_plus1:
                break
            cf = np.where(np.isfinite(self.cash_flows[:, q_idx]),
                          self.cash_flows[:, q_idx], 0.0)
            cumcash[:, t:] += cf[:, np.newaxis]

        return cumcash

    @property
    def cumulative_wealth_curves(self) -> np.ndarray:
        """
        Equity curve + cumulative cash flows extracted so far, shape (n_paths, n_days+1).

        When quarterly_reset=False, identical to equity_curves.
        When quarterly_reset=True, adds back the cash withdrawn each quarter
        so the curve represents the investor's true total wealth trajectory
        (what's in the fund + what's been taken out).
        """
        if self.cash_flows is None or not self.quarterly_reset:
            return self.equity_curves
        cumcash = self.cumulative_cash_flow_curves
        return self.equity_curves + cumcash

    @property
    def drawdown_curves(self) -> np.ndarray:
        """
        Reset-aware drawdown dollars for each path and day.

        With quarterly_reset=True, cash withdrawals are not treated as
        drawdowns. This is equivalent to computing drawdown on investor
        wealth (equity + extracted cash).
        """
        wealth = self.cumulative_wealth_curves
        hwm = np.maximum.accumulate(wealth, axis=1)
        return hwm - wealth

    @property
    def drawdown_pct_curves(self) -> np.ndarray:
        """
        Reset-aware drawdown percentage for each path and day.

        Dollar drawdown is measured on total investor wealth, while the
        denominator is the in-fund HWM after subtracting extracted cash. That
        keeps quarterly cash withdrawals from creating artificial drawdowns.
        """
        wealth = self.cumulative_wealth_curves
        wealth_hwm = np.maximum.accumulate(wealth, axis=1)
        in_fund_hwm = wealth_hwm - self.cumulative_cash_flow_curves
        dd = wealth_hwm - wealth
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(in_fund_hwm > 0, dd / in_fund_hwm, 0.0)

    @property
    def terminal_wealth(self) -> np.ndarray:
        """
        Total wealth per path = terminal equity + cumulative cash flows.

        When quarterly_reset=False, cash flows are zero so this equals
        the terminal equity exactly (no change from prior behaviour).

        When quarterly_reset=True, the equity curve is reset to initial_capital
        each profitable quarter — so terminal equity alone understates returns.
        Adding cumulative cash flows gives the investor's true total wealth.
        """
        return self.equity_curves[:, -1] + self.total_cash_flows

    @property
    def total_returns(self) -> np.ndarray:
        return self.terminal_wealth / self.initial_capital - 1.0

    @property
    def cagr(self) -> np.ndarray:
        return (self.terminal_wealth / self.initial_capital) ** (1 / self.years) - 1

    @property
    def max_drawdowns(self) -> np.ndarray:
        """
        Max drawdown per path in dollars, from running HWM.

        With quarterly_reset=True, the equity curve resets to initial_capital
        each profitable quarter, so the HWM also resets. Max drawdown here
        reflects the worst intra-quarter drawdown — consistent with the
        quarterly mandate (don't lose more than $X from quarterly start).
        """
        return self.drawdown_curves.max(axis=1)

    @property
    def max_drawdown_pct(self) -> np.ndarray:
        return self.drawdown_pct_curves.max(axis=1)

    @property
    def calmar(self) -> np.ndarray:
        dd_pct = self.max_drawdown_pct
        return np.where(dd_pct > 0, self.cagr / dd_pct, np.nan)

    @property
    def daily_equity_returns(self) -> np.ndarray:
        return np.diff(self.equity_curves, axis=1) / self.equity_curves[:, :-1]

    @property
    def sharpe(self) -> np.ndarray:
        daily = self.daily_equity_returns
        sd = daily.std(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(sd > 0, daily.mean(axis=1) / sd * np.sqrt(252), np.nan)

    def rolling_returns(
        self,
        window_days: int = 252,
        use_cumulative_wealth: bool = True,
    ) -> np.ndarray:
        """
        Rolling path returns across all start dates.

        use_cumulative_wealth=True reports investor total wealth, including
        extracted cash from quarterly resets. Set False for in-fund equity
        experience.
        """
        curve = self.cumulative_wealth_curves if use_cumulative_wealth else self.equity_curves
        if curve.shape[1] <= window_days:
            return np.empty((curve.shape[0], 0))
        return curve[:, window_days:] / curve[:, :-window_days] - 1.0

    def summary(self) -> pd.Series:
        tr = self.total_returns
        dd = self.max_drawdowns
        dd_pct = self.max_drawdown_pct
        s = pd.Series({
            'strategy': self.strategy_name,
            'rule': self.rule_name,
            'mean_total_return': tr.mean(),
            'median_total_return': np.median(tr),
            'p05_total_return': np.percentile(tr, 5),
            'p95_total_return': np.percentile(tr, 95),
            'mean_cagr': self.cagr.mean(),
            'mean_max_dd_$': dd.mean(),
            'p95_max_dd_$': np.percentile(dd, 95),
            'p99_max_dd_$': np.percentile(dd, 99),
            'mean_max_dd_pct': dd_pct.mean(),
            'p95_max_dd_pct': np.percentile(dd_pct, 95),
            'mean_sharpe': np.nanmean(self.sharpe),
            'prob_loss': (tr < 0).mean(),
            'prob_50pct_dd': (dd_pct > 0.5).mean(),
        })
        # Add cash flow summary when quarterly reset is active.
        if self.quarterly_reset and self.cash_flows is not None:
            s['mean_total_cash_out'] = self.total_cash_flows.mean()
            s['pct_quarters_positive'] = float(
                np.nanmean(self.cash_flows > 0)
            )
            s['mean_quarters_with_reset'] = float(
                np.isfinite(self.cash_flows).sum(axis=1).mean()
            )
        return s


def _apply_quarterly_reset(
    equity: np.ndarray,
    sizes: np.ndarray,
    initial_capital: float,
    reset_every_days: int = 63,
) -> np.ndarray:
    """
    Apply quarterly notional resets in-place and return cash flow matrix.

    Rule:
      - At the end of each quarter (every reset_every_days trading days),
        check the current position size and equity.
      - If size == 1.0 AND equity > initial_capital (profit quarter):
          * Record cash flow = equity - initial_capital (withdrawal)
          * Reset equity to initial_capital
      - Otherwise: do nothing. Specifically:
          * size < 1.0 or size == 0.0: stop is active, no reset
          * size == 1.0 but equity <= initial_capital: underperforming,
            no top-up. The loss is carried forward into the next quarter.

    Parameters
    ----------
    equity : (n_paths, n_days+1), modified IN PLACE
    sizes  : (n_paths, n_days)
    initial_capital : float
    reset_every_days : int, default 63 (~1 quarter)

    Returns
    -------
    cash_flows : (n_paths, n_quarters) — cash flow at each quarter-end.
        Positive = investor receives money (profit taken out).
        Negative = investor contributes money (loss topped up).
        NaN = no reset happened this quarter (size was reduced).
    """
    n_paths, n_days_plus1 = equity.shape
    n_days = n_days_plus1 - 1
    reset_days = list(range(reset_every_days - 1, n_days, reset_every_days))
    n_quarters = len(reset_days)

    cash_flows = np.full((n_paths, n_quarters), np.nan)

    for p in range(n_paths):
        for q_idx, t in enumerate(reset_days):
            size_at_reset = sizes[p, t]
            eq_before = equity[p, t + 1]
            if size_at_reset == 1.0 and eq_before > initial_capital:
                cf = eq_before - initial_capital
                cash_flows[p, q_idx] = cf
                delta = eq_before - initial_capital
                equity[p, t + 1:] -= delta

    return cash_flows


def _apply_transaction_costs(
    equity: np.ndarray,
    sizes: np.ndarray,
    cost: float,
) -> np.ndarray:
    """
    Apply one-way transaction costs to an equity curve array in-place.

    Two cost events per path:
      1. Intra-period: on every day where position size changes, deduct
         |Δsize| × equity_eod × cost from equity. Applied end-of-day
         (after the day's return, before recording equity[t+1]).
      2. Terminal: on the final day, deduct current_size × terminal_equity
         × cost to reflect liquidation of the remaining position. This
         levels the playing field: NoStop pays its exit cost at the end,
         stop rules that already exited (size=0) pay nothing extra.

    Parameters
    ----------
    equity : (n_paths, path_length+1) — modified IN PLACE
    sizes  : (n_paths, path_length)
    cost   : one-way cost as a fraction (e.g. 0.0005 for 5bps)

    Returns
    -------
    equity (same array, modified in place for efficiency)
    """
    if cost == 0.0:
        return equity

    n_paths, n_days = sizes.shape

    # --- Intra-period costs ---
    # Δsize[t] = sizes[t] - sizes[t-1], with sizes[-1] = 1.0 (full at start).
    prev_sizes = np.concatenate(
        [np.ones((n_paths, 1)), sizes[:, :-1]], axis=1
    )
    delta = np.abs(sizes - prev_sizes)          # (n_paths, n_days)
    # Equity after day t's return is equity[:, t+1] before cost adjustment.
    # Cost is applied to that post-return equity.
    cost_amount = delta * equity[:, 1:] * cost  # (n_paths, n_days)

    # Propagate: a cost on day t reduces equity on all subsequent days
    # because the compounding base is lower. We do this correctly by
    # working forward: subtract cost from equity[t+1], then let the
    # subsequent returns compound from the reduced base.
    # Vectorised forward propagation using cumulative cost ratios:
    #   equity_net[t] = equity_gross[t] × ∏_{s≤t} (1 - cost_s/equity_gross[s])
    # But that's O(n²) per path. Instead, adjust equity in-place iteratively
    # using a running cost multiplier — cheap because cost events are sparse.
    for t in range(n_days):
        col_cost = cost_amount[:, t]            # (n_paths,)
        has_cost = col_cost > 0
        if has_cost.any():
            equity[has_cost, t + 1:] -= col_cost[has_cost, np.newaxis]

    # --- Terminal liquidation cost ---
    # Everyone pays cost to unwind their remaining position on the last day.
    # Stop rules already at size=0 pay nothing (they liquidated during the run).
    terminal_sizes = sizes[:, -1]               # (n_paths,)
    terminal_equity = equity[:, -1]             # (n_paths,)
    terminal_cost = terminal_sizes * terminal_equity * cost
    equity[:, -1] -= terminal_cost

    return equity


def run_backtest(
    strategy_returns_paths: np.ndarray,
    rule: StopRule,
    strategy_name: str,
    initial_capital: float = 10_000_000.0,
    transaction_cost_bps: float = 0.0,
    quarterly_reset: bool = False,
    reset_every_days: int = 63,
) -> BacktestResult:
    """
    Apply a stop rule to a matrix of simulated returns.

    Parameters
    ----------
    strategy_returns_paths : np.ndarray, shape (n_paths, path_length)
        Daily returns for one strategy across all simulated paths.
    rule : StopRule
        The position-sizing / stop rule to apply.
    strategy_name : str
        For labeling results.
    initial_capital : float
        Starting equity per path (default $10m).
    transaction_cost_bps : float
        One-way transaction cost in basis points (default 0).
        Applied on every size change (intra-period) and on the final
        day's remaining position (terminal liquidation). 5bps = 0.05.
    quarterly_reset : bool
        If True, reset notional to initial_capital at the end of each
        quarter (every reset_every_days). Only fires when size == 1.0
        (fully invested). Cash flows are tracked and stored in result.
    reset_every_days : int
        Trading days per quarter (default 63).

    Returns
    -------
    BacktestResult
    """
    n_paths, path_length = strategy_returns_paths.shape
    returns = np.ascontiguousarray(strategy_returns_paths, dtype=np.float64)
    cost = transaction_cost_bps / 10_000.0  # convert bps to fraction

    reset_days = list(range(reset_every_days - 1, path_length, reset_every_days))
    reset_day_to_quarter = {t: i for i, t in enumerate(reset_days)}

    def _empty_cash_flows() -> np.ndarray | None:
        if not quarterly_reset:
            return None
        return np.full((n_paths, len(reset_days)), np.nan)

    def _finalise(equity, sizes, vol_mult_log=None, cash_flows=None):
        """Apply post-processing and return BacktestResult."""
        _apply_transaction_costs(equity, sizes, cost)
        return BacktestResult(
            strategy_name=strategy_name,
            rule_name=rule.name,
            equity_curves=equity,
            position_sizes=sizes,
            initial_capital=initial_capital,
            vol_mult_log=vol_mult_log,
            transaction_cost_bps=transaction_cost_bps,
            cash_flows=cash_flows,
            quarterly_reset=quarterly_reset,
            reset_every_days=reset_every_days,
        )

    # Fast path: NoStop is trivial.
    if isinstance(rule, NoStop) and not quarterly_reset:
        equity = np.empty((n_paths, path_length + 1))
        equity[:, 0] = initial_capital
        equity[:, 1:] = initial_capital * np.cumprod(1.0 + returns, axis=1)
        sizes = np.ones((n_paths, path_length))
        return _finalise(equity, sizes)

    # Optimized rules own their compiled path implementation.
    fast_path = getattr(rule, "run_fast_path", None)
    if fast_path is not None and not quarterly_reset:
        try:
            equity, sizes, vol_mult_log = fast_path(returns, float(initial_capital))
            return _finalise(equity, sizes, vol_mult_log)
        except NotImplementedError:
            pass

    # Generic fallback: Python loop, works for any StopRule subclass.
    equity = np.empty((n_paths, path_length + 1))
    equity[:, 0] = initial_capital
    sizes = np.empty((n_paths, path_length))
    cash_flows = _empty_cash_flows()

    is_vol_scaled = isinstance(rule, (VolScaledTrailingStop, RatioVolScaledTrailingStop))
    vol_mult_log = np.empty((n_paths, path_length)) if is_vol_scaled else None

    for p in range(n_paths):
        rule.reset(initial_capital)
        eq = initial_capital
        size = 1.0
        for t in range(path_length):
            r = strategy_returns_paths[p, t]
            rule.observe_return(r)
            sizes[p, t] = size
            eq = eq * (1 + size * r)
            equity[p, t + 1] = eq
            if quarterly_reset and t in reset_day_to_quarter and size == 1.0 and eq > initial_capital:
                cash_flows[p, reset_day_to_quarter[t]] = eq - initial_capital
                eq = initial_capital
                equity[p, t + 1] = eq
                rule.reset_notional(initial_capital)
                size = 1.0
            else:
                size = rule.update(eq)
            if is_vol_scaled:
                vol_mult_log[p, t] = rule.current_vol_mult

    return _finalise(equity, sizes, vol_mult_log, cash_flows)


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side comparison table of multiple backtest results."""
    return pd.DataFrame([r.summary() for r in results])
