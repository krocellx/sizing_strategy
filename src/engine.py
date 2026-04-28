"""
Backtest engine. Applies a StopRule to a matrix of simulated returns
and produces equity curves + summary statistics.

Vectorized over paths where possible; per-day loop is unavoidable because
stop rules are path-dependent (state depends on history).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .stop_rules import StopRule, TrailingStopRule, NoStop

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# Numba-jitted fast path for TrailingStopRule (the common case).
# Generic StopRule subclasses fall back to the Python loop.
if _HAS_NUMBA:
    @njit(cache=True)
    def _trailing_stop_loop(returns, initial_capital,
                            level_dds, level_sizes, reentry_recovery):
        n_paths, n_days = returns.shape
        n_levels = level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))

        for p in range(n_paths):
            eq = initial_capital
            equity[p, 0] = eq
            hwm = eq
            trough = eq
            cur_size = 1.0
            cur_level = -1

            for t in range(n_days):
                sizes[p, t] = cur_size
                eq = eq * (1.0 + cur_size * returns[p, t])
                equity[p, t + 1] = eq

                if eq > hwm:
                    hwm = eq
                    trough = eq
                    cur_size = 1.0
                    cur_level = -1
                    continue

                if eq < trough:
                    trough = eq

                dd = hwm - eq
                triggered = -1
                for i in range(n_levels):
                    if dd >= level_dds[i]:
                        triggered = i
                    else:
                        break

                if triggered > cur_level:
                    cur_level = triggered
                    cur_size = level_sizes[triggered]
                    trough = eq
                elif reentry_recovery > 0.0 and cur_level >= 0:
                    if eq - trough >= reentry_recovery:
                        new_level = -1
                        for i in range(n_levels):
                            if dd >= level_dds[i]:
                                new_level = i
                            else:
                                break
                        if new_level < cur_level:
                            cur_level = new_level
                            cur_size = 1.0 if new_level == -1 else level_sizes[new_level]
                            trough = eq

        return equity, sizes


@dataclass
class BacktestResult:
    strategy_name: str
    rule_name: str
    equity_curves: np.ndarray   # (n_paths, path_length + 1), starts at initial_capital
    position_sizes: np.ndarray  # (n_paths, path_length), size used each day
    initial_capital: float

    @property
    def terminal_wealth(self) -> np.ndarray:
        return self.equity_curves[:, -1]

    @property
    def total_returns(self) -> np.ndarray:
        return self.terminal_wealth / self.initial_capital - 1.0

    @property
    def max_drawdowns(self) -> np.ndarray:
        """Max drawdown per path, in dollars (from running HWM)."""
        hwm = np.maximum.accumulate(self.equity_curves, axis=1)
        dd = hwm - self.equity_curves
        return dd.max(axis=1)

    @property
    def max_drawdown_pct(self) -> np.ndarray:
        hwm = np.maximum.accumulate(self.equity_curves, axis=1)
        dd_pct = (hwm - self.equity_curves) / hwm
        return dd_pct.max(axis=1)

    def summary(self) -> pd.Series:
        tr = self.total_returns
        dd = self.max_drawdowns
        dd_pct = self.max_drawdown_pct
        # Annualized metrics assume path_length is in business days.
        n_days = self.equity_curves.shape[1] - 1
        years = n_days / 252
        cagr = (self.terminal_wealth / self.initial_capital) ** (1 / years) - 1
        # Sharpe from per-path daily equity returns.
        daily_eq_returns = np.diff(self.equity_curves, axis=1) / self.equity_curves[:, :-1]
        sharpe = daily_eq_returns.mean(axis=1) / daily_eq_returns.std(axis=1) * np.sqrt(252)
        return pd.Series({
            'strategy': self.strategy_name,
            'rule': self.rule_name,
            'mean_total_return': tr.mean(),
            'median_total_return': np.median(tr),
            'p05_total_return': np.percentile(tr, 5),
            'p95_total_return': np.percentile(tr, 95),
            'mean_cagr': cagr.mean(),
            'mean_max_dd_$': dd.mean(),
            'p95_max_dd_$': np.percentile(dd, 95),
            'p99_max_dd_$': np.percentile(dd, 99),
            'mean_max_dd_pct': dd_pct.mean(),
            'p95_max_dd_pct': np.percentile(dd_pct, 95),
            'mean_sharpe': sharpe.mean(),
            'prob_loss': (tr < 0).mean(),
            'prob_50pct_dd': (dd_pct > 0.5).mean(),
        })


def run_backtest(
    strategy_returns_paths: np.ndarray,
    rule: StopRule,
    strategy_name: str,
    initial_capital: float = 10_000_000.0,
) -> BacktestResult:
    """
    Apply a stop rule to a matrix of simulated returns.

    Parameters
    ----------
    strategy_returns_paths : np.ndarray, shape (n_paths, path_length)
        Daily returns for one strategy across all simulated paths.
        Get this from simulation.generate_scenarios()['paths'][strategy_name].
    rule : StopRule
        The position-sizing / stop rule to apply.
    strategy_name : str
        For labeling results.
    initial_capital : float
        Starting equity per path (default $10m).

    Returns
    -------
    BacktestResult
    """
    n_paths, path_length = strategy_returns_paths.shape
    returns = np.ascontiguousarray(strategy_returns_paths, dtype=np.float64)

    # Fast path: NoStop is trivial.
    if isinstance(rule, NoStop):
        equity = np.empty((n_paths, path_length + 1))
        equity[:, 0] = initial_capital
        equity[:, 1:] = initial_capital * np.cumprod(1.0 + returns, axis=1)
        sizes = np.ones((n_paths, path_length))
        return BacktestResult(strategy_name, rule.name, equity, sizes, initial_capital)

    # Fast path: TrailingStopRule via numba.
    if _HAS_NUMBA and isinstance(rule, TrailingStopRule):
        sorted_levels = sorted(rule.levels, key=lambda x: x[0])
        level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        equity, sizes = _trailing_stop_loop(
            returns, float(initial_capital),
            level_dds, level_sizes, float(rule.reentry_recovery),
        )
        return BacktestResult(strategy_name, rule.name, equity, sizes, initial_capital)

    # Generic fallback: Python loop, works for any StopRule subclass.
    equity = np.empty((n_paths, path_length + 1))
    equity[:, 0] = initial_capital
    sizes = np.empty((n_paths, path_length))

    # We need to drive the rule per-path because internal state is per-path.
    # The OUTER loop is over paths; the INNER loop is over days. This is the
    # opposite of what you'd want for pure vectorization, but stop rules
    # are inherently sequential within a path.
    #
    # For 10k paths x 1260 days this is ~12.6M rule.update() calls. On modern
    # hardware that's a few seconds for simple rules. If it ever becomes a
    # bottleneck, the trailing-stop logic could be re-implemented in numba.
    for p in range(n_paths):
        rule.reset(initial_capital)
        eq = initial_capital
        size = 1.0
        for t in range(path_length):
            sizes[p, t] = size
            eq = eq * (1 + size * strategy_returns_paths[p, t])
            equity[p, t + 1] = eq
            size = rule.update(eq)

    return BacktestResult(
        strategy_name=strategy_name,
        rule_name=rule.name,
        equity_curves=equity,
        position_sizes=sizes,
        initial_capital=initial_capital,
    )


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side comparison table of multiple backtest results."""
    return pd.DataFrame([r.summary() for r in results])
