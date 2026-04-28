"""
Core analysis functions operating on BacktestResult objects.

Organized into four categories:
  1. Distribution comparisons  — percentile_table, cvar, cvar_table,
                                  plot_distribution_overlay
  2. Drawdown dynamics         — time_under_water, recovery_times,
                                  drawdown_summary
  3. Regime-conditional        — conditional_comparison
  4. Paired / significance     — paired_comparison, bootstrap_ci

Institutional / allocator-facing metrics and the four-panel visual one-pager
live in institutional.py. Sensitivity sweeps (L, rule params, capital) live
in sensitivity.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from .engine import BacktestResult


# ---------------------------------------------------------------------------
# 1. Distribution comparisons
# ---------------------------------------------------------------------------

def percentile_table(
    results: Sequence[BacktestResult],
    quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> pd.DataFrame:
    """
    Terminal return and max-drawdown percentiles across all paths, per result.
    Shows the shape of each distribution at a glance.
    """
    rows = []
    for r in results:
        tr = r.total_returns
        dd = r.max_drawdowns
        dd_pct = r.max_drawdown_pct
        row = {'strategy': r.strategy_name, 'rule': r.rule_name}
        for q in quantiles:
            row[f'tr_p{int(q*100):02d}'] = np.quantile(tr, q)
        for q in quantiles:
            row[f'dd_$_p{int(q*100):02d}'] = np.quantile(dd, q)
        for q in quantiles:
            row[f'dd_%_p{int(q*100):02d}'] = np.quantile(dd_pct, q)
        rows.append(row)
    return pd.DataFrame(rows)


def cvar(result: BacktestResult, alpha: float = 0.05) -> dict:
    """
    Conditional VaR: mean terminal return AND mean max DD conditional on
    being in the worst alpha fraction of paths (ranked by terminal return).
    The headline tail-risk number.
    """
    tr = result.total_returns
    dd = result.max_drawdowns
    dd_pct = result.max_drawdown_pct
    threshold = np.quantile(tr, alpha)
    mask = tr <= threshold
    return {
        'alpha': alpha,
        'cvar_total_return': tr[mask].mean(),
        'cvar_max_dd_$': dd[mask].mean(),
        'cvar_max_dd_pct': dd_pct[mask].mean(),
        'var_total_return': threshold,
        'n_paths_in_tail': int(mask.sum()),
    }


def cvar_table(results: Sequence[BacktestResult],
               alphas: Sequence[float] = (0.01, 0.05, 0.10)) -> pd.DataFrame:
    """CVaR across multiple alpha levels, per result."""
    rows = []
    for r in results:
        row = {'strategy': r.strategy_name, 'rule': r.rule_name}
        for a in alphas:
            c = cvar(r, a)
            row[f'cvar{int(a*100):02d}_tr'] = c['cvar_total_return']
            row[f'cvar{int(a*100):02d}_dd_$'] = c['cvar_max_dd_$']
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distribution_overlay(
    results: Sequence[BacktestResult],
    metric: str = 'total_returns',
    bins: int = 60,
    ax=None,
    title: str | None = None,
):
    """
    Overlay histograms of a chosen metric across results. Quick diagnostic
    for whether a stop shifts the left tail without killing the right tail.

    metric: 'total_returns', 'max_drawdowns', 'max_drawdown_pct', 'terminal_wealth'
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        data = getattr(r, metric)
        label = f"{r.strategy_name} / {r.rule_name}"
        ax.hist(data, bins=bins, alpha=0.45, density=True, label=label)
    ax.legend(fontsize=9)
    ax.set_xlabel(metric)
    ax.set_ylabel('density')
    ax.set_title(title or f'Distribution of {metric}')
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# 2. Drawdown dynamics
# ---------------------------------------------------------------------------

def time_under_water(result: BacktestResult) -> np.ndarray:
    """
    Per-path fraction of days where equity is below its running HWM.
    Returns an array of length n_paths.
    """
    eq = result.equity_curves
    hwm = np.maximum.accumulate(eq, axis=1)
    underwater = eq < hwm
    return underwater.mean(axis=1)


def recovery_times(result: BacktestResult,
                   dd_threshold_dollars: float) -> np.ndarray:
    """
    For paths that ever breached the given DD threshold: days from first
    breach to next new HWM (or np.inf if never recovered by end of path).

    Returns an array of recovery times for paths that breached.
    """
    eq = result.equity_curves
    n_paths, n_days = eq.shape
    hwm = np.maximum.accumulate(eq, axis=1)
    dd = hwm - eq

    recoveries = []
    for p in range(n_paths):
        breaches = np.where(dd[p] >= dd_threshold_dollars)[0]
        if len(breaches) == 0:
            continue
        first_breach = breaches[0]
        hwm_at_breach = hwm[p, first_breach]
        recovered = np.where(eq[p, first_breach:] > hwm_at_breach)[0]
        recoveries.append(recovered[0] if len(recovered) else np.inf)
    return np.array(recoveries, dtype=float)


def drawdown_summary(result: BacktestResult,
                     levels_dollars: Sequence[float]) -> pd.Series:
    """Combined drawdown dynamics: hit rates + time underwater + recovery."""
    tuw = time_under_water(result)
    out = {
        'strategy': result.strategy_name,
        'rule': result.rule_name,
        'mean_time_underwater': tuw.mean(),
        'p95_time_underwater': np.quantile(tuw, 0.95),
    }
    for l in levels_dollars:
        hit = (result.max_drawdowns >= l).mean()
        out[f'pct_paths_hit_${int(l):,}'] = hit
        if hit > 0:
            rec = recovery_times(result, l)
            finite = rec[np.isfinite(rec)]
            out[f'median_recovery_days_${int(l):,}'] = (
                float(np.median(finite)) if len(finite) else np.nan
            )
            out[f'pct_unrecovered_${int(l):,}'] = float(np.mean(~np.isfinite(rec)))
    return pd.Series(out)


# ---------------------------------------------------------------------------
# 3. Regime-conditional analysis
# ---------------------------------------------------------------------------

def _path_realized_vol(result: BacktestResult) -> np.ndarray:
    """Realized vol of each path's daily equity returns, annualized."""
    daily = np.diff(result.equity_curves, axis=1) / result.equity_curves[:, :-1]
    return daily.std(axis=1) * np.sqrt(252)


def _path_trend(result: BacktestResult) -> np.ndarray:
    """Path trend: total return as a simple directional proxy."""
    return result.total_returns


def _path_worst_30d(result: BacktestResult) -> np.ndarray:
    """Worst rolling 30-day return per path (a tail-stress indicator)."""
    eq = result.equity_curves
    n_paths, n_days = eq.shape
    if n_days < 31:
        return np.zeros(n_paths)
    roll_ret = eq[:, 30:] / eq[:, :-30] - 1
    return roll_ret.min(axis=1)


def conditional_comparison(
    treated: BacktestResult,
    baseline: BacktestResult,
    bucket_by: str = 'vol',
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    Compare treated vs baseline bucketed by a path characteristic.

    Both results must share the same underlying scenarios (same idx matrix).
    We bucket BASELINE paths by their characteristic, then compute stats
    within each bucket for BOTH results. This isolates where the stop helps
    vs hurts as a function of market conditions.

    bucket_by: 'vol' (baseline realized vol), 'trend' (baseline total return),
               'worst_30d' (baseline worst 30-day return)
    """
    if treated.equity_curves.shape != baseline.equity_curves.shape:
        raise ValueError("Treated and baseline must have same path structure.")

    if bucket_by == 'vol':
        key = _path_realized_vol(baseline)
    elif bucket_by == 'trend':
        key = _path_trend(baseline)
    elif bucket_by == 'worst_30d':
        key = _path_worst_30d(baseline)
    else:
        raise ValueError(f"Unknown bucket_by: {bucket_by}")

    edges = np.quantile(key, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    buckets = np.digitize(key, edges[1:-1])

    rows = []
    for b in range(n_buckets):
        mask = buckets == b
        if mask.sum() == 0:
            continue
        rows.append({
            'bucket': b,
            'bucket_range': f"[{edges[b]:.3f}, {edges[b+1]:.3f}]",
            'n_paths': int(mask.sum()),
            'baseline_mean_tr': baseline.total_returns[mask].mean(),
            'treated_mean_tr':  treated.total_returns[mask].mean(),
            'delta_mean_tr':    treated.total_returns[mask].mean()
                                - baseline.total_returns[mask].mean(),
            'baseline_mean_dd_$': baseline.max_drawdowns[mask].mean(),
            'treated_mean_dd_$':  treated.max_drawdowns[mask].mean(),
            'delta_mean_dd_$':    treated.max_drawdowns[mask].mean()
                                  - baseline.max_drawdowns[mask].mean(),
            'pct_treated_better': (treated.total_returns[mask]
                                   > baseline.total_returns[mask]).mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3b. Sleeve combination (multi-strategy portfolio, independent stops)
# ---------------------------------------------------------------------------

def combine_sleeves(
    results: dict,
    strategies: Sequence[str],
    rule_label: str,
    capitals: dict,
    combined_name: str = 'combined',
) -> BacktestResult:
    """
    Sum per-strategy equity curves into a combined 'sleeve' result.

    Each strategy is assumed to run independently with its own stop rule and
    its own capital allocation. The combined equity curve at time t is the
    sum of each sleeve's equity at time t, scaled to its target allocation.

    Valid only when all strategies share the same scenarios (same idx) —
    otherwise the summed curves wouldn't represent a coherent joint history.

    Parameters
    ----------
    results : dict[(strategy, rule_label) -> BacktestResult]
        The nested dict built by your backtest loop.
    strategies : sequence of strategy names to combine.
    rule_label : which rule's results to use across all strategies.
    capitals : dict[strategy -> initial dollars]. Target allocation per sleeve.
    combined_name : label for the output result.

    Returns
    -------
    BacktestResult with equity_curves = sum of (scaled) per-strategy curves.

    Notes
    -----
    - position_sizes in the returned result is a weighted-average of sleeve
      sizes (useful for reporting but not used by most analyses).
    - All standard analysis functions (percentile_table, cvar_table,
      institutional_summary, paired_comparison, bootstrap_ci, etc.) work
      directly on the returned BacktestResult.
    - The combined drawdown reflects natural diversification between sleeves:
      a bad day for one strategy may be offset by another, so the combined
      max DD is usually well below the sum of per-sleeve max DDs.
    """
    if not strategies:
        raise ValueError("strategies must be non-empty")
    missing = [s for s in strategies if (s, rule_label) not in results]
    if missing:
        raise KeyError(f"Missing results for strategies {missing} under rule "
                       f"'{rule_label}'")

    # Validate shape alignment — all sleeves must share the scenario grid.
    shapes = {results[(s, rule_label)].equity_curves.shape for s in strategies}
    if len(shapes) > 1:
        raise ValueError(f"Sleeves have mismatched shapes {shapes}; they must "
                         f"share the same scenarios.")

    total_equity = None
    total_sizes_weighted = None
    total_initial = 0.0
    for s in strategies:
        r = results[(s, rule_label)]
        cap = capitals[s]
        scale = cap / r.initial_capital
        scaled_eq = r.equity_curves * scale       # (n_paths, n_days + 1)
        weight = cap                               # weight per sleeve in $ terms
        if total_equity is None:
            total_equity = scaled_eq.copy()
            total_sizes_weighted = r.position_sizes * weight
        else:
            total_equity += scaled_eq
            total_sizes_weighted += r.position_sizes * weight
        total_initial += cap

    avg_sizes = total_sizes_weighted / total_initial

    return BacktestResult(
        strategy_name=combined_name,
        rule_name=rule_label,
        equity_curves=total_equity,
        position_sizes=avg_sizes,
        initial_capital=total_initial,
    )


# ---------------------------------------------------------------------------
# 4. Paired comparison and significance
# ---------------------------------------------------------------------------

def paired_comparison(
    treated: BacktestResult,
    baseline: BacktestResult,
) -> pd.Series:
    """
    Path-wise paired comparison: for each simulated path, how did treated
    differ from baseline? Only meaningful if both share the same scenarios.
    """
    if treated.total_returns.shape != baseline.total_returns.shape:
        raise ValueError("Paired comparison requires same scenario set.")

    diff_tr = treated.total_returns - baseline.total_returns
    diff_dd = treated.max_drawdowns - baseline.max_drawdowns

    return pd.Series({
        'strategy': treated.strategy_name,
        'comparison': f'{treated.rule_name} vs {baseline.rule_name}',
        'mean_diff_tr': diff_tr.mean(),
        'median_diff_tr': np.median(diff_tr),
        'pct_paths_treated_better_tr': (diff_tr > 0).mean(),
        'mean_diff_dd_$': diff_dd.mean(),
        'pct_paths_treated_less_dd': (diff_dd < 0).mean(),
        'max_path_gain': diff_tr.max(),
        'max_path_loss': diff_tr.min(),
    })


def bootstrap_ci(
    treated: BacktestResult,
    baseline: BacktestResult,
    metric: str = 'total_returns',
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.Series:
    """
    Bootstrap CI for the difference in MEAN of `metric` between treated and
    baseline. Paths are resampled PAIRED (same path index) so this accounts
    for the shared scenario set.

    If the CI crosses zero, there is no statistical evidence that the
    difference is non-zero — even if the point estimate looks decisive.
    """
    rng = np.random.default_rng(seed)
    a = getattr(treated, metric)
    b = getattr(baseline, metric)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires same scenario set.")
    n = len(a)
    point = a.mean() - b.mean()

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return pd.Series({
        'metric': metric,
        'point_estimate': point,
        f'ci_lo_{int((1-alpha)*100)}': lo,
        f'ci_hi_{int((1-alpha)*100)}': hi,
        'significant': bool((lo > 0) or (hi < 0)),
        'n_resamples': n_resamples,
    })