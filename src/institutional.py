"""
Institutional / allocator-focused analysis.

Metrics and plots that show up in DDQs, IC memos, and allocator conversations:
  - Rolling 1yr / 3yr return distributions
  - DD-threshold breach probabilities
  - Stop activity statistics (how often does it actually trigger?)
  - Equity fan chart
  - Drawdown fan chart
  - Terminal return vs max DD scatter
  - Path-wise "did the stop help" histogram
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from .engine import BacktestResult


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rolling_return_stats(
    result: BacktestResult,
    window_days: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Distribution of rolling N-day returns across all (path, start-day) pairs.

    Key institutional number: the worst rolling 1-year return investors could
    have experienced if they'd invested at the worst possible moment.
    Reports p01/p05/p25/median and the single worst across all paths.
    """
    eq = result.equity_curves
    n_paths, n_days_plus_1 = eq.shape
    n_days = n_days_plus_1 - 1
    if n_days < window_days:
        return pd.Series({'error': f'path_length {n_days} < window {window_days}'})

    # Rolling returns: eq[t + window] / eq[t] - 1, over all valid (path, t).
    rolling = eq[:, window_days:] / eq[:, :-window_days] - 1.0
    flat = rolling.ravel()

    label = 'y' if window_days == periods_per_year else f'{window_days}d'
    return pd.Series({
        'strategy': result.strategy_name,
        'rule': result.rule_name,
        f'roll{label}_mean': flat.mean(),
        f'roll{label}_median': np.median(flat),
        f'roll{label}_p01': np.quantile(flat, 0.01),
        f'roll{label}_p05': np.quantile(flat, 0.05),
        f'roll{label}_p25': np.quantile(flat, 0.25),
        f'roll{label}_worst_any_path': flat.min(),
        f'roll{label}_prob_negative': (flat < 0).mean(),
    })


def dd_threshold_probabilities(
    result: BacktestResult,
    thresholds_pct: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
) -> pd.Series:
    """
    Probability of ever breaching each max DD threshold (as % of HWM).

    These map directly to institutional mandate triggers:
      5-10% soft review levels, 15-20% hard stops, 30%+ catastrophic.
    """
    dd_pct = result.max_drawdown_pct
    out = {'strategy': result.strategy_name, 'rule': result.rule_name}
    for t in thresholds_pct:
        out[f'P(maxDD>{int(t*100)}%)'] = float((dd_pct > t).mean())
    return pd.Series(out)


def stop_activity(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    How often does the stop actually change position size?

    Counts per-year:
      - size reductions (cuts)
      - size increases (re-entries / recoveries to HWM)
      - full stop-outs (size went to 0)
      - average days at reduced size per year

    A stop that only triggers 0.2x/year is mostly dormant; one that triggers
    5x/year has meaningful frictional/execution cost.
    """
    sizes = result.position_sizes  # (n_paths, n_days)
    n_paths, n_days = sizes.shape
    years = n_days / periods_per_year

    # Size change points: compare against previous day (first day vs 1.0 baseline).
    prev = np.concatenate([np.ones((n_paths, 1)), sizes[:, :-1]], axis=1)
    cuts = (sizes < prev).sum(axis=1) / years
    raises = (sizes > prev).sum(axis=1) / years
    stopouts = ((sizes == 0.0) & (prev > 0.0)).sum(axis=1) / years
    days_reduced_per_yr = (sizes < 1.0).sum(axis=1) / years

    return pd.Series({
        'strategy': result.strategy_name,
        'rule': result.rule_name,
        'mean_cuts_per_yr': float(cuts.mean()),
        'mean_raises_per_yr': float(raises.mean()),
        'mean_full_stopouts_per_yr': float(stopouts.mean()),
        'mean_days_reduced_per_yr': float(days_reduced_per_yr.mean()),
        'p95_days_reduced_per_yr': float(np.quantile(days_reduced_per_yr, 0.95)),
        'prob_any_full_stopout': float((stopouts > 0).mean()),
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_fan(
    results: Sequence[BacktestResult],
    percentiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    ax=None,
    periods_per_year: int = 252,
    log_y: bool = False,
    title: str | None = None,
):
    """
    Equity fan chart: median path + shaded bands across paths, one color per rule.

    The single most effective visualization in an institutional setting.
    Shows expected trajectory AND how the distribution evolves over time.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        eq = r.equity_curves / r.initial_capital  # normalize to 1.0 start
        t = np.arange(eq.shape[1]) / periods_per_year
        c = colors[i % len(colors)]
        q_low = np.quantile(eq, percentiles[0], axis=0)
        q_lo = np.quantile(eq, percentiles[1], axis=0)
        q_md = np.quantile(eq, percentiles[2], axis=0)
        q_hi = np.quantile(eq, percentiles[3], axis=0)
        q_high = np.quantile(eq, percentiles[4], axis=0)
        ax.fill_between(t, q_low, q_high, color=c, alpha=0.12)
        ax.fill_between(t, q_lo, q_hi, color=c, alpha=0.25)
        ax.plot(t, q_md, color=c, lw=2, label=f'{r.strategy_name} / {r.rule_name}')

    ax.axhline(1.0, color='k', ls=':', alpha=0.4)
    ax.set_xlabel('Years')
    ax.set_ylabel('Equity (normalized)')
    ax.set_title(title or 'Equity fan chart (median, 25/75, 5/95)')
    if log_y:
        ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    return ax


def plot_drawdown_fan(
    results: Sequence[BacktestResult],
    percentiles: Sequence[float] = (0.50, 0.75, 0.95, 0.99),
    ax=None,
    periods_per_year: int = 252,
    title: str | None = None,
):
    """
    Underwater fan chart: drawdown from running HWM over time, by percentile.

    Reads very naturally to investors — 'in a bad year (p95), you'd be down X%
    for Y days'. Pairs with the equity fan to show 'time underwater' pain.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        eq = r.equity_curves
        hwm = np.maximum.accumulate(eq, axis=1)
        dd_pct = (hwm - eq) / hwm  # positive numbers = drawdown magnitude
        t = np.arange(eq.shape[1]) / periods_per_year
        c = colors[i % len(colors)]
        for j, q in enumerate(percentiles):
            curve = np.quantile(dd_pct, q, axis=0)
            ls = ['-', '--', ':', '-.'][j % 4]
            ax.plot(t, curve, color=c, ls=ls, lw=1.5,
                    label=f'{r.strategy_name}/{r.rule_name} p{int(q*100)}')

    ax.set_xlabel('Years')
    ax.set_ylabel('Drawdown from HWM')
    ax.invert_yaxis()  # deeper drawdowns lower on chart, intuition-friendly
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(title or 'Drawdown fan chart (by percentile)')
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(alpha=0.3)
    return ax


def plot_return_vs_dd_scatter(
    results: Sequence[BacktestResult],
    ax=None,
    alpha: float = 0.15,
    title: str | None = None,
):
    """
    One dot per path: terminal total return vs max DD %. One color per rule.

    Visually slam-dunk for showing the stop's effect: with-stop cloud gets
    capped on the DD axis, and you see how much right-tail you gave up.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        ax.scatter(r.max_drawdown_pct, r.total_returns,
                   s=6, alpha=alpha, color=c,
                   label=f'{r.strategy_name}/{r.rule_name}')
    ax.axhline(0, color='k', ls=':', alpha=0.4)
    ax.set_xlabel('Max drawdown (%)')
    ax.set_ylabel('Total return')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_title(title or 'Terminal return vs max drawdown (one dot per path)')
    leg = ax.legend(fontsize=9, loc='best')
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax.grid(alpha=0.3)
    return ax


def plot_did_stop_help(
    treated: BacktestResult,
    baseline: BacktestResult,
    ax=None,
    bins: int = 80,
    title: str | None = None,
):
    """
    Path-wise: histogram of (treated terminal return - baseline terminal return).

    The shape tells the whole story. Right-skewed = stop earns its keep
    (saves bad paths more than it costs good paths). Left-skewed or symmetric
    around a negative mean = stop is just premium with no payoff.

    Both results must share the same scenario set.
    """
    if treated.total_returns.shape != baseline.total_returns.shape:
        raise ValueError("Paired plot requires same scenario set.")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    diff = treated.total_returns - baseline.total_returns
    # Split into helped / hurt / no-op bins for visual storytelling.
    helped = diff[diff > 0.005]
    hurt = diff[diff < -0.005]
    neutral = diff[(diff >= -0.005) & (diff <= 0.005)]

    ax.hist(hurt, bins=bins, alpha=0.7, color='tab:red',
            label=f'Stop hurt ({len(hurt) / len(diff):.0%} of paths)')
    ax.hist(neutral, bins=max(5, bins // 10), alpha=0.7, color='tab:gray',
            label=f'Neutral ({len(neutral) / len(diff):.0%})')
    ax.hist(helped, bins=bins, alpha=0.7, color='tab:green',
            label=f'Stop helped ({len(helped) / len(diff):.0%})')
    ax.axvline(0, color='k', ls=':', alpha=0.6)
    ax.axvline(diff.mean(), color='navy', ls='--', lw=2,
               label=f'Mean = {diff.mean():+.2%}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0%}'))
    ax.set_xlabel('Terminal return: with stop  –  without stop')
    ax.set_ylabel('Paths')
    ax.set_title(title or f'Did the stop help? ({treated.strategy_name})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Convenience: institutional one-pager table
# ---------------------------------------------------------------------------

def institutional_summary(
    results: Sequence[BacktestResult],
    dd_thresholds: Sequence[float] = (0.10, 0.15, 0.20, 0.30),
) -> pd.DataFrame:
    """
    One-row-per-result table with the metrics allocators actually ask for.
    Pairs well with the four plots above for a pitch-deck one-pager.
    """
    rows = []
    for r in results:
        tr = r.total_returns
        dd_pct = r.max_drawdown_pct
        eq = r.equity_curves
        years = (eq.shape[1] - 1) / 252
        cagr = (eq[:, -1] / eq[:, 0]) ** (1 / years) - 1

        # Rolling 1yr worst across all (path, start-day).
        if eq.shape[1] > 252:
            roll_1y = eq[:, 252:] / eq[:, :-252] - 1
            worst_1y = np.quantile(roll_1y.ravel(), 0.05)
        else:
            worst_1y = np.nan

        calmar = np.where(dd_pct > 0, cagr / dd_pct, np.nan)

        row = {
            'strategy': r.strategy_name,
            'rule': r.rule_name,
            'mean_CAGR': cagr.mean(),
            'median_CAGR': np.median(cagr),
            'mean_Calmar': np.nanmean(calmar),
            'mean_maxDD_%': dd_pct.mean(),
            'p95_maxDD_%': np.quantile(dd_pct, 0.95),
            'rolling_1yr_p05': worst_1y,
            'CVaR05_return': tr[tr <= np.quantile(tr, 0.05)].mean(),
            'prob_negative_5yr': float((tr < 0).mean()),
        }
        for t in dd_thresholds:
            row[f'P(maxDD>{int(t*100)}%)'] = float((dd_pct > t).mean())
        rows.append(row)
    return pd.DataFrame(rows)
