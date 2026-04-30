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
    percentiles: Sequence[float] = (0.25, 0.50, 0.75, 0.95),
    ax=None,
    periods_per_year: int = 252,
    title: str | None = None,
):
    """
    Underwater fan chart: drawdown from running HWM over time, by percentile.

    Percentiles here describe the DISTRIBUTION OF DRAWDOWNS across paths at
    each point in time:
      - Low percentile (e.g. 0.25): paths with small drawdown — near their HWM
      - High percentile (e.g. 0.95): paths with large drawdown — badly underwater

    With the y-axis inverted (deeper = lower), low-percentile lines plot near
    the top (near 0%), high-percentile lines plot deep.

    Note: since only ~3-5% of paths are at their exact HWM at any given moment,
    even the p25 line will typically show some drawdown. This is normal for any
    volatile strategy — it simply reflects that most paths are between highs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        eq = r.equity_curves
        hwm = np.maximum.accumulate(eq, axis=1)
        dd_pct = (hwm - eq) / hwm  # positive = drawdown magnitude
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


def plot_pct_at_hwm(
    results: Sequence[BacktestResult],
    ax=None,
    periods_per_year: int = 252,
    title: str | None = None,
):
    """
    Fraction of paths at their HWM (drawdown = 0%) at each point in time.

    This directly answers "how often is the strategy making new highs?"
    Complements the drawdown fan: the drawdown fan shows depth for paths
    that ARE in drawdown; this shows how many paths are NOT in drawdown.

    Typical pattern:
      - Starts at 100% (all paths at inception = HWM by definition)
      - Drops quickly as paths diverge
      - Stabilises around 3-5% for a volatile strategy (at any snapshot,
        only a small fraction happen to be exactly at their all-time high)
      - Stop rules show LOWER fraction than NoStop (paths go flat after
        stop-out and can never make new highs while flat)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        eq = r.equity_curves
        hwm = np.maximum.accumulate(eq, axis=1)
        at_hwm = (eq >= hwm).mean(axis=0)   # fraction at HWM each day
        t = np.arange(eq.shape[1]) / periods_per_year
        c = colors[i % len(colors)]
        ax.plot(t, at_hwm, color=c, lw=1.5,
                label=f'{r.strategy_name}/{r.rule_name}')
    ax.set_xlabel('Years')
    ax.set_ylabel('Fraction of paths at HWM')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(title or 'Fraction of paths at all-time high (HWM)')
    ax.legend(fontsize=9)
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
# New presentation-grade plot functions
# ---------------------------------------------------------------------------

def plot_calmar_bar(
    results: Sequence[BacktestResult],
    ax=None,
    title: str | None = None,
):
    """
    Grouped bar chart of mean Calmar ratio (CAGR / max DD) per strategy and rule.

    The single most important institutional metric in one glance. Groups by
    strategy, bars per rule. Higher = better risk-adjusted return.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    # Build DataFrame of Calmar values.
    rows = []
    for r in results:
        eq = r.equity_curves
        years = (eq.shape[1] - 1) / 252
        cagr = (eq[:, -1] / eq[:, 0]) ** (1 / years) - 1
        dd_pct = r.max_drawdown_pct
        calmar = np.where(dd_pct > 0, cagr / dd_pct, np.nan)
        rows.append({'strategy': r.strategy_name, 'rule': r.rule_name,
                     'calmar': float(np.nanmean(calmar))})
    df = pd.DataFrame(rows)

    strategies = df['strategy'].unique()
    rules = df['rule'].unique()
    x = np.arange(len(strategies))
    width = 0.8 / len(rules)
    colors = plt.cm.tab10.colors

    for i, rule in enumerate(rules):
        vals = [df[(df.strategy == s) & (df.rule == rule)]['calmar'].values
                for s in strategies]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - len(rules) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=rule, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('Mean Calmar Ratio (CAGR / Max DD)')
    ax.set_title(title or 'Calmar Ratio by Strategy and Rule')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_dd_breach_heatmap(
    results: Sequence[BacktestResult],
    thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    ax=None,
    title: str | None = None,
):
    """
    Heatmap of P(max DD > threshold) — strategy × threshold, one panel per rule.

    Colour = probability of breaching that DD level. Stop rules should show
    a hard colour boundary at their stop threshold (e.g. near-zero probability
    above 20% for a $2m stop on a $10m book).
    """
    rules = list(dict.fromkeys(r.rule_name for r in results))
    strategies = list(dict.fromkeys(r.strategy_name for r in results))

    n_rules = len(rules)
    fig_needed = ax is None
    if fig_needed:
        fig, axes = plt.subplots(1, n_rules, figsize=(4 * n_rules, 4),
                                 sharey=True)
        if n_rules == 1:
            axes = [axes]
    else:
        axes = [ax] * n_rules

    result_lookup = {(r.strategy_name, r.rule_name): r for r in results}
    labels = [f'{int(t*100)}%' for t in thresholds]

    for j, rule in enumerate(rules):
        matrix = []
        for strat in strategies:
            key = (strat, rule)
            if key in result_lookup:
                dd_pct = result_lookup[key].max_drawdown_pct
                row = [(dd_pct > t).mean() for t in thresholds]
            else:
                row = [np.nan] * len(thresholds)
            matrix.append(row)
        matrix = np.array(matrix)

        im = axes[j].imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                            vmin=0, vmax=1)
        axes[j].set_xticks(range(len(thresholds)))
        axes[j].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        if j == 0:
            axes[j].set_yticks(range(len(strategies)))
            axes[j].set_yticklabels(strategies, fontsize=8)
        axes[j].set_title(rule, fontsize=9)
        for row_i in range(len(strategies)):
            for col_i in range(len(thresholds)):
                val = matrix[row_i, col_i]
                if not np.isnan(val):
                    axes[j].text(col_i, row_i, f'{val:.0%}',
                                 ha='center', va='center', fontsize=7,
                                 color='white' if val > 0.6 else 'black')

    if fig_needed:
        fig.colorbar(im, ax=axes, label='P(max DD > threshold)')
        fig.suptitle(title or 'Drawdown Breach Probability Heatmap',
                     fontsize=11, y=1.02)
    return axes


def plot_rolling_return_violin(
    results: Sequence[BacktestResult],
    window_days: int = 252,
    ax=None,
    title: str | None = None,
    max_sample: int = 50_000,
):
    """
    Violin plot of rolling 1-year return distribution per rule, grouped by strategy.

    Shows the full distribution shape — width at each return level shows
    where mass concentrates. Much more informative than a table of percentiles.
    Immediately comparable across rules: a good stop narrows the left tail.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    strategies = list(dict.fromkeys(r.strategy_name for r in results))
    rules = list(dict.fromkeys(r.rule_name for r in results))
    colors = plt.cm.tab10.colors

    positions = []
    data_list = []
    labels = []
    tick_pos = []
    tick_labels = []

    group_width = len(rules) + 1
    for s_idx, strat in enumerate(strategies):
        group_center = s_idx * group_width + len(rules) / 2
        tick_pos.append(group_center)
        tick_labels.append(strat)
        for r_idx, rule in enumerate(rules):
            match = [r for r in results
                     if r.strategy_name == strat and r.rule_name == rule]
            if not match:
                continue
            r = match[0]
            eq = r.equity_curves
            if eq.shape[1] <= window_days:
                continue
            roll = (eq[:, window_days:] / eq[:, :-window_days] - 1).ravel()
            # Subsample if too large for violin.
            if len(roll) > max_sample:
                rng = np.random.default_rng(0)
                roll = rng.choice(roll, max_sample, replace=False)
            pos = s_idx * group_width + r_idx
            positions.append(pos)
            data_list.append(roll)
            labels.append(rule)

    if data_list:
        parts = ax.violinplot(data_list, positions=positions,
                              showmedians=True, showextrema=False, widths=0.8)
        for i, (pc, pos) in enumerate(zip(parts['bodies'], positions)):
            rule_idx = rules.index(labels[i])
            pc.set_facecolor(colors[rule_idx % len(colors)])
            pc.set_alpha(0.6)
        parts['cmedians'].set_colors('black')
        parts['cmedians'].set_linewidth(1.5)

    ax.axhline(0, color='k', ls='--', alpha=0.5, lw=0.8)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=15, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylabel(f'Rolling {window_days//252:.0f}-yr Return')
    ax.set_title(title or f'Rolling {window_days//252:.0f}-Year Return Distribution by Rule')

    # Legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)], alpha=0.6)
               for i, rule in enumerate(rules)]
    ax.legend(handles, rules, fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_stop_activity_bar(
    results: Sequence[BacktestResult],
    periods_per_year: int = 252,
    ax=None,
    title: str | None = None,
):
    """
    Stacked bar chart: days per year at full / reduced / stopped size, per result.

    Immediately shows the operational cost of each rule — how many business
    days per year the strategy runs at reduced or zero exposure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))

    labels, full_days, reduced_days, stopped_days = [], [], [], []
    for r in results:
        sizes = r.position_sizes
        n_days = sizes.shape[1]
        years = n_days / periods_per_year
        full    = (sizes == 1.0).mean(axis=1).mean() * periods_per_year
        stopped = (sizes == 0.0).mean(axis=1).mean() * periods_per_year
        reduced = periods_per_year - full - stopped
        labels.append(f'{r.strategy_name}\n{r.rule_name}')
        full_days.append(full)
        reduced_days.append(reduced)
        stopped_days.append(stopped)

    x = np.arange(len(labels))
    width = 0.6
    p1 = ax.bar(x, full_days,    width, label='Full size (1.0)',
                color='tab:green', alpha=0.8)
    p2 = ax.bar(x, reduced_days, width, bottom=full_days,
                label='Reduced (0 < size < 1)', color='tab:orange', alpha=0.8)
    bottom2 = [f + r for f, r in zip(full_days, reduced_days)]
    p3 = ax.bar(x, stopped_days, width, bottom=bottom2,
                label='Stopped (size = 0)', color='tab:red', alpha=0.8)

    # Value labels on reduced + stopped segments.
    for i, (rd, sd) in enumerate(zip(reduced_days, stopped_days)):
        if rd > 2:
            ax.text(x[i], full_days[i] + rd / 2, f'{rd:.0f}d',
                    ha='center', va='center', fontsize=7, color='white')
        if sd > 2:
            ax.text(x[i], bottom2[i] + sd / 2, f'{sd:.0f}d',
                    ha='center', va='center', fontsize=7, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Business days per year')
    ax.set_title(title or 'Stop Activity: Days at Each Size Level (per year)')
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(periods_per_year, color='k', ls=':', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_conditional_diverging(
    conditional_df: pd.DataFrame,
    ax=None,
    title: str | None = None,
):
    """
    Diverging horizontal bar chart of stop rule's per-bucket mean return delta.

    Input: output of conditional_comparison() from analysis.py.
    x-axis = delta_mean_tr (positive = stop helped, negative = stop hurt).
    One bar per bucket (sorted worst to best market environment).

    Immediately shows "the stop helps in crisis buckets, hurts in calm ones."
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df = conditional_df.copy()
    deltas = df['delta_mean_tr'].values
    buckets = df['bucket_range'].values
    n = len(df)
    y = np.arange(n)

    colors = ['tab:green' if d >= 0 else 'tab:red' for d in deltas]
    bars = ax.barh(y, deltas, color=colors, alpha=0.8, height=0.6)

    # Value labels.
    for bar, val in zip(bars, deltas):
        x_pos = val + (0.002 if val >= 0 else -0.002)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1%}', va='center', ha=ha, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([f'Bucket {df.iloc[i]["bucket"]} {buckets[i]}'
                        for i in range(n)], fontsize=8)
    ax.axvline(0, color='k', lw=1, alpha=0.7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0%}'))
    ax.set_xlabel('Mean terminal return delta (treated − baseline)')
    ax.set_title(title or 'Stop Rule Effect by Market Regime Bucket\n'
                           '(Bucket 0 = worst 20% of paths)')
    ax.grid(axis='x', alpha=0.3)
    return ax


def plot_size_change_frequency(
    results: Sequence[BacktestResult],
    window_days: int = 21,
    periods_per_year: int = 252,
    ax=None,
    title: str | None = None,
):
    """
    Rolling event density over simulated path time: average number of
    size-change events per month at each point in the simulated period,
    averaged across all paths.

    Signs to look for:
      - Spike at the start then near-zero: rule fires quickly, paths go flat.
        This is the permanent stopout problem — rule isn't actively managing,
        it's just stopping.
      - Roughly uniform density: rule actively manages throughout the period.
      - Clustered spikes: stress periods are concentrated in certain simulated
        regimes (expected if block bootstrap preserved crisis clustering).

    High frequency overall = rule may be whipsawing on noise.
    High frequency only in early months = stopout-dominated.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, r in enumerate(results):
        if r.rule_name == 'NoStop':
            continue
        sizes = r.position_sizes          # (n_paths, n_days)
        n_paths, n_days = sizes.shape
        prev = np.concatenate([np.ones((n_paths, 1)), sizes[:, :-1]], axis=1)
        events = (sizes != prev).astype(float)  # 1 where size changed

        # Rolling window sum averaged across paths.
        density = pd.DataFrame(events.T).rolling(window_days).sum().mean(axis=1)
        t = np.arange(len(density)) / periods_per_year
        c = colors[i % len(colors)]
        ax.plot(t, density, color=c, lw=1.5,
                label=f'{r.strategy_name}/{r.rule_name}')

    ax.set_xlabel('Years into simulated path')
    ax.set_ylabel(f'Mean events per {window_days}-day window (avg across paths)')
    ax.set_title(title or f'Size-Change Event Density Over Simulated Time\n'
                           f'(rolling {window_days}-day window)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_historical_events(
    historical_returns,
    rule,
    strategy_name: str,
    initial_capital: float,
    ax=None,
    title: str | None = None,
):
    """
    Apply the stop rule to the actual historical return series and plot:
      - Normalised equity curve
      - Position size as shaded area (right axis)
      - CUT events as red downward triangles
      - RAISE events as green upward triangles

    This is the most intuitive diagnostic for two questions:
      1. Validation: did the rule fire during known real stress periods?
         (2008 GFC, March 2020 COVID, 2022 rate shock)
      2. Whipsaw detection: are cuts and raises clustered tightly together
         in short windows? That signals the rule is reacting to noise
         rather than genuine drawdowns.

    Parameters
    ----------
    historical_returns : pd.Series
        Aligned daily returns for one strategy (same series used in simulation).
    rule : StopRule
        Fresh rule instance — will be run through the engine on the full history.
    strategy_name : str
    initial_capital : float
    """
    from .engine import run_backtest

    returns = historical_returns.values
    dates = historical_returns.index
    single_path = returns[np.newaxis, :]

    res = run_backtest(single_path, rule, strategy_name, initial_capital)
    eq = res.equity_curves[0]       # length n_days+1
    sizes = res.position_sizes[0]   # length n_days
    eq_norm = eq / eq[0]            # normalise to 1.0

    # Detect size-change events.
    prev = np.concatenate([[1.0], sizes[:-1]])
    cuts   = np.where(sizes < prev)[0]
    raises = np.where(sizes > prev)[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    # Equity curve on left axis.
    ax.plot(dates, eq_norm[:-1], color='black', lw=1.2,
            label='Equity (normalised)', zorder=3)
    ax.axhline(1.0, color='k', ls=':', alpha=0.3)

    # Position size as shaded area on right axis.
    ax2 = ax.twinx()
    ax2.fill_between(dates, sizes, alpha=0.12, color='steelblue')
    ax2.plot(dates, sizes, color='steelblue', lw=0.8, alpha=0.5,
             label='Position size')
    ax2.set_ylabel('Position size', color='steelblue')
    ax2.set_ylim(-0.1, 1.4)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Mark cut and raise events.
    if len(cuts):
        ax.scatter(dates[cuts], eq_norm[cuts],
                   color='tab:red', marker='v', s=70, zorder=5,
                   label=f'CUT ({len(cuts)})')
    if len(raises):
        ax.scatter(dates[raises], eq_norm[raises],
                   color='tab:green', marker='^', s=70, zorder=5,
                   label=f'RAISE ({len(raises)})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Equity (normalised to 1.0 at inception)')
    ax.set_title(title or f'{strategy_name} / {rule.name}: Historical Stop Events')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    # Print event log for inspection.
    years = len(dates) / 252
    print(f"\n{strategy_name} / {rule.name}  ({years:.1f} years of history)")
    print(f"  Cuts:   {len(cuts)}  ({len(cuts)/years:.1f}/yr)")
    print(f"  Raises: {len(raises)}  ({len(raises)/years:.1f}/yr)")
    if len(cuts):
        print(f"  First cut: {dates[cuts[0]].date()}   size → {sizes[cuts[0]]:.0%}")
        print(f"  Last cut:  {dates[cuts[-1]].date()}  size → {sizes[cuts[-1]]:.0%}")
    if len(cuts) > 1:
        gaps = np.diff(cuts)
        short_gaps = (gaps < 10).sum()
        print(f"  Mean days between cuts: {gaps.mean():.0f}")
        print(f"  Cuts within 10 days of previous cut: {short_gaps} "
              f"({'⚠ possible whipsaw' if short_gaps > 2 else 'OK'})")

    return ax, ax2


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
