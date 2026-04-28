"""
Main runner: end-to-end evaluation of a trailing stop on multiple strategies.

Produces:
  - core diagnostics (percentiles, CVaR, drawdown dynamics, conditional, paired)
  - institutional one-pager (summary table + 4-panel plot per strategy)
  - robustness checks (sensitivity to bootstrap L and rule parameters)

Replace the synthetic data in the SETUP section with your real strategy returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import (
    # simulation & engine
    generate_scenarios, NoStop, TrailingStopRule, run_backtest,
    # core analysis
    percentile_table, cvar_table,
    drawdown_summary, conditional_comparison,
    paired_comparison, bootstrap_ci,
    plot_distribution_overlay,
    # institutional
    institutional_summary, rolling_return_stats, dd_threshold_probabilities,
    stop_activity,
    plot_equity_fan, plot_drawdown_fan, plot_return_vs_dd_scatter,
    plot_did_stop_help,
    # sensitivity
    sensitivity_to_L, sensitivity_to_rule_params,
)


# =============================================================================
# SETUP — replace with your real data
# =============================================================================
dates = pd.bdate_range("2006-01-01", "2026-01-01")
n = len(dates)
rng = np.random.default_rng(0)

def ar1(n, mu, sigma, phi):
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = mu + phi * (r[i - 1] - mu) + eps[i]
    return r

historical_returns = {
    'momentum':    pd.Series(ar1(n, 0.0006, 0.014,  0.10), index=dates),
    'mean_revert': pd.Series(ar1(n, 0.0004, 0.009, -0.05), index=dates),
    'vol_carry':   pd.Series(ar1(n, 0.0008, 0.020,  0.08), index=dates),
}

INITIAL_CAPITAL = 10_000_000.0
STOP_LEVELS = [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.00)]
REENTRY = 300_000
TRIGGER_DOLLARS = [400_000, 1_100_000, 2_000_000]


# =============================================================================
# STEP 1: generate scenarios ONCE, reuse for all strategies and rules
# =============================================================================
print("Generating 10,000 scenarios...")
scenarios = generate_scenarios(
    historical_returns=historical_returns,
    n_paths=10_000, path_length=1260, L_mean=None, seed=42,
)
print(f"  Politis-White L per strategy: {scenarios['L_per_strategy']}")
print(f"  Block length used: L_mean = {scenarios['L_mean']:.1f}")

def make_trailing():
    return TrailingStopRule(levels=STOP_LEVELS, reentry_recovery=REENTRY,
                            label="TrailingStop")

results = {}
for strat, paths in scenarios['paths'].items():
    results[(strat, 'baseline')] = run_backtest(
        paths, NoStop(), strat, INITIAL_CAPITAL)
    results[(strat, 'trailing')] = run_backtest(
        paths, make_trailing(), strat, INITIAL_CAPITAL)
all_results = list(results.values())


# =============================================================================
# STEP 2: institutional summary + rolling / DD-threshold / activity tables
# =============================================================================
print("\n" + "=" * 100)
print("INSTITUTIONAL SUMMARY (the numbers an allocator actually asks for)")
print("=" * 100)
summary = institutional_summary(all_results, dd_thresholds=(0.10, 0.15, 0.20, 0.30))
with pd.option_context('display.max_columns', None, 'display.width', 220,
                       'display.float_format', '{:.3f}'.format):
    print(summary.to_string(index=False))
summary.to_csv('institutional_summary.csv', index=False)

roll_1y = pd.DataFrame([rolling_return_stats(r, 252) for r in all_results])
dd_probs = pd.DataFrame([
    dd_threshold_probabilities(r, (0.05, 0.10, 0.15, 0.20, 0.30, 0.50))
    for r in all_results
])
activity = pd.DataFrame([stop_activity(r) for r in all_results])

print("\n--- Rolling 1yr return distribution ---")
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(roll_1y.to_string(index=False))

print("\n--- Drawdown breach probabilities ---")
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(dd_probs.to_string(index=False))

print("\n--- Stop activity (how often the rule actually fires) ---")
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(activity.to_string(index=False))


# =============================================================================
# STEP 3: core diagnostics (percentiles, CVaR, drawdowns, conditional, paired)
# =============================================================================
print("\n" + "=" * 100)
print("CORE DIAGNOSTICS")
print("=" * 100)

pct_table = percentile_table(all_results)
cvar_t = cvar_table(all_results, alphas=(0.01, 0.05, 0.10))
print("\n--- Percentile table (terminal return and max DD) ---")
with pd.option_context('display.max_columns', None, 'display.width', 240,
                       'display.float_format', '{:.3f}'.format):
    print(pct_table.to_string(index=False))
print("\n--- CVaR table ---")
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(cvar_t.to_string(index=False))

print("\n--- Drawdown dynamics (hit rates, time underwater, recovery) ---")
dd_rows = pd.DataFrame([drawdown_summary(r, TRIGGER_DOLLARS) for r in all_results])
with pd.option_context('display.max_columns', None, 'display.width', 240,
                       'display.float_format', '{:.3f}'.format):
    print(dd_rows.to_string(index=False))

print("\n--- Paired comparisons and bootstrap CIs ---")
paired_rows, ci_rows = [], []
for strat in historical_returns:
    baseline = results[(strat, 'baseline')]
    treated = results[(strat, 'trailing')]
    paired_rows.append(paired_comparison(treated, baseline))
    ci = bootstrap_ci(treated, baseline, 'total_returns', n_resamples=2000)
    ci['strategy'] = strat
    ci_rows.append(ci)
with pd.option_context('display.float_format', '{:.4f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(pd.DataFrame(paired_rows).to_string(index=False))
    print("\nBootstrap 95% CI on mean-return difference (stop - baseline):")
    print(pd.DataFrame(ci_rows).to_string(index=False))

print("\n--- Conditional comparison (momentum, bucketed by baseline worst_30d) ---")
cond = conditional_comparison(
    results[('momentum', 'trailing')], results[('momentum', 'baseline')],
    bucket_by='worst_30d', n_buckets=5,
)
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(cond.to_string(index=False))
print("Question: does the stop help most in bucket 0 (worst-tail paths)?")


# =============================================================================
# STEP 4: plots
# =============================================================================
# Quick-diagnostic overlay plots for all strategies
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for j, strat in enumerate(historical_returns):
    plot_distribution_overlay(
        [results[(strat, 'baseline')], results[(strat, 'trailing')]],
        metric='total_returns', ax=axes[0, j],
        title=f'{strat}: terminal return distribution')
    plot_distribution_overlay(
        [results[(strat, 'baseline')], results[(strat, 'trailing')]],
        metric='max_drawdown_pct', ax=axes[1, j],
        title=f'{strat}: max DD % distribution')
plt.tight_layout()
plt.savefig('distribution_overlays.png', dpi=120)
plt.close()
print("\nSaved distribution_overlays.png")

# Institutional 4-panel one-pager per strategy
for strat in historical_returns:
    b = results[(strat, 'baseline')]
    t = results[(strat, 'trailing')]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_equity_fan([b, t], ax=axes[0, 0], title=f'{strat}: equity fan')
    plot_drawdown_fan([b, t], ax=axes[0, 1], title=f'{strat}: drawdown fan')
    plot_return_vs_dd_scatter([b, t], ax=axes[1, 0],
                              title=f'{strat}: return vs max DD')
    plot_did_stop_help(t, b, ax=axes[1, 1], title=f'{strat}: per-path delta')
    plt.tight_layout()
    plt.savefig(f'onepager_{strat}.png', dpi=130)
    plt.close()
    print(f"Saved onepager_{strat}.png")


# =============================================================================
# STEP 5: robustness sensitivity sweeps
# =============================================================================
print("\n" + "=" * 100)
print("SENSITIVITY TO BOOTSTRAP BLOCK LENGTH L")
print("=" * 100)
print("Re-running at L = 10, 30, 60, 120, 250 (uses 3000 paths for speed)...")
sens_L = sensitivity_to_L(
    historical_returns=historical_returns,
    rule_factory=make_trailing,
    L_values=[10, 30, 60, 120, 250],
    n_paths=3000, path_length=1260,
    initial_capital=INITIAL_CAPITAL, seed=42,
)
pivot_tr = sens_L.pivot_table(index='L_mean', columns=['strategy', 'rule'],
                              values='mean_total_return')
pivot_dd = sens_L.pivot_table(index='L_mean', columns=['strategy', 'rule'],
                              values='p95_max_dd_$')
with pd.option_context('display.float_format', '{:.4f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print("\nMean total return by L:")
    print(pivot_tr.to_string())
    print("\nP95 max DD ($) by L:")
    print(pivot_dd.to_string())

print("\n" + "=" * 100)
print("SENSITIVITY TO RULE PARAMETERS (-25% / base / +25% levels, 4 reentries)")
print("=" * 100)
level_variants = [
    [(300_000, 0.70), (825_000, 0.40), (1_500_000, 0.0)],   # tight
    [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.0)], # base
    [(500_000, 0.70), (1_375_000, 0.40), (2_500_000, 0.0)], # loose
]
sens_p = sensitivity_to_rule_params(
    scenario_paths=scenarios['paths'],
    level_variants=level_variants,
    reentry_variants=[0, 150_000, 300_000, 500_000],
    initial_capital=INITIAL_CAPITAL,
)
pivot_p = sens_p.pivot_table(
    index=['variant_idx', 'reentry_recovery'],
    columns='strategy', values='mean_total_return',
)
with pd.option_context('display.float_format', '{:.4f}'.format):
    print("Mean total return by (variant, reentry) per strategy:")
    print(pivot_p.to_string())
print("\nVariant 0 = tight (-25%), 1 = base, 2 = loose (+25%)")
print("Smooth gradients across variants = robust. Sharp differences = overfit.")

sens_L.to_csv('sensitivity_L.csv', index=False)
sens_p.to_csv('sensitivity_params.csv', index=False)
summary.to_csv('institutional_summary.csv', index=False)
activity.to_csv('stop_activity.csv', index=False)
print("\nAll CSVs saved in the current working directory.")
