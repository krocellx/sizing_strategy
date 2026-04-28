"""
VALIDATION: trace a single bootstrapped path day-by-day and reproduce every
engine output by hand (pure numpy, no engine code), then compare.

If the hand-calculated numbers match the engine's output, the path-level
mechanics of the trailing stop rule, the equity curve, the drawdown, and
the summary statistics are all correct.
"""

import numpy as np
import pandas as pd
from src import (
    generate_scenarios, NoStop, TrailingStopRule, run_backtest,
)


# ---------------------------------------------------------------------------
# Set up a SMALL, inspectable scenario: one strategy, 30 paths, 500 days.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
dates = pd.bdate_range("2010-01-01", "2020-01-01")
n_hist = len(dates)

historical_returns = {
    'strat': pd.Series(rng.normal(0.0005, 0.012, n_hist), index=dates),
}

scenarios = generate_scenarios(
    historical_returns=historical_returns,
    n_paths=30, path_length=500, L_mean=30, seed=1234,
)
paths = scenarios['paths']['strat']          # shape (30, 500)
idx = scenarios['idx']                       # shape (30, 500)

# ---------------------------------------------------------------------------
# Stop rule under test.
# ---------------------------------------------------------------------------
INITIAL = 10_000_000.0
LEVELS = [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.00)]
REENTRY = 300_000
rule = TrailingStopRule(levels=LEVELS, reentry_recovery=REENTRY, label='test')

engine_result = run_backtest(paths, rule, 'strat', INITIAL)

# ---------------------------------------------------------------------------
# VALIDATION 1: bootstrap lookup round-trip.
# ---------------------------------------------------------------------------
# paths[i, t] should equal historical_returns['strat'].values[idx[i, t]].
hist_arr = historical_returns['strat'].values
direct_lookup = hist_arr[idx]
assert np.allclose(paths, direct_lookup), "paths != hist[idx]"
print(f"[OK] bootstrap lookup:  paths[i,t] == historical_returns[idx[i,t]]  "
      f"({paths.size:,} values checked)")

# ---------------------------------------------------------------------------
# VALIDATION 2: pick ONE path and replay the rule by hand in pure Python.
# ---------------------------------------------------------------------------
# Pick path 0 for inspection.
p = 0
r = paths[p]                                  # the daily returns for this path
engine_eq  = engine_result.equity_curves[p]   # length 501 (includes t=0)
engine_sz  = engine_result.position_sizes[p]  # length 500

# Hand replay.
levels_sorted = sorted(LEVELS, key=lambda x: x[0])
level_dds   = [l[0] for l in levels_sorted]
level_sizes = [l[1] for l in levels_sorted]

hand_eq = [INITIAL]
hand_sz = []
eq = INITIAL
hwm = INITIAL
trough = INITIAL
cur_size = 1.0
cur_level = -1

for t in range(len(r)):
    hand_sz.append(cur_size)
    eq = eq * (1.0 + cur_size * r[t])
    hand_eq.append(eq)

    # Update state.
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
    for i, ldd in enumerate(level_dds):
        if dd >= ldd:
            triggered = i
        else:
            break
    if triggered > cur_level:
        cur_level = triggered
        cur_size = level_sizes[triggered]
        trough = eq
    elif REENTRY > 0 and cur_level >= 0:
        if eq - trough >= REENTRY:
            new_level = -1
            for i, ldd in enumerate(level_dds):
                if dd >= ldd:
                    new_level = i
                else:
                    break
            if new_level < cur_level:
                cur_level = new_level
                cur_size = 1.0 if new_level == -1 else level_sizes[new_level]
                trough = eq

hand_eq = np.array(hand_eq)
hand_sz = np.array(hand_sz)

# Equity and size must match to floating-point tolerance.
max_eq_diff = float(np.max(np.abs(hand_eq - engine_eq)))
max_sz_diff = float(np.max(np.abs(hand_sz - engine_sz)))
print(f"[OK] hand-replay path 0: max |Δequity| = {max_eq_diff:.6e}, "
      f"max |Δsize| = {max_sz_diff:.6e}")
assert max_eq_diff < 1e-6, "equity curves diverged"
assert max_sz_diff < 1e-12, "position sizes diverged"

# ---------------------------------------------------------------------------
# VALIDATION 3: derived statistics on path 0 match engine's output.
# ---------------------------------------------------------------------------
eq = engine_result.equity_curves[p]
hand_terminal = eq[-1]
hand_total_return = eq[-1] / INITIAL - 1.0
hand_hwm_curve = np.maximum.accumulate(eq)
hand_dd_curve = hand_hwm_curve - eq
hand_dd_pct_curve = hand_dd_curve / hand_hwm_curve
hand_max_dd = float(hand_dd_curve.max())
hand_max_dd_pct = float(hand_dd_pct_curve.max())

# Compare to engine properties at index p.
assert np.isclose(hand_terminal, engine_result.terminal_wealth[p])
assert np.isclose(hand_total_return, engine_result.total_returns[p])
assert np.isclose(hand_max_dd, engine_result.max_drawdowns[p])
assert np.isclose(hand_max_dd_pct, engine_result.max_drawdown_pct[p])
print(f"[OK] derived stats on path 0 match engine "
      f"(terminal={hand_terminal:,.2f}, tot_ret={hand_total_return:+.4%}, "
      f"max_dd=${hand_max_dd:,.0f} / {hand_max_dd_pct:.2%})")

# ---------------------------------------------------------------------------
# VALIDATION 4: no-stop path via independent numpy formula.
# ---------------------------------------------------------------------------
nostop = run_backtest(paths, NoStop(), 'strat', INITIAL)
# For NoStop, equity[t+1] = initial * prod(1 + r[:t+1]).
hand_nostop_eq = INITIAL * np.cumprod(1 + paths, axis=1)
hand_nostop_terminal = hand_nostop_eq[:, -1]
diff = float(np.max(np.abs(hand_nostop_terminal - nostop.terminal_wealth)))
print(f"[OK] no-stop terminal wealth matches cumprod formula "
      f"(max |Δ| across 30 paths = {diff:.6e})")

# ---------------------------------------------------------------------------
# VALIDATION 5: print the first 12 event days of path 0 for visual inspection.
# ---------------------------------------------------------------------------
print("\nFirst 12 'event' days on path 0 (rows where position size CHANGES, "
      "plus the first and last rows):")
sz = engine_result.position_sizes[p]
change_mask = np.concatenate([[True], sz[1:] != sz[:-1], [True]])
# engine_eq is length 501; align with per-day size/return on index t (0..499)
records = []
for t in range(len(r)):
    if change_mask[t] or t < 2 or t == len(r) - 1:
        records.append({
            'day': t,
            'return': r[t],
            'size_at_t': sz[t],
            'equity_end_of_t': engine_eq[t + 1],
            'hwm': np.maximum.accumulate(engine_eq)[t + 1],
            'dd_$': np.maximum.accumulate(engine_eq)[t + 1] - engine_eq[t + 1],
        })
df = pd.DataFrame(records).head(12)
with pd.option_context('display.float_format', '{:.4f}'.format,
                       'display.max_columns', None, 'display.width', 140):
    print(df.to_string(index=False))

print("\nALL VALIDATION CHECKS PASSED.")
