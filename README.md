# Trailing Stop Evaluation via Stationary Bootstrap

## Objective

Evaluate whether a tiered trailing-stop / position-sizing rule makes the
risk-return trade-off better or worse across several stock-picking strategies,
under realistic market dynamics. The testing rule in scope:

- Cut position to 70% when drawdown from HWM reaches **$400k**
- Cut further to 40% at **$1.1m** drawdown
- Fully exit at **$2m** drawdown
- Step back up one level if equity recovers **$300k** from its trough

This rule is path-dependent — its behavior depends on the *sequence* of returns,
not just the distribution — so back-testing on a single historical series is
too thin to draw conclusions. We use 10,000 simulated 5-year paths built from
20 years of history to map the full distribution of outcomes.

## Method

### Stationary bootstrap for path simulation

We generate simulated paths by stitching together random-length blocks of
consecutive historical days (Politis & Romano 1994). Block lengths are
geometric random variables with mean `L`, so the process is stationary and
preserves local dependence structure (autocorrelation, vol clustering) while
still producing path diversity.

The bootstrap is implemented as a functional pipeline: we generate one
`(n_paths × path_length)` index matrix into history, then apply the same
indices to every strategy's return series. This guarantees each strategy sees
the same simulated "market history," so stop-vs-no-stop comparisons aren't
contaminated by simulation noise.

`L` defaults to the Politis-White automatic estimate (typically 5–20 days on
daily equity data) but can be overridden. Sensitivity sweeps across
`L ∈ {10, 30, 60, 120, 250}` check whether conclusions are robust to this
choice.

### Backtest engine and stop rule

Each stop rule is a `StopRule` subclass with a `reset(initial_capital)` and
`update(equity)` interface. The engine drives the rule day-by-day, with a
numba-jit fast path for `TrailingStopRule` (~50-100× faster than a pure
Python loop). Each (strategy × rule) combination runs against the same
scenario set and produces a `BacktestResult` with per-path equity curves,
position sizes, and summary statistics.

## Project structure

```
src/
  simulation.py    functional bootstrap: generate_scenarios, politis_white_L
  stop_rules.py    StopRule ABC + NoStop + TrailingStopRule (OOP)
  engine.py        BacktestResult + run_backtest (numba-jit fast path)
  analysis.py      percentile tables, CVaR, drawdown dynamics,
                   conditional analysis, paired / bootstrap significance
  institutional.py allocator-facing metrics + four-panel visual one-pager
  sensitivity.py   robustness sweeps over L, rule parameters, capital
run.py             end-to-end runner: produces all tables and plots
validate.py        single-path validation: proves calculations are correct
```

## How to evaluate the rule

The analysis is organized around a core question:
**does the stop's insurance payoff (reduced left tail) justify its premium
(lost right tail), across realistic market conditions?**

### 1. Start with the institutional one-pager

`onepager_<strategy>.png` has four panels that together tell the story:

- **Equity fan chart** — median + 25/75 + 5/95 percentile bands over time,
  with and without the stop. Shows how the stop reshapes the distribution
  of trajectories. Good stops narrow the fan asymmetrically — cutting more
  downside than upside.
- **Drawdown fan chart** — drawdown from HWM by percentile. A binding stop
  produces flat percentile curves around the hard-stop level.
- **Return vs max DD scatter** — one dot per path. With-stop cloud is capped
  on the DD axis; the visual gap between the two clouds is exactly the
  trade-off.
- **"Did the stop help?" histogram** — per-path terminal-return delta.
  Right-skewed distribution with a fat right tail = stop earns its keep
  (saves bad paths). Symmetric or left-skewed = stop is just premium with
  no payoff.

### 2. The headline metrics an allocator asks about

`institutional_summary.csv`:
- **CAGR** — mean and median across paths
- **Calmar (CAGR / max DD)** — the single most-cited risk-adjusted metric in
  real allocator conversations. Good stops improve Calmar even when they
  reduce CAGR.
- **p95 max DD %** — "what does a bad drawdown look like?"
- **Rolling 1yr p05** — the worst 1-year return an investor could have
  experienced at the worst possible entry point.
- **CVaR at 5%** — mean terminal return in the worst 5% of paths.
- **P(max DD > {10, 15, 20, 30}%)** — mandate-trigger probabilities.

`stop_activity.csv` answers "how often does this rule actually fire?" —
cuts/raises/stopouts per year and days-at-reduced-size. A rule that triggers
5×/year has real frictional cost that should be pricing in.

### 3. Deeper diagnostics (in `run.py` stdout)

- **Conditional comparison by worst_30d** — does the stop help most in the
  bottom-quintile bucket (worst-stress paths)? If yes, it's a genuine tail
  hedge. If it helps uniformly across buckets, it's just reducing risk
  proportionally.
- **Paired comparison + bootstrap CI** — percentage of paths where the stop
  helped, and whether the mean-return difference is statistically
  distinguishable from zero. If the CI crosses zero, you don't have a robust
  edge.

### 4. Robustness checks

- **Sensitivity to L** — `sensitivity_L.csv`. If the stop's ranking vs no-stop
  flips as L varies, the rule is exploiting simulation artifacts rather than
  real dynamics.
- **Sensitivity to rule parameters** — `sensitivity_params.csv`. Sweeps the
  trigger levels ±25% and varies the re-entry recovery amount. Healthy rules
  show smooth performance gradients; sharp differences = overfit to specific
  thresholds.

### Interpretation framework

Put the facts together in this order:

1. Does the stop meaningfully compress drawdowns? (institutional_summary DD
   columns, drawdown fan chart)
2. What's the cost? (mean CAGR comparison, equity fan right-tail band)
3. Is the cost worth it for *this particular allocator's* risk tolerance?
   (CVaR, P(max DD > mandate trigger), rolling 1yr p05)
4. Is the conclusion robust? (sensitivity sweeps, bootstrap CI)
5. Does it work uniformly across strategies, or only some? (per-strategy
   "did it help" histograms)

A robust answer needs all five to line up. If they don't, the inconsistencies
themselves are the finding — usually they reveal that the rule works for one
strategy character (e.g. trend-following) but not another (e.g. mean-reversion).

## Usage

```python
from src import generate_scenarios, NoStop, TrailingStopRule, run_backtest

# Provide daily returns for each strategy, aligned to the same dates.
historical_returns = {
    'strategy_A': returns_A,  # pd.Series
    'strategy_B': returns_B,
}

# Generate 10k simulated 5-year paths, reused across all strategies.
scenarios = generate_scenarios(
    historical_returns=historical_returns,
    n_paths=10_000, path_length=1260, L_mean=None, seed=42,
)

# Define a stop rule.
rule = TrailingStopRule(
    levels=[(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.00)],
    reentry_recovery=300_000,
)

# Backtest.
result = run_backtest(
    scenarios['paths']['strategy_A'], rule, 'strategy_A',
    initial_capital=10_000_000,
)
```

See `run.py` for the full evaluation pipeline.

## Validation

`validate.py` proves the path-level mechanics are correct by:

1. Verifying the bootstrap lookup is consistent
   (`paths[i, t] == historical_returns[idx[i, t]]`)
2. Replaying the trailing-stop rule for one path in pure Python, outside the
   engine, and confirming every equity value and position size matches the
   engine's output bit-for-bit
3. Checking that terminal wealth, total return, max DD ($), and max DD (%)
   derived from the equity curve match the engine's summary properties
4. Cross-checking the NoStop case against `initial × cumprod(1 + r)`

Run `python validate.py` — all checks should pass with zero floating-point
error.
