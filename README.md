# Strategy Sizing and Stop-Rule Robustness

This project evaluates whether path-dependent position-sizing rules improve
the risk-return profile of a strategy or signal. The original use case is a
tiered trailing stop, but the framework is now structured around reusable
simulation, rule, backtest, metric, and reporting layers.

The key idea is still the same: a stop rule depends on the sequence of returns,
not just the final distribution. A single historical backtest is too thin, so
the project bootstraps many realistic return paths and measures how the rule
behaves across those paths.

## Current Rule Family

The baseline fixed trailing stop is:

- Cut position to 70% when drawdown from high-water mark reaches `$400k`
- Cut further to 40% at `$1.1m`
- Fully exit at `$2m`
- Step back up when drawdown recovers by the configured re-entry amount

The code also supports volatility-scaled variants:

- `VolScaledTrailingStop`: scales trigger levels by current realised vol versus
  a fixed reference vol.
- `RatioVolScaledTrailingStop`: scales trigger levels by a realised-vol ratio,
  defaulting to long-window vol over short-window vol. This is conservative
  when short-term volatility spikes because the multiplier falls and thresholds
  tighten.

For both vol-scaled rules, full stop-out levels (`size == 0`) remain hard
dollar stops. Nonzero reduction levels use the current volatility multiplier
until the first reduction, then lock that multiplier until the rule returns to
full size. This keeps the `$2m` full stop stable instead of letting it fluctuate
with the vol estimate.

## Architecture

The code is intentionally split into four layers:

1. Scenario generation creates bootstrapped return paths.
2. Stop rules own position-sizing state and any optimized fast path.
3. The backtest engine applies a rule to paths and returns a result object.
4. Analysis and plotting consume result metrics rather than recalculating them.

This keeps rule logic, metric logic, and presentation logic separate.

## Project Structure

```text
src/
  simulation.py       Stationary bootstrap and scenario generation.
  stop_rules.py       StopRule interface, concrete rules, and numba fast paths.
  engine.py           run_backtest, BacktestResult, transaction costs,
                      quarterly reset cash flows, result-level metrics.
  cache.py            Chunked disk-backed CachedResult for large runs.
  analysis.py         Distribution tables, CVaR, drawdown analysis,
                      paired comparisons, bootstrap confidence intervals.
  institutional.py    Allocator-facing summaries and plots. Uses result metrics.
  sensitivity.py      Robustness sweeps over block length, rule params, capital.
  example_data.py     Synthetic strategy-return generator for demos.
  validate.py         Independent path-level validation checks.
  helpers.py          Notebook scratch/helper code, not a core API.
  utility.py          Small experimental utilities.

run.py                End-to-end runner.
analysis.ipynb        Exploratory notebook.
requirements.txt      Python dependencies.
tests/                Unit tests for engine/rule/accounting semantics.
```

## Important Design Points

### Stop Rules Own Their Fast Paths

Each rule implements the common interface:

```python
rule.reset(initial_capital)
rule.observe_return(daily_return)
next_size = rule.update(equity)
```

Rules that have a numba implementation expose `run_fast_path()`. The engine
detects this method and uses it automatically. This keeps the optimized njit
logic next to the Python state machine, reducing the chance that the two drift
out of sync.

### Result Objects Own Metrics

`BacktestResult` is the single place for core calculations:

- `terminal_wealth`
- `total_returns`
- `cagr`
- `max_drawdowns`
- `max_drawdown_pct`
- `calmar`
- `sharpe`
- `rolling_returns()`
- `cumulative_wealth_curves`

This matters for quarterly reset mode. In that mode, profitable quarters can
withdraw cash and reset in-fund equity to initial capital. Raw terminal equity
can understate true investor wealth, so CAGR and total return must use:

```text
terminal wealth = terminal equity + cumulative cash flows
```

Plots and tables in `institutional.py` now consume these result-level metrics
instead of duplicating CAGR or Calmar calculations.

### Cached Results Match the Public Metric Surface

`CachedResult` mirrors the main metric properties while loading full curves
lazily from disk. Use `run_backtest_chunked()` when path counts are too large
to keep every curve in memory.

## Bootstrap Method

Scenario generation uses stationary bootstrap over historical daily returns.
It builds a single `(n_paths, path_length)` index matrix and applies that same
matrix to every strategy. This ensures each strategy sees the same simulated
market sequence, so paired comparisons are not polluted by different random
paths.

`L_mean=None` uses the automatic Politis-White estimate. You can also sweep
manual block lengths with `sensitivity_to_L()` to test whether conclusions are
stable across bootstrap assumptions.

For true signal robustness, the strongest test is to bootstrap underlying
market or feature data and rerun the signal on each simulated path. When that
is too expensive, bootstrapping realised strategy returns is still useful for
path-dependent stop-rule robustness, but it does not prove the signal itself is
stable under perturbed feature histories.

## Basic Usage

```python
from src import (
    generate_scenarios,
    NoStop,
    TrailingStopRule,
    RatioVolScaledTrailingStop,
    run_backtest,
    compare,
)

# Daily strategy returns as aligned pandas Series.
historical_returns = {
    "strategy_A": returns_A,
    "strategy_B": returns_B,
}

scenarios = generate_scenarios(
    historical_returns=historical_returns,
    n_paths=10_000,
    path_length=1260,
    L_mean=None,
    seed=42,
)

levels = [
    (400_000, 0.70),
    (1_100_000, 0.40),
    (2_000_000, 0.00),
]

baseline = run_backtest(
    scenarios["paths"]["strategy_A"],
    NoStop(),
    "strategy_A",
    initial_capital=10_000_000,
)

fixed_stop = run_backtest(
    scenarios["paths"]["strategy_A"],
    TrailingStopRule(levels=levels, reentry_recovery=300_000),
    "strategy_A",
    initial_capital=10_000_000,
)

ratio_stop = run_backtest(
    scenarios["paths"]["strategy_A"],
    RatioVolScaledTrailingStop(
        base_levels=levels,
        base_reentry_recovery=300_000,
        numerator_window=252,
        denominator_window=63,
    ),
    "strategy_A",
    initial_capital=10_000_000,
)

summary = compare([baseline, fixed_stop, ratio_stop])
```

## Quarterly Reset Mode

Use quarterly reset mode when the mandate takes profits out at quarter-end
while the strategy is fully invested:

```python
result = run_backtest(
    scenarios["paths"]["strategy_A"],
    rule,
    "strategy_A",
    initial_capital=10_000_000,
    quarterly_reset=True,
    reset_every_days=63,
)
```

When `quarterly_reset=True`, `result.terminal_wealth`, `result.total_returns`,
`result.cagr`, `result.calmar`, and `institutional_summary()` include extracted
cash flows.

## Institutional Analysis

`institutional_summary(results)` produces one row per result with:

- Mean and median CAGR
- Mean Calmar
- Mean and p95 max drawdown
- Rolling one-year p05 return
- CVaR at 5%
- Probability of negative terminal return
- Probability of breaching drawdown thresholds
- Cash-flow stats when quarterly reset is active

Common plots:

- `plot_equity_fan()`
- `plot_drawdown_fan()`
- `plot_return_vs_dd_scatter()`
- `plot_did_stop_help()`
- `plot_calmar_bar()`
- `plot_dd_breach_heatmap()`
- `plot_rolling_return_violin()`
- `plot_stop_activity_bar()`
- `plot_survival_curve()`
- `plot_stopout_pct()`

These functions should stay presentation-focused. If a new reusable metric is
needed, add it to `BacktestResult` first, then have plots consume it.

## Robustness Checks

Use the sensitivity module to test whether conclusions survive reasonable
changes in assumptions:

- `sensitivity_to_L()`: rerun on different bootstrap block lengths.
- `sensitivity_to_rule_params()`: sweep trigger levels and re-entry settings
  on a fixed scenario set.
- `sensitivity_to_capital()`: test absolute-dollar thresholds across different
  account sizes or allocation mixes.

Healthy rules usually show smooth gradients. Sharp cliffs around one parameter
set are a warning sign for overfitting.

## Validation

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the path-level validation:

```bash
python3 src/validate.py
```

The validation checks:

1. Bootstrapped paths equal direct historical-return lookup by index.
2. A single path replayed by hand matches engine equity and position sizes.
3. Terminal wealth, total return, max drawdown dollars, and max drawdown
   percent match `BacktestResult` properties.
4. `NoStop` matches the independent `initial * cumprod(1 + returns)` formula.

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

The tests cover quarterly reset semantics, cash-flow-aware sleeve combination,
reset-aware drawdowns, fast-path equivalence for the fixed trailing stop, and
explicit ratio-vol window configuration.

## Outputs

The current runner/notebook workflow may produce files such as:

- `backtest_results.csv`
- `institutional_summary.csv`
- `distribution_overlays.png`
- `onepager_<strategy>.png`

Generated outputs are analysis artifacts, not source files.
