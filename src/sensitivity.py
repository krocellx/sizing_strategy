"""
Sensitivity / robustness sweeps.

Two main questions:
  1. Are conclusions stable as the bootstrap block length L varies?
     -> sensitivity_to_L
  2. Are conclusions stable as the stop rule parameters vary?
     -> sensitivity_to_rule_params

Both produce DataFrames you can plot or pivot to see whether the stop's
advantage is robust or an artifact of specific parameter choices.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Callable, Union
from .simulation import generate_scenarios
from .stop_rules import StopRule, TrailingStopRule, NoStop
from .engine import run_backtest, BacktestResult


CapitalSpec = Union[float, dict]


def _resolve_capitals(initial_capital: CapitalSpec,
                      strategies) -> dict:
    """
    Normalize initial_capital into a dict[strategy -> float] lookup.

    - float: broadcast the same capital to every strategy
    - dict:  pass-through, but validate it covers every strategy
    """
    if isinstance(initial_capital, dict):
        missing = [s for s in strategies if s not in initial_capital]
        if missing:
            raise KeyError(f"initial_capital dict missing strategies: {missing}")
        return {s: float(initial_capital[s]) for s in strategies}
    return {s: float(initial_capital) for s in strategies}


def sensitivity_to_L(
    historical_returns: dict,
    rule_factory: Callable[[], StopRule],
    L_values: Sequence[float],
    baseline_factory: Callable[[], StopRule] = NoStop,
    n_paths: int = 10_000,
    path_length: int = 1260,
    initial_capital: float | dict = 10_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Re-run the full evaluation at multiple bootstrap block lengths.

    For each L, builds a fresh scenario set and backtests every strategy
    against both the treated rule and the baseline. Returns a long DataFrame
    with one row per (L, strategy, rule) combination.

    Rules are passed as FACTORIES (callables returning a fresh rule instance)
    because rules carry state and must be re-created for each backtest.

    Parameters
    ----------
    initial_capital : float or dict[str, float]
        If float, the same capital is used for every strategy (back-compat).
        If dict, it must contain an entry for every strategy in
        `historical_returns`; each strategy runs at its own capital. This
        matches the main analysis when stop thresholds are absolute dollars
        and capital differs across strategies.
    """
    cap_lookup = _resolve_capitals(initial_capital, historical_returns)

    rows = []
    for L in L_values:
        scenarios = generate_scenarios(
            historical_returns=historical_returns,
            n_paths=n_paths,
            path_length=path_length,
            L_mean=L,
            seed=seed,
        )
        for strat_name, paths in scenarios['paths'].items():
            cap = cap_lookup[strat_name]
            for factory in (baseline_factory, rule_factory):
                rule = factory()
                res = run_backtest(paths, rule, strat_name, cap)
                s = res.summary()
                s['L_mean'] = L
                rows.append(s)
    return pd.DataFrame(rows)


def sensitivity_to_rule_params(
    scenario_paths: dict,
    level_variants: Sequence[tuple],
    reentry_variants: Sequence[float],
    baseline: BacktestResult = None,
    initial_capital: CapitalSpec = 10_000_000.0,
) -> pd.DataFrame:
    """
    Sweep trailing-stop parameters against a fixed scenario set.

    Parameters
    ----------
    scenario_paths : dict[str, ndarray]
        Output of generate_scenarios()['paths']. Same scenarios used for all
        variants, so comparisons are apples-to-apples.
    level_variants : list of level-tuple-lists
        Each element is a list of (trigger_dd, size) tuples defining one rule.
        Example:
            [
                [(300_000, 0.70), (900_000, 0.40), (1_700_000, 0.0)],   # tight
                [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.0)], # base
                [(500_000, 0.70), (1_300_000, 0.40), (2_300_000, 0.0)], # loose
            ]
    reentry_variants : list of float
        Re-entry recovery amounts to try. Example: [0, 200_000, 300_000, 500_000].
    initial_capital : float or dict[str, float]
        If float, the same capital applies to every strategy (back-compat).
        If dict, must contain an entry for every strategy in scenario_paths.

    Returns
    -------
    DataFrame with one row per (strategy, level_variant, reentry_variant).

    Healthy rules show SMOOTH performance gradients as parameters vary. If
    base params look great but ±25% variants look terrible, you've overfit
    to arbitrary thresholds.
    """
    cap_lookup = _resolve_capitals(initial_capital, scenario_paths)

    rows = []
    for i, levels in enumerate(level_variants):
        for reentry in reentry_variants:
            for strat_name, paths in scenario_paths.items():
                rule = TrailingStopRule(
                    levels=list(levels),
                    reentry_recovery=reentry,
                    label=f'variant{i}_re{int(reentry/1000)}k',
                )
                res = run_backtest(paths, rule, strat_name,
                                   cap_lookup[strat_name])
                s = res.summary()
                s['variant_idx'] = i
                s['levels'] = str(levels)
                s['reentry_recovery'] = reentry
                rows.append(s)
    return pd.DataFrame(rows)


def sensitivity_to_capital(
    scenario_paths: dict,
    rule_factory: Callable[[], StopRule],
    capital_values: Sequence[CapitalSpec],
    baseline_factory: Callable[[], StopRule] = NoStop,
) -> pd.DataFrame:
    """
    Check how rule behavior changes with starting equity.

    Matters because absolute-dollar thresholds ($400k/$1.1m/$2m) behave very
    differently at $5m vs $20m starting capital. If the rule's relative
    advantage collapses at higher capital, the thresholds are sized for a
    specific account size and won't scale.

    Parameters
    ----------
    capital_values : sequence of float or dict
        Each entry defines one capital configuration to test. A float is
        broadcast to every strategy. A dict must contain an entry for every
        strategy and defines per-strategy capital (useful for sweeping
        allocation shapes, e.g. 0.5x / 1x / 2x of a base allocation).

    Returns
    -------
    DataFrame with a row per (capital_config, strategy, rule). Float inputs
    appear in the 'initial_capital' column as a number; dict inputs use a
    string repr so you can identify configurations in later pivot tables.
    """
    rows = []
    for cap_spec in capital_values:
        cap_lookup = _resolve_capitals(cap_spec, scenario_paths)
        # Label this capital configuration. For floats we keep the number
        # so pivots still work numerically; for dicts we use a repr string.
        cap_label = cap_spec if not isinstance(cap_spec, dict) else str(cap_spec)
        for strat_name, paths in scenario_paths.items():
            for factory in (baseline_factory, rule_factory):
                rule = factory()
                res = run_backtest(paths, rule, strat_name,
                                   cap_lookup[strat_name])
                s = res.summary()
                s['initial_capital'] = cap_label
                s['initial_capital_strat'] = cap_lookup[strat_name]
                rows.append(s)
    return pd.DataFrame(rows)
