"""
Stationary bootstrap simulation. Functional — no state worth encapsulating.

Generate ONE index matrix (n_paths x path_length) and reuse across all
strategies and stop rules for apples-to-apples comparison.
"""

import numpy as np
import pandas as pd
from arch.bootstrap import optimal_block_length


def politis_white_L(returns):
    """Politis-White optimal mean block length for the stationary bootstrap."""
    opt = optimal_block_length(np.asarray(returns))
    return float(opt['stationary'].iloc[0])


def stationary_bootstrap_indices(
    n_history: int,
    path_length: int,
    n_paths: int,
    L_mean: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate bootstrapped indices into the historical series.

    Returns an int matrix of shape (n_paths, path_length). Use it to look up
    returns for any strategy: strategy_returns[idx] -> simulated paths.

    Each path picks an independent random start; at every step there's a
    probability p = 1/L_mean of jumping to a fresh random point.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / L_mean

    jumps = rng.random((n_paths, path_length)) < p
    jumps[:, 0] = True  # force fresh random start at t=0 for each path
    new_starts = rng.integers(0, n_history, size=(n_paths, path_length))

    idx = np.empty((n_paths, path_length), dtype=np.int64)
    current = new_starts[:, 0].copy()
    idx[:, 0] = current
    for t in range(1, path_length):
        stepped = (current + 1) % n_history  # wraparound at end of history
        current = np.where(jumps[:, t], new_starts[:, t], stepped)
        idx[:, t] = current
    return idx


def apply_indices(strategy_returns, idx: np.ndarray) -> np.ndarray:
    """
    Look up returns from a strategy using the bootstrap index matrix.

    Parameters
    ----------
    strategy_returns : pd.Series or np.ndarray
        Daily returns aligned to the same historical dates used to build idx.
    idx : np.ndarray
        Output of stationary_bootstrap_indices().

    Returns
    -------
    np.ndarray of shape (n_paths, path_length): simulated daily returns.
    """
    arr = np.asarray(strategy_returns, dtype=np.float64)
    if idx.max() >= len(arr):
        raise ValueError(
            f"idx references position {idx.max()} but returns only have {len(arr)} entries."
        )
    return arr[idx]


def generate_scenarios(
    historical_returns: dict,
    n_paths: int = 10_000,
    path_length: int = 1260,
    L_mean: float | None = None,
    seed: int = 42,
) -> dict:
    """
    Top-level convenience: generate a scenario set usable across all strategies.

    Parameters
    ----------
    historical_returns : dict[str, pd.Series or np.ndarray]
        Mapping of strategy_name -> daily return series. All series must have
        the same length and be aligned to the same historical dates.
    n_paths : int
        Number of simulated paths (default 10,000).
    path_length : int
        Days per path (default 1260 ~= 5 business years).
    L_mean : float, optional
        Mean block length. If None, computed via Politis-White (max across
        strategies as a conservative default).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'idx'              : the index matrix (n_paths x path_length)
        'L_mean'           : the block length used
        'L_per_strategy'   : Politis-White L for each strategy
        'paths'            : dict[strategy_name -> ndarray of simulated returns]
    """
    lengths = {len(np.asarray(r)) for r in historical_returns.values()}
    if len(lengths) > 1:
        raise ValueError(f"All strategy return series must be same length, got {lengths}")
    n_history = lengths.pop()

    L_per_strategy = {name: politis_white_L(r) for name, r in historical_returns.items()}
    if L_mean is None:
        L_mean = max(L_per_strategy.values())
        L_mean = max(L_mean, 5.0)  # floor — Politis-White can return tiny values on noisy data

    idx = stationary_bootstrap_indices(n_history, path_length, n_paths, L_mean, seed=seed)
    paths = {name: apply_indices(r, idx) for name, r in historical_returns.items()}

    return {
        'idx': idx,
        'L_mean': L_mean,
        'L_per_strategy': L_per_strategy,
        'paths': paths,
    }
