"""
Disk-backed chunked backtest for large (n_paths, path_length) configurations.

Usage
-----
Use `run_backtest_chunked` when memory is tight — e.g., 10k paths over 10 years
across multiple strategies and rules. It produces a `CachedResult` that is a
drop-in replacement for `BacktestResult`: all standard analysis functions
(percentile_table, cvar_table, institutional_summary, paired_comparison,
bootstrap_ci, etc.) work without modification because summary arrays
(terminal_wealth, max_drawdowns, ...) are precomputed and cached in memory.

Equity curves and position sizes live on disk and are loaded lazily only when
a plot or a full-curve analysis needs them, then discarded from memory.

Use the original `run_backtest` when everything fits in RAM and you want
instant in-memory access (e.g., in a notebook iterating on plots).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import gc
import glob
import numpy as np
import pandas as pd

from .engine import run_backtest, BacktestResult
from .stop_rules import StopRule


DEFAULT_CACHE_DIR = Path('./_bt_cache')


@dataclass
class CachedResult:
    """
    Drop-in replacement for BacktestResult that holds summary arrays in memory
    but keeps equity_curves and position_sizes on disk.

    Exposes the same public surface as BacktestResult:
      - terminal_wealth, total_returns, cagr, calmar
      - max_drawdowns, max_drawdown_pct, rolling_returns
      - equity_curves, position_sizes  (lazy — loaded from disk on access)
      - summary()
      - strategy_name, rule_name, initial_capital

    Additional helpers:
      - load_curves(n_paths=None): load equity_curves array, optionally subset.
        Plots use this to subsample the fan (1000 paths is plenty for bands).
      - clear_curves_cache(): drop the in-memory curve cache after plotting.
    """

    strategy_name: str
    rule_name: str
    initial_capital: float
    cache_dir: Path

    # Pre-computed summary arrays (populated during chunked backtest).
    _terminal_wealth: np.ndarray = field(repr=False)
    _max_drawdowns: np.ndarray = field(repr=False)
    _max_drawdown_pct: np.ndarray = field(repr=False)
    # Per-path Sharpe (computed from daily equity-returns during chunking).
    _sharpe: np.ndarray = field(repr=False)
    _n_days: int = 0

    # Lazy-loaded curves (None until accessed).
    _equity_curves_cache: np.ndarray | None = field(default=None, repr=False)
    _position_sizes_cache: np.ndarray | None = field(default=None, repr=False)

    # ---- Eagerly-available summary statistics ------------------------------

    @property
    def n_days(self) -> int:
        return self._n_days

    @property
    def years(self) -> float:
        return self._n_days / 252

    @property
    def terminal_wealth(self) -> np.ndarray:
        return self._terminal_wealth

    @property
    def total_returns(self) -> np.ndarray:
        return self._terminal_wealth / self.initial_capital - 1.0

    @property
    def cagr(self) -> np.ndarray:
        return (self._terminal_wealth / self.initial_capital) ** (1 / self.years) - 1

    @property
    def max_drawdowns(self) -> np.ndarray:
        return self._max_drawdowns

    @property
    def max_drawdown_pct(self) -> np.ndarray:
        return self._max_drawdown_pct

    @property
    def calmar(self) -> np.ndarray:
        dd_pct = self._max_drawdown_pct
        return np.where(dd_pct > 0, self.cagr / dd_pct, np.nan)

    @property
    def sharpe(self) -> np.ndarray:
        return self._sharpe

    @property
    def cumulative_cash_flow_curves(self) -> np.ndarray:
        return np.zeros_like(self.equity_curves)

    @property
    def cumulative_wealth_curves(self) -> np.ndarray:
        return self.equity_curves

    @property
    def drawdown_curves(self) -> np.ndarray:
        eq = self.equity_curves
        hwm = np.maximum.accumulate(eq, axis=1)
        return hwm - eq

    @property
    def drawdown_pct_curves(self) -> np.ndarray:
        eq = self.equity_curves
        hwm = np.maximum.accumulate(eq, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(hwm > 0, (hwm - eq) / hwm, 0.0)

    def rolling_returns(
        self,
        window_days: int = 252,
        use_cumulative_wealth: bool = True,
    ) -> np.ndarray:
        curve = self.cumulative_wealth_curves if use_cumulative_wealth else self.equity_curves
        if curve.shape[1] <= window_days:
            return np.empty((curve.shape[0], 0))
        return curve[:, window_days:] / curve[:, :-window_days] - 1.0

    # ---- Lazy full-curve access --------------------------------------------

    @property
    def equity_curves(self) -> np.ndarray:
        if self._equity_curves_cache is None:
            self._equity_curves_cache = self.load_curves()
        return self._equity_curves_cache

    @property
    def position_sizes(self) -> np.ndarray:
        if self._position_sizes_cache is None:
            self._position_sizes_cache = self._load('sizes')
        return self._position_sizes_cache

    def load_curves(self, n_paths: int | None = None) -> np.ndarray:
        """
        Load equity curves from disk, optionally just the first n_paths.

        For fan plots, n_paths=1000 is plenty for p5/p25/p50/p75/p95 bands
        and massively reduces memory/IO cost.
        """
        return self._load('equity', n_paths)

    def _load(self, kind: str, n_paths: int | None = None) -> np.ndarray:
        chunks = sorted(self.cache_dir.glob(f'{kind}_*.npy'))
        if not chunks:
            raise FileNotFoundError(
                f"No cached {kind} files in {self.cache_dir}. "
                f"Was the backtest run with run_backtest_chunked?"
            )
        if n_paths is None:
            return np.concatenate([np.load(c) for c in chunks])
        # Only load as many chunks as needed.
        pieces, total = [], 0
        for c in chunks:
            arr = np.load(c)
            pieces.append(arr)
            total += len(arr)
            if total >= n_paths:
                break
        return np.concatenate(pieces)[:n_paths]

    def clear_curves_cache(self):
        """Drop the in-memory curve cache after plotting to free memory."""
        self._equity_curves_cache = None
        self._position_sizes_cache = None
        gc.collect()

    # ---- Summary ------------------------------------------------------------

    def summary(self) -> pd.Series:
        tr = self.total_returns
        dd = self._max_drawdowns
        dd_pct = self._max_drawdown_pct
        return pd.Series({
            'strategy': self.strategy_name,
            'rule': self.rule_name,
            'mean_total_return': tr.mean(),
            'median_total_return': np.median(tr),
            'p05_total_return': np.percentile(tr, 5),
            'p95_total_return': np.percentile(tr, 95),
            'mean_cagr': self.cagr.mean(),
            'mean_max_dd_$': dd.mean(),
            'p95_max_dd_$': np.percentile(dd, 95),
            'p99_max_dd_$': np.percentile(dd, 99),
            'mean_max_dd_pct': dd_pct.mean(),
            'p95_max_dd_pct': np.percentile(dd_pct, 95),
            'mean_sharpe': np.nanmean(self.sharpe),
            'prob_loss': (tr < 0).mean(),
            'prob_50pct_dd': (dd_pct > 0.5).mean(),
        })


def run_backtest_chunked(
    strategy_returns_paths: np.ndarray,
    rule: StopRule,
    strategy_name: str,
    initial_capital: float = 10_000_000.0,
    chunk_size: int = 1000,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    cache_key: str | None = None,
) -> CachedResult:
    """
    Run a backtest in chunks, persisting equity curves and position sizes
    to disk while keeping summary arrays in memory.

    Parameters
    ----------
    strategy_returns_paths : ndarray, shape (n_paths, path_length)
    rule : StopRule
        The rule to apply. Note: because run_backtest resets rule state per
        path internally, you can pass the same rule instance across chunks —
        but passing a fresh one is safer if you ever add stateful rules that
        don't clean up in reset().
    cache_key : str, optional
        Subdirectory name under cache_dir. Defaults to
        "{strategy_name}_{rule.name}". Stale files are removed at start.
    chunk_size : int
        Paths per chunk. Memory during backtest scales with chunk_size rather
        than n_paths.

    Returns
    -------
    CachedResult
        Drop-in replacement for BacktestResult with lazy-loaded equity_curves.
    """
    cache_dir = Path(cache_dir)
    key = cache_key or f'{strategy_name}_{rule.name}'.replace('/', '_').replace(' ', '_')
    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale cached files — we're starting fresh.
    for f in out_dir.glob('*.npy'):
        f.unlink()

    n_paths = strategy_returns_paths.shape[0]
    n_days = strategy_returns_paths.shape[1]

    terminal_list, maxdd_list, maxdd_pct_list, sharpe_list = [], [], [], []

    for i, start in enumerate(range(0, n_paths, chunk_size)):
        sub = strategy_returns_paths[start:start + chunk_size]
        res = run_backtest(sub, rule, strategy_name, initial_capital)

        # Persist the heavy arrays for this chunk.
        np.save(out_dir / f'equity_{i:04d}.npy', res.equity_curves.astype(np.float64))
        np.save(out_dir / f'sizes_{i:04d}.npy',  res.position_sizes.astype(np.float64))

        # Keep summary stats in memory — they're small and needed everywhere.
        terminal_list.append(res.terminal_wealth.copy())
        maxdd_list.append(res.max_drawdowns.copy())
        maxdd_pct_list.append(res.max_drawdown_pct.copy())

        # Compute per-path Sharpe now, while we have the curves in hand.
        daily = np.diff(res.equity_curves, axis=1) / res.equity_curves[:, :-1]
        mu = daily.mean(axis=1)
        sd = daily.std(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            sh = np.where(sd > 0, mu / sd * np.sqrt(252), np.nan)
        sharpe_list.append(sh)

        del res, daily, mu, sd, sh
        gc.collect()

    return CachedResult(
        strategy_name=strategy_name,
        rule_name=rule.name,
        initial_capital=initial_capital,
        cache_dir=out_dir,
        _terminal_wealth=np.concatenate(terminal_list),
        _max_drawdowns=np.concatenate(maxdd_list),
        _max_drawdown_pct=np.concatenate(maxdd_pct_list),
        _sharpe=np.concatenate(sharpe_list),
        _n_days=n_days,
    )
