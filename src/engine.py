"""
Backtest engine. Applies a StopRule to a matrix of simulated returns
and produces equity curves + summary statistics.

Vectorized over paths where possible; per-day loop is unavoidable because
stop rules are path-dependent (state depends on history).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .stop_rules import StopRule, TrailingStopRule, NoStop, VolScaledTrailingStop, RatioVolScaledTrailingStop

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# Numba-jitted fast path for TrailingStopRule (the common case).
# Generic StopRule subclasses fall back to the Python loop.
if _HAS_NUMBA:
    @njit(cache=True)
    def _trailing_stop_loop(returns, initial_capital,
                            level_dds, level_sizes, reentry_recovery):
        n_paths, n_days = returns.shape
        n_levels = level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))

        for p in range(n_paths):
            eq = initial_capital
            equity[p, 0] = eq
            hwm = eq
            cur_size = 1.0
            cur_level = -1

            for t in range(n_days):
                sizes[p, t] = cur_size
                eq = eq * (1.0 + cur_size * returns[p, t])
                equity[p, t + 1] = eq

                if eq > hwm:
                    hwm = eq
                    cur_size = 1.0
                    cur_level = -1
                    continue

                dd = hwm - eq

                # Single scan for both ratchet-down and re-entry.
                warranted = -1
                for i in range(n_levels):
                    if dd >= level_dds[i]:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = level_sizes[warranted]
                elif reentry_recovery > 0.0 and cur_level >= 0 and cur_size > 0.0:
                    # Threshold-based re-entry: DD retreated reentry_recovery
                    # below the current level's trigger.
                    recovery_from_trigger = level_dds[cur_level] - dd
                    if recovery_from_trigger >= reentry_recovery and warranted < cur_level:
                        cur_level = warranted
                        cur_size = 1.0 if warranted == -1 else level_sizes[warranted]

        return equity, sizes


    @njit(cache=True)
    def _vol_scaled_loop(returns, initial_capital,
                         base_level_dds, base_level_sizes,
                         base_reentry_recovery, reference_vol,
                         vol_window_days, refresh_mode_code,
                         monthly_days, seed_buffer):
        """
        Numba-jitted backtest for VolScaledTrailingStop.

        refresh_mode_code: 0 = 'monthly', 1 = 'hwm'

        seed_buffer: array of length up to vol_window_days containing the
            initial daily returns to seed the vol estimator. Pass a zero-
            length array if no seed.

        Also returns vol_mult_log: (n_paths, n_days) array of the vol_mult
        active when each day's size was decided (for diagnostics).
        """
        n_paths, n_days = returns.shape
        n_levels = base_level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))
        vol_mult_log = np.empty((n_paths, n_days))

        for p in range(n_paths):
            # Initialise ring buffer with seed.
            # Use float64 to match numba's default; size = vol_window_days.
            buf = np.zeros(vol_window_days, dtype=np.float64)
            buf_filled = 0   # how many real values are in buf
            buf_head = 0     # next write position

            n_seed = seed_buffer.shape[0]
            if n_seed > 0:
                take = min(n_seed, vol_window_days)
                for i in range(take):
                    buf[i] = seed_buffer[n_seed - take + i]
                buf_filled = take
                buf_head = take % vol_window_days

            # Compute initial vol_mult from seed (or 1.0 if no seed/too few obs).
            if buf_filled >= 2:
                mean = 0.0
                for i in range(buf_filled):
                    mean += buf[i]
                mean /= buf_filled
                var = 0.0
                for i in range(buf_filled):
                    diff = buf[i] - mean
                    var += diff * diff
                var /= (buf_filled - 1)
                vol = np.sqrt(var) * np.sqrt(252.0)
                vol_mult = vol / reference_vol
            else:
                vol_mult = 1.0

            eq = initial_capital
            equity[p, 0] = eq
            hwm = eq
            cur_size = 1.0
            cur_level = -1
            days_since_refresh = 0

            for t in range(n_days):
                # 1. Observe today's return — add to ring buffer.
                r = returns[p, t]
                buf[buf_head] = r
                buf_head = (buf_head + 1) % vol_window_days
                if buf_filled < vol_window_days:
                    buf_filled += 1
                days_since_refresh += 1

                # 2. Monthly refresh check.
                if refresh_mode_code == 0 and days_since_refresh >= monthly_days:
                    if buf_filled >= 2:
                        mean = 0.0
                        for i in range(buf_filled):
                            mean += buf[i]
                        mean /= buf_filled
                        var = 0.0
                        for i in range(buf_filled):
                            diff = buf[i] - mean
                            var += diff * diff
                        var /= (buf_filled - 1)
                        vol = np.sqrt(var) * np.sqrt(252.0)
                        vol_mult = vol / reference_vol
                    days_since_refresh = 0

                # 3. Record size decided for today.
                sizes[p, t] = cur_size
                vol_mult_log[p, t] = vol_mult

                # 4. Apply today's PnL.
                eq = eq * (1.0 + cur_size * r)
                equity[p, t + 1] = eq

                # 5. Update HWM (with optional vol_mult refresh on new high).
                if eq > hwm:
                    hwm = eq
                    cur_size = 1.0
                    cur_level = -1
                    if refresh_mode_code == 1 and buf_filled >= 2:
                        mean = 0.0
                        for i in range(buf_filled):
                            mean += buf[i]
                        mean /= buf_filled
                        var = 0.0
                        for i in range(buf_filled):
                            diff = buf[i] - mean
                            var += diff * diff
                        var /= (buf_filled - 1)
                        vol = np.sqrt(var) * np.sqrt(252.0)
                        vol_mult = vol / reference_vol
                        days_since_refresh = 0
                    continue

                dd = hwm - eq
                active_reentry = base_reentry_recovery * vol_mult

                # Single scan for both ratchet-down and re-entry.
                warranted = -1
                for i in range(n_levels):
                    if dd >= base_level_dds[i] * vol_mult:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = base_level_sizes[warranted]
                elif active_reentry > 0.0 and cur_level >= 0 and cur_size > 0.0:
                    # Threshold-based re-entry.
                    recovery_from_trigger = base_level_dds[cur_level] * vol_mult - dd
                    if recovery_from_trigger >= active_reentry and warranted < cur_level:
                        cur_level = warranted
                        cur_size = 1.0 if warranted == -1 else base_level_sizes[warranted]

        return equity, sizes, vol_mult_log


    @njit(cache=True)
    def _ratio_vol_scaled_loop(returns, initial_capital,
                               base_level_dds, base_level_sizes,
                               base_reentry_recovery,
                               short_window, long_window,
                               refresh_mode_code, monthly_days,
                               vol_mult_floor, vol_mult_cap):
        """
        Numba fast path for RatioVolScaledTrailingStop.

        vol_mult = clamp(σ_short / σ_long, floor, cap)

        During warmup (in-path days < long_window), vol_mult = 1.0.
        After warmup: ratio activates at the next refresh point (snap).

        refresh_mode_code: 0 = monthly, 1 = hwm
        No seed buffer needed — the warmup period handles initialisation.
        """
        n_paths, n_days = returns.shape
        n_levels = base_level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))
        vol_mult_log = np.empty((n_paths, n_days))

        for p in range(n_paths):
            # Ring buffers for short and long windows.
            short_buf = np.zeros(short_window)
            long_buf = np.zeros(long_window)
            short_filled = 0
            long_filled = 0
            short_head = 0
            long_head = 0

            vol_mult = 1.0
            vol_mult_locked = 1.0  # used to hold vol_mult during the day while we observe returns
            vol_locked = False  # whether vol_mult is currently locked (after observing return, before refresh)
            warmed_up = False
            days_since_refresh = 0
            in_path_days = 0

            eq = initial_capital
            equity[p, 0] = eq
            hwm = eq
            cur_size = 1.0
            cur_level = -1

            for t in range(n_days):
                r = returns[p, t]

                # --- observe_return: update ring buffers ---
                short_buf[short_head] = r
                short_head = (short_head + 1) % short_window
                if short_filled < short_window:
                    short_filled += 1

                long_buf[long_head] = r
                long_head = (long_head + 1) % long_window
                if long_filled < long_window:
                    long_filled += 1

                in_path_days += 1
                days_since_refresh += 1

                # Check warmup snap: activate on day long_window.
                if not warmed_up and in_path_days >= long_window:
                    warmed_up = True
                    # Compute vol_mult immediately (same as Python snap).
                    # Use full buffers at this point.
                    mean_s = 0.0
                    for i in range(short_filled):
                        mean_s += short_buf[i]
                    mean_s /= short_filled
                    var_s = 0.0
                    for i in range(short_filled):
                        d = short_buf[i] - mean_s
                        var_s += d * d
                    var_s /= (short_filled - 1)
                    sig_short = np.sqrt(var_s) * np.sqrt(252.0)

                    mean_l = 0.0
                    for i in range(long_filled):
                        mean_l += long_buf[i]
                    mean_l /= long_filled
                    var_l = 0.0
                    for i in range(long_filled):
                        d = long_buf[i] - mean_l
                        var_l += d * d
                    var_l /= (long_filled - 1)
                    sig_long = np.sqrt(var_l) * np.sqrt(252.0)

                    if sig_long > 0.0:
                        raw = sig_short / sig_long
                        if raw < vol_mult_floor:
                            raw = vol_mult_floor
                        if raw > vol_mult_cap:
                            raw = vol_mult_cap
                        vol_mult = raw
                    # Reset counter so next monthly refresh fires in monthly_days.
                    days_since_refresh = 0

                # Monthly refresh (only meaningful after warmup).
                if (refresh_mode_code == 0
                        and days_since_refresh >= monthly_days
                        and warmed_up
                        and not vol_locked):
                    # Compute σ_short.
                    mean_s = 0.0
                    for i in range(short_filled):
                        mean_s += short_buf[i]
                    mean_s /= short_filled
                    var_s = 0.0
                    for i in range(short_filled):
                        d = short_buf[i] - mean_s
                        var_s += d * d
                    var_s /= (short_filled - 1)
                    sig_short = np.sqrt(var_s) * np.sqrt(252.0)

                    # Compute σ_long.
                    mean_l = 0.0
                    for i in range(long_filled):
                        mean_l += long_buf[i]
                    mean_l /= long_filled
                    var_l = 0.0
                    for i in range(long_filled):
                        d = long_buf[i] - mean_l
                        var_l += d * d
                    var_l /= (long_filled - 1)
                    sig_long = np.sqrt(var_l) * np.sqrt(252.0)

                    if sig_long > 0.0:
                        raw = sig_short / sig_long
                        if raw < vol_mult_floor:
                            raw = vol_mult_floor
                        if raw > vol_mult_cap:
                            raw = vol_mult_cap
                        vol_mult = raw
                    days_since_refresh = 0

                # Record size and vol_mult for today.
                sizes[p, t] = cur_size
                vol_mult_log[p, t] = vol_mult

                # Apply today's PnL.
                eq = eq * (1.0 + cur_size * r)
                equity[p, t + 1] = eq

                # HWM update.
                if eq > hwm:
                    hwm = eq
                    cur_size = 1.0
                    cur_level = -1
                    vol_locked = False
                    # HWM refresh.
                    if refresh_mode_code == 1 and warmed_up and short_filled >= 2:
                        mean_s = 0.0
                        for i in range(short_filled):
                            mean_s += short_buf[i]
                        mean_s /= short_filled
                        var_s = 0.0
                        for i in range(short_filled):
                            d = short_buf[i] - mean_s
                            var_s += d * d
                        var_s /= (short_filled - 1)
                        sig_short = np.sqrt(var_s) * np.sqrt(252.0)

                        mean_l = 0.0
                        for i in range(long_filled):
                            mean_l += long_buf[i]
                        mean_l /= long_filled
                        var_l = 0.0
                        for i in range(long_filled):
                            d = long_buf[i] - mean_l
                            var_l += d * d
                        var_l /= (long_filled - 1)
                        sig_long = np.sqrt(var_l) * np.sqrt(252.0)

                        if sig_long > 0.0:
                            raw = sig_short / sig_long
                            if raw < vol_mult_floor:
                                raw = vol_mult_floor
                            if raw > vol_mult_cap:
                                raw = vol_mult_cap
                            vol_mult = raw
                        days_since_refresh = 0
                    continue
                
                vm = vol_mult_locked if vol_locked else vol_mult
                dd = hwm - eq
                active_reentry = base_reentry_recovery * vm

                # Single scan for both ratchet-down and re-entry.
                warranted = -1
                for i in range(n_levels):
                    trigger = base_level_dds[i]
                    if i < n_levels - 1:
                        trigger *= vm
                    if dd >= trigger:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = base_level_sizes[warranted]
                    if not vol_locked:
                        vol_mult_locked = vol_mult
                        vol_locked = True
                elif active_reentry > 0.0 and cur_level >= 0 and cur_size > 0.0:
                    # Threshold-based re-entry.
                    recovery_from_trigger = base_level_dds[cur_level] * vm - dd
                    if recovery_from_trigger >= active_reentry and warranted < cur_level:
                        cur_level = warranted
                        cur_size = 1.0 if warranted == -1 else base_level_sizes[warranted]

        return equity, sizes, vol_mult_log


@dataclass
class BacktestResult:
    strategy_name: str
    rule_name: str
    equity_curves: np.ndarray   # (n_paths, path_length + 1), starts at initial_capital
    position_sizes: np.ndarray  # (n_paths, path_length), size used each day
    initial_capital: float
    vol_mult_log: np.ndarray | None = None   # (n_paths, path_length), vol-scaled rules
    transaction_cost_bps: float = 0.0        # one-way cost applied, for reference
    cash_flows: np.ndarray | None = None     # (n_paths, n_quarters), NaN if no reset
    quarterly_reset: bool = False            # whether quarterly reset was applied

    @property
    def terminal_wealth(self) -> np.ndarray:
        return self.equity_curves[:, -1]

    @property
    def total_returns(self) -> np.ndarray:
        return self.terminal_wealth / self.initial_capital - 1.0

    @property
    def max_drawdowns(self) -> np.ndarray:
        """Max drawdown per path, in dollars (from running HWM)."""
        hwm = np.maximum.accumulate(self.equity_curves, axis=1)
        dd = hwm - self.equity_curves
        return dd.max(axis=1)

    @property
    def max_drawdown_pct(self) -> np.ndarray:
        hwm = np.maximum.accumulate(self.equity_curves, axis=1)
        dd_pct = (hwm - self.equity_curves) / hwm
        return dd_pct.max(axis=1)

    def summary(self) -> pd.Series:
        tr = self.total_returns
        dd = self.max_drawdowns
        dd_pct = self.max_drawdown_pct
        # Annualized metrics assume path_length is in business days.
        n_days = self.equity_curves.shape[1] - 1
        years = n_days / 252
        cagr = (self.terminal_wealth / self.initial_capital) ** (1 / years) - 1
        # Sharpe from per-path daily equity returns.
        daily_eq_returns = np.diff(self.equity_curves, axis=1) / self.equity_curves[:, :-1]
        sharpe = daily_eq_returns.mean(axis=1) / daily_eq_returns.std(axis=1) * np.sqrt(252)
        return pd.Series({
            'strategy': self.strategy_name,
            'rule': self.rule_name,
            'mean_total_return': tr.mean(),
            'median_total_return': np.median(tr),
            'p05_total_return': np.percentile(tr, 5),
            'p95_total_return': np.percentile(tr, 95),
            'mean_cagr': cagr.mean(),
            'mean_max_dd_$': dd.mean(),
            'p95_max_dd_$': np.percentile(dd, 95),
            'p99_max_dd_$': np.percentile(dd, 99),
            'mean_max_dd_pct': dd_pct.mean(),
            'p95_max_dd_pct': np.percentile(dd_pct, 95),
            'mean_sharpe': sharpe.mean(),
            'prob_loss': (tr < 0).mean(),
            'prob_50pct_dd': (dd_pct > 0.5).mean(),
        })


def _apply_quarterly_reset(
    equity: np.ndarray,
    sizes: np.ndarray,
    initial_capital: float,
    reset_every_days: int = 63,
) -> np.ndarray:
    """
    Apply quarterly notional resets in-place and return cash flow matrix.

    Rule:
      - At the end of each quarter (every reset_every_days trading days),
        check the current position size and equity.
      - If size == 1.0 AND equity > initial_capital (profit quarter):
          * Record cash flow = equity - initial_capital (withdrawal)
          * Reset equity to initial_capital
      - Otherwise: do nothing. Specifically:
          * size < 1.0 or size == 0.0: stop is active, no reset
          * size == 1.0 but equity <= initial_capital: underperforming,
            no top-up. The loss is carried forward into the next quarter.

    Parameters
    ----------
    equity : (n_paths, n_days+1), modified IN PLACE
    sizes  : (n_paths, n_days)
    initial_capital : float
    reset_every_days : int, default 63 (~1 quarter)

    Returns
    -------
    cash_flows : (n_paths, n_quarters) — cash flow at each quarter-end.
        Positive = investor receives money (profit taken out).
        Negative = investor contributes money (loss topped up).
        NaN = no reset happened this quarter (size was reduced).
    """
    n_paths, n_days_plus1 = equity.shape
    n_days = n_days_plus1 - 1
    reset_days = list(range(reset_every_days - 1, n_days, reset_every_days))
    n_quarters = len(reset_days)

    cash_flows = np.full((n_paths, n_quarters), np.nan)

    for p in range(n_paths):
        for q_idx, t in enumerate(reset_days):
            size_at_reset = sizes[p, t]
            eq_before = equity[p, t + 1]
            if size_at_reset == 1.0 and eq_before > initial_capital:
                cf = eq_before - initial_capital
                cash_flows[p, q_idx] = cf
                delta = eq_before - initial_capital
                equity[p, t + 1:] -= delta

    return cash_flows


def _apply_transaction_costs(
    equity: np.ndarray,
    sizes: np.ndarray,
    cost: float,
) -> np.ndarray:
    """
    Apply one-way transaction costs to an equity curve array in-place.

    Two cost events per path:
      1. Intra-period: on every day where position size changes, deduct
         |Δsize| × equity_eod × cost from equity. Applied end-of-day
         (after the day's return, before recording equity[t+1]).
      2. Terminal: on the final day, deduct current_size × terminal_equity
         × cost to reflect liquidation of the remaining position. This
         levels the playing field: NoStop pays its exit cost at the end,
         stop rules that already exited (size=0) pay nothing extra.

    Parameters
    ----------
    equity : (n_paths, path_length+1) — modified IN PLACE
    sizes  : (n_paths, path_length)
    cost   : one-way cost as a fraction (e.g. 0.0005 for 5bps)

    Returns
    -------
    equity (same array, modified in place for efficiency)
    """
    if cost == 0.0:
        return equity

    n_paths, n_days = sizes.shape

    # --- Intra-period costs ---
    # Δsize[t] = sizes[t] - sizes[t-1], with sizes[-1] = 1.0 (full at start).
    prev_sizes = np.concatenate(
        [np.ones((n_paths, 1)), sizes[:, :-1]], axis=1
    )
    delta = np.abs(sizes - prev_sizes)          # (n_paths, n_days)
    # Equity after day t's return is equity[:, t+1] before cost adjustment.
    # Cost is applied to that post-return equity.
    cost_amount = delta * equity[:, 1:] * cost  # (n_paths, n_days)

    # Propagate: a cost on day t reduces equity on all subsequent days
    # because the compounding base is lower. We do this correctly by
    # working forward: subtract cost from equity[t+1], then let the
    # subsequent returns compound from the reduced base.
    # Vectorised forward propagation using cumulative cost ratios:
    #   equity_net[t] = equity_gross[t] × ∏_{s≤t} (1 - cost_s/equity_gross[s])
    # But that's O(n²) per path. Instead, adjust equity in-place iteratively
    # using a running cost multiplier — cheap because cost events are sparse.
    for t in range(n_days):
        col_cost = cost_amount[:, t]            # (n_paths,)
        has_cost = col_cost > 0
        if has_cost.any():
            equity[has_cost, t + 1:] -= col_cost[has_cost, np.newaxis]

    # --- Terminal liquidation cost ---
    # Everyone pays cost to unwind their remaining position on the last day.
    # Stop rules already at size=0 pay nothing (they liquidated during the run).
    terminal_sizes = sizes[:, -1]               # (n_paths,)
    terminal_equity = equity[:, -1]             # (n_paths,)
    terminal_cost = terminal_sizes * terminal_equity * cost
    equity[:, -1] -= terminal_cost

    return equity


def run_backtest(
    strategy_returns_paths: np.ndarray,
    rule: StopRule,
    strategy_name: str,
    initial_capital: float = 10_000_000.0,
    transaction_cost_bps: float = 0.0,
    quarterly_reset: bool = False,
    reset_every_days: int = 63,
) -> BacktestResult:
    """
    Apply a stop rule to a matrix of simulated returns.

    Parameters
    ----------
    strategy_returns_paths : np.ndarray, shape (n_paths, path_length)
        Daily returns for one strategy across all simulated paths.
    rule : StopRule
        The position-sizing / stop rule to apply.
    strategy_name : str
        For labeling results.
    initial_capital : float
        Starting equity per path (default $10m).
    transaction_cost_bps : float
        One-way transaction cost in basis points (default 0).
        Applied on every size change (intra-period) and on the final
        day's remaining position (terminal liquidation). 5bps = 0.05.
    quarterly_reset : bool
        If True, reset notional to initial_capital at the end of each
        quarter (every reset_every_days). Only fires when size == 1.0
        (fully invested). Cash flows are tracked and stored in result.
    reset_every_days : int
        Trading days per quarter (default 63).

    Returns
    -------
    BacktestResult
    """
    n_paths, path_length = strategy_returns_paths.shape
    returns = np.ascontiguousarray(strategy_returns_paths, dtype=np.float64)
    cost = transaction_cost_bps / 10_000.0  # convert bps to fraction

    def _finalise(equity, sizes, vol_mult_log=None):
        """Apply post-processing and return BacktestResult."""
        _apply_transaction_costs(equity, sizes, cost)
        cfs = None
        if quarterly_reset:
            cfs = _apply_quarterly_reset(
                equity, sizes, initial_capital, reset_every_days
            )
        return BacktestResult(
            strategy_name=strategy_name,
            rule_name=rule.name,
            equity_curves=equity,
            position_sizes=sizes,
            initial_capital=initial_capital,
            vol_mult_log=vol_mult_log,
            transaction_cost_bps=transaction_cost_bps,
            cash_flows=cfs,
            quarterly_reset=quarterly_reset,
        )

    # Fast path: NoStop is trivial.
    if isinstance(rule, NoStop):
        equity = np.empty((n_paths, path_length + 1))
        equity[:, 0] = initial_capital
        equity[:, 1:] = initial_capital * np.cumprod(1.0 + returns, axis=1)
        sizes = np.ones((n_paths, path_length))
        return _finalise(equity, sizes)

    # Fast path: TrailingStopRule via numba.
    if _HAS_NUMBA and isinstance(rule, TrailingStopRule):
        sorted_levels = sorted(rule.levels, key=lambda x: x[0])
        level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        equity, sizes = _trailing_stop_loop(
            returns, float(initial_capital),
            level_dds, level_sizes, float(rule.reentry_recovery),
        )
        return _finalise(equity, sizes)

    # Fast path: VolScaledTrailingStop via numba.
    if _HAS_NUMBA and isinstance(rule, VolScaledTrailingStop):
        sorted_levels = sorted(rule.base_levels, key=lambda x: x[0])
        base_level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        base_level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        refresh_mode_code = 0 if rule.refresh_mode == 'monthly' else 1
        seed = (np.asarray(rule.initial_daily_returns, dtype=np.float64)
                if rule.initial_daily_returns is not None
                else np.zeros(0, dtype=np.float64))
        equity, sizes, vol_mult_log = _vol_scaled_loop(
            returns, float(initial_capital),
            base_level_dds, base_level_sizes,
            float(rule.base_reentry_recovery), float(rule.reference_vol),
            int(rule.vol_window_days), refresh_mode_code,
            int(rule.monthly_days), seed,
        )
        return _finalise(equity, sizes, vol_mult_log)

    # Fast path: RatioVolScaledTrailingStop via numba.
    if _HAS_NUMBA and isinstance(rule, RatioVolScaledTrailingStop):
        sorted_levels = sorted(rule.base_levels, key=lambda x: x[0])
        base_level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        base_level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        refresh_mode_code = 0 if rule.refresh_mode == 'monthly' else 1
        equity, sizes, vol_mult_log = _ratio_vol_scaled_loop(
            returns, float(initial_capital),
            base_level_dds, base_level_sizes,
            float(rule.base_reentry_recovery),
            int(rule.short_window), int(rule.long_window),
            refresh_mode_code, int(rule.monthly_days),
            float(rule.vol_mult_floor), float(rule.vol_mult_cap),
        )
        return _finalise(equity, sizes, vol_mult_log)

    # Generic fallback: Python loop, works for any StopRule subclass.
    equity = np.empty((n_paths, path_length + 1))
    equity[:, 0] = initial_capital
    sizes = np.empty((n_paths, path_length))

    is_vol_scaled = isinstance(rule, (VolScaledTrailingStop, RatioVolScaledTrailingStop))
    vol_mult_log = np.empty((n_paths, path_length)) if is_vol_scaled else None

    for p in range(n_paths):
        rule.reset(initial_capital)
        eq = initial_capital
        size = 1.0
        for t in range(path_length):
            r = strategy_returns_paths[p, t]
            rule.observe_return(r)
            sizes[p, t] = size
            eq = eq * (1 + size * r)
            equity[p, t + 1] = eq
            size = rule.update(eq)
            if is_vol_scaled:
                vol_mult_log[p, t] = rule.current_vol_mult

    return _finalise(equity, sizes, vol_mult_log)


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side comparison table of multiple backtest results."""
    return pd.DataFrame([r.summary() for r in results])
