"""
Stop / position-sizing rules. OOP because:
  - You'll have multiple rule types (trailing, fixed, vol-scaled, etc.)
  - Each rule has internal state (HWM, current size, drawdown level)
  - Rules need a common interface so the engine can plug any of them in

Each rule operates on ONE path at a time. The engine vectorizes over paths.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


class StopRule(ABC):
    """
    Abstract interface. A StopRule decides position size each day given
    the current equity. The engine drives it day-by-day.
    """

    @abstractmethod
    def reset(self, initial_capital: float) -> None:
        """Reset internal state at the start of a new path."""
        ...

    @abstractmethod
    def update(self, equity: float) -> float:
        """
        Called once per day AFTER the day's PnL is applied.
        Receives current equity, returns the position size multiplier
        for the NEXT day (1.0 = full, 0.0 = flat).
        """
        ...

    def observe_return(self, daily_return: float) -> None:
        """
        Optional hook called by engine BEFORE update() each day, passing the
        raw strategy daily return (not the post-stop return). Rules that
        track vol or other return-based diagnostics can override this.
        Default: do nothing.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NoStop(StopRule):
    """Baseline: always full size."""

    def reset(self, initial_capital: float) -> None:
        pass

    def update(self, equity: float) -> float:
        return 1.0

    @property
    def name(self) -> str:
        return "NoStop"


@dataclass
class TrailingStopRule(StopRule):
    """
    HWM-based trailing stop with tiered position reductions and re-entry.

    Drawdown is measured in dollars from the high-water mark (HWM).
    Levels are processed in ascending order of trigger_drawdown.

    Example matching your spec:
        levels = [
            (400_000, 0.70),    # at $400k DD from HWM, cut to 70%
            (1_100_000, 0.40),  # at $1.1m DD, cut to 40%
            (2_000_000, 0.00),  # at $2m DD, full stop out
        ]
        reentry_recovery = 300_000

    Re-entry: if equity recovers such that drawdown retreats reentry_recovery
    below the level's trigger threshold, step back up one level.
    """

    levels: list[tuple[float, float]]
    reentry_recovery: float = 0.0
    label: str = "TrailingStop"

    _hwm: float = field(default=0.0, init=False)
    _current_size: float = field(default=1.0, init=False)
    _current_level_idx: int = field(default=-1, init=False)
    _sorted_levels: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._sorted_levels = sorted(self.levels, key=lambda x: x[0])
        for i in range(1, len(self._sorted_levels)):
            if self._sorted_levels[i][1] >= self._sorted_levels[i - 1][1]:
                raise ValueError(
                    "Position sizes must be strictly decreasing as drawdown deepens."
                )

    def reset(self, initial_capital: float) -> None:
        self._hwm = initial_capital
        self._current_size = 1.0
        self._current_level_idx = -1

    def update(self, equity: float) -> float:
        if equity > self._hwm:
            self._hwm = equity
            self._current_size = 1.0
            self._current_level_idx = -1
            return 1.0

        drawdown = self._hwm - equity

        # Single scan — result used for both ratchet-down and re-entry.
        warranted_idx = -1
        for i, (trigger_dd, _) in enumerate(self._sorted_levels):
            if drawdown >= trigger_dd:
                warranted_idx = i
            else:
                break

        if warranted_idx > self._current_level_idx:
            # Ratchet down to deeper level.
            self._current_level_idx = warranted_idx
            self._current_size = self._sorted_levels[warranted_idx][1]

        elif self.reentry_recovery > 0 and self._current_level_idx >= 0:
            # Threshold-based re-entry:
            # Fire when DD has retreated reentry_recovery below the
            # current level's trigger — i.e., we've recovered enough
            # above the level that triggered this reduction.
            current_trigger = self._sorted_levels[self._current_level_idx][0]
            recovery_from_trigger = current_trigger - drawdown
            if recovery_from_trigger >= self.reentry_recovery:
                # warranted_idx already computed — step up to it.
                if warranted_idx < self._current_level_idx:
                    self._current_level_idx = warranted_idx
                    self._current_size = (
                        1.0 if warranted_idx == -1
                        else self._sorted_levels[warranted_idx][1]
                    )

        return self._current_size

    @property
    def name(self) -> str:
        return self.label

    def run_fast_path(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """Run the numba implementation that mirrors this rule's update logic."""
        if not _HAS_NUMBA:
            raise NotImplementedError("numba is not available")
        sorted_levels = sorted(self.levels, key=lambda x: x[0])
        level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        equity, sizes = _trailing_stop_loop(
            returns,
            float(initial_capital),
            level_dds,
            level_sizes,
            float(self.reentry_recovery),
        )
        return equity, sizes, None


@dataclass
class VolScaledTrailingStop(StopRule):
    """
    Vol-scaled trailing stop. Same tiered structure as TrailingStopRule, but
    trigger thresholds and reentry recovery scale with current realised vol.

    Idea
    ----
    A fixed-dollar stop is too tight in high-vol regimes (whipsaws on noise)
    and too loose in low-vol regimes (allows real damage to compound). By
    scaling thresholds to current vol, the rule maintains a constant
    "vol-units-of-drawdown" trigger across regimes.

    Mechanics
    ---------
    At each refresh point, compute:
        vol_mult = current_realised_vol / reference_vol
    Then the active trigger and reentry are:
        active_trigger_i = base_trigger_i * vol_mult
        active_reentry   = base_reentry   * vol_mult

    Parameters
    ----------
    base_levels : list of (base_trigger_dd, size)
        Trigger thresholds at reference_vol. e.g. [(700_000, 0.70), ...].
        These are the trigger sizes when current_vol == reference_vol.
    base_reentry_recovery : float
        Recovery threshold at reference_vol. Scaled the same way.
    reference_vol : float
        Annualised vol at which base_levels apply unscaled. Default 0.15.
        Use the strategy's long-run average vol for cleanest interpretation.
    vol_window_days : int
        Trailing window for realised vol estimation. Default 252.
    refresh_mode : {'monthly', 'hwm'}
        - 'monthly': vol_mult updates every `monthly_days` trading days.
        - 'hwm':     vol_mult updates only when a new HWM is set.
    monthly_days : int
        Refresh cadence for 'monthly' mode. Default 21.
    initial_daily_returns : np.ndarray or None
        Optional history of returns to seed the vol estimate at path start
        (so vol_mult is accurate from day 1 rather than after warmup). Use
        recent historical returns of the strategy.

    Notes
    -----
    - Vol uses STRATEGY returns (raw, pre-stop) via observe_return(). After
      a stop-out, equity is flat but the underlying strategy keeps moving;
      we want vol to reflect that.
    - Without seeding, the rule uses reference_vol as the initial estimate
      and updates as in-path returns accumulate.
    """

    base_levels: list[tuple[float, float]]
    base_reentry_recovery: float = 0.0
    reference_vol: float = 0.15
    vol_window_days: int = 252
    refresh_mode: str = "monthly"
    monthly_days: int = 21
    initial_daily_returns: np.ndarray | None = None
    label: str = "VolScaledTrailingStop"

    _hwm: float = field(default=0.0, init=False)
    _current_size: float = field(default=1.0, init=False)
    _current_level_idx: int = field(default=-1, init=False)
    _sorted_base_levels: list = field(default_factory=list, init=False)
    _vol_mult: float = field(default=1.0, init=False)
    _return_buffer: list = field(default_factory=list, init=False)
    _days_since_refresh: int = field(default=0, init=False)

    def __post_init__(self):
        self._sorted_base_levels = sorted(self.base_levels, key=lambda x: x[0])
        for i in range(1, len(self._sorted_base_levels)):
            if self._sorted_base_levels[i][1] >= self._sorted_base_levels[i - 1][1]:
                raise ValueError(
                    "Position sizes must be strictly decreasing as drawdown deepens."
                )
        if self.refresh_mode not in ("monthly", "hwm"):
            raise ValueError(
                f"refresh_mode must be 'monthly' or 'hwm', got {self.refresh_mode!r}"
            )

    def reset(self, initial_capital: float) -> None:
        self._hwm = initial_capital
        self._current_size = 1.0
        self._current_level_idx = -1

        # Seed return buffer with historical returns if provided.
        if self.initial_daily_returns is not None:
            seed = np.asarray(self.initial_daily_returns, dtype=float)
            self._return_buffer = list(seed[-self.vol_window_days:])
        else:
            self._return_buffer = []
        self._days_since_refresh = 0
        # Compute initial vol_mult from seed (or fall back to 1.0).
        self._refresh_vol_mult()

    def _current_realised_vol(self) -> float:
        """Annualised realised vol from buffer, or reference_vol if too few obs."""
        if len(self._return_buffer) < 2:
            return self.reference_vol
        return float(np.std(self._return_buffer, ddof=1) * np.sqrt(252))

    def _refresh_vol_mult(self) -> None:
        """Recompute vol_mult from current buffer."""
        current_vol = self._current_realised_vol()
        self._vol_mult = current_vol / self.reference_vol
        self._days_since_refresh = 0

    def observe_return(self, daily_return: float) -> None:
        """Track raw strategy return for vol estimation."""
        self._return_buffer.append(daily_return)
        if len(self._return_buffer) > self.vol_window_days:
            self._return_buffer.pop(0)
        self._days_since_refresh += 1

    def update(self, equity: float) -> float:
        # In monthly mode, refresh on cadence.
        if (self.refresh_mode == "monthly"
                and self._days_since_refresh >= self.monthly_days):
            self._refresh_vol_mult()

        active_reentry = self.base_reentry_recovery * self._vol_mult

        if equity > self._hwm:
            self._hwm = equity
            self._current_size = 1.0
            self._current_level_idx = -1
            if self.refresh_mode == "hwm":
                self._refresh_vol_mult()
            return 1.0

        drawdown = self._hwm - equity

        # Single scan — result used for both ratchet-down and re-entry.
        warranted_idx = -1
        for i, (base_trigger, _) in enumerate(self._sorted_base_levels):
            if drawdown >= base_trigger * self._vol_mult:
                warranted_idx = i
            else:
                break

        if warranted_idx > self._current_level_idx:
            self._current_level_idx = warranted_idx
            self._current_size = self._sorted_base_levels[warranted_idx][1]

        elif active_reentry > 0 and self._current_level_idx >= 0:
            # Threshold-based re-entry: DD has retreated active_reentry
            # below the current level's (vol-scaled) trigger.
            current_trigger = (
                self._sorted_base_levels[self._current_level_idx][0] * self._vol_mult
            )
            recovery_from_trigger = current_trigger - drawdown
            if recovery_from_trigger >= active_reentry:
                if warranted_idx < self._current_level_idx:
                    self._current_level_idx = warranted_idx
                    self._current_size = (
                        1.0 if warranted_idx == -1
                        else self._sorted_base_levels[warranted_idx][1]
                    )

        return self._current_size

    @property
    def name(self) -> str:
        return self.label

    @property
    def current_vol_mult(self) -> float:
        """Diagnostic: current vol multiplier."""
        return self._vol_mult

    def run_fast_path(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the numba implementation that mirrors this rule's update logic."""
        if not _HAS_NUMBA:
            raise NotImplementedError("numba is not available")
        sorted_levels = sorted(self.base_levels, key=lambda x: x[0])
        base_level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        base_level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        refresh_mode_code = 0 if self.refresh_mode == "monthly" else 1
        seed = (
            np.asarray(self.initial_daily_returns, dtype=np.float64)
            if self.initial_daily_returns is not None
            else np.zeros(0, dtype=np.float64)
        )
        return _vol_scaled_loop(
            returns,
            float(initial_capital),
            base_level_dds,
            base_level_sizes,
            float(self.base_reentry_recovery),
            float(self.reference_vol),
            int(self.vol_window_days),
            refresh_mode_code,
            int(self.monthly_days),
            seed,
        )


@dataclass
class RatioVolScaledTrailingStop(StopRule):
    """
    Path-dependent vol-scaled trailing stop.

    Vol multiplier is self-normalising — no reference vol parameter:

        VR_t = σ̂_63,t / σ̂_252,t

    where both are annualised trailing realised vols of the raw strategy
    daily returns (pre-stop). A ratio > 1 means the recent 3-month vol is
    elevated relative to the trailing year (stress regime) → thresholds widen.
    A ratio < 1 means unusually calm → thresholds tighten.

    Warmup: during the first 252 in-path days, insufficient history exists
    to compute σ̂_252 reliably. The rule uses fixed-dollar thresholds
    (vol_mult = 1.0) during this period. On day 252, the ratio activates
    immediately (snap, not blend).

    Vol_mult is clamped to [vol_mult_floor, vol_mult_cap] to prevent
    degenerate threshold collapse or explosion in tail scenarios.

    Parameters
    ----------
    base_levels : list of (base_trigger_dd, size)
        Trigger thresholds when VR_t = 1 (short-term vol = annual vol).
    base_reentry_recovery : float
        Recovery threshold when VR_t = 1. Scales same as triggers.
    short_window : int
        Short vol window in days. Default 63 (≈ 3 months).
    long_window : int
        Long vol window in days. Default 252 (≈ 1 year). Also sets warmup.
    refresh_mode : {'monthly', 'hwm'}
        When to recompute VR_t.
        - 'monthly': every `monthly_days` trading days.
        - 'hwm':     only when a new HWM is set.
    monthly_days : int
        Cadence for 'monthly' mode. Default 21.
    vol_mult_floor : float
        Minimum vol_mult. Default 0.25 (prevents thresholds shrinking too far
        in calm regimes). Triggers can't go below 25% of base.
    vol_mult_cap : float
        Maximum vol_mult. Default 4.0 (prevents thresholds growing >4x
        even in extreme stress).
    label : str

    Notes
    -----
    - Both windows use raw strategy daily returns via observe_return(), so vol
      estimates remain valid even after full stop-out (flat equity ≠ zero vol).
    - During warmup (days 0-251), vol_mult = 1.0 and the rule behaves exactly
      like TrailingStopRule with the same base_levels.
    - VR_t uses lag-1 estimates (yesterday's rolling vol computed after
      observe_return() updates the buffer). This ensures no look-ahead.
    """

    base_levels: list[tuple[float, float]]
    base_reentry_recovery: float = 0.0
    short_window: int = 63
    long_window: int = 252
    refresh_mode: str = "monthly"
    monthly_days: int = 21
    vol_mult_floor: float = 0.25
    vol_mult_cap: float = 4.0
    label: str = "RatioVolScaled"

    # Internal state
    _hwm: float = field(default=0.0, init=False)
    _current_size: float = field(default=1.0, init=False)
    _current_level_idx: int = field(default=-1, init=False)
    _sorted_base_levels: list = field(default_factory=list, init=False)
    _vol_mult: float = field(default=1.0, init=False)
    _short_buffer: list = field(default_factory=list, init=False)
    _long_buffer: list = field(default_factory=list, init=False)
    _days_since_refresh: int = field(default=0, init=False)
    _in_path_days: int = field(default=0, init=False)
    _warmed_up: bool = field(default=False, init=False)

    def __post_init__(self):
        self._sorted_base_levels = sorted(self.base_levels, key=lambda x: x[0])
        for i in range(1, len(self._sorted_base_levels)):
            if self._sorted_base_levels[i][1] >= self._sorted_base_levels[i - 1][1]:
                raise ValueError(
                    "Position sizes must be strictly decreasing as drawdown deepens."
                )
        if self.refresh_mode not in ("monthly", "hwm"):
            raise ValueError(
                f"refresh_mode must be 'monthly' or 'hwm', got {self.refresh_mode!r}"
            )
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be < long_window")

    def reset(self, initial_capital: float) -> None:
        self._hwm = initial_capital
        self._current_size = 1.0
        self._current_level_idx = -1
        self._vol_mult = 1.0
        self._short_buffer = []
        self._long_buffer = []
        self._days_since_refresh = 0
        self._in_path_days = 0
        self._warmed_up = False

    @staticmethod
    def _annualised_vol(buf: list) -> float:
        if len(buf) < 2:
            return np.nan
        return float(np.std(buf, ddof=1) * np.sqrt(252))

    def _compute_vol_mult(self) -> float:
        """
        Compute VR_t = σ_63 / σ_252.
        Returns 1.0 if either estimate is unreliable (nan or zero).
        """
        sig_short = self._annualised_vol(self._short_buffer)
        sig_long = self._annualised_vol(self._long_buffer)
        if np.isnan(sig_short) or np.isnan(sig_long) or sig_long == 0:
            return 1.0
        raw = sig_short / sig_long
        return float(np.clip(raw, self.vol_mult_floor, self.vol_mult_cap))

    def _refresh_vol_mult(self) -> None:
        if self._warmed_up:
            self._vol_mult = self._compute_vol_mult()
        else:
            self._vol_mult = 1.0
        self._days_since_refresh = 0

    def observe_return(self, daily_return: float) -> None:
        """Track raw strategy return (pre-stop) for vol estimation."""
        self._short_buffer.append(daily_return)
        if len(self._short_buffer) > self.short_window:
            self._short_buffer.pop(0)
        self._long_buffer.append(daily_return)
        if len(self._long_buffer) > self.long_window:
            self._long_buffer.pop(0)
        self._in_path_days += 1
        self._days_since_refresh += 1
        # Snap warmup to active on day long_window.
        if not self._warmed_up and self._in_path_days >= self.long_window:
            self._warmed_up = True
            # Immediately refresh so next update uses the ratio.
            self._refresh_vol_mult()

    def update(self, equity: float) -> float:
        if (self.refresh_mode == "monthly"
                and self._days_since_refresh >= self.monthly_days):
            self._refresh_vol_mult()

        active_reentry = self.base_reentry_recovery * self._vol_mult

        if equity > self._hwm:
            self._hwm = equity
            self._current_size = 1.0
            self._current_level_idx = -1
            if self.refresh_mode == "hwm":
                self._refresh_vol_mult()
            return 1.0

        drawdown = self._hwm - equity

        # Single scan — result used for both ratchet-down and re-entry.
        warranted_idx = -1
        for i, (base_trigger, _) in enumerate(self._sorted_base_levels):
            if drawdown >= base_trigger * self._vol_mult:
                warranted_idx = i
            else:
                break

        if warranted_idx > self._current_level_idx:
            self._current_level_idx = warranted_idx
            self._current_size = self._sorted_base_levels[warranted_idx][1]

        elif active_reentry > 0 and self._current_level_idx >= 0:
            # Threshold-based re-entry: DD has retreated active_reentry
            # below the current level's (vol-scaled) trigger.
            current_trigger = (
                self._sorted_base_levels[self._current_level_idx][0] * self._vol_mult
            )
            recovery_from_trigger = current_trigger - drawdown
            if recovery_from_trigger >= active_reentry:
                if warranted_idx < self._current_level_idx:
                    self._current_level_idx = warranted_idx
                    self._current_size = (
                        1.0 if warranted_idx == -1
                        else self._sorted_base_levels[warranted_idx][1]
                    )

        return self._current_size

    @property
    def name(self) -> str:
        return self.label

    @property
    def current_vol_mult(self) -> float:
        return self._vol_mult

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    def run_fast_path(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the numba implementation that mirrors this rule's update logic."""
        if not _HAS_NUMBA:
            raise NotImplementedError("numba is not available")
        sorted_levels = sorted(self.base_levels, key=lambda x: x[0])
        base_level_dds = np.array([l[0] for l in sorted_levels], dtype=np.float64)
        base_level_sizes = np.array([l[1] for l in sorted_levels], dtype=np.float64)
        refresh_mode_code = 0 if self.refresh_mode == "monthly" else 1
        return _ratio_vol_scaled_loop(
            returns,
            float(initial_capital),
            base_level_dds,
            base_level_sizes,
            float(self.base_reentry_recovery),
            int(self.short_window),
            int(self.long_window),
            refresh_mode_code,
            int(self.monthly_days),
            float(self.vol_mult_floor),
            float(self.vol_mult_cap),
        )


# Numba fast paths live with their owning rules so the optimized path is
# reviewed next to the Python state-machine implementation.
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
                warranted = -1
                for i in range(n_levels):
                    if dd >= level_dds[i]:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = level_sizes[warranted]
                elif reentry_recovery > 0.0 and cur_level >= 0:
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
        n_paths, n_days = returns.shape
        n_levels = base_level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))
        vol_mult_log = np.empty((n_paths, n_days))

        for p in range(n_paths):
            buf = np.zeros(vol_window_days, dtype=np.float64)
            buf_filled = 0
            buf_head = 0

            n_seed = seed_buffer.shape[0]
            if n_seed > 0:
                take = min(n_seed, vol_window_days)
                for i in range(take):
                    buf[i] = seed_buffer[n_seed - take + i]
                buf_filled = take
                buf_head = take % vol_window_days

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
                r = returns[p, t]
                buf[buf_head] = r
                buf_head = (buf_head + 1) % vol_window_days
                if buf_filled < vol_window_days:
                    buf_filled += 1
                days_since_refresh += 1

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

                sizes[p, t] = cur_size
                vol_mult_log[p, t] = vol_mult

                eq = eq * (1.0 + cur_size * r)
                equity[p, t + 1] = eq

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
                warranted = -1
                for i in range(n_levels):
                    if dd >= base_level_dds[i] * vol_mult:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = base_level_sizes[warranted]
                elif active_reentry > 0.0 and cur_level >= 0:
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
        n_paths, n_days = returns.shape
        n_levels = base_level_dds.shape[0]
        equity = np.empty((n_paths, n_days + 1))
        sizes = np.empty((n_paths, n_days))
        vol_mult_log = np.empty((n_paths, n_days))

        for p in range(n_paths):
            short_buf = np.zeros(short_window)
            long_buf = np.zeros(long_window)
            short_filled = 0
            long_filled = 0
            short_head = 0
            long_head = 0

            vol_mult = 1.0
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

                if not warmed_up and in_path_days >= long_window:
                    warmed_up = True
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

                if (refresh_mode_code == 0
                        and days_since_refresh >= monthly_days
                        and warmed_up):
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

                sizes[p, t] = cur_size
                vol_mult_log[p, t] = vol_mult

                eq = eq * (1.0 + cur_size * r)
                equity[p, t + 1] = eq

                if eq > hwm:
                    hwm = eq
                    cur_size = 1.0
                    cur_level = -1
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

                dd = hwm - eq
                active_reentry = base_reentry_recovery * vol_mult
                warranted = -1
                for i in range(n_levels):
                    if dd >= base_level_dds[i] * vol_mult:
                        warranted = i
                    else:
                        break

                if warranted > cur_level:
                    cur_level = warranted
                    cur_size = base_level_sizes[warranted]
                elif active_reentry > 0.0 and cur_level >= 0:
                    recovery_from_trigger = base_level_dds[cur_level] * vol_mult - dd
                    if recovery_from_trigger >= active_reentry and warranted < cur_level:
                        cur_level = warranted
                        cur_size = 1.0 if warranted == -1 else base_level_sizes[warranted]

        return equity, sizes, vol_mult_log
