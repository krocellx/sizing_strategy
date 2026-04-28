"""
Stop / position-sizing rules. OOP because:
  - You'll have multiple rule types (trailing, fixed, ATR-based, etc.)
  - Each rule has internal state (HWM, current size, drawdown level)
  - Rules need a common interface so the engine can plug any of them in

Each rule operates on ONE path at a time. The engine vectorizes over paths.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


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
            (400_000, 0.70),    # at $400k DD from HWM, cut to 70% (i.e., reduce by 30%)
            (1_100_000, 0.40),  # at $1.1m DD, cut to 40% (reduce by 60%)
            (2_000_000, 0.00),  # at $2m DD, full stop out
        ]
        reentry_recovery = 300_000  # if we recover $300k from the trough,
                                    # step back up one level

    Re-entry logic: tracks the lowest equity reached while in a reduced state.
    If equity recovers by `reentry_recovery` dollars from that trough AND
    drawdown from HWM is now below the previous level's trigger, step back up.
    """

    levels: list[tuple[float, float]]
    reentry_recovery: float = 0.0
    label: str = "TrailingStop"

    # Internal state — populated by reset()
    _hwm: float = field(default=0.0, init=False)
    _trough: float = field(default=0.0, init=False)
    _current_size: float = field(default=1.0, init=False)
    _current_level_idx: int = field(default=-1, init=False)  # -1 = no reduction active
    _sorted_levels: list = field(default_factory=list, init=False)

    def __post_init__(self):
        # Sort levels by trigger DD ascending and validate.
        self._sorted_levels = sorted(self.levels, key=lambda x: x[0])
        for i in range(1, len(self._sorted_levels)):
            if self._sorted_levels[i][1] >= self._sorted_levels[i - 1][1]:
                raise ValueError(
                    "Position sizes must be strictly decreasing as drawdown deepens."
                )

    def reset(self, initial_capital: float) -> None:
        self._hwm = initial_capital
        self._trough = initial_capital
        self._current_size = 1.0
        self._current_level_idx = -1

    def update(self, equity: float) -> float:
        # Update HWM. A new high resets everything — we're back to full size.
        if equity > self._hwm:
            self._hwm = equity
            self._trough = equity
            self._current_size = 1.0
            self._current_level_idx = -1
            return 1.0

        # Update trough (lowest equity since last HWM).
        if equity < self._trough:
            self._trough = equity

        drawdown = self._hwm - equity

        # Find the deepest level triggered by current drawdown.
        triggered_idx = -1
        for i, (trigger_dd, _) in enumerate(self._sorted_levels):
            if drawdown >= trigger_dd:
                triggered_idx = i
            else:
                break

        # Ratchet DOWN: if we've hit a deeper level than current, reduce size.
        if triggered_idx > self._current_level_idx:
            self._current_level_idx = triggered_idx
            self._current_size = self._sorted_levels[triggered_idx][1]
            self._trough = equity  # reset trough for re-entry tracking

        # Ratchet UP (re-entry): if equity has recovered enough from trough
        # AND drawdown has retreated below the current level's trigger, step up.
        elif self.reentry_recovery > 0 and self._current_level_idx >= 0:
            recovery = equity - self._trough
            if recovery >= self.reentry_recovery:
                # Determine what level we should be at given current DD.
                new_level_idx = -1
                for i, (trigger_dd, _) in enumerate(self._sorted_levels):
                    if drawdown >= trigger_dd:
                        new_level_idx = i
                    else:
                        break
                if new_level_idx < self._current_level_idx:
                    self._current_level_idx = new_level_idx
                    self._current_size = (
                        1.0 if new_level_idx == -1
                        else self._sorted_levels[new_level_idx][1]
                    )
                    self._trough = equity  # reset for next potential re-entry

        return self._current_size

    @property
    def name(self) -> str:
        return self.label
