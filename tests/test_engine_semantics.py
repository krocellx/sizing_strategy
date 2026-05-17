import unittest

import numpy as np

from src import (
    NoStop,
    RatioVolScaledTrailingStop,
    TrailingStopRule,
    combine_sleeves,
    run_backtest,
)


class SlowTrailingStopRule(TrailingStopRule):
    """Force the generic engine loop so tests can compare it to the fast path."""

    def run_fast_path(self, returns, initial_capital):
        raise NotImplementedError


class EngineSemanticsTest(unittest.TestCase):
    def test_quarterly_reset_resets_rule_hwm_and_notional(self):
        levels = [(280_000.0, 0.70)]
        returns = np.zeros((1, 80))
        returns[0, :63] = 0.0015
        returns[0, 63] = -0.027

        result = run_backtest(
            returns,
            TrailingStopRule(levels=levels, label="test"),
            "strategy",
            initial_capital=10_000_000.0,
            quarterly_reset=True,
            reset_every_days=63,
        )

        self.assertGreater(result.cash_flows[0, 0], 0.0)
        self.assertAlmostEqual(result.equity_curves[0, 63], 10_000_000.0)
        # If HWM were not reset, this day would breach the 280k trigger.
        self.assertEqual(result.position_sizes[0, 64], 1.0)

    def test_quarterly_reset_does_not_create_artificial_drawdown(self):
        returns = np.full((2, 252), 0.001)

        result = run_backtest(
            returns,
            NoStop(),
            "constant_positive",
            quarterly_reset=True,
            reset_every_days=63,
        )

        self.assertTrue(np.all(result.cash_flows > 0.0))
        self.assertTrue(np.allclose(result.max_drawdowns, 0.0))
        self.assertTrue(np.allclose(result.max_drawdown_pct, 0.0))
        self.assertTrue(
            np.allclose(
                result.terminal_wealth,
                result.equity_curves[:, -1] + result.total_cash_flows,
            )
        )

    def test_combined_sleeves_carry_cash_flows_into_cumulative_wealth(self):
        caps = {"A": 10_000_000.0, "B": 5_000_000.0}
        returns_a = np.full((3, 126), 0.001)
        returns_b = np.full((3, 126), 0.002)
        results = {
            ("A", "NoStop"): run_backtest(
                returns_a,
                NoStop(),
                "A",
                caps["A"],
                quarterly_reset=True,
                reset_every_days=63,
            ),
            ("B", "NoStop"): run_backtest(
                returns_b,
                NoStop(),
                "B",
                caps["B"],
                quarterly_reset=True,
                reset_every_days=63,
            ),
        }

        combined = combine_sleeves(results, ["A", "B"], "NoStop", caps)
        expected_wealth = (
            results[("A", "NoStop")].cumulative_wealth_curves
            + results[("B", "NoStop")].cumulative_wealth_curves
        )

        self.assertTrue(combined.quarterly_reset)
        self.assertTrue(np.allclose(combined.cumulative_wealth_curves, expected_wealth))
        self.assertGreater(combined.terminal_wealth[0], combined.equity_curves[0, -1])

    def test_trailing_stop_fast_path_matches_generic_loop(self):
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0002, 0.012, size=(20, 260))
        levels = [(400_000.0, 0.70), (1_100_000.0, 0.40), (2_000_000.0, 0.0)]

        fast = run_backtest(
            returns,
            TrailingStopRule(levels=levels, reentry_recovery=300_000.0),
            "strategy",
        )
        slow = run_backtest(
            returns,
            SlowTrailingStopRule(levels=levels, reentry_recovery=300_000.0),
            "strategy",
        )

        self.assertTrue(np.allclose(fast.equity_curves, slow.equity_curves))
        self.assertTrue(np.allclose(fast.position_sizes, slow.position_sizes))

    def test_ratio_vol_windows_are_explicit_and_unrestricted(self):
        rule = RatioVolScaledTrailingStop(
            base_levels=[(400_000.0, 0.70)],
            numerator_window=63,
            denominator_window=252,
        )

        self.assertEqual(rule.numerator_window, 63)
        self.assertEqual(rule.denominator_window, 252)


if __name__ == "__main__":
    unittest.main()
