import unittest

import numpy as np

from src import (
    BacktestResult,
    NoStop,
    RatioVolScaledTrailingStop,
    TrailingStopRule,
    VolScaledTrailingStop,
    combine_sleeves,
    run_backtest,
)
from src.institutional import stopout_pct_table


class SlowTrailingStopRule(TrailingStopRule):
    """Force the generic engine loop so tests can compare it to the fast path."""

    def run_fast_path(self, returns, initial_capital):
        raise NotImplementedError


class SlowRatioVolScaledTrailingStop(RatioVolScaledTrailingStop):
    """Force the generic engine loop for ratio-vol stop parity tests."""

    def run_fast_path(self, returns, initial_capital):
        raise NotImplementedError


class SlowVolScaledTrailingStop(VolScaledTrailingStop):
    """Force the generic engine loop for realised-vol stop parity tests."""

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

    def test_ratio_vol_full_stop_is_hard_not_vol_scaled(self):
        rule = RatioVolScaledTrailingStop(
            base_levels=[
                (700_000.0, 0.70),
                (1_400_000.0, 0.40),
                (2_000_000.0, 0.0),
            ],
            base_reentry_recovery=300_000.0,
        )
        rule.reset(10_000_000.0)
        rule._vol_mult = 0.25
        rule._warmed_up = True

        size = rule.update(9_000_000.0)

        self.assertEqual(size, 0.40)
        self.assertTrue(rule._vol_mult_locked)
        self.assertEqual(rule._locked_vol_mult, 0.25)

    def test_ratio_vol_multiplier_locks_until_back_to_full(self):
        rule = RatioVolScaledTrailingStop(
            base_levels=[
                (700_000.0, 0.70),
                (1_400_000.0, 0.40),
                (2_000_000.0, 0.0),
            ],
            base_reentry_recovery=300_000.0,
        )
        rule.reset(10_000_000.0)
        rule._vol_mult = 0.50
        rule._warmed_up = True

        self.assertEqual(rule.update(9_600_000.0), 0.70)
        rule._vol_mult = 4.00
        self.assertEqual(rule.update(9_200_000.0), 0.40)
        self.assertTrue(rule._vol_mult_locked)
        self.assertEqual(rule._locked_vol_mult, 0.50)

        self.assertEqual(rule.update(9_750_000.0), 1.00)
        self.assertFalse(rule._vol_mult_locked)

    def test_realised_vol_stop_uses_hard_full_stop_and_locked_multiplier(self):
        rule = VolScaledTrailingStop(
            base_levels=[
                (700_000.0, 0.70),
                (1_400_000.0, 0.40),
                (2_000_000.0, 0.0),
            ],
            base_reentry_recovery=300_000.0,
        )
        rule.reset(10_000_000.0)
        rule._vol_mult = 0.50

        self.assertEqual(rule.update(9_600_000.0), 0.70)
        rule._vol_mult = 4.00
        self.assertEqual(rule.update(9_200_000.0), 0.40)
        self.assertTrue(rule._vol_mult_locked)
        self.assertEqual(rule._locked_vol_mult, 0.50)

        self.assertEqual(rule.update(9_750_000.0), 1.00)
        self.assertFalse(rule._vol_mult_locked)

    def test_ratio_vol_fast_path_matches_generic_loop(self):
        rng = np.random.default_rng(456)
        returns = rng.normal(0.0001, 0.025, size=(12, 180))
        levels = [
            (700_000.0, 0.70),
            (1_400_000.0, 0.40),
            (2_000_000.0, 0.0),
        ]

        fast = run_backtest(
            returns,
            RatioVolScaledTrailingStop(
                base_levels=levels,
                base_reentry_recovery=300_000.0,
                numerator_window=21,
                denominator_window=63,
                monthly_days=5,
                vol_mult_floor=0.25,
                vol_mult_cap=4.0,
            ),
            "strategy",
        )
        slow = run_backtest(
            returns,
            SlowRatioVolScaledTrailingStop(
                base_levels=levels,
                base_reentry_recovery=300_000.0,
                numerator_window=21,
                denominator_window=63,
                monthly_days=5,
                vol_mult_floor=0.25,
                vol_mult_cap=4.0,
            ),
            "strategy",
        )

        self.assertTrue(np.allclose(fast.equity_curves, slow.equity_curves))
        self.assertTrue(np.allclose(fast.position_sizes, slow.position_sizes))

    def test_realised_vol_fast_path_matches_generic_loop(self):
        rng = np.random.default_rng(789)
        returns = rng.normal(0.0001, 0.02, size=(10, 160))
        levels = [
            (700_000.0, 0.70),
            (1_400_000.0, 0.40),
            (2_000_000.0, 0.0),
        ]

        fast = run_backtest(
            returns,
            VolScaledTrailingStop(
                base_levels=levels,
                base_reentry_recovery=300_000.0,
                reference_vol=0.15,
                vol_window_days=21,
                monthly_days=5,
            ),
            "strategy",
        )
        slow = run_backtest(
            returns,
            SlowVolScaledTrailingStop(
                base_levels=levels,
                base_reentry_recovery=300_000.0,
                reference_vol=0.15,
                vol_window_days=21,
                monthly_days=5,
            ),
            "strategy",
        )

        self.assertTrue(np.allclose(fast.equity_curves, slow.equity_curves))
        self.assertTrue(np.allclose(fast.position_sizes, slow.position_sizes))

    def test_combined_stopout_table_distinguishes_portfolio_and_sleeves(self):
        equity = np.full((2, 4), 10_000_000.0)
        sizes_a = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        sizes_b = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        results = {
            ("A", "rule"): BacktestResult("A", "Rule", equity, sizes_a, 10_000_000.0),
            ("B", "rule"): BacktestResult("B", "Rule", equity, sizes_b, 10_000_000.0),
        }
        results[("combined", "rule")] = combine_sleeves(
            results,
            ["A", "B"],
            "rule",
            {"A": 10_000_000.0, "B": 10_000_000.0},
        )

        portfolio = stopout_pct_table(
            results,
            ["combined"],
            ["rule"],
            component_strategies=["A", "B"],
            combined_mode="portfolio",
        )
        any_sleeve = stopout_pct_table(
            results,
            ["combined"],
            ["rule"],
            component_strategies=["A", "B"],
            combined_mode="any_sleeve",
        )
        capital_weighted = stopout_pct_table(
            results,
            ["combined"],
            ["rule"],
            component_strategies=["A", "B"],
            capitals={"A": 10_000_000.0, "B": 10_000_000.0},
            combined_mode="capital_weighted",
        )

        self.assertEqual(portfolio["stopout_pct"].iloc[0], 0.0)
        self.assertEqual(any_sleeve["stopout_pct"].iloc[0], 100.0)
        self.assertEqual(capital_weighted["stopout_pct"].iloc[0], 50.0)


if __name__ == "__main__":
    unittest.main()
