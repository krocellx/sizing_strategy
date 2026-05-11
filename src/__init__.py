from .simulation import (
    politis_white_L,
    stationary_bootstrap_indices,
    apply_indices,
    generate_scenarios,
)
from .stop_rules import StopRule, NoStop, TrailingStopRule, VolScaledTrailingStop
from .engine import BacktestResult, run_backtest, compare
from .cache import CachedResult, run_backtest_chunked
from .analysis import (
    percentile_table, cvar, cvar_table,
    plot_distribution_overlay, plot_qq,
    stop_trigger_frequency, time_under_water, recovery_times, drawdown_summary,
    conditional_comparison, combine_sleeves,
    risk_adjusted_metrics, risk_adjusted_table,
    paired_comparison, bootstrap_ci,
    full_report,
)
from .sensitivity import (
    sensitivity_to_L,
    sensitivity_to_rule_params,
    sensitivity_to_capital,
)
from .institutional import (
    rolling_return_stats,
    dd_threshold_probabilities,
    stop_activity,
    plot_equity_fan,
    plot_drawdown_fan,
    plot_pct_at_hwm,
    plot_return_vs_dd_scatter,
    plot_did_stop_help,
    institutional_summary,
    plot_calmar_bar,
    plot_dd_breach_heatmap,
    plot_rolling_return_violin,
    plot_stop_activity_bar,
    plot_conditional_diverging,
    plot_size_change_frequency,
    plot_historical_events,
)

__all__ = [
    # simulation
    "politis_white_L", "stationary_bootstrap_indices", "apply_indices",
    "generate_scenarios",
    # rules & engine
    "StopRule", "NoStop", "TrailingStopRule", "VolScaledTrailingStop",
    "BacktestResult", "run_backtest", "compare",
    "CachedResult", "run_backtest_chunked",
    # analysis
    "percentile_table", "cvar", "cvar_table",
    "plot_distribution_overlay", "plot_qq",
    "stop_trigger_frequency", "time_under_water", "recovery_times",
    "drawdown_summary", "conditional_comparison", "combine_sleeves",
    "risk_adjusted_metrics", "risk_adjusted_table",
    "paired_comparison", "bootstrap_ci", "full_report",
    # sensitivity
    "sensitivity_to_L", "sensitivity_to_rule_params", "sensitivity_to_capital",
    # institutional
    "rolling_return_stats", "dd_threshold_probabilities", "stop_activity",
    "plot_equity_fan", "plot_drawdown_fan", "plot_pct_at_hwm",
    "plot_return_vs_dd_scatter", "plot_did_stop_help", "institutional_summary",
    "plot_calmar_bar", "plot_dd_breach_heatmap", "plot_rolling_return_violin",
    "plot_stop_activity_bar", "plot_conditional_diverging", "plot_size_change_frequency",
    "plot_historical_events",
]