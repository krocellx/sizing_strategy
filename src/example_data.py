import numpy as np
import pandas as pd


def generate_random_strategy_returns(
    start_date="2005-01-01",
    end_date="2015-12-31",
    strategy_names=None,
    annual_return=None,
    annual_vol=None,
    seed=42,
):
    """
    Generate random daily returns for multiple strategies.

    Returns
    -------
    pd.DataFrame
        Index  : business-day dates
        Columns: strategy names
        Values : simulated daily returns
    """

    if strategy_names is None:
        strategy_names = [
            "US Ten Stocks",
            "Europe Ten Stocks",
            "Long-Short",
        ]

    if annual_return is None:
        annual_return = {
            "US Ten Stocks": 0.10,
            "Europe Ten Stocks": 0.08,
            "Long-Short": 0.06,
        }

    if annual_vol is None:
        annual_vol = {
            "US Ten Stocks": 0.18,
            "Europe Ten Stocks": 0.16,
            "Long-Short": 0.10,
        }

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    n_days = len(dates)

    data = {}

    for strat in strategy_names:

        mu_daily = annual_return[strat] / 252
        sigma_daily = annual_vol[strat] / np.sqrt(252)

        returns = np.random.normal(
            loc=mu_daily,
            scale=sigma_daily,
            size=n_days,
        )

        data[strat] = returns

    stra_returns = pd.DataFrame(
        data,
        index=dates,
    )

    return stra_returns