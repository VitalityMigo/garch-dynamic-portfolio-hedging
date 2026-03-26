from __future__ import annotations
import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame):
    """
    Compute simple daily returns from a price DataFrame.
    """
    returns = prices.pct_change()
    returns = returns.iloc[1:] 
    return returns

def build_portfolio(
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
):
    """
    Build a buy-and-hold, equally weighted portfolio.
    """
    n_stocks = prices.shape[1]

    # Initial price vector
    p0 = prices.iloc[0]

    # Equally weighted portfolio
    capital_per_stock = initial_capital / n_stocks
    shares = capital_per_stock / p0  

    # Invested capital vector
    position_values = prices.multiply(shares, axis="columns")
    position_values.index.name = "Date"

    # Portfolio NAV = sum of all position values
    portfolio_nav =position_values.sum(axis=1)
    portfolio_nav.name = "Portfolio_NAV"

    # Daily returns from NAV
    portfolio_returns= portfolio_nav.pct_change().iloc[1:]
    portfolio_returns.name = "Portfolio_Return"

    return portfolio_nav, position_values, portfolio_returns

def compute_weights(position_values: pd.DataFrame):
    """
    Compute the time-varying weight of each stock in the portfolio.
    """
    total_value = position_values.sum(axis=1)
    total_value = total_value.replace(0, np.nan)
    weights = position_values.div(total_value, axis="index")
    weights.index.name = "Date"
    return weights
