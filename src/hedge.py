from __future__ import annotations
import numpy as np
import pandas as pd


def compute_dynamic_hedge_ratio(
    correlation: pd.Series,
    portfolio_vol_forecast: pd.Series,
    hedge_vol_forecast: pd.Series,
    min_abs_vol: float = 1e-8,
    max_abs_ratio: float | None = None,
):
    """
    Compute the dynamic minimum-variance hedge ratio.
    """
    df = pd.concat(
        [correlation, portfolio_vol_forecast, hedge_vol_forecast],
        axis=1,
        join="inner",
    ).dropna()

    corr   = df.iloc[:, 0]
    p_vol  = df.iloc[:, 1]
    h_vol  = df.iloc[:, 2]

    # Guard denominator
    h_vol_safe = h_vol.where(h_vol.abs() >= min_abs_vol, np.nan)

    ratio = corr * p_vol / h_vol_safe

    if max_abs_ratio is not None:
        ratio = ratio.clip(-max_abs_ratio, max_abs_ratio)

    ratio.name = "hedge_ratio"
    return ratio.sort_index()


def compute_spy_hedge_position(
    portfolio_value: pd.Series,
    spy_price: pd.Series,
    portfolio_beta: pd.Series,
    target_beta: float,
    hedge_ratio: pd.Series,
    max_abs_position: float | None = None,
):
    """
    Compute the recommended SPY share position to reach target beta.
    """
    df = pd.concat(
        [portfolio_value, spy_price, portfolio_beta, hedge_ratio],
        axis=1,
        join="inner",
    ).dropna()

    pv = df.iloc[:, 0]   
    sp = df.iloc[:, 1] 
    pb = df.iloc[:, 2]
    hr = df.iloc[:, 3] 

    denom = hr * sp
    denom = denom.where(denom.abs() >= 1e-6, np.nan)

    position = -(pb - target_beta) * pv / denom

    if max_abs_position is not None:
        position = position.clip(-max_abs_position, max_abs_position)

    position.name = "spy_position"
    return position.sort_index()


def compute_trade_series(position_series: pd.Series):
    """
    Compute tomorrow trade.
    """
    trade = position_series.diff()
    trade.name = "spy_trade"
    return trade
