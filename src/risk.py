
from __future__ import annotations
import numpy as np
import pandas as pd



def _align(
    a: pd.Series,
    b: pd.Series,
):
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def _align_frame_series(
    df: pd.DataFrame,
    s: pd.Series,
):
    joined = df.join(s, how="inner").dropna()
    return joined.iloc[:, :-1], joined.iloc[:, -1]


def compute_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
):
    """Compute the rolling beta of the portfolio relative to a benchmark"""
    port, bench = _align(portfolio_returns, benchmark_returns)

    rolling_cov = port.rolling(window=window, min_periods=window).cov(bench)
    rolling_var = bench.rolling(window=window, min_periods=window).var()

    # Guard against zero variance (flat benchmark periods)
    rolling_var = rolling_var.replace(0.0, np.nan)

    beta = rolling_cov / rolling_var
    beta.name = "rolling_beta"
    return beta.sort_index()



# Correl
def compute_rolling_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
):
    """Compute the rolling correlation"""
    port, bench = _align(portfolio_returns, benchmark_returns)

    corr = port.rolling(window=window, min_periods=window).corr(bench)
    corr.name = "rolling_correlation"
    return corr.sort_index()


# Realized vol
def compute_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) :
    """Compute rolling realized vol"""
    clean = returns.dropna().sort_index()

    rv = clean.rolling(window=window, min_periods=window).std()

    if annualize:
        rv = rv * np.sqrt(trading_days)

    rv.name = "realized_volatility"
    return rv



# Rolling OLS
def compute_rolling_regression_stats(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
):
    """Compute rolling OLS regression """
    port, bench = _align(portfolio_returns, benchmark_returns)
    n = len(port)

    alphas     = np.full(n, np.nan)
    betas      = np.full(n, np.nan)
    r_squareds = np.full(n, np.nan)

    port_arr  = port.to_numpy()
    bench_arr = bench.to_numpy()

    for i in range(window - 1, n):
        y = port_arr[i - window + 1 : i + 1]  
        x = bench_arr[i - window + 1 : i + 1]

        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < window:
            continue

        y_w = y[mask]
        x_w = x[mask]

        # OLS via equations with intercept
        x_mat = np.column_stack([np.ones_like(x_w), x_w])  
        xtx = x_mat.T @ x_mat
        xty = x_mat.T @ y_w

        try:
            coeffs = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            continue

        a, b = coeffs[0], coeffs[1]

        y_hat     = x_mat @ coeffs
        ss_res    = np.sum((y_w - y_hat) ** 2)
        ss_tot    = np.sum((y_w - y_w.mean()) ** 2)
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        alphas[i]     = a
        betas[i]      = b
        r_squareds[i] = r2

    result = pd.DataFrame(
        {"alpha": alphas, "beta": betas, "r_squared": r_squareds},
        index=port.index,
    )
    return result.sort_index()



# Rolling beta decomposition
def compute_component_betas(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 126,
):
    """Compute rolling beta of constituants"""
    df, bench = _align_frame_series(stock_returns, benchmark_returns)

    rolling_var = bench.rolling(window=window, min_periods=window).var()
    rolling_var = rolling_var.replace(0.0, np.nan)

    component_betas: dict[str, pd.Series] = {}

    for ticker in df.columns:
        cov = df[ticker].rolling(window=window, min_periods=window).cov(bench)
        component_betas[ticker] = cov / rolling_var

    result = pd.DataFrame(component_betas, index=df.index)
    return result.sort_index()
