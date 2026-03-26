from __future__ import annotations
import numpy as np
import pandas as pd

from src.garch_model import fit_garch_model, scale_returns_for_garch
from src.hedge import (
    compute_dynamic_hedge_ratio,
    compute_spy_hedge_position,
    compute_trade_series,
)

# Minimum observations for GARCH
_MIN_HISTORY: int = 100


def build_historical_vol_forecast_series(
    returns: pd.Series,
    lookback: int = 252,
    annualize: bool = True,
    trading_days: int = 252,
    refit_every: int = 21,
    scale: float = 100.0,
):
    clean = returns.dropna().sort_index()
    n = len(clean)

    forecast_arr = np.full(n, np.nan)

    refit_indices = range(lookback, n, refit_every)

    for t_k in refit_indices:

        train_raw = clean.iloc[max(0, t_k - lookback) : t_k + 1]

        # How many forward dates to cover until the next refit?
        next_refit = t_k + refit_every
        horizon = min(refit_every, n - t_k)  # number of forward dates to cover

        if len(train_raw) < _MIN_HISTORY or horizon <= 0:
            continue

        try:
            scaled_train = scale_returns_for_garch(train_raw, scale=scale)

            model = fit_garch_model(scaled_train)

            fc = model.forecast(horizon=horizon, method="analytic", reindex=False)
            scaled_variances = fc.variance.values[-1]

            for h_idx in range(horizon):
                scaled_var = float(scaled_variances[h_idx])
                daily_vol = np.sqrt(max(scaled_var, 0.0)) / scale

                if annualize:
                    daily_vol *= np.sqrt(trading_days)

                target_idx = t_k + h_idx
                if target_idx < n:
                    forecast_arr[target_idx] = daily_vol

        except Exception as exc:
            continue

    series = pd.Series(
        forecast_arr,
        index=clean.index,
        name="garch_next_day_forecast",
    )
    return series.sort_index()


def run_hedge_backtest(
    portfolio_nav: pd.Series,
    portfolio_returns: pd.Series,
    spy_prices: pd.Series,
    spy_returns: pd.Series,
    rolling_beta: pd.Series,
    rolling_correlation: pd.Series,
    target_beta: float = 0.0,
    garch_lookback: int = 252,
    refit_every: int = 7,
    trading_days: int = 252,
    max_abs_position: float | None = None,
):
    """Simulate the hedge historically in walk-forward"""

    port_vol_forecast = build_historical_vol_forecast_series(
        portfolio_returns,
        lookback=garch_lookback,
        annualize=False,
        trading_days=trading_days,
        refit_every=refit_every,
    )

    spy_vol_forecast = build_historical_vol_forecast_series(
        spy_returns,
        lookback=garch_lookback,
        annualize=False,
        trading_days=trading_days,
        refit_every=refit_every,
    )

    # Compute hedge ratio and pos
    hedge_ratio = compute_dynamic_hedge_ratio(
        correlation=rolling_correlation,
        portfolio_vol_forecast=port_vol_forecast,
        hedge_vol_forecast=spy_vol_forecast,
    )

    recommended_position = compute_spy_hedge_position(
        portfolio_value=portfolio_nav,
        spy_price=spy_prices,
        portfolio_beta=rolling_beta,
        target_beta=target_beta,
        hedge_ratio=hedge_ratio,
        max_abs_position=max_abs_position,
    )

    spy_trade = compute_trade_series(recommended_position)

    combined = pd.concat(
        {
            "portfolio_nav": portfolio_nav,
            "portfolio_returns": portfolio_returns,
            "spy_price": spy_prices,
            "spy_returns": spy_returns,
            "rolling_beta": rolling_beta,
            "rolling_correlation": rolling_correlation,
            "portfolio_vol_forecast": port_vol_forecast,
            "spy_vol_forecast": spy_vol_forecast,
            "hedge_ratio": hedge_ratio,
            "recommended_spy_position": recommended_position,
            "spy_trade": spy_trade,
        },
        axis=1,
        join="outer",
    ).sort_index()

    combined["applied_spy_position"] = combined["recommended_spy_position"].shift(1)

    # PNL
    spy_price_diff = combined["spy_price"].diff()

    hedge_pnl = combined["applied_spy_position"] * spy_price_diff
    hedge_pnl = hedge_pnl.fillna(0.0)  # first row / NaN rows → 0 (no position)
    combined["hedge_pnl"] = hedge_pnl

    combined["portfolio_pnl"] = combined["portfolio_nav"].diff().fillna(0.0)
    combined["hedged_pnl"] = combined["portfolio_pnl"] + combined["hedge_pnl"]
    combined["hedge_overlay_cum_pnl"] = combined["hedge_pnl"].cumsum()
    combined["hedged_nav"] = (
        combined["portfolio_nav"] + combined["hedge_overlay_cum_pnl"]
    )

    col_order = [
        "portfolio_nav",
        "portfolio_returns",
        "spy_price",
        "spy_returns",
        "rolling_beta",
        "rolling_correlation",
        "portfolio_vol_forecast",
        "spy_vol_forecast",
        "hedge_ratio",
        "recommended_spy_position",
        "applied_spy_position",
        "spy_trade",
        "hedge_pnl",
        "portfolio_pnl",
        "hedged_pnl",
        "hedge_overlay_cum_pnl",
        "hedged_nav",
    ]
    existing_cols = [c for c in col_order if c in combined.columns]
    return combined[existing_cols]


def compute_hedge_metrics(
    backtest_df: pd.DataFrame,
    trading_days: int = 252,
) -> dict[str, float]:
    """Measure hedge perf"""
    df = backtest_df.dropna(subset=["portfolio_returns", "hedged_pnl"])

    port_ret = df["portfolio_returns"].dropna()
    nav = df["portfolio_nav"].dropna()

    # Hedged returns
    nav_lagged = df["portfolio_nav"].shift(1)
    hedged_ret = (df["hedged_pnl"] / nav_lagged).dropna()

    # Annualised returns
    unhedged_ann_ret = (1 + port_ret).prod() ** (trading_days / len(port_ret)) - 1
    hedged_ann_ret = (1 + hedged_ret).prod() ** (trading_days / len(hedged_ret)) - 1

    # Vol
    unhedged_ann_vol = port_ret.std() * np.sqrt(trading_days)
    hedged_ann_vol = hedged_ret.std() * np.sqrt(trading_days)

    var_unhedged = port_ret.var()
    var_hedged = hedged_ret.var()
    hedge_effectiveness = (
        1.0 - var_hedged / var_unhedged if var_unhedged > 0 else float("nan")
    )

    cumulative_hedge_pnl = (
        df["hedge_overlay_cum_pnl"].dropna().iloc[-1]
        if "hedge_overlay_cum_pnl" in df.columns
        else float("nan")
    )
    avg_abs_position = (
        df["applied_spy_position"].abs().mean()
        if "applied_spy_position" in df.columns
        else float("nan")
    )
    turnover = (
        df["spy_trade"].abs().sum() if "spy_trade" in df.columns else float("nan")
    )

    # Realized beta (before hedge)
    spy_ret = df["spy_returns"].dropna()
    aligned = pd.concat([port_ret, spy_ret], axis=1, join="inner").dropna()
    var_spy = aligned.iloc[:, 1].var()
    realized_beta_before = (
        aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / var_spy
        if var_spy > 0
        else float("nan")
    )

    # Realized beta (after hedge)
    h_aligned = pd.concat([hedged_ret, spy_ret], axis=1, join="inner").dropna()
    realized_beta_after = (
        h_aligned.iloc[:, 0].cov(h_aligned.iloc[:, 1]) / h_aligned.iloc[:, 1].var()
        if h_aligned.iloc[:, 1].var() > 0
        else float("nan")
    )

    if "hedged_nav" in df.columns:
        h_nav = df["hedged_nav"].dropna()
        peak = h_nav.cummax()
        drawdown = (h_nav - peak) / peak
        max_drawdown_hedged = float(drawdown.min())
    else:
        max_drawdown_hedged = float("nan")

    p_nav = nav
    p_peak = p_nav.cummax()
    p_drawdown = (p_nav - p_peak) / p_peak
    max_drawdown_unhedged = float(p_drawdown.min())

    return {
        "unhedged_ann_return": float(unhedged_ann_ret),
        "hedged_ann_return": float(hedged_ann_ret),
        "unhedged_ann_vol": float(unhedged_ann_vol),
        "hedged_ann_vol": float(hedged_ann_vol),
        "hedge_effectiveness": float(hedge_effectiveness),
        "cumulative_hedge_pnl": float(cumulative_hedge_pnl),
        "avg_abs_spy_position": float(avg_abs_position),
        "turnover": float(turnover),
        "realized_beta_before": float(realized_beta_before),
        "realized_beta_after": float(realized_beta_after),
        "max_drawdown_unhedged": max_drawdown_unhedged,
        "max_drawdown_hedged": max_drawdown_hedged,
    }
