
from __future__ import annotations
import numpy as np
import pandas as pd

# Min. observation
MIN_OBS_FOR_GARCH: int = 100


def scale_returns_for_garch(
    returns: pd.Series,
    scale: float = 100.0,
):
    """
    Scale daily returns for numerical stability before GARCH fitting.
    """
    scaled = returns.dropna() * scale
    scaled.name = returns.name
    return scaled


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
):
    """
    Fit a GARCH(p,q) model to a return series.
    """
    from arch import arch_model

    clean = returns.dropna()
    if len(clean) < MIN_OBS_FOR_GARCH:
        raise ValueError(
            f"Not enough observations to fit GARCH: need ≥ {MIN_OBS_FOR_GARCH}, "
            f"got {len(clean)}."
        )

    am = arch_model(
        clean,
        mean="Constant",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
    )

    result = am.fit(disp="off", show_warning=False, options={"maxiter": 500})

    return result

def extract_garch_parameters(fitted_model):
    """
    Extract parameters from a GARCH model result.
    """
    params = fitted_model.params
    pvals  = fitted_model.pvalues

    def _get(keys: list[str]) -> float:
        for k in keys:
            if k in params.index:
                return float(params[k])
        return float("nan")

    def _getp(keys: list[str]) -> float:
        for k in keys:
            if k in pvals.index:
                return float(pvals[k])
        return float("nan")

    omega   = _get(["omega"])
    alpha_1 = _get(["alpha[1]", "alpha"])
    beta_1  = _get(["beta[1]",  "beta"])

    return {
        "omega":          omega,
        "alpha_1":        alpha_1,
        "beta_1":         beta_1,
        "persistence":    alpha_1 + beta_1,
        "loglikelihood":  float(fitted_model.loglikelihood),
        "aic":            float(fitted_model.aic),
        "bic":            float(fitted_model.bic),
        "omega_pval":     _getp(["omega"]),
        "alpha_1_pval":   _getp(["alpha[1]", "alpha"]),
        "beta_1_pval":    _getp(["beta[1]", "beta"]),
    }

def compute_conditional_volatility(
    fitted_model,
    scale: float = 100.0,
    annualize: bool = True,
    trading_days: int = 252,
):
    """
    Extract the in-sample conditional volatility from a GARCH result.
    """
    cond_vol = fitted_model.conditional_volatility.copy() 

    cond_vol = cond_vol / scale

    if annualize:
        cond_vol = cond_vol * np.sqrt(trading_days)

    cond_vol.name = "conditional_vol"
    cond_vol.index = pd.to_datetime(cond_vol.index)
    return cond_vol.sort_index()

def forecast_next_day_volatility(
    fitted_model,
    scale: float = 100.0,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """
    Produce the 1-step-ahead conditional volatility forecast.
    """
    forecast = fitted_model.forecast(horizon=1, method="analytic", reindex=False)

    scaled_variance = float(forecast.variance.iloc[-1, 0])
    scaled_vol      = np.sqrt(max(scaled_variance, 0.0))

    # Convert back to original units
    daily_vol = scaled_vol / scale

    if annualize:
        return daily_vol * np.sqrt(trading_days)

    return daily_vol

def build_garch_summary(
    returns: pd.Series,
    scale: float = 100.0,
    annualize: bool = True,
    trading_days: int = 252,
):
    """
    Fit GARCH(1,1) and return all key outputs in a single dictionary.
    """
    scaled  = scale_returns_for_garch(returns, scale=scale)
    model   = fit_garch_model(scaled)
    params  = extract_garch_parameters(model)
    cond_vol = compute_conditional_volatility(
        model, scale=scale, annualize=annualize, trading_days=trading_days
    )
    next_day = forecast_next_day_volatility(
        model, scale=scale, annualize=annualize, trading_days=trading_days
    )

    return {
        "model":                 model,
        "params":                params,
        "conditional_vol":       cond_vol,
        "next_day_vol_forecast": next_day,
    }
