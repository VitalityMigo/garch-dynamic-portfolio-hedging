"""
app.py — Dynamic Equity Hedging Dashboard
Run with:  streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import streamlit.components.v1 as components

from src import config
from src.backtest import compute_hedge_metrics, run_hedge_backtest
from src.data_loader import prepare_price_panel
from src.garch_model import build_garch_summary
from src.plots import (
    bar_chart,
    garch_vol_surface,
    hedge_pnl_chart,
    line_chart,
    multi_line_chart,
    nav_comparison_chart,
    regression_scatter,
    rolling_stats_chart,
    vol_comparison_chart,
)
from src.portfolio import (
    build_portfolio,
    compute_weights,
    compute_returns,
)
from src.risk import (
    compute_component_betas,
    compute_realized_volatility,
    compute_rolling_beta,
    compute_rolling_correlation,
    compute_rolling_regression_stats,
)

# Page config

st.set_page_config(
    page_title="Dynamic Hedging",
    page_icon=":streamlit:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

/* ── Global terminal font & base ──────────────────────────── */
html, body, [class*="css"], input, button, select, textarea,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"],
[data-testid="stCaption"],
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stSelectbox, .stMultiSelect,
div[data-baseweb="select"], div[data-baseweb="input"] {
    font-family: 'IBM Plex Mono', 'Roboto Mono', 'Courier New', monospace !important;
}

/* ── Markdown section headers — terminal green ─────────────── */
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #00FF88 !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(0,255,136,0.15) !important;
    padding-bottom: 0.3rem !important;
    margin-bottom: 0.6rem !important;
}

/* ── Dataframe / table terminal look ───────────────────────── */
[data-testid="stDataFrame"] *, 
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stDataFrame"] th {
    color: #F7931E !important;
    background: rgba(247,147,30,0.06) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Metrics ───────────────────────────────────────────────── */
[data-testid="stMetricLabel"] {
    font-size: 0.60rem !important;
    color: #4a5568 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    line-height: 1.2 !important;
}
[data-testid="stMetricValue"] {
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    color: #F7931E;
    font-family: 'IBM Plex Mono', monospace !important;
    line-height: 1.3 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.62rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
    padding: 0.5rem 0.6rem !important;
}

/* ── Caption / small text ──────────────────────────────────── */
[data-testid="stCaption"] p {
    font-size: 0.62rem !important;
    color: #4a5568 !important;
    letter-spacing: 0.06em;
}

/* ── Labels (sliders, inputs) ──────────────────────────────── */
[data-testid="stWidgetLabel"] p {
    font-size: 0.66rem !important;
    color: #8898AA !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
}

/* ── App background — slightly lighter dark ───────────────── */
[data-testid="stApp"] {
    background-color: #111419 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(22,27,39,0.96) !important;
    backdrop-filter: blur(4px) !important;
}

/* ── Layout ────────────────────────────────────────────────── */
/* push content below the Streamlit toolbar so tabs aren't hidden */
.block-container {
    padding-top: 3.8rem !important;
    padding-bottom: 1rem;
    max-width: 100% !important;
}

/* ── Tab bar — terminal style ──────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.10);
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    padding: 0.68rem 2.8rem !important;
    color: #4a5568 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #F7931E !important;
    border-bottom: 2px solid #F7931E !important;
    background: rgba(247,147,30,0.04) !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #F7931E !important;
    background: rgba(247,147,30,0.07) !important;
}

/* ── Sidebar heading ───────────────────────────────────────── */
[data-testid="stSidebar"] h1 {
    font-size: 0.76rem !important;
    letter-spacing: 0.20em !important;
    color: #F7931E !important;
    text-transform: uppercase !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(247,147,30,0.25) !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important;
    letter-spacing: 0.14em !important;
    color: #8898AA !important;
    text-transform: uppercase !important;
    margin-top: 1.1rem !important;
}

/* ── Number input +/- buttons — remove red hover ──────────── */
[data-testid="stNumberInput"] button:hover,
[data-testid="stNumberInput"] button:focus,
[data-testid="stNumberInput"] button:active {
    background-color: rgba(247,147,30,0.12) !important;
    border-color: rgba(247,147,30,0.35) !important;
    color: #F7931E !important;
}
[data-testid="stNumberInput"] button {
    color: #8898AA !important;
    border-color: rgba(255,255,255,0.08) !important;
    background-color: transparent !important;
    transition: background-color 0.15s, border-color 0.15s !important;
}

/* ── Tab animated underline — disabled ────────────────────── */
[data-baseweb="tab-highlight"] {
    display: none !important;
    transition: none !important;
}
</style>
""", unsafe_allow_html=True)

# Cached helpers

@st.cache_data(show_spinner=False)
def _load_prices(tickers: tuple[str, ...], start_date: str) -> pd.DataFrame:
    return prepare_price_panel(
        portfolio_tickers=list(tickers),
        hedge_ticker=config.HEDGE_TICKER,
        start_date=start_date,
        end_date=config.DEFAULT_END_DATE,
    )


@st.cache_data(show_spinner=False)
def _run_pipeline(
    tickers: tuple[str, ...],
    start_date: str,
    initial_capital: float,
    target_beta: float,
    rolling_window: int,
    realized_vol_window: int,
    garch_lookback: int,
    max_abs_position: float,
) -> dict:
    price_panel = _load_prices(tickers, start_date)
    stock_prices = price_panel[list(tickers)]
    spy_prices = price_panel[config.HEDGE_TICKER]

    all_returns = compute_returns(price_panel)
    stock_returns = all_returns[list(tickers)]
    spy_returns = all_returns[config.HEDGE_TICKER]

    portfolio_nav, position_values, portfolio_returns = build_portfolio(
        prices=stock_prices,
        initial_capital=initial_capital,
    )
    current_weights = compute_weights(position_values)

    rolling_beta = compute_rolling_beta(
        portfolio_returns, spy_returns, window=rolling_window
    )
    rolling_corr = compute_rolling_correlation(
        portfolio_returns, spy_returns, window=rolling_window
    )
    port_realized_vol = compute_realized_volatility(
        portfolio_returns,
        window=realized_vol_window,
        annualize=True,
        trading_days=config.TRADING_DAYS,
    )
    spy_realized_vol = compute_realized_volatility(
        spy_returns,
        window=realized_vol_window,
        annualize=True,
        trading_days=config.TRADING_DAYS,
    )
    regression_stats = compute_rolling_regression_stats(
        portfolio_returns, spy_returns, window=rolling_window
    )
    component_betas = compute_component_betas(
        stock_returns, spy_returns, window=rolling_window
    )

    port_garch = build_garch_summary(
        portfolio_returns, annualize=True, trading_days=config.TRADING_DAYS
    )
    spy_garch = build_garch_summary(
        spy_returns, annualize=True, trading_days=config.TRADING_DAYS
    )

    backtest_df = run_hedge_backtest(
        portfolio_nav=portfolio_nav,
        portfolio_returns=portfolio_returns,
        spy_prices=spy_prices,
        spy_returns=spy_returns,
        rolling_beta=rolling_beta,
        rolling_correlation=rolling_corr,
        target_beta=target_beta,
        garch_lookback=garch_lookback,
        trading_days=config.TRADING_DAYS,
        max_abs_position=max_abs_position,
    )
    metrics = compute_hedge_metrics(backtest_df, trading_days=config.TRADING_DAYS)

    return dict(
        price_panel=price_panel,
        stock_prices=stock_prices,
        spy_prices=spy_prices,
        stock_returns=stock_returns,
        spy_returns=spy_returns,
        portfolio_nav=portfolio_nav,
        portfolio_returns=portfolio_returns,
        position_values=position_values,
        current_weights=current_weights,
        rolling_beta=rolling_beta,
        rolling_corr=rolling_corr,
        port_realized_vol=port_realized_vol,
        spy_realized_vol=spy_realized_vol,
        regression_stats=regression_stats,
        component_betas=component_betas,
        port_garch=port_garch,
        spy_garch=spy_garch,
        backtest_df=backtest_df,
        metrics=metrics,
    )


# Sidebar 
# Fixed parameters (no sidebar)
selected_tickers   = list(config.DEFAULT_TICKERS)
start_date         = config.DEFAULT_START_DATE
initial_capital    = config.DEFAULT_INITIAL_CAPITAL
target_beta        = config.DEFAULT_TARGET_BETA
max_abs_position   = config.MAX_SPY_POSITION
rolling_window     = config.ROLLING_WINDOW
realized_vol_window = config.REALIZED_VOL_WINDOW
garch_lookback     = config.GARCH_LOOKBACK

# Run pipeline 

_loading_slot = st.empty()
_loading_slot.markdown("""
<style>
@keyframes _spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes _pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
._loader-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 82vh;
    gap: 1.6rem;
}
._spinner-ring {
    width: 52px; height: 52px;
    border-radius: 50%;
    border: 2px solid rgba(247,147,30,0.12);
    border-top-color: #F7931E;
    animation: _spin 0.9s linear infinite;
}
._loader-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #F7931E;
    animation: _pulse 2s ease-in-out infinite;
}
._loader-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    margin-top: -1rem;
}
</style>
<div class="_loader-wrap">
  <div class="_spinner-ring"></div>
  <div class="_loader-title">COMPUTING</div>
  <div class="_loader-sub">fetching data &bull; fitting GARCH &bull; running backtest</div>
</div>
""", unsafe_allow_html=True)

d = _run_pipeline(
    tickers=tuple(selected_tickers),
    start_date=start_date,
    initial_capital=float(initial_capital),
    target_beta=target_beta,
    rolling_window=rolling_window,
    realized_vol_window=realized_vol_window,
    garch_lookback=garch_lookback,
    max_abs_position=float(max_abs_position),
)
_loading_slot.empty()

# Convenience aliases 

nav       = d["portfolio_nav"]
port_ret  = d["portfolio_returns"]
spy_ret   = d["spy_returns"]
spy_px    = d["spy_prices"]
rb        = d["rolling_beta"]
rc        = d["rolling_corr"]
port_vol  = d["port_realized_vol"]
spy_vol   = d["spy_realized_vol"]
reg       = d["regression_stats"]
comp_beta = d["component_betas"]
pg        = d["port_garch"]
sg        = d["spy_garch"]
bt        = d["backtest_df"]
metrics   = d["metrics"]


def _last(s: "pd.Series | pd.DataFrame") -> float:
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else float("nan")


#  Tabs 

tab_overview, tab_history, tab_volmodel, tab_portfolio = st.tabs([
    "OVERVIEW",
    "HISTORY",
    "VOLATILITY MODEL",
    "PORTFOLIO",
])

# TAB 1
with tab_overview:

    _hedged_ret       = (bt["hedged_pnl"] / bt["portfolio_nav"].shift(1)).dropna()
    _hedged_beta      = compute_rolling_beta(_hedged_ret, spy_ret, window=126).dropna()
    _hedged_beta_last = float(_hedged_beta.iloc[-1]) if len(_hedged_beta) else 0.0
    _hedged_ann_vol   = float(_hedged_ret.rolling(21).std().iloc[-1]) * np.sqrt(config.TRADING_DAYS) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Portfolio NAV", f"${nav.iloc[-1]:,.0f}")

    _applied_pos = bt["applied_spy_position"].dropna()
    _fut_exp = float(_applied_pos.iloc[-1]) * float(spy_px.iloc[-1]) if len(_applied_pos) else 0.0
    c2.metric("Future Exposure", f"${_fut_exp:+,.0f}", delta_color="off")

    c3.metric("Hedged Volatility", f"{_hedged_ann_vol:.1f}%", delta_color="off")

    c4.metric("Hedged Beta (6M)", f"{_hedged_beta_last:+.3f}", delta_color="off")

    corr_val = _last(rc)
    c5.metric("Correlation (6M)", f"{corr_val:.3f}", delta_color="off")

    latest_pos_series = bt["recommended_spy_position"].dropna()
    pos_val = float(latest_pos_series.iloc[-1]) if len(latest_pos_series) else 0.0
    c6.metric("Latest Hedge", f"{pos_val:+.0f} sh", delta_color="off")

    st.markdown('<div style="margin-top:-0.5rem"></div>', unsafe_allow_html=True)

    #  NAV 
    col_a, col_b = st.columns([5, 3])

    with col_a:
        hedged_nav_series = bt["hedged_nav"].dropna()
        st.plotly_chart(
            nav_comparison_chart(nav, hedged_nav_series, "Unhedged vs Hedged NAV"),
            width='stretch',
        )

    with col_b:
        st.markdown("#### Live Hedge Sizing")

        # Interactive target beta 
        live_target_beta = st.number_input(
            "Target beta (β*)",
            min_value=0.0, max_value=1.5,
            value=float(target_beta),
            step=0.05,
            format="%.2f",
            key="ov_target_beta",
        )

        port_vol_snap = float(pg["next_day_vol_forecast"])
        spy_vol_snap  = float(sg["next_day_vol_forecast"])
        corr_snap     = _last(rc)
        beta_snap     = _last(rb)
        nav_snap      = float(nav.iloc[-1])
        spy_px_snap   = float(spy_px.iloc[-1])

        h_raw_snap = corr_snap * port_vol_snap / spy_vol_snap if spy_vol_snap > 1e-8 else 1.0
        h_snap     = float(np.clip(h_raw_snap, -config.MAX_HEDGE_RATIO, config.MAX_HEDGE_RATIO))
        if abs(h_snap) < 1e-6:
            h_snap = 1e-6

        n_spy_snap  = -(_hedged_beta_last - live_target_beta) * nav_snap / (h_snap * spy_px_snap)
        n_spy_snap  = float(np.clip(n_spy_snap, -max_abs_position, max_abs_position))
        direction   = "SHORT" if n_spy_snap < 0 else ("LONG" if n_spy_snap > 0 else "FLAT")
        notional    = abs(n_spy_snap) * spy_px_snap
        delta_beta  = _hedged_beta_last - live_target_beta
        sig_color   = "#F5365C" if n_spy_snap < 0 else "#2DCE89"
        db_color    = "#F5365C" if delta_beta > 0.01 else ("#2DCE89" if delta_beta < -0.01 else "#8898AA")

        gc1, gc2 = st.columns(2)
        gc1.metric("PORT VOL. t+1",   f"{port_vol_snap*100:.2f}%")
        gc2.metric("BETA DIFF.",    f"{delta_beta:+.3f}")
        gc3, gc4 = st.columns(2)
        gc3.metric("SPY VOL t+1",    f"{spy_vol_snap*100:.2f}%")
        gc4.metric("HEDGE RATIO",   f"{h_snap:+.3f}")

        # Signal block 
        st.markdown(
            f"""<div style="margin-top:0.5rem;padding:0.7rem 0.75rem;
                 border:1px solid rgba(247,147,30,0.18);border-radius:2px;
                 background:rgba(0,0,0,0.15);font-family:'IBM Plex Mono',monospace;">
              <div style="font-size:0.50rem;color:#4a5568;letter-spacing:0.15em;
                   text-transform:uppercase;margin-bottom:0.3rem">SIGNAL</div>
              <div style="font-size:1.3rem;font-weight:500;color:{sig_color}">
                   {direction}&nbsp;{abs(n_spy_snap):.0f}&nbsp;shares</div>
              <div style="font-size:0.58rem;color:#4a5568;margin-top:0.1rem">
                   &approx;&nbsp;${notional:,.0f}&nbsp;notional</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Hedged profit & hedged rolling beta
    col_c, col_d = st.columns(2)

    with col_c:
        _hn = bt["hedged_nav"].dropna()
        hedged_total_pnl = (_hn - _hn.iloc[0]).rename("hedged_total_pnl")
        st.plotly_chart(
            hedge_pnl_chart(hedged_total_pnl, "Cumulative Hedged Portfolio PnL"),
            width='stretch',
        )

    with col_d:
        st.plotly_chart(
            rolling_stats_chart(
                _hedged_beta,
                title=f"Hedged Portfolio Rolling Beta (6M)",
                h_line=target_beta,
                h_label=f"Target β = {target_beta:.2f}",
            ),
            width='stretch',
        )

# History
with tab_history:

    st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)

    col_g, col_h = st.columns(2)

    with col_g:
        cum_pnl = bt["hedge_overlay_cum_pnl"].dropna()
        st.plotly_chart(
            hedge_pnl_chart(cum_pnl, "Cumulative Hedge: Unhedged vs Hedged"),
            width='stretch',
        )

    with col_h:

        _port_rv = port_vol.dropna()

        _hedged_pnl = bt["hedged_pnl"].dropna()
        _hedged_nav_lag = bt["portfolio_nav"].shift(1).reindex(_hedged_pnl.index)
        _hedged_ret_series = (_hedged_pnl / _hedged_nav_lag).dropna()
        from src.risk import compute_realized_volatility as _crv
        _hedged_rv = _crv(
            _hedged_ret_series,
            window=realized_vol_window,
            annualize=True,
            trading_days=config.TRADING_DAYS,
        ).dropna()

        _shared = _port_rv.index.intersection(_hedged_rv.index)
        _unh = _port_rv.loc[_shared] * 100
        _hed = _hedged_rv.loc[_shared] * 100

        _vfig = go.Figure()

        _vfig.add_trace(go.Scatter(
            x=list(_unh.index) + list(_unh.index[::-1]),
            y=list(_unh.values) + list(_hed.values[::-1]),
            fill="toself",
            fillcolor="rgba(255,255,255,0.04)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Unhedged
        _vfig.add_trace(go.Scatter(
            x=_unh.index, y=_unh.values,
            mode="lines", name="Unhedged",
            line=dict(color="#F7931E", width=1.8),
        ))
        # Hedged
        _vfig.add_trace(go.Scatter(
            x=_hed.index, y=_hed.values,
            mode="lines", name="Hedged",
            line=dict(color="#4C9EEB", width=1.8),
        ))
        from src.plots import _apply_base as _ab
        _vfig.update_yaxes(title_text="Ann. Vol (%)")
        st.plotly_chart(_ab(_vfig, "Realized Vol: Unhedged vs Hedged"), width='stretch')

    st.markdown("---")

    col_i, col_j = st.columns(2)

    with col_i:
        trade_series = bt["spy_trade"].dropna()
        trade_series.name = "Trade (shares)"
        st.plotly_chart(
            bar_chart(trade_series, title="Daily SPY Trade (Shares)", color_by_sign=True),
            width='stretch',
        )

    with col_j:
        hr_series = bt["hedge_ratio"].dropna()
        hr_series.name = "h_t"
        st.plotly_chart(
            rolling_stats_chart(
                hr_series,
                title="Dynamic Hedge Ratio",
                h_line=1.0,
                h_label="h = 1",
                color="#F7931E",
            ),
            width='stretch',
        )

# VOL MODEL
with tab_volmodel:

    # ── GARCH parameters 
    _GPTH  = ("padding:0.30rem 0.55rem;text-align:right;color:#F7931E;"
              "font-size:0.64rem;letter-spacing:0.12em;text-transform:uppercase;"
              "border-bottom:1px solid rgba(247,147,30,0.30);white-space:nowrap;")
    _GPTHL = _GPTH.replace("text-align:right", "text-align:left")
    _GPTD  = ("padding:0.22rem 0.55rem;text-align:right;font-size:0.74rem;"
              "color:rgba(154,165,180,1);border-bottom:1px solid rgba(255,255,255,0.04);")
    _GPTDL = _GPTD.replace("text-align:right", "text-align:left") + "color:rgba(247,147,30,0.85);font-weight:500;"

    _gpp, _gsp = pg["params"], sg["params"]

    def _pval_cell(v: float) -> str:
        """Format p-value with color: green <0.01, orange <0.05, red ≥0.05."""
        if np.isnan(v):
            return '<span style="color:#2d3748">–</span>'
        c = "#2DCE89" if v < 0.01 else ("#F7931E" if v < 0.05 else "#F5365C")
        return f'<span style="color:{c}">{v:.4f}</span>'

    _garch_param_rows = [
        ("ω (OMEGA)",   f"{_gpp['omega']:.6f}",       _pval_cell(_gpp['omega_pval']),
                        f"{_gsp['omega']:.6f}",        _pval_cell(_gsp['omega_pval'])),
        ("α (ARCH)",    f"{_gpp['alpha_1']:.4f}",     _pval_cell(_gpp['alpha_1_pval']),
                        f"{_gsp['alpha_1']:.4f}",      _pval_cell(_gsp['alpha_1_pval'])),
        ("β (GARCH)",   f"{_gpp['beta_1']:.4f}",      _pval_cell(_gpp['beta_1_pval']),
                        f"{_gsp['beta_1']:.4f}",       _pval_cell(_gsp['beta_1_pval'])),
        ("PERSISTENCE", f"{_gpp['persistence']:.4f}", '<span style="color:#2d3748">–</span>',
                        f"{_gsp['persistence']:.4f}",  '<span style="color:#2d3748">–</span>'),
    ]
    _GPTH2 = (_GPTH
              .replace("border-bottom:1px solid rgba(247,147,30,0.30)",
                       "border-bottom:1px solid rgba(247,147,30,0.15)")
              .replace("color:#F7931E", "color:rgba(247,147,30,0.85)"))
    
    _gprows_html = ""
    for _row in _garch_param_rows:
        _glabel, _gpv, _gppv, _gsv, _gspv = _row
        _gprows_html += (
            f'<tr>'
            f'<td style="{_GPTDL}">{_glabel}</td>'
            f'<td style="{_GPTD}">{_gpv}</td>'
            f'<td style="{_GPTD}">{_gppv}</td>'
            f'<td style="{_GPTD}">{_gsv}</td>'
            f'<td style="{_GPTD}">{_gspv}</td>'
            f'</tr>'
        )

    _GPHDR_PORT = f'<th colspan="2" style="{_GPTH};text-align:center;border-left:1px solid rgba(247,147,30,0.15);">PORTFOLIO</th>'
    _GPHDR_SPY  = f'<th colspan="2" style="{_GPTH};text-align:center;border-left:1px solid rgba(247,147,30,0.15);">SPY</th>'
    _GPHDR2 = (f'<th style="{_GPTH2};border-left:1px solid rgba(247,147,30,0.15);">VALUE</th>'
               f'<th style="{_GPTH2}">P-VALUE</th>'
               f'<th style="{_GPTH2};border-left:1px solid rgba(247,147,30,0.15);">VALUE</th>'
               f'<th style="{_GPTH2}">P-VALUE</th>')
    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono',monospace;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead>
  <tr>
    <th style="{_GPTHL}" rowspan="2">PARAMETERS</th>
    {_GPHDR_PORT}
    {_GPHDR_SPY}
  </tr>
  <tr>{_GPHDR2}</tr>
</thead>
<tbody>{_gprows_html}</tbody>
</table></div>""",
        unsafe_allow_html=True,
    )

    # Next-day
    _gfc1, _gfc2 = st.columns(2)
    with _gfc1:
        st.metric("NEXT-DAY VOL — PORTFOLIO", f"{pg['next_day_vol_forecast']*100:.2f}%")
    with _gfc2:
        st.metric("NEXT-DAY VOL — SPY", f"{sg['next_day_vol_forecast']*100:.2f}%")

    st.markdown("---")

    # History
    p_cond = pg["conditional_vol"].dropna()
    s_cond = sg["conditional_vol"].dropna()
    col_m, col_n = st.columns(2)

    port_fcst_col = "portfolio_vol_forecast"
    spy_fcst_col  = "spy_vol_forecast"

    with col_m:

        garch_port_fcst = (
            bt[port_fcst_col].dropna() * np.sqrt(config.TRADING_DAYS)
            if port_fcst_col in bt.columns
            else p_cond
        )
        st.plotly_chart(
            vol_comparison_chart(
                garch_port_fcst,
                port_vol.dropna(),
                title="Portfolio: GARCH Forecast vs Realized Vol",
            ),
            width='stretch',
        )

    with col_n:
        garch_spy_fcst = (
            bt[spy_fcst_col].dropna() * np.sqrt(config.TRADING_DAYS)
            if spy_fcst_col in bt.columns
            else s_cond
        )
        st.plotly_chart(
            vol_comparison_chart(
                garch_spy_fcst,
                spy_vol.dropna(),
                title="SPY: GARCH Forecast vs Realized Vol",
            ),
            width='stretch',
        )

    st.markdown("---")

    col_cv, col_vs = st.columns(2)

    vol_df = pd.concat(
        [p_cond.rename("Portfolio (ann.)"), s_cond.rename("SPY (ann.)")], axis=1
    ).dropna() * 100

    with col_cv:
        st.plotly_chart(
            multi_line_chart(
                vol_df,
                title="GARCH Conditional Volatility",
                y_label="Ann. Vol (%)",
            ),
            width='stretch',
        )

    with col_vs:
        pp = pg["params"]
        st.plotly_chart(
            garch_vol_surface(
                omega=pp["omega"],
                alpha=pp["alpha_1"],
                beta=pp["beta_1"],
                title="Next-Day Vol Sensitivity Surface",
            ),
            width='stretch',
        )

# Portfolio

with tab_portfolio:

    if "pt_excluded" not in st.session_state:
        st.session_state["pt_excluded"] = []

    active_pt = [t for t in selected_tickers if t not in st.session_state["pt_excluded"]]

    _spx   = d["stock_prices"]
    _sret  = d["stock_returns"]
    _pval  = d["position_values"]
    _curw  = d["current_weights"]

    table_rows = []
    for t in active_pt:
        _px  = float(_spx[t].iloc[-1])
        _sh  = float(_pval[t].iloc[-1] / _px) if _px > 0 else 0.0
        _w   = float(_curw[t].iloc[-1]) * 100
        _r   = _sret[t].dropna()
        _n   = len(_r)
        r3m  = float(((1 + _r.iloc[max(0, _n - 63):]).prod()  - 1) * 100) if _n >= 1 else float("nan")
        r6m  = float(((1 + _r.iloc[max(0, _n - 126):]).prod() - 1) * 100) if _n >= 1 else float("nan")
        r1y  = float(((1 + _r.iloc[max(0, _n - 252):]).prod() - 1) * 100) if _n >= 1 else float("nan")
        vol1y = float(_r.iloc[max(0, _n - 252):].std() * np.sqrt(config.TRADING_DAYS) * 100) if _n >= 2 else float("nan")
        _cb  = comp_beta[t].dropna()
        beta_t = float(_cb.iloc[-1]) if len(_cb) else float("nan")
        table_rows.append(dict(t=t, px=_px, sh=_sh, w=_w, r3m=r3m, r6m=r6m, r1y=r1y, vol=vol1y, b=beta_t))

    def _rc(v: float) -> str:
        """Colour-coded return cell."""
        if np.isnan(v):
            return '<span style="color:#2d3748">–</span>'
        c = "#2DCE89" if v >= 0 else "#F5365C"
        return f'<span style="color:{c}">{v:+.1f}%</span>'

    def _bc(v: float) -> str:
        """Beta cell — orange normal, red if high."""
        if np.isnan(v):
            return '<span style="color:#2d3748">–</span>'
        c = "#F5365C" if v > 1.5 else "#F7931E"
        return f'<span style="color:{c}">{v:+.2f}</span>'

    _TH = ("padding:0.30rem 0.55rem;text-align:right;color:#F7931E;"
           "font-size:0.64rem;letter-spacing:0.12em;text-transform:uppercase;"
           "border-bottom:1px solid rgba(247,147,30,0.30);white-space:nowrap;")
    _THL = _TH.replace("text-align:right", "text-align:left")
    _TD  = ("padding:0.18rem 0.55rem;text-align:right;font-size:0.74rem;"
            "color:#9aa5b4;border-bottom:1px solid rgba(255,255,255,0.04);")
    _TDL = _TD.replace("text-align:right", "text-align:left") + "color:rgba(247,147,30,0.9);font-weight:500;"

    rows_html = ""
    for r in table_rows:
        rows_html += (
            f'<tr>'
            f'<td style="{_TDL}">{r["t"]}</td>'
            f'<td style="{_TD}">${r["px"]:.2f}</td>'
            f'<td style="{_TD}">{r["sh"]:.1f}</td>'
            f'<td style="{_TD}">{r["w"]:.1f}%</td>'
            f'<td style="{_TD}">{_rc(r["r3m"])}</td>'
            f'<td style="{_TD}">{_rc(r["r6m"])}</td>'
            f'<td style="{_TD}">{_rc(r["r1y"])}</td>'
            f'<td style="{_TD}">{r["vol"]:.1f}%</td>'
            f'<td style="{_TD}">{_bc(r["b"])}</td>'
            f'</tr>'
        )

    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono',monospace;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr>
  <th style="{_THL}">TICKER</th>
  <th style="{_TH}">PRICE</th>
  <th style="{_TH}"># SHARES</th>
  <th style="{_TH}">WEIGHT</th>
  <th style="{_TH}">RET 3M</th>
  <th style="{_TH}">RET 6M</th>
  <th style="{_TH}">RET 1Y</th>
  <th style="{_TH}">VOL 1Y</th>
  <th style="{_TH}">BETA 6M</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_q, col_r = st.columns(2)

    _spy_clean  = spy_ret.dropna()
    _port_clean = port_ret.dropna()
    _shared_idx = _spy_clean.index.intersection(_port_clean.index)
    _spy_clean  = _spy_clean.loc[_shared_idx]
    _port_clean = _port_clean.loc[_shared_idx]
    _n_ret      = len(_spy_clean)
    spy_6m      = _spy_clean.iloc[max(0, _n_ret - rolling_window):]
    port_6m     = _port_clean.iloc[max(0, _n_ret - rolling_window):]

    reg_latest  = reg.dropna().iloc[-1]

    with col_q:
        st.plotly_chart(
            regression_scatter(
                x=spy_6m,
                y=port_6m,
                alpha=float(reg_latest["alpha"]),
                beta=float(reg_latest["beta"]),
                r_squared=float(reg_latest["r_squared"]),
                title=f"Portfolio vs SPY — Last {rolling_window}d Regression",
            ),
            width='stretch',
        )

    with col_r:
        st.plotly_chart(
            rolling_stats_chart(
                rb.dropna(),
                title="Rolling Beta vs SPY (6M)",
                color="#F7931E",
            ),
            width='stretch',
        )

    st.markdown("---")

    cum_stock = ((1 + d["stock_returns"].dropna()).cumprod() - 1) * 100
    cum_stock.columns = [str(c) for c in cum_stock.columns]
    st.plotly_chart(
        multi_line_chart(
            cum_stock,
            title="Individual Stock Cumulative Returns",
            y_label="Return (%)",
        ),
        width='stretch',
    )
