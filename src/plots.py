from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Colors
C_BLUE    = "#F7931E"   
C_GREEN   = "#2DCE89"
C_RED     = "#F5365C"
C_ORANGE  = "#E85D04" 
C_PURPLE  = "#9B59B6"
C_GREY    = "#8898AA"
C_WHITE   = "#E8ECF0"

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_WHITE, size=11, family="IBM Plex Mono, Roboto Mono, monospace"),
    margin=dict(l=40, r=16, t=42, b=36),
    legend=dict(
        bgcolor="rgba(0,0,0,0.45)",
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.98,
        font=dict(size=9, color="#8898AA", family="IBM Plex Mono, monospace"),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(_LAYOUT_BASE)
    if title:
        layout["title"] = dict(
            text=title.upper(),
            font=dict(size=11, color="#ff8418", family="IBM Plex Mono, Roboto Mono, monospace"),
            x=0.0,
            xanchor="left",
            pad=dict(b=2),
        )
    fig.update_layout(**layout)
    return fig


# Line chart
def line_chart(
    series: pd.Series,
    title: str = "",
    y_label: str = "",
    color: str = C_BLUE,
    fill: bool = False,
):
    """Single-series line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy" if fill else "none",
        fillcolor=color.replace(")", ", 0.10)").replace("rgb", "rgba") if fill else None,
        name=series.name or "",
    ))
    fig.update_yaxes(title_text=y_label)
    return _apply_base(fig, title)


# Multi linbe
def multi_line_chart(
    df: pd.DataFrame,
    title: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
    dash_second: bool = False,
):
    """Multiple series on one chart."""
    palette = colors or [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_RED, C_GREY]
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        dash = "dot" if (dash_second and i > 0) else "solid"
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col].values,
            mode="lines",
            name=str(col),
            line=dict(color=palette[i % len(palette)], width=1.8, dash=dash),
        ))
    fig.update_yaxes(title_text=y_label)
    return _apply_base(fig, title)


# NAV Chart
# Terminal colours for NAV chart
_C_NAV_UNHEDGED = "#F7931E"   # Bloomberg orange  — unhedged (with fill)
_C_NAV_HEDGED   = "#00BFFF"   # CRT phosphor blue — hedged (no fill)

def nav_comparison_chart(
    unhedged_nav: pd.Series,
    hedged_nav: pd.Series,
    title: str = "Unhedged vs Hedged NAV",
):
    """Unhedged vs Hedged NAV."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=unhedged_nav.index, y=unhedged_nav.values,
        mode="lines", name="UNHEDGED",
        line=dict(color=_C_NAV_UNHEDGED, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(247,147,30,0.10)",
    ))

    fig.add_trace(go.Scatter(
        x=hedged_nav.index, y=hedged_nav.values,
        mode="lines", name="HEDGED",
        line=dict(color=_C_NAV_HEDGED, width=1.8),
    ))
    fig.update_yaxes(title_text="USD")
    return _apply_base(fig, title)


# Hedge PnL chart
def hedge_pnl_chart(
    cum_pnl: pd.Series,
    title: str = "Hedge Overlay Cumulative PnL",
):
    """Area chart for cumulative PnL."""
    fig = go.Figure()

    # Fill regions
    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl.clip(lower=0).values,
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tozeroy", fillcolor="rgba(45,206,137,0.18)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl.clip(upper=0).values,
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tozeroy", fillcolor="rgba(245,54,92,0.18)",
        showlegend=False, hoverinfo="skip",
    ))

    # Build color-switching line: split at zero crossings
    s    = cum_pnl.dropna()
    vals = s.values.astype(float)
    idx  = s.index
    n    = len(vals)

    def _zero_cross(k: int):
        frac = abs(vals[k]) / (abs(vals[k]) + abs(vals[k + 1]))
        return idx[k] + frac * (idx[k + 1] - idx[k]), 0.0

    i = 0
    first_seg = True
    while i < n:
        j = i
        while j < n and vals[j] == 0.0:
            j += 1
        if j == n:
            break
        color = C_GREEN if vals[j] >= 0 else C_RED

        seg_x: list = []
        seg_y: list = []

        if i > 0 and vals[i - 1] * vals[i] < 0:
            zx, zy = _zero_cross(i - 1)
            seg_x.append(zx)
            seg_y.append(zy)

        k = i
        while k < n:
            seg_x.append(idx[k])
            seg_y.append(vals[k])
            if k + 1 < n and vals[k] * vals[k + 1] < 0:

                zx, zy = _zero_cross(k)
                seg_x.append(zx)
                seg_y.append(zy)
                k += 1
                break
            k += 1

        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            name="Cum. PnL" if first_seg else None,
            showlegend=first_seg,
            line=dict(color=color, width=1.8),
            hovertemplate="%{y:,.0f}<extra></extra>",
        ))
        first_seg = False
        i = k

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.update_yaxes(title_text="USD")
    return _apply_base(fig, title)


# Rolling beta
def rolling_stats_chart(
    series: pd.Series,
    title: str = "",
    h_line: float | None = None,
    h_label: str = "",
    color: str = C_ORANGE,
):
    """Single rolling series"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=series.name or "",
        line=dict(color=color, width=2.2),
    ))
    if h_line is not None:
        fig.add_hline(
            y=h_line,
            line=dict(color=C_GREY, width=1.0, dash="dash"),
            annotation_text=h_label,
            annotation_font_color=C_GREY,
        )
    return _apply_base(fig, title)


# Fitted reg line
def regression_scatter(
    x: pd.Series,
    y: pd.Series,
    alpha: float,
    beta: float,
    r_squared: float,
    x_label: str = "SPY Returns",
    y_label: str = "Portfolio Returns",
    title: str = "Portfolio vs SPY",
):
    """Scatter plot with fitted line"""
    fig = go.Figure()

    # Scatter
    fig.add_trace(go.Scatter(
        x=x.values * 100, y=y.values * 100,
        mode="markers",
        name="Daily Returns",
        marker=dict(color=C_BLUE, size=6, opacity=0.65),
    ))

    # Regression line
    x_range = np.linspace(x.min(), x.max(), 100)
    y_hat   = alpha + beta * x_range
    fig.add_trace(go.Scatter(
        x=x_range * 100, y=y_hat * 100,
        mode="lines", name=f"β={beta:.3f}  α={alpha:.4f}  R²={r_squared:.3f}",
        line=dict(color=C_RED, width=2),
    ))

    fig.update_xaxes(title_text=x_label + " (%)")
    fig.update_yaxes(title_text=y_label + " (%)")
    return _apply_base(fig, title)



# bar chart for trades
def bar_chart(
    series: pd.Series,
    title: str = "",
    y_label: str = "",
    color: str = C_BLUE,
    color_by_sign: bool = False,
):
    """Classic bar chart"""
    if color_by_sign:
        colors = [C_GREEN if v >= 0 else C_RED for v in series.values]
    else:
        colors = color

    fig = go.Figure(go.Bar(
        x=series.index.astype(str),
        y=series.values,
        marker_color=colors,
        name=series.name or "",
    ))
    fig.update_yaxes(title_text=y_label)
    return _apply_base(fig, title)


# Vol comp.
def vol_comparison_chart(
    forecast_vol: pd.Series,
    realized_vol: pd.Series,
    title: str = "Forecast vs Realized Volatility",
):
    """GARCH forecast vol vs rolling realized vol."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_vol.index, y=(forecast_vol * 100).values,
        mode="lines", name="GARCH Forecast (next-day)",
        line=dict(color=C_ORANGE, width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=realized_vol.index, y=(realized_vol * 100).values,
        mode="lines", name="Realized Vol (20d)",
        line=dict(color=C_GREY, width=1.2, dash="dot"),
    ))
    fig.update_yaxes(title_text="Annualised Vol (%)")
    return _apply_base(fig, title)



# Garch surface
def garch_vol_surface(
    omega: float,
    alpha: float,
    beta: float,
    scale: float = 100.0,
    n_points: int = 30,
    title: str = "GARCH(1,1) — Next-Day Vol Surface",
) -> go.Figure:
    """2D heatmap showing next-day forecast vol as a function of
    """

    prev_sigma = np.linspace(0.2, 4.0, n_points)  
    prev_shock = np.linspace(0.0, 4.0, n_points)   

    Z = np.zeros((n_points, n_points))
    for i, sig in enumerate(prev_sigma):
        for j, eps in enumerate(prev_shock):
            var_next = omega + alpha * (eps ** 2) + beta * (sig ** 2)
            Z[i, j] = np.sqrt(max(var_next, 0.0)) / scale * np.sqrt(252) * 100 

    fig = go.Figure(go.Surface(
        x=np.round(prev_shock, 2),
        y=np.round(prev_sigma, 2),
        z=Z,
        colorscale="Plasma",
        colorbar=dict(
            title=dict(text="Ann. Vol (%)", font=dict(color="#8898AA", size=9)),
            tickfont=dict(color="#8898AA", size=9),
            thickness=12,
        ),
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3, roughness=0.5),
    ))
    fig = _apply_base(fig, title)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="Innovation", font=dict(color="#8898AA", size=9)),
                tickfont=dict(color="#8898AA", size=8),
                gridcolor="rgba(255,255,255,0.08)",
                backgroundcolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                title=dict(text="Conditional Vol.", font=dict(color="#8898AA", size=9)),
                tickfont=dict(color="#8898AA", size=8),
                gridcolor="rgba(255,255,255,0.08)",
                backgroundcolor="rgba(0,0,0,0)",
            ),
            zaxis=dict(
                title=dict(text="Ann. Vol (%)", font=dict(color="#8898AA", size=9)),
                tickfont=dict(color="#8898AA", size=8),
                gridcolor="rgba(255,255,255,0.08)",
                backgroundcolor="rgba(0,0,0,0)",
            ),
            bgcolor="rgba(17,20,25,0.0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=42, b=0),
    )
    return fig

