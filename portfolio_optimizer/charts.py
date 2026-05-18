from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "#4C9BE8"
ORANGE = "#F08030"
YELLOW = "#F8D347"
GREEN = "#7EC8A4"
GRAY = "#8B949E"


def plot_frontier(
    ef_vols: np.ndarray,
    ef_rets: np.ndarray,
    mc_vols: np.ndarray,
    mc_rets: np.ndarray,
    mc_sharpes: np.ndarray,
    mv_vol: float,
    mv_ret: float,
    mv_sharpe: float,
    ms_vol: float,
    ms_ret: float,
    ms_sharpe: float,
    rf_rate: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mc_vols,
            y=mc_rets,
            mode="markers",
            marker=dict(
                color=mc_sharpes,
                colorscale="Viridis",
                size=3,
                opacity=0.5,
                colorbar=dict(title="Sharpe", thickness=12),
            ),
            name="Monte Carlo",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ef_vols,
            y=ef_rets,
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            name="Efficient Frontier",
        )
    )

    cml_vols = np.linspace(0, ef_vols.max() * 1.15, 100)
    cml_rets = rf_rate + (ms_ret - rf_rate) / ms_vol * cml_vols
    fig.add_trace(
        go.Scatter(
            x=cml_vols,
            y=cml_rets,
            mode="lines",
            line=dict(color=YELLOW, width=1.5, dash="dash"),
            name="Capital Market Line",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers",
            marker=dict(
                symbol="diamond", size=14, color=BLUE, line=dict(width=2, color="white")
            ),
            name=f"Min Variance (σ={mv_vol:.2%})",
            hovertemplate=f"Min Variance<br>Return: {mv_ret:.2%}<br>Vol: {mv_vol:.2%}<br>Sharpe: {mv_sharpe:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[ms_vol],
            y=[ms_ret],
            mode="markers",
            marker=dict(
                symbol="star", size=18, color=ORANGE, line=dict(width=2, color="white")
            ),
            name=f"Max Sharpe ({ms_sharpe:.2f})",
            hovertemplate=f"Max Sharpe<br>Return: {ms_ret:.2%}<br>Vol: {ms_vol:.2%}<br>Sharpe: {ms_sharpe:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[rf_rate],
            mode="markers",
            marker=dict(symbol="circle", size=10, color=GREEN),
            name=f"Risk-free ({rf_rate:.2%})",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Efficient Frontier & Capital Market Line",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(orientation="v", x=1.12, y=1),
        height=560,
        margin=dict(r=160),
    )
    return fig


def plot_allocations(
    tickers: list[str],
    mv_weights: np.ndarray,
    ms_weights: np.ndarray,
) -> tuple[go.Figure, go.Figure, pd.DataFrame, pd.DataFrame]:
    figures, dataframes = [], []

    for weights, label, color in [
        (mv_weights, "Minimum Variance", BLUE),
        (ms_weights, "Maximum Sharpe", ORANGE),
    ]:
        df = (
            pd.DataFrame({"Ticker": tickers, "Weight": weights})
            .sort_values("Weight", ascending=False)
            .reset_index(drop=True)
        )
        df["Weight (%)"] = (df["Weight"] * 100).round(2)

        fig = go.Figure(
            go.Pie(
                labels=df["Ticker"],
                values=df["Weight"],
                hole=0.4,
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            title=label, template="plotly_dark", height=380, showlegend=False
        )

        figures.append(fig)
        dataframes.append(df)

    return figures[0], figures[1], dataframes[0], dataframes[1]


def plot_correlation(
    returns: pd.DataFrame,
    tickers: list[str],
    rf_rate: float,
) -> tuple[go.Figure, pd.DataFrame]:
    corr = returns.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=tickers,
            y=tickers,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(
        title="Asset Correlation Matrix", template="plotly_dark", height=520
    )

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    summary = pd.DataFrame(
        {
            "Ticker": tickers,
            "Ann. Return": ann_ret.values,
            "Ann. Vol": ann_vol.values,
            "Sharpe": ((ann_ret - rf_rate) / ann_vol).values,
        }
    )
    return fig, summary


def plot_var_histogram(
    port_daily_rets: np.ndarray,
    var_param_1d: float,
    var_hist_1d: float,
    confidence: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=port_daily_rets,
            nbinsx=80,
            name="Historical daily P&L",
            marker_color=BLUE,
            opacity=0.7,
        )
    )
    for val, label, color in [
        (-var_param_1d, f"Param VaR 1d ({confidence:.0%})", YELLOW),
        (-var_hist_1d, f"Hist  VaR 1d ({confidence:.0%})", ORANGE),
    ]:
        fig.add_vline(
            x=val,
            line_color=color,
            line_dash="dash",
            line_width=2,
            annotation_text=label,
            annotation_position="top left",
        )
    fig.update_layout(
        template="plotly_dark",
        title="Max Sharpe Portfolio — Daily Return Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        xaxis_tickformat=".1%",
        height=420,
    )
    return fig


def plot_mc_var_histogram(
    sim_rets: np.ndarray,
    var_mc: float,
    confidence: float,
    horizon: int,
    n_sim: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sim_rets,
            nbinsx=100,
            name=f"MC {horizon}d P&L",
            marker_color=GREEN,
            opacity=0.7,
        )
    )
    fig.add_vline(
        x=-var_mc,
        line_color=ORANGE,
        line_dash="dash",
        line_width=2,
        annotation_text=f"MC VaR {horizon}d ({confidence:.0%})",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Monte Carlo GBM — {horizon}-day Portfolio Return Distribution ({n_sim:,} simulations)",
        xaxis_title=f"{horizon}-day Return",
        yaxis_title="Frequency",
        xaxis_tickformat=".1%",
        height=380,
    )
    return fig
