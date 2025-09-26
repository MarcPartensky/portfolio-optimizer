"""
Markowitz Mean-Variance Portfolio Optimizer
============================================
- Efficient frontier via scipy SLSQP
- Minimum variance & maximum Sharpe portfolios
- Capital Market Line
- Monte Carlo cloud (5 000 portfolios)
- Allocation pie charts + weight tables
- Correlation heatmap

Run:
    pip install -r requirements.txt
    streamlit run optimizer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark theme overrides */
    [data-testid="stSidebar"] {background: #0d1117;}
    .block-container {padding-top: 1.5rem;}
    h1 {font-size: 1.8rem !important;}
    .stTabs [data-baseweb="tab"] {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"].to_frame(tickers[0])
    return prices.dropna().pct_change().dropna()

# ── Core Markowitz ─────────────────────────────────────────────────────────────

def portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """Returns (annualised_return, annualised_volatility, sharpe_ratio)."""
    ret = float(weights @ mean_returns) * trading_days
    vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(trading_days))
    sharpe = (ret / vol) if vol > 0 else 0.0
    return ret, vol, sharpe


def _base_constraints_bounds(n: int):
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    return bounds, constraints


def min_variance_weights(mean_returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mean_returns)
    bounds, constraints = _base_constraints_bounds(n)
    result = minimize(
        lambda w: portfolio_stats(w, mean_returns, cov)[1],
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def max_sharpe_weights(mean_returns: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    n = len(mean_returns)
    bounds, constraints = _base_constraints_bounds(n)
    result = minimize(
        lambda w: -(portfolio_stats(w, mean_returns, cov)[0] - rf) / max(portfolio_stats(w, mean_returns, cov)[1], 1e-9),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def efficient_frontier(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    n_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(mean_returns)
    ann_returns = mean_returns * 252
    frontier_vols, frontier_rets = [], []

    for target in np.linspace(ann_returns.min(), ann_returns.max(), n_points):
        bounds, base_constraints = _base_constraints_bounds(n)
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, t=target: float(w @ mean_returns) * 252 - t}
        ]
        result = minimize(
            lambda w: portfolio_stats(w, mean_returns, cov)[1],
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)

    return np.array(frontier_vols), np.array(frontier_rets)


def monte_carlo(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(mean_returns)
    rets, vols, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w, mean_returns, cov)
        rets.append(r); vols.append(v); sharpes.append(s)
    return np.array(rets), np.array(vols), np.array(sharpes)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parameters")

    raw_tickers = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, JPM, GS, XOM, JNJ",
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    rf_rate = st.slider("Risk-free rate (%/yr)", 0.0, 10.0, 4.5, 0.1) / 100
    n_mc = st.select_slider("Monte Carlo portfolios", options=[1_000, 2_000, 5_000, 10_000], value=5_000)

    run = st.button("🚀 Optimise", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📈 Markowitz Portfolio Optimizer")
st.caption("Mean-variance optimisation · Efficient Frontier · Max Sharpe · Min Variance")

if not run:
    st.info("Configure parameters in the sidebar and click **Optimise**.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Downloading price data…"):
    returns = load_returns(tickers, str(start_date), str(end_date))

valid_tickers = [t for t in tickers if t in returns.columns]
missing = set(tickers) - set(valid_tickers)

if missing:
    st.warning(f"Tickers not found / no data: {', '.join(missing)}")

if len(valid_tickers) < 2:
    st.error("Need at least 2 valid tickers.")
    st.stop()

returns = returns[valid_tickers]
mean_ret = returns.mean().values
cov = returns.cov().values

# ── Optimise ──────────────────────────────────────────────────────────────────

with st.spinner("Running optimisation…"):
    mv_w = min_variance_weights(mean_ret, cov)
    ms_w = max_sharpe_weights(mean_ret, cov, rf=rf_rate / 252)
    ef_vols, ef_rets = efficient_frontier(mean_ret, cov)
    mc_rets, mc_vols, mc_sharpes = monte_carlo(mean_ret, cov, n_portfolios=n_mc)

mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv_w, mean_ret, cov)
ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_w, mean_ret, cov)

# ── KPI row ───────────────────────────────────────────────────────────────────

kpi_cols = st.columns(6)
kpi_cols[0].metric("Assets", len(valid_tickers))
kpi_cols[1].metric("Min-Var Return", f"{mv_ret:.2%}")
kpi_cols[2].metric("Min-Var Vol", f"{mv_vol:.2%}")
kpi_cols[3].metric("Max-Sharpe Return", f"{ms_ret:.2%}")
kpi_cols[4].metric("Max-Sharpe Vol", f"{ms_vol:.2%}")
kpi_cols[5].metric("Max Sharpe Ratio", f"{ms_sharpe:.2f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Efficient Frontier", "🥧 Allocations", "🔗 Correlation"])

# Tab 1 — Efficient Frontier ─────────────────────────────────────────────────

with tab1:
    fig_ef = go.Figure()

    # Monte Carlo cloud
    fig_ef.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets,
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
    ))

    # Efficient frontier
    fig_ef.add_trace(go.Scatter(
        x=ef_vols, y=ef_rets,
        mode="lines",
        line=dict(color="#4C9BE8", width=2.5),
        name="Efficient Frontier",
    ))

    # Capital Market Line
    cml_vols = np.linspace(0, ef_vols.max() * 1.15, 100)
    cml_rets = rf_rate + (ms_ret - rf_rate) / ms_vol * cml_vols
    fig_ef.add_trace(go.Scatter(
        x=cml_vols, y=cml_rets,
        mode="lines",
        line=dict(color="#F8D347", width=1.5, dash="dash"),
        name="Capital Market Line",
    ))

    # Min Variance
    fig_ef.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret],
        mode="markers",
        marker=dict(symbol="diamond", size=14, color="#4C9BE8", line=dict(width=2, color="white")),
        name=f"Min Variance (σ={mv_vol:.2%})",
        hovertemplate=f"Min Variance<br>Return: {mv_ret:.2%}<br>Vol: {mv_vol:.2%}<br>Sharpe: {mv_sharpe:.2f}<extra></extra>",
    ))

    # Max Sharpe
    fig_ef.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret],
        mode="markers",
        marker=dict(symbol="star", size=18, color="#F08030", line=dict(width=2, color="white")),
        name=f"Max Sharpe ({ms_sharpe:.2f})",
        hovertemplate=f"Max Sharpe<br>Return: {ms_ret:.2%}<br>Vol: {ms_vol:.2%}<br>Sharpe: {ms_sharpe:.2f}<extra></extra>",
    ))

    # Risk-free
    fig_ef.add_trace(go.Scatter(
        x=[0], y=[rf_rate],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="#7EC8A4"),
        name=f"Risk-free ({rf_rate:.2%})",
    ))

    fig_ef.update_layout(
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
    st.plotly_chart(fig_ef, use_container_width=True)

# Tab 2 — Allocations ─────────────────────────────────────────────────────────

with tab2:
    col_mv, col_ms = st.columns(2)

    for col, weights, label, color in [
        (col_mv, mv_w, "Minimum Variance", "#4C9BE8"),
        (col_ms, ms_w, "Maximum Sharpe",   "#F08030"),
    ]:
        df_alloc = (
            pd.DataFrame({"Ticker": valid_tickers, "Weight": weights})
            .sort_values("Weight", ascending=False)
            .reset_index(drop=True)
        )
        df_alloc["Weight (%)"] = (df_alloc["Weight"] * 100).round(2)

        fig_pie = go.Figure(go.Pie(
            labels=df_alloc["Ticker"],
            values=df_alloc["Weight"],
            hole=0.4,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            title=label,
            template="plotly_dark",
            height=380,
            showlegend=False,
        )

        with col:
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(df_alloc[["Ticker", "Weight (%)"]].style.bar(
                subset="Weight (%)", color=color + "88"
            ), use_container_width=True, hide_index=True)

# Tab 3 — Correlation ─────────────────────────────────────────────────────────

with tab3:
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=valid_tickers, y=valid_tickers,
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title="ρ"),
    ))
    fig_corr.update_layout(
        title="Asset Correlation Matrix",
        template="plotly_dark",
        height=520,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Summary stats
    st.subheader("📋 Summary Statistics")
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    summary = pd.DataFrame({
        "Ticker": valid_tickers,
        "Ann. Return": ann_ret.values,
        "Ann. Volatility": ann_vol.values,
        "Sharpe (vs Rf)": ((ann_ret - rf_rate) / ann_vol).values,
    })
    st.dataframe(
        summary.style
            .format({"Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe (vs Rf)": "{:.2f}"})
            .background_gradient(subset="Sharpe (vs Rf)", cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )
