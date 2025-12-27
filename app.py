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

import logging
import logging.config
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

# ── Logging ───────────────────────────────────────────────────────────────────

_conf = Path(__file__).parent / "logger.conf"
if _conf.exists():
    logging.config.fileConfig(_conf, disable_existing_loggers=False)
log = logging.getLogger("app")

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
    log.info("Downloading price data: %s  [%s → %s]", tickers, start, end)
    t0 = time.perf_counter()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"].to_frame(tickers[0])
    df = prices.dropna().pct_change().dropna()
    log.info("Price data loaded: %d assets, %d trading days (%.2fs)", df.shape[1], df.shape[0], time.perf_counter() - t0)
    return df


@st.cache_data(ttl=3600)
def fetch_euribor_3m() -> float | None:
    log.info("Fetching Euribor 3M (EURIBOR3MD=)")
    try:
        raw = yf.download("EURIBOR3MD=", period="5d", progress=False, auto_adjust=True)
        val = float(raw["Close"].dropna().iloc[-1]) / 100
        log.info("Euribor 3M = %.4f", val)
        return val
    except Exception as e:
        log.warning("Euribor 3M fetch failed: %s", e)
        return None

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


def _base_constraints_bounds(n: int, leverage: bool = False):
    # leverage=True: unconstrained weights (short selling + leverage allowed)
    # leverage=False: long-only, wᵢ ∈ [0, 1]
    bounds = tuple((-np.inf, np.inf) if leverage else (0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    return bounds, constraints


def min_variance_weights(mean_returns: np.ndarray, cov: np.ndarray, leverage: bool = False) -> np.ndarray:
    log.info("min_variance: n=%d  leverage=%s", len(mean_returns), leverage)
    t0 = time.perf_counter()
    n = len(mean_returns)
    bounds, constraints = _base_constraints_bounds(n, leverage)
    result = minimize(
        lambda w: portfolio_stats(w, mean_returns, cov)[1],
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret, vol, _ = portfolio_stats(result.x, mean_returns, cov)
    log.info("min_variance: converged=%s  ret=%.3f  vol=%.3f  (%.3fs)", result.success, ret, vol, time.perf_counter() - t0)
    return result.x


def max_sharpe_weights(mean_returns: np.ndarray, cov: np.ndarray, rf: float = 0.0, leverage: bool = False) -> np.ndarray:
    log.info("max_sharpe: n=%d  rf=%.5f  leverage=%s", len(mean_returns), rf, leverage)
    t0 = time.perf_counter()
    n = len(mean_returns)
    bounds, constraints = _base_constraints_bounds(n, leverage)
    result = minimize(
        lambda w: -(portfolio_stats(w, mean_returns, cov)[0] - rf) / max(portfolio_stats(w, mean_returns, cov)[1], 1e-9),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret, vol, sharpe = portfolio_stats(result.x, mean_returns, cov)
    log.info("max_sharpe: converged=%s  ret=%.3f  vol=%.3f  sharpe=%.3f  (%.3fs)", result.success, ret, vol, sharpe, time.perf_counter() - t0)
    return result.x


def efficient_frontier(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    n_points: int = 80,
    leverage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    log.info("efficient_frontier: %d points  leverage=%s", n_points, leverage)
    t0 = time.perf_counter()
    n = len(mean_returns)
    ann_returns = mean_returns * 252
    frontier_vols, frontier_rets = [], []

    for target in np.linspace(ann_returns.min(), ann_returns.max(), n_points):
        bounds, base_constraints = _base_constraints_bounds(n, leverage)
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

    log.info("efficient_frontier: %d/%d points converged (%.2fs)", len(frontier_vols), n_points, time.perf_counter() - t0)
    return np.array(frontier_vols), np.array(frontier_rets)


def monte_carlo(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log.info("monte_carlo: %d portfolios", n_portfolios)
    t0 = time.perf_counter()
    n = len(mean_returns)
    rets, vols, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w, mean_returns, cov)
        rets.append(r); vols.append(v); sharpes.append(s)
    log.info("monte_carlo: done (%.2fs)  max_sharpe=%.3f", time.perf_counter() - t0, max(sharpes))
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

    euribor = fetch_euribor_3m()
    rf_default = round(euribor * 100, 2) if euribor else 4.5
    rf_label = f"Risk-free rate (%/yr)  — Euribor 3M live: {euribor:.2%}" if euribor else "Risk-free rate (%/yr)  — Euribor 3M unavailable"
    rf_rate = st.slider(rf_label, 0.0, 10.0, rf_default, 0.1) / 100
    n_mc = st.select_slider("Monte Carlo portfolios", options=[1_000, 2_000, 5_000, 10_000], value=5_000)
    allow_leverage = st.toggle(
        "Allow leverage & short selling",
        value=False,
        help="Removes the wᵢ ∈ [0,1] constraint. Weights can be negative (short) or >1 (leveraged).",
    )

    run = st.button("🚀 Optimise", width="stretch", type="primary")

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
    mv_w = min_variance_weights(mean_ret, cov, leverage=allow_leverage)
    ms_w = max_sharpe_weights(mean_ret, cov, rf=rf_rate / 252, leverage=allow_leverage)
    ef_vols, ef_rets = efficient_frontier(mean_ret, cov, leverage=allow_leverage)
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Efficient Frontier", "🥧 Allocations", "🔗 Correlation", "⚠️ VaR"])

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
    st.plotly_chart(fig_ef, width="stretch")

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
            st.plotly_chart(fig_pie, width="stretch")
            st.dataframe(df_alloc[["Ticker", "Weight (%)"]].style.bar(
                subset="Weight (%)", color=color + "88"
            ), width="stretch", hide_index=True)

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
    st.plotly_chart(fig_corr, width="stretch")

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
        width="stretch",
        hide_index=True,
    )

# Tab 4 — VaR ─────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("⚠️ Value at Risk — Max Sharpe Portfolio")
    st.caption("Trois méthodes : paramétrique (normale), historique, Monte Carlo GBM")

    var_col1, var_col2 = st.columns([1, 2])
    with var_col1:
        confidence = st.selectbox("Confidence level", [0.95, 0.99, 0.999], index=1, format_func=lambda x: f"{x:.1%}")
        horizon = st.selectbox("Horizon (days)", [1, 5, 10, 21], index=0)
        n_var_sim = st.select_slider("MC simulations", options=[10_000, 50_000, 100_000], value=10_000)
        notional = st.number_input("Portfolio value (€)", value=1_000_000, step=100_000)

    # Portfolio daily returns for the Max Sharpe weights
    port_daily_rets = returns[valid_tickers].values @ ms_w  # shape (T,)

    # ── 1. Parametric VaR (normal distribution) ──────────────────────────────
    from scipy.stats import norm
    port_mu  = float(port_daily_rets.mean())
    port_sig = float(port_daily_rets.std())
    z = norm.ppf(1 - confidence)
    var_param_1d  = -(port_mu + z * port_sig)
    var_param_hd  = var_param_1d * np.sqrt(horizon)   # square-root-of-time scaling
    cvar_param_1d = port_sig * norm.pdf(z) / (1 - confidence) - port_mu

    # ── 2. Historical VaR ────────────────────────────────────────────────────
    var_hist_1d = float(-np.percentile(port_daily_rets, (1 - confidence) * 100))
    var_hist_hd = var_hist_1d * np.sqrt(horizon)
    cvar_hist_1d = float(-port_daily_rets[port_daily_rets <= -var_hist_1d].mean())

    # ── 3. Monte Carlo VaR (GBM) ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    # Simulate horizon-day portfolio P&L via GBM on each asset, then aggregate
    dt = 1 / 252
    chol = np.linalg.cholesky(returns[valid_tickers].cov().values)
    mu_vec = returns[valid_tickers].mean().values
    sim_rets = np.zeros(n_var_sim)
    for _ in range(horizon):
        z_draws = rng.standard_normal((len(valid_tickers), n_var_sim))
        daily = mu_vec[:, None] * dt + chol @ z_draws * np.sqrt(dt)
        sim_rets += (ms_w @ daily)          # cumulate log-approx returns
    var_mc_hd  = float(-np.percentile(sim_rets, (1 - confidence) * 100))
    cvar_mc_hd = float(-sim_rets[sim_rets <= -var_mc_hd].mean())

    # ── Display ───────────────────────────────────────────────────────────────
    with var_col2:
        df_var = pd.DataFrame({
            "Method":        ["Parametric (Normal)", "Historical", "Monte Carlo (GBM)"],
            f"VaR {horizon}d (%)":  [var_param_hd, var_hist_hd, var_mc_hd],
            f"VaR {horizon}d (€)":  [v * notional for v in [var_param_hd, var_hist_hd, var_mc_hd]],
            f"CVaR 1d (%)": [cvar_param_1d, cvar_hist_1d, float("nan")],
        })
        st.dataframe(
            df_var.style
                .format({
                    f"VaR {horizon}d (%)":  "{:.3%}",
                    f"VaR {horizon}d (€)":  "{:,.0f} €",
                    f"CVaR 1d (%)": "{:.3%}",
                })
                .background_gradient(subset=[f"VaR {horizon}d (%)"], cmap="Reds"),
            width="stretch",
            hide_index=True,
        )

    # ── P&L histogram with VaR lines ─────────────────────────────────────────
    fig_var = go.Figure()
    fig_var.add_trace(go.Histogram(
        x=port_daily_rets,
        nbinsx=80,
        name="Historical daily P&L",
        marker_color="#4C9BE8",
        opacity=0.7,
    ))
    for val, label, color in [
        (-var_param_1d, f"Param VaR 1d ({confidence:.0%})", "#F8D347"),
        (-var_hist_1d,  f"Hist  VaR 1d ({confidence:.0%})", "#F08030"),
    ]:
        fig_var.add_vline(x=val, line_color=color, line_dash="dash", line_width=2,
                          annotation_text=label, annotation_position="top left")
    fig_var.update_layout(
        template="plotly_dark",
        title="Max Sharpe Portfolio — Daily Return Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        xaxis_tickformat=".1%",
        height=420,
    )
    st.plotly_chart(fig_var, width="stretch")

    # ── MC P&L distribution ───────────────────────────────────────────────────
    fig_mc_var = go.Figure()
    fig_mc_var.add_trace(go.Histogram(
        x=sim_rets,
        nbinsx=100,
        name=f"MC {horizon}d P&L",
        marker_color="#7EC8A4",
        opacity=0.7,
    ))
    fig_mc_var.add_vline(x=-var_mc_hd, line_color="#F08030", line_dash="dash", line_width=2,
                         annotation_text=f"MC VaR {horizon}d ({confidence:.0%})",
                         annotation_position="top left")
    fig_mc_var.update_layout(
        template="plotly_dark",
        title=f"Monte Carlo GBM — {horizon}-day Portfolio Return Distribution ({n_var_sim:,} simulations)",
        xaxis_title=f"{horizon}-day Return",
        yaxis_title="Frequency",
        xaxis_tickformat=".1%",
        height=380,
    )
    st.plotly_chart(fig_mc_var, width="stretch")
