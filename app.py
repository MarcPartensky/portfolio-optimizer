"""
Markowitz Mean-Variance Portfolio Optimizer
============================================
Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio_optimizer.data import load_returns, fetch_euribor_3m
from portfolio_optimizer.optimizer import (
    min_variance,
    max_sharpe,
    efficient_frontier,
    monte_carlo,
    portfolio_stats,
)

from portfolio_optimizer.settings import cfg

from portfolio_optimizer.var import parametric_var, historical_var, montecarlo_var
from portfolio_optimizer.charts import (
    plot_frontier,
    plot_allocations,
    plot_correlation,
    plot_var_histogram,
    plot_mc_var_histogram,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=cfg["ui"]["page_title"],
    page_icon=cfg["ui"]["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] {background: #0d1117;}
    .block-container {padding-top: 1.5rem;}
    h1 {font-size: 1.8rem !important;}
    .stTabs [data-baseweb="tab"] {font-size: 0.9rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parameters")

    raw_tickers = st.text_input(
        "Tickers (comma-separated)", value=cfg["data"]["default_tickers"]
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    euribor = fetch_euribor_3m()
    rf_default = (
        round(euribor * 100, 2) if euribor else cfg["portfolio"]["default_rf_pct"]
    )
    rf_label = (
        f"Risk-free rate (%/yr)  — Euribor 3M live: {euribor:.2%}"
        if euribor
        else "Risk-free rate (%/yr)  — Euribor 3M unavailable"
    )
    rf_rate = st.slider(rf_label, 0.0, 10.0, rf_default, 0.1) / 100

    n_mc = st.select_slider(
        "Monte Carlo portfolios",
        options=[1_000, 2_000, 5_000, 10_000],
        value=cfg["portfolio"]["mc_portfolios"],
    )
    allow_leverage = st.toggle(
        "Allow leverage & short selling",
        value=False,
        help="Removes the wᵢ ∈ [0,1] constraint. Weights can be negative (short) or >1 (leveraged).",
    )
    run = st.button("🚀 Optimise", type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📈 Markowitz Portfolio Optimizer")
st.caption(
    "Mean-variance optimisation · Efficient Frontier · Max Sharpe · Min Variance"
)

if not run:
    st.info("Configure parameters in the sidebar and click **Optimise**.")
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────

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
    mv_w = min_variance(mean_ret, cov, leverage=allow_leverage)
    ms_w = max_sharpe(mean_ret, cov, rf=rf_rate / 252, leverage=allow_leverage)
    ef_vols, ef_rets = efficient_frontier(mean_ret, cov, leverage=allow_leverage)
    mc_rets, mc_vols, mc_sharpes = monte_carlo(mean_ret, cov, n_portfolios=n_mc)

mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv_w, mean_ret, cov)
ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_w, mean_ret, cov)

# ── KPI row ───────────────────────────────────────────────────────────────────

kpi = st.columns(6)
kpi[0].metric("Assets", len(valid_tickers))
kpi[1].metric("Min-Var Return", f"{mv_ret:.2%}")
kpi[2].metric("Min-Var Vol", f"{mv_vol:.2%}")
kpi[3].metric("Max-Sharpe Return", f"{ms_ret:.2%}")
kpi[4].metric("Max-Sharpe Vol", f"{ms_vol:.2%}")
kpi[5].metric("Max Sharpe Ratio", f"{ms_sharpe:.2f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Efficient Frontier", "🥧 Allocations", "🔗 Correlation", "⚠️ VaR"]
)

# Tab 1 — Efficient Frontier ──────────────────────────────────────────────────

with tab1:
    st.plotly_chart(
        plot_frontier(
            ef_vols,
            ef_rets,
            mc_vols,
            mc_rets,
            mc_sharpes,
            mv_vol,
            mv_ret,
            mv_sharpe,
            ms_vol,
            ms_ret,
            ms_sharpe,
            rf_rate,
        ),
        width="stretch",
    )

# Tab 2 — Allocations ─────────────────────────────────────────────────────────

with tab2:
    fig_mv, fig_ms, df_mv, df_ms = plot_allocations(valid_tickers, mv_w, ms_w)
    col_mv, col_ms = st.columns(2)
    with col_mv:
        st.plotly_chart(fig_mv, width="stretch")
        st.dataframe(
            df_mv[["Ticker", "Weight (%)"]].style.bar(
                subset="Weight (%)", color="#4C9BE888"
            ),
            width="stretch",
            hide_index=True,
        )
    with col_ms:
        st.plotly_chart(fig_ms, width="stretch")
        st.dataframe(
            df_ms[["Ticker", "Weight (%)"]].style.bar(
                subset="Weight (%)", color="#F0803088"
            ),
            width="stretch",
            hide_index=True,
        )

# Tab 3 — Correlation ─────────────────────────────────────────────────────────

with tab3:
    fig_corr, summary = plot_correlation(returns, valid_tickers, rf_rate)
    st.plotly_chart(fig_corr, width="stretch")
    st.subheader("📋 Summary Statistics")
    st.dataframe(
        summary.style.format(
            {"Ann. Return": "{:.2%}", "Ann. Vol": "{:.2%}", "Sharpe": "{:.2f}"}
        ).background_gradient(subset="Sharpe", cmap="RdYlGn"),
        width="stretch",
        hide_index=True,
    )

# Tab 4 — VaR ─────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("⚠️ Value at Risk — Max Sharpe Portfolio")
    st.caption("Three methods: parametric (normal), historical, Monte Carlo GBM")

    var_col1, var_col2 = st.columns([1, 2])
    with var_col1:
        confidence = st.selectbox(
            "Confidence level",
            [0.95, 0.99, 0.999],
            index=1,
            format_func=lambda x: f"{x:.1%}",
        )
        horizon = st.selectbox("Horizon (days)", [1, 5, 10, 21], index=0)
        n_var_sim = st.select_slider(
            "MC simulations",
            options=[10_000, 50_000, 100_000],
            value=cfg["var"]["mc_simulations"],
        )
        notional = st.number_input(
            "Portfolio value (€)",
            value=cfg["var"]["default_notional"],
            step=100_000,
        )

    pvar = parametric_var(returns, ms_w, confidence, horizon)
    hvar = historical_var(returns, ms_w, confidence, horizon)
    mvar = montecarlo_var(returns, ms_w, confidence, horizon, n_sim=n_var_sim)

    with var_col2:
        df_var = pd.DataFrame(
            {
                "Method": ["Parametric (Normal)", "Historical", "Monte Carlo (GBM)"],
                f"VaR {horizon}d (%)": [pvar["var"], hvar["var"], mvar["var"]],
                f"VaR {horizon}d (€)": [
                    v * notional for v in [pvar["var"], hvar["var"], mvar["var"]]
                ],
                f"CVaR 1d (%)": [pvar["cvar"], hvar["cvar"], float("nan")],
            }
        )
        st.dataframe(
            df_var.style.format(
                {
                    f"VaR {horizon}d (%)": "{:.3%}",
                    f"VaR {horizon}d (€)": "{:,.0f} €",
                    f"CVaR 1d (%)": "{:.3%}",
                }
            ).background_gradient(subset=[f"VaR {horizon}d (%)"], cmap="Reds"),
            width="stretch",
            hide_index=True,
        )

    port_daily_rets = returns.values @ ms_w
    var_param_1d = pvar["var"] / (horizon**0.5)
    var_hist_1d = hvar["var"] / (horizon**0.5)

    st.plotly_chart(
        plot_var_histogram(port_daily_rets, var_param_1d, var_hist_1d, confidence),
        width="stretch",
    )
    st.plotly_chart(
        plot_mc_var_histogram(mvar["sim"], mvar["var"], confidence, horizon, n_var_sim),
        width="stretch",
    )
