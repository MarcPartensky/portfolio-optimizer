# Markowitz Portfolio Optimizer

Live demo: [portfolio-optimizer.marcpartensky.com](https://portfolio-optimizer.marcpartensky.com)

Mean-variance portfolio optimization with efficient frontier, Monte Carlo simulation, and Value at Risk — built with Python and Streamlit.

---

## Theory

### Markowitz mean-variance optimization

Harry Markowitz (1952) showed that for a given level of expected return, there exists a portfolio that minimizes risk (volatility). The set of all such portfolios forms the **efficient frontier**.

For a portfolio with weights **w**, expected returns **μ** and covariance matrix **Σ**:

```
Return    : r = w · μ · 252
Volatility: σ = √(wᵀ Σ w) · √252
Sharpe    : S = (r - rf) / σ
```

Two special portfolios are identified:

- **Minimum Variance** — lowest volatility regardless of return
- **Maximum Sharpe** — best risk-adjusted return given a risk-free rate

Both are found via constrained optimization (SLSQP) with the constraint `Σwᵢ = 1` and optionally `wᵢ ∈ [0, 1]` (long-only). Enabling leverage removes the upper bound, allowing short positions (`wᵢ < 0`) and leverage (`wᵢ > 1`).

### Capital Market Line

The CML connects the risk-free rate to the Maximum Sharpe portfolio. Any portfolio on this line dominates portfolios on the efficient frontier for the same level of risk.

```
CML: r(σ) = rf + (r_ms - rf) / σ_ms · σ
```

### Monte Carlo cloud

5000 random portfolios are generated via Dirichlet sampling and plotted in risk-return space. This visualizes the feasible set and shows how the efficient frontier bounds it.

### Value at Risk

VaR at confidence level α over horizon h is the loss not exceeded with probability α. Three methods are implemented:

**Parametric (Normal)** — assumes returns follow a normal distribution. Fast but underestimates tail risk.

```
VaR_1d = -(μ + z_α · σ)
VaR_hd = VaR_1d · √h          (square-root-of-time scaling)
CVaR   = σ · φ(z_α) / (1-α) - μ
```

**Historical** — uses the empirical percentile of past returns. No distributional assumption, limited by history length.

```
VaR_1d = -percentile(returns, (1-α) · 100)
```

**Monte Carlo GBM** — simulates future asset paths via Geometric Brownian Motion with Cholesky decomposition to preserve inter-asset correlations.

```
dS = μ·S·dt + σ·S·dW
```

The three methods give different results by design — each reflects different assumptions about the return distribution.

---

## Project structure

```
portfolio-optimizer/
├── app.py          # Streamlit UI — routing only, no business logic
├── optimizer.py    # Markowitz core: efficient frontier, min variance, max sharpe
├── var.py          # VaR: parametric, historical, Monte Carlo GBM
├── data.py         # yfinance download + Euribor fetch with caching
├── charts.py       # Plotly figures
├── settings.py     # YAML config loader
├── settings.yml    # All parameters (tickers, windows, VaR defaults)
└── Justfile        # Task runner
```

---

## Quickstart

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/marcpartensky/portfolio-optimizer
cd portfolio-optimizer
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Available commands

```bash
just app     # run the app
just lint    # ruff linter
just clean   # remove venv and cache
```

---

## Configuration

All parameters are in `settings.yml`:

```yaml
data:
  default_tickers: "AAPL, MSFT, GOOGL, AMZN, JPM, GS, XOM, JNJ"
  cache_ttl: 3600

portfolio:
  trading_days: 252
  default_rf_pct: 4.5
  frontier_points: 80
  mc_portfolios: 5000

var:
  default_confidence: 0.99
  default_horizon: 1
  mc_simulations: 10000
  default_notional: 1000000
```

---

## Stack

- [Streamlit](https://streamlit.io) — UI
- [scipy](https://scipy.org) — SLSQP optimizer
- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- [Plotly](https://plotly.com) — charts
- [uv](https://github.com/astral-sh/uv) — package manager
