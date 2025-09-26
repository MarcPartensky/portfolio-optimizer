# portfolio-optimizer

Markowitz mean-variance portfolio optimizer with an interactive Streamlit interface. Computes the efficient frontier, minimum variance portfolio, and maximum Sharpe ratio portfolio for any set of Yahoo Finance tickers.

## Features

- **Efficient frontier** — solved via `scipy.optimize.minimize` (SLSQP), not just random sampling
- **Minimum variance portfolio** — lowest achievable volatility, long-only
- **Maximum Sharpe portfolio** — optimal risk/return trade-off given a risk-free rate
- **Capital Market Line** — tangent line from risk-free rate to max Sharpe portfolio
- **Monte Carlo cloud** — up to 10 000 random portfolios coloured by Sharpe ratio
- **Allocation breakdown** — pie charts and weight tables for both optimal portfolios
- **Correlation matrix** — identify diversification opportunities
- **Summary statistics** — per-asset annualised return, volatility, Sharpe ratio

## Usage

```bash
pip install -r requirements.txt
streamlit run optimizer.py
```

Configure in the sidebar:

- Enter any valid Yahoo Finance tickers (comma-separated)
- Set the date range and risk-free rate
- Adjust the Monte Carlo sample size
- Click **Optimise**

## Theory

**Mean-variance optimisation** (Markowitz, 1952) finds portfolio weights $w$ that minimise volatility for a given target return:

$$\min_{w} \; w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mathbf{1} = 1, \; w^\top \mu = \mu^*, \; w_i \geq 0$$

The **Sharpe ratio** measures risk-adjusted return:

$$S = \frac{\mu_p - r_f}{\sigma_p}$$

The **efficient frontier** is the upper boundary of the minimum-variance set: no other portfolio offers higher return for the same volatility.

The **Capital Market Line** is the line from the risk-free asset tangent to the efficient frontier, touching it at the maximum Sharpe portfolio.

## Limitations

- Long-only (weights $\in [0, 1]$)
- Historical returns used as proxy for expected returns (estimation risk not addressed)
- No transaction costs or liquidity constraints
- Assumes normal return distribution (fat tails not modelled)
