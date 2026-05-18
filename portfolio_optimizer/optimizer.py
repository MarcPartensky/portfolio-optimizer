from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from portfolio_optimizer.settings import cfg

TRADING_DAYS = cfg["portfolio"]["trading_days"]
FRONTIER_POINTS = cfg["portfolio"]["frontier_points"]


def portfolio_stats(
    weights: np.ndarray,
    mean_ret: np.ndarray,
    cov: np.ndarray,
) -> tuple[float, float, float]:
    """Returns (annualised_return, annualised_volatility, sharpe_ratio)."""
    ret = float(weights @ mean_ret) * TRADING_DAYS
    vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(TRADING_DAYS))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _bounds_constraints(n: int, leverage: bool) -> tuple:
    bounds = tuple((-np.inf, np.inf) if leverage else (0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    return bounds, constraints


def min_variance(
    mean_ret: np.ndarray,
    cov: np.ndarray,
    leverage: bool = False,
) -> np.ndarray:
    n = len(mean_ret)
    bounds, constraints = _bounds_constraints(n, leverage)
    res = minimize(
        lambda w: portfolio_stats(w, mean_ret, cov)[1],
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x


def max_sharpe(
    mean_ret: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    leverage: bool = False,
) -> np.ndarray:
    n = len(mean_ret)
    bounds, constraints = _bounds_constraints(n, leverage)
    res = minimize(
        lambda w: -(portfolio_stats(w, mean_ret, cov)[0] - rf)
        / max(portfolio_stats(w, mean_ret, cov)[1], 1e-9),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x


def efficient_frontier(
    mean_ret: np.ndarray,
    cov: np.ndarray,
    leverage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(mean_ret)
    ann_ret = mean_ret * TRADING_DAYS
    vols, rets = [], []

    for target in np.linspace(ann_ret.min(), ann_ret.max(), FRONTIER_POINTS):
        bounds, base = _bounds_constraints(n, leverage)
        constraints = base + [
            {
                "type": "eq",
                "fun": lambda w, t=target: float(w @ mean_ret) * TRADING_DAYS - t,
            }
        ]
        res = minimize(
            lambda w: portfolio_stats(w, mean_ret, cov)[1],
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            vols.append(res.fun)
            rets.append(target)

    return np.array(vols), np.array(rets)


def monte_carlo(
    mean_ret: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_portfolios is None:
        n_portfolios = cfg["portfolio"]["mc_portfolios"]
    n = len(mean_ret)
    rets, vols, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w, mean_ret, cov)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)
    return np.array(rets), np.array(vols), np.array(sharpes)
