from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from settings import cfg


def parametric_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float,
    horizon: int,
) -> dict:
    port = returns.values @ weights
    mu, sig = port.mean(), port.std()
    z = norm.ppf(1 - confidence)
    var_1d = -(mu + z * sig)
    return {
        "var": var_1d * np.sqrt(horizon),
        "cvar": sig * norm.pdf(z) / (1 - confidence) - mu,
    }


def historical_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float,
    horizon: int,
) -> dict:
    port = returns.values @ weights
    var_1d = float(-np.percentile(port, (1 - confidence) * 100))
    tail = port[port <= -var_1d]
    return {
        "var": var_1d * np.sqrt(horizon),
        "cvar": float(-tail.mean()) if len(tail) > 0 else float("nan"),
    }


def montecarlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float,
    horizon: int,
    n_sim: int | None = None,
) -> dict:
    if n_sim is None:
        n_sim = cfg["var"]["mc_simulations"]

    rng = np.random.default_rng(42)
    chol = np.linalg.cholesky(returns.cov().values)
    mu_vec = returns.mean().values
    dt = 1 / cfg["portfolio"]["trading_days"]
    sim = np.zeros(n_sim)

    for _ in range(horizon):
        z = rng.standard_normal((returns.shape[1], n_sim))
        sim += weights @ (mu_vec[:, None] * dt + chol @ z * np.sqrt(dt))

    var = float(-np.percentile(sim, (1 - confidence) * 100))
    return {
        "var": var,
        "cvar": float(-sim[sim <= -var].mean()),
        "sim": sim,
    }
