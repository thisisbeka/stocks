"""Portfolio optimization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioSummary:
    weights: Dict[str, float]
    expected_return: float
    variance: float
    sharpe_ratio: float


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.dropna(how="all")
    if cleaned.empty or cleaned.shape[1] == 0:
        raise ValueError("Portfolio optimization requires at least one price series.")
    return cleaned


def optimize_portfolio(prices: pd.DataFrame) -> PortfolioSummary:
    """Compute minimum-variance weights for the supplied price history."""

    cleaned = _validate_prices(prices)
    returns = cleaned.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough data to compute price returns.")

    mean_returns = returns.mean() * 252
    if (returns.var(ddof=0) <= 0).all():
        raise ValueError("Asset returns must exhibit variance for optimisation.")

    cov_matrix = returns.cov() * 252
    cov_values = cov_matrix.to_numpy()
    if not np.isfinite(cov_values).all():
        raise ValueError("Covariance matrix contains non-finite values.")
    if np.allclose(cov_values, 0):
        raise ValueError("Covariance matrix is singular; add more variance to the inputs.")

    num_assets = len(mean_returns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights.T @ cov_values @ weights)

    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    optimization = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not optimization.success:
        raise RuntimeError("Portfolio optimization did not converge.")

    weights = optimization.x
    expected_return = float(np.dot(mean_returns, weights))
    variance = portfolio_variance(weights)
    sharpe_ratio = expected_return / np.sqrt(variance) if variance > 0 else 0.0

    weight_map = {ticker: weight for ticker, weight in zip(cleaned.columns, weights)}

    return PortfolioSummary(
        weights=weight_map,
        expected_return=expected_return,
        variance=variance,
        sharpe_ratio=sharpe_ratio,
    )
