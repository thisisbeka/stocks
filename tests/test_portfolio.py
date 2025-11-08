import numpy as np
import pandas as pd
import pytest

from stocks_dashboard.portfolio import PortfolioSummary, optimize_portfolio


def generate_price_series(rows: int = 30) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    data = {
        "AAA": np.linspace(100, 110, num=rows),
        "BBB": np.linspace(50, 60, num=rows) + np.random.default_rng(1).normal(0, 1, size=rows),
    }
    return pd.DataFrame(data, index=index)


def test_optimize_portfolio_returns_valid_summary():
    prices = generate_price_series()
    summary = optimize_portfolio(prices)
    assert isinstance(summary, PortfolioSummary)
    assert set(summary.weights.keys()) == set(prices.columns)
    assert pytest.approx(sum(summary.weights.values()), abs=1e-6) == 1.0
    assert summary.variance >= 0


def test_optimize_portfolio_with_insufficient_data_raises():
    prices = pd.DataFrame({"AAA": [1.0, 1.0]})
    with pytest.raises(ValueError):
        optimize_portfolio(prices)
