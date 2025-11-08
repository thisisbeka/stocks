"""Analytics helpers for computing indicators and forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the simple moving average for each column."""

    if window <= 0:
        raise ValueError("Window size must be positive.")
    if window > len(prices.index):
        raise ValueError("Window is larger than the number of available samples.")
    return prices.rolling(window=window, min_periods=window).mean()


def linear_regression_forecast(prices: pd.DataFrame) -> pd.DataFrame:
    """Fit a linear regression for each symbol and return the fitted curve."""

    if prices.empty:
        raise ValueError("Cannot train a regression model on an empty price series.")

    x = np.arange(len(prices), dtype=float).reshape(-1, 1)
    forecasts: dict[str, np.ndarray] = {}
    for column in prices.columns:
        series = prices[column].dropna()
        if series.empty:
            raise ValueError(f"Ticker '{column}' does not contain any observations.")
        model = LinearRegression().fit(x[: len(series)], series.to_numpy())
        predictions = model.predict(x)
        forecasts[column] = predictions

    forecast_df = pd.DataFrame(forecasts, index=prices.index)
    return forecast_df
