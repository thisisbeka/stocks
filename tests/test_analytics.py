import numpy as np
import pandas as pd
import pytest

from stocks_dashboard.analytics import compute_sma, linear_regression_forecast


def test_compute_sma_returns_expected_values():
    prices = pd.DataFrame({
        "AAA": [1, 2, 3, 4, 5],
        "BBB": [5, 4, 3, 2, 1],
    })
    sma = compute_sma(prices, window=3)

    expected = pd.DataFrame({
        "AAA": [np.nan, np.nan, 2.0, 3.0, 4.0],
        "BBB": [np.nan, np.nan, 4.0, 3.0, 2.0],
    })
    pd.testing.assert_frame_equal(sma, expected)


@pytest.mark.parametrize("window", [0, -1])
def test_compute_sma_with_invalid_window_raises(window):
    prices = pd.DataFrame({"AAA": [1, 2, 3]})
    with pytest.raises(ValueError):
        compute_sma(prices, window=window)


def test_linear_regression_forecast_matches_input_shape():
    prices = pd.DataFrame({"AAA": np.linspace(10, 20, num=10)})
    forecast = linear_regression_forecast(prices)
    assert forecast.shape == prices.shape
    assert forecast.columns.tolist() == ["AAA"]


def test_linear_regression_forecast_with_empty_frame_raises():
    prices = pd.DataFrame(columns=["AAA"])
    with pytest.raises(ValueError):
        linear_regression_forecast(prices)
