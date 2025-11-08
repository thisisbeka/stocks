"""Streamlit application entry point."""
from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd
import streamlit as st

from .analytics import compute_sma, linear_regression_forecast
from .data import (
    PriceRequest,
    available_regions,
    available_sectors,
    fetch_price_history,
    parse_tickers,
)
from .portfolio import PortfolioSummary, optimize_portfolio

INSTRUMENT_PLACEHOLDERS = {
    "Stocks": "AAPL, MSFT, GOOGL",
    "Bonds": "TLT, SHY, IEF",
    "Futures": "ES=F, NQ=F, YM=F",
    "Options": "AAPL230318C00145000, MSFT230318P00260000",
}


@st.cache_data(show_spinner=False)
def load_data(price_request: PriceRequest) -> pd.DataFrame:
    return fetch_price_history(price_request)


def _render_sidebar() -> tuple[Sequence[str], date, date, int, float, float]:
    st.sidebar.header("Configuration")
    instrument_type = st.sidebar.selectbox(
        "Instrument type",
        options=list(INSTRUMENT_PLACEHOLDERS.keys()),
        index=0,
    )

    default_tickers = INSTRUMENT_PLACEHOLDERS[instrument_type]
    raw_tickers = st.sidebar.text_input(
        "Tickers",
        value=default_tickers,
        help="Provide a comma-separated list of Yahoo! Finance symbols.",
    )
    tickers = parse_tickers(raw_tickers)

    start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.sidebar.date_input("End date", value=date.today())

    st.sidebar.selectbox("Sector", options=list(available_sectors()), index=0)
    st.sidebar.selectbox("Region", options=list(available_regions()), index=0)

    initial_capital = st.sidebar.number_input(
        "Initial capital ($)", min_value=1000.0, value=10_000.0, step=1000.0
    )
    commission = st.sidebar.number_input(
        "Commission per trade ($)", min_value=0.0, value=0.0, step=0.1
    )

    sma_window = st.sidebar.slider("SMA window", min_value=5, max_value=50, value=20)
    return tickers, start_date, end_date, sma_window, initial_capital, commission


def _render_portfolio_summary(summary: PortfolioSummary) -> None:
    st.subheader("Portfolio allocation")
    weights_df = pd.DataFrame(
        {"Ticker": list(summary.weights.keys()), "Weight": list(summary.weights.values())}
    )
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

    metrics = pd.DataFrame(
        {
            "Metric": ["Expected annual return", "Variance", "Sharpe ratio"],
            "Value": [
                f"{summary.expected_return:.2%}",
                f"{summary.variance:.4f}",
                f"{summary.sharpe_ratio:.2f}",
            ],
        }
    )
    st.table(metrics)


def run() -> None:
    """Render the dashboard UI."""

    st.set_page_config(page_title="Stocks Dashboard", layout="wide")
    st.title("Portfolio optimization and forecasting dashboard")

    (
        tickers,
        start_date,
        end_date,
        sma_window,
        _initial_capital,
        _commission,
    ) = _render_sidebar()

    if not tickers:
        st.info("Enter at least one ticker symbol to begin.")
        return

    try:
        price_request = PriceRequest(tickers=tickers, start=start_date, end=end_date)
        prices = load_data(price_request)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unable to load data: {exc}")
        return

    st.subheader("Historical adjusted close prices")
    st.line_chart(prices)

    try:
        sma = compute_sma(prices, sma_window)
        st.subheader(f"{sma_window}-day simple moving average")
        st.line_chart(sma)
    except ValueError as exc:
        st.warning(str(exc))

    try:
        forecast = linear_regression_forecast(prices)
        st.subheader("Linear regression forecast")
        st.line_chart(forecast)
    except ValueError as exc:
        st.warning(str(exc))

    try:
        summary = optimize_portfolio(prices)
        _render_portfolio_summary(summary)
    except (ValueError, RuntimeError) as exc:
        st.warning(f"Portfolio optimization unavailable: {exc}")


if __name__ == "__main__":  # pragma: no cover - entry point for `streamlit run`
    run()
