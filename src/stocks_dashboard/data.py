"""Data access helpers for the stocks dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceRequest:
    """Parameters describing a Yahoo! Finance price history request."""

    tickers: Sequence[str]
    start: date
    end: date

    def normalized_tickers(self) -> list[str]:
        return [ticker.strip().upper() for ticker in self.tickers if ticker.strip()]


def fetch_price_history(price_request: PriceRequest) -> pd.DataFrame:
    """Download adjusted close price history for the requested tickers.

    Parameters
    ----------
    price_request:
        Request descriptor specifying tickers and date range.

    Returns
    -------
    pandas.DataFrame
        Data frame indexed by date with one column per ticker.
    """

    tickers = price_request.normalized_tickers()
    if not tickers:
        raise ValueError("At least one ticker symbol must be provided.")

    data = yf.download(
        tickers,
        start=price_request.start,
        end=price_request.end,
        progress=False,
    )

    if data.empty:
        raise ValueError("No pricing data returned for the requested parameters.")

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance returns a multi-index when querying multiple tickers
        data = data["Adj Close"].copy()
    else:
        data = data.rename(columns={"Adj Close": tickers[0]}).copy()

    data = data.sort_index()
    return data


def parse_tickers(raw_tickers: str) -> list[str]:
    """Parse a comma-separated list of tickers from user input."""

    if not raw_tickers:
        return []
    return [token.strip().upper() for token in raw_tickers.split(",") if token.strip()]


def available_sectors() -> Iterable[str]:
    return (
        "Technology",
        "Healthcare",
        "Financials",
        "Energy",
        "Utilities",
        "Consumer",
    )


def available_regions() -> Iterable[str]:
    return ("United States", "Canada", "United Kingdom", "Germany", "Japan")
