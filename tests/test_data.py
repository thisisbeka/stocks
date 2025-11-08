from datetime import date

from stocks_dashboard.data import PriceRequest, parse_tickers


def test_parse_tickers_strips_deduplicates_and_uppercases():
    result = parse_tickers(" aapl, msft , AAPL ,,, goog ")
    assert result == ["AAPL", "MSFT", "GOOG"]


def test_parse_tickers_handles_empty_input():
    assert parse_tickers("") == []


def test_price_request_normalized_tickers_preserves_order():
    request = PriceRequest(
        tickers=[" aapl", "MSFT", "AAPL", ""],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
    )

    assert request.normalized_tickers() == ["AAPL", "MSFT"]
