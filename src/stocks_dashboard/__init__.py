"""Stocks Dashboard package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("stocks_dashboard")
except PackageNotFoundError:  # pragma: no cover - best effort for runtime metadata
    __version__ = "0.0.0"
