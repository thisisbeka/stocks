# Stocks Dashboard

A clean and modular Streamlit dashboard for exploring Yahoo! Finance data, computing technical indicators, and performing minimum-variance portfolio optimisation.

## Features

- 📈 Interactive Streamlit UI with configurable instrument presets
- 🧮 Simple moving averages and linear regression trend modelling
- 💼 Minimum-variance portfolio optimisation with annualised metrics
- 🧪 Automated pytest suite that validates core analytics utilities
- 🧱 Modular project layout under `src/` for easy maintenance and extension

## Project layout

```
├── main.py                 # Streamlit entrypoint (`streamlit run main.py`)
├── requirements.txt        # Application and testing dependencies
├── src/stocks_dashboard    # Application package
│   ├── __init__.py
│   ├── analytics.py        # Indicators and forecasting helpers
│   ├── app.py              # Streamlit UI composition
│   ├── data.py             # Data fetching and metadata helpers
│   └── portfolio.py        # Portfolio optimisation logic
└── tests                   # Automated test suite (pytest)
    ├── test_analytics.py
    └── test_portfolio.py
```

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Run the automated tests**

   ```bash
   pytest
   ```

3. **Launch the dashboard**

   ```bash
   streamlit run main.py
   ```

   Streamlit will output a local URL (typically <http://localhost:8501>) where you can interact with the dashboard.

## Development tips

- The data download is cached via `st.cache_data`, so repeated requests with the same parameters are fast.
- Feel free to add additional indicators by extending `src/stocks_dashboard/analytics.py` and rendering the output in `app.py`.
- To ensure reproducible test runs, keep any random number generation seeded (see `tests/test_portfolio.py`).
- Ticker inputs are normalised and deduplicated before fetching prices, so you can paste messy comma-separated lists without
  worrying about duplicate requests.

## License


