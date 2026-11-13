"""Shared team configuration; freeze splits and universe before backtesting."""
from pathlib import Path
import os
from dotenv import load_dotenv

LARGE_CAP_CANDIDATES = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM', 'WMT', 'XOM', 'V', 'MA', 'COST', 'ORCL', 'NFLX', 'JNJ', 'ABBV', 'HD', 'PG', 'BAC', 'KO', 'CRM', 'CVX', 'MRK', 'AMD', 'PEP', 'CSCO', 'ACN', 'MCD', 'LIN', 'TMO', 'ADBE', 'WFC', 'GS', 'CAT', 'IBM', 'GE', 'DIS', 'QCOM', 'TXN', 'INTU', 'AMGN', 'PM', 'UNH']
RANDOM_SEED = 42
TICKERS = ['AMGN', 'AVGO', 'BRK-B', 'CAT', 'COST', 'JNJ', 'LLY', 'MA', 'META', 'MSFT', 'NVDA', 'ORCL', 'PEP', 'QCOM', 'WFC']

# Provisional; confirm first date is after Gemini knowledge cutoff.
DATE_SPLITS = {
    "development": {"start": "2025-03-01", "end": "2025-06-30"},
    "validation": {"start": "2025-07-01", "end": "2025-09-30"},
    "test": {"start": "2025-10-01", "end": "2026-11-09"},
}
TIMEFRAME = "1Day"
PRICE_ADJUSTMENT = "all"
DEFAULT_STOCK_FEED = "iex"
NEWS_PAGE_LIMIT = 50
BARS_PAGE_LIMIT = 10000
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30

def load_settings():
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")
    return {
        "api_key": key, "api_secret": secret,
        "data_url": os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/"),
        "database_path": Path(os.getenv("DATABASE_PATH", "data/market_data.sqlite")),
        "stock_feed": os.getenv("ALPACA_STOCK_FEED", DEFAULT_STOCK_FEED),
    }
