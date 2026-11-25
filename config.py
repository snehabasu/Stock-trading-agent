"""Shared team configuration; freeze splits and universe before backtesting."""
from pathlib import Path
import os
from dotenv import load_dotenv

LARGE_CAP_CANDIDATES = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM', 'WMT', 'XOM', 'V', 'MA', 'COST', 'ORCL', 'NFLX', 'JNJ', 'ABBV', 'HD', 'PG', 'BAC', 'KO', 'CRM', 'CVX', 'MRK', 'AMD', 'PEP', 'CSCO', 'ACN', 'MCD', 'LIN', 'TMO', 'ADBE', 'WFC', 'GS', 'CAT', 'IBM', 'GE', 'DIS', 'QCOM', 'TXN', 'INTU', 'AMGN', 'PM', 'UNH']
RANDOM_SEED = 42
TICKERS = ['AMGN', 'AVGO', 'BRK-B', 'CAT', 'COST', 'JNJ', 'LLY', 'MA', 'META', 'MSFT', 'NVDA', 'ORCL', 'PEP', 'QCOM', 'WFC']

# FROZEN 2026-11-23 (team sign-off). Do not change without team agreement. Alpaca news (source fiscal.ai) only exists
# from 2026-04-29, and the LLM (gemini-3.5-flash-lite) has a March 2026 knowledge cutoff,
# so every split starts after both.
DATE_SPLITS = {
    "development": {"start": "2026-05-01", "end": "2026-07-31"},
    "validation": {"start": "2026-08-01", "end": "2026-09-30"},
    "test": {"start": "2026-10-01", "end": "2026-11-20"},
}
TIMEFRAME = "1Day"
PRICE_ADJUSTMENT = "all"
DEFAULT_STOCK_FEED = "iex"
NEWS_PAGE_LIMIT = 50
BARS_PAGE_LIMIT = 10000
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30

# --- LLM stages (Task 2) -------------------------------------------------------
# gemini-3.5-flash-lite (checked 2026-11-23): stable, no shutdown date, free tier free of
# charge, knowledge cutoff March 2026 (some domains January 2025) -> before every split.
# Gemini 2.5 is closed to new API users, and 2.5 Flash reroutes to 3.5 Flash-Lite from
# 2026-12-01. LLMClient refuses any response whose model_version differs from the request.
TRIAGE_MODEL = "gemini-3.5-flash-lite"
EXTRACTOR_MODEL = "gemini-3.5-flash-lite"
# 3.5 Flash-Lite rejects thinking_budget=0 (400); None = model default (no thought tokens
# observed). Controlling Gemini 3 thinking_level needs google-genai > 1.47 / Python >= 3.10.
TRIAGE_THINKING_BUDGET = None
EXTRACTOR_THINKING_BUDGET = None
TRIAGE_BATCH_SIZE = 20          # articles per triage call
TRIAGE_RELEVANT_LEVELS = ("minor", "material")  # triage-v2 grades counted as relevant (drop only "none")
TRIAGE_SAMPLES = 1               # self-consistency samples per batch (1 = single greedy call)
TRIAGE_SAMPLE_TEMPERATURE = 1.0  # used only when TRIAGE_SAMPLES > 1
EXTRACTOR_BATCH_SIZE = 10       # (article, ticker) pairs per extraction call
LLM_MAX_ATTEMPTS = 6
LLM_BASE_DELAY_SECONDS = 5.0  # backoff 5, 10, 20, 40, 60s (Gemini 503 overloads come in spells)
LLM_TIMEOUT_SECONDS = 120
EXPERIMENT_CONDITIONS = 6

# Alpaca symbol spellings (verified 2026-11-23): bars accept BRK.B and reject BRK-B;
# Berkshire news is tagged BRK.A. Everything inside the project uses BRK-B.
ALPACA_BAR_SYMBOLS = {"BRK-B": ["BRK.B"]}
ALPACA_NEWS_SYMBOLS = {"BRK-B": ["BRK.A", "BRK.B"]}
TICKER_ALIASES = {"BRK.A": "BRK-B", "BRK.B": "BRK-B", "BRK/B": "BRK-B", "BRKB": "BRK-B"}

def alpaca_symbols(tickers, mapping):
    return [sym for t in tickers for sym in mapping.get(t, [t])]

def canonical_ticker(symbol):
    s = str(symbol).upper().strip()
    return TICKER_ALIASES.get(s, s)

# --- Decision agent (Task 3) -----------------------------------------------------
DECISION_MODEL = "gemini-3.5-flash-lite"
DECISION_PROMPT_VERSION = "decide-v2"
DECISION_USE_NEWS = True         # False = "same LLM without news" baseline / remove-news robustness test
DECISION_MASK_ENTITIES = False   # True = anonymize names/tickers as CO01..CO15 (H2)
DECISION_RISK_PROFILE = "moderate"
DECISION_UNIVERSE = "all"        # key of UNIVERSE_GROUPS
DECISION_EVENT_LOOKBACK_DAYS = 3
DECISION_MAX_EVENTS = 60
DECISION_MEMORY_SIZE = 5         # past decisions shown to the agent (thesis memory)
DECISION_PRICE_LOOKBACK_BARS = 60
DECISION_REVIEW = "always"       # what-if review round: "always" | "on_violation" | "never"
TRANSACTION_COST_BPS = 5.0       # per unit of traded notional, used by the what-if tool and tracker
# Long-only. min_cash caps gross exposure; max_turnover is one-way, as a fraction of NAV per day;
# vol_ceiling is annualized ex-ante volatility (a warning in the what-if tool, not enforced).
RISK_PROFILES = {
    "conservative": {"max_position": 0.10, "min_cash": 0.40, "max_turnover": 0.20, "vol_ceiling": 0.12},
    "moderate": {"max_position": 0.20, "min_cash": 0.10, "max_turnover": 0.35, "vol_ceiling": 0.20},
    "aggressive": {"max_position": 0.30, "min_cash": 0.00, "max_turnover": 0.60, "vol_ceiling": 0.30},
}
# Tech vs non-tech comparison on the frozen 15 (team choice 2026-11-23). META is GICS
# Communication Services but grouped with tech here as a heavily covered platform company;
# MA moved from IT to Financials in 2023 and is non-tech here.
UNIVERSE_GROUPS = {
    "all": TICKERS,
    "tech": ["AVGO", "META", "MSFT", "NVDA", "ORCL", "QCOM"],
    "non_tech": ["AMGN", "BRK-B", "CAT", "COST", "JNJ", "LLY", "MA", "PEP", "WFC"],
}
assert sorted(UNIVERSE_GROUPS["tech"] + UNIVERSE_GROUPS["non_tech"]) == sorted(TICKERS)

def load_llm_settings():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env")
    return {"api_key": key, "cache_path": Path(os.getenv("LLM_CACHE_PATH", "data/llm_cache.sqlite")),
            "max_attempts": LLM_MAX_ATTEMPTS, "timeout_seconds": LLM_TIMEOUT_SECONDS,
            "base_delay": LLM_BASE_DELAY_SECONDS}

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
