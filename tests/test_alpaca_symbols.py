from config import ALPACA_BAR_SYMBOLS, ALPACA_NEWS_SYMBOLS, TICKERS, alpaca_symbols
from src.news_collector import normalize_article


def test_requests_use_alpaca_spellings():
    bars, news = alpaca_symbols(TICKERS, ALPACA_BAR_SYMBOLS), alpaca_symbols(TICKERS, ALPACA_NEWS_SYMBOLS)
    assert "BRK.B" in bars and "BRK-B" not in bars
    assert {"BRK.A", "BRK.B"} <= set(news) and "BRK-B" not in news
    assert len(bars) == 15 and len(news) == 16


def test_news_tickers_mapped_back_to_project_spelling():
    a = normalize_article({"id": 1, "symbols": ["BRK.A", "brk.b", "MSFT"], "headline": "h",
                           "created_at": "2026-05-01T12:00:00Z"})
    assert a["tickers"] == ["BRK-B", "MSFT"]
