import json
from datetime import date
import jsonschema
import pytest
from src.database import Database
from src.event_extractor import EVENT_FIELDS
from src.event_pipeline import get_events, run_window, utc_day_window
from src.llm_client import LLMClient, LLMValidationError, ResponseCache
from src.stage_utils import candidate_tickers, expect_pairs
from src.triage import triage
from tests.fake_llm import FakeGemini
from tests.helpers import make_news

EVENT_SCHEMA = json.load(open("schemas/event.schema.json"))
DAY = date(2025, 10, 15)


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "m.sqlite"); db.initialize()
    db.upsert_news(make_news("a1", "2025-10-15T13:00:00Z", tickers=["MSFT"], headline="MSFT earnings beat"))
    db.upsert_news(make_news("a2", "2025-10-15T14:00:00Z", tickers=["NVDA", "AAPL"], headline="Analyst upgrade NVDA"))
    db.upsert_news(make_news("a3", "2025-10-15T15:00:00Z", tickers=["MSFT", "META", "TSLA"], headline="Market roundup"))
    db.upsert_news(make_news("a4", "2025-10-15T16:00:00Z", tickers=["TSLA"], headline="Not in universe"))
    db.upsert_news(make_news("a5", "2025-10-16T01:00:00Z", tickers=["MSFT"], headline="Next day earnings"))
    db.upsert_news(make_news("a6", "2025-10-15T17:00:00Z", tickers=["BRK.B"], headline="BRK earnings beat"))
    # published in-window but updated next day -> belongs to the next day's window
    db.upsert_news(make_news("a7", "2025-10-15T18:00:00Z", "2025-10-16T02:00:00Z", tickers=["COST"], headline="COST earnings"))
    return db


def make_llm(tmp_path, fake):
    return LLMClient(fake, ResponseCache(tmp_path / "cache.sqlite"), sleep=lambda s: None)


def test_day_produces_valid_event_json_and_rerun_is_cached(tmp_path, db):
    fake = FakeGemini()
    start, end = utc_day_window(DAY)
    first = run_window(db, make_llm(tmp_path, fake), start, end)
    calls_after_first = len(fake.calls)
    assert calls_after_first == 2  # one triage batch + one extraction batch

    events = first["events"]
    assert {(e["article_id"], e["ticker"]) for e in events} == {("a1", "MSFT"), ("a2", "NVDA"), ("a6", "BRK-B")}
    for e in events:
        jsonschema.validate(e, EVENT_SCHEMA)
        assert list(e) == EVENT_FIELDS
    assert first["pairs_triaged"] == 5  # a1 MSFT, a2 NVDA, a3 MSFT+META, a6 BRK-B; a4/a5/a7 excluded

    second = run_window(db, make_llm(tmp_path, fake), start, end)  # fresh client, same cache file
    assert len(fake.calls) == calls_after_first
    assert second["events"] == events and second["cache_hits"] == 2 and second["api_calls"] == 0


def test_article_id_passed_through_unchanged(tmp_path, db):
    start, end = utc_day_window(DAY)
    ids = {e["article_id"] for e in run_window(db, make_llm(tmp_path, FakeGemini()), start, end)["events"]}
    assert ids <= {"a1", "a2", "a6"}


def test_get_events_is_point_in_time(tmp_path, db):
    llm = make_llm(tmp_path, FakeGemini())
    run_window(db, llm, *utc_day_window(DAY))
    run_window(db, llm, *utc_day_window(date(2025, 10, 16)))
    # Mid-day 15th: nothing yet; the batch only becomes available at the window end.
    assert get_events(db, "MSFT", "2025-10-15T20:00:00Z") == []
    assert [e["article_id"] for e in get_events(db, "MSFT", "2025-10-16T00:00:00Z")] == ["a1"]
    assert [e["article_id"] for e in get_events(db, "MSFT", "2025-10-17T00:00:00Z")] == ["a5", "a1"]
    assert [e["article_id"] for e in get_events(db, "COST", "2025-10-17T00:00:00Z")] == ["a7"]
    assert get_events(db, "BRK.B", "2025-10-16T00:00:00Z")[0]["ticker"] == "BRK-B"


def test_window_with_no_universe_news_makes_no_calls(tmp_path, db):
    fake = FakeGemini()
    out = run_window(db, make_llm(tmp_path, fake), "2025-10-20T00:00:00Z", "2025-10-21T00:00:00Z")
    assert out["events"] == [] and fake.calls == []


def test_triage_batches_and_covers_every_pair(tmp_path, db):
    fake = FakeGemini()
    items = [make_news(f"x{i}", "2025-10-15T13:00:00Z", tickers=["MSFT", "NVDA"]) for i in range(5)]
    out = triage(make_llm(tmp_path, fake), items, batch_size=2)
    assert len(fake.calls) == 3 and len(out) == 10


def test_triage_rejects_missing_pairs(tmp_path):
    fake = FakeGemini(script=['{"results": []}'] * 4)
    from src.llm_client import LLMError
    with pytest.raises(LLMError, match="missing"):
        triage(make_llm(tmp_path, fake), [make_news("x", "2025-10-15T13:00:00Z", tickers=["MSFT"])])


def test_expect_pairs_flags_duplicates():
    with pytest.raises(LLMValidationError, match="duplicated"):
        expect_pairs([{"article_id": "a", "ticker": "T"}] * 2, [("a", "T")], "results")


def test_candidate_tickers_alias_aware():
    assert candidate_tickers({"tickers": ["brk.b", "AAPL", "MSFT"]}, ["BRK-B", "MSFT"]) == ["BRK-B", "MSFT"]
