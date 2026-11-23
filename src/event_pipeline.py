"""News window -> triage -> event extraction -> stored Event JSON.

A "window" is [start, end). Every article that first becomes visible to get_news during the
window is processed together, and its events get available_at = end. Use get_events(as_of)
in the backtest: it only returns events with available_at <= as_of, so batching a window
into one LLM call never lets an article's classification be used before every article in
that batch was public.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Sequence

from config import EXTRACTOR_MODEL, TICKERS, TRIAGE_MODEL
from src.event_extractor import EVENT_FIELDS, EXTRACTOR_PROMPT_VERSION, extract_events
from src.llm_client import LLMClient
from src.point_in_time import _cutoff, get_news_window
from src.stage_utils import canonical_ticker
from src.time_utils import now_utc
from src.triage import TRIAGE_PROMPT_VERSION, triage


def utc_day_window(day: date):
    start = datetime.combine(day, time(0), tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def run_window(db, llm: LLMClient, start, end, universe: Sequence[str] = TICKERS,
               triage_model: str = TRIAGE_MODEL, extractor_model: str = EXTRACTOR_MODEL,
               require_ingested_before: bool = False) -> Dict:
    start_s, end_s = _cutoff(start), _cutoff(end)
    items = get_news_window(db, start_s, end_s, require_ingested_before=require_ingested_before)
    by_id = {it["id"]: it for it in items}

    triaged = triage(llm, items, universe, model=triage_model)
    _store_triage(db, triaged, start_s, end_s)

    relevant = [(by_id[t["article_id"]], t["ticker"]) for t in triaged if t["relevant"]]
    events = extract_events(llm, relevant, model=extractor_model)
    _store_events(db, events, end_s)

    return {"window_start": start_s, "window_end": end_s, "articles": len(items),
            "pairs_triaged": len(triaged), "pairs_relevant": len(relevant),
            "events": [{k: e[k] for k in EVENT_FIELDS} for e in events],
            "cache_hits": llm.cache_hits, "api_calls": llm.api_calls}


def _store_triage(db, rows, start_s, end_s):
    created = now_utc()
    with db.connect() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO triage_results (article_id,ticker,relevant,reason,model,
               model_version,prompt_version,prompt_hash,input_hash,window_start,window_end,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(r["article_id"], r["ticker"], int(r["relevant"]), r["reason"], r["model"],
              r["model_version"], r["prompt_version"], r["prompt_hash"], r["input_hash"],
              start_s, end_s, created) for r in rows])


def _store_events(db, events, available_at):
    created = now_utc()
    with db.connect() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO news_events (article_id,ticker,event_type,direction,horizon,
               confidence,model,model_version,prompt_version,prompt_hash,input_hash,available_at,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(e["article_id"], e["ticker"], e["event_type"], e["direction"], e["horizon"],
              e["confidence"], e["model"], e["model_version"], e["prompt_version"], e["prompt_hash"],
              e["input_hash"], available_at, created) for e in events])


def get_events(db, ticker: str, as_of, model: str = EXTRACTOR_MODEL,
               prompt_version: str = EXTRACTOR_PROMPT_VERSION, since=None) -> List[Dict]:
    """Point-in-time Event JSON for one ticker: only events with available_at <= as_of
    (optionally available_at > since), newest first."""
    sql = """SELECT e.* FROM news_events e JOIN news_articles a ON a.id=e.article_id
             WHERE e.ticker=? AND e.model=? AND e.prompt_version=? AND e.available_at<=?"""
    args = [canonical_ticker(ticker), model, prompt_version, _cutoff(as_of)]
    if since is not None:
        sql += " AND e.available_at>?"; args.append(_cutoff(since))
    sql += " ORDER BY e.available_at DESC, a.published_at DESC"
    with db.connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [{k: r[k] for k in EVENT_FIELDS} for r in rows]

