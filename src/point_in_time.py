import json
from datetime import datetime, timezone

def _cutoff(as_of):
    if isinstance(as_of,str): as_of=datetime.fromisoformat(as_of.replace("Z","+00:00"))
    if as_of.tzinfo is None: raise ValueError("as_of must be timezone-aware")
    return as_of.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def get_news(db,ticker,as_of,limit=500,require_ingested_before=False):
    """Strictly exclude future publications and known later edits.

    Set require_ingested_before=True for prospective/live replay. Leave False
    for historical backfills, whose ingestion necessarily occurs after publication.
    """
    cutoff=_cutoff(as_of); ticker=ticker.upper().strip()
    sql="""SELECT id,tickers_json,headline,summary,source,url,published_at,updated_at,ingested_at
      FROM news_articles WHERE published_at < ?
      AND (updated_at IS NULL OR updated_at <= ?)"""
    args=[cutoff,cutoff]
    if require_ingested_before:
        sql+=" AND ingested_at <= ?"; args.append(cutoff)
    sql+=" ORDER BY published_at DESC"
    with db.connect() as conn: rows=conn.execute(sql,args).fetchall()
    out=[]
    for r in rows:
        item=_row_to_item(r)
        if ticker not in item["tickers"]: continue
        out.append(item)
        if len(out)>=limit: break
    return out

def _row_to_item(r):
    return {"id":r["id"],"tickers":json.loads(r["tickers_json"]),"headline":r["headline"],
      "summary":r["summary"],"source":r["source"],"url":r["url"],
      "published_at":r["published_at"],"updated_at":r["updated_at"],"ingested_at":r["ingested_at"]}

# An article becomes visible to get_news at the later of published_at / updated_at.
_VISIBLE_AT="CASE WHEN updated_at IS NOT NULL AND updated_at>published_at THEN updated_at ELSE published_at END"

def get_news_window(db,start,end,require_ingested_before=False):
    """All articles (any ticker) that FIRST become visible to get_news(..., as_of=end)
    during [start, end). Every returned item is visible as of `end` under the same rules
    as get_news, and each article falls in exactly one window of a contiguous series,
    so batching a window's articles into one LLM call cannot leak later news."""
    start,end=_cutoff(start),_cutoff(end)
    sql=f"""SELECT id,tickers_json,headline,summary,source,url,published_at,updated_at,ingested_at
      FROM news_articles WHERE published_at < ? AND (updated_at IS NULL OR updated_at <= ?)
      AND {_VISIBLE_AT} >= ? AND {_VISIBLE_AT} < ?"""
    args=[end,end,start,end]
    if require_ingested_before:
        sql+=" AND ingested_at <= ?"; args.append(end)
    sql+=" ORDER BY published_at, id"
    with db.connect() as conn: rows=conn.execute(sql,args).fetchall()
    return [_row_to_item(r) for r in rows]
