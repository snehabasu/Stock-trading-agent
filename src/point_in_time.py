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
        symbols=json.loads(r["tickers_json"])
        if ticker not in symbols: continue
        out.append({"id":r["id"],"tickers":symbols,"headline":r["headline"],
          "summary":r["summary"],"source":r["source"],"url":r["url"],
          "published_at":r["published_at"],"updated_at":r["updated_at"],"ingested_at":r["ingested_at"]})
        if len(out)>=limit: break
    return out
