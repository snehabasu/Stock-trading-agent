import hashlib, json
from src.time_utils import utc_iso, now_utc

def normalize_article(a):
    published = a.get("created_at") or a.get("published_at")
    if not published: raise ValueError("Alpaca article has no publication timestamp")
    headline = (a.get("headline") or "").strip()
    summary, url = a.get("summary"), a.get("url")
    article_id = str(a["id"]) if a.get("id") is not None else hashlib.sha256(
        f"{headline}|{utc_iso(published)}|{url or ''}".encode()).hexdigest()
    version_text = json.dumps({"headline":headline,"summary":summary,
       "updated_at":a.get("updated_at"),"url":url},sort_keys=True,default=str)
    return {"id":article_id,"tickers":sorted({str(x).upper() for x in a.get("symbols",[])}),
      "headline":headline,"summary":summary,"source":a.get("source") or "Alpaca","url":url,
      "published_at":utc_iso(published),"updated_at":utc_iso(a.get("updated_at")),
      "ingested_at":now_utc(),"version_hash":hashlib.sha256(version_text.encode()).hexdigest(),"raw":a}

def fetch_news(client, symbols, start, end, limit=50):
    params={"symbols":",".join(symbols),"start":start,"end":end,"limit":limit,
            "sort":"asc","include_content":"false"}
    while True:
        payload=client.get("/v1beta1/news",params)
        yield from (normalize_article(a) for a in payload.get("news",[]))
        token=payload.get("next_page_token")
        if not token: break
        params["page_token"]=token

def collect_news(db,client,symbols,start,end):
    seen=inserted=0
    for article in fetch_news(client,symbols,start,end):
        seen+=1
        inserted+=int(db.upsert_news(article))
    return seen,inserted
