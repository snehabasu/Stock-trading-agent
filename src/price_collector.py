from config import BARS_PAGE_LIMIT, PRICE_ADJUSTMENT, TIMEFRAME
from src.time_utils import now_utc

def fetch_bars(client,symbols,start,end,feed="iex"):
    params={"symbols":",".join(symbols),"timeframe":TIMEFRAME,"start":start,"end":end,
      "limit":BARS_PAGE_LIMIT,"adjustment":PRICE_ADJUSTMENT,"feed":feed,"sort":"asc"}
    while True:
        payload=client.get("/v2/stocks/bars",params)
        for ticker,bars in payload.get("bars",{}).items():
            for b in bars:
                yield {"ticker":ticker,"timestamp":b["t"],"timeframe":TIMEFRAME,
                  "adjustment":PRICE_ADJUSTMENT,"feed":feed,"open":b.get("o"),"high":b.get("h"),
                  "low":b.get("l"),"close":b.get("c"),"volume":b.get("v"),
                  "trade_count":b.get("n"),"vwap":b.get("vw"),"ingested_at":now_utc()}
        token=payload.get("next_page_token")
        if not token: break
        params["page_token"]=token

def collect_bars(db,client,symbols,start,end,feed="iex"):
    count=0
    for bar in fetch_bars(client,symbols,start,end,feed):
        db.upsert_bar(bar); count+=1
    return count,count
