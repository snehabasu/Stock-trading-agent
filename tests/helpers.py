import hashlib
from src.time_utils import now_utc

def make_news(article_id,published_at,updated_at=None,tickers=None,headline=None):
    headline=headline or article_id
    return {"id":article_id,"tickers":tickers or ["AAPL"],"headline":headline,
      "summary":"test summary","source":"test","url":None,"published_at":published_at,
      "updated_at":updated_at,"ingested_at":now_utc(),
      "version_hash":hashlib.sha256(headline.encode()).hexdigest(),"raw":{}}
