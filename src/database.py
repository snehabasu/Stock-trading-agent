import json, sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS news_articles (
 id TEXT PRIMARY KEY, tickers_json TEXT NOT NULL, headline TEXT NOT NULL,
 summary TEXT, source TEXT, url TEXT, published_at TEXT NOT NULL,
 updated_at TEXT, ingested_at TEXT NOT NULL, raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE TABLE IF NOT EXISTS news_article_versions (
 version_id INTEGER PRIMARY KEY AUTOINCREMENT,
 article_id TEXT NOT NULL REFERENCES news_articles(id),
 version_hash TEXT NOT NULL, headline TEXT NOT NULL, summary TEXT,
 updated_at TEXT, observed_at TEXT NOT NULL, raw_json TEXT,
 UNIQUE(article_id,version_hash,observed_at)
);
CREATE INDEX IF NOT EXISTS idx_versions_article ON news_article_versions(article_id,observed_at);
CREATE TABLE IF NOT EXISTS price_bars (
 ticker TEXT NOT NULL, timestamp TEXT NOT NULL, timeframe TEXT NOT NULL,
 adjustment TEXT NOT NULL, feed TEXT NOT NULL, open REAL, high REAL, low REAL,
 close REAL, volume REAL, trade_count INTEGER, vwap REAL, ingested_at TEXT NOT NULL,
 PRIMARY KEY(ticker,timestamp,timeframe,adjustment,feed)
);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_time ON price_bars(ticker,timestamp);
CREATE TABLE IF NOT EXISTS ingestion_runs (
 run_id INTEGER PRIMARY KEY AUTOINCREMENT, data_type TEXT NOT NULL,
 started_at TEXT NOT NULL, finished_at TEXT, records_seen INTEGER DEFAULT 0,
 records_inserted INTEGER DEFAULT 0, status TEXT NOT NULL, error TEXT
);
"""

class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    def initialize(self):
        with self.connect() as conn: conn.executescript(SCHEMA)
    def upsert_news(self, item):
        raw = json.dumps(item.get("raw", {}), default=str)
        symbols = json.dumps(sorted(set(item["tickers"])))
        with self.connect() as conn:
            cur = conn.execute("""INSERT OR IGNORE INTO news_articles
              (id,tickers_json,headline,summary,source,url,published_at,updated_at,ingested_at,raw_json)
              VALUES (?,?,?,?,?,?,?,?,?,?)""",
              (item["id"],symbols,item["headline"],item.get("summary"),item.get("source"),
               item.get("url"),item["published_at"],item.get("updated_at"),item["ingested_at"],raw))
            conn.execute("""UPDATE news_articles SET tickers_json=?,headline=?,summary=?,source=?,
              url=?,updated_at=?,ingested_at=?,raw_json=? WHERE id=?""",
              (symbols,item["headline"],item.get("summary"),item.get("source"),item.get("url"),
               item.get("updated_at"),item["ingested_at"],raw,item["id"]))
            conn.execute("""INSERT OR IGNORE INTO news_article_versions
              (article_id,version_hash,headline,summary,updated_at,observed_at,raw_json)
              VALUES (?,?,?,?,?,?,?)""",
              (item["id"],item["version_hash"],item["headline"],item.get("summary"),
               item.get("updated_at"),item["ingested_at"],raw))
            return cur.rowcount == 1
    def upsert_bar(self, b):
        with self.connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO price_bars
              (ticker,timestamp,timeframe,adjustment,feed,open,high,low,close,volume,trade_count,vwap,ingested_at)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              tuple(b.get(k) for k in ("ticker","timestamp","timeframe","adjustment","feed","open",
               "high","low","close","volume","trade_count","vwap","ingested_at")))
    def log_run(self, kind, started, finished, seen, inserted, status, error=None):
        with self.connect() as conn:
            conn.execute("""INSERT INTO ingestion_runs
              (data_type,started_at,finished_at,records_seen,records_inserted,status,error)
              VALUES (?,?,?,?,?,?,?)""",(kind,started,finished,seen,inserted,status,error))
