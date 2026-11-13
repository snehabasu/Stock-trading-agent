import json

def build_coverage_report(db):
    report=[]
    with db.connect() as conn:
        tickers=[r["ticker"] for r in conn.execute("SELECT DISTINCT ticker FROM price_bars ORDER BY ticker")]
        for ticker in tickers:
            b=conn.execute("SELECT COUNT(*) n,MIN(timestamp) first_ts,MAX(timestamp) last_ts FROM price_bars WHERE ticker=?",(ticker,)).fetchone()
            rows=conn.execute("SELECT tickers_json,published_at FROM news_articles").fetchall()
            dates=[r["published_at"] for r in rows if ticker in json.loads(r["tickers_json"])]
            report.append({"ticker":ticker,"bar_count":b["n"],"first_bar":b["first_ts"],"last_bar":b["last_ts"],
              "news_count":len(dates),"first_news":min(dates) if dates else None,"last_news":max(dates) if dates else None})
    return report

def print_coverage_report(db):
    for row in build_coverage_report(db): print(row)
