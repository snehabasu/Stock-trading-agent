import argparse, os
from datetime import datetime, timezone, date
from pathlib import Path
from dotenv import load_dotenv
from config import TICKERS, DATE_SPLITS, load_settings
from src.database import Database
from src.alpaca_client import AlpacaClient
from src.news_collector import collect_news
from src.price_collector import collect_bars
from src.coverage_report import print_coverage_report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["init-db","backfill","update","coverage"])
    p.add_argument("--start",default=DATE_SPLITS["development"]["start"])
    p.add_argument("--end",default=DATE_SPLITS["test"]["end"])
    a=p.parse_args(); load_dotenv()
    if a.command=="init-db":
        db=Database(Path(os.getenv("DATABASE_PATH","data/market_data.sqlite"))); db.initialize()
        print(f"Initialized {db.path}"); return
    s=load_settings(); db=Database(s["database_path"]); db.initialize()
    if a.command=="coverage": print_coverage_report(db); return
    client=AlpacaClient(s["api_key"],s["api_secret"],s["data_url"])
    start=a.start
    end=date.today().isoformat() if a.command=="update" else a.end
    jobs=[("news",lambda:collect_news(db,client,TICKERS,start,end)),
          ("bars",lambda:collect_bars(db,client,TICKERS,start,end,s["stock_feed"]))]
    for kind,job in jobs:
        started=datetime.now(timezone.utc).isoformat()
        try:
            seen,inserted=job()
            db.log_run(kind,started,datetime.now(timezone.utc).isoformat(),seen,inserted,"success")
            print(f"{kind}: seen={seen}, inserted={inserted}")
        except Exception as e:
            db.log_run(kind,started,datetime.now(timezone.utc).isoformat(),0,0,"failed",str(e)); raise

if __name__=="__main__": main()
