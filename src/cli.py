import argparse, json, os
from datetime import datetime, timezone, date
from pathlib import Path
from dotenv import load_dotenv
from config import TICKERS, DATE_SPLITS, load_settings, load_llm_settings
from src.database import Database

INGEST = ["init-db", "backfill", "update", "coverage"]
LLM = ["run-day", "run-window", "label-sample", "label-eval", "estimate-calls", "cache-stats"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=INGEST+LLM)
    p.add_argument("--start",default=None)
    p.add_argument("--end",default=None)
    p.add_argument("--date",help="run-day: UTC calendar day YYYY-MM-DD")
    p.add_argument("--out",help="run-day/run-window/label-sample: output path")
    p.add_argument("--labels",default="data/labels/triage_labels.csv")
    p.add_argument("--n",type=int,default=50)
    p.add_argument("--live",action="store_true",help="also require ingested_at <= window end")
    p.add_argument("--relevance-rate",type=float,default=0.5)
    p.add_argument("--articles-per-day",type=float,default=None)
    p.add_argument("--triage-rpd",type=int,default=None)
    p.add_argument("--extract-rpd",type=int,default=None)
    a=p.parse_args(); load_dotenv()
    db=Database(Path(os.getenv("DATABASE_PATH","data/market_data.sqlite"))); db.initialize()
    if a.command=="init-db": print(f"Initialized {db.path}"); return
    if a.command in INGEST: return run_ingest(a, db)
    return run_llm(a, db)

def run_ingest(a, db):
    from src.alpaca_client import AlpacaClient
    from src.news_collector import collect_news
    from src.price_collector import collect_bars
    from src.coverage_report import print_coverage_report
    s=load_settings()
    if a.command=="coverage": print_coverage_report(db); return
    client=AlpacaClient(s["api_key"],s["api_secret"],s["data_url"])
    start=a.start or DATE_SPLITS["development"]["start"]
    end=date.today().isoformat() if a.command=="update" else (a.end or DATE_SPLITS["test"]["end"])
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

def run_llm(a, db):
    if a.command=="estimate-calls":
        from src.call_estimate import estimate, format_estimate
        r=estimate(db,relevance_rate=a.relevance_rate,articles_per_day=a.articles_per_day)
        print(format_estimate(r,a.triage_rpd,a.extract_rpd)); return
    if a.command=="label-sample":
        from src.labeling import sample_for_labeling, write_label_csv
        start=a.start or DATE_SPLITS["development"]["start"]
        end=a.end or DATE_SPLITS["development"]["end"]
        rows=sample_for_labeling(db,f"{start}T00:00:00Z",f"{end}T23:59:59Z",n=a.n)
        out=Path(a.out or a.labels)
        if out.exists(): raise SystemExit(f"{out} exists; refusing to overwrite labels (pass --out)")
        write_label_csv(rows,out); print(f"Wrote {len(rows)} pairs to {out}; fill the 'relevant' column (1/0)"); return
    from src.llm_client import ResponseCache, build_llm_client
    s=load_llm_settings()
    if a.command=="cache-stats":
        for row in ResponseCache(s["cache_path"]).stats(): print(row)
        return
    llm=build_llm_client(s)
    if a.command=="label-eval":
        from src.labeling import evaluate_triage, format_report
        print(format_report(evaluate_triage(db,llm,a.labels))); return
    from src.event_pipeline import run_window, utc_day_window
    if a.command=="run-day":
        if not a.date: raise SystemExit("run-day needs --date YYYY-MM-DD")
        start,end=utc_day_window(date.fromisoformat(a.date))
    else:
        if not (a.start and a.end): raise SystemExit("run-window needs --start and --end (ISO, UTC)")
        start,end=a.start,a.end
    result=run_window(db,llm,start,end,require_ingested_before=a.live)
    text=json.dumps(result["events"],indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(text+"\n")
    else:
        print(text)
    print(f"window {result['window_start']} -> {result['window_end']}: {result['articles']} articles, "
          f"{result['pairs_triaged']} pairs triaged, {result['pairs_relevant']} relevant, "
          f"{len(result['events'])} events; API calls={result['api_calls']}, cache hits={result['cache_hits']}",
          flush=True, file=__import__("sys").stderr)

if __name__=="__main__": main()
