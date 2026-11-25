import argparse, json, os
from datetime import datetime, timezone, date
from pathlib import Path
from dotenv import load_dotenv
from config import TICKERS, DATE_SPLITS, load_settings, load_llm_settings
from src.database import Database

INGEST = ["init-db", "backfill", "update", "coverage"]
LLM = ["run-day", "run-window", "run-range", "label-sample", "label-eval", "estimate-calls", "cache-stats",
       "decide", "audit-decisions", "evaluate-runs"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=INGEST+LLM)
    p.add_argument("--start",default=None)
    p.add_argument("--end",default=None)
    p.add_argument("--date",help="run-day: UTC calendar day YYYY-MM-DD")
    p.add_argument("--out",help="run-day/run-window/label-sample: output path")
    p.add_argument("--labels",default="data/labels/triage_labels.csv")
    p.add_argument("--n",type=int,default=50)
    p.add_argument("--triage-version",default=None,help="triage prompt version, e.g. triage-v1 (default: latest)")
    p.add_argument("--samples",type=int,default=None,help="self-consistency samples for triage (default: config)")
    p.add_argument("--no-news",action="store_true",help="decide: same LLM without news (baseline / remove-news test)")
    p.add_argument("--mask",action="store_true",help="decide: anonymize companies and tickers (H2)")
    p.add_argument("--profile",default=None,help="decide: risk profile (conservative|moderate|aggressive)")
    p.add_argument("--universe",default=None,help="decide: universe group (all|tech|non_tech)")
    p.add_argument("--allow-test",action="store_true",help="decide: allow test-split dates (final evaluation only)")
    p.add_argument("--run",default=None,action="append",
                   help="audit-decisions/evaluate-runs: run name (repeatable for evaluate-runs; default: from the flags above)")
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
    if a.command=="evaluate-runs":
        from src.evaluate_runs import evaluate, format_evaluation
        print(format_evaluation(evaluate(db,a.run or [decision_config(a).run_name],a.start,a.end))); return
    from src.llm_client import ResponseCache, build_llm_client
    s=load_llm_settings()
    if a.command=="cache-stats":
        for row in ResponseCache(s["cache_path"]).stats(): print(row)
        return
    if a.command=="audit-decisions":
        from src.decision_audit import audit_run, format_audit
        report=audit_run(db,(a.run or [decision_config(a).run_name])[0],a.start,a.end)
        print(format_audit(report)); raise SystemExit(1 if report["problems"] else 0)
    llm=build_llm_client(s)
    if a.command=="decide":
        return run_decide(a, db, llm)
    if a.command=="run-range":
        return run_range(a, db, llm)
    if a.command=="label-eval":
        from src.labeling import evaluate_triage, format_report
        kw={"version":a.triage_version} if a.triage_version else {}
        if a.samples: kw["samples"]=a.samples
        print(format_report(evaluate_triage(db,llm,a.labels,**kw))); return
    from src.event_pipeline import run_window, utc_day_window
    if a.command=="run-day":
        if not a.date: raise SystemExit("run-day needs --date YYYY-MM-DD")
        start,end=utc_day_window(date.fromisoformat(a.date))
    else:
        if not (a.start and a.end): raise SystemExit("run-window needs --start and --end (ISO, UTC)")
        start,end=a.start,a.end
    result=run_window(db,llm,start,end,require_ingested_before=a.live,triage_version=a.triage_version,
                      triage_samples=a.samples)
    text=json.dumps(result["events"],indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(text+"\n")
    else:
        print(text)
    print(f"window {result['window_start']} -> {result['window_end']}: {result['articles']} articles, "
          f"{result['pairs_triaged']} pairs triaged, {result['pairs_relevant']} relevant, "
          f"{len(result['events'])} events; API calls={result['api_calls']}, cache hits={result['cache_hits']}",
          flush=True, file=__import__("sys").stderr)

def decision_config(a):
    from dataclasses import replace
    from src.decision_agent import DecisionConfig
    cfg=DecisionConfig()
    over={}
    if a.no_news: over["use_news"]=False
    if a.mask: over["mask"]=True
    if a.profile: over["profile"]=a.profile
    if a.universe: over["universe"]=a.universe
    return replace(cfg,**over)

def run_decide(a, db, llm):
    import sys
    from src.decision_agent import run_decisions
    if not (a.start and a.end): raise SystemExit("decide needs --start and --end (YYYY-MM-DD)")
    cfg=decision_config(a)
    print(f"run {cfg.run_name}",file=sys.stderr)
    def show(day,r):
        p=r["proposal"]
        held=", ".join(f"{t} {w:.0%}" for t,w in p["weights"].items()) or "all cash"
        print(f"{day} {'HOLD' if p['hold'] else 'TRADE'} cash {p['cash']:.0%} | {held} | cites {len(p['cited_article_ids'])}"
              f"{' | adjusted' if r['adjustments'] else ''} | calls={r['api_calls']} hits={r['cache_hits']}",flush=True)
    run_decisions(db,llm,cfg,a.start,a.end,allow_test=a.allow_test,on_day=show,day_retries=DAY_RETRIES,
                  day_wait_seconds=DAY_WAIT_SECONDS)

DAY_RETRIES, DAY_WAIT_SECONDS = 5, 120  # run-range/decide: extra patience for Gemini 503 spells

def run_range(a, db, llm):
    import sys
    from datetime import date, timedelta
    from src.event_pipeline import run_window, utc_day_window
    import time
    from src.llm_client import QuotaExhaustedError, RetriesExhaustedError
    if not (a.start and a.end): raise SystemExit("run-range needs --start and --end (YYYY-MM-DD)")
    d,end=date.fromisoformat(a.start),date.fromisoformat(a.end)
    while d<=end:
        for attempt in range(1,DAY_RETRIES+1):
            try:
                r=run_window(db,llm,*utc_day_window(d),require_ingested_before=a.live,
                             triage_version=a.triage_version,triage_samples=a.samples)
                break
            except QuotaExhaustedError as e:
                raise SystemExit(f"{d}: daily quota exhausted; rerun the same command after the reset ({e})")
            except RetriesExhaustedError as e:
                if attempt==DAY_RETRIES: raise
                print(f"{d}: Gemini unavailable ({str(e)[-60:]}); attempt {attempt}/{DAY_RETRIES}, waiting {DAY_WAIT_SECONDS}s, "
                      f"finished batches stay cached",file=sys.stderr,flush=True)
                time.sleep(DAY_WAIT_SECONDS)
        print(f"{d}: {r['articles']} articles, {r['pairs_relevant']}/{r['pairs_triaged']} relevant, "
              f"{len(r['events'])} events | calls={r['api_calls']} hits={r['cache_hits']}",flush=True)
        d+=timedelta(days=1)

if __name__=="__main__": main()
