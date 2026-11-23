# Alpaca News Collection System

Python + SQLite ingestion for Alpaca news and daily adjusted price bars.

## Reproducible 15-stock sample
`AMGN, AVGO, BRK-B, CAT, COST, JNJ, LLY, MA, META, MSFT, NVDA, ORCL, PEP, QCOM, WFC`. Selected using `random.Random(42).sample()` from the curated candidate list in `config.py`. The candidate list is a documented sampling frame, not an independently verified index constituent list.

## Frozen splits (team sign-off 2026-11-23)
Development 2026-05-01–2026-07-31; validation 2026-08-01–2026-09-30; test 2026-10-01–2026-11-20.
Why: Alpaca's news API (source `fiscal.ai`) returns nothing before 2026-04-29 (checked 2026-11-23; ~5,300 articles for the 15 tickers through 2026-11-22, ~650–950/month), and `gemini-3.5-flash-lite` has a March 2026 knowledge cutoff. Every split starts after both. Freeze the same splits team-wide.

## Install and run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set Alpaca keys in .env; never commit .env
python -m src.cli init-db
pytest -q
python -m src.cli backfill --start 2026-04-29 --end 2026-11-21
python -m src.cli coverage
```
For daily updates, pass a narrow start date: `python -m src.cli update --start YYYY-MM-DD`.

## Sneha's interface
```python
from datetime import datetime, timezone
from src.database import Database
from src.point_in_time import get_news

db = Database("data/market_data.sqlite")
items = get_news(db, "AAPL", datetime(2025,10,15,20,0,tzinfo=timezone.utc))
```
Keep each returned `id` unchanged as `article_id` in Event JSON.

For prospective/live replay, use `require_ingested_before=True`. For historical backfills, leave it False because ingestion necessarily occurs after the original publication.

## Point-in-time limitation
The query excludes `published_at >= as_of` and known `updated_at > as_of`. Alpaca's historical endpoint is not a full historical revision archive. A backfilled article may contain text edited after its original publication. This code preserves versions it observes from collection onward, but cannot recreate revisions never captured. Automated tests prove future publication timestamps and known future updates are excluded, not exact historical text reconstruction.

## Prices and scope
Bars use `adjustment=all` (split/dividend adjusted); do not adjust again. Berkshire is `BRK-B` everywhere in the project; the collectors request `BRK.B` bars and `BRK.A`/`BRK.B` news from Alpaca and map them back. IEX is default; SIP may require subscription access. Ingestion does not classify earnings, sentiment, relevance, or event type; the LLM stages below do that.

## LLM stages: triage + event extraction (Task 2)
```
get_news_window(day) -> triage (gemini-3.5-flash-lite) -> relevant (article, ticker) pairs
                     -> event extractor (gemini-3.5-flash-lite) -> Event JSON (schemas/event.schema.json)
```
All calls go through `src/llm_client.py`: a SQLite response cache keyed by a SHA-256 hash of the full request (`LLM_CACHE_PATH`), bounded retries with backoff, JSON-schema validation plus a check that each input (article, ticker) pair is answered exactly once, and a **model guard** that rejects any response whose reported `model_version` is not the requested model.

```bash
# add GEMINI_API_KEY to .env
python -m src.cli run-day --date 2025-10-15 --out data/events/2025-10-15.json   # rerun = 0 API calls
python -m src.cli estimate-calls --triage-rpd 20 --extract-rpd 20               # uses DB volume once backfilled
python -m src.cli cache-stats
```

Event contract: `{"article_id","ticker","event_type","direction","horizon","confidence"}`.
`event_type` ∈ earnings, guidance, analyst_action, merger_acquisition, regulatory, legal, product, management, capital_return, partnership_contract, macro_sector, other. `direction` ∈ positive, negative, neutral. `horizon` ∈ short_term (≤~2 weeks), medium_term (≤~1 quarter), long_term. `confidence` = model's probability that `direction` is right.

**Backtest consumption (Punit):** use `src.event_pipeline.get_events(db, ticker, as_of)`. A UTC-day window's articles are classified in shared batches, so their events carry `available_at` = window end and are hidden until then. Articles enter the window in which they first become visible to `get_news` (later of `published_at`/`updated_at`).

**Model facts (checked 2026-11-23):** `gemini-3.5-flash-lite` is stable with no announced shutdown, free of charge on the free tier, knowledge cutoff March 2026 (the model card says some domains stop at January 2025). Gemini 2.5 returns 404 for new API users, and `gemini-2.5-flash` auto-routes to 3.5 Flash-Lite from 2026-12-01, so 2.5 is not an option. The model guard rejects any response from a model other than the one requested. 3.5 Flash-Lite rejects `thinking_budget=0`, so the default thinking setting is used. Free-tier rate limits are only shown in your AI Studio project.

Triage evaluation: see `docs/triage_labeling_guide.md`. Default prompt is `triage-v2`, which grades each pair none / minor / material; pairs graded minor or material go to extraction (`TRIAGE_RELEVANT_LEVELS`).
```bash
python -m src.cli label-eval                            # triage-v2, one greedy call per batch
python -m src.cli label-eval --triage-version triage-v1 # yes/no baseline
python -m src.cli label-eval --samples 5                # self-consistency: 5 seeded samples, median vote
```
Results on the first 27 hand labels (dev split): v1 precision 52% / recall 92%; v2 57% / 100% (AUC 0.72); v2 with 5-sample self-consistency 55% / 100% (AUC 0.71). The 5 samples agreed unanimously on 23 of 27 articles, so the errors are systematic (where the minor/material line falls) rather than sampling noise; the pipeline therefore uses one call per batch (`TRIAGE_SAMPLES = 1`).
