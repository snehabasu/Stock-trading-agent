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

## Decision agent (Task 3)
Each trading day D the agent decides at **D 00:00 UTC** (after the previous close; orders fill at D's open):
```
events (available_at <= as_of, last 3 days) + price features (bars < as_of) + current paper portfolio
+ risk profile + thesis memory (this run's past decisions, as_of < now)
  -> draft Proposal -> what-if tool (trades, turnover, cost, ex-ante vol, limit breaches)
  -> review round -> deterministic enforcement of hard limits -> Proposal JSON + decision_log row
```
Proposal (`schemas/proposal.schema.json`):
`{"as_of", "run", "hold", "weights": {ticker: w}, "cash", "rationale", "cited_article_ids"}`. Long-only; `hold=true` keeps the drifted portfolio (allowed up to 2 points past a limit, trimmed beyond that).

```bash
python -m src.cli run-range --start 2026-07-29 --end 2026-09-29     # events needed by the validation window
python -m src.cli decide --start 2026-08-01 --end 2026-09-30        # news, named, moderate, all 15
python -m src.cli decide --start 2026-08-01 --end 2026-09-30 --no-news   # same LLM without news (baseline)
python -m src.cli decide --start 2026-08-01 --end 2026-09-30 --mask      # anonymized companies (H2)
python -m src.cli decide --start 2026-08-01 --end 2026-09-30 --universe tech      # or non_tech
python -m src.cli audit-decisions --start 2026-08-01 --end 2026-09-30   # same flags select the run
python -m src.cli evaluate-runs --start 2026-08-01 --end 2026-09-30 \
  --run tech-news-named-moderate-decide-v2 --run tech-nonews-named-moderate-decide-v2   # score runs
```
- **Runs** are named `<universe>-<news|nonews>-<named|masked>-<profile>-<prompt version>`; memory never crosses runs. Proposals are exported to `data/decisions/<run>.jsonl`.
- **No-news mode** removes `news_events` from the prompt and requires an empty citation list.
- **Masking** maps tickers, company names, well-known executives and flagship brands to stable IDs (`CO01`..`CO15`, `COxx-exec`, `COxx-brand`; seeded shuffle) in everything the model sees, including the what-if output, and maps the answer back. Price levels are never shown (returns/volatility only) so they cannot identify a company. Extend `src/masking.py: ALIASES` if review finds identifying words leaking through.
- **Universe groups** (`config.UNIVERSE_GROUPS`): `tech` = AVGO, META, MSFT, NVDA, ORCL, QCOM; `non_tech` = the other 9. META (GICS Communication Services) is grouped with tech as a heavily covered platform company.
- **Prompt versions:** `decide-v1` parked ~65% in cash and held on 40 of 42 validation days (audit PASS, but useless as a strategy). `decide-v2` (default) adds a benchmark mandate: start from equal weight in the run's universe ((1 - min_cash)/N per stock) and tilt on evidence; cash above `min_cash` is an active bet that needs a reason. Funding a new all-cash portfolio is exempt from `max_turnover` (the limit governs rebalancing); every other limit still applies.
- **Scoring** (`src/evaluate_runs.py`): return, Sharpe, max drawdown, beta/alpha and information ratio against the run's own equal-weight buy-and-hold benchmark (bought at the first execution open with the same cost), plus the news effect = news minus no-news daily returns with a t-statistic. Judge tech vs non-tech on excess/alpha and on the news effect, not raw return (raw return mostly measures the sector). A quick read only; ~40 days is short.
- **Validation results** (2026-08-03..09-30, 42 days, gemini-3.5-flash-lite, `decide-v2`, moderate; all audits PASS; `data/decisions/`):

  | run | return | benchmark | excess | Sharpe | invested | holds |
  |---|---|---|---|---|---|---|
  | tech, news | +10.49% | +12.34% | -1.85% | 2.69 | 90% | 14% |
  | tech, no news | +10.65% | +12.34% | -1.69% | 2.79 | 88% | 67% |
  | non_tech, news | -2.62% | -1.90% | -0.72% | -1.67 | 86% | 29% |
  | non_tech, no news | -3.63% | -1.90% | -1.72% | -2.09 | 90% | 83% |

  News effect (news minus no-news): tech -0.16% (t = -0.08), non_tech +1.01% (t = +0.47); neither is significant over 42 days. Most of the negative excess is the 10% `min_cash` against a fully invested benchmark. Known issue: the v2 prompt does not tell the model the initial deployment is exempt from `max_turnover`, so non_tech-news ramped in over 6 days (both non_tech runs earned about the same over that stretch); fix in the next prompt version before the test-split run.
- **Tuning on validation only:** `decide` refuses any range overlapping the test split unless `--allow-test` is passed.
- **Audit** checks every trading day has a schema-valid Proposal within limits, and every cited article was shown to the agent and was published (and any known update made) before the decision time.
- The paper tracker in `src/market_features.py` (overnight drift, fills at the open, 5 bps cost) only feeds the agent's portfolio and memory; Punit's backtester is the authoritative P&L.
