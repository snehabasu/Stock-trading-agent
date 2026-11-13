# Alpaca News Collection System

Python + SQLite ingestion for Alpaca news and daily adjusted price bars.

## Reproducible 15-stock sample
`AMGN, AVGO, BRK-B, CAT, COST, JNJ, LLY, MA, META, MSFT, NVDA, ORCL, PEP, QCOM, WFC`. Selected using `random.Random(42).sample()` from the curated candidate list in `config.py`. The candidate list is a documented sampling frame, not an independently verified index constituent list.

## Provisional splits
Development 2025-03-01–2025-06-30; validation 2025-07-01–2025-09-30; test 2025-10-01–2026-11-09. Confirm all dates are after Gemini's knowledge cutoff and freeze the same splits team-wide.

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
python -m src.cli backfill --start 2025-03-01 --end 2026-11-09
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
Bars use `adjustment=all` (split/dividend adjusted); do not adjust again. IEX is default; SIP may require subscription access. Ingestion does not classify earnings, sentiment, relevance, or event type; Sneha's LLM does that.
