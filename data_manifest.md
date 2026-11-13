# Data Manifest

- Provider: Alpaca only.
- News: `GET /v1beta1/news`.
- Prices: `GET /v2/stocks/bars`, daily timeframe.
- Universe: reproducible seed-42 sample: AMGN, AVGO, BRK-B, CAT, COST, JNJ, LLY, MA, META, MSFT, NVDA, ORCL, PEP, QCOM, WFC.
- Splits (provisional): development 2025-03-01–2025-06-30; validation 2025-07-01–2025-09-30; test 2025-10-01–2026-11-09. Confirm dates postdate Gemini knowledge cutoff.
- News timestamps: publication (`created_at` from Alpaca), publisher update (`updated_at` if supplied), and ingestion (`ingested_at`) stored separately; UTC.
- Deduplication: Alpaca article ID; versions observed by this collector retained in `news_article_versions`.
- Prices: `adjustment=all`; store feed with each bar to avoid mixing IEX/SIP silently.
- Point-in-time: require `published_at < decision_time`; if `updated_at` exists, require `updated_at <= decision_time`. For prospective replay additionally require `ingested_at <= decision_time`.
- Historical limitation: Alpaca historical news is not a full revision archive; exact historical article text may not be reconstructible from backfill.
- Coverage gaps: plan/feed entitlements can limit history; missing article does not prove no event occurred. Run `python -m src.cli coverage`.
- No prefiltering or LLM classification during ingestion; all available ticker-linked news is retained.
