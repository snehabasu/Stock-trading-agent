# Data Manifest

- Provider: Alpaca only.
- News: `GET /v1beta1/news`.
- Prices: `GET /v2/stocks/bars`, daily timeframe.
- Universe: reproducible seed-42 sample: AMGN, AVGO, BRK-B, CAT, COST, JNJ, LLY, MA, META, MSFT, NVDA, ORCL, PEP, QCOM, WFC.
- Splits (frozen 2026-11-23, team sign-off): development 2026-05-01–2026-07-31; validation 2026-08-01–2026-09-30; test 2026-10-01–2026-11-20. All after Alpaca's first available news (2026-04-29) and the LLM's March 2026 knowledge cutoff.
- News source: every article Alpaca returned was `fiscal.ai`; nothing before 2026-04-29 (checked 2026-11-23 with free paper keys).
- Symbols: Berkshire is `BRK-B` internally; Alpaca bars use `BRK.B`, Alpaca news tags `BRK.A`.
- News timestamps: publication (`created_at` from Alpaca), publisher update (`updated_at` if supplied), and ingestion (`ingested_at`) stored separately; UTC.
- Deduplication: Alpaca article ID; versions observed by this collector retained in `news_article_versions`.
- Prices: `adjustment=all`; store feed with each bar to avoid mixing IEX/SIP silently.
- Point-in-time: require `published_at < decision_time`; if `updated_at` exists, require `updated_at <= decision_time`. For prospective replay additionally require `ingested_at <= decision_time`.
- Historical limitation: Alpaca historical news is not a full revision archive; exact historical article text may not be reconstructible from backfill.
- Coverage gaps: plan/feed entitlements can limit history; missing article does not prove no event occurred. Run `python -m src.cli coverage`.
- No prefiltering or LLM classification during ingestion; all available ticker-linked news is retained.
