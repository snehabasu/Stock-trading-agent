# Triage labeling guide

Goal: ~50 hand labels to measure triage precision and recall. Label **before** looking at model output.

```bash
python -m src.cli label-sample            # writes data/labels/triage_labels.csv (dev split, seed 42)
# fill the `relevant` column with 1 or 0 (add a note for borderline cases)
python -m src.cli label-eval              # runs triage on those rows (cached), prints P/R/F1 + disagreements
```

Each row is one (article, ticker) pair, one ticker per article, balanced across the 15 tickers.
Sample from the **development** split only, so labels never tune decisions on validation/test.

## Label 1 (relevant) when the article carries company-specific information that could move *that* ticker
- earnings, guidance, pre-announcements
- analyst rating / price-target / coverage change on the company
- M&A, divestiture, spin-off, stake the company is party to
- regulatory, legal or government action aimed at the company (antitrust, export rules, FDA, lawsuits)
- product launch, recall, trial result, approval, outage
- executive or board change
- major contract, customer or partnership
- dividend, buyback, share or debt issuance

## Label 0 (not relevant)
- the ticker is one of many in a list: market wrap, top movers, "stocks to watch", ETF/fund holdings
- the company appears only in passing, or as a peer or comparison
- automated data recaps with no new company information: unusual options activity, short-interest tables, technical levels, 13F changes
- broad macro or market news that does not single the company out

Use only the headline and summary shown (that is all the model sees). With n≈50, expect 95% intervals of roughly ±10–15 points; `label-eval` prints Wilson intervals.
