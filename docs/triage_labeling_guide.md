# Triage labeling guide

Goal: ~50 hand labels to measure triage precision and recall. Label **before** looking at model output.

```bash
python -m src.cli label-sample            # writes data/labels/triage_labels.csv (dev split, seed 42)
# fill the `relevant` column with 1 or 0; use `notes` for borderline cases
python -m src.cli label-eval              # runs triage on those rows (cached), prints P/R/F1 + disagreements
```

Each row is one (article, ticker) pair, one ticker per article, spread across the 15 tickers.
The sample comes from the **development** split only, so labels never tune decisions on
validation or test. It skips articles that already have triage output, so you label blind.

## Definition: relevant = material

Label **1** only if the article contains new, company-specific information that a professional
trader would plausibly expect to **move this stock or change its earnings estimates**.

Ask: *"If I held this stock, would I want to read this before the next open?"*
If the honest answer is "no, it's noise", label 0, even if the article is about the company.

## Usually 1
- earnings, guidance, pre-announcements
- analyst rating or price-target change **on this company** (also "maintains rating, changes target")
- M&A, divestiture or spin-off where this company is the acquirer, target or seller of a significant business
- regulatory, legal or government action aimed at the company with real money or business at stake (antitrust, export controls, FDA decisions, large lawsuits/settlements, court rulings on key patents)
- safety signals, recalls or trial results for a significant product
- a major contract, customer or partnership that is large relative to the company
- CEO/CFO change; large buyback or new common dividend policy; large share or debt issuance
- a well-known investor disclosing a large new stake or activist campaign

## Usually 0
- the ticker is one of many in a list: market wraps, top movers, "stocks to watch", ETF/fund holdings, a third-party ETF launched on the stock
- the company appears only in passing, as a peer or comparison, or the vendor tagged the wrong company
- the company is a **service provider in someone else's deal**: a bank acting as advisor, underwriter, IPO co-manager or lender; a card network on another bank's card; a retailer stocking a third-party product
- routine, scheduled or tiny items: preferred-stock dividends, regular dividends declared at the usual amount, conference attendance or presentations, surveys and white papers, awards, local or regional marketing campaigns, minor product tweaks
- executives' or founders' personal or charitable activities
- automated data recaps with no new company information: unusual options activity, short-interest tables, technical levels, routine 13F changes
- broad macro or market news that does not single the company out

## Borderline or unsure
Write your best guess with a question mark: `1?` or `0?`, plus a short `notes` line if you can.
Don't leave a row blank because it's hard: skipping the hard rows makes triage look better than
it is. `label-eval` reports metrics with and without the `?` rows.

Quick tie-breakers:
- Is the company the **actor**, or just a vendor or service provider in someone else's story? (vendor → 0)
- Is it **new**, or scheduled and routine? (routine → 0)
- Is the money or business at stake **big relative to the company**? A $50M deal is noise for Microsoft.
- Would it make a **headline about this company** on a major financial news site? (yes → 1)

Don't look up how the stock moved afterwards. Labels must reflect what was knowable when the
article came out; using the realized price move would leak hindsight into the evaluation.

Judge from the headline and summary shown (that is all the model sees). With n≈50, expect 95%
intervals of roughly ±10–15 points; `label-eval` prints Wilson intervals.
