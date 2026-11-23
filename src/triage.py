"""Triage stage (Gemini 2.5 Flash-Lite): is this article relevant to this universe ticker?

Only (article, ticker) pairs where Alpaca tagged a universe ticker are sent; an article
tagged with no universe ticker is irrelevant by construction and costs no call.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

from config import TICKERS, TRIAGE_BATCH_SIZE, TRIAGE_MODEL, TRIAGE_THINKING_BUDGET
from src.llm_client import LLMClient, LLMRequest
from src.stage_utils import article_view, candidate_tickers, chunks, expect_pairs, input_hash

TRIAGE_PROMPT_VERSION = "triage-v1"

TRIAGE_SYSTEM = """You triage financial news for an equity trading system.
For every (article, ticker) pair you are given, decide whether the article is RELEVANT to that ticker.

RELEVANT: the article carries company-specific information a trader could expect to move that
company's stock. The company is the main subject or a principal party, e.g. its earnings or
guidance; an analyst rating / price-target change on it; a merger, acquisition or divestiture it is
party to; regulatory, legal or government action aimed at it; its product launches, recalls, trial
results or approvals; executive changes; major contracts or partnerships; dividends or buybacks.

NOT RELEVANT: the ticker appears only in a list (market wraps, top movers, "stocks to watch",
ETF/fund holdings); the company is mentioned in passing, as a peer or comparison; automated data
recaps with no new company information (unusual options activity, short-interest tables, technical
levels, 13F holdings changes); broad market or macro news that does not single the company out.

Judge each pair independently using only the headline and summary. Do not use outside knowledge
of what happened later. Keep "reason" under 20 words."""

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"},
                    "ticker": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["article_id", "ticker", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_triage_request(batch: Sequence[dict], universe: Sequence[str],
                         model: str = TRIAGE_MODEL) -> LLMRequest:
    articles = []
    for item in batch:
        view = article_view(item)
        view["all_tagged_tickers"] = sorted(item.get("tickers", []))
        view["tickers_to_judge"] = candidate_tickers(item, universe)
        articles.append(view)
    prompt = ("Return one result per (article_id, ticker in tickers_to_judge).\n"
              "all_tagged_tickers shows every symbol the news vendor tagged; a long list usually "
              "means a roundup.\n\nARTICLES:\n" + json.dumps(articles, indent=1, ensure_ascii=False))
    return LLMRequest(model=model, system=TRIAGE_SYSTEM, prompt=prompt, schema=TRIAGE_SCHEMA,
                      temperature=0.0, thinking_budget=TRIAGE_THINKING_BUDGET)


def triage(llm: LLMClient, items: Sequence[dict], universe: Sequence[str] = TICKERS,
           model: str = TRIAGE_MODEL, batch_size: int = TRIAGE_BATCH_SIZE) -> List[Dict]:
    """Return one record per (article, universe ticker) pair, in input order."""
    work = [it for it in items if candidate_tickers(it, universe)]
    by_id = {it["id"]: it for it in work}
    out: List[Dict] = []
    for batch in chunks(work, batch_size):
        request = build_triage_request(batch, universe, model)
        expected = [(it["id"], t) for it in batch for t in candidate_tickers(it, universe)]
        result = llm.generate_json(request, stage="triage",
                                   validator=lambda d, e=expected: expect_pairs(d["results"], e, "results"))
        decided = {(r["article_id"], r["ticker"]): r for r in result.data["results"]}
        for article_id, ticker in expected:
            r = decided[(article_id, ticker)]
            out.append({"article_id": article_id, "ticker": ticker, "relevant": bool(r["relevant"]),
                        "reason": r["reason"], "model": model, "model_version": result.model_version,
                        "prompt_version": TRIAGE_PROMPT_VERSION, "prompt_hash": result.prompt_hash,
                        "input_hash": input_hash(by_id[article_id]), "cached": result.cached})
    return out
