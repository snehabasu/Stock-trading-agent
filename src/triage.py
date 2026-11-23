"""Triage stage: is this article relevant (material) to this universe ticker?

Only (article, ticker) pairs where the vendor tagged a universe ticker are sent; an article
tagged with no universe ticker is irrelevant by construction and costs no call.

Prompt versions:
- triage-v1: yes/no relevance. Kept for comparison (precision ~52%, recall ~92% on the
  first 27 hand labels).
- triage-v2 (default): grades materiality none / minor / material; a pair is relevant when
  its grade is in TRIAGE_RELEVANT_LEVELS. Rules follow docs/triage_labeling_guide.md.

Self-consistency (samples > 1): the same batch is sampled k times at temperature T with
seeds 0..k-1 (each seed is its own cache entry, so reruns are free and reproducible). The
final grade is the median of the k ordinal grades; `agreement` is the share of samples that
agree with the final relevant/not-relevant decision and `score` is the mean grade scaled to
[0, 1] (v1: share of samples voting relevant), usable for ranking metrics and calibration.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from config import (TICKERS, TRIAGE_BATCH_SIZE, TRIAGE_MODEL, TRIAGE_RELEVANT_LEVELS,
                    TRIAGE_SAMPLE_TEMPERATURE, TRIAGE_SAMPLES, TRIAGE_THINKING_BUDGET)
from src.llm_client import LLMClient, LLMRequest
from src.stage_utils import article_view, candidate_tickers, chunks, expect_pairs, input_hash

TRIAGE_PROMPT_VERSION = "triage-v2"
MATERIALITY_LEVELS = ["none", "minor", "material"]

_SYSTEM_V1 = """You triage financial news for an equity trading system.
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

_SCHEMA_V1 = {
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

_SYSTEM_V2 = """You triage financial news for an equity trading system.
For every (article, ticker) pair, grade how MATERIAL the article is for THAT ticker's stock.

material: new, company-specific information that a professional investor holding this stock would
plausibly expect to move its price or change its earnings estimates. The company must be the main
actor or directly affected, and the stakes must be significant relative to the company's size.
Typical: earnings; guidance; an analyst rating or price-target change on this company; mergers,
acquisitions or divestitures of significant size; regulatory, legal or government action with real
money or business at stake; important product, clinical-trial, approval, safety or recall news;
large contracts, customers or partnerships; CEO or CFO changes; major buyback, dividend-policy or
financing decisions; large stake changes by the company's controlling or best-known shareholders.

minor: genuinely about the company, but unlikely to matter for the stock. Routine or scheduled
items (regular or preferred dividends at the usual rate, conference or trade-show appearances,
event scheduling); orders, contracts, investments or portfolio moves that are small relative to
the company; awards and rankings; marketing campaigns; local or regional launches; incremental
product, feature or app updates; small insider transactions; subsidiary updates with no clear
financial impact.

none: not really about this company's business. The ticker is one of many in a list or roundup;
it is mentioned in passing or as a comparison; the company is only a vendor, supplier, platform or
service provider in another company's story (a bank acting as advisor, underwriter or lender; a
retailer stocking someone else's product; another firm's product built on its components); the
vendor tagged the wrong company; executives' personal or charitable activities; automated data
recaps (options activity, short interest, technical levels, routine holdings filings); macro news
that does not single the company out.

Scale matters: what is material for a small company can be minor for a mega-cap.
Judge each pair independently from the headline and summary only. Do not use outside knowledge of
what happened later. Write "reason" first (under 20 words), then choose "materiality"."""

_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"},
                    "ticker": {"type": "string"},
                    "reason": {"type": "string"},
                    "materiality": {"type": "string", "enum": MATERIALITY_LEVELS},
                },
                "required": ["article_id", "ticker", "reason", "materiality"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

PROMPTS = {"triage-v1": (_SYSTEM_V1, _SCHEMA_V1), "triage-v2": (_SYSTEM_V2, _SCHEMA_V2)}
# Back-compat names for the default version.
TRIAGE_SYSTEM, TRIAGE_SCHEMA = PROMPTS[TRIAGE_PROMPT_VERSION]


GRADE_VALUE = {g: i for i, g in enumerate(MATERIALITY_LEVELS)}


def build_triage_request(batch: Sequence[dict], universe: Sequence[str],
                         model: str = TRIAGE_MODEL, version: str = TRIAGE_PROMPT_VERSION,
                         temperature: float = 0.0, seed: Optional[int] = None) -> LLMRequest:
    system, schema = PROMPTS[version]
    articles = []
    for item in batch:
        view = article_view(item)
        view["all_tagged_tickers"] = sorted(item.get("tickers", []))
        view["tickers_to_judge"] = candidate_tickers(item, universe)
        articles.append(view)
    prompt = ("Return one result per (article_id, ticker in tickers_to_judge).\n"
              "all_tagged_tickers shows every symbol the news vendor tagged; a long list usually "
              "means a roundup.\n\nARTICLES:\n" + json.dumps(articles, indent=1, ensure_ascii=False))
    return LLMRequest(model=model, system=system, prompt=prompt, schema=schema,
                      temperature=temperature, thinking_budget=TRIAGE_THINKING_BUDGET, seed=seed)


def _aggregate(answers: List[dict], relevant_levels: Sequence[str]) -> dict:
    """Combine k sampled answers for one (article, ticker) pair."""
    if answers[0].get("materiality") is None:  # v1: majority vote on yes/no
        votes = [bool(a["relevant"]) for a in answers]
        relevant = sum(votes) * 2 > len(votes)
        return {"materiality": None, "relevant": relevant, "votes": votes,
                "score": sum(votes) / len(votes), "p_relevant": sum(votes) / len(votes),
                "agreement": sum(v == relevant for v in votes) / len(votes),
                "reason": answers[0]["reason"]}
    grades = sorted((a["materiality"] for a in answers), key=GRADE_VALUE.get)
    median = grades[(len(grades) - 1) // 2]  # lower median for even k (conservative)
    relevant = median in relevant_levels
    reason = next(a["reason"] for a in answers if a["materiality"] == median)
    return {"materiality": median, "relevant": relevant, "votes": [a["materiality"] for a in answers],
            "score": sum(GRADE_VALUE[g] for g in grades) / (2 * len(grades)),
            "p_relevant": sum(g in relevant_levels for g in grades) / len(grades),
            "agreement": sum((g in relevant_levels) == relevant for g in grades) / len(grades),
            "reason": reason}


def triage(llm: LLMClient, items: Sequence[dict], universe: Sequence[str] = TICKERS,
           model: str = TRIAGE_MODEL, batch_size: int = TRIAGE_BATCH_SIZE,
           version: str = TRIAGE_PROMPT_VERSION,
           relevant_levels: Sequence[str] = TRIAGE_RELEVANT_LEVELS,
           samples: int = TRIAGE_SAMPLES, temperature: float = TRIAGE_SAMPLE_TEMPERATURE) -> List[Dict]:
    """Return one record per (article, universe ticker) pair, in input order.
    `materiality` is None for triage-v1, which only answers yes/no."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    work = [it for it in items if candidate_tickers(it, universe)]
    by_id = {it["id"]: it for it in work}
    label = version if samples == 1 else f"{version}+sc{samples}"
    out: List[Dict] = []
    for batch in chunks(work, batch_size):
        expected = [(it["id"], t) for it in batch for t in candidate_tickers(it, universe)]
        settings = [(0.0, None)] if samples == 1 else [(temperature, i) for i in range(samples)]
        answers, hashes, versions, cached = defaultdict(list), [], set(), []
        for temp, seed in settings:
            request = build_triage_request(batch, universe, model, version, temp, seed)
            result = llm.generate_json(request, stage="triage",
                                       validator=lambda d, e=expected: expect_pairs(d["results"], e, "results"))
            for r in result.data["results"]:
                answers[(r["article_id"], r["ticker"])].append(r)
            hashes.append(result.prompt_hash); versions.add(result.model_version); cached.append(result.cached)
        for article_id, ticker in expected:
            agg = _aggregate(answers[(article_id, ticker)], relevant_levels)
            out.append({"article_id": article_id, "ticker": ticker, **agg, "n_samples": samples,
                        "model": model, "model_version": ",".join(sorted(v for v in versions if v)),
                        "prompt_version": label, "prompt_hash": hashes[0] if samples == 1 else
                        hashlib.sha256("|".join(hashes).encode()).hexdigest(),
                        "input_hash": input_hash(by_id[article_id]), "cached": all(cached)})
    return out
