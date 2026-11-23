"""Event extractor (Gemini 2.5 Flash): event type, direction, horizon, confidence for
each (article, ticker) pair that triage marked relevant.

Output matches schemas/event.schema.json exactly:
{"article_id", "ticker", "event_type", "direction", "horizon", "confidence"}
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

from config import EXTRACTOR_BATCH_SIZE, EXTRACTOR_MODEL, EXTRACTOR_THINKING_BUDGET
from src.llm_client import LLMClient, LLMRequest
from src.stage_utils import article_view, chunks, expect_pairs, input_hash

EXTRACTOR_PROMPT_VERSION = "extract-v1"

EVENT_TYPES = ["earnings", "guidance", "analyst_action", "merger_acquisition", "regulatory",
               "legal", "product", "management", "capital_return", "partnership_contract",
               "macro_sector", "other"]
DIRECTIONS = ["positive", "negative", "neutral"]
HORIZONS = ["short_term", "medium_term", "long_term"]
EVENT_FIELDS = ["article_id", "ticker", "event_type", "direction", "horizon", "confidence"]

EXTRACTOR_SYSTEM = f"""You extract structured trading events from financial news.
For each (article, ticker) item, classify the news FROM THAT TICKER'S POINT OF VIEW
(in an acquisition the acquirer and the target can have different directions).

event_type, pick the single best fit:
- earnings: reported quarterly/annual results
- guidance: company outlook or forecast changes (including pre-announcements)
- analyst_action: rating, price-target or coverage change by an analyst or bank
- merger_acquisition: mergers, acquisitions, divestitures, spin-offs, stake purchases
- regulatory: government/regulator actions, approvals, antitrust, export controls, tariffs aimed at the company
- legal: lawsuits, settlements, investigations, court rulings
- product: launches, recalls, clinical trial results, drug approvals, outages
- management: executive or board changes
- capital_return: dividends, buybacks, share issuance, debt offerings
- partnership_contract: major customers, contracts, partnerships, supply deals
- macro_sector: sector/macro news with a direct, specific effect on this company
- other: none of the above

direction: expected effect on THIS company's stock price: positive, negative, or neutral
(neutral = no clear effect, mixed news, or already fully expected).

horizon: how long the effect is likely to matter:
- short_term: days, up to about two weeks (e.g. earnings surprise, rating change)
- medium_term: weeks to about one quarter
- long_term: beyond a quarter (structural, e.g. major acquisition, new business line)

confidence: your probability (0.0-1.0) that the direction label is correct. Use lower values for
ambiguous, speculative or minor news.

Use only the headline and summary. Do not use outside knowledge of what happened later."""

EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"},
                    "ticker": {"type": "string"},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "direction": {"type": "string", "enum": DIRECTIONS},
                    "horizon": {"type": "string", "enum": HORIZONS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": EVENT_FIELDS,
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

# Contract for a single Event (also written to schemas/event.schema.json).
EVENT_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Event",
                **EXTRACTOR_SCHEMA["properties"]["events"]["items"]}


def build_extractor_request(batch: Sequence[Tuple[dict, str]],
                            model: str = EXTRACTOR_MODEL) -> LLMRequest:
    items = []
    for item, ticker in batch:
        view = article_view(item)
        view["ticker"] = ticker
        items.append(view)
    prompt = ("Return one event per item, echoing article_id and ticker exactly.\n\nITEMS:\n"
              + json.dumps(items, indent=1, ensure_ascii=False))
    return LLMRequest(model=model, system=EXTRACTOR_SYSTEM, prompt=prompt, schema=EXTRACTOR_SCHEMA,
                      temperature=0.0, thinking_budget=EXTRACTOR_THINKING_BUDGET)


def extract_events(llm: LLMClient, pairs: Sequence[Tuple[dict, str]], model: str = EXTRACTOR_MODEL,
                   batch_size: int = EXTRACTOR_BATCH_SIZE) -> List[Dict]:
    """pairs: (NewsItem, ticker) for relevant pairs. Returns Event records + provenance
    fields (model, prompt_hash, ...); use to_event_json() for the bare contract."""
    out: List[Dict] = []
    for batch in chunks(list(pairs), batch_size):
        request = build_extractor_request(batch, model)
        expected = [(item["id"], ticker) for item, ticker in batch]
        result = llm.generate_json(request, stage="extract",
                                   validator=lambda d, e=expected: expect_pairs(d["events"], e, "events"))
        got = {(e["article_id"], e["ticker"]): e for e in result.data["events"]}
        for item, ticker in batch:
            ev = to_event_json(got[(item["id"], ticker)])
            ev.update({"model": model, "model_version": result.model_version,
                       "prompt_version": EXTRACTOR_PROMPT_VERSION, "prompt_hash": result.prompt_hash,
                       "input_hash": input_hash(item), "cached": result.cached})
            out.append(ev)
    return out


def to_event_json(record: Dict) -> Dict:
    """Strip provenance; return exactly the Event contract fields in contract order."""
    ev = {k: record[k] for k in EVENT_FIELDS}
    ev["confidence"] = round(float(ev["confidence"]), 4)
    return ev
