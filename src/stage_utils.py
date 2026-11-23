"""Helpers shared by the LLM stages."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Iterator, List, Sequence, TypeVar

from config import canonical_ticker

T = TypeVar("T")


def candidate_tickers(item: dict, universe: Iterable[str]) -> List[str]:
    """Universe tickers tagged on the article (alias-aware, sorted)."""
    wanted = {canonical_ticker(t) for t in universe}
    return sorted({canonical_ticker(t) for t in item.get("tickers", [])} & wanted)


def input_hash(item: dict) -> str:
    """Hash of the article text the LLM saw; changes if a new version is observed."""
    blob = json.dumps({"headline": item.get("headline"), "summary": item.get("summary")}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chunks(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def article_view(item: dict) -> dict:
    """What the LLM sees. Dates, URLs and sources are deliberately omitted: they add no
    classification signal and could cue the model to recall what happened next."""
    return {"article_id": item["id"], "headline": item.get("headline") or "",
            "summary": (item.get("summary") or "").strip()}


def expect_pairs(rows, expected, label: str):
    """Raise LLMValidationError unless `rows` covers each expected (article_id, ticker) exactly once."""
    from src.llm_client import LLMValidationError
    got = [(r["article_id"], r["ticker"]) for r in rows]
    dupes = sorted({p for p in got if got.count(p) > 1})
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    problems = []
    if missing: problems.append(f"missing {missing[:5]}")
    if extra: problems.append(f"unexpected {extra[:5]}")
    if dupes: problems.append(f"duplicated {dupes[:5]}")
    if problems:
        raise LLMValidationError(f"{label} must contain exactly one entry per requested "
                                 f"(article_id, ticker): " + "; ".join(problems))
