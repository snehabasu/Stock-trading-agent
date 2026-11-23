"""Estimate LLM call volume: days x conditions x stages, and what caching/batching does to it.

Naive formula (one call per day per condition per stage): days * conditions * stages.
Actual calls: triage and extraction prompts do not depend on the experimental condition, so
conditions 2..6 are served from the prompt-hash cache; per day the call count is set by
article volume and batch size instead.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta
from typing import Dict, Optional

from config import (DATE_SPLITS, EXPERIMENT_CONDITIONS, EXTRACTOR_BATCH_SIZE, TICKERS,
                    TRIAGE_BATCH_SIZE)
from src.point_in_time import get_news_window
from src.stage_utils import candidate_tickers


def split_days(splits=DATE_SPLITS, weekdays_only: bool = False) -> Dict[str, int]:
    out = {}
    for name, s in splits.items():
        d, end, n = date.fromisoformat(s["start"]), date.fromisoformat(s["end"]), 0
        while d <= end:
            n += (not weekdays_only) or d.weekday() < 5
            d += timedelta(days=1)
        out[name] = n
    return out


def estimate(db=None, conditions: int = EXPERIMENT_CONDITIONS, stages: int = 2,
             relevance_rate: float = 0.5, articles_per_day: Optional[float] = None,
             prompt_variants: int = 1, splits=DATE_SPLITS) -> Dict:
    days = split_days(splits)
    total_days = sum(days.values())
    report = {"days_by_split": days, "total_days": total_days, "conditions": conditions,
              "stages": stages, "naive_calls": total_days * conditions * stages}

    triage_calls = extract_calls = articles = pairs = 0
    source = "assumed"
    if db is not None:
        start = min(date.fromisoformat(s["start"]) for s in splits.values())
        end = max(date.fromisoformat(s["end"]) for s in splits.values()) + timedelta(days=1)
        per_day_articles, per_day_pairs = Counter(), Counter()
        for item in get_news_window(db, f"{start}T00:00:00Z", f"{end}T00:00:00Z"):
            c = candidate_tickers(item, TICKERS)
            if not c:
                continue
            day = max(item["published_at"], item["updated_at"] or "")[:10]  # UTC day it becomes visible
            per_day_articles[day] += 1
            per_day_pairs[day] += len(c)
        if per_day_articles:
            source = "database"
            for day, n in per_day_articles.items():
                triage_calls += math.ceil(n / TRIAGE_BATCH_SIZE)
                extract_calls += math.ceil(per_day_pairs[day] * relevance_rate / EXTRACTOR_BATCH_SIZE)
            articles, pairs = sum(per_day_articles.values()), sum(per_day_pairs.values())
    if source == "assumed":
        apd = articles_per_day if articles_per_day is not None else 40.0
        articles, pairs = apd * total_days, apd * 1.2 * total_days
        triage_calls = total_days * math.ceil(apd / TRIAGE_BATCH_SIZE)
        extract_calls = total_days * math.ceil(apd * 1.2 * relevance_rate / EXTRACTOR_BATCH_SIZE)
        report["assumed_articles_per_day"] = apd

    report.update({"volume_source": source, "articles": int(articles), "pairs": int(pairs),
                   "relevance_rate": relevance_rate, "prompt_variants": prompt_variants,
                   "triage_calls": triage_calls * prompt_variants,
                   "extract_calls": extract_calls * prompt_variants})
    report["cached_calls"] = report["triage_calls"] + report["extract_calls"]
    return report


def format_estimate(r: Dict, triage_rpd: Optional[int] = None, extract_rpd: Optional[int] = None) -> str:
    lines = [f"Days: {r['days_by_split']} = {r['total_days']} calendar days",
             f"Naive: {r['total_days']} days x {r['conditions']} conditions x {r['stages']} stages "
             f"= {r['naive_calls']:,} calls",
             f"With prompt-hash cache + batching ({r['volume_source']} volume: {r['articles']:,} articles, "
             f"{r['pairs']:,} ticker pairs, relevance {r['relevance_rate']:.0%}, "
             f"{r['prompt_variants']} prompt variant(s)):",
             f"  triage  {r['triage_calls']:,} calls (batch {TRIAGE_BATCH_SIZE} articles)",
             f"  extract {r['extract_calls']:,} calls (batch {EXTRACTOR_BATCH_SIZE} pairs)",
             f"  total   {r['cached_calls']:,} API calls; conditions 2-{r['conditions']} hit the cache"]
    if triage_rpd:
        lines.append(f"  triage at {triage_rpd} RPD: {math.ceil(r['triage_calls'] / triage_rpd)} days")
    if extract_rpd:
        lines.append(f"  extract at {extract_rpd} RPD: {math.ceil(r['extract_calls'] / extract_rpd)} days")
    return "\n".join(lines)
