"""Hand-labelled triage evaluation.

1. sample_for_labeling() writes a CSV of ~50 (article, ticker) pairs with an empty `relevant`
   column. Label it by hand BEFORE looking at model output, using the definition in
   docs/triage_labeling_guide.md (same definition as the triage prompt).
2. evaluate_triage() runs triage on exactly those articles (cached) and reports precision,
   recall and F1 for the positive class "relevant".
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from config import RANDOM_SEED, TICKERS, TRIAGE_MODEL
from src.point_in_time import _row_to_item, get_news_window
from src.stage_utils import candidate_tickers
from src.triage import triage

LABEL_COLUMNS = ["article_id", "ticker", "published_at", "all_tagged_tickers", "headline",
                 "summary", "relevant", "notes"]
_TRUE, _FALSE = {"1", "y", "yes", "true", "t"}, {"0", "n", "no", "false", "f"}


def sample_for_labeling(db, start, end, n: int = 50, seed: int = RANDOM_SEED,
                        universe: Sequence[str] = TICKERS) -> List[Dict]:
    """One pair per article, round-robin across tickers so every ticker is represented."""
    rng = random.Random(seed)
    per_ticker = defaultdict(list)
    for item in get_news_window(db, start, end):
        for t in candidate_tickers(item, universe):
            per_ticker[t].append(item)
    for t in per_ticker:
        rng.shuffle(per_ticker[t])
    order = sorted(per_ticker)
    rng.shuffle(order)
    used, rows = set(), []
    while len(rows) < n and any(per_ticker[t] for t in order):
        for t in order:
            while per_ticker[t] and per_ticker[t][-1]["id"] in used:
                per_ticker[t].pop()
            if not per_ticker[t] or len(rows) >= n:
                continue
            item = per_ticker[t].pop()
            used.add(item["id"])
            rows.append({"article_id": item["id"], "ticker": t, "published_at": item["published_at"],
                         "all_tagged_tickers": " ".join(item["tickers"]), "headline": item["headline"],
                         "summary": item.get("summary") or "", "relevant": "", "notes": ""})
    return rows


def write_label_csv(rows: List[Dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def read_labels(path) -> Dict[tuple, bool]:
    labels = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            value = (row.get("relevant") or "").strip().lower()
            if not value:
                continue
            if value not in _TRUE | _FALSE:
                raise ValueError(f"{path}:{i}: relevant must be 1/0 or yes/no, got {value!r}")
            labels[(row["article_id"], row["ticker"])] = value in _TRUE
    return labels


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def score(labels: Dict[tuple, bool], predictions: Dict[tuple, bool]) -> Dict:
    tp = fp = fn = tn = 0
    disagreements = []
    for pair, truth in sorted(labels.items()):
        pred = predictions[pair]
        if pred and truth: tp += 1
        elif pred and not truth: fp += 1; disagreements.append({"pair": pair, "label": truth, "model": pred})
        elif not pred and truth: fn += 1; disagreements.append({"pair": pair, "label": truth, "model": pred})
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {"n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "precision_95ci": wilson_interval(tp, tp + fp),
            "recall": recall, "recall_95ci": wilson_interval(tp, tp + fn),
            "f1": f1, "accuracy": (tp + tn) / len(labels) if labels else None,
            "disagreements": disagreements}


def evaluate_triage(db, llm, labels_path, model: str = TRIAGE_MODEL) -> Dict:
    labels = read_labels(labels_path)
    if not labels:
        raise ValueError(f"No labelled rows in {labels_path}; fill the 'relevant' column first.")
    ids = sorted({a for a, _ in labels})
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM news_articles WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    items = sorted((_row_to_item(r) for r in rows), key=lambda x: (x["published_at"], x["id"]))
    missing = set(ids) - {it["id"] for it in items}
    if missing:
        raise ValueError(f"Labelled articles not in database: {sorted(missing)[:5]}")
    universe = sorted({t for _, t in labels})
    results = triage(llm, items, universe, model=model)
    predictions = {(r["article_id"], r["ticker"]): r["relevant"] for r in results}
    reasons = {(r["article_id"], r["ticker"]): r["reason"] for r in results}
    report = score(labels, predictions)
    for d in report["disagreements"]:
        d["model_reason"] = reasons[d["pair"]]
    report["model"] = model
    return report


def format_report(report: Dict) -> str:
    def pct(x): return "n/a" if x is None else f"{x:.1%}"
    lines = [f"Triage vs hand labels ({report['model']}), n={report['n']}",
             f"  TP={report['tp']} FP={report['fp']} FN={report['fn']} TN={report['tn']}",
             f"  precision {pct(report['precision'])}  95% CI {report['precision_95ci']}",
             f"  recall    {pct(report['recall'])}  95% CI {report['recall_95ci']}",
             f"  F1 {pct(report['f1'])}   accuracy {pct(report['accuracy'])}"]
    if report["disagreements"]:
        lines.append("  Disagreements (label vs model):")
        for d in report["disagreements"]:
            lines.append(f"    {d['pair'][0]} {d['pair'][1]}: label={int(d['label'])} "
                         f"model={int(d['model'])} — {d['model_reason']}")
    return "\n".join(lines)
