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
from src.triage import TRIAGE_PROMPT_VERSION, triage

LABEL_COLUMNS = ["article_id", "ticker", "published_at", "all_tagged_tickers", "headline",
                 "summary", "relevant", "notes"]
_TRUE, _FALSE = {"1", "y", "yes", "true", "t"}, {"0", "n", "no", "false", "f"}


def already_triaged_ids(db) -> set:
    with db.connect() as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT article_id FROM triage_results")}


def sample_for_labeling(db, start, end, n: int = 50, seed: int = RANDOM_SEED,
                        universe: Sequence[str] = TICKERS, exclude_seen: bool = True) -> List[Dict]:
    """One pair per article, round-robin across tickers so every ticker is represented.
    exclude_seen skips articles that already have triage output, so labels stay blind."""
    rng = random.Random(seed)
    skip = already_triaged_ids(db) if exclude_seen else set()
    per_ticker = defaultdict(list)
    for item in get_news_window(db, start, end):
        if item["id"] in skip:
            continue
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


def read_labels(path, include_unsure: bool = True) -> Dict[tuple, bool]:
    """`relevant` is 1/0 (or yes/no). A trailing '?' (e.g. '1?') marks a best guess;
    include_unsure=False drops those rows. Blank rows are skipped."""
    labels = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            value = (row.get("relevant") or "").strip().lower()
            if not value:
                continue
            unsure = value.endswith("?")
            value = value.rstrip("?").strip()
            if value not in _TRUE | _FALSE:
                raise ValueError(f"{path}:{i}: relevant must be 1/0 (optionally 1?/0?), got {row['relevant']!r}")
            if unsure and not include_unsure:
                continue
            labels[(row["article_id"], row["ticker"])] = value in _TRUE
    return labels


def count_label_rows(path) -> Dict[str, int]:
    counts = {"rows": 0, "sure": 0, "unsure": 0, "blank": 0}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("relevant") or "").strip()
            counts["rows"] += 1
            counts["blank" if not v else "unsure" if v.endswith("?") else "sure"] += 1
    return counts


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def auc(labels: Dict[tuple, bool], scores: Dict[tuple, float]):
    """Mann-Whitney AUC (ties count 1/2): P(score of a material pair > score of a non-material one)."""
    pos = [scores[p] for p, y in labels.items() if y]
    neg = [scores[p] for p, y in labels.items() if not y]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def brier(labels: Dict[tuple, bool], probs: Dict[tuple, float]):
    return sum((probs[p] - float(y)) ** 2 for p, y in labels.items()) / len(labels) if labels else None


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


def evaluate_triage(db, llm, labels_path, model: str = TRIAGE_MODEL,
                    version: str = TRIAGE_PROMPT_VERSION, samples: int = 1, **triage_kw) -> Dict:
    labels = read_labels(labels_path)
    if not labels:
        raise ValueError(f"No labelled rows in {labels_path}; fill the 'relevant' column first.")
    sure = read_labels(labels_path, include_unsure=False)
    ids = sorted({a for a, _ in labels})
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM news_articles WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    items = sorted((_row_to_item(r) for r in rows), key=lambda x: (x["published_at"], x["id"]))
    missing = set(ids) - {it["id"] for it in items}
    if missing:
        raise ValueError(f"Labelled articles not in database: {sorted(missing)[:5]}")
    universe = sorted({t for _, t in labels})
    results = triage(llm, items, universe, model=model, version=version, samples=samples, **triage_kw)
    predictions = {(r["article_id"], r["ticker"]): r["relevant"] for r in results}
    reasons = {(r["article_id"], r["ticker"]): r["reason"] for r in results}
    report = score(labels, predictions)
    for d in report["disagreements"]:
        d["model_reason"] = reasons[d["pair"]]
        d["unsure"] = d["pair"] not in sure
    report["sure_only"] = score(sure, predictions) if sure else None
    report["counts"] = count_label_rows(labels_path)
    report["model"], report["version"], report["samples"] = model, version, samples
    by_pair = {(r["article_id"], r["ticker"]): r for r in results}
    report["auc"] = auc(labels, {p: by_pair[p]["score"] for p in labels})
    report["brier"] = brier(labels, {p: by_pair[p]["p_relevant"] for p in labels})
    if samples > 1:
        groups = {"unanimous": [p for p in labels if by_pair[p]["agreement"] == 1.0],
                  "split": [p for p in labels if by_pair[p]["agreement"] < 1.0]}
        report["by_agreement"] = {k: (len(v), sum(predictions[p] == labels[p] for p in v) / len(v) if v else None)
                                  for k, v in groups.items()}
        report["split_pairs"] = [(p, by_pair[p]["votes"], labels[p]) for p in groups["split"]]
    grades = {(r["article_id"], r["ticker"]): r["materiality"] for r in results}
    if any(g is not None for g in grades.values()):
        report["crosstab"] = {lab: {g: sum(1 for p, t in labels.items() if t == lab and grades[p] == g)
                                    for g in ("none", "minor", "material")} for lab in (True, False)}
        report["cutoffs"] = {
            "material": score(labels, {p: grades[p] == "material" for p in labels}),
            "minor_or_material": score(labels, {p: grades[p] in ("minor", "material") for p in labels})}
        for d in report["disagreements"]:
            d["grade"] = grades[d["pair"]]
    return report


def format_report(report: Dict) -> str:
    def pct(x): return "n/a" if x is None else f"{x:.1%}"

    def block(title, r):
        return [f"{title}: n={r['n']}  TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}",
                f"  precision {pct(r['precision'])}  95% CI {r['precision_95ci']}",
                f"  recall    {pct(r['recall'])}  95% CI {r['recall_95ci']}",
                f"  F1 {pct(r['f1'])}   accuracy {pct(r['accuracy'])}"]
    c = report["counts"]
    sc = f" x{report['samples']} samples" if report.get("samples", 1) > 1 else ""
    lines = [f"Triage {report.get('version', '')}{sc} vs hand labels ({report['model']}): "
             f"{c['sure']} sure, {c['unsure']} unsure (?), "
             f"{c['blank']} blank (skipped)"]
    lines += block("All labels (best guesses included)", report)
    lines.append(f"  AUC of model score {pct(report.get('auc'))}   Brier score "
                 f"{'n/a' if report.get('brier') is None else round(report['brier'], 3)} (lower is better)")
    if report.get("by_agreement"):
        for k, (n, acc) in report["by_agreement"].items():
            lines.append(f"  {k} votes: n={n}, accuracy {pct(acc)}")
    if report.get("sure_only") and c["unsure"]:
        lines += block("Sure labels only", report["sure_only"])
    if report.get("crosstab"):
        lines.append("Model grade by your label:        none  minor  material")
        for lab, name in ((True, "label 1 (material)"), (False, "label 0 (not)     ")):
            row = report["crosstab"][lab]
            lines.append(f"  {name}             {row['none']:>4}  {row['minor']:>5}  {row['material']:>8}")
        for key, title in (("material", "cutoff = material only"), ("minor_or_material", "cutoff = minor or material")):
            r = report["cutoffs"][key]
            lines.append(f"  {title}: precision {pct(r['precision'])}, recall {pct(r['recall'])}, F1 {pct(r['f1'])}")
    if report["disagreements"]:
        lines.append("Disagreements (label vs model):")
        for d in report["disagreements"]:
            flag = " [unsure]" if d.get("unsure") else ""
            grade = f" ({d['grade']})" if d.get("grade") else ""
            lines.append(f"  {d['pair'][0]} {d['pair'][1]}: label={int(d['label'])}{flag} "
                         f"model={int(d['model'])}{grade} - {d['model_reason']}")
    return "\n".join(lines)
