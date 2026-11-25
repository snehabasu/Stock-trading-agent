"""Audit a decision run: valid Proposal JSON every trading day, limits respected, and every
citation points to an article that existed (published, and any known update made) before the
decision time and was actually shown to the agent."""
from __future__ import annotations

import json
from typing import Dict, List

import jsonschema

from config import RISK_PROFILES
from src.decision_agent import PROPOSAL_SCHEMA, out_of_band
from src.market_features import trading_days
from src.what_if import violations


def audit_run(db, run_name: str, start: str = None, end: str = None) -> Dict:
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM decision_log WHERE run_name=? ORDER BY as_of", (run_name,))]
    if start or end:
        rows = [r for r in rows if (not start or r["as_of"][:10] >= start) and (not end or r["as_of"][:10] <= end)]
    problems: List[str] = []
    citations = 0
    if not rows:
        return {"run": run_name, "decisions": 0, "citations": 0, "problems": ["no decisions logged"]}
    cfg = json.loads(rows[0]["config_json"])
    profile = RISK_PROFILES[cfg["profile"]]
    expected = trading_days(db, start or rows[0]["as_of"][:10], end or rows[-1]["as_of"][:10])
    missing = sorted(set(expected) - {r["as_of"][:10] for r in rows})
    if missing:
        problems.append(f"no decision on trading days {missing[:10]}")
    with db.connect() as conn:
        for r in rows:
            day, as_of = r["as_of"][:10], r["as_of"]
            p = json.loads(r["proposal_json"])
            try:
                jsonschema.validate(p, PROPOSAL_SCHEMA)
            except jsonschema.ValidationError as e:
                problems.append(f"{day}: invalid Proposal JSON: {e.message}")
                continue
            if abs(sum(p["weights"].values()) + p["cash"] - 1) > 1e-4:
                problems.append(f"{day}: weights + cash != 1")
            if p["hold"] and out_of_band(p["weights"], profile):
                problems.append(f"{day}: hold kept weights beyond the drift tolerance band")
            if not p["hold"]:
                before = json.loads(r["portfolio_before_json"])["weights"]
                v = violations(p["weights"], before, profile)
                if any("limits beat turnover" in a for a in json.loads(r["adjustments_json"] or "[]")):
                    v = [x for x in v if not x.startswith("turnover")]
                problems += [f"{day}: {x}" for x in v]
            shown = set(json.loads(r["context_article_ids_json"]))
            if not cfg["use_news"] and (shown or p["cited_article_ids"]):
                problems.append(f"{day}: no-news run saw or cited news")
            for aid in p["cited_article_ids"]:
                citations += 1
                if aid not in shown:
                    problems.append(f"{day}: cited {aid} was not in the agent's context")
                a = conn.execute("SELECT published_at, updated_at FROM news_articles WHERE id=?", (aid,)).fetchone()
                if a is None:
                    problems.append(f"{day}: cited {aid} does not exist")
                elif not (a["published_at"] < as_of and (a["updated_at"] is None or a["updated_at"] <= as_of)):
                    problems.append(f"{day}: cited {aid} published {a['published_at']} / updated "
                                    f"{a['updated_at']} is not before decision time {as_of}")
    return {"run": run_name, "decisions": len(rows), "trading_days": len(expected), "citations": citations,
            "holds": sum(json.loads(r["proposal_json"])["hold"] for r in rows),
            "adjusted": sum(bool(json.loads(r["adjustments_json"] or "[]")) for r in rows),
            "problems": problems}


def format_audit(a: Dict) -> str:
    lines = [f"Audit {a['run']}: {a['decisions']} decisions over {a.get('trading_days', '?')} trading days, "
             f"{a['citations']} citations, {a.get('holds', 0)} holds, {a.get('adjusted', 0)} with limit adjustments"]
    lines.append("  PASS: valid output every day; every citation existed at decision time"
                 if not a["problems"] else f"  FAIL: {len(a['problems'])} problems")
    lines += [f"    {p}" for p in a["problems"][:30]]
    return "\n".join(lines)
