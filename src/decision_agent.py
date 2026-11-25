"""Decision agent (Task 3).

Each trading day D the agent decides at as_of = D 00:00 UTC (after the previous close; orders
execute at D's open). Inputs, all point-in-time:
  - news events with available_at in (as_of - lookback, as_of]   (omitted in no-news mode)
  - price features from bars with timestamp < as_of (returns/volatility, never price levels)
  - the current paper portfolio (previous decision drifted to the last close)
  - the risk profile
  - thesis memory: this run's previous decisions (as_of < current) and the return since each
Loop: draft proposal -> what-if tool on the draft -> review round -> deterministic enforcement
of hard limits -> Proposal JSON + decision_log row.

Config flags: use_news=False is the "same LLM without news" baseline / remove-news test;
mask=True anonymizes companies (H2). Test-split dates are refused unless allow_test=True, so
prompts can only be tuned on validation data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config import (DATE_SPLITS, DECISION_EVENT_LOOKBACK_DAYS, DECISION_MASK_ENTITIES, DECISION_MAX_EVENTS,
                    DECISION_MEMORY_SIZE, DECISION_MODEL, DECISION_PRICE_LOOKBACK_BARS, DECISION_PROMPT_VERSION,
                    DECISION_REVIEW, DECISION_RISK_PROFILE, DECISION_UNIVERSE, DECISION_USE_NEWS,
                    EXTRACTOR_MODEL, RISK_PROFILES, TRANSACTION_COST_BPS, UNIVERSE_GROUPS)
from src.event_extractor import EXTRACTOR_PROMPT_VERSION
from src.llm_client import LLMClient, LLMRequest, LLMValidationError, QuotaExhaustedError, RetriesExhaustedError
from src.market_features import (PortfolioState, advance, bar_on_or_after, features, load_bars,
                                 trading_days)
from src.masking import Masker
from src.time_utils import now_utc
from src.what_if import enforce, violations, what_if

DRIFT_BAND = 0.02  # a hold may keep weights that drifted up to 2 points past a limit


def out_of_band(weights: Dict[str, float], risk: dict) -> bool:
    return (any(w > risk["max_position"] + DRIFT_BAND for w in weights.values())
            or 1 - sum(weights.values()) < risk["min_cash"] - DRIFT_BAND)


PROPOSAL_FIELDS = ["as_of", "run", "hold", "weights", "cash", "rationale", "cited_article_ids"]

PROPOSAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Proposal", "type": "object",
    "properties": {
        "as_of": {"type": "string", "format": "date-time"},
        "run": {"type": "string"},
        "hold": {"type": "boolean"},
        "weights": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
        "cash": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "minLength": 1},
        "cited_article_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "required": PROPOSAL_FIELDS, "additionalProperties": False,
}


@dataclass(frozen=True)
class DecisionConfig:
    use_news: bool = DECISION_USE_NEWS
    mask: bool = DECISION_MASK_ENTITIES
    profile: str = DECISION_RISK_PROFILE
    universe: str = DECISION_UNIVERSE
    model: str = DECISION_MODEL
    review: str = DECISION_REVIEW
    prompt_version: str = DECISION_PROMPT_VERSION
    event_lookback_days: int = DECISION_EVENT_LOOKBACK_DAYS
    max_events: int = DECISION_MAX_EVENTS
    memory_size: int = DECISION_MEMORY_SIZE
    price_lookback: int = DECISION_PRICE_LOOKBACK_BARS

    @property
    def run_name(self) -> str:
        return "-".join([self.universe, "news" if self.use_news else "nonews",
                         "masked" if self.mask else "named", self.profile, self.prompt_version])

    @property
    def tickers(self) -> List[str]:
        return list(UNIVERSE_GROUPS[self.universe])

    @property
    def risk(self) -> dict:
        return RISK_PROFILES[self.profile]


PROMPTS = {
    "decide-v1": """You are the decision agent of a long-only equity portfolio. Once per trading day, after the
market close, you set target portfolio weights for the next session.

You receive JSON with:
- risk_profile: hard limits. max_position = max weight per stock; min_cash = minimum cash weight;
  max_turnover = max one-way turnover per day (fraction of NAV); vol_ceiling = ceiling for annualized
  portfolio volatility.
- portfolio: current weights (fractions of NAV) and cash.
- market: recent returns and 20-day annualized volatility per stock (price levels are not given).
- news_events: {news}
- memory: your recent decisions, newest first, with the portfolio return since each one.
{masking}
Decide:
- rationale: under 120 words. State the thesis behind each position you add, cut or keep. Stay
  consistent with your memory unless new information changes the thesis, and say so when it does.
- hold: true keeps the current portfolio unchanged (weights are then ignored). Every trade costs about
  {cost:g} bps of the traded amount, so hold when nothing material has changed.
- weights: target weight per stock; leave a stock out for no position. Cash = 1 - sum of weights.
  Respect every risk_profile limit.
- cited_article_ids: {cite}

Use only the information provided. Do not use outside knowledge about these companies or later events.""",
    # v2 (tuned on validation): v1 parked ~65% in cash and held for 40/42 days; v2 adds an
    # equal-weight benchmark mandate so cash and every position are active tilts that need a reason.
    "decide-v2": """You are the decision agent of a long-only equity portfolio. Once per trading day, after the
market close, you set target portfolio weights for the next session.

Mandate: you are measured against a passive benchmark that holds every stock in `market` at equal
weight and is fully invested. The neutral position is {neutral:.3f} of NAV per stock with cash at
min_cash. Start from neutral and tilt: overweight stocks where the evidence is favourable, underweight
or drop stocks where it is unfavourable. Cash above min_cash is an active bet against the whole market
and needs a stated reason, like any other position. Sitting in cash without one is a failure of the mandate.

You receive JSON with:
- risk_profile: hard limits. max_position = max weight per stock; min_cash = minimum cash weight;
  max_turnover = max one-way turnover per day (fraction of NAV); vol_ceiling = guidance for annualized
  portfolio volatility.
- portfolio: current weights (fractions of NAV) and cash.
- market: recent returns and 20-day annualized volatility per stock (price levels are not given).
- news_events: {news}
- memory: your recent decisions, newest first, with the portfolio return since each one.
{masking}
Decide:
- rationale: under 120 words. State the thesis behind each tilt you add, cut or keep. Stay consistent
  with your memory unless new information changes the thesis, and say so when it does.
- hold: true keeps the current portfolio unchanged (weights are then ignored). Every trade costs about
  {cost:g} bps of the traded amount, so hold when the portfolio already matches your intended tilts and
  nothing material has changed; do not hold a portfolio that is far from your intended allocation.
- weights: target weight per stock; leave a stock out for no position. Cash = 1 - sum of weights.
  Respect every risk_profile limit.
- cited_article_ids: {cite}

Use only the information provided. Do not use outside knowledge about these companies or later events.""",
}

_NEWS_ON = ("point-in-time news events: article_id, ticker, headline, event_type, direction, horizon, "
            "confidence (that the direction is right) and age_days.")
_NEWS_OFF = "not provided in this run. Decide from market data, portfolio and memory only."
_CITE_ON = ("article_ids of the news events that motivated your decision, only ids that appear in "
            "news_events. Empty if no news drove the decision.")
_CITE_OFF = "must be an empty list (no news is provided)."
_MASK_NOTE = ("Companies are anonymized as CO01..CO15 (their people as COxx-exec, their brands as COxx-brand);\n"
              "the IDs are consistent across days. Do not try to identify the companies.\n")


class DecisionAgent:
    def __init__(self, db, llm: LLMClient, cfg: DecisionConfig = DecisionConfig()):
        self.db, self.llm, self.cfg = db, llm, cfg
        self.masker = Masker(cfg.mask)

    # ---------- context ----------
    def events(self, as_of: str) -> List[dict]:
        since = (datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                 - timedelta(days=self.cfg.event_lookback_days)).isoformat().replace("+00:00", "Z")
        q = ",".join("?" * len(self.cfg.tickers))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT e.article_id, e.ticker, e.event_type, e.direction, e.horizon, e.confidence,
                    e.available_at, a.headline FROM news_events e JOIN news_articles a ON a.id = e.article_id
                    WHERE e.ticker IN ({q}) AND e.model=? AND e.prompt_version=?
                    AND e.available_at <= ? AND e.available_at > ?
                    AND a.published_at < ? AND (a.updated_at IS NULL OR a.updated_at <= ?)
                    ORDER BY e.available_at DESC, e.confidence DESC, e.article_id LIMIT ?""",
                [*self.cfg.tickers, EXTRACTOR_MODEL, EXTRACTOR_PROMPT_VERSION, as_of, since, as_of, as_of,
                 self.cfg.max_events]).fetchall()
        return [dict(r) for r in rows]

    def previous(self, as_of: str) -> Optional[dict]:
        with self.db.connect() as conn:
            r = conn.execute("""SELECT * FROM decision_log WHERE run_name=? AND as_of<? ORDER BY as_of DESC
                                LIMIT 1""", (self.cfg.run_name, as_of)).fetchone()
        return dict(r) if r else None

    def portfolio_at(self, as_of: str) -> PortfolioState:
        prev = self.previous(as_of)
        if prev is None:
            return PortfolioState()
        before = json.loads(prev["portfolio_before_json"])
        return advance(self.db, PortfolioState(before["weights"], before["nav"]), json.loads(prev["target_json"]),
                       prev["as_of"], prev["exec_day"], as_of)

    def memory(self, as_of: str, nav_now: float) -> List[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("""SELECT as_of, nav_before, final_llm_json, target_json FROM decision_log
                                   WHERE run_name=? AND as_of<? ORDER BY as_of DESC LIMIT ?""",
                                (self.cfg.run_name, as_of, self.cfg.memory_size)).fetchall()
        now = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        out = []
        for r in rows:
            llm_out = json.loads(r["final_llm_json"])
            then = datetime.fromisoformat(r["as_of"].replace("Z", "+00:00"))
            out.append({"days_ago": (now - then).days, "hold": llm_out["hold"],
                        "target_weights": {self.masker.ticker(t): round(w, 4)
                                           for t, w in json.loads(r["target_json"]).items()},
                        "rationale": llm_out["rationale"],  # already in the run's (masked) namespace
                        "portfolio_return_since": round(nav_now / r["nav_before"] - 1, 4)})
        return out

    def build_context(self, as_of: str, state: PortfolioState):
        m = self.masker
        bars = load_bars(self.db, self.cfg.tickers, as_of, self.cfg.price_lookback)
        market = []
        for t in self.cfg.tickers:
            f = features(bars[t])
            if f is not None:
                market.append({"ticker": m.ticker(t), **{k: round(v, 4) for k, v in f.items()}})
        visible = [x["ticker"] for x in market]
        events = self.events(as_of) if self.cfg.use_news else []
        now = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        news = [{"article_id": e["article_id"], "ticker": m.ticker(e["ticker"]), "headline": m.text(e["headline"]),
                 "event_type": e["event_type"], "direction": e["direction"], "horizon": e["horizon"],
                 "confidence": e["confidence"],
                 "age_days": round((now - datetime.fromisoformat(e["available_at"].replace("Z", "+00:00")))
                                   .total_seconds() / 86400, 1)} for e in events]
        context = {"risk_profile": self.cfg.risk,
                   "portfolio": {"cash": round(state.cash, 4),
                                 "positions": [{"ticker": m.ticker(t), "weight": round(w, 4)}
                                               for t, w in sorted(state.weights.items())]},
                   "market": market}
        if self.cfg.use_news:
            context["news_events"] = news
        context["memory"] = self.memory(as_of, state.nav)
        context_ids = sorted({e["article_id"] for e in events})
        return context, bars, visible, context_ids

    # ---------- LLM ----------
    def _system(self) -> str:
        neutral = (1 - self.cfg.risk["min_cash"]) / len(self.cfg.tickers)
        return PROMPTS[self.cfg.prompt_version].format(
            news=_NEWS_ON if self.cfg.use_news else _NEWS_OFF, cite=_CITE_ON if self.cfg.use_news else _CITE_OFF,
            masking=_MASK_NOTE if self.cfg.mask else "", cost=TRANSACTION_COST_BPS, neutral=neutral)

    def _schema(self, visible: List[str], context_ids: List[str]) -> dict:
        cite_items = {"type": "string", "enum": context_ids} if context_ids else {"type": "string"}
        return {"type": "object", "properties": {
            "rationale": {"type": "string"},
            "hold": {"type": "boolean"},
            "weights": {"type": "array", "items": {"type": "object", "properties": {
                "ticker": {"type": "string", "enum": visible},
                "weight": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["ticker", "weight"], "additionalProperties": False}},
            "cited_article_ids": {"type": "array", "items": cite_items}},
            "required": ["rationale", "hold", "weights", "cited_article_ids"], "additionalProperties": False}

    def _validator(self, context_ids: List[str]):
        allowed = set(context_ids)

        def check(d):
            tickers = [w["ticker"] for w in d["weights"]]
            if len(tickers) != len(set(tickers)):
                raise LLMValidationError("weights list a ticker more than once")
            bad = [c for c in d["cited_article_ids"] if c not in allowed]
            if bad:
                raise LLMValidationError(f"cited_article_ids not in news_events: {bad[:5]}"
                                         + ("" if allowed else " (no news is provided; cite nothing)"))
            if len(d["cited_article_ids"]) != len(set(d["cited_article_ids"])):
                raise LLMValidationError("cited_article_ids contains duplicates")
            if not d["rationale"].strip() or len(d["rationale"]) > 1500:
                raise LLMValidationError("rationale must be non-empty and under ~120 words")
        return check

    def _ask(self, prompt: str, visible, context_ids, stage):
        req = LLMRequest(model=self.cfg.model, system=self._system(), prompt=prompt,
                         schema=self._schema(visible, context_ids), temperature=0.0)
        return self.llm.generate_json(req, stage=stage, validator=self._validator(context_ids))

    def _weights(self, llm_out) -> Dict[str, float]:
        return {self.masker.unticker(w["ticker"]): float(w["weight"]) for w in llm_out["weights"] if w["weight"] > 0}

    # ---------- one decision ----------
    def decide(self, as_of: str) -> dict:
        state = self.portfolio_at(as_of)
        context, bars, visible, context_ids = self.build_context(as_of, state)
        ctx = json.dumps(context, indent=1, ensure_ascii=False)
        draft = self._ask("Write rationale first, then the other fields.\n\nCONTEXT:\n" + ctx,
                          visible, context_ids, "decide")
        hashes, versions = [draft.prompt_hash], {draft.model_version}
        draft_w = state.weights if draft.data["hold"] else self._weights(draft.data)
        wi = what_if(state.weights, draft_w, bars, self.cfg.risk)
        m = self.masker  # the agent sees the what-if in its own (possibly masked) namespace
        shown_wi = what_if({m.ticker(t): w for t, w in state.weights.items()},
                           {m.ticker(t): w for t, w in draft_w.items()},
                           {m.ticker(t): b for t, b in bars.items()}, self.cfg.risk)
        final = draft
        review = (not draft.data["hold"]) and (
            self.cfg.review == "always" or (self.cfg.review == "on_violation" and (wi["violations"] or wi["warnings"])))
        if review:
            prompt = ("CONTEXT:\n" + ctx + "\n\nDRAFT:\n" + json.dumps(draft.data, ensure_ascii=False)
                      + "\n\nWHAT_IF (effect of the draft on the portfolio):\n" + json.dumps(shown_wi)
                      + "\n\nReview the draft against the what-if result and the risk profile. Fix every "
                        "violation and reconsider trades whose cost or risk is not justified. Write rationale "
                        "first, then return the final proposal.")
            final = self._ask(prompt, visible, context_ids, "decide-review")
            hashes.append(final.prompt_hash); versions.add(final.model_version)
        hold = final.data["hold"]
        if hold and out_of_band(state.weights, self.cfg.risk):
            hold = False
            target, adjustments = enforce(state.weights, state.weights, self.cfg.risk)
            adjustments.insert(0, f"hold overridden: drift beyond the {DRIFT_BAND:.0%} tolerance band")
        elif hold:
            target, adjustments = dict(state.weights), []
        else:
            target, adjustments = enforce(self._weights(final.data), state.weights, self.cfg.risk)
        ref = self.masker.unticker(visible[0]) if visible else self.cfg.tickers[0]
        exec_bar = bar_on_or_after(self.db, ref, as_of)
        exec_day = exec_bar["timestamp"][:10] if exec_bar else as_of[:10]
        proposal = {"as_of": as_of, "run": self.cfg.run_name, "hold": hold,
                    "weights": {t: round(w, 6) for t, w in sorted(target.items())},
                    "cash": round(1 - sum(target.values()), 6),
                    "rationale": self.masker.untext(final.data["rationale"]),
                    "cited_article_ids": list(dict.fromkeys(final.data["cited_article_ids"]))}
        with self.db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO decision_log (run_name, as_of, exec_day, config_json, model, model_version,
                   prompt_version, portfolio_before_json, nav_before, context_article_ids_json, draft_json,
                   what_if_json, final_llm_json, adjustments_json, proposal_json, target_json, prompt_hashes_json,
                   created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.cfg.run_name, as_of, exec_day, json.dumps(asdict(self.cfg)), self.cfg.model,
                 ",".join(sorted(v for v in versions if v)), self.cfg.prompt_version,
                 json.dumps({"weights": state.weights, "nav": state.nav}), state.nav, json.dumps(context_ids),
                 json.dumps(draft.data), json.dumps(wi), json.dumps(final.data), json.dumps(adjustments),
                 json.dumps(proposal), json.dumps(target), json.dumps(hashes), now_utc()))
        return {"proposal": proposal, "reviewed": review, "adjustments": adjustments, "what_if": wi,
                "api_calls": self.llm.api_calls, "cache_hits": self.llm.cache_hits}


def in_test_split(day: str) -> bool:
    t = DATE_SPLITS["test"]
    return t["start"] <= day <= t["end"]


def run_decisions(db, llm: LLMClient, cfg: DecisionConfig, start: str, end: str, allow_test: bool = False,
                  out_dir: Path = Path("data/decisions"), on_day=None, day_retries: int = 1,
                  day_wait_seconds: float = 120, sleep=None) -> List[dict]:
    """day_retries > 1 retries a day whose calls all failed transiently (overload/timeout);
    finished calls stay cached."""
    days = trading_days(db, start, end)
    t = DATE_SPLITS["test"]
    if not allow_test and start <= t["end"] and end >= t["start"]:
        raise ValueError("Range overlaps the test split; tune prompts on validation only "
                         "(pass allow_test=True / --allow-test for the final evaluation run).")
    agent = DecisionAgent(db, llm, cfg)
    results = []
    import time
    sleep = sleep or time.sleep
    for d in days:
        for attempt in range(1, day_retries + 1):
            try:
                r = agent.decide(f"{d}T00:00:00Z")
                break
            except QuotaExhaustedError:
                raise
            except RetriesExhaustedError:
                if attempt == day_retries:
                    raise
                sleep(day_wait_seconds)
        results.append(r)
        if on_day:
            on_day(d, r)
    export_run(db, cfg.run_name, out_dir / f"{cfg.run_name}.jsonl")
    return results


def export_run(db, run_name: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with db.connect() as conn:
        rows = conn.execute("SELECT proposal_json FROM decision_log WHERE run_name=? ORDER BY as_of",
                            (run_name,)).fetchall()
    path.write_text("".join(r["proposal_json"] + "\n" for r in rows))
    return path
