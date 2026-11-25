import json
from dataclasses import replace
from datetime import date, timedelta

import jsonschema
import pytest

from config import TICKERS
from src.database import Database
from src.decision_agent import PROPOSAL_FIELDS, PROPOSAL_SCHEMA, DecisionAgent, DecisionConfig, run_decisions
from src.decision_audit import audit_run
from src.llm_client import LLMClient, ResponseCache
from src.market_features import PortfolioState, advance
from src.masking import Masker, mask_ids
from src.what_if import enforce, turnover, violations, what_if
from tests.fake_llm import FakeGemini
from tests.helpers import make_news

NAMES = ["MSFT", "NVDA", "WFC"]
START = date(2026, 7, 1)


def weekdays(start, n):
    d, out = start, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "m.sqlite"); db.initialize()
    days = weekdays(START, 45)
    for i, day in enumerate(days):
        for k, t in enumerate(NAMES):
            px = 100 * (1 + 0.002 * (k + 1)) ** i
            db.upsert_bar({"ticker": t, "timestamp": f"{day}T04:00:00Z", "timeframe": "1Day", "adjustment": "all",
                           "feed": "iex", "open": px * 0.999, "high": px * 1.01, "low": px * 0.99, "close": px,
                           "volume": 1e6, "trade_count": 1, "vwap": px, "ingested_at": "2026-11-23T00:00:00Z"})
    # positive MSFT news visible 2026-08-04 00:00Z; a future article must never be shown or cited
    db.upsert_news(make_news("n1", "2026-08-03T14:00:00Z", tickers=["MSFT"], headline="Microsoft earnings beat"))
    db.upsert_news(make_news("future", "2026-08-10T14:00:00Z", tickers=["NVDA"], headline="Nvidia upgrade"))
    with db.connect() as c:
        for aid, t, avail in [("n1", "MSFT", "2026-08-04T00:00:00Z"), ("future", "NVDA", "2026-08-11T00:00:00Z")]:
            c.execute("""INSERT INTO news_events (article_id,ticker,event_type,direction,horizon,confidence,model,
                         prompt_version,prompt_hash,input_hash,available_at,created_at)
                         VALUES (?,?,'earnings','positive','short_term',0.9,'gemini-3.5-flash-lite','extract-v1',
                         'h','i',?,'t')""", (aid, t, avail))
    return db


def llm(tmp_path, fake):
    return LLMClient(fake, ResponseCache(tmp_path / "c.sqlite"), sleep=lambda s: None)


def test_validation_window_runs_valid_and_audit_passes(tmp_path, db):
    fake = FakeGemini()
    cfg = DecisionConfig()
    out = run_decisions(db, llm(tmp_path, fake), cfg, "2026-08-03", "2026-08-07", out_dir=tmp_path / "dec")
    assert [r["proposal"]["as_of"][:10] for r in out] == ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
    for r in out:
        p = r["proposal"]
        jsonschema.validate(p, PROPOSAL_SCHEMA)
        assert list(p) == PROPOSAL_FIELDS
        slack = 0.02 if p["hold"] else 1e-9  # holds may keep small drift
        assert all(w <= cfg.risk["max_position"] + slack for w in p["weights"].values())
        assert p["cash"] >= cfg.risk["min_cash"] - slack
    day1, day2 = out[0]["proposal"], out[1]["proposal"]
    assert day1["cited_article_ids"] == [] and not day1["hold"]
    assert day2["cited_article_ids"] == ["n1"] and "MSFT" in day2["weights"]   # news visible from 08-04
    assert all("future" not in r["proposal"]["cited_article_ids"] for r in out)
    a = audit_run(db, cfg.run_name, "2026-08-03", "2026-08-07")
    assert a["problems"] == [] and a["decisions"] == 5 and a["citations"] >= 1
    assert (tmp_path / "dec" / f"{cfg.run_name}.jsonl").read_text().count("\n") == 5


def test_rerun_is_served_from_cache(tmp_path, db):
    fake = FakeGemini()
    run_decisions(db, llm(tmp_path, fake), DecisionConfig(), "2026-08-03", "2026-08-05", out_dir=tmp_path / "d")
    n = len(fake.calls)
    run_decisions(db, llm(tmp_path, fake), DecisionConfig(), "2026-08-03", "2026-08-05", out_dir=tmp_path / "d")
    assert len(fake.calls) == n


def test_review_round_sees_what_if_and_fixes_violation(tmp_path, db):
    fake = FakeGemini()
    run_decisions(db, llm(tmp_path, fake), DecisionConfig(), "2026-08-03", "2026-08-04", out_dir=tmp_path / "d")
    review = [c for c in fake.calls if "WHAT_IF" in c.prompt]
    assert review and any("exceeds max_position" in c.prompt for c in review)
    with db.connect() as c:
        row = c.execute("SELECT * FROM decision_log WHERE as_of='2026-08-04T00:00:00Z'").fetchone()
    assert json.loads(row["what_if_json"])["violations"]                      # draft violated (30% > 20%)
    assert json.loads(row["proposal_json"])["weights"]["MSFT"] <= 0.2 + 1e-9  # final respects the limit


def test_memory_reads_only_past_decisions(tmp_path, db):
    fake = FakeGemini()
    run_decisions(db, llm(tmp_path, fake), DecisionConfig(), "2026-08-03", "2026-08-05", out_dir=tmp_path / "d")
    agent = DecisionAgent(db, llm(tmp_path, fake), DecisionConfig())
    mem = agent.memory("2026-08-05T00:00:00Z", 1.0)
    assert [m["days_ago"] for m in mem] == [1, 2]
    assert agent.memory("2026-08-03T00:00:00Z", 1.0) == []


def test_no_news_mode_hides_news_and_cites_nothing(tmp_path, db):
    fake = FakeGemini()
    cfg = DecisionConfig(use_news=False)
    out = run_decisions(db, llm(tmp_path, fake), cfg, "2026-08-03", "2026-08-07", out_dir=tmp_path / "d")
    assert all(r["proposal"]["cited_article_ids"] == [] for r in out)
    assert all("news_events" not in c.prompt and "Microsoft earnings" not in c.prompt for c in fake.calls)
    assert cfg.run_name.startswith("all-nonews") and audit_run(db, cfg.run_name)["problems"] == []


def test_masked_mode_prompt_has_no_names_and_output_is_unmasked(tmp_path, db):
    fake = FakeGemini()
    cfg = DecisionConfig(mask=True)
    out = run_decisions(db, llm(tmp_path, fake), cfg, "2026-08-03", "2026-08-05", out_dir=tmp_path / "d")
    for c in fake.calls:
        for word in ["Microsoft", "MSFT", "Nvidia", "NVDA", "WFC"]:
            assert word not in c.prompt.split("CONTEXT:", 1)[1], word
    assert set(out[1]["proposal"]["weights"]) <= set(NAMES)  # mapped back to real tickers
    assert out[1]["proposal"]["cited_article_ids"] == ["n1"]


def test_test_split_is_refused_without_flag(tmp_path, db):
    with pytest.raises(ValueError, match="test split"):
        run_decisions(db, llm(tmp_path, FakeGemini()), DecisionConfig(), "2026-09-28", "2026-10-02")


def test_universe_groups_restrict_tickers(tmp_path, db):
    fake = FakeGemini()
    cfg = DecisionConfig(universe="non_tech")
    out = run_decisions(db, llm(tmp_path, fake), cfg, "2026-08-03", "2026-08-04", out_dir=tmp_path / "d")
    assert set().union(*(r["proposal"]["weights"] for r in out)) <= {"WFC"}
    assert all("MSFT" not in c.prompt for c in fake.calls)


def test_bad_citation_is_rejected_then_retried(tmp_path, db):
    bad = json.dumps({"rationale": "x", "hold": False, "weights": [{"ticker": "MSFT", "weight": 0.1}],
                      "cited_article_ids": ["future"]})
    fake = FakeGemini(script=[bad])
    agent = DecisionAgent(db, llm(tmp_path, fake), DecisionConfig(review="never"))
    r = agent.decide("2026-08-04T00:00:00Z")
    assert "future" not in r["proposal"]["cited_article_ids"]
    fb = fake.calls[1].feedback
    assert "'future' is not one of" in fb or "not in news_events" in fb


# ---------- masking ----------
def test_mask_ids_stable_and_complete():
    ids = mask_ids()
    assert ids == mask_ids() and sorted(ids) == sorted(TICKERS) and len(set(ids.values())) == 15


def test_masker_text():
    m = Masker(True)
    msft, meta, brk = m.ticker("MSFT"), m.ticker("META"), m.ticker("BRK-B")
    s = m.text("Microsoft's Azure grows; MSFT up. Satya Nadella spoke. Metabolic study. Meta Platforms and BRK.B")
    assert s == (f"{msft}'s {msft}-brand grows; {msft} up. {msft}-exec spoke. Metabolic study. "
                 f"{meta} and {brk}")
    assert m.untext(f"Buy {msft}, trim {meta}") == "Buy MSFT, trim META"
    assert Masker(False).text("Microsoft") == "Microsoft"


# ---------- what-if and limits ----------
PROFILE = {"max_position": 0.2, "min_cash": 0.1, "max_turnover": 0.35, "vol_ceiling": 0.2}


def test_turnover_and_violations():
    assert turnover({}, {"A": 0.2, "B": 0.2}) == pytest.approx(0.4)
    cur = {"A": 0.05}
    v = violations({"A": 0.25, "B": 0.7}, cur, PROFILE)
    assert any("max_position" in x for x in v) and any("min_cash" in x for x in v) and any("turnover" in x for x in v)
    # funding an all-cash portfolio is exempt from max_turnover, but not from the other limits
    v0 = violations({"A": 0.25, "B": 0.7}, {}, PROFILE)
    assert not any("turnover" in x for x in v0) and any("max_position" in x for x in v0)


def test_enforce_projects_onto_limits():
    cur = {"A": 0.05}
    w, notes = enforce({"A": 0.5, "B": 0.5, "C": 0.5}, cur, PROFILE)
    assert all(x <= 0.2 + 1e-9 for x in w.values())
    assert turnover(cur, w) <= 0.35 + 1e-6 and 1 - sum(w.values()) >= 0.1 - 1e-6
    assert violations(w, cur, PROFILE) == [] and notes


def test_initial_deployment_from_cash_is_not_turnover_limited():
    w, notes = enforce({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}, {}, PROFILE)
    assert sum(w.values()) == pytest.approx(0.9, abs=1e-5)       # only min_cash binds
    assert not any("turnover" in n for n in notes)


def test_what_if_reports_trades_costs_and_vol(db):
    from src.market_features import load_bars
    bars = load_bars(db, NAMES, "2026-08-04T00:00:00Z", 60)
    wi = what_if({"MSFT": 0.1}, {"MSFT": 0.2, "NVDA": 0.1}, bars, PROFILE)
    assert [t["ticker"] for t in wi["trades"]] == ["MSFT", "NVDA"]
    assert wi["turnover"] == pytest.approx(0.2) and wi["est_cost_pct_nav"] == pytest.approx(0.01)
    assert wi["ex_ante_vol_after"] is not None and wi["violations"] == []


def test_advance_nav_arithmetic(db):
    # all cash -> buy 20% MSFT at the 08-04 open; next decision 08-05 sees the 08-04 close
    prev = PortfolioState()
    s = advance(db, prev, {"MSFT": 0.2}, "2026-08-04T00:00:00Z", "2026-08-04", "2026-08-05T00:00:00Z", cost_bps=0)
    from src.market_features import bar_on_or_after
    b = bar_on_or_after(db, "MSFT", "2026-08-04T00:00:00Z")
    growth = b["close"] / b["open"]
    assert s.nav == pytest.approx(0.8 + 0.2 * growth)
    assert s.weights["MSFT"] == pytest.approx(0.2 * growth / s.nav)


def test_hold_with_large_drift_is_trimmed(tmp_path, db):
    from src.decision_agent import out_of_band
    risk = DecisionConfig().risk
    assert not out_of_band({"MSFT": 0.21}, risk) and out_of_band({"MSFT": 0.25}, risk)


def test_day_is_retried_after_overload(tmp_path, db):
    from src.llm_client import TransientLLMError
    fake = FakeGemini(script=[TransientLLMError("503 UNAVAILABLE")] * 2)
    client = LLMClient(fake, ResponseCache(tmp_path / "c.sqlite"), max_attempts=2, sleep=lambda s: None)
    waits = []
    out = run_decisions(db, client, DecisionConfig(), "2026-08-03", "2026-08-03", out_dir=tmp_path / "d",
                        day_retries=2, sleep=waits.append)
    assert len(out) == 1 and waits == [120]


def test_prompt_versions_v2_states_neutral_weight_v1_unchanged(tmp_path, db):
    v2 = DecisionAgent(db, llm(tmp_path, FakeGemini()), DecisionConfig(universe="tech", prompt_version="decide-v2"))
    assert "0.150 of NAV per stock" in v2._system()            # (1 - 0.10 min_cash) / 6 tech names
    v1 = DecisionAgent(db, llm(tmp_path, FakeGemini()), DecisionConfig(prompt_version="decide-v1"))
    assert "Mandate" not in v1._system() and v1._system().startswith("You are the decision agent")


def test_evaluate_runs_benchmark_and_news_pairing(tmp_path, db):
    from src.evaluate_runs import evaluate, format_evaluation
    fake = FakeGemini()
    news, nonews = DecisionConfig(), DecisionConfig(use_news=False)
    for cfg in (news, nonews):
        run_decisions(db, llm(tmp_path, fake), cfg, "2026-08-03", "2026-08-14", out_dir=tmp_path / "d")
    r = evaluate(db, [news.run_name, nonews.run_name], "2026-08-03", "2026-08-14")
    m = r["runs"][0]
    assert m["days"] == 10                                     # 10 decisions -> 10 close-to-close returns
    # benchmark: equal weight in the 3 names with bars, bought at the 08-03 open, held to 08-14 close
    with db.connect() as c:
        px = {t: (c.execute("SELECT open FROM price_bars WHERE ticker=? AND timestamp LIKE '2026-08-03%'", (t,)).fetchone()[0],
                  c.execute("SELECT close FROM price_bars WHERE ticker=? AND timestamp LIKE '2026-08-14%'", (t,)).fetchone()[0])
              for t in NAMES}
    expect = (1 - 5 / 1e4) * sum(cl / op for op, cl in px.values()) / 3 - 1
    assert m["bench_return"] == pytest.approx(expect)
    assert 0 < m["avg_invested"] <= 0.9 + 1e-9
    assert len(r["news_vs_nonews"]) == 1 and r["news_vs_nonews"][0]["a"] == news.run_name
    assert "News effect" in format_evaluation(r)
