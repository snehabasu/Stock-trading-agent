"""Score decision runs from the paper tracker (quick read before Punit's backtester).

NAV series: the NAV logged before each decision (= value at the previous close) plus the value
after the last decision drifts to its execution day's close. The benchmark is what the mandate
names: equal weight in the run's universe, bought at the first execution open (paying the same
cost) and held. Comparing each universe to its own benchmark separates the agent's skill from
the sector's return, which is what matters for tech vs non-tech.

All numbers come from bars inside the evaluated range; with ~40 days, treat differences as
indicative and look at the t-statistics before reading anything into them.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence

from config import TRANSACTION_COST_BPS, UNIVERSE_GROUPS
from src.market_features import PortfolioState, advance, bar_on_or_after, load_bars
from src.what_if import turnover

ANN = 252


def _rows(db, run: str, start: Optional[str], end: Optional[str]) -> List[dict]:
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM decision_log WHERE run_name=? ORDER BY as_of", (run,))]
    return [r for r in rows if (not start or r["as_of"][:10] >= start) and (not end or r["as_of"][:10] <= end)]


def _last_close_date(db, tickers: Sequence[str], before: str) -> Optional[str]:
    dates = [b[-1]["timestamp"][:10] for b in load_bars(db, tickers, before, 1).values() if b]
    return max(dates) if dates else None


def _close_on(db, ticker: str, day: str) -> Optional[float]:
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat() + "T00:00:00Z"
    b = load_bars(db, [ticker], nxt, 1)[ticker]
    return b[-1]["close"] if b and b[-1]["timestamp"][:10] == day else None


def nav_series(db, run: str, start: str = None, end: str = None) -> dict:
    rows = _rows(db, run, start, end)
    if not rows:
        raise ValueError(f"no decisions for run {run}")
    cfg = json.loads(rows[0]["config_json"])
    tickers = list(UNIVERSE_GROUPS[cfg["universe"]])
    dates, navs, invested, turnovers, holds, cites = [], [], [], [], 0, 0
    for r in rows:
        dates.append(_last_close_date(db, tickers, r["as_of"]) or "")
        navs.append(r["nav_before"])
        before = json.loads(r["portfolio_before_json"])["weights"]
        target = json.loads(r["target_json"])
        p = json.loads(r["proposal_json"])
        turnovers.append(turnover(before, target))
        invested.append(sum(target.values()))
        holds += bool(p["hold"])
        cites += len(p["cited_article_ids"])
    last = rows[-1]
    after = (date.fromisoformat(last["exec_day"]) + timedelta(days=1)).isoformat() + "T00:00:00Z"
    before = json.loads(last["portfolio_before_json"])
    final = advance(db, PortfolioState(before["weights"], before["nav"]), json.loads(last["target_json"]),
                    last["as_of"], last["exec_day"], after)
    dates.append(last["exec_day"])
    navs.append(final.nav)
    # benchmark: equal weight, bought at the first execution open, held
    first_exec = rows[0]["exec_day"]
    opens = {t: (bar_on_or_after(db, t, f"{first_exec}T00:00:00Z") or {}).get("open") for t in tickers}
    held = [t for t in tickers if opens[t]]
    bench = []
    for i, d in enumerate(dates):
        if d < first_exec:
            bench.append(1.0)
            continue
        vals = [_close_on(db, t, d) for t in held]
        bench.append((1 - TRANSACTION_COST_BPS / 1e4)
                     * sum(v / opens[t] for t, v in zip(held, vals) if v) / max(1, sum(1 for v in vals if v)))
    return {"run": run, "config": cfg, "dates": dates, "nav": navs, "bench": bench, "invested": invested,
            "turnover": turnovers, "holds": holds, "decisions": len(rows), "citations": cites}


def _rets(x: Sequence[float]) -> List[float]:
    return [x[i] / x[i - 1] - 1 for i in range(1, len(x))]


def _mean(x): return sum(x) / len(x) if x else 0.0


def _sd(x):
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _max_dd(x):
    peak, dd = x[0], 0.0
    for v in x:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
    return dd


def metrics(s: dict) -> dict:
    ra, rb = _rets(s["nav"]), _rets(s["bench"])
    ex = [a - b for a, b in zip(ra, rb)]
    vb = _sd(rb) ** 2
    beta = (sum((a - _mean(ra)) * (b - _mean(rb)) for a, b in zip(ra, rb)) / (len(ra) - 1) / vb) if vb else 0.0
    te = _sd(ex) * math.sqrt(ANN)
    return {
        "run": s["run"], "config": s["config"], "universe": s["config"]["universe"], "news": s["config"]["use_news"],
        "days": len(ra), "total_return": s["nav"][-1] / s["nav"][0] - 1,
        "bench_return": s["bench"][-1] / s["bench"][0] - 1,
        "excess_return": s["nav"][-1] / s["nav"][0] - s["bench"][-1] / s["bench"][0],
        "vol_ann": _sd(ra) * math.sqrt(ANN),
        "sharpe": (_mean(ra) / _sd(ra) * math.sqrt(ANN)) if _sd(ra) else 0.0,
        "bench_sharpe": (_mean(rb) / _sd(rb) * math.sqrt(ANN)) if _sd(rb) else 0.0,
        "max_drawdown": _max_dd(s["nav"]), "beta": beta,
        "alpha_ann": (_mean(ra) - beta * _mean(rb)) * ANN,
        "info_ratio": (_mean(ex) * ANN / te) if te else 0.0,
        "avg_invested": _mean(s["invested"]), "avg_turnover": _mean(s["turnover"]),
        "hold_rate": s["holds"] / s["decisions"], "cites_per_day": s["citations"] / s["decisions"],
        "daily": ra, "daily_bench": rb,
    }


def paired(a: dict, b: dict) -> dict:
    """a minus b on the days both runs have (e.g. news minus no-news, same universe)."""
    n = min(len(a["daily"]), len(b["daily"]))
    d = [x - y for x, y in zip(a["daily"][:n], b["daily"][:n])]
    sd = _sd(d)
    return {"a": a["run"], "b": b["run"], "days": n, "diff_total": a["total_return"] - b["total_return"],
            "diff_mean_daily": _mean(d), "t_stat": (_mean(d) / (sd / math.sqrt(n))) if sd else 0.0}


def evaluate(db, runs: Sequence[str], start: str = None, end: str = None) -> dict:
    ms = [metrics(nav_series(db, r, start, end)) for r in runs]
    def key(m, news):
        c = m["config"]
        return (c["universe"], c["mask"], c["profile"], c["prompt_version"], c["model"], news)
    by_key = {key(m, m["news"]): m for m in ms}
    pairs = [paired(m, by_key[key(m, False)]) for m in ms if m["news"] and key(m, False) in by_key]
    return {"runs": ms, "news_vs_nonews": pairs}


def format_evaluation(r: dict) -> str:
    cols = [("total_return", "return", "{:+.2%}"), ("bench_return", "bench", "{:+.2%}"),
            ("excess_return", "excess", "{:+.2%}"), ("sharpe", "sharpe", "{:.2f}"),
            ("bench_sharpe", "b.sharpe", "{:.2f}"), ("vol_ann", "vol", "{:.1%}"),
            ("max_drawdown", "maxDD", "{:.2%}"), ("beta", "beta", "{:.2f}"), ("alpha_ann", "alpha/yr", "{:+.1%}"),
            ("info_ratio", "IR", "{:+.2f}"), ("avg_invested", "invested", "{:.0%}"),
            ("avg_turnover", "turn/day", "{:.1%}"), ("hold_rate", "holds", "{:.0%}"),
            ("cites_per_day", "cites/d", "{:.1f}")]
    w = max(len(m["run"]) for m in r["runs"])
    lines = [f"{'run':<{w}}  " + "  ".join(f"{h:>8}" for _, h, _ in cols)]
    for m in r["runs"]:
        lines.append(f"{m['run']:<{w}}  " + "  ".join(f"{f.format(m[k]):>8}" for k, _, f in cols))
    if r["news_vs_nonews"]:
        lines.append("\nNews effect (news minus no-news, same universe; t on daily return differences):")
        for p in r["news_vs_nonews"]:
            lines.append(f"  {p['a']}: {p['diff_total']:+.2%} total over {p['days']} days, "
                         f"{p['diff_mean_daily'] * 1e4:+.1f} bps/day, t = {p['t_stat']:+.2f}")
    return "\n".join(lines)
