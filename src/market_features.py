"""Point-in-time price features and a simple paper-portfolio tracker for the decision agent.

Daily bars are stamped at the session date (e.g. 2026-08-03T04:00:00Z). A decision at as_of
(midnight UTC, after the previous close) may only use bars with timestamp < as_of.

The tracker is deliberately simple (it gives the agent a current portfolio and feedback for
its thesis memory); Punit's backtester is the authoritative P&L:
  decision at as_of -> old weights ride overnight to the next session's open -> rebalance at
  that open, paying TRANSACTION_COST_BPS on traded notional -> new weights drift to the last
  close before the next decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from config import DEFAULT_STOCK_FEED, PRICE_ADJUSTMENT, TIMEFRAME, TRANSACTION_COST_BPS


def load_bars(db, tickers: Sequence[str], before: str, lookback: int, feed: str = DEFAULT_STOCK_FEED) -> Dict[str, List[dict]]:
    """Last `lookback` bars per ticker with timestamp < before, oldest first."""
    out = {}
    with db.connect() as conn:
        for t in tickers:
            rows = conn.execute(
                """SELECT timestamp, open, close FROM price_bars WHERE ticker=? AND timeframe=?
                   AND adjustment=? AND feed=? AND timestamp < ? ORDER BY timestamp DESC LIMIT ?""",
                (t, TIMEFRAME, PRICE_ADJUSTMENT, feed, before, lookback)).fetchall()
            out[t] = [dict(r) for r in reversed(rows)]
    return out


def bar_on_or_after(db, ticker: str, day_start: str, feed: str = DEFAULT_STOCK_FEED) -> Optional[dict]:
    with db.connect() as conn:
        r = conn.execute(
            """SELECT timestamp, open, close FROM price_bars WHERE ticker=? AND timeframe=? AND adjustment=?
               AND feed=? AND timestamp >= ? ORDER BY timestamp LIMIT 1""",
            (ticker, TIMEFRAME, PRICE_ADJUSTMENT, feed, day_start)).fetchone()
    return dict(r) if r else None


def trading_days(db, start: str, end: str, feed: str = DEFAULT_STOCK_FEED) -> List[str]:
    """Session dates (YYYY-MM-DD) with bars between start and end inclusive."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT substr(timestamp,1,10) d FROM price_bars WHERE timeframe=? AND adjustment=?
               AND feed=? AND substr(timestamp,1,10) BETWEEN ? AND ? ORDER BY d""",
            (TIMEFRAME, PRICE_ADJUSTMENT, feed, start, end)).fetchall()
    return [r["d"] for r in rows]


def returns(bars: List[dict]) -> List[float]:
    closes = [b["close"] for b in bars]
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]


def features(bars: List[dict]) -> Optional[dict]:
    if len(bars) < 21:
        return None
    c = [b["close"] for b in bars]
    r20 = returns(bars[-21:])
    mean = sum(r20) / len(r20)
    vol = math.sqrt(sum((x - mean) ** 2 for x in r20) / (len(r20) - 1)) * math.sqrt(252)
    return {"ret_1d": c[-1] / c[-2] - 1, "ret_5d": c[-1] / c[-6] - 1, "ret_20d": c[-1] / c[-21] - 1,
            "vol_20d_ann": vol, "below_20d_high": c[-1] / max(c[-20:]) - 1}


def portfolio_vol(weights: Dict[str, float], bars: Dict[str, List[dict]], window: int = 60) -> Optional[float]:
    """Annualized ex-ante volatility from the sample covariance of the last `window` daily returns."""
    names = [t for t, w in weights.items() if w > 0]
    if not names:
        return 0.0
    rets = {t: returns(bars[t])[-window:] for t in names}
    n = min(len(r) for r in rets.values())
    if n < 20:
        return None
    rets = {t: r[-n:] for t, r in rets.items()}
    means = {t: sum(r) / n for t, r in rets.items()}
    var = 0.0
    for a in names:
        for b in names:
            cov = sum((rets[a][k] - means[a]) * (rets[b][k] - means[b]) for k in range(n)) / (n - 1)
            var += weights[a] * weights[b] * cov
    return math.sqrt(max(var, 0.0) * 252)


@dataclass
class PortfolioState:
    weights: Dict[str, float] = field(default_factory=dict)  # fraction of NAV per ticker
    nav: float = 1.0

    @property
    def cash(self) -> float:
        return 1.0 - sum(self.weights.values())


def advance(db, prev_before: PortfolioState, prev_target: Dict[str, float], prev_as_of: str,
            exec_day: str, as_of: str, cost_bps: float = TRANSACTION_COST_BPS) -> PortfolioState:
    """Portfolio at `as_of` given the previous decision (made at prev_as_of, executed at the
    open of exec_day). Uses only bars with timestamp < as_of."""
    names = sorted(set(prev_before.weights) | set(prev_target))
    nav = prev_before.nav
    # 1) old weights ride from the last close before prev_as_of to the execution open
    gross_open, w_open = 0.0, {}
    for t in names:
        w = prev_before.weights.get(t, 0.0)
        last = load_bars(db, [t], prev_as_of, 1)[t]
        ex = bar_on_or_after(db, t, f"{exec_day}T00:00:00Z")
        f = (ex["open"] / last[-1]["close"]) if (w and last and ex) else 1.0
        w_open[t] = w * f
        gross_open += w * f
    nav_open = nav * (prev_before.cash + gross_open)
    w_open = {t: v * nav / nav_open for t, v in w_open.items()}
    # 2) rebalance to target at the open, paying costs on traded notional
    traded = sum(abs(prev_target.get(t, 0.0) - w_open.get(t, 0.0)) for t in names)
    nav_after_cost = nav_open * (1 - cost_bps / 1e4 * traded)
    # 3) target weights drift from the execution open to the last close before as_of
    gross_end, w_end = 0.0, {}
    cash = 1.0 - sum(prev_target.values())
    for t in names:
        w = prev_target.get(t, 0.0)
        if not w:
            continue
        ex = bar_on_or_after(db, t, f"{exec_day}T00:00:00Z")
        last = load_bars(db, [t], as_of, 1)[t]
        f = last[-1]["close"] / ex["open"] if (ex and last) else 1.0
        w_end[t] = w * f
        gross_end += w * f
    total = cash + gross_end
    return PortfolioState(weights={t: v / total for t, v in w_end.items() if v > 1e-9},
                          nav=nav_after_cost * total)
