"""What-if tool: preview the portfolio effect of a proposed set of target weights before
committing, and the deterministic risk-limit enforcement applied to the final proposal."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from config import TRANSACTION_COST_BPS
from src.market_features import portfolio_vol

EPS = 1e-6


def initial_deployment(current: Dict[str, float]) -> bool:
    """Funding a new, all-cash portfolio is exempt from max_turnover (the limit governs
    rebalancing; applying it to day one would force a slow ramp out of cash)."""
    return not any(w > EPS for w in current.values())


def turnover(current: Dict[str, float], target: Dict[str, float]) -> float:
    """One-way turnover as a fraction of NAV (half the total absolute weight change, cash included)."""
    names = set(current) | set(target)
    d_stock = sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in names)
    d_cash = abs((1 - sum(target.values())) - (1 - sum(current.values())))
    return (d_stock + d_cash) / 2


def violations(target: Dict[str, float], current: Dict[str, float], profile: dict) -> List[str]:
    out = []
    for t, w in sorted(target.items()):
        if w < -EPS:
            out.append(f"{t}: negative weight {w:.3f} (long-only)")
        if w > profile["max_position"] + EPS:
            out.append(f"{t}: weight {w:.3f} exceeds max_position {profile['max_position']:.2f}")
    cash = 1 - sum(target.values())
    if cash < profile["min_cash"] - EPS:
        out.append(f"cash {cash:.3f} below min_cash {profile['min_cash']:.2f}")
    to = turnover(current, target)
    if to > profile["max_turnover"] + EPS and not initial_deployment(current):
        out.append(f"turnover {to:.3f} exceeds max_turnover {profile['max_turnover']:.2f}")
    return out


def what_if(current: Dict[str, float], target: Dict[str, float], bars, profile: dict,
            cost_bps: float = TRANSACTION_COST_BPS) -> dict:
    names = sorted(set(current) | set(target))
    trades = [{"ticker": t, "from": round(current.get(t, 0.0), 4), "to": round(target.get(t, 0.0), 4),
               "change": round(target.get(t, 0.0) - current.get(t, 0.0), 4)}
              for t in names if abs(target.get(t, 0.0) - current.get(t, 0.0)) > 1e-4]
    traded = sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in names)
    vol_now, vol_new = portfolio_vol(current, bars), portfolio_vol(target, bars)
    hard = violations(target, current, profile)
    soft = []
    if vol_new is not None and vol_new > profile["vol_ceiling"] + EPS:
        soft.append(f"ex-ante volatility {vol_new:.3f} above vol_ceiling {profile['vol_ceiling']:.2f}")
    held = {t: w for t, w in target.items() if w > 1e-4}
    return {"trades": trades, "turnover": round(turnover(current, target), 4),
            "est_cost_pct_nav": round(cost_bps / 1e4 * traded * 100, 4),
            "cash_after": round(1 - sum(target.values()), 4), "n_positions": len(held),
            "largest_position": max(held.values()) if held else 0.0,
            "hhi": round(sum(w * w for w in held.values()), 4),
            "ex_ante_vol_now": None if vol_now is None else round(vol_now, 4),
            "ex_ante_vol_after": None if vol_new is None else round(vol_new, 4),
            "violations": hard, "warnings": soft}


def enforce(target: Dict[str, float], current: Dict[str, float], profile: dict) -> Tuple[Dict[str, float], List[str]]:
    """Project a proposal onto the hard limits: long-only, max_position, min_cash, max_turnover
    (except for the initial deployment from all cash). Returns (weights, adjustments made)."""
    notes = []
    w = {t: max(0.0, v) for t, v in target.items()}
    if any(v < 0 for v in target.values()):
        notes.append("clipped negative weights to 0")
    capped = {t: min(v, profile["max_position"]) for t, v in w.items()}
    if capped != w:
        notes.append(f"capped positions at max_position {profile['max_position']:.2f}")
    w = capped
    gross_cap = 1 - profile["min_cash"]
    if sum(w.values()) > gross_cap + EPS:
        scale = gross_cap / sum(w.values())
        w = {t: v * scale for t, v in w.items()}
        notes.append(f"scaled gross exposure to {gross_cap:.2f} (min_cash)")
    to = turnover(current, w)
    if to > profile["max_turnover"] + EPS and not initial_deployment(current):
        alpha = profile["max_turnover"] / to
        names = set(current) | set(w)
        w = {t: current.get(t, 0.0) + alpha * (w.get(t, 0.0) - current.get(t, 0.0)) for t in names}
        w = {t: min(max(v, 0.0), profile["max_position"]) for t, v in w.items()}
        notes.append(f"moved {alpha:.0%} of the way to target to respect max_turnover {profile['max_turnover']:.2f}")
        # position and cash limits take priority over turnover if a drifted portfolio forces a choice
        if sum(w.values()) > gross_cap + EPS:
            scale = gross_cap / sum(w.values())
            w = {t: v * scale for t, v in w.items()}
            notes.append("re-scaled gross exposure after the turnover step (limits beat turnover)")
    # round DOWN so rounding can never push a position, gross exposure or turnover over a limit
    return {t: math.floor(v * 1e6) / 1e6 for t, v in w.items() if v > 1e-6}, notes
