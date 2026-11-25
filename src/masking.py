"""Entity masking for H2: swap company names, tickers, well-known people and flagship brands
for consistent anonymous IDs (CO01..CO15), so the decision LLM cannot lean on what it
already "knows" about a company.

The ticker -> ID mapping is a seeded shuffle of the universe, so it is identical across days,
runs and machines. Replacements:
  ticker or company name -> CO07
  person (founder/CEO)   -> CO07-exec
  flagship brand/product -> CO07-brand
Longer aliases are replaced first; matching respects word boundaries. Extend ALIASES if
review of masked prompts shows identifying words leaking through.
"""
from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Optional

from config import RANDOM_SEED, TICKERS, canonical_ticker

# name aliases: matched case-sensitively as written (plus an ALL-CAPS variant)
ALIASES: Dict[str, Dict[str, List[str]]] = {
    "AMGN": {"names": ["Amgen"], "people": [], "brands": ["Repatha", "Prolia", "Enbrel", "Tavneos"]},
    "AVGO": {"names": ["Broadcom"], "people": ["Hock Tan"], "brands": ["VMware"]},
    "BRK-B": {"names": ["Berkshire Hathaway", "Berkshire"], "people": ["Warren Buffett", "Buffett", "Greg Abel"],
              "brands": ["GEICO", "BNSF", "Dairy Queen"]},
    "CAT": {"names": ["Caterpillar"], "people": [], "brands": []},
    "COST": {"names": ["Costco Wholesale", "Costco"], "people": [], "brands": ["Kirkland Signature"]},
    "JNJ": {"names": ["Johnson & Johnson", "Johnson and Johnson", "J&J"], "people": [],
            "brands": ["Janssen", "Darzalex", "Stelara"]},
    "LLY": {"names": ["Eli Lilly and Company", "Eli Lilly", "Lilly"], "people": [],
            "brands": ["Mounjaro", "Zepbound", "Alimta"]},
    "MA": {"names": ["Mastercard", "MasterCard"], "people": [], "brands": []},
    "META": {"names": ["Meta Platforms", "Meta"], "people": ["Mark Zuckerberg", "Zuckerberg"],
             "brands": ["Facebook", "Instagram", "WhatsApp", "Threads", "Oculus"]},
    "MSFT": {"names": ["Microsoft"], "people": ["Satya Nadella", "Nadella", "Bill Gates"],
             "brands": ["Azure", "Windows", "Xbox", "LinkedIn", "GitHub", "Copilot"]},
    "NVDA": {"names": ["NVIDIA", "Nvidia"], "people": ["Jensen Huang", "Huang"], "brands": ["GeForce", "CUDA"]},
    "ORCL": {"names": ["Oracle"], "people": ["Larry Ellison", "Ellison", "Safra Catz"], "brands": []},
    "PEP": {"names": ["PepsiCo", "Pepsi"], "people": [],
            "brands": ["Frito-Lay", "Gatorade", "Doritos", "Muscle Milk", "Quaker"]},
    "QCOM": {"names": ["Qualcomm"], "people": ["Cristiano Amon"], "brands": ["Snapdragon"]},
    "WFC": {"names": ["Wells Fargo"], "people": ["Charlie Scharf"], "brands": []},
}
# extra ticker spellings that appear in vendor text
TICKER_SPELLINGS = {"BRK-B": ["BRK-B", "BRK.B", "BRK/B", "BRK.A"]}


def mask_ids(universe: Iterable[str] = TICKERS, seed: int = RANDOM_SEED) -> Dict[str, str]:
    tickers = sorted(canonical_ticker(t) for t in universe)
    order = tickers[:]
    random.Random(seed).shuffle(order)
    return {t: f"CO{i + 1:02d}" for i, t in enumerate(order)}


class Masker:
    """Identity when enabled=False, so callers never branch on the mode."""

    def __init__(self, enabled: bool, universe: Iterable[str] = TICKERS, seed: int = RANDOM_SEED):
        self.enabled = enabled
        self.to_id = mask_ids(universe, seed)
        self.to_ticker = {v: k for k, v in self.to_id.items()}
        rules = []  # (alias, replacement, case_sensitive)
        for ticker, mid in self.to_id.items():
            a = ALIASES.get(ticker, {})
            for name in a.get("names", []):
                rules += [(name, mid, True), (name.upper(), mid, True)]
            for person in a.get("people", []):
                rules.append((person, f"{mid}-exec", True))
            for brand in a.get("brands", []):
                rules += [(brand, f"{mid}-brand", True), (brand.upper(), f"{mid}-brand", True)]
            for spelling in TICKER_SPELLINGS.get(ticker, [ticker]):
                rules.append((spelling, mid, True))
        rules = sorted(set(rules), key=lambda r: -len(r[0]))
        self._rules = [(re.compile(rf"(?<![\w&]){re.escape(alias)}(?![\w&])"), rep) for alias, rep, _ in rules]

    def ticker(self, t: str) -> str:
        return self.to_id[canonical_ticker(t)] if self.enabled else canonical_ticker(t)

    def unticker(self, shown: str) -> str:
        return self.to_ticker.get(shown, shown) if self.enabled else shown

    def text(self, s: Optional[str]) -> Optional[str]:
        if not self.enabled or not s:
            return s
        for pattern, rep in self._rules:
            s = pattern.sub(rep, s)
        return s

    def untext(self, s: Optional[str]) -> Optional[str]:
        """Map anonymous IDs in model output back to tickers (for human-readable logs)."""
        if not self.enabled or not s:
            return s
        return re.sub(r"\bCO(\d{2})(-exec|-brand)?\b",
                      lambda m: self.to_ticker.get(f"CO{m.group(1)}", m.group(0)) + (m.group(2) or ""), s)

    def leaks(self, s: str) -> List[str]:
        """Aliases still present after masking (for audits)."""
        if not self.enabled:
            return []
        return sorted({p.pattern for p, _ in self._rules if p.search(s)})
