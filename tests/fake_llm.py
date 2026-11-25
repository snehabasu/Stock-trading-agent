"""Offline stand-in for Gemini: answers triage/extraction prompts deterministically."""
import json
from src.llm_client import LLMResponse, TransientLLMError


def _payload(prompt, marker):
    return json.loads(prompt.split(marker, 1)[1].split("\n\nYour previous response", 1)[0])


class FakeGemini:
    def __init__(self, model_version=None, script=None):
        self.calls = []
        self.model_version = model_version      # None -> echo requested model
        self.script = list(script or [])        # queued exceptions / raw texts to return first

    def generate(self, request):
        self.calls.append(request)
        version = self.model_version or request.model
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception): raise step
            return LLMResponse(text=step, model_version=version)
        prompt = request.full_prompt()
        if "CONTEXT:\n" in prompt:
            return LLMResponse(json.dumps(self._decide(prompt)), version, 300, 80)
        if "ARTICLES:" in prompt:
            graded = "materiality" in request.schema["properties"]["results"]["items"]["properties"]
            results = []
            for a in _payload(prompt, "ARTICLES:\n"):
                h = a["headline"].lower()
                for t in a["tickers_to_judge"]:
                    r = {"article_id": a["article_id"], "ticker": t, "reason": "fake"}
                    if graded and "flip" in h and request.seed is not None:
                        r["materiality"] = ["none", "minor", "material", "none", "none"][request.seed % 5]
                    elif graded:
                        r["materiality"] = "none" if "roundup" in h else "minor" if "minor" in h else "material"
                    else:
                        r["relevant"] = "roundup" not in h
                    results.append(r)
            return LLMResponse(json.dumps({"results": results}), version, 100, 20)
        events = []
        for it in _payload(prompt, "ITEMS:\n"):
            h = it["headline"].lower()
            events.append({"article_id": it["article_id"], "ticker": it["ticker"],
                           "event_type": "earnings" if "earnings" in h else "analyst_action",
                           "direction": "positive" if "beat" in h or "upgrade" in h else "negative",
                           "horizon": "short_term", "confidence": 0.85})
        return LLMResponse(json.dumps({"events": events}), version, 100, 20)


    @staticmethod
    def _decide(prompt):
        """Buy 30% of each stock with positive news (deliberately over max_position, to exercise
        the review round and enforcement), cite that news; otherwise start with 10% in the first
        three stocks; otherwise hold."""
        ctx = json.loads(prompt.split("CONTEXT:\n", 1)[1].split("\n\nDRAFT:", 1)[0].split("\n\nYour previous", 1)[0])
        review = "\n\nDRAFT:" in prompt
        cap = ctx["risk_profile"]["max_position"] if review else 1.0
        pos = [e for e in ctx.get("news_events", []) if e["direction"] == "positive"]
        if pos:
            tickers = sorted({e["ticker"] for e in pos})
            return {"rationale": "Positive news on " + ", ".join(tickers), "hold": False,
                    "weights": [{"ticker": t, "weight": min(0.3, cap)} for t in tickers],
                    "cited_article_ids": sorted({e["article_id"] for e in pos})}
        if not ctx["portfolio"]["positions"]:
            return {"rationale": "Initial allocation", "hold": False,
                    "weights": [{"ticker": m["ticker"], "weight": 0.1} for m in ctx["market"][:3]],
                    "cited_article_ids": []}
        return {"rationale": "Nothing changed; hold", "hold": True, "weights": [], "cited_article_ids": []}


def transient(msg="503 unavailable"):
    return TransientLLMError(msg)
