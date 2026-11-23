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


def transient(msg="503 unavailable"):
    return TransientLLMError(msg)
