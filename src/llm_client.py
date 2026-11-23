"""Shared LLM client used by every LLM stage (triage, event extraction, later stages).

Guarantees:
- Response cache keyed by a SHA-256 hash of the full request (model, system prompt,
  user prompt, JSON schema, generation settings). Only validated responses are cached,
  so a rerun of the same request never calls the API again.
- Bounded retries: transient API errors back off exponentially; invalid JSON is
  re-requested with the validation error appended. Daily-quota exhaustion stops
  immediately (retrying cannot help) and everything finished so far stays cached.
- JSON validation: every response is parsed, checked against the JSON schema, then
  checked by an optional stage-specific validator (e.g. "one result per input").
- Model guard: the model version reported by the API must match the requested model.
  Google reroutes deprecated models to newer ones (gemini-2.5-flash -> gemini-3.5-flash-lite
  from 2026-12-01); a newer model has a later knowledge cutoff and would leak test-period
  knowledge, so a mismatch raises and is never cached.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import jsonschema

from src.time_utils import now_utc


class LLMError(RuntimeError):
    """Request failed permanently (after bounded retries, or a non-retryable error)."""


class LLMValidationError(LLMError):
    """Response was not valid JSON or failed schema / stage validation."""


class ModelMismatchError(LLMError):
    """API answered with a different model than requested (e.g. silent rerouting)."""


class QuotaExhaustedError(LLMError):
    """Daily quota exhausted; rerun after the quota resets (cached work is kept)."""


class TransientLLMError(LLMError):
    """Rate limit, timeout or 5xx; safe to retry."""


# Keys Gemini's response_json_schema does not need; they stay in the local schema.
_LOCAL_ONLY_KEYS = {"$schema", "$id", "title", "additionalProperties"}


def _strip_local_keys(schema):
    if isinstance(schema, dict):
        return {k: _strip_local_keys(v) for k, v in schema.items() if k not in _LOCAL_ONLY_KEYS}
    if isinstance(schema, list):
        return [_strip_local_keys(v) for v in schema]
    return schema


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    prompt: str
    schema: dict
    temperature: float = 0.0
    thinking_budget: Optional[int] = None
    seed: Optional[int] = None  # distinct seeds => distinct cache entries (self-consistency samples)
    feedback: Optional[str] = field(default=None, compare=False)  # retry hint, not hashed

    def cache_payload(self) -> dict:
        payload = {"model": self.model, "system": self.system, "prompt": self.prompt,
                   "schema": self.schema, "temperature": self.temperature,
                   "thinking_budget": self.thinking_budget}
        if self.seed is not None:  # omitted when unset so pre-existing cache keys stay valid
            payload["seed"] = self.seed
        return payload

    @property
    def prompt_hash(self) -> str:
        blob = json.dumps(self.cache_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def api_schema(self) -> dict:
        return _strip_local_keys(self.schema)

    def full_prompt(self) -> str:
        if not self.feedback:
            return self.prompt
        return (f"{self.prompt}\n\nYour previous response was rejected: {self.feedback}\n"
                "Return corrected JSON only.")


@dataclass
class LLMResponse:
    text: str
    model_version: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class LLMResult:
    data: Any
    prompt_hash: str
    model: str
    model_version: Optional[str]
    cached: bool


class ResponseCache:
    """SQLite cache of validated responses, keyed by prompt hash."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS llm_cache (
     prompt_hash TEXT PRIMARY KEY, stage TEXT, model TEXT NOT NULL, model_version TEXT,
     request_json TEXT NOT NULL, response_text TEXT NOT NULL,
     input_tokens INTEGER, output_tokens INTEGER, created_at TEXT NOT NULL
    );
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, prompt_hash: str):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM llm_cache WHERE prompt_hash=?", (prompt_hash,)).fetchone()

    def put(self, request: LLMRequest, stage: str, response: LLMResponse):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_cache (prompt_hash,stage,model,model_version,request_json,
                   response_text,input_tokens,output_tokens,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (request.prompt_hash, stage, request.model, response.model_version,
                 json.dumps(request.cache_payload(), sort_keys=True), response.text,
                 response.input_tokens, response.output_tokens, now_utc()))

    def delete(self, prompt_hash: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM llm_cache WHERE prompt_hash=?", (prompt_hash,))

    def stats(self):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT stage, model, model_version, COUNT(*) n, SUM(input_tokens) input_tokens,
                   SUM(output_tokens) output_tokens FROM llm_cache GROUP BY stage, model, model_version""")]


def model_version_matches(requested: str, reported: Optional[str]) -> bool:
    """Accept the exact model or a pinned revision of it (gemini-2.5-flash-001),
    but not a different model that shares a prefix (gemini-2.5-flash-lite)."""
    if not reported:
        return False
    reported = reported.split("/")[-1]
    return reported == requested or re.fullmatch(re.escape(requested) + r"-\d{3}", reported) is not None


def parse_json_text(text: str):
    text = (text or "").strip()
    if text.startswith("```"):  # tolerate fenced output
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMValidationError(f"response is not valid JSON ({e.msg} at char {e.pos})") from e


class LLMClient:
    def __init__(self, backend, cache: ResponseCache, max_attempts: int = 4,
                 base_delay: float = 2.0, max_delay: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep, require_model_version: bool = True):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.backend, self.cache = backend, cache
        self.max_attempts, self.base_delay, self.max_delay = max_attempts, base_delay, max_delay
        self.sleep, self.require_model_version = sleep, require_model_version
        self.api_calls = 0
        self.cache_hits = 0

    def _validate(self, data, request: LLMRequest, validator):
        try:
            jsonschema.validate(data, request.schema)
        except jsonschema.ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "<root>"
            raise LLMValidationError(f"schema violation at {path}: {e.message}") from e
        if validator is not None:
            validator(data)

    def _check_model(self, request: LLMRequest, response: LLMResponse):
        if response.model_version is None and not self.require_model_version:
            return
        if not model_version_matches(request.model, response.model_version):
            raise ModelMismatchError(
                f"requested {request.model!r} but API reported {response.model_version!r}; "
                "refusing to use or cache it (a newer model may know about the test period)")

    def generate_json(self, request: LLMRequest, stage: str,
                      validator: Optional[Callable[[Any], None]] = None) -> LLMResult:
        row = self.cache.get(request.prompt_hash)
        if row is not None:
            try:
                data = parse_json_text(row["response_text"])
                self._validate(data, request, validator)
                self.cache_hits += 1
                return LLMResult(data, request.prompt_hash, request.model, row["model_version"], True)
            except LLMValidationError:
                self.cache.delete(request.prompt_hash)  # stale under new validation rules

        last_error: Optional[Exception] = None
        attempt_request = request
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.api_calls += 1
                response = self.backend.generate(attempt_request)
            except TransientLLMError as e:
                last_error = e
                if attempt < self.max_attempts:
                    delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
                    self.sleep(delay * (0.5 + random.random() / 2))
                continue
            self._check_model(request, response)
            try:
                data = parse_json_text(response.text)
                self._validate(data, request, validator)
            except LLMValidationError as e:
                last_error = e
                attempt_request = replace(request, feedback=str(e)[:500])
                continue
            self.cache.put(request, stage, response)
            return LLMResult(data, request.prompt_hash, request.model, response.model_version, False)
        raise LLMError(f"{stage}: gave up after {self.max_attempts} attempts; last error: {last_error}")


class GeminiBackend:
    """google-genai backend. Maps SDK errors onto the retry taxonomy above."""

    RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str, timeout_seconds: int = 60):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(timeout=timeout_seconds * 1000))

    def generate(self, request: LLMRequest) -> LLMResponse:
        import httpx
        from google.genai import errors
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            seed=request.seed,
            response_mime_type="application/json",
            response_json_schema=request.api_schema,
            thinking_config=(types.ThinkingConfig(thinking_budget=request.thinking_budget)
                             if request.thinking_budget is not None else None),
        )
        try:
            r = self._client.models.generate_content(
                model=request.model, contents=request.full_prompt(), config=config)
        except errors.APIError as e:
            message = str(e)
            if e.code == 429 and re.search(r"per ?day", message, re.IGNORECASE):
                raise QuotaExhaustedError(f"daily quota exhausted for {request.model}: {message}") from e
            if e.code in self.RETRYABLE_CODES:
                raise TransientLLMError(message) from e
            raise LLMError(message) from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientLLMError(f"{type(e).__name__}: {e}") from e
        usage = r.usage_metadata
        return LLMResponse(
            text=r.text or "",
            model_version=r.model_version,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=((getattr(usage, "candidates_token_count", None) or 0)
                           + (getattr(usage, "thoughts_token_count", None) or 0)) if usage else None,
        )


def build_llm_client(settings: dict) -> LLMClient:
    backend = GeminiBackend(settings["api_key"], settings["timeout_seconds"])
    return LLMClient(backend, ResponseCache(settings["cache_path"]), max_attempts=settings["max_attempts"],
                     base_delay=settings.get("base_delay", 2.0))
