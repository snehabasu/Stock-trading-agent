import json
import pytest
from src.llm_client import (LLMClient, LLMError, LLMRequest, ModelMismatchError, QuotaExhaustedError,
                            ResponseCache, model_version_matches)
from tests.fake_llm import FakeGemini, transient

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
          "additionalProperties": False}


def req(prompt="p", model="gemini-2.5-flash-lite"):
    return LLMRequest(model=model, system="s", prompt=prompt, schema=SCHEMA)


def client(tmp_path, backend, **kw):
    return LLMClient(backend, ResponseCache(tmp_path / "c.sqlite"), sleep=lambda s: None, **kw)


def test_cache_hit_on_identical_prompt(tmp_path):
    fake = FakeGemini(script=['{"x": 1}'])
    llm = client(tmp_path, fake)
    first, second = llm.generate_json(req(), "t"), llm.generate_json(req(), "t")
    assert first.data == second.data == {"x": 1}
    assert (first.cached, second.cached) == (False, True) and len(fake.calls) == 1


def test_cache_survives_new_client(tmp_path):
    client(tmp_path, FakeGemini(script=['{"x": 1}'])).generate_json(req(), "t")
    fake = FakeGemini()
    assert client(tmp_path, fake).generate_json(req(), "t").cached and fake.calls == []


def test_prompt_hash_changes_with_any_input():
    base = req()
    variants = [req(prompt="q"), req(model="gemini-2.5-flash"),
                LLMRequest(model=base.model, system="other", prompt="p", schema=SCHEMA),
                LLMRequest(model=base.model, system="s", prompt="p", schema=SCHEMA, temperature=0.5)]
    assert len({base.prompt_hash, *(v.prompt_hash for v in variants)}) == 5


def test_invalid_json_retried_with_feedback_then_cached(tmp_path):
    fake = FakeGemini(script=["not json", '{"x": "str"}', '{"x": 2}'])
    r = client(tmp_path, fake).generate_json(req(), "t")
    assert r.data == {"x": 2} and len(fake.calls) == 3
    assert fake.calls[0].feedback is None and "schema violation" in fake.calls[2].feedback


def test_transient_errors_bounded(tmp_path):
    fake = FakeGemini(script=[transient()] * 10)
    with pytest.raises(LLMError, match="after 3 attempts"):
        client(tmp_path, fake, max_attempts=3).generate_json(req(), "t")
    assert len(fake.calls) == 3


def test_transient_then_success(tmp_path):
    fake = FakeGemini(script=[transient(), '{"x": 5}'])
    assert client(tmp_path, fake).generate_json(req(), "t").data == {"x": 5}


def test_failed_validation_never_cached(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = LLMClient(FakeGemini(script=["bad"] * 2), cache, max_attempts=2, sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm.generate_json(req(), "t")
    assert cache.get(req().prompt_hash) is None


def test_stage_validator_enforced(tmp_path):
    def must_be_seven(d):
        from src.llm_client import LLMValidationError
        if d["x"] != 7: raise LLMValidationError("x must be 7")
    fake = FakeGemini(script=['{"x": 1}', '{"x": 7}'])
    assert client(tmp_path, fake).generate_json(req(), "t", must_be_seven).data == {"x": 7}


def test_rerouted_model_rejected_and_not_cached(tmp_path):
    fake = FakeGemini(model_version="gemini-3.5-flash-lite", script=['{"x": 1}'])
    llm = client(tmp_path, fake)
    with pytest.raises(ModelMismatchError):
        llm.generate_json(req(model="gemini-2.5-flash"), "t")
    assert llm.cache.get(req(model="gemini-2.5-flash").prompt_hash) is None


def test_quota_exhaustion_not_retried(tmp_path):
    fake = FakeGemini(script=[QuotaExhaustedError("per day")])
    with pytest.raises(QuotaExhaustedError):
        client(tmp_path, fake).generate_json(req(), "t")
    assert len(fake.calls) == 1


def test_model_version_matching():
    assert model_version_matches("gemini-2.5-flash", "gemini-2.5-flash")
    assert model_version_matches("gemini-2.5-flash", "models/gemini-2.5-flash-001")
    assert not model_version_matches("gemini-2.5-flash", "gemini-2.5-flash-lite")
    assert not model_version_matches("gemini-2.5-flash", "gemini-3.5-flash-lite")
    assert not model_version_matches("gemini-2.5-flash", None)


def test_api_schema_strips_local_only_keys():
    assert "additionalProperties" not in json.dumps(req().api_schema)
