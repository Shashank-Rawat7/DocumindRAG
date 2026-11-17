"""Adapter tests for the Groq provider using the real SDK over httpx.MockTransport.

No network.  These prove the adapter handles the SDK's behaviour correctly; they
do NOT prove Groq's live API accepts every parameter (that is C9.7).
"""

from __future__ import annotations

import json

import groq
import httpx
import pytest
from pydantic import SecretStr

from app.core import chat_config
from app.core.chat_config import ChatSettings
from app.rag import chat_provider_groq as gp
from app.rag.chat_provider import ProviderResultKind as K
from app.rag.chat_provider import ProviderUsage, SettlementAction, settlement_action
from app.rag.chat_provider_groq import GroqChatProvider

KEY = "gsk_test_secret_key_123456"
SYSTEM = "You are a helpful assistant."
USER = "What is X?"
REASONING = "SECRET-REASONING-TRACE"


def make_settings(model="openai/gpt-oss-20b"):
    return ChatSettings(
        groq_api_key=SecretStr(KEY), daily_token_limit=200_000, model=model
    )


def body(**over):
    base = {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Answer [1]",
                    "reasoning": REASONING,
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    base.update(over)
    return base


class Recorder:
    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request)


def provider(handler, settings=None):
    rec = Recorder(handler)
    client = httpx.Client(transport=httpx.MockTransport(rec))
    return GroqChatProvider(settings or make_settings(), http_client=client), rec


def ok_handler(payload=None):
    return lambda request: httpx.Response(
        200, json=payload if payload is not None else body()
    )


def gen(handler, **kw):
    p, rec = provider(handler, **kw)
    return p.generate(SYSTEM, USER), rec


# ---------------------------------------------------------------- request shape
def test_request_shape():
    result, rec = gen(ok_handler())
    assert result.kind is K.COMPLETED
    assert len(rec.requests) == 1
    req = rec.requests[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    sent = json.loads(req.content)
    assert sent == {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1024,
        "reasoning_effort": "low",
        "stream": False,
    }
    for forbidden in ("user", "include_reasoning", "reasoning_format", "max_tokens"):
        assert forbidden not in sent


def test_model_comes_from_settings():
    _, rec = gen(ok_handler(), settings=make_settings("openai/gpt-oss-120b"))
    assert json.loads(rec.requests[0].content)["model"] == "openai/gpt-oss-120b"


def test_env_base_url_is_ignored(monkeypatch):
    monkeypatch.setenv("GROQ_BASE_URL", "https://evil.example.com")
    _, rec = gen(ok_handler())
    assert rec.requests[0].url.host == "api.groq.com"


def test_client_configuration():
    p, _ = provider(ok_handler())
    assert p._client.max_retries == 0
    t = p._client.timeout
    assert (t.connect, t.read, t.write, t.pool) == (3, 30, 3, 2)
    assert str(p._client.base_url).startswith("https://api.groq.com")


def test_timeout_arithmetic_and_documentation():
    assert gp.WORST_CASE_PHASE_SECONDS == 38
    assert gp.READ_TIMEOUT_SECONDS == chat_config.PROVIDER_TIMEOUT_SECONDS
    assert (
        gp.WORST_CASE_PHASE_SECONDS + chat_config.LEASE_MARGIN_SECONDS
        <= chat_config.LEASE_SECONDS
    )
    doc = " ".join((gp.__doc__ or "").split())
    assert "NOT a strict wall-clock bound" in doc


def test_message_validation_does_not_call_provider():
    p, rec = provider(ok_handler())
    for bad in ("", "  ", None, b"x"):
        with pytest.raises(ValueError):
            p.generate(bad, USER)
        with pytest.raises(ValueError):
            p.generate(SYSTEM, bad)
    assert rec.requests == []


def test_settings_type_checked():
    with pytest.raises(TypeError):
        GroqChatProvider(object())


def test_repr_hides_key():
    p, _ = provider(ok_handler())
    assert KEY not in repr(p)
    assert "gpt-oss-20b" in repr(p)


# ---------------------------------------------------------------- success
def test_success_returns_content_and_usage_unstripped():
    payload = body()
    payload["choices"][0]["message"]["content"] = "  padded answer \n"
    result, _ = gen(ok_handler(payload))
    assert result.kind is K.COMPLETED
    assert result.content == "  padded answer \n"
    assert result.usage == ProviderUsage(100, 20)
    assert settlement_action(result) is SettlementAction.PUBLISH


def test_reasoning_never_leaks():
    result, _ = gen(ok_handler())
    assert REASONING not in repr(result)
    assert REASONING not in (result.content or "")
    assert not hasattr(result, "reasoning")


# ---------------------------------------------------------------- malformed 200
def _mut(fn):
    payload = body()
    fn(payload)
    return payload


def _del(key):
    return _mut(lambda p: p.pop(key))


UNUSABLE_WITH_USAGE = {
    "length": _mut(lambda p: p["choices"][0].update(finish_reason="length")),
    "tool_calls": _mut(lambda p: p["choices"][0].update(finish_reason="tool_calls")),
    "no_finish": _mut(lambda p: p["choices"][0].pop("finish_reason")),
    "null_finish": _mut(lambda p: p["choices"][0].update(finish_reason=None)),
    "two_choices": _mut(lambda p: p["choices"].append(dict(p["choices"][0]))),
    "empty_choices": _mut(lambda p: p.update(choices=[])),
    "null_choices": _mut(lambda p: p.update(choices=None)),
    "dict_choices": _mut(lambda p: p.update(choices={})),
    "choice_not_dict": _mut(lambda p: p.update(choices=["x"])),
    "no_message": _mut(lambda p: p["choices"][0].pop("message")),
    "role_user": _mut(lambda p: p["choices"][0]["message"].update(role="user")),
    "no_role": _mut(lambda p: p["choices"][0]["message"].pop("role")),
    "null_content": _mut(lambda p: p["choices"][0]["message"].update(content=None)),
    "blank_content": _mut(lambda p: p["choices"][0]["message"].update(content=" \n\t")),
    "empty_content": _mut(lambda p: p["choices"][0]["message"].update(content="")),
    "list_content": _mut(lambda p: p["choices"][0]["message"].update(content=["a"])),
    "int_content": _mut(lambda p: p["choices"][0]["message"].update(content=5)),
    "missing_content": _mut(lambda p: p["choices"][0]["message"].pop("content")),
}


@pytest.mark.parametrize("name", sorted(UNUSABLE_WITH_USAGE))
def test_unusable_keeps_valid_usage(name):
    result, _ = gen(ok_handler(UNUSABLE_WITH_USAGE[name]))
    assert result.kind is K.UNUSABLE
    assert result.error_code == "invalid_response"
    assert result.usage == ProviderUsage(100, 20)
    assert settlement_action(result) is SettlementAction.FAIL_WITH_USAGE


BAD_USAGE_PAYLOADS = {
    "no_usage": _del("usage"),
    "null_usage": _mut(lambda p: p.update(usage=None)),
    "usage_list": _mut(lambda p: p.update(usage=[1, 2])),
    "usage_partial": _mut(lambda p: p.update(usage={"prompt_tokens": 5})),
    "usage_negative": _mut(
        lambda p: p.update(usage={"prompt_tokens": -1, "completion_tokens": 2})
    ),
    "usage_bool": _mut(
        lambda p: p.update(usage={"prompt_tokens": True, "completion_tokens": 2})
    ),
    "usage_float": _mut(
        lambda p: p.update(usage={"prompt_tokens": 5.0, "completion_tokens": 2})
    ),
    "usage_str": _mut(
        lambda p: p.update(usage={"prompt_tokens": "5", "completion_tokens": 2})
    ),
    "usage_overflow": _mut(
        lambda p: p.update(usage={"prompt_tokens": 2**31 - 1, "completion_tokens": 1})
    ),
}


@pytest.mark.parametrize("name", sorted(BAD_USAGE_PAYLOADS))
def test_valid_answer_stays_completed_when_usage_is_missing_or_malformed(name):
    result, _ = gen(ok_handler(BAD_USAGE_PAYLOADS[name]))
    assert result.kind is K.COMPLETED
    assert result.content == "Answer [1]"
    assert result.usage is None
    assert result.error_code is None
    assert settlement_action(result) is SettlementAction.PUBLISH


def test_sparse_body_is_unusable_without_usage():
    result, _ = gen(ok_handler({}))
    assert result.kind is K.UNUSABLE
    assert result.usage is None
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN


@pytest.mark.parametrize("name", sorted(UNUSABLE_WITH_USAGE))
def test_unusable_with_bad_usage_has_no_usage(name):
    payload = json.loads(json.dumps(UNUSABLE_WITH_USAGE[name]))
    payload["usage"] = {"prompt_tokens": True, "completion_tokens": 2}
    result, _ = gen(ok_handler(payload))
    assert result.kind is K.UNUSABLE
    assert result.usage is None
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, text=""),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json="str"),
        httpx.Response(200, json=None),
        httpx.Response(200, content=b"\xff\xfe"),
    ],
)
def test_non_object_200_is_unusable_without_usage(response):
    result, _ = gen(lambda request: response)
    assert result.kind is K.UNUSABLE
    assert result.usage is None
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN


# ---------------------------------------------------------------- status table
def status_handler(status, headers=None):
    return lambda request: httpx.Response(
        status, headers=headers or {}, json={"error": {"message": "SECRET-ERR " + KEY}}
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_rejected_statuses(status):
    result, rec = gen(status_handler(status))
    assert (result.kind, result.error_code) == (K.REJECTED, "provider_rejected")
    assert settlement_action(result) is SettlementAction.MARK_REJECTED
    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    "status",
    [
        402,
        405,
        406,
        408,
        409,
        410,
        418,
        424,
        425,
        426,
        451,
        499,
        500,
        501,
        502,
        503,
        504,
        529,
        599,
    ],
)
def test_other_statuses_are_uncertain(status):
    result, rec = gen(status_handler(status))
    assert (result.kind, result.error_code) == (K.UNCERTAIN, "provider_unavailable")
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN
    assert len(rec.requests) == 1


@pytest.mark.parametrize("status", [300, 301, 302, 307])
def test_redirects_are_never_zero_charge(status):
    result, rec = gen(
        lambda request: httpx.Response(status, headers={"location": "http://x/"})
    )
    assert result.kind in (K.UNCERTAIN, K.UNUSABLE)
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN
    assert len(rec.requests) == 1


def test_204_is_not_a_completion():
    result, _ = gen(lambda request: httpx.Response(204))
    assert result.kind is not K.COMPLETED
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7000),
        ("0.5", 500),
        ("15", 15000),
        ("16", None),
        ("abc", None),
        ("Wed, 21 Oct 2015 07:28:00 GMT", None),
        ("-3", None),
        (None, None),
    ],
)
def test_429_retry_after(header, expected):
    headers = {"retry-after": header} if header is not None else {}
    result, rec = gen(status_handler(429, headers))
    assert result.kind is K.RATE_LIMITED
    assert result.error_code == "provider_rate_limited"
    assert result.retry_after_ms == expected
    assert settlement_action(result) is SettlementAction.MARK_RATE_LIMITED
    assert len(rec.requests) == 1


def test_error_body_and_key_never_surface():
    for status in (400, 429, 500):
        result, _ = gen(status_handler(status))
        text = repr(result)
        assert "SECRET-ERR" not in text and KEY not in text


# ---------------------------------------------------------------- transport
def raising(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    "exc",
    [httpx.ConnectError("c"), httpx.ConnectTimeout("ct"), httpx.PoolTimeout("pt")],
)
def test_never_sent_failures_are_zero_charge(exc):
    result, rec = gen(raising(exc))
    assert (result.kind, result.error_code) == (K.REJECTED, "provider_unavailable")
    assert settlement_action(result) is SettlementAction.MARK_REJECTED
    assert len(rec.requests) == 1


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("r"), httpx.WriteTimeout("w")])
def test_read_and_write_timeouts_are_uncertain(exc):
    result, rec = gen(raising(exc))
    assert (result.kind, result.error_code) == (K.UNCERTAIN, "provider_timeout")
    assert settlement_action(result) is SettlementAction.MARK_UNCERTAIN
    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("re"),
        httpx.WriteError("we"),
        httpx.RemoteProtocolError("rp"),
        httpx.LocalProtocolError("lp"),
        httpx.CloseError("cl"),
        httpx.ProxyError("px"),
        httpx.UnsupportedProtocol("up"),
        httpx.DecodingError("de"),
        RuntimeError("boom"),
        ValueError("v"),
    ],
)
def test_other_transport_failures_are_uncertain_never_zero_charge(exc):
    result, rec = gen(raising(exc))
    assert result.kind is K.UNCERTAIN
    assert result.error_code == "provider_unavailable"
    assert len(rec.requests) == 1


# ---------------------------------------------------------------- classify directly
def _status_error(cls, status):
    req = httpx.Request("POST", "https://api.groq.com/x")
    return cls("m", response=httpx.Response(status, request=req), body=None)


def test_classify_unknown_exception_types():
    for exc in (RuntimeError("x"), KeyError("x"), ValueError("v")):
        with pytest.raises(TypeError):
            gp.classify_exception(exc)
    assert (
        gp.classify_exception(
            groq.APIError("m", request=httpx.Request("POST", "http://x"), body=None)
        ).kind
        is K.UNCERTAIN
    )


def test_classify_connection_error_without_cause_is_uncertain():
    exc = groq.APIConnectionError(request=httpx.Request("POST", "http://x"))
    assert gp.classify_exception(exc).kind is K.UNCERTAIN


def test_classify_timeout_without_cause_is_uncertain_timeout():
    exc = groq.APITimeoutError(request=httpx.Request("POST", "http://x"))
    result = gp.classify_exception(exc)
    assert (result.kind, result.error_code) == (K.UNCERTAIN, "provider_timeout")


def test_classify_response_validation_error_is_unusable():
    req = httpx.Request("POST", "http://x")
    exc = groq.APIResponseValidationError(
        response=httpx.Response(200, request=req), body=None, message="m"
    )
    result = gp.classify_exception(exc)
    assert result.kind is K.UNUSABLE and result.usage is None


def test_classify_status_error_classes():
    assert (
        gp.classify_exception(_status_error(groq.BadRequestError, 400)).kind
        is K.REJECTED
    )
    assert (
        gp.classify_exception(_status_error(groq.InternalServerError, 503)).kind
        is K.UNCERTAIN
    )
    assert (
        gp.classify_exception(_status_error(groq.APIStatusError, 408)).kind
        is K.UNCERTAIN
    )
    assert (
        gp.classify_exception(_status_error(groq.APIStatusError, 413)).kind
        is K.REJECTED
    )


def test_rejected_allowlist_is_exact():
    assert gp.REJECTED_STATUSES == frozenset({400, 401, 403, 404, 413, 422})


# ---------------------------------------------------------------- normalize_body direct
def test_normalize_body_non_dict():
    for raw in (None, [], "x", 5):
        assert gp.normalize_body(raw).kind is K.UNUSABLE


def test_normalize_body_bool_and_float_usage_direct():
    for bad in (True, 5.0):
        payload = body(usage={"prompt_tokens": bad, "completion_tokens": 2})
        result = gp.normalize_body(payload)
        assert result.kind is K.COMPLETED
        assert result.usage is None
        assert result.content == "Answer [1]"


def test_normalize_body_does_not_read_reasoning():
    class Trap(dict):
        def get(self, key, default=None):
            if key in ("reasoning", "reasoning_content"):
                raise AssertionError("reasoning was read")
            return super().get(key, default)

    payload = body()
    payload["choices"][0]["message"] = Trap(payload["choices"][0]["message"])
    assert gp.normalize_body(payload).kind is K.COMPLETED


# ---------------------------------------------------------------- single call guarantees
def test_one_http_call_on_success_and_close():
    p, rec = provider(ok_handler())
    p.generate(SYSTEM, USER)
    assert len(rec.requests) == 1
    p.close()


class _Unexpected(Exception):
    pass


@pytest.mark.parametrize(
    "exc", [_Unexpected("bug"), RuntimeError("bug"), KeyError("k"), TypeError("t")]
)
def test_unexpected_exception_propagates_unchanged(exc):
    p, rec = provider(ok_handler())

    def boom(**kwargs):
        raise exc

    p._client.chat.completions.with_raw_response.create = boom
    with pytest.raises(type(exc)) as info:
        p.generate(SYSTEM, USER)
    assert info.value is exc
    assert rec.requests == []


def test_unexpected_exception_while_decoding_body_propagates():
    class Raw:
        class http_response:
            @staticmethod
            def json():
                raise _Unexpected("bug")

    p, _ = provider(ok_handler())
    p._client.chat.completions.with_raw_response.create = lambda **kw: Raw()
    with pytest.raises(_Unexpected):
        p.generate(SYSTEM, USER)
