"""Pure tests for the provider-neutral contract (C9.3). No SDK, no network."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.rag import chat_error_codes as codes
from app.rag import chat_provider as cp
from app.rag.chat_provider import (
    ProviderResult,
    ProviderResultKind,
    ProviderUsage,
    SettlementAction,
    normalize_usage,
    parse_retry_after_ms,
    retry_after_note,
    settlement_action,
)
from tests.chat_fakes import ScriptedProvider

ROOT = Path(__file__).resolve().parent.parent
RAG = ROOT / "app" / "rag"
INT4 = 2**31 - 1


# ---------------------------------------------------------------- error codes
def test_error_code_vocabulary_is_exactly_the_ten_approved():
    assert codes.ALL_ERROR_CODES == {
        "conversation_not_found",
        "no_context",
        "quota_exceeded",
        "provider_rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "provider_rejected",
        "invalid_response",
        "lease_expired",
        "internal_error",
    }
    assert "source_unavailable" not in codes.ALL_ERROR_CODES


def test_error_codes_match_c8_pattern():
    c8 = re.compile(r"[a-z][a-z0-9_]{0,63}")
    assert all(c8.fullmatch(c) for c in codes.ALL_ERROR_CODES)


# ---------------------------------------------------------------- retry-after
@pytest.mark.parametrize(
    ("value", "ms", "note"),
    [
        (None, None, "missing"),
        ("0", 0, "ok"),
        ("1", 1000, "ok"),
        ("7", 7000, "ok"),
        ("0.5", 500, "ok"),
        ("1.25", 1250, "ok"),
        ("2.5", 2500, "ok"),
        ("0.001", 1, "ok"),
        ("007", 7000, "ok"),
        (" 3 ", 3000, "ok"),
        ("\t3\t", 3000, "ok"),
        ("15", 15000, "ok"),
        ("15.000", 15000, "ok"),
        ("15.001", None, "too_large"),
        ("16", None, "too_large"),
        ("999999999", None, "too_large"),
        ("1000000000", None, "invalid"),
        ("", None, "invalid"),
        ("   ", None, "invalid"),
        ("-1", None, "invalid"),
        ("+1", None, "invalid"),
        ("1e3", None, "invalid"),
        (".5", None, "invalid"),
        ("5.", None, "invalid"),
        ("1.2345", None, "invalid"),
        ("1,5", None, "invalid"),
        ("NaN", None, "invalid"),
        ("inf", None, "invalid"),
        ("0x10", None, "invalid"),
        ("1 2", None, "invalid"),
        ("١", None, "invalid"),  # non-ASCII digit
        ("Wed, 21 Oct 2015 07:28:00 GMT", None, "invalid"),
        ("5\n", None, "invalid"),
        (5, None, "invalid"),
        (5.0, None, "invalid"),
        (b"5", None, "invalid"),
    ],
)
def test_retry_after_table(value, ms, note):
    assert retry_after_note(value) == (ms, note)
    assert parse_retry_after_ms(value) == ms


# ---------------------------------------------------------------- usage
@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        "x",
        {},
        {"prompt_tokens": 1},
        {"completion_tokens": 1},
        {"prompt_tokens": None, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": None},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": False},
        {"prompt_tokens": 5.0, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 2.5},
        {"prompt_tokens": "5", "completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": -1},
        {"prompt_tokens": INT4, "completion_tokens": 1},
        {"prompt_tokens": 2**40, "completion_tokens": 0},
    ],
)
def test_usage_doubtful_becomes_none(raw):
    assert normalize_usage(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0)),
        ({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 999}, (10, 20)),
        ({"prompt_tokens": INT4, "completion_tokens": 0}, (INT4, 0)),
        ({"prompt_tokens": INT4 - 5, "completion_tokens": 5}, (INT4 - 5, 5)),
    ],
)
def test_usage_valid(raw, expected):
    usage = normalize_usage(raw)
    assert usage == ProviderUsage(*expected)
    assert usage.total_tokens == sum(expected)


def test_usage_dataclass_rejects_bad_values():
    for args in ((True, 1), (1, 1.0), (-1, 0), (INT4, 1)):
        with pytest.raises(ValueError):
            ProviderUsage(*args)


# ---------------------------------------------------------------- invariants
U = ProviderUsage(3, 4)


def test_completed_usage_is_optional():
    r = cp.completed("hi")
    assert r.usage is None
    assert settlement_action(r) is SettlementAction.PUBLISH


def test_factories_build_valid_results():
    assert cp.completed("hi", U).kind is ProviderResultKind.COMPLETED
    assert cp.unusable().usage is None
    assert cp.unusable(U).usage == U
    assert cp.rate_limited(500).retry_after_ms == 500
    assert cp.rejected().error_code == "provider_rejected"
    assert cp.rejected(codes.PROVIDER_UNAVAILABLE).error_code == "provider_unavailable"
    assert cp.uncertain().error_code == "provider_unavailable"
    assert cp.uncertain(codes.PROVIDER_TIMEOUT).error_code == "provider_timeout"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": ProviderResultKind.COMPLETED},
        {"kind": ProviderResultKind.COMPLETED, "content": "  ", "usage": U},
        {
            "kind": ProviderResultKind.COMPLETED,
            "content": "x",
            "usage": U,
            "error_code": "invalid_response",
        },
        {"kind": ProviderResultKind.COMPLETED, "content": b"x", "usage": U},
        {
            "kind": ProviderResultKind.UNUSABLE,
            "content": "x",
            "error_code": "invalid_response",
        },
        {"kind": ProviderResultKind.UNUSABLE},
        {"kind": ProviderResultKind.UNUSABLE, "error_code": "provider_unavailable"},
        {"kind": ProviderResultKind.RATE_LIMITED, "error_code": "provider_unavailable"},
        {
            "kind": ProviderResultKind.RATE_LIMITED,
            "error_code": "provider_rate_limited",
            "usage": U,
        },
        {
            "kind": ProviderResultKind.RATE_LIMITED,
            "error_code": "provider_rate_limited",
            "retry_after_ms": 15001,
        },
        {
            "kind": ProviderResultKind.RATE_LIMITED,
            "error_code": "provider_rate_limited",
            "retry_after_ms": -1,
        },
        {
            "kind": ProviderResultKind.RATE_LIMITED,
            "error_code": "provider_rate_limited",
            "retry_after_ms": True,
        },
        {"kind": ProviderResultKind.REJECTED, "error_code": "provider_timeout"},
        {
            "kind": ProviderResultKind.REJECTED,
            "error_code": "provider_rejected",
            "usage": U,
        },
        {"kind": ProviderResultKind.UNCERTAIN, "error_code": "provider_rejected"},
        {
            "kind": ProviderResultKind.UNCERTAIN,
            "error_code": "provider_unavailable",
            "usage": U,
        },
        {
            "kind": ProviderResultKind.UNCERTAIN,
            "error_code": "provider_unavailable",
            "retry_after_ms": 1,
        },
        {"kind": "completed", "content": "x", "usage": U},
        {
            "kind": ProviderResultKind.UNUSABLE,
            "error_code": "invalid_response",
            "usage": (1, 2),
        },
    ],
)
def test_result_invariants_reject(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ProviderResult(**kwargs)


def test_result_is_frozen_and_slotted():
    r = cp.rejected()
    with pytest.raises(AttributeError):
        r.kind = ProviderResultKind.COMPLETED
    assert not hasattr(r, "__dict__")


# ---------------------------------------------------------------- settlement
@pytest.mark.parametrize(
    ("result", "action"),
    [
        (cp.completed("x", U), SettlementAction.PUBLISH),
        (cp.unusable(U), SettlementAction.FAIL_WITH_USAGE),
        (cp.unusable(), SettlementAction.MARK_UNCERTAIN),
        (cp.rate_limited(), SettlementAction.MARK_RATE_LIMITED),
        (cp.rate_limited(1000), SettlementAction.MARK_RATE_LIMITED),
        (cp.rejected(), SettlementAction.MARK_REJECTED),
        (cp.rejected(codes.PROVIDER_UNAVAILABLE), SettlementAction.MARK_REJECTED),
        (cp.uncertain(), SettlementAction.MARK_UNCERTAIN),
        (cp.uncertain(codes.PROVIDER_TIMEOUT), SettlementAction.MARK_UNCERTAIN),
    ],
)
def test_settlement_action(result, action):
    assert settlement_action(result) is action


def test_settlement_action_rejects_non_results():
    with pytest.raises(TypeError):
        settlement_action("completed")


def test_every_kind_has_an_action():
    assert {k for k in ProviderResultKind} == {
        ProviderResultKind.COMPLETED,
        ProviderResultKind.UNUSABLE,
        ProviderResultKind.RATE_LIMITED,
        ProviderResultKind.REJECTED,
        ProviderResultKind.UNCERTAIN,
    }


# ---------------------------------------------------------------- fake
def test_scripted_provider_replays_and_records():
    seen = []
    ok = cp.completed("a", U)
    p = ScriptedProvider([ok, RuntimeError("boom")], on_call=seen.append)
    assert p.generate("s", "u") is ok
    with pytest.raises(RuntimeError):
        p.generate("s2", "u2")
    assert p.calls == [("s", "u"), ("s2", "u2")]
    assert seen == [1, 2]
    with pytest.raises(AssertionError):
        p.generate("s", "u")


# ---------------------------------------------------------------- isolation
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_neutral_modules_import_no_sdk_http_db_time_or_c8():
    forbidden = {"groq", "httpx", "time", "sqlalchemy", "asyncio", "logging", "random"}
    for name in ("chat_provider.py", "chat_error_codes.py"):
        imported = _imports(RAG / name)
        assert not imported & forbidden, (name, imported & forbidden)
        assert not any(
            "chat_attempt_lifecycle" in i or "chat_request_finalization" in i
            for i in imported
        )
        assert not any(i.startswith("app.db") for i in imported)


def test_only_the_groq_module_imports_the_sdk():
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        if path.name == "chat_provider_groq.py":
            continue
        if _imports(path) & {"groq"}:
            offenders.append(path.name)
    assert offenders == []


def test_groq_module_has_no_sleep_retry_db_or_c8_imports():
    imported = _imports(RAG / "chat_provider_groq.py")
    assert not imported & {"time", "sqlalchemy", "asyncio", "logging"}
    assert not any("lifecycle" in i or "finalization" in i for i in imported)
    src = (RAG / "chat_provider_groq.py").read_text(encoding="utf-8")
    assert "sleep(" not in src


def test_error_code_module_has_no_source_unavailable_constant():
    assert not any("SOURCE_UNAVAILABLE" in name for name in dir(codes))
