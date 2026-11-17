"""Provider-neutral contract for the chat answer pipeline (C9.3).

This module classifies and normalizes only.  It deliberately imports no HTTP
client, no SDK, no database code and no C8 lifecycle module, and it never
sleeps, retries or manages leases: the pipeline (C9.5) decides what to do with
a :class:`ProviderResult`.

Settlement semantics (what the pipeline must do with each outcome):

* ``COMPLETED``    - a structurally valid answer, usage optional (``None`` when
  missing or malformed; C8 then settles from the reservation): publish.
* ``UNUSABLE``     - the provider answered but the answer must not be
  published (truncated, malformed, empty ...).  With valid usage the call is
  settled with that usage; without it the outcome is uncertain.
* ``RATE_LIMITED`` - HTTP 429: zero-charge, optional ``retry_after_ms``.
* ``REJECTED``     - generation reliably did not happen: zero-charge.
* ``UNCERTAIN``    - generation may have happened: charge conservatively.

Anything not explicitly established as a rejection is ``UNCERTAIN``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol

from app.rag import chat_error_codes as codes

# Largest value a PostgreSQL int4 token column can hold.
INT4_MAX: Final = 2**31 - 1
# Upper bound for a honoured ``retry-after`` (mirrors chat_config.RETRY_AFTER_MAX_SECONDS).
RETRY_AFTER_MAX_MS: Final = 15_000

_RETRY_AFTER_RE: Final = re.compile(r"([0-9]{1,9})(?:\.([0-9]{1,3}))?")


class ProviderResultKind(str, Enum):
    COMPLETED = "completed"
    UNUSABLE = "unusable"
    RATE_LIMITED = "rate_limited"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class SettlementAction(str, Enum):
    PUBLISH = "publish"
    FAIL_WITH_USAGE = "fail_with_usage"
    MARK_UNCERTAIN = "mark_uncertain"
    MARK_RATE_LIMITED = "mark_rate_limited"
    MARK_REJECTED = "mark_rejected"


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        for value in (self.prompt_tokens, self.completion_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("usage values must be non-negative ints")
        if self.prompt_tokens + self.completion_tokens > INT4_MAX:
            raise ValueError("usage total exceeds int4")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_ERROR_CODES_BY_KIND: Final[dict[ProviderResultKind, frozenset[str]]] = {
    ProviderResultKind.COMPLETED: frozenset(),
    ProviderResultKind.UNUSABLE: frozenset({codes.INVALID_RESPONSE}),
    ProviderResultKind.RATE_LIMITED: frozenset({codes.PROVIDER_RATE_LIMITED}),
    ProviderResultKind.REJECTED: frozenset(
        {codes.PROVIDER_REJECTED, codes.PROVIDER_UNAVAILABLE}
    ),
    ProviderResultKind.UNCERTAIN: frozenset(
        {codes.PROVIDER_TIMEOUT, codes.PROVIDER_UNAVAILABLE}
    ),
}


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Outcome of one provider call.  Invariants are enforced at construction."""

    kind: ProviderResultKind
    content: str | None = None
    usage: ProviderUsage | None = None
    error_code: str | None = None
    retry_after_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProviderResultKind):
            raise TypeError("kind must be a ProviderResultKind")
        if self.usage is not None and type(self.usage) is not ProviderUsage:
            raise TypeError("usage must be a ProviderUsage")
        kind = self.kind
        if kind is ProviderResultKind.COMPLETED:
            if type(self.content) is not str or not self.content.strip():
                raise ValueError("completed results need non-blank content")
            if self.error_code is not None:
                raise ValueError("completed results carry no error code")
        else:
            if self.content is not None:
                raise ValueError("only completed results carry content")
            if self.error_code not in _ERROR_CODES_BY_KIND[kind]:
                raise ValueError("error code not allowed for this kind")
            if kind is not ProviderResultKind.UNUSABLE and self.usage is not None:
                raise ValueError("only completed/unusable results carry usage")
        if self.retry_after_ms is not None:
            if kind is not ProviderResultKind.RATE_LIMITED:
                raise ValueError("retry_after_ms only applies to rate-limited results")
            if (
                type(self.retry_after_ms) is not int
                or not 0 <= self.retry_after_ms <= RETRY_AFTER_MAX_MS
            ):
                raise ValueError("retry_after_ms out of range")


def completed(content: str, usage: ProviderUsage | None = None) -> ProviderResult:
    return ProviderResult(ProviderResultKind.COMPLETED, content=content, usage=usage)


def unusable(usage: ProviderUsage | None = None) -> ProviderResult:
    return ProviderResult(
        ProviderResultKind.UNUSABLE, usage=usage, error_code=codes.INVALID_RESPONSE
    )


def rate_limited(retry_after_ms: int | None = None) -> ProviderResult:
    return ProviderResult(
        ProviderResultKind.RATE_LIMITED,
        error_code=codes.PROVIDER_RATE_LIMITED,
        retry_after_ms=retry_after_ms,
    )


def rejected(error_code: str = codes.PROVIDER_REJECTED) -> ProviderResult:
    return ProviderResult(ProviderResultKind.REJECTED, error_code=error_code)


def uncertain(error_code: str = codes.PROVIDER_UNAVAILABLE) -> ProviderResult:
    return ProviderResult(ProviderResultKind.UNCERTAIN, error_code=error_code)


def settlement_action(result: ProviderResult) -> SettlementAction:
    """Map a provider outcome to the C8 settlement the pipeline must perform."""
    if type(result) is not ProviderResult:
        raise TypeError("result must be a ProviderResult")
    kind = result.kind
    if kind is ProviderResultKind.COMPLETED:
        return SettlementAction.PUBLISH
    if kind is ProviderResultKind.UNUSABLE:
        if result.usage is not None:
            return SettlementAction.FAIL_WITH_USAGE
        return SettlementAction.MARK_UNCERTAIN
    if kind is ProviderResultKind.RATE_LIMITED:
        return SettlementAction.MARK_RATE_LIMITED
    if kind is ProviderResultKind.REJECTED:
        return SettlementAction.MARK_REJECTED
    return SettlementAction.MARK_UNCERTAIN


def normalize_usage(raw: object) -> ProviderUsage | None:
    """Strictly normalize a raw ``usage`` object; anything doubtful is ``None``.

    Both counts must be present, exact ``int`` (not bool, float or str),
    non-negative, and their sum must fit int4.  Missing usage is never inferred.
    """
    if not isinstance(raw, Mapping):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    for value in (prompt, completion):
        if type(value) is not int or value < 0:
            return None
    assert isinstance(prompt, int) and isinstance(completion, int)
    if prompt + completion > INT4_MAX:
        return None
    return ProviderUsage(prompt, completion)


def retry_after_note(value: object) -> tuple[int | None, str]:
    """Parse a ``retry-after`` header (decimal seconds only).

    Returns ``(milliseconds | None, note)`` with note in
    ``{"missing", "invalid", "too_large", "ok"}``.  HTTP-date forms, signs,
    exponents and anything else are ``invalid``.
    """
    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid"
    match = _RETRY_AFTER_RE.fullmatch(value.strip(" \t"))
    if match is None:
        return None, "invalid"
    whole, frac = match.group(1), match.group(2) or ""
    millis = int(whole) * 1000 + int(frac.ljust(3, "0")) if frac else int(whole) * 1000
    if millis > RETRY_AFTER_MAX_MS:
        return None, "too_large"
    return millis, "ok"


def parse_retry_after_ms(value: object) -> int | None:
    return retry_after_note(value)[0]


class ChatProvider(Protocol):
    """One non-streaming chat completion.  Implementations must not retry."""

    def generate(self, system_message: str, user_message: str) -> ProviderResult: ...


__all__: Final[list[str]] = [
    "INT4_MAX",
    "RETRY_AFTER_MAX_MS",
    "ChatProvider",
    "ProviderResult",
    "ProviderResultKind",
    "ProviderUsage",
    "SettlementAction",
    "completed",
    "normalize_usage",
    "parse_retry_after_ms",
    "rate_limited",
    "rejected",
    "retry_after_note",
    "settlement_action",
    "uncertain",
    "unusable",
]
