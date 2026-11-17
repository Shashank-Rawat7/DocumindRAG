"""Groq adapter for the chat provider contract (C9.3).

This is the only module that imports the Groq SDK and httpx.  It performs one
HTTP call per :meth:`GroqChatProvider.generate` (``max_retries=0``), then
classifies the outcome.  It never retries, sleeps, touches the database or
calls C8 lifecycle functions.

Classification rules (conservative: unknown means uncertain):

* HTTP 429 -> RATE_LIMITED (zero-charge), ``retry-after`` honoured if valid.
* HTTP 400/401/403/404/413/422 -> REJECTED / ``provider_rejected``.
* Any other HTTP status (including 408, 5xx) -> UNCERTAIN / ``provider_unavailable``.
* Transport failures whose cause is exactly ``httpx.ConnectError``,
  ``httpx.ConnectTimeout`` or ``httpx.PoolTimeout`` -> REJECTED /
  ``provider_unavailable`` (the request never reached the provider).
* Other timeouts -> UNCERTAIN / ``provider_timeout``; every other SDK or
  connection error -> UNCERTAIN / ``provider_unavailable``.
* Non-SDK exceptions (programming errors) propagate unchanged.
* A 200 is COMPLETED only with exactly one choice, ``finish_reason == "stop"``,
  an assistant message with non-blank string content.  Usage never decides
  publishability: missing or malformed usage yields ``usage=None`` and C8
  settles from the reservation.  Otherwise it is UNUSABLE
  (``invalid_response``), keeping usage if valid.
  Truncated answers (``finish_reason == "length"``) are never published.

The response body is parsed strictly from raw JSON so SDK type coercion cannot
turn ``true`` or ``5.0`` into usage values.  Reasoning content is never read,
returned, persisted or logged.

Timeouts: per-phase httpx timeouts (connect 3 s, write 3 s, pool 2 s, read
30 s) give a worst case of 38 s of *phase* time.  This is NOT a strict
wall-clock bound: the read timeout measures inactivity between chunks, so a
response that trickles bytes can run longer.  The lease arithmetic checked at
import (38 s + margin <= lease) covers the phase budget only.

A passing mock-transport test shows this adapter handles the SDK's behaviour
correctly; it does not show that Groq's live API accepts every parameter.
Live checks belong to C9.7.
"""

from __future__ import annotations

from typing import Any, Final

import groq
import httpx

from app.core.chat_config import (
    LEASE_MARGIN_SECONDS,
    LEASE_SECONDS,
    MAX_COMPLETION_TOKENS,
    PROVIDER_TIMEOUT_SECONDS,
    ChatSettings,
)
from app.rag import chat_error_codes as codes
from app.rag.chat_provider import (
    ProviderResult,
    completed,
    normalize_usage,
    rate_limited,
    rejected,
    retry_after_note,
    uncertain,
    unusable,
)

GROQ_BASE_URL: Final = "https://api.groq.com"
TEMPERATURE: Final = 0.2
REASONING_EFFORT: Final = "low"

READ_TIMEOUT_SECONDS: Final = PROVIDER_TIMEOUT_SECONDS
CONNECT_TIMEOUT_SECONDS: Final = 3
WRITE_TIMEOUT_SECONDS: Final = 3
POOL_TIMEOUT_SECONDS: Final = 2
WORST_CASE_PHASE_SECONDS: Final = (
    READ_TIMEOUT_SECONDS
    + CONNECT_TIMEOUT_SECONDS
    + WRITE_TIMEOUT_SECONDS
    + POOL_TIMEOUT_SECONDS
)

if WORST_CASE_PHASE_SECONDS + LEASE_MARGIN_SECONDS > LEASE_SECONDS:  # pragma: no cover
    raise RuntimeError("provider phase timeouts plus margin exceed the lease")

REJECTED_STATUSES: Final[frozenset[int]] = frozenset({400, 401, 403, 404, 413, 422})
_NEVER_SENT_CAUSES: Final = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def classify_exception(exc: groq.APIError) -> ProviderResult:
    """Map an SDK error to a provider result (never zero-charge by default).

    Only ``groq.APIError`` (the SDK wraps transport failures in it) is handled;
    anything else is a programming error and is rejected here, never classified.
    """
    if not isinstance(exc, groq.APIError):
        raise TypeError("classify_exception handles groq.APIError only")
    if isinstance(exc, groq.APIStatusError):
        status = exc.status_code
        if status == 429:
            headers = getattr(exc.response, "headers", None)
            raw = headers.get("retry-after") if headers is not None else None
            return rate_limited(retry_after_note(raw)[0])
        if status in REJECTED_STATUSES:
            return rejected(codes.PROVIDER_REJECTED)
        return uncertain(codes.PROVIDER_UNAVAILABLE)
    if isinstance(exc, groq.APIResponseValidationError):
        return unusable()
    if isinstance(exc, groq.APIConnectionError):
        if isinstance(exc.__cause__, _NEVER_SENT_CAUSES):
            return rejected(codes.PROVIDER_UNAVAILABLE)
        if isinstance(exc, groq.APITimeoutError):
            return uncertain(codes.PROVIDER_TIMEOUT)
        return uncertain(codes.PROVIDER_UNAVAILABLE)
    return uncertain(codes.PROVIDER_UNAVAILABLE)


def normalize_body(body: object) -> ProviderResult:
    """Normalize a decoded 200 response body.  Reasoning fields are never read."""
    if not isinstance(body, dict):
        return unusable()
    usage = normalize_usage(body.get("usage"))
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return unusable(usage)
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        return unusable(usage)
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return unusable(usage)
    content = message.get("content")
    if type(content) is not str or not content.strip():
        return unusable(usage)
    return completed(content, usage)


class GroqChatProvider:
    """One non-streaming Groq chat completion per call; no retries."""

    def __init__(
        self, settings: ChatSettings, *, http_client: httpx.Client | None = None
    ) -> None:
        if type(settings) is not ChatSettings:
            raise TypeError("settings must be ChatSettings")
        self._model = settings.model
        self._client = groq.Groq(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=GROQ_BASE_URL,
            max_retries=0,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
                write=WRITE_TIMEOUT_SECONDS,
                pool=POOL_TIMEOUT_SECONDS,
            ),
            http_client=http_client,
        )

    def __repr__(self) -> str:
        return f"GroqChatProvider(model={self._model!r})"

    def close(self) -> None:
        self._client.close()

    def generate(self, system_message: str, user_message: str) -> ProviderResult:
        for text in (system_message, user_message):
            if type(text) is not str or not text.strip():
                raise ValueError("messages must be non-blank strings")
        try:
            raw: Any = self._client.chat.completions.with_raw_response.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                stream=False,
            )
            try:
                body = raw.http_response.json()
            except ValueError:
                return unusable()
        except groq.APIError as exc:
            return classify_exception(exc)
        return normalize_body(body)
