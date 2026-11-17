"""Chat (C9) configuration: validated settings and fixed limits.

Design rules (C9.1):

* Nothing here reads the environment at import time. ``load_chat_settings`` is a pure
  function of the mapping it is given (the process environment by default), so importing
  this module never requires chat variables and never imports ``app.core.config``.
* ``CHAT_DAILY_TOKEN_LIMIT`` and ``GROQ_API_KEY`` are required and have no default. A
  missing or invalid value raises ``ChatConfigError``; chat must not run without them.
* The variables are read from the *process environment only*. They must NOT be put in the
  ``.env`` file: the application's existing ``Settings`` forbids unknown keys, so an extra
  key there makes ``Settings()`` fail at import.

Limits that are not environment-driven are module constants.

MVP limitation (explicit): ``PER_REQUEST_TOKEN_CEILING`` bounds a *single* request only. It
is a sanity check, not a safeguard for Groq's tokens-per-minute (TPM) limit, which is shared
across all requests. Concurrent requests can exceed TPM even when each one respects the
ceiling. The first version accepts that and relies on 429 handling (one bounded retry, see
``RETRY_AFTER_MAX_SECONDS``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import SecretStr

ENV_DAILY_TOKEN_LIMIT = "CHAT_DAILY_TOKEN_LIMIT"
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_MODEL = "CHAT_MODEL"

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Retrieval and generation limits (approved C9 decisions D1-D4).
TOP_K = 4
MAX_COMPLETION_TOKENS = 1024
PROMPT_TOKEN_BUDGET = 3500
PROVIDER_TIMEOUT_SECONDS = 30
LEASE_SECONDS = 60
# A 429 is retried once only if Groq's retry-after is present and within [0, this] seconds.
RETRY_AFTER_MAX_SECONDS = 15

# The reservation is estimate(full prompt) + MAX_COMPLETION_TOKENS, and the estimate is
# capped by PROMPT_TOKEN_BUDGET, so no request reserves more than this.
MAX_REQUEST_RESERVATION = PROMPT_TOKEN_BUDGET + MAX_COMPLETION_TOKENS
# Per-request sanity ceiling (NOT a TPM safeguard; see the module docstring).
PER_REQUEST_TOKEN_CEILING = 6000
# Margin the lease must keep over the provider timeout.
LEASE_MARGIN_SECONDS = 15

# Token columns are 32-bit integers in practice; keep the configured limit inside int4.
MAX_DAILY_TOKEN_LIMIT = 2**31 - 1

_DAILY_LIMIT_PATTERN = re.compile(r"[1-9][0-9]*")
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:\-]{0,99}")
_WHITESPACE = re.compile(r"\s")


class ChatConfigError(Exception):
    """Chat configuration is missing or invalid. Messages name variables, never values."""


@dataclass(frozen=True, slots=True)
class ChatSettings:
    """Validated chat settings. ``groq_api_key`` is a ``SecretStr`` (masked in repr/str)."""

    groq_api_key: SecretStr
    daily_token_limit: int
    model: str


def _check_constants() -> None:
    # Explicit raises, not ``assert``: asserts disappear under ``python -O``.
    if LEASE_SECONDS < PROVIDER_TIMEOUT_SECONDS + LEASE_MARGIN_SECONDS:
        raise RuntimeError(
            "chat lease must outlast the provider timeout plus the margin"
        )
    if MAX_REQUEST_RESERVATION > PER_REQUEST_TOKEN_CEILING:
        raise RuntimeError(
            "maximum request reservation exceeds the per-request ceiling"
        )
    if MAX_DAILY_TOKEN_LIMIT < MAX_REQUEST_RESERVATION:
        raise RuntimeError("daily limit bound is below the maximum request reservation")


_check_constants()


def _read_daily_limit(environ: Mapping[str, str], problems: list[str]) -> int | None:
    raw = environ.get(ENV_DAILY_TOKEN_LIMIT)
    if raw is None:
        problems.append(f"{ENV_DAILY_TOKEN_LIMIT} is required and has no default")
        return None
    if type(raw) is not str:
        problems.append(f"{ENV_DAILY_TOKEN_LIMIT} must be a string")
        return None
    if raw == "":
        problems.append(f"{ENV_DAILY_TOKEN_LIMIT} must not be empty")
        return None
    if _DAILY_LIMIT_PATTERN.fullmatch(raw) is None:
        problems.append(
            f"{ENV_DAILY_TOKEN_LIMIT} must be a positive base-10 integer with ASCII digits "
            "only (no sign, spaces, separators, decimal point or leading zeros)"
        )
        return None
    value = int(raw)
    if value > MAX_DAILY_TOKEN_LIMIT:
        problems.append(
            f"{ENV_DAILY_TOKEN_LIMIT} must be at most {MAX_DAILY_TOKEN_LIMIT}"
        )
        return None
    if value < MAX_REQUEST_RESERVATION:
        problems.append(
            f"{ENV_DAILY_TOKEN_LIMIT} must be at least {MAX_REQUEST_RESERVATION}, the largest "
            "single-request reservation, otherwise every request would be denied"
        )
        return None
    return value


def _read_api_key(environ: Mapping[str, str], problems: list[str]) -> SecretStr | None:
    raw = environ.get(ENV_GROQ_API_KEY)
    if raw is None:
        problems.append(f"{ENV_GROQ_API_KEY} is required")
        return None
    if type(raw) is not str:
        problems.append(f"{ENV_GROQ_API_KEY} must be a string")
        return None
    if raw == "":
        problems.append(f"{ENV_GROQ_API_KEY} must not be empty")
        return None
    if _WHITESPACE.search(raw) is not None:
        problems.append(
            f"{ENV_GROQ_API_KEY} must not contain whitespace (check for a pasted newline)"
        )
        return None
    return SecretStr(raw)


def _read_model(environ: Mapping[str, str], problems: list[str]) -> str | None:
    raw = environ.get(ENV_MODEL)
    if raw is None:
        return DEFAULT_MODEL
    if type(raw) is not str:
        problems.append(f"{ENV_MODEL} must be a string")
        return None
    if _MODEL_PATTERN.fullmatch(raw) is None:
        problems.append(
            f"{ENV_MODEL} must be 1-100 characters from letters, digits and . _ / : - "
            "when set (unset it to use the default)"
        )
        return None
    return raw


def load_chat_settings(environ: Mapping[str, str] | None = None) -> ChatSettings:
    """Validate chat settings; raise ``ChatConfigError`` naming every problem found."""
    source = os.environ if environ is None else environ
    problems: list[str] = []
    daily_limit = _read_daily_limit(source, problems)
    api_key = _read_api_key(source, problems)
    model = _read_model(source, problems)
    if problems or daily_limit is None or api_key is None or model is None:
        raise ChatConfigError("invalid chat configuration: " + "; ".join(problems))
    return ChatSettings(
        groq_api_key=api_key, daily_token_limit=daily_limit, model=model
    )
