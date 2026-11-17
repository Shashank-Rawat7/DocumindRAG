"""Idempotency-key validation and canonical request hashing (Component 12 / C8.1, step 1).

Pure and standard-library only: no database, no session, no settings, no HTTP.

What is hashed (decision D9). The request hash covers exactly four top-level fields:

    operation        what the client asked for (non-empty string)
    conversation_id  None (new conversation) or the target conversation's integer id
    question         the question text, exactly as supplied
    options          every EFFECTIVE option that can change retrieval or generation

The idempotency key itself is NOT part of the hash: it is scoped by owner_id through
uq_chat_requests_owner_id_idempotency_key, and the hash is what detects a reused key
that now carries a different request. Tokens, timestamps and transport metadata are
not part of the payload either.

Canonical form: ``json.dumps(payload, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)`` encoded as UTF-8, then SHA-256, lowercase hex.
Keys are sorted recursively. Text is hashed as supplied: nothing is trimmed, case-folded
or Unicode-normalized, so canonically equivalent but differently encoded strings hash
differently.

Contracts the caller must keep (this module cannot enforce them):
  * ``options`` must already hold the effective value of every option, defaults
    included, so an omitted default and an explicit default hash the same.
  * Option values keep their Python type: ``1`` and ``1.0`` serialize differently and
    therefore hash differently, as do ``True`` and ``1``. Resolve each option to one
    consistent type before calling.
  * A tuple serializes exactly like a list (order preserved) and hashes the same.

Errors. Every failure raises a ChatIdempotencyError subclass. No message contains the
key, the question or any option value (the same rule as app/rag/retrieval.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import re

# Same pattern as ck_chat_requests_idempotency_key_format, without the ^ and $ anchors
# (re.fullmatch supplies them). tests/test_chat_idempotency.py compares this string
# with the CHECK text in the C7 model and migration so the two cannot drift apart.
IDEMPOTENCY_KEY_PATTERN = r"[A-Za-z0-9._:-]{1,128}"

__all__ = [
    "IDEMPOTENCY_KEY_PATTERN",
    "CanonicalRequestError",
    "ChatIdempotencyError",
    "IdempotencyKeyError",
    "canonical_request_sha256",
    "validate_idempotency_key",
]


class ChatIdempotencyError(Exception):
    """Base class for every error raised by this module."""


class IdempotencyKeyError(ChatIdempotencyError):
    """The idempotency key is not a string of 1-128 characters from [A-Za-z0-9._:-]."""


class CanonicalRequestError(ChatIdempotencyError):
    """An argument of canonical_request_sha256 has an unsupported type or value."""


def validate_idempotency_key(key: object) -> str:
    """Return ``key`` unchanged if it is valid, else raise IdempotencyKeyError.

    Exact ``str`` only (a subclass is rejected, like the other validators in
    app/rag). No stripping and no normalization: " a" and "a\\n" are invalid, as in the
    database CHECK.
    """
    if type(key) is not str:
        raise IdempotencyKeyError("idempotency key must be a string")
    if re.fullmatch(IDEMPOTENCY_KEY_PATTERN, key) is None:
        raise IdempotencyKeyError(
            "idempotency key must be 1-128 characters from A-Z, a-z, 0-9, '.', '_', ':' and '-'"
        )
    return key


def _check_json_value(value: object) -> None:
    """Raise CanonicalRequestError unless ``value`` is a supported JSON value.

    Exact types only, so subclasses (IntEnum, OrderedDict, str subclasses) and every
    other object are rejected rather than converted. May raise RecursionError for a
    cyclic or extremely deep structure; the caller turns that into a typed error.
    """
    kind = type(value)
    if value is None or kind is bool or kind is int or kind is str:
        return
    if kind is float:
        if not math.isfinite(value):
            raise CanonicalRequestError("options must not contain NaN or infinity")
        return
    if kind is list or kind is tuple:
        for item in value:
            _check_json_value(item)
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalRequestError("options keys must be strings")
            _check_json_value(item)
        return
    raise CanonicalRequestError("options contains a value of an unsupported type")


def canonical_request_sha256(
    operation: object,
    conversation_id: object,
    question: object,
    options: object,
) -> str:
    """Return the lowercase 64-character hex SHA-256 of the canonical request payload.

    See the module docstring for the exact contract. Nothing is converted: a wrong
    type is an error, never a ``str()`` of the value.
    """
    if type(operation) is not str or not operation:
        raise CanonicalRequestError("operation must be a non-empty string")
    if conversation_id is not None and type(conversation_id) is not int:
        raise CanonicalRequestError("conversation_id must be None or an integer")
    if type(question) is not str:
        raise CanonicalRequestError("question must be a string")
    if type(options) is not dict:
        raise CanonicalRequestError("options must be a dict")

    payload = {
        "operation": operation,
        "conversation_id": conversation_id,
        "question": question,
        "options": options,
    }
    try:
        _check_json_value(options)
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        data = text.encode("utf-8")
    except CanonicalRequestError:
        raise
    except RecursionError:
        raise CanonicalRequestError(
            "options are nested too deeply or contain a cycle"
        ) from None
    except UnicodeEncodeError:
        raise CanonicalRequestError(
            "request text is not valid Unicode (lone surrogate)"
        ) from None
    except ValueError:
        # e.g. an integer with more digits than Python's int-to-str limit allows.
        raise CanonicalRequestError("request cannot be serialized") from None
    return hashlib.sha256(data).hexdigest()
