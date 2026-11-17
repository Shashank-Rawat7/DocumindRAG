"""Stable error-code vocabulary for the chat pipeline (C9).

These strings are persisted by the C8 lifecycle (``error_code`` columns) and
replayed to clients, so they are a contract: add codes deliberately and never
rename one.  Every code must satisfy the C8 pattern ``[a-z][a-z0-9_]{0,63}``.
"""

from __future__ import annotations

import re
from typing import Final

CONVERSATION_NOT_FOUND: Final = "conversation_not_found"
NO_CONTEXT: Final = "no_context"
QUOTA_EXCEEDED: Final = "quota_exceeded"
PROVIDER_RATE_LIMITED: Final = "provider_rate_limited"
PROVIDER_TIMEOUT: Final = "provider_timeout"
PROVIDER_UNAVAILABLE: Final = "provider_unavailable"
PROVIDER_REJECTED: Final = "provider_rejected"
INVALID_RESPONSE: Final = "invalid_response"
LEASE_EXPIRED: Final = "lease_expired"
INTERNAL_ERROR: Final = "internal_error"

ALL_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        CONVERSATION_NOT_FOUND,
        NO_CONTEXT,
        QUOTA_EXCEEDED,
        PROVIDER_RATE_LIMITED,
        PROVIDER_TIMEOUT,
        PROVIDER_UNAVAILABLE,
        PROVIDER_REJECTED,
        INVALID_RESPONSE,
        LEASE_EXPIRED,
        INTERNAL_ERROR,
    }
)

# Same pattern the C8 lifecycle enforces on persisted error codes.
ERROR_CODE_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]{0,63}")

for _code in ALL_ERROR_CODES:
    if ERROR_CODE_PATTERN.fullmatch(_code) is None:  # pragma: no cover - import guard
        raise RuntimeError("chat error code violates the C8 pattern")
del _code
