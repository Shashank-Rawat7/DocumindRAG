"""Idempotent chat-request registration (Component 12 / C8.1, step 2).

``register_request`` records that a user asked for one chat operation under one
idempotency key, or reports what already happened to that key. It writes only a
``chat_requests`` row: no attempt, no quota reservation and no message.

Transactions. Like every service in app/rag, the function is exactly one transaction
opened with ``with db.begin():``. The session must have no transaction in progress
(SQLAlchemy raises InvalidRequestError otherwise); the function commits on success
and rolls back on any error. It never commits or rolls back anything else.

Concurrency. The row is created with a single
``INSERT ... ON CONFLICT (owner_id, idempotency_key) DO NOTHING RETURNING id``, so the
database's unique constraint decides who creates it. Under READ COMMITTED a concurrent,
still-uncommitted insert of the same key makes the statement wait; once that
transaction commits it inserts nothing, and the SELECT that follows sees the winner's
row. There is no check-then-insert step, so duplicate rows cannot appear.

Outcomes (the stored row is never modified by a replay or a mismatch):
    CREATED                       a new ``in_progress`` row was inserted
    IN_PROGRESS_REPLAY            same key and hash, stored status in_progress
    SUCCEEDED_REPLAY              same key and hash, stored status succeeded
    FAILED_REPLAY                 same key and hash, stored status failed
    KEY_REUSED_DIFFERENT_REQUEST  the key exists with a different hash (any status)

Whether a failed request may be re-executed under the same key is not decided here:
FAILED_REPLAY only reports it.

Not done here: this function does not check that ``conversation_id`` belongs to
``owner_id`` (the foreign key does not either). Whoever acts on the target conversation
must check ownership. An owner_id that is not a user surfaces as the database's
IntegrityError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.chat_request import ChatRequest
from app.models.chat_request_attempt import ChatRequestAttempt  # noqa: F401  (see below)
from app.rag.chat_idempotency import ChatIdempotencyError, validate_idempotency_key

__all__ = [
    "RegistrationArgumentError",
    "RegistrationOutcome",
    "RegistrationResult",
    "register_request",
]


class RegistrationArgumentError(ChatIdempotencyError):
    """owner_id, request_sha256 or conversation_id has an invalid type or format."""


class RegistrationOutcome(str, Enum):
    CREATED = "created"
    IN_PROGRESS_REPLAY = "in_progress_replay"
    SUCCEEDED_REPLAY = "succeeded_replay"
    FAILED_REPLAY = "failed_replay"
    KEY_REUSED_DIFFERENT_REQUEST = "key_reused_different_request"


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    outcome: RegistrationOutcome
    request_id: int  # the new row for CREATED, otherwise the existing row for the key


# ChatRequest.attempts refers to ChatRequestAttempt by name, so that class must be imported
# before the first query or SQLAlchemy cannot configure the mappers. The import above does
# that without relying on the application having imported every model already.

_REPLAY_BY_STATUS = {
    "in_progress": RegistrationOutcome.IN_PROGRESS_REPLAY,
    "succeeded": RegistrationOutcome.SUCCEEDED_REPLAY,
    "failed": RegistrationOutcome.FAILED_REPLAY,
}

# Same format as ck_chat_requests_request_sha256_format (^[0-9a-f]{64}$).
_SHA256_PATTERN = r"[0-9a-f]{64}"


def register_request(
    db: Session,
    owner_id: int,
    idempotency_key: str,
    request_sha256: str,
    *,
    conversation_id: int | None,
) -> RegistrationResult:
    """Register the request, or report what the key already stands for.

    ``conversation_id`` is the target conversation, or None for a request that will
    create a new one (``creates_conversation`` is derived from it). It is required, with
    no default, so a forgotten target cannot silently mean "new conversation".
    """
    validate_idempotency_key(idempotency_key)
    if type(owner_id) is not int:
        raise RegistrationArgumentError("owner_id must be an integer")
    if (
        type(request_sha256) is not str
        or re.fullmatch(_SHA256_PATTERN, request_sha256) is None
    ):
        raise RegistrationArgumentError(
            "request_sha256 must be 64 lowercase hex characters"
        )
    if conversation_id is not None and type(conversation_id) is not int:
        raise RegistrationArgumentError("conversation_id must be None or an integer")

    with db.begin():
        created_id = db.execute(
            pg_insert(ChatRequest)
            .values(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                status="in_progress",
                creates_conversation=conversation_id is None,
                conversation_id=conversation_id,
            )
            .on_conflict_do_nothing(index_elements=["owner_id", "idempotency_key"])
            .returning(ChatRequest.id)
        ).scalar_one_or_none()
        if created_id is not None:
            return RegistrationResult(RegistrationOutcome.CREATED, created_id)
        existing = db.execute(
            select(
                ChatRequest.id, ChatRequest.request_sha256, ChatRequest.status
            ).where(
                ChatRequest.owner_id == owner_id,
                ChatRequest.idempotency_key == idempotency_key,
            )
        ).one()

    if existing.request_sha256 != request_sha256:
        return RegistrationResult(
            RegistrationOutcome.KEY_REUSED_DIFFERENT_REQUEST, existing.id
        )
    return RegistrationResult(_REPLAY_BY_STATUS[existing.status], existing.id)
