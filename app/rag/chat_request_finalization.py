"""Request finalization (Component 12 / C8.5): publish a success, fail a request, replay.

``publish_success`` is the only way to settle a successful attempt. In ONE transaction it
(1) checks the request, (2) settles the attempt through the fenced helper, (3) creates or
locks the conversation, (4) inserts the user and assistant messages, and (5) marks the
request ``succeeded`` with its message ids (and the new conversation id). Either all of it
commits or none of it does: no conversation, no message, no attempt left ``succeeded`` and
no request update can survive a failure at any step.

Rollback. Inside ``with db.begin()`` nothing that must be undone is ever turned into a
normal return value. An outcome decided AFTER the attempt was settled
(``CONVERSATION_NOT_FOUND``) is raised as an internal exception, which rolls the transaction
back; it is converted to the public result only after the transaction has exited. Outcomes
returned from inside the transaction (``ALREADY_SUCCEEDED``, ``REQUEST_FAILED``,
``REQUEST_NOT_FOUND`` and the fence rejections) are decided before any write, so committing
that transaction changes nothing.

Fencing. The attempt is settled with the C8.4 fenced update (status running, lease live,
started). A stale, expired or taken-over worker is rejected before any conversation or
message is written.

Locks. The request row is locked first, then the attempt row (by the fenced UPDATE), then
the conversation row. ``start_attempt`` and ``fail_request`` also lock the request row first,
so publish, fail and start for one request run one after another.

``reject_and_fail_request`` is the only way to settle a provider-rejected attempt (charge 0):
in ONE transaction it checks owner, request and attempt together, settles the attempt through
the same fence, and fails the request with ``_set_failed`` (also used by ``fail_request``).
Same lock order. No committed state can hold a rejected attempt with an in-progress request.

Citations (C9.4b). ``publish_success`` also takes the resolved ``citations`` and the
``retrieved_chunks`` of THIS request (both derived by the caller from the same
``BuiltPrompt``). Before anything is written it checks, in memory, that every citation is
well formed (contiguous ordinals, unique markers 1..999, unique chunks) and cites a chunk
that is in ``retrieved_chunks`` and equal to it; a mismatch returns ``INVALID_CITATIONS``
with nothing written. After the message insert, in the SAME transaction, it locks the cited
documents and chunks ``FOR SHARE`` (documents first, then chunks, each ordered by id, the
order the indexer already uses: document row, then its chunks), requires every one to exist,
belong to the owner and agree with the retrieved chunk on identity and immutable metadata,
and inserts the ``message_citations`` rows. Any mismatch rolls the whole publish back
(``INVALID_CITATIONS`` after the rollback); a database failure rolls it back by raising. The
database cannot prove that retrieval happened: that guarantee belongs to the caller.

``fail_request_after_generation`` is the atomic failure for an answer that was generated but
cannot be published (for example ``INVALID_CITATIONS``): in ONE transaction, request row
first and the attempt second, it settles the attempt with the actual usage
(``failed_with_usage``) or, without usage, the reservation (``uncertain``), and fails the
request. A charge of 0 is never used, because the provider did generate.

Not done here: choosing titles, size limits (C9/C10); checking conversation
ownership BEFORE the provider call (the chat flow should, to avoid wasting tokens; this
module checks again at publish); settling the attempt when the conversation turns out to be
unavailable (publish rolls back; the caller then uses ``fail_with_usage`` or
``mark_uncertain`` and ``fail_request``).

Like every service in app/rag, each public function is one transaction opened with
``with db.begin():``; the session must have no transaction in progress.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import bindparam, exists, func, select, text, update
from sqlalchemy.orm import Session

from app.models import user as _user_model  # noqa: F401  (see the note below)
from app.models.chat_request import ChatRequest
from app.models.chat_request_attempt import ChatRequestAttempt
from app.rag.chat_attempt_lifecycle import (
    TransitionOutcome,
    settle_failed_in_transaction,
    settle_rejected_in_transaction,
    settle_succeeded_in_transaction,
    validate_optional_usage,
)
from app.rag.chat_citations import Citation
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.retrieval import RetrievedChunk

__all__ = [
    "FailAfterGenerationOutcome",
    "FailAfterGenerationResult",
    "FailOutcome",
    "FailResult",
    "FinalizationArgumentError",
    "PublishOutcome",
    "PublishResult",
    "RejectOutcome",
    "RejectResult",
    "RequestResult",
    "fail_request",
    "fail_request_after_generation",
    "get_request_result",
    "publish_success",
    "reject_and_fail_request",
]

# Same format as the attempt error codes (chat_attempt_lifecycle); a test keeps them equal.
_ERROR_CODE_PATTERN = r"[a-z][a-z0-9_]{0,63}"
_INT32_MAX = 2_147_483_647
_MAX_MARKER = 999  # the parser accepts 1 to 3 digits; ck_message_citations_marker_valid

# Importing the retrieval types pulls in the Document model, whose relationships name the
# User model; without ``app.models.user`` registered, the first ORM query in a process that
# imports only this module fails to configure the mappers (app.main registers it normally).
attempt = ChatRequestAttempt


class FinalizationArgumentError(ChatIdempotencyError):
    """An argument has an invalid type, format or value."""


class PublishOutcome(str, Enum):
    PUBLISHED = "published"
    ALREADY_SUCCEEDED = (
        "already_succeeded"  # replay: nothing written, stored ids returned
    )
    REQUEST_FAILED = "request_failed"  # the request was already finalized as failed
    REQUEST_NOT_FOUND = (
        "request_not_found"  # no such request/attempt pair for this owner
    )
    CONVERSATION_NOT_FOUND = (
        "conversation_not_found"  # missing or not the owner's; rolled back
    )
    NOT_RUNNING = "not_running"  # fence: the attempt is already settled or expired
    LEASE_EXPIRED = "lease_expired"  # fence: running but past its lease
    WRONG_PHASE = "wrong_phase"  # fence: the attempt never recorded llm_started_at
    INVALID_CITATIONS = "invalid_citations"  # provenance/metadata mismatch; nothing written or committed


class FailOutcome(str, Enum):
    FAILED = "failed"
    NOT_FOUND = "not_found"
    ALREADY_FINALIZED = "already_finalized"  # current_status says which
    ATTEMPT_RUNNING = "attempt_running"  # settle or expire the running attempt first


class RejectOutcome(str, Enum):
    REJECTED = (
        "rejected"  # attempt settled (charge 0) and request failed, in one commit
    )
    REQUEST_NOT_FOUND = (
        "request_not_found"  # no such owner/request/attempt triple; nothing written
    )
    ALREADY_FINALIZED = "already_finalized"  # request not in progress; nothing written
    NOT_RUNNING = "not_running"  # fence: the attempt is already settled or expired
    LEASE_EXPIRED = "lease_expired"  # fence: running but past its lease
    WRONG_PHASE = "wrong_phase"  # fence: the attempt never recorded llm_started_at


class FailAfterGenerationOutcome(str, Enum):
    FAILED = "failed"  # attempt settled (actual usage or reservation), request failed
    REQUEST_NOT_FOUND = (
        "request_not_found"  # no such owner/request/attempt triple; nothing written
    )
    ALREADY_FINALIZED = "already_finalized"  # request not in progress; nothing written
    NOT_RUNNING = "not_running"  # fence: the attempt is already settled or expired
    LEASE_EXPIRED = "lease_expired"  # fence: running but past its lease
    WRONG_PHASE = "wrong_phase"  # fence: the attempt never recorded llm_started_at


@dataclass(frozen=True, slots=True)
class FailAfterGenerationResult:
    outcome: FailAfterGenerationOutcome
    current_status: str | None = None  # ALREADY_FINALIZED only


@dataclass(frozen=True, slots=True)
class RejectResult:
    outcome: RejectOutcome
    current_status: str | None = None  # ALREADY_FINALIZED only


@dataclass(frozen=True, slots=True)
class PublishResult:
    outcome: PublishOutcome
    conversation_id: int | None = None  # PUBLISHED and ALREADY_SUCCEEDED
    user_message_id: int | None = None
    assistant_message_id: int | None = None


@dataclass(frozen=True, slots=True)
class FailResult:
    outcome: FailOutcome
    current_status: str | None = None  # ALREADY_FINALIZED only


@dataclass(frozen=True, slots=True)
class RequestResult:
    status: str
    conversation_id: int | None
    user_message_id: int | None
    assistant_message_id: int | None
    error_code: str | None


class _Rollback(Exception):
    """Raised inside the transaction to undo earlier writes; carries the public result."""

    def __init__(self, result: PublishResult) -> None:
        super().__init__()
        self.result = result


def _check_id(name: str, value: object) -> None:
    if type(value) is not int or value < 1 or value > _INT32_MAX:
        raise FinalizationArgumentError(f"{name} must be a positive integer")


def _check_content(name: str, value: object) -> None:
    # Mirrors ck_messages_content_not_blank (length(trim(content)) > 0, trim removes spaces
    # only) and rejects what PostgreSQL text cannot store (NUL, unencodable surrogates).
    if type(value) is not str or value.strip(" ") == "" or "\x00" in value:
        raise FinalizationArgumentError(f"{name} must be a non-blank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise FinalizationArgumentError(f"{name} must be a non-blank string") from None


def _check_error_code(value: object) -> None:
    if type(value) is not str or re.fullmatch(_ERROR_CODE_PATTERN, value) is None:
        raise FinalizationArgumentError("error_code has an invalid format")


def publish_success(
    db: Session,
    owner_id: int,
    request_id: int,
    attempt_id: int,
    *,
    user_content: str,
    assistant_content: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    citations: Sequence[Citation] = (),
    retrieved_chunks: Sequence[RetrievedChunk] = (),
) -> PublishResult:
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    _check_id("attempt_id", attempt_id)
    _check_content("user_content", user_content)
    _check_content("assistant_content", assistant_content)
    validate_optional_usage(prompt_tokens, completion_tokens)
    citations = _check_citation_arguments(citations, retrieved_chunks)
    retrieved = tuple(retrieved_chunks)

    try:
        with db.begin():
            return _publish(
                db,
                owner_id,
                request_id,
                attempt_id,
                user_content,
                assistant_content,
                prompt_tokens,
                completion_tokens,
                citations,
                retrieved,
            )
    except _Rollback as rollback:  # the transaction has exited (rolled back) by now
        return rollback.result


def _check_citation_arguments(
    citations: object, retrieved_chunks: object
) -> tuple[Citation, ...]:
    """Argument types and the one retrieved-set invariant that is a caller bug."""
    if type(citations) not in (tuple, list) or any(
        type(c) is not Citation for c in citations
    ):
        raise FinalizationArgumentError("citations must be a sequence of Citation")
    if type(retrieved_chunks) not in (tuple, list) or any(
        type(c) is not RetrievedChunk for c in retrieved_chunks
    ):
        raise FinalizationArgumentError(
            "retrieved_chunks must be a sequence of RetrievedChunk"
        )
    chunk_ids = [c.chunk_id for c in retrieved_chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise FinalizationArgumentError("retrieved_chunks must not repeat a chunk")
    return tuple(citations)


def _citations_match_retrieved(
    citations: tuple[Citation, ...], retrieved_chunks: tuple[RetrievedChunk, ...]
) -> bool:
    """Pure provenance check: well-formed citations of chunks retrieved for this request."""
    retrieved = {c.chunk_id: c for c in retrieved_chunks}
    if any(type(c.ordinal) is not int for c in citations):
        return False
    if [c.ordinal for c in citations] != list(range(1, len(citations) + 1)):
        return False
    markers = [c.marker for c in citations]
    if len(set(markers)) != len(markers):
        return False
    if any(type(m) is not int or not 1 <= m <= _MAX_MARKER for m in markers):
        return False
    chunk_ids = [c.chunk.chunk_id for c in citations]
    if len(set(chunk_ids)) != len(chunk_ids):
        return False
    return all(retrieved.get(c.chunk.chunk_id) == c.chunk for c in citations)


_CHUNK_FIELDS = (
    "document_id",
    "chunk_index",
    "start_offset",
    "end_offset",
    "page_start",
    "page_end",
    "span_sha256",
)


def _persist_citations(
    db: Session,
    owner_id: int,
    message_id: int,
    citations: tuple[Citation, ...],
) -> bool:
    """Verify the cited chunks against the database and insert the rows; False on mismatch.

    Runs inside the publish transaction. The cited documents are locked ``FOR SHARE`` first
    and the cited chunks second, each ordered by id, so a concurrent delete or re-index of
    a cited chunk waits for this transaction (or, if it won, makes the row vanish here and
    this returns False). Documents first matches the indexer (document row, then chunks).
    The owner is enforced on the document rows; a chunk must belong to a locked, owned
    document. Every database value must equal the retrieved chunk's value. Nothing is
    inferred from ownership alone.
    """
    if not citations:
        return True
    document_ids = sorted({c.chunk.document_id for c in citations})
    chunk_ids = sorted(c.chunk.chunk_id for c in citations)
    documents = {
        row.id: row
        for row in db.execute(
            text(
                "SELECT id, indexed_source_version, source_sha256 FROM documents"
                " WHERE id IN :ids AND owner_id = :owner_id ORDER BY id FOR SHARE"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": document_ids, "owner_id": owner_id},
        )
    }
    stored = {
        row.id: row
        for row in db.execute(
            text(
                "SELECT id, document_id, chunk_index, start_offset, end_offset,"
                " page_start, page_end, span_sha256 FROM document_chunks"
                " WHERE id IN :ids AND document_id IN :document_ids"
                " ORDER BY id FOR SHARE"
            ).bindparams(
                bindparam("ids", expanding=True),
                bindparam("document_ids", expanding=True),
            ),
            {"ids": chunk_ids, "document_ids": document_ids},
        )
    }
    for citation in citations:
        chunk = citation.chunk
        row = stored.get(chunk.chunk_id)
        document = documents.get(chunk.document_id)
        if row is None or document is None:
            return False
        if any(getattr(row, f) != getattr(chunk, f) for f in _CHUNK_FIELDS):
            return False
        if (
            document.indexed_source_version != chunk.source_version
            or document.source_sha256 != chunk.source_sha256
        ):
            return False
    insert = text(
        "INSERT INTO message_citations (message_id, ordinal, marker, chunk_id,"
        " document_id, chunk_index, start_offset, end_offset, page_start, page_end,"
        " span_sha256, source_version, source_sha256)"
        " VALUES (:message_id, :ordinal, :marker, :chunk_id, :document_id, :chunk_index,"
        " :start_offset, :end_offset, :page_start, :page_end, :span_sha256,"
        " :source_version, :source_sha256)"
    )
    for citation in citations:
        chunk = citation.chunk
        inserted = db.execute(
            insert,
            {
                "message_id": message_id,
                "ordinal": citation.ordinal,
                "marker": citation.marker,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "span_sha256": chunk.span_sha256,
                "source_version": chunk.source_version,
                "source_sha256": chunk.source_sha256,
            },
        )
        if inserted.rowcount != 1:
            raise RuntimeError("citation could not be inserted")  # rolls back
    return True


def _publish(
    db: Session,
    owner_id: int,
    request_id: int,
    attempt_id: int,
    user_content: str,
    assistant_content: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    citations: tuple[Citation, ...],
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> PublishResult:
    # 1. The request must be this owner's, own this attempt, and still be in progress.
    request = db.execute(
        select(
            ChatRequest.status,
            ChatRequest.creates_conversation,
            ChatRequest.conversation_id,
            ChatRequest.user_message_id,
            ChatRequest.assistant_message_id,
        )
        .join(attempt, attempt.request_id == ChatRequest.id)
        .where(
            ChatRequest.id == request_id,
            ChatRequest.owner_id == owner_id,
            attempt.id == attempt_id,
        )
        .with_for_update(of=ChatRequest)
    ).first()
    if request is None:
        return PublishResult(PublishOutcome.REQUEST_NOT_FOUND)
    if request.status == "succeeded":
        return PublishResult(
            PublishOutcome.ALREADY_SUCCEEDED,
            request.conversation_id,
            request.user_message_id,
            request.assistant_message_id,
        )
    if request.status != "in_progress":
        return PublishResult(PublishOutcome.REQUEST_FAILED)

    # 1b. Provenance, in memory, before any write: every citation must cite a chunk that was
    # retrieved for this request. Nothing has been written yet, so returning commits nothing.
    if not _citations_match_retrieved(citations, retrieved_chunks):
        return PublishResult(PublishOutcome.INVALID_CITATIONS)

    # 2. Fenced settlement of the attempt (a stale or expired worker stops here, no writes).
    settled = settle_succeeded_in_transaction(
        db, attempt_id, prompt_tokens, completion_tokens
    )
    if settled is not TransitionOutcome.APPLIED:
        if settled is TransitionOutcome.NOT_FOUND:
            return PublishResult(PublishOutcome.REQUEST_NOT_FOUND)
        return PublishResult(PublishOutcome(settled.value))

    # 3. The conversation: create it, or lock the owner's existing one.
    if request.creates_conversation:
        conversation_id = db.execute(
            text(
                "INSERT INTO conversations (owner_id) VALUES (:owner_id) RETURNING id"
            ),
            {"owner_id": owner_id},
        ).scalar_one()
        next_seq = 1
    else:
        conversation_id = db.execute(
            text(
                "SELECT id FROM conversations WHERE id = :conversation_id"
                " AND owner_id = :owner_id FOR UPDATE"
            ),
            {"conversation_id": request.conversation_id, "owner_id": owner_id},
        ).scalar_one_or_none()
        if conversation_id is None:
            # The attempt was settled above: roll that back too, report after the rollback.
            raise _Rollback(PublishResult(PublishOutcome.CONVERSATION_NOT_FOUND))
        next_seq = (
            db.execute(
                text(
                    "SELECT coalesce(max(seq), 0) FROM messages WHERE conversation_id = :c"
                ),
                {"c": conversation_id},
            ).scalar_one()
            + 1
        )

    # 4. The message pair (sequence numbers are protected by the conversation lock/insert).
    insert_message = text(
        "INSERT INTO messages (conversation_id, seq, role, content)"
        " VALUES (:c, :seq, :role, :content) RETURNING id"
    )
    user_message_id = db.execute(
        insert_message,
        {
            "c": conversation_id,
            "seq": next_seq,
            "role": "user",
            "content": user_content,
        },
    ).scalar_one()
    assistant_message_id = db.execute(
        insert_message,
        {
            "c": conversation_id,
            "seq": next_seq + 1,
            "role": "assistant",
            "content": assistant_content,
        },
    ).scalar_one()
    # 4b. The citations, in this transaction: checked against the database, then inserted.
    if not _persist_citations(db, owner_id, assistant_message_id, citations):
        raise _Rollback(PublishResult(PublishOutcome.INVALID_CITATIONS))
    if not request.creates_conversation:
        db.execute(
            text(
                "UPDATE conversations SET updated_at = clock_timestamp() WHERE id = :c"
            ),
            {"c": conversation_id},
        )

    # 5. Finalize the request (the row is locked, so this must match; if not, roll back).
    finalized = db.execute(
        update(ChatRequest)
        .where(ChatRequest.id == request_id, ChatRequest.status == "in_progress")
        .values(
            status="succeeded",
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            completed_at=func.clock_timestamp(),
        )
        .returning(ChatRequest.id)
    ).first()
    if finalized is None:
        raise RuntimeError("request could not be finalized")  # rolls everything back
    return PublishResult(
        PublishOutcome.PUBLISHED, conversation_id, user_message_id, assistant_message_id
    )


def fail_request(
    db: Session, owner_id: int, request_id: int, error_code: str
) -> FailResult:
    """Finalize an in-progress request as failed. Refused while any attempt is running.

    The request row is locked first (``start_attempt`` and ``publish_success`` lock it too),
    then the running-attempt check runs as a fresh statement, so a start or a publish that
    committed first is always seen. Nothing but the request row is written; a failed
    new-conversation request leaves no conversation behind.
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    _check_error_code(error_code)
    with db.begin():
        status = db.execute(
            select(ChatRequest.status)
            .where(ChatRequest.id == request_id, ChatRequest.owner_id == owner_id)
            .with_for_update()
        ).scalar_one_or_none()
        if status is None:
            return FailResult(FailOutcome.NOT_FOUND)
        if status != "in_progress":
            return FailResult(FailOutcome.ALREADY_FINALIZED, status)
        running = db.execute(
            select(
                exists().where(
                    attempt.request_id == request_id, attempt.status == "running"
                )
            )
        ).scalar_one()
        if running:
            return FailResult(FailOutcome.ATTEMPT_RUNNING)
        _set_failed(db, request_id, error_code)
        return FailResult(FailOutcome.FAILED)


def _set_failed(db: Session, request_id: int, error_code: str) -> bool:
    """Mark an in-progress request failed, inside the caller's transaction.

    The one place that decides what a failed request looks like. The caller must already
    hold the request row lock. Returns whether a row was updated.
    """
    return (
        db.execute(
            update(ChatRequest)
            .where(ChatRequest.id == request_id, ChatRequest.status == "in_progress")
            .values(
                status="failed",
                error_code=error_code,
                completed_at=func.clock_timestamp(),
            )
            .returning(ChatRequest.id)
        ).first()
        is not None
    )


def reject_and_fail_request(
    db: Session, owner_id: int, request_id: int, attempt_id: int, error_code: str
) -> RejectResult:
    """Settle a provider-rejected attempt (charge 0) and fail its request, atomically.

    One transaction, request row locked first and the attempt row second (the same order as
    ``publish_success``, ``fail_request`` and ``start_attempt``):

    1. The request must be this owner's AND own this attempt (one joined, locking SELECT;
       a mismatch is ``REQUEST_NOT_FOUND`` and nothing is written).
    2. The request must still be in progress, else ``ALREADY_FINALIZED``.
    3. The attempt is settled through the C8.4 fenced update (running, lease live, started);
       a stale, expired or taken-over worker stops here with no writes.
    4. The request is marked failed with the same code.

    Either both rows change and commit together, or neither does. The same ``error_code``
    is stored on the attempt and the request (it is what a replay returns).
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    with db.begin():
        status = db.execute(
            select(ChatRequest.status)
            .join(attempt, attempt.request_id == ChatRequest.id)
            .where(
                ChatRequest.id == request_id,
                ChatRequest.owner_id == owner_id,
                attempt.id == attempt_id,
            )
            .with_for_update(of=ChatRequest)
        ).scalar_one_or_none()
        if status is None:
            return RejectResult(RejectOutcome.REQUEST_NOT_FOUND)
        if status != "in_progress":
            return RejectResult(RejectOutcome.ALREADY_FINALIZED, status)
        settled = settle_rejected_in_transaction(db, attempt_id, error_code)
        if settled is not TransitionOutcome.APPLIED:
            if settled is TransitionOutcome.NOT_FOUND:
                return RejectResult(RejectOutcome.REQUEST_NOT_FOUND)
            return RejectResult(RejectOutcome(settled.value))
        if not _set_failed(db, request_id, error_code):
            raise RuntimeError("request could not be failed")  # rolls everything back
        return RejectResult(RejectOutcome.REJECTED)


def fail_request_after_generation(
    db: Session,
    owner_id: int,
    request_id: int,
    attempt_id: int,
    error_code: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> FailAfterGenerationResult:
    """Settle an attempt whose generated answer cannot be published, and fail its request.

    One transaction, request row locked first and the attempt row second (the same order as
    ``publish_success``, ``fail_request``, ``reject_and_fail_request`` and
    ``start_attempt``):

    1. The request must be this owner's AND own this attempt (one joined, locking SELECT;
       a mismatch is ``REQUEST_NOT_FOUND`` and nothing is written).
    2. The request must still be in progress, else ``ALREADY_FINALIZED``.
    3. The attempt is settled through the C8.4 fenced update (running, lease live, started):
       ``failed_with_usage`` charging the actual usage, or, when the provider reported no
       usage (both None), ``uncertain`` charging the reservation. Never a charge of 0.
    4. The request is marked failed with the same code.

    Either both rows change and commit together, or neither does. A stale, expired or
    taken-over worker stops at step 3 with no writes. No committed state can hold a settled
    attempt with an in-progress request through this path.
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    validate_optional_usage(prompt_tokens, completion_tokens)
    with db.begin():
        status = db.execute(
            select(ChatRequest.status)
            .join(attempt, attempt.request_id == ChatRequest.id)
            .where(
                ChatRequest.id == request_id,
                ChatRequest.owner_id == owner_id,
                attempt.id == attempt_id,
            )
            .with_for_update(of=ChatRequest)
        ).scalar_one_or_none()
        if status is None:
            return FailAfterGenerationResult(
                FailAfterGenerationOutcome.REQUEST_NOT_FOUND
            )
        if status != "in_progress":
            return FailAfterGenerationResult(
                FailAfterGenerationOutcome.ALREADY_FINALIZED, status
            )
        settled = settle_failed_in_transaction(
            db, attempt_id, error_code, prompt_tokens, completion_tokens
        )
        if settled is not TransitionOutcome.APPLIED:
            if settled is TransitionOutcome.NOT_FOUND:
                return FailAfterGenerationResult(
                    FailAfterGenerationOutcome.REQUEST_NOT_FOUND
                )
            return FailAfterGenerationResult(FailAfterGenerationOutcome(settled.value))
        if not _set_failed(db, request_id, error_code):
            raise RuntimeError("request could not be failed")  # rolls everything back
        return FailAfterGenerationResult(FailAfterGenerationOutcome.FAILED)


def get_request_result(
    db: Session, owner_id: int, request_id: int
) -> RequestResult | None:
    """Read-only stored state of a request, or None if it is not this owner's.

    For a succeeded request this is the result to replay: the pointers are NULL if the
    conversation or messages were deleted afterwards.
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    with db.begin():
        row = db.execute(
            select(
                ChatRequest.status,
                ChatRequest.conversation_id,
                ChatRequest.user_message_id,
                ChatRequest.assistant_message_id,
                ChatRequest.error_code,
            ).where(ChatRequest.id == request_id, ChatRequest.owner_id == owner_id)
        ).first()
    if row is None:
        return None
    return RequestResult(*row)
