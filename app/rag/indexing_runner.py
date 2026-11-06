"""Indexing orchestration for one document (Component 12 / C5.2).

``run_indexing`` performs ONE indexing attempt by combining the C5.1 database
operations (app/rag/indexing_service.py) with the pure in-memory build
(app/rag/indexing.py):

    claim -> load source -> page mapping -> build -> publish
                                   \\-> on an ordinary failure: record_failure

It does not queue (request_indexing), retry, schedule or heartbeat. The caller
queues the document first; a document that is not 'pending' is simply not claimed.

IMPORTANT FOR CALLERS: a returned IndexingRunResult is a report of what THIS run
did, never proof that the document is indexed. Inspect ``result.outcome``; the
documents row is the authority on the document's real state. Only PUBLISHED means
this run published. Do not call record_failure yourself.

Sessions and transactions. Every database step uses a fresh ``SessionLocal()``
that is closed when the step ends (each C5.1 function opens and finishes its own
transaction with ``db.begin()``, so the session must have none in progress).
No session, connection or transaction exists while pages are mapped, text is
tokenized or vectors are computed: those steps can take minutes and the only
thing protecting them is the lease.

Exceptions. Only ordinary ``Exception`` is handled: it is classified and recorded
through record_failure, and a structured result is returned. KeyboardInterrupt,
SystemExit and cancellation (BaseException) are never caught: they propagate, the
sessions are still closed, and the document stays 'indexing' until its lease
expires, which is the V1 recovery path. A database error while claiming also
propagates: nothing was claimed, so there is nothing to record.

Logging never contains exception messages, SQL parameters, document text or
source-code lines: only ids, the failure reason, the exception class name and
frame metadata (file name, function, line number).
"""
from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import storage
from app.core.database import SessionLocal
from app.models.document import Document
from app.rag.embedder import Embedder, get_embedder
from app.rag.indexing import DocumentIndexBuild, build_document_index
from app.rag.indexing_service import (
    FailureOutcome,
    FailureReason,
    IndexingClaim,
    PublishOutcome,
    claim_document,
    classify_failure,
    publish_build,
    record_failure,
)
from app.rag.pdf_pages import map_document_pages

logger = logging.getLogger("documind")

_T = TypeVar("_T")
_MAX_LOGGED_FRAMES = 8


# ---------- result ----------

class RunOutcome(str, Enum):
    PUBLISHED = "published"                         # this run published; the document is 'indexed'
    NOT_CLAIMED = "not_claimed"                     # nothing to do (not pending, missing, live lease...); no writes
    STALE = "stale"                                 # another attempt owns the document or its source changed
    GONE = "gone"                                   # the document was deleted
    FAILED = "failed"                               # the attempt failed and the failure was recorded
    FAILURE_NOT_RECORDED = "failure_not_recorded"   # the attempt failed and recording it failed too


@dataclass(frozen=True, slots=True)
class IndexingRunResult:
    """What one run did. Holds ids, enums and counts only: never document text."""

    document_id: int
    outcome: RunOutcome
    attempt: int | None = None                      # the attempt number this run claimed
    chunk_count: int | None = None                  # only when PUBLISHED
    reason: FailureReason | None = None             # FAILED and FAILURE_NOT_RECORDED
    failure_outcome: FailureOutcome | None = None   # FAILED: FAILED / RESTORED / DISCARDED


@dataclass(frozen=True, slots=True)
class _Source:
    text: str
    stored_name: str | None


# ---------- the run ----------

def run_indexing(document_id: int, *, embedder: Embedder | None = None) -> IndexingRunResult:
    """Run one indexing attempt for ``document_id`` and report what happened.

    ``embedder`` defaults to the process-wide get_embedder(), which is requested
    only after a successful claim. A substitute must provide source_units(),
    embed_passages(), signature_fields() and embedding_id, like Embedder.

    Raises: a database error while claiming; KeyboardInterrupt, SystemExit and
    other BaseException unchanged. Every other failure of the attempt is recorded
    and returned as a result.
    """
    started = time.monotonic()
    claim = _in_fresh_session(lambda session: claim_document(session, document_id))
    if claim is None:
        logger.info("Indexing not started, nothing to claim | document_id=%s", document_id)
        return IndexingRunResult(document_id=document_id, outcome=RunOutcome.NOT_CLAIMED)

    build: DocumentIndexBuild | None = None
    try:
        source = _in_fresh_session(lambda session: _read_source(session, claim.document_id))
        if source is None:
            outcome = PublishOutcome.GONE            # deleted after the claim: nothing to build or record
        else:
            build = _build(source, embedder)         # slow; no session, connection or transaction is open
            outcome = _in_fresh_session(lambda session: publish_build(session, claim, build))
    except Exception as exc:  # noqa: BLE001
        return _record_failed_attempt(claim, exc, started)

    # publish_build has returned (or the document was gone): nothing below can lead to record_failure.
    return _result_for_publish(claim, outcome, build, started)


# ---------- steps ----------

def _in_fresh_session(operation: Callable[[Session], _T]) -> _T:
    """Run ``operation(session)`` in a new session and close the session afterwards.

    Errors raised by ``operation`` propagate. If the operation already RETURNED
    and only closing the session fails, its (committed) result is kept and the
    cleanup error is logged: a finished database step is never turned into a failure.
    """
    returned = False
    value: _T | None = None
    try:
        with SessionLocal() as session:
            value = operation(session)
            returned = True
    except Exception as exc:
        if not returned:
            raise
        logger.warning(
            "Session cleanup failed after the operation completed | error_type=%s frames=%s",
            type(exc).__name__,
            _frames(exc),
        )
    return value  # type: ignore[return-value]


def _read_source(session: Session, document_id: int) -> _Source | None:
    """The document's text and stored file name, or None if it no longer exists."""
    row = session.execute(
        select(Document.extracted_text, Document.stored_name).where(Document.id == document_id)
    ).first()
    if row is None:
        return None
    return _Source(text=row.extracted_text or "", stored_name=row.stored_name)


def _build(source: _Source, embedder: Embedder | None) -> DocumentIndexBuild:
    """Page mapping and the in-memory index build. Touches no database session."""
    path = storage.get_file_path(source.stored_name) if source.stored_name else None
    page_mapping = map_document_pages(path, source.text)
    if embedder is None:
        embedder = get_embedder()
    return build_document_index(source.text, embedder=embedder, page_mapping=page_mapping)


def _result_for_publish(
    claim: IndexingClaim,
    outcome: PublishOutcome,
    build: DocumentIndexBuild | None,
    started: float,
) -> IndexingRunResult:
    elapsed = time.monotonic() - started
    if outcome is PublishOutcome.PUBLISHED:
        assert build is not None
        logger.info(
            "Indexing published | document_id=%s attempt=%s chunks=%s elapsed_seconds=%.1f",
            claim.document_id, claim.attempt, len(build.chunks), elapsed,
        )
        return IndexingRunResult(
            document_id=claim.document_id,
            outcome=RunOutcome.PUBLISHED,
            attempt=claim.attempt,
            chunk_count=len(build.chunks),
        )
    run_outcome = RunOutcome.STALE if outcome is PublishOutcome.STALE else RunOutcome.GONE
    logger.info(
        "Indexing not published | document_id=%s attempt=%s outcome=%s elapsed_seconds=%.1f",
        claim.document_id, claim.attempt, run_outcome.value, elapsed,
    )
    return IndexingRunResult(document_id=claim.document_id, outcome=run_outcome, attempt=claim.attempt)


def _record_failed_attempt(claim: IndexingClaim, exc: Exception, started: float) -> IndexingRunResult:
    """Classify ``exc``, record it through record_failure and report the result."""
    reason = classify_failure(exc)
    logger.warning(
        "Indexing failed | document_id=%s attempt=%s reason=%s error_type=%s frames=%s elapsed_seconds=%.1f",
        claim.document_id, claim.attempt, reason.name, type(exc).__name__, _frames(exc),
        time.monotonic() - started,
    )
    try:
        failure = _in_fresh_session(lambda session: record_failure(session, claim, reason))
    except Exception as record_exc:  # noqa: BLE001
        # The failure could not be stored. The document stays 'indexing' until its lease expires.
        logger.error(
            "Could not record indexing failure | document_id=%s attempt=%s reason=%s error_type=%s frames=%s",
            claim.document_id, claim.attempt, reason.name, type(record_exc).__name__, _frames(record_exc),
        )
        return IndexingRunResult(
            document_id=claim.document_id,
            outcome=RunOutcome.FAILURE_NOT_RECORDED,
            attempt=claim.attempt,
            reason=reason,
        )

    if failure is FailureOutcome.STALE:
        return IndexingRunResult(document_id=claim.document_id, outcome=RunOutcome.STALE, attempt=claim.attempt)
    if failure is FailureOutcome.GONE:
        return IndexingRunResult(document_id=claim.document_id, outcome=RunOutcome.GONE, attempt=claim.attempt)
    return IndexingRunResult(
        document_id=claim.document_id,
        outcome=RunOutcome.FAILED,
        attempt=claim.attempt,
        reason=reason,
        failure_outcome=failure,
    )


def _frames(exc: BaseException) -> str:
    """Innermost frames as 'file:function:line', never messages or source lines."""
    frames = [
        f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{line}"
        for frame, line in traceback.walk_tb(exc.__traceback__)
    ]
    return " > ".join(frames[-_MAX_LOGGED_FRAMES:]) or "-"