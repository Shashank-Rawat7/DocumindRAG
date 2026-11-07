"""Automatic indexing after text extraction (Component 12 / C5.3).

``index_after_extraction`` is meant to run as a FastAPI background task registered
AFTER ``process_document`` on the same BackgroundTasks object. Starlette runs those
tasks one after another, so by the time this function starts, process_document has
returned and its final commit (extracted_text + 'completed', or 'failed') is visible.

    request_indexing  ->  run_indexing
    (fresh session)       (its own fresh sessions)

The gate is request_indexing itself: its single UPDATE requires
processing_status = 'completed', so a document whose extraction failed (or that does
not exist) is refused and nothing else happens. The hook adds no second check.

What this is NOT. request_indexing means "(re)queue". Calling this function again
for a document that is already 'indexed' or 'failed' queues it again and publishes
again. The upload endpoint calls it once per upload, for a document that was never
queued, so that does not happen in practice; the function is not an "index once"
guard and must not be described as one.

Recovery gap (V1, accepted until C5.4 / C5.5). Nothing here retries or schedules.
If this task never runs, or fails before the runner claims the document, the document
stays 'completed' with indexing_status NULL or 'pending'; lease expiry does not help
because it was never claimed. A later request / run (the C5.4 endpoint or the C5.5
backfill) is the recovery. An attempt that was claimed and then lost (process death,
FAILURE_NOT_RECORDED) stays 'indexing' until its lease expires and can then be
taken over.

Failure handling. The function never raises an ordinary Exception: the upload
response has long been sent and there is nobody to receive it. KeyboardInterrupt,
SystemExit and other BaseException propagate, as in the runner. The hook holds no session
while run_indexing works; the runner manages its own sessions. Logs hold ids, enum names,
counts, the exception class name and frame metadata only: never exception messages,
SQL parameters or document text.

Manual indexing (C5.4.2). ``run_requested_indexing`` is the background task of
POST /documents/{id}/indexing. That endpoint has already queued the document with
request_indexing, so the wrapper calls run_indexing only: it never requeues and opens
no session itself. The runner manages its own sessions. run_indexing claims only a
'pending' row, so a task that starts late finds nothing to claim and does not publish
again. Failure handling is the same as above, and its log messages say
"Requested indexing" so manual runs can be told apart from upload-triggered ones.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from app.core.database import SessionLocal
from app.rag.indexing_runner import IndexingRunResult, RunOutcome, run_indexing
from app.rag.indexing_service import request_indexing

logger = logging.getLogger("documind")

_MAX_LOGGED_FRAMES = 8

# FAILED is already logged as a warning with details by the runner. FAILURE_NOT_RECORDED
# is the one outcome that leaves the document 'indexing' with nobody to fix it until
# its lease expires, so it is an error. An outcome missing from this table is logged
# as a warning.
_RESULT_LOG_LEVELS = {
    RunOutcome.PUBLISHED: logging.INFO,
    RunOutcome.NOT_CLAIMED: logging.INFO,
    RunOutcome.STALE: logging.INFO,
    RunOutcome.GONE: logging.INFO,
    RunOutcome.FAILED: logging.WARNING,
    RunOutcome.FAILURE_NOT_RECORDED: logging.ERROR,
}


def index_after_extraction(document_id: int) -> None:
    """Queue ``document_id`` for indexing and run one indexing attempt.

    Does nothing for a document that is not 'completed' or is being indexed under a
    live lease. Never raises an ordinary Exception; BaseException propagates.
    """
    try:
        if not _request(document_id):
            logger.info("Indexing not queued | document_id=%s", document_id)
            return
        result = run_indexing(document_id)
        _log_result(result)
    # Background-task boundary: log unexpected ordinary failures without
    # allowing them to escape after the upload response has been sent.
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Automatic indexing failed | document_id=%s error_type=%s frames=%s",
            document_id,
            type(exc).__name__,
            _frames(exc),
        )
        return



def run_requested_indexing(document_id: int) -> None:
    """Run one indexing attempt for a document the manual endpoint just queued.

    Calls run_indexing only. The endpoint has already moved the document to 'pending'
    with request_indexing, and run_indexing claims only a 'pending' row, so a task that
    starts late finds nothing to claim (NOT_CLAIMED) and does not publish again. This
    function never requeues and opens no session itself. The runner manages its own
    sessions. Never raises an ordinary Exception; BaseException propagates.
    """
    try:
        result = run_indexing(document_id)
        _log_requested_result(result)
    # Background-task boundary: the response has long been sent.
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Requested indexing failed | document_id=%s error_type=%s frames=%s",
            document_id,
            type(exc).__name__,
            _frames(exc),
        )



def _request(document_id: int) -> bool:
    """request_indexing in its own session, closed before this function returns.

    request_indexing commits its own transaction. If it already returned and only
    closing the session fails, its result is kept and the cleanup error is logged:
    a finished (committed) queueing step is not turned into a failure.
    """
    session = SessionLocal()
    try:
        return request_indexing(session, document_id)
    finally:
        try:
            session.close()
        # Cleanup must not overwrite the result of an already completed request.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Session cleanup failed after request_indexing | document_id=%s error_type=%s frames=%s",
                document_id,
                type(exc).__name__,
                _frames(exc),
            )


def _log_result(result: IndexingRunResult) -> None:
    logger.log(
        _RESULT_LOG_LEVELS.get(result.outcome, logging.WARNING),
        "Automatic indexing finished | document_id=%s outcome=%s attempt=%s chunks=%s reason=%s failure=%s",
        result.document_id,
        result.outcome.value,
        result.attempt,
        result.chunk_count,
        result.reason.name if result.reason is not None else "-",
        result.failure_outcome.name if result.failure_outcome is not None else "-",
    )

def _log_requested_result(result: IndexingRunResult) -> None:
    logger.log(
        _RESULT_LOG_LEVELS.get(result.outcome, logging.WARNING),
        "Requested indexing finished | document_id=%s outcome=%s attempt=%s chunks=%s reason=%s failure=%s",
        result.document_id,
        result.outcome.value,
        result.attempt,
        result.chunk_count,
        result.reason.name if result.reason is not None else "-",
        result.failure_outcome.name if result.failure_outcome is not None else "-",
    )


def _frames(exc: BaseException) -> str:
    """Innermost frames as 'file:function:line', never messages or source lines."""
    frames = [
        f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{line}"
        for frame, line in traceback.walk_tb(exc.__traceback__)
    ]
    return " > ".join(frames[-_MAX_LOGGED_FRAMES:]) or "-"
