import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.core.deps import get_current_user
from app.core.rate_limit import limiter
from app.crud import document as crud
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag.indexing_hook import run_requested_indexing
from app.rag.indexing_service import request_indexing
from app.schemas.indexing import IndexingRequestResponse, IndexingStatusResponse

router = APIRouter()

logger = logging.getLogger("documind")

_EXTRACTION_UNFINISHED = "Text extraction has not finished yet"
_EXTRACTION_FAILED = "Text extraction failed for this document"
_NO_EXTRACTED_TEXT = "This document has no extracted text"
_STATE_CHANGED = "Indexing state changed; please retry"


def _read_status(db: Session, document_id: int, owner_id: int):
    """One SELECT returning every response field, or None if the document is gone.

    Plain columns and the three computed booleans come from a single statement, so
    they describe one moment. Selecting columns (not the Document entity) means no
    value is taken from the session's identity map. Each boolean is built so that it
    can never be NULL: ``and_`` with an ``IS NOT NULL`` term is false, not NULL, when
    the column is NULL, and EXISTS is always true or false.
    """
    now = func.clock_timestamp()  # the database clock, like request_indexing and claim_document
    statement = select(
        Document.id.label("document_id"),
        Document.indexing_status,
        Document.indexing_error,
        Document.indexing_attempt,
        Document.indexed_at,
        and_(
            Document.indexed_source_version.is_not(None),
            Document.indexed_source_version == Document.source_version,
        ).label("index_current"),
        exists()
        .where(DocumentChunk.document_id == Document.id)
        .correlate(Document)
        .label("published_chunks_present"),
        and_(
            Document.indexing_status == "indexing",
            Document.indexing_lease_expires_at.is_not(None),
            Document.indexing_lease_expires_at > now,
        ).label("in_progress"),
    ).where(Document.id == document_id, Document.owner_id == owner_id)
    return db.execute(statement).first()


@router.get("/documents/{document_id}/indexing", response_model=IndexingStatusResponse)
def get_indexing_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ownership gate: None for a missing document and for another user's alike.
    if crud.get_document(db, document_id, owner_id=current_user.id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    row = _read_status(db, document_id, current_user.id)
    if row is None:
        # Deleted between the gate and the snapshot: indistinguishable from "missing".
        raise HTTPException(status_code=404, detail="Document not found")

    return IndexingStatusResponse(**row._mapping)


def _already_current(row) -> bool:
    """Indexed, error-free, current and with chunk rows present: nothing to do."""
    return (
        row.indexing_status == "indexed"
        and row.indexing_error is None
        and row.index_current is True
        and row.published_chunks_present is True
    )


def _conflict_detail(processing_status: str | None) -> str:
    """The fixed 409 message for a document whose extraction is not completed.

    Valid only for a state that is not 'completed'. Calling it for 'completed', or for
    a value the table's check constraint does not allow, is a programming error.
    """
    if processing_status is None:
        return _NO_EXTRACTED_TEXT
    if processing_status in ("pending", "processing"):
        return _EXTRACTION_UNFINISHED
    if processing_status == "failed":
        return _EXTRACTION_FAILED
    raise ValueError("processing_status is not an extraction conflict")


def _close_session(session: Session, document_id: int, step: str) -> None:
    """Close a fresh session; a close failure after finished work is logged, not raised."""
    try:
        session.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Session cleanup failed after %s | document_id=%s error_type=%s",
            step,
            document_id,
            type(exc).__name__,
        )


def _request_in_fresh_session(document_id: int) -> bool:
    """request_indexing in its own short-lived session, closed before returning.

    The request session already holds the transaction opened by authentication, and
    request_indexing opens its own, so it can never use that session.
    """
    session = SessionLocal()
    try:
        return request_indexing(session, document_id)
    finally:
        _close_session(session, document_id, "request_indexing")


def _fresh_status(document_id: int, owner_id: int):
    """The status snapshot from a new session, so it reflects writes just committed."""
    session = SessionLocal()
    try:
        return _read_status(session, document_id, owner_id)
    finally:
        _close_session(session, document_id, "status read")


def _fresh_processing_status(document_id: int, owner_id: int):
    """The owner-filtered extraction state from a new session: a one-column Row, or None.

    None means the document is gone or not owned by the caller. A Row whose
    ``processing_status`` is None means the document exists and its extraction status
    is NULL. Returning the Row keeps those two cases distinguishable, which a bare
    ``str | None`` would not. It selects a column, not the entity, in a new session,
    so nothing can come from the request session's identity map.
    """
    session = SessionLocal()
    try:
        return session.execute(
            select(Document.processing_status).where(
                Document.id == document_id, Document.owner_id == owner_id
            )
        ).first()
    finally:
        _close_session(session, document_id, "extraction read")


def _respond(
    response: Response,
    status_code: int,
    outcome: str,
    row,
) -> IndexingRequestResponse:
    response.status_code = status_code
    return IndexingRequestResponse(
        outcome=outcome,
        indexing=IndexingStatusResponse(**row._mapping),
    )


@router.post(
    "/documents/{document_id}/indexing",
    response_model=IndexingRequestResponse,
    status_code=202,
)
@limiter.shared_limit("5/minute", scope="documents_indexing_post")
def request_document_indexing(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    owner_id = current_user.id

    # Ownership first, so a foreign document never reveals its extraction state.
    document = crud.get_document(db, document_id, owner_id=owner_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.processing_status != "completed":
        raise HTTPException(
            status_code=409,
            detail=_conflict_detail(document.processing_status),
        )

    snapshot = _read_status(db, document_id, owner_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if _already_current(snapshot):
        return _respond(response, 200, "already_indexed", snapshot)
    if snapshot.in_progress:
        return _respond(response, 202, "already_running", snapshot)

    queued = _request_in_fresh_session(document_id)
    fresh = _fresh_status(document_id, owner_id)
    if fresh is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if queued:
        background_tasks.add_task(run_requested_indexing, document_id)
        return _respond(response, 202, "queued", fresh)

    # request_indexing refused. Explain the refusal from current database state, never
    # from the ORM object read at the start: extraction may have changed since then.
    extraction = _fresh_processing_status(document_id, owner_id)
    if extraction is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if extraction.processing_status != "completed":
        raise HTTPException(
            status_code=409,
            detail=_conflict_detail(extraction.processing_status),
        )
    if _already_current(fresh):
        return _respond(response, 200, "already_indexed", fresh)
    if fresh.in_progress:
        return _respond(response, 202, "already_running", fresh)
    raise HTTPException(status_code=409, detail=_STATE_CHANGED)
