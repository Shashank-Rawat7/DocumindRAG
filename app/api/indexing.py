from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.crud import document as crud
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.schemas.indexing import IndexingStatusResponse

router = APIRouter()


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