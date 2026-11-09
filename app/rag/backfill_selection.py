"""Read-only selection of backfill candidates (Component 12 / C5.5, CP2).

One SELECT returns, for every document whose text extraction is completed, the
metadata the pure classifier in app/rag/backfill.py needs, as DocumentState values.
Nothing is written, nothing is committed, and the extracted text is never selected:
``has_text`` is a boolean computed by PostgreSQL, so only that boolean reaches Python.

The caller owns the session or connection and its transaction. To make the read-only
intent a database guarantee, a caller can hand in a connection configured BEFORE its
first statement (psycopg2 cannot change the setting inside an open transaction):

    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        states = select_document_states(connection)

PDF source files are not checked here. A PDF-like document is returned with
``SourceStatus.PDF_UNCHECKED``, which the classifier treats as blocked until a later
step resolves it, so this module can never cause a PDF to be indexed by itself.
"""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Select, and_, case, exists, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User  # noqa: F401  (registers the mapper that Document.owner refers to)
from app.rag.backfill import DocumentState, LeaseState, SourceStatus, is_pdf_like

_LEASE_NONE = LeaseState.NONE.value
_LEASE_LIVE = LeaseState.LIVE.value
_LEASE_EXPIRED = LeaseState.EXPIRED.value


def _clean_ids(document_ids: Sequence[int] | None) -> tuple[int, ...] | None:
    """Validate an id filter: None stays None; otherwise a tuple of real ints.

    ``bool`` is rejected (it is an int subclass), and so is any other type, so a stray
    string such as "12" cannot be iterated into the characters "1" and "2".
    """
    if document_ids is None:
        return None
    ids = tuple(document_ids)
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"document ids must be ints, not {type(value).__name__}")
    return ids


def build_selection_statement(document_ids: Sequence[int] | None = None) -> Select:
    """The selection statement. Building it has no side effects and touches no database."""
    now = func.clock_timestamp()  # the database clock, like request_indexing and claim_document

    lease = case(
        (Document.indexing_lease_expires_at.is_(None), _LEASE_NONE),
        (Document.indexing_lease_expires_at > now, _LEASE_LIVE),
        else_=_LEASE_EXPIRED,
    ).label("lease")

    # False, never NULL: "IS NOT NULL" is false for a NULL text, and false AND NULL is false.
    has_text = and_(
        Document.extracted_text.is_not(None),
        func.octet_length(Document.extracted_text) > 0,
    ).label("has_text")

    # EXISTS is always true or false, and runs inside the same statement (no N+1).
    chunks_present = (
        exists()
        .where(DocumentChunk.document_id == Document.id)
        .correlate(Document)
        .label("chunks_present")
    )

    statement = (
        select(
            Document.id.label("document_id"),
            Document.processing_status,
            Document.indexing_status,
            Document.indexing_error,
            lease,
            Document.source_version,
            Document.indexed_source_version,
            Document.indexed_embedding_id,
            has_text,
            chunks_present,
            Document.filename,
            Document.stored_name,
            Document.content_type,
        )
        .where(Document.processing_status == "completed")
        .order_by(Document.id)
    )

    ids = _clean_ids(document_ids)
    if ids is not None:
        statement = statement.where(Document.id.in_(ids))
    return statement


def _flag(value: object, name: str) -> bool:
    """A computed boolean, accepted only if it really is a bool.

    They cannot be NULL by construction. Coercing None to False would turn a current
    document into a stale one (no chunks), so a surprise raises instead.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def _state_from_row(row) -> DocumentState:
    """Map one selection row to a DocumentState. Pure: attribute reads only."""
    source = (
        SourceStatus.PDF_UNCHECKED
        if is_pdf_like(row.filename, row.stored_name, row.content_type)
        else SourceStatus.NOT_PDF
    )
    return DocumentState(
        document_id=row.document_id,
        processing_status=row.processing_status,
        indexing_status=row.indexing_status,
        indexing_error=row.indexing_error,
        lease=LeaseState(row.lease),  # an unexpected value raises ValueError
        source_version=row.source_version,
        indexed_source_version=row.indexed_source_version,
        indexed_embedding_id=row.indexed_embedding_id,
        has_text=_flag(row.has_text, "has_text"),
        chunks_present=_flag(row.chunks_present, "chunks_present"),
        filename=row.filename,
        stored_name=row.stored_name,
        content_type=row.content_type,
        source=source,
    )


def select_document_states(
    conn: Session | Connection,
    *,
    document_ids: Sequence[int] | None = None,
) -> list[DocumentState]:
    """Run the one read-only SELECT and return the states, sorted by document id.

    ``document_ids=None`` returns every completed document. A list narrows the result;
    an id that is unknown, deleted or no longer completed is simply absent, and an empty
    list returns [] without querying. The caller owns ``conn`` and its transaction: this
    function never commits, rolls back or closes anything.
    """
    ids = _clean_ids(document_ids)
    if ids is not None and not ids:
        return []
    rows = conn.execute(build_selection_statement(ids)).all()
    return [_state_from_row(row) for row in rows]