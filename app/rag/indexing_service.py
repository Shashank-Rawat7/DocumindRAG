"""Database half of the indexing service (Component 12 / C5.1).

Everything that moves a document through the indexing lifecycle lives here:

    request_indexing  -> pending     (a caller wants the document (re)indexed)
    claim_document    -> indexing    (one worker wins a lease and an attempt number)
    publish_build     -> indexed     (replace the chunks and record the publication)
    record_failure    -> indexed / failed   (end an attempt that did not publish)

It builds nothing: the pure in-memory build is app/rag/indexing.py, and running
build + publish is C5.2.

Transactions. Every public function is exactly one transaction, opened with
``with db.begin():``. The session must therefore have no transaction in progress
(SQLAlchemy raises InvalidRequestError otherwise). The slow work (tokenizing,
embedding) happens between calls and holds no lock and no transaction; the lease
is the only thing protecting it.

Clock. Every timestamp and every lease comparison uses the database's
clock_timestamp(), so all workers agree regardless of their own clocks.

Retrieval. A document's published chunks are searchable while ``retrievable_filter``
holds, whatever indexing_status says, so they stay searchable during a re-index.
That is only sound while two application invariants hold:

  I-1  After processing_status = 'completed', documents.extracted_text is never
       modified. Any future code that rewrites it must increment source_version in
       the same UPDATE. (Today only app/core/processing.py writes it, once, before
       'completed'; tests/test_indexing_service.py guards this.)
  I-2  Chunks are inserted only by publish_build, which verifies version, attempt
       and sha256(current text) under a row lock and records the published
       version and hash on the document.

With I-1 and I-2, ``indexed_source_version = source_version`` implies the chunks
still describe the current text. record_failure re-checks the hash of the current
text as a backstop and discards a publication that no longer matches. Direct SQL
edits of extracted_text are outside the guarantee (accepted V1 limitation).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from sqlalchemy import (
    ColumnElement,
    and_,
    delete,
    exists,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.rag.chunking import ChunkingError
from app.rag.embedder import (
    EmbedderUnavailableError,
    EmbeddingInputError,
    EmbeddingInputTooLongError,
    EmbeddingOutputError,
)
from app.rag.indexing import (
    NO_INDEXABLE_TEXT,
    DocumentIndexBuild,
    IndexBuildError,
    source_sha256,
)
from app.rag.pdf_pages import PageMappingError
from app.rag.token_offsets import OffsetMappingError

# ---------- types ----------

@dataclass(frozen=True, slots=True)
class IndexingClaim:
    """The right to publish one document's index, valid while the attempt matches."""

    document_id: int
    attempt: int           # documents.indexing_attempt after this claim incremented it
    source_version: int    # documents.source_version when the claim was made


class PublishOutcome(str, Enum):
    PUBLISHED = "published"
    STALE = "stale"        # attempt, version, status or source text changed; nothing written
    GONE = "gone"          # the document was deleted; nothing written


class FailureOutcome(str, Enum):
    FAILED = "failed"                      # nothing had ever been published
    RESTORED = "restored_indexed"          # the earlier publication is still valid and is kept
    DISCARDED = "failed_index_discarded"   # the earlier publication was invalid and was removed
    STALE = "stale"                        # another attempt owns the document; nothing written
    GONE = "gone"                          # the document was deleted; nothing written


class FailureReason(str, Enum):
    """The only values that may be stored in documents.indexing_error.

    Each value is a fixed, user-safe message: never document text, a filename or
    the text of an exception.
    """

    NO_INDEXABLE_TEXT = NO_INDEXABLE_TEXT
    BUILD_REJECTED = "The document could not be indexed"
    CHUNKING = "The document text could not be split into chunks safely"
    OFFSET_MAPPING = "The document text could not be aligned with the tokenizer"
    CHUNK_TOO_LONG = "A chunk exceeded the embedding model's input limit"
    EMBEDDING_INVALID = "The embedding model returned an invalid result"
    PAGE_MAPPING = "Page numbers could not be assigned to the chunks"
    EMBEDDER_UNAVAILABLE = "The embedding model is not available"
    DATABASE = "The index could not be saved"
    UNEXPECTED = "Indexing failed unexpectedly"


# ---------- classification ----------

def classify_failure(exc: Exception) -> FailureReason:
    """Map an exception to a fixed FailureReason by type alone.

    The exception's message is never used to build the result. The one message
    comparison (IndexBuildError) only selects between two fixed reasons.
    """
    if isinstance(exc, IndexBuildError):
        return FailureReason.NO_INDEXABLE_TEXT if exc.args == (NO_INDEXABLE_TEXT,) else FailureReason.BUILD_REJECTED
    if isinstance(exc, ChunkingError):
        return FailureReason.CHUNKING
    if isinstance(exc, OffsetMappingError):
        return FailureReason.OFFSET_MAPPING
    if isinstance(exc, EmbeddingInputTooLongError):      # before its base class
        return FailureReason.CHUNK_TOO_LONG
    if isinstance(exc, (EmbeddingInputError, EmbeddingOutputError)):
        return FailureReason.EMBEDDING_INVALID
    if isinstance(exc, PageMappingError):
        return FailureReason.PAGE_MAPPING
    if isinstance(exc, EmbedderUnavailableError):
        return FailureReason.EMBEDDER_UNAVAILABLE
    if isinstance(exc, SQLAlchemyError):
        return FailureReason.DATABASE
    return FailureReason.UNEXPECTED


# ---------- retrieval predicate ----------

def retrievable_filter(embedding_id: str) -> ColumnElement[bool]:
    """WHERE-clause for documents whose published chunks may be searched.

    Deliberately ignores indexing_status: published chunks stay searchable while
    a re-index is pending or running. ``embedding_id`` is the id of the embedder
    that will embed the query; vectors from another model are not comparable.
    """
    return and_(
        Document.indexed_source_version.is_not(None),
        Document.indexed_source_version == Document.source_version,
        Document.indexed_embedding_id == embedding_id,
    )


# ---------- transitions ----------

def request_indexing(db: Session, document_id: int) -> bool:
    """Queue a document for (re)indexing. True if it is now pending.

    Idempotent for an already pending document. Refused (False) for a missing
    document, a document whose text extraction has not completed, and a document
    whose current attempt still holds a live lease. An attempt whose lease has
    expired is abandoned and the document becomes pending again. indexing_error
    and all published data are left untouched.
    """
    with db.begin():
        now = func.clock_timestamp()
        queued = db.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.processing_status == "completed",
                or_(
                    Document.indexing_status.is_(None),
                    Document.indexing_status.in_(("pending", "failed", "indexed")),
                    and_(Document.indexing_status == "indexing", Document.indexing_lease_expires_at <= now),
                ),
            )
            .values(indexing_status="pending", indexing_lease_expires_at=None)
            .returning(Document.id)
            .execution_options(synchronize_session=False)
        ).first()
        return queued is not None


def claim_document(
    db: Session, document_id: int, *, lease_minutes: int | None = None
) -> IndexingClaim | None:
    """Try to take a pending (or lease-expired) document. None if not taken.

    One conditional UPDATE ... RETURNING: of any number of concurrent callers,
    exactly one matches the WHERE clause after the row lock is released, so
    exactly one wins. The attempt number it receives is what makes late writes
    from an abandoned worker harmless.
    """
    minutes = settings.indexing_lease_minutes if lease_minutes is None else lease_minutes
    if minutes < 1:
        raise ValueError("lease_minutes must be at least 1")
    with db.begin():
        now = func.clock_timestamp()
        claimed = db.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.processing_status == "completed",
                or_(
                    Document.indexing_status == "pending",
                    and_(Document.indexing_status == "indexing", Document.indexing_lease_expires_at <= now),
                ),
            )
            .values(
                indexing_status="indexing",
                indexing_attempt=Document.indexing_attempt + 1,
                indexing_lease_expires_at=now + timedelta(minutes=minutes),
            )
            .returning(Document.indexing_attempt, Document.source_version)
            .execution_options(synchronize_session=False)
        ).first()
    if claimed is None:
        return None
    return IndexingClaim(document_id=document_id, attempt=claimed.indexing_attempt, source_version=claimed.source_version)


def publish_build(db: Session, claim: IndexingClaim, build: DocumentIndexBuild) -> PublishOutcome:
    """Atomically replace the document's chunks with ``build`` and mark it indexed.

    Under a row lock it re-checks that this claim still owns the document and that
    the text is the text the build was made from. Any error rolls back everything
    (old chunks and metadata stay exactly as they were) and propagates.
    """
    if not build.chunks:
        raise ValueError("a build must contain at least one chunk")
    with db.begin():
        row = db.execute(
            select(
                Document.indexing_status,
                Document.indexing_attempt,
                Document.source_version,
                Document.extracted_text,
            )
            .where(Document.id == claim.document_id)
            .with_for_update()
        ).first()
        if row is None:
            return PublishOutcome.GONE
        if (
            row.indexing_status != "indexing"
            or row.indexing_attempt != claim.attempt
            or row.source_version != claim.source_version
            or row.extracted_text is None
            or source_sha256(row.extracted_text) != build.source_sha256
        ):
            return PublishOutcome.STALE

        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == claim.document_id))
        db.execute(
            insert(DocumentChunk),
            [
                {
                    "document_id": claim.document_id,
                    "chunk_index": record.chunk_index,
                    "start_offset": record.start_offset,
                    "end_offset": record.end_offset,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                    "text": record.text,
                    "span_sha256": record.span_sha256,
                    "embedding": record.embedding,
                }
                for record in build.chunks
            ],
        )
        db.execute(
            update(Document)
            .where(Document.id == claim.document_id)
            .values(
                indexing_status="indexed",
                indexing_error=None,
                indexing_lease_expires_at=None,
                indexed_source_version=claim.source_version,
                source_sha256=build.source_sha256,
                indexed_signature=build.signature,
                indexed_embedding_id=build.embedding_id,
                indexed_at=func.clock_timestamp(),
            )
            .execution_options(synchronize_session=False)
        )
        return PublishOutcome.PUBLISHED


def record_failure(db: Session, claim: IndexingClaim, reason: FailureReason) -> FailureOutcome:
    """End an attempt that did not publish, recording ``reason``.

    If the document still has a valid earlier publication it is kept and the
    document returns to 'indexed' (the failed attempt only sets indexing_error).
    If the earlier publication is no longer valid it is removed, so it can no
    longer be retrieved. Otherwise the document is simply 'failed'.
    """
    if not isinstance(reason, FailureReason):
        raise TypeError("reason must be a FailureReason")
    with db.begin():
        row = db.execute(
            select(
                Document.indexing_status,
                Document.indexing_attempt,
                Document.source_version,
                Document.source_sha256,
                Document.indexed_source_version,
                Document.indexed_signature,
                Document.indexed_embedding_id,
                Document.indexed_at,
                Document.extracted_text,
            )
            .where(Document.id == claim.document_id)
            .with_for_update()
        ).first()
        if row is None:
            return FailureOutcome.GONE
        if (
            row.indexing_status != "indexing"
            or row.indexing_attempt != claim.attempt
            or row.source_version != claim.source_version
        ):
            return FailureOutcome.STALE

        has_chunks = bool(
            db.scalar(select(exists().where(DocumentChunk.document_id == claim.document_id)))
        )
        if _publication_is_valid(row, has_chunks):
            db.execute(
                update(Document)
                .where(Document.id == claim.document_id)
                .values(indexing_status="indexed", indexing_error=reason.value, indexing_lease_expires_at=None)
                .execution_options(synchronize_session=False)
            )
            return FailureOutcome.RESTORED

        had_publication = has_chunks or any(
            value is not None
            for value in (
                row.source_sha256,
                row.indexed_source_version,
                row.indexed_signature,
                row.indexed_embedding_id,
                row.indexed_at,
            )
        )
        if had_publication:
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == claim.document_id))
        db.execute(
            update(Document)
            .where(Document.id == claim.document_id)
            .values(
                indexing_status="failed",
                indexing_error=reason.value,
                indexing_lease_expires_at=None,
                source_sha256=None,
                indexed_source_version=None,
                indexed_signature=None,
                indexed_embedding_id=None,
                indexed_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        return FailureOutcome.DISCARDED if had_publication else FailureOutcome.FAILED


def _publication_is_valid(row, has_chunks: bool) -> bool:
    """Is the document's last publication still a faithful index of its current text?

    ``row`` carries the locked document columns (including extracted_text).
    """
    return (
        has_chunks
        and row.indexed_source_version is not None
        and row.indexed_source_version == row.source_version
        and row.indexed_at is not None
        and row.indexed_signature is not None
        and row.indexed_embedding_id is not None
        and row.source_sha256 is not None
        and row.extracted_text is not None
        and source_sha256(row.extracted_text) == row.source_sha256
    )