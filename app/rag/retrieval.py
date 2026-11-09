"""Retrieval (Component 12 / C6): contract, validation and exact vector search.

C6.1 defines the types, limits and argument validation. C6.2 adds
``retrieve_chunks``: one read-only SELECT that joins document_chunks to
documents, always filters on the owner, applies ``retrievable_filter`` and ranks
by exact cosine distance. The query-embedding wrapper arrives in C6.3.

Rules that hold for every call:
  * Ownership is mandatory and independent of ``document_ids``, which can only
    narrow the result.
  * Only published chunks are returned; the publication rules live in
    ``indexing_service.retrievable_filter``.
  * The call never begins, commits or rolls back a transaction.
  * A row that fails verification aborts the whole call; no partial list is
    ever returned.
  * No message raised here contains document text or query text.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.rag.chunking import sha256_text
from app.rag.embedder import EMBEDDING_DIM, NORM_TOLERANCE
from app.rag.indexing_service import retrievable_filter

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MAX_DOCUMENT_IDS = 100


# ---------- errors ----------

class RetrievalError(Exception):
    """Base class for retrieval failures."""


class RetrievalRequestError(RetrievalError):
    """The caller's arguments are invalid. The message names the argument only."""


class EmptyQueryError(RetrievalRequestError):
    """The query is empty or contains only whitespace."""


class RetrievalIntegrityError(RetrievalError):
    """A stored chunk failed verification; the whole retrieval call is aborted."""

    def __init__(self, chunk_id: int, check: str) -> None:
        self.chunk_id = chunk_id
        self.check = check
        super().__init__(f"chunk {chunk_id} failed the {check} integrity check")


# ---------- result ----------

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk, with everything a citation needs.

    source_version is documents.indexed_source_version: the version of
    extracted_text these chunks were published from (not merely the current
    version). score is 1 - cosine distance, so higher means closer.
    """

    chunk_id: int
    document_id: int
    chunk_index: int
    start_offset: int
    end_offset: int
    page_start: int | None
    page_end: int | None
    text: str
    span_sha256: str
    score: float
    source_version: int
    source_sha256: str

# ---------- validation ----------

def _is_plain_int(value: object) -> bool:
    return type(value) is int


def validate_owner_id(owner_id: int) -> int:
    if not _is_plain_int(owner_id):
        raise RetrievalRequestError("owner_id must be an int")
    return owner_id


def validate_top_k(top_k: int) -> int:
    if not _is_plain_int(top_k):
        raise RetrievalRequestError("top_k must be an int")
    if not 1 <= top_k <= MAX_TOP_K:
        raise RetrievalRequestError(f"top_k must be between 1 and {MAX_TOP_K}")
    return top_k


def validate_embedding_id(embedding_id: str) -> str:
    if type(embedding_id) is not str or not embedding_id:
        raise RetrievalRequestError("embedding_id must be a non-empty str")
    return embedding_id


def validate_query_vector(query_vector: np.ndarray) -> np.ndarray:
    if not isinstance(query_vector, np.ndarray):
        raise RetrievalRequestError("query_vector must be a numpy array")
    if query_vector.dtype != np.float32 or query_vector.shape != (EMBEDDING_DIM,):
        raise RetrievalRequestError(
            f"query_vector must be float32 with shape ({EMBEDDING_DIM},)"
        )
    if not np.isfinite(query_vector).all():
        raise RetrievalRequestError("query_vector contains non-finite values")
    norm = np.linalg.norm(query_vector[np.newaxis, :], axis=1)[0]
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise RetrievalRequestError("query_vector is not unit length")
    return query_vector


def normalize_document_ids(document_ids: Sequence[int] | None) -> tuple[int, ...] | None:
    if document_ids is None:
        return None
    if isinstance(document_ids, (str, bytes, bytearray)) or not isinstance(document_ids, Sequence):
        raise RetrievalRequestError("document_ids must be a list or tuple of int, or None")
    if len(document_ids) > MAX_DOCUMENT_IDS:
        raise RetrievalRequestError(f"document_ids may contain at most {MAX_DOCUMENT_IDS} items")
    for item in document_ids:
        if not _is_plain_int(item):
            raise RetrievalRequestError("document_ids must contain only int values")
    return tuple(dict.fromkeys(document_ids))


def validate_query_text(query: str) -> str:
    if type(query) is not str:
        raise RetrievalRequestError("query must be a str")
    if query.strip() == "":
        raise EmptyQueryError("query is empty")
    return query


def verify_chunk_integrity(
    *, chunk_id: int, text: str, start_offset: int, end_offset: int, span_sha256: str
) -> None:
    if len(text) != end_offset - start_offset:
        raise RetrievalIntegrityError(chunk_id, "length")
    if sha256_text(text) != span_sha256:
        raise RetrievalIntegrityError(chunk_id, "sha256")

def validate_source_sha256(source_sha256: str) -> str:
    if (
        type(source_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise RetrievalRequestError(
            "source_sha256 must be a lowercase SHA-256 hex digest"
        )
    return source_sha256



# ---------- retrieval ----------

def _build_retrieval_statement(
    *,
    owner_id: int,
    query_vector: np.ndarray,
    embedding_id: str,
    top_k: int,
    document_ids: tuple[int, ...] | None,
) -> Select:
    """The single retrieval SELECT. Pure: it builds SQL and never touches a database.

    Arguments must already be validated. The owner predicate belongs to the base
    statement; ``document_ids`` can only add a further AND.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.start_offset,
            DocumentChunk.end_offset,
            DocumentChunk.page_start,
            DocumentChunk.page_end,
            DocumentChunk.text,
            DocumentChunk.span_sha256,
            distance,
            Document.indexed_source_version.label("source_version"),
            Document.source_sha256,
        )
        .select_from(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.owner_id == owner_id, retrievable_filter(embedding_id))
    )
    if document_ids is not None:
        statement = statement.where(Document.id.in_(document_ids))
    return statement.order_by(
        distance, DocumentChunk.document_id, DocumentChunk.chunk_index
    ).limit(top_k)


def _stored_source_sha256(chunk_id: int, value: Any) -> str:
    """A stored documents.source_sha256 that is present and well formed.

    Reuses validate_source_sha256 (the one definition of "well formed") but
    reports a bad stored value as an integrity failure, not as a caller error.
    """
    try:
        return validate_source_sha256(value)
    except RetrievalRequestError:
        raise RetrievalIntegrityError(chunk_id, "source_sha256") from None


def _row_to_chunk(row: Any) -> RetrievedChunk:
    """Verify one result row and turn it into a RetrievedChunk.

    Fail closed: the first failed check raises RetrievalIntegrityError.
    """
    verify_chunk_integrity(
        chunk_id=row.chunk_id,
        text=row.text,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        span_sha256=row.span_sha256,
    )
    source_sha256 = _stored_source_sha256(row.chunk_id, row.source_sha256)
    distance = row.distance
    if distance is None or not math.isfinite(distance):
        raise RetrievalIntegrityError(row.chunk_id, "distance")
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        page_start=row.page_start,
        page_end=row.page_end,
        text=row.text,
        span_sha256=row.span_sha256,
        score=1.0 - float(distance),
        source_version=row.source_version,
        source_sha256=source_sha256,
    )


def retrieve_chunks(
    db: Session,
    *,
    owner_id: int,
    query_vector: np.ndarray,
    embedding_id: str,
    top_k: int = DEFAULT_TOP_K,
    document_ids: Sequence[int] | None = None,
) -> list[RetrievedChunk]:
    """The owner's published chunks nearest to ``query_vector``, best first.

    One read-only SELECT in the caller's transaction: this function never
    begins, commits or rolls back. Database errors propagate unchanged.
    """
    owner_id = validate_owner_id(owner_id)
    top_k = validate_top_k(top_k)
    embedding_id = validate_embedding_id(embedding_id)
    query_vector = validate_query_vector(query_vector)
    ids = normalize_document_ids(document_ids)
    if ids is not None and not ids:
        return []
    statement = _build_retrieval_statement(
        owner_id=owner_id,
        query_vector=query_vector,
        embedding_id=embedding_id,
        top_k=top_k,
        document_ids=ids,
    )
    rows = db.execute(statement).all()
    return [_row_to_chunk(row) for row in rows]
