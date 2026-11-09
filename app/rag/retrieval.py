"""Retrieval contract (Component 12 / C6.1): types, limits and validation.

Pure: no database access and no model loading. The SQL arrives in C6.2 and the
query-embedding wrapper in C6.3. No message raised here contains document text
or query text.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.rag.chunking import sha256_text
from app.rag.embedder import EMBEDDING_DIM, NORM_TOLERANCE

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
