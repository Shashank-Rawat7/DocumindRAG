"""In-memory index build for one document (Component 12 / C2.5).

Pure: no database access, no status changes, no file I/O. Given a document's
exact extracted text, an embedder and a page mapping, it produces everything
needed to publish that document's chunks:

    source_sha256  - SHA-256 of extracted_text.encode("utf-8")
    signature      - canonical JSON naming every rule that shaped the chunks
                     and vectors (documents.indexed_signature)
    embedding_id   - model id @ revision (documents.indexed_embedding_id)
    chunks         - ordered ChunkRecord values, one per document_chunks row

The text is used exactly as given: it is never stripped, normalised or copied
into another form, so every offset refers to documents.extracted_text.

Known V1 limitation: the signature does not cover the PDF page-mapping rules
(app/rag/pdf_pages.py has no version). Pages do not affect chunk text or
vectors, but a change to those rules would not mark existing builds stale.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.rag import token_offsets
from app.rag.chunking import (
    DEFAULT_CHUNKER_CONFIG,
    ChunkerConfig,
    ChunkSpan,
    chunk_text,
    sha256_text,
)
from app.rag.embedder import (
    EMBEDDING_DIM,
    NORM_TOLERANCE,
    Embedder,
    EmbeddingOutputError,
)
from app.rag.pdf_pages import PageMapping, assign_chunk_pages

# Bump when indexing semantics change: chunk construction, offset interpretation,
# embedding preparation or record semantics. Not for EMBED_GROUP_SIZE alone,
# which changes neither the vectors nor their order.
INDEX_BUILD_VERSION = "c2.5-v1"
EMBED_GROUP_SIZE = 32             # chunks per embed_passages call (bounds how long the inference lock is held)


class IndexBuildError(Exception):
    """The document cannot be indexed. The message is safe to show to the user
    and never contains document text."""


NO_INDEXABLE_TEXT = "No indexable text found in this document"


# ---------- identity ----------

def build_signature(chunker_config: ChunkerConfig, embedder: Embedder) -> str:
    """Canonical JSON identifying how a document's chunks and vectors were built.

    Two builds with equal signatures used the same chunking rules, the same
    offset-mapping rules and the same embedding model and configuration.
    Calling this loads the embedder (its signature reports the verified
    attention implementation).
    """
    fields = {
        "index_build_version": INDEX_BUILD_VERSION,
        "chunker": chunker_config.signature_fields(),
        "offset_mapping": token_offsets.signature_fields(),
        "embedder": embedder.signature_fields(),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def source_sha256(text: str) -> str:
    """documents.source_sha256: lowercase hex SHA-256 of text.encode("utf-8")."""
    return sha256_text(text)


# ---------- step 2: chunk planning ----------

@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Ordered chunk spans of one text, before pages and embeddings."""

    chunks: tuple[ChunkSpan, ...]  # non-empty, in document order, chunk.index == position
    source_tokens: int             # content tokens of the whole text (om-v1 total)


def plan_chunks(
    text: str,
    embedder: Embedder,
    chunker_config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG,
) -> ChunkPlan:
    """Split ``text`` into chunks measured in real model tokens.

    ``text`` is passed unchanged to both the tokenizer and the chunker, so every
    chunk offset indexes this exact string. Errors from source_units()
    (EmbeddingInputError, EmbeddingOutputError, EmbedderUnavailableError,
    OffsetMappingError) and from chunk_text() (ChunkingError) propagate
    unchanged; IndexBuildError means the text yielded no chunk at all.
    """
    mapping = embedder.source_units(text)
    spans = [(unit.start, unit.end) for unit in mapping.units]
    counts = [unit.token_count for unit in mapping.units]
    chunks = chunk_text(text, spans, chunker_config, token_counts=counts)
    if not chunks:
        raise IndexBuildError(NO_INDEXABLE_TEXT)
    return ChunkPlan(chunks=tuple(chunks), source_tokens=mapping.total_tokens)


# ---------- step 3: page assignment ----------

@dataclass(frozen=True, slots=True)
class PagedChunk:
    """A planned chunk with its page range (both None when pages are unavailable)."""

    chunk: ChunkSpan               # the planned ChunkSpan itself, unchanged
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True, slots=True)
class PagedChunkPlan:
    """The chunk plan with pages, in the same order, ready for embedding."""

    chunks: tuple[PagedChunk, ...]
    source_tokens: int
    pages_available: bool          # False: the mapping was unavailable and every page is None


def assign_pages(plan: ChunkPlan, page_mapping: PageMapping) -> PagedChunkPlan:
    """Attach page_start/page_end to every planned chunk.

    ``page_mapping`` comes from the caller (normally
    pdf_pages.map_document_pages(path, extracted_text) for the same text the
    plan was built from); this function does no file I/O. An unavailable
    mapping gives every chunk NULL pages. PageMappingError (the mapping and the
    chunks disagree: a bug) and TypeError (not a PageMapping) propagate
    unchanged, and no partial result is returned.
    """
    pages = assign_chunk_pages(page_mapping, plan.chunks)
    paged = tuple(
        PagedChunk(chunk=chunk, page_start=page.page_start, page_end=page.page_end)
        for chunk, page in zip(plan.chunks, pages, strict=True)
    )
    return PagedChunkPlan(
        chunks=paged,
        source_tokens=plan.source_tokens,
        pages_available=page_mapping.is_available,
    )


# ---------- results ----------

@dataclass(frozen=True, slots=True, eq=False)
class ChunkRecord:
    """Values for one document_chunks row (field names match the columns).

    ``eq=False``: records hold a numpy array, which has no single truth value
    under ``==``; compare fields explicitly.
    """

    chunk_index: int
    start_offset: int
    end_offset: int
    page_start: int | None
    page_end: int | None
    text: str
    span_sha256: str
    embedding: np.ndarray          # shape (384,), float32, L2-normalised, read-only


@dataclass(frozen=True, slots=True)
class IndexBuildStats:
    """Facts about the build, for logs and measurement (not stored in the DB)."""

    chunk_count: int
    source_tokens: int             # content tokens of the whole text (om-v1 total)
    max_chunk_tokens: int          # largest ChunkSpan.token_count
    pages_available: bool          # False when the page mapping was unavailable (all pages NULL)


@dataclass(frozen=True, slots=True, eq=False)
class DocumentIndexBuild:
    """A complete, validated build of one document, ready to publish."""

    source_sha256: str
    signature: str
    embedding_id: str
    chunks: tuple[ChunkRecord, ...]
    stats: IndexBuildStats

# ---------- step 4: grouped embedding and record assembly ----------

def embed_chunks(paged: PagedChunkPlan, embedder: Embedder) -> tuple[np.ndarray, ...]:
    """One read-only (384,) float32 vector per chunk, in chunk order.

    Chunks are embedded EMBED_GROUP_SIZE at a time, so the embedder's inference
    lock is released between groups. Every group's output is checked against
    the embedder contract; a violation raises EmbeddingOutputError. Errors from
    embed_passages() (for example EmbeddingInputTooLongError or
    EmbedderUnavailableError) propagate unchanged.
    """
    texts = [item.chunk.text for item in paged.chunks]
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), EMBED_GROUP_SIZE):
        group = texts[start:start + EMBED_GROUP_SIZE]
        output = embedder.embed_passages(group)
        _check_group(output, len(group))
        for row in output:
            vector = np.array(row, dtype=np.float32, copy=True)   # own memory, independent of the group
            vector.setflags(write=False)
            vectors.append(vector)
    return tuple(vectors)


def _check_group(output: object, expected_rows: int) -> None:
    """The embed_passages() contract: float32 (n, 384), finite, unit-length rows."""
    if not isinstance(output, np.ndarray):
        raise EmbeddingOutputError(f"embed_passages returned {type(output).__name__}, expected a numpy array")
    if output.dtype != np.float32 or output.shape != (expected_rows, EMBEDDING_DIM):
        raise EmbeddingOutputError(
            f"embed_passages returned dtype {output.dtype} and shape {output.shape}, "
            f"expected float32 and {(expected_rows, EMBEDDING_DIM)}"
        )
    if not np.isfinite(output).all():
        raise EmbeddingOutputError("embed_passages returned non-finite values")
    if np.any(np.abs(np.linalg.norm(output, axis=1) - 1.0) > NORM_TOLERANCE):
        raise EmbeddingOutputError("embed_passages returned vectors that are not unit length")


def assemble_records(paged: PagedChunkPlan, vectors: Sequence[np.ndarray]) -> tuple[ChunkRecord, ...]:
    """document_chunks values: the i-th vector belongs to the i-th chunk."""
    return tuple(
        ChunkRecord(
            chunk_index=item.chunk.index,
            start_offset=item.chunk.start,
            end_offset=item.chunk.end,
            page_start=item.page_start,
            page_end=item.page_end,
            text=item.chunk.text,
            span_sha256=item.chunk.span_sha256,
            embedding=vector,
        )
        for item, vector in zip(paged.chunks, vectors, strict=True)
    )


def build_document_index(
    text: str,
    *,
    embedder: Embedder,
    page_mapping: PageMapping,
    chunker_config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG,
) -> DocumentIndexBuild:
    """Build everything needed to publish one document; nothing is written anywhere.

    ``text`` must be the document's exact extracted_text and ``page_mapping``
    must have been built from that same text (map_document_pages). Pages are assigned before full model initialization and embedding inference, so a PageMappingError fails the build before those operations.. Every error from the steps propagates unchanged.
    """
    plan = plan_chunks(text, embedder, chunker_config)
    paged = assign_pages(plan, page_mapping)
    signature = build_signature(chunker_config, embedder)
    records = assemble_records(paged, embed_chunks(paged, embedder))
    stats = IndexBuildStats(
        chunk_count=len(records),
        source_tokens=plan.source_tokens,
        max_chunk_tokens=max(chunk.token_count for chunk in plan.chunks),
        pages_available=paged.pages_available,
    )
    return DocumentIndexBuild(
        source_sha256=source_sha256(text),
        signature=signature,
        embedding_id=embedder.embedding_id,
        chunks=records,
        stats=stats,
    )
