"""Candidate pool and ``RetrievedChunk`` -> ``RankedChunk`` conversion (E1.7.2).

Pure module: it opens no database, imports no SQLAlchemy and runs no retrieval. The caller (the
E1.7.5 wiring) reads rows and hands them in; this module verifies them and fails closed with one
``EvaluationDataError`` that carries every finding. No issue holds a quote, a question or any text.

Document mapping (``map_documents``). A database ``document_id`` is never assumed. The corpus
ingest stores each document with ``description = "<corpus_version>:<corpus_key>"``; a row maps to
the manifest document whose key and the manifest's ``corpus_version`` reproduce that description
exactly. Optionally the stored ``filename`` is also compared with the corpus file names the caller
read. A row that matches no manifest document (``DOC_UNKNOWN``), a manifest document with no row
(``DOC_MISSING``) or with two rows (``DOC_AMBIGUOUS``), a repeated id, a foreign owner, an unpublished
or stale document (``indexed_source_version`` must be present and equal ``source_version``), another
embedder's index and a ``source_sha256`` that differs from the manifest's ``extracted_text_sha256``
are all refused. ``is_retrievable`` mirrors ``indexing_service.retrievable_filter`` (the dense query's
predicate); the E1.7.5 pool query must use that same SQL filter, and ``convert_retrieved`` re-checks
every dense chunk against the pool, so pool and dense retrieval cannot silently diverge.

Pool (``build_pool``). Every published chunk of the six documents. Each chunk must belong to a mapped
document (ownership is a property of the document, not of the chunk), carry that document's
published version and source hash, satisfy ``len(text) == end - start`` and ``span_sha256 == sha256(text)``,
and equal ``texts[corpus_key][start:end]`` of the verified source text. Duplicate chunk ids,
duplicate ``(document, chunk_index)`` and duplicate canonical keys are refused. Completeness: each
document has chunks with indexes ``0..n-1``, valid and strictly increasing ranges, and every
meaningful character of its text lies in some chunk (whitespace and ignorable characters may not).

Conversion keeps the existing ``RankedChunk`` semantics unchanged: ``corpus_key``, ``start`` and
``end`` are the half-open code-point range, and ``text_sha256`` is the chunk's ``span_sha256``.
Rank order is preserved and scores are not used.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from app.rag.chunking import sha256_text
from app.rag.evaluation.baselines import canonical_order
from app.rag.evaluation.metrics import RankedChunk
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    EvaluationDataError,
    Issue,
    Manifest,
)
from app.rag.text_boundaries import is_ignorable

DOC_MANIFEST = "DOC_MANIFEST"
DOC_FIELD_INVALID = "DOC_FIELD_INVALID"
DOC_UNKNOWN = "DOC_UNKNOWN"
DOC_MISSING = "DOC_MISSING"
DOC_AMBIGUOUS = "DOC_AMBIGUOUS"
DOC_DUPLICATE_ID = "DOC_DUPLICATE_ID"
DOC_FOREIGN_OWNER = "DOC_FOREIGN_OWNER"
DOC_FILENAME_MISMATCH = "DOC_FILENAME_MISMATCH"
DOC_NOT_PUBLISHED = "DOC_NOT_PUBLISHED"
DOC_STALE_VERSION = "DOC_STALE_VERSION"
DOC_EMBEDDING_MISMATCH = "DOC_EMBEDDING_MISMATCH"
DOC_SOURCE_HASH = "DOC_SOURCE_HASH"

POOL_CHUNK_INVALID = "POOL_CHUNK_INVALID"
POOL_FOREIGN_DOCUMENT = "POOL_FOREIGN_DOCUMENT"
POOL_STALE_VERSION = "POOL_STALE_VERSION"
POOL_SOURCE_HASH = "POOL_SOURCE_HASH"
POOL_INTEGRITY = "POOL_INTEGRITY"
POOL_TEXT_MISMATCH = "POOL_TEXT_MISMATCH"
POOL_DUPLICATE_CHUNK = "POOL_DUPLICATE_CHUNK"
POOL_DOCUMENT_EMPTY = "POOL_DOCUMENT_EMPTY"
POOL_INDEX_GAP = "POOL_INDEX_GAP"
POOL_RANGE_ORDER = "POOL_RANGE_ORDER"
POOL_COVERAGE = "POOL_COVERAGE"
POOL_TEXT_MISSING = "POOL_TEXT_MISSING"
RANK_NOT_IN_POOL = "RANK_NOT_IN_POOL"
RANK_DUPLICATE = "RANK_DUPLICATE"
RANK_MISMATCH = "RANK_MISMATCH"

_MESSAGES = {
    DOC_MANIFEST: "the manifest is not an ingested manifest with a hash for every document",
    DOC_FIELD_INVALID: "a document row has a field of the wrong type",
    DOC_UNKNOWN: "a document row matches no manifest document",
    DOC_MISSING: "a manifest document has no database row",
    DOC_AMBIGUOUS: "a manifest document has more than one database row",
    DOC_DUPLICATE_ID: "two document rows have the same id",
    DOC_FOREIGN_OWNER: "a document row belongs to another owner",
    DOC_FILENAME_MISMATCH: "a document row's filename does not match the corpus file",
    DOC_NOT_PUBLISHED: "a document has no published chunks",
    DOC_STALE_VERSION: "a document's published version is not its current version",
    DOC_EMBEDDING_MISMATCH: "a document was indexed by another embedder",
    DOC_SOURCE_HASH: "a document's source hash differs from the manifest",
    POOL_CHUNK_INVALID: "a chunk has a missing or wrongly typed field or an invalid range",
    POOL_FOREIGN_DOCUMENT: "a chunk belongs to a document that is not in the corpus mapping",
    POOL_STALE_VERSION: "a chunk's source version is not the document's published version",
    POOL_SOURCE_HASH: "a chunk's source hash differs from the manifest",
    POOL_INTEGRITY: "a chunk's length or text hash does not match its range",
    POOL_TEXT_MISMATCH: "a chunk's text is not the source text at its range",
    POOL_DUPLICATE_CHUNK: "a chunk repeats another chunk",
    POOL_DOCUMENT_EMPTY: "a corpus document has no chunk in the pool",
    POOL_INDEX_GAP: "a document's chunk indexes are not 0..n-1",
    POOL_RANGE_ORDER: "a document's chunk ranges are out of bounds or do not strictly increase",
    POOL_COVERAGE: "meaningful source characters of a document are in no chunk",
    POOL_TEXT_MISSING: "a corpus document has no verified source text or no mapped database id",
    RANK_NOT_IN_POOL: "a retrieved chunk is not in the candidate pool",
    RANK_DUPLICATE: "a retrieved ranking repeats a chunk",
    RANK_MISMATCH: "a retrieved chunk differs from the pool's chunk with the same id",
}


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_ERROR)


def _plain_int(value: object) -> bool:
    return type(value) is int


@dataclass(frozen=True, slots=True)
class PoolDocument:
    """One ``documents`` row as read by the caller (never the extracted text)."""

    document_id: int
    owner_id: int
    filename: str
    description: str | None
    source_version: int
    indexed_source_version: int | None
    indexed_embedding_id: str | None
    source_sha256: str | None


@dataclass(frozen=True, slots=True)
class DocumentMap:
    """The verified mapping between database ids and corpus keys (one to one, manifest order)."""

    corpus_version: str
    owner_id: int
    embedding_id: str
    key_by_id: Mapping[int, str]
    id_by_key: Mapping[str, int]
    source_version_by_id: Mapping[int, int]
    source_sha256_by_key: Mapping[str, str]

    @property
    def document_ids(self) -> tuple[int, ...]:
        """The ids to pass as ``document_ids`` to dense retrieval, in manifest order."""
        return tuple(self.id_by_key.values())


def is_retrievable(document: PoolDocument, embedding_id: str) -> bool:
    """Python mirror of ``indexing_service.retrievable_filter`` (published, current, same embedder)."""
    return (
        document.indexed_source_version is not None
        and document.indexed_source_version == document.source_version
        and document.indexed_embedding_id == embedding_id
    )


def _expected_description(manifest: Manifest, corpus_key: str) -> str:
    return f"{manifest.corpus_version}:{corpus_key}"


def _document_fields_ok(row: object) -> bool:
    if not isinstance(row, PoolDocument):
        return False
    return (
        _plain_int(row.document_id)
        and _plain_int(row.owner_id)
        and type(row.filename) is str
        and (row.description is None or type(row.description) is str)
        and _plain_int(row.source_version)
        and (row.indexed_source_version is None or _plain_int(row.indexed_source_version))
        and (row.indexed_embedding_id is None or type(row.indexed_embedding_id) is str)
        and (row.source_sha256 is None or type(row.source_sha256) is str)
    )


def map_documents(
    manifest: Manifest,
    documents: Iterable[PoolDocument],
    *,
    owner_id: int,
    embedding_id: str,
    expected_filenames: Mapping[str, str] | None = None,
) -> DocumentMap:
    """Verify the owner's document rows against the manifest and return the id <-> key mapping.

    Raises ``EvaluationDataError`` with every finding; ``TypeError`` for a wrongly typed argument.
    """
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    if not _plain_int(owner_id) or type(embedding_id) is not str or not embedding_id:
        raise TypeError("owner_id must be an int and embedding_id a non-empty str")
    if expected_filenames is not None and not isinstance(expected_filenames, Mapping):
        raise TypeError("expected_filenames must be a Mapping or None")
    rows = list(documents)

    if manifest.stage != "ingested" or any(
        type(d.extracted_text_sha256) is not str or not d.extracted_text_sha256 for d in manifest.documents
    ):
        raise EvaluationDataError([_issue(DOC_MANIFEST, "manifest")])

    issues: list[Issue] = []
    wanted = {_expected_description(manifest, d.corpus_key): d.corpus_key for d in manifest.documents}
    by_key: dict[str, list[tuple[int, PoolDocument]]] = {d.corpus_key: [] for d in manifest.documents}
    filename_keys = {name: key for key, name in (expected_filenames or {}).items()}

    seen_ids: set[int] = set()
    for index, row in enumerate(rows):
        where = f"documents[{index}]"
        if not _document_fields_ok(row):
            issues.append(_issue(DOC_FIELD_INVALID, where))
            continue
        if row.document_id in seen_ids:
            issues.append(_issue(DOC_DUPLICATE_ID, where))
            continue
        seen_ids.add(row.document_id)
        key = wanted.get(row.description) if row.description is not None else None
        if key is None:
            # the right file under a wrong description is a mismatch, anything else is unknown
            code = DOC_FILENAME_MISMATCH if row.filename in filename_keys else DOC_UNKNOWN
            issues.append(_issue(code, where))
            continue
        if expected_filenames is not None and expected_filenames.get(key) != row.filename:
            issues.append(_issue(DOC_FILENAME_MISMATCH, where))
            continue
        by_key[key].append((index, row))

    chosen: dict[str, PoolDocument] = {}
    for position, document in enumerate(manifest.documents):
        where = f"manifest.documents[{position}]"
        matches = by_key[document.corpus_key]
        if not matches:
            issues.append(_issue(DOC_MISSING, where))
            continue
        if len(matches) > 1:
            issues.append(_issue(DOC_AMBIGUOUS, where))
            continue
        index, row = matches[0]
        found = len(issues)
        if row.owner_id != owner_id:
            issues.append(_issue(DOC_FOREIGN_OWNER, f"documents[{index}]"))
        if row.indexed_source_version is None:
            issues.append(_issue(DOC_NOT_PUBLISHED, f"documents[{index}]"))
        elif row.indexed_source_version != row.source_version:
            issues.append(_issue(DOC_STALE_VERSION, f"documents[{index}]"))
        elif row.indexed_embedding_id != embedding_id:
            issues.append(_issue(DOC_EMBEDDING_MISMATCH, f"documents[{index}]"))
        if row.source_sha256 != document.extracted_text_sha256:
            issues.append(_issue(DOC_SOURCE_HASH, f"documents[{index}]"))
        if len(issues) == found:
            chosen[document.corpus_key] = row

    if issues:
        raise EvaluationDataError(issues)
    ordered = {d.corpus_key: chosen[d.corpus_key] for d in manifest.documents}
    return DocumentMap(
        corpus_version=manifest.corpus_version,
        owner_id=owner_id,
        embedding_id=embedding_id,
        key_by_id=MappingProxyType({row.document_id: key for key, row in ordered.items()}),
        id_by_key=MappingProxyType({key: row.document_id for key, row in ordered.items()}),
        source_version_by_id=MappingProxyType({row.document_id: row.indexed_source_version for row in ordered.values()}),
        source_sha256_by_key=MappingProxyType({key: row.source_sha256 for key, row in ordered.items()}),
    )


# ---------- chunks ----------

@dataclass(frozen=True, slots=True)
class _Fields:
    chunk_id: int
    document_id: int
    chunk_index: int
    start: int
    end: int
    text: str
    span_sha256: str
    source_version: int
    source_sha256: str


def _fields(chunk: Any) -> _Fields | None:
    """The fields a ``RetrievedChunk`` (or a pool row with the same names) must carry, or ``None``."""
    try:
        values = _Fields(
            chunk.chunk_id, chunk.document_id, chunk.chunk_index, chunk.start_offset, chunk.end_offset,
            chunk.text, chunk.span_sha256, chunk.source_version, chunk.source_sha256,
        )
    except AttributeError:
        return None
    ints = (values.chunk_id, values.document_id, values.chunk_index, values.start, values.end, values.source_version)
    if not all(_plain_int(v) for v in ints):
        return None
    if type(values.text) is not str or type(values.span_sha256) is not str or type(values.source_sha256) is not str:
        return None
    if values.chunk_index < 0 or not 0 <= values.start < values.end:
        return None
    return values


def _ranked(values: _Fields, corpus_key: str) -> RankedChunk:
    return RankedChunk(corpus_key, values.start, values.end, values.span_sha256)


def _self_consistent(values: _Fields) -> bool:
    return len(values.text) == values.end - values.start and sha256_text(values.text) == values.span_sha256


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Every published chunk of the corpus, verified. ``ranked`` is in canonical order."""

    ranked: tuple[RankedChunk, ...]
    by_chunk_id: Mapping[int, RankedChunk]
    document_map: DocumentMap
    pool_sha256: str

    def __len__(self) -> int:
        return len(self.ranked)


def _coverage_problem(text: str, ranges: Sequence[tuple[int, int]]) -> str | None:
    """``POOL_*`` code for ranges that do not validly cover ``text``, or ``None`` (pure)."""
    for position, (start, end) in enumerate(ranges):
        if end > len(text) or (position and not (start > ranges[position - 1][0] and end > ranges[position - 1][1])):
            return POOL_RANGE_ORDER
    cursor = 0
    for start, end in ranges:
        if any(not is_ignorable(text[position]) for position in range(cursor, start)):
            return POOL_COVERAGE
        cursor = max(cursor, end)
    if any(not is_ignorable(text[position]) for position in range(cursor, len(text))):
        return POOL_COVERAGE
    return None


def build_pool(
    manifest: Manifest,
    document_map: DocumentMap,
    chunks: Iterable[Any],
    texts: Mapping[str, str],
) -> CandidatePool:
    """Verify ``chunks`` (every published chunk of the corpus) and return the candidate pool.

    ``texts`` are the verified source texts (``DevInputs.texts``). Raises ``EvaluationDataError``
    with every finding; ``TypeError`` for a wrongly typed argument.
    """
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    if not isinstance(document_map, DocumentMap):
        raise TypeError("document_map must be a DocumentMap")
    if not isinstance(texts, Mapping):
        raise TypeError("texts must be a Mapping")
    rows = list(chunks)

    issues: list[Issue] = []
    for position, document in enumerate(manifest.documents):
        if document.corpus_key not in texts or document.corpus_key not in document_map.id_by_key:
            issues.append(_issue(POOL_TEXT_MISSING, f"manifest.documents[{position}]"))
    if issues:
        raise EvaluationDataError(issues)

    seen_ids: set[int] = set()
    seen_slots: set[tuple[int, int]] = set()
    seen_keys: set[tuple[str, int, int, str]] = set()
    per_document: dict[str, list[tuple[int, int, int]]] = {d.corpus_key: [] for d in manifest.documents}
    accepted: list[tuple[_Fields, RankedChunk]] = []

    for index, row in enumerate(rows):
        where = f"chunks[{index}]"
        values = _fields(row)
        if values is None:
            issues.append(_issue(POOL_CHUNK_INVALID, where))
            continue
        key = document_map.key_by_id.get(values.document_id)
        if key is None:
            issues.append(_issue(POOL_FOREIGN_DOCUMENT, where))
            continue
        bad = len(issues)
        if values.source_version != document_map.source_version_by_id[values.document_id]:
            issues.append(_issue(POOL_STALE_VERSION, where))
        if values.source_sha256 != document_map.source_sha256_by_key[key]:
            issues.append(_issue(POOL_SOURCE_HASH, where))
        if not _self_consistent(values):
            issues.append(_issue(POOL_INTEGRITY, where))
        elif texts[key][values.start:values.end] != values.text:
            issues.append(_issue(POOL_TEXT_MISMATCH, where))
        slot = (values.document_id, values.chunk_index)
        canonical = (key, values.start, values.end, values.span_sha256)
        if values.chunk_id in seen_ids or slot in seen_slots or canonical in seen_keys:
            issues.append(_issue(POOL_DUPLICATE_CHUNK, where))
        seen_ids.add(values.chunk_id)
        seen_slots.add(slot)
        seen_keys.add(canonical)
        if len(issues) != bad:
            continue
        accepted.append((values, _ranked(values, key)))
        per_document[key].append((values.chunk_index, values.start, values.end))

    if issues:
        raise EvaluationDataError(issues)

    for position, document in enumerate(manifest.documents):
        where = f"manifest.documents[{position}]"
        entries = sorted(per_document[document.corpus_key])
        if not entries:
            issues.append(_issue(POOL_DOCUMENT_EMPTY, where))
            continue
        if [e[0] for e in entries] != list(range(len(entries))):
            issues.append(_issue(POOL_INDEX_GAP, where))
            continue
        problem = _coverage_problem(texts[document.corpus_key], [(e[1], e[2]) for e in entries])
        if problem is not None:
            issues.append(_issue(problem, where))
    if issues:
        raise EvaluationDataError(issues)

    try:
        ordered = canonical_order([ranked for _, ranked in accepted])
    except ValueError:  # same canonical key: already reported above, kept as a final guard
        raise EvaluationDataError([_issue(POOL_DUPLICATE_CHUNK, "chunks")]) from None
    digest = hashlib.sha256(
        json.dumps([[c.corpus_key, c.start, c.end, c.text_sha256] for c in ordered], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CandidatePool(
        ranked=ordered,
        by_chunk_id=MappingProxyType({values.chunk_id: ranked for values, ranked in accepted}),
        document_map=document_map,
        pool_sha256=digest,
    )


def convert_retrieved(pool: CandidatePool, retrieved: Iterable[Any]) -> tuple[RankedChunk, ...]:
    """Convert a dense ranking (``RetrievedChunk`` objects, best first) to ``RankedChunk``.

    Every chunk must be in the pool, with the same document, range, hash and source identity; a chunk
    may not appear twice. Rank order is kept; the score is not used. Raises ``EvaluationDataError``.
    """
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    issues: list[Issue] = []
    result: list[RankedChunk] = []
    seen: set[int] = set()
    for index, row in enumerate(retrieved):
        where = f"retrieved[{index}]"
        values = _fields(row)
        if values is None:
            issues.append(_issue(POOL_CHUNK_INVALID, where))
            continue
        key = pool.document_map.key_by_id.get(values.document_id)
        in_pool = pool.by_chunk_id.get(values.chunk_id)
        if key is None or in_pool is None:
            issues.append(_issue(RANK_NOT_IN_POOL, where))
            continue
        if values.chunk_id in seen:
            issues.append(_issue(RANK_DUPLICATE, where))
            continue
        seen.add(values.chunk_id)
        converted = _ranked(values, key)
        if (
            converted != in_pool
            or not _self_consistent(values)
            or values.source_version != pool.document_map.source_version_by_id[values.document_id]
            or values.source_sha256 != pool.document_map.source_sha256_by_key[key]
        ):
            issues.append(_issue(RANK_MISMATCH, where))
            continue
        result.append(converted)
    if issues:
        raise EvaluationDataError(issues)
    return tuple(result)
