"""Read-only database access of the validity pilot (E1.7.5): pool query, parity check, dense retriever.

The module has two parts. The PURE part (no database import, usable in tests with fakes) holds
``documents_from_rows`` (document rows -> ``pool.PoolDocument``), ``PoolChunkRow`` and
``check_pool_parity``. The DATABASE part (SQLAlchemy and the models are imported only inside the
functions that need them) holds ``build_pool_statement``, ``DbSession`` and ``read_only_session``.

Read-only guarantees. ``read_only_session`` opens one connection and applies
``isolation_level="REPEATABLE READ"`` and ``postgresql_readonly=True`` BEFORE the first statement, so
PostgreSQL starts the transaction as ``READ ONLY``: it rejects any INSERT, UPDATE, DELETE or DDL. The
transaction is never committed; it is always rolled back and the connection closed. ``DbSession``
issues only SELECTs: its own statements are fixed text or the SELECT built by ``build_pool_statement``,
the corpus verification and the document-row query are the existing read-only ``corpus_ingest``
functions, the snapshot is ``snapshot.collect_snapshot`` (SELECTs only), and the dense retrieval is the
unmodified ``retrieval.retrieve_for_query`` (one SELECT). ``assert_read_only`` checks ``SHOW
transaction_read_only`` and the isolation level and refuses otherwise. A fresh read-only transaction
takes the post-run snapshot, because a REPEATABLE READ transaction cannot see later changes.

Pool parity (the pool and the dense retrieval must search the same chunks).
* ``build_pool_statement`` filters with the actual ``indexing_service.retrievable_filter(embedding_id)``
  and the owner predicate, exactly like ``retrieval._build_retrieval_statement`` (a test compiles both
  and compares the predicates). It is not the Python ``pool.is_retrievable`` copy.
* ``check_pool_parity`` then compares the SQL result with the Python reading of the document rows:
  a document that Python considers retrievable must have SQL chunks, and ALL of its stored chunks
  (the unfiltered ``chunk_count`` of the document row) must be in the SQL pool; a document that
  Python considers not retrievable must have none; no chunk may belong to an unmapped document.
* ``pool.build_pool`` verifies each chunk against the verified source text, and ``engine.run_pilot``
  refuses any dense chunk that is not in the pool, so a divergence cannot go unnoticed at run time.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.rag.evaluation.pool import PoolDocument, is_retrievable
from app.rag.evaluation.schema import SEVERITY_ERROR, EvaluationDataError, Issue

PARITY_PYTHON_ONLY = "PARITY_PYTHON_ONLY"
PARITY_SQL_ONLY = "PARITY_SQL_ONLY"
PARITY_COUNT = "PARITY_COUNT"
PARITY_FOREIGN = "PARITY_FOREIGN"
PARITY_INPUT = "PARITY_INPUT"
ROWS_INVALID = "ROWS_INVALID"
ROWS_SIGNATURE = "ROWS_SIGNATURE"

_MESSAGES = {
    PARITY_PYTHON_ONLY: "a document is retrievable by the Python rule but the SQL pool has none of its chunks",
    PARITY_SQL_ONLY: "the SQL pool has chunks of a document that the Python rule does not consider retrievable",
    PARITY_COUNT: "the SQL pool does not hold every stored chunk of a retrievable document",
    PARITY_FOREIGN: "the SQL pool has a chunk of a document that is not in the corpus mapping",
    PARITY_INPUT: "the parity check input is malformed",
    ROWS_INVALID: "a document row is missing a field or has a wrongly typed field",
    ROWS_SIGNATURE: "the corpus documents were not all indexed with the same embedding signature",
}


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_ERROR)


# ---------- pure part ----------

@dataclass(frozen=True, slots=True)
class PoolChunkRow:
    """One row of the pool query: the fields of ``RetrievedChunk`` except the score."""

    chunk_id: int
    document_id: int
    chunk_index: int
    start_offset: int
    end_offset: int
    page_start: int | None
    page_end: int | None
    text: str
    span_sha256: str
    source_version: int
    source_sha256: str


@dataclass(frozen=True, slots=True)
class DocumentFacts:
    """The mapped document rows, the stored chunk counts and the one embedding signature."""

    documents: tuple[PoolDocument, ...]
    chunk_counts: Mapping[int, int]
    signature: str


_ROW_FIELDS = (
    "id", "owner_id", "filename", "description", "source_version", "indexed_source_version",
    "indexed_embedding_id", "source_sha256", "indexed_signature", "chunk_count",
)


def documents_from_rows(rows: Sequence[Mapping[str, Any]]) -> DocumentFacts:
    """Convert ``corpus_ingest.fetch_rows`` rows to ``PoolDocument`` values.

    Raises ``EvaluationDataError`` for a row that lacks a field or has a wrongly typed one, and when
    the rows carry different (or no) ``indexed_signature``. Ownership and publication are judged
    later by ``pool.map_documents``, not here.
    """
    documents: list[PoolDocument] = []
    counts: dict[int, int] = {}
    signatures: set[str | None] = set()
    issues: list[Issue] = []
    for index, row in enumerate(rows):
        where = f"document_rows[{index}]"
        try:
            values = {name: row[name] for name in _ROW_FIELDS}
        except (KeyError, TypeError):
            issues.append(_issue(ROWS_INVALID, where))
            continue
        if type(values["chunk_count"]) is not int or type(values["id"]) is not int:
            issues.append(_issue(ROWS_INVALID, where))
            continue
        signatures.add(values["indexed_signature"] if type(values["indexed_signature"]) is str else None)
        documents.append(PoolDocument(
            values["id"], values["owner_id"], values["filename"], values["description"],
            values["source_version"], values["indexed_source_version"], values["indexed_embedding_id"],
            values["source_sha256"],
        ))
        counts[values["id"]] = values["chunk_count"]
    if issues:
        raise EvaluationDataError(issues)
    if len(signatures) != 1 or None in signatures or "" in signatures:
        raise EvaluationDataError([_issue(ROWS_SIGNATURE, "document_rows")])
    return DocumentFacts(tuple(documents), counts, next(iter(signatures)))


def check_pool_parity(
    documents: Sequence[PoolDocument],
    chunk_counts: Mapping[int, int],
    embedding_id: str,
    chunks: Sequence[Any],
) -> None:
    """Compare the SQL pool with the Python reading of the document rows (see the module docs).

    ``chunk_counts`` are the unfiltered numbers of stored chunks per document id; ``chunks`` are the
    rows of the pool query. Raises ``EvaluationDataError`` with every disagreement.
    """
    if (
        isinstance(documents, (str, bytes)) or not isinstance(chunk_counts, Mapping)
        or type(embedding_id) is not str or isinstance(chunks, (str, bytes))
        or not all(isinstance(d, PoolDocument) for d in documents)
    ):
        raise EvaluationDataError([_issue(PARITY_INPUT, "parity")])
    per_document: dict[int, int] = {}
    for chunk in chunks:
        document_id = getattr(chunk, "document_id", None)
        if type(document_id) is not int:
            raise EvaluationDataError([_issue(PARITY_INPUT, "parity")])
        per_document[document_id] = per_document.get(document_id, 0) + 1

    issues: list[Issue] = []
    known = {d.document_id for d in documents}
    for index, document in enumerate(documents):
        where = f"documents[{index}]"
        in_sql = per_document.get(document.document_id, 0)
        if is_retrievable(document, embedding_id):
            if in_sql == 0:
                issues.append(_issue(PARITY_PYTHON_ONLY, where))
            elif in_sql != chunk_counts.get(document.document_id):
                issues.append(_issue(PARITY_COUNT, where))
        elif in_sql:
            issues.append(_issue(PARITY_SQL_ONLY, where))
    for position, document_id in enumerate(sorted(set(per_document) - known)):
        issues.append(_issue(PARITY_FOREIGN, f"pool_documents[{position}]"))
    if issues:
        raise EvaluationDataError(issues)


# ---------- database part ----------

def build_pool_statement(*, owner_id: int, embedding_id: str, document_ids: Sequence[int]):
    """The pool SELECT: every published chunk of the owner's given documents (no vector, no limit).

    Uses the same predicates as ``retrieval._build_retrieval_statement``: the owner and the real
    ``indexing_service.retrievable_filter(embedding_id)``; ``document_ids`` only narrows.
    """
    from sqlalchemy import select

    from app.models.chunk import DocumentChunk
    from app.models.document import Document
    from app.rag.indexing_service import retrievable_filter
    from app.rag.retrieval import (
        RetrievalRequestError,
        normalize_document_ids,
        validate_embedding_id,
        validate_owner_id,
    )

    owner_id = validate_owner_id(owner_id)
    embedding_id = validate_embedding_id(embedding_id)
    ids = normalize_document_ids(document_ids)
    if not ids:
        raise RetrievalRequestError("document_ids must not be empty")
    return (
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
            Document.indexed_source_version.label("source_version"),
            Document.source_sha256,
        )
        .select_from(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.owner_id == owner_id, retrievable_filter(embedding_id), Document.id.in_(ids))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )


class DbSession:
    """The operations of the pilot over ONE read-only REPEATABLE READ connection (SELECTs only)."""

    def __init__(self, connection: Any, *, upload_root: Path) -> None:
        self._connection = connection
        self._upload_root = Path(upload_root)

    def assert_read_only(self) -> str | None:
        """None if the transaction is READ ONLY and REPEATABLE READ; otherwise a fixed message."""
        from sqlalchemy import text

        try:
            read_only = self._connection.execute(text("SHOW transaction_read_only")).scalar_one()
            isolation = self._connection.execute(text("SHOW transaction_isolation")).scalar_one()
        except Exception as exc:  # noqa: BLE001 - only the class name is shown
            return f"the transaction mode could not be checked ({type(exc).__name__})"
        if read_only != "on":
            return "the database transaction is not read-only"
        if str(isolation).lower() != "repeatable read":
            return "the database transaction is not REPEATABLE READ"
        return None

    def check_live(self, expected_database: str) -> str | None:
        from app.rag.backfill_safety import check_live_database

        failure = check_live_database(self._connection, expected_database)
        return None if failure is None else failure.message

    def take_snapshot(self, clock: Callable[[], datetime]) -> dict:
        from app.rag.evaluation.snapshot import collect_snapshot

        def run_query(sql: str) -> list[dict]:
            return [dict(row) for row in self._connection.exec_driver_sql(sql).mappings().all()]

        return collect_snapshot(run_query, created_at=clock())

    def verify_corpus(self, files: Sequence[Any], *, owner_id: int, owner_sha256: str, embedding_id: str) -> Any:
        from app.rag import corpus_ingest as ci

        return ci.verify(
            self._connection, files, owner_id=owner_id, owner_sha256=owner_sha256,
            upload_root=self._upload_root, embedding_id=embedding_id,
        )

    def document_rows(self, files: Sequence[Any]) -> list[dict]:
        from app.rag import corpus_ingest as ci

        return ci.fetch_rows(self._connection, files)

    def pool_chunks(self, *, owner_id: int, embedding_id: str, document_ids: Sequence[int]) -> list[PoolChunkRow]:
        statement = build_pool_statement(owner_id=owner_id, embedding_id=embedding_id, document_ids=document_ids)
        return [
            PoolChunkRow(
                row.chunk_id, row.document_id, row.chunk_index, row.start_offset, row.end_offset,
                row.page_start, row.page_end, row.text, row.span_sha256, row.source_version, row.source_sha256,
            )
            for row in self._connection.execute(statement).all()
        ]

    def dense_retriever(
        self, embedder: Any, *, owner_id: int, document_ids: Sequence[int], top_k: int
    ) -> Callable[[str], list]:
        """The real dense retrieval: ``retrieval.retrieve_for_query`` over this connection (one SELECT)."""
        from app.rag.retrieval import retrieve_for_query

        ids = tuple(document_ids)

        def retrieve(query: str) -> list:
            return retrieve_for_query(
                self._connection, owner_id=owner_id, query=query, embedder=embedder, top_k=top_k, document_ids=ids
            )

        return retrieve


@contextmanager
def read_only_session(engine: Any, *, upload_root: Path) -> Iterator[DbSession]:
    """A ``DbSession`` over a connection whose transaction starts READ ONLY and REPEATABLE READ.

    The options are applied before the first statement; the transaction is never committed, always
    rolled back, and the connection is closed.
    """
    with engine.connect() as connection:
        connection = connection.execution_options(isolation_level="REPEATABLE READ", postgresql_readonly=True)
        try:
            yield DbSession(connection, upload_root=upload_root)
        finally:
            try:
                connection.rollback()
            except Exception:  # noqa: BLE001, S110
                pass
