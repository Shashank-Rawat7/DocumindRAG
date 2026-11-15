"""Corpus ingestion driver for the six E1.5 evaluation documents (library; no command line).

What it does
  * ``load_corpus``   reads the manifest, maps every manifest entry to exactly one corpus file by
                      SHA-256 of the file bytes, and computes the expected extracted text with the
                      application's own ``extract_text`` (offline; no database).
  * ``inspect``       read-only: classifies every document against the database.
  * ``run_ingest``    ``execute=False`` is a read-only plan; ``execute=True`` creates the missing
                      documents through the production functions ``storage.save_upload``,
                      ``crud.create_uploaded_document`` and ``processing.process_document``.
  * ``verify``        read-only: checks ingestion, extraction and indexing of every document.
  * ``build_candidate`` / ``check_candidate``  produce and validate a candidate ``ingested`` manifest
                      (as text; this module never writes a manifest).

What it deliberately does not do: index anything (indexing is the unmodified backfill CLI's job; no
indexing module is imported here), delete, overwrite, adopt orphan files, re-extract, or retry.
Every state it did not itself create in this run, and does not recognise as complete, is reported
and the run stops; recovery is manual.

Hash semantics (never interchangeable)
  * ``source_file_sha256`` (manifest): SHA-256 of the FILE BYTES. Compared with the corpus file and
    the stored copy only.
  * ``extracted_text_sha256`` / ``documents.source_sha256``: SHA-256 of the extracted TEXT (UTF-8).
    The expected value is computed offline from the corpus file with the same ``extract_text``;
    for PDFs it depends on the installed pypdf version.

Nothing returned or printed contains a password, a URL, an email, extracted text or an embedding.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import text
from sqlalchemy.orm import configure_mappers

from app.core import processing, storage
from app.crud.document import create_uploaded_document
from app.models.chunk import EMBEDDING_DIM
from app.models.user import User  # noqa: F401  (registers the mapper that Document.owner refers to)
from app.rag.backfill_safety import check_live_database
from app.rag.backfill_wiring import write_target_failure
from app.rag.chunking import sha256_text
from app.rag.evaluation.parsing import EXTRACTION_LIBRARIES, parse_manifest
from app.rag.evaluation.schema import EvaluationDataError, Manifest
from app.rag.evaluation.validation import validate_manifest
from app.rag.text_boundaries import is_ignorable

CORPUS_VERSION = "eval-corpus-v1"
EXPECTED_DOCUMENT_COUNT = 6
CONTENT_TYPES = {"pdf": "application/pdf", "md": "text/markdown", "txt": "text/plain"}
# Files in the corpus directory that are not corpus documents.
NON_CORPUS_FILES = frozenset({"LICENSES.md", "manifest.json"})

# One advisory lock for the whole driver. Deterministic, so every invocation uses the same key.
LOCK_KEY = int.from_bytes(hashlib.sha256(b"documind:e15:corpus-ingest-v1").digest()[:8], "big", signed=True)

ACTION_CREATE = "create"
ACTION_SKIP = "skip"
ACTION_REFUSE = "refuse"

_READ_CHUNK = 1024 * 1024


class CorpusError(Exception):
    """The corpus could not be loaded. ``problems`` are fixed, safe strings."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


# ---------- hashing ----------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> tuple[str, int]:
    """(SHA-256 hex of the file bytes, size). Streams the file."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# ---------- the corpus ----------

@dataclass(frozen=True, slots=True)
class CorpusFile:
    corpus_key: str
    format: str
    filename: str                 # the corpus file's own name; stored as documents.filename
    path: Path
    sha256: str                   # SHA-256 of the file bytes (== manifest source_file_sha256)
    size: int
    content_type: str
    description: str              # "<corpus_version>:<corpus_key>"
    expected_text_sha256: str     # SHA-256 of the extracted text, computed offline
    expected_text_length: int


@dataclass(frozen=True, slots=True)
class Corpus:
    manifest: Manifest
    manifest_text: str
    files: tuple[CorpusFile, ...]


def load_corpus(
    manifest_path: Path, corpus_dir: Path, *, expected_count: int = EXPECTED_DOCUMENT_COUNT
) -> Corpus:
    """Load and cross-check the manifest and the corpus files. Raises CorpusError."""
    try:
        manifest_text = Path(manifest_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise CorpusError(["the manifest could not be read"]) from None
    try:
        manifest = parse_manifest(manifest_text)
        issues = list(validate_manifest(manifest))
    except EvaluationDataError as exc:
        issues = list(exc.issues)
    if issues:
        raise CorpusError([f"manifest {i.code} at {i.location}" for i in issues])

    problems: list[str] = []
    if manifest.corpus_version != CORPUS_VERSION:
        problems.append(f"the manifest corpus_version is not {CORPUS_VERSION!r}")
    if len(manifest.documents) != expected_count:
        problems.append(f"the manifest lists {len(manifest.documents)} documents, expected {expected_count}")
    for document in manifest.documents:
        if document.format not in CONTENT_TYPES:
            problems.append(f"{document.corpus_key}: format {document.format!r} is not supported by this driver")
    if problems:
        raise CorpusError(problems)

    # Inspect EVERY entry. Only the manifest-matched corpus files, LICENSES.md and manifest.json are
    # allowed; anything else (any extension, a directory, a symbolic link) is refused. Links are
    # never followed.
    hashed: list[tuple[Path, str, int]] = []
    try:
        entries = sorted(Path(corpus_dir).iterdir())
        for entry in entries:
            if entry.is_symlink():
                problems.append(f"symbolic link in the corpus directory (refused, not followed): {entry.name}")
            elif not entry.is_file():
                problems.append(f"unexpected non-file entry in the corpus directory: {entry.name}")
            elif entry.name in NON_CORPUS_FILES:
                continue
            elif entry.suffix.lower() not in (".pdf", ".txt", ".md"):
                problems.append(f"unexpected file in the corpus directory: {entry.name}")
            else:
                hashed.append((entry, *hash_file(entry)))
    except OSError:
        raise CorpusError(["the corpus directory could not be read"]) from None

    files: list[CorpusFile] = []
    used: set[Path] = set()
    for document in manifest.documents:
        matches = [
            (entry, sha, size) for entry, sha, size in hashed
            if sha == document.source_file_sha256 and entry.suffix.lower() == f".{document.format}"
        ]
        if len(matches) != 1:
            problems.append(
                f"{document.corpus_key}: {len(matches)} corpus files match the manifest source hash and format "
                "(exactly one is required)"
            )
            continue
        entry, sha, size = matches[0]
        if entry in used:
            problems.append(f"{document.corpus_key}: the file {entry.name} is claimed by more than one manifest entry")
            continue
        used.add(entry)
        try:
            extracted = processing.extract_text(entry)
        except Exception as exc:  # noqa: BLE001 - class name only
            problems.append(f"{document.corpus_key}: offline text extraction failed ({type(exc).__name__})")
            continue
        files.append(
            CorpusFile(
                corpus_key=document.corpus_key,
                format=document.format,
                filename=entry.name,
                path=entry,
                sha256=sha,
                size=size,
                content_type=CONTENT_TYPES[document.format],
                description=f"{manifest.corpus_version}:{document.corpus_key}",
                expected_text_sha256=sha256_text(extracted),
                expected_text_length=len(extracted),
            )
        )
    for entry, _sha, _size in hashed:
        if entry not in used:
            problems.append(f"unexpected file in the corpus directory: {entry.name}")
    if problems:
        raise CorpusError(problems)
    return Corpus(manifest=manifest, manifest_text=manifest_text, files=tuple(files))


# ---------- read-only database access ----------

@contextmanager
def read_only(engine: Any) -> Iterator[Any]:
    """A connection whose transaction starts read-only; it is always rolled back and closed."""
    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        try:
            yield connection
        finally:
            try:
                connection.rollback()
            except Exception:  # noqa: BLE001
                pass


_ROWS_SQL = text(
    """
    SELECT d.id, d.owner_id, d.filename, d.description, d.stored_name, d.content_type, d.size_bytes,
           d.processing_status, d.processing_error,
           (d.extracted_text IS NOT NULL) AS has_text,
           encode(sha256(convert_to(d.extracted_text, 'UTF8')), 'hex') AS text_sha,
           length(d.extracted_text) AS text_len,
           d.source_version, d.source_sha256,
           d.indexing_status, d.indexing_error, d.indexing_lease_expires_at,
           d.indexed_source_version, d.indexed_embedding_id, d.indexed_signature,
           (SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunk_count
    FROM documents d
    WHERE d.filename = ANY(:filenames) OR d.description = ANY(:descriptions)
    ORDER BY d.id
    """
)

_OWNER_SQL = text(
    "SELECT 1 FROM users WHERE id = :owner_id "
    "AND encode(sha256(convert_to(email, 'UTF8')), 'hex') = :digest"
)

_OTHER_DOCUMENTS_SQL = text(
    "SELECT id FROM documents WHERE owner_id = :owner_id "
    "AND (description IS NULL OR NOT (description = ANY(:descriptions))) ORDER BY id"
)

_CHUNKS_SQL = text(
    """
    SELECT count(c.id) AS n,
           min(c.chunk_index) AS first_index, max(c.chunk_index) AS last_index,
           count(DISTINCT c.chunk_index) AS distinct_indexes,
           count(c.id) FILTER (WHERE c.embedding IS NULL) AS null_embeddings,
           count(c.id) FILTER (WHERE vector_dims(c.embedding) <> :dim) AS wrong_dimension,
           count(c.id) FILTER (WHERE c.text IS DISTINCT FROM
                 substr(d.extracted_text, c.start_offset + 1, c.end_offset - c.start_offset)) AS text_mismatch,
           count(c.id) FILTER (WHERE c.span_sha256 <> encode(sha256(convert_to(c.text, 'UTF8')), 'hex')) AS span_mismatch,
           count(c.id) FILTER (WHERE c.page_start IS NOT NULL) AS with_pages,
           min(c.page_start) AS min_page, max(c.page_end) AS max_page
    FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id
    WHERE d.id = :document_id
    GROUP BY d.id
    """
)

_RANGES_SQL = text(
    "SELECT start_offset, end_offset FROM document_chunks WHERE document_id = :document_id ORDER BY chunk_index"
)

_TEXT_SQL = text("SELECT extracted_text FROM documents WHERE id = :document_id")


def fetch_rows(conn: Any, files: Sequence[CorpusFile]) -> list[dict]:
    """Every document, of any owner, whose filename or description matches one of ``files``."""
    result = conn.execute(
        _ROWS_SQL,
        {"filenames": [f.filename for f in files], "descriptions": [f.description for f in files]},
    )
    return [dict(row) for row in result.mappings().all()]


def _matches(file: CorpusFile, rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r["filename"] == file.filename or r["description"] == file.description]


def _stored_file_problem(file: CorpusFile, row: dict, upload_root: Path) -> str | None:
    """None if the stored copy exists and has the corpus file's bytes; otherwise a fixed text."""
    stored_name = row["stored_name"]
    if not stored_name:
        return "no stored file name"
    try:
        path = (Path(upload_root) / stored_name).resolve()
        if path.parent != Path(upload_root).resolve():
            return "the stored file name is not inside the upload directory"
        sha, size = hash_file(path)
    except (OSError, ValueError):
        return "the stored file is missing or unreadable"
    if size != file.size or sha != file.sha256:
        return "the stored file differs from the corpus file (SHA-256 or size)"
    return None


# ---------- plan ----------

@dataclass(frozen=True, slots=True)
class DocPlan:
    corpus_key: str
    filename: str
    code: str
    action: str
    document_id: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class Plan:
    documents: tuple[DocPlan, ...]
    problems: tuple[str, ...]          # conditions that are not about one document
    orphans: tuple[str, ...]           # stored files with a corpus hash and no document row

    @property
    def blocked(self) -> bool:
        return bool(self.problems or self.orphans or any(d.action == ACTION_REFUSE for d in self.documents))

    @property
    def to_create(self) -> tuple[DocPlan, ...]:
        return tuple(d for d in self.documents if d.action == ACTION_CREATE)


def classify(file: CorpusFile, rows: Sequence[dict], owner_id: int, upload_root: Path) -> DocPlan:
    """Decide what to do for one corpus file. Pure apart from reading the stored file."""

    def plan(code: str, action: str, row: dict | None, detail: str) -> DocPlan:
        return DocPlan(file.corpus_key, file.filename, code, action, row["id"] if row else None, detail)

    matches = _matches(file, rows)
    if not matches:
        return plan("new", ACTION_CREATE, None, "no matching document; it will be created")
    if len(matches) > 1:
        ids = ",".join(str(m["id"]) for m in matches)
        return plan("duplicate", ACTION_REFUSE, None, f"{len(matches)} documents match (ids {ids})")
    row = matches[0]
    if row["owner_id"] != owner_id:
        return plan("conflict_owner", ACTION_REFUSE, row, "the matching document belongs to another user")
    if row["filename"] != file.filename or row["description"] != file.description:
        return plan(
            "conflict_identity", ACTION_REFUSE, row,
            "the filename or the corpus-key description matches, but not both",
        )
    state = row["processing_status"] or "unknown"
    if state != "completed":
        return plan(
            f"partial_{state}", ACTION_REFUSE, row,
            f"extraction is {state}; recovery is manual (nothing is retried or deleted)",
        )
    if row["processing_error"] is not None:
        return plan(
            "inconsistent_error", ACTION_REFUSE, row,
            "marked completed but carries a processing error; recovery is manual",
        )
    problem = _stored_file_problem(file, row, upload_root)
    if problem:
        return plan("mismatch_stored_file", ACTION_REFUSE, row, problem)
    if row["content_type"] != file.content_type or row["size_bytes"] != file.size:
        return plan("mismatch_metadata", ACTION_REFUSE, row, "content type or size differs from the corpus file")
    if (
        not row["has_text"]
        or row["text_sha"] != file.expected_text_sha256
        or row["text_len"] != file.expected_text_length
        or row["source_version"] != 1
    ):
        return plan("mismatch_text", ACTION_REFUSE, row, "extracted text, its hash or source_version is not as expected")
    return plan(
        "present_extracted", ACTION_SKIP, row,
        f"already extracted; indexing_status={row['indexing_status']}, chunks={row['chunk_count']} (not touched)",
    )


def find_orphans(conn: Any, files: Sequence[CorpusFile], upload_root: Path) -> list[str]:
    """Stored files whose bytes equal a corpus file and that no document row references."""
    sizes = {f.size for f in files}
    hashes = {f.sha256 for f in files}
    names: list[str] = []
    with os.scandir(upload_root) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False) and entry.stat().st_size in sizes:
                if hash_file(Path(entry.path))[0] in hashes:
                    names.append(entry.name)
    if not names:
        return []
    referenced = {
        name for (name,) in conn.execute(
            text("SELECT stored_name FROM documents WHERE stored_name = ANY(:names)"), {"names": names}
        )
    }
    return sorted(set(names) - referenced)


def _owner_problems(conn: Any, files: Sequence[CorpusFile], owner_id: int, owner_sha256: str) -> list[str]:
    problems: list[str] = []
    if conn.execute(_OWNER_SQL, {"owner_id": owner_id, "digest": owner_sha256}).first() is None:
        problems.append("the evaluation owner does not exist or its email digest does not match")
        return problems
    others = [
        row[0] for row in conn.execute(
            _OTHER_DOCUMENTS_SQL, {"owner_id": owner_id, "descriptions": [f.description for f in files]}
        )
    ]
    if others:
        problems.append(f"the evaluation owner owns {len(others)} document(s) outside the corpus (ids {others})")
    return problems


def inspect(
    conn: Any, files: Sequence[CorpusFile], *, owner_id: int, owner_sha256: str, upload_root: Path
) -> Plan:
    """Read-only classification of every document, the owner and the upload directory."""
    problems = _owner_problems(conn, files, owner_id, owner_sha256)
    rows = fetch_rows(conn, files)
    documents = tuple(classify(f, rows, owner_id, upload_root) for f in files)
    try:
        orphans = tuple(find_orphans(conn, files, upload_root))
    except OSError:
        orphans = ()
        problems.append("the upload directory could not be read")
    return Plan(documents=documents, problems=tuple(problems), orphans=orphans)


# ---------- the write gate ----------

def _same_path(first: Path, second: Path) -> bool:
    def normal(path: Path) -> str:
        return os.path.normcase(os.path.normpath(os.fspath(path)))

    return normal(Path(first).resolve()) == normal(Path(second).resolve())


def live_database_failure(engine: Any, expected_database: str) -> str | None:
    """None if the server really is ``expected_database`` (one SELECT current_database())."""
    try:
        with read_only(engine) as conn:
            failure = check_live_database(conn, expected_database)
    except Exception as exc:  # noqa: BLE001 - class name only
        return f"could not connect to the database ({type(exc).__name__})"
    return failure.message if failure is not None else None


def write_gate_problems(
    engine: Any, session_factory: Callable[[], Any], expected_database: str, upload_root: Path
) -> list[str]:
    """Everything that must hold before the first write. An empty list means the gate is open.

    The live database name, the session factory used to create documents, the session factory that
    ``process_document`` uses internally (``processing.SessionLocal``) and the upload directory that
    ``storage.save_upload`` writes to are all proved to be the verified target.
    """
    problems: list[str] = []
    try:
        configure_mappers()  # a broken ORM mapping must refuse BEFORE a file is stored
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the ORM models do not configure ({type(exc).__name__}: {str(exc)[:200]})")
    failure = live_database_failure(engine, expected_database)
    if failure:
        problems.append(failure)
    binding = write_target_failure(engine, session_factory, processing.SessionLocal)
    if binding:
        problems.append(f"write target: {binding}")
    if not _same_path(storage.UPLOAD_ROOT, upload_root):
        problems.append("storage.UPLOAD_ROOT is not the verified upload directory")
    return problems


# ---------- advisory lock ----------

@contextmanager
def advisory_lock(engine: Any) -> Iterator[bool]:
    """Hold the driver's session-level advisory lock; yields False if another run holds it.

    The lock lives on a dedicated autocommit connection for the whole block, so it spans every
    commit the pipeline makes. It is released explicitly and, as a backstop, by closing the connection.
    """
    connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    acquired = False
    try:
        acquired = bool(connection.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": LOCK_KEY}).scalar_one())
        yield acquired
    finally:
        try:
            if acquired:
                connection.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
        finally:
            connection.close()


# ---------- ingest ----------

@dataclass(frozen=True, slots=True)
class DocResult:
    corpus_key: str
    status: str
    document_id: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    outcome: str          # refused | locked | dry_run | nothing_to_do | ingested | stopped
    plan: Plan | None
    documents: tuple[DocResult, ...]
    problems: tuple[str, ...]


def _refused(problems: Sequence[str], plan: Plan | None = None) -> IngestResult:
    return IngestResult("refused", plan, (), tuple(problems))


def _fresh_state_problem(file: CorpusFile, rows: Sequence[dict], owner_id: int, expect_row: int | None) -> str | None:
    """Check one document immediately before a state-changing step.

    ``expect_row`` None: no document may match. Otherwise exactly that row must match and be an
    untouched pending row.
    """
    matches = _matches(file, rows)
    if expect_row is None:
        return None if not matches else "a matching document appeared"
    if len(matches) != 1 or matches[0]["id"] != expect_row:
        return "the document row is not the one just created"
    row = matches[0]
    if (
        row["owner_id"] != owner_id
        or row["filename"] != file.filename
        or row["description"] != file.description
        or row["processing_status"] != "pending"
        or row["has_text"]
        or row["chunk_count"] != 0
        or row["indexing_status"] is not None
        or row["source_version"] != 1
        or row["processing_error"] is not None
    ):
        return "the document row is not an untouched pending row"
    return None


def run_ingest(
    *,
    engine: Any,
    session_factory: Callable[[], Any],
    files: Sequence[CorpusFile],
    owner_id: int,
    owner_sha256: str,
    upload_root: Path,
    expected_database: str,
    execute: bool,
    process: Callable[[int], None] | None = None,
) -> IngestResult:
    """Plan (``execute=False``, read-only) or ingest (``execute=True``) the corpus documents.

    Nothing is deleted, overwritten, re-extracted or retried. The run stops at the first problem.
    """
    process = process if process is not None else processing.process_document

    gate = write_gate_problems(engine, session_factory, expected_database, upload_root)
    if gate:
        return _refused(gate)

    with read_only(engine) as conn:
        plan = inspect(conn, files, owner_id=owner_id, owner_sha256=owner_sha256, upload_root=upload_root)
    if plan.blocked:
        return IngestResult("refused", plan, (), ())
    if not plan.to_create:
        return IngestResult("nothing_to_do", plan, (), ())
    if not execute:
        return IngestResult("dry_run", plan, (), ())

    with advisory_lock(engine) as acquired:
        if not acquired:
            return IngestResult("locked", plan, (), ())

        # Under the lock: look again. The plan above may be stale.
        with read_only(engine) as conn:
            plan = inspect(conn, files, owner_id=owner_id, owner_sha256=owner_sha256, upload_root=upload_root)
        if plan.blocked:
            return IngestResult("refused", plan, (), ())

        results: list[DocResult] = []
        todo = {d.corpus_key for d in plan.to_create}
        stopped = False
        for file in files:
            if stopped:
                results.append(DocResult(file.corpus_key, "not_attempted", None, "an earlier document stopped the run"))
                continue
            if file.corpus_key not in todo:
                results.append(DocResult(file.corpus_key, "skipped_present", None, "already extracted; untouched"))
                continue
            result = _ingest_one(
                engine, session_factory, file, owner_id=owner_id, upload_root=upload_root, process=process
            )
            results.append(result)
            if result.status != "created_extracted":
                stopped = True

    outcome = "stopped" if stopped else "ingested"
    return IngestResult(outcome, plan, tuple(results), ())


def _ingest_one(
    engine: Any,
    session_factory: Callable[[], Any],
    file: CorpusFile,
    *,
    owner_id: int,
    upload_root: Path,
    process: Callable[[int], None],
) -> DocResult:
    def fail(status: str, detail: str, document_id: int | None = None) -> DocResult:
        return DocResult(file.corpus_key, status, document_id, detail)

    # 1. The corpus file's bytes, read once; the same bytes are hashed and stored.
    try:
        data = file.path.read_bytes()
    except OSError:
        return fail("source_unreadable", "the corpus file could not be read")
    if len(data) != file.size or sha256_bytes(data) != file.sha256:
        return fail("source_changed", "the corpus file no longer matches the manifest hash; nothing was written")

    # 2. No document may match right now.
    with read_only(engine) as conn:
        problem = _fresh_state_problem(file, fetch_rows(conn, [file]), owner_id, None)
    if problem:
        return fail("state_changed", problem + "; nothing was written")

    # 3. Store the copy (production function), then re-hash what was stored.
    try:
        stored_name, stored_size = storage.save_upload(UploadFile(file=io.BytesIO(data), filename=file.filename))
    except Exception as exc:  # noqa: BLE001 - class name only
        return fail("store_failed", f"saving the file failed ({type(exc).__name__}); no document row was created")
    try:
        stored_sha, measured_size = hash_file(storage.get_file_path(stored_name))
    except (OSError, ValueError):
        return fail("store_failed", f"the stored file {stored_name} could not be read back; no row was created")
    if stored_size != file.size or measured_size != file.size or stored_sha != file.sha256:
        return fail("store_failed", f"the stored file {stored_name} does not match the corpus file; no row was created")

    # 4. Create the document row (production function).
    db = session_factory()
    try:
        document = create_uploaded_document(
            db,
            filename=file.filename,
            description=file.description,
            stored_name=stored_name,
            content_type=file.content_type,
            size_bytes=stored_size,
            owner_id=owner_id,
        )
        document_id = document.id
    except Exception as exc:  # noqa: BLE001
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return fail(
            "row_failed",
            f"creating the document row failed ({type(exc).__name__}); the stored file {stored_name} is an orphan",
        )
    finally:
        db.close()

    # 5. Immediately before extraction: re-hash the stored file and re-check the row.
    try:
        sha_again, size_again = hash_file(storage.get_file_path(stored_name))
    except (OSError, ValueError):
        return fail("stored_file_changed", "the stored file could not be read before extraction", document_id)
    if sha_again != file.sha256 or size_again != file.size:
        return fail("stored_file_changed", "the stored file changed before extraction; extraction was not run", document_id)
    with read_only(engine) as conn:
        problem = _fresh_state_problem(file, fetch_rows(conn, [file]), owner_id, document_id)
    if problem:
        return fail("state_changed", problem + "; extraction was not run", document_id)

    # 6. Extract (production function). Called once, only for the row created above.
    process(document_id)

    # 7. Read the result back.
    with read_only(engine) as conn:
        rows = [r for r in fetch_rows(conn, [file]) if r["id"] == document_id]
    if len(rows) != 1:
        return fail("failed_extraction", "the document could not be read back after extraction", document_id)
    row = rows[0]
    if row["processing_status"] != "completed":
        reason = row["processing_error"] or "no error text recorded"
        return fail("failed_extraction", f"extraction ended as {row['processing_status']}: {reason}", document_id)
    if row["text_sha"] != file.expected_text_sha256 or row["text_len"] != file.expected_text_length:
        return fail("text_mismatch", "the extracted text differs from the offline expectation", document_id)
    return DocResult(file.corpus_key, "created_extracted", document_id, "extracted; text hash matches the expectation")


# ---------- verify ----------

@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DocVerification:
    corpus_key: str
    document_id: int | None
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)


@dataclass(frozen=True, slots=True)
class VerifyReport:
    problems: tuple[str, ...]
    documents: tuple[DocVerification, ...]

    @property
    def ok(self) -> bool:
        return not self.problems and all(d.ok for d in self.documents)


def coverage_checks(text: str, ranges: Sequence[tuple[int, int]]) -> list["CheckResult"]:
    """The chunk-offset contract of app/rag/chunking.py, checked against ``text`` (pure).

    The chunker makes OVERLAPPING chunks and trims whitespace from both edges of each chunk, so
    chunks are neither adjacent nor gapless. What it guarantees, and what is checked here:
      * chunk_offsets_valid   every range is 0 <= start < end <= len(text); starts strictly
                              increase and ends strictly increase (overlap is allowed);
      * chunk_edges_trimmed   no chunk begins or ends with whitespace;
      * chunk_coverage        every character outside all chunks (before the first, between chunks,
                              after the last) is ignorable (whitespace, or Unicode Cc/Cf/Zs/Zl/Zp);
                              a meaningful character that no chunk contains is a failure.
    Details name offsets and counts only, never document text.
    """
    results: list[CheckResult] = []
    length = len(text)

    problems: list[str] = []
    if not ranges:
        problems.append("there are no chunks")
    for position, (start, end) in enumerate(ranges):
        if not (0 <= start < end <= length):
            problems.append(f"chunk {position} has an invalid range ({start}, {end}) for a text of length {length}")
        if position and not (start > ranges[position - 1][0] and end > ranges[position - 1][1]):
            problems.append(f"chunk {position} does not start and end after chunk {position - 1}")
    results.append(CheckResult("chunk_offsets_valid", not problems, "; ".join(problems[:3]) or f"{len(ranges)} range(s)"))
    if problems:
        return results  # the other two checks are meaningless on broken ranges

    untrimmed = [i for i, (s, e) in enumerate(ranges) if text[s:e] != text[s:e].strip()]
    results.append(CheckResult(
        "chunk_edges_trimmed", not untrimmed,
        f"chunk(s) {untrimmed[:5]} begin or end with whitespace" if untrimmed else "",
    ))

    uncovered: list[int] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            uncovered.extend(pos for pos in range(cursor, start) if not is_ignorable(text[pos]))
        cursor = max(cursor, end)
    uncovered.extend(pos for pos in range(cursor, length) if not is_ignorable(text[pos]))
    results.append(CheckResult(
        "chunk_coverage", not uncovered,
        f"{len(uncovered)} meaningful character(s) are in no chunk, first at offset {uncovered[0]}" if uncovered
        else "every meaningful character is in a chunk",
    ))
    return results


def verify(
    conn: Any,
    files: Sequence[CorpusFile],
    *,
    owner_id: int,
    owner_sha256: str,
    upload_root: Path,
    embedding_id: str,
) -> VerifyReport:
    """Read-only verification of ingestion, extraction and indexing of every corpus document.

    This does not re-run the chunker or recompute embeddings; it checks what is stored.
    """
    problems = _owner_problems(conn, files, owner_id, owner_sha256)
    rows = fetch_rows(conn, files)
    return VerifyReport(
        problems=tuple(problems),
        documents=tuple(_verify_one(conn, f, rows, owner_id, upload_root, embedding_id) for f in files),
    )


def _verify_one(
    conn: Any, file: CorpusFile, rows: Sequence[dict], owner_id: int, upload_root: Path, embedding_id: str
) -> DocVerification:
    checks: list[CheckResult] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append(CheckResult(name, bool(ok), detail))
        return bool(ok)

    matches = _matches(file, rows)
    exact = [r for r in matches if r["filename"] == file.filename and r["description"] == file.description]
    if not check("exactly_one_document", len(matches) == 1 and len(exact) == 1, f"{len(matches)} matching row(s)"):
        return DocVerification(file.corpus_key, None, tuple(checks))
    row = exact[0]
    check("owner", row["owner_id"] == owner_id)
    problem = _stored_file_problem(file, row, upload_root)
    check("stored_file_hash_and_size", problem is None, problem or "")
    check("content_type_and_size", row["content_type"] == file.content_type and row["size_bytes"] == file.size)
    check("extraction_completed", row["processing_status"] == "completed" and row["processing_error"] is None,
          f"processing_status={row['processing_status']}")
    check("text_sha256", row["has_text"] and row["text_sha"] == file.expected_text_sha256)
    check("text_length", row["text_len"] == file.expected_text_length)
    check("source_version", row["source_version"] == 1, f"source_version={row['source_version']}")
    check("indexing_status", row["indexing_status"] == "indexed" and row["indexing_error"] is None
          and row["indexing_lease_expires_at"] is None,
          f"indexing_status={row['indexing_status']}, error={'set' if row['indexing_error'] else 'none'}, "
          f"lease={'set' if row['indexing_lease_expires_at'] else 'none'}")
    check("indexed_source_version",
          row["indexed_source_version"] is not None and row["indexed_source_version"] == row["source_version"],
          f"indexed_source_version={row['indexed_source_version']}, source_version={row['source_version']}")
    check("indexed_embedding_id", row["indexed_embedding_id"] == embedding_id,
          f"expected {embedding_id!r}, found {row['indexed_embedding_id']!r}")
    check("indexed_signature", bool(row["indexed_signature"]),
          "present" if row["indexed_signature"] else "missing or empty")
    check("indexed_source_sha256", row["source_sha256"] == file.expected_text_sha256,
          "equals the expected extracted-text hash" if row["source_sha256"] == file.expected_text_sha256
          else f"source_sha256 is {'missing' if not row['source_sha256'] else 'different from the expected text hash'}")

    chunks = conn.execute(_CHUNKS_SQL, {"document_id": row["id"], "dim": EMBEDDING_DIM}).mappings().one()
    count = chunks["n"]
    check("chunks_contiguous",
          count > 0 and chunks["first_index"] == 0 and chunks["last_index"] == count - 1
          and chunks["distinct_indexes"] == count, f"{count} chunk(s)")
    ranges = [(r[0], r[1]) for r in conn.execute(_RANGES_SQL, {"document_id": row["id"]})]
    stored_text = conn.execute(_TEXT_SQL, {"document_id": row["id"]}).scalar_one() or ""
    checks.extend(coverage_checks(stored_text, ranges))
    check("chunk_text_matches_extracted_text", count > 0 and chunks["text_mismatch"] == 0,
          f"{chunks['text_mismatch']} mismatching chunk(s)")
    check("chunk_span_sha256", count > 0 and chunks["span_mismatch"] == 0,
          f"{chunks['span_mismatch']} mismatching chunk(s)")
    check("embeddings_present_and_384d",
          count > 0 and chunks["null_embeddings"] == 0 and chunks["wrong_dimension"] == 0,
          f"null={chunks['null_embeddings']}, wrong_dimension={chunks['wrong_dimension']}")
    if file.format == "pdf":
        # Reported, not gated: a chunk may legitimately have no page numbers.
        checks.append(CheckResult(
            "pdf_page_numbers_reported", True,
            f"{chunks['with_pages']} of {count} chunk(s) carry pages; range {chunks['min_page']}-{chunks['max_page']}",
        ))
    return DocVerification(file.corpus_key, row["id"], tuple(checks))


# ---------- candidate manifest (text only; never written here) ----------

def build_candidate(
    manifest_text: str,
    files: Sequence[CorpusFile],
    *,
    processing_ref: str,
    python_version: str,
    library_versions: dict[str, str],
) -> str:
    """The draft manifest with stage ``ingested`` and the three ingestion fields filled in.

    ``library_versions`` maps a library name (``pypdf``) to its version; only the libraries that
    apply to a document's format are written.
    """
    raw = json.loads(manifest_text)
    by_key = {f.corpus_key: f for f in files}
    raw["stage"] = "ingested"
    for document in raw["documents"]:
        file = by_key[document["corpus_key"]]
        extraction = {"processing_ref": processing_ref, "python": python_version}
        for name in EXTRACTION_LIBRARIES[file.format]:
            extraction[name] = library_versions[name]
        document["extracted_text_sha256"] = file.expected_text_sha256
        document["extracted_text_length"] = file.expected_text_length
        document["extraction"] = extraction
    return json.dumps(raw, indent=2, ensure_ascii=False) + "\n"


def check_candidate(original_text: str, candidate_text: str) -> list[str]:
    """Problems with a candidate manifest; empty means it is valid and changes only what it may."""
    problems: list[str] = []
    try:
        original = parse_manifest(original_text)
        candidate = parse_manifest(candidate_text)
    except EvaluationDataError as exc:
        return [f"parse {i.code} at {i.location}" for i in exc.issues]
    for label, manifest in (("original", original), ("candidate", candidate)):
        problems.extend(f"{label} {i.code} at {i.location}" for i in validate_manifest(manifest))
    if original.stage != "draft":
        problems.append("the original manifest is not a draft")
    if candidate.stage != "ingested":
        problems.append("the candidate manifest stage is not 'ingested'")
    if (original.schema_version, original.corpus_version) != (candidate.schema_version, candidate.corpus_version):
        problems.append("schema_version or corpus_version changed")
    stripped = [replace(d, extracted_text_sha256=None, extracted_text_length=None, extraction=None)
                for d in candidate.documents]
    if [d.corpus_key for d in original.documents] != [d.corpus_key for d in candidate.documents]:
        problems.append("the document list or its order changed")
    elif stripped != list(original.documents):
        problems.append("a field other than the three ingestion fields changed")
    return problems
