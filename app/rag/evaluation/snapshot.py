"""Deterministic database snapshots and their comparison (E1.5.3).

A snapshot records every persisted row of ``users``, ``documents`` and ``document_chunks``
in a form that can be compared byte for byte, without ever containing a secret or a text:

  * users: ``email`` and ``hashed_password`` appear only as SHA-256 digests;
  * documents: ``extracted_text`` appears only as a descriptor (null flag, character
    length, byte length, SHA-256 of its UTF-8 bytes);
  * chunks: ``text`` has the same descriptor, and ``embedding`` appears as a descriptor
    (null flag, dimension count, SHA-256 of its ``vector`` text form).

NULL is always distinct from an empty value: a NULL column is JSON ``null``, an empty text
has a descriptor with ``chars == 0`` and the SHA-256 of the empty string.

The digests are computed by PostgreSQL itself (``sha256(bytea)`` is built in from version
11; no extension is used), so a raw secret or text never reaches this process. This module
is pure: it contains the SQL as plain strings and the logic that checks and assembles the
rows it is given, but it opens no connection. ``collect_snapshot`` receives a ``run_query``
function and the caller (``app.rag.snapshot_cli``) is responsible for supplying a
read-only, REPEATABLE READ connection; ``collect_snapshot`` verifies those two facts
before it reads any data.

Fingerprints:
  * a row fingerprint is not stored; rows are compared through their canonical JSON;
  * a section fingerprint is the SHA-256 of the canonical JSON of the table name, its
    column allow-list and its rows ordered by primary key;
  * the combined fingerprint is the SHA-256 of the canonical JSON of the schema version
    and the three section fingerprints.
The creation time and the database identity are kept outside every fingerprint, so the same
data gives the same fingerprint in a different container, database name or minute.

``compare_snapshots`` is generic. It first checks that the two snapshots were produced in
compatible environments (see ENVIRONMENT_EXACT_KEYS: the pgvector version, extra_float_digits
and server encoding must be equal and the PostgreSQL major version must match; the database
name and the session time zone never matter). The ``database`` block is identity metadata
and stays outside every fingerprint; it is only consulted for that compatibility check.
``check_evaluation_delta`` applies the E1.5 invariants (no pre-existing row removed or
changed, exactly the expected new users and documents, ownership of the additions, and the
identity of the new user by the SHA-256 of its email) as explicit, separate checks.

Timestamps: every timestamp column is ``timestamptz`` (migrations). ``collect_snapshot``
reads the column types from ``information_schema`` and refuses anything else, because
``col AT TIME ZONE 'UTC'`` is independent of the session ``TimeZone`` only for ``timestamptz``;
on a ``timestamp`` (without time zone) the same expression would depend on it.

Messages and issue locations are fixed text plus table names, integer ids and column names.
They never contain a value read from the database.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from app.rag.evaluation.parsing import canonical_json
from app.rag.evaluation.schema import EvaluationDataError, Issue

SNAPSHOT_SCHEMA_VERSION = "eval-snapshot-v1"

TABLES = ("users", "documents", "document_chunks")

# ---------- column allow-lists ----------

# The persisted columns this module knows. collect_snapshot refuses a database whose tables
# have any other column set, so a new column can never be silently left out of a snapshot.
PERSISTED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "users": ("id", "email", "hashed_password", "created_at"),
    "documents": (
        "id", "filename", "description", "stored_name", "content_type", "size_bytes",
        "processing_status", "processing_error", "extracted_text", "owner_id", "created_at",
        "source_version", "source_sha256", "indexing_status", "indexing_error",
        "indexing_attempt", "indexing_lease_expires_at", "indexed_source_version",
        "indexed_signature", "indexed_embedding_id", "indexed_at",
    ),
    "document_chunks": (
        "id", "document_id", "chunk_index", "start_offset", "end_offset", "page_start",
        "page_end", "text", "span_sha256", "embedding", "created_at",
    ),
}

_INT, _STR, _TS, _SHA, _TEXT, _VECTOR = "int", "str", "timestamp", "sha256", "text", "vector"

# Record fields in order: (name, kind, may_be_null). For _TEXT and _VECTOR the flag says
# whether the descriptor may describe a NULL column.
RECORD_FIELDS: Mapping[str, tuple[tuple[str, str, bool], ...]] = {
    "users": (
        ("id", _INT, False),
        ("email_sha256", _SHA, False),
        ("hashed_password_sha256", _SHA, False),
        ("created_at", _TS, False),
    ),
    "documents": (
        ("id", _INT, False),
        ("filename", _STR, False),
        ("description", _STR, True),
        ("stored_name", _STR, True),
        ("content_type", _STR, True),
        ("size_bytes", _INT, True),
        ("processing_status", _STR, True),
        ("processing_error", _STR, True),
        ("extracted_text", _TEXT, True),
        ("owner_id", _INT, False),
        ("created_at", _TS, False),
        ("source_version", _INT, False),
        ("source_sha256", _SHA, True),
        ("indexing_status", _STR, True),
        ("indexing_error", _STR, True),
        ("indexing_attempt", _INT, False),
        ("indexing_lease_expires_at", _TS, True),
        ("indexed_source_version", _INT, True),
        ("indexed_signature", _STR, True),
        ("indexed_embedding_id", _STR, True),
        ("indexed_at", _TS, True),
    ),
    "document_chunks": (
        ("id", _INT, False),
        ("document_id", _INT, False),
        ("chunk_index", _INT, False),
        ("start_offset", _INT, False),
        ("end_offset", _INT, False),
        ("page_start", _INT, True),
        ("page_end", _INT, True),
        ("text", _TEXT, False),
        ("span_sha256", _SHA, False),
        ("embedding", _VECTOR, False),
        ("created_at", _TS, False),
    ),
}

# The timestamp columns of each table, derived from the record fields above.
TIMESTAMP_COLUMNS: Mapping[str, tuple[str, ...]] = {
    table: tuple(name for name, kind, _ in fields if kind == _TS) for table, fields in RECORD_FIELDS.items()
}
TIMESTAMP_COLUMN_TYPE = "timestamp with time zone"

# Recorded properties of the database that decide whether two snapshots are comparable.
# ``embedding::text`` (hashed into every embedding descriptor) is produced by the pgvector
# output function, which depends on the extension version and on ``extra_float_digits``.
# The encoding fixes how text is converted to bytes before hashing. Informational only, and
# never compared: database_name (a restored database is named differently), time_zone
# (timestamps are rendered with AT TIME ZONE 'UTC' from timestamptz) and the PostgreSQL
# minor version.
ENVIRONMENT_EXACT_KEYS = ("pgvector_version", "extra_float_digits", "server_encoding")

_TEXT_PARTS = ("is_null", "chars", "bytes", "sha256")
_VECTOR_PARTS = ("is_null", "dims", "sha256")

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")

# ---------- SQL (plain strings; run by the caller inside one read-only transaction) ----------


def _ts(column: str) -> str:
    """UTC wall time of a ``timestamptz`` column; independent of the session TimeZone.

    Only valid for ``timestamp with time zone``: collect_snapshot refuses any other type.
    """
    return f"to_char({column} AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"')"


def _sha(expression: str) -> str:
    return f"encode(sha256(convert_to({expression}, 'UTF8')), 'hex')"


SESSION_QUERY = (
    "SELECT current_setting('transaction_isolation') AS isolation, "
    "current_setting('transaction_read_only') AS read_only"
)

IDENTITY_QUERY = (
    "SELECT current_database() AS database_name, "
    "current_setting('server_version') AS server_version, "
    "current_setting('server_encoding') AS server_encoding, "
    "current_setting('TimeZone') AS time_zone, "
    "current_setting('extra_float_digits') AS extra_float_digits, "
    "(SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector_version"
)

SCHEMA_QUERY = (
    "SELECT table_name, column_name, data_type FROM information_schema.columns "
    "WHERE table_schema = 'public' "
    "AND table_name IN ('users', 'documents', 'document_chunks') "
    "ORDER BY table_name, ordinal_position"
)

COUNT_QUERY = (
    "SELECT (SELECT count(*) FROM users) AS users, "
    "(SELECT count(*) FROM documents) AS documents, "
    "(SELECT count(*) FROM document_chunks) AS document_chunks"
)

TABLE_QUERIES: Mapping[str, str] = {
    "users": (
        "SELECT id, "
        f"{_sha('email')} AS email_sha256, "
        f"{_sha('hashed_password')} AS hashed_password_sha256, "
        f"{_ts('created_at')} AS created_at "
        "FROM users ORDER BY id"
    ),
    "documents": (
        "SELECT id, filename, description, stored_name, content_type, size_bytes, "
        "processing_status, processing_error, "
        "(extracted_text IS NULL) AS extracted_text_is_null, "
        "char_length(extracted_text) AS extracted_text_chars, "
        "octet_length(extracted_text) AS extracted_text_bytes, "
        f"{_sha('extracted_text')} AS extracted_text_sha256, "
        f"owner_id, {_ts('created_at')} AS created_at, source_version, source_sha256, "
        "indexing_status, indexing_error, indexing_attempt, "
        f"{_ts('indexing_lease_expires_at')} AS indexing_lease_expires_at, "
        "indexed_source_version, indexed_signature, indexed_embedding_id, "
        f"{_ts('indexed_at')} AS indexed_at "
        "FROM documents ORDER BY id"
    ),
    "document_chunks": (
        "SELECT id, document_id, chunk_index, start_offset, end_offset, page_start, page_end, "
        "(text IS NULL) AS text_is_null, "
        "char_length(text) AS text_chars, "
        "octet_length(text) AS text_bytes, "
        f"{_sha('text')} AS text_sha256, "
        "span_sha256, "
        "(embedding IS NULL) AS embedding_is_null, "
        "vector_dims(embedding) AS embedding_dims, "
        f"{_sha('embedding::text')} AS embedding_sha256, "
        f"{_ts('created_at')} AS created_at "
        "FROM document_chunks ORDER BY id"
    ),
}

# ---------- issue codes ----------

SNAPSHOT_STRUCTURE = "snapshot_structure"
SNAPSHOT_FINGERPRINT = "snapshot_fingerprint"
SNAPSHOT_ORDER = "snapshot_order"
SNAPSHOT_SESSION = "snapshot_session"
SNAPSHOT_SCHEMA_DRIFT = "snapshot_schema_drift"
SNAPSHOT_COUNT_MISMATCH = "snapshot_count_mismatch"
SNAPSHOT_PARSE = "snapshot_parse"
SNAPSHOT_TIMESTAMP_TYPE = "snapshot_timestamp_type"
COMPARE_ENVIRONMENT = "compare_environment"
COMPARE_INCOMPARABLE = "compare_incomparable"
DELTA_PREEXISTING_REMOVED = "delta_preexisting_removed"
DELTA_PREEXISTING_MODIFIED = "delta_preexisting_modified"
DELTA_USER_COUNT = "delta_user_count"
DELTA_DOCUMENT_COUNT = "delta_document_count"
DELTA_UNEXPECTED_USER = "delta_unexpected_user"
DELTA_DOCUMENT_OWNER = "delta_document_owner"
DELTA_CHUNK_PARENT = "delta_chunk_parent"
DELTA_IDENTITY_INVALID = "delta_identity_invalid"

_MESSAGES = {
    SNAPSHOT_STRUCTURE: "The snapshot structure is invalid",
    SNAPSHOT_FINGERPRINT: "A stored fingerprint does not match the content",
    SNAPSHOT_ORDER: "Rows are not in strictly increasing primary-key order",
    SNAPSHOT_SESSION: "The database session is not read-only REPEATABLE READ on a UTF8 server",
    SNAPSHOT_SCHEMA_DRIFT: "The table columns differ from the snapshot allow-list",
    SNAPSHOT_COUNT_MISMATCH: "The row count differs from the number of rows read",
    SNAPSHOT_PARSE: "The snapshot text is not valid JSON",
    SNAPSHOT_TIMESTAMP_TYPE: "A timestamp column is not of type timestamp with time zone",
    COMPARE_ENVIRONMENT: "The snapshots come from incompatible database environments",
    COMPARE_INCOMPARABLE: "The two snapshots have different schema versions or columns",
    DELTA_PREEXISTING_REMOVED: "A pre-existing row is missing from the later snapshot",
    DELTA_PREEXISTING_MODIFIED: "A pre-existing row changed",
    DELTA_USER_COUNT: "The number of new users is not the expected number",
    DELTA_DOCUMENT_COUNT: "The number of new documents is not the expected number",
    DELTA_UNEXPECTED_USER: "The new user is not the expected evaluation user",
    DELTA_DOCUMENT_OWNER: "A new document is not owned by a new user",
    DELTA_CHUNK_PARENT: "A new chunk does not belong to a new document",
    DELTA_IDENTITY_INVALID: "The evaluation user identity digest is missing or malformed",
}


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code])


# ---------- reference descriptors (the semantics the SQL mirrors) ----------


def sha256_hex(value: str) -> str:
    """Lowercase hex SHA-256 of the UTF-8 encoding of ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_descriptor(value: str | None) -> dict:
    """The descriptor of a text column. NULL and the empty string are different."""
    if value is None:
        return {"null": True, "chars": None, "bytes": None, "sha256": None}
    return {"null": False, "chars": len(value), "bytes": len(value.encode("utf-8")), "sha256": sha256_hex(value)}


def vector_descriptor(text_form: str | None, dims: int | None) -> dict:
    """The descriptor of an embedding, from its ``vector`` text form and dimension count."""
    if text_form is None:
        return {"null": True, "dims": None, "sha256": None}
    return {"null": False, "dims": dims, "sha256": sha256_hex(text_form)}


def format_timestamp(value: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SS.ffffffZ`` in UTC; a naive value is refused."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a timezone-aware datetime is required")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------- record validation ----------


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _is_ts(value: object) -> bool:
    return isinstance(value, str) and _TS_RE.fullmatch(value) is not None


def _check_value(kind: str, may_be_null: bool, value: object) -> bool:
    if value is None:
        return may_be_null and kind in (_INT, _STR, _TS, _SHA)
    if kind == _INT:
        return _is_int(value)
    if kind == _STR:
        return isinstance(value, str)
    if kind == _TS:
        return _is_ts(value)
    if kind == _SHA:
        return _is_sha(value)
    if kind in (_TEXT, _VECTOR):
        return _check_descriptor(kind, may_be_null, value)
    return False


def _check_descriptor(kind: str, may_be_null: bool, value: object) -> bool:
    names = ("null", "chars", "bytes", "sha256") if kind == _TEXT else ("null", "dims", "sha256")
    if not isinstance(value, dict) or set(value) != set(names):
        return False
    if type(value["null"]) is not bool:
        return False
    if value["null"]:
        return may_be_null and all(value[name] is None for name in names[1:])
    if not _is_sha(value["sha256"]):
        return False
    if kind == _TEXT:
        chars, size = value["chars"], value["bytes"]
        return _is_int(chars) and _is_int(size) and 0 <= chars <= size
    return _is_int(value["dims"]) and value["dims"] >= 0


def _record_issues(table: str, record: object, location: str) -> list[Issue]:
    fields = RECORD_FIELDS[table]
    if not isinstance(record, dict) or set(record) != {name for name, _, _ in fields}:
        return [_issue(SNAPSHOT_STRUCTURE, location)]
    return [
        _issue(SNAPSHOT_STRUCTURE, f"{location}.{name}")
        for name, kind, may_be_null in fields
        if not _check_value(kind, may_be_null, record[name])
    ]


# ---------- building ----------


def _column(row: Mapping[str, object], name: str, table: str, location: str) -> object:
    if name not in row:
        raise EvaluationDataError([_issue(SNAPSHOT_STRUCTURE, f"{location}.{name}")])
    return row[name]


def _record_from_row(table: str, row: Mapping[str, object], location: str) -> dict:
    """One record from one row returned by TABLE_QUERIES (exactly the expected columns)."""
    expected: list[str] = []
    record: dict = {}
    for name, kind, _ in RECORD_FIELDS[table]:
        if kind == _TEXT:
            parts = {part: f"{name}_{part}" for part in _TEXT_PARTS}
        elif kind == _VECTOR:
            parts = {part: f"{name}_{part}" for part in _VECTOR_PARTS}
        else:
            parts = None
        if parts is None:
            expected.append(name)
            record[name] = _column(row, name, table, location)
        else:
            expected.extend(parts.values())
            values = {part: _column(row, alias, table, location) for part, alias in parts.items()}
            descriptor = {"null": values["is_null"]}
            for part in parts:
                if part != "is_null":
                    descriptor[part] = values[part]
            record[name] = descriptor
    if set(row) != set(expected):
        raise EvaluationDataError([_issue(SNAPSHOT_STRUCTURE, location)])
    return record


def _section_fingerprint(table: str, columns: Sequence[str], rows: Sequence[Mapping]) -> str:
    payload = {"table": table, "columns": list(columns), "rows": list(rows)}
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _combined_fingerprint(section_fingerprints: Mapping[str, str]) -> str:
    payload = {"schema_version": SNAPSHOT_SCHEMA_VERSION, "tables": dict(section_fingerprints)}
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def build_snapshot(
    table_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    created_at: datetime,
    database: Mapping[str, str | None],
) -> dict:
    """Assemble and verify a snapshot from the rows returned by TABLE_QUERIES.

    Rows are re-sorted by primary key here; the order the database returned them in is not
    relied on. Raises EvaluationDataError if a row or the assembled snapshot is invalid.
    """
    if set(table_rows) != set(TABLES):
        raise EvaluationDataError([_issue(SNAPSHOT_STRUCTURE, "tables")])
    if not isinstance(database, Mapping) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str)) for key, value in database.items()
    ):
        raise EvaluationDataError([_issue(SNAPSHOT_STRUCTURE, "database")])

    tables: dict = {}
    for table in TABLES:
        records = [
            _record_from_row(table, row, f"{table}[{index}]") for index, row in enumerate(table_rows[table])
        ]
        for index, record in enumerate(records):
            issues = _record_issues(table, record, f"{table}[{index}]")
            if issues:
                raise EvaluationDataError(issues)
        records.sort(key=lambda record: record["id"])
        for earlier, later in zip(records, records[1:]):
            if earlier["id"] == later["id"]:
                raise EvaluationDataError([_issue(SNAPSHOT_ORDER, table)])
        columns = list(PERSISTED_COLUMNS[table])
        tables[table] = {
            "columns": columns,
            "row_count": len(records),
            "rows": records,
            "fingerprint": _section_fingerprint(table, columns, records),
        }

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at_utc": format_timestamp(created_at),
        "database": {key: database[key] for key in sorted(database)},
        "fingerprint": _combined_fingerprint({table: tables[table]["fingerprint"] for table in TABLES}),
        "tables": tables,
    }
    issues = verify_snapshot(snapshot)
    if issues:
        raise EvaluationDataError(issues)
    return snapshot


# ---------- verification and serialisation ----------


def verify_snapshot(snapshot: object) -> list[Issue]:
    """Every structural and fingerprint problem of ``snapshot``; empty means it is sound."""
    top = {"schema_version", "created_at_utc", "database", "fingerprint", "tables"}
    if not isinstance(snapshot, dict) or set(snapshot) != top:
        return [_issue(SNAPSHOT_STRUCTURE, "snapshot")]
    issues: list[Issue] = []
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        issues.append(_issue(SNAPSHOT_STRUCTURE, "schema_version"))
    if not _is_ts(snapshot["created_at_utc"]):
        issues.append(_issue(SNAPSHOT_STRUCTURE, "created_at_utc"))
    database = snapshot["database"]
    if not isinstance(database, dict) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str)) for key, value in database.items()
    ):
        issues.append(_issue(SNAPSHOT_STRUCTURE, "database"))
    tables = snapshot["tables"]
    if not isinstance(tables, dict) or set(tables) != set(TABLES):
        issues.append(_issue(SNAPSHOT_STRUCTURE, "tables"))
        return issues

    section_fingerprints: dict[str, str] = {}
    for table in TABLES:
        section = tables[table]
        where = f"tables.{table}"
        if not isinstance(section, dict) or set(section) != {"columns", "row_count", "rows", "fingerprint"}:
            issues.append(_issue(SNAPSHOT_STRUCTURE, where))
            continue
        if section["columns"] != list(PERSISTED_COLUMNS[table]):
            issues.append(_issue(SNAPSHOT_SCHEMA_DRIFT, f"{where}.columns"))
        rows = section["rows"]
        if not isinstance(rows, list):
            issues.append(_issue(SNAPSHOT_STRUCTURE, f"{where}.rows"))
            continue
        if not _is_int(section["row_count"]) or section["row_count"] != len(rows):
            issues.append(_issue(SNAPSHOT_COUNT_MISMATCH, f"{where}.row_count"))
        row_issues = [issue for index, row in enumerate(rows) for issue in _record_issues(table, row, f"{where}.rows[{index}]")]
        issues.extend(row_issues)
        if row_issues:
            continue
        ids = [row["id"] for row in rows]
        if any(earlier >= later for earlier, later in zip(ids, ids[1:])):
            issues.append(_issue(SNAPSHOT_ORDER, f"{where}.rows"))
            continue
        expected = _section_fingerprint(table, section["columns"], rows)
        if section["fingerprint"] != expected:
            issues.append(_issue(SNAPSHOT_FINGERPRINT, f"{where}.fingerprint"))
        section_fingerprints[table] = expected
    if len(section_fingerprints) == len(TABLES) and snapshot["fingerprint"] != _combined_fingerprint(section_fingerprints):
        issues.append(_issue(SNAPSHOT_FINGERPRINT, "fingerprint"))
    return issues


def snapshot_to_text(snapshot: Mapping) -> str:
    """The snapshot as canonical JSON plus a newline: the same snapshot is the same bytes."""
    return canonical_json(snapshot) + "\n"


def load_snapshot(text: str) -> dict:
    """Parse and verify snapshot text. Raises EvaluationDataError; nothing from the text is quoted."""
    if not isinstance(text, str):
        raise EvaluationDataError([_issue(SNAPSHOT_PARSE, "snapshot")])

    def reject_duplicates(pairs: list) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_name: str) -> None:
        raise ValueError("non-finite number")

    try:
        snapshot = json.loads(text, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        raise EvaluationDataError([_issue(SNAPSHOT_PARSE, "snapshot")]) from None
    issues = verify_snapshot(snapshot)
    if issues:
        raise EvaluationDataError(issues)
    return snapshot


# ---------- collecting (pure orchestration over a caller-supplied run_query) ----------

RunQuery = Callable[[str], Sequence[Mapping[str, object]]]


def _single(rows: Sequence[Mapping[str, object]], location: str) -> Mapping[str, object]:
    if len(rows) != 1:
        raise EvaluationDataError([_issue(SNAPSHOT_STRUCTURE, location)])
    return rows[0]


def collect_snapshot(run_query: RunQuery, *, created_at: datetime) -> dict:
    """Read all three tables through ``run_query`` and return the verified snapshot.

    ``run_query(sql)`` returns the result rows as mappings. The caller must have opened a
    read-only REPEATABLE READ transaction; this function checks that, and that the server
    encoding is UTF8 and the three tables have exactly the allow-listed columns, before it
    reads any data. Only SELECT statements from this module are ever passed to ``run_query``.
    """
    session = _single(run_query(SESSION_QUERY), "session")
    identity = _single(run_query(IDENTITY_QUERY), "identity")
    if (
        session.get("isolation") != "repeatable read"
        or session.get("read_only") != "on"
        or identity.get("server_encoding") != "UTF8"
    ):
        raise EvaluationDataError([_issue(SNAPSHOT_SESSION, "session")])

    actual: dict[str, list[str]] = {table: [] for table in TABLES}
    column_types: dict[tuple[str, str], object] = {}
    for row in run_query(SCHEMA_QUERY):
        table = row.get("table_name")
        if table in actual and isinstance(row.get("column_name"), str):
            actual[table].append(row["column_name"])
            column_types[(table, row["column_name"])] = row.get("data_type")
    drift = [
        _issue(SNAPSHOT_SCHEMA_DRIFT, table)
        for table in TABLES
        if sorted(actual[table]) != sorted(PERSISTED_COLUMNS[table])
    ]
    if drift:
        raise EvaluationDataError(drift)
    wrong_type = [
        _issue(SNAPSHOT_TIMESTAMP_TYPE, f"{table}.{column}")
        for table in TABLES
        for column in TIMESTAMP_COLUMNS[table]
        if column_types.get((table, column)) != TIMESTAMP_COLUMN_TYPE
    ]
    if wrong_type:
        raise EvaluationDataError(wrong_type)

    table_rows = {table: list(run_query(TABLE_QUERIES[table])) for table in TABLES}
    counts = _single(run_query(COUNT_QUERY), "counts")
    mismatched = [
        _issue(SNAPSHOT_COUNT_MISMATCH, table)
        for table in TABLES
        if counts.get(table) != len(table_rows[table])
    ]
    if mismatched:
        raise EvaluationDataError(mismatched)

    database = {key: (None if identity[key] is None else str(identity[key])) for key in sorted(identity)}
    return build_snapshot(table_rows, created_at=created_at, database=database)


# ---------- comparison ----------


@dataclass(frozen=True, slots=True)
class RecordChange:
    """A pre-existing row whose content differs, and the record fields that differ."""

    record_id: int
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableComparison:
    table: str
    count_before: int
    count_after: int
    fingerprint_before: str
    fingerprint_after: str
    added: tuple[int, ...]
    removed: tuple[int, ...]
    modified: tuple[RecordChange, ...]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.modified)


@dataclass(frozen=True, slots=True)
class SnapshotComparison:
    fingerprint_before: str
    fingerprint_after: str
    tables: tuple[TableComparison, ...]

    @property
    def identical(self) -> bool:
        """Same combined fingerprint and no difference found in any table."""
        return self.fingerprint_before == self.fingerprint_after and not any(t.changed for t in self.tables)

    @property
    def preexisting_unchanged(self) -> bool:
        """No row of the earlier snapshot was removed or modified in the later one."""
        return not any(t.removed or t.modified for t in self.tables)


_MAJOR_RE = re.compile(r"([0-9]{1,3})(?:[.\s]|$)")


def _pg_major(version: object) -> int | None:
    """The PostgreSQL major version of a ``server_version`` text such as ``16.15``; None if unreadable."""
    if not isinstance(version, str):
        return None
    match = _MAJOR_RE.match(version)
    return int(match.group(1)) if match else None


def environment_issues(before_database: object, after_database: object) -> list[Issue]:
    """Why two snapshots' recorded environments are not comparable; empty means they are.

    Fails closed: a missing, empty or unreadable value in either snapshot is incompatible.
    The issues name the property only, never a value.
    """
    if not isinstance(before_database, Mapping) or not isinstance(after_database, Mapping):
        return [_issue(COMPARE_ENVIRONMENT, "database")]
    issues: list[Issue] = []
    for key in ENVIRONMENT_EXACT_KEYS:
        old, new = before_database.get(key), after_database.get(key)
        if not (isinstance(old, str) and old and isinstance(new, str) and new) or old != new:
            issues.append(_issue(COMPARE_ENVIRONMENT, f"database.{key}"))
    old_major, new_major = _pg_major(before_database.get("server_version")), _pg_major(after_database.get("server_version"))
    if old_major is None or new_major is None or old_major != new_major:
        issues.append(_issue(COMPARE_ENVIRONMENT, "database.server_version"))
    return issues


def compare_snapshots(before: object, after: object) -> SnapshotComparison:
    """Compare two snapshots by primary key and row content.

    Both are verified first, and their recorded environments must be compatible
    (EvaluationDataError otherwise). Equal row counts prove nothing
    here: every id of ``before`` is looked up in ``after`` and its content compared.
    """
    issues = verify_snapshot(before) + verify_snapshot(after)
    if issues:
        raise EvaluationDataError(issues)
    assert isinstance(before, dict) and isinstance(after, dict)
    if before["schema_version"] != after["schema_version"]:
        raise EvaluationDataError([_issue(COMPARE_INCOMPARABLE, "schema_version")])
    environment = environment_issues(before["database"], after["database"])
    if environment:
        raise EvaluationDataError(environment)

    results: list[TableComparison] = []
    for table in TABLES:
        old_section, new_section = before["tables"][table], after["tables"][table]
        if old_section["columns"] != new_section["columns"]:
            raise EvaluationDataError([_issue(COMPARE_INCOMPARABLE, f"tables.{table}.columns")])
        old = {row["id"]: row for row in old_section["rows"]}
        new = {row["id"]: row for row in new_section["rows"]}
        field_names = [name for name, _, _ in RECORD_FIELDS[table]]
        modified = []
        for record_id in sorted(old.keys() & new.keys()):
            if canonical_json(old[record_id]) != canonical_json(new[record_id]):
                differing = tuple(
                    name for name in field_names
                    if canonical_json(old[record_id][name]) != canonical_json(new[record_id][name])
                )
                modified.append(RecordChange(record_id, differing))
        results.append(
            TableComparison(
                table=table,
                count_before=old_section["row_count"],
                count_after=new_section["row_count"],
                fingerprint_before=old_section["fingerprint"],
                fingerprint_after=new_section["fingerprint"],
                added=tuple(sorted(new.keys() - old.keys())),
                removed=tuple(sorted(old.keys() - new.keys())),
                modified=tuple(modified),
            )
        )
    return SnapshotComparison(before["fingerprint"], after["fingerprint"], tuple(results))


def check_evaluation_delta(
    before: object,
    after: object,
    *,
    evaluation_email_sha256: str,
    expected_new_users: int = 1,
    expected_new_documents: int = 6,
) -> list[Issue]:
    """The E1.5 invariants for an ingestion, as explicit checks. Empty means they all hold.

    1. No row of ``before`` is removed or modified, in any table.
    2. Exactly ``expected_new_users`` users and ``expected_new_documents`` documents were added.
    3. Every added document is owned by an added user.
    4. Every added chunk belongs to an added document (any number of chunks may be added).
    5. Every added user has the email digest ``evaluation_email_sha256`` (required: 64 lowercase
       hex characters; the digest is never reported, and no raw email is ever needed here).
    Nothing is assumed about the baseline: it is whatever ``before`` contains.
    """
    if not _is_sha(evaluation_email_sha256):
        raise EvaluationDataError([_issue(DELTA_IDENTITY_INVALID, "evaluation_email_sha256")])
    comparison = compare_snapshots(before, after)
    by_table = {table.table: table for table in comparison.tables}
    issues: list[Issue] = []

    for table in comparison.tables:
        for record_id in table.removed:
            issues.append(_issue(DELTA_PREEXISTING_REMOVED, f"{table.table}[{record_id}]"))
        for change in table.modified:
            for name in change.fields:
                issues.append(_issue(DELTA_PREEXISTING_MODIFIED, f"{table.table}[{change.record_id}].{name}"))

    new_users, new_documents, new_chunks = (
        by_table["users"].added, by_table["documents"].added, by_table["document_chunks"].added
    )
    if len(new_users) != expected_new_users:
        issues.append(_issue(DELTA_USER_COUNT, "users"))
    if len(new_documents) != expected_new_documents:
        issues.append(_issue(DELTA_DOCUMENT_COUNT, "documents"))

    after_users = {row["id"]: row for row in after["tables"]["users"]["rows"]}
    after_documents = {row["id"]: row for row in after["tables"]["documents"]["rows"]}
    after_chunks = {row["id"]: row for row in after["tables"]["document_chunks"]["rows"]}
    for document_id in new_documents:
        if after_documents[document_id]["owner_id"] not in new_users:
            issues.append(_issue(DELTA_DOCUMENT_OWNER, f"documents[{document_id}]"))
    for chunk_id in new_chunks:
        if after_chunks[chunk_id]["document_id"] not in new_documents:
            issues.append(_issue(DELTA_CHUNK_PARENT, f"document_chunks[{chunk_id}]"))
    for user_id in new_users:
        if after_users[user_id]["email_sha256"] != evaluation_email_sha256:
            issues.append(_issue(DELTA_UNEXPECTED_USER, f"users[{user_id}]"))
    return issues


# ---------- reporting ----------

_LIST_LIMIT = 50


def _ids(values: Sequence[int]) -> str:
    shown = ", ".join(str(value) for value in values[:_LIST_LIMIT])
    extra = len(values) - _LIST_LIMIT
    return shown + (f" ... (+{extra} more)" if extra > 0 else "")


def format_report(comparison: SnapshotComparison, issues: Sequence[Issue] = ()) -> str:
    """A deterministic text report. It holds ids, counts, field names and fingerprints only."""
    lines = [
        f"before fingerprint: {comparison.fingerprint_before}",
        f"after fingerprint:  {comparison.fingerprint_after}",
        f"identical: {'yes' if comparison.identical else 'no'}",
        f"pre-existing rows unchanged: {'yes' if comparison.preexisting_unchanged else 'no'}",
    ]
    for table in comparison.tables:
        lines.append(f"{table.table}: {table.count_before} -> {table.count_after} rows"
                     f"{'' if table.fingerprint_before == table.fingerprint_after else ' (section fingerprint changed)'}")
        if table.added:
            lines.append(f"  added ({len(table.added)}): {_ids(table.added)}")
        if table.removed:
            lines.append(f"  removed ({len(table.removed)}): {_ids(table.removed)}")
        for change in table.modified[:_LIST_LIMIT]:
            lines.append(f"  modified id {change.record_id}: {', '.join(change.fields)}")
        if len(table.modified) > _LIST_LIMIT:
            lines.append(f"  ... (+{len(table.modified) - _LIST_LIMIT} more modified)")
    if issues:
        lines.append(f"findings ({len(issues)}):")
        lines.extend(f"  {issue.code} at {issue.location}: {issue.message}" for issue in issues)
    else:
        lines.append("findings: none")
    return "\n".join(lines) + "\n"
