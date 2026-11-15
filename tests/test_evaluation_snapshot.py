"""Tests for app.rag.evaluation.snapshot (E1.5.3). Pure: no database, no files.

A  allow-lists, descriptors and text hashing
B  building: determinism, ordering, timestamps, NULL versus empty, empty / single / many rows
C  embedding fingerprint
D  verification, serialisation and loading
E  collect_snapshot over a fake run_query (session guards, schema drift, counts, SELECT only)
F  comparison
G  evaluation delta invariants, including the documented baseline exceptions
H  report
I  import isolation
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.rag.evaluation import snapshot as sn
from app.rag.evaluation.schema import EvaluationDataError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 12, 13, 0, 0, 123456, tzinfo=timezone.utc)
STAMP = "2026-11-01T08:00:00.000001Z"
DB = {"database_name": "snapshot_test", "server_version": "16.15", "server_encoding": "UTF8",
      "time_zone": "UTC", "extra_float_digits": "1", "pgvector_version": "0.8.7"}
CANARY = "CANARY-secret-text-7731"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------- row builders (the shape TABLE_QUERIES returns) ----------

def user_row(user_id: int, email: str | None = None, password: str = "pw") -> dict:
    return {
        "id": user_id,
        "email_sha256": sha(email if email is not None else f"user{user_id}@example.com"),
        "hashed_password_sha256": sha(password),
        "created_at": STAMP,
    }


def doc_row(doc_id: int, owner_id: int = 1, **override) -> dict:
    row = {
        "id": doc_id, "filename": f"doc{doc_id}.txt", "description": None,
        "stored_name": f"{doc_id:032x}.txt", "content_type": "text/plain", "size_bytes": 10,
        "processing_status": "completed", "processing_error": None,
        "extracted_text_is_null": False, "extracted_text_chars": 5, "extracted_text_bytes": 5,
        "extracted_text_sha256": sha("hello"),
        "owner_id": owner_id, "created_at": STAMP, "source_version": 1, "source_sha256": sha("hello"),
        "indexing_status": "indexed", "indexing_error": None, "indexing_attempt": 1,
        "indexing_lease_expires_at": None, "indexed_source_version": 1, "indexed_signature": "sig",
        "indexed_embedding_id": "emb", "indexed_at": STAMP,
    }
    row.update(override)
    return row


def chunk_row(chunk_id: int, document_id: int, index: int = 0, text: str = "hello", vector: str = "[1,2,3]", **override) -> dict:
    row = {
        "id": chunk_id, "document_id": document_id, "chunk_index": index,
        "start_offset": 0, "end_offset": len(text), "page_start": None, "page_end": None,
        "text_is_null": False, "text_chars": len(text), "text_bytes": len(text.encode("utf-8")),
        "text_sha256": sha(text), "span_sha256": sha(text),
        "embedding_is_null": False, "embedding_dims": 3, "embedding_sha256": sha(vector),
        "created_at": STAMP,
    }
    row.update(override)
    return row


def rows(users=(), documents=(), chunks=()) -> dict:
    return {"users": list(users), "documents": list(documents), "document_chunks": list(chunks)}


def build(table_rows: dict, created_at: datetime = T0, database=DB) -> dict:
    return sn.build_snapshot(table_rows, created_at=created_at, database=database)


def baseline_rows() -> dict:
    return rows(
        users=[user_row(1), user_row(2)],
        documents=[doc_row(10, 1), doc_row(11, 2)],
        chunks=[chunk_row(100, 10, 0), chunk_row(101, 10, 1, text="world")],
    )


def codes(error: EvaluationDataError) -> list[str]:
    return [issue.code for issue in error.issues]


# ---------- A. allow-lists, descriptors, text hashing ----------

def test_the_allow_lists_have_the_expected_sizes_and_unique_names():
    assert {t: len(c) for t, c in sn.PERSISTED_COLUMNS.items()} == {"users": 4, "documents": 21, "document_chunks": 11}
    for columns in sn.PERSISTED_COLUMNS.values():
        assert len(set(columns)) == len(columns)
    assert sn.TABLES == ("users", "documents", "document_chunks")


def test_the_allow_lists_match_the_sqlalchemy_models():
    from app.models.chunk import DocumentChunk
    from app.models.document import Document
    from app.models.user import User

    for table, model in (("users", User), ("documents", Document), ("document_chunks", DocumentChunk)):
        assert sorted(sn.PERSISTED_COLUMNS[table]) == sorted(model.__table__.columns.keys())


def test_every_record_field_traces_to_a_persisted_column():
    for table, fields in sn.RECORD_FIELDS.items():
        for name, kind, _ in fields:
            if name in ("email_sha256", "hashed_password_sha256"):
                assert name.removesuffix("_sha256") in sn.PERSISTED_COLUMNS[table]
            else:
                assert name in sn.PERSISTED_COLUMNS[table]
        assert len(fields) == len(sn.PERSISTED_COLUMNS[table])


def test_sha256_hex_is_the_utf8_digest():
    assert sn.sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sn.sha256_hex("caf\u00e9") == hashlib.sha256("caf\u00e9".encode("utf-8")).hexdigest()


def test_text_descriptor_distinguishes_null_empty_and_text():
    null, empty, text = sn.text_descriptor(None), sn.text_descriptor(""), sn.text_descriptor("h\u00e9llo \U0001F600")
    assert null == {"null": True, "chars": None, "bytes": None, "sha256": None}
    assert empty == {"null": False, "chars": 0, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    assert text["chars"] == 7 and text["bytes"] == 11 and text["sha256"] == sha("h\u00e9llo \U0001F600")
    assert len({json.dumps(x, sort_keys=True) for x in (null, empty, text)}) == 3


def test_vector_descriptor_is_stable_and_distinguishes_null():
    assert sn.vector_descriptor("[1,2]", 2) == sn.vector_descriptor("[1,2]", 2)
    assert sn.vector_descriptor("[1,2]", 2)["sha256"] == sha("[1,2]")
    assert sn.vector_descriptor("[1,2.0000001]", 2) != sn.vector_descriptor("[1,2]", 2)
    assert sn.vector_descriptor(None, None) == {"null": True, "dims": None, "sha256": None}


def test_format_timestamp_is_utc_with_microseconds_and_refuses_naive_values():
    assert sn.format_timestamp(T0) == "2026-11-12T13:00:00.123456Z"
    ist = timezone(timedelta(hours=5, minutes=30))
    assert sn.format_timestamp(T0.astimezone(ist)) == "2026-11-12T13:00:00.123456Z"
    with pytest.raises(ValueError):
        sn.format_timestamp(datetime(2026, 1, 1))


def test_the_queries_are_select_only_and_name_only_allow_listed_tables():
    statements = [sn.SESSION_QUERY, sn.IDENTITY_QUERY, sn.SCHEMA_QUERY, sn.COUNT_QUERY, *sn.TABLE_QUERIES.values()]
    for sql in statements:
        assert sql.lstrip().upper().startswith("SELECT")
        for word in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "GRANT ", "COPY "):
            assert word not in sql.upper()
        assert ";" not in sql
    for table, sql in sn.TABLE_QUERIES.items():
        assert f"FROM {table} ORDER BY id" in sql
    assert "pgcrypto" not in " ".join(statements).lower() and "digest(" not in " ".join(statements).lower()
    # raw secret and text columns are only ever hashed or measured, never selected as values
    users_sql = sn.TABLE_QUERIES["users"]
    assert "email_sha256" in users_sql and "hashed_password_sha256" in users_sql
    assert " email," not in users_sql and " hashed_password," not in users_sql


# ---------- B. building ----------

def test_the_snapshot_has_the_documented_shape():
    snap = build(baseline_rows())
    assert set(snap) == {"schema_version", "created_at_utc", "database", "fingerprint", "tables"}
    assert snap["schema_version"] == sn.SNAPSHOT_SCHEMA_VERSION == "eval-snapshot-v1"
    assert snap["created_at_utc"] == "2026-11-12T13:00:00.123456Z"
    assert snap["database"] == DB
    for table in sn.TABLES:
        section = snap["tables"][table]
        assert set(section) == {"columns", "row_count", "rows", "fingerprint"}
        assert section["columns"] == list(sn.PERSISTED_COLUMNS[table])
    assert [snap["tables"][t]["row_count"] for t in sn.TABLES] == [2, 2, 2]


def test_rows_are_stored_in_primary_key_order_whatever_order_they_arrive_in():
    ordered = build(baseline_rows())
    shuffled = baseline_rows()
    for table in shuffled:
        shuffled[table].reverse()
    other = build(shuffled)
    assert other["fingerprint"] == ordered["fingerprint"]
    assert [r["id"] for r in other["tables"]["document_chunks"]["rows"]] == [100, 101]
    assert sn.snapshot_to_text(other) == sn.snapshot_to_text(ordered)


def test_the_content_fingerprint_ignores_the_creation_time_and_the_database_identity():
    first = build(baseline_rows(), created_at=T0)
    second = build(baseline_rows(), created_at=T0 + timedelta(days=3),
                   database={**DB, "database_name": "restored", "server_version": "16.99", "pgvector_version": "9.9.9"})
    assert first["created_at_utc"] != second["created_at_utc"]
    assert first["database"] != second["database"]
    assert first["fingerprint"] == second["fingerprint"]
    assert all(first["tables"][t]["fingerprint"] == second["tables"][t]["fingerprint"] for t in sn.TABLES)


def test_serialisation_is_deterministic_and_the_same_data_is_the_same_bytes():
    first, second = build(baseline_rows()), build(baseline_rows())
    text = sn.snapshot_to_text(first)
    assert text == sn.snapshot_to_text(second) and text.endswith("\n") and text.isascii()
    assert json.loads(text) == first


def test_any_content_change_changes_the_section_and_the_combined_fingerprint():
    base = build(baseline_rows())
    changed = baseline_rows()
    changed["documents"][0]["filename"] = "renamed.txt"
    other = build(changed)
    assert other["tables"]["documents"]["fingerprint"] != base["tables"]["documents"]["fingerprint"]
    assert other["tables"]["users"]["fingerprint"] == base["tables"]["users"]["fingerprint"]
    assert other["fingerprint"] != base["fingerprint"]


def test_null_and_empty_string_are_different_for_text_columns_and_fingerprints():
    with_null = rows(users=[user_row(1)], documents=[doc_row(10, description=None)])
    with_empty = rows(users=[user_row(1)], documents=[doc_row(10, description="")])
    a, b = build(with_null), build(with_empty)
    assert a["tables"]["documents"]["rows"][0]["description"] is None
    assert b["tables"]["documents"]["rows"][0]["description"] == ""
    assert a["fingerprint"] != b["fingerprint"]


def test_null_extracted_text_is_different_from_empty_and_from_text():
    def snap(**cols):
        return build(rows(users=[user_row(1)], documents=[doc_row(10, **cols)]))
    null = snap(extracted_text_is_null=True, extracted_text_chars=None, extracted_text_bytes=None, extracted_text_sha256=None)
    empty = snap(extracted_text_is_null=False, extracted_text_chars=0, extracted_text_bytes=0, extracted_text_sha256=sha(""))
    text = snap()
    fingerprints = {s["fingerprint"] for s in (null, empty, text)}
    assert len(fingerprints) == 3
    document = lambda s: s["tables"]["documents"]["rows"][0]["extracted_text"]
    assert document(null) == sn.text_descriptor(None)
    assert document(empty) == sn.text_descriptor("")
    assert document(text) == sn.text_descriptor("hello")


def test_empty_tables_build_and_have_a_stable_fingerprint():
    snap = build(rows())
    assert [snap["tables"][t]["row_count"] for t in sn.TABLES] == [0, 0, 0]
    assert all(snap["tables"][t]["rows"] == [] for t in sn.TABLES)
    assert snap["fingerprint"] == build(rows(), created_at=T0 + timedelta(hours=1))["fingerprint"]
    assert sn.verify_snapshot(snap) == []
    # an empty snapshot differs from a populated one
    assert snap["fingerprint"] != build(baseline_rows())["fingerprint"]


def test_single_row_tables_and_many_row_tables_build():
    one = build(rows(users=[user_row(1)], documents=[doc_row(10)], chunks=[chunk_row(100, 10)]))
    assert [one["tables"][t]["row_count"] for t in sn.TABLES] == [1, 1, 1]
    many = build(rows(users=[user_row(i) for i in range(1, 51)],
                      documents=[doc_row(i, owner_id=1 + i % 50) for i in range(1, 201)],
                      chunks=[chunk_row(i, 1 + i % 200, index=i) for i in range(1, 501)]))
    assert [many["tables"][t]["row_count"] for t in sn.TABLES] == [50, 200, 500]
    assert sn.verify_snapshot(many) == []


def test_extra_missing_or_wrongly_typed_columns_are_refused():
    extra = baseline_rows(); extra["users"][0]["unexpected"] = 1
    with pytest.raises(EvaluationDataError):
        build(extra)
    missing = baseline_rows(); del missing["documents"][0]["filename"]
    with pytest.raises(EvaluationDataError):
        build(missing)
    missing_part = baseline_rows(); del missing_part["document_chunks"][0]["text_sha256"]
    with pytest.raises(EvaluationDataError):
        build(missing_part)
    for bad in ({"id": True}, {"id": "10"}, {"size_bytes": "10"}, {"created_at": "2026-11-01"}, {"source_sha256": "XYZ"},
                {"extracted_text_is_null": 1}, {"owner_id": None}, {"filename": None}):
        broken = baseline_rows(); broken["documents"][0].update(bad)
        with pytest.raises(EvaluationDataError):
            build(broken)


def test_inconsistent_descriptors_are_refused():
    for bad in ({"extracted_text_is_null": True},                                      # null flag with data
                {"extracted_text_chars": 9},                                           # more chars than bytes
                {"extracted_text_is_null": False, "extracted_text_sha256": None}):     # data flag without a digest
        broken = baseline_rows(); broken["documents"][0].update(bad)
        with pytest.raises(EvaluationDataError):
            build(broken)
    null_chunk_text = baseline_rows(); null_chunk_text["document_chunks"][0].update(
        text_is_null=True, text_chars=None, text_bytes=None, text_sha256=None)
    with pytest.raises(EvaluationDataError):          # chunk text is NOT NULL
        build(null_chunk_text)


def test_duplicate_primary_keys_and_missing_tables_are_refused():
    duplicate = baseline_rows(); duplicate["users"].append(user_row(1))
    with pytest.raises(EvaluationDataError) as error:
        build(duplicate)
    assert codes(error.value) == [sn.SNAPSHOT_ORDER]
    with pytest.raises(EvaluationDataError):
        build({"users": [], "documents": []})
    with pytest.raises(ValueError):
        build(baseline_rows(), created_at=datetime(2026, 1, 1))
    with pytest.raises(EvaluationDataError):
        build(baseline_rows(), database={"name": 5})


def test_a_snapshot_never_contains_a_secret_or_a_text():
    raw = baseline_rows()
    raw["users"][0] = user_row(1, email=f"{CANARY}@example.com", password=CANARY)
    raw["documents"][0]["filename"] = "file.txt"
    text = sn.snapshot_to_text(build(raw))
    assert CANARY not in text and "@example.com" not in text
    assert sha(f"{CANARY}@example.com") in text and sha(CANARY) in text


# ---------- C. embedding fingerprint ----------

def test_the_embedding_fingerprint_is_stable_and_changes_with_the_vector():
    def chunk_snapshot(vector: str):
        return build(rows(users=[user_row(1)], documents=[doc_row(10)], chunks=[chunk_row(100, 10, vector=vector)]))
    a1, a2, b = chunk_snapshot("[1,2,3]"), chunk_snapshot("[1,2,3]"), chunk_snapshot("[1,2,3.0000002]")
    embedding = lambda s: s["tables"]["document_chunks"]["rows"][0]["embedding"]
    assert embedding(a1) == embedding(a2)
    assert a1["fingerprint"] == a2["fingerprint"]
    assert embedding(a1)["sha256"] != embedding(b)["sha256"]
    assert a1["fingerprint"] != b["fingerprint"]
    assert embedding(a1)["dims"] == 3 and embedding(a1)["null"] is False
    assert "[1,2,3]" not in sn.snapshot_to_text(a1)


# ---------- D. verification, serialisation, loading ----------

def test_a_built_snapshot_verifies_and_round_trips_through_text():
    snap = build(baseline_rows())
    assert sn.verify_snapshot(snap) == []
    loaded = sn.load_snapshot(sn.snapshot_to_text(snap))
    assert loaded == snap and loaded["fingerprint"] == snap["fingerprint"]


def test_verification_detects_tampering_with_content_fingerprints_order_and_counts():
    snap = build(baseline_rows())

    edited = copy.deepcopy(snap); edited["tables"]["documents"]["rows"][0]["filename"] = "tampered"
    assert any(i.code == sn.SNAPSHOT_FINGERPRINT for i in sn.verify_snapshot(edited))

    wrong_section = copy.deepcopy(snap); wrong_section["tables"]["users"]["fingerprint"] = "0" * 64
    assert any(i.code == sn.SNAPSHOT_FINGERPRINT for i in sn.verify_snapshot(wrong_section))

    wrong_combined = copy.deepcopy(snap); wrong_combined["fingerprint"] = "f" * 64
    assert [i.code for i in sn.verify_snapshot(wrong_combined)] == [sn.SNAPSHOT_FINGERPRINT]

    reordered = copy.deepcopy(snap); reordered["tables"]["users"]["rows"].reverse()
    assert any(i.code == sn.SNAPSHOT_ORDER for i in sn.verify_snapshot(reordered))

    miscounted = copy.deepcopy(snap); miscounted["tables"]["users"]["row_count"] = 5
    assert any(i.code == sn.SNAPSHOT_COUNT_MISMATCH for i in sn.verify_snapshot(miscounted))

    drifted = copy.deepcopy(snap); drifted["tables"]["users"]["columns"].append("extra")
    assert any(i.code == sn.SNAPSHOT_SCHEMA_DRIFT for i in sn.verify_snapshot(drifted))

    version = copy.deepcopy(snap); version["schema_version"] = "other"
    assert sn.verify_snapshot(version)

    unknown_key = copy.deepcopy(snap); unknown_key["extra"] = 1
    assert sn.verify_snapshot(unknown_key)


def test_loading_rejects_bad_json_duplicate_keys_non_finite_numbers_and_non_text():
    good = sn.snapshot_to_text(build(rows()))
    for bad in ("", "not json", "[]", "{", good[:-5], '{"a":1,"a":2}', '{"a":NaN}'):
        with pytest.raises(EvaluationDataError):
            sn.load_snapshot(bad)
    duplicate = good.replace('"fingerprint":', '"fingerprint":"x","fingerprint":', 1)
    with pytest.raises(EvaluationDataError):
        sn.load_snapshot(duplicate)
    with pytest.raises(EvaluationDataError):
        sn.load_snapshot(b"{}")  # type: ignore[arg-type]


def test_error_messages_never_quote_the_input():
    with pytest.raises(EvaluationDataError) as error:
        sn.load_snapshot(CANARY)
    shown = str(error.value) + " ".join(f"{i.code}{i.location}{i.message}" for i in error.value.issues)
    assert CANARY not in shown


# ---------- E. collect_snapshot over a fake run_query ----------

class FakeDatabase:
    """Answers the module's own queries with canned rows and records every statement."""

    def __init__(self, table_rows=None, *, isolation="repeatable read", read_only="on", encoding="UTF8",
                 columns=None, counts=None, identity_extra=None, column_types=None):
        self.table_rows = table_rows if table_rows is not None else baseline_rows()
        self.statements: list[str] = []
        self.session = {"isolation": isolation, "read_only": read_only}
        self.identity = {"database_name": "fake", "server_version": "16.15", "server_encoding": encoding,
                         "time_zone": "UTC", "extra_float_digits": "1", "pgvector_version": "0.8.7"}
        self.identity.update(identity_extra or {})
        self.columns = columns if columns is not None else {t: list(c) for t, c in sn.PERSISTED_COLUMNS.items()}
        self.counts = counts
        # information_schema data_type: timestamptz for the timestamp columns, text for the rest
        self.column_types = {
            (t, c): "timestamp with time zone" if c in sn.TIMESTAMP_COLUMNS[t] else "text"
            for t, cols in sn.PERSISTED_COLUMNS.items() for c in cols
        }
        self.column_types.update(column_types or {})

    def __call__(self, sql: str):
        self.statements.append(sql)
        if sql == sn.SESSION_QUERY:
            return [self.session]
        if sql == sn.IDENTITY_QUERY:
            return [self.identity]
        if sql == sn.SCHEMA_QUERY:
            return [{"table_name": t, "column_name": c, "data_type": self.column_types.get((t, c))}
                    for t, cols in self.columns.items() for c in cols]
        if sql == sn.COUNT_QUERY:
            return [self.counts or {t: len(r) for t, r in self.table_rows.items()}]
        for table, query in sn.TABLE_QUERIES.items():
            if sql == query:
                return self.table_rows[table]
        raise AssertionError("unexpected statement")


def test_collect_snapshot_builds_the_same_snapshot_as_build_snapshot():
    fake = FakeDatabase()
    collected = sn.collect_snapshot(fake, created_at=T0)
    assert collected["fingerprint"] == build(baseline_rows())["fingerprint"]
    assert collected["database"]["database_name"] == "fake" and collected["database"]["pgvector_version"] == "0.8.7"


def test_collect_snapshot_checks_the_session_before_reading_any_data_and_issues_only_selects():
    fake = FakeDatabase()
    sn.collect_snapshot(fake, created_at=T0)
    assert fake.statements[0] == sn.SESSION_QUERY
    assert fake.statements.index(sn.SESSION_QUERY) < fake.statements.index(sn.TABLE_QUERIES["users"])
    assert all(s.lstrip().upper().startswith("SELECT") for s in fake.statements)
    assert set(fake.statements) <= {sn.SESSION_QUERY, sn.IDENTITY_QUERY, sn.SCHEMA_QUERY, sn.COUNT_QUERY, *sn.TABLE_QUERIES.values()}


@pytest.mark.parametrize("kwargs", [
    {"isolation": "read committed"}, {"isolation": "serializable"}, {"read_only": "off"}, {"encoding": "LATIN1"},
])
def test_collect_snapshot_refuses_a_session_that_is_not_read_only_repeatable_read_utf8(kwargs):
    fake = FakeDatabase(**kwargs)
    with pytest.raises(EvaluationDataError) as error:
        sn.collect_snapshot(fake, created_at=T0)
    assert codes(error.value) == [sn.SNAPSHOT_SESSION]
    assert not any(s in fake.statements for s in sn.TABLE_QUERIES.values())


def test_collect_snapshot_refuses_schema_drift_before_reading_data():
    extra = {t: list(c) for t, c in sn.PERSISTED_COLUMNS.items()}
    extra["documents"].append("new_column")
    missing = {t: list(c) for t, c in sn.PERSISTED_COLUMNS.items()}
    missing["users"].remove("email")
    for columns in (extra, missing):
        fake = FakeDatabase(columns=columns)
        with pytest.raises(EvaluationDataError) as error:
            sn.collect_snapshot(fake, created_at=T0)
        assert codes(error.value) == [sn.SNAPSHOT_SCHEMA_DRIFT]
        assert not any(s in fake.statements for s in sn.TABLE_QUERIES.values())


def test_collect_snapshot_refuses_a_count_that_disagrees_with_the_rows_read():
    fake = FakeDatabase(counts={"users": 2, "documents": 3, "document_chunks": 2})
    with pytest.raises(EvaluationDataError) as error:
        sn.collect_snapshot(fake, created_at=T0)
    assert codes(error.value) == [sn.SNAPSHOT_COUNT_MISMATCH]


def test_collect_snapshot_refuses_malformed_session_or_identity_answers():
    class Broken(FakeDatabase):
        def __call__(self, sql):
            return [] if sql == sn.SESSION_QUERY else super().__call__(sql)
    with pytest.raises(EvaluationDataError):
        sn.collect_snapshot(Broken(), created_at=T0)


# ---------- F. comparison ----------

def test_identical_snapshots_compare_as_identical_even_with_different_timestamps():
    a, b = build(baseline_rows(), created_at=T0), build(baseline_rows(), created_at=T0 + timedelta(days=1))
    result = sn.compare_snapshots(a, b)
    assert result.identical and result.preexisting_unchanged
    assert all(not t.changed and t.count_before == t.count_after for t in result.tables)
    assert result.fingerprint_before == result.fingerprint_after


def test_an_added_user_document_and_chunk_are_reported_as_added():
    later = baseline_rows()
    later["users"].append(user_row(3))
    later["documents"].append(doc_row(12, owner_id=3))
    later["document_chunks"].append(chunk_row(102, 12))
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    by = {t.table: t for t in result.tables}
    assert by["users"].added == (3,) and by["documents"].added == (12,) and by["document_chunks"].added == (102,)
    assert all(not by[t].removed and not by[t].modified for t in by)
    assert result.preexisting_unchanged and not result.identical
    assert (by["users"].count_before, by["users"].count_after) == (2, 3)
    assert all(by[t].fingerprint_before != by[t].fingerprint_after for t in by)
    assert result.fingerprint_before != result.fingerprint_after


def test_a_removed_record_is_reported_and_breaks_preexisting_unchanged():
    later = baseline_rows()
    del later["documents"][1]
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    by = {t.table: t for t in result.tables}
    assert by["documents"].removed == (11,) and not by["documents"].added
    assert not result.preexisting_unchanged


def test_a_modified_record_reports_exactly_the_changed_fields():
    later = baseline_rows()
    later["documents"][0].update(indexing_status="failed", indexing_error="boom", indexed_at=None)
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    change = {t.table: t for t in result.tables}["documents"].modified
    assert len(change) == 1 and change[0].record_id == 10
    assert change[0].fields == ("indexing_status", "indexing_error", "indexed_at")   # record-field order


def test_same_row_count_but_different_identities_is_not_identical():
    later = baseline_rows()
    later["documents"][1] = doc_row(99, owner_id=2)       # id 11 replaced by id 99: counts equal
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    by = {t.table: t for t in result.tables}
    assert by["documents"].count_before == by["documents"].count_after == 2
    assert by["documents"].added == (99,) and by["documents"].removed == (11,)
    assert not result.identical and not result.preexisting_unchanged


def test_same_ids_with_modified_content_is_not_identical():
    later = baseline_rows()
    later["document_chunks"][0].update(text_sha256=sha("different"), text_chars=9, text_bytes=9)
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    change = {t.table: t for t in result.tables}["document_chunks"].modified
    assert [(c.record_id, c.fields) for c in change] == [(100, ("text",))]
    assert not result.identical


def test_changed_document_ownership_is_a_modification_of_owner_id():
    later = baseline_rows()
    later["documents"][0]["owner_id"] = 2
    result = sn.compare_snapshots(build(baseline_rows()), build(later))
    change = {t.table: t for t in result.tables}["documents"].modified
    assert [(c.record_id, c.fields) for c in change] == [(10, ("owner_id",))]


def test_a_changed_embedding_and_a_changed_password_digest_are_modifications():
    later = baseline_rows()
    later["document_chunks"][1]["embedding_sha256"] = sha("[9,9,9]")
    later["users"][0]["hashed_password_sha256"] = sha("new")
    by = {t.table: t for t in sn.compare_snapshots(build(baseline_rows()), build(later)).tables}
    assert [(c.record_id, c.fields) for c in by["document_chunks"].modified] == [(101, ("embedding",))]
    assert [(c.record_id, c.fields) for c in by["users"].modified] == [(1, ("hashed_password_sha256",))]


def test_comparison_refuses_an_invalid_snapshot_and_incomparable_schemas():
    good = build(baseline_rows())
    tampered = copy.deepcopy(good); tampered["tables"]["users"]["rows"][0]["created_at"] = "2020-01-01T00:00:00.000000Z"
    with pytest.raises(EvaluationDataError):
        sn.compare_snapshots(good, tampered)
    with pytest.raises(EvaluationDataError):
        sn.compare_snapshots(good, {"not": "a snapshot"})


# ---------- G. evaluation delta ----------

EVAL_EMAIL = "e1-eval@example.com"


def evaluated(base: dict, *, new_users=1, new_docs=6, chunks_per_doc=2, owner=None) -> dict:
    """``base`` plus an evaluation ingestion: new users, new documents and new chunks."""
    later = copy.deepcopy(base)
    next_user = max((u["id"] for u in later["users"]), default=0) + 1
    users = list(range(next_user, next_user + new_users))
    for user_id in users:
        later["users"].append(user_row(user_id, email=EVAL_EMAIL))
    next_doc = max((d["id"] for d in later["documents"]), default=0) + 1
    next_chunk = max((c["id"] for c in later["document_chunks"]), default=0) + 1
    for offset in range(new_docs):
        doc_id = next_doc + offset
        later["documents"].append(doc_row(doc_id, owner_id=owner if owner is not None else users[0]))
        for index in range(chunks_per_doc):
            later["document_chunks"].append(chunk_row(next_chunk, doc_id, index=index))
            next_chunk += 1
    return later


def delta(before_rows: dict, after_rows: dict, **kwargs):
    kwargs.setdefault("evaluation_email_sha256", sha(EVAL_EMAIL))
    return sn.check_evaluation_delta(build(before_rows), build(after_rows), **kwargs)


def test_a_correct_ingestion_passes_every_invariant():
    assert delta(baseline_rows(), evaluated(baseline_rows())) == []
    assert delta(baseline_rows(), evaluated(baseline_rows()), evaluation_email_sha256=sha(EVAL_EMAIL)) == []


def test_zero_new_chunks_is_allowed_and_the_baseline_size_is_not_assumed():
    assert delta(baseline_rows(), evaluated(baseline_rows(), chunks_per_doc=0)) == []
    assert delta(rows(), evaluated(rows())) == []                      # an empty baseline
    assert delta(rows(users=[user_row(1)]), evaluated(rows(users=[user_row(1)]))) == []
    big = rows(users=[user_row(i) for i in range(1, 10)], documents=[doc_row(i, owner_id=1 + i % 9) for i in range(1, 120)])
    assert delta(big, evaluated(big)) == []


def test_no_change_at_all_fails_the_expected_counts():
    found = delta(baseline_rows(), baseline_rows())
    assert {i.code for i in found} == {sn.DELTA_USER_COUNT, sn.DELTA_DOCUMENT_COUNT}


def test_the_expected_counts_are_checked_exactly():
    assert {i.code for i in delta(baseline_rows(), evaluated(baseline_rows(), new_docs=5))} == {sn.DELTA_DOCUMENT_COUNT}
    assert {i.code for i in delta(baseline_rows(), evaluated(baseline_rows(), new_docs=7))} == {sn.DELTA_DOCUMENT_COUNT}
    assert {i.code for i in delta(baseline_rows(), evaluated(baseline_rows(), new_users=2))} == {sn.DELTA_USER_COUNT}
    assert delta(baseline_rows(), evaluated(baseline_rows(), new_users=2, new_docs=3), expected_new_users=2, expected_new_documents=3) == []


def test_a_new_document_owned_by_a_preexisting_user_is_reported():
    found = delta(baseline_rows(), evaluated(baseline_rows(), owner=1))
    assert {i.code for i in found} == {sn.DELTA_DOCUMENT_OWNER}
    assert len(found) == 6


def test_a_new_chunk_on_a_preexisting_document_is_reported():
    later = evaluated(baseline_rows())
    later["document_chunks"].append(chunk_row(9999, 10, index=5))
    found = delta(baseline_rows(), later)
    assert [(i.code, i.location) for i in found] == [(sn.DELTA_CHUNK_PARENT, "document_chunks[9999]")]


def test_the_evaluation_user_identity_is_always_checked():
    found = delta(baseline_rows(), evaluated(baseline_rows()), evaluation_email_sha256=sha("someone-else@example.com"))
    assert [(i.code, i.location) for i in found] == [(sn.DELTA_UNEXPECTED_USER, "users[3]")]
    assert delta(baseline_rows(), evaluated(baseline_rows()), evaluation_email_sha256=sha(EVAL_EMAIL)) == []


def test_the_delta_check_cannot_be_called_without_an_identity_digest():
    before, after = build(baseline_rows()), build(evaluated(baseline_rows()))
    with pytest.raises(TypeError):
        sn.check_evaluation_delta(before, after)                       # keyword-only and required
    for bad in (None, "", "abc", sha(EVAL_EMAIL).upper(), sha(EVAL_EMAIL)[:-1], 7, sha(EVAL_EMAIL) + "0"):
        with pytest.raises(EvaluationDataError) as error:
            sn.check_evaluation_delta(before, after, evaluation_email_sha256=bad)
        assert codes(error.value) == [sn.DELTA_IDENTITY_INVALID]


def test_a_wrong_arbitrary_new_user_is_not_accepted_even_when_every_count_is_right():
    later = evaluated(baseline_rows())
    later["users"][-1] = user_row(3, email="arbitrary-new-user@example.com")
    found = delta(baseline_rows(), later)
    assert [(i.code, i.location) for i in found] == [(sn.DELTA_UNEXPECTED_USER, "users[3]")]


def test_every_new_user_must_match_the_digest_when_several_are_expected():
    later = evaluated(baseline_rows(), new_users=2)
    later["users"][-1] = user_row(4, email="other@example.com")
    found = delta(baseline_rows(), later, expected_new_users=2)
    assert [(i.code, i.location) for i in found] == [(sn.DELTA_UNEXPECTED_USER, "users[4]")]


def test_identity_findings_never_contain_the_digest_or_an_email():
    later = evaluated(baseline_rows())
    shown = sn.format_report(sn.compare_snapshots(build(baseline_rows()), build(later)),
                             delta(baseline_rows(), later, evaluation_email_sha256=sha("someone-else@example.com")))
    assert sha(EVAL_EMAIL) not in shown and sha("someone-else@example.com") not in shown
    assert "example.com" not in shown


def test_unexpected_changes_to_preexisting_rows_are_each_reported():
    later = evaluated(baseline_rows())
    later["users"][0]["hashed_password_sha256"] = sha("changed")
    later["documents"][1]["indexing_status"] = "failed"
    later["document_chunks"][0]["embedding_sha256"] = sha("[0]")
    found = {(i.code, i.location) for i in delta(baseline_rows(), later)}
    assert found == {
        (sn.DELTA_PREEXISTING_MODIFIED, "users[1].hashed_password_sha256"),
        (sn.DELTA_PREEXISTING_MODIFIED, "documents[11].indexing_status"),
        (sn.DELTA_PREEXISTING_MODIFIED, "document_chunks[100].embedding"),
    }


def test_a_removed_preexisting_row_is_reported_even_when_the_counts_are_made_to_match():
    later = evaluated(baseline_rows(), new_docs=7)       # one extra new document to hide the loss
    del later["documents"][0]                            # a pre-existing document disappears
    later["document_chunks"] = [c for c in later["document_chunks"] if c["document_id"] != 10]
    found = delta(baseline_rows(), later)
    assert any(i.code == sn.DELTA_PREEXISTING_REMOVED and i.location == "documents[10]" for i in found)
    assert any(i.code == sn.DELTA_PREEXISTING_REMOVED and i.location.startswith("document_chunks[") for i in found)


def test_swapping_identities_while_keeping_counts_equal_is_caught():
    later = evaluated(baseline_rows())
    later["users"][1] = user_row(77, email="other@example.com")      # pre-existing user 2 replaced, count unchanged
    found = delta(baseline_rows(), later)
    assert any(i.code == sn.DELTA_PREEXISTING_REMOVED and i.location == "users[2]" for i in found)
    assert any(i.code == sn.DELTA_USER_COUNT for i in found)


# The document ids named by the E1.5.2 preflight as baseline exceptions: 14 documents with a NULL
# stored_name and 4 documents whose stored file is missing. Only these ids (and, for document 39,
# the failed status, the error text and the 22-character extracted text) come from the preflight.
# Every other value of the fixture records below is invented for the test. The fixture is NOT the
# live baseline; the real baseline is whatever the S0 snapshot records.
NULL_NAME_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 31, 51)
MISSING_FILE_IDS = (19, 20, 30, 39)
EXCEPTION_IDS = NULL_NAME_IDS + MISSING_FILE_IDS


def no_text(**extra) -> dict:
    return {"extracted_text_is_null": True, "extracted_text_chars": None, "extracted_text_bytes": None,
            "extracted_text_sha256": None, **extra}


def representative_exceptions_baseline() -> dict:
    """Synthetic records that carry the documented exception ids, plus a few invented healthy rows.

    It does not reproduce the complete live baseline (4 users, 37 documents, 173 chunks) or the
    real field values of any record. It only lets the tests show that records with these ids are
    reported when changed or removed and are left alone by a correct ingestion.
    """
    documents = [doc_row(i, stored_name=None, processing_status=None, indexing_status=None, source_sha256=None,
                         indexed_source_version=None, indexed_signature=None, indexed_embedding_id=None,
                         indexed_at=None, indexing_attempt=0, **no_text()) for i in NULL_NAME_IDS]
    for i in (19, 20, 30):
        documents.append(doc_row(i, stored_name=f"{i:032x}.txt", processing_status=None, indexing_status=None,
                                 source_sha256=None, indexed_source_version=None, indexed_signature=None,
                                 indexed_embedding_id=None, indexed_at=None, indexing_attempt=0, **no_text()))
    documents.append(doc_row(39, stored_name=f"{39:032x}.txt", processing_status="failed",
                             processing_error="Stored file not found", indexing_status=None, source_sha256=None,
                             indexed_source_version=None, indexed_signature=None, indexed_embedding_id=None,
                             indexed_at=None, indexing_attempt=0,
                             extracted_text_is_null=False, extracted_text_chars=22, extracted_text_bytes=22,
                             extracted_text_sha256=sha("x" * 22)))
    healthy = [doc_row(i, owner_id=1) for i in (32, 33, 34)]
    return rows(users=[user_row(1), user_row(2), user_row(3), user_row(4)],
                documents=sorted(documents + healthy, key=lambda r: r["id"]),
                chunks=[chunk_row(1000 + i, 32 + i % 3, index=i) for i in range(12)])


def test_the_representative_fixture_carries_the_documented_exception_ids():
    base = representative_exceptions_baseline()
    assert len(EXCEPTION_IDS) == 18
    by_id = {d["id"]: d for d in base["documents"]}
    assert sum(1 for i in NULL_NAME_IDS if by_id[i]["stored_name"] is None) == 14
    assert by_id[39]["processing_error"] == "Stored file not found" and by_id[39]["extracted_text_chars"] == 22
    assert not any(c["document_id"] in EXCEPTION_IDS for c in base["document_chunks"])


def test_representative_records_with_the_exception_ids_are_preserved_by_a_correct_ingestion():
    before = representative_exceptions_baseline()
    result = sn.compare_snapshots(build(before), build(evaluated(before)))
    by = {t.table: t for t in result.tables}
    assert by["documents"].added and not by["documents"].modified and not by["documents"].removed
    assert result.preexisting_unchanged
    assert delta(before, evaluated(before)) == []
    after_documents = {d["id"] for d in build(evaluated(before))["tables"]["documents"]["rows"]}
    assert set(EXCEPTION_IDS) <= after_documents


@pytest.mark.parametrize("record_id,change,field", [
    (1, {"stored_name": "0" * 32 + ".txt"}, "stored_name"),                                  # "repairing" a NULL name
    (2, {"processing_status": "pending"}, "processing_status"),                              # re-queueing
    (19, {"stored_name": None}, "stored_name"),                                             # clearing a missing-file name
    (39, {"processing_error": None, "processing_status": "completed"}, "processing_error"), # retrying document 39
    (39, {"extracted_text_chars": 0, "extracted_text_bytes": 0, "extracted_text_sha256": sha("")}, "extracted_text"),
    (51, {"indexing_status": "pending"}, "indexing_status"),
])
def test_a_repair_of_a_representative_record_with_an_exception_id_is_detected(record_id, change, field):
    before = representative_exceptions_baseline()
    after = evaluated(before)
    next(d for d in after["documents"] if d["id"] == record_id).update(change)
    found = delta(before, after)
    assert (sn.DELTA_PREEXISTING_MODIFIED, f"documents[{record_id}].{field}") in {(i.code, i.location) for i in found}


@pytest.mark.parametrize("record_id", EXCEPTION_IDS)
def test_deleting_a_representative_record_with_an_exception_id_is_detected(record_id):
    before = representative_exceptions_baseline()
    after = evaluated(before)
    after["documents"] = [d for d in after["documents"] if d["id"] != record_id]
    found = delta(before, after)
    assert (sn.DELTA_PREEXISTING_REMOVED, f"documents[{record_id}]") in {(i.code, i.location) for i in found}


# ---------- H. report ----------

def test_the_report_is_deterministic_and_contains_only_ids_counts_fields_and_fingerprints():
    before = representative_exceptions_baseline()
    after = evaluated(before)
    after["documents"][0]["stored_name"] = "changed.txt"
    snapshots = (build(before), build(after))
    comparison = sn.compare_snapshots(*snapshots)
    issues = sn.check_evaluation_delta(*snapshots, evaluation_email_sha256=sha(EVAL_EMAIL))
    report = sn.format_report(comparison, issues)
    assert report == sn.format_report(sn.compare_snapshots(*snapshots), sn.check_evaluation_delta(*snapshots, evaluation_email_sha256=sha(EVAL_EMAIL)))
    assert "identical: no" in report and "pre-existing rows unchanged: no" in report
    assert "modified id 1: stored_name" in report and "delta_preexisting_modified at documents[1].stored_name" in report
    assert "changed.txt" not in report and "@example.com" not in report
    assert "findings: none" in sn.format_report(sn.compare_snapshots(snapshots[0], snapshots[0]))


def test_long_id_lists_are_truncated_in_the_report():
    big = rows(users=[user_row(1)], documents=[doc_row(i) for i in range(1, 120)])
    report = sn.format_report(sn.compare_snapshots(build(rows(users=[user_row(1)])), build(big)))
    assert "added (119):" in report and "(+69 more)" in report


# ---------- I. import isolation ----------

def test_the_module_loads_no_database_or_heavy_dependency_and_no_application_module():
    code = (
        "import sys\n"
        "import app.rag.evaluation.snapshot as m\n"
        "m.build_snapshot({'users': [], 'documents': [], 'document_chunks': []}, "
        "created_at=__import__('datetime').datetime(2026, 1, 1, tzinfo=__import__('datetime').timezone.utc), database={})\n"
        "heavy = ('sqlalchemy', 'psycopg2', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf', 'pgvector')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy))\n"
        "print(sorted(x for x in sys.modules if x.startswith('app.') and not x.startswith('app.rag')))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    heavy, app_modules = result.stdout.strip().splitlines()
    assert heavy == "[]"
    assert app_modules == "[]"


# ---------- J. environment compatibility ----------

def identity(**override) -> dict:
    return {**DB, **override}


def compare_environments(first: dict, second: dict):
    return sn.compare_snapshots(build(baseline_rows(), database=first), build(baseline_rows(), database=second))


def test_a_restored_database_with_a_different_name_time_zone_and_minor_version_is_comparable():
    restored = identity(database_name="documind_restore_check", time_zone="Asia/Kolkata",
                        server_version="16.4 (Debian 16.4-1.pgdg120+2)")
    result = compare_environments(identity(database_name="documind_db"), restored)
    assert result.identical and result.preexisting_unchanged
    assert build(baseline_rows(), database=restored)["fingerprint"] == build(baseline_rows())["fingerprint"]


@pytest.mark.parametrize("key,value", [
    ("pgvector_version", "0.8.6"),
    ("pgvector_version", "0.9.0"),
    ("extra_float_digits", "3"),
    ("extra_float_digits", "0"),
    ("server_encoding", "LATIN1"),
    ("server_version", "17.2"),
    ("server_version", "15.9"),
])
def test_incompatible_environments_are_rejected_in_both_directions(key, value):
    other = identity(**{key: value})
    for first, second in ((DB, other), (other, DB)):
        with pytest.raises(EvaluationDataError) as error:
            compare_environments(first, second)
        assert [(i.code, i.location) for i in error.value.issues] == [(sn.COMPARE_ENVIRONMENT, f"database.{key}")]
        shown = " ".join(f"{i.location} {i.message}" for i in error.value.issues)
        assert value not in shown and DB[key] not in shown


def test_every_incompatibility_is_reported_not_only_the_first():
    with pytest.raises(EvaluationDataError) as error:
        compare_environments(DB, identity(pgvector_version="0.8.6", extra_float_digits="2", server_version="17.1"))
    assert {i.location for i in error.value.issues} == {
        "database.pgvector_version", "database.extra_float_digits", "database.server_version"}


@pytest.mark.parametrize("key", ["pgvector_version", "extra_float_digits", "server_encoding", "server_version"])
@pytest.mark.parametrize("bad", [None, "missing"])
def test_a_missing_or_null_environment_property_is_incompatible_even_when_both_sides_agree(key, bad):
    broken = {k: v for k, v in DB.items() if k != key}
    if bad is None:
        broken[key] = None
    with pytest.raises(EvaluationDataError) as error:
        compare_environments(broken, dict(broken))            # equal, but unusable
    assert [(i.code, i.location) for i in error.value.issues] == [(sn.COMPARE_ENVIRONMENT, f"database.{key}")]


@pytest.mark.parametrize("version", ["", "unknown", "17beta1", "v16", ".16"])
def test_an_unreadable_server_version_is_incompatible(version):
    with pytest.raises(EvaluationDataError):
        compare_environments(identity(server_version=version), identity(server_version=version))


def test_snapshots_with_no_recorded_environment_at_all_are_not_comparable():
    with pytest.raises(EvaluationDataError) as error:
        compare_environments({}, {})
    assert {i.code for i in error.value.issues} == {sn.COMPARE_ENVIRONMENT}
    assert len(error.value.issues) == 4


def test_the_environment_is_identity_metadata_and_stays_out_of_the_content_fingerprint():
    other = build(baseline_rows(), database=identity(pgvector_version="0.8.6", extra_float_digits="3"))
    assert other["fingerprint"] == build(baseline_rows())["fingerprint"]        # same content, same fingerprint
    assert other["database"] != build(baseline_rows())["database"]
    with pytest.raises(EvaluationDataError):                                    # but not comparable
        sn.compare_snapshots(build(baseline_rows()), other)


def test_the_evaluation_delta_check_also_rejects_incompatible_environments():
    before = build(baseline_rows())
    after = build(evaluated(baseline_rows()), database=identity(pgvector_version="0.8.6"))
    with pytest.raises(EvaluationDataError) as error:
        sn.check_evaluation_delta(before, after, evaluation_email_sha256=sha(EVAL_EMAIL))
    assert codes(error.value) == [sn.COMPARE_ENVIRONMENT]


def test_environment_issues_handles_non_mapping_input():
    assert [i.code for i in sn.environment_issues("x", {})] == [sn.COMPARE_ENVIRONMENT]


# ---------- K. timestamp semantics ----------

def test_the_timestamp_columns_are_the_five_known_ones_and_are_persisted_columns():
    assert sn.TIMESTAMP_COLUMNS == {
        "users": ("created_at",),
        "documents": ("created_at", "indexing_lease_expires_at", "indexed_at"),
        "document_chunks": ("created_at",),
    }
    for table, columns in sn.TIMESTAMP_COLUMNS.items():
        assert set(columns) <= set(sn.PERSISTED_COLUMNS[table])


def test_every_timestamp_column_is_read_through_the_utc_conversion_and_never_bare():
    assert "AT TIME ZONE 'UTC'" in sn._ts("x")
    for table, columns in sn.TIMESTAMP_COLUMNS.items():
        sql = sn.TABLE_QUERIES[table]
        for column in columns:
            converted = f"{sn._ts(column)} AS {column}"
            assert converted in sql
            assert re.search(rf"(?<![A-Za-z_]){column}(?![A-Za-z_])", sql.replace(converted, "")) is None


def test_collect_snapshot_accepts_timestamptz_columns():
    assert sn.collect_snapshot(FakeDatabase(), created_at=T0)["fingerprint"] == build(baseline_rows())["fingerprint"]


@pytest.mark.parametrize("table,column", [(t, c) for t, cols in sn.TIMESTAMP_COLUMNS.items() for c in cols])
@pytest.mark.parametrize("data_type", ["timestamp without time zone", "date", "text", None])
def test_a_timestamp_column_that_is_not_timestamptz_is_refused_before_any_data_is_read(table, column, data_type):
    fake = FakeDatabase(column_types={(table, column): data_type})
    with pytest.raises(EvaluationDataError) as error:
        sn.collect_snapshot(fake, created_at=T0)
    assert [(i.code, i.location) for i in error.value.issues] == [(sn.SNAPSHOT_TIMESTAMP_TYPE, f"{table}.{column}")]
    assert not any(s in fake.statements for s in sn.TABLE_QUERIES.values())


def test_the_types_of_non_timestamp_columns_do_not_matter():
    fake = FakeDatabase(column_types={("users", "id"): "bigint", ("documents", "filename"): "character varying"})
    assert sn.collect_snapshot(fake, created_at=T0)["tables"]["users"]["row_count"] == 2


def test_formatting_a_timestamp_does_not_depend_on_the_process_time_zone(monkeypatch):
    import time
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    results = set()
    try:
        for zone in ("UTC", "Asia/Kolkata", "America/New_York", "Pacific/Auckland"):
            monkeypatch.setenv("TZ", zone)
            time.tzset()
            results.add(sn.format_timestamp(T0))
    finally:
        monkeypatch.undo()
        time.tzset()
    assert results == {"2026-11-12T13:00:00.123456Z"}


def test_the_session_time_zone_is_recorded_as_information_only():
    a = build(baseline_rows(), database=identity(time_zone="UTC"))
    b = build(baseline_rows(), database=identity(time_zone="Asia/Kolkata"))
    assert a["database"]["time_zone"] != b["database"]["time_zone"]
    assert sn.compare_snapshots(a, b).identical
