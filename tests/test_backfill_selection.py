"""Tests for the read-only backfill selection query (C5.5 / CP2).

Real test database through the existing conftest fixtures. Rows are inserted with
SessionLocal, as the other indexing tests do; leases are set with the database clock
so the Python process is never the source of "now".
"""
import dataclasses
import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError

from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag.backfill import (
    Action,
    Category,
    DocumentState,
    LeaseState,
    SourceStatus,
    build_plan,
)
from app.rag.backfill_selection import (
    _state_from_row,
    build_selection_statement,
    select_document_states,
)

EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"

EXPECTED_COLUMNS = [
    "document_id",
    "processing_status",
    "indexing_status",
    "indexing_error",
    "lease",
    "source_version",
    "indexed_source_version",
    "indexed_embedding_id",
    "has_text",
    "chunks_present",
    "filename",
    "stored_name",
    "content_type",
]


# ---------- helpers ----------

def execute(sql, **params):
    with engine.begin() as connection:
        connection.execute(text(sql), params)


def make_document(
    *,
    text_="hello world",
    processing_status="completed",
    filename="notes.txt",
    stored_name="abc.txt",
    content_type="text/plain",
    **columns,
) -> int:
    with SessionLocal() as session:
        user = User(email=f"u{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        document = Document(
            filename=filename,
            owner_id=user.id,
            processing_status=processing_status,
            stored_name=stored_name,
            content_type=content_type,
            extracted_text=text_,
            **columns,
        )
        session.add(document)
        session.commit()
        return document.id


def make_indexed(**kwargs) -> int:
    """A document whose published-metadata columns are complete (the DB requires them)."""
    return make_document(
        indexing_status="indexed",
        indexed_source_version=1,
        source_sha256="a" * 64,
        indexed_signature="{}",
        indexed_embedding_id=EMBEDDING_ID,
        indexed_at=datetime.now(timezone.utc),
        **kwargs,
    )


def make_indexing(seconds_from_now: int) -> int:
    """An 'indexing' document whose lease expires seconds_from_now on the DB clock."""
    document_id = make_document(
        indexing_status="indexing",
        indexing_lease_expires_at=datetime.now(timezone.utc),  # placeholder: the DB requires a lease
    )
    execute(
        "UPDATE documents SET indexing_lease_expires_at = "
        "clock_timestamp() + (:seconds * interval '1 second') WHERE id = :id",
        seconds=seconds_from_now,
        id=document_id,
    )
    return document_id


def add_chunk(document_id: int, index: int = 0) -> None:
    with SessionLocal() as session:
        session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                start_offset=0,
                end_offset=5,
                text="hello",
                span_sha256="0" * 64,
                embedding=[0.0] * 384,
            )
        )
        session.commit()


def load(**kwargs) -> list[DocumentState]:
    with SessionLocal() as session:
        return select_document_states(session, **kwargs)


def by_id(states) -> dict[int, DocumentState]:
    return {state.document_id: state for state in states}


def snapshot():
    """Row versions of every document: xmin changes whenever a row is written."""
    with engine.connect() as connection:
        return [tuple(row) for row in connection.execute(text("SELECT id, xmin::text FROM documents ORDER BY id"))]


# ---------- filter and ordering ----------

def test_only_completed_documents_are_selected():
    completed = make_document()
    make_document(processing_status="pending")
    make_document(processing_status="processing")
    make_document(processing_status="failed")
    make_document(processing_status=None)
    assert [state.document_id for state in load()] == [completed]
    assert load()[0].processing_status == "completed"


def test_an_empty_database_returns_nothing():
    assert load() == []


def test_the_result_is_sorted_by_document_id():
    ids = [make_document() for _ in range(4)]
    assert [state.document_id for state in load()] == sorted(ids)


# ---------- field mapping ----------

def test_a_fully_indexed_document_maps_every_documentstate_field():
    document_id = make_indexed()
    add_chunk(document_id)
    expected = DocumentState(
        document_id=document_id,
        processing_status="completed",
        indexing_status="indexed",
        indexing_error=None,
        lease=LeaseState.NONE,
        source_version=1,
        indexed_source_version=1,
        indexed_embedding_id=EMBEDDING_ID,
        has_text=True,
        chunks_present=True,
        filename="notes.txt",
        stored_name="abc.txt",
        content_type="text/plain",
        source=SourceStatus.NOT_PDF,
    )
    assert load() == [expected]


def test_a_never_queued_document_has_null_indexing_fields():
    document_id = make_document()
    state = by_id(load())[document_id]
    assert state.indexing_status is None
    assert state.indexing_error is None
    assert state.indexed_source_version is None
    assert state.indexed_embedding_id is None
    assert state.chunks_present is False
    assert state.lease is LeaseState.NONE


def test_a_failed_document_carries_its_status_and_error():
    document_id = make_document(indexing_status="failed", indexing_error="boom")
    state = by_id(load())[document_id]
    assert state.indexing_status == "failed"
    assert state.indexing_error == "boom"


def test_a_retained_error_on_an_indexed_document_is_carried():
    document_id = make_indexed(indexing_error="last attempt failed")
    state = by_id(load())[document_id]
    assert state.indexing_status == "indexed"
    assert state.indexing_error == "last attempt failed"
    assert state.indexed_embedding_id == EMBEDDING_ID
    assert state.indexed_source_version == 1


def test_a_different_embedding_id_is_carried_as_stored():
    document_id = make_indexed()
    execute("UPDATE documents SET indexed_embedding_id = 'other@rev' WHERE id = :id", id=document_id)
    assert by_id(load())[document_id].indexed_embedding_id == "other@rev"


# ---------- lease, on the database clock ----------

@pytest.mark.parametrize("seconds", [3600, 60])
def test_a_lease_in_the_future_is_live(seconds):
    document_id = make_indexing(seconds)
    assert by_id(load())[document_id].lease is LeaseState.LIVE


@pytest.mark.parametrize("seconds", [-1, -3600])
def test_a_lease_in_the_past_is_expired(seconds):
    document_id = make_indexing(seconds)
    assert by_id(load())[document_id].lease is LeaseState.EXPIRED


def test_documents_that_are_not_indexing_have_no_lease():
    pending = make_document(indexing_status="pending")
    indexed = make_indexed()
    states = by_id(load())
    assert states[pending].lease is LeaseState.NONE
    assert states[indexed].lease is LeaseState.NONE


# ---------- has_text ----------

@pytest.mark.parametrize(
    "text_, expected",
    [
        pytest.param(None, False, id="null"),
        pytest.param("", False, id="empty"),
        pytest.param("x", True, id="one-character"),
        pytest.param("héllo 日本語", True, id="multibyte"),
        pytest.param("   \n", True, id="whitespace-only-counts-as-text"),
    ],
)
def test_has_text_is_computed_by_the_database(text_, expected):
    document_id = make_document(text_=text_)
    assert by_id(load())[document_id].has_text is expected


def test_the_text_itself_never_reaches_the_states():
    marker = "SECRET-MARKER-"
    make_document(text_=marker * 1000)
    (state,) = load()
    assert all(marker not in str(value) for value in dataclasses.astuple(state))


# ---------- chunks_present ----------

def test_chunks_present_is_false_without_chunks_and_true_with_one():
    without = make_indexed()
    with_chunk = make_indexed()
    add_chunk(with_chunk)
    states = by_id(load())
    assert states[without].chunks_present is False
    assert states[with_chunk].chunks_present is True


def test_chunks_present_is_per_document():
    first = make_document()
    second = make_document()
    third = make_document()
    add_chunk(second, 0)
    add_chunk(second, 1)
    states = by_id(load())
    assert [states[i].chunks_present for i in (first, second, third)] == [False, True, False]


# ---------- initial source status (no filesystem) ----------

@pytest.mark.parametrize(
    "filename, stored_name, content_type, expected",
    [
        pytest.param("a.txt", "x.txt", "text/plain", SourceStatus.NOT_PDF, id="text"),
        pytest.param("a.pdf", "x.txt", "text/plain", SourceStatus.PDF_UNCHECKED, id="pdf-filename"),
        pytest.param("a.txt", "x.pdf", "text/plain", SourceStatus.PDF_UNCHECKED, id="pdf-stored-name"),
        pytest.param("a.txt", "x.txt", "application/pdf", SourceStatus.PDF_UNCHECKED, id="pdf-content-type"),
        pytest.param("a.txt", None, "text/plain", SourceStatus.NOT_PDF, id="non-pdf-without-stored-name"),
        pytest.param("a.pdf", None, "application/pdf", SourceStatus.PDF_UNCHECKED, id="pdf-without-stored-name"),
        pytest.param("a.txt", "x.txt", None, SourceStatus.NOT_PDF, id="no-content-type"),
    ],
)
def test_the_initial_source_status_uses_only_the_metadata(filename, stored_name, content_type, expected):
    document_id = make_document(filename=filename, stored_name=stored_name, content_type=content_type)
    assert by_id(load())[document_id].source is expected


# ---------- the id filter ----------

def test_the_id_filter_returns_only_the_requested_documents():
    ids = [make_document() for _ in range(4)]
    assert [s.document_id for s in load(document_ids=[ids[3], ids[1]])] == [ids[1], ids[3]]


def test_unknown_and_non_completed_ids_are_absent():
    completed = make_document()
    pending = make_document(processing_status="pending")
    states = load(document_ids=[completed, pending, 987654321])
    assert [s.document_id for s in states] == [completed]


def test_an_empty_id_list_returns_nothing_without_querying():
    class NoQueries:
        def execute(self, *args, **kwargs):
            raise AssertionError("an empty id list must not query")

    assert select_document_states(NoQueries(), document_ids=[]) == []


@pytest.mark.parametrize("bad", [[True], ["1"], [1.5], "12", [None]])
def test_non_integer_ids_are_rejected_before_any_query(bad):
    class NoQueries:
        def execute(self, *args, **kwargs):
            raise AssertionError("invalid ids must not query")

    with pytest.raises(TypeError):
        select_document_states(NoQueries(), document_ids=bad)


# ---------- one statement ----------

def test_the_selection_is_one_statement_for_many_documents():
    for index in range(20):
        document_id = make_indexed() if index % 2 else make_document()
        if index % 4 == 1:
            add_chunk(document_id)

    statements = []

    def record(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        states = load()
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert len(states) == 20
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT")


# ---------- extracted_text is never selected ----------

def test_extracted_text_is_not_a_selected_column():
    statement = build_selection_statement()
    column = Document.__table__.c.extracted_text
    assert list(statement.selected_columns.keys()) == EXPECTED_COLUMNS
    for selected in statement.selected_columns:
        assert selected is not column
        assert getattr(selected, "element", None) is not column


def test_the_executed_result_has_exactly_the_expected_columns():
    make_document()
    with SessionLocal() as session:
        result = session.execute(build_selection_statement())
        assert list(result.keys()) == EXPECTED_COLUMNS
        result.all()


def test_extracted_text_appears_only_inside_the_has_text_predicate():
    sql = str(build_selection_statement().compile(dialect=postgresql.dialect()))
    select_list = sql.split("\nFROM", 1)[0]
    assert "octet_length(documents.extracted_text)" in select_list
    # Every use is either "... IS NOT NULL" or the argument of octet_length(...).
    assert re.findall(r"documents\.extracted_text(?! IS NOT NULL|\))", select_list) == []


# ---------- read-only ----------

def test_postgresql_readonly_rejects_an_update_on_a_configured_connection():
    document_id = make_document()
    before = snapshot()

    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "on"
        with pytest.raises(DBAPIError) as excinfo:
            connection.execute(
                text("UPDATE documents SET indexing_error = 'x' WHERE id = :id"), {"id": document_id}
            )
        assert excinfo.value.orig.pgcode == "25006"  # read_only_sql_transaction
        connection.rollback()

    assert snapshot() == before
    with SessionLocal() as session:
        assert session.get(Document, document_id).indexing_error is None


def test_a_normal_connection_is_writable():
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "off"


def test_the_read_only_setting_is_reset_when_the_connection_is_returned():
    # One pooled connection, so the second checkout is certainly the same DBAPI connection.
    single = create_engine(engine.url, pool_size=1, max_overflow=0)
    try:
        with single.connect() as connection:
            connection = connection.execution_options(postgresql_readonly=True)
            assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "on"
        with single.connect() as connection:
            assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "off"
    finally:
        single.dispose()


def test_the_loader_runs_on_a_read_only_connection():
    make_indexed()
    add_chunk(make_indexed())
    expected = load()
    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        assert select_document_states(connection) == expected
        connection.rollback()


# ---------- the loader never writes ----------

def test_loading_changes_no_rows():
    make_document()
    make_indexed()
    make_indexing(3600)
    before = snapshot()
    load()
    with engine.connect() as connection:
        select_document_states(connection)
        connection.rollback()
    assert snapshot() == before


def test_a_session_and_a_connection_give_the_same_result():
    make_document()
    add_chunk(make_indexed())
    make_indexing(-60)
    via_session = load()
    with engine.connect() as connection:
        via_connection = select_document_states(connection)
        connection.rollback()
    assert via_session == via_connection


# ---------- the row mapper alone (no database) ----------

def fake_row(**overrides):
    values = dict(
        document_id=1,
        processing_status="completed",
        indexing_status=None,
        indexing_error=None,
        lease="none",
        source_version=1,
        indexed_source_version=None,
        indexed_embedding_id=None,
        has_text=True,
        chunks_present=False,
        filename="a.txt",
        stored_name="a.txt",
        content_type="text/plain",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value, expected", [("none", LeaseState.NONE), ("live", LeaseState.LIVE), ("expired", LeaseState.EXPIRED)])
def test_the_mapper_converts_the_lease_string(value, expected):
    assert _state_from_row(fake_row(lease=value)).lease is expected


def test_an_unexpected_lease_value_raises():
    with pytest.raises(ValueError):
        _state_from_row(fake_row(lease="weird"))


@pytest.mark.parametrize("field", ["has_text", "chunks_present"])
@pytest.mark.parametrize("bad", [None, 1, 0, "t", "true"])
def test_the_computed_booleans_must_really_be_bools(field, bad):
    with pytest.raises(TypeError):
        _state_from_row(fake_row(**{field: bad}))


def test_the_mapper_marks_pdf_like_rows_unchecked():
    assert _state_from_row(fake_row(filename="r.pdf")).source is SourceStatus.PDF_UNCHECKED
    assert _state_from_row(fake_row()).source is SourceStatus.NOT_PDF


# ---------- end to end with the CP1 planner ----------

def test_a_pdf_stays_blocked_until_its_source_is_checked():
    text_id = make_document()
    pdf_id = make_document(filename="report.pdf", stored_name="x.pdf", content_type="application/pdf")

    plan = build_plan(load(), configured_embedding_id=EMBEDDING_ID)
    items = {item.classification.document_id: item.classification for item in plan.items}

    assert items[text_id].category is Category.NEVER_QUEUED
    assert items[text_id].action is Action.INDEX
    assert items[pdf_id].category is Category.NEVER_QUEUED
    assert items[pdf_id].action is Action.BLOCKED
    assert items[pdf_id].note == "pdf_unchecked"