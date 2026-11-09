"""Tests for the backfill wiring (C5.5 / CP6c.1).

Real test database through the conftest fixtures. Real select_document_states,
attach_source_status, request_indexing and build_plan are used; no model is loaded
and run_indexing is only ever a recording fake here (the real runner is exercised
end to end in CP6c.2).

Resource behaviour is observed on a private engine with ONE pooled connection and a
short pool timeout (``pool_engine``): a leaked connection makes the next checkout
time out, so a leak fails the test instead of passing silently, and
``pool.checkedout()`` is asserted as well. Rows are created with SessionLocal and
leases are set with the database clock, as in the other backfill tests.
"""
import ast
import dataclasses
import inspect
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from app.core import storage
from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag.backfill import (
    NOTE_NO_TEXT,
    NOTE_REQUIRES_INCLUDE_FAILED,
    Action,
    Category,
    SourceStatus,
    StaleReason,
)
from app.rag.backfill_selection import select_document_states
from app.rag.backfill_sources import attach_source_status
from app.rag.backfill_wiring import (
    FAILURE_RUNNER_FACTORY_MISMATCH,
    FAILURE_RUNNER_FACTORY_UNCHECKABLE,
    FAILURE_SESSION_FACTORY_MISMATCH,
    FAILURE_SESSION_FACTORY_UNCHECKABLE,
    make_reclassify,
    make_request,
    make_run,
    write_target_failure,
)
from app.rag.indexing_service import request_indexing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"


# ---------- helpers (same shape as tests/test_backfill_selection.py) ----------

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


def row(document_id: int):
    with engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT indexing_status, indexing_attempt, indexing_error, "
                "indexing_lease_expires_at IS NOT NULL AS leased "
                "FROM documents WHERE id = :id"
            ),
            {"id": document_id},
        ).one()


def real_state(document_id: int):
    with SessionLocal() as session:
        return select_document_states(session, document_ids=[document_id])[0]


# ---------- fixtures and doubles ----------

@pytest.fixture
def pool_engine():
    """A private engine with ONE pooled connection: a leak shows up as a pool timeout."""
    private = create_engine(engine.url, pool_size=1, max_overflow=0, pool_timeout=3)
    try:
        with private.connect():  # warm up: first-connect initialisation is out of the way
            pass
        assert private.pool.checkedout() == 0
        yield private
    finally:
        private.dispose()


@pytest.fixture
def other_engine():
    """A second, distinct engine for the same database."""
    other = create_engine(engine.url)
    try:
        yield other
    finally:
        other.dispose()


class TrackedSession(Session):
    close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()


class RecordingFactory:
    """A session factory that remembers every session it handed out."""

    def __init__(self, bind):
        self._maker = sessionmaker(bind=bind, class_=TrackedSession, autoflush=False)
        self.sessions = []

    def __call__(self):
        session = self._maker()
        self.sessions.append(session)
        return session


class SpySelect:
    """Delegates to the real select_document_states and records how it was called."""

    def __init__(self, *, probe_readonly=False):
        self.probe_readonly = probe_readonly
        self.connections = []
        self.document_ids = []
        self.readonly_seen = []

    def __call__(self, connection, *, document_ids):
        self.connections.append(connection)
        self.document_ids.append(document_ids)
        if self.probe_readonly:
            self.readonly_seen.append(connection.exec_driver_sql("SHOW transaction_read_only").scalar())
        return select_document_states(connection, document_ids=document_ids)


def reclassifier(bind, *, include_failed=False, embedding_id=EMBEDDING_ID, select=None, attach=None):
    return make_reclassify(
        bind,
        configured_embedding_id=embedding_id,
        include_failed=include_failed,
        select_states=select if select is not None else select_document_states,
        attach_sources=attach if attach is not None else attach_source_status,
    )


def assert_pool_idle(bind):
    assert bind.pool.checkedout() == 0


# ======================================================================
# make_reclassify: classification of real rows
# ======================================================================

def test_a_never_queued_document_is_an_index_candidate(pool_engine):
    document_id = make_document()
    result = reclassifier(pool_engine)(document_id)
    assert (result.document_id, result.category, result.action) == (
        document_id, Category.NEVER_QUEUED, Action.INDEX,
    )


def test_a_current_document_is_skipped(pool_engine):
    document_id = make_indexed()
    add_chunk(document_id)
    result = reclassifier(pool_engine)(document_id)
    assert (result.category, result.action, result.note) == (Category.CURRENT, Action.SKIP, None)


def test_the_configured_embedding_id_decides_staleness(pool_engine):
    document_id = make_indexed()
    add_chunk(document_id)
    result = reclassifier(pool_engine, embedding_id="another/model@rev2")(document_id)
    assert (result.category, result.action) == (Category.INDEXED_STALE, Action.INDEX)
    assert result.stale_reasons == (StaleReason.EMBEDDING_ID,)


@pytest.mark.parametrize(
    "include_failed, action, note",
    [(False, Action.REPORT, NOTE_REQUIRES_INCLUDE_FAILED), (True, Action.INDEX, None)],
)
def test_a_failed_document_follows_include_failed(pool_engine, include_failed, action, note):
    document_id = make_document(indexing_status="failed", indexing_error="embedding_failed")
    result = reclassifier(pool_engine, include_failed=include_failed)(document_id)
    assert (result.category, result.action, result.note) == (Category.FAILED, action, note)


@pytest.mark.parametrize(
    "include_failed, action, note",
    [(False, Action.REPORT, NOTE_REQUIRES_INCLUDE_FAILED), (True, Action.INDEX, None)],
)
def test_an_indexed_document_with_a_retained_error_follows_include_failed(
    pool_engine, include_failed, action, note
):
    document_id = make_indexed(indexing_error="embedding_failed")
    add_chunk(document_id)
    result = reclassifier(pool_engine, include_failed=include_failed)(document_id)
    assert (result.category, result.action, result.note) == (Category.INDEXED_WITH_ERROR, action, note)


@pytest.mark.parametrize("empty", ["", None])
def test_a_document_without_text_is_reported_not_indexed(pool_engine, empty):
    document_id = make_document(text_=empty)
    result = reclassifier(pool_engine)(document_id)
    assert (result.category, result.action, result.note) == (Category.NO_TEXT, Action.REPORT, NOTE_NO_TEXT)


def test_a_pdf_whose_file_is_missing_is_blocked(pool_engine):
    document_id = make_document(filename="a.pdf", stored_name="missing.pdf", content_type="application/pdf")
    result = reclassifier(pool_engine)(document_id)
    assert (result.action, result.note) == (Action.BLOCKED, SourceStatus.PDF_FILE_MISSING.value)


def test_a_pdf_without_a_stored_name_is_blocked(pool_engine):
    document_id = make_document(filename="a.pdf", stored_name=None, content_type="application/pdf")
    result = reclassifier(pool_engine)(document_id)
    assert (result.action, result.note) == (Action.BLOCKED, SourceStatus.PDF_NO_STORED_NAME.value)


def test_a_pdf_whose_file_exists_is_an_index_candidate(
    pool_engine, tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "UPLOAD_ROOT", tmp_path)
    (tmp_path / "present.pdf").write_bytes(b"%PDF-1.4\n")

    document_id = make_document(
        filename="a.pdf",
        stored_name="present.pdf",
        content_type="application/pdf",
    )

    result = reclassifier(pool_engine)(document_id)

    assert (result.action, result.note) == (Action.INDEX, None)


def test_a_live_lease_is_skipped_and_an_expired_one_is_abandoned(pool_engine):
    reclassify = reclassifier(pool_engine)
    live = reclassify(make_indexing(300))
    abandoned = reclassify(make_indexing(-5))
    assert (live.category, live.action) == (Category.LIVE_LEASE, Action.SKIP)
    assert (abandoned.category, abandoned.action) == (Category.ABANDONED_LEASE, Action.INDEX)


def test_every_call_reads_the_database_state_of_that_moment(pool_engine):
    reclassify = reclassifier(pool_engine)
    document_id = make_document()
    assert reclassify(document_id).action is Action.INDEX

    execute(
        "UPDATE documents SET indexing_status='indexed', indexed_source_version=1, "
        "source_sha256=:sha, indexed_signature='{}', indexed_embedding_id=:emb, indexed_at=now() "
        "WHERE id=:id",
        sha="a" * 64, emb=EMBEDDING_ID, id=document_id,
    )
    add_chunk(document_id)
    assert reclassify(document_id).category is Category.CURRENT


def test_only_the_requested_document_is_selected_and_classified(pool_engine):
    first, second, third = make_document(), make_document(), make_document()
    spy = SpySelect()
    result = reclassifier(pool_engine, select=spy)(second)
    assert result.document_id == second
    assert spy.document_ids == [[second]]
    assert first != second != third


# ======================================================================
# make_reclassify: None, RuntimeError and argument checks
# ======================================================================

def test_an_unknown_document_returns_none(pool_engine):
    assert reclassifier(pool_engine)(999_999) is None


@pytest.mark.parametrize("status", ["pending", "processing", "failed", None])
def test_a_document_that_is_not_completed_returns_none(pool_engine, status):
    document_id = make_document(processing_status=status)
    assert reclassifier(pool_engine)(document_id) is None


def test_more_than_one_state_raises_before_sources_are_resolved(pool_engine):
    document_id = make_document()
    state = real_state(document_id)
    attached = []
    reclassify = reclassifier(
        pool_engine,
        select=lambda connection, *, document_ids: [state, state],
        attach=lambda states: attached.append(states) or states,
    )
    with pytest.raises(RuntimeError, match="more than one state"):
        reclassify(document_id)
    assert attached == []
    assert_pool_idle(pool_engine)


def test_a_state_for_another_document_raises(pool_engine):
    document_id = make_document()
    wrong = dataclasses.replace(real_state(document_id), document_id=document_id + 1)
    attached = []
    reclassify = reclassifier(
        pool_engine,
        select=lambda connection, *, document_ids: [wrong],
        attach=lambda states: attached.append(states) or states,
    )
    with pytest.raises(RuntimeError, match="different document"):
        reclassify(document_id)
    assert attached == []
    assert_pool_idle(pool_engine)


@pytest.mark.parametrize("changed", [lambda states: [], lambda states: [states[0], states[0]]])
def test_source_resolution_that_changes_the_number_of_states_raises(pool_engine, changed):
    document_id = make_document()
    reclassify = reclassifier(pool_engine, attach=changed)
    with pytest.raises(RuntimeError, match="source resolution"):
        reclassify(document_id)
    assert_pool_idle(pool_engine)


@pytest.mark.parametrize("bad", ["", None, 7])
def test_the_embedding_id_must_be_a_non_empty_string(pool_engine, bad):
    with pytest.raises(ValueError):
        reclassifier(pool_engine, embedding_id=bad)


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_include_failed_must_be_a_real_bool(pool_engine, bad):
    with pytest.raises(TypeError):
        reclassifier(pool_engine, include_failed=bad)


# ======================================================================
# make_reclassify: read-only connection and resource handling
# ======================================================================

def test_the_connection_is_read_only_before_the_first_statement(pool_engine):
    document_id = make_document()
    spy = SpySelect(probe_readonly=True)
    statements = []

    def record(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(pool_engine, "before_cursor_execute", record)
    try:
        reclassifier(pool_engine, select=spy)(document_id)
    finally:
        event.remove(pool_engine, "before_cursor_execute", record)

    assert spy.readonly_seen == ["on"]
    # The probe was the connection's very first statement, so the setting preceded all SQL.
    assert statements[0].startswith("SHOW transaction_read_only")
    assert any("FROM documents" in statement for statement in statements[1:])


def test_a_write_through_the_reclassify_connection_is_refused_by_the_database(pool_engine):
    document_id = make_document()
    before = row(document_id)

    def writing(connection, *, document_ids):
        connection.execute(
            text("UPDATE documents SET indexing_error = 'x' WHERE id = :id"), {"id": document_id}
        )
        return []

    with pytest.raises(DBAPIError):
        reclassifier(pool_engine, select=writing)(document_id)
    assert row(document_id) == before
    assert_pool_idle(pool_engine)


def test_the_connection_is_closed_and_released_after_success(pool_engine):
    document_id = make_document()
    spy = SpySelect()
    reclassifier(pool_engine, select=spy)(document_id)
    assert len(spy.connections) == 1
    assert spy.connections[0].closed
    assert_pool_idle(pool_engine)


def test_the_connection_is_closed_and_released_when_selection_raises(pool_engine):
    seen = []
    failure = ValueError("selection exploded")

    def raising(connection, *, document_ids):
        seen.append(connection)
        raise failure

    with pytest.raises(ValueError) as info:
        reclassifier(pool_engine, select=raising)(1)
    assert info.value is failure
    assert seen[0].closed
    assert_pool_idle(pool_engine)


def test_every_call_uses_its_own_connection(pool_engine):
    document_id = make_document()
    spy = SpySelect()
    reclassify = reclassifier(pool_engine, select=spy)
    reclassify(document_id)
    reclassify(document_id)
    assert spy.connections[0] is not spy.connections[1]
    assert all(connection.closed for connection in spy.connections)


def test_a_write_through_the_same_pool_succeeds_after_a_read_only_call(pool_engine):
    document_id = make_document()
    reclassifier(pool_engine)(document_id)

    # One pooled connection: this is certainly the DBAPI connection that was read-only.
    with pool_engine.connect() as connection:
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "off"
    with pool_engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET indexing_error = 'written' WHERE id = :id"), {"id": document_id}
        )
    assert row(document_id).indexing_error == "written"
    assert_pool_idle(pool_engine)


def test_repeated_calls_never_leak_a_connection(pool_engine):
    good = make_document()
    reclassify = reclassifier(pool_engine)

    def raising(connection, *, document_ids):
        raise RuntimeError("boom")

    failing = reclassifier(pool_engine, select=raising)
    for _ in range(5):  # the pool holds ONE connection: a leak would time out here
        assert reclassify(good).action is Action.INDEX
        assert reclassify(999_999) is None
        with pytest.raises(RuntimeError):
            failing(good)
    assert_pool_idle(pool_engine)


# ======================================================================
# make_request
# ======================================================================

def test_each_request_uses_a_fresh_session_with_no_transaction_open(pool_engine):
    document_id = make_document()
    factory = RecordingFactory(pool_engine)
    seen = []

    def spy(session, requested_id):
        seen.append((session, session.in_transaction()))
        return request_indexing(session, requested_id)

    request = make_request(factory, spy)
    assert request(document_id) is True
    assert request(document_id) is True  # idempotent for an already pending document

    assert len(factory.sessions) == 2
    assert factory.sessions[0] is not factory.sessions[1]
    assert [session for session, _ in seen] == factory.sessions
    assert [in_transaction for _, in_transaction in seen] == [False, False]
    assert [session.close_calls for session in factory.sessions] == [1, 1]
    assert_pool_idle(pool_engine)


@pytest.mark.parametrize("result", [True, False, None, "queued", 1])
def test_the_request_result_is_returned_unchanged(pool_engine, result):
    request = make_request(RecordingFactory(pool_engine), lambda session, document_id: result)
    assert request(1) is result


def test_the_request_is_committed_and_visible_from_another_connection(pool_engine):
    document_id = make_document()
    assert row(document_id).indexing_status is None

    request = make_request(RecordingFactory(pool_engine), request_indexing)
    assert request(document_id) is True

    # ``row`` uses a different engine, so only committed data can be seen here.
    after = row(document_id)
    assert after.indexing_status == "pending"
    assert after.leased is False
    assert_pool_idle(pool_engine)


def test_an_expired_lease_is_requeued(pool_engine):
    document_id = make_indexing(-5)
    request = make_request(RecordingFactory(pool_engine), request_indexing)
    assert request(document_id) is True
    assert row(document_id).indexing_status == "pending"


def test_a_live_lease_returns_false_and_changes_nothing(pool_engine):
    document_id = make_indexing(300)
    before = row(document_id)
    request = make_request(RecordingFactory(pool_engine), request_indexing)
    assert request(document_id) is False
    assert row(document_id) == before
    assert_pool_idle(pool_engine)


def test_a_missing_document_returns_false(pool_engine):
    request = make_request(RecordingFactory(pool_engine), request_indexing)
    assert request(999_999) is False
    assert_pool_idle(pool_engine)


def test_a_document_whose_extraction_is_not_completed_returns_false(pool_engine):
    document_id = make_document(processing_status="pending")
    request = make_request(RecordingFactory(pool_engine), request_indexing)
    assert request(document_id) is False
    assert row(document_id).indexing_status is None


def test_the_session_is_closed_and_the_work_rolled_back_when_the_service_raises(pool_engine):
    document_id = make_document()
    factory = RecordingFactory(pool_engine)
    failure = RuntimeError("service exploded")

    def failing(session, requested_id):
        with session.begin():
            session.execute(
                text("UPDATE documents SET indexing_status = 'pending' WHERE id = :id"),
                {"id": requested_id},
            )
            raise failure

    with pytest.raises(RuntimeError) as info:
        make_request(factory, failing)(document_id)

    assert info.value is failure
    assert factory.sessions[0].close_calls == 1
    assert row(document_id).indexing_status is None  # nothing was committed
    assert_pool_idle(pool_engine)


def test_repeated_requests_never_leak_a_connection(pool_engine):
    document_id = make_document()
    request = make_request(RecordingFactory(pool_engine), request_indexing)
    for _ in range(5):  # one pooled connection: a leak would time out
        assert request(document_id) is True
        assert request(999_999) is False
    assert_pool_idle(pool_engine)


# ======================================================================
# make_run
# ======================================================================

def test_run_passes_the_exact_embedder_and_returns_the_result_unchanged():
    embedder = object()  # never touched: only passed through
    result = object()
    calls = []

    def fake_run_indexing(document_id, *, embedder):
        calls.append((document_id, embedder))
        return result

    run = make_run(fake_run_indexing, embedder)
    assert run(41) is result
    assert run(42) is result
    assert len(calls) == 2
    assert [document_id for document_id, _ in calls] == [41, 42]
    assert all(passed is embedder for _, passed in calls)


def test_run_rejects_a_none_embedder_but_not_a_falsy_one():
    with pytest.raises(ValueError):
        make_run(lambda document_id, *, embedder: None, None)

    class FalsyEmbedder:
        def __len__(self):
            return 0

    falsy = FalsyEmbedder()
    seen = []
    make_run(lambda document_id, *, embedder: seen.append(embedder), falsy)(1)
    assert seen == [falsy] and seen[0] is falsy


@pytest.mark.parametrize("failure", [ValueError("x"), KeyboardInterrupt()])
def test_run_lets_every_exception_through_unchanged(failure):
    def raising(document_id, *, embedder):
        raise failure

    with pytest.raises(type(failure)) as info:
        make_run(raising, object())(1)
    assert info.value is failure


def test_the_real_run_indexing_accepts_the_keyword_make_run_passes():
    from app.rag.indexing_runner import run_indexing

    parameter = inspect.signature(run_indexing).parameters["embedder"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# ======================================================================
# write_target_failure
# ======================================================================

def test_both_factories_bound_to_the_engine_pass(pool_engine):
    queue, runner = RecordingFactory(pool_engine), RecordingFactory(pool_engine)
    assert write_target_failure(pool_engine, queue, runner) is None
    assert [s.close_calls for s in queue.sessions + runner.sessions] == [1, 1]


def test_the_real_application_factories_are_bound_to_the_application_engine():
    from app.rag import indexing_runner

    assert write_target_failure(engine, SessionLocal, indexing_runner.SessionLocal) is None


def test_a_queuing_factory_on_another_engine_is_refused(pool_engine, other_engine):
    result = write_target_failure(
        pool_engine, RecordingFactory(other_engine), RecordingFactory(pool_engine)
    )
    assert result == FAILURE_SESSION_FACTORY_MISMATCH


def test_a_runner_factory_on_another_engine_is_refused(pool_engine, other_engine):
    result = write_target_failure(
        pool_engine, RecordingFactory(pool_engine), RecordingFactory(other_engine)
    )
    assert result == FAILURE_RUNNER_FACTORY_MISMATCH


def test_the_queuing_factory_is_reported_first_when_both_are_wrong(pool_engine, other_engine):
    result = write_target_failure(
        pool_engine, RecordingFactory(other_engine), RecordingFactory(other_engine)
    )
    assert result == FAILURE_SESSION_FACTORY_MISMATCH


def test_identity_not_equality_decides(pool_engine):
    twin = create_engine(pool_engine.url)  # same URL, different object
    try:
        result = write_target_failure(pool_engine, RecordingFactory(twin), RecordingFactory(pool_engine))
    finally:
        twin.dispose()
    assert result == FAILURE_SESSION_FACTORY_MISMATCH


def _raising_factory():
    raise RuntimeError("password=hunter2 host=db.internal")


class _NoBindSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "broken",
    [
        _raising_factory,                 # the factory itself raises
        None,                             # not callable
        lambda: _NoBindSession(),         # a session without get_bind
        sessionmaker(),                   # a real, UNBOUND sessionmaker: get_bind raises
    ],
    ids=["raises", "none", "no-get_bind", "unbound-sessionmaker"],
)
def test_an_uncheckable_factory_is_refused_with_a_fixed_text(pool_engine, broken):
    good = RecordingFactory(pool_engine)
    queue_result = write_target_failure(pool_engine, broken, good)
    runner_result = write_target_failure(pool_engine, good, broken)
    assert queue_result == FAILURE_SESSION_FACTORY_UNCHECKABLE
    assert runner_result == FAILURE_RUNNER_FACTORY_UNCHECKABLE
    for message in (queue_result, runner_result):
        assert "hunter2" not in message and "db.internal" not in message


def test_the_four_refusal_texts_are_distinct_and_carry_no_url_or_name():
    messages = {
        FAILURE_SESSION_FACTORY_MISMATCH,
        FAILURE_RUNNER_FACTORY_MISMATCH,
        FAILURE_SESSION_FACTORY_UNCHECKABLE,
        FAILURE_RUNNER_FACTORY_UNCHECKABLE,
    }
    assert len(messages) == 4
    for message in messages:
        assert "://" not in message and "documind" not in message.lower()


def test_inspecting_the_binding_opens_no_connection():
    fresh = create_engine(engine.url, pool_size=1, max_overflow=0, pool_timeout=3)
    events = []
    event.listen(fresh, "connect", lambda *args: events.append("connect"))
    event.listen(fresh, "checkout", lambda *args: events.append("checkout"))
    event.listen(fresh, "before_cursor_execute", lambda *args: events.append("statement"))
    try:
        queue, runner = RecordingFactory(fresh), RecordingFactory(fresh)
        assert write_target_failure(fresh, queue, runner) is None
        assert events == []
        assert fresh.pool.checkedout() == 0
        assert [s.close_calls for s in queue.sessions + runner.sessions] == [1, 1]
    finally:
        fresh.dispose()


# ======================================================================
# import isolation
# ======================================================================

FORBIDDEN_MODULES = (
    "app.core.config", "app.core.database", "app.core.storage", "app.models", "app.main",
    "app.rag.embedder", "app.rag.indexing", "app.rag.indexing_service", "app.rag.indexing_runner",
    "app.rag.indexing_hook", "app.rag.pdf_pages", "app.rag.backfill_selection",
    "app.rag.backfill_sources", "sqlalchemy", "psycopg2", "fastapi",
    "torch", "transformers", "sentence_transformers", "huggingface_hub", "tokenizers",
)


def _modules_loaded_by(statement: str) -> set[str]:
    code = f"import sys, json\n{statement}\nprint(json.dumps(sorted(sys.modules)))\n"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


def _forbidden_in(modules: set[str]) -> list[str]:
    return sorted(
        name for name in modules
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES)
    )


def test_importing_the_module_loads_no_application_database_or_model_code():
    loaded = _modules_loaded_by("import app.rag.backfill_wiring")
    assert "app.rag.backfill_wiring" in loaded and "app.rag.backfill" in loaded  # really imported
    assert _forbidden_in(loaded) == []


def test_the_isolation_check_itself_detects_an_import():
    loaded = _modules_loaded_by("import sqlalchemy")  # a positive control for the detector
    assert "sqlalchemy" in _forbidden_in(loaded)


def test_the_module_source_imports_only_the_classifier_and_the_standard_library():
    source = (PROJECT_ROOT / "app" / "rag" / "backfill_wiring.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {"__future__", "collections.abc", "typing", "app.rag.backfill"}