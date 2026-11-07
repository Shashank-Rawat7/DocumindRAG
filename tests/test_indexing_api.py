"""C5.4.1 tests: GET /api/v1/documents/{document_id}/indexing.

Runs against documind_test (conftest builds it with `alembic upgrade head`). States
are written straight to the database, honoring the table's check constraints; lease
times are computed by PostgreSQL, never by Python. No test needs the embedding model.
"""
import hashlib
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, text, update

from app.api import indexing as indexing_api
from app.core.database import SessionLocal, engine
from app.core.rate_limit import limiter
from app.models.chunk import EMBEDDING_DIM, DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import indexing_hook as hook
from app.rag import indexing_runner as runner
from app.rag import indexing_service as svc

DOCUMENTS = "/api/v1/documents"

FIELDS = {
    "document_id", "indexing_status", "indexing_error", "indexing_attempt",
    "indexed_at", "index_current", "published_chunks_present", "in_progress",
}
NOT_FOUND = {"detail": "Document not found"}

LIVE_SECONDS = 3600   # a lease that cannot expire during a test
EXPIRED_SECONDS = -1  # a lease already in the past

PUBLISHED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PUBLICATION = dict(
    indexing_status="indexed",
    indexed_source_version=1,
    source_sha256=hashlib.sha256(b"source text").hexdigest(),
    indexed_signature="test-signature",
    indexed_embedding_id="intfloat/multilingual-e5-small@384",
    indexed_at=PUBLISHED_AT,
)
CHUNK_TEXT = "FastAPI dependency injection"


# ---------- helpers ----------

def create_document(client, headers, filename="notes.txt"):
    response = client.post(DOCUMENTS, json={"filename": filename}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def get_status(client, headers, document_id):
    return client.get(f"{DOCUMENTS}/{document_id}/indexing", headers=headers)


def execute(sql, **params):
    with engine.begin() as connection:
        connection.execute(text(sql), params)


def set_state(document_id, **values):
    """Write columns directly. The caller must respect the table's check constraints."""
    with engine.begin() as connection:
        result = connection.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
    assert result.rowcount == 1


def mark_extracted(document_id):
    """request_indexing only accepts documents whose extraction completed."""
    set_state(document_id, processing_status="completed")


def set_lease(document_id, seconds, attempt=1):
    """'indexing' with a lease that ends `seconds` from now on the DATABASE clock."""
    execute(
        "UPDATE documents SET indexing_status = 'indexing', indexing_attempt = :attempt, "
        "indexing_lease_expires_at = clock_timestamp() "
        "+ make_interval(secs => CAST(:seconds AS double precision)) WHERE id = :id",
        attempt=attempt, seconds=seconds, id=document_id,
    )


def add_chunk(document_id, chunk_index=0, content=CHUNK_TEXT):
    embedding = [0.0] * EMBEDDING_DIM
    embedding[0] = 1.0
    with SessionLocal() as session:
        session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                start_offset=0,
                end_offset=len(content),
                text=content,
                span_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                embedding=embedding,
            )
        )
        session.commit()


def user_id(email):
    with SessionLocal() as session:
        return session.execute(select(User.id).where(User.email == email)).scalar_one()


def indexing_status_of(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(Document.indexing_status).where(Document.id == document_id)
        ).scalar_one()


def row_snapshot(document_id):
    """Every indexing column plus xmin, which changes whenever the row is rewritten."""
    with SessionLocal() as session:
        row = session.execute(
            text(
                "SELECT xmin::text, processing_status, indexing_status, indexing_error, "
                "indexing_attempt, indexing_lease_expires_at, source_version, "
                "source_sha256, indexed_source_version, indexed_signature, "
                "indexed_embedding_id, indexed_at FROM documents WHERE id = :id"
            ),
            {"id": document_id},
        ).one()
    return tuple(row)


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def expected(document_id, **overrides):
    """The never-queued response, without indexed_at (checked separately)."""
    body = {
        "document_id": document_id,
        "indexing_status": None,
        "indexing_error": None,
        "indexing_attempt": 0,
        "index_current": False,
        "published_chunks_present": False,
        "in_progress": False,
    }
    body.update(overrides)
    return body


def assert_status(body, document_id, *, indexed_at=None, **fields):
    """Compare a response with the expected fields; indexed_at compares as a datetime."""
    body = dict(body)
    published = body.pop("indexed_at")
    if indexed_at is None:
        assert published is None
    else:
        assert parse_time(published) == indexed_at
    assert body == expected(document_id, **fields)


@contextmanager
def capture_sql():
    """Collect every statement, bound value and commit the engine sees."""
    seen = SimpleNamespace(statements=[], bound_values=[], commits=0)

    def before_execute(connection, cursor, statement, parameters, context, executemany):
        seen.statements.append(statement)
        values = parameters.values() if isinstance(parameters, dict) else parameters
        seen.bound_values.extend(values or ())

    def on_commit(connection):
        seen.commits += 1

    event.listen(engine, "before_cursor_execute", before_execute)
    event.listen(engine, "commit", on_commit)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", before_execute)
        event.remove(engine, "commit", on_commit)


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def no_indexing_machinery(monkeypatch):
    """GET must never reach the runner, the embedder or any lifecycle write.

    Only references used by the hook and the runner are replaced. request_indexing in
    indexing_service stays real: the lease tests call it directly to compare results.
    """
    calls = []

    def forbidden(name):
        def refuse(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called by the status endpoint")
        return refuse

    monkeypatch.setattr(runner, "get_embedder", forbidden("runner.get_embedder"))
    monkeypatch.setattr(runner, "run_indexing", forbidden("runner.run_indexing"))
    monkeypatch.setattr(hook, "run_indexing", forbidden("hook.run_indexing"))
    monkeypatch.setattr(hook, "request_indexing", forbidden("hook.request_indexing"))
    for name in ("claim_document", "publish_build", "record_failure"):
        monkeypatch.setattr(svc, name, forbidden(f"indexing_service.{name}"))
    yield
    assert calls == []


@pytest.fixture
def rate_limits_on():
    previous = limiter.enabled
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.reset()
    limiter.enabled = previous


# ---------- 1. response contract ----------

def test_a_never_queued_document_has_null_fields_and_false_booleans(client, auth_headers):
    doc = create_document(client, auth_headers)

    response = get_status(client, auth_headers, doc)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": doc,
        "indexing_status": None,
        "indexing_error": None,
        "indexing_attempt": 0,
        "indexed_at": None,
        "index_current": False,
        "published_chunks_present": False,
        "in_progress": False,
    }


def test_field_types_and_datetime_serialization(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_state(doc, **PUBLICATION, indexing_attempt=1)
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert set(body) == FIELDS
    assert type(body["document_id"]) is int and body["document_id"] == doc
    assert type(body["indexing_attempt"]) is int
    assert type(body["indexing_status"]) is str
    for name in ("index_current", "published_chunks_present", "in_progress"):
        assert type(body[name]) is bool, name
    published = parse_time(body["indexed_at"])
    assert published.tzinfo is not None
    assert published == PUBLISHED_AT


# ---------- 2. indexing lifecycle states ----------

def test_a_pending_document(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_state(doc, processing_status="completed", indexing_status="pending")

    body = get_status(client, auth_headers, doc).json()

    assert_status(body, doc, indexing_status="pending")


def test_an_indexing_document_with_a_live_lease_is_in_progress(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_lease(doc, LIVE_SECONDS, attempt=1)

    body = get_status(client, auth_headers, doc).json()

    assert_status(body, doc, indexing_status="indexing", indexing_attempt=1, in_progress=True)


def test_an_indexing_document_with_an_expired_lease_is_not_in_progress(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_lease(doc, EXPIRED_SECONDS, attempt=2)

    body = get_status(client, auth_headers, doc).json()

    # The stored status is still 'indexing': only in_progress knows the lease ended.
    assert_status(body, doc, indexing_status="indexing", indexing_attempt=2, in_progress=False)


def test_a_failed_document_reports_the_stored_error(client, auth_headers):
    doc = create_document(client, auth_headers)
    reason = svc.FailureReason.EMBEDDER_UNAVAILABLE.value
    set_state(
        doc, processing_status="completed", indexing_status="failed",
        indexing_error=reason, indexing_attempt=2,
    )

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexing_status="failed", indexing_error=reason, indexing_attempt=2
    )


def test_an_indexed_document_with_chunks_is_current_and_has_chunks(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_state(doc, **PUBLICATION, indexing_attempt=1)
    for index in range(3):  # several chunks must still give one plain boolean
        add_chunk(doc, chunk_index=index)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="indexed", indexing_attempt=1,
        index_current=True, published_chunks_present=True,
    )


def test_an_indexed_document_can_carry_an_indexing_error(client, auth_headers):
    # The state record_failure leaves when an earlier publication is restored.
    doc = create_document(client, auth_headers)
    reason = svc.FailureReason.DATABASE.value
    set_state(doc, **PUBLICATION, indexing_attempt=2, indexing_error=reason)
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="indexed", indexing_attempt=2,
        indexing_error=reason, index_current=True, published_chunks_present=True,
    )


def test_indexed_metadata_without_chunk_rows(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_state(doc, **PUBLICATION, indexing_attempt=1)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="indexed", indexing_attempt=1,
        index_current=True, published_chunks_present=False,
    )


def test_chunk_rows_without_publication_metadata(client, auth_headers):
    doc = create_document(client, auth_headers)
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert_status(body, doc, index_current=False, published_chunks_present=True)


def test_a_source_version_mismatch_is_not_current(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_state(doc, **PUBLICATION, indexing_attempt=1, source_version=2)
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="indexed", indexing_attempt=1,
        index_current=False, published_chunks_present=True,
    )


def test_chunks_of_another_document_do_not_count(client, auth_headers):
    with_chunk = create_document(client, auth_headers, "a.txt")
    without_chunk = create_document(client, auth_headers, "b.txt")
    add_chunk(with_chunk)

    assert get_status(client, auth_headers, with_chunk).json()["published_chunks_present"] is True
    assert get_status(client, auth_headers, without_chunk).json()["published_chunks_present"] is False

def test_a_requeued_document_keeps_its_earlier_publication(client, auth_headers):
    # What request_indexing leaves on an indexed document: status 'pending', lease
    # NULL, and the earlier publication and its chunks untouched.
    doc = create_document(client, auth_headers)
    set_state(
        doc, **{**PUBLICATION, "indexing_status": "pending"},
        processing_status="completed", indexing_attempt=1,
    )
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="pending", indexing_attempt=1,
        index_current=True, published_chunks_present=True, in_progress=False,
    )


def test_a_reindex_in_progress_keeps_its_earlier_publication(client, auth_headers):
    # What claim_document leaves on a re-queued indexed document: 'indexing' with a
    # live lease, attempt incremented, and the earlier publication still in place.
    doc = create_document(client, auth_headers)
    mark_extracted(doc)
    set_state(doc, **PUBLICATION, indexing_attempt=1)
    set_lease(doc, LIVE_SECONDS, attempt=2)
    add_chunk(doc)

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexed_at=PUBLISHED_AT, indexing_status="indexing", indexing_attempt=2,
        index_current=True, published_chunks_present=True, in_progress=True,
    )


def test_a_requeued_failed_document_keeps_its_earlier_error(client, auth_headers):
    # request_indexing never clears indexing_error, so 'pending' can still carry it.
    doc = create_document(client, auth_headers)
    reason = svc.FailureReason.EMBEDDER_UNAVAILABLE.value
    set_state(
        doc, processing_status="completed", indexing_status="pending",
        indexing_error=reason, indexing_attempt=1,
    )

    body = get_status(client, auth_headers, doc).json()

    assert_status(
        body, doc, indexing_status="pending", indexing_error=reason, indexing_attempt=1,
        index_current=False, published_chunks_present=False, in_progress=False,
    )

# ---------- 3. ownership and authentication ----------

def test_a_missing_document_returns_404(client, auth_headers):
    response = get_status(client, auth_headers, 999999)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_a_foreign_document_is_indistinguishable_from_a_missing_one(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_document(client, alice)
    reason = svc.FailureReason.DATABASE.value
    set_state(doc, **PUBLICATION, indexing_error=reason)  # state that must not leak
    add_chunk(doc)

    foreign = get_status(client, bob, doc)
    missing = get_status(client, bob, 999999)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json() == NOT_FOUND
    assert reason not in foreign.text
    assert get_status(client, alice, doc).status_code == 200


def test_a_missing_token_returns_401(client, auth_headers):
    doc = create_document(client, auth_headers)

    response = client.get(f"{DOCUMENTS}/{doc}/indexing")

    assert response.status_code == 401


def test_an_invalid_token_returns_401(client, auth_headers):
    doc = create_document(client, auth_headers)

    response = client.get(
        f"{DOCUMENTS}/{doc}/indexing", headers={"Authorization": "Bearer not-a-real-token"}
    )

    assert response.status_code == 401


def test_a_non_integer_document_id_returns_422(client, auth_headers):
    response = client.get(f"{DOCUMENTS}/abc/indexing", headers=auth_headers)

    assert response.status_code == 422


def test_the_snapshot_query_enforces_ownership_on_its_own(client, make_user, monkeypatch):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_document(client, alice)
    set_state(doc, **PUBLICATION)
    # Let the ownership gate wave everyone through: only the snapshot's own owner
    # filter can now keep Bob out of Alice's document.
    monkeypatch.setattr(
        indexing_api.crud, "get_document", lambda db, document_id, owner_id: object()
    )

    response = get_status(client, bob, doc)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_an_ownership_change_between_the_gate_and_the_snapshot_returns_404(
    client, make_user, monkeypatch
):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    doc = create_document(client, alice)
    bob_id = user_id("bob@example.com")
    real_gate = indexing_api.crud.get_document

    def gate_then_reassign(db, document_id, owner_id):
        found = real_gate(db, document_id, owner_id=owner_id)
        execute("UPDATE documents SET owner_id = :owner WHERE id = :id", owner=bob_id, id=document_id)
        return found

    monkeypatch.setattr(indexing_api.crud, "get_document", gate_then_reassign)

    response = get_status(client, alice, doc)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_a_deletion_between_the_gate_and_the_snapshot_returns_404(
    client, auth_headers, monkeypatch
):
    doc = create_document(client, auth_headers)
    real_gate = indexing_api.crud.get_document

    def gate_then_delete(db, document_id, owner_id):
        found = real_gate(db, document_id, owner_id=owner_id)
        execute("DELETE FROM documents WHERE id = :id", id=document_id)
        return found

    monkeypatch.setattr(indexing_api.crud, "get_document", gate_then_delete)

    response = get_status(client, auth_headers, doc)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


# ---------- 4. database clock and lease semantics ----------

def test_a_live_lease_is_in_progress_and_request_indexing_refuses_it(client, auth_headers):
    doc = create_document(client, auth_headers)
    mark_extracted(doc)
    set_lease(doc, LIVE_SECONDS)

    in_progress = get_status(client, auth_headers, doc).json()["in_progress"]
    with SessionLocal() as session:
        queued = svc.request_indexing(session, doc)

    assert (in_progress, queued) == (True, False)
    assert indexing_status_of(doc) == "indexing"


def test_an_expired_lease_is_not_in_progress_and_request_indexing_takes_it_over(
    client, auth_headers
):
    doc = create_document(client, auth_headers)
    mark_extracted(doc)
    set_lease(doc, EXPIRED_SECONDS)

    in_progress = get_status(client, auth_headers, doc).json()["in_progress"]
    with SessionLocal() as session:
        queued = svc.request_indexing(session, doc)

    assert (in_progress, queued) == (False, True)
    assert indexing_status_of(doc) == "pending"


def test_the_lease_is_compared_with_the_database_clock(client, auth_headers):
    doc = create_document(client, auth_headers)
    set_lease(doc, LIVE_SECONDS)

    with capture_sql() as seen:
        response = get_status(client, auth_headers, doc)

    assert response.json()["in_progress"] is True
    snapshots = [s for s in seen.statements if "document_chunks" in s]
    assert len(snapshots) == 1
    assert "clock_timestamp()" in snapshots[0].lower()
    # No Python-side time travels to the database as a bound value.
    assert not any(isinstance(value, datetime) for value in seen.bound_values)


def test_the_lease_boundary_matches_request_indexing_expiry(client, auth_headers):
    # request_indexing treats `lease <= clock_timestamp()` as expired, so a live
    # lease must be the strict `lease > clock_timestamp()`. An exact tie cannot be
    # produced against a moving clock, so the rendered comparison is checked instead.
    doc = create_document(client, auth_headers)

    with capture_sql() as seen:
        get_status(client, auth_headers, doc)

    snapshot = next(s for s in seen.statements if "document_chunks" in s).lower()
    assert re.search(
        r"(?:\w+\.)?indexing_lease_expires_at\s*>\s*clock_timestamp\(\)", snapshot
    )
    assert not re.search(r"indexing_lease_expires_at\s*<=", snapshot)


# ---------- 5. read-only behavior ----------

def test_get_leaves_every_indexing_column_unchanged(client, auth_headers):
    pending = create_document(client, auth_headers, "pending.txt")
    set_state(pending, processing_status="completed", indexing_status="pending")
    live = create_document(client, auth_headers, "live.txt")
    mark_extracted(live)
    set_lease(live, LIVE_SECONDS)
    expired = create_document(client, auth_headers, "expired.txt")
    mark_extracted(expired)
    set_lease(expired, EXPIRED_SECONDS, attempt=3)  # the state a careless GET might "recover"
    failed = create_document(client, auth_headers, "failed.txt")
    set_state(
        failed, processing_status="completed", indexing_status="failed",
        indexing_error=svc.FailureReason.UNEXPECTED.value, indexing_attempt=1,
    )
    documents = [pending, live, expired, failed]
    before = {doc: row_snapshot(doc) for doc in documents}

    for doc in documents:
        assert get_status(client, auth_headers, doc).status_code == 200

    assert {doc: row_snapshot(doc) for doc in documents} == before

def test_capture_sql_records_statements_and_commits():
    # Positive control: the read-only test's `commits == 0` only means something if
    # this listener is known to count a real commit. No application table is touched.
    with capture_sql() as seen:
        with engine.begin() as connection:
            connection.execute(text("SELECT 1"))

    assert seen.commits == 1
    assert any("SELECT 1" in statement for statement in seen.statements)

def test_get_issues_only_selects_and_never_commits(client, auth_headers):
    doc = create_document(client, auth_headers)
    mark_extracted(doc)
    set_lease(doc, EXPIRED_SECONDS)

    with capture_sql() as seen:
        response = get_status(client, auth_headers, doc)

    assert response.status_code == 200
    assert seen.statements
    for statement in seen.statements:
        assert statement.lstrip().upper().startswith("SELECT"), statement
        assert not re.search(r"\bfor\s+(no\s+key\s+)?update\b", statement, re.IGNORECASE)
    assert seen.commits == 0


# ---------- 6. operational behavior ----------

def test_the_status_endpoint_is_not_rate_limited(client, auth_headers, rate_limits_on):
    doc = create_document(client, auth_headers)

    codes = {get_status(client, auth_headers, doc).status_code for _ in range(20)}

    assert codes == {200}


def test_database_connections_return_to_the_pool(client, auth_headers):
    doc = create_document(client, auth_headers)
    assert get_status(client, auth_headers, doc).status_code == 200  # warm-up
    baseline = engine.pool.checkedout()

    for _ in range(5):
        assert get_status(client, auth_headers, doc).status_code == 200
    assert get_status(client, auth_headers, 999999).status_code == 404
    assert client.get(f"{DOCUMENTS}/{doc}/indexing").status_code == 401

    assert engine.pool.checkedout() == baseline
