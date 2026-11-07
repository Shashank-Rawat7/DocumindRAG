"""C5.4.2 tests: POST /api/v1/documents/{document_id}/indexing.

Runs against documind_test. States are written straight to the database, honoring the
table's check constraints; lease times are computed by PostgreSQL. The endpoint's
background task is replaced by a strict recorder, so no test indexes anything unless it
says so, and the real embedding model is never loaded. Helpers are local to this file
(tests/ has no shared helper module).
"""
import hashlib
import logging
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update
from sqlalchemy.exc import OperationalError

from app.api import indexing as indexing_api
from app.core.database import SessionLocal, engine
from app.core.rate_limit import limiter
from app.main import app
from app.models.chunk import EMBEDDING_DIM, DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import indexing_hook as hook
from app.rag import indexing_runner as runner
from app.rag import indexing_service as svc
from app.rag.embedder import EmbedderUnavailableError
from app.rag.text_boundaries import is_ignorable
from app.rag.token_offsets import map_token_offsets

DOCUMENTS = "/api/v1/documents"

FIELDS = {
    "document_id", "indexing_status", "indexing_error", "indexing_attempt",
    "indexed_at", "index_current", "published_chunks_present", "in_progress",
}
NOT_FOUND = {"detail": "Document not found"}
EXTRACTION_UNFINISHED = "Text extraction has not finished yet"
EXTRACTION_FAILED = "Text extraction failed for this document"
NO_EXTRACTED_TEXT = "This document has no extracted text"
STATE_CHANGED = "Indexing state changed; please retry"

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
TEXT = " ".join(f"word{k:04d}" for k in range(700))  # several default-size chunks

# Positions in the tuple returned by row_snapshot().
S_PROCESSING, S_STATUS, S_ERROR, S_ATTEMPT, S_LEASE = 1, 2, 3, 4, 5
S_PUBLICATION = slice(6, None)  # source_version and every indexed_* column


# ---------- helpers ----------

def create_document(client, headers, filename="notes.txt"):
    response = client.post(DOCUMENTS, json={"filename": filename}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def post_indexing(client, headers, document_id):
    return client.post(f"{DOCUMENTS}/{document_id}/indexing", headers=headers)


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


def ready_document(client, headers, extracted_text=None):
    """A document whose extraction completed and that was never queued for indexing."""
    document_id = create_document(client, headers)
    if extracted_text is None:
        mark_extracted(document_id)
    else:
        set_state(
            document_id, processing_status="completed",
            extracted_text=extracted_text, stored_name="notes.txt",
        )
    return document_id


def publish(document_id, *, chunk=True, **overrides):
    """A fully current publication (plus one chunk row unless chunk=False)."""
    set_state(document_id, processing_status="completed", **{**PUBLICATION, "indexing_attempt": 1, **overrides})
    if chunk:
        add_chunk(document_id)


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
    """The never-queued status body, without indexed_at (checked separately)."""
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
    """Compare a status body with the expected fields; indexed_at compares as a datetime."""
    body = dict(body)
    published = body.pop("indexed_at")
    if indexed_at is None:
        assert published is None
    else:
        assert parse_time(published) == indexed_at
    assert body == expected(document_id, **fields)


def delete_or_reassign(kind, document_id, other_owner_id):
    if kind == "delete":
        execute("DELETE FROM documents WHERE id = :id", id=document_id)
    else:
        execute(
            "UPDATE documents SET owner_id = :owner WHERE id = :id",
            owner=other_owner_id, id=document_id,
        )


_REAL_REQUEST = indexing_api._request_in_fresh_session


def intercept_request(monkeypatch, *, before=None, after=None, call_real=True, result=False):
    """Wrap indexing_api._request_in_fresh_session.

    `before` and `after` run around the call with the document id. With call_real the
    real helper runs (so a refusal or acceptance is genuine) and its answer is returned;
    without it `result` is returned and the database is never asked. Returns the list of
    answers given, one per call.
    """
    answers = []

    def wrapper(document_id):
        if before is not None:
            before(document_id)
        answer = _REAL_REQUEST(document_id) if call_real else result
        answers.append(answer)
        if after is not None:
            after(document_id)
        return answer

    monkeypatch.setattr(indexing_api, "_request_in_fresh_session", wrapper)
    return answers


def forbid_runner(monkeypatch):
    """Any attempt to run the indexer through the runner or the hook is recorded and fails."""
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the indexer must not run in this test")

    monkeypatch.setattr(runner, "run_indexing", refuse)
    monkeypatch.setattr(hook, "run_indexing", refuse)
    return calls


def db_error():
    return OperationalError("SELECT secret_sql", {"p": "secret-param"}, Exception("secret-driver-message"))


# ---------- fake embedder (same behaviour as the hook tests' fake, trimmed) ----------

def word_spans(value):
    spans, start = [], None
    for position, character in enumerate(value + " "):
        if is_ignorable(character):
            if start is not None:
                spans.append((start, position))
                start = None
        elif start is None:
            start = position
    return spans


def fake_vector(value):
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big") % EMBEDDING_DIM] = 1.0
    return vector


class FakeEmbedder:
    embedding_id = "test/fake@" + "0" * 40

    def __init__(self, *, embed_error=None):
        self.embed_error = embed_error
        self.calls = []

    def signature_fields(self):
        self.calls.append("signature_fields")
        return {"model_id": "test/fake", "embedding_dim": EMBEDDING_DIM}

    def source_units(self, value):
        self.calls.append("source_units")
        spans = word_spans(value)
        return map_token_offsets(value, spans, [value[start:end] for start, end in spans])

    def embed_passages(self, texts):
        self.calls.append("embed_passages")
        if self.embed_error is not None:
            raise self.embed_error
        return np.stack([fake_vector(item) for item in texts]).astype(np.float32)


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def scheduled(monkeypatch):
    """Strict stand-in for the endpoint's background task: records the id, never indexes."""
    calls = []

    def record(document_id):  # a single argument: any extra one would fail the call
        calls.append(document_id)

    monkeypatch.setattr(indexing_api, "run_requested_indexing", record)
    return calls


@pytest.fixture(autouse=True)
def forbid_real_embedder(monkeypatch):
    """The runner may only get an embedder the test hands it."""
    requested = []

    def forbidden():
        requested.append(True)
        raise AssertionError("the real embedder must not be requested in tests")

    monkeypatch.setattr(runner, "get_embedder", forbidden)
    yield
    assert requested == []


@pytest.fixture
def rate_limits_on():
    previous = limiter.enabled
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.reset()
    limiter.enabled = previous


@pytest.fixture
def lenient_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def embedder(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(runner, "get_embedder", lambda: fake)
    return fake


class SessionTracker:
    """Wraps indexing_api.SessionLocal: records every fresh session and its close."""

    def __init__(self, monkeypatch, *, close_error=None):
        self.records = []
        self.baseline = engine.pool.checkedout()
        real_factory = indexing_api.SessionLocal

        def factory():
            session = real_factory()
            record = {"closed": False}
            real_close = session.close

            def close():
                real_close()
                record["closed"] = True
                if close_error is not None:
                    raise close_error

            session.close = close
            self.records.append(record)
            return session

        monkeypatch.setattr(indexing_api, "SessionLocal", factory)


# ======================================================================
# D2a: contract, authentication, ownership, extraction state, lifecycle states
# ======================================================================

def test_a_queued_request_returns_202_with_the_approved_body(client, auth_headers, scheduled):
    doc = ready_document(client, auth_headers)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"outcome", "indexing"}
    assert body["outcome"] == "queued"
    assert set(body["indexing"]) == FIELDS
    assert_status(body["indexing"], doc, indexing_status="pending")
    assert body["indexing"] == get_status(client, auth_headers, doc).json()
    assert scheduled == [doc]


def test_authentication_and_path_validation_errors(client, auth_headers, scheduled):
    doc = ready_document(client, auth_headers)
    before = row_snapshot(doc)

    no_token = client.post(f"{DOCUMENTS}/{doc}/indexing")
    bad_token = client.post(
        f"{DOCUMENTS}/{doc}/indexing", headers={"Authorization": "Bearer not-a-real-token"}
    )
    not_an_integer = post_indexing(client, auth_headers, "abc")

    assert no_token.status_code == 401
    assert bad_token.status_code == 401
    assert not_an_integer.status_code == 422
    assert scheduled == []
    assert row_snapshot(doc) == before


def test_a_missing_document_returns_404(client, auth_headers, scheduled):
    response = post_indexing(client, auth_headers, 999999)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert scheduled == []


def test_a_foreign_document_is_indistinguishable_from_a_missing_one(
    client, make_user, scheduled
):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = ready_document(client, alice)
    before = row_snapshot(doc)

    foreign = post_indexing(client, bob, doc)
    missing = post_indexing(client, bob, 999999)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json() == NOT_FOUND
    assert scheduled == []
    assert row_snapshot(doc) == before  # xmin included: the row was not even rewritten


def test_ownership_is_checked_before_the_extraction_state(client, make_user, scheduled):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_document(client, alice)
    set_state(doc, processing_status="failed")

    response = post_indexing(client, bob, doc)

    assert response.status_code == 404  # not 409: Bob must not learn the extraction state
    assert response.json() == NOT_FOUND


@pytest.mark.parametrize(
    "processing_status, message",
    [
        ("pending", EXTRACTION_UNFINISHED),
        ("processing", EXTRACTION_UNFINISHED),
        ("failed", EXTRACTION_FAILED),
        (None, NO_EXTRACTED_TEXT),
    ],
    ids=["pending", "processing", "failed", "null"],
)
def test_extraction_not_completed_returns_the_approved_409(
    client, auth_headers, scheduled, processing_status, message
):
    doc = create_document(client, auth_headers)
    set_state(doc, processing_status=processing_status)
    before = row_snapshot(doc)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 409
    assert response.json() == {"detail": message}
    assert scheduled == []
    assert row_snapshot(doc) == before


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"indexing_status": "pending"},
        {
            "indexing_status": "failed", "indexing_attempt": 1,
            "indexing_error": svc.FailureReason.EMBEDDER_UNAVAILABLE.value,
        },
    ],
    ids=["never-queued", "pending", "failed"],
)
def test_never_queued_pending_and_failed_documents_are_queued(
    client, auth_headers, scheduled, state
):
    doc = ready_document(client, auth_headers)
    if state:  # an empty UPDATE ... SET would be invalid
        set_state(doc, **state)
    before = row_snapshot(doc)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert response.json()["outcome"] == "queued"
    after = row_snapshot(doc)
    assert after[S_STATUS] == "pending" and after[S_LEASE] is None
    assert after[S_ERROR] == before[S_ERROR]      # request_indexing never touches the error
    assert after[S_ATTEMPT] == before[S_ATTEMPT]  # nor the attempt counter
    assert response.json()["indexing"]["indexing_error"] == before[S_ERROR]
    assert scheduled == [doc]


def test_an_expired_lease_is_queued_again(client, auth_headers, scheduled):
    doc = ready_document(client, auth_headers)
    set_lease(doc, EXPIRED_SECONDS, attempt=2)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert response.json()["outcome"] == "queued"
    after = row_snapshot(doc)
    assert after[S_STATUS] == "pending" and after[S_LEASE] is None and after[S_ATTEMPT] == 2
    assert response.json()["indexing"]["in_progress"] is False
    assert scheduled == [doc]


@pytest.mark.parametrize(
    "variant",
    ["with-error", "without-chunks", "source-version-mismatch"],
)
def test_an_indexed_document_that_is_not_fully_current_is_queued(
    client, auth_headers, scheduled, variant
):
    doc = create_document(client, auth_headers)
    if variant == "with-error":
        publish(doc, indexing_error=svc.FailureReason.DATABASE.value)
    elif variant == "without-chunks":
        publish(doc, chunk=False)
    else:
        publish(doc, source_version=2)
    before = row_snapshot(doc)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert response.json()["outcome"] == "queued"
    after = row_snapshot(doc)
    assert after[S_STATUS] == "pending"
    assert after[S_ERROR] == before[S_ERROR]
    assert after[S_PUBLICATION] == before[S_PUBLICATION]  # the published data stays untouched
    assert scheduled == [doc]


def test_an_already_current_document_returns_200_and_writes_nothing(
    client, auth_headers, scheduled
):
    doc = create_document(client, auth_headers)
    publish(doc)
    before = row_snapshot(doc)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "already_indexed"
    assert_status(
        body["indexing"], doc, indexed_at=PUBLISHED_AT, indexing_status="indexed",
        indexing_attempt=1, index_current=True, published_chunks_present=True,
    )
    assert scheduled == []
    assert row_snapshot(doc) == before


def test_a_live_lease_returns_already_running_and_writes_nothing(
    client, auth_headers, scheduled
):
    doc = ready_document(client, auth_headers)
    set_lease(doc, LIVE_SECONDS, attempt=1)
    before = row_snapshot(doc)

    responses = [post_indexing(client, auth_headers, doc) for _ in range(2)]

    for response in responses:
        assert response.status_code == 202
        body = response.json()
        assert body["outcome"] == "already_running"
        assert_status(
            body["indexing"], doc, indexing_status="indexing", indexing_attempt=1,
            in_progress=True,
        )
    assert scheduled == []
    assert row_snapshot(doc) == before


# ======================================================================
# D2b: refusal races, ordering, sessions, errors, helper unit tests, duplicates (16 functions)
# ======================================================================

def test_a_refusal_explained_by_a_live_lease_returns_already_running(
    client, auth_headers, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)
    answers = intercept_request(
        monkeypatch, call_real=True, result=False,
        before=lambda document_id: set_lease(document_id, LIVE_SECONDS),
    )

    response = post_indexing(client, auth_headers, doc)

    assert answers == [False]
    assert response.status_code == 202
    body = response.json()
    assert body["outcome"] == "already_running"
    assert body["indexing"]["in_progress"] is True
    assert scheduled == []



def test_a_publication_racing_with_request_may_be_requeued(
    client, auth_headers, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)
    published = []

    def publish_before_request(document_id):
        publish(document_id)
        published.append(row_snapshot(document_id))

    answers = intercept_request(
        monkeypatch,
        call_real=True,
        before=publish_before_request,
    )

    response = post_indexing(client, auth_headers, doc)

    assert answers == [True]
    assert response.status_code == 202
    assert response.json()["outcome"] == "queued"
    assert scheduled == [doc]

    after = row_snapshot(doc)
    assert after[S_STATUS] == "pending"
    assert after[S_PUBLICATION] == published[0][S_PUBLICATION]


def test_an_unexplained_refusal_returns_the_fallback_409(
    client, auth_headers, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)
    before = row_snapshot(doc)
    answers = intercept_request(monkeypatch, call_real=False, result=False)

    response = post_indexing(client, auth_headers, doc)

    assert answers == [False]
    assert response.status_code == 409
    assert response.json() == {"detail": STATE_CHANGED}
    assert scheduled == []
    assert row_snapshot(doc) == before


def test_extraction_failing_after_the_first_check_returns_the_failed_409(
    client, auth_headers, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)
    runner_calls = forbid_runner(monkeypatch)
    # Extraction fails after the endpoint's first check; the REAL request then refuses.
    answers = intercept_request(
        monkeypatch, call_real=True,
        before=lambda document_id: set_state(document_id, processing_status="failed"),
    )

    response = post_indexing(client, auth_headers, doc)

    assert answers == [False]  # a genuine refusal, not a stub
    assert response.status_code == 409
    assert response.json() == {"detail": EXTRACTION_FAILED}
    assert scheduled == []
    assert runner_calls == []
    assert indexing_status_of(doc) is None
    assert row_snapshot(doc)[S_PROCESSING] == "failed"


def test_a_refusal_after_a_vanished_document_returns_404(
    client, auth_headers, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)
    answers = intercept_request(
        monkeypatch, call_real=True,
        before=lambda document_id: execute("DELETE FROM documents WHERE id = :id", id=document_id),
    )

    response = post_indexing(client, auth_headers, doc)

    assert answers == [False]
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert scheduled == []


@pytest.mark.parametrize("kind", ["delete", "reassign"])
def test_a_document_deleted_or_reassigned_after_the_request_schedules_nothing(
    client, make_user, monkeypatch, scheduled, kind
):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    bob_id = user_id("bob@example.com")
    doc = ready_document(client, alice)
    # The real request succeeds; the document then vanishes before the fresh snapshot.
    answers = intercept_request(
        monkeypatch, call_real=True,
        after=lambda document_id: delete_or_reassign(kind, document_id, bob_id),
    )

    response = post_indexing(client, alice, doc)

    assert answers == [True]
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert scheduled == []  # the approved ordering: no task before the fresh snapshot passes


@pytest.mark.parametrize("kind", ["delete", "reassign"])
def test_a_change_between_the_gate_and_the_pre_read_returns_404_without_a_request(
    client, make_user, monkeypatch, scheduled, kind
):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    bob_id = user_id("bob@example.com")
    doc = ready_document(client, alice)
    real_gate = indexing_api.crud.get_document

    def gate_then_change(db, document_id, owner_id):
        found = real_gate(db, document_id, owner_id=owner_id)
        delete_or_reassign(kind, document_id, bob_id)
        return found

    monkeypatch.setattr(indexing_api.crud, "get_document", gate_then_change)
    requests = []
    monkeypatch.setattr(
        indexing_api, "_request_in_fresh_session",
        lambda document_id: requests.append(document_id) or True,
    )

    response = post_indexing(client, alice, doc)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert requests == []
    assert scheduled == []


def test_the_task_is_registered_only_after_the_fresh_snapshot(
    client, auth_headers, monkeypatch
):
    doc = ready_document(client, auth_headers)
    events = []
    real_request = indexing_api._request_in_fresh_session
    real_fresh = indexing_api._fresh_status
    real_add_task = BackgroundTasks.add_task

    def request_step(document_id):
        events.append("request")
        return real_request(document_id)

    def fresh_step(document_id, owner_id):
        events.append("fresh_status")
        return real_fresh(document_id, owner_id)

    def add_task(self, func, *args, **kwargs):
        events.append("schedule")
        return real_add_task(self, func, *args, **kwargs)

    monkeypatch.setattr(indexing_api, "_request_in_fresh_session", request_step)
    monkeypatch.setattr(indexing_api, "_fresh_status", fresh_step)
    monkeypatch.setattr(BackgroundTasks, "add_task", add_task)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert events == ["request", "fresh_status", "schedule"]


def test_fresh_sessions_have_no_open_transaction_and_are_closed(
    client, auth_headers, monkeypatch
):
    queued_doc = ready_document(client, auth_headers)
    refused_doc = ready_document(client, auth_headers)
    tracker = SessionTracker(monkeypatch)
    in_transaction = []
    real_request = indexing_api.request_indexing

    def request_step(session, document_id):
        in_transaction.append(session.in_transaction())
        if document_id == refused_doc:
            set_lease(document_id, LIVE_SECONDS)  # a live lease appears: a genuine refusal
        return real_request(session, document_id)

    monkeypatch.setattr(indexing_api, "request_indexing", request_step)

    queued = post_indexing(client, auth_headers, queued_doc)
    assert queued.json()["outcome"] == "queued"
    assert len(tracker.records) == 2  # request_indexing + the status snapshot
    refused = post_indexing(client, auth_headers, refused_doc)
    assert refused.json()["outcome"] == "already_running"
    assert len(tracker.records) == 5  # + request_indexing, status snapshot, extraction recheck

    assert in_transaction == [False, False]  # request_indexing always got a fresh session
    assert all(record["closed"] for record in tracker.records)
    assert engine.pool.checkedout() == tracker.baseline


def test_a_session_close_failure_keeps_the_result(
    client, auth_headers, monkeypatch, scheduled, caplog
):
    caplog.set_level(logging.DEBUG, logger="documind")
    doc = ready_document(client, auth_headers)
    tracker = SessionTracker(monkeypatch, close_error=RuntimeError("secret-cleanup"))

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert response.json()["outcome"] == "queued"
    assert scheduled == [doc]
    warnings = [r for r in caplog.records if r.getMessage().startswith("Session cleanup failed after")]
    assert len(warnings) == len(tracker.records) == 2
    assert all(r.levelno == logging.WARNING for r in warnings)
    assert all("error_type=RuntimeError" in r.getMessage() for r in warnings)
    assert "secret-cleanup" not in caplog.text
    assert all(r.exc_info is None for r in warnings)


def test_a_request_indexing_error_returns_a_generic_500_and_schedules_nothing(
    client, auth_headers, lenient_client, monkeypatch, scheduled
):
    doc = ready_document(client, auth_headers)

    def explode(session, document_id):
        raise db_error()

    monkeypatch.setattr(indexing_api, "request_indexing", explode)

    response = lenient_client.post(f"{DOCUMENTS}/{doc}/indexing", headers=auth_headers)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert scheduled == []
    assert indexing_status_of(doc) is None


@pytest.mark.parametrize(
    "processing_status, message",
    [
        (None, NO_EXTRACTED_TEXT),
        ("pending", EXTRACTION_UNFINISHED),
        ("processing", EXTRACTION_UNFINISHED),
        ("failed", EXTRACTION_FAILED),
    ],
    ids=["null", "pending", "processing", "failed"],
)
def test_conflict_detail_maps_every_extraction_conflict(processing_status, message):
    assert indexing_api._conflict_detail(processing_status) == message


@pytest.mark.parametrize("value", ["completed", "unknown"], ids=["completed", "invalid"])
def test_conflict_detail_rejects_states_that_are_not_conflicts(value):
    with pytest.raises(ValueError):
        indexing_api._conflict_detail(value)


def test_fresh_processing_status_distinguishes_missing_from_null(client, make_user):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    alice_id, bob_id = user_id("alice@example.com"), user_id("bob@example.com")
    doc = create_document(client, alice)

    assert indexing_api._fresh_processing_status(999999, alice_id) is None  # missing
    assert indexing_api._fresh_processing_status(doc, bob_id) is None       # foreign

    for stored in (None, "pending", "processing", "failed", "completed"):
        set_state(doc, processing_status=stored)
        row = indexing_api._fresh_processing_status(doc, alice_id)
        assert row is not None
        assert row.processing_status == stored


def test_two_requests_on_a_never_queued_document_both_queue_it(
    client, auth_headers, scheduled
):
    doc = ready_document(client, auth_headers)

    first = post_indexing(client, auth_headers, doc)
    second = post_indexing(client, auth_headers, doc)

    assert first.status_code == second.status_code == 202
    assert first.json()["outcome"] == "queued"
    assert second.json()["outcome"] == "queued"
    # Duplicate tasks are allowed: the runner's conditional claim is the real protection.
    assert scheduled and set(scheduled) == {doc} and len(scheduled) <= 2
    after = row_snapshot(doc)
    assert after[S_STATUS] == "pending" and after[S_ATTEMPT] == 0


def test_a_request_after_a_claim_is_already_running(client, auth_headers, scheduled):
    doc = ready_document(client, auth_headers)
    assert post_indexing(client, auth_headers, doc).json()["outcome"] == "queued"
    set_lease(doc, LIVE_SECONDS, attempt=1)  # what claim_document would do

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    assert response.json()["outcome"] == "already_running"
    assert scheduled == [doc]  # only the first request scheduled anything


# ======================================================================
# D2c: rate limit and end to end
# ======================================================================

def test_the_sixth_post_in_a_minute_is_rate_limited(
    client, auth_headers, rate_limits_on, scheduled
):
    doc = ready_document(client, auth_headers)

    codes = [post_indexing(client, auth_headers, doc).status_code for _ in range(6)]

    assert codes == [202, 202, 202, 202, 202, 429]
    assert scheduled == [doc] * 5  # the limited request never reached the handler


def test_the_post_limit_counts_every_request_that_reaches_the_handler(
    client, auth_headers, rate_limits_on
):
    failed = create_document(client, auth_headers)
    set_state(failed, processing_status="failed")
    current = create_document(client, auth_headers)
    publish(current)
    ready = ready_document(client, auth_headers)

    codes = [
        post_indexing(client, auth_headers, 999999).status_code,   # 404
        post_indexing(client, auth_headers, failed).status_code,   # 409
        post_indexing(client, auth_headers, current).status_code,  # 200
        post_indexing(client, auth_headers, ready).status_code,    # 202
        post_indexing(client, auth_headers, 999999).status_code,   # 404
        post_indexing(client, auth_headers, ready).status_code,    # over the limit
    ]

    assert codes == [404, 409, 200, 202, 404, 429]


def test_exhausting_the_post_limit_does_not_limit_get(client, auth_headers, rate_limits_on):
    doc = ready_document(client, auth_headers)
    codes = [post_indexing(client, auth_headers, doc).status_code for _ in range(6)]
    assert codes[-1] == 429

    get_codes = {get_status(client, auth_headers, doc).status_code for _ in range(20)}

    assert get_codes == {200}


def test_401_and_422_do_not_consume_the_post_limit(client, auth_headers, rate_limits_on):
    doc = ready_document(client, auth_headers)

    for _ in range(7):
        assert client.post(f"{DOCUMENTS}/{doc}/indexing").status_code == 401
    for _ in range(7):
        assert post_indexing(client, auth_headers, "abc").status_code == 422
    codes = [post_indexing(client, auth_headers, doc).status_code for _ in range(6)]

    assert codes == [202, 202, 202, 202, 202, 429]


def test_the_manual_endpoint_runs_the_real_wrapper_and_publishes(
    client, auth_headers, monkeypatch, embedder
):
    monkeypatch.setattr(indexing_api, "run_requested_indexing", hook.run_requested_indexing)
    doc = ready_document(client, auth_headers, extracted_text=TEXT)

    first = post_indexing(client, auth_headers, doc)  # the background task has finished on return

    assert first.status_code == 202
    assert first.json()["outcome"] == "queued"
    status = get_status(client, auth_headers, doc).json()
    assert status["indexing_status"] == "indexed"
    assert status["indexing_attempt"] == 1
    assert status["indexing_error"] is None
    assert status["index_current"] is True
    assert status["published_chunks_present"] is True
    assert status["in_progress"] is False
    assert status["indexed_at"] is not None
    calls_after_first = list(embedder.calls)
    assert calls_after_first
    snapshot = row_snapshot(doc)

    second = post_indexing(client, auth_headers, doc)

    assert second.status_code == 200
    assert second.json()["outcome"] == "already_indexed"
    assert second.json()["indexing"]["indexed_at"] == status["indexed_at"]
    assert embedder.calls == calls_after_first
    assert row_snapshot(doc) == snapshot


def test_a_failing_embedder_is_visible_in_the_status(client, auth_headers, monkeypatch):
    error = EmbedderUnavailableError("secret-detail")
    failing = FakeEmbedder(embed_error=error)
    monkeypatch.setattr(runner, "get_embedder", lambda: failing)
    monkeypatch.setattr(indexing_api, "run_requested_indexing", hook.run_requested_indexing)
    doc = ready_document(client, auth_headers, extracted_text=TEXT)

    response = post_indexing(client, auth_headers, doc)

    assert response.status_code == 202
    status_response = get_status(client, auth_headers, doc)
    status = status_response.json()
    assert status["indexing_status"] == "failed"
    assert status["indexing_attempt"] == 1
    assert status["indexing_error"] == svc.classify_failure(error).value
    assert status["index_current"] is False and status["in_progress"] is False
    assert "secret-detail" not in status_response.text
    assert "secret-detail" not in response.text

def test_the_post_limit_is_shared_across_documents(
    client, auth_headers, rate_limits_on
):
    docs = [ready_document(client, auth_headers) for _ in range(5)]

    codes = [
        post_indexing(client, auth_headers, doc).status_code
        for doc in docs
    ]
    codes.append(post_indexing(client, auth_headers, 999999).status_code)

    assert codes == [202, 202, 202, 202, 202, 429]