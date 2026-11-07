"""C5.3 automatic indexing hook (app/rag/indexing_hook.py).

Two kinds of tests:

* Unit tests replace request_indexing / run_indexing inside the hook module with
  recording fakes: they pin the call order, the logging and the exception policy.
* End-to-end tests run the real hook, the real runner and the real C5.1 functions
  against the test database with a fake embedder.

The real embedding model is never loaded: an autouse fixture makes the runner's
get_embedder() fail the test if it is ever requested without the test patching it.
No network, no sleeps and no background threads are involved: the hook is called
directly, exactly as Starlette would call it after process_document.

Helpers are local to this file (tests/ has no shared helper module). Helper reads and
writes use their own short sessions, so no test leaves a transaction open for the
TRUNCATE in conftest's cleanup fixture.
"""
import hashlib
import logging
import uuid

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import indexing_hook as hook
from app.rag import indexing_runner as runner
from app.rag import indexing_service as svc
from app.rag.embedder import EMBEDDING_DIM, EmbedderUnavailableError
from app.rag.text_boundaries import is_ignorable
from app.rag.token_offsets import map_token_offsets

TEXT = " ".join(f"word{k:04d}" for k in range(700))      # several default-size chunks
RunOutcome = runner.RunOutcome
FINISHED = "Automatic indexing finished"
FAILED_MESSAGE = "Automatic indexing failed"
NOT_QUEUED = "Indexing not queued"
CLEANUP = "Session cleanup failed"


# ---------- fake embedder (same behaviour as the C5.2 tests' fake, trimmed) ----------

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


# ---------- database helpers ----------

def execute(sql, **params):
    with engine.begin() as connection:
        connection.execute(text(sql), params)


def make_document(text_=TEXT, processing_status="completed", stored_name="notes.txt", **columns):
    with SessionLocal() as session:
        user = User(email=f"u{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        document = Document(
            filename="a.txt",
            owner_id=user.id,
            processing_status=processing_status,
            stored_name=stored_name,
            extracted_text=text_,
            **columns,
        )
        session.add(document)
        session.commit()
        return document.id


def row_of(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(
                Document.indexing_status,
                Document.indexing_error,
                Document.indexing_attempt,
                Document.indexing_lease_expires_at,
                Document.indexed_source_version,
                Document.indexed_embedding_id,
            ).where(Document.id == document_id)
        ).one()


def chunk_texts(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(DocumentChunk.text)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).scalars().all()


def delete_document(document_id):
    execute("DELETE FROM documents WHERE id = :id", id=document_id)


# ---------- fixtures and test doubles ----------

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
def embedder(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(runner, "get_embedder", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def capture_logs(caplog):
    caplog.set_level(logging.DEBUG, logger="documind")
    return caplog


def hook_records(caplog, prefix=""):
    return [r for r in caplog.records if r.name == "documind" and r.getMessage().startswith(prefix)]


class SessionTracker:
    """Wraps hook.SessionLocal: records every session and whether it was closed."""

    def __init__(self, monkeypatch, *, close_error=None):
        self.records = []
        self.baseline = engine.pool.checkedout()
        real_factory = hook.SessionLocal

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

        monkeypatch.setattr(hook, "SessionLocal", factory)

    @property
    def all_closed(self):
        return bool(self.records) and all(record["closed"] for record in self.records)

    @property
    def pool_is_back_to_baseline(self):
        return engine.pool.checkedout() == self.baseline


def patch_steps(monkeypatch, *, queued=True, result=None, request_error=None, run_error=None):
    """Replace both steps inside the hook with recorders; returns the ordered call list."""
    steps = []

    def fake_request(session, document_id):
        steps.append(("request", document_id))
        if request_error is not None:
            raise request_error
        return queued

    def fake_run(document_id):
        steps.append(("run", document_id))
        if run_error is not None:
            raise run_error
        return result

    monkeypatch.setattr(hook, "request_indexing", fake_request)
    monkeypatch.setattr(hook, "run_indexing", fake_run)
    return steps


def result_for(outcome, document_id=7):
    """A result shaped like the runner's for ``outcome``."""
    if outcome is RunOutcome.PUBLISHED:
        return runner.IndexingRunResult(document_id=document_id, outcome=outcome, attempt=1, chunk_count=4)
    if outcome is RunOutcome.NOT_CLAIMED:
        return runner.IndexingRunResult(document_id=document_id, outcome=outcome)
    if outcome in (RunOutcome.STALE, RunOutcome.GONE):
        return runner.IndexingRunResult(document_id=document_id, outcome=outcome, attempt=2)
    reason = next(iter(svc.FailureReason))
    failure = svc.FailureOutcome.FAILED if outcome is RunOutcome.FAILED else None
    return runner.IndexingRunResult(
        document_id=document_id, outcome=outcome, attempt=1, reason=reason, failure_outcome=failure
    )


def db_error():
    return OperationalError("SELECT secret_sql", {"p": "secret-param"}, Exception("secret-driver-message"))


SECRETS = ("secret_sql", "secret-param", "secret-driver-message", "secret-boom", "secret-cleanup")


def assert_no_secrets(caplog):
    for record in caplog.records:
        assert record.exc_info is None
    assert not any(secret in caplog.text for secret in SECRETS)


# ---------- unit: order and gating ----------

def test_a_queued_document_is_run_once_and_only_after_the_request(monkeypatch):
    steps = patch_steps(monkeypatch, result=result_for(RunOutcome.PUBLISHED))

    assert hook.index_after_extraction(7) is None

    assert steps == [("request", 7), ("run", 7)]


def test_a_refused_request_does_not_run_the_runner(monkeypatch, capture_logs):
    steps = patch_steps(monkeypatch, queued=False)

    hook.index_after_extraction(7)

    assert steps == [("request", 7)]
    [record] = hook_records(capture_logs, NOT_QUEUED)
    assert record.levelno == logging.INFO and "document_id=7" in record.getMessage()


# ---------- unit: all six runner outcomes ----------

OUTCOME_LEVELS = [
    (RunOutcome.PUBLISHED, logging.INFO),
    (RunOutcome.NOT_CLAIMED, logging.INFO),
    (RunOutcome.STALE, logging.INFO),
    (RunOutcome.GONE, logging.INFO),
    (RunOutcome.FAILED, logging.WARNING),
    (RunOutcome.FAILURE_NOT_RECORDED, logging.ERROR),
]


def test_every_runner_outcome_is_covered_by_the_parametrized_test():
    assert {outcome for outcome, _ in OUTCOME_LEVELS} == set(RunOutcome)


@pytest.mark.parametrize("outcome, level", OUTCOME_LEVELS, ids=[o.value for o, _ in OUTCOME_LEVELS])
def test_each_outcome_is_logged_once_at_its_level_and_never_raises(monkeypatch, capture_logs, outcome, level):
    patch_steps(monkeypatch, result=result_for(outcome))

    assert hook.index_after_extraction(7) is None

    [record] = hook_records(capture_logs, FINISHED)
    message = record.getMessage()
    assert record.levelno == level
    assert "document_id=7" in message and f"outcome={outcome.value}" in message
    assert not hook_records(capture_logs, FAILED_MESSAGE)
    assert_no_secrets(capture_logs)


def test_failure_not_recorded_is_logged_at_error_level_with_its_reason(monkeypatch, capture_logs):
    result = result_for(RunOutcome.FAILURE_NOT_RECORDED)
    patch_steps(monkeypatch, result=result)

    hook.index_after_extraction(7)

    [record] = [r for r in hook_records(capture_logs) if r.levelno >= logging.ERROR]
    message = record.getMessage()
    assert "outcome=failure_not_recorded" in message and f"reason={result.reason.name}" in message


def test_failed_result_logs_the_reason_and_failure_outcome_names(monkeypatch, capture_logs):
    result = result_for(RunOutcome.FAILED)
    patch_steps(monkeypatch, result=result)

    hook.index_after_extraction(7)

    [record] = hook_records(capture_logs, FINISHED)
    message = record.getMessage()
    assert f"reason={result.reason.name}" in message and "failure=FAILED" in message


def test_an_outcome_without_a_configured_level_is_logged_as_a_warning(monkeypatch, capture_logs):
    patch_steps(monkeypatch, result=result_for(RunOutcome.PUBLISHED))
    monkeypatch.setattr(hook, "_RESULT_LOG_LEVELS", {})

    hook.index_after_extraction(7)

    [record] = hook_records(capture_logs, FINISHED)
    assert record.levelno == logging.WARNING


# ---------- unit: ordinary exceptions ----------

def test_a_database_error_in_request_is_logged_without_details_and_does_not_run(monkeypatch, capture_logs):
    steps = patch_steps(monkeypatch, request_error=db_error())

    assert hook.index_after_extraction(7) is None

    assert steps == [("request", 7)]
    [record] = hook_records(capture_logs, FAILED_MESSAGE)
    message = record.getMessage()
    assert record.levelno == logging.ERROR
    assert "document_id=7" in message and "error_type=OperationalError" in message and "frames=" in message
    assert_no_secrets(capture_logs)


def test_an_unexpected_exception_from_the_runner_is_logged_without_details(monkeypatch, capture_logs):
    steps = patch_steps(monkeypatch, run_error=RuntimeError("secret-boom"))

    assert hook.index_after_extraction(7) is None

    assert steps == [("request", 7), ("run", 7)]
    [record] = hook_records(capture_logs, FAILED_MESSAGE)
    assert record.levelno == logging.ERROR and "error_type=RuntimeError" in record.getMessage()
    assert not hook_records(capture_logs, FINISHED)
    assert_no_secrets(capture_logs)

def test_an_invalid_runner_result_is_logged_without_raising(monkeypatch, capture_logs):
    patch_steps(monkeypatch, result=None)

    assert hook.index_after_extraction(7) is None

    [record] = hook_records(capture_logs, FAILED_MESSAGE)
    assert record.levelno == logging.ERROR
    assert "document_id=7" in record.getMessage()
    assert "error_type=AttributeError" in record.getMessage()
    assert_no_secrets(capture_logs)


# ---------- unit: BaseException ----------

@pytest.mark.parametrize(
    "where, error",
    [("request", KeyboardInterrupt), ("run", KeyboardInterrupt), ("run", SystemExit)],
    ids=["request-keyboard-interrupt", "run-keyboard-interrupt", "run-system-exit"],
)
def test_base_exceptions_propagate_and_are_not_logged_as_failures(monkeypatch, capture_logs, where, error):
    kwargs = {"request_error": error()} if where == "request" else {"run_error": error()}
    steps = patch_steps(monkeypatch, **kwargs)
    tracker = SessionTracker(monkeypatch)

    with pytest.raises(error):
        hook.index_after_extraction(7)

    assert steps[0] == ("request", 7)
    assert (("run", 7) in steps) is (where == "run")
    assert not hook_records(capture_logs, FAILED_MESSAGE)
    assert tracker.all_closed and tracker.pool_is_back_to_baseline


# ---------- unit: the session used for request_indexing ----------

def test_the_request_session_is_closed_even_when_the_request_fails(monkeypatch):
    patch_steps(monkeypatch, request_error=db_error())
    tracker = SessionTracker(monkeypatch)

    hook.index_after_extraction(7)

    assert len(tracker.records) == 1 and tracker.all_closed


def test_a_cleanup_error_after_a_successful_request_keeps_the_result(monkeypatch, capture_logs):
    steps = patch_steps(monkeypatch, result=result_for(RunOutcome.PUBLISHED))
    SessionTracker(monkeypatch, close_error=RuntimeError("secret-cleanup"))

    assert hook.index_after_extraction(7) is None

    assert steps == [("request", 7), ("run", 7)]
    [warning] = hook_records(capture_logs, CLEANUP)
    assert warning.levelno == logging.WARNING and "error_type=RuntimeError" in warning.getMessage()
    assert not hook_records(capture_logs, FAILED_MESSAGE)
    assert_no_secrets(capture_logs)


def test_a_cleanup_error_never_hides_the_original_request_error(monkeypatch, capture_logs):
    steps = patch_steps(monkeypatch, request_error=db_error())
    SessionTracker(monkeypatch, close_error=RuntimeError("secret-cleanup"))

    hook.index_after_extraction(7)

    assert steps == [("request", 7)]
    [record] = hook_records(capture_logs, FAILED_MESSAGE)
    assert "error_type=OperationalError" in record.getMessage()
    assert_no_secrets(capture_logs)


# ---------- end to end: real hook, real runner, real database, fake embedder ----------

def test_a_completed_document_is_queued_claimed_and_published(embedder, capture_logs):
    document_id = make_document()

    assert hook.index_after_extraction(document_id) is None

    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_attempt == 1
    assert row.indexing_error is None and row.indexing_lease_expires_at is None
    assert row.indexed_embedding_id == FakeEmbedder.embedding_id
    assert len(chunk_texts(document_id)) >= 2
    [record] = hook_records(capture_logs, FINISHED)
    assert record.levelno == logging.INFO and "outcome=published" in record.getMessage()
    assert "word0001" not in capture_logs.text


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
def test_a_document_whose_extraction_did_not_complete_is_not_indexed(embedder, capture_logs, status):
    document_id = make_document(text_=None, processing_status=status)

    hook.index_after_extraction(document_id)

    row = row_of(document_id)
    assert row.indexing_status is None and row.indexing_attempt == 0
    assert chunk_texts(document_id) == []
    assert embedder.calls == []
    assert len(hook_records(capture_logs, NOT_QUEUED)) == 1


def test_a_deleted_document_is_ignored(embedder, capture_logs):
    document_id = make_document()
    delete_document(document_id)

    assert hook.index_after_extraction(document_id) is None

    assert embedder.calls == []
    assert len(hook_records(capture_logs, NOT_QUEUED)) == 1


def test_a_document_under_a_live_lease_is_left_alone(embedder, capture_logs):
    document_id = make_document(indexing_status="pending")
    with SessionLocal() as session:
        assert svc.claim_document(session, document_id).attempt == 1
    before = row_of(document_id)

    hook.index_after_extraction(document_id)

    assert row_of(document_id) == before
    assert before.indexing_status == "indexing"
    assert embedder.calls == [] and chunk_texts(document_id) == []
    assert len(hook_records(capture_logs, NOT_QUEUED)) == 1


def test_an_embedder_failure_is_recorded_by_the_runner_and_does_not_raise(monkeypatch, capture_logs):
    error = EmbedderUnavailableError("secret-boom")
    monkeypatch.setattr(runner, "get_embedder", lambda: FakeEmbedder(embed_error=error))
    document_id = make_document()

    assert hook.index_after_extraction(document_id) is None

    row = row_of(document_id)
    assert row.indexing_status == "failed" and row.indexing_attempt == 1
    assert row.indexing_error == svc.classify_failure(error).value
    assert row.indexing_lease_expires_at is None
    assert chunk_texts(document_id) == []
    [record] = hook_records(capture_logs, FINISHED)
    assert record.levelno == logging.WARNING and "outcome=failed" in record.getMessage()
    assert "secret-boom" not in capture_logs.text


def test_a_pending_document_is_accepted_and_published_once(embedder):
    document_id = make_document(indexing_status="pending")

    hook.index_after_extraction(document_id)

    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_attempt == 1


def test_a_failed_document_is_queued_again_and_the_old_error_is_cleared_by_the_publish(embedder):
    document_id = make_document(indexing_status="failed", indexing_error="old_error", indexing_attempt=1)

    hook.index_after_extraction(document_id)

    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_attempt == 2 and row.indexing_error is None


def test_the_hook_is_not_an_index_once_guard_an_indexed_document_is_published_again(embedder):
    """request_indexing means '(re)queue': a second call after a publish publishes again.

    The upload endpoint calls the hook once per upload, so this does not happen there.
    The test records the behaviour so nobody mistakes the hook for an idempotent guard.
    """
    document_id = make_document()

    hook.index_after_extraction(document_id)
    first = chunk_texts(document_id)
    hook.index_after_extraction(document_id)

    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_attempt == 2
    assert chunk_texts(document_id) == first                      # same source, same chunks


def test_the_runner_starts_only_after_the_request_is_committed_and_its_session_closed(monkeypatch):
    document_id = make_document()
    tracker = SessionTracker(monkeypatch)
    seen = {}

    def spy_run(requested_id):
        row = row_of(requested_id)
        seen.update(
            status=row.indexing_status,
            attempt=row.indexing_attempt,
            closed=tracker.all_closed,
            pool=tracker.pool_is_back_to_baseline,
            sessions=len(tracker.records),
        )
        return result_for(RunOutcome.PUBLISHED, requested_id)

    monkeypatch.setattr(hook, "run_indexing", spy_run)

    hook.index_after_extraction(document_id)

    assert seen == {"status": "pending", "attempt": 0, "closed": True, "pool": True, "sessions": 1}
    assert tracker.pool_is_back_to_baseline


def test_no_session_or_connection_is_left_after_a_full_run(embedder, monkeypatch):
    document_id = make_document()
    tracker = SessionTracker(monkeypatch)

    hook.index_after_extraction(document_id)

    assert tracker.all_closed and tracker.pool_is_back_to_baseline
    assert row_of(document_id).indexing_status == "indexed"