"""C5.2 indexing orchestration (app/rag/indexing_runner.py).

Real test database, fake embedder: the real embedding model is never loaded.
An autouse fixture makes the module-level get_embedder() fail the test if it is
ever requested without the test patching it first.

Helpers are deliberately local to this file (tests/ has no shared helper module,
and the C5.1 tests stay unchanged). Helper reads and writes use their own short
sessions or connections, so no test leaves a transaction open for the TRUNCATE
in conftest's cleanup fixture.
"""
import hashlib
import logging
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from app.core import storage
from app.core.database import SessionLocal, engine
from app.core.processing import extract_text
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import indexing_runner as runner
from app.rag import indexing_service as svc
from app.rag.chunking import ChunkingError
from app.rag.embedder import (
    EMBEDDING_DIM,
    EmbedderUnavailableError,
    EmbeddingInputError,
    EmbeddingInputTooLongError,
    EmbeddingOutputError,
)
from app.rag.indexing import build_document_index
from app.rag.pdf_pages import PageMapping, PageMappingError, UnavailableReason
from app.rag.text_boundaries import is_ignorable
from app.rag.token_offsets import OffsetMappingError, map_token_offsets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
TEXT = " ".join(f"word{k:04d}" for k in range(700))      # 700 one-token words: several default-size chunks
NOT_A_PDF = PageMapping.unavailable(UnavailableReason.NOT_PDF)


# ---------- fake embedder ----------

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
    """The four members build_document_index uses, over runs of non-ignorable characters.

    ``source_error`` / ``embed_error`` are raised by source_units() / embed_passages().
    ``hooks`` maps a method name to a callable that runs once, on the first call of
    that method (used to change the database while the build is running).
    ``probe(name)`` runs on every call of every method.
    """

    embedding_id = "test/fake@" + "0" * 40

    def __init__(self, *, source_error=None, embed_error=None, hooks=None, probe=None):
        self.source_error = source_error
        self.embed_error = embed_error
        self.hooks = dict(hooks or {})
        self.probe = probe
        self.calls = []
        self._fired = set()

    def _enter(self, name, size):
        self.calls.append((name, size))
        if self.probe is not None:
            self.probe(name)
        hook = self.hooks.get(name)
        if hook is not None and name not in self._fired:
            self._fired.add(name)
            hook()

    def signature_fields(self):
        self._enter("signature_fields", 0)
        return {"model_id": "test/fake", "embedding_dim": EMBEDDING_DIM}

    def source_units(self, value):
        self._enter("source_units", len(value))
        if self.source_error is not None:
            raise self.source_error
        spans = word_spans(value)
        return map_token_offsets(value, spans, [value[start:end] for start, end in spans])

    def embed_passages(self, texts):
        self._enter("embed_passages", len(texts))
        if self.embed_error is not None:
            raise self.embed_error
        return np.stack([fake_vector(item) for item in texts]).astype(np.float32)


def _explode(secret):
    raise RuntimeError(secret)  # LINE_MARKER_7f3a91


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


def queue(document_id):
    with SessionLocal() as session:
        assert svc.request_indexing(session, document_id) is True


def row_of(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(
                Document.indexing_status,
                Document.indexing_error,
                Document.indexing_attempt,
                Document.indexing_lease_expires_at,
                Document.source_sha256,
                Document.indexed_source_version,
                Document.indexed_signature,
                Document.indexed_embedding_id,
                Document.indexed_at,
            ).where(Document.id == document_id)
        ).one()


def chunks_of(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(
                DocumentChunk.chunk_index,
                DocumentChunk.start_offset,
                DocumentChunk.end_offset,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.text,
                DocumentChunk.span_sha256,
                DocumentChunk.embedding,
            )
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()


def snapshot(document_id):
    row = row_of(document_id)
    return (
        row.source_sha256,
        row.indexed_source_version,
        row.indexed_signature,
        row.indexed_embedding_id,
        row.indexed_at,
        [(c.chunk_index, c.start_offset, c.end_offset, c.text, c.span_sha256, tuple(c.embedding))
         for c in chunks_of(document_id)],
    )


def retrievable(document_id, embedding_id=FakeEmbedder.embedding_id):
    with SessionLocal() as session:
        return (
            session.execute(
                select(Document.id).where(Document.id == document_id, svc.retrievable_filter(embedding_id))
            ).first()
            is not None
        )


def expire_lease(document_id):
    execute(
        "UPDATE documents SET indexing_lease_expires_at = clock_timestamp() - interval '1 minute' WHERE id = :id",
        id=document_id,
    )


def delete_document(document_id):
    execute("DELETE FROM documents WHERE id = :id", id=document_id)


def take_over(document_id):
    """Another worker takes the document over: expire the lease, claim again (attempt 2)."""
    expire_lease(document_id)
    with SessionLocal() as session:
        claim = svc.claim_document(session, document_id)
    assert claim is not None and claim.attempt == 2
    return claim


def publish_once(document_id):
    """Queue and index a document with a plain run; it must publish."""
    queue(document_id)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert result.outcome is runner.RunOutcome.PUBLISHED
    return result


# ---------- fixtures ----------

class Spies:
    def __init__(self):
        self.calls = {"claim_document": 0, "publish_build": 0, "record_failure": 0, "get_embedder": 0}


@pytest.fixture(autouse=True)
def spies(monkeypatch):
    """Count calls of the C5.1 functions (delegating to the real ones) and forbid the real embedder."""
    spy = Spies()

    def wrap(name, real):
        def inner(*args, **kwargs):
            spy.calls[name] += 1
            return real(*args, **kwargs)
        monkeypatch.setattr(runner, name, inner)

    wrap("claim_document", svc.claim_document)
    wrap("publish_build", svc.publish_build)
    wrap("record_failure", svc.record_failure)

    def forbidden():
        spy.calls["get_embedder"] += 1
        raise AssertionError("the real embedder must not be requested in tests")

    monkeypatch.setattr(runner, "get_embedder", forbidden)
    yield spy
    assert spy.calls["get_embedder"] == 0


class SessionTracker:
    """Wraps runner.SessionLocal: records every session and when it was closed."""

    def __init__(self, monkeypatch):
        self.records = []
        self.probes = []
        self.baseline = engine.pool.checkedout()
        real_factory = runner.SessionLocal

        def factory():
            session = real_factory()
            record = {"closed": False, "in_transaction_at_close": None}
            real_close = session.close

            def close():
                record["in_transaction_at_close"] = session.in_transaction()
                record["closed"] = True
                return real_close()

            session.close = close
            self.records.append(record)
            return session

        monkeypatch.setattr(runner, "SessionLocal", factory)

    def probe(self, name):
        """Called by the fake embedder: what is open while the build runs?"""
        self.probes.append((name, all(r["closed"] for r in self.records), engine.pool.checkedout()))

    @property
    def all_closed(self):
        return all(record["closed"] for record in self.records)


def pool_is_back_to_baseline(tracker):
    return engine.pool.checkedout() == tracker.baseline


# ---------- success and inputs ----------

def test_success_publishes_every_chunk_and_reports_counts(spies):
    document_id = make_document()
    queue(document_id)
    embedder = FakeEmbedder()
    result = runner.run_indexing(document_id, embedder=embedder)

    stored = chunks_of(document_id)
    assert result == runner.IndexingRunResult(
        document_id=document_id, outcome=runner.RunOutcome.PUBLISHED, attempt=1, chunk_count=len(stored)
    )
    assert len(stored) >= 2
    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_error is None and row.indexing_lease_expires_at is None
    assert row.indexed_embedding_id == FakeEmbedder.embedding_id and row.indexed_source_version == 1
    assert row.source_sha256 == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    for chunk in stored:
        assert chunk.text == TEXT[chunk.start_offset:chunk.end_offset]
        assert chunk.page_start is None and chunk.page_end is None
        assert len(chunk.embedding) == EMBEDDING_DIM                       # pgvector 0.5.1 returns a list
        assert np.allclose(np.asarray(chunk.embedding, dtype=np.float32), fake_vector(chunk.text), atol=1e-6)
    names = [name for name, _ in embedder.calls]
    assert set(names) == {"source_units", "signature_fields", "embed_passages"} and names[0] == "source_units"
    assert spies.calls["record_failure"] == 0
    assert retrievable(document_id)


def test_sample_pdf_chunks_get_real_page_numbers(tmp_path):
    shutil.copy(SAMPLE_PDF, tmp_path / "sample.pdf")                      # conftest points UPLOAD_ROOT at tmp_path
    document_id = make_document(text_=extract_text(storage.get_file_path("sample.pdf")), stored_name="sample.pdf")
    queue(document_id)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert result.outcome is runner.RunOutcome.PUBLISHED
    stored = chunks_of(document_id)
    assert stored and all(c.page_start is not None and 1 <= c.page_start <= c.page_end for c in stored)


@pytest.mark.parametrize("stored_name", ["notes.txt", None, "ghost.pdf"], ids=["non-pdf", "no-stored-name", "missing-pdf"])
def test_documents_without_a_page_mapping_publish_with_null_pages(stored_name):
    document_id = make_document(stored_name=stored_name)
    queue(document_id)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert result.outcome is runner.RunOutcome.PUBLISHED
    assert all(c.page_start is None and c.page_end is None for c in chunks_of(document_id))


def test_an_invalid_stored_filename_is_a_recorded_failure():
    document_id = make_document(stored_name="../escape.txt")
    queue(document_id)
    embedder = FakeEmbedder()
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result.outcome is runner.RunOutcome.FAILED and result.reason is svc.FailureReason.UNEXPECTED
    assert row_of(document_id).indexing_error == svc.FailureReason.UNEXPECTED.value
    assert embedder.calls == []


@pytest.mark.parametrize("value", ["", "  \n\t ", None], ids=["empty", "whitespace", "null"])
def test_text_without_content_fails_with_no_indexable_text(value):
    document_id = make_document(text_=value)
    queue(document_id)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert result.outcome is runner.RunOutcome.FAILED and result.reason is svc.FailureReason.NO_INDEXABLE_TEXT
    row = row_of(document_id)
    assert row.indexing_status == "failed" and row.indexing_error == "No indexable text found in this document"


# ---------- not claimed, lazy embedder ----------

def _not_claimable(kind):
    if kind == "never-requested":
        return make_document()
    if kind == "missing":
        return 987654
    if kind == "incomplete":
        return make_document(processing_status="processing", indexing_status="pending")
    document_id = make_document()                                          # live lease held by someone else
    queue(document_id)
    with SessionLocal() as session:
        assert svc.claim_document(session, document_id) is not None
    return document_id


@pytest.mark.parametrize("kind", ["never-requested", "missing", "incomplete", "live-lease"])
def test_not_claimed_cases_do_nothing_and_never_touch_the_embedder(kind, spies, monkeypatch):
    document_id = _not_claimable(kind)
    before = None if kind == "missing" else (row_of(document_id), chunks_of(document_id))
    embedder = FakeEmbedder()
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result == runner.IndexingRunResult(document_id=document_id, outcome=runner.RunOutcome.NOT_CLAIMED)
    assert embedder.calls == []
    assert spies.calls["get_embedder"] == 0 and spies.calls["record_failure"] == 0 and spies.calls["publish_build"] == 0
    if before is not None:
        assert (row_of(document_id), chunks_of(document_id)) == before


def test_get_embedder_is_not_called_when_unclaimed_and_called_once_after_a_claim(monkeypatch):
    requested = []
    fake = FakeEmbedder()
    monkeypatch.setattr(runner, "get_embedder", lambda: requested.append(1) or fake)

    runner.run_indexing(make_document())                                   # never queued: not claimed
    assert requested == []

    document_id = make_document()
    queue(document_id)
    result = runner.run_indexing(document_id)                              # default embedder
    assert result.outcome is runner.RunOutcome.PUBLISHED
    assert requested == [1] and fake.calls


# ---------- publication outcomes ----------

def test_publication_exception_rolls_back_and_the_valid_publication_survives(monkeypatch, spies):
    document_id = make_document()
    publish_once(document_id)
    before = snapshot(document_id)
    queue(document_id)

    def corrupt_the_last_chunk_then_publish(session, claim, build):
        broken = replace(build.chunks[-1], page_start=0, page_end=0)        # violates ck_document_chunks_pages_valid
        return svc.publish_build(session, claim, replace(build, chunks=build.chunks[:-1] + (broken,)))

    monkeypatch.setattr(runner, "publish_build", corrupt_the_last_chunk_then_publish)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())

    assert result.outcome is runner.RunOutcome.FAILED and result.reason is svc.FailureReason.DATABASE
    assert result.failure_outcome is svc.FailureOutcome.RESTORED
    assert snapshot(document_id) == before                                  # nothing from the rolled-back publish
    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_error == svc.FailureReason.DATABASE.value
    assert spies.calls["record_failure"] == 1


def test_stale_when_the_document_is_taken_over_during_the_build(spies):
    document_id = make_document()
    queue(document_id)
    taken = []
    embedder = FakeEmbedder(hooks={"embed_passages": lambda: taken.append(take_over(document_id))})
    result = runner.run_indexing(document_id, embedder=embedder)

    assert result.outcome is runner.RunOutcome.STALE and result.attempt == 1
    assert spies.calls["record_failure"] == 0 and chunks_of(document_id) == []
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 2 and row.indexing_error is None
    build = build_document_index(TEXT, embedder=FakeEmbedder(), page_mapping=NOT_A_PDF)
    with SessionLocal() as session:
        assert svc.publish_build(session, taken[0], build) is svc.PublishOutcome.PUBLISHED


def test_stale_when_the_source_text_changes_during_the_build(spies):
    document_id = make_document()
    queue(document_id)
    embedder = FakeEmbedder(hooks={"embed_passages": lambda: execute(
        "UPDATE documents SET extracted_text = extracted_text || ' edited' WHERE id = :id", id=document_id)})
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result.outcome is runner.RunOutcome.STALE
    assert spies.calls["record_failure"] == 0 and chunks_of(document_id) == []
    assert row_of(document_id).indexing_status == "indexing"


def test_gone_when_the_document_is_deleted_during_the_build(spies):
    document_id = make_document()
    queue(document_id)
    embedder = FakeEmbedder(hooks={"embed_passages": lambda: delete_document(document_id)})
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result.outcome is runner.RunOutcome.GONE
    assert spies.calls["record_failure"] == 0 and chunks_of(document_id) == []


def test_gone_when_the_document_is_deleted_between_claim_and_load(monkeypatch, spies):
    document_id = make_document()
    queue(document_id)

    def claim_then_delete(session, requested_id, **kwargs):
        claim = svc.claim_document(session, requested_id, **kwargs)
        delete_document(requested_id)
        return claim

    monkeypatch.setattr(runner, "claim_document", claim_then_delete)
    embedder = FakeEmbedder()
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result.outcome is runner.RunOutcome.GONE and embedder.calls == []
    assert spies.calls["record_failure"] == 0


def test_a_session_cleanup_error_after_publication_does_not_become_a_failure(monkeypatch, spies, caplog):
    document_id = make_document()
    queue(document_id)
    tracker = SessionTracker(monkeypatch)

    def publish_then_break_close(session, claim, build):
        outcome = svc.publish_build(session, claim, build)
        real_close = session.close

        def close():
            real_close()
            raise RuntimeError("SECRET_CLOSE_55")

        session.close = close
        return outcome

    monkeypatch.setattr(runner, "publish_build", publish_then_break_close)
    caplog.set_level(logging.DEBUG, logger="documind")
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())

    assert result.outcome is runner.RunOutcome.PUBLISHED and result.chunk_count == len(chunks_of(document_id))
    assert spies.calls["record_failure"] == 0
    assert row_of(document_id).indexing_status == "indexed"
    assert "RuntimeError" in caplog.text and "SECRET_CLOSE_55" not in caplog.text
    assert pool_is_back_to_baseline(tracker)


def test_a_session_cleanup_error_after_recording_a_failure_is_still_a_recorded_failure(monkeypatch):
    document_id = make_document()
    queue(document_id)

    def record_then_break_close(session, claim, reason):
        outcome = svc.record_failure(session, claim, reason)
        real_close = session.close

        def close():
            real_close()
            raise RuntimeError("cleanup")

        session.close = close
        return outcome

    monkeypatch.setattr(runner, "record_failure", record_then_break_close)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=EmbedderUnavailableError("x")))
    assert result.outcome is runner.RunOutcome.FAILED and result.failure_outcome is svc.FailureOutcome.FAILED
    assert row_of(document_id).indexing_status == "failed"


# ---------- failures ----------

def _pages_error(monkeypatch):
    def broken(path, extracted_text):
        raise PageMappingError("page mapping bug")
    monkeypatch.setattr(runner, "map_document_pages", broken)


@pytest.mark.parametrize(
    ("where", "error", "reason"),
    [
        ("source", OffsetMappingError("uncovered_character", 3), svc.FailureReason.OFFSET_MAPPING),
        ("source", ChunkingError("no safe boundary"), svc.FailureReason.CHUNKING),
        ("source", EmbedderUnavailableError("model missing"), svc.FailureReason.EMBEDDER_UNAVAILABLE),
        ("embed", EmbeddingInputTooLongError(0, 600), svc.FailureReason.CHUNK_TOO_LONG),
        ("embed", EmbeddingInputError("wrong type"), svc.FailureReason.EMBEDDING_INVALID),
        ("embed", EmbeddingOutputError("bad vectors"), svc.FailureReason.EMBEDDING_INVALID),
        ("embed", RuntimeError("anything"), svc.FailureReason.UNEXPECTED),
        ("pages", PageMappingError("bug"), svc.FailureReason.PAGE_MAPPING),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
def test_build_failures_are_recorded_with_a_fixed_reason(where, error, reason, monkeypatch):
    document_id = make_document()
    queue(document_id)
    if where == "pages":
        _pages_error(monkeypatch)
        embedder = FakeEmbedder()
    else:
        embedder = FakeEmbedder(**{"source_error" if where == "source" else "embed_error": error})
    result = runner.run_indexing(document_id, embedder=embedder)

    assert result.outcome is runner.RunOutcome.FAILED and result.reason is reason
    assert result.failure_outcome is svc.FailureOutcome.FAILED and result.attempt == 1
    row = row_of(document_id)
    assert row.indexing_status == "failed" and row.indexing_error == reason.value
    assert row.indexing_lease_expires_at is None and chunks_of(document_id) == []


def test_logs_and_database_never_contain_exception_messages_or_source_lines(caplog):
    document_id = make_document()
    queue(document_id)
    embedder = FakeEmbedder()
    embedder.source_units = lambda value: _explode("SECRET_MESSAGE_9c1d")
    caplog.set_level(logging.DEBUG, logger="documind")

    result = runner.run_indexing(document_id, embedder=embedder)

    assert result.reason is svc.FailureReason.UNEXPECTED
    assert row_of(document_id).indexing_error == svc.FailureReason.UNEXPECTED.value
    logged = caplog.text
    assert "SECRET_MESSAGE_9c1d" not in logged and "LINE_MARKER_7f3a91" not in logged
    assert f"document_id={document_id}" in logged and "attempt=1" in logged and "reason=UNEXPECTED" in logged
    assert "error_type=RuntimeError" in logged
    assert re.search(r"test_indexing_runner\.py:_explode:\d+", logged)


def test_database_errors_do_not_leak_sql_parameters_into_logs(monkeypatch, caplog):
    error = OperationalError("INSERT INTO t VALUES (%(p)s)", {"p": "SECRET_PARAM_77"}, Exception("SECRET_DRIVER_88"))
    assert "SECRET_PARAM_77" in str(error)                                  # premise: the message does carry them

    def failing_publish(session, claim, build):
        raise error

    monkeypatch.setattr(runner, "publish_build", failing_publish)
    document_id = make_document()
    queue(document_id)
    caplog.set_level(logging.DEBUG, logger="documind")
    result = runner.run_indexing(document_id, embedder=FakeEmbedder())

    assert result.outcome is runner.RunOutcome.FAILED and result.reason is svc.FailureReason.DATABASE
    assert "SECRET_PARAM_77" not in caplog.text and "SECRET_DRIVER_88" not in caplog.text
    assert "error_type=OperationalError" in caplog.text


def test_a_failed_reindex_keeps_the_valid_previous_publication():
    document_id = make_document()
    publish_once(document_id)
    before = snapshot(document_id)
    queue(document_id)

    result = runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=EmbedderUnavailableError("x")))

    assert result.outcome is runner.RunOutcome.FAILED and result.failure_outcome is svc.FailureOutcome.RESTORED
    assert snapshot(document_id) == before
    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_error == svc.FailureReason.EMBEDDER_UNAVAILABLE.value
    assert retrievable(document_id)


def test_a_failed_reindex_discards_an_invalid_previous_publication():
    document_id = make_document()
    publish_once(document_id)
    execute("UPDATE documents SET extracted_text = 'a completely different text' WHERE id = :id", id=document_id)
    queue(document_id)

    result = runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=RuntimeError("x")))

    assert result.outcome is runner.RunOutcome.FAILED and result.failure_outcome is svc.FailureOutcome.DISCARDED
    assert chunks_of(document_id) == [] and not retrievable(document_id)
    row = row_of(document_id)
    assert row.indexing_status == "failed" and row.source_sha256 is None and row.indexed_at is None


def test_when_recording_the_failure_fails_the_lease_recovers_the_document(monkeypatch):
    document_id = make_document()
    queue(document_id)

    def unavailable(session, claim, reason):
        raise OperationalError("UPDATE documents", {}, Exception("connection lost"))

    monkeypatch.setattr(runner, "record_failure", unavailable)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=RuntimeError("x")))

    assert result.outcome is runner.RunOutcome.FAILURE_NOT_RECORDED and result.reason is svc.FailureReason.UNEXPECTED
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 1 and row.indexing_error is None

    expire_lease(document_id)                                              # the lease is the recovery path
    queue(document_id)
    recovered = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert recovered.outcome is runner.RunOutcome.PUBLISHED and recovered.attempt == 2


def test_a_failure_by_a_superseded_attempt_reports_stale_and_changes_nothing(spies):
    document_id = make_document()
    queue(document_id)
    embedder = FakeEmbedder(embed_error=RuntimeError("x"), hooks={"embed_passages": lambda: take_over(document_id)})
    result = runner.run_indexing(document_id, embedder=embedder)
    assert result.outcome is runner.RunOutcome.STALE
    assert spies.calls["record_failure"] == 1
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 2 and row.indexing_error is None


def test_a_database_error_while_claiming_propagates_and_changes_nothing(monkeypatch):
    document_id = make_document()
    queue(document_id)

    def unavailable(session, requested_id, **kwargs):
        raise OperationalError("UPDATE documents", {}, Exception("connection lost"))

    monkeypatch.setattr(runner, "claim_document", unavailable)
    embedder = FakeEmbedder()
    with pytest.raises(OperationalError):
        runner.run_indexing(document_id, embedder=embedder)
    assert row_of(document_id).indexing_status == "pending" and embedder.calls == []


# ---------- interruptions ----------

@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(3)], ids=["KeyboardInterrupt", "SystemExit"])
def test_interruptions_propagate_and_the_lease_recovers_the_document(interruption, spies, monkeypatch):
    document_id = make_document()
    queue(document_id)
    tracker = SessionTracker(monkeypatch)

    with pytest.raises(type(interruption)):
        runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=interruption, probe=tracker.probe))

    assert spies.calls["record_failure"] == 0
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 1 and row.indexing_error is None
    assert row.indexing_lease_expires_at is not None
    assert tracker.all_closed and pool_is_back_to_baseline(tracker)

    expire_lease(document_id)
    queue(document_id)
    recovered = runner.run_indexing(document_id, embedder=FakeEmbedder())
    assert recovered.outcome is runner.RunOutcome.PUBLISHED and recovered.attempt == 2


# ---------- session and connection discipline ----------

SCENARIOS = [
    "success", "not-claimed", "build-failure", "stale", "gone", "publish-exception",
    "record-failure-raises", "claim-raises", "keyboard-interrupt", "system-exit",
]
NO_BUILD = {"not-claimed", "claim-raises"}


def _drive(kind, monkeypatch, tracker):
    document_id = make_document()
    if kind != "not-claimed":
        queue(document_id)
    embedder = FakeEmbedder(probe=tracker.probe)
    if kind == "build-failure":
        embedder.embed_error = EmbedderUnavailableError("x")
    elif kind == "stale":
        embedder.hooks["embed_passages"] = lambda: take_over(document_id)
    elif kind == "gone":
        embedder.hooks["embed_passages"] = lambda: delete_document(document_id)
    elif kind == "keyboard-interrupt":
        embedder.embed_error = KeyboardInterrupt()
    elif kind == "system-exit":
        embedder.embed_error = SystemExit(1)
    elif kind == "publish-exception":
        monkeypatch.setattr(runner, "publish_build", lambda *a, **k: (_ for _ in ()).throw(
            OperationalError("INSERT", {}, Exception("lost"))))
    elif kind == "record-failure-raises":
        embedder.embed_error = RuntimeError("x")
        monkeypatch.setattr(runner, "record_failure", lambda *a, **k: (_ for _ in ()).throw(
            OperationalError("UPDATE", {}, Exception("lost"))))
    elif kind == "claim-raises":
        monkeypatch.setattr(runner, "claim_document", lambda *a, **k: (_ for _ in ()).throw(
            OperationalError("UPDATE", {}, Exception("lost"))))
    try:
        outcome = runner.run_indexing(document_id, embedder=embedder)
    except (KeyboardInterrupt, SystemExit, OperationalError) as exc:
        outcome = exc
    return embedder, outcome


@pytest.mark.parametrize("kind", SCENARIOS)
def test_every_session_is_closed_and_no_connection_is_held_during_the_build(kind, monkeypatch):
    tracker = SessionTracker(monkeypatch)
    embedder, _ = _drive(kind, monkeypatch, tracker)

    assert tracker.records and tracker.all_closed                          # every session closed on every path
    assert pool_is_back_to_baseline(tracker)                               # and no connection is left checked out
    if kind in NO_BUILD:
        assert embedder.calls == [] and tracker.probes == []
    else:
        assert tracker.probes and tracker.probes[0][0] == "source_units"
        for name, sessions_closed, checked_out in tracker.probes:          # during every embedder call
            assert sessions_closed, name                                   # no session is open
            assert checked_out == tracker.baseline, name                   # no pooled connection is held


def test_no_connection_is_idle_in_transaction_while_the_build_runs():
    """Supplementary: PostgreSQL's own view. The instrumented tests above are the primary proof."""
    document_id = make_document()
    queue(document_id)
    seen = []

    def probe(name):
        with engine.connect() as connection:
            seen.append(connection.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND pid <> pg_backend_pid() AND state LIKE 'idle in transaction%'")).scalar_one())

    result = runner.run_indexing(document_id, embedder=FakeEmbedder(probe=probe))
    assert result.outcome is runner.RunOutcome.PUBLISHED
    assert seen and set(seen) == {0}


# ---------- concurrency and structure ----------

def test_concurrent_runs_publish_exactly_once():
    document_id = make_document()
    queue(document_id)
    workers = 4
    barrier = threading.Barrier(workers)

    def attempt():
        barrier.wait()
        return runner.run_indexing(document_id, embedder=FakeEmbedder())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [future.result() for future in [pool.submit(attempt) for _ in range(workers)]]

    outcomes = sorted(result.outcome.value for result in results)
    assert outcomes == ["not_claimed"] * (workers - 1) + ["published"]
    assert row_of(document_id).indexing_attempt == 1 and row_of(document_id).indexing_status == "indexed"


def test_results_hold_only_ids_enums_and_counts():
    document_id = make_document()
    queue(document_id)
    result = runner.run_indexing(document_id, embedder=FakeEmbedder(embed_error=RuntimeError("x")))
    assert {field.name for field in fields(runner.IndexingRunResult)} == {
        "document_id", "outcome", "attempt", "chunk_count", "reason", "failure_outcome"}
    for field in fields(result):
        assert isinstance(getattr(result, field.name), (int, runner.RunOutcome, svc.FailureReason, svc.FailureOutcome, type(None)))
    assert "word0001" not in repr(result)


def test_the_runner_never_catches_base_exception():
    """Tripwire: only ordinary Exception may be handled (interruptions must propagate)."""
    source = (PROJECT_ROOT / "app" / "rag" / "indexing_runner.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"except\s+BaseException|except\s*:", code)
    assert len(re.findall(r"except\s+Exception", code)) >= 1