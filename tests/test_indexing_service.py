"""C5.1 indexing service: lifecycle transitions on the real test database.

No model is loaded. Builds are DocumentIndexBuild values assembled by hand from
slices of the document's extracted_text, with fake unit vectors.

Every service call gets its own fresh session (as C5.2 will), because each
function opens its own transaction with ``db.begin()``. Helpers that inspect or
change rows also use their own short-lived connections, so no test leaves a
transaction open for the TRUNCATE in conftest's cleanup fixture.
"""
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import (
    IntegrityError,
    InvalidRequestError,
    OperationalError,
    SQLAlchemyError,
)

from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import indexing_service as svc
from app.rag.chunking import ChunkingError, sha256_text
from app.rag.embedder import (
    EmbedderError,
    EmbedderUnavailableError,
    EmbeddingInputError,
    EmbeddingInputTooLongError,
    EmbeddingOutputError,
)
from app.rag.indexing import (
    NO_INDEXABLE_TEXT,
    ChunkRecord,
    DocumentIndexBuild,
    IndexBuildError,
    IndexBuildStats,
)
from app.rag.pdf_pages import PageMappingError
from app.rag.token_offsets import OffsetMappingError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ID = "test-model@0000000000000000000000000000000000000000"
TEXT = " ".join(f"word{k:03d}" for k in range(80))      # 639 characters, plain ASCII


# ---------- helpers ----------

def call(function, *args, **kwargs):
    """Run one service call in its own fresh session."""
    with SessionLocal() as session:
        return function(session, *args, **kwargs)


def execute(sql, **params):
    with engine.begin() as connection:
        connection.execute(text(sql), params)


def make_document(text_=TEXT, processing_status="completed", **columns):
    with SessionLocal() as session:
        user = User(email=f"u{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        document = Document(
            filename="a.txt",
            owner_id=user.id,
            processing_status=processing_status,
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
                Document.processing_status,
                Document.source_version,
                Document.source_sha256,
                Document.indexing_status,
                Document.indexing_error,
                Document.indexing_attempt,
                Document.indexing_lease_expires_at,
                Document.indexed_source_version,
                Document.indexed_signature,
                Document.indexed_embedding_id,
                Document.indexed_at,
                Document.extracted_text,
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


def db_now():
    with engine.connect() as connection:
        return connection.execute(text("SELECT clock_timestamp()")).scalar_one()


def retrievable(document_id, embedding_id=EMBEDDING_ID):
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


def make_build(text_=TEXT, count=3, embedding_id=EMBEDDING_ID, signature="sig-v1", page=None):
    """A build of ``count`` equal slices of ``text_`` with one-hot unit vectors."""
    size = len(text_) // count
    records = []
    for index in range(count):
        start = index * size
        end = (index + 1) * size if index < count - 1 else len(text_)
        piece = text_[start:end]
        vector = np.zeros(384, dtype=np.float32)
        vector[index] = 1.0
        vector.setflags(write=False)
        records.append(
            ChunkRecord(
                chunk_index=index,
                start_offset=start,
                end_offset=end,
                page_start=page,
                page_end=page,
                text=piece,
                span_sha256=sha256_text(piece),
                embedding=vector,
            )
        )
    return DocumentIndexBuild(
        source_sha256=sha256_text(text_),
        signature=signature,
        embedding_id=embedding_id,
        chunks=tuple(records),
        stats=IndexBuildStats(chunk_count=count, source_tokens=count, max_chunk_tokens=1, pages_available=page is not None),
    )


def claim(document_id):
    claimed = call(svc.claim_document, document_id)
    assert claimed is not None
    return claimed


def publish(document_id, build=None):
    """pending -> indexing -> indexed through the service; returns the claim."""
    assert call(svc.request_indexing, document_id) is True
    claimed = claim(document_id)
    assert call(svc.publish_build, claimed, build or make_build()) is svc.PublishOutcome.PUBLISHED
    return claimed


def reindex_and_claim(document_id):
    assert call(svc.request_indexing, document_id) is True
    return claim(document_id)


def set_state(document_id, state):
    """Put a completed document straight into ``state`` (None, pending, failed, indexed, live, expired)."""
    if state is None:
        return
    if state in ("pending", "failed"):
        execute("UPDATE documents SET indexing_status = :s WHERE id = :id", s=state, id=document_id)
    elif state == "indexed":
        execute(
            "UPDATE documents SET indexing_status = 'indexed', indexed_source_version = 1, source_sha256 = :h,"
            " indexed_signature = 'sig', indexed_embedding_id = 'emb', indexed_at = clock_timestamp() WHERE id = :id",
            h="0" * 64,
            id=document_id,
        )
    elif state in ("live", "expired"):
        offset = "+ interval '10 minutes'" if state == "live" else "- interval '1 minute'"
        execute(
            f"UPDATE documents SET indexing_status = 'indexing', indexing_attempt = 1,"
            f" indexing_lease_expires_at = clock_timestamp() {offset} WHERE id = :id",
            id=document_id,
        )
    else:
        raise AssertionError(state)


def snapshot(document_id):
    row = row_of(document_id)
    return (
        row.source_sha256,
        row.indexed_source_version,
        row.indexed_signature,
        row.indexed_embedding_id,
        row.indexed_at,
        [(c.chunk_index, c.start_offset, c.end_offset, c.text, c.span_sha256, tuple(c.embedding)) for c in chunks_of(document_id)],
    )


# ---------- request_indexing ----------

@pytest.mark.parametrize("state", [None, "failed", "indexed", "expired", "pending"])
def test_request_indexing_moves_eligible_states_to_pending(state):
    document_id = make_document()
    set_state(document_id, state)
    assert call(svc.request_indexing, document_id) is True
    row = row_of(document_id)
    assert row.indexing_status == "pending" and row.indexing_lease_expires_at is None


def test_request_indexing_is_refused_without_changing_anything():
    live = make_document()
    set_state(live, "live")
    unextracted = make_document(processing_status="processing")
    before = row_of(live)
    assert call(svc.request_indexing, live) is False
    assert call(svc.request_indexing, unextracted) is False
    assert call(svc.request_indexing, 987654) is False
    assert row_of(live) == before
    assert row_of(unextracted).indexing_status is None


@pytest.mark.parametrize("state", ["failed", "expired", "indexed"])
def test_request_indexing_keeps_the_previous_error(state):
    document_id = make_document(indexing_error="old error")
    set_state(document_id, state)
    assert call(svc.request_indexing, document_id) is True
    assert row_of(document_id).indexing_error == "old error"


# ---------- claim_document ----------

def test_claim_from_pending_sets_attempt_status_and_database_clock_lease(monkeypatch):
    document_id = make_document(indexing_error="old error")
    call(svc.request_indexing, document_id)
    before = db_now()
    claimed = call(svc.claim_document, document_id)
    after = db_now()
    assert claimed == svc.IndexingClaim(document_id=document_id, attempt=1, source_version=1)
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 1
    assert row.indexing_error == "old error"                                   # claiming keeps the error
    minutes = settings.indexing_lease_minutes
    assert minutes == 15
    assert before + timedelta(minutes=minutes) <= row.indexing_lease_expires_at <= after + timedelta(minutes=minutes)


def test_claim_lease_minutes_default_comes_from_settings_and_can_be_overridden(monkeypatch):
    monkeypatch.setattr(settings, "indexing_lease_minutes", 3)
    first = make_document()
    call(svc.request_indexing, first)
    before = db_now()
    claim(first)
    assert row_of(first).indexing_lease_expires_at - before < timedelta(minutes=3, seconds=30)
    second = make_document()
    call(svc.request_indexing, second)
    before = db_now()
    call(svc.claim_document, second, lease_minutes=1)
    assert row_of(second).indexing_lease_expires_at - before < timedelta(minutes=1, seconds=30)


@pytest.mark.parametrize("minutes", [0, -5])
def test_claim_rejects_a_lease_below_one_minute(minutes):
    document_id = make_document()
    call(svc.request_indexing, document_id)
    with pytest.raises(ValueError):
        call(svc.claim_document, document_id, lease_minutes=minutes)
    assert row_of(document_id).indexing_status == "pending"


@pytest.mark.parametrize("state", [None, "failed", "indexed", "live"])
def test_claim_is_refused_unless_pending_or_expired(state):
    document_id = make_document()
    set_state(document_id, state)
    before = row_of(document_id)
    assert call(svc.claim_document, document_id) is None
    assert row_of(document_id) == before


def test_claim_is_refused_for_unextracted_and_missing_documents():
    document_id = make_document(processing_status="processing", indexing_status="pending")
    assert call(svc.claim_document, document_id) is None
    assert call(svc.claim_document, 987654) is None
    assert row_of(document_id).indexing_attempt == 0


def test_concurrent_claims_have_exactly_one_winner():
    workers = 8
    for _ in range(5):
        document_id = make_document()
        call(svc.request_indexing, document_id)
        barrier = threading.Barrier(workers)

        def attempt(document_id=document_id, barrier=barrier):
            with SessionLocal() as session:
                barrier.wait()
                return svc.claim_document(session, document_id)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in [pool.submit(attempt) for _ in range(workers)]]
        winners = [result for result in results if result is not None]
        assert len(winners) == 1 and winners[0].attempt == 1
        assert results.count(None) == workers - 1
        row = row_of(document_id)
        assert row.indexing_attempt == 1 and row.indexing_status == "indexing"


def test_expired_lease_can_be_taken_over_with_a_new_attempt():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    first = claim(document_id)
    assert call(svc.claim_document, document_id) is None                        # live lease: refused
    expire_lease(document_id)
    second = claim(document_id)
    assert (first.attempt, second.attempt) == (1, 2)
    assert row_of(document_id).indexing_attempt == 2


# ---------- publish_build ----------

def test_publish_writes_every_chunk_field_and_the_publication_metadata():
    document_id = make_document(indexing_error="old error")
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    build = make_build(count=4, page=2)
    before = db_now()
    assert call(svc.publish_build, claimed, build) is svc.PublishOutcome.PUBLISHED
    after = db_now()

    row = row_of(document_id)
    assert row.indexing_status == "indexed"
    assert row.indexing_error is None and row.indexing_lease_expires_at is None
    assert row.indexed_source_version == 1 and row.source_sha256 == build.source_sha256 == sha256_text(TEXT)
    assert row.indexed_signature == "sig-v1" and row.indexed_embedding_id == EMBEDDING_ID
    assert before <= row.indexed_at <= after
    assert row.indexing_attempt == 1

    stored = chunks_of(document_id)
    assert len(stored) == len(build.chunks) == 4
    for got, want in zip(stored, build.chunks, strict=True):
        assert (got.chunk_index, got.start_offset, got.end_offset) == (want.chunk_index, want.start_offset, want.end_offset)
        assert (got.page_start, got.page_end) == (2, 2)
        assert got.text == want.text == TEXT[got.start_offset:got.end_offset]
        assert got.span_sha256 == want.span_sha256
        assert len(got.embedding) == 384                                         # pgvector 0.5.1 returns a list
        assert np.allclose(np.asarray(got.embedding, dtype=np.float32), want.embedding, atol=1e-6)


def test_publish_stores_null_pages_when_pages_are_unavailable():
    document_id = make_document()
    publish(document_id, make_build(page=None))
    assert all(c.page_start is None and c.page_end is None for c in chunks_of(document_id))


def test_republish_replaces_the_previous_chunks():
    document_id = make_document()
    publish(document_id, make_build(count=5))
    assert len(chunks_of(document_id)) == 5
    publish(document_id, make_build(count=3, signature="sig-v2"))
    stored = chunks_of(document_id)
    assert [c.chunk_index for c in stored] == [0, 1, 2]
    assert row_of(document_id).indexed_signature == "sig-v2"


def test_publish_with_a_stale_attempt_writes_nothing_even_after_the_new_attempt_is_running():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    old = claim(document_id)
    expire_lease(document_id)
    new = claim(document_id)
    assert call(svc.publish_build, old, make_build()) is svc.PublishOutcome.STALE
    assert call(svc.record_failure, old, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.STALE
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_attempt == 2 and row.indexing_error is None
    assert chunks_of(document_id) == []
    assert call(svc.publish_build, new, make_build()) is svc.PublishOutcome.PUBLISHED


def test_publish_is_stale_when_the_source_text_changed_after_the_claim():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    execute("UPDATE documents SET extracted_text = extracted_text || ' changed' WHERE id = :id", id=document_id)
    assert call(svc.publish_build, claimed, make_build()) is svc.PublishOutcome.STALE
    assert chunks_of(document_id) == [] and row_of(document_id).indexing_status == "indexing"


def test_publish_is_stale_when_the_build_was_made_from_other_text():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    other = make_build(TEXT + " extra text")
    assert call(svc.publish_build, claimed, other) is svc.PublishOutcome.STALE
    assert chunks_of(document_id) == []


def test_publish_is_stale_when_the_source_version_changed():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    execute("UPDATE documents SET source_version = 2 WHERE id = :id", id=document_id)
    assert call(svc.publish_build, claimed, make_build()) is svc.PublishOutcome.STALE


def test_publish_and_failure_report_gone_for_a_deleted_document():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    execute("DELETE FROM documents WHERE id = :id", id=document_id)
    assert call(svc.publish_build, claimed, make_build()) is svc.PublishOutcome.GONE
    assert call(svc.record_failure, claimed, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.GONE
    assert chunks_of(document_id) == []


def test_publish_rolls_back_completely_when_a_chunk_is_rejected_by_the_database():
    document_id = make_document()
    publish(document_id, make_build(count=3))
    before = snapshot(document_id)
    claimed = reindex_and_claim(document_id)
    bad = make_build(count=4)
    last = bad.chunks[3]
    broken = ChunkRecord(
        chunk_index=last.chunk_index, start_offset=last.start_offset, end_offset=last.end_offset,
        page_start=0, page_end=0,                                             # violates ck_document_chunks_pages_valid
        text=last.text, span_sha256=last.span_sha256, embedding=last.embedding,
    )
    build = DocumentIndexBuild(
        source_sha256=bad.source_sha256, signature="sig-new", embedding_id=bad.embedding_id,
        chunks=bad.chunks[:3] + (broken,), stats=bad.stats,
    )
    with pytest.raises(IntegrityError):
        call(svc.publish_build, claimed, build)
    assert snapshot(document_id) == before                                      # old chunks and metadata intact
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexed_signature == "sig-v1"


def test_publish_rejects_a_build_without_chunks():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    empty = DocumentIndexBuild(
        source_sha256=sha256_text(TEXT), signature="s", embedding_id=EMBEDDING_ID, chunks=(),
        stats=IndexBuildStats(chunk_count=0, source_tokens=0, max_chunk_tokens=0, pages_available=False),
    )
    with pytest.raises(ValueError):
        call(svc.publish_build, claimed, empty)
    assert row_of(document_id).indexing_status == "indexing"


# ---------- record_failure ----------

def test_first_attempt_failure_marks_the_document_failed():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    assert call(svc.record_failure, claimed, svc.FailureReason.EMBEDDER_UNAVAILABLE) is svc.FailureOutcome.FAILED
    row = row_of(document_id)
    assert row.indexing_status == "failed" and row.indexing_error == "The embedding model is not available"
    assert row.indexing_lease_expires_at is None
    assert row.indexed_source_version is None and row.source_sha256 is None and row.indexed_at is None
    assert chunks_of(document_id) == []


def test_failed_reindex_restores_the_valid_publication_untouched():
    document_id = make_document()
    publish(document_id)
    before = snapshot(document_id)
    claimed = reindex_and_claim(document_id)
    assert call(svc.record_failure, claimed, svc.FailureReason.CHUNKING) is svc.FailureOutcome.RESTORED
    row = row_of(document_id)
    assert row.indexing_status == "indexed" and row.indexing_lease_expires_at is None
    assert row.indexing_error == svc.FailureReason.CHUNKING.value
    assert snapshot(document_id) == before                                      # byte-for-byte unchanged
    assert retrievable(document_id)


def test_failure_discards_a_publication_whose_text_changed_without_a_version_bump():
    document_id = make_document()
    publish(document_id)
    execute("UPDATE documents SET extracted_text = 'something else entirely' WHERE id = :id", id=document_id)
    claimed = reindex_and_claim(document_id)
    assert call(svc.record_failure, claimed, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.DISCARDED
    assert_nothing_published(document_id)
    assert row_of(document_id).indexing_status == "failed"


def test_failure_discards_a_publication_of_an_older_source_version():
    document_id = make_document()
    publish(document_id)
    execute("UPDATE documents SET source_version = 2 WHERE id = :id", id=document_id)
    claimed = reindex_and_claim(document_id)
    assert claimed.source_version == 2
    assert call(svc.record_failure, claimed, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.DISCARDED
    assert_nothing_published(document_id)


def test_failure_discards_a_publication_whose_chunks_are_missing():
    document_id = make_document()
    publish(document_id)
    execute("DELETE FROM document_chunks WHERE document_id = :id", id=document_id)
    claimed = reindex_and_claim(document_id)
    assert call(svc.record_failure, claimed, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.DISCARDED
    assert row_of(document_id).indexing_status == "failed"
    assert_nothing_published(document_id)


def assert_nothing_published(document_id):
    row = row_of(document_id)
    assert row.source_sha256 is None and row.indexed_source_version is None
    assert row.indexed_signature is None and row.indexed_embedding_id is None and row.indexed_at is None
    assert row.indexing_lease_expires_at is None
    assert chunks_of(document_id) == []
    assert not retrievable(document_id)


def test_record_failure_accepts_only_a_failure_reason():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    with pytest.raises(TypeError):
        call(svc.record_failure, claimed, "raw exception text with document content")
    row = row_of(document_id)
    assert row.indexing_status == "indexing" and row.indexing_error is None


# ---------- recovery ----------

def test_an_abandoned_attempt_is_recovered_through_lease_expiry():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    abandoned = claim(document_id)                       # the worker dies here: nothing is ever recorded
    assert call(svc.request_indexing, document_id) is False
    expire_lease(document_id)
    assert call(svc.request_indexing, document_id) is True
    assert row_of(document_id).indexing_status == "pending"
    recovered = claim(document_id)
    assert recovered.attempt == 2
    assert call(svc.record_failure, abandoned, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.STALE
    assert call(svc.publish_build, abandoned, make_build()) is svc.PublishOutcome.STALE
    assert call(svc.publish_build, recovered, make_build()) is svc.PublishOutcome.PUBLISHED


def test_a_database_error_inside_record_failure_rolls_back_and_leaves_the_lease_for_recovery():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    claimed = claim(document_id)
    before = row_of(document_id)
    with SessionLocal() as session:
        real_execute, calls = session.execute, {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:                          # the write that follows the row lock (execute #1)
                raise OperationalError("UPDATE documents", {}, Exception("connection lost"))
            return real_execute(*args, **kwargs)

        session.execute = flaky
        with pytest.raises(OperationalError):
            svc.record_failure(session, claimed, svc.FailureReason.CHUNKING)
    assert calls["n"] == 2
    assert row_of(document_id) == before                 # still 'indexing' with its lease: expiry will recover it
    expire_lease(document_id)
    assert claim(document_id).attempt == 2


# ---------- retrieval predicate ----------

def test_retrievable_filter_follows_publication_not_status():
    document_id = make_document()
    assert not retrievable(document_id)                                          # never indexed
    publish(document_id)
    assert retrievable(document_id)
    assert not retrievable(document_id, "another-model@revision")               # vectors from a different model
    call(svc.request_indexing, document_id)
    assert row_of(document_id).indexing_status == "pending" and retrievable(document_id)
    claim(document_id)
    assert row_of(document_id).indexing_status == "indexing" and retrievable(document_id)
    execute("UPDATE documents SET source_version = 2 WHERE id = :id", id=document_id)
    assert not retrievable(document_id)                                          # newer source than the index


def test_a_first_failure_is_not_retrievable():
    document_id = make_document()
    call(svc.request_indexing, document_id)
    call(svc.record_failure, claim(document_id), svc.FailureReason.UNEXPECTED)
    assert not retrievable(document_id)


# ---------- classification ----------

@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (IndexBuildError(NO_INDEXABLE_TEXT), svc.FailureReason.NO_INDEXABLE_TEXT),
        (IndexBuildError("anything else, with document text"), svc.FailureReason.BUILD_REJECTED),
        (ChunkingError("boom"), svc.FailureReason.CHUNKING),
        (OffsetMappingError("uncovered_character", 3), svc.FailureReason.OFFSET_MAPPING),
        (EmbeddingInputTooLongError(0, 600), svc.FailureReason.CHUNK_TOO_LONG),
        (EmbeddingInputError("boom"), svc.FailureReason.EMBEDDING_INVALID),
        (EmbeddingOutputError("boom"), svc.FailureReason.EMBEDDING_INVALID),
        (PageMappingError("boom"), svc.FailureReason.PAGE_MAPPING),
        (EmbedderUnavailableError("boom"), svc.FailureReason.EMBEDDER_UNAVAILABLE),
        (EmbedderError("boom"), svc.FailureReason.UNEXPECTED),
        (SQLAlchemyError("boom"), svc.FailureReason.DATABASE),
        (OperationalError("stmt", {}, Exception("boom")), svc.FailureReason.DATABASE),
        (RuntimeError("boom"), svc.FailureReason.UNEXPECTED),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else value.name,
)
def test_classify_failure_maps_exception_types_to_fixed_reasons(exception, reason):
    assert svc.classify_failure(exception) is reason


def test_classification_never_exposes_the_exception_text():
    secret = "SECRET-DOCUMENT-TEXT /home/alice/private.pdf"
    for exception in (
        IndexBuildError(secret), ChunkingError(secret), OffsetMappingError(secret), EmbeddingInputError(secret),
        EmbeddingOutputError(secret), PageMappingError(secret), EmbedderUnavailableError(secret),
        SQLAlchemyError(secret), RuntimeError(secret),
    ):
        reason = svc.classify_failure(exception)
        assert isinstance(reason, svc.FailureReason) and "SECRET" not in reason.value and "private" not in reason.value


def test_failure_reasons_are_safe_and_distinct():
    values = [reason.value for reason in svc.FailureReason]
    assert len(set(values)) == len(values)
    assert all(value.isascii() and 0 < len(value) <= 500 for value in values)
    assert svc.FailureReason.NO_INDEXABLE_TEXT.value == "No indexable text found in this document"


# ---------- transaction guard and invariant guard ----------

def test_service_functions_require_a_session_without_an_open_transaction():
    document_id = make_document()
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))                                        # SQLAlchemy autobegins here
        with pytest.raises(InvalidRequestError):
            svc.request_indexing(session, document_id)
    assert row_of(document_id).indexing_status is None


def test_only_the_extraction_step_writes_extracted_text():
    """Invariant I-1 tripwire: nothing else in the application assigns extracted_text.

    Any new writer must also bump source_version in the same UPDATE; add it to
    ALLOWED only together with that logic.
    """
    allowed = {"app/core/processing.py", "app/models/document.py"}
    assignment = re.compile(r"\bextracted_text\s*=(?!=)", re.IGNORECASE)
    offenders = []
    for folder in ("app", "scripts"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative in allowed:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.lstrip().startswith("#") and assignment.search(line):
                    offenders.append(f"{relative}:{number}")
    assert offenders == []
    processing = (PROJECT_ROOT / "app/core/processing.py").read_text(encoding="utf-8")
    assert len(assignment.findall(processing)) == 1                              # the single extraction write