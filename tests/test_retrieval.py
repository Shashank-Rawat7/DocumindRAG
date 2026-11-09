"""C6.2 retrieval: exact cosine search, ownership isolation and publication rules.

A  statement builder (compile only)        G  ranking, ties and top_k
B  argument handling                       H  integrity failures against the database
C  row conversion with fake rows           I  Unicode, offsets and page metadata.
D  ownership isolation                     J  SQL shape and transaction behaviour
E  field mapping and large document IDs.   K  reused conventions
F  lifecycle through the real C5 service

Every helper that touches the database opens a short-lived session or connection
and closes it before returning, so nothing holds a lock when conftest's cleanup
TRUNCATEs the tables.
"""
import dataclasses
import inspect
import math
import re
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import delete, event, insert, text, update
from sqlalchemy.dialects import postgresql

from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import chunking, embedder, indexing
from app.rag import indexing_service as svc
from app.rag import retrieval as r
from app.rag.chunking import sha256_text
from app.rag.embedder import EMBEDDING_DIM
from app.rag.indexing import ChunkRecord, DocumentIndexBuild, IndexBuildStats

EMBEDDING_ID = "test-model@" + "0" * 40
TEXT = " ".join(f"word{k:03d}" for k in range(80))                 # 639 ASCII characters
UNICODE_TEXT = (
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e "   # Hindi greeting
    "\U0001F600 caf\u00e9 e\u0301 \u2014 "       # emoji, precomposed e, combining accent, em dash
    "hello world \U0001F389 end of the text"
)
MAX_INT4 = 2_147_483_647
SOURCE_HASH = sha256_text("the whole document")


# ---------- helpers ----------

def direction(*pairs):
    """A unit float32 vector pointing along the given (index, weight) pairs."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for index, weight in pairs:
        vector[index] = weight
    return vector / np.linalg.norm(vector)


def zero_vector():
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)


Q = direction((0, 1))                                              # the default query


def spans(source, count):
    """``count`` consecutive (start, end) slices covering ``source``."""
    size = len(source) // count
    return [(i * size, (i + 1) * size if i < count - 1 else len(source)) for i in range(count)]


def execute(statement, **params):
    with engine.begin() as connection:
        connection.execute(text(statement), params)


def scalar(statement, **params):
    with engine.connect() as connection:
        return connection.execute(text(statement), params).scalar_one()


def call(function, *args, **kwargs):
    """Run one service call in its own fresh session."""
    with SessionLocal() as session:
        return function(session, *args, **kwargs)


def new_user():
    with SessionLocal() as session:
        user = User(email=f"user-{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        user_id = user.id
        session.commit()
    return user_id


def published_columns(source, **overrides):
    values = {
        "indexing_status": "indexed",
        "indexed_source_version": 1,
        "source_sha256": sha256_text(source),
        "indexed_signature": "sig",
        "indexed_embedding_id": EMBEDDING_ID,
        "indexed_at": datetime.now(UTC),
    }
    values.update(overrides)
    return values


def chunk_rows(document_id, source, vectors, pages):
    if not vectors:
        return []
    page_start, page_end = pages if pages else (None, None)
    rows = []
    for index, ((start, end), vector) in enumerate(zip(spans(source, len(vectors)), vectors)):
        piece = source[start:end]
        rows.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                start_offset=start,
                end_offset=end,
                page_start=page_start,
                page_end=page_end,
                text=piece,
                span_sha256=sha256_text(piece),
                embedding=vector,
            )
        )
    return rows


def seed_document(
    owner_id, source=TEXT, vectors=(), *, publish=True, pages=None, reverse_insert=False, **columns
):
    """Insert a completed document with one chunk per vector (equal slices of ``source``).

    publish=True also records a full publication. ``columns`` override any column.
    """
    values = {
        "filename": "a.txt",
        "owner_id": owner_id,
        "processing_status": "completed",
        "extracted_text": source,
    }
    if publish:
        values.update(published_columns(source))
    values.update(columns)
    with SessionLocal() as session:
        document = Document(**values)
        session.add(document)
        session.flush()
        document_id = document.id
        rows = chunk_rows(document_id, source, vectors, pages)
        if reverse_insert:
            rows.reverse()
        session.add_all(rows)
        session.commit()
    return document_id


def make_build(source, vectors, *, signature="sig-v1"):
    records = []
    for index, ((start, end), vector) in enumerate(zip(spans(source, len(vectors)), vectors)):
        piece = source[start:end]
        frozen = np.array(vector, dtype=np.float32)
        frozen.setflags(write=False)
        records.append(
            ChunkRecord(
                chunk_index=index,
                start_offset=start,
                end_offset=end,
                page_start=None,
                page_end=None,
                text=piece,
                span_sha256=sha256_text(piece),
                embedding=frozen,
            )
        )
    return DocumentIndexBuild(
        source_sha256=sha256_text(source),
        signature=signature,
        embedding_id=EMBEDDING_ID,
        chunks=tuple(records),
        stats=IndexBuildStats(
            chunk_count=len(records), source_tokens=len(records), max_chunk_tokens=1,
            pages_available=False,
        ),
    )


def reindex_and_claim(document_id):
    assert call(svc.request_indexing, document_id) is True
    claim = call(svc.claim_document, document_id)
    assert claim is not None
    return claim


def service_publish(document_id, vectors, source=TEXT, signature="sig-v1"):
    claim = reindex_and_claim(document_id)
    outcome = call(svc.publish_build, claim, make_build(source, vectors, signature=signature))
    assert outcome is svc.PublishOutcome.PUBLISHED
    return claim


def published_document(owner_id, vectors, source=TEXT):
    """A document published through the real service (request, claim, publish)."""
    document_id = seed_document(owner_id, source, publish=False)
    service_publish(document_id, vectors, source)
    return document_id


def expire_lease(document_id):
    execute(
        "UPDATE documents SET indexing_lease_expires_at = clock_timestamp() - interval '1 minute'"
        " WHERE id = :id",
        id=document_id,
    )


def retrieve(owner_id, query_vector=Q, **kwargs):
    kwargs.setdefault("embedding_id", EMBEDDING_ID)
    with SessionLocal() as session:
        return r.retrieve_chunks(session, owner_id=owner_id, query_vector=query_vector, **kwargs)


def ranked(chunks):
    return [(c.document_id, c.chunk_index) for c in chunks]


def chunk_id_of(document_id, chunk_index):
    return scalar(
        "SELECT id FROM document_chunks WHERE document_id = :d AND chunk_index = :i",
        d=document_id, i=chunk_index,
    )


@contextmanager
def capture_statements():
    captured = []

    def listener(connection, cursor, statement, parameters, context, executemany):
        captured.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", listener)
    try:
        yield captured
    finally:
        event.remove(engine, "before_cursor_execute", listener)


def fake_row(**overrides):
    """A result row with the 12 selected columns, valid unless overridden."""
    body = "hello world"
    values = {
        "chunk_id": 7, "document_id": 3, "chunk_index": 0, "start_offset": 0,
        "end_offset": len(body), "page_start": None, "page_end": None, "text": body,
        "span_sha256": sha256_text(body), "distance": 0.25, "source_version": 1,
        "source_sha256": SOURCE_HASH,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------- A. statement builder (compile only, no database) ----------

def build(**overrides):
    arguments = {
        "owner_id": 7, "query_vector": Q, "embedding_id": EMBEDDING_ID, "top_k": 13,
        "document_ids": None,
    }
    arguments.update(overrides)
    statement = r._build_retrieval_statement(**arguments)
    compiled = statement.compile(dialect=postgresql.dialect())
    return statement, compiled, " ".join(str(compiled).split())


def where_clause(flat):
    return flat.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0]


def scalar_params(compiled):
    return [v for v in compiled.params.values() if not isinstance(v, np.ndarray)]


@pytest.mark.parametrize("document_ids", [None, (1,), (1, 2, 3)])
def test_the_owner_predicate_is_always_in_the_base_where_clause(document_ids):
    _, compiled, flat = build(document_ids=document_ids)
    match = re.search(r"documents\.owner_id = %\((\w+)\)s", where_clause(flat))
    assert match is not None
    assert compiled.params[match.group(1)] == 7


def test_document_ids_only_narrow_the_result():
    _, compiled, flat = build(document_ids=(4, 5))
    where = where_clause(flat)
    assert "documents.id IN (" in where
    assert where.index("documents.owner_id") < where.index("documents.id IN (")
    assert (4, 5) in [tuple(v) for v in compiled.params.values() if isinstance(v, (tuple, list))]
    _, _, without = build(document_ids=None)
    assert "documents.id IN" not in where_clause(without)


def test_the_publication_predicate_and_embedding_id_are_in_the_where_clause():
    _, compiled, flat = build()
    where = where_clause(flat)
    assert "documents.indexed_source_version IS NOT NULL" in where
    assert "documents.indexed_source_version = documents.source_version" in where
    match = re.search(r"documents\.indexed_embedding_id = %\((\w+)\)s", where)
    assert match is not None
    assert compiled.params[match.group(1)] == EMBEDDING_ID


def test_ordering_is_distance_then_document_then_chunk_index_with_a_limit():
    _, compiled, flat = build()
    assert (
        "ORDER BY distance, document_chunks.document_id, document_chunks.chunk_index" in flat
    )
    assert "LIMIT" in flat
    assert 13 in scalar_params(compiled)


def test_one_distance_expression_and_one_vector_bind():
    _, compiled, flat = build(document_ids=(1, 2))
    assert flat.count("<=>") == 1
    assert flat.count("document_chunks.embedding") == 1
    assert re.search(r"document_chunks\.embedding(?! <=>)", flat) is None
    vector_names = [k for k, v in compiled.params.items() if isinstance(v, np.ndarray)]
    assert len(vector_names) == 1
    assert flat.count(f"%({vector_names[0]})s") == 1


def test_the_statement_never_mentions_extracted_text():
    _, _, flat = build(document_ids=(1,))
    assert "extracted_text" not in flat


def test_the_select_list_has_the_twelve_columns_in_dataclass_order():
    statement, _, _ = build()
    fields = [f.name for f in dataclasses.fields(r.RetrievedChunk)]
    expected = ["distance" if name == "score" else name for name in fields]
    assert len(fields) == 12
    assert list(statement.selected_columns.keys()) == expected

def test_source_version_is_selected_from_indexed_source_version():
    statement, _, _ = build()
    assert (
        statement.selected_columns["source_version"].element.name
        == "indexed_source_version"
    )


# ---------- B. argument handling ----------

def valid_arguments(**overrides):
    arguments = {"owner_id": 1, "query_vector": Q, "embedding_id": EMBEDDING_ID}
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize(
    "overrides",
    [
        {"owner_id": True},
        {"owner_id": "1"},
        {"top_k": 0},
        {"top_k": 21},
        {"top_k": True},
        {"embedding_id": ""},
        {"query_vector": Q.astype(np.float64)},
        {"query_vector": zero_vector()},
        {"document_ids": "12"},
        {"document_ids": [True]},
        {"document_ids": list(range(101))},
    ],
    ids=[
        "owner-bool", "owner-str", "top_k-0", "top_k-21", "top_k-bool", "embedding-empty",
        "vector-float64", "vector-zero", "ids-str", "ids-bool", "ids-too-many",
    ],
)
def test_invalid_arguments_raise_before_the_database_is_touched(overrides):
    with pytest.raises(r.RetrievalRequestError):
        r.retrieve_chunks(None, **valid_arguments(**overrides))      # db=None: any use would fail


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_document_ids_return_an_empty_list_without_using_the_database(empty):
    assert r.retrieve_chunks(None, **valid_arguments(document_ids=empty)) == []


def test_an_invalid_top_k_is_still_rejected_when_document_ids_is_empty():
    with pytest.raises(r.RetrievalRequestError):
        r.retrieve_chunks(None, **valid_arguments(document_ids=[], top_k=21))


def test_empty_document_ids_issue_no_statement_at_all():
    with SessionLocal() as session, capture_statements() as captured:
        assert r.retrieve_chunks(session, **valid_arguments(document_ids=[])) == []
    assert captured == []


# ---------- C. row conversion (fake rows, no database) ----------

def test_a_valid_row_becomes_a_retrieved_chunk():
    chunk = r._row_to_chunk(fake_row())
    assert chunk == r.RetrievedChunk(
        chunk_id=7, document_id=3, chunk_index=0, start_offset=0, end_offset=11,
        page_start=None, page_end=None, text="hello world",
        span_sha256=sha256_text("hello world"), score=0.75, source_version=1,
        source_sha256=SOURCE_HASH,
    )


@pytest.mark.parametrize("distance", [0.0, 1.0, 2.0, 0.123456789])
def test_score_is_one_minus_the_distance(distance):
    assert r._row_to_chunk(fake_row(distance=distance)).score == 1.0 - distance


def test_pages_are_carried_through():
    chunk = r._row_to_chunk(fake_row(page_start=2, page_end=3))
    assert (chunk.page_start, chunk.page_end) == (2, 3)


def test_a_length_mismatch_is_an_integrity_failure():
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(end_offset=12))
    assert (caught.value.chunk_id, caught.value.check) == (7, "length")


def test_a_span_hash_mismatch_is_an_integrity_failure():
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(span_sha256=sha256_text("other")))
    assert (caught.value.chunk_id, caught.value.check) == (7, "sha256")


@pytest.mark.parametrize(
    "stored",
    [None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 123, b"a" * 64],
    ids=["none", "empty", "short", "long", "uppercase", "non-hex", "int", "bytes"],
)
def test_a_null_or_malformed_stored_source_hash_is_an_integrity_failure(stored):
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(source_sha256=stored))
    assert (caught.value.chunk_id, caught.value.check) == (7, "source_sha256")
    assert not isinstance(caught.value, r.RetrievalRequestError)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), float("-inf"), None])
def test_a_non_finite_distance_is_an_integrity_failure(distance):
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(distance=distance))
    assert (caught.value.chunk_id, caught.value.check) == (7, "distance")


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"end_offset": 99, "span_sha256": sha256_text("x"), "source_sha256": None, "distance": math.nan}, "length"),
        ({"span_sha256": sha256_text("x"), "source_sha256": None, "distance": math.nan}, "sha256"),
        ({"source_sha256": None, "distance": math.nan}, "source_sha256"),
        ({"distance": math.nan}, "distance"),
    ],
    ids=["length-first", "sha256-second", "source-third", "distance-last"],
)
def test_checks_run_in_the_fixed_order(overrides, expected):
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(**overrides))
    assert caught.value.check == expected


def test_integrity_messages_never_contain_the_text():
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        r._row_to_chunk(fake_row(text="SECRET-TEXT", end_offset=11, span_sha256="0" * 64))
    assert "SECRET" not in str(caught.value)


# ---------- D. ownership isolation ----------

OTHER_BEST = [direction((0, 1))]                                   # identical to the query
OWN_WEAKER = [direction((0, 1), (1, 2))]


def test_a_user_passing_only_another_users_document_ids_gets_nothing():
    alice, bob = new_user(), new_user()
    seed_document(alice, vectors=OWN_WEAKER)
    bobs = seed_document(bob, vectors=OTHER_BEST)                  # Bob's chunk is the best match
    assert retrieve(alice, document_ids=[bobs]) == []


def test_mixed_own_and_foreign_ids_return_only_own_chunks():
    alice, bob = new_user(), new_user()
    alices = seed_document(alice, vectors=OWN_WEAKER)
    bobs = seed_document(bob, vectors=OTHER_BEST)
    mixed = retrieve(alice, document_ids=[alices, bobs])
    assert {c.document_id for c in mixed} == {alices}
    assert mixed == retrieve(alice)


def test_identical_vectors_and_texts_never_cross_owners():
    alice, bob = new_user(), new_user()
    vectors = [direction((0, 1)), direction((0, 1), (1, 1))]
    alices = seed_document(alice, vectors=vectors)
    bobs = seed_document(bob, vectors=vectors)
    for_alice, for_bob = retrieve(alice), retrieve(bob)
    assert {c.document_id for c in for_alice} == {alices}
    assert {c.document_id for c in for_bob} == {bobs}
    assert not {c.chunk_id for c in for_alice} & {c.chunk_id for c in for_bob}
    assert [c.text for c in for_alice] == [c.text for c in for_bob]
    assert retrieve(alice, document_ids=[bobs]) == []
    assert retrieve(bob, document_ids=[alices]) == []


def test_foreign_and_nonexistent_ids_are_indistinguishable():
    alice, bob = new_user(), new_user()
    alices = seed_document(alice, vectors=OWN_WEAKER)
    bobs = seed_document(bob, vectors=OTHER_BEST)
    missing = 999_999
    assert retrieve(alice, document_ids=[bobs]) == retrieve(alice, document_ids=[missing]) == []
    assert retrieve(alice, document_ids=[alices, bobs]) == retrieve(
        alice, document_ids=[alices, missing]
    )


def test_a_document_deleted_after_indexing_is_gone_and_others_are_unaffected():
    alice = new_user()
    doomed = published_document(alice, [direction((0, 1))])
    kept = published_document(alice, [direction((0, 1), (1, 1))])
    assert {c.document_id for c in retrieve(alice)} == {doomed, kept}
    execute("DELETE FROM documents WHERE id = :id", id=doomed)
    assert {c.document_id for c in retrieve(alice)} == {kept}
    assert retrieve(alice, document_ids=[doomed]) == []


def test_a_deleted_owner_returns_nothing_and_other_owners_are_unaffected():
    alice, bob = new_user(), new_user()
    seed_document(alice, vectors=OTHER_BEST)
    seed_document(bob, vectors=OWN_WEAKER)
    before = retrieve(bob)
    assert len(before) == 1 and len(retrieve(alice)) == 1
    execute("DELETE FROM users WHERE id = :id", id=alice)
    assert retrieve(alice) == []
    assert retrieve(bob) == before


def test_another_users_reindex_failure_does_not_change_my_results():
    alice, bob = new_user(), new_user()
    seed_document(alice, vectors=OWN_WEAKER)
    bobs = published_document(bob, OTHER_BEST)
    before = retrieve(alice)
    claim = reindex_and_claim(bobs)
    execute("UPDATE documents SET extracted_text = 'something else entirely' WHERE id = :id", id=bobs)
    assert call(svc.record_failure, claim, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.DISCARDED
    assert retrieve(alice) == before
    assert retrieve(bob) == []


def test_each_user_in_a_mixed_corpus_sees_exactly_their_own_chunks():
    vectors = [direction((0, 1)), direction((0, 1), (1, 1))]
    owned = {}
    for _ in range(3):
        user = new_user()
        owned[user] = {seed_document(user, vectors=vectors) for _ in range(2)}
    for user, documents in owned.items():
        found = retrieve(user)
        assert len(found) == 4
        assert {c.document_id for c in found} == documents


# ---------- E. publication filter (seeded) ----------

def test_a_never_indexed_document_and_unpublished_chunks_are_excluded():
    owner = new_user()
    seed_document(owner, publish=False)                            # no chunks, no publication
    seed_document(owner, vectors=[Q], publish=False)               # chunks but no publication metadata
    assert retrieve(owner) == []


def test_a_stale_source_version_is_excluded_not_reported_as_the_old_version():
    owner = new_user()
    document_id = seed_document(owner, vectors=[Q])
    assert len(retrieve(owner)) == 1
    execute("UPDATE documents SET source_version = 2 WHERE id = :id", id=document_id)
    assert retrieve(owner) == []


def test_the_published_version_is_what_gets_reported():
    owner = new_user()
    seed_document(owner, vectors=[Q], source_version=2, indexed_source_version=2)
    chunk = retrieve(owner)[0]
    assert chunk.source_version == 2
    assert chunk.source_sha256 == sha256_text(TEXT)


def test_vectors_from_another_embedding_id_are_excluded():
    owner = new_user()
    seed_document(owner, vectors=[Q])
    assert len(retrieve(owner)) == 1
    assert retrieve(owner, embedding_id="other-model@" + "1" * 40) == []


def test_every_field_comes_from_the_right_column():
    owner = new_user()
    document_id = seed_document(
        owner, vectors=[direction((0, 1)), direction((1, 1))], pages=(2, 3)
    )
    result = retrieve(owner)
    assert isinstance(result, list)
    assert all(isinstance(c, r.RetrievedChunk) for c in result)
    first, second = result
    start, end = spans(TEXT, 2)[0]
    assert first == r.RetrievedChunk(
        chunk_id=chunk_id_of(document_id, 0), document_id=document_id, chunk_index=0,
        start_offset=start, end_offset=end, page_start=2, page_end=3, text=TEXT[start:end],
        span_sha256=sha256_text(TEXT[start:end]), score=first.score, source_version=1,
        source_sha256=sha256_text(TEXT),
    )
    assert first.score == pytest.approx(1.0, abs=1e-6)
    assert second.score == pytest.approx(0.0, abs=1e-6)


def test_the_largest_integer_document_id_works():
    owner = new_user()
    seed_document(owner, vectors=[Q], id=MAX_INT4)
    assert [c.document_id for c in retrieve(owner, document_ids=[MAX_INT4])] == [MAX_INT4]
    assert retrieve(owner, document_ids=[MAX_INT4 - 1]) == []


# ---------- F. lifecycle through the real C5 service ----------

OLD = [direction((0, 1)), direction((0, 1), (1, 1)), direction((0, 1), (1, 3))]


def test_a_first_failed_indexing_is_not_retrievable():
    owner = new_user()
    document_id = seed_document(owner, publish=False)
    claim = reindex_and_claim(document_id)
    assert call(svc.record_failure, claim, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.FAILED
    assert retrieve(owner) == []


@pytest.mark.parametrize(
    "state, status",
    [("pending", "pending"), ("running", "indexing"), ("expired", "indexing")],
)
def test_old_chunks_stay_retrievable_while_a_reindex_is_pending_or_running(state, status):
    owner = new_user()
    document_id = published_document(owner, OLD)
    old = retrieve(owner)
    assert len(old) == 3
    assert call(svc.request_indexing, document_id) is True
    if state != "pending":
        assert call(svc.claim_document, document_id) is not None
    if state == "expired":
        expire_lease(document_id)
    assert scalar("SELECT indexing_status FROM documents WHERE id = :id", id=document_id) == status
    assert retrieve(owner) == old


def test_a_failed_reindex_with_a_valid_publication_keeps_serving_the_old_chunks():
    owner = new_user()
    document_id = published_document(owner, OLD)
    old = retrieve(owner)
    claim = reindex_and_claim(document_id)
    assert call(svc.record_failure, claim, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.RESTORED
    assert scalar("SELECT indexing_status FROM documents WHERE id = :id", id=document_id) == "indexed"
    assert retrieve(owner) == old


def test_a_successful_reindex_replaces_the_old_chunks_completely():
    owner = new_user()
    document_id = published_document(owner, [direction((0, 1)), direction((1, 1))])
    old = retrieve(owner)
    claim = reindex_and_claim(document_id)
    new_vectors = [direction((2, 1)), direction((3, 1)), direction((0, 1), (2, 1))]
    published = call(svc.publish_build, claim, make_build(TEXT, new_vectors, signature="sig-v2"))
    assert published is svc.PublishOutcome.PUBLISHED
    new = retrieve(owner)
    assert [c.chunk_index for c in new] == [2, 0, 1]
    assert not {c.chunk_id for c in old} & {c.chunk_id for c in new}
    assert new[0].score == pytest.approx(1 / math.sqrt(2), abs=1e-6)


def test_a_failed_reindex_with_an_invalid_publication_discards_the_chunks():
    owner = new_user()
    document_id = published_document(owner, OLD)
    execute("UPDATE documents SET extracted_text = 'something else entirely' WHERE id = :id", id=document_id)
    claim = reindex_and_claim(document_id)
    assert call(svc.record_failure, claim, svc.FailureReason.UNEXPECTED) is svc.FailureOutcome.DISCARDED
    assert retrieve(owner) == []
    assert scalar("SELECT count(*) FROM document_chunks WHERE document_id = :id", id=document_id) == 0


def test_a_manual_uncommitted_chunk_swap_is_invisible_then_fully_visible_on_commit():
    """PostgreSQL visibility, not publish_build().

    A second connection performs by hand the same kinds of writes publish_build makes
    (delete the old chunks, insert the new ones, update the document row) inside one
    open transaction. Retrieval, being one SELECT, must see the whole old set until
    that transaction commits and the whole new set afterwards. The service's own
    checks (lease, attempt, source hash) are not exercised here.
    """
    owner = new_user()
    document_id = published_document(owner, OLD)                    # starting state: the real service
    old = retrieve(owner)
    assert len(old) == 3
    build_v2 = make_build(
        TEXT, [direction((0, 1)), direction((0, 1), (1, 1))], signature="sig-v2"
    )
    rows = [
        {
            "document_id": document_id, "chunk_index": c.chunk_index,
            "start_offset": c.start_offset, "end_offset": c.end_offset,
            "page_start": c.page_start, "page_end": c.page_end, "text": c.text,
            "span_sha256": c.span_sha256, "embedding": c.embedding,
        }
        for c in build_v2.chunks
    ]
    # Hand-written replacement transaction on its own connection (not publish_build).
    connection = engine.connect()
    transaction = connection.begin()
    try:
        connection.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        connection.execute(insert(DocumentChunk), rows)
        connection.execute(
            update(Document).where(Document.id == document_id).values(indexed_signature="sig-v2")
        )
        during = retrieve(owner)  # the manual transaction is still open
        transaction.commit()
    finally:
        if transaction.is_active:
            transaction.rollback()
        connection.close()
    after = retrieve(owner)
    assert during == old
    assert len(after) == 2
    assert not {c.chunk_id for c in old} & {c.chunk_id for c in after}


# ---------- G. ranking, ties and top_k ----------

def test_exact_cosine_ranking_with_scores():
    owner = new_user()
    document_id = seed_document(
        owner,
        vectors=[
            direction((1, 1)),                       # chunk 0: orthogonal      -> 0
            direction((0, 1)),                       # chunk 1: same direction  -> 1
            direction((0, 1), (1, 1)),               # chunk 2: 45 degrees      -> 0.7071
            direction((0, -1)),                      # chunk 3: opposite        -> -1
        ],
    )
    result = retrieve(owner)
    assert ranked(result) == [(document_id, 1), (document_id, 2), (document_id, 0), (document_id, 3)]
    assert [c.score for c in result] == pytest.approx([1.0, 1 / math.sqrt(2), 0.0, -1.0], abs=1e-6)
    assert [c.score for c in result] == sorted((c.score for c in result), reverse=True)


def test_ranking_interleaves_documents_by_score():
    owner = new_user()
    first = seed_document(owner, vectors=[direction((0, 1), (1, 1)), direction((0, 1), (1, 3))])
    second = seed_document(owner, vectors=[direction((0, 1), (1, 2)), direction((0, 1), (1, 4))])
    assert ranked(retrieve(owner)) == [(first, 0), (second, 0), (first, 1), (second, 1)]


def test_exact_ties_are_ordered_by_document_then_chunk_index_whatever_the_insert_order():
    owner = new_user()
    same = [Q, Q, Q]
    first = seed_document(owner, vectors=same, reverse_insert=True)
    second = seed_document(owner, vectors=same, reverse_insert=True)
    expected = [(first, 0), (first, 1), (first, 2), (second, 0), (second, 1)]
    runs = [ranked(retrieve(owner, top_k=5)) for _ in range(3)]
    assert runs == [expected] * 3


GRADED = [direction((0, 1), (1, i)) for i in range(25)]            # strictly decreasing similarity


@pytest.mark.parametrize("kwargs, expected", [({"top_k": 1}, 1), ({}, 5), ({"top_k": 20}, 20)])
def test_top_k_returns_exactly_the_k_best(kwargs, expected):
    owner = new_user()
    document_id = seed_document(owner, vectors=GRADED)
    result = retrieve(owner, **kwargs)
    assert ranked(result) == [(document_id, i) for i in range(expected)]


def test_fewer_chunks_than_top_k_returns_all_of_them():
    owner = new_user()
    seed_document(owner, vectors=GRADED[:3])
    assert len(retrieve(owner)) == 3


# ---------- H. integrity failures against the database ----------

THREE = [direction((0, 1)), direction((0, 1), (1, 1)), direction((0, 1), (1, 3))]


@pytest.mark.parametrize("tampered_rank", [0, 1, 2])
def test_an_integrity_failure_on_any_result_returns_no_partial_list(tampered_rank):
    owner = new_user()
    document_id = seed_document(owner, vectors=THREE)
    assert [c.chunk_index for c in retrieve(owner)] == [0, 1, 2]    # ranked best first
    victim = chunk_id_of(document_id, tampered_rank)
    execute("UPDATE document_chunks SET text = repeat('x', length(text)) WHERE id = :id", id=victim)
    result = None
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        result = retrieve(owner)
    assert result is None
    assert (caught.value.chunk_id, caught.value.check) == (victim, "sha256")


def test_a_wrong_stored_span_hash_fails_the_whole_call():
    owner = new_user()
    document_id = seed_document(owner, vectors=THREE)
    victim = chunk_id_of(document_id, 2)
    execute(
        "UPDATE document_chunks SET span_sha256 = :h WHERE id = :id",
        h=sha256_text("something else"), id=victim,
    )
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        retrieve(owner)
    assert (caught.value.chunk_id, caught.value.check) == (victim, "sha256")


def require_nan_for_a_zero_vector():
    zero = "[" + ",".join(["0"] * EMBEDDING_DIM) + "]"
    one = "[1" + ",0" * (EMBEDDING_DIM - 1) + "]"
    observed = scalar(f"SELECT CAST('{zero}' AS vector) <=> CAST('{one}' AS vector)")
    if observed is None or not math.isnan(observed):
        pytest.skip(
            f"this pgvector build returns {observed!r} for a zero vector, not NaN; the "
            "non-finite branch is covered by the fake-row tests"
        )


def test_a_stored_zero_vector_gives_a_non_finite_distance_and_fails_closed():
    require_nan_for_a_zero_vector()
    owner = new_user()
    document_id = seed_document(owner, vectors=[zero_vector()])
    result = None
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        result = retrieve(owner)
    assert result is None
    assert (caught.value.chunk_id, caught.value.check) == (chunk_id_of(document_id, 0), "distance")


def test_a_non_finite_distance_on_the_last_ranked_row_returns_no_partial_list():
    require_nan_for_a_zero_vector()                                 # NaN sorts after every real distance
    owner = new_user()
    document_id = seed_document(
        owner, vectors=[direction((0, 1)), direction((0, 1), (1, 1)), zero_vector()]
    )
    result = None
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        result = retrieve(owner)
    assert result is None
    assert (caught.value.chunk_id, caught.value.check) == (chunk_id_of(document_id, 2), "distance")


# ---------- I. Unicode and source integrity ----------

def test_offsets_count_code_points_and_both_hashes_hold_for_unicode_text():
    assert scalar("SHOW server_encoding") == "UTF8"
    assert any(ord(ch) > 0xFFFF for ch in UNICODE_TEXT)             # astral characters are present
    owner = new_user()
    document_id = seed_document(owner, UNICODE_TEXT, vectors=[Q, Q, Q, Q])
    result = retrieve(owner)
    assert len(result) == 4
    for chunk in result:
        assert UNICODE_TEXT[chunk.start_offset:chunk.end_offset] == chunk.text
        assert sha256_text(chunk.text) == chunk.span_sha256
        assert sha256_text(UNICODE_TEXT) == chunk.source_sha256
    assert scalar(
        "SELECT bool_and(substr(d.extracted_text, c.start_offset + 1,"
        " c.end_offset - c.start_offset) = c.text)"
        " FROM document_chunks c JOIN documents d ON d.id = c.document_id WHERE d.id = :id",
        id=document_id,
    ) is True


def test_pages_are_null_when_the_page_mapping_was_unavailable():
    owner = new_user()
    seed_document(owner, vectors=[Q])
    chunk = retrieve(owner)[0]
    assert (chunk.page_start, chunk.page_end) == (None, None)


# ---------- J. SQL shape and transaction behaviour ----------

def test_one_select_per_call_with_the_vector_bound_once_and_no_extra_data_loaded():
    owner = new_user()
    seed_document(owner, vectors=[direction((0, 1)), direction((1, 1))])
    retrieve(owner)                                                 # warm-up: connection set-up happens here
    with capture_statements() as captured:
        result = retrieve(owner)
    assert len(result) == 2
    assert len(captured) == 1
    statement, parameters = captured[0]
    flat = " ".join(statement.split())
    assert flat.upper().startswith("SELECT")
    assert flat.upper().count("SELECT ") == 1
    assert "extracted_text" not in flat
    assert flat.count("document_chunks.embedding") == 1
    assert re.search(r"document_chunks\.embedding(?! <=>)", flat) is None
    assert flat.count("<=>") == 1
    assert isinstance(parameters, dict)
    vector_names = [
        k for k, v in parameters.items() if isinstance(v, str) and v.startswith("[")
    ]
    assert len(vector_names) == 1
    assert flat.count(f"%({vector_names[0]})s") == 1


def test_retrieval_does_not_commit_roll_back_or_flush():
    owner = new_user()
    seed_document(owner, vectors=[Q])
    events = []
    with SessionLocal() as session:
        pending = User(email=f"pending-{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(pending)
        event.listen(session, "before_flush", lambda *args: events.append("flush"))
        event.listen(session, "after_commit", lambda *args: events.append("commit"))
        event.listen(session, "after_soft_rollback", lambda *args: events.append("rollback"))
        result = r.retrieve_chunks(
            session, owner_id=owner, query_vector=Q, embedding_id=EMBEDDING_ID
        )
        assert len(result) == 1
        assert events == []
        assert pending in session.new                               # still pending: nothing was flushed
        session.rollback()


def test_retrieval_reads_inside_the_callers_transaction_and_leaves_it_open():
    owner = new_user()
    with SessionLocal() as session:
        document = Document(
            filename="uncommitted.txt", owner_id=owner, processing_status="completed",
            extracted_text=TEXT, **published_columns(TEXT),
        )
        session.add(document)
        session.flush()
        document_id = document.id
        session.add_all(chunk_rows(document_id, TEXT, [Q], None))
        session.flush()
        result = r.retrieve_chunks(
            session, owner_id=owner, query_vector=Q, embedding_id=EMBEDDING_ID
        )
        assert [c.document_id for c in result] == [document_id]     # sees uncommitted rows: no rollback
        assert session.in_transaction()                             # and nothing was committed or ended
        session.rollback()
    assert retrieve(owner) == []                                    # nothing was committed behind our back


# ---------- K. reused conventions ----------

def test_retrieval_reuses_the_existing_helpers_and_constants():
    assert r.sha256_text is chunking.sha256_text
    assert r.retrievable_filter is svc.retrievable_filter
    assert r.NORM_TOLERANCE is embedder.NORM_TOLERANCE
    assert r.EMBEDDING_DIM == embedder.EMBEDDING_DIM
    assert "hashlib" not in inspect.getsource(r)


def test_the_source_hash_is_the_one_the_real_publish_path_writes():
    assert indexing.source_sha256(TEXT) == sha256_text(TEXT)
    owner = new_user()
    document_id = published_document(owner, [Q])
    chunk = retrieve(owner)[0]
    stored = scalar("SELECT source_sha256 FROM documents WHERE id = :id", id=document_id)
    assert chunk.source_sha256 == stored == indexing.source_sha256(TEXT)
    assert chunk.span_sha256 == sha256_text(chunk.text)