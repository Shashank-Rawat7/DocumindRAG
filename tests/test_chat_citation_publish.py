"""C9.4b tests: citation persistence in publish_success and fail_request_after_generation.

Runs against documind_test. Attempts come from the real start_attempt and mark_llm_started;
documents and chunks are real rows; retrieval results are built from those rows, so the
retrieved chunk and the database row agree unless a test makes them disagree. Failures are
injected with temporary triggers; locking is observed through real concurrent sessions.
"""

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError

import app.rag.chat_attempt_lifecycle as lifecycle_module
import app.rag.chat_request_finalization as finalization_module
from app.core.database import SessionLocal, engine
from app.rag.chat_attempt_lifecycle import (
    AttemptLifecycleArgumentError,
    expire_stale_attempt,
    mark_llm_started,
)
from app.rag.chat_attempt_start import AttemptStartOutcome, start_attempt
from app.rag.chat_citations import Citation
from app.rag.chat_request_finalization import (
    FailAfterGenerationOutcome,
    FailAfterGenerationResult,
    FailOutcome,
    FinalizationArgumentError,
    PublishOutcome,
    RequestResult,
    fail_request,
    fail_request_after_generation,
    get_request_result,
    publish_success,
)
from app.rag.retrieval import RetrievedChunk

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
LIMIT = 1_000_000
QUESTION = "What is DocuMind?"
ANSWER = "A RAG service [1][2]."
CODE = "internal_error"
SOURCE_SHA = "b" * 64
TABLES = (
    "conversations",
    "messages",
    "chat_requests",
    "chat_request_attempts",
    "message_citations",
)
_counter = [0]
P = PublishOutcome
F = FailAfterGenerationOutcome
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _not_near_utc_midnight():
    now = datetime.now(UTC)
    if now.hour == 23 and now.minute == 59 and now.second >= 30:
        pytest.skip("too close to UTC midnight")


# ---------------------------------------------------------------- helpers


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def make_user(email="alice@example.com"):
    return execute(
        "INSERT INTO users (email, hashed_password) VALUES (:e, 'x') RETURNING id",
        e=email,
    ).scalar_one()


def make_conversation(owner, messages=0):
    cid = execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=owner
    ).scalar_one()
    for seq in range(1, messages + 1):
        execute(
            "INSERT INTO messages (conversation_id, seq, role, content)"
            " VALUES (:c, :s, :r, :t)",
            c=cid,
            s=seq,
            r="user" if seq % 2 else "assistant",
            t=f"old {seq}",
        )
    return cid


def make_request(owner, conversation_id=None):
    _counter[0] += 1
    return execute(
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status,"
        " creates_conversation, conversation_id)"
        " VALUES (:o, :k, :h, 'in_progress', :n, :c) RETURNING id",
        o=owner,
        k=f"k{_counter[0]}",
        h="a" * 64,
        n=conversation_id is None,
        c=conversation_id,
    ).scalar_one()


def make_document(owner, source_version=1):
    return execute(
        "INSERT INTO documents (filename, owner_id, source_version, source_sha256,"
        " indexing_status, indexed_source_version, indexed_signature,"
        " indexed_embedding_id, indexed_at)"
        " VALUES ('d.txt', :o, :v, :s, 'indexed', :v, 'sig', 'emb', now()) RETURNING id",
        o=owner,
        v=source_version,
        s=SOURCE_SHA,
    ).scalar_one()


def make_chunk(document_id, index, pages=None):
    content = f"chunk number {index}"
    start = index * 100
    return execute(
        "INSERT INTO document_chunks (document_id, chunk_index, start_offset, end_offset,"
        " page_start, page_end, text, span_sha256, embedding)"
        " VALUES (:d, :i, :s, :e, :ps, :pe, :t, :h,"
        " array_fill(0::real, ARRAY[384])::vector(384)) RETURNING id",
        d=document_id,
        i=index,
        s=start,
        e=start + len(content),
        ps=pages[0] if pages else None,
        pe=pages[1] if pages else None,
        t=content,
        h=f"{index:064x}",
    ).scalar_one()


def retrieved(chunk_id):
    """The RetrievedChunk that retrieval would have produced for this row."""
    row = execute(
        "SELECT c.id, c.document_id, c.chunk_index, c.start_offset, c.end_offset,"
        " c.page_start, c.page_end, c.text, c.span_sha256, d.indexed_source_version,"
        " d.source_sha256 FROM document_chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE c.id = :c",
        c=chunk_id,
    ).one()
    return RetrievedChunk(
        chunk_id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        page_start=row.page_start,
        page_end=row.page_end,
        text=row.text,
        span_sha256=row.span_sha256,
        score=0.9,
        source_version=row.indexed_source_version,
        source_sha256=row.source_sha256,
    )


def cite(chunks, markers=None):
    markers = markers or range(1, len(chunks) + 1)
    return tuple(
        Citation(ordinal=i, marker=marker, chunk=chunk)
        for i, (marker, chunk) in enumerate(zip(markers, chunks, strict=True), start=1)
    )


class World:
    """An owner's document with three indexed chunks, as retrieval would return them."""

    def __init__(self, owner):
        self.owner = owner
        self.document = make_document(owner)
        self.ids = [make_chunk(self.document, i) for i in range(3)]
        self.chunks = [retrieved(i) for i in self.ids]


def call(function, *args, **kwargs):
    with SessionLocal() as db:
        return function(db, *args, **kwargs)


def ready(owner, conversation_id=None, reserved=500, started=True, request_id=None):
    request_id = request_id or make_request(owner, conversation_id)
    result = call(
        start_attempt,
        owner,
        request_id,
        reserved_tokens=reserved,
        daily_limit_tokens=LIMIT,
        lease_seconds=600,
    )
    assert result.outcome is AttemptStartOutcome.STARTED, result
    if started:
        call(mark_llm_started, result.attempt_id)
    return request_id, result.attempt_id


def publish(owner, request_id, attempt_id, citations=(), retrieved_chunks=(), **kw):
    arguments = {
        "user_content": QUESTION,
        "assistant_content": ANSWER,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
    } | kw
    with SessionLocal() as db:
        return publish_success(db, owner, request_id, attempt_id, **arguments)


def publish_ok(owner, rid, aid, world, picks=(0, 1), **kw):
    chunks = [world.chunks[i] for i in picks]
    return publish(owner, rid, aid, cite(chunks), tuple(world.chunks), **kw)


def fail_gen(owner, rid, aid, code=CODE, **usage):
    with SessionLocal() as db:
        return fail_request_after_generation(db, owner, rid, aid, code, **usage)


def expire_lease(attempt_id):
    execute(
        "UPDATE chat_request_attempts SET lease_expires_at = clock_timestamp()"
        " - interval '1 second' WHERE id = :i",
        i=attempt_id,
    )


def snapshot():
    with engine.begin() as conn:
        return {
            table: [
                tuple(r)
                for r in conn.execute(text(f"SELECT * FROM {table} ORDER BY id"))
            ]
            for table in TABLES
        }


def count(table):
    return execute(f"SELECT count(*) FROM {table}").scalar_one()


def request_row(request_id):
    return execute("SELECT * FROM chat_requests WHERE id = :i", i=request_id).one()


def attempt_row(attempt_id):
    return execute(
        "SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id
    ).one()


def citation_rows(message_id=None):
    sql = (
        "SELECT ordinal, marker, chunk_id, document_id, chunk_index, start_offset,"
        " end_offset, page_start, page_end, span_sha256, source_version, source_sha256"
        " FROM message_citations"
    )
    if message_id is not None:
        return [
            tuple(r)
            for r in execute(
                sql + " WHERE message_id = :m ORDER BY ordinal", m=message_id
            )
        ]
    return [tuple(r) for r in execute(sql + " ORDER BY message_id, ordinal")]


def untouched(rid, aid):
    return (
        attempt_row(aid).status == "running"
        and request_row(rid).status == "in_progress"
    )


@contextmanager
def boom_trigger(table, when, event_name):
    execute(
        "CREATE FUNCTION c94_boom() RETURNS trigger LANGUAGE plpgsql AS"
        " $$ BEGIN RAISE EXCEPTION 'c94 injected failure'; END $$"
    )
    execute(
        f"CREATE TRIGGER c94_boom BEFORE {event_name} ON {table} FOR EACH ROW"
        f" WHEN ({when}) EXECUTE FUNCTION c94_boom()"
    )
    try:
        yield
    finally:
        execute(f"DROP TRIGGER c94_boom ON {table}")
        execute("DROP FUNCTION c94_boom()")


def run_threads(n, work):
    barrier = threading.Barrier(n)

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return work(db, i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))


def wait_for_lock_wait(pattern, timeout=15):
    """Block until some session is waiting on a lock in a statement matching pattern."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiting = execute(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
            " AND query ILIKE :p AND pid <> pg_backend_pid()",
            p=pattern,
        ).scalar_one()
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError(f"nobody is waiting on a lock for {pattern!r}")


@pytest.fixture
def owner():
    return make_user()


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


@pytest.fixture
def world(owner):
    return World(owner)


# ---------------------------------------------------------------- valid citations


@pytest.mark.parametrize("existing_conversation", [False, True])
def test_citations_persist_with_the_message_pair_and_match_the_chunks(
    owner, world, existing_conversation
):
    cid = make_conversation(owner, messages=2) if existing_conversation else None
    rid, aid = ready(owner, conversation_id=cid)
    result = publish_ok(
        owner, rid, aid, world, picks=(1, 0), prompt_tokens=90, completion_tokens=10
    )
    assert result.outcome is P.PUBLISHED
    expected = []
    for ordinal, chunk in enumerate((world.chunks[1], world.chunks[0]), start=1):
        expected.append(
            (
                ordinal,
                ordinal,
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.start_offset,
                chunk.end_offset,
                chunk.page_start,
                chunk.page_end,
                chunk.span_sha256,
                chunk.source_version,
                chunk.source_sha256,
            )
        )
    assert citation_rows(result.assistant_message_id) == expected
    assert (
        citation_rows(result.user_message_id) == []
    )  # only the assistant message cites
    assert count("message_citations") == 2
    assert (request_row(rid).status, attempt_row(aid).status) == (
        "succeeded",
        "succeeded",
    )
    assert attempt_row(aid).charged_tokens == 100


def test_markers_keep_the_first_appearance_order_and_need_not_be_contiguous(
    owner, world
):
    rid, aid = ready(owner)
    chunks = [world.chunks[2], world.chunks[0], world.chunks[1]]
    result = publish(
        owner, rid, aid, cite(chunks, markers=[3, 1, 999]), tuple(world.chunks)
    )
    assert result.outcome is P.PUBLISHED
    rows = citation_rows(result.assistant_message_id)
    assert [(r[0], r[1], r[2]) for r in rows] == [
        (1, 3, world.ids[2]),
        (2, 1, world.ids[0]),
        (3, 999, world.ids[1]),
    ]


def test_pages_are_stored_when_the_chunk_has_them(owner):
    document = make_document(owner)
    chunk_id = make_chunk(document, 0, pages=(2, 3))
    chunk = retrieved(chunk_id)
    rid, aid = ready(owner)
    result = publish(owner, rid, aid, cite([chunk]), (chunk,))
    assert result.outcome is P.PUBLISHED
    [row] = citation_rows(result.assistant_message_id)
    assert (row[7], row[8]) == (2, 3)


@pytest.mark.parametrize("pass_retrieved", [False, True])
def test_no_citations_behaves_exactly_as_before(owner, world, pass_retrieved):
    rid, aid = ready(owner)
    retrieved_chunks = tuple(world.chunks) if pass_retrieved else ()
    result = publish(owner, rid, aid, (), retrieved_chunks)
    assert result.outcome is P.PUBLISHED
    assert count("message_citations") == 0
    assert (request_row(rid).status, attempt_row(aid).status) == (
        "succeeded",
        "succeeded",
    )
    assert count("messages") == 2 and count("conversations") == 1


@pytest.mark.parametrize("pass_retrieved", [False, True])
def test_a_publish_without_citations_runs_no_citation_sql_and_takes_no_chunk_locks(
    owner, world, pass_retrieved
):
    rid, aid = ready(owner)
    retrieved_chunks = tuple(world.chunks) if pass_retrieved else ()
    seen = statements_of(lambda: publish(owner, rid, aid, (), retrieved_chunks))
    assert not [s for s in seen if "message_citations" in s or "FOR SHARE" in s]
    assert not [s for s in seen if "documents" in s or "document_chunks" in s]


def test_omitting_the_new_arguments_is_the_old_call(owner):
    rid, aid = ready(owner)
    with SessionLocal() as db:
        result = publish_success(
            db, owner, rid, aid, user_content=QUESTION, assistant_content=ANSWER
        )
    assert result.outcome is P.PUBLISHED and count("message_citations") == 0


# ---------------------------------------------------------------- provenance (in memory)


def assert_invalid_and_nothing_written(owner, world, citations, retrieved_chunks):
    rid, aid = ready(owner)
    before = snapshot()
    result = publish(owner, rid, aid, citations, retrieved_chunks)
    assert result.outcome is P.INVALID_CITATIONS
    assert (result.conversation_id, result.assistant_message_id) == (None, None)
    assert snapshot() == before
    assert untouched(rid, aid)
    # Nothing stayed locked or half-done: the same request still publishes correctly.
    good = publish_ok(owner, rid, aid, world)
    assert good.outcome is P.PUBLISHED
    return rid, aid


def test_a_chunk_the_user_owns_but_was_not_retrieved_is_rejected(owner, world):
    retrieved_subset = (world.chunks[0],)  # chunk 1 is owned, but was not retrieved
    assert_invalid_and_nothing_written(
        owner, world, cite([world.chunks[0], world.chunks[1]]), retrieved_subset
    )


def test_citations_with_an_empty_retrieved_set_are_rejected(owner, world):
    assert_invalid_and_nothing_written(owner, world, cite([world.chunks[0]]), ())


def test_a_citation_whose_chunk_differs_from_the_retrieved_one_is_rejected(
    owner, world
):
    altered = dataclasses.replace(world.chunks[0], text="tampered text")
    assert_invalid_and_nothing_written(
        owner, world, cite([altered]), tuple(world.chunks)
    )


def test_a_different_score_is_a_different_retrieved_chunk(owner, world):
    altered = dataclasses.replace(world.chunks[0], score=0.1)
    assert_invalid_and_nothing_written(
        owner, world, cite([altered]), tuple(world.chunks)
    )


def citations_with(world, changes):
    first, second = world.chunks[0], world.chunks[1]
    base = [
        Citation(ordinal=1, marker=1, chunk=first),
        Citation(ordinal=2, marker=2, chunk=second),
    ]
    return tuple(
        dataclasses.replace(c, **changes.get(i, {})) for i, c in enumerate(base)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {0: {"ordinal": 2}},  # ordinals 2, 2
        {1: {"ordinal": 3}},  # 1, 3: not contiguous
        {0: {"ordinal": 0}, 1: {"ordinal": 1}},  # 0, 1: must start at 1
        {1: {"marker": 1}},  # duplicate marker
        {0: {"marker": 0}},
        {0: {"marker": 1000}},
        {0: {"marker": True}},  # a bool is not an int marker
        {0: {"ordinal": True}},
    ],
)
def test_malformed_citation_numbering_is_rejected(owner, world, changes):
    assert_invalid_and_nothing_written(
        owner, world, citations_with(world, changes), tuple(world.chunks)
    )


def test_citing_the_same_chunk_twice_is_rejected(owner, world):
    twice = (
        Citation(ordinal=1, marker=1, chunk=world.chunks[0]),
        Citation(ordinal=2, marker=2, chunk=world.chunks[0]),
    )
    assert_invalid_and_nothing_written(owner, world, twice, tuple(world.chunks))


def test_out_of_order_ordinals_are_rejected(owner, world):
    swapped = (
        Citation(ordinal=2, marker=1, chunk=world.chunks[0]),
        Citation(ordinal=1, marker=2, chunk=world.chunks[1]),
    )
    assert_invalid_and_nothing_written(owner, world, swapped, tuple(world.chunks))


# ---------------------------------------------------------------- argument errors


@pytest.mark.parametrize(
    "kwargs",
    [
        {"citations": None},
        {"citations": "cite"},
        {"citations": {1}},
        {"citations": ("x",)},
        {"retrieved_chunks": None},
        {"retrieved_chunks": ({"chunk_id": 1},)},
        {"retrieved_chunks": "chunks"},
    ],
)
def test_argument_types_are_checked_before_any_database_work(owner, kwargs):
    rid, aid = ready(owner)
    before = snapshot()
    with pytest.raises(FinalizationArgumentError):
        publish(owner, rid, aid, **kwargs)
    assert snapshot() == before and untouched(rid, aid)


def test_a_chunk_repeated_in_the_retrieved_set_is_a_caller_bug(owner, world):
    rid, aid = ready(owner)
    with pytest.raises(FinalizationArgumentError):
        publish(owner, rid, aid, (), (world.chunks[0], world.chunks[0]))
    assert untouched(rid, aid)


def test_a_citation_that_is_not_a_citation_instance_is_rejected(owner, world):
    class Impostor:
        ordinal, marker, chunk = 1, 1, world.chunks[0]

    rid, aid = ready(owner)
    with pytest.raises(FinalizationArgumentError):
        publish(owner, rid, aid, (Impostor(),), tuple(world.chunks))


def test_lists_are_accepted_like_tuples(owner, world):
    rid, aid = ready(owner)
    result = publish(owner, rid, aid, list(cite(world.chunks[:2])), list(world.chunks))
    assert result.outcome is P.PUBLISHED and count("message_citations") == 2


# ---------------------------------------------------------------- database checks


def assert_rolled_back(owner, rid, aid, citations, retrieved_chunks):
    before = snapshot()
    result = publish(owner, rid, aid, citations, retrieved_chunks)
    assert result.outcome is P.INVALID_CITATIONS
    assert (
        snapshot() == before
    )  # the attempt settlement and the messages rolled back too
    assert untouched(rid, aid)
    assert count("message_citations") == 0


def test_a_foreign_owners_chunk_is_rejected_and_everything_rolls_back(
    owner, other_owner
):
    foreign = World(other_owner)
    mine = World(owner)
    rid, aid = ready(owner)
    # The caller wrongly includes the foreign chunk in the retrieved set.
    assert_rolled_back(
        owner,
        rid,
        aid,
        cite([foreign.chunks[0]]),
        (foreign.chunks[0], *mine.chunks),
    )
    assert publish_ok(owner, rid, aid, mine).outcome is P.PUBLISHED


def test_a_nonexistent_chunk_is_rejected_and_everything_rolls_back(owner, world):
    ghost = dataclasses.replace(world.chunks[0], chunk_id=world.ids[-1] + 1000)
    rid, aid = ready(owner)
    assert_rolled_back(owner, rid, aid, cite([ghost]), (ghost,))


@pytest.mark.parametrize(
    "changes",
    [
        {"document_id": 999_999},
        {"chunk_index": 7},
        {"start_offset": 1},
        {"end_offset": 5000},
        {"page_start": 1, "page_end": 2},
        {"span_sha256": "f" * 64},
        {"source_version": 2},
        {"source_sha256": "c" * 64},
        {"chunk_id": "other"},  # a real chunk id whose row has different data
    ],
)
def test_every_metadata_field_must_agree_with_the_database(owner, world, changes):
    changes = dict(changes)
    if changes.get("chunk_id") == "other":
        changes["chunk_id"] = world.ids[1]
    altered = dataclasses.replace(world.chunks[0], **changes)
    rid, aid = ready(owner)
    # The retrieved set agrees with the citation, so only the database can disagree.
    assert_rolled_back(owner, rid, aid, cite([altered]), (altered,))


def test_a_document_reindexed_after_retrieval_is_rejected(owner, world):
    execute(
        "UPDATE documents SET source_version = 2, indexed_source_version = 2"
        " WHERE id = :d",
        d=world.document,
    )
    rid, aid = ready(owner)
    assert_rolled_back(owner, rid, aid, cite(world.chunks[:1]), tuple(world.chunks))


def test_a_chunk_deleted_after_retrieval_is_rejected(owner, world):
    execute("DELETE FROM document_chunks WHERE id = :c", c=world.ids[0])
    rid, aid = ready(owner)
    assert_rolled_back(owner, rid, aid, cite(world.chunks[:1]), tuple(world.chunks))


def test_a_document_deleted_after_retrieval_is_rejected(owner, world):
    execute("DELETE FROM documents WHERE id = :d", d=world.document)
    rid, aid = ready(owner)
    assert_rolled_back(owner, rid, aid, cite(world.chunks[:1]), tuple(world.chunks))


def test_one_bad_citation_among_good_ones_rejects_all(owner, world):
    execute("DELETE FROM document_chunks WHERE id = :c", c=world.ids[1])
    rid, aid = ready(owner)
    assert_rolled_back(owner, rid, aid, cite(world.chunks[:2]), tuple(world.chunks))


# ---------------------------------------------------------------- replays and fences


def test_a_replay_returns_the_stored_ids_and_adds_no_citation_rows(owner, world):
    rid, aid = ready(owner)
    first = publish_ok(owner, rid, aid, world)
    assert first.outcome is P.PUBLISHED
    before = snapshot()
    replays = [
        publish_ok(owner, rid, aid, world),  # same citations
        publish_ok(owner, rid, aid, world, picks=(2,)),  # different citations
        publish(owner, rid, aid, cite([world.chunks[0]]), ()),  # not even retrieved
    ]
    for replay in replays:
        assert replay.outcome is P.ALREADY_SUCCEEDED
        assert replay.assistant_message_id == first.assistant_message_id
    assert snapshot() == before
    assert count("message_citations") == 2


def test_a_failed_request_publishes_nothing_even_with_citations(owner, world):
    rid, aid = ready(owner)
    assert fail_gen(owner, rid, aid).outcome is F.FAILED
    before = snapshot()
    assert publish_ok(owner, rid, aid, world).outcome is P.REQUEST_FAILED
    assert snapshot() == before


def test_an_expired_lease_stops_a_citing_publish_without_writes(owner, world):
    rid, aid = ready(owner)
    expire_lease(aid)
    before = snapshot()
    assert publish_ok(owner, rid, aid, world).outcome is P.LEASE_EXPIRED
    assert snapshot() == before


def test_an_unstarted_attempt_cannot_publish_citations(owner, world):
    rid, aid = ready(owner, started=False)
    before = snapshot()
    assert publish_ok(owner, rid, aid, world).outcome is P.WRONG_PHASE
    assert snapshot() == before


def test_a_missing_conversation_rolls_back_citations_too(owner, world):
    cid = make_conversation(owner, messages=2)
    rid, aid = ready(owner, conversation_id=cid)
    execute("DELETE FROM conversations WHERE id = :c", c=cid)
    before = snapshot()
    assert publish_ok(owner, rid, aid, world).outcome is P.CONVERSATION_NOT_FOUND
    assert snapshot() == before and count("message_citations") == 0


# ---------------------------------------------------------------- atomic rollback


@pytest.mark.parametrize(
    ("table", "when", "event_name"),
    [
        ("message_citations", "NEW.ordinal = 2", "INSERT"),  # the second citation fails
        ("message_citations", "NEW.ordinal = 1", "INSERT"),  # the first one fails
        ("chat_requests", "NEW.status = 'succeeded'", "UPDATE"),  # after the citations
        ("messages", "NEW.role = 'assistant'", "INSERT"),  # before the citations
        ("chat_request_attempts", "NEW.status = 'succeeded'", "UPDATE"),
    ],
)
def test_a_database_failure_anywhere_rolls_back_the_whole_publish(
    owner, world, table, when, event_name
):
    rid, aid = ready(owner)
    before = snapshot()
    with boom_trigger(table, when, event_name), pytest.raises(DBAPIError):
        publish_ok(owner, rid, aid, world)
    assert snapshot() == before
    assert untouched(rid, aid)
    assert publish_ok(owner, rid, aid, world).outcome is P.PUBLISHED  # nothing stuck
    assert count("message_citations") == 2


def test_a_python_failure_after_the_citations_rolls_everything_back(
    owner, world, monkeypatch
):
    rid, aid = ready(owner)
    before = snapshot()
    original = finalization_module._persist_citations

    def persist_then_crash(*args, **kwargs):
        assert original(*args, **kwargs) is True
        raise KeyboardInterrupt("simulated crash after the citation inserts")

    monkeypatch.setattr(finalization_module, "_persist_citations", persist_then_crash)
    with pytest.raises(KeyboardInterrupt):
        publish_ok(owner, rid, aid, world)
    assert snapshot() == before
    monkeypatch.undo()
    assert publish_ok(owner, rid, aid, world).outcome is P.PUBLISHED


PUBLISH_CRASH_SCRIPT = """
import json, os, sys
import app.rag.chat_request_finalization as f
from app.core.database import SessionLocal
from app.rag.chat_citations import Citation
from app.rag.retrieval import RetrievedChunk

original = f._persist_citations

def persist_then_die(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(17)        # dies with the citation rows inserted but nothing committed

owner, request_id, attempt_id = (int(v) for v in sys.argv[1:4])
chunks = tuple(RetrievedChunk(**d) for d in json.loads(sys.argv[4]))
f._persist_citations = persist_then_die
with SessionLocal() as db:
    f.publish_success(
        db, owner, request_id, attempt_id, user_content="q", assistant_content="a [1]",
        citations=(Citation(ordinal=1, marker=1, chunk=chunks[0]),), retrieved_chunks=chunks,
    )
os._exit(0)
"""


def test_a_process_crash_after_the_citation_inserts_commits_nothing(owner, world):
    rid, aid = ready(owner)
    before = snapshot()
    payload = json.dumps([dataclasses.asdict(c) for c in world.chunks])
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            PUBLISH_CRASH_SCRIPT,
            str(owner),
            str(rid),
            str(aid),
            payload,
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 17, done.stderr  # it really died mid-transaction
    assert snapshot() == before and untouched(rid, aid)
    assert publish_ok(owner, rid, aid, world).outcome is P.PUBLISHED


# ---------------------------------------------------------------- locking and concurrency


def hold_publish_open(monkeypatch):
    """Make publish_success pause (locks held, nothing committed) after its citation inserts."""
    held, release = threading.Event(), threading.Event()
    original = finalization_module._persist_citations

    def persist_then_wait(*args, **kwargs):
        ok = original(*args, **kwargs)
        held.set()
        release.wait(timeout=30)
        return ok

    monkeypatch.setattr(finalization_module, "_persist_citations", persist_then_wait)
    return held, release


def nowait_lock_fails(sql, **params):
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text(sql), params)
        except DBAPIError as error:
            assert "could not obtain lock" in str(error.orig)
            return True
        finally:
            transaction.rollback()
        return False


def test_publish_locks_exactly_the_cited_documents_and_chunks(
    owner, world, monkeypatch
):
    other_world = World(owner)  # another owned document that is not cited
    rid, aid = ready(owner)
    held, release = hold_publish_open(monkeypatch)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(publish_ok, owner, rid, aid, world, (0, 1))
        try:
            assert held.wait(timeout=20)
            lock = "SELECT 1 FROM {t} WHERE id = :i FOR UPDATE NOWAIT"
            for chunk_id in world.ids[:2]:  # cited chunks are locked
                assert nowait_lock_fails(lock.format(t="document_chunks"), i=chunk_id)
            assert nowait_lock_fails(lock.format(t="documents"), i=world.document)
            # A non-key UPDATE of the document (re-index metadata) is blocked too.
            assert nowait_lock_fails(
                "SELECT 1 FROM documents WHERE id = :i FOR NO KEY UPDATE NOWAIT",
                i=world.document,
            )
            # Rows that were not cited are not locked by the publish.
            assert not nowait_lock_fails(
                lock.format(t="document_chunks"), i=world.ids[2]
            )
            assert not nowait_lock_fails(
                lock.format(t="document_chunks"), i=other_world.ids[0]
            )
            assert not nowait_lock_fails(
                lock.format(t="documents"), i=other_world.document
            )
        finally:
            release.set()
        assert future.result(timeout=30).outcome is P.PUBLISHED


@pytest.mark.parametrize(
    ("delete_sql", "pattern"),
    [
        ("DELETE FROM document_chunks WHERE id = :c", "DELETE FROM document_chunks%"),
        ("DELETE FROM documents WHERE id = :d", "DELETE FROM documents%"),
    ],
)
def test_a_delete_during_publish_waits_and_leaves_a_consistent_citation(
    owner, world, monkeypatch, delete_sql, pattern
):
    rid, aid = ready(owner)
    held, release = hold_publish_open(monkeypatch)
    with ThreadPoolExecutor(2) as pool:
        publisher = pool.submit(publish_ok, owner, rid, aid, world, (0, 1))
        try:
            assert held.wait(timeout=20)
            deleter = pool.submit(execute, delete_sql, c=world.ids[0], d=world.document)
            wait_for_lock_wait(pattern)  # the delete is blocked by the publish's locks
            assert not deleter.done()
        finally:
            release.set()
        result = publisher.result(timeout=30)
        deleter.result(timeout=30)
    assert result.outcome is P.PUBLISHED
    rows = execute(
        "SELECT chunk_id, document_id, chunk_index FROM message_citations ORDER BY ordinal"
    ).all()
    assert [tuple(r) for r in rows] == [
        (None, world.document, 0),
        *(
            [(None, world.document, 1)]
            if "documents" in delete_sql
            else [(world.ids[1], world.document, 1)]
        ),
    ]
    assert request_row(rid).status == "succeeded"


@pytest.mark.parametrize(
    ("statement", "pattern"),
    [
        (
            "DELETE FROM document_chunks WHERE id = :c",
            "%FROM document_chunks%FOR SHARE%",
        ),
        ("DELETE FROM documents WHERE id = :d", "%FROM documents%FOR SHARE%"),
        (
            "UPDATE documents SET source_sha256 = repeat('e', 64) WHERE id = :d",
            "%FROM documents%FOR SHARE%",
        ),
    ],
)
def test_a_publish_that_waits_behind_an_uncommitted_change_rolls_back(
    owner, world, statement, pattern
):
    rid, aid = ready(owner)
    before = snapshot()
    writer = engine.connect()
    pool = ThreadPoolExecutor(1)
    transaction = writer.begin()
    try:
        writer.execute(text(statement), {"c": world.ids[0], "d": world.document})
        publisher = pool.submit(publish_ok, owner, rid, aid, world, (0, 1))
        wait_for_lock_wait(pattern)  # the publish is blocked on the cited row
        transaction.commit()
        result = publisher.result(timeout=30)
    finally:
        if transaction.is_active:
            transaction.rollback()
        pool.shutdown(wait=True)
        writer.close()
    assert result.outcome is P.INVALID_CITATIONS
    assert snapshot() == before and untouched(rid, aid)


def test_a_publish_behind_a_rolled_back_change_succeeds(owner, world):
    rid, aid = ready(owner)
    writer = engine.connect()
    pool = ThreadPoolExecutor(1)
    transaction = writer.begin()
    try:
        writer.execute(
            text("DELETE FROM document_chunks WHERE id = :c"), {"c": world.ids[0]}
        )
        publisher = pool.submit(publish_ok, owner, rid, aid, world, (0, 1))
        wait_for_lock_wait("%FROM document_chunks%FOR SHARE%")
        transaction.rollback()
        result = publisher.result(timeout=30)
    finally:
        if transaction.is_active:
            transaction.rollback()
        pool.shutdown(wait=True)
        writer.close()
    assert result.outcome is P.PUBLISHED and count("message_citations") == 2


def test_concurrent_publishes_of_one_request_write_the_citations_exactly_once(
    owner, world
):
    rid, aid = ready(owner)
    results = run_threads(
        6,
        lambda db, i: publish_success(
            db,
            owner,
            rid,
            aid,
            user_content=QUESTION,
            assistant_content=ANSWER,
            citations=cite(world.chunks[:2]),
            retrieved_chunks=tuple(world.chunks),
        ),
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(P.PUBLISHED) == 1 and outcomes.count(P.ALREADY_SUCCEEDED) == 5
    assert count("message_citations") == 2 and count("messages") == 2


def test_publish_and_fail_after_generation_never_both_win(owner, world):
    for _ in range(8):
        rid, aid = ready(owner)
        citations_before = citation_rows()
        messages_before = count("messages")
        conversations_before = count("conversations")

        def work(db, i, rid=rid, aid=aid):
            if i == 0:
                return publish_success(
                    db,
                    owner,
                    rid,
                    aid,
                    user_content=QUESTION,
                    assistant_content=ANSWER,
                    citations=cite(world.chunks[:1]),
                    retrieved_chunks=tuple(world.chunks),
                )
            return fail_request_after_generation(db, owner, rid, aid, CODE, 10, 5)

        published, failed = run_threads(2, work)
        request, attempt = request_row(rid), attempt_row(aid)
        if published.outcome is P.PUBLISHED:
            assert failed.outcome is F.ALREADY_FINALIZED
            assert (request.status, attempt.status) == ("succeeded", "succeeded")
            # Exactly one citation row was added, on the published assistant message.
            added = citation_rows()[len(citations_before) :]
            assert citation_rows()[: len(citations_before)] == citations_before
            assert len(added) == 1 and added[0][2] == world.ids[0]
            assert citation_rows(published.assistant_message_id) == added
            assert count("messages") == messages_before + 2
            assert count("conversations") == conversations_before + 1
        else:
            assert published.outcome is P.REQUEST_FAILED
            assert failed.outcome is F.FAILED
            assert (request.status, attempt.status) == ("failed", "failed_with_usage")
            # The failed side wrote no citation, message or conversation.
            assert citation_rows() == citations_before
            assert count("messages") == messages_before
            assert count("conversations") == conversations_before


def statements_of(function):
    seen = []

    def record(conn, cursor, statement, parameters, context, executemany):
        seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", record)
    try:
        function()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return seen


def first_index(statements, *needles):
    for index, statement in enumerate(statements):
        if all(n in statement for n in needles):
            return index
    raise AssertionError(f"no statement contains {needles}")


def test_publish_takes_its_locks_in_the_documented_order(owner, world):
    rid, aid = ready(owner)
    seen = statements_of(lambda: publish_ok(owner, rid, aid, world))
    order = [
        first_index(seen, "FROM chat_requests", "FOR UPDATE"),  # request first
        first_index(seen, "UPDATE chat_request_attempts"),  # then the attempt
        first_index(seen, "INSERT INTO conversations"),  # then the conversation
        first_index(seen, "INSERT INTO messages"),
        first_index(seen, "FROM documents", "FOR SHARE"),  # then documents
        first_index(seen, "FROM document_chunks", "FOR SHARE"),  # then chunks
        first_index(seen, "INSERT INTO message_citations"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    chunk_lock = seen[order[5]]
    assert "ORDER BY id" in chunk_lock and "ORDER BY id" in seen[order[4]]


def test_fail_after_generation_locks_the_request_before_the_attempt(owner):
    rid, aid = ready(owner)
    seen = statements_of(
        lambda: fail_gen(owner, rid, aid, prompt_tokens=1, completion_tokens=1)
    )
    assert (
        first_index(seen, "FROM chat_requests", "FOR UPDATE")
        < first_index(seen, "UPDATE chat_request_attempts")
        < first_index(seen, "UPDATE chat_requests")
    )


# ---------------------------------------------------------------- fail_request_after_generation


@pytest.mark.parametrize("existing_conversation", [False, True])
def test_fail_after_generation_with_usage_charges_the_actual_usage(
    owner, existing_conversation
):
    cid = make_conversation(owner, messages=2) if existing_conversation else None
    rid, aid = ready(owner, conversation_id=cid, reserved=500)
    conversations, messages = count("conversations"), count("messages")
    result = fail_gen(owner, rid, aid, prompt_tokens=120, completion_tokens=30)
    assert result == FailAfterGenerationResult(F.FAILED)
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.prompt_tokens, attempt.completion_tokens) == (
        "failed_with_usage",
        120,
        30,
    )
    assert (attempt.charged_tokens, attempt.error_code) == (150, CODE)
    request = request_row(rid)
    assert (request.status, request.error_code) == ("failed", CODE)
    assert request.completed_at is not None
    assert (count("conversations"), count("messages")) == (conversations, messages)
    assert count("message_citations") == 0


def test_fail_after_generation_without_usage_charges_the_reservation(owner):
    rid, aid = ready(owner, reserved=500)
    assert fail_gen(owner, rid, aid).outcome is F.FAILED
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.charged_tokens, attempt.error_code) == (
        "uncertain",
        500,
        CODE,
    )
    assert (attempt.prompt_tokens, attempt.completion_tokens) == (None, None)
    assert request_row(rid).status == "failed"


def test_it_never_charges_zero(owner):
    for usage in ({}, {"prompt_tokens": 0, "completion_tokens": 0}):
        rid, aid = ready(owner, reserved=500)
        assert fail_gen(owner, rid, aid, **usage).outcome is F.FAILED
        attempt = attempt_row(aid)
        # Reported usage of 0 + 0 is the provider's own figure; without usage it is 500.
        assert attempt.charged_tokens == (0 if usage else 500)
        assert attempt.status == ("failed_with_usage" if usage else "uncertain")


def daily_charged():
    return execute(
        "SELECT coalesce(sum(charged_tokens), 0) FROM chat_request_attempts"
    ).scalar_one()


def test_the_quota_ledger_reflects_the_settlement(owner):
    rid, aid = ready(owner, reserved=500)
    assert (
        fail_gen(owner, rid, aid, prompt_tokens=200, completion_tokens=40).outcome
        is F.FAILED
    )
    rid2, aid2 = ready(owner, reserved=700)
    assert fail_gen(owner, rid2, aid2).outcome is F.FAILED
    assert daily_charged() == 240 + 700


def test_an_invalid_citation_then_the_atomic_fail_is_the_full_story(owner, world):
    cid = make_conversation(owner, messages=2)
    rid, aid = ready(owner, conversation_id=cid, reserved=500)
    before_messages = count("messages")
    bad = dataclasses.replace(world.chunks[0], source_sha256="c" * 64)
    result = publish(
        owner, rid, aid, cite([bad]), (bad,), prompt_tokens=90, completion_tokens=40
    )
    assert result.outcome is P.INVALID_CITATIONS and untouched(rid, aid)
    failed = fail_gen(owner, rid, aid, prompt_tokens=90, completion_tokens=40)
    assert failed.outcome is F.FAILED
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.charged_tokens) == ("failed_with_usage", 130)
    assert count("messages") == before_messages and count("message_citations") == 0
    stored = call(get_request_result, owner, rid)
    assert stored == RequestResult("failed", cid, None, None, CODE)
    assert call(fail_request, owner, rid, "x").outcome is FailOutcome.ALREADY_FINALIZED


def test_a_crash_before_the_atomic_fail_leaves_a_running_attempt_that_expiry_charges(
    owner, world
):
    rid, aid = ready(owner, reserved=500)
    bad = dataclasses.replace(world.chunks[0], span_sha256="f" * 64)
    assert publish(owner, rid, aid, cite([bad]), (bad,)).outcome is P.INVALID_CITATIONS
    # ...the worker dies here. Expiry is the existing conservative recovery.
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.charged_tokens) == ("uncertain", 500)
    assert call(fail_request, owner, rid, "lease_expired").outcome is FailOutcome.FAILED


@pytest.mark.parametrize("reserved", [1, 500, 777, 123_456])
def test_missing_usage_charges_exactly_this_attempts_reservation(owner, reserved):
    other_rid, other_aid = ready(
        owner, reserved=9_999
    )  # another attempt must not leak in
    rid, aid = ready(owner, reserved=reserved)
    assert attempt_row(aid).reserved_tokens == reserved
    result = fail_gen(owner, rid, aid)
    assert result == FailAfterGenerationResult(F.FAILED)
    attempt = attempt_row(aid)
    assert attempt.status == "uncertain"
    assert attempt.charged_tokens == reserved == attempt.reserved_tokens
    assert (attempt.prompt_tokens, attempt.completion_tokens) == (None, None)
    assert attempt.error_code == CODE and attempt.finished_at is not None
    assert attempt.lease_expires_at is None
    request = request_row(rid)
    assert (request.status, request.error_code) == ("failed", CODE)
    assert request.completed_at is not None
    # The other request's attempt is untouched.
    assert untouched(other_rid, other_aid)
    assert attempt_row(other_aid).charged_tokens is None


def test_the_reservation_comes_from_the_fenced_update_on_the_locked_row(owner):
    rid, aid = ready(owner, reserved=777)
    seen = statements_of(lambda: fail_gen(owner, rid, aid))
    update_index = first_index(seen, "UPDATE chat_request_attempts")
    update = seen[update_index]
    # One statement settles the attempt: the database copies the row's own reservation,
    # under the same live-lease, running and started-phase predicates as every worker write.
    assert "charged_tokens=chat_request_attempts.reserved_tokens" in update
    for predicate in (
        "chat_request_attempts.status = ",  # running
        "chat_request_attempts.lease_expires_at > clock_timestamp()",  # live lease
        "chat_request_attempts.llm_started_at IS NOT NULL",  # started phase
        "RETURNING chat_request_attempts.id",
    ):
        assert predicate in update, predicate
    # No separate read of the attempt row happens between the request lock and that update,
    # so the lock order stays request, then attempt.
    before_update = seen[:update_index]
    assert not [s for s in before_update if "reserved_tokens" in s]
    assert not [
        s
        for s in before_update
        if "FROM chat_request_attempts" in s and "FOR UPDATE" in s
    ]
    assert first_index(seen, "FROM chat_requests", "FOR UPDATE") < update_index
    assert update_index < first_index(seen, "UPDATE chat_requests")
    assert attempt_row(aid).charged_tokens == 777


def test_an_expired_lease_with_invalid_citations_changes_nothing_at_either_step(
    owner, world
):
    rid, aid = ready(owner, reserved=500)
    expire_lease(aid)
    # Not in the retrieved set: detected in memory, before the fence.
    not_retrieved = cite([world.chunks[1]])
    before = snapshot()
    published = publish(
        owner,
        rid,
        aid,
        not_retrieved,
        (world.chunks[0],),
        prompt_tokens=90,
        completion_tokens=40,
    )
    assert published.outcome is P.INVALID_CITATIONS
    assert snapshot() == before
    failed = fail_gen(owner, rid, aid, prompt_tokens=90, completion_tokens=40)
    assert failed.outcome is F.LEASE_EXPIRED  # the fence still decides, with no writes
    assert snapshot() == before
    attempt = attempt_row(aid)
    assert attempt.status == "running" and attempt.charged_tokens is None
    assert attempt.lease_expires_at is not None  # still the expiry path's to settle
    assert request_row(rid).status == "in_progress"
    # Only expiry may settle it, conservatively, as before.
    assert call(expire_stale_attempt, rid).attempt_id == aid
    assert attempt_row(aid).charged_tokens == 500
    assert attempt_row(aid).status == "uncertain"


def test_an_expired_lease_with_a_database_level_mismatch_hits_the_fence_first(
    owner, world
):
    # Consistent with the retrieved set, but not with the database: only the in-transaction
    # check can see it, and that runs after the fence, so the fence outcome wins.
    rid, aid = ready(owner)
    expire_lease(aid)
    bad = dataclasses.replace(world.chunks[0], span_sha256="f" * 64)
    before = snapshot()
    assert publish(owner, rid, aid, cite([bad]), (bad,)).outcome is P.LEASE_EXPIRED
    assert fail_gen(owner, rid, aid).outcome is F.LEASE_EXPIRED
    assert snapshot() == before and request_row(rid).status == "in_progress"


def test_an_expired_lease_with_valid_citations_is_the_fence_not_a_provenance_error(
    owner, world
):
    rid, aid = ready(owner)
    expire_lease(aid)
    before = snapshot()
    assert publish_ok(owner, rid, aid, world).outcome is P.LEASE_EXPIRED
    assert fail_gen(owner, rid, aid).outcome is F.LEASE_EXPIRED
    assert snapshot() == before


# ---- association, validation and the fence


def test_a_wrong_owner_request_or_attempt_changes_nothing(owner, other_owner):
    rid, aid = ready(owner)
    other_rid, other_aid = ready(other_owner)
    before = snapshot()
    for args in (
        (other_owner, rid, aid),
        (owner, other_rid, aid),
        (owner, rid, other_aid),
        (owner, rid + 999, aid),
        (owner, rid, aid + 999),
    ):
        assert fail_gen(*args).outcome is F.REQUEST_NOT_FOUND
    assert snapshot() == before


def test_an_already_finalized_request_is_reported_with_its_status(owner, world):
    rid, aid = ready(owner)
    assert publish_ok(owner, rid, aid, world).outcome is P.PUBLISHED
    before = snapshot()
    result = fail_gen(owner, rid, aid)
    assert (result.outcome, result.current_status) == (F.ALREADY_FINALIZED, "succeeded")
    assert snapshot() == before
    rid2, aid2 = ready(owner)
    assert fail_gen(owner, rid2, aid2).outcome is F.FAILED
    again = fail_gen(owner, rid2, aid2)
    assert (again.outcome, again.current_status) == (F.ALREADY_FINALIZED, "failed")


def test_an_unstarted_attempt_cannot_be_failed_this_way(owner):
    rid, aid = ready(owner, started=False)
    before = snapshot()
    assert fail_gen(owner, rid, aid).outcome is F.WRONG_PHASE
    assert snapshot() == before


def test_an_expired_lease_cannot_be_failed_this_way(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    before = snapshot()
    assert fail_gen(owner, rid, aid).outcome is F.LEASE_EXPIRED
    assert snapshot() == before and request_row(rid).status == "in_progress"


def test_a_reaped_attempt_cannot_be_failed_this_way(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    before = snapshot()
    assert fail_gen(owner, rid, aid).outcome is F.NOT_RUNNING
    assert snapshot() == before


def test_a_stale_worker_cannot_fail_after_takeover(owner):
    rid, old_aid = ready(owner)
    expire_lease(old_aid)
    assert call(expire_stale_attempt, rid).attempt_id == old_aid
    _, new_aid = ready(owner, request_id=rid)
    before = snapshot()
    assert fail_gen(owner, rid, old_aid).outcome is F.NOT_RUNNING
    assert snapshot() == before and attempt_row(new_aid).status == "running"
    assert (
        fail_gen(owner, rid, new_aid).outcome is F.FAILED
    )  # the live worker still can


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 1},
        {"completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1.5, "completion_tokens": 1},
        {"prompt_tokens": 2**31, "completion_tokens": 1},
    ],
)
def test_usage_is_both_or_neither_and_a_valid_integer_pair(owner, usage):
    rid, aid = ready(owner)
    before = snapshot()
    with pytest.raises(AttemptLifecycleArgumentError):
        fail_gen(owner, rid, aid, **usage)
    assert snapshot() == before


@pytest.mark.parametrize("code", ["", "Bad", "1abc", "has space", "x" * 65, None, 5])
def test_the_error_code_format_is_checked(owner, code):
    rid, aid = ready(owner)
    with pytest.raises(FinalizationArgumentError):
        fail_gen(owner, rid, aid, code)
    assert untouched(rid, aid)


@pytest.mark.parametrize("bad", [0, -1, True, "1", 2**31, None])
def test_ids_are_checked(owner, bad):
    rid, aid = ready(owner)
    for args in ((bad, rid, aid), (owner, bad, aid), (owner, rid, bad)):
        with pytest.raises(FinalizationArgumentError):
            fail_gen(*args)
    assert untouched(rid, aid)


# ---- atomicity


USAGE = {"prompt_tokens": 10, "completion_tokens": 5}
NO_USAGE = {}
BOTH_USAGE_FORMS = pytest.mark.parametrize(
    "usage", [USAGE, NO_USAGE], ids=["with_usage", "without_usage"]
)


@pytest.mark.parametrize(
    ("table", "when"),
    [
        ("chat_request_attempts", "NEW.status IN ('failed_with_usage', 'uncertain')"),
        ("chat_requests", "NEW.status = 'failed'"),
    ],
)
@BOTH_USAGE_FORMS
def test_a_database_failure_at_either_update_leaves_nothing_behind(
    owner, table, when, usage
):
    rid, aid = ready(owner)
    before = snapshot()
    with boom_trigger(table, when, "UPDATE"), pytest.raises(DBAPIError):
        fail_gen(owner, rid, aid, **usage)
    assert snapshot() == before and untouched(rid, aid)
    assert attempt_row(aid).charged_tokens is None  # not even a charge survived
    assert fail_gen(owner, rid, aid, **usage).outcome is F.FAILED


@BOTH_USAGE_FORMS
def test_a_python_failure_between_the_two_updates_rolls_the_attempt_back(
    owner, monkeypatch, usage
):
    rid, aid = ready(owner)
    before = snapshot()

    def explode(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash between the two updates")

    monkeypatch.setattr(finalization_module, "_set_failed", explode)
    with pytest.raises(KeyboardInterrupt):
        fail_gen(owner, rid, aid, **usage)
    assert snapshot() == before
    assert attempt_row(aid).charged_tokens is None
    monkeypatch.undo()
    assert fail_gen(owner, rid, aid, **usage).outcome is F.FAILED


@BOTH_USAGE_FORMS
def test_a_failed_request_update_after_a_settled_attempt_rolls_everything_back(
    owner, monkeypatch, usage
):
    rid, aid = ready(owner)
    before = snapshot()
    monkeypatch.setattr(finalization_module, "_set_failed", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        fail_gen(owner, rid, aid, **usage)
    assert snapshot() == before and untouched(rid, aid)
    assert attempt_row(aid).charged_tokens is None


def test_a_failure_while_settling_the_attempt_leaves_the_request_untouched(
    owner, monkeypatch
):
    rid, aid = ready(owner)
    before = snapshot()

    def explode(*args, **kwargs):
        raise RuntimeError("settlement failed")

    monkeypatch.setattr(finalization_module, "settle_failed_in_transaction", explode)
    with pytest.raises(RuntimeError):
        fail_gen(owner, rid, aid)
    assert snapshot() == before


FAIL_CRASH_SCRIPT = """
import os, sys
import app.rag.chat_request_finalization as f
from app.core.database import SessionLocal

def die(*args, **kwargs):
    os._exit(17)          # dies between the attempt update and the request update

f._set_failed = die
owner, request_id, attempt_id = (int(v) for v in sys.argv[1:4])
usage = (9, 9) if sys.argv[4] == "usage" else (None, None)
with SessionLocal() as db:
    f.fail_request_after_generation(db, owner, request_id, attempt_id, "internal_error", *usage)
os._exit(0)
"""


@pytest.mark.parametrize("usage", ["usage", "no_usage"])
def test_a_process_crash_between_the_updates_commits_neither_row(owner, usage):
    rid, aid = ready(owner)
    before = snapshot()
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            FAIL_CRASH_SCRIPT,
            str(owner),
            str(rid),
            str(aid),
            usage,
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 17, done.stderr
    assert snapshot() == before and untouched(rid, aid)
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    assert attempt_row(aid).charged_tokens == 500  # the reservation, never zero


def settled_but_in_progress():
    """Settled attempts whose request is still in progress, ignoring ordinary expiry."""
    return execute(
        "SELECT count(*) FROM chat_request_attempts a JOIN chat_requests r"
        " ON r.id = a.request_id WHERE r.status = 'in_progress'"
        " AND a.status IN ('failed_with_usage', 'uncertain') AND a.error_code = :code",
        code=CODE,
    ).scalar_one()


def test_no_outcome_of_the_function_leaves_a_settled_attempt_with_an_open_request(
    owner,
):
    rid, aid = ready(owner)
    expired_rid, expired_aid = ready(owner)
    expire_lease(expired_aid)
    unstarted_rid, unstarted_aid = ready(owner, started=False)
    for args in (
        (owner, rid, aid),
        (owner, rid, aid),
        (owner, rid + 999, aid),
        (owner, expired_rid, expired_aid),
        (owner, unstarted_rid, unstarted_aid),
    ):
        fail_gen(*args, prompt_tokens=1, completion_tokens=1)
        assert settled_but_in_progress() == 0
    for args in (  # the same, without usage (the reservation path)
        ready(owner),
        ready(owner),
        ready(owner, started=False),
    ):
        rid_, aid_ = args
        fail_gen(owner, rid_, aid_)
        fail_gen(owner, rid_, aid_)
        assert settled_but_in_progress() == 0


def test_the_helper_stays_private_and_only_the_atomic_operation_uses_it():
    app_dir = PROJECT_ROOT / "app"
    users = {
        p.name
        for p in app_dir.rglob("*.py")
        if "settle_failed_in_transaction" in p.read_text(encoding="utf-8")
    }
    assert users == {"chat_attempt_lifecycle.py", "chat_request_finalization.py"}
    assert "fail_request_after_generation" in finalization_module.__all__
    assert "settle_failed_in_transaction" not in finalization_module.__all__
    assert "settle_failed_in_transaction" not in lifecycle_module.__all__


def test_fail_request_and_fail_with_usage_keep_their_behaviour(owner):
    rid, aid = ready(owner)
    assert call(fail_request, owner, rid, "x").outcome is FailOutcome.ATTEMPT_RUNNING
    assert (
        call(lifecycle_module.fail_with_usage, aid, CODE, 4, 6)
        is lifecycle_module.TransitionOutcome.APPLIED
    )
    assert (
        request_row(rid).status == "in_progress"
    )  # still the two-step path, unchanged
    assert call(fail_request, owner, rid, CODE).outcome is FailOutcome.FAILED
    assert attempt_row(aid).charged_tokens == 10


# ---- concurrency


def test_concurrent_calls_apply_exactly_once(owner):
    rid, aid = ready(owner)
    results = run_threads(
        6, lambda db, i: fail_request_after_generation(db, owner, rid, aid, CODE, 7, 3)
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(F.FAILED) == 1 and outcomes.count(F.ALREADY_FINALIZED) == 5
    assert (request_row(rid).status, attempt_row(aid).status) == (
        "failed",
        "failed_with_usage",
    )
    assert attempt_row(aid).charged_tokens == 10


def test_fail_after_generation_and_fail_request_never_conflict(owner):
    for _ in range(5):
        rid, aid = ready(owner)

        def work(db, i, rid=rid, aid=aid):
            if i == 0:
                return fail_request_after_generation(db, owner, rid, aid, CODE, 5, 5)
            return fail_request(db, owner, rid, "gave_up")

        failed, other = run_threads(2, work)
        assert failed.outcome is F.FAILED  # fail_request refuses while the attempt runs
        assert other.outcome in (
            FailOutcome.ATTEMPT_RUNNING,
            FailOutcome.ALREADY_FINALIZED,
        )
        assert request_row(rid).error_code == CODE


def test_fail_after_generation_and_expiry_are_mutually_exclusive(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    failed, expired = run_threads(
        2,
        lambda db, i: (
            fail_request_after_generation(db, owner, rid, aid, CODE, 5, 5)
            if i == 0
            else expire_stale_attempt(db, rid)
        ),
    )
    assert failed.outcome in (F.LEASE_EXPIRED, F.NOT_RUNNING)
    assert expired.attempt_id == aid
    assert (attempt_row(aid).status, request_row(rid).status) == (
        "uncertain",
        "in_progress",
    )


def test_a_live_lease_is_never_expired_by_a_concurrent_fail(owner):
    rid, aid = ready(owner)
    failed, expired = run_threads(
        2,
        lambda db, i: (
            fail_request_after_generation(db, owner, rid, aid, CODE, 5, 5)
            if i == 0
            else expire_stale_attempt(db, rid)
        ),
    )
    assert failed.outcome is F.FAILED and expired.attempt_id is None
    assert attempt_row(aid).status == "failed_with_usage"
