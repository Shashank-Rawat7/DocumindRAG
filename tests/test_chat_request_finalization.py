"""C8.5 tests: publish_success, fail_request and get_request_result.

Runs against documind_test. Attempts come from the real start_attempt and mark_llm_started;
leases are moved with plain SQL; failures are injected with temporary database triggers.
"""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InvalidRequestError

import app.rag.chat_attempt_lifecycle as lifecycle_module
import app.rag.chat_request_finalization as finalization_module
from app.core.database import SessionLocal, engine
from app.rag.chat_attempt_lifecycle import (
    expire_stale_attempt,
    mark_llm_started,
    mark_uncertain,
)
from app.rag.chat_attempt_start import AttemptStartOutcome, start_attempt
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_quota_usage import get_daily_usage
from app.rag.chat_request_finalization import (
    FailOutcome,
    FailResult,
    FinalizationArgumentError,
    PublishOutcome,
    RejectOutcome,
    RequestResult,
    fail_request,
    get_request_result,
    publish_success,
    reject_and_fail_request,
)
from app.rag.chat_request_registration import RegistrationOutcome, register_request

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
LIMIT = 1_000_000
QUESTION = "What is DocuMind?"
ANSWER = "A RAG service."
_counter = [0]
P = PublishOutcome
R = RejectOutcome
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _not_near_utc_midnight():
    now = datetime.now(UTC)
    if now.hour == 23 and now.minute == 59 and now.second >= 30:
        pytest.skip("too close to UTC midnight")


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def make_user(email="alice@example.com"):
    return execute(
        "INSERT INTO users (email, hashed_password) VALUES (:e, 'x') RETURNING id",
        e=email,
    ).scalar_one()


def make_conversation(owner, messages=0, updated_at="2020-01-01T00:00:00+00:00"):
    cid = execute(
        "INSERT INTO conversations (owner_id, updated_at) VALUES (:o, :u) RETURNING id",
        o=owner,
        u=updated_at,
    ).scalar_one()
    for seq in range(1, messages + 1):
        execute(
            "INSERT INTO messages (conversation_id, seq, role, content) VALUES (:c, :s, :r, :t)",
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


def call(function, *args, **kwargs):
    with SessionLocal() as db:
        return function(db, *args, **kwargs)


def ready(owner, conversation_id=None, reserved=500, started=True, request_id=None):
    """(request_id, attempt_id): a request with a running (and by default started) attempt."""
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


def publish(owner, request_id, attempt_id, **overrides):
    arguments = {"user_content": QUESTION, "assistant_content": ANSWER} | overrides
    with SessionLocal() as db:
        return publish_success(db, owner, request_id, attempt_id, **arguments)


def expire_lease(attempt_id):
    execute(
        "UPDATE chat_request_attempts SET lease_expires_at = clock_timestamp()"
        " - interval '1 second' WHERE id = :i",
        i=attempt_id,
    )


def snapshot():
    """Every row of the four tables, as a comparable value."""
    with engine.begin() as conn:
        return {
            table: [
                tuple(r)
                for r in conn.execute(text(f"SELECT * FROM {table} ORDER BY id"))
            ]
            for table in (
                "conversations",
                "messages",
                "chat_requests",
                "chat_request_attempts",
            )
        }


def count(table):
    return execute(f"SELECT count(*) FROM {table}").scalar_one()


def request_row(request_id):
    return execute("SELECT * FROM chat_requests WHERE id = :i", i=request_id).one()


def attempt_row(attempt_id):
    return execute(
        "SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id
    ).one()


@contextmanager
def boom_trigger(table, when):
    execute(
        "CREATE FUNCTION c85_boom() RETURNS trigger LANGUAGE plpgsql AS"
        " $$ BEGIN RAISE EXCEPTION 'c85 injected failure'; END $$"
    )
    event = "INSERT" if table == "messages" else "UPDATE"
    execute(
        f"CREATE TRIGGER c85_boom BEFORE {event} ON {table} FOR EACH ROW"
        f" WHEN ({when}) EXECUTE FUNCTION c85_boom()"
    )
    try:
        yield
    finally:
        execute(f"DROP TRIGGER c85_boom ON {table}")
        execute("DROP FUNCTION c85_boom()")


@pytest.fixture
def owner():
    return make_user()


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


# ---------------------------------------------------------------- publishing


def test_publish_into_a_new_conversation(owner):
    rid, aid = ready(owner)
    result = publish(owner, rid, aid, prompt_tokens=120, completion_tokens=30)
    assert result.outcome is P.PUBLISHED
    conversation = execute("SELECT * FROM conversations").one()
    assert (conversation.id, conversation.owner_id, conversation.title) == (
        result.conversation_id,
        owner,
        None,
    )
    messages = execute("SELECT * FROM messages ORDER BY seq").all()
    assert [(m.conversation_id, m.seq, m.role, m.content) for m in messages] == [
        (conversation.id, 1, "user", QUESTION),
        (conversation.id, 2, "assistant", ANSWER),
    ]
    assert (messages[0].id, messages[1].id) == (
        result.user_message_id,
        result.assistant_message_id,
    )
    request = request_row(rid)
    assert (request.status, request.conversation_id, request.error_code) == (
        "succeeded",
        conversation.id,
        None,
    )
    assert (request.user_message_id, request.assistant_message_id) == (
        messages[0].id,
        messages[1].id,
    )
    assert request.creates_conversation is True and request.completed_at is not None
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.prompt_tokens, attempt.completion_tokens) == (
        "succeeded",
        120,
        30,
    )
    assert attempt.charged_tokens == 150 and attempt.finished_at is not None


def test_publish_without_usage_charges_the_reservation(owner):
    rid, aid = ready(owner, reserved=500)
    assert publish(owner, rid, aid).outcome is P.PUBLISHED
    attempt = attempt_row(aid)
    assert (attempt.status, attempt.prompt_tokens, attempt.charged_tokens) == (
        "succeeded",
        None,
        500,
    )


def test_content_is_stored_exactly(owner):
    rid, aid = ready(owner)
    question, answer = "  क्या है?\n", "\n  उत्तर 😀  "
    publish(owner, rid, aid, user_content=question, assistant_content=answer)
    assert [
        r.content for r in execute("SELECT content FROM messages ORDER BY seq").all()
    ] == [
        question,
        answer,
    ]


def test_publish_into_an_existing_conversation(owner):
    cid = make_conversation(owner, messages=2)
    rid, aid = ready(owner, conversation_id=cid)
    result = publish(owner, rid, aid)
    assert (result.outcome, result.conversation_id) == (P.PUBLISHED, cid)
    assert [r.seq for r in execute("SELECT seq FROM messages ORDER BY seq").all()] == [
        1,
        2,
        3,
        4,
    ]
    request = request_row(rid)
    assert (request.conversation_id, request.creates_conversation) == (cid, False)
    updated = execute("SELECT updated_at FROM conversations").scalar_one()
    assert updated > datetime(2026, 1, 1, tzinfo=UTC)  # moved forward from 2020


def test_sequence_continues_after_the_highest_existing_seq(owner):
    cid = make_conversation(owner)
    execute(
        "INSERT INTO messages (conversation_id, seq, role, content) VALUES (:c, 5, 'user', 'x')",
        c=cid,
    )
    rid, aid = ready(owner, conversation_id=cid)
    publish(owner, rid, aid)
    assert [r.seq for r in execute("SELECT seq FROM messages ORDER BY seq").all()] == [
        5,
        6,
        7,
    ]


def test_publish_goes_through_the_whole_c8_flow(owner):
    key, sha = "flow-1", "b" * 64
    with SessionLocal() as db:
        registered = register_request(db, owner, key, sha, conversation_id=None)
    assert registered.outcome is RegistrationOutcome.CREATED
    rid, aid = ready(owner, request_id=registered.request_id)
    assert (
        publish(owner, rid, aid, prompt_tokens=10, completion_tokens=5).outcome
        is P.PUBLISHED
    )
    with SessionLocal() as db:
        replay = register_request(db, owner, key, sha, conversation_id=None)
        usage = get_daily_usage(db, owner, datetime.now(UTC).date())
    assert replay.outcome is RegistrationOutcome.SUCCEEDED_REPLAY
    assert (usage.settled_tokens, usage.running_reserved_tokens) == (15, 0)
    stored = call(get_request_result, owner, replay.request_id)
    assert stored.status == "succeeded" and stored.assistant_message_id is not None


# ---------------------------------------------------------------- no-write outcomes


def test_a_second_publish_is_a_replay_and_changes_nothing(owner):
    rid, aid = ready(owner)
    first = publish(owner, rid, aid)
    before = snapshot()
    again = publish(
        owner, rid, aid, user_content="other q", assistant_content="other a"
    )
    assert again.outcome is P.ALREADY_SUCCEEDED
    assert (
        again.conversation_id,
        again.user_message_id,
        again.assistant_message_id,
    ) == (
        first.conversation_id,
        first.user_message_id,
        first.assistant_message_id,
    )
    assert snapshot() == before


def test_publish_on_a_failed_request_changes_nothing(owner):
    rid, aid = ready(owner)
    assert (
        call(mark_uncertain, aid, "timeout")
        is lifecycle_module.TransitionOutcome.APPLIED
    )
    assert call(fail_request, owner, rid, "gave_up").outcome is FailOutcome.FAILED
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.REQUEST_FAILED
    assert snapshot() == before


def test_unknown_or_foreign_request_and_attempt_pairs_are_not_found(owner, other_owner):
    rid, aid = ready(owner)
    other_rid, other_aid = ready(other_owner)
    before = snapshot()
    assert (
        publish(other_owner, rid, aid).outcome is P.REQUEST_NOT_FOUND
    )  # not their request
    assert (
        publish(owner, rid, other_aid).outcome is P.REQUEST_NOT_FOUND
    )  # attempt of another
    assert publish(owner, rid + 1000, aid).outcome is P.REQUEST_NOT_FOUND
    assert publish(owner, rid, aid + 1000).outcome is P.REQUEST_NOT_FOUND
    assert snapshot() == before
    assert other_rid != rid


# ---------------------------------------------------------------- fencing


def test_expired_lease_cannot_publish(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.LEASE_EXPIRED
    assert snapshot() == before
    assert (count("conversations"), count("messages")) == (0, 0)


def test_reaped_attempt_cannot_publish(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.NOT_RUNNING
    assert snapshot() == before


def test_unstarted_attempt_cannot_publish(owner):
    rid, aid = ready(owner, started=False)
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.WRONG_PHASE
    assert snapshot() == before


def test_stale_worker_cannot_publish_after_its_attempt_was_expired(owner):
    # C9.5a: an expired, started attempt is never replaced, so there is no takeover; the
    # stale worker simply finds its attempt settled and writes nothing.
    rid, old_attempt = ready(owner)
    expire_lease(old_attempt)
    call(expire_stale_attempt, rid)
    before = snapshot()
    assert publish(owner, rid, old_attempt).outcome is P.NOT_RUNNING  # the stale worker
    assert snapshot() == before
    old = attempt_row(old_attempt)
    # Settled by expiry: its charge is the retained reservation, and nothing changed it.
    assert (old.status, old.charged_tokens, old.reserved_tokens) == (
        "uncertain",
        500,
        500,
    )
    assert request_row(rid).status == "in_progress"


# ---------------------------------------------------------------- ownership and rollback


def test_existing_conversation_of_another_owner_rolls_everything_back(
    owner, other_owner
):
    foreign = make_conversation(other_owner, messages=2)
    rid, aid = ready(owner, conversation_id=foreign)
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.CONVERSATION_NOT_FOUND
    assert snapshot() == before  # the attempt settlement was rolled back too
    assert attempt_row(aid).status == "running"
    assert request_row(rid).status == "in_progress"


def test_deleted_conversation_rolls_everything_back(owner):
    cid = make_conversation(owner, messages=2)
    rid, aid = ready(owner, conversation_id=cid)
    execute(
        "DELETE FROM conversations WHERE id = :c", c=cid
    )  # request pointer becomes NULL
    before = snapshot()
    assert publish(owner, rid, aid).outcome is P.CONVERSATION_NOT_FOUND
    assert snapshot() == before
    assert attempt_row(aid).status == "running"


@pytest.mark.parametrize(
    "table, when, existing",
    [
        ("messages", "NEW.role = 'assistant'", False),  # user message already inserted
        ("messages", "NEW.role = 'assistant'", True),
        ("chat_requests", "NEW.status = 'succeeded'", False),  # everything else written
        ("chat_requests", "NEW.status = 'succeeded'", True),
    ],
)
def test_a_database_failure_at_any_step_leaves_nothing_behind(
    owner, table, when, existing
):
    cid = make_conversation(owner, messages=2) if existing else None
    rid, aid = ready(owner, conversation_id=cid)
    before = snapshot()
    with boom_trigger(table, when), pytest.raises(DBAPIError):
        publish(owner, rid, aid)
    assert snapshot() == before
    assert attempt_row(aid).status == "running"
    assert request_row(rid).status == "in_progress"
    # Nothing was left locked or half-done: the same publish works once the fault is gone.
    assert publish(owner, rid, aid).outcome is P.PUBLISHED


# ---------------------------------------------------------------- concurrency


def run_threads(n, work):
    barrier = threading.Barrier(n)

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return work(db, i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))


def test_concurrent_publishes_of_one_request_write_exactly_once(owner):
    rid, aid = ready(owner)
    results = run_threads(
        6,
        lambda db, i: publish_success(
            db, owner, rid, aid, user_content=QUESTION, assistant_content=ANSWER
        ),
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(P.PUBLISHED) == 1
    assert outcomes.count(P.ALREADY_SUCCEEDED) == 5
    assert len({(r.conversation_id, r.assistant_message_id) for r in results}) == 1
    assert (count("conversations"), count("messages")) == (1, 2)


def test_concurrent_publishes_to_one_conversation_get_distinct_sequence_numbers(owner):
    cid = make_conversation(owner)
    pairs = [ready(owner, conversation_id=cid) for _ in range(5)]
    results = run_threads(
        5,
        lambda db, i: publish_success(
            db,
            owner,
            pairs[i][0],
            pairs[i][1],
            user_content=QUESTION,
            assistant_content=ANSWER,
        ),
    )
    assert all(r.outcome is P.PUBLISHED for r in results)
    seqs = [r.seq for r in execute("SELECT seq FROM messages ORDER BY seq").all()]
    assert seqs == list(range(1, 11))
    # Each request's pair is adjacent: user seq is odd, assistant seq is the next one.
    for request_id, _ in pairs:
        row = request_row(request_id)
        user_seq = execute(
            "SELECT seq FROM messages WHERE id = :i", i=row.user_message_id
        )
        assistant_seq = execute(
            "SELECT seq FROM messages WHERE id = :i", i=row.assistant_message_id
        )
        assert assistant_seq.scalar_one() == user_seq.scalar_one() + 1


def test_publish_and_fail_request_never_both_win(owner):
    for _ in range(10):
        rid, aid = ready(owner)

        def work(db, i, rid=rid, aid=aid):
            if i == 0:
                return publish_success(
                    db, owner, rid, aid, user_content=QUESTION, assistant_content=ANSWER
                )
            return fail_request(db, owner, rid, "gave_up")

        published, failed = run_threads(2, work)
        assert published.outcome is P.PUBLISHED
        # fail_request refuses while the attempt runs, and a finished request stays succeeded.
        assert failed.outcome in (
            FailOutcome.ATTEMPT_RUNNING,
            FailOutcome.ALREADY_FINALIZED,
        )
        assert request_row(rid).status == "succeeded"


# ---------------------------------------------------------------- fail_request


def test_fail_request_without_attempts(owner):
    rid = make_request(owner)
    assert call(fail_request, owner, rid, "no_context") == FailResult(
        FailOutcome.FAILED
    )
    request = request_row(rid)
    assert (request.status, request.error_code) == ("failed", "no_context")
    assert request.completed_at is not None and request.conversation_id is None
    assert (count("conversations"), count("messages")) == (0, 0)


def test_fail_request_after_settled_attempts(owner):
    rid, aid = ready(owner)
    call(mark_uncertain, aid, "timeout")
    assert call(fail_request, owner, rid, "gave_up").outcome is FailOutcome.FAILED
    assert attempt_row(aid).status == "uncertain"  # attempts are not touched


@pytest.mark.parametrize("expired", [False, True])
def test_fail_request_is_refused_while_an_attempt_is_running(owner, expired):
    rid, aid = ready(owner)
    if expired:
        expire_lease(aid)
    before = snapshot()
    assert (
        call(fail_request, owner, rid, "gave_up").outcome is FailOutcome.ATTEMPT_RUNNING
    )
    assert snapshot() == before


def test_fail_request_does_not_touch_finalized_requests(owner):
    rid, aid = ready(owner)
    publish(owner, rid, aid)
    failed_rid = make_request(owner)
    call(fail_request, owner, failed_rid, "first")
    before = snapshot()
    assert call(fail_request, owner, rid, "second") == FailResult(
        FailOutcome.ALREADY_FINALIZED, "succeeded"
    )
    assert call(fail_request, owner, failed_rid, "second") == FailResult(
        FailOutcome.ALREADY_FINALIZED, "failed"
    )
    assert snapshot() == before


def test_fail_request_not_found(owner, other_owner):
    rid = make_request(owner)
    before = snapshot()
    assert call(fail_request, other_owner, rid, "code").outcome is FailOutcome.NOT_FOUND
    assert (
        call(fail_request, owner, rid + 1000, "code").outcome is FailOutcome.NOT_FOUND
    )
    assert snapshot() == before


@pytest.mark.parametrize(
    "bad", ["", "Code", "1code", "co de", "a" * 65, "code\n", None, 5]
)
def test_fail_request_rejects_invalid_error_codes(owner, bad):
    rid = make_request(owner)
    before = snapshot()
    with pytest.raises(FinalizationArgumentError):
        call(fail_request, owner, rid, bad)
    assert snapshot() == before


def test_error_code_format_matches_the_attempt_lifecycle():
    assert (
        finalization_module._ERROR_CODE_PATTERN == lifecycle_module._ERROR_CODE_PATTERN
    )


# ---------------------------------------------------------------- get_request_result


def test_get_request_result_for_every_state(owner, other_owner):
    pending = make_request(owner)
    done_rid, done_aid = ready(owner)
    published = publish(owner, done_rid, done_aid)
    failed = make_request(owner)
    call(fail_request, owner, failed, "gave_up")
    assert call(get_request_result, owner, pending) == RequestResult(
        "in_progress", None, None, None, None
    )
    assert call(get_request_result, owner, done_rid) == RequestResult(
        "succeeded",
        published.conversation_id,
        published.user_message_id,
        published.assistant_message_id,
        None,
    )
    assert call(get_request_result, owner, failed) == RequestResult(
        "failed", None, None, None, "gave_up"
    )
    assert call(get_request_result, other_owner, done_rid) is None
    assert call(get_request_result, owner, done_rid + 1000) is None


def test_get_request_result_after_the_conversation_is_deleted(owner):
    rid, aid = ready(owner)
    published = publish(owner, rid, aid)
    execute("DELETE FROM conversations WHERE id = :c", c=published.conversation_id)
    assert call(get_request_result, owner, rid) == RequestResult(
        "succeeded", None, None, None, None
    )


def test_get_request_result_is_read_only(owner):
    rid, aid = ready(owner)
    publish(owner, rid, aid)
    before = snapshot()
    call(get_request_result, owner, rid)
    assert snapshot() == before


# ---------------------------------------------------------------- validation, transactions


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1", 2_147_483_648])
def test_invalid_ids_are_rejected(owner, bad):
    rid, aid = ready(owner)
    for args in ((bad, rid, aid), (owner, bad, aid), (owner, rid, bad)):
        with pytest.raises(FinalizationArgumentError):
            publish(*args)
    for function in (get_request_result,):
        with pytest.raises(FinalizationArgumentError):
            call(function, bad, rid)
        with pytest.raises(FinalizationArgumentError):
            call(function, owner, bad)


@pytest.mark.parametrize(
    "bad", ["", " ", "   ", None, 5, b"x", "a\x00b", "\ud800", "ok\ud800"]
)
def test_invalid_content_is_rejected_and_nothing_is_written(owner, bad):
    rid, aid = ready(owner)
    before = snapshot()
    with pytest.raises(FinalizationArgumentError):
        publish(owner, rid, aid, user_content=bad)
    with pytest.raises(FinalizationArgumentError):
        publish(owner, rid, aid, assistant_content=bad)
    assert snapshot() == before


def test_content_that_only_the_python_strip_would_call_blank_is_accepted(owner):
    # The database CHECK trims spaces only, so a lone newline is valid content.
    rid, aid = ready(owner)
    assert publish(
        owner, rid, aid, user_content="\n", assistant_content="\t"
    ).outcome is (P.PUBLISHED)


@pytest.mark.parametrize(
    "usage",
    [
        (5, None),
        (None, 5),
        (-1, 1),
        (1, -1),
        (True, 1),
        (1, 1.5),
        (2_000_000_000, 2_000_000_000),
    ],
)
def test_invalid_usage_is_rejected_before_anything_runs(owner, usage):
    rid, aid = ready(owner)
    before = snapshot()
    with pytest.raises(ChatIdempotencyError):
        publish(owner, rid, aid, prompt_tokens=usage[0], completion_tokens=usage[1])
    assert snapshot() == before


def test_argument_error_shares_the_step1_base_class():
    assert issubclass(FinalizationArgumentError, ChatIdempotencyError)


def test_session_with_open_transaction_fails_loudly(owner):
    rid, aid = ready(owner)
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
        with pytest.raises(InvalidRequestError):
            publish_success(
                db, owner, rid, aid, user_content="q", assistant_content="a"
            )
    assert (count("messages"), request_row(rid).status) == (0, "in_progress")


def test_every_call_commits_and_leaves_no_open_transaction(owner):
    rid, aid = ready(owner)
    other_rid = make_request(owner)
    with SessionLocal() as db:
        assert publish_success(
            db, owner, rid, aid, user_content="q", assistant_content="a"
        ).outcome is (P.PUBLISHED)
        assert not db.in_transaction()
        assert publish_success(
            db, owner, rid, aid, user_content="q", assistant_content="a"
        ).outcome is (P.ALREADY_SUCCEEDED)
        assert not db.in_transaction()
        assert (
            fail_request(db, owner, other_rid, "gave_up").outcome is FailOutcome.FAILED
        )
        assert not db.in_transaction()
        assert get_request_result(db, owner, rid).status == "succeeded"
        assert not db.in_transaction()
    assert count("messages") == 2  # committed: visible from another connection


# ---------------------------------------------------------------- reject_and_fail_request
# C9.4a: one transaction settles a provider-rejected attempt (charge 0) AND fails its request.

CODE = "provider_rejected"


def reject(owner, request_id, attempt_id, code=CODE):
    with SessionLocal() as db:
        return reject_and_fail_request(db, owner, request_id, attempt_id, code)


def half_settled_count():
    """Rejected attempts whose request is still in progress: must always be zero."""
    return execute(
        "SELECT count(*) FROM chat_request_attempts a JOIN chat_requests r"
        " ON r.id = a.request_id WHERE a.status = 'rejected' AND r.status = 'in_progress'"
    ).scalar_one()


@pytest.mark.parametrize("existing_conversation", [False, True])
def test_reject_settles_zero_charge_and_fails_the_request_together(
    owner, existing_conversation
):
    cid = make_conversation(owner, messages=2) if existing_conversation else None
    rid, aid = ready(owner, conversation_id=cid, reserved=500)
    messages_before = count("messages")
    result = reject(owner, rid, aid)
    assert result.outcome is R.REJECTED and result.current_status is None
    a = attempt_row(aid)
    assert (a.status, a.charged_tokens, a.error_code) == ("rejected", 0, CODE)
    assert (a.prompt_tokens, a.completion_tokens, a.lease_expires_at) == (
        None,
        None,
        None,
    )
    assert a.llm_started_at is not None and a.finished_at is not None
    r = request_row(rid)
    assert (r.status, r.error_code) == ("failed", CODE)
    assert r.completed_at is not None
    assert (r.user_message_id, r.assistant_message_id) == (None, None)
    assert count("messages") == messages_before  # nothing published
    assert count("conversations") == (1 if existing_conversation else 0)
    assert half_settled_count() == 0


def test_the_stored_failure_is_what_a_replay_returns(owner):
    rid, aid = ready(owner)
    assert reject(owner, rid, aid, "provider_unavailable").outcome is R.REJECTED
    with SessionLocal() as db:
        assert get_request_result(db, owner, rid) == RequestResult(
            "failed", None, None, None, "provider_unavailable"
        )
    assert attempt_row(aid).error_code == "provider_unavailable"


# ---- association: owner, request and attempt are validated together, before any write


def test_attempt_request_owner_mismatches_are_not_found_and_change_nothing(
    owner, other_owner
):
    rid, aid = ready(owner)
    other_rid, other_aid = ready(other_owner)
    before = snapshot()
    assert (
        reject(other_owner, rid, aid).outcome is R.REQUEST_NOT_FOUND
    )  # not their request
    assert (
        reject(owner, rid, other_aid).outcome is R.REQUEST_NOT_FOUND
    )  # attempt of another
    assert (
        reject(owner, other_rid, aid).outcome is R.REQUEST_NOT_FOUND
    )  # request of another
    assert reject(owner, other_rid, other_aid).outcome is R.REQUEST_NOT_FOUND
    assert reject(owner, rid + 1000, aid).outcome is R.REQUEST_NOT_FOUND
    assert reject(owner, rid, aid + 1000).outcome is R.REQUEST_NOT_FOUND
    assert snapshot() == before  # both rows of both pairs untouched
    assert (
        reject(owner, rid, aid).outcome is R.REJECTED
    )  # the correct triple still works
    assert request_row(other_rid).status == "in_progress"
    assert attempt_row(other_aid).status == "running"


def test_a_second_request_of_the_owner_is_not_touched(owner):
    rid, aid = ready(owner)
    other_rid, other_aid = ready(owner)
    before = {t: r for t, r in snapshot().items()}
    assert reject(owner, rid, other_aid).outcome is R.REQUEST_NOT_FOUND
    assert snapshot() == before
    assert reject(owner, rid, aid).outcome is R.REJECTED
    assert request_row(other_rid).status == "in_progress"
    assert attempt_row(other_aid).status == "running"


# ---- the request must still be in progress


def test_a_second_identical_call_is_already_finalized_and_changes_nothing(owner):
    rid, aid = ready(owner)
    assert reject(owner, rid, aid).outcome is R.REJECTED
    after_first = snapshot()
    again = reject(owner, rid, aid)
    assert (again.outcome, again.current_status) == (R.ALREADY_FINALIZED, "failed")
    assert reject(owner, rid, aid, "other_code").outcome is R.ALREADY_FINALIZED
    assert snapshot() == after_first
    assert request_row(rid).error_code == CODE and attempt_row(aid).error_code == CODE


def test_a_succeeded_request_cannot_be_rejected(owner):
    rid, aid = ready(owner)
    assert publish(owner, rid, aid).outcome is P.PUBLISHED
    before = snapshot()
    result = reject(owner, rid, aid)
    assert (result.outcome, result.current_status) == (R.ALREADY_FINALIZED, "succeeded")
    assert snapshot() == before


def test_a_request_failed_by_fail_request_cannot_be_rejected(owner):
    rid, aid = ready(owner)
    assert (
        call(mark_uncertain, aid, "timeout")
        is lifecycle_module.TransitionOutcome.APPLIED
    )
    assert call(fail_request, owner, rid, "gave_up").outcome is FailOutcome.FAILED
    before = snapshot()
    result = reject(owner, rid, aid)
    assert (result.outcome, result.current_status) == (R.ALREADY_FINALIZED, "failed")
    assert snapshot() == before


# ---- the C8.4 fence is unchanged


def test_an_unstarted_attempt_cannot_be_rejected(owner):
    rid, aid = ready(owner, started=False)
    before = snapshot()
    assert reject(owner, rid, aid).outcome is R.WRONG_PHASE
    assert snapshot() == before and half_settled_count() == 0


def test_an_expired_lease_cannot_be_rejected(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    before = snapshot()
    assert reject(owner, rid, aid).outcome is R.LEASE_EXPIRED
    assert snapshot() == before  # still running until it is expired explicitly
    assert request_row(rid).status == "in_progress"


def test_a_reaped_attempt_cannot_be_rejected(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    before = snapshot()
    assert reject(owner, rid, aid).outcome is R.NOT_RUNNING
    assert snapshot() == before
    assert attempt_row(aid).status == "uncertain"
    assert request_row(rid).status == "in_progress"


@pytest.mark.parametrize("settle", ["uncertain", "rate_limited"])
def test_an_attempt_settled_by_another_transition_cannot_be_rejected(owner, settle):
    rid, aid = ready(owner)
    function = {
        "uncertain": lifecycle_module.mark_uncertain,
        "rate_limited": lifecycle_module.mark_rate_limited,
    }[settle]
    assert call(function, aid, "code") is lifecycle_module.TransitionOutcome.APPLIED
    before = snapshot()
    assert reject(owner, rid, aid).outcome is R.NOT_RUNNING
    assert snapshot() == before


def test_a_stale_worker_cannot_reject_after_its_attempt_was_expired(owner):
    # C9.5a: no takeover exists any more; the expired attempt is settled and stays so.
    rid, old_aid = ready(owner)
    expire_lease(old_aid)
    assert call(expire_stale_attempt, rid).attempt_id == old_aid
    before = snapshot()
    assert reject(owner, rid, old_aid).outcome is R.NOT_RUNNING
    assert snapshot() == before
    old = attempt_row(old_aid)
    assert (old.status, old.charged_tokens, old.reserved_tokens) == (
        "uncertain",
        500,
        500,
    )
    assert request_row(rid).status == "in_progress"


# ---- atomicity: both rows change together or not at all


@pytest.mark.parametrize(
    "table, when",
    [
        ("chat_request_attempts", "NEW.status = 'rejected'"),  # attempt update fails
        (
            "chat_requests",
            "NEW.status = 'failed'",
        ),  # request update fails after the attempt's
    ],
)
def test_a_database_failure_at_either_update_leaves_nothing_behind(owner, table, when):
    rid, aid = ready(owner)
    before = snapshot()
    with boom_trigger(table, when), pytest.raises(DBAPIError):
        reject(owner, rid, aid)
    assert snapshot() == before
    assert attempt_row(aid).status == "running"
    assert request_row(rid).status == "in_progress"
    assert half_settled_count() == 0
    # Nothing stayed locked or half-done: the same call works once the fault is gone.
    assert reject(owner, rid, aid).outcome is R.REJECTED


def test_a_python_failure_between_the_two_updates_rolls_the_attempt_back(
    owner, monkeypatch
):
    rid, aid = ready(owner)
    before = snapshot()

    def explode(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash between the two updates")

    monkeypatch.setattr(finalization_module, "_set_failed", explode)
    with pytest.raises(KeyboardInterrupt):
        reject(owner, rid, aid)
    assert snapshot() == before
    monkeypatch.undo()
    assert reject(owner, rid, aid).outcome is R.REJECTED


def test_a_failed_request_update_after_a_settled_attempt_rolls_everything_back(
    owner, monkeypatch
):
    rid, aid = ready(owner)
    before = snapshot()
    monkeypatch.setattr(finalization_module, "_set_failed", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        reject(owner, rid, aid)
    assert snapshot() == before
    assert half_settled_count() == 0


def test_a_failure_while_settling_the_attempt_leaves_the_request_untouched(
    owner, monkeypatch
):
    rid, aid = ready(owner)
    before = snapshot()

    def explode(*args, **kwargs):
        raise RuntimeError("settlement failed")

    monkeypatch.setattr(finalization_module, "settle_rejected_in_transaction", explode)
    with pytest.raises(RuntimeError):
        reject(owner, rid, aid)
    assert snapshot() == before


# ---- crash safety


CRASH_SCRIPT = """
import os, sys
import app.rag.chat_request_finalization as f
from app.core.database import SessionLocal

def die(*args, **kwargs):
    os._exit(17)          # the process dies between the attempt update and the request update

f._set_failed = die
owner, request_id, attempt_id = (int(v) for v in sys.argv[1:4])
with SessionLocal() as db:
    f.reject_and_fail_request(db, owner, request_id, attempt_id, "provider_rejected")
os._exit(0)
"""


def test_a_process_crash_between_the_updates_commits_neither_row(owner):
    rid, aid = ready(owner, reserved=500)
    before = snapshot()
    done = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT, str(owner), str(rid), str(aid)],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 17, done.stderr  # it really died mid-transaction
    assert snapshot() == before  # the server rolled the dead session back
    assert half_settled_count() == 0
    assert (
        attempt_row(aid).status == "running"
        and request_row(rid).status == "in_progress"
    )

    # Recovery is the existing conservative path: expire (uncertain, reservation kept), fail.
    assert call(fail_request, owner, rid, "x").outcome is FailOutcome.ATTEMPT_RUNNING
    expire_lease(aid)
    assert call(expire_stale_attempt, rid).attempt_id == aid
    a = attempt_row(aid)
    assert (a.status, a.charged_tokens, a.error_code) == (
        "uncertain",
        500,
        "lease_expired",
    )
    assert call(fail_request, owner, rid, "lease_expired").outcome is FailOutcome.FAILED
    assert half_settled_count() == 0


def test_the_half_settled_check_detects_a_violation_and_the_function_never_makes_one(
    owner,
):
    rid, aid = ready(owner)
    other = make_request(owner)
    execute(  # manufactured violation, bypassing the function (control for the query)
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status, reserved_tokens,"
        " charged_tokens, llm_started_at, finished_at, error_code)"
        " VALUES (:r, 1, 'rejected', 100, 0, now(), now(), 'x')",
        r=other,
    )
    assert half_settled_count() == 1
    execute("DELETE FROM chat_request_attempts WHERE request_id = :r", r=other)
    # Every outcome of the function leaves the invariant intact.
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
        reject(*args)
        assert half_settled_count() == 0


def test_rejection_is_only_reachable_through_the_atomic_operation():
    app_dir = PROJECT_ROOT / "app"
    users = {
        p.name
        for p in app_dir.rglob("*.py")
        if "settle_rejected_in_transaction" in p.read_text(encoding="utf-8")
    }
    assert users == {"chat_attempt_lifecycle.py", "chat_request_finalization.py"}
    assert not any(
        "def mark_rejected" in p.read_text(encoding="utf-8")
        for p in app_dir.rglob("*.py")
    )
    assert "reject_and_fail_request" in finalization_module.__all__
    assert "settle_rejected_in_transaction" not in finalization_module.__all__
    assert "settle_rejected_in_transaction" not in lifecycle_module.__all__


# ---- concurrency


def test_concurrent_rejects_of_one_attempt_apply_exactly_once(owner):
    rid, aid = ready(owner)
    results = run_threads(
        6, lambda db, i: reject_and_fail_request(db, owner, rid, aid, CODE)
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(R.REJECTED) == 1
    assert outcomes.count(R.ALREADY_FINALIZED) == 5
    assert (request_row(rid).status, attempt_row(aid).status) == ("failed", "rejected")


def test_reject_and_publish_never_both_win(owner):
    for _ in range(10):
        rid, aid = ready(owner)

        def work(db, i, rid=rid, aid=aid):
            if i == 0:
                return publish_success(
                    db, owner, rid, aid, user_content=QUESTION, assistant_content=ANSWER
                )
            return reject_and_fail_request(db, owner, rid, aid, CODE)

        published, rejected = run_threads(2, work)
        request, attempt = request_row(rid), attempt_row(aid)
        if published.outcome is P.PUBLISHED:
            assert rejected.outcome is R.ALREADY_FINALIZED
            assert (request.status, attempt.status) == ("succeeded", "succeeded")
        else:
            assert published.outcome is P.REQUEST_FAILED
            assert rejected.outcome is R.REJECTED
            assert (request.status, attempt.status) == ("failed", "rejected")
        assert half_settled_count() == 0


def test_reject_and_fail_request_never_conflict(owner):
    for _ in range(5):
        rid, aid = ready(owner)

        def work(db, i, rid=rid, aid=aid):
            if i == 0:
                return reject_and_fail_request(db, owner, rid, aid, CODE)
            return fail_request(db, owner, rid, "gave_up")

        rejected, failed = run_threads(2, work)
        assert (
            rejected.outcome is R.REJECTED
        )  # fail_request refuses while the attempt runs
        assert failed.outcome in (
            FailOutcome.ATTEMPT_RUNNING,
            FailOutcome.ALREADY_FINALIZED,
        )
        assert (request_row(rid).error_code, attempt_row(aid).status) == (
            CODE,
            "rejected",
        )


def test_reject_and_expiry_are_mutually_exclusive(owner):
    rid, aid = ready(owner)
    expire_lease(aid)
    rejected, expired = run_threads(
        2,
        lambda db, i: (
            reject_and_fail_request(db, owner, rid, aid, CODE)
            if i == 0
            else expire_stale_attempt(db, rid)
        ),
    )
    assert rejected.outcome in (R.LEASE_EXPIRED, R.NOT_RUNNING)
    assert expired.attempt_id == aid
    assert (attempt_row(aid).status, request_row(rid).status) == (
        "uncertain",
        "in_progress",
    )
    assert half_settled_count() == 0


def test_a_live_lease_is_never_expired_by_a_concurrent_reject(owner):
    rid, aid = ready(owner)
    rejected, expired = run_threads(
        2,
        lambda db, i: (
            reject_and_fail_request(db, owner, rid, aid, CODE)
            if i == 0
            else expire_stale_attempt(db, rid)
        ),
    )
    assert rejected.outcome is R.REJECTED
    assert expired.outcome is lifecycle_module.ExpireOutcome.NOTHING_TO_EXPIRE
    assert (attempt_row(aid).status, request_row(rid).status) == ("rejected", "failed")


# ---- quota accounting and what the request can do afterwards


def test_a_rejection_releases_the_reservation_and_charges_nothing(owner, other_owner):
    rid, aid = ready(owner, reserved=500)
    other_rid, other_aid = ready(other_owner, reserved=700)
    today = datetime.now(UTC).date()
    with SessionLocal() as db:
        before = get_daily_usage(db, owner, today)
    assert (before.running_reserved_tokens, before.settled_tokens) == (500, 0)
    assert reject(owner, rid, aid).outcome is R.REJECTED
    with SessionLocal() as db:
        after = get_daily_usage(db, owner, today)
        other = get_daily_usage(db, other_owner, today)
    assert (
        after.running_reserved_tokens,
        after.settled_tokens,
        after.total_tokens,
    ) == (0, 0, 0)
    assert (
        other.running_reserved_tokens == 700
    )  # another owner's reservation is untouched
    assert attempt_row(other_aid).status == "running"
    assert request_row(other_rid).status == "in_progress"


def test_the_request_cannot_start_another_attempt_after_a_rejection(owner):
    rid, aid = ready(owner)
    assert reject(owner, rid, aid).outcome is R.REJECTED
    result = call(
        start_attempt,
        owner,
        rid,
        reserved_tokens=500,
        daily_limit_tokens=LIMIT,
        lease_seconds=600,
    )
    assert result.outcome is AttemptStartOutcome.REQUEST_NOT_IN_PROGRESS
    assert count("chat_request_attempts") == 1


def test_a_rejection_after_a_429_retry_fails_the_request_with_two_settled_attempts(
    owner,
):
    rid, first = ready(owner)
    assert call(lifecycle_module.mark_rate_limited, first, "http_429") is (
        lifecycle_module.TransitionOutcome.APPLIED
    )
    _, second = ready(owner, request_id=rid)
    assert reject(owner, rid, second).outcome is R.REJECTED
    statuses = [
        r.status
        for r in execute(
            "SELECT status FROM chat_request_attempts ORDER BY attempt_number"
        ).all()
    ]
    assert statuses == ["rate_limited", "rejected"]
    assert request_row(rid).status == "failed"


# ---- arguments and transactions


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1", 2_147_483_648])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_invalid_ids_are_rejected_before_anything_runs(owner, bad, position):
    rid, aid = ready(owner)
    before = snapshot()
    args = [owner, rid, aid]
    args[position] = bad
    with pytest.raises(FinalizationArgumentError):
        reject(*args)
    assert snapshot() == before


@pytest.mark.parametrize(
    "bad", ["", "A", "1code", "co de", "co-de", "a" * 65, "code\n", None, 5]
)
def test_invalid_error_codes_are_rejected_before_anything_runs(owner, bad):
    rid, aid = ready(owner)
    before = snapshot()
    with pytest.raises(FinalizationArgumentError):
        reject(owner, rid, aid, bad)
    assert snapshot() == before


def test_reject_session_with_open_transaction_fails_loudly(owner):
    rid, aid = ready(owner)
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
        with pytest.raises(InvalidRequestError):
            reject_and_fail_request(db, owner, rid, aid, CODE)
    assert (request_row(rid).status, attempt_row(aid).status) == (
        "in_progress",
        "running",
    )


def test_reject_commits_and_leaves_no_open_transaction(owner):
    rid, aid = ready(owner)
    with SessionLocal() as db:
        assert reject_and_fail_request(db, owner, rid, aid, CODE).outcome is R.REJECTED
        assert not db.in_transaction()
        assert (
            reject_and_fail_request(db, owner, rid, aid, CODE).outcome
            is R.ALREADY_FINALIZED
        )
        assert not db.in_transaction()
        assert reject_and_fail_request(db, owner, rid + 999, aid, CODE).outcome is (
            R.REQUEST_NOT_FOUND
        )
        assert not db.in_transaction()
    assert (request_row(rid).status, attempt_row(aid).status) == ("failed", "rejected")


def test_fail_request_still_writes_the_same_failed_state(owner):
    """Regression for the _set_failed extraction: fail_request behaves as before."""
    rid = make_request(owner)
    assert call(fail_request, owner, rid, "gave_up").outcome is FailOutcome.FAILED
    r = request_row(rid)
    assert (r.status, r.error_code) == ("failed", "gave_up")
    assert r.completed_at is not None
    assert (
        call(fail_request, owner, rid, "again").outcome is FailOutcome.ALREADY_FINALIZED
    )
