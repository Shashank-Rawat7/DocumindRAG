"""C8.1 step 2 tests: register_request (database-backed idempotent registration).

Runs against documind_test (built by conftest with `alembic upgrade head`). Each test
body uses fresh sessions, because register_request owns its transaction.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, InvalidRequestError

from app.core.database import SessionLocal, engine
from app.rag.chat_idempotency import ChatIdempotencyError, IdempotencyKeyError
from app.rag.chat_request_registration import (
    RegistrationArgumentError,
    RegistrationOutcome,
    RegistrationResult,
    register_request,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
KEY = "key-1"


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def rows(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params).all()


def make_user(email):
    return execute(
        "INSERT INTO users (email, hashed_password) VALUES (:e, 'x') RETURNING id",
        e=email,
    ).scalar_one()


def make_conversation(owner_id):
    return execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=owner_id
    ).scalar_one()


def register(owner_id, key=KEY, sha=HASH_A, conversation_id=None):
    with SessionLocal() as db:
        return register_request(db, owner_id, key, sha, conversation_id=conversation_id)


def request_rows():
    return rows(
        "SELECT id, owner_id, idempotency_key, request_sha256, status, creates_conversation,"
        " conversation_id, error_code, completed_at FROM chat_requests ORDER BY id"
    )


@pytest.fixture
def owner():
    return make_user("alice@example.com")


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


def set_status(status):
    if status == "in_progress":
        return
    if status == "succeeded":
        execute("UPDATE chat_requests SET status='succeeded', completed_at=now()")
    else:
        execute(
            "UPDATE chat_requests SET status='failed', completed_at=now(), error_code='boom'"
        )


# ---------------------------------------------------------------- creation


def test_new_request_creates_exactly_one_row(owner):
    result = register(owner)
    assert isinstance(result, RegistrationResult)
    assert result.outcome is RegistrationOutcome.CREATED
    (row,) = request_rows()
    assert result.request_id == row.id
    assert (row.owner_id, row.idempotency_key, row.request_sha256) == (
        owner,
        KEY,
        HASH_A,
    )
    assert row.status == "in_progress"
    assert row.creates_conversation is True
    assert row.conversation_id is None
    assert row.error_code is None and row.completed_at is None


def test_existing_conversation_request_stores_target(owner):
    conv = make_conversation(owner)
    result = register(owner, conversation_id=conv)
    assert result.outcome is RegistrationOutcome.CREATED
    (row,) = request_rows()
    assert row.creates_conversation is False
    assert row.conversation_id == conv


def test_conversation_id_is_required_keyword(owner):
    with SessionLocal() as db, pytest.raises(TypeError):
        register_request(db, owner, KEY, HASH_A)  # type: ignore[call-arg]


# ---------------------------------------------------------------- replays


@pytest.mark.parametrize(
    "status, outcome",
    [
        ("in_progress", RegistrationOutcome.IN_PROGRESS_REPLAY),
        ("succeeded", RegistrationOutcome.SUCCEEDED_REPLAY),
        ("failed", RegistrationOutcome.FAILED_REPLAY),
    ],
)
def test_replay_reports_stored_status_and_leaves_row_unchanged(owner, status, outcome):
    first = register(owner)
    set_status(status)
    before = request_rows()
    again = register(owner)
    assert again == RegistrationResult(outcome, first.request_id)
    assert request_rows() == before


@pytest.mark.parametrize("status", ["in_progress", "succeeded", "failed"])
def test_different_hash_is_a_mismatch_and_leaves_row_unchanged(owner, status):
    first = register(owner)
    set_status(status)
    before = request_rows()
    again = register(owner, sha=HASH_B)
    assert again == RegistrationResult(
        RegistrationOutcome.KEY_REUSED_DIFFERENT_REQUEST, first.request_id
    )
    assert request_rows() == before


def test_replay_ignores_conversation_target_and_does_not_change_row(owner):
    conv = make_conversation(owner)
    register(owner)
    before = request_rows()
    again = register(owner, conversation_id=conv)
    assert again.outcome is RegistrationOutcome.IN_PROGRESS_REPLAY
    assert request_rows() == before


# ---------------------------------------------------------------- isolation


def test_different_owners_get_independent_rows(owner, other_owner):
    a = register(owner)
    b = register(other_owner, sha=HASH_B)
    assert a.outcome is b.outcome is RegistrationOutcome.CREATED
    assert a.request_id != b.request_id
    assert len(request_rows()) == 2
    # Each owner's key is checked only against that owner's rows.
    assert (
        register(other_owner, sha=HASH_B).outcome
        is RegistrationOutcome.IN_PROGRESS_REPLAY
    )
    assert register(other_owner, sha=HASH_A).outcome is (
        RegistrationOutcome.KEY_REUSED_DIFFERENT_REQUEST
    )


def test_different_keys_for_one_owner_are_independent(owner):
    assert register(owner, key="k1").outcome is RegistrationOutcome.CREATED
    assert register(owner, key="k2").outcome is RegistrationOutcome.CREATED
    assert len(request_rows()) == 2


# ---------------------------------------------------------------- concurrency


def run_threads(n, work):
    barrier = threading.Barrier(n)

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return work(db, i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))


def test_concurrent_same_key_same_hash_creates_exactly_one_row(owner):
    results = run_threads(
        8, lambda db, i: register_request(db, owner, KEY, HASH_A, conversation_id=None)
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(RegistrationOutcome.CREATED) == 1
    assert outcomes.count(RegistrationOutcome.IN_PROGRESS_REPLAY) == 7
    assert len({r.request_id for r in results}) == 1
    assert len(request_rows()) == 1


@pytest.mark.parametrize("n", [2, 8])
def test_concurrent_same_key_different_hashes_creates_one_row(owner, n):
    hashes = [HASH_A if i % 2 == 0 else HASH_B for i in range(n)]
    results = run_threads(
        n,
        lambda db, i: register_request(db, owner, KEY, hashes[i], conversation_id=None),
    )
    (row,) = request_rows()
    winner_hash = row.request_sha256
    outcomes = [r.outcome for r in results]
    assert outcomes.count(RegistrationOutcome.CREATED) == 1
    for result, sha in zip(results, hashes):
        assert result.request_id == row.id
        if result.outcome is RegistrationOutcome.CREATED:
            assert sha == winner_hash
        elif sha == winner_hash:
            assert result.outcome is RegistrationOutcome.IN_PROGRESS_REPLAY
        else:
            assert result.outcome is RegistrationOutcome.KEY_REUSED_DIFFERENT_REQUEST
    # Both hashes were submitted, so at least one caller must have seen a mismatch.
    assert RegistrationOutcome.KEY_REUSED_DIFFERENT_REQUEST in outcomes


def test_concurrent_different_owners_same_key_do_not_interfere(owner, other_owner):
    owners = [owner, other_owner] * 3
    results = run_threads(
        6,
        lambda db, i: register_request(
            db, owners[i], KEY, HASH_A, conversation_id=None
        ),
    )
    assert [r.outcome for r in results].count(RegistrationOutcome.CREATED) == 2
    assert len(request_rows()) == 2


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", ["", "a b", "x" * 129, "é", None, 5, b"k"])
def test_invalid_key_is_rejected(owner, bad):
    with pytest.raises(IdempotencyKeyError):
        register(owner, key=bad)
    assert request_rows() == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        " " + "a" * 63,
        "a" * 64 + "\n",
        None,
        5,
    ],
)
def test_invalid_hash_is_rejected(owner, bad):
    with pytest.raises(RegistrationArgumentError):
        register(owner, sha=bad)
    assert request_rows() == []


@pytest.mark.parametrize("bad", [None, True, 1.0, "1"])
def test_invalid_owner_is_rejected(bad):
    with pytest.raises(RegistrationArgumentError):
        register(bad)


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_invalid_conversation_id_is_rejected(owner, bad):
    with pytest.raises(RegistrationArgumentError):
        register(owner, conversation_id=bad)
    assert request_rows() == []


def test_argument_errors_share_the_step1_base_class():
    assert issubclass(RegistrationArgumentError, ChatIdempotencyError)


# ---------------------------------------------------------------- transactions


def test_session_with_open_transaction_fails_loudly(owner):
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))  # autobegin
        with pytest.raises(InvalidRequestError):
            register_request(db, owner, KEY, HASH_A, conversation_id=None)
    assert request_rows() == []


def test_commits_and_leaves_no_open_transaction(owner):
    with SessionLocal() as db:
        register_request(db, owner, KEY, HASH_A, conversation_id=None)
        assert not db.in_transaction()
        # A replay (no insert) also leaves the session clean.
        register_request(db, owner, KEY, HASH_A, conversation_id=None)
        assert not db.in_transaction()
    assert len(request_rows()) == 1  # visible from another connection: it committed


def test_nonexistent_owner_raises_integrity_error_and_leaves_no_row():
    with SessionLocal() as db:
        with pytest.raises(IntegrityError):
            register_request(db, 999_999, KEY, HASH_A, conversation_id=None)
        assert not db.in_transaction()
    assert request_rows() == []
