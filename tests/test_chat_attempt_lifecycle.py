"""C8.4 tests: the fenced attempt lifecycle (app/rag/chat_attempt_lifecycle.py).

Runs against documind_test. Attempts are created through the real start_attempt; leases are
moved with plain SQL so expiry can be tested without sleeping.
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

import app.rag.chat_attempt_lifecycle as lifecycle_module
from app.core.database import SessionLocal, engine
from app.rag.chat_attempt_lifecycle import (
    LEASE_EXPIRED_CODE,
    AttemptLifecycleArgumentError,
    ExpireOutcome,
    ExpireResult,
    TransitionOutcome,
    expire_stale_attempt,
    fail_before_call,
    fail_with_usage,
    mark_llm_started,
    mark_rate_limited,
    mark_uncertain,
    settle_rejected_in_transaction,
    settle_succeeded_in_transaction,
)
from app.rag.chat_attempt_start import AttemptStartOutcome, start_attempt
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_quota_usage import get_daily_usage

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
LIMIT = 100_000
_counter = [0]
APPLIED = TransitionOutcome.APPLIED


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


def make_request(owner):
    _counter[0] += 1
    return execute(
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status,"
        " creates_conversation) VALUES (:o, :k, :h, 'in_progress', true) RETURNING id",
        o=owner,
        k=f"k{_counter[0]}",
        h="a" * 64,
    ).scalar_one()


def complete_succeeded(db, attempt_id, prompt_tokens=None, completion_tokens=None):
    """Test stand-in for the removed public function: settle success in its own transaction.

    Production code can only settle a success inside publish_success (see
    test_chat_request_finalization.py); this wrapper exists to exercise the fence.
    """
    with db.begin():
        return settle_succeeded_in_transaction(
            db, attempt_id, prompt_tokens, completion_tokens
        )


def complete_rejected(db, attempt_id, error_code):
    """Test stand-in: settle a rejection in its own transaction (production: only inside
    reject_and_fail_request, see test_chat_request_finalization.py)."""
    with db.begin():
        return settle_rejected_in_transaction(db, attempt_id, error_code)


def call(function, *args, **kwargs):
    with SessionLocal() as db:
        return function(db, *args, **kwargs)


def new_attempt(owner, reserved=100, lease=600, started=False, request_id=None):
    """A running attempt made by the real start_attempt, optionally already started."""
    request_id = request_id or make_request(owner)
    result = call(
        start_attempt,
        owner,
        request_id,
        reserved_tokens=reserved,
        daily_limit_tokens=LIMIT,
        lease_seconds=lease,
    )
    assert result.outcome is AttemptStartOutcome.STARTED, result
    if started:
        assert call(mark_llm_started, result.attempt_id) is APPLIED
    return result.attempt_id, request_id


def expire_lease(attempt_id):
    execute(
        "UPDATE chat_request_attempts SET lease_expires_at = clock_timestamp() - interval"
        " '1 second' WHERE id = :i",
        i=attempt_id,
    )


def row(attempt_id):
    return tuple(
        execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    )


@pytest.fixture
def owner():
    return make_user()


# every worker transition, as (name, callable(attempt_id), needs_started)
WORKER_CALLS = {
    "mark_llm_started": lambda a: call(mark_llm_started, a),
    "succeeded": lambda a: call(complete_succeeded, a, 10, 5),
    "succeeded_no_usage": lambda a: call(complete_succeeded, a),
    "fail_before_call": lambda a: call(fail_before_call, a, "retrieval_failed"),
    "fail_with_usage": lambda a: call(fail_with_usage, a, "bad_output", 7, 3),
    "uncertain": lambda a: call(mark_uncertain, a, "timeout"),
    "rate_limited": lambda a: call(mark_rate_limited, a, "http_429"),
    "rejected": lambda a: call(complete_rejected, a, "provider_rejected"),
}


# ---------------------------------------------------------------- mark_llm_started


def test_mark_llm_started_stamps_once_and_is_committed(owner):
    attempt_id, _ = new_attempt(owner)
    before = datetime.now(UTC)
    assert call(mark_llm_started, attempt_id) is APPLIED
    after = datetime.now(UTC)
    stamped = execute(
        "SELECT status, llm_started_at FROM chat_request_attempts WHERE id = :i",
        i=attempt_id,
    ).one()  # a different connection: the stamp is committed
    assert stamped.status == "running"
    assert before - timedelta(seconds=1) <= stamped.llm_started_at <= after
    # A second call must not move the timestamp (it would move the accounting day).
    assert call(mark_llm_started, attempt_id) is TransitionOutcome.WRONG_PHASE
    assert (
        execute(
            "SELECT llm_started_at FROM chat_request_attempts WHERE id = :i",
            i=attempt_id,
        ).scalar_one()
        == stamped.llm_started_at
    )


# ---------------------------------------------------------------- settlements


def test_succeeded_with_usage(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(complete_succeeded, attempt_id, 120, 30) is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.prompt_tokens, r.completion_tokens, r.charged_tokens) == (
        "succeeded",
        120,
        30,
        150,
    )
    assert (
        r.lease_expires_at is None
        and r.finished_at is not None
        and r.error_code is None
    )


def test_succeeded_without_usage_charges_the_reservation(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(complete_succeeded, attempt_id) is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.prompt_tokens, r.completion_tokens, r.charged_tokens) == (
        "succeeded",
        None,
        None,
        500,
    )


def test_usage_may_exceed_the_reservation(owner):
    attempt_id, _ = new_attempt(owner, reserved=10, started=True)
    assert call(complete_succeeded, attempt_id, 400, 600) is APPLIED
    assert row_charged(attempt_id) == 1000


def row_charged(attempt_id):
    return execute(
        "SELECT charged_tokens FROM chat_request_attempts WHERE id = :i", i=attempt_id
    ).scalar_one()


def test_fail_before_call(owner):
    attempt_id, _ = new_attempt(owner, reserved=500)
    assert call(fail_before_call, attempt_id, "retrieval_failed") is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.charged_tokens, r.error_code) == (
        "failed_before_call",
        0,
        "retrieval_failed",
    )
    assert (
        r.llm_started_at is None
        and r.lease_expires_at is None
        and r.finished_at is not None
    )


def test_fail_with_usage(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(fail_with_usage, attempt_id, "bad_output", 40, 2) is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (
        r.status,
        r.prompt_tokens,
        r.completion_tokens,
        r.charged_tokens,
        r.error_code,
    ) == (
        "failed_with_usage",
        40,
        2,
        42,
        "bad_output",
    )


def test_uncertain_keeps_the_reservation(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(mark_uncertain, attempt_id, "timeout") is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.charged_tokens, r.prompt_tokens, r.error_code) == (
        "uncertain",
        500,
        None,
        "timeout",
    )


def test_rate_limited_charges_nothing(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(mark_rate_limited, attempt_id, "http_429") is APPLIED
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.charged_tokens, r.error_code) == ("rate_limited", 0, "http_429")


def test_rejected_charges_nothing_and_records_no_usage(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    assert call(complete_rejected, attempt_id, "provider_rejected") is APPLIED
    r = row_named(attempt_id)
    assert (r.status, r.charged_tokens, r.error_code) == (
        "rejected",
        0,
        "provider_rejected",
    )
    assert (r.prompt_tokens, r.completion_tokens, r.lease_expires_at) == (
        None,
        None,
        None,
    )
    assert r.llm_started_at is not None and r.finished_at is not None
    assert call(get_daily_usage, owner, datetime.now(UTC).date()).total_tokens == 0


def row_named(attempt_id):
    return execute(
        "SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id
    ).one()


# ---------------------------------------------------------------- phases


@pytest.mark.parametrize(
    "name",
    [
        "succeeded",
        "succeeded_no_usage",
        "fail_with_usage",
        "uncertain",
        "rate_limited",
        "rejected",
    ],
)
def test_started_only_transitions_reject_an_unstarted_attempt(owner, name):
    attempt_id, _ = new_attempt(owner)
    before = row(attempt_id)
    assert WORKER_CALLS[name](attempt_id) is TransitionOutcome.WRONG_PHASE
    assert row(attempt_id) == before


@pytest.mark.parametrize("name", ["mark_llm_started", "fail_before_call"])
def test_unstarted_only_transitions_reject_a_started_attempt(owner, name):
    attempt_id, _ = new_attempt(owner, started=True)
    before = row(attempt_id)
    assert WORKER_CALLS[name](attempt_id) is TransitionOutcome.WRONG_PHASE
    assert row(attempt_id) == before


# ---------------------------------------------------------------- the fence


@pytest.mark.parametrize(
    "settle", ["succeeded", "uncertain", "rate_limited", "fail_with_usage", "rejected"]
)
def test_a_settled_attempt_cannot_be_changed_by_any_stale_worker(owner, settle):
    attempt_id, _ = new_attempt(owner, started=True)
    assert WORKER_CALLS[settle](attempt_id) is APPLIED
    settled = row(attempt_id)
    for name, worker_call in WORKER_CALLS.items():
        assert worker_call(attempt_id) is TransitionOutcome.NOT_RUNNING, name
    assert row(attempt_id) == settled


def test_a_settled_unstarted_attempt_is_also_fenced(owner):
    attempt_id, _ = new_attempt(owner)
    assert WORKER_CALLS["fail_before_call"](attempt_id) is APPLIED
    settled = row(attempt_id)
    for name, worker_call in WORKER_CALLS.items():
        assert worker_call(attempt_id) is TransitionOutcome.NOT_RUNNING, name
    assert row(attempt_id) == settled


@pytest.mark.parametrize("started", [False, True])
def test_expired_lease_rejects_every_worker_write(owner, started):
    attempt_id, _ = new_attempt(owner, started=started)
    expire_lease(attempt_id)
    before = row(attempt_id)
    for name, worker_call in WORKER_CALLS.items():
        assert worker_call(attempt_id) is TransitionOutcome.LEASE_EXPIRED, name
    assert (
        row(attempt_id) == before
    )  # still running, untouched, until expired explicitly


@pytest.mark.parametrize("started", [False, True])
def test_stale_worker_cannot_change_an_attempt_after_expiry(owner, started):
    attempt_id, request_id = new_attempt(owner, started=started)
    expire_lease(attempt_id)
    assert call(expire_stale_attempt, request_id).attempt_id == attempt_id
    expired = row(attempt_id)
    for name, worker_call in WORKER_CALLS.items():
        assert worker_call(attempt_id) is TransitionOutcome.NOT_RUNNING, name
    assert row(attempt_id) == expired


def test_unknown_attempt_is_not_found():
    for name, worker_call in WORKER_CALLS.items():
        assert worker_call(999_999) is TransitionOutcome.NOT_FOUND, name


# ---------------------------------------------------------------- expiry


def test_expiring_an_unstarted_attempt_abandons_it(owner):
    attempt_id, request_id = new_attempt(owner, reserved=500)
    expire_lease(attempt_id)
    assert call(expire_stale_attempt, request_id) == ExpireResult(
        ExpireOutcome.ABANDONED, attempt_id
    )
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.charged_tokens, r.error_code) == (
        "abandoned",
        0,
        LEASE_EXPIRED_CODE,
    )
    assert (
        r.llm_started_at is None
        and r.lease_expires_at is None
        and r.finished_at is not None
    )


def test_expiring_a_started_attempt_makes_it_uncertain(owner):
    attempt_id, request_id = new_attempt(owner, reserved=500, started=True)
    expire_lease(attempt_id)
    assert call(expire_stale_attempt, request_id) == ExpireResult(
        ExpireOutcome.UNCERTAIN, attempt_id
    )
    r = execute("SELECT * FROM chat_request_attempts WHERE id = :i", i=attempt_id).one()
    assert (r.status, r.charged_tokens, r.error_code) == (
        "uncertain",
        500,
        LEASE_EXPIRED_CODE,
    )
    assert r.llm_started_at is not None and r.finished_at is not None


def test_a_live_lease_is_never_expired(owner):
    attempt_id, request_id = new_attempt(owner, started=True)
    before = row(attempt_id)
    assert call(expire_stale_attempt, request_id) == ExpireResult(
        ExpireOutcome.NOTHING_TO_EXPIRE
    )
    assert row(attempt_id) == before


def test_expire_ignores_settled_attempts_other_requests_and_missing_requests(owner):
    settled_id, settled_request = new_attempt(owner, started=True)
    assert WORKER_CALLS["succeeded"](settled_id) is APPLIED
    other_id, other_request = new_attempt(owner, started=True)
    expire_lease(other_id)
    before_settled = row(settled_id)
    assert (
        call(expire_stale_attempt, settled_request).outcome
        is ExpireOutcome.NOTHING_TO_EXPIRE
    )
    assert (
        call(expire_stale_attempt, 999_999).outcome is ExpireOutcome.NOTHING_TO_EXPIRE
    )
    assert row(settled_id) == before_settled
    assert (
        row(other_id)[3] == "running"
    )  # status column: other request untouched so far
    assert call(expire_stale_attempt, other_request).outcome is ExpireOutcome.UNCERTAIN


def test_takeover_after_expiry_counts_the_retained_reservation(owner):
    attempt_id, request_id = new_attempt(owner, reserved=500, started=True)
    expire_lease(attempt_id)
    # Not yet expired: a new attempt is refused.
    refused = call(
        start_attempt,
        owner,
        request_id,
        reserved_tokens=100,
        daily_limit_tokens=LIMIT,
        lease_seconds=60,
    )
    assert refused.outcome is AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING
    assert call(expire_stale_attempt, request_id).outcome is ExpireOutcome.UNCERTAIN
    second = call(
        start_attempt,
        owner,
        request_id,
        reserved_tokens=100,
        daily_limit_tokens=LIMIT,
        lease_seconds=60,
    )
    assert second.outcome is AttemptStartOutcome.STARTED
    assert (second.attempt_number, second.used_tokens) == (2, 500)


# ---------------------------------------------------------------- end to end


def test_full_flow_charge_is_what_the_quota_sees(owner):
    attempt_id, request_id = new_attempt(owner, reserved=500, started=True)
    assert call(complete_succeeded, attempt_id, 120, 30) is APPLIED
    with SessionLocal() as db:
        usage = get_daily_usage(db, owner, datetime.now(UTC).date())
    assert (usage.settled_tokens, usage.running_reserved_tokens) == (150, 0)
    # The request itself is untouched: finalization is C8.5.
    assert execute(
        "SELECT status FROM chat_requests WHERE id = :i", i=request_id
    ).scalar_one() == ("in_progress")


# ---------------------------------------------------------------- concurrency


def test_concurrent_settlements_of_one_attempt_apply_exactly_once(owner):
    attempt_id, _ = new_attempt(owner, reserved=500, started=True)
    barrier = threading.Barrier(10)
    calls = [
        lambda db: complete_succeeded(db, attempt_id, 10, 5),
        lambda db: mark_uncertain(db, attempt_id, "timeout"),
        lambda db: mark_rate_limited(db, attempt_id, "http_429"),
        lambda db: fail_with_usage(db, attempt_id, "bad_output", 1, 1),
        lambda db: complete_rejected(db, attempt_id, "provider_rejected"),
    ] * 2

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return calls[i](db)

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(task, range(10)))
    assert outcomes.count(APPLIED) == 1
    assert outcomes.count(TransitionOutcome.NOT_RUNNING) == 9
    final = execute("SELECT status FROM chat_request_attempts").scalar_one()
    assert final in {
        "succeeded",
        "uncertain",
        "rate_limited",
        "failed_with_usage",
        "rejected",
    }


def race_worker_against_expiry(attempt_id, request_id, lease_ms, jitter_ms):
    """One round: a worker settles while another session tries to expire the same attempt."""
    barrier = threading.Barrier(2)
    results = {}

    def worker():
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            time.sleep(max(0, lease_ms - jitter_ms) / 1000)
            results["worker"] = complete_succeeded(db, attempt_id, 10, 5)

    def expirer():
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                result = expire_stale_attempt(db, request_id)
                if result.outcome is not ExpireOutcome.NOTHING_TO_EXPIRE:
                    results["expire"] = result
                    return
                status = execute(
                    "SELECT status FROM chat_request_attempts WHERE id = :i",
                    i=attempt_id,
                ).scalar_one()
                if status != "running":
                    return  # the worker settled it first
                time.sleep(0.002)

    threads = [threading.Thread(target=worker), threading.Thread(target=expirer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return results


def test_worker_settlement_and_expiry_are_mutually_exclusive_at_the_lease_boundary(
    owner,
):
    """Hammer the lease boundary: exactly one of settlement and expiry may win."""
    rng = random.Random(7)
    wins = {"worker": 0, "expire": 0}
    for _ in range(25):
        attempt_id, request_id = new_attempt(owner, reserved=500, started=True)
        lease_ms = rng.randint(5, 60)
        execute(
            "UPDATE chat_request_attempts SET lease_expires_at = clock_timestamp()"
            " + make_interval(secs => :s) WHERE id = :i",
            s=lease_ms / 1000,
            i=attempt_id,
        )
        results = race_worker_against_expiry(
            attempt_id, request_id, lease_ms, rng.randint(0, 10)
        )
        applied = results["worker"] is APPLIED
        expired = "expire" in results
        assert applied != expired, (results, lease_ms)  # exactly one wins
        final = execute(
            "SELECT status, charged_tokens FROM chat_request_attempts WHERE id = :i",
            i=attempt_id,
        ).one()
        status, charged = final.status, final.charged_tokens
        if applied:
            wins["worker"] += 1
            assert (status, charged) == ("succeeded", 15)
        else:
            wins["expire"] += 1
            assert (status, charged) == ("uncertain", 500)
            assert results["worker"] in (
                TransitionOutcome.LEASE_EXPIRED,
                TransitionOutcome.NOT_RUNNING,
            )
    assert sum(wins.values()) == 25


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1", 2_147_483_648])
def test_invalid_attempt_and_request_ids_are_rejected(bad):
    for function, args in (
        (mark_llm_started, ()),
        (complete_succeeded, ()),
        (fail_before_call, ("code",)),
        (fail_with_usage, ("code", 1, 1)),
        (mark_uncertain, ("code",)),
        (mark_rate_limited, ("code",)),
        (complete_rejected, ("code",)),
        (expire_stale_attempt, ()),
    ):
        with pytest.raises(AttemptLifecycleArgumentError):
            call(function, bad, *args)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "A",
        "Code",
        "1code",
        "_code",
        "co de",
        "co-de",
        "a" * 65,
        "code\n",
        "é",
        None,
        5,
    ],
)
def test_invalid_error_codes_are_rejected(owner, bad):
    attempt_id, _ = new_attempt(owner, started=True)
    before = row(attempt_id)
    for function in (
        mark_uncertain,
        mark_rate_limited,
        fail_before_call,
        complete_rejected,
    ):
        with pytest.raises(AttemptLifecycleArgumentError):
            call(function, attempt_id, bad)
    with pytest.raises(AttemptLifecycleArgumentError):
        call(fail_with_usage, attempt_id, bad, 1, 1)
    assert row(attempt_id) == before


@pytest.mark.parametrize("code", ["a", "http_429", "a" * 64, "x1_y2"])
def test_valid_error_codes_are_accepted(owner, code):
    attempt_id, _ = new_attempt(owner, started=True)
    assert call(mark_uncertain, attempt_id, code) is APPLIED


@pytest.mark.parametrize(
    "prompt, completion",
    [
        (-1, 1),
        (1, -1),
        (True, 1),
        (1, 1.5),
        ("1", 1),
        (2_147_483_648, 0),
        (2_000_000_000, 2_000_000_000),
    ],
)
def test_invalid_usage_is_rejected(owner, prompt, completion):
    attempt_id, _ = new_attempt(owner, started=True)
    before = row(attempt_id)
    with pytest.raises(AttemptLifecycleArgumentError):
        call(complete_succeeded, attempt_id, prompt, completion)
    with pytest.raises(AttemptLifecycleArgumentError):
        call(fail_with_usage, attempt_id, "code", prompt, completion)
    assert row(attempt_id) == before


@pytest.mark.parametrize("usage", [(5, None), (None, 5)])
def test_partial_usage_is_rejected_for_success(owner, usage):
    attempt_id, _ = new_attempt(owner, started=True)
    with pytest.raises(AttemptLifecycleArgumentError):
        call(complete_succeeded, attempt_id, *usage)


def test_argument_error_shares_the_step1_base_class():
    assert issubclass(AttemptLifecycleArgumentError, ChatIdempotencyError)


def test_every_transition_commits_and_leaves_no_open_transaction(owner):
    attempt_id, request_id = new_attempt(owner)
    with SessionLocal() as db:
        assert mark_llm_started(db, attempt_id) is APPLIED
        assert not db.in_transaction()
        assert complete_succeeded(db, attempt_id) is APPLIED
        assert not db.in_transaction()
        assert complete_succeeded(db, attempt_id) is TransitionOutcome.NOT_RUNNING
        assert not db.in_transaction()
        assert (
            expire_stale_attempt(db, request_id).outcome
            is ExpireOutcome.NOTHING_TO_EXPIRE
        )
        assert not db.in_transaction()


def test_standalone_rejection_is_not_public():
    assert "settle_rejected_in_transaction" not in lifecycle_module.__all__
    assert not hasattr(lifecycle_module, "mark_rejected")
    assert not hasattr(lifecycle_module, "complete_rejected")


def test_standalone_success_settlement_is_not_public():
    assert "complete_succeeded" not in lifecycle_module.__all__
    assert not hasattr(lifecycle_module, "complete_succeeded")
    assert "settle_succeeded_in_transaction" not in lifecycle_module.__all__
