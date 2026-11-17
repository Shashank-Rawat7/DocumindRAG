"""C8.3 tests: start_attempt (atomic quota reservation and attempt start).

Runs against documind_test. Rows are seeded with plain SQL that satisfies the C7.2 CHECKs.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import InvalidRequestError

from app.core.database import SessionLocal, engine
from app.rag.chat_attempt_start import (
    AttemptStartArgumentError,
    AttemptStartOutcome,
    AttemptStartResult,
    start_attempt,
)
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_quota_usage import get_daily_usage

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
LIMIT = 1000
LEASE = 600
_counter = [0]


@pytest.fixture(autouse=True)
def _not_near_utc_midnight():
    """Tests seed rows relative to 'now'; avoid straddling a UTC day change."""
    now = datetime.now(UTC)
    if now.hour == 23 and now.minute == 59 and now.second >= 30:
        pytest.skip("too close to UTC midnight")


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def rows(sql, **params):
    return execute(sql, **params).all()


def make_user(email):
    return execute(
        "INSERT INTO users (email, hashed_password) VALUES (:e, 'x') RETURNING id",
        e=email,
    ).scalar_one()


def make_request(owner, status="in_progress"):
    _counter[0] += 1
    sql = (
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status,"
        " creates_conversation, completed_at, error_code)"
        " VALUES (:o, :k, :h, :s, true, :c, :e) RETURNING id"
    )
    return execute(
        sql,
        o=owner,
        k=f"k{_counter[0]}",
        h="a" * 64,
        s=status,
        c=None if status == "in_progress" else datetime.now(UTC),
        e="boom" if status == "failed" else None,
    ).scalar_one()


def add_attempt(
    request_id,
    status,
    number=1,
    started="now",
    reserved=100,
    charged=None,
    lease_delta=timedelta(hours=1),
):
    """Insert one attempt that satisfies every CHECK. started: 'now', None or a datetime."""
    running = status == "running"
    if started == "now":
        started = datetime.now(UTC) - timedelta(seconds=5)
    execute(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status,"
        " lease_expires_at, reserved_tokens, charged_tokens, error_code, llm_started_at,"
        " finished_at) VALUES (:r, :n, :s, :lease, :res, :ch, :err, :st, :fin)",
        r=request_id,
        n=number,
        s=status,
        lease=datetime.now(UTC) + lease_delta if running else None,
        res=reserved,
        ch=charged,
        err=None if status in ("running", "succeeded") else "code",
        st=started,
        fin=None if running else datetime.now(UTC),
    )


def seed_used(owner, tokens, status="uncertain", started="now"):
    """Seed `tokens` of usage for the owner through a fresh request."""
    rid = make_request(owner)
    if status == "running":
        add_attempt(rid, "running", started=started, reserved=tokens)
    else:
        add_attempt(rid, status, started=started, reserved=tokens, charged=tokens)
    return rid


def start(owner, request_id, **overrides):
    arguments = {
        "reserved_tokens": 100,
        "daily_limit_tokens": LIMIT,
        "lease_seconds": LEASE,
    } | overrides
    with SessionLocal() as db:
        return start_attempt(db, owner, request_id, **arguments)


def attempt_rows():
    return rows(
        "SELECT id, request_id, attempt_number, status, reserved_tokens, charged_tokens,"
        " llm_started_at, lease_expires_at, finished_at FROM chat_request_attempts ORDER BY id"
    )


@pytest.fixture
def owner():
    return make_user("alice@example.com")


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


# ---------------------------------------------------------------- starting


def test_started_attempt_row(owner):
    rid = make_request(owner)
    before = datetime.now(UTC)
    result = start(owner, rid, reserved_tokens=123)
    after = datetime.now(UTC)
    (row,) = attempt_rows()
    assert result == AttemptStartResult(
        AttemptStartOutcome.STARTED, attempt_id=row.id, attempt_number=1, used_tokens=0
    )
    assert (row.request_id, row.attempt_number, row.status) == (rid, 1, "running")
    assert row.reserved_tokens == 123
    assert row.charged_tokens is None and row.llm_started_at is None
    assert row.finished_at is None
    assert before + timedelta(seconds=LEASE) <= row.lease_expires_at
    assert row.lease_expires_at <= after + timedelta(seconds=LEASE)
    # The request itself is untouched.
    assert rows("SELECT status, completed_at FROM chat_requests") == [
        ("in_progress", None)
    ]


def test_next_attempt_number_follows_earlier_attempts(owner):
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    add_attempt(rid, "uncertain", number=2, reserved=50, charged=50)
    result = start(owner, rid, reserved_tokens=10)
    assert result.outcome is AttemptStartOutcome.STARTED
    assert result.attempt_number == 3
    assert result.used_tokens == 50


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_finished_request_is_rejected(owner, status):
    rid = make_request(owner, status)
    assert start(owner, rid).outcome is AttemptStartOutcome.REQUEST_NOT_IN_PROGRESS
    assert attempt_rows() == []


def test_unknown_request_and_other_owners_request_are_not_found(owner, other_owner):
    rid = make_request(other_owner)
    assert start(owner, rid).outcome is AttemptStartOutcome.REQUEST_NOT_FOUND
    assert start(owner, rid + 1000).outcome is AttemptStartOutcome.REQUEST_NOT_FOUND
    assert start(owner + 1000, rid).outcome is AttemptStartOutcome.REQUEST_NOT_FOUND
    assert attempt_rows() == []


@pytest.mark.parametrize("lease_delta", [timedelta(hours=1), timedelta(hours=-1)])
def test_running_attempt_blocks_a_new_one_even_if_expired(owner, lease_delta):
    rid = make_request(owner)
    add_attempt(rid, "running", started=None, lease_delta=lease_delta)
    assert start(owner, rid).outcome is AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING
    assert len(attempt_rows()) == 1  # no takeover, no second row


# ---------------------------------------------------------------- quota rule


@pytest.mark.parametrize(
    "reserved, outcome",
    [
        (100, AttemptStartOutcome.STARTED),  # used 900 + 100 == limit: allowed (<=)
        (101, AttemptStartOutcome.QUOTA_EXCEEDED),
    ],
)
def test_limit_comparison_is_inclusive(owner, reserved, outcome):
    seed_used(owner, 900)
    rid = make_request(owner)
    result = start(owner, rid, reserved_tokens=reserved)
    assert result.outcome is outcome
    assert result.used_tokens == 900


def test_denial_creates_nothing_and_leaves_request_in_progress(owner):
    seed_used(owner, 950)
    rid = make_request(owner)
    before = attempt_rows()
    result = start(owner, rid, reserved_tokens=100)
    assert result == AttemptStartResult(
        AttemptStartOutcome.QUOTA_EXCEEDED, used_tokens=950
    )
    assert attempt_rows() == before
    assert rows("SELECT status FROM chat_requests WHERE id = :i", i=rid) == [
        ("in_progress",)
    ]


def test_counted_usage_components(owner):
    seed_used(owner, 10, status="succeeded")  # settled charge
    seed_used(owner, 20, status="uncertain")  # reservation retained
    seed_used(owner, 30, status="running")  # running, started
    seed_used(owner, 40, status="running", started=None)  # running, unstarted, live
    rid = make_request(owner)
    assert start(owner, rid, reserved_tokens=1).used_tokens == 100


def test_expired_unstarted_running_attempt_is_not_counted(owner):
    r1 = make_request(owner)
    add_attempt(
        r1, "running", started=None, reserved=500, lease_delta=timedelta(hours=-1)
    )
    rid = make_request(owner)
    assert start(owner, rid, reserved_tokens=1).used_tokens == 0


def test_live_unstarted_attempt_blocks_the_quota(owner):
    seed_used(owner, 950, status="running", started=None)
    rid = make_request(owner)
    assert (
        start(owner, rid, reserved_tokens=100).outcome
        is AttemptStartOutcome.QUOTA_EXCEEDED
    )


def test_expired_but_started_running_attempt_still_counts(owner):
    r1 = make_request(owner)
    add_attempt(
        r1, "running", started="now", reserved=500, lease_delta=timedelta(hours=-1)
    )
    rid = make_request(owner)
    assert start(owner, rid, reserved_tokens=1).used_tokens == 500


def test_other_days_and_other_owners_and_zero_charges_are_not_counted(
    owner, other_owner
):
    seed_used(owner, 700, started=datetime.now(UTC) - timedelta(days=1, hours=1))
    seed_used(other_owner, 700)
    seed_used(other_owner, 700, status="running", started=None)
    r_zero = make_request(owner)
    add_attempt(r_zero, "failed_before_call", started=None, charged=0, reserved=900)
    rid = make_request(owner)
    assert start(owner, rid, reserved_tokens=1).used_tokens == 0


def test_started_usage_matches_the_historical_calculation(owner):
    seed_used(owner, 10, status="succeeded")
    seed_used(owner, 20, status="uncertain")
    seed_used(owner, 30, status="running")
    today = datetime.now(UTC).date()
    with SessionLocal() as db:
        historical = get_daily_usage(db, owner, today).total_tokens
    rid = make_request(owner)
    assert start(owner, rid, reserved_tokens=1).used_tokens == historical == 60


# ---------------------------------------------------------------- 429 retry cap


def test_first_429_allows_exactly_one_retry(owner):
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    result = start(owner, rid)
    assert result.outcome is AttemptStartOutcome.STARTED
    assert result.attempt_number == 2


def test_second_429_ends_retries(owner):
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    add_attempt(rid, "rate_limited", number=2, charged=0)
    before = attempt_rows()
    assert start(owner, rid).outcome is AttemptStartOutcome.RETRY_LIMIT_REACHED
    assert attempt_rows() == before
    assert rows("SELECT status FROM chat_requests WHERE id = :i", i=rid) == [
        ("in_progress",)
    ]


def test_only_rate_limited_attempts_count_towards_the_cap(owner):
    rid = make_request(owner)
    add_attempt(rid, "uncertain", number=1, reserved=1, charged=1)
    add_attempt(rid, "failed_before_call", number=2, started=None, charged=0)
    add_attempt(rid, "succeeded", number=3, reserved=1, charged=1)
    assert start(owner, rid).outcome is AttemptStartOutcome.STARTED


def test_a_rejected_attempt_does_not_count_towards_the_cap(owner):
    # Seeded directly: in production a rejection also fails its request (C9.4a). This pins
    # that the 429 cap counts 'rate_limited' attempts only.
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    add_attempt(rid, "rejected", number=2, charged=0)
    assert start(owner, rid).outcome is AttemptStartOutcome.STARTED


def test_a_rejected_attempt_consumes_no_quota(owner):
    done = make_request(owner)
    add_attempt(
        done, "rejected", number=1, reserved=999, charged=0
    )  # charged 0, whatever reserved
    rid = make_request(owner)
    assert (
        start(owner, rid, reserved_tokens=LIMIT).outcome is AttemptStartOutcome.STARTED
    )


def test_the_cap_is_per_request(owner):
    capped = make_request(owner)
    add_attempt(capped, "rate_limited", number=1, charged=0)
    add_attempt(capped, "rate_limited", number=2, charged=0)
    fresh = make_request(owner)
    assert start(owner, capped).outcome is AttemptStartOutcome.RETRY_LIMIT_REACHED
    assert start(owner, fresh).outcome is AttemptStartOutcome.STARTED


def test_a_retry_still_goes_through_the_quota_check(owner):
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    seed_used(owner, 950)
    assert (
        start(owner, rid, reserved_tokens=100).outcome
        is AttemptStartOutcome.QUOTA_EXCEEDED
    )


# ---------------------------------------------------------------- concurrency


def run_threads(n, work):
    barrier = threading.Barrier(n)

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return work(db, i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))


def test_concurrent_reservations_never_exceed_the_limit(owner):
    request_ids = [make_request(owner) for _ in range(8)]
    results = run_threads(
        8,
        lambda db, i: start_attempt(
            db,
            owner,
            request_ids[i],
            reserved_tokens=100,
            daily_limit_tokens=350,
            lease_seconds=LEASE,
        ),
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(AttemptStartOutcome.STARTED) == 3
    assert outcomes.count(AttemptStartOutcome.QUOTA_EXCEEDED) == 5
    assert sum(r.reserved_tokens for r in attempt_rows()) == 300
    assert sorted(r.used_tokens for r in results if r.attempt_id) == [0, 100, 200]


def test_concurrent_starts_of_one_request_create_one_attempt(owner):
    rid = make_request(owner)
    results = run_threads(
        6,
        lambda db, i: start_attempt(
            db,
            owner,
            rid,
            reserved_tokens=10,
            daily_limit_tokens=LIMIT,
            lease_seconds=LEASE,
        ),
    )
    outcomes = [r.outcome for r in results]
    assert outcomes.count(AttemptStartOutcome.STARTED) == 1
    assert outcomes.count(AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING) == 5
    assert len(attempt_rows()) == 1


def test_owners_do_not_block_or_count_each_other_concurrently(owner, other_owner):
    owners = [owner, other_owner] * 3
    request_ids = [make_request(o) for o in owners]
    results = run_threads(
        6,
        lambda db, i: start_attempt(
            db,
            owners[i],
            request_ids[i],
            reserved_tokens=100,
            daily_limit_tokens=200,
            lease_seconds=LEASE,
        ),
    )
    # Each owner fits two reservations of 100 under a 200 limit.
    assert [r.outcome for r in results].count(AttemptStartOutcome.STARTED) == 4
    assert [r.outcome for r in results].count(AttemptStartOutcome.QUOTA_EXCEEDED) == 2


# ---------------------------------------------------------------- validation, transactions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reserved_tokens": 0},
        {"reserved_tokens": -1},
        {"reserved_tokens": True},
        {"reserved_tokens": 1.5},
        {"reserved_tokens": 2_147_483_648},
        {"daily_limit_tokens": 0},
        {"daily_limit_tokens": None},
        {"daily_limit_tokens": "5"},
        {"lease_seconds": 0},
        {"lease_seconds": None},
        {"lease_seconds": 1.0},
    ],
)
def test_invalid_numeric_arguments_are_rejected(owner, kwargs):
    rid = make_request(owner)
    with pytest.raises(AttemptStartArgumentError):
        start(owner, rid, **kwargs)
    assert attempt_rows() == []


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1"])
def test_invalid_ids_are_rejected(owner, bad):
    rid = make_request(owner)
    with pytest.raises(AttemptStartArgumentError):
        start(bad, rid)
    with pytest.raises(AttemptStartArgumentError):
        start(owner, bad)


def test_limit_and_lease_are_required_keywords(owner):
    rid = make_request(owner)
    with SessionLocal() as db, pytest.raises(TypeError):
        start_attempt(db, owner, rid, reserved_tokens=1)  # type: ignore[call-arg]


def test_argument_error_shares_the_step1_base_class():
    assert issubclass(AttemptStartArgumentError, ChatIdempotencyError)


def test_session_with_open_transaction_fails_loudly(owner):
    rid = make_request(owner)
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
        with pytest.raises(InvalidRequestError):
            start_attempt(
                db,
                owner,
                rid,
                reserved_tokens=1,
                daily_limit_tokens=LIMIT,
                lease_seconds=LEASE,
            )
    assert attempt_rows() == []


def test_commits_and_leaves_no_open_transaction(owner):
    rid = make_request(owner)
    with SessionLocal() as db:
        start_attempt(
            db,
            owner,
            rid,
            reserved_tokens=1,
            daily_limit_tokens=LIMIT,
            lease_seconds=LEASE,
        )
        assert not db.in_transaction()
    assert len(attempt_rows()) == 1  # visible from another connection
