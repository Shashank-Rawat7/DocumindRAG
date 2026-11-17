"""C8.3 tests: start_attempt (atomic quota reservation and attempt start).

Runs against documind_test. Rows are seeded with plain SQL that satisfies the C7.2 CHECKs.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, text
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
    usage=None,
):
    """Insert one attempt that satisfies every CHECK. started: 'now', None or a datetime.

    usage: optional (prompt_tokens, completion_tokens), required by failed_with_usage.
    """
    running = status == "running"
    if started == "now":
        started = datetime.now(UTC) - timedelta(seconds=5)
    execute(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status,"
        " lease_expires_at, reserved_tokens, charged_tokens, error_code, llm_started_at,"
        " finished_at, prompt_tokens, completion_tokens)"
        " VALUES (:r, :n, :s, :lease, :res, :ch, :err, :st, :fin, :pt, :ct)",
        r=request_id,
        n=number,
        s=status,
        lease=datetime.now(UTC) + lease_delta if running else None,
        res=reserved,
        ch=charged,
        err=None if status in ("running", "succeeded") else "code",
        st=started,
        fin=None if running else datetime.now(UTC),
        pt=None if usage is None else usage[0],
        ct=None if usage is None else usage[1],
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
    # Only attempts that provably never reached the provider may precede a new one (C9.5a).
    # The 50 tokens of usage live on another request, so the quota figure is still pinned.
    seed_used(owner, 50)
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    add_attempt(rid, "abandoned", number=2, started=None, charged=0)
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
    # Unstarted attempts never count towards the 429 cap (paid ones are handled by the
    # admission guard, see the C9.5a section below).
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    add_attempt(rid, "abandoned", number=2, started=None, charged=0)
    add_attempt(rid, "failed_before_call", number=3, started=None, charged=0)
    assert start(owner, rid).outcome is AttemptStartOutcome.STARTED


def test_a_rejected_attempt_blocks_admission_and_is_not_a_429(owner):
    # Seeded directly: in production a rejection also fails its request (C9.4a). A rejected
    # attempt is not a 429 (the cap is not what answers), and it is a started attempt, so
    # the C9.5a guard answers.
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    add_attempt(rid, "rejected", number=2, charged=0)
    before = attempt_rows()
    result = start(owner, rid)
    assert result.outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    assert attempt_rows() == before


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


# ---------------------------------------------------------------- C9.5a admission guard

PAID = ("succeeded", "failed_with_usage", "uncertain", "rejected")


def add_paid(request_id, status, number=1):
    """Seed an attempt that ended after llm_started_at was set and was not a 429."""
    if status == "failed_with_usage":
        add_attempt(
            request_id, status, number=number, reserved=10, charged=10, usage=(7, 3)
        )
    elif status == "rejected":
        add_attempt(request_id, status, number=number, reserved=10, charged=0)
    else:  # succeeded / uncertain without usage are charged the reservation
        add_attempt(request_id, status, number=number, reserved=10, charged=10)


def settle_running(request_id):
    """What expiry or a worker does to a running, started attempt: running -> uncertain."""
    execute(
        "UPDATE chat_request_attempts SET status = 'uncertain', charged_tokens ="
        " reserved_tokens, error_code = 'lease_expired', lease_expires_at = NULL,"
        " finished_at = clock_timestamp()"
        " WHERE request_id = :r AND status = 'running'",
        r=request_id,
    )


def request_status(request_id):
    return rows("SELECT status FROM chat_requests WHERE id = :i", i=request_id)


@pytest.mark.parametrize("status", PAID)
def test_a_paid_attempt_blocks_any_new_admission(owner, status):
    rid = make_request(owner)
    add_paid(rid, status)
    before = attempt_rows()
    result = start(owner, rid)
    assert result == AttemptStartResult(
        AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    )
    assert attempt_rows() == before
    assert request_status(rid) == [("in_progress",)]


@pytest.mark.parametrize("status", PAID)
def test_a_paid_attempt_after_a_429_still_blocks(owner, status):
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    add_paid(rid, status, number=2)
    before = attempt_rows()
    assert start(owner, rid).outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    assert attempt_rows() == before


@pytest.mark.parametrize("status", ["failed_before_call", "abandoned"])
def test_a_prior_unstarted_attempt_allows_a_legitimate_new_one(owner, status):
    rid = make_request(owner)
    add_attempt(rid, status, number=1, started=None, charged=0)
    result = start(owner, rid)
    assert result.outcome is AttemptStartOutcome.STARTED
    assert result.attempt_number == 2


def test_several_unstarted_attempts_then_a_paid_one_then_blocked(owner):
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    add_attempt(rid, "abandoned", number=2, started=None, charged=0)
    add_attempt(rid, "failed_before_call", number=3, started=None, charged=0)
    first = start(owner, rid)
    assert first.outcome is AttemptStartOutcome.STARTED
    assert first.attempt_number == 4
    # The fourth attempt reaches the provider and ends uncertain (what expiry does).
    execute(
        "UPDATE chat_request_attempts SET llm_started_at = clock_timestamp() WHERE id = :i",
        i=first.attempt_id,
    )
    settle_running(rid)
    before = attempt_rows()
    assert start(owner, rid).outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    assert attempt_rows() == before


def test_the_429_retry_policy_is_unchanged(owner):
    once = make_request(owner)
    add_attempt(once, "rate_limited", number=1, charged=0)
    twice = make_request(owner)
    add_attempt(twice, "rate_limited", number=1, charged=0)
    add_attempt(twice, "rate_limited", number=2, charged=0)
    assert start(owner, once).outcome is AttemptStartOutcome.STARTED
    assert start(owner, twice).outcome is AttemptStartOutcome.RETRY_LIMIT_REACHED


def test_the_guard_answers_before_the_429_cap_and_the_quota_check(owner):
    capped = make_request(owner)
    add_attempt(capped, "rate_limited", number=1, charged=0)
    add_attempt(capped, "rate_limited", number=2, charged=0)
    add_paid(capped, "uncertain", number=3)
    assert (
        start(owner, capped).outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    )
    seed_used(owner, 950)
    paid = make_request(owner)
    add_paid(paid, "succeeded")
    assert (
        start(owner, paid, reserved_tokens=100).outcome
        is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    )


def test_the_running_check_answers_before_the_guard(owner):
    rid = make_request(owner)
    add_paid(rid, "uncertain", number=1)
    add_attempt(rid, "running", number=2)  # seeded: cannot occur in production
    assert start(owner, rid).outcome is AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING


def test_the_guard_is_per_request(owner):
    paid = make_request(owner)
    add_paid(paid, "uncertain")
    fresh = make_request(owner)
    assert (
        start(owner, paid).outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    )
    assert start(owner, fresh).outcome is AttemptStartOutcome.STARTED


def test_a_finished_request_is_still_reported_as_finished(owner):
    rid = make_request(owner, status="succeeded")
    add_paid(rid, "succeeded")
    assert start(owner, rid).outcome is AttemptStartOutcome.REQUEST_NOT_IN_PROGRESS


# --- statement order and isolation (the guard's exactness depends on both)


@contextmanager
def sql_trace():
    """Every statement start_attempt sends, with the isolation level it ran under."""
    log = []

    def before(conn, cursor, statement, parameters, context, executemany):
        raw = cursor.connection.cursor()  # same connection, same transaction
        raw.execute("SHOW transaction_isolation")
        isolation = raw.fetchone()[0]
        raw.close()
        log.append((" ".join(statement.split()), dict(parameters or {}), isolation))

    event.listen(engine, "before_cursor_execute", before)
    try:
        yield log
    finally:
        event.remove(engine, "before_cursor_execute", before)


def index_of(log, *needles):
    found = [i for i, (sql, _, _) in enumerate(log) if all(n in sql for n in needles)]
    assert len(found) == 1, (needles, found)
    return found[0]


def is_guard(entry):
    sql, params, _ = entry
    return "chat_request_attempts.status IN" in sql and set(params.values()) >= set(
        PAID
    )


def is_running_check(entry):
    sql, params, _ = entry
    return (
        "chat_request_attempts.status = " in sql
        and "count(*)" not in sql
        and "LIMIT" in sql
        and "running" in params.values()
    )


def test_guard_statement_order_and_isolation_level(owner):
    rid = make_request(owner)
    with sql_trace() as log:
        assert start(owner, rid).outcome is AttemptStartOutcome.STARTED
    lock_user = index_of(log, "FROM users", "FOR UPDATE")
    lock_request = index_of(log, "FROM chat_requests", "FOR UPDATE")
    running = [i for i, e in enumerate(log) if is_running_check(e)]
    guard = [i for i, e in enumerate(log) if is_guard(e)]
    cap = index_of(log, "count(*)")
    quota = index_of(log, "coalesce(sum(CASE")
    insert = index_of(log, "INSERT INTO chat_request_attempts")
    # Two separate statements, one each, in exactly this order.
    assert len(running) == 1 and len(guard) == 1 and running != guard
    assert lock_user < lock_request < running[0] < guard[0] < cap < quota < insert
    # The guard predicate is its own statement: it never mentions 'running' ...
    assert "running" not in log[guard[0]][1].values()
    # ... and the running check never mentions the paid statuses.
    assert not set(PAID) & set(log[running[0]][1].values())
    # One transaction at PostgreSQL's READ COMMITTED for every statement (no override).
    assert {isolation for _, _, isolation in log} == {"read committed"}
    with engine.connect() as connection:
        assert connection.get_isolation_level().upper() == "READ COMMITTED"
    assert rows("SHOW default_transaction_isolation") == [("read committed",)]


def test_a_denied_guard_runs_the_same_prefix_and_writes_nothing(owner):
    rid = make_request(owner)
    add_paid(rid, "uncertain")
    with sql_trace() as log:
        result = start(owner, rid)
    assert result.outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    guard = [i for i, e in enumerate(log) if is_guard(e)]
    assert len(guard) == 1 and guard[0] == len(log) - 1  # nothing runs after the guard
    assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql, _, _ in log)
    assert {isolation for _, _, isolation in log} == {"read committed"}


# --- concurrency


def hold_request_lock(request_id):
    conn = engine.connect()
    tx = conn.begin()
    conn.execute(
        text("SELECT id FROM chat_requests WHERE id = :i FOR UPDATE"), {"i": request_id}
    )
    return conn, tx


def wait_for_lock_waiter(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiting = rows(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
            " AND wait_event_type = 'Lock' AND pid <> pg_backend_pid()"
        )[0][0]
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("no start_attempt is waiting on the request lock")


@pytest.mark.parametrize(
    ("settle_while_waiting", "expected"),
    [
        (True, AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED),
        (False, AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING),
    ],
)
def test_start_that_waited_for_the_request_lock_sees_a_settlement_committed_meanwhile(
    owner, settle_while_waiting, expected
):
    rid = make_request(owner)
    add_attempt(rid, "running", number=1, reserved=10)  # started, lease live
    conn, tx = hold_request_lock(rid)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(start, owner, rid)
            wait_for_lock_waiter()
            assert not future.done()
            if settle_while_waiting:
                settle_running(rid)  # committed while the start waits
            tx.rollback()  # now the start gets the request lock
            result = future.result(timeout=10)
    finally:
        if tx.is_active:
            tx.rollback()
        conn.close()
    assert result.outcome is expected
    assert len(attempt_rows()) == 1


def test_racing_starts_on_a_settled_paid_request_admit_nobody(owner):
    rid = make_request(owner)
    add_paid(rid, "uncertain")
    before = attempt_rows()
    results = run_threads(
        8,
        lambda db, i: start_attempt(
            db,
            owner,
            rid,
            reserved_tokens=10,
            daily_limit_tokens=LIMIT,
            lease_seconds=LEASE,
        ),
    )
    assert {r.outcome for r in results} == {
        AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED
    }
    assert attempt_rows() == before


def test_two_waves_of_racing_starts_admit_exactly_one_attempt(owner):
    rid = make_request(owner)

    def wave():
        return run_threads(
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

    first = [r.outcome for r in wave()]
    assert first.count(AttemptStartOutcome.STARTED) == 1
    # The one admitted attempt reaches the provider and ends uncertain.
    execute(
        "UPDATE chat_request_attempts SET llm_started_at = clock_timestamp()"
        " WHERE request_id = :r",
        r=rid,
    )
    settle_running(rid)
    second = [r.outcome for r in wave()]
    assert second == [AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED] * 6
    assert len(attempt_rows()) == 1


def test_settlement_racing_a_start_never_lets_a_second_attempt_in(owner):
    outcomes = set()
    for _ in range(25):
        rid = make_request(owner)
        add_attempt(rid, "running", number=1, reserved=10)  # started, lease live
        results = run_threads(
            2,
            lambda db, i, rid=rid: (
                settle_running(rid)
                if i == 0
                else start_attempt(
                    db,
                    owner,
                    rid,
                    reserved_tokens=10,
                    daily_limit_tokens=LIMIT,
                    lease_seconds=LEASE,
                )
            ),
        )
        outcomes.add(results[1].outcome)
        assert (
            len(
                rows(
                    "SELECT id FROM chat_request_attempts WHERE request_id = :r", r=rid
                )
            )
            == 1
        )
    assert AttemptStartOutcome.STARTED not in outcomes
    assert outcomes <= {
        AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING,
        AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED,
    }


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
