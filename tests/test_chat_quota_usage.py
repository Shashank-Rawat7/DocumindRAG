"""C8.2 tests: get_daily_usage (read-only quota usage calculation).

Runs against documind_test. Rows are seeded with plain SQL that satisfies the C7.2 CHECKs.
"""

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, engine
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_quota_usage import (
    DailyUsage,
    QuotaUsageArgumentError,
    get_daily_usage,
    utc_day_bounds,
)

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
DAY = date(2026, 11, 17)
NOON = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
START = datetime(2026, 11, 17, 0, 0, tzinfo=UTC)
NEXT = START + timedelta(days=1)
LEASE = datetime(2030, 1, 1, tzinfo=UTC)
_counter = [0]


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def make_user(email):
    return execute(
        "INSERT INTO users (email, hashed_password) VALUES (:e, 'x') RETURNING id",
        e=email,
    ).scalar_one()


def add_attempt(
    owner,
    status,
    started=NOON,
    reserved=100,
    prompt=None,
    completion=None,
    charged=None,
):
    """Insert one attempt (with its own request) that satisfies every CHECK."""
    _counter[0] += 1
    request_id = execute(
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status,"
        " creates_conversation) VALUES (:o, :k, :h, 'in_progress', true) RETURNING id",
        o=owner,
        k=f"k{_counter[0]}",
        h="a" * 64,
    ).scalar_one()
    running = status == "running"
    execute(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status,"
        " lease_expires_at, reserved_tokens, prompt_tokens, completion_tokens,"
        " charged_tokens, error_code, llm_started_at, finished_at)"
        " VALUES (:r, 1, :s, :lease, :res, :p, :c, :ch, :err, :st, :fin)",
        r=request_id,
        s=status,
        lease=LEASE if running else None,
        res=reserved,
        p=prompt,
        c=completion,
        ch=charged,
        err=None if status in ("running", "succeeded") else "code",
        st=started,
        fin=None if running else NOON,
    )


def usage(owner, day=DAY):
    with SessionLocal() as db:
        return get_daily_usage(db, owner, day)


@pytest.fixture
def owner():
    return make_user("alice@example.com")


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


def seed_one_of_each(owner):
    add_attempt(owner, "running", reserved=7)  # running: counts reserved
    add_attempt(owner, "succeeded", reserved=50, prompt=10, completion=5, charged=15)
    add_attempt(owner, "succeeded", reserved=40, charged=40)  # no usage: reservation
    add_attempt(
        owner, "failed_with_usage", reserved=60, prompt=20, completion=1, charged=21
    )
    add_attempt(owner, "uncertain", reserved=30, charged=30)  # retains reservation
    add_attempt(owner, "rate_limited", reserved=25, charged=0)
    add_attempt(owner, "failed_before_call", started=None, reserved=99, charged=0)
    add_attempt(owner, "abandoned", started=None, reserved=98, charged=0)


# ---------------------------------------------------------------- counting rules


def test_empty_day_is_all_zero(owner):
    assert usage(owner) == DailyUsage(0, 0, 0, 0)


def test_every_status_contributes_what_the_rules_say(owner):
    seed_one_of_each(owner)
    # settled = 15 + 40 + 21 + 30 + 0 (rate_limited) + 0 + 0; running = 7
    assert usage(owner) == DailyUsage(
        settled_tokens=106,
        running_reserved_tokens=7,
        total_tokens=113,
        unstarted_running_attempts=0,
    )


def test_expired_unreaped_running_attempt_still_counts_its_reservation(owner):
    add_attempt(owner, "running", reserved=11)
    execute(
        "UPDATE chat_request_attempts SET lease_expires_at = :t", t=START
    )  # long expired
    assert usage(owner).running_reserved_tokens == 11


def test_running_attempt_without_llm_started_at_is_reported_not_counted(owner):
    add_attempt(owner, "running", started=None, reserved=500)
    add_attempt(owner, "running", started=None, reserved=600)
    add_attempt(owner, "succeeded", reserved=5, charged=5)
    result = usage(owner)
    assert result == DailyUsage(5, 0, 5, 2)
    # Not tied to any day: the diagnostic does not depend on usage_day.
    assert usage(owner, DAY + timedelta(days=30)).unstarted_running_attempts == 2


def test_unstarted_count_is_per_owner(owner, other_owner):
    add_attempt(owner, "running", started=None)
    add_attempt(other_owner, "running", started=None)
    add_attempt(other_owner, "running", started=None)
    assert usage(owner).unstarted_running_attempts == 1


# ---------------------------------------------------------------- the UTC day


def test_day_bounds_are_half_open_utc():
    start, end = utc_day_bounds(DAY)
    assert (start, end) == (START, NEXT)
    assert start.utcoffset() == timedelta(0) and end.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "started, counted",
    [
        (START - timedelta(microseconds=1), False),
        (START, True),
        (NEXT - timedelta(microseconds=1), True),
        (NEXT, False),
    ],
)
def test_day_boundaries(owner, started, counted):
    add_attempt(owner, "succeeded", started=started, reserved=9, charged=9)
    assert usage(owner).total_tokens == (9 if counted else 0)


def test_day_is_decided_by_utc_not_by_the_timestamp_offset(owner):
    ist = timezone(timedelta(hours=5, minutes=30))
    # 2026-11-17 03:00 IST == 2026-11-16 21:30 UTC: previous UTC day.
    add_attempt(
        owner,
        "succeeded",
        started=datetime(2026, 11, 17, 3, 0, tzinfo=ist),
        reserved=4,
        charged=4,
    )
    # 2026-11-18 04:00 IST == 2026-11-17 22:30 UTC: this UTC day.
    add_attempt(
        owner,
        "succeeded",
        started=datetime(2026, 11, 18, 4, 0, tzinfo=ist),
        reserved=6,
        charged=6,
    )
    assert usage(owner, DAY).total_tokens == 6
    assert usage(owner, DAY - timedelta(days=1)).total_tokens == 4


def test_session_time_zone_does_not_change_the_result(owner):
    add_attempt(owner, "succeeded", started=NOON, reserved=8, charged=8)
    with SessionLocal() as db:
        with db.begin():
            db.execute(text("SET LOCAL TIME ZONE 'Pacific/Kiritimati'"))
        db.execute(text("SET TIME ZONE 'Pacific/Kiritimati'"))
        db.commit()
        assert get_daily_usage(db, owner, DAY).total_tokens == 8


def test_other_days_are_excluded(owner):
    add_attempt(
        owner, "succeeded", started=NOON - timedelta(days=1), reserved=3, charged=3
    )
    add_attempt(owner, "running", started=NOON + timedelta(days=1), reserved=3)
    assert usage(owner).total_tokens == 0


# ---------------------------------------------------------------- scope


def test_other_owners_attempts_are_excluded(owner, other_owner):
    add_attempt(owner, "succeeded", reserved=5, charged=5)
    add_attempt(other_owner, "succeeded", reserved=70, charged=70)
    add_attempt(other_owner, "running", reserved=80)
    assert usage(owner).total_tokens == 5
    assert usage(other_owner).total_tokens == 150


def test_multiple_attempts_of_one_request_all_count(owner):
    add_attempt(owner, "uncertain", reserved=20, charged=20)
    request_id = execute("SELECT id FROM chat_requests").scalar_one()
    execute(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status,"
        " lease_expires_at, reserved_tokens, llm_started_at)"
        " VALUES (:r, 2, 'running', :l, 30, :s)",
        r=request_id,
        l=LEASE,
        s=NOON,
    )
    assert usage(owner) == DailyUsage(20, 30, 50, 0)


# ---------------------------------------------------------------- read-only and transactions


def table_state():
    with engine.begin() as conn:
        return conn.execute(
            text(
                "SELECT (SELECT count(*) FROM chat_requests),"
                " (SELECT count(*) FROM chat_request_attempts),"
                " (SELECT coalesce(sum(xmin::text::bigint), 0) FROM chat_request_attempts),"
                " (SELECT coalesce(sum(xmin::text::bigint), 0) FROM chat_requests)"
            )
        ).one()


def test_calculation_changes_nothing(owner):
    seed_one_of_each(owner)
    before = table_state()
    usage(owner)
    assert table_state() == before


def test_works_in_a_read_only_transaction(owner):
    seed_one_of_each(owner)
    # One explicit connection, so the setting cannot land on a different pooled connection.
    with engine.connect() as conn:
        try:
            conn.execute(text("SET default_transaction_read_only = on"))
            conn.commit()
            with Session(bind=conn) as db:
                assert get_daily_usage(db, owner, DAY).total_tokens == 113
        finally:  # the pooled connection must not keep the setting
            conn.execute(text("SET default_transaction_read_only = off"))
            conn.commit()


def test_leaves_no_open_transaction(owner):
    with SessionLocal() as db:
        get_daily_usage(db, owner, DAY)
        assert not db.in_transaction()


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [None, True, 1.0, "1"])
def test_invalid_owner_is_rejected(bad):
    with SessionLocal() as db, pytest.raises(QuotaUsageArgumentError):
        get_daily_usage(db, bad, DAY)


@pytest.mark.parametrize("bad", [None, "2026-11-17", NOON, 20261117])
def test_invalid_usage_day_is_rejected(owner, bad):
    with SessionLocal() as db, pytest.raises(QuotaUsageArgumentError):
        get_daily_usage(db, owner, bad)


def test_argument_error_shares_the_step1_base_class():
    assert issubclass(QuotaUsageArgumentError, ChatIdempotencyError)
