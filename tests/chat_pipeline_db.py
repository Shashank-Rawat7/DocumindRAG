"""Shared database helpers for the C9.5b pipeline tests (documind_test, plain SQL seeding).

Rows are seeded with SQL that satisfies every C7.2 CHECK; attempts that must come from the real
services are created with ``start_attempt`` / ``mark_llm_started`` by the tests themselves.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.core.database import SessionLocal, engine

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo
TABLES = ("conversations", "messages", "chat_requests", "chat_request_attempts")
_counter = [0]


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


def make_request(owner, status="in_progress", error_code=None):
    _counter[0] += 1
    return execute(
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status,"
        " creates_conversation, completed_at, error_code)"
        " VALUES (:o, :k, :h, :s, true, :c, :e) RETURNING id",
        o=owner,
        k=f"k{_counter[0]}",
        h="a" * 64,
        s=status,
        c=None if status == "in_progress" else datetime.now(UTC),
        e=(error_code or "boom") if status == "failed" else None,
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
    error_code="code",
):
    """Insert one attempt that satisfies every CHECK. started: 'now', None or a datetime."""
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
        err=None if status in ("running", "succeeded") else error_code,
        st=started,
        fin=None if running else datetime.now(UTC),
        pt=None if usage is None else usage[0],
        ct=None if usage is None else usage[1],
    )


def add_terminal(request_id, status, number=1, error_code="code"):
    """Seed an attempt that reached the provider and was not a 429."""
    if status == "failed_with_usage":
        add_attempt(
            request_id,
            status,
            number=number,
            reserved=10,
            charged=10,
            usage=(7, 3),
            error_code=error_code,
        )
    elif status == "rejected":
        add_attempt(
            request_id,
            status,
            number=number,
            reserved=10,
            charged=0,
            error_code=error_code,
        )
    else:  # succeeded / uncertain without usage are charged the reservation
        add_attempt(
            request_id,
            status,
            number=number,
            reserved=10,
            charged=10,
            error_code=error_code,
        )


def expire_lease(attempt_id):
    execute(
        "UPDATE chat_request_attempts SET lease_expires_at = clock_timestamp()"
        " - interval '1 second' WHERE id = :i",
        i=attempt_id,
    )


def snapshot():
    """Every row of the four chat tables, as a comparable value."""
    with engine.begin() as conn:
        return {
            table: [
                tuple(r)
                for r in conn.execute(text(f"SELECT * FROM {table} ORDER BY id"))
            ]
            for table in TABLES
        }


def request_row(request_id):
    return execute("SELECT * FROM chat_requests WHERE id = :i", i=request_id).one()


def attempt_history(request_id):
    return rows(
        "SELECT id, attempt_number, status, error_code, charged_tokens, reserved_tokens,"
        " llm_started_at FROM chat_request_attempts WHERE request_id = :r"
        " ORDER BY attempt_number",
        r=request_id,
    )


def run_threads(n, work):
    barrier = threading.Barrier(n)

    def task(i):
        with SessionLocal() as db:
            barrier.wait(timeout=10)
            return work(db, i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))
