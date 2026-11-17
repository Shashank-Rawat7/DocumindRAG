"""Atomic quota reservation and attempt start (Component 12 / C8.3).

``start_attempt`` reserves ``reserved_tokens`` against the owner's daily quota and, in the
same transaction, creates the next ``running`` attempt for an ``in_progress`` request. It
does not know a daily limit, a reservation formula or a lease duration: the caller passes
``daily_limit_tokens``, ``reserved_tokens`` and ``lease_seconds`` and none has a default.

Atomicity. The transaction first locks the owner's ``users`` row (``SELECT ... FOR UPDATE``),
so reservations of one owner are checked and inserted one after another; another owner is
never blocked. Under READ COMMITTED the usage query that follows sees every reservation
committed before the lock was granted. The request row is locked next, which also keeps two
starts for one request apart.

Quota check (approved decisions):
    used = settled charges of attempts started in the current UTC day
         + reserved_tokens of running attempts started in the current UTC day
         + reserved_tokens of running attempts that have NOT started (llm_started_at NULL)
           and whose lease is still live
    allowed  <=>  used + reserved_tokens <= daily_limit_tokens
The current UTC day comes from the database clock. An unstarted running attempt with an
expired lease is not counted: llm_started_at is committed before the provider call, so it
provably issued none. Once llm_started_at is set, ordinary UTC-day accounting applies and
there is no re-check at that point. This differs on purpose from ``get_daily_usage``
(C8.2), which stays the historical definition and excludes unstarted attempts; its code is
not touched or called here (it would open a second transaction).

Outcomes (nothing is written for any outcome except STARTED):
    STARTED                the attempt row was inserted (status running, llm_started_at NULL)
    QUOTA_EXCEEDED         the reservation does not fit; the request stays in_progress
    REQUEST_NOT_FOUND      no such request for this owner (missing or someone else's)
    REQUEST_NOT_IN_PROGRESS  the request already succeeded or failed
    ATTEMPT_ALREADY_RUNNING  a running attempt exists (live or expired): no takeover here
    RETRY_LIMIT_REACHED    the request already has two rate_limited attempts (one 429 retry)

Not done here: stamping llm_started_at, finishing or abandoning an attempt, reaping and any
request state change (see chat_attempt_lifecycle). The 429 retry cap counts only
rate_limited attempts: the first 429 allows one retry, a second ends retries.
The lease is returned to the database as ``clock_timestamp() + lease_seconds``.

Like every service in app/rag, the function is one transaction opened with
``with db.begin():``; the session must have no transaction in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from sqlalchemy import case, func, insert, select, text
from sqlalchemy.orm import Session

from app.models.chat_request import ChatRequest
from app.models.chat_request_attempt import ChatRequestAttempt
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_quota_usage import UTC, utc_day_bounds

__all__ = [
    "AttemptStartArgumentError",
    "AttemptStartOutcome",
    "AttemptStartResult",
    "start_attempt",
]

_INT32_MAX = 2_147_483_647  # reserved_tokens is a 32-bit INTEGER column
_MAX_RATE_LIMITED_ATTEMPTS = (
    2  # one 429 retry: a second rate_limited attempt ends retries
)


class AttemptStartArgumentError(ChatIdempotencyError):
    """An argument has an invalid type or value."""


class AttemptStartOutcome(str, Enum):
    STARTED = "started"
    QUOTA_EXCEEDED = "quota_exceeded"
    REQUEST_NOT_FOUND = "request_not_found"
    REQUEST_NOT_IN_PROGRESS = "request_not_in_progress"
    ATTEMPT_ALREADY_RUNNING = "attempt_already_running"
    RETRY_LIMIT_REACHED = "retry_limit_reached"


@dataclass(frozen=True, slots=True)
class AttemptStartResult:
    outcome: AttemptStartOutcome
    attempt_id: int | None = None  # STARTED only
    attempt_number: int | None = None  # STARTED only
    used_tokens: int | None = (
        None  # STARTED and QUOTA_EXCEEDED: usage before this reservation
    )


def _check_int(
    name: str, value: object, minimum: int, maximum: int | None = None
) -> None:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise AttemptStartArgumentError(
            f"{name} must be an integer of at least {minimum}"
        )


def start_attempt(
    db: Session,
    owner_id: int,
    request_id: int,
    *,
    reserved_tokens: int,
    daily_limit_tokens: int,
    lease_seconds: int,
) -> AttemptStartResult:
    _check_int("owner_id", owner_id, 1)
    _check_int("request_id", request_id, 1)
    _check_int("reserved_tokens", reserved_tokens, 1, _INT32_MAX)
    _check_int("daily_limit_tokens", daily_limit_tokens, 1)
    _check_int("lease_seconds", lease_seconds, 1)

    attempt = ChatRequestAttempt
    with db.begin():
        # 1. Serialize this owner's reservations. No such user means no such request.
        # (Plain SQL: the User ORM class would drag every model it relates to into mapper
        # configuration.)
        locked_owner = db.execute(
            text("SELECT id FROM users WHERE id = :owner_id FOR UPDATE"),
            {"owner_id": owner_id},
        ).scalar_one_or_none()
        if locked_owner is None:
            return AttemptStartResult(AttemptStartOutcome.REQUEST_NOT_FOUND)

        # 2. The request must be this owner's and still in progress.
        request_status = db.execute(
            select(ChatRequest.status)
            .where(ChatRequest.id == request_id, ChatRequest.owner_id == owner_id)
            .with_for_update()
        ).scalar_one_or_none()
        if request_status is None:
            return AttemptStartResult(AttemptStartOutcome.REQUEST_NOT_FOUND)
        if request_status != "in_progress":
            return AttemptStartResult(AttemptStartOutcome.REQUEST_NOT_IN_PROGRESS)

        # 3. One running attempt per request; an expired one is not taken over here.
        has_running = db.execute(
            select(attempt.id)
            .where(attempt.request_id == request_id, attempt.status == "running")
            .limit(1)
        ).first()
        if has_running is not None:
            return AttemptStartResult(AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING)

        # 3b. At most one retry after a 429 (counted under the request lock held above).
        rate_limited = db.execute(
            select(func.count())
            .select_from(attempt)
            .where(attempt.request_id == request_id, attempt.status == "rate_limited")
        ).scalar_one()
        if rate_limited >= _MAX_RATE_LIMITED_ATTEMPTS:
            return AttemptStartResult(AttemptStartOutcome.RETRY_LIMIT_REACHED)

        # 4. Usage for the current UTC day (database clock), including live unstarted holds.
        now = db.execute(select(func.clock_timestamp())).scalar_one()
        day_start, day_end = utc_day_bounds(now.astimezone(UTC).date())
        in_day = (attempt.llm_started_at >= day_start) & (
            attempt.llm_started_at < day_end
        )
        is_running = attempt.status == "running"
        live_unstarted = (
            is_running
            & attempt.llm_started_at.is_(None)
            & (attempt.lease_expires_at > func.clock_timestamp())
        )
        used = int(
            db.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (~is_running & in_day, attempt.charged_tokens),
                                (is_running & in_day, attempt.reserved_tokens),
                                (live_unstarted, attempt.reserved_tokens),
                                else_=0,
                            )
                        ),
                        0,
                    )
                )
                .select_from(attempt)
                .join(ChatRequest, ChatRequest.id == attempt.request_id)
                .where(ChatRequest.owner_id == owner_id)
            ).scalar_one()
        )
        if used + reserved_tokens > daily_limit_tokens:
            return AttemptStartResult(
                AttemptStartOutcome.QUOTA_EXCEEDED, used_tokens=used
            )

        # 5. Reserve: insert the next running attempt (llm_started_at stays NULL).
        attempt_number = db.execute(
            select(func.coalesce(func.max(attempt.attempt_number), 0) + 1).where(
                attempt.request_id == request_id
            )
        ).scalar_one()
        attempt_id = db.execute(
            insert(attempt)
            .values(
                request_id=request_id,
                attempt_number=attempt_number,
                status="running",
                lease_expires_at=func.clock_timestamp()
                + timedelta(seconds=lease_seconds),
                reserved_tokens=reserved_tokens,
            )
            .returning(attempt.id)
        ).scalar_one()
        return AttemptStartResult(
            AttemptStartOutcome.STARTED,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            used_tokens=used,
        )
