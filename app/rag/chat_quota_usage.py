"""Read-only daily token usage for one owner (Component 12 / C8.2).

``get_daily_usage`` adds up the quota ledger (``chat_request_attempts``) for one owner and
one UTC day. It only reads: it creates no reservation, enforces no limit, reaps no
attempt and changes no row. It does not know the daily limit or the reservation formula.

Rules (approved for C8.2):
    * Scope is the owner: attempts are joined to ``chat_requests`` on ``owner_id``.
    * The accounting day is the half-open UTC range [day 00:00, next day 00:00) applied to
      ``llm_started_at``. The bounds are UTC-aware datetimes built here, so the result does
      not depend on the session time zone.
    * A settled attempt (any status except ``running``) counts its ``charged_tokens``. The
      schema fixes that value per status: actual usage, the reservation for an uncertain
      attempt, or 0 for a proven non-execution or a rate limit.
    * A ``running`` attempt counts its ``reserved_tokens``. An expired but unreaped running
      attempt still counts: nothing here looks at the lease.
    * A running attempt whose ``llm_started_at`` is NULL belongs to no day. It is excluded
      from the totals and only counted in ``unstarted_running_attempts``. How such an
      attempt should be accounted is an open decision (D3) and is not made here.

The result is advisory. It is one statement under READ COMMITTED and takes no locks, so
it is not a safe check-then-reserve by itself: the reservation step must serialize
separately.

Like every service in app/rag, the function is one transaction opened with
``with db.begin():``; the session must have no transaction in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from app.models.chat_request import ChatRequest
from app.models.chat_request_attempt import ChatRequestAttempt
from app.rag.chat_idempotency import ChatIdempotencyError

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatible, like the rest of the repo

__all__ = [
    "DailyUsage",
    "QuotaUsageArgumentError",
    "get_daily_usage",
    "utc_day_bounds",
]


class QuotaUsageArgumentError(ChatIdempotencyError):
    """owner_id or usage_day has an invalid type."""


@dataclass(frozen=True, slots=True)
class DailyUsage:
    settled_tokens: int  # charged_tokens of non-running attempts started in the day
    running_reserved_tokens: (
        int  # reserved_tokens of running attempts started in the day
    )
    total_tokens: int  # settled_tokens + running_reserved_tokens
    unstarted_running_attempts: int  # diagnostic only; not part of total_tokens


def utc_day_bounds(usage_day: date) -> tuple[datetime, datetime]:
    """Half-open UTC range [start, end) of the given UTC day."""
    if type(usage_day) is not date:
        raise QuotaUsageArgumentError("usage_day must be a date")
    start = datetime.combine(usage_day, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def get_daily_usage(db: Session, owner_id: int, usage_day: date) -> DailyUsage:
    if type(owner_id) is not int:
        raise QuotaUsageArgumentError("owner_id must be an integer")
    start, end = utc_day_bounds(usage_day)

    attempt = ChatRequestAttempt
    started_in_day = (attempt.llm_started_at >= start) & (attempt.llm_started_at < end)
    is_running = attempt.status == "running"

    statement = (
        select(
            func.coalesce(
                func.sum(attempt.charged_tokens).filter(~is_running & started_in_day), 0
            ),
            func.coalesce(
                func.sum(attempt.reserved_tokens).filter(is_running & started_in_day), 0
            ),
            func.coalesce(
                func.sum(
                    case((is_running & attempt.llm_started_at.is_(None), 1), else_=0)
                ),
                0,
            ).cast(Integer),
        )
        .select_from(attempt)
        .join(ChatRequest, ChatRequest.id == attempt.request_id)
        .where(ChatRequest.owner_id == owner_id)
    )
    with db.begin():
        settled, running, unstarted = db.execute(statement).one()

    return DailyUsage(
        settled_tokens=int(settled),
        running_reserved_tokens=int(running),
        total_tokens=int(settled) + int(running),
        unstarted_running_attempts=int(unstarted),
    )
