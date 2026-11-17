"""Fenced attempt lifecycle (Component 12 / C8.4).

Transitions of one ``chat_request_attempts`` row after ``start_attempt`` created it:

    mark_llm_started   running, not started  -> running, llm_started_at stamped
    (success)          running, started      -> succeeded, ONLY inside publish_success
    fail_before_call   running, not started  -> failed_before_call   (charge 0)
    fail_with_usage    running, started      -> failed_with_usage    (charge = actual usage)
    mark_uncertain     running, started      -> uncertain            (charge = reserved)
    mark_rate_limited  running, started      -> rate_limited         (charge 0)
    (rejected)         running, started      -> rejected             (charge 0), ONLY inside
                                                 reject_and_fail_request
    (failed after generation) running, started -> failed_with_usage (charge = actual usage)
                                                 or uncertain (no usage, charge = reserved),
                                                 ONLY inside fail_request_after_generation
    expire_stale_attempt  running, lease expired -> abandoned (not started, charge 0)
                                                 or uncertain (started, charge = reserved)

The fence. The attempt row is its own fencing token: a takeover creates a NEW attempt and
settles the old one, so a stale worker finds its row no longer running. Every worker write
is ONE conditional statement::

    UPDATE ... WHERE id = :id AND status = 'running'
                 AND lease_expires_at > clock_timestamp() [AND phase condition]
    RETURNING id

Nothing is read first and then written unconditionally. Expiry is the same kind of
statement with the opposite lease predicate (``lease_expires_at <= clock_timestamp()``).
Both take the row lock; under READ COMMITTED the loser re-evaluates its WHERE after the
winner commits, finds the row no longer running, and changes nothing. Worker settlement
and expiry are therefore mutually exclusive, and a stale worker can never change an
attempt that was settled, expired or taken over. Times come from the database clock.

When no row matches, one read-only follow-up names the reason: NOT_FOUND, NOT_RUNNING
(already settled or expired), LEASE_EXPIRED (still running but past its lease, not yet
expired) or WRONG_PHASE (llm_started_at is not in the state the transition needs).

Phases. ``llm_started_at`` is committed by ``mark_llm_started`` BEFORE the provider call, in
its own transaction. NULL proves no call was issued, so only then may an attempt end as
failed_before_call or abandoned; every attempt that may have reached the provider ends as
succeeded, failed_with_usage, rate_limited, rejected or uncertain. ``mark_llm_started`` never moves an
existing timestamp (a second call is WRONG_PHASE), because that would move the UTC day.
There is no lease extension: the caller's lease must outlast the provider call.

Success is not a public transition here: ``settle_succeeded_in_transaction`` has no
transaction of its own and is used only by ``chat_request_finalization.publish_success``.
Rejection is deliberately not public either: ``settle_rejected_in_transaction`` is used only by
``chat_request_finalization.reject_and_fail_request``, which settles the attempt and fails its
request in ONE transaction, so no committed state has a rejected attempt with a request that is
still in progress. Failing after the provider generated an answer that cannot be published
(for example invalid citations) is private for the same reason:
``settle_failed_in_transaction`` is used only by
``chat_request_finalization.fail_request_after_generation``.

Not done here: any change to ``chat_requests``, messages or conversations (request
finalization is C8.5), taking over after expiry (the caller expires, then calls
``start_attempt``), and the error-code vocabulary. Error codes are supplied by the caller
and only their format is checked; ``lease_expired`` is the one fixed code, written by
``expire_stale_attempt``.

Like every service in app/rag, each public function is one transaction opened with
``with db.begin():``; the session must have no transaction in progress.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from app.models.chat_request_attempt import ChatRequestAttempt
from app.rag.chat_idempotency import ChatIdempotencyError

__all__ = [
    "LEASE_EXPIRED_CODE",
    "AttemptLifecycleArgumentError",
    "ExpireOutcome",
    "ExpireResult",
    "TransitionOutcome",
    "expire_stale_attempt",
    "fail_before_call",
    "fail_with_usage",
    "mark_llm_started",
    "mark_rate_limited",
    "mark_uncertain",
]

LEASE_EXPIRED_CODE = "lease_expired"
_ERROR_CODE_PATTERN = r"[a-z][a-z0-9_]{0,63}"
_INT32_MAX = 2_147_483_647  # token columns are 32-bit INTEGERs

attempt = ChatRequestAttempt


class AttemptLifecycleArgumentError(ChatIdempotencyError):
    """An argument has an invalid type, format or value."""


class TransitionOutcome(str, Enum):
    APPLIED = "applied"
    NOT_FOUND = "not_found"
    NOT_RUNNING = (
        "not_running"  # already settled or expired: the fence rejected the write
    )
    LEASE_EXPIRED = "lease_expired"  # running, but the lease has passed
    WRONG_PHASE = (
        "wrong_phase"  # llm_started_at is not in the state this transition needs
    )


class ExpireOutcome(str, Enum):
    ABANDONED = "abandoned"  # not started: proven no provider call, charge 0
    UNCERTAIN = "uncertain"  # started: reservation retained
    NOTHING_TO_EXPIRE = "nothing_to_expire"  # no running attempt with an expired lease


@dataclass(frozen=True, slots=True)
class ExpireResult:
    outcome: ExpireOutcome
    attempt_id: int | None = None


def _check_id(name: str, value: object) -> None:
    if type(value) is not int or value < 1 or value > _INT32_MAX:
        raise AttemptLifecycleArgumentError(f"{name} must be a positive integer")


def _check_error_code(value: object) -> None:
    if type(value) is not str or re.fullmatch(_ERROR_CODE_PATTERN, value) is None:
        raise AttemptLifecycleArgumentError("error_code has an invalid format")


def _check_usage(prompt_tokens: object, completion_tokens: object) -> None:
    for name, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
    ):
        if type(value) is not int or value < 0 or value > _INT32_MAX:
            raise AttemptLifecycleArgumentError(
                f"{name} must be a non-negative integer"
            )
    if prompt_tokens + completion_tokens > _INT32_MAX:  # type: ignore[operator]
        raise AttemptLifecycleArgumentError("total tokens are too large")


def _why_not_applied(db: Session, attempt_id: int) -> TransitionOutcome:
    """Read-only diagnosis after a fenced UPDATE matched no row."""
    row = db.execute(
        select(
            attempt.status,
            attempt.lease_expires_at <= func.clock_timestamp(),
        ).where(attempt.id == attempt_id)
    ).first()
    if row is None:
        return TransitionOutcome.NOT_FOUND
    status, lease_passed = row
    if status != "running":
        return TransitionOutcome.NOT_RUNNING
    if lease_passed:
        return TransitionOutcome.LEASE_EXPIRED
    return TransitionOutcome.WRONG_PHASE


def _fenced_update(
    db: Session, attempt_id: int, values: dict, *, started: bool
) -> TransitionOutcome:
    """Apply ``values`` iff the attempt is running, its lease is live and the phase matches.

    ``started`` is the required phase: True needs llm_started_at set, False needs it NULL.
    Must run inside the caller's transaction.
    """
    phase = (
        attempt.llm_started_at.is_not(None)
        if started
        else attempt.llm_started_at.is_(None)
    )
    applied = db.execute(
        update(attempt)
        .where(
            attempt.id == attempt_id,
            attempt.status == "running",
            attempt.lease_expires_at > func.clock_timestamp(),
            phase,
        )
        .values(**values)
        .returning(attempt.id)
    ).first()
    if applied is not None:
        return TransitionOutcome.APPLIED
    return _why_not_applied(db, attempt_id)


def _settled(status: str, **extra) -> dict:
    return {
        "status": status,
        "lease_expires_at": None,
        "finished_at": func.clock_timestamp(),
        **extra,
    }


def mark_llm_started(db: Session, attempt_id: int) -> TransitionOutcome:
    """Stamp llm_started_at; call this and commit BEFORE the provider call."""
    _check_id("attempt_id", attempt_id)
    with db.begin():
        return _fenced_update(
            db, attempt_id, {"llm_started_at": func.clock_timestamp()}, started=False
        )


def validate_optional_usage(
    prompt_tokens: int | None, completion_tokens: int | None
) -> None:
    """Usage is both-or-neither; when given, both are non-negative 32-bit integers."""
    if prompt_tokens is None and completion_tokens is None:
        return
    if prompt_tokens is None or completion_tokens is None:
        raise AttemptLifecycleArgumentError(
            "usage needs both prompt_tokens and completion_tokens"
        )
    _check_usage(prompt_tokens, completion_tokens)


def settle_succeeded_in_transaction(
    db: Session,
    attempt_id: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> TransitionOutcome:
    """Settle a started attempt as succeeded, inside the CALLER's transaction.

    Deliberately not public API (not in ``__all__``) and without a transaction of its own:
    a success must never be settled apart from publishing its messages, so only
    ``chat_request_finalization.publish_success`` calls this, in the same transaction as the
    message and request writes. With usage the charge is prompt + completion; with no usage
    at all (both None) the provider reported none and the charge is the reservation.
    """
    _check_id("attempt_id", attempt_id)
    validate_optional_usage(prompt_tokens, completion_tokens)
    if prompt_tokens is None:
        values = _settled("succeeded", charged_tokens=attempt.reserved_tokens)
    else:
        values = _settled(
            "succeeded",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            charged_tokens=prompt_tokens + completion_tokens,
        )
    return _fenced_update(db, attempt_id, values, started=True)


def settle_rejected_in_transaction(
    db: Session, attempt_id: int, error_code: str
) -> TransitionOutcome:
    """Settle a started attempt the provider rejected before generating (charge 0).

    Deliberately not public API (not in ``__all__``) and without a transaction of its own:
    a rejection must never be settled apart from failing its request, so only
    ``chat_request_finalization.reject_and_fail_request`` calls this, in the same
    transaction as the request update. Same fence as every worker write: the attempt must
    be running, its lease live and ``llm_started_at`` set (the call was about to be issued).
    """
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    return _fenced_update(
        db,
        attempt_id,
        _settled("rejected", charged_tokens=0, error_code=error_code),
        started=True,
    )


def settle_failed_in_transaction(
    db: Session,
    attempt_id: int,
    error_code: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> TransitionOutcome:
    """Settle a started attempt whose generated answer could not be published.

    Deliberately not public API (not in ``__all__``) and without a transaction of its own:
    it must never be settled apart from failing its request, so only
    ``chat_request_finalization.fail_request_after_generation`` calls this, in the same
    transaction as the request update. Same fence as every worker write (running, lease
    live, ``llm_started_at`` set). With usage the attempt is ``failed_with_usage`` and the
    charge is prompt + completion; with no usage at all (both None) the provider reported
    none, so it is ``uncertain`` and the charge is the reservation, exactly as a success
    without usage is charged. The answer was generated either way, so a charge of 0 is
    never an option here.
    """
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    validate_optional_usage(prompt_tokens, completion_tokens)
    if prompt_tokens is None:
        # ``attempt.reserved_tokens`` is a column expression, not a value read earlier: the
        # database copies the row's own reservation inside the fenced UPDATE below (the same
        # pattern as mark_uncertain and expire_stale_attempt), on the row that statement locks.
        values = _settled(
            "uncertain", charged_tokens=attempt.reserved_tokens, error_code=error_code
        )
    else:
        values = _settled(
            "failed_with_usage",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            charged_tokens=prompt_tokens + completion_tokens,
            error_code=error_code,
        )
    return _fenced_update(db, attempt_id, values, started=True)


def fail_before_call(
    db: Session, attempt_id: int, error_code: str
) -> TransitionOutcome:
    """Settle an attempt whose provider call was never issued (charge 0)."""
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    with db.begin():
        return _fenced_update(
            db,
            attempt_id,
            _settled("failed_before_call", charged_tokens=0, error_code=error_code),
            started=False,
        )


def fail_with_usage(
    db: Session,
    attempt_id: int,
    error_code: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> TransitionOutcome:
    """Settle a started attempt that failed after the provider reported usage."""
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    _check_usage(prompt_tokens, completion_tokens)
    with db.begin():
        return _fenced_update(
            db,
            attempt_id,
            _settled(
                "failed_with_usage",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                charged_tokens=prompt_tokens + completion_tokens,
                error_code=error_code,
            ),
            started=True,
        )


def mark_uncertain(db: Session, attempt_id: int, error_code: str) -> TransitionOutcome:
    """Settle a started attempt whose provider outcome is unknown: the reservation is kept."""
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    with db.begin():
        return _fenced_update(
            db,
            attempt_id,
            _settled(
                "uncertain",
                charged_tokens=attempt.reserved_tokens,
                error_code=error_code,
            ),
            started=True,
        )


def mark_rate_limited(
    db: Session, attempt_id: int, error_code: str
) -> TransitionOutcome:
    """Settle a started attempt the provider rejected with a rate limit (charge 0)."""
    _check_id("attempt_id", attempt_id)
    _check_error_code(error_code)
    with db.begin():
        return _fenced_update(
            db,
            attempt_id,
            _settled("rate_limited", charged_tokens=0, error_code=error_code),
            started=True,
        )


def expire_stale_attempt(db: Session, request_id: int) -> ExpireResult:
    """Settle the request's running attempt if (and only if) its lease has expired.

    One conditional UPDATE, the mirror image of the worker fence: it matches only a running
    attempt with ``lease_expires_at <= clock_timestamp()``, so it can never touch an attempt
    a worker still holds or one that is already settled. Not started: abandoned, charge 0.
    Started: uncertain, charge = reservation.
    """
    _check_id("request_id", request_id)
    not_started = attempt.llm_started_at.is_(None)
    with db.begin():
        row = db.execute(
            update(attempt)
            .where(
                attempt.request_id == request_id,
                attempt.status == "running",
                attempt.lease_expires_at <= func.clock_timestamp(),
            )
            .values(
                status=case((not_started, "abandoned"), else_="uncertain"),
                charged_tokens=case((not_started, 0), else_=attempt.reserved_tokens),
                error_code=LEASE_EXPIRED_CODE,
                lease_expires_at=None,
                finished_at=func.clock_timestamp(),
            )
            .returning(attempt.id, attempt.status)
        ).first()
    if row is None:
        return ExpireResult(ExpireOutcome.NOTHING_TO_EXPIRE)
    return ExpireResult(ExpireOutcome(row.status), row.id)
