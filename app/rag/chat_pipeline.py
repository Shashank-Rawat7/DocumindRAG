"""Chat answer pipeline (Component 12 / C9.5b).

Part 1 is the recovery protocol R1-R4 that every pipeline entry for an EXISTING request runs
first (plain functions over the C8 services). Part 2 is the LangGraph orchestration
``run_chat_pipeline`` that calls it and then runs one generation (see its docstring).

Recovery protocol (approved C9.5 plan, revised D3)
--------------------------------------------------
Replays carry the same request body, so the pipeline is always handed the question; the request
row stores only its hash. Every entry runs:

R1  Load the request, scoped to its owner.
      not found / another owner's -> NOT_FOUND
      succeeded                   -> REPLAY_SUCCEEDED (the stored result)
      failed                      -> REPLAY_FAILED (the stored error code)
R2  ``expire_stale_attempt``: one fenced UPDATE, idempotent. After it returns, an attempt whose
    lease had expired is settled by someone (this worker or a concurrent one).
R3  Read the attempt history in one read-only statement.
      a running attempt exists (its lease is live) -> IN_PROGRESS_ELSEWHERE, nothing written
      an attempt that reached the provider and is not a 429 (a "terminal-started" attempt:
      succeeded, failed_with_usage, uncertain, rejected) exists -> R4
      otherwise -> PROCEED (no attempt, or only unstarted / rate_limited ones: states S1, S3, S4)
R4  Finalize: ``fail_request`` with the terminal attempt's own error code (``lease_expired``
    after an expiry), ``internal_error`` if it has none. This path never retrieves, never calls
    ``start_attempt`` and never calls the provider, so recovery can never start a second paid
    generation; C9.5a enforces the same rule again inside ``start_attempt``.
      FAILED (this call)            -> FINALIZED
      already finalized by another  -> the stored state (REPLAY_FAILED / REPLAY_SUCCEEDED)
      an attempt is running again   -> IN_PROGRESS_ELSEWHERE

State S5 (the request is in_progress and the terminal attempt is ``uncertain`` with
``lease_expired``) is the one intermediate state this protocol creates: it exists between R2
committing and R4 committing. A crash there is harmless, because every later entry runs R1-R4
again and finalizes. A ``succeeded`` attempt on an in_progress request cannot be produced by the
C8 services (publishing settles the attempt and the request in one transaction); the only
legitimate way to see it here is a publish that committed between R1 and R3, which is re-read
and reported as a replay. Anything else is reported as INCONSISTENT and nothing is written.

Logging: ids and outcome names only, never the question, an answer, a chunk or an exception text.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.chat_config import LEASE_SECONDS, TOP_K, ChatSettings
from app.models.chat_request_attempt import ChatRequestAttempt
from app.rag import chat_error_codes as codes
from app.rag.chat_attempt_lifecycle import (
    TransitionOutcome,
    expire_stale_attempt,
    fail_before_call,
    mark_llm_started,
    mark_rate_limited,
)
from app.rag.chat_attempt_start import (
    _NON_RETRYABLE_STARTED_STATUSES as TERMINAL_STARTED_STATUSES,
)
from app.rag.chat_attempt_start import AttemptStartOutcome, start_attempt
from app.rag.chat_citations import resolve_citations
from app.rag.chat_idempotency import ChatIdempotencyError
from app.rag.chat_prompt import (
    BuiltPrompt,
    NoUsableContextError,
    PromptBuildError,
    build_prompt,
)
from app.rag.chat_provider import (
    ChatProvider,
    ProviderResult,
    SettlementAction,
    settlement_action,
)
from app.rag.chat_request_finalization import (
    FailAfterGenerationOutcome,
    FailOutcome,
    PublishOutcome,
    PublishResult,
    RejectOutcome,
    RequestResult,
    fail_request,
    fail_request_after_generation,
    get_request_result,
    publish_success,
    reject_and_fail_request,
)
from app.rag.embedder import Embedder, EmbeddingInputTooLongError, get_embedder
from app.rag.retrieval import RetrievalRequestError, retrieve_chunks

__all__ = [
    "AttemptSummary",
    "FailureKind",
    "PipelineArgumentError",
    "PipelineOutcome",
    "PipelineResult",
    "RecoveryOutcome",
    "RecoveryResult",
    "TerminalAttempt",
    "finalize_after_terminal_attempt",
    "read_attempt_summary",
    "recover_request",
    "run_chat_pipeline",
]

logger = logging.getLogger("documind")

_INT32_MAX: Final = 2_147_483_647


class PipelineArgumentError(ChatIdempotencyError):
    """An argument has an invalid type or value."""


class RecoveryOutcome(str, Enum):
    NOT_FOUND = "not_found"
    REPLAY_SUCCEEDED = "replay_succeeded"  # request.* holds the stored result
    REPLAY_FAILED = "replay_failed"  # error_code holds the stored code
    IN_PROGRESS_ELSEWHERE = (
        "in_progress_elsewhere"  # a live attempt exists; nothing written
    )
    FINALIZED = "finalized"  # this call failed the request (error_code)
    PROCEED = "proceed"  # no terminal-started attempt, nothing running: normal flow
    INCONSISTENT = (
        "inconsistent"  # a state the C8 services cannot produce; nothing written
    )


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    outcome: RecoveryOutcome
    request: RequestResult | None = None  # replay outcomes: the stored state
    error_code: str | None = None  # REPLAY_FAILED and FINALIZED


@dataclass(frozen=True, slots=True)
class TerminalAttempt:
    attempt_id: int
    status: str  # one of TERMINAL_STARTED_STATUSES
    error_code: str | None


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    running: bool  # any attempt of the request is running
    terminal: (
        TerminalAttempt | None
    )  # the lowest-numbered terminal-started attempt, if any


def _check_id(name: str, value: object) -> None:
    if type(value) is not int or value < 1 or value > _INT32_MAX:
        raise PipelineArgumentError(f"{name} must be a positive integer")


def read_attempt_summary(db: Session, request_id: int) -> AttemptSummary:
    """One read-only statement over the request's whole attempt history (own transaction)."""
    _check_id("request_id", request_id)
    attempt = ChatRequestAttempt
    with db.begin():
        history = db.execute(
            select(attempt.id, attempt.status, attempt.error_code)
            .where(attempt.request_id == request_id)
            .order_by(attempt.attempt_number)
        ).all()
    running = any(row.status == "running" for row in history)
    terminal = next(
        (
            TerminalAttempt(row.id, row.status, row.error_code)
            for row in history
            if row.status in TERMINAL_STARTED_STATUSES
        ),
        None,
    )
    return AttemptSummary(running=running, terminal=terminal)


def _replay(stored: RequestResult) -> RecoveryResult:
    if stored.status == "succeeded":
        return RecoveryResult(RecoveryOutcome.REPLAY_SUCCEEDED, request=stored)
    if stored.status == "failed":
        return RecoveryResult(
            RecoveryOutcome.REPLAY_FAILED, request=stored, error_code=stored.error_code
        )
    return RecoveryResult(RecoveryOutcome.INCONSISTENT, request=stored)


def _finalization_code(terminal: TerminalAttempt) -> str:
    code = terminal.error_code
    if code is not None and codes.ERROR_CODE_PATTERN.fullmatch(code) is not None:
        return code
    return codes.INTERNAL_ERROR


def finalize_after_terminal_attempt(
    db: Session, owner_id: int, request_id: int
) -> RecoveryResult:
    """R3 + R4: finalize a request whose attempt already reached the provider.

    PRECONDITION. This function does NOT perform R2 (``expire_stale_attempt``). It may be
    called only (a) right after ``recover_request`` has run R2, or (b) after ``start_attempt``
    has answered GENERATION_ALREADY_ATTEMPTED (a worker raced past an earlier look at the
    history). It is not a substitute for ``recover_request``: an expired-but-still-running
    attempt must go through R2 first, otherwise R3 sees it as running and reports
    IN_PROGRESS_ELSEWHERE for a lease nobody will ever settle.

    Writes at most one thing: ``fail_request``.
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    summary = read_attempt_summary(db, request_id)
    if summary.running:
        return RecoveryResult(RecoveryOutcome.IN_PROGRESS_ELSEWHERE)
    terminal = summary.terminal
    if terminal is None:
        return RecoveryResult(RecoveryOutcome.PROCEED)
    if terminal.status == "succeeded":
        # Only a publish that committed after R1 can explain this: re-read and report it.
        stored = get_request_result(db, owner_id, request_id)
        if stored is None:
            return RecoveryResult(RecoveryOutcome.NOT_FOUND)
        result = _replay(stored)
        if result.outcome is RecoveryOutcome.INCONSISTENT:
            logger.error("chat recovery request_id=%s inconsistent", request_id)
        return result

    code = _finalization_code(terminal)
    failed = fail_request(db, owner_id, request_id, code)
    if failed.outcome is FailOutcome.FAILED:
        logger.info("chat recovery request_id=%s finalized code=%s", request_id, code)
        return RecoveryResult(RecoveryOutcome.FINALIZED, error_code=code)
    if failed.outcome is FailOutcome.ATTEMPT_RUNNING:
        return RecoveryResult(RecoveryOutcome.IN_PROGRESS_ELSEWHERE)
    if failed.outcome is FailOutcome.NOT_FOUND:
        return RecoveryResult(RecoveryOutcome.NOT_FOUND)
    # ALREADY_FINALIZED: another worker finalized it first; report what is stored.
    stored = get_request_result(db, owner_id, request_id)
    if stored is None:
        return RecoveryResult(RecoveryOutcome.NOT_FOUND)
    return _replay(stored)


def recover_request(db: Session, owner_id: int, request_id: int) -> RecoveryResult:
    """R1-R4 for one existing request; see the module docstring."""
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    stored = get_request_result(db, owner_id, request_id)  # R1
    if stored is None:
        return RecoveryResult(RecoveryOutcome.NOT_FOUND)
    if stored.status != "in_progress":
        return _replay(stored)
    expire_stale_attempt(db, request_id)  # R2
    return finalize_after_terminal_attempt(db, owner_id, request_id)  # R3, R4


# ---------------------------------------------------------------------------------------------
# Part 2: orchestration
# ---------------------------------------------------------------------------------------------


class PipelineOutcome(str, Enum):
    ANSWERED = "answered"  # this call published the answer (``published``)
    REPLAYED_SUCCESS = "replayed_success"  # already succeeded (``request``)
    REPLAYED_FAILURE = "replayed_failure"  # already failed (``error_code``)
    FAILED = "failed"  # this call failed the request (``error_code``, ``failure_kind``)
    IN_PROGRESS_ELSEWHERE = (
        "in_progress_elsewhere"  # a live attempt is running; nothing written
    )
    STOPPED = "stopped"  # this worker lost its fence; nothing it did is persisted
    NOT_FOUND = "not_found"
    INCONSISTENT = (
        "inconsistent"  # a state the C8 services cannot produce; nothing written
    )


class FailureKind(str, Enum):
    """Why a pre-provider failure happened (both persist ``internal_error``)."""

    INVALID_INPUT = (
        "invalid_input"  # expected validation failure (bad question, bad argument)
    )
    UNEXPECTED = "unexpected"  # embedder / retrieval-integrity / anything else


@dataclass(frozen=True, slots=True)
class PipelineResult:
    outcome: PipelineOutcome
    error_code: str | None = None  # FAILED and REPLAYED_FAILURE
    failure_kind: FailureKind | None = None  # FAILED, only for D8 failures
    request: RequestResult | None = None  # replays
    published: PublishResult | None = None  # ANSWERED


class _State(TypedDict, total=False):
    route: str
    result: PipelineResult
    prompt: BuiltPrompt
    attempt_id: int
    provider_result: ProviderResult
    retried: bool


_Node = Callable[[_State], dict[str, Any]]


def _is_storable(text: object) -> bool:
    """True when ``text`` is safe to persist AND has visible content.

    Deliberately a little stricter than ``_check_content`` / ck_messages_content_not_blank,
    which treat only spaces as blank: here ANY whitespace-only string (tabs, newlines, NBSP...)
    is refused, so such an answer fails as ``invalid_response`` instead of being published as an
    empty-looking message. Everything ``_is_storable`` accepts, ``publish_success`` accepts, so
    ``FinalizationArgumentError`` cannot escape from content.
    """
    if type(text) is not str or text.strip() == "" or "\x00" in text:
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _recovery_to_result(recovered: RecoveryResult) -> PipelineResult | None:
    """Map a terminal recovery outcome; None means PROCEED."""
    outcome = recovered.outcome
    if outcome is RecoveryOutcome.PROCEED:
        return None
    if outcome is RecoveryOutcome.NOT_FOUND:
        return PipelineResult(PipelineOutcome.NOT_FOUND)
    if outcome is RecoveryOutcome.REPLAY_SUCCEEDED:
        return PipelineResult(
            PipelineOutcome.REPLAYED_SUCCESS, request=recovered.request
        )
    if outcome is RecoveryOutcome.REPLAY_FAILED:
        return PipelineResult(
            PipelineOutcome.REPLAYED_FAILURE,
            error_code=recovered.error_code,
            request=recovered.request,
        )
    if outcome is RecoveryOutcome.FINALIZED:
        return PipelineResult(PipelineOutcome.FAILED, error_code=recovered.error_code)
    if outcome is RecoveryOutcome.IN_PROGRESS_ELSEWHERE:
        return PipelineResult(PipelineOutcome.IN_PROGRESS_ELSEWHERE)
    return PipelineResult(PipelineOutcome.INCONSISTENT)


def _stored_replay(db: Session, owner_id: int, request_id: int) -> PipelineResult:
    """Another worker finalized the request first: report what is stored."""
    stored = get_request_result(db, owner_id, request_id)
    if stored is None:
        return PipelineResult(PipelineOutcome.NOT_FOUND)
    mapped = _recovery_to_result(_replay(stored))
    return (
        mapped if mapped is not None else PipelineResult(PipelineOutcome.INCONSISTENT)
    )


def _fail_without_attempt(
    db: Session,
    owner_id: int,
    request_id: int,
    code: str,
    kind: FailureKind | None = None,
) -> PipelineResult:
    """``fail_request``: no attempt of this worker is running, so nothing is charged."""
    failed = fail_request(db, owner_id, request_id, code)
    if failed.outcome is FailOutcome.FAILED:
        logger.info("chat pipeline request_id=%s failed code=%s", request_id, code)
        return PipelineResult(
            PipelineOutcome.FAILED, error_code=code, failure_kind=kind
        )
    if failed.outcome is FailOutcome.ATTEMPT_RUNNING:
        return PipelineResult(PipelineOutcome.IN_PROGRESS_ELSEWHERE)
    if failed.outcome is FailOutcome.NOT_FOUND:
        return PipelineResult(PipelineOutcome.NOT_FOUND)
    return _stored_replay(db, owner_id, request_id)


def _fail_after_generation(
    db: Session,
    owner_id: int,
    request_id: int,
    attempt_id: int,
    code: str,
    result: ProviderResult | None = None,
) -> PipelineResult:
    """Settle this worker's started attempt (usage if reported, else reservation) and fail."""
    usage = result.usage if result is not None else None
    settled = fail_request_after_generation(
        db,
        owner_id,
        request_id,
        attempt_id,
        code,
        usage.prompt_tokens if usage is not None else None,
        usage.completion_tokens if usage is not None else None,
    )
    if settled.outcome is FailAfterGenerationOutcome.FAILED:
        logger.info("chat pipeline request_id=%s failed code=%s", request_id, code)
        return PipelineResult(PipelineOutcome.FAILED, error_code=code)
    if settled.outcome is FailAfterGenerationOutcome.ALREADY_FINALIZED:
        return _stored_replay(db, owner_id, request_id)
    # REQUEST_NOT_FOUND, NOT_RUNNING, LEASE_EXPIRED, WRONG_PHASE: this worker lost its fence.
    logger.warning(
        "chat pipeline request_id=%s stopped outcome=%s",
        request_id,
        settled.outcome.value,
    )
    return PipelineResult(PipelineOutcome.STOPPED)


def _rollback_if_open(db: Session) -> None:
    if db.in_transaction():
        db.rollback()


def run_chat_pipeline(
    db: Session,
    *,
    owner_id: int,
    request_id: int,
    question: str,
    provider: ChatProvider,
    settings: ChatSettings,
    embedder: Embedder | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    """Answer one registered request. Every call runs recovery first, so replays are safe.

    Graph: recover -> prepare -> start -> mark_started -> generate -> settle (-> start, once,
    after a 429 with a usable retry-after). Each provider outcome reaches exactly one
    settlement path (``settlement_action``):

        PUBLISH          publish_success (zero citations are published; D9)
        FAIL_WITH_USAGE  fail_request_after_generation (UNUSABLE; usage if reported)
        MARK_UNCERTAIN   fail_request_after_generation (reservation stays charged)
        MARK_REJECTED    reject_and_fail_request (one transaction, charge 0)
        MARK_RATE_LIMITED mark_rate_limited, then one retry or fail_request

    A raised exception from ``provider.generate`` is settled as UNCERTAIN / ``internal_error``.
    Retrieval and generation never run inside a database transaction: the question is embedded
    first (D11), retrieval is one short read-only transaction, and the session must arrive with
    no transaction open. Nothing is logged but ids, outcome names and exception class names.
    """
    _check_id("owner_id", owner_id)
    _check_id("request_id", request_id)
    if db.in_transaction():
        raise PipelineArgumentError("the session must have no transaction in progress")

    def done(result: PipelineResult) -> dict[str, Any]:
        return {"route": "end", "result": result}

    def recover(state: _State) -> dict[str, Any]:
        mapped = _recovery_to_result(recover_request(db, owner_id, request_id))
        return {"route": "prepare"} if mapped is None else done(mapped)

    def prepare(state: _State) -> dict[str, Any]:
        if not _is_storable(question):
            return done(
                _fail_without_attempt(
                    db,
                    owner_id,
                    request_id,
                    codes.INTERNAL_ERROR,
                    FailureKind.INVALID_INPUT,
                )
            )
        try:
            engine = embedder if embedder is not None else get_embedder()
            vector = engine.embed_query(question)  # no transaction is open here (D11)
            with db.begin():
                chunks = retrieve_chunks(
                    db,
                    owner_id=owner_id,
                    query_vector=vector,
                    embedding_id=engine.embedding_id,
                    top_k=TOP_K,
                )
            prompt = build_prompt(question, chunks)
        except NoUsableContextError:
            return done(
                _fail_without_attempt(db, owner_id, request_id, codes.NO_CONTEXT)
            )
        # EmbeddingInputTooLongError is the one embedder error that is the caller's doing (an over-long
        # question, rejected and never truncated). The base EmbeddingInputError means a non-str was
        # passed, which this module never does, so it is a bug and falls through to UNEXPECTED.
        except (
            PromptBuildError,
            RetrievalRequestError,
            EmbeddingInputTooLongError,
        ) as exc:
            logger.warning(
                "chat pipeline request_id=%s invalid input %s",
                request_id,
                type(exc).__name__,
            )
            return done(
                _fail_without_attempt(
                    db,
                    owner_id,
                    request_id,
                    codes.INTERNAL_ERROR,
                    FailureKind.INVALID_INPUT,
                )
            )
        except Exception as exc:  # noqa: BLE001 - settle, never leak
            _rollback_if_open(db)
            logger.error(
                "chat pipeline request_id=%s unexpected %s",
                request_id,
                type(exc).__name__,
            )
            return done(
                _fail_without_attempt(
                    db,
                    owner_id,
                    request_id,
                    codes.INTERNAL_ERROR,
                    FailureKind.UNEXPECTED,
                )
            )
        return {"route": "start", "prompt": prompt}

    def start(state: _State) -> dict[str, Any]:
        prompt = state["prompt"]
        started = start_attempt(
            db,
            owner_id,
            request_id,
            reserved_tokens=prompt.reservation_tokens,
            daily_limit_tokens=settings.daily_token_limit,
            lease_seconds=LEASE_SECONDS,
        )
        outcome = started.outcome
        if outcome is AttemptStartOutcome.STARTED:
            return {"route": "mark_started", "attempt_id": started.attempt_id}
        if outcome is AttemptStartOutcome.QUOTA_EXCEEDED:
            return done(
                _fail_without_attempt(db, owner_id, request_id, codes.QUOTA_EXCEEDED)
            )
        if outcome is AttemptStartOutcome.RETRY_LIMIT_REACHED:
            return done(
                _fail_without_attempt(
                    db, owner_id, request_id, codes.PROVIDER_RATE_LIMITED
                )
            )
        if outcome is AttemptStartOutcome.ATTEMPT_ALREADY_RUNNING:
            return done(PipelineResult(PipelineOutcome.IN_PROGRESS_ELSEWHERE))
        if outcome is AttemptStartOutcome.GENERATION_ALREADY_ATTEMPTED:
            # R2 already ran at entry (or the attempt settled since): R3 + R4 only.
            mapped = _recovery_to_result(
                finalize_after_terminal_attempt(db, owner_id, request_id)
            )
            return done(
                mapped
                if mapped is not None
                else PipelineResult(PipelineOutcome.INCONSISTENT)
            )
        if outcome is AttemptStartOutcome.REQUEST_NOT_FOUND:
            return done(PipelineResult(PipelineOutcome.NOT_FOUND))
        return done(_stored_replay(db, owner_id, request_id))  # REQUEST_NOT_IN_PROGRESS

    def mark_started(state: _State) -> dict[str, Any]:
        attempt_id = state["attempt_id"]
        try:
            marked = mark_llm_started(db, attempt_id)
        except Exception as exc:  # noqa: BLE001 - settle, never leak
            _rollback_if_open(db)
            logger.error(
                "chat pipeline request_id=%s mark_started %s",
                request_id,
                type(exc).__name__,
            )
            return done(_abort_before_call(attempt_id))
        if marked is TransitionOutcome.APPLIED:
            return {"route": "generate"}
        return done(_stopped(f"mark_started={marked.value}"))

    def _abort_before_call(attempt_id: int) -> PipelineResult:
        # Only a still-unstarted attempt can be failed here. Otherwise leave it to lease expiry.
        if (
            fail_before_call(db, attempt_id, codes.INTERNAL_ERROR)
            is TransitionOutcome.APPLIED
        ):
            return _fail_without_attempt(
                db, owner_id, request_id, codes.INTERNAL_ERROR, FailureKind.UNEXPECTED
            )
        return _stopped("abort_before_call")

    def _stopped(why: str) -> PipelineResult:
        logger.warning("chat pipeline request_id=%s stopped %s", request_id, why)
        return PipelineResult(PipelineOutcome.STOPPED)

    def generate(state: _State) -> dict[str, Any]:
        prompt = state["prompt"]
        try:
            result = provider.generate(prompt.system_message, prompt.user_message)
        except Exception as exc:  # noqa: BLE001 - settle, never leak
            # Uncertain: the call may have been billed. Reservation stays charged.
            logger.error(
                "chat pipeline request_id=%s provider raised %s",
                request_id,
                type(exc).__name__,
            )
            _rollback_if_open(db)
            return done(
                _fail_after_generation(
                    db, owner_id, request_id, state["attempt_id"], codes.INTERNAL_ERROR
                )
            )
        return {"route": "settle", "provider_result": result}

    def settle(state: _State) -> dict[str, Any]:
        result = state["provider_result"]
        attempt_id = state["attempt_id"]
        action = settlement_action(result)
        if action is SettlementAction.PUBLISH:
            return done(_publish(state))
        if action in (
            SettlementAction.FAIL_WITH_USAGE,
            SettlementAction.MARK_UNCERTAIN,
        ):
            return done(
                _fail_after_generation(
                    db, owner_id, request_id, attempt_id, result.error_code, result
                )
            )
        if action is SettlementAction.MARK_REJECTED:
            rejected = reject_and_fail_request(
                db, owner_id, request_id, attempt_id, result.error_code
            )
            if rejected.outcome is RejectOutcome.REJECTED:
                return done(
                    PipelineResult(PipelineOutcome.FAILED, error_code=result.error_code)
                )
            if rejected.outcome is RejectOutcome.ALREADY_FINALIZED:
                return done(_stored_replay(db, owner_id, request_id))
            return done(_stopped(f"reject={rejected.outcome.value}"))
        # MARK_RATE_LIMITED
        marked = mark_rate_limited(db, attempt_id, codes.PROVIDER_RATE_LIMITED)
        if marked is not TransitionOutcome.APPLIED:
            return done(_stopped(f"rate_limited={marked.value}"))
        if state.get("retried") or result.retry_after_ms is None:
            return done(
                _fail_without_attempt(
                    db, owner_id, request_id, codes.PROVIDER_RATE_LIMITED
                )
            )
        sleep(result.retry_after_ms / 1000)  # no transaction open, no attempt running
        return {"route": "start", "retried": True}

    def _publish(state: _State) -> PipelineResult:
        result = state["provider_result"]
        prompt = state["prompt"]
        attempt_id = state["attempt_id"]
        try:
            resolved = resolve_citations(result.content, prompt.sources)
        except Exception as exc:  # noqa: BLE001 - settle, never leak
            logger.error(
                "chat pipeline request_id=%s citations %s",
                request_id,
                type(exc).__name__,
            )
            return _fail_after_generation(
                db, owner_id, request_id, attempt_id, codes.INTERNAL_ERROR, result
            )
        if not _is_storable(resolved.text):
            return _fail_after_generation(
                db, owner_id, request_id, attempt_id, codes.INVALID_RESPONSE, result
            )
        usage = result.usage
        try:
            published = publish_success(
                db,
                owner_id,
                request_id,
                attempt_id,
                user_content=question,
                assistant_content=resolved.text,
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                completion_tokens=usage.completion_tokens
                if usage is not None
                else None,
                citations=resolved.citations,
                retrieved_chunks=tuple(source.chunk for source in prompt.sources),
            )
        except Exception as exc:  # noqa: BLE001 - settle, never leak
            _rollback_if_open(db)
            logger.error(
                "chat pipeline request_id=%s publish %s", request_id, type(exc).__name__
            )
            return _fail_after_generation(
                db, owner_id, request_id, attempt_id, codes.INTERNAL_ERROR, result
            )
        outcome = published.outcome
        if outcome is PublishOutcome.PUBLISHED:
            logger.info("chat pipeline request_id=%s answered", request_id)
            return PipelineResult(PipelineOutcome.ANSWERED, published=published)
        if outcome is PublishOutcome.ALREADY_SUCCEEDED:
            return _stored_replay(db, owner_id, request_id)
        if outcome is PublishOutcome.REQUEST_FAILED:
            return _stored_replay(db, owner_id, request_id)
        if outcome is PublishOutcome.INVALID_CITATIONS:
            return _fail_after_generation(
                db, owner_id, request_id, attempt_id, codes.INVALID_RESPONSE, result
            )
        if outcome is PublishOutcome.CONVERSATION_NOT_FOUND:
            return _fail_after_generation(
                db,
                owner_id,
                request_id,
                attempt_id,
                codes.CONVERSATION_NOT_FOUND,
                result,
            )
        # REQUEST_NOT_FOUND, NOT_RUNNING, LEASE_EXPIRED, WRONG_PHASE: stop, never re-generate.
        return _stopped(f"publish={outcome.value}")

    nodes: dict[str, _Node] = {
        "recover": recover,
        "prepare": prepare,
        "start": start,
        "mark_started": mark_started,
        "generate": generate,
        "settle": settle,
    }
    graph = StateGraph(_State)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "recover")
    targets = {**{name: name for name in nodes}, "end": END}
    for name in nodes:
        graph.add_conditional_edges(name, lambda state: state["route"], targets)
    final = graph.compile().invoke({"retried": False})
    return final["result"]
