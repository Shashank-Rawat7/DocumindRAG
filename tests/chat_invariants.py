"""Whole-history invariant observer for chat requests (C9.5b observer, hardened in C9.5c).

``check_history`` is a PURE function: it takes one request row and its COMPLETE ordered attempt
history and returns every violation it finds, so negative self-tests can feed it hand-built
invalid histories. ``check_database`` is the database-backed half: it follows the request's
message and citation rows. ``observe`` runs both; ``assert_valid`` raises.

Every rule below is derived from the schema CHECKs (chat_request / chat_request_attempt /
message / message_citation models) and from what the C8/C9 code actually writes; none is
invented. Where the code or schema leaves something open, the rule is NOT asserted and the
gap is named in "Deliberately not asserted".

Attempt classes
    UNSTARTED    failed_before_call, abandoned      llm_started_at NULL, charge 0
    RATE_LIMITED rate_limited                       llm_started_at set,  charge 0
    RETRYABLE    UNSTARTED + RATE_LIMITED           a later attempt may follow; nothing billed
    RUNNING      running
    PAID         succeeded, failed_with_usage, uncertain, rejected
                                                    reached the provider and are final: no
                                                    attempt may ever follow (C9.5a guard)

States (request status + attempt history)
    S1  in_progress, no attempts yet
    S2  in_progress, the last attempt is running (live lease, or expired and not yet expired-out)
    S3  in_progress, the last attempt is rate_limited          (the one 429 retry window)
    S4  in_progress, the last attempt is unstarted             (retry is safe)
    S5  in_progress, the last attempt is uncertain with error_code 'lease_expired' and charge =
        reservation: the single permitted intermediate state, between ``expire_stale_attempt``
        and ``fail_request``. Recoverable by R4; it can neither publish nor admit an attempt.
    S6  failed
    S7  succeeded

Invariants
    I1  attempt numbers are 1..n in order, with no gaps.
    I2  at most one attempt is running; it is the last one, and the request is in_progress.
    I3  retry boundary, stated over ALL attempts: every attempt that has a successor is
        RETRYABLE; there are at most two rate_limited attempts; and no attempt starts once two
        rate_limited attempts already exist (start_attempt's RETRY_LIMIT_REACHED).
    I4  each attempt row is internally consistent: numeric domains, usage pairing, status vs
        llm_started_at vs usage vs charge vs error_code (``_check_attempt``).
    I5  a request is succeeded iff exactly one attempt succeeded, and that attempt is the last.
    I6  an in_progress request is in S1-S5: the only settled PAID attempt it may hold is S5's.
    I7  a failed request has no running attempt; its error_code is in the frozen vocabulary;
        when its last attempt is PAID the request's error_code equals that attempt's; when the
        last attempt is missing / unstarted / rate_limited the code is one of the codes the
        pipeline can write in that position (``PRE_ATTEMPT_CODES``, ``RATE_LIMITED_CODES``).
    I8  the request row agrees with its status (completed_at, error_code, message ids,
        conversation), and in the database the messages, citations and completion time agree
        with the request and its terminal attempt (``check_database``).

Deliberately not asserted
    * A running attempt's llm_started_at: no schema rule fixes it, and both values are real
      (NULL before ``mark_llm_started``, set after), so either is legal.
    * Which chunks were retrieved for the request: no retrieval snapshot is persisted (C9.4b
      decision), so citations are checked against the database, not against retrieval.
    * Marker order inside answers that contain a backtick: finding code spans is the production
      parser's job and is not re-implemented, so those answers get the presence check only. All
      other answers must list their distinct markers in ordinal order (``_check_marker_order``).
    * marker == ordinal: not an invariant. ordinal is the order of first appearance, marker is
      the n in ``[n]``; they differ whenever the answer cites ``[2]`` before ``[1]``.
    * source_version / source_sha256 of a citation against the document: the columns snapshot the
      version the answer was built from (publication checks them against the document under
      FOR SHARE), so a later re-index legitimately makes them differ. The schema CHECKs fix their
      format; nothing else is asserted.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.core.database import engine

UNSTARTED = ("failed_before_call", "abandoned")
RATE_LIMITED = "rate_limited"
RETRYABLE = (*UNSTARTED, RATE_LIMITED)
PAID = ("succeeded", "failed_with_usage", "uncertain", "rejected")
ALL_STATUSES = (*RETRYABLE, "running", *PAID)
MAX_RATE_LIMITED = 2
LEASE_EXPIRED = "lease_expired"

# The frozen error-code vocabulary (app/rag/chat_error_codes.py), pinned here as literals so a
# change to the production module cannot silently change what the observer accepts.
VOCABULARY = frozenset(
    {
        "conversation_not_found",
        "no_context",
        "quota_exceeded",
        "provider_rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "provider_rejected",
        "invalid_response",
        "lease_expired",
        "internal_error",
    }
)
# What the pipeline writes with ``fail_request`` when no paid attempt exists (every call site is
# ``_fail_without_attempt`` in app/rag/chat_pipeline.py): no_context (NoUsableContextError),
# internal_error (D8 prepare failures, a failed pre-call mark), quota_exceeded (start_attempt
# QUOTA_EXCEEDED, including a refused 429 retry). Recovery R4 finalizes only a PAID terminal
# attempt, so an expired unstarted attempt (abandoned, error_code lease_expired) is never the
# source of a request's code: the next entry proceeds and one of these three ends the request.
PRE_ATTEMPT_CODES = frozenset({"no_context", "internal_error", "quota_exceeded"})
# provider_rate_limited only ever ends a request whose last attempt is rate_limited (a second
# 429, a 429 without retry-after, or RETRY_LIMIT_REACHED after two of them).
RATE_LIMITED_CODES = PRE_ATTEMPT_CODES | {"provider_rate_limited"}


@dataclass(frozen=True)
class AttemptRow:
    number: int
    status: str
    reserved: int
    charged: int | None
    error_code: str | None = None
    started: bool = False  # llm_started_at IS NOT NULL
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class RequestRow:
    status: str
    error_code: str | None = None
    completed: bool = False
    user_message_id: int | None = None
    assistant_message_id: int | None = None
    conversation_id: int | None = None


@dataclass(frozen=True)
class Violation:
    invariant: str  # "I1".."I8", or "S" when the history fits no state
    detail: str


def _usage_total(a: AttemptRow) -> int | None:
    if a.prompt_tokens is None or a.completion_tokens is None:
        return None
    return a.prompt_tokens + a.completion_tokens


def _check_attempt(a: AttemptRow) -> list[str]:
    """I4 for one attempt; returns problem descriptions."""
    n = a.number
    if a.status not in ALL_STATUSES:
        return [f"attempt {n}: unknown status {a.status!r}"]
    problems = []

    # numeric domains (ck_..._reserved_tokens_positive, ck_..._usage_valid)
    if a.reserved is None or a.reserved <= 0:
        problems.append(
            f"attempt {n}: reserved_tokens must be positive, got {a.reserved}"
        )
    for name, value in (
        ("prompt_tokens", a.prompt_tokens),
        ("completion_tokens", a.completion_tokens),
        ("charged_tokens", a.charged),
    ):
        if value is not None and value < 0:
            problems.append(f"attempt {n}: {name} must not be negative, got {value}")

    # usage is reported as a pair (ck_..._usage_valid)
    paired = (a.prompt_tokens is None) == (a.completion_tokens is None)
    has_usage = a.prompt_tokens is not None and a.completion_tokens is not None
    if not paired:
        problems.append(
            f"attempt {n}: prompt_tokens and completion_tokens must be both set or both NULL"
        )

    # llm_started_at vs status (ck_..._llm_started_consistent); a running attempt may be either
    if a.status in UNSTARTED and a.started:
        problems.append(f"attempt {n}: {a.status} must not have llm_started_at")
    if a.status not in UNSTARTED and a.status != "running" and not a.started:
        problems.append(f"attempt {n}: {a.status} must have llm_started_at")

    # usage by status (ck_..._charge_consistent)
    if a.status == "failed_with_usage":
        if paired and not has_usage:
            problems.append(f"attempt {n}: failed_with_usage must carry usage")
    elif a.status != "succeeded" and (
        a.prompt_tokens is not None or a.completion_tokens is not None
    ):
        problems.append(f"attempt {n}: {a.status} must not carry usage")

    # charge by status (ck_..._charge_consistent, ck_..._charged_tokens_null_iff_running)
    if a.status == "running":
        if a.charged is not None:
            problems.append(f"attempt {n}: running must not be charged")
    elif a.status in (*UNSTARTED, RATE_LIMITED, "rejected"):
        if a.charged != 0:
            problems.append(f"attempt {n}: {a.status} must charge 0, got {a.charged}")
    elif a.status == "uncertain":
        if a.charged != a.reserved:
            problems.append(f"attempt {n}: uncertain must charge its reservation")
    elif paired:  # failed_with_usage / succeeded, with a well-formed usage state
        if a.status == "failed_with_usage" and a.charged != _usage_total(a):
            problems.append(f"attempt {n}: failed_with_usage must charge its usage")
        if a.status == "succeeded":
            expected = _usage_total(a) if has_usage else a.reserved
            if a.charged != expected:
                problems.append(
                    f"attempt {n}: succeeded must charge usage or reservation"
                )

    # error_code is NULL exactly for running and succeeded (ck_..._error_code_consistent)
    if a.status in ("running", "succeeded"):
        if a.error_code is not None:
            problems.append(f"attempt {n}: {a.status} must not carry an error_code")
    elif a.error_code is None:
        problems.append(f"attempt {n}: {a.status} must carry an error_code")
    return problems


def classify(request: RequestRow, attempts: list[AttemptRow]) -> str | None:
    """The state S1-S7 the history is in, or None when it is in none of them."""
    if request.status == "succeeded":
        return "S7"
    if request.status == "failed":
        return "S6"
    if request.status != "in_progress":
        return None
    if not attempts:
        return "S1"
    last = attempts[-1]
    if last.status == "running":
        return "S2"
    if last.status == RATE_LIMITED:
        return "S3"
    if last.status in UNSTARTED:
        return "S4"
    if (
        last.status == "uncertain"
        and last.error_code == LEASE_EXPIRED
        and last.charged == last.reserved
    ):
        return "S5"
    return None


def _failed_code_problem(request: RequestRow, attempts: list[AttemptRow]) -> str | None:
    """I7's code rule for a failed request (None when fine)."""
    code = request.error_code
    if code is None:
        return None  # I8 reports a failed request without a code
    if code not in VOCABULARY:
        return f"request error_code {code!r} is not in the frozen vocabulary"
    last = attempts[-1] if attempts else None
    if last is not None and last.status in PAID and last.status != "succeeded":
        if code != last.error_code:
            return (
                f"request error_code {code!r} differs from its last attempt's"
                f" {last.error_code!r}"
            )
        return None
    if last is None or last.status in UNSTARTED:
        if code not in PRE_ATTEMPT_CODES:
            where = "no attempt" if last is None else f"a last {last.status} attempt"
            return f"error_code {code!r} is not a pre-attempt code, but the request has {where}"
    elif last.status == RATE_LIMITED and code not in RATE_LIMITED_CODES:
        return f"error_code {code!r} cannot end a request whose last attempt is rate_limited"
    return None


def check_history(request: RequestRow, attempts: list[AttemptRow]) -> list[Violation]:
    found: list[Violation] = []

    def add(invariant: str, detail: str) -> None:
        found.append(Violation(invariant, detail))

    # I1 - contiguous numbering in order.
    if [a.number for a in attempts] != list(range(1, len(attempts) + 1)):
        add(
            "I1",
            f"attempt numbers {[a.number for a in attempts]} are not 1..n in order",
        )

    # I4 - every attempt on its own.
    for attempt in attempts:
        for problem in _check_attempt(attempt):
            add("I4", problem)

    # I2 - the running slot.
    running = [a for a in attempts if a.status == "running"]
    if len(running) > 1:
        add("I2", f"{len(running)} attempts are running")
    for attempt in running:
        if attempt is not attempts[-1]:
            add("I2", f"running attempt {attempt.number} is not the last attempt")
        if request.status != "in_progress":
            add(
                "I2",
                f"attempt {attempt.number} is running on a {request.status} request",
            )

    # I3 - the retry boundary, strictly over ALL attempts, not just the latest.
    for attempt in attempts[:-1]:
        if attempt.status not in RETRYABLE:
            add(
                "I3",
                f"attempt {attempt.number} ({attempt.status}) has a successor but is not retryable",
            )
    limited = sum(1 for a in attempts if a.status == RATE_LIMITED)
    if limited > MAX_RATE_LIMITED:
        add(
            "I3",
            f"{limited} rate_limited attempts exceed the cap of {MAX_RATE_LIMITED}",
        )
    for index, attempt in enumerate(attempts):
        before = sum(
            1 for earlier in attempts[:index] if earlier.status == RATE_LIMITED
        )
        if before >= MAX_RATE_LIMITED:
            add(
                "I3",
                f"attempt {attempt.number} started after {before} rate_limited attempts"
                " (the retry limit)",
            )

    # I5 - success and the succeeded attempt are one and the same fact.
    succeeded = [a for a in attempts if a.status == "succeeded"]
    if request.status == "succeeded":
        if len(succeeded) != 1:
            add("I5", f"succeeded request has {len(succeeded)} succeeded attempts")
        elif succeeded[0] is not attempts[-1]:
            add("I5", "the succeeded attempt is not the last attempt")
    elif succeeded:
        add("I5", f"{request.status} request has a succeeded attempt")

    # I6 - the in_progress states.
    if request.status == "in_progress" and classify(request, attempts) is None:
        add(
            "I6",
            "in_progress request holds a settled paid attempt that is not state S5",
        )

    # I7 - failed requests.
    if request.status == "failed":
        if running:
            add("I7", "failed request still has a running attempt")
        problem = _failed_code_problem(request, attempts)
        if problem is not None:
            add("I7", problem)

    # I8 - the request row itself.
    ids = (request.user_message_id, request.assistant_message_id)
    any_id = any(i is not None for i in ids)
    if request.status == "in_progress":
        if request.completed or request.error_code is not None or any_id:
            add("I8", "in_progress request has completion data")
    elif request.status == "failed":
        if not request.completed or request.error_code is None or any_id:
            add(
                "I8",
                "failed request needs completed_at and error_code and no message ids",
            )
    elif request.status == "succeeded":
        if not request.completed or request.error_code is not None:
            add("I8", "succeeded request needs completed_at and no error_code")
        pointers = (*ids, request.conversation_id)
        # Either all three pointers are set, or all are NULL: deleting the conversation sets
        # them NULL together (chat_request model: the row outlives the conversation).
        if any(p is None for p in pointers) and not all(p is None for p in pointers):
            add(
                "I8",
                "succeeded request has only some of conversation, user and assistant ids",
            )
    else:
        add("I8", f"unknown request status {request.status!r}")

    if classify(request, attempts) is None and request.status != "in_progress":
        add("S", "history fits no state")
    return found


# ---------------------------------------------------------------- the database half of I8


@dataclass(frozen=True)
class _Message:
    id: int
    conversation_id: int
    seq: int
    role: str
    content: str


_MARKER = re.compile(r"\[([0-9]{1,3})\]")


def _check_marker_order(
    answer: str, citations: list[Any], add: Callable[[str], None]
) -> None:
    """ordinal = order of first appearance of a distinct marker; marker = the n in ``[n]``.

    They are different numbers (an answer that cites ``[2]`` first stores ordinal 1, marker 2),
    so equality of the two is NOT an invariant. What is: the stored answer keeps every valid
    marker and removes every invalid one, so outside code spans its distinct markers, in order of
    first appearance, are exactly the citations' markers by ordinal. Code spans make a literal
    ``[n]`` uncited, and finding them is the production scanner's job, so an answer containing a
    backtick is skipped rather than re-implemented here.
    """
    if "`" in answer:
        return
    seen: list[int] = []
    for match in _MARKER.finditer(answer):
        raw = match.group(1)
        number = int(raw)
        if raw == str(number) and number not in seen:
            seen.append(number)
    cited = [c.marker for c in citations]
    if seen != cited:
        add(
            f"citation markers {cited} (by ordinal) are not the answer's markers {seen} "
            "in order of first appearance"
        )


def check_database(request_id: int) -> list[Violation]:
    """I8 against the messages, citations and attempts the request actually points at."""
    found: list[Violation] = []

    def add(detail: str) -> None:
        found.append(Violation("I8", detail))

    with engine.begin() as conn:
        req = conn.execute(
            text(
                "SELECT owner_id, status, conversation_id, user_message_id,"
                " assistant_message_id, completed_at FROM chat_requests WHERE id = :i"
            ),
            {"i": request_id},
        ).one()

        # The conversation the request points at belongs to the request's owner, whatever the
        # request's status (a failed request may name an existing conversation too).
        if req.conversation_id is not None:
            owner = conn.execute(
                text("SELECT owner_id FROM conversations WHERE id = :c"),
                {"c": req.conversation_id},
            ).scalar()
            if owner != req.owner_id:
                add("the request's conversation belongs to another user")

        # A terminal request is completed after its last attempt was settled. The schema already
        # forces finished_at NULL exactly for a running attempt
        # (ck_chat_request_attempts_finished_at_consistent), so a missing finished_at here means
        # the last attempt is still running under a terminal request: say so, never skip it.
        if req.status != "in_progress":
            last = conn.execute(
                text(
                    "SELECT status, finished_at FROM chat_request_attempts"
                    " WHERE request_id = :i ORDER BY attempt_number DESC LIMIT 1"
                ),
                {"i": request_id},
            ).first()
            if last is not None:
                if last.finished_at is None:
                    add(
                        f"terminal request whose last attempt ({last.status}) has no finished_at"
                    )
                elif (
                    req.completed_at is not None and req.completed_at < last.finished_at
                ):
                    add("request completed before its last attempt was settled")

        if req.status != "succeeded" or req.user_message_id is None:
            return found  # nothing more to follow (a deleted conversation nulls every pointer)

        def message(message_id: int | None) -> _Message | None:
            if message_id is None:
                return None
            row = conn.execute(
                text(
                    "SELECT id, conversation_id, seq, role, content FROM messages"
                    " WHERE id = :m"
                ),
                {"m": message_id},
            ).first()
            return None if row is None else _Message(*row)

        user = message(req.user_message_id)
        assistant = message(req.assistant_message_id)
        if user is None or assistant is None:
            add("a message the succeeded request points at does not exist")
            return found
        if user.id == assistant.id:
            add("the user and assistant message are the same row")
        if user.role != "user":
            add(f"user_message_id points at a {user.role!r} message")
        if assistant.role != "assistant":
            add(f"assistant_message_id points at a {assistant.role!r} message")
        for label, msg in (("user", user), ("assistant", assistant)):
            if msg.conversation_id != req.conversation_id:
                add(f"the {label} message is not in the request's conversation")
        if (
            user.conversation_id == assistant.conversation_id
            and assistant.seq != user.seq + 1
        ):
            add(f"assistant seq {assistant.seq} does not follow user seq {user.seq}")
        # Citations hang off the assistant message only, and are well formed.
        stray = conn.execute(
            text("SELECT count(*) FROM message_citations WHERE message_id = :m"),
            {"m": user.id},
        ).scalar()
        if stray and user.id != assistant.id:
            add("the user message has citations")
        citations = conn.execute(
            text(
                "SELECT c.ordinal, c.marker, c.chunk_id, c.document_id, c.chunk_index,"
                " c.start_offset, c.end_offset, c.page_start, c.page_end, c.span_sha256,"
                " k.document_id AS k_document, k.chunk_index AS k_index,"
                " k.start_offset AS k_start, k.end_offset AS k_end,"
                " k.page_start AS k_page_start, k.page_end AS k_page_end,"
                " k.span_sha256 AS k_span"
                " FROM message_citations c LEFT JOIN document_chunks k ON k.id = c.chunk_id"
                " WHERE c.message_id = :m ORDER BY c.ordinal"
            ),
            {"m": assistant.id},
        ).all()
        if [c.ordinal for c in citations] != list(range(1, len(citations) + 1)):
            add(f"citation ordinals {[c.ordinal for c in citations]} are not 1..n")
        _check_marker_order(assistant.content, citations, add)
        for c in citations:
            if f"[{c.marker}]" not in assistant.content:
                add(
                    f"citation {c.ordinal}: marker [{c.marker}] does not appear in the answer"
                )
            document_owner = conn.execute(
                text("SELECT owner_id FROM documents WHERE id = :d"),
                {"d": c.document_id},
            ).scalar()
            if document_owner is not None and document_owner != req.owner_id:
                add(f"citation {c.ordinal}: its document belongs to another user")
            if c.chunk_id is not None:
                # The same fields publication compares before it inserts (_CHUNK_FIELDS). The
                # source_version / source_sha256 snapshot is deliberately not compared: it
                # records the document version the answer was built from and may legitimately
                # differ from the document's current value after a re-index.
                snapshot = (
                    c.document_id,
                    c.chunk_index,
                    c.start_offset,
                    c.end_offset,
                    c.page_start,
                    c.page_end,
                )
                current = (
                    c.k_document,
                    c.k_index,
                    c.k_start,
                    c.k_end,
                    c.k_page_start,
                    c.k_page_end,
                )
                if snapshot != current or c.span_sha256 != c.k_span:
                    add(
                        f"citation {c.ordinal}: its snapshot disagrees with the chunk it cites"
                    )
    return found


# ---------------------------------------------------------------- loading and asserting


def load(request_id: int) -> tuple[RequestRow, list[AttemptRow]]:
    with engine.begin() as conn:
        r = conn.execute(
            text(
                "SELECT status, error_code, completed_at, user_message_id,"
                " assistant_message_id, conversation_id FROM chat_requests WHERE id = :i"
            ),
            {"i": request_id},
        ).one()
        attempts = conn.execute(
            text(
                "SELECT attempt_number, status, reserved_tokens, charged_tokens, error_code,"
                " llm_started_at, prompt_tokens, completion_tokens"
                " FROM chat_request_attempts WHERE request_id = :i ORDER BY attempt_number"
            ),
            {"i": request_id},
        ).all()
    request = RequestRow(
        r.status,
        r.error_code,
        r.completed_at is not None,
        r.user_message_id,
        r.assistant_message_id,
        r.conversation_id,
    )
    rows = [
        AttemptRow(
            a.attempt_number,
            a.status,
            a.reserved_tokens,
            a.charged_tokens,
            a.error_code,
            a.llm_started_at is not None,
            a.prompt_tokens,
            a.completion_tokens,
        )
        for a in attempts
    ]
    return request, rows


def observe(request_id: int) -> list[Violation]:
    return [*check_history(*load(request_id)), *check_database(request_id)]


def state_of(request_id: int) -> str | None:
    return classify(*load(request_id))


def assert_valid(request_id: int) -> None:
    violations = observe(request_id)
    assert not violations, f"request {request_id}: {violations}"


def all_request_ids() -> list[int]:
    with engine.begin() as conn:
        return [
            row[0]
            for row in conn.execute(text("SELECT id FROM chat_requests ORDER BY id"))
        ]


def assert_every_request_valid() -> None:
    for request_id in all_request_ids():
        assert_valid(request_id)
