"""Sequential execution engine for the indexing backfill (Component 12 / C5.5, CP6b).

Pure orchestration: the database, the runner and the embedder are reached only through
three injected callables, so everything here is testable with plain fakes and this
module imports nothing from the application beyond the pure classifier.

    reclassify(id) -> Classification | None   fresh read; None = gone or no longer completed
    request(id)    -> bool                    queue the document (request_indexing)
    run(id)        -> an indexing run result  (run_indexing); its ``outcome``, ``chunk_count``
                                              and ``reason`` are read, nothing else

For every document the plan says to index, in id order, one at a time: re-read and
reclassify, queue, run, and record exactly one ItemResult. Failures never stop the loop;
KeyboardInterrupt does, and the summary is still returned. A result's ``detail`` holds
fixed vocabulary only (a category, a source status, a failure-reason name or an exception
CLASS name): exception messages are never read.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from app.rag.backfill import Action, BackfillPlan, Classification

DETAIL_NO_LONGER_SELECTED = "no_longer_selected"
DETAIL_UNRECOGNIZED_OUTCOME = "unrecognized_run_outcome"


class ItemOutcome(str, Enum):
    """What happened to one planned document. Exactly one per document."""

    INDEXED = "indexed"                          # published
    FAILED = "failed"                            # the attempt failed and the failure was recorded
    FAILURE_NOT_RECORDED = "failure_not_recorded"
    ERROR = "error"                              # an exception, or a result we could not read
    BLOCKED = "blocked"                          # reclassified as BLOCKED just before indexing
    CHANGED = "changed"                          # reclassified as SKIP/REPORT, or no longer selected
    NOT_QUEUED = "not_queued"                    # request_indexing declined
    NOT_CLAIMED = "not_claimed"                  # run_indexing found nothing to claim
    STALE = "stale"                              # another attempt owns it, or its source changed
    GONE = "gone"                                # deleted meanwhile
    INTERRUPTED = "interrupted"                  # Ctrl-C while the document may have been written
    NOT_ATTEMPTED = "not_attempted"              # never started (or Ctrl-C before any write)


_FAILURES = frozenset({ItemOutcome.FAILED, ItemOutcome.FAILURE_NOT_RECORDED, ItemOutcome.ERROR})
_CHANGED = frozenset(
    {
        ItemOutcome.CHANGED,
        ItemOutcome.NOT_QUEUED,
        ItemOutcome.NOT_CLAIMED,
        ItemOutcome.STALE,
        ItemOutcome.GONE,
    }
)

# RunOutcome values (app/rag/indexing_runner.py) by string, so this module need not import the
# runner. A test checks the keys against the real enum. Anything else maps to ERROR.
RUN_OUTCOME_TO_ITEM_OUTCOME = MappingProxyType(
    {
        "published": ItemOutcome.INDEXED,
        "not_claimed": ItemOutcome.NOT_CLAIMED,
        "stale": ItemOutcome.STALE,
        "gone": ItemOutcome.GONE,
        "failed": ItemOutcome.FAILED,
        "failure_not_recorded": ItemOutcome.FAILURE_NOT_RECORDED,
    }
)


@dataclass(frozen=True, slots=True)
class ItemResult:
    """The result for one document."""

    document_id: int
    outcome: ItemOutcome
    detail: str | None = None           # fixed vocabulary only (see the module docstring)
    chunk_count: int | None = None      # INDEXED only
    elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("detail must be a str or None")
        if self.chunk_count is not None and self.outcome is not ItemOutcome.INDEXED:
            raise ValueError("only an INDEXED result can carry a chunk count")


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Every plan document accounted for exactly once.

    ``results`` has one entry per document the plan told us to index (in id order); the
    ``plan_*`` fields count the rest of the plan. The constructor checks that they add up.
    """

    examined: int
    plan_deferred: int
    plan_blocked: int
    plan_reported: int
    plan_skipped: int
    results: tuple[ItemResult, ...]
    interrupted: bool
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name in ("examined", "plan_deferred", "plan_blocked", "plan_reported", "plan_skipped"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple")
        ids = [result.document_id for result in self.results]
        if len(set(ids)) != len(ids):
            raise ValueError("a document appears more than once in the results")
        accounted = (
            len(self.results)
            + self.plan_deferred
            + self.plan_blocked
            + self.plan_reported
            + self.plan_skipped
        )
        if accounted != self.examined:
            raise ValueError(f"accounting does not add up: {accounted} accounted for, {self.examined} examined")

    @property
    def planned(self) -> int:
        """Documents the plan told us to index (deferred ones excluded)."""
        return len(self.results)

    def counts(self) -> dict[ItemOutcome, int]:
        """Results per outcome, with a zero for every outcome."""
        found = {outcome: 0 for outcome in ItemOutcome}
        for result in self.results:
            found[result.outcome] += 1
        return found

    @property
    def indexed(self) -> int:
        return self.counts()[ItemOutcome.INDEXED]

    @property
    def failures(self) -> int:
        return sum(count for outcome, count in self.counts().items() if outcome in _FAILURES)

    @property
    def blocked(self) -> int:
        """Documents found BLOCKED at run time (those blocked in the plan are ``plan_blocked``)."""
        return self.counts()[ItemOutcome.BLOCKED]

    @property
    def changed(self) -> int:
        """Documents someone else changed or owns (skipped, not failures)."""
        return sum(count for outcome, count in self.counts().items() if outcome in _CHANGED)

    @property
    def needs_attention(self) -> bool:
        """True when the run had failures or blocked documents (the exit-code-1 condition)."""
        return self.failures > 0 or self.blocked + self.plan_blocked > 0


class _Marker:
    """Set once the loop may have written to the database for the current document."""

    __slots__ = ("writing",)

    def __init__(self) -> None:
        self.writing = False


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _class_name(exc: BaseException) -> str:
    return _ascii(type(exc).__name__)


def _from_run_result(result: object) -> tuple[ItemOutcome, str | None, int | None]:
    """Translate a run result. May raise for a malformed one; the caller records an ERROR."""
    raw = result.outcome  # type: ignore[attr-defined]
    mapped = RUN_OUTCOME_TO_ITEM_OUTCOME.get(getattr(raw, "value", raw))
    if mapped is None:
        return ItemOutcome.ERROR, DETAIL_UNRECOGNIZED_OUTCOME, None
    if mapped is ItemOutcome.INDEXED:
        count = getattr(result, "chunk_count", None)
        return ItemOutcome.INDEXED, None, (count if type(count) is int and count >= 0 else None)
    if mapped in (ItemOutcome.FAILED, ItemOutcome.FAILURE_NOT_RECORDED):
        reason = getattr(getattr(result, "reason", None), "name", None)
        return mapped, (_ascii(reason) if isinstance(reason, str) else None), None
    return mapped, None, None


def _attempt(
    document_id: int,
    *,
    reclassify: Callable[[int], Classification | None],
    request: Callable[[int], bool],
    run: Callable[[int], object],
    marker: _Marker,
) -> tuple[ItemOutcome, str | None, int | None]:
    """Process one document. Exceptions (not BaseExceptions) become ERROR results."""
    try:
        classification = reclassify(document_id)
    except Exception as exc:  # noqa: BLE001 - only the class name is kept
        return ItemOutcome.ERROR, _class_name(exc), None

    if classification is None:
        return ItemOutcome.CHANGED, DETAIL_NO_LONGER_SELECTED, None
    if classification.action is Action.BLOCKED:
        return ItemOutcome.BLOCKED, classification.note or classification.category.value, None
    if classification.action is not Action.INDEX:
        return ItemOutcome.CHANGED, classification.category.value, None

    marker.writing = True  # from here on the database may be written
    try:
        queued = request(document_id)
    except Exception as exc:  # noqa: BLE001
        return ItemOutcome.ERROR, _class_name(exc), None
    if queued is not True:  # strict: anything but True is "not queued", and run is never called
        return ItemOutcome.NOT_QUEUED, None, None

    try:
        return _from_run_result(run(document_id))
    except Exception as exc:  # noqa: BLE001
        return ItemOutcome.ERROR, _class_name(exc), None


def execute_plan(
    plan: BackfillPlan,
    *,
    reclassify: Callable[[int], Classification | None],
    request: Callable[[int], bool],
    run: Callable[[int], object],
    on_result: Callable[[int, int, ItemResult], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ExecutionSummary:
    """Index the plan's documents one at a time and account for every document.

    ``on_result(position, total, result)`` is called after each document that completes,
    best effort: an Exception it raises is ignored. KeyboardInterrupt stops the loop and
    the summary is still returned with ``interrupted=True``; any other BaseException
    propagates.
    """
    ids = tuple(item.classification.document_id for item in plan.to_run())
    by_action = plan.counts_by_action()
    started = clock()

    results: list[ItemResult] = []
    interrupted = False
    pending: tuple[int, _Marker] | None = None

    try:
        for position, document_id in enumerate(ids, start=1):
            marker = _Marker()
            pending = (document_id, marker)
            item_started = clock()
            try:
                outcome, detail, chunk_count = _attempt(
                    document_id, reclassify=reclassify, request=request, run=run, marker=marker
                )
                result = ItemResult(document_id, outcome, detail, chunk_count, clock() - item_started)
            except Exception as exc:  # noqa: BLE001 - a malformed answer is an ERROR, never a lost summary
                result = ItemResult(
                    document_id, ItemOutcome.ERROR, _class_name(exc), None, clock() - item_started
                )
            results.append(result)
            pending = None
            if on_result is not None:
                try:
                    on_result(position, len(ids), result)
                except Exception:  # noqa: BLE001 - progress output is best effort
                    pass
    except KeyboardInterrupt:
        interrupted = True
        if pending is not None and not (results and results[-1].document_id == pending[0]):
            document_id, marker = pending
            kind = ItemOutcome.INTERRUPTED if marker.writing else ItemOutcome.NOT_ATTEMPTED
            results.append(ItemResult(document_id, kind))
        for document_id in ids[len(results):]:
            results.append(ItemResult(document_id, ItemOutcome.NOT_ATTEMPTED))

    return ExecutionSummary(
        examined=len(plan.items),
        plan_deferred=len(plan.deferred()),
        plan_blocked=by_action[Action.BLOCKED],
        plan_reported=by_action[Action.REPORT],
        plan_skipped=by_action[Action.SKIP],
        results=tuple(results),
        interrupted=interrupted,
        elapsed_seconds=clock() - started,
    )