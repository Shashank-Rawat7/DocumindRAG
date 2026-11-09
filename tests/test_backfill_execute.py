"""Tests for the backfill execution engine (C5.5 / CP6b).

No database and no model: the three callables are fakes driven by a script. Real
IndexingRunResult / RunOutcome / FailureReason values are used so the outcome mapping is
checked against the real runner vocabulary.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.backfill import (
    Action,
    Category,
    Classification,
    DocumentState,
    LeaseState,
    SourceStatus,
    build_plan,
)
from app.rag.backfill_execute import (
    DETAIL_NO_LONGER_SELECTED,
    DETAIL_UNRECOGNIZED_OUTCOME,
    RUN_OUTCOME_TO_ITEM_OUTCOME,
    ExecutionSummary,
    ItemOutcome,
    ItemResult,
    execute_plan,
)
from app.rag.indexing_runner import IndexingRunResult, RunOutcome
from app.rag.indexing_service import FailureOutcome, FailureReason

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ID = "test/fake@" + "0" * 40
SECRET = "s3cr3t-p4ssw0rd-XYZ"


# ---------- builders ----------

def state(document_id: int, kind: str = "index") -> DocumentState:
    values = dict(
        document_id=document_id, processing_status="completed", indexing_status=None, indexing_error=None,
        lease=LeaseState.NONE, source_version=1, indexed_source_version=None, indexed_embedding_id=None,
        has_text=True, chunks_present=False, filename="notes.txt", stored_name="a.txt",
        content_type="text/plain", source=SourceStatus.NOT_PDF,
    )
    if kind == "blocked":
        values.update(source=SourceStatus.PDF_FILE_MISSING, filename="x.pdf", stored_name="x.pdf",
                      content_type="application/pdf")
    elif kind == "reported":
        values.update(indexing_status="failed", indexing_error="boom")
    elif kind == "skipped":
        values.update(indexing_status="indexed", indexed_source_version=1,
                      indexed_embedding_id=EMBEDDING_ID, chunks_present=True)
    return DocumentState(**values)


def make_plan(index=(), blocked=(), reported=(), skipped=(), limit=None):
    states = (
        [state(i) for i in index] + [state(i, "blocked") for i in blocked]
        + [state(i, "reported") for i in reported] + [state(i, "skipped") for i in skipped]
    )
    return build_plan(states, configured_embedding_id=EMBEDDING_ID, limit=limit)


def decision(document_id, action=Action.INDEX, category=Category.NEVER_QUEUED, note=None):
    return Classification(document_id, category, action, (), note)


def published(document_id, chunks=3):
    return IndexingRunResult(document_id=document_id, outcome=RunOutcome.PUBLISHED, attempt=1, chunk_count=chunks)


class Harness:
    """Scripted reclassify/request/run. An answer may be a value, an exception to raise, or a callable."""

    def __init__(self, *, reclassify=None, request=None, run=None):
        self.calls = []
        self.tables = {"reclassify": reclassify or {}, "request": request or {}, "run": run or {}}

    def _answer(self, step, document_id, default):
        self.calls.append((step, document_id))
        answer = self.tables[step].get(document_id, default)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer()
        return answer

    def reclassify(self, document_id):
        return self._answer("reclassify", document_id, decision(document_id))

    def request(self, document_id):
        return self._answer("request", document_id, True)

    def run(self, document_id):
        return self._answer("run", document_id, published(document_id))

    def execute(self, plan, **kwargs):
        return execute_plan(plan, reclassify=self.reclassify, request=self.request, run=self.run, **kwargs)


def by_id(summary):
    return {result.document_id: result for result in summary.results}


def assert_accounting(summary, plan):
    """Every plan document is accounted for exactly once."""
    assert [r.document_id for r in summary.results] == [i.classification.document_id for i in plan.to_run()]
    assert summary.examined == len(plan.items)
    assert summary.planned == len(summary.results)
    assert sum(summary.counts().values()) == summary.planned
    assert summary.examined == (
        summary.planned + summary.plan_deferred + summary.plan_blocked
        + summary.plan_reported + summary.plan_skipped
    )


# ---------- the happy path and ordering ----------

def test_every_planned_document_is_processed_once_in_id_order_and_sequentially():
    plan = make_plan(index=[9, 2, 5])
    harness = Harness()
    summary = harness.execute(plan)

    assert [r.outcome for r in summary.results] == [ItemOutcome.INDEXED] * 3
    assert [r.document_id for r in summary.results] == [2, 5, 9]
    assert all(r.chunk_count == 3 for r in summary.results)
    assert harness.calls == [
        ("reclassify", 2), ("request", 2), ("run", 2),
        ("reclassify", 5), ("request", 5), ("run", 5),
        ("reclassify", 9), ("request", 9), ("run", 9),
    ]
    assert summary.interrupted is False
    assert summary.needs_attention is False
    assert_accounting(summary, plan)


def test_only_documents_the_plan_will_run_are_touched():
    plan = make_plan(index=[1, 2, 3, 4], blocked=[10], reported=[20], skipped=[30, 31], limit=2)
    harness = Harness()
    summary = harness.execute(plan)

    assert {call[1] for call in harness.calls} == {1, 2}  # deferred 3 and 4, blocked, reported, skipped: untouched
    assert (summary.planned, summary.plan_deferred, summary.plan_blocked) == (2, 2, 1)
    assert (summary.plan_reported, summary.plan_skipped, summary.examined) == (1, 2, 8)
    assert_accounting(summary, plan)


def test_an_empty_plan_does_nothing():
    plan = make_plan(skipped=[1, 2])
    harness = Harness()
    summary = harness.execute(plan)
    assert summary.results == () and harness.calls == []
    assert summary.interrupted is False and summary.needs_attention is False
    assert_accounting(summary, plan)


# ---------- every RunOutcome ----------

RUN_CASES = [
    (dict(outcome=RunOutcome.PUBLISHED, attempt=1, chunk_count=5), ItemOutcome.INDEXED, None, 5),
    (dict(outcome=RunOutcome.NOT_CLAIMED), ItemOutcome.NOT_CLAIMED, None, None),
    (dict(outcome=RunOutcome.STALE, attempt=1), ItemOutcome.STALE, None, None),
    (dict(outcome=RunOutcome.GONE, attempt=1), ItemOutcome.GONE, None, None),
    (dict(outcome=RunOutcome.FAILED, attempt=1, reason=FailureReason.EMBEDDER_UNAVAILABLE,
          failure_outcome=FailureOutcome.FAILED), ItemOutcome.FAILED, "EMBEDDER_UNAVAILABLE", None),
    (dict(outcome=RunOutcome.FAILURE_NOT_RECORDED, attempt=1, reason=FailureReason.DATABASE),
     ItemOutcome.FAILURE_NOT_RECORDED, "DATABASE", None),
]


@pytest.mark.parametrize("fields, outcome, detail, chunks", RUN_CASES, ids=[c[0]["outcome"].value for c in RUN_CASES])
def test_each_run_outcome_is_mapped(fields, outcome, detail, chunks):
    plan = make_plan(index=[1, 2])
    harness = Harness(run={1: IndexingRunResult(document_id=1, **fields)})
    summary = harness.execute(plan)

    first = by_id(summary)[1]
    assert (first.outcome, first.detail, first.chunk_count) == (outcome, detail, chunks)
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED  # the loop continues
    assert_accounting(summary, plan)


def test_the_mapping_covers_exactly_the_real_run_outcomes():
    assert set(RUN_OUTCOME_TO_ITEM_OUTCOME) == {outcome.value for outcome in RunOutcome}


def test_an_unrecognized_run_outcome_is_an_error():
    class Result:
        outcome = "something_new"

    harness = Harness(run={1: Result()})
    summary = harness.execute(make_plan(index=[1]))
    assert (summary.results[0].outcome, summary.results[0].detail) == (ItemOutcome.ERROR, DETAIL_UNRECOGNIZED_OUTCOME)


def test_a_malformed_run_result_is_an_error_with_only_the_class_name():
    harness = Harness(run={1: object()})
    summary = harness.execute(make_plan(index=[1, 2]))
    assert summary.results[0].outcome is ItemOutcome.ERROR
    assert summary.results[0].detail == "AttributeError"
    assert summary.results[1].outcome is ItemOutcome.INDEXED

def test_a_malformed_reclassify_result_is_an_error_and_the_loop_continues():
    plan = make_plan(index=[1, 2])
    harness = Harness(reclassify={1: object()})
    summary = harness.execute(plan)

    first = by_id(summary)[1]
    assert (first.outcome, first.detail) == (ItemOutcome.ERROR, "AttributeError")
    assert ("request", 1) not in harness.calls and ("run", 1) not in harness.calls
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED
    assert summary.interrupted is False and summary.needs_attention is True
    assert_accounting(summary, plan)


def test_a_blocked_classification_with_a_non_string_note_is_an_error_and_the_loop_continues():
    plan = make_plan(index=[1, 2])
    harness = Harness(reclassify={1: decision(1, Action.BLOCKED, Category.NEVER_QUEUED, note=5)})
    summary = harness.execute(plan)

    first = by_id(summary)[1]
    assert (first.outcome, first.detail) == (ItemOutcome.ERROR, "TypeError")
    assert summary.blocked == 0 and summary.failures == 1   # not misreported as BLOCKED
    assert ("request", 1) not in harness.calls and ("run", 1) not in harness.calls
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED
    assert_accounting(summary, plan)

def test_a_bad_chunk_count_is_dropped_not_trusted():
    class Result:
        outcome = RunOutcome.PUBLISHED
        chunk_count = "many"

    summary = Harness(run={1: Result()}).execute(make_plan(index=[1]))
    assert (summary.results[0].outcome, summary.results[0].chunk_count) == (ItemOutcome.INDEXED, None)


# ---------- races found by the re-read ----------

RECLASSIFY_CASES = [
    (None, ItemOutcome.CHANGED, DETAIL_NO_LONGER_SELECTED),
    (decision(1, Action.SKIP, Category.CURRENT), ItemOutcome.CHANGED, "current"),
    (decision(1, Action.SKIP, Category.LIVE_LEASE), ItemOutcome.CHANGED, "live_lease"),
    (decision(1, Action.REPORT, Category.FAILED, note="requires --include-failed"), ItemOutcome.CHANGED, "failed"),
    (decision(1, Action.REPORT, Category.NO_TEXT, note="no extracted text"), ItemOutcome.CHANGED, "no_text"),
    (decision(1, Action.BLOCKED, Category.NEVER_QUEUED, note="pdf_file_missing"), ItemOutcome.BLOCKED, "pdf_file_missing"),
    (decision(1, Action.BLOCKED, Category.NEVER_QUEUED), ItemOutcome.BLOCKED, "never_queued"),
]


@pytest.mark.parametrize("answer, outcome, detail", RECLASSIFY_CASES)
def test_a_changed_document_is_never_requested_or_run(answer, outcome, detail):
    plan = make_plan(index=[1, 2])
    harness = Harness(reclassify={1: answer})
    summary = harness.execute(plan)

    first = by_id(summary)[1]
    assert (first.outcome, first.detail) == (outcome, detail)
    assert ("request", 1) not in harness.calls and ("run", 1) not in harness.calls
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED
    assert_accounting(summary, plan)


@pytest.mark.parametrize("answer", [False, None, 0, "yes"])
def test_a_request_that_is_not_exactly_true_means_not_queued_and_no_run(answer):
    plan = make_plan(index=[1, 2])
    harness = Harness(request={1: answer})
    summary = harness.execute(plan)

    assert by_id(summary)[1].outcome is ItemOutcome.NOT_QUEUED
    assert ("run", 1) not in harness.calls
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED
    assert_accounting(summary, plan)


# ---------- exceptions ----------

@pytest.mark.parametrize("step", ["reclassify", "request", "run"])
def test_an_exception_is_an_error_with_the_class_name_only_and_the_loop_continues(step):
    plan = make_plan(index=[1, 2])
    harness = Harness(**{step: {1: RuntimeError(f"postgresql://u:{SECRET}@host/db")}})
    summary = harness.execute(plan)

    first = by_id(summary)[1]
    assert (first.outcome, first.detail) == (ItemOutcome.ERROR, "RuntimeError")
    assert SECRET not in repr(summary)
    assert by_id(summary)[2].outcome is ItemOutcome.INDEXED
    assert summary.failures == 1 and summary.needs_attention is True
    if step == "reclassify":
        assert ("request", 1) not in harness.calls
    if step == "request":
        assert ("run", 1) not in harness.calls
    assert_accounting(summary, plan)


def test_a_non_ascii_exception_name_is_made_ascii():
    error = type("Caf\u00e9Error", (Exception,), {})("x")
    summary = Harness(run={1: error}).execute(make_plan(index=[1]))
    assert summary.results[0].detail.isascii()


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        Harness(run={1: SystemExit(3)}).execute(make_plan(index=[1, 2]))


# ---------- interrupts ----------

def test_an_interrupt_before_any_write_leaves_that_document_not_attempted():
    plan = make_plan(index=[1, 2, 3, 4])
    harness = Harness(reclassify={3: KeyboardInterrupt()})
    summary = harness.execute(plan)

    assert summary.interrupted is True
    assert [r.outcome for r in summary.results] == [
        ItemOutcome.INDEXED, ItemOutcome.INDEXED, ItemOutcome.NOT_ATTEMPTED, ItemOutcome.NOT_ATTEMPTED,
    ]
    assert harness.calls[-1] == ("reclassify", 3)  # nothing ran after the interrupt
    assert_accounting(summary, plan)


@pytest.mark.parametrize("step", ["request", "run"])
def test_an_interrupt_after_the_write_may_have_started_is_interrupted(step):
    plan = make_plan(index=[1, 2, 3])
    harness = Harness(**{step: {2: KeyboardInterrupt()}})
    summary = harness.execute(plan)

    assert summary.interrupted is True
    assert [r.outcome for r in summary.results] == [
        ItemOutcome.INDEXED, ItemOutcome.INTERRUPTED, ItemOutcome.NOT_ATTEMPTED,
    ]
    assert harness.calls[-1] == (step, 2)
    assert_accounting(summary, plan)


def test_an_interrupt_in_the_progress_callback_keeps_the_finished_document():
    plan = make_plan(index=[1, 2, 3])

    def on_result(position, total, result):
        if position == 2:
            raise KeyboardInterrupt

    summary = Harness().execute(plan, on_result=on_result)

    assert summary.interrupted is True
    assert [r.outcome for r in summary.results] == [
        ItemOutcome.INDEXED, ItemOutcome.INDEXED, ItemOutcome.NOT_ATTEMPTED,
    ]
    assert_accounting(summary, plan)  # document 2 is counted once, not twice


# ---------- the progress callback and the clock ----------

def test_the_callback_sees_each_finished_document_in_order():
    seen = []
    plan = make_plan(index=[1, 2])
    Harness(run={2: IndexingRunResult(document_id=2, outcome=RunOutcome.STALE, attempt=1)}).execute(
        plan, on_result=lambda position, total, result: seen.append((position, total, result.document_id, result.outcome))
    )
    assert seen == [(1, 2, 1, ItemOutcome.INDEXED), (2, 2, 2, ItemOutcome.STALE)]


def test_a_failing_callback_does_not_stop_the_run_or_change_the_results():
    def on_result(position, total, result):
        raise OSError("broken pipe")

    plan = make_plan(index=[1, 2])
    summary = Harness().execute(plan, on_result=on_result)
    assert [r.outcome for r in summary.results] == [ItemOutcome.INDEXED] * 2
    assert_accounting(summary, plan)


def test_the_callback_is_not_called_for_interrupted_or_not_attempted_documents():
    seen = []
    Harness(run={2: KeyboardInterrupt()}).execute(
        make_plan(index=[1, 2, 3]), on_result=lambda position, total, result: seen.append(result.document_id)
    )
    assert seen == [1]


def test_timing_uses_the_injected_clock():
    ticks = iter(float(n) for n in range(100))
    summary = Harness().execute(make_plan(index=[1, 2]), clock=lambda: next(ticks))
    # start=0; doc 1: 1..2; doc 2: 3..4; end=5
    assert [r.elapsed_seconds for r in summary.results] == [1.0, 1.0]
    assert summary.elapsed_seconds == 5.0


# ---------- exact accounting under a mixed run ----------

def test_a_mixed_run_accounts_for_every_document():
    plan = make_plan(index=[1, 2, 3, 4, 5, 6], blocked=[10, 11], reported=[20], skipped=[30, 31, 32], limit=5)
    harness = Harness(
        request={2: False},
        run={3: IndexingRunResult(document_id=3, outcome=RunOutcome.FAILED, attempt=1,
                                  reason=FailureReason.UNEXPECTED, failure_outcome=FailureOutcome.FAILED),
             5: RuntimeError("x")},
        reclassify={4: None},
    )
    summary = harness.execute(plan)

    counts = summary.counts()
    assert counts[ItemOutcome.INDEXED] == 1 and counts[ItemOutcome.NOT_QUEUED] == 1
    assert counts[ItemOutcome.FAILED] == 1 and counts[ItemOutcome.CHANGED] == 1 and counts[ItemOutcome.ERROR] == 1
    assert (summary.planned, summary.plan_deferred, summary.plan_blocked) == (5, 1, 2)
    assert (summary.plan_reported, summary.plan_skipped, summary.examined) == (1, 3, 12)
    assert summary.failures == 2 and summary.changed == 2
    assert_accounting(summary, plan)


# ---------- what needs attention (the exit-code-1 condition) ----------

ATTENTION_CASES = [
    ("all indexed", dict(index=[1]), {}, False),
    ("failed", dict(index=[1]), dict(run={1: IndexingRunResult(
        document_id=1, outcome=RunOutcome.FAILED, attempt=1, reason=FailureReason.DATABASE,
        failure_outcome=FailureOutcome.FAILED)}), True),
    ("failure not recorded", dict(index=[1]), dict(run={1: IndexingRunResult(
        document_id=1, outcome=RunOutcome.FAILURE_NOT_RECORDED, attempt=1, reason=FailureReason.DATABASE)}), True),
    ("error", dict(index=[1]), dict(run={1: RuntimeError("x")}), True),
    ("blocked at run time", dict(index=[1]), dict(reclassify={1: decision(1, Action.BLOCKED, note="pdf_file_missing")}), True),
    ("blocked in the plan", dict(index=[1], blocked=[2]), {}, True),
    ("not queued", dict(index=[1]), dict(request={1: False}), False),
    ("changed", dict(index=[1]), dict(reclassify={1: None}), False),
    ("stale", dict(index=[1]), dict(run={1: IndexingRunResult(document_id=1, outcome=RunOutcome.STALE, attempt=1)}), False),
    ("gone", dict(index=[1]), dict(run={1: IndexingRunResult(document_id=1, outcome=RunOutcome.GONE, attempt=1)}), False),
    ("not claimed", dict(index=[1]), dict(run={1: IndexingRunResult(document_id=1, outcome=RunOutcome.NOT_CLAIMED)}), False),
    ("reported only", dict(index=[1], reported=[2]), {}, False),
    ("interrupted only", dict(index=[1, 2]), dict(run={1: KeyboardInterrupt()}), False),
]


@pytest.mark.parametrize("label, plan_kwargs, script, expected", ATTENTION_CASES, ids=[c[0] for c in ATTENTION_CASES])
def test_needs_attention_follows_the_approved_exit_code_rule(label, plan_kwargs, script, expected):
    summary = Harness(**script).execute(make_plan(**plan_kwargs))
    assert summary.needs_attention is expected


# ---------- the data classes validate themselves ----------

def test_a_summary_whose_numbers_do_not_add_up_is_rejected():
    results = (ItemResult(1, ItemOutcome.INDEXED),)
    good = dict(examined=1, plan_deferred=0, plan_blocked=0, plan_reported=0, plan_skipped=0,
                results=results, interrupted=False, elapsed_seconds=0.0)
    ExecutionSummary(**good)
    with pytest.raises(ValueError):
        ExecutionSummary(**{**good, "examined": 2})
    with pytest.raises(ValueError):
        ExecutionSummary(**{**good, "plan_deferred": -1, "examined": 0})
    with pytest.raises(ValueError):
        ExecutionSummary(**{**good, "results": results * 2, "examined": 2})  # a duplicate id
    with pytest.raises(TypeError):
        ExecutionSummary(**{**good, "results": list(results)})


def test_a_chunk_count_is_only_allowed_on_indexed_results():
    ItemResult(1, ItemOutcome.INDEXED, chunk_count=3)
    with pytest.raises(ValueError):
        ItemResult(1, ItemOutcome.FAILED, chunk_count=3)
    with pytest.raises(TypeError):
        ItemResult(1, ItemOutcome.FAILED, detail=5)


# ---------- the module itself ----------

def test_importing_the_module_loads_no_configuration_database_or_model():
    code = (
        "import sys\n"
        "import app.rag.backfill_execute\n"
        "names = ('app.core.config', 'app.core.database', 'app.core.storage', 'app.rag.indexing_runner',\n"
        "         'app.rag.indexing_service', 'app.rag.embedder', 'sqlalchemy', 'torch', 'transformers')\n"
        "bad = [m for m in names if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr