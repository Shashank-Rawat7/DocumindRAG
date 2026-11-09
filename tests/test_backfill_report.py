"""Pure tests for the backfill text report (C5.5 / CP5). No database, no files."""
import re

import pytest

from app.rag.backfill import DocumentState, LeaseState, SourceStatus, build_plan
from app.rag.backfill_report import DRY_RUN_FOOTER, DRY_RUN_MODE, NO_STATUS_LABEL, render_report
from app.rag.backfill_execute import ExecutionSummary, ItemOutcome, ItemResult
from app.rag.backfill_report import (
    DRY_RUN_FOOTER,
    DRY_RUN_MODE,
    EXECUTE_FOOTER,
    EXECUTE_MODE,
    NO_STATUS_LABEL,
    format_progress_line,
    render_execution_summary,
    render_report,
)

EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"

NEVER = dict(
    indexing_status=None, indexed_source_version=None, indexed_embedding_id=None, chunks_present=False
)


def make_state(**overrides) -> DocumentState:
    values = dict(
        document_id=1,
        processing_status="completed",
        indexing_status="indexed",
        indexing_error=None,
        lease=LeaseState.NONE,
        source_version=1,
        indexed_source_version=1,
        indexed_embedding_id=EMBEDDING_ID,
        has_text=True,
        chunks_present=True,
        filename="notes.txt",
        stored_name="abc.txt",
        content_type="text/plain",
        source=SourceStatus.NOT_PDF,
    )
    values.update(overrides)
    return DocumentState(**values)


def mixed_states() -> list[DocumentState]:
    return [
        make_state(document_id=1, **NEVER),                                       # INDEX
        make_state(document_id=2, **NEVER, source=SourceStatus.PDF_FILE_MISSING,  # BLOCKED
                   filename="secret-name.pdf", stored_name="secret-stored.pdf"),
        make_state(document_id=3, indexing_status="failed", indexing_error="boom"),  # REPORT
        make_state(document_id=4),                                                # SKIP (current)
        make_state(document_id=5, source_version=2, indexed_embedding_id="other@rev"),  # INDEX, stale
        make_state(document_id=6, **NEVER, has_text=False),                       # REPORT (no text)
        make_state(document_id=7, **NEVER),                                       # INDEX
    ]


def render(states=None, *, limit=None, include_failed=False, not_completed=None, target="localhost:5434/documind_db as u",
           verified="documind_db", mode=DRY_RUN_MODE, **kwargs) -> str:
    plan = build_plan(
        mixed_states() if states is None else states,
        configured_embedding_id=EMBEDDING_ID,
        include_failed=include_failed,
        limit=limit,
    )
    return render_report(
        plan,
        mode=mode,
        target=target,
        database_verified=verified,
        include_failed=include_failed,
        not_completed={} if not_completed is None else not_completed,
        **kwargs,
    )


def section(text: str, title: str) -> list[str]:
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    body = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        body.append(line.strip())
    return body


def value(text: str, label: str) -> str:
    match = re.search(rf"^\s+{re.escape(label)}\s*:\s*(.+)$", text, re.M)
    assert match, f"no summary row {label!r}"
    return match.group(1).strip()


def has_doc(lines: list[str], document_id: int) -> bool:
    return any(re.match(rf"^doc {document_id}\s", line) for line in lines)


# ---------- header ----------

def test_the_header_shows_mode_target_database_and_options():
    text = render(limit=2, include_failed=True)
    assert text.split("\n")[0] == f"DocuMind indexing backfill - {DRY_RUN_MODE}"
    assert "Target   : localhost:5434/documind_db as u" in text
    assert "Database : verified as 'documind_db' (live check)" in text
    assert "Options  : include-failed=yes, limit=2" in text


def test_defaults_show_no_limit_unverified_database_and_unknown_target():
    text = render(verified=None, target=None)
    assert "Options  : include-failed=no, limit=none" in text
    assert "Database : not verified (no --database given)" in text
    assert "Target   : unknown" in text


# ---------- summary ----------

def test_the_summary_counts_match_the_plan():
    text = render()
    assert value(text, "Completed documents examined") == "7"
    assert value(text, "Will index") == "3"
    assert value(text, "Deferred by --limit") == "0"
    assert value(text, "Blocked") == "1"
    assert value(text, "Reported (needs attention)") == "2"
    assert value(text, "Skipped (nothing to do)") == "1"


@pytest.mark.parametrize("limit", [None, 1, 2, 3, 10])
@pytest.mark.parametrize("include_failed", [False, True])
def test_the_summary_always_adds_up(limit, include_failed):
    text = render(limit=limit, include_failed=include_failed)
    parts = [
        int(value(text, "Will index")),
        int(value(text, "Deferred by --limit")),
        int(value(text, "Blocked")),
        int(value(text, "Reported (needs attention)")),
        int(value(text, "Skipped (nothing to do)")),
    ]
    assert sum(parts) == int(value(text, "Completed documents examined"))


def test_the_category_row_lists_only_non_zero_categories():
    text = render()
    assert value(text, "By category") == "never_queued 3, failed 1, indexed_stale 1, current 1, no_text 1"


def test_an_empty_plan_reports_zeros_and_none():
    text = render(states=[])
    assert value(text, "Completed documents examined") == "0"
    assert value(text, "By category") == "none"
    for title in ("Will index", "Blocked", "Reported"):
        assert section(text, title) == ["none"]


# ---------- sections ----------

def test_will_index_lists_candidates_in_id_order_with_stale_reasons():
    lines = section(render(), "Will index")
    assert [bool(re.match(r"^doc (\d+)", line)) for line in lines] == [True, True, True]
    ids = [int(re.match(r"^doc (\d+)", line).group(1)) for line in lines]
    assert ids == [1, 5, 7]
    assert any("indexed_stale (source_version, embedding_id)" in line for line in lines)


def test_blocked_and_reported_show_their_notes():
    text = render()
    blocked = section(text, "Blocked")
    reported = section(text, "Reported")
    assert re.match(r"^doc 2\s+never_queued\s+\[pdf_file_missing\]$", blocked[0])
    assert any(re.match(r"^doc 3\s+failed\s+\[requires --include-failed\]$", line) for line in reported)
    assert any(re.match(r"^doc 6\s+no_text\s+\[no extracted text\]$", line) for line in reported)


def test_skipped_documents_are_counted_but_never_listed():
    text = render()
    assert not has_doc(text.split("\n"), 4)
    assert not re.search(r"^\s*doc 4\s", text, re.M)
    assert "current" in value(text, "By category")


def test_include_failed_moves_the_failed_document_into_will_index():
    text = render(include_failed=True)
    assert has_doc(section(text, "Will index"), 3)
    assert not has_doc(section(text, "Reported"), 3)


# ---------- the limit ----------

def test_the_limit_defers_only_index_candidates_beyond_n():
    text = render(limit=2)
    assert [int(re.match(r"^doc (\d+)", l).group(1)) for l in section(text, "Will index")] == [1, 5]
    deferred = section(text, "Deferred by --limit 2")
    assert [int(re.match(r"^doc (\d+)", l).group(1)) for l in deferred] == [7]
    assert value(text, "Deferred by --limit") == "1"
    # Blocked, reported and skipped items are never deferred.
    assert has_doc(section(text, "Blocked"), 2)
    assert has_doc(section(text, "Reported"), 3) and has_doc(section(text, "Reported"), 6)


def test_without_a_limit_the_deferred_section_is_empty():
    assert section(render(), "Deferred by --limit") == ["none"]


# ---------- extraction not completed ----------

def test_the_not_completed_breakdown_is_sorted_with_no_status_last():
    text = render(not_completed={"pending": 1, None: 16, "failed": 7})
    assert value(text, "Not examined (extraction not completed)") == f"24 (failed 7, pending 1, {NO_STATUS_LABEL} 16)"


def test_zero_not_completed_documents_show_a_plain_zero():
    assert value(render(not_completed={}), "Not examined (extraction not completed)") == "0"
    assert value(render(not_completed={"failed": 0}), "Not examined (extraction not completed)") == "0"


# ---------- stability, privacy, footer ----------

def test_the_output_is_deterministic_and_independent_of_input_order():
    states = mixed_states()
    assert render(states) == render(list(reversed(states)))
    assert render(states) == render(states)


def test_no_filenames_or_stored_names_appear():
    text = render()
    assert "secret-name" not in text and "secret-stored" not in text
    assert "notes.txt" not in text and "abc.txt" not in text


def test_the_output_is_plain_ascii():
    assert render(not_completed={None: 3}).isascii()


def test_the_footer_defaults_to_the_dry_run_message_and_can_be_replaced():
    assert render().endswith(DRY_RUN_FOOTER)
    assert render(footer="custom footer").endswith("custom footer")

# ---------- execution: progress lines ----------

def item(document_id, outcome, detail=None, chunks=None, seconds=None):
    return ItemResult(document_id, outcome, detail, chunks, seconds)


@pytest.mark.parametrize(
    "result, pattern",
    [
        (item(7, ItemOutcome.INDEXED, chunks=12, seconds=41.23), r"\[3/13\] doc 7\s+indexed \(12 chunks, 41\.2s\)"),
        (item(7, ItemOutcome.INDEXED), r"\[3/13\] doc 7\s+indexed"),
        (item(7, ItemOutcome.FAILED, "EMBEDDER_UNAVAILABLE"), r"\[3/13\] doc 7\s+failed \(EMBEDDER_UNAVAILABLE\)"),
        (item(7, ItemOutcome.FAILURE_NOT_RECORDED, "DATABASE"), r"\[3/13\] doc 7\s+failure not recorded \(DATABASE\)"),
        (item(7, ItemOutcome.ERROR, "RuntimeError"), r"\[3/13\] doc 7\s+error \(RuntimeError\)"),
        (item(7, ItemOutcome.BLOCKED, "pdf_file_missing"), r"\[3/13\] doc 7\s+blocked \(pdf_file_missing\)"),
        (item(7, ItemOutcome.NOT_QUEUED), r"\[3/13\] doc 7\s+not queued"),
        (item(7, ItemOutcome.STALE), r"\[3/13\] doc 7\s+stale"),
    ],
)
def test_progress_lines(result, pattern):
    line = format_progress_line(3, 13, result)
    assert re.fullmatch(pattern, line), line
    assert "\n" not in line and line.isascii()


# ---------- execution: the closing summary ----------

def make_summary(results=(), *, deferred=0, blocked=0, reported=0, skipped=0, interrupted=False, elapsed=12.34):
    results = tuple(results)
    return ExecutionSummary(
        examined=len(results) + deferred + blocked + reported + skipped,
        plan_deferred=deferred, plan_blocked=blocked, plan_reported=reported, plan_skipped=skipped,
        results=results, interrupted=interrupted, elapsed_seconds=elapsed,
    )


def mixed_summary():
    return make_summary(
        [
            item(1, ItemOutcome.INDEXED, chunks=4), item(2, ItemOutcome.INDEXED, chunks=2),
            item(3, ItemOutcome.FAILED, "EMBEDDER_UNAVAILABLE"), item(4, ItemOutcome.ERROR, "RuntimeError"),
            item(5, ItemOutcome.BLOCKED, "pdf_file_missing"), item(6, ItemOutcome.CHANGED, "current"),
            item(7, ItemOutcome.NOT_QUEUED), item(8, ItemOutcome.STALE),
            item(9, ItemOutcome.INTERRUPTED), item(10, ItemOutcome.NOT_ATTEMPTED),
        ],
        deferred=2, blocked=1, reported=3, skipped=4, interrupted=True,
    )


def test_the_execution_summary_rows():
    text = render_execution_summary(mixed_summary())
    expected = {
        "Completed documents examined": "20", "Planned for indexing": "10", "Indexed": "2", "Failed": "2",
        "Blocked (found at run time)": "1", "Changed since the plan": "3", "Interrupted in progress": "1",
        "Not attempted": "1", "Deferred by --limit": "2", "Blocked (in the plan)": "1",
        "Reported (needs attention)": "3", "Skipped (nothing to do)": "4", "Elapsed": "12.3s",
    }
    for label, shown in expected.items():
        assert value(text, label) == shown, label


def test_the_execution_summary_adds_up():
    text = render_execution_summary(mixed_summary())
    run_parts = ["Indexed", "Failed", "Blocked (found at run time)", "Changed since the plan",
                 "Interrupted in progress", "Not attempted"]
    assert sum(int(value(text, label)) for label in run_parts) == int(value(text, "Planned for indexing"))
    plan_parts = ["Planned for indexing", "Deferred by --limit", "Blocked (in the plan)",
                  "Reported (needs attention)", "Skipped (nothing to do)"]
    assert sum(int(value(text, label)) for label in plan_parts) == int(value(text, "Completed documents examined"))


def test_the_execution_sections_list_the_right_documents():
    text = render_execution_summary(mixed_summary())
    failures = section(text, "Failures (2)")
    assert re.match(r"^doc 3\s+failed \(EMBEDDER_UNAVAILABLE\)$", failures[0])
    assert re.match(r"^doc 4\s+error \(RuntimeError\)$", failures[1])
    assert [bool(re.match(r"^doc 5\s+blocked \(pdf_file_missing\)$", l)) for l in section(text, "Blocked at run time (1)")] == [True]
    changed = section(text, "Changed since the plan (3)")
    assert [int(re.match(r"^doc (\d+)", l).group(1)) for l in changed] == [6, 7, 8]
    assert [int(re.match(r"^doc (\d+)", l).group(1)) for l in section(text, "Not completed (2)")] == [9, 10]


def test_empty_execution_sections_say_none():
    text = render_execution_summary(make_summary([item(1, ItemOutcome.INDEXED, chunks=1)]))
    for title in ("Failures (0)", "Blocked at run time (0)", "Changed since the plan (0)", "Not completed (0)"):
        assert section(text, title) == ["none"]


def test_the_result_line_for_each_state():
    clean = render_execution_summary(make_summary([item(1, ItemOutcome.INDEXED, chunks=1)]))
    failed = render_execution_summary(make_summary([item(1, ItemOutcome.FAILED, "DATABASE")]))
    plan_blocked = render_execution_summary(make_summary([item(1, ItemOutcome.INDEXED, chunks=1)], blocked=1))
    interrupted = render_execution_summary(mixed_summary())
    assert "Result: finished without failures" in clean
    assert "Result: finished with failures or blocked documents" in failed
    assert "Result: finished with failures or blocked documents" in plan_blocked
    assert "Result: interrupted before completion" in interrupted


def test_the_lease_note_appears_only_when_a_document_was_interrupted_in_progress():
    assert "stays locked until its lease expires" in render_execution_summary(mixed_summary())
    not_started = make_summary([item(1, ItemOutcome.NOT_ATTEMPTED)], interrupted=True)
    assert "lease" not in render_execution_summary(not_started)


def test_the_execution_summary_is_ascii_deterministic_and_has_no_filenames():
    text = render_execution_summary(mixed_summary())
    assert text.isascii() and text == render_execution_summary(mixed_summary())
    assert "notes.txt" not in text and ".pdf" not in text


def test_the_execute_header_and_footer_work_with_the_existing_report():
    text = render(mode=EXECUTE_MODE, footer=EXECUTE_FOOTER)
    assert text.split("\n")[0] == f"DocuMind indexing backfill - {EXECUTE_MODE}"
    assert text.endswith(EXECUTE_FOOTER)
    assert DRY_RUN_FOOTER not in text
