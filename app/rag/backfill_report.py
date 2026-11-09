"""Text report for the indexing backfill (Component 12 / C5.5, CP5).

Pure: it turns an already-built plan into text. No I/O, no database, no clock. Only
document ids, categories, stale reasons and short notes are shown (never filenames, text
or stored names), and the target is passed in as a ready-made string that cannot carry a
password. Output is plain ASCII.
"""
from __future__ import annotations

from collections.abc import Mapping

from app.rag.backfill import Action, BackfillPlan, Category, PlannedItem
from app.rag.backfill_execute import ExecutionSummary, ItemOutcome, ItemResult

NO_STATUS_LABEL = "no status"
DRY_RUN_MODE = "DRY RUN (nothing will be changed)"
DRY_RUN_FOOTER = "Dry run: nothing was changed."
EXECUTE_MODE = "EXECUTE (documents will be indexed)"
EXECUTE_FOOTER = "Indexing starts now. Progress follows, one line per document."


def _item_line(item: PlannedItem) -> str:
    classification = item.classification
    line = f"  doc {classification.document_id:<6} {classification.category.value}"
    if classification.stale_reasons:
        line += " (" + ", ".join(reason.value for reason in classification.stale_reasons) + ")"
    if classification.note:
        line += f"  [{classification.note}]"
    return line.rstrip()


def _section(title: str, items: list[PlannedItem]) -> list[str]:
    lines = [f"{title} ({len(items)})"]
    if items:
        lines.extend(_item_line(item) for item in items)
    else:
        lines.append("  none")
    return lines


def _not_completed_text(not_completed: Mapping[str | None, int]) -> str:
    entries = sorted(
        ((status, count) for status, count in not_completed.items() if count),
        key=lambda entry: (entry[0] is None, entry[0] or ""),  # named statuses first, "no status" last
    )
    if not entries:
        return "0"
    total = sum(count for _, count in entries)
    detail = ", ".join(
        f"{NO_STATUS_LABEL if status is None else status} {count}" for status, count in entries
    )
    return f"{total} ({detail})"


def _category_text(counts: Mapping[Category, int]) -> str:
    parts = [f"{category.value} {count}" for category, count in counts.items() if count]
    return ", ".join(parts) if parts else "none"


def render_report(
    plan: BackfillPlan,
    *,
    mode: str,
    target: str | None,
    database_verified: str | None,
    include_failed: bool,
    not_completed: Mapping[str | None, int],
    footer: str = DRY_RUN_FOOTER,
) -> str:
    """Render the plan. ``database_verified`` is the name confirmed by the live check, or None."""
    by_action = plan.counts_by_action()
    will_index = list(plan.to_run())
    deferred = list(plan.deferred())
    blocked = [item for item in plan.items if item.classification.action is Action.BLOCKED]
    reported = [item for item in plan.items if item.classification.action is Action.REPORT]

    limit_text = "none" if plan.limit is None else str(plan.limit)
    verified_text = (
        f"verified as '{database_verified}' (live check)"
        if database_verified
        else "not verified (no --database given)"
    )

    rows = [
        ("Completed documents examined", str(len(plan.items))),
        ("Will index", str(len(will_index))),
        ("Deferred by --limit", str(len(deferred))),
        ("Blocked", str(len(blocked))),
        ("Reported (needs attention)", str(len(reported))),
        ("Skipped (nothing to do)", str(by_action[Action.SKIP])),
        ("Not examined (extraction not completed)", _not_completed_text(not_completed)),
        ("By category", _category_text(plan.counts_by_category())),
    ]
    width = max(len(label) for label, _ in rows)

    deferred_title = "Deferred by --limit" if plan.limit is None else f"Deferred by --limit {plan.limit}"

    lines = [
        f"DocuMind indexing backfill - {mode}",
        f"Target   : {target if target else 'unknown'}",
        f"Database : {verified_text}",
        f"Options  : include-failed={'yes' if include_failed else 'no'}, limit={limit_text}",
        "",
        "Summary",
    ]
    lines.extend(f"  {label:<{width}} : {value}" for label, value in rows)
    lines.append("")
    lines.extend(_section("Will index", will_index))
    lines.append("")
    lines.extend(_section(deferred_title, deferred))
    lines.append("")
    lines.extend(_section("Blocked", blocked))
    lines.append("")
    lines.extend(_section("Reported", reported))
    lines.append("")
    lines.append(footer)
    return "\n".join(lines)

def _result_line(result: ItemResult) -> str:
    line = f"  doc {result.document_id:<6} {result.outcome.value.replace('_', ' ')}"
    if result.detail:
        line += f" ({result.detail})"
    return line


def _result_section(title: str, results: list[ItemResult]) -> list[str]:
    lines = [f"{title} ({len(results)})"]
    lines.extend(_result_line(result) for result in results) if results else lines.append("  none")
    return lines


def format_progress_line(position: int, total: int, result: ItemResult) -> str:
    """One line of progress for a finished document, for example ``[3/13] doc 7 indexed (12 chunks, 4.1s)``."""
    extras: list[str] = []
    if result.outcome is ItemOutcome.INDEXED:
        if result.chunk_count is not None:
            extras.append(f"{result.chunk_count} chunks")
        if result.elapsed_seconds is not None:
            extras.append(f"{result.elapsed_seconds:.1f}s")
    elif result.detail:
        extras.append(result.detail)
    line = f"[{position}/{total}] doc {result.document_id:<6} {result.outcome.value.replace('_', ' ')}"
    if extras:
        line += " (" + ", ".join(extras) + ")"
    return line


def render_execution_summary(summary: ExecutionSummary) -> str:
    """The closing report of an execution run. Ids, outcomes and fixed vocabulary only."""
    counts = summary.counts()
    rows = [
        ("Completed documents examined", str(summary.examined)),
        ("Planned for indexing", str(summary.planned)),
        ("Indexed", str(summary.indexed)),
        ("Failed", str(summary.failures)),
        ("Blocked (found at run time)", str(summary.blocked)),
        ("Changed since the plan", str(summary.changed)),
        ("Interrupted in progress", str(counts[ItemOutcome.INTERRUPTED])),
        ("Not attempted", str(counts[ItemOutcome.NOT_ATTEMPTED])),
        ("Deferred by --limit", str(summary.plan_deferred)),
        ("Blocked (in the plan)", str(summary.plan_blocked)),
        ("Reported (needs attention)", str(summary.plan_reported)),
        ("Skipped (nothing to do)", str(summary.plan_skipped)),
        ("Elapsed", f"{summary.elapsed_seconds:.1f}s"),
    ]
    width = max(len(label) for label, _ in rows)

    failures = [r for r in summary.results if r.outcome in (
        ItemOutcome.FAILED, ItemOutcome.FAILURE_NOT_RECORDED, ItemOutcome.ERROR)]
    blocked = [r for r in summary.results if r.outcome is ItemOutcome.BLOCKED]
    changed = [r for r in summary.results if r.outcome in (
        ItemOutcome.CHANGED, ItemOutcome.NOT_QUEUED, ItemOutcome.NOT_CLAIMED,
        ItemOutcome.STALE, ItemOutcome.GONE)]
    incomplete = [r for r in summary.results if r.outcome in (
        ItemOutcome.INTERRUPTED, ItemOutcome.NOT_ATTEMPTED)]

    if summary.interrupted:
        verdict = "Result: interrupted before completion"
    elif summary.needs_attention:
        verdict = "Result: finished with failures or blocked documents"
    else:
        verdict = "Result: finished without failures"

    lines = ["Execution summary"]
    lines.extend(f"  {label:<{width}} : {value}" for label, value in rows)
    lines.append("")
    lines.extend(_result_section("Failures", failures))
    lines.append("")
    lines.extend(_result_section("Blocked at run time", blocked))
    lines.append("")
    lines.extend(_result_section("Changed since the plan", changed))
    lines.append("")
    lines.extend(_result_section("Not completed", incomplete))
    lines.append("")
    lines.append(verdict)
    if counts[ItemOutcome.INTERRUPTED]:
        lines.append(
            "Note: a document that was in progress stays locked until its lease expires; "
            "a rerun skips it until then."
        )
    return "\n".join(lines)