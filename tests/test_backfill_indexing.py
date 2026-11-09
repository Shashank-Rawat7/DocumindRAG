"""Pure unit tests for the backfill classifier and planner (C5.5 / CP1).

No database rows, SQL, files, model or CLI: every test builds DocumentState values in
memory. (The autouse fixtures in conftest.py still run around each test.)
"""
import dataclasses
import inspect

import pytest

from app.rag import backfill
from app.rag.backfill import (
    Action,
    BackfillPlan,
    Category,
    Classification,
    DocumentState,
    LeaseState,
    PlannedItem,
    SourceStatus,
    StaleReason,
    build_plan,
    classify,
    is_pdf_like,
)

EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"
OTHER_EMBEDDING_ID = "someone/else@rev2"


def make_state(**overrides) -> DocumentState:
    """A fully current, valid indexed document; override fields to build any other state."""
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
        stored_name="abc123.txt",
        content_type="text/plain",
        source=SourceStatus.NOT_PDF,
    )
    values.update(overrides)
    return DocumentState(**values)


def run(state: DocumentState, include_failed: bool = False) -> Classification:
    return classify(state, configured_embedding_id=EMBEDDING_ID, include_failed=include_failed)


def assert_result(result, category, action, *, reasons=(), note=None):
    assert result.category is category
    assert result.action is action
    assert result.stale_reasons == tuple(reasons)
    assert result.note == note


# Never-indexed leftovers must not matter for these states.
NEVER_INDEXED = dict(
    indexing_status=None, indexed_source_version=None, indexed_embedding_id=None, chunks_present=False
)

ALL_INDEXING_STATUSES = [None, "pending", "indexing", "indexed", "failed"]


# ---------- DocumentState is strict ----------

def test_every_documentstate_field_is_required():
    for field in dataclasses.fields(DocumentState):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name


@pytest.mark.parametrize("missing", ["lease", "source"])
def test_omitting_lease_or_source_is_a_type_error(missing):
    values = {f.name: None for f in dataclasses.fields(DocumentState)}
    values.pop(missing)
    with pytest.raises(TypeError):
        DocumentState(**values)


def test_states_and_results_are_immutable():
    state = make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.lease = LeaseState.LIVE
    with pytest.raises(dataclasses.FrozenInstanceError):
        run(state).action = Action.INDEX


# ---------- extraction state ----------

@pytest.mark.parametrize("indexing_status", ALL_INDEXING_STATUSES)
@pytest.mark.parametrize("include_failed", [False, True])
def test_null_processing_status_is_skipped(indexing_status, include_failed):
    result = run(make_state(processing_status=None, indexing_status=indexing_status), include_failed)
    assert_result(result, Category.NO_PROCESSING_STATUS, Action.SKIP)


@pytest.mark.parametrize("processing_status", ["pending", "processing", "failed"])
@pytest.mark.parametrize("indexing_status", ALL_INDEXING_STATUSES)
@pytest.mark.parametrize("include_failed", [False, True])
def test_unfinished_or_failed_extraction_is_skipped_even_with_include_failed(
    processing_status, indexing_status, include_failed
):
    result = run(
        make_state(processing_status=processing_status, indexing_status=indexing_status),
        include_failed,
    )
    assert_result(result, Category.EXTRACTION_INCOMPLETE, Action.SKIP)


def test_unknown_processing_status_is_reported():
    assert_result(run(make_state(processing_status="weird")), Category.UNRECOGNIZED, Action.REPORT)


def test_unknown_indexing_status_is_reported():
    assert_result(run(make_state(indexing_status="weird")), Category.UNRECOGNIZED, Action.REPORT)


# ---------- completed, not yet indexed ----------

def test_never_queued_is_indexed():
    assert_result(run(make_state(**NEVER_INDEXED)), Category.NEVER_QUEUED, Action.INDEX)


def test_pending_is_indexed():
    state = make_state(**{**NEVER_INDEXED, "indexing_status": "pending"})
    assert_result(run(state), Category.PENDING, Action.INDEX)


@pytest.mark.parametrize("include_failed", [False, True])
def test_a_live_lease_is_skipped(include_failed):
    state = make_state(indexing_status="indexing", lease=LeaseState.LIVE)
    assert_result(run(state, include_failed), Category.LIVE_LEASE, Action.SKIP)


def test_an_expired_lease_is_indexed():
    state = make_state(indexing_status="indexing", lease=LeaseState.EXPIRED)
    assert_result(run(state), Category.ABANDONED_LEASE, Action.INDEX)


@pytest.mark.parametrize("include_failed", [False, True])
def test_an_indexing_row_without_a_lease_is_reported_never_indexed(include_failed):
    state = make_state(indexing_status="indexing", lease=LeaseState.NONE)
    assert_result(run(state, include_failed), Category.LEASE_AMBIGUOUS, Action.REPORT)


# ---------- failed ----------

def test_failed_is_reported_without_the_flag():
    result = run(make_state(indexing_status="failed", indexing_error="boom"))
    assert_result(result, Category.FAILED, Action.REPORT, note="requires --include-failed")


def test_failed_is_indexed_with_the_flag():
    result = run(make_state(indexing_status="failed", indexing_error="boom"), include_failed=True)
    assert_result(result, Category.FAILED, Action.INDEX)


# ---------- indexed ----------

def test_a_fully_current_document_is_skipped():
    assert_result(run(make_state()), Category.CURRENT, Action.SKIP)


@pytest.mark.parametrize("include_failed", [False, True])
def test_current_is_skipped_regardless_of_the_flag(include_failed):
    assert_result(run(make_state(), include_failed), Category.CURRENT, Action.SKIP)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({"indexed_source_version": None}, (StaleReason.SOURCE_VERSION,), id="indexed-version-null"),
        pytest.param({"indexed_source_version": 0}, (StaleReason.SOURCE_VERSION,), id="indexed-version-older"),
        pytest.param({"source_version": 2}, (StaleReason.SOURCE_VERSION,), id="source-moved-on"),
        pytest.param({"indexed_source_version": 3}, (StaleReason.SOURCE_VERSION,), id="indexed-version-newer-defensive"),
        pytest.param({"chunks_present": False}, (StaleReason.NO_CHUNKS,), id="no-chunks"),
        pytest.param({"indexed_embedding_id": OTHER_EMBEDDING_ID}, (StaleReason.EMBEDDING_ID,), id="other-embedding"),
        pytest.param({"indexed_embedding_id": None}, (StaleReason.EMBEDDING_ID,), id="embedding-null"),
        pytest.param(
            {"source_version": 2, "chunks_present": False, "indexed_embedding_id": OTHER_EMBEDDING_ID},
            (StaleReason.SOURCE_VERSION, StaleReason.NO_CHUNKS, StaleReason.EMBEDDING_ID),
            id="all-three-in-fixed-order",
        ),
    ],
)
def test_each_stale_reason_is_indexed_and_reported(overrides, expected):
    assert_result(run(make_state(**overrides)), Category.INDEXED_STALE, Action.INDEX, reasons=expected)


def test_staleness_is_found_without_the_flag_even_when_an_error_is_retained():
    state = make_state(source_version=2, indexing_error="boom")
    assert_result(
        run(state), Category.INDEXED_STALE, Action.INDEX, reasons=(StaleReason.SOURCE_VERSION,)
    )


def test_a_valid_indexed_document_with_an_error_is_reported_without_the_flag():
    result = run(make_state(indexing_error="boom"))
    assert_result(result, Category.INDEXED_WITH_ERROR, Action.REPORT, note="requires --include-failed")


def test_a_valid_indexed_document_with_an_error_is_indexed_with_the_flag():
    result = run(make_state(indexing_error="boom"), include_failed=True)
    assert_result(result, Category.INDEXED_WITH_ERROR, Action.INDEX)


def test_an_empty_error_string_still_counts_as_an_error():
    # The rule is "error is not NULL", the same test the API uses.
    result = run(make_state(indexing_error=""))
    assert_result(result, Category.INDEXED_WITH_ERROR, Action.REPORT, note="requires --include-failed")


# ---------- gates ----------

# (id, overrides, include_failed, category) for every state that would otherwise be INDEX.
INDEX_CASES = [
    pytest.param(NEVER_INDEXED, False, Category.NEVER_QUEUED, id="never-queued"),
    pytest.param({**NEVER_INDEXED, "indexing_status": "pending"}, False, Category.PENDING, id="pending"),
    pytest.param(
        {"indexing_status": "indexing", "lease": LeaseState.EXPIRED}, False, Category.ABANDONED_LEASE, id="abandoned"
    ),
    pytest.param({"indexing_status": "failed", "indexing_error": "boom"}, True, Category.FAILED, id="failed-with-flag"),
    pytest.param({"source_version": 2}, False, Category.INDEXED_STALE, id="stale"),
    pytest.param({"indexing_error": "boom"}, True, Category.INDEXED_WITH_ERROR, id="error-with-flag"),
]


@pytest.mark.parametrize("overrides, include_failed, category", INDEX_CASES)
def test_every_index_candidate_with_text_and_a_usable_source_is_indexed(overrides, include_failed, category):
    result = run(make_state(**overrides), include_failed)
    assert result.action is Action.INDEX
    assert result.category is category


@pytest.mark.parametrize("overrides, include_failed, category", INDEX_CASES)
@pytest.mark.parametrize("has_text", [False])
def test_no_text_is_reported_and_never_indexed(overrides, include_failed, category, has_text):
    result = run(make_state(**overrides, has_text=has_text), include_failed)
    assert result.category is Category.NO_TEXT
    assert result.action is Action.REPORT
    assert result.note == "no extracted text"


def test_no_text_keeps_the_stale_reasons():
    result = run(make_state(source_version=2, has_text=False))
    assert result.category is Category.NO_TEXT
    assert result.stale_reasons == (StaleReason.SOURCE_VERSION,)


@pytest.mark.parametrize("overrides, include_failed, category", INDEX_CASES)
@pytest.mark.parametrize(
    "source",
    [SourceStatus.PDF_FILE_MISSING, SourceStatus.PDF_NO_STORED_NAME, SourceStatus.PDF_UNCHECKED],
)
def test_an_unusable_source_blocks_every_index_candidate(overrides, include_failed, category, source):
    result = run(make_state(**overrides, source=source), include_failed)
    assert result.category is category          # the state category is kept
    assert result.action is Action.BLOCKED
    assert result.note == source.value


def test_blocking_keeps_the_stale_reasons():
    result = run(make_state(source_version=2, source=SourceStatus.PDF_FILE_MISSING))
    assert_result(
        result,
        Category.INDEXED_STALE,
        Action.BLOCKED,
        reasons=(StaleReason.SOURCE_VERSION,),
        note="pdf_file_missing",
    )


@pytest.mark.parametrize("source", [SourceStatus.NOT_PDF, SourceStatus.PDF_PRESENT])
def test_a_usable_source_does_not_block(source):
    assert run(make_state(**NEVER_INDEXED, source=source)).action is Action.INDEX


def test_no_text_wins_over_a_blocked_source():
    result = run(make_state(**NEVER_INDEXED, has_text=False, source=SourceStatus.PDF_FILE_MISSING))
    assert result.category is Category.NO_TEXT
    assert result.action is Action.REPORT


@pytest.mark.parametrize(
    "overrides, include_failed, category, action",
    [
        pytest.param({}, False, Category.CURRENT, Action.SKIP, id="current"),
        pytest.param(
            {"indexing_status": "indexing", "lease": LeaseState.LIVE}, False, Category.LIVE_LEASE, Action.SKIP, id="live"
        ),
        pytest.param(
            {"indexing_status": "failed", "indexing_error": "boom"},
            False,
            Category.FAILED,
            Action.REPORT,
            id="failed-without-flag",
        ),
        pytest.param(
            {"indexing_error": "boom"}, False, Category.INDEXED_WITH_ERROR, Action.REPORT, id="error-without-flag"
        ),
        pytest.param(
            {"processing_status": "failed"}, True, Category.EXTRACTION_INCOMPLETE, Action.SKIP, id="extraction-failed"
        ),
        pytest.param({"processing_status": None}, False, Category.NO_PROCESSING_STATUS, Action.SKIP, id="no-status"),
    ],
)
def test_gates_do_not_touch_documents_that_were_not_going_to_be_indexed(
    overrides, include_failed, category, action
):
    state = make_state(**overrides, has_text=False, source=SourceStatus.PDF_FILE_MISSING)
    result = run(state, include_failed)
    assert result.category is category
    assert result.action is action
    assert result.note != "no extracted text"
    assert result.note != "pdf_file_missing"


def test_a_completed_pdf_without_a_stored_name_is_blocked_not_indexed():
    state = make_state(
        **NEVER_INDEXED,
        filename="report.pdf",
        stored_name=None,
        content_type="application/pdf",
        source=SourceStatus.PDF_NO_STORED_NAME,
    )
    assert_result(run(state), Category.NEVER_QUEUED, Action.BLOCKED, note="pdf_no_stored_name")


def test_a_non_pdf_without_a_stored_name_is_indexed():
    state = make_state(**NEVER_INDEXED, stored_name=None, source=SourceStatus.NOT_PDF)
    assert_result(run(state), Category.NEVER_QUEUED, Action.INDEX)


# ---------- classify input validation and determinism ----------

@pytest.mark.parametrize("bad", ["", None, 5])
def test_the_configured_embedding_id_must_be_a_non_empty_string(bad):
    with pytest.raises(ValueError):
        classify(make_state(), configured_embedding_id=bad)


def test_classify_is_deterministic():
    state = make_state(source_version=2)
    assert run(state) == run(state)


# ---------- is_pdf_like ----------

@pytest.mark.parametrize(
    "filename, stored_name, content_type, expected",
    [
        pytest.param("a.pdf", None, None, True, id="filename-suffix"),
        pytest.param("A.PDF", None, None, True, id="filename-uppercase"),
        pytest.param(None, "x.pdf", None, True, id="stored-suffix"),
        pytest.param(None, "x.PDF", None, True, id="stored-uppercase"),
        pytest.param(None, None, "application/pdf", True, id="mime"),
        pytest.param(None, None, "Application/PDF; charset=binary", True, id="mime-with-parameters"),
        pytest.param("report.docx", None, "application/pdf", True, id="any-signal-is-enough"),
        pytest.param("a.txt", "b.txt", "text/plain", False, id="plain-text"),
        pytest.param(None, None, None, False, id="nothing"),
        pytest.param("", "", "", False, id="empty-strings"),
        pytest.param("pdf", None, None, False, id="no-dot"),
        pytest.param("a.pdf.txt", None, None, False, id="pdf-not-last-suffix"),
        pytest.param(".pdf", None, None, False, id="hidden-file-has-no-suffix"),
        pytest.param("notes.md", None, "text/markdown", False, id="markdown"),
    ],
)
def test_is_pdf_like(filename, stored_name, content_type, expected):
    assert is_pdf_like(filename, stored_name, content_type) is expected


# ---------- build_plan ----------

def plan_of(states, **kwargs) -> BackfillPlan:
    return build_plan(states, configured_embedding_id=EMBEDDING_ID, **kwargs)


def index_candidate(document_id: int) -> DocumentState:
    return make_state(document_id=document_id, **NEVER_INDEXED)


def ids(items) -> list[int]:
    return [item.classification.document_id for item in items]


def test_an_empty_input_gives_an_empty_plan():
    plan = plan_of([])
    assert plan.items == ()
    assert plan.limit is None
    assert plan.to_run() == ()
    assert plan.deferred() == ()
    assert sum(plan.counts_by_category().values()) == 0


def test_the_plan_is_sorted_by_document_id_whatever_the_input_order():
    plan = plan_of([index_candidate(9), make_state(document_id=2), index_candidate(5)])
    assert ids(plan.items) == [2, 5, 9]


def test_a_generator_input_is_accepted():
    plan = plan_of(index_candidate(i) for i in (3, 1, 2))
    assert ids(plan.items) == [1, 2, 3]


def test_duplicate_document_ids_are_rejected():
    with pytest.raises(ValueError):
        plan_of([index_candidate(4), index_candidate(4)])


def test_counts_cover_every_category_and_action_with_zeros():
    plan = plan_of(
        [
            index_candidate(1),
            make_state(document_id=2),                          # CURRENT
            make_state(document_id=3, processing_status=None),  # NO_PROCESSING_STATUS
            make_state(document_id=4, indexing_status="failed", indexing_error="boom"),  # FAILED, REPORT
        ]
    )
    by_category = plan.counts_by_category()
    assert set(by_category) == set(Category)
    assert by_category[Category.NEVER_QUEUED] == 1
    assert by_category[Category.CURRENT] == 1
    assert by_category[Category.NO_PROCESSING_STATUS] == 1
    assert by_category[Category.FAILED] == 1
    assert by_category[Category.PENDING] == 0
    assert sum(by_category.values()) == 4

    by_action = plan.counts_by_action()
    assert by_action == {Action.INDEX: 1, Action.SKIP: 2, Action.REPORT: 1, Action.BLOCKED: 0}


def test_without_a_limit_nothing_is_deferred_and_every_index_item_runs():
    plan = plan_of([index_candidate(i) for i in (1, 2, 3)])
    assert ids(plan.to_run()) == [1, 2, 3]
    assert plan.deferred() == ()


def test_the_limit_keeps_the_lowest_ids_and_defers_the_rest():
    plan = plan_of([index_candidate(i) for i in (7, 3, 5, 1)], limit=2)
    assert ids(plan.to_run()) == [1, 3]
    assert ids(plan.deferred()) == [5, 7]
    assert plan.limit == 2
    assert [item.deferred_by_limit for item in plan.items] == [False, False, True, True]


def test_the_plan_still_lists_every_document_when_a_limit_is_set():
    plan = plan_of([index_candidate(i) for i in (1, 2, 3)], limit=1)
    assert ids(plan.items) == [1, 2, 3]


def test_a_limit_equal_to_the_candidate_count_defers_nothing():
    plan = plan_of([index_candidate(i) for i in (1, 2, 3)], limit=3)
    assert plan.deferred() == ()
    assert len(plan.to_run()) == 3


def test_a_limit_larger_than_the_candidate_count_defers_nothing():
    plan = plan_of([index_candidate(1)], limit=100)
    assert plan.deferred() == ()
    assert ids(plan.to_run()) == [1]


def test_only_index_candidates_count_toward_and_are_deferred_by_the_limit():
    states = [
        make_state(document_id=1),                                                     # CURRENT, SKIP
        index_candidate(2),                                                            # INDEX
        make_state(document_id=3, indexing_status="failed", indexing_error="boom"),   # REPORT
        make_state(document_id=4, **NEVER_INDEXED, source=SourceStatus.PDF_FILE_MISSING),  # BLOCKED
        index_candidate(5),                                                            # INDEX
        make_state(document_id=6, **NEVER_INDEXED, has_text=False),                    # NO_TEXT, REPORT
        index_candidate(7),                                                            # INDEX
    ]
    plan = plan_of(states, limit=1)
    assert ids(plan.to_run()) == [2]
    assert ids(plan.deferred()) == [5, 7]
    # Skip, report and blocked items are listed and never deferred.
    for item in plan.items:
        if item.classification.action is not Action.INDEX:
            assert item.deferred_by_limit is False
    assert plan.counts_by_action()[Action.BLOCKED] == 1


def test_blocked_items_are_not_counted_against_the_limit():
    states = [make_state(document_id=1, **NEVER_INDEXED, source=SourceStatus.PDF_UNCHECKED), index_candidate(2)]
    plan = plan_of(states, limit=1)
    assert ids(plan.to_run()) == [2]
    assert plan.deferred() == ()


def test_include_failed_is_passed_through_to_the_classifier():
    state = make_state(document_id=1, indexing_status="failed", indexing_error="boom")
    assert plan_of([state]).to_run() == ()
    assert ids(plan_of([state], include_failed=True).to_run()) == [1]


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_a_zero_or_negative_limit_is_rejected(bad):
    with pytest.raises(ValueError):
        plan_of([index_candidate(1)], limit=bad)


@pytest.mark.parametrize("bad", [1.0, 2.5, "3", True, False, [1], b"1"])
def test_a_non_integer_limit_is_rejected(bad):
    with pytest.raises(TypeError):
        plan_of([index_candidate(1)], limit=bad)


def test_the_limit_is_validated_before_the_states_are_consumed():
    states = iter([index_candidate(1)])
    with pytest.raises(ValueError):
        plan_of(states, limit=0)
    assert next(states).document_id == 1  # still unconsumed


def test_an_invalid_embedding_id_is_rejected_by_build_plan():
    with pytest.raises(ValueError):
        build_plan([index_candidate(1)], configured_embedding_id="")


# ---------- PlannedItem ----------

def test_only_index_items_can_be_deferred():
    skipped = Classification(1, Category.CURRENT, Action.SKIP)
    with pytest.raises(ValueError):
        PlannedItem(skipped, deferred_by_limit=True)
    assert PlannedItem(skipped).deferred_by_limit is False

    indexed = Classification(2, Category.NEVER_QUEUED, Action.INDEX)
    assert PlannedItem(indexed, deferred_by_limit=True).deferred_by_limit is True


# ---------- the module stays pure ----------

def test_the_module_depends_only_on_the_standard_library():
    allowed = {
        "__future__",
        "builtins",
        "collections",
        "dataclasses",
        "enum",
        "pathlib",
        "typing",
        backfill.__name__,
    }
    allowed_roots = {a.split(".")[0] for a in allowed}
    offenders = []
    for name, value in vars(backfill).items():
        if name.startswith("__"):  # interpreter-set dunders such as __loader__ and __builtins__
            continue
        origin = value.__name__ if inspect.ismodule(value) else getattr(value, "__module__", None)
        if origin is None:
            continue
        if origin.split(".")[0] not in allowed_roots:
            offenders.append((name, origin))
        if origin.startswith(("app.", "sqlalchemy", "torch", "transformers")) and origin != backfill.__name__:
            offenders.append((name, origin))
    assert offenders == []