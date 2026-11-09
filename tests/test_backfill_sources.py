"""Tests for PDF source verification in the backfill (C5.5 / CP3).

No database rows. Files live under a per-test upload root inside pytest's tmp_path, which
the tests point storage.UPLOAD_ROOT at, so the real storage.get_file_path does the path
resolution. Filesystem calls are also injected where a test needs a specific failure.
"""
import builtins
import io
from dataclasses import replace
from pathlib import Path

import pytest

from app.core import storage
from app.rag import backfill_sources
from app.rag.backfill import (
    Action,
    Category,
    DocumentState,
    LeaseState,
    SourceStatus,
    build_plan,
)
from app.rag.backfill_sources import attach_source_status, resolve_source_status

EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"


# ---------- helpers ----------

@pytest.fixture
def upload_root(tmp_path, monkeypatch) -> Path:
    """An upload root with a sibling directory, so escape attempts have real targets."""
    root = (tmp_path / "root").resolve()
    root.mkdir()
    monkeypatch.setattr(storage, "UPLOAD_ROOT", root)
    return root


@pytest.fixture
def outside_file(tmp_path, upload_root) -> Path:
    """A file that exists outside the upload root: reaching it would be an escape."""
    path = tmp_path / "outside.pdf"
    path.write_bytes(b"outside")
    return path


def make_state(**overrides) -> DocumentState:
    """A never-queued, PDF-like document whose source has not been checked."""
    values = dict(
        document_id=1,
        processing_status="completed",
        indexing_status=None,
        indexing_error=None,
        lease=LeaseState.NONE,
        source_version=1,
        indexed_source_version=None,
        indexed_embedding_id=None,
        has_text=True,
        chunks_present=False,
        filename="report.pdf",
        stored_name="abc.pdf",
        content_type="application/pdf",
        source=SourceStatus.PDF_UNCHECKED,
    )
    values.update(overrides)
    return DocumentState(**values)


def forbidden(_):
    raise AssertionError("the filesystem must not be consulted here")


def recording(function):
    """Wrap a one-argument callable and record every argument it receives."""
    calls = []

    def wrapper(argument):
        calls.append(argument)
        return function(argument)

    wrapper.calls = calls
    return wrapper


def raising(exception):
    def function(_):
        raise exception

    return function


def listing(directory: Path):
    """Names and modification times of everything under a directory."""
    return sorted((str(p.relative_to(directory)), p.stat().st_mtime_ns) for p in directory.rglob("*"))


# ---------- states that are never examined ----------

def test_a_non_pdf_state_is_returned_unchanged_and_never_touches_the_filesystem():
    state = make_state(
        filename="notes.txt", stored_name="gone.txt", content_type="text/plain", source=SourceStatus.NOT_PDF
    )
    assert resolve_source_status(state, locate=forbidden, is_file=forbidden) is state


@pytest.mark.parametrize(
    "source",
    [SourceStatus.PDF_PRESENT, SourceStatus.PDF_FILE_MISSING, SourceStatus.PDF_NO_STORED_NAME],
)
def test_an_already_resolved_state_is_returned_unchanged_and_never_touches_the_filesystem(source):
    state = make_state(source=source)
    assert resolve_source_status(state, locate=forbidden, is_file=forbidden) is state


# ---------- present, missing, directory ----------

def test_an_existing_file_is_present(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"%PDF-1.4 stand-in")
    result = resolve_source_status(make_state())
    assert result.source is SourceStatus.PDF_PRESENT


def test_the_file_is_not_parsed_so_garbage_content_is_still_present(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"this is not a pdf at all")
    assert resolve_source_status(make_state()).source is SourceStatus.PDF_PRESENT


def test_an_empty_file_is_present(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"")
    assert resolve_source_status(make_state()).source is SourceStatus.PDF_PRESENT


def test_a_missing_file_is_reported_missing(upload_root):
    assert resolve_source_status(make_state()).source is SourceStatus.PDF_FILE_MISSING


def test_a_directory_with_the_stored_name_is_reported_missing(upload_root):
    (upload_root / "abc.pdf").mkdir()
    assert resolve_source_status(make_state()).source is SourceStatus.PDF_FILE_MISSING


def test_a_missing_upload_directory_blocks_the_pdf_and_is_not_created(tmp_path, monkeypatch):
    absent = (tmp_path / "never-created").resolve()
    monkeypatch.setattr(storage, "UPLOAD_ROOT", absent)
    assert resolve_source_status(make_state()).source is SourceStatus.PDF_FILE_MISSING
    assert not absent.exists()


# ---------- no stored name ----------

@pytest.mark.parametrize("name", [None, "", "   ", "\t\n"])
def test_a_null_or_blank_stored_name_has_no_stored_name_and_no_filesystem_access(name):
    state = make_state(stored_name=name)
    result = resolve_source_status(state, locate=forbidden, is_file=forbidden)
    assert result.source is SourceStatus.PDF_NO_STORED_NAME


# ---------- path traversal and other rejected names (real get_file_path) ----------

@pytest.mark.parametrize("name", ["../outside.pdf", "./../outside.pdf", "sub/inner.pdf", "sub/../../outside.pdf"])
def test_a_traversal_or_subdirectory_name_is_rejected_even_when_the_target_exists(
    upload_root, outside_file, name
):
    inner = upload_root / "sub"
    inner.mkdir()
    (inner / "inner.pdf").write_bytes(b"inner")  # exists, but only directly under the root counts
    result = resolve_source_status(make_state(stored_name=name))
    assert result.source is SourceStatus.PDF_NO_STORED_NAME


def test_an_absolute_path_outside_the_root_is_rejected_even_when_it_exists(upload_root, outside_file):
    result = resolve_source_status(make_state(stored_name=str(outside_file)))
    assert result.source is SourceStatus.PDF_NO_STORED_NAME


def test_a_backslash_traversal_name_never_reaches_the_outside_file(upload_root, outside_file):
    # On Windows this is traversal (rejected); on POSIX it is just an odd file name (absent).
    result = resolve_source_status(make_state(stored_name="..\\outside.pdf"))
    assert result.source in {SourceStatus.PDF_NO_STORED_NAME, SourceStatus.PDF_FILE_MISSING}


def test_a_name_with_a_nul_byte_is_never_present(upload_root):
    result = resolve_source_status(make_state(stored_name="abc\x00.pdf"))
    assert result.source in {SourceStatus.PDF_NO_STORED_NAME, SourceStatus.PDF_FILE_MISSING}


def test_a_symlink_escaping_the_root_is_rejected(upload_root, outside_file):
    link = upload_root / "link.pdf"
    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    result = resolve_source_status(make_state(stored_name="link.pdf"))
    assert result.source is SourceStatus.PDF_NO_STORED_NAME


def test_a_symlink_to_a_file_inside_the_root_resolves_to_the_target_and_is_rejected(upload_root):
    # get_file_path resolves symlinks, so a link inside the root points at its target; the
    # target here is in a subdirectory, which the runner would also refuse.
    target_dir = upload_root / "sub"
    target_dir.mkdir()
    (target_dir / "real.pdf").write_bytes(b"x")
    link = upload_root / "alias.pdf"
    try:
        link.symlink_to(target_dir / "real.pdf")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    assert resolve_source_status(make_state(stored_name="alias.pdf")).source is SourceStatus.PDF_NO_STORED_NAME


# ---------- fail closed on filesystem errors (injected) ----------

@pytest.mark.parametrize(
    "error",
    [PermissionError("denied"), OSError("io error"), ValueError("bad path")],
    ids=["permission-denied", "os-error", "value-error"],
)
def test_an_is_file_failure_is_reported_missing(upload_root, error):
    result = resolve_source_status(make_state(), is_file=raising(error))
    assert result.source is SourceStatus.PDF_FILE_MISSING


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(ValueError("rejected"), SourceStatus.PDF_NO_STORED_NAME, id="value-error"),
        pytest.param(OSError("cannot resolve"), SourceStatus.PDF_FILE_MISSING, id="os-error"),
        pytest.param(PermissionError("denied"), SourceStatus.PDF_FILE_MISSING, id="permission-error"),
        pytest.param(RuntimeError("symlink loop"), SourceStatus.PDF_FILE_MISSING, id="runtime-error"),
    ],
)
def test_a_locate_failure_is_classified_and_never_raised(error, expected):
    result = resolve_source_status(make_state(), locate=raising(error), is_file=forbidden)
    assert result.source is expected


def test_an_unexpected_exception_type_is_not_swallowed():
    # Only filesystem-shaped errors fail closed; a genuine bug still surfaces.
    with pytest.raises(KeyError):
        resolve_source_status(make_state(), locate=raising(KeyError("bug")))


def test_is_file_returning_false_is_missing(upload_root):
    result = resolve_source_status(make_state(), locate=lambda name: upload_root / name, is_file=lambda p: False)
    assert result.source is SourceStatus.PDF_FILE_MISSING


def test_is_file_returning_true_is_present(upload_root):
    result = resolve_source_status(make_state(), locate=lambda name: upload_root / name, is_file=lambda p: True)
    assert result.source is SourceStatus.PDF_PRESENT

@pytest.mark.parametrize("value", [None, 1, "yes", [], object()])
def test_is_file_returning_non_bool_raises_type_error(value):
    with pytest.raises(TypeError, match="is_file must return a bool"):
        resolve_source_status(
            make_state(),
            locate=lambda name: Path(name),
            is_file=lambda path: value,
        )

# ---------- what is passed where ----------

def test_locate_receives_the_raw_stored_name_unchanged(upload_root):
    locate = recording(lambda name: upload_root / "x.pdf")
    resolve_source_status(make_state(stored_name=" abc.pdf "), locate=locate, is_file=lambda p: True)
    assert locate.calls == [" abc.pdf "]


def test_is_file_receives_exactly_the_path_that_locate_returned(upload_root):
    chosen = upload_root / "chosen.pdf"
    is_file = recording(lambda path: True)
    resolve_source_status(make_state(), locate=lambda name: chosen, is_file=is_file)
    assert is_file.calls == [chosen]


def test_the_default_locate_is_looked_up_at_call_time(upload_root, monkeypatch):
    elsewhere = upload_root / "elsewhere.pdf"
    elsewhere.write_bytes(b"x")
    monkeypatch.setattr(storage, "get_file_path", lambda name: elsewhere)
    assert resolve_source_status(make_state(stored_name="whatever.pdf")).source is SourceStatus.PDF_PRESENT


# ---------- the result ----------

def test_only_the_source_field_changes(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"x")
    original = make_state(indexing_status="pending", has_text=True, document_id=77)
    result = resolve_source_status(original)
    assert result is not original
    assert result.source is SourceStatus.PDF_PRESENT
    assert replace(result, source=original.source) == original
    assert original.source is SourceStatus.PDF_UNCHECKED  # the input object is untouched


def test_a_present_file_can_be_checked_without_opening_any_file(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"x")

    def refuse(*args, **kwargs):
        raise AssertionError("a file must not be opened")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", refuse)
        patch.setattr(io, "open", refuse)
        patch.setattr(Path, "open", refuse)
        patch.setattr(Path, "read_bytes", refuse)
        patch.setattr(Path, "read_text", refuse)
        result = resolve_source_status(make_state())
    assert result.source is SourceStatus.PDF_PRESENT


def test_checking_never_writes_to_the_upload_directory(upload_root):
    (upload_root / "abc.pdf").write_bytes(b"x")
    (upload_root / "dir.pdf").mkdir()
    before = listing(upload_root)

    states = [
        make_state(document_id=1),                          # present
        make_state(document_id=2, stored_name="gone.pdf"),  # missing
        make_state(document_id=3, stored_name="dir.pdf"),   # directory
        make_state(document_id=4, stored_name="../x.pdf"),  # rejected
        make_state(document_id=5, stored_name=None),        # no name
    ]
    attach_source_status(states)

    assert listing(upload_root) == before


# ---------- attach_source_status ----------

def test_attach_resolves_a_mixed_list_in_order(upload_root):
    (upload_root / "ok.pdf").write_bytes(b"x")
    non_pdf = make_state(document_id=1, filename="a.txt", stored_name="a.txt", content_type="text/plain", source=SourceStatus.NOT_PDF)
    present = make_state(document_id=2, stored_name="ok.pdf")
    missing = make_state(document_id=3, stored_name="gone.pdf")
    no_name = make_state(document_id=4, stored_name=None)
    resolved = make_state(document_id=5, source=SourceStatus.PDF_PRESENT)

    locate = recording(storage.get_file_path)
    is_file = recording(Path.is_file)
    result = attach_source_status([non_pdf, present, missing, no_name, resolved], locate=locate, is_file=is_file)

    assert [s.document_id for s in result] == [1, 2, 3, 4, 5]
    assert [s.source for s in result] == [
        SourceStatus.NOT_PDF,
        SourceStatus.PDF_PRESENT,
        SourceStatus.PDF_FILE_MISSING,
        SourceStatus.PDF_NO_STORED_NAME,
        SourceStatus.PDF_PRESENT,
    ]
    assert result[0] is non_pdf and result[4] is resolved       # untouched objects are preserved
    assert locate.calls == ["ok.pdf", "gone.pdf"]               # only unchecked PDFs with a name are located
    assert len(is_file.calls) == 2                              # and each is checked exactly once


def test_attach_returns_a_new_list_and_leaves_the_input_alone(upload_root):
    states = [make_state(document_id=1), make_state(document_id=2)]
    snapshot = list(states)
    result = attach_source_status(states)
    assert result is not states
    assert states == snapshot
    assert all(s.source is SourceStatus.PDF_UNCHECKED for s in states)


def test_attach_accepts_an_empty_input_and_a_generator(upload_root):
    assert attach_source_status([]) == []
    result = attach_source_status(make_state(document_id=i) for i in (3, 1, 2))
    assert [s.document_id for s in result] == [3, 1, 2]  # order of the input, not sorted


# ---------- with the CP1 classifier ----------

def test_resolved_states_drive_the_classifier(upload_root):
    (upload_root / "ok.pdf").write_bytes(b"x")
    states = [
        make_state(document_id=1, stored_name="ok.pdf"),
        make_state(document_id=2, stored_name="gone.pdf"),
        make_state(document_id=3, stored_name=None),
    ]
    unresolved = make_state(document_id=4, stored_name="ok.pdf")  # never passed through CP3

    plan = build_plan(attach_source_status(states) + [unresolved], configured_embedding_id=EMBEDDING_ID)
    items = {item.classification.document_id: item.classification for item in plan.items}

    assert (items[1].category, items[1].action, items[1].note) == (Category.NEVER_QUEUED, Action.INDEX, None)
    assert (items[2].category, items[2].action, items[2].note) == (
        Category.NEVER_QUEUED,
        Action.BLOCKED,
        "pdf_file_missing",
    )
    assert (items[3].category, items[3].action, items[3].note) == (
        Category.NEVER_QUEUED,
        Action.BLOCKED,
        "pdf_no_stored_name",
    )
    assert (items[4].category, items[4].action, items[4].note) == (
        Category.NEVER_QUEUED,
        Action.BLOCKED,
        "pdf_unchecked",
    )
    assert [item.classification.document_id for item in plan.to_run()] == [1]


def test_a_missing_pdf_never_counts_against_the_limit(upload_root):
    (upload_root / "ok.pdf").write_bytes(b"x")
    states = attach_source_status(
        [
            make_state(document_id=1, stored_name="gone.pdf"),
            make_state(document_id=2, stored_name="ok.pdf"),
            make_state(document_id=3, filename="n.txt", stored_name="n.txt", content_type="text/plain", source=SourceStatus.NOT_PDF),
        ]
    )
    plan = build_plan(states, configured_embedding_id=EMBEDDING_ID, limit=1)
    assert [item.classification.document_id for item in plan.to_run()] == [2]
    assert [item.classification.document_id for item in plan.deferred()] == [3]
    assert plan.counts_by_action()[Action.BLOCKED] == 1


def test_a_signal_mismatch_with_an_existing_file_is_accepted_as_present(upload_root):
    # The documented V1 limitation: PDF-like by filename, but the stored file is not a .pdf.
    (upload_root / "abc.txt").write_bytes(b"x")
    state = make_state(filename="report.pdf", stored_name="abc.txt", content_type="text/plain")
    assert resolve_source_status(state).source is SourceStatus.PDF_PRESENT


# ---------- module hygiene ----------

def test_the_module_exposes_only_its_two_functions_as_public_api():
    public = {name for name in vars(backfill_sources) if not name.startswith("_")}
    assert {"resolve_source_status", "attach_source_status"} <= public
    assert "SessionLocal" not in public and "engine" not in public