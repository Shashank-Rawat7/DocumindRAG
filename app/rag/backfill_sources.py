"""PDF source-file verification for the indexing backfill (Component 12 / C5.5, CP3).

The selection query (backfill_selection.py) hands over every PDF-like document with
``SourceStatus.PDF_UNCHECKED``, which the classifier treats as blocked. This module
replaces that status with a definite one by asking the filesystem whether the stored
file exists:

    PDF_PRESENT         the stored name resolves inside the upload directory and the
                        path is an existing regular file
    PDF_FILE_MISSING    the name resolves, but no regular file is there, or the
                        filesystem could not confirm it (fail closed)
    PDF_NO_STORED_NAME  stored_name is NULL or blank, or storage.get_file_path rejects it

The check is a stat only. It never opens, reads or parses a PDF, never writes, and never
raises for a filesystem condition. The path comes from ``storage.get_file_path``, the
same function the indexing runner uses, so a name that passes here resolves to the same
file when the document is indexed.

Only PDF_UNCHECKED states are examined. NOT_PDF and already-resolved states are returned
as the very same object, so non-PDF documents never need their file.

Known V1 limitations:
  * Existence only. A PDF that exists but cannot be read, or whose page text does not
    match the extracted text, still indexes with NULL page numbers (the runner tolerates
    that).
  * A document can be PDF-like by filename or content type while its stored file has a
    different extension. If that file exists it is reported PDF_PRESENT, yet the runner
    treats the stored suffix as non-PDF and publishes NULL pages. Real uploads cannot
    produce this (the stored extension comes from the filename), so it is accepted.
  * ``storage.UPLOAD_ROOT`` is fixed when app.core.storage is imported, relative to the
    current directory at that moment. From the wrong directory every PDF would report
    PDF_FILE_MISSING (safe, but misleading); the CLI safety checks must prevent that.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from app.core import storage
from app.rag.backfill import DocumentState, SourceStatus


def resolve_source_status(
    state: DocumentState,
    *,
    locate: Callable[[str], Path] | None = None,
    is_file: Callable[[Path], bool] | None = None,
) -> DocumentState:
    """Resolve ``PDF_UNCHECKED`` to PDF_PRESENT, PDF_FILE_MISSING or PDF_NO_STORED_NAME.

    Any other source status returns ``state`` itself, unchanged and uninspected.
    Otherwise a copy with only ``source`` replaced is returned.

    ``locate`` maps a stored name to a path (default: storage.get_file_path) and
    ``is_file`` tells whether a path is an existing regular file (default: Path.is_file).
    Both are looked up at call time, so they can be injected or monkeypatched.
    """
    if state.source is not SourceStatus.PDF_UNCHECKED:
        return state

    locate = locate if locate is not None else storage.get_file_path
    is_file = is_file if is_file is not None else Path.is_file

    name = state.stored_name
    if name is None or not name.strip():
        return replace(state, source=SourceStatus.PDF_NO_STORED_NAME)

    try:
        path = locate(name)  # the raw stored name, exactly as the runner passes it
    except ValueError:
        # Rejected: traversal, a subdirectory, a symlink escaping the root, a NUL byte.
        return replace(state, source=SourceStatus.PDF_NO_STORED_NAME)
    except (OSError, RuntimeError):
        # The filesystem could not even resolve the path (RuntimeError: a symlink loop
        # on Python 3.12). Fail closed.
        return replace(state, source=SourceStatus.PDF_FILE_MISSING)

    try:
        present = is_file(path)
        if not isinstance(present, bool):
            raise TypeError("is_file must return a bool")
    except (OSError, ValueError):
        present = False  # for example permission denied: presence cannot be confirmed

    return replace(
        state,
        source=SourceStatus.PDF_PRESENT if present else SourceStatus.PDF_FILE_MISSING,
    )


def attach_source_status(
    states: Iterable[DocumentState],
    *,
    locate: Callable[[str], Path] | None = None,
    is_file: Callable[[Path], bool] | None = None,
) -> list[DocumentState]:
    """Resolve every state, preserving order. Returns a new list; the input is not changed."""
    return [resolve_source_status(state, locate=locate, is_file=is_file) for state in states]