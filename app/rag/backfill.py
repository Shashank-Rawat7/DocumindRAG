"""Pure selection and categorization for the indexing backfill (Component 12 / C5.5, CP1).

Given one document's state and metadata, decide a category and an action. Nothing in
this module touches the database, the filesystem, the clock, the network or the
embedding model: it only reads the values it is handed, so every rule can be
unit-tested with plain objects.

Later checkpoints add the read-only SELECT that builds ``DocumentState`` values, the
source-file check that fills ``DocumentState.source``, the safety checks, the preflight,
the execution loop and the CLI. They live outside this pure section and call into it.

Rules mirrored from app/rag/indexing_service.py:
    * only ``processing_status == 'completed'`` documents can be indexed;
    * a live lease means another worker owns the document;
    * a published build is current when its source version, chunks and embedding id all
      match (build-signature drift is deliberately not detected in V1).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath

# ---------- vocabulary ----------


class Category(str, Enum):
    """Why a document is in the state it is in. Exactly one per document."""

    NO_PROCESSING_STATUS = "no_processing_status"      # processing_status is NULL
    EXTRACTION_INCOMPLETE = "extraction_incomplete"    # pending, processing or failed extraction
    UNRECOGNIZED = "unrecognized"                      # a status value the schema does not allow
    NEVER_QUEUED = "never_queued"                      # completed, indexing_status NULL
    PENDING = "pending"                                # completed, indexing_status 'pending'
    ABANDONED_LEASE = "abandoned_lease"                # 'indexing' with an expired lease
    LIVE_LEASE = "live_lease"                          # 'indexing' with a live lease
    LEASE_AMBIGUOUS = "lease_ambiguous"                # 'indexing' with no lease (the DB forbids it)
    FAILED = "failed"                                  # completed, indexing_status 'failed'
    INDEXED_STALE = "indexed_stale"                    # 'indexed' but version, chunks or embedding id is off
    INDEXED_WITH_ERROR = "indexed_with_error"          # valid publication that retains an indexing error
    CURRENT = "current"                                # fully current: nothing to do
    NO_TEXT = "no_text"                                # would be indexed, but has no extracted text


class Action(str, Enum):
    """What the backfill does with a document."""

    INDEX = "index"        # request and run indexing
    SKIP = "skip"          # leave alone, only counted
    REPORT = "report"      # leave alone, listed for a human
    BLOCKED = "blocked"    # eligible, but cannot be indexed yet (listed, never attempted)


class StaleReason(str, Enum):
    """Why an 'indexed' document is not current. A document may have several."""

    SOURCE_VERSION = "source_version"  # indexed_source_version is NULL or differs from source_version
    NO_CHUNKS = "no_chunks"            # no published document_chunks rows
    EMBEDDING_ID = "embedding_id"      # indexed_embedding_id differs from the configured one


class LeaseState(str, Enum):
    """Lease of an 'indexing' document, evaluated by the database against its own clock."""

    NONE = "none"
    LIVE = "live"
    EXPIRED = "expired"


class SourceStatus(str, Enum):
    """Whether the stored source file needed for page mapping is available.

    Only PDFs need the file (pages come from it). Anything but NOT_PDF and PDF_PRESENT
    blocks indexing, so a forgotten check (PDF_UNCHECKED) can never index a PDF.
    """

    NOT_PDF = "not_pdf"                        # no PDF signal: the file is not needed
    PDF_PRESENT = "pdf_present"                # PDF and its stored file exists
    PDF_FILE_MISSING = "pdf_file_missing"      # PDF, valid stored name, file absent
    PDF_NO_STORED_NAME = "pdf_no_stored_name"  # PDF signal, but no usable stored name
    PDF_UNCHECKED = "pdf_unchecked"            # PDF signal, availability not yet checked


# ---------- inputs and outputs ----------


@dataclass(frozen=True, slots=True)
class DocumentState:
    """Everything the classifier knows about one document. No defaults: every value must
    be supplied, so a missing fact is a TypeError, never a silent guess."""

    document_id: int
    processing_status: str | None
    indexing_status: str | None
    indexing_error: str | None
    lease: LeaseState                   # computed in SQL on the database clock
    source_version: int
    indexed_source_version: int | None
    indexed_embedding_id: str | None
    has_text: bool                      # computed in SQL; the text itself is never loaded
    chunks_present: bool                # EXISTS over document_chunks
    filename: str | None
    stored_name: str | None
    content_type: str | None
    source: SourceStatus                # resolved by the shell layer; required (fail closed)


@dataclass(frozen=True, slots=True)
class Classification:
    """The decision for one document."""

    document_id: int
    category: Category
    action: Action
    stale_reasons: tuple[StaleReason, ...] = ()
    note: str | None = None             # e.g. "requires --include-failed", "pdf_file_missing"

    # Reserved for later checkpoints: nothing here is computed beyond the constructor.


@dataclass(frozen=True, slots=True)
class PlannedItem:
    """A classification plus whether --limit defers it."""

    classification: Classification
    deferred_by_limit: bool = False

    def __post_init__(self) -> None:
        # Only INDEX candidates can be deferred; deferring anything else is a bug.
        if self.deferred_by_limit and self.classification.action is not Action.INDEX:
            raise ValueError("only INDEX items can be deferred by the limit")


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    """The complete, deterministic plan: every document, sorted by id."""

    items: tuple[PlannedItem, ...]
    limit: int | None

    def counts_by_category(self) -> dict[Category, int]:
        """Number of documents per category, with a zero for every category."""
        counts = Counter(item.classification.category for item in self.items)
        return {category: counts.get(category, 0) for category in Category}

    def counts_by_action(self) -> dict[Action, int]:
        """Number of documents per action, with a zero for every action."""
        counts = Counter(item.classification.action for item in self.items)
        return {action: counts.get(action, 0) for action in Action}

    def to_run(self) -> tuple[PlannedItem, ...]:
        """INDEX items that execution will process: not deferred, in id order."""
        return tuple(
            item
            for item in self.items
            if item.classification.action is Action.INDEX and not item.deferred_by_limit
        )

    def deferred(self) -> tuple[PlannedItem, ...]:
        """INDEX items held back by the limit, in id order."""
        return tuple(item for item in self.items if item.deferred_by_limit)


# ---------- constants ----------

NOTE_REQUIRES_INCLUDE_FAILED = "requires --include-failed"
NOTE_NO_TEXT = "no extracted text"

_PDF_SUFFIX = ".pdf"
_PDF_MIME = "application/pdf"
_SOURCE_USABLE = frozenset({SourceStatus.NOT_PDF, SourceStatus.PDF_PRESENT})
_EXTRACTION_UNFINISHED = frozenset({"pending", "processing", "failed"})


# ---------- PDF detection ----------


def _suffix(name: str | None) -> str:
    """Lowercase extension of a name, using PurePath (string work only, no filesystem)."""
    return PurePath(name).suffix.lower() if name else ""


def is_pdf_like(filename: str | None, stored_name: str | None, content_type: str | None) -> bool:
    """True when ANY signal says the document is a PDF.

    Any signal is enough on purpose: a completed PDF whose ``stored_name`` is NULL must
    not be mistaken for a non-PDF and indexed without its source file. The three signals
    agree for real uploads (the stored extension comes from the filename).
    """
    mime = (content_type or "").split(";")[0].strip().lower()
    return (
        _suffix(stored_name) == _PDF_SUFFIX
        or _suffix(filename) == _PDF_SUFFIX
        or mime == _PDF_MIME
    )


# ---------- classification ----------


def classify(
    state: DocumentState,
    *,
    configured_embedding_id: str,
    include_failed: bool = False,
) -> Classification:
    """Decide the category and action for one document. Pure and deterministic.

    ``include_failed`` covers indexing failures after a successful extraction only. A
    document whose extraction failed is never eligible, with or without the flag.
    """
    if not isinstance(configured_embedding_id, str) or not configured_embedding_id:
        raise ValueError("configured_embedding_id must be a non-empty string")

    def result(
        category: Category,
        action: Action,
        reasons: tuple[StaleReason, ...] = (),
        note: str | None = None,
    ) -> Classification:
        return Classification(state.document_id, category, action, reasons, note)

    # 1. Extraction state decides first; indexing columns are irrelevant until it completes.
    if state.processing_status is None:
        return result(Category.NO_PROCESSING_STATUS, Action.SKIP)
    if state.processing_status in _EXTRACTION_UNFINISHED:
        return result(Category.EXTRACTION_INCOMPLETE, Action.SKIP)
    if state.processing_status != "completed":
        return result(Category.UNRECOGNIZED, Action.REPORT)

    # 2. Completed extraction: classify the indexing state.
    status = state.indexing_status
    reasons: tuple[StaleReason, ...] = ()

    if status is None:
        category = Category.NEVER_QUEUED
    elif status == "pending":
        category = Category.PENDING
    elif status == "indexing":
        if state.lease is LeaseState.LIVE:
            return result(Category.LIVE_LEASE, Action.SKIP)
        if state.lease is LeaseState.NONE:
            return result(Category.LEASE_AMBIGUOUS, Action.REPORT)
        category = Category.ABANDONED_LEASE
    elif status == "failed":
        if not include_failed:
            return result(Category.FAILED, Action.REPORT, note=NOTE_REQUIRES_INCLUDE_FAILED)
        category = Category.FAILED
    elif status == "indexed":
        found: list[StaleReason] = []
        if state.indexed_source_version is None or state.indexed_source_version != state.source_version:
            found.append(StaleReason.SOURCE_VERSION)
        if not state.chunks_present:
            found.append(StaleReason.NO_CHUNKS)
        if state.indexed_embedding_id != configured_embedding_id:
            found.append(StaleReason.EMBEDDING_ID)

        if found:
            # Structural staleness wins over a retained error and needs no flag.
            category = Category.INDEXED_STALE
            reasons = tuple(found)
        elif state.indexing_error is not None:
            # Valid publication whose last re-index attempt failed.
            if not include_failed:
                return result(
                    Category.INDEXED_WITH_ERROR, Action.REPORT, note=NOTE_REQUIRES_INCLUDE_FAILED
                )
            category = Category.INDEXED_WITH_ERROR
        else:
            return result(Category.CURRENT, Action.SKIP)
    else:
        return result(Category.UNRECOGNIZED, Action.REPORT)

    # 3. Gates, for would-be INDEX results only. Order matters: no text first.
    if not state.has_text:
        return result(Category.NO_TEXT, Action.REPORT, reasons, NOTE_NO_TEXT)
    if state.source not in _SOURCE_USABLE:
        return result(category, Action.BLOCKED, reasons, state.source.value)
    return result(category, Action.INDEX, reasons)


# ---------- planning ----------


def _validate_limit(limit: object) -> None:
    """None means unlimited; otherwise a real int of at least 1 (bool is not accepted)."""
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(f"limit must be an int or None, not {type(limit).__name__}")
    if limit < 1:
        raise ValueError("limit must be at least 1")


def build_plan(
    states: Iterable[DocumentState],
    *,
    configured_embedding_id: str,
    include_failed: bool = False,
    limit: int | None = None,
) -> BackfillPlan:
    """Classify every state and apply the optional limit to INDEX candidates only.

    The plan lists every document, sorted by id. With a limit, the first ``limit`` INDEX
    candidates (by document id) are kept and the rest are marked ``deferred_by_limit``.
    Documents with any other action are never deferred.
    """
    _validate_limit(limit)  # before consuming ``states``

    classifications = [
        classify(
            state,
            configured_embedding_id=configured_embedding_id,
            include_failed=include_failed,
        )
        for state in states
    ]
    classifications.sort(key=lambda item: item.document_id)

    for earlier, later in zip(classifications, classifications[1:]):
        if earlier.document_id == later.document_id:
            raise ValueError(f"duplicate document id {earlier.document_id}")

    index_ids = [item.document_id for item in classifications if item.action is Action.INDEX]
    deferred = frozenset(index_ids[limit:]) if limit is not None else frozenset()

    items = tuple(
        PlannedItem(item, deferred_by_limit=item.document_id in deferred)
        for item in classifications
    )
    return BackfillPlan(items=items, limit=limit)