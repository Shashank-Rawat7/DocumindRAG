"""Evaluation data types and structural span validation (E1.2a).

This module holds the shapes of the evaluation corpus manifest and dataset (E1.1
contract), the constants that bound them, the issue and error types that report
problems, and the checks on ONE evidence span that need neither a manifest nor any
source text. It is pure: no I/O, no database, no model.

Deliberately NOT here (later checkpoints): parsing JSON into these types, checking a
span against the stored text, the cross-record dataset rules (duplicate IDs, split
collisions, the unit cap per question, and which optional ``Question`` fields may or
must be set for an answerable or an unanswerable question), near-duplicate detection
and metrics. The dataclasses below do NOT validate themselves, so an object built
from them is not yet known to be valid. In particular, ``Question`` uses optional
fields for both kinds of question; the dataset validator (E1.2d) must reject a
question whose fields do not fit its kind.

Offsets are Python code-point offsets, half-open: ``text[start:end]``. No message
raised or reported here ever contains a quote or any other document text.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.chunking import sha256_text

# ---------- constants ----------

MANIFEST_SCHEMA_VERSION = "eval-manifest-v1"
DATASET_SCHEMA_VERSION = "eval-dataset-v1"

# E1.1 contract: an answerable question has at most 20 required evidence units.
MAX_UNITS_PER_QUESTION = 20

LANGUAGES = frozenset({"en", "hi"})
SPLITS = ("dev", "test")
DOCUMENT_FORMATS = ("pdf", "txt", "md", "docx")
MANIFEST_STAGES = ("draft", "ingested")
AUTHOR_TYPES = ("human", "llm_assisted")
DIFFICULTIES = ("direct", "paraphrase")
UNANSWERABLE_TYPES = ("off_topic", "on_topic_not_covered")
SEMANTIC_REVIEW_RESULTS = ("confirmed_unanswerable",)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
_SEVERITIES = (SEVERITY_ERROR, SEVERITY_WARNING)

# Used with fullmatch(), so a trailing newline never slips through.
CORPUS_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
QUESTION_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")  # same pattern as the database CHECKs

# Issue codes reported by validate_span.
SPAN_FIELD_TYPE = "SPAN_FIELD_TYPE"
SPAN_CORPUS_KEY = "SPAN_CORPUS_KEY"
SPAN_SOURCE_SHA256_FORMAT = "SPAN_SOURCE_SHA256_FORMAT"
SPAN_QUOTE_SHA256_FORMAT = "SPAN_QUOTE_SHA256_FORMAT"
SPAN_OFFSET_RANGE = "SPAN_OFFSET_RANGE"
SPAN_QUOTE_EMPTY = "SPAN_QUOTE_EMPTY"
SPAN_QUOTE_LENGTH = "SPAN_QUOTE_LENGTH"
SPAN_QUOTE_ENCODING = "SPAN_QUOTE_ENCODING"
SPAN_QUOTE_SHA256_MISMATCH = "SPAN_QUOTE_SHA256_MISMATCH"


# ---------- evidence ----------

@dataclass(frozen=True, slots=True)
class Span:
    """One labelled evidence span: ``text[start:end]`` of one corpus document.

    ``source_sha256`` is the SHA-256 of that document's whole extracted text when
    the span was labelled; ``quote`` is the exact span text and ``quote_sha256`` its
    SHA-256 (hex, over the UTF-8 encoding). Nothing here needs the source text.
    """

    corpus_key: str
    source_sha256: str
    start: int
    end: int
    quote: str
    quote_sha256: str

    @property
    def interval(self) -> tuple[int, int]:
        """The half-open ``(start, end)`` interval, the only part metrics need."""
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """One required fact. ``spans`` are alternative places that each state it."""

    unit_id: str
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class SemanticReview:
    """A second reader's review of an unanswerable question."""

    reviewer: str
    date: str
    result: str


@dataclass(frozen=True, slots=True)
class RetrievalAssistance:
    """How retrieval output was used while checking an unanswerable question."""

    date: str
    method: str | None
    run_ref: str | None
    description: str


@dataclass(frozen=True, slots=True)
class Question:
    """One evaluation question, answerable or not.

    Answerable questions use ``expected_answer``, ``units``, ``source_language``,
    ``difficulty`` and ``close_distractor``. Unanswerable questions use
    ``unanswerable_type``, ``semantic_review`` and ``retrieval_assistance`` and have
    no units. Fields that do not apply are ``None`` (``units`` is empty). This class
    does not enforce that split; the dataset validator does.
    """

    question_id: str
    split: str
    answerable: bool
    question: str
    query_language: str
    author_type: str
    hindi_reviewed: bool
    hindi_reviewer: str | None = None
    notes: str | None = None
    expected_answer: str | None = None
    units: tuple[EvidenceUnit, ...] = ()
    source_language: str | None = None
    difficulty: str | None = None
    close_distractor: bool | None = None
    unanswerable_type: str | None = None
    semantic_review: SemanticReview | None = None
    retrieval_assistance: RetrievalAssistance | None = None


# ---------- corpus manifest ----------

@dataclass(frozen=True, slots=True)
class Extraction:
    """What produced a document's extracted text. Which optional fields are required
    depends on the document format; the manifest validator enforces that."""

    processing_ref: str
    python: str
    pypdf: str | None = None
    python_docx: str | None = None
    lxml: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """One corpus document. ``corpus_key`` is its stable identity (never a database
    id). ``extracted_text_sha256``, ``extracted_text_length`` and ``extraction`` are
    ``None`` in a draft manifest and filled in after ingestion."""

    corpus_key: str
    title: str
    publisher: str
    source_url: str
    license: str
    license_url: str
    format: str
    language: str
    source_file_sha256: str
    extracted_text_sha256: str | None
    extracted_text_length: int | None
    extraction: Extraction | None
    redistributable: bool


@dataclass(frozen=True, slots=True)
class Manifest:
    schema_version: str
    corpus_version: str
    stage: str
    documents: tuple[ManifestDocument, ...]


# ---------- issues and errors ----------

@dataclass(frozen=True, slots=True)
class Issue:
    """One validation finding. ``location`` names where (for example
    ``q12.units[0].spans[1].start``); ``message`` is fixed text. Neither may contain
    a quote or document text."""

    code: str
    location: str
    message: str
    severity: str = SEVERITY_ERROR

    def __post_init__(self) -> None:
        for value in (self.code, self.location, self.message):
            if type(value) is not str:
                raise ValueError("code, location and message must be str")
        if self.severity not in _SEVERITIES:
            raise ValueError("severity must be 'error' or 'warning'")


class EvaluationDataError(ValueError):
    """Raised when evaluation data fails validation. ``issues`` holds every finding.
    The message states only how many there are: an Issue field may hold any string,
    so none of them is ever copied into the message."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        items = tuple(issues)
        if not all(isinstance(item, Issue) for item in items):
            raise TypeError("issues must contain only Issue objects")
        self.issues = items
        super().__init__(f"{len(items)} evaluation data issue(s)")


# ---------- structural span validation ----------

def validate_span(span: Span, *, location: str) -> list[Issue]:
    """Structural problems of one span, found without the source text or manifest.

    Returns every problem, in a fixed order: field types, then formats (corpus key,
    source hash, quote hash), then the offset range, then the quote checks (empty,
    length, encoding or hash). An empty list means the span is structurally valid;
    it does not mean the span matches any document. A problem that makes a later
    check meaningless suppresses that check, so one mistake is reported once.
    ``location`` is a caller-chosen prefix and must not contain document text.
    """
    if not isinstance(span, Span):
        raise TypeError("span must be a Span")
    if type(location) is not str:
        raise TypeError("location must be a str")

    issues: list[Issue] = []

    def add(code: str, field: str, message: str) -> None:
        issues.append(Issue(code, f"{location}.{field}", message))

    # 1. field types
    text_ok = {}
    for name in ("corpus_key", "source_sha256", "quote", "quote_sha256"):
        text_ok[name] = type(getattr(span, name)) is str
        if not text_ok[name]:
            add(SPAN_FIELD_TYPE, name, f"{name} must be a str")
    offsets_ok = True
    for name in ("start", "end"):
        if type(getattr(span, name)) is not int:
            offsets_ok = False
            add(SPAN_FIELD_TYPE, name, f"{name} must be an int")

    # 2. formats
    if text_ok["corpus_key"] and CORPUS_KEY_PATTERN.fullmatch(span.corpus_key) is None:
        add(SPAN_CORPUS_KEY, "corpus_key", "corpus_key does not match the corpus key pattern")
    if text_ok["source_sha256"] and SHA256_HEX_PATTERN.fullmatch(span.source_sha256) is None:
        add(SPAN_SOURCE_SHA256_FORMAT, "source_sha256", "source_sha256 must be 64 lowercase hex characters")
    quote_hash_ok = text_ok["quote_sha256"]
    if quote_hash_ok and SHA256_HEX_PATTERN.fullmatch(span.quote_sha256) is None:
        quote_hash_ok = False
        add(SPAN_QUOTE_SHA256_FORMAT, "quote_sha256", "quote_sha256 must be 64 lowercase hex characters")

    # 3. offset range
    range_ok = False
    if offsets_ok:
        range_ok = 0 <= span.start < span.end
        if not range_ok:
            add(SPAN_OFFSET_RANGE, "offsets", "offsets must satisfy 0 <= start < end")

    # 4. quote checks
    if text_ok["quote"]:
        if span.quote == "":
            add(SPAN_QUOTE_EMPTY, "quote", "quote must not be empty")
        elif range_ok and len(span.quote) != span.end - span.start:
            add(
                SPAN_QUOTE_LENGTH,
                "quote",
                f"quote length {len(span.quote)} differs from end - start {span.end - span.start}",
            )
        try:
            digest = sha256_text(span.quote)
        except UnicodeEncodeError:
            add(SPAN_QUOTE_ENCODING, "quote", "quote cannot be encoded as UTF-8")
        else:
            if quote_hash_ok and digest != span.quote_sha256:
                add(SPAN_QUOTE_SHA256_MISMATCH, "quote_sha256", "quote_sha256 does not match the quote")

    return issues
