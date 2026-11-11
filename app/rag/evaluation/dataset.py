"""Dataset rules for evaluation questions (E1.2d).

Step 3 holds ``validate_question``: the record-level rules for ONE ``Question``. It needs no
manifest and no source text, and it never calls ``validate_span`` (a span's own fields are
checked there; here a span is only a member of a unit, with an identity of
``(corpus_key, start, end)``). Step 4 adds ``validate_dataset``: the rules across a whole
dataset, built from ``validate_question``, ``validate_span``/``resolve_question``, the manifest
and the step 2 duplicate detector (see its docstring). The dataset metadata file is a later
step.

Like ``validate_manifest`` the check is type-defensive: dataclasses built by hand may hold
any value, so a wrong type is reported once and the checks that depend on it are skipped.
Types are exact (``type(x) is str``; ``bool`` is never an ``int``). All problems are
collected; nothing is raised for bad data.

Order of issues: the common fields in schema order (the unanswerable-split rule comes right
after the split check), the Hindi review rules, then the remaining fields in schema order
(each applying to the kind of question it belongs to), then the units in order, each with its
spans in order.

Privacy: every message is fixed text from ``_MESSAGES``. A location is the caller's
``location`` prefix plus fixed schema field names and numeric indexes. Question text,
answers, identifiers, reviewer names, corpus keys and quotes never appear in an issue. The
``location`` prefix is chosen by the caller: it is trusted, must be fixed text, and must
never contain user-controlled content (the dataset validator passes ``dev.questions[3]``-style
positions).
"""
from __future__ import annotations

import dataclasses
import datetime
import re
from collections.abc import Mapping, Sequence

from app.rag.evaluation.schema import (
    AUTHOR_TYPES,
    DIFFICULTIES,
    LANGUAGES,
    MAX_UNITS_PER_QUESTION,
    QUESTION_ID_PATTERN,
    SEMANTIC_REVIEW_RESULTS,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    SPLITS,
    UNANSWERABLE_TYPES,
    EvidenceUnit,
    Issue,
    Manifest,
    Question,
    RetrievalAssistance,
    SemanticReview,
    Span,
)

# ---------- issue codes (record level) ----------

Q_FIELD_TYPE = "Q_FIELD_TYPE"
Q_QUESTION_ID = "Q_QUESTION_ID"
Q_ENUM = "Q_ENUM"
Q_BLANK = "Q_BLANK"
Q_DATE = "Q_DATE"
Q_DESCRIPTION_LENGTH = "Q_DESCRIPTION_LENGTH"
Q_FIELD_MISSING = "Q_FIELD_MISSING"
Q_INAPPLICABLE_FIELD = "Q_INAPPLICABLE_FIELD"
Q_UNANSWERABLE_SPLIT = "Q_UNANSWERABLE_SPLIT"
Q_UNIT_COUNT = "Q_UNIT_COUNT"
Q_UNITS_TYPE = "Q_UNITS_TYPE"
Q_UNIT_TYPE = "Q_UNIT_TYPE"
Q_SPANS_TYPE = "Q_SPANS_TYPE"
Q_SPAN_TYPE = "Q_SPAN_TYPE"
Q_UNIT_ID_DUPLICATE = "Q_UNIT_ID_DUPLICATE"
Q_UNIT_NO_SPANS = "Q_UNIT_NO_SPANS"
Q_SPAN_DUPLICATE_IN_UNIT = "Q_SPAN_DUPLICATE_IN_UNIT"
Q_SPAN_REUSED_ACROSS_UNITS = "Q_SPAN_REUSED_ACROSS_UNITS"
Q_HINDI_REVIEW_REQUIRED = "Q_HINDI_REVIEW_REQUIRED"
Q_HINDI_REVIEW_INAPPLICABLE = "Q_HINDI_REVIEW_INAPPLICABLE"
Q_HINDI_REVIEWER_REQUIRED = "Q_HINDI_REVIEWER_REQUIRED"
Q_HINDI_REVIEWER_INAPPLICABLE = "Q_HINDI_REVIEWER_INAPPLICABLE"

# Every code is an error except this one.
_WARNING_CODES = frozenset({Q_SPAN_REUSED_ACROSS_UNITS})

MAX_RETRIEVAL_DESCRIPTION_LENGTH = 500  # code points

_MESSAGES = {
    Q_FIELD_TYPE: "a field has the wrong type",
    Q_QUESTION_ID: "question_id does not match the question id pattern",
    Q_ENUM: "a field is not one of its allowed values",
    Q_BLANK: "a field must not be blank",
    Q_DATE: "a date must be a real calendar date written as YYYY-MM-DD",
    Q_DESCRIPTION_LENGTH: "the description is longer than the allowed length",
    Q_FIELD_MISSING: "a field required for this kind of question is missing",
    Q_INAPPLICABLE_FIELD: "a field does not apply to this kind of question",
    Q_UNANSWERABLE_SPLIT: "an unanswerable question must be in the dev split",
    Q_UNIT_COUNT: "an answerable question needs between 1 and the maximum number of evidence units",
    Q_UNITS_TYPE: "units must be a tuple",
    Q_UNIT_TYPE: "a unit must be an evidence unit",
    Q_SPANS_TYPE: "a unit's spans must be a tuple",
    Q_SPAN_TYPE: "a span must be an evidence span",
    Q_UNIT_ID_DUPLICATE: "a unit id is already used by an earlier unit of this question",
    Q_UNIT_NO_SPANS: "a unit must have at least one span",
    Q_SPAN_DUPLICATE_IN_UNIT: "this span repeats an earlier span of the same unit",
    Q_SPAN_REUSED_ACROSS_UNITS: "this span is also used by an earlier unit of this question; review it",
    Q_HINDI_REVIEW_REQUIRED: "a Hindi query must be marked as Hindi-reviewed",
    Q_HINDI_REVIEW_INAPPLICABLE: "an English query must not be marked as Hindi-reviewed",
    Q_HINDI_REVIEWER_REQUIRED: "a Hindi query needs a named Hindi reviewer",
    Q_HINDI_REVIEWER_INAPPLICABLE: "an English query must not have a Hindi reviewer",
}

_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")  # used with fullmatch()


# ---------- small helpers ----------

def _is_str(value: object) -> bool:
    return type(value) is str


def _is_int(value: object) -> bool:
    return type(value) is int


def _blank(value: str) -> bool:
    return value.strip() == ""


def is_valid_iso_date(value: object) -> bool:
    """True only for an exact ``str`` of the form ``YYYY-MM-DD`` that names a real date.

    The digits must be ASCII ``0``-``9`` (the pattern spells the range out, so no other
    script's digits match), and the date must exist on the calendar: month 1-12, a day that
    month has (29 February only in a leap year), and year 0001 or later, so ``0000-01-01`` is
    refused. There is no future-date rule. Anything that is not exactly a ``str`` (``None``,
    numbers, ``bytes``, ``datetime.date``, a ``str`` subclass) gives ``False``; the function
    never raises.
    """
    if type(value) is not str or _DATE_PATTERN.fullmatch(value) is None:
        return False
    try:
        datetime.date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        return False
    return True


class _Collector:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, code: str, location: str) -> None:
        severity = SEVERITY_WARNING if code in _WARNING_CODES else SEVERITY_ERROR
        self.issues.append(Issue(code, location, _MESSAGES[code], severity))

    def text(self, value: object, location: str) -> None:
        """A required non-blank string."""
        if not _is_str(value):
            self.add(Q_FIELD_TYPE, location)
        elif _blank(value):
            self.add(Q_BLANK, location)

    def optional_text(self, value: object, location: str) -> None:
        """None, or a non-blank string."""
        if value is not None:
            self.text(value, location)

    def enum(self, value: object, allowed: object, location: str) -> bool:
        """True when ``value`` is a str in ``allowed``; otherwise the right issue is added."""
        if not _is_str(value):
            self.add(Q_FIELD_TYPE, location)
            return False
        if value not in allowed:  # type: ignore[operator]
            self.add(Q_ENUM, location)
            return False
        return True

    def flag(self, value: object, location: str) -> bool:
        if type(value) is not bool:
            self.add(Q_FIELD_TYPE, location)
            return False
        return True

    def date(self, value: object, location: str) -> None:
        if not _is_str(value):
            self.add(Q_FIELD_TYPE, location)
        elif not is_valid_iso_date(value):
            self.add(Q_DATE, location)


# ---------- the checks ----------

def _check_hindi(c: _Collector, q: Question, location: str) -> None:
    """The Hindi review rules, for answerable and unanswerable questions alike.

    A rule is skipped when its own field has the wrong type (already reported) or when
    ``query_language`` is not a valid language (already reported). The two fields are
    judged independently: a wrong-typed ``hindi_reviewed`` does not skip the reviewer rule.

    Two different things apply to the reviewer. The language-dependent rules (required and
    non-blank for ``hi``; must be ``None`` for ``en``) need a valid ``query_language``. The
    general rule that an optional reviewer, when it is a string, is not blank does not: it
    applies exactly when no language-dependent rule does, i.e. when ``query_language`` is
    invalid or wrong-typed. A valid language therefore never gives ``Q_BLANK`` for the reviewer.
    """
    language = q.query_language if _is_str(q.query_language) and q.query_language in LANGUAGES else None
    reviewed, reviewer = q.hindi_reviewed, q.hindi_reviewer
    reviewed_ok = type(reviewed) is bool
    reviewer_ok = reviewer is None or _is_str(reviewer)
    if language == "hi":
        if reviewed_ok and reviewed is not True:
            c.add(Q_HINDI_REVIEW_REQUIRED, f"{location}.hindi_reviewed")
        if reviewer_ok and not (_is_str(reviewer) and not _blank(reviewer)):
            c.add(Q_HINDI_REVIEWER_REQUIRED, f"{location}.hindi_reviewer")
    elif language == "en":
        if reviewed_ok and reviewed is not False:
            c.add(Q_HINDI_REVIEW_INAPPLICABLE, f"{location}.hindi_reviewed")
        if reviewer_ok and reviewer is not None:
            c.add(Q_HINDI_REVIEWER_INAPPLICABLE, f"{location}.hindi_reviewer")
    elif _is_str(reviewer) and _blank(reviewer):
        c.add(Q_BLANK, f"{location}.hindi_reviewer")  # no Hindi rule applies: a reviewer must still be non-blank


def _check_semantic_review(c: _Collector, value: object, location: str) -> None:
    if not isinstance(value, SemanticReview):
        c.add(Q_FIELD_TYPE, location)
        return
    c.text(value.reviewer, f"{location}.reviewer")
    c.date(value.date, f"{location}.date")
    c.enum(value.result, SEMANTIC_REVIEW_RESULTS, f"{location}.result")


def _check_retrieval_assistance(c: _Collector, value: object, location: str) -> None:
    if not isinstance(value, RetrievalAssistance):
        c.add(Q_FIELD_TYPE, location)
        return
    c.date(value.date, f"{location}.date")
    description_location = f"{location}.description"
    c.text(value.description, description_location)
    if _is_str(value.description) and not _blank(value.description):
        if len(value.description) > MAX_RETRIEVAL_DESCRIPTION_LENGTH:
            c.add(Q_DESCRIPTION_LENGTH, description_location)
    c.optional_text(value.method, f"{location}.method")
    c.optional_text(value.run_ref, f"{location}.run_ref")


def _span_identity(span: Span) -> tuple[str, int, int] | None:
    """``(corpus_key, start, end)`` when those three fields are well typed, else None."""
    if _is_str(span.corpus_key) and _is_int(span.start) and _is_int(span.end):
        return (span.corpus_key, span.start, span.end)
    return None


def _check_units(c: _Collector, units: tuple, location: str) -> None:
    seen_unit_ids: set[str] = set()
    spans_of_earlier_units: set[tuple[str, int, int]] = set()
    for i, unit in enumerate(units):
        unit_location = f"{location}[{i}]"
        if not isinstance(unit, EvidenceUnit):
            c.add(Q_UNIT_TYPE, unit_location)
            continue

        id_location = f"{unit_location}.unit_id"
        if not _is_str(unit.unit_id):
            c.add(Q_FIELD_TYPE, id_location)
        elif _blank(unit.unit_id):
            c.add(Q_BLANK, id_location)
        elif unit.unit_id in seen_unit_ids:
            c.add(Q_UNIT_ID_DUPLICATE, id_location)
        else:
            seen_unit_ids.add(unit.unit_id)

        spans_location = f"{unit_location}.spans"
        if type(unit.spans) is not tuple:
            c.add(Q_SPANS_TYPE, spans_location)
            continue
        if len(unit.spans) == 0:
            c.add(Q_UNIT_NO_SPANS, spans_location)
            continue

        spans_of_this_unit: set[tuple[str, int, int]] = set()
        for j, span in enumerate(unit.spans):
            span_location = f"{spans_location}[{j}]"
            if not isinstance(span, Span):
                c.add(Q_SPAN_TYPE, span_location)
                continue
            identity = _span_identity(span)
            if identity is None:
                continue  # an ill-typed span has no identity; validate_span reports its fields
            if identity in spans_of_this_unit:
                c.add(Q_SPAN_DUPLICATE_IN_UNIT, span_location)
            else:
                spans_of_this_unit.add(identity)
                if identity in spans_of_earlier_units:
                    c.add(Q_SPAN_REUSED_ACROSS_UNITS, span_location)
        spans_of_earlier_units |= spans_of_this_unit


def _check_units_field(c: _Collector, q: Question, answerable: bool, location: str) -> None:
    units_location = f"{location}.units"
    if type(q.units) is not tuple:
        c.add(Q_UNITS_TYPE, units_location)
        return
    if not answerable:
        if len(q.units) != 0:
            c.add(Q_INAPPLICABLE_FIELD, units_location)
        return
    if not 1 <= len(q.units) <= MAX_UNITS_PER_QUESTION:
        c.add(Q_UNIT_COUNT, units_location)
    _check_units(c, q.units, units_location)


def _check_kind_fields(c: _Collector, q: Question, answerable: bool, location: str) -> None:
    """The fields whose rules depend on the kind of question, in schema order."""

    def required(value: object, field: str, check) -> None:
        if not answerable:
            if value is not None:
                c.add(Q_INAPPLICABLE_FIELD, f"{location}.{field}")
        elif value is None:
            c.add(Q_FIELD_MISSING, f"{location}.{field}")
        else:
            check(value, f"{location}.{field}")

    required(q.expected_answer, "expected_answer", c.text)
    _check_units_field(c, q, answerable, location)
    required(q.source_language, "source_language", lambda v, loc: c.enum(v, LANGUAGES, loc))
    required(q.difficulty, "difficulty", lambda v, loc: c.enum(v, DIFFICULTIES, loc))
    required(q.close_distractor, "close_distractor", c.flag)

    def only_when_unanswerable(value: object, field: str, check) -> None:
        if answerable:
            if value is not None:
                c.add(Q_INAPPLICABLE_FIELD, f"{location}.{field}")
        elif value is None:
            if field != "retrieval_assistance":  # retrieval_assistance is optional here
                c.add(Q_FIELD_MISSING, f"{location}.{field}")
        else:
            check(value, f"{location}.{field}")

    only_when_unanswerable(q.unanswerable_type, "unanswerable_type", lambda v, loc: c.enum(v, UNANSWERABLE_TYPES, loc))
    only_when_unanswerable(q.semantic_review, "semantic_review", lambda v, loc: _check_semantic_review(c, v, loc))
    only_when_unanswerable(
        q.retrieval_assistance, "retrieval_assistance", lambda v, loc: _check_retrieval_assistance(c, v, loc)
    )


def validate_question(q: Question, *, location: str = "question") -> list[Issue]:
    """Every record-level problem in ``q``, in a fixed order; an empty list means none.

    ``location`` is a trusted, caller-chosen prefix for every issue location. It must be
    fixed text and must never contain user-controlled content. Raises ``TypeError`` if ``q``
    is not a ``Question`` or ``location`` is not a ``str``. ``q`` is not modified.
    """
    if not isinstance(q, Question):
        raise TypeError("q must be a Question")
    if type(location) is not str:
        raise TypeError("location must be a str")

    c = _Collector()

    # The common fields.
    if not _is_str(q.question_id):
        c.add(Q_FIELD_TYPE, f"{location}.question_id")
    elif QUESTION_ID_PATTERN.fullmatch(q.question_id) is None:
        c.add(Q_QUESTION_ID, f"{location}.question_id")
    c.text(q.question, f"{location}.question")
    c.enum(q.split, SPLITS, f"{location}.split")
    if q.answerable is False and _is_str(q.split) and q.split == "test":
        c.add(Q_UNANSWERABLE_SPLIT, f"{location}.split")
    c.enum(q.query_language, LANGUAGES, f"{location}.query_language")
    c.enum(q.author_type, AUTHOR_TYPES, f"{location}.author_type")
    answerable_ok = c.flag(q.answerable, f"{location}.answerable")
    c.flag(q.hindi_reviewed, f"{location}.hindi_reviewed")
    if not (q.hindi_reviewer is None or _is_str(q.hindi_reviewer)):
        c.add(Q_FIELD_TYPE, f"{location}.hindi_reviewer")
    if not (q.notes is None or _is_str(q.notes)):
        c.add(Q_FIELD_TYPE, f"{location}.notes")
    elif q.notes is not None and _blank(q.notes):
        c.add(Q_BLANK, f"{location}.notes")

    _check_hindi(c, q, location)

    # The fields that depend on the kind of question; unknown kind means they cannot be judged.
    if answerable_ok:
        _check_kind_fields(c, q, q.answerable is True, location)
    return c.issues


# ---------- dataset level (step 4) ----------

# Issue codes reported by validate_dataset itself. Issues from validate_question (Q_*), from
# the span checks (SPAN_*, ANCHOR_*) and from the duplicate detector (DATASET_DUPLICATE_QUESTION_TEXT,
# DATASET_SIMILAR_QUESTION, defined in near_duplicates) are passed through unchanged.
DATASET_SPLITS = "DATASET_SPLITS"
DATASET_QUESTION_TYPE = "DATASET_QUESTION_TYPE"
DATASET_MANIFEST_INVALID = "DATASET_MANIFEST_INVALID"
DATASET_MANIFEST_NOT_INGESTED = "DATASET_MANIFEST_NOT_INGESTED"
DATASET_SPLIT_MISMATCH = "DATASET_SPLIT_MISMATCH"
DATASET_SPAN_KEY_UNKNOWN = "DATASET_SPAN_KEY_UNKNOWN"
DATASET_SPAN_HASH_MISMATCH = "DATASET_SPAN_HASH_MISMATCH"
DATASET_MIXED_LANGUAGE_SPANS = "DATASET_MIXED_LANGUAGE_SPANS"
DATASET_SOURCE_LANGUAGE_MISMATCH = "DATASET_SOURCE_LANGUAGE_MISMATCH"
DATASET_DUPLICATE_QUESTION_ID = "DATASET_DUPLICATE_QUESTION_ID"
DATASET_SPLIT_COLLISION = "DATASET_SPLIT_COLLISION"

# Kept apart from ``_MESSAGES`` (the record-level mapping), which holds exactly the Q_* codes.
_DATASET_MESSAGES = {
    DATASET_SPLITS: "the dataset must have exactly the collections dev and test",
    DATASET_QUESTION_TYPE: "the item is not a question",
    DATASET_MANIFEST_INVALID: "the manifest is not valid, so spans cannot be checked against it",
    DATASET_MANIFEST_NOT_INGESTED: "the manifest is not at the ingested stage, so spans cannot be checked against it",
    DATASET_SPLIT_MISMATCH: "the question's split differs from the collection it is in",
    DATASET_SPAN_KEY_UNKNOWN: "the span's corpus key is not a document in the manifest",
    DATASET_SPAN_HASH_MISMATCH: "the span's source_sha256 differs from the manifest's extracted_text_sha256",
    DATASET_MIXED_LANGUAGE_SPANS: "the question's spans are in documents of different languages",
    DATASET_SOURCE_LANGUAGE_MISMATCH: "source_language differs from the language of the question's documents",
    DATASET_DUPLICATE_QUESTION_ID: "the question id is already used by an earlier question",
    DATASET_SPLIT_COLLISION: "this test span overlaps or touches a dev evidence span of the same document",
}


def _dataset_issue(code: str, location: str) -> Issue:
    return Issue(code, location, _DATASET_MESSAGES[code])


def _placeholder_question(split: str) -> Question:
    """A question the duplicate detector counts but skips (its text is not a ``str``)."""
    return Question(
        question_id="", split=split, answerable=False, question=None, query_language=None,  # type: ignore[arg-type]
        author_type="", hindi_reviewed=False,
    )


def _containers_are_well_formed(q: Question) -> bool:
    """``units`` is a tuple of units, each with a tuple of spans that are all ``Span`` objects."""
    if type(q.units) is not tuple:
        return False
    for unit in q.units:
        if not isinstance(unit, EvidenceUnit) or type(unit.spans) is not tuple:
            return False
        if not all(isinstance(span, Span) for span in unit.spans):
            return False
    return True


def _is_checkable_answerable(q: object) -> bool:
    """An answerable question whose unit and span containers can be walked."""
    return isinstance(q, Question) and q.answerable is True and _containers_are_well_formed(q)


def _located_spans(q: Question, location: str):
    """``(span location, span)`` for every span of a checkable question, in order."""
    for u, unit in enumerate(q.units):
        for s, span in enumerate(unit.spans):
            yield f"{location}.units[{u}].spans[{s}]", span


def _valid_interval(span: Span) -> bool:
    return (
        _is_str(span.corpus_key)
        and _is_int(span.start)
        and _is_int(span.end)
        and 0 <= span.start < span.end
    )


def _supplied_text_hash(texts: Mapping[str, str] | None, key: str, cache: dict[str, str | None]) -> str | None:
    """The hash of the supplied text for ``key``, or ``None`` if it is absent or unusable.

    Each key is hashed at most once per ``validate_dataset`` call; ``cache`` holds the results.
    """
    from app.rag.chunking import sha256_text

    if key in cache:
        return cache[key]
    digest = None
    if texts is not None and key in texts:
        text = texts[key]
        if _is_str(text) and text != "":
            try:
                digest = sha256_text(text)
            except UnicodeEncodeError:
                digest = None
    cache[key] = digest
    return digest


def validate_dataset(
    questions: Mapping[str, Sequence[Question]],
    manifest: Manifest,
    *,
    near_duplicate_threshold: float,
    texts: Mapping[str, str] | None = None,
) -> list[Issue]:
    """Every dataset-level problem, in a fixed order; an empty list means none.

    ``questions`` maps ``"dev"`` and ``"test"`` to sequences of ``Question``. ``manifest`` is
    required. ``near_duplicate_threshold`` is required (a number from 0 to 1, see
    ``near_duplicates``). ``texts`` maps corpus keys to extracted text.

    Official-run contract (binding on the future runner; NOT enforced here). An official
    evaluation run must supply ``texts`` for every document in the manifest, call
    ``verify_source_text`` on each document, and reject absent or incomplete texts.
    ``texts=None`` is for structural, development and testing use only: spans are then checked
    only for structure and against the manifest, never against the source text.

    Raises ``TypeError`` for a wrongly typed argument (``questions`` not a mapping, a ``dev`` or
    ``test`` collection that is not a sequence, ``manifest`` not a ``Manifest``, ``texts``
    neither ``None`` nor a mapping) and the threshold errors of ``find_duplicate_questions``.
    Bad data is reported as issues, never raised.

    Phases, in order:

    A. Split shape: the keys must be exactly ``dev`` and ``test`` (``DATASET_SPLITS`` at
       ``dataset``); the later phases still run on whichever of the two are present.
    B. Manifest gate, once: ``DATASET_MANIFEST_INVALID`` at ``manifest`` if ``validate_manifest``
       reports an error, else ``DATASET_MANIFEST_NOT_INGESTED`` at ``manifest.stage`` if the stage
       is not ``ingested``. A failed gate skips only the manifest-reference and language checks.
    C. Each question, dev then test, in input order. An item that is not a ``Question`` gets one
       ``DATASET_QUESTION_TYPE`` error at its own location and no further checks. Otherwise:
       (1) ``validate_question`` and split consistency; (2) span validation, for answerable questions with well-formed unit and span
       containers: ``resolve_question`` when ``texts`` is given, otherwise ``validate_span`` on
       each span, never both; (3) if the gate passed, manifest references (unknown document key,
       source hash mismatch) and language checks (mixed document languages, else source-language
       mismatch).
    D. Across the dataset: duplicate question ids; dev/test evidence collisions (one error per
       colliding test span, using ``collides``); exact duplicate question text; similar questions.
    E. Deduplication by ``(code, location)``, then two suppressions: ``ANCHOR_TEXT_MISSING`` is
       dropped where ``DATASET_SPAN_KEY_UNKNOWN`` is at the same location, and
       ``ANCHOR_SOURCE_HASH_MISMATCH`` is dropped where ``DATASET_SPAN_HASH_MISMATCH`` is at the
       same location and the supplied text hashes to the manifest's recorded hash (if the text
       matches neither hash, both issues stay).

    Locations are ``dev.questions[i]`` / ``test.questions[i]`` (position within the collection)
    plus fixed field names and indexes; no issue contains question text, identifiers, corpus
    keys or document text. Locations of the duplicate and similarity issues also use the original
    collection index, including when earlier items are not questions or a question's ``split``
    differs from its collection.
    """
    # Imported here so that importing this module stays light (see the import-isolation test).
    from app.rag.evaluation import near_duplicates
    from app.rag.evaluation.anchoring import ANCHOR_SOURCE_HASH_MISMATCH, ANCHOR_TEXT_MISSING, resolve_question
    from app.rag.evaluation.intervals import collides
    from app.rag.evaluation.schema import validate_span
    from app.rag.evaluation.validation import validate_manifest

    # Phase 0: arguments.
    if not isinstance(questions, Mapping):
        raise TypeError("questions must be a Mapping")
    for name in ("dev", "test"):
        if name in questions:
            collection = questions[name]
            if isinstance(collection, (str, bytes)) or not isinstance(collection, Sequence):
                raise TypeError("each questions collection must be a sequence of Question")
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    if texts is not None and not isinstance(texts, Mapping):
        raise TypeError("texts must be a Mapping or None")
    near_duplicates.find_duplicate_questions([], threshold=near_duplicate_threshold)  # validates the threshold

    collections = {name: questions[name] if name in questions else () for name in ("dev", "test")}
    issues: list[Issue] = []

    # Phase A: split shape.
    if set(questions) != {"dev", "test"}:
        issues.append(_dataset_issue(DATASET_SPLITS, "dataset"))

    # Phase B: manifest gate.
    gate_open = False
    documents: dict[str, object] = {}
    if any(item.severity == "error" for item in validate_manifest(manifest)):
        issues.append(_dataset_issue(DATASET_MANIFEST_INVALID, "manifest"))
    elif manifest.stage != "ingested":
        issues.append(_dataset_issue(DATASET_MANIFEST_NOT_INGESTED, "manifest.stage"))
    else:
        gate_open = True
        documents = {document.corpus_key: document for document in manifest.documents}

    # Phase C: each question.
    recorded_hashes: dict[str, tuple[str, object]] = {}  # span source_sha256 location -> (key, manifest hash)
    for split, collection in collections.items():
        for i, q in enumerate(collection):
            location = f"{split}.questions[{i}]"
            if not isinstance(q, Question):
                issues.append(_dataset_issue(DATASET_QUESTION_TYPE, location))
                continue

            # C1: record rules and split consistency.
            issues.extend(validate_question(q, location=location))
            if _is_str(q.split) and q.split in SPLITS and q.split != split:
                issues.append(_dataset_issue(DATASET_SPLIT_MISMATCH, f"{location}.split"))
            if not _is_checkable_answerable(q):
                continue

            # C2: span validation, one mode per question.
            if texts is not None:
                issues.extend(resolve_question(q, texts, location=location))
            else:
                for span_location, span in _located_spans(q, location):
                    issues.extend(validate_span(span, location=span_location))

            # C3: manifest references and languages.
            if not gate_open:
                continue
            languages = set()
            for span_location, span in _located_spans(q, location):
                if not _is_str(span.corpus_key):
                    continue  # an ill-typed span is reported by the span phase
                document = documents.get(span.corpus_key)
                if document is None:
                    issues.append(_dataset_issue(DATASET_SPAN_KEY_UNKNOWN, f"{span_location}.corpus_key"))
                    continue
                languages.add(document.language)
                if _is_str(span.source_sha256) and span.source_sha256 != document.extracted_text_sha256:
                    hash_location = f"{span_location}.source_sha256"
                    issues.append(_dataset_issue(DATASET_SPAN_HASH_MISMATCH, hash_location))
                    recorded_hashes[hash_location] = (span.corpus_key, document.extracted_text_sha256)
            if len(languages) > 1:
                issues.append(_dataset_issue(DATASET_MIXED_LANGUAGE_SPANS, f"{location}.units"))
            elif len(languages) == 1 and _is_str(q.source_language) and q.source_language in LANGUAGES:
                if q.source_language not in languages:
                    issues.append(_dataset_issue(DATASET_SOURCE_LANGUAGE_MISMATCH, f"{location}.source_language"))

    # Phase D1: duplicate question ids, dev before test.
    seen_ids: set[str] = set()
    for split, collection in collections.items():
        for i, q in enumerate(collection):
            if not isinstance(q, Question) or not _is_str(q.question_id):
                continue
            if q.question_id in seen_ids:
                issues.append(_dataset_issue(DATASET_DUPLICATE_QUESTION_ID, f"{split}.questions[{i}].question_id"))
            else:
                seen_ids.add(q.question_id)

    # Phase D2: dev/test evidence collisions, one error per colliding test span.
    dev_intervals: dict[str, list[tuple[int, int]]] = {}
    for i, q in enumerate(collections["dev"]):
        if _is_checkable_answerable(q):
            for _, span in _located_spans(q, f"dev.questions[{i}]"):
                if _valid_interval(span):
                    dev_intervals.setdefault(span.corpus_key, []).append((span.start, span.end))
    for i, q in enumerate(collections["test"]):
        if _is_checkable_answerable(q):
            for span_location, span in _located_spans(q, f"test.questions[{i}]"):
                if _valid_interval(span) and any(
                    collides((span.start, span.end), other) for other in dev_intervals.get(span.corpus_key, ())
                ):
                    issues.append(_dataset_issue(DATASET_SPLIT_COLLISION, span_location))

    # Phases D3 and D4: exact duplicate text, then similar questions (the step 2 detector, unchanged).
    # The detector derives each position from the question's own split and does not count items
    # that are not questions. So it is given one entry per collection item, in collection order,
    # with ``split`` set to the collection: every location then uses the original collection index.
    detector_input = [
        dataclasses.replace(q, split=name) if isinstance(q, Question) else _placeholder_question(name)
        for name in ("dev", "test")
        for q in collections[name]
    ]
    found = near_duplicates.find_duplicate_questions(detector_input, threshold=near_duplicate_threshold)
    issues.extend(item for item in found if item.code == near_duplicates.DATASET_DUPLICATE_QUESTION_TEXT)
    issues.extend(item for item in found if item.code != near_duplicates.DATASET_DUPLICATE_QUESTION_TEXT)

    # Phase E: deduplicate by (code, location), then the two suppressions.
    key_unknown = {item.location for item in issues if item.code == DATASET_SPAN_KEY_UNKNOWN}
    seen: set[tuple[str, str]] = set()
    hash_cache: dict[str, str | None] = {}
    result: list[Issue] = []
    for item in issues:
        identity = (item.code, item.location)
        if identity in seen:
            continue
        seen.add(identity)
        if item.code == ANCHOR_TEXT_MISSING and item.location in key_unknown:
            continue
        if item.code == ANCHOR_SOURCE_HASH_MISMATCH and item.location in recorded_hashes:
            key, recorded = recorded_hashes[item.location]
            digest = _supplied_text_hash(texts, key, hash_cache)
            if digest is not None and digest == recorded:
                continue
        result.append(item)
    return result
