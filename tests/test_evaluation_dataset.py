"""E1.2d step 3: record-level rules (validate_question).

A  valid questions
B  common fields
C  dates
D  answerable and unanswerable applicability
E  retrieval assistance
F  Hindi review matrix
G  units and spans
H  malformed types
I  severity, messages and privacy
J  determinism, no mutation, arguments
L  Hindi rules with an invalid or wrong-typed query_language
M  issue order: split rule, Hindi rules, kind fields
N  lone surrogates and very long strings
O  nested field and container types
P  is_valid_iso_date
Q  validate_dataset: arguments and split shape
R  manifest gate
S  per-question order and split consistency
T  manifest references and languages
U  the span phase: one mode per question
V  suppression and deduplication
W  dataset-level phases
X  whole-result properties
K  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import dataset as d
from app.rag.evaluation import near_duplicates as nd
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHA = "a" * 64
CANARY = "SECRET-CANARY-7731"
HINDI = chr(0x0939) + chr(0x093F) + chr(0x0928) + chr(0x094D) + chr(0x0926) + chr(0x0940)
ALL_CODES = (
    "Q_FIELD_TYPE Q_QUESTION_ID Q_ENUM Q_BLANK Q_DATE Q_DESCRIPTION_LENGTH Q_FIELD_MISSING Q_INAPPLICABLE_FIELD "
    "Q_UNANSWERABLE_SPLIT Q_UNIT_COUNT Q_UNITS_TYPE Q_UNIT_TYPE Q_SPANS_TYPE Q_SPAN_TYPE Q_UNIT_ID_DUPLICATE "
    "Q_UNIT_NO_SPANS Q_SPAN_DUPLICATE_IN_UNIT Q_SPAN_REUSED_ACROSS_UNITS Q_HINDI_REVIEW_REQUIRED "
    "Q_HINDI_REVIEW_INAPPLICABLE Q_HINDI_REVIEWER_REQUIRED Q_HINDI_REVIEWER_INAPPLICABLE"
).split()


def span(key="doc-a", start=0, end=5, **overrides):
    values = dict(corpus_key=key, source_sha256=SHA, start=start, end=end, quote="quote", quote_sha256=SHA)
    values.update(overrides)
    return s.Span(**values)


def unit(unit_id="u1", spans=None):
    return s.EvidenceUnit(unit_id, (span(),) if spans is None else spans)


def answerable(**overrides):
    values = dict(
        question_id="q1",
        split="dev",
        answerable=True,
        question="What is the notice period?",
        query_language="en",
        author_type="human",
        hindi_reviewed=False,
        expected_answer="Thirty days.",
        units=(unit(),),
        source_language="en",
        difficulty="direct",
        close_distractor=False,
    )
    values.update(overrides)
    return s.Question(**values)


def review(**overrides):
    values = dict(reviewer="reader", date="2026-11-11", result="confirmed_unanswerable")
    values.update(overrides)
    return s.SemanticReview(**values)


def assistance(**overrides):
    values = dict(date="2026-11-11", method=None, run_ref=None, description="looked at the top results")
    values.update(overrides)
    return s.RetrievalAssistance(**values)


def unanswerable(**overrides):
    values = dict(
        question_id="q2",
        split="dev",
        answerable=False,
        question="What is the refund policy?",
        query_language="en",
        author_type="human",
        hindi_reviewed=False,
        unanswerable_type="off_topic",
        semantic_review=review(),
    )
    values.update(overrides)
    return s.Question(**values)


def hindi(base, **overrides):
    """The same question as a Hindi query with a valid review."""
    values = dict(query_language="hi", hindi_reviewed=True, hindi_reviewer="reviewer", question=HINDI)
    values.update(overrides)
    return dataclasses.replace(base, **values)


def check(q, location="question"):
    return [(i.code, i.location) for i in d.validate_question(q, location=location)]


def codes(q):
    return [code for code, _ in check(q)]


# ---------- A. valid questions ----------

@pytest.mark.parametrize(
    "q",
    [
        pytest.param(answerable(), id="answerable"),
        pytest.param(unanswerable(), id="unanswerable"),
        pytest.param(unanswerable(retrieval_assistance=assistance(method="m", run_ref="r")), id="unanswerable-assisted"),
        pytest.param(answerable(notes="a note", units=(unit(spans=(span(), span(start=7, end=9))),)), id="notes-and-alternatives"),
        pytest.param(hindi(answerable()), id="hindi-answerable"),
        pytest.param(hindi(unanswerable()), id="hindi-unanswerable"),
        pytest.param(answerable(source_language="hi"), id="english-query-hindi-source"),
        pytest.param(answerable(question_id="0", split="test"), id="test-split-single-char-id"),
    ],
)
def test_valid_questions_have_no_issues(q):
    assert d.validate_question(q) == []


def test_the_issue_codes_are_exactly_the_approved_set():
    assert sorted(d._MESSAGES) == sorted(ALL_CODES)
    for code in ALL_CODES:
        assert getattr(d, code) == code


# ---------- B. common fields ----------

@pytest.mark.parametrize(
    "overrides, code, field",
    [
        pytest.param({"question_id": "Q1"}, "Q_QUESTION_ID", "question_id", id="id-uppercase"),
        pytest.param({"question_id": ""}, "Q_QUESTION_ID", "question_id", id="id-empty"),
        pytest.param({"question_id": "q1\n"}, "Q_QUESTION_ID", "question_id", id="id-trailing-newline"),
        pytest.param({"question_id": "-q1"}, "Q_QUESTION_ID", "question_id", id="id-leading-hyphen"),
        pytest.param({"question_id": "q" * 65}, "Q_QUESTION_ID", "question_id", id="id-too-long"),
        pytest.param({"question": ""}, "Q_BLANK", "question", id="question-empty"),
        pytest.param({"question": " \t\n"}, "Q_BLANK", "question", id="question-whitespace"),
        pytest.param({"split": "train"}, "Q_ENUM", "split", id="split-unknown"),
        pytest.param({"query_language": "fr"}, "Q_ENUM", "query_language", id="language-unknown"),
        pytest.param({"author_type": "robot"}, "Q_ENUM", "author_type", id="author-unknown"),
        pytest.param({"notes": ""}, "Q_BLANK", "notes", id="notes-empty"),
        pytest.param({"notes": "  "}, "Q_BLANK", "notes", id="notes-whitespace"),
        pytest.param({"notes": 5}, "Q_FIELD_TYPE", "notes", id="notes-int"),
    ],
)
def test_common_field_rules(overrides, code, field):
    assert check(answerable(**overrides)) == [(code, f"question.{field}")]


def test_a_valid_question_id_of_64_characters_is_accepted():
    assert check(answerable(question_id="a" + "-" * 63)) == []


@pytest.mark.parametrize("language", ["en", "hi"])
@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("author", ["human", "llm_assisted"])
def test_every_valid_enum_value_is_accepted(language, split, author):
    q = answerable(split=split, author_type=author)
    q = hindi(q) if language == "hi" else q
    assert check(q) == []


def test_the_caller_location_prefix_is_used_everywhere():
    q = answerable(question_id="BAD", units=(unit(spans=()),), source_language="xx")
    assert check(q, location="dev.questions[3]") == [
        ("Q_QUESTION_ID", "dev.questions[3].question_id"),
        ("Q_UNIT_NO_SPANS", "dev.questions[3].units[0].spans"),
        ("Q_ENUM", "dev.questions[3].source_language"),
    ]


# ---------- C. dates ----------

DATE_FIELDS = [
    pytest.param(lambda v: unanswerable(semantic_review=review(date=v)), "question.semantic_review.date", id="semantic-review"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(date=v)), "question.retrieval_assistance.date", id="retrieval-assistance"),
]


@pytest.mark.parametrize("build, location", DATE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("2026-02-29", id="not-a-leap-year"),
        pytest.param("2026-13-01", id="month-13"),
        pytest.param("2026-00-10", id="month-0"),
        pytest.param("2026-04-31", id="day-31-of-april"),
        pytest.param("0000-01-01", id="year-0"),
        pytest.param("2026-1-1", id="not-zero-padded"),
        pytest.param("2026/11/11", id="slashes"),
        pytest.param("2026-11-11\n", id="trailing-newline"),
        pytest.param(" 2026-11-11", id="leading-space"),
        pytest.param("", id="empty"),
        pytest.param(chr(0x0662) + "026-11-11", id="arabic-indic-digit"),
        pytest.param(chr(0xFF12) + "026-11-11", id="fullwidth-digit"),
    ],
)
def test_an_invalid_date_is_reported(build, location, value):
    assert check(build(value)) == [("Q_DATE", location)]


@pytest.mark.parametrize("build, location", DATE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("2024-02-29", id="leap-day"),
        pytest.param("2999-12-31", id="future-date-allowed"),
        pytest.param("0001-01-01", id="year-1"),
    ],
)
def test_a_valid_date_is_accepted(build, location, value):
    assert check(build(value)) == []


@pytest.mark.parametrize("build, location", DATE_FIELDS)
def test_a_non_string_date_is_a_type_error_issue(build, location):
    assert check(build(20261111)) == [("Q_FIELD_TYPE", location)]


# ---------- D. answerable and unanswerable applicability ----------

@pytest.mark.parametrize(
    "field",
    [
        pytest.param("expected_answer", id="expected-answer"),
        pytest.param("source_language", id="source-language"),
        pytest.param("difficulty", id="difficulty"),
        pytest.param("close_distractor", id="close-distractor"),
    ],
)
def test_an_answerable_question_must_have_its_answerable_fields(field):
    assert check(answerable(**{field: None})) == [("Q_FIELD_MISSING", f"question.{field}")]


@pytest.mark.parametrize(
    "overrides, code, field",
    [
        pytest.param({"expected_answer": ""}, "Q_BLANK", "expected_answer", id="answer-blank"),
        pytest.param({"expected_answer": 5}, "Q_FIELD_TYPE", "expected_answer", id="answer-int"),
        pytest.param({"source_language": "fr"}, "Q_ENUM", "source_language", id="source-language-unknown"),
        pytest.param({"difficulty": "hard"}, "Q_ENUM", "difficulty", id="difficulty-unknown"),
        pytest.param({"close_distractor": 1}, "Q_FIELD_TYPE", "close_distractor", id="distractor-int"),
    ],
)
def test_answerable_field_values(overrides, code, field):
    assert check(answerable(**overrides)) == [(code, f"question.{field}")]


@pytest.mark.parametrize(
    "overrides, field",
    [
        pytest.param({"unanswerable_type": "off_topic"}, "unanswerable_type", id="unanswerable-type"),
        pytest.param({"semantic_review": review()}, "semantic_review", id="semantic-review"),
        pytest.param({"retrieval_assistance": assistance()}, "retrieval_assistance", id="retrieval-assistance"),
    ],
)
def test_unanswerable_fields_do_not_apply_to_an_answerable_question(overrides, field):
    assert check(answerable(**overrides)) == [("Q_INAPPLICABLE_FIELD", f"question.{field}")]


def test_inapplicable_content_is_not_validated():
    q = answerable(retrieval_assistance=assistance(date="bad", description=""), semantic_review=review(date="bad"))
    assert check(q) == [
        ("Q_INAPPLICABLE_FIELD", "question.semantic_review"),
        ("Q_INAPPLICABLE_FIELD", "question.retrieval_assistance"),
    ]


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("unanswerable_type", id="unanswerable-type"),
        pytest.param("semantic_review", id="semantic-review"),
    ],
)
def test_an_unanswerable_question_must_have_its_fields(field):
    assert check(unanswerable(**{field: None})) == [("Q_FIELD_MISSING", f"question.{field}")]


@pytest.mark.parametrize(
    "overrides, field",
    [
        pytest.param({"expected_answer": "x"}, "expected_answer", id="expected-answer"),
        pytest.param({"source_language": "en"}, "source_language", id="source-language"),
        pytest.param({"difficulty": "direct"}, "difficulty", id="difficulty"),
        pytest.param({"close_distractor": False}, "close_distractor", id="close-distractor"),
        pytest.param({"units": (unit(),)}, "units", id="units"),
    ],
)
def test_answerable_fields_do_not_apply_to_an_unanswerable_question(overrides, field):
    assert check(unanswerable(**overrides)) == [("Q_INAPPLICABLE_FIELD", f"question.{field}")]


@pytest.mark.parametrize(
    "overrides, code, field",
    [
        pytest.param({"unanswerable_type": "other"}, "Q_ENUM", "unanswerable_type", id="type-unknown"),
        pytest.param({"unanswerable_type": 3}, "Q_FIELD_TYPE", "unanswerable_type", id="type-int"),
        pytest.param({"semantic_review": "reviewed"}, "Q_FIELD_TYPE", "semantic_review", id="review-wrong-class"),
    ],
)
def test_unanswerable_field_values(overrides, code, field):
    assert check(unanswerable(**overrides)) == [(code, f"question.{field}")]


@pytest.mark.parametrize("kind", ["off_topic", "on_topic_not_covered"])
def test_both_unanswerable_types_are_accepted(kind):
    assert check(unanswerable(unanswerable_type=kind)) == []


@pytest.mark.parametrize(
    "overrides, code, field",
    [
        pytest.param({"reviewer": ""}, "Q_BLANK", "reviewer", id="reviewer-blank"),
        pytest.param({"reviewer": None}, "Q_FIELD_TYPE", "reviewer", id="reviewer-none"),
        pytest.param({"result": "unsure"}, "Q_ENUM", "result", id="result-unknown"),
        pytest.param({"result": None}, "Q_FIELD_TYPE", "result", id="result-none"),
    ],
)
def test_semantic_review_fields(overrides, code, field):
    q = unanswerable(semantic_review=review(**overrides))
    assert check(q) == [(code, f"question.semantic_review.{field}")]


def test_an_unanswerable_question_must_be_in_dev():
    assert check(unanswerable(split="test")) == [("Q_UNANSWERABLE_SPLIT", "question.split")]


def test_an_unknown_split_is_only_an_enum_issue():
    assert check(unanswerable(split="train")) == [("Q_ENUM", "question.split")]


def test_an_answerable_question_may_be_in_either_split():
    assert check(answerable(split="test")) == [] and check(answerable(split="dev")) == []


# ---------- E. retrieval assistance ----------

def test_retrieval_assistance_may_be_absent_for_an_unanswerable_question():
    assert unanswerable().retrieval_assistance is None
    assert check(unanswerable()) == []


@pytest.mark.parametrize("length, expected", [(500, []), (501, [("Q_DESCRIPTION_LENGTH", "question.retrieval_assistance.description")])])
def test_description_length_is_counted_in_code_points(length, expected):
    assert check(unanswerable(retrieval_assistance=assistance(description="x" * length))) == expected


def test_description_length_counts_code_points_not_bytes():
    assert check(unanswerable(retrieval_assistance=assistance(description=HINDI[0] * 500))) == []
    assert codes(unanswerable(retrieval_assistance=assistance(description=HINDI[0] * 501))) == ["Q_DESCRIPTION_LENGTH"]


@pytest.mark.parametrize(
    "overrides, code, field",
    [
        pytest.param({"description": ""}, "Q_BLANK", "description", id="description-empty"),
        pytest.param({"description": "  "}, "Q_BLANK", "description", id="description-whitespace"),
        pytest.param({"description": 5}, "Q_FIELD_TYPE", "description", id="description-int"),
        pytest.param({"description": None}, "Q_FIELD_TYPE", "description", id="description-none"),
        pytest.param({"method": ""}, "Q_BLANK", "method", id="method-empty"),
        pytest.param({"method": 5}, "Q_FIELD_TYPE", "method", id="method-int"),
        pytest.param({"run_ref": " "}, "Q_BLANK", "run_ref", id="run-ref-blank"),
        pytest.param({"run_ref": 5}, "Q_FIELD_TYPE", "run_ref", id="run-ref-int"),
    ],
)
def test_retrieval_assistance_fields(overrides, code, field):
    q = unanswerable(retrieval_assistance=assistance(**overrides))
    assert check(q) == [(code, f"question.retrieval_assistance.{field}")]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"method": None, "run_ref": None}, id="neither"),
        pytest.param({"method": "dense", "run_ref": None}, id="method-only"),
        pytest.param({"method": None, "run_ref": "run-1"}, id="run-ref-only"),
        pytest.param({"method": "dense", "run_ref": "run-1"}, id="both"),
    ],
)
def test_method_and_run_ref_are_independent(overrides):
    assert check(unanswerable(retrieval_assistance=assistance(**overrides))) == []


def test_retrieval_assistance_of_the_wrong_class_is_a_type_issue():
    assert check(unanswerable(retrieval_assistance="note")) == [("Q_FIELD_TYPE", "question.retrieval_assistance")]


# ---------- F. Hindi review matrix ----------

HINDI_ROWS = [
    # language, hindi_reviewed, hindi_reviewer, expected codes
    pytest.param("hi", True, "reviewer", [], id="hi-valid"),
    pytest.param("hi", False, "reviewer", ["Q_HINDI_REVIEW_REQUIRED"], id="hi-not-reviewed"),
    pytest.param("hi", True, None, ["Q_HINDI_REVIEWER_REQUIRED"], id="hi-no-reviewer"),
    pytest.param("hi", True, "", ["Q_HINDI_REVIEWER_REQUIRED"], id="hi-empty-reviewer"),
    pytest.param("hi", True, "  ", ["Q_HINDI_REVIEWER_REQUIRED"], id="hi-blank-reviewer"),
    pytest.param("hi", False, None, ["Q_HINDI_REVIEW_REQUIRED", "Q_HINDI_REVIEWER_REQUIRED"], id="hi-neither"),
    pytest.param("en", False, None, [], id="en-valid"),
    pytest.param("en", True, None, ["Q_HINDI_REVIEW_INAPPLICABLE"], id="en-reviewed"),
    pytest.param("en", False, "reviewer", ["Q_HINDI_REVIEWER_INAPPLICABLE"], id="en-reviewer"),
    pytest.param("en", False, "", ["Q_HINDI_REVIEWER_INAPPLICABLE"], id="en-empty-reviewer"),
    pytest.param("en", True, "reviewer", ["Q_HINDI_REVIEW_INAPPLICABLE", "Q_HINDI_REVIEWER_INAPPLICABLE"], id="en-both"),
]


@pytest.mark.parametrize("language, reviewed, reviewer, expected", HINDI_ROWS)
@pytest.mark.parametrize("kind", [pytest.param(answerable, id="answerable"), pytest.param(unanswerable, id="unanswerable")])
def test_hindi_review_matrix(kind, language, reviewed, reviewer, expected):
    q = kind(query_language=language, hindi_reviewed=reviewed, hindi_reviewer=reviewer)
    assert codes(q) == expected


def test_hindi_issue_locations():
    q = answerable(query_language="hi", hindi_reviewed=False, hindi_reviewer=None)
    assert check(q) == [
        ("Q_HINDI_REVIEW_REQUIRED", "question.hindi_reviewed"),
        ("Q_HINDI_REVIEWER_REQUIRED", "question.hindi_reviewer"),
    ]


@pytest.mark.parametrize("source", ["en", "hi"])
def test_the_source_language_does_not_trigger_hindi_review(source):
    assert check(answerable(source_language=source)) == []


def test_a_hindi_source_with_an_english_query_needs_no_review():
    assert check(answerable(query_language="en", source_language="hi", hindi_reviewed=False)) == []


def test_an_invalid_query_language_skips_the_hindi_rules():
    q = answerable(query_language="fr", hindi_reviewed=True, hindi_reviewer="reviewer")
    assert check(q) == [("Q_ENUM", "question.query_language")]


def test_a_non_string_query_language_skips_the_hindi_rules():
    q = answerable(query_language=None, hindi_reviewed=True)
    assert check(q) == [("Q_FIELD_TYPE", "question.query_language")]


def test_a_blank_reviewer_is_still_reported_when_no_hindi_rule_applies():
    assert check(answerable(query_language="fr", hindi_reviewer=" ")) == [
        ("Q_ENUM", "question.query_language"),
        ("Q_BLANK", "question.hindi_reviewer"),
    ]


def test_a_non_bool_hindi_reviewed_is_reported_once_and_its_rule_skipped():
    for language, reviewer in (("hi", "reviewer"), ("en", None)):
        q = answerable(query_language=language, hindi_reviewed=1, hindi_reviewer=reviewer)
        assert check(q) == [("Q_FIELD_TYPE", "question.hindi_reviewed")]


def test_a_reviewer_of_the_wrong_type_is_reported_once_and_its_rule_skipped():
    for language, reviewed in (("hi", True), ("en", False)):
        q = answerable(query_language=language, hindi_reviewed=reviewed, hindi_reviewer=5)
        assert check(q) == [("Q_FIELD_TYPE", "question.hindi_reviewer")]


def test_the_two_hindi_fields_are_judged_independently():
    q = answerable(query_language="hi", hindi_reviewed=1, hindi_reviewer=None)
    assert check(q) == [
        ("Q_FIELD_TYPE", "question.hindi_reviewed"),
        ("Q_HINDI_REVIEWER_REQUIRED", "question.hindi_reviewer"),
    ]


# ---------- G. units and spans ----------

@pytest.mark.parametrize(
    "count, expected",
    [
        pytest.param(0, ["Q_UNIT_COUNT"], id="0-units"),
        pytest.param(1, [], id="1-unit"),
        pytest.param(20, [], id="20-units"),
        pytest.param(21, ["Q_UNIT_COUNT"], id="21-units"),
    ],
)
def test_unit_count_limits(count, expected):
    units = tuple(unit(unit_id=f"u{i}", spans=(span(start=i, end=i + 1),)) for i in range(count))
    assert codes(answerable(units=units)) == expected


def test_unit_count_location():
    assert check(answerable(units=())) == [("Q_UNIT_COUNT", "question.units")]


def test_units_must_be_a_tuple():
    assert check(answerable(units=[unit()])) == [("Q_UNITS_TYPE", "question.units")]
    assert check(unanswerable(units=[])) == [("Q_UNITS_TYPE", "question.units")]
    assert check(answerable(units=None)) == [("Q_UNITS_TYPE", "question.units")]


def test_a_unit_must_be_an_evidence_unit():
    q = answerable(units=(unit(), "not a unit", unit(unit_id="u3", spans=(span(start=9, end=12),))))
    assert check(q) == [("Q_UNIT_TYPE", "question.units[1]")]


@pytest.mark.parametrize(
    "unit_id, expected",
    [
        pytest.param("", [("Q_BLANK", "question.units[0].unit_id")], id="empty"),
        pytest.param("  ", [("Q_BLANK", "question.units[0].unit_id")], id="whitespace"),
        pytest.param(5, [("Q_FIELD_TYPE", "question.units[0].unit_id")], id="int"),
        pytest.param(None, [("Q_FIELD_TYPE", "question.units[0].unit_id")], id="none"),
    ],
)
def test_unit_id_rules(unit_id, expected):
    assert check(answerable(units=(unit(unit_id=unit_id),))) == expected


def test_a_duplicate_unit_id_is_reported_at_the_later_unit():
    units = (unit("a", (span(start=0, end=1),)), unit("b", (span(start=2, end=3),)), unit("a", (span(start=4, end=5),)))
    assert check(answerable(units=units)) == [("Q_UNIT_ID_DUPLICATE", "question.units[2].unit_id")]


def test_unit_ids_are_compared_exactly():
    units = (unit("a", (span(start=0, end=1),)), unit("A", (span(start=2, end=3),)), unit("a ", (span(start=4, end=5),)))
    assert check(answerable(units=units)) == []


def test_a_unit_without_spans_is_reported():
    assert check(answerable(units=(unit(spans=()),))) == [("Q_UNIT_NO_SPANS", "question.units[0].spans")]


def test_spans_must_be_a_tuple():
    assert check(answerable(units=(unit(spans=[span()]),))) == [("Q_SPANS_TYPE", "question.units[0].spans")]


def test_a_span_must_be_a_span():
    q = answerable(units=(unit(spans=(span(), "not a span", span(start=7, end=9))),))
    assert check(q) == [("Q_SPAN_TYPE", "question.units[0].spans[1]")]


def test_a_span_repeated_within_a_unit_is_an_error_at_the_later_span():
    q = answerable(units=(unit(spans=(span(), span(start=7, end=9), span())),))
    issues = d.validate_question(q)
    assert [(i.code, i.location, i.severity) for i in issues] == [
        ("Q_SPAN_DUPLICATE_IN_UNIT", "question.units[0].spans[2]", "error")
    ]


def test_span_identity_ignores_the_other_span_fields():
    q = answerable(units=(unit(spans=(span(quote="one"), span(quote="two", quote_sha256="b" * 64))),))
    assert codes(q) == ["Q_SPAN_DUPLICATE_IN_UNIT"]


def test_spans_with_different_identities_are_not_duplicates():
    spans = (span(), span(key="doc-b"), span(start=1), span(end=6))
    assert check(answerable(units=(unit(spans=spans),))) == []


def test_the_same_span_in_two_units_is_a_warning_at_the_later_unit():
    units = (unit("a"), unit("b"))
    issues = d.validate_question(answerable(units=units))
    assert [(i.code, i.location, i.severity) for i in issues] == [
        ("Q_SPAN_REUSED_ACROSS_UNITS", "question.units[1].spans[0]", "warning")
    ]


def test_reuse_is_reported_once_for_each_later_span():
    units = (
        unit("a", (span(start=0, end=1), span(start=2, end=3))),
        unit("b", (span(start=0, end=1), span(start=5, end=6))),
        unit("c", (span(start=2, end=3),)),
    )
    assert check(answerable(units=units)) == [
        ("Q_SPAN_REUSED_ACROSS_UNITS", "question.units[1].spans[0]"),
        ("Q_SPAN_REUSED_ACROSS_UNITS", "question.units[2].spans[0]"),
    ]


def test_overlapping_spans_in_different_units_are_not_reported():
    units = (unit("a", (span(start=0, end=10),)), unit("b", (span(start=5, end=15),)))
    assert check(answerable(units=units)) == []


def test_ill_typed_span_fields_have_no_identity_and_no_issue_here():
    bad = (span(start="0"), span(start=True, end=True), span(key=5), span(end=1.5))
    assert check(answerable(units=(unit(spans=bad),))) == []
    assert check(answerable(units=(unit("a", bad), unit("b", bad)))) == []


def test_validate_span_is_never_called(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("validate_span must not be called")

    monkeypatch.setattr(s, "validate_span", boom)
    assert d.validate_question(answerable(units=(unit(spans=(span(start=9, end=2),)),))) == []


def test_the_units_of_a_question_of_unknown_kind_are_not_judged():
    q = answerable(answerable=None, units=(unit(spans=()),))
    assert check(q) == [("Q_FIELD_TYPE", "question.answerable")]


# ---------- H. malformed types ----------

@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("question_id", 5, id="question-id-int"),
        pytest.param("question", None, id="question-none"),
        pytest.param("split", None, id="split-none"),
        pytest.param("query_language", 5, id="language-int"),
        pytest.param("author_type", None, id="author-none"),
        pytest.param("answerable", "yes", id="answerable-str"),
        pytest.param("answerable", 1, id="answerable-int"),
        pytest.param("answerable", None, id="answerable-none"),
        pytest.param("hindi_reviewed", "no", id="reviewed-str"),
        pytest.param("hindi_reviewer", 5, id="reviewer-int"),
        pytest.param("notes", ["x"], id="notes-list"),
    ],
)
def test_wrong_typed_common_fields_are_reported_once(field, value):
    assert check(answerable(**{field: value})) == [("Q_FIELD_TYPE", f"question.{field}")]


class StrSubclass(str):
    pass


@pytest.mark.parametrize(
    "field",
    ["question_id", "question", "split", "query_language", "author_type", "expected_answer", "source_language", "difficulty"],
)
def test_a_str_subclass_is_a_type_issue(field):
    value = StrSubclass(getattr(answerable(), field))
    assert check(answerable(**{field: value})) == [("Q_FIELD_TYPE", f"question.{field}")]


def test_a_validation_run_never_raises_on_garbage_fields():
    q = s.Question(
        question_id=object(), split=object(), answerable=object(), question=object(), query_language=object(),
        author_type=object(), hindi_reviewed=object(), hindi_reviewer=object(), notes=object(),
        expected_answer=object(), units=object(), source_language=object(), difficulty=object(),
        close_distractor=object(), unanswerable_type=object(), semantic_review=object(), retrieval_assistance=object(),
    )
    assert {i.code for i in d.validate_question(q)} == {"Q_FIELD_TYPE"}


@pytest.mark.parametrize("answerable_flag", [True, False])
def test_garbage_in_nested_objects_never_raises(answerable_flag):
    q = answerable(units=(s.EvidenceUnit(object(), object()), s.EvidenceUnit("u", (s.Span(None, None, None, None, None, None),))))
    assert isinstance(d.validate_question(q), list)
    q = unanswerable(semantic_review=s.SemanticReview(None, None, None), retrieval_assistance=s.RetrievalAssistance(None, 1, 2, None))
    assert {i.code for i in d.validate_question(q)} == {"Q_FIELD_TYPE"}


# ---------- I. severity, messages and privacy ----------

def test_only_span_reuse_is_a_warning():
    assert d._WARNING_CODES == {"Q_SPAN_REUSED_ACROSS_UNITS"}
    assert set(d._MESSAGES) == set(ALL_CODES)


def _many_issue_questions():
    return [
        answerable(question_id="BAD", question=" ", split="x", query_language="x", author_type="x", notes=" "),
        answerable(units=(unit("a"), unit("a", (span(start=0, end=5),)), unit("b", ()), unit("c", (span(), span())))),
        answerable(units=(), expected_answer=None, source_language=None, difficulty=None, close_distractor=None,
                   unanswerable_type="off_topic", semantic_review=review(), retrieval_assistance=assistance()),
        unanswerable(split="test", expected_answer="x", units=(unit(),), source_language="en", difficulty="direct",
                     close_distractor=True, retrieval_assistance=assistance(date="x", description="x" * 501, method=" ", run_ref=" ")),
        unanswerable(unanswerable_type=None, semantic_review=review(reviewer=" ", date="x", result="x")),
        hindi(answerable(), hindi_reviewed=False, hindi_reviewer=None),
        answerable(hindi_reviewed=True, hindi_reviewer="x"),
        answerable(answerable=None, hindi_reviewed=None, hindi_reviewer=3, notes=3),
        answerable(units=(unit("a"), "not a unit", unit("b", [span()]), unit("c", ("not a span",)))),
        answerable(units=[unit()]),
    ]


def test_every_emitted_message_is_the_fixed_text_for_its_code_and_every_code_is_emitted():
    emitted = set()
    for q in _many_issue_questions():
        for issue in d.validate_question(q):
            assert issue.message == d._MESSAGES[issue.code]
            emitted.add(issue.code)
    assert emitted == set(ALL_CODES)


def test_no_user_controlled_value_appears_in_any_issue():
    q = answerable(
        question_id=CANARY, question=" ", expected_answer=CANARY.lower() + " ", notes=" ",
        query_language="hi", hindi_reviewed=True, hindi_reviewer=CANARY,
        units=(
            unit(CANARY, (span(key=CANARY, start=0, end=1, quote=CANARY),)),
            unit(CANARY, (span(key=CANARY, start=0, end=1, quote=CANARY), span(key=CANARY, start=0, end=1))),
        ),
    )
    q2 = unanswerable(
        question=CANARY, semantic_review=review(reviewer=CANARY, date=CANARY, result=CANARY),
        retrieval_assistance=assistance(date=CANARY, method=CANARY, run_ref=CANARY, description=CANARY * 100),
    )
    for question in (q, q2):
        issues = d.validate_question(question, location="dev.questions[4]")
        assert issues
        for issue in issues:
            text = repr(issue).lower()
            assert "canary" not in text and "secret" not in text


def test_locations_contain_only_field_names_and_indexes():
    allowed = set("abcdefghijklmnopqrstuvwxyz_[].0123456789")
    for q in _many_issue_questions():
        for issue in d.validate_question(q, location="dev.questions[4]"):
            assert set(issue.location) <= allowed, issue.location


# ---------- J. determinism, no mutation, arguments ----------

def test_results_are_deterministic_and_the_question_is_not_modified():
    for q in _many_issue_questions():
        before = dataclasses.replace(q)
        first = d.validate_question(q)
        assert d.validate_question(q) == first
        assert q == before


def test_issue_order_is_fixed():
    q = answerable(
        question_id="BAD", query_language="hi", hindi_reviewed=False, hindi_reviewer=None, difficulty=None,
        units=(unit(spans=()),),
    )
    assert codes(q) == [
        "Q_QUESTION_ID", "Q_HINDI_REVIEW_REQUIRED", "Q_HINDI_REVIEWER_REQUIRED", "Q_UNIT_NO_SPANS", "Q_FIELD_MISSING",
    ]


@pytest.mark.parametrize("bad", [None, "question", {}, [], 5, answerable().units], ids=["none", "str", "dict", "list", "int", "tuple"])
def test_a_non_question_is_a_type_error(bad):
    with pytest.raises(TypeError, match="^q must be a Question$"):
        d.validate_question(bad)


@pytest.mark.parametrize("bad", [None, 5, b"question", ["question"]], ids=["none", "int", "bytes", "list"])
def test_a_non_string_location_is_a_type_error(bad):
    with pytest.raises(TypeError, match="^location must be a str$"):
        d.validate_question(answerable(), location=bad)


def test_location_is_keyword_only_and_defaults_to_question():
    with pytest.raises(TypeError):
        d.validate_question(answerable(question_id="BAD"), "x")
    assert check(answerable(question_id="BAD")) == [("Q_QUESTION_ID", "question.question_id")]


# ---------- L. Hindi rules: invalid or wrong-typed query_language ----------
# The language-dependent rules need a valid query_language. The general rule that a reviewer
# string is not blank applies exactly when they do not.

BAD_LANGUAGES = [
    pytest.param("fr", "Q_ENUM", id="language-unknown"),
    pytest.param("", "Q_ENUM", id="language-empty"),
    pytest.param("EN", "Q_ENUM", id="language-wrong-case"),
    pytest.param("hi ", "Q_ENUM", id="language-trailing-space"),
    pytest.param(None, "Q_FIELD_TYPE", id="language-none"),
    pytest.param(5, "Q_FIELD_TYPE", id="language-int"),
    pytest.param(b"hi", "Q_FIELD_TYPE", id="language-bytes"),
    pytest.param(StrSubclass("hi"), "Q_FIELD_TYPE", id="language-str-subclass"),
]
REVIEWERS = [
    pytest.param(None, [], id="reviewer-absent"),
    pytest.param("Asha", [], id="reviewer-nonblank"),
    pytest.param("", [("Q_BLANK", "question.hindi_reviewer")], id="reviewer-empty"),
    pytest.param("  ", [("Q_BLANK", "question.hindi_reviewer")], id="reviewer-whitespace"),
    pytest.param("\t\n", [("Q_BLANK", "question.hindi_reviewer")], id="reviewer-tab-newline"),
]


@pytest.mark.parametrize("language, language_code", BAD_LANGUAGES)
@pytest.mark.parametrize("reviewer, reviewer_issues", REVIEWERS)
@pytest.mark.parametrize("reviewed", [pytest.param(True, id="reviewed"), pytest.param(False, id="not-reviewed")])
@pytest.mark.parametrize("kind", [pytest.param(answerable, id="answerable"), pytest.param(unanswerable, id="unanswerable")])
def test_with_a_bad_language_no_hindi_rule_applies_but_a_reviewer_string_must_be_nonblank(
    kind, reviewed, reviewer, reviewer_issues, language, language_code
):
    q = kind(query_language=language, hindi_reviewed=reviewed, hindi_reviewer=reviewer)
    assert check(q) == [(language_code, "question.query_language")] + reviewer_issues


@pytest.mark.parametrize("reviewer", [pytest.param("", id="empty"), pytest.param("  ", id="whitespace")])
def test_with_a_valid_language_a_blank_reviewer_is_a_hindi_issue_never_q_blank(reviewer):
    assert codes(answerable(query_language="hi", hindi_reviewed=True, hindi_reviewer=reviewer)) == ["Q_HINDI_REVIEWER_REQUIRED"]
    assert codes(answerable(query_language="en", hindi_reviewed=False, hindi_reviewer=reviewer)) == ["Q_HINDI_REVIEWER_INAPPLICABLE"]


@pytest.mark.parametrize("language, language_code", BAD_LANGUAGES)
def test_a_wrong_typed_reviewer_is_one_type_issue_whatever_the_language(language, language_code):
    q = answerable(query_language=language, hindi_reviewed=True, hindi_reviewer=5)
    assert check(q) == [(language_code, "question.query_language"), ("Q_FIELD_TYPE", "question.hindi_reviewer")]


# ---------- M. issue order: split rule, Hindi rules, kind fields ----------

def test_the_unanswerable_split_rule_comes_right_after_the_split_check_and_before_the_hindi_rules():
    q = unanswerable(split="test", query_language="hi", hindi_reviewed=False, hindi_reviewer=None, unanswerable_type=None)
    assert check(q) == [
        ("Q_UNANSWERABLE_SPLIT", "question.split"),
        ("Q_HINDI_REVIEW_REQUIRED", "question.hindi_reviewed"),
        ("Q_HINDI_REVIEWER_REQUIRED", "question.hindi_reviewer"),
        ("Q_FIELD_MISSING", "question.unanswerable_type"),
    ]


def test_the_full_issue_order_common_fields_then_hindi_then_kind_fields():
    q = unanswerable(
        question_id="BAD", question=" ", split="test", author_type="x", notes=" ",
        query_language="hi", hindi_reviewed=False, hindi_reviewer=None,
        unanswerable_type="other", semantic_review=None,
    )
    assert check(q) == [
        ("Q_QUESTION_ID", "question.question_id"),
        ("Q_BLANK", "question.question"),
        ("Q_UNANSWERABLE_SPLIT", "question.split"),
        ("Q_ENUM", "question.author_type"),
        ("Q_BLANK", "question.notes"),
        ("Q_HINDI_REVIEW_REQUIRED", "question.hindi_reviewed"),
        ("Q_HINDI_REVIEWER_REQUIRED", "question.hindi_reviewer"),
        ("Q_ENUM", "question.unanswerable_type"),
        ("Q_FIELD_MISSING", "question.semantic_review"),
    ]


def test_an_exact_test_split_on_an_unanswerable_question_gives_the_split_issue():
    assert check(unanswerable(split="test")) == [("Q_UNANSWERABLE_SPLIT", "question.split")]


def test_a_str_subclass_split_is_only_a_type_issue_not_a_split_restriction_issue():
    assert check(unanswerable(split=StrSubclass("test"))) == [("Q_FIELD_TYPE", "question.split")]


def test_a_split_whose_equality_raises_does_not_make_validation_raise():
    class RaisingEq:
        def __eq__(self, other):
            raise RuntimeError("equality must not be used")

        __hash__ = None

    assert check(unanswerable(split=RaisingEq())) == [("Q_FIELD_TYPE", "question.split")]
    assert check(answerable(split=RaisingEq())) == [("Q_FIELD_TYPE", "question.split")]


def test_the_split_rule_needs_a_known_unanswerable_kind():
    assert check(answerable(answerable=None, split="test")) == [("Q_FIELD_TYPE", "question.answerable")]
    assert check(answerable(split="test")) == []


# ---------- N. lone surrogates and very long strings ----------

SURROGATE = chr(0xD800)
LONG = "x" * 100_000

TEXT_SLOTS = [
    pytest.param(lambda v: answerable(question=v), id="question"),
    pytest.param(lambda v: answerable(notes=v), id="notes"),
    pytest.param(lambda v: answerable(expected_answer=v), id="expected-answer"),
    pytest.param(lambda v: answerable(query_language="hi", hindi_reviewed=True, hindi_reviewer=v), id="hindi-reviewer"),
    pytest.param(lambda v: answerable(units=(unit(unit_id=v),)), id="unit-id"),
    pytest.param(lambda v: answerable(units=(unit(spans=(span(key=v),)),)), id="corpus-key"),
    pytest.param(lambda v: unanswerable(semantic_review=review(reviewer=v)), id="semantic-reviewer"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(method=v)), id="method"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(run_ref=v)), id="run-ref"),
]
CONSTRAINED_SLOTS = [
    pytest.param(lambda v: answerable(question_id=v), "Q_QUESTION_ID", "question_id", id="question-id"),
    pytest.param(lambda v: answerable(split=v), "Q_ENUM", "split", id="split"),
    pytest.param(lambda v: answerable(query_language=v), "Q_ENUM", "query_language", id="query-language"),
    pytest.param(lambda v: answerable(author_type=v), "Q_ENUM", "author_type", id="author-type"),
    pytest.param(lambda v: answerable(source_language=v), "Q_ENUM", "source_language", id="source-language"),
    pytest.param(lambda v: answerable(difficulty=v), "Q_ENUM", "difficulty", id="difficulty"),
    pytest.param(lambda v: unanswerable(unanswerable_type=v), "Q_ENUM", "unanswerable_type", id="unanswerable-type"),
    pytest.param(lambda v: unanswerable(semantic_review=review(result=v)), "Q_ENUM", "semantic_review.result", id="review-result"),
    pytest.param(lambda v: unanswerable(semantic_review=review(date=v)), "Q_DATE", "semantic_review.date", id="review-date"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(date=v)), "Q_DATE", "retrieval_assistance.date", id="assistance-date"),
]
UNUSUAL = [pytest.param(SURROGATE, id="lone-surrogate"), pytest.param(LONG, id="100000-characters")]


def _no_value_leaked(issues, value):
    for issue in issues:
        text = repr(issue)
        assert value not in text and "xxxxxxxx" not in text and "ud800" not in text.lower()


@pytest.mark.parametrize("value", UNUSUAL)
@pytest.mark.parametrize("build", TEXT_SLOTS)
def test_text_fields_accept_unusual_but_nonblank_strings(build, value):
    assert d.validate_question(build(value)) == []


@pytest.mark.parametrize("value", UNUSUAL)
@pytest.mark.parametrize("build, code, field", CONSTRAINED_SLOTS)
def test_constrained_fields_reject_unusual_strings_with_fixed_issues(build, code, field, value):
    issues = d.validate_question(build(value))
    assert [(i.code, i.location, i.message) for i in issues] == [(code, f"question.{field}", d._MESSAGES[code])]
    _no_value_leaked(issues, value)


@pytest.mark.parametrize("blank_slot", [pytest.param(lambda v: answerable(question=v), id="question"), pytest.param(lambda v: answerable(notes=v), id="notes")])
def test_a_very_long_whitespace_string_is_blank(blank_slot):
    assert codes(blank_slot(" " * 100_000)) == ["Q_BLANK"]


@pytest.mark.parametrize(
    "description, expected",
    [
        pytest.param(SURROGATE, [], id="one-surrogate"),
        pytest.param(SURROGATE * 500, [], id="500-surrogates"),
        pytest.param(SURROGATE * 501, ["Q_DESCRIPTION_LENGTH"], id="501-surrogates"),
        pytest.param(LONG, ["Q_DESCRIPTION_LENGTH"], id="100000-characters"),
    ],
)
def test_description_length_with_unusual_strings(description, expected):
    assert codes(unanswerable(retrieval_assistance=assistance(description=description))) == expected


def test_very_long_unit_ids_are_compared_without_error():
    units = (unit(LONG, (span(start=0, end=1),)), unit(LONG, (span(start=2, end=3),)))
    assert check(answerable(units=units)) == [("Q_UNIT_ID_DUPLICATE", "question.units[1].unit_id")]
    units = (unit(SURROGATE, (span(start=0, end=1),)), unit(SURROGATE, (span(start=2, end=3),)))
    assert check(answerable(units=units)) == [("Q_UNIT_ID_DUPLICATE", "question.units[1].unit_id")]


# ---------- O. nested field and container types ----------

WRONG_TYPES = [
    pytest.param(5, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(b"text", id="bytes"),
    pytest.param(["text"], id="list"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(object(), id="object"),
]
NESTED_STR_SLOTS = [
    pytest.param(lambda v: unanswerable(semantic_review=review(reviewer=v)), "semantic_review.reviewer", id="review-reviewer"),
    pytest.param(lambda v: unanswerable(semantic_review=review(date=v)), "semantic_review.date", id="review-date"),
    pytest.param(lambda v: unanswerable(semantic_review=review(result=v)), "semantic_review.result", id="review-result"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(date=v)), "retrieval_assistance.date", id="assistance-date"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(description=v)), "retrieval_assistance.description", id="assistance-description"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(method=v)), "retrieval_assistance.method", id="assistance-method"),
    pytest.param(lambda v: unanswerable(retrieval_assistance=assistance(run_ref=v)), "retrieval_assistance.run_ref", id="assistance-run-ref"),
    pytest.param(lambda v: answerable(units=(unit(unit_id=v),)), "units[0].unit_id", id="unit-id"),
]


@pytest.mark.parametrize("value", WRONG_TYPES)
@pytest.mark.parametrize("build, field", NESTED_STR_SLOTS)
def test_a_wrong_typed_nested_field_is_one_type_issue(build, field, value):
    assert check(build(value)) == [("Q_FIELD_TYPE", f"question.{field}")]


@pytest.mark.parametrize("value", [pytest.param(v, id=i) for v, i in [("text", "str"), (5, "int"), ({"date": "2026-11-11"}, "dict"), (assistance(), "other-class"), (unit(), "unit")]])
def test_a_semantic_review_of_the_wrong_class_is_a_type_issue(value):
    assert check(unanswerable(semantic_review=value)) == [("Q_FIELD_TYPE", "question.semantic_review")]


@pytest.mark.parametrize("value", [pytest.param(v, id=i) for v, i in [("text", "str"), (5, "int"), ({"date": "2026-11-11"}, "dict"), (review(), "other-class"), (unit(), "unit")]])
def test_retrieval_assistance_of_the_wrong_class_is_a_type_issue_for_any_value(value):
    assert check(unanswerable(retrieval_assistance=value)) == [("Q_FIELD_TYPE", "question.retrieval_assistance")]


@pytest.mark.parametrize("units", [pytest.param(v, id=i) for v, i in [(None, "none"), ("text", "str"), (5, "int"), ({}, "dict"), ([unit()], "list"), ({unit()}, "set"), (iter([unit()]), "iterator")]])
@pytest.mark.parametrize("kind", [pytest.param(answerable, id="answerable"), pytest.param(unanswerable, id="unanswerable")])
def test_units_of_any_non_tuple_type_is_one_issue(kind, units):
    assert check(kind(units=units)) == [("Q_UNITS_TYPE", "question.units")]


@pytest.mark.parametrize("bad", [pytest.param(v, id=i) for v, i in [(None, "none"), (5, "int"), ("unit", "str"), ({}, "dict"), (span(), "span"), ((unit(),), "tuple")]])
def test_a_unit_of_any_other_type_is_a_unit_type_issue(bad):
    assert check(answerable(units=(unit(), bad))) == [("Q_UNIT_TYPE", "question.units[1]")]


@pytest.mark.parametrize("bad", [pytest.param(v, id=i) for v, i in [(None, "none"), ("text", "str"), (5, "int"), ([span()], "list"), ({span()}, "set"), ({}, "dict")]])
def test_spans_of_any_non_tuple_type_is_a_spans_type_issue(bad):
    assert check(answerable(units=(s.EvidenceUnit("u1", bad),))) == [("Q_SPANS_TYPE", "question.units[0].spans")]


@pytest.mark.parametrize("bad", [pytest.param(v, id=i) for v, i in [(None, "none"), (5, "int"), ("span", "str"), ({}, "dict"), (unit(), "unit"), ((span(),), "tuple")]])
def test_a_span_of_any_other_type_is_a_span_type_issue(bad):
    assert check(answerable(units=(unit(spans=(span(), bad)),))) == [("Q_SPAN_TYPE", "question.units[0].spans[1]")]


# ---------- P. is_valid_iso_date ----------

@pytest.mark.parametrize(
    "value",
    [
        pytest.param("2026-11-11", id="ordinary"),
        pytest.param("2024-02-29", id="leap-day"),
        pytest.param("2000-02-29", id="leap-century"),
        pytest.param("0001-01-01", id="first-year"),
        pytest.param("9999-12-31", id="last-date"),
        pytest.param("2999-06-15", id="far-future"),
        pytest.param("2026-04-30", id="last-day-of-april"),
    ],
)
def test_valid_dates(value):
    assert d.is_valid_iso_date(value) is True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("2026-02-29", id="not-a-leap-year"),
        pytest.param("1900-02-29", id="not-a-leap-century"),
        pytest.param("2026-02-30", id="february-30"),
        pytest.param("2026-04-31", id="april-31"),
        pytest.param("2026-13-01", id="month-13"),
        pytest.param("2026-00-01", id="month-0"),
        pytest.param("2026-01-00", id="day-0"),
        pytest.param("2026-01-32", id="day-32"),
        pytest.param("0000-01-01", id="year-0"),
        pytest.param("0000-12-31", id="year-0-december"),
    ],
)
def test_invalid_calendar_dates(value):
    assert d.is_valid_iso_date(value) is False


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param("2026-1-1", id="not-zero-padded"),
        pytest.param("26-11-11", id="two-digit-year"),
        pytest.param("20261111", id="no-separators"),
        pytest.param("2026/11/11", id="slashes"),
        pytest.param("2026-11-11\n", id="trailing-newline"),
        pytest.param("2026-11-11 ", id="trailing-space"),
        pytest.param(" 2026-11-11", id="leading-space"),
        pytest.param("2026-11-11T00:00:00", id="datetime"),
        pytest.param("2026-11-111", id="three-digit-day"),
        pytest.param("+026-11-11", id="plus-sign"),
        pytest.param(SURROGATE, id="lone-surrogate"),
        pytest.param("2026-11-1" + SURROGATE, id="surrogate-in-date"),
        pytest.param(LONG, id="100000-characters"),
        pytest.param("2026-11-11" + "x" * 100_000, id="valid-prefix-then-long-tail"),
    ],
)
def test_malformed_dates(value):
    assert d.is_valid_iso_date(value) is False


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(chr(0x0662) + chr(0x0660) + "26-11-11", id="arabic-indic-digits"),
        pytest.param(chr(0x0966) + "026-11-11", id="devanagari-digit"),
        pytest.param(chr(0xFF12) + "026-11-11", id="fullwidth-digit"),
        pytest.param("2026-1" + chr(0x00B9) + "-11", id="superscript-one"),
        pytest.param("2026" + chr(0x2212) + "11-11", id="minus-sign-separator"),
    ],
)
def test_only_ascii_digits_are_accepted(value):
    assert d.is_valid_iso_date(value) is False


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param(20261111, id="int"),
        pytest.param(2026.1111, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(b"2026-11-11", id="bytes"),
        pytest.param(bytearray(b"2026-11-11"), id="bytearray"),
        pytest.param(["2026-11-11"], id="list"),
        pytest.param(("2026", "11", "11"), id="tuple"),
        pytest.param({"date": "2026-11-11"}, id="dict"),
        pytest.param(object(), id="object"),
        pytest.param(__import__("datetime").date(2026, 11, 11), id="date-object"),
        pytest.param(StrSubclass("2026-11-11"), id="str-subclass"),
    ],
)
def test_non_strings_are_rejected_without_raising(value):
    assert d.is_valid_iso_date(value) is False


def test_the_helper_is_public_and_returns_exact_booleans():
    assert "is_valid_iso_date" in dir(d) and not d.is_valid_iso_date.__name__.startswith("_")
    assert type(d.is_valid_iso_date("2026-11-11")) is bool and type(d.is_valid_iso_date(None)) is bool


# ---------- Q. validate_dataset: arguments and split shape ----------

TEXT_A = "Alpha clause one. Beta clause two. Gamma clause three."
TEXT_B = (HINDI + " ") * 6
TEXTS = {"doc-a": TEXT_A, "doc-b": TEXT_B}
QTEXT = {
    "d1": "What does the first clause say about alpha?",
    "d2": "How long is the notice period in days?",
    "d3": "Which party pays the refund amount?",
    "t1": "Who may terminate the agreement early?",
    "t2": "Where is the governing court located?",
    "t3": "When does the warranty period expire?",
    "x1": "Explain the data retention schedule.",
    "x2": "List every permitted subcontractor name.",
}
DUP_TEXT = d_nd_code = nd.DATASET_DUPLICATE_QUESTION_TEXT
SIMILAR = nd.DATASET_SIMILAR_QUESTION


def ds_doc(key, text, language="en"):
    return s.ManifestDocument(
        corpus_key=key, title="T", publisher="P", source_url="https://example.invalid/a", license="MIT",
        license_url="https://example.invalid/l", format="txt", language=language, source_file_sha256=SHA,
        extracted_text_sha256=sha256_text(text), extracted_text_length=len(text),
        extraction=s.Extraction(processing_ref="d" * 40, python="3.12.4"), redistributable=True,
    )


MANIFEST = s.Manifest("eval-manifest-v1", "c1", "ingested", (ds_doc("doc-a", TEXT_A), ds_doc("doc-b", TEXT_B, "hi")))
DRAFT_MANIFEST = dataclasses.replace(MANIFEST, stage="draft")
INVALID_MANIFEST = dataclasses.replace(MANIFEST, schema_version="nope")


def rspan(key="doc-a", start=0, end=17):
    text = TEXTS[key]
    return s.Span(key, sha256_text(text), start, end, text[start:end], sha256_text(text[start:end]))


def ghost_span():
    return s.Span("doc-zzz", "a" * 64, 0, 5, "hello", sha256_text("hello"))


def bad_hash_span():
    return s.Span("doc-a", sha256_text("other"), 0, 17, TEXT_A[0:17], sha256_text(TEXT_A[0:17]))


def dq(qid, spans=None, split="dev", text=None, **overrides):
    spans = (rspan(),) if spans is None else spans
    values = dict(
        question_id=qid, split=split, answerable=True,
        question=text if text is not None else QTEXT.get(qid, "A fallback question about something else entirely"),
        query_language="en", author_type="human", hindi_reviewed=False, expected_answer="Answer",
        units=(s.EvidenceUnit("u1", spans),), source_language="en", difficulty="direct", close_distractor=False,
    )
    values.update(overrides)
    return s.Question(**values)


def tq(qid="t1", spans=None, **overrides):
    overrides.setdefault("split", "test")
    return dq(qid, spans=(rspan(start=35, end=54),) if spans is None else spans, **overrides)


def run_ds(dev=None, test=None, manifest=MANIFEST, texts=TEXTS, threshold=0.9, questions=None):
    if questions is None:
        questions = {"dev": [dq("d1")] if dev is None else dev, "test": [tq()] if test is None else test}
    return d.validate_dataset(questions, manifest, near_duplicate_threshold=threshold, texts=texts)


def got(*args, **kwargs):
    return [(i.code, i.location) for i in run_ds(*args, **kwargs)]


def test_the_fixture_dataset_manifests_are_what_the_tests_assume():
    from app.rag.evaluation import validation as val

    assert val.validate_manifest(MANIFEST) == [] and val.validate_manifest(DRAFT_MANIFEST) == []
    assert any(i.severity == "error" for i in val.validate_manifest(INVALID_MANIFEST))


@pytest.mark.parametrize("texts", [pytest.param(TEXTS, id="with-texts"), pytest.param(None, id="without-texts")])
def test_a_valid_dataset_is_clean(texts):
    assert run_ds(texts=texts) == []


def test_empty_collections_are_clean_and_a_mapping_proxy_is_accepted():
    from types import MappingProxyType

    assert run_ds(dev=[], test=[]) == []
    assert run_ds(questions=MappingProxyType({"dev": (dq("d1"),), "test": (tq(),)})) == []


@pytest.mark.parametrize("bad", [[], None, "dev", 5, ("dev", "test")], ids=["list", "none", "str", "int", "tuple"])
def test_questions_must_be_a_mapping(bad):
    with pytest.raises(TypeError, match="^questions must be a Mapping$"):
        d.validate_dataset(bad, MANIFEST, near_duplicate_threshold=0.9)


@pytest.mark.parametrize("bad", [5, "abc", b"abc", None, {"a"}], ids=["int", "str", "bytes", "none", "set"])
@pytest.mark.parametrize("name", ["dev", "test"])
def test_each_collection_must_be_a_sequence(name, bad):
    questions = {"dev": [], "test": [], name: bad}
    with pytest.raises(TypeError, match="^each questions collection must be a sequence of Question$"):
        d.validate_dataset(questions, MANIFEST, near_duplicate_threshold=0.9)


@pytest.mark.parametrize("bad", [None, "manifest", {}, [], 5], ids=["none", "str", "dict", "list", "int"])
def test_manifest_must_be_a_manifest(bad):
    with pytest.raises(TypeError, match="^manifest must be a Manifest$"):
        d.validate_dataset({"dev": [], "test": []}, bad, near_duplicate_threshold=0.9)


@pytest.mark.parametrize("bad", [[], "text", 5, ("a",)], ids=["list", "str", "int", "tuple"])
def test_texts_must_be_none_or_a_mapping(bad):
    with pytest.raises(TypeError, match="^texts must be a Mapping or None$"):
        d.validate_dataset({"dev": [], "test": []}, MANIFEST, near_duplicate_threshold=0.9, texts=bad)


def test_the_threshold_is_required_and_keyword_only():
    with pytest.raises(TypeError):
        d.validate_dataset({"dev": [], "test": []}, MANIFEST)
    with pytest.raises(TypeError):
        d.validate_dataset({"dev": [], "test": []}, MANIFEST, 0.9)


@pytest.mark.parametrize(
    "bad, error",
    [
        pytest.param(True, TypeError, id="bool"),
        pytest.param("0.5", TypeError, id="str"),
        pytest.param(None, TypeError, id="none"),
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="inf"),
        pytest.param(2, ValueError, id="above-one"),
        pytest.param(-0.1, ValueError, id="below-zero"),
    ],
)
def test_a_bad_threshold_is_refused_before_any_checking(bad, error):
    with pytest.raises(error):
        d.validate_dataset({"dev": [], "test": []}, MANIFEST, near_duplicate_threshold=bad)
    with pytest.raises(error):
        run_ds(dev=[dq("d1", question_id="BAD")], threshold=bad)


@pytest.mark.parametrize(
    "questions",
    [
        pytest.param({"dev": [dq("d1")]}, id="test-missing"),
        pytest.param({"test": [tq()]}, id="dev-missing"),
        pytest.param({}, id="both-missing"),
        pytest.param({"dev": [dq("d1")], "test": [tq()], "train": []}, id="extra-key"),
        pytest.param({"dev": [dq("d1")], "test": [tq()], 5: []}, id="non-str-key"),
        pytest.param({"DEV": [dq("d1")], "test": [tq()]}, id="wrong-case-key"),
    ],
)
def test_a_wrong_split_shape_is_one_issue_not_an_exception(questions):
    assert got(questions=questions) == [("DATASET_SPLITS", "dataset")]


def test_the_present_collections_are_still_validated_after_a_shape_issue():
    questions = {"dev": [dq("d1", question_id="BAD")]}
    assert got(questions=questions) == [("DATASET_SPLITS", "dataset"), ("Q_QUESTION_ID", "dev.questions[0].question_id")]


# ---------- R. manifest gate ----------

def test_a_draft_manifest_gives_one_not_ingested_issue():
    assert got(manifest=DRAFT_MANIFEST) == [("DATASET_MANIFEST_NOT_INGESTED", "manifest.stage")]


def test_an_invalid_manifest_gives_one_invalid_issue():
    assert got(manifest=INVALID_MANIFEST) == [("DATASET_MANIFEST_INVALID", "manifest")]
    assert got(manifest=dataclasses.replace(MANIFEST, documents=())) == [("DATASET_MANIFEST_INVALID", "manifest")]


def test_an_invalid_manifest_is_reported_as_invalid_even_when_it_is_also_a_draft():
    both = dataclasses.replace(MANIFEST, stage="draft", schema_version="nope")
    assert got(manifest=both) == [("DATASET_MANIFEST_INVALID", "manifest")]


def _gate_dataset():
    bad_offsets = s.Span("doc-a", sha256_text(TEXT_A), 9, 3, "abcdef", sha256_text("abcdef"))
    return [dq("d1", question_id="BAD", spans=(bad_offsets, ghost_span()))]


@pytest.mark.parametrize("manifest", [pytest.param(DRAFT_MANIFEST, id="draft"), pytest.param(INVALID_MANIFEST, id="invalid")])
def test_a_failed_gate_skips_only_the_manifest_reference_checks(manifest):
    codes_found = {i.code for i in run_ds(dev=_gate_dataset(), test=[], manifest=manifest, texts=None)}
    assert {"Q_QUESTION_ID", "SPAN_OFFSET_RANGE"} <= codes_found
    assert "DATASET_SPAN_KEY_UNKNOWN" not in codes_found


def test_an_open_gate_runs_the_reference_checks():
    codes_found = {i.code for i in run_ds(dev=_gate_dataset(), test=[], texts=None)}
    assert {"Q_QUESTION_ID", "SPAN_OFFSET_RANGE", "DATASET_SPAN_KEY_UNKNOWN"} <= codes_found


def test_a_failed_gate_still_runs_the_text_anchoring_phase():
    wrong_quote = s.Span("doc-a", sha256_text(TEXT_A), 0, 17, "x" * 17, sha256_text("x" * 17))
    assert got(dev=[dq("d1", spans=(wrong_quote,))], manifest=DRAFT_MANIFEST) == [
        ("DATASET_MANIFEST_NOT_INGESTED", "manifest.stage"),
        ("ANCHOR_QUOTE_MISMATCH", "dev.questions[0].units[0].spans[0].quote"),
    ]


# ---------- S. per-question order and split consistency ----------

def test_per_question_order_is_record_then_span_then_manifest_reference():
    bad_offsets = s.Span("doc-a", sha256_text(TEXT_A), 9, 3, "abcdef", sha256_text("abcdef"))
    q = dq("d1", question_id="BAD", spans=(bad_offsets, ghost_span()))
    assert got(dev=[q], test=[], texts=None) == [
        ("Q_QUESTION_ID", "dev.questions[0].question_id"),
        ("SPAN_OFFSET_RANGE", "dev.questions[0].units[0].spans[0].offsets"),
        ("DATASET_SPAN_KEY_UNKNOWN", "dev.questions[0].units[0].spans[1].corpus_key"),
    ]


def test_dev_is_checked_before_test_whatever_the_key_order_and_locations_use_positions():
    dev = [dq("d1"), dq("d2", spans=(rspan(start=18, end=34),)), dq("d3", question_id="BAD", spans=(rspan(start=18, end=34),))]
    test = [tq("t1", question_id="BAD2")]
    questions = {"test": test, "dev": dev}
    assert got(questions=questions) == [
        ("Q_QUESTION_ID", "dev.questions[2].question_id"),
        ("Q_QUESTION_ID", "test.questions[0].question_id"),
    ]


@pytest.mark.parametrize(
    "dev, test, expected",
    [
        pytest.param([dq("d1", split="test")], [tq()], [("DATASET_SPLIT_MISMATCH", "dev.questions[0].split")], id="test-record-in-dev"),
        pytest.param([dq("d1")], [tq(split="dev")], [("DATASET_SPLIT_MISMATCH", "test.questions[0].split")], id="dev-record-in-test"),
        pytest.param([dq("d1", split="train")], [tq()], [("Q_ENUM", "dev.questions[0].split")], id="unknown-split-is-an-enum-issue-only"),
        pytest.param([dq("d1", split=None)], [tq()], [("Q_FIELD_TYPE", "dev.questions[0].split")], id="non-str-split-is-a-type-issue-only"),
    ],
)
def test_split_consistency(dev, test, expected):
    assert got(dev=dev, test=test) == expected


QUESTION_TYPE = "DATASET_QUESTION_TYPE"


def test_an_item_that_is_not_a_question_is_one_type_error_at_its_original_location():
    result = run_ds(dev=[dq("d1"), "not a question", None], test=[tq(), 5, {"question": "x"}])
    assert [(i.code, i.location, i.severity) for i in result] == [
        (QUESTION_TYPE, "dev.questions[1]", "error"),
        (QUESTION_TYPE, "dev.questions[2]", "error"),
        (QUESTION_TYPE, "test.questions[1]", "error"),
        (QUESTION_TYPE, "test.questions[2]", "error"),
    ]
    assert {i.message for i in result} == {d._DATASET_MESSAGES[QUESTION_TYPE]}


def test_a_non_question_item_gets_no_further_validation_and_the_rest_is_still_checked():
    result = got(dev=[None, dq("d2", spans=(ghost_span(),))])
    assert result[0] == (QUESTION_TYPE, "dev.questions[0]")
    assert not any(loc == "dev.questions[0]" or loc.startswith("dev.questions[0].") for _, loc in result[1:])
    assert any(loc.startswith("dev.questions[1].") for _, loc in result[1:])


def test_a_subclass_of_question_is_not_a_type_error():
    class Sub(s.Question):
        pass

    import dataclasses

    base = dq("d1")
    sub = Sub(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
    assert QUESTION_TYPE not in {c for c, _ in got(dev=[sub])}


def test_duplicate_text_keeps_original_indices_after_earlier_non_question_items():
    dev = [None, "x", dq("d1")]
    test = [7, tq(text=QTEXT["d1"].upper())]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=test)]
    assert (DUP_TEXT, "test.questions[1].question") in result
    assert not any("questions[0].question" in loc and "test" in loc for _, loc in result)
    assert (QUESTION_TYPE, "test.questions[0]") in result


def test_similar_pair_keeps_original_indices_after_earlier_non_question_items():
    dev = [None, dq("d1", text="abcdefg")]
    test = [None, None, tq(text="abcdefgh")]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=test, threshold=0.75)]
    assert (SIMILAR, "test.questions[2].question~dev.questions[1].question") in result


def test_duplicate_text_within_one_collection_keeps_original_indices():
    dev = [None, dq("d1", text="same words here please"), "junk", dq("d2", text="SAME words here please")]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=[])]
    assert (DUP_TEXT, "dev.questions[3].question") in result


def test_detector_locations_use_the_collection_index_even_when_split_disagrees():
    dev = [None, dq("d1", split="test", text="abcdefg")]
    test = [tq(text="abcdefgh", split="dev")]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=test, threshold=0.75)]
    assert (SIMILAR, "test.questions[0].question~dev.questions[1].question") in result
    assert not any(loc.endswith("dev.questions[0].question") for code, loc in result if code == SIMILAR)


def test_the_callers_questions_are_not_modified_by_the_detector_input():
    q = dq("d1", split="test")
    run_ds(dev=[q], test=[])
    assert q.split == "test"


# ---------- T. manifest references and languages ----------

def _loc(i=0, u=0, s_=0, field=None, split="dev"):
    base = f"{split}.questions[{i}].units[{u}].spans[{s_}]"
    return base if field is None else f"{base}.{field}"


@pytest.mark.parametrize("texts", [pytest.param(TEXTS, id="with-texts"), pytest.param(None, id="without-texts")])
def test_an_unknown_document_key_is_reported_once(texts):
    assert got(dev=[dq("d1", spans=(ghost_span(),))], texts=texts) == [("DATASET_SPAN_KEY_UNKNOWN", _loc(field="corpus_key"))]


@pytest.mark.parametrize("texts", [pytest.param(TEXTS, id="with-texts"), pytest.param(None, id="without-texts")])
def test_a_source_hash_mismatch_is_reported_once(texts):
    assert got(dev=[dq("d1", spans=(bad_hash_span(),))], texts=texts) == [("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256"))]


def test_a_known_document_with_the_right_hash_is_clean_including_the_hindi_document():
    hindi_q = dq("d2", spans=(rspan("doc-b", 0, 6),), source_language="hi")
    assert got(dev=[dq("d1"), hindi_q]) == []


def test_the_unknown_key_check_does_not_apply_to_unanswerable_questions():
    assert got(dev=[dq("d1"), unanswerable(question_id="u1", question="Something unrelated entirely?")]) == []


def test_mixed_document_languages_give_one_issue_at_units():
    spans = (rspan(), rspan("doc-b", 0, 6), rspan(start=18, end=34))
    assert got(dev=[dq("d1", spans=spans)]) == [("DATASET_MIXED_LANGUAGE_SPANS", "dev.questions[0].units")]
    units = (s.EvidenceUnit("u1", (rspan(),)), s.EvidenceUnit("u2", (rspan("doc-b", 0, 6),)))
    assert got(dev=[dq("d1", units=units)]) == [("DATASET_MIXED_LANGUAGE_SPANS", "dev.questions[0].units")]


@pytest.mark.parametrize(
    "spans, source, expected",
    [
        pytest.param((rspan(),), "hi", [("DATASET_SOURCE_LANGUAGE_MISMATCH", "dev.questions[0].source_language")], id="english-document-hindi-source"),
        pytest.param((rspan("doc-b", 0, 6),), "en", [("DATASET_SOURCE_LANGUAGE_MISMATCH", "dev.questions[0].source_language")], id="hindi-document-english-source"),
        pytest.param((rspan(),), "en", [], id="english-match"),
        pytest.param((rspan("doc-b", 0, 6),), "hi", [], id="hindi-match"),
        pytest.param((rspan(), rspan("doc-b", 0, 6)), "hi", [("DATASET_MIXED_LANGUAGE_SPANS", "dev.questions[0].units")], id="mixed-wins-over-mismatch"),
        pytest.param((rspan(), ghost_span()), "en", [("DATASET_SPAN_KEY_UNKNOWN", "dev.questions[0].units[0].spans[1].corpus_key")], id="unknown-keys-are-left-out-of-the-language-check"),
        pytest.param((rspan(), ghost_span()), "hi", [("DATASET_SPAN_KEY_UNKNOWN", "dev.questions[0].units[0].spans[1].corpus_key"), ("DATASET_SOURCE_LANGUAGE_MISMATCH", "dev.questions[0].source_language")], id="known-language-still-compared"),
        pytest.param((rspan(),), "fr", [("Q_ENUM", "dev.questions[0].source_language")], id="invalid-source-language-is-not-compared"),
    ],
)
def test_language_checks(spans, source, expected):
    assert got(dev=[dq("d1", spans=spans, source_language=source)]) == expected


def test_a_cross_language_question_is_valid_when_the_source_language_matches_the_document():
    q = dq("d1", spans=(rspan("doc-b", 0, 6),), source_language="hi")
    assert got(dev=[q]) == []


# ---------- U. the span phase: one mode per question ----------

def test_with_texts_resolve_question_is_used_and_validate_span_is_not(monkeypatch):
    from app.rag.evaluation import anchoring as an

    calls = []

    def recorder(q, texts, *, location):
        calls.append(location)
        return []

    def forbidden(*args, **kwargs):
        raise AssertionError("validate_span must not be called when texts are given")

    monkeypatch.setattr(an, "resolve_question", recorder)
    monkeypatch.setattr(s, "validate_span", forbidden)
    assert got() == []
    assert calls == ["dev.questions[0]", "test.questions[0]"]


def test_without_texts_validate_span_is_used_and_resolve_question_is_not(monkeypatch):
    from app.rag.evaluation import anchoring as an

    seen = []
    real = s.validate_span

    def counting(span, *, location):
        seen.append(location)
        return real(span, location=location)

    def forbidden(*args, **kwargs):
        raise AssertionError("resolve_question must not be called without texts")

    monkeypatch.setattr(an, "resolve_question", forbidden)
    monkeypatch.setattr(s, "validate_span", counting)
    assert got(texts=None) == []
    assert seen == ["dev.questions[0].units[0].spans[0]", "test.questions[0].units[0].spans[0]"]


def test_only_answerable_questions_are_span_checked(monkeypatch):
    from app.rag.evaluation import anchoring as an

    calls = []
    monkeypatch.setattr(an, "resolve_question", lambda q, texts, *, location: calls.append(location) or [])
    got(dev=[dq("d1"), unanswerable(question_id="u1", question="Something unrelated entirely?")])
    assert calls == ["dev.questions[0]", "test.questions[0]"]


def test_with_texts_a_quote_mismatch_is_reported_and_without_texts_it_is_not():
    wrong = s.Span("doc-a", sha256_text(TEXT_A), 0, 17, "x" * 17, sha256_text("x" * 17))
    assert got(dev=[dq("d1", spans=(wrong,))]) == [("ANCHOR_QUOTE_MISMATCH", _loc(field="quote"))]
    assert got(dev=[dq("d1", spans=(wrong,))], texts=None) == []


@pytest.mark.parametrize("texts", [pytest.param(TEXTS, id="with-texts"), pytest.param(None, id="without-texts")])
def test_a_structural_span_problem_is_reported_exactly_once(texts):
    bad = s.Span("doc-a", sha256_text(TEXT_A), 9, 3, "abcdef", sha256_text("abcdef"))
    result = [i.code for i in run_ds(dev=[dq("d1", spans=(bad,))], texts=texts)]
    assert result.count("SPAN_OFFSET_RANGE") == 1


@pytest.mark.parametrize(
    "units, expected",
    [
        pytest.param([s.EvidenceUnit("u1", (rspan(),))], "Q_UNITS_TYPE", id="units-list"),
        pytest.param((s.EvidenceUnit("u1", [rspan()]),), "Q_SPANS_TYPE", id="spans-list"),
        pytest.param(("not a unit",), "Q_UNIT_TYPE", id="unit-wrong-class"),
        pytest.param((s.EvidenceUnit("u1", (rspan(), "not a span")),), "Q_SPAN_TYPE", id="span-wrong-class"),
    ],
)
@pytest.mark.parametrize("texts", [pytest.param(TEXTS, id="with-texts"), pytest.param(None, id="without-texts")])
def test_malformed_containers_get_only_the_record_issue(units, expected, texts):
    result = [i.code for i in run_ds(dev=[dq("d1", units=units)], texts=texts)]
    assert result == [expected]


def test_a_known_document_whose_text_is_not_supplied_is_an_anchoring_issue():
    assert got(texts={"doc-b": TEXT_B}) == [("ANCHOR_TEXT_MISSING", _loc(field="corpus_key")), ("ANCHOR_TEXT_MISSING", _loc(split="test", field="corpus_key"))]


# ---------- V. suppression and deduplication ----------

def test_p1_a_missing_text_for_an_unknown_key_is_not_reported_twice():
    result = got(dev=[dq("d1", spans=(ghost_span(),))], texts=TEXTS)
    assert result == [("DATASET_SPAN_KEY_UNKNOWN", _loc(field="corpus_key"))]


def test_p1_does_not_apply_when_the_key_is_known():
    result = got(dev=[dq("d1")], test=[], texts={"doc-b": TEXT_B})
    assert result == [("ANCHOR_TEXT_MISSING", _loc(field="corpus_key"))]


def test_p2_a_text_that_matches_the_manifest_hash_drops_the_anchoring_hash_mismatch():
    result = got(dev=[dq("d1", spans=(bad_hash_span(),))], texts=TEXTS)
    assert result == [("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256"))]


def test_p2_a_text_that_matches_neither_hash_keeps_both_issues():
    texts = {**TEXTS, "doc-a": TEXT_A + "x"}
    result = got(dev=[dq("d1", spans=(bad_hash_span(),))], test=[], texts=texts)
    assert result == [
        ("ANCHOR_SOURCE_HASH_MISMATCH", _loc(field="source_sha256")),
        ("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256")),
    ]


def test_p2_a_text_that_matches_the_span_but_not_the_manifest_reports_only_the_manifest_mismatch():
    longer = TEXT_A + "x"
    span = s.Span("doc-a", sha256_text(longer), 0, 17, longer[0:17], sha256_text(longer[0:17]))
    result = got(dev=[dq("d1", spans=(span,))], test=[], texts={**TEXTS, "doc-a": longer})
    assert result == [("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256"))]


def test_p2_a_missing_text_does_not_suppress_the_manifest_hash_mismatch():
    result = got(dev=[dq("d1", spans=(bad_hash_span(),))], test=[], texts={"doc-b": TEXT_B})
    assert result == [("ANCHOR_TEXT_MISSING", _loc(field="corpus_key")), ("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256"))]


def test_identical_code_and_location_pairs_are_reported_once(monkeypatch):
    from app.rag.evaluation import near_duplicates as nd_module

    issue = s.Issue(nd.DATASET_DUPLICATE_QUESTION_TEXT, "test.questions[0].question", "fixed")
    monkeypatch.setattr(nd_module, "find_duplicate_questions", lambda qs, **kw: [issue, issue] if qs else [])
    assert [(i.code, i.location) for i in run_ds()] == [(nd.DATASET_DUPLICATE_QUESTION_TEXT, "test.questions[0].question")]


def test_a_different_code_at_the_same_location_is_never_suppressed():
    span = s.Span("doc-a", sha256_text("other"), 0, 17, "x" * 17, sha256_text("x" * 17))
    result = got(dev=[dq("d1", spans=(span,))], test=[], texts={**TEXTS, "doc-a": TEXT_A + "x"})
    assert ("DATASET_SPAN_HASH_MISMATCH", _loc(field="source_sha256")) in result
    assert ("ANCHOR_SOURCE_HASH_MISMATCH", _loc(field="source_sha256")) in result


# ---------- W. dataset-level phases ----------

def test_duplicate_ids_are_reported_at_the_later_question():
    dev = [dq("d1"), dq("d1", text=QTEXT["d2"])]
    assert got(dev=dev) == [("DATASET_DUPLICATE_QUESTION_ID", "dev.questions[1].question_id")]


def test_a_duplicate_id_across_splits_is_reported_in_test():
    assert got(test=[tq("d1", text=QTEXT["t1"])]) == [("DATASET_DUPLICATE_QUESTION_ID", "test.questions[0].question_id")]


def test_three_identical_ids_report_the_second_and_third():
    dev = [dq("d1"), dq("d1", text=QTEXT["d2"]), dq("d1", text=QTEXT["d3"])]
    assert got(dev=dev, test=[]) == [
        ("DATASET_DUPLICATE_QUESTION_ID", "dev.questions[1].question_id"),
        ("DATASET_DUPLICATE_QUESTION_ID", "dev.questions[2].question_id"),
    ]


def test_a_non_string_id_is_a_type_issue_and_never_a_duplicate():
    dev = [dq("d1"), dq("d2", question_id=5), dq("d3", question_id=5)]
    result = got(dev=dev, test=[])
    assert ("DATASET_DUPLICATE_QUESTION_ID", "dev.questions[2].question_id") not in result
    assert {c for c, _ in result} == {"Q_FIELD_TYPE"}


@pytest.mark.parametrize(
    "start, end, collides_",
    [
        pytest.param(10, 20, True, id="overlap"),
        pytest.param(17, 20, True, id="touching-after"),
        pytest.param(0, 17, True, id="identical"),
        pytest.param(3, 5, True, id="nested"),
        pytest.param(18, 25, False, id="one-position-gap"),
    ],
)
def test_dev_test_evidence_collisions(start, end, collides_):
    result = got(test=[tq(spans=(rspan(start=start, end=end),))])
    assert result == ([("DATASET_SPLIT_COLLISION", _loc(split="test"))] if collides_ else [])


def test_touching_before_a_dev_span_also_collides():
    dev = [dq("d1", spans=(rspan(start=20, end=30),))]
    assert got(dev=dev, test=[tq(spans=(rspan(start=10, end=20),))]) == [("DATASET_SPLIT_COLLISION", _loc(split="test"))]


def test_a_different_document_does_not_collide():
    assert got(test=[tq(spans=(rspan("doc-b", 0, 6),), source_language="hi")]) == []


def test_one_error_per_colliding_test_span():
    test = [tq(spans=(rspan(start=0, end=5), rspan(start=10, end=12), rspan(start=40, end=50)))]
    assert got(test=test) == [
        ("DATASET_SPLIT_COLLISION", _loc(split="test", s_=0)),
        ("DATASET_SPLIT_COLLISION", _loc(split="test", s_=1)),
    ]


def test_a_test_span_colliding_with_two_dev_spans_is_reported_once():
    dev = [dq("d1", spans=(rspan(start=0, end=8), rspan(start=8, end=17)))]
    assert got(dev=dev, test=[tq(spans=(rspan(start=5, end=12),))]) == [("DATASET_SPLIT_COLLISION", _loc(split="test"))]


def test_spans_within_one_split_never_collide_with_each_other():
    dev = [dq("d1"), dq("d2")]
    test = [tq("t1"), tq("t2")]
    assert got(dev=dev, test=test) == []


def test_collisions_skip_invalid_intervals_without_raising():
    odd = (
        s.Span("doc-a", sha256_text(TEXT_A), 20, 10, "abcdefghij", sha256_text("abcdefghij")),
        s.Span("doc-a", sha256_text(TEXT_A), -1, 3, "abcd", sha256_text("abcd")),
        s.Span("doc-a", sha256_text(TEXT_A), 5, 5, "", sha256_text("")),
        s.Span("doc-a", sha256_text(TEXT_A), True, 3, "ab", sha256_text("ab")),
        s.Span("doc-a", sha256_text(TEXT_A), 0.5, 3, "ab", sha256_text("ab")),
        s.Span(5, sha256_text(TEXT_A), 0, 3, "abc", sha256_text("abc")),
    )
    result = {c for c, _ in got(dev=[dq("d1", spans=odd)], test=[tq(spans=odd)], texts=None)}
    assert "DATASET_SPLIT_COLLISION" not in result and "SPAN_OFFSET_RANGE" in result


def test_unanswerable_and_malformed_questions_have_no_dev_spans_for_collisions():
    dev = [unanswerable(question_id="u1", question="Something unrelated entirely?"), dq("d2", units=[1])]
    assert "DATASET_SPLIT_COLLISION" not in {c for c, _ in got(dev=dev, test=[tq(spans=(rspan(start=0, end=17),))])}


def test_exact_duplicate_text_across_splits_is_an_error_at_the_test_question():
    result = run_ds(test=[tq(text=QTEXT["d1"].upper())])
    assert [(i.code, i.location, i.severity) for i in result] == [(DUP_TEXT, "test.questions[0].question", "error")]


def test_similar_text_is_a_warning_and_the_threshold_is_passed_through():
    dev, test = [dq("d1", text="abcdefg")], [tq(text="abcdefgh")]
    result = run_ds(dev=dev, test=test, threshold=0.75)
    assert [(i.code, i.location, i.severity) for i in result] == [(SIMILAR, "test.questions[0].question~dev.questions[0].question", "warning")]
    assert run_ds(dev=dev, test=test, threshold=0.76) == []


def test_the_duplicate_detector_is_called_unchanged_with_one_flat_sequence(monkeypatch):
    from app.rag.evaluation import near_duplicates as nd_module

    calls = []
    monkeypatch.setattr(nd_module, "find_duplicate_questions", lambda qs, **kw: calls.append((list(qs), kw)) or [])
    dev, test = [dq("d1"), dq("d2")], [tq()]
    run_ds(dev=dev, test=test, threshold=0.8)
    assert calls[-1] == (dev + test, {"threshold": 0.8})


def test_supplied_text_is_hashed_at_most_once_per_document_key(monkeypatch):
    import app.rag.chunking as chunking

    real = chunking.sha256_text
    calls = []
    monkeypatch.setattr(chunking, "sha256_text", lambda value: calls.append(value) or real(value))
    dev = [dq(f"d{i}", spans=(bad_hash_span(),)) for i in range(1, 6)]
    result = got(dev=dev, test=[])
    assert [c for c, _ in result].count("DATASET_SPAN_HASH_MISMATCH") == 5
    assert calls.count(TEXT_A) == 1


def test_the_hash_cache_does_not_change_the_p2_suppression_outcome():
    matching = dq("d1", spans=(bad_hash_span(),))
    other = dq("d2", spans=(bad_hash_span(),))
    stale = {**TEXTS, "doc-a": TEXT_A + " edited"}
    assert "ANCHOR_SOURCE_HASH_MISMATCH" not in {c for c, _ in got(dev=[matching, other], test=[])}
    assert {c for c, _ in got(dev=[matching, other], test=[], texts=stale)} >= {
        "ANCHOR_SOURCE_HASH_MISMATCH", "DATASET_SPAN_HASH_MISMATCH"}


def test_the_detector_gets_one_entry_per_collection_item_with_the_collection_as_split(monkeypatch):
    from app.rag.evaluation import near_duplicates as nd_module

    calls = []
    monkeypatch.setattr(nd_module, "find_duplicate_questions", lambda qs, **kw: calls.append(list(qs)) or [])
    run_ds(dev=[None, dq("d1", split="test")], test=[5, tq("t1", split="dev")])
    entries = calls[-1]
    assert [e.split for e in entries] == ["dev", "dev", "test", "test"]
    assert [e.question_id for e in entries][1] == "d1" and [e.question_id for e in entries][3] == "t1"
    assert entries[0].question is None and entries[2].question is None


def test_dataset_phases_are_in_the_approved_order():
    dev = [dq("d1", text="abcdefg")]
    test = [
        tq("d1", spans=(rspan(start=10, end=20),), text="abcdefg"),
        tq("t2", text="abcdefgh"),
    ]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=test, threshold=0.75)]
    assert result[:3] == [
        ("DATASET_DUPLICATE_QUESTION_ID", "test.questions[0].question_id"),
        ("DATASET_SPLIT_COLLISION", _loc(split="test")),
        (DUP_TEXT, "test.questions[0].question"),
    ]
    assert [c for c, _ in result[3:]] == [SIMILAR, SIMILAR]


def test_exact_duplicates_are_listed_before_similar_pairs_even_when_the_detector_finds_a_similar_pair_first():
    dev = [dq("d1", text="abcdefg"), dq("d2", spans=(rspan(start=18, end=34),), text="abcdefgh")]
    test = [tq("t1", text="abcdefg")]
    result = [(i.code, i.location) for i in run_ds(dev=dev, test=test, threshold=0.75)]
    assert result == [
        (DUP_TEXT, "test.questions[0].question"),
        (SIMILAR, "dev.questions[1].question~dev.questions[0].question"),
        (SIMILAR, "test.questions[0].question~dev.questions[1].question"),
    ]
    raw = nd.find_duplicate_questions([*dev, *test], threshold=0.75)
    assert [i.code for i in raw][0] == SIMILAR  # the detector alone would list the similar pair first


def test_per_question_issues_come_before_the_dataset_phases():
    dev = [dq("d1", question_id="BAD", text="abcdefg")]
    test = [tq("t1", text="abcdefg")]
    assert [c for c, _ in got(dev=dev, test=test)] == ["Q_QUESTION_ID", DUP_TEXT]


# ---------- X. whole-result properties ----------

def _kitchen_sink_open_gate():
    dev = [
        dq("d1", split="test"),
        dq("d2", spans=(ghost_span(),)),
        dq("d3", spans=(bad_hash_span(),)),
        dq("x1", spans=(rspan(start=18, end=34), rspan("doc-b", 0, 6))),
        dq("x2", spans=(rspan(start=18, end=34),), source_language="hi"),
    ]
    test = [tq("d1", spans=(rspan(start=0, end=17),), text=QTEXT["d1"].upper())]
    test.append("not a question")
    return {"dev": dev, "test": test, "train": []}


def _all_dataset_codes():
    emitted = {i.code for i in run_ds(questions=_kitchen_sink_open_gate())}
    emitted |= {i.code for i in run_ds(manifest=DRAFT_MANIFEST)}
    emitted |= {i.code for i in run_ds(manifest=INVALID_MANIFEST)}
    return emitted


def test_every_new_dataset_code_is_reachable_with_its_fixed_message_and_error_severity():
    new_codes = set(d._DATASET_MESSAGES)
    assert new_codes == {
        "DATASET_SPLITS", "DATASET_QUESTION_TYPE", "DATASET_MANIFEST_INVALID", "DATASET_MANIFEST_NOT_INGESTED", "DATASET_SPLIT_MISMATCH",
        "DATASET_SPAN_KEY_UNKNOWN", "DATASET_SPAN_HASH_MISMATCH", "DATASET_MIXED_LANGUAGE_SPANS",
        "DATASET_SOURCE_LANGUAGE_MISMATCH", "DATASET_DUPLICATE_QUESTION_ID", "DATASET_SPLIT_COLLISION",
    }
    assert new_codes <= _all_dataset_codes()
    for manifest, questions in (
        (MANIFEST, _kitchen_sink_open_gate()),
        (DRAFT_MANIFEST, None),
        (INVALID_MANIFEST, None),
    ):
        for issue in run_ds(manifest=manifest, questions=questions):
            if issue.code in new_codes:
                assert issue.message == d._DATASET_MESSAGES[issue.code] and issue.severity == "error"
        assert d._MESSAGES.keys().isdisjoint(new_codes)


def test_the_record_message_mapping_still_holds_exactly_the_record_codes():
    assert sorted(d._MESSAGES) == sorted(ALL_CODES)


def test_warnings_flow_through_and_do_not_fail_require_valid():
    from app.rag.evaluation import validation as val

    dev, test = [dq("d1", text="abcdefg")], [tq(text="abcdefgh")]
    result = run_ds(dev=dev, test=test, threshold=0.75)
    assert result and all(i.severity == "warning" for i in result)
    val.require_valid(result)


def test_results_are_deterministic_and_inputs_are_not_modified():
    questions = _kitchen_sink_open_gate()
    snapshot = {k: list(v) for k, v in questions.items()}
    first = run_ds(questions=questions)
    assert run_ds(questions=questions) == first
    assert {k: list(v) for k, v in questions.items()} == snapshot
    texts = dict(TEXTS)
    run_ds(texts=texts)
    assert texts == TEXTS


def test_no_user_controlled_value_appears_in_any_dataset_issue():
    secret = "secret-canary-7731"
    canary_span = s.Span(secret, sha256_text(secret), 0, 5, secret[:5], sha256_text(secret[:5]))
    dev = [
        dq(secret, spans=(canary_span, rspan()), text=secret + " question", question_id=secret),
        dq(secret, spans=(rspan(start=18, end=34),), text=secret + " question"),
    ]
    test = [tq(secret, text=secret + " question", spans=(rspan(start=0, end=17),))]
    for texts in (TEXTS, {**TEXTS, secret: secret}, None):
        for manifest in (MANIFEST, DRAFT_MANIFEST):
            issues = run_ds(dev=dev, test=test, texts=texts, manifest=manifest)
            assert issues
            for issue in issues:
                assert "canary" not in repr(issue).lower()


def test_locations_contain_only_field_names_indexes_and_separators():
    allowed = set("abcdefghijklmnopqrstuvwxyz_[].0123456789~")
    for manifest, questions in ((MANIFEST, _kitchen_sink_open_gate()), (DRAFT_MANIFEST, None), (INVALID_MANIFEST, None)):
        for issue in run_ds(manifest=manifest, questions=questions):
            assert set(issue.location) <= allowed, issue.location


def test_texts_none_never_produces_an_anchoring_issue():
    wrong = s.Span("doc-a", sha256_text(TEXT_A), 0, 17, "x" * 17, sha256_text("x" * 17))
    assert not any(i.code.startswith("ANCHOR_") for i in run_ds(dev=[dq("d1", spans=(wrong, ghost_span()))], texts=None))


def test_the_docstring_states_the_official_run_contract():
    doc = d.validate_dataset.__doc__
    for phrase in ("Official-run contract", "every document", "verify_source_text", "texts=None", "NOT enforced"):
        assert phrase in doc


# ---------- K. import isolation ----------

def test_importing_the_module_loads_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import app.rag.evaluation.dataset\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in heavy)\n"
        "later = sorted(m for m in sys.modules if m in ('app.rag.evaluation.validation', 'app.rag.evaluation.anchoring', "
        "'app.rag.evaluation.parsing', 'app.rag.evaluation.intervals'))\n"
        "print(loaded, later)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[] []"
