"""E1.2d step 1: derive_structure, QuestionStructure and the structure constants.

A  constants
B  retrieval type (all four directions)
C  unit structure
D  alternatives
E  every combination of the three facts
F  unanswerable questions
G  invalid input
H  the result type
I  purity: no mutation, determinism, order independence, single-change effects
J  error messages never contain question content
K  import isolation
"""
import dataclasses
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import schema as s
from app.rag.evaluation import structure as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHA_A = "a" * 64
UNANSWERABLE_MESSAGE = "an unanswerable question has no structure"
INVALID_MESSAGE = "the answerable question cannot be classified"
CANARY = "SECRET-CANARY-7731"


class StrSubclass(str):
    pass


def make_span(start: int = 0, end: int = 5, corpus_key: str = "doc-a") -> s.Span:
    quote = "x" * (end - start)
    return s.Span(
        corpus_key=corpus_key,
        source_sha256=SHA_A,
        start=start,
        end=end,
        quote=quote,
        quote_sha256=sha256_text(quote),
    )


def make_unit(span_count: int, unit_id: str = "u1", first_start: int = 0) -> s.EvidenceUnit:
    """A unit with ``span_count`` alternative spans at distinct, separate offsets."""
    spans = tuple(make_span(first_start + 10 * i, first_start + 10 * i + 5) for i in range(span_count))
    return s.EvidenceUnit(unit_id=unit_id, spans=spans)


def make_units(*span_counts: int) -> tuple:
    """One unit per entry; each entry is that unit's number of alternative spans."""
    return tuple(
        make_unit(count, unit_id=f"u{index}", first_start=1000 * index)
        for index, count in enumerate(span_counts)
    )


_DEFAULT_UNITS = object()  # lets a test pass units=None on purpose


def make_question(units=_DEFAULT_UNITS, **overrides) -> s.Question:
    """A valid answerable question (one unit, one span, en to en) unless ``overrides`` change it."""
    values = {
        "question_id": "q1",
        "split": "test",
        "answerable": True,
        "question": "What is it?",
        "query_language": "en",
        "author_type": "human",
        "hindi_reviewed": False,
        "expected_answer": "An answer.",
        "units": make_units(1) if units is _DEFAULT_UNITS else units,
        "source_language": "en",
        "difficulty": "direct",
        "close_distractor": False,
    }
    values.update(overrides)
    return s.Question(**values)


def make_unanswerable(**overrides) -> s.Question:
    values = {
        "question_id": "q2",
        "split": "dev",
        "answerable": False,
        "question": "Is this covered?",
        "query_language": "en",
        "author_type": "human",
        "hindi_reviewed": False,
        "unanswerable_type": "off_topic",
        "semantic_review": s.SemanticReview(
            reviewer="reader", date="2026-11-11", result="confirmed_unanswerable"
        ),
    }
    values.update(overrides)
    return s.Question(**values)


# ---------- A: constants ----------

def test_structure_version_is_the_string_one():
    assert st.STRUCTURE_VERSION == "1"
    assert type(st.STRUCTURE_VERSION) is str


def test_the_structure_version_is_separate_from_the_schema_versions():
    assert st.STRUCTURE_VERSION not in (s.MANIFEST_SCHEMA_VERSION, s.DATASET_SCHEMA_VERSION)


def test_unit_structures_are_the_two_approved_values():
    assert st.UNIT_STRUCTURES == ("single_unit", "multi_unit")


def test_retrieval_types_are_the_four_approved_values():
    assert st.RETRIEVAL_TYPES == ("en_to_en", "hi_to_hi", "en_to_hi", "hi_to_en")


def test_retrieval_types_cover_every_pair_of_languages():
    expected = {f"{query}_to_{source}" for query in s.LANGUAGES for source in s.LANGUAGES}
    assert set(st.RETRIEVAL_TYPES) == expected


def test_cross_language_types_are_the_two_mixed_directions():
    assert st.CROSS_LANGUAGE_RETRIEVAL_TYPES == frozenset({"en_to_hi", "hi_to_en"})
    assert type(st.CROSS_LANGUAGE_RETRIEVAL_TYPES) is frozenset
    assert st.CROSS_LANGUAGE_RETRIEVAL_TYPES <= set(st.RETRIEVAL_TYPES)


# ---------- B: retrieval type ----------

@pytest.mark.parametrize(
    "query, source, expected, cross",
    [
        pytest.param("en", "en", "en_to_en", False, id="english-to-english"),
        pytest.param("hi", "hi", "hi_to_hi", False, id="hindi-to-hindi"),
        pytest.param("en", "hi", "en_to_hi", True, id="english-query-hindi-source"),
        pytest.param("hi", "en", "hi_to_en", True, id="hindi-query-english-source"),
    ],
)
def test_retrieval_type_follows_query_then_source_language(query, source, expected, cross):
    result = st.derive_structure(make_question(query_language=query, source_language=source))
    assert result.retrieval_type == expected
    assert result.retrieval_type in st.RETRIEVAL_TYPES
    assert (result.retrieval_type in st.CROSS_LANGUAGE_RETRIEVAL_TYPES) is cross


def test_the_two_cross_language_directions_are_different_values():
    english_to_hindi = st.derive_structure(make_question(query_language="en", source_language="hi"))
    hindi_to_english = st.derive_structure(make_question(query_language="hi", source_language="en"))
    assert english_to_hindi.retrieval_type != hindi_to_english.retrieval_type


# ---------- C: unit structure ----------

@pytest.mark.parametrize(
    "span_counts, expected",
    [
        pytest.param((1,), "single_unit", id="one-unit"),
        pytest.param((6,), "single_unit", id="one-unit-with-six-spans"),
        pytest.param((1, 1), "multi_unit", id="two-units"),
        pytest.param((1, 1, 1), "multi_unit", id="three-units"),
        pytest.param((1,) * 20, "multi_unit", id="twenty-units"),
    ],
)
def test_unit_structure_counts_units_not_spans(span_counts, expected):
    result = st.derive_structure(make_question(units=make_units(*span_counts)))
    assert result.unit_structure == expected
    assert result.unit_structure in st.UNIT_STRUCTURES


def test_two_units_over_the_same_span_are_still_multi_unit():
    span = make_span()
    units = (s.EvidenceUnit("a", (span,)), s.EvidenceUnit("b", (span,)))
    result = st.derive_structure(make_question(units=units))
    assert result.unit_structure == "multi_unit"
    assert result.has_alternatives is False


def test_units_in_different_documents_are_counted_the_same_way():
    units = (
        s.EvidenceUnit("a", (make_span(corpus_key="doc-a"),)),
        s.EvidenceUnit("b", (make_span(corpus_key="doc-b"),)),
    )
    assert st.derive_structure(make_question(units=units)).unit_structure == "multi_unit"


# ---------- D: alternatives ----------

@pytest.mark.parametrize(
    "span_counts, expected",
    [
        pytest.param((1,), False, id="one-span"),
        pytest.param((2,), True, id="two-spans-in-one-unit"),
        pytest.param((1, 1), False, id="two-units-one-span-each"),
        pytest.param((2, 1), True, id="alternatives-in-the-first-unit"),
        pytest.param((1, 2), True, id="alternatives-in-the-last-unit"),
        pytest.param((1, 3, 1), True, id="alternatives-in-the-middle-unit"),
        pytest.param((2, 2), True, id="alternatives-in-every-unit"),
    ],
)
def test_has_alternatives_is_true_when_any_unit_has_more_than_one_span(span_counts, expected):
    result = st.derive_structure(make_question(units=make_units(*span_counts)))
    assert result.has_alternatives is expected


@pytest.mark.parametrize(
    "spans",
    [
        pytest.param((make_span(0, 10), make_span(5, 15)), id="overlapping"),
        pytest.param((make_span(0, 5), make_span(5, 10)), id="touching"),
        pytest.param((make_span(0, 20), make_span(5, 10)), id="nested"),
        pytest.param((make_span(0, 5), make_span(0, 5)), id="identical"),
    ],
)
def test_alternatives_are_counted_whatever_their_positions(spans):
    units = (s.EvidenceUnit("a", spans),)
    assert st.derive_structure(make_question(units=units)).has_alternatives is True


# ---------- E: every combination ----------

_LAYOUTS = {
    "single-no-alternatives": ((1,), "single_unit", False),
    "single-with-alternatives": ((2,), "single_unit", True),
    "multi-no-alternatives": ((1, 1), "multi_unit", False),
    "multi-with-alternatives": ((2, 1), "multi_unit", True),
}
_LANGUAGE_PAIRS = ("en", "en"), ("hi", "hi"), ("en", "hi"), ("hi", "en")


@pytest.mark.parametrize(
    "layout, pair",
    [
        pytest.param(layout, pair, id=f"{layout}-{pair[0]}-to-{pair[1]}")
        for layout, pair in itertools.product(_LAYOUTS, _LANGUAGE_PAIRS)
    ],
)
def test_all_sixteen_combinations(layout, pair):
    span_counts, unit_structure, has_alternatives = _LAYOUTS[layout]
    query, source = pair
    result = st.derive_structure(
        make_question(units=make_units(*span_counts), query_language=query, source_language=source)
    )
    assert result == st.QuestionStructure(
        unit_structure=unit_structure,
        has_alternatives=has_alternatives,
        retrieval_type=f"{query}_to_{source}",
    )


# ---------- F: unanswerable questions ----------

@pytest.mark.parametrize(
    "unanswerable_type, language",
    [
        pytest.param(kind, language, id=f"{kind}-{language}")
        for kind, language in itertools.product(s.UNANSWERABLE_TYPES, ("en", "hi"))
    ],
)
def test_an_unanswerable_question_has_no_structure(unanswerable_type, language):
    question = make_unanswerable(unanswerable_type=unanswerable_type, query_language=language)
    with pytest.raises(ValueError) as caught:
        st.derive_structure(question)
    assert str(caught.value) == UNANSWERABLE_MESSAGE


def test_an_unanswerable_question_that_wrongly_carries_units_is_still_refused_as_unanswerable():
    question = make_unanswerable(units=make_units(1), source_language="en")
    with pytest.raises(ValueError) as caught:
        st.derive_structure(question)
    assert str(caught.value) == UNANSWERABLE_MESSAGE


# ---------- G: invalid input ----------

@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"units": ()}, id="no-units"),
        pytest.param({"units": [make_unit(1)]}, id="units-as-a-list"),
        pytest.param({"units": None}, id="units-is-none"),
        pytest.param({"units": (object(),)}, id="a-unit-that-is-not-an-evidence-unit"),
        pytest.param({"units": (s.EvidenceUnit("u", ()),)}, id="a-unit-with-no-spans"),
        pytest.param({"units": (s.EvidenceUnit("u", [make_span()]),)}, id="spans-as-a-list"),
        pytest.param({"answerable": 1}, id="answerable-is-an-int"),
        pytest.param({"answerable": None}, id="answerable-is-none"),
        pytest.param({"query_language": "fr"}, id="unknown-query-language"),
        pytest.param({"query_language": "EN"}, id="query-language-in-capitals"),
        pytest.param({"query_language": None}, id="query-language-is-none"),
        pytest.param({"query_language": StrSubclass("en")}, id="query-language-is-a-str-subclass"),
        pytest.param({"source_language": None}, id="source-language-is-none"),
        pytest.param({"source_language": "fr"}, id="unknown-source-language"),
        pytest.param({"source_language": StrSubclass("hi")}, id="source-language-is-a-str-subclass"),
    ],
)
def test_an_answerable_question_that_cannot_be_classified_is_refused(overrides):
    with pytest.raises(ValueError) as caught:
        st.derive_structure(make_question(**overrides))
    assert str(caught.value) == INVALID_MESSAGE


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param({}, id="dict"),
        pytest.param("q1", id="string"),
        pytest.param(42, id="int"),
        pytest.param(make_span(), id="span"),
        pytest.param(make_unit(1), id="evidence-unit"),
    ],
)
def test_a_value_that_is_not_a_question_raises_type_error(value):
    with pytest.raises(TypeError) as caught:
        st.derive_structure(value)
    assert str(caught.value) == "q must be a Question"


# ---------- H: the result type ----------

def test_the_result_has_exactly_the_three_fields_in_order():
    names = [field.name for field in dataclasses.fields(st.QuestionStructure)]
    assert names == ["unit_structure", "has_alternatives", "retrieval_type"]


def test_the_result_values_have_the_expected_types():
    result = st.derive_structure(make_question())
    assert type(result) is st.QuestionStructure
    assert type(result.unit_structure) is str
    assert type(result.has_alternatives) is bool
    assert type(result.retrieval_type) is str


def test_the_result_is_frozen_and_slotted():
    result = st.derive_structure(make_question())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.unit_structure = "multi_unit"
    assert "__slots__" in vars(st.QuestionStructure)
    assert not hasattr(result, "__dict__")


def test_the_result_is_hashable_and_compares_by_value():
    first = st.derive_structure(make_question())
    second = st.derive_structure(make_question())
    other = st.derive_structure(make_question(units=make_units(1, 1)))
    assert first == second
    assert hash(first) == hash(second)
    assert first != other
    assert len({first, second, other}) == 2


# ---------- I: purity ----------

def test_the_question_is_not_modified():
    units = make_units(2, 1)
    question = make_question(units=units)
    snapshot = dataclasses.replace(question)
    snapshot_repr = repr(question)
    st.derive_structure(question)
    assert question == snapshot
    assert repr(question) == snapshot_repr
    assert question.units is units


def test_the_result_is_deterministic():
    question = make_question(units=make_units(2, 1), query_language="hi", source_language="en")
    assert st.derive_structure(question) == st.derive_structure(question)


@pytest.mark.parametrize(
    "span_counts",
    [
        pytest.param((2, 1), id="two-then-one"),
        pytest.param((1, 3, 1), id="one-three-one"),
        pytest.param((1, 1, 1), id="all-single"),
    ],
)
def test_unit_and_span_order_does_not_change_the_result(span_counts):
    units = make_units(*span_counts)
    reordered = tuple(
        s.EvidenceUnit(unit.unit_id, tuple(reversed(unit.spans))) for unit in reversed(units)
    )
    assert st.derive_structure(make_question(units=units)) == st.derive_structure(
        make_question(units=reordered)
    )


def test_adding_a_span_to_a_unit_changes_only_has_alternatives():
    before = st.derive_structure(make_question(units=make_units(1, 1)))
    after = st.derive_structure(make_question(units=make_units(2, 1)))
    assert (before.unit_structure, before.retrieval_type) == (after.unit_structure, after.retrieval_type)
    assert (before.has_alternatives, after.has_alternatives) == (False, True)


def test_adding_a_unit_changes_only_the_unit_structure():
    before = st.derive_structure(make_question(units=make_units(1)))
    after = st.derive_structure(make_question(units=make_units(1, 1)))
    assert (before.has_alternatives, before.retrieval_type) == (after.has_alternatives, after.retrieval_type)
    assert (before.unit_structure, after.unit_structure) == ("single_unit", "multi_unit")


def test_changing_the_languages_changes_only_the_retrieval_type():
    before = st.derive_structure(make_question(units=make_units(2, 1)))
    after = st.derive_structure(make_question(units=make_units(2, 1), query_language="hi", source_language="en"))
    assert (before.unit_structure, before.has_alternatives) == (after.unit_structure, after.has_alternatives)
    assert (before.retrieval_type, after.retrieval_type) == ("en_to_en", "hi_to_en")


def test_fields_other_than_units_and_languages_have_no_effect():
    base = st.derive_structure(make_question(units=make_units(2, 1)))
    changed = st.derive_structure(
        make_question(
            units=make_units(2, 1),
            question_id="another-id",
            split="dev",
            question="A different question?",
            author_type="llm_assisted",
            expected_answer="Another answer.",
            difficulty="paraphrase",
            close_distractor=True,
            notes="a note",
        )
    )
    assert base == changed


# ---------- J: messages ----------

def test_the_two_error_messages_are_fixed_and_different():
    assert UNANSWERABLE_MESSAGE != INVALID_MESSAGE


def test_error_messages_never_contain_question_content():
    unanswerable = make_unanswerable(question=CANARY, question_id="canary-id")
    invalid = make_question(
        units=(), question=CANARY, expected_answer=CANARY, question_id="canary-id", notes=CANARY
    )
    for question, message in ((unanswerable, UNANSWERABLE_MESSAGE), (invalid, INVALID_MESSAGE)):
        with pytest.raises(ValueError) as caught:
            st.derive_structure(question)
        assert str(caught.value) == message
        assert CANARY not in repr(caught.value)


# ---------- K: import isolation ----------

def test_importing_the_module_loads_no_heavy_code_and_no_interval_helpers():
    code = (
        "import sys; import app.rag.evaluation.structure; "
        "names = ['torch', 'transformers', 'sqlalchemy', 'fastapi', 'numpy', 'pypdf', "
        "'app.rag.embedder', 'app.rag.retrieval', 'app.rag.indexing', 'app.core.database', 'app.main', "
        "'app.rag.evaluation.intervals']; "
        "print(sorted(n for n in names if n in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
