"""E1.2a schema: constants, dataclasses, issue and error types, structural span validation.

A  constants and patterns
B  dataclasses (frozen, slotted, defaults, hashable)
C  Issue and EvaluationDataError
D  validate_span: valid spans
E  validate_span: invalid spans (codes, order, no cascades)
F  determinism, locations and messages that never leak the quote
G  validate_span argument guards
H  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHA_A = "a" * 64
SHA_B = "b" * 64

DEVANAGARI_KSSA = chr(0x0915) + chr(0x094D) + chr(0x0937)  # one grapheme cluster, 3 code points
EMOJI = chr(0x1F600)  # one code point, two UTF-16 units
LIGATURE_FI = chr(0xFB01)
E_NFC = chr(0x00E9)
E_NFD = "e" + chr(0x0301)
CRLF = chr(13) + chr(10)
LONE_SURROGATE = chr(0xD800)


class StrSubclass(str):
    pass


def digest_or_placeholder(quote):
    try:
        return sha256_text(quote)
    except (AttributeError, UnicodeEncodeError):
        return SHA_B


def make_span(**overrides) -> s.Span:
    """A structurally valid span unless ``overrides`` break it."""
    quote = overrides.get("quote", "hello")
    values = {
        "corpus_key": "doc-a",
        "source_sha256": SHA_A,
        "start": 10,
        "end": 10 + (len(quote) if type(quote) is str else 5),
        "quote": quote,
        "quote_sha256": digest_or_placeholder(quote),
    }
    values.update(overrides)
    return s.Span(**values)


def codes(issues):
    return [issue.code for issue in issues]


def validate(**overrides):
    return s.validate_span(make_span(**overrides), location="q1.units[0].spans[0]")


def make_question(**overrides) -> s.Question:
    values = {
        "question_id": "q1",
        "split": "test",
        "answerable": True,
        "question": "What is it?",
        "query_language": "en",
        "author_type": "human",
        "hindi_reviewed": False,
    }
    values.update(overrides)
    return s.Question(**values)


# ---------- A: constants and patterns ----------

def test_unit_cap_is_twenty():
    assert s.MAX_UNITS_PER_QUESTION == 20
    assert type(s.MAX_UNITS_PER_QUESTION) is int


def test_languages_are_an_immutable_set():
    assert s.LANGUAGES == frozenset({"en", "hi"})
    assert isinstance(s.LANGUAGES, frozenset)


def test_enumerations_are_the_approved_values():
    assert s.SPLITS == ("dev", "test")
    assert s.DOCUMENT_FORMATS == ("pdf", "txt", "md", "docx")
    assert s.MANIFEST_STAGES == ("draft", "ingested")
    assert s.AUTHOR_TYPES == ("human", "llm_assisted")
    assert s.DIFFICULTIES == ("direct", "paraphrase")
    assert s.UNANSWERABLE_TYPES == ("off_topic", "on_topic_not_covered")
    assert s.SEMANTIC_REVIEW_RESULTS == ("confirmed_unanswerable",)
    assert (s.SEVERITY_ERROR, s.SEVERITY_WARNING) == ("error", "warning")


def test_schema_versions():
    assert s.MANIFEST_SCHEMA_VERSION == "eval-manifest-v1"
    assert s.DATASET_SCHEMA_VERSION == "eval-dataset-v1"


@pytest.mark.parametrize(
    "value, ok",
    [
        ("a", True),
        ("fastapi-tutorial-en", True),
        ("py3.12_docs", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("", False),
        ("Doc", False),
        ("-doc", False),
        (".doc", False),
        ("_doc", False),
        ("doc space", False),
        ("doc/slash", False),
        ("doc" + chr(10), False),
        ("doc" + chr(0x0928), False),
    ],
)
def test_corpus_key_pattern(value, ok):
    assert (s.CORPUS_KEY_PATTERN.fullmatch(value) is not None) is ok


@pytest.mark.parametrize(
    "value, ok",
    [
        ("q1", True),
        ("test-001_a", True),
        ("0", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("", False),
        ("Q1", False),
        ("-q", False),
        (".q", False),
        ("q.1", False),
        ("q1" + chr(10), False),
    ],
)
def test_question_id_pattern(value, ok):
    assert (s.QUESTION_ID_PATTERN.fullmatch(value) is not None) is ok


@pytest.mark.parametrize(
    "value, ok",
    [
        ("0" * 64, True),
        ("abcdef0123456789" * 4, True),
        ("A" * 64, False),
        ("a" * 63, False),
        ("a" * 65, False),
        ("g" * 64, False),
        ("a" * 64 + chr(10), False),
        (" " + "a" * 63, False),
        ("", False),
    ],
)
def test_sha256_pattern(value, ok):
    assert (s.SHA256_HEX_PATTERN.fullmatch(value) is not None) is ok


# ---------- B: dataclasses ----------

def sample_objects():
    span = make_span()
    unit = s.EvidenceUnit(unit_id="u1", spans=(span,))
    extraction = s.Extraction(processing_ref="c" * 40, python="3.12.4")
    document = s.ManifestDocument(
        corpus_key="doc-a",
        title="T",
        publisher="P",
        source_url="https://example.invalid/a",
        license="MIT",
        license_url="https://example.invalid/license",
        format="txt",
        language="en",
        source_file_sha256=SHA_A,
        extracted_text_sha256=None,
        extracted_text_length=None,
        extraction=None,
        redistributable=True,
    )
    return [
        span,
        unit,
        s.SemanticReview(reviewer="r1", date="2026-11-10", result="confirmed_unanswerable"),
        s.RetrievalAssistance(date="2026-11-10", method="m", run_ref=None, description="d"),
        make_question(units=(unit,)),
        extraction,
        document,
        s.Manifest(schema_version=s.MANIFEST_SCHEMA_VERSION, corpus_version="1", stage="draft",
                   documents=(document,)),
        s.Issue("CODE", "where", "what"),
    ]


@pytest.mark.parametrize("obj", sample_objects(), ids=lambda o: type(o).__name__)
def test_dataclasses_are_frozen_and_slotted(obj):
    field = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, "changed")
    assert not hasattr(obj, "__dict__")


@pytest.mark.parametrize("obj", sample_objects(), ids=lambda o: type(o).__name__)
def test_dataclasses_are_hashable_and_comparable(obj):
    assert hash(obj) == hash(obj)
    assert obj == obj
    assert len({obj, obj}) == 1


def test_span_interval_is_the_half_open_pair():
    assert make_span(start=3, end=8, quote="abcde").interval == (3, 8)


def test_question_defaults_leave_every_optional_field_unset():
    q = make_question()
    assert q.units == ()
    for name in (
        "hindi_reviewer", "notes", "expected_answer", "source_language", "difficulty",
        "close_distractor", "unanswerable_type", "semantic_review", "retrieval_assistance",
    ):
        assert getattr(q, name) is None


def test_extraction_optional_libraries_default_to_none():
    e = s.Extraction(processing_ref="c" * 40, python="3.12.4")
    assert (e.pypdf, e.python_docx, e.lxml) == (None, None, None)


def test_manifest_document_has_no_defaults_for_the_post_ingestion_fields():
    names = {f.name for f in dataclasses.fields(s.ManifestDocument) if f.default is dataclasses.MISSING}
    assert {"extracted_text_sha256", "extracted_text_length", "extraction"} <= names


# ---------- C: Issue and EvaluationDataError ----------

def test_issue_defaults_to_an_error():
    assert s.Issue("C", "l", "m").severity == "error"
    assert s.Issue("C", "l", "m", s.SEVERITY_WARNING).severity == "warning"


@pytest.mark.parametrize("severity", ["", "ERROR", "info", None, 1])
def test_issue_rejects_an_unknown_severity(severity):
    with pytest.raises(ValueError, match="severity"):
        s.Issue("C", "l", "m", severity)


@pytest.mark.parametrize("args", [(1, "l", "m"), ("C", None, "m"), ("C", "l", b"m"), ("C", "l", StrSubclass("m"))])
def test_issue_rejects_non_str_text(args):
    with pytest.raises(ValueError, match="must be str"):
        s.Issue(*args)


def test_error_carries_every_issue_and_states_only_the_count():
    issues = [s.Issue("A_CODE", "loc1", "secret message one"), s.Issue("B_CODE", "loc2", "secret message two")]
    err = s.EvaluationDataError(issues)
    assert isinstance(err, ValueError)
    assert err.issues == tuple(issues)
    assert [i.code for i in err.issues] == ["A_CODE", "B_CODE"]
    assert str(err) == "2 evaluation data issue(s)"


def test_error_message_never_contains_an_issue_code_location_or_message():
    secret = "SECRET-DOCUMENT-TEXT"
    issue = s.Issue(secret + "-code", secret + "-location", secret + "-message")
    err = s.EvaluationDataError([issue])
    assert secret not in str(err)
    assert secret not in repr(err)
    assert str(err) == "1 evaluation data issue(s)"
    assert err.issues[0].code == secret + "-code"


def test_error_with_no_issues_and_a_list_input_is_stored_as_a_tuple():
    err = s.EvaluationDataError([])
    assert err.issues == ()
    assert str(err) == "0 evaluation data issue(s)"


def test_error_rejects_non_issue_entries():
    with pytest.raises(TypeError, match="Issue"):
        s.EvaluationDataError(["not an issue"])


# ---------- D: valid spans ----------

@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="ascii"),
        pytest.param({"quote": DEVANAGARI_KSSA}, id="devanagari-conjunct-3-code-points"),
        pytest.param({"quote": EMOJI}, id="astral-emoji-1-code-point"),
        pytest.param({"quote": LIGATURE_FI}, id="ligature-fi-no-nfkc"),
        pytest.param({"quote": E_NFC}, id="nfc-e-acute-1"),
        pytest.param({"quote": E_NFD}, id="nfd-e-acute-2"),
        pytest.param({"quote": "a" + CRLF + "b"}, id="crlf-inside-quote"),
        pytest.param({"quote": " padded "}, id="leading-and-trailing-space"),
        pytest.param({"start": 0, "end": 5}, id="start-at-zero"),
        pytest.param({"start": 2**31 - 6, "end": 2**31 - 1}, id="large-offsets"),
        pytest.param({"corpus_key": "a" * 64}, id="longest-corpus-key"),
        pytest.param({"quote": "x" * 5000}, id="long-quote"),
    ],
)
def test_valid_spans_have_no_issues(overrides):
    assert validate(**overrides) == []


def test_offsets_count_code_points_not_utf16_units():
    assert validate(quote=EMOJI, start=4, end=5) == []
    assert codes(validate(quote=EMOJI, start=4, end=6)) == [s.SPAN_QUOTE_LENGTH]


def test_unicode_is_never_normalised():
    # Same visible letter, different code points: the hash and the length both notice.
    assert codes(validate(quote=E_NFC, quote_sha256=sha256_text(E_NFD))) == [s.SPAN_QUOTE_SHA256_MISMATCH]
    assert codes(validate(quote=E_NFD, start=4, end=5)) == [s.SPAN_QUOTE_LENGTH]
    # U+FB01 is not expanded to "fi" (which would be 2 code points).
    assert codes(validate(quote=LIGATURE_FI, start=4, end=6)) == [s.SPAN_QUOTE_LENGTH]


# ---------- E: invalid spans ----------

GOOD_HASH = sha256_text("hello")


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # field types
        pytest.param({"start": True}, [s.SPAN_FIELD_TYPE], id="start-bool-true"),
        pytest.param({"start": False}, [s.SPAN_FIELD_TYPE], id="start-bool-false"),
        pytest.param({"end": True}, [s.SPAN_FIELD_TYPE], id="end-bool"),
        pytest.param({"start": 10.0}, [s.SPAN_FIELD_TYPE], id="start-float"),
        pytest.param({"end": 15.0}, [s.SPAN_FIELD_TYPE], id="end-float"),
        pytest.param({"start": "10"}, [s.SPAN_FIELD_TYPE], id="start-str"),
        pytest.param({"end": None}, [s.SPAN_FIELD_TYPE], id="end-none"),
        pytest.param({"corpus_key": 5}, [s.SPAN_FIELD_TYPE], id="corpus-key-int"),
        pytest.param({"corpus_key": StrSubclass("doc-a")}, [s.SPAN_FIELD_TYPE], id="corpus-key-str-subclass"),
        pytest.param({"source_sha256": 1}, [s.SPAN_FIELD_TYPE], id="source-sha-int"),
        pytest.param({"quote": b"hello"}, [s.SPAN_FIELD_TYPE], id="quote-bytes-no-cascade"),
        pytest.param({"quote_sha256": None}, [s.SPAN_FIELD_TYPE], id="quote-sha-none"),
        # offset range
        pytest.param({"start": -1, "end": 4}, [s.SPAN_OFFSET_RANGE], id="negative-start"),
        pytest.param({"start": 10, "end": 10}, [s.SPAN_OFFSET_RANGE], id="start-equals-end"),
        pytest.param({"start": 15, "end": 10}, [s.SPAN_OFFSET_RANGE], id="start-after-end"),
        # quote
        pytest.param({"quote": "", "end": 15}, [s.SPAN_QUOTE_EMPTY], id="empty-quote"),
        pytest.param({"quote": "", "end": 15, "quote_sha256": SHA_B},
                     [s.SPAN_QUOTE_EMPTY, s.SPAN_QUOTE_SHA256_MISMATCH], id="empty-quote-wrong-hash"),
        pytest.param({"end": 16}, [s.SPAN_QUOTE_LENGTH], id="end-too-large"),
        pytest.param({"end": 14}, [s.SPAN_QUOTE_LENGTH], id="end-too-small"),
        pytest.param({"quote_sha256": SHA_B}, [s.SPAN_QUOTE_SHA256_MISMATCH], id="wrong-quote-hash"),
        pytest.param({"end": 16, "quote_sha256": SHA_B},
                     [s.SPAN_QUOTE_LENGTH, s.SPAN_QUOTE_SHA256_MISMATCH], id="length-and-hash"),
        pytest.param({"quote": LONE_SURROGATE, "end": 11}, [s.SPAN_QUOTE_ENCODING], id="lone-surrogate-no-crash"),
        # quote hash format (the mismatch check is skipped: reported once)
        pytest.param({"quote_sha256": GOOD_HASH.upper()}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-uppercase"),
        pytest.param({"quote_sha256": GOOD_HASH[:63]}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-63"),
        pytest.param({"quote_sha256": GOOD_HASH + "0"}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-65"),
        pytest.param({"quote_sha256": "g" * 64}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-non-hex"),
        pytest.param({"quote_sha256": GOOD_HASH + chr(10)}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-newline"),
        pytest.param({"quote_sha256": ""}, [s.SPAN_QUOTE_SHA256_FORMAT], id="quote-hash-empty"),
        # source hash format
        pytest.param({"source_sha256": "A" * 64}, [s.SPAN_SOURCE_SHA256_FORMAT], id="source-hash-uppercase"),
        pytest.param({"source_sha256": "a" * 63}, [s.SPAN_SOURCE_SHA256_FORMAT], id="source-hash-63"),
        pytest.param({"source_sha256": "a" * 64 + chr(10)}, [s.SPAN_SOURCE_SHA256_FORMAT], id="source-hash-newline"),
        # corpus key pattern
        pytest.param({"corpus_key": ""}, [s.SPAN_CORPUS_KEY], id="corpus-key-empty"),
        pytest.param({"corpus_key": "Doc"}, [s.SPAN_CORPUS_KEY], id="corpus-key-uppercase"),
        pytest.param({"corpus_key": "-doc"}, [s.SPAN_CORPUS_KEY], id="corpus-key-leading-hyphen"),
        pytest.param({"corpus_key": ".doc"}, [s.SPAN_CORPUS_KEY], id="corpus-key-leading-dot"),
        pytest.param({"corpus_key": "a" * 65}, [s.SPAN_CORPUS_KEY], id="corpus-key-65"),
        pytest.param({"corpus_key": "doc space"}, [s.SPAN_CORPUS_KEY], id="corpus-key-space"),
        pytest.param({"corpus_key": "doc/slash"}, [s.SPAN_CORPUS_KEY], id="corpus-key-slash"),
        # UTF-16 versus code points, NFC versus NFD
        pytest.param({"quote": EMOJI, "end": 12}, [s.SPAN_QUOTE_LENGTH], id="emoji-counted-as-utf16"),
        pytest.param({"quote": E_NFD, "end": 11}, [s.SPAN_QUOTE_LENGTH], id="nfd-counted-as-one"),
    ],
)
def test_invalid_spans_report_exactly_these_codes(overrides, expected):
    assert codes(validate(**overrides)) == expected


def test_a_mistake_that_makes_a_later_check_meaningless_is_reported_once():
    # Bad type for end: no range check and no length check follow.
    assert codes(validate(end=15.0)) == [s.SPAN_FIELD_TYPE]
    # Invalid range: no length check follows.
    assert codes(validate(start=20, end=10)) == [s.SPAN_OFFSET_RANGE]
    # Bad quote hash format: no mismatch check follows.
    assert codes(validate(quote_sha256="zz")) == [s.SPAN_QUOTE_SHA256_FORMAT]


def test_all_six_wrong_types_are_reported_in_field_order():
    span = s.Span(corpus_key=1, source_sha256=2, start="3", end=4.0, quote=5, quote_sha256=6)
    issues = s.validate_span(span, location="q")
    assert codes(issues) == [s.SPAN_FIELD_TYPE] * 6
    assert [i.location for i in issues] == [
        "q.corpus_key", "q.source_sha256", "q.quote", "q.quote_sha256", "q.start", "q.end",
    ]


def test_several_problems_come_out_in_the_documented_order():
    issues = validate(corpus_key="Bad", source_sha256="x", quote_sha256="y", start=-1, end=4)
    assert codes(issues) == [
        s.SPAN_CORPUS_KEY,
        s.SPAN_SOURCE_SHA256_FORMAT,
        s.SPAN_QUOTE_SHA256_FORMAT,
        s.SPAN_OFFSET_RANGE,
    ]
    assert [i.location for i in issues] == [
        "q1.units[0].spans[0].corpus_key",
        "q1.units[0].spans[0].source_sha256",
        "q1.units[0].spans[0].quote_sha256",
        "q1.units[0].spans[0].offsets",
    ]


# ---------- F: determinism, severity, no leaks ----------

def test_results_are_deterministic_and_all_errors():
    first = validate(corpus_key="Bad", end=16, quote_sha256=SHA_B)
    second = validate(corpus_key="Bad", end=16, quote_sha256=SHA_B)
    assert first == second
    assert all(issue.severity == s.SEVERITY_ERROR for issue in first)


def test_hash_mismatch_message_does_not_depend_on_the_quote():
    one = validate(quote="alpha", quote_sha256=SHA_B)
    two = validate(quote="omega", quote_sha256=SHA_B)
    assert [i.message for i in one] == [i.message for i in two]


def test_no_issue_or_error_text_contains_the_quote():
    secret = "SECRET-QUOTE-TEXT"
    issues = []
    issues += validate(quote=secret, quote_sha256=SHA_B, end=99)          # length + hash mismatch
    issues += validate(quote=secret, quote_sha256="not-a-hash")           # hash format
    issues += validate(quote=secret + LONE_SURROGATE, quote_sha256=SHA_B)  # encoding
    assert issues
    for issue in issues:
        assert secret not in issue.location
        assert secret not in issue.message
    assert secret not in str(s.EvaluationDataError(issues))


# ---------- G: argument guards ----------

def test_validate_span_rejects_a_non_span():
    with pytest.raises(TypeError, match="Span"):
        s.validate_span("not a span", location="q")


@pytest.mark.parametrize("location", [None, 5, b"q"])
def test_validate_span_rejects_a_non_str_location(location):
    with pytest.raises(TypeError, match="location"):
        s.validate_span(make_span(), location=location)


def test_location_is_keyword_only():
    with pytest.raises(TypeError):
        s.validate_span(make_span(), "q")


# ---------- H: import isolation ----------

def test_importing_the_schema_loads_no_model_database_or_retrieval_code():
    code = (
        "import sys; import app.rag.evaluation.schema; "
        "names = ['torch', 'transformers', 'sqlalchemy', 'fastapi', 'numpy', 'pypdf', "
        "'app.rag.embedder', 'app.rag.retrieval', 'app.rag.indexing', 'app.core.database', 'app.main']; "
        "print(sorted(n for n in names if n in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
