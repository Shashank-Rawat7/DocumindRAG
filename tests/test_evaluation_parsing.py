"""E1.2b parsing: strict JSON and JSONL into the E1.2a dataclasses, and canonical hashing.

A  canonical_json
B  parse_manifest: accepted input
C  parse_manifest: rejected input (exact issue codes and locations)
D  parse_questions: accepted input
E  parse_questions: rejected input
F  manifest_hash and dataset_hash
G  information leakage
H  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import parsing as p
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHA_A = "a" * 64
SHA_B = "b" * 64
REF = "c" * 40
SECRET = "SECRET-INPUT-TEXT"
HINDI = chr(0x0939) + chr(0x093F) + chr(0x0928) + chr(0x094D) + chr(0x0926) + chr(0x0940)
LS = chr(0x2028)  # LINE SEPARATOR: legal inside a JSON string, must not split a JSONL record
CRLF = chr(13) + chr(10)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def found(fn, *args, **kwargs):
    """The (code, location) pairs of the EvaluationDataError that fn raises."""
    with pytest.raises(s.EvaluationDataError) as caught:
        fn(*args, **kwargs)
    return [(i.code, i.location) for i in caught.value.issues]


# ---------- builders ----------

def doc(**overrides) -> dict:
    base = {
        "corpus_key": "doc-a",
        "title": "Title",
        "publisher": "Publisher",
        "source_url": "https://example.invalid/a",
        "license": "MIT",
        "license_url": "https://example.invalid/license",
        "format": "txt",
        "language": "en",
        "source_file_sha256": SHA_A,
        "extracted_text_sha256": None,
        "extracted_text_length": None,
        "extraction": None,
        "redistributable": True,
    }
    base.update(overrides)
    return base


def manifest(documents=None, **overrides) -> dict:
    base = {
        "schema_version": "eval-manifest-v1",
        "corpus_version": "c1",
        "stage": "draft",
        "documents": [doc()] if documents is None else documents,
    }
    base.update(overrides)
    return base


def dumps(value) -> str:
    return json.dumps(value)


def question(**overrides) -> dict:
    base = {
        "question_id": "q1",
        "split": "test",
        "answerable": True,
        "question": "What is it?",
        "query_language": "en",
        "author_type": "human",
        "hindi_reviewed": False,
    }
    base.update(overrides)
    return base


def span_dict(**overrides) -> dict:
    base = {"corpus_key": "doc-a", "source_sha256": SHA_A, "start": 3, "end": 8, "quote": "hello", "quote_sha256": sha("hello")}
    base.update(overrides)
    return base


def jsonl(*records, eol="\n", final=True) -> str:
    text = eol.join(json.dumps(r) if not isinstance(r, str) else r for r in records)
    return text + (eol if final else "")


# ---------- A: canonical_json ----------

def test_canonical_json_sorts_keys_and_is_compact():
    assert p.canonical_json({"b": 1, "a": [1, 2, {"d": None, "c": True}]}) == '{"a":[1,2,{"c":true,"d":null}],"b":1}'


def test_canonical_json_escapes_non_ascii():
    assert p.canonical_json({"k": HINDI}) == '{"k":"' + "".join(f"\\u{ord(c):04x}" for c in HINDI) + '"}'
    assert p.canonical_json(chr(0x1F600)) == '"\\ud83d\\ude00"'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(bad):
    with pytest.raises(ValueError):
        p.canonical_json({"x": bad})


# ---------- B: parse_manifest accepted ----------

def test_minimal_draft_manifest():
    m = p.parse_manifest(dumps(manifest()))
    assert type(m) is s.Manifest
    assert (m.schema_version, m.corpus_version, m.stage) == ("eval-manifest-v1", "c1", "draft")
    assert type(m.documents) is tuple and len(m.documents) == 1
    d = m.documents[0]
    assert type(d) is s.ManifestDocument
    assert d.corpus_key == "doc-a" and d.format == "txt" and d.redistributable is True
    assert (d.extracted_text_sha256, d.extracted_text_length, d.extraction) == (None, None, None)


@pytest.mark.parametrize(
    "fmt, extra",
    [
        ("txt", {}),
        ("md", {}),
        ("pdf", {"pypdf": "5.0.0"}),
        ("docx", {"python_docx": "1.1.0", "lxml": "5.0.0"}),
    ],
)
def test_ingested_documents_for_every_format(fmt, extra):
    extraction = {"processing_ref": REF, "python": "3.12.4", **extra}
    raw = manifest(
        [doc(format=fmt, extracted_text_sha256=SHA_B, extracted_text_length=10, extraction=extraction)],
        stage="ingested",
    )
    d = p.parse_manifest(dumps(raw)).documents[0]
    assert type(d.extraction) is s.Extraction
    assert d.extraction.processing_ref == REF and d.extraction.python == "3.12.4"
    for name, value in extra.items():
        assert getattr(d.extraction, name) == value
    assert d.extracted_text_length == 10 and type(d.extracted_text_length) is int


def test_pretty_printed_and_crlf_input_parse_the_same():
    raw = manifest([doc(), doc(corpus_key="doc-b")])
    lf = json.dumps(raw, indent=2)
    crlf = lf.replace("\n", CRLF)
    assert p.parse_manifest(lf) == p.parse_manifest(crlf) == p.parse_manifest(dumps(raw))


def test_non_ascii_metadata_is_kept_exactly():
    m = p.parse_manifest(dumps(manifest([doc(title=HINDI, publisher=chr(0x1F600))])))
    assert m.documents[0].title == HINDI and m.documents[0].publisher == chr(0x1F600)


def test_applicable_library_may_be_null_at_parse_time():
    # Parsing accepts the key; validate_manifest rejects the null value.
    raw = manifest([doc(format="pdf", extraction={"processing_ref": REF, "python": "3.12", "pypdf": None})])
    assert p.parse_manifest(dumps(raw)).documents[0].extraction.pypdf is None


def test_values_are_not_checked_at_parse_time():
    raw = manifest([doc(corpus_key="NOT A KEY", format="rtf", language="fr", title="")], stage="bogus",
                   schema_version="v9")
    m = p.parse_manifest(dumps(raw))
    assert (m.stage, m.schema_version, m.documents[0].corpus_key, m.documents[0].format) == ("bogus", "v9", "NOT A KEY", "rtf")


def test_empty_document_list_parses():
    assert p.parse_manifest(dumps(manifest([]))).documents == ()


def test_unknown_format_allows_any_library_key():
    raw = manifest([doc(format="rtf", extraction={"processing_ref": REF, "python": "3", "pypdf": "1", "lxml": "2"})])
    assert p.parse_manifest(dumps(raw)).documents[0].extraction.lxml == "2"


# ---------- C: parse_manifest rejected ----------

ROW = pytest.param

MANIFEST_REJECTED = [
    ROW("", [(p.PARSE_SYNTAX, "manifest line 1 column 1")], id="empty"),
    ROW('{"a": 1', [(p.PARSE_SYNTAX, "manifest line 1 column 8")], id="truncated"),
    ROW(dumps(manifest()) + " x", [(p.PARSE_SYNTAX, None)], id="trailing-garbage"),
    ROW("[]", [(p.PARSE_NOT_OBJECT, "manifest")], id="array-top-level"),
    ROW("5", [(p.PARSE_NOT_OBJECT, "manifest")], id="number-top-level"),
    ROW("null", [(p.PARSE_NOT_OBJECT, "manifest")], id="null-top-level"),
    ROW(chr(0xFEFF) + dumps(manifest()), [(p.PARSE_BOM, "manifest")], id="bom"),
    ROW(dumps(manifest()).replace('"stage": "draft"', '"stage": "draft", "stage": "draft"'),
        [(p.PARSE_DUPLICATE_KEY, "manifest")], id="duplicate-top-level-key"),
    ROW(dumps(manifest()).replace('"title": "Title"', '"title": "A", "title": "B"'),
        [(p.PARSE_DUPLICATE_KEY, "manifest")], id="duplicate-nested-key"),
    ROW(dumps(manifest()).replace('"c1"', 'NaN'), [(p.PARSE_NON_FINITE, "manifest")], id="nan"),
    ROW(dumps(manifest()).replace('"c1"', 'Infinity'), [(p.PARSE_NON_FINITE, "manifest")], id="infinity"),
    ROW(dumps(manifest()).replace('"c1"', '-Infinity'), [(p.PARSE_NON_FINITE, "manifest")], id="negative-infinity"),
    ROW(dumps(manifest()).replace('"c1"', '1e999'), [(p.PARSE_NON_FINITE, "manifest")], id="overflowing-float"),
    ROW(dumps(manifest()).replace('"c1"', '-1e999'), [(p.PARSE_NON_FINITE, "manifest")], id="negative-overflowing-float"),
    ROW("[" * 100000, [(p.PARSE_TOO_DEEP, "manifest")], id="too-deep"),
    # closed schema
    ROW(dumps(manifest(extra=1)), [(p.PARSE_UNKNOWN_FIELD, "manifest")], id="unknown-top-level-field"),
    ROW(dumps(manifest([doc(extra=1)])), [(p.PARSE_UNKNOWN_FIELD, "manifest.documents[0]")], id="unknown-document-field"),
    ROW(dumps(manifest([doc(extraction={"processing_ref": REF, "python": "3", "extra": 1})])),
        [(p.PARSE_UNKNOWN_FIELD, "manifest.documents[0].extraction")], id="unknown-extraction-field"),
    ROW(dumps(manifest(a=1, b=2)),
        [(p.PARSE_UNKNOWN_FIELD, "manifest")] * 2, id="two-unknown-fields-two-issues"),
    # missing keys
    ROW("{}", [(p.PARSE_MISSING_FIELD, f"manifest.{n}") for n in ("schema_version", "corpus_version", "stage", "documents")],
        id="empty-object"),
    ROW(dumps({k: v for k, v in manifest().items() if k != "stage"}), [(p.PARSE_MISSING_FIELD, "manifest.stage")],
        id="missing-stage"),
    ROW(dumps(manifest([{k: v for k, v in doc().items() if k != "extraction"}])),
        [(p.PARSE_MISSING_FIELD, "manifest.documents[0].extraction")], id="missing-extraction-key"),
    ROW(dumps(manifest([{k: v for k, v in doc().items() if k != "extracted_text_sha256"}])),
        [(p.PARSE_MISSING_FIELD, "manifest.documents[0].extracted_text_sha256")], id="missing-hash-key"),
    # types
    ROW(dumps(manifest(stage=None)), [(p.PARSE_FIELD_TYPE, "manifest.stage")], id="null-stage"),
    ROW(dumps(manifest(corpus_version=1)), [(p.PARSE_FIELD_TYPE, "manifest.corpus_version")], id="int-corpus-version"),
    ROW(dumps(manifest({"a": 1})), [(p.PARSE_FIELD_TYPE, "manifest.documents")], id="documents-is-object"),
    ROW(dumps(manifest(["x"])), [(p.PARSE_NOT_OBJECT, "manifest.documents[0]")], id="document-is-string"),
    ROW(dumps(manifest([doc(title=5)])), [(p.PARSE_FIELD_TYPE, "manifest.documents[0].title")], id="int-title"),
    ROW(dumps(manifest([doc(title=None)])), [(p.PARSE_FIELD_TYPE, "manifest.documents[0].title")], id="null-title"),
    ROW(dumps(manifest([doc(redistributable="true")])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].redistributable")], id="string-bool"),
    ROW(dumps(manifest([doc(redistributable=1)])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].redistributable")], id="int-as-bool"),
    ROW(dumps(manifest([doc(extracted_text_length=True)])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].extracted_text_length")], id="bool-as-int"),
    ROW(dumps(manifest([doc(extracted_text_length=1.0)])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].extracted_text_length")], id="float-as-int"),
    ROW(dumps(manifest([doc(extracted_text_length="5")])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].extracted_text_length")], id="string-as-int"),
    ROW(dumps(manifest([doc(extracted_text_sha256=5)])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].extracted_text_sha256")], id="int-hash"),
    ROW(dumps(manifest([doc(extraction=[])])), [(p.PARSE_NOT_OBJECT, "manifest.documents[0].extraction")], id="extraction-list"),
    ROW(dumps(manifest([doc(extraction="x")])), [(p.PARSE_NOT_OBJECT, "manifest.documents[0].extraction")], id="extraction-string"),
    ROW(dumps(manifest([doc(extraction={"processing_ref": 1, "python": None})])),
        [(p.PARSE_FIELD_TYPE, "manifest.documents[0].extraction.processing_ref"),
         (p.PARSE_FIELD_TYPE, "manifest.documents[0].extraction.python")], id="extraction-wrong-types"),
    ROW(dumps(manifest([doc(extraction={"python": "3"})])),
        [(p.PARSE_MISSING_FIELD, "manifest.documents[0].extraction.processing_ref")], id="extraction-missing-ref"),
    # raw extraction key presence by format
    ROW(dumps(manifest([doc(format="txt", extraction={"processing_ref": REF, "python": "3", "pypdf": "1"})])),
        [(p.PARSE_INAPPLICABLE_FIELD, "manifest.documents[0].extraction.pypdf")], id="txt-with-pypdf"),
    ROW(dumps(manifest([doc(format="md", extraction={"processing_ref": REF, "python": "3", "lxml": None})])),
        [(p.PARSE_INAPPLICABLE_FIELD, "manifest.documents[0].extraction.lxml")], id="md-with-null-lxml-key"),
    ROW(dumps(manifest([doc(format="pdf", extraction={"processing_ref": REF, "python": "3"})])),
        [(p.PARSE_MISSING_FIELD, "manifest.documents[0].extraction.pypdf")], id="pdf-missing-pypdf"),
    ROW(dumps(manifest([doc(format="pdf", extraction={"processing_ref": REF, "python": "3", "pypdf": "1", "lxml": "2"})])),
        [(p.PARSE_INAPPLICABLE_FIELD, "manifest.documents[0].extraction.lxml")], id="pdf-with-lxml"),
    ROW(dumps(manifest([doc(format="docx", extraction={"processing_ref": REF, "python": "3", "python_docx": "1"})])),
        [(p.PARSE_MISSING_FIELD, "manifest.documents[0].extraction.lxml")], id="docx-missing-lxml"),
    ROW(dumps(manifest([doc(format="docx", extraction={"processing_ref": REF, "python": "3", "python_docx": "1",
                                                        "lxml": "2", "pypdf": "3"})])),
        [(p.PARSE_INAPPLICABLE_FIELD, "manifest.documents[0].extraction.pypdf")], id="docx-with-pypdf"),
    # independent problems are all reported, in a stable order
    ROW(dumps(manifest([doc(title=1), doc(corpus_key=2, extra=1)], stage=None)),
        [(p.PARSE_FIELD_TYPE, "manifest.stage"),
         (p.PARSE_FIELD_TYPE, "manifest.documents[0].title"),
         (p.PARSE_UNKNOWN_FIELD, "manifest.documents[1]"),
         (p.PARSE_FIELD_TYPE, "manifest.documents[1].corpus_key")], id="several-problems-in-order"),
]


@pytest.mark.parametrize("text, expected", MANIFEST_REJECTED)
def test_manifest_rejections(text, expected):
    got = found(p.parse_manifest, text)
    if expected == [(p.PARSE_SYNTAX, None)]:
        assert len(got) == 1 and got[0][0] == p.PARSE_SYNTAX and got[0][1].startswith("manifest line ")
    else:
        assert got == expected


@pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits") or not 0 < sys.get_int_max_str_digits() <= 4300,
    reason="interpreter integer digit limit is not at its default",
)
def test_integer_beyond_the_digit_limit_is_rejected():
    text = '{"x": ' + "1" * 5000 + "}"
    assert found(p.parse_manifest, text) == [(p.PARSE_NUMBER, "manifest")]


def test_parse_manifest_requires_str():
    for bad in (b"{}", None, 5, bytearray(b"{}")):
        with pytest.raises(TypeError):
            p.parse_manifest(bad)


# ---------- C2: every duplicate key, non-finite number and BOM is collected ----------

TOKEN = '"@@"'


def mtext(replacement, **fields):
    """Manifest JSON whose value "@@" is replaced by raw JSON text (for example NaN)."""
    return dumps(manifest(**fields)).replace(TOKEN, replacement)


def qtext(replacement, **fields):
    return jsonl(question(**fields)).replace(TOKEN, replacement)


def test_duplicate_keys_in_different_objects_are_all_reported():
    text = dumps(manifest([doc(), doc(corpus_key="doc-b")]))
    text = text.replace('"title": "Title"', '"title": "Title", "title": "Other"')      # in both documents
    text = text.replace('"stage": "draft"', '"stage": "draft", "stage": "draft"')       # at the top level
    assert text.count("Other") == 2
    assert found(p.parse_manifest, text) == [(p.PARSE_DUPLICATE_KEY, "manifest")] * 3


def test_several_duplicates_of_keys_inside_one_object_are_each_reported():
    text = dumps(manifest()).replace('"title": "Title"', '"title": "A", "title": "B", "title": "C"')
    assert found(p.parse_manifest, text) == [(p.PARSE_DUPLICATE_KEY, "manifest")] * 2


def test_several_non_finite_numbers_in_one_manifest_are_all_reported():
    text = dumps(manifest([doc(), doc(corpus_key="doc-b")]))
    text = text.replace('"c1"', "NaN", 1)
    text = text.replace('"Title"', "-Infinity", 1)
    text = text.replace("null, \"extracted_text_length\": null", "Infinity, \"extracted_text_length\": 1e999", 1)
    assert text.count("NaN") == 1 and text.count("1e999") == 1
    assert found(p.parse_manifest, text) == [(p.PARSE_NON_FINITE, "manifest")] * 4


def test_a_manifest_with_a_bom_and_a_structural_problem_reports_both():
    text = chr(0xFEFF) + dumps(manifest(extra=1))
    assert found(p.parse_manifest, text) == [(p.PARSE_BOM, "manifest"), (p.PARSE_UNKNOWN_FIELD, "manifest")]
    text = chr(0xFEFF) + dumps(manifest([doc(title=5)]))
    assert found(p.parse_manifest, text) == [(p.PARSE_BOM, "manifest"), (p.PARSE_FIELD_TYPE, "manifest.documents[0].title")]


def test_a_bom_alone_is_still_an_error_and_a_valid_manifest_behind_it_is_never_returned():
    with pytest.raises(s.EvaluationDataError) as caught:
        p.parse_manifest(chr(0xFEFF) + dumps(manifest()))
    assert [i.code for i in caught.value.issues] == [p.PARSE_BOM]


def test_a_bom_then_unreadable_json_reports_both():
    assert found(p.parse_manifest, chr(0xFEFF) + "{") == [(p.PARSE_BOM, "manifest"), (p.PARSE_SYNTAX, "manifest line 1 column 2")]


def test_manifest_issue_order_is_bom_then_loader_issues_then_structural_issues():
    text = dumps(manifest([doc(), doc(corpus_key="doc-b")], extra=1))
    text = text.replace('"c1"', "NaN", 1)
    text = text.replace('"title": "Title"', '"title": "Title", "title": "X"', 1)
    text = text.replace('"Publisher"', "NaN", 2)
    text = chr(0xFEFF) + text
    got = found(p.parse_manifest, text)
    assert [c for c, _ in got] == [
        p.PARSE_BOM, p.PARSE_NON_FINITE, p.PARSE_NON_FINITE, p.PARSE_DUPLICATE_KEY, p.PARSE_NON_FINITE,
        p.PARSE_UNKNOWN_FIELD,
    ]
    assert got == found(p.parse_manifest, text)  # stable across calls


def test_loader_issues_follow_the_order_the_json_text_is_read():
    # NaN is met while scanning; a duplicate is recorded when its object closes.
    text = (
        '{"schema_version": "eval-manifest-v1", "corpus_version": NaN, "stage": "draft", "documents": ['
        + dumps(doc()).replace('"title": "Title"', '"title": "Title", "title": "X"')
        + ", "
        + dumps(doc(corpus_key="doc-b")).replace('"Publisher"', "NaN")
        + "]}"
    )
    assert [c for c, _ in found(p.parse_manifest, text)] == [
        p.PARSE_NON_FINITE, p.PARSE_DUPLICATE_KEY, p.PARSE_NON_FINITE,
    ]


def test_unreadable_json_reports_only_the_syntax_problem():
    assert [c for c, _ in found(p.parse_manifest, '{"a": NaN, "a": 1, "b": }')] == [p.PARSE_SYNTAX]
    assert [c for c, _ in found(p.parse_manifest, '{"a": NaN, "a": 1} trailing')] == [p.PARSE_SYNTAX]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(mtext("NaN", documents=TOKEN[1:-1]), id="documents"),
        pytest.param(mtext("NaN", documents=[TOKEN[1:-1]]), id="document-element"),
        pytest.param(mtext("NaN", documents=[doc(extraction=TOKEN[1:-1])]), id="extraction-object"),
        pytest.param(mtext("Infinity", documents=[doc(extracted_text_length=TOKEN[1:-1])]), id="int-field"),
        pytest.param(mtext("-Infinity", documents=[doc(redistributable=TOKEN[1:-1])]), id="bool-field"),
        pytest.param(mtext("1e999", documents=[doc(title=TOKEN[1:-1])]), id="string-field"),
    ],
)
def test_a_non_finite_number_is_reported_once_and_causes_no_extra_type_issue(text):
    assert found(p.parse_manifest, text) == [(p.PARSE_NON_FINITE, "manifest")]


def test_a_non_finite_number_in_the_top_level_position_is_reported_once():
    assert found(p.parse_manifest, "NaN") == [(p.PARSE_NON_FINITE, "manifest")]


def test_questions_collect_loader_issues_bom_and_structure_in_line_order():
    text = (
        chr(0xFEFF)
        + json.dumps(question(extra=1)).replace('"What is it?"', "NaN") + "\n"
        + json.dumps(question(question_id="q2")).replace('"question_id": "q2"', '"question_id": "q2", "question_id": "q3"') + "\n"
        + "\n"
        + "{\n"
    )
    assert found(p.parse_questions, text, split="test") == [
        (p.PARSE_BOM, "line 1"),
        (p.PARSE_NON_FINITE, "line 1"),
        (p.PARSE_UNKNOWN_FIELD, "line 1"),
        (p.PARSE_DUPLICATE_KEY, "line 2"),
        (p.PARSE_BLANK_LINE, "line 3"),
        (p.PARSE_SYNTAX, "line 4 column 2"),
    ]


def test_questions_report_every_duplicate_and_non_finite_value_on_a_line():
    line = (json.dumps(question()).replace('"split": "test"', '"split": "test", "split": "test"')
            .replace('"What is it?"', "NaN").replace('"human"', "Infinity"))
    # numbers are met while scanning; the duplicate is recorded when the object closes
    assert found(p.parse_questions, line, split="test") == [
        (p.PARSE_NON_FINITE, "line 1"), (p.PARSE_NON_FINITE, "line 1"), (p.PARSE_DUPLICATE_KEY, "line 1"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(qtext("NaN", units=TOKEN[1:-1]), id="units"),
        pytest.param(qtext("NaN", units=[TOKEN[1:-1]]), id="unit"),
        pytest.param(qtext("NaN", units=[{"unit_id": "u", "spans": [TOKEN[1:-1]]}]), id="span"),
        pytest.param(qtext("Infinity", units=[{"unit_id": "u", "spans": [span_dict(start=TOKEN[1:-1])]}]), id="span-int"),
        pytest.param(qtext("NaN", semantic_review=TOKEN[1:-1]), id="review"),
        pytest.param(qtext("-Infinity", retrieval_assistance={"date": "d", "method": TOKEN[1:-1], "run_ref": None, "description": "x"}),
                     id="assistance-field"),
        pytest.param(qtext("1e999", answerable=TOKEN[1:-1]), id="bool-field"),
    ],
)
def test_a_non_finite_question_value_is_reported_once_and_causes_no_extra_issue(text):
    assert found(p.parse_questions, text, split="test") == [(p.PARSE_NON_FINITE, "line 1")]


def test_collected_loader_issues_never_leak_input():
    texts = [
        chr(0xFEFF) + '{"%s": NaN, "%s": 1, "%s": 2}' % (SECRET, SECRET, SECRET),
        dumps(manifest()).replace('"title": "Title"', '"title": "%s", "title": "%s"' % (SECRET, SECRET))
        .replace('"c1"', "NaN").replace('"Publisher"', "Infinity"),
        chr(0xFEFF) + dumps(manifest(**{SECRET: 1})).replace('"c1"', "NaN"),
    ]
    for text in texts:
        with pytest.raises(s.EvaluationDataError) as caught:
            p.parse_manifest(text)
        assert len(caught.value.issues) >= 2
        for piece in every_text(caught.value):
            assert SECRET not in piece
    line = (json.dumps(question(**{SECRET: 1, "notes": SECRET}))
            .replace('"notes": "%s"' % SECRET, '"notes": "%s", "notes": "%s"' % (SECRET, SECRET))
            .replace('"What is it?"', "NaN"))
    with pytest.raises(s.EvaluationDataError) as caught:
        p.parse_questions(chr(0xFEFF) + line, split="test")
    assert len(caught.value.issues) >= 3
    for piece in every_text(caught.value):
        assert SECRET not in piece
    assert str(caught.value) == f"{len(caught.value.issues)} evaluation data issue(s)"


# ---------- D: parse_questions accepted ----------

def test_minimal_question_takes_dataclass_defaults():
    (q,) = p.parse_questions(jsonl(question()), split="test")
    assert type(q) is s.Question
    assert (q.question_id, q.split, q.answerable, q.query_language, q.author_type, q.hindi_reviewed) == (
        "q1", "test", True, "en", "human", False)
    assert q.units == () and q.expected_answer is None and q.semantic_review is None and q.retrieval_assistance is None


def test_full_answerable_question():
    unit = {"unit_id": "u1", "spans": [span_dict(), span_dict(start=20, end=25)]}
    raw = question(expected_answer="Answer", units=[unit, {"unit_id": "u2", "spans": [span_dict(start=40, end=45)]}],
                   source_language="en", difficulty="direct", close_distractor=True, notes="n",
                   hindi_reviewer=None)
    (q,) = p.parse_questions(jsonl(raw), split="test")
    assert type(q.units) is tuple and [u.unit_id for u in q.units] == ["u1", "u2"]
    assert type(q.units[0].spans) is tuple and type(q.units[0].spans[0]) is s.Span
    assert q.units[0].spans[1].interval == (20, 25)
    assert q.close_distractor is True and q.difficulty == "direct" and q.notes == "n"


def test_unanswerable_question_with_review_and_assistance():
    raw = question(
        split="dev", answerable=False, unanswerable_type="off_topic",
        semantic_review={"reviewer": "r1", "date": "2026-11-10", "result": "confirmed_unanswerable"},
        retrieval_assistance={"date": "2026-11-10", "method": None, "run_ref": None, "description": "d"},
    )
    (q,) = p.parse_questions(jsonl(raw), split="dev")
    assert type(q.semantic_review) is s.SemanticReview and q.semantic_review.reviewer == "r1"
    assert type(q.retrieval_assistance) is s.RetrievalAssistance
    assert q.retrieval_assistance.method is None and q.retrieval_assistance.description == "d"


def test_explicit_null_where_the_schema_allows_it():
    raw = question(hindi_reviewer=None, notes=None, expected_answer=None, source_language=None, difficulty=None,
                   close_distractor=None, unanswerable_type=None, semantic_review=None, retrieval_assistance=None)
    (q,) = p.parse_questions(jsonl(raw), split="test")
    assert q.notes is None and q.semantic_review is None and q.close_distractor is None


@pytest.mark.parametrize("eol", ["\n", CRLF])
@pytest.mark.parametrize("final", [True, False])
def test_line_endings_and_final_terminator(eol, final):
    text = jsonl(question(question_id="q1"), question(question_id="q2"), eol=eol, final=final)
    assert [q.question_id for q in p.parse_questions(text, split="test")] == ["q1", "q2"]


def test_file_order_is_preserved():
    text = jsonl(question(question_id="q9"), question(question_id="q1"), question(question_id="q5"))
    assert [q.question_id for q in p.parse_questions(text, split="test")] == ["q9", "q1", "q5"]


def test_empty_file_is_an_empty_tuple():
    assert p.parse_questions("", split="test") == ()
    assert p.parse_questions("", split="dev") == ()


def test_unicode_line_separators_inside_strings_do_not_split_records():
    raw = question(question="a" + LS + "b" + chr(0x2029) + "c" + chr(0x85) + chr(0x0B) + chr(0x0C) + chr(0x1C))
    text = json.dumps(raw, ensure_ascii=False) + "\n"
    assert LS in text
    (q,) = p.parse_questions(text, split="test")
    assert q.question == raw["question"]


def test_hindi_text_and_embedded_crlf_in_a_quote_are_exact():
    quote = HINDI + "\r\n" + HINDI
    unit = {"unit_id": "u1", "spans": [span_dict(quote=quote, quote_sha256=sha(quote), start=0, end=len(quote))]}
    (q,) = p.parse_questions(jsonl(question(question=HINDI, query_language="hi", units=[unit])), split="test")
    assert q.question == HINDI and q.units[0].spans[0].quote == quote


def test_empty_units_list_parses_to_an_empty_tuple():
    (q,) = p.parse_questions(jsonl(question(units=[])), split="test")
    assert q.units == ()


def test_question_values_are_not_checked_at_parse_time():
    raw = question(question_id="NOT AN ID", query_language="fr", author_type="robot", question="")
    (q,) = p.parse_questions(jsonl(raw), split="test")
    assert q.question_id == "NOT AN ID"


# ---------- E: parse_questions rejected ----------

def q_unit(**o):
    return {"unit_id": "u1", "spans": [span_dict()], **o}


QUESTION_REJECTED = [
    ROW(chr(0xFEFF) + jsonl(question()), [(p.PARSE_BOM, "line 1")], id="bom"),
    ROW("\n", [(p.PARSE_BLANK_LINE, "line 1")], id="only-a-newline"),
    ROW(jsonl(question()) + "\n", [(p.PARSE_BLANK_LINE, "line 2")], id="extra-blank-final-line"),
    ROW(jsonl(question(), "", question(question_id="q2")),
        [(p.PARSE_BLANK_LINE, "line 2")], id="blank-line-between-records"),
    ROW(jsonl(question(), "  \t ", question(question_id="q2")),
        [(p.PARSE_BLANK_LINE, "line 2")], id="whitespace-only-line"),
    ROW("{", [(p.PARSE_SYNTAX, "line 1 column 2")], id="truncated-record"),
    ROW(jsonl(question(), "{bad", question(question_id="q3"), "[1", final=False),
        [(p.PARSE_SYNTAX, "line 2 column 2"), (p.PARSE_SYNTAX, "line 4 column 3")], id="bad-lines-among-good"),
    ROW(jsonl("[1]"), [(p.PARSE_NOT_OBJECT, "line 1")], id="record-is-array"),
    ROW(jsonl(question()).replace('"question_id": "q1"', '"question_id": "q1", "question_id": "q2"'),
        [(p.PARSE_DUPLICATE_KEY, "line 1")], id="duplicate-key"),
    ROW(jsonl(question()).replace('"What is it?"', "NaN"), [(p.PARSE_NON_FINITE, "line 1")], id="nan"),
    ROW(jsonl(question()).replace('"What is it?"', "1e999"), [(p.PARSE_NON_FINITE, "line 1")], id="overflowing-float"),
    ROW(jsonl("[" * 100000), [(p.PARSE_TOO_DEEP, "line 1")], id="too-deep"),
    ROW(jsonl(question(split="dev")), [(p.PARSE_SPLIT_MISMATCH, "line 1.split")], id="split-mismatch"),
    ROW(jsonl(question(), question(question_id="q2", split="dev")),
        [(p.PARSE_SPLIT_MISMATCH, "line 2.split")], id="split-mismatch-second-line"),
    # closed schemas at every level
    ROW(jsonl(question(extra=1)), [(p.PARSE_UNKNOWN_FIELD, "line 1")], id="unknown-question-field"),
    ROW(jsonl(question(units=[q_unit(extra=1)])), [(p.PARSE_UNKNOWN_FIELD, "line 1.units[0]")], id="unknown-unit-field"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [span_dict(extra=1)]}])),
        [(p.PARSE_UNKNOWN_FIELD, "line 1.units[0].spans[0]")], id="unknown-span-field"),
    ROW(jsonl(question(semantic_review={"reviewer": "r", "date": "d", "result": "x", "extra": 1})),
        [(p.PARSE_UNKNOWN_FIELD, "line 1.semantic_review")], id="unknown-review-field"),
    ROW(jsonl(question(retrieval_assistance={"date": "d", "method": None, "run_ref": None, "description": "x", "extra": 1})),
        [(p.PARSE_UNKNOWN_FIELD, "line 1.retrieval_assistance")], id="unknown-assistance-field"),
    # required fields
    ROW(jsonl({}), [(p.PARSE_MISSING_FIELD, f"line 1.{n}") for n in (
        "question_id", "split", "answerable", "question", "query_language", "author_type", "hindi_reviewed")],
        id="empty-record"),
    ROW(jsonl({k: v for k, v in question().items() if k != "hindi_reviewed"}),
        [(p.PARSE_MISSING_FIELD, "line 1.hindi_reviewed")], id="missing-hindi-reviewed"),
    ROW(jsonl(question(units=[{"spans": []}])), [(p.PARSE_MISSING_FIELD, "line 1.units[0].unit_id")], id="unit-missing-id"),
    ROW(jsonl(question(units=[{"unit_id": "u"}])), [(p.PARSE_MISSING_FIELD, "line 1.units[0].spans")], id="unit-missing-spans"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [{k: v for k, v in span_dict().items() if k != "quote"}]}])),
        [(p.PARSE_MISSING_FIELD, "line 1.units[0].spans[0].quote")], id="span-missing-quote"),
    ROW(jsonl(question(retrieval_assistance={"date": "d", "description": "x"})),
        [(p.PARSE_MISSING_FIELD, "line 1.retrieval_assistance.method"),
         (p.PARSE_MISSING_FIELD, "line 1.retrieval_assistance.run_ref")], id="assistance-missing-keys"),
    # types
    ROW(jsonl(question(answerable="yes")), [(p.PARSE_FIELD_TYPE, "line 1.answerable")], id="string-bool"),
    ROW(jsonl(question(answerable=1)), [(p.PARSE_FIELD_TYPE, "line 1.answerable")], id="int-as-bool"),
    ROW(jsonl(question(hindi_reviewed=None)), [(p.PARSE_FIELD_TYPE, "line 1.hindi_reviewed")], id="null-bool"),
    ROW(jsonl(question(question=5)), [(p.PARSE_FIELD_TYPE, "line 1.question")], id="int-text"),
    ROW(jsonl(question(question=None)), [(p.PARSE_FIELD_TYPE, "line 1.question")], id="null-text"),
    ROW(jsonl(question(notes=5)), [(p.PARSE_FIELD_TYPE, "line 1.notes")], id="int-notes"),
    ROW(jsonl(question(close_distractor="no")), [(p.PARSE_FIELD_TYPE, "line 1.close_distractor")], id="string-distractor"),
    ROW(jsonl(question(units=None)), [(p.PARSE_FIELD_TYPE, "line 1.units")], id="null-units"),
    ROW(jsonl(question(units={"a": 1})), [(p.PARSE_FIELD_TYPE, "line 1.units")], id="units-object"),
    ROW(jsonl(question(units=["x"])), [(p.PARSE_NOT_OBJECT, "line 1.units[0]")], id="unit-string"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": "x"}])), [(p.PARSE_FIELD_TYPE, "line 1.units[0].spans")], id="spans-string"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [5]}])), [(p.PARSE_NOT_OBJECT, "line 1.units[0].spans[0]")], id="span-int"),
    ROW(jsonl(question(units=[{"unit_id": 1, "spans": []}])), [(p.PARSE_FIELD_TYPE, "line 1.units[0].unit_id")], id="int-unit-id"),
    ROW(jsonl(question(units=[q_unit(spans=[span_dict(start=True)])])),
        [(p.PARSE_FIELD_TYPE, "line 1.units[0].spans[0].start")], id="span-start-bool"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [span_dict(start=1.0)]}])),
        [(p.PARSE_FIELD_TYPE, "line 1.units[0].spans[0].start")], id="span-start-float"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [span_dict(end="8")]}])),
        [(p.PARSE_FIELD_TYPE, "line 1.units[0].spans[0].end")], id="span-end-string"),
    ROW(jsonl(question(units=[{"unit_id": "u", "spans": [span_dict(quote=None)]}])),
        [(p.PARSE_FIELD_TYPE, "line 1.units[0].spans[0].quote")], id="span-null-quote"),
    ROW(jsonl(question(semantic_review="x")), [(p.PARSE_NOT_OBJECT, "line 1.semantic_review")], id="review-string"),
    ROW(jsonl(question(retrieval_assistance=[])), [(p.PARSE_NOT_OBJECT, "line 1.retrieval_assistance")], id="assistance-list"),
    ROW(jsonl(question(retrieval_assistance={"date": "d", "method": 5, "run_ref": None, "description": "x"})),
        [(p.PARSE_FIELD_TYPE, "line 1.retrieval_assistance.method")], id="assistance-int-method"),
    # every invalid record is reported, none skipped
    ROW(jsonl(question(answerable="x"), question(question_id="q2"), question(question_id="q3", units=None)),
        [(p.PARSE_FIELD_TYPE, "line 1.answerable"), (p.PARSE_FIELD_TYPE, "line 3.units")], id="two-bad-records"),
]


@pytest.mark.parametrize("text, expected", QUESTION_REJECTED)
def test_question_rejections(text, expected):
    assert found(p.parse_questions, text, split="test") == expected


def test_a_bad_record_never_yields_a_partial_result():
    text = jsonl(question(), question(question_id="q2", answerable="x"))
    with pytest.raises(s.EvaluationDataError):
        p.parse_questions(text, split="test")


@pytest.mark.parametrize("bad", ["train", "", None, "Dev", ["dev"]])
def test_split_argument_must_be_dev_or_test(bad):
    with pytest.raises(ValueError, match="split"):
        p.parse_questions("", split=bad)


def test_split_is_keyword_only_and_text_must_be_str():
    with pytest.raises(TypeError):
        p.parse_questions("", "test")
    for bad in (b"", None, 5):
        with pytest.raises(TypeError):
            p.parse_questions(bad, split="test")


# ---------- F: hashing ----------

def make_manifest(**overrides):
    return p.parse_manifest(dumps(manifest(**overrides)))


def test_manifest_hash_matches_a_hand_built_canonical_document():
    m = p.parse_manifest(dumps(manifest()))
    expected = (
        '{"corpus_version":"c1","documents":[{"corpus_key":"doc-a","extracted_text_length":null,'
        '"extracted_text_sha256":null,"extraction":null,"format":"txt","language":"en","license":"MIT",'
        '"license_url":"https://example.invalid/license","publisher":"Publisher","redistributable":true,'
        '"source_file_sha256":"' + SHA_A + '","source_url":"https://example.invalid/a","title":"Title"}],'
        '"schema_version":"eval-manifest-v1","stage":"draft"}'
    )
    assert p.manifest_hash(m) == sha(expected)


def test_manifest_hash_includes_every_extraction_field_with_explicit_nulls():
    raw = manifest([doc(extracted_text_sha256=SHA_B, extracted_text_length=7,
                        extraction={"processing_ref": REF, "python": "3.12"})], stage="ingested")
    expected_extraction = (
        '{"lxml":null,"processing_ref":"' + REF + '","pypdf":null,"python":"3.12","python_docx":null}'
    )
    m = p.parse_manifest(dumps(raw))
    top = p._record(m)
    assert p.canonical_json(top["documents"][0]["extraction"]) == expected_extraction
    assert p.manifest_hash(m) == sha(p.canonical_json(top))


def test_manifest_hash_ignores_key_order_document_order_and_line_endings():
    docs = [doc(corpus_key="doc-b"), doc(corpus_key="doc-a", title="Other")]
    base = p.manifest_hash(p.parse_manifest(dumps(manifest(docs))))
    reordered = dict(reversed(list(manifest(docs).items())))
    reordered["documents"] = [dict(reversed(list(d.items()))) for d in reversed(docs)]
    assert p.manifest_hash(p.parse_manifest(dumps(reordered))) == base
    pretty = json.dumps(manifest(docs), indent=2)
    assert p.manifest_hash(p.parse_manifest(pretty.replace("\n", CRLF))) == base


def test_manifest_hash_orders_documents_by_code_point():
    # Upper case sorts before lower case by code point (hashing does not validate keys).
    a, b = doc(corpus_key="a"), doc(corpus_key="B")
    one = p.manifest_hash(p.parse_manifest(dumps(manifest([a, b]))))
    two = p.manifest_hash(p.parse_manifest(dumps(manifest([b, a]))))
    assert one == two
    top = p._record(p.parse_manifest(dumps(manifest([a, b]))))
    top["documents"] = [top["documents"][1], top["documents"][0]]  # "B" first
    assert one == sha(p.canonical_json(top))


def test_duplicate_keys_still_hash_deterministically():
    d1, d2 = doc(title="one"), doc(title="two")
    assert (p.manifest_hash(p.parse_manifest(dumps(manifest([d1, d2]))))
            == p.manifest_hash(p.parse_manifest(dumps(manifest([d2, d1])))))


def test_draft_and_ingested_manifests_hash_differently():
    draft = p.parse_manifest(dumps(manifest()))
    ingested_raw = manifest([doc(extracted_text_sha256=SHA_B, extracted_text_length=5,
                                 extraction={"processing_ref": REF, "python": "3"})], stage="ingested")
    ingested = p.parse_manifest(dumps(ingested_raw))
    assert p.manifest_hash(draft) != p.manifest_hash(ingested)
    # the stage alone is enough to change the hash
    assert p.manifest_hash(make_manifest(stage="draft")) != p.manifest_hash(make_manifest(stage="ingested"))


def test_null_versus_value_changes_the_hash():
    one = p.manifest_hash(p.parse_manifest(dumps(manifest([doc(extracted_text_length=None, extracted_text_sha256=None)]))))
    two = p.manifest_hash(p.parse_manifest(dumps(manifest([doc(extracted_text_length=1, extracted_text_sha256=SHA_B)]))))
    assert one != two


def test_manifest_hash_is_a_lowercase_sha256_and_stable():
    m = p.parse_manifest(dumps(manifest()))
    h = p.manifest_hash(m)
    assert s.SHA256_HEX_PATTERN.fullmatch(h) and h == p.manifest_hash(m)


def test_manifest_hash_does_not_validate():
    m = p.parse_manifest(dumps(manifest([doc(corpus_key="BAD KEY", format="rtf")], stage="nope")))
    assert s.SHA256_HEX_PATTERN.fullmatch(p.manifest_hash(m))


def test_manifest_hash_rejects_unserialisable_content_without_leaking_it():
    good = p.parse_manifest(dumps(manifest()))
    bad_docs = s.Manifest(good.schema_version, good.corpus_version, good.stage, (SECRET,))
    for bad in (
        bad_docs,
        s.Manifest("v", "c", "draft", "not a tuple"),
        s.Manifest("v", "c", "draft", (dataclasses.replace(good.documents[0], title=float("nan")),)),
    ):
        with pytest.raises(s.EvaluationDataError) as caught:
            p.manifest_hash(bad)
        assert [i.code for i in caught.value.issues] == [p.HASH_NOT_SERIALISABLE]
        assert SECRET not in str(caught.value) and SECRET not in caught.value.issues[0].message


def test_manifest_hash_requires_a_manifest():
    with pytest.raises(TypeError):
        p.manifest_hash("nope")


def q_obj(**o):
    return p.parse_questions(jsonl(question(**o)), split=o.get("split", "test"))[0]


def test_dataset_hash_of_nothing_is_the_hash_of_the_empty_byte_string():
    assert p.dataset_hash(()) == hashlib.sha256(b"").hexdigest()
    assert p.dataset_hash([]) == hashlib.sha256(b"").hexdigest()


def test_dataset_hash_matches_a_hand_built_canonical_document():
    q = q_obj()
    expected = (
        '{"answerable":true,"author_type":"human","close_distractor":null,"difficulty":null,'
        '"expected_answer":null,"hindi_reviewed":false,"hindi_reviewer":null,"notes":null,'
        '"query_language":"en","question":"What is it?","question_id":"q1","retrieval_assistance":null,'
        '"semantic_review":null,"source_language":null,"split":"test","unanswerable_type":null,"units":[]}'
    )
    assert p.dataset_hash([q]) == sha(expected)


def test_dataset_hash_joins_with_a_newline_and_orders_by_question_id():
    a, b = q_obj(question_id="a"), q_obj(question_id="b")
    ca, cb = (p.canonical_json(p._record(x)) for x in (a, b))
    assert p.dataset_hash([a, b]) == sha(ca + "\n" + cb)
    assert p.dataset_hash([b, a]) == p.dataset_hash([a, b])
    assert p.dataset_hash([a]) == sha(ca)  # no trailing newline


def test_dataset_hash_is_stable_across_key_order_and_line_endings():
    raw1 = question(question_id="a", expected_answer="x", units=[q_unit()])
    raw2 = question(question_id="b")
    text_lf = jsonl(raw1, raw2)
    shuffled = jsonl(dict(reversed(list(raw2.items()))), dict(reversed(list(raw1.items()))), eol=CRLF)
    assert (p.dataset_hash(p.parse_questions(text_lf, split="test"))
            == p.dataset_hash(p.parse_questions(shuffled, split="test")))


def test_dataset_hash_absent_and_explicit_null_are_the_same_record():
    one = p.parse_questions(jsonl(question()), split="test")
    two = p.parse_questions(jsonl(question(notes=None, units=[])), split="test")
    assert p.dataset_hash(one) == p.dataset_hash(two)


def test_split_and_content_are_part_of_the_hash():
    assert p.dataset_hash([q_obj(split="test")]) != p.dataset_hash([q_obj(split="dev")])
    assert p.dataset_hash([q_obj(question="a")]) != p.dataset_hash([q_obj(question="b")])
    quote = "hello"
    one = q_obj(units=[{"unit_id": "u", "spans": [span_dict()]}])
    two = q_obj(units=[{"unit_id": "u", "spans": [span_dict(quote="jello")]}])
    assert quote and p.dataset_hash([one]) != p.dataset_hash([two])


def test_dataset_hash_uses_ascii_escapes_for_non_ascii_text():
    q = q_obj(question=HINDI)
    assert p.dataset_hash([q]) == sha(p.canonical_json(p._record(q)))
    assert HINDI not in p.canonical_json(p._record(q))


def test_duplicate_question_ids_still_hash_deterministically():
    a, b = q_obj(question="one"), q_obj(question="two")
    assert p.dataset_hash([a, b]) == p.dataset_hash([b, a])


def test_dataset_hash_rejects_bad_input_without_leaking_it():
    good = q_obj()
    bad = s.Question(question_id=None, split="test", answerable=True, question=SECRET, query_language="en",
                     author_type="human", hindi_reviewed=False)
    with pytest.raises(s.EvaluationDataError) as caught:
        p.dataset_hash([good, bad])
    assert [i.code for i in caught.value.issues] == [p.HASH_NOT_SERIALISABLE]
    assert SECRET not in str(caught.value)
    with pytest.raises(TypeError):
        p.dataset_hash("nope")
    with pytest.raises(s.EvaluationDataError):
        p.dataset_hash([5])


# ---------- G: information leakage ----------

def every_text(error):
    for issue in error.issues:
        yield issue.code
        yield issue.location
        yield issue.message
    yield str(error)
    yield repr(error)


LEAK_MANIFEST = [
    '{"%s": 1}' % SECRET,
    '{"%s": 1, "%s": 2}' % (SECRET, SECRET),
    '{"schema_version": "%s", "schema_version": "x"}' % SECRET,
    '{"%s": NaN}' % SECRET,
    '{"%s": ' % SECRET,
    '{"%s" 5}' % SECRET,
    '[ "%s" ]' % SECRET,
    dumps(manifest(**{SECRET: 1})),
    dumps({**manifest(), "stage": 5, "corpus_version": {SECRET: SECRET}}),
    dumps(manifest([doc(**{SECRET: SECRET})])),
    dumps(manifest([doc(title=[SECRET])])),
    dumps(manifest([doc(extraction={"processing_ref": SECRET, "python": SECRET, SECRET: 1})])),
    dumps(manifest([doc(format="txt", extraction={"processing_ref": REF, "python": "3", "pypdf": SECRET})])),
    chr(0xFEFF) + SECRET,
]


@pytest.mark.parametrize("text", LEAK_MANIFEST)
def test_manifest_issues_never_contain_input_content(text):
    with pytest.raises(s.EvaluationDataError) as caught:
        p.parse_manifest(text)
    assert caught.value.issues
    for piece in every_text(caught.value):
        assert SECRET not in piece


LEAK_JSONL = [
    jsonl('{"%s": 1}' % SECRET),
    jsonl('{"question_id": "a", "%s": 1, "%s": 2}' % (SECRET, SECRET)),
    jsonl('{"%s" ' % SECRET, final=False),
    jsonl('{"question": NaN, "%s": 1}' % SECRET),
    jsonl(question(**{SECRET: SECRET})),
    jsonl(question(question=[SECRET], split=SECRET)),
    jsonl(question(units=[{"unit_id": SECRET, "spans": [span_dict(quote=[SECRET], **{SECRET: 1})]}])),
    jsonl(question(semantic_review={SECRET: SECRET}, retrieval_assistance={"date": SECRET})),
    jsonl(question(split=SECRET)),
]


@pytest.mark.parametrize("text", LEAK_JSONL)
def test_question_issues_never_contain_input_content(text):
    with pytest.raises(s.EvaluationDataError) as caught:
        p.parse_questions(text, split="test")
    assert caught.value.issues
    for piece in every_text(caught.value):
        assert SECRET not in piece


def test_error_message_is_still_count_only():
    with pytest.raises(s.EvaluationDataError) as caught:
        p.parse_manifest(dumps(manifest(a=1, b=2)))
    assert str(caught.value) == "2 evaluation data issue(s)"


def test_parser_issues_use_fixed_messages_per_code():
    messages = {}
    for text in LEAK_MANIFEST + [dumps(manifest(extra=1)), dumps(manifest([doc(extra=1)]))]:
        with pytest.raises(s.EvaluationDataError) as caught:
            p.parse_manifest(text)
        for issue in caught.value.issues:
            messages.setdefault(issue.code, set()).add(issue.message)
    assert all(len(v) == 1 for v in messages.values())


# ---------- H: import isolation ----------

def test_importing_parsing_loads_no_model_database_or_retrieval_code():
    code = (
        "import sys; import app.rag.evaluation.parsing; "
        "names = ['torch', 'transformers', 'sqlalchemy', 'fastapi', 'numpy', 'pypdf', "
        "'app.rag.embedder', 'app.rag.retrieval', 'app.rag.indexing', 'app.core.database', 'app.main']; "
        "print(sorted(n for n in names if n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
