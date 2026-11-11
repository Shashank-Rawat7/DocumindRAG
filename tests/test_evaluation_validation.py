"""E1.2b validation: manifest value rules and require_valid.

A  valid manifests
B  top-level rules
C  document rules
D  extraction metadata by format
E  draft and ingested stage rules
F  cross-document uniqueness and issue order
G  type-defensive behaviour
H  require_valid
I  determinism and information leakage
J  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import parsing as p
from app.rag.evaluation import schema as s
from app.rag.evaluation import validation as v

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REF = "d" * 40
SECRET = "SECRET-MANIFEST-TEXT"
HINDI = chr(0x0939) + chr(0x093F) + chr(0x0928) + chr(0x094D) + chr(0x0926) + chr(0x0940)
IDEOGRAPHIC_SPACE = chr(0x3000)
D = "manifest.documents[0]"


def extraction(fmt="txt", **overrides):
    values = {"processing_ref": REF, "python": "3.12.4"}
    if fmt == "pdf":
        values["pypdf"] = "5.0.0"
    if fmt == "docx":
        values["python_docx"] = "1.1.0"
        values["lxml"] = "5.0.0"
    values.update(overrides)
    return s.Extraction(**values)


def document(**overrides):
    """A valid DRAFT document: null hash, length and extraction."""
    values = {
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
    values.update(overrides)
    return s.ManifestDocument(**values)


def ingested_document(fmt="txt", **overrides):
    values = {"format": fmt, "extracted_text_sha256": SHA_B, "extracted_text_length": 100, "extraction": extraction(fmt)}
    values.update(overrides)
    return document(**values)


def raw_manifest(**overrides):
    """A valid draft manifest with ``overrides`` applied exactly as given (no conversion)."""
    values = {"schema_version": "eval-manifest-v1", "corpus_version": "c1", "stage": "draft", "documents": (document(),)}
    values.update(overrides)
    return s.Manifest(**values)


def make(documents=None, **overrides):
    """Like raw_manifest, but ``documents`` may be any iterable of documents (it becomes a tuple)."""
    if documents is not None:
        overrides["documents"] = tuple(documents)
    return raw_manifest(**overrides)


def problems(manifest):
    return [(i.code, i.location) for i in v.validate_manifest(manifest)]


def replaced(doc, **overrides):
    return dataclasses.replace(doc, **overrides)


# ---------- A: valid manifests ----------

def test_minimal_draft_manifest_is_valid():
    assert v.validate_manifest(make()) == []


@pytest.mark.parametrize("fmt", ["txt", "md", "pdf", "docx"])
def test_ingested_document_of_every_format_is_valid(fmt):
    assert problems(make([ingested_document(fmt)], stage="ingested")) == []


def test_draft_extraction_is_independent_of_hash_and_length():
    assert problems(make([document(extraction=extraction())])) == []                      # extraction only
    assert problems(make([document(extracted_text_sha256=SHA_B, extracted_text_length=5)])) == []  # hash + length only


def test_several_documents_and_non_ascii_metadata_are_valid():
    docs = [document(corpus_key="a"), document(corpus_key="b", title=HINDI, language="hi"),
            ingested_document("pdf", corpus_key="c", extracted_text_sha256=SHA_C)]
    assert problems(make(docs)) == []


def test_a_parsed_manifest_validates_cleanly():
    raw = {
        "schema_version": "eval-manifest-v1", "corpus_version": "c1", "stage": "ingested",
        "documents": [{
            "corpus_key": "doc-a", "title": "T", "publisher": "P", "source_url": "u", "license": "l",
            "license_url": "lu", "format": "docx", "language": "en", "source_file_sha256": SHA_A,
            "extracted_text_sha256": SHA_B, "extracted_text_length": 9,
            "extraction": {"processing_ref": REF, "python": "3", "python_docx": "1", "lxml": "2"},
            "redistributable": False,
        }],
    }
    assert v.validate_manifest(p.parse_manifest(json.dumps(raw))) == []


def test_validate_manifest_requires_a_manifest():
    for bad in (None, "x", {}, make().documents):
        with pytest.raises(TypeError):
            v.validate_manifest(bad)


# ---------- B: top-level rules ----------

TOP_ROWS = [
    pytest.param({"schema_version": "eval-manifest-v2"}, [(v.MANIFEST_SCHEMA_VERSION_MISMATCH, "manifest.schema_version")], id="version-mismatch"),
    pytest.param({"schema_version": ""}, [(v.MANIFEST_SCHEMA_VERSION_MISMATCH, "manifest.schema_version")], id="version-empty"),
    pytest.param({"schema_version": 1}, [(v.MANIFEST_FIELD_TYPE, "manifest.schema_version")], id="version-int"),
    pytest.param({"schema_version": None}, [(v.MANIFEST_FIELD_TYPE, "manifest.schema_version")], id="version-null"),
    pytest.param({"corpus_version": ""}, [(v.MANIFEST_BLANK, "manifest.corpus_version")], id="corpus-version-empty"),
    pytest.param({"corpus_version": " \t\n"}, [(v.MANIFEST_BLANK, "manifest.corpus_version")], id="corpus-version-whitespace"),
    pytest.param({"corpus_version": IDEOGRAPHIC_SPACE}, [(v.MANIFEST_BLANK, "manifest.corpus_version")], id="corpus-version-unicode-space"),
    pytest.param({"corpus_version": 5}, [(v.MANIFEST_FIELD_TYPE, "manifest.corpus_version")], id="corpus-version-int"),
    pytest.param({"stage": "final"}, [(v.MANIFEST_STAGE, "manifest.stage")], id="stage-unknown"),
    pytest.param({"stage": "Draft"}, [(v.MANIFEST_STAGE, "manifest.stage")], id="stage-case"),
    pytest.param({"stage": None}, [(v.MANIFEST_FIELD_TYPE, "manifest.stage")], id="stage-null"),
    pytest.param({"stage": ["draft"]}, [(v.MANIFEST_FIELD_TYPE, "manifest.stage")], id="stage-list"),
    pytest.param({"documents": ()}, [(v.MANIFEST_NO_DOCUMENTS, "manifest.documents")], id="no-documents"),
    pytest.param({"documents": [document()]}, [(v.MANIFEST_FIELD_TYPE, "manifest.documents")], id="documents-list"),
    pytest.param({"documents": None}, [(v.MANIFEST_FIELD_TYPE, "manifest.documents")], id="documents-null"),
]


@pytest.mark.parametrize("overrides, expected", TOP_ROWS)
def test_top_level_rules(overrides, expected):
    assert problems(raw_manifest(**overrides)) == expected


def test_every_top_level_problem_is_reported_in_order():
    m = raw_manifest(schema_version="x", corpus_version="", stage="y", documents=())
    assert problems(m) == [
        (v.MANIFEST_SCHEMA_VERSION_MISMATCH, "manifest.schema_version"),
        (v.MANIFEST_BLANK, "manifest.corpus_version"),
        (v.MANIFEST_STAGE, "manifest.stage"),
        (v.MANIFEST_NO_DOCUMENTS, "manifest.documents"),
    ]


# ---------- C: document rules ----------

DOC_ROWS = [
    # corpus key
    pytest.param({"corpus_key": ""}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-empty"),
    pytest.param({"corpus_key": "Doc"}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-uppercase"),
    pytest.param({"corpus_key": "-a"}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-leading-hyphen"),
    pytest.param({"corpus_key": "a" * 65}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-65"),
    pytest.param({"corpus_key": "a b"}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-space"),
    pytest.param({"corpus_key": "a" + chr(10)}, [(v.DOC_CORPUS_KEY, f"{D}.corpus_key")], id="key-trailing-newline"),
    pytest.param({"corpus_key": None}, [(v.DOC_FIELD_TYPE, f"{D}.corpus_key")], id="key-null"),
    pytest.param({"corpus_key": 5}, [(v.DOC_FIELD_TYPE, f"{D}.corpus_key")], id="key-int"),
    # blank / type metadata
    *[pytest.param({name: value}, [(code, f"{D}.{name}")], id=f"{name}-{label}")
      for name in ("title", "publisher", "source_url", "license", "license_url")
      for label, value, code in (("empty", "", v.DOC_BLANK), ("spaces", "   ", v.DOC_BLANK),
                                 ("unicode-space", IDEOGRAPHIC_SPACE, v.DOC_BLANK),
                                 ("null", None, v.DOC_FIELD_TYPE), ("int", 5, v.DOC_FIELD_TYPE))],
    # enums
    pytest.param({"format": "rtf"}, [(v.DOC_FORMAT, f"{D}.format")], id="format-unknown"),
    pytest.param({"format": "PDF"}, [(v.DOC_FORMAT, f"{D}.format")], id="format-case"),
    pytest.param({"format": None}, [(v.DOC_FIELD_TYPE, f"{D}.format")], id="format-null"),
    pytest.param({"language": "fr"}, [(v.DOC_LANGUAGE, f"{D}.language")], id="language-unknown"),
    pytest.param({"language": "EN"}, [(v.DOC_LANGUAGE, f"{D}.language")], id="language-case"),
    pytest.param({"language": None}, [(v.DOC_FIELD_TYPE, f"{D}.language")], id="language-null"),
    # hashes
    pytest.param({"source_file_sha256": "A" * 64}, [(v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256")], id="source-hash-uppercase"),
    pytest.param({"source_file_sha256": "a" * 63}, [(v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256")], id="source-hash-63"),
    pytest.param({"source_file_sha256": "a" * 65}, [(v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256")], id="source-hash-65"),
    pytest.param({"source_file_sha256": "g" * 64}, [(v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256")], id="source-hash-non-hex"),
    pytest.param({"source_file_sha256": "a" * 64 + chr(10)}, [(v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256")], id="source-hash-newline"),
    pytest.param({"source_file_sha256": None}, [(v.DOC_FIELD_TYPE, f"{D}.source_file_sha256")], id="source-hash-null"),
    pytest.param({"extracted_text_sha256": "B" * 64, "extracted_text_length": 5},
                 [(v.DOC_SHA256_FORMAT, f"{D}.extracted_text_sha256")], id="text-hash-uppercase"),
    pytest.param({"extracted_text_sha256": SHA_B + chr(10), "extracted_text_length": 5},
                 [(v.DOC_SHA256_FORMAT, f"{D}.extracted_text_sha256")], id="text-hash-newline"),
    pytest.param({"extracted_text_sha256": 5, "extracted_text_length": 5},
                 [(v.DOC_FIELD_TYPE, f"{D}.extracted_text_sha256")], id="text-hash-int"),
    # length
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": 0}, [(v.DOC_LENGTH, f"{D}.extracted_text_length")], id="length-zero"),
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": -3}, [(v.DOC_LENGTH, f"{D}.extracted_text_length")], id="length-negative"),
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": True}, [(v.DOC_FIELD_TYPE, f"{D}.extracted_text_length")], id="length-bool"),
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": 5.0}, [(v.DOC_FIELD_TYPE, f"{D}.extracted_text_length")], id="length-float"),
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": "5"}, [(v.DOC_FIELD_TYPE, f"{D}.extracted_text_length")], id="length-string"),
    pytest.param({"extracted_text_sha256": SHA_B, "extracted_text_length": 1}, [], id="length-one-is-valid"),
    # hash/length pairing
    pytest.param({"extracted_text_sha256": SHA_B}, [(v.DOC_HASH_LENGTH_PAIR, D)], id="hash-without-length"),
    pytest.param({"extracted_text_length": 5}, [(v.DOC_HASH_LENGTH_PAIR, D)], id="length-without-hash"),
    # flag
    pytest.param({"redistributable": "yes"}, [(v.DOC_FIELD_TYPE, f"{D}.redistributable")], id="redistributable-string"),
    pytest.param({"redistributable": 1}, [(v.DOC_FIELD_TYPE, f"{D}.redistributable")], id="redistributable-int"),
    pytest.param({"redistributable": None}, [(v.DOC_FIELD_TYPE, f"{D}.redistributable")], id="redistributable-null"),
    pytest.param({"redistributable": False}, [], id="redistributable-false-is-valid"),
]


@pytest.mark.parametrize("overrides, expected", DOC_ROWS)
def test_document_rules(overrides, expected):
    assert problems(make([document(**overrides)])) == expected


def test_a_document_with_several_problems_reports_them_in_field_order():
    doc = document(corpus_key="Bad", title="", format="rtf", language="fr", source_file_sha256="x",
                   redistributable=3)
    assert problems(make([doc])) == [
        (v.DOC_CORPUS_KEY, f"{D}.corpus_key"),
        (v.DOC_BLANK, f"{D}.title"),
        (v.DOC_FORMAT, f"{D}.format"),
        (v.DOC_LANGUAGE, f"{D}.language"),
        (v.DOC_SHA256_FORMAT, f"{D}.source_file_sha256"),
        (v.DOC_FIELD_TYPE, f"{D}.redistributable"),
    ]


# ---------- D: extraction metadata by format ----------

X = f"{D}.extraction"


def with_extraction(fmt, ex):
    return make([document(format=fmt, extraction=ex)])


EXTRACTION_ROWS = [
    # processing_ref
    pytest.param("txt", {"processing_ref": "D" * 40}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-uppercase"),
    pytest.param("txt", {"processing_ref": "d" * 39}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-39"),
    pytest.param("txt", {"processing_ref": "d" * 41}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-41"),
    pytest.param("txt", {"processing_ref": "g" * 40}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-non-hex"),
    pytest.param("txt", {"processing_ref": "d" * 40 + chr(10)}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-newline"),
    pytest.param("txt", {"processing_ref": ""}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-empty"),
    pytest.param("txt", {"processing_ref": None}, [(v.DOC_FIELD_TYPE, f"{X}.processing_ref")], id="ref-null"),
    pytest.param("txt", {"processing_ref": "d" * 64}, [(v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref")], id="ref-is-not-a-sha256"),
    # python
    pytest.param("txt", {"python": ""}, [(v.EXTRACTION_BLANK, f"{X}.python")], id="python-empty"),
    pytest.param("txt", {"python": "  "}, [(v.EXTRACTION_BLANK, f"{X}.python")], id="python-spaces"),
    pytest.param("txt", {"python": None}, [(v.DOC_FIELD_TYPE, f"{X}.python")], id="python-null"),
    pytest.param("txt", {"python": 3}, [(v.DOC_FIELD_TYPE, f"{X}.python")], id="python-int"),
    # txt / md: no library applies
    pytest.param("txt", {"pypdf": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.pypdf")], id="txt-with-pypdf"),
    pytest.param("txt", {"python_docx": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.python_docx")], id="txt-with-python-docx"),
    pytest.param("md", {"lxml": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.lxml")], id="md-with-lxml"),
    pytest.param("md", {"pypdf": ""}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.pypdf")], id="md-with-empty-pypdf-is-still-inapplicable"),
    # pdf: pypdf required, docx libraries inapplicable
    pytest.param("pdf", {"pypdf": None}, [(v.EXTRACTION_FIELD_MISSING, f"{X}.pypdf")], id="pdf-missing-pypdf"),
    pytest.param("pdf", {"pypdf": ""}, [(v.EXTRACTION_BLANK, f"{X}.pypdf")], id="pdf-blank-pypdf"),
    pytest.param("pdf", {"pypdf": 5}, [(v.DOC_FIELD_TYPE, f"{X}.pypdf")], id="pdf-int-pypdf"),
    pytest.param("pdf", {"lxml": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.lxml")], id="pdf-with-lxml"),
    pytest.param("pdf", {"python_docx": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.python_docx")], id="pdf-with-python-docx"),
    # docx: python_docx and lxml required, pypdf inapplicable
    pytest.param("docx", {"python_docx": None}, [(v.EXTRACTION_FIELD_MISSING, f"{X}.python_docx")], id="docx-missing-python-docx"),
    pytest.param("docx", {"lxml": None}, [(v.EXTRACTION_FIELD_MISSING, f"{X}.lxml")], id="docx-missing-lxml"),
    pytest.param("docx", {"lxml": "  "}, [(v.EXTRACTION_BLANK, f"{X}.lxml")], id="docx-blank-lxml"),
    pytest.param("docx", {"pypdf": "1"}, [(v.EXTRACTION_FIELD_INAPPLICABLE, f"{X}.pypdf")], id="docx-with-pypdf"),
    pytest.param("docx", {"python_docx": None, "lxml": None},
                 [(v.EXTRACTION_FIELD_MISSING, f"{X}.python_docx"), (v.EXTRACTION_FIELD_MISSING, f"{X}.lxml")],
                 id="docx-missing-both-in-order"),
]


@pytest.mark.parametrize("fmt, overrides, expected", EXTRACTION_ROWS)
def test_extraction_rules_by_format(fmt, overrides, expected):
    ex = dataclasses.replace(extraction(fmt), **overrides)
    assert problems(with_extraction(fmt, ex)) == expected


def test_extraction_must_be_an_extraction_object():
    for bad in ("x", {"processing_ref": REF}, 5, [extraction()]):
        assert problems(with_extraction("txt", bad)) == [(v.DOC_FIELD_TYPE, X)]


def test_unknown_format_skips_the_library_checks_and_reports_only_the_format():
    ex = s.Extraction(processing_ref=REF, python="3", pypdf="1", lxml=None)
    assert problems(with_extraction("rtf", ex)) == [(v.DOC_FORMAT, f"{D}.format")]


def test_extraction_with_bad_ref_and_missing_library_reports_both_in_order():
    ex = s.Extraction(processing_ref="bad", python="", pypdf=None)
    assert problems(with_extraction("pdf", ex)) == [
        (v.EXTRACTION_PROCESSING_REF, f"{X}.processing_ref"),
        (v.EXTRACTION_BLANK, f"{X}.python"),
        (v.EXTRACTION_FIELD_MISSING, f"{X}.pypdf"),
    ]


# ---------- E: stage rules ----------

def test_ingested_requires_hash_length_and_extraction():
    m = make([document()], stage="ingested")
    assert problems(m) == [
        (v.DOC_STAGE_INCOMPLETE, f"{D}.extracted_text_sha256"),
        (v.DOC_STAGE_INCOMPLETE, f"{D}.extracted_text_length"),
        (v.DOC_STAGE_INCOMPLETE, f"{D}.extraction"),
    ]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"extraction": None}, [(v.DOC_STAGE_INCOMPLETE, f"{D}.extraction")]),
        ({"extracted_text_sha256": None, "extracted_text_length": None},
         [(v.DOC_STAGE_INCOMPLETE, f"{D}.extracted_text_sha256"), (v.DOC_STAGE_INCOMPLETE, f"{D}.extracted_text_length")]),
        ({"extracted_text_sha256": None},
         [(v.DOC_HASH_LENGTH_PAIR, D), (v.DOC_STAGE_INCOMPLETE, f"{D}.extracted_text_sha256")]),
    ],
)
def test_ingested_documents_missing_one_part(overrides, expected):
    assert problems(make([ingested_document("txt", **overrides)], stage="ingested")) == expected


def test_one_incomplete_document_among_valid_ones_is_located():
    docs = [ingested_document("txt", corpus_key="a"), document(corpus_key="b"), ingested_document("pdf", corpus_key="c", extracted_text_sha256=SHA_C)]
    got = problems(make(docs, stage="ingested"))
    assert got == [
        (v.DOC_STAGE_INCOMPLETE, "manifest.documents[1].extracted_text_sha256"),
        (v.DOC_STAGE_INCOMPLETE, "manifest.documents[1].extracted_text_length"),
        (v.DOC_STAGE_INCOMPLETE, "manifest.documents[1].extraction"),
    ]


def test_draft_does_not_require_the_post_ingestion_fields():
    assert problems(make([document(), document(corpus_key="b")], stage="draft")) == []


def test_draft_extraction_is_complete_or_null_never_partial():
    partial = s.Extraction(processing_ref=REF, python="3", pypdf=None)
    assert problems(with_extraction("pdf", partial)) == [(v.EXTRACTION_FIELD_MISSING, f"{X}.pypdf")]


def test_an_unknown_stage_skips_the_stage_rules():
    assert problems(make([document()], stage="done")) == [(v.MANIFEST_STAGE, "manifest.stage")]


# ---------- F: uniqueness and order ----------

def test_duplicate_corpus_key_is_reported_at_each_later_document():
    docs = [document(corpus_key="a"), document(corpus_key="b"), document(corpus_key="a"), document(corpus_key="a")]
    assert problems(make(docs)) == [
        (v.DOC_DUPLICATE_KEY, "manifest.documents[2].corpus_key"),
        (v.DOC_DUPLICATE_KEY, "manifest.documents[3].corpus_key"),
    ]


def test_duplicate_non_null_text_hash_is_an_error():
    docs = [ingested_document(corpus_key="a"), ingested_document(corpus_key="b"), ingested_document(corpus_key="c", extracted_text_sha256=SHA_C)]
    assert problems(make(docs, stage="ingested")) == [(v.DOC_DUPLICATE_TEXT_HASH, "manifest.documents[1].extracted_text_sha256")]


def test_null_text_hashes_are_not_duplicates():
    assert problems(make([document(corpus_key="a"), document(corpus_key="b"), document(corpus_key="c")])) == []


def test_invalid_keys_and_hashes_are_not_also_reported_as_duplicates():
    docs = [document(corpus_key="A"), document(corpus_key="A")]
    assert problems(make(docs)) == [(v.DOC_CORPUS_KEY, "manifest.documents[0].corpus_key"),
                                    (v.DOC_CORPUS_KEY, "manifest.documents[1].corpus_key")]


def test_issue_order_is_top_level_then_documents_then_cross_document():
    docs = [document(corpus_key="a", title=""), document(corpus_key="a", extracted_text_sha256=SHA_B, extracted_text_length=3),
            document(corpus_key="b", extracted_text_sha256=SHA_B, extracted_text_length=3, language="fr")]
    got = problems(make(docs, corpus_version=""))
    assert got == [
        (v.MANIFEST_BLANK, "manifest.corpus_version"),
        (v.DOC_BLANK, "manifest.documents[0].title"),
        (v.DOC_LANGUAGE, "manifest.documents[2].language"),
        (v.DOC_DUPLICATE_KEY, "manifest.documents[1].corpus_key"),
        (v.DOC_DUPLICATE_TEXT_HASH, "manifest.documents[2].extracted_text_sha256"),
    ]


# ---------- G: type-defensive behaviour ----------

def test_a_non_document_element_is_reported_and_does_not_crash():
    m = make([document(), "not a document", None, 5])
    assert problems(m) == [(v.DOC_FIELD_TYPE, "manifest.documents[1]"), (v.DOC_FIELD_TYPE, "manifest.documents[2]"),
                           (v.DOC_FIELD_TYPE, "manifest.documents[3]")]


def test_garbage_in_every_field_gives_issues_and_no_crash():
    garbage = s.ManifestDocument(*([[1], {2: 3}, 4.5, (6,), None, b"x", object(), 7, {}, [], (), 1j, "x"]))
    m = s.Manifest(schema_version=[1], corpus_version={}, stage=(1,), documents=(garbage,))
    got = v.validate_manifest(m)
    assert got and all(i.severity == s.SEVERITY_ERROR for i in got)


def test_unhashable_values_do_not_break_the_duplicate_checks():
    docs = [document(corpus_key=["a"]), document(corpus_key=["a"], extracted_text_sha256=["b"], extracted_text_length=1)]
    got = problems(make(docs))
    assert (v.DOC_FIELD_TYPE, "manifest.documents[0].corpus_key") in got
    assert not any(code in (v.DOC_DUPLICATE_KEY, v.DOC_DUPLICATE_TEXT_HASH) for code, _ in got)


# ---------- H: require_valid ----------

def err(code="E", loc="l", msg="m"):
    return s.Issue(code, loc, msg)


def warn(code="W"):
    return s.Issue(code, "l", "m", s.SEVERITY_WARNING)


def test_require_valid_returns_none_for_no_issues_and_for_warnings_only():
    assert v.require_valid([]) is None
    assert v.require_valid([warn(), warn("W2")]) is None
    assert v.require_valid(()) is None


def test_require_valid_raises_when_any_error_is_present():
    issues = [warn("W1"), err("E1"), warn("W2"), err("E2")]
    with pytest.raises(s.EvaluationDataError) as caught:
        v.require_valid(issues)
    assert caught.value.issues == tuple(issues)  # warnings stay available, order preserved
    assert str(caught.value) == "4 evaluation data issue(s)"


def test_require_valid_does_not_modify_its_argument():
    issues = [err()]
    with pytest.raises(s.EvaluationDataError):
        v.require_valid(issues)
    assert issues == [err()]


def test_require_valid_error_message_is_count_only_even_with_secret_codes():
    issues = [s.Issue(SECRET, SECRET, SECRET)]
    with pytest.raises(s.EvaluationDataError) as caught:
        v.require_valid(issues)
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)
    assert caught.value.issues[0].code == SECRET


@pytest.mark.parametrize("bad", [["x"], [err(), "x"], [None], [{"code": "E"}]])
def test_require_valid_rejects_non_issue_entries(bad):
    with pytest.raises(TypeError, match="Issue"):
        v.require_valid(bad)


def test_require_valid_rejects_a_non_iterable():
    with pytest.raises(TypeError):
        v.require_valid(5)


def test_require_valid_accepts_validate_manifest_output():
    assert v.require_valid(v.validate_manifest(make())) is None
    with pytest.raises(s.EvaluationDataError):
        v.require_valid(v.validate_manifest(make(documents=())))


# ---------- I: determinism and leakage ----------

def test_results_are_deterministic_and_all_errors():
    m = make([document(title="", format="rtf"), document(corpus_key="Bad")], stage="x")
    first, second = v.validate_manifest(m), v.validate_manifest(m)
    assert first == second and first
    assert all(i.severity == s.SEVERITY_ERROR for i in first)


def test_each_code_has_one_fixed_message():
    messages = {}
    rows = [raw_manifest(**r.values[0]) for r in TOP_ROWS] + [make([document(**r.values[0])]) for r in DOC_ROWS]
    for m in rows:
        for issue in v.validate_manifest(m):
            messages.setdefault(issue.code, set()).add(issue.message)
    assert messages and all(len(texts) == 1 for texts in messages.values())


def test_no_issue_contains_manifest_content():
    bad = s.ManifestDocument(
        corpus_key=SECRET, title=SECRET, publisher=5, source_url=SECRET, license=SECRET, license_url=SECRET,
        format=SECRET, language=SECRET, source_file_sha256=SECRET, extracted_text_sha256=SECRET,
        extracted_text_length=SECRET,
        extraction=s.Extraction(processing_ref=SECRET, python=SECRET, pypdf=SECRET, python_docx=SECRET, lxml=SECRET),
        redistributable=SECRET,
    )
    m = s.Manifest(schema_version=SECRET, corpus_version=SECRET, stage=SECRET, documents=(bad, bad))
    issues = v.validate_manifest(m)
    assert len(issues) > 10
    for issue in issues:
        assert SECRET not in issue.code and SECRET not in issue.location and SECRET not in issue.message
    with pytest.raises(s.EvaluationDataError) as caught:
        v.require_valid(issues)
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)


# ---------- J: import isolation ----------

def test_importing_validation_loads_no_model_database_or_retrieval_code():
    code = (
        "import sys; import app.rag.evaluation.validation; "
        "names = ['torch', 'transformers', 'sqlalchemy', 'fastapi', 'numpy', 'pypdf', "
        "'app.rag.embedder', 'app.rag.retrieval', 'app.rag.indexing', 'app.core.database', 'app.main']; "
        "print(sorted(n for n in names if n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
