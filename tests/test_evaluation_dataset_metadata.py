"""Tests for dataset metadata parsing and validation (E1.2d, step 5).

Sections:
  A. DatasetMetadata
  B. parse_dataset_metadata: accepted input
  C. parse_dataset_metadata: rejected input
  D. parse_dataset_metadata: numbers, non-finite values, duplicate keys
  E. validate_dataset_metadata: field rules
  F. validate_dataset_metadata: manifest gate and comparisons
  G. whole-result properties and privacy
  H. import isolation
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import dataset_metadata as m
from app.rag.evaluation import parsing as p
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64
CANARY = "SECRET-CANARY-7731"
TEXT_A = "Alpha clause one. Beta clause two."


def doc(key="doc-a", text=TEXT_A):
    return s.ManifestDocument(
        corpus_key=key, title="T", publisher="P", source_url="https://example.invalid/a", license="MIT",
        license_url="https://example.invalid/l", format="txt", language="en", source_file_sha256=SHA,
        extracted_text_sha256=sha256_text(text), extracted_text_length=len(text),
        extraction=s.Extraction(processing_ref="d" * 40, python="3.12.4"), redistributable=True,
    )


MANIFEST = s.Manifest("eval-manifest-v1", "c1", "ingested", (doc(),))
DRAFT = dataclasses.replace(MANIFEST, stage="draft")
INVALID = dataclasses.replace(MANIFEST, schema_version="nope")
GOOD_HASH = p.manifest_hash(MANIFEST)
DRAFT_HASH = p.manifest_hash(DRAFT)


def meta(**overrides):
    values = dict(
        schema_version=s.DATASET_SCHEMA_VERSION, dataset_version="v1", corpus_version="c1",
        corpus_manifest_hash=GOOD_HASH, created="2026-11-11",
    )
    values.update(overrides)
    return m.DatasetMetadata(**values)


def good_json(**overrides):
    values = dict(
        schema_version=s.DATASET_SCHEMA_VERSION, dataset_version="v1", corpus_version="c1",
        corpus_manifest_hash=GOOD_HASH, created="2026-11-11",
    )
    values.update(overrides)
    return json.dumps(values)


def parse_issues(text):
    with pytest.raises(s.EvaluationDataError) as caught:
        m.parse_dataset_metadata(text)
    return [(i.code, i.location) for i in caught.value.issues]


def check(metadata=None, manifest=MANIFEST):
    return [(i.code, i.location) for i in m.validate_dataset_metadata(meta() if metadata is None else metadata, manifest)]


# ---------- A. DatasetMetadata ----------

def test_dataset_metadata_has_exactly_the_five_fields_in_order():
    assert [f.name for f in dataclasses.fields(m.DatasetMetadata)] == [
        "schema_version", "dataset_version", "corpus_version", "corpus_manifest_hash", "created"
    ]
    assert all(f.type in (str, "str") for f in dataclasses.fields(m.DatasetMetadata))


def test_dataset_metadata_is_frozen_and_slotted():
    value = meta()
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.created = "x"
    assert not hasattr(value, "__dict__")


# ---------- B. parse: accepted ----------

def test_a_valid_document_parses_to_the_exact_object():
    assert m.parse_dataset_metadata(good_json()) == meta()


def test_key_order_and_whitespace_do_not_matter():
    text = "\n  " + json.dumps(dict(reversed(list(json.loads(good_json()).items()))), indent=3) + "\n\n"
    assert m.parse_dataset_metadata(text) == meta()


def test_parsing_does_not_validate_values():
    parsed = m.parse_dataset_metadata(good_json(schema_version="whatever", created="not a date", dataset_version=""))
    assert parsed.schema_version == "whatever" and parsed.created == "not a date" and parsed.dataset_version == ""


def test_non_ascii_values_are_kept():
    value = chr(0x0939) + chr(0x093F)
    assert m.parse_dataset_metadata(good_json(dataset_version=value)).dataset_version == value


# ---------- C. parse: rejected ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("", [("PARSE_SYNTAX", "dataset_metadata line 1 column 1")], id="empty"),
        pytest.param("{", [("PARSE_SYNTAX", "dataset_metadata line 1 column 2")], id="truncated"),
        pytest.param("[1]", [("PARSE_NOT_OBJECT", "dataset_metadata")], id="array"),
        pytest.param('"x"', [("PARSE_NOT_OBJECT", "dataset_metadata")], id="string"),
        pytest.param("null", [("PARSE_NOT_OBJECT", "dataset_metadata")], id="null"),
        pytest.param("5", [("PARSE_NOT_OBJECT", "dataset_metadata")], id="integer-root"),
        pytest.param("5.5", [("PARSE_NOT_OBJECT", "dataset_metadata")], id="float-root"),
        pytest.param("{}", [("PARSE_MISSING_FIELD", f"dataset_metadata.{n}") for n in m.FIELDS], id="empty-object"),
    ],
)
def test_structural_rejections(text, expected):
    assert parse_issues(text) == expected


def test_a_bom_is_reported_and_the_rest_is_still_inspected():
    assert parse_issues("﻿" + good_json(created=1)) == [
        ("PARSE_BOM", "dataset_metadata"), ("PARSE_FIELD_TYPE", "dataset_metadata.created")
    ]
    assert parse_issues("﻿{") == [("PARSE_BOM", "dataset_metadata"), ("PARSE_SYNTAX", "dataset_metadata line 1 column 2")]


def test_unknown_fields_are_reported_without_naming_them():
    result = parse_issues(good_json(**{CANARY: "x", "other": "y"}))
    assert result == [("PARSE_UNKNOWN_FIELD", "dataset_metadata")] * 2


def test_missing_fields_are_reported_in_field_order():
    data = json.loads(good_json())
    del data["created"], data["dataset_version"]
    assert parse_issues(json.dumps(data)) == [
        ("PARSE_MISSING_FIELD", "dataset_metadata.dataset_version"), ("PARSE_MISSING_FIELD", "dataset_metadata.created")
    ]


@pytest.mark.parametrize(
    "value", [pytest.param(None, id="null"), pytest.param(True, id="bool"), pytest.param([], id="list"),
              pytest.param({}, id="object"), pytest.param(7, id="int"), pytest.param(1.5, id="float")]
)
def test_every_field_must_be_a_string(value):
    assert parse_issues(good_json(dataset_version=value)) == [("PARSE_FIELD_TYPE", "dataset_metadata.dataset_version")]


def test_all_independent_problems_are_collected_together():
    text = json.dumps({"schema_version": 1, "dataset_version": "v", "corpus_version": None, "extra": 1})
    assert parse_issues(text) == [
        ("PARSE_UNKNOWN_FIELD", "dataset_metadata"),
        ("PARSE_MISSING_FIELD", "dataset_metadata.corpus_manifest_hash"),
        ("PARSE_MISSING_FIELD", "dataset_metadata.created"),
        ("PARSE_FIELD_TYPE", "dataset_metadata.schema_version"),
        ("PARSE_FIELD_TYPE", "dataset_metadata.corpus_version"),
    ]


def test_deep_nesting_is_one_too_deep_issue():
    assert parse_issues("[" * 100000) == [("PARSE_TOO_DEEP", "dataset_metadata")]


def test_non_str_input_raises_type_error():
    for bad in (None, b"{}", 5):
        with pytest.raises(TypeError):
            m.parse_dataset_metadata(bad)


def test_a_failed_parse_never_returns_a_partial_object():
    with pytest.raises(s.EvaluationDataError):
        m.parse_dataset_metadata(good_json(created=3))


# ---------- D. parse: numbers, non-finite values, duplicate keys ----------

@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", "1E400"])
def test_non_finite_numbers_are_rejected_once_and_not_also_as_a_type_error(literal):
    text = good_json().replace('"2026-11-11"', literal)
    assert parse_issues(text) == [("PARSE_NON_FINITE", "dataset_metadata")]


def test_a_non_finite_root_is_reported_once():
    assert parse_issues("NaN") == [("PARSE_NON_FINITE", "dataset_metadata")]


def test_a_huge_integer_is_a_field_type_issue_not_a_digit_limit_failure():
    text = good_json().replace('"2026-11-11"', "9" * 100000)
    assert parse_issues(text) == [("PARSE_FIELD_TYPE", "dataset_metadata.created")]


def test_a_huge_integer_root_is_not_an_object_and_a_huge_float_is_finite_text():
    assert parse_issues("9" * 100000) == [("PARSE_NOT_OBJECT", "dataset_metadata")]
    assert parse_issues("1" + "0" * 300 + ".5") == [("PARSE_NOT_OBJECT", "dataset_metadata")]


def test_number_tokens_in_every_form_are_field_type_issues():
    for token in ("0", "-0", "12", "1.0", "1e3", "-2.5E-3"):
        text = good_json().replace('"v1"', token)
        assert parse_issues(text) == [("PARSE_FIELD_TYPE", "dataset_metadata.dataset_version")], token


def test_parsing_does_not_depend_on_the_interpreter_digit_limit():
    limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else None
    if limit is None:
        pytest.skip("no digit limit on this interpreter")
    sys.set_int_max_str_digits(640)
    try:
        text = good_json().replace('"v1"', "1" * 1000)
        assert parse_issues(text) == [("PARSE_FIELD_TYPE", "dataset_metadata.dataset_version")]
    finally:
        sys.set_int_max_str_digits(limit)


def test_a_duplicate_key_is_reported_and_the_first_value_is_kept():
    text = good_json().replace("}", ', "created": "2030-01-01"}')
    assert parse_issues(text) == [("PARSE_DUPLICATE_KEY", "dataset_metadata")]


def test_duplicate_keys_in_a_nested_object_are_reported_too():
    text = '{"schema_version": {"a": 1, "a": 2}, "dataset_version": "v", "corpus_version": "c", ' \
           '"corpus_manifest_hash": "h", "created": "d"}'
    assert parse_issues(text) == [("PARSE_DUPLICATE_KEY", "dataset_metadata"), ("PARSE_FIELD_TYPE", "dataset_metadata.schema_version")]


def test_duplicate_keys_and_non_finite_numbers_are_collected_together():
    text = '{"created": NaN, "created": "x", "dataset_version": 1e999, "schema_version": "a", "corpus_version": "b", ' \
           '"corpus_manifest_hash": "c"}'
    # Values are read before their object is assembled, so a duplicate key follows the numbers inside it.
    assert parse_issues(text) == [
        ("PARSE_NON_FINITE", "dataset_metadata"), ("PARSE_NON_FINITE", "dataset_metadata"), ("PARSE_DUPLICATE_KEY", "dataset_metadata")
    ]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_a_non_finite_number_in_an_unknown_field_gives_only_unknown_field(literal):
    text = good_json()[:-1] + ', "extra": ' + literal + "}"
    assert parse_issues(text) == [("PARSE_UNKNOWN_FIELD", "dataset_metadata")]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("[NaN, 1e999]", id="list"),
        pytest.param('{"a": Infinity, "b": [-Infinity]}', id="nested-object"),
        pytest.param('[[{"x": NaN}]]', id="deep"),
    ],
)
def test_non_finite_numbers_nested_in_an_unknown_field_are_not_reported(value):
    text = good_json()[:-1] + ', "extra": ' + value + "}"
    assert parse_issues(text) == [("PARSE_UNKNOWN_FIELD", "dataset_metadata")]


def test_unknown_field_with_non_finite_does_not_hide_other_problems():
    text = '{"extra": NaN, "schema_version": 1, "dataset_version": "v", "corpus_version": "c", "corpus_manifest_hash": "h"}'
    assert parse_issues(text) == [
        ("PARSE_UNKNOWN_FIELD", "dataset_metadata"),
        ("PARSE_MISSING_FIELD", "dataset_metadata.created"),
        ("PARSE_FIELD_TYPE", "dataset_metadata.schema_version"),
    ]


def test_non_finite_numbers_in_known_fields_are_still_reported_alongside_an_unknown_one():
    text = '{"extra": NaN, "created": Infinity, "dataset_version": "v", "schema_version": "a", "corpus_version": "b", ' \
           '"corpus_manifest_hash": -1e999}'
    assert parse_issues(text) == [("PARSE_NON_FINITE", "dataset_metadata")] * 2 + [("PARSE_UNKNOWN_FIELD", "dataset_metadata")]


def test_non_finite_numbers_in_a_known_fields_list_value_are_reported_with_the_type_issue():
    assert parse_issues(good_json(dataset_version=[1]).replace("[1]", "[NaN]")) == [
        ("PARSE_NON_FINITE", "dataset_metadata"), ("PARSE_FIELD_TYPE", "dataset_metadata.dataset_version")
    ]


def test_a_repeated_known_key_with_a_non_finite_dropped_value_is_still_reported():
    text = good_json()[:-1] + ', "created": NaN}'
    assert parse_issues(text) == [("PARSE_NON_FINITE", "dataset_metadata"), ("PARSE_DUPLICATE_KEY", "dataset_metadata")]


def test_a_repeated_unknown_key_with_non_finite_values_is_not_reported_as_non_finite():
    text = good_json()[:-1] + ', "extra": 1, "extra": NaN}'
    assert parse_issues(text) == [("PARSE_DUPLICATE_KEY", "dataset_metadata"), ("PARSE_UNKNOWN_FIELD", "dataset_metadata")]


def test_a_non_finite_inside_a_non_object_root_is_still_reported():
    assert parse_issues("[NaN]") == [("PARSE_NON_FINITE", "dataset_metadata"), ("PARSE_NOT_OBJECT", "dataset_metadata")]


def test_no_issue_location_uses_the_old_metadata_prefix():
    bad_parse = [
        "", "{", "[1]", "\ufeff{}", "NaN", good_json(created=1), '{"extra": 1}',
        good_json()[:-1] + ', "created": "x"}',
    ]
    locations = []
    for text in bad_parse:
        locations += [loc for _, loc in parse_issues(text)]
    locations += [loc for _, loc in check(m.DatasetMetadata(1, "", None, "zz", "nope"), INVALID)]
    assert locations
    for location in locations:
        assert location.split(" ")[0].split(".")[0] in ("dataset_metadata", "manifest")
        assert not location.startswith("metadata")


def test_parse_issues_never_contain_input_text():
    text = json.dumps({CANARY: 1, "schema_version": CANARY, "dataset_version": [CANARY], "created": CANARY})
    with pytest.raises(s.EvaluationDataError) as caught:
        m.parse_dataset_metadata(text)
    for issue in caught.value.issues:
        assert CANARY not in issue.code + issue.location + issue.message
    with pytest.raises(s.EvaluationDataError) as caught:
        m.parse_dataset_metadata("{" + CANARY)
    assert all(CANARY not in i.location + i.message for i in caught.value.issues)
    assert CANARY not in str(caught.value)


def test_parse_messages_match_the_shared_parse_messages():
    for code, message in m._PARSE_MESSAGES.items():
        assert p._MESSAGES[code] == message


def test_parse_codes_are_the_existing_parse_codes():
    used = {c for c in m._PARSE_MESSAGES}
    assert used <= {v for k, v in vars(p).items() if k.startswith("PARSE_")}


# ---------- E. validate: field rules ----------

def test_valid_metadata_has_no_issues_for_ingested_and_draft_manifests():
    assert check() == []
    assert check(meta(corpus_manifest_hash=DRAFT_HASH), DRAFT) == []


def test_validated_parse_output_round_trips():
    assert check(m.parse_dataset_metadata(good_json())) == []


@pytest.mark.parametrize(
    "field, value, code",
    [
        pytest.param("schema_version", "eval-dataset-v2", "META_SCHEMA_VERSION", id="schema-version"),
        pytest.param("schema_version", "", "META_SCHEMA_VERSION", id="schema-version-empty"),
        pytest.param("schema_version", None, "META_FIELD_TYPE", id="schema-version-none"),
        pytest.param("dataset_version", "", "META_BLANK", id="dataset-version-empty"),
        pytest.param("dataset_version", " \t\n", "META_BLANK", id="dataset-version-space"),
        pytest.param("dataset_version", 3, "META_FIELD_TYPE", id="dataset-version-int"),
        pytest.param("corpus_version", "  ", "META_BLANK", id="corpus-version-blank"),
        pytest.param("corpus_version", True, "META_FIELD_TYPE", id="corpus-version-bool"),
        pytest.param("corpus_manifest_hash", "A" * 64, "META_HASH_FORMAT", id="hash-upper"),
        pytest.param("corpus_manifest_hash", "a" * 63, "META_HASH_FORMAT", id="hash-short"),
        pytest.param("corpus_manifest_hash", "a" * 65, "META_HASH_FORMAT", id="hash-long"),
        pytest.param("corpus_manifest_hash", "a" * 63 + "g", "META_HASH_FORMAT", id="hash-non-hex"),
        pytest.param("corpus_manifest_hash", "a" * 64 + "\n", "META_HASH_FORMAT", id="hash-newline"),
        pytest.param("corpus_manifest_hash", b"a" * 64, "META_FIELD_TYPE", id="hash-bytes"),
        pytest.param("created", "2026-02-30", "META_DATE", id="created-impossible"),
        pytest.param("created", "2026-1-1", "META_DATE", id="created-short"),
        pytest.param("created", "2026-11-11T00:00:00", "META_DATE", id="created-datetime"),
        pytest.param("created", "2026-11-11\n", "META_DATE", id="created-newline"),
        pytest.param("created", "", "META_DATE", id="created-empty"),
        pytest.param("created", 20261111, "META_FIELD_TYPE", id="created-int"),
    ],
)
def test_each_field_rule(field, value, code):
    result = check(meta(**{field: value}))
    assert (code, f"dataset_metadata.{field}") in result
    assert not [c for c, loc in result if loc == f"dataset_metadata.{field}" and c != code]


def test_a_str_subclass_is_a_type_issue():
    class S(str):
        pass

    assert ("META_FIELD_TYPE", "dataset_metadata.dataset_version") in check(meta(dataset_version=S("v1")))
    assert ("META_FIELD_TYPE", "dataset_metadata.schema_version") in check(meta(schema_version=S(s.DATASET_SCHEMA_VERSION)))


def test_lone_surrogates_and_very_long_strings_do_not_raise():
    assert check(meta(dataset_version="\ud800", created="\ud800")) == [("META_DATE", "dataset_metadata.created")]
    assert check(meta(dataset_version="x" * 100000)) == []
    assert ("META_DATE", "dataset_metadata.created") in check(meta(created="9" * 100000))


def test_field_issues_come_in_field_order_and_all_are_collected():
    bad = m.DatasetMetadata(1, "", None, "zz", "nope")
    assert check(bad) == [
        ("META_FIELD_TYPE", "dataset_metadata.schema_version"),
        ("META_BLANK", "dataset_metadata.dataset_version"),
        ("META_FIELD_TYPE", "dataset_metadata.corpus_version"),
        ("META_HASH_FORMAT", "dataset_metadata.corpus_manifest_hash"),
        ("META_DATE", "dataset_metadata.created"),
    ]


def test_created_uses_the_public_dataset_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "is_valid_iso_date", lambda value: calls.append(value) or False)
    assert ("META_DATE", "dataset_metadata.created") in check()
    assert calls == ["2026-11-11"]


def test_arguments_are_type_checked():
    for bad in (None, "x", {"created": "2026-11-11"}):
        with pytest.raises(TypeError):
            m.validate_dataset_metadata(bad, MANIFEST)
    for bad in (None, "x", {}):
        with pytest.raises(TypeError):
            m.validate_dataset_metadata(meta(), bad)


# ---------- F. validate: manifest gate and comparisons ----------

def test_an_invalid_manifest_is_exactly_one_error_at_manifest_and_skips_the_comparisons():
    result = check(meta(corpus_version="different", corpus_manifest_hash="b" * 64), INVALID)
    assert result == [("META_MANIFEST_INVALID", "manifest")]


def test_an_invalid_manifest_still_reports_field_issues_first():
    assert check(meta(created="nope"), INVALID) == [
        ("META_DATE", "dataset_metadata.created"), ("META_MANIFEST_INVALID", "manifest")
    ]


def test_a_draft_manifest_is_accepted_with_no_stage_requirement():
    assert check(meta(corpus_manifest_hash=DRAFT_HASH), DRAFT) == []
    assert not any(c == "META_MANIFEST_INVALID" for c, _ in check(meta(corpus_manifest_hash=DRAFT_HASH), DRAFT))


def test_manifest_warnings_do_not_trip_the_gate(monkeypatch):
    warning = s.Issue("W", "manifest", "m", s.SEVERITY_WARNING)
    monkeypatch.setattr(m, "validate_manifest", lambda manifest: [warning])
    assert check() == []


def test_corpus_version_mismatch():
    assert check(meta(corpus_version="c2")) == [("META_CORPUS_VERSION_MISMATCH", "dataset_metadata.corpus_version")]


def test_corpus_version_comparison_is_exact():
    assert ("META_CORPUS_VERSION_MISMATCH", "dataset_metadata.corpus_version") in check(meta(corpus_version="C1"))
    assert ("META_CORPUS_VERSION_MISMATCH", "dataset_metadata.corpus_version") in check(meta(corpus_version="c1 "))


def test_manifest_hash_mismatch():
    assert check(meta(corpus_manifest_hash="b" * 64)) == [("META_MANIFEST_HASH_MISMATCH", "dataset_metadata.corpus_manifest_hash")]


def test_the_hash_is_of_the_whole_manifest_so_a_stage_change_is_a_mismatch():
    assert check(meta(corpus_manifest_hash=GOOD_HASH), DRAFT) == [
        ("META_MANIFEST_HASH_MISMATCH", "dataset_metadata.corpus_manifest_hash")
    ]


def test_both_comparisons_can_fail_together_in_order():
    assert check(meta(corpus_version="c2", corpus_manifest_hash="b" * 64)) == [
        ("META_CORPUS_VERSION_MISMATCH", "dataset_metadata.corpus_version"),
        ("META_MANIFEST_HASH_MISMATCH", "dataset_metadata.corpus_manifest_hash"),
    ]


def test_a_comparison_is_made_only_if_its_field_passed_validation():
    assert check(meta(corpus_version="")) == [("META_BLANK", "dataset_metadata.corpus_version")]
    assert check(meta(corpus_version=5)) == [("META_FIELD_TYPE", "dataset_metadata.corpus_version")]
    assert check(meta(corpus_manifest_hash="B" * 64)) == [("META_HASH_FORMAT", "dataset_metadata.corpus_manifest_hash")]
    assert check(meta(corpus_manifest_hash=None)) == [("META_FIELD_TYPE", "dataset_metadata.corpus_manifest_hash")]


def test_the_hash_is_not_computed_when_the_manifest_is_invalid_or_the_field_is_bad(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "manifest_hash", lambda manifest: calls.append(1) or GOOD_HASH)
    check(meta(), INVALID)
    check(meta(corpus_manifest_hash="zz"))
    assert calls == []
    check()
    assert calls == [1]


def test_a_manifest_hash_failure_is_one_manifest_invalid_issue(monkeypatch):
    def boom(manifest):
        raise s.EvaluationDataError([s.Issue("HASH_NOT_SERIALISABLE", "manifest", "m")])

    monkeypatch.setattr(m, "manifest_hash", boom)
    assert check(meta(corpus_version="c2")) == [
        ("META_CORPUS_VERSION_MISMATCH", "dataset_metadata.corpus_version"), ("META_MANIFEST_INVALID", "manifest")
    ]


def test_the_gate_and_a_hash_failure_never_give_two_manifest_invalid_issues(monkeypatch):
    def boom(manifest):
        raise s.EvaluationDataError([s.Issue("HASH_NOT_SERIALISABLE", "manifest", "m")])

    monkeypatch.setattr(m, "manifest_hash", boom)
    assert check(meta(), INVALID) == [("META_MANIFEST_INVALID", "manifest")]


def test_other_errors_from_hashing_are_not_swallowed(monkeypatch):
    def boom(manifest):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(m, "manifest_hash", boom)
    with pytest.raises(RuntimeError):
        check()


# ---------- G. whole-result properties and privacy ----------

def test_every_meta_code_is_reachable_with_a_fixed_message_and_error_severity():
    assert set(m._META_MESSAGES) == {
        "META_FIELD_TYPE", "META_BLANK", "META_SCHEMA_VERSION", "META_HASH_FORMAT", "META_DATE",
        "META_MANIFEST_INVALID", "META_CORPUS_VERSION_MISMATCH", "META_MANIFEST_HASH_MISMATCH",
    }
    seen = set()
    cases = [
        (m.DatasetMetadata(1, "", "c2", "zz", "nope"), MANIFEST),
        (meta(schema_version="x", corpus_version="c2", corpus_manifest_hash="b" * 64), MANIFEST),
        (meta(), INVALID),
    ]
    for metadata, manifest in cases:
        for issue in m.validate_dataset_metadata(metadata, manifest):
            seen.add(issue.code)
            assert issue.message == m._META_MESSAGES[issue.code] and issue.severity == "error"
    assert seen == set(m._META_MESSAGES)


def test_the_literal_code_strings_are_pinned():
    assert (m.META_MANIFEST_INVALID, m.META_FIELD_TYPE, m.META_BLANK, m.META_SCHEMA_VERSION) == (
        "META_MANIFEST_INVALID", "META_FIELD_TYPE", "META_BLANK", "META_SCHEMA_VERSION"
    )
    assert (m.META_HASH_FORMAT, m.META_DATE, m.META_CORPUS_VERSION_MISMATCH, m.META_MANIFEST_HASH_MISMATCH) == (
        "META_HASH_FORMAT", "META_DATE", "META_CORPUS_VERSION_MISMATCH", "META_MANIFEST_HASH_MISMATCH"
    )


def test_no_user_controlled_value_appears_in_any_validation_issue():
    metadata = m.DatasetMetadata(CANARY, CANARY, CANARY, CANARY, CANARY)
    for manifest in (MANIFEST, INVALID):
        for issue in m.validate_dataset_metadata(metadata, manifest):
            assert CANARY not in issue.code + issue.location + issue.message


def test_validation_is_deterministic_and_does_not_modify_its_inputs():
    metadata, manifest = meta(corpus_version="c2"), MANIFEST
    before = (dataclasses.asdict(metadata), dataclasses.asdict(manifest))
    first = m.validate_dataset_metadata(metadata, manifest)
    assert m.validate_dataset_metadata(metadata, manifest) == first
    assert (dataclasses.asdict(metadata), dataclasses.asdict(manifest)) == before


def test_parse_is_deterministic():
    assert parse_issues("{") == parse_issues("{")
    assert m.parse_dataset_metadata(good_json()) == m.parse_dataset_metadata(good_json())


def test_the_module_does_not_use_private_parsing_helpers():
    source = (PROJECT_ROOT / "app" / "rag" / "evaluation" / "dataset_metadata.py").read_text(encoding="utf-8")
    import re

    for line in re.findall(r"from app\.rag\.evaluation\.parsing import \((.*?)\)", source, re.S):
        assert not [n for n in re.findall(r"\w+", line) if n.startswith("_")]


# ---------- H. import isolation ----------

def test_importing_the_module_loads_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import app.rag.evaluation.dataset_metadata\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "print(sorted(mod for mod in sys.modules if mod.split('.')[0] in heavy))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
