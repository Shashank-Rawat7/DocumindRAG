"""Dataset metadata (``dataset.json``): parsing and validation (E1.2d, step 5).

``DatasetMetadata`` has exactly five fields. ``parse_dataset_metadata`` reads JSON text with
its own strict loader and checks structure and types only; ``validate_dataset_metadata``
checks values and compares them with a manifest. The freeze-hash fields belong to E1.8.

Parsing: a leading BOM, malformed JSON, duplicate keys, ``NaN``, ``Infinity`` and ``1e999``
and over-deep nesting are rejected. Numbers are never converted to Python integers (an
integer hook keeps only a marker), so no interpreter digit limit applies; any number where a
string is required is a field-type issue. A non-finite number is reported once and the value
it stands for is not reported again; one inside an unknown top-level field is not reported at
all, because the field is reported as unknown. Every independently detectable problem is collected and
``EvaluationDataError`` carries all of them; a partial object is never returned.

Validation: ``schema_version`` equals ``DATASET_SCHEMA_VERSION``; ``dataset_version`` and
``corpus_version`` are non-blank; ``corpus_manifest_hash`` is 64 lowercase hexadecimal
characters; ``created`` is a real ``YYYY-MM-DD`` date. The manifest gate runs
``validate_manifest``: a manifest with errors gives exactly one ``META_MANIFEST_INVALID`` at
``manifest`` and skips the two comparisons. There is no stage requirement, so a draft manifest
is accepted. Otherwise ``corpus_version`` must equal the manifest's, and
``corpus_manifest_hash`` must equal ``manifest_hash(manifest)``; each comparison is made only
if that field passed its own checks. If hashing the manifest fails, that is reported as
``META_MANIFEST_INVALID`` too (still at most one).

No issue contains user-controlled content: messages are fixed, locations are the fixed names
``dataset_metadata``, ``dataset_metadata.<field>`` and ``manifest`` plus syntax line and column numbers.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from app.rag.evaluation.dataset import is_valid_iso_date
from app.rag.evaluation.parsing import (
    PARSE_BOM,
    PARSE_DUPLICATE_KEY,
    PARSE_FIELD_TYPE,
    PARSE_MISSING_FIELD,
    PARSE_NON_FINITE,
    PARSE_NOT_OBJECT,
    PARSE_SYNTAX,
    PARSE_TOO_DEEP,
    PARSE_UNKNOWN_FIELD,
    manifest_hash,
)
from app.rag.evaluation.schema import (
    DATASET_SCHEMA_VERSION,
    SHA256_HEX_PATTERN,
    EvaluationDataError,
    Issue,
    Manifest,
)
from app.rag.evaluation.validation import validate_manifest

META_FIELD_TYPE = "META_FIELD_TYPE"
META_BLANK = "META_BLANK"
META_SCHEMA_VERSION = "META_SCHEMA_VERSION"
META_HASH_FORMAT = "META_HASH_FORMAT"
META_DATE = "META_DATE"
META_MANIFEST_INVALID = "META_MANIFEST_INVALID"
META_CORPUS_VERSION_MISMATCH = "META_CORPUS_VERSION_MISMATCH"
META_MANIFEST_HASH_MISMATCH = "META_MANIFEST_HASH_MISMATCH"

FIELDS = ("schema_version", "dataset_version", "corpus_version", "corpus_manifest_hash", "created")

_PARSE_MESSAGES = {
    PARSE_BOM: "input must not start with a byte order mark",
    PARSE_SYNTAX: "input is not valid JSON",
    PARSE_DUPLICATE_KEY: "a JSON object has a duplicate key",
    PARSE_NON_FINITE: "NaN and infinite numbers are not allowed",
    PARSE_TOO_DEEP: "input is nested too deeply",
    PARSE_NOT_OBJECT: "a JSON object is required here",
    PARSE_UNKNOWN_FIELD: "an object has a field that is not part of the schema",
    PARSE_MISSING_FIELD: "a required field is missing",
    PARSE_FIELD_TYPE: "a field has the wrong type",
}

_META_MESSAGES = {
    META_FIELD_TYPE: "a metadata field has the wrong type",
    META_BLANK: "a metadata field must not be blank",
    META_SCHEMA_VERSION: "schema_version is not the supported dataset schema version",
    META_HASH_FORMAT: "corpus_manifest_hash must be 64 lowercase hexadecimal characters",
    META_DATE: "created must be a real date in YYYY-MM-DD form",
    META_MANIFEST_INVALID: "the manifest is not valid, so it cannot be compared with the metadata",
    META_CORPUS_VERSION_MISMATCH: "corpus_version differs from the manifest's corpus_version",
    META_MANIFEST_HASH_MISMATCH: "corpus_manifest_hash differs from the hash of the manifest",
}


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """The five metadata fields of a dataset. Not validated on construction."""

    schema_version: str
    dataset_version: str
    corpus_version: str
    corpus_manifest_hash: str
    created: str


# ---------- parsing ----------

_BOM = "﻿"


class _Poison:
    """Stands in for a non-finite number; it is reported unless it sits inside an unknown field."""


class _Object(dict):
    """A parsed JSON object. ``dropped`` holds the values of repeated keys, in reading order."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped: list[tuple[str, object]] = []


class _Number:
    """Stands in for a JSON number: its value is never built or kept."""


_NUMBER = _Number()


def _parse_issue(code: str, location: str) -> Issue:
    return Issue(code, location, _PARSE_MESSAGES[code])


def _poisons_in(value: object, found: list[_Poison]) -> None:
    """Every ``_Poison`` reachable from ``value``, including values dropped for repeated keys."""
    if isinstance(value, _Poison):
        found.append(value)
    elif isinstance(value, _Object):
        for item in value.values():
            _poisons_in(item, found)
        for _, item in value.dropped:
            _poisons_in(item, found)
    elif isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            _poisons_in(item, found)


def _load(text: str, where: str) -> tuple[object, list[Issue], bool]:
    """Parse one JSON document. Returns (value, issues, readable).

    Duplicate keys and non-finite numbers do not stop the parse. Both are recorded in reading
    order, but a non-finite number is only reported once the whole document is read, so that
    those inside an unknown top-level field (reported as unknown instead) can be left out.
    """
    events: list[Issue | _Poison] = []

    def pairs(items: list) -> _Object:
        result = _Object()
        for key, value in items:
            if key in result:
                events.append(_parse_issue(PARSE_DUPLICATE_KEY, where))
                result.dropped.append((key, value))
            else:
                result[key] = value
        return result

    def constant(_name: str) -> _Poison:
        events.append(_Poison())
        return events[-1]

    def integer(_literal: str) -> _Number:
        return _NUMBER  # never builds a Python int

    def real(literal: str) -> _Number | _Poison:
        if math.isfinite(float(literal)):
            return _NUMBER
        events.append(_Poison())  # "1e999" overflows without using parse_constant
        return events[-1]

    try:
        value = json.loads(
            text, object_pairs_hook=pairs, parse_int=integer, parse_float=real, parse_constant=constant
        )
    except json.JSONDecodeError as error:
        return None, [_parse_issue(PARSE_SYNTAX, f"{where} line {error.lineno} column {error.colno}")], False
    except RecursionError:
        return None, [_parse_issue(PARSE_TOO_DEEP, where)], False

    if isinstance(value, _Object):
        counted: list[_Poison] = []
        for name in FIELDS:
            if name in value:
                _poisons_in(value[name], counted)
            for key, item in value.dropped:
                if key == name:
                    _poisons_in(item, counted)
        reported = {id(item) for item in counted}
    else:
        reported = {id(item) for item in events if isinstance(item, _Poison)}
    issues = [
        item if isinstance(item, Issue) else _parse_issue(PARSE_NON_FINITE, where)
        for item in events
        if isinstance(item, Issue) or id(item) in reported
    ]
    return value, issues, True


def parse_dataset_metadata(text: str) -> DatasetMetadata:
    """Read dataset metadata. Raises ``EvaluationDataError`` listing every structural problem."""
    if type(text) is not str:
        raise TypeError("text must be a str")
    issues: list[Issue] = []
    if text.startswith(_BOM):
        issues.append(_parse_issue(PARSE_BOM, "dataset_metadata"))
        text = text[len(_BOM):]
    value, strict_issues, readable = _load(text, "dataset_metadata")
    issues.extend(strict_issues)
    if not readable:
        raise EvaluationDataError(issues)
    if isinstance(value, _Poison):
        raise EvaluationDataError(issues)  # already reported as a non-finite number
    if type(value) is not _Object:
        issues.append(_parse_issue(PARSE_NOT_OBJECT, "dataset_metadata"))
        raise EvaluationDataError(issues)
    for key in value:
        if key not in FIELDS:
            issues.append(_parse_issue(PARSE_UNKNOWN_FIELD, "dataset_metadata"))  # the key itself is never reported
    for name in FIELDS:
        if name not in value:
            issues.append(_parse_issue(PARSE_MISSING_FIELD, f"dataset_metadata.{name}"))
    for name in FIELDS:
        if name in value and not isinstance(value[name], _Poison) and type(value[name]) is not str:
            issues.append(_parse_issue(PARSE_FIELD_TYPE, f"dataset_metadata.{name}"))
    if issues:
        raise EvaluationDataError(issues)
    return DatasetMetadata(**{name: value[name] for name in FIELDS})


# ---------- validation ----------


def _meta_issue(code: str, location: str) -> Issue:
    return Issue(code, location, _META_MESSAGES[code])


def _text_ok(issues: list[Issue], value: object, location: str) -> bool:
    if type(value) is not str:
        issues.append(_meta_issue(META_FIELD_TYPE, location))
        return False
    if value.strip() == "":
        issues.append(_meta_issue(META_BLANK, location))
        return False
    return True


def validate_dataset_metadata(metadata: DatasetMetadata, manifest: Manifest) -> list[Issue]:
    """Every problem in ``metadata``, in a fixed order; an empty list means none.

    Raises ``TypeError`` if ``metadata`` is not a ``DatasetMetadata`` or ``manifest`` is not a
    ``Manifest``. Bad data is reported as issues, never raised. Neither argument is modified.
    """
    if not isinstance(metadata, DatasetMetadata):
        raise TypeError("metadata must be a DatasetMetadata")
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")

    issues: list[Issue] = []

    version = metadata.schema_version
    if type(version) is not str:
        issues.append(_meta_issue(META_FIELD_TYPE, "dataset_metadata.schema_version"))
    elif version != DATASET_SCHEMA_VERSION:
        issues.append(_meta_issue(META_SCHEMA_VERSION, "dataset_metadata.schema_version"))

    _text_ok(issues, metadata.dataset_version, "dataset_metadata.dataset_version")
    corpus_version_ok = _text_ok(issues, metadata.corpus_version, "dataset_metadata.corpus_version")

    digest = metadata.corpus_manifest_hash
    hash_ok = False
    if type(digest) is not str:
        issues.append(_meta_issue(META_FIELD_TYPE, "dataset_metadata.corpus_manifest_hash"))
    elif SHA256_HEX_PATTERN.fullmatch(digest) is None:
        issues.append(_meta_issue(META_HASH_FORMAT, "dataset_metadata.corpus_manifest_hash"))
    else:
        hash_ok = True

    created = metadata.created
    if type(created) is not str:
        issues.append(_meta_issue(META_FIELD_TYPE, "dataset_metadata.created"))
    elif not is_valid_iso_date(created):
        issues.append(_meta_issue(META_DATE, "dataset_metadata.created"))

    # Manifest gate: errors only, any stage.
    if any(item.severity == "error" for item in validate_manifest(manifest)):
        issues.append(_meta_issue(META_MANIFEST_INVALID, "manifest"))
        return issues

    if corpus_version_ok and metadata.corpus_version != manifest.corpus_version:
        issues.append(_meta_issue(META_CORPUS_VERSION_MISMATCH, "dataset_metadata.corpus_version"))
    if hash_ok:
        try:
            expected = manifest_hash(manifest)
        except EvaluationDataError:
            issues.append(_meta_issue(META_MANIFEST_INVALID, "manifest"))
        else:
            if digest != expected:
                issues.append(_meta_issue(META_MANIFEST_HASH_MISMATCH, "dataset_metadata.corpus_manifest_hash"))
    return issues
