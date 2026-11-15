"""Freeze-binding gate of the retrieval-only baseline (E2.1).

A baseline run must be provably tied to the frozen dataset ``eval-dataset-v1.0.0``. This module checks
``eval/dataset/freeze_record.json`` against the files it describes and returns a frozen ``FreezeBinding``
with the verified hashes. It is a gate only: it runs no retrieval, builds no report and changes nothing.

What is verified, for the DEVELOPMENT side only:

* the record itself: strict JSON (no BOM, duplicate keys, NaN or infinity), a closed schema for the parts
  the gate uses (top level, ``files``, ``corpus``, ``label_binding_hashes``), ``record_format`` equal to
  ``eval-freeze-record-v1`` and ``status`` equal to ``frozen_dataset_not_baseline``. The informational
  sections (``counts``, ``review_provenance``, ``audit``, ``pilot_reference``, ``git_state``,
  ``limitations``) must be present and of the right JSON type; their contents are not interpreted;
* the raw SHA-256 of ``dev.jsonl``, ``dataset.json`` and ``manifest.json`` against ``files``;
* the dev label-binding hash, recomputed with the existing ``parsing.dataset_hash`` over the questions
  ``parsing.parse_questions`` reads from the same ``dev.jsonl`` bytes (no second implementation);
* the canonical manifest hash, recomputed with the existing ``parsing.manifest_hash`` from the same
  ``manifest.json`` bytes, against ``corpus.canonical_manifest_hash``.

The held-out test split is never touched. The record lists the hash of ``test.jsonl`` and of other private
files; those values are only checked for their format (64 lowercase hex digits). The module never opens,
hashes, stats, lists or resolves any of those files, and ``load_freeze_binding`` reads exactly four fixed
paths (``READ_PATHS``). A matching hash shows the file is the frozen file; it does not show that its labels
are correct.

``build_freeze_binding`` is pure: bytes in, ``FreezeBinding`` out. Every file is read once and the same
bytes are hashed and parsed, so a file cannot change between the hash check and the label-binding check.

Filesystem safety. Before any file is opened, ``load_freeze_binding`` calls ``os.lstat`` (which never follows
a link) on the directories ``eval``, ``eval/dataset`` and ``eval/corpus`` and then on the four input files. A
symbolic link, a Windows reparse point (junction, symlink, cloud placeholder) or an entry of the wrong type
(a directory where a file belongs, a FIFO, a device) is refused with ``FREEZE_PATH_UNSAFE``, so a link such as
``dev.jsonl -> test.jsonl`` is never followed. ``resolve()`` and ``realpath()`` are not used, because they
would stat the link target. The repository root itself may be a link. Not covered: hard links (the gate does
not detect them) and a path swapped between the check and the read.

Errors. ``TypeError`` for a wrongly typed argument. ``EvaluationDataError`` carries every finding, with
fixed messages and schema-defined locations. No issue, message or exception holds a question, an answer,
evidence, document text, a record value or a field name read from the data.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.evaluation.parsing import (
    dataset_hash,
    manifest_hash,
    parse_manifest,
    parse_questions,
)
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    SHA256_HEX_PATTERN,
    EvaluationDataError,
    Issue,
)

RECORD_FORMAT = "eval-freeze-record-v1"
FROZEN_STATUS = "frozen_dataset_not_baseline"

RECORD_PATH = "eval/dataset/freeze_record.json"
DEV_PATH = "eval/dataset/dev.jsonl"
DATASET_JSON_PATH = "eval/dataset/dataset.json"
MANIFEST_PATH = "eval/corpus/manifest.json"

# The only files this module ever reads.
READ_PATHS = (RECORD_PATH, DEV_PATH, DATASET_JSON_PATH, MANIFEST_PATH)
# Files whose raw SHA-256 is verified against ``files``.
VERIFIED_FILE_PATHS = (DEV_PATH, DATASET_JSON_PATH, MANIFEST_PATH)
# Entries of ``files`` that are recorded but never read, hashed or stat'ed here (the first one is the
# held-out test split).
RECORDED_ONLY_PATHS = (
    "eval/dataset/test.jsonl",
    "eval/dataset/audit.jsonl",
    "eval/dataset/guidelines.md",
    "eval/dataset/draft/README.md",
)

_TOP_LEVEL = (
    "record_format", "dataset_version", "frozen_on", "status", "anchoring", "files", "corpus",
    "label_binding_hashes", "counts", "review_provenance", "audit", "pilot_reference", "git_state",
    "limitations",
)
_OBJECT_SECTIONS = ("counts", "review_provenance", "audit", "pilot_reference", "git_state")
_CORPUS_FIELDS = ("corpus_version", "canonical_manifest_hash")
_LABEL_BINDING_FIELDS = ("dev", "test")

_VERSION_PATTERN = re.compile(r"eval-dataset-v\d+\.\d+\.\d+")
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

FREEZE_FILE_UNREADABLE = "FREEZE_FILE_UNREADABLE"
FREEZE_RECORD_SYNTAX = "FREEZE_RECORD_SYNTAX"
FREEZE_FIELD_MISSING = "FREEZE_FIELD_MISSING"
FREEZE_FIELD_UNKNOWN = "FREEZE_FIELD_UNKNOWN"
FREEZE_FIELD_TYPE = "FREEZE_FIELD_TYPE"
FREEZE_RECORD_FORMAT = "FREEZE_RECORD_FORMAT"
FREEZE_STATUS = "FREEZE_STATUS"
FREEZE_VERSION_FORMAT = "FREEZE_VERSION_FORMAT"
FREEZE_DATE_FORMAT = "FREEZE_DATE_FORMAT"
FREEZE_HASH_FORMAT = "FREEZE_HASH_FORMAT"
FREEZE_FILE_HASH_MISMATCH = "FREEZE_FILE_HASH_MISMATCH"
FREEZE_DEV_UNPARSEABLE = "FREEZE_DEV_UNPARSEABLE"
FREEZE_LABEL_BINDING_MISMATCH = "FREEZE_LABEL_BINDING_MISMATCH"
FREEZE_MANIFEST_UNPARSEABLE = "FREEZE_MANIFEST_UNPARSEABLE"
FREEZE_MANIFEST_HASH_MISMATCH = "FREEZE_MANIFEST_HASH_MISMATCH"
FREEZE_PATH_UNSAFE = "FREEZE_PATH_UNSAFE"

_MESSAGES = {
    FREEZE_FILE_UNREADABLE: "a file of the freeze binding is missing or cannot be read",
    FREEZE_RECORD_SYNTAX: "the freeze record is not strict UTF-8 JSON",
    FREEZE_FIELD_MISSING: "a required field of the freeze record is missing",
    FREEZE_FIELD_UNKNOWN: "the freeze record holds a field that its schema does not define",
    FREEZE_FIELD_TYPE: "a field of the freeze record has the wrong type or an empty value",
    FREEZE_RECORD_FORMAT: "the freeze record has an unsupported record_format",
    FREEZE_STATUS: "the freeze record does not have the status of a frozen, non-baseline dataset",
    FREEZE_VERSION_FORMAT: "dataset_version is not of the form eval-dataset-vMAJOR.MINOR.PATCH",
    FREEZE_DATE_FORMAT: "frozen_on is not a valid YYYY-MM-DD date",
    FREEZE_HASH_FORMAT: "a recorded hash is not 64 lowercase hexadecimal digits",
    FREEZE_FILE_HASH_MISMATCH: "the SHA-256 of a file differs from the freeze record",
    FREEZE_DEV_UNPARSEABLE: "the dev file could not be read as development-split questions",
    FREEZE_LABEL_BINDING_MISMATCH: "the dev label-binding hash differs from the freeze record",
    FREEZE_MANIFEST_UNPARSEABLE: "the corpus manifest could not be parsed",
    FREEZE_MANIFEST_HASH_MISMATCH: "the canonical manifest hash differs from the freeze record",
    FREEZE_PATH_UNSAFE: "a path of the freeze binding is a link, a reparse point or not an ordinary file or directory",
}


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_ERROR)


@dataclass(frozen=True, slots=True)
class FreezeBinding:
    """The verified identity of the frozen development inputs of one baseline run.

    Every hash was recomputed from the bytes of the files named in the record; none is merely copied from
    it. ``test_label_binding_sha256`` is not here: the test split is never read, so it cannot be verified.
    """

    dataset_version: str
    frozen_on: str
    corpus_version: str
    record_sha256: str
    dev_file_sha256: str
    dataset_json_sha256: str
    manifest_file_sha256: str
    dev_label_binding_sha256: str
    canonical_manifest_sha256: str


# ---------- strict reading of the record ----------

class _Duplicate(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise _Duplicate
        result[key] = value
    return result


def _constant(_name: str) -> Any:
    raise _NonFinite


def _number(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):  # "1e999" overflows without going through parse_constant
        raise _NonFinite
    return value


_NOT_JSON = object()  # JSON ``null`` parses to None, so None cannot mean "unreadable"


def _load_record(data: bytes) -> Any:
    """The parsed record, or ``_NOT_JSON`` for anything that is not strict UTF-8 JSON.

    ``ValueError`` covers bad UTF-8, a BOM or other syntax errors, duplicate keys, non-finite numbers and
    over-long integers; ``RecursionError`` covers over-deep nesting.
    """
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_number
        )
    except (ValueError, RecursionError):
        return _NOT_JSON


def _is_hash(value: object) -> bool:
    return type(value) is str and SHA256_HEX_PATTERN.fullmatch(value) is not None


def _closed_keys(section: object, expected: tuple[str, ...], where: str, issues: list[Issue]) -> bool:
    """True when ``section`` is an object with exactly the expected keys. Reports each problem once."""
    if not isinstance(section, dict):
        issues.append(_issue(FREEZE_FIELD_TYPE, where))
        return False
    ok = True
    for name in expected:
        if name not in section:
            issues.append(_issue(FREEZE_FIELD_MISSING, f"{where}.{name}"))
            ok = False
    if any(name not in expected for name in section):  # the unknown names are never copied
        issues.append(_issue(FREEZE_FIELD_UNKNOWN, where))
        ok = False
    return ok


def _check_record(record: object) -> list[Issue]:
    """Structural findings of the record. An empty list means the record may be trusted."""
    if not isinstance(record, dict):
        return [_issue(FREEZE_FIELD_TYPE, "freeze_record")]
    issues: list[Issue] = []
    # Missing or unknown top-level fields are reported, and everything else that can still be judged
    # is judged too, so a wrong value does not hide behind a missing neighbour.
    _closed_keys(record, _TOP_LEVEL, "freeze_record", issues)

    for name in ("record_format", "dataset_version", "frozen_on", "status", "anchoring"):
        if name in record and (type(record[name]) is not str or record[name] == ""):
            issues.append(_issue(FREEZE_FIELD_TYPE, f"freeze_record.{name}"))
    if type(record.get("record_format")) is str and record["record_format"] != RECORD_FORMAT:
        issues.append(_issue(FREEZE_RECORD_FORMAT, "freeze_record.record_format"))
    if type(record.get("status")) is str and record["status"] != FROZEN_STATUS:
        issues.append(_issue(FREEZE_STATUS, "freeze_record.status"))
    version = record.get("dataset_version")
    if type(version) is str and version and _VERSION_PATTERN.fullmatch(version) is None:
        issues.append(_issue(FREEZE_VERSION_FORMAT, "freeze_record.dataset_version"))
    frozen_on = record.get("frozen_on")
    if type(frozen_on) is str and frozen_on and not _valid_date(frozen_on):
        issues.append(_issue(FREEZE_DATE_FORMAT, "freeze_record.frozen_on"))

    if "files" in record and _closed_keys(
        record["files"], VERIFIED_FILE_PATHS + RECORDED_ONLY_PATHS, "freeze_record.files", issues
    ):
        for path in VERIFIED_FILE_PATHS + RECORDED_ONLY_PATHS:
            if not _is_hash(record["files"][path]):
                issues.append(_issue(FREEZE_HASH_FORMAT, f"freeze_record.files.{path}"))

    if "corpus" in record and _closed_keys(record["corpus"], _CORPUS_FIELDS, "freeze_record.corpus", issues):
        version_field = record["corpus"]["corpus_version"]
        if type(version_field) is not str or version_field == "":
            issues.append(_issue(FREEZE_FIELD_TYPE, "freeze_record.corpus.corpus_version"))
        if not _is_hash(record["corpus"]["canonical_manifest_hash"]):
            issues.append(_issue(FREEZE_HASH_FORMAT, "freeze_record.corpus.canonical_manifest_hash"))

    if "label_binding_hashes" in record and _closed_keys(
        record["label_binding_hashes"], _LABEL_BINDING_FIELDS, "freeze_record.label_binding_hashes", issues
    ):
        for name in _LABEL_BINDING_FIELDS:
            if not _is_hash(record["label_binding_hashes"][name]):
                issues.append(_issue(FREEZE_HASH_FORMAT, f"freeze_record.label_binding_hashes.{name}"))

    for name in _OBJECT_SECTIONS:
        if name in record and not isinstance(record[name], dict):
            issues.append(_issue(FREEZE_FIELD_TYPE, f"freeze_record.{name}"))
    limitations = record.get("limitations")
    if "limitations" in record and (
        not isinstance(limitations, list) or any(type(item) is not str for item in limitations)
    ):
        issues.append(_issue(FREEZE_FIELD_TYPE, "freeze_record.limitations"))
    return issues


def _valid_date(text: str) -> bool:
    if _DATE_PATTERN.fullmatch(text) is None:
        return False
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


# ---------- the gate ----------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_freeze_binding(
    record_bytes: bytes, dev_bytes: bytes, dataset_json_bytes: bytes, manifest_bytes: bytes
) -> FreezeBinding:
    """Verify the freeze record against the raw bytes of the files it describes.

    Pure: no file is opened. Raises ``EvaluationDataError`` with every finding when anything fails and
    ``TypeError`` when an argument is not ``bytes``. When the record itself is malformed, the files are
    not checked against it (a malformed record cannot be trusted).
    """
    for value in (record_bytes, dev_bytes, dataset_json_bytes, manifest_bytes):
        if type(value) is not bytes:
            raise TypeError("every input must be bytes")

    record = _load_record(record_bytes)
    if record is _NOT_JSON:
        raise EvaluationDataError([_issue(FREEZE_RECORD_SYNTAX, "freeze_record")])
    issues = _check_record(record)
    if issues:
        raise EvaluationDataError(issues)

    files: Mapping[str, str] = record["files"]
    actual = {DEV_PATH: _sha256(dev_bytes), DATASET_JSON_PATH: _sha256(dataset_json_bytes),
              MANIFEST_PATH: _sha256(manifest_bytes)}
    for path in VERIFIED_FILE_PATHS:
        if actual[path] != files[path]:
            issues.append(_issue(FREEZE_FILE_HASH_MISMATCH, f"freeze_record.files.{path}"))

    label_binding = _dev_label_binding(dev_bytes, issues)
    if label_binding is not None and label_binding != record["label_binding_hashes"]["dev"]:
        issues.append(_issue(FREEZE_LABEL_BINDING_MISMATCH, "freeze_record.label_binding_hashes.dev"))

    canonical_manifest = _canonical_manifest(manifest_bytes, issues)
    if canonical_manifest is not None and canonical_manifest != record["corpus"]["canonical_manifest_hash"]:
        issues.append(_issue(FREEZE_MANIFEST_HASH_MISMATCH, "freeze_record.corpus.canonical_manifest_hash"))

    if issues or label_binding is None or canonical_manifest is None:
        raise EvaluationDataError(issues)  # a None above always came with an issue
    return FreezeBinding(
        dataset_version=record["dataset_version"],
        frozen_on=record["frozen_on"],
        corpus_version=record["corpus"]["corpus_version"],
        record_sha256=_sha256(record_bytes),
        dev_file_sha256=actual[DEV_PATH],
        dataset_json_sha256=actual[DATASET_JSON_PATH],
        manifest_file_sha256=actual[MANIFEST_PATH],
        dev_label_binding_sha256=label_binding,
        canonical_manifest_sha256=canonical_manifest,
    )


def _dev_label_binding(dev_bytes: bytes, issues: list[Issue]) -> str | None:
    """The existing dataset hash of the dev questions in ``dev_bytes``; ``None`` after reporting."""
    try:
        questions = parse_questions(dev_bytes.decode("utf-8"), split="dev")
        return dataset_hash(questions)
    except (UnicodeDecodeError, EvaluationDataError):
        issues.append(_issue(FREEZE_DEV_UNPARSEABLE, DEV_PATH))
        return None


def _canonical_manifest(manifest_bytes: bytes, issues: list[Issue]) -> str | None:
    """The existing canonical manifest hash of ``manifest_bytes``; ``None`` after reporting."""
    try:
        return manifest_hash(parse_manifest(manifest_bytes.decode("utf-8")))
    except (UnicodeDecodeError, EvaluationDataError):
        issues.append(_issue(FREEZE_MANIFEST_UNPARSEABLE, MANIFEST_PATH))
        return None


_CHECK_GROUPS = (("eval",), ("eval/dataset", "eval/corpus"), READ_PATHS)


def _is_unsafe_entry(info: os.stat_result, *, is_file: bool) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    if getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return True
    return not (stat.S_ISREG(info.st_mode) if is_file else stat.S_ISDIR(info.st_mode))


def _filesystem_issues(root: Path) -> list[Issue]:
    """lstat the parent directories and the four files, without following links; never opens a file."""
    for group in _CHECK_GROUPS:
        found: list[Issue] = []
        for relative in group:
            is_file = relative in READ_PATHS
            try:
                info = os.lstat(root / relative)
            except OSError:
                if is_file:
                    found.append(_issue(FREEZE_FILE_UNREADABLE, relative))
                continue  # a missing directory is reported through the files below it
            if _is_unsafe_entry(info, is_file=is_file):
                found.append(_issue(FREEZE_PATH_UNSAFE, relative))
        if found:
            return found
    return []


def load_freeze_binding(repo_root: str | Path) -> FreezeBinding:
    """Read the four files of ``READ_PATHS`` under ``repo_root`` and run ``build_freeze_binding``.

    Only the parent directories and these four files are lstat'ed (before any read); no other path is read,
    listed, hashed or stat'ed; in particular never ``test.jsonl``.
    """
    if not isinstance(repo_root, (str, Path)):
        raise TypeError("repo_root must be a str or Path")
    root = Path(repo_root)
    unsafe = _filesystem_issues(root)
    if unsafe:
        raise EvaluationDataError(unsafe)
    contents: list[bytes] = []
    issues: list[Issue] = []
    for relative in READ_PATHS:
        try:
            contents.append((root / relative).read_bytes())
        except OSError:
            issues.append(_issue(FREEZE_FILE_UNREADABLE, relative))
    if issues:
        raise EvaluationDataError(issues)
    return build_freeze_binding(*contents)
