"""Strict JSON and JSONL parsing, and canonical hashing, for evaluation data (E1.2b).

``parse_manifest`` and ``parse_questions`` turn text into the frozen E1.2a dataclasses.
They check structure and types only: closed schemas, required keys, plain types (a
``bool`` is never an ``int``), no coercion, no discarded fields. Value rules (enums,
patterns, lengths, dates, spans against text, cross-record rules) belong to
``validation`` and to later checkpoints, so a parsed object is not yet known to be valid.

Input is ``str``. A leading BOM, malformed JSON, duplicate object keys, ``NaN`` and
``Infinity`` (also ``1e999``), over-deep nesting and over-long integers are rejected.
Every independently detectable problem is collected; a parser returns complete objects
or raises ``EvaluationDataError`` carrying all issues, never a partial result. A BOM is
reported and then removed so the rest of the input is still inspected. Duplicate keys and
non-finite numbers are all collected from one syntactically valid document, in the order
the JSON text is read, followed by its structural issues. Text that is not readable JSON
(syntax error, over-deep nesting, over-long integer) yields that one issue only, because
nothing after it can be trusted.

No issue contains user-controlled content. Messages are fixed text. Locations are
schema-defined paths, line numbers and syntax columns. Field names, values, duplicate
keys and ``json`` exception text are never copied.

The hash functions are total over the dataclasses but do not validate or detect
duplicates; a fixed secondary sort key keeps them deterministic regardless.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from types import MappingProxyType

from app.rag.chunking import sha256_text
from app.rag.evaluation.schema import (
    DOCUMENT_FORMATS,
    SPLITS,
    EvaluationDataError,
    EvidenceUnit,
    Extraction,
    Issue,
    Manifest,
    ManifestDocument,
    Question,
    RetrievalAssistance,
    SemanticReview,
    Span,
)

# Which extraction library keys apply to which document format (E1.1 contract).
EXTRACTION_LIBRARIES = MappingProxyType(
    {
        "pdf": ("pypdf",),
        "docx": ("python_docx", "lxml"),
        "txt": (),
        "md": (),
    }
)
_ALL_LIBRARIES = ("pypdf", "python_docx", "lxml")

# ---------- issue codes ----------

PARSE_BOM = "PARSE_BOM"
PARSE_SYNTAX = "PARSE_SYNTAX"
PARSE_DUPLICATE_KEY = "PARSE_DUPLICATE_KEY"
PARSE_NON_FINITE = "PARSE_NON_FINITE"
PARSE_TOO_DEEP = "PARSE_TOO_DEEP"
PARSE_NUMBER = "PARSE_NUMBER"
PARSE_NOT_OBJECT = "PARSE_NOT_OBJECT"
PARSE_UNKNOWN_FIELD = "PARSE_UNKNOWN_FIELD"
PARSE_MISSING_FIELD = "PARSE_MISSING_FIELD"
PARSE_INAPPLICABLE_FIELD = "PARSE_INAPPLICABLE_FIELD"
PARSE_FIELD_TYPE = "PARSE_FIELD_TYPE"
PARSE_BLANK_LINE = "PARSE_BLANK_LINE"
PARSE_SPLIT_MISMATCH = "PARSE_SPLIT_MISMATCH"
HASH_NOT_SERIALISABLE = "HASH_NOT_SERIALISABLE"

_MESSAGES = {
    PARSE_BOM: "input must not start with a byte order mark",
    PARSE_SYNTAX: "input is not valid JSON",
    PARSE_DUPLICATE_KEY: "a JSON object has a duplicate key",
    PARSE_NON_FINITE: "NaN and infinite numbers are not allowed",
    PARSE_TOO_DEEP: "input is nested too deeply",
    PARSE_NUMBER: "a number is too long to read",
    PARSE_NOT_OBJECT: "a JSON object is required here",
    PARSE_UNKNOWN_FIELD: "an object has a field that is not part of the schema",
    PARSE_MISSING_FIELD: "a required field is missing",
    PARSE_INAPPLICABLE_FIELD: "a field does not apply to this document format",
    PARSE_FIELD_TYPE: "a field has the wrong type",
    PARSE_BLANK_LINE: "a blank line is not a record",
    PARSE_SPLIT_MISMATCH: "the record's split differs from the file's split",
    HASH_NOT_SERIALISABLE: "the data cannot be serialised for hashing",
}

# ---------- canonical JSON and hashing ----------


def canonical_json(value: object) -> str:
    """The one canonical serialisation: sorted keys, compact, ASCII only, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _record(value: object) -> object:
    """Dataclasses become dicts of ALL their fields (nulls included), tuples become lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _record(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [_record(item) for item in value]
    return value


def _ordered(records: list, key_field: str) -> list[tuple[str, object]]:
    """(canonical text, record) pairs ordered by ``key_field``, then by canonical text."""
    keyed = [(record[key_field], canonical_json(record), index) for index, record in enumerate(records)]
    keyed.sort()
    return [(text, records[index]) for _, text, index in keyed]


def manifest_hash(manifest: Manifest) -> str:
    """SHA-256 of the canonical manifest, documents ordered by corpus key.

    Every manifest field is included, with explicit nulls. The hash covers the parsed
    content, so key order and line endings in a file do not affect it.
    """
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    try:
        top = _record(manifest)
        documents = top["documents"]
        if not isinstance(documents, list):
            raise TypeError
        top["documents"] = [record for _, record in _ordered(documents, "corpus_key")]
        return sha256_text(canonical_json(top))
    except (TypeError, ValueError, KeyError, AttributeError):
        raise EvaluationDataError(
            [Issue(HASH_NOT_SERIALISABLE, "manifest", _MESSAGES[HASH_NOT_SERIALISABLE])]
        ) from None


def dataset_hash(questions: Sequence[Question]) -> str:
    """SHA-256 of one split's canonical question records, ordered by question id.

    Records are joined with a newline. The ``split`` field is part of each record. No
    questions hash as the empty byte string.
    """
    if isinstance(questions, (str, bytes)):
        raise TypeError("questions must be a sequence of Question")
    try:
        records = [_record(question) for question in questions]
        texts = [text for text, _ in _ordered(records, "question_id")]
        return sha256_text("\n".join(texts))
    except (TypeError, ValueError, KeyError, AttributeError):
        raise EvaluationDataError(
            [Issue(HASH_NOT_SERIALISABLE, "dataset", _MESSAGES[HASH_NOT_SERIALISABLE])]
        ) from None


# ---------- strict JSON loading ----------


class _Poison:
    """Stands in for a non-finite number that was already reported, so it is not reported twice."""


_POISON = _Poison()


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code])


def _load(text: str, where: str, *, multiline: bool) -> tuple[object, list[Issue], bool]:
    """Parse one JSON document. Returns (value, issues, readable).

    Duplicate keys and non-finite numbers do not stop the parse: each is recorded and
    reading continues, so all of them are returned together (the first value of a
    duplicated key is kept; a non-finite number becomes ``_POISON``). If the text is not
    readable JSON, ``readable`` is False and ``issues`` holds only that one problem.
    """
    found: list[Issue] = []

    def pairs(items: list) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                found.append(_issue(PARSE_DUPLICATE_KEY, where))
            else:
                result[key] = value
        return result

    def constant(_name: str) -> _Poison:
        found.append(_issue(PARSE_NON_FINITE, where))
        return _POISON

    def number(literal: str) -> float | _Poison:
        value = float(literal)
        if math.isfinite(value):
            return value
        found.append(_issue(PARSE_NON_FINITE, where))  # "1e999" overflows without using parse_constant
        return _POISON

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    except json.JSONDecodeError as error:
        position = f"line {error.lineno} column {error.colno}" if multiline else f"column {error.colno}"
        return None, [_issue(PARSE_SYNTAX, f"{where} {position}")], False
    except RecursionError:
        return None, [_issue(PARSE_TOO_DEEP, where)], False
    except ValueError:  # an integer longer than the interpreter's digit limit
        return None, [_issue(PARSE_NUMBER, where)], False
    return value, found, True


# ---------- structural reading ----------

_ABSENT = object()
_BAD = object()


class _Reader:
    """Collects issues while reading raw JSON values into typed fields."""

    def __init__(self) -> None:
        self.issues: list[Issue] = []
        self.skipped = 0  # values left unread because they were already reported as non-finite

    def mark(self) -> tuple[int, int]:
        return len(self.issues), self.skipped

    def changed(self, mark: tuple[int, int]) -> bool:
        """True if an issue was added or a value was skipped since ``mark``: nothing may be built."""
        return self.mark() != mark

    def add(self, code: str, location: str) -> None:
        self.issues.append(_issue(code, location))

    def obj(self, raw: object, location: str, required: Sequence[str], optional: Sequence[str]):
        """``raw`` as a closed object: unknown keys, then missing keys, are reported."""
        if raw is _POISON:
            self.skipped += 1
            return None  # already reported as a non-finite number
        if type(raw) is not dict:
            self.add(PARSE_NOT_OBJECT, location)
            return None
        allowed = set(required) | set(optional)
        for key in raw:
            if key not in allowed:
                self.add(PARSE_UNKNOWN_FIELD, location)  # the key itself is never reported
        for name in required:
            if name not in raw:
                self.add(PARSE_MISSING_FIELD, f"{location}.{name}")
        return raw

    def field(self, raw: dict, name: str, location: str, kind: type, *, nullable: bool = False):
        """The value of ``raw[name]`` if it has exactly type ``kind`` (or is an allowed null)."""
        if name not in raw:
            return _ABSENT
        value = raw[name]
        if value is _POISON:
            self.skipped += 1
            return _BAD  # already reported as a non-finite number
        if value is None and nullable:
            return None
        if type(value) is kind:
            return value
        self.add(PARSE_FIELD_TYPE, f"{location}.{name}")
        return _BAD


def _read_span(reader: _Reader, raw: object, location: str) -> Span | None:
    before = reader.mark()
    names = ("corpus_key", "source_sha256", "start", "end", "quote", "quote_sha256")
    if reader.obj(raw, location, names, ()) is None:
        return None
    kinds = {"start": int, "end": int}
    values = {name: reader.field(raw, name, location, kinds.get(name, str)) for name in names}
    if reader.changed(before):
        return None
    return Span(**values)


def _read_unit(reader: _Reader, raw: object, location: str) -> EvidenceUnit | None:
    before = reader.mark()
    if reader.obj(raw, location, ("unit_id", "spans"), ()) is None:
        return None
    unit_id = reader.field(raw, "unit_id", location, str)
    raw_spans = reader.field(raw, "spans", location, list)
    spans = []
    if raw_spans is not _ABSENT and raw_spans is not _BAD:
        for index, item in enumerate(raw_spans):
            spans.append(_read_span(reader, item, f"{location}.spans[{index}]"))
    if reader.changed(before):
        return None
    return EvidenceUnit(unit_id=unit_id, spans=tuple(spans))


def _read_semantic_review(reader: _Reader, raw: object, location: str) -> SemanticReview | None:
    before = reader.mark()
    names = ("reviewer", "date", "result")
    if reader.obj(raw, location, names, ()) is None:
        return None
    values = {name: reader.field(raw, name, location, str) for name in names}
    if reader.changed(before):
        return None
    return SemanticReview(**values)


def _read_assistance(reader: _Reader, raw: object, location: str) -> RetrievalAssistance | None:
    before = reader.mark()
    names = ("date", "method", "run_ref", "description")
    if reader.obj(raw, location, names, ()) is None:
        return None
    nullable = {"method", "run_ref"}
    values = {name: reader.field(raw, name, location, str, nullable=name in nullable) for name in names}
    if reader.changed(before):
        return None
    return RetrievalAssistance(**values)


# (name, kind, nullable). ``kind`` strings name the nested readers. Order = dataclass order.
_QUESTION_FIELDS = (
    ("question_id", str, False),
    ("split", str, False),
    ("answerable", bool, False),
    ("question", str, False),
    ("query_language", str, False),
    ("author_type", str, False),
    ("hindi_reviewed", bool, False),
    ("hindi_reviewer", str, True),
    ("notes", str, True),
    ("expected_answer", str, True),
    ("units", "units", False),
    ("source_language", str, True),
    ("difficulty", str, True),
    ("close_distractor", bool, True),
    ("unanswerable_type", str, True),
    ("semantic_review", "review", True),
    ("retrieval_assistance", "assistance", True),
)
_QUESTION_REQUIRED = (
    "question_id", "split", "answerable", "question", "query_language", "author_type", "hindi_reviewed",
)
_QUESTION_OPTIONAL = tuple(name for name, _, _ in _QUESTION_FIELDS if name not in _QUESTION_REQUIRED)


def _read_question(reader: _Reader, raw: object, location: str, split: str) -> Question | None:
    before = reader.mark()
    if reader.obj(raw, location, _QUESTION_REQUIRED, _QUESTION_OPTIONAL) is None:
        return None
    values: dict = {}
    for name, kind, nullable in _QUESTION_FIELDS:
        if name not in raw:
            continue
        if kind == "units":
            items = reader.field(raw, name, location, list)
            if items is _BAD:
                continue
            units = [_read_unit(reader, item, f"{location}.units[{i}]") for i, item in enumerate(items)]
            values[name] = tuple(units)
            continue
        if kind in ("review", "assistance"):
            if raw[name] is None:
                values[name] = None
                continue
            reader_fn = _read_semantic_review if kind == "review" else _read_assistance
            values[name] = reader_fn(reader, raw[name], f"{location}.{name}")
            continue
        value = reader.field(raw, name, location, kind, nullable=nullable)
        if value is _BAD:
            continue
        if name == "split" and value != split:
            reader.add(PARSE_SPLIT_MISMATCH, f"{location}.split")
        values[name] = value
    if reader.changed(before):
        return None
    return Question(**values)


def _read_extraction(reader: _Reader, raw: object, location: str, document_format: object) -> Extraction | None:
    before = reader.mark()
    known = type(document_format) is str and document_format in EXTRACTION_LIBRARIES
    applicable = EXTRACTION_LIBRARIES[document_format] if known else ()
    required = ("processing_ref", "python") + applicable
    optional = tuple(name for name in _ALL_LIBRARIES if name not in applicable)
    if reader.obj(raw, location, required, optional) is None:
        return None
    if known:
        for name in _ALL_LIBRARIES:
            if name not in applicable and name in raw:
                reader.add(PARSE_INAPPLICABLE_FIELD, f"{location}.{name}")
    values = {
        "processing_ref": reader.field(raw, "processing_ref", location, str),
        "python": reader.field(raw, "python", location, str),
    }
    for name in _ALL_LIBRARIES:
        if name in raw:
            values[name] = reader.field(raw, name, location, str, nullable=True)
    if reader.changed(before):
        return None
    return Extraction(**values)


_DOCUMENT_REQUIRED = (
    "corpus_key", "title", "publisher", "source_url", "license", "license_url", "format", "language",
    "source_file_sha256", "extracted_text_sha256", "extracted_text_length", "extraction", "redistributable",
)


def _read_document(reader: _Reader, raw: object, location: str) -> ManifestDocument | None:
    before = reader.mark()
    if reader.obj(raw, location, _DOCUMENT_REQUIRED, ()) is None:
        return None
    values: dict = {}
    for name in _DOCUMENT_REQUIRED:
        if name == "extraction":
            continue
        if name == "extracted_text_sha256":
            values[name] = reader.field(raw, name, location, str, nullable=True)
        elif name == "extracted_text_length":
            values[name] = reader.field(raw, name, location, int, nullable=True)
        elif name == "redistributable":
            values[name] = reader.field(raw, name, location, bool)
        else:
            values[name] = reader.field(raw, name, location, str)
    if "extraction" in raw:
        if raw["extraction"] is None:
            values["extraction"] = None
        else:
            values["extraction"] = _read_extraction(
                reader, raw["extraction"], f"{location}.extraction", values.get("format")
            )
    if reader.changed(before):
        return None
    return ManifestDocument(**values)


# ---------- public parsers ----------

_BOM = "﻿"


def parse_manifest(text: str) -> Manifest:
    """Read a corpus manifest. Raises ``EvaluationDataError`` listing every structural problem."""
    if type(text) is not str:
        raise TypeError("text must be a str")
    reader = _Reader()
    if text.startswith(_BOM):
        reader.add(PARSE_BOM, "manifest")
        text = text[len(_BOM):]
    value, strict_issues, readable = _load(text, "manifest", multiline=True)
    reader.issues.extend(strict_issues)
    if not readable:
        raise EvaluationDataError(reader.issues)
    names = ("schema_version", "corpus_version", "stage", "documents")
    if reader.obj(value, "manifest", names, ()) is None:
        raise EvaluationDataError(reader.issues)
    schema_version = reader.field(value, "schema_version", "manifest", str)
    corpus_version = reader.field(value, "corpus_version", "manifest", str)
    stage = reader.field(value, "stage", "manifest", str)
    raw_documents = reader.field(value, "documents", "manifest", list)
    documents = []
    if raw_documents is not _ABSENT and raw_documents is not _BAD:
        for index, item in enumerate(raw_documents):
            documents.append(_read_document(reader, item, f"manifest.documents[{index}]"))
    if reader.issues:
        raise EvaluationDataError(reader.issues)
    return Manifest(
        schema_version=schema_version, corpus_version=corpus_version, stage=stage, documents=tuple(documents)
    )


def parse_questions(text: str, *, split: str) -> tuple[Question, ...]:
    """Read one JSONL question file. ``split`` is the split the file belongs to.

    Records are split on LF only, never on U+2028 or other separators inside strings.
    A CR before the LF is JSON whitespace, so CRLF files parse exactly like LF files.
    Blank lines are errors except a final line terminator.
    An empty file gives an empty tuple.
    """
    if type(text) is not str:
        raise TypeError("text must be a str")
    if split not in SPLITS:
        raise ValueError("split must be 'dev' or 'test'")

    reader = _Reader()
    if text.startswith(_BOM):
        reader.add(PARSE_BOM, "line 1")
        text = text[len(_BOM):]
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the final line terminator
    questions = []
    for number, line in enumerate(lines, start=1):
        location = f"line {number}"
        if line.strip(" \t\r") == "":
            reader.add(PARSE_BLANK_LINE, location)
            continue
        value, strict_issues, readable = _load(line, location, multiline=False)
        reader.issues.extend(strict_issues)
        if not readable:
            continue
        question = _read_question(reader, value, location, split)
        if question is not None:
            questions.append(question)
    if reader.issues:
        raise EvaluationDataError(reader.issues)
    return tuple(questions)
