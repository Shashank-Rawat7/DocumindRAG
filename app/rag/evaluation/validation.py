"""Corpus manifest validation and ``require_valid`` (E1.2b).

``validate_manifest`` checks the VALUE rules of a manifest: versions, enums, patterns,
non-blank metadata, uniqueness, hash and length consistency, the draft and ingested
stage rules, and format-specific extraction metadata. It never needs document text and
never touches a database. It is type-defensive, because dataclasses built by hand are not
guaranteed to hold the right types: a wrong type is reported and the checks that depend
on it are skipped, so one mistake is reported once and nothing crashes.

Issues are returned in a fixed order: top-level fields, then each document in order (its
fields in a fixed order), then the cross-document duplicate checks in document order.
Messages are fixed text and locations are schema-defined paths with list indexes, so no
issue contains document content. Dataset-level rules are a later checkpoint (E1.2d).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from app.rag.evaluation.parsing import EXTRACTION_LIBRARIES
from app.rag.evaluation.schema import (
    CORPUS_KEY_PATTERN,
    DOCUMENT_FORMATS,
    LANGUAGES,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_STAGES,
    SEVERITY_ERROR,
    SHA256_HEX_PATTERN,
    EvaluationDataError,
    Extraction,
    Issue,
    Manifest,
    ManifestDocument,
)

# processing_ref: a 40-character lowercase hex git commit id (E1.1 contract).
PROCESSING_REF_PATTERN = re.compile(r"[0-9a-f]{40}")

MANIFEST_FIELD_TYPE = "MANIFEST_FIELD_TYPE"
MANIFEST_SCHEMA_VERSION_MISMATCH = "MANIFEST_SCHEMA_VERSION_MISMATCH"
MANIFEST_STAGE = "MANIFEST_STAGE"
MANIFEST_BLANK = "MANIFEST_BLANK"
MANIFEST_NO_DOCUMENTS = "MANIFEST_NO_DOCUMENTS"
DOC_FIELD_TYPE = "DOC_FIELD_TYPE"
DOC_CORPUS_KEY = "DOC_CORPUS_KEY"
DOC_BLANK = "DOC_BLANK"
DOC_FORMAT = "DOC_FORMAT"
DOC_LANGUAGE = "DOC_LANGUAGE"
DOC_SHA256_FORMAT = "DOC_SHA256_FORMAT"
DOC_LENGTH = "DOC_LENGTH"
DOC_HASH_LENGTH_PAIR = "DOC_HASH_LENGTH_PAIR"
DOC_STAGE_INCOMPLETE = "DOC_STAGE_INCOMPLETE"
DOC_DUPLICATE_KEY = "DOC_DUPLICATE_KEY"
DOC_DUPLICATE_TEXT_HASH = "DOC_DUPLICATE_TEXT_HASH"
EXTRACTION_PROCESSING_REF = "EXTRACTION_PROCESSING_REF"
EXTRACTION_BLANK = "EXTRACTION_BLANK"
EXTRACTION_FIELD_MISSING = "EXTRACTION_FIELD_MISSING"
EXTRACTION_FIELD_INAPPLICABLE = "EXTRACTION_FIELD_INAPPLICABLE"

_MESSAGES = {
    MANIFEST_FIELD_TYPE: "a manifest field has the wrong type",
    MANIFEST_SCHEMA_VERSION_MISMATCH: "schema_version is not the supported manifest schema version",
    MANIFEST_STAGE: "stage must be 'draft' or 'ingested'",
    MANIFEST_BLANK: "a manifest field must not be blank",
    MANIFEST_NO_DOCUMENTS: "the manifest must list at least one document",
    DOC_FIELD_TYPE: "a document field has the wrong type",
    DOC_CORPUS_KEY: "corpus_key does not match the corpus key pattern",
    DOC_BLANK: "a document field must not be blank",
    DOC_FORMAT: "format must be one of the supported document formats",
    DOC_LANGUAGE: "language must be one of the supported languages",
    DOC_SHA256_FORMAT: "a hash must be 64 lowercase hex characters",
    DOC_LENGTH: "extracted_text_length must be an integer of at least 1",
    DOC_HASH_LENGTH_PAIR: "extracted_text_sha256 and extracted_text_length must both be null or both be set",
    DOC_STAGE_INCOMPLETE: "an ingested manifest requires this field to be set",
    DOC_DUPLICATE_KEY: "corpus_key is already used by an earlier document",
    DOC_DUPLICATE_TEXT_HASH: "extracted_text_sha256 is already used by an earlier document",
    EXTRACTION_PROCESSING_REF: "processing_ref must be 40 lowercase hex characters",
    EXTRACTION_BLANK: "an extraction field must not be blank",
    EXTRACTION_FIELD_MISSING: "an extraction field required for this format is missing",
    EXTRACTION_FIELD_INAPPLICABLE: "an extraction field does not apply to this document format",
}


def require_valid(issues: Sequence[Issue]) -> None:
    """Raise ``EvaluationDataError`` if any issue has error severity; warnings alone pass.

    The raised error carries every issue (warnings included) in ``.issues``. Its message
    is only the count, as for every ``EvaluationDataError``.
    """
    items = tuple(issues)
    if not all(isinstance(item, Issue) for item in items):
        raise TypeError("issues must contain only Issue objects")
    if any(item.severity == SEVERITY_ERROR for item in items):
        raise EvaluationDataError(items)


def _blank(value: str) -> bool:
    return value.strip() == ""


def _is_plain_int(value: object) -> bool:
    return type(value) is int


class _Collector:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, code: str, location: str) -> None:
        self.issues.append(Issue(code, location, _MESSAGES[code]))

    def text(self, value: object, location: str, type_code: str, blank_code: str) -> bool:
        """True when ``value`` is a non-blank str; otherwise the right issue is added."""
        if type(value) is not str:
            self.add(type_code, location)
            return False
        if _blank(value):
            self.add(blank_code, location)
            return False
        return True

    def sha256(self, value: object, location: str) -> bool:
        if type(value) is not str:
            self.add(DOC_FIELD_TYPE, location)
            return False
        if SHA256_HEX_PATTERN.fullmatch(value) is None:
            self.add(DOC_SHA256_FORMAT, location)
            return False
        return True


def _check_extraction(c: _Collector, extraction: object, document_format: object, location: str) -> None:
    if not isinstance(extraction, Extraction):
        c.add(DOC_FIELD_TYPE, location)
        return
    ref_location = f"{location}.processing_ref"
    if type(extraction.processing_ref) is not str:
        c.add(DOC_FIELD_TYPE, ref_location)
    elif PROCESSING_REF_PATTERN.fullmatch(extraction.processing_ref) is None:
        c.add(EXTRACTION_PROCESSING_REF, ref_location)
    c.text(extraction.python, f"{location}.python", DOC_FIELD_TYPE, EXTRACTION_BLANK)

    if type(document_format) is not str or document_format not in EXTRACTION_LIBRARIES:
        return  # the format problem is reported on the document; applicability is undecidable
    applicable = EXTRACTION_LIBRARIES[document_format]
    for name in ("pypdf", "python_docx", "lxml"):
        value = getattr(extraction, name)
        where = f"{location}.{name}"
        if name in applicable:
            if value is None:
                c.add(EXTRACTION_FIELD_MISSING, where)
            else:
                c.text(value, where, DOC_FIELD_TYPE, EXTRACTION_BLANK)
        elif value is not None:
            c.add(EXTRACTION_FIELD_INAPPLICABLE, where)


def _check_document(c: _Collector, document: object, stage: object, location: str) -> None:
    if not isinstance(document, ManifestDocument):
        c.add(DOC_FIELD_TYPE, location)
        return
    key_location = f"{location}.corpus_key"
    if type(document.corpus_key) is not str:
        c.add(DOC_FIELD_TYPE, key_location)
    elif CORPUS_KEY_PATTERN.fullmatch(document.corpus_key) is None:
        c.add(DOC_CORPUS_KEY, key_location)
    for name in ("title", "publisher", "source_url", "license", "license_url"):
        c.text(getattr(document, name), f"{location}.{name}", DOC_FIELD_TYPE, DOC_BLANK)

    format_location = f"{location}.format"
    if type(document.format) is not str:
        c.add(DOC_FIELD_TYPE, format_location)
    elif document.format not in DOCUMENT_FORMATS:
        c.add(DOC_FORMAT, format_location)
    language_location = f"{location}.language"
    if type(document.language) is not str:
        c.add(DOC_FIELD_TYPE, language_location)
    elif document.language not in LANGUAGES:
        c.add(DOC_LANGUAGE, language_location)

    c.sha256(document.source_file_sha256, f"{location}.source_file_sha256")
    if type(document.redistributable) is not bool:
        c.add(DOC_FIELD_TYPE, f"{location}.redistributable")

    hash_location = f"{location}.extracted_text_sha256"
    length_location = f"{location}.extracted_text_length"
    if document.extracted_text_sha256 is not None:
        c.sha256(document.extracted_text_sha256, hash_location)
    if document.extracted_text_length is not None:
        if not _is_plain_int(document.extracted_text_length):
            c.add(DOC_FIELD_TYPE, length_location)
        elif document.extracted_text_length < 1:
            c.add(DOC_LENGTH, length_location)
    if (document.extracted_text_sha256 is None) != (document.extracted_text_length is None):
        c.add(DOC_HASH_LENGTH_PAIR, location)

    if document.extraction is not None:
        _check_extraction(c, document.extraction, document.format, f"{location}.extraction")

    if stage == "ingested":
        if document.extracted_text_sha256 is None:
            c.add(DOC_STAGE_INCOMPLETE, hash_location)
        if document.extracted_text_length is None:
            c.add(DOC_STAGE_INCOMPLETE, length_location)
        if document.extraction is None:
            c.add(DOC_STAGE_INCOMPLETE, f"{location}.extraction")


def validate_manifest(manifest: Manifest) -> list[Issue]:
    """Every value-rule problem of ``manifest``, in a fixed order. Empty means valid."""
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    c = _Collector()

    if type(manifest.schema_version) is not str:
        c.add(MANIFEST_FIELD_TYPE, "manifest.schema_version")
    elif manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        c.add(MANIFEST_SCHEMA_VERSION_MISMATCH, "manifest.schema_version")
    c.text(manifest.corpus_version, "manifest.corpus_version", MANIFEST_FIELD_TYPE, MANIFEST_BLANK)
    if type(manifest.stage) is not str:
        c.add(MANIFEST_FIELD_TYPE, "manifest.stage")
    elif manifest.stage not in MANIFEST_STAGES:
        c.add(MANIFEST_STAGE, "manifest.stage")

    if type(manifest.documents) is not tuple:
        c.add(MANIFEST_FIELD_TYPE, "manifest.documents")
        return c.issues
    if not manifest.documents:
        c.add(MANIFEST_NO_DOCUMENTS, "manifest.documents")

    for index, document in enumerate(manifest.documents):
        _check_document(c, document, manifest.stage, f"manifest.documents[{index}]")

    # Cross-document uniqueness: report each repeat at the later document.
    seen_keys: set[str] = set()
    seen_hashes: set[str] = set()
    for index, document in enumerate(manifest.documents):
        if not isinstance(document, ManifestDocument):
            continue
        key = document.corpus_key
        if type(key) is str and CORPUS_KEY_PATTERN.fullmatch(key) is not None:
            if key in seen_keys:
                c.add(DOC_DUPLICATE_KEY, f"manifest.documents[{index}].corpus_key")
            seen_keys.add(key)
        digest = document.extracted_text_sha256
        if type(digest) is str and SHA256_HEX_PATTERN.fullmatch(digest) is not None:
            if digest in seen_hashes:
                c.add(DOC_DUPLICATE_TEXT_HASH, f"manifest.documents[{index}].extracted_text_sha256")
            seen_hashes.add(digest)
    return c.issues
