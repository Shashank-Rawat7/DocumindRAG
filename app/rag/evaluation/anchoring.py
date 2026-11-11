"""Source-text verification and span anchoring (E1.2c).

Both functions take the document text as an explicit argument. They read no file, no
database and no service, and they never change a span: anchoring only CHECKS the stored
offsets and quote against the stored extracted text. A mismatch is a failure, never a cue
to search for the quote elsewhere or to adjust an offset.

* ``verify_source_text`` is the fail-closed gate for one document: the text must be a
  non-empty ``str`` whose SHA-256 equals the manifest's ``extracted_text_sha256`` and, when
  recorded, whose length equals ``extracted_text_length``.
* ``resolve_question`` checks every span of one question against the texts it is given.

Text is compared exactly. Offsets are Python code-point offsets (never UTF-16 units),
half-open: ``text[start:end]``. Nothing is normalised (no NFC, NFD, NFKC or NFKD), a
ligature such as U+FB01 is never expanded, and CR LF is whatever the stored text holds.

Both functions RETURN issues (the E1.2 convention) and raise only ``TypeError`` for a
wrongly typed argument. An empty list means every check passed; a caller turns issues
into an exception with ``validation.require_valid``. No issue contains text, a quote, a
corpus key or any other document content: messages are fixed and locations are
positional. A span edge inside a grapheme cluster is a WARNING, never an error.
"""
from __future__ import annotations

from collections.abc import Mapping

from app.rag.chunking import sha256_text
from app.rag.evaluation import schema
from app.rag.evaluation.schema import (
    SEVERITY_WARNING,
    SHA256_HEX_PATTERN,
    EvidenceUnit,
    Issue,
    ManifestDocument,
    Question,
    Span,
)
from app.rag.text_boundaries import grapheme_boundaries

# ---------- issue codes ----------

SOURCE_TEXT_TYPE = "SOURCE_TEXT_TYPE"
SOURCE_TEXT_EMPTY = "SOURCE_TEXT_EMPTY"
SOURCE_TEXT_ENCODING = "SOURCE_TEXT_ENCODING"
SOURCE_ENTRY_HASH_MISSING = "SOURCE_ENTRY_HASH_MISSING"
SOURCE_ENTRY_HASH_FORMAT = "SOURCE_ENTRY_HASH_FORMAT"
SOURCE_ENTRY_LENGTH_TYPE = "SOURCE_ENTRY_LENGTH_TYPE"
SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
SOURCE_LENGTH_MISMATCH = "SOURCE_LENGTH_MISMATCH"

ANCHOR_UNITS_TYPE = "ANCHOR_UNITS_TYPE"
ANCHOR_UNIT_TYPE = "ANCHOR_UNIT_TYPE"
ANCHOR_SPANS_TYPE = "ANCHOR_SPANS_TYPE"
ANCHOR_SPAN_TYPE = "ANCHOR_SPAN_TYPE"
ANCHOR_TEXT_MISSING = "ANCHOR_TEXT_MISSING"
ANCHOR_TEXT_INVALID = "ANCHOR_TEXT_INVALID"
ANCHOR_SOURCE_HASH_MISMATCH = "ANCHOR_SOURCE_HASH_MISMATCH"
ANCHOR_END_BEYOND_TEXT = "ANCHOR_END_BEYOND_TEXT"
ANCHOR_QUOTE_MISMATCH = "ANCHOR_QUOTE_MISMATCH"
ANCHOR_START_INSIDE_GRAPHEME = "ANCHOR_START_INSIDE_GRAPHEME"
ANCHOR_END_INSIDE_GRAPHEME = "ANCHOR_END_INSIDE_GRAPHEME"

_MESSAGES = {
    SOURCE_TEXT_TYPE: "the source text must be a str",
    SOURCE_TEXT_EMPTY: "the source text must not be empty",
    SOURCE_TEXT_ENCODING: "the source text cannot be encoded as UTF-8",
    SOURCE_ENTRY_HASH_MISSING: "the manifest entry has no extracted_text_sha256 to verify against",
    SOURCE_ENTRY_HASH_FORMAT: "extracted_text_sha256 must be 64 lowercase hex characters",
    SOURCE_ENTRY_LENGTH_TYPE: "extracted_text_length must be an integer or null",
    SOURCE_HASH_MISMATCH: "the source text does not match extracted_text_sha256",
    SOURCE_LENGTH_MISMATCH: "the source text length does not match extracted_text_length",
    ANCHOR_UNITS_TYPE: "units must be a tuple",
    ANCHOR_UNIT_TYPE: "a unit must be an EvidenceUnit",
    ANCHOR_SPANS_TYPE: "spans must be a tuple",
    ANCHOR_SPAN_TYPE: "a span must be a Span",
    ANCHOR_TEXT_MISSING: "no source text was supplied for this span's corpus key",
    ANCHOR_TEXT_INVALID: "the supplied source text is not a usable non-empty str",
    ANCHOR_SOURCE_HASH_MISMATCH: "the source text does not match the span's source_sha256",
    ANCHOR_END_BEYOND_TEXT: "the span ends after the end of the source text",
    ANCHOR_QUOTE_MISMATCH: "the source text at the span's offsets differs from the quote",
    ANCHOR_START_INSIDE_GRAPHEME: "the span starts inside a grapheme cluster",
    ANCHOR_END_INSIDE_GRAPHEME: "the span ends inside a grapheme cluster",
}

# Structural problems after which an anchoring check would be meaningless. Quote-level
# problems (empty, length, encoding, quote hash) do not block: the text check stands alone.
_BLOCKING_STRUCTURE = frozenset(
    {
        schema.SPAN_FIELD_TYPE,
        schema.SPAN_CORPUS_KEY,
        schema.SPAN_SOURCE_SHA256_FORMAT,
        schema.SPAN_OFFSET_RANGE,
    }
)


def _error(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code])


def _warning(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_WARNING)


# ---------- source-text verification ----------

def verify_source_text(entry: ManifestDocument, text: str) -> list[Issue]:
    """Check ``text`` against one manifest entry. An empty list means it is the recorded text.

    Fails closed: a missing or malformed ``extracted_text_sha256`` is an error, so an entry
    that cannot be verified is never accepted. The recorded length is checked only when it
    is set. Issues come in a fixed order: text type, empty, then the entry's hash and length
    metadata, then the encoding, hash and length comparisons. A problem that makes a later
    check meaningless suppresses that check, so one mistake is reported once.
    """
    if not isinstance(entry, ManifestDocument):
        raise TypeError("entry must be a ManifestDocument")
    issues: list[Issue] = []

    text_is_str = type(text) is str
    text_usable = False
    if not text_is_str:
        issues.append(_error(SOURCE_TEXT_TYPE, "source_text"))
    elif text == "":
        issues.append(_error(SOURCE_TEXT_EMPTY, "source_text"))
    else:
        text_usable = True

    recorded_hash = entry.extracted_text_sha256
    hash_usable = False
    if recorded_hash is None:
        issues.append(_error(SOURCE_ENTRY_HASH_MISSING, "entry.extracted_text_sha256"))
    elif type(recorded_hash) is not str or SHA256_HEX_PATTERN.fullmatch(recorded_hash) is None:
        issues.append(_error(SOURCE_ENTRY_HASH_FORMAT, "entry.extracted_text_sha256"))
    else:
        hash_usable = True

    recorded_length = entry.extracted_text_length
    length_usable = recorded_length is not None
    if length_usable and type(recorded_length) is not int:
        issues.append(_error(SOURCE_ENTRY_LENGTH_TYPE, "entry.extracted_text_length"))
        length_usable = False

    if text_usable:
        try:
            digest = sha256_text(text)  # the project's one hash convention: SHA-256 of the UTF-8 bytes
        except UnicodeEncodeError:  # for example a lone surrogate
            issues.append(_error(SOURCE_TEXT_ENCODING, "source_text"))
        else:
            if hash_usable and digest != recorded_hash:
                issues.append(_error(SOURCE_HASH_MISMATCH, "source_text"))
        if length_usable and len(text) != recorded_length:
            issues.append(_error(SOURCE_LENGTH_MISMATCH, "source_text"))
    return issues


# ---------- question span anchoring ----------

class _Sources:
    """The supplied texts, each hashed and segmented at most once per call."""

    def __init__(self, texts: Mapping[str, str]) -> None:
        self._texts = texts
        self._checked: dict[str, tuple[str | None, str | None]] = {}
        self._boundaries: dict[str, bytearray] = {}

    def check(self, key: str) -> tuple[str | None, str | None]:
        """(error code, digest). The code is None when the text exists and can be hashed."""
        if key not in self._checked:
            if key not in self._texts:
                result: tuple[str | None, str | None] = (ANCHOR_TEXT_MISSING, None)
            else:
                text = self._texts[key]
                if type(text) is not str or text == "":
                    result = (ANCHOR_TEXT_INVALID, None)
                else:
                    try:
                        result = (None, sha256_text(text))
                    except UnicodeEncodeError:
                        result = (ANCHOR_TEXT_INVALID, None)
            self._checked[key] = result
        return self._checked[key]

    def text(self, key: str) -> str:
        return self._texts[key]

    def boundaries(self, key: str) -> bytearray:
        if key not in self._boundaries:
            self._boundaries[key] = grapheme_boundaries(self._texts[key])
        return self._boundaries[key]


def _anchor_span(sources: _Sources, span: Span, location: str) -> list[Issue]:
    """Anchoring checks for one structurally sound span, in a fixed order."""
    code, digest = sources.check(span.corpus_key)
    if code is not None:
        return [_error(code, f"{location}.corpus_key")]
    if digest != span.source_sha256:
        return [_error(ANCHOR_SOURCE_HASH_MISMATCH, f"{location}.source_sha256")]
    text = sources.text(span.corpus_key)
    if span.end > len(text):
        return [_error(ANCHOR_END_BEYOND_TEXT, f"{location}.end")]
    if text[span.start:span.end] != span.quote:
        return [_error(ANCHOR_QUOTE_MISMATCH, f"{location}.quote")]
    # Anchored. Edges inside a grapheme cluster are worth a look, not a rejection.
    found: list[Issue] = []
    boundaries = sources.boundaries(span.corpus_key)
    if not boundaries[span.start]:
        found.append(_warning(ANCHOR_START_INSIDE_GRAPHEME, f"{location}.start"))
    if not boundaries[span.end]:
        found.append(_warning(ANCHOR_END_INSIDE_GRAPHEME, f"{location}.end"))
    return found


def resolve_question(q: Question, texts: Mapping[str, str], *, location: str = "question") -> list[Issue]:
    """Check every span of ``q`` against ``texts`` (corpus key -> extracted text).

    Returns the issues found, or an empty list when every span is anchored. Spans and
    offsets are never modified or relocated. For each span, in unit then span order:
    the structural issues of ``schema.validate_span``, then, if the structure allows it,
    the anchoring checks: text present and usable, text hash equal to the span's
    ``source_sha256``, ``end`` within the text, and ``text[start:end] == quote`` exactly;
    only a span that passes all of those can add the grapheme-edge warnings. Each
    document's text is hashed once per call.

    ``location`` is a caller-chosen prefix and must not contain document text. A question
    without units gives no issues; whether it should have units is a dataset rule, not
    an anchoring rule.
    """
    if not isinstance(q, Question):
        raise TypeError("q must be a Question")
    if not isinstance(texts, Mapping):
        raise TypeError("texts must be a Mapping")
    if type(location) is not str:
        raise TypeError("location must be a str")

    issues: list[Issue] = []
    if type(q.units) is not tuple:
        return [_error(ANCHOR_UNITS_TYPE, f"{location}.units")]
    sources = _Sources(texts)
    for u, unit in enumerate(q.units):
        unit_location = f"{location}.units[{u}]"
        if not isinstance(unit, EvidenceUnit):
            issues.append(_error(ANCHOR_UNIT_TYPE, unit_location))
            continue
        if type(unit.spans) is not tuple:
            issues.append(_error(ANCHOR_SPANS_TYPE, f"{unit_location}.spans"))
            continue
        for s, span in enumerate(unit.spans):
            span_location = f"{unit_location}.spans[{s}]"
            if not isinstance(span, Span):
                issues.append(_error(ANCHOR_SPAN_TYPE, span_location))
                continue
            structural = schema.validate_span(span, location=span_location)
            issues.extend(structural)
            if not any(item.code in _BLOCKING_STRUCTURE for item in structural):
                issues.extend(_anchor_span(sources, span, span_location))
    return issues
