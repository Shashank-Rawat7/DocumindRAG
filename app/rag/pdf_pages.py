"""PDF page mapping for chunks (Component 12 / C2.2).

Rebuilds where each physical PDF page lies inside ``documents.extracted_text``
and assigns ``page_start``/``page_end`` to chunks. Extraction itself
(app/core/processing.py) is not changed; this module re-reads the stored PDF
at indexing time and must stay in step with it:

    processing.extract_text():  "\\n".join(page.extract_text() or "" for each page)
                                then .replace("\\x00", "")

Rules:
- Page numbers are 1-based physical page indexes (page 1 is the first page in
  the file). PDF page labels are ignored.
- Offsets are Python code-point offsets into extracted_text, half-open
  [start, end), the same units as app.rag.chunking.ChunkSpan.
- Page k covers exactly its NUL-free extracted text. The single "\\n" joining
  two pages belongs to no page. An empty page has a zero-length span and so
  contains no position.
- A page contains position ``pos`` iff ``start <= pos < end``.
- The mapping is accepted only when the rebuilt text equals extracted_text
  exactly; otherwise it is unavailable and every chunk gets NULL pages.
- page_start is the page containing chunk.start; page_end is the page
  containing chunk.end - 1. Pages are assigned to all chunks or to none.
- Mapping unavailability is reported as a PageMapping with a reason and never
  raised. PageMappingError means a violated invariant (a bug) and is raised.
"""
from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.rag.chunking import ChunkSpan

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n"

# Expected failures when re-reading a stored PDF. Anything else propagates.
#   PyPdfError: malformed / truncated / empty file (PdfReadError, PdfStreamError,
#               EmptyFileError, ParseError), encrypted (FileNotDecryptedError), limits.
#   OSError:    file vanished, permission denied, path is a directory, I/O error.
PDF_READ_ERRORS: tuple[type[BaseException], ...] = (PyPdfError, OSError)


class PageMappingError(Exception):
    """A violated page-mapping invariant (a bug). Never means "unavailable"."""


class UnavailableReason(str, Enum):
    NOT_PDF = "not_pdf"
    FILE_MISSING = "file_missing"
    PDF_READ_ERROR = "pdf_read_error"
    NO_PAGES = "no_pages"
    TEXT_MISMATCH = "text_mismatch"


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise PageMappingError(f"{name} must be an int, got {value!r}")


@dataclass(frozen=True, slots=True)
class PageSpan:
    """One physical page: extracted_text[start:end] is its NUL-free text."""

    page_number: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _require_int("page_number", self.page_number)
        _require_int("start", self.start)
        _require_int("end", self.end)
        if self.page_number < 1:
            raise PageMappingError(f"page_number must be >= 1, got {self.page_number}")
        if not 0 <= self.start <= self.end:
            raise PageMappingError(
                f"page {self.page_number}: span ({self.start}, {self.end}) "
                "must satisfy 0 <= start <= end"
            )

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end


@dataclass(frozen=True, slots=True)
class PageMapping:
    """Either available (spans, no reason) or unavailable (reason, no spans).

    Build with PageMapping.mapped(...) or PageMapping.unavailable(...).
    """

    spans: tuple[PageSpan, ...]
    reason: UnavailableReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if type(self.spans) is not tuple:
            raise PageMappingError("spans must be a tuple")
        if self.reason is not None and not isinstance(self.reason, UnavailableReason):
            raise PageMappingError(f"reason must be an UnavailableReason, got {self.reason!r}")
        if self.detail is not None and (type(self.detail) is not str or not self.detail):
            raise PageMappingError("detail must be None or a non-empty str")

        if self.reason is not None:
            if self.detail is not None and self.reason is not UnavailableReason.PDF_READ_ERROR:
                raise PageMappingError(
                    f"detail is only allowed for {UnavailableReason.PDF_READ_ERROR.value}, "
                    f"not {self.reason.value}"
                )
            if self.spans:
                raise PageMappingError("an unavailable mapping must not have spans")
            return

        if self.detail is not None:
            raise PageMappingError("an available mapping must not have a detail")
        if not self.spans:
            raise PageMappingError("an available mapping must have at least one span")
        expected_start = 0
        for k, span in enumerate(self.spans):
            if type(span) is not PageSpan:
                raise PageMappingError(f"spans[{k}] must be a PageSpan, got {span!r}")
            if span.page_number != k + 1:
                raise PageMappingError(
                    f"spans[{k}] has page_number {span.page_number}; pages must be 1..n in order"
                )
            if span.start != expected_start:
                raise PageMappingError(
                    f"page {span.page_number} starts at {span.start}, expected {expected_start} "
                    "(first page at 0, then one separator after each page)"
                )
            expected_start = span.end + len(PAGE_SEPARATOR)

    @classmethod
    def mapped(cls, spans: Sequence[PageSpan]) -> PageMapping:
        return cls(spans=tuple(spans))

    @classmethod
    def unavailable(cls, reason: UnavailableReason, detail: str | None = None) -> PageMapping:
        if not isinstance(reason, UnavailableReason):
            raise PageMappingError(f"reason must be an UnavailableReason, got {reason!r}")
        return cls(spans=(), reason=reason, detail=detail)

    @property
    def is_available(self) -> bool:
        return self.reason is None

    @property
    def page_count(self) -> int:
        return len(self.spans)


@dataclass(frozen=True, slots=True)
class ChunkPages:
    """Values for document_chunks.page_start / page_end (both None or both set)."""

    page_start: int | None
    page_end: int | None


NULL_PAGES = ChunkPages(None, None)


# ---------- building page spans ----------

def read_pdf_page_texts(path: Path) -> list[str]:
    """Per-page text exactly as processing.extract_text() obtains it.

    NULs are NOT removed and pages are NOT joined here. pypdf and OS errors
    propagate to the caller.
    """
    reader = PdfReader(path)
    return [(page.extract_text() or "") for page in reader.pages]


def page_mapping_from_texts(page_texts: Sequence[str], extracted_text: str) -> PageMapping:
    """Page spans for ``extracted_text`` built from raw per-page texts (pure)."""
    if type(extracted_text) is not str:
        raise TypeError("extracted_text must be a str")
    if isinstance(page_texts, (str, bytes, bytearray, memoryview)) or not isinstance(
        page_texts, Sequence
    ):
        raise TypeError("page_texts must be a sequence of str (one item per page)")
    for k, page_text in enumerate(page_texts):
        if type(page_text) is not str:
            raise TypeError(f"page_texts[{k}] must be a str, got {type(page_text).__name__}")

    if len(page_texts) == 0:
        return PageMapping.unavailable(UnavailableReason.NO_PAGES)

    cleaned = [page_text.replace("\x00", "") for page_text in page_texts]
    if PAGE_SEPARATOR.join(cleaned) != extracted_text:
        return PageMapping.unavailable(UnavailableReason.TEXT_MISMATCH)

    spans: list[PageSpan] = []
    cursor = 0
    for k, page_text in enumerate(cleaned):
        span = PageSpan(page_number=k + 1, start=cursor, end=cursor + len(page_text))
        spans.append(span)
        cursor = span.end + len(PAGE_SEPARATOR)

    # Invariant checks (cannot fail after the equality check above; kept on purpose).
    for k, span in enumerate(spans):
        if extracted_text[span.start:span.end] != cleaned[k]:
            raise PageMappingError(f"page {span.page_number}: slice does not equal its page text")
        if k < len(spans) - 1:
            separator = extracted_text[span.end:span.end + len(PAGE_SEPARATOR)]
            if separator != PAGE_SEPARATOR:
                raise PageMappingError(
                    f"expected a newline separator after page {span.page_number} "
                    f"at offset {span.end}, found {separator!r}"
                )
    if spans[-1].end != len(extracted_text):
        raise PageMappingError("the last page does not end at the end of extracted_text")

    return PageMapping.mapped(spans)


def map_document_pages(path: Path | None, extracted_text: str) -> PageMapping:
    """Page mapping for a stored document; unavailability is returned, not raised.

    Non-PDF documents (by the stored file's suffix, as in processing.py) are
    never opened.
    """
    if path is None:
        return _unavailable(None, UnavailableReason.FILE_MISSING)
    if path.suffix.lower() != ".pdf":
        logger.debug("PDF page mapping skipped for %s: not a PDF", path.name)
        return PageMapping.unavailable(UnavailableReason.NOT_PDF)
    if not path.is_file():
        return _unavailable(path, UnavailableReason.FILE_MISSING)

    try:
        page_texts = read_pdf_page_texts(path)
    except PDF_READ_ERRORS as exc:
        return _unavailable(path, UnavailableReason.PDF_READ_ERROR, type(exc).__name__)

    mapping = page_mapping_from_texts(page_texts, extracted_text)
    if not mapping.is_available:
        _log_unavailable(path, mapping.reason, mapping.detail)
    return mapping


def _unavailable(
    path: Path | None, reason: UnavailableReason, detail: str | None = None
) -> PageMapping:
    _log_unavailable(path, reason, detail)
    return PageMapping.unavailable(reason, detail)


def _log_unavailable(path: Path | None, reason: UnavailableReason, detail: str | None) -> None:
    logger.warning(
        "PDF page mapping unavailable for %s: %s%s",
        path.name if path is not None else "<no stored file>",
        reason.value,
        f" ({detail})" if detail else "",
    )


# ---------- assigning pages to chunks ----------

def assign_chunk_pages(mapping: PageMapping, chunks: Sequence[ChunkSpan]) -> tuple[ChunkPages, ...]:
    """page_start/page_end for every chunk, in order, or PageMappingError.

    Unavailable mapping: NULL pages for every chunk, offsets not inspected.
    Available mapping: every chunk is mapped or nothing is returned.
    """
    if not isinstance(mapping, PageMapping):
        raise TypeError("mapping must be a PageMapping")
    if not mapping.is_available:
        return tuple(NULL_PAGES for _ in chunks)

    # Zero-length (empty) pages contain no position, so they are not candidates.
    non_empty = [span for span in mapping.spans if span.start < span.end]
    starts = [span.start for span in non_empty]
    text_length = mapping.spans[-1].end

    def page_containing(pos: int, what: str, chunk_index: int) -> int:
        i = bisect_right(starts, pos) - 1  # candidate only; containment decides
        if i >= 0 and non_empty[i].contains(pos):
            return non_empty[i].page_number
        where = "outside the text" if pos >= text_length else "on a page separator"
        raise PageMappingError(
            f"chunk {chunk_index}: {what} offset {pos} is {where}, not inside any page"
        )

    result: list[ChunkPages] = []
    for k, chunk in enumerate(chunks):
        start, end = chunk.start, chunk.end
        _require_int(f"chunk {k} start", start)
        _require_int(f"chunk {k} end", end)
        if not 0 <= start < end:
            raise PageMappingError(f"chunk {k}: span ({start}, {end}) must satisfy 0 <= start < end")
        page_start = page_containing(start, "start", k)
        page_end = page_containing(end - 1, "last character", k)
        if not 1 <= page_start <= page_end <= mapping.page_count:
            raise PageMappingError(
                f"chunk {k}: pages ({page_start}, {page_end}) violate "
                f"1 <= page_start <= page_end <= {mapping.page_count}"
            )
        result.append(ChunkPages(page_start, page_end))
    return tuple(result)