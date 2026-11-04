"""C2.2: PDF page mapping (app/rag/pdf_pages.py).

Pure page-span and page-assignment tests, plus real PDFs built in memory from
tests/fixtures/sample.pdf. No database access, no new fixture files.
"""
import io
import logging
import random
import re
from pathlib import Path

import pypdf
import pytest
from pypdf import PdfWriter
from pypdf.errors import PyPdfError

from app.core.processing import extract_text
from app.rag import pdf_pages
from app.rag.chunking import ChunkerConfig, ChunkSpan, chunk_text, sha256_text
from app.rag.pdf_pages import (
    NULL_PAGES,
    ChunkPages,
    PageMapping,
    PageMappingError,
    PageSpan,
    UnavailableReason,
    assign_chunk_pages,
    map_document_pages,
    page_mapping_from_texts,
    read_pdf_page_texts,
)

SAMPLE_PDF = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
LOGGER_NAME = "app.rag.pdf_pages"
TINY = ChunkerConfig(target_tokens=4, max_tokens=6, overlap_tokens=1)
LETTER = (612, 792)


# ---------- helpers ----------

def word_spans(text):
    """Fake tokenizer: one content token per whitespace-delimited word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def make_chunk(start, end, text=""):
    """A ChunkSpan with the given offsets (other fields are irrelevant here)."""
    piece = text[start:end] if text else ""
    return ChunkSpan(
        index=0,
        start=start,
        end=end,
        text=piece,
        span_sha256=sha256_text(piece),
        token_start=0,
        token_end=1,
        token_count=1,
    )


def spans_of(mapping):
    return [(s.page_number, s.start, s.end) for s in mapping.spans]


def pages_of(result):
    return [(p.page_start, p.page_end) for p in result]


def mapping_for(page_texts):
    """Available mapping for page texts joined the way processing.py joins them."""
    text = "\n".join(t.replace("\x00", "") for t in page_texts)
    mapping = page_mapping_from_texts(page_texts, text)
    assert mapping.is_available
    return text, mapping


def oracle_page(mapping, pos):
    """Independent linear-scan oracle for the page containing ``pos``."""
    found = [s.page_number for s in mapping.spans if s.start <= pos < s.end]
    assert len(found) <= 1
    return found[0] if found else None


def write_pdf(path, layout):
    """Write a PDF whose pages are copies of sample.pdf ("sample") or blank pages."""
    writer = PdfWriter()
    for item in layout:
        if item == "sample":
            writer.append(SAMPLE_PDF)
        elif item == "blank":
            writer.add_blank_page(*LETTER)
        else:
            raise ValueError(item)
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


@pytest.fixture(scope="module")
def sample_length():
    return len(extract_text(SAMPLE_PDF))


def warnings_from_module(caplog):
    return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]


# ---------- A. building page spans (pure) ----------

@pytest.mark.parametrize(
    "page_texts, extracted, expected",
    [
        (["Hello world"], "Hello world", [(1, 0, 11)]),
        (["abc", "de", "fgh"], "abc\nde\nfgh", [(1, 0, 3), (2, 4, 6), (3, 7, 10)]),
        (["", "abc"], "\nabc", [(1, 0, 0), (2, 1, 4)]),
        (["ab", "", "cd"], "ab\n\ncd", [(1, 0, 2), (2, 3, 3), (3, 4, 6)]),
        (["ab", ""], "ab\n", [(1, 0, 2), (2, 3, 3)]),
        (["ab", "", "", "cd"], "ab\n\n\ncd", [(1, 0, 2), (2, 3, 3), (3, 4, 4), (4, 5, 7)]),
        (["ab", "c\x00d\x00", "ef"], "ab\ncd\nef", [(1, 0, 2), (2, 3, 5), (3, 6, 8)]),
        (["ab", "\x00\x00", "cd"], "ab\n\ncd", [(1, 0, 2), (2, 3, 3), (3, 4, 6)]),
        (["ab\n", "\ncd"], "ab\n\n\ncd", [(1, 0, 3), (2, 4, 7)]),
        (["ab", "  ", "cd"], "ab\n  \ncd", [(1, 0, 2), (2, 3, 5), (3, 6, 8)]),
    ],
    ids=[
        "single", "three", "empty-first", "empty-middle", "empty-last",
        "consecutive-empty", "nul-shifts-offsets", "nul-only-page",
        "newlines-inside-pages", "whitespace-only-page",
    ],
)
def test_page_spans_are_built_exactly(page_texts, extracted, expected):
    mapping = page_mapping_from_texts(page_texts, extracted)
    assert mapping.is_available
    assert mapping.reason is None and mapping.detail is None
    assert mapping.page_count == len(page_texts)
    assert spans_of(mapping) == expected
    for span, raw in zip(mapping.spans, page_texts):
        assert extracted[span.start:span.end] == raw.replace("\x00", "")


def test_unicode_pages_use_code_point_offsets():
    pages = ["नमस्ते", "\U0001F44D\U0001F3FDx", "e\u0301"]
    assert [len(p) for p in pages] == [6, 3, 2]
    text, mapping = mapping_for(pages)
    assert spans_of(mapping) == [(1, 0, 6), (2, 7, 10), (3, 11, 13)]
    for span, page in zip(mapping.spans, pages):
        assert text[span.start:span.end] == page


def test_page_texts_may_be_a_tuple():
    mapping = page_mapping_from_texts(("ab", "cd"), "ab\ncd")
    assert spans_of(mapping) == [(1, 0, 2), (2, 3, 5)]


@pytest.mark.parametrize(
    "page_texts, extracted",
    [
        (["abc", "de"], "abc\ndf"),          # one character differs
        (["abc", "de"], "abc\nde\n"),        # extra trailing newline
        (["abc", "de"], "abcde"),            # missing separator
        (["abc", "de"], "abc\n\nde"),        # pages joined with two newlines
        (["abc", "de"], "abc"),              # a page missing from the text
        (["abc"], "abc\nde"),                # a page missing from the PDF
        (["a\x00bc", "de"], "a\x00bc\nde"),  # NULs still present in the text
        (["  abc", "de  "], "abc\nde"),      # stripped text
    ],
    ids=[
        "char-differs", "extra-trailing-newline", "missing-separator", "double-separator",
        "text-missing-page", "pdf-missing-page", "nul-not-removed", "stripped",
    ],
)
def test_any_difference_from_extracted_text_is_a_mismatch(page_texts, extracted):
    mapping = page_mapping_from_texts(page_texts, extracted)
    assert not mapping.is_available
    assert mapping.reason is UnavailableReason.TEXT_MISMATCH
    assert mapping.spans == ()
    assert mapping.page_count == 0


def test_no_pages_is_unavailable():
    mapping = page_mapping_from_texts([], "anything")
    assert mapping.reason is UnavailableReason.NO_PAGES
    assert mapping.spans == ()


@pytest.mark.parametrize(
    "page_texts",
    [
        "abc",
        b"abc",
        bytearray(b"abc"),
        memoryview(b"abc"),
        iter(["abc"]),
        {"abc"},
        None,
        [b"abc"],
        ["abc", None],
        ["abc", 1],
    ],
    ids=[
        "str", "bytes", "bytearray", "memoryview", "iterator", "set", "none",
        "bytes-item", "none-item", "int-item",
    ],
)
def test_page_texts_must_be_a_sequence_of_str(page_texts):
    with pytest.raises(TypeError):
        page_mapping_from_texts(page_texts, "abc")


@pytest.mark.parametrize("extracted", [None, b"abc", 1])
def test_extracted_text_must_be_a_str(extracted):
    with pytest.raises(TypeError):
        page_mapping_from_texts(["abc"], extracted)


# ---------- B. data-structure invariants ----------

@pytest.mark.parametrize(
    "page_number, start, end",
    [
        (0, 0, 0),
        (-1, 0, 0),
        (1, -1, 0),
        (1, 3, 2),
        (True, 0, 0),
        (1, False, 0),
        (1, 0, True),
        (1.0, 0, 0),
        (1, 0.0, 1),
        (1, 0, 2.0),
        ("1", 0, 0),
        (None, 0, 0),
    ],
    ids=[
        "page-zero", "page-negative", "negative-start", "reversed", "bool-page",
        "bool-start", "bool-end", "float-page", "float-start", "float-end",
        "str-page", "none-page",
    ],
)
def test_invalid_page_span_is_rejected(page_number, start, end):
    with pytest.raises(PageMappingError):
        PageSpan(page_number, start, end)


def test_valid_page_spans_including_empty():
    assert PageSpan(1, 0, 0).contains(0) is False
    span = PageSpan(3, 4, 6)
    assert [span.contains(p) for p in (3, 4, 5, 6)] == [False, True, True, False]


@pytest.mark.parametrize(
    "build",
    [
        lambda: PageMapping(spans=()),
        lambda: PageMapping.mapped([]),
        lambda: PageMapping(spans=[PageSpan(1, 0, 1)]),
        lambda: PageMapping.mapped([PageSpan(2, 0, 1)]),
        lambda: PageMapping.mapped([PageSpan(2, 0, 1), PageSpan(1, 2, 3)]),
        lambda: PageMapping.mapped([PageSpan(1, 0, 1), PageSpan(3, 2, 3)]),
        lambda: PageMapping.mapped([PageSpan(1, 1, 2)]),
        lambda: PageMapping.mapped([PageSpan(1, 0, 2), PageSpan(2, 2, 3)]),
        lambda: PageMapping.mapped([PageSpan(1, 0, 2), PageSpan(2, 4, 5)]),
        lambda: PageMapping.mapped([PageSpan(1, 0, 2), PageSpan(2, 1, 3)]),
        lambda: PageMapping(spans=((1, 0, 2),)),
        lambda: PageMapping(spans=(PageSpan(1, 0, 1),), detail="x"),
        lambda: PageMapping(spans=(PageSpan(1, 0, 1),), reason=UnavailableReason.TEXT_MISMATCH),
        lambda: PageMapping(spans=(), reason="text_mismatch"),
        lambda: PageMapping.unavailable("text_mismatch"),
        lambda: PageMapping.unavailable(None),
        lambda: PageMapping.unavailable(UnavailableReason.PDF_READ_ERROR, ""),
        lambda: PageMapping.unavailable(UnavailableReason.PDF_READ_ERROR, 42),
        lambda: PageMapping.unavailable(UnavailableReason.NOT_PDF, "x"),
        lambda: PageMapping.unavailable(UnavailableReason.FILE_MISSING, "x"),
        lambda: PageMapping.unavailable(UnavailableReason.NO_PAGES, "x"),
        lambda: PageMapping.unavailable(UnavailableReason.TEXT_MISMATCH, "x"),
        lambda: PageMapping(spans=(), reason=UnavailableReason.TEXT_MISMATCH, detail="x"),
    ],
    ids=[
        "available-without-spans", "mapped-empty", "spans-not-tuple", "starts-at-page-2",
        "out-of-order", "page-number-gap", "first-not-at-zero", "no-separator-gap",
        "two-char-gap", "overlapping", "not-a-pagespan", "available-with-detail",
        "unavailable-with-spans", "reason-plain-str", "unavailable-plain-str",
        "unavailable-without-reason", "empty-detail", "non-str-detail",
        "detail-with-not-pdf", "detail-with-file-missing", "detail-with-no-pages",
        "detail-with-text-mismatch", "detail-with-text-mismatch-direct",
    ],
)
def test_invalid_page_mapping_is_rejected(build):
    with pytest.raises(PageMappingError):
        build()


def test_valid_page_mappings():
    mapped = PageMapping.mapped([PageSpan(1, 0, 0), PageSpan(2, 1, 4), PageSpan(3, 5, 5)])
    assert mapped.is_available and mapped.page_count == 3
    gone = PageMapping.unavailable(UnavailableReason.PDF_READ_ERROR, "PdfStreamError")
    assert not gone.is_available and gone.page_count == 0
    assert gone.detail == "PdfStreamError"
    for reason in UnavailableReason:
        assert PageMapping.unavailable(reason).detail is None


# ---------- C. assigning pages to chunks (pure) ----------

THREE = ["abc", "de", "fgh"]  # "abc\nde\nfgh": pages (0,3) (4,6) (7,10)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (0, 3, (1, 1)),   # ends exactly at page 1's end: page 1, not page 2
        (4, 6, (2, 2)),   # starts exactly at page 2's start
        (7, 10, (3, 3)),  # ends at the end of the text
        (2, 5, (1, 2)),
        (0, 10, (1, 3)),
        (1, 2, (1, 1)),
    ],
)
def test_pages_come_from_the_first_and_last_character(start, end, expected):
    text, mapping = mapping_for(THREE)
    assert pages_of(assign_chunk_pages(mapping, [make_chunk(start, end, text)])) == [expected]


def test_empty_middle_page_can_only_be_inside_a_range():
    text, mapping = mapping_for(["ab", "", "cd"])
    result = assign_chunk_pages(mapping, [make_chunk(0, 6, text), make_chunk(4, 6, text)])
    assert pages_of(result) == [(1, 3), (3, 3)]


def test_empty_first_page_is_never_assigned():
    text, mapping = mapping_for(["", "abc"])
    assert pages_of(assign_chunk_pages(mapping, [make_chunk(1, 4, text)])) == [(2, 2)]


def test_result_is_chunk_pages_in_input_order():
    text, mapping = mapping_for(THREE)
    chunks = [make_chunk(7, 10, text), make_chunk(0, 3, text), make_chunk(4, 6, text)]
    result = assign_chunk_pages(mapping, chunks)
    assert type(result) is tuple
    assert all(type(p) is ChunkPages for p in result)
    assert pages_of(result) == [(3, 3), (1, 1), (2, 2)]


@pytest.mark.parametrize(
    "page_texts, start, end",
    [
        (THREE, 3, 6),           # starts on the separator after page 1
        (THREE, 0, 4),           # last character is the separator after page 1
        (THREE, 6, 10),          # starts on the separator after page 2
        (THREE, -1, 3),          # negative start
        (THREE, 0, 11),          # past the end of the text
        (THREE, 10, 11),         # entirely outside
        (THREE, 2, 2),           # empty chunk
        (THREE, 5, 2),           # reversed
        (["ab", "", "cd"], 3, 6),  # offset 3 is the empty page's start, but a separator
        (["ab", "", "cd"], 0, 4),  # last character on that separator
        (["", "abc"], 0, 4),       # offset 0 is the empty first page, but a separator
    ],
    ids=[
        "start-on-separator", "end-on-separator", "start-on-second-separator",
        "negative-start", "past-end", "outside", "empty", "reversed",
        "empty-page-start", "empty-page-end", "empty-first-page",
    ],
)
def test_chunk_edges_outside_every_page_raise(page_texts, start, end):
    _, mapping = mapping_for(page_texts)
    with pytest.raises(PageMappingError):
        assign_chunk_pages(mapping, [make_chunk(start, end)])


@pytest.mark.parametrize("start, end", [(True, 3), (0, True), (0.0, 3), (0, 3.0)])
def test_chunk_offsets_must_be_ints(start, end):
    _, mapping = mapping_for(THREE)
    with pytest.raises(PageMappingError):
        assign_chunk_pages(mapping, [make_chunk(start, end)])


@pytest.mark.parametrize("order", ["valid-first", "invalid-first"])
def test_no_partial_result_when_any_chunk_fails(order):
    text, mapping = mapping_for(THREE)
    chunks = [make_chunk(0, 3, text), make_chunk(3, 5)]
    if order == "invalid-first":
        chunks.reverse()
    with pytest.raises(PageMappingError):
        assign_chunk_pages(mapping, chunks)


def test_unavailable_mapping_gives_null_pages_without_checking_offsets():
    mapping = PageMapping.unavailable(UnavailableReason.TEXT_MISMATCH)
    chunks = [make_chunk(0, 3), make_chunk(5, 2), make_chunk(-4, 99)]
    result = assign_chunk_pages(mapping, chunks)
    assert result == (NULL_PAGES, NULL_PAGES, NULL_PAGES)
    assert pages_of(result) == [(None, None)] * 3


@pytest.mark.parametrize("available", [True, False])
def test_no_chunks_gives_empty_result(available):
    if available:
        _, mapping = mapping_for(THREE)
    else:
        mapping = PageMapping.unavailable(UnavailableReason.NOT_PDF)
    assert assign_chunk_pages(mapping, []) == ()


def test_mapping_must_be_a_page_mapping():
    with pytest.raises(TypeError):
        assign_chunk_pages(None, [])


# ---------- D. end to end with the C2.1 chunker ----------

PIECES = [
    "alpha", "beta.", "gamma!", "नमस्ते", "दुनिया।", "\U0001F44D\U0001F3FD",
    "e\u0301clair", "x\x00y", "\x00", "",
]
GAPS = [" ", "  ", "\n", "\n\n", "\t"]


def random_page(rng):
    kind = rng.random()
    if kind < 0.15:
        return ""
    if kind < 0.25:
        return rng.choice(["  ", "\n", "\n\n", " \t ", "\x00"])
    words = [rng.choice(PIECES) for _ in range(rng.randint(1, 30))]
    page = ""
    for word in words:
        page += word + rng.choice(GAPS)
    return rng.choice(["", " ", "\n"]) + page.rstrip(" ") if rng.random() < 0.3 else page


def test_random_multi_page_documents_match_the_oracle():
    rng = random.Random(20261104)
    tested = 0
    for _ in range(250):
        pages = [random_page(rng) for _ in range(rng.randint(1, 8))]
        text = "\n".join(p.replace("\x00", "") for p in pages)
        if not text.strip():
            continue
        tested += 1
        mapping = page_mapping_from_texts(pages, text)
        assert mapping.is_available
        chunks = chunk_text(text, word_spans(text), TINY)
        assert chunks
        result = assign_chunk_pages(mapping, chunks)
        assert len(result) == len(chunks)
        previous = (1, 1)
        for chunk, pages_ in zip(chunks, result):
            assert pages_.page_start == oracle_page(mapping, chunk.start)
            assert pages_.page_end == oracle_page(mapping, chunk.end - 1)
            # ck_document_chunks_pages_valid, plus the page-count bound
            assert 1 <= pages_.page_start <= pages_.page_end <= mapping.page_count
            assert pages_.page_start >= previous[0] and pages_.page_end >= previous[1]
            previous = (pages_.page_start, pages_.page_end)
        assert assign_chunk_pages(page_mapping_from_texts(pages, text), chunks) == result
    assert tested >= 200


def test_chunks_crossing_an_empty_page_never_use_it_as_an_edge():
    pages = ["one two three four five", "", "six seven eight nine ten"]
    text, mapping = mapping_for(pages)
    chunks = chunk_text(text, word_spans(text), TINY)
    result = pages_of(assign_chunk_pages(mapping, chunks))
    assert all(2 not in pair for pair in result)
    assert (1, 3) in result


# ---------- E. real PDFs ----------

def test_sample_pdf_maps_to_one_page(sample_length):
    text = extract_text(SAMPLE_PDF)
    mapping = map_document_pages(SAMPLE_PDF, text)
    assert spans_of(mapping) == [(1, 0, sample_length)]
    chunks = chunk_text(text, word_spans(text), TINY)
    assert len(chunks) > 1
    assert set(pages_of(assign_chunk_pages(mapping, chunks))) == {(1, 1)}


@pytest.mark.parametrize(
    "layout, expected",
    [
        (["sample"], lambda n: [(1, 0, n)]),
        (["sample", "sample"], lambda n: [(1, 0, n), (2, n + 1, 2 * n + 1)]),
        (
            ["sample", "blank", "sample"],
            lambda n: [(1, 0, n), (2, n + 1, n + 1), (3, n + 2, 2 * n + 2)],
        ),
        (["blank", "sample"], lambda n: [(1, 0, 0), (2, 1, n + 1)]),
    ],
    ids=["one", "two", "blank-middle", "blank-first"],
)
def test_pdf_page_spans_stay_in_step_with_processing(tmp_path, sample_length, layout, expected):
    path = write_pdf(tmp_path / "doc.pdf", layout)
    text = extract_text(path)

    raw = read_pdf_page_texts(path)
    assert len(raw) == len(layout)
    assert "\n".join(t.replace("\x00", "") for t in raw) == text

    mapping = map_document_pages(path, text)
    assert spans_of(mapping) == expected(sample_length)

    chunks = chunk_text(text, word_spans(text), TINY)
    result = pages_of(assign_chunk_pages(mapping, chunks))
    blank_pages = {k + 1 for k, item in enumerate(layout) if item == "blank"}
    for page_start, page_end in result:
        assert page_start not in blank_pages and page_end not in blank_pages
        assert 1 <= page_start <= page_end <= len(layout)


def test_nul_and_none_page_texts_stay_in_step_with_processing(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "four.pdf", ["sample"] * 4)
    per_page = {0: "", 1: "x\x00y", 2: None, 3: "z"}
    monkeypatch.setattr(
        pypdf.PageObject,
        "extract_text",
        lambda self, *args, **kwargs: per_page[self.page_number],
    )
    assert extract_text(path) == "\nxy\n\nz"
    assert read_pdf_page_texts(path) == ["", "x\x00y", "", "z"]
    mapping = map_document_pages(path, extract_text(path))
    assert spans_of(mapping) == [(1, 0, 0), (2, 1, 3), (3, 4, 4), (4, 5, 6)]


# ---------- F. failure handling ----------

def _forbid_pdf_reader(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("PdfReader must not be called for this document")

    monkeypatch.setattr(pdf_pages, "PdfReader", forbidden)


@pytest.mark.parametrize("suffix", [".txt", ".md", ".docx", ".TXT", ".DocX", ""])
@pytest.mark.parametrize("exists", [True, False])
def test_non_pdf_documents_are_never_opened(tmp_path, monkeypatch, suffix, exists):
    _forbid_pdf_reader(monkeypatch)
    path = tmp_path / f"stored{suffix}"
    if exists:
        path.write_bytes(SAMPLE_PDF.read_bytes())  # PDF bytes, but not a .pdf file
    mapping = map_document_pages(path, "anything")
    assert mapping.reason is UnavailableReason.NOT_PDF
    assert mapping.spans == ()


def test_pdf_suffix_is_case_insensitive(tmp_path, sample_length):
    path = tmp_path / "STORED.PDF"
    path.write_bytes(SAMPLE_PDF.read_bytes())
    mapping = map_document_pages(path, extract_text(path))
    assert spans_of(mapping) == [(1, 0, sample_length)]


def test_no_stored_file_is_file_missing(monkeypatch):
    _forbid_pdf_reader(monkeypatch)
    assert map_document_pages(None, "text").reason is UnavailableReason.FILE_MISSING


def test_missing_pdf_is_file_missing(tmp_path, monkeypatch):
    _forbid_pdf_reader(monkeypatch)
    mapping = map_document_pages(tmp_path / "gone.pdf", "text")
    assert mapping.reason is UnavailableReason.FILE_MISSING


def test_directory_named_like_a_pdf_is_file_missing(tmp_path, monkeypatch):
    _forbid_pdf_reader(monkeypatch)
    (tmp_path / "d.pdf").mkdir()
    assert map_document_pages(tmp_path / "d.pdf", "text").reason is UnavailableReason.FILE_MISSING


def _encrypted_pdf_bytes():
    writer = PdfWriter()
    writer.append(SAMPLE_PDF)
    writer.encrypt("pw")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        b"not a pdf",
        b"",
        SAMPLE_PDF.read_bytes()[:500],
        _encrypted_pdf_bytes(),
    ],
    ids=["junk", "zero-bytes", "truncated", "encrypted"],
)
def test_unreadable_pdf_is_pdf_read_error(tmp_path, content):
    path = tmp_path / "bad.pdf"
    path.write_bytes(content)
    mapping = map_document_pages(path, "text")
    assert mapping.reason is UnavailableReason.PDF_READ_ERROR
    assert mapping.spans == ()
    # detail is the pypdf exception class name, never its message
    assert issubclass(getattr(pypdf.errors, mapping.detail), PyPdfError)


def test_os_error_while_reading_is_pdf_read_error(tmp_path, monkeypatch):
    path = tmp_path / "locked.pdf"
    path.write_bytes(SAMPLE_PDF.read_bytes())

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(pdf_pages, "PdfReader", denied)
    mapping = map_document_pages(path, "text")
    assert mapping.reason is UnavailableReason.PDF_READ_ERROR
    assert mapping.detail == "PermissionError"


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyError("k"), TypeError("t")])
def test_unexpected_errors_propagate(tmp_path, monkeypatch, error):
    path = tmp_path / "doc.pdf"
    path.write_bytes(SAMPLE_PDF.read_bytes())

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(pdf_pages, "PdfReader", broken)
    with pytest.raises(type(error)):
        map_document_pages(path, "text")


def test_pdf_with_other_text_is_a_mismatch(tmp_path):
    path = write_pdf(tmp_path / "doc.pdf", ["sample", "sample"])
    mapping = map_document_pages(path, extract_text(SAMPLE_PDF))
    assert mapping.reason is UnavailableReason.TEXT_MISMATCH
    assert mapping.spans == ()


# ---------- G. logging and determinism ----------

@pytest.mark.parametrize("case", ["no-stored-file", "missing", "unreadable", "mismatch"])
def test_unavailable_mapping_logs_one_warning(tmp_path, caplog, case):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    if case == "no-stored-file":
        path, text = None, "text"
    elif case == "missing":
        path, text = tmp_path / "gone.pdf", "text"
    elif case == "unreadable":
        path, text = tmp_path / "bad.pdf", "text"
        path.write_bytes(b"not a pdf")
    else:
        path, text = SAMPLE_PDF, "SECRET-MARKER document text"
    mapping = map_document_pages(path, text)

    records = warnings_from_module(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert mapping.reason.value in message
    assert "SECRET-MARKER" not in caplog.text
    assert str(tmp_path) not in message


def test_non_pdf_logs_no_warning(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    map_document_pages(tmp_path / "notes.txt", "text")
    assert warnings_from_module(caplog) == []
    assert any(r.name == LOGGER_NAME and r.levelno == logging.DEBUG for r in caplog.records)


def test_available_mapping_logs_no_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    assert map_document_pages(SAMPLE_PDF, extract_text(SAMPLE_PDF)).is_available
    assert warnings_from_module(caplog) == []


def test_mapping_is_deterministic(tmp_path):
    path = write_pdf(tmp_path / "doc.pdf", ["blank", "sample", "sample"])
    text = extract_text(path)
    assert map_document_pages(path, text) == map_document_pages(path, text)