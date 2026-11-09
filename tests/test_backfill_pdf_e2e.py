"""Real-PDF end-to-end validation of the indexing backfill (C5.5 / CP6c.3).

A deterministic three-page PDF is built in memory, uploaded through the real API, extracted
by the real processing code, and then indexed by the real backfill CLI (--execute). The real
test database, wiring, request_indexing, run_indexing, PDF page mapping, chunker and publish
step are used. Only the embedding MODEL is replaced by a fake (same shape as the CP6c.2
fakes), and autouse guards fail the test if the real model is ever requested.

Expectations come from independent oracles, not from the code under test: the designed word
sequence of every page, and page markers located with str.index in the extracted text.

Helpers are local to this file, as in the other backfill test modules.
"""
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader
from sqlalchemy import select, text

from app.core import storage
from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.rag import embedder as embedder_module
from app.rag import indexing_runner, pdf_pages, token_offsets
from app.rag.backfill_cli import EXIT_OK, NOTICE_MODEL_OK, Runtime, main
from app.rag.backfill_report import DRY_RUN_FOOTER
from app.rag.chunking import DEFAULT_CHUNKER_CONFIG
from app.rag.embedder import EMBEDDING_DIM
from app.rag.indexing import EMBED_GROUP_SIZE, INDEX_BUILD_VERSION
from app.rag.text_boundaries import is_ignorable

UPLOAD = "/api/v1/documents/upload"
DOCUMENTS = "/api/v1/documents"
SECRET = "s3cr3t-p4ssw0rd-XYZ"
TEST_DATABASE = engine.url.database
FAKE_EMBEDDING_ID = "test/fake@" + "0" * 40
RUNBOOK = Path(__file__).resolve().parent.parent / "docs" / "backfill_indexing_runbook.md"

PAGES = 3
WORDS_PER_LINE = 12
BODY_LINES = 20  # 240 body words + 5 marker words per page: 735 words in all, above max_tokens (400)


# ---------- guards: the real model must never be touched ----------

@pytest.fixture(autouse=True)
def forbid_the_real_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the real embedding model must not be requested")

    monkeypatch.setattr(embedder_module.Embedder, "_ensure_loaded", boom)
    monkeypatch.setattr(embedder_module, "load_pinned_model", boom)
    monkeypatch.setattr(embedder_module, "get_embedder", boom)
    monkeypatch.setattr(indexing_runner, "get_embedder", boom)


# ---------- the fixture PDF ----------

def page_lines(number):
    """The designed text of page ``number``: a start marker, body lines, an end marker."""
    lines = [f"PAGE{number}START fixture page {number}"]
    for line in range(BODY_LINES):
        first = line * WORDS_PER_LINE
        lines.append(" ".join(f"p{number}w{first + k + 1:03d}" for k in range(WORDS_PER_LINE)))
    lines.append(f"PAGE{number}END")
    return lines


def designed_words(number):
    return " ".join(page_lines(number)).split()


LAYOUT_TEXT = [page_lines(1), page_lines(2), page_lines(3)]
LAYOUT_GAP = [page_lines(1), [], page_lines(3)]  # page 2 is blank


def pdf_bytes(pages):
    """A minimal, fully deterministic PDF: no dates, no ids, no randomness.

    ``pages`` is a list of pages; each page is a list of text lines ([] is a blank page).
    Objects: 1 catalog, 2 page tree, 3 font, then a page and its content stream per page.
    """
    for lines in pages:
        for line in lines:
            assert line.isascii() and not set("()\\") & set(line)
    count = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for i, lines in enumerate(pages):
        bodies.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
            ).encode("ascii")
        )
        if lines:
            moves = [f"72 750 Td ({lines[0]}) Tj"] + [f"0 -30 Td ({line}) Tj" for line in lines[1:]]
            stream = ("BT /F1 10 Tf " + " ".join(moves) + " ET").encode("ascii")
        else:
            stream = b""
        bodies.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode("ascii") + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def reference_text(path):
    """The documented extraction contract (app/core/processing.py), re-stated independently."""
    return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages).replace("\x00", "")


# ---------- independent page oracle ----------

def page_starts(extracted, numbers):
    """{page number: offset of its start marker}, strictly increasing."""
    starts = {}
    for number in numbers:
        marker = f"PAGE{number}START"
        assert extracted.count(marker) == 1, marker
        starts[number] = extracted.index(marker)
    assert list(starts.values()) == sorted(starts.values())
    return starts


def page_at(starts, position):
    return max(number for number, start in starts.items() if start <= position)


def page_regions(extracted, starts):
    ordered = sorted(starts.items(), key=lambda item: item[1])
    regions = {}
    for index, (number, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(extracted)
        regions[number] = extracted[start:end]
    return regions


# ---------- fake embedder (same shape as the CP6c.2 tests) ----------

def word_spans(value):
    spans, start = [], None
    for position, character in enumerate(value + " "):
        if is_ignorable(character):
            if start is not None:
                spans.append((start, position))
                start = None
        elif start is None:
            start = position
    return spans


def fake_vector(value):
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big") % EMBEDDING_DIM] = 1.0
    return vector


class FakeEmbedder:
    embedding_id = FAKE_EMBEDDING_ID

    def __init__(self):
        self.calls = []

    def signature_fields(self):
        self.calls.append(("signature_fields", 0))
        return {"model_id": "test/fake", "embedding_dim": EMBEDDING_DIM}

    def source_units(self, value):
        self.calls.append(("source_units", len(value)))
        spans = word_spans(value)
        return token_offsets.map_token_offsets(value, spans, [value[start:end] for start, end in spans])

    def embed_passages(self, texts):
        self.calls.append(("embed_passages", len(texts)))
        return np.stack([fake_vector(item) for item in texts]).astype(np.float32)


# ---------- database and CLI helpers ----------

def db_snapshot():
    with engine.connect() as connection:
        documents = [tuple(r) for r in connection.execute(text("SELECT id, xmin::text FROM documents ORDER BY id"))]
        chunks = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
    return documents, chunks


def document_row(document_id):
    with engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT processing_status, processing_error, stored_name, content_type, size_bytes, "
                "extracted_text, source_version, source_sha256, indexing_status, indexing_error, "
                "indexing_attempt, indexing_lease_expires_at IS NOT NULL AS leased, "
                "indexed_source_version, indexed_signature, indexed_embedding_id, indexed_at "
                "FROM documents WHERE id = :id"
            ),
            {"id": document_id},
        ).one()


def chunks_of(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(
                DocumentChunk.chunk_index, DocumentChunk.start_offset, DocumentChunk.end_offset,
                DocumentChunk.page_start, DocumentChunk.page_end, DocumentChunk.text,
                DocumentChunk.span_sha256, DocumentChunk.embedding,
            )
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / ".env").write_text("", encoding="utf-8")  # created by the test, empty
    uploads = root / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(storage, "UPLOAD_ROOT", uploads)
    assert uploads.is_relative_to(tmp_path.resolve())        # uploads can only land in the temp folder
    assert (engine.url.database or "").endswith("_test")     # and only the test database is used
    return root


def make_runtime(repo, **overrides):
    values = dict(
        engine=engine,
        database_url=f"postgresql://documind_user:{SECRET}@localhost:5434/{TEST_DATABASE}",
        environment="development",
        environment_explicit=True,
        upload_dir="uploads",
        upload_root=repo / "uploads",
        embedding_id=FAKE_EMBEDDING_ID,
        embedder=FakeEmbedder(),
        session_factory=SessionLocal,
    )
    values.update(overrides)
    return Runtime(**values)


def run_cli(repo, embedder, *argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        list(argv),
        runtime_loader=lambda: make_runtime(repo, embedder=embedder),
        environ={},
        cwd=repo,
        repo_root=repo,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def execute_args(*extra):
    return ("--execute", "--database", TEST_DATABASE, *extra)


def value(text_, label):
    match = re.search(rf"^\s+{re.escape(label)}\s*:\s*(.+)$", text_, re.M)
    assert match, f"no summary row {label!r}"
    return match.group(1).strip()


def upload_pdf(client, headers, data, name="fixture.pdf"):
    response = client.post(UPLOAD, files={"file": (name, data, "application/pdf")}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


@dataclass
class Indexed:
    document_id: int
    pdf: bytes
    path: Path
    text: str
    embedder: FakeEmbedder
    code: int
    out: str
    err: str


@pytest.fixture
def indexed(repo, client, auth_headers):
    """Upload the fixture PDF (real extraction), then index it with the real CLI."""
    data = pdf_bytes(LAYOUT_TEXT)
    document_id = upload_pdf(client, auth_headers, data)
    row = document_row(document_id)
    embedder = FakeEmbedder()
    code, out, err = run_cli(repo, embedder, *execute_args())
    return Indexed(document_id, data, repo / "uploads" / row.stored_name, row.extracted_text, embedder, code, out, err)


# ======================================================================
# A. The fixture and the real extraction
# ======================================================================

def test_the_fixture_pdf_is_deterministic_and_has_exactly_the_designed_pages(tmp_path):
    assert pdf_bytes(LAYOUT_TEXT) == pdf_bytes(LAYOUT_TEXT)
    path = tmp_path / "fixture.pdf"
    path.write_bytes(pdf_bytes(LAYOUT_TEXT))
    reader = PdfReader(path)
    assert len(reader.pages) == PAGES
    for number, page in enumerate(reader.pages, start=1):
        assert page.extract_text().split() == designed_words(number)  # whitespace-agnostic, otherwise exact


def test_upload_stores_the_pdf_and_extraction_joins_the_pages_with_one_newline(repo, client, auth_headers):
    data = pdf_bytes(LAYOUT_TEXT)
    document_id = upload_pdf(client, auth_headers, data)

    row = document_row(document_id)
    stored = repo / "uploads" / row.stored_name
    assert row.stored_name.endswith(".pdf") and stored.read_bytes() == data
    assert (row.processing_status, row.processing_error) == ("completed", None)
    assert (row.content_type, row.size_bytes) == ("application/pdf", len(data))

    extracted = row.extracted_text
    assert extracted == reference_text(stored)
    assert "\x00" not in extracted
    starts = page_starts(extracted, range(1, PAGES + 1))
    for number in range(2, PAGES + 1):
        assert extracted[starts[number] - 1] == "\n"  # a page begins right after the joining newline
    regions = page_regions(extracted, starts)
    for number in range(1, PAGES + 1):
        assert regions[number].split() == designed_words(number)  # each page holds exactly its own words


def test_the_page_mapping_agrees_with_the_extracted_text(repo, client, auth_headers):
    document_id = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_TEXT))
    row = document_row(document_id)
    mapping = pdf_pages.map_document_pages(repo / "uploads" / row.stored_name, row.extracted_text)
    assert mapping.is_available and mapping.page_count == PAGES
    starts = page_starts(row.extracted_text, range(1, PAGES + 1))
    for span in mapping.spans:
        assert span.start <= starts[span.page_number] < span.end
        assert row.extracted_text[span.start:span.end].split() == designed_words(span.page_number)


# ======================================================================
# B. Dry run
# ======================================================================

def test_a_dry_run_lists_the_uploaded_pdf_and_changes_nothing(repo, client, auth_headers):
    document_id = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_TEXT))
    embedder = FakeEmbedder()
    before = db_snapshot()

    code, out, err = run_cli(repo, embedder, "--database", TEST_DATABASE)

    assert code == EXIT_OK
    assert value(out, "Will index") == "1" and value(out, "Blocked") == "0"
    assert re.search(rf"^  doc {document_id}\s+never_queued$", out, re.M)  # a PDF with its file present is not blocked
    assert DRY_RUN_FOOTER in out
    assert embedder.calls == []
    assert db_snapshot() == before and before[1] == 0


# ======================================================================
# C. Execute: what is published
# ======================================================================

def test_the_backfill_indexes_the_pdf_with_the_right_identity(indexed):
    assert indexed.code == EXIT_OK, indexed.err
    row = document_row(indexed.document_id)
    assert (row.indexing_status, row.indexing_error, row.indexing_attempt, row.leased) == ("indexed", None, 1, False)
    assert row.indexed_source_version == row.source_version == 1
    assert row.indexed_embedding_id == FAKE_EMBEDDING_ID
    assert row.source_sha256 == hashlib.sha256(row.extracted_text.encode("utf-8")).hexdigest()
    assert row.indexed_at is not None
    assert row.extracted_text == indexed.text  # the backfill never rewrites the text

    def normalized(data):
        return json.loads(json.dumps(data))

    assert json.loads(row.indexed_signature) == {
        "index_build_version": INDEX_BUILD_VERSION,
        "chunker": normalized(DEFAULT_CHUNKER_CONFIG.signature_fields()),
        "offset_mapping": normalized(token_offsets.signature_fields()),
        "embedder": {"model_id": "test/fake", "embedding_dim": EMBEDDING_DIM},
    }


def test_the_run_reports_the_pdf_as_indexed(indexed):
    chunks = chunks_of(indexed.document_id)
    match = re.search(rf"^\[1/1\] doc {indexed.document_id}\s+indexed \((\d+) chunks", indexed.out, re.M)
    assert match and int(match.group(1)) == len(chunks)
    assert value(indexed.out, "Planned for indexing") == "1"
    assert value(indexed.out, "Indexed") == "1" and value(indexed.out, "Failed") == "0"
    assert "Result: finished without failures" in indexed.out
    assert NOTICE_MODEL_OK in indexed.err


def test_chunks_are_exact_slices_of_the_extracted_text(indexed):
    chunks = chunks_of(indexed.document_id)
    extracted = indexed.text
    assert len(chunks) >= 2  # 735 tokens cannot fit in one chunk of at most 400
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert 0 <= c.start_offset < c.end_offset <= len(extracted)
        assert c.text == extracted[c.start_offset:c.end_offset]
        assert c.span_sha256 == hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        assert 1 <= len(c.text.split()) <= DEFAULT_CHUNKER_CONFIG.max_tokens  # one fake token per word
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.start_offset < later.start_offset and earlier.end_offset < later.end_offset
    covered = bytearray(len(extracted))
    for c in chunks:
        covered[c.start_offset:c.end_offset] = b"\x01" * (c.end_offset - c.start_offset)
    for word in re.finditer(r"\S+", extracted):
        assert all(covered[word.start():word.end()]), f"word at {word.start()} is in no chunk"


def test_every_chunk_has_the_384_dimension_embedding_of_its_own_text(indexed):
    for c in chunks_of(indexed.document_id):
        vector = np.asarray(c.embedding, dtype=np.float32)
        assert vector.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
        assert np.array_equal(vector, fake_vector(c.text))  # the stored vector belongs to the stored text


def test_the_embedder_received_the_exact_extracted_text_once(indexed):
    chunks = chunks_of(indexed.document_id)
    calls = indexed.embedder.calls
    assert calls[:2] == [("signature_fields", 0), ("embed_passages", 1)]  # the preflight
    assert [name for name, _ in calls].count("source_units") == 1
    assert ("source_units", len(indexed.text)) in calls                  # the text, unstripped
    sizes = [size for name, size in calls[2:] if name == "embed_passages"]
    count = len(chunks)
    assert sizes == [min(EMBED_GROUP_SIZE, count - start) for start in range(0, count, EMBED_GROUP_SIZE)]


# ======================================================================
# D. Execute: pages
# ======================================================================

def test_chunk_pages_match_an_independent_marker_oracle(indexed):
    chunks = chunks_of(indexed.document_id)
    starts = page_starts(indexed.text, range(1, PAGES + 1))
    for c in chunks:
        assert (c.page_start, c.page_end) == (
            page_at(starts, c.start_offset),
            page_at(starts, c.end_offset - 1),
        )
    for number in range(1, PAGES + 1):  # a chunk holding a page's start marker covers that page
        for c in chunks:
            if f"PAGE{number}START" in c.text:
                assert c.page_start <= number <= c.page_end
    assert chunks[0].page_start == 1 and chunks[-1].page_end == PAGES
    assert any(c.page_start < c.page_end for c in chunks)  # at least one chunk spans a page break


# ======================================================================
# E. Rerun and cleanup
# ======================================================================

def test_a_rerun_finds_nothing_to_do_and_changes_nothing(indexed, repo):
    before = db_snapshot()
    rerun = FakeEmbedder()

    code, out, err = run_cli(repo, rerun, *execute_args())

    assert code == EXIT_OK and rerun.calls == []
    assert value(out, "Planned for indexing") == "0" and value(out, "Skipped (nothing to do)") == "1"
    assert db_snapshot() == before
    code, out, err = run_cli(repo, FakeEmbedder(), "--database", TEST_DATABASE)
    assert value(out, "Will index") == "0" and value(out, "By category") == "current 1"


def test_deleting_the_document_removes_its_chunks_and_its_stored_file(indexed, client, auth_headers):
    assert chunks_of(indexed.document_id) and indexed.path.is_file()

    response = client.delete(f"{DOCUMENTS}/{indexed.document_id}", headers=auth_headers)

    assert response.status_code == 204
    assert chunks_of(indexed.document_id) == []
    assert not indexed.path.exists()
    assert db_snapshot() == ([], 0)


# ======================================================================
# OPTIONAL HARDENING
# ======================================================================

def test_optional_a_blank_page_between_text_pages(repo, client, auth_headers):
    document_id = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_GAP))
    row = document_row(document_id)
    extracted = row.extracted_text
    mapping = pdf_pages.map_document_pages(repo / "uploads" / row.stored_name, extracted)
    assert mapping.is_available and mapping.page_count == 3
    assert mapping.spans[1].start == mapping.spans[1].end  # the blank page is a zero-length span

    code, out, err = run_cli(repo, FakeEmbedder(), *execute_args())

    assert code == EXIT_OK, err
    starts = page_starts(extracted, (1, 3))
    chunks = chunks_of(document_id)
    for c in chunks:
        assert (c.page_start, c.page_end) == (page_at(starts, c.start_offset), page_at(starts, c.end_offset - 1))
        assert 2 not in (c.page_start, c.page_end)  # no chunk can start or end on the blank page
    assert any((c.page_start, c.page_end) == (1, 3) for c in chunks)


def test_optional_a_pdf_whose_text_no_longer_matches_is_indexed_with_null_pages(repo, client, auth_headers):
    document_id = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_TEXT))
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET extracted_text = extracted_text || ' appended' WHERE id = :id"),
            {"id": document_id},
        )

    code, out, err = run_cli(repo, FakeEmbedder(), *execute_args())

    assert code == EXIT_OK, err
    extracted = document_row(document_id).extracted_text
    chunks = chunks_of(document_id)
    assert chunks and all(c.page_start is None and c.page_end is None for c in chunks)  # the V1 fallback
    assert all(c.text == extracted[c.start_offset:c.end_offset] for c in chunks)


def test_optional_limit_one_indexes_the_lowest_pdf_first_and_a_later_run_the_rest(repo, client, auth_headers):
    first = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_TEXT))
    second = upload_pdf(client, auth_headers, pdf_bytes(LAYOUT_TEXT))

    code, out, err = run_cli(repo, FakeEmbedder(), *execute_args("--limit", "1"))
    assert code == EXIT_OK
    assert document_row(first).indexing_status == "indexed"
    assert document_row(second).indexing_status is None and chunks_of(second) == []
    assert value(out, "Deferred by --limit") == "1"
    published_at = document_row(first).indexed_at

    code, out, err = run_cli(repo, FakeEmbedder(), *execute_args())
    assert code == EXIT_OK and document_row(second).indexing_status == "indexed"
    assert document_row(first).indexed_at == published_at and document_row(first).indexing_attempt == 1


def test_optional_the_runbook_verification_queries_run_and_find_nothing_wrong(indexed):
    blocks = re.findall(r"```sql\n(.*?)```", RUNBOOK.read_text(encoding="utf-8"), re.S)
    checks = [block for block in blocks if block.lstrip().startswith("-- verify:")]
    assert len(checks) >= 8  # the runbook really contains its checks
    with engine.connect() as connection:
        for block in checks:
            kind = block.lstrip().splitlines()[0].split(":", 1)[1].strip()
            assert kind in ("none", "info"), block
            assert "%" not in block  # psycopg2 would read a percent sign as a parameter
            rows = connection.exec_driver_sql(block).all()
            if kind == "none":
                assert rows == [], block