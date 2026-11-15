"""Integration tests for the corpus ingestion driver against the test database (documind_test).

A small synthetic corpus (md, txt, pdf) stands in for the six real documents. The evaluation user is
created through the real registration endpoint. No embedding model is loaded and no indexing code is
run: where a test needs an "indexed" document, the published rows are written with plain SQL.
"""
import hashlib
import json
import re
import shutil
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.core import processing, storage
from app.core.database import SessionLocal, engine
from app.crud.document import create_uploaded_document
from app.models.user import User
from app.rag import corpus_ingest as ci
from app.rag import corpus_ingest_cli as cli
from app.rag.chunking import ChunkerConfig, chunk_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample.pdf"
EVAL_EMAIL = "evaluation@example.com"
EMBEDDING_ID = "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
KEYS = ["alpha-guide", "beta-notes", "gamma-sample"]


def make_corpus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "alpha-guide.md").write_bytes(b"# Alpha\n\nAlpha explains containers.\n")
    (directory / "beta-notes.txt").write_bytes(b"Beta notes about startup order.\nSecond line.\n")
    shutil.copy(SAMPLE_PDF, directory / "gamma-sample.pdf")
    documents = []
    for key, name, fmt in (("alpha-guide", "alpha-guide.md", "md"), ("beta-notes", "beta-notes.txt", "txt"),
                           ("gamma-sample", "gamma-sample.pdf", "pdf")):
        documents.append({
            "corpus_key": key, "title": key, "publisher": "Test", "source_url": "https://example.com/" + key,
            "license": "MIT", "license_url": "https://example.com/license", "format": fmt, "language": "en",
            "source_file_sha256": ci.hash_file(directory / name)[0],
            "extracted_text_sha256": None, "extracted_text_length": None, "extraction": None,
            "redistributable": True,
        })
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": "eval-manifest-v1", "corpus_version": "eval-corpus-v1",
                                "stage": "draft", "documents": documents}, indent=2) + "\n", encoding="utf-8")
    return path


class Ctx:
    def __init__(self, files, owner_id, other_id, upload_root, corpus_dir):
        self.files = files
        self.owner_id = owner_id
        self.other_id = other_id
        self.upload_root = upload_root
        self.corpus_dir = corpus_dir
        self.digest = hashlib.sha256(EVAL_EMAIL.encode()).hexdigest()

    def kwargs(self, **override):
        values = dict(
            engine=engine, session_factory=SessionLocal, files=self.files, owner_id=self.owner_id,
            owner_sha256=self.digest, upload_root=self.upload_root, expected_database=engine.url.database,
            execute=True,
        )
        values.update(override)
        return values

    def ingest(self, **override):
        return ci.run_ingest(**self.kwargs(**override))

    def file(self, key):
        return next(f for f in self.files if f.corpus_key == key)


@pytest.fixture
def ctx(make_user, db, tmp_path_factory):
    make_user(EVAL_EMAIL)
    make_user("someone-else@example.com")
    owner = db.query(User).filter(User.email == EVAL_EMAIL).one().id
    other = db.query(User).filter(User.email == "someone-else@example.com").one().id
    corpus_dir = tmp_path_factory.mktemp("corpus")
    manifest = make_corpus(corpus_dir)
    loaded = ci.load_corpus(manifest, corpus_dir, expected_count=3)
    return Ctx(loaded.files, owner, other, storage.UPLOAD_ROOT, corpus_dir)


def table_state():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, owner_id, filename, description, stored_name, content_type, size_bytes, processing_status, "
            "processing_error, encode(sha256(convert_to(extracted_text,'UTF8')),'hex') AS text_sha, source_version, "
            "indexing_status, created_at FROM documents ORDER BY id"
        )).all()
        chunks = conn.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
    return [tuple(r) for r in rows], chunks


def stored_files():
    return sorted(p.name for p in Path(storage.UPLOAD_ROOT).iterdir() if p.is_file())


def seed_row(ctx, file, *, owner=None, status="pending", **fields):
    """A document row created with the production function; the stored copy is a plain file copy."""
    stored = f"seed-{file.corpus_key}-{status}.bin"
    shutil.copy(file.path, Path(storage.UPLOAD_ROOT) / stored)
    session = SessionLocal()
    try:
        document = create_uploaded_document(
            session, filename=fields.pop("filename", file.filename),
            description=fields.pop("description", file.description), stored_name=stored,
            content_type=file.content_type, size_bytes=file.size, owner_id=owner or ctx.owner_id,
        )
        document.processing_status = status
        session.commit()
        return document.id
    finally:
        session.close()


# ---------- read-only behaviour ----------

def test_dry_run_and_plan_write_nothing(ctx):
    before = (table_state(), stored_files())
    dry = ctx.ingest(execute=False)
    assert dry.outcome == "dry_run"
    assert [d.code for d in dry.plan.documents] == ["new", "new", "new"]
    assert (table_state(), stored_files()) == before
    assert before[0][0] == [] and before[1] == []


def test_inspection_uses_a_read_only_transaction(ctx):
    with ci.read_only(engine) as conn:
        assert conn.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        with pytest.raises(Exception):
            conn.execute(text("DELETE FROM documents"))


# ---------- safe ingestion of new documents ----------

def test_ingestion_creates_stores_and_extracts_each_document(ctx):
    result = ctx.ingest()
    assert result.outcome == "ingested"
    assert [d.status for d in result.documents] == ["created_extracted"] * 3

    rows, chunks = table_state()
    assert len(rows) == 3 and chunks == 0
    assert len(stored_files()) == 3
    by_name = {row[2]: row for row in rows}
    for file in ctx.files:
        row = by_name[file.filename]
        (_id, owner, _fn, description, stored_name, content_type, size, status, error, text_sha, version,
         indexing_status, _created) = row
        assert owner == ctx.owner_id
        assert description == f"eval-corpus-v1:{file.corpus_key}"
        assert content_type == file.content_type
        assert size == file.size
        assert (status, error, version) == ("completed", None, 1)
        assert text_sha == file.expected_text_sha256
        assert indexing_status is None               # the driver never indexes
        assert ci.hash_file(Path(storage.UPLOAD_ROOT) / stored_name) == (file.sha256, file.size)


def test_second_invocation_is_a_no_op_and_never_extracts_again(ctx, monkeypatch):
    assert ctx.ingest().outcome == "ingested"
    before = (table_state(), stored_files())

    def forbidden(_id):
        raise AssertionError("extraction must not be invoked for a completed document")

    monkeypatch.setattr(processing, "process_document", forbidden)
    second = ctx.ingest(process=forbidden)
    assert second.outcome == "nothing_to_do"
    assert [d.code for d in second.plan.documents] == ["present_extracted"] * 3
    assert (table_state(), stored_files()) == before


def test_a_partial_corpus_only_creates_what_is_missing(ctx):
    first = ctx.file("alpha-guide")
    assert ctx.ingest(files=[first]).outcome == "ingested"
    calls = []

    def spy(document_id):
        calls.append(document_id)
        processing.process_document(document_id)

    result = ctx.ingest(process=spy)
    assert result.outcome == "ingested"
    assert [d.status for d in result.documents] == ["skipped_present", "created_extracted", "created_extracted"]
    assert len(calls) == 2
    assert len(table_state()[0]) == 3


# ---------- refusals ----------

def test_wrong_database_is_refused_before_any_write(ctx):
    before = (table_state(), stored_files())
    result = ctx.ingest(expected_database="documind_other_test")
    assert result.outcome == "refused"
    assert "documind_other_test" in " ".join(result.problems)
    assert (table_state(), stored_files()) == before


def test_wrong_session_factory_is_refused(ctx):
    other_engine = create_engine(str(engine.url.render_as_string(hide_password=False)))
    try:
        from sqlalchemy.orm import sessionmaker

        result = ctx.ingest(session_factory=sessionmaker(bind=other_engine))
    finally:
        other_engine.dispose()
    assert result.outcome == "refused"
    assert "not bound to the verified database engine" in " ".join(result.problems)
    assert table_state()[0] == [] and stored_files() == []


def test_wrong_upload_directory_is_refused(ctx, tmp_path_factory):
    result = ctx.ingest(upload_root=tmp_path_factory.mktemp("elsewhere"))
    assert result.outcome == "refused"
    assert "storage.UPLOAD_ROOT" in " ".join(result.problems)
    assert table_state()[0] == [] and stored_files() == []


@pytest.mark.parametrize("case", ["other_user", "wrong_digest", "missing_user"])
def test_wrong_owner_is_refused(ctx, case):
    owner = {"other_user": ctx.other_id, "wrong_digest": ctx.owner_id, "missing_user": 99999}[case]
    digest = "0" * 64 if case == "wrong_digest" else ctx.digest
    result = ctx.ingest(owner_id=owner, owner_sha256=digest)
    assert result.outcome == "refused"
    assert "evaluation owner does not exist or its email digest does not match" in " ".join(result.plan.problems)
    assert table_state()[0] == [] and stored_files() == []


def test_owner_with_unrelated_documents_is_refused(ctx):
    session = SessionLocal()
    try:
        create_uploaded_document(session, filename="mine.txt", description="not a corpus doc", stored_name="x.txt",
                                 content_type="text/plain", size_bytes=1, owner_id=ctx.owner_id)
    finally:
        session.close()
    result = ctx.ingest()
    assert result.outcome == "refused"
    assert "outside the corpus" in " ".join(result.plan.problems)
    assert len(table_state()[0]) == 1


def test_duplicate_rows_are_refused_and_nothing_is_written(ctx):
    file = ctx.file("alpha-guide")
    seed_row(ctx, file, status="completed")
    session = SessionLocal()
    try:
        create_uploaded_document(session, filename=file.filename, description=None, stored_name="dup.md",
                                 content_type=file.content_type, size_bytes=1, owner_id=ctx.other_id)
    finally:
        session.close()
    before = (table_state(), stored_files())
    result = ctx.ingest()
    assert result.outcome == "refused"
    assert result.plan.documents[0].code == "duplicate"
    assert (table_state(), stored_files()) == before


def test_matching_filename_with_other_owner_or_other_key_is_refused(ctx):
    file = ctx.file("beta-notes")
    seed_row(ctx, file, owner=ctx.other_id)
    result = ctx.ingest(execute=False)
    assert result.outcome == "refused" and result.plan.documents[1].code == "conflict_owner"


def test_matching_key_with_other_filename_is_refused(ctx):
    file = ctx.file("beta-notes")
    seed_row(ctx, file, filename="renamed.txt")
    result = ctx.ingest(execute=False)
    assert result.outcome == "refused" and result.plan.documents[1].code == "conflict_identity"


@pytest.mark.parametrize("state", ["pending", "processing", "failed"])
def test_partial_rows_stop_the_run_and_are_not_retried(ctx, state):
    seed_row(ctx, ctx.file("alpha-guide"), status=state)
    calls = []
    before = (table_state(), stored_files())
    result = ctx.ingest(process=lambda i: calls.append(i))
    assert result.outcome == "refused"
    assert result.plan.documents[0].code == f"partial_{state}"
    assert calls == []
    assert (table_state(), stored_files()) == before


def test_orphan_file_is_reported_and_blocks_ingestion(ctx):
    shutil.copy(ctx.file("alpha-guide").path, Path(storage.UPLOAD_ROOT) / "orphan.md")
    result = ctx.ingest()
    assert result.outcome == "refused"
    assert result.plan.orphans == ("orphan.md",)
    assert table_state()[0] == []
    assert stored_files() == ["orphan.md"]            # nothing adopted, nothing deleted


def test_a_stored_file_that_changed_blocks_the_next_run(ctx):
    assert ctx.ingest().outcome == "ingested"
    stored = table_state()[0][0][4]
    (Path(storage.UPLOAD_ROOT) / stored).write_bytes(b"tampered")
    before = table_state()
    result = ctx.ingest()
    assert result.outcome == "refused"
    assert result.plan.documents[0].code == "mismatch_stored_file"
    assert table_state() == before


def test_a_corpus_file_changed_after_loading_stops_before_any_write(ctx):
    (ctx.corpus_dir / "alpha-guide.md").write_bytes(b"# Alpha\n\nChanged after the manifest check.\n")
    result = ctx.ingest()
    assert result.outcome == "stopped"
    assert result.documents[0].status == "source_changed"
    assert [d.status for d in result.documents[1:]] == ["not_attempted"] * 2
    assert table_state()[0] == [] and stored_files() == []


# ---------- extraction failure ----------

def test_extraction_failure_stops_the_run_and_nothing_is_retried_or_deleted(ctx):
    calls = []

    def failing(document_id):
        calls.append(document_id)
        session = SessionLocal()
        try:
            processing._mark_failed(session, document_id, "No extractable text found (the file may be empty or scanned)")
        finally:
            session.close()

    result = ctx.ingest(process=failing)
    assert result.outcome == "stopped"
    assert [d.status for d in result.documents] == ["failed_extraction", "not_attempted", "not_attempted"]
    assert "No extractable text found" in result.documents[0].detail
    assert len(calls) == 1
    rows, _ = table_state()
    assert [r[7] for r in rows] == ["failed"]
    assert len(stored_files()) == 1                    # the stored copy is kept for manual recovery

    # A later run sees a partial row and refuses; it does not retry or clean up.
    again = ctx.ingest(process=lambda i: calls.append(i))
    assert again.outcome == "refused" and again.plan.documents[0].code == "partial_failed"
    assert len(calls) == 1 and table_state()[0] == rows


def test_a_text_that_differs_from_the_expectation_is_reported(ctx):
    def wrong_text(document_id):
        session = SessionLocal()
        try:
            session.execute(text("UPDATE documents SET extracted_text = 'something else', "
                                 "processing_status = 'completed' WHERE id = :i"), {"i": document_id})
            session.commit()
        finally:
            session.close()

    result = ctx.ingest(process=wrong_text)
    assert result.outcome == "stopped"
    assert result.documents[0].status == "text_mismatch"


def test_row_creation_failure_is_reported_with_the_orphan_name(ctx, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ci, "create_uploaded_document", broken)
    result = ctx.ingest()
    assert result.outcome == "stopped"
    assert result.documents[0].status == "row_failed"
    assert "orphan" in result.documents[0].detail
    assert table_state()[0] == [] and len(stored_files()) == 1


# ---------- advisory lock and concurrency ----------

def test_a_held_lock_blocks_a_second_invocation(ctx):
    with ci.advisory_lock(engine) as first:
        assert first is True
        with ci.advisory_lock(engine) as second:
            assert second is False
        result = ctx.ingest()
        assert result.outcome == "locked"
        assert table_state()[0] == [] and stored_files() == []
    # Released on exit: a later run works.
    assert ctx.ingest(execute=False).outcome == "dry_run"
    with ci.advisory_lock(engine) as again:
        assert again is True


def test_the_lock_is_released_after_a_run(ctx):
    assert ctx.ingest().outcome == "ingested"
    with ci.advisory_lock(engine) as acquired:
        assert acquired is True


def test_concurrent_invocations_never_duplicate_documents(ctx):
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        outcomes.append(ctx.ingest().outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert len(outcomes) == 2
    assert "ingested" in outcomes
    assert set(outcomes) <= {"ingested", "locked", "nothing_to_do", "refused"}
    rows, _ = table_state()
    assert len(rows) == 3 and len({r[3] for r in rows}) == 3
    assert len(stored_files()) == 3


# ---------- verification ----------

# A small chunker config makes even the short test documents produce several OVERLAPPING, edge-trimmed chunks,
# exactly like production output (production: 300/400/50 tokens; here words stand in for tokens).
_SMALL_CONFIG = ChunkerConfig(target_tokens=3, max_tokens=4, overlap_tokens=1)


def overlapping_ranges(body: str, config: ChunkerConfig = _SMALL_CONFIG) -> list[tuple[int, int]]:
    spans = [m.span() for m in re.finditer(r"\S+", body)]
    return [(c.start, c.end) for c in chunk_text(body, spans, config)]


def document_text(document_id: int) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT extracted_text FROM documents WHERE id = :i"), {"i": document_id}).scalar_one()


def stored_ranges(document_id: int) -> list[tuple[int, int]]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT start_offset, end_offset FROM document_chunks WHERE document_id = :i "
                                 "ORDER BY chunk_index"), {"i": document_id}).all()
    return [(r[0], r[1]) for r in rows]


def publish_chunks(document_id: int, embedding_id: str = EMBEDDING_ID, ranges=None) -> None:
    """Stand-in for the indexer's publish step: overlapping, edge-trimmed chunks plus the indexed metadata."""
    body = document_text(document_id)
    with engine.begin() as conn:
        for index, (start, end) in enumerate(ranges if ranges is not None else overlapping_ranges(body)):
            conn.execute(text(
                "INSERT INTO document_chunks (document_id, chunk_index, start_offset, end_offset, page_start, page_end, "
                "text, span_sha256, embedding) "
                "SELECT d.id, :n, :s, :e, 1, 1, substr(d.extracted_text, :s + 1, :e - :s), "
                "encode(sha256(convert_to(substr(d.extracted_text, :s + 1, :e - :s), 'UTF8')), 'hex'), "
                "array_fill(0::real, ARRAY[384])::vector FROM documents d WHERE d.id = :i"
            ), {"n": index, "s": start, "e": end, "i": document_id})
        conn.execute(text(
            "UPDATE documents SET indexing_status = 'indexed', indexed_source_version = 1, "
            "source_sha256 = encode(sha256(convert_to(extracted_text,'UTF8')),'hex'), indexed_signature = 'sig', "
            "indexed_embedding_id = :e, indexed_at = now(), indexing_attempt = 1 WHERE id = :i"
        ), {"e": embedding_id, "i": document_id})


def run_verify(ctx, embedding_id=EMBEDDING_ID):
    with ci.read_only(engine) as conn:
        return ci.verify(conn, ctx.files, owner_id=ctx.owner_id, owner_sha256=ctx.digest,
                         upload_root=ctx.upload_root, embedding_id=embedding_id)


def failed_checks(report):
    return {(d.corpus_key, c.name) for d in report.documents for c in d.checks if not c.ok}


def test_verify_passes_only_for_extracted_and_indexed_documents(ctx):
    assert ctx.ingest().outcome == "ingested"
    report = run_verify(ctx)
    assert not report.ok
    assert {name for _key, name in failed_checks(report)} >= {
        "indexing_status", "indexed_embedding_id", "indexed_signature", "chunk_offsets_valid", "chunks_contiguous"}
    assert not any(name in ("stored_file_hash_and_size", "text_sha256", "extraction_completed")
                   for _key, name in failed_checks(report))

    for row in table_state()[0]:
        publish_chunks(row[0])
    report = run_verify(ctx)
    assert report.ok, failed_checks(report)
    pdf = next(d for d in report.documents if d.corpus_key == "gamma-sample")
    count = len(stored_ranges(pdf.document_id))
    assert count > 2
    assert any(c.name == "pdf_page_numbers_reported" and f"{count} of {count}" in c.detail for c in pdf.checks)


def test_verify_detects_each_kind_of_damage(ctx):
    assert ctx.ingest().outcome == "ingested"
    ids = {row[3].split(":")[1]: row[0] for row in table_state()[0]}
    for document_id in ids.values():
        publish_chunks(document_id)
    assert run_verify(ctx).ok

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM document_chunks WHERE document_id = :i AND chunk_index = 0"), {"i": ids["alpha-guide"]})
        conn.execute(text("UPDATE document_chunks SET text = repeat('x', length(text)) "
                          "WHERE document_id = :i AND chunk_index = 1"), {"i": ids["beta-notes"]})
        conn.execute(text("UPDATE document_chunks SET span_sha256 = repeat('a', 64) "
                          "WHERE document_id = :i AND chunk_index = 0"), {"i": ids["gamma-sample"]})
    failed = failed_checks(run_verify(ctx))
    assert ("alpha-guide", "chunks_contiguous") in failed
    assert ("beta-notes", "chunk_text_matches_extracted_text") in failed
    assert ("gamma-sample", "chunk_span_sha256") in failed

    assert ("alpha-guide", "indexed_embedding_id") in failed_checks(run_verify(ctx, embedding_id="another-model"))

    with engine.begin() as conn:
        conn.execute(text("UPDATE documents SET source_sha256 = repeat('b', 64) WHERE id = :i"), {"i": ids["alpha-guide"]})
    assert ("alpha-guide", "indexed_source_sha256") in failed_checks(run_verify(ctx))


def test_verify_reports_a_missing_or_duplicated_document(ctx):
    report = run_verify(ctx)
    assert {c.name for d in report.documents for c in d.checks if not c.ok} == {"exactly_one_document"}
    assert ctx.ingest().outcome == "ingested"
    session = SessionLocal()
    try:
        create_uploaded_document(session, filename="alpha-guide.md", description=None, stored_name="d.md",
                                 content_type="text/markdown", size_bytes=1, owner_id=ctx.other_id)
    finally:
        session.close()
    assert ("alpha-guide", "exactly_one_document") in failed_checks(run_verify(ctx))


def test_verify_is_read_only(ctx):
    assert ctx.ingest().outcome == "ingested"
    before = (table_state(), stored_files())
    run_verify(ctx)
    assert (table_state(), stored_files()) == before


# ---------- regression: completed with a processing error ----------

def test_completed_with_a_processing_error_blocks_and_is_not_skipped(ctx):
    assert ctx.ingest().outcome == "ingested"
    document_id = table_state()[0][0][0]
    with engine.begin() as conn:
        conn.execute(text("UPDATE documents SET processing_error = 'left over' WHERE id = :i"), {"i": document_id})
    before = (table_state(), stored_files())
    calls = []
    result = ctx.ingest(process=lambda i: calls.append(i))
    assert result.outcome == "refused"
    assert result.plan.documents[0].code == "inconsistent_error"
    assert result.plan.documents[0].action == ci.ACTION_REFUSE
    assert calls == [] and (table_state(), stored_files()) == before
    assert ("alpha-guide", "extraction_completed") in failed_checks(run_verify(ctx))


# ---------- chunk offsets: overlapping, edge-trimmed chunks are what the indexer produces ----------

def rewrite_chunk(document_id: int, index: int, start: int, end: int) -> None:
    """Change a chunk's range and keep its text and hash consistent, so only the offsets are wrong."""
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE document_chunks c SET start_offset = :s, end_offset = :e, "
            "text = substr(d.extracted_text, :s + 1, :e - :s), "
            "span_sha256 = encode(sha256(convert_to(substr(d.extracted_text, :s + 1, :e - :s), 'UTF8')), 'hex') "
            "FROM documents d WHERE d.id = c.document_id AND c.document_id = :i AND c.chunk_index = :n"
        ), {"s": start, "e": end, "i": document_id, "n": index})


def indexed_alpha(ctx):
    assert ctx.ingest().outcome == "ingested"
    ids = {row[3].split(":")[1]: row[0] for row in table_state()[0]}
    for document_id in ids.values():
        publish_chunks(document_id)
    assert run_verify(ctx).ok
    document_id = ids["alpha-guide"]
    return document_id, document_text(document_id), stored_ranges(document_id)


def check_detail(report, key, name):
    return next(c.detail for d in report.documents if d.corpus_key == key for c in d.checks if c.name == name)


def test_overlapping_trimmed_chunks_verify_and_the_trailing_newline_is_not_a_gap(ctx):
    _document_id, body, ranges = indexed_alpha(ctx)
    assert len(ranges) >= 2
    assert all(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges)))    # every transition overlaps
    assert body.endswith("\n") and ranges[-1][1] == len(body) - 1                  # trimmed tail, as in production
    assert ranges[0][0] == 0
    report = run_verify(ctx)
    assert report.ok
    assert check_detail(report, "alpha-guide", "chunk_coverage") == "every meaningful character is in a chunk"


def test_an_uncovered_meaningful_character_at_the_tail_fails_only_coverage(ctx):
    document_id, body, ranges = indexed_alpha(ctx)
    rewrite_chunk(document_id, len(ranges) - 1, ranges[-1][0], ranges[-1][1] - 1)   # drops the final "."
    report = run_verify(ctx)
    assert failed_checks(report) == {("alpha-guide", "chunk_coverage")}
    assert f"first at offset {ranges[-1][1] - 1}" in check_detail(report, "alpha-guide", "chunk_coverage")


def test_an_uncovered_meaningful_character_before_the_first_chunk_fails_only_coverage(ctx):
    indexed_alpha(ctx)
    beta_id = next(row[0] for row in table_state()[0] if row[2] == "beta-notes.txt")
    body, ranges = document_text(beta_id), stored_ranges(beta_id)
    assert body[:5] == "Beta " and ranges[0][0] == 0 and ranges[1][0] > 5
    rewrite_chunk(beta_id, 0, 5, ranges[0][1])                                        # "Beta " is now in no chunk
    report = run_verify(ctx)
    assert failed_checks(report) == {("beta-notes", "chunk_coverage")}
    assert "first at offset 0" in check_detail(report, "beta-notes", "chunk_coverage")


def test_non_increasing_starts_fail_only_the_offset_check(ctx):
    document_id, _body, ranges = indexed_alpha(ctx)
    rewrite_chunk(document_id, 1, ranges[0][0], ranges[1][1])                         # same start as chunk 0
    report = run_verify(ctx)
    assert failed_checks(report) == {("alpha-guide", "chunk_offsets_valid")}
    assert "chunk 1 does not start and end after chunk 0" in check_detail(report, "alpha-guide", "chunk_offsets_valid")


def test_a_chunk_that_starts_with_whitespace_fails_only_the_trim_check(ctx):
    document_id, body, ranges = indexed_alpha(ctx)
    index = next(i for i in range(1, len(ranges)) if body[ranges[i][0] - 1].isspace())
    rewrite_chunk(document_id, index, ranges[index][0] - 1, ranges[index][1])
    assert failed_checks(run_verify(ctx)) == {("alpha-guide", "chunk_edges_trimmed")}


@pytest.mark.parametrize("case", ["negative_start", "empty_range", "end_beyond_text"])
def test_verify_rejects_invalid_ranges_even_if_the_database_constraints_were_missing(ctx, case):
    document_id, body, ranges = indexed_alpha(ctx)
    length = len(body)
    values = {"negative_start": (0, -1, ranges[0][1]), "empty_range": (0, 5, 5),
              "end_beyond_text": (len(ranges) - 1, ranges[-1][0], length + 5)}
    index, start, end = values[case]
    connection = engine.connect()
    transaction = connection.begin()
    try:
        connection.execute(text("ALTER TABLE document_chunks DROP CONSTRAINT ck_document_chunks_offsets_valid"))
        connection.execute(text("ALTER TABLE document_chunks DROP CONSTRAINT ck_document_chunks_text_length_matches_offsets"))
        connection.execute(text("UPDATE document_chunks SET start_offset = :s, end_offset = :e "
                                "WHERE document_id = :i AND chunk_index = :n"),
                           {"s": start, "e": end, "i": document_id, "n": index})
        report = ci.verify(connection, ctx.files, owner_id=ctx.owner_id, owner_sha256=ctx.digest,
                           upload_root=ctx.upload_root, embedding_id=EMBEDDING_ID)
    finally:
        transaction.rollback()
        connection.close()
    assert ("alpha-guide", "chunk_offsets_valid") in failed_checks(report)
    assert "invalid range" in check_detail(report, "alpha-guide", "chunk_offsets_valid")
    assert run_verify(ctx).ok          # the rolled-back transaction left the data and constraints intact


def test_a_document_with_no_chunks_fails_cleanly_instead_of_crashing(ctx):
    assert ctx.ingest().outcome == "ingested"
    report = run_verify(ctx)
    assert {("alpha-guide", "chunk_offsets_valid"), ("alpha-guide", "chunks_contiguous")} <= failed_checks(report)
    assert check_detail(report, "alpha-guide", "chunk_offsets_valid") == "there are no chunks"


# ---------- indexed metadata: each part is checked and named ----------

def test_the_embedding_id_must_be_the_full_model_at_revision_value(ctx):
    indexed_alpha(ctx)
    model_only = EMBEDDING_ID.split("@")[0]                    # what the first verifier version compared against
    report = run_verify(ctx, embedding_id=model_only)
    assert failed_checks(report) == {(key, "indexed_embedding_id") for key in KEYS}
    detail = check_detail(report, "alpha-guide", "indexed_embedding_id")
    assert repr(model_only) in detail and repr(EMBEDDING_ID) in detail


@pytest.mark.parametrize("column, value, expected", [
    ("indexed_signature", "''", {"indexed_signature"}),
    ("source_sha256", "repeat('b', 64)", {"indexed_source_sha256"}),
    ("indexed_embedding_id", "'intfloat/multilingual-e5-small'", {"indexed_embedding_id"}),
])
def test_malformed_indexed_metadata_is_reported_precisely(ctx, column, value, expected):
    document_id, _body, _ranges = indexed_alpha(ctx)
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE documents SET {column} = {value} WHERE id = :i"), {"i": document_id})
    assert failed_checks(run_verify(ctx)) == {("alpha-guide", name) for name in expected}


def test_a_changed_source_version_is_reported(ctx):
    document_id, _body, _ranges = indexed_alpha(ctx)
    with engine.begin() as conn:
        conn.execute(text("UPDATE documents SET source_version = 2 WHERE id = :i"), {"i": document_id})
    failed = failed_checks(run_verify(ctx))
    assert {("alpha-guide", "source_version"), ("alpha-guide", "indexed_source_version")} <= failed


# ---------- production chunker AND tokenizer ----------

def test_rows_built_with_the_production_chunker_and_tokenizer_verify(ctx):
    try:
        from app.rag.embedder import Embedder, EmbedderConfig, EmbedderUnavailableError
        from app.rag.indexing import plan_chunks

        embedder = Embedder(EmbedderConfig.from_settings())
        assert ctx.ingest().outcome == "ingested"
        plans = {row[0]: plan_chunks(document_text(row[0]), embedder) for row in table_state()[0]}
    except (ImportError, EmbedderUnavailableError) as exc:
        pytest.skip(f"the pinned tokenizer/model is not available here ({type(exc).__name__})")
    assert embedder.embedding_id == cli.expected_embedding_id()
    for document_id, plan in plans.items():
        publish_chunks(document_id, embedding_id=embedder.embedding_id,
                       ranges=[(c.start, c.end) for c in plan.chunks])
    report = run_verify(ctx, embedding_id=cli.expected_embedding_id())
    assert report.ok, failed_checks(report)


# ---------- regression: live failure "row_failed | creating the document row failed (InvalidRequestError)" ----------
# pytest imports app.main (through conftest), which registers every ORM model. The real command line imports
# only the driver, so Document.owner -> 'User' could not be resolved and the first Document(...) raised
# InvalidRequestError after the file had already been stored. These tests therefore use a FRESH interpreter.

_FRESH_INTERPRETER_SCRIPT = """
import json, sys
sys.path.insert(0, ".")
import app.rag.corpus_ingest_cli  # exactly what scripts/e15_ingest_corpus.py imports
from pathlib import Path
from app.core import storage
from app.core.database import SessionLocal, engine
from app.rag import corpus_ingest as ci

loaded = ci.load_corpus(Path(sys.argv[1]), Path(sys.argv[2]), expected_count=3)
result = ci.run_ingest(
    engine=engine, session_factory=SessionLocal, files=loaded.files[:1], owner_id=int(sys.argv[3]),
    owner_sha256=sys.argv[4], upload_root=storage.UPLOAD_ROOT, expected_database=engine.url.database, execute=True)
print(json.dumps({"outcome": result.outcome, "documents": [[d.status, d.detail] for d in result.documents],
                  "problems": list(result.problems), "user_model_loaded": "app.models.user" in sys.modules}))
"""


def test_a_fresh_interpreter_that_imports_only_the_cli_can_ingest(ctx):
    import os
    import subprocess
    import sys

    environment = dict(os.environ, UPLOAD_DIR=str(storage.UPLOAD_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_INTERPRETER_SCRIPT, str(ctx.corpus_dir / "manifest.json"), str(ctx.corpus_dir),
         str(ctx.owner_id), ctx.digest],
        cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["user_model_loaded"] is True
    assert report["outcome"] == "ingested", report
    assert report["documents"][0][0] == "created_extracted"
    rows, _chunks = table_state()
    assert [(r[2], r[7]) for r in rows] == [("alpha-guide.md", "completed")]
    assert len(stored_files()) == 1


def test_an_orm_mapping_that_does_not_configure_is_refused_before_any_file_is_stored(ctx, monkeypatch):
    from sqlalchemy.exc import InvalidRequestError

    def broken():
        raise InvalidRequestError("expression 'User' failed to locate a name")

    monkeypatch.setattr(ci, "configure_mappers", broken)
    result = ctx.ingest()
    assert result.outcome == "refused"
    assert "the ORM models do not configure (InvalidRequestError" in " ".join(result.problems)
    assert table_state()[0] == [] and stored_files() == []
