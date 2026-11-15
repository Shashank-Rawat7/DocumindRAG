"""Unit tests for app/rag/corpus_ingest.py that need no database rows (the shared conftest still
prepares documind_test, as for every test)."""
import json
import shutil
from pathlib import Path

import pytest

from app.rag import corpus_ingest as ci
from app.rag.evaluation.parsing import parse_manifest
from app.rag.evaluation.validation import validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CORPUS = PROJECT_ROOT / "eval" / "corpus"
REAL_MANIFEST = REAL_CORPUS / "manifest.json"
SAMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample.pdf"

OWNER = 7


def make_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A small synthetic corpus (md, txt, pdf) with a valid draft manifest."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha-guide.md").write_bytes(b"# Alpha\n\nAlpha explains containers.\n")
    (corpus / "beta-notes.txt").write_bytes(b"Beta notes about startup order.\nSecond line.\n")
    shutil.copy(SAMPLE_PDF, corpus / "gamma-sample.pdf")
    (corpus / "LICENSES.md").write_bytes(b"licence notes, not a corpus document\n")
    documents = []
    for key, name, fmt in (
        ("alpha-guide", "alpha-guide.md", "md"),
        ("beta-notes", "beta-notes.txt", "txt"),
        ("gamma-sample", "gamma-sample.pdf", "pdf"),
    ):
        documents.append({
            "corpus_key": key, "title": key, "publisher": "Test", "source_url": "https://example.com/" + key,
            "license": "MIT", "license_url": "https://example.com/license", "format": fmt, "language": "en",
            "source_file_sha256": ci.hash_file(corpus / name)[0],
            "extracted_text_sha256": None, "extracted_text_length": None, "extraction": None,
            "redistributable": True,
        })
    manifest = {"schema_version": "eval-manifest-v1", "corpus_version": "eval-corpus-v1",
                "stage": "draft", "documents": documents}
    path = corpus / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, corpus


@pytest.fixture
def corpus(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    return ci.load_corpus(manifest, directory, expected_count=3), manifest, directory


def row_for(file: ci.CorpusFile, **overrides) -> dict:
    row = {
        "id": 1, "owner_id": OWNER, "filename": file.filename, "description": file.description,
        "stored_name": "stored.bin", "content_type": file.content_type, "size_bytes": file.size,
        "processing_status": "completed", "processing_error": None, "has_text": True,
        "text_sha": file.expected_text_sha256, "text_len": file.expected_text_length,
        "source_version": 1, "source_sha256": None, "indexing_status": None, "indexing_error": None,
        "indexing_lease_expires_at": None, "indexed_source_version": None, "indexed_embedding_id": None,
        "indexed_signature": None, "chunk_count": 0,
    }
    row.update(overrides)
    return row


# ---------- corpus-file mapping ----------

def derived_real_draft(tmp_path: Path) -> Path:
    """The real corpus manifest as it was before E1.5 published it: a draft, written under ``tmp_path``.

    The manifest in the repository is the published ``ingested`` one. Only ``stage`` and the three
    ingestion fields differ from the draft, so the draft is derived from it; nothing is written to the repository.
    """
    raw = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    raw["stage"] = "draft"
    for document in raw["documents"]:
        document["extracted_text_sha256"] = None
        document["extracted_text_length"] = None
        document["extraction"] = None
    directory = tmp_path / "derived-draft"
    directory.mkdir()
    path = directory / "manifest.json"
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_real_corpus_maps_six_files_to_the_expected_names(tmp_path):
    loaded = ci.load_corpus(derived_real_draft(tmp_path), REAL_CORPUS)
    mapping = {f.corpus_key: (f.filename, f.content_type, f.description) for f in loaded.files}
    assert mapping == {
        "docker-build-best-practices": ("docker-build-best-practices.md", "text/markdown",
                                        "eval-corpus-v1:docker-build-best-practices"),
        "docker-compose-startup-order": ("docker-compose-startup-order.md", "text/markdown",
                                         "eval-corpus-v1:docker-compose-startup-order"),
        "fastapi-deployment-docker-en": ("fastapi-deployment-docker-en.md", "text/markdown",
                                         "eval-corpus-v1:fastapi-deployment-docker-en"),
        "pg18-concurrency": ("postgresql-18.6-concurrency-control.pdf", "application/pdf",
                             "eval-corpus-v1:pg18-concurrency"),
        "pg18-indexes": ("postgresql-18.6-indexes.pdf", "application/pdf", "eval-corpus-v1:pg18-indexes"),
        "python314-asyncio-dev": ("python314-asyncio-dev.txt", "text/plain", "eval-corpus-v1:python314-asyncio-dev"),
    }
    # Manifest order is kept, and LICENSES.md / manifest.json are not corpus documents.
    assert [f.corpus_key for f in loaded.files] == [d.corpus_key for d in loaded.manifest.documents]
    assert loaded.manifest.stage == "draft"


def test_source_hash_and_size_come_from_the_file_bytes(corpus):
    loaded, _manifest, directory = corpus
    for file in loaded.files:
        sha, size = ci.hash_file(directory / file.filename)
        assert (file.sha256, file.size) == (sha, size)
        assert file.sha256 == ci.sha256_bytes((directory / file.filename).read_bytes())
    markdown = next(f for f in loaded.files if f.format == "md")
    # Text files: extracted-text hash and file hash coincide (a property of these files, not a rule).
    assert markdown.expected_text_sha256 == markdown.sha256
    pdf = next(f for f in loaded.files if f.format == "pdf")
    assert pdf.expected_text_sha256 != pdf.sha256


def test_changed_source_file_is_refused(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    (directory / "alpha-guide.md").write_bytes(b"# Alpha\n\nTampered.\n")
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    problems = " ".join(caught.value.problems)
    assert "alpha-guide: 0 corpus files match" in problems
    assert "unexpected file in the corpus directory: alpha-guide.md" in problems


def test_extra_file_and_wrong_count_are_refused(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    (directory / "stray.txt").write_bytes(b"not in the manifest\n")
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    assert "unexpected file in the corpus directory: stray.txt" in " ".join(caught.value.problems)
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory)  # default: six documents
    assert "expected 6" in " ".join(caught.value.problems)


def test_offline_extraction_failure_is_refused_before_any_database_use(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    blank = directory / "beta-notes.txt"
    blank.write_bytes(b"   \n\n")
    data = json.loads(manifest.read_text())
    for document in data["documents"]:
        if document["corpus_key"] == "beta-notes":
            document["source_file_sha256"] = ci.hash_file(blank)[0]
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    assert "beta-notes: offline text extraction failed (ExtractionError)" in " ".join(caught.value.problems)


def test_a_non_draft_or_invalid_manifest_does_not_load(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    data = json.loads(manifest.read_text())
    data["documents"][0]["format"] = "docx"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ci.CorpusError):
        ci.load_corpus(manifest, directory, expected_count=3)
    with pytest.raises(ci.CorpusError):
        ci.load_corpus(tmp_path / "missing.json", directory, expected_count=3)


# ---------- classification (pure) ----------

def test_a_missing_document_is_planned_for_creation(corpus):
    loaded, _m, directory = corpus
    plan = ci.classify(loaded.files[0], [], OWNER, directory)
    assert (plan.code, plan.action) == ("new", ci.ACTION_CREATE)


def test_duplicates_wrong_owner_and_identity_conflicts_are_refused(corpus):
    loaded, _m, directory = corpus
    file = loaded.files[0]
    other = loaded.files[1]
    duplicate = ci.classify(file, [row_for(file, id=1), row_for(file, id=2)], OWNER, directory)
    assert (duplicate.code, duplicate.action) == ("duplicate", ci.ACTION_REFUSE)
    # The same filename and a row with the corpus-key description but another filename are also two matches.
    split = ci.classify(file, [row_for(file, id=1), row_for(file, id=2, filename="other.md")], OWNER, directory)
    assert split.code == "duplicate"
    assert ci.classify(file, [row_for(file, owner_id=OWNER + 1)], OWNER, directory).code == "conflict_owner"
    assert ci.classify(file, [row_for(file, description="eval-corpus-v1:other")], OWNER, directory).code == "conflict_identity"
    assert ci.classify(file, [row_for(file, filename="renamed.md")], OWNER, directory).code == "conflict_identity"
    # A row that belongs to a different corpus document does not match this one.
    assert ci.classify(file, [row_for(other)], OWNER, directory).code == "new"


@pytest.mark.parametrize("state", ["pending", "processing", "failed"])
def test_partially_processed_rows_are_refused_not_retried(corpus, state):
    loaded, _m, directory = corpus
    plan = ci.classify(loaded.files[0], [row_for(loaded.files[0], processing_status=state, has_text=False)], OWNER, directory)
    assert (plan.code, plan.action) == (f"partial_{state}", ci.ACTION_REFUSE)


def test_completed_rows_are_skipped_only_when_every_hash_matches(corpus, tmp_path):
    loaded, _m, directory = corpus
    file = loaded.files[0]
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    shutil.copy(file.path, upload_root / "stored.bin")

    ok = ci.classify(file, [row_for(file)], OWNER, upload_root)
    assert (ok.code, ok.action) == ("present_extracted", ci.ACTION_SKIP)

    assert ci.classify(file, [row_for(file, text_sha="0" * 64)], OWNER, upload_root).code == "mismatch_text"
    assert ci.classify(file, [row_for(file, text_len=1)], OWNER, upload_root).code == "mismatch_text"
    assert ci.classify(file, [row_for(file, source_version=2)], OWNER, upload_root).code == "mismatch_text"
    assert ci.classify(file, [row_for(file, size_bytes=1)], OWNER, upload_root).code == "mismatch_metadata"
    assert ci.classify(file, [row_for(file, content_type="text/plain")], OWNER, upload_root).code == "mismatch_metadata"
    assert ci.classify(file, [row_for(file, stored_name="gone.bin")], OWNER, upload_root).code == "mismatch_stored_file"
    (upload_root / "stored.bin").write_bytes(b"tampered")
    assert ci.classify(file, [row_for(file)], OWNER, upload_root).code == "mismatch_stored_file"
    assert ci.classify(file, [row_for(file, stored_name="../escape.bin")], OWNER, upload_root).code == "mismatch_stored_file"


# ---------- candidate manifest ----------

def candidate_for(loaded, manifest_text):
    return ci.build_candidate(
        manifest_text, loaded.files, processing_ref="a" * 40, python_version="3.13.0",
        library_versions={"pypdf": "6.20.0"},
    )


def test_the_real_candidate_validates_and_changes_only_the_ingestion_fields(tmp_path):
    loaded = ci.load_corpus(derived_real_draft(tmp_path), REAL_CORPUS)
    text = candidate_for(loaded, loaded.manifest_text)
    assert ci.check_candidate(loaded.manifest_text, text) == []
    manifest = parse_manifest(text)
    assert validate_manifest(manifest) == []
    assert manifest.stage == "ingested"
    by_key = {d.corpus_key: d for d in manifest.documents}
    for file in loaded.files:
        document = by_key[file.corpus_key]
        assert document.extracted_text_sha256 == file.expected_text_sha256
        assert document.extracted_text_length == file.expected_text_length
        assert document.extraction.processing_ref == "a" * 40
        assert (document.extraction.pypdf is not None) == (file.format == "pdf")
    # The derived draft is the "original" here. The manifest in the repository is the published one: it stays
    # "ingested" and is a valid ingestion result of that draft.
    real_text = REAL_MANIFEST.read_text(encoding="utf-8")
    assert parse_manifest(real_text).stage == "ingested"
    assert ci.check_candidate(loaded.manifest_text, real_text) == []


def test_candidate_check_rejects_unexpected_changes(corpus):
    loaded, manifest, _directory = corpus
    text = candidate_for(loaded, loaded.manifest_text)
    assert ci.check_candidate(loaded.manifest_text, text) == []

    changed = json.loads(text)
    changed["documents"][0]["title"] = "A different title"
    assert "a field other than the three ingestion fields changed" in ci.check_candidate(
        loaded.manifest_text, json.dumps(changed))

    still_draft = json.loads(text)
    still_draft["stage"] = "draft"
    assert "the candidate manifest stage is not 'ingested'" in ci.check_candidate(
        loaded.manifest_text, json.dumps(still_draft))

    reordered = json.loads(text)
    reordered["documents"].reverse()
    assert "the document list or its order changed" in ci.check_candidate(loaded.manifest_text, json.dumps(reordered))

    bad_hash = json.loads(text)
    bad_hash["documents"][0]["extracted_text_sha256"] = "not-a-hash"
    assert ci.check_candidate(loaded.manifest_text, json.dumps(bad_hash))

    assert "the original manifest is not a draft" in ci.check_candidate(text, text)
    assert ci.check_candidate(loaded.manifest_text, "{not json")


# ---------- the driver has no indexing ----------

def test_the_driver_modules_do_not_reference_indexing_code():
    forbidden = ("indexing_runner", "indexing_service", "indexing_hook", "request_indexing", "run_indexing",
                 "backfill_execute", "index_after_extraction", "get_embedder")
    for name in ("corpus_ingest.py", "corpus_ingest_cli.py"):
        source = (PROJECT_ROOT / "app" / "rag" / name).read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in source, f"{name} mentions {word}"


def test_the_lock_key_is_a_fixed_signed_bigint():
    assert -(2**63) <= ci.LOCK_KEY < 2**63
    assert ci.LOCK_KEY == 8289887559535460316


# ---------- regression: every corpus-directory entry is inspected; links are never followed ----------

@pytest.mark.parametrize("name", ["notes.json", "data.bin", "README", ".hidden", "image.PNG", "extra.docx", "archive.tar.gz"])
def test_unexpected_files_are_refused_whatever_their_extension(tmp_path, name):
    manifest, directory = make_corpus(tmp_path)
    (directory / name).write_bytes(b"not a corpus document\n")
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    assert f"unexpected file in the corpus directory: {name}" in caught.value.problems


def test_a_subdirectory_is_refused(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    (directory / "nested").mkdir()
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    assert "unexpected non-file entry in the corpus directory: nested" in caught.value.problems


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")


def test_symbolic_links_are_refused_not_followed(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    _symlink(directory / "alias.md", directory / "alpha-guide.md")
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    assert "symbolic link in the corpus directory (refused, not followed): alias.md" in caught.value.problems


def test_a_corpus_file_replaced_by_a_link_is_refused_even_with_the_right_bytes(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    real = tmp_path / "elsewhere.md"
    (directory / "alpha-guide.md").replace(real)
    _symlink(directory / "alpha-guide.md", real)
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(manifest, directory, expected_count=3)
    problems = " ".join(caught.value.problems)
    assert "symbolic link in the corpus directory (refused, not followed): alpha-guide.md" in problems
    assert "alpha-guide: 0 corpus files match" in problems


@pytest.mark.parametrize("name", ["LICENSES.md", "manifest.json"])
def test_the_two_allowed_extra_names_may_not_be_links(tmp_path, name):
    manifest, directory = make_corpus(tmp_path)
    real = tmp_path / ("moved-" + name)
    (directory / name).replace(real)
    _symlink(directory / name, real)
    with pytest.raises(ci.CorpusError) as caught:
        ci.load_corpus(directory / "manifest.json" if name != "manifest.json" else real, directory, expected_count=3)
    assert f"symbolic link in the corpus directory (refused, not followed): {name}" in caught.value.problems


def test_only_the_manifest_files_and_the_two_allowed_names_pass(tmp_path):
    manifest, directory = make_corpus(tmp_path)
    loaded = ci.load_corpus(manifest, directory, expected_count=3)
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [f.filename for f in loaded.files] + ["LICENSES.md", "manifest.json"])


# ---------- regression: a completed document with a processing error is never "already extracted" ----------

def test_completed_with_a_processing_error_is_a_blocking_inconsistent_state(corpus, tmp_path):
    loaded, _m, _directory = corpus
    file = loaded.files[0]
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    shutil.copy(file.path, upload_root / "stored.bin")
    assert ci.classify(file, [row_for(file)], OWNER, upload_root).action == ci.ACTION_SKIP
    plan = ci.classify(file, [row_for(file, processing_error="Failed to extract text")], OWNER, upload_root)
    assert (plan.code, plan.action) == ("inconsistent_error", ci.ACTION_REFUSE)
    assert plan.document_id == 1


# ---------- chunk-offset contract: overlapping, edge-trimmed chunks (pure) ----------

def _names(results):
    return {r.name: r for r in results}


def test_overlap_and_trimmed_tail_satisfy_the_contract():
    body = "alpha beta gamma delta epsilon\n"
    ranges = [(0, 16), (11, 30)]          # "alpha beta gamma" / "gamma delta epsilon"; chunks overlap on "gamma"
    results = ci.coverage_checks(body, ranges)
    assert [r.name for r in results] == ["chunk_offsets_valid", "chunk_edges_trimmed", "chunk_coverage"]
    assert all(r.ok for r in results), results
    assert ranges[-1][1] == len(body) - 1


def test_whitespace_and_ignorable_characters_may_be_uncovered():
    # A gap of spaces/newlines, a no-break space, a zero-width space (Cf) and a leading blank line.
    body = "\n\nalpha\n\n ​beta  \n"
    ranges = [(2, 7), (11, 15)]
    assert body[2:7] == "alpha" and body[11:15] == "beta"
    assert all(r.ok for r in ci.coverage_checks(body, ranges))


def test_a_meaningful_character_in_a_gap_a_head_or_a_tail_fails_coverage_with_its_offset():
    secret = "SECRETWORD"
    gap = f"alpha {secret} beta"
    results = _names(ci.coverage_checks(gap, [(0, 5), (len(gap) - 4, len(gap))]))
    assert not results["chunk_coverage"].ok and "first at offset 6" in results["chunk_coverage"].detail
    assert secret not in results["chunk_coverage"].detail                     # offsets only, never document text

    head = _names(ci.coverage_checks("# alpha beta", [(2, 12)]))
    assert not head["chunk_coverage"].ok and "first at offset 0" in head["chunk_coverage"].detail

    tail = _names(ci.coverage_checks("alpha beta.", [(0, 10)]))
    assert not tail["chunk_coverage"].ok and "first at offset 10" in tail["chunk_coverage"].detail
    assert all(r.ok for r in ci.coverage_checks("alpha beta.", [(0, 11)]))


@pytest.mark.parametrize("ranges", [
    [],                            # no chunks
    [(-1, 5)],                     # negative start
    [(3, 3)],                      # empty range
    [(5, 3)],                      # end before start
    [(0, 99)],                     # end beyond the text
    [(0, 5), (0, 8)],              # equal starts
    [(0, 8), (3, 8)],              # equal ends
    [(3, 8), (0, 11)],             # start goes backwards
    [(0, 11), (3, 8)],             # end goes backwards
])
def test_invalid_or_non_increasing_ranges_fail_the_offset_check(ranges):
    results = ci.coverage_checks("alpha beta g", ranges)
    assert not results[0].ok and results[0].name == "chunk_offsets_valid"
    assert len(results) == 1       # the other two checks are skipped on broken ranges


def test_leading_and_trailing_whitespace_in_a_chunk_fails_the_trim_check():
    body = "alpha beta"
    trailing = _names(ci.coverage_checks(body, [(0, 6)]))          # "alpha "
    leading = _names(ci.coverage_checks(body, [(0, 5), (5, 10)]))   # " beta"
    assert not trailing["chunk_edges_trimmed"].ok and "[0]" in trailing["chunk_edges_trimmed"].detail
    assert not leading["chunk_edges_trimmed"].ok and "[1]" in leading["chunk_edges_trimmed"].detail


# ---------- the production chunker, tokenizer and configuration on the six real documents ----------

def _real_texts():
    from app.core import processing

    loaded = ci.load_corpus(REAL_MANIFEST, REAL_CORPUS)
    return [(f.corpus_key, processing.extract_text(f.path)) for f in loaded.files]


def test_production_chunker_and_config_with_a_stand_in_tokenizer_satisfy_the_contract():
    import re

    from app.rag.chunking import DEFAULT_CHUNKER_CONFIG, chunk_text

    for key, body in _real_texts():
        spans = [m.span() for m in re.finditer(r"\S+", body)]
        counts = [max(1, -(-(e - s) // 4)) for s, e in spans]          # ~4 characters per token
        chunks = chunk_text(body, spans, DEFAULT_CHUNKER_CONFIG, token_counts=counts)
        ranges = [(c.start, c.end) for c in chunks]
        results = ci.coverage_checks(body, ranges)
        assert all(r.ok for r in results), (key, results)
        if len(chunks) > 1:
            assert all(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges))), key   # overlap, never adjacent


def test_production_chunker_tokenizer_and_config_satisfy_the_contract_on_the_six_real_documents():
    try:
        from app.rag.embedder import Embedder, EmbedderConfig, EmbedderUnavailableError
        from app.rag.indexing import plan_chunks

        embedder = Embedder(EmbedderConfig.from_settings())
        planned = [(key, body, plan_chunks(body, embedder)) for key, body in _real_texts()]
    except (ImportError, EmbedderUnavailableError) as exc:
        pytest.skip(f"the pinned tokenizer/model is not available here ({type(exc).__name__})")
    for key, body, plan in planned:
        results = ci.coverage_checks(body, [(c.start, c.end) for c in plan.chunks])
        assert all(r.ok for r in results), (key, results)


def test_the_expected_embedding_id_is_what_the_embedder_stores():
    from app.rag import corpus_ingest_cli as cli
    from app.rag.embedder import Embedder, EmbedderConfig

    expected = cli.expected_embedding_id()
    assert expected == Embedder(EmbedderConfig.from_settings()).embedding_id
    assert "@" in expected and expected.split("@")[1] == EmbedderConfig.from_settings().revision
