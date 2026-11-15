"""Tests for the corpus ingestion command line (app/rag/corpus_ingest_cli.py) against documind_test.

The runtime is injected: a temporary "repository" with a .env marker and an uploads folder, an
environment that says "development", and the real test engine. A synthetic six-document corpus is
used; no embedding model is loaded and no indexing code is run (published rows are written with SQL).
"""
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

import pytest
from sqlalchemy import text

from app.core import storage
from app.core.database import SessionLocal, engine
from app.models.user import User
from app.rag import corpus_ingest as ci
from app.rag import corpus_ingest_cli as cli
from app.rag.chunking import ChunkerConfig, chunk_text
from app.rag.evaluation.parsing import parse_manifest
from app.rag.evaluation.validation import validate_manifest
from app.rag.snapshot_cli import Runtime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample.pdf"
EVAL_EMAIL = "evaluation@example.com"
EMBEDDING_ID = "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIGEST = hashlib.sha256(EVAL_EMAIL.encode()).hexdigest()
COMMIT = "c" * 40

DOCUMENTS = (
    ("alpha-guide", "alpha-guide.md", "md", b"# Alpha\n\nAlpha explains containers.\n"),
    ("beta-notes", "beta-notes.txt", "txt", b"Beta notes about startup order.\nSecond line.\n"),
    ("gamma-sample", "gamma-sample.pdf", "pdf", None),
    ("delta-guide", "delta-guide.md", "md", b"# Delta\n\nDelta is about volumes.\n"),
    ("epsilon-notes", "epsilon-notes.txt", "txt", b"Epsilon notes about networks.\n"),
    ("zeta-guide", "zeta-guide.md", "md", b"# Zeta\n\nZeta is about secrets.\n"),
)


def make_corpus(directory: Path) -> Path:
    directory.mkdir(parents=True)
    documents = []
    for key, name, fmt, content in DOCUMENTS:
        if content is None:
            shutil.copy(SAMPLE_PDF, directory / name)
        else:
            (directory / name).write_bytes(content)
        documents.append({
            "corpus_key": key, "title": key, "publisher": "Test", "source_url": "https://example.com/" + key,
            "license": "MIT", "license_url": "https://example.com/license", "format": fmt, "language": "en",
            "source_file_sha256": ci.hash_file(directory / name)[0],
            "extracted_text_sha256": None, "extracted_text_length": None, "extraction": None,
            "redistributable": True,
        })
    (directory / "LICENSES.md").write_bytes(b"licences\n")
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": "eval-manifest-v1", "corpus_version": "eval-corpus-v1",
                                "stage": "draft", "documents": documents}, indent=2) + "\n", encoding="utf-8")
    return path


class Env:
    def __init__(self, tmp_path, monkeypatch, make_user, db):
        make_user(EVAL_EMAIL)
        make_user("someone-else@example.com")
        self.owner_id = db.query(User).filter(User.email == EVAL_EMAIL).one().id
        self.other_id = db.query(User).filter(User.email == "someone-else@example.com").one().id
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        (self.repo / ".env").write_text("placeholder\n")
        self.uploads = self.repo / "uploads"
        self.uploads.mkdir()
        monkeypatch.setattr(storage, "UPLOAD_ROOT", self.uploads.resolve())
        self.manifest = make_corpus(tmp_path / "corpus")
        self.corpus = self.manifest.parent
        self.outside = tmp_path / "outside"
        self.outside.mkdir()
        self.runtime = Runtime(
            engine=engine, database_url=str(engine.url.render_as_string(hide_password=False)),
            environment="development", environment_explicit=True, upload_dir="uploads",
            upload_root=self.uploads.resolve(),
        )
        self.database = engine.url.database

    def run(self, *args, runtime=None, environ=None, database=None, owner=None, digest=None, common=True):
        argv = list(args)
        if common:
            argv += ["--database", database or self.database, "--owner-id", str(owner or self.owner_id),
                     "--evaluation-email-sha256", digest or DIGEST,
                     "--manifest", str(self.manifest), "--corpus-dir", str(self.corpus)]
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(
            argv,
            runtime_loader=lambda: runtime or self.runtime,
            tools_loader=lambda: cli.Tools(session_factory=SessionLocal, embedding_id=EMBEDDING_ID),
            environ={} if environ is None else environ,
            cwd=self.repo, repo_root=self.repo, stdout=out, stderr=err,
        )
        return code, out.getvalue(), err.getvalue()

    def authorization(self):
        return cli.authorization_phrase(self.database, self.owner_id)

    def ingest(self):
        return self.run("ingest", "--execute", "--authorization", self.authorization())

    def state(self):
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT id, filename, processing_status FROM documents ORDER BY id")).all()
        return [tuple(r) for r in rows], sorted(p.name for p in self.uploads.iterdir())

    def manifest_sha(self):
        return hashlib.sha256(self.manifest.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch, make_user, db):
    return Env(tmp_path, monkeypatch, make_user, db)


_SMALL_CONFIG = ChunkerConfig(target_tokens=3, max_tokens=4, overlap_tokens=1)


def publish_all(documents_ids):
    """Stand-in for the indexer's publish step: overlapping, edge-trimmed chunks plus the indexed metadata."""
    for document_id in documents_ids:
        with engine.begin() as conn:
            body = conn.execute(text("SELECT extracted_text FROM documents WHERE id = :i"), {"i": document_id}).scalar_one()
            spans = [m.span() for m in re.finditer(r"\S+", body)]
            for index, chunk in enumerate(chunk_text(body, spans, _SMALL_CONFIG)):
                conn.execute(text(
                    "INSERT INTO document_chunks (document_id, chunk_index, start_offset, end_offset, page_start, "
                    "page_end, text, span_sha256, embedding) "
                    "SELECT d.id, :n, :s, :e, 1, 1, substr(d.extracted_text, :s + 1, :e - :s), "
                    "encode(sha256(convert_to(substr(d.extracted_text, :s + 1, :e - :s), 'UTF8')), 'hex'), "
                    "array_fill(0::real, ARRAY[384])::vector FROM documents d WHERE d.id = :i"
                ), {"n": index, "s": chunk.start, "e": chunk.end, "i": document_id})
            conn.execute(text(
                "UPDATE documents SET indexing_status = 'indexed', indexed_source_version = 1, "
                "source_sha256 = encode(sha256(convert_to(extracted_text,'UTF8')),'hex'), indexed_signature = 'sig', "
                "indexed_embedding_id = :e, indexed_at = now(), indexing_attempt = 1 WHERE id = :i"
            ), {"e": EMBEDDING_ID, "i": document_id})


# ---------- the command surface ----------

def test_there_are_exactly_four_commands_and_no_indexing_command():
    parser = cli.build_parser()
    commands = next(a for a in parser._actions if a.dest == "command").choices
    assert sorted(commands) == ["candidate", "ingest", "plan", "verify"]
    assert not any("index" in name for name in commands)


def test_abbreviated_and_unknown_options_are_rejected(env):
    assert env.run("ingest", "--exec")[0] == 2
    assert env.run("index")[0] == 2
    assert env.run("ingest", "--authorization", "x")[0] == 2        # meaningless without --execute
    assert env.run("verify", "--execute")[0] == 2
    assert env.state() == ([], [])


def test_the_default_command_is_read_only(env):
    code, out, err = env.run("ingest")
    assert code == 0, err
    assert "dry run: nothing was written" in out
    assert "6 to create" in out
    assert env.state() == ([], [])


# ---------- plan ----------

def test_plan_is_read_only_and_describes_every_document(env):
    before = env.manifest_sha()
    code, out, err = env.run("plan")
    assert code == 0, err
    for key, *_ in DOCUMENTS:
        assert f"{key}: new -> create" in out
    assert env.state() == ([], [])
    assert env.manifest_sha() == before


def test_plan_reports_a_blocking_state_with_a_failure_exit_code(env):
    assert env.ingest()[0] == 0
    stored = sorted(env.uploads.iterdir())[0]
    stored.write_bytes(b"tampered")
    code, out, err = env.run("plan")
    assert code == 1
    assert "mismatch_stored_file -> refuse" in out
    assert "Recovery is manual" in err


# ---------- ingest ----------

def test_execute_requires_the_exact_authorization(env):
    for args in (("ingest", "--execute"), ("ingest", "--execute", "--authorization", "yes"),
                 ("ingest", "--execute", "--authorization", f"ingest-{env.database}-owner-{env.other_id}")):
        code, _out, err = env.run(*args)
        assert code == 2 and "--authorization" in err
    assert env.state() == ([], [])


def test_authorized_ingest_creates_all_six_and_a_second_run_does_nothing(env):
    code, out, err = env.ingest()
    assert code == 0, err
    assert out.count("created_extracted") == 6
    assert "NOT indexed" in out
    rows, files = env.state()
    assert len(rows) == 6 and all(r[2] == "completed" for r in rows) and len(files) == 6

    code, out, err = env.ingest()
    assert code == 0 and "nothing to do" in out
    assert env.state() == (rows, files)


def test_wrong_database_name_is_refused_by_the_static_checks(env):
    code, _out, err = env.run("plan", database="documind_db")
    assert code == 2 and "database_name" in err
    code, _out, err = env.run("ingest", "--execute", "--authorization",
                              cli.authorization_phrase("documind_db", env.owner_id), database="documind_db")
    assert code == 2 and "database_name" in err
    assert env.state() == ([], [])


@pytest.mark.parametrize("environment, explicit", [("production", True), ("development", False), (None, False)])
def test_a_non_development_environment_is_refused(env, environment, explicit):
    runtime = Runtime(engine=engine, database_url=env.runtime.database_url, environment=environment,
                      environment_explicit=explicit, upload_dir="uploads", upload_root=env.uploads.resolve())
    code, _out, err = env.run("ingest", "--execute", "--authorization", env.authorization(), runtime=runtime)
    assert code == 2 and "[environment]" in err
    assert env.state() == ([], [])


def test_unsafe_runtime_settings_are_refused(env):
    remote = Runtime(engine=engine, database_url="postgresql://u:p@db.example.com:5432/" + env.database,
                     environment="development", environment_explicit=True, upload_dir="uploads",
                     upload_root=env.uploads.resolve())
    assert "[database_url]" in env.run("plan", runtime=remote)[2]
    assert "[database_url]" in env.run("plan", environ={"PGHOST": "elsewhere"})[2]
    wrong_uploads = Runtime(engine=engine, database_url=env.runtime.database_url, environment="development",
                            environment_explicit=True, upload_dir="uploads", upload_root=env.outside.resolve())
    assert "[upload_directory]" in env.run("plan", runtime=wrong_uploads)[2]
    (env.repo / ".env").unlink()
    assert "[env_file]" in env.run("plan")[2]
    assert env.state() == ([], [])


def test_wrong_owner_is_refused(env):
    for owner, digest in ((env.other_id, None), (env.owner_id, "1" * 64), (424242, None)):
        code, out, _err = env.run(
            "ingest", "--execute", "--authorization", cli.authorization_phrase(env.database, owner),
            owner=owner, digest=digest)
        assert code == 1 and "evaluation owner does not exist or its email digest does not match" in out
    assert env.state() == ([], [])


def test_a_corpus_that_does_not_load_is_refused_before_the_database_is_used(env):
    (env.corpus / "beta-notes.txt").write_bytes(b"changed\n")
    code, _out, err = env.run("plan")
    assert code == 2 and "the corpus did not load" in err
    assert env.state() == ([], [])


def test_an_ingested_manifest_is_not_accepted_for_ingestion(env):
    data = json.loads(env.manifest.read_text())
    data["stage"] = "ingested"
    for number, document in enumerate(data["documents"], start=1):
        document["extracted_text_sha256"] = f"{number:064x}"
        document["extracted_text_length"] = 1
        document["extraction"] = {"processing_ref": COMMIT, "python": "3.13.0",
                                  **({"pypdf": "1"} if document["format"] == "pdf" else {})}
    env.manifest.write_text(json.dumps(data, indent=2) + "\n")
    code, _out, err = env.run("plan")
    assert code == 2 and "'draft'" in err


def test_a_concurrent_run_is_refused_with_exit_code_two(env):
    with ci.advisory_lock(engine) as held:
        assert held
        code, _out, err = env.ingest()
    assert code == 2 and "another run holds the ingestion lock" in err
    assert env.state() == ([], [])


def test_an_extraction_failure_stops_with_exit_code_one(env, monkeypatch):
    from app.core import processing

    def failing(document_id):
        session = SessionLocal()
        try:
            processing._mark_failed(session, document_id, "Failed to extract text from this file")
        finally:
            session.close()

    monkeypatch.setattr(processing, "process_document", failing)
    code, out, err = env.ingest()
    assert code == 1 and "STOPPED" in err and "failed_extraction" in out and out.count("not_attempted") == 5
    rows, _files = env.state()
    assert [r[2] for r in rows] == ["failed"]


# ---------- verify and candidate ----------

def test_verify_fails_until_the_documents_are_indexed(env):
    assert env.ingest()[0] == 0
    code, out, _err = env.run("verify")
    assert code == 1 and "verification: FAILED (0/6 documents)" in out
    publish_all([r[0] for r in env.state()[0]])
    code, out, err = env.run("verify")
    assert code == 0, out + err
    assert "verification: PASSED (6/6 documents)" in out
    assert "pdf_page_numbers_reported" in out


def test_candidate_is_not_written_when_verification_fails(env):
    assert env.ingest()[0] == 0
    target = env.outside / "candidate.json"
    code, _out, err = env.run("candidate", "--processing-ref", COMMIT, "--out", str(target))
    assert code == 1 and "Not generating a candidate manifest" in err
    assert not target.exists()


def test_candidate_is_written_validated_and_never_published(env):
    assert env.ingest()[0] == 0
    publish_all([r[0] for r in env.state()[0]])
    before = env.manifest_sha()
    target = env.outside / "candidate.json"
    code, out, err = env.run("candidate", "--processing-ref", COMMIT, "--out", str(target))
    assert code == 0, out + err
    assert "NOT published" in out
    text_ = target.read_text(encoding="utf-8")
    assert ci.sha256_bytes(text_.encode("utf-8")) in out
    manifest = parse_manifest(text_)
    assert validate_manifest(manifest) == [] and manifest.stage == "ingested"
    assert ci.check_candidate(env.manifest.read_text(encoding="utf-8"), text_) == []
    assert all(d.extraction.processing_ref == COMMIT for d in manifest.documents)
    assert env.manifest_sha() == before                       # the manifest was not modified


def test_candidate_output_path_is_checked(env):
    assert env.ingest()[0] == 0
    publish_all([r[0] for r in env.state()[0]])
    existing = env.outside / "exists.json"
    existing.write_text("keep me")
    cases = {
        "relative": "candidate.json",
        "inside_repo": str(env.repo / "candidate.json"),
        "the_manifest": str(env.manifest),
        "existing": str(existing),
        "missing_folder": str(env.outside / "nope" / "candidate.json"),
    }
    for name, path in cases.items():
        code, _out, err = env.run("candidate", "--processing-ref", COMMIT, "--out", path)
        assert code == 2, name
        assert "REFUSED" in err, name
    assert existing.read_text() == "keep me"
    assert not (env.repo / "candidate.json").exists()


def test_candidate_needs_a_full_commit_id(env):
    assert env.run("candidate", "--processing-ref", "abc", "--out", str(env.outside / "c.json"))[0] == 2


def test_the_cli_expects_the_full_model_at_revision_embedding_id():
    assert cli.load_tools().embedding_id == cli.expected_embedding_id() == EMBEDDING_ID
    assert "@" in cli.expected_embedding_id()
