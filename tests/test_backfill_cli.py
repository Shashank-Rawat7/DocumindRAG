"""Tests for the backfill CLI dry run (C5.5 / CP5).

Parser tests are pure. The main() tests run against the test database with an injected
Runtime, a temporary "repository" (an empty .env and an uploads directory) and patched
storage. Every test runs with the lifecycle functions and the model loader replaced by
failing stand-ins: CP5 must never index, load a model or write.
"""
import io
import re
import runpy
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

from app.core import storage
from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import backfill_selection
from app.rag import embedder as embedder_module
from app.rag import indexing_runner, indexing_service
from app.rag.backfill_cli import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_REFUSED,
    Runtime,
    count_not_completed,
    load_runtime,
    main,
    parse_arguments,
)

SECRET = "s3cr3t-p4ssw0rd-XYZ"
EMBEDDING_ID = "intfloat/multilingual-e5-small@rev1"
TEST_DATABASE = engine.url.database


# ---------- guards: CP5 may not index, load a model or write ----------

@pytest.fixture(autouse=True)
def forbid_indexing_and_models(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("CP5 must not index or load an embedding model")

    monkeypatch.setattr(indexing_service, "request_indexing", boom)
    monkeypatch.setattr(indexing_runner, "run_indexing", boom)
    monkeypatch.setattr(embedder_module.Embedder, "_ensure_loaded", boom)
    monkeypatch.setattr(embedder_module, "load_pinned_model", boom)


# ---------- helpers ----------

def make_document(
    *, text_="hello world", processing_status="completed", filename="notes.txt",
    stored_name="abc.txt", content_type="text/plain", **columns,
) -> int:
    with SessionLocal() as session:
        user = User(email=f"u{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        document = Document(
            filename=filename, owner_id=user.id, processing_status=processing_status,
            stored_name=stored_name, content_type=content_type, extracted_text=text_, **columns,
        )
        session.add(document)
        session.commit()
        return document.id


def make_indexed(**kwargs) -> int:
    return make_document(
        indexing_status="indexed", indexed_source_version=1, source_sha256="a" * 64,
        indexed_signature="{}", indexed_embedding_id=EMBEDDING_ID,
        indexed_at=datetime.now(timezone.utc), **kwargs,
    )


def add_chunk(document_id: int) -> None:
    with SessionLocal() as session:
        session.add(DocumentChunk(
            document_id=document_id, chunk_index=0, start_offset=0, end_offset=5,
            text="hello", span_sha256="0" * 64, embedding=[0.0] * 384,
        ))
        session.commit()


def db_snapshot():
    with engine.connect() as connection:
        documents = [tuple(r) for r in connection.execute(text("SELECT id, xmin::text FROM documents ORDER BY id"))]
        chunks = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
        users = connection.execute(text("SELECT count(*) FROM users")).scalar_one()
    return documents, chunks, users


def tree(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.is_dir()) for p in root.rglob("*"))


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / ".env").write_text("", encoding="utf-8")  # created by the test, empty
    uploads = root / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(storage, "UPLOAD_ROOT", uploads)
    return root


def make_runtime(repo: Path, **overrides) -> Runtime:
    values = dict(
        engine=engine,
        database_url=f"postgresql://documind_user:{SECRET}@localhost:5434/{TEST_DATABASE}",
        environment="development",
        environment_explicit=True,
        upload_dir="uploads",
        upload_root=repo / "uploads",
        embedding_id=EMBEDDING_ID,
    )
    values.update(overrides)
    return Runtime(**values)


def invoke(repo: Path, *argv, runtime=None, runtime_loader=None, environ=None, cwd=None):
    out, err = io.StringIO(), io.StringIO()
    loader = runtime_loader or (lambda: runtime if runtime is not None else make_runtime(repo))
    code = main(
        list(argv),
        runtime_loader=loader,
        environ={} if environ is None else environ,
        cwd=repo if cwd is None else cwd,
        repo_root=repo,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


class NoConnectEngine:
    """An engine that must never be used."""

    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True
        raise AssertionError("must not connect")


class FailingEngine:
    def __init__(self, error):
        self.error = error

    def connect(self):
        raise self.error


def section(text_: str, title: str) -> list[str]:
    lines = text_.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    body = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        body.append(line.strip())
    return body


def value(text_: str, label: str) -> str:
    match = re.search(rf"^\s+{re.escape(label)}\s*:\s*(.+)$", text_, re.M)
    assert match, f"no summary row {label!r}"
    return match.group(1).strip()


def has_doc(lines, document_id) -> bool:
    return any(re.match(rf"^doc {document_id}\s", line) for line in lines)


def no_select(monkeypatch):
    """Replace the selection with a failing stand-in; returns the list of calls."""
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise AssertionError("the selection must not run")

    monkeypatch.setattr(backfill_selection, "select_document_states", boom)
    return calls


# ---------- argument parsing ----------

def test_the_defaults_are_a_dry_run_with_no_database_and_no_limit():
    args = parse_arguments([])
    assert (args.execute, args.database, args.include_failed, args.limit) == (False, None, False, None)


def test_valid_arguments_are_parsed():
    args = parse_arguments(["--database", "documind_db", "--include-failed", "--limit", "3"])
    assert (args.database, args.include_failed, args.limit, args.execute) == ("documind_db", True, 3, False)
    assert parse_arguments(["--execute", "--database", "x"]).execute is True


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", ""])
def test_an_invalid_limit_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit) as info:
        parse_arguments(["--limit", bad])
    assert info.value.code == 2
    capsys.readouterr()


@pytest.mark.parametrize("bad", ["", "   "])
def test_a_blank_database_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit) as info:
        parse_arguments(["--database", bad])
    assert info.value.code == 2
    capsys.readouterr()


def test_execute_without_a_database_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as info:
        parse_arguments(["--execute"])
    assert info.value.code == 2
    assert "--database" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--exec", "--incl", "--lim=3", "--bogus"])
def test_abbreviations_and_unknown_flags_are_rejected(flag, capsys):
    with pytest.raises(SystemExit) as info:
        parse_arguments([flag])
    assert info.value.code == 2
    capsys.readouterr()


def test_help_exits_zero_and_lists_the_options(capsys):
    with pytest.raises(SystemExit) as info:
        parse_arguments(["--help"])
    assert info.value.code == 0
    shown = capsys.readouterr().out
    for flag in ("--database", "--include-failed", "--limit", "--execute"):
        assert flag in shown


@pytest.mark.parametrize("argv, expected", [(["--help"], 0), (["--limit", "0"], 2), (["--execute"], 2), (["--nope"], 2)])
def test_main_maps_parser_exits_to_codes_without_loading_configuration(argv, expected, capsys):
    def loader():
        raise AssertionError("the configuration must not be loaded for argument errors")

    assert main(argv, runtime_loader=loader) == expected
    capsys.readouterr()


# ---------- a default dry run ----------

def test_a_default_dry_run_reports_and_changes_nothing(repo):
    make_document()
    add_chunk(make_indexed())
    before_db, before_files = db_snapshot(), tree(repo)

    code, out, err = invoke(repo)

    assert code == EXIT_OK
    assert err == ""
    assert "DRY RUN" in out and "Dry run: nothing was changed." in out
    assert "not verified (no --database given)" in out
    assert db_snapshot() == before_db
    assert tree(repo) == before_files


def test_a_dry_run_with_the_right_database_verifies_it(repo):
    make_document()
    code, out, err = invoke(repo, "--database", TEST_DATABASE)
    assert code == EXIT_OK and err == ""
    assert f"verified as '{TEST_DATABASE}' (live check)" in out


def test_the_target_line_shows_host_database_and_user_but_no_password(repo):
    code, out, err = invoke(repo)
    assert f"localhost:5434/{TEST_DATABASE} as documind_user" in out
    assert SECRET not in out + err


# ---------- data scenarios ----------

def seed_scenario(repo: Path) -> dict[str, int]:
    (repo / "uploads" / "ok.pdf").write_bytes(b"not parsed")
    ids = {
        "text": make_document(stored_name="a1.txt"),
        "pdf_ok": make_document(filename="ok.pdf", stored_name="ok.pdf", content_type="application/pdf"),
        "pdf_missing": make_document(filename="gone.pdf", stored_name="gone.pdf", content_type="application/pdf"),
        "failed": make_document(indexing_status="failed", indexing_error="boom"),
        "current": make_indexed(),
        "extraction_pending": make_document(processing_status="pending"),
        "extraction_failed": make_document(processing_status="failed"),
        "no_status": make_document(processing_status=None),
    }
    add_chunk(ids["current"])
    return ids


def test_each_document_lands_in_the_right_section(repo):
    ids = seed_scenario(repo)
    code, out, err = invoke(repo)

    assert code == EXIT_OK  # blocked and reported items do not fail a dry run
    assert err == ""
    will_index = section(out, "Will index")
    assert has_doc(will_index, ids["text"]) and has_doc(will_index, ids["pdf_ok"])
    blocked = section(out, "Blocked")
    assert has_doc(blocked, ids["pdf_missing"]) and any("pdf_file_missing" in line for line in blocked)
    assert has_doc(section(out, "Reported"), ids["failed"])
    assert not re.search(rf"^\s*doc {ids['current']}\s", out, re.M)  # skipped items are only counted

    assert value(out, "Completed documents examined") == "5"
    assert value(out, "Will index") == "2"
    assert value(out, "Blocked") == "1"
    assert value(out, "Reported (needs attention)") == "1"
    assert value(out, "Skipped (nothing to do)") == "1"
    assert value(out, "Not examined (extraction not completed)") == "3 (failed 1, pending 1, no status 1)"


def test_include_failed_moves_the_failed_document_into_will_index(repo):
    ids = seed_scenario(repo)
    code, out, err = invoke(repo, "--include-failed")
    assert code == EXIT_OK
    assert has_doc(section(out, "Will index"), ids["failed"])
    assert section(out, "Reported") == ["none"]
    assert "include-failed=yes" in out


def test_the_limit_defers_only_index_candidates(repo):
    candidates = [make_document() for _ in range(4)]
    blocked = make_document(filename="gone.pdf", stored_name="gone.pdf", content_type="application/pdf")
    reported = make_document(indexing_status="failed", indexing_error="boom")

    code, out, err = invoke(repo, "--limit", "2")

    assert code == EXIT_OK
    assert "limit=2" in out
    assert [bool(has_doc(section(out, "Will index"), i)) for i in candidates] == [True, True, False, False]
    deferred = section(out, "Deferred by --limit 2")
    assert has_doc(deferred, candidates[2]) and has_doc(deferred, candidates[3])
    assert value(out, "Deferred by --limit") == "2"
    assert has_doc(section(out, "Blocked"), blocked)      # never deferred
    assert has_doc(section(out, "Reported"), reported)    # never deferred


def test_a_limit_larger_than_the_candidate_count_defers_nothing(repo):
    make_document()
    make_document()
    code, out, err = invoke(repo, "--limit", "50")
    assert code == EXIT_OK
    assert value(out, "Deferred by --limit") == "0"
    assert value(out, "Will index") == "2"


def test_count_not_completed_groups_by_status_and_includes_null():
    make_document()
    make_document(processing_status="pending")
    make_document(processing_status="failed")
    make_document(processing_status="failed")
    make_document(processing_status=None)
    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        assert count_not_completed(connection) == {"pending": 1, "failed": 2, None: 1}
        connection.rollback()


# ---------- safety refusals: exit 2 and no connection at all ----------

def _remove_env_file(repo, tmp_path):
    (repo / ".env").unlink()
    return {}


def _remove_uploads(repo, tmp_path):
    (repo / "uploads").rmdir()
    return {}


REFUSALS = {
    "production": lambda repo, tmp: {"runtime": dict(environment="production")},
    "no-environment": lambda repo, tmp: {"runtime": dict(environment=None)},
    "implicit-environment": lambda repo, tmp: {"runtime": dict(environment_explicit=False)},
    "remote-host": lambda repo, tmp: {"runtime": dict(database_url=f"postgresql://u:{SECRET}@db.example.com:5432/{TEST_DATABASE}")},
    "url-query": lambda repo, tmp: {"runtime": dict(database_url=f"postgresql://u:{SECRET}@localhost:5434/{TEST_DATABASE}?sslmode=disable")},
    "malformed-url": lambda repo, tmp: {"runtime": dict(database_url=f"://{SECRET}")},
    "other-driver": lambda repo, tmp: {"runtime": dict(database_url=f"sqlite://u:{SECRET}@localhost/x")},
    "missing-env-file": lambda repo, tmp: (_remove_env_file(repo, tmp), {})[1],
    "missing-uploads": lambda repo, tmp: (_remove_uploads(repo, tmp), {})[1],
    "wrong-directory": lambda repo, tmp: {"cwd": tmp},
    "libpq-variable": lambda repo, tmp: {"environ": {"PGHOSTADDR": "10.0.0.5"}},
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_unsafe_configurations_are_refused_without_connecting(case, repo, tmp_path):
    blocker = NoConnectEngine()
    options = REFUSALS[case](repo, tmp_path)
    runtime = make_runtime(repo, engine=blocker, **options.get("runtime", {}))

    code, out, err = invoke(repo, runtime=runtime, environ=options.get("environ"), cwd=options.get("cwd"))

    assert code == EXIT_REFUSED
    assert "REFUSED" in err and "Nothing was changed." in err
    assert out == ""
    assert blocker.connected is False
    assert SECRET not in out + err


def test_a_static_database_name_mismatch_is_refused_without_connecting(repo, monkeypatch):
    calls = no_select(monkeypatch)
    blocker = NoConnectEngine()
    code, out, err = invoke(repo, "--database", "documind_prod", runtime=make_runtime(repo, engine=blocker))
    assert code == EXIT_REFUSED
    assert "documind_prod" in err and TEST_DATABASE in err
    assert blocker.connected is False and calls == []


def test_a_live_database_mismatch_is_refused_before_any_selection(repo, monkeypatch):
    calls = no_select(monkeypatch)
    # The URL names documind_x, so the static check passes; the real server is the test database.
    runtime = make_runtime(repo, database_url=f"postgresql://u:{SECRET}@localhost:5434/documind_x")
    before = db_snapshot()
    code, out, err = invoke(repo, "--database", "documind_x", runtime=runtime)
    assert code == EXIT_REFUSED
    assert "documind_x" in err and TEST_DATABASE in err
    assert out == "" and calls == []
    assert db_snapshot() == before


# ---------- errors and interrupts ----------

def test_a_configuration_failure_is_refused_with_only_the_class_name(repo):
    def loader():
        raise ValueError(f"bad config {SECRET}")

    code, out, err = invoke(repo, runtime_loader=loader)
    assert code == EXIT_REFUSED
    assert "ValueError" in err and SECRET not in out + err


def test_a_connect_error_with_a_database_is_a_refusal(repo):
    error = OperationalError("SELECT 1", {}, Exception(f"postgresql://u:{SECRET}@10.0.0.5/db"))
    code, out, err = invoke(repo, "--database", TEST_DATABASE, runtime=make_runtime(repo, engine=FailingEngine(error)))
    assert code == EXIT_REFUSED
    assert "OperationalError" in err and SECRET not in out + err and "SELECT" not in err


def test_a_connect_error_without_a_database_is_an_operational_failure(repo):
    error = OperationalError("SELECT 1", {}, Exception(SECRET))
    code, out, err = invoke(repo, runtime=make_runtime(repo, engine=FailingEngine(error)))
    assert code == EXIT_FAILURE
    assert "OperationalError" in err and SECRET not in out + err


def test_a_selection_error_is_an_operational_failure_with_only_the_class_name(repo, monkeypatch):
    def broken(connection, **kwargs):
        raise RuntimeError(f"postgresql://u:{SECRET}@host/db")

    monkeypatch.setattr(backfill_selection, "select_document_states", broken)
    code, out, err = invoke(repo)
    assert code == EXIT_FAILURE
    assert "RuntimeError" in err and "nothing was changed" in err
    assert out == "" and SECRET not in err


def test_a_keyboard_interrupt_exits_130(repo, monkeypatch):
    def interrupted(connection, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(backfill_selection, "select_document_states", interrupted)
    code, out, err = invoke(repo)
    assert code == EXIT_INTERRUPTED
    assert "Interrupted" in err and out == ""


# ---------- --execute refusals that never reach execution ----------

def test_execute_still_runs_the_live_check(repo):
    runtime = make_runtime(repo, database_url=f"postgresql://u:{SECRET}@localhost:5434/documind_x")
    code, out, err = invoke(repo, "--execute", "--database", "documind_x", runtime=runtime)
    assert code == EXIT_REFUSED
    assert "REFUSED" in err and "documind_x" in err


def test_execute_without_a_database_is_a_usage_error_through_main(repo, capsys):
    code, out, err = invoke(repo, "--execute")
    assert code == 2
    capsys.readouterr()


# ---------- the connection is read-only ----------

def test_the_selection_receives_a_read_only_connection(repo, monkeypatch):
    make_document()
    before = db_snapshot()
    attempts = []

    def probing(connection, **kwargs):
        with pytest.raises(DBAPIError) as info:
            connection.execute(text("UPDATE documents SET indexing_error = 'x'"))
        attempts.append(info.value.orig.pgcode)
        connection.rollback()
        return []

    monkeypatch.setattr(backfill_selection, "select_document_states", probing)
    code, out, err = invoke(repo)

    assert code == EXIT_OK
    assert attempts == ["25006"]  # read_only_sql_transaction
    assert db_snapshot() == before


# ---------- no secrets in any output ----------

def test_no_scenario_ever_prints_the_password(repo):
    seed_scenario(repo)
    outputs = []
    for argv in ([], ["--include-failed"], ["--limit", "1"], ["--database", TEST_DATABASE], ["--database", "other"]):
        code, out, err = invoke(repo, *argv)
        outputs.append(out + err)
    assert all(SECRET not in text_ for text_ in outputs)
    assert all(text_.isascii() for text_ in outputs)


# ---------- load_runtime ----------

def test_load_runtime_derives_the_embedding_id_without_loading_a_model(monkeypatch):
    monkeypatch.setattr(embedder_module, "_EMBEDDER", None)  # a fresh, construct-only singleton
    runtime = load_runtime()

    assert runtime.embedding_id == f"{settings.embedding_model_id}@{settings.embedding_model_revision}"
    assert runtime.engine is engine
    assert runtime.upload_root == storage.UPLOAD_ROOT
    assert runtime.environment == settings.environment
    assert runtime.environment_explicit is True  # conftest sets ENVIRONMENT
    assert runtime.embedder is embedder_module.get_embedder()
    assert runtime.embedder.embedding_id == runtime.embedding_id
    assert runtime.session_factory is SessionLocal
    assert embedder_module.get_embedder().is_loaded is False


def test_the_runtime_repr_hides_the_url_and_engine():
    runtime = make_runtime(Path("."))
    shown = repr(runtime)
    assert SECRET not in shown and "documind_user" not in shown and "Engine" not in shown


# ---------- the script entry point ----------

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backfill_indexing.py"


def test_the_script_prints_help_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["backfill_indexing.py", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert info.value.code == 0
    assert "--limit" in capsys.readouterr().out


def test_the_script_exits_two_on_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["backfill_indexing.py", "--limit", "0"])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert info.value.code == 2
    capsys.readouterr()