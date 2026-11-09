"""Tests for --execute wiring in the backfill CLI (C5.5 / CP6c.2a).

Safety, ordering and exit codes. The real test database, the real wiring, request_indexing
and run_indexing are used with a FAKE embedder; the real embedding model is never loaded
(autouse guards fail the test if it is requested). The full indexing lifecycle (failures,
races, interruption, leases) belongs to CP6c.2b and is deliberately not here.

Helpers are local to this file, as in the other backfill test modules.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path



import numpy as np
import pytest

from sqlalchemy import create_engine, event, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core import storage
from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import backfill_cli, backfill_selection, indexing_runner
from app.rag import embedder as embedder_module
from app.rag.backfill_cli import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_REFUSED,
    MESSAGE_NO_EXECUTION_RUNTIME,
    NOTE_OFFLINE_SET_LATE,
    Runtime,
    build_parser,
    exit_code_for,
    main,
)
from app.rag.backfill import Category, SourceStatus
from app.rag.backfill_execute import DETAIL_NO_LONGER_SELECTED, ExecutionSummary, ItemOutcome, ItemResult
from app.rag.indexing import source_sha256
from app.rag.indexing_service import FailureReason
from app.rag.backfill_execute import ExecutionSummary, ItemOutcome, ItemResult
from app.rag.backfill_report import EXECUTE_FOOTER, EXECUTE_MODE
from app.rag.backfill_wiring import FAILURE_RUNNER_FACTORY_MISMATCH, FAILURE_SESSION_FACTORY_MISMATCH
from app.rag.embedder import EMBEDDING_DIM
from app.rag.text_boundaries import is_ignorable
from app.rag.token_offsets import map_token_offsets
from app.rag import backfill_cli, backfill_selection, indexing_runner, indexing_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET = "s3cr3t-p4ssw0rd-XYZ"
TEST_DATABASE = engine.url.database
FAKE_EMBEDDING_ID = "test/fake@" + "0" * 40
TEXT = " ".join(f"word{k:04d}" for k in range(700))  # the same known-good input as the runner tests


# ---------- guards: the real model must never be touched ----------

@pytest.fixture(autouse=True)
def forbid_the_real_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the real embedding model must not be requested")

    monkeypatch.setattr(embedder_module.Embedder, "_ensure_loaded", boom)
    monkeypatch.setattr(embedder_module, "load_pinned_model", boom)
    monkeypatch.setattr(embedder_module, "get_embedder", boom)
    monkeypatch.setattr(indexing_runner, "get_embedder", boom)


# ---------- fake embedder (same shape as tests/test_indexing_runner.py) ----------

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
    """The members the preflight and build_document_index use.

    ``probe(name)`` runs on every call of every method; ``hooks`` maps a method name to a
    callable that runs once, on the first call of that method. ``signature_dim`` and
    ``embed_result`` exist to make the preflight fail in specific ways.
    """

    embedding_id = FAKE_EMBEDDING_ID

    def __init__(self, *, embed_error=None, signature_dim=EMBEDDING_DIM, embed_result=None,
                 probe=None, hooks=None):
        self.embed_error = embed_error
        self.signature_dim = signature_dim
        self.embed_result = embed_result
        self.probe = probe
        self.hooks = dict(hooks or {})
        self.calls = []
        self._fired = set()

    def _enter(self, name, size):
        self.calls.append((name, size))
        if self.probe is not None:
            self.probe(name)
        hook = self.hooks.get(name)
        if hook is not None and name not in self._fired:
            self._fired.add(name)
            hook()

    def signature_fields(self):
        self._enter("signature_fields", 0)
        return {"model_id": "test/fake", "embedding_dim": self.signature_dim}

    def source_units(self, value):
        self._enter("source_units", len(value))
        spans = word_spans(value)
        return map_token_offsets(value, spans, [value[start:end] for start, end in spans])

    def embed_passages(self, texts):
        self._enter("embed_passages", len(texts))
        if self.embed_error is not None:
            raise self.embed_error
        if self.embed_result is not None:
            return self.embed_result
        return np.stack([fake_vector(item) for item in texts]).astype(np.float32)

    def names(self):
        return [name for name, _ in self.calls]


# ---------- database helpers ----------

def make_document(
    *, text_=TEXT, processing_status="completed", filename="notes.txt",
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


def make_current() -> int:
    """A document that is already indexed under the fake embedding id, with a chunk."""
    document_id = make_document(
        indexing_status="indexed", indexed_source_version=1, source_sha256="a" * 64,
        indexed_signature="{}", indexed_embedding_id=FAKE_EMBEDDING_ID,
        indexed_at=datetime.now(timezone.utc),
    )
    with SessionLocal() as session:
        session.add(DocumentChunk(
            document_id=document_id, chunk_index=0, start_offset=0, end_offset=5,
            text="hello", span_sha256="0" * 64, embedding=[0.0] * EMBEDDING_DIM,
        ))
        session.commit()
    return document_id


def db_snapshot():
    with engine.connect() as connection:
        documents = [tuple(r) for r in connection.execute(text("SELECT id, xmin::text FROM documents ORDER BY id"))]
        chunks = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
    return documents, chunks


def indexing_status(document_id: int):
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT indexing_status FROM documents WHERE id = :id"), {"id": document_id}
        ).scalar_one()


# ---------- fixtures and doubles ----------

@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / ".env").write_text("", encoding="utf-8")  # created by the test, empty
    uploads = root / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(storage, "UPLOAD_ROOT", uploads)
    return root


@pytest.fixture
def other_engine():
    other = create_engine(engine.url)
    try:
        yield other
    finally:
        other.dispose()


class RecordingFactory:
    """A session factory that remembers every session it handed out."""

    def __init__(self, bind):
        self._maker = sessionmaker(bind=bind)
        self.sessions = []

    def __call__(self):
        session = self._maker()
        self.sessions.append(session)
        return session


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


def make_runtime(repo: Path, **overrides) -> Runtime:
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


def invoke(repo: Path, *argv, runtime=None, environ=None, cwd=None, out=None, err=None):
    out = out if out is not None else io.StringIO()
    err = err if err is not None else io.StringIO()
    code = main(
        list(argv),
        runtime_loader=lambda: runtime if runtime is not None else make_runtime(repo),
        environ={} if environ is None else environ,
        cwd=repo if cwd is None else cwd,
        repo_root=repo,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def execute_args(*extra):
    return ("--execute", "--database", TEST_DATABASE, *extra)


def no_select(monkeypatch):
    """Replace the selection with a failing stand-in; returns the list of calls."""
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise AssertionError("the selection must not run")

    monkeypatch.setattr(backfill_selection, "select_document_states", boom)
    return calls


def value(text_: str, label: str) -> str:
    import re

    match = re.search(rf"^\s+{re.escape(label)}\s*:\s*(.+)$", text_, re.M)
    assert match, f"no summary row {label!r}"
    return match.group(1).strip()


# ======================================================================
# static safety refusals: exit 2 and NO database connection at all
# ======================================================================

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
def test_a_static_refusal_never_connects_selects_embeds_or_opens_a_session(case, repo, tmp_path, monkeypatch):
    make_document()
    calls = no_select(monkeypatch)
    blocker = NoConnectEngine()
    embedder = FakeEmbedder()
    factory = RecordingFactory(engine)
    options = REFUSALS[case](repo, tmp_path)
    runtime = make_runtime(
        repo, engine=blocker, embedder=embedder, session_factory=factory, **options.get("runtime", {})
    )
    before = db_snapshot()

    code, out, err = invoke(
        repo, *execute_args(), runtime=runtime, environ=options.get("environ"), cwd=options.get("cwd")
    )

    assert code == EXIT_REFUSED
    assert "REFUSED" in err and "Nothing was changed." in err
    assert out == ""
    assert blocker.connected is False
    assert calls == [] and embedder.calls == [] and factory.sessions == []
    assert SECRET not in out + err
    assert db_snapshot() == before


def test_a_static_database_name_mismatch_never_connects(repo, monkeypatch):
    calls = no_select(monkeypatch)
    blocker = NoConnectEngine()
    embedder = FakeEmbedder()
    code, out, err = invoke(
        repo, "--execute", "--database", "documind_prod",
        runtime=make_runtime(repo, engine=blocker, embedder=embedder),
    )
    assert code == EXIT_REFUSED
    assert "documind_prod" in err and TEST_DATABASE in err
    assert blocker.connected is False and calls == [] and embedder.calls == []
    assert out == ""


# ======================================================================
# live refusals: a read-only connection is needed, nothing else may happen
# ======================================================================

def test_a_live_database_mismatch_refuses_before_selection_model_sessions_or_writes(repo, monkeypatch):
    make_document()
    calls = no_select(monkeypatch)
    embedder = FakeEmbedder()
    queue_factory, runner_factory = RecordingFactory(engine), RecordingFactory(engine)
    monkeypatch.setattr(indexing_runner, "SessionLocal", runner_factory)
    # The URL names documind_x, so the static check passes; the real server is the test database.
    runtime = make_runtime(
        repo, embedder=embedder, session_factory=queue_factory,
        database_url=f"postgresql://u:{SECRET}@localhost:5434/documind_x",
    )
    before = db_snapshot()
    statements = []

    def record(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        code, out, err = invoke(repo, "--execute", "--database", "documind_x", runtime=runtime)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert code == EXIT_REFUSED
    assert "documind_x" in err and TEST_DATABASE in err and out == ""
    assert any("current_database" in statement for statement in statements)
    assert not any("documents" in statement or "document_chunks" in statement for statement in statements)
    assert calls == [] and embedder.calls == []
    assert queue_factory.sessions == [] and runner_factory.sessions == []
    assert engine.pool.checkedout() == 0  # the read-only connection was released
    assert db_snapshot() == before


def test_a_connect_error_in_execute_mode_is_a_refusal_with_only_the_class_name(repo):
    error = OperationalError("SELECT 1", {}, Exception(f"postgresql://u:{SECRET}@10.0.0.5/db"))
    embedder, factory = FakeEmbedder(), RecordingFactory(engine)
    code, out, err = invoke(
        repo, *execute_args(),
        runtime=make_runtime(repo, engine=FailingEngine(error), embedder=embedder, session_factory=factory),
    )
    assert code == EXIT_REFUSED
    assert "OperationalError" in err and SECRET not in out + err and "SELECT" not in err
    assert embedder.calls == [] and factory.sessions == []


# ======================================================================
# the engine the writers are bound to must be the engine that was verified
# ======================================================================

def _assert_refused_without_effects(code, out, err, expected_text, calls, embedder, before):
    assert code == EXIT_REFUSED
    assert expected_text in err and "Nothing was changed." in err
    assert f"localhost:5434/{TEST_DATABASE}" in err and SECRET not in out + err
    assert out == ""
    assert calls == [] and embedder.calls == []
    assert db_snapshot() == before


def test_a_queuing_factory_on_another_engine_prevents_execution(repo, other_engine, monkeypatch):
    make_document()
    calls = no_select(monkeypatch)
    embedder = FakeEmbedder()
    runner_factory = RecordingFactory(engine)
    monkeypatch.setattr(indexing_runner, "SessionLocal", runner_factory)
    before = db_snapshot()
    code, out, err = invoke(
        repo, *execute_args(),
        runtime=make_runtime(repo, embedder=embedder, session_factory=RecordingFactory(other_engine)),
    )
    _assert_refused_without_effects(code, out, err, FAILURE_SESSION_FACTORY_MISMATCH, calls, embedder, before)
    assert runner_factory.sessions == []  # the queuing factory is checked first


def test_a_runner_factory_on_another_engine_prevents_execution(repo, other_engine, monkeypatch):
    make_document()
    calls = no_select(monkeypatch)
    embedder = FakeEmbedder()
    monkeypatch.setattr(indexing_runner, "SessionLocal", RecordingFactory(other_engine))
    before = db_snapshot()
    code, out, err = invoke(
        repo, *execute_args(),
        runtime=make_runtime(repo, embedder=embedder, session_factory=RecordingFactory(engine)),
    )
    _assert_refused_without_effects(code, out, err, FAILURE_RUNNER_FACTORY_MISMATCH, calls, embedder, before)


@pytest.mark.parametrize(
    "missing",
    [{"embedder": None}, {"session_factory": None}, {"embedder": None, "session_factory": None}],
    ids=["no-embedder", "no-session-factory", "neither"],
)
def test_a_runtime_without_an_embedder_or_session_factory_prevents_execution(repo, monkeypatch, missing):
    make_document()
    calls = no_select(monkeypatch)
    before = db_snapshot()
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, **missing))
    assert code == EXIT_REFUSED
    assert MESSAGE_NO_EXECUTION_RUNTIME in err and "Nothing was changed." in err
    assert out == "" and calls == []
    assert db_snapshot() == before


# ======================================================================
# preflight: before any mutation, on the same embedder, never with a connection held
# ======================================================================

def test_the_preflight_runs_before_any_mutation(repo):
    make_document()
    baseline = db_snapshot()
    snapshots = {}

    def probe(name):
        if name in ("signature_fields", "embed_passages") and name not in snapshots:
            snapshots[name] = db_snapshot()  # the first call of each is the preflight's

    embedder = FakeEmbedder(probe=probe)
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))

    assert code == EXIT_OK
    assert snapshots == {"signature_fields": baseline, "embed_passages": baseline}
    assert embedder.calls[:2] == [("signature_fields", 0), ("embed_passages", 1)]
    assert len(embedder.calls) > 2  # the builds came after
    assert db_snapshot() != baseline  # and then the document was indexed


def test_no_database_connection_is_held_while_the_model_embeds(repo):
    make_document()
    make_document()
    checked_out = []
    embedder = FakeEmbedder(probe=lambda name: checked_out.append(engine.pool.checkedout()))

    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))

    assert code == EXIT_OK
    assert {"signature_fields", "embed_passages", "source_units"} <= set(embedder.names())
    assert checked_out and set(checked_out) == {0}


def test_the_same_embedder_instance_is_used_for_the_preflight_and_every_document(repo, monkeypatch):
    make_document()
    make_document()
    embedder = FakeEmbedder()
    seen = []
    real = indexing_runner.run_indexing

    def spy(document_id, *, embedder=None):
        seen.append(embedder)
        return real(document_id, embedder=embedder)

    monkeypatch.setattr(indexing_runner, "run_indexing", spy)
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))

    assert code == EXIT_OK
    assert len(seen) == 2 and all(item is embedder for item in seen)
    assert embedder.calls[:2] == [("signature_fields", 0), ("embed_passages", 1)]  # the preflight used it too


PREFLIGHT_FAILURES = {
    "embedding-raises": (lambda: FakeEmbedder(embed_error=RuntimeError(SECRET)), {}, "embedding", "RuntimeError"),
    "identity": (FakeEmbedder, {"embedding_id": "other/model@rev"}, "identity", None),
    "dimension": (lambda: FakeEmbedder(signature_dim=EMBEDDING_DIM + 1), {}, "result", None),
    "shape": (lambda: FakeEmbedder(embed_result=np.zeros((2, EMBEDDING_DIM), dtype=np.float32)), {}, "result", None),
}


@pytest.mark.parametrize("case", sorted(PREFLIGHT_FAILURES))
def test_a_failed_preflight_refuses_and_changes_nothing(case, repo):
    make_document()
    build, overrides, stage, error_type = PREFLIGHT_FAILURES[case]
    embedder = build()
    before = db_snapshot()

    code, out, err = invoke(
        repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder, **overrides)
    )

    assert code == EXIT_REFUSED
    assert f"REFUSED: embedding preflight failed at the {stage} step" in err
    if error_type:
        assert error_type in err
    assert "Nothing was changed." in err
    assert SECRET not in out + err
    assert "source_units" not in embedder.names()
    assert EXECUTE_MODE in out                    # the plan was shown
    assert EXECUTE_FOOTER not in out              # but indexing never started
    assert "[1/1]" not in out and "Execution summary" not in out
    assert db_snapshot() == before


def test_an_interrupt_during_the_preflight_changes_nothing(repo):
    make_document()

    def interrupt():
        raise KeyboardInterrupt

    embedder = FakeEmbedder(hooks={"signature_fields": interrupt})
    before = db_snapshot()
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))
    assert code == EXIT_INTERRUPTED
    assert "Interrupted; nothing was changed." in err
    assert "source_units" not in embedder.names()
    assert db_snapshot() == before


def test_offline_is_forced_in_the_given_environment_and_not_in_the_process(repo, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    make_document()
    environ = {"HF_HUB_OFFLINE": "0"}  # an inherited value must not win
    code, out, err = invoke(repo, *execute_args(), environ=environ)
    assert code == EXIT_OK
    assert environ["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_the_late_offline_note_appears_only_when_the_libraries_were_already_imported(repo, monkeypatch):
    make_document()
    for module in ("transformers", "huggingface_hub"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_OK and NOTE_OFFLINE_SET_LATE not in err

    make_document()
    monkeypatch.setitem(sys.modules, "transformers", object())
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_OK and NOTE_OFFLINE_SET_LATE in err


# ======================================================================
# output order
# ======================================================================

def test_the_plan_precedes_the_preflight_and_the_footer_follows_it(repo):
    make_document()
    out, err = io.StringIO(), io.StringIO()
    seen = {}

    def probe(name):
        if name == "signature_fields" and not seen:  # the first call is the preflight's
            seen["out"], seen["err"] = out.getvalue(), err.getvalue()

    embedder = FakeEmbedder(probe=probe)
    code, out_text, err_text = invoke(
        repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder), out=out, err=err
    )

    assert code == EXIT_OK
    assert EXECUTE_MODE in seen["out"] and EXECUTE_FOOTER not in seen["out"] and "[1/1]" not in seen["out"]
    assert "Checking the embedding model" in seen["err"] and "check passed" not in seen["err"]
    assert "check passed" in err_text
    positions = [out_text.index(marker) for marker in (EXECUTE_MODE, EXECUTE_FOOTER, "[1/1]", "Execution summary")]
    assert positions == sorted(positions)


# ======================================================================
# nothing to index
# ======================================================================

def test_a_run_with_nothing_to_index_skips_the_preflight_and_exits_zero(repo):
    make_current()
    embedder = FakeEmbedder()
    before = db_snapshot()
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))
    assert code == EXIT_OK
    assert embedder.calls == []
    assert "Checking the embedding model" not in err and EXECUTE_FOOTER not in out
    assert "Execution summary" in out and value(out, "Planned for indexing") == "0"
    assert db_snapshot() == before


def test_an_empty_database_exits_zero_without_touching_the_model(repo):
    embedder = FakeEmbedder()
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))
    assert code == EXIT_OK and embedder.calls == []
    assert "finished without failures" in out


def test_only_blocked_documents_exit_one_without_touching_the_model(repo):
    blocked = make_document(filename="gone.pdf", stored_name="gone.pdf", content_type="application/pdf")
    embedder = FakeEmbedder()
    before = db_snapshot()
    code, out, err = invoke(repo, *execute_args(), runtime=make_runtime(repo, embedder=embedder))
    assert code == EXIT_FAILURE
    assert embedder.calls == []
    assert value(out, "Blocked (in the plan)") == "1" and value(out, "Planned for indexing") == "0"
    assert "finished with failures or blocked documents" in out
    assert indexing_status(blocked) is None
    assert db_snapshot() == before


# ======================================================================
# failures outside the engine, and the summary print
# ======================================================================

def test_an_unexpected_error_after_writes_began_says_the_database_may_have_changed(repo, monkeypatch):
    make_document()

    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(backfill_cli, "execute_plan", broken)
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_FAILURE
    assert "Unexpected error (RuntimeError)" in err
    assert "the database may have been changed" in err
    assert "nothing was changed" not in err.lower()
    assert SECRET not in out + err


def test_an_interrupt_after_writes_began_says_the_database_may_have_changed(repo, monkeypatch):
    make_document()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(backfill_cli, "execute_plan", interrupted)
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_INTERRUPTED
    assert "Interrupted after indexing started" in err
    assert "nothing was changed" not in err.lower()


def test_an_unexpected_error_with_nothing_to_index_still_says_nothing_was_changed(repo, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(backfill_cli, "execute_plan", broken)  # no documents: writes never started
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_FAILURE
    assert "Unexpected error (RuntimeError); nothing was changed." in err
    assert SECRET not in out + err


def test_a_summary_that_cannot_be_printed_keeps_the_computed_exit_code(repo, monkeypatch):
    document_id = make_document()

    def broken(summary):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(backfill_cli, "render_execution_summary", broken)
    code, out, err = invoke(repo, *execute_args())
    assert code == EXIT_OK
    assert "Could not print the execution summary (RuntimeError)" in err
    assert SECRET not in out + err
    assert indexing_status(document_id) == "indexed"


# ======================================================================
# exit_code_for (pure)
# ======================================================================

def make_summary(outcomes=(), *, interrupted=False, plan_blocked=0, plan_reported=0, plan_skipped=0, plan_deferred=0):
    results = tuple(ItemResult(index + 1, outcome) for index, outcome in enumerate(outcomes))
    return ExecutionSummary(
        examined=len(results) + plan_blocked + plan_reported + plan_skipped + plan_deferred,
        plan_deferred=plan_deferred,
        plan_blocked=plan_blocked,
        plan_reported=plan_reported,
        plan_skipped=plan_skipped,
        results=results,
        interrupted=interrupted,
        elapsed_seconds=0.0,
    )


@pytest.mark.parametrize(
    "summary, expected",
    [
        (make_summary([ItemOutcome.INDEXED, ItemOutcome.INDEXED]), EXIT_OK),
        (make_summary([ItemOutcome.CHANGED, ItemOutcome.STALE, ItemOutcome.GONE]), EXIT_OK),
        (make_summary([ItemOutcome.NOT_QUEUED, ItemOutcome.NOT_CLAIMED]), EXIT_OK),
        (make_summary([], plan_reported=2, plan_skipped=3, plan_deferred=1), EXIT_OK),
        (make_summary([ItemOutcome.INDEXED, ItemOutcome.FAILED]), EXIT_FAILURE),
        (make_summary([ItemOutcome.ERROR]), EXIT_FAILURE),
        (make_summary([ItemOutcome.FAILURE_NOT_RECORDED]), EXIT_FAILURE),
        (make_summary([ItemOutcome.INDEXED], plan_blocked=1), EXIT_FAILURE),
        (make_summary([ItemOutcome.BLOCKED]), EXIT_FAILURE),
        (make_summary([ItemOutcome.INDEXED, ItemOutcome.NOT_ATTEMPTED], interrupted=True), EXIT_INTERRUPTED),
        (make_summary([ItemOutcome.FAILED, ItemOutcome.INTERRUPTED], interrupted=True), EXIT_INTERRUPTED),
        (make_summary([ItemOutcome.INDEXED], interrupted=True, plan_blocked=1), EXIT_INTERRUPTED),
    ],
)
def test_exit_code_for(summary, expected):
    assert exit_code_for(summary) == expected


# ======================================================================
# module hygiene
# ======================================================================

def test_importing_the_cli_loads_no_configuration_database_or_model_code():
    forbidden = (
        "app.core.config", "app.core.database", "app.rag.embedder", "app.rag.indexing_runner",
        "app.rag.indexing_service", "app.rag.backfill_preflight", "torch", "transformers",
    )
    code = "import sys, json\nimport app.rag.backfill_cli\nprint(json.dumps(sorted(sys.modules)))\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    assert "app.rag.backfill_cli" in loaded and "app.rag.backfill_wiring" in loaded  # really imported
    assert [name for name in loaded if name in forbidden] == []


def test_the_execute_help_no_longer_calls_it_reserved_or_unavailable():
    shown = build_parser().format_help()
    assert "--execute" in shown
    assert "reserved" not in shown and "not available" not in shown

# ======================================================================
# CP6c.2b: the real lifecycle through the CLI (fake embedder, test database)
# ======================================================================

# ---------- helpers ----------

def execute(sql, **params):
    with engine.begin() as connection:
        connection.execute(text(sql), params)


def set_live_lease(document_id, seconds=300):
    """Another worker holds the document: 'indexing' with a live lease (database clock)."""
    execute(
        "UPDATE documents SET indexing_status = 'indexing', "
        "indexing_lease_expires_at = clock_timestamp() + (:seconds * interval '1 second') WHERE id = :id",
        seconds=seconds, id=document_id,
    )


def expire_lease(document_id):
    execute(
        "UPDATE documents SET indexing_lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = :id",
        id=document_id,
    )


def bump_attempt(document_id):
    """Another worker claimed the document again: the attempt number moves on."""
    execute("UPDATE documents SET indexing_attempt = indexing_attempt + 1 WHERE id = :id", id=document_id)


def delete_document(document_id):
    execute("DELETE FROM documents WHERE id = :id", id=document_id)


def uncomplete_document(document_id):
    execute("UPDATE documents SET processing_status = 'processing' WHERE id = :id", id=document_id)


def doc_row(document_id):
    """The lifecycle columns of one document plus its chunk count, or None if it is gone."""
    with engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT d.indexing_status AS status, d.indexing_attempt AS attempt, "
                "d.indexing_error AS error, d.indexing_lease_expires_at IS NOT NULL AS leased, "
                "d.indexed_embedding_id AS embedding_id, d.source_sha256 AS sha, "
                "(SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunks "
                "FROM documents d WHERE d.id = :id"
            ),
            {"id": document_id},
        ).first()


def chunk_rows(document_id):
    with SessionLocal() as session:
        return session.execute(
            select(
                DocumentChunk.chunk_index, DocumentChunk.start_offset, DocumentChunk.end_offset,
                DocumentChunk.page_start, DocumentChunk.page_end, DocumentChunk.embedding,
            )
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()


def progress_for(out, document_id):
    """The text after 'doc N' on that document's progress line, or None if it has no line."""
    match = re.search(rf"^\[\d+/\d+\] doc {document_id}\s+(.+)$", out, re.M)
    return match.group(1).strip() if match else None


def run_cli(repo, embedder, *argv, **runtime_overrides):
    runtime = make_runtime(repo, embedder=embedder, **runtime_overrides)
    return invoke(repo, *execute_args(*argv), runtime=runtime)


def three_documents():
    return make_document(), make_document(), make_document()


def assert_benign_run(out, *, changed):
    """Documents others changed or own are skipped, never failures."""
    assert value(out, "Changed since the plan") == str(changed)
    assert value(out, "Failed") == "0"
    assert "Result: finished without failures" in out


LEAK_TOKENS = (SECRET, "TOPSECRET", "CONFIDENTIAL", "stored-6f1d", "topsecret-stored")


def assert_no_leaks(repo, *texts):
    combined = "".join(texts)
    for token in LEAK_TOKENS:
        assert token not in combined
    assert str(repo) not in combined
    assert combined.isascii()


class MarkerEmbedder(FakeEmbedder):
    """Fails or interrupts only for documents whose text carries a marker word."""

    def __init__(self, *, fail_marker=None, interrupt_marker=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_marker = fail_marker
        self.interrupt_marker = interrupt_marker

    def source_units(self, value):
        if self.fail_marker and self.fail_marker in value:
            self._enter("source_units", len(value))
            raise RuntimeError(SECRET)
        if self.interrupt_marker and self.interrupt_marker in value:
            self._enter("source_units", len(value))
            raise KeyboardInterrupt
        return super().source_units(value)


class Trace:
    """Records the order of selections, requests, runs and embedder calls."""

    def __init__(self):
        self.events = []
        self.selections = []  # (document_ids or None, transaction_read_only, connection)

    def probe(self, name):
        self.events.append(f"embedder:{name}")

    def without_embedder(self):
        return [event_name for event_name in self.events if not event_name.startswith("embedder:")]

    def install(self, monkeypatch):
        real_select = backfill_selection.select_document_states
        real_request = indexing_service.request_indexing
        real_run = indexing_runner.run_indexing

        def select_spy(connection, *, document_ids=None):
            readonly = connection.exec_driver_sql("SHOW transaction_read_only").scalar()
            ids = None if document_ids is None else tuple(document_ids)
            self.selections.append((ids, readonly, connection))
            self.events.append("select:plan" if ids is None else f"select:{ids[0]}")
            return real_select(connection, document_ids=document_ids)

        def request_spy(session, document_id):
            self.events.append(f"request:{document_id}")
            return real_request(session, document_id)

        def run_spy(document_id, *, embedder=None):
            self.events.append(f"run:{document_id}")
            return real_run(document_id, embedder=embedder)

        monkeypatch.setattr(backfill_selection, "select_document_states", select_spy)
        monkeypatch.setattr(indexing_service, "request_indexing", request_spy)
        monkeypatch.setattr(indexing_runner, "run_indexing", run_spy)


class _ClosingSession(Session):
    fail_on_close = False

    def close(self):
        super().close()  # really closed (and anything committed stays committed) ...
        if self.fail_on_close:
            raise RuntimeError(SECRET)  # ... and then the close reports an error


class FlakyCloseFactory:
    """Session factory whose sessions fail on close() once ``healthy_sessions`` have been made."""

    def __init__(self, bind, *, healthy_sessions):
        self._maker = sessionmaker(bind=bind, class_=_ClosingSession)
        self._healthy = healthy_sessions
        self.created = 0

    def __call__(self):
        session = self._maker()
        self.created += 1
        session.fail_on_close = self.created > self._healthy
        return session


# ======================================================================
# A. indexing through the CLI
# ======================================================================

def test_a_full_run_publishes_real_chunks_and_a_rerun_has_nothing_to_do(repo):
    ids = [make_document() for _ in range(3)]

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_OK
    total_chunks = 0
    for position, document_id in enumerate(ids, start=1):
        row = doc_row(document_id)
        assert (row.status, row.attempt, row.error, row.leased) == ("indexed", 1, None, False)
        assert row.embedding_id == FAKE_EMBEDDING_ID
        assert row.sha == source_sha256(TEXT)
        assert row.chunks >= 1
        match = re.search(rf"^\[{position}/3\] doc {document_id}\s+indexed \((\d+) chunks", out, re.M)
        assert match is not None and int(match.group(1)) == row.chunks
        chunks = chunk_rows(document_id)
        assert [chunk.chunk_index for chunk in chunks] == list(range(row.chunks))
        assert all(0 <= chunk.start_offset < chunk.end_offset <= len(TEXT) for chunk in chunks)
        assert all(chunk.page_start is None and chunk.page_end is None for chunk in chunks)  # not a PDF
        assert all(len(chunk.embedding) == EMBEDDING_DIM for chunk in chunks)
        total_chunks += row.chunks
    assert db_snapshot()[1] == total_chunks
    assert value(out, "Indexed") == "3" and value(out, "Failed") == "0"
    assert "Result: finished without failures" in out

    rerun_embedder = FakeEmbedder()
    before = db_snapshot()
    code, out, err = run_cli(repo, rerun_embedder)
    assert code == EXIT_OK and rerun_embedder.calls == []
    assert value(out, "Planned for indexing") == "0" and value(out, "Skipped (nothing to do)") == "3"
    assert db_snapshot() == before


def test_limit_indexes_only_the_lowest_candidates_and_leaves_the_rest_untouched(repo):
    ids = [make_document() for _ in range(4)]
    code, out, err = run_cli(repo, FakeEmbedder(), "--limit", "2")
    assert code == EXIT_OK
    assert [doc_row(document_id).status for document_id in ids] == ["indexed", "indexed", None, None]
    assert [doc_row(document_id).attempt for document_id in ids[2:]] == [0, 0]
    assert value(out, "Planned for indexing") == "2" and value(out, "Deferred by --limit") == "2"


def test_a_failed_document_is_left_alone_without_include_failed_and_retried_with_it(repo):
    failed = make_document(indexing_status="failed", indexing_error=FailureReason.UNEXPECTED.value)
    plain = make_document()
    before_failed = doc_row(failed)

    code, out, err = run_cli(repo, FakeEmbedder())
    assert code == EXIT_OK  # a reported document never fails a run
    assert doc_row(plain).status == "indexed"
    assert doc_row(failed) == before_failed
    assert value(out, "Reported (needs attention)") == "1"

    code, out, err = run_cli(repo, FakeEmbedder(), "--include-failed")
    assert code == EXIT_OK
    row = doc_row(failed)
    assert (row.status, row.error, row.attempt) == ("indexed", None, 1)


# ======================================================================
# B. order, connections and engines
# ======================================================================

def test_the_lifecycle_runs_in_the_documented_order(repo, monkeypatch):
    d1, d2 = make_document(), make_document()
    trace = Trace()
    trace.install(monkeypatch)

    code, out, err = run_cli(repo, FakeEmbedder(probe=trace.probe))

    assert code == EXIT_OK
    # The plan is read once, then the preflight, and only then the per-document work.
    assert trace.events[:3] == ["select:plan", "embedder:signature_fields", "embedder:embed_passages"]
    assert trace.without_embedder() == [
        "select:plan",
        f"select:{d1}", f"request:{d1}", f"run:{d1}",   # re-read, queue, run: immediately in that order
        f"select:{d2}", f"request:{d2}", f"run:{d2}",
    ]


def test_selection_connections_are_read_only_and_closed_before_any_embedding(repo, monkeypatch):
    make_document()
    make_document()
    trace = Trace()
    trace.install(monkeypatch)
    closed_at_each_embedder_call = []

    def probe(name):
        closed_at_each_embedder_call.append(all(connection.closed for _, _, connection in trace.selections))

    code, out, err = run_cli(repo, FakeEmbedder(probe=probe))

    assert code == EXIT_OK
    assert [ids is None for ids, _, _ in trace.selections] == [True, False, False]  # plan, then one per document
    assert {readonly for _, readonly, _ in trace.selections} == {"on"}
    assert closed_at_each_embedder_call and all(closed_at_each_embedder_call)
    assert engine.pool.checkedout() == 0


def test_every_database_step_runs_on_the_runtime_engine_through_one_pooled_connection(repo, monkeypatch):
    d1, d2 = make_document(), make_document()
    pooled = create_engine(engine.url, pool_size=1, max_overflow=0, pool_timeout=3)
    queue_factory, runner_factory = RecordingFactory(pooled), RecordingFactory(pooled)
    monkeypatch.setattr(indexing_runner, "SessionLocal", runner_factory)
    pooled_statements, default_statements = [], []

    def record_pooled(connection, cursor, statement, parameters, context, executemany):
        pooled_statements.append(statement)

    def record_default(connection, cursor, statement, parameters, context, executemany):
        default_statements.append(statement)

    event.listen(pooled, "before_cursor_execute", record_pooled)
    event.listen(engine, "before_cursor_execute", record_default)
    try:
        runtime = make_runtime(repo, engine=pooled, session_factory=queue_factory)
        code, out, err = invoke(repo, *execute_args(), runtime=runtime)
        idle = pooled.pool.checkedout()
    finally:
        event.remove(engine, "before_cursor_execute", record_default)
        event.remove(pooled, "before_cursor_execute", record_pooled)
        pooled.dispose()

    # One pooled connection served the read-only planning, every re-read, every request and
    # every runner step: a connection that was not released would have timed out the next step.
    assert code == EXIT_OK
    assert doc_row(d1).status == "indexed" and doc_row(d2).status == "indexed"
    assert default_statements == []                      # nothing used the application's default engine
    assert any("current_database" in statement for statement in pooled_statements)
    assert any("UPDATE documents" in statement for statement in pooled_statements)
    assert any("INSERT INTO document_chunks" in statement for statement in pooled_statements)
    assert len(queue_factory.sessions) == 1 + 2          # the target check, then one session per request
    assert len(runner_factory.sessions) == 1 + 3 * 2  # One target-check session, then claim, read-source and publish per document.
    assert idle == 0

@pytest.mark.parametrize("failing", ["embedding", "identity"])
def test_a_failed_preflight_makes_no_request_run_or_per_document_read(repo, monkeypatch, failing):
    document_id = make_document()
    trace = Trace()
    trace.install(monkeypatch)
    if failing == "embedding":
        runtime = make_runtime(repo, embedder=FakeEmbedder(embed_error=RuntimeError(SECRET)))
    else:
        runtime = make_runtime(repo, embedding_id="other/model@rev")

    code, out, err = invoke(repo, *execute_args(), runtime=runtime)

    assert code == EXIT_REFUSED
    assert trace.events == ["select:plan"]  # the plan only: no re-read, no request, no run
    row = doc_row(document_id)
    assert (row.status, row.attempt) == (None, 0)


# ======================================================================
# C. races and other benign outcomes are skipped, never failures
# ======================================================================

def test_a_document_leased_by_someone_else_after_planning_is_skipped(repo):
    d1, d2, d3 = three_documents()
    embedder = FakeEmbedder(hooks={"source_units": lambda: set_live_lease(d2)})

    code, out, err = run_cli(repo, embedder)

    assert code == EXIT_OK
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    row = doc_row(d2)
    assert (row.status, row.attempt, row.leased) == ("indexing", 0, True)  # the other worker's lease, untouched
    assert progress_for(out, d2) == f"changed ({Category.LIVE_LEASE.value})"
    assert_benign_run(out, changed=1)


@pytest.mark.parametrize("mutate", [delete_document, uncomplete_document], ids=["deleted", "no-longer-completed"])
def test_a_document_that_leaves_the_selection_after_planning_is_skipped(repo, mutate):
    d1, d2, d3 = three_documents()
    embedder = FakeEmbedder(hooks={"source_units": lambda: mutate(d2)})

    code, out, err = run_cli(repo, embedder)

    assert code == EXIT_OK
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    assert progress_for(out, d2) == f"changed ({DETAIL_NO_LONGER_SELECTED})"
    assert_benign_run(out, changed=1)


def test_a_request_that_loses_a_race_is_not_queued_and_not_a_failure(repo, monkeypatch):
    d1, d2, d3 = three_documents()
    real_request, real_run = indexing_service.request_indexing, indexing_runner.run_indexing
    runs = []

    def racing_request(session, document_id):
        if document_id == d2:
            set_live_lease(d2)  # another worker takes it between our re-read and our request
        return real_request(session, document_id)

    def counting_run(document_id, *, embedder=None):
        runs.append(document_id)
        return real_run(document_id, embedder=embedder)

    monkeypatch.setattr(indexing_service, "request_indexing", racing_request)
    monkeypatch.setattr(indexing_runner, "run_indexing", counting_run)

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_OK
    assert runs == [d1, d3]  # run is never called for a document that was not queued
    assert progress_for(out, d2) == "not queued"
    row = doc_row(d2)
    assert (row.status, row.attempt) == ("indexing", 0)
    assert_benign_run(out, changed=1)


def test_a_run_that_loses_the_claim_is_not_claimed_and_not_a_failure(repo, monkeypatch):
    d1, d2, d3 = three_documents()
    real_run = indexing_runner.run_indexing

    def racing_run(document_id, *, embedder=None):
        if document_id == d2:
            set_live_lease(d2)  # another worker claims it between our request and our claim
        return real_run(document_id, embedder=embedder)

    monkeypatch.setattr(indexing_runner, "run_indexing", racing_run)

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_OK
    assert progress_for(out, d2) == "not claimed"
    row = doc_row(d2)
    assert (row.status, row.leased, row.chunks) == ("indexing", True, 0)
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    assert_benign_run(out, changed=1)


def test_a_document_whose_attempt_is_taken_over_during_the_build_is_stale_not_failed(repo):
    d1, d2, d3 = three_documents()
    embedder = FakeEmbedder(hooks={"source_units": lambda: bump_attempt(d1)})  # during d1's build

    code, out, err = run_cli(repo, embedder)

    assert code == EXIT_OK
    assert progress_for(out, d1) == "stale"
    row = doc_row(d1)
    assert (row.status, row.attempt, row.chunks) == ("indexing", 2, 0)  # nothing was published
    assert doc_row(d2).status == "indexed" and doc_row(d3).status == "indexed"
    assert_benign_run(out, changed=1)


def test_a_document_deleted_during_its_own_build_is_gone_not_failed(repo):
    d1, d2, d3 = three_documents()
    embedder = FakeEmbedder(hooks={"source_units": lambda: delete_document(d1)})

    code, out, err = run_cli(repo, embedder)

    assert code == EXIT_OK
    assert progress_for(out, d1) == "gone"
    assert doc_row(d1) is None and chunk_rows(d1) == []
    assert doc_row(d2).status == "indexed" and doc_row(d3).status == "indexed"
    assert_benign_run(out, changed=1)


# ======================================================================
# D. failures: recorded, reported, and the run goes on
# ======================================================================

def marked_documents():
    d1 = make_document()
    d2 = make_document(text_=TEXT + " FAILME", filename="TOPSECRET-notes.txt", stored_name="topsecret-stored.txt")
    d3 = make_document()
    return d1, d2, d3


def test_a_failing_document_is_recorded_and_the_later_documents_are_still_indexed(repo):
    d1, d2, d3 = marked_documents()

    code, out, err = run_cli(repo, MarkerEmbedder(fail_marker="FAILME"))

    assert code == EXIT_FAILURE
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"  # d3 came after the failure
    failed = doc_row(d2)
    assert (failed.status, failed.error, failed.attempt, failed.chunks, failed.leased) == (
        "failed", FailureReason.UNEXPECTED.value, 1, 0, False,
    )
    assert progress_for(out, d2) == f"failed ({FailureReason.UNEXPECTED.name})"
    assert value(out, "Indexed") == "2" and value(out, "Failed") == "1"
    assert "Result: finished with failures or blocked documents" in out
    assert_no_leaks(repo, out, err)


def test_a_failed_document_is_retried_only_with_include_failed(repo):
    d1, d2, d3 = marked_documents()
    run_cli(repo, MarkerEmbedder(fail_marker="FAILME"))
    assert doc_row(d2).status == "failed"
    before = doc_row(d2)

    code, out, err = run_cli(repo, FakeEmbedder())  # same text, healthy embedder, no flag
    assert code == EXIT_OK and doc_row(d2) == before
    assert value(out, "Reported (needs attention)") == "1"

    code, out, err = run_cli(repo, FakeEmbedder(), "--include-failed")
    assert code == EXIT_OK
    row = doc_row(d2)
    assert (row.status, row.error, row.attempt) == ("indexed", None, 2)


def test_a_session_that_fails_to_close_after_the_commit_is_an_error_and_the_document_is_picked_up_later(repo):
    d1, d2 = make_document(), make_document()
    factory = FlakyCloseFactory(engine, healthy_sessions=1)  # the first session is the target check's
    embedder = FakeEmbedder()

    code, out, err = run_cli(repo, embedder, session_factory=factory)

    assert code == EXIT_FAILURE
    for document_id in (d1, d2):
        row = doc_row(document_id)
        assert (row.status, row.attempt) == ("pending", 0)  # queued and committed, then the close failed
        assert progress_for(out, document_id) == "error (RuntimeError)"
    assert "source_units" not in embedder.names()  # nothing was built for them
    assert value(out, "Failed") == "2"
    assert_no_leaks(repo, out, err)

    code, out, err = run_cli(repo, FakeEmbedder())  # a later backfill, with a healthy factory
    assert code == EXIT_OK
    assert doc_row(d1).status == "indexed" and doc_row(d2).status == "indexed"


# ======================================================================
# E. interruption
# ======================================================================

def interrupted_run(repo):
    d1 = make_document()
    d2 = make_document(text_=TEXT + " INTERRUPTME")
    d3 = make_document()
    code, out, err = run_cli(repo, MarkerEmbedder(interrupt_marker="INTERRUPTME"))
    return (d1, d2, d3), code, out, err


def test_an_interrupt_mid_run_gives_an_accurate_partial_summary(repo):
    (d1, d2, d3), code, out, err = interrupted_run(repo)

    assert code == EXIT_INTERRUPTED
    assert doc_row(d1).status == "indexed" and doc_row(d1).chunks >= 1
    in_progress = doc_row(d2)
    assert (in_progress.status, in_progress.attempt, in_progress.leased, in_progress.chunks) == ("indexing", 1, True, 0)
    untouched = doc_row(d3)
    assert (untouched.status, untouched.attempt) == (None, 0)
    assert progress_for(out, d1).startswith("indexed")
    assert progress_for(out, d2) is None and progress_for(out, d3) is None  # only finished documents get a line
    assert value(out, "Indexed") == "1"
    assert value(out, "Interrupted in progress") == "1" and value(out, "Not attempted") == "1"
    assert "Result: interrupted before completion" in out
    assert "stays locked until its lease expires" in out
    assert_no_leaks(repo, out, err)


def test_a_rerun_skips_the_interrupted_document_while_its_lease_is_live(repo):
    (d1, d2, d3), code, out, err = interrupted_run(repo)

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_OK
    assert doc_row(d3).status == "indexed"
    row = doc_row(d2)
    assert (row.status, row.attempt, row.leased) == ("indexing", 1, True)  # untouched
    assert value(out, "Planned for indexing") == "1" and value(out, "Skipped (nothing to do)") == "2"


def test_a_rerun_recovers_the_interrupted_document_once_its_lease_has_expired(repo):
    (d1, d2, d3), code, out, err = interrupted_run(repo)
    run_cli(repo, FakeEmbedder())
    expire_lease(d2)

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_OK
    row = doc_row(d2)
    assert (row.status, row.attempt, row.error, row.leased) == ("indexed", 2, None, False)
    assert progress_for(out, d2).startswith("indexed")


# ======================================================================
# F. blocked documents
# ======================================================================

def test_a_blocked_pdf_is_never_queued_while_the_other_documents_are_indexed(repo):
    d1 = make_document()
    blocked = make_document(
        filename="CONFIDENTIAL-quarterly.pdf", stored_name="stored-6f1d.pdf", content_type="application/pdf"
    )
    d3 = make_document()

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_FAILURE
    row = doc_row(blocked)
    assert (row.status, row.attempt) == (None, 0)
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    assert value(out, "Blocked (in the plan)") == "1" and value(out, "Indexed") == "2"
    assert progress_for(out, blocked) is None
    assert f"[{SourceStatus.PDF_FILE_MISSING.value}]" in out
    assert "Result: finished with failures or blocked documents" in out
    assert_no_leaks(repo, out, err)


def test_a_pdf_whose_file_disappears_after_planning_is_blocked_at_run_time(repo):
    stored = repo / "uploads" / "present.pdf"
    stored.write_bytes(b"%PDF-1.4\n")
    d1 = make_document()
    d2 = make_document(filename="a.pdf", stored_name="present.pdf", content_type="application/pdf")
    d3 = make_document()
    embedder = FakeEmbedder(hooks={"source_units": stored.unlink})  # during d1's build

    code, out, err = run_cli(repo, embedder)

    assert code == EXIT_FAILURE
    row = doc_row(d2)
    assert (row.status, row.attempt) == (None, 0)  # never queued
    assert progress_for(out, d2) == f"blocked ({SourceStatus.PDF_FILE_MISSING.value})"
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    assert value(out, "Blocked (found at run time)") == "1"
    assert value(out, "Blocked (in the plan)") == "0" and value(out, "Failed") == "0"


# ======================================================================
# G. no leaks across a mixed run
# ======================================================================

def test_a_mixed_run_leaks_no_secret_filename_stored_name_or_path(repo):
    make_document()
    make_document(text_=TEXT + " FAILME", filename="TOPSECRET-notes.txt", stored_name="topsecret-stored.txt")
    make_document(filename="CONFIDENTIAL-quarterly.pdf", stored_name="stored-6f1d.pdf", content_type="application/pdf")

    code, out, err = run_cli(repo, MarkerEmbedder(fail_marker="FAILME"))

    assert code == EXIT_FAILURE
    assert value(out, "Indexed") == "1" and value(out, "Failed") == "1"
    assert value(out, "Blocked (in the plan)") == "1"
    assert_no_leaks(repo, out, err)


# ======================================================================
# OPTIONAL HARDENING
# ======================================================================

def test_optional_a_failure_that_cannot_be_recorded_is_reported_and_the_run_continues(repo, monkeypatch):
    d1, d2, d3 = marked_documents()

    def cannot_record(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(indexing_runner, "record_failure", cannot_record)

    code, out, err = run_cli(repo, MarkerEmbedder(fail_marker="FAILME"))

    assert code == EXIT_FAILURE
    assert progress_for(out, d2) == f"failure not recorded ({FailureReason.UNEXPECTED.name})"
    row = doc_row(d2)
    assert (row.status, row.leased, row.attempt) == ("indexing", True, 1)  # stays locked until the lease expires
    assert doc_row(d1).status == "indexed" and doc_row(d3).status == "indexed"
    assert_no_leaks(repo, out, err)


def test_optional_an_unexpected_exception_from_the_runner_is_an_error_and_the_run_continues(repo, monkeypatch):
    d1, d2, d3 = three_documents()
    real_run = indexing_runner.run_indexing

    def exploding_run(document_id, *, embedder=None):
        if document_id == d2:
            raise RuntimeError(SECRET)
        return real_run(document_id, embedder=embedder)

    monkeypatch.setattr(indexing_runner, "run_indexing", exploding_run)

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_FAILURE
    assert progress_for(out, d2) == "error (RuntimeError)"
    row = doc_row(d2)
    assert (row.status, row.attempt) == ("pending", 0)  # queued, never claimed
    assert doc_row(d3).status == "indexed"
    assert_no_leaks(repo, out, err)


def test_optional_an_interrupt_before_any_write_leaves_everything_untouched(repo, monkeypatch):
    make_document()
    make_document()
    real_select = backfill_selection.select_document_states

    def selecting(connection, *, document_ids=None):
        if document_ids is not None:  # the per-document re-read of the first document
            raise KeyboardInterrupt
        return real_select(connection, document_ids=document_ids)

    monkeypatch.setattr(backfill_selection, "select_document_states", selecting)
    before = db_snapshot()

    code, out, err = run_cli(repo, FakeEmbedder())

    assert code == EXIT_INTERRUPTED
    assert db_snapshot() == before
    assert value(out, "Not attempted") == "2" and value(out, "Interrupted in progress") == "0"
    assert "stays locked" not in out
    assert engine.pool.checkedout() == 0