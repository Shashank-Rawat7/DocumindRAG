"""E1.7.5: the pilot command line, run with fakes (no database, no embedder, no real corpus).

Safety refusals use the real ``check_static_safety`` on a temporary repository. The database side is a fake
session that implements only the read-only methods of ``DbSession``; ``compare_snapshots`` is replaced.

Run focused: pytest tests/test_pilot_cli.py --noconftest -p no:cacheprovider
"""
import dataclasses
import io
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_evaluation_engine as fx

from app.rag import pilot_cli as cli
from app.rag.evaluation import report as rep
from app.rag.snapshot_cli import Runtime

OWNER_ID = 7770111
DIGEST = "d" * 64
IDENTITY = rep.combine_identity({"app.rag.evaluation.engine": "e" * 64})
FINGERPRINT = "f" * 64
ADJ = ["--adjudicate", "q4:u1"]
QUESTIONS = (fx.Q1, fx.Q2, fx.Q3, fx.Q4, fx.U1, fx.U2)


class FakeSession:
    def __init__(self, ctx):
        self.ctx = ctx

    def check_live(self, name):
        return self.ctx.live_problem

    def assert_read_only(self):
        return self.ctx.readonly_problem

    def take_snapshot(self, clock):
        self.ctx.calls.append("snapshot")
        return {"fingerprint": FINGERPRINT}

    def verify_corpus(self, files, *, owner_id, owner_sha256, embedding_id):
        self.ctx.calls.append("verify")
        self.ctx.verify_args = (owner_id, owner_sha256, embedding_id)
        return self.ctx.verification

    def document_rows(self, files):
        return self.ctx.rows

    def pool_chunks(self, *, owner_id, embedding_id, document_ids):
        self.ctx.pool_args = (owner_id, embedding_id, tuple(document_ids))
        return self.ctx.chunks

    def dense_retriever(self, embedder, *, owner_id, document_ids, top_k):
        self.ctx.top_k = top_k
        return self.ctx.retriever


class Ctx:
    def __init__(self, tmp_path):
        self.repo = (tmp_path / "repo").resolve()
        (self.repo / "uploads").mkdir(parents=True)
        (self.repo / ".env").write_text("x", encoding="utf-8")
        (self.repo / "eval" / "dataset").mkdir(parents=True)
        (self.repo / "eval" / "dataset" / "dev.jsonl").write_text("", encoding="utf-8")
        self.outside = (tmp_path / "outside").resolve()
        self.outside.mkdir()
        self.out = self.outside / "pilot.json"
        self.calls, self.sessions = [], 0
        self.live_problem = self.readonly_problem = None
        self.verification = SimpleNamespace(ok=True, problems=[], documents=[])
        self.rows = [self.row(fx.ID_A, "doc-a"), self.row(fx.ID_B, "doc-b")]
        self.chunks = list(fx.C.values())
        self.retriever = fx.Retriever()
        self.url = "postgresql://user:hunter2@localhost:5432/documind_db"
        self.environment, self.explicit = "development", True
        self.cwd = self.repo
        self.embedder_id = fx.EMB
        self.corpus_manifest = fx.MANIFEST
        self.git = ("c" * 40, True)
        self.git_hook = None
        self.identical = True
        self.inputs = fx.inputs_of(*QUESTIONS)
        self.inputs_loaded = 0
        self.tools_embedding_id = fx.EMB

    @staticmethod
    def row(doc_id, key):
        return {"id": doc_id, "owner_id": OWNER_ID, "filename": f"{key}.txt", "description": f"c1:{key}",
                "source_version": 1, "indexed_source_version": 1, "indexed_embedding_id": fx.EMB,
                "source_sha256": fx.sha256_text(fx.TEXTS[key]), "indexed_signature": "sig-1", "chunk_count": 3}

    def runtime(self):
        return Runtime(engine=None, database_url=self.url, environment=self.environment,
                       environment_explicit=self.explicit, upload_dir="uploads", upload_root=self.repo / "uploads")

    def tools(self, runtime):
        @contextmanager
        def open_session():
            self.sessions += 1
            yield FakeSession(self)

        return cli.Tools(self.tools_embedding_id, open_session,
                         lambda: SimpleNamespace(embedding_id=self.embedder_id))

    def load_inputs(self, dev, corpus):
        self.inputs_loaded += 1
        self.dev_path = dev
        return self.inputs

    def corpus(self, directory):
        files = [SimpleNamespace(corpus_key=k, filename=f"{k}.txt") for k in ("doc-a", "doc-b")]
        return SimpleNamespace(manifest=self.corpus_manifest, files=files)

    def git_reader(self, repo):
        if self.git_hook:
            self.git_hook()
        return self.git

    def run(self, *extra, out=None, monkeypatch=None):
        argv = ["--database", "documind_db", "--owner-id", str(OWNER_ID), "--evaluation-email-sha256", DIGEST,
                "--out", str(self.out if out is None else out), "--resamples", "1000", *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        code = cli.main(
            argv, runtime_loader=self.runtime, tools_loader=self.tools, inputs_loader=self.load_inputs,
            corpus_loader=self.corpus, git_reader=self.git_reader, code_identity=lambda: IDENTITY,
            environ={}, cwd=self.cwd, repo_root=self.repo, stdout=stdout, stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    c = Ctx(tmp_path)
    monkeypatch.setattr("app.rag.evaluation.snapshot.compare_snapshots",
                        lambda before, after: SimpleNamespace(identical=c.identical))
    return c


# ---------- success and privacy ----------

def test_a_successful_run_writes_one_report_outside_the_repository(ctx):
    code, out, err = ctx.run(*ADJ)
    assert (code, err) == (0, "")
    assert ctx.out.is_file() and ctx.sessions == 2 and ctx.calls == ["snapshot", "verify", "snapshot"]
    assert rep.load_report(ctx.out.read_bytes()) is not None
    assert "Validity pilot" in out and "not the V1 baseline" in out
    assert ctx.verify_args == (OWNER_ID, DIGEST, fx.EMB) and ctx.top_k == 20
    assert ctx.pool_args[:2] == (OWNER_ID, fx.EMB) and set(ctx.pool_args[2]) == {fx.ID_A, fx.ID_B}
    assert not any(ctx.repo in p.parents for p in [ctx.out])


def test_dev_only_and_the_default_dev_path(ctx):
    ctx.run(*ADJ)
    assert ctx.dev_path.name == "dev.jsonl"
    assert "test" not in ctx.dev_path.name


def test_the_summary_holds_no_question_text_ids_or_credentials(ctx):
    _, out, err = ctx.run(*ADJ)
    text = out + err + ctx.out.read_text(encoding="utf-8")
    for question in QUESTIONS:
        assert question.question not in text
    for secret in ("hunter2", "user:", str(OWNER_ID), DIGEST, "evaluation@"):
        assert secret not in out + err
    assert str(OWNER_ID) not in ctx.out.read_text(encoding="utf-8")
    for doc_id in (fx.ID_A, fx.ID_B):
        assert not re.search(rf"\b{doc_id}\b", out + err)
    assert "localhost:5432/documind_db" in out


def test_the_retriever_never_sees_unanswerable_text(ctx):
    ctx.run(*ADJ)
    assert fx.U1.question not in ctx.retriever.queries and fx.U2.question not in ctx.retriever.queries
    assert len(ctx.retriever.queries) == 4


def test_the_report_records_identical_snapshots_and_code_identity(ctx):
    ctx.run(*ADJ)
    document = __import__("json").loads(ctx.out.read_text(encoding="utf-8"))
    assert document["database_snapshots"] == {
        "before_fingerprint": FINGERPRINT, "after_fingerprint": FINGERPRINT, "unchanged": True}


# ---------- output path ----------

def test_the_output_path_must_be_absolute_outside_new_and_in_an_existing_folder(ctx, tmp_path):
    assert ctx.run(*ADJ, out="relative.json")[0] == 2
    assert ctx.run(*ADJ, out=ctx.repo / "report.json")[0] == 2
    assert ctx.run(*ADJ, out=ctx.outside / "missing" / "r.json")[0] == 2
    ctx.out.write_text("keep", encoding="utf-8")
    assert ctx.run(*ADJ)[0] == 2 and ctx.out.read_text(encoding="utf-8") == "keep"
    link = ctx.outside / "dangling.json"
    try:
        os.symlink(ctx.outside / "nowhere", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    assert ctx.run(*ADJ, out=link)[0] == 2
    assert ctx.sessions == 0 and ctx.inputs_loaded == 0


def test_a_report_created_during_the_run_is_never_overwritten(ctx):
    ctx.git_hook = lambda: ctx.out.write_text("someone else", encoding="utf-8")
    code, _, _ = ctx.run(*ADJ)
    assert code == 2 and ctx.out.read_text(encoding="utf-8") == "someone else"


# ---------- refusals (exit 2, nothing read) ----------

@pytest.mark.parametrize("name", ["test.jsonl", "dev.json", "other.jsonl", "dev.jsonl.bak"])
def test_only_a_file_named_dev_jsonl_is_accepted(ctx, name):
    code, _, _ = ctx.run(*ADJ, "--dev-file", str(ctx.repo / "eval" / "dataset" / name))
    assert code == 2 and ctx.inputs_loaded == 0 and ctx.sessions == 0 and not ctx.out.exists()


@pytest.mark.parametrize("change", [
    {"url": "postgresql://u:p@db.example.com:5432/documind_db"},
    {"url": "postgresql://u:p@localhost:5432/production"},
    {"url": "postgresql://u:p@localhost:5432/documind_db?host=elsewhere"},
    {"url": "mysql://u:p@localhost/documind_db"},
    {"environment": "production"}, {"environment": None}, {"explicit": False},
])
def test_the_static_safety_gates_refuse_before_anything_is_read(ctx, change):
    for name, value in change.items():
        setattr(ctx, name, value)
    code, _, err = ctx.run(*ADJ)
    assert code == 2 and ctx.inputs_loaded == 0 and ctx.sessions == 0 and not ctx.out.exists()
    assert "hunter2" not in err


def test_a_wrong_working_directory_or_a_missing_env_file_is_refused(ctx):
    ctx.cwd = ctx.outside
    assert ctx.run(*ADJ)[0] == 2
    ctx.cwd = ctx.repo
    (ctx.repo / ".env").unlink()
    assert ctx.run(*ADJ)[0] == 2 and ctx.sessions == 0


def test_the_database_argument_must_match_the_url(ctx):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        ["--database", "other_db", "--owner-id", "1", "--evaluation-email-sha256", DIGEST, "--out", str(ctx.out)],
        runtime_loader=ctx.runtime, tools_loader=ctx.tools, inputs_loader=ctx.load_inputs, corpus_loader=ctx.corpus,
        git_reader=ctx.git_reader, code_identity=lambda: IDENTITY, environ={}, cwd=ctx.repo, repo_root=ctx.repo,
        stdout=stdout, stderr=stderr)
    assert code == 2 and ctx.sessions == 0


def test_an_embedder_id_that_differs_from_the_stored_one_is_refused(ctx):
    ctx.embedder_id = "other-model@rev"
    assert ctx.run(*ADJ)[0] == 2 and ctx.sessions == 0


def test_fewer_than_one_thousand_seeds_are_refused(ctx):
    assert ctx.run(*ADJ, "--random-seeds", "999")[0] == 2 and ctx.sessions == 0


def test_a_corpus_manifest_that_is_not_the_one_of_the_inputs_is_refused(ctx):
    ctx.corpus_manifest = dataclasses.replace(fx.MANIFEST, corpus_version="c2")
    assert ctx.run(*ADJ)[0] == 2 and ctx.sessions == 0
    ctx.corpus_manifest = dataclasses.replace(fx.MANIFEST, stage="frozen")
    assert ctx.run(*ADJ)[0] == 2 and ctx.sessions == 0


@pytest.mark.parametrize("bad", ["1", "q4", ":u1", "q4:", "a:b:c"])
def test_a_malformed_adjudication_is_a_usage_error(ctx, bad):
    assert ctx.run("--adjudicate", bad)[0] == 2


def test_a_missing_required_argument_is_a_usage_error():
    assert cli.main(["--database", "documind_db"], stdout=io.StringIO(), stderr=io.StringIO()) == 2


# ---------- failures after reading started (exit 1) ----------

def test_a_live_database_name_mismatch_is_refused_before_any_snapshot(ctx):
    ctx.live_problem = "The connected database is 'production', not 'documind_db'"
    code, _, _ = ctx.run(*ADJ)
    assert code == 2 and ctx.calls == [] and not ctx.out.exists()


def test_a_read_write_transaction_is_refused(ctx):
    ctx.readonly_problem = "the database transaction is not read-only"
    code, _, _ = ctx.run(*ADJ)
    assert code == 2 and ctx.calls == [] and not ctx.out.exists()


def test_a_corpus_that_does_not_verify_stops_the_run(ctx):
    check = SimpleNamespace(ok=False, name="owner")
    ctx.verification = SimpleNamespace(ok=False, problems=["ownership"], documents=[
        SimpleNamespace(corpus_key="doc-a", checks=[check])])
    code, _, err = ctx.run(*ADJ)
    assert code == 1 and not ctx.out.exists() and "doc-a: owner" in err and ctx.sessions == 1


def test_a_document_owned_by_someone_else_stops_the_run(ctx):
    ctx.rows[0]["owner_id"] = OWNER_ID + 1
    assert ctx.run(*ADJ)[0] == 1 and not ctx.out.exists()


def test_documents_are_found_by_their_stored_description_not_by_id(ctx):
    ctx.rows = [ctx.row(5001, "doc-a"), ctx.row(5002, "doc-b")]
    moved = {i: dataclasses.replace(c, document_id=5001 if c.document_id == fx.ID_A else 5002) for i, c in fx.C.items()}
    ctx.chunks = list(moved.values())
    ctx.retriever = lambda query: [moved[i] for i in fx.DENSE[query]]
    code, _, _ = ctx.run(*ADJ)
    assert code == 0 and set(ctx.pool_args[2]) == {5001, 5002}


def test_a_description_that_names_the_wrong_corpus_version_is_refused(ctx):
    ctx.rows[0]["description"] = "c9:doc-a"
    assert ctx.run(*ADJ)[0] == 1


def test_a_sql_pool_that_lacks_a_stored_chunk_fails_parity(ctx):
    ctx.chunks = list(fx.C.values())[:-1]
    assert ctx.run(*ADJ)[0] == 1 and not ctx.out.exists()


def test_a_sql_pool_with_a_chunk_of_a_foreign_document_fails_parity(ctx):
    ctx.chunks = [*fx.C.values(), dataclasses.replace(fx.C[1], chunk_id=99, document_id=123456)]
    assert ctx.run(*ADJ)[0] == 1 and not ctx.out.exists()


def test_a_dense_ranking_outside_the_pool_is_refused(ctx):
    stranger = dataclasses.replace(fx.C[1], chunk_id=77)
    ctx.retriever = lambda query: [stranger, *[fx.C[i] for i in (2, 3, 4, 5, 6)]]
    assert ctx.run(*ADJ)[0] == 1 and not ctx.out.exists()


def test_a_database_change_between_the_snapshots_blocks_the_report(ctx):
    ctx.identical = False
    code, _, err = ctx.run(*ADJ)
    assert code == 1 and not ctx.out.exists() and "changed" in err and ctx.sessions == 2


def test_failed_oracle_validity_still_writes_the_report_but_exits_one(ctx):
    code, out, err = ctx.run()  # q4's unreachable unit is not adjudicated
    assert code == 1 and ctx.out.is_file() and "oracle validity" in err and "FAILED" in out


def test_an_unexpected_error_shows_only_the_class_name(ctx):
    def boom(dev, corpus):
        raise RuntimeError("password=hunter2")

    ctx.load_inputs = boom
    code, out, err = ctx.run(*ADJ)
    assert code == 1 and "hunter2" not in out + err and "RuntimeError" in err


def test_the_cli_modules_never_commit_or_write_to_the_database():
    root = Path(__file__).resolve().parent.parent
    for name in ("app/rag/pilot_cli.py", "scripts/e17_run_pilot.py"):
        source = (root / name).read_text(encoding="utf-8")
        for word in (".commit(", "session.add", "INSERT", "UPDATE ", "DELETE FROM"):
            assert word not in source
        assert "test.jsonl" not in source


def test_a_stored_chunk_count_that_the_sql_pool_does_not_match_fails_parity_alone(ctx):
    ctx.rows[1]["chunk_count"] = 4  # the pool still holds three chunks of this document
    code, _, err = ctx.run(*ADJ)
    assert code == 1 and not ctx.out.exists() and "PARITY_COUNT" in err
