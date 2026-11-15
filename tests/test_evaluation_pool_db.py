"""E1.7.5: the database part of the pilot, tested with fakes and by compiling the real SQL.

A  documents_from_rows
B  check_pool_parity (Python reading versus the SQL pool)
C  the pool statement uses the real retrievable_filter and the dense query's predicates
D  DbSession over a fake connection: only SELECTs, read-only checks
E  read_only_session: options before the first statement, rollback, never commit

Run focused: pytest tests/test_evaluation_pool_db.py --noconftest -p no:cacheprovider
"""
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.evaluation import pool as p
from app.rag.evaluation import pool_db as db
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB = "emb-1"
SHA = "a" * 64


def row(doc_id, **over):
    base = {
        "id": doc_id, "owner_id": 5, "filename": f"f{doc_id}.txt", "description": f"c1:k{doc_id}",
        "source_version": 1, "indexed_source_version": 1, "indexed_embedding_id": EMB,
        "source_sha256": SHA, "indexed_signature": "sig-1", "chunk_count": 3,
    }
    base.update(over)
    return base


def codes(error):
    return {i.code for i in error.issues}


def refused(code, fn, *args):
    with pytest.raises(s.EvaluationDataError) as info:
        fn(*args)
    assert codes(info.value) == {code}


# ---------- A ----------

def test_rows_become_pool_documents_with_counts_and_one_signature():
    facts = db.documents_from_rows([row(10), row(20, chunk_count=4)])
    assert [d.document_id for d in facts.documents] == [10, 20]
    assert facts.documents[0] == p.PoolDocument(10, 5, "f10.txt", "c1:k10", 1, 1, EMB, SHA)
    assert dict(facts.chunk_counts) == {10: 3, 20: 4} and facts.signature == "sig-1"


def test_a_row_with_a_missing_or_mistyped_field_is_refused():
    bad = row(10)
    del bad["owner_id"]
    refused(db.ROWS_INVALID, db.documents_from_rows, [bad])
    refused(db.ROWS_INVALID, db.documents_from_rows, [row(10, chunk_count="3")])
    refused(db.ROWS_INVALID, db.documents_from_rows, [row(True)])
    refused(db.ROWS_INVALID, db.documents_from_rows, [None])


def test_documents_must_share_one_nonempty_signature():
    refused(db.ROWS_SIGNATURE, db.documents_from_rows, [row(10), row(20, indexed_signature="sig-2")])
    refused(db.ROWS_SIGNATURE, db.documents_from_rows, [row(10, indexed_signature=None)])
    refused(db.ROWS_SIGNATURE, db.documents_from_rows, [row(10, indexed_signature="")])
    refused(db.ROWS_SIGNATURE, db.documents_from_rows, [])


# ---------- B ----------

def chunk(doc_id, n=1):
    return [SimpleNamespace(document_id=doc_id) for _ in range(n)]


def docs():
    return facts_docs(db.documents_from_rows([row(10), row(20)]))


def facts_docs(facts):
    return facts.documents, dict(facts.chunk_counts)


def test_parity_holds_when_sql_has_every_chunk_of_every_retrievable_document():
    documents, counts = docs()
    db.check_pool_parity(documents, counts, EMB, chunk(10, 3) + chunk(20, 3))


def test_a_document_that_python_accepts_but_sql_returns_nothing_for_is_refused():
    documents, counts = docs()
    refused(db.PARITY_PYTHON_ONLY, db.check_pool_parity, documents, counts, EMB, chunk(10, 3))


def test_sql_chunks_of_a_document_python_rejects_are_refused():
    facts = db.documents_from_rows([row(10), row(20, indexed_source_version=None)])
    refused(db.PARITY_SQL_ONLY, db.check_pool_parity, facts.documents, dict(facts.chunk_counts), EMB,
            chunk(10, 3) + chunk(20, 1))
    other = db.documents_from_rows([row(10), row(20, indexed_embedding_id="other")])
    refused(db.PARITY_SQL_ONLY, db.check_pool_parity, other.documents, dict(other.chunk_counts), EMB,
            chunk(10, 3) + chunk(20, 1))


def test_a_document_without_sql_chunks_is_fine_when_python_also_rejects_it():
    facts = db.documents_from_rows([row(10), row(20, indexed_source_version=None)])
    db.check_pool_parity(facts.documents, dict(facts.chunk_counts), EMB, chunk(10, 3))


def test_the_sql_pool_must_hold_every_stored_chunk():
    documents, counts = docs()
    refused(db.PARITY_COUNT, db.check_pool_parity, documents, counts, EMB, chunk(10, 3) + chunk(20, 2))
    refused(db.PARITY_COUNT, db.check_pool_parity, documents, counts, EMB, chunk(10, 3) + chunk(20, 4))


def test_a_chunk_of_an_unmapped_document_is_refused():
    documents, counts = docs()
    refused(db.PARITY_FOREIGN, db.check_pool_parity, documents, counts, EMB, chunk(10, 3) + chunk(20, 3) + chunk(99))


def test_parity_input_is_validated():
    documents, counts = docs()
    for args in (("x", counts, EMB, []), (documents, [], EMB, []), (documents, counts, 5, []),
                 (documents, counts, EMB, [object()]), ([1], counts, EMB, [])):
        refused(db.PARITY_INPUT, db.check_pool_parity, *args)


# ---------- C ----------

_COMPILE = r"""
import json
from sqlalchemy.dialects import postgresql
import numpy as np
import app.models.user, app.models.document, app.models.chunk  # noqa: F401
from app.rag.evaluation.pool_db import build_pool_statement
from app.rag.indexing_service import retrievable_filter
from app.rag.retrieval import _build_retrieval_statement

def sql(statement):
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

pool = sql(build_pool_statement(owner_id=7, embedding_id="emb-1", document_ids=[3, 4]))
vector = np.zeros(384, dtype="float32"); vector[0] = 1.0
dense = sql(_build_retrieval_statement(owner_id=7, query_vector=vector, embedding_id="emb-1", top_k=20, document_ids=(3, 4)))
filt = sql(retrievable_filter("emb-1"))
print(json.dumps({"pool": pool, "dense": dense, "filter": filt}))
"""


@pytest.fixture(scope="module")
def compiled():
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                "DATABASE_URL": "postgresql://u:p@localhost/x", "SECRET_KEY": "k" * 40})
    done = subprocess.run([sys.executable, "-c", _COMPILE], cwd=PROJECT_ROOT, env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode != 0:
        pytest.skip("the retrieval code is not importable here (for example numpy is missing)")
    return json.loads(done.stdout)


def _normal(text):
    return " ".join(text.split())


def test_the_pool_statement_embeds_the_real_retrievable_filter(compiled):
    assert _normal(compiled["filter"]) in _normal(compiled["pool"])
    assert "indexed_source_version IS NOT NULL" in compiled["filter"]
    assert "indexed_embedding_id = 'emb-1'" in compiled["filter"]


def test_the_pool_and_the_dense_query_share_the_filter_and_the_owner_predicate(compiled):
    for text in (compiled["pool"], compiled["dense"]):
        assert _normal(compiled["filter"]) in _normal(text)
        assert "documents.owner_id = 7" in text
        assert "documents.id IN (3, 4)" in text


def test_the_pool_statement_is_a_plain_select(compiled):
    sql = compiled["pool"].lstrip().upper()
    assert sql.startswith("SELECT")
    for word in ("INSERT", "UPDATE ", "DELETE", "DROP", "ALTER", "TRUNCATE", "LIMIT"):
        assert word not in sql
    assert "embedding" not in compiled["pool"].replace("indexed_embedding_id", "")  # no vector column


def test_the_pool_statement_refuses_bad_arguments():
    code = (
        "from app.rag.evaluation.pool_db import build_pool_statement as b\n"
        "bad = 0\n"
        "for kw in ({'owner_id': True, 'embedding_id': 'e', 'document_ids': [1]},"
        " {'owner_id': 1, 'embedding_id': '', 'document_ids': [1]},"
        " {'owner_id': 1, 'embedding_id': 'e', 'document_ids': []},"
        " {'owner_id': 1, 'embedding_id': 'e', 'document_ids': ['1']}):\n"
        "    try:\n        b(**kw)\n    except Exception:\n        bad += 1\n"
        "print(bad)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                "DATABASE_URL": "postgresql://u:p@localhost/x", "SECRET_KEY": "k" * 40})
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip("the retrieval code is not importable here")
    assert done.stdout.strip() == "4"


# ---------- D ----------

class FakeResult:
    def __init__(self, rows=None, scalar=None):
        self._rows, self._scalar = rows or [], scalar

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar


class FakeConnection:
    def __init__(self, answers=None, fail=None):
        self.statements, self.answers, self.fail = [], answers or {}, fail
        self.options = None
        self.rolled_back = 0

    def execute(self, statement, *args, **kwargs):
        text = str(statement)
        self.statements.append(text)
        if self.fail:
            raise self.fail
        for needle, result in self.answers.items():
            if needle in text:
                return result
        return FakeResult()

    def commit(self):
        raise AssertionError("a read-only session must never commit")

    def rollback(self):
        self.rolled_back += 1


def test_assert_read_only_accepts_only_a_read_only_repeatable_read_transaction():
    ok = FakeConnection({"transaction_read_only": FakeResult(scalar="on"), "transaction_isolation": FakeResult(scalar="repeatable read")})
    assert db.DbSession(ok, upload_root=Path(".")).assert_read_only() is None
    assert [x.strip() for x in ok.statements] == ["SHOW transaction_read_only", "SHOW transaction_isolation"]
    off = FakeConnection({"transaction_read_only": FakeResult(scalar="off"), "transaction_isolation": FakeResult(scalar="repeatable read")})
    assert "not read-only" in db.DbSession(off, upload_root=Path(".")).assert_read_only()
    low = FakeConnection({"transaction_read_only": FakeResult(scalar="on"), "transaction_isolation": FakeResult(scalar="read committed")})
    assert "REPEATABLE READ" in db.DbSession(low, upload_root=Path(".")).assert_read_only()


def test_a_failing_mode_check_reports_only_the_exception_class():
    broken = FakeConnection(fail=RuntimeError("password=hunter2 host=10.0.0.1"))
    message = db.DbSession(broken, upload_root=Path(".")).assert_read_only()
    assert "RuntimeError" in message and "hunter2" not in message and "10.0.0.1" not in message


def test_check_live_compares_the_server_database_name():
    good = FakeConnection({"current_database": FakeResult(scalar="documind_db")})
    assert db.DbSession(good, upload_root=Path(".")).check_live("documind_db") is None
    wrong = db.DbSession(FakeConnection({"current_database": FakeResult(scalar="production")}), upload_root=Path("."))
    assert wrong.check_live("documind_db")
    boom = db.DbSession(FakeConnection(fail=RuntimeError("secret detail")), upload_root=Path("."))
    assert "secret detail" not in boom.check_live("documind_db")


def test_pool_chunks_runs_one_select_and_converts_rows(monkeypatch):
    marker = SimpleNamespace()
    seen = {}

    def fake_statement(**kwargs):
        seen.update(kwargs)
        return marker

    monkeypatch.setattr(db, "build_pool_statement", fake_statement)
    rows = [SimpleNamespace(chunk_id=1, document_id=10, chunk_index=0, start_offset=0, end_offset=5, page_start=None,
                            page_end=None, text="hello", span_sha256=SHA, source_version=1, source_sha256=SHA)]

    class Conn(FakeConnection):
        def execute(self, statement, *a, **k):
            assert statement is marker
            self.statements.append("SELECT")
            return FakeResult(rows)

    conn = Conn()
    result = db.DbSession(conn, upload_root=Path(".")).pool_chunks(owner_id=5, embedding_id=EMB, document_ids=[10, 20])
    assert conn.statements == ["SELECT"] and seen == {"owner_id": 5, "embedding_id": EMB, "document_ids": [10, 20]}
    assert result == [db.PoolChunkRow(1, 10, 0, 0, 5, None, None, "hello", SHA, 1, SHA)]


def test_the_dense_retriever_calls_the_unmodified_retrieval_with_owner_ids_and_top_k(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(retrieve_for_query=lambda conn, **kw: calls.append((conn, kw)) or ["chunk"])
    monkeypatch.setitem(sys.modules, "app.rag.retrieval", fake_module)
    conn, embedder = object(), object()
    retriever = db.DbSession(conn, upload_root=Path(".")).dense_retriever(
        embedder, owner_id=5, document_ids=[10, 20], top_k=20)
    assert retriever("a question") == ["chunk"]
    assert calls == [(conn, {"owner_id": 5, "query": "a question", "embedder": embedder, "top_k": 20,
                             "document_ids": (10, 20)})]


# ---------- E ----------

class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()
        self.events = []

    @contextmanager
    def connect(self):
        engine = self

        class Conn(FakeConnection):
            def execution_options(self, **options):
                engine.events.append(("options", options))
                return self

            def execute(self, statement, *a, **k):
                engine.events.append(("execute", str(statement)))
                return FakeResult(scalar="repeatable read" if "isolation" in str(statement) else "on")

            def rollback(self):
                engine.events.append(("rollback", None))

        try:
            yield Conn()
        finally:
            engine.events.append(("closed", None))


def test_the_session_applies_read_only_repeatable_read_before_any_statement_and_rolls_back():
    engine = FakeEngine()
    with db.read_only_session(engine, upload_root=Path(".")) as session:
        assert session.assert_read_only() is None
    kinds = [kind for kind, _ in engine.events]
    assert kinds[0] == "options"
    assert engine.events[0][1] == {"isolation_level": "REPEATABLE READ", "postgresql_readonly": True}
    assert kinds[-2:] == ["rollback", "closed"] and "commit" not in kinds


def test_the_session_rolls_back_even_when_the_body_raises():
    engine = FakeEngine()
    with pytest.raises(RuntimeError), db.read_only_session(engine, upload_root=Path(".")):
        raise RuntimeError("boom")
    assert [k for k, _ in engine.events][-2:] == ["rollback", "closed"]


def test_take_snapshot_uses_only_the_snapshot_module(monkeypatch):
    seen = {}

    def collect(run_query, *, created_at):
        seen["created_at"] = created_at
        return {"fingerprint": SHA}

    monkeypatch.setattr("app.rag.evaluation.snapshot.collect_snapshot", collect)
    stamp = datetime(2026, 11, 14, tzinfo=timezone.utc)  # noqa: UP017
    assert db.DbSession(object(), upload_root=Path(".")).take_snapshot(lambda: stamp) == {"fingerprint": SHA}
    assert seen["created_at"] == stamp


def test_the_database_modules_never_commit_or_write():
    source = (PROJECT_ROOT / "app" / "rag" / "evaluation" / "pool_db.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(source)
    strings = {id(n) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    code = "\n".join(ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Call, ast.Attribute)))
    del strings
    for word in (".commit(", ".flush(", ".merge(", "session.add(", "insert(", "update(", "delete(", ):
        assert word not in code
