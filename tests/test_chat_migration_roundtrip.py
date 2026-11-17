"""C7.3 migration round trip for migrations #6 and #7 (conversations, messages,
chat_requests, chat_request_attempts), with existing C1 data in the database.

Frozen R3 section 15 sequence:
  upgrade to 0b0212c1cca6 + seed user/document/chunk
  -> upgrade to #6 + seed conversation/messages
  -> upgrade to #7 + seed requests/attempts
  -> downgrade to #6 -> downgrade to 0b0212c1cca6 -> re-upgrade to head.

Runs ONLY against the dedicated database named documind_migration_test, given by
MIGRATION_TEST_DATABASE_URL, and is skipped when that variable is unset. It never reads
.env, DATABASE_URL or TEST_DATABASE_URL for its own target, and it never touches
documind_test or documind_db (the two conftest fixtures that reset documind_test are
overridden below, for this module only).

Safety (the four guards of tests/test_migration_roundtrip.py, copied unchanged): before
the schema reset or any Alembic command, the fixture proves that
(1) the URL names exactly documind_migration_test, (2) migrations/env.py still takes its
URL from app.core.config.settings.database_url, (3) that settings path, resolved in a
subprocess with the exact environment and working directory the Alembic subprocesses will
use, yields the same database, host and port, and (4) the server reports
current_database() == documind_migration_test.

Downgrades delete chat data by design (frozen section 15); this test asserts that loss.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_ENV_PY = PROJECT_ROOT / "migrations" / "env.py"
MIGRATION_TEST_DATABASE_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")
MIGRATION_TEST_DATABASE_NAME = "documind_migration_test"

C1_HEAD = "0b0212c1cca6"   # add pgvector and document chunks
MIGRATION_6 = "4a06e23062ae"   # conversations, messages
MIGRATION_7 = "e9749572298d"   # chat_requests, chat_request_attempts
MIGRATION_8 = "c4f1a7d92b60"   # attempt status 'rejected' (three CHECKs)
MIGRATION_9 = "a7d3e5f10c48"   # message_citations
CITATION_TABLE = "message_citations"

CHAT_TABLES_6 = ["conversations", "messages"]
CHAT_TABLES_7 = ["chat_requests", "chat_request_attempts"]
ALL_CHAT_TABLES = CHAT_TABLES_6 + CHAT_TABLES_7

# The exact lines in migrations/env.py that decide which database Alembic uses.
ENV_PY_URL_SOURCE_LINES = (
    "from app.core.config import settings",
    'config.set_main_option("sqlalchemy.url", settings.database_url)',
)

# Imports the same settings object env.py imports and reports where it points.
RESOLVE_SETTINGS_URL = (
    "import json\n"
    "from sqlalchemy.engine import make_url\n"
    "from app.core.config import settings\n"
    "url = make_url(settings.database_url)\n"
    "print(json.dumps({'database': url.database, 'host': url.host, 'port': url.port}))\n"
)

pytestmark = pytest.mark.skipif(
    not MIGRATION_TEST_DATABASE_URL,
    reason=(
        "Required C7.3 check: set MIGRATION_TEST_DATABASE_URL to the "
        f"'{MIGRATION_TEST_DATABASE_NAME}' database to run."
    ),
)

_safety_checks_passed = False


# ---------- fixtures that must NOT run for this module ----------
# tests/conftest.py has two autouse fixtures that connect to documind_test: a session
# fixture that drops and rebuilds its schema, and a per-test one that truncates it.
# Defining fixtures with the same names here replaces them for this module only, so this
# test never touches documind_test (tests in other modules still use the real ones).

@pytest.fixture(scope="session", autouse=True)
def _database():
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    yield


# ---------- the four safety guards (copied from tests/test_migration_roundtrip.py) ----------

def alembic_subprocess_env():
    """The environment every Alembic subprocess (and the preflight) runs with."""
    return {**os.environ, "DATABASE_URL": MIGRATION_TEST_DATABASE_URL}


def assert_env_py_reads_settings_database_url():
    source = ALEMBIC_ENV_PY.read_text(encoding="utf-8")
    for line in ENV_PY_URL_SOURCE_LINES:
        if line not in source:
            raise RuntimeError(
                f"Refusing to run: {ALEMBIC_ENV_PY} no longer contains {line!r}, so the "
                "database Alembic targets can no longer be verified."
            )


def resolve_alembic_target():
    """Resolve settings.database_url exactly as an Alembic subprocess would."""
    result = subprocess.run(
        [sys.executable, "-c", RESOLVE_SETTINGS_URL],
        cwd=PROJECT_ROOT,
        env=alembic_subprocess_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_alembic(*args):
    if not _safety_checks_passed:
        raise RuntimeError("Refusing to run Alembic before the migration database safety checks.")
    subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=PROJECT_ROOT,
        env=alembic_subprocess_env(),
        check=True,
    )


def run_alembic_expecting_failure(*args):
    """Run Alembic and return its exit code and combined output; the caller asserts failure."""
    if not _safety_checks_passed:
        raise RuntimeError("Refusing to run Alembic before the migration database safety checks.")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=PROJECT_ROOT,
        env=alembic_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


REJECTED_ATTEMPT_SQL = text(
    "INSERT INTO chat_request_attempts (request_id, attempt_number, status, reserved_tokens, "
    "charged_tokens, llm_started_at, finished_at, error_code) "
    "VALUES (:r, 2, 'rejected', 100, 0, now(), now(), 'provider_rejected')")


@pytest.fixture(scope="module")
def migration_engine():
    global _safety_checks_passed

    # 1. The URL we were given names exactly the dedicated database.
    url = make_url(MIGRATION_TEST_DATABASE_URL)
    if url.database != MIGRATION_TEST_DATABASE_NAME:
        raise RuntimeError(
            f"Refusing to run: MIGRATION_TEST_DATABASE_URL must name exactly "
            f"'{MIGRATION_TEST_DATABASE_NAME}', got '{url.database}'."
        )

    # 2. Alembic's env.py still takes its URL from settings.database_url.
    assert_env_py_reads_settings_database_url()

    # 3. That settings path, under the Alembic subprocess environment, resolves to it.
    target = resolve_alembic_target()
    expected = {"database": MIGRATION_TEST_DATABASE_NAME, "host": url.host, "port": url.port}
    if target != expected:
        raise RuntimeError(
            f"Refusing to run: Alembic would target {target}, expected {expected}."
        )

    # 4. The server confirms the database before anything is reset.
    engine = create_engine(url)
    with engine.begin() as conn:
        actual = conn.execute(text("SELECT current_database()")).scalar_one()
        if actual != MIGRATION_TEST_DATABASE_NAME:
            raise RuntimeError(
                f"Refusing to reset '{actual}'; expected '{MIGRATION_TEST_DATABASE_NAME}'."
            )
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    _safety_checks_passed = True
    try:
        yield engine
    finally:
        _safety_checks_passed = False
        engine.dispose()


# ---------- helpers ----------

def current_revision(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def table_names(engine):
    return set(inspect(engine).get_table_names())


def extension_version(engine):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()


def row_count(engine, table):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def chat_request_objects_left(engine):
    """Names of constraints or indexes belonging to the #7 tables that still exist."""
    with engine.connect() as conn:
        constraints = conn.execute(text(
            "SELECT conname FROM pg_constraint "
            "WHERE conname LIKE '%chat\\_request%' ORDER BY conname")).scalars().all()
        indexes = conn.execute(text(
            "SELECT indexname FROM pg_indexes "
            "WHERE indexname LIKE '%chat\\_request%' ORDER BY indexname")).scalars().all()
    return constraints + indexes


def schema_snapshot(engine, tables):
    """Primary key, constraint, foreign key and index names (with definitions) of the tables."""
    insp = inspect(engine)
    snapshot = {}
    for table in tables:
        primary_key = insp.get_pk_constraint(table)
        snapshot[table] = {
            "primary_key": (primary_key["name"], tuple(primary_key["constrained_columns"])),
            "checks": sorted(c["name"] for c in insp.get_check_constraints(table)),
            "uniques": sorted((u["name"], tuple(u["column_names"]))
                              for u in insp.get_unique_constraints(table)),
            "foreign_keys": sorted(
                (f["name"], tuple(f["constrained_columns"]), f["referred_table"],
                 f["options"].get("ondelete"))
                for f in insp.get_foreign_keys(table)),
            "indexes": sorted(
                (i["name"], tuple(i["column_names"]), bool(i["unique"]),
                 i.get("dialect_options", {}).get("postgresql_where") or "")
                for i in insp.get_indexes(table)),
        }
    return snapshot


def seed_c1_data(engine):
    """A user, a document and one chunk with an embedding (the pre-chat C1 data)."""
    with engine.begin() as conn:
        user_id = conn.execute(text(
            "INSERT INTO users (email, hashed_password) "
            "VALUES ('c1-user@example.com', 'x') RETURNING id")).scalar_one()
        doc_id = conn.execute(text(
            "INSERT INTO documents (filename, owner_id, processing_status, extracted_text) "
            "VALUES ('c1.txt', :owner, 'completed', 'c1 source text') RETURNING id"
        ), {"owner": user_id}).scalar_one()
        chunk_id = conn.execute(text(
            "INSERT INTO document_chunks "
            "(document_id, chunk_index, start_offset, end_offset, text, span_sha256, embedding) "
            "VALUES (:doc, 0, 0, 2, 'c1', :hash, CAST(:embedding AS vector)) RETURNING id"
        ), {
            "doc": doc_id,
            "hash": hashlib.sha256(b"c1").hexdigest(),
            "embedding": "[" + ",".join(["0.5"] * 384) + "]",
        }).scalar_one()
    return {"user": user_id, "document": doc_id, "chunk": chunk_id}


def assert_c1_data_intact(engine, ids, installed_version):
    """The seeded C1 rows are unchanged, and pgvector is the same version and works."""
    with engine.connect() as conn:
        user = conn.execute(text("SELECT email FROM users WHERE id = :id"),
                            {"id": ids["user"]}).one()
        document = conn.execute(text(
            "SELECT filename, owner_id, processing_status, extracted_text "
            "FROM documents WHERE id = :id"), {"id": ids["document"]}).one()
        chunk = conn.execute(text(
            "SELECT document_id, chunk_index, start_offset, end_offset, text, span_sha256 "
            "FROM document_chunks WHERE id = :id"), {"id": ids["chunk"]}).one()
        # the stored embedding is the seeded vector: its cosine distance to it is ~0
        stored_distance = conn.execute(text(
            "SELECT embedding <=> CAST(:embedding AS vector) FROM document_chunks WHERE id = :id"
        ), {"id": ids["chunk"], "embedding": "[" + ",".join(["0.5"] * 384) + "]"}).scalar_one()
        # vector operators still work on fresh literals
        orthogonal_distance = conn.execute(
            text("SELECT '[1,0]'::vector <=> '[0,1]'::vector")).scalar_one()
    assert tuple(user) == ("c1-user@example.com",)
    assert tuple(document) == ("c1.txt", ids["user"], "completed", "c1 source text")
    assert tuple(chunk) == (ids["document"], 0, 0, 2, "c1", hashlib.sha256(b"c1").hexdigest())
    assert stored_distance == pytest.approx(0.0, abs=1e-6)
    assert orthogonal_distance == pytest.approx(1.0)
    assert extension_version(engine) == installed_version
    assert row_count(engine, "users") == 1
    assert row_count(engine, "documents") == 1
    assert row_count(engine, "document_chunks") == 1


def seed_chat_6_data(engine, user_id):
    """A conversation with a user and an assistant message (tables from migration #6)."""
    with engine.begin() as conn:
        conversation_id = conn.execute(text(
            "INSERT INTO conversations (owner_id, title) "
            "VALUES (:owner, 'Round trip chat') RETURNING id"), {"owner": user_id}).scalar_one()
        user_message_id = conn.execute(text(
            "INSERT INTO messages (conversation_id, seq, role, content) "
            "VALUES (:c, 1, 'user', 'What is pgvector?') RETURNING id"),
            {"c": conversation_id}).scalar_one()
        assistant_message_id = conn.execute(text(
            "INSERT INTO messages (conversation_id, seq, role, content) "
            "VALUES (:c, 2, 'assistant', 'A Postgres extension.') RETURNING id"),
            {"c": conversation_id}).scalar_one()
    return {"conversation": conversation_id, "user_message": user_message_id,
            "assistant_message": assistant_message_id}


def chat_6_content(engine):
    """Everything seeded by seed_chat_6_data, read back for comparison."""
    with engine.connect() as conn:
        conversations = conn.execute(text(
            "SELECT id, owner_id, title FROM conversations ORDER BY id")).all()
        messages = conn.execute(text(
            "SELECT id, conversation_id, seq, role, content FROM messages ORDER BY id")).all()
    return [tuple(r) for r in conversations], [tuple(r) for r in messages]


INSERT_CITATION_SQL = text(
    "INSERT INTO message_citations (message_id, ordinal, marker, chunk_id, document_id, "
    "chunk_index, start_offset, end_offset, page_start, page_end, span_sha256, "
    "source_version, source_sha256) "
    "VALUES (:m, 1, 1, :c, :d, 0, 0, 2, NULL, NULL, :h, 1, :h)"
).bindparams(h=hashlib.sha256(b"c1").hexdigest())


def citation_objects_left(engine):
    """Constraints or indexes named for message_citations that survived a downgrade."""
    with engine.connect() as conn:
        constraints = conn.execute(text(
            "SELECT conname FROM pg_constraint WHERE conname LIKE '%message\\_citations%'"
        )).scalars().all()
        indexes = conn.execute(text(
            "SELECT indexname FROM pg_indexes WHERE indexname LIKE '%message\\_citations%'"
        )).scalars().all()
    return sorted([*constraints, *indexes])


def seed_chat_7_data(engine, user_id, chat):
    """Two requests and three attempts (tables from migration #7), linked to chat 6 data."""
    sha = "a" * 64
    with engine.begin() as conn:
        pending = conn.execute(text(
            "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status, "
            "creates_conversation, conversation_id) "
            "VALUES (:o, 'pending-key', :h, 'in_progress', false, :c) RETURNING id"
        ), {"o": user_id, "h": sha, "c": chat["conversation"]}).scalar_one()
        done = conn.execute(text(
            "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status, "
            "creates_conversation, conversation_id, user_message_id, assistant_message_id, "
            "completed_at) "
            "VALUES (:o, 'done-key', :h, 'succeeded', false, :c, :u, :a, now()) RETURNING id"
        ), {"o": user_id, "h": sha, "c": chat["conversation"], "u": chat["user_message"],
            "a": chat["assistant_message"]}).scalar_one()
        # running attempt of the pending request
        conn.execute(text(
            "INSERT INTO chat_request_attempts "
            "(request_id, attempt_number, status, lease_expires_at, reserved_tokens) "
            "VALUES (:r, 1, 'running', now() + interval '1 hour', 100)"), {"r": pending})
        # the done request: an uncertain attempt, then the succeeded one
        conn.execute(text(
            "INSERT INTO chat_request_attempts "
            "(request_id, attempt_number, status, reserved_tokens, charged_tokens, "
            "llm_started_at, finished_at, error_code) "
            "VALUES (:r, 1, 'uncertain', 100, 100, now(), now(), 'timeout')"), {"r": done})
        conn.execute(text(
            "INSERT INTO chat_request_attempts "
            "(request_id, attempt_number, status, reserved_tokens, prompt_tokens, "
            "completion_tokens, charged_tokens, llm_started_at, finished_at) "
            "VALUES (:r, 2, 'succeeded', 100, 10, 5, 15, now(), now())"), {"r": done})
    return {"pending": pending, "done": done}


def assert_chat_requests_seeded(engine, requests, chat):
    """The #7 rows exist exactly as seeded (so their later disappearance is a real loss)."""
    assert row_count(engine, "chat_requests") == 2
    assert row_count(engine, "chat_request_attempts") == 3
    with engine.connect() as conn:
        done = conn.execute(text(
            "SELECT status, conversation_id, user_message_id, assistant_message_id "
            "FROM chat_requests WHERE id = :id"), {"id": requests["done"]}).one()
        statuses = conn.execute(text(
            "SELECT status FROM chat_request_attempts ORDER BY request_id, attempt_number"
        )).scalars().all()
    assert tuple(done) == ("succeeded", chat["conversation"], chat["user_message"],
                           chat["assistant_message"])
    assert statuses == ["running", "uncertain", "succeeded"]


# ---------- the round trip ----------

def test_c7_round_trip_drops_only_chat_data_and_keeps_c1_data_and_pgvector(migration_engine):
    engine = migration_engine

    # 1. C1 head (no chat tables yet) with C1 data: a user, a document and a chunk.
    run_alembic("upgrade", C1_HEAD)
    assert current_revision(engine) == C1_HEAD
    assert table_names(engine).isdisjoint(ALL_CHAT_TABLES)
    installed_version = extension_version(engine)
    assert installed_version is not None
    c1 = seed_c1_data(engine)
    assert_c1_data_intact(engine, c1, installed_version)

    # 2. Migration #6: conversations and messages appear (and only those); seed chat data.
    run_alembic("upgrade", MIGRATION_6)
    assert current_revision(engine) == MIGRATION_6
    assert set(CHAT_TABLES_6) <= table_names(engine)
    assert table_names(engine).isdisjoint(CHAT_TABLES_7)
    chat = seed_chat_6_data(engine, c1["user"])
    chat_6_before = chat_6_content(engine)
    assert len(chat_6_before[0]) == 1 and len(chat_6_before[1]) == 2
    snapshot_6 = schema_snapshot(engine, CHAT_TABLES_6)
    assert_c1_data_intact(engine, c1, installed_version)

    # 3. Migration #7: the request and attempt tables appear; seed them, linked to chat data.
    run_alembic("upgrade", MIGRATION_7)
    assert current_revision(engine) == MIGRATION_7
    assert set(ALL_CHAT_TABLES) <= table_names(engine)
    requests = seed_chat_7_data(engine, c1["user"], chat)
    assert_chat_requests_seeded(engine, requests, chat)
    snapshot_7 = schema_snapshot(engine, ALL_CHAT_TABLES)
    # #7 must not have changed the #6 tables
    assert {t: snapshot_7[t] for t in CHAT_TABLES_6} == snapshot_6
    # ... and both partial unique indexes are really partial
    attempt_indexes = {name: (unique, where) for name, _, unique, where
                       in snapshot_7["chat_request_attempts"]["indexes"]}
    assert attempt_indexes["uq_chat_request_attempts_one_running"][0] is True
    assert "'running'" in attempt_indexes["uq_chat_request_attempts_one_running"][1]
    assert attempt_indexes["uq_chat_request_attempts_one_succeeded"][0] is True
    assert "'succeeded'" in attempt_indexes["uq_chat_request_attempts_one_succeeded"][1]
    assert "ix_chat_request_attempts_llm_started_at" in attempt_indexes
    assert_c1_data_intact(engine, c1, installed_version)

    # 3b. Migration #8 changes only the three attempt CHECKs: names, FKs and indexes are
    #     identical, the seeded #7 rows survive, and 'rejected' becomes a valid status.
    run_alembic("upgrade", MIGRATION_8)
    assert current_revision(engine) == MIGRATION_8
    assert_chat_requests_seeded(engine, requests, chat)
    assert schema_snapshot(engine, ALL_CHAT_TABLES) == snapshot_7
    with engine.connect() as conn:
        transaction = conn.begin()
        conn.execute(REJECTED_ATTEMPT_SQL, {"r": requests["pending"]})   # accepted at #8
        transaction.rollback()
    assert_c1_data_intact(engine, c1, installed_version)

    # 3c. Downgrade #8 refuses while a 'rejected' attempt exists and changes nothing.
    with engine.begin() as conn:
        conn.execute(REJECTED_ATTEMPT_SQL, {"r": requests["pending"]})
    code, output = run_alembic_expecting_failure("downgrade", MIGRATION_7)
    assert code != 0
    assert "cannot downgrade" in output and "'rejected'" in output
    assert current_revision(engine) == MIGRATION_8
    assert row_count(engine, "chat_request_attempts") == 4
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM chat_request_attempts WHERE status = 'rejected'"))

    # 3d. Without rejected rows, downgrade #8 restores #7 exactly: seeded data intact and
    #     'rejected' is invalid again; then upgrade to #8 once more.
    run_alembic("downgrade", MIGRATION_7)
    assert current_revision(engine) == MIGRATION_7
    assert_chat_requests_seeded(engine, requests, chat)
    assert schema_snapshot(engine, ALL_CHAT_TABLES) == snapshot_7
    with engine.connect() as conn:
        transaction = conn.begin()
        with pytest.raises(IntegrityError) as old_status:
            conn.execute(REJECTED_ATTEMPT_SQL, {"r": requests["pending"]})
        transaction.rollback()
    # 'rejected' breaks several #7 CHECKs at once; PostgreSQL names whichever it checks first.
    assert old_status.value.orig.diag.constraint_name in {
        "ck_chat_request_attempts_status_valid",
        "ck_chat_request_attempts_charge_consistent",
    }
    run_alembic("upgrade", MIGRATION_8)
    assert current_revision(engine) == MIGRATION_8
    assert_chat_requests_seeded(engine, requests, chat)

    # 3e. Migration #9 only adds message_citations: every earlier table is identical and its
    #     data survives. The downgrade refuses while a citation exists (evidence is never
    #     destroyed silently), changes nothing, and drops the table once it is empty.
    run_alembic("upgrade", MIGRATION_9)
    assert current_revision(engine) == MIGRATION_9
    assert CITATION_TABLE in table_names(engine)
    assert schema_snapshot(engine, ALL_CHAT_TABLES) == snapshot_7
    assert {t: snapshot_7[t] for t in CHAT_TABLES_6} == snapshot_6
    assert_chat_requests_seeded(engine, requests, chat)
    assert_c1_data_intact(engine, c1, installed_version)
    with engine.begin() as conn:
        conn.execute(INSERT_CITATION_SQL, {"m": chat["assistant_message"], "c": c1["chunk"],
                                           "d": c1["document"]})
    code, output = run_alembic_expecting_failure("downgrade", MIGRATION_8)
    assert code != 0
    assert "Refusing to downgrade" in output and CITATION_TABLE in output
    assert current_revision(engine) == MIGRATION_9
    assert row_count(engine, CITATION_TABLE) == 1
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM message_citations"))
    run_alembic("downgrade", MIGRATION_8)
    assert current_revision(engine) == MIGRATION_8
    assert CITATION_TABLE not in table_names(engine)
    assert citation_objects_left(engine) == []
    assert schema_snapshot(engine, ALL_CHAT_TABLES) == snapshot_7
    assert_chat_requests_seeded(engine, requests, chat)
    run_alembic("upgrade", MIGRATION_9)
    assert current_revision(engine) == MIGRATION_9
    assert row_count(engine, CITATION_TABLE) == 0

    # 4. Downgrade #9, #8 and #7: request and attempt data is LOST (by design); chat 6 data survives.
    run_alembic("downgrade", MIGRATION_6)
    assert current_revision(engine) == MIGRATION_6
    assert CITATION_TABLE not in table_names(engine)
    assert citation_objects_left(engine) == []
    assert table_names(engine).isdisjoint(CHAT_TABLES_7)          # tables, rows and ledger gone
    assert chat_request_objects_left(engine) == []                # no constraint/index left over
    assert set(CHAT_TABLES_6) <= table_names(engine)
    assert chat_6_content(engine) == chat_6_before                # conversation and messages intact
    assert schema_snapshot(engine, CHAT_TABLES_6) == snapshot_6
    assert_c1_data_intact(engine, c1, installed_version)

    # 5. Downgrade #6: conversations and messages are LOST; C1 data and pgvector survive.
    run_alembic("downgrade", C1_HEAD)
    assert current_revision(engine) == C1_HEAD
    assert table_names(engine).isdisjoint(ALL_CHAT_TABLES)
    assert chat_request_objects_left(engine) == []
    with engine.connect() as conn:
        left = conn.execute(text(
            "SELECT conname FROM pg_constraint WHERE conname LIKE '%conversations%' "
            "OR conname LIKE '%\\_messages\\_%' OR conname LIKE 'ck\\_messages%' "
            "OR conname LIKE 'uq\\_messages%'")).scalars().all()
    assert left == []
    assert_c1_data_intact(engine, c1, installed_version)

    # 6. Re-upgrade to head: all four tables return EMPTY, identical in structure.
    run_alembic("upgrade", "head")
    assert current_revision(engine) == MIGRATION_9
    assert set(ALL_CHAT_TABLES) <= table_names(engine)
    assert CITATION_TABLE in table_names(engine)
    for table in [*ALL_CHAT_TABLES, CITATION_TABLE]:
        assert row_count(engine, table) == 0, table               # nothing came back
    assert schema_snapshot(engine, ALL_CHAT_TABLES) == snapshot_7  # names, FKs, indexes unchanged
    assert_c1_data_intact(engine, c1, installed_version)

    # 7. The re-created constraints still bite: a terminal attempt with a NULL charge fails
    #    (T-S1), while the same row with its correct charge is accepted (control). Both run
    #    in transactions that are rolled back, so the tables stay empty.
    insert_request = text(
        "INSERT INTO chat_requests (owner_id, idempotency_key, request_sha256, status, "
        "creates_conversation) VALUES (:o, 'k', :h, 'in_progress', false) RETURNING id")
    insert_attempt = text(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status, reserved_tokens, "
        "charged_tokens, llm_started_at, finished_at, error_code) "
        "VALUES (:r, 1, 'uncertain', 100, :charge, now(), now(), 'timeout')")
    params = {"o": c1["user"], "h": "a" * 64}

    with engine.connect() as conn:
        transaction = conn.begin()
        request_id = conn.execute(insert_request, params).scalar_one()
        conn.execute(insert_attempt, {"r": request_id, "charge": 100})   # control: accepted
        transaction.rollback()

    with engine.connect() as conn:
        transaction = conn.begin()
        request_id = conn.execute(insert_request, params).scalar_one()
        with pytest.raises(IntegrityError) as excinfo:
            conn.execute(insert_attempt, {"r": request_id, "charge": None})
        transaction.rollback()
    assert excinfo.value.orig.diag.constraint_name in {
        "ck_chat_request_attempts_charge_consistent",
        "ck_chat_request_attempts_charged_tokens_null_iff_running",
    }

    # The NULL charge above is also caught by the separate charged_tokens_null_iff_running
    # check, so it cannot show that charge_consistent itself was re-created. A wrong,
    # non-NULL charge (50 where the frozen rule requires the reserved 100) is rejected by
    # charge_consistent alone.
    with engine.connect() as conn:
        transaction = conn.begin()
        request_id = conn.execute(insert_request, params).scalar_one()
        with pytest.raises(IntegrityError) as wrong_charge:
            conn.execute(insert_attempt, {"r": request_id, "charge": 50})
        transaction.rollback()
    assert wrong_charge.value.orig.diag.constraint_name == "ck_chat_request_attempts_charge_consistent"

    # The re-created #8 constraints accept a rejected zero-charge attempt (control) and
    # reject a rejected attempt charged the reservation (charge_consistent alone).
    insert_rejected = text(
        "INSERT INTO chat_request_attempts (request_id, attempt_number, status, reserved_tokens, "
        "charged_tokens, llm_started_at, finished_at, error_code) "
        "VALUES (:r, 1, 'rejected', 100, :charge, now(), now(), 'provider_rejected')")
    with engine.connect() as conn:
        transaction = conn.begin()
        request_id = conn.execute(insert_request, params).scalar_one()
        conn.execute(insert_rejected, {"r": request_id, "charge": 0})
        transaction.rollback()
    with engine.connect() as conn:
        transaction = conn.begin()
        request_id = conn.execute(insert_request, params).scalar_one()
        with pytest.raises(IntegrityError) as rejected_charge:
            conn.execute(insert_rejected, {"r": request_id, "charge": 100})
        transaction.rollback()
    assert rejected_charge.value.orig.diag.constraint_name == (
        "ck_chat_request_attempts_charge_consistent")

    for table in ALL_CHAT_TABLES:
        assert row_count(engine, table) == 0, table
    assert_c1_data_intact(engine, c1, installed_version)
