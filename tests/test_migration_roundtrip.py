"""C1 migration round trip: upgrade -> downgrade -> re-upgrade, with existing data.

Required C1.6 verification. Runs ONLY against the dedicated database named
documind_migration_test, given by MIGRATION_TEST_DATABASE_URL; it is skipped
in normal runs where that variable is unset. It never touches documind_test
or documind_db.

Safety: before the schema reset or any Alembic command, the fixture proves that
(1) the URL names exactly documind_migration_test, (2) migrations/env.py still
takes its URL from app.core.config.settings.database_url, (3) that settings path,
resolved in a subprocess with the exact environment and working directory the
Alembic subprocesses will use, yields the same database, host and port, and
(4) the server reports current_database() == documind_migration_test.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_ENV_PY = PROJECT_ROOT / "migrations" / "env.py"
MIGRATION_TEST_DATABASE_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")
MIGRATION_TEST_DATABASE_NAME = "documind_migration_test"
BASE_REVISION = "b4413de55d93"  # last revision before C1

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

C1_DOCUMENT_COLUMNS = {
    "source_version", "source_sha256", "indexing_status", "indexing_error",
    "indexing_attempt", "indexing_lease_expires_at", "indexed_source_version",
    "indexed_signature", "indexed_embedding_id", "indexed_at",
}
PRE_C1_DOCUMENT_CHECKS = {
    "ck_documents_filename_not_blank",
    "ck_documents_processing_status_valid",
}
C1_DOCUMENT_CHECKS = {
    "ck_documents_indexing_status_valid",
    "ck_documents_source_version_positive",
    "ck_documents_indexing_attempt_nonnegative",
    "ck_documents_indexed_source_version_valid",
    "ck_documents_source_sha256_format",
    "ck_documents_indexing_lease_consistent",
    "ck_documents_indexed_metadata_complete",
}

pytestmark = pytest.mark.skipif(
    not MIGRATION_TEST_DATABASE_URL,
    reason=(
        "Required C1.6 check: set MIGRATION_TEST_DATABASE_URL to the "
        f"'{MIGRATION_TEST_DATABASE_NAME}' database to run."
    ),
)

_safety_checks_passed = False


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


def extension_version(engine):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()


def document_columns(engine):
    return {c["name"] for c in inspect(engine).get_columns("documents")}


def document_checks(engine):
    return {c["name"] for c in inspect(engine).get_check_constraints("documents")}


def table_names(engine):
    return set(inspect(engine).get_table_names())


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


def test_c1_round_trip_preserves_data_and_never_drops_pgvector(migration_engine):
    engine = migration_engine

    # 1. Pre-C1 schema with an existing ("legacy") document.
    run_alembic("upgrade", BASE_REVISION)
    assert "document_chunks" not in table_names(engine)
    with engine.begin() as conn:
        user_id = conn.execute(text(
            "INSERT INTO users (email, hashed_password) "
            "VALUES ('legacy@example.com', 'x') RETURNING id"
        )).scalar_one()
        doc_id = conn.execute(text(
            "INSERT INTO documents (filename, owner_id, processing_status, extracted_text) "
            "VALUES ('legacy.txt', :owner, 'completed', 'legacy text') RETURNING id"
        ), {"owner": user_id}).scalar_one()

    # 2. Upgrade: new schema present; the existing row receives the server defaults.
    run_alembic("upgrade", "head")
    installed_version = extension_version(engine)
    assert installed_version is not None
    assert "document_chunks" in table_names(engine)
    assert C1_DOCUMENT_COLUMNS <= document_columns(engine)
    assert document_checks(engine) == PRE_C1_DOCUMENT_CHECKS | C1_DOCUMENT_CHECKS
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT source_version, indexing_attempt, indexing_status, source_sha256, "
            "indexing_lease_expires_at, indexed_source_version, indexed_at "
            "FROM documents WHERE id = :id"
        ), {"id": doc_id}).one()
    assert tuple(row) == (1, 0, None, None, None, None, None)

    # 3. A chunk exists, so downgrade must drop a non-empty table.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO document_chunks "
            "(document_id, chunk_index, start_offset, end_offset, text, span_sha256, embedding) "
            "VALUES (:doc, 0, 0, 6, 'legacy', :hash, CAST(:embedding AS vector))"
        ), {
            "doc": doc_id,
            "hash": hashlib.sha256(b"legacy").hexdigest(),
            "embedding": "[" + ",".join(["0.5"] * 384) + "]",
        })

    # 4. Downgrade: C1 objects gone, original data intact, extension kept and usable.
    run_alembic("downgrade", BASE_REVISION)
    assert "document_chunks" not in table_names(engine)
    assert document_columns(engine).isdisjoint(C1_DOCUMENT_COLUMNS)
    assert document_checks(engine) == PRE_C1_DOCUMENT_CHECKS
    with engine.connect() as conn:
        legacy = conn.execute(text(
            "SELECT filename, processing_status, extracted_text FROM documents WHERE id = :id"
        ), {"id": doc_id}).one()
        revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        distance = conn.execute(text("SELECT '[1,0]'::vector <=> '[0,1]'::vector")).scalar_one()
    assert tuple(legacy) == ("legacy.txt", "completed", "legacy text")
    assert revision == BASE_REVISION
    assert extension_version(engine) == installed_version
    assert distance == pytest.approx(1.0)

    # 5. Re-upgrade with the extension already present.
    run_alembic("upgrade", "head")
    assert "document_chunks" in table_names(engine)
    assert C1_DOCUMENT_COLUMNS <= document_columns(engine)
    assert document_checks(engine) == PRE_C1_DOCUMENT_CHECKS | C1_DOCUMENT_CHECKS
    assert extension_version(engine) == installed_version
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM document_chunks")).scalar_one() == 0
        row = conn.execute(text(
            "SELECT source_version, indexing_attempt FROM documents WHERE id = :id"
        ), {"id": doc_id}).one()
    assert tuple(row) == (1, 0)