"""C9.4b schema tests: message_citations (migration #9).

Runs against documind_test, which conftest builds with `alembic upgrade head`, so these
assertions check the migrated database, not just ORM metadata.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, inspect, text
from sqlalchemy.exc import IntegrityError

from app.core.database import engine
from app.models.message_citation import MessageCitation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_9_FILE = (
    PROJECT_ROOT / "migrations" / "versions" / "a7d3e5f10c48_add_message_citations.py"
)
SPAN = hashlib.sha256(b"span").hexdigest()
SOURCE = hashlib.sha256(b"source").hexdigest()
EXPECTED_COLUMNS = {
    "id",
    "message_id",
    "ordinal",
    "marker",
    "chunk_id",
    "document_id",
    "chunk_index",
    "start_offset",
    "end_offset",
    "page_start",
    "page_end",
    "span_sha256",
    "source_version",
    "source_sha256",
    "created_at",
}
NULLABLE = {"chunk_id", "page_start", "page_end"}
EXPECTED_CHECKS = {
    "ck_message_citations_ordinal_positive",
    "ck_message_citations_marker_valid",
    "ck_message_citations_chunk_index_nonnegative",
    "ck_message_citations_offsets_valid",
    "ck_message_citations_pages_valid",
    "ck_message_citations_span_sha256_format",
    "ck_message_citations_source_version_positive",
    "ck_message_citations_source_sha256_format",
}


def execute(sql, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)


def load_migration_9():
    spec = importlib.util.spec_from_file_location("migration_9", MIGRATION_9_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seeded():
    """A user, a conversation with an assistant message, a document and two chunks."""
    owner = execute(
        "INSERT INTO users (email, hashed_password) VALUES ('a@example.com', 'x')"
        " RETURNING id"
    ).scalar_one()
    conversation = execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=owner
    ).scalar_one()
    messages = [
        execute(
            "INSERT INTO messages (conversation_id, seq, role, content)"
            " VALUES (:c, :s, 'assistant', 'answer') RETURNING id",
            c=conversation,
            s=seq,
        ).scalar_one()
        for seq in (1, 2)
    ]
    document = execute(
        "INSERT INTO documents (filename, owner_id) VALUES ('d.txt', :o) RETURNING id",
        o=owner,
    ).scalar_one()
    chunks = [
        execute(
            "INSERT INTO document_chunks (document_id, chunk_index, start_offset,"
            " end_offset, text, span_sha256, embedding)"
            " VALUES (:d, :i, 0, 2, 'ab', :h,"
            " array_fill(0::real, ARRAY[384])::vector(384)) RETURNING id",
            d=document,
            i=index,
            h=SPAN,
        ).scalar_one()
        for index in (0, 1)
    ]
    return {
        "conversation": conversation,
        "messages": messages,
        "document": document,
        "chunks": chunks,
    }


def citation_values(seeded, **overrides):
    values = {
        "message_id": seeded["messages"][0],
        "ordinal": 1,
        "marker": 1,
        "chunk_id": seeded["chunks"][0],
        "document_id": seeded["document"],
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": 2,
        "page_start": None,
        "page_end": None,
        "span_sha256": SPAN,
        "source_version": 1,
        "source_sha256": SOURCE,
    }
    return values | overrides


def insert_citation(seeded, **overrides):
    values = citation_values(seeded, **overrides)
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    return execute(
        f"INSERT INTO message_citations ({columns}) VALUES ({placeholders}) RETURNING id",
        **values,
    ).scalar_one()


def citation_count():
    return execute("SELECT count(*) FROM message_citations").scalar_one()


# ---------------------------------------------------------------- structure


def test_models_are_registered_in_app_main_and_alembic_env():
    import app.main

    assert app.main.message_citation.MessageCitation is MessageCitation
    env_source = (PROJECT_ROOT / "migrations" / "env.py").read_text(encoding="utf-8")
    [import_line] = [
        ln for ln in env_source.splitlines() if ln.startswith("from app.models import")
    ]
    names = {n.strip() for n in import_line.split("#")[0].split("import")[1].split(",")}
    assert "message_citation" in names


def test_model_and_migration_check_expressions_are_identical():
    migration = load_migration_9()
    model_checks = {
        c.name: str(c.sqltext)
        for c in MessageCitation.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert model_checks == dict(migration.CITATION_CHECKS)
    assert set(model_checks) == EXPECTED_CHECKS


def test_columns_types_and_nullability():
    columns = {c["name"]: c for c in inspect(engine).get_columns("message_citations")}
    assert set(columns) == EXPECTED_COLUMNS
    for name, column in columns.items():
        assert column["nullable"] == (name in NULLABLE), name
    assert columns["span_sha256"]["type"].length == 64
    assert columns["source_sha256"]["type"].length == 64


def test_foreign_keys_unique_constraints_and_index():
    inspector = inspect(engine)
    foreign_keys = {
        fk["name"]: fk for fk in inspector.get_foreign_keys("message_citations")
    }
    assert set(foreign_keys) == {
        "fk_message_citations_message_id_messages",
        "fk_message_citations_chunk_id_document_chunks",
    }  # document_id has deliberately no foreign key
    message_fk = foreign_keys["fk_message_citations_message_id_messages"]
    assert (message_fk["referred_table"], message_fk["options"]["ondelete"]) == (
        "messages",
        "CASCADE",
    )
    chunk_fk = foreign_keys["fk_message_citations_chunk_id_document_chunks"]
    assert (chunk_fk["referred_table"], chunk_fk["options"]["ondelete"]) == (
        "document_chunks",
        "SET NULL",
    )
    uniques = {
        u["name"]: tuple(u["column_names"])
        for u in inspector.get_unique_constraints("message_citations")
    }
    assert uniques == {
        "uq_message_citations_message_id_ordinal": ("message_id", "ordinal"),
        "uq_message_citations_message_id_marker": ("message_id", "marker"),
    }
    indexes = {
        i["name"]: i["column_names"] for i in inspector.get_indexes("message_citations")
    }
    assert indexes["ix_message_citations_chunk_id"] == ["chunk_id"]


def test_the_model_declares_the_same_keys_and_index_as_the_database():
    table = MessageCitation.__table__
    model_fks = {
        fk.name: (fk.referred_table.name, fk.ondelete)
        for fk in table.foreign_key_constraints
    }
    database_fks = {
        fk["name"]: (fk["referred_table"], fk["options"]["ondelete"])
        for fk in inspect(engine).get_foreign_keys("message_citations")
    }
    assert model_fks == database_fks
    model_uniques = {
        c.name: tuple(col.name for col in c.columns)
        for c in table.constraints
        if isinstance(c, UniqueConstraint)
    }
    database_uniques = {
        u["name"]: tuple(u["column_names"])
        for u in inspect(engine).get_unique_constraints("message_citations")
    }
    assert model_uniques == database_uniques
    assert {(i.name, tuple(c.name for c in i.columns)) for i in table.indexes} == {
        ("ix_message_citations_chunk_id", ("chunk_id",))
    }
    model_columns = {
        c.name: (c.nullable, c.type.__class__.__name__) for c in table.columns
    }
    database_columns = {
        c["name"]: (c["nullable"], c["type"].__class__.__name__)
        for c in inspect(engine).get_columns("message_citations")
    }
    assert {k: v[0] for k, v in model_columns.items()} == {
        k: v[0] for k, v in database_columns.items()
    }


# ---------------------------------------------------------------- constraints


def test_a_valid_citation_is_accepted_with_and_without_pages(seeded):
    insert_citation(seeded)
    insert_citation(
        seeded,
        ordinal=2,
        marker=2,
        chunk_id=seeded["chunks"][1],
        chunk_index=1,
        page_start=2,
        page_end=3,
    )
    assert citation_count() == 2


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"ordinal": 0}, "ck_message_citations_ordinal_positive"),
        ({"marker": 0}, "ck_message_citations_marker_valid"),
        ({"marker": 1000}, "ck_message_citations_marker_valid"),
        ({"chunk_index": -1}, "ck_message_citations_chunk_index_nonnegative"),
        ({"start_offset": -1}, "ck_message_citations_offsets_valid"),
        ({"start_offset": 2, "end_offset": 2}, "ck_message_citations_offsets_valid"),
        ({"page_start": 1}, "ck_message_citations_pages_valid"),
        ({"page_end": 1}, "ck_message_citations_pages_valid"),
        ({"page_start": 0, "page_end": 1}, "ck_message_citations_pages_valid"),
        ({"page_start": 3, "page_end": 2}, "ck_message_citations_pages_valid"),
        ({"span_sha256": "A" * 64}, "ck_message_citations_span_sha256_format"),
        ({"span_sha256": "a" * 63}, "ck_message_citations_span_sha256_format"),
        ({"source_version": 0}, "ck_message_citations_source_version_positive"),
        ({"source_sha256": "g" * 64}, "ck_message_citations_source_sha256_format"),
    ],
)
def test_invalid_values_are_rejected_by_the_named_check(seeded, overrides, constraint):
    with pytest.raises(IntegrityError) as raised:
        insert_citation(seeded, **overrides)
    assert raised.value.orig.diag.constraint_name == constraint
    assert citation_count() == 0


def test_the_marker_bounds_are_inclusive(seeded):
    insert_citation(seeded, marker=1)
    insert_citation(seeded, ordinal=2, marker=999)
    assert citation_count() == 2


def test_ordinal_and_marker_are_unique_per_message_only(seeded):
    insert_citation(seeded)
    with pytest.raises(IntegrityError) as same_ordinal:
        insert_citation(seeded, marker=2)
    assert (
        same_ordinal.value.orig.diag.constraint_name
        == "uq_message_citations_message_id_ordinal"
    )
    with pytest.raises(IntegrityError) as same_marker:
        insert_citation(seeded, ordinal=2)
    assert (
        same_marker.value.orig.diag.constraint_name
        == "uq_message_citations_message_id_marker"
    )
    insert_citation(
        seeded, message_id=seeded["messages"][1]
    )  # another message: allowed
    assert citation_count() == 2


def test_a_citation_needs_an_existing_message_and_chunk(seeded):
    with pytest.raises(IntegrityError):
        insert_citation(seeded, message_id=seeded["messages"][1] + 999)
    with pytest.raises(IntegrityError):
        insert_citation(seeded, chunk_id=seeded["chunks"][1] + 999)
    assert citation_count() == 0


def test_the_snapshot_document_id_has_no_foreign_key(seeded):
    insert_citation(seeded, document_id=seeded["document"] + 12345)
    assert citation_count() == 1


# ---------------------------------------------------------------- deletes


def test_deleting_the_message_deletes_its_citations_only(seeded):
    insert_citation(seeded)
    insert_citation(seeded, message_id=seeded["messages"][1])
    execute("DELETE FROM messages WHERE id = :m", m=seeded["messages"][0])
    assert citation_count() == 1
    assert (
        execute("SELECT message_id FROM message_citations").scalar_one()
        == (seeded["messages"][1])
    )


def test_deleting_the_conversation_deletes_the_citations(seeded):
    insert_citation(seeded)
    insert_citation(seeded, message_id=seeded["messages"][1])
    execute("DELETE FROM conversations WHERE id = :c", c=seeded["conversation"])
    assert citation_count() == 0


def snapshot_of(citation_id):
    return tuple(
        execute(
            "SELECT document_id, chunk_index, start_offset, end_offset, page_start,"
            " page_end, span_sha256, source_version, source_sha256, ordinal, marker"
            " FROM message_citations WHERE id = :i",
            i=citation_id,
        ).one()
    )


def test_deleting_the_chunk_nulls_chunk_id_and_keeps_the_snapshot(seeded):
    cid = insert_citation(seeded)
    before = snapshot_of(cid)
    execute("DELETE FROM document_chunks WHERE id = :c", c=seeded["chunks"][0])
    assert execute("SELECT chunk_id FROM message_citations").scalar_one() is None
    assert snapshot_of(cid) == before


def test_deleting_the_document_nulls_chunk_id_and_keeps_the_snapshot(seeded):
    cid = insert_citation(seeded)
    before = snapshot_of(cid)
    execute("DELETE FROM documents WHERE id = :d", d=seeded["document"])
    assert execute("SELECT chunk_id FROM message_citations").scalar_one() is None
    assert snapshot_of(cid) == before
    assert before[0] == seeded["document"]  # the document id outlives the document


def test_deleting_the_owner_removes_everything(seeded):
    insert_citation(seeded)
    execute("TRUNCATE TABLE documents, users RESTART IDENTITY CASCADE")
    assert citation_count() == 0
