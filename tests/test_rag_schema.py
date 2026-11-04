"""C1 schema tests: pgvector, documents indexing metadata, document_chunks.

Runs against documind_test, which conftest builds with `alembic upgrade head`,
so these assertions check the migrated database, not just ORM metadata.
"""
import hashlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.core.database import engine
from app.models.chunk import EMBEDDING_DIM, DocumentChunk
from app.models.document import Document
from app.models.user import User

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = "/api/v1/documents"
ME = "/api/v1/auth/me"

NOW = datetime.now(timezone.utc)
CHUNK_TEXT = "FastAPI dependency injection"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def vec(*pairs):
    """A 384-dim vector that is zero except for the given (index, value) pairs."""
    values = [0.0] * EMBEDDING_DIM
    for index, value in pairs:
        values[index] = value
    return values


VALID_SHA = sha("source text")
INDEXED = dict(
    indexing_status="indexed",
    indexed_source_version=1,
    source_sha256=VALID_SHA,
    indexed_signature="test-signature",
    indexed_embedding_id="intfloat/multilingual-e5-small@384",
    indexed_at=NOW,
)

NEW_DOCUMENT_COLUMNS = {
    "source_version", "source_sha256", "indexing_status", "indexing_error",
    "indexing_attempt", "indexing_lease_expires_at", "indexed_source_version",
    "indexed_signature", "indexed_embedding_id", "indexed_at",
}
EXPECTED_DOCUMENT_CHECKS = {
    "ck_documents_filename_not_blank",
    "ck_documents_processing_status_valid",
    "ck_documents_indexing_status_valid",
    "ck_documents_source_version_positive",
    "ck_documents_indexing_attempt_nonnegative",
    "ck_documents_indexed_source_version_valid",
    "ck_documents_source_sha256_format",
    "ck_documents_indexing_lease_consistent",
    "ck_documents_indexed_metadata_complete",
}
CHUNK_COLUMNS = {
    "id", "document_id", "chunk_index", "start_offset", "end_offset",
    "page_start", "page_end", "text", "span_sha256", "embedding", "created_at",
}
EXPECTED_CHUNK_CHECKS = {
    "ck_document_chunks_chunk_index_nonnegative",
    "ck_document_chunks_offsets_valid",
    "ck_document_chunks_text_length_matches_offsets",
    "ck_document_chunks_pages_valid",
    "ck_document_chunks_span_sha256_format",
}


# ---------- helpers ----------

def model_check_names(model):
    return {c.name for c in model.__table__.constraints if isinstance(c, CheckConstraint)}


def make_chunk(document_id, chunk_index=0, content=CHUNK_TEXT, start_offset=0, **overrides):
    """A chunk that satisfies every constraint unless overridden."""
    fields = dict(
        document_id=document_id,
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
        text=content,
        span_sha256=sha(content),
        embedding=vec((0, 1.0)),
    )
    fields.update(overrides)
    return DocumentChunk(**fields)


def assert_violates(db, instance, constraint_name):
    """Insert must fail, and fail on exactly the named constraint."""
    db.add(instance)
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.diag.constraint_name == constraint_name


def add_chunks(db, document_id, count):
    for index in range(count):
        db.add(make_chunk(document_id, chunk_index=index))
    db.commit()


def chunk_count(db, document_id=None):
    query = db.query(DocumentChunk)
    if document_id is not None:
        query = query.filter(DocumentChunk.document_id == document_id)
    return query.count()


def create_document(client, headers, filename):
    response = client.post(DOCUMENTS, json={"filename": filename}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def owner_id(db):
    user = User(email="owner@example.com", hashed_password="not-a-real-hash")
    db.add(user)
    db.commit()
    return user.id


@pytest.fixture
def document_id(db, owner_id):
    document = Document(filename="doc.txt", owner_id=owner_id)
    db.add(document)
    db.commit()
    return document.id


# ---------- pgvector and migration state ----------

def test_pgvector_extension_is_installed_at_the_pinned_version():
    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()

    assert version == "0.8.7"


def test_database_is_at_the_single_migration_head():
    script = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))
    heads = script.get_heads()

    assert len(heads) == 1
    with engine.connect() as conn:
        current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    assert current == heads[0]


# ---------- documents: schema ----------

def test_documents_has_the_new_indexing_columns():
    columns = {c["name"]: c for c in inspect(engine).get_columns("documents")}

    assert NEW_DOCUMENT_COLUMNS <= set(columns)
    assert columns["source_version"]["nullable"] is False
    assert columns["source_version"]["default"] == "1"
    assert columns["indexing_attempt"]["nullable"] is False
    assert columns["indexing_attempt"]["default"] == "0"
    for name in NEW_DOCUMENT_COLUMNS - {"source_version", "indexing_attempt"}:
        assert columns[name]["nullable"] is True, name
    assert columns["source_sha256"]["type"].length == 64
    for name in ("indexing_lease_expires_at", "indexed_at"):
        assert columns[name]["type"].timezone is True, name


def test_documents_check_constraints_match_the_model():
    in_database = {c["name"] for c in inspect(engine).get_check_constraints("documents")}

    assert in_database == model_check_names(Document) == EXPECTED_DOCUMENT_CHECKS


def test_a_new_document_gets_the_version_and_attempt_defaults(db, owner_id):
    document = Document(filename="new.txt", owner_id=owner_id)
    db.add(document)
    db.commit()
    db.refresh(document)

    assert document.source_version == 1
    assert document.indexing_attempt == 0
    assert document.indexing_status is None
    assert document.indexing_lease_expires_at is None
    assert document.source_sha256 is None
    assert document.indexed_source_version is None
    assert document.indexed_at is None
    assert document.processing_status is None  # existing behaviour unchanged


# ---------- documents: indexing constraints ----------

DOCUMENT_REJECTIONS = [
    pytest.param({"indexing_status": "done"},
                 "ck_documents_indexing_status_valid", id="unknown-status"),
    pytest.param({"source_version": 0},
                 "ck_documents_source_version_positive", id="version-zero"),
    pytest.param({"indexing_attempt": -1},
                 "ck_documents_indexing_attempt_nonnegative", id="attempt-negative"),
    pytest.param({"indexed_source_version": 2},
                 "ck_documents_indexed_source_version_valid", id="indexed-version-ahead"),
    pytest.param({"indexed_source_version": 0},
                 "ck_documents_indexed_source_version_valid", id="indexed-version-zero"),
    pytest.param({"source_sha256": VALID_SHA.upper()},
                 "ck_documents_source_sha256_format", id="hash-uppercase"),
    pytest.param({"source_sha256": "abc"},
                 "ck_documents_source_sha256_format", id="hash-short"),
    pytest.param({"indexing_status": "indexing"},
                 "ck_documents_indexing_lease_consistent", id="indexing-without-lease"),
    pytest.param({"indexing_lease_expires_at": NOW},
                 "ck_documents_indexing_lease_consistent", id="lease-with-null-status"),
    pytest.param({"indexing_status": "failed", "indexing_lease_expires_at": NOW},
                 "ck_documents_indexing_lease_consistent", id="lease-on-failed"),
    pytest.param({**INDEXED, "indexed_at": None},
                 "ck_documents_indexed_metadata_complete", id="indexed-without-indexed-at"),
    pytest.param({**INDEXED, "source_sha256": None},
                 "ck_documents_indexed_metadata_complete", id="indexed-without-hash"),
]


@pytest.mark.parametrize("fields, constraint", DOCUMENT_REJECTIONS)
def test_invalid_document_indexing_state_is_rejected(db, owner_id, fields, constraint):
    assert_violates(db, Document(filename="bad.txt", owner_id=owner_id, **fields), constraint)


DOCUMENT_ACCEPTED = [
    pytest.param({"indexing_status": "pending"}, id="pending"),
    pytest.param({"indexing_status": "indexing", "indexing_attempt": 1,
                  "indexing_lease_expires_at": NOW + timedelta(minutes=15)},
                 id="indexing-with-lease"),
    pytest.param({"indexing_status": "failed", "indexing_attempt": 1,
                  "indexing_error": "Embedding failed"}, id="initial-index-failed"),
    pytest.param(INDEXED, id="indexed"),
    pytest.param({**INDEXED, "indexing_attempt": 2, "indexing_error": "Re-index failed"},
                 id="indexed-restored-after-failed-reindex"),
]


@pytest.mark.parametrize("fields", DOCUMENT_ACCEPTED)
def test_valid_document_indexing_state_is_accepted(db, owner_id, fields):
    db.add(Document(filename="ok.txt", owner_id=owner_id, **fields))
    db.commit()


# ---------- document_chunks: schema ----------

def test_document_chunks_has_the_expected_columns_and_no_owner():
    columns = {c["name"]: c for c in inspect(engine).get_columns("document_chunks")}

    assert set(columns) == CHUNK_COLUMNS  # in particular: no owner_id
    assert {name for name, c in columns.items() if c["nullable"]} == {"page_start", "page_end"}


def test_embedding_column_is_vector_384():
    with engine.connect() as conn:
        column_type = conn.execute(text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'"
        )).scalar_one()

    assert column_type == f"vector({EMBEDDING_DIM})"


def test_document_chunks_constraints_match_the_model():
    insp = inspect(engine)

    in_database = {c["name"] for c in insp.get_check_constraints("document_chunks")}
    assert in_database == model_check_names(DocumentChunk) == EXPECTED_CHUNK_CHECKS

    [unique] = insp.get_unique_constraints("document_chunks")
    assert unique["name"] == "uq_document_chunks_document_id_chunk_index"
    assert unique["column_names"] == ["document_id", "chunk_index"]

    [fk] = insp.get_foreign_keys("document_chunks")
    assert fk["name"] == "fk_document_chunks_document_id_documents"
    assert fk["constrained_columns"] == ["document_id"]
    assert fk["referred_table"] == "documents"
    assert fk["options"].get("ondelete") == "CASCADE"


def test_document_chunks_has_no_approximate_vector_index():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks'"
        )).all()

    assert {name for name, _ in rows} == {
        "document_chunks_pkey",
        "uq_document_chunks_document_id_chunk_index",
    }
    assert not any(method in definition.lower()
                   for _, definition in rows for method in ("hnsw", "ivfflat"))


# ---------- document_chunks: constraints ----------

CHUNK_REJECTIONS = [
    pytest.param({"chunk_index": -1},
                 "ck_document_chunks_chunk_index_nonnegative", id="negative-index"),
    pytest.param({"start_offset": -1},
                 "ck_document_chunks_offsets_valid", id="negative-start"),
    pytest.param({"content": "", "start_offset": 3},
                 "ck_document_chunks_offsets_valid", id="empty-span"),
    pytest.param({"end_offset": len(CHUNK_TEXT) + 5},
                 "ck_document_chunks_text_length_matches_offsets", id="length-mismatch"),
    pytest.param({"page_start": 1},
                 "ck_document_chunks_pages_valid", id="page-end-missing"),
    pytest.param({"page_end": 1},
                 "ck_document_chunks_pages_valid", id="page-start-missing"),
    pytest.param({"page_start": 0, "page_end": 1},
                 "ck_document_chunks_pages_valid", id="page-zero"),
    pytest.param({"page_start": 3, "page_end": 2},
                 "ck_document_chunks_pages_valid", id="pages-reversed"),
    pytest.param({"span_sha256": sha(CHUNK_TEXT).upper()},
                 "ck_document_chunks_span_sha256_format", id="hash-uppercase"),
    pytest.param({"span_sha256": "0" * 63},
                 "ck_document_chunks_span_sha256_format", id="hash-short"),
]


@pytest.mark.parametrize("fields, constraint", CHUNK_REJECTIONS)
def test_invalid_chunk_is_rejected(db, document_id, fields, constraint):
    assert_violates(db, make_chunk(document_id, **fields), constraint)


def test_unicode_chunk_offsets_slice_the_extracted_text_exactly(db, owner_id):
    # Emoji (1 code point, 4 UTF-8 bytes, 2 UTF-16 units) before the chunk and
    # Devanagari combining marks inside it: byte- or UTF-16-based offsets would
    # land in the wrong place.
    prefix = "Intro 🙂 पहले "
    hindi = "नमस्ते दुनिया"
    source = f"{prefix}🙂{hindi} — अंत 🙂 end"
    document = Document(filename="unicode.txt", owner_id=owner_id, extracted_text=source)
    db.add(document)
    db.commit()
    doc_id = document.id

    start_offset = source.index(hindi)  # Python str (code point) index
    end_offset = start_offset + len(hindi)
    chunk = make_chunk(doc_id, content=hindi, start_offset=start_offset,
                       page_start=1, page_end=2)
    db.add(chunk)
    db.commit()
    chunk_id = chunk.id

    db.expire_all()
    stored_chunk = db.get(DocumentChunk, chunk_id)
    stored_document = db.get(Document, doc_id)

    assert (stored_chunk.start_offset, stored_chunk.end_offset) == (start_offset, end_offset)
    assert stored_chunk.text == stored_document.extracted_text[
        stored_chunk.start_offset:stored_chunk.end_offset
    ]
    assert stored_chunk.span_sha256 == sha(stored_chunk.text)

    with engine.connect() as conn:
        db_length, span, db_slice_matches = conn.execute(
            text(
                "SELECT length(c.text), c.end_offset - c.start_offset, "
                "substr(d.extracted_text, c.start_offset + 1, c.end_offset - c.start_offset) = c.text "
                "FROM document_chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id = :id"
            ),
            {"id": chunk_id},
        ).one()
    assert db_length == span == len(hindi)
    assert db_slice_matches is True


def test_duplicate_chunk_index_in_one_document_is_rejected(db, document_id):
    add_chunks(db, document_id, 1)

    assert_violates(db, make_chunk(document_id, chunk_index=0),
                    "uq_document_chunks_document_id_chunk_index")


def test_same_chunk_index_in_different_documents_is_allowed(db, owner_id):
    first = Document(filename="one.txt", owner_id=owner_id)
    second = Document(filename="two.txt", owner_id=owner_id)
    db.add_all([first, second])
    db.commit()

    db.add_all([make_chunk(first.id, chunk_index=0), make_chunk(second.id, chunk_index=0)])
    db.commit()

    assert chunk_count(db) == 2


def test_chunk_for_a_missing_document_is_rejected(db):
    assert_violates(db, make_chunk(999_999), "fk_document_chunks_document_id_documents")


def test_chunk_without_an_embedding_is_rejected(db, document_id):
    db.add(make_chunk(document_id, embedding=None))
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()

    assert excinfo.value.orig.diag.column_name == "embedding"


# ---------- vectors ----------

def test_embedding_round_trips_with_384_dimensions(db, document_id):
    values = [i / 1000 for i in range(EMBEDDING_DIM)]
    chunk = make_chunk(document_id, embedding=values)
    db.add(chunk)
    db.commit()

    stored = db.execute(
        select(DocumentChunk.embedding).where(DocumentChunk.id == chunk.id)
    ).scalar_one()

    assert isinstance(stored, list)
    assert len(stored) == EMBEDDING_DIM
    assert stored == pytest.approx(values, rel=1e-6)  # stored as float4


@pytest.mark.parametrize("dimensions", [EMBEDDING_DIM - 1, EMBEDDING_DIM + 1])
def test_embedding_with_the_wrong_dimension_is_rejected(db, document_id, dimensions):
    db.add(make_chunk(document_id, embedding=[0.1] * dimensions))
    with pytest.raises(DBAPIError) as excinfo:
        db.flush()

    db.rollback()

    assert excinfo.value.orig.pgcode == "22000"  # data_exception (pgvector CheckExpectedDim)
    assert chunk_count(db, document_id) == 0


def test_cosine_distance_orders_chunks_by_similarity(db, document_id):
    # Stored out of order on purpose: chunk_index differs from the expected rank.
    embeddings = {
        0: vec((1, 1.0)),            # orthogonal to the query  -> distance 1
        1: vec((0, 1.0)),            # same direction           -> distance 0
        2: vec((0, 1.0), (1, 1.0)),  # 45 degrees               -> distance 1 - 1/sqrt(2)
    }
    for index, embedding in embeddings.items():
        db.add(make_chunk(document_id, chunk_index=index, embedding=embedding))
    db.commit()

    distance = DocumentChunk.embedding.cosine_distance(vec((0, 1.0)))
    rows = db.execute(
        select(DocumentChunk.chunk_index, distance)
        .where(DocumentChunk.document_id == document_id)
        .order_by(distance)
    ).all()

    assert [index for index, _ in rows] == [1, 2, 0]
    assert [d for _, d in rows] == pytest.approx([0.0, 1 - 1 / math.sqrt(2), 1.0], abs=1e-6)


# ---------- cascades ----------

def test_deleting_a_document_deletes_all_its_chunks(client, make_user, db):
    alice = make_user("alice@example.com")
    doomed = create_document(client, alice, "doomed.txt")
    kept = create_document(client, alice, "kept.txt")
    add_chunks(db, doomed, 3)
    add_chunks(db, kept, 2)

    assert client.delete(f"{DOCUMENTS}/{doomed}", headers=alice).status_code == 204

    assert chunk_count(db, doomed) == 0
    assert chunk_count(db, kept) == 2


def test_deleting_a_user_deletes_their_documents_and_chunks(client, make_user, db):
    alice = make_user("alice@example.com")
    alice_id = db.query(User).filter_by(email="alice@example.com").one().id
    alice_docs = [create_document(client, alice, name) for name in ("a.txt", "b.txt")]
    for doc in alice_docs:
        add_chunks(db, doc, 2)
    alice_chunk_ids = {
        c.id for c in db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(alice_docs))
    }
    assert len(alice_chunk_ids) == 4

    assert client.delete(ME, headers=alice).status_code == 204

    db.expire_all()
    assert db.get(User, alice_id) is None
    for doc in alice_docs:
        assert db.get(Document, doc) is None, doc
    assert db.query(DocumentChunk).filter(DocumentChunk.id.in_(alice_chunk_ids)).count() == 0
    assert db.query(Document).count() == 0
    assert chunk_count(db) == 0


def test_deleting_one_users_data_leaves_other_users_chunks_untouched(client, make_user, db):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    alice_docs = [create_document(client, alice, name) for name in ("a1.txt", "a2.txt")]
    bob_doc = create_document(client, bob, "bob.txt")
    for doc in alice_docs:
        add_chunks(db, doc, 2)
    add_chunks(db, bob_doc, 3)
    bob_chunk_ids = {c.id for c in db.query(DocumentChunk).filter_by(document_id=bob_doc)}

    assert client.delete(f"{DOCUMENTS}/{alice_docs[0]}", headers=alice).status_code == 204
    assert client.delete(ME, headers=alice).status_code == 204

    remaining = db.query(DocumentChunk).all()
    assert {c.id for c in remaining} == bob_chunk_ids
    assert client.get(f"{DOCUMENTS}/{bob_doc}", headers=bob).status_code == 200