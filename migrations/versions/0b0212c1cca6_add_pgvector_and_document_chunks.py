"""add pgvector and document chunks

Revision ID: 0b0212c1cca6
Revises: b4413de55d93
Create Date: 2026-11-04 11:02:17.866694

Component 12 / C1. Hand-written: do NOT regenerate with --autogenerate.
- Enables the pgvector extension (shared infrastructure; never dropped on downgrade).
- Adds indexing metadata columns and CHECK constraints to documents.
- Creates document_chunks with vector(384) embeddings (exact search only; no ANN index).
Names and CHECK expressions must stay identical to app/models/document.py and
app/models/chunk.py.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '0b0212c1cca6'
down_revision: Union[str, Sequence[str], None] = 'b4413de55d93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Frozen at this revision on purpose (do not import app constants into migrations).
EMBEDDING_DIM = 384

DOCUMENTS_CHECKS = [
    (
        "ck_documents_indexing_status_valid",
        "indexing_status IS NULL OR indexing_status IN ('pending', 'indexing', 'indexed', 'failed')",
    ),
    (
        "ck_documents_source_version_positive",
        "source_version >= 1",
    ),
    (
        "ck_documents_indexing_attempt_nonnegative",
        "indexing_attempt >= 0",
    ),
    (
        "ck_documents_indexed_source_version_valid",
        "indexed_source_version IS NULL OR "
        "(indexed_source_version >= 1 AND indexed_source_version <= source_version)",
    ),
    (
        "ck_documents_source_sha256_format",
        "source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$'",
    ),
    (
        "ck_documents_indexing_lease_consistent",
        "("
        "indexing_status IS NOT DISTINCT FROM 'indexing' "
        "AND indexing_lease_expires_at IS NOT NULL"
        ") OR ("
        "indexing_status IS DISTINCT FROM 'indexing' "
        "AND indexing_lease_expires_at IS NULL"
        ")",
    ),
    (
        "ck_documents_indexed_metadata_complete",
        "indexing_status IS DISTINCT FROM 'indexed' OR ("
        "indexed_source_version IS NOT NULL AND source_sha256 IS NOT NULL AND "
        "indexed_signature IS NOT NULL AND indexed_embedding_id IS NOT NULL AND "
        "indexed_at IS NOT NULL)",
    ),
]


# Added in this order on upgrade, dropped in reverse order on downgrade.
DOCUMENTS_NEW_COLUMNS = [
    sa.Column("source_version", sa.Integer(), server_default=sa.text("1"), nullable=False),
    sa.Column("source_sha256", sa.String(length=64), nullable=True),
    sa.Column("indexing_status", sa.String(), nullable=True),
    sa.Column("indexing_error", sa.String(), nullable=True),
    sa.Column("indexing_attempt", sa.Integer(), server_default=sa.text("0"), nullable=False),
    sa.Column("indexing_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("indexed_source_version", sa.Integer(), nullable=True),
    sa.Column("indexed_signature", sa.String(), nullable=True),
    sa.Column("indexed_embedding_id", sa.String(), nullable=True),
    sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
]


def upgrade() -> None:
    """Upgrade schema."""
    # 1. pgvector extension. IF NOT EXISTS: safe whether or not it is already installed.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Indexing metadata on documents. Existing rows receive source_version=1 and
    #    indexing_attempt=0 from the server defaults; every other new column is NULL.
    for column in DOCUMENTS_NEW_COLUMNS:
        op.add_column("documents", column)

    # 3. CHECK constraints on documents (validated against existing rows when added).
    for name, condition in DOCUMENTS_CHECKS:
        op.create_check_constraint(name, "documents", condition)

    # 4. Chunk table. The unique (document_id, chunk_index) index also serves
    #    document_id lookups and cascade deletes; no separate index, no HNSW/IVFFlat.
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("span_sha256", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_id_chunk_index"
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_document_chunks_chunk_index_nonnegative"),
        sa.CheckConstraint(
            "start_offset >= 0 AND start_offset < end_offset",
            name="ck_document_chunks_offsets_valid",
        ),
        sa.CheckConstraint(
            "length(text) = end_offset - start_offset",
            name="ck_document_chunks_text_length_matches_offsets",
        ),
        sa.CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start >= 1 AND page_start <= page_end)",
            name="ck_document_chunks_pages_valid",
        ),
        sa.CheckConstraint(
            "span_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_chunks_span_sha256_format",
        ),
    )


def downgrade() -> None:
    """Downgrade schema.

    Removes everything this revision added, in reverse dependency order.
    The pgvector extension is intentionally NOT dropped: it is shared database
    infrastructure, and re-upgrading relies on CREATE EXTENSION IF NOT EXISTS.
    Chunks and indexing metadata are derived data and can be rebuilt by re-indexing.
    """
    # 1. Chunk table first: it references documents (FK) and uses the vector type.
    #    Dropping it also drops its PK, FK, unique index and CHECK constraints.
    op.drop_table("document_chunks")

    # 2. documents CHECK constraints, before the columns they reference.
    for name, _ in reversed(DOCUMENTS_CHECKS):
        op.drop_constraint(name, "documents", type_="check")

    # 3. documents columns, reverse order of addition.
    for column in reversed(DOCUMENTS_NEW_COLUMNS):
        op.drop_column("documents", column.name)

    # 4. Deliberately no: op.execute("DROP EXTENSION vector")