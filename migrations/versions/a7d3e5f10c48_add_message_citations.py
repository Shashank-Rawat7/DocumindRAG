"""add message citations

Revision ID: a7d3e5f10c48
Revises: c4f1a7d92b60
Create Date: 2026-11-17 19:00:00.000000

Component C9 / C9.4b (migration #9). Hand-written: do NOT regenerate with --autogenerate.
Creates message_citations: the sources an assistant message cites, written only by
publish_success in the same transaction as the message. Nothing else changes: no existing
table, column, constraint or index is touched.

- message_id -> messages.id ON DELETE CASCADE (a citation never outlives its message).
- chunk_id -> document_chunks.id ON DELETE SET NULL, and a snapshot of the chunk's identity
  and immutable metadata (document_id WITHOUT a foreign key, offsets, pages, span and
  source hashes, source_version), so a citation survives re-indexing and document deletion.
  There is no text snapshot.
- Unique (message_id, ordinal) and (message_id, marker); marker 1..999.
Names and CHECK expressions must stay identical to app/models/message_citation.py. Do not
import app constants here.

Downgrade drops the table and REFUSES to run while any citation row exists: dropping it
would silently destroy persisted citation evidence.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7d3e5f10c48"
down_revision: str | Sequence[str] | None = "c4f1a7d92b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "message_citations"

# Frozen at this revision on purpose. Tests compare these exact texts with the model's.
CITATION_CHECKS = [
    ("ck_message_citations_ordinal_positive", "ordinal >= 1"),
    ("ck_message_citations_marker_valid", "marker >= 1 AND marker <= 999"),
    ("ck_message_citations_chunk_index_nonnegative", "chunk_index >= 0"),
    (
        "ck_message_citations_offsets_valid",
        "start_offset >= 0 AND start_offset < end_offset",
    ),
    (
        "ck_message_citations_pages_valid",
        (
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start >= 1 AND page_start <= page_end)"
        ),
    ),
    ("ck_message_citations_span_sha256_format", "span_sha256 ~ '^[0-9a-f]{64}$'"),
    ("ck_message_citations_source_version_positive", "source_version >= 1"),
    ("ck_message_citations_source_sha256_format", "source_sha256 ~ '^[0-9a-f]{64}$'"),
]


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("marker", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("span_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_version", sa.Integer(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name="fk_message_citations_message_id_messages",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_message_citations_chunk_id_document_chunks",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "message_id", "ordinal", name="uq_message_citations_message_id_ordinal"
        ),
        sa.UniqueConstraint(
            "message_id", "marker", name="uq_message_citations_message_id_marker"
        ),
        *[
            sa.CheckConstraint(condition, name=name)
            for name, condition in CITATION_CHECKS
        ],
    )
    op.create_index("ix_message_citations_chunk_id", TABLE, ["chunk_id"])


def downgrade() -> None:
    """Downgrade schema. Refuses while any citation exists (see module docstring)."""
    existing = (
        op.get_bind().exec_driver_sql(f"SELECT count(*) FROM {TABLE}").scalar_one()
    )
    if existing:
        raise RuntimeError(
            f"Refusing to downgrade: {TABLE} holds {existing} citation row(s). "
            "Dropping the table would destroy persisted citation evidence."
        )
    op.drop_table(TABLE)
