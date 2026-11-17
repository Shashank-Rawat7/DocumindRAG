"""add conversations and messages

Revision ID: 4a06e23062ae
Revises: 0b0212c1cca6
Create Date: 2026-11-16 19:00:00.000000

Component C7 / C7.1 (migration #6). Hand-written: do NOT regenerate with --autogenerate.
- Creates conversations (owned by users, ON DELETE CASCADE).
- Creates messages (owned by conversations, ON DELETE CASCADE).
Names and CHECK expressions must stay identical to app/models/conversation.py and
app/models/message.py. The chat request/attempt tables belong to migration #7.

WARNING: downgrade drops both tables and every chat conversation and message in them.
Chat history is user-generated data and cannot be rebuilt; take a backup first.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a06e23062ae'
down_revision: Union[str, Sequence[str], None] = '0b0212c1cca6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # conversations first: messages references it.
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_conversations_owner_id_users",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "title IS NULL OR length(trim(title)) > 0",
            name="ck_conversations_title_not_blank",
        ),
    )
    op.create_index(
        "ix_conversations_owner_id_updated_at",
        "conversations",
        ["owner_id", "updated_at"],
    )

    # The unique (conversation_id, seq) constraint also serves conversation_id
    # lookups and cascade deletes; no separate index.
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("conversation_id", "seq", name="uq_messages_conversation_id_seq"),
        sa.CheckConstraint("seq >= 1", name="ck_messages_seq_positive"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role_valid"),
        sa.CheckConstraint(
            "length(trim(content)) > 0", name="ck_messages_content_not_blank"
        ),
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops both tables (and with them their constraints and index), child first.
    All conversations and messages are lost; see the module docstring.
    """
    op.drop_table("messages")
    op.drop_index("ix_conversations_owner_id_updated_at", table_name="conversations")
    op.drop_table("conversations")
