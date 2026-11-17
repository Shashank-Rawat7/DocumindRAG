"""add chat requests and attempts

Revision ID: e9749572298d
Revises: 4a06e23062ae
Create Date: 2026-11-16 21:00:00.000000

Component C7 / C7.2 (migration #7). Hand-written: do NOT regenerate with --autogenerate.
- Creates chat_requests: one logical chat request per (owner, idempotency key).
- Creates chat_request_attempts: one execution attempt per row, and the quota ledger.
Names and CHECK expressions must stay identical to app/models/chat_request.py and
app/models/chat_request_attempt.py. Schema only: lease, fencing, reservation and quota
behaviour belong to C8.

Every attempt CHECK is written to be NULL-safe: PostgreSQL accepts a row whose CHECK
evaluates to NULL, so each comparison is guarded by IS NOT NULL (see
ATTEMPT_CHARGE_CONSISTENT). The only deliberate NULL is ck_..._usage_valid.

WARNING: downgrade drops both tables and every chat request and quota-ledger row.
That history cannot be rebuilt; take a backup first.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9749572298d'
down_revision: Union[str, Sequence[str], None] = '4a06e23062ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Frozen at this revision on purpose (do not import app constants into migrations).
# Tests evaluate these exact texts, and compare them with the models' CHECK texts.
ATTEMPT_CHARGE_CONSISTENT = (
    "(status = 'running' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NULL) "
    "OR (status = 'succeeded' AND charged_tokens IS NOT NULL AND ("
    "(prompt_tokens IS NOT NULL AND completion_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens = reserved_tokens))) "
    "OR (status = 'failed_with_usage' AND prompt_tokens IS NOT NULL "
    "AND completion_tokens IS NOT NULL AND charged_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (status IN ('failed_before_call', 'abandoned', 'rate_limited') "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = 0) "
    "OR (status = 'uncertain' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = reserved_tokens)"
)

REQUEST_CHECKS = [
    (
        "ck_chat_requests_status_valid",
        "status IN ('in_progress', 'succeeded', 'failed')",
    ),
    (
        "ck_chat_requests_idempotency_key_format",
        "idempotency_key ~ '^[A-Za-z0-9._:-]{1,128}$'",
    ),
    (
        "ck_chat_requests_request_sha256_format",
        "request_sha256 ~ '^[0-9a-f]{64}$'",
    ),
    (
        "ck_chat_requests_completed_at_consistent",
        "(status = 'in_progress') = (completed_at IS NULL)",
    ),
    (
        "ck_chat_requests_error_code_consistent",
        "(status = 'failed') = (error_code IS NOT NULL)",
    ),
    (
        "ck_chat_requests_new_conversation_unset_until_success",
        "NOT creates_conversation OR status = 'succeeded' OR conversation_id IS NULL",
    ),
]

ATTEMPT_CHECKS = [
    (
        "ck_chat_request_attempts_attempt_number_positive",
        "attempt_number >= 1",
    ),
    (
        "ck_chat_request_attempts_status_valid",
        "status IN ('running', 'succeeded', 'failed_before_call', 'failed_with_usage', "
        "'rate_limited', 'uncertain', 'abandoned')",
    ),
    (
        "ck_chat_request_attempts_lease_consistent",
        "(status = 'running') = (lease_expires_at IS NOT NULL)",
    ),
    (
        "ck_chat_request_attempts_finished_at_consistent",
        "(status = 'running') = (finished_at IS NULL)",
    ),
    (
        "ck_chat_request_attempts_error_code_consistent",
        "(status IN ('running', 'succeeded')) = (error_code IS NULL)",
    ),
    (
        "ck_chat_request_attempts_reserved_tokens_positive",
        "reserved_tokens > 0",
    ),
    (
        # NULL is deliberate here: absent usage (both NULL) is valid.
        "ck_chat_request_attempts_usage_valid",
        "(prompt_tokens IS NULL) = (completion_tokens IS NULL) "
        "AND prompt_tokens >= 0 AND completion_tokens >= 0",
    ),
    (
        "ck_chat_request_attempts_llm_started_consistent",
        "status = 'running' "
        "OR (status IN ('succeeded', 'failed_with_usage', 'rate_limited', 'uncertain') "
        "AND llm_started_at IS NOT NULL) "
        "OR (status IN ('failed_before_call', 'abandoned') AND llm_started_at IS NULL)",
    ),
    (
        "ck_chat_request_attempts_charged_tokens_null_iff_running",
        "(status = 'running') = (charged_tokens IS NULL)",
    ),
    (
        "ck_chat_request_attempts_charge_consistent",
        ATTEMPT_CHARGE_CONSISTENT,
    ),
]


def upgrade() -> None:
    """Upgrade schema."""
    # chat_requests first: attempts reference it. conversation_id, user_message_id and
    # assistant_message_id are ON DELETE SET NULL, so deleting a conversation or
    # message keeps the request (and its attempts, the quota ledger).
    op.create_table(
        "chat_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("creates_conversation", sa.Boolean(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("user_message_id", sa.Integer(), nullable=True),
        sa.Column("assistant_message_id", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_chat_requests_owner_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_chat_requests_conversation_id_conversations",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_message_id"],
            ["messages.id"],
            name="fk_chat_requests_user_message_id_messages",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["assistant_message_id"],
            ["messages.id"],
            name="fk_chat_requests_assistant_message_id_messages",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "owner_id", "idempotency_key", name="uq_chat_requests_owner_id_idempotency_key"
        ),
        *[sa.CheckConstraint(condition, name=name) for name, condition in REQUEST_CHECKS],
    )
    op.create_index("ix_chat_requests_conversation_id", "chat_requests", ["conversation_id"])
    op.create_index("ix_chat_requests_user_message_id", "chat_requests", ["user_message_id"])
    op.create_index(
        "ix_chat_requests_assistant_message_id", "chat_requests", ["assistant_message_id"]
    )

    op.create_table(
        "chat_request_attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Integer(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reserved_tokens", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("charged_tokens", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("llm_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["chat_requests.id"],
            name="fk_chat_request_attempts_request_id_chat_requests",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "request_id",
            "attempt_number",
            name="uq_chat_request_attempts_request_id_attempt_number",
        ),
        *[sa.CheckConstraint(condition, name=name) for name, condition in ATTEMPT_CHECKS],
    )
    # Partial unique indexes: at most one running and at most one succeeded attempt
    # per request.
    op.create_index(
        "uq_chat_request_attempts_one_running",
        "chat_request_attempts",
        ["request_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index(
        "uq_chat_request_attempts_one_succeeded",
        "chat_request_attempts",
        ["request_id"],
        unique=True,
        postgresql_where=sa.text("status = 'succeeded'"),
    )
    op.create_index(
        "ix_chat_request_attempts_llm_started_at",
        "chat_request_attempts",
        ["llm_started_at"],
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops both tables (and with them their constraints and indexes), child first.
    Every chat request and quota-ledger row is lost; see the module docstring.
    """
    op.drop_table("chat_request_attempts")
    op.drop_table("chat_requests")
