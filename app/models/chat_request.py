from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ChatRequest(Base):
    """One logical chat request, identified per user by an idempotency key.

    A request is executed by one or more attempts (chat_request_attempts). It stores
    no question or answer text: a successful request only points at the messages it
    produced, so deleting a conversation leaves this row (and its attempts, which are
    the quota ledger) behind with those pointers set to NULL.

    status and creates_conversation have no server default: the C8/C9 code sets them.
    conversation_id: for an existing-conversation request it is the target; for a
    new-conversation request (creates_conversation) it stays NULL until success.
    """

    __tablename__ = "chat_requests"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "idempotency_key", name="uq_chat_requests_owner_id_idempotency_key"
        ),
        CheckConstraint(
            "status IN ('in_progress', 'succeeded', 'failed')",
            name="ck_chat_requests_status_valid",
        ),
        CheckConstraint(
            "idempotency_key ~ '^[A-Za-z0-9._:-]{1,128}$'",
            name="ck_chat_requests_idempotency_key_format",
        ),
        CheckConstraint(
            "request_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_chat_requests_request_sha256_format",
        ),
        CheckConstraint(
            "(status = 'in_progress') = (completed_at IS NULL)",
            name="ck_chat_requests_completed_at_consistent",
        ),
        CheckConstraint(
            "(status = 'failed') = (error_code IS NOT NULL)",
            name="ck_chat_requests_error_code_consistent",
        ),
        CheckConstraint(
            "NOT creates_conversation OR status = 'succeeded' OR conversation_id IS NULL",
            name="ck_chat_requests_new_conversation_unset_until_success",
        ),
        # The three ON DELETE SET NULL foreign keys get an index so the database's
        # FK action finds the referencing rows without scanning the table.
        Index("ix_chat_requests_conversation_id", "conversation_id"),
        Index("ix_chat_requests_user_message_id", "user_message_id"),
        Index("ix_chat_requests_assistant_message_id", "assistant_message_id"),
    )

    id = Column(Integer, primary_key=True)
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_chat_requests_owner_id_users"),
        nullable=False,
    )
    idempotency_key = Column(String(128), nullable=False)
    # Lowercase hex SHA-256 (computed in Python by C8) of the canonical request payload.
    request_sha256 = Column(String(64), nullable=False)
    status = Column(String, nullable=False)
    creates_conversation = Column(Boolean, nullable=False)
    conversation_id = Column(
        Integer,
        ForeignKey(
            "conversations.id",
            ondelete="SET NULL",
            name="fk_chat_requests_conversation_id_conversations",
        ),
        nullable=True,
    )
    user_message_id = Column(
        Integer,
        ForeignKey(
            "messages.id",
            ondelete="SET NULL",
            name="fk_chat_requests_user_message_id_messages",
        ),
        nullable=True,
    )
    assistant_message_id = Column(
        Integer,
        ForeignKey(
            "messages.id",
            ondelete="SET NULL",
            name="fk_chat_requests_assistant_message_id_messages",
        ),
        nullable=True,
    )
    # Fixed reason codes only; set if and only if status = 'failed'.
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Set if and only if the request is terminal (status <> 'in_progress').
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # DB-level ON DELETE CASCADE removes attempts; the ORM must not load or null them.
    attempts = relationship(
        "ChatRequestAttempt",
        back_populates="request",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatRequestAttempt.attempt_number",
    )
