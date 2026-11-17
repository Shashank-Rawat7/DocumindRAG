from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ChatRequestAttempt(Base):
    """One execution attempt of a chat request, and the quota ledger row for it.

    status is the attempt's outcome (8 values). charged_tokens is NULL only while the
    attempt is running; every other status has a non-NULL charge (see the CHECKs).
    llm_started_at is committed BEFORE the Groq call: NULL proves no call was issued.
    Every CHECK below is written so it can never evaluate to NULL (PostgreSQL accepts
    a row when a CHECK is NULL); the one deliberate exception is usage_valid, where
    absent usage is valid. This model holds the schema only: lease and fencing
    behaviour belongs to C8.
    """

    __tablename__ = "chat_request_attempts"
    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "attempt_number",
            name="uq_chat_request_attempts_request_id_attempt_number",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_chat_request_attempts_attempt_number_positive",
        ),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed_before_call', 'failed_with_usage', "
            "'rate_limited', 'uncertain', 'abandoned', 'rejected')",
            name="ck_chat_request_attempts_status_valid",
        ),
        CheckConstraint(
            "(status = 'running') = (lease_expires_at IS NOT NULL)",
            name="ck_chat_request_attempts_lease_consistent",
        ),
        CheckConstraint(
            "(status = 'running') = (finished_at IS NULL)",
            name="ck_chat_request_attempts_finished_at_consistent",
        ),
        CheckConstraint(
            "(status IN ('running', 'succeeded')) = (error_code IS NULL)",
            name="ck_chat_request_attempts_error_code_consistent",
        ),
        CheckConstraint(
            "reserved_tokens > 0",
            name="ck_chat_request_attempts_reserved_tokens_positive",
        ),
        # NULL is deliberate here: absent usage (both NULL) is valid.
        CheckConstraint(
            "(prompt_tokens IS NULL) = (completion_tokens IS NULL) "
            "AND prompt_tokens >= 0 AND completion_tokens >= 0",
            name="ck_chat_request_attempts_usage_valid",
        ),
        CheckConstraint(
            "status = 'running' "
            "OR (status IN ('succeeded', 'failed_with_usage', 'rate_limited', 'uncertain', "
            "'rejected') AND llm_started_at IS NOT NULL) "
            "OR (status IN ('failed_before_call', 'abandoned') AND llm_started_at IS NULL)",
            name="ck_chat_request_attempts_llm_started_consistent",
        ),
        CheckConstraint(
            "(status = 'running') = (charged_tokens IS NULL)",
            name="ck_chat_request_attempts_charged_tokens_null_iff_running",
        ),
        CheckConstraint(
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
            "OR (status IN ('failed_before_call', 'abandoned', 'rate_limited', 'rejected') "
            "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
            "AND charged_tokens IS NOT NULL AND charged_tokens = 0) "
            "OR (status = 'uncertain' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
            "AND charged_tokens IS NOT NULL AND charged_tokens = reserved_tokens)",
            name="ck_chat_request_attempts_charge_consistent",
        ),
        # Partial unique indexes: the database backstop for "one live attempt" and
        # "one success" per request (first partial indexes in this project).
        Index(
            "uq_chat_request_attempts_one_running",
            "request_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
        Index(
            "uq_chat_request_attempts_one_succeeded",
            "request_id",
            unique=True,
            postgresql_where=text("status = 'succeeded'"),
        ),
        # Half-open UTC day-range quota scans by llm_started_at.
        Index("ix_chat_request_attempts_llm_started_at", "llm_started_at"),
    )

    id = Column(Integer, primary_key=True)
    request_id = Column(
        Integer,
        ForeignKey(
            "chat_requests.id",
            ondelete="CASCADE",
            name="fk_chat_request_attempts_request_id_chat_requests",
        ),
        nullable=False,
    )
    # 1-based per request.
    attempt_number = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    # Set if and only if status = 'running'. Compared against clock_timestamp() by C8.
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    # Recorded per attempt so history survives later cap changes; the value comes from C8.
    reserved_tokens = Column(Integer, nullable=False)
    # Actual usage, when the provider reported it.
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    # Settled amount; NULL only while running.
    charged_tokens = Column(Integer, nullable=True)
    # Fixed reason codes only; NULL for running and succeeded attempts.
    error_code = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Written and committed before the Groq call; selects the UTC accounting day.
    llm_started_at = Column(DateTime(timezone=True), nullable=True)
    # Set when the attempt leaves 'running'.
    finished_at = Column(DateTime(timezone=True), nullable=True)

    request = relationship("ChatRequest", back_populates="attempts")
