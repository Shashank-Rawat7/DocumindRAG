from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Conversation(Base):
    """A chat thread owned by one user.

    Ownership is stored here (owner_id); messages carry no owner and are always
    reached through a join to their conversation. Deleting a conversation deletes
    its messages in the database (ON DELETE CASCADE). The C7 chat request/attempt
    tables (migration #7) reference conversations with ON DELETE SET NULL.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "title IS NULL OR length(trim(title)) > 0",
            name="ck_conversations_title_not_blank",
        ),
        # Serves the owner's conversation list newest-first and owner lookups.
        Index("ix_conversations_owner_id_updated_at", "owner_id", "updated_at"),
    )

    id = Column(Integer, primary_key=True)
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_conversations_owner_id_users"),
        nullable=False,
    )
    # Title policy is decided later (C9/C10); NULL means no title yet.
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # No ORM onupdate and no trigger in this project: the code that stores a message
    # pair sets this explicitly.
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # DB-level ON DELETE CASCADE removes messages; the ORM must not load or null them.
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.seq",
    )
