from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Message(Base):
    """One persisted chat message (a user question or an assistant answer).

    Messages are written only after a request completes successfully, so there is
    no status column. seq orders messages inside a conversation independently of
    timestamps. Ownership is NOT stored here: always join to conversations and
    filter owner_id. Citations are not part of this table (added later by C9).
    """

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uq_messages_conversation_id_seq"),
        CheckConstraint("seq >= 1", name="ck_messages_seq_positive"),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role_valid"),
        CheckConstraint("length(trim(content)) > 0", name="ck_messages_content_not_blank"),
    )

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer,
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
            name="fk_messages_conversation_id_conversations",
        ),
        nullable=False,
    )
    seq = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
