from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from app.core.database import Base


class MessageCitation(Base):
    """One source cited by an assistant message (C9.4b).

    Rows are written only by chat_request_finalization.publish_success, in the same
    transaction as the message pair, after the cited chunk was checked against the chunks
    retrieved for THAT request and against the database (existence, owner, identity).
    ordinal is the 1-based order of first appearance in the answer; marker is the n of the
    [n] the model wrote (1..999, the parser's limit). Both are unique per message.

    The snapshot columns copy the chunk's identity and immutable metadata at publication
    time, so a citation survives re-indexing and document deletion: chunk_id is then set to
    NULL (ON DELETE SET NULL) and the snapshot stays. document_id deliberately has NO
    foreign key for the same reason. There is no text snapshot: span_sha256 and the offsets
    identify the exact slice of the document's extracted text. Ownership is NOT stored
    here: join messages to conversations and filter owner_id.
    """

    __tablename__ = "message_citations"
    __table_args__ = (
        UniqueConstraint(
            "message_id", "ordinal", name="uq_message_citations_message_id_ordinal"
        ),
        UniqueConstraint(
            "message_id", "marker", name="uq_message_citations_message_id_marker"
        ),
        CheckConstraint("ordinal >= 1", name="ck_message_citations_ordinal_positive"),
        CheckConstraint(
            "marker >= 1 AND marker <= 999", name="ck_message_citations_marker_valid"
        ),
        CheckConstraint(
            "chunk_index >= 0", name="ck_message_citations_chunk_index_nonnegative"
        ),
        CheckConstraint(
            "start_offset >= 0 AND start_offset < end_offset",
            name="ck_message_citations_offsets_valid",
        ),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start >= 1 AND page_start <= page_end)",
            name="ck_message_citations_pages_valid",
        ),
        CheckConstraint(
            "span_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_message_citations_span_sha256_format",
        ),
        CheckConstraint(
            "source_version >= 1", name="ck_message_citations_source_version_positive"
        ),
        CheckConstraint(
            "source_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_message_citations_source_sha256_format",
        ),
        Index("ix_message_citations_chunk_id", "chunk_id"),
    )

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer,
        ForeignKey(
            "messages.id",
            ondelete="CASCADE",
            name="fk_message_citations_message_id_messages",
        ),
        nullable=False,
    )
    ordinal = Column(Integer, nullable=False)
    marker = Column(Integer, nullable=False)
    chunk_id = Column(
        Integer,
        ForeignKey(
            "document_chunks.id",
            ondelete="SET NULL",
            name="fk_message_citations_chunk_id_document_chunks",
        ),
        nullable=True,
    )
    # Snapshot of the retrieved chunk (no foreign key on document_id, on purpose).
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    span_sha256 = Column(String(64), nullable=False)
    source_version = Column(Integer, nullable=False)
    source_sha256 = Column(String(64), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
