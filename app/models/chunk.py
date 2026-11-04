from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import deferred, relationship
from sqlalchemy.sql import func

from app.core.database import Base

EMBEDDING_DIM = 384  # intfloat/multilingual-e5-small


class DocumentChunk(Base):
    """One exact slice of documents.extracted_text plus its embedding.

    Invariants (enforced by the indexer in C3/C4; the length rule also by the DB):
      text == document.extracted_text[start_offset:end_offset]
      span_sha256 == sha256(text.encode("utf-8")).hexdigest()
    Offsets are Python str indices (Unicode code points), half-open
    [start_offset, end_offset), which match PostgreSQL length()/substr() on a
    UTF-8 database.
    Ownership is NOT stored here: always join to documents and filter owner_id.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_id_chunk_index"
        ),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_chunk_index_nonnegative"),
        CheckConstraint(
            "start_offset >= 0 AND start_offset < end_offset",
            name="ck_document_chunks_offsets_valid",
        ),
        CheckConstraint(
            "length(text) = end_offset - start_offset",
            name="ck_document_chunks_text_length_matches_offsets",
        ),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start >= 1 AND page_start <= page_end)",
            name="ck_document_chunks_pages_valid",
        ),
        CheckConstraint(
            "span_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_chunks_span_sha256_format",
        ),
    )

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk_document_chunks_document_id_documents",
        ),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    span_sha256 = Column(String(64), nullable=False)
    embedding = deferred(Column(Vector(EMBEDDING_DIM), nullable=False))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")