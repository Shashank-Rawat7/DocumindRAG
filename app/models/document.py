from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship, deferred
from sqlalchemy.sql import func
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("length(trim(filename)) > 0", name="ck_documents_filename_not_blank"),
        CheckConstraint(
            "processing_status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_documents_processing_status_valid",
        ),
        # --- Component 12: indexing metadata ---
        CheckConstraint(
            "indexing_status IS NULL OR indexing_status IN ('pending', 'indexing', 'indexed', 'failed')",
            name="ck_documents_indexing_status_valid",
        ),
        CheckConstraint("source_version >= 1", name="ck_documents_source_version_positive"),
        CheckConstraint("indexing_attempt >= 0", name="ck_documents_indexing_attempt_nonnegative"),
        CheckConstraint(
            "indexed_source_version IS NULL OR "
            "(indexed_source_version >= 1 AND indexed_source_version <= source_version)",
            name="ck_documents_indexed_source_version_valid",
        ),
        CheckConstraint(
            "source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_documents_source_sha256_format",
        ),
        CheckConstraint(
            "("
            "indexing_status IS NOT DISTINCT FROM 'indexing' "
            "AND indexing_lease_expires_at IS NOT NULL"
            ") OR ("
            "indexing_status IS DISTINCT FROM 'indexing' "
            "AND indexing_lease_expires_at IS NULL"
            ")",
            name="ck_documents_indexing_lease_consistent",
        ),
        CheckConstraint(
            "indexing_status IS DISTINCT FROM 'indexed' OR ("
            "indexed_source_version IS NOT NULL AND source_sha256 IS NOT NULL AND "
            "indexed_signature IS NOT NULL AND indexed_embedding_id IS NOT NULL AND "
            "indexed_at IS NOT NULL)",
            name="ck_documents_indexed_metadata_complete",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    description = Column(String, nullable=True)
    stored_name = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    processing_status = Column(String, nullable=True)
    processing_error = Column(String, nullable=True)
    extracted_text = deferred(Column(Text, nullable=True))
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_documents_owner_id_users"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # --- Component 12: source identity ---
    # Version of extracted_text. Always 1 in V1 (no replacement/reprocessing yet).
    source_version = Column(Integer, nullable=False, server_default=text("1"))
    # Lowercase hex SHA-256 of extracted_text.encode("utf-8") (NOT the uploaded file
    # bytes). Computed by the indexer and written only in the atomic publish, so it
    # describes the text the current chunks were built from. Citation snapshots copy
    # this value; citation preview recomputes it from extracted_text and compares.
    source_sha256 = Column(String(64), nullable=True)

    # --- Component 12: indexing lifecycle (separate from processing_status) ---
    # NULL = never queued for indexing.
    indexing_status = Column(String, nullable=True)
    indexing_error = Column(String, nullable=True)
    # Attempt generation: incremented on every claim. A worker may publish only if
    # the row still holds the attempt it claimed (stale-worker protection).
    indexing_attempt = Column(Integer, nullable=False, server_default=text("0"))
    # Deadline of the current claim; set if and only if indexing_status = 'indexing'.
    # Duration comes from config (default 15 min).
    indexing_lease_expires_at = Column(DateTime(timezone=True), nullable=True)

    # --- Component 12: last successful publication ---
    # Written only by a successful publish. A failed attempt keeps these values,
    # source_sha256 and the published chunks only while that publication is still
    # valid for the current text; an invalid one is discarded (chunks deleted, these
    # columns and source_sha256 set to NULL).
    indexed_source_version = Column(Integer, nullable=True)
    indexed_signature = Column(String, nullable=True)
    indexed_embedding_id = Column(String, nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    owner = relationship("User", back_populates="documents")
    # DB-level ON DELETE CASCADE removes chunks; the ORM must not load or null them.
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunk.chunk_index",
    )