from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint
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

    owner = relationship("User", back_populates="documents")
