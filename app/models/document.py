from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("length(trim(filename)) > 0", name="ck_documents_filename_not_blank"),
    )

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    description = Column(String, nullable=True)
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_documents_owner_id_users"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    owner = relationship("User", back_populates="documents")
