from sqlalchemy.orm import Session
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentUpdate


def create_document(db: Session, data: DocumentCreate) -> Document:
    new_doc = Document(filename=data.filename, description=data.description)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return new_doc


def get_document(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()


def get_documents(db: Session, skip: int = 0, limit: int = 10) -> list[Document]:
    return db.query(Document).offset(skip).limit(limit).all()


def update_document(db: Session, document_id: int, data: DocumentUpdate) -> Document | None:
    doc = get_document(db, document_id)
    if not doc:
        return None
    if data.filename is not None:
        doc.filename = data.filename
    if data.description is not None:
        doc.description = data.description
    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, document_id: int) -> bool:
    doc = get_document(db, document_id)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True