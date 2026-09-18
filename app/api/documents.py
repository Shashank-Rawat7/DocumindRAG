from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from app.crud import document as crud

router = APIRouter()


@router.post("/documents", response_model=DocumentResponse, status_code=201)
def create_document(data: DocumentCreate, db: Session = Depends(get_db)):
    return crud.create_document(db, data)


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.get_documents(db, skip=skip, limit=limit)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = crud.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document(document_id: int, data: DocumentUpdate, db: Session = Depends(get_db)):
    doc = crud.update_document(db, document_id, data)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    success = crud.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")