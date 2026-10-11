from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core import storage
from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.rate_limit import limiter
from app.models.user import User
from app.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from app.crud import document as crud

router = APIRouter()

@router.post("/documents", response_model=DocumentResponse, status_code=201)
def create_document(
    data: DocumentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return crud.create_document(db, data, owner_id=current_user.id)

@router.post("/documents/upload", response_model=DocumentResponse, status_code=201)
@limiter.limit("5/minute")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(None, max_length=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        stored_name, size = storage.save_upload(file)
    except storage.UnsupportedFileTypeError:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    except storage.FileTooLargeError:
        raise HTTPException(status_code=413, detail="File too large")

    display_name = Path((file.filename or "").replace("\\", "/")).name.strip()[:255]
    clean_description = description.strip() if description else None

    try:
        return crud.create_uploaded_document(
            db,
            filename=display_name,
            description=clean_description or None,
            stored_name=stored_name,
            content_type=file.content_type,
            size_bytes=size,
            owner_id=current_user.id,
        )
    except Exception:
        db.rollback()
        storage.delete_file(stored_name)
        raise


@router.get("/documents/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = crud.get_document(db, document_id, owner_id=current_user.id)
    if not doc or not doc.stored_name:
        raise HTTPException(status_code=404, detail="File not found")

    path = storage.get_file_path(doc.stored_name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, filename=doc.filename)

@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return crud.get_documents(db, owner_id=current_user.id, skip=skip, limit=limit)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = crud.get_document(
        db,
        document_id,
        owner_id=current_user.id
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: int,
    data: DocumentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = crud.update_document(
        db,
        document_id,
        data,
        owner_id=current_user.id
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    success = crud.delete_document(
        db,
        document_id,
        owner_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")