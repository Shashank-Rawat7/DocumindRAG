import logging

from app.core import storage
from app.core.database import SessionLocal
from app.core.text_extraction import ExtractionError, extract_text
from app.models.document import Document

logger = logging.getLogger("documind")


def _mark_failed(db, document_id: int, reason: str) -> None:
    doc = db.get(Document, document_id)
    if doc is None:
        return
    doc.processing_status = "failed"
    doc.processing_error = reason[:500]
    db.commit()


def _run(db, document_id: int) -> None:
    doc = db.get(Document, document_id)
    if doc is None or not doc.stored_name:
        return

    doc.processing_status = "processing"
    doc.processing_error = None
    db.commit()

    try:
        text = extract_text(storage.get_file_path(doc.stored_name))
        doc.extracted_text = text
        doc.processing_status = "completed"
        db.commit()
    except ExtractionError as exc:
        db.rollback()
        _mark_failed(db, document_id, str(exc))
    except Exception:
        db.rollback()
        logger.error(
            "Document processing failed | document_id=%s", document_id, exc_info=True
        )
        _mark_failed(db, document_id, "Failed to extract text from this file")


def process_document(document_id: int) -> None:
    db = SessionLocal()
    try:
        _run(db, document_id)
    except Exception:
        logger.error(
            "Document processing crashed | document_id=%s", document_id, exc_info=True
        )
        try:
            db.rollback()
            _mark_failed(db, document_id, "Processing failed unexpectedly")
        except Exception:
            logger.error(
                "Could not record failure | document_id=%s", document_id, exc_info=True
            )
    finally:
        db.close()