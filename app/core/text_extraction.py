"""Offline text extraction shared by document processing and the evaluation tools.

This module holds ``extract_text`` and ``ExtractionError`` exactly as they were in ``app/core/processing.py``
(moved, not changed). It imports only the standard library, ``pypdf`` and ``python-docx``: no application
settings, no database engine and no model, so offline callers (the corpus ingest, the labelling tool and the
evaluation runner) can extract text without ``DATABASE_URL`` or ``SECRET_KEY``. ``app.core.processing`` imports
both names from here, so ``processing.extract_text`` is the same function object.
"""
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


class ExtractionError(Exception):
    """A failure whose message is safe to show to the user."""


def extract_text(path: Path) -> str:
    if not path.is_file():
        raise ExtractionError("Stored file not found")

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    elif ext == ".pdf":
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                parts.append(" | ".join(c for c in cells if c))
        text = "\n".join(parts)
    else:
        raise ExtractionError("Unsupported file type")

    text = text.replace("\x00", "")
    if not text.strip():
        raise ExtractionError("No extractable text found (the file may be empty or scanned)")

    return text
