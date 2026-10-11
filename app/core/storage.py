import uuid
import logging
logger = logging.getLogger("documind")

from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings

UPLOAD_ROOT = Path(settings.upload_dir).resolve()
CHUNK_SIZE = 1024 * 1024  # 1 MB


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


def save_upload(file: UploadFile) -> tuple[str, int]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(ext)

    stored_name = f"{uuid.uuid4().hex}{ext}"
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_ROOT / stored_name

    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise FileTooLargeError()
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    return stored_name, size


def delete_file(stored_name: str | None) -> None:
    if not stored_name:
        return

    try:
        path = get_file_path(stored_name)
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "Failed to delete stored file %s",
            stored_name,
            exc_info=True,
        )


def get_file_path(stored_name: str) -> Path:
    path = (UPLOAD_ROOT / stored_name).resolve()

    if path.parent != UPLOAD_ROOT:
        raise ValueError("Invalid stored filename")

    return path
