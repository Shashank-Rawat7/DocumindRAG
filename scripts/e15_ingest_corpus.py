"""Ingest and verify the six E1.5 evaluation corpus documents.

Usage (PowerShell, from the project root; see app/rag/corpus_ingest_cli.py for every option):

    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py plan   --database documind_db --owner-id N --evaluation-email-sha256 <64 hex>
    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py verify --database documind_db --owner-id N --evaluation-email-sha256 <64 hex>

There is no indexing command. Index with the unmodified scripts/backfill_indexing.py.
All logic lives in app/rag/corpus_ingest_cli.py; this file only makes the project importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.corpus_ingest_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
