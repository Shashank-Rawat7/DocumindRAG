"""Indexing backfill (Component 12 / C5.5).

Dry run is the default and changes nothing. Usage (PowerShell, from the project root):

    venv\\Scripts\\python.exe scripts\\backfill_indexing.py
    venv\\Scripts\\python.exe scripts\\backfill_indexing.py --database documind_db --limit 5

--execute indexes the documents the plan lists, one at a time. It requires --database NAME,
which is checked against the database URL and against the live server before anything is
written. Read a dry run first. Then, for example, index a single document as a canary:

    venv\\Scripts\\python.exe scripts\\backfill_indexing.py --database documind_db --limit 1 --execute

Things to know before using --execute:
  * The embedding model is loaded offline from the local cache, once, and checked with one
    real embedding before any document is touched. A failed check changes nothing (exit 2).
  * Ctrl-C stops after the current document. A document that was in progress stays locked
    until its lease expires, and a rerun skips it until then (exit 130).
  * If a document was queued but its database session then failed to close, it is reported
    as an error and stays pending; a later run picks it up.
  * Library warnings are written to stderr as they happen, next to the progress lines.
  * Exit codes: 0 success or dry run; 1 failures or blocked documents; 2 refusal or failed
    model check; 130 interrupted.

All logic lives in app/rag/backfill_cli.py; this file only makes the project importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.backfill_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())