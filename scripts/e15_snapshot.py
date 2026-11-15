"""Snapshot and compare for E1.5 (take reads the database read-only; compare needs no database).

Usage (PowerShell, from the project root):

    venv\\Scripts\\python.exe scripts\\e15_snapshot.py take --database documind_db --out C:\\path\\outside\\repo\\s0.json
    venv\\Scripts\\python.exe scripts\\e15_snapshot.py compare S0.json S1.json --expect-evaluation --evaluation-email-sha256 <64 hex>

All logic lives in app/rag/snapshot_cli.py; this file only makes the project importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.snapshot_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
