"""Run the development-split validity pilot (E1.7). Read-only database; the report goes outside the repository.

Usage (PowerShell, from the project root; the application must not be running):

    venv\\Scripts\\python.exe scripts\\e17_run_pilot.py --database documind_db --owner-id N ^
        --evaluation-email-sha256 <64 hex> --out C:\\path\\outside\\repo\\e17-pilot-report.json

All logic lives in app/rag/pilot_cli.py; this file only makes the project importable. This is a validity
pilot, not the V1 baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pilot_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
