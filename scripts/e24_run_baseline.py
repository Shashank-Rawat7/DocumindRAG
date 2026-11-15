"""Run the retrieval-only dense baseline on the development split (E2.4). Read-only database; the report goes
outside the repository.

Usage (PowerShell, from the project root; the application must not be running; the working tree must be clean
and committed):

    venv\\Scripts\\python.exe scripts\\e24_run_baseline.py --database documind_db --owner-id N ^
        --evaluation-email-sha256 <64 hex> --expected-freeze-record-sha256 <64 lowercase hex> ^
        --out C:\\path\\outside\\repo\\e24-baseline-report.json

``--expected-freeze-record-sha256`` is the SHA-256 of the raw eval/dataset/freeze_record.json, taken from outside
the repository (for example from the frozen-dataset record you wrote down at freeze time). It is the trust anchor
of the run and is never read from the repository. Exit codes: 0 report written, 1 failure after reading started,
2 refusal or invalid input.

All logic lives in app/rag/baseline_cli.py; this file only makes the project importable. The report is the
retrieval-only dense baseline of the dev split, not the full V1 RAG baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.baseline_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
