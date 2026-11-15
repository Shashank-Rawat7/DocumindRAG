"""Label the E1 evaluation dataset (quote-anchored; see app/rag/evaluation/labelling_cli.py).

Usage (PowerShell, from the project root):

    venv\\Scripts\\python.exe scripts\\e16_label.py show
    venv\\Scripts\\python.exe scripts\\e16_label.py show --key docker-compose-startup-order --start 0 --end 1500
    venv\\Scripts\\python.exe scripts\\e16_label.py check

All logic lives in app/rag/evaluation/labelling_cli.py; this file only makes the project importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.labelling_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
