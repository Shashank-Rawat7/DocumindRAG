"""C5.5 standalone import path: the backfill must not depend on app.main for its models."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLI_IMPORT_SEQUENCE = (
    "import app.rag.backfill_cli as cli\n"
    "cli.load_runtime()\n"                                   # constructs the embedder; no model load, no connection
    "from app.rag.backfill_selection import select_document_states\n"
    "from app.rag.backfill_sources import attach_source_status\n"
)


@pytest.mark.parametrize(
    "imports",
    ["import app.rag.backfill_selection\n", CLI_IMPORT_SEQUENCE],
    ids=["selection_module", "cli_import_sequence"],
)
def test_the_mappers_configure_without_app_main_or_the_api_imports(imports):
    code = (
        "import sys\n"
        + imports
        + "assert 'app.main' not in sys.modules\n"
        + "from sqlalchemy.orm import configure_mappers\n"
        + "configure_mappers()\n"
        + "print('MAPPERS OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MAPPERS OK" in result.stdout