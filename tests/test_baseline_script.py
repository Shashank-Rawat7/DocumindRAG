"""E2.4e: the operator wrapper ``scripts/e24_run_baseline.py`` (a thin entry point; all logic is in baseline_cli)."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "e24_run_baseline.py"
BOGUS_ANCHOR = "0123456789abcdef" * 4
EMAIL_SHA = "ef" * 32
FORBIDDEN = ("sqlalchemy", "app.core", "app.db", "app.database", "app.models", "app.main", "app.rag.embedder",
             "app.rag.retrieval", "app.rag.pilot_cli", "app.rag.snapshot_cli", "app.rag.backfill_safety",
             "app.rag.baseline_run", "app.rag.evaluation.engine", "app.rag.evaluation.pool",
             "app.rag.evaluation.pool_db", "app.rag.evaluation.report", "torch", "sentence_transformers",
             "psycopg", "psycopg2", "pgvector")

# Runs the wrapper as ``__main__`` in a fresh isolated interpreter from an unrelated working directory and reports
# the exit code and which forbidden modules were imported at the end.
CHILD = r"""
import json, runpy, sys
script, forbidden = sys.argv[1], json.loads(sys.argv[2])
sys.argv = [script, *sys.argv[3:]]
code = None
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exit_request:
    code = exit_request.code
loaded = sorted(n for n in sys.modules if any(n == f or n.startswith(f + ".") for f in forbidden))
print(json.dumps({"code": code, "loaded": loaded}))
"""


def run_wrapper(tmp_path: Path, *arguments: str) -> tuple[dict, str, str]:
    done = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(SCRIPT), json.dumps(list(FORBIDDEN)), *arguments],
        capture_output=True, text=True, timeout=120, check=False, cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "HOME")},
    )
    lines = done.stdout.strip().splitlines()
    return json.loads(lines[-1]), done.stdout, done.stderr


def full_arguments(tmp_path: Path, **override: str) -> list[str]:
    values = {"--database": "documind_db", "--owner-id": "7", "--evaluation-email-sha256": EMAIL_SHA,
              "--expected-freeze-record-sha256": BOGUS_ANCHOR, "--out": str(tmp_path / "report.json")}
    values.update(override)
    return [item for pair in values.items() for item in pair]


def test_the_wrapper_is_only_a_path_bootstrap_and_a_call_to_main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [(n.module, [a.name for a in n.names]) for n in imports if isinstance(n, ast.ImportFrom)] == [
        ("__future__", ["annotations"]), ("pathlib", ["Path"]), ("app.rag.baseline_cli", ["main"])]
    assert [a.name for n in imports if isinstance(n, ast.Import) for a in n.names] == ["sys"]
    assert not [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for needle in ("test" + ".jsonl", "audit" + ".jsonl", "subprocess", "open(", "environ"):
        assert needle not in source  # no file access, no process, no environment
    assert source.rstrip().endswith("raise SystemExit(main())")


def test_help_works_from_another_directory_and_names_the_anchor(tmp_path: Path) -> None:
    result, out, err = run_wrapper(tmp_path, "--help")
    assert result["code"] == 0 and result["loaded"] == []
    assert "--expected-freeze-record-sha256" in out and "e24_run_baseline.py" in out
    assert "--dev-file" not in out and "--corpus-dir" not in out


def test_a_missing_anchor_is_a_usage_error_and_nothing_heavy_is_imported(tmp_path: Path) -> None:
    arguments = full_arguments(tmp_path)
    index = arguments.index("--expected-freeze-record-sha256")
    del arguments[index:index + 2]
    result, _, err = run_wrapper(tmp_path, *arguments)
    assert result["code"] == 2 and result["loaded"] == []
    assert "--expected-freeze-record-sha256" in err


@pytest.mark.parametrize("bad", [BOGUS_ANCHOR.upper(), BOGUS_ANCHOR[:-1], " " + BOGUS_ANCHOR, "x" * 64])
def test_a_malformed_anchor_is_refused_before_any_file_is_read(tmp_path: Path, bad: str) -> None:
    result, _, _ = run_wrapper(tmp_path, *full_arguments(tmp_path, **{"--expected-freeze-record-sha256": bad}))
    assert result["code"] == 2 and result["loaded"] == []
    assert not (tmp_path / "report.json").exists()


def test_a_wrong_anchor_is_refused_through_the_wrapper_before_any_heavy_import(tmp_path: Path) -> None:
    # The wrapper uses the real repository root: only the four permitted frozen files are read by the freeze check.
    # Whether the record verifies (and then differs from the bogus anchor) or cannot be verified, the run is refused
    # with exit 2, nothing else is imported or read, nothing is written and no hash or detail is printed.
    result, out, err = run_wrapper(tmp_path, *full_arguments(tmp_path))
    assert result["code"] == 2 and result["loaded"] == []
    assert not (tmp_path / "report.json").exists() and list(tmp_path.iterdir()) == []
    assert BOGUS_ANCHOR not in out + err and "FREEZE_" not in out + err
    assert err.strip().splitlines()[-1] in {
        "REFUSED: the freeze record does not match --expected-freeze-record-sha256. Nothing else was read.",
        "REFUSED: the frozen dataset record could not be verified. Nothing else was read.",
    }


def test_the_pilot_wrapper_still_runs_the_pilot() -> None:
    pilot = (REPO / "scripts" / "e17_run_pilot.py").read_text(encoding="utf-8")
    assert "from app.rag.pilot_cli import main" in pilot and "baseline_cli" not in pilot
