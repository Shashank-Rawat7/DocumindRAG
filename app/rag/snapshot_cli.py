"""Snapshot command line for E1.5 (``scripts/e15_snapshot.py`` only makes the project importable).

    venv\\Scripts\\python.exe scripts\\e15_snapshot.py take --database documind_db --out C:\\...\\s0.json
    venv\\Scripts\\python.exe scripts\\e15_snapshot.py compare BEFORE.json AFTER.json
    venv\\Scripts\\python.exe scripts\\e15_snapshot.py compare S0.json S1.json --expect-evaluation --evaluation-email-sha256 <64 hex>

``take`` reads the database in one read-only REPEATABLE READ transaction (SELECT statements
only, always rolled back) and writes one snapshot file. It refuses unless the same static
safety checks as the indexing backfill pass and the live database name matches --database.
The output file must be an absolute path outside the repository, in an existing folder, and must
not exist (not even as a dangling link). "Outside" is decided on fully resolved paths: symbolic
links, Windows junctions, ``..`` parts and (where the OS ignores case) letter case cannot be used to
reach the repository. Nothing is ever overwritten.

``compare`` needs no database. Two files are verified and compared; the two snapshots must also come from compatible environments (same pgvector version,
extra_float_digits and encoding, same PostgreSQL major version; the database name is ignored).
--expect-evaluation additionally checks the E1.5 invariants (nothing pre-existing removed or
changed, exactly --users new users and --documents new documents, ownership of the additions) and
REQUIRES --evaluation-email-sha256: the SHA-256 of the evaluation user's email. The email itself is
never accepted, printed or stored. --users, --documents and --evaluation-email-sha256 are refused
without --expect-evaluation.

Exit codes: 0 success (identical, or every expected invariant holds); 1 differences or failed
invariants, or a failure while reading; 2 refusal, unreadable or invalid input.
Nothing printed contains a password, a connection URL, an email, extracted text, chunk text
or an embedding.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from app.rag.backfill_safety import (
    SafetyFailure,
    TargetInfo,
    check_live_database,
    check_static_safety,
    default_repo_root,
    environment_is_explicit,
    inspect_filesystem,
)
from app.rag.evaluation.schema import EvaluationDataError, Issue
from app.rag.evaluation.snapshot import (
    TABLES,
    check_evaluation_delta,
    collect_snapshot,
    compare_snapshots,
    format_report,
    load_snapshot,
    snapshot_to_text,
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2

_SHA_HEX = 64
_DEFAULT_NEW_USERS = 1
_DEFAULT_NEW_DOCUMENTS = 6


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _sha256_hex(value: str) -> str:
    lowered = value.strip().lower()
    if len(lowered) != _SHA_HEX or any(char not in "0123456789abcdef" for char in lowered):
        raise argparse.ArgumentTypeError("must be 64 hexadecimal characters")
    return lowered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e15_snapshot.py",
        description="Take or compare deterministic DocuMind database snapshots (E1.5).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    take = commands.add_parser("take", help="read the database and write one snapshot file")
    take.add_argument("--database", required=True, type=_non_blank,
                      help="database name; must equal the name in the database URL and on the server")
    take.add_argument("--out", required=True, type=_non_blank,
                      help="snapshot file to create (outside the repository; must not exist)")

    compare = commands.add_parser("compare", help="compare two snapshot files (no database access)")
    compare.add_argument("before", type=_non_blank)
    compare.add_argument("after", type=_non_blank)
    compare.add_argument("--expect-evaluation", action="store_true",
                         help="also check the E1.5 ingestion invariants")
    compare.add_argument("--users", type=_non_negative_int, default=None,
                         help="expected number of new users (default 1; only with --expect-evaluation)")
    compare.add_argument("--documents", type=_non_negative_int, default=None,
                         help="expected number of new documents (default 6; only with --expect-evaluation)")
    compare.add_argument("--evaluation-email-sha256", type=_sha256_hex, default=None,
                         help="SHA-256 of the evaluation user's email (required with --expect-evaluation); "
                              "every new user must match it")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------- runtime ----------

@dataclass(frozen=True, slots=True)
class Runtime:
    """Application configuration the ``take`` command needs. Secrets are kept out of repr."""

    engine: Any = field(repr=False)
    database_url: str = field(repr=False)
    environment: str | None
    environment_explicit: bool
    upload_dir: str
    upload_root: Path


def load_runtime() -> Runtime:
    """Import the application configuration (this builds the engine; it does not connect)."""
    from app.core import storage
    from app.core.config import settings
    from app.core.database import engine

    return Runtime(
        engine=engine,
        database_url=settings.database_url,
        environment=settings.environment,
        environment_explicit=environment_is_explicit(settings),
        upload_dir=settings.upload_dir,
        upload_root=storage.UPLOAD_ROOT,
    )


# ---------- helpers ----------

def _ascii(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


def _resolve(path: Path) -> Path | None:
    """The path with every symbolic link and Windows junction resolved, whether or not it exists yet.

    ``os.path.realpath`` (non-strict) resolves the part that exists and appends the rest, so a
    file that is about to be created is judged by where its folder really is. None if the
    path cannot be resolved.
    """
    try:
        return Path(os.path.realpath(path))
    except (OSError, ValueError, RuntimeError):
        return None


def _comparable(value: Path) -> str:
    """A string for comparing resolved paths: no extended-length prefix, OS case rules applied."""
    text = str(value)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(text)


def _inside(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or below it once both are fully resolved.

    Case-insensitive where the OS is. If either path cannot be resolved the answer is True
    (fail closed: it cannot be shown to be outside).
    """
    resolved_path, resolved_root = _resolve(path), _resolve(root)
    if resolved_path is None or resolved_root is None:
        return True
    try:
        return os.path.commonpath([_comparable(resolved_path), _comparable(resolved_root)]) == _comparable(resolved_root)
    except ValueError:  # different drives
        return False


def _print_issues(issues: Sequence[Issue], err: TextIO) -> None:
    for issue in issues:
        print(f"  {issue.code} at {issue.location}: {issue.message}", file=err)


def _print_failures(failures: Sequence[SafetyFailure], target: TargetInfo | None, err: TextIO) -> None:
    print("REFUSED: the safety checks failed. Nothing was read or written.", file=err)
    if target is not None:
        print(f"  target: {_ascii(target.describe())}", file=err)
    for failure in failures:
        print(f"  [{failure.check.value}] {_ascii(failure.message)}", file=err)


def _check_output_path(raw: str, repo_root: Path, err: TextIO) -> Path | None:
    """The fully resolved path to create, or None after printing why it was refused."""
    path = Path(raw)
    if not path.is_absolute():
        print("REFUSED: --out must be an absolute path.", file=err)
        return None
    if _inside(path, repo_root):
        print("REFUSED: --out must be outside the repository.", file=err)
        return None
    if not path.parent.is_dir():
        print("REFUSED: the folder of --out does not exist.", file=err)
        return None
    if os.path.lexists(path):  # lexists: a dangling link is also "exists"
        print("REFUSED: --out already exists; a snapshot is never overwritten.", file=err)
        return None
    return _resolve(path)


# ---------- take ----------

def _take(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[], Runtime],
    environ: Mapping[str, str],
    cwd: Path,
    repo_root: Path,
    clock: Callable[[], datetime],
    out: TextIO,
    err: TextIO,
) -> int:
    path = _check_output_path(args.out, repo_root, err)
    if path is None:
        return EXIT_REFUSED

    try:
        runtime = runtime_loader()
    except Exception as exc:  # noqa: BLE001 - only the class name is ever shown
        print(f"REFUSED: could not load the application configuration ({type(exc).__name__})", file=err)
        return EXIT_REFUSED

    filesystem = inspect_filesystem(repo_root, runtime.upload_dir, runtime.upload_root)
    report = check_static_safety(
        database_url=runtime.database_url,
        environment=runtime.environment,
        environment_explicit=runtime.environment_explicit,
        expected_database=args.database,
        cwd=cwd,
        repo_root=repo_root,
        upload_root=runtime.upload_root,
        filesystem=filesystem,
        environ=environ,
    )
    if not report.ok:
        _print_failures(report.failures, report.target, err)
        return EXIT_REFUSED

    try:
        connection = runtime.engine.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: could not connect to the database ({type(exc).__name__}). Nothing was written.", file=err)
        return EXIT_FAILURE

    with connection:
        try:
            # Applied before the first statement, so the transaction begins read-only and REPEATABLE READ.
            connection = connection.execution_options(isolation_level="REPEATABLE READ", postgresql_readonly=True)
            failure = check_live_database(connection, args.database)
            if failure is not None:
                _print_failures([failure], report.target, err)
                return EXIT_REFUSED

            def run_query(sql: str) -> list[dict]:
                return [dict(row) for row in connection.exec_driver_sql(sql).mappings().all()]

            snapshot = collect_snapshot(run_query, created_at=clock())
        except EvaluationDataError as exc:
            print("FAILED: the snapshot could not be taken. Nothing was written.", file=err)
            _print_issues(exc.issues, err)
            return EXIT_FAILURE
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: reading the database failed ({type(exc).__name__}). Nothing was written.", file=err)
            return EXIT_FAILURE
        finally:
            try:
                connection.rollback()  # the transaction is read-only; it is never committed
            except Exception:  # noqa: BLE001
                pass

    text = snapshot_to_text(snapshot)
    # The folder may have changed while the database was read: check again, immediately before
    # creating the file, and create it at the resolved location.
    path = _check_output_path(args.out, repo_root, err)
    if path is None:
        return EXIT_REFUSED
    try:
        load_snapshot(text)  # the text that will be written is read back and verified
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except EvaluationDataError as exc:
        print("FAILED: the snapshot did not verify after serialisation. Nothing was written.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_FAILURE
    except OSError as exc:
        print(f"FAILED: the snapshot file could not be written ({type(exc).__name__}).", file=err)
        return EXIT_FAILURE

    file_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"snapshot written: {path}", file=out)
    print(f"database: {_ascii(report.target.describe()) if report.target else '?'}", file=out)
    print(f"created (UTC): {snapshot['created_at_utc']}", file=out)
    for table in TABLES:
        print(f"{table}: {snapshot['tables'][table]['row_count']} rows, "
              f"fingerprint {snapshot['tables'][table]['fingerprint']}", file=out)
    print(f"content fingerprint: {snapshot['fingerprint']}", file=out)
    print(f"file bytes: {len(text.encode('utf-8'))}  file sha256: {file_sha}", file=out)
    return EXIT_OK


# ---------- compare ----------

def _read_snapshot(raw: str, err: TextIO) -> dict | None:
    try:
        text = Path(raw).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"REFUSED: a snapshot file could not be read ({type(exc).__name__}).", file=err)
        return None
    try:
        return load_snapshot(text)
    except EvaluationDataError as exc:
        print("REFUSED: a snapshot file is not a valid snapshot.", file=err)
        _print_issues(exc.issues, err)
        return None


def _compare(args: argparse.Namespace, *, out: TextIO, err: TextIO) -> int:
    evaluation_options = (args.users, args.documents, args.evaluation_email_sha256)
    if not args.expect_evaluation and any(option is not None for option in evaluation_options):
        print("REFUSED: --users, --documents and --evaluation-email-sha256 are only valid with --expect-evaluation.",
              file=err)
        return EXIT_REFUSED
    if args.expect_evaluation and args.evaluation_email_sha256 is None:
        print("REFUSED: --expect-evaluation requires --evaluation-email-sha256 (the SHA-256 of the "
              "evaluation user's email; the email itself is never accepted).", file=err)
        return EXIT_REFUSED
    before = _read_snapshot(args.before, err)
    after = _read_snapshot(args.after, err)
    if before is None or after is None:
        return EXIT_REFUSED
    try:
        comparison = compare_snapshots(before, after)
        issues: list[Issue] = []
        if args.expect_evaluation:
            issues = check_evaluation_delta(
                before,
                after,
                evaluation_email_sha256=args.evaluation_email_sha256,
                expected_new_users=_DEFAULT_NEW_USERS if args.users is None else args.users,
                expected_new_documents=_DEFAULT_NEW_DOCUMENTS if args.documents is None else args.documents,
            )
    except EvaluationDataError as exc:
        print("REFUSED: the snapshots cannot be compared.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_REFUSED
    out.write(format_report(comparison, issues))
    if args.expect_evaluation:
        return EXIT_OK if not issues else EXIT_FAILURE
    return EXIT_OK if comparison.identical else EXIT_FAILURE


# ---------- entry point ----------

def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: Callable[[], Runtime] | None = None,
    environ: MutableMapping[str, str] | None = None,
    cwd: Path | None = None,
    repo_root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the command line and return its exit code. The keyword arguments exist for tests."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        args = parse_arguments(argv)
    except SystemExit as exit_request:  # argparse: 0 for --help, 2 for a usage error
        code = exit_request.code
        if code is None:
            return EXIT_OK
        return code if isinstance(code, int) else EXIT_REFUSED

    try:
        if args.command == "compare":
            return _compare(args, out=out, err=err)
        return _take(
            args,
            runtime_loader=runtime_loader if runtime_loader is not None else load_runtime,
            environ=os.environ if environ is None else environ,
            cwd=Path.cwd().resolve() if cwd is None else cwd,
            repo_root=default_repo_root() if repo_root is None else repo_root,
            clock=clock if clock is not None else (lambda: datetime.now(timezone.utc)),
            out=out,
            err=err,
        )
    except KeyboardInterrupt:
        print("Interrupted. Nothing was written to the database.", file=err)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error ({_ascii(type(exc).__name__)}).", file=err)
        return EXIT_FAILURE
