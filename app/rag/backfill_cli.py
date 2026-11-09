"""Command-line driver for the indexing backfill (Component 12 / C5.5, CP5 and CP6c).

Dry run is the default. It never indexes, never loads the embedding model, and never
writes to the database or to files: the database work happens on a connection opened
read-only (``postgresql_readonly``), and the only output is text on stdout/stderr.

``--execute`` (which requires ``--database NAME``) indexes the documents the plan lists,
one at a time. Its order is fixed:

  1. static safety checks (no database connection exists yet);
  2. a read-only connection and the live ``current_database()`` check;
  3. a check that the session factories that will write are bound to that same engine;
  4. read and plan; the planning connection is closed;
  5. print the plan;
  6. the offline embedding preflight (only if something is to be indexed);
  7. per document: re-read and reclassify, queue, run (app/rag/backfill_execute.py).

No database connection or transaction is open while the model embeds. Nothing is written
before step 7. The same embedder object that passed the preflight indexes every document.

Imports that load application configuration (settings, the engine, the models, the indexing
runner) happen lazily, after the arguments are parsed, so ``--help`` and usage errors work
without a .env file. Every failure message is fixed text plus an exception CLASS name at
most: exception text from pydantic, SQLAlchemy or the driver can echo credentials and is
never printed. Library logging is not configured here; its warnings reach the real stderr.

Known behaviour, deliberately not "fixed" here:
  * If request_indexing commits and the session then fails to close, the exception
    propagates. The document is recorded as an error, stays 'pending', and a later
    backfill picks it up.
  * Ctrl-C while a document is in progress leaves it locked ('indexing') until its lease
    expires; a rerun skips it until then.

Exit codes: 0 success, a dry run, or an execution without failures or blocked documents;
1 an operational error, or an execution with indexing failures or blocked documents;
2 a safety refusal, invalid arguments, a configuration that cannot be loaded, or a failed
embedding preflight; 130 interrupted.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy import func, select
from sqlalchemy.engine import Connection, Engine

from app.rag.backfill import BackfillPlan, build_plan
from app.rag.backfill_execute import ExecutionSummary, execute_plan
from app.rag.backfill_report import (
    DRY_RUN_MODE,
    EXECUTE_FOOTER,
    EXECUTE_MODE,
    format_progress_line,
    render_execution_summary,
    render_report,
)
from app.rag.backfill_safety import (
    SafetyFailure,
    TargetInfo,
    check_live_database,
    check_static_safety,
    default_repo_root,
    environment_is_explicit,
    inspect_filesystem,
)
from app.rag.backfill_wiring import make_reclassify, make_request, make_run, write_target_failure

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2
EXIT_INTERRUPTED = 130

# Fixed texts: no exception message, URL, path or name is ever part of them.
MESSAGE_NO_EXECUTION_RUNTIME = (
    "execution needs an embedder and a session factory, and this runtime has none"
)
NOTE_OFFLINE_SET_LATE = (
    "Note: the offline setting was applied after the model libraries were imported; "
    "offline loading relies on the pinned loader (local files only)."
)
NOTICE_CHECKING_MODEL = "Checking the embedding model (offline); this can take a while."
NOTICE_MODEL_OK = "Embedding model check passed."
INTERRUPTED_BEFORE_WRITES = "Interrupted; nothing was changed."
INTERRUPTED_AFTER_WRITES = (
    "Interrupted after indexing started; the database may have been changed. "
    "Run a dry run to see the current state."
)
ERROR_AFTER_WRITES = "the database may have been changed. Run a dry run to see the current state."


# ---------- arguments ----------

def _positive_int(value: str) -> int:
    message = "must be a whole number of at least 1"
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if number < 1:
        raise argparse.ArgumentTypeError(message)
    return number


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: "--exec" must never be accepted as "--execute".
    parser = argparse.ArgumentParser(
        prog="backfill_indexing.py",
        description=(
            "Report (dry run, the default) which documents the indexing backfill would index; "
            "with --execute, index them."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--database",
        metavar="NAME",
        type=_non_blank,
        help="expected database name; enables the database-name checks, including a live check",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="also consider indexing failures after a successful extraction",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=_positive_int,
        help="index at most N candidates (lowest document ids first); other documents are not deferred",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "index the documents listed in the plan, one at a time "
            "(requires --database; loads the embedding model offline)"
        ),
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute and not args.database:
        parser.error("--execute requires --database NAME")
    return args


# ---------- runtime ----------

@dataclass(frozen=True, slots=True)
class Runtime:
    """Everything taken from application configuration. The URL, engine, embedder and
    session factory are kept out of repr.

    ``embedder`` and ``session_factory`` are only needed by --execute; they default to None
    so a dry-run runtime does not have to provide them, and --execute refuses without them.
    """

    engine: Engine = field(repr=False)
    database_url: str = field(repr=False)
    environment: str | None
    environment_explicit: bool
    upload_dir: str
    upload_root: Path
    embedding_id: str
    embedder: Any = field(default=None, repr=False)
    session_factory: Callable[[], Any] | None = field(default=None, repr=False)


def load_runtime() -> Runtime:
    """Import application configuration and collect the runtime facts. May raise."""
    from app.core import storage
    from app.core.config import settings
    from app.core.database import SessionLocal, engine
    from app.rag.embedder import get_embedder

    # get_embedder() only constructs the Embedder; the model loads on first use, which
    # this function never triggers. The very same object is later preflighted and used to
    # index, and its embedding_id is the single source of the id format.
    embedder = get_embedder()
    return Runtime(
        engine=engine,
        database_url=settings.database_url,
        environment=settings.environment,
        environment_explicit=environment_is_explicit(settings),
        upload_dir=settings.upload_dir,
        upload_root=storage.UPLOAD_ROOT,
        embedding_id=embedder.embedding_id,
        embedder=embedder,
        session_factory=SessionLocal,
    )


# ---------- read-only queries ----------

def count_not_completed(conn: Connection) -> dict[str | None, int]:
    """Documents whose extraction is not completed, counted by status (NULL is a key)."""
    from app.models.document import Document

    statement = (
        select(Document.processing_status, func.count())
        .where(Document.processing_status.is_distinct_from("completed"))
        .group_by(Document.processing_status)
    )
    return {status: count for status, count in conn.execute(statement).all()}


# ---------- run state and exit codes ----------

@dataclass(slots=True)
class _RunState:
    """Tells the error handlers in main() whether the database may already have been written."""

    writes_started: bool = False


def exit_code_for(summary: ExecutionSummary) -> int:
    """130 if interrupted (even with failures), else 1 on failures or blocked documents, else 0."""
    if summary.interrupted:
        return EXIT_INTERRUPTED
    if summary.needs_attention:
        return EXIT_FAILURE
    return EXIT_OK


# ---------- output helpers ----------

def _ascii(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


def _print_refusal(failures: Sequence[SafetyFailure], target: TargetInfo | None, err: TextIO) -> None:
    for failure in failures:
        print(f"REFUSED: {failure.message}", file=err)
    if target is not None:
        print(f"Target: {target.describe()}", file=err)
    print("Nothing was changed.", file=err)


def _refuse_text(message: str, target: TargetInfo | None, err: TextIO) -> None:
    """A refusal with a ready-made fixed message (not a SafetyFailure)."""
    print(f"REFUSED: {message}", file=err)
    if target is not None:
        print(f"Target: {target.describe()}", file=err)
    print("Nothing was changed.", file=err)


def _connect_failure(args: argparse.Namespace, exc: Exception, err: TextIO) -> int:
    name = type(exc).__name__  # the exception text can contain connection details
    if args.database:
        print(f"REFUSED: the live database check failed ({name})", file=err)
        print("Nothing was changed.", file=err)
        return EXIT_REFUSED
    print(f"Could not connect to the database ({name}); nothing was changed.", file=err)
    return EXIT_FAILURE


# ---------- execution ----------

@dataclass(frozen=True, slots=True)
class _ExecutionTools:
    """The application functions that --execute injects into the wiring."""

    request_indexing: Callable[..., bool]
    run_indexing: Callable[..., Any]


def _prepare_execution(runtime: Runtime) -> tuple[_ExecutionTools | None, str | None]:
    """Check what --execute needs. Returns (tools, None) or (None, a fixed refusal text).

    Runs after the live database check and before anything is selected or written. The
    indexing modules are imported here, at call time, so dry runs and --help never load
    them, and so each function is looked up on its module when it is needed.
    """
    if runtime.embedder is None or runtime.session_factory is None:
        return None, MESSAGE_NO_EXECUTION_RUNTIME

    from app.rag import indexing_runner, indexing_service

    # Opens no connection: both factories must be bound to the engine that was just verified.
    mismatch = write_target_failure(
        runtime.engine, runtime.session_factory, indexing_runner.SessionLocal
    )
    if mismatch is not None:
        return None, mismatch
    return _ExecutionTools(indexing_service.request_indexing, indexing_runner.run_indexing), None


def _execute(
    plan: BackfillPlan,
    args: argparse.Namespace,
    runtime: Runtime,
    tools: _ExecutionTools,
    *,
    select_states: Callable[..., Any],
    attach_sources: Callable[..., Any],
    environ: MutableMapping[str, str],
    target_text: str | None,
    not_completed: dict[str | None, int],
    out: TextIO,
    err: TextIO,
    state: _RunState,
) -> int:
    """Print the plan, preflight the model, index the plan's documents, report, return the exit code."""
    # The plan comes first, so it is visible even if the model check fails. The footer is
    # printed later, after the preflight, when "indexing starts now" is true.
    print(
        render_report(
            plan,
            mode=EXECUTE_MODE,
            target=target_text,
            database_verified=args.database,
            include_failed=args.include_failed,
            not_completed=not_completed,
            footer="",
        ),
        file=out,
        flush=True,
    )

    embedder = runtime.embedder
    to_run = plan.to_run()
    if to_run:
        # Imported here: numpy is not needed for --help, usage errors or dry runs.
        from app.rag.backfill_preflight import PreflightError, run_preflight

        print(NOTICE_CHECKING_MODEL, file=err, flush=True)
        try:
            result = run_preflight(
                runtime.embedder,
                expected_embedding_id=runtime.embedding_id,
                environ=environ,
            )
        except PreflightError as exc:
            # The message is the stage name and at most an exception CLASS name.
            print(f"REFUSED: {_ascii(str(exc))}", file=err)
            print("Nothing was changed.", file=err)
            return EXIT_REFUSED
        print(NOTICE_MODEL_OK, file=err, flush=True)
        if not result.offline_set_before_imports:
            print(NOTE_OFFLINE_SET_LATE, file=err, flush=True)
        print(EXECUTE_FOOTER, file=out, flush=True)
        embedder = result.embedder  # the object the preflight validated, used for every document

    reclassify = make_reclassify(
        runtime.engine,
        configured_embedding_id=runtime.embedding_id,
        include_failed=args.include_failed,
        select_states=select_states,
        attach_sources=attach_sources,
    )
    request = make_request(runtime.session_factory, tools.request_indexing)
    run = make_run(tools.run_indexing, embedder)

    def progress(position: int, total: int, result: Any) -> None:
        print(format_progress_line(position, total, result), file=out, flush=True)

    # From here on the database may be written (only if there is anything to index).
    state.writes_started = bool(to_run)
    summary = execute_plan(plan, reclassify=reclassify, request=request, run=run, on_result=progress)

    code = exit_code_for(summary)
    try:
        print(render_execution_summary(summary), file=out, flush=True)
    except Exception as exc:  # noqa: BLE001 - the run is over; keep its exit code
        print(f"Could not print the execution summary ({_ascii(type(exc).__name__)}).", file=err)
    return code


# ---------- the run ----------

def _run(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[], Runtime],
    environ: MutableMapping[str, str],
    cwd: Path,
    repo_root: Path,
    out: TextIO,
    err: TextIO,
    state: _RunState,
) -> int:
    # 1. Application configuration (lazy; nothing connects to any database yet).
    try:
        runtime = runtime_loader()
        from app.rag.backfill_selection import select_document_states
        from app.rag.backfill_sources import attach_source_status
    except Exception as exc:  # noqa: BLE001 - only the class name is ever shown
        print(
            f"REFUSED: could not load the application configuration ({type(exc).__name__})",
            file=err,
        )
        print("Nothing was changed.", file=err)
        return EXIT_REFUSED

    # 2. Static safety checks. No connection exists before they pass.
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
        _print_refusal(report.failures, report.target, err)
        return EXIT_REFUSED
    target_text = report.target.describe() if report.target is not None else None

    # 3. A read-only connection, configured before its first statement.
    try:
        connection = runtime.engine.connect()
    except Exception as exc:  # noqa: BLE001
        return _connect_failure(args, exc, err)

    tools: _ExecutionTools | None = None
    with connection:
        connection = connection.execution_options(postgresql_readonly=True)

        # 4. The live check (only when --database was given; --execute requires it).
        if args.database:
            failure = check_live_database(connection, args.database)
            if failure is not None:
                _print_refusal([failure], report.target, err)
                return EXIT_REFUSED

        # 5. --execute: what execution needs, and the proof that the factories that will
        #    write are bound to the engine verified above. Still nothing selected or written.
        if args.execute:
            tools, refusal = _prepare_execution(runtime)
            if refusal is not None:
                _refuse_text(refusal, report.target, err)
                return EXIT_REFUSED

        # 6. Read. Everything below only reads, and the connection closes at the end of the block.
        states = attach_source_status(select_document_states(connection))
        not_completed = count_not_completed(connection)

    plan = build_plan(
        states,
        configured_embedding_id=runtime.embedding_id,
        include_failed=args.include_failed,
        limit=args.limit,
    )

    if args.execute:
        assert tools is not None  # _prepare_execution returned tools when it did not refuse
        return _execute(
            plan,
            args,
            runtime,
            tools,
            select_states=select_document_states,
            attach_sources=attach_source_status,
            environ=environ,
            target_text=target_text,
            not_completed=not_completed,
            out=out,
            err=err,
            state=state,
        )

    print(
        render_report(
            plan,
            mode=DRY_RUN_MODE,
            target=target_text,
            database_verified=args.database,
            include_failed=args.include_failed,
            not_completed=not_completed,
        ),
        file=out,
    )
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: Callable[[], Runtime] | None = None,
    environ: MutableMapping[str, str] | None = None,
    cwd: Path | None = None,
    repo_root: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the backfill CLI and return its exit code. The keyword arguments exist for tests."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    try:
        args = parse_arguments(argv)
    except SystemExit as exit_request:  # argparse: 0 for --help, 2 for a usage error
        code = exit_request.code
        if code is None:
            return EXIT_OK
        return code if isinstance(code, int) else EXIT_REFUSED

    state = _RunState()
    try:
        return _run(
            args,
            runtime_loader=runtime_loader if runtime_loader is not None else load_runtime,
            environ=os.environ if environ is None else environ,
            cwd=Path.cwd().resolve() if cwd is None else cwd,
            repo_root=default_repo_root() if repo_root is None else repo_root,
            out=out,
            err=err,
            state=state,
        )
    except KeyboardInterrupt:
        print(INTERRUPTED_AFTER_WRITES if state.writes_started else INTERRUPTED_BEFORE_WRITES, file=err)
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001
        outcome = ERROR_AFTER_WRITES if state.writes_started else "nothing was changed."
        print(f"Unexpected error ({_ascii(type(exc).__name__)}); {outcome}", file=err)
        return EXIT_FAILURE