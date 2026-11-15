"""Command line for the E1.5 corpus ingestion driver (``scripts/e15_ingest_corpus.py`` only makes the
project importable).

    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py plan      --database documind_db --owner-id N --evaluation-email-sha256 <64 hex>
    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py ingest    <same>                      (read-only dry run)
    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py ingest    <same> --execute --authorization ingest-<database>-owner-<N>
    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py verify    <same>
    venv\\Scripts\\python.exe scripts\\e15_ingest_corpus.py candidate <same> --processing-ref <40 hex> --out C:\\outside\\repo\\manifest.candidate.json

Commands
  plan       read-only: what ``ingest --execute`` would do for each of the six documents.
  ingest     without --execute it is the same read-only plan. With --execute it creates the missing
             documents (store file -> create row -> extract text), one at a time, and stops at the
             first problem. --execute also requires --authorization with the exact phrase above.
  verify     read-only: ingestion, extraction and indexing checks for every document.
  candidate  runs ``verify``; when everything passes, writes a candidate manifest (stage ``ingested``)
             to --out (absolute, outside the repository, must not exist). It never touches
             eval/corpus/manifest.json: publishing the candidate is a separate, manual step.

There is no indexing command. Index the documents with the unmodified scripts/backfill_indexing.py.

Every command first passes the same static safety checks as the backfill and snapshot tools
(development environment, loopback database host, database name equals --database, upload
directory resolved from the repository root, run from the repository root), then the live
``current_database()`` check. ``ingest --execute`` also proves that the session factories used to
create rows and to extract text, and ``storage.UPLOAD_ROOT``, point at the verified target.

Exit codes: 0 success (or nothing to do); 1 a blocking state, a failed check or a failure while
running; 2 refusal, invalid input, or another run holds the lock.
Nothing printed contains a password, a connection URL, an email, extracted text or an embedding.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from app.rag import corpus_ingest as ci
from app.rag.backfill_safety import (
    check_static_safety,
    default_repo_root,
    inspect_filesystem,
)
from app.rag.snapshot_cli import (
    Runtime,
    _ascii,
    _inside,
    _non_blank,
    _print_failures,
    _resolve,
    _sha256_hex,
    load_runtime,
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2

_COMMIT_HEX = 40


# ---------- arguments ----------

def _commit_id(value: str) -> str:
    lowered = value.strip().lower()
    if len(lowered) != _COMMIT_HEX or any(c not in "0123456789abcdef" for c in lowered):
        raise argparse.ArgumentTypeError("must be a 40-character lowercase hexadecimal git commit id")
    return lowered


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def authorization_phrase(database: str, owner_id: int) -> str:
    return f"ingest-{database}-owner-{owner_id}"


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: "--exec" must never be accepted as "--execute".
    parser = argparse.ArgumentParser(
        prog="e15_ingest_corpus.py",
        description="Ingest and verify the six E1.5 evaluation corpus documents. Read-only unless "
                    "'ingest --execute' is given with its authorization.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--database", required=True, type=_non_blank,
                         help="database name; must equal the name in the database URL and on the server")
        sub.add_argument("--owner-id", required=True, type=_positive_int,
                         help="id of the dedicated evaluation user")
        sub.add_argument("--evaluation-email-sha256", required=True, type=_sha256_hex,
                         help="SHA-256 of the evaluation user's email (the email itself is never accepted)")
        sub.add_argument("--manifest", type=_non_blank, default=None,
                         help="corpus manifest (default: eval/corpus/manifest.json in the repository)")
        sub.add_argument("--corpus-dir", type=_non_blank, default=None,
                         help="corpus directory (default: eval/corpus in the repository)")

    common(commands.add_parser("plan", help="read-only: show what ingest --execute would do", allow_abbrev=False))

    ingest = commands.add_parser("ingest", help="plan (default) or create the missing documents", allow_abbrev=False)
    common(ingest)
    ingest.add_argument("--execute", action="store_true", help="really create the documents (writes!)")
    ingest.add_argument("--authorization", default=None,
                        help="required with --execute: ingest-<database>-owner-<owner id>")

    common(commands.add_parser("verify", help="read-only: verify all six documents", allow_abbrev=False))

    candidate = commands.add_parser("candidate", help="verify, then write a candidate manifest", allow_abbrev=False)
    common(candidate)
    candidate.add_argument("--processing-ref", required=True, type=_commit_id,
                           help="40-character git commit id of the code that did the extraction")
    candidate.add_argument("--out", required=True, type=_non_blank,
                           help="candidate manifest to create (absolute, outside the repository, must not exist)")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.command == "ingest" and not args.execute and args.authorization is not None:
        build_parser().error("--authorization is only meaningful with --execute")
    return args


# ---------- tools ----------

@dataclass(frozen=True, slots=True)
class Tools:
    """What the commands need beyond the static runtime: the session factory and the embedding id."""

    session_factory: Callable[[], Any] = field(repr=False)
    embedding_id: str


def expected_embedding_id() -> str:
    """The value the indexer stores in documents.indexed_embedding_id: ``<model id>@<revision>``.

    It is built from the same two settings as ``Embedder.embedding_id`` (a test asserts they agree);
    no model is loaded.
    """
    from app.core.config import settings

    return f"{settings.embedding_model_id}@{settings.embedding_model_revision}"


def load_tools() -> Tools:
    from app.core.database import SessionLocal

    return Tools(session_factory=SessionLocal, embedding_id=expected_embedding_id())


# ---------- output ----------

def _print_plan(plan: ci.Plan, out: TextIO) -> None:
    for doc in plan.documents:
        ident = f" id={doc.document_id}" if doc.document_id is not None else ""
        print(f"  {doc.corpus_key}: {doc.code} -> {doc.action}{ident} | {_ascii(doc.detail)}", file=out)
    for problem in plan.problems:
        print(f"  PROBLEM: {_ascii(problem)}", file=out)
    for orphan in plan.orphans:
        print(f"  PROBLEM: stored file {_ascii(orphan)} has a corpus hash and no document row (orphan; manual recovery)", file=out)
    creates = len(plan.to_create)
    print(f"summary: {creates} to create, "
          f"{sum(1 for d in plan.documents if d.action == ci.ACTION_SKIP)} already extracted, "
          f"{sum(1 for d in plan.documents if d.action == ci.ACTION_REFUSE)} blocking, "
          f"{len(plan.problems)} general problem(s), {len(plan.orphans)} orphan file(s)", file=out)


def _print_verify(report: ci.VerifyReport, out: TextIO) -> None:
    for problem in report.problems:
        print(f"  PROBLEM: {_ascii(problem)}", file=out)
    for doc in report.documents:
        ident = f" id={doc.document_id}" if doc.document_id is not None else ""
        print(f"  {doc.corpus_key}{ident}: {'OK' if doc.ok else 'FAILED'}", file=out)
        for item in doc.checks:
            if not item.ok or item.name == "pdf_page_numbers_reported":
                print(f"    [{'ok' if item.ok else 'FAIL'}] {item.name}: {_ascii(item.detail)}", file=out)
    print(f"verification: {'PASSED' if report.ok else 'FAILED'} "
          f"({sum(1 for d in report.documents if d.ok)}/{len(report.documents)} documents)", file=out)


# ---------- main ----------

def _check_candidate_path(raw: str, repo_root: Path, manifest_path: Path, err: TextIO) -> Path | None:
    path = Path(raw)
    if not path.is_absolute():
        print("REFUSED: --out must be an absolute path.", file=err)
        return None
    if _inside(path, repo_root):
        print("REFUSED: --out must be outside the repository; the candidate is never written over the manifest.", file=err)
        return None
    resolved = _resolve(path)
    if resolved is None or resolved == _resolve(manifest_path):
        print("REFUSED: --out must not be the manifest.", file=err)
        return None
    if not path.parent.is_dir():
        print("REFUSED: the folder of --out does not exist.", file=err)
        return None
    if os.path.lexists(path):
        print("REFUSED: --out already exists; a candidate is never overwritten.", file=err)
        return None
    return resolved


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: Callable[[], Runtime] | None = None,
    tools_loader: Callable[[], Tools] | None = None,
    environ: MutableMapping[str, str] | None = None,
    cwd: Path | None = None,
    repo_root: Path | None = None,
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
        return _run(
            args,
            runtime_loader=runtime_loader if runtime_loader is not None else load_runtime,
            tools_loader=tools_loader if tools_loader is not None else load_tools,
            environ=os.environ if environ is None else environ,
            cwd=Path.cwd().resolve() if cwd is None else cwd,
            repo_root=default_repo_root() if repo_root is None else repo_root,
            out=out,
            err=err,
        )
    except KeyboardInterrupt:
        print("Interrupted. Re-run 'plan' to see the current state; nothing is retried automatically.", file=err)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error ({_ascii(type(exc).__name__)}). Run 'plan' to see the current state.", file=err)
        return EXIT_FAILURE


def _run(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[], Runtime],
    tools_loader: Callable[[], Tools],
    environ: Mapping[str, str],
    cwd: Path,
    repo_root: Path,
    out: TextIO,
    err: TextIO,
) -> int:
    execute = args.command == "ingest" and args.execute
    if execute and args.authorization != authorization_phrase(args.database, args.owner_id):
        print("REFUSED: --execute requires --authorization "
              f"{authorization_phrase(args.database, args.owner_id)!r}. Nothing was read or written.", file=err)
        return EXIT_REFUSED

    manifest_path = Path(args.manifest) if args.manifest else repo_root / "eval" / "corpus" / "manifest.json"
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else repo_root / "eval" / "corpus"
    candidate_path = None
    if args.command == "candidate":
        candidate_path = _check_candidate_path(args.out, repo_root, manifest_path, err)
        if candidate_path is None:
            return EXIT_REFUSED

    try:
        runtime = runtime_loader()
        tools = tools_loader()
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
        corpus = ci.load_corpus(manifest_path, corpus_dir)
    except ci.CorpusError as exc:
        print("REFUSED: the corpus did not load. Nothing was read from or written to the database.", file=err)
        for problem in exc.problems:
            print(f"  {_ascii(problem)}", file=err)
        return EXIT_REFUSED
    if args.command in ("plan", "ingest", "candidate") and corpus.manifest.stage != "draft":
        print(f"REFUSED: the manifest stage is {corpus.manifest.stage!r}; this command needs a 'draft' manifest.", file=err)
        return EXIT_REFUSED

    target = _ascii(report.target.describe()) if report.target else "?"
    print(f"command: {args.command}{' --execute' if execute else ''}", file=out)
    print(f"target: {target}", file=out)
    print(f"corpus: {len(corpus.files)} documents, manifest stage {corpus.manifest.stage}", file=out)

    engine = runtime.engine
    common = dict(
        owner_id=args.owner_id, owner_sha256=args.evaluation_email_sha256, upload_root=runtime.upload_root
    )

    if args.command in ("plan", "ingest"):
        result = ci.run_ingest(
            engine=engine, session_factory=tools.session_factory, files=corpus.files,
            expected_database=args.database, execute=execute, **common,
        )
        return _report_ingest(result, execute=execute, out=out, err=err)

    # verify and candidate: read-only.
    failure = ci.live_database_failure(engine, args.database)
    if failure:
        print(f"REFUSED: {_ascii(failure)}", file=err)
        return EXIT_REFUSED
    with ci.read_only(engine) as conn:
        verification = ci.verify(conn, corpus.files, embedding_id=tools.embedding_id, **common)
    _print_verify(verification, out)
    if not verification.ok:
        print("Not generating a candidate manifest: verification did not pass.", file=err)
        return EXIT_FAILURE
    if args.command == "verify":
        return EXIT_OK
    return _write_candidate(corpus, args.processing_ref, candidate_path, repo_root, manifest_path, out, err)


def _report_ingest(result: ci.IngestResult, *, execute: bool, out: TextIO, err: TextIO) -> int:
    if result.plan is not None:
        _print_plan(result.plan, out)
    for item in result.documents:
        ident = f" id={item.document_id}" if item.document_id is not None else ""
        print(f"  result {item.corpus_key}: {item.status}{ident} | {_ascii(item.detail)}", file=out)

    if result.outcome == "refused":
        print("REFUSED: nothing was written.", file=err)
        for problem in result.problems:
            print(f"  {_ascii(problem)}", file=err)
        if result.plan is not None:
            print("  A blocking state was found (see above). Recovery is manual; nothing is deleted, "
                  "overwritten or retried.", file=err)
        return EXIT_REFUSED if result.plan is None else EXIT_FAILURE
    if result.outcome == "locked":
        print("REFUSED: another run holds the ingestion lock. Nothing was written.", file=err)
        return EXIT_REFUSED
    if result.outcome == "stopped":
        print("STOPPED: a document failed. Later documents were not attempted. Run 'plan' to see the state; "
              "recovery is manual and nothing is retried.", file=err)
        return EXIT_FAILURE
    if result.outcome == "dry_run":
        print("dry run: nothing was written. Use 'ingest --execute --authorization ...' to create the documents.", file=out)
    elif result.outcome == "nothing_to_do":
        print("nothing to do: every document is already extracted. Nothing was written.", file=out)
    else:
        print("ingested: documents created and extracted. They are NOT indexed; use the backfill CLI.", file=out)
    return EXIT_OK


def _write_candidate(
    corpus: ci.Corpus, processing_ref: str, path: Path, repo_root: Path, manifest_path: Path,
    out: TextIO, err: TextIO,
) -> int:
    import pypdf

    text = ci.build_candidate(
        corpus.manifest_text,
        corpus.files,
        processing_ref=processing_ref,
        python_version=platform.python_version(),
        library_versions={"pypdf": pypdf.__version__},
    )
    problems = ci.check_candidate(corpus.manifest_text, text)
    if problems:
        print("FAILED: the candidate manifest did not validate. Nothing was written.", file=err)
        for problem in problems:
            print(f"  {_ascii(problem)}", file=err)
        return EXIT_FAILURE
    # The folder may have changed while the database was read: check again, immediately before creating.
    again = _check_candidate_path(str(path), repo_root, manifest_path, err)
    if again is None:
        return EXIT_REFUSED
    try:
        with open(again, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as exc:
        print(f"FAILED: the candidate could not be written ({type(exc).__name__}).", file=err)
        return EXIT_FAILURE
    print(f"candidate manifest written: {again}", file=out)
    print(f"candidate file sha256: {ci.sha256_bytes(text.encode('utf-8'))}", file=out)
    print("It is NOT published. Review it, take S1 and compare it with S0, then publish it manually.", file=out)
    return EXIT_OK
