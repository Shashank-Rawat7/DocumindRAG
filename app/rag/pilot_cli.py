"""Command line of the development-split validity pilot (E1.7.5). ``scripts/e17_run_pilot.py`` only makes
the project importable; all logic is here.

    venv\\Scripts\\python.exe scripts\\e17_run_pilot.py --database documind_db --owner-id N ^
        --evaluation-email-sha256 <64 hex> --out C:\\outside\\repo\\pilot-report.json [--adjudicate q12:u1 ...]

It connects the existing pieces: the dev-only input gate (``runner_inputs``), the verified candidate pool
(``pool`` and ``pool_db``), the real dense retrieval (``retrieval.retrieve_for_query``), the run engine
(``engine``) and the report (``report``). This is a VALIDITY PILOT, not the V1 baseline.

Safety, in this order; the first failure stops the run with exit code 2 and nothing is read or written:
 1. ``--out`` must be an absolute path outside the repository, in an existing folder, and must not exist
    (not even as a dangling link); it is checked again immediately before the file is created, and the
    file is created exclusively (never overwritten);
 2. the dev file must be named ``dev.jsonl`` (the other split's file is refused by name); nothing else of
    the dataset folder is read, and the test split is never opened;
 3. the same static safety checks as the backfill, snapshot and ingest tools: development environment,
    explicitly configured, loopback database host, expected database name, upload directory and working
    directory under the repository (``check_static_safety``);
 4. the dev inputs and the corpus load and verify (``load_dev_inputs``; the corpus manifest hash equals the
    one of the inputs); the embedder id equals the stored one;
 5. every database read happens in a READ ONLY, REPEATABLE READ transaction (``pool_db.read_only_session``),
    after ``SELECT current_database()`` equals ``--database`` and ``SHOW transaction_read_only`` is on.
Then: snapshot 1; the existing read-only corpus verification (the evaluation user is identified by
``--owner-id`` and the SHA-256 of its email, never by assumed document ids; every document must be owned by it,
extracted, indexed and chunk-verified); the six documents are mapped from their stored description and file
name and the pool is read with the real ``retrievable_filter``; the pool is checked against the Python reading
(parity), verified against the source texts, and the dense retrieval is run through the engine, which refuses any
chunk outside the pool. A second read-only transaction takes snapshot 2: the two must be identical, else no
report is written (exit 1). The report is written last. Exit codes: 0 report written, snapshots identical
and the oracle validity checks passed; 1 failure after reading started, a failed validity or snapshot check
(a report is still written when only the oracle validity failed); 2 refusal or invalid input.

The summary printed holds counts, hashes, metric values and paths only: no question text, evidence, credential,
owner id or database id.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from app.rag.backfill_safety import (
    check_static_safety,
    default_repo_root,
    inspect_filesystem,
)
from app.rag.evaluation import engine
from app.rag.evaluation import report as rep
from app.rag.evaluation import statistics as stats
from app.rag.evaluation.metrics import MAX_RANK
from app.rag.evaluation.parsing import manifest_hash
from app.rag.evaluation.pool import build_pool, map_documents
from app.rag.evaluation.pool_db import check_pool_parity, documents_from_rows
from app.rag.evaluation.runner_inputs import (
    DEV_FILE_NAME,
    DevInputs,
    check_dev_path,
    load_dev_inputs,
)
from app.rag.evaluation.schema import EvaluationDataError, Issue
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

DEFAULT_BOOTSTRAP_SEED = 20261114
_SUMMARY_METRICS = ("raw_hit_at_1_half_coverage", "raw_hit_at_5_half_coverage", "raw_unit_recall_at_5_half_coverage")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _adjudication(value: str) -> tuple[str, str]:
    question_id, separator, unit_id = value.partition(":")
    if not separator or not question_id.strip() or not unit_id.strip() or ":" in unit_id:
        raise argparse.ArgumentTypeError("must be QUESTION_ID:UNIT_ID")
    return question_id.strip(), unit_id.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e17_run_pilot.py",
        description="Run the development-split validity pilot (E1.7): read-only database, report outside the "
                    "repository. This is not the V1 baseline.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", required=True, type=_non_blank,
                        help="database name; must equal the name in the database URL and on the server")
    parser.add_argument("--owner-id", required=True, type=_positive_int, help="id of the dedicated evaluation user")
    parser.add_argument("--evaluation-email-sha256", required=True, type=_sha256_hex,
                        help="SHA-256 of the evaluation user's email (the email itself is never accepted)")
    parser.add_argument("--out", required=True, type=_non_blank,
                        help="report file to create (absolute, outside the repository, must not exist)")
    parser.add_argument("--dev-file", type=_non_blank, default=None,
                        help="the dev split file, named dev.jsonl (default: eval/dataset/dev.jsonl)")
    parser.add_argument("--corpus-dir", type=_non_blank, default=None,
                        help="corpus directory (default: eval/corpus in the repository)")
    parser.add_argument("--adjudicate", action="append", type=_adjudication, default=[], metavar="QUESTION:UNIT",
                        help="a unit a human adjudicated as unreachable (repeatable)")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--resamples", type=_positive_int, default=stats.DEFAULT_RESAMPLES)
    parser.add_argument("--random-seeds", type=_positive_int, default=1000,
                        help="number of random-baseline seeds, 0..N-1 (at least 1000)")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------- tools (injectable) ----------

@dataclass(frozen=True, slots=True)
class Tools:
    """What the run needs beyond the static runtime. Real implementations are in ``load_tools``."""

    embedding_id: str
    open_session: Callable[[], AbstractContextManager[Any]] = field(repr=False)
    load_embedder: Callable[[], Any] = field(repr=False)


def load_tools(runtime: Runtime) -> Tools:
    from app.core.config import settings
    from app.rag.evaluation.pool_db import read_only_session

    def load_embedder() -> Any:
        from app.rag.embedder import get_embedder

        return get_embedder()

    return Tools(
        embedding_id=f"{settings.embedding_model_id}@{settings.embedding_model_revision}",
        open_session=lambda: read_only_session(runtime.engine, upload_root=runtime.upload_root),
        load_embedder=load_embedder,
    )


def load_corpus_files(corpus_dir: Path) -> Any:
    from app.rag import corpus_ingest as ci

    return ci.load_corpus(corpus_dir / "manifest.json", corpus_dir)


def git_info(repo_root: Path) -> tuple[str | None, bool | None]:
    """The HEAD commit and whether the working tree is clean; ``(None, None)`` if git cannot answer."""
    try:
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=15, check=False)
        status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"], capture_output=True,
                                text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None, None
    commit = head.stdout.strip()
    if head.returncode != 0 or len(commit) != 40 or status.returncode != 0:
        return None, None
    return commit, status.stdout.strip() == ""


# ---------- output path ----------

def _check_report_path(raw: str, repo_root: Path, err: TextIO) -> Path | None:
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
    if os.path.lexists(path):  # a dangling link also counts as existing
        print("REFUSED: --out already exists; a report is never overwritten.", file=err)
        return None
    return _resolve(path)


def _print_issues(issues: Sequence[Issue], err: TextIO) -> None:
    for issue in issues:
        print(f"  {issue.code} at {issue.location}: {issue.message}", file=err)


def _target_text(target: Any) -> str:
    if target is None:
        return "?"
    port = f":{target.port}" if target.port is not None else ""
    return _ascii(f"{target.host or '?'}{port}/{target.database or '?'}")


# ---------- main ----------

def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: Callable[[], Runtime] | None = None,
    tools_loader: Callable[[Runtime], Tools] | None = None,
    inputs_loader: Callable[[Path, Path], DevInputs] | None = None,
    corpus_loader: Callable[[Path], Any] | None = None,
    git_reader: Callable[[Path], tuple[str | None, bool | None]] | None = None,
    code_identity: Callable[[], rep.CodeIdentity] | None = None,
    clock: Callable[[], datetime] | None = None,
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
            runtime_loader=runtime_loader or load_runtime,
            tools_loader=tools_loader or load_tools,
            inputs_loader=inputs_loader or (lambda dev, corpus: load_dev_inputs(dev, corpus)),
            corpus_loader=corpus_loader or load_corpus_files,
            git_reader=git_reader or git_info,
            code_identity=code_identity or rep.compute_code_identity,
            clock=clock or (lambda: datetime.now(timezone.utc)),  # noqa: UP017
            environ=os.environ if environ is None else environ,
            cwd=Path.cwd().resolve() if cwd is None else cwd,
            repo_root=default_repo_root() if repo_root is None else repo_root,
            out=out,
            err=err,
        )
    except KeyboardInterrupt:
        print("Interrupted. Nothing was written to the database and no report was written.", file=err)
        return 130
    except Exception as exc:  # noqa: BLE001 - only the class name is ever shown
        print(f"Unexpected error ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE


def _run(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[], Runtime],
    tools_loader: Callable[[Runtime], Tools],
    inputs_loader: Callable[[Path, Path], DevInputs],
    corpus_loader: Callable[[Path], Any],
    git_reader: Callable[[Path], tuple[str | None, bool | None]],
    code_identity: Callable[[], rep.CodeIdentity],
    clock: Callable[[], datetime],
    environ: Mapping[str, str],
    cwd: Path,
    repo_root: Path,
    out: TextIO,
    err: TextIO,
) -> int:
    # 1. the output path, before anything is read
    report_path = _check_report_path(args.out, repo_root, err)
    if report_path is None:
        return EXIT_REFUSED

    # 2. the dev file: only a file named dev.jsonl; the other split is never touched
    dev_raw = Path(args.dev_file) if args.dev_file else repo_root / "eval" / "dataset" / DEV_FILE_NAME
    try:
        dev_path = check_dev_path(dev_raw)
    except EvaluationDataError as exc:
        print("REFUSED: the dev file is not acceptable. Nothing was read.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_REFUSED
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else repo_root / "eval" / "corpus"

    # 3. runtime and the static safety checks
    try:
        runtime = runtime_loader()
        tools = tools_loader(runtime)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: could not load the application configuration ({type(exc).__name__})", file=err)
        return EXIT_REFUSED
    filesystem = inspect_filesystem(repo_root, runtime.upload_dir, runtime.upload_root)
    safety = check_static_safety(
        database_url=runtime.database_url, environment=runtime.environment,
        environment_explicit=runtime.environment_explicit, expected_database=args.database, cwd=cwd,
        repo_root=repo_root, upload_root=runtime.upload_root, filesystem=filesystem, environ=environ,
    )
    if not safety.ok:
        _print_failures(safety.failures, safety.target, err)
        return EXIT_REFUSED

    # 4. inputs, corpus, embedder (no database yet)
    try:
        inputs = inputs_loader(dev_path, corpus_dir)
    except EvaluationDataError as exc:
        print("REFUSED: the dev inputs did not pass the gate. Nothing was read from the database.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_REFUSED
    try:
        corpus = corpus_loader(corpus_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: the corpus did not load ({_ascii(type(exc).__name__)}).", file=err)
        for problem in getattr(exc, "problems", ()):
            print(f"  {_ascii(str(problem))}", file=err)
        return EXIT_REFUSED
    if manifest_hash(corpus.manifest) != inputs.manifest_sha256 or corpus.manifest.stage != "ingested":
        print("REFUSED: the corpus manifest is not the ingested manifest of the dev inputs.", file=err)
        return EXIT_REFUSED
    try:
        embedder = tools.load_embedder()
        embedder_id = embedder.embedding_id
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: the embedder could not be loaded ({_ascii(type(exc).__name__)}).", file=err)
        return EXIT_REFUSED
    if embedder_id != tools.embedding_id:
        print("REFUSED: the embedder id differs from the embedding id the corpus was indexed with.", file=err)
        return EXIT_REFUSED
    seeds = tuple(range(args.random_seeds)) if args.random_seeds >= 1000 else None
    if seeds is None:
        print("REFUSED: --random-seeds must be at least 1000.", file=err)
        return EXIT_REFUSED
    adjudicated: dict[str, list[str]] = {}
    for question_id, unit_id in args.adjudicate:
        adjudicated.setdefault(question_id, []).append(unit_id)

    print(f"target: {_target_text(safety.target)}", file=out)
    print(f"corpus: {len(corpus.files)} documents; dev: {len(inputs.answerable)} answerable, "
          f"{len(inputs.unanswerable)} unanswerable", file=out)

    # 5. the read-only run
    try:
        with tools.open_session() as session:
            problem = session.check_live(args.database)
            if problem is None:
                problem = session.assert_read_only()
            if problem:
                print(f"REFUSED: {_ascii(problem)}", file=err)
                return EXIT_REFUSED
            before = session.take_snapshot(clock)
            verification = session.verify_corpus(
                corpus.files, owner_id=args.owner_id, owner_sha256=args.evaluation_email_sha256,
                embedding_id=tools.embedding_id,
            )
            if not verification.ok:
                print("FAILED: the corpus documents did not verify (ownership, extraction, indexing or chunks).", file=err)
                for text in verification.problems:
                    print(f"  PROBLEM: {_ascii(str(text))}", file=err)
                for document in verification.documents:
                    for item in document.checks:
                        if not item.ok:
                            print(f"  {document.corpus_key}: {item.name}", file=err)
                return EXIT_FAILURE
            facts = documents_from_rows(session.document_rows(corpus.files))
            document_map = map_documents(
                inputs.manifest, facts.documents, owner_id=args.owner_id, embedding_id=tools.embedding_id,
                expected_filenames={f.corpus_key: f.filename for f in corpus.files},
            )
            chunks = session.pool_chunks(
                owner_id=args.owner_id, embedding_id=tools.embedding_id, document_ids=document_map.document_ids
            )
            mapped = tuple(d for d in facts.documents if d.document_id in document_map.key_by_id)
            check_pool_parity(mapped, facts.chunk_counts, tools.embedding_id, chunks)
            pool = build_pool(inputs.manifest, document_map, chunks, inputs.texts)
            retriever = session.dense_retriever(
                embedder, owner_id=args.owner_id, document_ids=document_map.document_ids, top_k=MAX_RANK
            )
            run = engine.run_pilot(
                inputs, pool, retriever, seeds=seeds, bootstrap_seed=args.bootstrap_seed,
                resamples=args.resamples, adjudicated_unreachable=adjudicated or None,
            )
    except EvaluationDataError as exc:
        print("FAILED: an integrity check failed. No report was written.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - the message of a database error can hold connection details
        print(f"FAILED: reading or running failed ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE

    # 6. the second snapshot, in a new read-only transaction
    try:
        from app.rag.evaluation.snapshot import compare_snapshots

        with tools.open_session() as session:
            problem = session.check_live(args.database) or session.assert_read_only()
            if problem:
                print(f"FAILED: {_ascii(problem)}. No report was written.", file=err)
                return EXIT_FAILURE
            after = session.take_snapshot(clock)
        comparison = compare_snapshots(before, after)
    except EvaluationDataError as exc:
        print("FAILED: the snapshots could not be compared. No report was written.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: the second snapshot failed ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE
    if not comparison.identical:
        print("FAILED: the database changed between the two snapshots (is the application running?). "
              "No report was written.", file=err)
        return EXIT_FAILURE

    # 7. metadata and report
    commit, clean = git_reader(repo_root)
    try:
        metadata = rep.RunMetadata(
            run_timestamp_utc=rep.utc_timestamp(clock()), python_version=rep.current_python_version(),
            embedder_id=embedder_id, embedder_signature=facts.signature, code_identity=code_identity(),
            git_commit=commit, working_tree_clean=clean,
            database_snapshots=rep.DatabaseSnapshotRecord(
                before["fingerprint"], after["fingerprint"], comparison.identical
            ),
        )
        rendered = rep.render_report(run, inputs, pool, metadata)
    except EvaluationDataError as exc:
        print("FAILED: the report could not be built. No report was written.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_FAILURE

    final_path = _check_report_path(args.out, repo_root, err)  # the folder may have changed meanwhile
    if final_path is None:
        return EXIT_REFUSED
    try:
        with open(final_path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered.json_bytes.decode("utf-8"))
        written = final_path.read_bytes()
    except OSError as exc:
        print(f"FAILED: the report file could not be written ({_ascii(type(exc).__name__)}).", file=err)
        return EXIT_FAILURE
    if written != rendered.json_bytes:
        print("FAILED: the written report differs from the rendered report.", file=err)
        return EXIT_FAILURE

    _print_summary(run, rendered, final_path, comparison, out)
    if not run.oracle_validity.ok:
        print("FAILED: the oracle validity checks did not pass (see the report). The report was written.", file=err)
        return EXIT_FAILURE
    return EXIT_OK


def _headline(report: Any, name: str) -> str:
    for result in report.overall.series:
        if result.name == name:
            return "n/a" if result.rate_or_mean is None else f"{result.rate_or_mean:.3f}"
    return "n/a"


def _print_summary(run: engine.PilotRun, rendered: rep.SerializedReport, path: Path, comparison: Any, out: TextIO) -> None:
    print(f"label: {rep.REPORT_LABEL}", file=out)
    print(f"report written: {path}", file=out)
    print(f"report bytes: {len(rendered.json_bytes)}  report sha256: {rendered.sha256}", file=out)
    print(f"manifest sha256: {run.manifest_sha256}", file=out)
    print(f"dev dataset sha256: {run.dev_dataset_sha256}", file=out)
    print(f"candidate pool sha256: {run.pool_sha256}  chunks: {run.pool_size}", file=out)
    print(f"questions: {len(run.questions)} answerable scored, {len(run.unanswerable)} unanswerable listed apart", file=out)
    print(f"random seeds: {len(run.seeds)}; bootstrap seed {run.dense_report.seed}; resamples {run.dense_report.resamples}",
          file=out)
    print(f"database snapshots identical: {'yes' if comparison.identical else 'no'}", file=out)
    validity = run.oracle_validity
    print(f"oracle validity ({validity.rule}): {'passed' if validity.ok else 'FAILED'}; "
          f"{sum(1 for c in validity.checks if c.unadjudicated)} question(s) with unadjudicated unreachable units; "
          f"{sum(1 for c in validity.checks if c.adjudicated)} adjudicated", file=out)
    for name in _SUMMARY_METRICS:
        print(f"  {name}: dense {_headline(run.dense_report, name)}, oracle {_headline(run.oracle_report, name)}, "
              f"random {_headline(run.random_report, name)}", file=out)
    print("dense-versus-random: reported in the file only; no pass/fail threshold is applied.", file=out)
