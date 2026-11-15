"""Command line of the retrieval-only dense baseline on the dev split (E2.4c gates, E2.4d two-pass run).

Preflight, in this order; the first failure stops with exit code 2 and nothing else is read or written:
 1. argparse; ``--expected-freeze-record-sha256`` is mandatory and must be exactly 64 lowercase hex characters;
 2. ``load_freeze_binding(repo_root)`` (reads the four frozen files only), then the SHA-256 of the raw
    ``freeze_record.json`` is compared with the expected value by ``hmac.compare_digest``. The expected value
    is supplied by the operator from outside the repository and is never stored in the code. Any failure
    prints one fixed message without issue details, paths or hashes;
 3. ``--out``: absolute, outside the repository, existing folder, does not exist;
 4. the fixed dev file ``eval/dataset/dev.jsonl`` (path check only; the other split is never named or opened);
 5. the tracked-only Git state: git's top level is the repository root, HEAD resolves and neither the index nor
    the tracked files differ from it. Untracked paths are never listed.

Run (E2.4d), only after the preflight: application configuration and the static safety checks of the pilot; the
dev inputs, the corpus and the embedder; then ONE read-only REPEATABLE READ transaction in which the corpus is
verified, the pool is built once and ``baseline_run.run_two_passes`` runs the complete engine twice (fresh
retriever and recorder per pass, same inputs, pool, seeds and snapshot); the D7 ranking hashes must be identical
or no report is written. A second read-only transaction takes the after-snapshot (must equal the first). Then,
immediately before rendering, the freeze binding is loaded again and must equal the first one and still match the
expected SHA-256; the report is rendered; immediately before writing, the Git state is read again (clean, same
HEAD) and the output path is checked again; the file is created exclusively and read back. A failed or partial
write never deletes the file.
Exit codes: 0 report written; 1 failure after reading started (a pass, the determinism gate, the snapshots, the
write, or an oracle validity failure after the report was written); 2 refusal or invalid input (the preflight,
and the final freeze, Git and output-path rechecks, all of which stop before anything is written); 130 interrupt.

Import boundary: this module imports only the standard library, ``freeze_binding`` and ``runner_inputs`` at the
top level. Every other module is imported lazily, after the anchor has passed (``_preflight`` imports ``pilot_cli``
and ``_heavy`` imports the run modules), so the anchor comparison runs before any application configuration,
database engine, runtime, embedder or retriever module is imported, and before the SQLAlchemy library itself.
"""
from __future__ import annotations

import argparse
import hmac
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

from app.rag.evaluation.freeze_binding import FreezeBinding, load_freeze_binding
from app.rag.evaluation.runner_inputs import DEV_FILE_NAME, check_dev_path
from app.rag.evaluation.schema import EvaluationDataError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2

_EXPECTED_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEAD = re.compile(r"[0-9a-f]{40}\Z")
_GIT_TIMEOUT_SECONDS = 30

GIT_CLEAN = "clean"
GIT_STAGED = "staged"
GIT_UNSTAGED = "unstaged"
GIT_UNKNOWN = "unknown"


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


# ---------- argument types (local, so that parsing needs no heavy import) ----------

def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value


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


def _lenient_sha256(value: str) -> str:
    lowered = value.strip().lower()
    if not _EXPECTED_SHA.match(lowered):
        raise argparse.ArgumentTypeError("must be 64 hexadecimal characters")
    return lowered


def _strict_sha256(value: str) -> str:
    """Exactly 64 lowercase hex characters: no case folding, no whitespace, no trailing newline."""
    if not _EXPECTED_SHA.match(value):
        raise argparse.ArgumentTypeError("must be exactly 64 lowercase hexadecimal characters")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e24_run_baseline.py",
        description="Run the retrieval-only dense baseline on the dev split (E2.4). Read-only database, report "
                    "outside the repository. Only the preflight gates exist in this checkpoint.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", required=True, type=_non_blank,
                        help="database name; must equal the name in the database URL and on the server")
    parser.add_argument("--owner-id", required=True, type=_positive_int, help="id of the dedicated evaluation user")
    parser.add_argument("--evaluation-email-sha256", required=True, type=_lenient_sha256,
                        help="SHA-256 of the evaluation user's email (the email itself is never accepted)")
    parser.add_argument("--expected-freeze-record-sha256", required=True, type=_strict_sha256,
                        help="SHA-256 of the raw eval/dataset/freeze_record.json, supplied from outside the "
                             "repository (exactly 64 lowercase hex characters)")
    parser.add_argument("--out", required=True, type=_non_blank,
                        help="report file to create (absolute, outside the repository, must not exist)")
    parser.add_argument("--adjudicate", action="append", type=_adjudication, default=[], metavar="QUESTION:UNIT",
                        help="a unit a human adjudicated as unreachable (repeatable)")
    # The defaults of the next three are resolved from the pilot's defaults when the run is implemented.
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--resamples", type=_positive_int, default=None)
    parser.add_argument("--random-seeds", type=_positive_int, default=None,
                        help="number of random-baseline seeds, 0..N-1 (at least 1000)")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------- tracked-only Git state ----------

def _git_environment() -> dict[str, str]:
    """The inherited environment without any ``GIT_*`` variable.

    ``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, ``GIT_COMMON_DIR`` and the like could redirect git to another
    repository or index, so none is passed on; git then finds the repository and its real index from ``-C``.
    Names are compared case-insensitively (Windows).
    """
    return {name: value for name, value in os.environ.items() if not name.upper().startswith("GIT_")}


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # No shell; every call names the repository explicitly with ``-C``, runs in a ``GIT_*``-free environment
    # and never lists untracked paths. ``core.fsmonitor=false`` keeps a repository-configured filesystem monitor
    # (an arbitrary program) from running during any check.
    return subprocess.run(
        ["git", "-C", str(repo_root), "--no-optional-locks", "-c", "core.fsmonitor=false", *arguments],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, check=False, shell=False,
        env=_git_environment(),
    )


def _same_directory(reported: str, repo_root: Path) -> bool:
    """True if git's reported top-level directory is ``repo_root`` once both are fully resolved (fail closed)."""
    if not reported:
        return False
    try:
        wanted = os.path.normcase(os.path.realpath(repo_root))
        return os.path.normcase(os.path.realpath(reported)) == wanted
    except (OSError, ValueError):
        return False


def tracked_git_state(repo_root: Path) -> tuple[str | None, str]:
    """``(HEAD commit, state)``; the state is one of the four ``GIT_*`` constants.

    ``clean`` means git's top-level directory is ``repo_root`` itself (not an enclosing repository), HEAD resolves
    and neither the index nor the tracked files differ from it. Untracked paths are ignored.
    Any other outcome of git (not a repository, no commit, an error, a timeout) is ``unknown``.
    """
    try:
        top = _git(repo_root, "rev-parse", "--show-toplevel")
        if top.returncode != 0 or not _same_directory(top.stdout.rstrip("\r\n"), repo_root):
            return None, GIT_UNKNOWN  # git answers for another (for example an enclosing) repository
        head = _git(repo_root, "rev-parse", "--verify", "HEAD")
        commit = head.stdout.strip()
        if head.returncode != 0 or not _HEAD.match(commit):
            return None, GIT_UNKNOWN
        staged = _git(repo_root, "diff", "--cached", "--quiet", "--no-ext-diff")
        unstaged = _git(repo_root, "diff", "--quiet", "--no-ext-diff")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, GIT_UNKNOWN
    if staged.returncode not in (0, 1) or unstaged.returncode not in (0, 1):
        return commit, GIT_UNKNOWN
    if staged.returncode == 1:
        return commit, GIT_STAGED
    if unstaged.returncode == 1:
        return commit, GIT_UNSTAGED
    return commit, GIT_CLEAN


# ---------- the run (E2.4d) ----------

@dataclass(frozen=True, slots=True)
class Preflight:
    """What the gates established; the argument of the run."""

    args: argparse.Namespace
    binding: FreezeBinding
    repo_root: Path
    report_path: Path
    dev_path: Path
    head_commit: str


@dataclass(frozen=True, slots=True)
class Dependencies:
    """Collaborators of the run. ``None`` means the real implementation, imported lazily. For tests."""

    freeze_loader: Callable[[Path], FreezeBinding]
    git_reader: Callable[[Path], tuple[str | None, str]]
    runtime_loader: Callable[[], Any] | None = None
    tools_loader: Callable[[Any], Any] | None = None
    inputs_loader: Callable[[Path, Path], Any] | None = None
    corpus_loader: Callable[[Path], Any] | None = None
    code_identity: Callable[[], Any] | None = None
    clock: Callable[[], datetime] | None = None
    environ: Mapping[str, str] | None = None
    cwd: Path | None = None


def _heavy() -> SimpleNamespace:
    """The modules of the run, imported only now (after the anchor)."""
    from app.rag import baseline_run, pilot_cli  # noqa: PLC0415 - deliberately lazy (import boundary)
    from app.rag.backfill_safety import check_static_safety, inspect_filesystem  # noqa: PLC0415
    from app.rag.evaluation import pool, pool_db, snapshot  # noqa: PLC0415
    from app.rag.evaluation import report as rep  # noqa: PLC0415
    from app.rag.evaluation import statistics as stats  # noqa: PLC0415
    from app.rag.evaluation.metrics import MAX_RANK  # noqa: PLC0415
    from app.rag.evaluation.parsing import manifest_hash  # noqa: PLC0415

    return SimpleNamespace(
        run=baseline_run, pc=pilot_cli, rep=rep, stats=stats, pool=pool, pool_db=pool_db, snapshot=snapshot,
        check_static_safety=check_static_safety, inspect_filesystem=inspect_filesystem, max_rank=MAX_RANK,
        manifest_hash=manifest_hash,
    )


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything that is loaded before the database is read."""

    runtime: Any
    tools: Any
    target: Any
    inputs: Any
    corpus: Any
    embedder: Any
    embedder_id: str
    seeds: tuple[int, ...]
    bootstrap_seed: int
    resamples: int
    adjudicated: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class Ran:
    """What the read-only run produced."""

    two: Any
    pool: Any
    facts: Any
    before: Any
    after: Any
    comparison: Any


def _execute(preflight: Preflight, out: TextIO, err: TextIO, deps: Dependencies) -> int:
    h = _heavy()
    prepared = _prepare(preflight, deps, h, out, err)
    if isinstance(prepared, int):
        return prepared
    ran = _read_and_run(preflight, prepared, deps, h, out, err)
    if isinstance(ran, int):
        return ran
    return _finish(preflight, prepared, ran, deps, h, out, err)


def _prepare(pre: Preflight, deps: Dependencies, h: SimpleNamespace, out: TextIO, err: TextIO) -> Prepared | int:
    pc, args = h.pc, pre.args
    try:
        runtime = (deps.runtime_loader or pc.load_runtime)()
        tools = (deps.tools_loader or pc.load_tools)(runtime)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: could not load the application configuration ({_ascii(type(exc).__name__)})", file=err)
        return EXIT_REFUSED
    filesystem = h.inspect_filesystem(pre.repo_root, runtime.upload_dir, runtime.upload_root)
    safety = h.check_static_safety(
        database_url=runtime.database_url, environment=runtime.environment,
        environment_explicit=runtime.environment_explicit, expected_database=args.database,
        cwd=Path.cwd().resolve() if deps.cwd is None else deps.cwd, repo_root=pre.repo_root,
        upload_root=runtime.upload_root, filesystem=filesystem,
        environ=os.environ if deps.environ is None else deps.environ,
    )
    if not safety.ok:
        pc._print_failures(safety.failures, safety.target, err)
        return EXIT_REFUSED
    corpus_dir = pre.repo_root / "eval" / "corpus"
    try:
        inputs = (deps.inputs_loader or (lambda dev, corpus: pc.load_dev_inputs(dev, corpus)))(pre.dev_path, corpus_dir)
    except EvaluationDataError as exc:
        print("REFUSED: the dev inputs did not pass the gate. Nothing was read from the database.", file=err)
        pc._print_issues(exc.issues, err)
        return EXIT_REFUSED
    try:
        corpus = (deps.corpus_loader or pc.load_corpus_files)(corpus_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: the corpus did not load ({_ascii(type(exc).__name__)}).", file=err)
        return EXIT_REFUSED
    if h.manifest_hash(corpus.manifest) != inputs.manifest_sha256 or corpus.manifest.stage != "ingested":
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
    random_seeds = 1000 if args.random_seeds is None else args.random_seeds
    if random_seeds < 1000:
        print("REFUSED: --random-seeds must be at least 1000.", file=err)
        return EXIT_REFUSED
    adjudicated: dict[str, list[str]] = {}
    for question_id, unit_id in args.adjudicate:
        adjudicated.setdefault(question_id, []).append(unit_id)
    print(f"target: {pc._target_text(safety.target)}", file=out)
    print(f"corpus: {len(corpus.files)} documents; dev: {len(inputs.answerable)} answerable, "
          f"{len(inputs.unanswerable)} unanswerable", file=out)
    return Prepared(
        runtime=runtime, tools=tools, target=safety.target, inputs=inputs, corpus=corpus, embedder=embedder,
        embedder_id=embedder_id, seeds=tuple(range(random_seeds)),
        bootstrap_seed=pc.DEFAULT_BOOTSTRAP_SEED if args.bootstrap_seed is None else args.bootstrap_seed,
        resamples=h.stats.DEFAULT_RESAMPLES if args.resamples is None else args.resamples, adjudicated=adjudicated,
    )


def _read_and_run(
    pre: Preflight, prep: Prepared, deps: Dependencies, h: SimpleNamespace, out: TextIO, err: TextIO,
) -> Ran | int:
    pc, args, tools = h.pc, pre.args, prep.tools
    clock = deps.clock or (lambda: datetime.now(timezone.utc))  # noqa: UP017 - Python 3.10
    try:
        # One read-only REPEATABLE READ transaction holds the verification, the pool and BOTH passes.
        with tools.open_session() as session:
            problem = session.check_live(args.database)
            if problem is None:
                problem = session.assert_read_only()
            if problem:
                print(f"REFUSED: {_ascii(problem)}", file=err)
                return EXIT_REFUSED
            before = session.take_snapshot(clock)
            verification = session.verify_corpus(
                prep.corpus.files, owner_id=args.owner_id, owner_sha256=args.evaluation_email_sha256,
                embedding_id=tools.embedding_id,
            )
            if not verification.ok:
                print("FAILED: the corpus documents did not verify (ownership, extraction, indexing or chunks).",
                      file=err)
                for text in verification.problems:
                    print(f"  PROBLEM: {_ascii(str(text))}", file=err)
                for document in verification.documents:
                    for item in document.checks:
                        if not item.ok:
                            print(f"  {document.corpus_key}: {item.name}", file=err)
                return EXIT_FAILURE
            facts = h.pool_db.documents_from_rows(session.document_rows(prep.corpus.files))
            document_map = h.pool.map_documents(
                prep.inputs.manifest, facts.documents, owner_id=args.owner_id, embedding_id=tools.embedding_id,
                expected_filenames={f.corpus_key: f.filename for f in prep.corpus.files},
            )
            chunks = session.pool_chunks(
                owner_id=args.owner_id, embedding_id=tools.embedding_id, document_ids=document_map.document_ids
            )
            mapped = tuple(d for d in facts.documents if d.document_id in document_map.key_by_id)
            h.pool_db.check_pool_parity(mapped, facts.chunk_counts, tools.embedding_id, chunks)
            pool = h.pool.build_pool(prep.inputs.manifest, document_map, chunks, prep.inputs.texts)
            two = h.run.run_two_passes(
                lambda: session.dense_retriever(
                    prep.embedder, owner_id=args.owner_id, document_ids=document_map.document_ids,
                    top_k=h.max_rank,
                ),
                prep.inputs, pool, seeds=prep.seeds, bootstrap_seed=prep.bootstrap_seed, resamples=prep.resamples,
                adjudicated=prep.adjudicated,
            )
    except EvaluationDataError as exc:
        print("FAILED: an integrity or determinism check failed. No report was written.", file=err)
        pc._print_issues(exc.issues, err)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - the message of a database error can hold connection details
        print(f"FAILED: reading or running failed ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE

    # the after-snapshot, in a new read-only transaction
    try:
        with tools.open_session() as session:
            problem = session.check_live(args.database) or session.assert_read_only()
            if problem:
                print(f"FAILED: {_ascii(problem)}. No report was written.", file=err)
                return EXIT_FAILURE
            after = session.take_snapshot(clock)
        comparison = h.snapshot.compare_snapshots(before, after)
    except EvaluationDataError as exc:
        print("FAILED: the snapshots could not be compared. No report was written.", file=err)
        pc._print_issues(exc.issues, err)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: the second snapshot failed ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE
    if not comparison.identical:
        print("FAILED: the database changed between the two snapshots (is the application running?). "
              "No report was written.", file=err)
        return EXIT_FAILURE
    return Ran(two=two, pool=pool, facts=facts, before=before, after=after, comparison=comparison)


def _finish(
    pre: Preflight, prep: Prepared, ran: Ran, deps: Dependencies, h: SimpleNamespace, out: TextIO, err: TextIO,
) -> int:
    pc, rep, args = h.pc, h.rep, pre.args
    # 1. the freeze binding again, immediately before rendering: the same files, the same bytes, the same anchor
    try:
        final_binding = deps.freeze_loader(pre.repo_root)
    except Exception:  # noqa: BLE001 - fixed text only
        print("REFUSED: the frozen dataset record could not be verified again. No report was written.", file=err)
        return EXIT_REFUSED
    if final_binding != pre.binding or not _anchor_matches(final_binding, args.expected_freeze_record_sha256):
        print("REFUSED: the frozen dataset record changed during the run. No report was written.", file=err)
        return EXIT_REFUSED

    # 2. render (pure); the first pass is the reported run, the determinism section is the verified D7 hash
    clock = deps.clock or (lambda: datetime.now(timezone.utc))  # noqa: UP017 - Python 3.10
    two, run = ran.two, ran.two.first.run
    try:
        metadata = rep.RunMetadata(
            run_timestamp_utc=rep.utc_timestamp(clock()), python_version=rep.current_python_version(),
            embedder_id=prep.embedder_id, embedder_signature=ran.facts.signature,
            code_identity=(deps.code_identity or rep.compute_code_identity)(),
            git_commit=pre.head_commit, working_tree_clean=True,
            database_snapshots=rep.DatabaseSnapshotRecord(
                ran.before["fingerprint"], ran.after["fingerprint"], ran.comparison.identical
            ),
        )
        rendered = rep.render_report(
            run, prep.inputs, ran.pool, metadata, kind=rep.BASELINE_KIND, freeze=final_binding,
            score_diagnostics=two.score_diagnostics,
            ranking_determinism=rep.ranking_determinism_section(two.ranking_sha256),
        )
    except EvaluationDataError as exc:
        print("FAILED: the report could not be built. No report was written.", file=err)
        pc._print_issues(exc.issues, err)
        return EXIT_FAILURE

    # 3. the Git state again, immediately before writing: same HEAD, clean tracked tree
    commit, state = deps.git_reader(pre.repo_root)
    if state != GIT_CLEAN or commit != pre.head_commit:
        print("REFUSED: the tracked Git state changed or could not be determined during the run. "
              "No report was written.", file=err)
        return EXIT_REFUSED

    # 4. the output path again (the folder may have changed meanwhile), then exclusive create and read-back
    final_path = pc._check_report_path(args.out, pre.repo_root, err)
    if final_path is None:
        return EXIT_REFUSED
    code = _write_report(final_path, rendered.json_bytes, err)
    if code != EXIT_OK:
        return code
    _print_summary(two, run, rendered, final_path, ran.comparison, out, rep)
    if not run.oracle_validity.ok:
        print("FAILED: the oracle validity checks did not pass (see the report). The report was written.", file=err)
        return EXIT_FAILURE
    return EXIT_OK


def _write_report(path: Path, data: bytes, err: TextIO) -> int:
    """Create ``path`` exclusively, write, read back. A file that exists afterwards is never deleted."""
    shown = _ascii(str(path))
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(data.decode("utf-8"))
    except FileExistsError:
        print("REFUSED: the output file appeared during the run; it was not touched and nothing was written.",
              file=err)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print(f"Interrupted while writing. A partial report may exist at {shown}; it was left for inspection.",
              file=err)
        return 130
    except OSError as exc:
        print(f"FAILED: the report file could not be written completely ({_ascii(type(exc).__name__)}). A partial "
              f"file may exist at {shown}; it was left in place for inspection and nothing was deleted.", file=err)
        return EXIT_FAILURE
    try:
        written = path.read_bytes()
    except OSError as exc:
        print(f"FAILED: the report file could not be read back ({_ascii(type(exc).__name__)}). It was left in "
              f"place at {shown} for inspection.", file=err)
        return EXIT_FAILURE
    if written != data:
        print(f"FAILED: the written report differs from the rendered report. It was left in place at {shown} "
              "for inspection and nothing was deleted.", file=err)
        return EXIT_FAILURE
    return EXIT_OK


def _headline(report: Any, name: str) -> str:
    for result in report.overall.series:
        if result.name == name:
            return "n/a" if result.rate_or_mean is None else f"{result.rate_or_mean:.3f}"
    return "n/a"


def _print_summary(two: Any, run: Any, rendered: Any, path: Path, comparison: Any, out: TextIO, rep: Any) -> None:
    """Counts, hashes, metric values and the path only: no question text, evidence, credential or id."""
    print(f"label: {rep.BASELINE_LABEL}", file=out)
    print(f"report written: {path}", file=out)
    print(f"report bytes: {len(rendered.json_bytes)}  report sha256: {rendered.sha256}", file=out)
    print(f"manifest sha256: {run.manifest_sha256}", file=out)
    print(f"dev dataset sha256: {run.dev_dataset_sha256}", file=out)
    print(f"candidate pool sha256: {run.pool_sha256}  chunks: {run.pool_size}", file=out)
    print(f"questions: {len(run.questions)} answerable scored, {len(run.unanswerable)} unanswerable scored apart "
          "(scores only)", file=out)
    print(f"ranking determinism: {rep.DETERMINISM_PASSES} passes identical, ranking sha256 {two.ranking_sha256}",
          file=out)
    identical = two.score_diagnostics["scores_identical_across_passes"]
    print(f"score digests identical across passes: {'yes' if identical else 'no'} (diagnostic only)", file=out)
    print(f"database snapshots identical: {'yes' if comparison.identical else 'no'}", file=out)
    for name in ("raw_hit_at_1_half_coverage", "raw_hit_at_5_half_coverage", "raw_unit_recall_at_5_half_coverage"):
        print(f"  {name}: dense {_headline(run.dense_report, name)}", file=out)


# ---------- main ----------

def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_issues(issues: Sequence[Any], err: TextIO) -> None:
    for issue in issues:
        print(f"  {issue.code} at {issue.location}: {issue.message}", file=err)


def main(
    argv: Sequence[str] | None = None,
    *,
    freeze_loader: Callable[[Path], FreezeBinding] | None = None,
    git_reader: Callable[[Path], tuple[str | None, str]] | None = None,
    executor: Callable[[Preflight, TextIO, TextIO], int] | None = None,
    runtime_loader: Callable[[], Any] | None = None,
    tools_loader: Callable[[Any], Any] | None = None,
    inputs_loader: Callable[[Path, Path], Any] | None = None,
    corpus_loader: Callable[[Path], Any] | None = None,
    code_identity: Callable[[], Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
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
    root = default_repo_root() if repo_root is None else repo_root
    freeze = freeze_loader or load_freeze_binding
    git = git_reader or tracked_git_state
    try:
        preflight = _preflight(args, root, freeze, git, err)
    except KeyboardInterrupt:
        print("Interrupted. Nothing was written.", file=err)
        return 130
    except Exception as exc:  # noqa: BLE001 - only the class name is ever shown
        print(f"REFUSED: the preflight failed unexpectedly ({_ascii(type(exc).__name__)}). Nothing was read "
              "from the database and no report was written.", file=err)
        return EXIT_REFUSED
    if isinstance(preflight, int):
        return preflight
    try:
        if executor is not None:
            return executor(preflight, out, err)
        deps = Dependencies(
            freeze_loader=freeze, git_reader=git, runtime_loader=runtime_loader, tools_loader=tools_loader,
            inputs_loader=inputs_loader, corpus_loader=corpus_loader, code_identity=code_identity, clock=clock,
            environ=environ, cwd=cwd,
        )
        return _execute(preflight, out, err, deps)
    except KeyboardInterrupt:
        print("Interrupted. Nothing was written to the database and no report was written.", file=err)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error ({_ascii(type(exc).__name__)}). No report was written.", file=err)
        return EXIT_FAILURE


def _anchor_matches(binding: FreezeBinding, expected: str) -> bool:
    try:
        return hmac.compare_digest(binding.record_sha256.encode("ascii"), expected.encode("ascii"))
    except (UnicodeError, AttributeError, TypeError):
        return False


def _preflight(
    args: argparse.Namespace,
    repo_root: Path,
    freeze_loader: Callable[[Path], FreezeBinding],
    git_reader: Callable[[Path], tuple[str | None, str]],
    err: TextIO,
) -> Preflight | int:
    # 1. the external trust anchor, before anything else is imported or read beyond the four frozen files
    try:
        binding = freeze_loader(repo_root)
    except Exception:  # noqa: BLE001 - any failure; KeyboardInterrupt is not caught here
        # Fixed text only: no exception class, message, issue code or location, path or hash.
        print("REFUSED: the frozen dataset record could not be verified. Nothing else was read.", file=err)
        return EXIT_REFUSED
    if not _anchor_matches(binding, args.expected_freeze_record_sha256):
        print("REFUSED: the freeze record does not match --expected-freeze-record-sha256. Nothing else was read.",
              file=err)
        return EXIT_REFUSED

    # 2. the output path (``pilot_cli`` is imported only now: it pulls in the SQLAlchemy library)
    from app.rag.pilot_cli import _check_report_path  # noqa: PLC0415 - deliberately lazy (import boundary)

    report_path = _check_report_path(args.out, repo_root, err)
    if report_path is None:
        return EXIT_REFUSED

    # 3. the fixed dev file: a path check only
    try:
        dev_path = check_dev_path(repo_root / "eval" / "dataset" / DEV_FILE_NAME)
    except EvaluationDataError as exc:
        print("REFUSED: the dev file is not acceptable. Nothing was read.", file=err)
        _print_issues(exc.issues, err)
        return EXIT_REFUSED

    # 4. committed HEAD and a clean tracked tree (untracked paths are ignored)
    commit, state = git_reader(repo_root)
    if state != GIT_CLEAN or commit is None:
        reason = {
            GIT_STAGED: "tracked changes are staged",
            GIT_UNSTAGED: "tracked files have unstaged changes",
        }.get(state, "the Git state could not be determined")
        print(f"REFUSED: the run needs a committed HEAD and a clean tracked tree ({reason}).", file=err)
        return EXIT_REFUSED
    return Preflight(args=args, binding=binding, repo_root=repo_root, report_path=report_path, dev_path=dev_path,
                     head_commit=commit)


if __name__ == "__main__":  # pragma: no cover - the wrapper script comes in a later checkpoint
    raise SystemExit(main())
