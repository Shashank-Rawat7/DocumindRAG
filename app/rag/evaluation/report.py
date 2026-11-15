"""Deterministic report and run metadata of the development-split validity pilot (E1.7.4).

Pure module: it reads no database and no dataset file, runs no retrieval and never touches the test
split. ``build_report`` turns a ``PilotRun`` (E1.7.3) plus the run metadata into one JSON-ready document;
``serialize_report`` writes it as canonical UTF-8 JSON (sorted keys, no whitespace, no NaN or infinity,
floats exactly as Python prints them); ``render_report`` does both and adds the SHA-256 of the bytes. The
same run and metadata always give the same bytes. ``compute_code_identity`` is the only function that
reads files: the source of the evaluation, dense-retrieval and extraction modules.

Label. Every report says ``"Validity pilot — not the V1 baseline"`` and ``is_v1_baseline: false``. The
V1 baseline is a later, separate, frozen run (E2).

Contents. Hashes of the manifest, the dev questions and the candidate pool; the run time and Python
version; the code identity; the embedder id and signature; the content fingerprints of the
read-only database snapshots taken before and after the run (when supplied); cutoffs, relevance rules, seeds, bootstrap
seed and resamples; per-question dense, oracle and random results (rankings as ``corpus_key``, offsets
and chunk text hash only); the three aggregate reports; the random macro statistics; the dense-minus-
random comparison, reported only (no pass/fail threshold is approved or applied); the oracle validity
checks with the unreachable units and adjudications; the unanswerable questions, listed apart with no
retrieval metric; the warnings of the dev gate; truncation counts; and every suppressed statistical slice.

Code identity method (``CODE_IDENTITY_METHOD``). For each module in ``CODE_IDENTITY_MODULES`` the source
file is read as bytes, every ``CRLF`` and lone ``CR`` is replaced by ``LF`` (so a Windows checkout and a
Linux checkout of the same commit agree), and the SHA-256 is taken. The combined hash is the SHA-256 of the
UTF-8 text ``"<module>\\t<sha256>\\n"`` for each module sorted by name. The files are located with
``importlib.util.find_spec`` without importing them. The caller may add the git commit and whether the
working tree was clean; they are recorded as given. The hashes identify the code that ran even when the
tree was dirty.

Integrity. ``build_report`` refuses (``EvaluationDataError``, fixed messages) when the run's manifest,
dev-dataset or pool hash disagrees with the inputs and the pool it is given (the two hashes are also
recomputed from the manifest and the questions), when the embedder id differs from the pool's mapped
embedder, when the question ids of any version differ from the dev answerable questions, when a value is
not finite, or when any string in the report contains a question, an expected answer or an evidence quote
of the dev split. ``check_comparable`` must be called before two reports are compared: it refuses when
their manifest hash, dev-dataset hash (and by default pool hash), question ids, series or cutoffs
differ, so two runs against different labels are never compared.

Privacy. The report holds no question text, source text, quote, expected answer, database id, owner id,
file path or credential. Question ids, corpus keys, unit ids, offsets and hashes are identifiers, not text.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.rag.evaluation import baselines, engine
from app.rag.evaluation import statistics as stats
from app.rag.evaluation.aggregation import (
    MIN_SLICE_QUESTIONS,
    STATUS_SUPPRESSED,
    Report,
)
from app.rag.evaluation.metrics import MAX_RANK, RELEVANCE_RULES, RankedChunk
from app.rag.evaluation.parsing import dataset_hash, manifest_hash
from app.rag.evaluation.pool import CandidatePool
from app.rag.evaluation.runner_inputs import DevInputs
from app.rag.evaluation.schema import SEVERITY_ERROR, EvaluationDataError, Issue
from app.rag.evaluation.structure import STRUCTURE_VERSION, derive_structure

REPORT_FORMAT_VERSION = "e1.7-report-1"
REPORT_KIND = "validity_pilot"
REPORT_LABEL = "Validity pilot — not the V1 baseline"
COMPARISON_NOTE = "Reported only: no dense-versus-random pass/fail threshold is approved or applied."
UNANSWERABLE_NOTE = "Listed apart: no retrieval was run and no answerable retrieval metric is claimed."

CODE_IDENTITY_MODULES = (
    "app.core.text_extraction",
    "app.rag.chunking",
    "app.rag.embedder",
    "app.rag.evaluation.aggregation",
    "app.rag.evaluation.anchoring",
    "app.rag.evaluation.baselines",
    "app.rag.evaluation.dataset",
    "app.rag.evaluation.engine",
    "app.rag.evaluation.intervals",
    "app.rag.evaluation.metrics",
    "app.rag.evaluation.parsing",
    "app.rag.evaluation.passages",
    "app.rag.evaluation.pool",
    "app.rag.evaluation.report",
    "app.rag.evaluation.runner_inputs",
    "app.rag.evaluation.schema",
    "app.rag.evaluation.statistics",
    "app.rag.evaluation.structure",
    "app.rag.evaluation.validation",
    "app.rag.indexing_service",
    "app.rag.retrieval",
    "app.rag.text_boundaries",
)
CODE_IDENTITY_METHOD = (
    "sha256 of each module source with CRLF/CR normalised to LF; combined = sha256 of "
    "'<module>\\t<sha256>\\n' lines sorted by module name"
)

REPORT_HASH_MISMATCH = "REPORT_HASH_MISMATCH"
REPORT_EMBEDDER_MISMATCH = "REPORT_EMBEDDER_MISMATCH"
REPORT_QUESTION_MISMATCH = "REPORT_QUESTION_MISMATCH"
REPORT_METADATA_INVALID = "REPORT_METADATA_INVALID"
REPORT_NONFINITE = "REPORT_NONFINITE"
REPORT_UNSERIALIZABLE = "REPORT_UNSERIALIZABLE"
REPORT_LEAK = "REPORT_LEAK"
REPORT_NOT_COMPARABLE = "REPORT_NOT_COMPARABLE"
REPORT_INVALID = "REPORT_INVALID"
REPORT_IDENTITY_UNAVAILABLE = "REPORT_IDENTITY_UNAVAILABLE"

_MESSAGES = {
    REPORT_HASH_MISMATCH: "a hash of the run disagrees with the inputs or the pool it is reported with",
    REPORT_EMBEDDER_MISMATCH: "the embedder id differs from the embedder of the candidate pool",
    REPORT_QUESTION_MISMATCH: "the question ids of the run differ from the dev answerable questions",
    REPORT_METADATA_INVALID: "the run metadata has an invalid field",
    REPORT_NONFINITE: "a value in the report is not a finite number",
    REPORT_UNSERIALIZABLE: "a value in the report has a type that cannot be serialised",
    REPORT_LEAK: "the report contains text that belongs to the dev questions or their evidence",
    REPORT_NOT_COMPARABLE: "the two reports were not produced against the same labels, questions or series",
    REPORT_INVALID: "the report is not a valid validity-pilot report",
    REPORT_IDENTITY_UNAVAILABLE: "the source of an identity module could not be read",
}

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MIN_FORBIDDEN = 6


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_ERROR)


def _fail(code: str, location: str) -> EvaluationDataError:
    return EvaluationDataError([_issue(code, location)])


# ---------- metadata ----------

def utc_timestamp(now: datetime | None = None) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` for ``now`` (default: the current UTC time). Not called by the report."""
    moment = (datetime.now(timezone.utc)  # noqa: UP017
              if now is None else now)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def current_python_version() -> str:
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def hash_source_bytes(data: bytes) -> str:
    """SHA-256 of ``data`` with CRLF and CR normalised to LF (the code identity method)."""
    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    method: str
    files: tuple[tuple[str, str], ...]  # (module, sha256), sorted by module
    combined_sha256: str


def combine_identity(files: Mapping[str, str]) -> CodeIdentity:
    ordered = tuple(sorted(files.items()))
    text = "".join(f"{module}\t{digest}\n" for module, digest in ordered)
    return CodeIdentity(CODE_IDENTITY_METHOD, ordered, hashlib.sha256(text.encode("utf-8")).hexdigest())


def compute_code_identity(modules: Sequence[str] = CODE_IDENTITY_MODULES) -> CodeIdentity:
    """Hash the source files of ``modules`` (located, not imported). The only file reads of this module."""
    if isinstance(modules, (str, bytes)) or not isinstance(modules, Sequence):
        raise TypeError("modules must be a sequence of module names")
    hashes: dict[str, str] = {}
    for name in modules:
        if type(name) is not str:
            raise TypeError("a module name must be a str")
        try:
            spec = importlib.util.find_spec(name)
            origin = None if spec is None else spec.origin
            if origin is None or not origin.endswith(".py"):
                raise OSError
            with open(origin, "rb") as handle:
                hashes[name] = hash_source_bytes(handle.read())
        except (OSError, ImportError, ValueError):
            raise _fail(REPORT_IDENTITY_UNAVAILABLE, "code_identity") from None
    return combine_identity(hashes)


@dataclass(frozen=True, slots=True)
class DatabaseSnapshotRecord:
    """Content fingerprints of the read-only snapshots taken before and after the run (E1.5 snapshots).

    ``unchanged`` is True when the two snapshots compare as identical. Only fingerprints are recorded.
    """

    before_fingerprint: str
    after_fingerprint: str
    unchanged: bool


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """What identifies this run apart from its results. Supplied by the caller, never read here."""

    run_timestamp_utc: str
    python_version: str
    embedder_id: str
    embedder_signature: str
    code_identity: CodeIdentity
    git_commit: str | None = None
    working_tree_clean: bool | None = None
    database_snapshots: DatabaseSnapshotRecord | None = None


def _snapshots_ok(record: object) -> bool:
    if record is None:
        return True
    return (
        isinstance(record, DatabaseSnapshotRecord)
        and all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in (record.before_fingerprint, record.after_fingerprint))
        and type(record.unchanged) is bool
    )


def _check_metadata(metadata: RunMetadata) -> None:
    bad = (
        not isinstance(metadata, RunMetadata)
        or type(metadata.run_timestamp_utc) is not str
        or _TIMESTAMP.fullmatch(metadata.run_timestamp_utc) is None
        or type(metadata.python_version) is not str
        or re.fullmatch(r"\d+\.\d+\.\d+", metadata.python_version) is None
        or type(metadata.embedder_id) is not str
        or not metadata.embedder_id
        or type(metadata.embedder_signature) is not str
        or not metadata.embedder_signature
        or not isinstance(metadata.code_identity, CodeIdentity)
        or not metadata.code_identity.files
        or (metadata.git_commit is not None and (type(metadata.git_commit) is not str
                                                 or _COMMIT.fullmatch(metadata.git_commit) is None))
        or (metadata.working_tree_clean is not None and type(metadata.working_tree_clean) is not bool)
        or not _snapshots_ok(metadata.database_snapshots)
    )
    if bad:
        raise _fail(REPORT_METADATA_INVALID, "metadata")


# ---------- plain conversion ----------

def _plain(value: Any, path: str = "report") -> Any:
    """Recursively convert dataclasses, tuples and mappings to JSON types; refuse anything else."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(REPORT_NONFINITE, path)
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), f"{path}.{f.name}") for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        if not all(type(k) is str for k in value):
            raise _fail(REPORT_UNSERIALIZABLE, path)
        return {k: _plain(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise _fail(REPORT_UNSERIALIZABLE, path)


def _chunk(chunk: RankedChunk) -> dict[str, Any]:
    return {"corpus_key": chunk.corpus_key, "start": chunk.start, "end": chunk.end, "text_sha256": chunk.text_sha256}


def _metrics(ids: Sequence[baselines.MetricId], values: Sequence[float]) -> dict[str, float]:
    return {engine.series_name(m): v for m, v in zip(ids, values)}


# ---------- verification ----------

def _verify(run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool, metadata: RunMetadata) -> None:
    if not isinstance(run, engine.PilotRun):
        raise TypeError("run must be a PilotRun")
    if not isinstance(inputs, DevInputs):
        raise TypeError("inputs must be DevInputs")
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    _check_metadata(metadata)

    if (
        run.manifest_sha256 != inputs.manifest_sha256
        or run.manifest_sha256 != manifest_hash(inputs.manifest)
        or run.dev_dataset_sha256 != inputs.dev_dataset_sha256
        or run.dev_dataset_sha256 != dataset_hash(inputs.questions)
        or run.pool_sha256 != pool.pool_sha256
        or run.pool_size != len(pool)
    ):
        raise _fail(REPORT_HASH_MISMATCH, "run")
    if metadata.embedder_id != pool.document_map.embedding_id:
        raise _fail(REPORT_EMBEDDER_MISMATCH, "metadata.embedder_id")
    expected = [q.question_id for q in inputs.answerable]
    groups = (
        [r.question_id for r in run.questions],
        [o.question_id for o in run.dense_observations],
        [o.question_id for o in run.oracle_observations],
        [o.question_id for o in run.random_observations],
        [u.question_id for u in run.unanswerable],
    )
    if list(groups[:4]) != [expected] * 4 or groups[4] != [q.question_id for q in inputs.unanswerable]:
        raise _fail(REPORT_QUESTION_MISMATCH, "run.questions")


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _forbidden(inputs: DevInputs) -> list[str]:
    found: set[str] = set()
    for question in inputs.questions:
        found.add(question.question)
        if question.expected_answer:
            found.add(question.expected_answer)
        for unit in question.units or ():
            found.update(span.quote for span in unit.spans)
    return [text for text in found if isinstance(text, str) and len(text) >= _MIN_FORBIDDEN]


def _check_no_leak(document: dict[str, Any], inputs: DevInputs) -> None:
    forbidden = _forbidden(inputs)
    for leaf in _strings(document):
        if any(text in leaf for text in forbidden):
            raise _fail(REPORT_LEAK, "report")


# ---------- document ----------

def _slice_status(report: Report, name: str) -> list[dict[str, Any]]:
    return [
        {"report": name, "retrieval_type": sl.retrieval_type, "n": sl.n,
         "reason": f"fewer than {MIN_SLICE_QUESTIONS} questions"}
        for sl in report.slices
        if sl.series and all(result.status == STATUS_SUPPRESSED for result in sl.series)
    ]


def _truncation(report: Report) -> list[list[int]]:
    return [[cutoff, count] for cutoff, count in report.overall.truncation]


def build_report(run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool, metadata: RunMetadata) -> dict[str, Any]:
    """The JSON-ready report document (plain dicts, lists, strings, ints and finite floats)."""
    _verify(run, inputs, pool, metadata)
    structure = {q.question_id: derive_structure(q) for q in inputs.answerable}
    ids = run.questions[0].dense_scores.metric_ids if run.questions else ()

    dense_questions, oracle_questions, random_questions = [], [], []
    for item, dense_o, oracle_o, random_o in zip(
        run.questions, run.dense_observations, run.oracle_observations, run.random_observations
    ):
        base = {
            "question_id": item.question_id,
            "retrieval_type": item.retrieval_type,
            "unit_structure": structure[item.question_id].unit_structure,
            "has_alternatives": structure[item.question_id].has_alternatives,
        }
        dense_questions.append({
            **base, "ranking": [_chunk(c) for c in item.dense],
            "metrics": dict(dense_o.values), "truncated_at": list(dense_o.truncated_at),
        })
        oracle = item.oracle
        oracle_questions.append({
            **base, "ranking": [_chunk(c) for c in oracle.ranking],
            "metrics": dict(oracle_o.values), "truncated_at": list(oracle_o.truncated_at),
            "chunks_needed": oracle.chunks_needed, "method": oracle.method, "unit_ids": list(oracle.unit_ids),
            "unreachable_by_rule": {rule: list(units) for rule, units in oracle.unreachable_by_rule},
            "reachable_only": {
                r.rule: {
                    "unit_count": r.unit_count,
                    "metrics": None if r.scores is None else _metrics(r.scores.metric_ids, r.scores.values),
                }
                for r in oracle.reachable_only
            },
        })
        summaries = item.random.summaries
        random_questions.append({
            **base, "n_seeds": len(item.random.seeds),
            "metrics_mean": dict(random_o.values), "truncated_at": list(random_o.truncated_at),
            "metrics_p2_5": {engine.series_name(s.metric): s.p2_5 for s in summaries},
            "metrics_p95": {engine.series_name(s.metric): s.p95 for s in summaries},
            "metrics_p97_5": {engine.series_name(s.metric): s.p97_5 for s in summaries},
        })

    record = run.questions[0].random.reproducibility if run.questions else None
    suppressed = (
        _slice_status(run.dense_report, "dense") + _slice_status(run.oracle_report, "oracle")
        + _slice_status(run.random_report, "random")
    )
    document = {
        "report_format_version": REPORT_FORMAT_VERSION,
        "kind": REPORT_KIND,
        "label": REPORT_LABEL,
        "is_v1_baseline": False,
        "run": {
            "timestamp_utc": metadata.run_timestamp_utc, "python_version": metadata.python_version,
            "git_commit": metadata.git_commit, "working_tree_clean": metadata.working_tree_clean,
        },
        "code_identity": metadata.code_identity,
        "embedder": {"id": metadata.embedder_id, "signature": metadata.embedder_signature},
        "database_snapshots": metadata.database_snapshots,
        "inputs": {
            "manifest": {
                "sha256": run.manifest_sha256, "schema_version": inputs.manifest.schema_version,
                "corpus_version": inputs.manifest.corpus_version, "stage": inputs.manifest.stage,
                "document_count": len(inputs.manifest.documents),
            },
            "dev_dataset": {
                "sha256": run.dev_dataset_sha256, "split": "dev", "test_split_read": False,
                "answerable_count": len(inputs.answerable), "unanswerable_count": len(inputs.unanswerable),
            },
            "candidate_pool": {"sha256": run.pool_sha256, "size": run.pool_size},
            "corpus_keys": [d.corpus_key for d in inputs.manifest.documents],
        },
        "settings": {
            "cutoffs": list(run.cutoffs), "max_rank": MAX_RANK, "relevance_rules": list(RELEVANCE_RULES),
            "primary_rule": baselines.PRIMARY_RULE, "random_seeds": list(run.seeds),
            "random_seed_count": len(run.seeds), "min_random_seeds": baselines.MIN_RANDOM_SEEDS,
            "bootstrap_seed": run.dense_report.seed, "resamples": run.dense_report.resamples,
            "confidence_level": stats.CONFIDENCE_LEVEL, "min_slice_questions": MIN_SLICE_QUESTIONS,
            "percentile_convention": stats.PERCENTILE_CONVENTION, "structure_version": STRUCTURE_VERSION,
            "oracle_method": baselines.ORACLE_METHOD, "random_reproducibility": record,
            "series": {"binary_hit": list(run.series), "all_continuous": list(run.series_continuous)},
            "metric_count": len(ids),
        },
        "warnings": [{"code": w.code, "location": w.location, "severity": w.severity} for w in inputs.warnings],
        "dense": {"questions": dense_questions, "report": run.dense_report},
        "oracle": {"questions": oracle_questions, "report": run.oracle_report, "validity": run.oracle_validity},
        "random": {"questions": random_questions, "report": run.random_report, "macro": run.random_macro},
        "dense_vs_random": {
            "note": COMPARISON_NOTE, "baseline": "random", "candidate": "dense", "report": run.dense_vs_random,
        },
        "unanswerable": {
            "note": UNANSWERABLE_NOTE, "count": len(run.unanswerable), "retrieval_metrics": None,
            "questions": list(run.unanswerable),
        },
        "truncation_counts": {
            "dense": _truncation(run.dense_report), "oracle": _truncation(run.oracle_report),
            "random": _truncation(run.random_report),
        },
        "suppressed_slices": suppressed,
    }
    plain = _plain(document)
    _check_no_leak(plain, inputs)
    return plain


# ---------- serialization ----------

@dataclass(frozen=True, slots=True)
class SerializedReport:
    document: dict[str, Any]
    json_bytes: bytes
    sha256: str


def serialize_report(document: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, no whitespace, no NaN, a final newline."""
    try:
        text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise _fail(REPORT_UNSERIALIZABLE, "report") from None
    return (text + "\n").encode("utf-8")


def render_report(run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool, metadata: RunMetadata) -> SerializedReport:
    document = build_report(run, inputs, pool, metadata)
    data = serialize_report(document)
    return SerializedReport(document, data, hashlib.sha256(data).hexdigest())


def load_report(data: bytes) -> dict[str, Any]:
    """Parse a serialised report and check its kind, version and label (no other validation)."""
    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _fail(REPORT_INVALID, "report") from None
    if (
        not isinstance(document, dict)
        or document.get("kind") != REPORT_KIND
        or document.get("report_format_version") != REPORT_FORMAT_VERSION
        or document.get("label") != REPORT_LABEL
        or document.get("is_v1_baseline") is not False
    ):
        raise _fail(REPORT_INVALID, "report")
    return document


# ---------- comparison guard ----------

def _identity(document: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        inputs = document["inputs"]
        settings = document["settings"]
        return (
            inputs["manifest"]["sha256"], inputs["dev_dataset"]["sha256"],
            tuple(q["question_id"] for q in document["dense"]["questions"]),
            tuple(settings["cutoffs"]), tuple(settings["relevance_rules"]),
            tuple(s["name"] for s in settings["series"]["binary_hit"]),
        )
    except (KeyError, TypeError):
        raise _fail(REPORT_INVALID, "report") from None


def check_comparable(first: Mapping[str, Any], second: Mapping[str, Any], *, require_same_pool: bool = True) -> None:
    """Refuse (``EvaluationDataError``) unless the two reports used the same labels and questions.

    Compared: manifest hash, dev-dataset hash, question ids in order, cutoffs, relevance rules and series
    names, and by default the candidate-pool hash. Both must be validity-pilot reports. Call this before
    any comparison of two runs; the hashes are the proof that both were scored against the same labels.
    """
    for document in (first, second):
        if (
            not isinstance(document, Mapping) or document.get("kind") != REPORT_KIND
            or document.get("report_format_version") != REPORT_FORMAT_VERSION
        ):
            raise _fail(REPORT_INVALID, "report")
    if _identity(first) != _identity(second):
        raise _fail(REPORT_NOT_COMPARABLE, "reports")
    if require_same_pool and first["inputs"]["candidate_pool"]["sha256"] != second["inputs"]["candidate_pool"]["sha256"]:
        raise _fail(REPORT_NOT_COMPARABLE, "reports.candidate_pool")
