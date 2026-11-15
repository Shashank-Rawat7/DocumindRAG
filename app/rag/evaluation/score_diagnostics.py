"""Descriptive retrieval-score diagnostics of the retrieval-only dense baseline (E2.3).

Pure module: it reads no file and no database, runs no SQL and never touches the test split. It turns the
dense scores of the development questions into numbers. It reports; nothing in it selects a cutoff, tunes,
calibrates, predicts answerability or applies a pass/fail rule.

Where the scores come from. ``retrieval.RetrievedChunk.score`` (1 - cosine distance) is dropped by
``pool.convert_retrieved``. ``ScoreRecorder`` wraps the retriever handed to ``engine.run_pilot`` and
snapshots every call (a hash of the query, never its text; the identity of each returned chunk; its score)
without changing the returned object. ``build_score_diagnostics`` then checks that the recorded calls match
the answerable dev questions one by one (call count, order, query hash) and that each recorded ranking is
exactly the dense ranking kept in the ``PilotRun`` (chunk count, order and identity). That identity data
exists only in memory and is never written to the report. The unanswerable dev questions are retrieved
separately by ``collect_unanswerable_scores`` (dev-file order, a plain retriever, never a recorder): their
scores are kept for these diagnostics only and no relevance or retrieval metric is computed for them.

Statistics (every one is numbers only). A score vector ``s1..sL`` holds the scores of one ranking, best
first, with ``L = min(20, pool size)``. It must be non-empty, of length ``L``, made of finite ``float``
values and non-increasing; it is never sorted or repaired. Per question: top-1 ``s1``; gap ``s1 - s2``
(valid only when ``L >= 2``); spread ``s1 - sL``. Per group (answerable, unanswerable) and per statistic:
the number of valid observations, min, max, mean (``math.fsum(values) / n``) and the 25th, 50th and 75th
percentiles by nearest rank (``sorted[ceil(p * n / 100) - 1]``, integer arithmetic; the median is the
50th percentile, the lower middle value for an even count). Also per rank ``1..L`` the number of
contributing questions, mean and median; ``tied_adjacent_pairs`` (adjacent scores that are exactly equal)
and ``top_two_tied_questions`` (questions whose first two scores are exactly equal). There is no tolerance
anywhere. An empty group is valid and has null statistics. Every non-null result must be finite: a
subtraction or an aggregation that overflows is refused (``SCORES_OVERFLOW``), never clamped.

Digest. ``scores_sha256`` is the SHA-256 of the UTF-8 bytes of the compact JSON text of
``["e2.3-scores-digest-1", L, [[question_id, [scores...]], ...], [[question_id, [scores...]], ...]]``: the
answerable rows first, then the unanswerable rows, each in dev-file order, floats printed by Python's
shortest round-trip ``repr``, ``separators=(",", ":")``, ``ensure_ascii=False``, ``allow_nan=False``, no
trailing newline. It is recorded and, when a second pass is supplied, compared with the second pass's
digest; a mismatch is reported only.

Errors: ``TypeError`` for a wrongly typed argument, ``EvaluationDataError`` (fixed messages, no value, no
text) for everything about the data.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.evaluation import engine
from app.rag.evaluation.metrics import MAX_RANK
from app.rag.evaluation.pool import CandidatePool, convert_retrieved
from app.rag.evaluation.runner_inputs import DevInputs
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    SHA256_HEX_PATTERN,
    EvaluationDataError,
    Issue,
)

SECTION_FORMAT = "e2.3-score-diagnostics-1"
DIGEST_FORMAT = "e2.3-scores-digest-1"
PERCENTILES = (25, 50, 75)

SCORES_RECORDER_STATE = "SCORES_RECORDER_STATE"
SCORES_CALL_COUNT = "SCORES_CALL_COUNT"
SCORES_QUERY_MISMATCH = "SCORES_QUERY_MISMATCH"
SCORES_RANKING_MISMATCH = "SCORES_RANKING_MISMATCH"
SCORES_RETURN = "SCORES_RETURN"
SCORES_EMPTY = "SCORES_EMPTY"
SCORES_LENGTH = "SCORES_LENGTH"
SCORES_MISSING = "SCORES_MISSING"
SCORES_NONFINITE = "SCORES_NONFINITE"
SCORES_ORDER = "SCORES_ORDER"
SCORES_OVERFLOW = "SCORES_OVERFLOW"
SCORES_QUESTION_IDS = "SCORES_QUESTION_IDS"
SCORES_DIGEST_INVALID = "SCORES_DIGEST_INVALID"

MESSAGES = {
    SCORES_RECORDER_STATE: "the score recorder was already taken and cannot be used again",
    SCORES_CALL_COUNT: "the recorded retriever calls do not match the answerable dev questions in number",
    SCORES_QUERY_MISMATCH: "a recorded retriever call does not belong to its answerable dev question",
    SCORES_RANKING_MISMATCH: "the recorded ranking differs from the dense ranking of the run",
    SCORES_RETURN: "the retriever must return a list or tuple of chunks",
    SCORES_EMPTY: "a score vector is empty",
    SCORES_LENGTH: "a score vector does not have min(20, pool size) entries",
    SCORES_MISSING: "a chunk has no score or its score is not a float",
    SCORES_NONFINITE: "a score is not a finite number",
    SCORES_ORDER: "the scores of a ranking are not in non-increasing order",
    SCORES_OVERFLOW: "a score statistic is not a finite number",
    SCORES_QUESTION_IDS: "the scored question ids differ from the dev questions",
    SCORES_DIGEST_INVALID: "the second-pass scores hash is not 64 lowercase hexadecimal digits",
}

_ABSENT = None
_CHUNK_FIELDS = ("document_id", "start_offset", "end_offset", "span_sha256", "score")
_SECTION_KEYS = frozenset({
    "format", "ranking_length", "scores_sha256", "second_pass_scores_sha256", "scores_identical_across_passes",
    "groups", "questions",
})
_ROW_KEYS = frozenset({"question_id", "scores"})


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, MESSAGES[code], SEVERITY_ERROR)


def _fail(code: str, location: str) -> EvaluationDataError:
    return EvaluationDataError([_issue(code, location)])


# ---------- score vectors ----------

@dataclass(frozen=True, slots=True)
class QuestionScores:
    """The score vector of one question's ranking, best first (question id and numbers only)."""

    question_id: str
    scores: tuple[float, ...]


def expected_length(pool_size: int) -> int:
    """``min(20, pool size)``: the length of every ranking (the engine enforces the same rule)."""
    return min(MAX_RANK, pool_size)


def _check_vector(scores: object, length: int, where: str) -> tuple[float, ...]:
    """The scores as a tuple, or ``EvaluationDataError``. Never sorts or repairs anything."""
    if not isinstance(scores, (list, tuple)) or len(scores) == 0:
        raise _fail(SCORES_EMPTY, where)
    if len(scores) != length:
        raise _fail(SCORES_LENGTH, where)
    if any(type(value) is not float for value in scores):
        raise _fail(SCORES_MISSING, where)
    if not all(math.isfinite(value) for value in scores):
        raise _fail(SCORES_NONFINITE, where)
    if any(earlier < later for earlier, later in itertools.pairwise(scores)):
        raise _fail(SCORES_ORDER, where)
    return tuple(scores)


# ---------- statistics ----------

def _finite(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise _fail(SCORES_OVERFLOW, where)
    return value


def _mean(values: Sequence[float], where: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError:
        raise _fail(SCORES_OVERFLOW, where) from None
    return _finite(total / len(values), where)


def _nearest_rank(ordered: Sequence[float], percentile: int) -> float:
    rank = (percentile * len(ordered) + 99) // 100
    return ordered[rank - 1]


def _stat(values: Sequence[float], where: str) -> dict[str, Any]:
    if not values:
        return {"valid_count": 0, "min": None, "max": None, "mean": None, "p25": None, "p50": None, "p75": None}
    ordered = sorted(values)
    return {
        "valid_count": len(values), "min": ordered[0], "max": ordered[-1], "mean": _mean(values, where),
        **{f"p{p}": _nearest_rank(ordered, p) for p in PERCENTILES},
    }


def summarize_group(vectors: Sequence[Sequence[float]], length: int) -> dict[str, Any]:
    """The statistics of one group of score vectors (see the module docstring). An empty group is valid."""
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
        raise TypeError("vectors must be a sequence of score vectors")
    if type(length) is not int:
        raise TypeError("length must be an int")
    if length < 1:
        raise _fail(SCORES_EMPTY, "ranking_length")
    checked = [_check_vector(vector, length, f"vectors[{i}]") for i, vector in enumerate(vectors)]

    top1 = [vector[0] for vector in checked]
    gap = [_finite(vector[0] - vector[1], "gap") for vector in checked if length >= 2]
    spread = [_finite(vector[0] - vector[-1], "spread") for vector in checked]
    per_rank = []
    for index in range(length):
        column = [vector[index] for vector in checked]
        ordered = sorted(column)
        per_rank.append({
            "rank": index + 1, "question_count": len(column),
            "mean": _mean(column, f"per_rank[{index + 1}]") if column else None,
            "median": _nearest_rank(ordered, 50) if column else None,
        })
    return {
        "question_count": len(checked),
        "top1": _stat(top1, "top1"), "gap": _stat(gap, "gap"), "spread": _stat(spread, "spread"),
        "per_rank": per_rank,
        "tied_adjacent_pairs": sum(
            1 for vector in checked for earlier, later in itertools.pairwise(vector) if earlier == later
        ),
        "top_two_tied_questions": sum(1 for vector in checked if length >= 2 and vector[0] == vector[1]),
    }


# ---------- digest ----------

def scores_digest(length: int, answerable: Sequence[QuestionScores], unanswerable: Sequence[QuestionScores]) -> str:
    """SHA-256 of the canonical payload described in the module docstring."""
    payload = [
        DIGEST_FORMAT, length,
        [[row.question_id, list(row.scores)] for row in answerable],
        [[row.question_id, list(row.scores)] for row in unanswerable],
    ]
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- the section ----------

def _section(
    length: int, answerable: Sequence[QuestionScores], unanswerable: Sequence[QuestionScores],
    second_pass: str | None,
) -> dict[str, Any]:
    first = scores_digest(length, answerable, unanswerable)
    return {
        "format": SECTION_FORMAT,
        "ranking_length": length,
        "scores_sha256": first,
        "second_pass_scores_sha256": second_pass,
        "scores_identical_across_passes": None if second_pass is None else second_pass == first,
        "groups": {
            "answerable": summarize_group([row.scores for row in answerable], length),
            "unanswerable": summarize_group([row.scores for row in unanswerable], length),
        },
        "questions": {
            "answerable": [{"question_id": row.question_id, "scores": list(row.scores)} for row in answerable],
            "unanswerable": [{"question_id": row.question_id, "scores": list(row.scores)} for row in unanswerable],
        },
    }


def _same_json(first: Any, second: Any) -> bool:
    """Equal as canonical JSON, so ``0`` is not ``0.0``."""
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)


def _rows(raw: object, ids: Sequence[str], length: int, where: str) -> list[QuestionScores]:
    if not isinstance(raw, (list, tuple)) or len(raw) != len(ids):
        raise _fail(SCORES_QUESTION_IDS, where)
    rows = []
    for index, (item, expected_id) in enumerate(zip(raw, ids, strict=True)):
        if not isinstance(item, Mapping) or set(item) != _ROW_KEYS:
            raise _fail(SCORES_QUESTION_IDS, f"{where}[{index}]")
        if type(item["question_id"]) is not str or item["question_id"] != expected_id:
            raise _fail(SCORES_QUESTION_IDS, f"{where}[{index}]")
        rows.append(QuestionScores(expected_id, _check_vector(item["scores"], length, f"{where}[{index}]")))
    return rows


def _section_ok(
    section: object, answerable_ids: Sequence[str], unanswerable_ids: Sequence[str], pool_size: int,
    answerable_ranking_lengths: Sequence[int],
) -> bool:
    if not isinstance(section, Mapping) or set(section) != _SECTION_KEYS or type(pool_size) is not int:
        return False
    length = expected_length(pool_size)
    if length < 1 or section["format"] != SECTION_FORMAT:
        return False
    if type(section["ranking_length"]) is not int or section["ranking_length"] != length:
        return False
    second = section["second_pass_scores_sha256"]
    if second is not None and (type(second) is not str or SHA256_HEX_PATTERN.fullmatch(second) is None):
        return False
    questions = section["questions"]
    if not isinstance(questions, Mapping) or set(questions) != {"answerable", "unanswerable"}:
        return False
    answerable = _rows(questions["answerable"], answerable_ids, length, "questions.answerable")
    unanswerable = _rows(questions["unanswerable"], unanswerable_ids, length, "questions.unanswerable")
    if len(answerable_ranking_lengths) != len(answerable) or any(
        type(n) is not int or len(row.scores) != n
        for row, n in zip(answerable, answerable_ranking_lengths, strict=True)
    ):
        return False
    return _same_json(section, _section(length, answerable, unanswerable, second))


def section_ok(
    section: object, *, answerable_ids: Sequence[str], unanswerable_ids: Sequence[str], pool_size: int,
    answerable_ranking_lengths: Sequence[int],
) -> bool:
    """True only if ``section`` is exactly what ``build_score_diagnostics`` would produce for these ids.

    Checks the keys and types, the format, ``ranking_length == min(20, pool_size)``, the question ids and
    their order, that each answerable vector is as long as that question's ranking, every vector (finite
    floats, non-increasing), and then recomputes every statistic, the digest and the cross-pass fields from
    the vectors and requires exact equality. Never raises; any problem is ``False``.
    """
    try:
        return _section_ok(section, answerable_ids, unanswerable_ids, pool_size, answerable_ranking_lengths)
    except (KeyError, TypeError, ValueError, AttributeError, EvaluationDataError):
        return False


# ---------- recording (answerable questions) ----------

@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One snapshot: the SHA-256 of the query and ``(document_id, start, end, span_sha256, score)`` per chunk.

    Temporary: it holds database ids and is used only to check a ranking; it is never written to a report.
    ``chunks`` is ``None`` when the retriever did not return a list or tuple; a field a chunk lacks is ``None``.
    """

    query_sha256: str
    chunks: tuple[tuple[Any, ...], ...] | None


def _snapshot(result: object) -> tuple[tuple[Any, ...], ...] | None:
    if not isinstance(result, (list, tuple)):
        return None
    return tuple(tuple(getattr(chunk, name, _ABSENT) for name in _CHUNK_FIELDS) for chunk in result)


class ScoreRecorder:
    """Wraps the retriever of ``engine.run_pilot``: forwards every call unchanged and snapshots it.

    One recorder serves one run: ``take()`` returns the snapshots once and seals the recorder.
    """

    __slots__ = ("_calls", "_retriever", "_sealed")

    def __init__(self, retriever: Callable[[str], Sequence[Any]]) -> None:
        if not callable(retriever):
            raise TypeError("retriever must be callable")
        self._retriever = retriever
        self._calls: list[RecordedCall] = []
        self._sealed = False

    def __call__(self, query: str) -> Sequence[Any]:
        if self._sealed:
            raise _fail(SCORES_RECORDER_STATE, "recorder")
        if type(query) is not str:
            raise TypeError("query must be a str")
        result = self._retriever(query)  # an error of the retriever propagates and records nothing
        self._calls.append(RecordedCall(hashlib.sha256(query.encode("utf-8")).hexdigest(), _snapshot(result)))
        return result

    def take(self) -> tuple[RecordedCall, ...]:
        if self._sealed:
            raise _fail(SCORES_RECORDER_STATE, "recorder")
        self._sealed = True
        return tuple(self._calls)


# ---------- unanswerable questions (a separate path) ----------

def collect_unanswerable_scores(
    inputs: DevInputs, pool: CandidatePool, retriever: Callable[[str], Sequence[Any]],
) -> tuple[QuestionScores, ...]:
    """Retrieve every unanswerable dev question once, in dev-file order, and keep only its scores.

    The ranking is checked with ``pool.convert_retrieved`` (every chunk must be in the pool) and must have
    ``min(20, pool size)`` chunks. No relevance or retrieval metric is computed. A ``ScoreRecorder`` is
    refused so the answerable call count can never be mixed with these calls.
    """
    if not isinstance(inputs, DevInputs):
        raise TypeError("inputs must be DevInputs")
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    if not callable(retriever) or isinstance(retriever, ScoreRecorder):
        raise TypeError("retriever must be a plain callable, not a ScoreRecorder")
    length = expected_length(len(pool))
    rows: list[QuestionScores] = []
    for index, question in enumerate(inputs.unanswerable):
        where = f"unanswerable[{index}]"
        retrieved = retriever(question.question)
        if not isinstance(retrieved, (list, tuple)):
            raise _fail(SCORES_RETURN, where)
        try:
            ranking = convert_retrieved(pool, retrieved)
        except EvaluationDataError as error:
            raise EvaluationDataError(
                [Issue(i.code, f"{where}.{i.location}", i.message, i.severity) for i in error.issues]
            ) from None
        if len(ranking) != length:
            raise _fail(SCORES_LENGTH, where)
        scores = [getattr(chunk, "score", _ABSENT) for chunk in retrieved]
        rows.append(QuestionScores(question.question_id, _check_vector(scores, length, where)))
    return tuple(rows)


# ---------- building the section ----------

def _answerable_rows(
    run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool, calls: Sequence[RecordedCall], length: int,
) -> list[QuestionScores]:
    questions = inputs.answerable
    if len(calls) != len(questions) or len(run.questions) != len(questions):
        raise _fail(SCORES_CALL_COUNT, "answerable")
    rows = []
    for index, (question, call, item) in enumerate(zip(questions, calls, run.questions, strict=True)):
        where = f"answerable[{index}]"
        if not isinstance(call, RecordedCall):
            raise TypeError("answerable_calls must hold RecordedCall objects")
        if call.query_sha256 != hashlib.sha256(question.question.encode("utf-8")).hexdigest():
            raise _fail(SCORES_QUERY_MISMATCH, where)
        if item.question_id != question.question_id or call.chunks is None:
            raise _fail(SCORES_RANKING_MISMATCH, where)
        recorded = []
        for entry in call.chunks:
            try:
                key = pool.document_map.key_by_id.get(entry[0])
            except TypeError:
                key = None
            recorded.append((key, entry[1], entry[2], entry[3]))
        kept = [(c.corpus_key, c.start, c.end, c.text_sha256) for c in item.dense]
        if recorded != kept:
            raise _fail(SCORES_RANKING_MISMATCH, where)
        rows.append(QuestionScores(question.question_id, _check_vector([e[4] for e in call.chunks], length, where)))
    return rows


def build_score_diagnostics(
    run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool, answerable_calls: Sequence[RecordedCall],
    unanswerable: Sequence[QuestionScores], *, second_pass_scores_sha256: str | None = None,
) -> dict[str, Any]:
    """The ``score_diagnostics`` section of a ``retrieval_baseline`` report (see the module docstring).

    ``answerable_calls`` is ``ScoreRecorder.take()`` of the recorder the run used; ``unanswerable`` is
    ``collect_unanswerable_scores``. ``second_pass_scores_sha256`` is the ``scores_sha256`` of a second,
    independent pass (or ``None``); when given, the section records whether it equals this pass's digest.
    The comparison is reported only: no tolerance, no gate.
    """
    if not isinstance(run, engine.PilotRun):
        raise TypeError("run must be a PilotRun")
    if not isinstance(inputs, DevInputs):
        raise TypeError("inputs must be DevInputs")
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    if isinstance(answerable_calls, (str, bytes)) or not isinstance(answerable_calls, Sequence):
        raise TypeError("answerable_calls must be a sequence of RecordedCall")
    if isinstance(unanswerable, (str, bytes)) or not isinstance(unanswerable, Sequence):
        raise TypeError("unanswerable must be a sequence of QuestionScores")
    if second_pass_scores_sha256 is not None and (
        type(second_pass_scores_sha256) is not str or SHA256_HEX_PATTERN.fullmatch(second_pass_scores_sha256) is None
    ):
        raise _fail(SCORES_DIGEST_INVALID, "second_pass_scores_sha256")
    length = expected_length(len(pool))
    if length < 1:
        raise _fail(SCORES_EMPTY, "ranking_length")
    answerable = _answerable_rows(run, inputs, pool, answerable_calls, length)
    if not all(isinstance(row, QuestionScores) for row in unanswerable) or [
        row.question_id for row in unanswerable
    ] != [question.question_id for question in inputs.unanswerable]:
        raise _fail(SCORES_QUESTION_IDS, "unanswerable")
    rows = [
        QuestionScores(row.question_id, _check_vector(row.scores, length, f"unanswerable[{i}]"))
        for i, row in enumerate(unanswerable)
    ]
    return _section(length, answerable, rows, second_pass_scores_sha256)
