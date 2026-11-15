"""The D7 ranking hash of the retrieval-only baseline (E2.4a).

D7 requires two independent dense runs over the same frozen dev inputs to produce identical rankings. This
module reduces the dense rankings of ONE run to one SHA-256 so two runs can be compared exactly. It is pure:
no file, database, model or clock is touched, and nothing is written.

Scope. The answerable dense rankings retained in ``PilotRun.questions[*].dense`` only. Unanswerable questions
are not covered: their rankings are not retained anywhere, and this module does not capture them.

Identity. A chunk is its existing ``metrics.RankedChunk`` identity ``(corpus_key, start, end, text_sha256)``;
no parallel representation exists and no database id (chunk, document, owner) enters the payload.

Canonical payload (format ``e2.4-ranking-digest-1``), a JSON array::

    ["e2.4-ranking-digest-1", L, [[question_id, [[corpus_key, start, end, text_sha256], ...]], ...]]

``L = min(20, pool size)``; the questions are in dev-file order; the chunks of a question are in their original
rank order. The text is compact UTF-8 JSON (separators ``","`` and ``":"``, ``ensure_ascii=False``,
``allow_nan=False``, no trailing newline) and the digest is its SHA-256 in lowercase hexadecimal.

Validation. Nothing is sorted, deduplicated or repaired; a malformed input is refused. Every ranking must be a
non-empty list or tuple of exactly ``L`` ``RankedChunk`` objects (the exact class, with a non-empty ``str`` key
that encodes as UTF-8, plain ``int`` offsets with ``0 <= start < end`` and a lowercase 64-hex hash), with no
repeated identity. Question ids are non-empty, unique ``str`` values that encode as UTF-8 (an unpaired surrogate
is refused with the fixed message, never a ``UnicodeEncodeError``). ``ranking_sha256`` additionally ties the
questions of the run to the answerable dev questions of the inputs, in order, and the pool size of the run to
the pool.

Errors. ``TypeError`` for a wrongly typed argument (a non-``int`` length, a bool included) and ``ValueError`` for
a length below 1, as in ``expected_length``. ``EvaluationDataError`` for malformed data: one fixed message per
code and a schema-defined location (``answerable[i]``, ``answerable[i].ranking[j]``). No message
holds a question id, a corpus key, an offset or a hash.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from app.rag.evaluation import engine
from app.rag.evaluation.metrics import MAX_RANK, RankedChunk
from app.rag.evaluation.pool import CandidatePool
from app.rag.evaluation.runner_inputs import DevInputs
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    SHA256_HEX_PATTERN,
    EvaluationDataError,
    Issue,
)

DIGEST_FORMAT = "e2.4-ranking-digest-1"

RANKING_QUESTIONS = "RANKING_QUESTIONS"
RANKING_EMPTY = "RANKING_EMPTY"
RANKING_LENGTH = "RANKING_LENGTH"
RANKING_CHUNK = "RANKING_CHUNK"
RANKING_DUPLICATE = "RANKING_DUPLICATE"

MESSAGES = {
    RANKING_QUESTIONS: "the ranked questions are missing, malformed, repeated or differ from the dev questions",
    RANKING_EMPTY: "a ranking is missing or empty",
    RANKING_LENGTH: "a ranking does not have min(20, pool size) chunks",
    RANKING_CHUNK: "a ranked chunk is not a well-formed ranked chunk identity",
    RANKING_DUPLICATE: "a chunk appears twice in one ranking",
}


def _fail(code: str, location: str) -> EvaluationDataError:
    return EvaluationDataError([Issue(code, location, MESSAGES[code], SEVERITY_ERROR)])


def expected_length(pool_size: int) -> int:
    """``min(20, pool_size)``: the number of chunks every dense ranking of a run must have."""
    if type(pool_size) is not int:
        raise TypeError("pool_size must be an int")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    return min(MAX_RANK, pool_size)


def _encodable(text: str) -> bool:
    """Whether ``text`` can be written as UTF-8 (an unpaired surrogate cannot)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _identity(chunk: object, where: str) -> list[Any]:
    """``[corpus_key, start, end, text_sha256]`` of a well-formed ``RankedChunk``, else a refusal."""
    if type(chunk) is not RankedChunk:
        raise _fail(RANKING_CHUNK, where)
    key, start, end, digest = chunk.corpus_key, chunk.start, chunk.end, chunk.text_sha256
    if (
        type(key) is not str
        or not key
        or not _encodable(key)
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or type(digest) is not str
        or SHA256_HEX_PATTERN.fullmatch(digest) is None
    ):
        raise _fail(RANKING_CHUNK, where)
    return [key, start, end, digest]


def ranking_payload(length: int, rows: Sequence[tuple[str, Sequence[RankedChunk]]]) -> list[Any]:
    """The validated canonical payload for ``rows`` = ``(question_id, ranking)`` pairs in dev-file order."""
    if type(length) is not int:
        raise TypeError("length must be an int")
    if length < 1:
        raise ValueError("length must be positive")
    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be a list or tuple")
    if not rows:
        raise _fail(RANKING_QUESTIONS, "answerable")
    questions: list[Any] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"answerable[{index}]"
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise _fail(RANKING_QUESTIONS, where)
        question_id, ranking = row
        if type(question_id) is not str or not question_id or not _encodable(question_id) or question_id in seen_ids:
            raise _fail(RANKING_QUESTIONS, where)
        seen_ids.add(question_id)
        if not isinstance(ranking, (list, tuple)) or not ranking:
            raise _fail(RANKING_EMPTY, f"{where}.ranking")
        if len(ranking) != length:
            raise _fail(RANKING_LENGTH, f"{where}.ranking")
        identities = [_identity(chunk, f"{where}.ranking[{rank}]") for rank, chunk in enumerate(ranking)]
        if len({tuple(identity) for identity in identities}) != len(identities):
            raise _fail(RANKING_DUPLICATE, f"{where}.ranking")
        questions.append([question_id, identities])
    return [DIGEST_FORMAT, length, questions]


def canonical_bytes(payload: Sequence[Any]) -> bytes:
    """The canonical UTF-8 JSON of a payload: compact, ``ensure_ascii=False``, ``allow_nan=False``."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def ranking_digest(length: int, rows: Sequence[tuple[str, Sequence[RankedChunk]]]) -> str:
    """The lowercase SHA-256 of the canonical payload of ``rows`` (validated first)."""
    return hashlib.sha256(canonical_bytes(ranking_payload(length, rows))).hexdigest()


def ranking_sha256(run: engine.PilotRun, inputs: DevInputs, pool: CandidatePool) -> str:
    """The D7 ranking hash of one run: its answerable dense rankings, tied to the inputs and the pool.

    The run's questions must be exactly the answerable dev questions, in order, and its pool size must be the
    pool's; the rankings are ``run.questions[*].dense`` as retained by the engine.
    """
    if not isinstance(run, engine.PilotRun):
        raise TypeError("run must be a PilotRun")
    if not isinstance(inputs, DevInputs):
        raise TypeError("inputs must be DevInputs")
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    if run.pool_size != len(pool):
        raise _fail(RANKING_LENGTH, "pool")
    if [item.question_id for item in run.questions] != [question.question_id for question in inputs.answerable]:
        raise _fail(RANKING_QUESTIONS, "answerable")
    return ranking_digest(expected_length(len(pool)), [(item.question_id, item.dense) for item in run.questions])
