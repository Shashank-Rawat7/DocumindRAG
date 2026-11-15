"""The two-pass dense run and its determinism gate (E2.4d). Pure orchestration: no file, no environment, no CLI.

``run_two_passes`` executes the complete ``engine.run_pilot`` twice over the same immutable inputs, candidate
pool, seeds and configuration. Every pass gets
 * a fresh retriever (``new_retriever()`` is called again; for the real database it builds a new closure over the
   SAME connection, so both passes read inside the same read-only REPEATABLE READ transaction),
 * a fresh ``ScoreRecorder`` (a recorder is single-use: ``take()`` seals it),
 * a fresh copy of the adjudication mapping,
 * its own, separate plain retriever for the unanswerable questions (never the recorder).
The embedder, the inputs and the pool are shared on purpose: they are what must not change between the passes.

Gate (D7): the ranking hash of ``ranking_hash.ranking_sha256`` is computed for each pass from its own ``PilotRun``;
if the two differ nothing is returned (``EvaluationDataError`` with ``BASELINE_RANKINGS_DIFFER``). No ranking is
sorted, repaired, normalised or deduplicated, and nothing is retried. The score digests are only compared: the
score-diagnostics section records whether they are equal (a diagnostic, never a gate).
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.evaluation import engine
from app.rag.evaluation import ranking_hash as rh
from app.rag.evaluation import score_diagnostics as sd
from app.rag.evaluation.pool import CandidatePool
from app.rag.evaluation.runner_inputs import DevInputs
from app.rag.evaluation.schema import SEVERITY_ERROR, EvaluationDataError, Issue

PASSES = 2
BASELINE_RANKINGS_DIFFER = "BASELINE_RANKINGS_DIFFER"
_MESSAGES = {
    BASELINE_RANKINGS_DIFFER: "the dense rankings of the two passes are not identical; no report may be written",
}


def _rankings_differ() -> EvaluationDataError:
    return EvaluationDataError(
        [Issue(BASELINE_RANKINGS_DIFFER, "ranking_determinism", _MESSAGES[BASELINE_RANKINGS_DIFFER], SEVERITY_ERROR)]
    )


@dataclass(frozen=True, slots=True)
class PassResult:
    """Everything one complete pass produced. ``calls`` and ``unanswerable`` hold database ids: memory only."""

    run: engine.PilotRun
    calls: tuple[sd.RecordedCall, ...]
    unanswerable: tuple[sd.QuestionScores, ...]
    ranking_sha256: str


@dataclass(frozen=True, slots=True)
class TwoPassResult:
    """The first pass (the run that is reported), the second pass, and the verified determinism facts."""

    first: PassResult
    second: PassResult
    ranking_sha256: str
    score_diagnostics: dict[str, Any]


def run_pass(
    new_retriever: Callable[[], Callable[[str], Sequence[Any]]],
    inputs: DevInputs,
    pool: CandidatePool,
    *,
    seeds: Sequence[int],
    bootstrap_seed: int,
    resamples: int,
    adjudicated: Mapping[str, Collection[str]] | None,
) -> PassResult:
    """One complete pass: a fresh recorder over a fresh retriever, the full ``run_pilot``, then the unanswerable
    scores through another fresh plain retriever."""
    recorder = sd.ScoreRecorder(new_retriever())
    run = engine.run_pilot(
        inputs, pool, recorder, seeds=seeds, bootstrap_seed=bootstrap_seed, resamples=resamples,
        adjudicated_unreachable=None if not adjudicated else {key: list(units) for key, units in adjudicated.items()},
    )
    calls = recorder.take()
    unanswerable = sd.collect_unanswerable_scores(inputs, pool, new_retriever())
    return PassResult(run, calls, unanswerable, rh.ranking_sha256(run, inputs, pool))


def run_two_passes(
    new_retriever: Callable[[], Callable[[str], Sequence[Any]]],
    inputs: DevInputs,
    pool: CandidatePool,
    *,
    seeds: Sequence[int],
    bootstrap_seed: int,
    resamples: int,
    adjudicated: Mapping[str, Collection[str]] | None = None,
) -> TwoPassResult:
    """Run both passes, require identical D7 ranking hashes, and build the score-diagnostics section.

    Raises ``EvaluationDataError`` (``BASELINE_RANKINGS_DIFFER``) if the hashes differ, before anything is built.
    """
    if not callable(new_retriever):
        raise TypeError("new_retriever must be callable")
    seeds = tuple(seeds)  # one immutable tuple, shared by both passes
    kwargs = {"seeds": seeds, "bootstrap_seed": bootstrap_seed, "resamples": resamples, "adjudicated": adjudicated}
    first = run_pass(new_retriever, inputs, pool, **kwargs)
    second = run_pass(new_retriever, inputs, pool, **kwargs)
    if first.ranking_sha256 != second.ranking_sha256:
        raise _rankings_differ()
    # The second pass's own diagnostics are built (and so fully validated) only to obtain its score digest.
    second_digest = sd.build_score_diagnostics(
        second.run, inputs, pool, second.calls, second.unanswerable
    )["scores_sha256"]
    section = sd.build_score_diagnostics(
        first.run, inputs, pool, first.calls, first.unanswerable, second_pass_scores_sha256=second_digest
    )
    return TwoPassResult(first, second, first.ranking_sha256, section)
