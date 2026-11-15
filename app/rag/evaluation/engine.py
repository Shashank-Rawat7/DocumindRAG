"""The pure evaluation run engine of the development-split pilot (E1.7.3).

It connects ``runner_inputs`` (the gated dev questions), ``pool`` (the verified candidate pool and the
dense conversion), ``baselines`` (scoring, oracle, random), ``metrics`` and ``aggregation``. It opens no
database, reads no file, imports no retrieval code and serialises nothing: a caller-supplied
``retriever`` stands in for dense retrieval, so everything here runs on fake data in tests.

One path for everything. For every answerable dev question, in dev-file order, the engine builds three
rankings over the SAME ``DevInputs`` questions and the SAME ``CandidatePool``:

* dense: ``retriever(query_text)`` returns the dense ranking (``RetrievedChunk``-like objects, best first).
  The retriever sees only the query text, never a label. ``pool.convert_retrieved`` checks every chunk
  against the pool and converts it to ``RankedChunk``. The ranking must have exactly
  ``min(MAX_RANK, len(pool))`` chunks, because exact search over the pool always fills the top 20;
  anything else is refused as a malformed ranking. A retriever error other than ``EvaluationDataError``
  propagates unchanged;
* oracle: ``baselines.oracle_baseline`` (a deterministic greedy reference, not a proven upper bound);
* random: ``baselines.random_baseline`` for the supplied seeds (at least 1,000 unique, non-negative).

Dense and oracle are scored by ``baselines.score_ranking`` / its identical internal scorer, so every
ranking yields the same ``MetricId`` list (checked); each metric value is a series
``<family>_<metric>[_at_<k>][_<rule>]``. Observations are ``aggregation.QuestionObservation`` (retrieval
type from ``structure.derive_structure``; ``truncated_at`` from the distinct truncation flags). The random
baseline's observation per question holds the MEAN over seeds of each metric, and its ``truncated_at``
lists the cutoffs at which at least one seed's ranking was truncated.

Two series declarations. ``series`` has Hit as ``binary`` and everything else ``continuous``; it fits dense
and oracle. A random observation holds means (a hit mean is not 0 or 1), so random observations, and every
comparison that includes them, use ``series_continuous`` (all series continuous; the paired bootstrap then
works on the per-question differences).

Reports. ``aggregate`` is applied to dense, oracle and random, and ``compare`` to random (baseline) and
dense (candidate). The comparison is only reported: no pass/fail threshold is approved or applied. The
random baseline is also summarised over seeds (``random_macro``): for each seed the mean over all answerable
questions, then the mean and the 2.5th, 95th and 97.5th percentiles over seeds.

Oracle validity (``OracleValidity``), judged under the primary relevance rule (``half_coverage``) and the
raw family only: a question is valid when every unreachable unit is explicitly adjudicated (the caller
passes ``adjudicated_unreachable``; a listed unit must really be unreachable) and, when it has reachable
units, Hit@1 is 1.0 and Unit Recall@20 over the reachable units is 1.0. Unreachable units under the other
rules are reported, not gated. Nothing is raised for a failed validity check; the verdict is in the result
(``ok``), because deciding what to do about it is not the engine's job.

Unanswerable questions are listed separately (id, language, type) and are not retrieved, scored or
aggregated: no answerable retrieval metric is claimed for them.

Errors: ``TypeError`` for a wrongly typed argument, ``EvaluationDataError`` (fixed messages, no text) for
malformed rankings, a pool that does not match the inputs and unknown adjudications.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.evaluation import baselines
from app.rag.evaluation import statistics as stats
from app.rag.evaluation.aggregation import (
    BINARY,
    CONTINUOUS,
    ComparisonReport,
    QuestionObservation,
    Report,
    SeriesSpec,
    aggregate,
    compare,
)
from app.rag.evaluation.baselines import (
    MIN_RANDOM_SEEDS,
    PRIMARY_RULE,
    MetricId,
    MetricSummary,
    OracleResult,
    RandomBaselineResult,
    RankingScores,
)
from app.rag.evaluation.metrics import CUTOFFS, MAX_RANK, RankedChunk
from app.rag.evaluation.pool import CandidatePool, convert_retrieved
from app.rag.evaluation.runner_inputs import DevInputs
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    EvaluationDataError,
    Issue,
    Question,
)
from app.rag.evaluation.structure import derive_structure

ENGINE_POOL_MISMATCH = "ENGINE_POOL_MISMATCH"
ENGINE_RETRIEVER_RETURN = "ENGINE_RETRIEVER_RETURN"
ENGINE_RANKING_LENGTH = "ENGINE_RANKING_LENGTH"
ENGINE_NO_ANSWERABLE = "ENGINE_NO_ANSWERABLE"
ENGINE_ADJUDICATION = "ENGINE_ADJUDICATION"
ENGINE_SERIES_MISMATCH = "ENGINE_SERIES_MISMATCH"

_MESSAGES = {
    ENGINE_POOL_MISMATCH: "the candidate pool does not cover exactly the corpus documents of the inputs",
    ENGINE_RETRIEVER_RETURN: "the retriever must return a list or tuple of chunks",
    ENGINE_RANKING_LENGTH: "a dense ranking must hold exactly min(20, pool size) chunks",
    ENGINE_NO_ANSWERABLE: "the dev split has no answerable question",
    ENGINE_ADJUDICATION: "an adjudicated unit is unknown or is not unreachable under the primary rule",
    ENGINE_SERIES_MISMATCH: "the rankings did not produce the same metric list",
}

Retriever = Callable[[str], Sequence[Any]]


def _issue(code: str, location: str) -> Issue:
    return Issue(code, location, _MESSAGES[code], SEVERITY_ERROR)


# ---------- series ----------

def series_name(metric: MetricId) -> str:
    """``<family>_<metric>`` plus ``_at_<k>`` and ``_<rule>`` where the metric has them."""
    name = f"{metric.family}_{metric.metric}"
    if metric.k is not None:
        name += f"_at_{metric.k}"
    if metric.rule is not None:
        name += f"_{metric.rule}"
    return name


def _series(ids: Sequence[MetricId], *, hit_binary: bool) -> tuple[SeriesSpec, ...]:
    return tuple(
        SeriesSpec(series_name(m), BINARY if hit_binary and m.metric == "hit" else CONTINUOUS) for m in ids
    )


def default_random_seeds(count: int = MIN_RANDOM_SEEDS) -> tuple[int, ...]:
    """The seeds ``0 .. count - 1`` (at least ``MIN_RANDOM_SEEDS``), so a run needs no hidden state."""
    if type(count) is not int:
        raise TypeError("count must be an int")
    if count < MIN_RANDOM_SEEDS:
        raise ValueError("too few seeds")
    return tuple(range(count))


# ---------- result types ----------

@dataclass(frozen=True, slots=True)
class OracleCheck:
    """The oracle validity of one question under the primary rule (raw family)."""

    question_id: str
    unit_ids: tuple[str, ...]
    unreachable_by_rule: tuple[tuple[str, tuple[str, ...]], ...]
    reachable_units: int
    adjudicated: tuple[str, ...]
    unadjudicated: tuple[str, ...]
    hit_at_1: float | None
    unit_recall_at_20: float | None
    ok: bool


@dataclass(frozen=True, slots=True)
class OracleValidity:
    checks: tuple[OracleCheck, ...]
    rule: str
    ok: bool
    method: str


@dataclass(frozen=True, slots=True)
class QuestionRun:
    """The three rankings of one answerable question and what they scored."""

    question_id: str
    retrieval_type: str
    dense: tuple[RankedChunk, ...]
    dense_scores: RankingScores
    oracle: OracleResult
    random: RandomBaselineResult


@dataclass(frozen=True, slots=True)
class UnansweredQuestion:
    """An unanswerable dev question, listed apart. It is not retrieved or scored."""

    question_id: str
    query_language: str
    unanswerable_type: str | None


@dataclass(frozen=True, slots=True)
class PilotRun:
    """Everything one run produced, in dev-file order. Nothing is serialised here."""

    manifest_sha256: str
    dev_dataset_sha256: str
    pool_sha256: str
    pool_size: int
    seeds: tuple[int, ...]
    cutoffs: tuple[int, ...]
    series: tuple[SeriesSpec, ...]
    series_continuous: tuple[SeriesSpec, ...]
    questions: tuple[QuestionRun, ...]
    unanswerable: tuple[UnansweredQuestion, ...]
    dense_observations: tuple[QuestionObservation, ...]
    oracle_observations: tuple[QuestionObservation, ...]
    random_observations: tuple[QuestionObservation, ...]
    random_macro: tuple[MetricSummary, ...]
    oracle_validity: OracleValidity
    dense_report: Report
    oracle_report: Report
    random_report: Report
    dense_vs_random: ComparisonReport


# ---------- building blocks ----------

def _truncated(scores: RankingScores) -> tuple[int, ...]:
    return tuple(k for k, flag in zip(scores.cutoffs, scores.distinct_truncated) if flag)


def _observation(question_id: str, retrieval_type: str, series: Sequence[SeriesSpec], values: Sequence[float],
                 truncated: tuple[int, ...]) -> QuestionObservation:
    return QuestionObservation(
        question_id, retrieval_type, tuple((spec.name, float(value)) for spec, value in zip(series, values)), truncated
    )


def _check_pool(inputs: DevInputs, pool: CandidatePool) -> None:
    keys = {d.corpus_key for d in inputs.manifest.documents}
    if set(pool.document_map.id_by_key) != keys or {c.corpus_key for c in pool.ranked} != keys:
        raise EvaluationDataError([_issue(ENGINE_POOL_MISMATCH, "pool")])


def _dense_ranking(pool: CandidatePool, retriever: Retriever, question: Question, index: int) -> tuple[RankedChunk, ...]:
    where = f"answerable[{index}]"
    retrieved = retriever(question.question)
    if not isinstance(retrieved, (list, tuple)):
        raise EvaluationDataError([_issue(ENGINE_RETRIEVER_RETURN, where)])
    try:
        ranking = convert_retrieved(pool, retrieved)
    except EvaluationDataError as error:
        raise EvaluationDataError(
            [Issue(i.code, f"{where}.{i.location}", i.message, i.severity) for i in error.issues]
        ) from None
    if len(ranking) != min(MAX_RANK, len(pool)):
        raise EvaluationDataError([_issue(ENGINE_RANKING_LENGTH, where)])
    return ranking


def _oracle_check(
    question: Question, oracle: OracleResult, adjudicated: Collection[str]
) -> OracleCheck:
    unreachable = dict(oracle.unreachable_by_rule)[PRIMARY_RULE]
    covered = tuple(u for u in unreachable if u in adjudicated)
    open_ = tuple(u for u in unreachable if u not in adjudicated)
    reachable = len(oracle.unit_ids) - len(unreachable)
    hit = recall = None
    if reachable:
        hit = oracle.scores.get("raw", "hit", 1, PRIMARY_RULE)
        reachable_scores = next(r for r in oracle.reachable_only if r.rule == PRIMARY_RULE).scores
        recall = reachable_scores.get("raw", "unit_recall", MAX_RANK, PRIMARY_RULE)
    ok = not open_ and (not reachable or (hit == 1.0 and recall == 1.0))
    return OracleCheck(question.question_id, oracle.unit_ids, oracle.unreachable_by_rule, reachable,
                       covered, open_, hit, recall, ok)


def _check_adjudication(
    adjudicated: Mapping[str, Collection[str]], oracles: Mapping[str, OracleResult]
) -> None:
    issues = []
    for position, (question_id, units) in enumerate(sorted(adjudicated.items())):
        oracle = oracles.get(question_id)
        unreachable = set(dict(oracle.unreachable_by_rule)[PRIMARY_RULE]) if oracle else set()
        if oracle is None or not set(units) <= unreachable:
            issues.append(_issue(ENGINE_ADJUDICATION, f"adjudicated_unreachable[{position}]"))
    if issues:
        raise EvaluationDataError(issues)


def _macro(runs: Sequence[QuestionRun], ids: Sequence[MetricId]) -> tuple[MetricSummary, ...]:
    """Per seed, the mean over questions; then mean and percentiles over seeds."""
    seeds = len(runs[0].random.seeds)
    summaries = []
    for column_index, metric in enumerate(ids):
        column = [
            math.fsum(run.random.per_seed[s].values[column_index] for run in runs) / len(runs)
            for s in range(seeds)
        ]
        summaries.append(baselines._summary(metric, column))  # same summary and percentile rule as per question
    return tuple(summaries)


# ---------- the engine ----------

def run_pilot(
    inputs: DevInputs,
    pool: CandidatePool,
    retriever: Retriever,
    *,
    seeds: Sequence[int],
    bootstrap_seed: int,
    resamples: int = stats.DEFAULT_RESAMPLES,
    adjudicated_unreachable: Mapping[str, Collection[str]] | None = None,
) -> PilotRun:
    """Run dense, oracle and random over the answerable dev questions and aggregate them.

    ``seeds`` are the random baseline's explicit seeds (``default_random_seeds`` gives 0..999).
    ``bootstrap_seed`` and ``resamples`` go unchanged to every ``aggregate`` and ``compare``.
    ``adjudicated_unreachable`` maps a question id to the units a human adjudicated as unreachable.
    """
    if not isinstance(inputs, DevInputs):
        raise TypeError("inputs must be DevInputs")
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    if not callable(retriever):
        raise TypeError("retriever must be callable")
    if adjudicated_unreachable is not None and not isinstance(adjudicated_unreachable, Mapping):
        raise TypeError("adjudicated_unreachable must be a Mapping or None")
    adjudicated = {} if adjudicated_unreachable is None else adjudicated_unreachable
    for units in adjudicated.values():
        if isinstance(units, (str, bytes)) or not isinstance(units, Collection):
            raise TypeError("adjudicated units must be a collection of unit ids")
    if not isinstance(seeds, (list, tuple)):
        raise TypeError("seeds must be a list or tuple of ints")
    seed_list = tuple(seeds)

    _check_pool(inputs, pool)
    if not inputs.answerable:
        raise EvaluationDataError([_issue(ENGINE_NO_ANSWERABLE, "dev")])

    runs: list[QuestionRun] = []
    for index, question in enumerate(inputs.answerable):
        dense = _dense_ranking(pool, retriever, question, index)
        dense_scores = baselines.score_ranking(question, dense)
        oracle = baselines.oracle_baseline(question, pool.ranked)
        random_result = baselines.random_baseline(question, pool.ranked, seed_list)
        if not (dense_scores.metric_ids == oracle.scores.metric_ids == random_result.per_seed[0].metric_ids):
            raise EvaluationDataError([_issue(ENGINE_SERIES_MISMATCH, f"answerable[{index}]")])
        runs.append(QuestionRun(
            question.question_id, derive_structure(question).retrieval_type, dense, dense_scores, oracle, random_result,
        ))

    ids = runs[0].dense_scores.metric_ids
    series = _series(ids, hit_binary=True)
    series_continuous = _series(ids, hit_binary=False)

    dense_obs = tuple(
        _observation(r.question_id, r.retrieval_type, series, r.dense_scores.values, _truncated(r.dense_scores))
        for r in runs
    )
    oracle_obs = tuple(
        _observation(r.question_id, r.retrieval_type, series, r.oracle.scores.values, _truncated(r.oracle.scores))
        for r in runs
    )
    random_obs = tuple(
        _observation(
            r.question_id, r.retrieval_type, series_continuous,
            [summary.mean for summary in r.random.summaries],
            tuple(sorted({k for row in r.random.per_seed for k in _truncated(row)})),
        )
        for r in runs
    )

    oracles = {r.question_id: r.oracle for r in runs}
    _check_adjudication(adjudicated, oracles)
    checks = tuple(
        _oracle_check(q, oracles[q.question_id], adjudicated.get(q.question_id, ())) for q in inputs.answerable
    )
    validity = OracleValidity(checks, PRIMARY_RULE, all(c.ok for c in checks), baselines.ORACLE_METHOD)

    return PilotRun(
        manifest_sha256=inputs.manifest_sha256,
        dev_dataset_sha256=inputs.dev_dataset_sha256,
        pool_sha256=pool.pool_sha256,
        pool_size=len(pool),
        seeds=seed_list,
        cutoffs=CUTOFFS,
        series=series,
        series_continuous=series_continuous,
        questions=tuple(runs),
        unanswerable=tuple(
            UnansweredQuestion(q.question_id, q.query_language, q.unanswerable_type) for q in inputs.unanswerable
        ),
        dense_observations=dense_obs,
        oracle_observations=oracle_obs,
        random_observations=random_obs,
        random_macro=_macro(runs, ids),
        oracle_validity=validity,
        dense_report=aggregate(dense_obs, series, bootstrap_seed, resamples),
        oracle_report=aggregate(oracle_obs, series, bootstrap_seed, resamples),
        random_report=aggregate(random_obs, series_continuous, bootstrap_seed, resamples),
        dense_vs_random=compare(random_obs, dense_obs, series_continuous, bootstrap_seed, resamples),
    )
