"""Oracle and random baselines, and the scoring of one ranking (E1.3.5).

These are pure, evaluation-only functions. They work on ``RankedChunk`` rankings and on a
caller-supplied candidate pool, and they never touch the database, a chunk ID, an index or a
retrieval run. Building the pool (all published chunks of the manifest's documents, the same pool
the dense retrieval searches) and checking source versions and hashes belong to the runner (E1.7).
Nothing here modifies ``metrics`` or ``passages``; every metric value comes from them.

Scoring (``score_ranking``). One ranking is scored with the existing raw and distinct-passage
metrics at the cutoffs ``CUTOFFS`` (1, 3, 5, 10, 20): Hit@k, Unit Recall@k and Evidence Coverage@k,
and, once per ranking (not per cutoff), the reciprocal rank over the first ``MAX_RANK`` chunks. Hit,
Unit Recall and the reciprocal rank are scored under all three relevance rules of
``RELEVANCE_RULES`` (``any_overlap``, ``half_coverage`` and ``full_coverage``); Evidence Coverage has
no relevance rule. A ``MetricId`` names each value (family ``raw`` or ``distinct``, metric, cutoff or
``None``, rule or ``None``), and the distinct truncation flag of ``passages.is_truncated`` is kept
for every cutoff. The primary rule is ``PRIMARY_RULE``, the 50% coverage rule.

Canonical order. ``canonical_order`` sorts a pool by ``(corpus_key, start, end, text_sha256)`` (the
contract's ``start_offset`` and ``end_offset`` are the ``start`` and ``end`` of ``RankedChunk``),
comparing strings by Unicode code point, never by a database ID. Two chunks with the same sort key
are an error, so the order is total.

Oracle (``oracle_baseline``). A deterministic GREEDY REFERENCE for one answerable question over a
pool. It is not a proven upper bound: the greedy choice is not shown to be optimal for any metric.
It is built under the primary rule. At each step, among the chunks of the pool not yet chosen, it
picks the one with (a) the most newly satisfied required units, then (b) the largest exact gain in
question coverage (Evidence Coverage as ``metrics`` defines it: the mean over units of the best
span coverage by the union of the chosen chunks, computed with exact ``Fraction`` arithmetic and
never from a rounded float), then (c) the first in canonical order. It stops choosing when no chunk
gives a gain (a > 0 or b > 0), then appends the rest of the pool in canonical order, so the ranking
has ``min(MAX_RANK, pool size)`` chunks. ``chunks_needed`` is the number of greedy picks that
satisfied at least one new unit under the primary rule: the greedy count, not a proven minimum.
Every reachable unit is satisfied by such a pick before any coverage-only pick is made, and at most
20 units mean at most 20 such picks, so the rank cap never cuts the count short. It is 0 when no unit
is reachable.

A unit is unreachable under a rule when no chunk of the pool is relevant to any of its alternative
spans under that rule. The unreachable unit IDs are reported separately for each of the three rules
(the sets nest: any_overlap within half_coverage within full_coverage). Besides the scores over all
units, the oracle reports, for each rule, Unit Recall and Evidence Coverage (raw and distinct)
recomputed over the reachable units only (``None`` when none is reachable). Hit and the reciprocal
rank do not change, because an unreachable unit can never be satisfied.

Random baseline (``random_baseline``). For each of at least ``MIN_RANDOM_SEEDS`` (1,000) explicit,
unique, non-negative integer seeds, the canonical pool is shuffled by a seeded partial Fisher-Yates
and its first ``MAX_RANK`` chunks form one ranking, which is scored at all three rules. Step ``i``
swaps position ``i`` with ``i + r``, where ``r`` is uniform on ``0 .. n - i - 1`` and is drawn by
rejection sampling from the 53-bit integer ``random.Random(seed).random() * 2**53`` (exact, because
``random()`` returns an integer over 2**53). Only ``random()`` is used because Python guarantees
its sequence for a given seed across versions; ``randrange``, ``shuffle`` and ``getrandbits`` carry
no such guarantee. Rejection removes the bias that ``int(random() * m)`` has when ``m`` does not
divide 2**53; a draw is rejected with probability below ``m / 2**53``, and the number of draws used
is still fixed by the seed. The first ``MAX_RANK`` steps of a forward Fisher-Yates use only their own
draws, so the result is exactly the first ``MAX_RANK`` items of the full shuffle on the same draws.
All per-seed scores are kept; for every metric the mean and the 2.5th, 95th and 97.5th percentiles
over seeds are also given (nearest-rank, as in ``statistics``: the value at rank
``ceil(seeds * permille / 1000)`` of the sorted values; the mean is the ordinary mean). The Python
version is recorded; it does not change a result. No numerical dense-versus-random pass criterion is
defined here: judging a retrieval against these baselines belongs to E1.7.

Errors follow the evaluation package convention: ``TypeError`` for a wrong type (a ``bool`` is never
an int), ``ValueError`` for a wrong value, with the arguments' types checked before their values.
Messages are fixed text and never contain input values. Nothing here modifies its arguments.
"""
from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from app.rag.evaluation import metrics
from app.rag.evaluation.metrics import (
    CUTOFFS,
    DEFAULT_RULE,
    MAX_RANK,
    RELEVANCE_RULES,
    RankedChunk,
    _check_chunk,
    _top,
    _units,
    is_relevant,
    overlap_length,
    span_coverage,
)
from app.rag.evaluation.passages import is_truncated
from app.rag.evaluation.schema import EvidenceUnit, Question
from app.rag.evaluation.statistics import PERCENTILE_CONVENTION

PRIMARY_RULE = DEFAULT_RULE
MIN_RANDOM_SEEDS = 1000
ORACLE_METHOD = "deterministic_greedy_reference"
RANDOM_ALGORITHM = "seeded_partial_fisher_yates_rejection_v1"
RANDOM_SOURCE = "random.Random(seed).random() * 2**53, rejection sampling"
CANONICAL_KEY = ("corpus_key", "start_offset", "end_offset", "text_sha256")

FAMILIES = ("raw", "distinct")
METRIC_NAMES = ("hit", "unit_recall", "coverage", "mrr")

_SCALE = 1 << 53
_PERMILLES = (25, 950, 975)
_REACHABLE_METRICS = ("unit_recall", "coverage")

_QUESTION_MESSAGE = "question must be a Question"
_POOL_TYPE_MESSAGE = "pool must be a list or tuple of RankedChunk"
_POOL_EMPTY_MESSAGE = "the pool must not be empty"
_POOL_DUPLICATE_MESSAGE = "the pool has two chunks with the same canonical sort key"
_SEED_TYPE_MESSAGE = "a seed must be an int"
_SEEDS_TYPE_MESSAGE = "seeds must be a list or tuple of ints"
_SEED_NEGATIVE_MESSAGE = "a seed must not be negative"
_SEED_DUPLICATE_MESSAGE = "seeds must be unique"
_SEED_COUNT_MESSAGE = "too few seeds"
_BOUND_TYPE_MESSAGE = "bound must be an int"
_BOUND_RANGE_MESSAGE = "bound must be from 1 to 2**53"
_DRAW_MESSAGE = "a draw must be an int from 0 to 2**53 - 1"
_NO_METRIC_MESSAGE = "no such metric in these scores"


# ---------- result types ----------

@dataclass(frozen=True, slots=True)
class MetricId:
    """Names one metric value: ``family`` is ``raw`` or ``distinct``; ``metric`` is ``hit``,
    ``unit_recall``, ``coverage`` or ``mrr``; ``k`` is the cutoff (``None`` for ``mrr``); ``rule`` is
    the relevance rule (``None`` for ``coverage``)."""

    family: str
    metric: str
    k: int | None
    rule: str | None


@dataclass(frozen=True, slots=True)
class RankingScores:
    """The values of one ranking, aligned with ``metric_ids``, and the distinct truncation flag of
    each cutoff in ``cutoffs``. ``get`` looks one value up."""

    cutoffs: tuple[int, ...]
    metric_ids: tuple[MetricId, ...]
    values: tuple[float, ...]
    distinct_truncated: tuple[bool, ...]

    def get(self, family: str, metric: str, k: int | None = None, rule: str | None = None) -> float:
        try:
            return self.values[self.metric_ids.index(MetricId(family, metric, k, rule))]
        except ValueError:
            raise ValueError(_NO_METRIC_MESSAGE) from None


@dataclass(frozen=True, slots=True)
class ReachableScores:
    """Unit Recall and Evidence Coverage over only the units reachable under ``rule``.

    ``scores`` is ``None`` when ``unit_count`` (the number of reachable units) is 0.
    """

    rule: str
    unit_count: int
    scores: RankingScores | None


@dataclass(frozen=True, slots=True)
class OracleResult:
    """The greedy reference ranking of one question and how it scores.

    ``unreachable_by_rule`` is ``(rule, unit ids)`` for every rule in ``RELEVANCE_RULES`` order.
    ``chunks_needed`` is the greedy count under the primary rule. ``method`` states that this is a
    deterministic greedy reference and not a proven upper bound.
    """

    ranking: tuple[RankedChunk, ...]
    chunks_needed: int
    unit_ids: tuple[str, ...]
    unreachable_by_rule: tuple[tuple[str, tuple[str, ...]], ...]
    scores: RankingScores
    reachable_only: tuple[ReachableScores, ...]
    method: str


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One metric over the seeds: the mean and the nearest-rank 2.5th, 95th and 97.5th percentiles."""

    metric: MetricId
    n_seeds: int
    mean: float
    p2_5: float
    p95: float
    p97_5: float


@dataclass(frozen=True, slots=True)
class ReproducibilityRecord:
    """What is needed to reproduce a random baseline (the seeds themselves are in the result)."""

    python_version: str
    algorithm: str
    random_source: str
    canonical_key: tuple[str, ...]
    pool_size: int
    rank_cap: int
    percentile_convention: str


@dataclass(frozen=True, slots=True)
class RandomBaselineResult:
    """The random baseline of one question: every seed's scores, in seed order, and their summaries."""

    seeds: tuple[int, ...]
    per_seed: tuple[RankingScores, ...]
    summaries: tuple[MetricSummary, ...]
    reproducibility: ReproducibilityRecord


# ---------- metric ids and scoring ----------

def _metric_ids(rules: Sequence[str], names: Sequence[str]) -> tuple[MetricId, ...]:
    ids: list[MetricId] = []
    for family in FAMILIES:
        for name in ("hit", "unit_recall"):
            if name in names:
                ids.extend(MetricId(family, name, k, rule) for rule in rules for k in CUTOFFS)
        if "coverage" in names:
            ids.extend(MetricId(family, "coverage", k, None) for k in CUTOFFS)
        if "mrr" in names:
            ids.extend(MetricId(family, "mrr", None, rule) for rule in rules)
    return tuple(ids)


_FULL_IDS = _metric_ids(RELEVANCE_RULES, METRIC_NAMES)
_REACHABLE_IDS = {rule: _metric_ids((rule,), _REACHABLE_METRICS) for rule in RELEVANCE_RULES}


def _value(question: Question, ranking: Sequence[RankedChunk], metric: MetricId) -> float:
    raw = metric.family == "raw"
    if metric.metric == "hit":
        function = metrics.hit_at_k if raw else metrics.distinct_hit_at_k
        return float(function(question, ranking, metric.k, metric.rule))
    if metric.metric == "unit_recall":
        function = metrics.unit_recall_at_k if raw else metrics.distinct_unit_recall_at_k
        return float(function(question, ranking, metric.k, metric.rule))
    if metric.metric == "coverage":
        function = metrics.evidence_coverage_at_k if raw else metrics.distinct_evidence_coverage_at_k
        return float(function(question, ranking, metric.k))
    function = metrics.reciprocal_rank if raw else metrics.distinct_reciprocal_rank
    return float(function(question, ranking, metric.rule))


def _score(
    question: Question, ranking: Sequence[RankedChunk], ids: tuple[MetricId, ...]
) -> RankingScores:
    values = tuple(_value(question, ranking, metric) for metric in ids)
    flags = tuple(is_truncated(ranking, k) for k in CUTOFFS)
    return RankingScores(CUTOFFS, ids, values, flags)


def score_ranking(question: Question, ranking: Sequence[RankedChunk]) -> RankingScores:
    """Score one ranking of one answerable question with every raw and distinct metric.

    The values are in the order of ``RankingScores.metric_ids``: for each family, Hit@k and Unit
    Recall@k under each of the three rules (rule by rule, then by cutoff), Evidence Coverage@k, and
    the reciprocal rank under each rule. Only the first ``MAX_RANK`` chunks are used.
    """
    if not isinstance(question, Question):
        raise TypeError(_QUESTION_MESSAGE)
    _top(ranking, MAX_RANK)
    _units(question)
    return _score(question, ranking, _FULL_IDS)


# ---------- the pool and canonical order ----------

def _check_pool_types(pool: object) -> None:
    if not isinstance(pool, (list, tuple)):
        raise TypeError(_POOL_TYPE_MESSAGE)
    for chunk in pool:
        _check_chunk(chunk)


def _sort_key(chunk: RankedChunk) -> tuple[str, int, int, str]:
    return (chunk.corpus_key, chunk.start, chunk.end, chunk.text_sha256)


def canonical_order(pool: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]:
    """The pool sorted by ``(corpus_key, start, end, text_sha256)``.

    Raises ``TypeError`` for a pool that is not a list or tuple of ``RankedChunk``, and
    ``ValueError`` for an empty pool or two chunks with the same sort key.
    """
    _check_pool_types(pool)
    if not pool:
        raise ValueError(_POOL_EMPTY_MESSAGE)
    ordered = sorted(pool, key=_sort_key)
    for earlier, later in zip(ordered, ordered[1:]):
        if _sort_key(earlier) == _sort_key(later):
            raise ValueError(_POOL_DUPLICATE_MESSAGE)
    return tuple(ordered)


# ---------- oracle ----------

def _coverage(units: Sequence[EvidenceUnit], chunks: list[RankedChunk]) -> Fraction:
    """Evidence Coverage of ``chunks`` as an exact fraction: the mean over units of the best span
    coverage. The same definition as ``metrics.evidence_coverage_at_k``, before its final ``float``."""
    total = sum((max(span_coverage(chunks, span) for span in unit.spans) for unit in units), Fraction(0))
    return total / len(units)


def _reachable(unit: EvidenceUnit, contenders: Sequence[RankedChunk], rule: str) -> bool:
    return any(is_relevant(chunk, span, rule) for span in unit.spans for chunk in contenders)


def _greedy(
    units: Sequence[EvidenceUnit], contenders: Sequence[RankedChunk], canonical: Sequence[RankedChunk]
) -> tuple[tuple[RankedChunk, ...], int]:
    limit = min(MAX_RANK, len(canonical))
    chosen: list[RankedChunk] = []
    taken: set[RankedChunk] = set()
    satisfied: set[int] = set()
    coverage = Fraction(0)
    needed = 0
    while len(chosen) < limit:
        best: RankedChunk | None = None
        best_key: tuple[int, Fraction] | None = None
        best_new: list[int] = []
        best_gain = Fraction(0)
        for candidate in contenders:  # canonical order, so a strictly greater key keeps the first tie
            if candidate in taken:
                continue
            new = [
                index
                for index, unit in enumerate(units)
                if index not in satisfied and any(is_relevant(candidate, span, PRIMARY_RULE) for span in unit.spans)
            ]
            gain = _coverage(units, chosen + [candidate]) - coverage
            if not new and gain == 0:
                continue
            key = (len(new), gain)
            if best_key is None or key > best_key:
                best, best_key, best_new, best_gain = candidate, key, new, gain
        if best is None:
            break
        chosen.append(best)
        taken.add(best)
        satisfied.update(best_new)
        coverage += best_gain
        if best_new:
            needed += 1
    for candidate in canonical:
        if len(chosen) >= limit:
            break
        if candidate not in taken:
            chosen.append(candidate)
    return tuple(chosen), needed


def oracle_baseline(question: Question, pool: Sequence[RankedChunk]) -> OracleResult:
    """The greedy reference ranking of ``question`` over ``pool`` and its scores (see the module docs).

    ``question`` must be answerable with at least one unit. Raises ``TypeError`` or ``ValueError``
    for invalid input.
    """
    if not isinstance(question, Question):
        raise TypeError(_QUESTION_MESSAGE)
    _check_pool_types(pool)
    units = _units(question)
    canonical = canonical_order(pool)

    spans = [span for unit in units for span in unit.spans]
    contenders = tuple(
        chunk
        for chunk in canonical
        if any(
            chunk.corpus_key == span.corpus_key
            and overlap_length((chunk.start, chunk.end), (span.start, span.end)) > 0
            for span in spans
        )
    )  # a chunk that shares no position with any gold span has no gain and is never relevant
    reachable = {rule: [_reachable(unit, contenders, rule) for unit in units] for rule in RELEVANCE_RULES}
    unreachable_by_rule = tuple(
        (rule, tuple(unit.unit_id for unit, ok in zip(units, reachable[rule]) if not ok))
        for rule in RELEVANCE_RULES
    )

    ranking, needed = _greedy(units, contenders, canonical)
    scores = _score(question, ranking, _FULL_IDS)
    reachable_only = []
    for rule in RELEVANCE_RULES:
        kept = tuple(unit for unit, ok in zip(units, reachable[rule]) if ok)
        if kept:
            subset = _score(replace(question, units=kept), ranking, _REACHABLE_IDS[rule])
            reachable_only.append(ReachableScores(rule, len(kept), subset))
        else:
            reachable_only.append(ReachableScores(rule, 0, None))
    return OracleResult(
        ranking, needed, tuple(unit.unit_id for unit in units), unreachable_by_rule, scores,
        tuple(reachable_only), ORACLE_METHOD,
    )


# ---------- random baseline ----------

def _draw53(rng: random.Random) -> int:
    """A uniform integer from 0 to 2**53 - 1: ``random()`` is such an integer over 2**53, and the
    product with a power of two is exact."""
    return int(rng.random() * _SCALE)


def _bounded(draw: Callable[[], int], bound: int) -> int:
    """A uniform integer from 0 to ``bound - 1`` from 53-bit draws, by rejection sampling.

    ``bound`` must be an int from 1 to 2**53 (so the loop always has an accepting draw). A draw is
    kept only if it is below the largest multiple of ``bound`` that fits in 2**53; the rest would
    favour the smaller results.
    """
    if type(bound) is not int:
        raise TypeError(_BOUND_TYPE_MESSAGE)
    if not 1 <= bound <= _SCALE:
        raise ValueError(_BOUND_RANGE_MESSAGE)
    limit = _SCALE - _SCALE % bound
    while True:
        value = draw()
        if type(value) is not int or not 0 <= value < _SCALE:
            raise ValueError(_DRAW_MESSAGE)
        if value < limit:
            return value % bound


def _partial_shuffle(canonical: Sequence[RankedChunk], seed: int) -> tuple[RankedChunk, ...]:
    items = list(canonical)
    size = len(items)
    rng = random.Random(seed)

    def draw() -> int:
        return _draw53(rng)

    for index in range(min(MAX_RANK, size - 1)):
        other = index + _bounded(draw, size - index)
        items[index], items[other] = items[other], items[index]
    return tuple(items[:MAX_RANK])


def _check_seed_type(seed: object) -> None:
    if type(seed) is not int:
        raise TypeError(_SEED_TYPE_MESSAGE)


def random_ranking(pool: Sequence[RankedChunk], seed: int) -> tuple[RankedChunk, ...]:
    """The first ``MAX_RANK`` chunks of the seeded shuffle of the canonical pool (all of a smaller pool)."""
    _check_pool_types(pool)
    _check_seed_type(seed)
    canonical = canonical_order(pool)
    if seed < 0:
        raise ValueError(_SEED_NEGATIVE_MESSAGE)
    return _partial_shuffle(canonical, seed)


def _nearest_rank(count: int, permille: int) -> int:
    """The 1-based nearest rank ``ceil(count * permille / 1000)``, by integer arithmetic."""
    return -(-count * permille // 1000)


def _summary(metric: MetricId, column: list[float]) -> MetricSummary:
    ordered = sorted(column)
    count = len(ordered)
    low, mid, high = (ordered[_nearest_rank(count, permille) - 1] for permille in _PERMILLES)
    return MetricSummary(metric, count, math.fsum(ordered) / count, low, mid, high)


def random_baseline(
    question: Question, pool: Sequence[RankedChunk], seeds: Sequence[int]
) -> RandomBaselineResult:
    """The random baseline of ``question`` over ``pool`` for every seed (see the module docs).

    ``seeds`` must be a list or tuple of at least ``MIN_RANDOM_SEEDS`` unique, non-negative ints.
    """
    if not isinstance(question, Question):
        raise TypeError(_QUESTION_MESSAGE)
    _check_pool_types(pool)
    if not isinstance(seeds, (list, tuple)):
        raise TypeError(_SEEDS_TYPE_MESSAGE)
    for seed in seeds:
        _check_seed_type(seed)
    _units(question)
    canonical = canonical_order(pool)
    if any(seed < 0 for seed in seeds):
        raise ValueError(_SEED_NEGATIVE_MESSAGE)
    if len(set(seeds)) != len(seeds):
        raise ValueError(_SEED_DUPLICATE_MESSAGE)
    if len(seeds) < MIN_RANDOM_SEEDS:
        raise ValueError(_SEED_COUNT_MESSAGE)

    per_seed = tuple(_score(question, _partial_shuffle(canonical, seed), _FULL_IDS) for seed in seeds)
    summaries = tuple(
        _summary(metric, [row.values[index] for row in per_seed]) for index, metric in enumerate(_FULL_IDS)
    )
    version = sys.version_info
    record = ReproducibilityRecord(
        f"{version.major}.{version.minor}.{version.micro}", RANDOM_ALGORITHM, RANDOM_SOURCE,
        CANONICAL_KEY, len(canonical), MAX_RANK, PERCENTILE_CONVENTION,
    )
    return RandomBaselineResult(tuple(seeds), per_seed, summaries, record)
