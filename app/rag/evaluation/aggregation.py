"""Aggregation and in-memory reporting of per-question evaluation observations (E1.3.4.d).

This module is a pure aggregation layer. It takes observations that were already computed
(one ``QuestionObservation`` per question and retrieval version) and summarises them overall and
for each of the four retrieval types. It never reads a ``Question`` or a ranking, never computes
a retrieval metric, never builds an observation, and never runs or orchestrates an evaluation:
those belong to the caller. All statistics come from ``statistics.py`` (``wilson_interval``,
``bootstrap_mean_interval``, ``paired_bootstrap_interval``); nothing is recalculated here.

What is reported
  * ``aggregate`` gives, for the overall set and for each retrieval type in ``RETRIEVAL_TYPES``
    (always all four, always in that order, an empty one with ``n = 0``), the count of questions,
    the per-cutoff truncation counts, and one ``SeriesResult`` per declared series.
  * ``compare`` gives the same layout for candidate minus baseline: one ``PairedSeriesResult``
    per series, from the paired bootstrap. Overlapping or separate Wilson intervals are never
    used as a comparison.

Series. The caller declares ``SeriesSpec(name, kind)`` entries. Names are opaque unique strings
(for example ``"hit_at_5"`` or ``"distinct_unit_recall_at_3"``); the kind picks the method. A
``binary`` series (values exactly 0 or 1, such as Hit@k) gets a Wilson interval; a
``continuous`` series (values from 0 to 1) gets the ordinary question-level bootstrap. Every
observation must carry exactly one value for every declared series, no more and no fewer.

Status of a series result
  * ``ok``: the rate or mean and the interval are present.
  * ``not_estimable``: valid data from which no interval can be made. The overall set with
    ``n = 0``; a continuous series with ``n = 1`` (the observed mean is kept, the interval is
    not); a paired series with ``n < 2`` (the observed mean difference is kept when ``n = 1``).
    A binary series with ``n = 1`` is ``ok``, since a Wilson interval exists for one trial.
  * ``suppressed``: a retrieval-type slice with fewer than ``MIN_SLICE_QUESTIONS`` (10)
    questions, empty ones included. It carries ``n``, the truncation counts and the individual
    outcomes, and nothing else: no rate, mean, interval or paired difference. The overall set is
    not subject to this rule.
  A non-estimable or suppressed result never makes the report fail; invalid data always does,
  including in a slice that would be suppressed.

Truncation. ``truncated_at`` of an observation lists the cutoffs at which the distinct ranking
had fewer than ``k`` distinct passage representatives in its first ``MAX_RANK`` chunks (the
``is_truncated`` flag of E1.3.3). The report gives, per slice and for every declared cutoff
(default ``CUTOFFS``, zero counts included), the number of questions truncated at that cutoff.
The count only annotates a slice. It never changes a value or removes a question. It concerns
the distinct ``@k`` metrics only; the raw metrics and distinct reciprocal rank are not affected.

Reproducibility. ``seed`` and ``resamples`` are explicit, are validated here (so a report whose
slices are all suppressed still rejects a bad setting), and are passed unchanged to every
interval of the report. The same seed is reused for every series and slice. No global random
state is used. Questions about the same source document are probably correlated, so every
interval is likely narrower than the real uncertainty (see ``statistics.py``).

Comparison. ``compare`` requires the two versions to hold exactly the same question ids, with the
same retrieval type for each id, and never compares an intersection. Matching ids do not prove
that the labels behind them are the same: the runner must verify that both versions were scored
against the identical locked dataset and labels (E1.7) before calling it.

Errors follow the evaluation package convention: ``TypeError`` for a wrong type (a ``bool`` is
never an int or a number), ``ValueError`` for a wrong value, with every type check on every
argument before any value check. Messages are fixed text and never contain input values. The
value checks run in this order: ``seed``, ``resamples``, the series, the cutoffs, then each
observation list (baseline first) over all its observations in these passes: retrieval type, the
set of series names, the values, ``truncated_at``, duplicate question ids; and for ``compare``
last, the id sets, then the retrieval types.

Nothing is serialised, hashed or exported here; that is E1.7. The file imports the sibling
``statistics`` module and never the standard-library one.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.evaluation import statistics as stats
from app.rag.evaluation.metrics import CUTOFFS, MAX_RANK
from app.rag.evaluation.structure import RETRIEVAL_TYPES

MIN_SLICE_QUESTIONS = 10

BINARY = "binary"
CONTINUOUS = "continuous"
SERIES_KINDS = (BINARY, CONTINUOUS)

STATUS_OK = "ok"
STATUS_NOT_ESTIMABLE = "not_estimable"
STATUS_SUPPRESSED = "suppressed"

_OBSERVATIONS_MESSAGE = "observations must be a list or tuple of QuestionObservation"
_OBSERVATION_FIELDS_MESSAGE = (
    "a QuestionObservation needs str question_id and retrieval_type and tuple values and truncated_at"
)
_VALUE_PAIR_MESSAGE = "values must be a tuple of (str, int or float) pairs"
_TRUNCATED_TYPE_MESSAGE = "truncated_at must hold only ints"
_SERIES_MESSAGE = "series must be a list or tuple of SeriesSpec"
_SPEC_FIELDS_MESSAGE = "a SeriesSpec needs a str name and a str kind"
_CUTOFFS_TYPE_MESSAGE = "cutoffs must be a list or tuple of ints"
_INT_MESSAGE = "{name} must be an int"

_SEED_VALUE_MESSAGE = "seed must not be negative"
_RESAMPLES_VALUE_MESSAGE = "resamples is below the minimum"
_SERIES_NAME_MESSAGE = "a series name must not be empty"
_SERIES_DUPLICATE_MESSAGE = "series names must be unique"
_SERIES_KIND_MESSAGE = "a series kind must be 'binary' or 'continuous'"
_CUTOFFS_VALUE_MESSAGE = "cutoffs must be ascending, unique and from 1 to the maximum rank"
_RETRIEVAL_TYPE_MESSAGE = "an observation has an invalid retrieval type"
_SERIES_SET_MESSAGE = "each observation must have exactly one value for every declared series"
_BINARY_VALUE_MESSAGE = "a binary value must be 0 or 1"
_CONTINUOUS_VALUE_MESSAGE = "a continuous value must be finite and from 0 to 1"
_TRUNCATED_VALUE_MESSAGE = "truncated_at must be ascending and drawn from the declared cutoffs"
_DUPLICATE_ID_MESSAGE = "question ids must be unique"
_ID_SET_MESSAGE = "baseline and candidate must have the same question ids"
_RETRIEVAL_MATCH_MESSAGE = "baseline and candidate must have the same retrieval type for every question"


# ---------- input and result types ----------

@dataclass(frozen=True, slots=True)
class QuestionObservation:
    """What one retrieval version scored for one question.

    ``values`` holds one ``(series name, value)`` pair per declared series. ``truncated_at`` lists
    the cutoffs at which this question's distinct ranking was truncated (see the module docs).
    """

    question_id: str
    retrieval_type: str
    values: tuple[tuple[str, int | float], ...]
    truncated_at: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """A declared series: an opaque ``name`` and a ``kind`` of ``binary`` or ``continuous``."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class SeriesResult:
    """One series over one slice.

    ``outcomes`` always holds the individual ``(question_id, value)`` pairs, ordered by id.
    ``successes`` (binary series) and ``rate_or_mean`` (the observed rate of a binary series, the
    observed mean of a continuous one) are ``None`` when the status is ``suppressed``.
    ``interval`` is a ``WilsonInterval`` (binary) or a ``BootstrapInterval`` (continuous) only when
    the status is ``ok``.
    """

    name: str
    kind: str
    n: int
    status: str
    outcomes: tuple[tuple[str, int | float], ...]
    successes: int | None
    rate_or_mean: float | None
    interval: stats.WilsonInterval | stats.BootstrapInterval | None


@dataclass(frozen=True, slots=True)
class SliceReport:
    """The overall set (``retrieval_type is None``) or one retrieval type.

    ``truncation`` is ``(cutoff, number of questions truncated at that cutoff)`` for every
    declared cutoff.
    """

    retrieval_type: str | None
    n: int
    truncation: tuple[tuple[int, int], ...]
    series: tuple[SeriesResult, ...]


@dataclass(frozen=True, slots=True)
class Report:
    """The overall slice and the four retrieval-type slices, with the settings that made them."""

    overall: SliceReport
    slices: tuple[SliceReport, ...]
    seed: int
    resamples: int
    confidence_level: float
    min_slice_questions: int
    cutoffs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PairedSeriesResult:
    """One series compared over one slice (candidate minus baseline).

    ``outcomes`` holds ``(question_id, baseline value, candidate value)`` ordered by id.
    ``mean_difference`` is ``None`` when the status is ``suppressed`` or the slice is empty;
    ``interval`` is a ``PairedBootstrapInterval`` only when the status is ``ok``.
    """

    name: str
    kind: str
    n: int
    status: str
    outcomes: tuple[tuple[str, int | float, int | float], ...]
    mean_difference: float | None
    interval: stats.PairedBootstrapInterval | None


@dataclass(frozen=True, slots=True)
class PairedSliceReport:
    """One slice of a comparison, with the truncation counts of both versions."""

    retrieval_type: str | None
    n: int
    baseline_truncation: tuple[tuple[int, int], ...]
    candidate_truncation: tuple[tuple[int, int], ...]
    series: tuple[PairedSeriesResult, ...]


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The overall and four retrieval-type slices of a candidate-minus-baseline comparison."""

    overall: PairedSliceReport
    slices: tuple[PairedSliceReport, ...]
    seed: int
    resamples: int
    confidence_level: float
    min_slice_questions: int
    cutoffs: tuple[int, ...]


# ---------- validation ----------

def _is_list_or_tuple(value: object) -> bool:
    return type(value) is list or type(value) is tuple


def _check_observation_types(observations: object) -> None:
    if not _is_list_or_tuple(observations):
        raise TypeError(_OBSERVATIONS_MESSAGE)
    for observation in observations:
        if type(observation) is not QuestionObservation:
            raise TypeError(_OBSERVATIONS_MESSAGE)
    for observation in observations:
        if (
            type(observation.question_id) is not str
            or type(observation.retrieval_type) is not str
            or type(observation.values) is not tuple
            or type(observation.truncated_at) is not tuple
        ):
            raise TypeError(_OBSERVATION_FIELDS_MESSAGE)
        for pair in observation.values:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) not in (int, float)
            ):
                raise TypeError(_VALUE_PAIR_MESSAGE)
        for cutoff in observation.truncated_at:
            if type(cutoff) is not int:
                raise TypeError(_TRUNCATED_TYPE_MESSAGE)


def _check_series_types(series: object) -> None:
    if not _is_list_or_tuple(series):
        raise TypeError(_SERIES_MESSAGE)
    for spec in series:
        if type(spec) is not SeriesSpec:
            raise TypeError(_SERIES_MESSAGE)
    for spec in series:
        if type(spec.name) is not str or type(spec.kind) is not str:
            raise TypeError(_SPEC_FIELDS_MESSAGE)


def _check_cutoff_types(cutoffs: object) -> None:
    if not _is_list_or_tuple(cutoffs):
        raise TypeError(_CUTOFFS_TYPE_MESSAGE)
    for cutoff in cutoffs:
        if type(cutoff) is not int:
            raise TypeError(_CUTOFFS_TYPE_MESSAGE)


def _check_series_values(series: Sequence[SeriesSpec]) -> None:
    for spec in series:
        if spec.name == "":
            raise ValueError(_SERIES_NAME_MESSAGE)
    if len({spec.name for spec in series}) != len(series):
        raise ValueError(_SERIES_DUPLICATE_MESSAGE)
    for spec in series:
        if spec.kind not in SERIES_KINDS:
            raise ValueError(_SERIES_KIND_MESSAGE)


def _check_cutoff_values(cutoffs: Sequence[int]) -> None:
    if any(not 1 <= cutoff <= MAX_RANK for cutoff in cutoffs) or any(
        earlier >= later for earlier, later in zip(cutoffs, cutoffs[1:])
    ):
        raise ValueError(_CUTOFFS_VALUE_MESSAGE)


def _check_observation_values(
    observations: Sequence[QuestionObservation], series: Sequence[SeriesSpec], cutoffs: Sequence[int]
) -> None:
    kinds = {spec.name: spec.kind for spec in series}
    for observation in observations:
        if observation.retrieval_type not in RETRIEVAL_TYPES:
            raise ValueError(_RETRIEVAL_TYPE_MESSAGE)
    for observation in observations:
        if sorted(name for name, _ in observation.values) != sorted(kinds):
            raise ValueError(_SERIES_SET_MESSAGE)
    for observation in observations:
        for name, value in observation.values:
            if kinds[name] == BINARY:
                if value != 0 and value != 1:
                    raise ValueError(_BINARY_VALUE_MESSAGE)
            elif not 0 <= value <= 1:
                raise ValueError(_CONTINUOUS_VALUE_MESSAGE)
    for observation in observations:
        flagged = observation.truncated_at
        if any(cutoff not in cutoffs for cutoff in flagged) or any(
            earlier >= later for earlier, later in zip(flagged, flagged[1:])
        ):
            raise ValueError(_TRUNCATED_VALUE_MESSAGE)
    if len({observation.question_id for observation in observations}) != len(observations):
        raise ValueError(_DUPLICATE_ID_MESSAGE)


def _check_settings_types(seed: object, resamples: object) -> None:
    if type(seed) is not int:
        raise TypeError(_INT_MESSAGE.format(name="seed"))
    if type(resamples) is not int:
        raise TypeError(_INT_MESSAGE.format(name="resamples"))


def _check_settings_values(seed: int, resamples: int) -> None:
    if seed < 0:
        raise ValueError(_SEED_VALUE_MESSAGE)
    if resamples < stats.MIN_RESAMPLES:
        raise ValueError(_RESAMPLES_VALUE_MESSAGE)


# ---------- building one slice ----------

def _truncation_counts(
    members: Sequence[QuestionObservation], cutoffs: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (cutoff, sum(1 for observation in members if cutoff in observation.truncated_at))
        for cutoff in cutoffs
    )


def _series_result(
    spec: SeriesSpec,
    outcomes: tuple[tuple[str, int | float], ...],
    suppressed: bool,
    seed: int,
    resamples: int,
) -> SeriesResult:
    n = len(outcomes)
    if suppressed:
        return SeriesResult(spec.name, spec.kind, n, STATUS_SUPPRESSED, outcomes, None, None, None)
    if n == 0:
        return SeriesResult(spec.name, spec.kind, n, STATUS_NOT_ESTIMABLE, outcomes, None, None, None)
    values = [value for _, value in outcomes]
    if spec.kind == BINARY:
        interval = stats.wilson_interval(sum(1 for value in values if value == 1), n)
        return SeriesResult(spec.name, spec.kind, n, STATUS_OK, outcomes, interval.successes,
                            interval.rate, interval)
    if n < 2:
        return SeriesResult(spec.name, spec.kind, n, STATUS_NOT_ESTIMABLE, outcomes, None,
                            math.fsum(values) / n, None)
    interval = stats.bootstrap_mean_interval(outcomes, seed, resamples)
    return SeriesResult(spec.name, spec.kind, n, STATUS_OK, outcomes, None, interval.mean, interval)


def _slice_report(
    retrieval_type: str | None,
    members: Sequence[QuestionObservation],
    series: Sequence[SeriesSpec],
    cutoffs: Sequence[int],
    seed: int,
    resamples: int,
) -> SliceReport:
    ordered = sorted(members, key=lambda observation: observation.question_id)
    maps = [(observation.question_id, dict(observation.values)) for observation in ordered]
    suppressed = retrieval_type is not None and len(ordered) < MIN_SLICE_QUESTIONS
    results = tuple(
        _series_result(spec, tuple((question_id, values[spec.name]) for question_id, values in maps),
                       suppressed, seed, resamples)
        for spec in series
    )
    return SliceReport(retrieval_type, len(ordered), _truncation_counts(ordered, cutoffs), results)


def _paired_series_result(
    spec: SeriesSpec,
    outcomes: tuple[tuple[str, int | float, int | float], ...],
    suppressed: bool,
    seed: int,
    resamples: int,
) -> PairedSeriesResult:
    n = len(outcomes)
    if suppressed:
        return PairedSeriesResult(spec.name, spec.kind, n, STATUS_SUPPRESSED, outcomes, None, None)
    if n == 0:
        return PairedSeriesResult(spec.name, spec.kind, n, STATUS_NOT_ESTIMABLE, outcomes, None, None)
    if n < 2:
        _, before, after = outcomes[0]
        return PairedSeriesResult(spec.name, spec.kind, n, STATUS_NOT_ESTIMABLE, outcomes,
                                  float(after) - float(before), None)
    interval = stats.paired_bootstrap_interval(
        [(question_id, before) for question_id, before, _ in outcomes],
        [(question_id, after) for question_id, _, after in outcomes],
        seed, resamples, binary=spec.kind == BINARY,
    )
    return PairedSeriesResult(spec.name, spec.kind, n, STATUS_OK, outcomes, interval.mean_difference, interval)


def _paired_slice_report(
    retrieval_type: str | None,
    baseline: Sequence[QuestionObservation],
    candidate: Sequence[QuestionObservation],
    series: Sequence[SeriesSpec],
    cutoffs: Sequence[int],
    seed: int,
    resamples: int,
) -> PairedSliceReport:
    before_maps = {o.question_id: dict(o.values) for o in baseline}
    after_maps = {o.question_id: dict(o.values) for o in candidate}
    ids = sorted(before_maps)
    suppressed = retrieval_type is not None and len(ids) < MIN_SLICE_QUESTIONS
    results = tuple(
        _paired_series_result(
            spec,
            tuple((i, before_maps[i][spec.name], after_maps[i][spec.name]) for i in ids),
            suppressed, seed, resamples,
        )
        for spec in series
    )
    return PairedSliceReport(retrieval_type, len(ids), _truncation_counts(baseline, cutoffs),
                             _truncation_counts(candidate, cutoffs), results)


def _by_retrieval_type(observations: Sequence[QuestionObservation]) -> dict[str, list[QuestionObservation]]:
    return {kind: [o for o in observations if o.retrieval_type == kind] for kind in RETRIEVAL_TYPES}


# ---------- public functions ----------

def aggregate(
    observations: Sequence[QuestionObservation],
    series: Sequence[SeriesSpec],
    seed: int,
    resamples: int = stats.DEFAULT_RESAMPLES,
    *,
    cutoffs: Sequence[int] = CUTOFFS,
) -> Report:
    """Summarise one retrieval version overall and per retrieval type (see the module docs).

    Raises ``TypeError`` or ``ValueError`` for invalid input, and never for a valid slice that is
    small, empty or not estimable.
    """
    _check_observation_types(observations)
    _check_series_types(series)
    _check_settings_types(seed, resamples)
    _check_cutoff_types(cutoffs)
    _check_settings_values(seed, resamples)
    _check_series_values(series)
    _check_cutoff_values(cutoffs)
    _check_observation_values(observations, series, cutoffs)

    groups = _by_retrieval_type(observations)
    return Report(
        _slice_report(None, observations, series, cutoffs, seed, resamples),
        tuple(_slice_report(kind, groups[kind], series, cutoffs, seed, resamples) for kind in RETRIEVAL_TYPES),
        seed, resamples, stats.CONFIDENCE_LEVEL, MIN_SLICE_QUESTIONS, tuple(cutoffs),
    )


def compare(
    baseline: Sequence[QuestionObservation],
    candidate: Sequence[QuestionObservation],
    series: Sequence[SeriesSpec],
    seed: int,
    resamples: int = stats.DEFAULT_RESAMPLES,
    *,
    cutoffs: Sequence[int] = CUTOFFS,
) -> ComparisonReport:
    """Compare two retrieval versions, candidate minus baseline (see the module docs).

    Both versions must hold exactly the same question ids with the same retrieval types.
    """
    _check_observation_types(baseline)
    _check_observation_types(candidate)
    _check_series_types(series)
    _check_settings_types(seed, resamples)
    _check_cutoff_types(cutoffs)
    _check_settings_values(seed, resamples)
    _check_series_values(series)
    _check_cutoff_values(cutoffs)
    _check_observation_values(baseline, series, cutoffs)
    _check_observation_values(candidate, series, cutoffs)
    baseline_types = {o.question_id: o.retrieval_type for o in baseline}
    candidate_types = {o.question_id: o.retrieval_type for o in candidate}
    if baseline_types.keys() != candidate_types.keys():
        raise ValueError(_ID_SET_MESSAGE)
    if baseline_types != candidate_types:
        raise ValueError(_RETRIEVAL_MATCH_MESSAGE)

    before_groups = _by_retrieval_type(baseline)
    after_groups = _by_retrieval_type(candidate)
    return ComparisonReport(
        _paired_slice_report(None, baseline, candidate, series, cutoffs, seed, resamples),
        tuple(
            _paired_slice_report(kind, before_groups[kind], after_groups[kind], series, cutoffs, seed, resamples)
            for kind in RETRIEVAL_TYPES
        ),
        seed, resamples, stats.CONFIDENCE_LEVEL, MIN_SLICE_QUESTIONS, tuple(cutoffs),
    )
