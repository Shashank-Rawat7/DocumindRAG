"""Confidence intervals for retrieval-evaluation results (E1.3.4).

This module holds the statistical primitives. It works on numbers only: it never computes a
retrieval metric, never reads a ``Question`` or a ranking, and does not depend on ``metrics``.
Which metric a number comes from, and whether it is a raw or a distinct-passage metric, is the
caller's business (the reporting layer).

Checkpoint a: ``wilson_interval``, the Wilson score interval for a proportion such as Hit@k,
where each question either hit (1) or missed (0). The interval describes one retrieval version
on its own. It is not a test of the difference between two versions, and two overlapping Wilson
intervals do not show that a difference is unimportant; comparisons use the paired bootstrap
(a later checkpoint).

Checkpoint b: ``bootstrap_mean_interval``, the question-level percentile bootstrap interval for
the mean of a continuous per-question metric (MRR, Unit Recall, Evidence Coverage; raw or
distinct, the caller decides). The questions are resampled with replacement and nothing else is:
chunks and evidence units are never treated as independent questions. The method, in order:

  1. The observations are sorted by question id, so the order they arrive in cannot matter.
  2. ``random.Random(seed)`` is created for the call (no shared random state). Each of the
     ``resamples`` resamples draws n indices as ``int(random() * n)``; only ``random()`` is used
     because Python guarantees its sequence for a given seed across versions.
  3. Each resample's mean is ``math.fsum`` of the drawn values divided by n.
  4. The resampled means are sorted and the limits are the nearest-rank percentiles: the value
     at 1-based rank ceil(B * 0.025) is the lower limit and the one at rank ceil(B * 0.975) the
     upper limit (ranks 250 and 9,750 when B is 10,000). The ranks are computed with integers.
     There is no interpolation, so a limit is always one of the resampled means.

``mean`` in the result is the mean of the observed values, not of the resamples. The seed is
required and has no default; ``resamples`` defaults to ``DEFAULT_RESAMPLES`` and must be at
least ``MIN_RESAMPLES``, with no upper limit. The same seed and the same number of observations
always draw the same indices, whatever the values are.

A bootstrap needs at least two observations: with one, every resample is that value, and the
zero-width interval would look like certainty although it says nothing about uncertainty. Fewer
than two raises ``ValueError``; the reporting layer shows the count and the observed value and
marks the interval as not estimable. With two or more identical values a zero-width interval is
legitimate, but it still only describes the sample. As with Wilson, questions about the same
source document are probably correlated, so the interval is likely narrower than the real
uncertainty; no document-level resampling is done.

Checkpoint c: ``paired_bootstrap_interval``, the question-level percentile bootstrap interval for
the mean difference between a candidate and a baseline retrieval version (candidate minus
baseline), for a continuous metric or for a binary one such as Hit@k (``binary=True``). The two
versions must have been scored on exactly the same questions. The pairing is what makes the
comparison valid, and it works like this:

  1. Each input is aligned by question id (sorted), after both id sets are checked to be
     identical, so the order the observations arrive in cannot matter.
  2. The per-question difference ``candidate - baseline`` is formed. The differences may be
     negative; only the inputs are range-checked, never the differences.
  3. The differences are resampled with the same machinery as the ordinary bootstrap. Each
     resample draws one set of question indices, and one drawn index stands for both versions'
     values for that question, so the mean of the drawn differences is exactly the candidate's
     resampled mean minus the baseline's under the same indices. The two versions are never
     resampled separately.
  4. The nearest-rank percentile limits are the same as for the ordinary bootstrap. Because of
     that convention, swapping the two versions negates ``mean_difference`` and every resampled
     difference exactly, but it does not always mirror the two limits exactly: the lower and
     upper ranks (250 and 9,750 at 10,000 resamples) do not add up to B + 1.

The interval is not a test of significance and is not a combination of the two versions'
own intervals. This module checks only that the observations have the same question ids. It
cannot know whether the labels behind them are the same: the caller (the evaluation runner)
must verify that both versions were scored against the identical locked dataset and labels
before comparing them.

The confidence level is fixed at 95% (``CONFIDENCE_LEVEL``); the normal quantile used is
1.959963984540054. This is the plain Wilson score interval, without a continuity correction.
Its limits are clamped to [0, 1], and a limit is exactly 0.0 when there are no successes and
exactly 1.0 when every trial succeeded. The questions are treated as independent trials. Questions
about the same source document are probably correlated, so the interval is likely narrower than
the real uncertainty; the 50-question test set gives exploratory evidence only.

Errors follow the evaluation package convention: ``TypeError`` for an argument of the wrong type
(an ``int`` argument must be a plain ``int``, so a ``bool`` is a type error), ``ValueError`` for
a value of the right type that is not allowed. Type checks come before value checks. Messages are
fixed text and never contain input values. Nothing passed in is modified.

The file name is the one in the approved plan and equals the stdlib ``statistics`` module's name,
so this module never imports the stdlib one.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

CONFIDENCE_LEVEL = 0.95
WILSON_METHOD = "wilson_score"

# The standard normal quantile for a two-sided 95% interval, i.e. the 97.5th percentile.
_Z_95 = 1.959963984540054

# 2.5% in each tail leaves the 95% in CONFIDENCE_LEVEL; kept as an integer per mille so the
# nearest-rank positions below are exact.
_TAIL_PERMILLE = 25

_INT_MESSAGE = "{name} must be an int"
_TRIALS_MESSAGE = "n must be at least 1"
_SUCCESSES_MESSAGE = "successes must be from 0 to n"


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """A Wilson interval and what it was computed from.

    ``rate`` is the observed proportion ``successes / n`` (not the centre of the interval).
    """

    successes: int
    n: int
    rate: float
    lower: float
    upper: float
    confidence_level: float
    method: str


def wilson_interval(successes: int, n: int) -> WilsonInterval:
    """The 95% Wilson score interval for ``successes`` out of ``n`` trials.

    ``n`` is the number of questions with a valid observation: a question whose ranking was
    empty is a valid miss. Raises ``TypeError`` if either argument is not an ``int`` (a ``bool``
    is not), and ``ValueError`` if ``n < 1`` or ``successes`` is not from 0 to ``n``. A single
    trial (``n == 1``) is valid; its interval is simply very wide.
    """
    if type(successes) is not int:
        raise TypeError(_INT_MESSAGE.format(name="successes"))
    if type(n) is not int:
        raise TypeError(_INT_MESSAGE.format(name="n"))
    if n < 1:
        raise ValueError(_TRIALS_MESSAGE)
    if not 0 <= successes <= n:
        raise ValueError(_SUCCESSES_MESSAGE)

    rate = successes / n
    z_squared = _Z_95 * _Z_95
    denominator = 1 + z_squared / n
    centre = (rate + z_squared / (2 * n)) / denominator
    half_width = _Z_95 * math.sqrt(rate * (1 - rate) / n + z_squared / (4 * n * n)) / denominator

    # Mathematically the limit is exactly 0 (or 1) at the extremes; floating point can leave a
    # tiny error, so those two cases are fixed, and the rest is clamped to the unit interval.
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == n else min(1.0, centre + half_width)
    return WilsonInterval(successes, n, rate, lower, upper, CONFIDENCE_LEVEL, WILSON_METHOD)


# ---------- ordinary (unpaired) question-level bootstrap ----------

DEFAULT_RESAMPLES = 10_000
MIN_RESAMPLES = 1_000
BOOTSTRAP_METHOD = "percentile_bootstrap"
PERCENTILE_CONVENTION = "nearest_rank"

_OBSERVATIONS_MESSAGE = "observations must be a list or tuple of (question_id, value) pairs"
_ITEM_MESSAGE = "each observation must be a (question_id, value) tuple"
_ID_MESSAGE = "question_id must be a str"
_VALUE_TYPE_MESSAGE = "an observation value must be an int or float"
_VALUE_RANGE_MESSAGE = "an observation value must be finite and from 0 to 1"
_DUPLICATE_MESSAGE = "question ids must be unique"
_SAMPLE_SIZE_MESSAGE = "a bootstrap needs at least 2 observations"
_SEED_MESSAGE = "seed must not be negative"
_RESAMPLES_MESSAGE = "resamples is below the minimum"


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile bootstrap interval for a mean, and what it was computed from.

    ``mean`` is the mean of the observed values; ``lower`` and ``upper`` are nearest-rank
    percentiles of the resampled means.
    """

    n: int
    mean: float
    lower: float
    upper: float
    resamples: int
    seed: int
    confidence_level: float
    method: str
    percentile_convention: str


def _observation_pairs(observations: object) -> list[tuple[str, int | float]]:
    """The type checks for ``observations``; returns the pairs without changing their order."""
    if not isinstance(observations, (list, tuple)):
        raise TypeError(_OBSERVATIONS_MESSAGE)
    pairs = []
    for item in observations:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(_ITEM_MESSAGE)
        question_id, value = item
        if type(question_id) is not str:
            raise TypeError(_ID_MESSAGE)
        if type(value) is not int and type(value) is not float:
            raise TypeError(_VALUE_TYPE_MESSAGE)
        pairs.append(item)
    return pairs


def _sorted_values(pairs: list[tuple[str, int | float]]) -> list[float]:
    """The value checks for the pairs; returns the values ordered by question id."""
    for _, value in pairs:
        # A NaN fails both comparisons and an infinity fails one, so this also rejects every
        # non-finite value; it never converts a huge int to a float.
        if not 0 <= value <= 1:
            raise ValueError(_VALUE_RANGE_MESSAGE)
    if len({question_id for question_id, _ in pairs}) != len(pairs):
        raise ValueError(_DUPLICATE_MESSAGE)
    if len(pairs) < 2:
        raise ValueError(_SAMPLE_SIZE_MESSAGE)
    return [float(value) for _, value in sorted(pairs, key=lambda pair: pair[0])]


def _resample_means(values: Sequence[float], seed: int, resamples: int) -> list[float]:
    """The sorted means of ``resamples`` resamples (with replacement) of ``values``.

    Does no validation and no range check, so a later checkpoint can also use it on
    differences, which can be negative.
    """
    n = len(values)
    draw = random.Random(seed).random
    means = [math.fsum([values[int(draw() * n)] for _ in range(n)]) / n for _ in range(resamples)]
    means.sort()
    return means


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _percentile_limits(sorted_means: Sequence[float]) -> tuple[float, float]:
    """The nearest-rank 2.5th and 97.5th percentiles of an ascending list.

    The 1-based ranks are ceil(B * 0.025) and ceil(B * 0.975), computed with integers.
    """
    count = len(sorted_means)
    lower_rank = _ceil_div(count * _TAIL_PERMILLE, 1000)
    upper_rank = _ceil_div(count * (1000 - _TAIL_PERMILLE), 1000)
    return sorted_means[lower_rank - 1], sorted_means[upper_rank - 1]


def bootstrap_mean_interval(
    observations: Sequence[tuple[str, int | float]],
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
) -> BootstrapInterval:
    """The 95% percentile bootstrap interval for the mean of per-question values.

    ``observations`` is a list or tuple of ``(question_id, value)`` pairs, one per question:
    ids are unique ``str``, values are finite ``int`` or ``float`` from 0 to 1. Raises
    ``TypeError`` for a wrong type (a ``bool`` is not an int or a number here) and ``ValueError``
    for a value outside 0..1 or non-finite, a duplicate id, fewer than 2 observations, a negative
    ``seed`` or ``resamples`` below ``MIN_RESAMPLES``. The type checks all come first; the value
    checks then run in the order just listed.
    """
    pairs = _observation_pairs(observations)
    if type(seed) is not int:
        raise TypeError(_INT_MESSAGE.format(name="seed"))
    if type(resamples) is not int:
        raise TypeError(_INT_MESSAGE.format(name="resamples"))
    values = _sorted_values(pairs)
    if seed < 0:
        raise ValueError(_SEED_MESSAGE)
    if resamples < MIN_RESAMPLES:
        raise ValueError(_RESAMPLES_MESSAGE)

    lower, upper = _percentile_limits(_resample_means(values, seed, resamples))
    return BootstrapInterval(
        len(values), math.fsum(values) / len(values), lower, upper, resamples, seed,
        CONFIDENCE_LEVEL, BOOTSTRAP_METHOD, PERCENTILE_CONVENTION,
    )


# ---------- paired bootstrap: candidate minus baseline on the same questions ----------

PAIRED_BOOTSTRAP_METHOD = "paired_percentile_bootstrap"

_ID_SET_MESSAGE = "baseline and candidate must have the same question ids"
_BINARY_FLAG_MESSAGE = "binary must be a bool"
_BINARY_VALUE_MESSAGE = "a binary observation must be 0 or 1"


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    """A percentile bootstrap interval for a mean difference (candidate minus baseline).

    ``mean_difference`` is the mean of the observed per-question differences; ``lower`` and
    ``upper`` are nearest-rank percentiles of the resampled mean differences. Both may be
    negative.
    """

    n: int
    mean_difference: float
    lower: float
    upper: float
    resamples: int
    seed: int
    confidence_level: float
    method: str
    percentile_convention: str


def paired_bootstrap_interval(
    baseline: Sequence[tuple[str, int | float]],
    candidate: Sequence[tuple[str, int | float]],
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    *,
    binary: bool = False,
) -> PairedBootstrapInterval:
    """The 95% paired percentile bootstrap interval for ``candidate - baseline``.

    ``baseline`` and ``candidate`` are lists or tuples of ``(question_id, value)`` pairs with
    exactly the same question ids, each id once. Values are finite ``int`` or ``float`` from 0
    to 1; with ``binary=True`` they must be exactly 0 or 1 (``0``, ``1``, ``0.0`` and ``1.0``
    are all accepted). ``binary`` only chooses this validation; it does not change the method.

    Raises ``TypeError`` for a wrong type (a ``bool`` is not an int or a number here, and
    ``binary`` must be a ``bool``). Every type check, on both observation inputs, ``seed``,
    ``resamples`` and ``binary``, comes before any value check. Raises ``ValueError`` for, in
    this order: a baseline value outside 0..1 or non-finite, a duplicate baseline id, fewer than
    2 baseline observations, the same three for the candidate, differing id sets, a value that
    is not 0 or 1 when ``binary`` is set, a negative ``seed``, ``resamples`` below
    ``MIN_RESAMPLES``.
    """
    baseline_pairs = _observation_pairs(baseline)
    candidate_pairs = _observation_pairs(candidate)
    if type(seed) is not int:
        raise TypeError(_INT_MESSAGE.format(name="seed"))
    if type(resamples) is not int:
        raise TypeError(_INT_MESSAGE.format(name="resamples"))
    if type(binary) is not bool:
        raise TypeError(_BINARY_FLAG_MESSAGE)
    baseline_values = _sorted_values(baseline_pairs)
    candidate_values = _sorted_values(candidate_pairs)
    if sorted(question_id for question_id, _ in baseline_pairs) != sorted(
        question_id for question_id, _ in candidate_pairs
    ):
        raise ValueError(_ID_SET_MESSAGE)
    if binary and any(value != 0.0 and value != 1.0 for value in baseline_values + candidate_values):
        raise ValueError(_BINARY_VALUE_MESSAGE)
    if seed < 0:
        raise ValueError(_SEED_MESSAGE)
    if resamples < MIN_RESAMPLES:
        raise ValueError(_RESAMPLES_MESSAGE)

    # Both lists are ordered by the same ids, so position i is one question in both versions.
    differences = [after - before for before, after in zip(baseline_values, candidate_values)]
    lower, upper = _percentile_limits(_resample_means(differences, seed, resamples))
    return PairedBootstrapInterval(
        len(differences), math.fsum(differences) / len(differences), lower, upper, resamples, seed,
        CONFIDENCE_LEVEL, PAIRED_BOOTSTRAP_METHOD, PERCENTILE_CONVENTION,
    )
