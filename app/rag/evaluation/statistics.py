"""Confidence intervals for retrieval-evaluation results (E1.3.4).

This module holds the statistical primitives. It works on numbers only: it never computes a
retrieval metric, never reads a ``Question`` or a ranking, and does not depend on ``metrics``.
Which metric a number comes from, and whether it is a raw or a distinct-passage metric, is the
caller's business (the reporting layer).

Checkpoint a (this part): ``wilson_interval``, the Wilson score interval for a proportion such
as Hit@k, where each question either hit (1) or missed (0). The interval describes one
retrieval version on its own. It is not a test of the difference between two versions, and two
overlapping Wilson intervals do not show that a difference is unimportant; comparisons use the
paired bootstrap (a later checkpoint).

The confidence level is fixed at 95% (``CONFIDENCE_LEVEL``); the normal quantile used is
1.959963984540054. This is the plain Wilson score interval, without a continuity correction.
Its limits are clamped to [0, 1], and a limit is exactly 0.0 when there are no successes and
exactly 1.0 when every trial succeeded. The questions are treated as independent trials. Questions
about the same source document are probably correlated, so the interval is likely narrower than
the real uncertainty; the 50-question test set gives exploratory evidence only.

Errors follow the evaluation package convention: ``TypeError`` for an argument that is not a
plain ``int`` (a ``bool`` is a type error), ``ValueError`` for an int that is not allowed.
Messages are fixed text and never contain input values.

The file name is the one in the approved plan and equals the stdlib ``statistics`` module's name,
so this module never imports the stdlib one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CONFIDENCE_LEVEL = 0.95
WILSON_METHOD = "wilson_score"

# The standard normal quantile for a two-sided 95% interval, i.e. the 97.5th percentile.
_Z_95 = 1.959963984540054

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
