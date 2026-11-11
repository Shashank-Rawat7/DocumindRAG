"""Half-open character intervals for the evaluation package (E1.2a).

An interval ``(start, end)`` covers the positions ``start, start + 1, ..., end - 1``
of a text, exactly like the Python slice ``text[start:end]``. Positions are Python
code-point offsets. Only the relations the evaluation needs are defined here:

  * ``overlaps``: the two intervals share at least one position.
  * ``touches``:  one interval ends exactly where the other starts (no shared
                  position, no gap).
  * ``collides``: ``overlaps`` or ``touches``. This is the "disjoint" rule of the
                  E1.1 contract: two evidence spans are disjoint only if they do not
                  collide.

Every argument must be a ``tuple`` or ``list`` of exactly two plain ``int`` values
(``bool``, ``float``, ``str`` and other types are rejected, never coerced) with
``0 <= start < end``. Zero-length and reversed intervals are rejected. Violations
raise ``ValueError`` with a fixed message that contains no values. The functions
are pure and never modify their arguments.
"""
from __future__ import annotations

Interval = tuple[int, int] | list[int]


def _check(interval: object) -> tuple[int, int]:
    if (
        not isinstance(interval, (tuple, list))
        or len(interval) != 2
        or type(interval[0]) is not int
        or type(interval[1]) is not int
    ):
        raise ValueError("interval must be a (start, end) pair of int")
    start, end = interval
    if not 0 <= start < end:
        raise ValueError("interval must satisfy 0 <= start < end")
    return start, end


def overlaps(a: Interval, b: Interval) -> bool:
    """True if ``a`` and ``b`` share at least one position."""
    a_start, a_end = _check(a)
    b_start, b_end = _check(b)
    return a_start < b_end and b_start < a_end


def touches(a: Interval, b: Interval) -> bool:
    """True if one interval ends exactly where the other starts."""
    a_start, a_end = _check(a)
    b_start, b_end = _check(b)
    return a_end == b_start or b_end == a_start


def collides(a: Interval, b: Interval) -> bool:
    """True if ``a`` and ``b`` overlap or touch."""
    return overlaps(a, b) or touches(a, b)
