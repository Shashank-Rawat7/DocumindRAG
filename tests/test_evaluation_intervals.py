"""E1.2a interval helpers: overlap, touching and collision of half-open intervals.

A  relation truth table, in both argument orders
B  exhaustive agreement with an independent set-based definition
C  accepted input forms (tuple, list, mixed) and no mutation of the inputs
D  rejected inputs (bool, float, str, None, wrong length, negative, empty, reversed)
E  module isolation
"""
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import intervals as iv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BIG = 2**31 - 1

# (id, a, b, overlaps, touches)
RELATIONS = [
    ("identical", (0, 5), (0, 5), True, False),
    ("nested", (0, 10), (2, 5), True, False),
    ("partial", (0, 5), (3, 8), True, False),
    ("one-shared-position", (0, 5), (4, 9), True, False),
    ("shared-start", (0, 5), (0, 3), True, False),
    ("shared-end", (2, 5), (0, 5), True, False),
    ("adjacent", (0, 5), (5, 8), False, True),
    ("gap-of-one", (0, 5), (6, 8), False, False),
    ("far-apart", (0, 5), (100, 200), False, False),
    ("single-positions-adjacent", (0, 1), (1, 2), False, True),
    ("single-positions-gap", (0, 1), (2, 3), False, False),
    ("large-values-adjacent", (BIG - 1, BIG), (BIG, BIG + 1), False, True),
    ("large-values-gap", (BIG - 2, BIG - 1), (BIG, BIG + 1), False, False),
]


# ---------- A: truth table ----------

@pytest.mark.parametrize(
    "a, b, expected_overlaps, expected_touches",
    [pytest.param(a, b, o, t, id=name) for name, a, b, o, t in RELATIONS],
)
def test_relations_in_both_orders(a, b, expected_overlaps, expected_touches):
    for x, y in ((a, b), (b, a)):
        assert iv.overlaps(x, y) is expected_overlaps
        assert iv.touches(x, y) is expected_touches
        assert iv.collides(x, y) is (expected_overlaps or expected_touches)


# ---------- B: independent definition ----------

GRID = [(s, e) for s in range(0, 6) for e in range(s + 1, 7)]


def test_agrees_with_a_set_based_definition_on_every_small_pair():
    for a, b in itertools.product(GRID, GRID):
        positions_a, positions_b = set(range(*a)), set(range(*b))
        union = sorted(positions_a | positions_b)
        contiguous = union == list(range(union[0], union[-1] + 1))
        shared = bool(positions_a & positions_b)
        assert iv.overlaps(a, b) is shared
        assert iv.collides(a, b) is contiguous
        assert iv.touches(a, b) is (contiguous and not shared)


# ---------- C: accepted forms ----------

def test_lists_and_mixed_forms_are_accepted_and_not_modified():
    a, b = [0, 5], (5, 8)
    assert iv.collides(a, b) is True
    assert iv.touches(b, a) is True
    assert iv.overlaps(a, b) is False
    assert a == [0, 5] and b == (5, 8)


@pytest.mark.parametrize("func", [iv.overlaps, iv.touches, iv.collides], ids=lambda f: f.__name__)
def test_results_are_real_booleans(func):
    assert type(func((0, 5), (3, 8))) is bool
    assert type(func((0, 5), (9, 12))) is bool


# ---------- D: rejected inputs ----------

BAD_TYPE = [
    (True, 5),
    (0, True),
    (False, 5),
    (0.0, 5),
    (0, 5.0),
    ("0", 5),
    (None, 5),
    (0, None),
    5,
    None,
    "05",
    b"\x00\x05",
    {0, 5},
    {"start": 0, "end": 5},
    (),
    (1,),
    (0, 5, 6),
    [0],
    [],
]

BAD_RANGE = [(-1, 5), (-5, -1), (3, 3), (0, 0), (5, 3)]


@pytest.mark.parametrize("func", [iv.overlaps, iv.touches, iv.collides], ids=lambda f: f.__name__)
@pytest.mark.parametrize("bad", BAD_TYPE, ids=repr)
def test_wrong_type_or_shape_is_rejected_in_either_position(func, bad):
    with pytest.raises(ValueError, match="pair of int"):
        func(bad, (0, 5))
    with pytest.raises(ValueError, match="pair of int"):
        func((0, 5), bad)


@pytest.mark.parametrize("func", [iv.overlaps, iv.touches, iv.collides], ids=lambda f: f.__name__)
@pytest.mark.parametrize("bad", BAD_RANGE, ids=repr)
def test_invalid_range_is_rejected_in_either_position(func, bad):
    with pytest.raises(ValueError, match="0 <= start < end"):
        func(bad, (0, 5))
    with pytest.raises(ValueError, match="0 <= start < end"):
        func((0, 5), bad)


def test_error_messages_contain_no_values():
    with pytest.raises(ValueError) as caught:
        iv.overlaps((-12345, 5), (0, 5))
    assert "12345" not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        iv.overlaps(("secret-value", 5), (0, 5))
    assert "secret-value" not in str(caught.value)


# ---------- E: isolation ----------

def test_module_imports_nothing_else_from_the_application():
    code = (
        "import sys; import app.rag.evaluation.intervals; "
        "print(sorted(name for name in sys.modules if name.startswith('app.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == (
        "['app.rag', 'app.rag.evaluation', 'app.rag.evaluation.intervals']"
    )
