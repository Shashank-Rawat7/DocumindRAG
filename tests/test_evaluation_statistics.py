"""E1.3.4.a: the Wilson score interval.

A  constants and the result type
B  numerically verified values (Decimal reference, and published values)
C  edge cases: no successes, all successes, one trial
D  properties: mirror symmetry, monotonicity, containment, narrowing with n
E  invalid input: TypeError for a wrong type, ValueError for a bad value
F  purity, exact types and fixed messages
G  seeded comparison with an independent Decimal implementation
H  imports: no heavy dependencies, no stdlib statistics import
"""
import dataclasses
import random
import re
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import pytest

from app.rag.evaluation import statistics as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANARY = "SECRET-CANARY-7731"
TOLERANCE = 1e-12


class IntSubclass(int):
    pass


# ---------- A. constants and the result type ----------

def test_the_constants_are_the_approved_ones():
    assert st.CONFIDENCE_LEVEL == 0.95
    assert st.WILSON_METHOD == "wilson_score"


def test_the_result_has_exactly_the_documented_fields_and_is_frozen_and_slotted():
    assert [f.name for f in dataclasses.fields(st.WilsonInterval)] == [
        "successes", "n", "rate", "lower", "upper", "confidence_level", "method"]
    value = st.wilson_interval(3, 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.lower = 0.0
    assert not hasattr(value, "__dict__")


def test_the_result_records_its_inputs_and_settings():
    value = st.wilson_interval(3, 10)
    assert (value.successes, value.n, value.rate) == (3, 10, 0.3)
    assert value.confidence_level == 0.95 and value.method == "wilson_score"


# ---------- B. numerically verified values ----------
# Reference values computed once with 60-digit Decimal arithmetic from the Wilson formula
# (centre = (p + z^2/2n) / (1 + z^2/n), half-width = z * sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)),
# z = 1.959963984540054. Four of them match published tables (see the ids).

@pytest.mark.parametrize(
    "successes, n, lower, upper",
    [
        pytest.param(0, 10, 0.0, 0.277532799862889, id="0-of-10-published-0.2775"),
        pytest.param(5, 10, 0.236593090512564, 0.763406909487436, id="5-of-10-published-0.2366-0.7634"),
        pytest.param(10, 10, 0.722467200137111, 1.0, id="10-of-10"),
        pytest.param(81, 263, 0.255288519878274, 0.366209576982800, id="81-of-263-published-0.2553-0.3662"),
        pytest.param(40, 50, 0.669628940677746, 0.887562499842239, id="40-of-50"),
        pytest.param(25, 50, 0.366445143168286, 0.633554856831714, id="25-of-50"),
        pytest.param(1, 20, 0.008881448800795, 0.236131193446742, id="1-of-20"),
        pytest.param(7, 20, 0.181191824101082, 0.567145723314764, id="7-of-20"),
        pytest.param(0, 50, 0.0, 0.071347599133359, id="0-of-50"),
        pytest.param(1, 2, 0.094531205734231, 0.905468794265769, id="1-of-2"),
    ],
)
def test_the_interval_matches_independently_computed_values(successes, n, lower, upper):
    value = st.wilson_interval(successes, n)
    assert value.lower == pytest.approx(lower, abs=1e-12)
    assert value.upper == pytest.approx(upper, abs=1e-12)
    assert value.rate == successes / n


# ---------- C. edge cases ----------

@pytest.mark.parametrize("n", [1, 2, 10, 50, 263])
def test_no_successes_gives_a_lower_limit_of_exactly_zero(n):
    value = st.wilson_interval(0, n)
    assert value.lower == 0.0 and 0.0 < value.upper < 1.0 and value.rate == 0.0


@pytest.mark.parametrize("n", [1, 2, 10, 50, 263])
def test_all_successes_gives_an_upper_limit_of_exactly_one(n):
    value = st.wilson_interval(n, n)
    assert value.upper == 1.0 and 0.0 < value.lower < 1.0 and value.rate == 1.0


def test_a_single_trial_is_valid_and_its_interval_is_wide():
    miss = st.wilson_interval(0, 1)
    hit = st.wilson_interval(1, 1)
    assert (miss.lower, miss.upper) == (0.0, pytest.approx(0.793450685622763, abs=1e-12))
    assert (hit.lower, hit.upper) == (pytest.approx(0.206549314377237, abs=1e-12), 1.0)
    assert (miss.n, hit.n) == (1, 1) and (miss.rate, hit.rate) == (0.0, 1.0)


def test_a_large_n_does_not_break_the_arithmetic():
    value = st.wilson_interval(500_000, 1_000_000)
    assert 0.4990 < value.lower < 0.5 < value.upper < 0.5010
    assert (st.wilson_interval(0, 10**12).lower, st.wilson_interval(10**12, 10**12).upper) == (0.0, 1.0)


# ---------- D. properties ----------

@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 50])
def test_the_interval_is_the_mirror_image_for_x_and_n_minus_x(n):
    for x in range(n + 1):
        a = st.wilson_interval(x, n)
        b = st.wilson_interval(n - x, n)
        assert a.lower == pytest.approx(1 - b.upper, abs=TOLERANCE)
        assert a.upper == pytest.approx(1 - b.lower, abs=TOLERANCE)


@pytest.mark.parametrize("n", [1, 5, 50])
def test_both_limits_rise_with_the_number_of_successes(n):
    values = [st.wilson_interval(x, n) for x in range(n + 1)]
    assert [v.lower for v in values] == sorted(v.lower for v in values)
    assert [v.upper for v in values] == sorted(v.upper for v in values)
    assert len({v.lower for v in values}) == n + 1 and len({v.upper for v in values}) == n + 1


@pytest.mark.parametrize("n", [1, 2, 5, 20, 50, 263])
def test_the_limits_are_ordered_within_the_unit_interval_and_around_the_rate(n):
    for x in range(n + 1):
        value = st.wilson_interval(x, n)
        assert 0.0 <= value.lower <= value.rate <= value.upper <= 1.0
        assert value.lower < value.upper


def test_the_interval_narrows_as_n_grows_at_the_same_rate():
    widths = []
    for n in (10, 20, 50, 100, 1000):
        value = st.wilson_interval(n // 2, n)
        widths.append(value.upper - value.lower)
    assert widths == sorted(widths, reverse=True) and widths[0] > 2 * widths[-1]


def test_the_interval_is_not_centred_on_the_rate_except_at_one_half():
    value = st.wilson_interval(1, 20)
    assert value.upper - value.rate > value.rate - value.lower  # skewed towards the middle
    half = st.wilson_interval(10, 20)
    assert half.upper - half.rate == pytest.approx(half.rate - half.lower, abs=TOLERANCE)


# ---------- E. invalid input ----------

@pytest.mark.parametrize("bad", [True, False, 1.0, 5.0, "5", None, b"5", Fraction(5), [5], IntSubclass(5)])
def test_successes_of_the_wrong_type_is_a_type_error(bad):
    with pytest.raises(TypeError):
        st.wilson_interval(bad, 10)


@pytest.mark.parametrize("bad", [True, False, 10.0, "10", None, b"10", Fraction(10), [10], IntSubclass(10)])
def test_n_of_the_wrong_type_is_a_type_error(bad):
    with pytest.raises(TypeError):
        st.wilson_interval(0, bad)


@pytest.mark.parametrize("n", [0, -1, -10])
def test_n_below_one_is_a_value_error(n):
    with pytest.raises(ValueError):
        st.wilson_interval(0, n)


@pytest.mark.parametrize(
    "successes, n",
    [pytest.param(-1, 10, id="negative"), pytest.param(11, 10, id="above-n"), pytest.param(2, 1, id="above-one-trial"),
     pytest.param(-5, 1, id="negative-one-trial")],
)
def test_successes_outside_zero_to_n_is_a_value_error(successes, n):
    with pytest.raises(ValueError):
        st.wilson_interval(successes, n)


def test_the_type_check_comes_before_the_value_check():
    with pytest.raises(TypeError):
        st.wilson_interval(1.5, 0)  # bad type for successes wins over n < 1
    with pytest.raises(TypeError):
        st.wilson_interval(-1, "x")
    with pytest.raises(ValueError):
        st.wilson_interval(-1, 0)   # both ints: n is checked before successes


# ---------- F. purity, exact types and fixed messages ----------

def test_the_result_types_are_exact_and_the_call_is_deterministic():
    value = st.wilson_interval(7, 20)
    assert type(value) is st.WilsonInterval
    assert type(value.successes) is int and type(value.n) is int
    assert all(type(x) is float for x in (value.rate, value.lower, value.upper, value.confidence_level))
    assert type(value.method) is str
    assert st.wilson_interval(7, 20) == value


def test_zero_and_full_successes_still_return_plain_floats():
    for value in (st.wilson_interval(0, 5), st.wilson_interval(5, 5)):
        assert all(type(x) is float for x in (value.rate, value.lower, value.upper))


def test_error_messages_are_fixed_and_contain_no_input():
    calls = [
        lambda: st.wilson_interval(CANARY, 10),
        lambda: st.wilson_interval(0, CANARY),
        lambda: st.wilson_interval(10**30 + 1, 10**30),
        lambda: st.wilson_interval(-10**30, 10),
        lambda: st.wilson_interval(0, -10**30),
    ]
    messages = set()
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        text = str(caught.value)
        assert CANARY not in text and "10" not in text and "30" not in text
        messages.add(text)
    assert messages == {"successes must be an int", "n must be an int", "successes must be from 0 to n",
                        "n must be at least 1"}


# ---------- G. independent comparison ----------

def reference_interval(successes, n):
    """Wilson limits with 80-digit Decimal arithmetic (floats only at the end)."""
    getcontext().prec = 80
    z = Decimal("1.959963984540054")
    p = Decimal(successes) / Decimal(n)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)).sqrt() / d
    lower, upper = max(Decimal(0), centre - half), min(Decimal(1), centre + half)
    return float(lower), float(upper)


def test_the_interval_matches_a_decimal_reference_on_random_inputs():
    rng = random.Random(20261112)
    for _ in range(3000):
        n = rng.choice([1, 2, 3, 5, 9, 10, 11, 25, 50, 51, 100, 263, 1000, 99999])
        x = rng.choice([0, n, rng.randint(0, n), rng.randint(0, n)])
        lower, upper = reference_interval(x, n)
        value = st.wilson_interval(x, n)
        assert value.lower == pytest.approx(lower, abs=1e-12)
        assert value.upper == pytest.approx(upper, abs=1e-12)


def test_the_reference_agrees_with_a_published_case():
    lower, upper = reference_interval(81, 263)
    assert (round(lower, 4), round(upper, 4)) == (0.2553, 0.3662)


# ---------- H. imports ----------

def test_importing_the_module_loads_no_heavy_dependencies_or_other_evaluation_modules():
    code = (
        "import sys\n"
        "import app.rag.evaluation.statistics as m\n"
        "assert m.wilson_interval(1, 2).n == 2\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "evaluation = sorted(x for x in sys.modules if x.startswith('app.rag.evaluation.') "
        "and x != 'app.rag.evaluation.statistics')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy), evaluation)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[] []"


def test_the_module_does_not_import_the_stdlib_statistics_module():
    source = (PROJECT_ROOT / "app" / "rag" / "evaluation" / "statistics.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+statistics\b", source, flags=re.MULTILINE)
