"""E1.3.4.a: the Wilson score interval. E1.3.4.b: the ordinary question-level bootstrap interval.
E1.3.4.c: the paired bootstrap interval.

A  constants and the result type
B  numerically verified values (Decimal reference, and published values)
C  edge cases: no successes, all successes, one trial
D  properties: mirror symmetry, monotonicity, containment, narrowing with n
E  invalid input: TypeError for a wrong type, ValueError for a bad value
F  purity, exact types and fixed messages
G  seeded comparison with an independent Decimal implementation
H  imports: no heavy dependencies, no stdlib statistics import
I  bootstrap: constants and the result type
J  bootstrap: hand-derived values and the empirical resampling distribution
K  bootstrap: the nearest-rank convention
L  bootstrap: determinism, reproducibility and shared resampling indices
M  bootstrap: properties that always hold, and two fixed-fixture regression checks
N  bootstrap: invalid input
O  bootstrap: purity, exact types and fixed messages
P  bootstrap: seeded comparison with an independent Fraction-based implementation
Q  bootstrap: imports
R  paired: constants, signature and the result type
S  paired: direction and hand-derived values
T  paired: shared resampling indices and true pairing
U  paired: determinism, order independence and pinned values
V  paired: question-id matching
W  paired: validation of types and values, binary mode, error order
X  paired: seed, resamples and the nearest-rank limits
Y  paired: swapping the versions (negated differences, rank-based endpoint relationship)
Z  paired: purity, exact types, fixed messages and imports
"""
import dataclasses
import inspect
import math
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


# ---------- I. bootstrap: constants and the result type ----------

def obs(*values):
    """Observations q01, q02, ... in the order given."""
    return [(f"q{index:02d}", value) for index, value in enumerate(values, start=1)]


def test_the_bootstrap_constants_are_the_approved_ones():
    assert st.DEFAULT_RESAMPLES == 10_000
    assert st.MIN_RESAMPLES == 1_000
    assert st.BOOTSTRAP_METHOD == "percentile_bootstrap"
    assert st.PERCENTILE_CONVENTION == "nearest_rank"
    assert st.CONFIDENCE_LEVEL == 0.95


def test_the_two_tails_leave_the_documented_confidence_level():
    assert (1000 - 2 * st._TAIL_PERMILLE) / 1000 == st.CONFIDENCE_LEVEL


def test_the_bootstrap_result_has_exactly_the_documented_fields_and_is_frozen_and_slotted():
    assert [f.name for f in dataclasses.fields(st.BootstrapInterval)] == [
        "n", "mean", "lower", "upper", "resamples", "seed", "confidence_level", "method", "percentile_convention"]
    value = st.bootstrap_mean_interval(obs(0.2, 0.8), 1, 1000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.lower = 0.0
    assert not hasattr(value, "__dict__")


def test_the_result_records_its_inputs_and_settings():
    value = st.bootstrap_mean_interval(obs(0.2, 0.8, 0.5), 7, 1500)
    assert (value.n, value.seed, value.resamples) == (3, 7, 1500)
    assert value.mean == 0.5
    assert (value.confidence_level, value.method, value.percentile_convention) == (
        0.95, "percentile_bootstrap", "nearest_rank")


def test_resamples_defaults_to_the_standard_setting_and_the_seed_has_no_default():
    assert st.bootstrap_mean_interval(obs(0.2, 0.8), 1).resamples == 10_000
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(obs(0.2, 0.8))


# ---------- J. bootstrap: hand-derived values and the empirical resampling distribution ----------
# For small samples the distribution of the resampled mean can be written down exactly, so the
# 2.5% and 97.5% nearest-rank limits are known in advance. Each limit sits many standard
# deviations from the next possible value at B = 10,000, so the result does not depend on the seed.

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_two_observations_zero_and_one_give_the_full_unit_interval(seed):
    # The mean of two draws from {0, 1} is 0, 0.5 or 1 with probability 1/4, 1/2 and 1/4.
    value = st.bootstrap_mean_interval(obs(0.0, 1.0), seed)
    assert (value.lower, value.upper, value.mean) == (0.0, 1.0, 0.5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_three_observations_two_zeros_and_a_one(seed):
    # Means are 0, 1/3, 2/3, 1 with probability 8/27, 12/27, 6/27, 1/27; the cumulative
    # probabilities 0.296, 0.741, 0.963, 1 put the 2.5% limit at 0 and the 97.5% limit at 1.
    value = st.bootstrap_mean_interval(obs(0.0, 0.0, 1.0), seed)
    assert (value.lower, value.upper) == (0.0, 1.0)
    assert value.mean == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_six_observations_three_zeros_and_three_ones(seed):
    # The resampled mean is Binomial(6, 1/2) / 6. Cumulative probabilities: P(0) = 1/64 = 0.016
    # (below 2.5%), P(<= 1/6) = 7/64 = 0.109; P(<= 4/6) = 57/64 = 0.89 (below 97.5%),
    # P(<= 5/6) = 63/64 = 0.984. So the limits are 1/6 and 5/6.
    value = st.bootstrap_mean_interval(obs(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), seed)
    assert value.lower == pytest.approx(1 / 6, abs=1e-12)
    assert value.upper == pytest.approx(5 / 6, abs=1e-12)
    assert value.mean == 0.5


@pytest.mark.parametrize("constant", [0.0, 0.3, 1.0, 0, 1])
def test_identical_values_give_a_zero_width_interval_at_that_value(constant):
    # Legitimate for n >= 2, but it only describes the sample (see the module docstring).
    value = st.bootstrap_mean_interval(obs(constant, constant, constant), 5)
    assert value.lower == value.upper == value.mean == pytest.approx(constant, abs=1e-12)


def test_the_resampling_draws_from_the_observed_values_with_the_expected_frequencies():
    # This checks resampling from the observed empirical distribution only: for the sample
    # {0, 1} the resampled mean is 0, 0.5 or 1 with probability 1/4, 1/2, 1/4. It says nothing
    # about the distribution of the underlying population.
    means = st._resample_means([0.0, 1.0], 20261112, 10_000)
    assert means == sorted(means) and len(means) == 10_000
    assert set(means) == {0.0, 0.5, 1.0}
    for point, expected in ((0.0, 0.25), (0.5, 0.5), (1.0, 0.25)):
        assert means.count(point) / 10_000 == pytest.approx(expected, abs=0.03)


def test_each_value_can_be_drawn_more_than_once_in_one_resample():
    # Sampling with replacement: with 3 distinct values the resampled means include 0 (all draws
    # the smallest value) and 1 (all draws the largest), which is impossible without replacement.
    means = st._resample_means([0.0, 0.5, 1.0], 1, 2000)
    assert means[0] == 0.0 and means[-1] == 1.0


# ---------- K. bootstrap: the nearest-rank convention ----------

@pytest.mark.parametrize(
    "count, lower_rank, upper_rank",
    [
        pytest.param(1000, 25, 975, id="B-1000"),
        pytest.param(1001, 26, 976, id="B-1001"),
        pytest.param(1039, 26, 1014, id="B-1039"),
        pytest.param(1040, 26, 1014, id="B-1040"),
        pytest.param(5000, 125, 4875, id="B-5000"),
        pytest.param(10000, 250, 9750, id="B-10000"),
        pytest.param(10001, 251, 9751, id="B-10001"),
    ],
)
def test_the_limits_are_the_values_at_the_nearest_ranks(count, lower_rank, upper_rank):
    ascending = list(range(1, count + 1))  # the value at rank r is r
    assert st._percentile_limits(ascending) == (lower_rank, upper_rank)


def test_the_standard_setting_uses_ranks_250_and_9750_of_10000():
    assert st._percentile_limits(list(range(1, 10_001))) == (250, 9750)


def test_the_integer_rank_arithmetic_equals_the_exact_ceiling_for_many_counts():
    for count in list(range(1000, 3000)) + [9999, 10000, 10001, 12345, 99999, 100000]:
        lower_rank = math.ceil(Fraction(count) * Fraction(25, 1000))
        upper_rank = math.ceil(Fraction(count) * Fraction(975, 1000))
        assert st._percentile_limits(list(range(1, count + 1))) == (lower_rank, upper_rank)


def test_the_limits_are_actual_resampled_means_and_are_not_interpolated():
    ascending = sorted([0.0, 1.0] * 500)
    lower, upper = st._percentile_limits(ascending)
    assert lower in ascending and upper in ascending
    spaced = [float(i) for i in range(1000)]
    assert st._percentile_limits(spaced) == (24.0, 974.0)


# ---------- L. bootstrap: determinism, reproducibility and shared resampling indices ----------

GOLDEN_SAMPLE = [("q1", 0.0), ("q2", 0.25), ("q3", 0.5), ("q4", 0.5),
                 ("q5", 0.75), ("q6", 1.0), ("q7", 0.1), ("q8", 0.9)]


@pytest.mark.parametrize(
    "seed, resamples, lower, upper",
    [
        pytest.param(0, 1000, 0.25625, 0.75625, id="seed-0-B-1000"),
        pytest.param(1, 1000, 0.25, 0.7375, id="seed-1-B-1000"),
        pytest.param(12345, 10000, 0.2625, 0.7375, id="seed-12345-B-10000"),
        pytest.param(20261112, 10000, 0.2625, 0.7375, id="seed-20261112-B-10000"),
    ],
)
def test_pinned_values_for_fixed_seeds(seed, resamples, lower, upper):
    # Computed once with a separately written script (exact Fractions, math.ceil ranks). The
    # sequence of random() for a seed is guaranteed by Python across versions, so these hold on
    # every supported interpreter. The observed mean of the sample is exactly 0.5.
    value = st.bootstrap_mean_interval(GOLDEN_SAMPLE, seed, resamples)
    assert value.lower == pytest.approx(lower, abs=1e-12)
    assert value.upper == pytest.approx(upper, abs=1e-12)
    assert value.mean == pytest.approx(0.5, abs=1e-12)


def test_the_same_inputs_always_give_the_same_result():
    first = st.bootstrap_mean_interval(GOLDEN_SAMPLE, 99, 2000)
    for _ in range(3):
        assert st.bootstrap_mean_interval(GOLDEN_SAMPLE, 99, 2000) == first
    assert st.bootstrap_mean_interval(tuple(GOLDEN_SAMPLE), 99, 2000) == first


def test_the_input_order_does_not_change_the_result():
    expected = st.bootstrap_mean_interval(GOLDEN_SAMPLE, 3, 1500)
    rng = random.Random(1)
    for _ in range(5):
        shuffled = list(GOLDEN_SAMPLE)
        rng.shuffle(shuffled)
        assert st.bootstrap_mean_interval(shuffled, 3, 1500) == expected
    assert st.bootstrap_mean_interval(list(reversed(GOLDEN_SAMPLE)), 3, 1500) == expected


def test_an_integer_value_equals_the_same_float_value():
    ints = [("a", 0), ("b", 1), ("c", 1), ("d", 0), ("e", 1)]
    floats = [(name, float(value)) for name, value in ints]
    assert st.bootstrap_mean_interval(ints, 4, 1000) == st.bootstrap_mean_interval(floats, 4, 1000)


def test_the_same_seed_and_size_draw_the_same_indices_whatever_the_values():
    # Same indices means a linear map of the values maps the resampled means and the limits the
    # same way. (The test does not claim that different seeds must give different endpoints.)
    base = [("q1", 0.0), ("q2", 0.2), ("q3", 0.4), ("q4", 0.6), ("q5", 0.8), ("q6", 1.0)]
    mapped = [(name, 0.5 * value + 0.25) for name, value in base]
    a = st.bootstrap_mean_interval(base, 11, 2000)
    b = st.bootstrap_mean_interval(mapped, 11, 2000)
    assert b.lower == pytest.approx(0.5 * a.lower + 0.25, abs=1e-12)
    assert b.upper == pytest.approx(0.5 * a.upper + 0.25, abs=1e-12)


def test_each_call_uses_its_own_random_state():
    random.seed(0)
    before = st.bootstrap_mean_interval(GOLDEN_SAMPLE, 8, 1000)
    random.seed(12345)
    random.random()
    assert st.bootstrap_mean_interval(GOLDEN_SAMPLE, 8, 1000) == before


def test_the_global_random_state_is_not_consumed():
    random.seed(77)
    expected = random.random()
    random.seed(77)
    st.bootstrap_mean_interval(GOLDEN_SAMPLE, 8, 1000)
    assert random.random() == expected


# ---------- M. bootstrap: properties that always hold, and two fixed-fixture regression checks ----------

def test_the_mean_is_the_exact_observed_mean_and_the_limits_stay_inside_the_data_range():
    rng = random.Random(5)
    for _ in range(25):
        values = [rng.random() for _ in range(rng.randint(2, 20))]
        value = st.bootstrap_mean_interval(obs(*values), rng.randint(0, 99), 1000)
        assert value.mean == math.fsum(values) / len(values)
        assert min(values) <= value.lower <= value.upper <= max(values)
        assert value.n == len(values)


# The next two are regression checks on one fixed fixture and seed each. They are NOT general
# statistical properties: a percentile bootstrap interval need not contain the observed mean,
# and an interval need not narrow every time the sample grows.

def test_regression_the_golden_sample_with_seed_2_has_its_mean_inside_the_interval():
    value = st.bootstrap_mean_interval(GOLDEN_SAMPLE, 2, 5000)
    assert value.lower < value.mean < value.upper


def test_regression_the_interval_narrows_for_this_fixed_pattern_and_seed_as_n_grows_4_16_64():
    pattern = [0.1, 0.9]
    widths = []
    for n in (4, 16, 64):
        value = st.bootstrap_mean_interval(obs(*(pattern * (n // 2))), 3, 3000)
        widths.append(value.upper - value.lower)
    assert widths == sorted(widths, reverse=True) and widths[0] > 2 * widths[-1]


# ---------- N. bootstrap: invalid input ----------

@pytest.mark.parametrize("bad", [None, "abc", "", {}, {"q1": 0.5}, set(), 5, b"ab", iter(()), (x for x in ()),
                                 range(0)])
def test_observations_must_be_a_list_or_tuple(bad):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(bad, 1, 1000)


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(["q1", 0.5], id="list-item"),
        pytest.param(("q1",), id="short-tuple"),
        pytest.param(("q1", 0.5, 0.1), id="long-tuple"),
        pytest.param("q1", id="str-item"),
        pytest.param(None, id="none-item"),
        pytest.param(5, id="int-item"),
        pytest.param({"q1": 0.5}, id="dict-item"),
    ],
)
def test_each_observation_must_be_a_two_item_tuple(item):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval([("q0", 0.1), item], 1, 1000)


@pytest.mark.parametrize("question_id", [1, None, b"q1", 1.5, ("q1",), True])
def test_a_question_id_must_be_a_str(question_id):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval([("q0", 0.1), (question_id, 0.5)], 1, 1000)


def test_a_str_subclass_is_not_accepted_as_a_question_id_and_the_empty_string_is_a_valid_one():
    class Name(str):
        pass

    with pytest.raises(TypeError):
        st.bootstrap_mean_interval([("q0", 0.1), (Name("q1"), 0.5)], 1, 1000)
    assert st.bootstrap_mean_interval([("", 0.1), ("q1", 0.5)], 1, 1000).n == 2


@pytest.mark.parametrize("value", [True, False, "0.5", None, b"1", Fraction(1, 2), [0.5], IntSubclass(1), 1 + 0j])
def test_a_value_must_be_an_int_or_float(value):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval([("q0", 0.1), ("q1", value)], 1, 1000)


@pytest.mark.parametrize("value", [-0.1, 1.1, 2, -1, float("nan"), float("inf"), float("-inf"), 10**30,
                                   10**400, -(10**400), 1e300, -1e-9, 1.0000000000000002])
def test_a_value_must_be_finite_and_within_zero_to_one(value):
    with pytest.raises(ValueError):
        st.bootstrap_mean_interval([("q0", 0.1), ("q1", value)], 1, 1000)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 5e-324, 0.9999999999999999])
def test_the_boundaries_of_the_value_range_are_accepted(value):
    assert st.bootstrap_mean_interval([("q0", 0.5), ("q1", value)], 1, 1000).n == 2


@pytest.mark.parametrize("observations", [[("q1", 0.1), ("q2", 0.2), ("q1", 0.3)], [("a", 0.1), ("a", 0.1)]])
def test_duplicate_question_ids_are_a_value_error(observations):
    with pytest.raises(ValueError):
        st.bootstrap_mean_interval(observations, 1, 1000)


@pytest.mark.parametrize("observations", [[], (), [("q1", 0.5)], (("q1", 0.5),)])
def test_fewer_than_two_observations_are_not_estimable(observations):
    with pytest.raises(ValueError):
        st.bootstrap_mean_interval(observations, 1, 1000)


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None, b"1", IntSubclass(1), [1]])
def test_a_seed_of_the_wrong_type_is_a_type_error(seed):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(obs(0.2, 0.8), seed, 1000)


@pytest.mark.parametrize("seed", [-1, -100, -(10**30)])
def test_a_negative_seed_is_a_value_error(seed):
    with pytest.raises(ValueError):
        st.bootstrap_mean_interval(obs(0.2, 0.8), seed, 1000)


@pytest.mark.parametrize("seed", [0, 1, 2**32, 10**30])
def test_zero_and_large_seeds_are_accepted(seed):
    assert st.bootstrap_mean_interval(obs(0.2, 0.8), seed, 1000).seed == seed


@pytest.mark.parametrize("resamples", [True, False, 1000.0, "1000", None, b"1000", IntSubclass(1000), [1000]])
def test_resamples_of_the_wrong_type_is_a_type_error(resamples):
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(obs(0.2, 0.8), 1, resamples)


@pytest.mark.parametrize("resamples", [999, 500, 1, 0, -1, -1000])
def test_resamples_below_the_minimum_is_a_value_error(resamples):
    with pytest.raises(ValueError):
        st.bootstrap_mean_interval(obs(0.2, 0.8), 1, resamples)


@pytest.mark.parametrize("resamples", [1000, 1001, 2500, 12_000])
def test_resamples_from_the_minimum_upward_are_accepted_with_no_upper_limit(resamples):
    assert st.bootstrap_mean_interval(obs(0.2, 0.8), 1, resamples).resamples == resamples


def test_every_type_check_comes_before_every_value_check():
    bad_value = [("q0", 2.0), ("q0", 0.5)]   # out of range and a duplicate id: both value errors
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(bad_value, 1.5, 1000)         # seed type beats the value errors
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval(bad_value, -1, "1000")        # resamples type beats value errors
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval([("q0", 0.5), ("q1", "x")], -1, 5)   # item type beats seed/resamples values
    with pytest.raises(TypeError):
        st.bootstrap_mean_interval("abc", -1, 5)


def test_the_value_checks_run_in_a_fixed_order():
    # range, then duplicates, then the sample size, then the seed, then resamples
    messages = []
    cases = [
        ([("q0", 2.0), ("q0", 0.5)], -1, 5),
        ([("q0", 0.2), ("q0", 0.5)], -1, 5),
        ([("q0", 0.2)], -1, 5),
        (obs(0.2, 0.5), -1, 5),
        (obs(0.2, 0.5), 1, 5),
    ]
    for observations, seed, resamples in cases:
        with pytest.raises(ValueError) as caught:
            st.bootstrap_mean_interval(observations, seed, resamples)
        messages.append(str(caught.value))
    assert messages == [
        "an observation value must be finite and from 0 to 1",
        "question ids must be unique",
        "a bootstrap needs at least 2 observations",
        "seed must not be negative",
        "resamples is below the minimum",
    ]


# ---------- O. bootstrap: purity, exact types and fixed messages ----------

def test_the_observations_are_not_modified_or_reordered():
    observations = [("q3", 0.9), ("q1", 0.1), ("q2", 0.5)]
    snapshot = list(observations)
    st.bootstrap_mean_interval(observations, 1, 1000)
    assert observations == snapshot and all(a is b for a, b in zip(observations, snapshot))
    as_tuple = tuple(observations)
    st.bootstrap_mean_interval(as_tuple, 1, 1000)
    assert as_tuple == tuple(snapshot)


def test_the_result_types_are_exact():
    value = st.bootstrap_mean_interval(obs(0, 1, 1), 5, 1000)
    assert type(value) is st.BootstrapInterval
    assert all(type(x) is int for x in (value.n, value.resamples, value.seed))
    assert all(type(x) is float for x in (value.mean, value.lower, value.upper, value.confidence_level))
    assert type(value.method) is str and type(value.percentile_convention) is str


def test_error_messages_are_fixed_and_contain_no_input():
    calls = [
        lambda: st.bootstrap_mean_interval(CANARY, 1, 1000),
        lambda: st.bootstrap_mean_interval([CANARY], 1, 1000),
        lambda: st.bootstrap_mean_interval([(CANARY, "x")], 1, 1000),
        lambda: st.bootstrap_mean_interval([(1, 0.5)], CANARY, 1000),
        lambda: st.bootstrap_mean_interval(obs(0.2, 0.8), 1, CANARY),
        lambda: st.bootstrap_mean_interval([(CANARY, 0.5), (CANARY, 0.5)], 1, 1000),
        lambda: st.bootstrap_mean_interval([(CANARY, 7.5), ("q", 0.5)], 1, 1000),
        lambda: st.bootstrap_mean_interval([(CANARY, 0.5)], 1, 1000),
        lambda: st.bootstrap_mean_interval(obs(0.2, 0.8), -123456, 1000),
        lambda: st.bootstrap_mean_interval(obs(0.2, 0.8), 1, 777),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        text = str(caught.value)
        assert CANARY not in text and "123456" not in text and "777" not in text and "7.5" not in text


# ---------- P. bootstrap: independent comparison ----------

def reference_bootstrap(pairs, seed, resamples):
    """Exact Fraction means and exact ceiling ranks; no fsum, no shared code with the module."""
    ordered = sorted(pairs)
    values = [Fraction(value) for _, value in ordered]
    n = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        picks = [values[int(rng.random() * n)] for _ in range(n)]
        means.append(sum(picks) / n)
    means.sort()
    lower = means[math.ceil(Fraction(resamples) * Fraction(25, 1000)) - 1]
    upper = means[math.ceil(Fraction(resamples) * Fraction(975, 1000)) - 1]
    return float(lower), float(upper), float(sum(values) / n)


def test_the_bootstrap_matches_an_independent_reference_on_random_datasets():
    rng = random.Random(20261113)
    for _ in range(60):
        n = rng.randint(2, 30)
        pairs = [(f"id{index}", rng.choice([0, 1, rng.random(), round(rng.random(), 2)])) for index in range(n)]
        rng.shuffle(pairs)
        seed = rng.choice([0, 1, rng.randint(0, 10**9)])
        resamples = rng.choice([1000, 1001, 1234])
        lower, upper, mean = reference_bootstrap(pairs, seed, resamples)
        value = st.bootstrap_mean_interval(pairs, seed, resamples)
        assert value.lower == pytest.approx(lower, abs=1e-12)
        assert value.upper == pytest.approx(upper, abs=1e-12)
        assert value.mean == pytest.approx(mean, abs=1e-12)


def test_the_reference_agrees_with_a_hand_derived_case():
    lower, upper, mean = reference_bootstrap(obs(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 0, 10_000)
    assert (round(lower, 12), round(upper, 12), mean) == (round(1 / 6, 12), round(5 / 6, 12), 0.5)


# ---------- Q. bootstrap: imports ----------

def test_the_bootstrap_adds_no_heavy_dependency_and_no_other_evaluation_module():
    code = (
        "import sys\n"
        "import app.rag.evaluation.statistics as m\n"
        "assert m.bootstrap_mean_interval([('a', 0.0), ('b', 1.0)], 1, 1000).n == 2\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "evaluation = sorted(x for x in sys.modules if x.startswith('app.rag.evaluation.') "
        "and x != 'app.rag.evaluation.statistics')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy), evaluation)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[] []"


# ---------- R. paired: constants, signature and the result type ----------

def paired(baseline_values, candidate_values):
    """Two observation lists over the same ids q01, q02, ... (baseline first)."""
    return obs(*baseline_values), obs(*candidate_values)


def test_the_paired_method_constant_and_the_reused_settings():
    assert st.PAIRED_BOOTSTRAP_METHOD == "paired_percentile_bootstrap"
    assert (st.DEFAULT_RESAMPLES, st.MIN_RESAMPLES, st.CONFIDENCE_LEVEL, st.PERCENTILE_CONVENTION) == (
        10_000, 1_000, 0.95, "nearest_rank")


def test_the_paired_result_has_exactly_the_documented_fields_and_is_frozen_and_slotted():
    assert [f.name for f in dataclasses.fields(st.PairedBootstrapInterval)] == [
        "n", "mean_difference", "lower", "upper", "resamples", "seed", "confidence_level", "method",
        "percentile_convention"]
    base, cand = paired([0.1, 0.5, 0.9], [0.2, 0.4, 1.0])
    value = st.paired_bootstrap_interval(base, cand, 1, 1000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.lower = 0.0
    assert not hasattr(value, "__dict__")


def test_the_paired_result_records_its_inputs_and_settings():
    base, cand = paired([0.0, 0.5, 0.5, 1.0], [0.25, 0.5, 0.75, 1.0])
    value = st.paired_bootstrap_interval(base, cand, 7, 1500)
    assert (value.n, value.seed, value.resamples) == (4, 7, 1500)
    assert value.mean_difference == pytest.approx(0.125, abs=1e-12)
    assert (value.confidence_level, value.method, value.percentile_convention) == (
        0.95, "paired_percentile_bootstrap", "nearest_rank")


def test_the_signature_has_a_required_seed_a_default_resample_count_and_a_keyword_only_binary_flag():
    parameters = inspect.signature(st.paired_bootstrap_interval).parameters
    assert list(parameters) == ["baseline", "candidate", "seed", "resamples", "binary"]
    assert parameters["seed"].default is inspect.Parameter.empty
    assert parameters["resamples"].default == 10_000
    assert parameters["binary"].kind is inspect.Parameter.KEYWORD_ONLY and parameters["binary"].default is False
    base, cand = paired([0.1, 0.5], [0.2, 0.4])
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(base, cand)                       # no seed
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(base, cand, 1, 1000, True)       # binary is not positional
    assert st.paired_bootstrap_interval(base, cand, 1).resamples == 10_000


# ---------- S. paired: direction and hand-derived values ----------

def test_a_constant_shift_gives_a_zero_width_interval_at_the_shift():
    # Every per-question difference is 0.25 (all values are exact binary fractions), so every
    # resampled mean difference is 0.25 whatever the indices: a mathematical guarantee.
    base, cand = paired([0.0, 0.25, 0.5, 0.5, 0.75], [0.25, 0.5, 0.75, 0.75, 1.0])
    for seed in (0, 1, 2, 12345):
        value = st.paired_bootstrap_interval(base, cand, seed, 1000)
        assert (value.lower, value.upper, value.mean_difference) == (0.25, 0.25, 0.25)


def test_the_difference_is_candidate_minus_baseline():
    higher = st.paired_bootstrap_interval(*paired([0.25, 0.25, 0.25], [0.75, 0.75, 0.75]), 1, 1000)
    lower = st.paired_bootstrap_interval(*paired([0.75, 0.75, 0.75], [0.25, 0.25, 0.25]), 1, 1000)
    assert (higher.lower, higher.upper, higher.mean_difference) == (0.5, 0.5, 0.5)
    assert (lower.lower, lower.upper, lower.mean_difference) == (-0.5, -0.5, -0.5)


def test_identical_versions_give_a_zero_difference_interval():
    values = [0.1, 0.3, 0.3, 0.9, 1.0, 0.0]
    value = st.paired_bootstrap_interval(*paired(values, values), 4, 1000)
    assert (value.lower, value.upper, value.mean_difference) == (0.0, 0.0, 0.0)


def test_the_extreme_differences_minus_one_and_plus_one_are_reachable():
    worse = st.paired_bootstrap_interval(*paired([1, 1, 1], [0, 0, 0]), 3, 1000, binary=True)
    better = st.paired_bootstrap_interval(*paired([0, 0, 0], [1, 1, 1]), 3, 1000, binary=True)
    assert (worse.lower, worse.upper, worse.mean_difference) == (-1.0, -1.0, -1.0)
    assert (better.lower, better.upper, better.mean_difference) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_two_questions_with_differences_plus_one_and_zero(seed):
    # Differences [1, 0]: the resampled mean is 0, 0.5 or 1 with probability 1/4, 1/2 and 1/4,
    # so the 2.5% and 97.5% limits are 0 and 1.
    value = st.paired_bootstrap_interval(*paired([0, 0], [1, 0]), seed, binary=True)
    assert (value.lower, value.upper, value.mean_difference) == (0.0, 1.0, 0.5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_two_questions_with_differences_minus_one_and_plus_one(seed):
    # Differences [-1, 1]: the resampled mean is -1, 0 or 1 with probability 1/4, 1/2 and 1/4.
    value = st.paired_bootstrap_interval(*paired([1, 0], [0, 1]), seed, binary=True)
    assert (value.lower, value.upper, value.mean_difference) == (-1.0, 1.0, 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_six_questions_three_improved_three_unchanged(seed):
    # Differences are 0 (three times) and 1 (three times): the resampled mean is Binomial(6, 1/2) / 6,
    # exactly as in the ordinary-bootstrap derivation, so the limits are 1/6 and 5/6.
    value = st.paired_bootstrap_interval(*paired([0] * 6, [0, 0, 0, 1, 1, 1]), seed, binary=True)
    assert value.lower == pytest.approx(1 / 6, abs=1e-12) and value.upper == pytest.approx(5 / 6, abs=1e-12)
    assert value.mean_difference == 0.5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 12345])
def test_six_questions_three_improved_three_regressed(seed):
    # Differences are -1 (three times) and +1 (three times): the resampled mean is
    # (2 * Binomial(6, 1/2) - 6) / 6, so the 1/6 and 5/6 ranks of the binomial map to -2/3 and 2/3.
    value = st.paired_bootstrap_interval(*paired([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]), seed, binary=True)
    assert value.lower == pytest.approx(-2 / 3, abs=1e-12) and value.upper == pytest.approx(2 / 3, abs=1e-12)
    assert value.mean_difference == 0.0


def test_a_binary_comparison_gives_differences_in_minus_one_zero_and_one():
    base = [1, 0, 0, 1, 0, 0, 1, 0]
    cand = [1, 1, 0, 1, 1, 0, 1, 0]
    value = st.paired_bootstrap_interval(*paired(base, cand), 5, 2000, binary=True)
    assert value.mean_difference == 0.25 and -1.0 <= value.lower <= value.upper <= 1.0
    means = st._resample_means([float(c - b) for b, c in zip(base, cand)], 5, 2000)
    assert all(abs(m * 8 - round(m * 8)) < 1e-9 for m in means)   # every resampled mean is a multiple of 1/8


def test_negative_differences_are_accepted_and_reported():
    value = st.paired_bootstrap_interval(*paired([0.9, 0.8, 0.7, 0.9], [0.1, 0.3, 0.2, 0.4]), 2, 1000)
    assert value.mean_difference < 0 and value.lower < 0 and value.upper < 0


# ---------- T. paired: shared resampling indices and true pairing ----------

def reference_paired(baseline, candidate, seed, resamples):
    """The paired bootstrap rebuilt from scratch: ONE set of indices per resample, applied to both
    versions, with the two resampled means computed separately in exact Fractions and subtracted.
    Shares no code with the module (no differences list, no fsum, no helper)."""
    ids = sorted(question_id for question_id, _ in baseline)
    before = {question_id: Fraction(value) for question_id, value in baseline}
    after = {question_id: Fraction(value) for question_id, value in candidate}
    n = len(ids)
    rng = random.Random(seed)
    results = []
    for _ in range(resamples):
        drawn = [ids[int(rng.random() * n)] for _ in range(n)]
        results.append(sum(after[i] for i in drawn) / n - sum(before[i] for i in drawn) / n)
    results.sort()
    lower = results[math.ceil(Fraction(resamples) * Fraction(25, 1000)) - 1]
    upper = results[math.ceil(Fraction(resamples) * Fraction(975, 1000)) - 1]
    observed = sum(after[i] for i in ids) / n - sum(before[i] for i in ids) / n
    return float(lower), float(upper), float(observed)


def random_pair(rng, binary):
    n = rng.randint(2, 25)
    ids = [f"id{index}" for index in range(n)]
    if binary:
        baseline = [(i, rng.choice([0, 1, 0.0, 1.0])) for i in ids]
        candidate = [(i, rng.choice([0, 1, 0.0, 1.0])) for i in ids]
    else:
        baseline = [(i, rng.choice([0, 1, rng.random(), round(rng.random(), 2)])) for i in ids]
        candidate = [(i, rng.choice([0, 1, rng.random(), round(rng.random(), 2)])) for i in ids]
    rng.shuffle(baseline)
    rng.shuffle(candidate)
    return baseline, candidate


@pytest.mark.parametrize("binary", [False, True], ids=["continuous", "binary"])
def test_the_paired_bootstrap_matches_an_independent_shared_index_reconstruction(binary):
    rng = random.Random(20261114 + binary)
    for _ in range(40):
        baseline, candidate = random_pair(rng, binary)
        seed = rng.choice([0, 1, rng.randint(0, 10**9)])
        resamples = rng.choice([1000, 1001, 1234])
        lower, upper, observed = reference_paired(baseline, candidate, seed, resamples)
        value = st.paired_bootstrap_interval(baseline, candidate, seed, resamples, binary=binary)
        assert value.lower == pytest.approx(lower, abs=1e-12)
        assert value.upper == pytest.approx(upper, abs=1e-12)
        assert value.mean_difference == pytest.approx(observed, abs=1e-12)


def test_the_reference_agrees_with_a_hand_derived_case():
    lower, upper, observed = reference_paired(obs(1, 1, 1, 0, 0, 0), obs(0, 0, 0, 1, 1, 1), 0, 10_000)
    assert (round(lower, 12), round(upper, 12), observed) == (round(-2 / 3, 12), round(2 / 3, 12), 0.0)


def test_the_two_versions_are_not_resampled_separately():
    # Independent resampling of the two versions (two ordinary bootstraps, then subtracting the
    # limits) cannot reproduce a zero-width interval for a constant shift; the paired one does.
    base_values = [0.0, 0.25, 0.5, 0.5, 0.75, 0.25]
    cand_values = [value + 0.25 for value in base_values]
    paired_result = st.paired_bootstrap_interval(*paired(base_values, cand_values), 3, 2000)
    assert paired_result.upper - paired_result.lower == 0.0
    ordinary_base = st.bootstrap_mean_interval(obs(*base_values), 3, 2000)
    ordinary_cand = st.bootstrap_mean_interval(obs(*cand_values), 3, 2000)
    assert (ordinary_cand.upper - ordinary_cand.lower) > 0 and (ordinary_base.upper - ordinary_base.lower) > 0
    # Subtracting the two separate intervals would give a wide, wrong interval for the difference.
    assert ordinary_cand.upper - ordinary_base.lower > ordinary_cand.lower - ordinary_base.upper


def test_the_pairing_follows_the_question_ids_not_the_positions_or_the_values():
    ids = ["a", "b", "c", "d", "e"]
    base_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    aligned = [(i, v) for i, v in zip(ids, base_values)]
    shifted = [(i, v + 0.125) for i, v in zip(ids, base_values)]
    misaligned = [(i, v + 0.125) for i, v in zip(ids, reversed(base_values))]  # same values, wrong questions
    right = st.paired_bootstrap_interval(aligned, shifted, 6, 1000)
    wrong = st.paired_bootstrap_interval(aligned, misaligned, 6, 1000)
    assert right.upper - right.lower == pytest.approx(0.0, abs=1e-12)
    assert wrong.upper - wrong.lower > 0.1 and wrong.mean_difference == pytest.approx(right.mean_difference, abs=1e-12)


def test_a_question_keeps_its_own_baseline_and_candidate_values_in_every_resample():
    # One question improves by 1, the others are unchanged. Every resampled mean difference is
    # (number of times that question is drawn) / n, so it is always a multiple of 1/n.
    baseline = obs(0.0, 0.0, 0.0, 0.0)
    candidate = obs(1.0, 0.0, 0.0, 0.0)
    means = st._resample_means([1.0, 0.0, 0.0, 0.0], 9, 2000)
    assert set(means) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    value = st.paired_bootstrap_interval(baseline, candidate, 9, 2000)
    assert value.mean_difference == 0.25 and value.lower == 0.0 and value.upper == 0.75


# ---------- U. paired: determinism, order independence and pinned values ----------

PAIRED_BASELINE = [("q1", 0.0), ("q2", 0.25), ("q3", 0.5), ("q4", 0.5),
                   ("q5", 0.75), ("q6", 1.0), ("q7", 0.1), ("q8", 0.9)]
PAIRED_CANDIDATE = [("q1", 0.1), ("q2", 0.4), ("q3", 0.45), ("q4", 0.7),
                    ("q5", 0.7), ("q6", 0.8), ("q7", 0.3), ("q8", 0.95)]


@pytest.mark.parametrize(
    "seed, resamples, lower, upper",
    [
        pytest.param(0, 1000, -0.05, 0.1375, id="seed-0-B-1000"),
        pytest.param(1, 1000, -0.04375, 0.1375, id="seed-1-B-1000"),
        pytest.param(12345, 10000, -0.04375, 0.1375, id="seed-12345-B-10000"),
        pytest.param(20261112, 10000, -0.04375, 0.1375, id="seed-20261112-B-10000"),
    ],
)
def test_pinned_paired_values_for_fixed_seeds(seed, resamples, lower, upper):
    # Computed once with a separately written script (shared indices, exact Fractions, math.ceil
    # ranks). The observed mean difference is 0.05.
    value = st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, seed, resamples)
    assert value.lower == pytest.approx(lower, abs=1e-12)
    assert value.upper == pytest.approx(upper, abs=1e-12)
    assert value.mean_difference == pytest.approx(0.05, abs=1e-12)


def test_the_same_paired_inputs_always_give_the_same_result():
    first = st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 99, 2000)
    for _ in range(3):
        assert st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 99, 2000) == first
    assert st.paired_bootstrap_interval(tuple(PAIRED_BASELINE), tuple(PAIRED_CANDIDATE), 99, 2000) == first


def test_the_input_order_of_either_version_does_not_change_the_result():
    expected = st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 3, 1500)
    rng = random.Random(2)
    for _ in range(6):
        shuffled_base = list(PAIRED_BASELINE)
        shuffled_cand = list(PAIRED_CANDIDATE)
        rng.shuffle(shuffled_base)
        rng.shuffle(shuffled_cand)
        assert st.paired_bootstrap_interval(shuffled_base, shuffled_cand, 3, 1500) == expected
    assert st.paired_bootstrap_interval(list(reversed(PAIRED_BASELINE)), PAIRED_CANDIDATE, 3, 1500) == expected
    assert st.paired_bootstrap_interval(PAIRED_BASELINE, list(reversed(PAIRED_CANDIDATE)), 3, 1500) == expected


def test_ints_and_the_same_floats_are_interchangeable():
    ints_b = [("a", 0), ("b", 1), ("c", 1), ("d", 0), ("e", 1)]
    ints_c = [("a", 1), ("b", 1), ("c", 0), ("d", 0), ("e", 1)]
    floats_b = [(k, float(v)) for k, v in ints_b]
    floats_c = [(k, float(v)) for k, v in ints_c]
    expected = st.paired_bootstrap_interval(ints_b, ints_c, 4, 1000, binary=True)
    assert st.paired_bootstrap_interval(floats_b, floats_c, 4, 1000, binary=True) == expected
    assert st.paired_bootstrap_interval(ints_b, floats_c, 4, 1000, binary=True) == expected


def test_the_paired_call_uses_its_own_random_state():
    random.seed(77)
    expected = random.random()
    random.seed(77)
    st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 8, 1000)
    assert random.random() == expected


def test_the_same_seed_gives_the_same_indices_as_the_ordinary_bootstrap_of_the_same_size():
    # A linear relationship between the series makes the paired limits equal the ordinary
    # limits of the candidate when the baseline is all zeros (differences = candidate values).
    cand = [0.1, 0.4, 0.45, 0.7, 0.7, 0.8, 0.3, 0.95]
    paired_result = st.paired_bootstrap_interval(*paired([0.0] * 8, cand), 21, 2000)
    ordinary = st.bootstrap_mean_interval(obs(*cand), 21, 2000)
    assert (paired_result.lower, paired_result.upper, paired_result.mean_difference) == (
        ordinary.lower, ordinary.upper, ordinary.mean)


# ---------- V. paired: question-id matching ----------

def test_the_same_ids_in_a_different_order_are_accepted():
    base = [("c", 0.5), ("a", 0.1), ("b", 0.2)]
    cand = [("b", 0.3), ("c", 0.4), ("a", 0.9)]
    assert st.paired_bootstrap_interval(base, cand, 1, 1000).n == 3


@pytest.mark.parametrize(
    "baseline, candidate",
    [
        pytest.param([("a", 0.1), ("b", 0.2)], [("a", 0.1), ("b", 0.2), ("c", 0.3)], id="candidate-has-an-extra-id"),
        pytest.param([("a", 0.1), ("b", 0.2), ("c", 0.3)], [("a", 0.1), ("b", 0.2)], id="candidate-is-missing-an-id"),
        pytest.param([("a", 0.1), ("b", 0.2)], [("a", 0.1), ("x", 0.2)], id="a-different-id"),
        pytest.param([("a", 0.1), ("b", 0.2)], [("x", 0.1), ("y", 0.2)], id="no-id-in-common"),
        pytest.param([("a", 0.1), ("b", 0.2)], [("A", 0.1), ("B", 0.2)], id="case-differs"),
        pytest.param([("a", 0.1), ("b", 0.2)], [("a ", 0.1), ("b", 0.2)], id="whitespace-differs"),
    ],
)
def test_different_question_id_sets_are_a_value_error_and_nothing_is_intersected(baseline, candidate):
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval(baseline, candidate, 1, 1000)
    assert str(caught.value) == "baseline and candidate must have the same question ids"


@pytest.mark.parametrize(
    "baseline, candidate",
    [
        pytest.param([("a", 0.1), ("b", 0.2), ("a", 0.3)], [("a", 0.1), ("b", 0.2), ("c", 0.3)], id="baseline-duplicate"),
        pytest.param([("a", 0.1), ("b", 0.2), ("c", 0.3)], [("a", 0.1), ("b", 0.2), ("b", 0.3)], id="candidate-duplicate"),
        pytest.param([("a", 0.1), ("a", 0.1)], [("a", 0.1), ("a", 0.1)], id="both-duplicate"),
    ],
)
def test_duplicate_ids_in_either_version_are_a_value_error(baseline, candidate):
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval(baseline, candidate, 1, 1000)
    assert str(caught.value) == "question ids must be unique"


def test_a_duplicate_is_not_hidden_by_equal_lengths_or_equal_sets():
    # {a, b} versus {a, a} has the same length as {a, b}, and a set comparison alone would see {a} vs {a, b}.
    with pytest.raises(ValueError):
        st.paired_bootstrap_interval([("a", 0.1), ("b", 0.2)], [("a", 0.1), ("a", 0.2)], 1, 1000)


@pytest.mark.parametrize("version", ["baseline", "candidate"])
@pytest.mark.parametrize("observations", [[], [("a", 0.5)]])
def test_fewer_than_two_observations_in_either_version_are_not_estimable(version, observations):
    good = [("a", 0.5), ("b", 0.4)]
    baseline, candidate = (observations, good) if version == "baseline" else (good, observations)
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval(baseline, candidate, 1, 1000)
    assert str(caught.value) == "a bootstrap needs at least 2 observations"


# ---------- W. paired: validation of types and values, binary mode, error order ----------

@pytest.mark.parametrize("bad", [None, "abc", {}, {"a": 0.5}, set(), 5, b"ab", iter(()), (x for x in ()), range(0)])
@pytest.mark.parametrize("which", ["baseline", "candidate"])
def test_each_version_must_be_a_list_or_tuple(bad, which):
    good = [("a", 0.5), ("b", 0.4)]
    arguments = (bad, good) if which == "baseline" else (good, bad)
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(*arguments, 1, 1000)


@pytest.mark.parametrize(
    "item",
    [pytest.param(["a", 0.5], id="list-item"), pytest.param(("a",), id="short"), pytest.param(("a", 0.5, 1), id="long"),
     pytest.param((1, 0.5), id="int-id"), pytest.param(("a", True), id="bool-value"),
     pytest.param(("a", "0.5"), id="str-value"), pytest.param(("a", None), id="none-value"),
     pytest.param(("a", Fraction(1, 2)), id="fraction-value")],
)
@pytest.mark.parametrize("which", ["baseline", "candidate"])
def test_each_observation_of_each_version_is_checked_for_type(item, which):
    good = [("a", 0.5), ("b", 0.4)]
    bad = [("b", 0.4), item]
    arguments = (bad, good) if which == "baseline" else (good, bad)
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(*arguments, 1, 1000)


@pytest.mark.parametrize("which", ["baseline", "candidate"])
@pytest.mark.parametrize("value", [-0.1, 1.1, 2, -1, float("nan"), float("inf"), float("-inf"), 10**400, 1e300])
def test_values_outside_zero_to_one_in_either_version_are_a_value_error(which, value):
    good = [("a", 0.5), ("b", 0.4)]
    bad = [("a", 0.5), ("b", value)]
    arguments = (bad, good) if which == "baseline" else (good, bad)
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval(*arguments, 1, 1000)
    assert str(caught.value) == "an observation value must be finite and from 0 to 1"


def test_the_differences_are_not_range_checked_only_the_inputs():
    # baseline 1.0 and candidate 0.0 give a difference of -1, outside [0, 1]; that is valid.
    assert st.paired_bootstrap_interval(*paired([1.0, 1.0], [0.0, 0.0]), 1, 1000).lower == -1.0


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None, b"1", IntSubclass(1), [1]])
def test_a_seed_of_the_wrong_type_is_a_type_error(seed):
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), seed, 1000)


@pytest.mark.parametrize("resamples", [True, False, 1000.0, "1000", None, b"1000", IntSubclass(1000), [1000]])
def test_resamples_of_the_wrong_type_is_a_type_error(resamples):
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), 1, resamples)


@pytest.mark.parametrize("flag", [1, 0, None, "True", 1.0, [True], IntSubclass(1)])
def test_the_binary_flag_must_be_a_bool(flag):
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(*paired([0, 1], [1, 0]), 1, 1000, binary=flag)


@pytest.mark.parametrize("which", ["baseline", "candidate"])
@pytest.mark.parametrize("value", [0.5, 0.25, 0.9999999999999999, 1e-9, 5e-324])
def test_binary_mode_rejects_any_value_that_is_not_zero_or_one_in_either_version(which, value):
    good = [("a", 0), ("b", 1)]
    bad = [("a", 0), ("b", value)]
    arguments = (bad, good) if which == "baseline" else (good, bad)
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval(*arguments, 1, 1000, binary=True)
    assert str(caught.value) == "a binary observation must be 0 or 1"


@pytest.mark.parametrize("zero, one", [(0, 1), (0.0, 1.0), (0, 1.0), (0.0, 1), (-0.0, 1)])
def test_binary_mode_accepts_integer_and_float_zero_and_one(zero, one):
    value = st.paired_bootstrap_interval([("a", zero), ("b", one)], [("a", one), ("b", one)], 1, 1000, binary=True)
    assert value.mean_difference == 0.5


def test_binary_mode_still_checks_the_range_first_and_a_bool_is_never_binary():
    with pytest.raises(ValueError) as caught:
        st.paired_bootstrap_interval([("a", 0), ("b", 2)], [("a", 0), ("b", 1)], 1, 1000, binary=True)
    assert str(caught.value) == "an observation value must be finite and from 0 to 1"
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval([("a", True), ("b", 0)], [("a", 0), ("b", 1)], 1, 1000, binary=True)


def test_without_binary_mode_continuous_values_are_accepted():
    base, cand = paired([0.3, 0.6, 0.45], [0.35, 0.5, 0.9])
    assert st.paired_bootstrap_interval(base, cand, 1, 1000, binary=False).n == 3


def test_every_type_check_on_every_argument_comes_before_every_value_check():
    # In each case the first argument holds a VALUE error and a later argument a TYPE error; the type error wins.
    bad_value = [("a", 5.0), ("a", 0.5)]      # out of range and a duplicate id
    good = [("a", 0.5), ("b", 0.4)]
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(bad_value, [("a", "x")], 1, 1000)                  # candidate item type
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(bad_value, "abc", 1, 1000)                         # candidate container type
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(bad_value, good, 1.5, 1000)                        # seed type
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(bad_value, good, -1, "1000")                       # resamples type
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(bad_value, good, -1, 5, binary="yes")              # binary type
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval(good, [("a", "x")], -1, 5)                         # candidate type vs seed value
    with pytest.raises(TypeError):
        st.paired_bootstrap_interval([("a", 9.0)], [("a", True)], 1, 1000, binary=True)  # candidate type vs baseline value


def test_the_value_checks_run_in_the_documented_order():
    ok_b, ok_c = [("a", 0.0), ("b", 1.0)], [("a", 1.0), ("b", 1.0)]
    cases = [
        ([("a", 2.0), ("a", 0.5)], [("a", 0.5), ("a", 0.5)], dict(seed=-1, resamples=5, binary=True)),     # range
        ([("a", 0.5), ("a", 0.5)], [("a", 0.5)], dict(seed=-1, resamples=5, binary=True)),                 # duplicate (baseline first)
        ([("a", 0.5)], [("a", 0.5), ("b", 0.5)], dict(seed=-1, resamples=5, binary=True)),                 # sample size
        ([("a", 0.5), ("b", 0.5)], [("x", 0.5), ("y", 0.5)], dict(seed=-1, resamples=5, binary=True)),     # id sets
        (ok_b, [("a", 0.5), ("b", 1.0)], dict(seed=-1, resamples=5, binary=True)),                         # binary values
        (ok_b, ok_c, dict(seed=-1, resamples=5, binary=True)),                                             # seed
        (ok_b, ok_c, dict(seed=1, resamples=5, binary=True)),                                              # resamples
    ]
    messages = []
    for baseline, candidate, options in cases:
        with pytest.raises(ValueError) as caught:
            st.paired_bootstrap_interval(baseline, candidate, options["seed"], options["resamples"],
                                         binary=options["binary"])
        messages.append(str(caught.value))
    assert messages == [
        "an observation value must be finite and from 0 to 1",
        "question ids must be unique",
        "a bootstrap needs at least 2 observations",
        "baseline and candidate must have the same question ids",
        "a binary observation must be 0 or 1",
        "seed must not be negative",
        "resamples is below the minimum",
    ]


# ---------- X. paired: seed, resamples and the nearest-rank limits ----------

@pytest.mark.parametrize("seed", [-1, -100, -(10**30)])
def test_a_negative_seed_is_a_value_error_for_the_paired_bootstrap(seed):
    with pytest.raises(ValueError):
        st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), seed, 1000)


@pytest.mark.parametrize("seed", [0, 1, 2**32, 10**30])
def test_zero_and_large_seeds_are_accepted_for_the_paired_bootstrap(seed):
    assert st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), seed, 1000).seed == seed


@pytest.mark.parametrize("resamples", [999, 500, 1, 0, -1, -1000])
def test_resamples_below_the_minimum_is_a_value_error_for_the_paired_bootstrap(resamples):
    with pytest.raises(ValueError):
        st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), 1, resamples)


@pytest.mark.parametrize("resamples", [1000, 1001, 2500, 12_000])
def test_resamples_from_the_minimum_upward_are_accepted_for_the_paired_bootstrap(resamples):
    assert st.paired_bootstrap_interval(*paired([0.1, 0.5], [0.2, 0.4]), 1, resamples).resamples == resamples


@pytest.mark.parametrize("resamples", [1000, 1001, 1039, 1040, 5000, 10000])
def test_the_limits_are_the_nearest_rank_values_of_the_resampled_differences(resamples):
    base_values = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 0.1, 0.9]
    cand_values = [0.1, 0.4, 0.45, 0.7, 0.7, 0.8, 0.3, 0.95]
    differences = [c - b for b, c in zip(base_values, cand_values)]
    means = st._resample_means(differences, 5, resamples)
    lower_rank = math.ceil(Fraction(resamples) * Fraction(25, 1000))
    upper_rank = math.ceil(Fraction(resamples) * Fraction(975, 1000))
    value = st.paired_bootstrap_interval(*paired(base_values, cand_values), 5, resamples)
    assert (value.lower, value.upper) == (means[lower_rank - 1], means[upper_rank - 1])


def test_the_limits_come_from_the_shared_percentile_helper_at_the_standard_setting():
    assert st._percentile_limits(list(range(1, 10_001))) == (250, 9750)
    value = st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 5)
    assert value.resamples == 10_000 and value.lower <= value.upper


# ---------- Y. paired: swapping the versions ----------
# Swapping baseline and candidate negates every per-question difference, so under the same seed
# every resampled mean difference is exactly negated. The two limits are NOT always an exact
# mirror image, because the nearest-rank positions (250 and 9,750 of 10,000) do not add up to B + 1.

def test_swapping_the_versions_negates_the_mean_difference_exactly():
    forward = st.paired_bootstrap_interval(PAIRED_BASELINE, PAIRED_CANDIDATE, 3, 1500)
    backward = st.paired_bootstrap_interval(PAIRED_CANDIDATE, PAIRED_BASELINE, 3, 1500)
    assert backward.mean_difference == -forward.mean_difference
    assert (backward.n, backward.seed, backward.resamples) == (forward.n, forward.seed, forward.resamples)


@pytest.mark.parametrize("resamples", [1000, 1001, 1500, 10000])
def test_swapping_the_versions_gives_exactly_negated_resampled_differences(resamples):
    base_values = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 0.1, 0.9]
    cand_values = [0.1, 0.4, 0.45, 0.7, 0.7, 0.8, 0.3, 0.95]
    forward = [c - b for b, c in zip(base_values, cand_values)]
    backward = [b - c for b, c in zip(base_values, cand_values)]
    means = st._resample_means(forward, 4, resamples)
    negated = st._resample_means(backward, 4, resamples)
    assert negated == [-m for m in reversed(means)]


@pytest.mark.parametrize("resamples", [1000, 1001, 1039, 1040, 5000, 10000])
def test_swapped_limits_follow_the_rank_formula_and_are_a_mirror_only_when_the_ranks_add_up(resamples):
    base_values = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 0.1, 0.9]
    cand_values = [0.1, 0.4, 0.45, 0.7, 0.7, 0.8, 0.3, 0.95]
    forward = st.paired_bootstrap_interval(*paired(base_values, cand_values), 4, resamples)
    backward = st.paired_bootstrap_interval(*paired(cand_values, base_values), 4, resamples)
    means = st._resample_means([c - b for b, c in zip(base_values, cand_values)], 4, resamples)
    lower_rank = math.ceil(Fraction(resamples) * Fraction(25, 1000))
    upper_rank = math.ceil(Fraction(resamples) * Fraction(975, 1000))
    # backward.lower is the lower-rank value of the negated list = the negation of forward's value at
    # 0-based position (B - lower_rank); backward.upper likewise at (B - upper_rank).
    assert backward.lower == -means[resamples - lower_rank]
    assert backward.upper == -means[resamples - upper_rank]
    if lower_rank + upper_rank == resamples + 1:
        assert (backward.lower, backward.upper) == (-forward.upper, -forward.lower)   # exact mirror
    else:
        assert resamples - lower_rank == upper_rank != upper_rank - 1                  # off by one rank


def test_at_the_standard_setting_the_swapped_interval_is_not_the_exact_mirror_image_by_construction():
    assert 250 + 9750 != 10_000 + 1
    means = list(range(10_000))
    assert st._percentile_limits(means) == (249, 9749)
    assert st._percentile_limits([-m for m in reversed(means)]) == (-9750, -250)   # not (-9749, -249)


# ---------- Z. paired: purity, exact types, fixed messages and imports ----------

def test_the_paired_inputs_are_not_modified_or_reordered():
    baseline = [("q3", 0.9), ("q1", 0.1), ("q2", 0.5)]
    candidate = [("q2", 0.6), ("q3", 0.2), ("q1", 0.1)]
    snapshots = (list(baseline), list(candidate))
    st.paired_bootstrap_interval(baseline, candidate, 1, 1000)
    assert baseline == snapshots[0] and candidate == snapshots[1]
    assert all(a is b for a, b in zip(baseline, snapshots[0])) and all(a is b for a, b in zip(candidate, snapshots[1]))
    as_tuples = (tuple(baseline), tuple(candidate))
    st.paired_bootstrap_interval(*as_tuples, 1, 1000)
    assert as_tuples == (tuple(snapshots[0]), tuple(snapshots[1]))


def test_the_paired_result_types_are_exact():
    value = st.paired_bootstrap_interval(*paired([0, 1, 1, 0], [1, 1, 0, 0]), 5, 1000, binary=True)
    assert type(value) is st.PairedBootstrapInterval
    assert all(type(x) is int for x in (value.n, value.resamples, value.seed))
    assert all(type(x) is float for x in (value.mean_difference, value.lower, value.upper, value.confidence_level))
    assert type(value.method) is str and type(value.percentile_convention) is str


def test_paired_error_messages_are_fixed_and_contain_no_input():
    good = [("a", 0.5), ("b", 0.4)]
    calls = [
        lambda: st.paired_bootstrap_interval(CANARY, good, 1, 1000),
        lambda: st.paired_bootstrap_interval(good, [CANARY], 1, 1000),
        lambda: st.paired_bootstrap_interval(good, [(CANARY, "x")], 1, 1000),
        lambda: st.paired_bootstrap_interval(good, good, CANARY, 1000),
        lambda: st.paired_bootstrap_interval(good, good, 1, CANARY),
        lambda: st.paired_bootstrap_interval(good, good, 1, 1000, binary=CANARY),
        lambda: st.paired_bootstrap_interval([(CANARY, 7.5), ("b", 0.4)], good, 1, 1000),
        lambda: st.paired_bootstrap_interval([(CANARY, 0.5), (CANARY, 0.4)], good, 1, 1000),
        lambda: st.paired_bootstrap_interval([(CANARY, 0.5), ("b", 0.4)], good, 1, 1000),
        lambda: st.paired_bootstrap_interval([(CANARY, 0.5)], good, 1, 1000),
        lambda: st.paired_bootstrap_interval([("a", 0.25), ("b", 0.4)], good, 1, 1000, binary=True),
        lambda: st.paired_bootstrap_interval(good, good, -123456, 1000),
        lambda: st.paired_bootstrap_interval(good, good, 1, 777),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        text = str(caught.value)
        assert CANARY not in text and "123456" not in text and "777" not in text and "7.5" not in text


def test_the_paired_bootstrap_adds_no_heavy_dependency_and_no_other_evaluation_module():
    code = (
        "import sys\n"
        "import app.rag.evaluation.statistics as m\n"
        "r = m.paired_bootstrap_interval([('a', 0), ('b', 1)], [('a', 1), ('b', 1)], 1, 1000, binary=True)\n"
        "assert r.n == 2\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "evaluation = sorted(x for x in sys.modules if x.startswith('app.rag.evaluation.') "
        "and x != 'app.rag.evaluation.statistics')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy), evaluation)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[] []"
