"""E1.3.4.a: the Wilson score interval. E1.3.4.b: the ordinary question-level bootstrap interval.

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
"""
import dataclasses
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
