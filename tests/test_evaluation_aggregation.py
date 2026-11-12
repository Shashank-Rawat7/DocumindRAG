"""Tests for app.rag.evaluation.aggregation (E1.3.4.d).

A  constants, input types and result types
B  overall aggregation (binary Wilson, continuous bootstrap, equal to the primitives)
C  retrieval-type slices and the minimum-10 rule
D  empty and one-question data (non-estimable, never a failure)
E  truncation counts
F  comparison (paired, candidate minus baseline)
G  comparison: matching of ids and retrieval types
H  validation of types and values, and the error order
I  determinism, order independence and purity
J  import isolation
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from app.rag.evaluation import aggregation as agg
from app.rag.evaluation import statistics as st
from app.rag.evaluation.metrics import CUTOFFS
from app.rag.evaluation.structure import RETRIEVAL_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANARY = "CANARY-xyzzy-7731"
SEED = 7
B = 1000
HIT = agg.SeriesSpec("hit", "binary")
SCORE = agg.SeriesSpec("score", "continuous")
SERIES = (HIT, SCORE)


class IntSubclass(int):
    pass


def build(counts, hit=lambda i: i % 2, score=lambda i: (i % 5) / 4, truncated=lambda rt, i: ()):
    """Observations for {retrieval_type: count}; ids are '<type>-<NN>'."""
    result = []
    for retrieval_type, count in counts.items():
        for i in range(count):
            result.append(agg.QuestionObservation(
                f"{retrieval_type}-{i:02d}", retrieval_type,
                (("hit", hit(i)), ("score", score(i))), tuple(truncated(retrieval_type, i))))
    return result


def one(question_id="q1", retrieval_type="en_to_en", hit=1, score=0.5, truncated=()):
    return agg.QuestionObservation(question_id, retrieval_type, (("hit", hit), ("score", score)), truncated)


def series_of(slice_report, name):
    return next(result for result in slice_report.series if result.name == name)


def run(observations=None, series=SERIES, seed=SEED, resamples=B, **options):
    return agg.aggregate(build({"en_to_en": 3}) if observations is None else observations,
                         series, seed, resamples, **options)


MIXED = {"en_to_en": 10, "hi_to_hi": 9, "en_to_hi": 0, "hi_to_en": 12}


# ---------- A. constants, input types and result types ----------

def test_the_constants():
    assert agg.MIN_SLICE_QUESTIONS == 10
    assert (agg.BINARY, agg.CONTINUOUS, agg.SERIES_KINDS) == ("binary", "continuous", ("binary", "continuous"))
    assert (agg.STATUS_OK, agg.STATUS_NOT_ESTIMABLE, agg.STATUS_SUPPRESSED) == ("ok", "not_estimable", "suppressed")


@pytest.mark.parametrize(
    "cls, names",
    [
        (agg.QuestionObservation, ["question_id", "retrieval_type", "values", "truncated_at"]),
        (agg.SeriesSpec, ["name", "kind"]),
        (agg.SeriesResult, ["name", "kind", "n", "status", "outcomes", "successes", "rate_or_mean", "interval"]),
        (agg.SliceReport, ["retrieval_type", "n", "truncation", "series"]),
        (agg.Report, ["overall", "slices", "seed", "resamples", "confidence_level", "min_slice_questions",
                      "cutoffs"]),
        (agg.PairedSeriesResult, ["name", "kind", "n", "status", "outcomes", "mean_difference", "interval"]),
        (agg.PairedSliceReport, ["retrieval_type", "n", "baseline_truncation", "candidate_truncation", "series"]),
        (agg.ComparisonReport, ["overall", "slices", "seed", "resamples", "confidence_level",
                                "min_slice_questions", "cutoffs"]),
    ],
)
def test_the_data_classes_have_the_documented_fields_and_are_frozen_and_slotted(cls, names):
    assert [f.name for f in dataclasses.fields(cls)] == names
    assert cls.__dataclass_params__.frozen and "__slots__" in vars(cls)


def test_a_result_cannot_be_changed():
    report = run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.seed = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        series_of(report.overall, "hit").status = "x"


def test_the_signatures_have_a_required_seed_default_resamples_and_keyword_only_cutoffs():
    for function, first in ((agg.aggregate, ["observations"]), (agg.compare, ["baseline", "candidate"])):
        parameters = inspect.signature(function).parameters
        assert list(parameters) == first + ["series", "seed", "resamples", "cutoffs"]
        assert parameters["seed"].default is inspect.Parameter.empty
        assert parameters["resamples"].default == 10_000
        assert parameters["cutoffs"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["cutoffs"].default == CUTOFFS


def test_the_report_records_the_settings():
    report = run(seed=11, resamples=1234, cutoffs=(3, 5))
    assert (report.seed, report.resamples, report.confidence_level, report.min_slice_questions,
            report.cutoffs) == (11, 1234, 0.95, 10, (3, 5))
    assert run().cutoffs == CUTOFFS


# ---------- B. overall aggregation ----------

def test_overall_binary_series_is_a_wilson_interval_equal_to_the_primitive():
    report = agg.aggregate(build({"en_to_en": 7, "hi_to_hi": 3}), SERIES, SEED, B)
    result = series_of(report.overall, "hit")
    assert (result.status, result.n, result.successes, result.rate_or_mean) == ("ok", 10, 4, 0.4)
    assert result.interval == st.wilson_interval(4, 10) and type(result.interval) is st.WilsonInterval
    assert report.overall.retrieval_type is None and report.overall.n == 10


def test_overall_continuous_series_is_a_bootstrap_interval_equal_to_the_primitive():
    observations = build({"en_to_en": 7, "hi_to_hi": 3})
    expected_pairs = sorted((o.question_id, dict(o.values)["score"]) for o in observations)
    result = series_of(agg.aggregate(observations, SERIES, SEED, B).overall, "score")
    expected = st.bootstrap_mean_interval(expected_pairs, SEED, B)
    assert result.interval == expected and type(result.interval) is st.BootstrapInterval
    assert (result.status, result.n, result.successes, result.rate_or_mean) == ("ok", 10, None, expected.mean)
    assert result.rate_or_mean == math.fsum(v for _, v in expected_pairs) / 10


def test_a_hand_derived_continuous_interval():
    # Six questions with values 0, 0, 0, 1, 1, 1: the resampled mean is Binomial(6, 1/2) / 6, so the
    # nearest-rank 2.5% and 97.5% limits are 1/6 and 5/6 (derived in the statistics tests).
    observations = build({"en_to_en": 6}, score=lambda i: 0 if i < 3 else 1)
    interval = series_of(agg.aggregate(observations, SERIES, 3, B).overall, "score").interval
    assert interval.lower == pytest.approx(1 / 6, abs=1e-12) and interval.upper == pytest.approx(5 / 6, abs=1e-12)


def test_a_hand_derived_wilson_interval_for_ten_questions():
    result = series_of(agg.aggregate(build({"en_to_en": 10}), SERIES, SEED, B).slices[0], "hit")
    assert result.successes == 5 and result.rate_or_mean == 0.5
    assert result.interval.lower == pytest.approx(0.2366, abs=1e-4)
    assert result.interval.upper == pytest.approx(0.7634, abs=1e-4)


def test_outcomes_hold_every_individual_value_ordered_by_question_id():
    observations = [one("b", hit=0, score=0.25), one("a", hit=1, score=0.75), one("c", hit=1, score=1.0)]
    report = agg.aggregate(observations, SERIES, SEED, B)
    assert series_of(report.overall, "hit").outcomes == (("a", 1), ("b", 0), ("c", 1))
    assert series_of(report.overall, "score").outcomes == (("a", 0.75), ("b", 0.25), ("c", 1.0))


def test_the_series_come_back_in_the_declared_order_and_with_their_kind():
    report = run(series=(SCORE, HIT))
    assert [(r.name, r.kind) for r in report.overall.series] == [("score", "continuous"), ("hit", "binary")]


def test_float_and_int_binary_values_are_the_same_outcome():
    ints = [one(f"q{i}", hit=i % 2) for i in range(6)]
    floats = [one(f"q{i}", hit=float(i % 2)) for i in range(6)]
    a = series_of(agg.aggregate(ints, SERIES, SEED, B).overall, "hit")
    b = series_of(agg.aggregate(floats, SERIES, SEED, B).overall, "hit")
    assert (a.successes, a.rate_or_mean, a.interval) == (b.successes, b.rate_or_mean, b.interval) == (3, 0.5, a.interval)


def test_a_declared_series_set_may_be_empty():
    report = agg.aggregate([agg.QuestionObservation("a", "en_to_en", (), ())], [], SEED, B)
    assert report.overall.series == () and report.overall.n == 1


def test_the_seed_and_resamples_reach_the_intervals():
    result = series_of(run(build({"en_to_en": 6}), seed=21, resamples=1500).overall, "score")
    assert (result.interval.seed, result.interval.resamples) == (21, 1500)


# ---------- C. retrieval-type slices and the minimum-10 rule ----------

def test_the_four_slices_are_always_present_in_the_approved_order():
    report = agg.aggregate(build(MIXED), SERIES, SEED, B)
    assert tuple(s.retrieval_type for s in report.slices) == RETRIEVAL_TYPES
    assert [s.n for s in report.slices] == [10, 9, 0, 12] and report.overall.n == 31


def test_a_slice_with_ten_or_more_questions_is_aggregated_like_the_overall_set():
    report = agg.aggregate(build(MIXED), SERIES, SEED, B)
    for slice_report, retrieval_type in ((report.slices[0], "en_to_en"), (report.slices[3], "hi_to_en")):
        members = build({retrieval_type: slice_report.n})
        direct = agg.aggregate(members, SERIES, SEED, B).overall
        for name in ("hit", "score"):
            assert series_of(slice_report, name) == series_of(direct, name)
            assert series_of(slice_report, name).status == "ok"


def test_the_boundary_nine_versus_ten_questions():
    nine = agg.aggregate(build({"en_to_en": 9}), SERIES, SEED, B).slices[0]
    ten = agg.aggregate(build({"en_to_en": 10}), SERIES, SEED, B).slices[0]
    assert {r.status for r in nine.series} == {"suppressed"}
    assert {r.status for r in ten.series} == {"ok"}


@pytest.mark.parametrize("count", [0, 1, 2, 5, 9])
def test_a_small_slice_reports_counts_and_outcomes_and_nothing_else(count):
    slice_report = agg.aggregate(build({"hi_to_hi": count}), SERIES, SEED, B).slices[1]
    assert slice_report.n == count
    for result in slice_report.series:
        assert result.status == "suppressed" and result.n == count
        assert (result.successes, result.rate_or_mean, result.interval) == (None, None, None)
        assert len(result.outcomes) == count


def test_the_individual_outcomes_of_a_suppressed_slice_are_kept_in_order():
    observations = [one("z", "hi_to_hi", hit=1, score=0.5), one("a", "hi_to_hi", hit=0, score=0.25)]
    slice_report = agg.aggregate(observations, SERIES, SEED, B).slices[1]
    assert series_of(slice_report, "hit").outcomes == (("a", 0), ("z", 1))
    assert series_of(slice_report, "score").outcomes == (("a", 0.25), ("z", 0.5))


def test_the_overall_set_is_not_suppressed_even_when_small():
    result = series_of(agg.aggregate(build({"en_to_en": 3}), SERIES, SEED, B).overall, "hit")
    assert result.status == "ok" and result.n == 3 and result.interval is not None


def test_an_empty_slice_is_shown_with_n_zero_not_omitted():
    report = agg.aggregate(build({"en_to_en": 10}), SERIES, SEED, B)
    empty = report.slices[2]
    assert (empty.retrieval_type, empty.n) == ("en_to_hi", 0)
    assert [(r.status, r.outcomes) for r in empty.series] == [("suppressed", ())] * 2


def test_slices_hold_only_their_own_questions():
    report = agg.aggregate(build(MIXED), SERIES, SEED, B)
    for slice_report in report.slices:
        for result in slice_report.series:
            assert all(question_id.startswith(slice_report.retrieval_type + "-") for question_id, _ in result.outcomes)


def test_every_question_lands_in_exactly_one_slice():
    report = agg.aggregate(build(MIXED), SERIES, SEED, B)
    assert sum(s.n for s in report.slices) == report.overall.n


# ---------- D. empty and one-question data ----------

def test_no_observations_at_all_is_a_valid_report():
    report = agg.aggregate([], SERIES, SEED, B)
    assert report.overall.n == 0 and [r.status for r in report.overall.series] == ["not_estimable"] * 2
    assert all(r.outcomes == () and (r.successes, r.rate_or_mean, r.interval) == (None, None, None)
               for r in report.overall.series)
    assert [s.n for s in report.slices] == [0, 0, 0, 0]
    assert all(r.status == "suppressed" for s in report.slices for r in s.series)


def test_a_single_question_does_not_fail_the_report():
    report = agg.aggregate([one(hit=1, score=0.75)], SERIES, SEED, B)
    hit, score = series_of(report.overall, "hit"), series_of(report.overall, "score")
    assert (hit.status, hit.successes, hit.rate_or_mean, hit.interval) == ("ok", 1, 1.0, st.wilson_interval(1, 1))
    assert (score.status, score.rate_or_mean, score.interval, score.outcomes) == (
        "not_estimable", 0.75, None, (("q1", 0.75),))
    assert report.slices[0].n == 1 and series_of(report.slices[0], "score").status == "suppressed"


def test_two_questions_are_the_first_estimable_continuous_case():
    report = agg.aggregate([one("a", score=0.0), one("b", score=1.0)], SERIES, SEED, B)
    result = series_of(report.overall, "score")
    assert result.status == "ok" and (result.interval.lower, result.interval.upper) == (0.0, 1.0)


# ---------- E. truncation counts ----------

def trunc(retrieval_type, i):
    return {0: (), 1: (5,), 2: (5, 10), 3: (1, 3, 5, 10, 20)}[i % 4]


def test_truncated_questions_are_counted_per_cutoff_for_every_default_cutoff():
    report = agg.aggregate(build({"en_to_en": 10}, truncated=trunc), SERIES, SEED, B)
    # i = 0..9 use the patterns 0,1,2,3,0,1,2,3,0,1 of trunc(); the counts are derived independently here.
    expected = {c: sum(1 for i in range(10) if c in trunc("x", i)) for c in CUTOFFS}
    assert expected == {1: 2, 3: 2, 5: 7, 10: 4, 20: 2}
    assert report.overall.truncation == tuple((c, expected[c]) for c in CUTOFFS)
    assert report.slices[0].truncation == report.overall.truncation


def test_zero_counts_are_reported_and_cutoffs_follow_the_declaration():
    report = agg.aggregate(build({"en_to_en": 4}), SERIES, SEED, B, cutoffs=(2, 4))
    assert report.overall.truncation == ((2, 0), (4, 0))
    assert agg.aggregate([], SERIES, SEED, B, cutoffs=()).overall.truncation == ()


def test_suppressed_and_empty_slices_still_report_truncation():
    observations = build({"hi_to_hi": 3}, truncated=trunc)
    report = agg.aggregate(observations, SERIES, SEED, B)
    assert report.slices[1].truncation == ((1, 0), (3, 0), (5, 2), (10, 1), (20, 0))
    assert report.slices[2].truncation == tuple((c, 0) for c in CUTOFFS)


def test_truncation_never_changes_the_values_or_the_questions():
    plain = agg.aggregate(build({"en_to_en": 10}), SERIES, SEED, B)
    flagged = agg.aggregate(build({"en_to_en": 10}, truncated=trunc), SERIES, SEED, B)
    assert flagged.overall.series == plain.overall.series and flagged.overall.n == plain.overall.n


def test_truncation_counts_split_by_slice():
    observations = build({"en_to_en": 2, "hi_to_en": 2}, truncated=lambda rt, i: (5,) if rt == "hi_to_en" else ())
    report = agg.aggregate(observations, SERIES, SEED, B, cutoffs=(5,))
    assert [s.truncation for s in report.slices] == [((5, 0),), ((5, 0),), ((5, 0),), ((5, 2),)]
    assert report.overall.truncation == ((5, 2),)


# ---------- F. comparison ----------

def pair(counts, shift=0.25):
    baseline = build(counts, hit=lambda i: 0, score=lambda i: (i % 3) / 4)
    candidate = build(counts, hit=lambda i: 1 if i % 2 else 0, score=lambda i: (i % 3) / 4 + shift)
    return baseline, candidate


def test_a_constant_improvement_gives_a_zero_width_interval_in_every_estimable_slice():
    baseline, candidate = pair({"en_to_en": 10, "hi_to_en": 11})
    report = agg.compare(baseline, candidate, SERIES, SEED, B)
    for slice_report in (report.overall, report.slices[0], report.slices[3]):
        result = series_of(slice_report, "score")
        assert result.status == "ok" and (result.interval.lower, result.interval.upper) == (0.25, 0.25)
        assert result.mean_difference == 0.25


def test_the_comparison_equals_the_paired_primitive_with_the_same_settings():
    baseline, candidate = pair({"en_to_en": 12})
    report = agg.compare(baseline, candidate, SERIES, 5, 1500)
    ids = sorted(o.question_id for o in baseline)
    for spec in SERIES:
        before = [(i, v) for i, v in sorted((o.question_id, dict(o.values)[spec.name]) for o in baseline)]
        after = [(i, v) for i, v in sorted((o.question_id, dict(o.values)[spec.name]) for o in candidate)]
        expected = st.paired_bootstrap_interval(before, after, 5, 1500, binary=spec.kind == "binary")
        result = series_of(report.slices[0], spec.name)
        assert result.interval == expected and result.mean_difference == expected.mean_difference
        assert result.outcomes == tuple((i, b, a) for (i, b), (_, a) in zip(before, after))
        assert type(result.interval) is st.PairedBootstrapInterval and result.n == len(ids)


def test_the_direction_is_candidate_minus_baseline():
    baseline, candidate = pair({"en_to_en": 10})
    forward = series_of(agg.compare(baseline, candidate, SERIES, SEED, B).overall, "hit")
    backward = series_of(agg.compare(candidate, baseline, SERIES, SEED, B).overall, "hit")
    assert forward.mean_difference == 0.5 and backward.mean_difference == -0.5


def test_a_small_slice_has_no_paired_difference_but_keeps_the_paired_outcomes():
    baseline, candidate = pair({"hi_to_hi": 9, "en_to_en": 10})
    report = agg.compare(baseline, candidate, SERIES, SEED, B)
    small = report.slices[1]
    assert small.n == 9
    for result in small.series:
        assert (result.status, result.mean_difference, result.interval) == ("suppressed", None, None)
        assert len(result.outcomes) == 9 and all(len(row) == 3 for row in result.outcomes)
    assert series_of(report.overall, "hit").status == "ok" and series_of(report.slices[0], "hit").status == "ok"


def test_empty_and_one_question_comparisons_do_not_fail():
    empty = agg.compare([], [], SERIES, SEED, B)
    assert empty.overall.n == 0 and {r.status for r in empty.overall.series} == {"not_estimable"}
    assert all(r.mean_difference is None for r in empty.overall.series)
    single = agg.compare([one(hit=0, score=0.25)], [one(hit=1, score=0.75)], SERIES, SEED, B)
    result = series_of(single.overall, "score")
    assert (result.status, result.mean_difference, result.interval, result.outcomes) == (
        "not_estimable", 0.5, None, (("q1", 0.25, 0.75),))
    assert series_of(single.slices[0], "hit").status == "suppressed"


def test_each_version_keeps_its_own_truncation_counts():
    baseline = build({"en_to_en": 10}, truncated=lambda rt, i: (5,) if i < 3 else ())
    candidate = build({"en_to_en": 10}, truncated=lambda rt, i: (5, 10) if i < 7 else ())
    report = agg.compare(baseline, candidate, SERIES, SEED, B, cutoffs=(5, 10))
    assert report.overall.baseline_truncation == ((5, 3), (10, 0))
    assert report.overall.candidate_truncation == ((5, 7), (10, 7))
    assert report.slices[0].baseline_truncation == report.overall.baseline_truncation


def test_the_comparison_records_its_settings_and_slice_order():
    baseline, candidate = pair({"en_to_en": 10})
    report = agg.compare(baseline, candidate, SERIES, 9, 1200, cutoffs=(1,))
    assert (report.seed, report.resamples, report.confidence_level, report.min_slice_questions,
            report.cutoffs) == (9, 1200, 0.95, 10, (1,))
    assert tuple(s.retrieval_type for s in report.slices) == RETRIEVAL_TYPES
    assert report.overall.retrieval_type is None


def test_the_comparison_is_independent_of_the_order_of_either_input():
    baseline, candidate = pair({"en_to_en": 10, "hi_to_en": 10})
    expected = agg.compare(baseline, candidate, SERIES, SEED, B)
    rng = random.Random(3)
    for _ in range(3):
        a, b = list(baseline), list(candidate)
        rng.shuffle(a)
        rng.shuffle(b)
        assert agg.compare(a, b, SERIES, SEED, B) == expected
    assert agg.compare(tuple(baseline), tuple(candidate), SERIES, SEED, B) == expected


def test_the_two_versions_are_paired_not_resampled_separately():
    # A constant shift gives a zero-width paired interval; two separate bootstraps could not.
    baseline, candidate = pair({"en_to_en": 12})
    result = series_of(agg.compare(baseline, candidate, SERIES, SEED, B).overall, "score")
    assert result.interval.upper - result.interval.lower == 0.0
    separate = series_of(agg.aggregate(candidate, SERIES, SEED, B).overall, "score").interval
    assert separate.upper - separate.lower > 0


# ---------- G. comparison: matching of ids and retrieval types ----------

@pytest.mark.parametrize(
    "baseline, candidate",
    [
        pytest.param([one("a"), one("b")], [one("a"), one("b"), one("c")], id="candidate-extra"),
        pytest.param([one("a"), one("b"), one("c")], [one("a"), one("b")], id="candidate-missing"),
        pytest.param([one("a"), one("b")], [one("a"), one("x")], id="different-id"),
        pytest.param([one("a"), one("b")], [one("A"), one("B")], id="case-differs"),
        pytest.param([one("a")], [], id="empty-candidate"),
        pytest.param([], [one("a")], id="empty-baseline"),
    ],
)
def test_different_id_sets_are_a_value_error_and_nothing_is_intersected(baseline, candidate):
    with pytest.raises(ValueError) as caught:
        agg.compare(baseline, candidate, SERIES, SEED, B)
    assert str(caught.value) == "baseline and candidate must have the same question ids"


def test_the_same_id_with_a_different_retrieval_type_is_a_value_error():
    with pytest.raises(ValueError) as caught:
        agg.compare([one("a", "en_to_en"), one("b")], [one("a", "hi_to_en"), one("b")], SERIES, SEED, B)
    assert str(caught.value) == "baseline and candidate must have the same retrieval type for every question"


@pytest.mark.parametrize("which", ["baseline", "candidate"])
def test_duplicate_ids_in_either_version_are_a_value_error(which):
    good = [one("a"), one("b")]
    bad = [one("a"), one("a")]
    arguments = (bad, good) if which == "baseline" else (good, bad)
    with pytest.raises(ValueError) as caught:
        agg.compare(*arguments, SERIES, SEED, B)
    assert str(caught.value) == "question ids must be unique"


def test_each_version_is_validated_against_the_declared_series():
    good = [one("a"), one("b")]
    missing = [agg.QuestionObservation("a", "en_to_en", (("hit", 1),), ()), one("b")]
    with pytest.raises(ValueError):
        agg.compare(good, missing, SERIES, SEED, B)
    with pytest.raises(ValueError):
        agg.compare(missing, good, SERIES, SEED, B)


# ---------- H. validation of types and values, and the error order ----------

@pytest.mark.parametrize("bad", [None, "abc", {}, 5, (x for x in ()), {one()}, b"ab"])
def test_observations_must_be_a_list_or_tuple(bad):
    with pytest.raises(TypeError):
        agg.aggregate(bad, SERIES, SEED, B)
    with pytest.raises(TypeError):
        agg.compare(bad, [one()], SERIES, SEED, B)
    with pytest.raises(TypeError):
        agg.compare([one()], bad, SERIES, SEED, B)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(("q", "en_to_en", (), ()), id="plain-tuple"),
        pytest.param({"question_id": "q"}, id="dict"),
        pytest.param(None, id="none"),
        pytest.param(agg.SeriesSpec("hit", "binary"), id="other-dataclass"),
        pytest.param(agg.QuestionObservation(1, "en_to_en", (), ()), id="int-id"),
        pytest.param(agg.QuestionObservation("q", None, (), ()), id="none-retrieval-type"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", [], ()), id="list-values"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (), []), id="list-truncated"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (["hit", 1],), ()), id="list-pair"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit",),), ()), id="short-pair"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit", 1, 2),), ()), id="long-pair"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", ((1, 1),), ()), id="int-name"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit", True),), ()), id="bool-value"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit", "1"),), ()), id="str-value"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit", None),), ()), id="none-value"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (("hit", Fraction(1)),), ()), id="fraction-value"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (), (True,)), id="bool-cutoff"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (), (5.0,)), id="float-cutoff"),
        pytest.param(agg.QuestionObservation("q", "en_to_en", (), (IntSubclass(5),)), id="int-subclass-cutoff"),
    ],
)
def test_each_observation_is_checked_for_type(bad):
    with pytest.raises(TypeError):
        agg.aggregate([one("a"), bad], SERIES, SEED, B)


@pytest.mark.parametrize(
    "bad",
    [None, "ab", {}, 3, [("hit", "binary")], (HIT, ("score", "continuous")), (HIT, None),
     (agg.SeriesSpec(1, "binary"),), (agg.SeriesSpec("hit", None),), (agg.SeriesSpec("hit", True),)],
)
def test_series_must_be_a_list_or_tuple_of_series_specs_with_str_fields(bad):
    with pytest.raises(TypeError):
        agg.aggregate([one()], bad, SEED, B)


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None, b"1", IntSubclass(1), [1]])
def test_a_seed_of_the_wrong_type_is_a_type_error(seed):
    with pytest.raises(TypeError):
        agg.aggregate([one()], SERIES, seed, B)


@pytest.mark.parametrize("resamples", [True, False, 1000.0, "1000", None, IntSubclass(1000), [1000]])
def test_resamples_of_the_wrong_type_is_a_type_error(resamples):
    with pytest.raises(TypeError):
        agg.aggregate([one()], SERIES, SEED, resamples)


@pytest.mark.parametrize("cutoffs", [None, 5, "5", {5}, [True], [5.0], ["5"], (IntSubclass(5),), [[5]]])
def test_cutoffs_must_be_a_list_or_tuple_of_ints(cutoffs):
    with pytest.raises(TypeError):
        agg.aggregate([one()], SERIES, SEED, B, cutoffs=cutoffs)


def test_the_settings_are_validated_even_when_every_slice_is_suppressed_or_empty():
    with pytest.raises(ValueError):
        agg.aggregate([], SERIES, -1, B)
    with pytest.raises(ValueError):
        agg.aggregate([], SERIES, SEED, 999)
    with pytest.raises(ValueError):
        agg.aggregate([one("a", "hi_to_hi")], SERIES, SEED, 5)


@pytest.mark.parametrize("seed", [-1, -(10**30)])
def test_a_negative_seed_is_a_value_error(seed):
    with pytest.raises(ValueError):
        agg.aggregate([one()], SERIES, seed, B)


@pytest.mark.parametrize("resamples", [999, 1, 0, -1000])
def test_resamples_below_the_minimum_is_a_value_error(resamples):
    with pytest.raises(ValueError):
        agg.aggregate([one()], SERIES, SEED, resamples)


@pytest.mark.parametrize("seed", [0, 1, 10**30])
def test_zero_and_large_seeds_are_accepted(seed):
    assert agg.aggregate([one()], SERIES, seed, B).seed == seed


@pytest.mark.parametrize(
    "series",
    [
        pytest.param([agg.SeriesSpec("", "binary")], id="empty-name"),
        pytest.param([HIT, agg.SeriesSpec("hit", "continuous")], id="duplicate-name"),
        pytest.param([agg.SeriesSpec("hit", "Binary")], id="wrong-case-kind"),
        pytest.param([agg.SeriesSpec("hit", "")], id="empty-kind"),
        pytest.param([agg.SeriesSpec("hit", "ordinal")], id="unknown-kind"),
    ],
)
def test_series_value_errors(series):
    with pytest.raises(ValueError):
        agg.aggregate([], series, SEED, B)


@pytest.mark.parametrize("cutoffs", [[0], [21], [-1], [5, 5], [10, 5], [1, 3, 3], [1, 22]])
def test_cutoff_value_errors(cutoffs):
    with pytest.raises(ValueError):
        agg.aggregate([], SERIES, SEED, B, cutoffs=cutoffs)


@pytest.mark.parametrize("cutoffs", [(), (1,), (20,), (1, 3, 5, 10, 20), [2, 4]])
def test_valid_cutoffs_are_accepted(cutoffs):
    assert agg.aggregate([], SERIES, SEED, B, cutoffs=cutoffs).cutoffs == tuple(cutoffs)


@pytest.mark.parametrize("retrieval_type", ["EN_TO_EN", "fr_to_en", "", "en-to-en", "en_to_en ", "en_to_fr", "unknown"])
def test_an_invalid_retrieval_type_is_a_value_error(retrieval_type):
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a"), one("b", retrieval_type)], SERIES, SEED, B)
    assert str(caught.value) == "an observation has an invalid retrieval type"


@pytest.mark.parametrize(
    "values",
    [
        pytest.param((("hit", 1),), id="missing-score"),
        pytest.param((("score", 0.5),), id="missing-hit"),
        pytest.param((), id="no-values"),
        pytest.param((("hit", 1), ("score", 0.5), ("extra", 0.5)), id="extra-series"),
        pytest.param((("hit", 1), ("hit", 1), ("score", 0.5)), id="duplicate-series"),
        pytest.param((("hit", 1), ("other", 0.5)), id="wrong-name"),
        pytest.param((("hit", 1), ("hit", 1)), id="duplicate-instead-of-missing"),
    ],
)
def test_every_observation_needs_exactly_one_value_per_declared_series(values):
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a"), agg.QuestionObservation("b", "en_to_en", values, ())], SERIES, SEED, B)
    assert str(caught.value) == "each observation must have exactly one value for every declared series"


def test_a_missing_series_is_rejected_even_in_a_suppressed_slice():
    bad = agg.QuestionObservation("a", "hi_to_hi", (("hit", 1),), ())
    with pytest.raises(ValueError):
        agg.aggregate([bad], SERIES, SEED, B)
    with pytest.raises(ValueError):
        agg.compare([bad], [bad], SERIES, SEED, B)


@pytest.mark.parametrize("value", [0.5, 2, -1, 0.9999999999999999, float("nan"), float("inf"), 10**400])
def test_a_binary_value_must_be_zero_or_one(value):
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a"), one("b", hit=value)], SERIES, SEED, B)
    assert str(caught.value) == "a binary value must be 0 or 1"


@pytest.mark.parametrize("value", [-0.1, 1.1, 2, -1, float("nan"), float("inf"), float("-inf"), 10**400, 1e300])
def test_a_continuous_value_must_be_finite_and_from_zero_to_one(value):
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a"), one("b", score=value)], SERIES, SEED, B)
    assert str(caught.value) == "a continuous value must be finite and from 0 to 1"


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, -0.0])
def test_binary_zero_and_one_in_int_and_float_form_are_accepted(value):
    assert agg.aggregate([one("a", hit=value), one("b", hit=1)], SERIES, SEED, B).overall.n == 2


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 0.5, 5e-324, 0.9999999999999999])
def test_continuous_values_from_zero_to_one_inclusive_are_accepted(value):
    assert agg.aggregate([one("a", score=value), one("b")], SERIES, SEED, B).overall.n == 2


def test_an_invalid_value_in_a_suppressed_slice_still_raises():
    with pytest.raises(ValueError):
        agg.aggregate([one("a", "hi_to_hi", score=7.0)], SERIES, SEED, B)


@pytest.mark.parametrize(
    "truncated",
    [(10, 5), (5, 5), (0,), (21,), (4,), (1, 3, 3)],
)
def test_truncated_at_must_be_ascending_and_drawn_from_the_declared_cutoffs(truncated):
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a", truncated=truncated)], SERIES, SEED, B)
    assert str(caught.value) == "truncated_at must be ascending and drawn from the declared cutoffs"


def test_truncated_at_is_checked_against_custom_cutoffs():
    agg.aggregate([one("a", truncated=(4,))], SERIES, SEED, B, cutoffs=(4,))
    with pytest.raises(ValueError):
        agg.aggregate([one("a", truncated=(5,))], SERIES, SEED, B, cutoffs=(4,))


def test_duplicate_question_ids_are_a_value_error_even_across_slices():
    with pytest.raises(ValueError) as caught:
        agg.aggregate([one("a", "en_to_en"), one("a", "hi_to_hi")], SERIES, SEED, B)
    assert str(caught.value) == "question ids must be unique"


def test_every_type_check_comes_before_every_value_check():
    bad_value = [one("a", score=9.0), one("a")]        # out of range and a duplicate id
    with pytest.raises(TypeError):
        agg.aggregate(bad_value, [HIT, None], SEED, B)
    with pytest.raises(TypeError):
        agg.aggregate(bad_value, SERIES, "x", B)
    with pytest.raises(TypeError):
        agg.aggregate(bad_value, SERIES, -1, 1.5)
    with pytest.raises(TypeError):
        agg.aggregate(bad_value, SERIES, -1, 5, cutoffs=(1.5,))
    with pytest.raises(TypeError):
        agg.aggregate([one("a", score=9.0), (1, 2)], SERIES, SEED, B)               # later item type vs earlier value
    with pytest.raises(TypeError):
        agg.compare(bad_value, [agg.QuestionObservation("q", "en_to_en", (), [])], SERIES, SEED, B)
    with pytest.raises(TypeError):
        agg.compare([one("a", score=9.0)], [one("a")], SERIES, SEED, True)
    with pytest.raises(TypeError):
        agg.compare(bad_value, [one("a")], SERIES, -1, 5, cutoffs=(1.5,))          # cutoff type vs seed value
    with pytest.raises(TypeError):
        agg.compare([one("a")], [one("a")], SERIES, -1, 5, cutoffs=(1.5,))


def test_the_value_checks_run_in_the_documented_order():
    good = [one("a"), one("b")]
    cases = [
        (good, [HIT, SCORE], dict(seed=-1, resamples=5, cutoffs=[0])),
        (good, [HIT, SCORE], dict(seed=1, resamples=5, cutoffs=[0])),
        (good, [agg.SeriesSpec("", "x"), agg.SeriesSpec("", "x")], dict(seed=1, resamples=B, cutoffs=[0])),
        (good, [agg.SeriesSpec("a", "x"), agg.SeriesSpec("a", "x")], dict(seed=1, resamples=B, cutoffs=[0])),
        (good, [agg.SeriesSpec("a", "x")], dict(seed=1, resamples=B, cutoffs=[0])),
        (good, SERIES, dict(seed=1, resamples=B, cutoffs=[0])),
        ([one("a", "bad", hit=5), one("a")], SERIES, dict(seed=1, resamples=B, cutoffs=(5,))),
        ([one("a", hit=5), agg.QuestionObservation("a", "en_to_en", (), ())], SERIES, dict(seed=1, resamples=B, cutoffs=(5,))),
        ([one("a", hit=5, truncated=(4,)), one("a")], SERIES, dict(seed=1, resamples=B, cutoffs=(5,))),
        ([one("a", truncated=(4,)), one("a")], SERIES, dict(seed=1, resamples=B, cutoffs=(5,))),
        ([one("a"), one("a")], SERIES, dict(seed=1, resamples=B, cutoffs=(5,))),
    ]
    messages = []
    for observations, series, options in cases:
        with pytest.raises(ValueError) as caught:
            agg.aggregate(observations, series, options["seed"], options["resamples"], cutoffs=options["cutoffs"])
        messages.append(str(caught.value))
    assert messages == [
        "seed must not be negative",
        "resamples is below the minimum",
        "a series name must not be empty",
        "series names must be unique",
        "a series kind must be 'binary' or 'continuous'",
        "cutoffs must be ascending, unique and from 1 to the maximum rank",
        "an observation has an invalid retrieval type",
        "each observation must have exactly one value for every declared series",
        "a binary value must be 0 or 1",
        "truncated_at must be ascending and drawn from the declared cutoffs",
        "question ids must be unique",
    ]


def test_in_a_comparison_the_baseline_is_validated_before_the_candidate_and_both_before_id_matching():
    bad_baseline = [one("a", hit=5), one("b")]
    bad_candidate = [one("a", score=5.0), one("b")]
    with pytest.raises(ValueError) as caught:
        agg.compare(bad_baseline, bad_candidate, SERIES, SEED, B)
    assert str(caught.value) == "a binary value must be 0 or 1"
    with pytest.raises(ValueError) as caught:
        agg.compare([one("a"), one("b")], bad_candidate, SERIES, SEED, B)
    assert str(caught.value) == "a continuous value must be finite and from 0 to 1"
    with pytest.raises(ValueError) as caught:
        agg.compare([one("a")], [one("a", score=5.0), one("zzz")], SERIES, SEED, B)
    assert str(caught.value) == "a continuous value must be finite and from 0 to 1"
    with pytest.raises(ValueError) as caught:
        agg.compare([one("a", "en_to_en")], [one("b", "hi_to_en")], SERIES, SEED, B)
    assert str(caught.value) == "baseline and candidate must have the same question ids"


def test_error_messages_are_fixed_and_contain_no_input():
    secret = [one(CANARY, retrieval_type=CANARY), one(CANARY, score=7.5)]
    calls = [
        lambda: agg.aggregate(CANARY, SERIES, SEED, B),
        lambda: agg.aggregate([CANARY], SERIES, SEED, B),
        lambda: agg.aggregate([agg.QuestionObservation(CANARY, 1, (), ())], SERIES, SEED, B),
        lambda: agg.aggregate([agg.QuestionObservation(CANARY, "en_to_en", ((CANARY, True),), ())], SERIES, SEED, B),
        lambda: agg.aggregate([], [agg.SeriesSpec(CANARY, 1)], SEED, B),
        lambda: agg.aggregate([], [agg.SeriesSpec(CANARY, CANARY)], SEED, B),
        lambda: agg.aggregate([], [agg.SeriesSpec("", CANARY)], SEED, B),
        lambda: agg.aggregate([], SERIES, CANARY, B),
        lambda: agg.aggregate([], SERIES, -123456, B),
        lambda: agg.aggregate([], SERIES, SEED, 777),
        lambda: agg.aggregate([], SERIES, SEED, B, cutoffs=[CANARY]),
        lambda: agg.aggregate([], SERIES, SEED, B, cutoffs=[4242]),
        lambda: agg.aggregate(secret, SERIES, SEED, B),
        lambda: agg.aggregate([one(CANARY, score=7.5)], SERIES, SEED, B),
        lambda: agg.aggregate([one(CANARY, hit=7.5)], SERIES, SEED, B),
        lambda: agg.aggregate([one(CANARY, truncated=(4242,))], SERIES, SEED, B),
        lambda: agg.aggregate([one(CANARY), one(CANARY)], SERIES, SEED, B),
        lambda: agg.compare([one(CANARY)], [one("x")], SERIES, SEED, B),
        lambda: agg.compare([one(CANARY, "en_to_en")], [one(CANARY, "hi_to_en")], SERIES, SEED, B),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        text = str(caught.value)
        assert all(token not in text for token in (CANARY, "123456", "777", "4242", "7.5"))


# ---------- I. determinism, order independence and purity ----------

def test_the_same_inputs_always_give_the_same_report():
    observations = build(MIXED)
    first = agg.aggregate(observations, SERIES, SEED, B)
    for _ in range(2):
        assert agg.aggregate(observations, SERIES, SEED, B) == first
    assert agg.aggregate(tuple(observations), tuple(SERIES), SEED, B) == first


def test_the_input_order_of_the_observations_does_not_change_the_report():
    observations = build(MIXED, truncated=trunc)
    expected = agg.aggregate(observations, SERIES, SEED, B)
    rng = random.Random(1)
    for _ in range(3):
        shuffled = list(observations)
        rng.shuffle(shuffled)
        assert agg.aggregate(shuffled, SERIES, SEED, B) == expected


def test_the_order_of_the_values_inside_an_observation_does_not_matter():
    forward = [one(f"q{i}", hit=i % 2, score=i / 10) for i in range(6)]
    reversed_pairs = [agg.QuestionObservation(o.question_id, o.retrieval_type, tuple(reversed(o.values)),
                                              o.truncated_at) for o in forward]
    assert agg.aggregate(reversed_pairs, SERIES, SEED, B) == agg.aggregate(forward, SERIES, SEED, B)


def test_the_global_random_state_is_neither_used_nor_changed():
    random.seed(77)
    expected = random.random()
    random.seed(77)
    agg.aggregate(build(MIXED), SERIES, SEED, B)
    agg.compare(*pair({"en_to_en": 10}), SERIES, SEED, B)
    assert random.random() == expected


def test_the_inputs_are_not_modified():
    observations = build(MIXED)
    series = list(SERIES)
    snapshot = (list(observations), list(series))
    agg.aggregate(observations, series, SEED, B)
    baseline, candidate = pair({"en_to_en": 10})
    snapshots = (list(baseline), list(candidate))
    agg.compare(baseline, candidate, series, SEED, B)
    assert (observations, series) == snapshot and (baseline, candidate) == snapshots
    assert all(a is b for a, b in zip(observations, snapshot[0]))


def test_result_types_are_exact():
    report = agg.aggregate(build(MIXED), SERIES, SEED, B)
    assert type(report) is agg.Report and type(report.overall) is agg.SliceReport
    assert type(report.slices) is tuple and type(report.overall.series) is tuple
    result = series_of(report.slices[0], "hit")
    assert type(result) is agg.SeriesResult and type(result.outcomes) is tuple
    assert type(result.n) is int and type(result.successes) is int and type(result.rate_or_mean) is float
    assert all(type(c) is int and type(k) is int for c, k in report.overall.truncation)
    comparison = agg.compare(*pair({"en_to_en": 10}), SERIES, SEED, B)
    assert type(comparison) is agg.ComparisonReport and type(comparison.slices[0]) is agg.PairedSliceReport
    assert type(series_of(comparison.slices[0], "hit")) is agg.PairedSeriesResult


# ---------- J. import isolation ----------

def test_the_module_adds_no_heavy_dependency_and_no_unrelated_evaluation_module():
    code = (
        "import sys\n"
        "import app.rag.evaluation.aggregation as m\n"
        "o = m.QuestionObservation('a', 'en_to_en', (('hit', 1),), ())\n"
        "r = m.aggregate([o], [m.SeriesSpec('hit', 'binary')], 1, 1000)\n"
        "assert r.overall.n == 1\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy))\n"
        "print(sorted(x for x in sys.modules if x.startswith('app.rag.evaluation.')))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    heavy, evaluation = result.stdout.strip().splitlines()
    assert heavy == "[]"
    # The only evaluation modules pulled in are the ones the module declares it imports (metrics and
    # structure, which bring schema, intervals and their own imports) plus statistics; no I/O or
    # unrelated evaluation module (dataset, parsing, validation, anchoring, near_duplicates).
    loaded = set(eval(evaluation))
    assert {"app.rag.evaluation.statistics", "app.rag.evaluation.metrics",
            "app.rag.evaluation.structure"} <= loaded
    assert not loaded & {f"app.rag.evaluation.{name}" for name in
                         ("dataset", "dataset_metadata", "parsing", "validation", "anchoring", "near_duplicates")}
