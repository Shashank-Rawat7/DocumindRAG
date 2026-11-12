"""E1.3.5: score_ranking, canonical order, the oracle and the random baseline.

A  constants and result types
B  score_ranking: the metric list, its order, and equality with the metric functions
C  canonical_order
D  exact coverage (the oracle's tie-break is never made from a rounded float)
E  oracle: hand-computed rankings
F  oracle: unreachable units, reachable-only scores, short pools and alternatives
G  oracle: an independent brute-force reference on random cases
H  _bounded: bound validation, rejection sampling and its uniformity by construction
I  random_ranking: an independent full-shuffle reference, pinned rankings, seeds
J  random_baseline: per-seed scores, summaries, percentiles, reproducibility record
K  validation: types before values, error order, fixed messages
L  purity and import isolation
"""
import dataclasses
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from app.rag.evaluation import baselines as bl
from app.rag.evaluation import metrics as mt
from app.rag.evaluation import passages as ps
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64
SHA_B = "b" * 64
CANARY = "SECRET-CANARY-7731"
RULES = ("any_overlap", "half_coverage", "full_coverage")
CUTOFFS = (1, 3, 5, 10, 20)


class IntSubclass(int):
    pass


def chunk(key, start, end, sha=SHA):
    return mt.RankedChunk(key, start, end, sha)


def span(key, start, end):
    return s.Span(key, SHA, start, end, "q", SHA)  # the quote plays no part in metrics


def unit(unit_id, *spans):
    return s.EvidenceUnit(unit_id, tuple(spans))


def question(*units, **overrides):
    values = dict(
        question_id="q1", split="dev", answerable=True, question="What?", query_language="en",
        author_type="human", hindi_reviewed=False, expected_answer="A", units=tuple(units),
        source_language="en", difficulty="direct", close_distractor=False,
    )
    values.update(overrides)
    return s.Question(**values)


# A pool of 30 chunks of one document with one gold unit, used by the random baseline tests.
RANDOM_POOL = [chunk("doc-a", i * 50, i * 50 + 80) for i in range(30)]
RANDOM_QUESTION = question(unit("u1", span("doc-a", 100, 200)), unit("u2", span("doc-a", 700, 760)))
SEEDS = tuple(range(1000))


@pytest.fixture(scope="module")
def baseline():
    return bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, SEEDS)


# ---------- A. constants and result types ----------

def test_the_constants():
    assert bl.PRIMARY_RULE == "half_coverage" == mt.DEFAULT_RULE
    assert bl.MIN_RANDOM_SEEDS == 1000
    assert bl.ORACLE_METHOD == "deterministic_greedy_reference"
    assert bl.CANONICAL_KEY == ("corpus_key", "start_offset", "end_offset", "text_sha256")
    assert bl.FAMILIES == ("raw", "distinct") and bl.METRIC_NAMES == ("hit", "unit_recall", "coverage", "mrr")


@pytest.mark.parametrize(
    "cls, names",
    [
        (bl.MetricId, ["family", "metric", "k", "rule"]),
        (bl.RankingScores, ["cutoffs", "metric_ids", "values", "distinct_truncated"]),
        (bl.ReachableScores, ["rule", "unit_count", "scores"]),
        (bl.OracleResult, ["ranking", "chunks_needed", "unit_ids", "unreachable_by_rule", "scores",
                           "reachable_only", "method"]),
        (bl.MetricSummary, ["metric", "n_seeds", "mean", "p2_5", "p95", "p97_5"]),
        (bl.ReproducibilityRecord, ["python_version", "algorithm", "random_source", "canonical_key", "pool_size",
                                    "rank_cap", "percentile_convention"]),
        (bl.RandomBaselineResult, ["seeds", "per_seed", "summaries", "reproducibility"]),
    ],
)
def test_the_data_classes_have_the_documented_fields_and_are_frozen_and_slotted(cls, names):
    assert [f.name for f in dataclasses.fields(cls)] == names
    assert cls.__dataclass_params__.frozen and "__slots__" in vars(cls)


def test_a_result_cannot_be_changed(baseline):
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.seeds = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.per_seed[0].values = ()


# ---------- B. score_ranking ----------

def expected_ids():
    ids = []
    for family in ("raw", "distinct"):
        for name in ("hit", "unit_recall"):
            ids += [bl.MetricId(family, name, k, rule) for rule in RULES for k in CUTOFFS]
        ids += [bl.MetricId(family, "coverage", k, None) for k in CUTOFFS]
        ids += [bl.MetricId(family, "mrr", None, rule) for rule in RULES]
    return ids


WORKED = question(
    unit("u1", span("doc-a", 0, 100)),
    unit("u2", span("doc-a", 200, 300), span("doc-b", 0, 50)),
    unit("u3", span("doc-b", 1000, 1100)),
)
WORKED_RANKING = [
    chunk("doc-c", 0, 500), chunk("doc-a", 0, 60), chunk("doc-a", 250, 300),
    chunk("doc-b", 0, 20), chunk("doc-b", 1000, 1060),
]


def test_score_ranking_has_76_values_in_the_documented_order():
    scores = bl.score_ranking(WORKED, WORKED_RANKING)
    assert list(scores.metric_ids) == expected_ids() and len(scores.values) == 76
    assert scores.cutoffs == CUTOFFS and len(scores.distinct_truncated) == 5
    assert all(type(v) is float for v in scores.values)


def test_coverage_has_no_rule_and_mrr_has_no_cutoff():
    scores = bl.score_ranking(WORKED, WORKED_RANKING)
    for metric_id in scores.metric_ids:
        assert (metric_id.rule is None) == (metric_id.metric == "coverage")
        assert (metric_id.k is None) == (metric_id.metric == "mrr")


def test_worked_example_values_are_hand_computed():
    scores = bl.score_ranking(WORKED, WORKED_RANKING)
    # rank 2: 60% of u1; rank 3: 50% of u2's first span; rank 4: 40% of u2's second span; rank 5: 60% of u3.
    assert scores.get("raw", "hit", 1, "half_coverage") == 0.0
    assert scores.get("raw", "hit", 3, "half_coverage") == 1.0
    assert scores.get("raw", "unit_recall", 3, "half_coverage") == pytest.approx(2 / 3)
    assert scores.get("raw", "unit_recall", 5, "half_coverage") == 1.0
    assert scores.get("raw", "unit_recall", 5, "full_coverage") == 0.0
    assert scores.get("raw", "unit_recall", 5, "any_overlap") == 1.0
    assert scores.get("raw", "coverage", 3) == pytest.approx((0.6 + 0.5 + 0) / 3)
    assert scores.get("raw", "mrr", None, "half_coverage") == 0.5
    assert scores.get("raw", "mrr", None, "full_coverage") == 0.0


def reference_value(question_, ranking, metric_id):
    """The same value from the metric functions, called by name."""
    k, rule = metric_id.k, metric_id.rule
    table = {
        ("raw", "hit"): lambda: mt.hit_at_k(question_, ranking, k, rule),
        ("raw", "unit_recall"): lambda: mt.unit_recall_at_k(question_, ranking, k, rule),
        ("raw", "coverage"): lambda: mt.evidence_coverage_at_k(question_, ranking, k),
        ("raw", "mrr"): lambda: mt.reciprocal_rank(question_, ranking, rule),
        ("distinct", "hit"): lambda: mt.distinct_hit_at_k(question_, ranking, k, rule),
        ("distinct", "unit_recall"): lambda: mt.distinct_unit_recall_at_k(question_, ranking, k, rule),
        ("distinct", "coverage"): lambda: mt.distinct_evidence_coverage_at_k(question_, ranking, k),
        ("distinct", "mrr"): lambda: mt.distinct_reciprocal_rank(question_, ranking, rule),
    }
    return float(table[(metric_id.family, metric_id.metric)]())


def test_every_value_equals_the_metric_function_on_random_rankings():
    rng = random.Random(5)
    for _ in range(25):
        ranking = [chunk(rng.choice("ab"), (a := rng.randrange(0, 150)), a + rng.randrange(1, 80), SHA)
                   for _ in range(rng.randrange(0, 25))]
        q = question(unit("u1", span("a", 20, 90), span("b", 10, 70)), unit("u2", span("a", 100, 160)))
        scores = bl.score_ranking(q, ranking)
        for metric_id, value in zip(scores.metric_ids, scores.values):
            assert value == reference_value(q, ranking, metric_id)
        assert scores.distinct_truncated == tuple(ps.is_truncated(ranking, k) for k in CUTOFFS)


def test_distinct_truncation_flags_are_reported_for_every_cutoff():
    scores = bl.score_ranking(WORKED, WORKED_RANKING)
    assert scores.distinct_truncated == (False, False, False, True, True)  # 5 representatives only
    assert bl.score_ranking(WORKED, []).distinct_truncated == (True,) * 5


def test_get_finds_a_value_and_rejects_an_unknown_metric():
    scores = bl.score_ranking(WORKED, WORKED_RANKING)
    assert scores.get("distinct", "hit", 5, "any_overlap") == reference_value(
        WORKED, WORKED_RANKING, bl.MetricId("distinct", "hit", 5, "any_overlap"))
    for arguments in (("raw", "hit", 2, "half_coverage"), ("raw", "coverage", 5, "half_coverage"), ("x", "hit", 1, "any_overlap")):
        with pytest.raises(ValueError) as caught:
            scores.get(*arguments)
        assert str(caught.value) == "no such metric in these scores"


def test_only_the_first_twenty_chunks_are_scored():
    ranking = [chunk("doc-x", i * 10, i * 10 + 5) for i in range(20)] + [chunk("doc-a", 0, 100)]
    assert bl.score_ranking(WORKED, ranking).get("raw", "hit", 20, "any_overlap") == 0.0


def test_an_empty_ranking_scores_zero():
    scores = bl.score_ranking(WORKED, [])
    assert all(v == 0.0 for v in scores.values)


# ---------- C. canonical order ----------

def test_the_canonical_order_is_corpus_key_start_end_hash():
    pool = [chunk("b", 0, 5), chunk("a", 10, 20), chunk("a", 0, 30), chunk("a", 0, 20, SHA_B), chunk("a", 0, 20, SHA)]
    assert bl.canonical_order(pool) == (
        chunk("a", 0, 20, SHA), chunk("a", 0, 20, SHA_B), chunk("a", 0, 30), chunk("a", 10, 20), chunk("b", 0, 5))


def test_the_canonical_order_does_not_depend_on_the_input_order():
    rng = random.Random(1)
    pool = [chunk(rng.choice("abc"), i, i + 3, SHA if i % 2 else SHA_B) for i in range(40)]
    expected = bl.canonical_order(pool)
    for _ in range(5):
        rng.shuffle(pool)
        assert bl.canonical_order(pool) == expected
    assert bl.canonical_order(tuple(pool)) == expected


def test_keys_compare_as_strings_not_numbers_or_locale():
    pool = [chunk("b", 0, 1), chunk("B", 0, 1), chunk("a", 0, 1), chunk("a-", 0, 1), chunk("é", 0, 1)]
    assert [c.corpus_key for c in bl.canonical_order(pool)] == ["B", "a", "a-", "b", "é"]


def test_offsets_compare_as_numbers():
    pool = [chunk("a", 100, 101), chunk("a", 9, 10)]
    assert [c.start for c in bl.canonical_order(pool)] == [9, 100]


def test_a_duplicate_sort_key_is_a_value_error():
    with pytest.raises(ValueError) as caught:
        bl.canonical_order([chunk("a", 0, 5), chunk("b", 0, 5), chunk("a", 0, 5)])
    assert str(caught.value) == "the pool has two chunks with the same canonical sort key"


def test_chunks_that_differ_only_in_the_hash_are_not_duplicates():
    assert len(bl.canonical_order([chunk("a", 0, 5, SHA), chunk("a", 0, 5, SHA_B)])) == 2


def test_an_empty_pool_is_a_value_error():
    for pool in ([], ()):
        with pytest.raises(ValueError) as caught:
            bl.canonical_order(pool)
        assert str(caught.value) == "the pool must not be empty"


@pytest.mark.parametrize("pool", [None, "abc", {chunk("a", 0, 1)}, 5, [chunk("a", 0, 1), "x"], [("a", 0, 1, SHA)], (x for x in ())])
def test_the_pool_must_be_a_list_or_tuple_of_ranked_chunks(pool):
    with pytest.raises(TypeError):
        bl.canonical_order(pool)


# ---------- D. exact coverage ----------

def test_the_oracle_coverage_equals_the_metric_function_before_its_final_float():
    rng = random.Random(8)
    for _ in range(40):
        units = [unit(f"u{i}", *[span(rng.choice("ab"), (a := rng.randrange(0, 100)), a + rng.randrange(1, 60))
                                  for _ in range(rng.randrange(1, 3))]) for i in range(rng.randrange(1, 5))]
        q = question(*units)
        chosen = [chunk(rng.choice("ab"), (a := rng.randrange(0, 140)), a + rng.randrange(1, 70)) for _ in range(rng.randrange(1, 21))]
        exact = bl._coverage(units, chosen)
        assert type(exact) is Fraction
        assert float(exact) == mt.evidence_coverage_at_k(q, chosen, len(chosen))


def test_the_oracle_breaks_a_coverage_tie_exactly_where_floats_would_collapse():
    # One span of length 2**60. A covers 2**59 + 1 positions and B covers 2**59 + 2: both are relevant
    # (50% rule) and have the same float gain, but B's exact gain is larger. A comes first in canonical
    # order, so a float-based tie-break would pick A.
    length = 1 << 60
    a_chunk, b_chunk = chunk("doc-a", 0, (1 << 59) + 1), chunk("doc-a", 0, (1 << 59) + 2)
    assert float(Fraction((1 << 59) + 1, length)) == float(Fraction((1 << 59) + 2, length))
    result = bl.oracle_baseline(question(unit("u1", span("doc-a", 0, length))), [a_chunk, b_chunk])
    assert result.ranking == (b_chunk, a_chunk) and result.chunks_needed == 1


# ---------- E. oracle: hand-computed rankings ----------

def test_a_single_unit_picks_the_relevant_chunks_then_coverage_then_the_rest_in_canonical_order():
    q = question(unit("u1", span("doc-a", 0, 100)))
    pool = [chunk("doc-b", 0, 500), chunk("doc-a", 40, 100), chunk("doc-a", 0, 40), chunk("doc-a", 0, 60)]
    result = bl.oracle_baseline(q, pool)
    # Step 1: [0,60) and [40,100) both satisfy u1 with gain 0.6; canonical order picks [0,60).
    # Step 2: [40,100) adds [60,100) (gain 0.4); [0,40) adds nothing. Fill: [0,40), then doc-b.
    assert result.ranking == (chunk("doc-a", 0, 60), chunk("doc-a", 40, 100), chunk("doc-a", 0, 40), chunk("doc-b", 0, 500))
    assert result.chunks_needed == 1 and result.unit_ids == ("u1",) and result.method == bl.ORACLE_METHOD
    assert result.scores.get("raw", "hit", 1, "half_coverage") == 1.0
    assert result.scores.get("raw", "coverage", 1) == pytest.approx(0.6)
    assert result.scores.get("raw", "coverage", 3) == 1.0
    assert result.scores.get("raw", "mrr", None, "half_coverage") == 1.0


def test_a_chunk_that_satisfies_two_units_goes_first_and_coverage_breaks_the_next_tie():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-a", 100, 200)), unit("u3", span("doc-a", 300, 400)))
    a_chunk, b_chunk, c_chunk = chunk("doc-a", 0, 200), chunk("doc-a", 0, 100), chunk("doc-a", 100, 200)
    d_chunk, e_chunk = chunk("doc-a", 300, 400), chunk("doc-a", 300, 350)
    result = bl.oracle_baseline(q, [e_chunk, d_chunk, c_chunk, b_chunk, a_chunk])
    # A satisfies two units. Then D and E satisfy u3 (E exactly 50%); D has the larger coverage gain
    # although E comes first in canonical order. Nothing else gains, so the rest follows canonically.
    assert result.ranking == (a_chunk, d_chunk, b_chunk, c_chunk, e_chunk)
    assert result.chunks_needed == 2


def test_coverage_only_chunks_are_chosen_after_the_units_are_satisfied():
    q = question(unit("u1", span("doc-a", 0, 100)))
    x, y, z = chunk("doc-a", 0, 60), chunk("doc-a", 60, 100), chunk("doc-a", 200, 300)
    result = bl.oracle_baseline(q, [z, y, x])
    assert result.ranking == (x, y, z) and result.chunks_needed == 1
    assert result.scores.get("raw", "coverage", 1) == pytest.approx(0.6)
    assert result.scores.get("raw", "coverage", 3) == 1.0


def test_equal_gains_are_broken_by_canonical_order_including_the_hash():
    q = question(unit("u1", span("doc-a", 0, 100)))
    first, second = chunk("doc-a", 0, 60, SHA), chunk("doc-a", 0, 60, SHA_B)
    assert bl.oracle_baseline(q, [second, first]).ranking == (first, second)


def test_a_unit_with_alternative_spans_is_satisfied_by_either():
    q = question(unit("u1", span("doc-a", 0, 100), span("doc-b", 0, 100)))
    result = bl.oracle_baseline(q, [chunk("doc-a", 0, 10), chunk("doc-b", 0, 100)])
    assert result.ranking[0] == chunk("doc-b", 0, 100) and result.chunks_needed == 1


def test_chunks_needed_records_the_greedy_count_without_claiming_optimality():
    # u1 [0,100), u2 [100,200), u3 [200,300). X covers u1 and u2, Y covers u2 and u3, Z covers u1.
    # Greedy takes X (two units, canonical first), then Y (u3): 2 picks. The test only checks the
    # reported count and the method label, which makes no optimality claim.
    q = question(unit("u1", span("a", 0, 100)), unit("u2", span("a", 100, 200)), unit("u3", span("a", 200, 300)))
    result = bl.oracle_baseline(q, [chunk("a", 0, 200), chunk("a", 100, 300), chunk("a", 0, 100)])
    assert result.chunks_needed == 2 and "upper bound" not in result.method


def test_the_oracle_ranking_is_independent_of_the_pool_order():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-b", 0, 100)))
    pool = [chunk("doc-a", i * 20, i * 20 + 60) for i in range(8)] + [chunk("doc-b", i * 30, i * 30 + 70) for i in range(5)]
    expected = bl.oracle_baseline(q, pool)
    rng = random.Random(4)
    for _ in range(4):
        rng.shuffle(pool)
        assert bl.oracle_baseline(q, pool) == expected


# ---------- F. oracle: unreachable units, reachable-only scores, short pools ----------

def test_unreachable_units_are_reported_separately_for_each_rule_and_nest():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-b", 0, 100)), unit("u3", span("doc-c", 0, 10)))
    pool = [chunk("doc-a", 0, 30), chunk("doc-b", 0, 60), chunk("doc-b", 0, 100)]
    result = bl.oracle_baseline(q, pool)
    assert result.unreachable_by_rule == (
        ("any_overlap", ("u3",)),
        ("half_coverage", ("u1", "u3")),
        ("full_coverage", ("u1", "u3")),
    )
    assert result.unit_ids == ("u1", "u2", "u3")


def test_full_coverage_can_make_a_unit_unreachable_that_half_coverage_reaches():
    q = question(unit("u1", span("doc-a", 0, 100)))
    result = bl.oracle_baseline(q, [chunk("doc-a", 0, 60), chunk("doc-a", 40, 100)])
    assert dict(result.unreachable_by_rule) == {"any_overlap": (), "half_coverage": (), "full_coverage": ("u1",)}


def test_with_no_reachable_unit_chunks_needed_is_zero_and_coverage_chunks_still_come_first():
    q = question(unit("u1", span("doc-a", 0, 100)))
    pool = [chunk("doc-a", 0, 30), chunk("doc-b", 0, 10), chunk("doc-a", 500, 600)]
    result = bl.oracle_baseline(q, pool)
    assert dict(result.unreachable_by_rule)["half_coverage"] == ("u1",)
    assert result.chunks_needed == 0
    assert result.ranking == (chunk("doc-a", 0, 30), chunk("doc-a", 500, 600), chunk("doc-b", 0, 10))
    assert result.scores.get("raw", "hit", 1, "half_coverage") == 0.0
    assert result.scores.get("raw", "hit", 1, "any_overlap") == 1.0
    by_rule = {r.rule: r for r in result.reachable_only}
    assert by_rule["any_overlap"].unit_count == 1 and by_rule["any_overlap"].scores is not None
    assert (by_rule["half_coverage"].unit_count, by_rule["half_coverage"].scores) == (0, None)
    assert (by_rule["full_coverage"].unit_count, by_rule["full_coverage"].scores) == (0, None)


def test_reachable_only_recomputes_unit_recall_and_coverage_over_the_reachable_units():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-b", 0, 100)))
    pool = [chunk("doc-a", 0, 100), chunk("doc-b", 0, 30)]            # u2 is unreachable under half coverage
    result = bl.oracle_baseline(q, pool)
    assert dict(result.unreachable_by_rule)["half_coverage"] == ("u2",)
    half = {r.rule: r for r in result.reachable_only}["half_coverage"]
    assert half.unit_count == 1
    sub = dataclasses.replace(q, units=(q.units[0],))
    for k in CUTOFFS:
        assert half.scores.get("raw", "unit_recall", k, "half_coverage") == mt.unit_recall_at_k(sub, result.ranking, k)
        assert half.scores.get("raw", "coverage", k) == mt.evidence_coverage_at_k(sub, result.ranking, k)
        assert half.scores.get("distinct", "unit_recall", k, "half_coverage") == mt.distinct_unit_recall_at_k(sub, result.ranking, k)
        assert half.scores.get("distinct", "coverage", k) == mt.distinct_evidence_coverage_at_k(sub, result.ranking, k)
    assert result.scores.get("raw", "unit_recall", 20, "half_coverage") == 0.5   # all units
    assert half.scores.get("raw", "unit_recall", 20, "half_coverage") == 1.0     # reachable units only
    assert not any(m.metric in ("hit", "mrr") for m in half.scores.metric_ids)


def test_the_validity_checks_hold_on_a_fixed_fixture():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-a", 200, 300)), unit("u3", span("doc-z", 0, 50)))
    pool = [chunk("doc-a", i * 25, i * 25 + 60) for i in range(14)] + [chunk("doc-b", 0, 80)]
    result = bl.oracle_baseline(q, pool)
    half = {r.rule: r for r in result.reachable_only}["half_coverage"]
    assert result.unreachable_by_rule[1] == ("half_coverage", ("u3",))
    assert result.scores.get("raw", "hit", 1, "half_coverage") == 1.0
    assert half.scores.get("raw", "unit_recall", 20, "half_coverage") == 1.0


def test_the_ranking_has_min_of_twenty_and_the_pool_size():
    q = question(unit("u1", span("doc-a", 0, 100)))
    for size, expected in ((1, 1), (3, 3), (20, 20), (21, 20), (45, 20)):
        pool = [chunk("doc-a", i * 10, i * 10 + 20) for i in range(size)]
        result = bl.oracle_baseline(q, pool)
        assert len(result.ranking) == expected and len(set(result.ranking)) == expected


def test_the_greedy_count_never_exceeds_twenty_for_twenty_units():
    units = [unit(f"u{i}", span("doc-a", i * 100, i * 100 + 50)) for i in range(20)]
    pool = [chunk("doc-a", i * 100, i * 100 + 50) for i in range(20)] + [chunk("doc-a", 5000 + i, 5010 + i) for i in range(10)]
    result = bl.oracle_baseline(question(*units), pool)
    assert result.chunks_needed == 20 and len(result.ranking) == 20
    assert result.scores.get("raw", "unit_recall", 20, "half_coverage") == 1.0


def test_distinct_oracle_scores_are_reported_but_may_be_lower_than_the_raw_ones():
    # Two adjacent relevant chunks merge into one passage: only the first stays in the distinct ranking.
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-a", 100, 200)))
    result = bl.oracle_baseline(q, [chunk("doc-a", 0, 100), chunk("doc-a", 100, 200)])
    assert result.scores.get("raw", "unit_recall", 3, "half_coverage") == 1.0
    assert result.scores.get("distinct", "unit_recall", 3, "half_coverage") == 0.5


# ---------- G. oracle: an independent brute-force reference ----------

def reference_oracle(question_, pool):
    """The greedy rule written from scratch over every pool chunk, with explicit position sets."""
    units = question_.units
    ordered = sorted(pool, key=lambda c: (c.corpus_key, c.start, c.end, c.text_sha256))

    def cells(c, sp):
        return set(range(max(c.start, sp.start), min(c.end, sp.end))) if c.corpus_key == sp.corpus_key else set()

    def relevant(c, sp, rule=bl.PRIMARY_RULE):
        shared, size = len(cells(c, sp)), sp.end - sp.start
        return {"any_overlap": shared > 0, "half_coverage": 2 * shared >= size, "full_coverage": shared == size}[rule]

    def coverage(chosen):
        total = Fraction(0)
        for u in units:
            total += max(Fraction(len(set().union(*[cells(c, sp) for c in chosen])), sp.end - sp.start) for sp in u.spans)
        return total / len(units)

    chosen, satisfied, needed = [], set(), 0
    limit = min(20, len(ordered))
    while len(chosen) < limit:
        best = None
        for c in ordered:
            if c in chosen:
                continue
            new = {i for i, u in enumerate(units) if i not in satisfied and any(relevant(c, sp) for sp in u.spans)}
            gain = coverage(chosen + [c]) - coverage(chosen)
            if not new and gain == 0:
                continue
            if best is None or (len(new), gain) > best[0]:
                best = ((len(new), gain), c, new)
        if best is None:
            break
        chosen.append(best[1])
        satisfied |= best[2]
        needed += bool(best[2])
    chosen += [c for c in ordered if c not in chosen][: limit - len(chosen)]
    unreachable = tuple(
        (rule, tuple(u.unit_id for u in units if not any(relevant(c, sp, rule) for sp in u.spans for c in ordered)))
        for rule in RULES)
    return tuple(chosen), needed, unreachable


def random_case(rng):
    keys = "ab"
    units = [unit(f"u{i}", *[span(rng.choice(keys), (a := rng.randrange(0, 50)), a + rng.randrange(1, 30))
                             for _ in range(rng.randrange(1, 3))]) for i in range(rng.randrange(1, 5))]
    pool = {}
    for _ in range(rng.randrange(3, 30)):
        start = rng.randrange(0, 60)
        c = chunk(rng.choice(keys), start, start + rng.randrange(1, 40), rng.choice([SHA, SHA_B]))
        pool[(c.corpus_key, c.start, c.end, c.text_sha256)] = c
    return question(*units), list(pool.values())


def test_the_oracle_matches_an_independent_brute_force_reference_on_random_cases():
    rng = random.Random(2026)
    for _ in range(60):
        q, pool = random_case(rng)
        ranking, needed, unreachable = reference_oracle(q, pool)
        result = bl.oracle_baseline(q, pool)
        assert result.ranking == ranking and result.chunks_needed == needed
        assert result.unreachable_by_rule == unreachable


# ---------- H. _bounded ----------

@pytest.mark.parametrize("bound", [0, -1, -5, (1 << 53) + 1, 1 << 54, 10**30])
def test_a_bound_outside_one_to_two_to_the_53rd_is_a_value_error_and_never_loops(bound):
    with pytest.raises(ValueError) as caught:
        bl._bounded(lambda: 0, bound)
    assert str(caught.value) == "bound must be from 1 to 2**53"


@pytest.mark.parametrize("bound", [True, False, 3.0, "3", None, IntSubclass(3), [3], b"3"])
def test_a_bound_that_is_not_an_exact_int_is_a_type_error(bound):
    with pytest.raises(TypeError) as caught:
        bl._bounded(lambda: 0, bound)
    assert str(caught.value) == "bound must be an int"


@pytest.mark.parametrize("bound", [1, 2, 3, 20, 1 << 52, 1 << 53])
def test_valid_bounds_including_the_extremes_are_accepted(bound):
    assert 0 <= bl._bounded(lambda: 5, bound) < bound


@pytest.mark.parametrize("draw", [-1, 1 << 53, 1.5, True, None, "7"])
def test_a_draw_outside_the_53_bit_range_is_a_value_error(draw):
    with pytest.raises(ValueError) as caught:
        bl._bounded(lambda: draw, 3)
    assert str(caught.value) == "a draw must be an int from 0 to 2**53 - 1"


def test_the_biased_tail_is_rejected_and_the_next_draw_decides():
    # 2**53 % 3 == 2, so the two largest draws are rejected for bound 3.
    scale = 1 << 53
    assert scale % 3 == 2
    draws = iter([scale - 1, scale - 2, 8])
    assert bl._bounded(lambda: next(draws), 3) == 8 % 3
    draws = iter([scale - 3])                                    # the largest accepted draw
    assert bl._bounded(lambda: next(draws), 3) == (scale - 3) % 3


def test_a_power_of_two_bound_never_rejects():
    draws = iter([(1 << 53) - 1])
    assert bl._bounded(lambda: next(draws), 1 << 10) == (1 << 10) - 1


@pytest.mark.parametrize("bound", [1, 2, 3, 5, 7, 20, 30, 1000, 10**6, (1 << 20) + 1, (1 << 52) + 1, 1 << 53])
def test_the_accepted_range_is_an_exact_multiple_of_the_bound(bound):
    # The accepted draws 0 .. limit - 1 map to each residue exactly limit / bound times: no bias.
    scale = 1 << 53
    limit = scale - scale % bound
    assert limit % bound == 0 and scale - limit < bound
    seen = iter([limit, limit + (scale - limit) // 2 if scale > limit else 0, 0])
    value = bl._bounded(lambda: next(seen) if limit < scale else 0, bound)
    assert 0 <= value < bound


def test_a_random_draw_is_the_53_bit_integer_behind_random():
    rng_a, rng_b = random.Random(11), random.Random(11)
    for _ in range(200):
        r = rng_a.random()
        w = bl._draw53(rng_b)
        assert Fraction(r) * (1 << 53) == w and w / (1 << 53) == r


# ---------- I. random_ranking ----------

def reference_prefix(n, seed):
    """A FULL forward Fisher-Yates shuffle of indices 0..n-1 with Fraction arithmetic and explicit
    rejection (sharing no code with the module), then its first 20 positions."""
    rng = random.Random(seed)
    items = list(range(n))
    for i in range(n - 1):
        m = n - i
        limit = (2**53 // m) * m
        while True:
            w = Fraction(rng.random()) * 2**53
            assert w.denominator == 1
            if int(w) < limit:
                break
        j = i + int(w) % m
        items[i], items[j] = items[j], items[i]
    return items[:20]


def pool_of(n):
    return [chunk("doc-a", i * 50, i * 50 + 80) for i in range(n)]   # canonical order = index order


@pytest.mark.parametrize("n", [1, 2, 3, 7, 19, 20, 21, 22, 30, 61])
def test_random_ranking_matches_a_full_independent_shuffle_for_many_seeds(n):
    pool = pool_of(n)
    for seed in [0, 1, 2, 3, 12345, 2**31, 2**40] + list(range(20, 30)):
        expected = tuple(pool[i] for i in reference_prefix(n, seed))
        assert bl.random_ranking(pool, seed) == expected


@pytest.mark.parametrize(
    "n, seed, indices",
    [
        pytest.param(30, 0, [22, 23, 26, 14, 25, 13, 27, 3, 28, 16, 4, 1, 7, 8, 20, 6, 18, 0, 2, 5], id="n30-seed0"),
        pytest.param(30, 1, [7, 22, 25, 17, 5, 12, 8, 23, 4, 24, 28, 0, 27, 20, 19, 10, 29, 26, 15, 3], id="n30-seed1"),
        pytest.param(30, 12345, [4, 1, 13, 16, 0, 8, 15, 20, 24, 18, 26, 5, 14, 29, 17, 2, 23, 9, 27, 22], id="n30-seed12345"),
        pytest.param(30, 2**40, [24, 9, 28, 7, 15, 0, 29, 21, 20, 2, 8, 17, 19, 10, 23, 26, 22, 18, 13, 27], id="n30-seed-2-40"),
        pytest.param(7, 3, [4, 6, 1, 5, 3, 0, 2], id="n7-seed3"),
        pytest.param(20, 5, [4, 7, 5, 15, 6, 12, 13, 14, 2, 16, 8, 18, 9, 1, 10, 19, 0, 11, 17, 3], id="n20-seed5"),
        pytest.param(21, 5, [12, 7, 8, 15, 1, 9, 2, 20, 10, 13, 17, 18, 4, 0, 5, 14, 11, 16, 3, 19], id="n21-seed5"),
        pytest.param(1, 0, [0], id="n1"),
        pytest.param(2, 9, [1, 0], id="n2-seed9"),
    ],
)
def test_pinned_random_rankings(n, seed, indices):
    pool = pool_of(n)
    assert bl.random_ranking(pool, seed) == tuple(pool[i] for i in indices)


def test_the_plain_scaled_float_index_would_give_different_rankings():
    rng = random.Random(0)
    items = list(range(30))
    for i in range(20):
        j = i + int(rng.random() * (30 - i))
        items[i], items[j] = items[j], items[i]
    assert items[:20] != reference_prefix(30, 0)


def test_the_ranking_is_a_without_replacement_sample_of_the_pool():
    pool = pool_of(50)
    for seed in range(12):
        ranking = bl.random_ranking(pool, seed)
        assert len(ranking) == 20 and len(set(ranking)) == 20 and set(ranking) <= set(pool)


def test_a_smaller_pool_is_returned_whole_in_a_seeded_order():
    pool = pool_of(6)
    ranking = bl.random_ranking(pool, 4)
    assert sorted(ranking, key=lambda c: c.start) == pool and ranking != tuple(pool)


def test_the_pool_input_order_and_container_do_not_change_the_ranking():
    pool = pool_of(40)
    expected = bl.random_ranking(pool, 9)
    shuffled = list(pool)
    random.Random(1).shuffle(shuffled)
    assert bl.random_ranking(shuffled, 9) == expected and bl.random_ranking(tuple(shuffled), 9) == expected


def test_the_same_seed_gives_the_same_ranking_and_different_seeds_usually_differ():
    pool = pool_of(40)
    assert bl.random_ranking(pool, 3) == bl.random_ranking(pool, 3)
    assert len({bl.random_ranking(pool, seed) for seed in range(10)}) == 10


def test_random_ranking_uses_its_own_random_state():
    random.seed(77)
    expected = random.random()
    random.seed(77)
    bl.random_ranking(pool_of(40), 5)
    assert random.random() == expected


def test_chunk_ids_play_no_part_the_order_depends_on_the_canonical_key_only():
    a = [chunk("a", 0, 10, SHA), chunk("b", 5, 15, SHA_B), chunk("a", 20, 30, SHA)]
    b = list(reversed(a))
    assert bl.random_ranking(a, 1) == bl.random_ranking(b, 1)


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None, IntSubclass(1), [1], b"1"])
def test_a_seed_of_the_wrong_type_is_a_type_error(seed):
    with pytest.raises(TypeError):
        bl.random_ranking(pool_of(5), seed)


@pytest.mark.parametrize("seed", [-1, -(10**30)])
def test_a_negative_seed_is_a_value_error(seed):
    with pytest.raises(ValueError):
        bl.random_ranking(pool_of(5), seed)


@pytest.mark.parametrize("seed", [0, 1, 2**64, 10**30])
def test_zero_and_large_seeds_are_accepted(seed):
    assert len(bl.random_ranking(pool_of(5), seed)) == 5


# ---------- J. random_baseline ----------

def test_the_result_has_one_score_row_per_seed_in_seed_order(baseline):
    assert baseline.seeds == SEEDS and len(baseline.per_seed) == 1000
    assert all(type(row) is bl.RankingScores and len(row.values) == 76 for row in baseline.per_seed)
    assert len(baseline.summaries) == 76
    assert [m.metric for m in baseline.summaries] == list(baseline.per_seed[0].metric_ids)


@pytest.mark.parametrize("index", [0, 1, 17, 499, 999])
def test_a_seeds_scores_equal_scoring_its_own_ranking(baseline, index):
    ranking = bl.random_ranking(RANDOM_POOL, SEEDS[index])
    assert baseline.per_seed[index] == bl.score_ranking(RANDOM_QUESTION, ranking)


def test_all_three_rules_are_scored_on_the_same_ranking_per_seed(baseline):
    row = baseline.per_seed[3]
    ranking = bl.random_ranking(RANDOM_POOL, 3)
    for rule in RULES:
        assert row.get("raw", "hit", 20, rule) == float(mt.hit_at_k(RANDOM_QUESTION, ranking, 20, rule))
        assert row.get("distinct", "mrr", None, rule) == mt.distinct_reciprocal_rank(RANDOM_QUESTION, ranking, rule)


def test_summaries_equal_an_independent_recomputation_from_the_per_seed_values(baseline):
    for index, summary in enumerate(baseline.summaries):
        column = sorted(row.values[index] for row in baseline.per_seed)
        ranks = [math.ceil(Fraction(1000 * permille, 1000)) for permille in (25, 950, 975)]
        assert ranks == [25, 950, 975]
        assert (summary.p2_5, summary.p95, summary.p97_5) == tuple(column[r - 1] for r in ranks)
        assert summary.mean == pytest.approx(float(sum(Fraction(v) for v in column) / 1000), abs=1e-12)
        assert summary.n_seeds == 1000
        assert summary.p2_5 <= summary.p95 <= summary.p97_5


@pytest.mark.parametrize(
    "count, ranks",
    [(1000, (25, 950, 975)), (1001, (26, 951, 976)), (1039, (26, 988, 1014)), (1500, (38, 1425, 1463)),
     (2000, (50, 1900, 1950)), (1, (1, 1, 1)), (40, (1, 38, 39)), (41, (2, 39, 40))],
)
def test_nearest_ranks_are_hand_derived_for_several_seed_counts(count, ranks):
    assert tuple(bl._nearest_rank(count, p) for p in (25, 950, 975)) == ranks
    assert tuple(math.ceil(Fraction(count * p, 1000)) for p in (25, 950, 975)) == ranks


@pytest.mark.parametrize("count", [1000, 1001, 1039, 1500, 2000, 40, 7, 1])
def test_a_summary_picks_the_nearest_rank_value_of_the_sorted_column(count):
    rng = random.Random(count)
    column = [rng.random() for _ in range(count)]
    summary = bl._summary(bl.MetricId("raw", "hit", 1, "any_overlap"), column)
    ordered = sorted(column)
    low, mid, high = (math.ceil(Fraction(count * p, 1000)) for p in (25, 950, 975))
    assert (summary.p2_5, summary.p95, summary.p97_5) == (ordered[low - 1], ordered[mid - 1], ordered[high - 1])
    assert summary.mean == math.fsum(column) / count and summary.n_seeds == count


@pytest.mark.parametrize("count", [1001, 1039, 1500])
def test_the_summaries_follow_the_nearest_rank_for_other_seed_counts_end_to_end(count):
    seeds = list(range(7, 7 + count))
    result = bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, seeds)
    ranks = [math.ceil(Fraction(count * p, 1000)) for p in (25, 950, 975)]
    for index in (0, 20, 41, 75):
        column = sorted(row.values[index] for row in result.per_seed)
        summary = result.summaries[index]
        assert (summary.p2_5, summary.p95, summary.p97_5) == tuple(column[r - 1] for r in ranks)
        assert summary.n_seeds == count


def test_a_summary_of_a_binary_metric_is_its_rate_over_the_seeds(baseline):
    index = list(baseline.per_seed[0].metric_ids).index(bl.MetricId("raw", "hit", 20, "any_overlap"))
    hits = sum(row.values[index] for row in baseline.per_seed)
    assert baseline.summaries[index].mean == pytest.approx(hits / 1000)
    assert 0.0 <= baseline.summaries[index].mean <= 1.0


def test_the_reproducibility_record(baseline):
    record = baseline.reproducibility
    version = sys.version_info
    assert record.python_version == f"{version.major}.{version.minor}.{version.micro}"
    assert record.algorithm == "seeded_partial_fisher_yates_rejection_v1"
    assert record.canonical_key == ("corpus_key", "start_offset", "end_offset", "text_sha256")
    assert (record.pool_size, record.rank_cap, record.percentile_convention) == (30, 20, "nearest_rank")
    assert "2**53" in record.random_source


def test_distinct_truncation_flags_are_kept_for_every_seed_and_cutoff(baseline):
    for index in (0, 5, 500):
        ranking = bl.random_ranking(RANDOM_POOL, index)
        assert baseline.per_seed[index].distinct_truncated == tuple(ps.is_truncated(ranking, k) for k in CUTOFFS)
    assert all(len(row.distinct_truncated) == 5 for row in baseline.per_seed)


def test_the_baseline_is_repeatable_and_independent_of_the_pool_order():
    seeds = tuple(range(5000, 6000))
    first = bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, seeds)
    shuffled = list(RANDOM_POOL)
    random.Random(2).shuffle(shuffled)
    assert bl.random_baseline(RANDOM_QUESTION, shuffled, list(seeds)) == first


def test_the_random_baseline_uses_its_own_random_state():
    random.seed(77)
    expected = random.random()
    random.seed(77)
    bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL[:25], SEEDS)
    assert random.random() == expected


def test_the_oracle_is_at_least_as_good_as_the_random_mean_on_this_fixture(baseline):
    oracle = bl.oracle_baseline(RANDOM_QUESTION, RANDOM_POOL)
    for index, metric_id in enumerate(baseline.per_seed[0].metric_ids):
        if metric_id.family == "raw" and metric_id.metric in ("hit", "unit_recall") and metric_id.rule == "half_coverage":
            assert oracle.scores.values[index] >= baseline.summaries[index].mean


# ---------- K. validation ----------

@pytest.mark.parametrize("seeds", [None, "abc", {1, 2}, 5, range(1000), (x for x in ())])
def test_seeds_must_be_a_list_or_tuple(seeds):
    with pytest.raises(TypeError):
        bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, seeds)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, IntSubclass(1), [1]])
def test_every_seed_must_be_an_exact_int(bad):
    with pytest.raises(TypeError):
        bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, list(range(999)) + [bad])


def test_seed_value_errors_and_their_order():
    cases = [
        (list(range(500)) + [-1], "a seed must not be negative"),
        ([-1] + [5] * 3, "a seed must not be negative"),
        ([1, 1] + list(range(2, 1000)), "seeds must be unique"),
        ([1, 1], "seeds must be unique"),
        (list(range(999)), "too few seeds"),
        ([], "too few seeds"),
    ]
    for seeds, message in cases:
        with pytest.raises(ValueError) as caught:
            bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL, seeds)
        assert str(caught.value) == message


def test_exactly_one_thousand_seeds_are_enough():
    assert len(bl.random_baseline(RANDOM_QUESTION, RANDOM_POOL[:22], range_list(1000)).seeds) == 1000


def range_list(count):
    return list(range(count))


@pytest.mark.parametrize("bad", [None, "q", {"q": 1}, 5, WORKED.units])
def test_the_question_must_be_a_question(bad):
    with pytest.raises(TypeError):
        bl.random_baseline(bad, RANDOM_POOL, SEEDS)
    with pytest.raises(TypeError):
        bl.oracle_baseline(bad, RANDOM_POOL)
    with pytest.raises(TypeError):
        bl.score_ranking(bad, [])


def test_an_unanswerable_question_or_one_without_units_is_a_value_error():
    for bad in (question(unit("u1", span("a", 0, 5)), answerable=False), question()):
        for call in (lambda: bl.score_ranking(bad, []), lambda: bl.oracle_baseline(bad, RANDOM_POOL),
                     lambda: bl.random_baseline(bad, RANDOM_POOL, SEEDS)):
            with pytest.raises(ValueError):
                call()


@pytest.mark.parametrize("ranking", [None, "abc", {}, 5, (x for x in ()), [chunk("a", 0, 1), "x"]])
def test_the_ranking_must_be_a_list_or_tuple_of_ranked_chunks(ranking):
    with pytest.raises(TypeError):
        bl.score_ranking(WORKED, ranking)


def test_every_type_check_comes_before_every_value_check():
    unanswerable = question(unit("u1", span("a", 0, 5)), answerable=False)
    with pytest.raises(TypeError):
        bl.random_baseline(unanswerable, [], ["x"])                       # seed type beats empty pool and question
    with pytest.raises(TypeError):
        bl.random_baseline(unanswerable, [chunk("a", 0, 1)] * 2, "abc")   # seeds container type beats duplicates
    with pytest.raises(TypeError):
        bl.random_baseline(RANDOM_QUESTION, [chunk("a", 0, 1), chunk("a", 0, 1), 7], [-1])
    with pytest.raises(TypeError):
        bl.oracle_baseline(unanswerable, [7])
    with pytest.raises(TypeError):
        bl.random_ranking([], True)                                       # seed type beats empty pool


def test_the_value_checks_run_in_the_documented_order():
    unanswerable = question(unit("u1", span("a", 0, 5)), answerable=False)
    twice = [chunk("a", 0, 1), chunk("a", 0, 1)]
    cases = [
        (unanswerable, [], [-1], "the question must be answerable"),
        (RANDOM_QUESTION, [], [-1], "the pool must not be empty"),
        (RANDOM_QUESTION, twice, [-1], "the pool has two chunks with the same canonical sort key"),
        (RANDOM_QUESTION, RANDOM_POOL, [-1], "a seed must not be negative"),
        (RANDOM_QUESTION, RANDOM_POOL, [1, 1], "seeds must be unique"),
        (RANDOM_QUESTION, RANDOM_POOL, [1, 2], "too few seeds"),
    ]
    for q, pool, seeds, message in cases:
        with pytest.raises(ValueError) as caught:
            bl.random_baseline(q, pool, seeds)
        assert str(caught.value) == message


def test_error_messages_are_fixed_and_contain_no_input():
    bad_chunk = chunk(CANARY, 0, 5)
    calls = [
        lambda: bl.canonical_order(CANARY),
        lambda: bl.canonical_order([CANARY]),
        lambda: bl.canonical_order([bad_chunk, bad_chunk]),
        lambda: bl.random_ranking([bad_chunk], CANARY),
        lambda: bl.random_ranking([bad_chunk], -123456),
        lambda: bl.random_baseline(RANDOM_QUESTION, [bad_chunk], [CANARY]),
        lambda: bl.random_baseline(RANDOM_QUESTION, [bad_chunk], [4242, 4242]),
        lambda: bl.random_baseline(RANDOM_QUESTION, [bad_chunk], [-4242]),
        lambda: bl.random_baseline(RANDOM_QUESTION, [bad_chunk], [4242]),
        lambda: bl.oracle_baseline(CANARY, [bad_chunk]),
        lambda: bl.oracle_baseline(RANDOM_QUESTION, CANARY),
        lambda: bl.score_ranking(WORKED, [CANARY]),
        lambda: bl.score_ranking(question(unit("u1", span(CANARY, 0, 5)), answerable=False), []),
        lambda: bl._bounded(lambda: 0, 10**30),
        lambda: bl._bounded(lambda: 0, CANARY),
        lambda: bl.score_ranking(WORKED, []).get(CANARY, "hit", 1, "any_overlap"),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        text = str(caught.value)
        assert all(token not in text for token in (CANARY, "123456", "4242", str(10**30)))


# ---------- L. purity and import isolation ----------

def test_inputs_are_not_modified():
    pool = list(RANDOM_POOL)
    snapshot = list(pool)
    ranking = [chunk("doc-a", 0, 60), chunk("doc-a", 250, 300)]
    ranking_snapshot = list(ranking)
    bl.canonical_order(pool)
    bl.random_ranking(pool, 3)
    bl.oracle_baseline(RANDOM_QUESTION, pool)
    bl.score_ranking(WORKED, ranking)
    assert pool == snapshot and ranking == ranking_snapshot
    assert all(a is b for a, b in zip(pool, snapshot))


def test_result_types_are_exact():
    result = bl.oracle_baseline(RANDOM_QUESTION, RANDOM_POOL)
    assert type(result) is bl.OracleResult and type(result.ranking) is tuple and type(result.chunks_needed) is int
    assert all(type(v) is float for v in result.scores.values)
    assert all(type(item) is bl.ReachableScores for item in result.reachable_only)
    assert [rule for rule, _ in result.unreachable_by_rule] == list(RULES)


def test_the_module_adds_no_heavy_dependency_and_no_unrelated_evaluation_module():
    code = (
        "import sys\n"
        "import app.rag.evaluation.baselines as m\n"
        "c = m.RankedChunk('a', 0, 10, 'x' * 64)\n"
        "assert len(m.random_ranking([c], 1)) == 1\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "print(sorted(x for x in sys.modules if x.split('.')[0] in heavy))\n"
        "print(sorted(x for x in sys.modules if x.startswith('app.rag.evaluation.')))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    heavy, evaluation = result.stdout.strip().splitlines()
    assert heavy == "[]"
    loaded = set(eval(evaluation))
    assert {"app.rag.evaluation.baselines", "app.rag.evaluation.metrics", "app.rag.evaluation.passages",
            "app.rag.evaluation.statistics"} <= loaded
    assert not loaded & {f"app.rag.evaluation.{name}" for name in
                         ("dataset", "dataset_metadata", "parsing", "validation", "anchoring", "near_duplicates",
                          "structure", "aggregation")}
