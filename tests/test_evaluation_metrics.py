"""E1.3.1: relevance, coverage, Hit@k, Unit Recall@k, Evidence Coverage@k and reciprocal rank.

A  constants and RankedChunk
B  overlap_length and half-open boundaries
C  relevance rules (exactly 50% versus 49%, any overlap, full coverage)
D  span coverage (union, duplicates, other documents)
E  the worked example: every metric at every cutoff
F  alternatives, independent units, a chunk spanning two spans
G  reciprocal rank and the rank-20 limit
H  empty rankings and rankings shorter than k
I  rule sensitivity
J  invalid input: TypeError for a wrong type, ValueError for a bad value
K  purity: no mutation, no reordering, determinism
L  error messages never contain input content
M  import isolation
N  distinct-passage metrics (distinct_hit_at_k, distinct_reciprocal_rank; the grouping itself is
   tested in test_evaluation_passages.py)
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import metrics as mt
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64
CANARY = "SECRET-CANARY-7731"


class StrSubclass(str):
    pass


def chunk(key, start, end, sha=SHA):
    return mt.RankedChunk(key, start, end, sha)


def span(key, start, end):
    return s.Span(key, SHA, start, end, "q" * (end - start), SHA)


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


def single(start, end, key="doc-a"):
    return question(unit("u1", span(key, start, end)))


# The worked example. Gold evidence, three required units:
#   u1: doc-a [0, 100)
#   u2: doc-a [200, 300)  or  doc-b [0, 50)
#   u3: doc-b [1000, 1100)
WORKED = question(
    unit("u1", span("doc-a", 0, 100)),
    unit("u2", span("doc-a", 200, 300), span("doc-b", 0, 50)),
    unit("u3", span("doc-b", 1000, 1100)),
)
# Ranking, rank 1 first:
#   1 doc-c [0,500)     other document                       -> nothing
#   2 doc-a [0,60)      60% of u1                            -> u1 relevant, coverage 0.6
#   3 doc-a [250,300)   50% of u2's first span               -> u2 relevant, coverage 0.5
#   4 doc-b [0,20)      40% of u2's second span              -> not relevant, coverage 0.4
#   5 doc-b [1000,1060) 60% of u3                            -> u3 relevant, coverage 0.6
WORKED_RANKING = [
    chunk("doc-c", 0, 500),
    chunk("doc-a", 0, 60),
    chunk("doc-a", 250, 300),
    chunk("doc-b", 0, 20),
    chunk("doc-b", 1000, 1060),
]


# ---------- A. constants and RankedChunk ----------

def test_the_cutoffs_and_limits_are_the_approved_ones():
    assert mt.CUTOFFS == (1, 3, 5, 10, 20)
    assert mt.MAX_RANK == 20
    assert mt.RELEVANCE_RULES == ("any_overlap", "half_coverage", "full_coverage")
    assert mt.DEFAULT_RULE == "half_coverage"


def test_a_ranked_chunk_has_exactly_the_four_fields_and_is_frozen_and_slotted():
    assert [f.name for f in dataclasses.fields(mt.RankedChunk)] == ["corpus_key", "start", "end", "text_sha256"]
    value = chunk("doc-a", 0, 5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.start = 1
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize(
    "key, start, end, sha",
    [
        pytest.param("doc-a", True, 5, SHA, id="bool-start"),
        pytest.param("doc-a", 0, True, SHA, id="bool-end"),
        pytest.param("doc-a", 0.0, 5, SHA, id="float-start"),
        pytest.param("doc-a", 0, 5.0, SHA, id="float-end"),
        pytest.param("doc-a", "0", 5, SHA, id="str-start"),
        pytest.param("doc-a", None, 5, SHA, id="none-start"),
        pytest.param(5, 0, 5, SHA, id="int-key"),
        pytest.param(None, 0, 5, SHA, id="none-key"),
        pytest.param(b"doc-a", 0, 5, SHA, id="bytes-key"),
        pytest.param(StrSubclass("doc-a"), 0, 5, SHA, id="str-subclass-key"),
        pytest.param("doc-a", 0, 5, None, id="none-hash"),
        pytest.param("doc-a", 0, 5, b"x", id="bytes-hash"),
        pytest.param("doc-a", 0, 5, StrSubclass(SHA), id="str-subclass-hash"),
    ],
)
def test_a_ranked_chunk_rejects_a_wrong_type_with_type_error(key, start, end, sha):
    with pytest.raises(TypeError):
        mt.RankedChunk(key, start, end, sha)


@pytest.mark.parametrize(
    "start, end",
    [
        pytest.param(-1, 5, id="negative-start"),
        pytest.param(5, 5, id="empty"),
        pytest.param(6, 5, id="reversed"),
        pytest.param(0, 0, id="zero-length-at-zero"),
        pytest.param(-5, -1, id="both-negative"),
    ],
)
def test_a_ranked_chunk_rejects_a_bad_interval_with_value_error(start, end):
    with pytest.raises(ValueError):
        mt.RankedChunk("doc-a", start, end, SHA)


def test_ranked_chunk_strings_are_only_type_checked_as_in_the_e1_2_contract():
    # The corpus key and hash patterns of E1.2 apply to labelled spans (validate_span), not to
    # chunks, so any exact str is accepted, including an empty one; it just never matches a gold span.
    odd = mt.RankedChunk("", 0, 5, "")
    assert (odd.corpus_key, odd.text_sha256) == ("", "")
    assert mt.RankedChunk("Not A Valid Key!", 0, 5, "xyz").corpus_key == "Not A Valid Key!"
    assert mt.is_relevant(odd, span("doc-a", 0, 5)) is False
    assert mt.hit_at_k(single(0, 5), [odd], 1) == 0


# ---------- B. overlap_length and half-open boundaries ----------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param((0, 10), (0, 10), 10, id="identical"),
        pytest.param((0, 10), (5, 15), 5, id="partial"),
        pytest.param((0, 10), (10, 20), 0, id="touching-after"),
        pytest.param((10, 20), (0, 10), 0, id="touching-before"),
        pytest.param((0, 10), (9, 20), 1, id="one-position"),
        pytest.param((0, 10), (11, 20), 0, id="gap"),
        pytest.param((0, 100), (20, 30), 10, id="contained"),
        pytest.param((20, 30), (0, 100), 10, id="containing"),
        pytest.param([0, 10], [5, 6], 1, id="lists"),
    ],
)
def test_overlap_length_is_half_open(a, b, expected):
    assert mt.overlap_length(a, b) == expected


@pytest.mark.parametrize("bad", ["ab", None, 5, {0: 1}, {0, 1}, (c for c in (0, 1))])
def test_overlap_length_rejects_a_non_sequence_interval_with_type_error(bad):
    with pytest.raises(TypeError):
        mt.overlap_length(bad, (0, 5))
    with pytest.raises(TypeError):
        mt.overlap_length((0, 5), bad)


@pytest.mark.parametrize("bad", [(True, 3), (0, True), (0.5, 3), (0, 3.0), ("0", 3), (None, 3)])
def test_overlap_length_rejects_non_int_offsets_with_type_error(bad):
    with pytest.raises(TypeError):
        mt.overlap_length(bad, (0, 5))
    with pytest.raises(TypeError):
        mt.overlap_length((0, 5), bad)


@pytest.mark.parametrize("bad", [(1,), (), (0, 3, 4), (0, 0), (5, 2), (-1, 3)])
def test_overlap_length_rejects_a_bad_interval_with_value_error(bad):
    with pytest.raises(ValueError):
        mt.overlap_length(bad, (0, 5))
    with pytest.raises(ValueError):
        mt.overlap_length((0, 5), bad)


# ---------- C. relevance rules ----------

def test_exactly_half_is_relevant_and_just_under_half_is_not():
    gold = span("doc-a", 0, 100)
    assert mt.is_relevant(chunk("doc-a", 0, 50), gold) is True
    assert mt.is_relevant(chunk("doc-a", 0, 49), gold) is False
    assert mt.is_relevant(chunk("doc-a", 50, 100), gold) is True
    assert mt.is_relevant(chunk("doc-a", 51, 100), gold) is False


def test_half_of_an_odd_length_span_needs_the_rounded_up_amount():
    gold = span("doc-a", 0, 5)
    assert mt.is_relevant(chunk("doc-a", 0, 2), gold) is False  # 40%
    assert mt.is_relevant(chunk("doc-a", 0, 3), gold) is True  # 60%


def test_a_chunk_larger_than_the_span_covers_it():
    assert mt.is_relevant(chunk("doc-a", 0, 1000), span("doc-a", 10, 20)) is True


def test_touching_is_not_overlap_under_any_rule():
    gold = span("doc-a", 10, 20)
    for rule in mt.RELEVANCE_RULES:
        assert mt.is_relevant(chunk("doc-a", 0, 10), gold, rule) is False
        assert mt.is_relevant(chunk("doc-a", 20, 30), gold, rule) is False


def test_the_last_position_of_a_span_counts_and_the_end_is_exclusive():
    gold = span("doc-a", 10, 20)
    assert mt.is_relevant(chunk("doc-a", 19, 30), gold, mt.ANY_OVERLAP) is True
    assert mt.is_relevant(chunk("doc-a", 0, 11), gold, mt.ANY_OVERLAP) is True
    assert mt.is_relevant(chunk("doc-a", 20, 30), gold, mt.ANY_OVERLAP) is False


def test_any_overlap_and_full_coverage_rules():
    gold = span("doc-a", 0, 100)
    assert mt.is_relevant(chunk("doc-a", 99, 120), gold, mt.ANY_OVERLAP) is True
    assert mt.is_relevant(chunk("doc-a", 99, 120), gold, mt.HALF_COVERAGE) is False
    assert mt.is_relevant(chunk("doc-a", 0, 99), gold, mt.FULL_COVERAGE) is False
    assert mt.is_relevant(chunk("doc-a", 0, 100), gold, mt.FULL_COVERAGE) is True
    assert mt.is_relevant(chunk("doc-a", 0, 150), gold, mt.FULL_COVERAGE) is True


def test_a_different_document_is_never_relevant():
    for rule in mt.RELEVANCE_RULES:
        assert mt.is_relevant(chunk("doc-b", 0, 100), span("doc-a", 0, 100), rule) is False


def test_the_threshold_is_not_met_by_adding_up_several_chunks():
    q = single(0, 100)
    ranking = [chunk("doc-a", 0, 30), chunk("doc-a", 70, 100)]  # 30% + 30%
    assert mt.hit_at_k(q, ranking, 5) == 0
    assert mt.reciprocal_rank(q, ranking) == 0.0
    assert mt.evidence_coverage_at_k(q, ranking, 5) == 0.6  # coverage does add them (disjoint)


def test_text_sha256_plays_no_part_in_relevance_or_coverage():
    q = single(0, 100)
    one = [chunk("doc-a", 0, 60, sha="b" * 64)]
    two = [chunk("doc-a", 0, 60, sha="c" * 64)]
    assert mt.hit_at_k(q, one, 5) == mt.hit_at_k(q, two, 5) == 1
    assert mt.evidence_coverage_at_k(q, one, 5) == mt.evidence_coverage_at_k(q, two, 5) == 0.6


# ---------- D. span coverage ----------

@pytest.mark.parametrize(
    "chunks, expected",
    [
        pytest.param([], 0, id="none"),
        pytest.param([(0, 40)], 0.4, id="one"),
        pytest.param([(0, 40), (30, 60)], 0.6, id="overlap-counted-once"),
        pytest.param([(0, 40), (0, 40)], 0.4, id="duplicate-counted-once"),
        pytest.param([(0, 40), (40, 70)], 0.7, id="touching-adds"),
        pytest.param([(0, 40), (60, 70)], 0.5, id="gap-between"),
        pytest.param([(10, 30), (0, 100)], 1.0, id="containing-chunk"),
        pytest.param([(0, 20), (10, 40), (30, 50)], 0.5, id="chain"),
        pytest.param([(200, 300)], 0, id="outside"),
        pytest.param([(90, 500)], 0.1, id="clipped-to-the-span"),
    ],
)
def test_span_coverage_is_the_union_over_the_span_length(chunks, expected):
    ranking = [chunk("doc-a", a, b) for a, b in chunks]
    assert float(mt.span_coverage(ranking, span("doc-a", 0, 100))) == expected


def test_span_coverage_ignores_other_documents_and_is_exact():
    result = mt.span_coverage([chunk("doc-b", 0, 100), chunk("doc-a", 0, 1)], span("doc-a", 0, 3))
    assert str(result) == "1/3"


def test_coverage_in_the_input_order_does_not_matter():
    spans = [(0, 40), (30, 60), (80, 90)]
    forward = [chunk("doc-a", a, b) for a, b in spans]
    assert mt.span_coverage(forward, span("doc-a", 0, 100)) == mt.span_coverage(forward[::-1], span("doc-a", 0, 100))


# ---------- E. the worked example ----------

@pytest.mark.parametrize(
    "k, hit, recall, coverage",
    [
        pytest.param(1, 0, 0.0, 0.0, id="k1"),
        pytest.param(3, 1, 2 / 3, 11 / 30, id="k3"),
        pytest.param(5, 1, 1.0, 17 / 30, id="k5"),
        pytest.param(10, 1, 1.0, 17 / 30, id="k10"),
        pytest.param(20, 1, 1.0, 17 / 30, id="k20"),
    ],
)
def test_the_worked_example_at_every_cutoff(k, hit, recall, coverage):
    assert mt.hit_at_k(WORKED, WORKED_RANKING, k) == hit
    assert mt.unit_recall_at_k(WORKED, WORKED_RANKING, k) == recall
    assert mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, k) == coverage


def test_the_worked_example_at_k_equal_2():
    assert mt.hit_at_k(WORKED, WORKED_RANKING, 2) == 1
    assert mt.unit_recall_at_k(WORKED, WORKED_RANKING, 2) == 1 / 3
    assert mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, 2) == pytest.approx(0.2)


def test_the_worked_example_reciprocal_rank_is_the_first_relevant_rank():
    assert mt.reciprocal_rank(WORKED, WORKED_RANKING) == 0.5


def test_the_worked_example_coverage_takes_the_best_alternative_for_unit_two():
    only_second_alternative = [chunk("doc-b", 0, 20)]  # 40% of u2's second span
    assert mt.evidence_coverage_at_k(WORKED, only_second_alternative, 5) == pytest.approx(0.4 / 3)
    both = [chunk("doc-b", 0, 20), chunk("doc-a", 250, 300)]  # 0.4 and 0.5 -> 0.5
    assert mt.evidence_coverage_at_k(WORKED, both, 5) == pytest.approx(0.5 / 3)


@pytest.mark.parametrize("k", mt.CUTOFFS)
def test_the_metrics_never_decrease_as_k_grows(k):
    smaller = [c for c in mt.CUTOFFS if c < k]
    for earlier in smaller:
        assert mt.hit_at_k(WORKED, WORKED_RANKING, earlier) <= mt.hit_at_k(WORKED, WORKED_RANKING, k)
        assert mt.unit_recall_at_k(WORKED, WORKED_RANKING, earlier) <= mt.unit_recall_at_k(WORKED, WORKED_RANKING, k)
        assert (
            mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, earlier)
            <= mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, k)
        )


# ---------- F. alternatives, independent units, a chunk spanning two spans ----------

def test_a_unit_is_satisfied_by_any_one_alternative():
    q = question(unit("u1", span("doc-a", 0, 10), span("doc-a", 100, 110), span("doc-b", 0, 10)))
    for ranking in ([chunk("doc-a", 0, 10)], [chunk("doc-a", 100, 110)], [chunk("doc-b", 0, 10)]):
        assert mt.unit_recall_at_k(q, ranking, 5) == 1.0
    assert mt.unit_recall_at_k(q, [chunk("doc-a", 200, 210)], 5) == 0.0


def test_satisfying_two_alternatives_of_one_unit_still_counts_one_unit():
    q = question(unit("u1", span("doc-a", 0, 10), span("doc-a", 100, 110)), unit("u2", span("doc-a", 500, 510)))
    ranking = [chunk("doc-a", 0, 10), chunk("doc-a", 100, 110)]
    assert mt.unit_recall_at_k(q, ranking, 5) == 0.5


def test_units_are_independently_required():
    q = question(unit("u1", span("doc-a", 0, 10)), unit("u2", span("doc-a", 100, 110)))
    assert mt.hit_at_k(q, [chunk("doc-a", 0, 10)], 5) == 1  # one unit is enough for a hit
    assert mt.unit_recall_at_k(q, [chunk("doc-a", 0, 10)], 5) == 0.5
    assert mt.unit_recall_at_k(q, [chunk("doc-a", 0, 10), chunk("doc-a", 100, 110)], 5) == 1.0


def test_a_chunk_spanning_two_gold_spans_can_satisfy_two_units():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-a", 100, 200)))
    ranking = [chunk("doc-a", 50, 150)]  # 50% of each
    assert mt.unit_recall_at_k(q, ranking, 1) == 1.0
    assert mt.evidence_coverage_at_k(q, ranking, 1) == 0.5
    assert mt.reciprocal_rank(q, ranking) == 1.0


def test_a_chunk_spanning_two_spans_with_just_under_half_of_one():
    q = question(unit("u1", span("doc-a", 0, 100)), unit("u2", span("doc-a", 100, 200)))
    ranking = [chunk("doc-a", 51, 149)]  # 49% of u1, 49% of u2
    assert mt.unit_recall_at_k(q, ranking, 1) == 0.0
    assert mt.hit_at_k(q, ranking, 1) == 0
    assert mt.evidence_coverage_at_k(q, ranking, 1) == 0.49


def test_overlapping_and_duplicate_chunks_do_not_inflate_coverage():
    q = single(0, 100)
    assert mt.evidence_coverage_at_k(q, [chunk("doc-a", 0, 50)] * 3, 5) == 0.5
    assert mt.evidence_coverage_at_k(q, [chunk("doc-a", 0, 50), chunk("doc-a", 25, 75)], 5) == 0.75


def test_a_single_unit_question_with_two_overlapping_spans_uses_the_best_one():
    q = question(unit("u1", span("doc-a", 0, 100), span("doc-a", 50, 150)))
    ranking = [chunk("doc-a", 50, 150)]
    assert mt.evidence_coverage_at_k(q, ranking, 1) == 1.0


# ---------- G. reciprocal rank and the rank-20 limit ----------

def padded(relevant_rank, total=25):
    items = [chunk("doc-z", 0, 10)] * total
    items[relevant_rank - 1] = chunk("doc-a", 0, 100)
    return items


@pytest.mark.parametrize("rank, expected", [(1, 1.0), (2, 0.5), (3, 1 / 3), (7, 1 / 7), (20, 1 / 20), (21, 0.0), (25, 0.0)])
def test_reciprocal_rank_by_position(rank, expected):
    assert mt.reciprocal_rank(single(0, 100), padded(rank)) == expected


def test_nothing_beyond_rank_20_is_looked_at_by_reciprocal_rank():
    ranking = padded(20) + [chunk("doc-a", 0, 100)]
    assert mt.reciprocal_rank(single(0, 100), ranking) == 1 / 20
    assert mt.reciprocal_rank(single(0, 100), padded(21)) == 0.0


def test_hit_at_20_ignores_a_relevant_chunk_at_rank_21():
    assert mt.hit_at_k(single(0, 100), padded(21), 20) == 0
    assert mt.hit_at_k(single(0, 100), padded(20), 20) == 1
    assert mt.hit_at_k(single(0, 100), padded(20), 10) == 0


def test_an_invalid_item_beyond_rank_20_is_never_inspected():
    ranking = padded(3, total=20) + ["not a chunk"]
    assert mt.reciprocal_rank(single(0, 100), ranking) == 1 / 3
    assert mt.hit_at_k(single(0, 100), ranking, 20) == 1


def test_reciprocal_rank_uses_the_first_relevant_chunk_across_all_units():
    q = question(unit("u1", span("doc-a", 0, 10)), unit("u2", span("doc-a", 100, 110)))
    ranking = [chunk("doc-a", 500, 510), chunk("doc-a", 100, 110), chunk("doc-a", 0, 10)]
    assert mt.reciprocal_rank(q, ranking) == 0.5


def test_reciprocal_rank_with_no_relevant_chunk_is_zero():
    assert mt.reciprocal_rank(single(0, 100), [chunk("doc-a", 0, 49), chunk("doc-b", 0, 100)]) == 0.0


# ---------- H. empty rankings and rankings shorter than k ----------

@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_ranking_scores_zero_everywhere(empty):
    for k in mt.CUTOFFS:
        assert mt.hit_at_k(WORKED, empty, k) == 0
        assert mt.unit_recall_at_k(WORKED, empty, k) == 0.0
        assert mt.evidence_coverage_at_k(WORKED, empty, k) == 0.0
    assert mt.reciprocal_rank(WORKED, empty) == 0.0


def test_a_ranking_shorter_than_k_uses_what_there_is():
    assert mt.unit_recall_at_k(WORKED, WORKED_RANKING, 20) == mt.unit_recall_at_k(WORKED, WORKED_RANKING, 5)


def test_a_tuple_ranking_is_accepted():
    assert mt.hit_at_k(WORKED, tuple(WORKED_RANKING), 5) == 1


# ---------- I. rule sensitivity ----------

def test_the_rule_changes_hit_and_recall_but_not_coverage():
    q = single(0, 100)
    ranking = [chunk("doc-a", 90, 200)]  # 10% of the span
    assert mt.hit_at_k(q, ranking, 5, mt.ANY_OVERLAP) == 1
    assert mt.hit_at_k(q, ranking, 5, mt.HALF_COVERAGE) == 0
    assert mt.hit_at_k(q, ranking, 5, mt.FULL_COVERAGE) == 0
    assert mt.unit_recall_at_k(q, ranking, 5, mt.ANY_OVERLAP) == 1.0
    assert mt.reciprocal_rank(q, ranking, mt.ANY_OVERLAP) == 1.0
    assert mt.reciprocal_rank(q, ranking, mt.HALF_COVERAGE) == 0.0
    assert mt.evidence_coverage_at_k(q, ranking, 5) == 0.1


def test_full_coverage_needs_one_chunk_to_cover_the_whole_span():
    q = single(0, 100)
    assert mt.hit_at_k(q, [chunk("doc-a", 0, 60), chunk("doc-a", 40, 100)], 5, mt.FULL_COVERAGE) == 0
    assert mt.hit_at_k(q, [chunk("doc-a", 0, 100)], 5, mt.FULL_COVERAGE) == 1


def test_the_ordering_of_the_rules_is_any_then_half_then_full():
    q = single(0, 100)
    for end in (1, 49, 50, 99, 100):
        ranking = [chunk("doc-a", 0, end)]
        hits = [mt.hit_at_k(q, ranking, 1, rule) for rule in mt.RELEVANCE_RULES]
        assert hits == sorted(hits, reverse=True)


# ---------- J. invalid input ----------

@pytest.mark.parametrize("k", [True, False, 1.0, 5.0, "5", None, [5]])
def test_k_of_the_wrong_type_is_a_type_error(k):
    with pytest.raises(TypeError):
        mt.hit_at_k(WORKED, WORKED_RANKING, k)
    with pytest.raises(TypeError):
        mt.unit_recall_at_k(WORKED, WORKED_RANKING, k)
    with pytest.raises(TypeError):
        mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, k)


@pytest.mark.parametrize("k", [0, -1, 21, 100])
def test_k_outside_1_to_20_is_a_value_error(k):
    with pytest.raises(ValueError):
        mt.hit_at_k(WORKED, WORKED_RANKING, k)
    with pytest.raises(ValueError):
        mt.unit_recall_at_k(WORKED, WORKED_RANKING, k)
    with pytest.raises(ValueError):
        mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, k)


@pytest.mark.parametrize("ranking", [None, "abc", "", {1: 2}, {chunk("doc-a", 0, 1)}, set(), 5, (c for c in [])])
def test_a_ranking_must_be_a_list_or_tuple(ranking):
    with pytest.raises(TypeError):
        mt.hit_at_k(WORKED, ranking, 5)
    with pytest.raises(TypeError):
        mt.reciprocal_rank(WORKED, ranking)


def test_a_ranking_item_must_be_a_ranked_chunk():
    for bad in (None, ("doc-a", 0, 5, SHA), {"corpus_key": "doc-a"}, WORKED.units[0].spans[0]):
        with pytest.raises(TypeError):
            mt.hit_at_k(WORKED, [bad], 5)


def test_a_question_must_be_a_question():
    for bad in (None, "q", {"units": ()}, WORKED.units):
        with pytest.raises(TypeError):
            mt.hit_at_k(bad, WORKED_RANKING, 5)


def test_an_unanswerable_question_is_a_value_error():
    q = question(answerable=False, units=(), expected_answer=None, source_language=None, difficulty=None,
                 close_distractor=None)
    with pytest.raises(ValueError):
        mt.hit_at_k(q, WORKED_RANKING, 5)
    with pytest.raises(ValueError):
        mt.reciprocal_rank(q, WORKED_RANKING)


@pytest.mark.parametrize("value", ["yes", 1, 0, None])
def test_a_non_bool_answerable_is_a_type_error(value):
    with pytest.raises(TypeError):
        mt.hit_at_k(dataclasses.replace(WORKED, answerable=value), WORKED_RANKING, 5)


def _every_metric(q):
    return (
        lambda: mt.hit_at_k(q, WORKED_RANKING, 5),
        lambda: mt.unit_recall_at_k(q, WORKED_RANKING, 5),
        lambda: mt.evidence_coverage_at_k(q, WORKED_RANKING, 5),
        lambda: mt.reciprocal_rank(q, WORKED_RANKING),
    )


@pytest.mark.parametrize(
    "units",
    [
        pytest.param([unit("u1", span("doc-a", 0, 5))], id="list-of-units"),
        pytest.param(None, id="none-units"),
        pytest.param("u1", id="str-units"),
        pytest.param(("u1",), id="not-a-unit"),
        pytest.param((s.EvidenceUnit("u1", [span("doc-a", 0, 5)]),), id="spans-not-a-tuple"),
        pytest.param((s.EvidenceUnit("u1", None),), id="spans-none"),
        pytest.param((s.EvidenceUnit("u1", ("span",)),), id="not-a-span"),
        pytest.param((unit("u1", s.Span("doc-a", SHA, True, 5, "x", SHA)),), id="bool-start"),
        pytest.param((unit("u1", s.Span("doc-a", SHA, 0, 5.0, "x", SHA)),), id="float-end"),
        pytest.param((unit("u1", s.Span(None, SHA, 0, 5, "x", SHA)),), id="none-key"),
        pytest.param((unit("u1", s.Span(StrSubclass("doc-a"), SHA, 0, 5, "x", SHA)),), id="str-subclass-key"),
    ],
)
def test_wrongly_typed_evidence_units_are_a_type_error_for_every_metric(units):
    q = dataclasses.replace(WORKED, units=units)
    for call in _every_metric(q):
        with pytest.raises(TypeError):
            call()


@pytest.mark.parametrize(
    "units",
    [
        pytest.param((), id="no-units"),
        pytest.param((unit("u1"),), id="unit-without-spans"),
        pytest.param((unit("u1", s.Span("doc-a", SHA, 5, 5, "x", SHA)),), id="empty-span"),
        pytest.param((unit("u1", s.Span("doc-a", SHA, 6, 5, "x", SHA)),), id="reversed-span"),
        pytest.param((unit("u1", s.Span("doc-a", SHA, -1, 5, "x", SHA)),), id="negative-start"),
    ],
)
def test_unusable_evidence_values_are_a_value_error_for_every_metric(units):
    q = dataclasses.replace(WORKED, units=units)
    for call in _every_metric(q):
        with pytest.raises(ValueError):
            call()


def test_a_bad_span_is_rejected_even_when_the_ranking_is_empty():
    bad_type = dataclasses.replace(WORKED, units=(unit("u1", s.Span("doc-a", SHA, True, 5, "x", SHA)),))
    bad_value = dataclasses.replace(WORKED, units=(unit("u1", s.Span("doc-a", SHA, 5, 5, "x", SHA)),))
    with pytest.raises(TypeError):
        mt.hit_at_k(bad_type, [], 5)
    with pytest.raises(ValueError):
        mt.hit_at_k(bad_value, [], 5)


@pytest.mark.parametrize("rule", [None, 0.5, 1, StrSubclass("half_coverage"), b"half_coverage"])
def test_a_rule_of_the_wrong_type_is_a_type_error(rule):
    with pytest.raises(TypeError):
        mt.hit_at_k(WORKED, WORKED_RANKING, 5, rule)
    with pytest.raises(TypeError):
        mt.reciprocal_rank(WORKED, WORKED_RANKING, rule)
    with pytest.raises(TypeError):
        mt.is_relevant(chunk("doc-a", 0, 5), span("doc-a", 0, 5), rule)


@pytest.mark.parametrize("rule", ["half", "ANY_OVERLAP", "", "any overlap"])
def test_an_unknown_rule_name_is_a_value_error(rule):
    with pytest.raises(ValueError):
        mt.hit_at_k(WORKED, WORKED_RANKING, 5, rule)
    with pytest.raises(ValueError):
        mt.reciprocal_rank(WORKED, WORKED_RANKING, rule)
    with pytest.raises(ValueError):
        mt.is_relevant(chunk("doc-a", 0, 5), span("doc-a", 0, 5), rule)


def test_is_relevant_checks_its_arguments():
    with pytest.raises(TypeError):
        mt.is_relevant("chunk", span("doc-a", 0, 5))
    with pytest.raises(TypeError):
        mt.is_relevant(chunk("doc-a", 0, 5), "span")
    with pytest.raises(TypeError):
        mt.is_relevant(chunk("doc-a", 0, 5), s.Span("doc-a", SHA, True, 5, "x", SHA))
    with pytest.raises(ValueError):
        mt.is_relevant(chunk("doc-a", 0, 5), s.Span("doc-a", SHA, 5, 5, "x", SHA))


def test_span_coverage_checks_the_span():
    with pytest.raises(TypeError):
        mt.span_coverage([], "span")
    with pytest.raises(TypeError):
        mt.span_coverage([], s.Span("doc-a", SHA, 0.0, 5, "x", SHA))
    with pytest.raises(ValueError):
        mt.span_coverage([], s.Span("doc-a", SHA, 0, 0, "", SHA))


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty-str"),
        pytest.param("abc", id="str"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param({}, id="empty-dict"),
        pytest.param({1: 2}, id="dict"),
        pytest.param(set(), id="empty-set"),
        pytest.param(frozenset(), id="empty-frozenset"),
        pytest.param({chunk("doc-a", 0, 5)}, id="set-of-chunks"),
        pytest.param(iter(()), id="empty-iterator"),
        pytest.param((c for c in (chunk("doc-a", 0, 5),)), id="generator"),
        pytest.param(range(0), id="empty-range"),
        pytest.param(0, id="int"),
        pytest.param(chunk("doc-a", 0, 5), id="single-chunk"),
    ],
)
def test_span_coverage_rejects_a_chunks_container_that_is_not_a_list_or_tuple(container):
    with pytest.raises(TypeError):
        mt.span_coverage(container, span("doc-a", 0, 5))


def test_span_coverage_rejects_a_non_chunk_item_and_accepts_empty_list_and_tuple():
    for bad in ("doc-a", None, (0, 5), span("doc-a", 0, 5)):
        with pytest.raises(TypeError):
            mt.span_coverage([chunk("doc-a", 0, 5), bad], span("doc-a", 0, 5))
    assert mt.span_coverage([], span("doc-a", 0, 5)) == 0
    assert mt.span_coverage((), span("doc-a", 0, 5)) == 0
    assert mt.span_coverage((chunk("doc-a", 0, 5),), span("doc-a", 0, 5)) == 1


# ---------- K. purity ----------

def test_inputs_are_not_modified_or_reordered():
    ranking = list(WORKED_RANKING)
    snapshot = list(ranking)
    question_before = dataclasses.asdict(WORKED)
    for k in mt.CUTOFFS:
        mt.hit_at_k(WORKED, ranking, k)
        mt.unit_recall_at_k(WORKED, ranking, k)
        mt.evidence_coverage_at_k(WORKED, ranking, k)
    mt.reciprocal_rank(WORKED, ranking)
    assert ranking == snapshot and all(a is b for a, b in zip(ranking, snapshot))
    assert dataclasses.asdict(WORKED) == question_before


def test_results_are_deterministic_and_have_exact_types():
    first = [mt.hit_at_k(WORKED, WORKED_RANKING, 3), mt.unit_recall_at_k(WORKED, WORKED_RANKING, 3),
             mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, 3), mt.reciprocal_rank(WORKED, WORKED_RANKING)]
    second = [mt.hit_at_k(WORKED, WORKED_RANKING, 3), mt.unit_recall_at_k(WORKED, WORKED_RANKING, 3),
              mt.evidence_coverage_at_k(WORKED, WORKED_RANKING, 3), mt.reciprocal_rank(WORKED, WORKED_RANKING)]
    assert first == second
    assert [type(x) for x in first] == [int, float, float, float]


def test_rank_order_matters_for_reciprocal_rank_only_through_the_first_relevant_chunk():
    relevant, other = chunk("doc-a", 0, 100), chunk("doc-z", 0, 5)
    q = single(0, 100)
    assert mt.reciprocal_rank(q, [other, relevant]) == 0.5
    assert mt.reciprocal_rank(q, [relevant, other]) == 1.0
    assert mt.hit_at_k(q, [other, relevant], 1) == 0 and mt.hit_at_k(q, [other, relevant], 2) == 1


# ---------- L. no input content in errors ----------

def test_error_messages_never_contain_input_content():
    calls = [
        lambda: mt.RankedChunk(CANARY, -1, 5, SHA),
        lambda: mt.RankedChunk(CANARY, 0, 5, 7),
        lambda: mt.hit_at_k(WORKED, [CANARY], 5),
        lambda: mt.hit_at_k(WORKED, CANARY, 5),
        lambda: mt.hit_at_k(WORKED, WORKED_RANKING, 5, CANARY),
        lambda: mt.hit_at_k(dataclasses.replace(WORKED, units=(unit(CANARY, s.Span(CANARY, SHA, 5, 5, "x", SHA)),)),
                            WORKED_RANKING, 5),
        lambda: mt.hit_at_k(CANARY, WORKED_RANKING, 5),
        lambda: mt.span_coverage(CANARY, span(CANARY, 0, 5)),
        lambda: mt.span_coverage([], s.Span(CANARY, SHA, True, 5, "x", SHA)),
        lambda: mt.overlap_length(CANARY, (0, 5)),
        lambda: mt.is_relevant(chunk(CANARY, 0, 5), CANARY),
        lambda: mt.hit_at_k(dataclasses.replace(WORKED, units=(CANARY,)), WORKED_RANKING, 5),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        assert CANARY not in str(caught.value)


# ---------- M. import isolation ----------

def test_importing_the_module_loads_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import app.rag.evaluation.metrics\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "later = ('app.rag.evaluation.statistics', 'app.rag.retrieval', 'app.models')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy or m in later))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# ---------- N. distinct-passage metrics ----------

def fillers(first_rank, last_rank):
    return [chunk("doc-z", 100 * rank, 100 * rank + 10) for rank in range(first_rank, last_rank + 1)]


def redundant_then_relevant():
    """Ranks 1-3 are three chunks of one passage of doc-b (irrelevant); rank 4 is the gold span."""
    return [
        chunk("doc-b", 0, 100), chunk("doc-b", 50, 150), chunk("doc-b", 100, 200),
        chunk("doc-a", 1000, 1100),
    ]


def test_distinct_hit_can_exceed_raw_hit_when_deduplication_moves_a_passage_up():
    q = single(1000, 1100)
    ranking = redundant_then_relevant()
    assert [mt.hit_at_k(q, ranking, k) for k in (1, 2, 3, 4)] == [0, 0, 0, 1]
    assert [mt.distinct_hit_at_k(q, ranking, k) for k in (1, 2, 3, 4)] == [0, 1, 1, 1]
    assert mt.distinct_hit_at_k(q, ranking, 2) > mt.hit_at_k(q, ranking, 2)


def test_distinct_reciprocal_rank_uses_the_compressed_position():
    q = single(1000, 1100)
    ranking = redundant_then_relevant()
    assert mt.reciprocal_rank(q, ranking) == 1 / 4
    assert mt.distinct_reciprocal_rank(q, ranking) == 1 / 2


def test_distinct_metrics_equal_the_raw_ones_when_no_chunks_merge():
    for k in mt.CUTOFFS:
        assert mt.distinct_hit_at_k(WORKED, WORKED_RANKING, k) == mt.hit_at_k(WORKED, WORKED_RANKING, k)
    assert mt.distinct_reciprocal_rank(WORKED, WORKED_RANKING) == mt.reciprocal_rank(WORKED, WORKED_RANKING) == 0.5


def test_a_redundant_chunk_ahead_of_the_first_relevant_one_improves_the_distinct_rank():
    ranking = [chunk("doc-c", 0, 500), chunk("doc-c", 100, 200), chunk("doc-a", 0, 60)]
    assert mt.reciprocal_rank(WORKED, ranking) == 1 / 3
    assert mt.distinct_reciprocal_rank(WORKED, ranking) == 1 / 2
    assert mt.hit_at_k(WORKED, ranking, 2) == 0
    assert mt.distinct_hit_at_k(WORKED, ranking, 2) == 1


def test_a_relevant_later_member_gets_no_credit_when_its_representative_is_not_relevant():
    q = single(20, 30)
    ranking = [chunk("doc-a", 0, 10), chunk("doc-a", 20, 30), chunk("doc-a", 10, 21)]  # one passage, rep rank 1
    assert mt.hit_at_k(q, ranking, 2) == 1 and mt.reciprocal_rank(q, ranking) == 1 / 2
    assert mt.distinct_hit_at_k(q, ranking, 3) == 0
    assert mt.distinct_reciprocal_rank(q, ranking) == 0.0


def test_the_intervals_of_a_passage_are_not_combined_to_reach_the_threshold():
    q = single(0, 100)
    ranking = [chunk("doc-a", 0, 40), chunk("doc-a", 35, 75)]  # one passage covering 75%, each part 40%
    assert mt.evidence_coverage_at_k(q, ranking, 2) == 0.75
    assert mt.distinct_hit_at_k(q, ranking, 2) == 0
    assert mt.distinct_reciprocal_rank(q, ranking) == 0.0


def test_a_relevant_representative_counts_whatever_its_later_members_are():
    q = single(0, 100)
    ranking = [chunk("doc-a", 0, 60), chunk("doc-a", 55, 70)]
    assert mt.distinct_hit_at_k(q, ranking, 1) == 1
    assert mt.distinct_reciprocal_rank(q, ranking) == 1.0


def bridge_ranking(total):
    """Gold [20, 30). Rank 1 (0,10) is irrelevant, rank 2 (20,30) is relevant, and a chunk
    (10,21) touches the first and overlaps the second by one position, so it bridges them."""
    return [chunk("doc-a", 0, 10), chunk("doc-a", 20, 30)] + fillers(3, total - 1) + [chunk("doc-a", 10, 21)]


def test_a_bridge_at_rank_20_changes_distinct_metrics_at_small_k():
    q = single(20, 30)
    ranking = bridge_ranking(20)
    assert len(ranking) == 20
    assert mt.hit_at_k(q, ranking, 2) == 1 and mt.reciprocal_rank(q, ranking) == 1 / 2  # raw: unaffected
    assert [mt.distinct_hit_at_k(q, ranking, k) for k in (1, 2, 5, 20)] == [0, 0, 0, 0]
    assert mt.distinct_reciprocal_rank(q, ranking) == 0.0


def test_the_same_chunk_at_rank_21_changes_nothing():
    q = single(20, 30)
    ranking = bridge_ranking(21)
    assert len(ranking) == 21
    assert [mt.distinct_hit_at_k(q, ranking, k) for k in (1, 2, 5, 20)] == [0, 1, 1, 1]
    assert mt.distinct_reciprocal_rank(q, ranking) == 1 / 2


def test_distinct_metrics_depend_only_on_the_first_20_chunks():
    q = single(1000, 1100)
    ranking = fillers(1, 20) + [chunk("doc-a", 1000, 1100)]
    assert mt.distinct_hit_at_k(q, ranking, 20) == 0 and mt.distinct_reciprocal_rank(q, ranking) == 0.0


def test_k_counts_compressed_positions_and_may_exceed_the_number_of_representatives():
    q = single(1000, 1100)
    ranking = redundant_then_relevant()
    assert mt.distinct_hit_at_k(q, ranking, 20) == 1
    assert mt.distinct_hit_at_k(q, ranking[:3], 20) == 0


@pytest.mark.parametrize(
    "end, any_overlap, half, full",
    [(1, 1, 0, 0), (49, 1, 0, 0), (50, 1, 1, 0), (99, 1, 1, 0), (100, 1, 1, 1)],
)
def test_all_three_relevance_rules_apply_to_the_representative(end, any_overlap, half, full):
    q = single(0, 100)
    ranking = [chunk("doc-b", 0, 5), chunk("doc-b", 3, 8), chunk("doc-a", 0, end)]
    got = [mt.distinct_hit_at_k(q, ranking, 2, rule) for rule in (mt.ANY_OVERLAP, mt.HALF_COVERAGE, mt.FULL_COVERAGE)]
    assert got == [any_overlap, half, full]
    rr = [mt.distinct_reciprocal_rank(q, ranking, rule) for rule in mt.RELEVANCE_RULES]
    assert rr == [0.5 * flag for flag in (any_overlap, half, full)]


def test_distinct_metrics_on_empty_and_short_rankings():
    for empty in ([], ()):
        assert mt.distinct_hit_at_k(WORKED, empty, 5) == 0
        assert mt.distinct_reciprocal_rank(WORKED, empty) == 0.0
    assert mt.distinct_hit_at_k(single(0, 100), [chunk("doc-a", 0, 100)], 1) == 1
    assert mt.distinct_reciprocal_rank(single(0, 100), (chunk("doc-a", 0, 100),)) == 1.0


def test_a_unit_can_be_satisfied_by_any_representative_and_units_stay_independent():
    q = question(unit("u1", span("doc-a", 0, 10)), unit("u2", span("doc-a", 500, 510)))
    ranking = [chunk("doc-a", 0, 10), chunk("doc-a", 5, 12), chunk("doc-a", 500, 510)]
    assert mt.distinct_hit_at_k(q, ranking, 1) == 1
    assert mt.distinct_reciprocal_rank(q, ranking) == 1.0


def test_distinct_results_have_exact_types():
    q = single(1000, 1100)
    assert type(mt.distinct_hit_at_k(q, redundant_then_relevant(), 2)) is int
    assert type(mt.distinct_reciprocal_rank(q, redundant_then_relevant())) is float


@pytest.mark.parametrize("k", [True, False, 1.0, "5", None])
def test_distinct_k_of_the_wrong_type_is_a_type_error(k):
    with pytest.raises(TypeError):
        mt.distinct_hit_at_k(WORKED, WORKED_RANKING, k)


@pytest.mark.parametrize("k", [0, -1, 21])
def test_distinct_k_outside_1_to_20_is_a_value_error(k):
    with pytest.raises(ValueError):
        mt.distinct_hit_at_k(WORKED, WORKED_RANKING, k)


@pytest.mark.parametrize("ranking", [None, "abc", "", {}, set(), 5, (c for c in [])])
def test_a_distinct_ranking_must_be_a_list_or_tuple(ranking):
    with pytest.raises(TypeError):
        mt.distinct_hit_at_k(WORKED, ranking, 5)
    with pytest.raises(TypeError):
        mt.distinct_reciprocal_rank(WORKED, ranking)


def test_a_distinct_ranking_item_among_the_first_20_must_be_a_ranked_chunk_and_later_ones_are_ignored():
    with pytest.raises(TypeError):
        mt.distinct_hit_at_k(WORKED, [chunk("doc-a", 0, 5), None], 5)
    with pytest.raises(TypeError):
        mt.distinct_reciprocal_rank(WORKED, fillers(1, 19) + ["x"])
    assert mt.distinct_reciprocal_rank(WORKED, fillers(1, 20) + ["x"]) == 0.0


def test_distinct_metrics_validate_the_question_and_the_rule():
    unanswerable = question(answerable=False, units=(), expected_answer=None, source_language=None,
                            difficulty=None, close_distractor=None)
    with pytest.raises(ValueError):
        mt.distinct_hit_at_k(unanswerable, WORKED_RANKING, 5)
    with pytest.raises(TypeError):
        mt.distinct_reciprocal_rank("q", WORKED_RANKING)
    with pytest.raises(ValueError):
        mt.distinct_hit_at_k(WORKED, WORKED_RANKING, 5, "half")
    with pytest.raises(TypeError):
        mt.distinct_reciprocal_rank(WORKED, WORKED_RANKING, None)
    bad = dataclasses.replace(WORKED, units=(unit("u1", s.Span("doc-a", SHA, True, 5, "x", SHA)),))
    with pytest.raises(TypeError):
        mt.distinct_hit_at_k(bad, [], 5)


def test_distinct_metrics_do_not_modify_or_reorder_their_inputs_and_are_deterministic():
    q = single(1000, 1100)
    ranking = redundant_then_relevant()
    snapshot = list(ranking)
    first = (mt.distinct_hit_at_k(q, ranking, 2), mt.distinct_reciprocal_rank(q, ranking))
    assert (mt.distinct_hit_at_k(q, ranking, 2), mt.distinct_reciprocal_rank(q, ranking)) == first
    assert ranking == snapshot and all(a is b for a, b in zip(ranking, snapshot))


def test_distinct_error_messages_never_contain_input_content():
    calls = [
        lambda: mt.distinct_hit_at_k(WORKED, [CANARY], 5),
        lambda: mt.distinct_hit_at_k(WORKED, CANARY, 5),
        lambda: mt.distinct_reciprocal_rank(WORKED, WORKED_RANKING, CANARY),
        lambda: mt.distinct_hit_at_k(CANARY, WORKED_RANKING, 5),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        assert CANARY not in str(caught.value)


def test_the_raw_metrics_are_unchanged_by_the_distinct_ones():
    ranking = redundant_then_relevant()
    q = single(1000, 1100)
    before = [mt.hit_at_k(q, ranking, k) for k in mt.CUTOFFS] + [mt.reciprocal_rank(q, ranking)]
    mt.distinct_hit_at_k(q, ranking, 2)
    mt.distinct_reciprocal_rank(q, ranking)
    assert [mt.hit_at_k(q, ranking, k) for k in mt.CUTOFFS] + [mt.reciprocal_rank(q, ranking)] == before
