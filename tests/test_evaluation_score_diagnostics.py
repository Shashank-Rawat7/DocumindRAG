"""E2.3: descriptive dense-score diagnostics (app/rag/evaluation/score_diagnostics.py).

The module is pure: these tests need no database, no model and no dataset file. They use the small
hand-computed fixtures of the engine tests, with binary fractions as scores so every expected number is exact.

A  group statistics (hand-computed) and their edge cases
B  numeric fail-closed rules (empty, missing, non-finite, wrong length, rising, overflow)
C  the digest payload
D  the recorder and the unanswerable collection
E  the section built for a real run, and its validator
F  determinism, fixed messages, purity and privacy

Run focused: pytest tests/test_evaluation_score_diagnostics.py --noconftest -p no:cacheprovider
"""
import ast
import builtins
import copy
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import types
from fractions import Fraction
from pathlib import Path

import pytest
import test_evaluation_engine as fx

from app.rag.evaluation import engine
from app.rag.evaluation import schema as s
from app.rag.evaluation import score_diagnostics as sd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Binary fractions only: every difference, sum and mean below is exact or the correctly rounded quotient.
VECTORS = {
    fx.Q1.question: [1.0, 0.75, 0.75, 0.5, 0.25, 0.0],
    fx.Q2.question: [0.875, 0.75, 0.5, 0.5, 0.25, 0.125],
    fx.Q3.question: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    fx.Q4.question: [0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    fx.U1.question: [0.5, 0.5, 0.25, 0.25, 0.125, 0.0],
    fx.U2.question: [0.375, 0.25, 0.25, 0.125, 0.0, -0.25],
}
ORDER = {**fx.DENSE, fx.U1.question: [1, 2, 3, 4, 5, 6], fx.U2.question: [1, 2, 3, 4, 5, 6]}
ANSWERABLE = [fx.Q1, fx.Q2, fx.Q3, fx.Q4]
UNANSWERABLE = [fx.U1, fx.U2]
SEED_RESAMPLES = {
    "seeds": fx.SEEDS, "bootstrap_seed": 7, "resamples": fx.RESAMPLES, "adjudicated_unreachable": {"q4": ["u1"]},
}


def scored(vectors=None, order=None):
    """A retriever returning the fixture chunks in ``order`` with the scores of ``vectors`` (by position)."""
    vectors = VECTORS if vectors is None else vectors
    order = ORDER if order is None else order

    def retrieve(query):
        return [dataclasses.replace(fx.C[i], score=v) for i, v in zip(order[query], vectors[query], strict=True)]

    return retrieve


def calls_of(retriever, questions=None):
    """What the engine does with a recorder: one call per answerable question, in dev-file order."""
    recorder = sd.ScoreRecorder(retriever)
    for question in ANSWERABLE if questions is None else questions:
        recorder(question.question)
    return recorder.take()


def refused(code, fn, *args, **kwargs):
    with pytest.raises(s.EvaluationDataError) as info:
        fn(*args, **kwargs)
    assert {i.code for i in info.value.issues} == {code}
    return info.value


@pytest.fixture(scope="module")
def pool():
    return fx.make_pool()


@pytest.fixture(scope="module")
def inputs():
    return fx.inputs_of(fx.Q1, fx.Q2, fx.Q3, fx.Q4, fx.U1, fx.U2)


@pytest.fixture(scope="module")
def flow(inputs, pool):
    """One engine run through a recorder (as E2.4 will do) plus the separate unanswerable collection."""
    recorder = sd.ScoreRecorder(scored())
    run = engine.run_pilot(inputs, pool, recorder, **SEED_RESAMPLES)
    return types.SimpleNamespace(
        run=run, calls=recorder.take(), unanswerable=sd.collect_unanswerable_scores(inputs, pool, scored()),
    )


@pytest.fixture(scope="module")
def section(flow, inputs, pool):
    return sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, flow.unanswerable)


def ids(section_, group):
    return [row["question_id"] for row in section_["questions"][group]]


# ---------- A: group statistics ----------

def nearest_rank(values, p):
    ordered = sorted(values)
    return ordered[-(-p * len(ordered) // 100) - 1]  # ceil(p * n / 100) - 1 with exact integer arithmetic


def test_the_group_statistics_are_hand_computed():
    group = sd.summarize_group([[1.0, 0.75, 0.75], [0.5, 0.25, 0.0], [0.5, 0.5, 0.5]], 3)
    assert group["question_count"] == 3
    assert group["top1"] == {"valid_count": 3, "min": 0.5, "max": 1.0, "mean": 2.0 / 3, "p25": 0.5, "p50": 0.5, "p75": 1.0}
    assert group["gap"] == {"valid_count": 3, "min": 0.0, "max": 0.25, "mean": 0.5 / 3, "p25": 0.0, "p50": 0.25, "p75": 0.25}
    assert group["spread"] == {"valid_count": 3, "min": 0.0, "max": 0.5, "mean": 0.75 / 3, "p25": 0.0, "p50": 0.25, "p75": 0.5}
    assert group["per_rank"] == [
        {"rank": 1, "question_count": 3, "mean": 2.0 / 3, "median": 0.5},
        {"rank": 2, "question_count": 3, "mean": 1.5 / 3, "median": 0.5},
        {"rank": 3, "question_count": 3, "mean": 1.25 / 3, "median": 0.5},
    ]
    assert group["tied_adjacent_pairs"] == 3 and group["top_two_tied_questions"] == 1


def test_the_statistics_of_the_fixture_groups(section):
    answerable, unanswerable = section["groups"]["answerable"], section["groups"]["unanswerable"]
    assert answerable["top1"] == {"valid_count": 4, "min": 0.5, "max": 1.0, "mean": 3.125 / 4,
                                  "p25": 0.5, "p50": 0.75, "p75": 0.875}
    assert answerable["gap"] == {"valid_count": 4, "min": 0.0, "max": 0.25, "mean": 0.5 / 4,
                                 "p25": 0.0, "p50": 0.125, "p75": 0.125}
    assert answerable["spread"] == {"valid_count": 4, "min": 0.0, "max": 1.0, "mean": 2.375 / 4,
                                    "p25": 0.0, "p50": 0.625, "p75": 0.75}
    assert answerable["tied_adjacent_pairs"] == 7 and answerable["top_two_tied_questions"] == 1
    assert answerable["per_rank"][0] == {"rank": 1, "question_count": 4, "mean": 3.125 / 4, "median": 0.75}
    assert answerable["per_rank"][5] == {"rank": 6, "question_count": 4, "mean": 0.75 / 4, "median": 0.125}
    assert unanswerable["top1"] == {"valid_count": 2, "min": 0.375, "max": 0.5, "mean": 0.4375,
                                    "p25": 0.375, "p50": 0.375, "p75": 0.5}
    assert unanswerable["gap"]["p50"] == 0.0 and unanswerable["gap"]["p75"] == 0.125
    assert unanswerable["spread"] == {"valid_count": 2, "min": 0.5, "max": 0.625, "mean": 0.5625,
                                      "p25": 0.5, "p50": 0.5, "p75": 0.625}
    assert unanswerable["tied_adjacent_pairs"] == 3 and unanswerable["top_two_tied_questions"] == 1
    assert unanswerable["per_rank"][5] == {"rank": 6, "question_count": 2, "mean": -0.125, "median": -0.25}


def test_the_two_groups_are_computed_separately(section):
    assert section["groups"]["answerable"]["question_count"] == 4
    assert section["groups"]["unanswerable"]["question_count"] == 2
    assert section["groups"]["answerable"]["per_rank"] != section["groups"]["unanswerable"]["per_rank"]


@pytest.mark.parametrize("count", range(1, 13))
@pytest.mark.parametrize("percentile", sd.PERCENTILES)
def test_the_nearest_rank_percentile_is_ceil_p_n_over_100(count, percentile):
    values = [Fraction(i * 37 % 101, 128) for i in range(count)]
    vectors = [[float(v)] for v in sorted(values, reverse=True)]
    stat = sd.summarize_group(vectors, 1)["top1"]
    assert stat[f"p{percentile}"] == float(nearest_rank(values, percentile))
    assert stat["valid_count"] == count and stat["min"] == float(min(values)) and stat["max"] == float(max(values))


def test_the_median_of_an_even_count_is_the_lower_middle_value_not_an_average():
    stat = sd.summarize_group([[0.75], [0.5], [0.25], [0.125]], 1)["top1"]
    assert stat["p50"] == 0.25  # sorted 0.125 0.25 0.5 0.75, nearest rank 2 (an average would be 0.375)


def test_a_single_chunk_ranking_has_a_top1_and_a_spread_but_no_gap():
    group = sd.summarize_group([[0.5], [0.25]], 1)
    assert group["gap"] == {"valid_count": 0, "min": None, "max": None, "mean": None,
                            "p25": None, "p50": None, "p75": None}
    assert group["top1"]["valid_count"] == 2 and group["spread"]["valid_count"] == 2
    assert group["spread"]["min"] == 0.0 and group["spread"]["max"] == 0.0
    assert group["top_two_tied_questions"] == 0 and group["tied_adjacent_pairs"] == 0
    assert group["per_rank"] == [{"rank": 1, "question_count": 2, "mean": 0.375, "median": 0.25}]


def test_a_single_question_has_every_percentile_equal_to_its_value():
    group = sd.summarize_group([[0.5, 0.25]], 2)
    assert group["top1"] == {"valid_count": 1, "min": 0.5, "max": 0.5, "mean": 0.5, "p25": 0.5, "p50": 0.5, "p75": 0.5}
    assert group["gap"]["valid_count"] == 1 and group["gap"]["p75"] == 0.25


def test_an_empty_group_is_valid_and_has_null_statistics():
    group = sd.summarize_group([], 3)
    null = {"valid_count": 0, "min": None, "max": None, "mean": None, "p25": None, "p50": None, "p75": None}
    assert group["question_count"] == 0
    assert group["top1"] == group["gap"] == group["spread"] == null
    assert group["per_rank"] == [{"rank": r, "question_count": 0, "mean": None, "median": None} for r in (1, 2, 3)]
    assert group["tied_adjacent_pairs"] == 0 and group["top_two_tied_questions"] == 0


def test_ties_are_exact_equality_without_any_tolerance():
    above = math.nextafter(0.5, 1.0)
    group = sd.summarize_group([[above, 0.5, 0.5], [0.5, 0.5, 0.5]], 3)
    assert group["tied_adjacent_pairs"] == 3  # (0.5, 0.5) once in the first vector, twice in the second
    assert group["top_two_tied_questions"] == 1  # the first vector's first two scores differ by one ulp
    assert group["gap"]["max"] == above - 0.5


def test_a_tied_ranking_is_never_reordered_or_deduplicated():
    group = sd.summarize_group([[0.5] * 6], 6)
    assert group["tied_adjacent_pairs"] == 5 and group["per_rank"][0]["question_count"] == 1
    assert group["gap"]["min"] == 0.0 and group["spread"]["max"] == 0.0


def test_every_non_null_statistic_of_a_valid_group_is_finite(section):
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, float):
            yield value

    numbers = list(walk(section["groups"]))
    assert numbers and all(math.isfinite(n) for n in numbers)


def test_large_but_finite_scores_are_accepted_because_no_score_range_is_imposed():
    group = sd.summarize_group([[1.0e308], [5.0e307]], 1)
    assert group["top1"]["max"] == 1.0e308 and math.isfinite(group["top1"]["mean"])


# ---------- B: numeric fail-closed rules ----------

class FloatSubclass(float):
    pass


@pytest.mark.parametrize("vector,code", [
    ([], sd.SCORES_EMPTY),
    ((), sd.SCORES_EMPTY),
    (None, sd.SCORES_EMPTY),
    ("abc", sd.SCORES_EMPTY),
    ([0.5, 0.25], sd.SCORES_LENGTH),
    ([0.5, 0.25, 0.125, 0.0], sd.SCORES_LENGTH),
    ([0.5, 1, 0.25], sd.SCORES_MISSING),
    ([0.5, True, 0.25], sd.SCORES_MISSING),
    ([0.5, None, 0.25], sd.SCORES_MISSING),
    ([0.5, "0.25", 0.25], sd.SCORES_MISSING),
    ([0.5, FloatSubclass(0.25), 0.25], sd.SCORES_MISSING),
    ([0.5, math.nan, 0.25], sd.SCORES_NONFINITE),
    ([math.inf, 0.5, 0.25], sd.SCORES_NONFINITE),
    ([0.5, 0.25, -math.inf], sd.SCORES_NONFINITE),
    ([0.25, 0.5, 0.125], sd.SCORES_ORDER),
    ([0.5, 0.25, 0.375], sd.SCORES_ORDER),
])
def test_an_invalid_score_vector_is_refused_with_a_fixed_message(vector, code):
    error = refused(code, sd.summarize_group, [vector], 3)
    assert [i.message for i in error.issues] == [sd.MESSAGES[code]]


def test_a_rising_sequence_is_refused_not_sorted():
    vector = [0.25, 0.5, 0.125]
    refused(sd.SCORES_ORDER, sd.summarize_group, [vector], 3)
    assert vector == [0.25, 0.5, 0.125]  # the input was not touched either


def test_the_first_invalid_vector_of_a_group_names_its_position():
    error = refused(sd.SCORES_ORDER, sd.summarize_group, [[0.5, 0.25], [0.25, 0.5]], 2)
    assert [i.location for i in error.issues] == ["vectors[1]"]


def test_a_ranking_length_below_one_is_refused():
    refused(sd.SCORES_EMPTY, sd.summarize_group, [], 0)


@pytest.mark.parametrize("vectors,length", [("x", 1), (b"x", 1), (5, 1), ([[0.5]], "1"), ([[0.5]], 1.0)])
def test_wrongly_typed_arguments_are_a_type_error(vectors, length):
    with pytest.raises(TypeError):
        sd.summarize_group(vectors, length)


def test_a_subtraction_that_overflows_the_spread_is_refused_not_clamped():
    # 1e308 - (-1e308) is infinite although both scores are finite; the gap (1e308 - 1e308) is fine
    error = refused(sd.SCORES_OVERFLOW, sd.summarize_group, [[1.0e308, 1.0e308, -1.0e308]], 3)
    assert [i.location for i in error.issues] == ["spread"]


def test_a_subtraction_that_overflows_the_gap_is_refused_not_clamped():
    error = refused(sd.SCORES_OVERFLOW, sd.summarize_group, [[1.0e308, -1.0e308, -1.0e308]], 3)
    assert [i.location for i in error.issues] == ["gap"]


def test_an_aggregation_that_overflows_is_refused_not_clamped():
    # math.fsum raises OverflowError for these finite values; the mean must not become infinite or be replaced
    with pytest.raises(OverflowError):
        math.fsum([1.7e308, 1.7e308])
    error = refused(sd.SCORES_OVERFLOW, sd.summarize_group, [[1.7e308], [1.7e308]], 1)
    assert [i.location for i in error.issues] in (["top1"], ["per_rank[1]"])
    refused(sd.SCORES_OVERFLOW, sd.summarize_group, [[-1.7e308], [-1.7e308]], 1)


def test_an_overflowing_per_rank_mean_is_refused():
    refused(sd.SCORES_OVERFLOW, sd.summarize_group, [[0.0, -1.7e308], [0.0, -1.7e308]], 2)


def test_no_non_finite_value_can_reach_a_section(flow, inputs, pool):
    with pytest.raises(s.EvaluationDataError):
        sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, [
            sd.QuestionScores("du1", (math.inf, 0.5, 0.25, 0.25, 0.125, 0.0)), flow.unanswerable[1]])


# ---------- C: the digest ----------

def test_the_digest_is_the_sha256_of_the_documented_payload():
    answerable = [sd.QuestionScores("a", (0.5, 0.25))]
    unanswerable = [sd.QuestionScores("b", (0.25, 0.0))]
    text = '["e2.3-scores-digest-1",2,[["a",[0.5,0.25]]],[["b",[0.25,0.0]]]]'
    assert sd.scores_digest(2, answerable, unanswerable) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_an_empty_unanswerable_group_has_a_digest_too():
    text = '["e2.3-scores-digest-1",1,[["a",[0.5]]],[]]'
    assert sd.scores_digest(1, [sd.QuestionScores("a", (0.5,))], []) == hashlib.sha256(text.encode()).hexdigest()


def test_the_digest_of_a_run_matches_an_independent_payload(section):
    answerable = [[q.question_id, VECTORS[q.question]] for q in ANSWERABLE]
    unanswerable = [[q.question_id, VECTORS[q.question]] for q in UNANSWERABLE]
    text = json.dumps(["e2.3-scores-digest-1", 6, answerable, unanswerable], separators=(",", ":"), ensure_ascii=False)
    assert section["scores_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_digest_changes_with_one_ulp_the_order_the_groups_and_the_ids():
    a, b = sd.QuestionScores("a", (0.5, 0.25)), sd.QuestionScores("b", (0.25, 0.125))
    base = sd.scores_digest(2, [a, b], [])
    assert sd.scores_digest(2, [a, b], []) == base
    assert sd.scores_digest(2, [b, a], []) != base
    assert sd.scores_digest(2, [a], [b]) != base
    assert sd.scores_digest(2, [], [a, b]) != base
    assert sd.scores_digest(2, [sd.QuestionScores("a", (math.nextafter(0.5, 1.0), 0.25)), b], []) != base
    assert sd.scores_digest(2, [sd.QuestionScores("c", (0.5, 0.25)), b], []) != base
    assert sd.scores_digest(3, [a, b], []) != base


def test_negative_zero_and_zero_are_different_scores_to_the_digest():
    assert sd.scores_digest(1, [sd.QuestionScores("a", (0.0,))], []) != sd.scores_digest(
        1, [sd.QuestionScores("a", (-0.0,))], [])


# ---------- D: the recorder and the unanswerable collection ----------

def test_the_recorder_returns_the_retriever_result_unchanged():
    result = [fx.C[1]]
    recorder = sd.ScoreRecorder(lambda query: result)
    assert recorder("anything") is result
    as_tuple = (fx.C[1],)
    assert sd.ScoreRecorder(lambda query: as_tuple)("anything") is as_tuple


def test_the_recorder_keeps_a_query_hash_and_never_the_text():
    recorder = sd.ScoreRecorder(scored())
    recorder(fx.Q1.question)
    (call,) = recorder.take()
    assert call.query_sha256 == hashlib.sha256(fx.Q1.question.encode("utf-8")).hexdigest()
    assert fx.Q1.question not in repr(call) and "alpha" not in repr(call).lower()
    assert [entry[4] for entry in call.chunks] == VECTORS[fx.Q1.question]
    assert call.chunks[0][0] == fx.C[1].document_id  # temporary identity data, never in a report


def test_the_recorder_is_single_use():
    recorder = sd.ScoreRecorder(scored())
    recorder(fx.Q1.question)
    assert len(recorder.take()) == 1
    refused(sd.SCORES_RECORDER_STATE, recorder.take)
    refused(sd.SCORES_RECORDER_STATE, recorder, fx.Q2.question)


def test_a_failing_retriever_records_nothing_and_its_error_propagates():
    def broken(query):
        raise RuntimeError("retriever down")

    recorder = sd.ScoreRecorder(broken)
    with pytest.raises(RuntimeError):
        recorder("q")
    assert recorder.take() == ()


@pytest.mark.parametrize("query", [None, 1, b"x", ["x"]])
def test_the_recorder_takes_only_a_str_query(query):
    with pytest.raises(TypeError):
        sd.ScoreRecorder(scored())(query)


@pytest.mark.parametrize("retriever", [None, "x", 1])
def test_the_recorder_wraps_only_a_callable(retriever):
    with pytest.raises(TypeError):
        sd.ScoreRecorder(retriever)


def test_a_non_list_result_and_a_chunk_without_fields_are_snapshotted_without_raising():
    odd = sd.ScoreRecorder(lambda query: "not a list")
    assert odd("q") == "not a list"
    bare = sd.ScoreRecorder(lambda query: [types.SimpleNamespace(document_id=1)])
    bare("q")
    assert odd.take()[0].chunks is None
    assert bare.take()[0].chunks == ((1, None, None, None, None),)


def test_the_unanswerable_questions_are_retrieved_in_dev_file_order(inputs, pool):
    asked = []
    retriever = scored()

    def spy(query):
        asked.append(query)
        return retriever(query)

    rows = sd.collect_unanswerable_scores(inputs, pool, spy)
    assert asked == [fx.U1.question, fx.U2.question]
    assert [row.question_id for row in rows] == ["du1", "du2"]
    assert [row.scores for row in rows] == [tuple(VECTORS[fx.U1.question]), tuple(VECTORS[fx.U2.question])]


def test_the_unanswerable_path_refuses_a_recorder(inputs, pool):
    with pytest.raises(TypeError):
        sd.collect_unanswerable_scores(inputs, pool, sd.ScoreRecorder(scored()))


@pytest.mark.parametrize("args", [(None, "pool", scored()), ("inputs", None, scored()), ("inputs", "pool", None),
                                  ("inputs", "pool", "x")])
def test_the_unanswerable_path_checks_its_argument_types(inputs, pool, args):
    chosen = [inputs if a == "inputs" else pool if a == "pool" else a for a in args]
    with pytest.raises(TypeError):
        sd.collect_unanswerable_scores(*chosen)


def test_an_unanswerable_ranking_must_come_from_the_pool(inputs, pool):
    foreign = dataclasses.replace(fx.C[1], chunk_id=99)

    def retrieve(query):
        return [foreign, *scored()(query)[1:]]

    error = refused("RANK_NOT_IN_POOL", sd.collect_unanswerable_scores, inputs, pool, retrieve)
    assert error.issues[0].location == "unanswerable[0].retrieved[0]"


def test_an_unanswerable_ranking_must_have_the_expected_length(inputs, pool):
    refused(sd.SCORES_LENGTH, sd.collect_unanswerable_scores, inputs, pool, lambda query: scored()(query)[:5])
    refused(sd.SCORES_LENGTH, sd.collect_unanswerable_scores, inputs, pool, lambda query: [])


@pytest.mark.parametrize("value", ["a string", None, 5, {"a": 1}])
def test_the_unanswerable_retriever_must_return_a_list_or_tuple(inputs, pool, value):
    refused(sd.SCORES_RETURN, sd.collect_unanswerable_scores, inputs, pool, lambda query: value)


@pytest.mark.parametrize("bad,code", [
    (1, sd.SCORES_MISSING), (None, sd.SCORES_MISSING), (True, sd.SCORES_MISSING), (math.nan, sd.SCORES_NONFINITE),
    (math.inf, sd.SCORES_NONFINITE), (-math.inf, sd.SCORES_NONFINITE),
])
def test_an_invalid_unanswerable_score_is_refused(inputs, pool, bad, code):
    def retrieve(query):
        chunks = scored()(query)
        chunks[2] = dataclasses.replace(chunks[2], score=bad)
        return chunks

    refused(code, sd.collect_unanswerable_scores, inputs, pool, retrieve)


def test_a_rising_unanswerable_ranking_is_refused(inputs, pool):
    vectors = {**VECTORS, fx.U2.question: [0.0, 0.125, 0.25, 0.25, 0.375, 0.5]}
    refused(sd.SCORES_ORDER, sd.collect_unanswerable_scores, inputs, pool, scored(vectors))


def test_an_unanswerable_chunk_without_a_score_is_refused(inputs, pool):
    def retrieve(query):
        return [types.SimpleNamespace(**{k: getattr(c, k) for k in vars(c) if k != "score"})
                for c in scored()(query)]

    refused(sd.SCORES_MISSING, sd.collect_unanswerable_scores, inputs, pool, retrieve)


# ---------- E: the section of a real run ----------

def test_the_section_has_exactly_the_documented_keys(section):
    assert set(section) == {"format", "ranking_length", "scores_sha256", "second_pass_scores_sha256",
                            "scores_identical_across_passes", "groups", "questions"}
    assert section["format"] == "e2.3-score-diagnostics-1" and section["ranking_length"] == 6
    assert set(section["groups"]) == {"answerable", "unanswerable"}
    assert set(section["groups"]["answerable"]) == {"question_count", "top1", "gap", "spread", "per_rank",
                                                    "tied_adjacent_pairs", "top_two_tied_questions"}
    assert ids(section, "answerable") == ["q1", "q2", "q3", "q4"] and ids(section, "unanswerable") == ["du1", "du2"]
    assert section["second_pass_scores_sha256"] is None and section["scores_identical_across_passes"] is None
    assert all(set(row) == {"question_id", "scores"} for g in section["questions"].values() for row in g)
    assert re.fullmatch(r"[0-9a-f]{64}", section["scores_sha256"])


def test_the_scores_are_the_retrieved_scores_in_rank_order(section):
    for question in ANSWERABLE + UNANSWERABLE:
        group = "answerable" if question.answerable else "unanswerable"
        row = next(r for r in section["questions"][group] if r["question_id"] == question.question_id)
        assert row["scores"] == VECTORS[question.question]


def test_the_section_is_valid_for_its_own_run(section, flow):
    assert sd.section_ok(section, **ok_arguments(flow))


def ok_arguments(flow):
    return {
        "answerable_ids": [q.question_id for q in flow.run.questions],
        "unanswerable_ids": [u.question_id for u in flow.run.unanswerable],
        "pool_size": flow.run.pool_size,
        "answerable_ranking_lengths": [len(q.dense) for q in flow.run.questions],
    }


def test_the_section_is_json_serialisable_with_exact_floats(section):
    text = json.dumps(section, sort_keys=True, allow_nan=False)
    assert json.loads(text) == section
    assert json.dumps(json.loads(text), sort_keys=True, allow_nan=False) == text


def test_the_second_pass_digest_is_recorded_and_compared_without_a_gate(flow, inputs, pool, section):
    same = sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, flow.unanswerable,
                                      second_pass_scores_sha256=section["scores_sha256"])
    assert same["scores_identical_across_passes"] is True
    assert same["second_pass_scores_sha256"] == section["scores_sha256"]
    other = sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, flow.unanswerable,
                                       second_pass_scores_sha256="0" * 64)
    assert other["scores_identical_across_passes"] is False and other["scores_sha256"] == section["scores_sha256"]
    assert sd.section_ok(same, **ok_arguments(flow)) and sd.section_ok(other, **ok_arguments(flow))


@pytest.mark.parametrize("value", ["", "A" * 64, "a" * 63, "g" * 64, 5, b"a" * 64])
def test_a_malformed_second_pass_digest_is_refused(flow, inputs, pool, value):
    refused(sd.SCORES_DIGEST_INVALID, sd.build_score_diagnostics, flow.run, inputs, pool, flow.calls,
            flow.unanswerable, second_pass_scores_sha256=value)


def test_an_empty_unanswerable_group_in_a_real_run(pool):
    only = fx.inputs_of(fx.Q1, fx.Q2, fx.Q3, fx.Q4)
    recorder = sd.ScoreRecorder(scored())
    run = engine.run_pilot(only, pool, recorder, **SEED_RESAMPLES)
    built = sd.build_score_diagnostics(run, only, pool, recorder.take(), sd.collect_unanswerable_scores(
        only, pool, scored()))
    assert built["groups"]["unanswerable"]["question_count"] == 0
    assert built["groups"]["unanswerable"]["top1"]["valid_count"] == 0 and built["questions"]["unanswerable"] == []
    assert sd.section_ok(built, answerable_ids=["q1", "q2", "q3", "q4"], unanswerable_ids=[], pool_size=run.pool_size,
                         answerable_ranking_lengths=[6, 6, 6, 6])


# --- alignment between the recorded calls and the run ---

def test_a_missing_or_an_extra_recorded_call_is_refused(flow, inputs, pool):
    for calls in (flow.calls[:-1], (*flow.calls, flow.calls[0]), ()):
        refused(sd.SCORES_CALL_COUNT, sd.build_score_diagnostics, flow.run, inputs, pool, calls, flow.unanswerable)


def test_calls_in_the_wrong_order_are_refused_by_their_query_hash(flow, inputs, pool):
    swapped = (flow.calls[1], flow.calls[0], *flow.calls[2:])
    refused(sd.SCORES_QUERY_MISMATCH, sd.build_score_diagnostics, flow.run, inputs, pool, swapped, flow.unanswerable)
    duplicated = (flow.calls[0], flow.calls[0], *flow.calls[2:])
    refused(sd.SCORES_QUERY_MISMATCH, sd.build_score_diagnostics, flow.run, inputs, pool, duplicated, flow.unanswerable)


def mismatching(change):
    """A recorder run whose Q1 ranking differs from the run's (the other questions are the same)."""
    base = scored()

    def retrieve(query):
        chunks = base(query)
        return change(chunks) if query == fx.Q1.question else chunks

    return calls_of(retrieve)


@pytest.mark.parametrize("change", [
    lambda chunks: [chunks[1], chunks[0], *chunks[2:]],                                    # order
    lambda chunks: chunks[:5],                                                             # count (shorter)
    lambda chunks: [*chunks, chunks[0]],                                                   # count (longer)
    lambda chunks: [dataclasses.replace(chunks[0], span_sha256="0" * 64), *chunks[1:]],    # identity (hash)
    lambda chunks: [dataclasses.replace(chunks[0], start_offset=chunks[0].start_offset + 1), *chunks[1:]],
    lambda chunks: [dataclasses.replace(chunks[0], end_offset=chunks[0].end_offset + 1), *chunks[1:]],
    lambda chunks: [dataclasses.replace(chunks[0], document_id=fx.C[4].document_id), *chunks[1:]],  # other document
    lambda chunks: [dataclasses.replace(chunks[0], document_id=424242), *chunks[1:]],               # unknown document
    lambda chunks: [dataclasses.replace(chunks[0], document_id=None), *chunks[1:]],
    lambda chunks: [dataclasses.replace(chunks[0], document_id=[1]), *chunks[1:]],                  # unhashable
])
def test_a_recorded_ranking_that_differs_from_the_run_is_refused(flow, inputs, pool, change):
    refused(sd.SCORES_RANKING_MISMATCH, sd.build_score_diagnostics, flow.run, inputs, pool, mismatching(change),
            flow.unanswerable)


def test_a_non_list_recorded_result_is_a_ranking_mismatch(flow, inputs, pool):
    recorder = sd.ScoreRecorder(lambda query: "nothing")
    for question in ANSWERABLE:
        recorder(question.question)
    refused(sd.SCORES_RANKING_MISMATCH, sd.build_score_diagnostics, flow.run, inputs, pool, recorder.take(),
            flow.unanswerable)


def with_score(index, value):
    base = scored()

    def retrieve(query):
        chunks = base(query)
        if query == fx.Q4.question:
            chunks[index] = dataclasses.replace(chunks[index], score=value)
        return chunks

    return calls_of(retrieve)


@pytest.mark.parametrize("value,code", [
    (1, sd.SCORES_MISSING), (None, sd.SCORES_MISSING), (False, sd.SCORES_MISSING), ("0.5", sd.SCORES_MISSING),
    (FloatSubclass(0.5), sd.SCORES_MISSING), (math.nan, sd.SCORES_NONFINITE), (math.inf, sd.SCORES_NONFINITE),
    (-math.inf, sd.SCORES_NONFINITE),
])
def test_an_invalid_recorded_score_is_refused(flow, inputs, pool, value, code):
    refused(code, sd.build_score_diagnostics, flow.run, inputs, pool, with_score(2, value), flow.unanswerable)


def test_a_rising_recorded_ranking_is_refused(flow, inputs, pool):
    refused(sd.SCORES_ORDER, sd.build_score_diagnostics, flow.run, inputs, pool, with_score(0, 0.0), flow.unanswerable)


def test_a_chunk_without_a_score_attribute_is_refused_even_though_its_identity_matches(flow, inputs, pool):
    base = scored()

    def retrieve(query):
        chunks = base(query)
        if query == fx.Q1.question:
            fields = ("document_id", "start_offset", "end_offset", "span_sha256")
            chunks[0] = types.SimpleNamespace(**{name: getattr(chunks[0], name) for name in fields})
        return chunks

    refused(sd.SCORES_MISSING, sd.build_score_diagnostics, flow.run, inputs, pool, calls_of(retrieve),
            flow.unanswerable)


def test_the_unanswerable_rows_must_be_the_unanswerable_dev_questions_in_order(flow, inputs, pool):
    for rows in (flow.unanswerable[::-1], flow.unanswerable[:1], (), (*flow.unanswerable, flow.unanswerable[0]),
                 [sd.QuestionScores("other", flow.unanswerable[0].scores), flow.unanswerable[1]]):
        refused(sd.SCORES_QUESTION_IDS, sd.build_score_diagnostics, flow.run, inputs, pool, flow.calls, rows)


def test_unanswerable_rows_are_validated_again_when_building(flow, inputs, pool):
    bad = [sd.QuestionScores("du1", (0.25, 0.5, 0.25, 0.25, 0.125, 0.0)), flow.unanswerable[1]]
    refused(sd.SCORES_ORDER, sd.build_score_diagnostics, flow.run, inputs, pool, flow.calls, bad)
    short = [sd.QuestionScores("du1", (0.5, 0.25)), flow.unanswerable[1]]
    refused(sd.SCORES_LENGTH, sd.build_score_diagnostics, flow.run, inputs, pool, flow.calls, short)


@pytest.mark.parametrize("position,value", [
    (0, "run"), (1, "inputs"), (2, "pool"), (3, "calls"), (4, "rows"),
])
def test_build_score_diagnostics_checks_its_argument_types(flow, inputs, pool, position, value):
    args = [flow.run, inputs, pool, flow.calls, flow.unanswerable]
    args[position] = value if value != "calls" and value != "rows" else "x"
    with pytest.raises(TypeError):
        sd.build_score_diagnostics(*args)


def test_a_non_recorded_call_object_is_a_type_error(flow, inputs, pool):
    with pytest.raises(TypeError):
        sd.build_score_diagnostics(flow.run, inputs, pool, [None] * 4, flow.unanswerable)


# --- the validator ---

@pytest.mark.parametrize("change", [
    lambda d: d.update(format="x"),
    lambda d: d.update(ranking_length=5),
    lambda d: d.update(ranking_length=True),
    lambda d: d.update(scores_sha256="0" * 64),
    lambda d: d.update(second_pass_scores_sha256="A" * 64),
    lambda d: d.update(scores_identical_across_passes=False),
    lambda d: d.update(extra=1),
    lambda d: d.pop("questions"),
    lambda d: d["groups"].pop("unanswerable"),
    lambda d: d["groups"]["answerable"].update(question_count=5),
    lambda d: d["groups"]["answerable"]["gap"].update(valid_count=3),
    lambda d: d["groups"]["answerable"]["top1"].update(p50=0.0),
    lambda d: d["groups"]["answerable"]["per_rank"][3].update(question_count=3),
    lambda d: d["groups"]["answerable"]["per_rank"].pop(),
    lambda d: d["groups"]["unanswerable"].update(top_two_tied_questions=0),
    lambda d: d["questions"].update(extra=[]),
    lambda d: d["questions"]["answerable"].pop(),
    lambda d: d["questions"]["answerable"][0].update(question_id="x"),
    lambda d: d["questions"]["answerable"][0].update(extra=1),
    lambda d: d["questions"]["answerable"][0]["scores"].append(0.0),
    lambda d: d["questions"]["answerable"][0]["scores"].__setitem__(5, 0.5),
    lambda d: d["questions"]["answerable"][0]["scores"].__setitem__(0, 1),
    lambda d: d["questions"]["answerable"][0]["scores"].__setitem__(0, math.nan),
    lambda d: d["questions"]["unanswerable"].reverse(),
])
def test_the_validator_refuses_every_tampering(section, flow, change):
    broken = copy.deepcopy(section)
    change(broken)
    assert sd.section_ok(broken, **ok_arguments(flow)) is False


@pytest.mark.parametrize("value", [None, 1, "x", [], {}, [1], {"format": "e2.3-score-diagnostics-1"}])
def test_the_validator_never_raises_on_garbage(flow, value):
    assert sd.section_ok(value, **ok_arguments(flow)) is False


def test_the_validator_checks_the_ids_lengths_and_pool_size_it_is_given(section, flow):
    base = ok_arguments(flow)
    assert sd.section_ok(section, **{**base, "answerable_ids": ["q1", "q2", "q3", "other"]}) is False
    assert sd.section_ok(section, **{**base, "unanswerable_ids": ["du2", "du1"]}) is False
    assert sd.section_ok(section, **{**base, "pool_size": 5}) is False
    assert sd.section_ok(section, **{**base, "pool_size": 0}) is False
    assert sd.section_ok(section, **{**base, "answerable_ranking_lengths": [6, 6, 6, 5]}) is False
    assert sd.section_ok(section, **{**base, "answerable_ranking_lengths": [6, 6, 6]}) is False
    assert sd.section_ok(section, **{**base, "pool_size": "6"}) is False


def test_a_pool_larger_than_twenty_chunks_expects_twenty_scores(section, flow):
    assert sd.expected_length(40) == 20 and sd.expected_length(6) == 6 and sd.expected_length(20) == 20
    assert sd.section_ok(section, **{**ok_arguments(flow), "pool_size": 40}) is False  # 6 scores, not 20


# ---------- F: determinism, messages, purity, privacy ----------

def test_building_twice_gives_identical_sections_and_bytes(flow, inputs, pool, section):
    again = sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, flow.unanswerable)
    assert again == section
    assert json.dumps(again, sort_keys=True).encode() == json.dumps(section, sort_keys=True).encode()


def test_a_changed_score_changes_the_digest_and_nothing_else_unexpected(flow, inputs, pool, section):
    vectors = {**VECTORS, fx.Q3.question: [0.5, 0.5, 0.5, 0.5, 0.5, 0.25]}
    calls = calls_of(scored(vectors))
    other = sd.build_score_diagnostics(flow.run, inputs, pool, calls, flow.unanswerable)
    assert other["scores_sha256"] != section["scores_sha256"]
    assert other["groups"]["unanswerable"] == section["groups"]["unanswerable"]
    assert other["groups"]["answerable"] != section["groups"]["answerable"]


def test_every_code_has_one_fixed_message_and_no_message_holds_a_value():
    codes = {value for key, value in vars(sd).items() if key.startswith("SCORES_") and isinstance(value, str)}
    assert set(sd.MESSAGES) == codes and len(codes) == 13
    assert all(isinstance(text, str) and text and not re.search(r"\d", text.replace("min(20", "").replace("64", "")) for text in
               sd.MESSAGES.values())


def test_errors_carry_schema_locations_and_fixed_messages_only(inputs, pool):
    def retrieve(query):
        chunks = scored()(query)
        chunks[0] = dataclasses.replace(chunks[0], score=math.nan)
        return chunks

    error = refused(sd.SCORES_NONFINITE, sd.collect_unanswerable_scores, inputs, pool, retrieve)
    for issue in error.issues:
        assert issue.severity == "error" and issue.message == sd.MESSAGES[sd.SCORES_NONFINITE]
        assert re.fullmatch(r"[A-Za-z0-9_.\[\]]+", issue.location)
    assert "nan" not in " ".join(i.message for i in error.issues).lower()


def test_the_section_holds_no_text_identifier_or_path(section):
    text = json.dumps(section)
    for question in ANSWERABLE + UNANSWERABLE:
        assert question.question not in text
    for forbidden in ("document_id", "chunk_id", "owner", "query", "span_sha256", "text_sha256", ".jsonl", "eval/",
                      "corpus_key", "doc-a", "doc-b"):
        assert forbidden not in text
    assert str(fx.OWNER) not in text
    # the document id as a standalone token (not inside a hex digest or a longer number)
    assert not re.search(rf"(?<![0-9A-Za-z.]){fx.ID_A}(?![0-9A-Za-z.])", text)


def test_building_and_collecting_open_no_file_and_never_touch_the_filesystem(flow, inputs, pool, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the diagnostics must not touch the filesystem")

    with monkeypatch.context() as patch:
        for owner, names in ((builtins, ("open",)), (io, ("open",)), (os, ("open", "stat", "lstat", "scandir",
                                                                           "listdir"))):
            for name in names:
                patch.setattr(owner, name, forbidden)
        for name in ("read_bytes", "read_text", "open", "stat", "exists", "iterdir", "glob", "resolve"):
            patch.setattr(Path, name, forbidden)
        assert sd.collect_unanswerable_scores(inputs, pool, scored())
        assert sd.build_score_diagnostics(flow.run, inputs, pool, flow.calls, flow.unanswerable)
        assert sd.summarize_group([[0.5, 0.25]], 2)


def _module_ast():
    return ast.parse(Path(sd.__file__).read_text(encoding="utf-8"))


def test_the_module_imports_only_the_standard_library_and_the_existing_evaluation_modules():
    allowed = {"__future__", "hashlib", "itertools", "json", "math", "collections.abc", "dataclasses", "typing",
               "app.rag.evaluation.metrics", "app.rag.evaluation.pool", "app.rag.evaluation.runner_inputs",
               "app.rag.evaluation.schema", "app.rag.evaluation"}
    found = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            found.add(node.module)
    assert found <= allowed
    assert not {"os", "pathlib", "io", "sqlalchemy", "app.rag.retrieval", "app.rag.evaluation.report"} & found


def test_the_module_has_no_file_access_and_never_names_a_private_file():
    source = Path(sd.__file__).read_text(encoding="utf-8")
    tree = _module_ast()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "open" not in names and "Path" not in names
    assert not attributes & {"read_bytes", "read_text", "open", "stat", "lstat", "exists", "iterdir", "glob",
                             "resolve", "realpath", "listdir", "scandir"}
    for private in ("test.jsonl", "audit.jsonl", "eval/dataset", "dev.jsonl"):
        assert private not in source


def test_scores_are_never_sorted_in_place_or_otherwise_repaired():
    tree = _module_ast()
    sorted_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "sorted"]
    # sorted() is used only to order values for the percentile statistics, never on a score vector
    assert len(sorted_calls) == 2
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("sort", "reverse")]
