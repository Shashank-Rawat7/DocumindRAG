"""E1.7.3: the pure evaluation run engine (fake retriever, hand-computed fixtures).

Fixture: two documents (doc-a, doc-b), three non-overlapping chunks each, so a pool of six chunks
c1..c6 (c1-c3 in doc-a, c4-c6 in doc-b). Every ranking therefore holds all six chunks.
Dev questions (all en_to_en): q1 gold c1; q2 gold c4; q3 two units, c2 and c6; q4 gold span that no single
chunk covers by half (unreachable under half_coverage); plus two unanswerable questions.

A  identical questions and pool for dense, oracle and random
B  hand-computed dense and oracle metrics at every cutoff
C  random: seeds minimum, uniqueness, determinism, exact and statistical values
D  oracle validity and unreachable units
E  unanswerable questions kept apart
F  observations, series and reports; stable ordering
G  malformed rankings and arguments
H  purity

Run focused: pytest tests/test_evaluation_engine.py --noconftest -p no:cacheprovider
"""
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import baselines
from app.rag.evaluation import engine as e
from app.rag.evaluation import pool as p
from app.rag.evaluation import runner_inputs as r
from app.rag.evaluation import schema as s
from app.rag.evaluation.metrics import CUTOFFS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OWNER, EMB = 10, "emb-1"
ID_A, ID_B = 931, 417
TEXT_A = "Alpha clause one. Beta clause two. Gamma clause three."
TEXT_B = "Delta section four. Epsilon section five. Zeta section six."
TEXTS = {"doc-a": TEXT_A, "doc-b": TEXT_B}
RESAMPLES = 1000


def mdoc(key, text):
    return s.ManifestDocument(
        corpus_key=key, title="T", publisher="P", source_url="https://example.invalid/a", license="MIT",
        license_url="https://example.invalid/l", format="txt", language="en",
        source_file_sha256=hashlib.sha256(text.encode()).hexdigest(),
        extracted_text_sha256=sha256_text(text), extracted_text_length=len(text),
        extraction=s.Extraction(processing_ref="d" * 40, python="3.12.4"), redistributable=True,
    )


MANIFEST = s.Manifest("eval-manifest-v1", "c1", "ingested", (mdoc("doc-a", TEXT_A), mdoc("doc-b", TEXT_B)))


@dataclasses.dataclass
class Fake:
    chunk_id: int
    document_id: int
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    span_sha256: str
    source_version: int = 1
    source_sha256: str = ""
    score: float = 0.5


def chunk(chunk_id, document_id, key, index, quote):
    start = TEXTS[key].index(quote)
    return Fake(chunk_id, document_id, index, start, start + len(quote), quote, sha256_text(quote),
                source_sha256=sha256_text(TEXTS[key]))


C = {
    1: chunk(1, ID_A, "doc-a", 0, "Alpha clause one."),
    2: chunk(2, ID_A, "doc-a", 1, "Beta clause two."),
    3: chunk(3, ID_A, "doc-a", 2, "Gamma clause three."),
    4: chunk(4, ID_B, "doc-b", 0, "Delta section four."),
    5: chunk(5, ID_B, "doc-b", 1, "Epsilon section five."),
    6: chunk(6, ID_B, "doc-b", 2, "Zeta section six."),
}


def row(document_id, key):
    return p.PoolDocument(document_id, OWNER, f"{key}.txt", f"c1:{key}", 1, 1, EMB, sha256_text(TEXTS[key]))


def make_pool(chunks=None):
    dmap = p.map_documents(MANIFEST, [row(ID_A, "doc-a"), row(ID_B, "doc-b")], owner_id=OWNER, embedding_id=EMB)
    return p.build_pool(MANIFEST, dmap, list(C.values()) if chunks is None else chunks, TEXTS)


def span(key, quote):
    text = TEXTS[key]
    start = text.index(quote)
    return s.Span(key, sha256_text(text), start, start + len(quote), quote, sha256_text(quote))


def question(qid, text, *spans):
    units = tuple(s.EvidenceUnit(f"u{i}", (sp,)) for i, sp in enumerate(spans, 1))
    return s.Question(
        question_id=qid, split="dev", answerable=True, question=text, query_language="en", author_type="human",
        hindi_reviewed=False, expected_answer="x", units=units, source_language="en", difficulty="direct",
        close_distractor=False,
    )


def unanswerable(qid, text):
    return s.Question(
        question_id=qid, split="dev", answerable=False, question=text, query_language="en", author_type="human",
        hindi_reviewed=False, unanswerable_type="off_topic",
    )


Q1 = question("q1", "What does the first alpha clause say?", span("doc-a", "Alpha clause one."))
Q2 = question("q2", "Which delta section comes fourth in the list?", span("doc-b", "Delta section four."))
Q3 = question("q3", "Which two clauses cover beta and zeta topics?",
              span("doc-a", "Beta clause two."), span("doc-b", "Zeta section six."))
Q4 = question("q4", "What spans several clauses across the middle?",
              span("doc-a", "clause one. Beta clause two. Gamma"))
U1 = unanswerable("du1", "How many employees work in the delivery department?")
U2 = unanswerable("du2", "What colour is the company vehicle fleet painted?")


def inputs_of(*questions):
    import json
    text = "".join(json.dumps(dataclasses.asdict(q), ensure_ascii=False) + "\n" for q in questions)
    return r.build_dev_inputs(text, MANIFEST, TEXTS, near_duplicate_threshold=0.6)


DENSE = {
    Q1.question: [1, 2, 3, 4, 5, 6],
    Q2.question: [1, 4, 2, 3, 5, 6],
    Q3.question: [2, 5, 3, 1, 4, 6],
    Q4.question: [3, 2, 1, 6, 5, 4],
}


class Retriever:
    def __init__(self, table=None):
        self.table = DENSE if table is None else table
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        return [C[i] for i in self.table[query]]


SEEDS = e.default_random_seeds()


def run(questions=(Q1, Q2, Q3, Q4, U1, U2), retriever=None, **kw):
    kw.setdefault("seeds", SEEDS)
    kw.setdefault("bootstrap_seed", 7)
    kw.setdefault("resamples", RESAMPLES)
    return e.run_pilot(inputs_of(*questions), make_pool(), retriever or Retriever(), **kw)


@pytest.fixture(scope="module")
def result():
    return run(adjudicated_unreachable={"q4": ["u1"]})


def key(rankingscores, family, metric, k=None, rule=None):
    return rankingscores.get(family, metric, k, rule)


HALF = "half_coverage"


# ---------- A: identical questions and pool ----------

def test_dense_oracle_and_random_use_the_same_questions_and_the_same_pool(monkeypatch):
    seen = {"oracle": [], "random": [], "dense": []}
    real_oracle, real_random, real_convert = baselines.oracle_baseline, baselines.random_baseline, e.convert_retrieved

    def oracle(q, pool):
        seen["oracle"].append((q, pool))
        return real_oracle(q, pool)

    def random(q, pool, seeds):
        seen["random"].append((q, pool, seeds))
        return real_random(q, pool, seeds)

    def convert(pool, retrieved):
        seen["dense"].append(pool)
        return real_convert(pool, retrieved)

    monkeypatch.setattr(baselines, "oracle_baseline", oracle)
    monkeypatch.setattr(baselines, "random_baseline", random)
    monkeypatch.setattr(e, "convert_retrieved", convert)
    inputs, pool = inputs_of(Q1, Q2, Q3, U1), make_pool()
    e.run_pilot(inputs, pool, Retriever(), seeds=SEEDS, bootstrap_seed=1, resamples=RESAMPLES)

    assert [q for q, _ in seen["oracle"]] == list(inputs.answerable) == [q for q, *_ in seen["random"]]
    assert all(q is a for (q, _), a in zip(seen["oracle"], inputs.answerable))
    assert all(pl is pool.ranked for _, pl in seen["oracle"]) and all(pl is pool.ranked for _, pl, _ in seen["random"])
    assert all(pl is pool for pl in seen["dense"]) and len(seen["dense"]) == 3
    assert all(sd == SEEDS for *_, sd in seen["random"])


def test_the_three_versions_have_the_same_question_ids_types_and_series(result):
    ids = [o.question_id for o in result.dense_observations]
    assert ids == [o.question_id for o in result.oracle_observations] == [o.question_id for o in result.random_observations]
    assert ids == ["q1", "q2", "q3", "q4"]
    assert {o.retrieval_type for o in result.dense_observations} == {"en_to_en"}
    names = [name for name, _ in result.dense_observations[0].values]
    assert names == [sp.name for sp in result.series] == [sp.name for sp in result.series_continuous]
    assert [name for name, _ in result.random_observations[0].values] == names


def test_the_result_is_bound_to_the_inputs_and_the_pool(result):
    inputs, pool = inputs_of(Q1, Q2, Q3, Q4, U1, U2), make_pool()
    assert result.manifest_sha256 == inputs.manifest_sha256
    assert result.dev_dataset_sha256 == inputs.dev_dataset_sha256
    assert result.pool_sha256 == pool.pool_sha256 and result.pool_size == 6
    assert result.seeds == SEEDS and result.cutoffs == CUTOFFS


def test_the_retriever_sees_only_query_text_of_answerable_questions_in_order():
    retriever = Retriever()
    run(retriever=retriever)
    assert retriever.queries == [Q1.question, Q2.question, Q3.question, Q4.question]


# ---------- B: hand-computed dense and oracle metrics ----------

def test_dense_hit_at_every_cutoff_for_q2(result):
    sc = result.questions[1].dense_scores  # ranking c1, c4, ...: the gold chunk is second
    assert [sc.get("raw", "hit", k, HALF) for k in CUTOFFS] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert sc.get("raw", "mrr", None, HALF) == 0.5
    assert [sc.get("raw", "unit_recall", k, HALF) for k in CUTOFFS] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert [sc.get("raw", "coverage", k) for k in CUTOFFS] == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_dense_unit_recall_and_coverage_at_every_cutoff_for_a_two_unit_question(result):
    sc = result.questions[2].dense_scores  # ranking c2, c5, c3, c1, c4, c6: unit one at rank 1, unit two at rank 6
    assert [sc.get("raw", "hit", k, HALF) for k in CUTOFFS] == [1.0] * 5
    assert [sc.get("raw", "unit_recall", k, HALF) for k in CUTOFFS] == [0.5, 0.5, 0.5, 1.0, 1.0]
    assert [sc.get("raw", "coverage", k) for k in CUTOFFS] == [0.5, 0.5, 0.5, 1.0, 1.0]
    assert sc.get("raw", "mrr", None, HALF) == 1.0
    # one passage per chunk, so the distinct metrics equal the raw ones here
    for k in CUTOFFS:
        assert sc.get("distinct", "unit_recall", k, HALF) == sc.get("raw", "unit_recall", k, HALF)
        assert sc.get("distinct", "hit", k, HALF) == sc.get("raw", "hit", k, HALF)


def test_dense_scores_are_identical_to_scoring_the_converted_ranking_directly(result):
    pool = make_pool()
    run_q = result.questions[2]
    direct = baselines.score_ranking(Q3, tuple(pool.by_chunk_id[i] for i in DENSE[Q3.question]))
    assert run_q.dense_scores == direct


def test_the_distinct_truncation_flags_follow_the_six_chunk_ranking(result):
    assert result.dense_observations[0].truncated_at == (10, 20)  # six passages cannot fill k = 10 or 20
    assert result.questions[0].dense_scores.distinct_truncated == (False, False, False, True, True)


def test_the_oracle_is_the_greedy_reference_at_every_cutoff(result):
    q3 = result.questions[2].oracle
    pool = make_pool()
    assert q3.ranking[:2] == (pool.by_chunk_id[2], pool.by_chunk_id[6])  # tie: canonical order puts doc-a first
    assert q3.chunks_needed == 2 and q3.method == baselines.ORACLE_METHOD
    assert [q3.scores.get("raw", "unit_recall", k, HALF) for k in CUTOFFS] == [0.5, 1.0, 1.0, 1.0, 1.0]
    q1 = result.questions[0].oracle
    assert q1.ranking[0] == pool.by_chunk_id[1] and q1.scores.get("raw", "hit", 1, HALF) == 1.0


def test_dense_oracle_and_random_expose_the_same_metric_list(result):
    q = result.questions[2]
    assert q.dense_scores.metric_ids == q.oracle.scores.metric_ids == q.random.per_seed[0].metric_ids
    assert len(result.series) == 76 and len({sp.name for sp in result.series}) == 76


def test_series_kinds_hit_is_binary_only_in_the_binary_declaration(result):
    assert {sp.kind for sp in result.series if "_hit_" in sp.name} == {"binary"}
    assert {sp.kind for sp in result.series if "_hit_" not in sp.name} == {"continuous"}
    assert {sp.kind for sp in result.series_continuous} == {"continuous"}
    assert e.series_name(result.questions[0].dense_scores.metric_ids[0]) == "raw_hit_at_1_any_overlap"


# ---------- C: random ----------

def test_the_default_seeds_are_one_thousand_unique_reproducible_seeds():
    assert len(SEEDS) == len(set(SEEDS)) == 1000 and SEEDS == tuple(range(1000))
    assert len(e.default_random_seeds(1500)) == 1500
    with pytest.raises(ValueError):
        e.default_random_seeds(999)
    with pytest.raises(TypeError):
        e.default_random_seeds(True)


def test_fewer_than_one_thousand_duplicate_negative_or_bool_seeds_are_refused():
    with pytest.raises(ValueError):
        run(seeds=tuple(range(999)))
    with pytest.raises(ValueError):
        run(seeds=tuple(range(999)) + (0,))
    with pytest.raises(ValueError):
        run(seeds=(-1,) + tuple(range(1, 1000)))
    with pytest.raises(TypeError):
        run(seeds=(True,) + tuple(range(2, 1001)))
    with pytest.raises(TypeError):
        run(seeds=range(1000))


def test_the_random_baseline_is_reproducible_and_matches_the_seeded_ranking(result):
    again = run(adjudicated_unreachable={"q4": ["u1"]})
    assert again == result
    pool = make_pool()
    first = baselines.score_ranking(Q1, baselines.random_ranking(pool.ranked, 0))
    assert result.questions[0].random.per_seed[0] == first
    assert result.questions[0].random.seeds == SEEDS


def test_a_ranking_with_every_chunk_gives_exact_random_values_at_k_ten_and_twenty(result):
    for observation in result.random_observations[:3]:  # q1..q3 have a reachable gold unit
        values = dict(observation.values)
        for k in (10, 20):
            assert values[f"raw_hit_at_{k}_half_coverage"] == 1.0
            assert values[f"raw_unit_recall_at_{k}_half_coverage"] == 1.0
            assert values[f"raw_coverage_at_{k}"] == 1.0


def test_the_random_means_estimate_the_hand_computed_probabilities(result):
    q1 = dict(result.random_observations[0].values)   # one gold chunk among six: P(hit@k) = k / 6
    assert abs(q1["raw_hit_at_1_half_coverage"] - 1 / 6) < 0.04
    assert abs(q1["raw_hit_at_3_half_coverage"] - 3 / 6) < 0.05
    assert abs(q1["raw_hit_at_5_half_coverage"] - 5 / 6) < 0.04
    q3 = dict(result.random_observations[2].values)   # two gold chunks: P(hit@1) = 2 / 6
    assert abs(q3["raw_hit_at_1_half_coverage"] - 2 / 6) < 0.05
    q4 = dict(result.random_observations[3].values)   # unreachable under half_coverage: never a hit
    assert q4["raw_hit_at_20_half_coverage"] == 0.0


def test_the_macro_summary_over_seeds_is_ordered_and_covers_every_metric(result):
    macro = result.random_macro
    assert len(macro) == 76 and all(m.n_seeds == 1000 for m in macro)
    assert all(m.p2_5 <= m.p95 <= m.p97_5 for m in macro)
    hit1 = next(m for m in macro if m.metric == baselines.MetricId("raw", "hit", 1, HALF))
    assert 0.1 < hit1.mean < 0.3  # (1/6 + 1/6 + 2/6 + 0) / 4 = 1/6 expected


# ---------- D: oracle validity ----------

def test_the_oracle_validity_checks_pass_for_reachable_questions(result):
    checks = {c.question_id: c for c in result.oracle_validity.checks}
    for qid in ("q1", "q2", "q3"):
        c = checks[qid]
        assert c.ok and c.hit_at_1 == 1.0 and c.unit_recall_at_20 == 1.0
        assert c.unadjudicated == () and c.reachable_units == len(c.unit_ids)
    assert result.oracle_validity.rule == HALF and result.oracle_validity.method == baselines.ORACLE_METHOD


def test_an_unreachable_unit_is_reported_and_blocks_validity_until_adjudicated():
    plain = run()
    c4 = next(c for c in plain.oracle_validity.checks if c.question_id == "q4")
    assert not c4.ok and c4.unadjudicated == ("u1",) and c4.reachable_units == 0
    assert c4.hit_at_1 is None and c4.unit_recall_at_20 is None
    rules = dict(c4.unreachable_by_rule)
    assert rules["any_overlap"] == () and rules[HALF] == ("u1",) and rules["full_coverage"] == ("u1",)
    assert not plain.oracle_validity.ok
    adjudicated = run(adjudicated_unreachable={"q4": ("u1",)})
    c4 = next(c for c in adjudicated.oracle_validity.checks if c.question_id == "q4")
    assert c4.ok and c4.adjudicated == ("u1",) and c4.unadjudicated == ()
    assert adjudicated.oracle_validity.ok


def test_an_adjudication_of_a_reachable_or_unknown_unit_or_question_is_refused():
    for bad in ({"q1": ["u1"]}, {"q4": ["u9"]}, {"nope": ["u1"]}):
        with pytest.raises(s.EvaluationDataError) as info:
            run(adjudicated_unreachable=bad)
        assert {i.code for i in info.value.issues} == {e.ENGINE_ADJUDICATION}


def test_a_failed_hit_at_one_gate_is_reported_not_raised(monkeypatch):
    real = baselines.oracle_baseline

    def bad(q, pool):
        result = real(q, pool)
        worse = baselines.score_ranking(q, tuple(reversed(result.ranking)))
        return dataclasses.replace(result, scores=worse)

    monkeypatch.setattr(baselines, "oracle_baseline", bad)
    out = run(questions=(Q1, Q2, Q3, U1))
    check = out.oracle_validity.checks[0]
    assert check.hit_at_1 == 0.0 and not check.ok and not out.oracle_validity.ok


def test_a_failed_unit_recall_at_twenty_gate_is_reported(monkeypatch):
    real = baselines.oracle_baseline

    def short(q, pool):
        result = real(q, pool)
        one = result.ranking[:1]
        partial = baselines._score(
            dataclasses.replace(q, units=q.units), one, baselines._REACHABLE_IDS[HALF]
        )
        reachable = tuple(
            dataclasses.replace(x, scores=partial) if x.rule == HALF else x for x in result.reachable_only
        )
        return dataclasses.replace(result, reachable_only=reachable)

    monkeypatch.setattr(baselines, "oracle_baseline", short)
    out = run(questions=(Q3, U1))
    check = out.oracle_validity.checks[0]
    assert check.hit_at_1 == 1.0 and check.unit_recall_at_20 == 0.5 and not check.ok


# ---------- E: unanswerable questions ----------

def test_unanswerable_questions_are_listed_apart_and_never_scored(result):
    assert [(u.question_id, u.query_language, u.unanswerable_type) for u in result.unanswerable] == [
        ("du1", "en", "off_topic"), ("du2", "en", "off_topic")]
    everything = (result.dense_observations + result.oracle_observations + result.random_observations)
    assert all(o.question_id not in ("du1", "du2") for o in everything)
    assert [q.question_id for q in result.questions] == ["q1", "q2", "q3", "q4"]
    assert result.dense_report.overall.n == 4


def test_the_retriever_is_not_called_for_unanswerable_questions():
    retriever = Retriever()
    run(retriever=retriever)
    assert U1.question not in retriever.queries and U2.question not in retriever.queries


def test_a_dev_split_with_no_answerable_question_is_refused():
    with pytest.raises(s.EvaluationDataError) as info:
        run(questions=(U1, U2))
    assert {i.code for i in info.value.issues} == {e.ENGINE_NO_ANSWERABLE}


# ---------- F: observations, reports, ordering ----------

def test_observation_values_are_the_scores_in_series_order(result):
    obs = result.dense_observations[2]
    sc = result.questions[2].dense_scores
    assert [n for n, _ in obs.values] == [sp.name for sp in result.series]
    assert [v for _, v in obs.values] == list(sc.values)
    assert dict(obs.values)["raw_unit_recall_at_5_half_coverage"] == 0.5


def test_aggregate_reports_cover_the_four_questions_and_keep_language_slices_suppressed(result):
    for report in (result.dense_report, result.oracle_report, result.random_report):
        assert report.overall.n == 4 and report.resamples == RESAMPLES and report.seed == 7
        by_type = {sl.retrieval_type: sl for sl in report.slices}
        assert by_type["en_to_en"].n == 4 and by_type["en_to_en"].series[0].status == "suppressed"
        assert by_type["hi_to_hi"].n == 0
    hit1 = next(sr for sr in result.dense_report.overall.series if sr.name == "raw_hit_at_1_half_coverage")
    assert hit1.successes == 2 and hit1.rate_or_mean == 0.5 and hit1.status == "ok"
    assert hit1.outcomes == (("q1", 1.0), ("q2", 0.0), ("q3", 1.0), ("q4", 0.0))


def test_the_comparison_is_dense_minus_random_and_only_reported(result):
    cmp = result.dense_vs_random
    hit1 = next(sr for sr in cmp.overall.series if sr.name == "raw_hit_at_1_half_coverage")
    assert hit1.status == "ok" and 0.28 < hit1.mean_difference < 0.39  # 0.5 - (1/6) with a seeded estimate
    assert [o[0] for o in hit1.outcomes] == ["q1", "q2", "q3", "q4"]
    assert not hasattr(result, "passed") and not hasattr(cmp, "passed")  # no threshold is applied


def test_output_order_follows_the_dev_file_order():
    a = run(questions=(Q1, Q2, Q3, U1))
    b = run(questions=(Q3, Q2, Q1, U1))
    assert [q.question_id for q in a.questions] == ["q1", "q2", "q3"]
    assert [q.question_id for q in b.questions] == ["q3", "q2", "q1"]  # dev-file order is kept, not sorted
    assert {o.question_id: o.values for o in a.dense_observations} == {o.question_id: o.values for o in b.dense_observations}


# ---------- G: malformed rankings and arguments ----------

def _codes(error):
    return {i.code for i in error.issues}


def test_a_ranking_with_a_missing_chunk_is_refused():
    table = {**DENSE, Q1.question: [1, 2, 3, 4, 5]}
    with pytest.raises(s.EvaluationDataError) as info:
        run(retriever=Retriever(table))
    assert _codes(info.value) == {e.ENGINE_RANKING_LENGTH}
    assert info.value.issues[0].location == "answerable[0]"


def test_a_ranking_with_a_repeated_chunk_is_refused_with_its_position():
    table = {**DENSE, Q2.question: [1, 4, 4, 3, 5, 6]}
    with pytest.raises(s.EvaluationDataError) as info:
        run(retriever=Retriever(table))
    assert _codes(info.value) == {p.RANK_DUPLICATE}
    assert info.value.issues[0].location == "answerable[1].retrieved[2]"


def test_a_chunk_outside_the_pool_is_refused():
    outsider = dataclasses.replace(C[1], chunk_id=99)

    def retriever(query):
        return [outsider] + [C[i] for i in (2, 3, 4, 5, 6)]

    with pytest.raises(s.EvaluationDataError) as info:
        e.run_pilot(inputs_of(Q1, U1), make_pool(), retriever, seeds=SEEDS, bootstrap_seed=1, resamples=RESAMPLES)
    assert _codes(info.value) == {p.RANK_NOT_IN_POOL}


def test_a_retriever_that_does_not_return_a_list_is_refused():
    for bad in (lambda q: (c for c in C.values()), lambda q: None, lambda q: "abc", lambda q: {1: C[1]}):
        with pytest.raises(s.EvaluationDataError) as info:
            e.run_pilot(inputs_of(Q1, U1), make_pool(), bad, seeds=SEEDS, bootstrap_seed=1, resamples=RESAMPLES)
        assert _codes(info.value) == {e.ENGINE_RETRIEVER_RETURN}


def test_a_retriever_error_propagates_unchanged():
    def boom(query):
        raise RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        e.run_pilot(inputs_of(Q1, U1), make_pool(), boom, seeds=SEEDS, bootstrap_seed=1, resamples=RESAMPLES)


def test_a_pool_that_does_not_cover_the_corpus_documents_is_refused():
    one = s.Manifest("eval-manifest-v1", "c1", "ingested", (mdoc("doc-a", TEXT_A),))
    dmap = p.map_documents(one, [row(ID_A, "doc-a")], owner_id=OWNER, embedding_id=EMB)
    small = p.build_pool(one, dmap, [C[1], C[2], C[3]], {"doc-a": TEXT_A})
    with pytest.raises(s.EvaluationDataError) as info:
        e.run_pilot(inputs_of(Q1, U1), small, Retriever(), seeds=SEEDS, bootstrap_seed=1, resamples=RESAMPLES)
    assert _codes(info.value) == {e.ENGINE_POOL_MISMATCH}


def test_argument_types_are_checked():
    inputs, pool = inputs_of(Q1, U1), make_pool()
    base = {"seeds": SEEDS, "bootstrap_seed": 1, "resamples": RESAMPLES}
    for args in (("x", pool, Retriever()), (inputs, "x", Retriever()), (inputs, pool, "x")):
        with pytest.raises(TypeError):
            e.run_pilot(*args, **base)
    with pytest.raises(TypeError):
        e.run_pilot(inputs, pool, Retriever(), adjudicated_unreachable=["q4"], **base)
    with pytest.raises(TypeError):
        e.run_pilot(inputs, pool, Retriever(), adjudicated_unreachable={"q4": "u1"}, **base)
    with pytest.raises(ValueError):
        e.run_pilot(inputs, pool, Retriever(), seeds=SEEDS, bootstrap_seed=-1, resamples=RESAMPLES)
    with pytest.raises(ValueError):
        e.run_pilot(inputs, pool, Retriever(), seeds=SEEDS, bootstrap_seed=1, resamples=10)


def test_no_finding_holds_query_or_document_text():
    table = {**DENSE, Q1.question: [1, 2, 3, 4, 5]}
    with pytest.raises(s.EvaluationDataError) as info:
        run(retriever=Retriever(table))
    blob = str(info.value) + " ".join(f"{i.code}{i.location}{i.message}" for i in info.value.issues)
    assert Q1.question not in blob and "Alpha" not in blob


# ---------- H: purity ----------

def test_the_engine_imports_no_database_retrieval_or_file_code():
    code = (
        "import sys\nimport app.rag.evaluation.engine\n"
        "bad = [m for m in ('sqlalchemy','app.core.database','app.core.config','app.models','app.rag.retrieval',"
        "'app.rag.embedder','numpy','torch') if m in sys.modules]\nprint('bad=' + ','.join(bad))\n"
    )
    import os
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.strip() == "bad="
