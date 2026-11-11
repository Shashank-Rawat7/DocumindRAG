"""E1.3.2: Representative and representatives() (distinct passages of a ranking).

A  Representative
B  connected components: transitivity, touching, gaps, duplicates, documents
C  representative choice and ordering; original rank versus compressed position
D  the first 20 chunks only: a bridge at rank 20 versus rank 21
E  invalid input
F  purity and fixed messages
G  seeded randomized comparison with an independent brute-force implementation
H  imports: no cycle in either order, no heavy dependencies
I  E1.3.3: is_truncated (P < k, P == k, P > k, empty, longer than 20, bridges, validation, purity)
"""
import dataclasses
import random
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import metrics as mt
from app.rag.evaluation import passages as ps
from app.rag.evaluation.intervals import collides

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64
CANARY = "SECRET-CANARY-7731"


class StrSubclass(str):
    pass


def c(key, start, end, sha=SHA):
    return mt.RankedChunk(key, start, end, sha)


def summary(result):
    """(original_rank, corpus_key, start, end) for each representative."""
    return [(r.original_rank, r.chunk.corpus_key, r.chunk.start, r.chunk.end) for r in result]


def ranks(result):
    return [r.original_rank for r in result]


def fillers(first_rank, last_rank):
    """Chunks of an unrelated document, one per rank, none touching another."""
    return [c("doc-z", 100 * rank, 100 * rank + 10) for rank in range(first_rank, last_rank + 1)]


# ---------- A. Representative ----------

def test_representative_has_exactly_two_fields_and_is_frozen_and_slotted():
    assert [f.name for f in dataclasses.fields(ps.Representative)] == ["original_rank", "chunk"]
    value = ps.Representative(1, c("doc-a", 0, 5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.original_rank = 2
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize("rank", [True, False, 1.0, "1", None])
def test_representative_rank_of_the_wrong_type_is_a_type_error(rank):
    with pytest.raises(TypeError):
        ps.Representative(rank, c("doc-a", 0, 5))


@pytest.mark.parametrize("rank", [1, 2, 19, 20])
def test_representative_ranks_from_1_to_max_rank_are_accepted(rank):
    assert ps.Representative(rank, c("doc-a", 0, 5)).original_rank == rank
    assert mt.MAX_RANK == 20


@pytest.mark.parametrize("rank", [0, -1, -20, 21, 22, 100, 2**31])
def test_representative_rank_outside_1_to_max_rank_is_a_value_error(rank):
    with pytest.raises(ValueError):
        ps.Representative(rank, c("doc-a", 0, 5))


def test_type_validation_takes_precedence_over_range_validation():
    with pytest.raises(TypeError):
        ps.Representative(21, None)
    with pytest.raises(TypeError):
        ps.Representative(0, "chunk")
    with pytest.raises(TypeError):
        ps.Representative(True, None)
    with pytest.raises(TypeError):
        ps.Representative(1.5, c("doc-a", 0, 5))
    with pytest.raises(ValueError):
        ps.Representative(21, c("doc-a", 0, 5))


def test_every_rank_representatives_returns_satisfies_the_bound():
    for item in ps.representatives(fillers(1, 60)):
        assert 1 <= item.original_rank <= mt.MAX_RANK


@pytest.mark.parametrize("chunk", [None, ("doc-a", 0, 5, SHA), "chunk"])
def test_representative_chunk_must_be_a_ranked_chunk(chunk):
    with pytest.raises(TypeError):
        ps.Representative(1, chunk)


# ---------- B. connected components ----------

def test_an_empty_ranking_has_no_representatives():
    assert ps.representatives([]) == ()
    assert ps.representatives(()) == ()


def test_one_chunk_is_its_own_representative():
    assert summary(ps.representatives([c("doc-a", 0, 5)])) == [(1, "doc-a", 0, 5)]


def test_separate_chunks_are_all_representatives_in_rank_order():
    ranking = [c("doc-a", 100, 110), c("doc-a", 0, 10), c("doc-a", 50, 60)]
    assert ranks(ps.representatives(ranking)) == [1, 2, 3]


def test_overlapping_chunks_form_one_passage():
    assert ranks(ps.representatives([c("doc-a", 0, 10), c("doc-a", 5, 15)])) == [1]


def test_touching_chunks_are_one_passage_in_either_order():
    assert ranks(ps.representatives([c("doc-a", 0, 10), c("doc-a", 10, 20)])) == [1]
    assert ranks(ps.representatives([c("doc-a", 10, 20), c("doc-a", 0, 10)])) == [1]


def test_a_gap_of_one_position_keeps_passages_apart():
    assert ranks(ps.representatives([c("doc-a", 0, 10), c("doc-a", 11, 20)])) == [1, 2]
    assert ranks(ps.representatives([c("doc-a", 11, 20), c("doc-a", 0, 10)])) == [1, 2]


def test_connectivity_is_transitive_through_overlaps():
    # (0,10) and (16,26) do not meet, but both meet (8,18).
    ranking = [c("doc-a", 0, 10), c("doc-a", 16, 26), c("doc-a", 8, 18)]
    assert ranks(ps.representatives(ranking)) == [1]


def test_connectivity_is_transitive_through_touching():
    ranking = [c("doc-a", 20, 30), c("doc-a", 0, 10), c("doc-a", 10, 20)]
    assert ranks(ps.representatives(ranking)) == [1]


def test_a_long_chunk_joins_far_apart_chunks():
    ranking = [c("doc-a", 0, 5), c("doc-a", 1000, 1005), c("doc-a", 2000, 2005), c("doc-a", 3, 2003)]
    assert ranks(ps.representatives(ranking)) == [1]


def test_a_chain_of_five_is_one_passage_whatever_the_rank_order():
    pieces = [(0, 10), (9, 20), (19, 30), (29, 40), (39, 50)]
    for order in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        ranking = [c("doc-a", *pieces[i]) for i in order]
        assert ranks(ps.representatives(ranking)) == [1]


def test_a_chunk_inside_another_is_one_passage():
    assert ranks(ps.representatives([c("doc-a", 40, 50), c("doc-a", 0, 100)])) == [1]


def test_duplicate_chunks_collapse_to_the_first():
    assert summary(ps.representatives([c("doc-a", 0, 10)] * 3)) == [(1, "doc-a", 0, 10)]
    assert ranks(ps.representatives([c("doc-a", 0, 10, "1" * 64), c("doc-a", 0, 10, "2" * 64)])) == [1]


def test_different_documents_never_join_even_with_identical_offsets():
    assert ranks(ps.representatives([c("doc-a", 0, 10), c("doc-b", 0, 10)])) == [1, 2]


def test_interleaved_documents_are_grouped_per_document():
    ranking = [
        c("doc-a", 0, 10),    # 1: passage A1 {1, 3}
        c("doc-b", 0, 10),    # 2: passage B1 {2}
        c("doc-a", 5, 15),    # 3: joins A1
        c("doc-b", 20, 30),   # 4: passage B2
        c("doc-a", 100, 110),  # 5: passage A2
        c("doc-b", 8, 22),    # 6: joins B1 and B2 (bridge) -> B1 {2, 4, 6}
    ]
    assert ranks(ps.representatives(ranking)) == [1, 2, 5]


# ---------- C. representative choice and ordering ----------

def test_the_representative_is_the_lowest_ranked_member_not_the_leftmost_or_largest():
    ranking = [c("doc-a", 50, 60), c("doc-a", 0, 55), c("doc-a", 58, 200)]
    result = ps.representatives(ranking)
    assert summary(result) == [(1, "doc-a", 50, 60)]


def test_representatives_keep_the_original_relative_order_across_documents():
    ranking = [c("doc-z", 0, 5), c("doc-a", 0, 5), c("doc-m", 0, 5), c("doc-a", 100, 105)]
    assert summary(ps.representatives(ranking)) == [
        (1, "doc-z", 0, 5), (2, "doc-a", 0, 5), (3, "doc-m", 0, 5), (4, "doc-a", 100, 105)
    ]


def test_original_rank_and_compressed_position_differ_after_merging():
    ranking = [c("doc-b", 0, 100), c("doc-b", 50, 150), c("doc-b", 100, 200), c("doc-a", 1000, 1100)]
    result = ps.representatives(ranking)
    assert [(position, item.original_rank) for position, item in enumerate(result, start=1)] == [(1, 1), (2, 4)]


def test_the_representative_chunk_is_the_input_object():
    ranking = [c("doc-a", 0, 10), c("doc-a", 5, 15)]
    assert ps.representatives(ranking)[0].chunk is ranking[0]


# ---------- D. the first 20 chunks only ----------

def bridge_ranking(total):
    """Rank 1 (0,10) and rank 2 (20,30) are separate passages; a chunk (10,21) bridges them."""
    ranking = [c("doc-a", 0, 10), c("doc-a", 20, 30)] + fillers(3, total - 1)
    return ranking + [c("doc-a", 10, 21)]


def test_a_bridge_at_rank_20_merges_earlier_passages():
    ranking = bridge_ranking(20)
    assert len(ranking) == 20
    result = ps.representatives(ranking)
    assert 2 not in ranks(result) and 20 not in ranks(result)
    assert ranks(result) == [1] + list(range(3, 20))


def test_a_bridge_at_rank_21_is_never_looked_at():
    ranking = bridge_ranking(21)
    assert len(ranking) == 21
    result = ps.representatives(ranking)
    assert ranks(result) == list(range(1, 21))  # rank 2 stays a passage of its own


def test_without_the_bridge_the_two_earlier_passages_stay_apart():
    assert ranks(ps.representatives(bridge_ranking(20)[:-1]))[:2] == [1, 2]


def test_only_the_first_20_chunks_are_inspected():
    ranking = fillers(1, 20) + ["not a chunk", None]
    assert ranks(ps.representatives(ranking)) == list(range(1, 21))


def test_the_result_never_has_more_than_20_representatives():
    assert len(ps.representatives(fillers(1, 60))) == 20


def test_a_ranking_of_exactly_20_and_a_short_ranking_work():
    assert len(ps.representatives(fillers(1, 20))) == 20
    assert len(ps.representatives(fillers(1, 3))) == 3


# ---------- E. invalid input ----------

@pytest.mark.parametrize("ranking", [None, "", "abc", b"", {}, {1: c("doc-a", 0, 5)}, set(), frozenset(), 5,
                                     iter(()), (x for x in ()), range(0)])
def test_a_ranking_must_be_a_list_or_tuple(ranking):
    with pytest.raises(TypeError):
        ps.representatives(ranking)


@pytest.mark.parametrize("bad", [None, "chunk", ("doc-a", 0, 5, SHA), {"corpus_key": "doc-a"}, 7])
def test_an_item_among_the_first_20_must_be_a_ranked_chunk(bad):
    with pytest.raises(TypeError):
        ps.representatives([c("doc-a", 0, 5), bad])
    with pytest.raises(TypeError):
        ps.representatives(fillers(1, 19) + [bad])


# ---------- F. purity and fixed messages ----------

def test_the_input_is_not_modified_or_reordered():
    ranking = [c("doc-b", 50, 60), c("doc-a", 0, 10), c("doc-b", 55, 70), c("doc-a", 10, 20)]
    snapshot = list(ranking)
    ps.representatives(ranking)
    assert ranking == snapshot and all(a is b for a, b in zip(ranking, snapshot))
    as_tuple = tuple(ranking)
    ps.representatives(as_tuple)
    assert as_tuple == tuple(snapshot)


def test_the_result_is_a_tuple_of_representatives_and_is_deterministic():
    ranking = [c("doc-b", 50, 60), c("doc-a", 0, 10), c("doc-b", 55, 70), c("doc-a", 10, 20)]
    first = ps.representatives(ranking)
    assert type(first) is tuple and all(type(item) is ps.Representative for item in first)
    assert ps.representatives(ranking) == first
    assert ps.representatives(tuple(ranking)) == first


def test_error_messages_never_contain_input_content():
    calls = [
        lambda: ps.representatives(CANARY),
        lambda: ps.representatives([CANARY]),
        lambda: ps.representatives([c(CANARY, 0, 5), CANARY]),
        lambda: ps.Representative(CANARY, c("doc-a", 0, 5)),
        lambda: ps.Representative(1, CANARY),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        assert CANARY not in str(caught.value)


# ---------- G. randomized comparison with an independent implementation ----------

def brute_force_ranks(ranking):
    """Union-find over every pair of the first 20 chunks, using intervals.collides directly."""
    chunks = ranking[:20]
    parent = list(range(len(chunks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if chunks[i].corpus_key == chunks[j].corpus_key and collides(
                (chunks[i].start, chunks[i].end), (chunks[j].start, chunks[j].end)
            ):
                parent[find(i)] = find(j)
    best = {}
    for index in range(len(chunks)):
        root = find(index)
        best[root] = min(best.get(root, index + 1), index + 1)
    return sorted(best.values())


def random_ranking(rng):
    keys = ["doc-a", "doc-b", "doc-c"][: rng.randint(1, 3)]
    ranking = []
    for _ in range(rng.randint(0, 26)):
        start = rng.randint(0, 60)
        ranking.append(c(rng.choice(keys), start, start + rng.randint(1, 12), rng.choice(["1", "2"]) * 64))
    return ranking


def test_representatives_match_an_independent_brute_force_on_random_rankings():
    rng = random.Random(20261111)
    multi = 0
    for _ in range(1500):
        ranking = random_ranking(rng)
        assert ranks(ps.representatives(ranking)) == brute_force_ranks(ranking)
        multi += len(brute_force_ranks(ranking)) < len(ranking[:20])
    assert multi > 500  # the generator really produces merged passages


def test_the_brute_force_reference_itself_agrees_with_a_hand_worked_case():
    ranking = [c("doc-a", 0, 10), c("doc-a", 16, 26), c("doc-a", 8, 18), c("doc-b", 0, 10)]
    assert brute_force_ranks(ranking) == [1, 4]


# ---------- H. imports ----------

@pytest.mark.parametrize("first", ["passages", "metrics"])
def test_the_modules_import_in_either_order_without_a_cycle_or_heavy_dependencies(first):
    second = "metrics" if first == "passages" else "passages"
    code = (
        "import sys\n"
        f"import app.rag.evaluation.{first}\n"
        f"import app.rag.evaluation.{second}\n"
        "from app.rag.evaluation import metrics, passages\n"
        "assert passages.representatives([]) == ()\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "later = ('app.rag.evaluation.statistics', 'app.rag.retrieval', 'app.models')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy or m in later))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_importing_metrics_alone_does_not_load_passages():
    code = "import sys, app.rag.evaluation.metrics\nprint('app.rag.evaluation.passages' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# ---------- I. is_truncated ----------

def test_is_truncated_is_true_exactly_when_there_are_fewer_than_k_representatives():
    ranking = fillers(1, 5)  # P = 5
    assert [ps.is_truncated(ranking, k) for k in (1, 3, 4, 5, 6, 10, 20)] == [
        False, False, False, False, True, True, True]


@pytest.mark.parametrize(
    "ranking, p",
    [
        pytest.param([c("doc-a", 0, 10)] * 5, 1, id="duplicates-collapse-to-one"),
        pytest.param([c("doc-a", 0, 10), c("doc-a", 10, 20), c("doc-a", 20, 30)], 1, id="touching-chain"),
        pytest.param([c("doc-a", 0, 10), c("doc-a", 11, 20), c("doc-b", 0, 10)], 3, id="gap-and-documents"),
        pytest.param(fillers(1, 20), 20, id="twenty-separate"),
    ],
)
def test_is_truncated_compares_the_representative_count_with_k(ranking, p):
    assert len(ps.representatives(ranking)) == p
    for k in range(1, 21):
        assert ps.is_truncated(ranking, k) is (p < k)


def test_p_less_than_p_equal_to_and_p_greater_than_k():
    ranking = [c("doc-a", 0, 10), c("doc-a", 5, 15), c("doc-b", 0, 10), c("doc-c", 0, 10)]  # P = 3
    assert ps.is_truncated(ranking, 4) is True    # P < k
    assert ps.is_truncated(ranking, 3) is False   # P == k
    assert ps.is_truncated(ranking, 2) is False   # P > k


def test_an_empty_ranking_is_truncated_at_every_k():
    for ranking in ([], ()):
        assert all(ps.is_truncated(ranking, k) is True for k in range(1, 21))


def test_a_ranking_longer_than_20_counts_representatives_of_the_first_20_chunks_only():
    assert ps.is_truncated(fillers(1, 60), 20) is False
    collapsed = [c("doc-a", 0, 10)] * 20 + fillers(21, 40)  # one passage; the rest are never looked at
    assert ps.is_truncated(collapsed, 1) is False and ps.is_truncated(collapsed, 2) is True
    assert ps.is_truncated([c("doc-a", 0, 10)] * 19 + fillers(20, 25), 2) is False  # chunk 20 is a second passage


def test_a_bridge_at_rank_20_makes_a_full_length_ranking_truncated_and_rank_21_does_not():
    at_20 = bridge_ranking(20)
    at_21 = bridge_ranking(21)
    assert len(at_20) == 20 and len(ps.representatives(at_20)) == 18
    assert [ps.is_truncated(at_20, k) for k in (18, 19, 20)] == [False, True, True]
    assert len(ps.representatives(at_21)) == 20
    assert ps.is_truncated(at_21, 20) is False


def test_the_flag_is_consistent_with_the_number_of_representatives_on_random_rankings():
    rng = random.Random(424242)
    truncated = full = 0
    for _ in range(800):
        ranking = random_ranking(rng)
        p = len(brute_force_ranks(ranking))
        for k in (1, 3, 5, 10, 20):
            result = ps.is_truncated(ranking, k)
            assert result is (p < k)
            truncated += result
            full += not result
    assert truncated > 500 and full > 500


def test_is_truncated_returns_a_plain_bool():
    assert type(ps.is_truncated([], 1)) is bool
    assert type(ps.is_truncated(fillers(1, 3), 3)) is bool


@pytest.mark.parametrize("k", [True, False, 1.0, "5", None, b"5"])
def test_is_truncated_k_of_the_wrong_type_is_a_type_error(k):
    with pytest.raises(TypeError):
        ps.is_truncated(fillers(1, 3), k)


@pytest.mark.parametrize("k", [0, -1, 21, 100])
def test_is_truncated_k_outside_1_to_20_is_a_value_error(k):
    with pytest.raises(ValueError):
        ps.is_truncated(fillers(1, 3), k)


@pytest.mark.parametrize("ranking", [None, "", "abc", b"", {}, set(), 5, iter(()), (x for x in ()), range(0)])
def test_is_truncated_ranking_must_be_a_list_or_tuple(ranking):
    with pytest.raises(TypeError):
        ps.is_truncated(ranking, 5)


def test_is_truncated_checks_chunks_among_the_first_20_only_and_k_before_the_ranking():
    with pytest.raises(TypeError):
        ps.is_truncated([c("doc-a", 0, 5), None], 5)
    with pytest.raises(TypeError):
        ps.is_truncated(fillers(1, 19) + ["x"], 5)
    assert ps.is_truncated(fillers(1, 20) + ["x"], 20) is False
    with pytest.raises(ValueError):
        ps.is_truncated(None, 0)   # k is checked first, as in the distinct metrics
    with pytest.raises(TypeError):
        ps.is_truncated(None, "5")
    with pytest.raises(TypeError):
        ps.is_truncated(None, 5)


def test_is_truncated_does_not_modify_its_input_and_is_deterministic():
    ranking = [c("doc-b", 50, 60), c("doc-a", 0, 10), c("doc-b", 55, 70)]
    snapshot = list(ranking)
    first = [ps.is_truncated(ranking, k) for k in (1, 2, 3)]
    assert [ps.is_truncated(ranking, k) for k in (1, 2, 3)] == first == [False, False, True]
    assert ranking == snapshot and all(a is b for a, b in zip(ranking, snapshot))
    assert [ps.is_truncated(tuple(ranking), k) for k in (1, 2, 3)] == first


def test_is_truncated_error_messages_never_contain_input_content():
    calls = [
        lambda: ps.is_truncated(CANARY, 5),
        lambda: ps.is_truncated([CANARY], 5),
        lambda: ps.is_truncated([], CANARY),
        lambda: ps.is_truncated([c(CANARY, 0, 5), CANARY], 5),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)) as caught:
            call()
        assert CANARY not in str(caught.value)


@pytest.mark.parametrize("first", ["passages", "metrics"])
def test_is_truncated_imports_in_either_order_without_heavy_dependencies(first):
    second = "metrics" if first == "passages" else "passages"
    code = (
        "import sys\n"
        f"import app.rag.evaluation.{first}\n"
        f"import app.rag.evaluation.{second}\n"
        "from app.rag.evaluation import passages\n"
        "assert passages.is_truncated([], 1) is True\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
