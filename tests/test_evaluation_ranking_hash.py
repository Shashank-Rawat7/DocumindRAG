"""E2.4a: the D7 ranking hash (app/rag/evaluation/ranking_hash.py).

The module is pure: these tests need no database, model or dataset file. The real-run tests reuse the small
engine fixtures; the digest tests use hand-built ``RankedChunk`` identities.

A  the canonical payload and a pinned literal digest
B  sensitivity (ranking order, question order, chunk identity, L)
C  refusals (rankings, chunks, question ids) with fixed messages
D  no database ids, determinism, input immutability
E  the digest of a real run, tied to the inputs and the pool
F  purity

Run focused: pytest tests/test_evaluation_ranking_hash.py --noconftest -p no:cacheprovider
"""
import ast
import dataclasses
import hashlib
import json
import re
import types
from pathlib import Path

import pytest
import test_evaluation_engine as fx

from app.rag.evaluation import engine
from app.rag.evaluation import ranking_hash as rh
from app.rag.evaluation.metrics import RankedChunk

H_A = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"  # sha256("a")
H_B = "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d"  # sha256("b")
H_C = "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6"  # sha256("c")
PINNED = "e8cdfb0d6605b6d7e6f46959cc6195948420766aaa22651e71ddbcc9e1d6c10d"

PINNED_TEXT = (
    '["e2.4-ranking-digest-1",2,[["q1",[["doc-a",0,17,"' + H_A + '"],["doc-b",5,40,"' + H_B + '"]]],'
    '["q2",[["doc-b",5,40,"' + H_B + '"],["doc-a",0,17,"' + H_C + '"]]]]]'
)

A1, B1, B2, A2 = RankedChunk("doc-a", 0, 17, H_A), RankedChunk("doc-b", 5, 40, H_B), RankedChunk("doc-b", 5, 40, H_B), \
    RankedChunk("doc-a", 0, 17, H_C)
ROWS = [("q1", [A1, B1]), ("q2", [B2, A2])]

SEED_RESAMPLES = {
    "seeds": fx.SEEDS, "bootstrap_seed": 7, "resamples": fx.RESAMPLES, "adjudicated_unreachable": {"q4": ["u1"]},
}


def refused(code, fn, *args, **kwargs):
    with pytest.raises(rh.EvaluationDataError) as info:
        fn(*args, **kwargs)
    assert {i.code for i in info.value.issues} == {code}
    assert [i.message for i in info.value.issues] == [rh.MESSAGES[code]]
    return info.value


def forged(chunk, **fields):
    """A RankedChunk whose fields were changed after construction (the constructor would refuse them)."""
    clone = RankedChunk(chunk.corpus_key, chunk.start, chunk.end, chunk.text_sha256)
    for name, value in fields.items():
        object.__setattr__(clone, name, value)
    return clone


# ---------- A: payload and pinned digest ----------

def test_the_payload_is_the_documented_array():
    payload = rh.ranking_payload(2, ROWS)
    assert payload == ["e2.4-ranking-digest-1", 2, [
        ["q1", [["doc-a", 0, 17, H_A], ["doc-b", 5, 40, H_B]]],
        ["q2", [["doc-b", 5, 40, H_B], ["doc-a", 0, 17, H_C]]],
    ]]
    assert rh.DIGEST_FORMAT == "e2.4-ranking-digest-1"


def test_the_canonical_bytes_are_compact_utf8_json_without_a_trailing_newline():
    data = rh.canonical_bytes(rh.ranking_payload(2, ROWS))
    assert data == PINNED_TEXT.encode("utf-8")
    assert not data.endswith(b"\n") and b" " not in data
    assert json.loads(data) == rh.ranking_payload(2, ROWS)


def test_the_digest_is_the_sha256_of_the_independently_written_payload():
    assert rh.ranking_digest(2, ROWS) == hashlib.sha256(PINNED_TEXT.encode("utf-8")).hexdigest()


def test_the_digest_is_pinned_to_a_literal_value():
    assert rh.ranking_digest(2, ROWS) == PINNED
    assert re.fullmatch(r"[0-9a-f]{64}", PINNED)


def test_non_ascii_keys_are_written_as_utf8_not_escaped():
    row = [("q1", [RankedChunk("dökument-ä", 0, 1, H_A)])]
    data = rh.canonical_bytes(rh.ranking_payload(1, row))
    assert "dökument-ä".encode() in data and b"\\u" not in data


def test_lists_and_tuples_give_the_same_digest():
    as_tuples = tuple((qid, tuple(chunks)) for qid, chunks in ROWS)
    assert rh.ranking_digest(2, as_tuples) == rh.ranking_digest(2, ROWS)


# ---------- B: sensitivity ----------

def digest_of(rows, length=2):
    return rh.ranking_digest(length, rows)


def test_the_digest_depends_on_the_rank_order_within_a_question():
    assert digest_of([("q1", [B1, A1]), ROWS[1]]) != PINNED


def test_the_digest_depends_on_the_question_order():
    assert digest_of([ROWS[1], ROWS[0]]) != PINNED


@pytest.mark.parametrize("field,value", [
    ("corpus_key", "doc-x"), ("start", 1), ("end", 18), ("text_sha256", "0" * 64),
])
def test_the_digest_depends_on_every_identity_field_of_a_chunk(field, value):
    changed = dataclasses.replace(A1, **{field: value})
    assert digest_of([("q1", [changed, B1]), ROWS[1]]) != PINNED


def test_the_digest_depends_on_the_question_id():
    assert digest_of([("q9", ROWS[0][1]), ROWS[1]]) != PINNED


def test_the_digest_depends_on_L():
    longer = [("q1", [A1, B1, RankedChunk("doc-a", 20, 30, H_C)]), ("q2", [B2, A2, RankedChunk("doc-b", 40, 50, H_A)])]
    assert rh.ranking_payload(3, longer)[1] == 3 and rh.ranking_payload(2, ROWS)[1] == 2
    assert digest_of(longer, 3) != PINNED
    # the same first two chunks, one more at the end: a different payload even though the prefixes agree
    assert rh.canonical_bytes(rh.ranking_payload(3, longer)) != rh.canonical_bytes(rh.ranking_payload(2, ROWS))


def test_a_different_L_with_the_same_rankings_is_refused_not_silently_hashed():
    refused(rh.RANKING_LENGTH, rh.ranking_digest, 3, ROWS)
    refused(rh.RANKING_LENGTH, rh.ranking_digest, 1, ROWS)


# ---------- C: refusals ----------

def test_no_questions_is_refused():
    for rows in ([], ()):
        refused(rh.RANKING_QUESTIONS, rh.ranking_digest, 2, rows)


@pytest.mark.parametrize("ranking", [[], (), None, "ab", 5, {"a": 1}])
def test_a_missing_or_empty_ranking_is_refused(ranking):
    refused(rh.RANKING_EMPTY, rh.ranking_digest, 2, [("q1", ranking)])


@pytest.mark.parametrize("chunks", [[A1], [A1, B1, A2], [A1, B1, A2, B2]])
def test_a_ranking_with_the_wrong_number_of_chunks_is_refused(chunks):
    refused(rh.RANKING_LENGTH, rh.ranking_digest, 2, [("q1", chunks)])


def test_a_repeated_chunk_identity_is_refused_not_deduplicated():
    refused(rh.RANKING_DUPLICATE, rh.ranking_digest, 2, [("q1", [A1, A1])])
    refused(rh.RANKING_DUPLICATE, rh.ranking_digest, 2, [("q1", [B1, B2])])  # equal identity, distinct objects


def test_the_same_chunk_in_two_different_questions_is_allowed():
    assert rh.ranking_digest(2, [("q1", [A1, B1]), ("q2", [A1, B1])])


class SubChunk(RankedChunk):
    pass


@pytest.mark.parametrize("chunk", [
    types.SimpleNamespace(corpus_key="doc-a", start=0, end=17, text_sha256=H_A),
    ("doc-a", 0, 17, H_A),
    {"corpus_key": "doc-a", "start": 0, "end": 17, "text_sha256": H_A},
    "doc-a",
    None,
    SubChunk("doc-a", 0, 17, H_A),
])
def test_a_chunk_that_is_not_exactly_a_ranked_chunk_is_refused(chunk):
    refused(rh.RANKING_CHUNK, rh.ranking_digest, 2, [("q1", [chunk, B1])])


@pytest.mark.parametrize("fields", [
    {"corpus_key": b"doc-a"}, {"corpus_key": 5}, {"corpus_key": None}, {"corpus_key": ""},
    {"corpus_key": "doc-\ud800"}, {"corpus_key": "\udfffdoc-a"},
    {"start": True}, {"start": 0.0}, {"start": "0"}, {"start": -1}, {"start": 17}, {"start": 18},
    {"end": False}, {"end": 17.0}, {"end": "17"}, {"end": 0},
    {"text_sha256": None}, {"text_sha256": b"x" * 64}, {"text_sha256": H_A.upper()}, {"text_sha256": H_A[:63]},
    {"text_sha256": H_A + "0"}, {"text_sha256": "g" * 64}, {"text_sha256": ""}, {"text_sha256": H_A + "\n"},
])
def test_a_chunk_with_an_invalid_field_is_refused(fields):
    refused(rh.RANKING_CHUNK, rh.ranking_digest, 2, [("q1", [forged(A1, **fields), B1])])


def test_the_first_bad_chunk_names_its_question_and_rank():
    error = refused(rh.RANKING_CHUNK, rh.ranking_digest, 2, [ROWS[0], ("q2", [B2, forged(A2, start=-1)])])
    assert [i.location for i in error.issues] == ["answerable[1].ranking[1]"]


@pytest.mark.parametrize("row", [
    ("q1",), ("q1", [A1, B1], "x"), "q1", None, 5, ["q1", [A1, B1], [A1, B1]],
])
def test_a_malformed_question_row_is_refused(row):
    refused(rh.RANKING_QUESTIONS, rh.ranking_digest, 2, [row])


@pytest.mark.parametrize("question_id", ["", None, 1, b"q1", ("q1",), True])
def test_a_bad_question_id_is_refused(question_id):
    refused(rh.RANKING_QUESTIONS, rh.ranking_digest, 2, [(question_id, [A1, B1])])


@pytest.mark.parametrize("question_id", ["q\ud800", "\udc00", "a\udbffb"])
def test_a_question_id_that_is_not_utf8_encodable_is_refused_with_the_fixed_message(question_id):
    error = refused(rh.RANKING_QUESTIONS, rh.ranking_digest, 2, [(question_id, [A1, B1])])
    assert [i.location for i in error.issues] == ["answerable[0]"]
    assert "ud800" not in repr(error.issues) and "udc00" not in repr(error.issues)


@pytest.mark.parametrize("key", ["doc-\ud800", "\udfff", "x\udc80y"])
def test_a_corpus_key_that_is_not_utf8_encodable_is_refused_with_the_fixed_message(key):
    chunk = forged(A1, corpus_key=key)
    error = refused(rh.RANKING_CHUNK, rh.ranking_digest, 2, [("q1", [chunk, B1])])
    assert [i.location for i in error.issues] == ["answerable[0].ranking[0]"]
    refused(rh.RANKING_CHUNK, rh.ranking_payload, 2, [("q1", [chunk, B1])])
    refused(rh.RANKING_CHUNK, rh.ranking_digest, 2, [("q1", [RankedChunk(key, 0, 17, H_A), B1])])


def test_valid_non_ascii_and_astral_text_is_still_accepted():
    rows = [("fråga-\U0001f600", [RankedChunk("dok-\u00e4\u4e2d\U0001f600", 0, 1, H_A)])]
    assert rh.ranking_digest(1, rows)


def test_a_repeated_question_id_is_refused():
    refused(rh.RANKING_QUESTIONS, rh.ranking_digest, 2, [("q1", [A1, B1]), ("q1", [B1, A1])])


@pytest.mark.parametrize("length", [2.0, "2", None, True, False, b"2"])
def test_a_non_int_length_is_a_type_error(length):
    with pytest.raises(TypeError):
        rh.ranking_digest(length, ROWS)
    with pytest.raises(TypeError):
        rh.ranking_payload(length, ROWS)


@pytest.mark.parametrize("length", [0, -1, -20])
def test_an_integer_length_below_one_is_a_value_error_like_expected_length(length):
    with pytest.raises(ValueError):
        rh.ranking_digest(length, ROWS)
    with pytest.raises(ValueError):
        rh.ranking_payload(length, ROWS)
    with pytest.raises(ValueError):
        rh.expected_length(length)


@pytest.mark.parametrize("rows", ["q1", None, 5, {"q1": [A1, B1]}, b"x"])
def test_rows_must_be_a_list_or_tuple(rows):
    with pytest.raises(TypeError):
        rh.ranking_digest(2, rows)


def test_expected_length_is_min_twenty_and_pool_size():
    assert [rh.expected_length(n) for n in (1, 6, 19, 20, 21, 500)] == [1, 6, 19, 20, 20, 20]
    for bad in (0, -3):
        with pytest.raises(ValueError):
            rh.expected_length(bad)
    for bad in (True, 2.0, "6", None):
        with pytest.raises(TypeError):
            rh.expected_length(bad)


def test_errors_are_fixed_messages_with_schema_locations_and_no_data():
    secret = "secret-corpus-key"
    chunk = RankedChunk(secret, 123456, 123999, H_A)
    error = refused(rh.RANKING_LENGTH, rh.ranking_digest, 2, [("secret-question-id", [chunk])])
    text = repr(error.issues) + str(error)
    for leaked in (secret, "secret-question-id", "123456", H_A):
        assert leaked not in text
    assert all(re.fullmatch(r"[A-Za-z0-9_.\[\]]+", i.location) and i.severity == "error" for i in error.issues)
    assert set(rh.MESSAGES) == {rh.RANKING_QUESTIONS, rh.RANKING_EMPTY, rh.RANKING_LENGTH, rh.RANKING_CHUNK,
                                rh.RANKING_DUPLICATE}
    assert all(isinstance(text, str) and text and not re.search(r"\d", text.replace("min(20", ""))
               for text in rh.MESSAGES.values())


# ---------- D: no database ids, determinism ----------

def test_the_payload_holds_only_the_four_identity_fields_and_no_database_id():
    assert [f.name for f in dataclasses.fields(RankedChunk)] == ["corpus_key", "start", "end", "text_sha256"]
    payload = rh.ranking_payload(2, ROWS)
    for _, identities in payload[2]:
        assert all(len(identity) == 4 for identity in identities)
    text = rh.canonical_bytes(payload).decode()
    for name in ("chunk_id", "document_id", "owner", "score", "distance", "span_sha256", "source"):
        assert name not in text


def test_identical_inputs_give_identical_digests_and_the_inputs_are_not_changed():
    rows = [(qid, list(chunks)) for qid, chunks in ROWS]
    before = [(qid, list(chunks)) for qid, chunks in rows]
    first, second = rh.ranking_digest(2, rows), rh.ranking_digest(2, rows)
    assert first == second == PINNED
    assert rows == before  # not sorted, not deduplicated, not repaired


def test_a_rising_or_arbitrary_order_is_kept_as_given():
    ordered = rh.ranking_payload(2, [("q1", [A1, B1])])[2][0][1]
    swapped = rh.ranking_payload(2, [("q1", [B1, A1])])[2][0][1]
    assert ordered == [["doc-a", 0, 17, H_A], ["doc-b", 5, 40, H_B]]
    assert swapped == list(reversed(ordered))


# ---------- E: a real run ----------

def retriever_with(table=None):
    return fx.Retriever(table)


@pytest.fixture(scope="module")
def inputs():
    return fx.inputs_of(fx.Q1, fx.Q2, fx.Q3, fx.Q4, fx.U1, fx.U2)


@pytest.fixture(scope="module")
def pool():
    return fx.make_pool()


@pytest.fixture(scope="module")
def run(inputs, pool):
    return engine.run_pilot(inputs, pool, retriever_with(), **SEED_RESAMPLES)


def test_the_run_digest_hashes_the_retained_dense_rankings_in_dev_order(run, inputs, pool):
    rows = [[q.question_id, [[c.corpus_key, c.start, c.end, c.text_sha256] for c in q.dense]] for q in run.questions]
    text = json.dumps(["e2.4-ranking-digest-1", 6, rows], separators=(",", ":"), ensure_ascii=False)
    assert rh.ranking_sha256(run, inputs, pool) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert [row[0] for row in rows] == ["q1", "q2", "q3", "q4"]  # answerable only: the unanswerable are not in it
    assert "du1" not in text and "du2" not in text


def test_two_independent_runs_with_fresh_retrievers_have_the_same_digest(run, inputs, pool):
    second = engine.run_pilot(inputs, pool, retriever_with(), **SEED_RESAMPLES)
    assert second is not run
    assert rh.ranking_sha256(second, inputs, pool) == rh.ranking_sha256(run, inputs, pool)


def test_a_different_dense_ranking_changes_the_digest(run, inputs, pool):
    table = {**fx.DENSE, fx.Q3.question: [5, 2, 3, 1, 4, 6]}  # first two chunks of Q3 swapped
    other = engine.run_pilot(inputs, pool, retriever_with(table), **SEED_RESAMPLES)
    assert rh.ranking_sha256(other, inputs, pool) != rh.ranking_sha256(run, inputs, pool)


def test_the_digest_ignores_scores_and_does_not_need_them(run, inputs, pool):
    class Scored(fx.Retriever):
        def __call__(self, query):
            return [dataclasses.replace(c, score=0.25) for c in super().__call__(query)]

    other = engine.run_pilot(inputs, pool, Scored(), **SEED_RESAMPLES)
    assert rh.ranking_sha256(other, inputs, pool) == rh.ranking_sha256(run, inputs, pool)


def test_the_run_digest_equals_the_digest_of_its_own_rows(run, inputs, pool):
    rows = [(q.question_id, q.dense) for q in run.questions]
    assert rh.ranking_sha256(run, inputs, pool) == rh.ranking_digest(6, rows)


def test_a_run_whose_questions_differ_from_the_dev_questions_is_refused(run, pool):
    fewer = fx.inputs_of(fx.Q1, fx.Q2, fx.Q3, fx.U1)
    refused(rh.RANKING_QUESTIONS, rh.ranking_sha256, run, fewer, pool)
    reordered = fx.inputs_of(fx.Q2, fx.Q1, fx.Q3, fx.Q4)
    refused(rh.RANKING_QUESTIONS, rh.ranking_sha256, run, reordered, pool)


def test_a_run_with_another_pool_size_is_refused(run, inputs, pool):
    shorter = dataclasses.replace(run, pool_size=run.pool_size + 1)
    refused(rh.RANKING_LENGTH, rh.ranking_sha256, shorter, inputs, pool)


def test_a_retained_ranking_of_the_wrong_length_is_refused(run, inputs, pool):
    first, *rest = run.questions
    cut = dataclasses.replace(first, dense=first.dense[:5])
    refused(rh.RANKING_LENGTH, rh.ranking_sha256, dataclasses.replace(run, questions=(cut, *rest)), inputs, pool)


def test_a_retained_ranking_with_a_repeated_chunk_is_refused(run, inputs, pool):
    first, *rest = run.questions
    repeated = dataclasses.replace(first, dense=(first.dense[0], *first.dense[:5]))
    broken = dataclasses.replace(run, questions=(repeated, *rest))
    refused(rh.RANKING_DUPLICATE, rh.ranking_sha256, broken, inputs, pool)


@pytest.mark.parametrize("position,value", [(0, "run"), (1, "inputs"), (2, "pool")])
def test_wrongly_typed_arguments_are_a_type_error(run, inputs, pool, position, value):
    args = [run, inputs, pool]
    args[position] = value
    with pytest.raises(TypeError):
        rh.ranking_sha256(*args)


# ---------- F: purity ----------

def _tree():
    return ast.parse(Path(rh.__file__).read_text(encoding="utf-8"))


def test_the_module_imports_only_the_standard_library_and_the_existing_evaluation_modules():
    allowed = {
        "__future__", "hashlib", "json", "collections.abc", "typing", "app.rag.evaluation",
        "app.rag.evaluation.metrics", "app.rag.evaluation.pool", "app.rag.evaluation.runner_inputs",
        "app.rag.evaluation.schema",
    }
    found = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            found.add(node.module)
    assert found <= allowed
    assert not {"os", "pathlib", "io", "sqlalchemy", "random", "app.rag.retrieval", "app.rag.evaluation.report",
                "app.rag.evaluation.score_diagnostics", "app.rag.evaluation.freeze_binding"} & found


def test_the_module_does_no_file_access_sorts_nothing_and_never_names_a_private_file():
    source = Path(rh.__file__).read_text(encoding="utf-8")
    tree = _tree()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & {"open", "Path", "sorted", "print"}
    assert not attributes & {"read_bytes", "read_text", "open", "stat", "lstat", "exists", "iterdir", "glob",
                             "resolve", "realpath", "listdir", "scandir", "sort", "reverse"}
    for private in ("test.jsonl", "audit.jsonl", "eval/dataset", "dev.jsonl"):
        assert private not in source
    json_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "dumps"]
    keywords = {kw.arg: ast.literal_eval(kw.value) for kw in json_calls[0].keywords if kw.arg}
    assert keywords == {"separators": (",", ":"), "ensure_ascii": False, "allow_nan": False}
