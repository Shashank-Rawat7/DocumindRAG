import dataclasses

import numpy as np
import pytest

import app.rag.embedder as embedder_module
from app.rag import retrieval as r
from app.rag.chunking import sha256_text
from app.rag.embedder import EMBEDDING_DIM, NORM_TOLERANCE, _check_output

FIELDS = [
    "chunk_id",
    "document_id",
    "chunk_index",
    "start_offset",
    "end_offset",
    "page_start",
    "page_end",
    "text",
    "span_sha256",
    "score",
    "source_version",
    "source_sha256",
]


def unit(index: int = 0) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_chunk(**overrides) -> r.RetrievedChunk:
    values = {
    "chunk_id": 1,
    "document_id": 2,
    "chunk_index": 0,
    "start_offset": 0,
    "end_offset": 5,
    "page_start": None,
    "page_end": None,
    "text": "hello",
    "span_sha256": sha256_text("hello"),
    "score": 0.5,
    "source_version": 1,
    "source_sha256": "a" * 64,
}
    values.update(overrides)
    return r.RetrievedChunk(**values)


# ---------- constants, errors, type ----------

def test_limits_are_the_approved_values():
    assert (r.DEFAULT_TOP_K, r.MAX_TOP_K, r.MAX_DOCUMENT_IDS) == (5, 20, 100)


def test_error_hierarchy():
    assert issubclass(r.RetrievalRequestError, r.RetrievalError)
    assert issubclass(r.EmptyQueryError, r.RetrievalRequestError)
    assert issubclass(r.RetrievalIntegrityError, r.RetrievalError)
    assert not issubclass(r.RetrievalIntegrityError, r.RetrievalRequestError)


def test_retrieved_chunk_has_the_approved_fields_in_order():
    assert [f.name for f in dataclasses.fields(r.RetrievedChunk)] == FIELDS


def test_retrieved_chunk_is_immutable_and_slotted():
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.score = 0.9
    assert not hasattr(chunk, "__dict__")
    assert chunk == make_chunk()


def test_the_norm_tolerance_is_the_embedders_own():
    assert r.NORM_TOLERANCE is embedder_module.NORM_TOLERANCE


# ---------- owner_id / top_k / embedding_id ----------

def test_owner_id_accepts_a_plain_int():
    assert r.validate_owner_id(7) == 7


@pytest.mark.parametrize("bad", [True, "1", 1.0, None, np.int64(1)])
def test_owner_id_rejects_other_types(bad):
    with pytest.raises(r.RetrievalRequestError):
        r.validate_owner_id(bad)


@pytest.mark.parametrize("good", [1, 5, 20])
def test_top_k_accepts_the_valid_range(good):
    assert r.validate_top_k(good) == good


@pytest.mark.parametrize("bad", [0, -1, 21, 1000, True, 5.0, "5", None, np.int64(5)])
def test_top_k_rejects_instead_of_clamping(bad):
    with pytest.raises(r.RetrievalRequestError):
        r.validate_top_k(bad)


def test_embedding_id():
    value = "intfloat/multilingual-e5-small@" + "0" * 40
    assert r.validate_embedding_id(value) == value
    for bad in ["", None, 5, b"x"]:
        with pytest.raises(r.RetrievalRequestError):
            r.validate_embedding_id(bad)


# ---------- query_vector ----------

def test_a_unit_vector_is_accepted_and_returned_unchanged():
    vector = unit(3)
    assert r.validate_query_vector(vector) is vector


def test_a_vector_that_passes_the_embedders_own_check_passes_here():
    rng = np.random.default_rng(0)
    for _ in range(50):
        raw = rng.normal(size=(1, EMBEDDING_DIM)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        _check_output(raw, 1)                       # the embedder accepts it...
        r.validate_query_vector(raw[0])             # ...so must we


def test_the_norm_boundary_follows_the_shared_tolerance():
    inside = unit(); inside[0] = 1.0 + NORM_TOLERANCE / 2
    outside = unit(); outside[0] = 1.0 + NORM_TOLERANCE * 2
    r.validate_query_vector(inside)
    with pytest.raises(r.RetrievalRequestError):
        r.validate_query_vector(outside)


@pytest.mark.parametrize(
    "bad",
    [
        unit().astype(np.float64),
        np.zeros(EMBEDDING_DIM - 1, dtype=np.float32),
        unit().reshape(1, EMBEDDING_DIM),
        unit().reshape(EMBEDDING_DIM, 1),
        np.zeros(EMBEDDING_DIM, dtype=np.float32),
        unit() * 1.01,
        unit().tolist(),
        None,
    ],
    ids=["float64", "short", "row2d", "column2d", "zero", "not-unit", "list", "none"],
)
def test_query_vector_rejects_invalid_values(bad):
    with pytest.raises(r.RetrievalRequestError):
        r.validate_query_vector(bad)


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_query_vector_rejects_non_finite_values(poison):
    vector = unit()
    vector[5] = poison
    with pytest.raises(r.RetrievalRequestError):
        r.validate_query_vector(vector)


# ---------- document_ids ----------

def test_document_ids_none_means_no_filter_and_empty_means_nothing():
    assert r.normalize_document_ids(None) is None
    assert r.normalize_document_ids([]) == ()
    assert r.normalize_document_ids(()) == ()


def test_document_ids_are_deduplicated_in_first_seen_order():
    assert r.normalize_document_ids([3, 1, 3, 2, 1]) == (3, 1, 2)
    assert r.normalize_document_ids((9,)) == (9,)


@pytest.mark.parametrize(
    "bad",
    ["12", b"12", {1, 2}, 5, [True], [1.0], ["1"], [None], [np.int64(1)], np.array([1, 2])],
)
def test_document_ids_reject_wrong_shapes_and_types(bad):
    with pytest.raises(r.RetrievalRequestError):
        r.normalize_document_ids(bad)


def test_document_ids_cap_applies_to_the_supplied_length():
    assert len(r.normalize_document_ids(list(range(100)))) == 100
    with pytest.raises(r.RetrievalRequestError):
        r.normalize_document_ids(list(range(101)))
    with pytest.raises(r.RetrievalRequestError):
        r.normalize_document_ids([1] * 101)


# ---------- query text ----------

def test_query_text_is_returned_exactly_as_given():
    assert r.validate_query_text("  नमस्ते world \n") == "  नमस्ते world \n"


@pytest.mark.parametrize("empty", ["", " ", "\n\t  ", "\u00a0"])
def test_empty_or_whitespace_queries_raise_empty_query_error(empty):
    with pytest.raises(r.EmptyQueryError):
        r.validate_query_text(empty)


@pytest.mark.parametrize("bad", [None, 5, b"hello"])
def test_a_non_string_query_is_a_request_error_not_an_empty_query(bad):
    with pytest.raises(r.RetrievalRequestError) as caught:
        r.validate_query_text(bad)
    assert not isinstance(caught.value, r.EmptyQueryError)


# ---------- integrity ----------

def check(**overrides):
    values = {
    "chunk_id": 7,
    "text": "hello",
    "start_offset": 0,
    "end_offset": 5,
    "span_sha256": sha256_text("hello"),
}
    values.update(overrides)
    r.verify_chunk_integrity(**values)


def test_integrity_passes_for_a_matching_chunk():
    check()


def test_integrity_counts_code_points_not_bytes():
    text = "नमस्ते 😀"
    check(text=text, start_offset=10, end_offset=10 + len(text), span_sha256=sha256_text(text))


def test_integrity_fails_on_a_length_mismatch():
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        check(end_offset=6)
    assert (caught.value.chunk_id, caught.value.check) == (7, "length")


@pytest.mark.parametrize(
    "wrong", [sha256_text("other"), sha256_text("hello").upper(), "0" * 64]
)
def test_integrity_fails_on_a_hash_mismatch(wrong):
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        check(span_sha256=wrong)
    assert (caught.value.chunk_id, caught.value.check) == (7, "sha256")


def test_integrity_errors_never_contain_the_text():
    with pytest.raises(r.RetrievalIntegrityError) as caught:
        check(text="SECRET-TEXT", end_offset=11, span_sha256="0" * 64)
    assert "SECRET" not in str(caught.value)

# ---------- source hash ----------

def test_source_sha256_accepts_valid_hash():
    value = "a" * 64
    assert r.validate_source_sha256(value) == value


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        None,
        123,
    ],
)
def test_source_sha256_rejects_invalid_hash(bad):
    with pytest.raises(r.RetrievalRequestError):
        r.validate_source_sha256(bad)
