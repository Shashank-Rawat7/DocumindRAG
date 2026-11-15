"""E1.7.2: candidate pool and RetrievedChunk -> RankedChunk conversion (pure, fake fixtures).

A  document mapping: correct, ids never assumed, missing, ambiguous, unknown, foreign, stale
B  the pool: conversion, canonical order, offsets and text hash
C  chunk checks: foreign document, stale version and hash, integrity, wrong offsets, duplicates
D  completeness
E  dense conversion: same eligible chunks as the pool
F  privacy and immutability
G  field compatibility with the real RetrievedChunk (from source) and import isolation

Run focused: pytest tests/test_evaluation_pool.py --noconftest -p no:cacheprovider
"""
import ast
import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import pool as p
from app.rag.evaluation import schema as s
from app.rag.evaluation.metrics import RankedChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OWNER = 10
EMB = "emb-1"
CANARY = "SECRET-CANARY-7731"

TEXT_A = "Alpha clause one. Beta clause two. Gamma clause three."
TEXT_B = "Delta section four. Epsilon section five. Zeta section six."


def mdoc(key, text):
    return s.ManifestDocument(
        corpus_key=key, title="T", publisher="P", source_url="https://example.invalid/a", license="MIT",
        license_url="https://example.invalid/l", format="txt", language="en",
        source_file_sha256=hashlib.sha256(text.encode()).hexdigest(),
        extracted_text_sha256=sha256_text(text), extracted_text_length=len(text),
        extraction=s.Extraction(processing_ref="d" * 40, python="3.12.4"), redistributable=True,
    )


MANIFEST = s.Manifest("eval-manifest-v1", "c1", "ingested", (mdoc("doc-a", TEXT_A), mdoc("doc-b", TEXT_B)))
TEXTS = {"doc-a": TEXT_A, "doc-b": TEXT_B}
ID_A, ID_B = 931, 417  # deliberately not 1, 2 and not in manifest order


@dataclasses.dataclass
class FakeChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    span_sha256: str
    source_version: int = 1
    source_sha256: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.5


def row(document_id, key, **over):
    base = {
        "document_id": document_id, "owner_id": OWNER, "filename": f"{key}.txt", "description": f"c1:{key}",
        "source_version": 1, "indexed_source_version": 1, "indexed_embedding_id": EMB,
        "source_sha256": sha256_text(TEXTS[key]),
    }
    base.update(over)
    return p.PoolDocument(**base)


def docs():
    return [row(ID_A, "doc-a"), row(ID_B, "doc-b")]


def mapping(rows=None, **kw):
    return p.map_documents(MANIFEST, docs() if rows is None else rows, owner_id=OWNER, embedding_id=EMB, **kw)


def chunk_of(chunk_id, document_id, key, index, quote, **over):
    text = TEXTS[key]
    start = text.index(quote)
    base = {
        "chunk_id": chunk_id, "document_id": document_id, "chunk_index": index, "start_offset": start,
        "end_offset": start + len(quote), "text": quote, "span_sha256": sha256_text(quote),
        "source_sha256": sha256_text(text),
    }
    base.update(over)
    return FakeChunk(**base)


def good_chunks():
    return [
        chunk_of(1, ID_A, "doc-a", 0, "Alpha clause one."),
        chunk_of(2, ID_A, "doc-a", 1, "Beta clause two."),
        chunk_of(3, ID_A, "doc-a", 2, "Gamma clause three."),
        chunk_of(4, ID_B, "doc-b", 0, "Delta section four."),
        chunk_of(5, ID_B, "doc-b", 1, "Epsilon section five."),
        chunk_of(6, ID_B, "doc-b", 2, "Zeta section six."),
    ]


def pool_of(chunks=None, document_map=None, texts=TEXTS):
    return p.build_pool(MANIFEST, document_map or mapping(), good_chunks() if chunks is None else chunks, texts)


def error_of(fn, *args, **kwargs):
    with pytest.raises(s.EvaluationDataError) as info:
        fn(*args, **kwargs)
    return info.value


def codes(error):
    return {i.code for i in error.issues}


# ---------- A: document mapping ----------

def test_the_mapping_is_built_from_descriptions_and_assumes_no_ids():
    m = mapping()
    assert {**m.key_by_id} == {ID_A: "doc-a", ID_B: "doc-b"}
    assert {**m.id_by_key} == {"doc-a": ID_A, "doc-b": ID_B}
    assert m.document_ids == (ID_A, ID_B)  # manifest order, whatever the ids are
    assert m.corpus_version == "c1" and m.owner_id == OWNER and m.embedding_id == EMB


def test_row_order_does_not_change_the_mapping():
    assert {**mapping(list(reversed(docs()))).key_by_id} == {**mapping().key_by_id}


def test_a_document_row_does_not_need_a_filename_check_unless_filenames_are_given():
    assert mapping([row(ID_A, "doc-a", filename="anything.txt"), row(ID_B, "doc-b")])


def test_expected_filenames_are_checked_when_given():
    names = {"doc-a": "doc-a.txt", "doc-b": "doc-b.txt"}
    assert mapping(expected_filenames=names)
    bad = [row(ID_A, "doc-a", filename="other.txt"), row(ID_B, "doc-b")]
    # the mismatching row is not used, so its document is also reported missing
    assert codes(error_of(mapping, bad, expected_filenames=names)) == {p.DOC_FILENAME_MISMATCH, p.DOC_MISSING}


def test_the_right_file_under_a_wrong_description_is_a_filename_mismatch_and_the_key_is_missing():
    bad = [row(ID_A, "doc-a", description="c1:something-else"), row(ID_B, "doc-b")]
    found = codes(error_of(mapping, bad, expected_filenames={"doc-a": "doc-a.txt", "doc-b": "doc-b.txt"}))
    assert found == {p.DOC_FILENAME_MISMATCH, p.DOC_MISSING}


def test_a_missing_document_is_refused():
    assert codes(error_of(mapping, [row(ID_A, "doc-a")])) == {p.DOC_MISSING}


def test_no_rows_at_all_is_refused_for_every_document():
    err = error_of(mapping, [])
    assert [i.code for i in err.issues] == [p.DOC_MISSING, p.DOC_MISSING]


def test_two_rows_for_one_document_are_ambiguous():
    rows = docs() + [row(555, "doc-a")]
    assert codes(error_of(mapping, rows)) == {p.DOC_AMBIGUOUS}


def test_a_repeated_document_id_is_refused():
    rows = docs() + [row(ID_A, "doc-b")]
    assert p.DOC_DUPLICATE_ID in codes(error_of(mapping, rows))


def test_a_row_that_maps_to_no_manifest_document_is_unknown():
    assert codes(error_of(mapping, docs() + [row(777, "doc-a", description="c1:doc-z")])) == {p.DOC_UNKNOWN}


def test_a_description_from_another_corpus_version_is_unknown_and_leaves_the_document_missing():
    rows = [row(ID_A, "doc-a", description="c0:doc-a"), row(ID_B, "doc-b")]
    assert codes(error_of(mapping, rows)) == {p.DOC_UNKNOWN, p.DOC_MISSING}


def test_a_row_without_a_description_is_unknown():
    assert codes(error_of(mapping, docs() + [row(778, "doc-a", description=None)])) == {p.DOC_UNKNOWN}


def test_a_foreign_owner_is_refused():
    rows = [row(ID_A, "doc-a", owner_id=OWNER + 1), row(ID_B, "doc-b")]
    assert codes(error_of(mapping, rows)) == {p.DOC_FOREIGN_OWNER}


def test_an_unpublished_document_is_refused():
    rows = [row(ID_A, "doc-a", indexed_source_version=None), row(ID_B, "doc-b")]
    assert codes(error_of(mapping, rows)) == {p.DOC_NOT_PUBLISHED}


def test_a_stale_source_version_is_refused():
    rows = [row(ID_A, "doc-a", source_version=2, indexed_source_version=1), row(ID_B, "doc-b")]
    assert codes(error_of(mapping, rows)) == {p.DOC_STALE_VERSION}


def test_another_embedder_is_refused():
    rows = [row(ID_A, "doc-a", indexed_embedding_id="emb-2"), row(ID_B, "doc-b")]
    assert codes(error_of(mapping, rows)) == {p.DOC_EMBEDDING_MISMATCH}


def test_a_stale_or_missing_source_hash_is_refused():
    for bad in ("0" * 64, None):
        rows = [row(ID_A, "doc-a", source_sha256=bad), row(ID_B, "doc-b")]
        assert codes(error_of(mapping, rows)) == {p.DOC_SOURCE_HASH}


def test_wrongly_typed_row_fields_are_refused():
    for over in ({"document_id": True}, {"owner_id": "10"}, {"source_version": 1.0}, {"filename": None}):
        assert p.DOC_FIELD_INVALID in codes(error_of(mapping, docs() + [dataclasses.replace(row(9, "doc-a"), **over)]))
    assert p.DOC_FIELD_INVALID in codes(error_of(mapping, docs() + [object()]))


def test_a_draft_manifest_is_refused():
    draft = dataclasses.replace(MANIFEST, stage="draft")
    assert codes(error_of(p.map_documents, draft, docs(), owner_id=OWNER, embedding_id=EMB)) == {p.DOC_MANIFEST}


def test_findings_are_collected_not_first_only():
    rows = [row(ID_A, "doc-a", owner_id=1, source_sha256="0" * 64), row(ID_B, "doc-b", indexed_source_version=None)]
    assert codes(error_of(mapping, rows)) == {p.DOC_FOREIGN_OWNER, p.DOC_SOURCE_HASH, p.DOC_NOT_PUBLISHED}


def test_argument_types():
    with pytest.raises(TypeError):
        p.map_documents("m", docs(), owner_id=OWNER, embedding_id=EMB)
    with pytest.raises(TypeError):
        p.map_documents(MANIFEST, docs(), owner_id=True, embedding_id=EMB)
    with pytest.raises(TypeError):
        p.map_documents(MANIFEST, docs(), owner_id=OWNER, embedding_id="")
    with pytest.raises(TypeError):
        p.map_documents(MANIFEST, docs(), owner_id=OWNER, embedding_id=EMB, expected_filenames=["x"])


def test_is_retrievable_mirrors_the_dense_predicate():
    assert p.is_retrievable(row(1, "doc-a"), EMB)
    assert not p.is_retrievable(row(1, "doc-a", indexed_source_version=None), EMB)
    assert not p.is_retrievable(row(1, "doc-a", source_version=2), EMB)
    assert not p.is_retrievable(row(1, "doc-a"), "other")


# ---------- B: the pool and the conversion ----------

def test_the_pool_holds_every_chunk_as_a_rankedchunk_in_canonical_order():
    pool = pool_of()
    assert len(pool) == 6
    assert [(c.corpus_key, c.start, c.end) for c in pool.ranked] == sorted(
        (c.corpus_key, c.start, c.end) for c in pool.ranked
    )
    assert pool.ranked[0] == RankedChunk("doc-a", 0, 17, sha256_text("Alpha clause one."))


def test_offsets_and_text_hash_are_converted_exactly():
    pool = pool_of()
    c = chunk_of(5, ID_B, "doc-b", 1, "Epsilon section five.")
    ranked = pool.by_chunk_id[5]
    assert (ranked.corpus_key, ranked.start, ranked.end) == ("doc-b", c.start_offset, c.end_offset)
    assert ranked.text_sha256 == c.span_sha256 == sha256_text(TEXT_B[ranked.start:ranked.end])


def test_the_pool_is_independent_of_chunk_input_order():
    first = pool_of()
    second = pool_of(list(reversed(good_chunks())))
    assert first.ranked == second.ranked and first.pool_sha256 == second.pool_sha256
    assert len(first.pool_sha256) == 64


def test_the_pool_hash_changes_with_the_chunking():
    merged = [chunk_of(1, ID_A, "doc-a", 0, "Alpha clause one. Beta clause two.")] + [
        dataclasses.replace(c, chunk_index=c.chunk_index - 1) for c in good_chunks()[2:3]
    ] + good_chunks()[3:]
    assert pool_of(merged).pool_sha256 != pool_of().pool_sha256


def test_overlapping_chunks_are_allowed():
    overlapped = good_chunks()[:2] + [chunk_of(3, ID_A, "doc-a", 2, "two. Gamma clause three.")] + good_chunks()[3:]
    assert len(pool_of(overlapped)) == 6


def test_extra_chunk_attributes_are_ignored_and_the_pool_is_read_only():
    pool = pool_of()
    with pytest.raises(TypeError):
        pool.by_chunk_id[99] = pool.ranked[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        pool.pool_sha256 = "x"


# ---------- C: chunk checks ----------

def test_a_chunk_of_a_foreign_document_is_refused():
    chunks = good_chunks() + [chunk_of(7, 12345, "doc-a", 3, "Alpha clause one.")]
    assert codes(error_of(pool_of, chunks)) == {p.POOL_FOREIGN_DOCUMENT}


def test_a_chunk_of_a_document_with_a_foreign_owner_cannot_be_mapped():
    # the foreign owner's document never enters the mapping, so its chunk is a foreign document
    assert codes(error_of(mapping, [row(ID_A, "doc-a", owner_id=99), row(ID_B, "doc-b")])) == {p.DOC_FOREIGN_OWNER}
    chunks = good_chunks() + [chunk_of(8, 4242, "doc-b", 3, "Zeta section six.")]
    assert codes(error_of(pool_of, chunks)) == {p.POOL_FOREIGN_DOCUMENT}


def test_a_stale_chunk_source_version_is_refused():
    chunks = good_chunks()
    chunks[0] = dataclasses.replace(chunks[0], source_version=2)
    assert codes(error_of(pool_of, chunks)) == {p.POOL_STALE_VERSION}


def test_a_stale_chunk_source_hash_is_refused():
    chunks = good_chunks()
    chunks[3] = dataclasses.replace(chunks[3], source_sha256="0" * 64)
    assert codes(error_of(pool_of, chunks)) == {p.POOL_SOURCE_HASH}


def test_a_chunk_hash_that_does_not_match_its_text_is_refused():
    chunks = good_chunks()
    chunks[1] = dataclasses.replace(chunks[1], span_sha256="0" * 64)
    assert codes(error_of(pool_of, chunks)) == {p.POOL_INTEGRITY}


def test_a_chunk_length_that_does_not_match_its_range_is_refused():
    chunks = good_chunks()
    chunks[1] = dataclasses.replace(chunks[1], end_offset=chunks[1].end_offset + 1)
    assert codes(error_of(pool_of, chunks)) == {p.POOL_INTEGRITY}


def test_wrong_offsets_with_a_consistent_text_are_refused():
    chunks = good_chunks()
    c = chunks[1]
    chunks[1] = dataclasses.replace(c, start_offset=c.start_offset + 1, end_offset=c.end_offset + 1)  # same length
    assert codes(error_of(pool_of, chunks)) == {p.POOL_TEXT_MISMATCH}


def test_chunk_text_that_differs_from_the_source_is_refused_even_with_a_matching_hash():
    chunks = good_chunks()
    c = chunks[0]
    fake = "Alpha clause 0ne."
    chunks[0] = dataclasses.replace(c, text=fake, span_sha256=sha256_text(fake))
    assert codes(error_of(pool_of, chunks)) == {p.POOL_TEXT_MISMATCH}


def test_a_chunk_past_the_end_of_the_text_is_refused():
    chunks = good_chunks()
    long_text = "x" * 10
    chunks[2] = dataclasses.replace(
        chunks[2], start_offset=len(TEXT_A) - 2, end_offset=len(TEXT_A) + 8, text=long_text, span_sha256=sha256_text(long_text)
    )
    assert codes(error_of(pool_of, chunks)) == {p.POOL_TEXT_MISMATCH}


def test_duplicate_chunk_ids_slots_and_ranges_are_refused():
    same_id = good_chunks() + [dataclasses.replace(good_chunks()[0], chunk_index=9)]
    assert codes(error_of(pool_of, same_id)) == {p.POOL_DUPLICATE_CHUNK}
    same_slot = good_chunks() + [dataclasses.replace(good_chunks()[0], chunk_id=50)]
    assert codes(error_of(pool_of, same_slot)) == {p.POOL_DUPLICATE_CHUNK}
    same_range = good_chunks() + [dataclasses.replace(good_chunks()[0], chunk_id=51, chunk_index=9)]
    assert codes(error_of(pool_of, same_range)) == {p.POOL_DUPLICATE_CHUNK}


def test_an_exact_duplicate_of_a_chunk_is_refused():
    assert codes(error_of(pool_of, good_chunks() + [good_chunks()[2]])) == {p.POOL_DUPLICATE_CHUNK}


def test_invalid_chunk_fields_are_refused():
    base = good_chunks()[0]
    for over in (
        {"chunk_id": True}, {"document_id": "931"}, {"start_offset": -1}, {"end_offset": 0}, {"chunk_index": -1},
        {"text": None}, {"span_sha256": 5}, {"source_version": 1.5}, {"source_sha256": None},
    ):
        assert codes(error_of(pool_of, good_chunks() + [dataclasses.replace(base, **over)])) == {p.POOL_CHUNK_INVALID}, over
    assert codes(error_of(pool_of, good_chunks() + [object()])) == {p.POOL_CHUNK_INVALID}


def test_a_document_without_a_verified_text_is_refused():
    assert codes(error_of(pool_of, None, None, {"doc-a": TEXT_A})) == {p.POOL_TEXT_MISSING}


# ---------- D: completeness ----------

def test_a_document_with_no_chunk_is_refused():
    assert codes(error_of(pool_of, good_chunks()[:3])) == {p.POOL_DOCUMENT_EMPTY}


def test_a_gap_in_chunk_indexes_is_refused():
    chunks = good_chunks()
    chunks[2] = dataclasses.replace(chunks[2], chunk_index=5)
    assert codes(error_of(pool_of, chunks)) == {p.POOL_INDEX_GAP}


def test_a_missing_chunk_leaves_meaningful_text_uncovered():
    chunks = good_chunks()
    del chunks[1]  # chunk 1 of doc-a: indexes become 0,2
    chunks[1] = dataclasses.replace(chunks[1], chunk_index=1)  # keep indexes contiguous
    assert codes(error_of(pool_of, chunks)) == {p.POOL_COVERAGE}


def test_uncovered_whitespace_is_allowed_but_uncovered_text_at_either_edge_is_not():
    padded = {"doc-a": "  " + TEXT_A + "  ", "doc-b": TEXT_B}
    manifest = s.Manifest("eval-manifest-v1", "c1", "ingested", (mdoc("doc-a", padded["doc-a"]), mdoc("doc-b", TEXT_B)))
    dmap = p.map_documents(
        manifest,
        [row(ID_A, "doc-a", source_sha256=sha256_text(padded["doc-a"])), row(ID_B, "doc-b")],
        owner_id=OWNER, embedding_id=EMB,
    )
    shifted = [dataclasses.replace(c, start_offset=c.start_offset + 2, end_offset=c.end_offset + 2,
                                   source_sha256=sha256_text(padded["doc-a"])) if c.document_id == ID_A else c
               for c in good_chunks()]
    assert len(p.build_pool(manifest, dmap, shifted, padded)) == 6
    tail = [c for c in shifted if c.chunk_id != 3]
    # dropping the last chunk of doc-a leaves "Gamma clause three." uncovered
    assert codes(error_of(p.build_pool, manifest, dmap, tail, padded)) == {p.POOL_COVERAGE}


def test_ranges_that_do_not_strictly_increase_are_refused():
    chunks = good_chunks()
    chunks[1], chunks[2] = (
        dataclasses.replace(chunks[1], chunk_index=2), dataclasses.replace(chunks[2], chunk_index=1),
    )
    assert codes(error_of(pool_of, chunks)) == {p.POOL_RANGE_ORDER}


# ---------- E: dense conversion ----------

def test_a_dense_ranking_is_converted_in_rank_order():
    pool = pool_of()
    dense = [good_chunks()[4], good_chunks()[0], good_chunks()[2]]
    assert p.convert_retrieved(pool, dense) == (pool.by_chunk_id[5], pool.by_chunk_id[1], pool.by_chunk_id[3])
    assert p.convert_retrieved(pool, []) == ()


def test_every_dense_chunk_is_in_the_pool_for_the_same_eligible_documents():
    pool = pool_of()
    ranking = p.convert_retrieved(pool, good_chunks())
    assert set(ranking) == set(pool.ranked)
    assert set(pool.document_map.document_ids) == {c.document_id for c in good_chunks()}


def test_a_dense_chunk_outside_the_pool_is_refused():
    pool = pool_of()
    assert codes(error_of(p.convert_retrieved, pool, [chunk_of(99, ID_A, "doc-a", 0, "Alpha clause one.")])) == {p.RANK_NOT_IN_POOL}
    foreign = dataclasses.replace(good_chunks()[0], document_id=12345)
    assert codes(error_of(p.convert_retrieved, pool, [foreign])) == {p.RANK_NOT_IN_POOL}


def test_a_repeated_dense_chunk_is_refused():
    pool = pool_of()
    assert codes(error_of(p.convert_retrieved, pool, [good_chunks()[0], good_chunks()[0]])) == {p.RANK_DUPLICATE}


def test_a_dense_chunk_that_differs_from_the_pool_is_refused():
    pool = pool_of()
    base = good_chunks()[0]
    for over in (
        {"start_offset": 1, "end_offset": 18},
        {"span_sha256": "0" * 64},
        {"source_version": 2},
        {"source_sha256": "0" * 64},
        {"text": "Alpha clause 0ne."},
    ):
        assert codes(error_of(p.convert_retrieved, pool, [dataclasses.replace(base, **over)])) == {p.RANK_MISMATCH}, over


def test_invalid_dense_chunks_and_types_are_refused():
    pool = pool_of()
    assert codes(error_of(p.convert_retrieved, pool, [object()])) == {p.POOL_CHUNK_INVALID}
    with pytest.raises(TypeError):
        p.convert_retrieved("pool", [])


def test_the_score_is_not_used():
    pool = pool_of()
    a = dataclasses.replace(good_chunks()[0], score=0.99)
    b = dataclasses.replace(good_chunks()[0], score=-3.0)
    assert p.convert_retrieved(pool, [a]) == p.convert_retrieved(pool, [b])


# ---------- F: privacy ----------

def test_no_finding_holds_chunk_or_document_text():
    leaky = TEXT_A + " " + CANARY
    manifest = s.Manifest("eval-manifest-v1", "c1", "ingested", (mdoc("doc-a", leaky), mdoc("doc-b", TEXT_B)))
    texts = {"doc-a": leaky, "doc-b": TEXT_B}
    dmap = p.map_documents(
        manifest, [row(ID_A, "doc-a", source_sha256=sha256_text(leaky)), row(ID_B, "doc-b")],
        owner_id=OWNER, embedding_id=EMB,
    )
    chunks = good_chunks()
    chunks[0] = dataclasses.replace(chunks[0], text=CANARY, span_sha256="0" * 64, source_version=7)
    err = error_of(p.build_pool, manifest, dmap, chunks, texts)
    blob = " ".join(f"{i.code} {i.location} {i.message}" for i in err.issues) + str(err)
    assert CANARY not in blob and "Alpha" not in blob


# ---------- G: compatibility and isolation ----------

def _run(code, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)


def test_the_pool_module_imports_no_database_or_retrieval_code():
    done = _run(
        "import sys\nimport app.rag.evaluation.pool\n"
        "bad = [m for m in ('sqlalchemy','app.core.database','app.core.config','app.models','app.rag.retrieval',"
        "'app.rag.embedder','numpy','torch') if m in sys.modules]\nprint('bad=' + ','.join(bad))\n"
    )
    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.strip() == "bad="


def test_the_fake_chunk_has_every_field_the_real_retrievedchunk_declares():
    # read from the source, so it runs where numpy (imported by app.rag.retrieval) is not installed
    tree = ast.parse((PROJECT_ROOT / "app" / "rag" / "retrieval.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetrievedChunk")
    declared = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert declared == {f.name for f in dataclasses.fields(FakeChunk)}
