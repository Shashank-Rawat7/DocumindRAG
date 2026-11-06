"""C2.5 in-memory index build (app/rag/indexing.py).

Step 2: chunk planning. A fake embedder supplies source units through the real
om-v1 mapper (map_token_offsets), so the real chunker sees exactly what
Embedder.source_units() would give it. No model.
Step 3: page assignment, with mappings from the real pdf_pages functions
(page_mapping_from_texts, and map_document_pages on tests/fixtures/sample.pdf).
Step 4: grouped embedding and the full build, with a fake embedder that returns
deterministic unit vectors. The last section uses the pinned model and runs only
when it is already in the local cache (skipped otherwise; nothing is downloaded).
"""
import hashlib
import zlib
from pathlib import Path

import numpy as np
import pytest

from app.core.processing import extract_text
from app.rag.chunking import DEFAULT_CHUNKER_CONFIG, ChunkerConfig, ChunkingError
from app.rag.embedder import (
    EMBEDDING_DIM,
    PASSAGE_PREFIX,
    Embedder,
    EmbedderConfig,
    EmbedderUnavailableError,
    EmbeddingInputTooLongError,
    EmbeddingOutputError,
)
from app.rag.indexing import (
    EMBED_GROUP_SIZE,
    NO_INDEXABLE_TEXT,
    ChunkRecord,
    DocumentIndexBuild,
    IndexBuildError,
    PagedChunk,
    PagedChunkPlan,
    assign_pages,
    build_document_index,
    build_signature,
    plan_chunks,
)
from app.rag.pdf_pages import (
    PageMapping,
    PageMappingError,
    UnavailableReason,
    map_document_pages,
    page_mapping_from_texts,
)
from app.rag.text_boundaries import is_ignorable
from app.rag.token_offsets import METASPACE, OffsetMappingError, map_token_offsets

SMALL = ChunkerConfig(target_tokens=10, max_tokens=14, overlap_tokens=3)
SAMPLE_PDF = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"


class FakeEmbedder:
    """source_units() over runs of non-ignorable characters, one token per run
    (whitespace and format characters such as ZWSP produce no token).

    ``metaspace_every=k`` makes every k-th word a verified Metaspace pair
    ("\u2581" at (s, s+1) followed by the word at (s, e)), i.e. one source unit
    worth 2 model tokens, as the real XLM-R tokenizer produces.
    """

    def __init__(self, metaspace_every=0, error=None):
        self.metaspace_every = metaspace_every
        self.error = error
        self.texts = []

    def source_units(self, text):
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        offsets, tokens = [], []
        for k, (s, e) in enumerate(word_spans(text)):
            if self.metaspace_every and k % self.metaspace_every == 0:
                offsets.append((s, s + 1))
                tokens.append(METASPACE)
            offsets.append((s, e))
            tokens.append(text[s:e])
        return map_token_offsets(text, offsets, tokens)


def word_spans(text):
    spans, start = [], None
    for pos, ch in enumerate(text + " "):
        if is_ignorable(ch):
            if start is not None:
                spans.append((start, pos))
                start = None
        elif start is None:
            start = pos
    return spans


def words(n, prefix="w"):
    return " ".join(f"{prefix}{k}" for k in range(n))


def test_chunks_are_exact_slices_of_the_unmodified_text():
    text = "  Leading spaces\r\n" + words(60) + "\n\nTrailing paragraph.  \n"
    embedder = FakeEmbedder()
    plan = plan_chunks(text, embedder, SMALL)
    assert embedder.texts == [text] and embedder.texts[0] is text      # tokenized once, unchanged
    assert len(plan.chunks) > 1 and isinstance(plan.chunks, tuple)
    for position, chunk in enumerate(plan.chunks):
        assert chunk.index == position
        assert chunk.text == text[chunk.start:chunk.end]
    assert plan.chunks[0].start == 2                                   # offsets count the leading spaces
    assert plan.chunks[-1].end == text.index("paragraph.") + len("paragraph.")   # trailing whitespace excluded


def test_budgets_and_totals_are_counted_in_model_tokens():
    text = words(80)
    plan = plan_chunks(text, FakeEmbedder(metaspace_every=3), SMALL)
    assert plan.source_tokens == 80 + 27                               # 27 of 80 words are 2-token units
    assert plan.chunks[0].token_start == 0 and plan.chunks[-1].token_end == plan.source_tokens
    assert all(1 <= chunk.token_count <= SMALL.max_tokens for chunk in plan.chunks)


def test_default_chunker_config_is_used():
    text = words(1000)
    plan = plan_chunks(text, FakeEmbedder())
    assert len(plan.chunks) > 1
    assert all(chunk.token_count <= DEFAULT_CHUNKER_CONFIG.max_tokens for chunk in plan.chunks)


@pytest.mark.parametrize("text", ["", "\u200b", "\u200b \ufeff\n\u200d"], ids=["empty", "zwsp", "only-ignorable"])
def test_text_without_indexable_content_raises_index_build_error(text):
    with pytest.raises(IndexBuildError) as excinfo:
        plan_chunks(text, FakeEmbedder())
    assert str(excinfo.value) == NO_INDEXABLE_TEXT == "No indexable text found in this document"


@pytest.mark.parametrize("error", [
    OffsetMappingError("uncovered_character", 3),
    EmbedderUnavailableError("cache missing"),
], ids=["offset-mapping", "embedder-unavailable"])
def test_source_unit_errors_propagate_unchanged(error):
    with pytest.raises(type(error)) as excinfo:
        plan_chunks("some text", FakeEmbedder(error=error), SMALL)
    assert excinfo.value is error


def test_chunking_errors_propagate_unchanged():
    # Every unit is a 2-token Metaspace pair; max_tokens=1 cannot hold any of them.
    tiny = ChunkerConfig(target_tokens=1, max_tokens=1, overlap_tokens=0)
    with pytest.raises(ChunkingError) as excinfo:
        plan_chunks(words(5), FakeEmbedder(metaspace_every=1), tiny)
    assert not isinstance(excinfo.value, IndexBuildError)

# ---------- step 3: page assignment ----------

PAGES = [words(25, "a"), words(25, "b"), words(25, "c")]
PAGED_TEXT = "\n".join(PAGES)            # joined exactly as processing.extract_text joins PDF pages


def page_of(mapping, pos):
    """Page containing ``pos``, by a linear scan of the mapping's spans."""
    [page] = [span.page_number for span in mapping.spans if span.start <= pos < span.end]
    return page


def test_pages_follow_each_chunk_across_page_boundaries():
    mapping = page_mapping_from_texts(PAGES, PAGED_TEXT)
    assert mapping.is_available and mapping.page_count == 3
    paged = assign_pages(plan_chunks(PAGED_TEXT, FakeEmbedder(), SMALL), mapping)
    assert isinstance(paged, PagedChunkPlan) and paged.pages_available
    for item in paged.chunks:
        assert item.page_start == page_of(mapping, item.chunk.start)
        assert item.page_end == page_of(mapping, item.chunk.end - 1)
    spans = {(item.page_start, item.page_end) for item in paged.chunks}
    assert {(1, 1), (2, 2), (3, 3)} <= spans                          # single-page chunks
    assert spans & {(1, 2), (2, 3)}                                   # and chunks crossing a page break


def test_page_assignment_keeps_every_planned_chunk_unchanged():
    plan = plan_chunks(PAGED_TEXT, FakeEmbedder(metaspace_every=3), SMALL)
    paged = assign_pages(plan, page_mapping_from_texts(PAGES, PAGED_TEXT))
    assert isinstance(paged.chunks, tuple) and all(type(item) is PagedChunk for item in paged.chunks)
    assert [item.chunk for item in paged.chunks] == list(plan.chunks)     # same order, same values
    assert all(item.chunk is chunk for item, chunk in zip(paged.chunks, plan.chunks))
    assert paged.source_tokens == plan.source_tokens


@pytest.mark.parametrize("mapping", [
    PageMapping.unavailable(UnavailableReason.NOT_PDF),
    page_mapping_from_texts([], PAGED_TEXT),                          # a PDF with no pages
    page_mapping_from_texts(["different text"], PAGED_TEXT),          # pages do not rebuild the text
], ids=["not-pdf", "no-pages", "text-mismatch"])
def test_unavailable_mapping_gives_null_pages(mapping):
    assert not mapping.is_available
    plan = plan_chunks(PAGED_TEXT, FakeEmbedder(), SMALL)
    paged = assign_pages(plan, mapping)
    assert not paged.pages_available and len(paged.chunks) == len(plan.chunks)
    assert all(item.page_start is None and item.page_end is None for item in paged.chunks)


def test_page_mapping_error_propagates_unchanged():
    # A mapping built for page 1 alone cannot place chunks of the full text: a bug, not "unavailable".
    mapping = page_mapping_from_texts(PAGES[:1], PAGES[0])
    assert mapping.is_available
    plan = plan_chunks(PAGED_TEXT, FakeEmbedder(), SMALL)
    with pytest.raises(PageMappingError, match="outside the text") as excinfo:
        assign_pages(plan, mapping)
    assert not isinstance(excinfo.value, IndexBuildError)


def test_a_mapping_of_the_wrong_type_is_rejected():
    plan = plan_chunks(PAGED_TEXT, FakeEmbedder(), SMALL)
    with pytest.raises(TypeError, match="PageMapping"):
        assign_pages(plan, None)


def test_sample_pdf_chunks_get_their_real_page():
    text = extract_text(SAMPLE_PDF)
    mapping = map_document_pages(SAMPLE_PDF, text)
    assert mapping.is_available
    paged = assign_pages(plan_chunks(text, FakeEmbedder()), mapping)
    assert paged.pages_available and paged.chunks
    for item in paged.chunks:
        assert item.chunk.text == text[item.chunk.start:item.chunk.end]
        assert 1 <= item.page_start <= item.page_end <= mapping.page_count

# ---------- step 4: grouped embedding and the full build ----------

def fake_vector(text):
    """Deterministic unit float32 vector for ``text`` (a stand-in for the model)."""
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    vector = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeVectorEmbedder(FakeEmbedder):
    """FakeEmbedder plus the rest of the Embedder interface used by the build.

    ``corrupt(vectors) -> output`` replaces a group's output (malformed cases);
    ``embed_error`` is raised by embed_passages().
    """

    embedding_id = "test/fake@" + "0" * 40

    def __init__(self, corrupt=None, embed_error=None, **kwargs):
        super().__init__(**kwargs)
        self.corrupt = corrupt
        self.embed_error = embed_error
        self.group_sizes = []

    def signature_fields(self):
        return {"model_id": "test/fake", "embedding_dim": EMBEDDING_DIM}

    def embed_passages(self, texts):
        self.group_sizes.append(len(texts))
        if self.embed_error is not None:
            raise self.embed_error
        vectors = np.stack([fake_vector(PASSAGE_PREFIX + t) for t in texts]).astype(np.float32)
        return self.corrupt(vectors) if self.corrupt else vectors


ONE_WORD = ChunkerConfig(target_tokens=1, max_tokens=1, overlap_tokens=0)   # one chunk per word
NO_PAGES = PageMapping.unavailable(UnavailableReason.NOT_PDF)


def test_chunks_are_embedded_in_groups_of_32_in_order():
    assert EMBED_GROUP_SIZE == 32
    embedder = FakeVectorEmbedder()
    build = build_document_index(words(70), embedder=embedder, page_mapping=NO_PAGES, chunker_config=ONE_WORD)
    assert embedder.group_sizes == [32, 32, 6]
    assert [record.text for record in build.chunks] == words(70).split()
    for record in build.chunks:                                        # every vector belongs to its own chunk
        np.testing.assert_array_equal(record.embedding, fake_vector(PASSAGE_PREFIX + record.text))


def test_records_carry_the_planned_chunks_and_pages_exactly():
    embedder = FakeVectorEmbedder(metaspace_every=3)
    mapping = page_mapping_from_texts(PAGES, PAGED_TEXT)
    build = build_document_index(PAGED_TEXT, embedder=embedder, page_mapping=mapping, chunker_config=SMALL)
    paged = assign_pages(plan_chunks(PAGED_TEXT, FakeVectorEmbedder(metaspace_every=3), SMALL), mapping)
    assert isinstance(build, DocumentIndexBuild) and isinstance(build.chunks, tuple)
    assert len(build.chunks) == len(paged.chunks)
    for record, item in zip(build.chunks, paged.chunks):
        assert type(record) is ChunkRecord
        assert (record.chunk_index, record.start_offset, record.end_offset, record.text, record.span_sha256) == (
            item.chunk.index, item.chunk.start, item.chunk.end, item.chunk.text, item.chunk.span_sha256)
        assert (record.page_start, record.page_end) == (item.page_start, item.page_end)
        assert record.text == PAGED_TEXT[record.start_offset:record.end_offset]


def test_embeddings_are_read_only_unit_float32_vectors():
    build = build_document_index(words(40), embedder=FakeVectorEmbedder(), page_mapping=NO_PAGES,
                                 chunker_config=SMALL)
    for record in build.chunks:
        vector = record.embedding
        assert vector.shape == (EMBEDDING_DIM,) and vector.dtype == np.float32
        assert np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1.0) <= 1e-5
        assert not vector.flags.writeable
        with pytest.raises(ValueError):
            vector[0] = 0.0
    first, second = build.chunks[0].embedding, build.chunks[1].embedding
    assert not np.shares_memory(first, second)                         # each row owns its memory


def test_build_identity_and_stats():
    text = "  " + PAGED_TEXT + "\n"
    embedder = FakeVectorEmbedder(metaspace_every=3)
    build = build_document_index(text, embedder=embedder, page_mapping=NO_PAGES, chunker_config=SMALL)
    plan = plan_chunks(text, FakeVectorEmbedder(metaspace_every=3), SMALL)
    assert build.source_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert build.signature == build_signature(SMALL, embedder)
    assert build.embedding_id == embedder.embedding_id
    assert build.stats.chunk_count == len(build.chunks) == len(plan.chunks)
    assert build.stats.source_tokens == plan.source_tokens
    assert build.stats.max_chunk_tokens == max(chunk.token_count for chunk in plan.chunks)
    assert build.stats.pages_available is False


@pytest.mark.parametrize("corrupt, message", [
    (lambda v: v[:-1], "shape"),                                     # one row missing
    (lambda v: v[:0], "shape"),                                      # empty output
    (lambda v: v[:, :-1], "shape"),                                  # wrong dimension
    (lambda v: v.astype(np.float64), "dtype"),
    (lambda v: np.where(np.arange(v.shape[1]) == 0, np.float32("nan"), v), "non-finite"),
    (lambda v: v * 2, "unit length"),
    (lambda v: v.tolist(), "numpy array"),
], ids=["missing-row", "empty", "wrong-dim", "float64", "nan", "not-normalised", "list"])
def test_malformed_embedder_output_is_rejected(corrupt, message):
    with pytest.raises(EmbeddingOutputError, match=message):
        build_document_index(words(40), embedder=FakeVectorEmbedder(corrupt=corrupt), page_mapping=NO_PAGES,
                             chunker_config=SMALL)


@pytest.mark.parametrize("error", [
    EmbeddingInputTooLongError(3, 600),
    EmbedderUnavailableError("cache missing"),
    EmbeddingOutputError("invalid model output"),
], ids=["too-long", "unavailable", "embedding-output-error",])
def test_embedder_errors_propagate_unchanged(error):
    with pytest.raises(type(error)) as excinfo:
        build_document_index(words(40), embedder=FakeVectorEmbedder(embed_error=error), page_mapping=NO_PAGES,
                             chunker_config=SMALL)
    assert excinfo.value is error



def test_page_errors_fail_the_build_before_any_embedding():
    embedder = FakeVectorEmbedder()
    with pytest.raises(PageMappingError):
        build_document_index(PAGED_TEXT, embedder=embedder, page_mapping=page_mapping_from_texts(PAGES[:1], PAGES[0]),
                             chunker_config=SMALL)
    assert embedder.group_sizes == []


# ---------- step 4: pinned-model measurements (skipped when the model is not cached) ----------

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
ENGLISH = (
    "PostgreSQL stores table rows in heap files and finds them through B-tree indexes. "
    "A vacuum pass reclaims space left behind by deleted rows and keeps statistics fresh. "
)
HINDI = (
    "\u0921\u0947\u091f\u093e\u092c\u0947\u0938 \u0924\u093e\u0932\u093f\u0915\u093e\u0913\u0902 "
    "\u092e\u0947\u0902 \u092a\u0902\u0915\u094d\u0924\u093f\u092f\u093e\u0901 "
    "\u0938\u0902\u0917\u094d\u0930\u0939\u0940\u0924 \u0915\u0930\u0924\u093e \u0939\u0948\u0964 "
)


@pytest.fixture(scope="module")
def pinned():
    with pytest.MonkeyPatch.context() as patch:          # restored after this module
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        embedder = Embedder(EmbedderConfig(MODEL_ID, REVISION, 1))
        try:
            embedder.signature_fields()                  # loads and validates
        except EmbedderUnavailableError as exc:
            if isinstance(exc.__cause__, OSError):       # not in the local cache
                pytest.skip(f"pinned model not in the local cache ({type(exc.__cause__).__name__})")
            raise                                        # anything else is a real failure
        yield embedder


def measure(embedder, build):
    """Real passage tokens ("passage: " + chunk text, special tokens included) per chunk."""
    return [embedder.count_passage_tokens(record.text) for record in build.chunks]


def test_pinned_sample_pdf_build_stays_within_the_model_limit(pinned, record_property):
    text = extract_text(SAMPLE_PDF)
    build = build_document_index(text, embedder=pinned, page_mapping=map_document_pages(SAMPLE_PDF, text))
    passage_tokens = measure(pinned, build)
    record_property("max_passage_tokens", max(passage_tokens))
    record_property("max_chunk_tokens", build.stats.max_chunk_tokens)
    print(f"sample.pdf: {build.stats.chunk_count} chunks, max chunk tokens {build.stats.max_chunk_tokens}, "
          f"max passage tokens {max(passage_tokens)} / 512")
    assert build.stats.pages_available and max(passage_tokens) <= 512
    assert all(record.text == text[record.start_offset:record.end_offset] for record in build.chunks)


def test_pinned_long_english_hindi_build_stays_within_the_model_limit(pinned, record_property):
    paragraphs = [(ENGLISH if k % 2 == 0 else HINDI) * 12 for k in range(16)]
    text = "\n\n".join(paragraphs)
    build = build_document_index(text, embedder=pinned, page_mapping=NO_PAGES)
    passage_tokens = measure(pinned, build)
    margins = [p - record_count for p, record_count in
               zip(passage_tokens, (c.token_count for c in plan_chunks(text, pinned).chunks))]
    record_property("max_passage_tokens", max(passage_tokens))
    record_property("max_passage_minus_chunk_tokens", max(margins))
    print(f"English/Hindi: {build.stats.chunk_count} chunks, max chunk tokens {build.stats.max_chunk_tokens}, "
          f"max passage tokens {max(passage_tokens)} / 512, passage - chunk tokens {min(margins)}..{max(margins)}")
    assert build.stats.chunk_count > 3 and max(passage_tokens) <= 512
    for record in build.chunks:
        assert abs(float(np.linalg.norm(record.embedding)) - 1.0) <= 1e-5 and not record.embedding.flags.writeable
