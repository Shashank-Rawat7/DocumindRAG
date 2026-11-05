"""C2.4 embedder (app/rag/embedder.py).

Normal tests use a fake word tokenizer and a tiny randomly initialised
BertModel (hidden size 384, SDPA) built from a BertConfig: no files, no network,
no Hugging Face cache. The last section uses the pinned model and runs only when
it is already in the local cache (skipped otherwise; nothing is downloaded).
om-v1 rules themselves are covered by test_token_offsets.py; here only the
embedder's integration with them is tested.
"""
import json
import re
import threading
import time
import types
import zlib

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from transformers import BertConfig, BertModel

from app.core.config import Settings
from app.rag.chunking import ChunkerConfig, chunk_text
from app.rag.embedder import (
    EMBEDDER_VERSION,
    EMBEDDING_DIM,
    MAX_INPUT_TOKENS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    Embedder,
    EmbedderConfig,
    EmbedderError,
    EmbedderUnavailableError,
    EmbeddingInputError,
    EmbeddingInputTooLongError,
    EmbeddingOutputError,
    load_pinned_model,
)
from app.rag.token_offsets import METASPACE, OffsetMappingError

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

BOS, PAD, EOS = 0, 1, 2
VOCAB = 1000
_UNSET = object()


# ---------- fakes and helpers ----------

class FakeTokenizer:
    """Whitespace tokenizer: one id per word, <s>/</s> when special tokens are
    requested. ``output`` / ``tokens`` replace the real results (malformed cases)."""

    is_fast = True
    pad_token_id = PAD
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, output=_UNSET, tokens=_UNSET):
        self.output = output
        self.tokens = tokens
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.output is not _UNSET:
            return self.output
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        ids = [3 + zlib.crc32(text[s:e].encode("utf-8")) % (VOCAB - 3) for s, e in spans]
        if kwargs.get("add_special_tokens"):
            ids = [BOS] + ids + [EOS]
        encoding = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            encoding["offset_mapping"] = spans
        return encoding

    def convert_ids_to_tokens(self, ids):
        return self.tokens if self.tokens is not _UNSET else [f"w{i}" for i in ids]


def tiny_model(attn="sdpa", seed=0):
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=EMBEDDING_DIM,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=MAX_INPUT_TOKENS,
        pad_token_id=PAD,
        attn_implementation=attn,
    )
    torch.manual_seed(seed)
    return BertModel(config, add_pooling_layer=False).eval()


def make_embedder(tokenizer=None, model=None, batch_size=1):
    """An Embedder with a counting fake loader; returns (embedder, loads)."""
    tokenizer = FakeTokenizer() if tokenizer is None else tokenizer
    model = tiny_model() if model is None else model
    loads = []

    def loader(config):
        loads.append(config)
        return tokenizer, model

    return Embedder(EmbedderConfig("test/tiny", REVISION, batch_size), loader=loader), loads


def count_forward_calls(model):
    calls = []
    model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True)
    return calls


def words(n, word="w"):
    return " ".join(f"{word}{k}" for k in range(n))


def run_threads(target, count=8):
    errors = []

    def wrapped(k):
        try:
            target(k)
        except Exception as exc:          # reported by the caller's assertion
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(k,)) for k in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


TEXTS = ["PostgreSQL stores table rows in heap files.", "x", words(40), "", "a b c d e f g"]


# ---------- 1. lazy loading, exactly once ----------

def test_construction_and_cheap_calls_do_not_load():
    embedder, loads = make_embedder()
    assert embedder.embedding_id == f"test/tiny@{REVISION}"
    assert embedder.embed_passages([]).shape == (0, EMBEDDING_DIM)
    with pytest.raises(EmbeddingInputError):
        embedder.embed_passages("not a list")
    assert not embedder.is_loaded and loads == []


def test_model_loads_once_across_operations():
    embedder, loads = make_embedder()
    embedder.embed_query("hello")
    embedder.embed_passages(["a", "b"])
    embedder.count_passage_tokens("a")
    embedder.source_units("a b")
    embedder.signature_fields()
    assert embedder.is_loaded and len(loads) == 1 and loads[0] is embedder.config


def test_failed_load_is_not_cached_and_is_retried():
    attempts = []
    model = tiny_model()

    def flaky_loader(config):
        attempts.append(1)
        if len(attempts) == 1:
            raise EmbedderUnavailableError("cache missing")
        return FakeTokenizer(), model

    embedder = Embedder(EmbedderConfig("test/tiny", REVISION, 1), loader=flaky_loader)
    with pytest.raises(EmbedderUnavailableError):
        embedder.embed_query("x")
    assert not embedder.is_loaded
    assert embedder.embed_query("x").shape == (EMBEDDING_DIM,)
    assert len(attempts) == 2


def test_load_pinned_model_requests_local_pinned_safetensors_sdpa(monkeypatch):
    import transformers

    seen = {}
    model = tiny_model().train()
    tokenizer = FakeTokenizer()

    def fake_tokenizer_loader(model_id, **kwargs):
        seen["tokenizer"] = (model_id, kwargs)
        return tokenizer

    def fake_model_loader(model_id, **kwargs):
        seen["model"] = (model_id, kwargs)
        return model

    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=fake_tokenizer_loader))
    monkeypatch.setattr(transformers, "AutoModel", types.SimpleNamespace(from_pretrained=fake_model_loader))

    assert load_pinned_model(EmbedderConfig(MODEL_ID, REVISION, 1)) == (tokenizer, model)
    common = {"revision": REVISION, "local_files_only": True, "trust_remote_code": False}
    assert seen["tokenizer"] == (MODEL_ID, common)
    assert seen["model"] == (MODEL_ID, {**common, "use_safetensors": True, "dtype": torch.float32,
                                        "attn_implementation": "sdpa"})
    assert not model.training


# ---------- 2. output contract ----------

def test_output_shape_dtype_finite_and_unit_norm():
    embedder, _ = make_embedder()
    passages = embedder.embed_passages(TEXTS)
    query = embedder.embed_query("How does PostgreSQL store rows?")
    assert passages.shape == (len(TEXTS), EMBEDDING_DIM) and passages.dtype == np.float32
    assert query.shape == (EMBEDDING_DIM,) and query.dtype == np.float32
    for vectors in (passages, query[None, :]):
        assert np.isfinite(vectors).all()
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_non_finite_model_output_is_rejected():
    model = tiny_model()

    def poison(module, args, output):
        output.last_hidden_state.fill_(float("nan"))     # in place; the hook returns None

    model.register_forward_hook(poison)
    embedder, _ = make_embedder(model=model)
    with pytest.raises(EmbeddingOutputError, match="non-finite"):
        embedder.embed_passages(["a b c"])


# ---------- 3. prefixes ----------

def test_prefixes_are_applied_once_and_source_units_use_raw_text():
    tokenizer = FakeTokenizer()
    embedder, _ = make_embedder(tokenizer=tokenizer)
    embedder.embed_passages(["first text", "second"])
    embedder.embed_query("a question")
    embedder.count_passage_tokens("counted")
    embedder.source_units("raw text")
    assert [text for text, _ in tokenizer.calls] == [
        PASSAGE_PREFIX + "first text", PASSAGE_PREFIX + "second",
        QUERY_PREFIX + "a question", PASSAGE_PREFIX + "counted", "raw text",
    ]
    assert tokenizer.calls[-1][1]["add_special_tokens"] is False
    assert (PASSAGE_PREFIX, QUERY_PREFIX) == ("passage: ", "query: ")


def test_same_text_as_passage_and_query_gives_different_vectors():
    embedder, _ = make_embedder()
    passage = embedder.embed_passages(["database storage"])[0]
    assert not np.allclose(passage, embedder.embed_query("database storage"), atol=1e-3)


# ---------- 4. the 512-token limit (prefix and special tokens included) ----------
# Fake tokenizer: "passage:"/"query:" is one word, plus <s> and </s>, so n words -> n + 3 tokens.

def test_exactly_512_tokens_is_accepted_and_counted_with_prefix_and_specials():
    tokenizer = FakeTokenizer()
    embedder, _ = make_embedder(tokenizer=tokenizer)
    assert embedder.count_passage_tokens("") == 3
    text = words(MAX_INPUT_TOKENS - 3)
    assert embedder.count_passage_tokens(text) == 512
    assert embedder.embed_passages([text]).shape == (1, EMBEDDING_DIM)
    assert embedder.embed_query(text).shape == (EMBEDDING_DIM,)
    assert tokenizer.calls[-1][1] == {"add_special_tokens": True, "truncation": False, "padding": False}


def test_513_tokens_is_rejected_before_the_model_runs():
    model = tiny_model()
    forward_calls = count_forward_calls(model)
    embedder, _ = make_embedder(model=model)
    too_long = words(MAX_INPUT_TOKENS - 2, "secretword")
    with pytest.raises(EmbeddingInputTooLongError) as excinfo:
        embedder.embed_passages(["short", too_long, "short"])
    error = excinfo.value
    assert (error.index, error.token_count, error.limit) == (1, 513, 512)
    assert isinstance(error, EmbeddingInputError) and "secretword" not in str(error)
    with pytest.raises(EmbeddingInputTooLongError):
        embedder.embed_query(too_long)
    assert forward_calls == []          # nothing embedded, nothing truncated


# ---------- 5. batching and input order ----------

def test_batching_matches_single_inputs_and_keeps_order():
    single, _ = make_embedder(batch_size=1)
    batched, _ = make_embedder(batch_size=3)
    reference = single.embed_passages(TEXTS)
    np.testing.assert_allclose(batched.embed_passages(TEXTS), reference, atol=1e-5)
    order = [3, 0, 4, 1, 2]
    np.testing.assert_allclose(batched.embed_passages([TEXTS[k] for k in order]), reference[order], atol=1e-5)


def test_model_receives_right_padded_ids_and_mask_only():
    model = tiny_model()
    forward_calls = count_forward_calls(model)
    tokenizer = FakeTokenizer()
    embedder, _ = make_embedder(tokenizer=tokenizer, model=model, batch_size=2)
    embedder.embed_passages(["a b c", "d", "e f"])
    assert len(forward_calls) == 2                        # ceil(3 / 2) batches
    first = forward_calls[0]
    assert set(first) == {"input_ids", "attention_mask"}
    assert first["input_ids"].dtype == torch.long and first["attention_mask"].dtype == torch.long
    long_ids = tokenizer(PASSAGE_PREFIX + "a b c", add_special_tokens=True)["input_ids"]   # 6 ids
    short_ids = tokenizer(PASSAGE_PREFIX + "d", add_special_tokens=True)["input_ids"]      # 4 ids
    pad = len(long_ids) - len(short_ids)
    assert first["input_ids"].tolist() == [long_ids, short_ids + [PAD] * pad]
    assert first["attention_mask"].tolist() == [[1] * len(long_ids), [1] * len(short_ids) + [0] * pad]
    assert forward_calls[1]["input_ids"].shape == (1, 5)          # last batch: one input, no padding


# ---------- 6. CPU / float32 / eval validation ----------

def test_valid_model_with_integer_buffers_passes():
    model = tiny_model()
    assert any(not buffer.dtype.is_floating_point for buffer in model.buffers())
    embedder, _ = make_embedder(model=model)
    embedder.embed_query("x")
    assert embedder.is_loaded


def _eager(model):
    return tiny_model(attn="eager")


def _float64(model):
    return model.double()


def _train_submodule(model):
    model.encoder.layer[0].train()
    return model


def _buffer_off_cpu(model):
    model.embeddings.position_ids = model.embeddings.position_ids.to("meta")
    return model


@pytest.mark.parametrize("break_model, message", [
    (_eager, "attention implementation"),
    (_float64, "float32 on cpu"),
    (_train_submodule, "training mode"),
    (_buffer_off_cpu, "buffers must be on cpu"),
], ids=["eager", "float64", "train-submodule", "buffer-off-cpu"])
def test_invalid_model_is_rejected_and_not_kept(break_model, message):
    embedder, _ = make_embedder(model=break_model(tiny_model()))
    with pytest.raises(EmbedderUnavailableError, match=message):
        embedder.embed_query("x")
    assert not embedder.is_loaded


# ---------- 7. concurrency ----------

def test_concurrent_first_use_loads_once_and_agrees():
    model = tiny_model()
    loads = []

    def slow_loader(config):
        loads.append(1)
        time.sleep(0.05)
        return FakeTokenizer(), model

    embedder = Embedder(EmbedderConfig("test/tiny", REVISION, 2), loader=slow_loader)
    barrier = threading.Barrier(8)
    results = [None] * 8

    def work(k):
        barrier.wait()
        results[k] = embedder.embed_passages(TEXTS)

    assert run_threads(work) == [] and len(loads) == 1
    for vectors in results[1:]:
        np.testing.assert_allclose(vectors, results[0], atol=1e-6)


def test_tokenizer_and_model_are_never_used_concurrently():
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def enter():
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.001)

    def leave():
        with guard:
            state["active"] -= 1

    class ProbedTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            enter()
            try:
                return super().__call__(text, **kwargs)
            finally:
                leave()

    model = tiny_model()
    model.register_forward_pre_hook(lambda module, args: enter())
    model.register_forward_hook(lambda module, args, output: leave())
    embedder, _ = make_embedder(tokenizer=ProbedTokenizer(), model=model)
    operations = [
        lambda: embedder.embed_passages(TEXTS[:3]),
        lambda: embedder.embed_query("question"),
        lambda: embedder.count_passage_tokens("some passage"),
        lambda: embedder.source_units("some raw text"),
    ]

    def work(k):
        for _ in range(5):
            operations[k % len(operations)]()

    assert run_threads(work) == [] and state["max"] == 1


# ---------- 8. signature and reproducibility ----------

def test_signature_fields():
    import tokenizers
    import transformers

    one, _ = make_embedder(batch_size=1)
    signature = one.signature_fields()
    assert signature == {
        "embedder_version": EMBEDDER_VERSION,
        "model_id": "test/tiny",
        "model_revision": REVISION,
        "weights_format": "safetensors",
        "embedding_dim": 384,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "pooling": "mean_over_attention_mask",
        "normalization": "l2",
        "max_input_tokens": 512,
        "special_tokens_counted": True,
        "truncation": "reject",
        "compute_dtype": "float32",
        "output_dtype": "float32",
        "device": "cpu",
        "attention_implementation": "sdpa",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
    }
    assert EMBEDDER_VERSION == "c2.4-v1"
    json.dumps(signature, sort_keys=True)                # storable as JSON (C2.5)
    eight, _ = make_embedder(batch_size=8)
    assert eight.signature_fields() == signature         # batch size is not part of the signature


def test_vectors_are_reproducible():
    embedder, _ = make_embedder()
    first = embedder.embed_passages(TEXTS)
    np.testing.assert_allclose(embedder.embed_passages(TEXTS), first, atol=1e-6)
    same_weights, _ = make_embedder(model=tiny_model(seed=0))
    np.testing.assert_allclose(same_weights.embed_passages(TEXTS), first, atol=1e-6)
    other_weights, _ = make_embedder(model=tiny_model(seed=1))
    assert not np.allclose(other_weights.embed_passages(TEXTS), first, atol=1e-3)


# ---------- 9. empty inputs and invalid input types ----------

def test_empty_inputs():
    embedder, _ = make_embedder()
    for empty in ([], ()):
        vectors = embedder.embed_passages(empty)
        assert vectors.shape == (0, EMBEDDING_DIM) and vectors.dtype == np.float32
    assert embedder.embed_passages([""]).shape == (1, EMBEDDING_DIM)   # "passage: " is still a real input
    assert embedder.embed_query("").shape == (EMBEDDING_DIM,)


def test_invalid_input_types_are_rejected_before_loading():
    embedder, loads = make_embedder()
    for call in (
        lambda: embedder.embed_passages("a string"),
        lambda: embedder.embed_passages(["ok", 7]),
        lambda: embedder.embed_query(None),
        lambda: embedder.count_passage_tokens(b"bytes"),
        lambda: embedder.source_units(None),
    ):
        with pytest.raises(EmbeddingInputError):
            call()
    assert loads == []


# ---------- configuration ----------

VALID_REQUIRED = {
    "database_url": "postgresql://user:password@localhost:5432/documind_test",
    "secret_key": "s" * 32,
}


@pytest.fixture
def clean_embedding_env(monkeypatch):
    for name in ("EMBEDDING_MODEL_ID", "EMBEDDING_MODEL_REVISION", "EMBEDDING_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)


def test_settings_embedding_defaults(clean_embedding_env):
    settings = Settings(_env_file=None, **VALID_REQUIRED)
    assert (settings.embedding_model_id, settings.embedding_model_revision, settings.embedding_batch_size) == (
        MODEL_ID, REVISION, 1)


@pytest.mark.parametrize("field, value", [
    ("embedding_model_id", " intfloat/multilingual-e5-small"),
    ("embedding_model_revision", "main"),
    ("embedding_batch_size", 0),
])
def test_settings_reject_invalid_embedding_value(clean_embedding_env, field, value):
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, **VALID_REQUIRED, **{field: value})
    assert [error["loc"] for error in excinfo.value.errors()] == [(field,)]


def test_embedder_config_validation():
    for args in ((" test/tiny", REVISION, 1), ("test/tiny", "main", 1), ("test/tiny", REVISION, 0),
                 ("test/tiny", REVISION, True)):
        with pytest.raises(ValueError):
            EmbedderConfig(*args)
    assert EmbedderConfig(MODEL_ID, REVISION, 1).model_id == MODEL_ID


# ---------- 10. source units: om-v1 and the weighted chunker ----------

def test_source_units_group_metaspace_pairs_for_the_weighted_chunker():
    text = "ab cd"
    tokenizer = FakeTokenizer(output={"input_ids": [7, 8, 9], "offset_mapping": [(0, 1), (0, 2), (3, 5)]},
                              tokens=[METASPACE, "ab", "cd"])
    embedder, _ = make_embedder(tokenizer=tokenizer)
    mapping = embedder.source_units(text)
    assert [(u.start, u.end, u.token_count) for u in mapping.units] == [(0, 2, 2), (3, 5, 1)]
    assert mapping.total_tokens == 3 and mapping.metaspace_groups == 1
    chunks = chunk_text(text, [(u.start, u.end) for u in mapping.units],
                        ChunkerConfig(target_tokens=4, max_tokens=4, overlap_tokens=0),
                        token_counts=[u.token_count for u in mapping.units])
    assert [(c.text, c.token_start, c.token_end) for c in chunks] == [("ab cd", 0, 3)]


@pytest.mark.parametrize("output, message", [
    ({"input_ids": [5, 6]}, "no 'offset_mapping'"),
    ({"input_ids": [5], "offset_mapping": [(0, 2), (3, 5)]}, "1 input_ids but 2 offsets"),
], ids=["missing-offsets", "length-mismatch"])
def test_source_units_reject_malformed_tokenizer_output(output, message):
    embedder, _ = make_embedder(tokenizer=FakeTokenizer(output=output))
    with pytest.raises(EmbeddingOutputError, match=message):
        embedder.source_units("ab cd")


def test_source_units_propagate_om_v1_errors_unchanged():
    tokenizer = FakeTokenizer(output={"input_ids": [5, 6], "offset_mapping": [(0, 6), (3, 11)]})
    embedder, _ = make_embedder(tokenizer=tokenizer)
    with pytest.raises(OffsetMappingError) as excinfo:
        embedder.source_units("secret text")
    assert excinfo.value.reason == "unverified_overlap"
    assert not isinstance(excinfo.value, EmbedderError) and "secret" not in str(excinfo.value)


# ---------- 11. the pinned model (skipped when it is not in the local cache) ----------

@pytest.fixture(scope="module")
def offline_hub():
    with pytest.MonkeyPatch.context() as patch:          # restored after this module
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        yield


@pytest.fixture(scope="module")
def pinned_tokenizer(offline_hub):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True,
                                             trust_remote_code=False)
    except (OSError, ValueError) as exc:
        pytest.skip(f"pinned tokenizer not in the local cache ({type(exc).__name__})")


@pytest.fixture(scope="module")
def pinned(pinned_tokenizer):
    embedder = Embedder(EmbedderConfig(MODEL_ID, REVISION, 1))
    try:
        embedder.signature_fields()                      # loads and validates
    except EmbedderUnavailableError as exc:
        if isinstance(exc.__cause__, OSError):           # weights missing from the cache
            pytest.skip(f"pinned model weights not in the local cache ({type(exc.__cause__).__name__})")
        raise                                            # anything else is a real failure
    return embedder


def test_pinned_model_loads_and_counts_like_its_tokenizer(pinned, pinned_tokenizer):
    signature = pinned.signature_fields()
    assert signature["attention_implementation"] == "sdpa" and signature["model_revision"] == REVISION
    for text in ["Hello world", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"]:
        ids = pinned_tokenizer(PASSAGE_PREFIX + text)["input_ids"]
        assert pinned.count_passage_tokens(text) == len(ids)
        assert ids[0] == pinned_tokenizer.cls_token_id and ids[-1] == pinned_tokenizer.sep_token_id
    vectors = pinned.embed_passages(["Hello world"])
    assert vectors.shape == (1, EMBEDDING_DIM) and abs(float(np.linalg.norm(vectors[0])) - 1.0) <= 1e-5


def test_pinned_multilingual_sanity(pinned):
    query = pinned.embed_query("How does PostgreSQL store table rows?")
    related, unrelated = pinned.embed_passages([
        "PostgreSQL stores table rows in heap files.",
        "The recipe needs two cups of flour and an egg.",
    ])
    assert float(query @ related) > float(query @ unrelated)
    hindi, english, other = pinned.embed_passages([
        "\u0906\u091c \u092e\u094c\u0938\u092e \u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u093e \u0939\u0948",
        "The weather is very nice today.",
        "The invoice must be paid within thirty days.",
    ])
    assert float(hindi @ english) > float(hindi @ other)
