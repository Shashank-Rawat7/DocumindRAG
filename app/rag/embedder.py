"""Pinned multilingual-e5-small embedder (Component 12 / C2.4).

One process-wide component that owns the pinned tokenizer and model:

* Loading is offline only (local Hugging Face cache, ``local_files_only=True``),
  at a pinned commit, safetensors weights, no remote code, float32, CPU, and an
  explicitly requested SDPA attention implementation that is verified after
  loading. Nothing is ever downloaded; a missing cache raises
  EmbedderUnavailableError.
* Each input is tokenized exactly once (prefix + text, special tokens added,
  no truncation, no padding). Inputs over MAX_INPUT_TOKENS are rejected before
  the model runs. Batches are built from those same token ids with manual
  right-padding and an explicit attention mask; only ``input_ids`` and
  ``attention_mask`` are passed to the model.
* Inference runs in eval mode under ``torch.inference_mode()``; vectors are
  attention-mask mean pooled, L2-normalised and returned as float32 arrays that
  are checked for shape, finiteness and unit norm.
* Three locks: one for creating the process-wide instance, one for loading the
  model (exactly once, retried after a failed load), one serialising every use
  of the shared tokenizer and model (tokenization included).

torch and transformers are imported lazily (inside functions) so importing this
module is cheap; the C2.3d benchmark measured 5-15 s for those imports.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.rag.token_offsets import OffsetMapping, map_token_offsets

EMBEDDING_DIM = 384
MAX_INPUT_TOKENS = 512                 # prefix + content + special tokens
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
EMBEDDER_VERSION = "c2.4-v1"           # bump when pooling/normalisation/validation code changes
ATTN_IMPLEMENTATION = "sdpa"
MODEL_INPUT_NAMES = ["input_ids", "attention_mask"]
NORM_TOLERANCE = 1e-5

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


# ---------- errors ----------

class EmbedderError(Exception):
    """Base class for embedder failures."""


class EmbedderUnavailableError(EmbedderError):
    """The pinned model cannot be loaded or failed a post-load check."""


class EmbeddingInputError(EmbedderError):
    """Inputs of the wrong type."""


class EmbeddingInputTooLongError(EmbeddingInputError):
    """An input exceeds MAX_INPUT_TOKENS (prefix and special tokens included)."""

    def __init__(self, index: int, token_count: int, limit: int = MAX_INPUT_TOKENS):
        self.index = index
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            f"input {index} has {token_count} tokens; the limit is {limit} "
            "(prefix and special tokens included); it is rejected, never truncated"
        )


class EmbeddingOutputError(EmbedderError):
    """The tokenizer or model produced something that fails validation."""


# ---------- configuration and loading ----------

@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    model_id: str
    revision: str
    batch_size: int

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id or self.model_id != self.model_id.strip():
            raise ValueError("model_id must be a non-empty str without leading or trailing whitespace")
        if type(self.revision) is not str or not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError("revision must be a 40-character lowercase hex commit hash")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be an int >= 1")

    @classmethod
    def from_settings(cls) -> "EmbedderConfig":
        from app.core.config import settings

        return cls(
            model_id=settings.embedding_model_id,
            revision=settings.embedding_model_revision,
            batch_size=settings.embedding_batch_size,
        )


Loader = Callable[[EmbedderConfig], tuple[Any, Any]]   # -> (tokenizer, model)


def load_pinned_model(config: EmbedderConfig) -> tuple[Any, Any]:
    """Load the pinned tokenizer and model from the local cache only."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise EmbedderUnavailableError(
            f"embedding dependencies are not importable ({type(exc).__name__})"
        ) from exc

    common = dict(revision=config.revision, local_files_only=True, trust_remote_code=False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
        model = AutoModel.from_pretrained(
            config.model_id,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation=ATTN_IMPLEMENTATION,
            **common,
        )
    except (OSError, ValueError, ImportError) as exc:
        raise EmbedderUnavailableError(
            f"cannot load {config.model_id}@{config.revision} from the local cache "
            f"({type(exc).__name__})"
        ) from exc
    model.eval()
    return tokenizer, model


# ---------- the embedder ----------

class Embedder:
    """Thread-safe, lazily loaded embedder. See the module docstring."""

    def __init__(self, config: EmbedderConfig, loader: Loader = load_pinned_model) -> None:
        if not isinstance(config, EmbedderConfig):
            raise TypeError("config must be an EmbedderConfig")
        self._config = config
        self._loader = loader
        self._load_lock = threading.Lock()        # model loading, exactly once
        self._inference_lock = threading.Lock()   # every use of tokenizer + model
        self._tokenizer: Any = None
        self._model: Any = None
        self._attn_implementation: str | None = None

    @property
    def config(self) -> EmbedderConfig:
        return self._config

    @property
    def embedding_id(self) -> str:
        """Value for documents.indexed_embedding_id."""
        return f"{self._config.model_id}@{self._config.revision}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ----- public operations -----

    def signature_fields(self) -> dict:
        """Everything that changes the stored vectors or which inputs are accepted.

        Batch size and torch thread count are deliberately excluded: with the
        attention mask they only cause floating-point-level differences.
        """
        self._ensure_loaded()
        import tokenizers
        import torch
        import transformers

        return {
            "embedder_version": EMBEDDER_VERSION,
            "model_id": self._config.model_id,
            "model_revision": self._config.revision,
            "weights_format": "safetensors",
            "embedding_dim": EMBEDDING_DIM,
            "passage_prefix": PASSAGE_PREFIX,
            "query_prefix": QUERY_PREFIX,
            "pooling": "mean_over_attention_mask",
            "normalization": "l2",
            "max_input_tokens": MAX_INPUT_TOKENS,
            "special_tokens_counted": True,
            "truncation": "reject",
            "compute_dtype": "float32",
            "output_dtype": "float32",
            "device": "cpu",
            "attention_implementation": self._attn_implementation,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
        }

    def source_units(self, text: str) -> OffsetMapping:
        """om-v1 source units for ``text`` (content tokens only, no prefix, no
        special tokens), ready for chunk_text(..., token_counts=...).

        Malformed tokenizer output raises EmbeddingOutputError; OffsetMappingError
        from map_token_offsets (om-v1 rule violations) propagates unchanged.
        """
        if type(text) is not str:
            raise EmbeddingInputError("text must be a str")
        tokenizer, _ = self._ensure_loaded()
        with self._inference_lock:
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                padding=False,
            )
            ids = _flat_int_sequence(_tokenizer_field(encoding, "input_ids"), "input_ids")
            raw_offsets = _tokenizer_field(encoding, "offset_mapping")
            if isinstance(raw_offsets, (str, bytes, bytearray)) or not isinstance(raw_offsets, Sequence):
                raise EmbeddingOutputError(
                    f"tokenizer offset_mapping must be a sequence of pairs, got {type(raw_offsets).__name__}"
                )
            if len(raw_offsets) != len(ids):
                raise EmbeddingOutputError(
                    f"tokenizer returned {len(ids)} input_ids but {len(raw_offsets)} offsets"
                )
            offsets = []
            for position, pair in enumerate(raw_offsets):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise EmbeddingOutputError(f"tokenizer offset_mapping[{position}] is not a pair")
                if type(pair[0]) is not int or type(pair[1]) is not int:
                    raise EmbeddingOutputError(f"tokenizer offset_mapping[{position}] is not a pair of ints")
                offsets.append((pair[0], pair[1]))
            tokens = tokenizer.convert_ids_to_tokens(ids)
            if isinstance(tokens, (str, bytes, bytearray)) or not isinstance(tokens, Sequence) \
                    or len(tokens) != len(ids) or any(type(token) is not str for token in tokens):
                raise EmbeddingOutputError(
                    "tokenizer convert_ids_to_tokens must return one str per input id"
                )
            tokens = list(tokens)
        return map_token_offsets(text, offsets, tokens)

    def count_passage_tokens(self, text: str) -> int:
        """Model-input tokens of PASSAGE_PREFIX + text, special tokens included."""
        if type(text) is not str:
            raise EmbeddingInputError("text must be a str")
        tokenizer, _ = self._ensure_loaded()
        with self._inference_lock:
            return len(self._encode(tokenizer, PASSAGE_PREFIX + text))

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """float32 array of shape (len(texts), 384), L2-normalised, input order."""
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise EmbeddingInputError("texts must be a sequence of str")
        for k, text in enumerate(texts):
            if type(text) is not str:
                raise EmbeddingInputError(f"texts[{k}] must be a str")
        if len(texts) == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return self._embed([PASSAGE_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """float32 array of shape (384,), L2-normalised."""
        if type(text) is not str:
            raise EmbeddingInputError("text must be a str")
        return self._embed([QUERY_PREFIX + text])[0]

    # ----- loading -----

    def _ensure_loaded(self) -> tuple[Any, Any]:
        model = self._model
        if model is not None:
            return self._tokenizer, model
        with self._load_lock:
            if self._model is None:
                tokenizer, model = self._loader(self._config)
                attn = _validate_loaded(tokenizer, model)
                self._tokenizer = tokenizer
                self._attn_implementation = attn
                self._model = model          # set last: the lock-free fast path checks it
        return self._tokenizer, self._model

    # ----- inference -----

    def _embed(self, prefixed_texts: list[str]) -> np.ndarray:
        tokenizer, model = self._ensure_loaded()
        with self._inference_lock:
            encoded = [self._encode(tokenizer, text) for text in prefixed_texts]
            for index, ids in enumerate(encoded):
                if len(ids) > MAX_INPUT_TOKENS:
                    raise EmbeddingInputTooLongError(index, len(ids))
            vectors = self._run_model(model, tokenizer.pad_token_id, encoded)
        _check_output(vectors, len(prefixed_texts))
        return vectors

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int]:
        """The single tokenization used for counting, validation and inference.

        Returns a non-empty list of non-negative Python ints, or raises
        EmbeddingOutputError naming what was wrong (never the text itself).
        """
        encoding = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        try:
            raw_ids = encoding["input_ids"]
        except (KeyError, TypeError, IndexError):
            raise EmbeddingOutputError(
                f"tokenizer output ({type(encoding).__name__}) has no 'input_ids'"
            ) from None
        if isinstance(raw_ids, (str, bytes, bytearray)) or not isinstance(raw_ids, Sequence):
            raise EmbeddingOutputError(
                f"tokenizer input_ids must be a flat list of ints, got {type(raw_ids).__name__}"
            )
        ids = list(raw_ids)
        if not ids:
            raise EmbeddingOutputError("tokenizer returned empty input_ids")
        for position, token_id in enumerate(ids):
            if type(token_id) is not int:
                raise EmbeddingOutputError(
                    f"tokenizer input_ids[{position}] is {type(token_id).__name__}, expected int"
                )
            if token_id < 0:
                raise EmbeddingOutputError(f"tokenizer input_ids[{position}] is negative")
        return ids

    def _run_model(self, model: Any, pad_token_id: int, encoded: list[list[int]]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = np.empty((len(encoded), EMBEDDING_DIM), dtype=np.float32)
        batch_size = self._config.batch_size
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            input_ids, attention_mask = _right_pad(batch, pad_token_id)
            with torch.inference_mode():
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                if tuple(hidden.shape) != (len(batch), input_ids.shape[1], EMBEDDING_DIM):
                    raise EmbeddingOutputError(
                        f"model returned hidden states of shape {tuple(hidden.shape)}"
                    )
                pooled = _masked_mean(hidden.to(torch.float32), attention_mask)
                normalized = F.normalize(pooled, p=2, dim=1)
            out[start:start + len(batch)] = normalized.to(torch.float32).cpu().numpy()
        return out


# ---------- helpers ----------

def _tokenizer_field(encoding: Any, name: str) -> Any:
    """encoding[name], or EmbeddingOutputError naming the missing field."""
    try:
        return encoding[name]
    except (KeyError, TypeError, IndexError):
        raise EmbeddingOutputError(
            f"tokenizer output ({type(encoding).__name__}) has no {name!r}"
        ) from None


def _flat_int_sequence(raw: Any, name: str) -> list[int]:
    """A flat sequence of non-negative Python ints (may be empty) as a list."""
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise EmbeddingOutputError(f"tokenizer {name} must be a flat list of ints, got {type(raw).__name__}")
    values = list(raw)
    for position, value in enumerate(values):
        if type(value) is not int:
            raise EmbeddingOutputError(f"tokenizer {name}[{position}] is {type(value).__name__}, expected int")
        if value < 0:
            raise EmbeddingOutputError(f"tokenizer {name}[{position}] is negative")
    return values


def _validate_loaded(tokenizer: Any, model: Any) -> str:
    """Post-load checks; returns the verified attention implementation."""
    import torch

    def fail(reason: str) -> None:
        raise EmbedderUnavailableError(f"loaded model failed validation: {reason}")

    config = getattr(model, "config", None)
    if config is None:
        fail("model has no config")
    if getattr(config, "model_type", None) != "bert":
        fail(f"model_type is {getattr(config, 'model_type', None)!r}, expected 'bert'")
    if getattr(config, "hidden_size", None) != EMBEDDING_DIM:
        fail(f"hidden_size is {getattr(config, 'hidden_size', None)!r}, expected {EMBEDDING_DIM}")
    if not isinstance(getattr(config, "max_position_embeddings", None), int) or \
            config.max_position_embeddings < MAX_INPUT_TOKENS:
        fail("max_position_embeddings is below the input limit")
    if getattr(config, "auto_map", None):
        fail("model config requests remote code (auto_map)")

    attn = getattr(config, "_attn_implementation", None)
    if type(attn) is not str or attn != ATTN_IMPLEMENTATION:
        fail(f"attention implementation is {attn!r}, expected {ATTN_IMPLEMENTATION!r}")
    for module in model.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and getattr(module_config, "_attn_implementation", None) != attn:
            fail("a submodule uses a different attention implementation")

    if any(module.training for module in model.modules()):
        fail("the model or one of its submodules is in training mode")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            fail("all parameters must be float32 on cpu")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            fail("all buffers must be on cpu")   # dtype not checked: integer buffers are valid

    if getattr(tokenizer, "is_fast", False) is not True:
        fail("tokenizer is not a fast tokenizer")
    if type(getattr(tokenizer, "pad_token_id", None)) is not int:
        fail("tokenizer has no integer pad_token_id")
    if list(getattr(tokenizer, "model_input_names", [])) != MODEL_INPUT_NAMES:
        fail(f"tokenizer model_input_names must be {MODEL_INPUT_NAMES}")
    return attn


def _right_pad(batch: list[list[int]], pad_token_id: int):
    """Right-padded input_ids and attention_mask (torch.long) built from the
    already validated token ids; real tokens keep positions 0..len-1."""
    import torch

    width = max(len(ids) for ids in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, ids in enumerate(batch):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
    return input_ids, attention_mask


def _masked_mean(hidden, attention_mask):
    """Mean over real tokens only: padded positions are replaced by 0 (not
    multiplied, so non-finite padding values cannot leak) and excluded from the count."""
    mask = attention_mask.bool().unsqueeze(-1)
    summed = hidden.masked_fill(~mask, 0.0).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1).to(hidden.dtype)
    return summed / counts


def _check_output(vectors: np.ndarray, expected_rows: int) -> None:
    if vectors.dtype != np.float32 or vectors.shape != (expected_rows, EMBEDDING_DIM):
        raise EmbeddingOutputError(f"embeddings have dtype {vectors.dtype} and shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise EmbeddingOutputError("embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
        raise EmbeddingOutputError("embeddings are not unit length")


# ---------- process-wide instance ----------

_SINGLETON_LOCK = threading.Lock()
_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """The process-wide Embedder built from settings (the model loads on first use)."""
    global _EMBEDDER
    embedder = _EMBEDDER
    if embedder is None:
        with _SINGLETON_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = Embedder(EmbedderConfig.from_settings())
            embedder = _EMBEDDER
    return embedder
