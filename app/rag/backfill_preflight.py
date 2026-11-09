"""Embedding preflight for the indexing backfill (Component 12 / C5.5, CP6a).

Before the backfill changes any document it must know the embedding model can really
load offline and produce a vector. The preflight does that once, with the exact embedder
object the indexing loop will then reuse, so the model is loaded a single time.

What it does, in order:
  1. set HF_HUB_OFFLINE=1 (the model is only ever read from the local cache);
  2. check the embedder's id equals the id the plan was classified with;
  3. signature_fields()   - this loads the model;
  4. embed_passages(["preflight"]);
  5. check the result is one finite floating-point vector of the size the signature declares.

Any failure raises PreflightError carrying a stage and, at most, an exception CLASS
name. The original exception text is not exposed: it is not in the error's message and
the original is not chained as its cause (``from None``). It remains reachable through
``__context__`` but is suppressed when a traceback is printed. KeyboardInterrupt and
other BaseExceptions are not caught.

This module imports only numpy and the standard library: it must not import torch,
transformers or the embedder module, so setting the offline variable here always
happens before those libraries are imported by anything this module triggers.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

OFFLINE_VARIABLE = "HF_HUB_OFFLINE"
PREFLIGHT_TEXT = "preflight"

# Libraries that read HF_HUB_OFFLINE when they are first imported.
_OFFLINE_AWARE_MODULES = ("transformers", "huggingface_hub")


class PreflightEmbedder(Protocol):
    """The three members of Embedder the preflight uses."""

    @property
    def embedding_id(self) -> str: ...

    def signature_fields(self) -> Mapping[str, object]: ...

    def embed_passages(self, texts: Sequence[str]) -> object: ...


class PreflightStage(str, Enum):
    IDENTITY = "identity"
    SIGNATURE = "signature"
    EMBEDDING = "embedding"
    RESULT = "result"


class PreflightError(Exception):
    """The preflight failed. Holds the stage and an exception class name, never exception text."""

    def __init__(self, stage: PreflightStage, error_type: str | None = None) -> None:
        self.stage = stage
        self.error_type = error_type
        message = f"embedding preflight failed at the {stage.value} step"
        if error_type:
            message += f" ({error_type})"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """A successful preflight. ``embedder`` is the object to reuse for every document."""

    embedder: PreflightEmbedder
    embedding_id: str
    embedding_dim: int
    offline_set_before_imports: bool


def _embedding_dim(signature: object) -> int:
    if not isinstance(signature, Mapping):
        raise PreflightError(PreflightStage.SIGNATURE)
    dim = signature.get("embedding_dim")
    if type(dim) is not int or dim < 1:  # type(...) is int: a bool or a str is not a size
        raise PreflightError(PreflightStage.SIGNATURE)
    return dim


def _check_vectors(vectors: object, dim: int) -> None:
    if getattr(vectors, "shape", None) != (1, dim):
        raise PreflightError(PreflightStage.RESULT)
    dtype = getattr(vectors, "dtype", None)
    try:
        is_float = dtype is not None and bool(np.issubdtype(dtype, np.floating))
    except TypeError:  # not a dtype numpy understands
        is_float = False
    if not is_float:
        raise PreflightError(PreflightStage.RESULT)
    try:
        finite = bool(np.isfinite(vectors).all())
    except Exception:  # noqa: BLE001 - defensive: a floating array should never fail here
        raise PreflightError(PreflightStage.RESULT) from None
    if not finite:
        raise PreflightError(PreflightStage.RESULT)

def run_preflight(
    embedder: PreflightEmbedder,
    *,
    expected_embedding_id: str,
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
) -> PreflightResult:
    """Prove ``embedder`` loads offline and embeds one text. Raises PreflightError otherwise."""
    env = os.environ if environ is None else environ
    imported = sys.modules if modules is None else modules
    already_imported = any(name in imported for name in _OFFLINE_AWARE_MODULES)

    # 1. Offline only. Forced (not setdefault): an inherited "0" must not win.
    env[OFFLINE_VARIABLE] = "1"

    # 2. The plan was classified under expected_embedding_id; this embedder must be that one.
    try:
        actual_id = embedder.embedding_id
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(PreflightStage.IDENTITY, type(exc).__name__) from None
    if actual_id != expected_embedding_id:
        raise PreflightError(PreflightStage.IDENTITY)

    # 3. The signature. This is the call that loads the model.
    try:
        signature = embedder.signature_fields()
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(PreflightStage.SIGNATURE, type(exc).__name__) from None
    dim = _embedding_dim(signature)

    # 4. One real embedding.
    try:
        vectors = embedder.embed_passages([PREFLIGHT_TEXT])
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(PreflightStage.EMBEDDING, type(exc).__name__) from None

    # 5. The result is what the indexing loop will rely on.
    _check_vectors(vectors, dim)

    return PreflightResult(
        embedder=embedder,
        embedding_id=actual_id,
        embedding_dim=dim,
        offline_set_before_imports=not already_imported,
    )