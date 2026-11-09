"""Tests for the backfill embedding preflight (C5.5 / CP6a).

No model is ever loaded: embedders are fakes, except one real Embedder whose loader
fails. The conftest database fixtures still apply, as for every test in this project.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.rag.backfill_preflight import (
    OFFLINE_VARIABLE,
    PREFLIGHT_TEXT,
    PreflightError,
    PreflightResult,
    PreflightStage,
    run_preflight,
)
from app.rag.embedder import Embedder, EmbedderConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ID = "test/fake@" + "0" * 40
SECRET = "s3cr3t-p4ssw0rd-XYZ"
DIM = 4


class FakeEmbedder:
    """The three members the preflight uses, recording calls and the environment it saw."""

    def __init__(self, *, embedding_id=EMBEDDING_ID, signature=None, vectors=None,
                 signature_error=None, embed_error=None, environ=None):
        self.embedding_id = embedding_id
        self.signature = {"embedding_dim": DIM} if signature is None else signature
        self.vectors = np.zeros((1, DIM), dtype=np.float32) if vectors is None else vectors
        self.signature_error = signature_error
        self.embed_error = embed_error
        self.environ = environ
        self.calls = []
        self.offline_seen = []

    def signature_fields(self):
        self.calls.append(("signature_fields",))
        if self.environ is not None:
            self.offline_seen.append(self.environ.get(OFFLINE_VARIABLE))
        if self.signature_error is not None:
            raise self.signature_error
        return self.signature

    def embed_passages(self, texts):
        self.calls.append(("embed_passages", list(texts)))
        if self.embed_error is not None:
            raise self.embed_error
        return self.vectors


def preflight(embedder, **overrides):
    options = dict(expected_embedding_id=EMBEDDING_ID, environ={}, modules={})
    options.update(overrides)
    return run_preflight(embedder, **options)


# ---------- success ----------

def test_success_returns_the_same_embedder_and_the_facts():
    embedder = FakeEmbedder()
    result = preflight(embedder)
    assert isinstance(result, PreflightResult)
    assert result.embedder is embedder
    assert result.embedding_id == EMBEDDING_ID
    assert result.embedding_dim == DIM
    assert result.offline_set_before_imports is True


def test_the_dimension_comes_from_the_signature_not_a_constant():
    embedder = FakeEmbedder(signature={"embedding_dim": 7}, vectors=np.zeros((1, 7), dtype=np.float32))
    assert preflight(embedder).embedding_dim == 7


def test_the_embedder_is_called_in_order_exactly_once_with_the_approved_text():
    embedder = FakeEmbedder()
    preflight(embedder)
    assert embedder.calls == [("signature_fields",), ("embed_passages", [PREFLIGHT_TEXT])]
    assert PREFLIGHT_TEXT == "preflight"


# ---------- offline ----------

def test_offline_is_set_before_the_model_is_touched():
    environ = {}
    embedder = FakeEmbedder(environ=environ)
    preflight(embedder, environ=environ)
    assert embedder.offline_seen == ["1"]
    assert environ[OFFLINE_VARIABLE] == "1"


def test_an_inherited_value_is_overridden():
    environ = {OFFLINE_VARIABLE: "0"}
    preflight(FakeEmbedder(environ=environ), environ=environ)
    assert environ[OFFLINE_VARIABLE] == "1"


def test_the_default_environment_is_the_process_environment(monkeypatch):
    monkeypatch.setenv(OFFLINE_VARIABLE, "0")  # restored (or removed) by monkeypatch afterwards
    run_preflight(FakeEmbedder(), expected_embedding_id=EMBEDDING_ID, modules={})
    assert os.environ[OFFLINE_VARIABLE] == "1"


def test_offline_is_still_set_when_the_preflight_fails():
    environ = {}
    with pytest.raises(PreflightError):
        preflight(FakeEmbedder(embedding_id="other@id"), environ=environ)
    assert environ[OFFLINE_VARIABLE] == "1"


@pytest.mark.parametrize("module", ["transformers", "huggingface_hub"])
def test_libraries_imported_earlier_are_reported(module):
    result = preflight(FakeEmbedder(), modules={module: object()})
    assert result.offline_set_before_imports is False


def test_unrelated_imported_modules_do_not_matter():
    assert preflight(FakeEmbedder(), modules={"numpy": object()}).offline_set_before_imports is True

@pytest.mark.parametrize("module", ["transformers", "huggingface_hub"])
def test_the_default_modules_are_the_real_sys_modules(monkeypatch, module):
    # Nothing imports these libraries during the test, and monkeypatch restores sys.modules.
    monkeypatch.setitem(sys.modules, module, object())
    result = run_preflight(FakeEmbedder(), expected_embedding_id=EMBEDDING_ID, environ={})
    assert result.offline_set_before_imports is False


def test_the_default_modules_report_true_when_the_libraries_are_absent(monkeypatch):
    for module in ("transformers", "huggingface_hub"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    result = run_preflight(FakeEmbedder(), expected_embedding_id=EMBEDDING_ID, environ={})
    assert result.offline_set_before_imports is True

# ---------- identity ----------

def test_a_different_embedding_id_fails_before_the_model_is_touched():
    embedder = FakeEmbedder(embedding_id="other/model@" + "1" * 40)
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.IDENTITY
    assert info.value.error_type is None
    assert embedder.calls == []

def test_an_embedding_id_that_raises_is_an_identity_failure_with_only_the_class_name():
    class RaisingId:
        @property
        def embedding_id(self):
            raise LookupError(SECRET)

        def signature_fields(self):
            raise AssertionError("the model must not be touched")

        def embed_passages(self, texts):
            raise AssertionError("the model must not be touched")

    with pytest.raises(PreflightError) as info:
        preflight(RaisingId())
    assert info.value.stage is PreflightStage.IDENTITY
    assert info.value.error_type == "LookupError"
    assert SECRET not in str(info.value)

# ---------- failures at each step ----------

def test_a_signature_error_is_reported_by_class_name_only():
    embedder = FakeEmbedder(signature_error=RuntimeError(f"cannot load {SECRET}"))
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.SIGNATURE
    assert info.value.error_type == "RuntimeError"
    assert SECRET not in str(info.value)
    assert embedder.calls == [("signature_fields",)]  # no embedding attempt after a failed load


def test_an_embedding_error_is_reported_by_class_name_only():
    embedder = FakeEmbedder(embed_error=ValueError(SECRET))
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.EMBEDDING
    assert info.value.error_type == "ValueError"
    assert SECRET not in str(info.value)


def test_the_original_exception_is_not_chained():
    embedder = FakeEmbedder(embed_error=ValueError(SECRET))
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "signature",
    [None, [], {}, {"embedding_dim": 0}, {"embedding_dim": -1}, {"embedding_dim": "4"},
     {"embedding_dim": True}, {"embedding_dim": 4.0}],
)
def test_a_malformed_signature_fails_before_any_embedding(signature):
    embedder = FakeEmbedder()
    embedder.signature = signature
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.SIGNATURE
    assert info.value.error_type is None
    assert embedder.calls == [("signature_fields",)]


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((2, DIM), dtype=np.float32),
        np.zeros((1, DIM - 1), dtype=np.float32),
        np.zeros(DIM, dtype=np.float32),
        np.zeros((1, DIM, 1), dtype=np.float32),
        np.zeros((0, DIM), dtype=np.float32),
        [[0.0] * DIM],
        np.array([[np.nan, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[np.inf, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([["a", "b", "c", "d"]]),
        np.ones((1, DIM), dtype=np.int64),
        np.ones((1, DIM), dtype=bool),
        np.ones((1, DIM), dtype=np.complex64),
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=object),
    ],
    ids=["two-rows", "short", "one-dimensional", "three-dimensional", "empty", "plain-list", "nan", "inf",
         "strings", "integers", "booleans", "complex", "objects"],)
def test_a_bad_result_fails_at_the_result_step(vectors):
    embedder = FakeEmbedder()
    embedder.vectors = vectors
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.RESULT


def test_a_none_result_fails_at_the_result_step():
    embedder = FakeEmbedder()
    embedder.vectors = None
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.RESULT

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_any_floating_dtype_is_accepted(dtype):
    embedder = FakeEmbedder(vectors=np.zeros((1, DIM), dtype=dtype))
    assert preflight(embedder).embedding_dim == DIM


def test_a_result_with_an_unreadable_dtype_fails_at_the_result_step():
    class Odd:
        shape = (1, DIM)
        dtype = "not a dtype"

    embedder = FakeEmbedder()
    embedder.vectors = Odd()
    with pytest.raises(PreflightError) as info:
        preflight(embedder)
    assert info.value.stage is PreflightStage.RESULT

# ---------- interrupts are never swallowed ----------

@pytest.mark.parametrize("where", ["signature_error", "embed_error"])
@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3)])
def test_base_exceptions_propagate_unchanged(where, exc):
    embedder = FakeEmbedder(**{where: exc})
    with pytest.raises(type(exc)) as info:
        preflight(embedder)
    assert info.value is exc

# ---------- the message ----------

def test_the_error_message_is_fixed_ascii_text():
    for stage in PreflightStage:
        shown = str(PreflightError(stage, "SomeError"))
        assert shown.isascii()
        assert stage.value in shown and "SomeError" in shown
    assert str(PreflightError(PreflightStage.RESULT)) == "embedding preflight failed at the result step"


# ---------- a real Embedder, with a loader that fails ----------

def test_a_real_embedder_whose_loader_fails_is_reported_and_never_loaded():
    def loader(config):
        raise RuntimeError(f"no model: {SECRET}")

    embedder = Embedder(EmbedderConfig.from_settings(), loader=loader)
    with pytest.raises(PreflightError) as info:
        run_preflight(embedder, expected_embedding_id=embedder.embedding_id, environ={}, modules={})
    assert info.value.stage is PreflightStage.SIGNATURE
    assert info.value.error_type == "RuntimeError"
    assert SECRET not in str(info.value)
    assert embedder.is_loaded is False


# ---------- the module itself ----------

def test_importing_the_module_loads_no_machine_learning_library():
    code = (
        "import sys\n"
        "import app.rag.backfill_preflight\n"
        "bad = [m for m in ('torch', 'transformers', 'huggingface_hub', 'tokenizers') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr