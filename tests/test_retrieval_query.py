"""C6.3 retrieve_for_query: query embedding integration.

A  wiring and call order                  E  faulty embedder outputs
B  no embedding on invalid input          F  default get_embedder() vs injected
C  empty document_ids short-circuit       G  real PostgreSQL round trip (fake embedder)
D  embedder errors propagate unchanged    H  real pinned model smoke test

Groups A-F use fakes and never load a model. Group H uses the pinned model and is
skipped only when it is not in the local cache. Embedding quality and ranking
quality are not tested here (E1). Every database helper opens a short-lived
session and closes it, so nothing holds a lock when conftest TRUNCATEs the tables.
"""
import inspect
import math
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import event

from app.core.database import SessionLocal, engine
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.rag import embedder as emb
from app.rag import retrieval as r
from app.rag.chunking import sha256_text

EMBEDDING_ID = "test-model@" + "0" * 40
OTHER_EMBEDDING_ID = "other-model@" + "1" * 40
TEXT = " ".join(f"word{k:03d}" for k in range(80))
UNICODE_QUERY = "  \u0928\u092e\u0938\u094d\u0924\u0947 world \U0001F600 \n"
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
_UNSET = object()


# ---------- helpers ----------

def direction(*pairs):
    """A unit float32 vector pointing along the given (index, weight) pairs."""
    vector = np.zeros(emb.EMBEDDING_DIM, dtype=np.float32)
    for index, weight in pairs:
        vector[index] = weight
    return vector / np.linalg.norm(vector)


Q = direction((0, 1))


def with_value(position, value):
    vector = Q.copy()
    vector[position] = value
    return vector


def scaled(factor):
    return (Q * np.float32(factor)).astype(np.float32)


class FakeEmbedder:
    """Records every use in ``events`` ("id" when embedding_id is read, "embed" per query)."""

    def __init__(self, vector=_UNSET, embedding_id=_UNSET, error=None, events=None):
        self.vector = Q if vector is _UNSET else vector
        self._embedding_id = EMBEDDING_ID if embedding_id is _UNSET else embedding_id
        self.error = error
        self.events = [] if events is None else events
        self.queries = []

    @property
    def embedding_id(self):
        self.events.append("id")
        return self._embedding_id

    def embed_query(self, text):
        self.events.append("embed")
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        return self.vector


class FalsyEmbedder(FakeEmbedder):
    def __bool__(self):
        return False


class ForbiddenDb:
    """Any attribute access fails: proves the database (or a commit/rollback) was not used."""

    def __getattr__(self, name):
        raise AssertionError(f"the database must not be used (db.{name})")


class RecordingDb:
    """Accepts exactly one kind of use, execute(...).all(), and logs it as "db"."""

    def __init__(self, events):
        self.events = events
        self.statements = []

    def execute(self, statement):
        self.events.append("db")
        self.statements.append(statement)
        return SimpleNamespace(all=list)

    def __getattr__(self, name):
        raise AssertionError(f"unexpected db.{name}: retrieval must not control transactions")


def run(db, embedder, **overrides):
    arguments = {"owner_id": 1, "query": "a question", "embedder": embedder}
    arguments.update(overrides)
    return r.retrieve_for_query(db, **arguments)


_BOUND_AT_IMPORT = r.get_embedder                       # before any test patches it


@pytest.fixture(autouse=True)
def _default_embedder_is_forbidden(monkeypatch):
    """No test may reach the real default embedder by accident. Group F overrides this
    inside the test body, which runs after fixtures, so its patch wins."""

    def forbidden():
        raise AssertionError("get_embedder() must not be called in this test")

    monkeypatch.setattr(r, "get_embedder", forbidden)


@pytest.fixture
def spy_retrieve_chunks(monkeypatch):
    seen = []
    result = [object()]

    def spy(db, **kwargs):
        seen.append((db, kwargs))
        return result

    monkeypatch.setattr(r, "retrieve_chunks", spy)
    return SimpleNamespace(seen=seen, result=result)


# ---------- A. wiring and call order ----------

def test_a_the_signature_matches_the_approved_contract():
    parameters = inspect.signature(r.retrieve_for_query).parameters
    assert list(parameters) == ["db", "owner_id", "query", "embedder", "top_k", "document_ids"]
    assert parameters["db"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ["owner_id", "query", "embedder", "top_k", "document_ids"]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["owner_id"].default is inspect.Parameter.empty
    assert parameters["query"].default is inspect.Parameter.empty
    assert parameters["embedder"].default is None
    assert parameters["top_k"].default == r.DEFAULT_TOP_K
    assert parameters["document_ids"].default is None


def test_a_the_embedder_receives_exactly_the_validated_string(spy_retrieve_chunks):
    fake = FakeEmbedder()
    run(ForbiddenDb(), fake, query=UNICODE_QUERY)
    assert fake.queries == [UNICODE_QUERY]
    assert fake.queries[0] is UNICODE_QUERY                 # no independent strip or copy
    assert fake.queries[0] is r.validate_query_text(UNICODE_QUERY)


def test_a_the_vector_id_and_arguments_reach_retrieve_chunks(spy_retrieve_chunks):
    fake = FakeEmbedder()
    db = ForbiddenDb()
    result = run(db, fake, owner_id=7, top_k=3, document_ids=[4, 4, 9])
    assert result is spy_retrieve_chunks.result
    (seen_db, kwargs), = spy_retrieve_chunks.seen
    assert seen_db is db
    assert set(kwargs) == {"owner_id", "query_vector", "embedding_id", "top_k", "document_ids"}
    assert kwargs["query_vector"] is fake.vector
    assert kwargs["embedding_id"] == EMBEDDING_ID
    assert kwargs["owner_id"] == 7 and kwargs["top_k"] == 3
    assert kwargs["document_ids"] == (4, 9)                 # normalized and de-duplicated


def test_a_defaults_are_top_k_5_and_no_document_filter(spy_retrieve_chunks):
    run(ForbiddenDb(), FakeEmbedder())
    (_, kwargs), = spy_retrieve_chunks.seen
    assert kwargs["top_k"] == r.DEFAULT_TOP_K == 5
    assert kwargs["document_ids"] is None


def test_a_embedding_finishes_before_the_database_is_touched():
    fake = FakeEmbedder()
    db = RecordingDb(fake.events)
    assert run(db, fake) == []
    assert fake.events == ["id", "embed", "db"]             # id validated, then embed, then SQL
    assert len(db.statements) == 1


# ---------- B. no embedding on invalid input ----------

@pytest.mark.parametrize("query", ["", " ", "\n\t ", "\u2003\u00a0"])
def test_b_an_empty_query_raises_empty_query_error_without_embedding(query):
    fake = FakeEmbedder()
    with pytest.raises(r.EmptyQueryError):
        run(ForbiddenDb(), fake, query=query)
    assert fake.events == []


@pytest.mark.parametrize("query", [None, b"query", 5, ["query"]])
def test_b_a_non_str_query_raises_request_error_without_embedding(query):
    fake = FakeEmbedder()
    with pytest.raises(r.RetrievalRequestError) as caught:
        run(ForbiddenDb(), fake, query=query)
    assert not isinstance(caught.value, r.EmptyQueryError)
    assert fake.events == []


@pytest.mark.parametrize("owner_id", [True, False, "1", 1.0, None, np.int64(1)])
def test_b_an_invalid_owner_id_raises_without_embedding(owner_id):
    fake = FakeEmbedder()
    with pytest.raises(r.RetrievalRequestError):
        run(ForbiddenDb(), fake, owner_id=owner_id)
    assert fake.events == []


@pytest.mark.parametrize("top_k", [True, "5", 5.0, None, 0, -1, 21, 100])
def test_b_an_invalid_top_k_raises_without_embedding(top_k):
    fake = FakeEmbedder()
    with pytest.raises(r.RetrievalRequestError):
        run(ForbiddenDb(), fake, top_k=top_k)
    assert fake.events == []


@pytest.mark.parametrize(
    "document_ids",
    [
        "12",
        b"1",
        5,
        {1, 2},
        pytest.param((k for k in [1]), id="generator"),
        [True],
        ["1"],
        [1.5],
        list(range(101)),
    ],
)
def test_b_invalid_document_ids_raise_without_embedding(document_ids):
    fake = FakeEmbedder()
    with pytest.raises(r.RetrievalRequestError):
        run(ForbiddenDb(), fake, document_ids=document_ids)
    assert fake.events == []


@pytest.mark.parametrize(
    "overrides",
    [{"top_k": 1}, {"top_k": 20}, {"document_ids": list(range(100))}],
    ids=["top_k_1", "top_k_20", "100_document_ids"],
)
def test_b_boundary_values_are_accepted_and_do_embed(overrides):
    fake = FakeEmbedder()
    assert run(RecordingDb(fake.events), fake, **overrides) == []
    assert fake.events == ["id", "embed", "db"]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"query": " "}, r.EmptyQueryError),
        ({"owner_id": True}, r.RetrievalRequestError),
        ({"top_k": 0}, r.RetrievalRequestError),
    ],
)
def test_b_arguments_are_validated_before_the_empty_document_ids_short_circuit(overrides, error):
    fake = FakeEmbedder()
    with pytest.raises(error):
        run(ForbiddenDb(), fake, document_ids=[], **overrides)
    assert fake.events == []


# ---------- C. empty document_ids short-circuit ----------

@pytest.mark.parametrize("empty", [[], ()], ids=["list", "tuple"])
def test_c_empty_document_ids_return_nothing_without_embedding_or_db_or_default(empty):
    fake = FakeEmbedder()
    assert run(ForbiddenDb(), fake, document_ids=empty) == []
    assert fake.events == []
    # embedder=None: the forbidden default (autouse fixture) proves it was not resolved either
    assert r.retrieve_for_query(ForbiddenDb(), owner_id=1, query="q", document_ids=empty) == []


def test_c_a_real_embedder_is_neither_loaded_nor_used_for_empty_document_ids():
    loads = []

    def loader(config):
        loads.append(config)
        raise AssertionError("the model loader must not run")

    embedder = emb.Embedder(emb.EmbedderConfig("test/tiny", REVISION, 1), loader=loader)
    assert run(ForbiddenDb(), embedder, document_ids=[]) == []
    assert loads == [] and not embedder.is_loaded


# ---------- D. embedder errors propagate unchanged ----------

ERRORS = [
    emb.EmbeddingInputError("bad input"),
    emb.EmbeddingInputTooLongError(0, 513),
    emb.EmbedderUnavailableError("model unavailable"),
    emb.EmbeddingOutputError("bad output"),
    emb.EmbedderError("generic embedder failure"),
    RuntimeError("anything else"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda error: type(error).__name__)
def test_d_embedder_errors_propagate_as_the_same_object_without_db_access(error):
    fake = FakeEmbedder(error=error)
    with pytest.raises(type(error)) as caught:
        run(ForbiddenDb(), fake)
    assert caught.value is error
    assert fake.events == ["id", "embed"]


def test_d_a_real_embedder_load_failure_propagates_unchanged():
    error = emb.EmbedderUnavailableError("weights missing")
    loads = []

    def loader(config):
        loads.append(config)
        raise error

    embedder = emb.Embedder(emb.EmbedderConfig("test/tiny", REVISION, 1), loader=loader)
    with pytest.raises(emb.EmbedderUnavailableError) as caught:
        run(ForbiddenDb(), embedder)
    assert caught.value is error
    assert len(loads) == 1 and not embedder.is_loaded      # embedding_id alone never loads


# ---------- E. faulty embedder outputs are rejected by the existing validators ----------

BAD_VECTORS = [
    pytest.param(Q.astype(np.float64), "float32 with shape", id="float64"),
    pytest.param(Q[:383], "float32 with shape", id="too_short"),
    pytest.param(Q.reshape(1, -1), "float32 with shape", id="two_dimensional"),
    pytest.param(with_value(5, np.nan), "non-finite", id="nan"),
    pytest.param(with_value(5, np.inf), "non-finite", id="inf"),
    pytest.param(scaled(1.001), "unit length", id="norm_too_large"),
    pytest.param(np.zeros(emb.EMBEDDING_DIM, dtype=np.float32), "unit length", id="zero_vector"),
    pytest.param(Q.tolist(), "numpy array", id="python_list"),
    pytest.param(None, "numpy array", id="none"),
]


@pytest.mark.parametrize(("vector", "message"), BAD_VECTORS)
def test_e_an_invalid_vector_is_rejected_before_any_db_access(vector, message):
    fake = FakeEmbedder(vector=vector)
    with pytest.raises(r.RetrievalRequestError, match=message):
        run(ForbiddenDb(), fake)
    assert fake.events == ["id", "embed"]


def test_e_a_norm_within_tolerance_is_accepted():
    fake = FakeEmbedder(vector=scaled(1.0000001))
    assert run(RecordingDb(fake.events), fake) == []
    assert fake.events == ["id", "embed", "db"]


@pytest.mark.parametrize("embedding_id", ["", None, 5, b"model@rev"])
def test_e_an_invalid_embedding_id_is_rejected_before_embedding(embedding_id):
    fake = FakeEmbedder(embedding_id=embedding_id)
    with pytest.raises(r.RetrievalRequestError, match="non-empty str"):
        run(ForbiddenDb(), fake)
    assert fake.events == ["id"]


# ---------- F. default get_embedder() versus an injected embedder ----------

def counting_default(fake):
    calls = []

    def factory():
        calls.append(1)
        return fake

    return calls, factory


@pytest.mark.parametrize("extra", [{}, {"embedder": None}], ids=["omitted", "explicit_none"])
def test_f_without_an_injected_embedder_get_embedder_is_used_exactly_once(monkeypatch, extra):
    fake = FakeEmbedder()
    calls, factory = counting_default(fake)
    monkeypatch.setattr(r, "get_embedder", factory)
    assert r.retrieve_for_query(RecordingDb(fake.events), owner_id=1, query="q", **extra) == []
    assert calls == [1]
    assert fake.events == ["id", "embed", "db"] and fake.queries == ["q"]


def test_f_an_injected_embedder_is_used_and_get_embedder_is_not_called(monkeypatch):
    injected, default = FakeEmbedder(), FakeEmbedder()
    calls, factory = counting_default(default)
    monkeypatch.setattr(r, "get_embedder", factory)
    assert run(RecordingDb(injected.events), injected) == []
    assert calls == [] and default.events == []
    assert injected.events == ["id", "embed", "db"]


def test_f_a_falsy_but_valid_injected_embedder_is_still_used(monkeypatch):
    injected, default = FalsyEmbedder(), FakeEmbedder()
    calls, factory = counting_default(default)
    monkeypatch.setattr(r, "get_embedder", factory)
    assert not injected
    assert run(RecordingDb(injected.events), injected) == []
    assert calls == [] and injected.events == ["id", "embed", "db"]


def test_f_the_module_default_is_the_real_process_wide_get_embedder():
    assert _BOUND_AT_IMPORT is emb.get_embedder


# ---------- G. real PostgreSQL round trip with a fake embedder ----------

def spans(source, count):
    size = len(source) // count
    return [(i * size, (i + 1) * size if i < count - 1 else len(source)) for i in range(count)]


def new_user():
    with SessionLocal() as session:
        user = User(email=f"user-{uuid.uuid4().hex}@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        user_id = user.id
        session.commit()
    return user_id


def seed_document(owner_id, vectors, *, source=TEXT, embedding_id=EMBEDDING_ID):
    """A completed, fully published document with one chunk per vector (equal slices)."""
    values = {
        "filename": "a.txt",
        "owner_id": owner_id,
        "processing_status": "completed",
        "extracted_text": source,
        "indexing_status": "indexed",
        "indexed_source_version": 1,
        "source_sha256": sha256_text(source),
        "indexed_signature": "sig",
        "indexed_embedding_id": embedding_id,
        "indexed_at": datetime.now(UTC),
    }
    with SessionLocal() as session:
        document = Document(**values)
        session.add(document)
        session.flush()
        document_id = document.id
        for index, ((start, end), vector) in enumerate(zip(spans(source, len(vectors)), vectors)):
            piece = source[start:end]
            session.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=index,
                    start_offset=start,
                    end_offset=end,
                    page_start=None,
                    page_end=None,
                    text=piece,
                    span_sha256=sha256_text(piece),
                    embedding=vector,
                )
            )
        session.commit()
    return document_id


def retrieve(owner_id, embedder, **kwargs):
    arguments = {"query": "a question", **kwargs}
    with SessionLocal() as session:
        return r.retrieve_for_query(
            session, owner_id=owner_id, embedder=embedder, **arguments
        )


@pytest.fixture
def corpus():
    owner, other = new_user(), new_user()
    vectors = [direction((0, 1)), direction((0, 1), (1, 1)), direction((1, 1))]
    owned = seed_document(owner, vectors)
    foreign = seed_document(other, [direction((0, 1))])         # identical to the query vector
    return SimpleNamespace(owner=owner, other=other, owned=owned, foreign=foreign)


def test_g_it_returns_the_owners_chunks_exactly_like_retrieve_chunks(corpus):
    result = retrieve(corpus.owner, FakeEmbedder())
    with SessionLocal() as session:
        expected = r.retrieve_chunks(
            session, owner_id=corpus.owner, query_vector=Q, embedding_id=EMBEDDING_ID
        )
    assert result == expected
    assert [c.chunk_index for c in result] == [0, 1, 2]
    assert {c.document_id for c in result} == {corpus.owned}
    assert [c.score for c in result] == pytest.approx([1.0, 2 ** -0.5, 0.0], abs=1e-6)


def test_g_ownership_isolation_holds_through_the_query_path(corpus):
    owner_result = retrieve(corpus.owner, FakeEmbedder())
    other_result = retrieve(corpus.other, FakeEmbedder())
    assert corpus.foreign not in {c.document_id for c in owner_result}
    assert {c.document_id for c in other_result} == {corpus.foreign}


def test_g_a_foreign_document_id_returns_nothing_and_a_mixed_list_returns_only_owned(corpus):
    assert retrieve(corpus.owner, FakeEmbedder(), document_ids=[corpus.foreign]) == []
    mixed = retrieve(corpus.owner, FakeEmbedder(), document_ids=[corpus.owned, corpus.foreign])
    assert {c.document_id for c in mixed} == {corpus.owned}


def test_g_top_k_is_passed_through(corpus):
    assert [c.chunk_index for c in retrieve(corpus.owner, FakeEmbedder(), top_k=2)] == [0, 1]


def test_g_the_embedding_id_reaches_the_sql_filter(corpus):
    assert retrieve(corpus.owner, FakeEmbedder(embedding_id=OTHER_EMBEDDING_ID)) == []


def test_g_empty_document_ids_return_nothing_even_with_data(corpus):
    fake = FakeEmbedder()
    assert retrieve(corpus.owner, fake, document_ids=[]) == []
    assert fake.events == []


def test_g_embedding_precedes_one_select_and_nothing_is_committed(corpus):
    fake = FakeEmbedder()
    events, statements, commits = fake.events, [], []

    def on_execute(conn, cursor, statement, parameters, context, executemany):
        events.append("db")
        statements.append(statement)

    def on_commit(conn):
        commits.append(True)

    event.listen(engine, "before_cursor_execute", on_execute)
    event.listen(engine, "commit", on_commit)
    try:
        result = retrieve(corpus.owner, fake)
    finally:
        event.remove(engine, "before_cursor_execute", on_execute)
        event.remove(engine, "commit", on_commit)
    assert len(result) == 3
    assert events == ["id", "embed", "db"]
    assert statements[0].lstrip().upper().startswith("SELECT")
    assert commits == []


# ---------- H. real pinned model smoke test (wiring only, no quality assertions) ----------

@pytest.fixture(scope="module")
def pinned():
    with pytest.MonkeyPatch.context() as patch:          # restored after this module
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        embedder = emb.Embedder(emb.EmbedderConfig(MODEL_ID, REVISION, 1))
        try:
            embedder.signature_fields()                  # loads and validates
        except emb.EmbedderUnavailableError as exc:
            if isinstance(exc.__cause__, OSError):       # not in the local cache
                pytest.skip(f"pinned model not in the local cache ({type(exc.__cause__).__name__})")
            raise                                        # anything else is a real failure
        yield embedder


def test_h_the_real_model_embeds_the_query_and_the_stored_chunk_comes_back(pinned):
    passage = "PostgreSQL stores table rows in heap files."
    owner = new_user()
    document_id = seed_document(
        owner, [pinned.embed_passages([passage])[0]], source=passage, embedding_id=pinned.embedding_id
    )
    result = retrieve(owner, pinned, query="Where does PostgreSQL store table rows?")
    assert [c.document_id for c in result] == [document_id]
    assert result[0].text == passage
    assert math.isfinite(result[0].score)