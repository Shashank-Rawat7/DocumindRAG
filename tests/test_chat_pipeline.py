"""C9.5b slice 2: the orchestration (chat_pipeline.run_chat_pipeline).

Runs against documind_test with a ScriptedProvider and a fake embedder. Every provider outcome
is checked against the rows it must leave behind (attempt status, charge, request status,
messages), and each test pins that exactly one settlement path ran.
"""

import hashlib
import logging

import numpy as np
import pytest
from pydantic import SecretStr
from sqlalchemy import event, select

import app.rag.chat_pipeline as pipeline
from app.core.chat_config import ChatSettings
from app.core.database import SessionLocal
from app.rag import chat_error_codes as codes
from app.rag.chat_attempt_lifecycle import TransitionOutcome
from app.rag.chat_pipeline import (
    FailureKind,
    PipelineArgumentError,
    PipelineOutcome,
    RecoveryOutcome,
    RecoveryResult,
    recover_request,
    run_chat_pipeline,
)
from app.rag.chat_prompt import PromptBuildError
from app.rag.chat_provider import (
    ProviderUsage,
    completed,
    rate_limited,
    rejected,
    uncertain,
    unusable,
)
from app.rag.chat_request_finalization import PublishOutcome, PublishResult
from app.rag.embedder import (
    EmbedderUnavailableError,
    EmbeddingInputError,
    EmbeddingInputTooLongError,
)
from app.rag.retrieval import RetrievalRequestError
from tests.chat_fakes import ScriptedProvider
from tests.chat_invariants import (
    assert_every_request_valid,
    assert_valid,
    state_of,
)
from tests.chat_pipeline_db import (
    add_attempt,
    attempt_history,
    execute,
    expire_lease,
    make_request,
    make_user,
    request_row,
    rows,
    snapshot,
)

P = PublishOutcome
SETTINGS = ChatSettings(
    groq_api_key=SecretStr("not-a-real-key"), daily_token_limit=10_000_000, model="m"
)
QUESTION = "What does chunk number 0 say?"
SECRET_ANSWER = "ANSWER-TEXT-MUST-NOT-LEAK"
SECRET_EXC = "EXCEPTION-TEXT-MUST-NOT-LEAK"
USAGE = ProviderUsage(prompt_tokens=50, completion_tokens=20)


# ---------------------------------------------------------------- fixtures and helpers


class FakeEmbedder:
    """Lazy like the real one: ``on_load`` runs on the first embed, ``on_embed`` on every one."""

    embedding_id = "emb"

    def __init__(self, on_load=None, on_embed=None, error=None):
        self._on_load = on_load
        self._on_embed = on_embed
        self._error = error
        self.loaded = False
        self.calls = []

    def embed_query(self, text):
        if not self.loaded:
            self.loaded = True
            if self._on_load is not None:
                self._on_load()
        if self._on_embed is not None:
            self._on_embed()
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


def vector_literal(index):
    values = [0.0] * 384
    values[0] = 1.0
    values[1] = 0.1 * index  # chunk 0 is nearest to the query, then 1, then 2
    return "[" + ",".join(str(v) for v in values) + "]"


def make_world(owner, chunks=3):
    document = execute(
        "INSERT INTO documents (filename, owner_id, source_version, source_sha256,"
        " indexing_status, indexed_source_version, indexed_signature,"
        " indexed_embedding_id, indexed_at)"
        " VALUES ('d.txt', :o, 1, :s, 'indexed', 1, 'sig', 'emb', now()) RETURNING id",
        o=owner,
        s="b" * 64,
    ).scalar_one()
    ids = []
    for i in range(chunks):
        text = f"chunk number {i}"
        ids.append(
            execute(
                "INSERT INTO document_chunks (document_id, chunk_index, start_offset,"
                " end_offset, text, span_sha256, embedding)"
                " VALUES (:d, :i, :s, :e, :t, :h, CAST(:v AS vector)) RETURNING id",
                d=document,
                i=i,
                s=i * 100,
                e=i * 100 + len(text),
                t=text,
                h=hashlib.sha256(text.encode()).hexdigest(),
                v=vector_literal(i),
            ).scalar_one()
        )
    return ids


# A test that deliberately seeds a history no C8 service can produce (to see the pipeline refuse
# to act on it) carries ``@invalid_history_by_design``: the observer would rightly flag the seed.
# Nothing else is exempt, and the observer itself is never relaxed.
invalid_history_by_design = pytest.mark.invalid_history_by_design


@pytest.fixture(autouse=True)
def _every_history_is_legal_afterwards(request):
    """After every test, every request's complete history satisfies S1-S7 and I1-I8."""
    yield
    if request.node.get_closest_marker("invalid_history_by_design") is None:
        assert_every_request_valid()


@pytest.fixture
def owner():
    return make_user("alice@example.com")


@pytest.fixture
def world(owner):
    return make_world(owner)


def run(
    owner, rid, provider, embedder=None, question=QUESTION, settings=SETTINGS, **kw
):
    sleeps = kw.pop("sleeps", None)
    with SessionLocal() as db:
        return run_chat_pipeline(
            db,
            owner_id=owner,
            request_id=rid,
            question=question,
            provider=provider,
            settings=settings,
            embedder=embedder or FakeEmbedder(),
            sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
            **kw,
        )


def statuses(rid):
    return [(r.status, r.charged_tokens) for r in attempt_history(rid)]


def latest_attempt_id(rid):
    return attempt_history(rid)[-1].id


def message_count():
    return rows("SELECT count(*) FROM messages")[0][0]


def citation_count():
    return rows("SELECT count(*) FROM message_citations")[0][0]


def assert_failed(rid, code):
    row = request_row(rid)
    assert (row.status, row.error_code) == ("failed", code)
    assert message_count() == 0


# ---------------------------------------------------------------- success paths


def test_answer_with_citations_is_published_once(owner, world):
    rid = make_request(owner)
    provider = ScriptedProvider([completed("It says so [1] and also [2].", USAGE)])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.ANSWERED
    assert result.published.outcome is P.PUBLISHED
    assert len(provider.calls) == 1
    assert request_row(rid).status == "succeeded"
    assert statuses(rid) == [("succeeded", 70)]
    assert message_count() == 2
    assert citation_count() == 2
    _, user_message = provider.calls[0]
    assert "chunk number 0" in user_message and QUESTION in user_message
    stored = rows("SELECT role, content FROM messages ORDER BY seq")
    assert stored[0] == ("user", QUESTION)
    assert (
        stored[1][1] == "It says so [1] and also [2]."
    )  # valid markers stay in the text


def test_zero_citation_answer_is_published_with_no_citations(owner, world):
    """D9: an answer that cites nothing is a legitimate 'the sources do not say'."""
    rid = make_request(owner)
    provider = ScriptedProvider([completed("The sources do not answer this.", USAGE)])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.ANSWERED
    assert request_row(rid).status == "succeeded"
    assert citation_count() == 0
    assert message_count() == 2
    assert statuses(rid) == [("succeeded", 70)]


def test_success_without_usage_charges_the_reservation(owner, world):
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([completed("Fine [1].")]))
    ((status, charged),) = statuses(rid)
    assert (status, charged) == ("succeeded", attempt_history(rid)[0].reserved_tokens)


def test_replay_after_success_never_calls_the_provider(owner, world):
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([completed("Fine [1].", USAGE)]))
    before = snapshot()
    provider = ScriptedProvider([])
    again = run(owner, rid, provider)
    assert again.outcome is PipelineOutcome.REPLAYED_SUCCESS
    assert again.request.status == "succeeded"
    assert provider.calls == [] and snapshot() == before


def test_replay_after_failure_never_calls_the_provider(owner, world):
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([uncertain(codes.PROVIDER_TIMEOUT)]))
    before = snapshot()
    provider = ScriptedProvider([])
    again = run(owner, rid, provider)
    assert again.outcome is PipelineOutcome.REPLAYED_FAILURE
    assert again.error_code == codes.PROVIDER_TIMEOUT
    assert provider.calls == [] and snapshot() == before


# ---------------------------------------------------------------- one settlement per outcome


def test_unusable_with_usage_is_failed_with_usage(owner, world):
    rid = make_request(owner)
    result = run(owner, rid, ScriptedProvider([unusable(ProviderUsage(7, 3))]))
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INVALID_RESPONSE,
    )
    assert statuses(rid) == [("failed_with_usage", 10)]
    assert_failed(rid, codes.INVALID_RESPONSE)


def test_unusable_without_usage_charges_the_reservation(owner, world):
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([unusable()]))
    ((_, charged),) = statuses(rid)
    assert charged == attempt_history(rid)[0].reserved_tokens > 0
    assert_failed(rid, codes.INVALID_RESPONSE)


@pytest.mark.parametrize("code", [codes.PROVIDER_TIMEOUT, codes.PROVIDER_UNAVAILABLE])
def test_uncertain_keeps_the_reservation(owner, world, code):
    rid = make_request(owner)
    result = run(owner, rid, ScriptedProvider([uncertain(code)]))
    assert (result.outcome, result.error_code) == (PipelineOutcome.FAILED, code)
    ((status, charged),) = statuses(rid)
    assert status == "uncertain"
    assert charged == attempt_history(rid)[0].reserved_tokens > 0
    assert_failed(rid, code)


@pytest.mark.parametrize("code", [codes.PROVIDER_REJECTED, codes.PROVIDER_UNAVAILABLE])
def test_rejected_charges_nothing_and_uses_the_atomic_path(owner, world, code):
    rid = make_request(owner)
    result = run(owner, rid, ScriptedProvider([rejected(code)]))
    assert (result.outcome, result.error_code) == (PipelineOutcome.FAILED, code)
    assert statuses(rid) == [("rejected", 0)]
    assert_failed(rid, code)


def test_a_raised_provider_exception_is_uncertain_internal_error(owner, world, caplog):
    rid = make_request(owner)
    caplog.set_level(logging.DEBUG)
    result = run(owner, rid, ScriptedProvider([RuntimeError(SECRET_EXC)]))
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INTERNAL_ERROR,
    )
    ((status, charged),) = statuses(rid)
    assert status == "uncertain" and charged == attempt_history(rid)[0].reserved_tokens
    assert_failed(rid, codes.INTERNAL_ERROR)
    assert SECRET_EXC not in caplog.text
    assert SECRET_EXC not in repr(rows("SELECT * FROM chat_request_attempts"))
    assert SECRET_EXC not in repr(rows("SELECT * FROM chat_requests"))


# ---------------------------------------------------------------- the 429 path


def test_a_429_with_retry_after_is_retried_once_then_answers(owner, world):
    rid = make_request(owner)
    sleeps = []
    provider = ScriptedProvider([rate_limited(2000), completed("Ok [1].", USAGE)])
    result = run(owner, rid, provider, sleeps=sleeps)
    assert result.outcome is PipelineOutcome.ANSWERED
    assert sleeps == [2.0]
    assert statuses(rid) == [("rate_limited", 0), ("succeeded", 70)]
    assert len(provider.calls) == 2
    assert (
        provider.calls[0] == provider.calls[1]
    )  # the same prompt, no second retrieval


def test_a_second_429_fails_with_provider_rate_limited(owner, world):
    rid = make_request(owner)
    sleeps = []
    provider = ScriptedProvider([rate_limited(1000), rate_limited(1000)])
    result = run(owner, rid, provider, sleeps=sleeps)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.PROVIDER_RATE_LIMITED,
    )
    assert sleeps == [1.0]
    assert statuses(rid) == [("rate_limited", 0), ("rate_limited", 0)]
    assert_failed(rid, codes.PROVIDER_RATE_LIMITED)


def test_no_third_attempt_ever_exists_after_two_429s_even_on_replay(owner, world):
    """The approved policy is one retry: after two rate_limited attempts nothing may start."""
    rid = make_request(owner)
    provider = ScriptedProvider([rate_limited(0), rate_limited(0)])
    run(owner, rid, provider)
    assert state_of(rid) == "S6" and len(attempt_history(rid)) == 2
    again = ScriptedProvider([])
    replay = run(owner, rid, again)
    assert replay.outcome is PipelineOutcome.REPLAYED_FAILURE
    assert again.calls == [] and len(attempt_history(rid)) == 2


def test_two_429s_left_in_progress_by_a_crash_admit_no_third_attempt(owner, world):
    """State S3 with the cap already reached: the entry finalizes, it does not start."""
    rid = make_request(owner)
    add_attempt(
        rid, "rate_limited", number=1, charged=0, error_code="provider_rate_limited"
    )
    add_attempt(
        rid, "rate_limited", number=2, charged=0, error_code="provider_rate_limited"
    )
    assert state_of(rid) == "S3"
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.PROVIDER_RATE_LIMITED,
    )
    assert (
        provider.calls == []
        and len(attempt_history(rid)) == 2
        and state_of(rid) == "S6"
    )


def test_a_429_without_retry_after_is_not_retried(owner, world):
    rid = make_request(owner)
    sleeps = []
    result = run(owner, rid, ScriptedProvider([rate_limited(None)]), sleeps=sleeps)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.PROVIDER_RATE_LIMITED,
    )
    assert sleeps == [] and statuses(rid) == [("rate_limited", 0)]
    assert_failed(rid, codes.PROVIDER_RATE_LIMITED)


def test_the_retry_sleeps_with_no_attempt_running_and_no_transaction(owner, world):
    rid = make_request(owner)
    seen = []
    with SessionLocal() as db:

        def sleeper(seconds):
            seen.append(db.in_transaction())
            seen.append(
                rows(
                    "SELECT count(*) FROM chat_request_attempts WHERE status = 'running'"
                )[0][0]
            )

        run_chat_pipeline(
            db,
            owner_id=owner,
            request_id=rid,
            question=QUESTION,
            provider=ScriptedProvider([rate_limited(0), completed("Ok.", USAGE)]),
            settings=SETTINGS,
            embedder=FakeEmbedder(),
            sleep=sleeper,
        )
    assert seen == [False, 0]


def test_the_retry_start_still_enforces_the_quota(owner, world, monkeypatch):
    rid = make_request(owner)
    original = pipeline.mark_rate_limited

    def mark_then_use_up_the_day(db, attempt_id, code):
        outcome = original(db, attempt_id, code)
        filler = make_request(owner)
        add_attempt(
            filler,
            "uncertain",
            reserved=1_000_000,
            charged=1_000_000,
            error_code="provider_timeout",
        )
        execute(
            "UPDATE chat_requests SET status = 'failed', error_code = 'provider_timeout',"
            " completed_at = clock_timestamp() + interval '1 second' WHERE id = :r",
            r=filler,
        )
        return outcome

    monkeypatch.setattr(pipeline, "mark_rate_limited", mark_then_use_up_the_day)
    provider = ScriptedProvider([rate_limited(0)])
    result = run(
        owner, rid, provider, settings=ChatSettings(SecretStr("k"), 1_000_000, "m")
    )
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.QUOTA_EXCEEDED,
    )
    assert len(provider.calls) == 1
    assert_failed(rid, codes.QUOTA_EXCEEDED)


# ---------------------------------------------------------------- failures before the provider


def test_no_documents_fails_with_no_context(owner):
    rid = make_request(owner)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code, result.failure_kind) == (
        PipelineOutcome.FAILED,
        codes.NO_CONTEXT,
        None,
    )
    assert provider.calls == [] and attempt_history(rid) == []
    assert_failed(rid, codes.NO_CONTEXT)


def test_another_owners_documents_are_not_context(owner, world):
    stranger = make_user("mallory@example.com")
    rid = make_request(stranger)
    result = run(stranger, rid, ScriptedProvider([]))
    assert result.error_code == codes.NO_CONTEXT


def test_quota_exceeded_finalizes_without_an_attempt(owner, world):
    rid = make_request(owner)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider, settings=ChatSettings(SecretStr("k"), 10, "m"))
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.QUOTA_EXCEEDED,
    )
    assert provider.calls == [] and attempt_history(rid) == []
    assert_failed(rid, codes.QUOTA_EXCEEDED)


@pytest.mark.parametrize(
    "question", ["", "   \n\t", "has a \x00 nul", "lone \ud800 surrogate"]
)
def test_an_unstorable_question_is_an_invalid_input_failure(owner, world, question):
    """D8: expected validation failure; no embedding, no attempt, nothing charged."""
    rid = make_request(owner)
    embedder, provider = FakeEmbedder(), ScriptedProvider([])
    result = run(owner, rid, provider, embedder=embedder, question=question)
    assert (result.outcome, result.error_code, result.failure_kind) == (
        PipelineOutcome.FAILED,
        codes.INTERNAL_ERROR,
        FailureKind.INVALID_INPUT,
    )
    assert embedder.calls == [] and provider.calls == [] and attempt_history(rid) == []
    assert_failed(rid, codes.INTERNAL_ERROR)


@pytest.mark.parametrize(
    "error", [EmbedderUnavailableError("model gone"), RuntimeError(SECRET_EXC)]
)
def test_embedder_failures_are_unexpected_and_charge_nothing(
    owner, world, error, caplog
):
    rid = make_request(owner)
    caplog.set_level(logging.DEBUG)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider, embedder=FakeEmbedder(error=error))
    assert (result.outcome, result.error_code, result.failure_kind) == (
        PipelineOutcome.FAILED,
        codes.INTERNAL_ERROR,
        FailureKind.UNEXPECTED,
    )
    assert provider.calls == [] and attempt_history(rid) == []
    assert_failed(rid, codes.INTERNAL_ERROR)
    assert SECRET_EXC not in caplog.text


def test_a_retrieval_integrity_failure_is_unexpected(owner, world):
    execute("UPDATE document_chunks SET text = 'tampered text!' WHERE chunk_index = 0")
    rid = make_request(owner)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.error_code, result.failure_kind) == (
        codes.INTERNAL_ERROR,
        FailureKind.UNEXPECTED,
    )
    assert provider.calls == [] and attempt_history(rid) == []


@pytest.mark.parametrize(
    "error",
    [EmbeddingInputError("text must be a str"), EmbeddingInputTooLongError(0, 300)],
    ids=["wrong_type_is_a_bug", "too_long_is_the_callers"],
)
def test_embedding_input_errors_are_classified_by_their_meaning(owner, world, error):
    """D8: only the over-long question is the caller's doing; a wrong type is a bug."""
    rid = make_request(owner)
    result = run(owner, rid, ScriptedProvider([]), embedder=FakeEmbedder(error=error))
    expected = (
        FailureKind.INVALID_INPUT
        if isinstance(error, EmbeddingInputTooLongError)
        else FailureKind.UNEXPECTED
    )
    assert (result.error_code, result.failure_kind) == (codes.INTERNAL_ERROR, expected)
    assert attempt_history(rid) == []
    assert_failed(rid, codes.INTERNAL_ERROR)


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("retrieve_chunks", RetrievalRequestError("bad request")),
        ("build_prompt", PromptBuildError("bad prompt")),
    ],
)
def test_validation_errors_from_retrieval_and_prompt_are_invalid_input(
    owner, world, monkeypatch, name, error
):
    def boom(*a, **k):
        raise error

    monkeypatch.setattr(pipeline, name, boom)
    rid = make_request(owner)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.error_code, result.failure_kind) == (
        codes.INTERNAL_ERROR,
        FailureKind.INVALID_INPUT,
    )
    assert provider.calls == [] and attempt_history(rid) == []


# ---------------------------------------------------------------- D10: unpublishable answers


@pytest.mark.parametrize("content", ["bad \x00 nul", "lone \ud800 surrogate", "[9]"])
def test_an_unstorable_answer_fails_as_invalid_response(owner, world, content):
    rid = make_request(owner)
    result = run(owner, rid, ScriptedProvider([completed(content, USAGE)]))
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INVALID_RESPONSE,
    )
    assert statuses(rid) == [("failed_with_usage", 70)]
    assert_failed(rid, codes.INVALID_RESPONSE)


def test_an_unstorable_answer_without_usage_charges_the_reservation(owner, world):
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([completed("bad \x00 nul")]))
    ((_, charged),) = statuses(rid)
    assert charged == attempt_history(rid)[0].reserved_tokens
    assert_failed(rid, codes.INVALID_RESPONSE)


BLANKS = ["  ", "\t", "\n", "\r\n", " \t\n ", "\u00a0", "\u2003", "\x0b\x0c"]


@pytest.mark.parametrize("blank", BLANKS)
def test_is_storable_rejects_every_whitespace_only_string(blank):
    assert pipeline._is_storable(blank) is False
    assert pipeline._is_storable("") is False
    assert pipeline._is_storable("a") is True
    assert pipeline._is_storable(f"{blank}a{blank}") is True


@pytest.mark.parametrize("blank", BLANKS)
@pytest.mark.parametrize("with_usage", [True, False], ids=["usage", "no_usage"])
def test_an_answer_that_is_blank_after_marker_removal_fails_as_invalid_response(
    owner, world, blank, with_usage
):
    """The invalid marker [9] is removed, leaving only whitespace: never published."""
    rid = make_request(owner)
    provider = ScriptedProvider(
        [completed(f"{blank}[9]{blank}", USAGE if with_usage else None)]
    )
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INVALID_RESPONSE,
    )
    (attempt,) = attempt_history(rid)
    if with_usage:
        assert (attempt.status, attempt.charged_tokens) == ("failed_with_usage", 70)
    else:
        assert attempt.charged_tokens == attempt.reserved_tokens > 0
    assert_failed(rid, codes.INVALID_RESPONSE)


def test_malformed_citation_markers_do_not_make_an_answer_unpublishable(owner, world):
    rid = make_request(owner)
    result = run(
        owner, rid, ScriptedProvider([completed("See [9] and [x] and [1].", USAGE)])
    )
    assert result.outcome is PipelineOutcome.ANSWERED
    assert citation_count() == 1


# ---------------------------------------------------------------- D11: transaction boundaries


def test_no_transaction_is_open_during_embedding_including_first_load(owner, world):
    rid = make_request(owner)
    observations = []
    with SessionLocal() as db:

        def check(label):
            return lambda: observations.append((label, db.in_transaction()))

        embedder = FakeEmbedder(on_load=check("load"), on_embed=check("embed"))
        provider = ScriptedProvider(
            [completed("Ok [1].", USAGE)],
            on_call=lambda n: observations.append(("provider", db.in_transaction())),
        )
        result = run_chat_pipeline(
            db,
            owner_id=owner,
            request_id=rid,
            question=QUESTION,
            provider=provider,
            settings=SETTINGS,
            embedder=embedder,
        )
    assert result.outcome is PipelineOutcome.ANSWERED
    assert observations == [("load", False), ("embed", False), ("provider", False)]


def test_embedding_sits_between_closed_transactions_and_retrieval_is_read_only(
    owner, world
):
    rid = make_request(owner)
    order = []
    statements = []
    with SessionLocal() as db:

        @event.listens_for(db, "after_begin")
        def began(session, transaction, connection):
            order.append("begin")

        @event.listens_for(db, "after_transaction_end")
        def ended(session, transaction):
            order.append("end")

        @event.listens_for(db.get_bind(), "before_cursor_execute")
        def saw(conn, cursor, statement, parameters, context, executemany):
            order.append("sql")
            statements.append(statement)

        try:
            run_chat_pipeline(
                db,
                owner_id=owner,
                request_id=rid,
                question=QUESTION,
                provider=ScriptedProvider([completed("Ok [1].", USAGE)]),
                settings=SETTINGS,
                embedder=FakeEmbedder(on_embed=lambda: order.append("embed")),
            )
        finally:
            event.remove(db.get_bind(), "before_cursor_execute", saw)
    before_embed = order[: order.index("embed")]
    assert before_embed.count("begin") == before_embed.count("end")  # nothing open
    after_embed = order[order.index("embed") + 1 :]
    retrieval_at = next(
        i for i, s in enumerate(statements) if "document_chunks" in s and "SELECT" in s
    )
    assert "document_chunks" in statements[retrieval_at]
    assert after_embed[0] == "begin"  # retrieval opens its own transaction


def test_a_session_with_a_transaction_open_is_refused(owner, world):
    rid = make_request(owner)
    with SessionLocal() as db:
        db.execute(select(1))
        with pytest.raises(PipelineArgumentError):
            run_chat_pipeline(
                db,
                owner_id=owner,
                request_id=rid,
                question=QUESTION,
                provider=ScriptedProvider([]),
                settings=SETTINGS,
                embedder=FakeEmbedder(),
            )


@pytest.mark.parametrize("bad", [0, -1, True, "1", None])
def test_bad_ids_are_rejected(bad):
    kwargs = {
        "question": QUESTION,
        "provider": ScriptedProvider([]),
        "settings": SETTINGS,
    }
    with SessionLocal() as db:
        with pytest.raises(PipelineArgumentError):
            run_chat_pipeline(db, owner_id=bad, request_id=1, **kwargs)
        with pytest.raises(PipelineArgumentError):
            run_chat_pipeline(db, owner_id=1, request_id=bad, **kwargs)


# ---------------------------------------------------------------- recovery and the guard race


def test_not_found_and_other_owner(owner, world):
    assert (
        run(owner, 999_999, ScriptedProvider([])).outcome is PipelineOutcome.NOT_FOUND
    )
    stranger = make_user("mallory@example.com")
    rid = make_request(owner)
    assert run(stranger, rid, ScriptedProvider([])).outcome is PipelineOutcome.NOT_FOUND


def test_a_live_attempt_elsewhere_means_no_generation(owner, world):
    rid = make_request(owner)
    add_attempt(rid, "running", started="now")
    provider = ScriptedProvider([])
    before = snapshot()
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.IN_PROGRESS_ELSEWHERE
    assert provider.calls == [] and snapshot() == before


def test_an_expired_started_attempt_is_finalized_without_a_second_generation(
    owner, world
):
    rid = make_request(owner)
    add_attempt(rid, "running", started="now")
    expire_lease(latest_attempt_id(rid))
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.LEASE_EXPIRED,
    )
    assert provider.calls == []
    assert [s for s, _ in statuses(rid)] == ["uncertain"]
    assert_failed(rid, codes.LEASE_EXPIRED)


def test_an_expired_unstarted_attempt_is_abandoned_and_generation_proceeds(
    owner, world
):
    rid = make_request(owner)
    add_attempt(rid, "running", started=None)
    expire_lease(latest_attempt_id(rid))
    result = run(owner, rid, ScriptedProvider([completed("Ok [1].", USAGE)]))
    assert result.outcome is PipelineOutcome.ANSWERED
    assert [s for s, _ in statuses(rid)] == ["abandoned", "succeeded"]


def test_the_admission_guard_race_falls_back_to_finalize_only(
    owner, world, monkeypatch
):
    """Recovery looked first and saw nothing; a paid attempt appeared before start_attempt."""
    rid = make_request(owner)

    def late_recovery(db, owner_id, request_id):
        add_attempt(
            request_id,
            "uncertain",
            reserved=10,
            charged=10,
            error_code="provider_timeout",
        )
        return RecoveryResult(RecoveryOutcome.PROCEED)

    monkeypatch.setattr(pipeline, "recover_request", late_recovery)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.PROVIDER_TIMEOUT,
    )
    assert provider.calls == []
    assert statuses(rid) == [("uncertain", 10)]  # nothing new was started or charged
    assert_failed(rid, codes.PROVIDER_TIMEOUT)


@invalid_history_by_design
def test_the_guard_fallback_never_publishes_on_an_impossible_succeeded_attempt(
    owner, world, monkeypatch
):
    rid = make_request(owner)

    def late_recovery(db, owner_id, request_id):
        add_attempt(request_id, "succeeded", reserved=10, charged=10)
        return RecoveryResult(RecoveryOutcome.PROCEED)

    monkeypatch.setattr(pipeline, "recover_request", late_recovery)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.INCONSISTENT
    assert provider.calls == [] and request_row(rid).status == "in_progress"


def test_a_worker_that_started_between_recovery_and_start_wins(
    owner, world, monkeypatch
):
    """Recovery saw nothing; another worker's live attempt appeared before start_attempt."""
    rid = make_request(owner)

    def late_recovery(db, owner_id, request_id):
        add_attempt(request_id, "running", started=None)
        return RecoveryResult(RecoveryOutcome.PROCEED)

    monkeypatch.setattr(pipeline, "recover_request", late_recovery)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.IN_PROGRESS_ELSEWHERE
    assert provider.calls == [] and request_row(rid).status == "in_progress"
    assert [s for s, _ in statuses(rid)] == ["running"]  # the other worker's, untouched


def test_two_earlier_429s_end_retries_and_finalize_as_rate_limited(owner, world):
    rid = make_request(owner)
    add_attempt(
        rid, "rate_limited", number=1, charged=0, error_code=codes.PROVIDER_RATE_LIMITED
    )
    add_attempt(
        rid, "rate_limited", number=2, charged=0, error_code=codes.PROVIDER_RATE_LIMITED
    )
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.PROVIDER_RATE_LIMITED,
    )
    assert provider.calls == [] and len(attempt_history(rid)) == 2
    assert_failed(rid, codes.PROVIDER_RATE_LIMITED)


# ---------------------------------------------------------------- stale workers stop


def test_a_lease_that_expires_during_generation_stops_publication(owner, world):
    rid = make_request(owner)
    provider = ScriptedProvider(
        [completed("Ok [1].", USAGE)],
        on_call=lambda n: expire_lease(latest_attempt_id(rid)),
    )
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.STOPPED
    assert request_row(rid).status == "in_progress" and message_count() == 0
    assert [s for s, _ in statuses(rid)] == ["running"]  # recovery settles it later
    assert len(provider.calls) == 1


def test_recovery_finishing_first_turns_a_late_publish_into_a_replay(owner, world):
    rid = make_request(owner)

    def takeover(n):
        expire_lease(latest_attempt_id(rid))
        with SessionLocal() as other:
            recover_request(other, owner, rid)

    provider = ScriptedProvider([completed("Ok [1].", USAGE)], on_call=takeover)
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.REPLAYED_FAILURE,
        codes.LEASE_EXPIRED,
    )
    assert message_count() == 0 and len(provider.calls) == 1
    assert [s for s, _ in statuses(rid)] == ["uncertain"]


@pytest.mark.parametrize(
    "result_",
    [uncertain(codes.PROVIDER_TIMEOUT), rejected(), unusable(ProviderUsage(1, 1))],
    ids=["uncertain", "rejected", "unusable"],
)
def test_a_stale_failure_settlement_writes_nothing(owner, world, result_):
    rid = make_request(owner)
    provider = ScriptedProvider(
        [result_], on_call=lambda n: expire_lease(latest_attempt_id(rid))
    )
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.STOPPED
    assert request_row(rid).status == "in_progress"
    assert [s for s, _ in statuses(rid)] == ["running"]


def test_a_stale_429_stops_without_sleeping(owner, world):
    rid = make_request(owner)
    sleeps = []
    provider = ScriptedProvider(
        [rate_limited(1000)], on_call=lambda n: expire_lease(latest_attempt_id(rid))
    )
    result = run(owner, rid, provider, sleeps=sleeps)
    assert result.outcome is PipelineOutcome.STOPPED and sleeps == []


@pytest.mark.parametrize(
    "outcome", [P.REQUEST_NOT_FOUND, P.NOT_RUNNING, P.LEASE_EXPIRED, P.WRONG_PHASE]
)
def test_fence_outcomes_from_publish_stop_without_further_writes(
    owner, world, monkeypatch, outcome
):
    rid = make_request(owner)
    monkeypatch.setattr(
        pipeline, "publish_success", lambda *a, **k: PublishResult(outcome)
    )
    result = run(owner, rid, ScriptedProvider([completed("Ok [1].", USAGE)]))
    assert result.outcome is PipelineOutcome.STOPPED
    assert [s for s, _ in statuses(rid)] == [
        "running"
    ]  # untouched; lease expiry settles it


def test_invalid_citations_from_publish_fail_as_invalid_response(
    owner, world, monkeypatch
):
    rid = make_request(owner)
    monkeypatch.setattr(
        pipeline, "publish_success", lambda *a, **k: PublishResult(P.INVALID_CITATIONS)
    )
    result = run(owner, rid, ScriptedProvider([completed("Ok [1].", USAGE)]))
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INVALID_RESPONSE,
    )
    assert statuses(rid) == [("failed_with_usage", 70)]
    assert_failed(rid, codes.INVALID_RESPONSE)


def test_a_conversation_deleted_during_generation_fails_and_charges_usage(owner, world):
    conversation = execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=owner
    ).scalar_one()
    rid = make_request(owner)
    execute(
        "UPDATE chat_requests SET creates_conversation = false, conversation_id = :c"
        " WHERE id = :r",
        c=conversation,
        r=rid,
    )
    provider = ScriptedProvider(
        [completed("Ok [1].", USAGE)],
        on_call=lambda n: execute(
            "DELETE FROM conversations WHERE id = :c", c=conversation
        ),
    )
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.CONVERSATION_NOT_FOUND,
    )
    assert statuses(rid) == [("failed_with_usage", 70)]


def test_a_publish_exception_settles_with_usage_and_leaks_nothing(
    owner, world, monkeypatch, caplog
):
    rid = make_request(owner)
    caplog.set_level(logging.DEBUG)

    def boom(*a, **k):
        raise RuntimeError(SECRET_EXC)

    monkeypatch.setattr(pipeline, "publish_success", boom)
    result = run(
        owner, rid, ScriptedProvider([completed(SECRET_ANSWER + " [1]", USAGE)])
    )
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.INTERNAL_ERROR,
    )
    assert statuses(rid) == [("failed_with_usage", 70)]
    assert SECRET_EXC not in caplog.text and SECRET_ANSWER not in caplog.text
    assert_failed(rid, codes.INTERNAL_ERROR)


# ---------------------------------------------------------------- failures between start and call


def test_a_failure_marking_the_attempt_started_fails_before_the_call(
    owner, world, monkeypatch
):
    rid = make_request(owner)

    def boom(db, attempt_id):
        raise RuntimeError(SECRET_EXC)

    monkeypatch.setattr(pipeline, "mark_llm_started", boom)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert (result.outcome, result.error_code, result.failure_kind) == (
        PipelineOutcome.FAILED,
        codes.INTERNAL_ERROR,
        FailureKind.UNEXPECTED,
    )
    assert provider.calls == []
    assert statuses(rid) == [("failed_before_call", 0)]
    assert_failed(rid, codes.INTERNAL_ERROR)


def test_an_error_after_the_start_marker_committed_is_left_for_lease_expiry(
    owner, world, monkeypatch
):
    """The mark committed, then the call raised: fail_before_call must refuse (WRONG_PHASE)."""
    rid = make_request(owner)
    real = pipeline.mark_llm_started

    def commit_then_raise(db, attempt_id):
        real(db, attempt_id)
        raise RuntimeError("connection dropped after commit")

    monkeypatch.setattr(pipeline, "mark_llm_started", commit_then_raise)
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.STOPPED
    assert provider.calls == []
    assert [s for s, _ in statuses(rid)] == ["running"]
    assert request_row(rid).status == "in_progress"
    # a later entry settles it through R2/R4, with no second generation
    monkeypatch.setattr(pipeline, "mark_llm_started", real)
    expire_lease(latest_attempt_id(rid))
    later_provider = ScriptedProvider([])
    later = run(owner, rid, later_provider)
    assert (later.outcome, later.error_code) == (
        PipelineOutcome.FAILED,
        codes.LEASE_EXPIRED,
    )
    assert later_provider.calls == []


def test_a_lost_mark_started_fence_stops(owner, world, monkeypatch):
    rid = make_request(owner)
    monkeypatch.setattr(
        pipeline,
        "mark_llm_started",
        lambda db, attempt_id: TransitionOutcome.LEASE_EXPIRED,
    )
    provider = ScriptedProvider([])
    result = run(owner, rid, provider)
    assert result.outcome is PipelineOutcome.STOPPED and provider.calls == []


# ---------------------------------------------------------------- logging and structure


def test_logs_carry_no_question_answer_chunk_or_secret(owner, world, caplog):
    caplog.set_level(logging.DEBUG)
    rid = make_request(owner)
    run(owner, rid, ScriptedProvider([completed(SECRET_ANSWER + " [1]", USAGE)]))
    rid2 = make_request(owner)
    run(owner, rid2, ScriptedProvider([RuntimeError(SECRET_EXC)]))
    for forbidden in (
        QUESTION,
        SECRET_ANSWER,
        SECRET_EXC,
        "chunk number",
        "not-a-real-key",
    ):
        assert forbidden not in caplog.text


def test_the_orchestration_is_a_langgraph_state_graph_without_checkpointer_or_callbacks():
    import inspect

    source = inspect.getsource(pipeline.run_chat_pipeline)
    assert "StateGraph" in source and ".compile()" in source
    assert "checkpointer" not in source and "callbacks" not in source


# ---------------------------------------------------------------- the observer, mid-flight


def test_the_observer_sees_the_expected_state_at_every_point_of_a_429_retry(
    owner, world
):
    rid = make_request(owner)
    seen = []

    def provider_called(n):
        seen.append(("provider", n, state_of(rid)))
        assert_valid(rid)

    def slept(seconds):
        seen.append(("sleep", None, state_of(rid)))
        assert_valid(rid)

    provider = ScriptedProvider(
        [rate_limited(0), completed("Ok [1].", USAGE)], on_call=provider_called
    )
    assert state_of(rid) == "S1"
    with SessionLocal() as db:
        run_chat_pipeline(
            db,
            owner_id=owner,
            request_id=rid,
            question=QUESTION,
            provider=provider,
            settings=SETTINGS,
            embedder=FakeEmbedder(),
            sleep=slept,
        )
    assert seen == [("provider", 1, "S2"), ("sleep", None, "S3"), ("provider", 2, "S2")]
    assert state_of(rid) == "S7"


def test_the_observer_sees_s2_during_generation_and_s6_after_a_failure(owner, world):
    rid = make_request(owner)
    seen = []
    provider = ScriptedProvider(
        [uncertain(codes.PROVIDER_TIMEOUT)],
        on_call=lambda n: seen.append(state_of(rid)),
    )
    run(owner, rid, provider)
    assert seen == ["S2"] and state_of(rid) == "S6"


# ---------------------------------------------------------------- concurrency and crashes


class SlowProvider:
    """Thread-safe: counts calls and holds each one open long enough for the others to arrive."""

    def __init__(self, seconds=0.4):
        self.calls = 0
        self._seconds = seconds
        self._lock = __import__("threading").Lock()

    def generate(self, system_message, user_message):
        import time

        with self._lock:
            self.calls += 1
        time.sleep(self._seconds)
        return completed("Ok [1].", USAGE)


def test_concurrent_entries_for_one_request_make_exactly_one_generation(owner, world):
    from tests.chat_pipeline_db import run_threads

    rid = make_request(owner)
    provider = SlowProvider()

    def work(db, i):
        return run_chat_pipeline(
            db,
            owner_id=owner,
            request_id=rid,
            question=QUESTION,
            provider=provider,
            settings=SETTINGS,
            embedder=FakeEmbedder(),
            sleep=lambda s: None,
        )

    results = run_threads(6, work)
    outcomes = [r.outcome for r in results]
    assert provider.calls == 1
    assert outcomes.count(PipelineOutcome.ANSWERED) == 1
    assert set(outcomes) <= {
        PipelineOutcome.ANSWERED,
        PipelineOutcome.IN_PROGRESS_ELSEWHERE,
        PipelineOutcome.REPLAYED_SUCCESS,
    }
    assert statuses(rid) == [("succeeded", 70)] and message_count() == 2


def test_a_late_entry_after_the_answer_replays_it_and_never_generates_again(
    owner, world
):
    rid = make_request(owner)
    first = SlowProvider(0)
    run(owner, rid, first)
    second = SlowProvider(0)
    again = run(owner, rid, second)
    assert again.outcome is PipelineOutcome.REPLAYED_SUCCESS
    assert (first.calls, second.calls) == (1, 0)
    assert len(attempt_history(rid)) == 1


CRASH_IN_PROVIDER = """
import os, sys
import numpy as np
from pydantic import SecretStr
from app.core.chat_config import ChatSettings
from app.core.database import SessionLocal
from app.rag.chat_pipeline import run_chat_pipeline

class Embedder:
    embedding_id = "emb"
    def embed_query(self, text):
        v = np.zeros(384, dtype=np.float32); v[0] = 1.0
        return v

class Provider:
    def generate(self, system_message, user_message):
        os._exit(17)  # the process dies mid-generation: attempt started, nothing settled

with SessionLocal() as db:
    run_chat_pipeline(
        db, owner_id=int(sys.argv[1]), request_id=int(sys.argv[2]), question=sys.argv[3],
        provider=Provider(), settings=ChatSettings(SecretStr("k"), 10_000_000, "m"),
        embedder=Embedder(),
    )
os._exit(0)
"""


def test_a_crash_during_generation_is_recovered_without_a_second_generation(
    owner, world
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    rid = make_request(owner)
    done = subprocess.run(
        [sys.executable, "-c", CRASH_IN_PROVIDER, str(owner), str(rid), QUESTION],
        cwd=Path(__file__).resolve().parent.parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 17, done.stderr
    assert state_of(rid) == "S2"  # started, running, lease still live
    assert_valid(rid)

    # while the lease is live, a replay must not start anything
    live = ScriptedProvider([])
    assert run(owner, rid, live).outcome is PipelineOutcome.IN_PROGRESS_ELSEWHERE
    assert live.calls == []

    # once the lease has expired, the replay settles it and fails the request: no new generation
    expire_lease(latest_attempt_id(rid))
    later = ScriptedProvider([])
    result = run(owner, rid, later)
    assert (result.outcome, result.error_code) == (
        PipelineOutcome.FAILED,
        codes.LEASE_EXPIRED,
    )
    assert later.calls == []
    assert [(s, c) for s, c in statuses(rid)] == [
        ("uncertain", attempt_history(rid)[0].reserved_tokens)
    ]
    assert state_of(rid) == "S6"
    assert (
        run(owner, rid, ScriptedProvider([])).outcome
        is PipelineOutcome.REPLAYED_FAILURE
    )
