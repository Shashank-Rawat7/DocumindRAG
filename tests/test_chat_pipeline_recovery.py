"""C9.5b slice 1: the recovery protocol R1-R4 (chat_pipeline.recover_request).

Runs against documind_test. Histories are seeded with SQL or built with the real C8 services.
"""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

import app.rag.chat_attempt_start as start_module
import app.rag.chat_pipeline as pipeline
from app.core.database import SessionLocal
from app.rag.chat_attempt_lifecycle import expire_stale_attempt, mark_llm_started
from app.rag.chat_attempt_start import AttemptStartOutcome, start_attempt
from app.rag.chat_pipeline import (
    PipelineArgumentError,
    RecoveryOutcome,
    RecoveryResult,
    finalize_after_terminal_attempt,
    read_attempt_summary,
    recover_request,
)
from app.rag.chat_request_finalization import (
    FailOutcome,
    FailResult,
    PublishOutcome,
    publish_success,
)
from tests.chat_pipeline_db import (
    add_attempt,
    add_terminal,
    attempt_history,
    execute,
    expire_lease,
    make_request,
    make_user,
    request_row,
    rows,
    run_threads,
    snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
O = RecoveryOutcome
HOUR = timedelta(hours=1)
TERMINAL = ("succeeded", "failed_with_usage", "uncertain", "rejected")


@pytest.fixture
def owner():
    return make_user("alice@example.com")


@pytest.fixture
def other_owner():
    return make_user("bob@example.com")


def recover(owner_id, request_id):
    with SessionLocal() as db:
        return recover_request(db, owner_id, request_id)


def real_started_attempt(owner_id, request_id, reserved=500):
    """A running, started attempt created through the real services."""
    with SessionLocal() as db:
        result = start_attempt(
            db,
            owner_id,
            request_id,
            reserved_tokens=reserved,
            daily_limit_tokens=10_000_000,
            lease_seconds=600,
        )
    assert result.outcome is AttemptStartOutcome.STARTED
    with SessionLocal() as db:
        mark_llm_started(db, result.attempt_id)
    return result.attempt_id


@pytest.fixture(autouse=True)
def _no_attempt_may_be_started(monkeypatch):
    """Recovery must never admit an attempt: any call to start_attempt fails the test."""

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery called start_attempt")

    monkeypatch.setattr(start_module, "start_attempt", forbidden)
    monkeypatch.setattr(pipeline, "start_attempt", forbidden, raising=False)


# ---------------------------------------------------------------- R1: the stored request


def test_unknown_and_foreign_requests_are_not_found(owner, other_owner):
    rid = make_request(owner)
    before = snapshot()
    assert recover(owner, 999_999) == RecoveryResult(O.NOT_FOUND)
    assert recover(other_owner, rid) == RecoveryResult(O.NOT_FOUND)
    assert snapshot() == before


def test_a_failed_request_replays_its_stored_code_and_writes_nothing(owner):
    rid = make_request(owner, status="failed", error_code="quota_exceeded")
    before = snapshot()
    result = recover(owner, rid)
    assert result.outcome is O.REPLAY_FAILED
    assert result.error_code == "quota_exceeded"
    assert result.request.status == "failed"
    assert snapshot() == before


def test_a_succeeded_request_replays_and_writes_nothing(owner):
    rid = make_request(owner, status="succeeded")
    before = snapshot()
    result = recover(owner, rid)
    assert result.outcome is O.REPLAY_SUCCEEDED
    assert result.request.status == "succeeded"
    assert snapshot() == before


# ---------------------------------------------------------------- R2/R3: nothing to finalize


def test_a_fresh_request_proceeds(owner):
    rid = make_request(owner)
    before = snapshot()
    assert recover(owner, rid) == RecoveryResult(O.PROCEED)
    assert snapshot() == before


@pytest.mark.parametrize(
    "history",
    [
        ["rate_limited"],  # S3: one 429
        ["rate_limited", "rate_limited"],  # two 429s
        ["failed_before_call"],  # S4
        ["abandoned", "failed_before_call"],
        ["failed_before_call", "rate_limited"],
    ],
)
def test_only_unstarted_or_429_attempts_proceed_without_writes(owner, history):
    rid = make_request(owner)
    for number, status in enumerate(history, start=1):
        # a 429 reached the provider (started); the unstarted statuses provably did not
        started = "now" if status == "rate_limited" else None
        add_attempt(rid, status, number=number, started=started, charged=0)
    before = snapshot()
    assert recover(owner, rid) == RecoveryResult(O.PROCEED)
    assert snapshot() == before


def test_a_live_running_attempt_means_someone_else_is_working(owner):
    rid = make_request(owner)
    add_attempt(rid, "running", number=1)  # started, lease live
    before = snapshot()
    assert recover(owner, rid) == RecoveryResult(O.IN_PROGRESS_ELSEWHERE)
    assert snapshot() == before


def test_a_live_unstarted_running_attempt_means_someone_else_is_working(owner):
    rid = make_request(owner)
    add_attempt(rid, "running", number=1, started=None)
    before = snapshot()
    assert recover(owner, rid) == RecoveryResult(O.IN_PROGRESS_ELSEWHERE)
    assert snapshot() == before


def test_an_expired_unstarted_attempt_is_abandoned_and_the_request_proceeds(owner):
    rid = make_request(owner)
    add_attempt(rid, "running", number=1, started=None, reserved=300, lease_delta=-HOUR)
    assert recover(owner, rid) == RecoveryResult(O.PROCEED)  # S4: retry is safe
    (only,) = attempt_history(rid)
    assert (only.status, only.charged_tokens, only.error_code) == (
        "abandoned",
        0,
        "lease_expired",
    )
    assert request_row(rid).status == "in_progress"


# ---------------------------------------------------------------- R2 + R4: expired started attempt


def test_an_expired_started_attempt_ends_the_request_without_another_attempt(owner):
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid, reserved=500)
    expire_lease(aid)
    result = recover(owner, rid)
    assert result == RecoveryResult(O.FINALIZED, error_code="lease_expired")
    (only,) = attempt_history(rid)  # no second attempt, ever
    assert (only.status, only.charged_tokens, only.reserved_tokens) == (
        "uncertain",
        500,
        500,
    )
    request = request_row(rid)
    assert (request.status, request.error_code) == ("failed", "lease_expired")
    assert recover(owner, rid).outcome is O.REPLAY_FAILED  # idempotent
    assert recover(owner, rid).error_code == "lease_expired"


def test_state_s5_is_finalized_by_the_next_entry(owner):
    # The window between expiry and finalization, built with the real expiry service.
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid)
    expire_lease(aid)
    with SessionLocal() as db:
        expire_stale_attempt(db, rid)
    (only,) = attempt_history(rid)
    assert (only.status, only.error_code) == ("uncertain", "lease_expired")
    assert request_row(rid).status == "in_progress"  # S5
    assert recover(owner, rid) == RecoveryResult(
        O.FINALIZED, error_code="lease_expired"
    )
    assert len(attempt_history(rid)) == 1


@pytest.mark.parametrize("status", ["failed_with_usage", "uncertain", "rejected"])
def test_a_settled_terminal_attempt_is_finalized_with_its_own_code(owner, status):
    rid = make_request(owner)
    add_terminal(rid, status, error_code="invalid_response")
    result = recover(owner, rid)
    assert result == RecoveryResult(O.FINALIZED, error_code="invalid_response")
    assert request_row(rid).error_code == "invalid_response"
    assert len(attempt_history(rid)) == 1


def test_a_terminal_attempt_after_429s_is_finalized(owner):
    rid = make_request(owner)
    add_attempt(rid, "rate_limited", number=1, charged=0)
    add_terminal(rid, "uncertain", number=2, error_code="provider_timeout")
    assert recover(owner, rid) == RecoveryResult(
        O.FINALIZED, error_code="provider_timeout"
    )
    assert [a.status for a in attempt_history(rid)] == ["rate_limited", "uncertain"]


def test_a_missing_or_malformed_code_falls_back_to_internal_error():
    # The CHECKs never let such a code reach the table, so the rule is pinned directly.
    assert pipeline._finalization_code(
        pipeline.TerminalAttempt(1, "uncertain", None)
    ) == ("internal_error")
    assert pipeline._finalization_code(
        pipeline.TerminalAttempt(1, "uncertain", "Bad Code")
    ) == ("internal_error")
    assert pipeline._finalization_code(
        pipeline.TerminalAttempt(1, "uncertain", "ok_code")
    ) == ("ok_code")


# ---------------------------------------------------------------- inconsistent / racing states


def test_a_succeeded_attempt_on_an_in_progress_request_is_inconsistent(owner):
    rid = make_request(owner)
    add_terminal(rid, "succeeded", error_code=None)
    before = snapshot()
    result = recover(owner, rid)
    assert result.outcome is O.INCONSISTENT
    assert (
        snapshot() == before
    )  # nothing is written for a state the services cannot produce


def test_a_publish_that_committed_after_r1_is_reported_as_a_replay(owner, monkeypatch):
    rid = make_request(owner)
    real_expire = pipeline.expire_stale_attempt

    def expire_then_publish_wins(db, request_id):
        outcome = real_expire(db, request_id)
        # Between R1 and R3 the other worker publishes (attempt + request in one commit).
        add_terminal(rid, "succeeded", error_code=None)
        execute(
            "UPDATE chat_requests SET status = 'succeeded', completed_at = clock_timestamp()"
            " WHERE id = :i",
            i=rid,
        )
        return outcome

    monkeypatch.setattr(pipeline, "expire_stale_attempt", expire_then_publish_wins)
    assert recover(owner, rid).outcome is O.REPLAY_SUCCEEDED


@pytest.mark.parametrize(
    ("fail_outcome", "expected"),
    [
        (FailOutcome.ATTEMPT_RUNNING, O.IN_PROGRESS_ELSEWHERE),
        (FailOutcome.NOT_FOUND, O.NOT_FOUND),
    ],
)
def test_the_other_fail_request_outcomes_are_mapped_without_writes(
    owner, monkeypatch, fail_outcome, expected
):
    # Unreachable through the real services (C9.5a stops any attempt after a terminal one), but
    # the mapping is part of the contract: a running attempt means someone else is working.
    rid = make_request(owner)
    add_terminal(rid, "uncertain")
    monkeypatch.setattr(
        pipeline, "fail_request", lambda *args, **kwargs: FailResult(fail_outcome)
    )
    before = snapshot()
    assert recover(owner, rid).outcome is expected
    assert snapshot() == before


def test_finalize_helper_proceeds_when_no_terminal_attempt_exists(owner):
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    with SessionLocal() as db:
        assert finalize_after_terminal_attempt(db, owner, rid) == RecoveryResult(
            O.PROCEED
        )


# ---------------------------------------------------------------- the attempt summary


def test_the_summary_reads_the_whole_history_not_the_last_row(owner):
    rid = make_request(owner)
    add_attempt(rid, "failed_before_call", number=1, started=None, charged=0)
    add_terminal(rid, "uncertain", number=2)
    # A (never produced) later unstarted attempt must not hide the terminal-started one.
    add_attempt(rid, "failed_before_call", number=3, started=None, charged=0)
    with SessionLocal() as db:
        summary = read_attempt_summary(db, rid)
    assert summary.running is False
    assert summary.terminal is not None and summary.terminal.status == "uncertain"


def test_the_summary_reports_running_and_picks_the_lowest_terminal(owner):
    rid = make_request(owner)
    add_terminal(rid, "uncertain", number=1)
    add_terminal(rid, "rejected", number=2)
    add_attempt(rid, "running", number=3)
    with SessionLocal() as db:
        summary = read_attempt_summary(db, rid)
    assert summary.running is True
    assert summary.terminal.status == "uncertain"


@pytest.mark.parametrize("status", TERMINAL)
def test_every_terminal_status_is_recognized(owner, status):
    rid = make_request(owner)
    add_terminal(rid, status, error_code=None if status == "succeeded" else "code")
    with SessionLocal() as db:
        assert read_attempt_summary(db, rid).terminal.status == status


def test_the_summary_is_one_read_only_statement_that_commits(owner):
    rid = make_request(owner)
    with SessionLocal() as db:
        read_attempt_summary(db, rid)
        assert not db.in_transaction()


def test_the_terminal_statuses_are_the_guards_statuses():
    assert (
        pipeline.TERMINAL_STARTED_STATUSES
        == start_module._NON_RETRYABLE_STARTED_STATUSES
    )


# ---------------------------------------------------------------- concurrency


def test_concurrent_recovery_of_one_expired_attempt_settles_and_finalizes_once(owner):
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid, reserved=500)
    expire_lease(aid)
    results = run_threads(8, lambda db, i: recover_request(db, owner, rid))
    outcomes = [r.outcome for r in results]
    assert outcomes.count(O.FINALIZED) == 1
    assert set(outcomes) <= {O.FINALIZED, O.REPLAY_FAILED}
    assert {r.error_code for r in results} == {"lease_expired"}
    (only,) = attempt_history(rid)  # one settlement, no new attempt
    assert (only.status, only.charged_tokens) == ("uncertain", 500)
    assert request_row(rid).status == "failed"


def test_concurrent_recovery_with_a_live_worker_writes_nothing(owner):
    rid = make_request(owner)
    add_attempt(rid, "running", number=1)
    before = snapshot()
    results = run_threads(6, lambda db, i: recover_request(db, owner, rid))
    assert {r.outcome for r in results} == {O.IN_PROGRESS_ELSEWHERE}
    assert snapshot() == before


# ---------------------------------------------------------------- the stale original worker


def publish(owner_id, request_id, attempt_id):
    with SessionLocal() as db:
        return publish_success(
            db,
            owner_id,
            request_id,
            attempt_id,
            user_content="question",
            assistant_content="answer",
        )


def test_a_stale_worker_cannot_publish_in_state_s5_or_after_finalization(owner):
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid, reserved=500)
    expire_lease(aid)
    with SessionLocal() as db:
        expire_stale_attempt(db, rid)  # S5
    before = snapshot()
    assert publish(owner, rid, aid).outcome is PublishOutcome.NOT_RUNNING
    assert snapshot() == before
    assert recover(owner, rid).outcome is O.FINALIZED
    after = snapshot()
    assert publish(owner, rid, aid).outcome is PublishOutcome.REQUEST_FAILED
    assert snapshot() == after
    (only,) = attempt_history(rid)
    assert (only.status, only.charged_tokens) == ("uncertain", 500)  # charge unchanged


def test_a_stale_worker_whose_lease_expired_is_stopped_before_recovery_runs(owner):
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid, reserved=500)
    expire_lease(aid)  # still running, past its lease, nobody has expired it yet
    before = snapshot()
    assert publish(owner, rid, aid).outcome is PublishOutcome.LEASE_EXPIRED
    assert snapshot() == before
    assert (
        recover(owner, rid).outcome is O.FINALIZED
    )  # recovery then settles and finalizes
    assert len(attempt_history(rid)) == 1


# ---------------------------------------------------------------- a real crash between R2 and R4

CRASH_SCRIPT = """
import os, sys
from app.core.database import SessionLocal
import app.rag.chat_pipeline as p

def die(*args, **kwargs):
    os._exit(17)  # the process dies after expire_stale_attempt committed, before fail_request

p.fail_request = die
with SessionLocal() as db:
    p.recover_request(db, int(sys.argv[1]), int(sys.argv[2]))
os._exit(0)
"""


def test_a_process_crash_between_expiry_and_finalization_is_recovered_by_the_next_entry(
    owner,
):
    rid = make_request(owner)
    aid = real_started_attempt(owner, rid, reserved=500)
    expire_lease(aid)
    done = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT, str(owner), str(rid)],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 17, done.stderr  # it really died between the two steps
    (only,) = attempt_history(rid)
    assert (only.status, only.error_code, only.charged_tokens) == (
        "uncertain",
        "lease_expired",
        500,
    )
    assert request_row(rid).status == "in_progress"  # state S5, as documented
    assert recover(owner, rid) == RecoveryResult(
        O.FINALIZED, error_code="lease_expired"
    )
    assert len(attempt_history(rid)) == 1  # the replay started nothing
    assert request_row(rid).status == "failed"


# ---------------------------------------------------------------- arguments


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1", 2**31])
def test_invalid_ids_are_rejected(owner, bad):
    rid = make_request(owner)
    with SessionLocal() as db:
        with pytest.raises(PipelineArgumentError):
            recover_request(db, bad, rid)
        with pytest.raises(PipelineArgumentError):
            recover_request(db, owner, bad)
        with pytest.raises(PipelineArgumentError):
            read_attempt_summary(db, bad)
        with pytest.raises(PipelineArgumentError):
            finalize_after_terminal_attempt(db, owner, bad)


def test_recovery_commits_and_leaves_no_transaction_open(owner):
    rid = make_request(owner)
    with SessionLocal() as db:
        recover_request(db, owner, rid)
        assert not db.in_transaction()
    assert rows("SELECT count(*) FROM chat_request_attempts") == [(0,)]
