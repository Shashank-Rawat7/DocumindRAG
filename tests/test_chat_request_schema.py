"""C7.2 schema tests: chat_requests and chat_request_attempts (migration #7).

Runs against documind_test, which conftest builds with `alembic upgrade head`,
so these assertions check the migrated database, not just ORM metadata.
Schema only: no lease, fencing, reservation or quota behaviour (that is C8).

The attempt CHECKs are tested hardest on NULL handling: PostgreSQL accepts a row
whose CHECK evaluates to NULL, so a terminal attempt with charged_tokens = NULL
must be rejected explicitly (T-S1) and every CHECK must be total (T-S4).
"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.core.database import engine
from app.models.chat_request import ChatRequest
from app.models.chat_request_attempt import ChatRequestAttempt
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_7_FILE = (
    PROJECT_ROOT / "migrations" / "versions" / "e9749572298d_add_chat_requests_and_attempts.py"
)
MIGRATION_7_REVISION = "e9749572298d"
MIGRATION_7_DOWN_REVISION = "4a06e23062ae"
MIGRATION_8_FILE = (
    PROJECT_ROOT / "migrations" / "versions" / "c4f1a7d92b60_allow_rejected_attempt_status.py"
)
MIGRATION_8_REVISION = "c4f1a7d92b60"
MIGRATION_8_CHANGED_CHECKS = {
    "ck_chat_request_attempts_status_valid",
    "ck_chat_request_attempts_llm_started_consistent",
    "ck_chat_request_attempts_charge_consistent",
}
ME = "/api/v1/auth/me"

NOW = datetime.now(timezone.utc)
FUTURE = NOW + timedelta(hours=1)
VALID_SHA = "a" * 64

REQUEST_COLUMNS = {
    "id", "owner_id", "idempotency_key", "request_sha256", "status", "creates_conversation",
    "conversation_id", "user_message_id", "assistant_message_id", "error_code",
    "created_at", "completed_at",
}
REQUEST_NULLABLE = {
    "conversation_id", "user_message_id", "assistant_message_id", "error_code", "completed_at",
}
ATTEMPT_COLUMNS = {
    "id", "request_id", "attempt_number", "status", "lease_expires_at", "reserved_tokens",
    "prompt_tokens", "completion_tokens", "charged_tokens", "error_code", "created_at",
    "llm_started_at", "finished_at",
}
ATTEMPT_NULLABLE = {
    "lease_expires_at", "prompt_tokens", "completion_tokens", "charged_tokens", "error_code",
    "llm_started_at", "finished_at",
}
EXPECTED_REQUEST_CHECKS = {
    "ck_chat_requests_status_valid",
    "ck_chat_requests_idempotency_key_format",
    "ck_chat_requests_request_sha256_format",
    "ck_chat_requests_completed_at_consistent",
    "ck_chat_requests_error_code_consistent",
    "ck_chat_requests_new_conversation_unset_until_success",
}
EXPECTED_ATTEMPT_CHECKS = {
    "ck_chat_request_attempts_attempt_number_positive",
    "ck_chat_request_attempts_status_valid",
    "ck_chat_request_attempts_lease_consistent",
    "ck_chat_request_attempts_finished_at_consistent",
    "ck_chat_request_attempts_error_code_consistent",
    "ck_chat_request_attempts_reserved_tokens_positive",
    "ck_chat_request_attempts_usage_valid",
    "ck_chat_request_attempts_llm_started_consistent",
    "ck_chat_request_attempts_charged_tokens_null_iff_running",
    "ck_chat_request_attempts_charge_consistent",
}
CHARGE = "ck_chat_request_attempts_charge_consistent"
NULL_IFF_RUNNING = "ck_chat_request_attempts_charged_tokens_null_iff_running"
LLM_STARTED = "ck_chat_request_attempts_llm_started_consistent"
LEASE = "ck_chat_request_attempts_lease_consistent"
FINISHED = "ck_chat_request_attempts_finished_at_consistent"
ERROR_CODE = "ck_chat_request_attempts_error_code_consistent"
USAGE = "ck_chat_request_attempts_usage_valid"

# R1's charge CHECK, kept only as a regression exhibit for T-S5: it differs from the
# frozen one by lacking the IS NOT NULL guards on terminal branches (and on
# completion_tokens), so a NULL charge made it evaluate to NULL, which PostgreSQL accepts.
R1_CHARGE_CONSISTENT_REGRESSION = (
    "(status = 'running' AND prompt_tokens IS NULL AND charged_tokens IS NULL) "
    "OR (status = 'succeeded' AND ((prompt_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (prompt_tokens IS NULL AND charged_tokens = reserved_tokens))) "
    "OR (status = 'failed_with_usage' AND prompt_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (status IN ('failed_before_call', 'abandoned', 'rate_limited') "
    "AND prompt_tokens IS NULL AND charged_tokens = 0) "
    "OR (status = 'uncertain' AND prompt_tokens IS NULL AND charged_tokens = reserved_tokens)"
)

# Valid terminal attempts, one per accounting outcome (reserved_tokens is always 100).
TERMINAL = {
    "succeeded_with_usage": dict(
        status="succeeded", prompt_tokens=10, completion_tokens=5, charged_tokens=15,
        llm_started_at=NOW, finished_at=NOW),
    "succeeded_zero_usage": dict(
        status="succeeded", prompt_tokens=0, completion_tokens=0, charged_tokens=0,
        llm_started_at=NOW, finished_at=NOW),
    "succeeded_without_usage": dict(
        status="succeeded", charged_tokens=100, llm_started_at=NOW, finished_at=NOW),
    "failed_with_usage": dict(
        status="failed_with_usage", prompt_tokens=10, completion_tokens=5, charged_tokens=15,
        llm_started_at=NOW, finished_at=NOW, error_code="unusable_response"),
    "failed_before_call": dict(
        status="failed_before_call", charged_tokens=0, llm_started_at=None, finished_at=NOW,
        error_code="pre_call_failed"),
    "rate_limited": dict(
        status="rate_limited", charged_tokens=0, llm_started_at=NOW, finished_at=NOW,
        error_code="rate_limited"),
    "uncertain": dict(
        status="uncertain", charged_tokens=100, llm_started_at=NOW, finished_at=NOW,
        error_code="timeout"),
    "abandoned": dict(
        status="abandoned", charged_tokens=0, llm_started_at=None, finished_at=NOW,
        error_code="lease_expired"),
    "rejected": {
        "status": "rejected", "charged_tokens": 0, "llm_started_at": NOW,
        "finished_at": NOW, "error_code": "provider_rejected"},
}
# succeeded rows carry no error_code
for _name in ("succeeded_with_usage", "succeeded_zero_usage", "succeeded_without_usage"):
    TERMINAL[_name]["error_code"] = None
TERMINAL_NAMES = list(TERMINAL)


# ---------- helpers ----------

def load_migration_7():
    spec = importlib.util.spec_from_file_location("migration_7_under_test", MIGRATION_7_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_migration_8():
    spec = importlib.util.spec_from_file_location("migration_8_under_test", MIGRATION_8_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_attempt_checks():
    """The attempt CHECKs in force at head: migration 7's, with migration 8's replacing three."""
    return {**dict(load_migration_7().ATTEMPT_CHECKS),
            **dict(load_migration_8().ATTEMPT_CHECKS_CHANGED)}


def model_checks(model):
    return {c.name: str(c.sqltext) for c in model.__table__.constraints
            if isinstance(c, CheckConstraint)}


def make_user(db, email):
    user = User(email=email, hashed_password="not-a-real-hash")
    db.add(user)
    db.commit()
    return user.id


def make_conversation(db, owner_id):
    conversation = Conversation(owner_id=owner_id)
    db.add(conversation)
    db.commit()
    return conversation.id


def make_message_pair(db, conversation_id):
    user_message = Message(conversation_id=conversation_id, seq=1, role="user", content="q")
    assistant_message = Message(conversation_id=conversation_id, seq=2, role="assistant",
                                content="a")
    db.add_all([user_message, assistant_message])
    db.commit()
    return user_message.id, assistant_message.id


def request_fields(owner_id, key="key-1", **overrides):
    """A chat request that satisfies every constraint unless overridden."""
    fields = dict(owner_id=owner_id, idempotency_key=key, request_sha256=VALID_SHA,
                  status="in_progress", creates_conversation=False)
    fields.update(overrides)
    return fields


def make_request(db, owner_id, key="key-1", **overrides):
    request = ChatRequest(**request_fields(owner_id, key, **overrides))
    db.add(request)
    db.commit()
    return request.id


def attempt_fields(request_id, attempt_number=1, **overrides):
    """A running attempt that satisfies every constraint unless overridden."""
    fields = dict(request_id=request_id, attempt_number=attempt_number, status="running",
                  lease_expires_at=FUTURE, reserved_tokens=100)
    fields.update(overrides)
    return fields


def terminal_attempt(request_id, name, attempt_number=1, **overrides):
    """A valid terminal attempt (no lease) of the named kind, with optional overrides."""
    fields = attempt_fields(request_id, attempt_number, lease_expires_at=None,
                            **TERMINAL[name])
    fields.update(overrides)
    return ChatRequestAttempt(**fields)


def make_attempt(db, request_id, attempt_number=1, **overrides):
    attempt = ChatRequestAttempt(**attempt_fields(request_id, attempt_number, **overrides))
    db.add(attempt)
    db.commit()
    return attempt.id


def assert_violates(db, instance, *constraint_names):
    """Insert must fail, and fail on one of the named constraints."""
    db.add(instance)
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.diag.constraint_name in constraint_names


def counts(db):
    return {
        "conversations": db.query(Conversation).count(),
        "messages": db.query(Message).count(),
        "requests": db.query(ChatRequest).count(),
        "attempts": db.query(ChatRequestAttempt).count(),
    }


@pytest.fixture
def owner_id(db):
    return make_user(db, "owner@example.com")


@pytest.fixture
def request_id(db, owner_id):
    return make_request(db, owner_id)


# ---------- migration #7 position and registration ----------

def test_migration_7_follows_migration_6_and_migration_8_is_the_newest_head():
    script = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))

    assert script.get_revision(MIGRATION_7_REVISION).down_revision == MIGRATION_7_DOWN_REVISION
    assert script.get_revision(MIGRATION_8_REVISION).down_revision == MIGRATION_7_REVISION
    assert script.get_heads() == [MIGRATION_8_REVISION]


def test_models_are_registered_in_app_main_and_alembic_env():
    import app.main

    assert app.main.chat_request.ChatRequest is ChatRequest
    assert app.main.chat_request_attempt.ChatRequestAttempt is ChatRequestAttempt

    env_source = (PROJECT_ROOT / "migrations" / "env.py").read_text(encoding="utf-8")
    [import_line] = [ln for ln in env_source.splitlines() if ln.startswith("from app.models import")]
    names = {n.strip() for n in import_line.split("#")[0].split("import")[1].split(",")}
    assert {"chat_request", "chat_request_attempt"} <= names


def test_model_and_migration_check_expressions_are_identical():
    migration = load_migration_7()

    assert model_checks(ChatRequest) == dict(migration.REQUEST_CHECKS)
    assert model_checks(ChatRequestAttempt) == current_attempt_checks()


# ---------- chat_requests: schema ----------

def test_chat_requests_has_the_expected_columns_types_and_nullability():
    columns = {c["name"]: c for c in inspect(engine).get_columns("chat_requests")}

    assert set(columns) == REQUEST_COLUMNS
    assert {n for n, c in columns.items() if c["nullable"]} == REQUEST_NULLABLE
    for name in ("id", "owner_id", "conversation_id", "user_message_id", "assistant_message_id"):
        assert isinstance(columns[name]["type"], sa.Integer), name
    assert isinstance(columns["idempotency_key"]["type"], sa.String)
    assert columns["idempotency_key"]["type"].length == 128
    assert isinstance(columns["request_sha256"]["type"], sa.String)
    assert columns["request_sha256"]["type"].length == 64
    assert isinstance(columns["status"]["type"], sa.String)
    assert isinstance(columns["error_code"]["type"], sa.String)
    assert isinstance(columns["creates_conversation"]["type"], sa.Boolean)
    for name in ("created_at", "completed_at"):
        assert isinstance(columns[name]["type"], sa.DateTime), name
        assert columns[name]["type"].timezone is True, name
    # Only created_at has a default: status and creates_conversation are set by code.
    assert "now()" in columns["created_at"]["default"]
    for name in REQUEST_COLUMNS - {"id", "created_at"}:
        assert columns[name]["default"] is None, name


def test_chat_requests_constraints_foreign_keys_and_indexes_match_the_model():
    insp = inspect(engine)

    in_database = {c["name"] for c in insp.get_check_constraints("chat_requests")}
    assert in_database == set(model_checks(ChatRequest)) == EXPECTED_REQUEST_CHECKS

    [unique] = insp.get_unique_constraints("chat_requests")
    assert unique["name"] == "uq_chat_requests_owner_id_idempotency_key"
    assert unique["column_names"] == ["owner_id", "idempotency_key"]

    foreign_keys = {fk["name"]: fk for fk in insp.get_foreign_keys("chat_requests")}
    expected = {
        "fk_chat_requests_owner_id_users": (["owner_id"], "users", "CASCADE"),
        "fk_chat_requests_conversation_id_conversations":
            (["conversation_id"], "conversations", "SET NULL"),
        "fk_chat_requests_user_message_id_messages":
            (["user_message_id"], "messages", "SET NULL"),
        "fk_chat_requests_assistant_message_id_messages":
            (["assistant_message_id"], "messages", "SET NULL"),
    }
    assert set(foreign_keys) == set(expected)
    for name, (columns, table, ondelete) in expected.items():
        assert foreign_keys[name]["constrained_columns"] == columns, name
        assert foreign_keys[name]["referred_table"] == table, name
        assert foreign_keys[name]["options"].get("ondelete") == ondelete, name

    indexes = {i["name"]: i["column_names"] for i in insp.get_indexes("chat_requests")
               if "duplicates_constraint" not in i}
    assert indexes == {
        "ix_chat_requests_conversation_id": ["conversation_id"],
        "ix_chat_requests_user_message_id": ["user_message_id"],
        "ix_chat_requests_assistant_message_id": ["assistant_message_id"],
    }


# ---------- chat_request_attempts: schema ----------

def test_chat_request_attempts_has_the_expected_columns_types_and_nullability():
    columns = {c["name"]: c for c in inspect(engine).get_columns("chat_request_attempts")}

    assert set(columns) == ATTEMPT_COLUMNS  # in particular: no owner_id
    assert {n for n, c in columns.items() if c["nullable"]} == ATTEMPT_NULLABLE
    for name in ("id", "request_id", "attempt_number", "reserved_tokens", "prompt_tokens",
                 "completion_tokens", "charged_tokens"):
        assert isinstance(columns[name]["type"], sa.Integer), name
    for name in ("status", "error_code"):
        assert isinstance(columns[name]["type"], sa.String), name
    for name in ("lease_expires_at", "created_at", "llm_started_at", "finished_at"):
        assert isinstance(columns[name]["type"], sa.DateTime), name
        assert columns[name]["type"].timezone is True, name
    assert "now()" in columns["created_at"]["default"]
    for name in ATTEMPT_COLUMNS - {"id", "created_at"}:
        assert columns[name]["default"] is None, name


def test_chat_request_attempts_constraints_foreign_key_and_indexes_match_the_model():
    insp = inspect(engine)

    in_database = {c["name"] for c in insp.get_check_constraints("chat_request_attempts")}
    assert in_database == set(model_checks(ChatRequestAttempt)) == EXPECTED_ATTEMPT_CHECKS

    [unique] = insp.get_unique_constraints("chat_request_attempts")
    assert unique["name"] == "uq_chat_request_attempts_request_id_attempt_number"
    assert unique["column_names"] == ["request_id", "attempt_number"]

    [fk] = insp.get_foreign_keys("chat_request_attempts")
    assert fk["name"] == "fk_chat_request_attempts_request_id_chat_requests"
    assert fk["constrained_columns"] == ["request_id"]
    assert fk["referred_table"] == "chat_requests"
    assert fk["options"].get("ondelete") == "CASCADE"

    indexes = {i["name"]: i for i in insp.get_indexes("chat_request_attempts")
               if "duplicates_constraint" not in i}
    assert set(indexes) == {
        "uq_chat_request_attempts_one_running",
        "uq_chat_request_attempts_one_succeeded",
        "ix_chat_request_attempts_llm_started_at",
    }
    running = indexes["uq_chat_request_attempts_one_running"]
    succeeded = indexes["uq_chat_request_attempts_one_succeeded"]
    assert running["unique"] and running["column_names"] == ["request_id"]
    assert succeeded["unique"] and succeeded["column_names"] == ["request_id"]
    assert "'running'" in running["dialect_options"]["postgresql_where"]
    assert "'succeeded'" in succeeded["dialect_options"]["postgresql_where"]
    assert indexes["ix_chat_request_attempts_llm_started_at"]["column_names"] == [
        "llm_started_at"]
    assert not indexes["ix_chat_request_attempts_llm_started_at"]["unique"]


# ---------- chat_requests: behaviour ----------

def test_a_new_request_gets_a_created_at_and_no_links(db, owner_id):
    request = ChatRequest(**request_fields(owner_id))
    db.add(request)
    db.commit()
    db.refresh(request)

    assert request.created_at is not None and request.created_at.tzinfo is not None
    assert (request.conversation_id, request.user_message_id, request.assistant_message_id,
            request.error_code, request.completed_at) == (None, None, None, None, None)


@pytest.mark.parametrize("column", ["owner_id", "idempotency_key", "request_sha256", "status",
                                    "creates_conversation"])
def test_request_required_columns_are_not_nullable(db, owner_id, column):
    fields = request_fields(owner_id)
    fields[column] = None
    request = ChatRequest(**fields)
    db.add(request)
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.diag.column_name == column


def test_duplicate_idempotency_key_for_one_owner_is_rejected(db, owner_id):
    make_request(db, owner_id, "same-key")

    assert_violates(db, ChatRequest(**request_fields(owner_id, "same-key")),
                    "uq_chat_requests_owner_id_idempotency_key")


def test_the_same_idempotency_key_for_different_owners_is_allowed(db, owner_id):
    other = make_user(db, "other@example.com")

    make_request(db, owner_id, "shared-key")
    make_request(db, other, "shared-key")

    assert db.query(ChatRequest).filter_by(idempotency_key="shared-key").count() == 2


@pytest.mark.parametrize("status", ["queued", "SUCCEEDED", "", "failed "])
def test_invalid_request_status_is_rejected(db, owner_id, status):
    # completed_at set and error_code NULL satisfy both consistency CHECKs for any status
    # other than 'in_progress'/'failed', so status_valid is the only rule broken.
    fields = request_fields(owner_id, status=status, completed_at=NOW, error_code=None)
    assert_violates(db, ChatRequest(**fields), "ck_chat_requests_status_valid")


@pytest.mark.parametrize("key", ["a", "A.b_c:d-e", "0" * 128, "550e8400-e29b-41d4-a716-446655440000"])
def test_valid_idempotency_keys_are_accepted(db, owner_id, key):
    make_request(db, owner_id, key)
    assert db.query(ChatRequest).filter_by(idempotency_key=key).count() == 1


@pytest.mark.parametrize("key", ["", "a b", "a/b", "a\nb", "é"])
def test_invalid_idempotency_keys_are_rejected(db, owner_id, key):
    assert_violates(db, ChatRequest(**request_fields(owner_id, key=key)),
                    "ck_chat_requests_idempotency_key_format")


@pytest.mark.parametrize("sha", ["", "a" * 63, "A" * 64, "g" * 64, " " + "a" * 63])
def test_invalid_request_sha256_is_rejected(db, owner_id, sha):
    assert_violates(db, ChatRequest(**request_fields(owner_id, request_sha256=sha)),
                    "ck_chat_requests_request_sha256_format")


@pytest.mark.parametrize("column,value", [("idempotency_key", "x" * 129),
                                          ("request_sha256", "a" * 65)])
def test_over_long_key_and_hash_are_rejected_by_the_column_length(db, owner_id, column, value):
    # String(128) / String(64): PostgreSQL rejects the value before any CHECK runs.
    db.add(ChatRequest(**request_fields(owner_id, **{column: value})
                       if column != "idempotency_key"
                       else request_fields(owner_id, key=value)))
    with pytest.raises(DBAPIError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.pgcode == "22001"       # string_data_right_truncation


@pytest.mark.parametrize("fields", [
    dict(status="in_progress", completed_at=NOW),
    dict(status="succeeded", completed_at=None),
    dict(status="failed", completed_at=None, error_code="attempts_exhausted"),
], ids=["in-progress-with-completed-at", "succeeded-without-completed-at",
        "failed-without-completed-at"])
def test_completed_at_must_be_set_exactly_when_the_request_is_terminal(db, owner_id, fields):
    assert_violates(db, ChatRequest(**request_fields(owner_id, **fields)),
                    "ck_chat_requests_completed_at_consistent")


@pytest.mark.parametrize("fields", [
    dict(status="in_progress", error_code="x"),
    dict(status="succeeded", completed_at=NOW, error_code="x"),
    dict(status="failed", completed_at=NOW, error_code=None),
], ids=["in-progress-with-error", "succeeded-with-error", "failed-without-error"])
def test_error_code_must_be_set_exactly_when_the_request_failed(db, owner_id, fields):
    assert_violates(db, ChatRequest(**request_fields(owner_id, **fields)),
                    "ck_chat_requests_error_code_consistent")


@pytest.mark.parametrize("fields", [
    dict(status="in_progress", completed_at=None),
    dict(status="succeeded", completed_at=NOW),
    dict(status="failed", completed_at=NOW, error_code="conversation_unavailable"),
], ids=["in-progress", "succeeded", "failed"])
def test_every_valid_request_status_is_accepted(db, owner_id, fields):
    make_request(db, owner_id, **fields)
    assert db.query(ChatRequest).count() == 1


def test_a_new_conversation_request_has_no_conversation_until_it_succeeds(db, owner_id):
    conversation_id = make_conversation(db, owner_id)

    # in progress or failed: must not point at a conversation
    assert_violates(db, ChatRequest(**request_fields(
        owner_id, creates_conversation=True, conversation_id=conversation_id)),
        "ck_chat_requests_new_conversation_unset_until_success")
    assert_violates(db, ChatRequest(**request_fields(
        owner_id, creates_conversation=True, conversation_id=conversation_id, status="failed",
        completed_at=NOW, error_code="publish_failed")),
        "ck_chat_requests_new_conversation_unset_until_success")

    # accepted: still unset while in progress; set once succeeded
    make_request(db, owner_id, "k-unset", creates_conversation=True)
    make_request(db, owner_id, "k-done", creates_conversation=True,
                 conversation_id=conversation_id, status="succeeded", completed_at=NOW)
    # an existing-conversation request may point at its target while in progress
    make_request(db, owner_id, "k-existing", creates_conversation=False,
                 conversation_id=conversation_id)


def test_request_for_a_missing_user_conversation_or_message_is_rejected(db, owner_id):
    assert_violates(db, ChatRequest(**request_fields(999_999)),
                    "fk_chat_requests_owner_id_users")
    assert_violates(db, ChatRequest(**request_fields(owner_id, conversation_id=999_999)),
                    "fk_chat_requests_conversation_id_conversations")
    assert_violates(db, ChatRequest(**request_fields(
        owner_id, status="succeeded", completed_at=NOW, user_message_id=999_999)),
        "fk_chat_requests_user_message_id_messages")
    assert_violates(db, ChatRequest(**request_fields(
        owner_id, status="succeeded", completed_at=NOW, assistant_message_id=999_999)),
        "fk_chat_requests_assistant_message_id_messages")


# ---------- chat_request_attempts: behaviour ----------

def test_a_running_attempt_is_accepted_with_and_without_llm_started_at(db, owner_id):
    first = make_request(db, owner_id, "k1")
    second = make_request(db, owner_id, "k2")

    make_attempt(db, first)                                 # reserved, call not yet issued
    make_attempt(db, second, llm_started_at=NOW)            # pre-call committed

    attempt = db.query(ChatRequestAttempt).filter_by(request_id=first).one()
    assert attempt.created_at.tzinfo is not None
    assert (attempt.charged_tokens, attempt.prompt_tokens, attempt.finished_at) == (
        None, None, None)


@pytest.mark.parametrize("name", TERMINAL_NAMES)
def test_every_valid_terminal_attempt_is_accepted(db, request_id, name):
    db.add(terminal_attempt(request_id, name))
    db.commit()

    attempt = db.query(ChatRequestAttempt).one()
    assert attempt.status == TERMINAL[name]["status"]
    assert attempt.charged_tokens == TERMINAL[name]["charged_tokens"]


@pytest.mark.parametrize("status", ["rejected", "failed", "in_progress", "RUNNING", "", "uncertain "])
def test_statuses_outside_the_seven_are_rejected(db, request_id, status):
    # 'rejected' was removed from V1; no other status may exist either.
    # Other CHECKs also reject these rows, so the exact status vocabulary is proven in
    # test_status_valid_expression_accepts_exactly_the_seven_statuses.
    attempt = ChatRequestAttempt(**attempt_fields(request_id, status=status))
    db.add(attempt)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_status_valid_expression_accepts_exactly_the_seven_statuses():
    expression = dict(load_migration_7().ATTEMPT_CHECKS)["ck_chat_request_attempts_status_valid"]
    candidates = ["running", "succeeded", "failed_before_call", "failed_with_usage",
                  "rate_limited", "uncertain", "abandoned",
                  "rejected", "failed", "in_progress", "RUNNING", "", "uncertain "]
    with engine.connect() as conn:
        result = {s: conn.exec_driver_sql(f"SELECT {expression} FROM (SELECT %s AS status) t", (s,))
                  .scalar_one() for s in candidates}

    assert {s for s, ok in result.items() if ok} == set(candidates[:7])
    assert result["rejected"] is False


def test_migration_8_changes_only_the_three_approved_checks_and_only_by_adding_rejected():
    m7 = dict(load_migration_7().ATTEMPT_CHECKS)
    m8 = load_migration_8()
    changed = dict(m8.ATTEMPT_CHECKS_CHANGED)

    assert set(changed) == MIGRATION_8_CHANGED_CHECKS <= set(m7)
    # The previous texts it restores on downgrade are exactly migration 7's.
    assert dict(m8.PREVIOUS_ATTEMPT_CHECKS_CHANGED) == {n: m7[n] for n in changed}
    expected_new = {
        "ck_chat_request_attempts_status_valid": m7[
            "ck_chat_request_attempts_status_valid"].replace(
            "'abandoned')", "'abandoned', 'rejected')"),
        "ck_chat_request_attempts_llm_started_consistent": m7[
            "ck_chat_request_attempts_llm_started_consistent"].replace(
            "'rate_limited', 'uncertain') ", "'rate_limited', 'uncertain', 'rejected') "),
        "ck_chat_request_attempts_charge_consistent": m7[
            "ck_chat_request_attempts_charge_consistent"].replace(
            "'abandoned', 'rate_limited')", "'abandoned', 'rate_limited', 'rejected')"),
    }
    assert changed == expected_new
    assert all(changed[n] != m7[n] for n in changed)


def test_status_valid_after_migration_8_accepts_exactly_the_eight_statuses():
    expression = current_attempt_checks()["ck_chat_request_attempts_status_valid"]
    candidates = ["running", "succeeded", "failed_before_call", "failed_with_usage",
                  "rate_limited", "uncertain", "abandoned", "rejected",
                  "failed", "in_progress", "REJECTED", "", "rejected "]
    with engine.connect() as conn:
        result = {s: conn.exec_driver_sql(f"SELECT {expression} FROM (SELECT %s AS status) t", (s,))
                  .scalar_one() for s in candidates}

    assert {s for s, ok in result.items() if ok} == set(candidates[:8])


def test_every_attempt_check_in_force_is_total_including_rejected():
    checks = list(current_attempt_checks().items())

    nulls = null_results(checks, ATTEMPT_MATRIX)

    assert {name: n for name, n in nulls.items() if name != USAGE} == {
        name: 0 for name in EXPECTED_ATTEMPT_CHECKS - {USAGE}}


def test_rejected_is_accepted_only_as_a_started_zero_charge_attempt_without_usage(db, request_id):
    db.add(terminal_attempt(request_id, "rejected"))
    db.commit()


@pytest.mark.parametrize("overrides,constraints", [
    ({"charged_tokens": 100}, (CHARGE,)),                               # reservation charged
    ({"charged_tokens": 1}, (CHARGE,)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "charged_tokens": 0}, (CHARGE,)),   # usage present
    ({"prompt_tokens": 0, "completion_tokens": 0, "charged_tokens": 0}, (CHARGE,)),
    ({"llm_started_at": None}, (LLM_STARTED,)),                         # never started
    ({"error_code": None}, (ERROR_CODE,)),                              # settled needs a code
    ({"charged_tokens": None}, (NULL_IFF_RUNNING, CHARGE)),
])
def test_rejected_violations_are_rejected_by_the_named_check(db, request_id, overrides, constraints):
    assert_violates(db, terminal_attempt(request_id, "rejected", **overrides), *constraints)


# --- T-S1: every terminal status with charged_tokens = NULL must fail ---

@pytest.mark.parametrize("name", TERMINAL_NAMES)
def test_ts1_a_terminal_attempt_with_a_null_charge_is_rejected(db, owner_id, name):
    # Control: the same row with its correct charge is accepted (so the failure below
    # cannot come from any other constraint).
    control_request = make_request(db, owner_id, "control")
    db.add(terminal_attempt(control_request, name))
    db.commit()

    failing_request = make_request(db, owner_id, "failing")
    assert_violates(db, terminal_attempt(failing_request, name, charged_tokens=None),
                    CHARGE, NULL_IFF_RUNNING)

    assert db.query(ChatRequestAttempt).count() == 1   # only the control row exists


# --- T-S2: wrong charge fails, right charge passes ---

WRONG_CHARGES = [
    ("succeeded_with_usage", dict(charged_tokens=14)),
    ("succeeded_with_usage", dict(charged_tokens=16)),
    ("succeeded_with_usage", dict(charged_tokens=100)),      # reservation instead of actual
    ("succeeded_with_usage", dict(charged_tokens=0)),
    ("succeeded_without_usage", dict(charged_tokens=0)),     # success without usage is NOT free
    ("succeeded_without_usage", dict(charged_tokens=99)),
    ("failed_with_usage", dict(charged_tokens=14)),
    ("failed_with_usage", dict(charged_tokens=0)),
    ("failed_with_usage", dict(prompt_tokens=None, completion_tokens=None, charged_tokens=100)),
    ("failed_before_call", dict(charged_tokens=1)),
    ("failed_before_call", dict(charged_tokens=100)),
    ("failed_before_call", dict(prompt_tokens=10, completion_tokens=5, charged_tokens=0)),
    ("rate_limited", dict(charged_tokens=1)),
    ("rate_limited", dict(prompt_tokens=10, completion_tokens=5, charged_tokens=0)),
    ("uncertain", dict(charged_tokens=0)),                   # uncertain is never free
    ("uncertain", dict(charged_tokens=99)),
    ("uncertain", dict(prompt_tokens=10, completion_tokens=5, charged_tokens=100)),
    ("abandoned", dict(charged_tokens=1)),
    ("abandoned", dict(charged_tokens=100)),
]


@pytest.mark.parametrize("name,overrides", WRONG_CHARGES,
                         ids=[f"{n}-{'-'.join(f'{k}={v}' for k, v in o.items())}"
                              for n, o in WRONG_CHARGES])
def test_ts2_a_wrong_charge_is_rejected(db, request_id, name, overrides):
    assert_violates(db, terminal_attempt(request_id, name, **overrides), CHARGE)


def test_ts2_with_llm_started_at_set_only_rate_limited_can_have_a_zero_charge(db, request_id):
    # No usage reported, call started, charge 0: accepted only as rate_limited.
    no_usage = dict(prompt_tokens=None, completion_tokens=None, charged_tokens=0,
                    llm_started_at=NOW)
    for status in ("succeeded", "failed_with_usage", "uncertain"):
        attempt = ChatRequestAttempt(**attempt_fields(
            request_id, lease_expires_at=None, status=status, finished_at=NOW,
            error_code=None if status == "succeeded" else "x", **no_usage))
        assert_violates(db, attempt, CHARGE)

    db.add(terminal_attempt(request_id, "rate_limited"))   # same shape, accepted
    db.commit()


# --- T-S3: running attempts ---

@pytest.mark.parametrize("overrides", [
    dict(charged_tokens=0),
    dict(charged_tokens=100),
    dict(prompt_tokens=10, completion_tokens=5),
    dict(prompt_tokens=10, completion_tokens=5, charged_tokens=15),
], ids=["charge-0", "charge-100", "usage-only", "usage-and-charge"])
def test_ts3_a_running_attempt_must_have_no_charge_and_no_usage(db, request_id, overrides):
    assert_violates(db, ChatRequestAttempt(**attempt_fields(request_id, **overrides)),
                    CHARGE, NULL_IFF_RUNNING)


# --- single-rule violations ---

@pytest.mark.parametrize("overrides,constraint", [
    (dict(lease_expires_at=None), LEASE),                                          # running, no lease
    ({**TERMINAL["uncertain"], "lease_expires_at": FUTURE}, LEASE),               # terminal, lease
    (dict(finished_at=NOW), FINISHED),                                             # running, finished
    (dict(error_code="x"), ERROR_CODE),                                            # running, error
], ids=["running-no-lease", "terminal-with-lease", "running-finished", "running-with-error"])
def test_running_and_terminal_state_columns_must_agree(db, request_id, overrides, constraint):
    assert_violates(db, ChatRequestAttempt(**attempt_fields(request_id, **overrides)), constraint)


def test_terminal_attempt_needs_finished_at_and_the_right_error_code(db, request_id):
    assert_violates(db, terminal_attempt(request_id, "uncertain", finished_at=None), FINISHED)
    assert_violates(db, terminal_attempt(request_id, "uncertain", error_code=None), ERROR_CODE)
    assert_violates(db, terminal_attempt(request_id, "failed_before_call", error_code=None),
                    ERROR_CODE)
    assert_violates(db, terminal_attempt(request_id, "succeeded_with_usage", error_code="x"),
                    ERROR_CODE)


@pytest.mark.parametrize("name,overrides", [
    ("succeeded_with_usage", dict(llm_started_at=None)),
    ("succeeded_without_usage", dict(llm_started_at=None)),
    ("failed_with_usage", dict(llm_started_at=None)),
    ("rate_limited", dict(llm_started_at=None)),
    ("uncertain", dict(llm_started_at=None)),
    ("failed_before_call", dict(llm_started_at=NOW)),
    ("abandoned", dict(llm_started_at=NOW)),
])
def test_llm_started_at_proves_whether_a_call_was_issued(db, request_id, name, overrides):
    assert_violates(db, terminal_attempt(request_id, name, **overrides), LLM_STARTED)


@pytest.mark.parametrize("value", [0, -1, -100])
def test_non_positive_reserved_tokens_are_rejected(db, request_id, value):
    assert_violates(db, ChatRequestAttempt(**attempt_fields(request_id, reserved_tokens=value)),
                    "ck_chat_request_attempts_reserved_tokens_positive")


@pytest.mark.parametrize("value", [0, -1, -100])
def test_non_positive_attempt_number_is_rejected(db, request_id, value):
    assert_violates(db, ChatRequestAttempt(**attempt_fields(request_id, attempt_number=value)),
                    "ck_chat_request_attempts_attempt_number_positive")


@pytest.mark.parametrize("overrides", [
    dict(prompt_tokens=-1, completion_tokens=6, charged_tokens=5),
    dict(prompt_tokens=6, completion_tokens=-1, charged_tokens=5),
], ids=["negative-prompt", "negative-completion"])
def test_negative_usage_is_rejected_by_usage_valid_alone(db, request_id, overrides):
    # The sum equals the charge, so charge_consistent passes and only usage_valid fails.
    assert_violates(db, terminal_attempt(request_id, "failed_with_usage", **overrides), USAGE)


@pytest.mark.parametrize("overrides", [
    dict(prompt_tokens=10, completion_tokens=None),
    dict(prompt_tokens=None, completion_tokens=5),
], ids=["only-prompt", "only-completion"])
def test_usage_must_be_reported_as_a_pair(db, request_id, overrides):
    assert_violates(db, terminal_attempt(request_id, "succeeded_without_usage", **overrides),
                    USAGE, CHARGE)


def test_attempt_required_columns_are_not_nullable(db, request_id):
    for column in ("request_id", "attempt_number", "status", "reserved_tokens"):
        fields = attempt_fields(request_id)
        fields[column] = None
        attempt = ChatRequestAttempt(**fields)
        db.add(attempt)
        with pytest.raises(IntegrityError) as excinfo:
            db.flush()
        db.rollback()
        assert excinfo.value.orig.diag.column_name == column


def test_attempt_for_a_missing_request_is_rejected(db):
    assert_violates(db, ChatRequestAttempt(**attempt_fields(999_999)),
                    "fk_chat_request_attempts_request_id_chat_requests")


# --- uniqueness and the partial unique indexes ---

def test_duplicate_attempt_number_in_one_request_is_rejected(db, request_id):
    db.add(terminal_attempt(request_id, "uncertain", attempt_number=1))
    db.commit()

    assert_violates(db, terminal_attempt(request_id, "abandoned", attempt_number=1),
                    "uq_chat_request_attempts_request_id_attempt_number")


def test_the_same_attempt_number_in_different_requests_is_allowed(db, owner_id):
    first = make_request(db, owner_id, "k1")
    second = make_request(db, owner_id, "k2")

    make_attempt(db, first, attempt_number=1)
    make_attempt(db, second, attempt_number=1)

    assert db.query(ChatRequestAttempt).count() == 2


def test_only_one_running_attempt_per_request(db, owner_id, request_id):
    make_attempt(db, request_id, attempt_number=1)

    assert_violates(db, ChatRequestAttempt(**attempt_fields(request_id, attempt_number=2)),
                    "uq_chat_request_attempts_one_running")
    # another request may have its own running attempt
    make_attempt(db, make_request(db, owner_id, "k2"), attempt_number=1)


def test_only_one_succeeded_attempt_per_request(db, request_id):
    db.add(terminal_attempt(request_id, "succeeded_with_usage", attempt_number=1))
    db.commit()

    assert_violates(db, terminal_attempt(request_id, "succeeded_without_usage", attempt_number=2),
                    "uq_chat_request_attempts_one_succeeded")


def test_a_request_may_have_many_settled_attempts_then_a_running_one_and_a_success(db, request_id):
    # uncertain, abandoned, rate_limited, failed_with_usage: no uniqueness limit on settled
    # attempts, and a settled history does not block a running attempt or one success.
    for number, name in enumerate(["uncertain", "uncertain", "abandoned", "rate_limited",
                                   "failed_before_call", "failed_with_usage"], start=1):
        db.add(terminal_attempt(request_id, name, attempt_number=number))
    db.commit()
    make_attempt(db, request_id, attempt_number=7)

    # the running attempt finishes as the single success
    attempt = db.query(ChatRequestAttempt).filter_by(attempt_number=7).one()
    for key, value in dict(TERMINAL["succeeded_with_usage"], lease_expires_at=None).items():
        setattr(attempt, key, value)
    db.commit()

    assert [a.attempt_number for a in db.get(ChatRequest, request_id).attempts] == list(range(1, 8))
    assert db.query(ChatRequestAttempt).filter_by(status="succeeded").count() == 1


def test_the_schema_does_not_forbid_a_running_attempt_next_to_a_succeeded_one(db, request_id):
    # The two partial indexes are independent: each limits only its own status, so the
    # schema does not forbid running+succeeded for one request. That sequencing rule
    # belongs to C8; this pins the schema's actual behaviour.
    db.add(terminal_attempt(request_id, "succeeded_with_usage", attempt_number=1))
    db.commit()
    make_attempt(db, request_id, attempt_number=2)

    assert db.query(ChatRequestAttempt).count() == 2


# ---------- T-S4: every CHECK is total ----------

STATUSES = ["running", "succeeded", "failed_before_call", "failed_with_usage",
            "rate_limited", "uncertain", "abandoned", "rejected"]


def null_results(checks, source_sql, only=None):
    """{constraint name: number of matrix rows where the expression evaluates to NULL}."""
    found = {}
    with engine.connect() as conn:
        for name, expression in checks:
            if only is not None and name not in only:
                continue
            found[name] = conn.exec_driver_sql(
                f"SELECT count(*) FROM (SELECT ({expression}) AS result FROM {source_sql}) m "
                "WHERE result IS NULL").scalar_one()
    return found


ATTEMPT_MATRIX = """
    (VALUES ('running'), ('succeeded'), ('failed_before_call'), ('failed_with_usage'),
            ('rate_limited'), ('uncertain'), ('abandoned'), ('rejected')) AS s(status)
    CROSS JOIN (VALUES (1), (0)) AS a(attempt_number)
    CROSS JOIN (VALUES (100), (0)) AS r(reserved_tokens)
    CROSS JOIN (VALUES (NULL::int), (10), (-1)) AS p(prompt_tokens)
    CROSS JOIN (VALUES (NULL::int), (5), (-1)) AS c(completion_tokens)
    CROSS JOIN (VALUES (NULL::int), (0), (15), (100)) AS g(charged_tokens)
    CROSS JOIN (VALUES (NULL::timestamptz), (now())) AS l(lease_expires_at)
    CROSS JOIN (VALUES (NULL::timestamptz), (now())) AS f(finished_at)
    CROSS JOIN (VALUES (NULL::text), ('code')) AS e(error_code)
    CROSS JOIN (VALUES (NULL::timestamptz), (now())) AS t(llm_started_at)
"""

REQUEST_MATRIX = """
    (VALUES ('in_progress'), ('succeeded'), ('failed'), ('bogus')) AS s(status)
    CROSS JOIN (VALUES (true), (false)) AS c(creates_conversation)
    CROSS JOIN (VALUES (NULL::int), (1)) AS v(conversation_id)
    CROSS JOIN (VALUES (NULL::timestamptz), (now())) AS d(completed_at)
    CROSS JOIN (VALUES (NULL::text), ('code')) AS e(error_code)
    CROSS JOIN (VALUES ('ok-key'), (''), ('a b')) AS k(idempotency_key)
    CROSS JOIN (VALUES ('%s'), ('zz')) AS h(request_sha256)
""" % VALID_SHA


def test_ts4_every_attempt_check_except_usage_valid_is_total():
    checks = load_migration_7().ATTEMPT_CHECKS

    nulls = null_results(checks, ATTEMPT_MATRIX)

    assert set(nulls) == EXPECTED_ATTEMPT_CHECKS
    assert {name: n for name, n in nulls.items() if name != USAGE} == {
        name: 0 for name in EXPECTED_ATTEMPT_CHECKS - {USAGE}}


def test_ts4_usage_valid_is_null_only_when_both_usage_values_are_absent():
    checks = dict(load_migration_7().ATTEMPT_CHECKS)
    expression = checks[USAGE]
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"SELECT prompt_tokens, completion_tokens, ({expression}) AS result FROM "
            "(VALUES (NULL::int), (10), (-1)) AS p(prompt_tokens) CROSS JOIN "
            "(VALUES (NULL::int), (5), (-1)) AS c(completion_tokens)").all()

    for prompt, completion, result in rows:
        if prompt is None and completion is None:
            assert result is None            # deliberate: absent usage is valid
        else:
            assert result in (True, False), (prompt, completion)


def test_ts4_every_request_check_is_total():
    nulls = null_results(load_migration_7().REQUEST_CHECKS, REQUEST_MATRIX)

    assert nulls == {name: 0 for name in EXPECTED_REQUEST_CHECKS}


# ---------- T-S5: the R1 loophole stays closed ----------

NULL_CHARGE_ROWS = [
    # (status, prompt_tokens, completion_tokens)
    ("succeeded", 10, 5), ("succeeded", None, None), ("failed_with_usage", 10, 5),
    ("failed_before_call", None, None), ("rate_limited", None, None),
    ("uncertain", None, None), ("abandoned", None, None),
]


@pytest.mark.parametrize("status,prompt,completion", NULL_CHARGE_ROWS)
def test_ts5_r1_charge_check_accepted_a_null_charge_and_the_frozen_check_does_not(
        status, prompt, completion):
    frozen = load_migration_7().ATTEMPT_CHARGE_CONSISTENT
    row = ("(SELECT %s::text AS status, %s::int AS prompt_tokens, %s::int AS completion_tokens, "
           "NULL::int AS charged_tokens, 100 AS reserved_tokens) t")
    with engine.connect() as conn:
        r1 = conn.exec_driver_sql(
            f"SELECT ({R1_CHARGE_CONSISTENT_REGRESSION}) FROM {row}",
            (status, prompt, completion)).scalar_one()
        r3 = conn.exec_driver_sql(
            f"SELECT ({frozen}) FROM {row}", (status, prompt, completion)).scalar_one()

    assert r1 is None      # NULL: PostgreSQL would have ACCEPTED the row
    assert r3 is False     # FALSE: the frozen expression rejects it


# ---------- deletion behaviour ----------

def seed_succeeded_request(db, owner_id, key="k-done"):
    """A succeeded existing-conversation request with its messages and one succeeded attempt."""
    conversation_id = make_conversation(db, owner_id)
    user_message_id, assistant_message_id = make_message_pair(db, conversation_id)
    request_id = make_request(
        db, owner_id, key, status="succeeded", completed_at=NOW, conversation_id=conversation_id,
        user_message_id=user_message_id, assistant_message_id=assistant_message_id)
    db.add(terminal_attempt(request_id, "succeeded_with_usage"))
    db.commit()
    return request_id, conversation_id, user_message_id, assistant_message_id


def test_deleting_a_conversation_keeps_its_requests_and_attempts_with_null_links(db, owner_id):
    request_id, conversation_id, _, _ = seed_succeeded_request(db, owner_id)
    # an in-progress request for the same conversation with a running attempt
    pending_id = make_request(db, owner_id, "k-pending", conversation_id=conversation_id)
    make_attempt(db, pending_id)

    db.delete(db.get(Conversation, conversation_id))
    db.commit()

    db.expire_all()
    assert db.query(Message).count() == 0                      # messages cascaded
    request = db.get(ChatRequest, request_id)
    assert request.status == "succeeded"
    assert (request.conversation_id, request.user_message_id, request.assistant_message_id) == (
        None, None, None)
    pending = db.get(ChatRequest, pending_id)
    assert (pending.status, pending.conversation_id) == ("in_progress", None)
    assert db.query(ChatRequestAttempt).count() == 2           # the quota ledger survives
    assert db.query(ChatRequestAttempt).filter_by(status="running").count() == 1


def test_deleting_one_message_nulls_only_that_link(db, owner_id):
    request_id, _, user_message_id, assistant_message_id = seed_succeeded_request(db, owner_id)

    db.execute(text("DELETE FROM messages WHERE id = :id"), {"id": assistant_message_id})
    db.commit()

    db.expire_all()
    request = db.get(ChatRequest, request_id)
    assert request.assistant_message_id is None
    assert request.user_message_id == user_message_id
    assert request.conversation_id is not None
    assert db.query(ChatRequestAttempt).count() == 1


def test_deleting_a_request_deletes_its_attempts_only(db, owner_id):
    doomed, _, _, _ = seed_succeeded_request(db, owner_id, "k-doomed")
    kept = make_request(db, owner_id, "k-kept")
    make_attempt(db, kept)

    db.delete(db.get(ChatRequest, doomed))
    db.commit()

    db.expire_all()
    assert db.query(ChatRequestAttempt).filter_by(request_id=doomed).count() == 0
    assert db.query(ChatRequestAttempt).filter_by(request_id=kept).count() == 1


def test_the_database_cascades_a_raw_request_delete_to_attempts(db, request_id):
    make_attempt(db, request_id)

    db.execute(text("DELETE FROM chat_requests WHERE id = :id"), {"id": request_id})
    db.commit()

    assert db.query(ChatRequestAttempt).count() == 0


def test_deleting_a_user_deletes_all_their_chat_data(client, make_user, db):
    alice = make_user("alice@example.com")
    alice_id = db.query(User).filter_by(email="alice@example.com").one().id
    seed_succeeded_request(db, alice_id, "a1")
    make_attempt(db, make_request(db, alice_id, "a2"))
    assert counts(db) == {"conversations": 1, "messages": 2, "requests": 2, "attempts": 2}

    assert client.delete(ME, headers=alice).status_code == 204

    db.expire_all()
    assert db.get(User, alice_id) is None
    assert counts(db) == {"conversations": 0, "messages": 0, "requests": 0, "attempts": 0}


def test_deleting_one_users_data_leaves_other_users_requests_and_attempts_untouched(
        client, make_user, db):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    alice_id = db.query(User).filter_by(email="alice@example.com").one().id
    bob_id = db.query(User).filter_by(email="bob@example.com").one().id
    seed_succeeded_request(db, alice_id, "shared-key")
    bob_request, bob_conversation, bob_user_msg, bob_assistant_msg = seed_succeeded_request(
        db, bob_id, "shared-key")      # the same key is fine for another owner
    make_attempt(db, make_request(db, bob_id, "bob-pending"))

    assert client.delete(ME, headers=alice).status_code == 204

    db.expire_all()
    assert counts(db) == {"conversations": 1, "messages": 2, "requests": 2, "attempts": 2}
    request = db.get(ChatRequest, bob_request)
    assert (request.conversation_id, request.user_message_id, request.assistant_message_id) == (
        bob_conversation, bob_user_msg, bob_assistant_msg)
    assert db.get(User, bob_id) is not None


# ---------- test-database cleanup ----------

def test_the_conftest_cleanup_statement_also_clears_the_new_tables(db, owner_id):
    seed_succeeded_request(db, owner_id)
    make_attempt(db, make_request(db, owner_id, "k-pending"))
    db.rollback()    # end the read transaction opened by the last attribute access, so
                     # TRUNCATE (ACCESS EXCLUSIVE) is not blocked by this session

    # The exact statement conftest's _clean_tables fixture runs after every test.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE documents, users RESTART IDENTITY CASCADE"))

    db.expire_all()
    assert counts(db) == {"conversations": 0, "messages": 0, "requests": 0, "attempts": 0}
