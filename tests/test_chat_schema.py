"""C7.1 schema tests: conversations and messages (migration #6).

Runs against documind_test, which conftest builds with `alembic upgrade head`,
so these assertions check the migrated database, not just ORM metadata.
Chat request/attempt tables (migration #7) are out of scope here.
"""
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, inspect, text
from sqlalchemy.exc import IntegrityError

from app.core.database import engine
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ME = "/api/v1/auth/me"

MIGRATION_6_REVISION = "4a06e23062ae"
MIGRATION_6_DOWN_REVISION = "0b0212c1cca6"

CONVERSATION_COLUMNS = {"id", "owner_id", "title", "created_at", "updated_at"}
MESSAGE_COLUMNS = {"id", "conversation_id", "seq", "role", "content", "created_at"}
EXPECTED_CONVERSATION_CHECKS = {"ck_conversations_title_not_blank"}
EXPECTED_MESSAGE_CHECKS = {
    "ck_messages_seq_positive",
    "ck_messages_role_valid",
    "ck_messages_content_not_blank",
}


# ---------- helpers ----------

def model_check_names(model):
    return {c.name for c in model.__table__.constraints if isinstance(c, CheckConstraint)}


def make_message(conversation_id, seq=1, role="user", content="What is pgvector?", **overrides):
    """A message that satisfies every constraint unless overridden."""
    fields = dict(conversation_id=conversation_id, seq=seq, role=role, content=content)
    fields.update(overrides)
    return Message(**fields)


def assert_violates(db, instance, constraint_name):
    """Insert must fail, and fail on exactly the named constraint."""
    db.add(instance)
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.diag.constraint_name == constraint_name


def make_user(db, email):
    user = User(email=email, hashed_password="not-a-real-hash")
    db.add(user)
    db.commit()
    return user.id


def make_conversation(db, owner_id, title=None):
    conversation = Conversation(owner_id=owner_id, title=title)
    db.add(conversation)
    db.commit()
    return conversation.id


def add_messages(db, conversation_id, count):
    """`count` alternating user/assistant messages with seq 1..count."""
    for seq in range(1, count + 1):
        role = "user" if seq % 2 else "assistant"
        db.add(make_message(conversation_id, seq=seq, role=role, content=f"message {seq}"))
    db.commit()


def message_count(db, conversation_id=None):
    query = db.query(Message)
    if conversation_id is not None:
        query = query.filter(Message.conversation_id == conversation_id)
    return query.count()


@pytest.fixture
def owner_id(db):
    return make_user(db, "owner@example.com")


@pytest.fixture
def conversation_id(db, owner_id):
    return make_conversation(db, owner_id)


# ---------- migration #6 position and registration ----------

def test_migration_6_follows_the_c1_head_and_is_the_single_head():
    script = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))

    revision = script.get_revision(MIGRATION_6_REVISION)
    assert revision.down_revision == MIGRATION_6_DOWN_REVISION
    # Later migrations (C7.2's #7, ...) build on #6, so only require a single head
    # that still has #6 in its history. The exact head is asserted by the newest test.
    assert len(script.get_heads()) == 1
    assert MIGRATION_6_REVISION in {r.revision for r in script.walk_revisions()}


def test_models_are_registered_in_app_main_and_alembic_env():
    import app.main

    assert app.main.conversation.Conversation is Conversation
    assert app.main.message.Message is Message

    env_source = (PROJECT_ROOT / "migrations" / "env.py").read_text(encoding="utf-8")
    [import_line] = [ln for ln in env_source.splitlines() if ln.startswith("from app.models import")]
    assert {"conversation", "message"} <= {n.strip() for n in import_line.split("#")[0].split("import")[1].split(",")}


# ---------- conversations: schema ----------

def test_conversations_has_the_expected_columns_types_and_nullability():
    columns = {c["name"]: c for c in inspect(engine).get_columns("conversations")}

    assert set(columns) == CONVERSATION_COLUMNS
    assert {name for name, c in columns.items() if c["nullable"]} == {"title"}
    assert isinstance(columns["id"]["type"], sa.Integer)
    assert isinstance(columns["owner_id"]["type"], sa.Integer)
    assert isinstance(columns["title"]["type"], sa.String)
    for name in ("created_at", "updated_at"):
        assert isinstance(columns[name]["type"], sa.DateTime), name
        assert columns[name]["type"].timezone is True, name
        assert "now()" in columns[name]["default"], name


def test_conversations_constraints_and_index_match_the_model():
    insp = inspect(engine)

    in_database = {c["name"] for c in insp.get_check_constraints("conversations")}
    assert in_database == model_check_names(Conversation) == EXPECTED_CONVERSATION_CHECKS

    [fk] = insp.get_foreign_keys("conversations")
    assert fk["name"] == "fk_conversations_owner_id_users"
    assert fk["constrained_columns"] == ["owner_id"]
    assert fk["referred_table"] == "users"
    assert fk["options"].get("ondelete") == "CASCADE"

    indexes = {i["name"]: i for i in insp.get_indexes("conversations")}
    assert set(indexes) == {"ix_conversations_owner_id_updated_at"}
    assert indexes["ix_conversations_owner_id_updated_at"]["column_names"] == [
        "owner_id", "updated_at",
    ]
    assert insp.get_unique_constraints("conversations") == []


# ---------- messages: schema ----------

def test_messages_has_the_expected_columns_types_and_nullability_and_no_owner():
    columns = {c["name"]: c for c in inspect(engine).get_columns("messages")}

    assert set(columns) == MESSAGE_COLUMNS  # in particular: no owner_id, no status
    assert {name for name, c in columns.items() if c["nullable"]} == set()
    assert isinstance(columns["id"]["type"], sa.Integer)
    assert isinstance(columns["conversation_id"]["type"], sa.Integer)
    assert isinstance(columns["seq"]["type"], sa.Integer)
    assert isinstance(columns["role"]["type"], sa.String)
    assert isinstance(columns["content"]["type"], sa.Text)
    assert isinstance(columns["created_at"]["type"], sa.DateTime)
    assert columns["created_at"]["type"].timezone is True
    assert "now()" in columns["created_at"]["default"]


def test_messages_constraints_match_the_model():
    insp = inspect(engine)

    in_database = {c["name"] for c in insp.get_check_constraints("messages")}
    assert in_database == model_check_names(Message) == EXPECTED_MESSAGE_CHECKS

    [unique] = insp.get_unique_constraints("messages")
    assert unique["name"] == "uq_messages_conversation_id_seq"
    assert unique["column_names"] == ["conversation_id", "seq"]

    [fk] = insp.get_foreign_keys("messages")
    assert fk["name"] == "fk_messages_conversation_id_conversations"
    assert fk["constrained_columns"] == ["conversation_id"]
    assert fk["referred_table"] == "conversations"
    assert fk["options"].get("ondelete") == "CASCADE"


# ---------- conversations: behaviour ----------

def test_a_new_conversation_gets_timestamps_and_no_title(db, owner_id):
    conversation = Conversation(owner_id=owner_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    assert conversation.title is None
    assert conversation.created_at is not None and conversation.created_at.tzinfo is not None
    assert conversation.updated_at is not None and conversation.updated_at.tzinfo is not None


@pytest.mark.parametrize("title", [None, "Questions about FastAPI", "x"])
def test_valid_conversation_titles_are_accepted(db, owner_id, title):
    make_conversation(db, owner_id, title=title)
    assert db.query(Conversation).filter_by(title=title).count() == 1


@pytest.mark.parametrize("title", ["", " ", "     "])
def test_blank_conversation_titles_are_rejected(db, owner_id, title):
    assert_violates(db, Conversation(owner_id=owner_id, title=title),
                    "ck_conversations_title_not_blank")


def test_conversation_for_a_missing_user_is_rejected(db):
    assert_violates(db, Conversation(owner_id=999_999), "fk_conversations_owner_id_users")


# ---------- messages: behaviour ----------

def test_a_valid_message_pair_is_accepted_and_ordered_by_seq(db, conversation_id):
    # Insert out of order to prove ordering comes from seq, not insertion order.
    db.add(make_message(conversation_id, seq=2, role="assistant", content="An extension."))
    db.add(make_message(conversation_id, seq=1, role="user", content="What is pgvector?"))
    db.commit()

    conversation = db.get(Conversation, conversation_id)
    assert [(m.seq, m.role) for m in conversation.messages] == [(1, "user"), (2, "assistant")]
    assert all(m.created_at is not None and m.created_at.tzinfo is not None
               for m in conversation.messages)


@pytest.mark.parametrize("seq", [0, -1, -100])
def test_non_positive_seq_is_rejected(db, conversation_id, seq):
    assert_violates(db, make_message(conversation_id, seq=seq), "ck_messages_seq_positive")


def test_seq_one_is_accepted(db, conversation_id):
    db.add(make_message(conversation_id, seq=1))
    db.commit()
    assert message_count(db, conversation_id) == 1


def test_duplicate_seq_in_one_conversation_is_rejected(db, conversation_id):
    add_messages(db, conversation_id, 2)

    assert_violates(db, make_message(conversation_id, seq=2, role="assistant"),
                    "uq_messages_conversation_id_seq")
    assert message_count(db, conversation_id) == 2


def test_same_seq_in_different_conversations_is_allowed(db, owner_id):
    first = make_conversation(db, owner_id)
    second = make_conversation(db, owner_id)

    add_messages(db, first, 2)
    add_messages(db, second, 2)

    assert message_count(db, first) == 2
    assert message_count(db, second) == 2


@pytest.mark.parametrize("role", ["system", "tool", "", "USER", "User", "assistant "])
def test_invalid_roles_are_rejected(db, conversation_id, role):
    assert_violates(db, make_message(conversation_id, role=role), "ck_messages_role_valid")


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_valid_roles_are_accepted(db, conversation_id, role):
    db.add(make_message(conversation_id, role=role))
    db.commit()
    assert message_count(db, conversation_id) == 1


# Note: PostgreSQL trim() removes spaces only (same rule as ck_documents_filename_not_blank),
# so the cases here are the empty string and space-only strings.
@pytest.mark.parametrize("content", ["", " ", "      "])
def test_blank_message_content_is_rejected(db, conversation_id, content):
    assert_violates(db, make_message(conversation_id, content=content),
                    "ck_messages_content_not_blank")


def test_message_content_is_stored_exactly(db, conversation_id):
    content = "  Leading and trailing spaces, unicode: héllo ✓\nand a second line.  "
    db.add(make_message(conversation_id, content=content))
    db.commit()
    db.expire_all()

    assert db.query(Message).one().content == content


def test_message_for_a_missing_conversation_is_rejected(db):
    assert_violates(db, make_message(999_999), "fk_messages_conversation_id_conversations")


@pytest.mark.parametrize("column", ["conversation_id", "seq", "role", "content"])
def test_message_columns_are_not_nullable(db, conversation_id, column):
    message = make_message(conversation_id)
    setattr(message, column, None)
    db.add(message)
    with pytest.raises(IntegrityError) as excinfo:
        db.flush()
    db.rollback()
    assert excinfo.value.orig.diag.column_name == column


# ---------- cascades ----------

def test_deleting_a_conversation_deletes_all_its_messages(db, owner_id):
    doomed = make_conversation(db, owner_id)
    kept = make_conversation(db, owner_id)
    add_messages(db, doomed, 4)
    add_messages(db, kept, 2)

    db.delete(db.get(Conversation, doomed))
    db.commit()

    db.expire_all()
    assert db.get(Conversation, doomed) is None
    assert message_count(db, doomed) == 0
    assert db.get(Conversation, kept) is not None
    assert message_count(db, kept) == 2


def test_the_database_cascades_a_raw_conversation_delete_to_messages(db, conversation_id):
    add_messages(db, conversation_id, 3)

    db.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conversation_id})
    db.commit()

    assert message_count(db) == 0


def test_deleting_a_user_deletes_their_conversations_and_messages(client, make_user, db):
    alice = make_user("alice@example.com")
    alice_id = db.query(User).filter_by(email="alice@example.com").one().id
    alice_conversations = [make_conversation(db, alice_id, title=t) for t in ("one", "two")]
    for conversation in alice_conversations:
        add_messages(db, conversation, 2)
    alice_message_ids = {
        m.id for m in db.query(Message).filter(Message.conversation_id.in_(alice_conversations))
    }
    assert len(alice_message_ids) == 4

    assert client.delete(ME, headers=alice).status_code == 204

    db.expire_all()
    assert db.get(User, alice_id) is None
    for conversation in alice_conversations:
        assert db.get(Conversation, conversation) is None, conversation
    assert db.query(Message).filter(Message.id.in_(alice_message_ids)).count() == 0
    assert db.query(Conversation).count() == 0
    assert message_count(db) == 0


def test_deleting_one_users_data_leaves_other_users_chat_data_untouched(client, make_user, db):
    alice = make_user("alice@example.com")
    make_user("bob@example.com")
    alice_id = db.query(User).filter_by(email="alice@example.com").one().id
    bob_id = db.query(User).filter_by(email="bob@example.com").one().id
    alice_conversations = [make_conversation(db, alice_id) for _ in range(2)]
    bob_conversation = make_conversation(db, bob_id, title="bob's chat")
    for conversation in alice_conversations:
        add_messages(db, conversation, 2)
    add_messages(db, bob_conversation, 3)
    bob_message_ids = {m.id for m in db.query(Message).filter_by(conversation_id=bob_conversation)}

    # A single conversation of Alice's, then the whole account.
    db.delete(db.get(Conversation, alice_conversations[0]))
    db.commit()
    assert client.delete(ME, headers=alice).status_code == 204

    db.expire_all()
    assert {m.id for m in db.query(Message).all()} == bob_message_ids
    assert [c.id for c in db.query(Conversation).all()] == [bob_conversation]
    assert db.get(Conversation, bob_conversation).title == "bob's chat"
    assert db.get(User, bob_id) is not None


# ---------- test-database cleanup ----------

def test_the_conftest_cleanup_statement_also_clears_the_chat_tables(db, owner_id):
    conversation = make_conversation(db, owner_id)
    add_messages(db, conversation, 2)

    # The exact statement conftest's _clean_tables fixture runs after every test.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE documents, users RESTART IDENTITY CASCADE"))

    db.expire_all()
    assert db.query(Conversation).count() == 0
    assert message_count(db) == 0
