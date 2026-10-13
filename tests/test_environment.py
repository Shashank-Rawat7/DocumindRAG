from sqlalchemy import inspect
from app.core.security import hash_password
from app.models.user import User
from app.core.database import engine


def test_app_is_connected_to_the_test_database():
    assert engine.url.database.endswith("_test")


def test_database_cleanup_removes_users(db):
    user = User(
        email="cleanup@example.com",
        hashed_password=hash_password("password123"),
    )
    db.add(user)
    db.commit()

    assert db.query(User).count() == 1


def test_schema_was_built_by_migrations():
    inspector = inspect(engine)

    assert {"users", "documents", "alembic_version"} <= set(inspector.get_table_names())

    columns = {c["name"] for c in inspector.get_columns("documents")}
    assert {
        "stored_name", "content_type", "size_bytes",
        "processing_status", "processing_error", "extracted_text",
    } <= columns

    constraints = {c["name"] for c in inspector.get_check_constraints("documents")}
    assert "ck_documents_filename_not_blank" in constraints
    assert "ck_documents_processing_status_valid" in constraints


def test_documents_foreign_key_cascades_on_user_delete():
    foreign_keys = inspect(engine).get_foreign_keys("documents")

    assert any(fk["options"].get("ondelete") == "CASCADE" for fk in foreign_keys)

def test_isolation_part_1_creates_a_user(make_user, db):
    make_user()

    assert db.query(User).count() == 1


def test_isolation_part_2_finds_nothing_left_behind(db):
    assert db.query(User).count() == 0


def test_database_is_clean_at_start_of_test(db):
    assert db.query(User).count() == 0
