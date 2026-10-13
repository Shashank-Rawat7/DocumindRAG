import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://documind_user:documind_pass@localhost:5432/documind_test",
)

if not (make_url(TEST_DATABASE_URL).database or "").endswith("_test"):
    raise RuntimeError("Refusing to run tests: the test database name must end with '_test'.")

# These MUST be set before the first `import app...`: Settings() and the
# database engine are created at import time.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["SECRET_KEY"] = "test-secret-key-for-pytest-only-0123456789"
os.environ["ENVIRONMENT"] = "test"

from fastapi.testclient import TestClient  # noqa: E402

from app.core import storage  # noqa: E402
from app.core.database import SessionLocal, engine # noqa: E402
from app.core.rate_limit import limiter  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    expected_database = make_url(TEST_DATABASE_URL).database

    if not expected_database or not expected_database.endswith("_test"):
        raise RuntimeError(
            "Refusing to reset database: expected database must end with '_test'."
        )

    if engine.url.database != expected_database:
        raise RuntimeError(
            "Refusing to reset database: the application engine URL "
            "does not match TEST_DATABASE_URL."
        )

    try:
        with engine.begin() as conn:
            actual_database = conn.execute(
                text("SELECT current_database()")
            ).scalar_one()

            if actual_database != expected_database:
                raise RuntimeError(
                    f"Refusing to reset database '{actual_database}'; "
                    f"expected '{expected_database}'."
                )

            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    except OperationalError:
        pytest.exit(
            "Cannot reach the test database. Is Docker running, and did you create "
            "'documind_test'? (see the CREATE DATABASE command)",
            returncode=2,
        )

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    yield
    engine.dispose()


@pytest.fixture(autouse=True)
def _clean_tables():
    yield
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE documents, users RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def _limiter_off():
    limiter.enabled = False
    yield
    limiter.reset()
    limiter.enabled = True


@pytest.fixture(autouse=True)
def _upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_ROOT", tmp_path.resolve())


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def make_user(client):
    def _make_user(email="alice@example.com", password="password123"):
        response = client.post(
            "/api/v1/auth/register", json={"email": email, "password": password}
        )
        assert response.status_code == 201, response.text

        response = client.post(
            "/api/v1/auth/login", json={"email": email, "password": password}
        )
        assert response.status_code == 200, response.text

        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return _make_user


@pytest.fixture
def auth_headers(make_user):
    return make_user()