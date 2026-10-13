from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.config import settings
from app.core.security import decode_access_token
from app.models.document import Document
from app.models.user import User

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/auth/me"
DOCUMENTS = "/api/v1/documents"

EMAIL = "alice@example.com"
PASSWORD = "password123"


def register(client, email=EMAIL, password=PASSWORD):
    return client.post(REGISTER, json={"email": email, "password": password})


def login(client, email=EMAIL, password=PASSWORD):
    return client.post(LOGIN, json={"email": email, "password": password})


def make_token(email=EMAIL, *, expires_in=timedelta(minutes=5), secret=None, include_sub=True):
    claims = {"exp": datetime.now(timezone.utc) + expires_in}
    if include_sub:
        claims["sub"] = email
    return jwt.encode(claims, secret or settings.secret_key, algorithm=settings.algorithm)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ---------- registration ----------

def test_register_returns_the_new_user_without_any_password_field(client):
    response = register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == EMAIL
    assert isinstance(body["id"], int)
    assert set(body) == {"id", "email"}


def test_register_stores_a_bcrypt_hash_not_the_password(client, db):
    register(client)

    user = db.query(User).filter(User.email == EMAIL).one()
    assert user.hashed_password != PASSWORD
    assert user.hashed_password.startswith("$2")


def test_register_normalizes_email(client):
    response = register(client, email="  Alice@Example.COM ")

    assert response.status_code == 201
    assert response.json()["email"] == EMAIL


def test_register_duplicate_email_is_rejected(client):
    assert register(client).status_code == 201

    response = register(client)

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


def test_register_duplicate_email_is_detected_regardless_of_case(client):
    register(client)

    response = register(client, email="ALICE@EXAMPLE.COM")

    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": EMAIL, "password": "short"},
        {"email": EMAIL, "password": "a" * 73},
        {"email": EMAIL},
        {"password": PASSWORD},
        {},
    ],
    ids=[
        "bad-email",
        "password-too-short",
        "password-too-long",
        "missing-password",
        "missing-email",
        "empty-body",
    ],
)
def test_register_rejects_invalid_input(client, payload):
    response = client.post(REGISTER, json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("length", [8, 72])
def test_register_accepts_password_length_boundaries(client, length):
    password = "a" * length

    assert register(client, password=password).status_code == 201
    assert login(client, password=password).status_code == 200


# ---------- login ----------

def test_login_returns_a_bearer_token_for_the_user(client):
    register(client)

    response = login(client)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"

    payload = decode_access_token(body["access_token"])
    assert payload["sub"] == EMAIL

    expected_exp = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    assert abs(payload["exp"] - expected_exp.timestamp()) < 60


def test_login_failures_do_not_reveal_whether_the_email_exists(client):
    register(client)

    wrong_password = login(client, password="wrong-password")
    unknown_email = login(client, email="nobody@example.com")

    assert wrong_password.status_code == 401
    assert unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json()["detail"] == "Incorrect email or password"


def test_login_email_is_case_insensitive(client):
    register(client)

    assert login(client, email="ALICE@Example.com").status_code == 200


# ---------- token handling ----------

def test_valid_token_grants_access(client, auth_headers):
    response = client.get(DOCUMENTS, headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_handmade_token_helper_produces_accepted_tokens(client):
    register(client)

    response = client.get(DOCUMENTS, headers=bearer(make_token()))

    assert response.status_code == 200


def test_protected_route_without_token_is_rejected(client):
    response = client.get(DOCUMENTS)

    assert response.status_code == 401


def test_garbage_token_is_rejected(client):
    response = client.get(DOCUMENTS, headers=bearer("not.a.jwt"))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_expired_token_is_rejected(client):
    register(client)

    response = client.get(DOCUMENTS, headers=bearer(make_token(expires_in=timedelta(minutes=-1))))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_token_signed_with_a_different_key_is_rejected(client):
    register(client)
    forged = make_token(secret="a-completely-different-secret-key-0123456789")

    response = client.get(DOCUMENTS, headers=bearer(forged))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_valid_token_without_a_subject_is_rejected(client):
    register(client)

    response = client.get(DOCUMENTS, headers=bearer(make_token(include_sub=False)))

    assert response.status_code == 401
    assert response.json()["detail"] == "User not found"


def test_valid_token_for_an_unknown_user_is_rejected(client):
    response = client.get(DOCUMENTS, headers=bearer(make_token("ghost@example.com")))

    assert response.status_code == 401
    assert response.json()["detail"] == "User not found"


# ---------- account deletion ----------

def test_delete_account_removes_the_user_and_blocks_login(client, auth_headers, db):
    response = client.delete(ME, headers=auth_headers)

    assert response.status_code == 204
    assert db.query(User).count() == 0
    assert login(client).status_code == 401


def test_token_stops_working_after_account_deletion(client, auth_headers):
    client.delete(ME, headers=auth_headers)

    response = client.get(DOCUMENTS, headers=auth_headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "User not found"


def test_delete_account_requires_authentication(client):
    assert client.delete(ME).status_code == 401


def test_delete_account_cascades_to_the_users_documents(client, auth_headers, db):
    client.post(DOCUMENTS, json={"filename": "a.txt"}, headers=auth_headers)
    client.post(DOCUMENTS, json={"filename": "b.txt"}, headers=auth_headers)
    assert db.query(Document).count() == 2

    client.delete(ME, headers=auth_headers)

    assert db.query(Document).count() == 0


def test_delete_account_leaves_other_users_untouched(client, make_user, db):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    client.post(DOCUMENTS, json={"filename": "alice.txt"}, headers=alice)
    client.post(DOCUMENTS, json={"filename": "bob.txt"}, headers=bob)

    assert client.delete(ME, headers=alice).status_code == 204

    remaining = client.get(DOCUMENTS, headers=bob).json()
    assert [d["filename"] for d in remaining] == ["bob.txt"]
    assert db.query(User).count() == 1