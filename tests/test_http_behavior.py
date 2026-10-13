import logging
import re
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.crud import document as crud_document
from app.main import app
from app.models.document import Document

HEALTH = "/api/v1/health"
REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
DOCUMENTS = "/api/v1/documents"
UPLOAD = f"{DOCUMENTS}/upload"

PASSWORD = "password123"
ALLOWED_ORIGIN = "http://localhost:5173"
UNKNOWN_ORIGIN = "http://evil.example"


def explode(*args, **kwargs):
    raise RuntimeError("secret database password")


def credentials(i, password=PASSWORD):
    return {"email": f"user{i}@example.com", "password": password}


def upload_text(client, headers, i):
    return client.post(
        UPLOAD,
        files={"file": (f"n{i}.txt", b"hello", "text/plain")},
        headers=headers,
    )


@pytest.fixture
def rate_limits_on():
    limiter.reset()
    limiter.enabled = True


@pytest.fixture
def lenient_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------- request id and security headers (unchanged) ----------

def test_health_check(client):
    response = client.get(HEALTH)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_every_response_has_a_valid_request_id(client):
    response = client.get(HEALTH)

    request_id = response.headers.get("X-Request-ID")
    assert request_id
    uuid.UUID(request_id)  # raises ValueError if it is not a valid UUID


def test_request_ids_are_unique_per_request(client):
    ids = {client.get(HEALTH).headers["X-Request-ID"] for _ in range(3)}

    assert len(ids) == 3


def test_security_headers_are_set(client):
    response = client.get(HEALTH)

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.parametrize(
    "path, status",
    [("/api/v1/does-not-exist", 404), (DOCUMENTS, 401)],
    ids=["unknown-route", "missing-token"],
)
def test_handled_error_responses_still_carry_request_id_and_security_headers(client, path, status):
    response = client.get(path)

    assert response.status_code == status
    uuid.UUID(response.headers["X-Request-ID"])
    assert response.headers["X-Frame-Options"] == "DENY"


# ---------- rate limiting ----------

def test_register_is_limited_to_5_requests_per_minute(client, rate_limits_on):
    codes = [client.post(REGISTER, json=credentials(i)).status_code for i in range(6)]

    assert codes == [201, 201, 201, 201, 201, 429]


def test_failed_logins_are_counted_and_block_even_the_right_password(client, rate_limits_on):
    client.post(REGISTER, json=credentials(0))

    wrong = [
        client.post(LOGIN, json=credentials(0, "wrong-password")).status_code for _ in range(5)
    ]
    blocked = client.post(LOGIN, json=credentials(0))

    assert wrong == [401] * 5
    assert blocked.status_code == 429
    assert "Rate limit exceeded" in blocked.json()["error"]
    assert "5 per 1 minute" in blocked.json()["error"]


def test_each_limited_route_has_its_own_counter(client, rate_limits_on):
    for i in range(5):
        assert client.post(REGISTER, json=credentials(i)).status_code == 201
    assert client.post(REGISTER, json=credentials(5)).status_code == 429

    assert client.post(LOGIN, json=credentials(0)).status_code == 200


def test_invalid_requests_do_not_consume_the_limit(client, rate_limits_on):
    invalid = {"email": "not-an-email", "password": PASSWORD}
    for _ in range(7):
        assert client.post(REGISTER, json=invalid).status_code == 422

    assert client.post(REGISTER, json=credentials(0)).status_code == 201


def test_upload_is_limited_to_5_requests_per_minute(
    client, auth_headers, rate_limits_on, tmp_path, db
):
    codes = [upload_text(client, auth_headers, i).status_code for i in range(6)]

    assert codes == [201, 201, 201, 201, 201, 429]
    assert len([p for p in tmp_path.iterdir() if p.is_file()]) == 5
    assert db.query(Document).count() == 5


def test_other_routes_are_not_rate_limited(client, auth_headers, rate_limits_on):
    codes = {client.get(DOCUMENTS, headers=auth_headers).status_code for _ in range(20)}

    assert codes == {200}


def test_the_limit_is_per_client_address_not_per_account(client, rate_limits_on):
    # Everyone behind one IP (an office or university network) shares this counter.
    client.post(REGISTER, json=credentials(0))
    client.post(REGISTER, json=credentials(1))
    for _ in range(5):
        client.post(LOGIN, json=credentials(0, "wrong-password"))

    other_account = client.post(LOGIN, json=credentials(1))

    assert other_account.status_code == 429


# ---------- the global 500 handler ----------

def test_unhandled_errors_return_a_generic_500_without_internal_details(
    lenient_client, auth_headers, monkeypatch
):
    monkeypatch.setattr(crud_document, "get_documents", explode)

    response = lenient_client.get(DOCUMENTS, headers=auth_headers)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    uuid.UUID(response.headers["X-Request-ID"])
    for leaked in ["secret", "RuntimeError", "Traceback"]:
        assert leaked not in response.text


def test_unhandled_errors_are_logged_with_request_details(
    lenient_client, auth_headers, monkeypatch, caplog
):
    monkeypatch.setattr(crud_document, "get_documents", explode)

    with caplog.at_level(logging.ERROR, logger="documind"):
        lenient_client.get(DOCUMENTS, headers=auth_headers)

    records = [r for r in caplog.records if "Unhandled error" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "path=/api/v1/documents" in message
    assert "method=GET" in message
    assert "secret database password" in message
    assert re.search(r"request_id=[0-9a-f-]{36}", message)
    assert records[0].exc_info is not None


def test_the_app_keeps_serving_after_an_unhandled_error(
    lenient_client, auth_headers, monkeypatch
):
    monkeypatch.setattr(crud_document, "get_documents", explode)
    assert lenient_client.get(DOCUMENTS, headers=auth_headers).status_code == 500

    assert lenient_client.get(HEALTH).status_code == 200


# ---------- CORS ----------

def test_allowed_origin_gets_cors_headers(client):
    response = client.get(HEALTH, headers={"Origin": ALLOWED_ORIGIN})

    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_unknown_origin_gets_no_cors_headers(client):
    response = client.get(HEALTH, headers={"Origin": UNKNOWN_ORIGIN})

    assert "access-control-allow-origin" not in response.headers


def test_requests_without_an_origin_get_no_cors_headers(client):
    response = client.get(HEALTH)

    assert "access-control-allow-origin" not in response.headers


def test_preflight_from_the_allowed_origin_is_accepted(client):
    response = client.options(
        DOCUMENTS,
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_preflight_from_an_unknown_origin_is_rejected(client):
    response = client.options(
        DOCUMENTS,
        headers={"Origin": UNKNOWN_ORIGIN, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_api_error_responses_still_carry_cors_headers(client):
    response = client.get(DOCUMENTS, headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
