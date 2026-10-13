import uuid


def test_health_check(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_every_response_has_a_valid_request_id(client):
    response = client.get("/api/v1/health")

    request_id = response.headers.get("X-Request-ID")
    assert request_id
    uuid.UUID(request_id)  # raises ValueError if it is not a valid UUID


def test_request_ids_are_unique_per_request(client):
    ids = {client.get("/api/v1/health").headers["X-Request-ID"] for _ in range(3)}

    assert len(ids) == 3


def test_security_headers_are_set(client):
    response = client.get("/api/v1/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"