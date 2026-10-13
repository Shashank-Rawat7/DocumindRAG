from datetime import datetime

import pytest

from app.models.document import Document
from app.models.user import User

DOCUMENTS = "/api/v1/documents"

RESPONSE_FIELDS = {
    "id", "filename", "description", "content_type",
    "size_bytes", "processing_status", "processing_error", "created_at",
}


def create_doc(client, headers, filename="notes.txt", description=None):
    payload = {"filename": filename}
    if description is not None:
        payload["description"] = description
    response = client.post(DOCUMENTS, json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def user_id(db, email):
    return db.query(User).filter(User.email == email).one().id


# ---------- create (JSON, metadata only) ----------

def test_create_document_returns_a_metadata_only_document(client, auth_headers):
    response = client.post(
        DOCUMENTS,
        json={"filename": "notes.txt", "description": "my notes"},
        headers=auth_headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body["id"], int)
    assert body["filename"] == "notes.txt"
    assert body["description"] == "my notes"
    assert body["content_type"] is None
    assert body["size_bytes"] is None
    assert body["processing_status"] is None
    assert body["processing_error"] is None
    created_at = datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))
    assert created_at.tzinfo is not None


def test_response_exposes_only_public_fields(client, auth_headers):
    body = create_doc(client, auth_headers)

    assert set(body) == RESPONSE_FIELDS


def test_create_document_trims_whitespace(client, auth_headers):
    body = create_doc(client, auth_headers, "  notes.txt  ", "  hello  ")

    assert body["filename"] == "notes.txt"
    assert body["description"] == "hello"


@pytest.mark.parametrize(
    "payload",
    [
        {"filename": ""},
        {"filename": "   "},
        {"filename": "a" * 256},
        {},
        {"filename": "ok.txt", "description": "d" * 1001},
        {"filename": 123},
    ],
    ids=[
        "empty-filename",
        "blank-filename",
        "filename-too-long",
        "missing-filename",
        "description-too-long",
        "filename-not-a-string",
    ],
)
def test_create_document_rejects_invalid_input(client, auth_headers, payload):
    response = client.post(DOCUMENTS, json=payload, headers=auth_headers)

    assert response.status_code == 422
    assert client.get(DOCUMENTS, headers=auth_headers).json() == []


def test_create_document_accepts_maximum_lengths(client, auth_headers):
    response = client.post(
        DOCUMENTS,
        json={"filename": "a" * 255, "description": "d" * 1000},
        headers=auth_headers,
    )

    assert response.status_code == 201


def test_create_document_ignores_a_client_supplied_owner_id(client, make_user, db):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    alice_id = user_id(db, "alice@example.com")

    response = client.post(
        DOCUMENTS, json={"filename": "x.txt", "owner_id": alice_id}, headers=bob
    )

    assert response.status_code == 201
    assert client.get(DOCUMENTS, headers=alice).json() == []
    assert [d["filename"] for d in client.get(DOCUMENTS, headers=bob).json()] == ["x.txt"]


def test_created_document_can_be_fetched(client, auth_headers):
    created = create_doc(client, auth_headers, "notes.txt", "hello")

    fetched = client.get(f"{DOCUMENTS}/{created['id']}", headers=auth_headers)

    assert fetched.status_code == 200
    assert fetched.json() == created


# ---------- list and pagination ----------

def test_list_is_empty_for_a_new_user(client, auth_headers):
    response = client.get(DOCUMENTS, headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_documents_in_creation_order(client, auth_headers):
    for name in ["c.txt", "a.txt", "b.txt"]:
        create_doc(client, auth_headers, name)

    names = [d["filename"] for d in client.get(DOCUMENTS, headers=auth_headers).json()]

    assert names == ["c.txt", "a.txt", "b.txt"]


def test_list_pagination_skip_and_limit(client, auth_headers):
    for i in range(5):
        create_doc(client, auth_headers, f"doc{i}.txt")

    def names(query):
        response = client.get(f"{DOCUMENTS}?{query}", headers=auth_headers)
        assert response.status_code == 200
        return [d["filename"] for d in response.json()]

    assert names("limit=2") == ["doc0.txt", "doc1.txt"]
    assert names("skip=2&limit=2") == ["doc2.txt", "doc3.txt"]
    assert names("skip=4&limit=10") == ["doc4.txt"]
    assert names("skip=10") == []


def test_list_default_page_size_is_10(client, auth_headers):
    for i in range(12):
        create_doc(client, auth_headers, f"doc{i}.txt")

    documents = client.get(DOCUMENTS, headers=auth_headers).json()

    assert len(documents) == 10
    assert documents[0]["filename"] == "doc0.txt"


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=101", "limit=-1", "skip=-1", "limit=abc"]
)
def test_list_rejects_invalid_pagination(client, auth_headers, query):
    response = client.get(f"{DOCUMENTS}?{query}", headers=auth_headers)

    assert response.status_code == 422


@pytest.mark.parametrize("limit", [1, 100])
def test_list_accepts_limit_boundaries(client, auth_headers, limit):
    create_doc(client, auth_headers)

    response = client.get(f"{DOCUMENTS}?limit={limit}", headers=auth_headers)

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_only_contains_the_callers_documents(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    create_doc(client, alice, "alice1.txt")
    create_doc(client, alice, "alice2.txt")
    create_doc(client, bob, "bob1.txt")

    alice_names = [d["filename"] for d in client.get(DOCUMENTS, headers=alice).json()]
    bob_names = [d["filename"] for d in client.get(DOCUMENTS, headers=bob).json()]

    assert alice_names == ["alice1.txt", "alice2.txt"]
    assert bob_names == ["bob1.txt"]


# ---------- get ----------

def test_get_nonexistent_document_returns_404(client, auth_headers):
    response = client.get(f"{DOCUMENTS}/999999", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_get_with_a_non_integer_id_returns_422(client, auth_headers):
    response = client.get(f"{DOCUMENTS}/abc", headers=auth_headers)

    assert response.status_code == 422


# ---------- update ----------

def test_update_changes_only_the_provided_fields(client, auth_headers):
    doc = create_doc(client, auth_headers, "old.txt", "old description")
    url = f"{DOCUMENTS}/{doc['id']}"

    renamed = client.patch(url, json={"filename": "new.txt"}, headers=auth_headers)
    assert renamed.status_code == 200
    assert renamed.json()["filename"] == "new.txt"
    assert renamed.json()["description"] == "old description"

    described = client.patch(url, json={"description": "new description"}, headers=auth_headers)
    assert described.json()["filename"] == "new.txt"
    assert described.json()["description"] == "new description"

    assert client.get(url, headers=auth_headers).json() == described.json()


def test_update_with_an_empty_body_changes_nothing(client, auth_headers):
    doc = create_doc(client, auth_headers, "notes.txt", "keep me")

    response = client.patch(f"{DOCUMENTS}/{doc['id']}", json={}, headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == doc


@pytest.mark.parametrize(
    "payload",
    [
        {"filename": ""},
        {"filename": "   "},
        {"filename": "a" * 256},
        {"description": "d" * 1001},
    ],
    ids=["empty-filename", "blank-filename", "filename-too-long", "description-too-long"],
)
def test_update_rejects_invalid_input(client, auth_headers, payload):
    doc = create_doc(client, auth_headers, "notes.txt", "keep me")
    url = f"{DOCUMENTS}/{doc['id']}"

    response = client.patch(url, json=payload, headers=auth_headers)

    assert response.status_code == 422
    assert client.get(url, headers=auth_headers).json() == doc


def test_update_with_null_description_leaves_it_unchanged(client, auth_headers):
    # Current behavior: null means "not provided", so PATCH cannot clear a description with null.
    doc = create_doc(client, auth_headers, "notes.txt", "keep me")

    response = client.patch(
        f"{DOCUMENTS}/{doc['id']}", json={"description": None}, headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["description"] == "keep me"


def test_update_with_empty_description_clears_the_text(client, auth_headers):
    doc = create_doc(client, auth_headers, "notes.txt", "remove me")

    response = client.patch(
        f"{DOCUMENTS}/{doc['id']}", json={"description": ""}, headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["description"] == ""


def test_update_ignores_a_client_supplied_owner_id(client, make_user, db):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    bob_id = user_id(db, "bob@example.com")
    doc = create_doc(client, alice)
    url = f"{DOCUMENTS}/{doc['id']}"

    response = client.patch(url, json={"owner_id": bob_id}, headers=alice)

    assert response.status_code == 200
    assert client.get(url, headers=alice).status_code == 200
    assert client.get(url, headers=bob).status_code == 404


def test_update_nonexistent_document_returns_404(client, auth_headers):
    response = client.patch(
        f"{DOCUMENTS}/999999", json={"filename": "x.txt"}, headers=auth_headers
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


# ---------- delete ----------

def test_delete_removes_the_document(client, auth_headers, db):
    doc = create_doc(client, auth_headers)
    url = f"{DOCUMENTS}/{doc['id']}"

    response = client.delete(url, headers=auth_headers)

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(url, headers=auth_headers).status_code == 404
    assert db.query(Document).count() == 0


def test_deleting_twice_returns_404_the_second_time(client, auth_headers):
    doc = create_doc(client, auth_headers)
    url = f"{DOCUMENTS}/{doc['id']}"

    assert client.delete(url, headers=auth_headers).status_code == 204

    second = client.delete(url, headers=auth_headers)
    assert second.status_code == 404
    assert second.json()["detail"] == "Document not found"


def test_delete_only_removes_the_target_document(client, auth_headers):
    first = create_doc(client, auth_headers, "first.txt")
    create_doc(client, auth_headers, "second.txt")

    client.delete(f"{DOCUMENTS}/{first['id']}", headers=auth_headers)

    names = [d["filename"] for d in client.get(DOCUMENTS, headers=auth_headers).json()]
    assert names == ["second.txt"]


# ---------- ownership isolation ----------

def test_other_user_cannot_read_a_document(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_doc(client, alice)

    response = client.get(f"{DOCUMENTS}/{doc['id']}", headers=bob)

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_other_user_cannot_update_a_document(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_doc(client, alice, "notes.txt")
    url = f"{DOCUMENTS}/{doc['id']}"

    response = client.patch(url, json={"filename": "hacked.txt"}, headers=bob)

    assert response.status_code == 404
    assert client.get(url, headers=alice).json()["filename"] == "notes.txt"


def test_other_user_cannot_delete_a_document(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_doc(client, alice)
    url = f"{DOCUMENTS}/{doc['id']}"

    assert client.delete(url, headers=bob).status_code == 404
    assert client.get(url, headers=alice).status_code == 200


@pytest.mark.parametrize("method", ["GET", "PATCH", "DELETE"])
def test_a_foreign_document_is_indistinguishable_from_a_missing_one(client, make_user, method):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = create_doc(client, alice)
    kwargs = {"json": {"filename": "hacked.txt"}} if method == "PATCH" else {}

    foreign = client.request(method, f"{DOCUMENTS}/{doc['id']}", headers=bob, **kwargs)
    missing = client.request(method, f"{DOCUMENTS}/999999", headers=bob, **kwargs)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


# ---------- authentication on every route ----------

@pytest.mark.parametrize(
    "method, path, body",
    [
        ("POST", "/documents", {"filename": "a.txt"}),
        ("GET", "/documents", None),
        ("GET", "/documents/1", None),
        ("PATCH", "/documents/1", {"filename": "b.txt"}),
        ("DELETE", "/documents/1", None),
        ("GET", "/documents/1/download", None),
    ],
)
def test_document_routes_require_authentication(client, method, path, body):
    response = client.request(method, f"/api/v1{path}", json=body)

    assert response.status_code == 401


def test_upload_requires_authentication(client):
    response = client.post(f"{DOCUMENTS}/upload", files={"file": ("a.txt", b"hello")})

    assert response.status_code == 401