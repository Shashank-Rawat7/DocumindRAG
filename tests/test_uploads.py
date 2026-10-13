import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import storage
from app.core.config import settings
from app.crud import document as crud_document
from app.main import app
from app.models.document import Document

DOCUMENTS = "/api/v1/documents"
UPLOAD = f"{DOCUMENTS}/upload"
ME = "/api/v1/auth/me"

ONE_MB = 1024 * 1024
STORED_NAME = re.compile(r"^[0-9a-f]{32}\.[a-z]+$")
RESPONSE_FIELDS = {
    "id", "filename", "description", "content_type",
    "size_bytes", "processing_status", "processing_error", "created_at",
}


def upload(
    client,
    headers,
    name="notes.txt",
    content=b"hello world",
    content_type="text/plain",
    description=None,
):
    data = {"description": description} if description is not None else None
    return client.post(
        UPLOAD,
        files={"file": (name, content, content_type)},
        data=data,
        headers=headers,
    )


def stored_files(tmp_path):
    return sorted(p for p in tmp_path.iterdir() if p.is_file())


@pytest.fixture
def one_mb_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)


# ---------- happy path ----------

def test_upload_creates_a_document_and_stores_the_file(client, auth_headers, tmp_path):
    response = upload(client, auth_headers, "notes.txt", b"hello world", "text/plain", "my notes")

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body["id"], int)
    assert body["filename"] == "notes.txt"
    assert body["description"] == "my notes"
    assert body["content_type"] == "text/plain"
    assert body["size_bytes"] == len(b"hello world")

    files = stored_files(tmp_path)
    assert len(files) == 1
    assert STORED_NAME.match(files[0].name)
    assert files[0].read_bytes() == b"hello world"


def test_response_exposes_no_storage_details(client, auth_headers):
    body = upload(client, auth_headers).json()

    assert set(body) == RESPONSE_FIELDS


def test_response_shows_pending_and_processing_finishes_afterwards(client, auth_headers, db):
    text = "héllo wörld — 日本語"
    response = upload(client, auth_headers, content=text.encode("utf-8"))
    assert response.json()["processing_status"] == "pending"
    doc_id = response.json()["id"]

    fetched = client.get(f"{DOCUMENTS}/{doc_id}", headers=auth_headers).json()

    assert fetched["processing_status"] == "completed"
    assert fetched["processing_error"] is None
    assert db.get(Document, doc_id).extracted_text == text


@pytest.mark.parametrize(
    "raw, expected",
    [("  hello  ", "hello"), ("   ", None), ("", None)],
    ids=["trimmed", "blank-becomes-null", "empty-becomes-null"],
)
def test_upload_cleans_the_description(client, auth_headers, raw, expected):
    response = upload(client, auth_headers, description=raw)

    assert response.status_code == 201
    assert response.json()["description"] == expected


def test_upload_accepts_a_description_of_exactly_1000_characters(client, auth_headers):
    assert upload(client, auth_headers, description="d" * 1000).status_code == 201


def test_upload_rejects_a_too_long_description_without_storing_anything(
    client, auth_headers, tmp_path, db
):
    response = upload(client, auth_headers, description="d" * 1001)

    assert response.status_code == 422
    assert stored_files(tmp_path) == []
    assert db.query(Document).count() == 0


def test_upload_requires_the_file_field(client, auth_headers):
    response = client.post(UPLOAD, data={"description": "no file"}, headers=auth_headers)

    assert response.status_code == 422


# ---------- file types ----------

@pytest.mark.parametrize("name", ["a.txt", "a.md", "a.pdf", "a.docx", "UPPER.TXT", "mixed.Md"])
def test_upload_accepts_allowed_extensions_case_insensitively(client, auth_headers, tmp_path, name):
    response = upload(client, auth_headers, name, b"some content")

    assert response.status_code == 201
    assert response.json()["filename"] == name
    stored = stored_files(tmp_path)
    assert len(stored) == 1
    assert stored[0].suffix == Path(name).suffix.lower()


@pytest.mark.parametrize(
    "name",
    ["malware.exe", "image.png", "page.html", "script.py", "archive.zip", "README", ".txt", "notes.txt.exe"],
    ids=["exe", "png", "html", "py", "zip", "no-extension", "dotfile-only", "double-extension"],
)
def test_upload_rejects_unsupported_types_without_storing_anything(
    client, auth_headers, tmp_path, db, name
):
    response = upload(client, auth_headers, name)

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type"
    assert stored_files(tmp_path) == []
    assert db.query(Document).count() == 0


# ---------- filename handling ----------

@pytest.mark.parametrize(
    "sent, shown",
    [
        ("../../evil.txt", "evil.txt"),
        ("C:\\Users\\me\\resume.txt", "resume.txt"),
        ("/etc/passwd.txt", "passwd.txt"),
    ],
    ids=["parent-traversal", "windows-path", "absolute-path"],
)
def test_upload_sanitizes_the_display_name_and_never_uses_it_on_disk(
    client, auth_headers, tmp_path, sent, shown
):
    response = upload(client, auth_headers, sent)

    assert response.status_code == 201
    assert response.json()["filename"] == shown
    stored = stored_files(tmp_path)
    assert len(stored) == 1
    assert STORED_NAME.match(stored[0].name)


def test_upload_truncates_a_very_long_display_name_to_255_characters(client, auth_headers):
    response = upload(client, auth_headers, "a" * 300 + ".txt")

    assert response.status_code == 201
    assert len(response.json()["filename"]) == 255


# ---------- size limit ----------

def test_upload_at_exactly_the_size_limit_is_accepted(client, auth_headers, one_mb_limit):
    response = upload(client, auth_headers, content=b"a" * ONE_MB)

    assert response.status_code == 201
    assert response.json()["size_bytes"] == ONE_MB


def test_upload_over_the_size_limit_is_rejected_and_leaves_nothing_behind(
    client, auth_headers, one_mb_limit, tmp_path, db
):
    response = upload(client, auth_headers, content=b"a" * (ONE_MB + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"
    assert stored_files(tmp_path) == []
    assert db.query(Document).count() == 0


def test_empty_file_is_accepted_but_processing_marks_it_failed(client, auth_headers):
    response = upload(client, auth_headers, content=b"")

    assert response.status_code == 201
    assert response.json()["size_bytes"] == 0
    fetched = client.get(f"{DOCUMENTS}/{response.json()['id']}", headers=auth_headers).json()
    assert fetched["processing_status"] == "failed"
    assert "No extractable text" in fetched["processing_error"]


# ---------- content type ----------

def test_claimed_content_type_is_recorded_but_download_ignores_it(client, auth_headers):
    uploaded = upload(client, auth_headers, "notes.txt", b"hello", "application/x-evil").json()
    assert uploaded["content_type"] == "application/x-evil"

    response = client.get(f"{DOCUMENTS}/{uploaded['id']}/download", headers=auth_headers)

    assert response.headers["content-type"].startswith("text/plain")


# ---------- download ----------

def test_download_returns_the_exact_bytes_as_an_attachment(client, auth_headers):
    content = "héllo wörld — 日本語".encode("utf-8")
    doc = upload(client, auth_headers, "notes.txt", content).json()

    response = client.get(f"{DOCUMENTS}/{doc['id']}/download", headers=auth_headers)

    assert response.status_code == 200
    assert response.content == content
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert 'filename="notes.txt"' in disposition


def test_download_uses_the_current_display_name_after_a_rename(client, auth_headers):
    doc = upload(client, auth_headers, "old.txt", b"data").json()
    url = f"{DOCUMENTS}/{doc['id']}"
    client.patch(url, json={"filename": "renamed.txt"}, headers=auth_headers)

    response = client.get(f"{url}/download", headers=auth_headers)

    assert response.content == b"data"
    assert 'filename="renamed.txt"' in response.headers["content-disposition"]


def test_download_of_a_metadata_only_document_returns_404(client, auth_headers):
    doc = client.post(DOCUMENTS, json={"filename": "no-file.txt"}, headers=auth_headers).json()

    response = client.get(f"{DOCUMENTS}/{doc['id']}/download", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_download_of_a_nonexistent_document_returns_404(client, auth_headers):
    response = client.get(f"{DOCUMENTS}/999999/download", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_download_when_the_file_is_missing_from_disk_returns_404(client, auth_headers, tmp_path):
    doc = upload(client, auth_headers).json()
    for file in stored_files(tmp_path):
        file.unlink()

    response = client.get(f"{DOCUMENTS}/{doc['id']}/download", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_other_user_cannot_download_a_file(client, make_user):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = upload(client, alice, content=b"secret").json()

    foreign = client.get(f"{DOCUMENTS}/{doc['id']}/download", headers=bob)
    missing = client.get(f"{DOCUMENTS}/999999/download", headers=bob)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    assert b"secret" not in foreign.content


def test_uploaded_document_appears_in_the_owners_list_without_internal_fields(client, auth_headers):
    doc = upload(client, auth_headers, "notes.txt", b"hello").json()

    listed = client.get(DOCUMENTS, headers=auth_headers).json()

    assert len(listed) == 1
    assert set(listed[0]) == RESPONSE_FIELDS
    assert listed[0]["id"] == doc["id"]
    assert listed[0]["size_bytes"] == 5


def test_same_filename_twice_creates_two_independent_files(client, auth_headers, tmp_path):
    first = upload(client, auth_headers, "same.txt", b"first").json()
    second = upload(client, auth_headers, "same.txt", b"second").json()

    assert len(stored_files(tmp_path)) == 2
    first_download = client.get(f"{DOCUMENTS}/{first['id']}/download", headers=auth_headers)
    second_download = client.get(f"{DOCUMENTS}/{second['id']}/download", headers=auth_headers)
    assert first_download.content == b"first"
    assert second_download.content == b"second"


# ---------- cleanup on delete ----------

def test_deleting_a_document_removes_its_file(client, auth_headers, tmp_path):
    doc = upload(client, auth_headers).json()
    assert len(stored_files(tmp_path)) == 1

    response = client.delete(f"{DOCUMENTS}/{doc['id']}", headers=auth_headers)

    assert response.status_code == 204
    assert stored_files(tmp_path) == []


def test_deleting_one_document_keeps_the_other_documents_file(client, auth_headers, tmp_path):
    first = upload(client, auth_headers, "first.txt", b"one").json()
    second = upload(client, auth_headers, "second.txt", b"two").json()

    client.delete(f"{DOCUMENTS}/{first['id']}", headers=auth_headers)

    assert len(stored_files(tmp_path)) == 1
    response = client.get(f"{DOCUMENTS}/{second['id']}/download", headers=auth_headers)
    assert response.content == b"two"


def test_other_user_cannot_delete_a_document_and_its_file_survives(client, make_user, tmp_path):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    doc = upload(client, alice).json()

    response = client.delete(f"{DOCUMENTS}/{doc['id']}", headers=bob)

    assert response.status_code == 404
    assert len(stored_files(tmp_path)) == 1


def test_deleting_a_document_whose_file_is_already_gone_still_succeeds(
    client, auth_headers, tmp_path
):
    doc = upload(client, auth_headers).json()
    for file in stored_files(tmp_path):
        file.unlink()

    response = client.delete(f"{DOCUMENTS}/{doc['id']}", headers=auth_headers)

    assert response.status_code == 204


def test_deleting_the_account_removes_all_of_the_users_files_and_only_theirs(
    client, make_user, tmp_path
):
    alice = make_user("alice@example.com")
    bob = make_user("bob@example.com")
    upload(client, alice, "a1.txt", b"a1")
    upload(client, alice, "a2.txt", b"a2")
    client.post(DOCUMENTS, json={"filename": "no-file.txt"}, headers=alice)
    bob_doc = upload(client, bob, "b1.txt", b"b1").json()
    assert len(stored_files(tmp_path)) == 3

    assert client.delete(ME, headers=alice).status_code == 204

    assert len(stored_files(tmp_path)) == 1
    response = client.get(f"{DOCUMENTS}/{bob_doc['id']}/download", headers=bob)
    assert response.content == b"b1"


# ---------- failure handling ----------

def test_file_is_removed_when_saving_the_document_row_fails(
    auth_headers, tmp_path, db, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("database exploded")

    monkeypatch.setattr(crud_document, "create_uploaded_document", explode)

    with TestClient(app, raise_server_exceptions=False) as lenient_client:
        response = upload(lenient_client, auth_headers)

    assert response.status_code == 500
    assert stored_files(tmp_path) == []
    assert db.query(Document).count() == 0


# ---------- storage module ----------

@pytest.mark.parametrize(
    "name",
    ["../evil.txt", "sub/evil.txt", "/etc/passwd"],
    ids=["parent-traversal", "subfolder", "absolute-path"],
)
def test_get_file_path_rejects_names_that_escape_the_upload_folder(name):
    with pytest.raises(ValueError):
        storage.get_file_path(name)


def test_get_file_path_accepts_a_plain_stored_name(tmp_path):
    assert storage.get_file_path("abc123.txt") == tmp_path.resolve() / "abc123.txt"


def test_delete_file_ignores_none_and_missing_files():
    storage.delete_file(None)
    storage.delete_file("does-not-exist.txt")


def test_delete_file_never_deletes_outside_the_upload_folder():
    victim = storage.UPLOAD_ROOT.parent / "victim.txt"
    victim.write_text("do not delete")
    try:
        with pytest.raises(ValueError):
            storage.delete_file("../victim.txt")
        assert victim.exists()
    finally:
        victim.unlink(missing_ok=True)