# DocuMind

## Running the tests

The suite uses real PostgreSQL (not SQLite or mocks), because the rules we care
about live in the database: cascades, CHECK constraints, unique emails.

**One-time setup**

```powershell
pip install -r requirements-dev.txt
docker compose up -d
docker compose exec db psql -U documind_user -d documind_db -c "CREATE DATABASE documind_test;"
```

**Run**

```powershell
pytest                       # everything
pytest -v                    # one line per test
pytest tests/test_auth.py    # one file
pytest -k upload             # tests whose name contains "upload"
pytest -x --lf               # stop at first failure / rerun last failures
```

**How it works**

- `tests/conftest.py` points the app at `documind_test` *before* the app is
  imported. It refuses to run unless the database name ends in `_test`, so the
  dev database can never be touched. Set `TEST_DATABASE_URL` to use a different
  server.
- The schema is built once per run by running `alembic upgrade head`, so the
  tests also verify the migrations.
- Tables are truncated after every test. Uploads go to a temporary folder per
  test. Rate limiting is off by default and switched on only in the tests that
  check it.
- Background tasks run inside the test client, so after an upload returns,
  processing has already finished.
- `tests/fixtures/sample.pdf` is a small real PDF used for text-extraction tests.

**What is covered**

| File | Covers |
|---|---|
| `test_auth.py` | registration, login, JWT handling, account deletion and cascades |
| `test_documents.py` | CRUD, pagination, ownership isolation, auth on every route |
| `test_uploads.py` | upload validation, size limit, filename safety, download, file cleanup |
| `test_processing.py` | text extraction per file type, task status transitions, failure handling |
| `test_http_behavior.py` | rate limits, error handler, CORS, request-ID and security headers |
| `test_environment.py` | test database wiring, migrated schema, test isolation |

## Known limitations

**File storage (local disk)**
- File type is checked by extension only; content is not sniffed.
- A crash between the database commit and file deletion can leave an orphaned
  file on disk; there is no cleanup job yet.
- Oversized request bodies are buffered before the size check runs; rejecting
  them early is a reverse-proxy job (e.g. nginx `client_max_body_size`).
- Files live on one machine's disk; moving to object storage means replacing
  `app/core/storage.py`.

**Background processing (FastAPI `BackgroundTasks`)**
- Not durable: a server restart can leave a document stuck in `pending` or
  `processing`. A startup sweep or a real queue (Celery/RQ) is the fix.
- No retries: `failed` is final. Bursts of large files share the request thread
  pool.
- No OCR: scanned PDFs fail with a clear message.
- DOCX extraction covers body paragraphs and tables only (not headers, footers,
  footnotes or text boxes).

**HTTP behavior**
- Rate limits are per client IP, so users behind one shared address share the
  counter. Limited responses carry no `Retry-After` header.
- Unhandled (500) errors are produced outside the middleware stack, so those
  responses lack the security and CORS headers. A browser frontend may show them
  as CORS errors during development.

## HTTP Behavior and Error Handling

DocuMind includes request IDs, security headers, request logging, rate limiting, CORS configuration, and centralized handling of unhandled exceptions.

### Error Responses

- **Handled API errors:** Tested responses include request IDs and security headers.
- **Unhandled exceptions:** Return a generic `500 Internal Server Error` response without exposing internal exception details. The response includes the `X-Request-ID` header for troubleshooting.
- **Logging:** Unhandled exceptions are logged with request details and exception information.

### CORS and Middleware Limitation

The current middleware configuration has a known limitation: unhandled `500` responses may not include CORS or security headers because exceptions can propagate beyond the inner middleware before the response is generated.

Handled API error responses have been tested to include CORS headers. This limitation is documented for awareness; the middleware architecture has not been refactored to address it.
