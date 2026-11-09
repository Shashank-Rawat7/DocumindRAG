# Indexing backfill runbook

> **STOP: do not run `--execute` against `documind_db`.**
> The real `--execute` run must not be performed until CP6c is formally complete **and** the run
> has been separately approved. Dry runs are safe at any time.
>
> Approval record: approved by ________  date ________  canary (`--limit 1`) result ________

## 1. What the backfill does

The backfill indexes documents whose text extraction is complete but whose index is missing or
out of date. For each document it queues indexing, builds chunks and embeddings, and publishes
them in one transaction.

It writes only `document_chunks` rows and the indexing columns of `documents`
(`indexing_status`, `indexing_error`, `indexing_attempt`, `indexing_lease_expires_at`,
`indexed_*`, `source_sha256`). It never changes `extracted_text`, uploaded files or user data.

## 2. Before you start

### Prerequisites

- Windows PowerShell, working directory **exactly the project root** (`E:\Projects\Documind`).
- The Python virtual environment: `venv\Scripts\python.exe`.
- PostgreSQL with pgvector running and migrated: `docker compose up -d` (port 127.0.0.1:5434).
- `.env` present in the project root. Never paste its contents anywhere.
- The embedding model (`intfloat/multilingual-e5-small`, pinned revision) in the local Hugging Face
  cache. The backfill never downloads anything.
- A backup of `documind_db` and of the `uploads` folder before any `--execute`.
  For example: `docker compose exec db pg_dump -U documind_user -d documind_db -Fc -f /tmp/before_backfill.dump`
  and then `docker compose cp db:/tmp/before_backfill.dump .`
- Recommended: stop the API server, so no upload hook is indexing while the backfill runs.

### Environment restrictions (checked automatically; a failure exits with 2 and changes nothing)

| Check | Requirement |
|---|---|
| Current directory | the repository root |
| `.env` | exists in the repository root (presence only, never read by the check) |
| Upload directory | exists, and equals `<repo root>/<UPLOAD_DIR>` |
| `ENVIRONMENT` | explicitly set to `development` (the default value does not count) |
| Database URL | PostgreSQL, loopback host (`localhost`, `127.0.0.1`), no query parameters |
| libpq variables | `PGHOST`, `PGHOSTADDR`, `PGSERVICE` must not be set |
| `--database NAME` | must equal the URL's database name, and then the live `current_database()` |

## 3. Dry run (the default; read-only)

A dry run never loads the model and never writes. It uses a read-only database connection.

```powershell
# Usage and options
venv\Scripts\python.exe scripts\backfill_indexing.py --help

# Plan only, no live database-name check
venv\Scripts\python.exe scripts\backfill_indexing.py

# Plan with the live database-name check (recommended)
venv\Scripts\python.exe scripts\backfill_indexing.py --database documind_db

# Include indexing failures; limit the plan to the first 5 candidates
venv\Scripts\python.exe scripts\backfill_indexing.py --database documind_db --include-failed
venv\Scripts\python.exe scripts\backfill_indexing.py --database documind_db --limit 5
```

Exit code 0 means the report was printed.

## 4. Reading the plan

Every completed document gets one category and one action. Documents whose extraction is not
completed appear only in the "Not examined" row.

| Action | Meaning |
|---|---|
| **INDEX** | `--execute` will index it. |
| **SKIP** | Left alone, only counted. |
| **REPORT** | Left alone and listed for a human. |
| **BLOCKED** | Would be indexed, but cannot be yet. Listed, never attempted. |

| Category | Meaning | Action |
|---|---|---|
| `no_processing_status`, `extraction_incomplete` | Extraction never started, or is pending, processing or failed | SKIP |
| `never_queued` | Completed and never queued for indexing | INDEX |
| `pending` | Queued, not yet claimed | INDEX |
| `abandoned_lease` | A worker claimed it and its lease expired | INDEX |
| `live_lease` | Another worker holds it right now | SKIP |
| `failed` | The last indexing attempt failed | REPORT `[requires --include-failed]`; INDEX with the flag |
| `indexed_stale` | Indexed, but the source version, chunks or embedding id is off (reasons are shown in brackets) | INDEX |
| `indexed_with_error` | A valid index whose last re-index attempt failed | REPORT `[requires --include-failed]`; INDEX with the flag |
| `current` | Source version, chunks and embedding id all match | SKIP |
| `no_text` | Would be indexed but has no extracted text | REPORT `[no extracted text]` |
| `unrecognized`, `lease_ambiguous` | A status value the schema does not allow, or `indexing` without a lease | REPORT |
| any INDEX category with an unusable PDF | The stored PDF file is missing, or its stored name is missing or invalid | BLOCKED `[pdf_file_missing]`, `[pdf_no_stored_name]` or `[pdf_unchecked]` |

- **BLOCKED:** a human fixes the cause (restore the file, correct the stored name). The backfill
  fixes nothing itself. Blocked documents make an execution exit with 1.
- **REPORT:** read the note. Nothing is changed for these documents.

## 5. Options

**`--include-failed`**

- Off by default. A `failed` or `indexed_with_error` document is then REPORT only.
- On, those two categories become INDEX.
- It covers indexing failures after a successful extraction only. A document whose extraction
  failed is never eligible, with or without the flag.

**`--limit N`**

- N must be a whole number of at least 1.
- The first N INDEX candidates by ascending document id are kept. The remaining INDEX candidates
  appear under "Deferred by --limit" and are not processed.
- SKIP, REPORT and BLOCKED documents never count against the limit and are never deferred.
- Every run re-plans from scratch, so a later run picks up the deferred documents.

## 6. Execution (when approved)

```powershell
# Canary: one document first
venv\Scripts\python.exe scripts\backfill_indexing.py --database documind_db --limit 1 --execute
```

`--execute` requires `--database`. Review a dry run first.

### Safety gates, in order

1. **Static checks** (section 2). All failures are reported at once. No database connection exists yet.
2. **Read-only connection and live check.** `current_database()` must equal `--database`.
3. **Factory check.** The session factories that will write must be bound to that same engine.
4. **Plan printed.** It stays visible even if a later step refuses.
5. **Model preflight**, only if something is to be indexed:
   - it sets `HF_HUB_OFFLINE=1`;
   - it checks the embedder's id against the id used for the plan;
   - it loads the model from the local cache;
   - it makes one real embedding and checks its shape.
   A failure exits with 2 and **nothing is changed**.
6. **Per document**, immediately before queueing, the document is re-read and reclassified. A plan
   that went stale cannot cause a wrong write. Then it is queued and run.

No database connection is held while the model embeds. The same embedder object that passed the
preflight indexes every document.

## 7. Expected output and exit codes

Illustrative dry run:

```text
DocuMind indexing backfill - DRY RUN (nothing will be changed)
Target   : localhost:5434/documind_db as documind_user
Database : verified as 'documind_db' (live check)
Options  : include-failed=no, limit=none

Summary
  Completed documents examined            : 12
  Will index                              : 3
  Deferred by --limit                     : 0
  Blocked                                 : 1
  Reported (needs attention)              : 1
  Skipped (nothing to do)                 : 7
  Not examined (extraction not completed) : 0
  By category                             : never_queued 3, pending 1, failed 1, current 7

Will index (3)
  doc 4      never_queued
  doc 5      never_queued
  doc 9      pending

Deferred by --limit (0)
  none

Blocked (1)
  doc 6      never_queued  [pdf_file_missing]

Reported (1)
  doc 8      failed  [requires --include-failed]

Dry run: nothing was changed.
```

An execution prints the same plan with the mode line `EXECUTE (documents will be indexed)`, then:

- the notices `Checking the embedding model (offline); this can take a while.` and
  `Embedding model check passed.` on the error stream;
- `Indexing starts now. Progress follows, one line per document.`;
- one progress line per document, for example `[1/3] doc 4      indexed (14 chunks, 6.3s)`;
- the **Execution summary** and a verdict line.

Progress outcomes:

| Outcome | Meaning | Counts as failure |
|---|---|---|
| `indexed` | Published | no |
| `failed (REASON)` | The attempt failed and the reason was recorded | **yes** |
| `failure not recorded (REASON)` | The attempt failed and recording it failed too | **yes** |
| `error (ClassName)` | An exception, for example a database error. Only the class name is shown | **yes** |
| `blocked (...)` | Became BLOCKED just before indexing | counts as blocked |
| `changed (...)`, `not queued`, `not claimed`, `stale`, `gone` | Another worker or process changed or took the document. Skipped | no |
| `interrupted`, `not attempted` | Ctrl-C | see exit 130 |

`REASON` is one of: `NO_INDEXABLE_TEXT`, `BUILD_REJECTED`, `CHUNKING`, `OFFSET_MAPPING`,
`CHUNK_TOO_LONG`, `EMBEDDING_INVALID`, `PAGE_MAPPING`, `EMBEDDER_UNAVAILABLE`, `DATABASE`,
`UNEXPECTED`. The stored `documents.indexing_error` holds the matching fixed sentence.

| Exit | Meaning |
|---|---|
| **0** | A dry run completed, or an execution finished with no failures and no blocked documents |
| **1** | An execution finished with failures or blocked documents, or an operational error (for example an unexpected error, or no connection without `--database`) |
| **2** | A refusal: safety check, invalid arguments, configuration that cannot load, live check, factory check, or failed model preflight. Nothing was changed |
| **130** | Interrupted |

Messages never contain passwords, URLs, filenames, stored names or exception text.

## 8. Failure, interruption and lease recovery

The default lease is `INDEXING_LEASE_MINUTES=15`. Builds are not heartbeated, so a lease must outlast the
slowest build.

| Situation | State left behind | Recovery |
|---|---|---|
| `failed (REASON)` | `indexing_status='failed'`, fixed reason stored | Fix the cause, then rerun with `--include-failed` |
| `failure not recorded` | Still `indexing` until the lease expires | Wait for expiry; it then shows as `abandoned_lease` and is INDEX |
| `error (ClassName)` after queueing | `pending` | A later run picks it up |
| Model preflight failed (exit 2) | Nothing changed | Fix the cache, id or environment; rerun |
| Ctrl-C (exit 130) | The document in progress stays `indexing` until its lease expires. Documents not yet reached are untouched | Run a dry run. The locked document shows as `live_lease` (SKIP) until expiry, then `abandoned_lease` (INDEX) |

A summary containing `Interrupted in progress` also prints a note that the document stays locked.
After an interruption, always run a dry run to see the current state.

Do not hand-edit `indexing_status` or the lease columns. If a document stays stuck well past
the lease time, stop and investigate. Indexing data is derived data; the source of truth is
`extracted_text` and the uploaded file, and the backup covers the rest.

## 9. Verifying the database

Run these after an execution. They are all read-only. Example:

```powershell
docker compose exec db psql -U documind_user -d documind_db -c "SELECT indexing_status, count(*) FROM documents GROUP BY indexing_status ORDER BY indexing_status;"
```

Queries marked `none` must return no rows after a clean full run. After a partial run, or with
REPORT or BLOCKED documents, some rows are expected and explain themselves. Queries marked `info`
are for inspection.

```sql
-- verify: info
SELECT indexing_status, count(*) FROM documents GROUP BY indexing_status ORDER BY indexing_status;
```

```sql
-- verify: none
SELECT d.id, d.indexing_status, d.indexing_error
FROM documents d
WHERE d.processing_status = 'completed'
  AND d.extracted_text IS NOT NULL AND length(trim(d.extracted_text)) > 0
  AND (d.indexing_status IS DISTINCT FROM 'indexed'
       OR NOT EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id))
ORDER BY d.id;
```

```sql
-- verify: none
SELECT id, indexing_status, indexing_attempt, indexing_lease_expires_at
FROM documents WHERE indexing_status = 'indexing' ORDER BY id;
```

```sql
-- verify: none
SELECT id, indexing_status, indexing_error, indexing_attempt
FROM documents WHERE indexing_error IS NOT NULL ORDER BY id;
```

```sql
-- verify: none
SELECT c.id, c.document_id, c.chunk_index
FROM document_chunks c JOIN documents d ON d.id = c.document_id
WHERE c.text <> substr(d.extracted_text, c.start_offset + 1, c.end_offset - c.start_offset);
```

```sql
-- verify: none
SELECT id FROM documents
WHERE indexing_status = 'indexed'
  AND source_sha256 IS DISTINCT FROM encode(sha256(convert_to(extracted_text, 'UTF8')), 'hex');
```

```sql
-- verify: info
SELECT indexed_embedding_id, count(*) FROM documents
WHERE indexing_status = 'indexed' GROUP BY indexed_embedding_id;
```

All rows must show the configured id, `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`.

```sql
-- verify: none
SELECT id, document_id, chunk_index, vector_dims(embedding)
FROM document_chunks WHERE vector_dims(embedding) <> 384;
```

```sql
-- verify: info
SELECT d.id, count(*) AS chunks, count(*) FILTER (WHERE c.page_start IS NULL) AS chunks_without_pages
FROM documents d JOIN document_chunks c ON c.document_id = d.id
WHERE right(lower(d.stored_name), 4) = '.pdf'
GROUP BY d.id
HAVING count(*) FILTER (WHERE c.page_start IS NULL) > 0
ORDER BY d.id;
```

Rows here are PDFs indexed **without page numbers** (for example the PDF could not be re-read, or its text
no longer matches). The CLI does not report this, and a rerun will not repair it.

```sql
-- verify: info
SELECT d.id, count(c.id) AS chunks, min(c.page_start) AS first_page, max(c.page_end) AS last_page
FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id
WHERE d.indexing_status = 'indexed' GROUP BY d.id ORDER BY d.id;
```

Finally, run a dry run again. After a clean full run it must show `Will index : 0`.

## 10. Known limitations

- The model is checked only inside `--execute` (after the plan prints). There is no model check without
  executing, so the first execute should be the `--limit 1` canary.
- A change in chunking or signature rules is not detected as "stale" (V1). Only the source
  version, missing chunks and the embedding id are.
- PDF page-mapping rules are unversioned, and the file check is existence only.
- Ctrl-C leaves the document in progress locked until its lease expires.
- If a document was queued and its session then failed to close, it is reported as an error and
  stays `pending`.
- Library warnings (for example the indexing runner's) go to the real stderr, next to the progress lines.