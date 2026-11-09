"""Wiring between the backfill engine and the real indexing code (C5.5, CP6c.1).

The CP6b engine (app/rag/backfill_execute.py) takes three callables. This module
builds them from application functions that the CALLER injects, so it imports no
configuration, database, model, indexing-service or indexing-runner module. Only
app.rag.backfill (the pure classifier) and the standard library are imported.

    reclassify(id) -> Classification | None   fresh READ-ONLY connection per call
    request(id)    -> bool                    fresh session; request_indexing commits
    run(id)        -> IndexingRunResult       run_indexing with the preflight-validated embedder

No indexing lifecycle logic lives here: queuing, claiming, building, publishing and
failure recording all stay in app/rag/indexing_service.py and indexing_runner.py.

Nothing here catches exceptions from the injected functions. The engine records
an exception from any of the three callables as an ERROR holding only the class
name, so messages never reach a report. The failure strings returned by
write_target_failure are fixed texts.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.rag.backfill import Classification, DocumentState, build_plan

# ---------- fixed refusal texts (no exception text, no URL, no name) ----------

FAILURE_SESSION_FACTORY_MISMATCH = (
    "the session factory used to queue documents is not bound to the verified database engine"
)
FAILURE_RUNNER_FACTORY_MISMATCH = (
    "the session factory used by the indexing runner is not bound to the verified database engine"
)
FAILURE_SESSION_FACTORY_UNCHECKABLE = (
    "could not determine which database the queuing session factory writes to"
)
FAILURE_RUNNER_FACTORY_UNCHECKABLE = (
    "could not determine which database the indexing runner session factory writes to"
)


# ---------- reclassify ----------

def make_reclassify(
    engine: Any,
    *,
    configured_embedding_id: str,
    include_failed: bool,
    select_states: Callable[..., Sequence[DocumentState]],
    attach_sources: Callable[[Sequence[DocumentState]], Sequence[DocumentState]],
) -> Callable[[int], Classification | None]:
    """Build ``reclassify(document_id)`` for the engine's ``reclassify`` parameter.

    ``select_states`` is select_document_states and ``attach_sources`` is
    attach_source_status (both injected). ``configured_embedding_id`` and
    ``include_failed`` must be the values the plan was built with.
    """
    if not isinstance(configured_embedding_id, str) or not configured_embedding_id:
        raise ValueError("configured_embedding_id must be a non-empty string")
    if not isinstance(include_failed, bool):
        raise TypeError("include_failed must be a bool")

    def reclassify(document_id: int) -> Classification | None:
        # A fresh connection per call, so the decision uses the database's state NOW.
        # The read-only characteristic is applied before the first statement
        # (psycopg2 cannot change it inside an open transaction). The with-block
        # closes the connection on success and on any exception.
        with engine.connect() as connection:
            read_only = connection.execution_options(postgresql_readonly=True)
            states = list(select_states(read_only, document_ids=[document_id]))
        # The connection is closed here; the file-system check below needs none.

        if not states:
            return None  # deleted, or no longer processing_status='completed'
        if len(states) != 1:
            raise RuntimeError("selection returned more than one state for a single id")
        if states[0].document_id != document_id:
            raise RuntimeError("selection returned a different document than requested")

        resolved = list(attach_sources(states))
        if len(resolved) != 1:
            raise RuntimeError("source resolution changed the number of states")

        # limit=None: the limit was applied once, when the plan was built. Here only
        # this document's current decision matters, and a limit must never defer it.
        plan = build_plan(
            resolved,
            configured_embedding_id=configured_embedding_id,
            include_failed=include_failed,
            limit=None,
        )
        return plan.items[0].classification

    return reclassify


# ---------- request ----------

def make_request(
    session_factory: Callable[[], Any],
    request_indexing: Callable[[Any, int], bool],
) -> Callable[[int], bool]:
    """Build ``request(document_id)`` for the engine's ``request`` parameter.

    request_indexing opens and commits its own transaction (``with db.begin()``)
    and requires a session with none in progress, which a fresh session satisfies.
    The result is returned unchanged: the engine accepts only exactly ``True``.
    """

    def request(document_id: int) -> bool:
        with session_factory() as session:
            return request_indexing(session, document_id)

    return request


# ---------- run ----------

def make_run(
    run_indexing: Callable[..., Any],
    embedder: Any,
) -> Callable[[int], Any]:
    """Build ``run(document_id)`` for the engine's ``run`` parameter.

    ``embedder`` must be the very instance the preflight validated; it is passed
    through untouched, so run_indexing never calls get_embedder().
    """
    if embedder is None:
        raise ValueError("an embedder instance is required")

    def run(document_id: int) -> Any:
        return run_indexing(document_id, embedder=embedder)

    return run


# ---------- write-target identity ----------

def write_target_failure(
    runtime_engine: Any,
    session_factory: Callable[[], Any],
    runner_session_factory: Callable[[], Any],
) -> str | None:
    """None if both factories write to ``runtime_engine``; otherwise a fixed text.

    The live check (current_database()) ran on ``runtime_engine``. The queuing
    session and the runner's SessionLocal are separate objects, so this proves
    they point at that same engine before anything is written.
    """
    for factory, mismatch, uncheckable in (
        (session_factory, FAILURE_SESSION_FACTORY_MISMATCH, FAILURE_SESSION_FACTORY_UNCHECKABLE),
        (runner_session_factory, FAILURE_RUNNER_FACTORY_MISMATCH, FAILURE_RUNNER_FACTORY_UNCHECKABLE),
    ):
        try:
            with factory() as session:
                bound = session.get_bind()  # no connection is opened
        except Exception:  # noqa: BLE001 - never expose the message
            return uncheckable
        if bound is not runtime_engine:
            return mismatch
    return None