"""Self-tests for the invariant observer: it must accept every legal history and reject
every illegal one, naming the invariant. A checker that cannot fail proves nothing."""

import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from tests.chat_invariants import (
    PRE_ATTEMPT_CODES,
    RATE_LIMITED_CODES,
    VOCABULARY,
    AttemptRow,
    RequestRow,
    assert_valid,
    check_history,
    classify,
    load,
    observe,
    state_of,
)
from tests.chat_pipeline_db import (
    add_attempt,
    attempt_history,
    execute,
    expire_lease,
    make_request,
    make_user,
)

R = 100  # a reservation


def a(number, status, **kw):
    """An internally consistent attempt of ``status``; keyword arguments override fields."""
    base = {
        "failed_before_call": {
            "charged": 0,
            "error_code": "internal_error",
            "started": False,
        },
        "abandoned": {"charged": 0, "error_code": "lease_expired", "started": False},
        "rate_limited": {
            "charged": 0,
            "error_code": "provider_rate_limited",
            "started": True,
        },
        "running": {"charged": None, "error_code": None, "started": True},
        "rejected": {"charged": 0, "error_code": "provider_rejected", "started": True},
        "uncertain": {"charged": R, "error_code": "provider_timeout", "started": True},
        "failed_with_usage": {
            "charged": 10,
            "error_code": "invalid_response",
            "started": True,
            "prompt_tokens": 7,
            "completion_tokens": 3,
        },
        "succeeded": {
            "charged": 70,
            "error_code": None,
            "started": True,
            "prompt_tokens": 50,
            "completion_tokens": 20,
        },
    }[status]
    return AttemptRow(number=number, status=status, **{"reserved": R, **base, **kw})


def progress():
    return RequestRow("in_progress")


def failed(code="provider_timeout"):
    return RequestRow("failed", error_code=code, completed=True)


def succeeded():
    return RequestRow(
        "succeeded",
        completed=True,
        user_message_id=1,
        assistant_message_id=2,
        conversation_id=3,
    )


def invariants(request, attempts):
    return {v.invariant for v in check_history(request, attempts)}


# ---------------------------------------------------------------- legal histories, per state

LEGAL = [
    ("S1", progress(), []),
    ("S2", progress(), [a(1, "running")]),
    (
        "S2",
        progress(),
        [a(1, "abandoned"), a(2, "failed_before_call"), a(3, "running")],
    ),
    ("S2", progress(), [a(1, "rate_limited"), a(2, "running")]),
    ("S3", progress(), [a(1, "rate_limited")]),
    ("S3", progress(), [a(1, "failed_before_call"), a(2, "rate_limited")]),
    ("S3", progress(), [a(1, "rate_limited"), a(2, "rate_limited")]),
    ("S4", progress(), [a(1, "failed_before_call")]),
    ("S4", progress(), [a(1, "abandoned"), a(2, "failed_before_call")]),
    ("S4", progress(), [a(1, "rate_limited"), a(2, "failed_before_call")]),
    ("S5", progress(), [a(1, "uncertain", error_code="lease_expired")]),
    (
        "S5",
        progress(),
        [a(1, "rate_limited"), a(2, "uncertain", error_code="lease_expired")],
    ),
    ("S6", failed("no_context"), []),
    ("S6", failed("no_context"), []),
    ("S6", failed("provider_rate_limited"), [a(1, "rate_limited")]),
    (
        "S6",
        failed("provider_rate_limited"),
        [a(1, "rate_limited"), a(2, "rate_limited")],
    ),
    ("S6", failed("provider_timeout"), [a(1, "uncertain")]),
    ("S6", failed("lease_expired"), [a(1, "uncertain", error_code="lease_expired")]),
    ("S6", failed("invalid_response"), [a(1, "failed_with_usage")]),
    ("S6", failed("provider_rejected"), [a(1, "rejected")]),
    (
        "S6",
        failed("quota_exceeded"),
        [a(1, "rate_limited")],
    ),  # a retry refused by quota
    ("S6", failed("internal_error"), [a(1, "failed_before_call")]),
    ("S6", failed("provider_timeout"), [a(1, "rate_limited"), a(2, "uncertain")]),
    ("S7", succeeded(), [a(1, "succeeded")]),
    ("S7", succeeded(), [a(1, "rate_limited"), a(2, "succeeded")]),
    (
        "S7",
        succeeded(),
        [
            a(1, "abandoned"),
            a(2, "succeeded", prompt_tokens=None, completion_tokens=None, charged=R),
        ],
    ),
]


@pytest.mark.parametrize(("state", "request_row", "attempts"), LEGAL)
def test_every_legal_history_is_accepted_and_classified(state, request_row, attempts):
    assert check_history(request_row, attempts) == []
    assert classify(request_row, attempts) == state


# ---------------------------------------------------------------- illegal histories, per invariant

ILLEGAL = [
    # I1: numbering
    ("I1", progress(), [a(2, "running")]),
    ("I1", failed(), [a(1, "rate_limited"), a(3, "rate_limited")]),
    ("I1", progress(), [a(1, "failed_before_call"), a(1, "running")]),
    # I2: the running slot
    ("I2", progress(), [a(1, "running"), a(2, "running")]),
    ("I2", progress(), [a(1, "running"), a(2, "failed_before_call")]),
    ("I2", failed(), [a(1, "running")]),
    ("I2", succeeded(), [a(1, "running"), a(2, "succeeded")]),
    # I3: a paid attempt is final, whatever follows and however many attempts there are
    ("I3", failed(), [a(1, "uncertain"), a(2, "rate_limited")]),
    ("I3", failed(), [a(1, "failed_with_usage"), a(2, "failed_before_call")]),
    ("I3", succeeded(), [a(1, "rejected"), a(2, "succeeded")]),
    (
        "I3",
        progress(),
        [a(1, "failed_before_call"), a(2, "succeeded"), a(3, "running")],
    ),
    ("I3", failed(), [a(1, "uncertain"), a(2, "uncertain")]),
    ("I3", failed(), [a(1, "succeeded"), a(2, "failed_before_call")]),
    (
        "I3",
        failed(),
        [a(1, "rate_limited"), a(2, "rate_limited"), a(3, "rate_limited")],
    ),
    (
        "I3",
        progress(),
        [a(1, "uncertain", error_code="lease_expired"), a(2, "running")],
    ),
    # I4: one attempt's own consistency
    ("I4", failed(), [a(1, "failed_before_call", started=True)]),
    ("I4", failed(), [a(1, "abandoned", started=True)]),
    ("I4", failed("provider_timeout"), [a(1, "uncertain", started=False)]),
    ("I4", failed("provider_rate_limited"), [a(1, "rate_limited", started=False)]),
    ("I4", failed("provider_rejected"), [a(1, "rejected", started=False)]),
    ("I4", failed("provider_rate_limited"), [a(1, "rate_limited", charged=5)]),
    ("I4", failed("provider_rejected"), [a(1, "rejected", charged=R)]),
    ("I4", failed("internal_error"), [a(1, "failed_before_call", charged=R)]),
    ("I4", failed("provider_timeout"), [a(1, "uncertain", charged=0)]),
    ("I4", failed("invalid_response"), [a(1, "failed_with_usage", charged=R)]),
    (
        "I4",
        failed("invalid_response"),
        [a(1, "failed_with_usage", prompt_tokens=None, completion_tokens=None)],
    ),
    ("I4", succeeded(), [a(1, "succeeded", charged=1)]),
    ("I4", progress(), [a(1, "running", charged=0)]),
    ("I4", failed("provider_timeout"), [a(1, "uncertain", error_code=None)]),
    ("I4", failed(), [AttemptRow(1, "teleported", R, 0, "x", True)]),
    # I5: success and the succeeded attempt
    ("I5", succeeded(), []),
    ("I5", succeeded(), [a(1, "uncertain")]),
    ("I5", succeeded(), [a(1, "succeeded"), a(2, "succeeded")]),
    ("I5", failed("provider_timeout"), [a(1, "succeeded")]),
    ("I5", progress(), [a(1, "succeeded")]),
    # I6: an in_progress request holding a settled paid attempt (other than S5)
    ("I6", progress(), [a(1, "uncertain")]),  # uncertain, but not lease_expired
    ("I6", progress(), [a(1, "uncertain", error_code="lease_expired", charged=1)]),
    ("I6", progress(), [a(1, "failed_with_usage")]),
    ("I6", progress(), [a(1, "rejected")]),
    # I7: failed requests
    ("I7", failed("provider_timeout"), [a(1, "uncertain", error_code="lease_expired")]),
    ("I7", failed("invalid_response"), [a(1, "rejected")]),
    ("I7", failed("lease_expired"), [a(1, "failed_with_usage")]),
    # I8: the request row
    ("I8", RequestRow("in_progress", completed=True), []),
    ("I8", RequestRow("in_progress", error_code="x"), []),
    ("I8", RequestRow("in_progress", user_message_id=1), []),
    ("I8", RequestRow("failed", error_code=None, completed=True), []),
    ("I8", RequestRow("failed", error_code="x", completed=False), []),
    (
        "I8",
        RequestRow("failed", error_code="x", completed=True, assistant_message_id=2),
        [],
    ),
    (
        "I8",
        RequestRow("succeeded", completed=True, user_message_id=1),
        [a(1, "succeeded")],
    ),
    (
        "I8",
        RequestRow(
            "succeeded", completed=False, user_message_id=1, assistant_message_id=2
        ),
        [a(1, "succeeded")],
    ),
    (
        "I8",
        RequestRow(
            "succeeded",
            error_code="x",
            completed=True,
            user_message_id=1,
            assistant_message_id=2,
        ),
        [a(1, "succeeded")],
    ),
    ("I8", RequestRow("pending"), []),
]


@pytest.mark.parametrize(("invariant", "request_row", "attempts"), ILLEGAL)
def test_every_illegal_history_is_rejected_naming_the_invariant(
    invariant, request_row, attempts
):
    assert invariant in invariants(request_row, attempts)


def test_every_invariant_has_a_negative_case():
    assert {case[0] for case in ILLEGAL} == {f"I{n}" for n in range(1, 9)}


def test_every_status_has_a_legal_case_and_every_state_a_legal_history():
    assert {s for s, _, _ in LEGAL} == {f"S{n}" for n in range(1, 8)}
    seen = {x.status for _, _, attempts in LEGAL for x in attempts}
    assert seen == {
        "failed_before_call",
        "abandoned",
        "rate_limited",
        "running",
        "rejected",
        "uncertain",
        "failed_with_usage",
        "succeeded",
    }


def test_the_checker_looks_at_every_attempt_not_just_the_latest():
    """A paid attempt buried in the middle of a long history must be seen."""
    history = [
        a(1, "failed_before_call"),
        a(2, "uncertain"),
        a(3, "failed_before_call"),
        a(4, "failed_before_call"),
    ]
    assert "I3" in invariants(failed("internal_error"), history)
    # ...even though the latest attempt on its own is a perfectly legal S4 tail.
    tail = dataclasses.replace(history[-1], number=1)
    assert invariants(progress(), [tail]) == set()


def test_a_legal_history_is_not_mutated_into_an_illegal_one_by_reordering():
    legal = [a(1, "rate_limited"), a(2, "succeeded")]
    assert check_history(succeeded(), legal) == []
    assert "I3" in invariants(succeeded(), list(reversed(legal)))


# ---------------------------------------------------------------- the DB loader agrees


def test_the_loader_reads_real_rows_into_the_same_model():
    owner = make_user("observer@example.com")
    rid = make_request(owner)
    assert state_of(rid) == "S1"
    add_attempt(
        rid, "rate_limited", number=1, charged=0, error_code="provider_rate_limited"
    )
    assert state_of(rid) == "S3"
    add_attempt(rid, "running", number=2, started="now")
    assert state_of(rid) == "S2"
    assert_valid(rid)
    request, attempts = load(rid)
    assert [x.status for x in attempts] == ["rate_limited", "running"]
    assert request.status == "in_progress"


def test_the_loader_flags_a_real_illegal_history():
    owner = make_user("observer2@example.com")
    rid = make_request(owner)
    add_attempt(
        rid,
        "uncertain",
        number=1,
        reserved=10,
        charged=10,
        error_code="provider_timeout",
    )
    assert {v.invariant for v in observe(rid)} == {
        "I6"
    }  # in_progress + settled paid attempt
    with pytest.raises(AssertionError):
        assert_valid(rid)


def test_an_expired_lease_alone_does_not_make_a_running_attempt_illegal():
    owner = make_user("observer3@example.com")
    rid = make_request(owner)
    add_attempt(rid, "running", number=1, started="now")
    expire_lease(attempt_history(rid)[0].id)
    assert state_of(rid) == "S2"
    assert_valid(rid)


# ================================================================ C9.5c: hardened contract
# ---------------------------------------------------------------- I4: the attempt row


def problems(request_row, attempts, invariant):
    return [
        v.detail
        for v in check_history(request_row, attempts)
        if v.invariant == invariant
    ]


@pytest.mark.parametrize("started", [True, False])
def test_a_running_attempt_may_be_started_or_not(started):
    """No schema rule fixes it: NULL before mark_llm_started, set after. Both are real."""
    assert problems(progress(), [a(1, "running", started=started)], "I4") == []


I4_BAD = [
    (
        "reservation_zero",
        "failed_before_call",
        {"reserved": 0},
        "reserved_tokens must be positive",
    ),
    (
        "reservation_negative",
        "uncertain",
        {"reserved": -5, "charged": -5},
        "reserved_tokens must be positive",
    ),
    (
        "prompt_negative",
        "failed_with_usage",
        {"prompt_tokens": -1, "completion_tokens": 11, "charged": 10},
        "must not be negative",
    ),
    (
        "completion_negative",
        "failed_with_usage",
        {"prompt_tokens": 11, "completion_tokens": -1, "charged": 10},
        "must not be negative",
    ),
    ("charge_negative", "rejected", {"charged": -1}, "must not be negative"),
    (
        "running_with_prompt_usage",
        "running",
        {"prompt_tokens": 1, "completion_tokens": 1},
        "running must not carry usage",
    ),
    (
        "running_with_error_code",
        "running",
        {"error_code": "internal_error"},
        "must not carry an error_code",
    ),
    ("running_charged", "running", {"charged": 0}, "running must not be charged"),
    (
        "succeeded_with_error_code",
        "succeeded",
        {"error_code": "internal_error"},
        "must not carry an error_code",
    ),
    (
        "succeeded_prompt_only",
        "succeeded",
        {"completion_tokens": None, "charged": 100},
        "both set or both NULL",
    ),
    (
        "succeeded_completion_only",
        "succeeded",
        {"prompt_tokens": None, "charged": 100},
        "both set or both NULL",
    ),
    (
        "succeeded_partial_charged_usage",
        "succeeded",
        {"completion_tokens": None, "charged": 50},
        "both set or both NULL",
    ),
    (
        "succeeded_wrong_total",
        "succeeded",
        {"charged": 71},
        "must charge usage or reservation",
    ),
    (
        "succeeded_no_usage_wrong_charge",
        "succeeded",
        {"prompt_tokens": None, "completion_tokens": None, "charged": 1},
        "must charge usage or reservation",
    ),
    (
        "failed_with_usage_no_usage",
        "failed_with_usage",
        {"prompt_tokens": None, "completion_tokens": None, "charged": R},
        "must carry usage",
    ),
    (
        "failed_with_usage_prompt_only",
        "failed_with_usage",
        {"completion_tokens": None, "charged": 7},
        "both set or both NULL",
    ),
    (
        "failed_with_usage_wrong_total",
        "failed_with_usage",
        {"charged": 11},
        "must charge its usage",
    ),
    (
        "uncertain_with_usage",
        "uncertain",
        {"prompt_tokens": 1, "completion_tokens": 1},
        "must not carry usage",
    ),
    (
        "rejected_with_usage",
        "rejected",
        {"prompt_tokens": 1, "completion_tokens": 1},
        "must not carry usage",
    ),
    (
        "rate_limited_with_usage",
        "rate_limited",
        {"prompt_tokens": 1, "completion_tokens": 1},
        "must not carry usage",
    ),
    (
        "unstarted_with_usage",
        "failed_before_call",
        {"prompt_tokens": 0, "completion_tokens": 0},
        "must not carry usage",
    ),
    (
        "abandoned_with_usage",
        "abandoned",
        {"prompt_tokens": 1, "completion_tokens": 1},
        "must not carry usage",
    ),
    (
        "unstarted_but_started",
        "abandoned",
        {"started": True},
        "must not have llm_started_at",
    ),
    (
        "paid_but_not_started",
        "uncertain",
        {"started": False},
        "must have llm_started_at",
    ),
    ("unstarted_charged", "failed_before_call", {"charged": 1}, "must charge 0"),
    (
        "uncertain_not_reservation",
        "uncertain",
        {"charged": 1},
        "must charge its reservation",
    ),
    (
        "settled_without_code",
        "rejected",
        {"error_code": None},
        "must carry an error_code",
    ),
]


@pytest.mark.parametrize(
    ("status", "overrides", "expected"),
    [(s, o, e) for _, s, o, e in I4_BAD],
    ids=[case[0] for case in I4_BAD],
)
def test_every_bad_attempt_row_is_flagged_by_i4(status, overrides, expected):
    row = a(1, status, **overrides)
    assert any(expected in d for d in problems(progress(), [row], "I4")), problems(
        progress(), [row], "I4"
    )


@pytest.mark.parametrize(
    "status",
    [
        "failed_before_call",
        "abandoned",
        "rate_limited",
        "running",
        "rejected",
        "uncertain",
        "failed_with_usage",
        "succeeded",
    ],
)
def test_the_well_formed_row_of_every_status_has_no_i4_problem(status):
    assert problems(progress(), [a(1, status)], "I4") == []


def test_succeeded_accepts_complete_usage_or_no_usage_and_nothing_between():
    usage = a(1, "succeeded")
    none = a(1, "succeeded", prompt_tokens=None, completion_tokens=None, charged=R)
    assert problems(succeeded(), [usage], "I4") == []
    assert problems(succeeded(), [none], "I4") == []
    zero = a(1, "succeeded", prompt_tokens=0, completion_tokens=0, charged=0)
    assert problems(succeeded(), [zero], "I4") == []  # zero is real usage, not absence


# ---------------------------------------------------------------- I7: codes by last attempt

LAST_KINDS = {
    "none": [],
    "failed_before_call": [a(1, "failed_before_call")],
    "abandoned": [a(1, "abandoned")],
    "rate_limited": [a(1, "rate_limited")],
    "after_a_429_then_unstarted": [a(1, "rate_limited"), a(2, "failed_before_call")],
}


def legal_codes(kind):
    return RATE_LIMITED_CODES if kind == "rate_limited" else PRE_ATTEMPT_CODES


@pytest.mark.parametrize("code", sorted(VOCABULARY))
@pytest.mark.parametrize("kind", sorted(LAST_KINDS))
def test_i7_allows_exactly_the_codes_the_pipeline_can_write_after_each_last_attempt(
    kind, code
):
    flagged = bool(problems(failed(code), LAST_KINDS[kind], "I7"))
    assert flagged is (code not in legal_codes(kind))


def test_i7_code_tables_are_what_the_pipeline_writes():
    assert PRE_ATTEMPT_CODES == {"no_context", "internal_error", "quota_exceeded"}
    assert RATE_LIMITED_CODES == PRE_ATTEMPT_CODES | {"provider_rate_limited"}


@pytest.mark.parametrize(
    "code", ["", "x", "boom", "Provider_Timeout", "provider timeout"]
)
def test_i7_rejects_codes_outside_the_frozen_vocabulary(code):
    assert any("frozen vocabulary" in d for d in problems(failed(code), [], "I7"))
    assert any(
        "frozen vocabulary" in d
        for d in problems(failed(code), [a(1, "uncertain")], "I7")
    )


@pytest.mark.parametrize("status", ["uncertain", "rejected", "failed_with_usage"])
def test_i7_paid_attempt_code_must_match_exactly(status):
    attempt = a(1, status)
    assert problems(failed(attempt.error_code), [attempt], "I7") == []
    other = next(c for c in sorted(VOCABULARY) if c != attempt.error_code)
    assert problems(failed(other), [attempt], "I7")


# ---------------------------------------------------------------- I3: the 429 retry boundary


@pytest.mark.parametrize(
    "third",
    [
        "running",
        "failed_before_call",
        "abandoned",
        "rate_limited",
        "uncertain",
        "rejected",
        "failed_with_usage",
        "succeeded",
    ],
)
def test_no_attempt_may_follow_two_rate_limited_attempts(third):
    history = [a(1, "rate_limited"), a(2, "rate_limited"), a(3, third)]
    request_row = succeeded() if third == "succeeded" else progress()
    assert any("retry limit" in d for d in problems(request_row, history, "I3"))


def test_two_429s_with_nothing_after_them_are_legal_and_a_third_attempt_is_not():
    assert (
        check_history(
            failed("provider_rate_limited"),
            [a(1, "rate_limited"), a(2, "rate_limited")],
        )
        == []
    )
    assert problems(
        progress(), [a(1, "rate_limited"), a(2, "rate_limited"), a(3, "running")], "I3"
    )


def test_a_second_429_after_an_unstarted_attempt_is_still_the_one_allowed_retry():
    history = [a(1, "rate_limited"), a(2, "failed_before_call"), a(3, "rate_limited")]
    assert check_history(failed("provider_rate_limited"), history) == []
    assert problems(progress(), history + [a(4, "running")], "I3")


# ---------------------------------------------------------------- I8: the request row


I8_POINTERS = [
    ("only_user", {"user_message_id": 1, "conversation_id": 3}),
    ("only_assistant", {"assistant_message_id": 2, "conversation_id": 3}),
    ("no_conversation", {"user_message_id": 1, "assistant_message_id": 2}),
    ("only_conversation", {"conversation_id": 3}),
    (
        "both_messages_no_conversation",
        {"user_message_id": 1, "assistant_message_id": 2},
    ),
]


@pytest.mark.parametrize(
    ("name", "pointers"), I8_POINTERS, ids=[n for n, _ in I8_POINTERS]
)
def test_a_succeeded_request_needs_all_three_pointers_or_none(name, pointers):
    row = RequestRow("succeeded", completed=True, **pointers)
    assert any("only some" in d for d in problems(row, [a(1, "succeeded")], "I8"))


def test_a_succeeded_request_whose_conversation_was_deleted_is_legal():
    """Deleting the conversation sets conversation_id and both message ids NULL together."""
    row = RequestRow("succeeded", completed=True)
    assert check_history(row, [a(1, "succeeded")]) == []


# ---------------------------------------------------------------- I8: against the database

SHA = hashlib.sha256(b"chunk").hexdigest()
ANSWER = "It says so [1]."


def vector():
    values = [0.0] * 384
    values[0] = 1.0
    return "[" + ",".join(str(v) for v in values) + "]"


def make_document(owner):
    document = execute(
        "INSERT INTO documents (filename, owner_id, source_version, source_sha256,"
        " indexing_status, indexed_source_version, indexed_signature,"
        " indexed_embedding_id, indexed_at)"
        " VALUES ('d.txt', :o, 1, :s, 'indexed', 1, 'sig', 'emb', now()) RETURNING id",
        o=owner,
        s="b" * 64,
    ).scalar_one()
    chunk = execute(
        "INSERT INTO document_chunks (document_id, chunk_index, start_offset, end_offset,"
        " text, span_sha256, embedding)"
        " VALUES (:d, 0, 0, 5, 'chunk', :h, CAST(:v AS vector)) RETURNING id",
        d=document,
        h=SHA,
        v=vector(),
    ).scalar_one()
    return document, chunk


class Built:
    """A legal succeeded request with its conversation, message pair, citation and attempt."""

    def __init__(self, owner, with_citation=True):
        self.owner = owner
        self.document, self.chunk = make_document(owner)
        self.conversation = execute(
            "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=owner
        ).scalar_one()
        self.user = execute(
            "INSERT INTO messages (conversation_id, seq, role, content)"
            " VALUES (:c, 1, 'user', 'a question') RETURNING id",
            c=self.conversation,
        ).scalar_one()
        self.assistant = execute(
            "INSERT INTO messages (conversation_id, seq, role, content)"
            " VALUES (:c, 2, 'assistant', :t) RETURNING id",
            c=self.conversation,
            # An answer with no citation row carries no valid marker (publication keeps every
            # valid marker and cites it), so the citation-less fixture says so without one.
            t=ANSWER if with_citation else "It says so.",
        ).scalar_one()
        self.request = make_request(owner)
        add_attempt(self.request, "succeeded", charged=70, usage=(50, 20))
        self.citation = None
        if with_citation:
            self.citation = execute(
                "INSERT INTO message_citations (message_id, ordinal, marker, chunk_id,"
                " document_id, chunk_index, start_offset, end_offset, span_sha256,"
                " source_version, source_sha256)"
                " VALUES (:m, 1, 1, :k, :d, 0, 0, 5, :h, 1, :s) RETURNING id",
                m=self.assistant,
                k=self.chunk,
                d=self.document,
                h=SHA,
                s="b" * 64,
            ).scalar_one()
        execute(
            "UPDATE chat_requests SET status = 'succeeded', conversation_id = :c,"
            " user_message_id = :u, assistant_message_id = :a, error_code = NULL,"
            " completed_at = :t WHERE id = :r",
            c=self.conversation,
            u=self.user,
            a=self.assistant,
            t=datetime.now(UTC) + timedelta(seconds=1),
            r=self.request,
        )

    def violations(self):
        return [f"{v.invariant}: {v.detail}" for v in observe(self.request)]


@pytest.fixture
def built():
    return Built(make_user("built@example.com"))


def test_a_legal_succeeded_request_has_no_violation(built):
    assert built.violations() == []
    assert_valid(built.request)


def test_a_legal_succeeded_request_without_citations_has_no_violation():
    assert (
        Built(make_user("nocite@example.com"), with_citation=False).violations() == []
    )


def test_deleting_the_conversation_leaves_a_legal_request(built):
    execute("DELETE FROM conversations WHERE id = :c", c=built.conversation)
    request, _ = load(built.request)
    assert (
        request.conversation_id,
        request.user_message_id,
        request.assistant_message_id,
    ) == (
        None,
        None,
        None,
    )
    assert built.violations() == []


def test_a_deleted_chunk_or_document_keeps_the_citation_legal(built):
    execute("DELETE FROM documents WHERE id = :d", d=built.document)
    assert (
        execute(
            "SELECT chunk_id FROM message_citations WHERE id = :c", c=built.citation
        ).scalar()
        is None
    )
    assert built.violations() == []


def other_user():
    return make_user("other@example.com")


CORRUPTIONS = [
    (
        "assistant_has_user_role",
        "UPDATE messages SET role = 'user' WHERE id = {assistant}",
        "assistant_message_id points at",
    ),
    (
        "user_has_assistant_role",
        "UPDATE messages SET role = 'assistant' WHERE id = {user}",
        "user_message_id points at",
    ),
    (
        "one_message_for_both",
        "UPDATE chat_requests SET assistant_message_id = user_message_id WHERE id = {request}",
        "same row",
    ),
    (
        "assistant_seq_gap",
        "UPDATE messages SET seq = 9 WHERE id = {assistant}",
        "does not follow user seq",
    ),
    (
        "assistant_before_user",
        "UPDATE messages SET seq = 3 WHERE id = {user}",
        "does not follow user seq",
    ),
    (
        "request_points_elsewhere",
        "UPDATE chat_requests SET conversation_id = {other_conversation} WHERE id = {request}",
        "not in the request's conversation",
    ),
    (
        "conversation_owned_by_another",
        "UPDATE conversations SET owner_id = {other_user} WHERE id = {conversation}",
        "belongs to another user",
    ),
    (
        "message_moved_to_another_conversation",
        "UPDATE messages SET conversation_id = {other_conversation}, seq = 1 WHERE id = {assistant}",
        "not in the request's conversation",
    ),
    (
        "citation_on_the_user_message",
        "INSERT INTO message_citations (message_id, ordinal, marker, document_id, chunk_index, start_offset, end_offset, span_sha256, source_version, source_sha256) VALUES ({user}, 1, 1, {document}, 0, 0, 5, '"
        + SHA
        + "', 1, '"
        + "b" * 64
        + "')",
        "user message has citations",
    ),
    (
        "ordinal_gap",
        "UPDATE message_citations SET ordinal = 3 WHERE id = {citation}",
        "are not 1..n",
    ),
    (
        "marker_not_in_the_answer",
        "UPDATE message_citations SET marker = 9 WHERE id = {citation}",
        "does not appear in the answer",
    ),
    (
        "snapshot_offsets_disagree",
        "UPDATE message_citations SET start_offset = 1 WHERE id = {citation}",
        "snapshot disagrees",
    ),
    (
        "snapshot_hash_disagrees",
        "UPDATE message_citations SET span_sha256 = '"
        + "f" * 64
        + "' WHERE id = {citation}",
        "snapshot disagrees",
    ),
    (
        "snapshot_pages_disagree",
        "UPDATE message_citations SET page_start = 1, page_end = 1 WHERE id = {citation}",
        "snapshot disagrees",
    ),
    (
        "snapshot_chunk_index_disagrees",
        "UPDATE message_citations SET chunk_index = 4 WHERE id = {citation}",
        "snapshot disagrees",
    ),
    (
        "citation_document_of_another_user",
        "UPDATE message_citations SET document_id = {other_document} WHERE id = {citation}",
        "document belongs to another user",
    ),
    (
        "completed_before_the_attempt_settled",
        "UPDATE chat_requests SET completed_at = completed_at - interval '1 hour' WHERE id = {request}",
        "completed before its last attempt",
    ),
    (
        "assistant_message_deleted",
        "DELETE FROM messages WHERE id = {assistant}",
        "only some",
    ),
    (
        "request_not_succeeded_but_has_ids",
        "UPDATE chat_requests SET status = 'failed', error_code = 'internal_error', creates_conversation = false WHERE id = {request}",
        "no message ids",
    ),
]


@pytest.mark.parametrize(
    ("sql", "expected"),
    [(c[1], c[2]) for c in CORRUPTIONS],
    ids=[c[0] for c in CORRUPTIONS],
)
def test_every_corrupted_database_fixture_is_flagged(built, sql, expected):
    assert built.violations() == []  # the fixture is legal until corrupted
    other = other_user()
    other_conversation = execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id", o=built.owner
    ).scalar_one()
    other_document, _ = make_document(other)
    execute(
        sql.format(
            assistant=built.assistant,
            user=built.user,
            request=built.request,
            conversation=built.conversation,
            other_conversation=other_conversation,
            other_user=other,
            document=built.document,
            other_document=other_document,
            citation=built.citation,
        )
    )
    found = built.violations()
    # Every corruption here is a database-half I8 failure: the reported invariant must be I8,
    # not merely some detail string that happens to match.
    assert any(v.startswith("I8: ") and expected in v for v in found), found


# ------------------------------------------- I8: ownership and settlement time, any status


def conversation_request(owner, conversation_owner, status, **fields):
    """A request naming an existing conversation (creates_conversation = false)."""
    conversation = execute(
        "INSERT INTO conversations (owner_id) VALUES (:o) RETURNING id",
        o=conversation_owner,
    ).scalar_one()
    request = make_request(owner, status=status, error_code="internal_error")
    execute(
        "UPDATE chat_requests SET creates_conversation = false, conversation_id = :c"
        " WHERE id = :r",
        c=conversation,
        r=request,
    )
    return request


@pytest.mark.parametrize("status", ["in_progress", "failed"])
def test_a_request_naming_another_users_conversation_is_flagged_in_any_status(status):
    owner, other = make_user("owner@example.com"), other_user()
    request = conversation_request(owner, other, status)
    found = [f"{v.invariant}: {v.detail}" for v in observe(request)]
    assert found == ["I8: the request's conversation belongs to another user"], found


@pytest.mark.parametrize("status", ["in_progress", "failed"])
def test_a_request_naming_its_own_conversation_is_legal_in_any_status(status):
    owner = make_user("owner@example.com")
    request = conversation_request(owner, owner, status)
    assert observe(request) == []


def test_the_schema_forces_finished_at_on_every_settled_attempt():
    """The dependency check_database relies on: only a running attempt may lack finished_at."""
    request = make_request(make_user("owner@example.com"))
    add_attempt(
        request,
        "failed_before_call",
        started=None,
        charged=0,
        error_code="internal_error",
    )
    with pytest.raises(IntegrityError) as excinfo:
        execute(
            "UPDATE chat_request_attempts SET finished_at = NULL WHERE request_id = :r",
            r=request,
        )
    assert "ck_chat_request_attempts_finished_at_consistent" in str(excinfo.value)


@pytest.mark.parametrize("status", ["failed", "succeeded"])
def test_a_terminal_request_whose_last_attempt_is_still_running_is_flagged(status):
    """finished_at is NULL only for a running attempt, so this is the missing-settlement case."""
    request = make_request(make_user("owner@example.com"))
    add_attempt(request, "running")
    execute(
        "UPDATE chat_requests SET status = :s, error_code = :e, completed_at = now()"
        " WHERE id = :r",
        s=status,
        e="internal_error" if status == "failed" else None,
        r=request,
    )
    found = [f"{v.invariant}: {v.detail}" for v in observe(request)]
    assert any(v.startswith("I8: ") and "has no finished_at" in v for v in found), found


# ---------------------------------------------------- I8: citation order and snapshot semantics


def cite(built, answer, citations):
    """Replace the answer and the citations: ``citations`` is [(ordinal, marker, chunk_index)]."""
    second = execute(
        "INSERT INTO document_chunks (document_id, chunk_index, start_offset, end_offset,"
        " text, span_sha256, embedding)"
        " VALUES (:d, 1, 5, 10, 'chunk', :h, CAST(:v AS vector)) RETURNING id",
        d=built.document,
        h=SHA,
        v=vector(),
    ).scalar_one()
    chunks = {0: (built.chunk, 0, 5), 1: (second, 5, 10)}
    execute("DELETE FROM message_citations WHERE message_id = :m", m=built.assistant)
    execute(
        "UPDATE messages SET content = :t WHERE id = :m", t=answer, m=built.assistant
    )
    for ordinal, marker, index in citations:
        chunk, start, end = chunks[index]
        execute(
            "INSERT INTO message_citations (message_id, ordinal, marker, chunk_id,"
            " document_id, chunk_index, start_offset, end_offset, span_sha256,"
            " source_version, source_sha256)"
            " VALUES (:m, :o, :k, :c, :d, :i, :s, :e, :h, 1, :v)",
            m=built.assistant,
            o=ordinal,
            k=marker,
            c=chunk,
            d=built.document,
            i=index,
            s=start,
            e=end,
            h=SHA,
            v="b" * 64,
        )


def test_marker_and_ordinal_are_different_numbers_and_may_differ(built):
    """The first marker to appear gets ordinal 1, whatever its number."""
    cite(built, "B [2] then A [1].", [(1, 2, 1), (2, 1, 0)])
    assert built.violations() == []


def test_a_repeated_marker_is_one_citation(built):
    cite(built, "A [1] and again [1].", [(1, 1, 0)])
    assert built.violations() == []


def test_ordinals_that_do_not_follow_first_appearance_are_flagged(built):
    cite(built, "B [2] then A [1].", [(1, 1, 0), (2, 2, 1)])
    found = built.violations()
    assert any(v.startswith("I8: ") and "first appearance" in v for v in found), found


def test_a_marker_kept_in_the_answer_without_a_citation_is_flagged(built):
    cite(built, "A [1] and B [2].", [(1, 1, 0)])
    found = built.violations()
    assert any(v.startswith("I8: ") and "first appearance" in v for v in found), found


def test_a_code_span_marker_is_not_held_against_the_answer(built):
    """The production parser leaves ``[2]`` in a code span uncited; the observer skips order."""
    cite(built, "Literal `[2]` but cited [1].", [(1, 1, 0)])
    assert built.violations() == []


def test_a_citation_snapshot_of_an_older_source_version_stays_legal(built):
    """source_version / source_sha256 record what the answer was built from; a later re-index
    of the document must not make the history illegal."""
    execute(
        "UPDATE documents SET source_version = 2, indexed_source_version = 2,"
        " source_sha256 = :s WHERE id = :d",
        s="c" * 64,
        d=built.document,
    )
    assert built.violations() == []


def test_a_leading_zero_bracket_is_not_a_marker(built):
    """The parser's marker has no leading zero, so ``[01]`` is never a citation: no row, no flag."""
    cite(built, "Only a literal [01] here.", [])
    assert built.violations() == []
