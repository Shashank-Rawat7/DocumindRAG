"""allow rejected attempt status

Revision ID: c4f1a7d92b60
Revises: e9749572298d
Create Date: 2026-11-17 16:30:00.000000

Component C9 / C9.4a (migration #8). Hand-written: do NOT regenerate with --autogenerate.
Adds the attempt status 'rejected': a started attempt the provider rejected before
generating anything, settled with charge 0 and used only by
reject_and_fail_request (which also fails the request in the same transaction).

Only the three attempt CHECKs that name the status vocabulary change; they are dropped and
recreated under the same names with their new frozen text:
- ck_chat_request_attempts_status_valid: 'rejected' is a valid status.
- ck_chat_request_attempts_llm_started_consistent: 'rejected' needs llm_started_at.
- ck_chat_request_attempts_charge_consistent: 'rejected' has no usage and charge 0.
No column, index, foreign key or data changes. Do not import app constants here.

Downgrade restores the previous texts and REFUSES to run while any 'rejected' attempt
exists: such rows would violate the old CHECKs, and rewriting them into another status
would corrupt the retry-cap and quota history.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4f1a7d92b60"
down_revision: str | Sequence[str] | None = "e9749572298d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "chat_request_attempts"

# Frozen at this revision on purpose. Tests evaluate these exact texts and compare them
# with the model's CHECK texts.
ATTEMPT_CHARGE_CONSISTENT = (
    "(status = 'running' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NULL) "
    "OR (status = 'succeeded' AND charged_tokens IS NOT NULL AND ("
    "(prompt_tokens IS NOT NULL AND completion_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens = reserved_tokens))) "
    "OR (status = 'failed_with_usage' AND prompt_tokens IS NOT NULL "
    "AND completion_tokens IS NOT NULL AND charged_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (status IN ('failed_before_call', 'abandoned', 'rate_limited', 'rejected') "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = 0) "
    "OR (status = 'uncertain' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = reserved_tokens)"
)

ATTEMPT_CHECKS_CHANGED = [
    (
        "ck_chat_request_attempts_status_valid",
        (
            "status IN ('running', 'succeeded', 'failed_before_call', "
            "'failed_with_usage', 'rate_limited', 'uncertain', 'abandoned', 'rejected')"
        ),
    ),
    (
        "ck_chat_request_attempts_llm_started_consistent",
        (
            "status = 'running' "
            "OR (status IN ('succeeded', 'failed_with_usage', 'rate_limited', 'uncertain', "
            "'rejected') AND llm_started_at IS NOT NULL) "
            "OR (status IN ('failed_before_call', 'abandoned') AND llm_started_at IS NULL)"
        ),
    ),
    (
        "ck_chat_request_attempts_charge_consistent",
        ATTEMPT_CHARGE_CONSISTENT,
    ),
]

# The texts migration #7 (e9749572298d) created, restored by downgrade().
PREVIOUS_ATTEMPT_CHARGE_CONSISTENT = (
    "(status = 'running' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NULL) "
    "OR (status = 'succeeded' AND charged_tokens IS NOT NULL AND ("
    "(prompt_tokens IS NOT NULL AND completion_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens = reserved_tokens))) "
    "OR (status = 'failed_with_usage' AND prompt_tokens IS NOT NULL "
    "AND completion_tokens IS NOT NULL AND charged_tokens IS NOT NULL "
    "AND charged_tokens = prompt_tokens + completion_tokens) "
    "OR (status IN ('failed_before_call', 'abandoned', 'rate_limited') "
    "AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = 0) "
    "OR (status = 'uncertain' AND prompt_tokens IS NULL AND completion_tokens IS NULL "
    "AND charged_tokens IS NOT NULL AND charged_tokens = reserved_tokens)"
)

PREVIOUS_ATTEMPT_CHECKS_CHANGED = [
    (
        "ck_chat_request_attempts_status_valid",
        (
            "status IN ('running', 'succeeded', 'failed_before_call', "
            "'failed_with_usage', 'rate_limited', 'uncertain', 'abandoned')"
        ),
    ),
    (
        "ck_chat_request_attempts_llm_started_consistent",
        (
            "status = 'running' "
            "OR (status IN ('succeeded', 'failed_with_usage', 'rate_limited', 'uncertain') "
            "AND llm_started_at IS NOT NULL) "
            "OR (status IN ('failed_before_call', 'abandoned') AND llm_started_at IS NULL)"
        ),
    ),
    (
        "ck_chat_request_attempts_charge_consistent",
        PREVIOUS_ATTEMPT_CHARGE_CONSISTENT,
    ),
]


def _replace_checks(checks: list[tuple[str, str]]) -> None:
    for name, _ in checks:
        op.drop_constraint(name, TABLE, type_="check")
    for name, condition in checks:
        op.create_check_constraint(name, TABLE, condition)


def upgrade() -> None:
    """Upgrade schema."""
    _replace_checks(ATTEMPT_CHECKS_CHANGED)


def downgrade() -> None:
    """Downgrade schema. Refuses while any 'rejected' attempt exists (see module docstring)."""
    rejected = (
        op.get_bind()
        .exec_driver_sql(f"SELECT count(*) FROM {TABLE} WHERE status = 'rejected'")
        .scalar_one()
    )
    if rejected:
        raise RuntimeError(
            f"cannot downgrade: {rejected} 'rejected' attempt row(s) exist; "
            "they would violate the previous CHECK constraints"
        )
    _replace_checks(PREVIOUS_ATTEMPT_CHECKS_CHANGED)
