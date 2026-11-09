"""Execution safety checks for the indexing backfill (Component 12 / C5.5, CP4).

The CLI (a later checkpoint) must call these before any write. Nothing here writes, opens
the ``.env`` file, or makes a network connection:

  * ``check_static_safety`` is pure: every input is an argument, including the filesystem
    facts. It evaluates all checks, in a fixed order, and returns ALL failures at once.
  * ``inspect_filesystem`` gathers the filesystem facts with stat calls only.
  * ``check_live_database`` runs one ``SELECT current_database()``. The CLI calls it only
    after the static report is clean, so a URL that failed the host checks is never
    contacted.

Failures are returned, never raised. Every message is a fixed string that may include
safe values (database names, the environment string, the host, paths) but never a URL, a
password, ``.env`` contents or raw exception text. SQLAlchemy and driver messages can echo
connection details, so only an exception's class name is ever reported.

Unsafe conditions are checked on facts the caller supplies; the caller decides what a
non-empty report means (the CLI exits with code 2).
"""
from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.orm import Session

REQUIRED_ENVIRONMENT = "development"

# libpq reads these from the process environment. PGHOSTADDR and PGSERVICE can redirect the
# real connection even when the URL says "localhost", so their presence is refused.
_LIBPQ_HOST_ENV = ("PGHOST", "PGHOSTADDR", "PGSERVICE")

_LIVE_QUERY = text("SELECT current_database()")


class Check(str, Enum):
    """The checks, in the order they are evaluated and reported."""

    CURRENT_DIRECTORY = "current_directory"
    ENV_FILE = "env_file"
    UPLOAD_DIRECTORY = "upload_directory"
    ENVIRONMENT = "environment"
    DATABASE_URL = "database_url"
    DATABASE_NAME = "database_name"
    LIVE_DATABASE = "live_database"


@dataclass(frozen=True, slots=True)
class SafetyFailure:
    """One failed check. ``message`` is fixed text, safe to print."""

    check: Check
    message: str


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """The parts of the database URL that may be shown. There is deliberately no password."""

    host: str | None
    port: int | None
    database: str | None
    username: str | None

    def describe(self) -> str:
        host = self.host or "?"
        port = f":{self.port}" if self.port is not None else ""
        return f"{host}{port}/{self.database or '?'} as {self.username or '?'}"


@dataclass(frozen=True, slots=True)
class FilesystemFacts:
    """Facts gathered with stat calls only (see inspect_filesystem)."""

    env_file_exists: bool                # a regular file named .env in the repo root; never opened
    upload_root_is_dir: bool
    expected_upload_root: Path | None    # (repo_root / UPLOAD_DIR).resolve(); None if it cannot resolve


@dataclass(frozen=True, slots=True)
class SafetyReport:
    failures: tuple[SafetyFailure, ...]
    target: TargetInfo | None            # None when the URL could not be parsed

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------- inputs the caller gathers ----------


def default_repo_root() -> Path:
    """The repository root, from this file's location (app/rag/backfill_safety.py)."""
    return Path(__file__).resolve().parents[2]


def environment_is_explicit(settings: object) -> bool:
    """True if ENVIRONMENT was supplied (environment variable or .env), not left at its default.

    Uses pydantic's record of which fields were set. Anything unexpected counts as "not
    explicit", so this fails closed.
    """
    fields_set = getattr(settings, "model_fields_set", None)
    return isinstance(fields_set, (set, frozenset)) and "environment" in fields_set


def inspect_filesystem(
    repo_root: Path,
    upload_dir_setting: str,
    upload_root: Path,
    *,
    is_file: Callable[[Path], bool] | None = None,
    is_dir: Callable[[Path], bool] | None = None,
    resolve: Callable[[Path], Path] | None = None,
) -> FilesystemFacts:
    """Collect the filesystem facts with stat calls only: nothing is opened, read or written.

    Any OSError (or ValueError) counts as "not present", so a filesystem problem makes the
    safety check fail rather than pass.
    """
    is_file = is_file if is_file is not None else Path.is_file
    is_dir = is_dir if is_dir is not None else Path.is_dir
    resolve = resolve if resolve is not None else Path.resolve

    def present(check: Callable[[Path], bool], path: Path) -> bool:
        try:
            return bool(check(path))
        except (OSError, ValueError):
            return False

    try:
        expected = resolve(Path(repo_root) / upload_dir_setting)
    except (OSError, RuntimeError, ValueError):  # RuntimeError: a symlink loop on Python 3.12
        expected = None

    return FilesystemFacts(
        env_file_exists=present(is_file, Path(repo_root) / ".env"),
        upload_root_is_dir=present(is_dir, Path(upload_root)),
        expected_upload_root=expected,
    )


# ---------- static checks ----------


def _same_path(first: Path, second: Path) -> bool:
    """Compare two already-resolved paths; normcase makes this case-insensitive on Windows."""

    def normal(path: Path) -> str:
        return os.path.normcase(os.path.normpath(os.fspath(path)))

    return normal(first) == normal(second)


def _is_loopback(host: str) -> bool:
    """localhost, or an IP address that is a loopback address. DNS names never qualify."""
    name = host.lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _check_database_url(
    database_url: object, environ: Mapping[str, str]
) -> tuple[list[SafetyFailure], TargetInfo | None, object | None]:
    """Parse and vet the URL. Returns (failures, target, parsed URL)."""

    def fail(message: str) -> SafetyFailure:
        return SafetyFailure(Check.DATABASE_URL, message)

    try:
        url = make_url(database_url)
    except Exception:  # noqa: BLE001 - the error text can echo the URL, so it is never used
        return [fail("The database URL could not be parsed")], None, None

    failures: list[SafetyFailure] = []
    driver = url.drivername or ""
    if driver != "postgresql" and not driver.startswith("postgresql+"):
        failures.append(fail("The database URL is not a PostgreSQL URL"))

    host = url.host
    if not host:
        failures.append(
            fail("The database URL has no host (a missing host falls back to PGHOST or a unix socket)")
        )
    elif not _is_loopback(host):
        failures.append(fail(f"The database host '{host}' is not a loopback address"))

    if url.query:
        failures.append(
            fail("The database URL must not carry query parameters (they can override the host)")
        )

    present = sorted({name.upper() for name in environ} & set(_LIBPQ_HOST_ENV))
    if present:
        failures.append(
            fail(f"Refusing to run with {', '.join(present)} set: libpq can redirect the connection")
        )

    target = TargetInfo(host=url.host, port=url.port, database=url.database, username=url.username)
    return failures, target, url


def check_static_safety(
    *,
    database_url: str,
    environment: str | None,
    environment_explicit: bool,
    expected_database: str | None,
    cwd: Path,
    repo_root: Path,
    upload_root: Path,
    filesystem: FilesystemFacts,
    environ: Mapping[str, str],
) -> SafetyReport:
    """Evaluate every static check, in order, and return all failures.

    ``expected_database=None`` skips only the database-name comparison (a dry run without
    --database); every other check still runs. Pure: no network, no file access, no writes.
    """
    failures: list[SafetyFailure] = []

    # 1. The current directory is the repository root.
    if not _same_path(cwd, repo_root):
        failures.append(
            SafetyFailure(
                Check.CURRENT_DIRECTORY,
                f"Run this from the repository root ({repo_root}); the current directory is {cwd}",
            )
        )

    # 2. .env exists in the repository root (presence only; it is never read here).
    if not filesystem.env_file_exists:
        failures.append(SafetyFailure(Check.ENV_FILE, "No .env file was found in the repository root"))

    # 3. The configured upload directory exists and was resolved from the repository root.
    if not filesystem.upload_root_is_dir:
        failures.append(
            SafetyFailure(Check.UPLOAD_DIRECTORY, f"The upload directory does not exist: {upload_root}")
        )
    elif filesystem.expected_upload_root is None or not _same_path(
        upload_root, filesystem.expected_upload_root
    ):
        failures.append(
            SafetyFailure(
                Check.UPLOAD_DIRECTORY,
                "The upload directory was not resolved from the repository root and the configured UPLOAD_DIR",
            )
        )

    # 4. ENVIRONMENT is explicitly development.
    if environment != REQUIRED_ENVIRONMENT:
        shown = "not set" if environment is None else repr(environment)
        failures.append(
            SafetyFailure(
                Check.ENVIRONMENT, f"ENVIRONMENT is {shown}; it must be {REQUIRED_ENVIRONMENT!r}"
            )
        )
    elif not environment_explicit:
        failures.append(
            SafetyFailure(
                Check.ENVIRONMENT,
                "ENVIRONMENT is not set explicitly (the default value does not count)",
            )
        )

    # 5. The database URL is a loopback PostgreSQL URL with no query overrides.
    url_failures, target, url = _check_database_url(database_url, environ)
    failures.extend(url_failures)

    # 6. The URL's database name matches --database (skipped when no --database was given).
    if expected_database is not None and url is not None:
        actual = url.database
        if not expected_database:
            failures.append(SafetyFailure(Check.DATABASE_NAME, "--database must not be empty"))
        elif not actual:
            failures.append(SafetyFailure(Check.DATABASE_NAME, "The database URL names no database"))
        elif actual != expected_database:
            failures.append(
                SafetyFailure(
                    Check.DATABASE_NAME,
                    f"The database URL names '{actual}' but --database is '{expected_database}'",
                )
            )

    return SafetyReport(failures=tuple(failures), target=target)


# ---------- live check ----------


def check_live_database(conn: Session | Connection, expected_database: str) -> SafetyFailure | None:
    """Confirm the server really is the expected database. Returns None on success.

    Runs exactly one statement, ``SELECT current_database()``, which works on a read-only
    connection. It never commits, rolls back or closes: the caller owns ``conn``.

    Any ordinary exception becomes a failure naming only the exception CLASS. The message
    text of SQLAlchemy and driver errors can contain connection details, so it is never
    used, and nothing is chained or logged. KeyboardInterrupt and other BaseExceptions
    still propagate.
    """

    def fail(message: str) -> SafetyFailure:
        return SafetyFailure(Check.LIVE_DATABASE, message)

    if not isinstance(expected_database, str) or not expected_database:
        return fail("No expected database name was given for the live check")

    try:
        actual = conn.execute(_LIVE_QUERY).scalar_one()
    except Exception as exc:  # noqa: BLE001 - deliberately broad; only the class name is reported
        return fail(f"The live database check failed ({type(exc).__name__})")

    if not isinstance(actual, str):
        return fail("The live database check returned an unexpected value")
    if actual != expected_database:
        return fail(f"The connected database is '{actual}', not '{expected_database}'")
    return None