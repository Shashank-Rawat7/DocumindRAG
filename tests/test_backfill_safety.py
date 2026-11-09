"""Tests for the backfill execution safety checks (C5.5 / CP4).

The static checks are pure and need no database. The live-check tests use the test
database through the existing fixtures. No test reads the real .env: files are created in
tmp_path, and Settings is always built with an explicit temporary env file or none.
"""
import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.core.database import engine
from app.rag import backfill_safety
from app.rag.backfill_safety import (
    Check,
    FilesystemFacts,
    SafetyFailure,
    SafetyReport,
    TargetInfo,
    check_live_database,
    check_static_safety,
    default_repo_root,
    environment_is_explicit,
    inspect_filesystem,
)

SECRET = "s3cr3t-p4ssw0rd-XYZ"


def url(host="localhost", *, port=5434, database="documind_db", user="documind_user", query=""):
    host_part = f"[{host}]" if ":" in host else host
    return f"postgresql://{user}:{SECRET}@{host_part}:{port}/{database}{query}"


GOOD_URL = url()


@pytest.fixture
def repo(tmp_path) -> Path:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "uploads").mkdir()
    return root


def good_kwargs(repo: Path) -> dict:
    uploads = repo / "uploads"
    return dict(
        database_url=GOOD_URL,
        environment="development",
        environment_explicit=True,
        expected_database="documind_db",
        cwd=repo,
        repo_root=repo,
        upload_root=uploads,
        filesystem=FilesystemFacts(env_file_exists=True, upload_root_is_dir=True, expected_upload_root=uploads),
        environ={},
    )


def run(repo: Path, **overrides) -> SafetyReport:
    kwargs = good_kwargs(repo)
    kwargs.update(overrides)
    return check_static_safety(**kwargs)


def failed(report: SafetyReport) -> list[Check]:
    return [failure.check for failure in report.failures]


def everything_printable(report: SafetyReport) -> str:
    parts = [repr(report), str(report)]
    parts += [failure.message for failure in report.failures]
    if report.target is not None:
        parts += [repr(report.target), report.target.describe()]
    return "\n".join(parts)


# ---------- the passing case ----------

def test_a_fully_safe_configuration_passes(repo):
    report = run(repo)
    assert report.ok is True
    assert report.failures == ()
    assert report.target == TargetInfo(host="localhost", port=5434, database="documind_db", username="documind_user")
    assert report.target.describe() == "localhost:5434/documind_db as documind_user"


def test_the_repo_root_is_found_from_the_module_location():
    root = default_repo_root()
    assert (root / "alembic.ini").is_file()
    assert (root / "app").is_dir()
    assert (root / "app" / "rag" / "backfill_safety.py").is_file()


# ---------- 1. current directory ----------

def test_a_different_current_directory_fails(repo, tmp_path):
    assert failed(run(repo, cwd=tmp_path)) == [Check.CURRENT_DIRECTORY]


def test_a_subdirectory_of_the_repo_root_fails(repo):
    sub = repo / "scripts"
    sub.mkdir()
    assert failed(run(repo, cwd=sub)) == [Check.CURRENT_DIRECTORY]


def test_equivalent_spellings_of_the_repo_root_pass(repo):
    (repo / "sub").mkdir()
    assert run(repo, cwd=repo / "sub" / "..").ok


# ---------- 2. .env file ----------

def test_a_missing_env_file_fails(repo):
    facts = FilesystemFacts(env_file_exists=False, upload_root_is_dir=True, expected_upload_root=repo / "uploads")
    assert failed(run(repo, filesystem=facts)) == [Check.ENV_FILE]


# ---------- 3. upload directory ----------

def test_a_missing_upload_directory_fails(repo):
    facts = FilesystemFacts(env_file_exists=True, upload_root_is_dir=False, expected_upload_root=repo / "uploads")
    assert failed(run(repo, filesystem=facts)) == [Check.UPLOAD_DIRECTORY]


def test_an_upload_root_that_is_not_the_expected_one_fails(repo, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert failed(run(repo, upload_root=elsewhere)) == [Check.UPLOAD_DIRECTORY]


def test_an_unresolvable_expected_upload_root_fails(repo):
    facts = FilesystemFacts(env_file_exists=True, upload_root_is_dir=True, expected_upload_root=None)
    assert failed(run(repo, filesystem=facts)) == [Check.UPLOAD_DIRECTORY]


# ---------- 4. environment ----------

@pytest.mark.parametrize("environment", ["Development", "DEVELOPMENT", "test", "production", "staging", "", " development"])
def test_any_environment_but_exactly_development_fails(repo, environment):
    assert failed(run(repo, environment=environment)) == [Check.ENVIRONMENT]


def test_an_unset_environment_fails(repo):
    report = run(repo, environment=None)
    assert failed(report) == [Check.ENVIRONMENT]
    assert "not set" in report.failures[0].message


def test_development_left_at_its_default_fails(repo):
    report = run(repo, environment="development", environment_explicit=False)
    assert failed(report) == [Check.ENVIRONMENT]
    assert "explicitly" in report.failures[0].message


# ---------- environment_is_explicit against the real Settings class ----------

def test_environment_set_by_a_variable_equal_to_the_default_is_explicit(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert environment_is_explicit(settings) is True


def test_environment_set_in_an_env_file_equal_to_the_default_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("ENVIRONMENT=development\n", encoding="utf-8")
    settings = Settings(_env_file=str(env_file))
    assert settings.environment == "development"
    assert environment_is_explicit(settings) is True


def test_a_lowercase_key_in_the_env_file_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("environment=development\n", encoding="utf-8")
    assert environment_is_explicit(Settings(_env_file=str(env_file))) is True


def test_a_variable_and_an_env_file_together_are_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("ENVIRONMENT", "development")
    env_file = tmp_path / "test.env"
    env_file.write_text("ENVIRONMENT=development\n", encoding="utf-8")
    assert environment_is_explicit(Settings(_env_file=str(env_file))) is True


def test_a_non_default_value_is_explicit(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    settings = Settings(_env_file=None)
    assert settings.environment == "production"
    assert environment_is_explicit(settings) is True


def test_an_unset_environment_is_not_explicit_even_though_the_default_is_development(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert environment_is_explicit(settings) is False


def test_unrelated_env_file_entries_do_not_make_the_environment_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("MAX_UPLOAD_MB=5\n", encoding="utf-8")
    assert environment_is_explicit(Settings(_env_file=str(env_file))) is False


def test_environment_is_explicit_fails_closed_on_unusable_objects():
    assert environment_is_explicit(SimpleNamespace()) is False
    assert environment_is_explicit(SimpleNamespace(model_fields_set=None)) is False
    assert environment_is_explicit(SimpleNamespace(model_fields_set=["environment"])) is False
    assert environment_is_explicit(SimpleNamespace(model_fields_set={"environment"})) is True
    assert environment_is_explicit(SimpleNamespace(model_fields_set=frozenset({"other"}))) is False


# ---------- 5. database URL ----------

@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "127.0.0.1", "127.0.0.2", "::1"])
def test_loopback_hosts_pass(repo, host):
    assert run(repo, database_url=url(host)).ok


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "10.0.0.5", "192.168.1.10", "db.internal", "example.com", "localhost.example.com", "127.0.0.1.evil.com", "::2"],
)
def test_non_loopback_hosts_fail(repo, host):
    assert failed(run(repo, database_url=url(host))) == [Check.DATABASE_URL]


@pytest.mark.parametrize(
    "database_url",
    [
        f"postgresql://u:{SECRET}@/documind_db",
        f"postgresql://u:{SECRET}@localhost:5434,evil.example.com:5432/documind_db",
        f"postgresql://u:{SECRET}@/documind_db?host=/var/run/postgresql",
    ],
    ids=["no-host", "multi-host", "unix-socket-style"],
)
def test_an_empty_or_multiple_host_fails(repo, database_url):
    assert Check.DATABASE_URL in failed(run(repo, database_url=database_url))


@pytest.mark.parametrize(
    "query",
    ["?host=evil.example.com", "?hostaddr=10.0.0.5", "?service=prod", "?port=5432", "?sslmode=disable", "?options=-c%20search_path%3Dx"],
)
def test_any_query_parameter_fails(repo, query):
    report = run(repo, database_url=url(query=query))
    assert failed(report) == [Check.DATABASE_URL]
    assert "query" in report.failures[0].message


@pytest.mark.parametrize("driver", ["postgresql", "postgresql+psycopg2", "postgresql+psycopg"])
def test_postgresql_drivers_pass(repo, driver):
    assert run(repo, database_url=GOOD_URL.replace("postgresql", driver, 1)).ok


@pytest.mark.parametrize("scheme", ["sqlite", "mysql+pymysql", "mssql"])
def test_other_drivers_fail(repo, scheme):
    other = GOOD_URL.replace("postgresql", scheme, 1)
    assert Check.DATABASE_URL in failed(run(repo, database_url=other))


@pytest.mark.parametrize("name", ["PGHOST", "PGHOSTADDR", "PGSERVICE", "pghostaddr"])
def test_libpq_host_variables_fail_whatever_their_value(repo, name):
    for value in ("10.0.0.5", ""):
        report = run(repo, environ={name: value})
        assert failed(report) == [Check.DATABASE_URL]
        assert "10.0.0.5" not in report.failures[0].message


@pytest.mark.parametrize("name", ["PGPORT", "PGUSER", "PGDATABASE", "PATH", "HOME"])
def test_other_environment_variables_are_ignored(repo, name):
    assert run(repo, environ={name: "x"}).ok


def test_each_libpq_variable_is_named_but_never_its_value(repo):
    report = run(repo, environ={"PGHOST": "10.9.8.7", "PGSERVICE": "prod-secret"})
    message = report.failures[0].message
    assert "PGHOST" in message and "PGSERVICE" in message
    assert "10.9.8.7" not in message and "prod-secret" not in message


# ---------- 6. database name ----------

def test_a_matching_database_name_passes(repo):
    assert run(repo, expected_database="documind_db").ok


def test_a_different_database_name_fails_and_names_both(repo):
    report = run(repo, expected_database="documind_prod")
    assert failed(report) == [Check.DATABASE_NAME]
    assert "documind_db" in report.failures[0].message and "documind_prod" in report.failures[0].message


def test_the_database_name_comparison_is_case_sensitive(repo):
    assert failed(run(repo, expected_database="Documind_DB")) == [Check.DATABASE_NAME]


def test_an_empty_expected_database_fails(repo):
    assert failed(run(repo, expected_database="")) == [Check.DATABASE_NAME]


def test_a_url_without_a_database_fails_the_name_check(repo):
    no_database = f"postgresql://documind_user:{SECRET}@localhost:5434"
    assert failed(run(repo, database_url=no_database)) == [Check.DATABASE_NAME]


def test_no_expected_database_skips_only_the_name_check(repo):
    assert run(repo, expected_database=None).ok
    remote = run(repo, expected_database=None, database_url=url("db.example.com"))
    assert failed(remote) == [Check.DATABASE_URL]


# ---------- malformed URLs ----------

@pytest.mark.parametrize(
    "database_url",
    ["", "not a url", f"postgresql://u:{SECRET}@localhost:notaport/db", f"://{SECRET}", None],
    ids=["empty", "garbage", "bad-port-with-secret", "no-scheme-with-secret", "none"],
)
def test_a_malformed_url_fails_without_echoing_it(repo, database_url):
    report = run(repo, database_url=database_url)
    assert Check.DATABASE_URL in failed(report)
    assert report.target is None
    assert SECRET not in everything_printable(report)


def test_a_malformed_url_skips_the_name_check_but_still_runs_the_others(repo, tmp_path):
    report = run(repo, database_url="not a url", cwd=tmp_path)
    assert failed(report) == [Check.CURRENT_DIRECTORY, Check.DATABASE_URL]


# ---------- all failures, in order ----------

def test_every_static_failure_is_reported_together_in_check_order(repo, tmp_path):
    facts = FilesystemFacts(env_file_exists=False, upload_root_is_dir=False, expected_upload_root=None)
    report = run(
        repo,
        cwd=tmp_path,
        filesystem=facts,
        environment="production",
        database_url=url("db.example.com", query="?sslmode=disable"),
        expected_database="documind_prod",
        environ={"PGHOSTADDR": "10.0.0.5"},
    )
    order = list(Check)
    assert failed(report) == sorted(failed(report), key=order.index)
    assert set(failed(report)) == {
        Check.CURRENT_DIRECTORY,
        Check.ENV_FILE,
        Check.UPLOAD_DIRECTORY,
        Check.ENVIRONMENT,
        Check.DATABASE_URL,
        Check.DATABASE_NAME,
    }
    assert failed(report).count(Check.DATABASE_URL) == 3  # host, query, PGHOSTADDR


# ---------- secrets ----------

def test_the_password_never_appears_in_any_report_even_when_everything_fails(repo, tmp_path):
    facts = FilesystemFacts(env_file_exists=False, upload_root_is_dir=False, expected_upload_root=None)
    report = run(
        repo,
        cwd=tmp_path,
        filesystem=facts,
        environment="production",
        database_url=url("db.example.com", query="?sslmode=disable"),
        expected_database="documind_prod",
        environ={"PGSERVICE": "x"},
    )
    assert len(report.failures) >= 6
    assert SECRET not in everything_printable(report)


def test_the_password_never_appears_in_a_passing_report(repo):
    assert SECRET not in everything_printable(run(repo))


def test_target_info_has_no_password_field():
    assert "password" not in {name.lower() for name in TargetInfo.__slots__}


def test_a_url_without_a_port_gives_no_port(repo):
    report = run(repo, database_url=f"postgresql://u:{SECRET}@localhost/documind_db")
    assert report.ok
    assert report.target.port is None
    assert report.target.describe() == "localhost/documind_db as u"


# ---------- inspect_filesystem ----------

def test_inspect_finds_the_env_file_and_the_upload_directory_without_opening_anything(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / ".env").write_text("", encoding="utf-8")  # created by the test, empty
    (root / "uploads").mkdir()

    def refuse(*args, **kwargs):
        raise AssertionError("no file may be opened")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", refuse)
        patch.setattr(io, "open", refuse)
        patch.setattr(Path, "open", refuse)
        patch.setattr(Path, "read_text", refuse)
        patch.setattr(Path, "read_bytes", refuse)
        facts = inspect_filesystem(root, "uploads", root / "uploads")

    assert facts.env_file_exists is True
    assert facts.upload_root_is_dir is True
    assert facts.expected_upload_root == (root / "uploads").resolve()


def test_inspect_reports_a_missing_env_file_and_upload_directory(tmp_path):
    facts = inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads")
    assert facts.env_file_exists is False
    assert facts.upload_root_is_dir is False


def test_a_directory_named_env_is_not_an_env_file(tmp_path):
    (tmp_path / ".env").mkdir()
    assert inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads").env_file_exists is False


def test_an_upload_path_that_is_a_file_is_not_a_directory(tmp_path):
    (tmp_path / "uploads").write_bytes(b"x")
    assert inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads").upload_root_is_dir is False


def test_an_absolute_upload_setting_resolves_to_itself(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    facts = inspect_filesystem(tmp_path, str(other), other)
    assert facts.expected_upload_root == other.resolve()


def test_inspect_never_creates_anything(tmp_path):
    before = sorted(p.name for p in tmp_path.rglob("*"))
    inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads")
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("io"), ValueError("bad")])
def test_stat_errors_count_as_not_present(tmp_path, error):
    def boom(path):
        raise error

    facts = inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads", is_file=boom, is_dir=boom)
    assert facts.env_file_exists is False
    assert facts.upload_root_is_dir is False


@pytest.mark.parametrize("error", [OSError("io"), RuntimeError("symlink loop"), ValueError("bad")])
def test_an_unresolvable_upload_setting_gives_none(tmp_path, error):
    def boom(path):
        raise error

    facts = inspect_filesystem(tmp_path, "uploads", tmp_path / "uploads", resolve=boom)
    assert facts.expected_upload_root is None


# ---------- live check ----------

class ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class Recorder:
    """A stand-in connection that records statements and returns a fixed value."""

    def __init__(self, value):
        self.value = value
        self.statements = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return ScalarResult(self.value)


class Raiser:
    def __init__(self, error):
        self.error = error

    def execute(self, statement):
        raise self.error


def test_the_live_check_passes_against_the_real_test_database():
    with engine.connect() as connection:
        assert check_live_database(connection, engine.url.database) is None
        connection.rollback()


def test_the_live_check_fails_for_another_database_and_names_both():
    with engine.connect() as connection:
        failure = check_live_database(connection, "documind_db")
        connection.rollback()
    assert isinstance(failure, SafetyFailure)
    assert failure.check is Check.LIVE_DATABASE
    assert engine.url.database in failure.message and "documind_db" in failure.message


def test_the_live_check_runs_on_a_read_only_connection():
    with engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        assert check_live_database(connection, engine.url.database) is None
        connection.rollback()


def test_the_live_check_issues_exactly_one_select():
    recorder = Recorder("documind_db")
    assert check_live_database(recorder, "documind_db") is None
    assert recorder.statements == ["SELECT current_database()"]


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("SELECT 1", {}, Exception(f"postgresql://u:{SECRET}@10.0.0.5/db")),
        RuntimeError(f"could not connect with password {SECRET}"),
        ValueError(f"bad url postgresql://u:{SECRET}@localhost/db"),
        Exception(SECRET),
    ],
    ids=["sqlalchemy-operational", "runtime", "value", "plain"],
)
def test_a_live_failure_reports_only_the_exception_class(error):
    failure = check_live_database(Raiser(error), "documind_db")
    assert failure.check is Check.LIVE_DATABASE
    assert type(error).__name__ in failure.message
    assert SECRET not in failure.message
    for fragment in ("postgresql://", "SELECT", "10.0.0.5", "password"):
        assert fragment not in failure.message
    assert SECRET not in repr(failure)
    assert failure.__cause__ is None if hasattr(failure, "__cause__") else True


def test_a_non_string_live_result_is_a_failure():
    failure = check_live_database(Recorder(42), "documind_db")
    assert failure.check is Check.LIVE_DATABASE
    assert "unexpected" in failure.message


@pytest.mark.parametrize("expected", ["", None, 5])
def test_an_invalid_expected_name_fails_without_querying(expected):
    recorder = Recorder("documind_db")
    failure = check_live_database(recorder, expected)
    assert failure.check is Check.LIVE_DATABASE
    assert recorder.statements == []


def test_a_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        check_live_database(Raiser(KeyboardInterrupt()), "documind_db")


# ---------- the module stays isolated ----------

def test_the_module_does_not_expose_the_application_database_objects():
    public = {name for name in vars(backfill_safety) if not name.startswith("_")}
    assert {"check_static_safety", "check_live_database", "inspect_filesystem"} <= public
    for forbidden in ("SessionLocal", "engine", "settings", "storage", "Settings"):
        assert forbidden not in public