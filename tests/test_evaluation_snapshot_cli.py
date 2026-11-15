"""Tests for app.rag.snapshot_cli (E1.5.3). No database and no network: the engine is a fake.

A  argument parsing
B  take: refusals that happen before any connection
C  take: a successful snapshot over a fake read-only connection
D  take: failures
E  compare
F  output path containment (symbolic links, junctions, "..", letter case, re-check, exclusive creation)
G  evaluation identity digest
H  environment compatibility through the command line
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.rag import snapshot_cli as cli
from app.rag.evaluation import snapshot as sn

PASSWORD = "SECRETPASS-7731"
NOW = datetime(2026, 11, 12, 14, 0, 0, 5, tzinfo=timezone.utc)
STAMP = "2026-11-01T08:00:00.000001Z"
EVAL_EMAIL = "e1-eval@example.com"
CLI_DB = {"database_name": "documind_db", "server_version": "16.15", "server_encoding": "UTF8",
          "time_zone": "UTC", "extra_float_digits": "1", "pgvector_version": "0.8.7"}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------- snapshot data ----------

def user_row(user_id: int, email: str | None = None) -> dict:
    return {"id": user_id, "email_sha256": sha(email or f"user{user_id}@example.com"),
            "hashed_password_sha256": sha("pw"), "created_at": STAMP}


def doc_row(doc_id: int, owner_id: int = 1) -> dict:
    return {
        "id": doc_id, "filename": f"doc{doc_id}.txt", "description": None, "stored_name": None,
        "content_type": None, "size_bytes": None, "processing_status": None, "processing_error": None,
        "extracted_text_is_null": True, "extracted_text_chars": None, "extracted_text_bytes": None,
        "extracted_text_sha256": None, "owner_id": owner_id, "created_at": STAMP, "source_version": 1,
        "source_sha256": None, "indexing_status": None, "indexing_error": None, "indexing_attempt": 0,
        "indexing_lease_expires_at": None, "indexed_source_version": None, "indexed_signature": None,
        "indexed_embedding_id": None, "indexed_at": None,
    }


def table_rows(extra_user=False) -> dict:
    users = [user_row(1), user_row(2)]
    documents = [doc_row(10), doc_row(11, 2)]
    if extra_user:
        users.append(user_row(3, EVAL_EMAIL))
        documents += [doc_row(20 + i, 3) for i in range(6)]
    return {"users": users, "documents": documents, "document_chunks": []}


def snapshot(extra_user=False, created_at=NOW, **identity) -> dict:
    return sn.build_snapshot(table_rows(extra_user), created_at=created_at, database={**CLI_DB, **identity})


def write_snapshot(path: Path, snap: dict) -> Path:
    path.write_text(sn.snapshot_to_text(snap), encoding="utf-8", newline="\n")
    return path


# ---------- a fake engine ----------

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)


class _Scalar:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class FakeConnection:
    def __init__(self, engine):
        self.engine = engine
        self.options: dict = {}
        self.statements: list[str] = []
        self.rolled_back = False
        self.committed = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def execution_options(self, **options):
        self.options = options
        return self

    def execute(self, statement):                      # check_live_database
        return _Scalar(self.engine.live_name)

    def exec_driver_sql(self, sql):
        self.statements.append(sql)
        if sql == sn.SESSION_QUERY:
            return _Result([{"isolation": self.engine.isolation if self.options.get("isolation_level") else "read committed",
                             "read_only": "on" if self.options.get("postgresql_readonly") else "off"}])
        if sql == sn.IDENTITY_QUERY:
            return _Result([{"database_name": self.engine.live_name, "server_version": "16.15", "server_encoding": "UTF8",
                             "time_zone": "UTC", "extra_float_digits": "1", "pgvector_version": "0.8.7"}])
        if sql == sn.SCHEMA_QUERY:
            return _Result([
                {"table_name": t, "column_name": c,
                 "data_type": "timestamp with time zone" if c in sn.TIMESTAMP_COLUMNS[t] else "text"}
                for t, cols in sn.PERSISTED_COLUMNS.items() for c in cols
            ])
        if sql == sn.COUNT_QUERY:
            return _Result([{t: len(r) for t, r in self.engine.rows.items()}])
        for table, query in sn.TABLE_QUERIES.items():
            if sql == query:
                return _Result(self.engine.rows[table])
        raise AssertionError("unexpected statement")

    def rollback(self):
        self.rolled_back = True

    def commit(self):
        self.committed = True


class FakeEngine:
    def __init__(self, *, live_name="documind_db", rows=None, isolation="repeatable read", fail_connect=False):
        self.live_name, self.rows, self.isolation, self.fail_connect = live_name, rows or table_rows(), isolation, fail_connect
        self.connections: list[FakeConnection] = []

    def connect(self):
        if self.fail_connect:
            raise OSError(f"could not connect with password {PASSWORD}")
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection


@pytest.fixture()
def world(tmp_path):
    repo = (tmp_path / "repo").resolve()
    (repo / "uploads").mkdir(parents=True)
    (repo / ".env").write_text("placeholder", encoding="utf-8")
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    return repo, outside


def runtime_for(repo: Path, engine: FakeEngine, **override) -> cli.Runtime:
    values = dict(engine=engine, database_url=f"postgresql://user:{PASSWORD}@localhost:5434/documind_db",
                  environment="development", environment_explicit=True, upload_dir="uploads",
                  upload_root=(repo / "uploads").resolve())
    values.update(override)
    return cli.Runtime(**values)


def run(argv, repo: Path, *, engine=None, loader=None, clock=None, **override):
    out, err = io.StringIO(), io.StringIO()
    engine = engine or FakeEngine()
    code = cli.main(
        argv,
        runtime_loader=loader or (lambda: runtime_for(repo, engine, **override)),
        environ={}, cwd=repo, repo_root=repo, clock=clock or (lambda: NOW), stdout=out, stderr=err,
    )
    return code, out.getvalue(), err.getvalue(), engine


def never_loaded():
    raise AssertionError("the runtime must not be loaded")


# ---------- A. arguments ----------

def test_the_parser_requires_a_command_and_its_arguments():
    parser = cli.build_parser()
    assert parser.parse_args(["take", "--database", "d", "--out", "o"]).command == "take"
    args = parser.parse_args(["compare", "a.json", "b.json"])
    assert (args.users, args.documents, args.expect_evaluation, args.evaluation_email_sha256) == (None, None, False, None)
    for bad in ([], ["take"], ["take", "--database", "d"], ["take", "--out", "o"], ["compare", "a"],
                ["take", "--database", " ", "--out", "o"], ["compare", "a", "b", "--users", "-1"],
                ["compare", "a", "b", "--users", "x"], ["compare", "a", "b", "--evaluation-email-sha256", "zz"], ["other"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_usage_errors_return_exit_code_two(world):
    repo, _ = world
    code, out, err, _ = run(["take"], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED == 2 and out == ""


def test_the_email_digest_argument_is_normalised_to_lower_case():
    digest = sha(EVAL_EMAIL)
    args = cli.build_parser().parse_args(["compare", "a", "b", "--evaluation-email-sha256", digest.upper()])
    assert args.evaluation_email_sha256 == digest


# ---------- B. take: refusals before any connection ----------

@pytest.mark.parametrize("where", ["relative", "inside", "missing_folder", "exists", "dot_dot"])
def test_take_refuses_an_unsafe_output_path_before_loading_anything(world, where):
    repo, outside = world
    if where == "relative":
        target = "snapshot.json"
    elif where == "inside":
        target = str(repo / "snapshot.json")
    elif where == "dot_dot":
        target = str(outside / ".." / "repo" / "snapshot.json")
    elif where == "missing_folder":
        target = str(outside / "no-such-folder" / "snapshot.json")
    else:
        existing = outside / "snapshot.json"
        existing.write_text("keep", encoding="utf-8")
        target = str(existing)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", target], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == "" and err.startswith("REFUSED")
    if where == "exists":
        assert existing.read_text(encoding="utf-8") == "keep"


def test_take_refuses_when_the_configuration_cannot_be_loaded(world):
    repo, outside = world

    def broken():
        raise RuntimeError(f"cannot load {PASSWORD}")
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(outside / "s.json")], repo, loader=broken)
    assert code == cli.EXIT_REFUSED and "RuntimeError" in err and PASSWORD not in err and not (outside / "s.json").exists()


@pytest.mark.parametrize("override,database,needle", [
    ({"environment": "production"}, "documind_db", "environment"),
    ({"environment_explicit": False}, "documind_db", "explicit"),
    ({"database_url": f"postgresql://u:{PASSWORD}@db.example.com:5434/documind_db"}, "documind_db", "loopback"),
    ({}, "other_db", "other_db"),
])
def test_take_refuses_when_a_static_safety_check_fails_and_never_connects(world, override, database, needle):
    repo, outside = world
    engine = FakeEngine()
    code, out, err, _ = run(["take", "--database", database, "--out", str(outside / "s.json")], repo, engine=engine, **override)
    assert code == cli.EXIT_REFUSED and needle in err and PASSWORD not in err + out
    assert engine.connections == [] and not (outside / "s.json").exists()


def test_take_refuses_when_the_live_database_is_not_the_expected_one(world):
    repo, outside = world
    engine = FakeEngine(live_name="documind_test")
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(outside / "s.json")], repo, engine=engine)
    assert code == cli.EXIT_REFUSED and "documind_test" in err and not (outside / "s.json").exists()
    connection = engine.connections[0]
    assert connection.statements == [] and connection.rolled_back and connection.closed and not connection.committed


# ---------- C. take: success ----------

def test_take_writes_one_verified_snapshot_over_a_read_only_repeatable_read_connection(world):
    repo, outside = world
    target = outside / "s0.json"
    code, out, err, engine = run(["take", "--database", "documind_db", "--out", str(target)], repo)
    assert code == cli.EXIT_OK and err == ""
    written = target.read_text(encoding="utf-8")
    snap = sn.load_snapshot(written)
    assert snap["created_at_utc"] == "2026-11-12T14:00:00.000005Z"
    assert snap["fingerprint"] == snapshot(created_at=NOW)["fingerprint"]
    assert f"content fingerprint: {snap['fingerprint']}" in out
    assert f"file sha256: {hashlib.sha256(written.encode('utf-8')).hexdigest()}" in out
    assert f"file bytes: {len(written.encode('utf-8'))}" in out
    for table in sn.TABLES:
        assert f"{table}: {len(snap['tables'][table]['rows'])} rows" in out
    connection = engine.connections[0]
    assert connection.options == {"isolation_level": "REPEATABLE READ", "postgresql_readonly": True}
    assert connection.rolled_back and connection.closed and not connection.committed
    assert connection.statements and all(s.lstrip().upper().startswith("SELECT") for s in connection.statements)
    assert PASSWORD not in out + err and "@example.com" not in out + written
    assert written.endswith("\n") and "\r" not in written


def test_two_snapshots_of_unchanged_data_have_the_same_fingerprint_and_different_times(world):
    repo, outside = world
    first, second = outside / "a.json", outside / "b.json"
    run(["take", "--database", "documind_db", "--out", str(first)], repo)
    out, err = io.StringIO(), io.StringIO()
    later = datetime(2026, 11, 13, tzinfo=timezone.utc)
    cli.main(["take", "--database", "documind_db", "--out", str(second)],
             runtime_loader=lambda: runtime_for(repo, FakeEngine()), environ={}, cwd=repo, repo_root=repo,
             clock=lambda: later, stdout=out, stderr=err)
    a, b = (sn.load_snapshot(p.read_text(encoding="utf-8")) for p in (first, second))
    assert a["created_at_utc"] != b["created_at_utc"] and a["fingerprint"] == b["fingerprint"]
    code, report, _, _ = run(["compare", str(first), str(second)], repo, loader=never_loaded)
    assert code == cli.EXIT_OK and "identical: yes" in report


def test_take_never_overwrites_an_existing_snapshot(world):
    repo, outside = world
    target = outside / "s0.json"
    assert run(["take", "--database", "documind_db", "--out", str(target)], repo)[0] == cli.EXIT_OK
    before = target.read_bytes()
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(target)], repo)
    assert code == cli.EXIT_REFUSED and target.read_bytes() == before


# ---------- D. take: failures ----------

def test_take_fails_and_writes_nothing_when_the_session_is_not_repeatable_read(world):
    repo, outside = world
    engine = FakeEngine(isolation="read committed")
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(outside / "s.json")], repo, engine=engine)
    assert code == cli.EXIT_FAILURE and "snapshot_session" in err and out == "" and not (outside / "s.json").exists()
    assert engine.connections[0].rolled_back and engine.connections[0].closed


def test_take_reports_a_connection_failure_without_its_message(world):
    repo, outside = world
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(outside / "s.json")], repo,
                            engine=FakeEngine(fail_connect=True))
    assert code == cli.EXIT_FAILURE and "OSError" in err and PASSWORD not in err and not (outside / "s.json").exists()


def test_take_reports_an_unexpected_read_error_by_class_name_only(world):
    repo, outside = world

    class Exploding(FakeEngine):
        def connect(self):
            connection = super().connect()
            original = connection.exec_driver_sql

            def boom(sql):
                if sql in sn.TABLE_QUERIES.values():
                    raise ValueError(f"boom {PASSWORD}")
                return original(sql)
            connection.exec_driver_sql = boom
            return connection
    code, out, err, engine = run(["take", "--database", "documind_db", "--out", str(outside / "s.json")], repo, engine=Exploding())
    assert code == cli.EXIT_FAILURE and "ValueError" in err and PASSWORD not in err + out
    assert not (outside / "s.json").exists() and engine.connections[0].rolled_back


# ---------- E. compare ----------

def test_compare_of_identical_snapshots_exits_zero_and_needs_no_runtime(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot(created_at=NOW))
    b = write_snapshot(outside / "b.json", snapshot(created_at=datetime(2026, 12, 1, tzinfo=timezone.utc)))
    code, out, err, _ = run(["compare", str(a), str(b)], repo, loader=never_loaded)
    assert code == cli.EXIT_OK and "identical: yes" in out and "findings: none" in out and err == ""


def test_compare_of_different_snapshots_exits_one_and_reports_the_difference(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    b = write_snapshot(outside / "b.json", snapshot(extra_user=True))
    code, out, _, _ = run(["compare", str(a), str(b)], repo, loader=never_loaded)
    assert code == cli.EXIT_FAILURE and "identical: no" in out and "added (1): 3" in out and "added (6): 20, 21, 22, 23, 24, 25" in out


def test_compare_with_expect_evaluation_passes_for_a_correct_ingestion(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    b = write_snapshot(outside / "b.json", snapshot(extra_user=True))
    code, out, err, _ = run(["compare", str(a), str(b), "--expect-evaluation",
                             "--evaluation-email-sha256", sha(EVAL_EMAIL)], repo, loader=never_loaded)
    assert code == cli.EXIT_OK and "findings: none" in out and "pre-existing rows unchanged: yes" in out


def test_compare_with_expect_evaluation_fails_for_each_violation(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    good = write_snapshot(outside / "good.json", snapshot(extra_user=True))
    digest = ["--evaluation-email-sha256", sha(EVAL_EMAIL)]
    cases = {
        "no change": (a, ["--expect-evaluation", *digest], "delta_user_count"),
        "wrong counts": (good, ["--expect-evaluation", "--documents", "5", *digest], "delta_document_count"),
        "wrong user": (good, ["--expect-evaluation", "--evaluation-email-sha256", sha("other@example.com")], "delta_unexpected_user"),
    }
    for name, (after, flags, needle) in cases.items():
        code, out, _, _ = run(["compare", str(a), str(after), *flags], repo, loader=never_loaded)
        assert code == cli.EXIT_FAILURE and needle in out, name

    modified = snapshot(extra_user=True)
    modified["tables"]["documents"]["rows"][0]["filename"] = "changed.txt"
    modified["tables"]["documents"]["fingerprint"] = sn._section_fingerprint(
        "documents", modified["tables"]["documents"]["columns"], modified["tables"]["documents"]["rows"])
    modified["fingerprint"] = sn._combined_fingerprint({t: modified["tables"][t]["fingerprint"] for t in sn.TABLES})
    bad = write_snapshot(outside / "bad.json", modified)
    code, out, _, _ = run(["compare", str(a), str(bad), "--expect-evaluation", *digest], repo, loader=never_loaded)
    assert code == cli.EXIT_FAILURE and "delta_preexisting_modified at documents[10].filename" in out
    assert "changed.txt" not in out


def test_compare_refuses_missing_corrupt_and_tampered_files(world):
    repo, outside = world
    good = write_snapshot(outside / "good.json", snapshot())
    code, out, err, _ = run(["compare", str(good), str(outside / "missing.json")], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == "" and "could not be read" in err

    corrupt = outside / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    code, out, err, _ = run(["compare", str(good), str(corrupt)], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and "not a valid snapshot" in err and "{not json" not in err

    tampered = json.loads(good.read_text(encoding="utf-8"))
    tampered["tables"]["users"]["rows"][0]["created_at"] = "2020-01-01T00:00:00.000000Z"
    path = outside / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    code, out, err, _ = run(["compare", str(good), str(path)], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and "snapshot_fingerprint" in err

    binary = outside / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    assert run(["compare", str(good), str(binary)], repo, loader=never_loaded)[0] == cli.EXIT_REFUSED


def test_the_compare_report_contains_no_email_or_text(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    b = write_snapshot(outside / "b.json", snapshot(extra_user=True))
    _, out, err, _ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", sha(EVAL_EMAIL)],
                         repo, loader=never_loaded)
    assert "@example.com" not in out + err and EVAL_EMAIL not in out + err


# ---------- F. output path containment ----------

def make_link(link: Path, target: Path) -> None:
    """A symbolic link to a folder (or a not-yet-existing target); the test is skipped where that is not allowed."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available in this environment")


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def refused_outside(code, out, err):
    return code == cli.EXIT_REFUSED and out == "" and "outside the repository" in err


@pytest.mark.parametrize("target_name", ["uploads", ""])
def test_an_output_folder_that_is_a_symlink_into_the_repository_is_refused(world, target_name):
    repo, outside = world
    link = outside / "link"
    make_link(link, repo / target_name if target_name else repo)
    before = tree(repo)
    code, out, err, engine = run(["take", "--database", "documind_db", "--out", str(link / "s.json")], repo, loader=never_loaded)
    assert refused_outside(code, out, err)
    assert tree(repo) == before and not (repo / target_name / "s.json").exists()


def test_a_symlink_into_the_repository_is_refused_for_a_folder_that_does_not_exist_yet(world):
    repo, outside = world
    link = outside / "link"
    make_link(link, repo)
    before = tree(repo)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(link / "new" / "deeper" / "s.json")],
                            repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == "" and err.startswith("REFUSED")      # refused (not created)
    assert tree(repo) == before


def test_the_repository_itself_reached_through_a_symlink_is_still_the_repository(world):
    repo, outside = world
    alias = outside / "alias"
    make_link(alias, repo)
    before = tree(repo)
    # the configured repository root is the alias, the output is spelled with the real path
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(repo / "s.json")], alias, loader=never_loaded)
    assert refused_outside(code, out, err) and tree(repo) == before
    # and the other way round: root spelled really, output spelled through the alias
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(alias / "s.json")], repo, loader=never_loaded)
    assert refused_outside(code, out, err) and tree(repo) == before


def test_a_symlinked_folder_that_resolves_outside_the_repository_is_allowed_and_written_at_its_real_place(world):
    repo, outside = world
    real = outside / "real"
    real.mkdir()
    link = outside / "link"
    make_link(link, real)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(link / "s.json")], repo)
    assert code == cli.EXIT_OK and err == ""
    assert (real / "s.json").is_file() and sn.load_snapshot((real / "s.json").read_text(encoding="utf-8"))
    assert str(real / "s.json") in out


def test_an_output_that_is_a_dangling_symlink_is_refused_and_never_followed(world):
    repo, outside = world
    into_repo, elsewhere = outside / "into_repo.json", outside / "elsewhere.json"
    planted, nowhere = repo / "planted.json", outside / "nowhere.json"          # neither exists
    try:
        os.symlink(planted, into_repo)
        os.symlink(nowhere, elsewhere)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available in this environment")
    before = tree(repo)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(into_repo)], repo, loader=never_loaded)
    assert refused_outside(code, out, err) and not planted.exists() and tree(repo) == before      # resolves into the repository
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(elsewhere)], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and "already exists" in err and not nowhere.exists()          # lexists, not exists


def test_dot_dot_segments_cannot_reach_the_repository(world):
    repo, outside = world
    before = tree(repo)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(outside / ".." / "repo" / "uploads" / "s.json")],
                            repo, loader=never_loaded)
    assert refused_outside(code, out, err) and tree(repo) == before


def test_inside_compares_resolved_paths_and_a_sibling_with_the_same_prefix_is_outside(world):
    repo, outside = world
    sibling = repo.parent / (repo.name + "2")
    sibling.mkdir()
    assert cli._inside(repo, repo) and cli._inside(repo / "uploads" / "x.json", repo)
    assert cli._inside(repo / "not" / "yet" / "there.json", repo)                  # a path that does not exist
    assert not cli._inside(sibling / "x.json", repo) and not cli._inside(outside / "x.json", repo)


def test_inside_fails_closed_when_a_path_cannot_be_resolved(world, monkeypatch):
    repo, outside = world
    monkeypatch.setattr(cli, "_resolve", lambda path: None)
    assert cli._inside(outside / "x.json", repo) is True


def test_inside_applies_the_case_rules_of_the_operating_system(world, monkeypatch):
    repo, _ = world
    shouted = Path(str(repo).upper()) / "S.JSON"
    if sys.platform == "linux":
        assert not cli._inside(shouted, repo)                          # a case-sensitive file system: another folder
    monkeypatch.setattr(os.path, "normcase", lambda value: value.lower())   # the case rule of ntpath.normcase
    assert cli._inside(shouted, repo)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(shouted)], repo, loader=never_loaded)
    assert refused_outside(code, out, err)


def test_the_extended_length_prefix_does_not_hide_the_repository():
    plain = os.path.normcase(r"C:\work\repo\s.json")
    assert cli._comparable(Path(r"\\?\C:\work\repo\s.json")) == plain
    assert cli._comparable(Path(r"\\?\UNC\server\share\s.json")) == os.path.normcase(r"\\server\share\s.json")


def test_the_folder_is_checked_again_immediately_before_the_file_is_created(world):
    repo, outside = world
    folder = outside / "d"
    folder.mkdir()
    swapped = {}

    def clock():                              # runs after the first check, while the database is read
        folder.rmdir()
        make_link(folder, repo / "uploads")
        swapped["done"] = True
        return NOW

    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(folder / "s.json")], repo, clock=clock)
    assert swapped and code == cli.EXIT_REFUSED and "outside the repository" in err and "snapshot written" not in out
    assert list((repo / "uploads").iterdir()) == []


def test_a_file_that_appears_while_the_database_is_read_is_not_overwritten(world):
    repo, outside = world
    target = outside / "s.json"

    def clock():
        target.write_text("planted", encoding="utf-8")
        return NOW

    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(target)], repo, clock=clock)
    assert code == cli.EXIT_REFUSED and "already exists" in err and target.read_text(encoding="utf-8") == "planted"


def test_exclusive_creation_still_refuses_when_the_second_check_is_bypassed(world, monkeypatch):
    repo, outside = world
    target = outside / "s.json"
    monkeypatch.setattr(cli, "_check_output_path", lambda raw, root, err: Path(raw))     # no path check at all
    target.write_text("keep", encoding="utf-8")
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(target)], repo)
    assert code == cli.EXIT_FAILURE and "FileExistsError" in err and target.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions exist only on Windows")
def test_windows_a_junction_into_the_repository_is_refused_whatever_the_letter_case(world):
    repo, outside = world
    junction = outside / "junction"
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(repo / "uploads")], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip("could not create a junction here")
    before = tree(repo)
    for spelled in (junction / "s.json", Path(str(junction).upper()) / "S.JSON"):
        code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(spelled)], repo, loader=never_loaded)
        assert refused_outside(code, out, err)
    code, out, err, _ = run(["take", "--database", "documind_db", "--out", str(Path(str(repo).upper()) / "S.JSON")],
                            repo, loader=never_loaded)
    assert refused_outside(code, out, err)
    assert tree(repo) == before


# ---------- G. evaluation identity digest ----------

def ingestion_pair(outside: Path):
    return (write_snapshot(outside / "a.json", snapshot()), write_snapshot(outside / "b.json", snapshot(extra_user=True)))


def test_expect_evaluation_without_a_digest_is_refused_before_anything_is_read(world):
    repo, outside = world
    code, out, err, _ = run(["compare", str(outside / "no-a.json"), str(outside / "no-b.json"), "--expect-evaluation"],
                            repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == ""
    assert "--expect-evaluation requires --evaluation-email-sha256" in err and "could not be read" not in err


def test_expect_evaluation_with_the_correct_digest_passes(world):
    repo, outside = world
    a, b = ingestion_pair(outside)
    code, out, err, _ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", sha(EVAL_EMAIL)],
                            repo, loader=never_loaded)
    assert code == cli.EXIT_OK and "findings: none" in out and err == ""


def test_the_digest_is_accepted_in_upper_case(world):
    repo, outside = world
    a, b = ingestion_pair(outside)
    code, *_ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", sha(EVAL_EMAIL).upper()],
                   repo, loader=never_loaded)
    assert code == cli.EXIT_OK


def test_expect_evaluation_with_a_wrong_digest_fails_and_reports_only_the_user_id(world):
    repo, outside = world
    a, b = ingestion_pair(outside)
    wrong = sha("arbitrary-new-user@example.com")
    code, out, err, _ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", wrong],
                            repo, loader=never_loaded)
    assert code == cli.EXIT_FAILURE and "delta_unexpected_user at users[3]" in out
    shown = out + err
    assert wrong not in shown and sha(EVAL_EMAIL) not in shown and "example.com" not in shown


def test_an_arbitrary_new_user_is_not_accepted_by_a_digest_that_does_not_match_it(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    other = snapshot(extra_user=True)
    users = other["tables"]["users"]
    users["rows"][-1]["email_sha256"] = sha("someone-else@example.com")
    users["fingerprint"] = sn._section_fingerprint("users", users["columns"], users["rows"])
    other["fingerprint"] = sn._combined_fingerprint({t: other["tables"][t]["fingerprint"] for t in sn.TABLES})
    b = write_snapshot(outside / "b.json", other)
    code, out, _, _ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", sha(EVAL_EMAIL)],
                          repo, loader=never_loaded)
    assert code == cli.EXIT_FAILURE and "delta_unexpected_user" in out


@pytest.mark.parametrize("bad", ["", "abc", "g" * 64, "a" * 63, "a" * 65])
def test_a_malformed_digest_is_a_usage_error(world, bad):
    repo, outside = world
    a, b = ingestion_pair(outside)
    code, out, *_ = run(["compare", str(a), str(b), "--expect-evaluation", "--evaluation-email-sha256", bad],
                        repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == ""


@pytest.mark.parametrize("flags", [["--users", "1"], ["--documents", "6"], ["--evaluation-email-sha256", "a" * 64]])
def test_the_evaluation_options_are_refused_without_expect_evaluation(world, flags):
    repo, outside = world
    a, b = ingestion_pair(outside)
    code, out, err, _ = run(["compare", str(a), str(b), *flags], repo, loader=never_loaded)
    assert code == cli.EXIT_REFUSED and out == "" and "only valid with --expect-evaluation" in err


def test_generic_comparison_without_expect_evaluation_is_unchanged(world):
    repo, outside = world
    a, b = ingestion_pair(outside)
    same = write_snapshot(outside / "same.json", snapshot(created_at=datetime(2027, 1, 1, tzinfo=timezone.utc)))
    assert run(["compare", str(a), str(same)], repo, loader=never_loaded)[0] == cli.EXIT_OK
    code, out, _, _ = run(["compare", str(a), str(b)], repo, loader=never_loaded)
    assert code == cli.EXIT_FAILURE and "identical: no" in out and "delta_" not in out


def test_expected_counts_can_be_overridden_together_with_the_digest(world):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    same = write_snapshot(outside / "same.json", snapshot())
    code, *_ = run(["compare", str(a), str(same), "--expect-evaluation", "--users", "0", "--documents", "0",
                    "--evaluation-email-sha256", sha(EVAL_EMAIL)], repo, loader=never_loaded)
    assert code == cli.EXIT_OK


def test_the_raw_evaluation_email_is_not_in_the_tool_sources():
    sources = [Path(cli.__file__), Path(sn.__file__), Path(cli.__file__).parents[2] / "scripts" / "e15_snapshot.py"]
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "e1-eval" not in text and "@example.com" not in text, source.name


# ---------- H. environment compatibility through the command line ----------

def test_a_restored_database_snapshot_is_comparable_with_the_original(world):
    repo, outside = world
    original = write_snapshot(outside / "s0.json", snapshot(database_name="documind_db"))
    restored = write_snapshot(outside / "r.json", snapshot(created_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
                                                           database_name="documind_restore_check", time_zone="Asia/Kolkata",
                                                           server_version="16.4"))
    code, out, err, _ = run(["compare", str(original), str(restored)], repo, loader=never_loaded)
    assert code == cli.EXIT_OK and "identical: yes" in out and err == ""


@pytest.mark.parametrize("key,value", [("pgvector_version", "0.8.6"), ("extra_float_digits", "3"),
                                       ("server_version", "17.2"), ("server_encoding", "LATIN1")])
def test_snapshots_from_incompatible_environments_are_refused_by_compare(world, key, value):
    repo, outside = world
    a = write_snapshot(outside / "a.json", snapshot())
    b = write_snapshot(outside / "b.json", snapshot(**{key: value}))
    for extra in ([], ["--expect-evaluation", "--users", "0", "--documents", "0", "--evaluation-email-sha256", sha(EVAL_EMAIL)]):
        code, out, err, _ = run(["compare", str(a), str(b), *extra], repo, loader=never_loaded)
        assert code == cli.EXIT_REFUSED and out == ""
        assert f"compare_environment at database.{key}" in err and value not in err
