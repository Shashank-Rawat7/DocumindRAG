"""E2.4c: the baseline command line, its external freeze anchor and its preflight gates (no run exists yet)."""
from __future__ import annotations

import ast
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag import baseline_cli as bc
from app.rag.evaluation.freeze_binding import FreezeBinding
from app.rag.evaluation.schema import EvaluationDataError, Issue

REPO = Path(__file__).resolve().parents[1]
RECORD_SHA = "ab" * 32
OTHER_SHA = "cd" * 32
EMAIL_SHA = "ef" * 32
HEAD = "1" * 40


def binding(record_sha: str = RECORD_SHA) -> FreezeBinding:
    return FreezeBinding(
        dataset_version="eval-dataset-v1.0.0", frozen_on="2026-11-14", corpus_version="c1", record_sha256=record_sha,
        dev_file_sha256="1" * 64, dataset_json_sha256="2" * 64, manifest_file_sha256="3" * 64,
        dev_label_binding_sha256="4" * 64, canonical_manifest_sha256="5" * 64,
    )


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.outside = tmp_path / "outside"
        self.outside.mkdir()
        self.out_path = self.outside / "baseline.json"
        self.calls: list[str] = []
        self.executed: list[bc.Preflight] = []

    def argv(self, **override: object) -> list[str]:
        values: dict[str, object] = {
            "--database": "documind_db", "--owner-id": "7", "--evaluation-email-sha256": EMAIL_SHA,
            "--expected-freeze-record-sha256": RECORD_SHA, "--out": str(self.out_path),
        }
        values.update({key if key.startswith("--") else f"--{key}": value for key, value in override.items()})
        result: list[str] = []
        for key, value in values.items():
            if value is not None:
                result += [key, str(value)]
        return result

    def freeze_loader(self, sha: str = RECORD_SHA):
        def load(root: Path) -> FreezeBinding:
            self.calls.append("freeze")
            assert root == self.repo
            return binding(sha)

        return load

    def git_reader(self, state: str = bc.GIT_CLEAN, commit: str | None = HEAD):
        def read(root: Path) -> tuple[str | None, str]:
            self.calls.append("git")
            assert root == self.repo
            return commit, state

        return read

    def executor(self, code: int = 0):
        def run(preflight: bc.Preflight, out, err) -> int:
            self.calls.append("execute")
            self.executed.append(preflight)
            return code

        return run

    def main(self, argv=None, *, freeze=None, git=None, executor=None):
        out, err = io.StringIO(), io.StringIO()
        code = bc.main(
            self.argv() if argv is None else argv,
            freeze_loader=freeze or self.freeze_loader(), git_reader=git or self.git_reader(),
            executor=executor or self.executor(), repo_root=self.repo, stdout=out, stderr=err,
        )
        return code, out.getvalue(), err.getvalue()


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


# ---------- argument contract ----------

def test_the_expected_sha_is_mandatory_and_nothing_is_called_without_it(
        h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = h.main(h.argv(**{"expected-freeze-record-sha256": None}))
    assert code == bc.EXIT_REFUSED
    assert "--expected-freeze-record-sha256" in capsys.readouterr().err  # argparse writes to the real stderr
    assert h.calls == []


@pytest.mark.parametrize("bad", [
    RECORD_SHA.upper(), RECORD_SHA[:-1], RECORD_SHA + "a", " " + RECORD_SHA, RECORD_SHA + " ", RECORD_SHA + "\n",
    "g" * 64, "", "0x" + "a" * 62, "ab" * 31 + "aB",
])
def test_a_malformed_expected_sha_is_refused_before_any_call(h: Harness, bad: str) -> None:
    code, out, err = h.main(h.argv(**{"expected-freeze-record-sha256": bad}))
    assert code == bc.EXIT_REFUSED
    assert h.calls == []
    assert out == ""
    assert not h.out_path.exists()


def test_the_strict_type_returns_the_value_unchanged() -> None:
    assert bc._strict_sha256(RECORD_SHA) == RECORD_SHA


def test_the_parser_has_no_dataset_path_options(h: Harness) -> None:
    for option in ("--dev-file", "--corpus-dir"):
        code, _, _ = h.main(h.argv(**{option: "x"}))
        assert code == bc.EXIT_REFUSED
    assert h.calls == []


def test_abbreviations_are_not_accepted(h: Harness) -> None:
    code, _, _ = h.main([a if a != "--expected-freeze-record-sha256" else "--expected" for a in h.argv()])
    assert code == bc.EXIT_REFUSED
    assert h.calls == []


def test_optional_arguments_parse_and_default_to_none() -> None:
    base = ["--database", "d", "--owner-id", "1", "--evaluation-email-sha256", EMAIL_SHA,
            "--expected-freeze-record-sha256", RECORD_SHA, "--out", "x"]
    args = bc.parse_arguments(base)
    assert (args.bootstrap_seed, args.resamples, args.random_seeds, args.adjudicate) == (None, None, None, [])
    args = bc.parse_arguments(base + ["--adjudicate", "q1:u1", "--resamples", "5", "--random-seeds", "1000",
                                      "--bootstrap-seed", "3"])
    assert (args.bootstrap_seed, args.resamples, args.random_seeds, args.adjudicate) == (3, 5, 1000, [("q1", "u1")])


# ---------- the trust anchor ----------

def test_a_mismatched_anchor_refuses_with_no_further_call_and_no_hash_printed(h: Harness) -> None:
    code, out, err = h.main(freeze=h.freeze_loader(OTHER_SHA))
    assert code == bc.EXIT_REFUSED
    assert h.calls == ["freeze"]
    assert RECORD_SHA not in out + err and OTHER_SHA not in out + err
    assert "does not match" in err
    assert not h.out_path.exists()


def test_a_correct_anchor_reaches_the_executor_exactly_once(h: Harness) -> None:
    code, _, _ = h.main(executor=h.executor(0))
    assert code == bc.EXIT_OK
    assert h.calls == ["freeze", "git", "execute"]
    (pre,) = h.executed
    assert pre.binding.record_sha256 == RECORD_SHA and pre.head_commit == HEAD
    assert pre.dev_path.name == "dev.jsonl" and pre.report_path.name == "baseline.json"


def test_the_executor_exit_code_is_propagated(h: Harness) -> None:
    assert h.main(executor=h.executor(1))[0] == bc.EXIT_FAILURE


def test_the_default_executor_is_the_real_run_and_a_refused_configuration_writes_nothing(h: Harness) -> None:
    out, err = io.StringIO(), io.StringIO()

    def refusing_runtime() -> object:
        h.calls.append("runtime")
        raise RuntimeError("secret detail")

    code = bc.main(h.argv(), freeze_loader=h.freeze_loader(), git_reader=h.git_reader(), repo_root=h.repo,
                   runtime_loader=refusing_runtime, stdout=out, stderr=err)
    assert code == bc.EXIT_REFUSED
    assert h.calls == ["freeze", "git", "runtime"]  # the runtime is first touched after every preflight gate
    assert "application configuration" in err.getvalue() and "secret detail" not in err.getvalue()
    assert not h.out_path.exists() and list(h.outside.iterdir()) == []


@pytest.mark.parametrize("value", [None, 5, b"x", "é" * 64])
def test_a_non_ascii_or_non_string_record_hash_is_a_mismatch(value: object) -> None:
    class Fake:
        record_sha256 = value

    assert bc._anchor_matches(Fake(), RECORD_SHA) is False  # type: ignore[arg-type]


def test_the_comparison_uses_compare_digest(monkeypatch: pytest.MonkeyPatch, h: Harness) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = bc.hmac.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(bc.hmac, "compare_digest", spy)
    assert h.main()[0] == bc.EXIT_OK
    assert seen == [(RECORD_SHA.encode(), RECORD_SHA.encode())]


FREEZE_REFUSAL = "REFUSED: the frozen dataset record could not be verified. Nothing else was read.\n"


def test_an_unverifiable_freeze_record_prints_only_the_fixed_refusal(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    out, err = io.StringIO(), io.StringIO()
    argv = ["--database", "d", "--owner-id", "1", "--evaluation-email-sha256", EMAIL_SHA,
            "--expected-freeze-record-sha256", RECORD_SHA, "--out", str(tmp_path / "r.json")]
    code = bc.main(argv, git_reader=lambda r: pytest.fail("git must not be read"), repo_root=root, stdout=out,
                   stderr=err)
    assert code == bc.EXIT_REFUSED
    assert err.getvalue() == FREEZE_REFUSAL  # the real loader failed (empty repository): nothing but the message
    assert out.getvalue() == ""


def test_freeze_failure_output_never_carries_issue_details(h: Harness) -> None:
    secret_path = "eval/dataset/secret-name.jsonl"
    issues = [Issue("FREEZE_SECRET_CODE", secret_path, f"leaks {RECORD_SHA} and {OTHER_SHA}"),
              Issue("FREEZE_OTHER", "elsewhere", "second")]

    def failing(root: Path) -> FreezeBinding:
        h.calls.append("freeze")
        raise EvaluationDataError(issues)

    code, out, err = h.main(freeze=failing)
    assert code == bc.EXIT_REFUSED
    assert err == FREEZE_REFUSAL and out == ""
    for needle in ("FREEZE_", secret_path, "elsewhere", RECORD_SHA, OTHER_SHA, "EvaluationDataError", "leaks"):
        assert needle not in err
    assert h.calls == ["freeze"]


@pytest.mark.parametrize("failure", [
    OSError(13, "permission denied for C:\\secret\\freeze_record.json"),
    RuntimeError("boom " + RECORD_SHA),
    ValueError("x"),
    EvaluationDataError([Issue("FREEZE_X", "somewhere", "m")]),
])
def test_every_freeze_loader_failure_prints_exactly_the_fixed_message(h: Harness, failure: Exception) -> None:
    def failing(root: Path) -> FreezeBinding:
        h.calls.append("freeze")
        raise failure

    code, out, err = h.main(freeze=failing)
    assert code == bc.EXIT_REFUSED
    assert err == FREEZE_REFUSAL and out == ""
    assert h.calls == ["freeze"]  # nothing after the failed load: no git, no executor


def test_an_interrupt_during_the_freeze_load_is_not_swallowed(h: Harness) -> None:
    def interrupted(root: Path) -> FreezeBinding:
        raise KeyboardInterrupt

    code, out, err = h.main(freeze=interrupted)
    assert code == 130
    assert "FREEZE" not in err and "could not be verified" not in err


def test_the_gate_order_is_anchor_then_output_then_dev_then_git(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.rag.pilot_cli as pilot  # noqa: PLC0415 - imported lazily on purpose

    real_report, real_dev = pilot._check_report_path, bc.check_dev_path

    def report(*a, **k):
        h.calls.append("report")
        return real_report(*a, **k)

    def dev(*a, **k):
        h.calls.append("dev")
        return real_dev(*a, **k)

    monkeypatch.setattr(pilot, "_check_report_path", report)
    monkeypatch.setattr(bc, "check_dev_path", dev)
    assert h.main()[0] == bc.EXIT_OK
    assert h.calls == ["freeze", "report", "dev", "git", "execute"]


# ---------- output path and git gate ----------

def test_output_path_refusals_stop_before_git_and_executor(h: Harness, tmp_path: Path) -> None:
    h.out_path.write_text("x", encoding="utf-8")
    cases = {
        "relative": "report.json",
        "inside the repository": str(h.repo / "r.json"),
        "existing": str(h.out_path),
        "missing folder": str(tmp_path / "nowhere" / "r.json"),
    }
    for name, path in cases.items():
        h.calls.clear()
        code, _, err = h.main(h.argv(out=path))
        assert code == bc.EXIT_REFUSED, name
        assert h.calls == ["freeze"], name
        assert "REFUSED" in err


@pytest.mark.parametrize("state", [bc.GIT_STAGED, bc.GIT_UNSTAGED, bc.GIT_UNKNOWN, "anything-else"])
def test_a_dirty_or_unknown_git_state_is_refused(h: Harness, state: str) -> None:
    code, _, err = h.main(git=h.git_reader(state))
    assert code == bc.EXIT_REFUSED
    assert h.calls == ["freeze", "git"]
    assert "REFUSED" in err


def test_a_clean_state_without_a_commit_is_refused(h: Harness) -> None:
    assert h.main(git=h.git_reader(bc.GIT_CLEAN, None))[0] == bc.EXIT_REFUSED


def test_an_unexpected_preflight_error_is_exit_2_with_the_class_name_only(h: Harness) -> None:
    def boom(root: Path):
        raise RuntimeError("secret detail")

    code, _, err = h.main(git=boom)
    assert code == bc.EXIT_REFUSED
    assert "RuntimeError" in err and "secret detail" not in err


def test_an_unexpected_execution_error_is_exit_1(h: Harness) -> None:
    def boom(pre, out, err):
        raise ValueError("secret detail")

    code, _, err = h.main(executor=boom)
    assert code == bc.EXIT_FAILURE
    assert "ValueError" in err and "secret detail" not in err


# ---------- the real tracked-only git reader ----------

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True)


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)


def make_repo(path: Path) -> Path:
    path.mkdir()
    run_git(path, "init", "-q")
    (path / "tracked.txt").write_text("one\n", encoding="utf-8", newline="\n")
    run_git(path, "add", "tracked.txt")
    run_git(path, "commit", "-q", "-m", "first")
    return path


@needs_git
def test_git_state_clean_ignores_untracked_files(tmp_path: Path, git_env: None) -> None:
    root = make_repo(tmp_path / "r")
    (root / "private.txt").write_text("x", encoding="utf-8")
    commit, state = bc.tracked_git_state(root)
    assert state == bc.GIT_CLEAN and commit is not None and len(commit) == 40


@needs_git
def test_git_state_unstaged_and_staged(tmp_path: Path, git_env: None) -> None:
    root = make_repo(tmp_path / "r")
    (root / "tracked.txt").write_text("two\n", encoding="utf-8", newline="\n")
    assert bc.tracked_git_state(root)[1] == bc.GIT_UNSTAGED
    run_git(root, "add", "tracked.txt")
    assert bc.tracked_git_state(root)[1] == bc.GIT_STAGED


@needs_git
def test_git_state_unknown_without_a_commit_or_repository(tmp_path: Path, git_env: None) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    run_git(empty, "init", "-q")
    assert bc.tracked_git_state(empty) == (None, bc.GIT_UNKNOWN)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert bc.tracked_git_state(plain)[1] == bc.GIT_UNKNOWN


@needs_git
def test_a_subdirectory_of_an_enclosing_repository_is_rejected(tmp_path: Path, git_env: None) -> None:
    outer = make_repo(tmp_path / "outer")
    inner = outer / "sub"
    inner.mkdir()
    (inner / "file.txt").write_text("x", encoding="utf-8")
    assert bc.tracked_git_state(outer)[1] == bc.GIT_CLEAN  # the enclosing repository itself is clean...
    assert bc.tracked_git_state(inner) == (None, bc.GIT_UNKNOWN)  # ...but it is not the supplied root
    assert bc.tracked_git_state(inner / "missing") == (None, bc.GIT_UNKNOWN)


@needs_git
def test_a_nested_repository_is_judged_by_its_own_toplevel(tmp_path: Path, git_env: None) -> None:
    outer = make_repo(tmp_path / "outer")
    nested = make_repo(outer / "nested")
    assert bc.tracked_git_state(nested)[1] == bc.GIT_CLEAN


@needs_git
def test_a_link_to_the_repository_root_is_accepted(tmp_path: Path, git_env: None) -> None:
    repo = make_repo(tmp_path / "r")
    link = tmp_path / "link"
    try:
        link.symlink_to(repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    assert bc.tracked_git_state(link)[1] == bc.GIT_CLEAN


def test_a_different_reported_toplevel_is_rejected_even_when_everything_else_is_clean(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        stdout = str(tmp_path.parent) + "\n" if "--show-toplevel" in argv else HEAD + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.tracked_git_state(tmp_path) == (None, bc.GIT_UNKNOWN)


def test_a_failing_or_empty_toplevel_answer_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for code, stdout in ((128, ""), (0, ""), (0, "\n")):
        monkeypatch.setattr(bc.subprocess, "run", lambda argv, _c=code, _o=stdout, **k:
                            subprocess.CompletedProcess(argv, _c, stdout=_o, stderr=""))
        assert bc.tracked_git_state(tmp_path) == (None, bc.GIT_UNKNOWN)


@needs_git
def test_git_state_does_not_depend_on_the_working_directory(tmp_path: Path, git_env: None,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    clean = make_repo(tmp_path / "clean")
    dirty = make_repo(tmp_path / "dirty")
    (dirty / "tracked.txt").write_text("changed\n", encoding="utf-8", newline="\n")
    monkeypatch.chdir(dirty)
    assert bc.tracked_git_state(clean)[1] == bc.GIT_CLEAN
    monkeypatch.chdir(clean)
    assert bc.tracked_git_state(dirty)[1] == bc.GIT_UNSTAGED


def test_the_help_description_describes_the_two_pass_run_and_is_not_a_stale_checkpoint_note() -> None:
    description = bc.build_parser().description or ""
    assert "two complete dense passes" in description.lower() and "no report is written" in description
    assert "preflight gates exist" not in description and "this checkpoint" not in description


def test_every_git_command_names_the_repository_and_never_lists_untracked_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        recorded.append((list(argv), kwargs))
        stdout = ""
        if "--show-toplevel" in argv:
            stdout = str(tmp_path) + "\n"
        elif "rev-parse" in argv:
            stdout = HEAD + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.tracked_git_state(tmp_path) == (HEAD, bc.GIT_CLEAN)
    assert len(recorded) == 4
    assert [a[0][6:8] for a in recorded][0] == ["rev-parse", "--show-toplevel"]
    for argv, kwargs in recorded:
        assert argv[:6] == ["git", "-C", str(tmp_path), "--no-optional-locks", "-c", "core.fsmonitor=false"]
        assert kwargs.get("shell") is False and "cwd" not in kwargs and "timeout" in kwargs
        assert not {"status", "ls-files", "--others", "clean", "add", "commit", "checkout"} & set(argv)
        assert not [name for name in kwargs["env"] if name.upper().startswith("GIT_")]


def test_all_four_git_checks_disable_the_filesystem_monitor_before_the_subcommand(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        stdout = str(tmp_path) + "\n" if "--show-toplevel" in argv else HEAD + "\n" if "rev-parse" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.tracked_git_state(tmp_path) == (HEAD, bc.GIT_CLEAN)
    assert [argv[6] for argv in recorded] == ["rev-parse", "rev-parse", "diff", "diff"]
    assert len(recorded) == 4
    for argv in recorded:
        position = argv.index("core.fsmonitor=false")
        assert argv[position - 1] == "-c"
        assert position < argv.index(argv[6])  # before the Git subcommand
        assert argv.count("core.fsmonitor=false") == 1


def test_the_git_environment_drops_every_git_variable_and_keeps_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "git_namespace", "GIT_CONFIG_COUNT"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("DOCUMIND_KEEP", "1")
    env = bc._git_environment()
    assert not [name for name in env if name.upper().startswith("GIT_")]
    assert env["DOCUMIND_KEEP"] == "1"


@needs_git
@pytest.mark.parametrize("variables", [
    ("GIT_DIR",), ("GIT_WORK_TREE",), ("GIT_INDEX_FILE",), ("GIT_COMMON_DIR",),
    ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"),
])
def test_misleading_git_variables_cannot_redirect_the_checks(
        tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch, variables: tuple[str, ...]) -> None:
    clean = make_repo(tmp_path / "clean")
    dirty = make_repo(tmp_path / "dirty")
    (dirty / "tracked.txt").write_text("changed\n", encoding="utf-8", newline="\n")
    run_git(dirty, "add", "tracked.txt")  # the dirty repository has a staged change in its own index
    misleading = {
        "GIT_DIR": str(dirty / ".git"), "GIT_WORK_TREE": str(dirty), "GIT_INDEX_FILE": str(dirty / ".git" / "index"),
        "GIT_COMMON_DIR": str(dirty / ".git"),
    }
    for name in variables:
        monkeypatch.setenv(name, misleading[name])
    assert bc.tracked_git_state(clean)[1] == bc.GIT_CLEAN
    assert bc.tracked_git_state(dirty)[1] == bc.GIT_STAGED


@needs_git
def test_a_missing_or_foreign_index_file_variable_changes_nothing(
        tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "r")
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "does-not-exist"))
    assert bc.tracked_git_state(repo)[1] == bc.GIT_UNSTAGED  # the real index is used, not the missing one
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-such-git-dir"))
    assert bc.tracked_git_state(repo)[1] == bc.GIT_UNSTAGED


@needs_git
def test_untracked_paths_stay_unlisted_under_isolation(tmp_path: Path, git_env: None,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "r")
    (repo / "private.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    assert bc.tracked_git_state(repo)[1] == bc.GIT_CLEAN


@pytest.mark.parametrize("failure", [OSError("x"), subprocess.TimeoutExpired("git", 1)])
def test_git_failures_are_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception) -> None:
    def fake_run(*a, **k):
        raise failure

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.tracked_git_state(tmp_path) == (None, bc.GIT_UNKNOWN)


def test_an_unexpected_diff_exit_code_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(argv, **kwargs):
        if "--show-toplevel" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=str(tmp_path) + "\n", stderr="")
        if "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=HEAD + "\n", stderr="")
        return subprocess.CompletedProcess(argv, 128, stdout="", stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.tracked_git_state(tmp_path)[1] == bc.GIT_UNKNOWN


# ---------- static properties of the module ----------

def module_tree() -> tuple[str, ast.Module]:
    source = (REPO / "app" / "rag" / "baseline_cli.py").read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_the_module_names_no_held_out_split_file_and_holds_no_hash_literal() -> None:
    source, tree = module_tree()
    for needle in ("test" + ".jsonl", "audit" + ".jsonl"):
        assert needle not in source.casefold()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not (len(node.value) == 64 and all(c in "0123456789abcdef" for c in node.value))


def test_top_level_imports_are_standard_library_and_two_light_modules() -> None:
    _, tree = module_tree()
    allowed_app = {"app.rag.evaluation.freeze_binding", "app.rag.evaluation.runner_inputs",
                   "app.rag.evaluation.schema"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in allowed_app or module.split(".")[0] in sys.stdlib_module_names, module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name


def test_lazy_imports_exist_only_in_two_functions_and_pilot_cli_only_after_the_anchor() -> None:
    _, tree = module_tree()
    by_function: dict[str, list[ast.AST]] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            nodes = [n for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))]
            if nodes:
                by_function[fn.name] = nodes
    assert set(by_function) == {"_preflight", "_heavy"}
    assert not [n for nodes in by_function.values() for n in nodes if isinstance(n, ast.Import)]
    (pilot_import,) = by_function["_preflight"]
    assert isinstance(pilot_import, ast.ImportFrom) and pilot_import.module == "app.rag.pilot_cli"
    preflight = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "_preflight")
    anchor_line = next(n.lineno for n in ast.walk(preflight) if isinstance(n, ast.Call)
                       and getattr(n.func, "id", "") == "_anchor_matches")
    assert pilot_import.lineno > anchor_line  # the import comes after the anchor comparison
    # ``_heavy`` is called only from ``_execute``, which main reaches only after the preflight
    callers = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               for n in ast.walk(f) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_heavy"]
    assert callers == ["_execute"]


# ---------- fresh-process import boundary ----------

PRE_ANCHOR_FORBIDDEN = (
    "sqlalchemy", "app.core", "app.db", "app.database", "app.models", "app.main", "app.rag.embedder",
    "app.rag.retrieval", "app.rag.pilot_cli", "app.rag.snapshot_cli", "app.rag.backfill_safety",
    "app.rag.evaluation.engine", "app.rag.evaluation.pool", "app.rag.evaluation.pool_db",
    "app.rag.evaluation.report", "app.rag.baseline_run", "torch", "sentence_transformers", "psycopg", "psycopg2",
    "pgvector",
)
NEVER_FORBIDDEN = tuple(n for n in PRE_ANCHOR_FORBIDDEN if n not in (
    "sqlalchemy", "app.rag.pilot_cli", "app.rag.snapshot_cli", "app.rag.backfill_safety",
    "app.rag.evaluation.engine", "app.rag.evaluation.pool", "app.rag.evaluation.pool_db",
    "app.rag.evaluation.report", "app.rag.baseline_run"))

CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
forbidden = json.loads(sys.argv[2])
mode, work = sys.argv[3], Path(sys.argv[4])

def loaded(names):
    return sorted(n for n in sys.modules if any(n == f or n.startswith(f + ".") for f in names))

result = {"at_import": loaded(forbidden)}
import app.rag.baseline_cli as bc
from app.rag.evaluation.freeze_binding import FreezeBinding
from app.rag.evaluation.schema import EvaluationDataError, Issue
result["after_import"] = loaded(forbidden)
(work / "repo").mkdir(); (work / "out").mkdir()
sha = "ab" * 32
def fake(root):
    result["at_anchor"] = loaded(forbidden)
    return FreezeBinding("eval-dataset-v1.0.0", "2026-11-14", "c1", sha if mode != "mismatch" else "cd" * 32,
                         "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
argv = ["--database", "d", "--owner-id", "1", "--evaluation-email-sha256", "ef" * 32,
        "--expected-freeze-record-sha256", sha, "--out", str(work / "out" / "r.json")]
kwargs = {"repo_root": work / "repo"}
if mode == "real":
    pass
else:
    kwargs["freeze_loader"] = fake
if mode == "match":
    kwargs["git_reader"] = lambda r: ("1" * 40, "clean")
    def refuse_runtime():
        result["at_runtime_loader"] = loaded(forbidden)
        raise RuntimeError("stop here")
    kwargs["runtime_loader"] = refuse_runtime
import io
err = io.StringIO()
result["code"] = bc.main(argv, stdout=io.StringIO(), stderr=err, **kwargs)
result["final"] = loaded(forbidden)
result["stderr_has_hash"] = sha in err.getvalue()
print(json.dumps(result))
"""


def child(tmp_path: Path, mode: str, forbidden: tuple[str, ...]) -> dict:
    work = tmp_path / mode
    work.mkdir()
    done = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(REPO), json.dumps(list(forbidden)), mode, str(work)],
        capture_output=True, text=True, timeout=120, check=False, env={k: v for k, v in os.environ.items()
                                                                       if k in ("PATH", "SYSTEMROOT", "HOME")},
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_importing_the_module_loads_none_of_the_forbidden_modules(tmp_path: Path) -> None:
    result = child(tmp_path, "mismatch", PRE_ANCHOR_FORBIDDEN)
    assert result["at_import"] == [] and result["after_import"] == []


def test_a_mismatched_anchor_stops_before_any_heavy_import(tmp_path: Path) -> None:
    result = child(tmp_path, "mismatch", PRE_ANCHOR_FORBIDDEN)
    assert result["at_anchor"] == []
    assert result["final"] == []
    assert result["code"] == bc.EXIT_REFUSED
    assert result["stderr_has_hash"] is False


def test_the_real_freeze_loader_failure_also_stops_before_any_heavy_import(tmp_path: Path) -> None:
    result = child(tmp_path, "real", PRE_ANCHOR_FORBIDDEN)
    assert result["code"] == bc.EXIT_REFUSED
    assert result["final"] == []


def test_after_a_match_nothing_configures_a_database_runtime_embedder_or_retriever(tmp_path: Path) -> None:
    result = child(tmp_path, "match", PRE_ANCHOR_FORBIDDEN)
    assert result["at_anchor"] == []  # the anchor is compared before any of them
    assert result["code"] == bc.EXIT_REFUSED  # the injected runtime loader refuses, as a missing configuration would
    for moment in ("at_runtime_loader", "final"):
        leaked = [n for n in result[moment] if any(n == f or n.startswith(f + ".") for f in NEVER_FORBIDDEN)]
        assert leaked == [], moment
    # the run modules (engine, report, pool, pilot helpers and the SQLAlchemy library) are loaded only after the anchor
    assert "app.rag.baseline_run" in result["final"] and "app.rag.pilot_cli" in result["final"]
