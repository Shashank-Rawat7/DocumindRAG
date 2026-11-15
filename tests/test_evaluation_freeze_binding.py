"""Tests of the freeze-binding gate (E2.1): app/rag/evaluation/freeze_binding.py.

The gate is pure, so these tests need no database, no model and no application config. They read only
public files of the repository (the freeze record, dev.jsonl, dataset.json and the corpus manifest) and
never the held-out test split: the test-file guard below makes any attempt to open, stat, list or hash
it a test failure.
"""
import ast
import builtins
import copy
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from app.rag.evaluation import freeze_binding as fb
from app.rag.evaluation import parsing
from app.rag.evaluation.schema import EvaluationDataError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANARY = "SECRET-CANARY-7731"

DEV = fb.DEV_PATH
DATASET_JSON = fb.DATASET_JSON_PATH
MANIFEST = fb.MANIFEST_PATH
TEST_FILE = "eval/dataset/test.jsonl"

# The values recorded when the dataset was frozen (eval/dataset/freeze_record.json, 2026-11-14).
RECORDED = {
    "dev_file": "09169e4f0bef3520fcbee1643af7d181681293201b76064f991744504a8795c0",
    "dataset_json": "3f23c286be3f4010d9595b971ebc1c9ae05efc882d6ec4406014c45a356436fa",
    "manifest_file": "1902c8cc1491cb3239617edc88fb70c401964bf46bb3ce4e3b9ee6edc88e6422",
    "dev_label_binding": "860078e2fdb8bfabcfaddc4f184eb6af941d52bfb004f8a61d8a7f6b8416951c",
    "canonical_manifest": "7f6d7473e5b632a4ba09f980c671f6970597e0b836e9a5a27875f360234c9cc9",
}
OTHER_HASH = "ab" * 32


def read(relative: str) -> bytes:
    assert relative != TEST_FILE
    return (PROJECT_ROOT / relative).read_bytes()


REAL = {path: read(path) for path in fb.READ_PATHS}
REAL_RECORD = json.loads(REAL[fb.RECORD_PATH].decode("utf-8"))
DEV_QUESTIONS = [q.question for q in parsing.parse_questions(REAL[DEV].decode("utf-8"), split="dev")]


def dump(record) -> bytes:
    return json.dumps(record, indent=2).encode("utf-8")


def mutated(mutate) -> bytes:
    record = copy.deepcopy(REAL_RECORD)
    mutate(record)
    return dump(record)


def run(record_bytes=None, dev=None, dataset_json=None, manifest=None):
    return fb.build_freeze_binding(
        REAL[fb.RECORD_PATH] if record_bytes is None else record_bytes,
        REAL[DEV] if dev is None else dev,
        REAL[DATASET_JSON] if dataset_json is None else dataset_json,
        REAL[MANIFEST] if manifest is None else manifest,
    )


def same_binding_but_for_the_record_hash(first: fb.FreezeBinding, second: fb.FreezeBinding) -> bool:
    return dataclasses.replace(first, record_sha256="") == dataclasses.replace(second, record_sha256="")


def codes(error: EvaluationDataError) -> list[tuple[str, str]]:
    return [(issue.code, issue.location) for issue in error.issues]


def failure(**kwargs) -> list[tuple[str, str]]:
    with pytest.raises(EvaluationDataError) as caught:
        run(**kwargs)
    return codes(caught.value)


# ---------- A: the real repository ----------

def test_the_real_freeze_record_passes_with_the_recorded_values():
    binding = fb.load_freeze_binding(PROJECT_ROOT)
    assert binding.dataset_version == "eval-dataset-v1.0.0"
    assert binding.frozen_on == "2026-11-14"
    assert binding.corpus_version == "eval-corpus-v1"
    assert binding.dev_file_sha256 == RECORDED["dev_file"]
    assert binding.dataset_json_sha256 == RECORDED["dataset_json"]
    assert binding.manifest_file_sha256 == RECORDED["manifest_file"]
    assert binding.dev_label_binding_sha256 == RECORDED["dev_label_binding"]
    assert binding.canonical_manifest_sha256 == RECORDED["canonical_manifest"]


def test_the_record_hash_is_the_sha256_of_the_record_bytes():
    assert fb.load_freeze_binding(PROJECT_ROOT).record_sha256 == hashlib.sha256(REAL[fb.RECORD_PATH]).hexdigest()


def test_the_pure_function_and_the_loader_agree():
    assert run() == fb.load_freeze_binding(PROJECT_ROOT)


def test_the_binding_is_frozen_and_has_no_test_split_field():
    binding = run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.dataset_version = "x"
    names = {f.name for f in dataclasses.fields(fb.FreezeBinding)}
    assert not any("test" in name for name in names)


def test_the_label_binding_is_the_existing_dataset_hash_not_a_second_implementation():
    questions = parsing.parse_questions(REAL[DEV].decode("utf-8"), split="dev")
    assert run().dev_label_binding_sha256 == parsing.dataset_hash(questions)
    assert run().canonical_manifest_sha256 == parsing.manifest_hash(
        parsing.parse_manifest(REAL[MANIFEST].decode("utf-8"))
    )


def test_the_real_record_has_exactly_the_schema_the_gate_expects():
    assert set(REAL_RECORD) == set(fb._TOP_LEVEL)
    assert set(REAL_RECORD["files"]) == set(fb.VERIFIED_FILE_PATHS + fb.RECORDED_ONLY_PATHS)


# ---------- B: record format and status ----------

@pytest.mark.parametrize("value", ["eval-freeze-record-v2", "", "EVAL-FREEZE-RECORD-V1"])
def test_a_wrong_record_format_fails(value):
    result = failure(record_bytes=mutated(lambda r: r.update(record_format=value)))
    assert ("FREEZE_RECORD_FORMAT", "freeze_record.record_format") in result or (
        value == "" and ("FREEZE_FIELD_TYPE", "freeze_record.record_format") in result
    )


@pytest.mark.parametrize("value", ["frozen", "baseline", "frozen_dataset_is_baseline", "FROZEN_DATASET_NOT_BASELINE"])
def test_a_wrong_status_fails(value):
    result = failure(record_bytes=mutated(lambda r: r.update(status=value)))
    assert result == [("FREEZE_STATUS", "freeze_record.status")]


@pytest.mark.parametrize("name", ["record_format", "status", "dataset_version", "frozen_on", "anchoring"])
@pytest.mark.parametrize("value", [1, None, True, ["x"], {}])
def test_a_non_string_scalar_field_fails_with_a_type_issue(name, value):
    result = failure(record_bytes=mutated(lambda r: r.update({name: value})))
    assert result == [("FREEZE_FIELD_TYPE", f"freeze_record.{name}")]


@pytest.mark.parametrize(
    "value", ["v1.0.0", "eval-dataset-v1", "eval-dataset-v1.0", "eval-dataset-v1.0.0-rc1", " eval-dataset-v1.0.0"]
)
def test_a_malformed_dataset_version_fails(value):
    assert failure(record_bytes=mutated(lambda r: r.update(dataset_version=value))) == [
        ("FREEZE_VERSION_FORMAT", "freeze_record.dataset_version")
    ]


@pytest.mark.parametrize(
    "value", ["2026-13-01", "2026-02-30", "20261114", "2026-11-14T00:00:00", "14-11-2026", "2026-1-14"]
)
def test_a_malformed_frozen_on_fails(value):
    assert failure(record_bytes=mutated(lambda r: r.update(frozen_on=value))) == [
        ("FREEZE_DATE_FORMAT", "freeze_record.frozen_on")
    ]


# ---------- C: malformed JSON ----------

DUPLICATE = REAL[fb.RECORD_PATH].replace(b'"status"', b'"status": "x",\n  "status"', 1)

@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"{",
        b"not json",
        b'{"a": 1,}',
        b"\xef\xbb\xbf" + REAL[fb.RECORD_PATH],
        b"\xff\xfe{}",
        DUPLICATE,
        REAL[fb.RECORD_PATH].replace(b'"anchoring"', b'"x": NaN, "anchoring"', 1),
        REAL[fb.RECORD_PATH].replace(b'"anchoring"', b'"x": Infinity, "anchoring"', 1),
        REAL[fb.RECORD_PATH].replace(b'"anchoring"', b'"x": 1e999, "anchoring"', 1),
        b"[" * 200_000,
    ],
    ids=["empty", "open-brace", "plain-text", "trailing-comma", "bom", "bad-utf8", "duplicate-key", "nan",
         "infinity", "overflow", "too-deep"],
)
def test_malformed_json_fails_with_one_syntax_issue(data):
    assert failure(record_bytes=data) == [("FREEZE_RECORD_SYNTAX", "freeze_record")]


@pytest.mark.parametrize("data", [b"[]", b"null", b"1", b'"text"', b"true"])
def test_a_record_that_is_not_an_object_fails(data):
    assert failure(record_bytes=data) == [("FREEZE_FIELD_TYPE", "freeze_record")]


# ---------- D: missing and unknown fields ----------

@pytest.mark.parametrize("name", list(REAL_RECORD))
def test_each_missing_top_level_field_fails(name):
    assert failure(record_bytes=mutated(lambda r: r.pop(name))) == [("FREEZE_FIELD_MISSING", f"freeze_record.{name}")]


def test_an_unknown_top_level_field_fails_and_its_name_is_not_reported():
    result = failure(record_bytes=mutated(lambda r: r.update({CANARY: 1})))
    assert result == [("FREEZE_FIELD_UNKNOWN", "freeze_record")]


@pytest.mark.parametrize("section", ["files", "corpus", "label_binding_hashes"])
def test_an_unknown_field_in_a_closed_section_fails(section):
    result = failure(record_bytes=mutated(lambda r: r[section].update({CANARY: OTHER_HASH})))
    assert result == [("FREEZE_FIELD_UNKNOWN", f"freeze_record.{section}")]


@pytest.mark.parametrize(
    "section,name",
    [("files", p) for p in fb.VERIFIED_FILE_PATHS + fb.RECORDED_ONLY_PATHS]
    + [("corpus", "corpus_version"), ("corpus", "canonical_manifest_hash"),
       ("label_binding_hashes", "dev"), ("label_binding_hashes", "test")],
)
def test_each_missing_field_in_a_closed_section_fails(section, name):
    result = failure(record_bytes=mutated(lambda r: r[section].pop(name)))
    assert result == [("FREEZE_FIELD_MISSING", f"freeze_record.{section}.{name}")]


@pytest.mark.parametrize("section", ["files", "corpus", "label_binding_hashes", "counts", "review_provenance",
                                     "audit", "pilot_reference", "git_state"])
@pytest.mark.parametrize("value", [[], "text", 1, None])
def test_a_section_that_is_not_an_object_fails(section, value):
    assert failure(record_bytes=mutated(lambda r: r.update({section: value}))) == [
        ("FREEZE_FIELD_TYPE", f"freeze_record.{section}")
    ]


@pytest.mark.parametrize("value", ["text", {}, None, 1, ["ok", 2]])
def test_limitations_must_be_a_list_of_strings(value):
    assert failure(record_bytes=mutated(lambda r: r.update(limitations=value))) == [
        ("FREEZE_FIELD_TYPE", "freeze_record.limitations")
    ]


def test_the_informational_sections_are_not_interpreted():
    record = copy.deepcopy(REAL_RECORD)
    record["counts"] = {"anything": [1, 2, {"x": None}]}
    record["audit"] = {}
    record["limitations"] = []
    assert same_binding_but_for_the_record_hash(run(record_bytes=dump(record)), run())


# ---------- E: recorded hash formats ----------

BAD_HASHES = [
    "", "AB" * 32, OTHER_HASH[:-1], OTHER_HASH + "0", "g" * 64, " " + OTHER_HASH[1:], 123, None, True, [OTHER_HASH],
]


@pytest.mark.parametrize("value", BAD_HASHES)
@pytest.mark.parametrize(
    "path", [("files", p) for p in fb.VERIFIED_FILE_PATHS + fb.RECORDED_ONLY_PATHS]
    + [("corpus", "canonical_manifest_hash"), ("label_binding_hashes", "dev"), ("label_binding_hashes", "test")],
)
def test_a_badly_formatted_recorded_hash_fails_and_nothing_is_verified(path, value):
    section, name = path
    result = failure(record_bytes=mutated(lambda r: r[section].update({name: value})))
    assert result == [("FREEZE_HASH_FORMAT", f"freeze_record.{section}.{name}")]


def test_several_structural_problems_are_all_reported_in_one_error():
    def many(record):
        record["status"] = "x"
        record["frozen_on"] = "nope"
        record["files"][DEV] = "zz"
        record["corpus"].pop("corpus_version")

    assert sorted(code for code, _ in failure(record_bytes=mutated(many))) == [
        "FREEZE_DATE_FORMAT", "FREEZE_FIELD_MISSING", "FREEZE_HASH_FORMAT", "FREEZE_STATUS",
    ]


# ---------- F: file hash mismatches ----------

@pytest.mark.parametrize("path", fb.VERIFIED_FILE_PATHS)
def test_a_recorded_raw_hash_that_differs_from_the_file_fails_with_a_stable_code(path):
    assert failure(record_bytes=mutated(lambda r: r["files"].update({path: OTHER_HASH}))) == [
        ("FREEZE_FILE_HASH_MISMATCH", f"freeze_record.files.{path}")
    ]


@pytest.mark.parametrize(
    "path,kwarg,change",
    [
        (DEV, "dev", lambda b: b.replace(b"\n", b"\r\n")),                 # same labels, different bytes
        (DATASET_JSON, "dataset_json", lambda b: b + b"\n"),
        (MANIFEST, "manifest", lambda b: b + b"\n"),                       # same canonical manifest
    ],
)
def test_changed_file_bytes_fail_only_the_raw_hash_check_when_the_content_is_unchanged(path, kwarg, change):
    assert failure(**{kwarg: change(REAL[path])}) == [("FREEZE_FILE_HASH_MISMATCH", f"freeze_record.files.{path}")]


def test_a_changed_dev_label_fails_both_the_raw_hash_and_the_label_binding():
    shorter = REAL[DEV].rstrip(b"\r\n").rsplit(b"\n", 1)[0] + b"\n"
    assert failure(dev=shorter) == [
        ("FREEZE_FILE_HASH_MISMATCH", f"freeze_record.files.{DEV}"),
        ("FREEZE_LABEL_BINDING_MISMATCH", "freeze_record.label_binding_hashes.dev"),
    ]


def test_a_recorded_label_binding_hash_that_differs_fails_alone():
    result = failure(record_bytes=mutated(lambda r: r["label_binding_hashes"].update(dev=OTHER_HASH)))
    assert result == [("FREEZE_LABEL_BINDING_MISMATCH", "freeze_record.label_binding_hashes.dev")]


def test_a_recorded_canonical_manifest_hash_that_differs_fails_alone():
    result = failure(record_bytes=mutated(lambda r: r["corpus"].update(canonical_manifest_hash=OTHER_HASH)))
    assert result == [("FREEZE_MANIFEST_HASH_MISMATCH", "freeze_record.corpus.canonical_manifest_hash")]


@pytest.mark.parametrize("dev", [b"", b"{", b"\xff\xfe", REAL[DEV][:-40], b'{"x": 1}\n'])
def test_an_unreadable_dev_file_fails_closed(dev):
    result = failure(dev=dev)
    label_mismatch = ("FREEZE_LABEL_BINDING_MISMATCH", "freeze_record.label_binding_hashes.dev")
    assert ("FREEZE_DEV_UNPARSEABLE", DEV) in result or label_mismatch in result
    assert ("FREEZE_FILE_HASH_MISMATCH", f"freeze_record.files.{DEV}") in result


@pytest.mark.parametrize("manifest", [b"", b"{", b"\xff", b"[]"])
def test_an_unreadable_manifest_fails_closed(manifest):
    result = failure(manifest=manifest)
    assert ("FREEZE_MANIFEST_UNPARSEABLE", MANIFEST) in result
    assert ("FREEZE_FILE_HASH_MISMATCH", f"freeze_record.files.{MANIFEST}") in result


def test_a_test_split_record_in_the_dev_file_is_refused():
    line = REAL[DEV].decode("utf-8").splitlines()[0]
    assert '"split"' in line
    swapped = re.sub(r'"split":\s*"dev"', '"split": "test"', line)
    assert swapped != line
    result = failure(dev=(swapped + "\n").encode("utf-8"))
    assert ("FREEZE_DEV_UNPARSEABLE", DEV) in result


def test_the_recorded_only_entries_are_not_verified():
    def change(record):
        record["files"][TEST_FILE] = OTHER_HASH
        record["files"]["eval/dataset/audit.jsonl"] = OTHER_HASH
        record["label_binding_hashes"]["test"] = OTHER_HASH

    assert same_binding_but_for_the_record_hash(run(record_bytes=mutated(change)), run())


# ---------- G: argument types ----------

@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("value", ["text", bytearray(b"x"), memoryview(b"x"), None, 1])
def test_every_input_must_be_bytes(position, value):
    args = [REAL[fb.RECORD_PATH], REAL[DEV], REAL[DATASET_JSON], REAL[MANIFEST]]
    args[position] = value
    with pytest.raises(TypeError):
        fb.build_freeze_binding(*args)


@pytest.mark.parametrize("value", [None, 1, b"x", ["a"]])
def test_the_repo_root_must_be_a_str_or_path(value):
    with pytest.raises(TypeError):
        fb.load_freeze_binding(value)


# ---------- H: the loader ----------

def make_repo(tmp_path: Path, *, with_test_file: bool = False) -> Path:
    for relative in fb.READ_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(REAL[relative])
    if with_test_file:
        (tmp_path / TEST_FILE).write_bytes(CANARY.encode("utf-8"))
    return tmp_path


def test_the_loader_reads_a_copy_of_the_files(tmp_path):
    assert fb.load_freeze_binding(make_repo(tmp_path)) == run()
    assert fb.load_freeze_binding(str(make_repo(tmp_path / "again"))) == run()


def test_the_loader_reports_every_missing_file(tmp_path):
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(tmp_path)
    assert codes(caught.value) == [("FREEZE_FILE_UNREADABLE", relative) for relative in fb.READ_PATHS]


def test_the_loader_reports_one_missing_file(tmp_path):
    repo = make_repo(tmp_path)
    (repo / DATASET_JSON).unlink()
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(repo)
    assert codes(caught.value) == [("FREEZE_FILE_UNREADABLE", DATASET_JSON)]


def test_the_read_paths_never_include_the_test_split():
    assert TEST_FILE not in fb.READ_PATHS
    assert not any(Path(p).name.casefold() == "test.jsonl" for p in fb.READ_PATHS)
    assert TEST_FILE in fb.RECORDED_ONLY_PATHS


# ---------- H2: the filesystem safety guard (links, reparse points, wrong types) ----------

ALL_DIRS = ("eval", "eval/dataset", "eval/corpus")


def link(target: Path, where: Path, *, directory: bool = False) -> None:
    where.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, where, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links cannot be created here")


def unsafe(repo: Path) -> list[tuple[str, str]]:
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(repo)
    return codes(caught.value)


@pytest.mark.parametrize("relative", fb.READ_PATHS)
def test_a_symlink_in_place_of_an_input_file_is_refused_and_never_followed(tmp_path, guard, relative):
    repo = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(CANARY.encode("utf-8"))
    (repo / relative).unlink()
    link(outside, repo / relative)
    guard.start()
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(repo)
    assert codes(caught.value) == [("FREEZE_PATH_UNSAFE", relative)]
    assert CANARY not in str(caught.value) + repr(caught.value.issues)
    assert guard.opened == set() and guard.statted == set() and guard.resolved == []


def test_a_dev_file_linked_to_the_test_split_is_refused_without_touching_the_test_split(tmp_path, guard):
    repo = make_repo(tmp_path / "repo", with_test_file=True)
    (repo / DEV).unlink()
    link(repo / TEST_FILE, repo / DEV)
    guard.start()  # the guard raises on any operation whose path names test.jsonl
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(repo)
    assert codes(caught.value) == [("FREEZE_PATH_UNSAFE", DEV)]
    assert guard.opened == set() and guard.statted == set()
    assert CANARY not in str(caught.value) + repr(caught.value.issues)


@pytest.mark.parametrize("relative", fb.READ_PATHS)
def test_a_broken_symlink_is_refused_as_unsafe(tmp_path, relative):
    repo = make_repo(tmp_path / "repo")
    (repo / relative).unlink()
    link(tmp_path / "does-not-exist", repo / relative)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", relative)]


@pytest.mark.parametrize("directory", ALL_DIRS)
def test_a_symlinked_parent_directory_is_refused(tmp_path, directory):
    repo = make_repo(tmp_path / "repo")
    moved = tmp_path / "moved"
    (repo / directory).rename(moved)
    link(moved, repo / directory, directory=True)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", directory)]


def test_a_broken_symlink_in_place_of_a_directory_is_refused(tmp_path):
    repo = make_repo(tmp_path / "repo")
    for relative in fb.READ_PATHS:
        if relative.startswith("eval/corpus"):
            (repo / relative).unlink()
    (repo / "eval/corpus").rmdir()
    link(tmp_path / "does-not-exist", repo / "eval/corpus", directory=True)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", "eval/corpus")]


def test_every_unsafe_directory_of_a_group_is_reported_together(tmp_path):
    repo = make_repo(tmp_path / "repo")
    for directory in ("eval/dataset", "eval/corpus"):
        moved = tmp_path / ("moved-" + directory.rsplit("/", 1)[-1])
        (repo / directory).rename(moved)
        link(moved, repo / directory, directory=True)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", "eval/dataset"), ("FREEZE_PATH_UNSAFE", "eval/corpus")]


def test_every_unsafe_input_file_is_reported_together(tmp_path):
    repo = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    for relative in (DEV, MANIFEST):
        (repo / relative).unlink()
        link(outside, repo / relative)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", DEV), ("FREEZE_PATH_UNSAFE", MANIFEST)]


@pytest.mark.parametrize("relative", fb.READ_PATHS)
def test_a_directory_in_place_of_an_input_file_is_refused(tmp_path, relative):
    repo = make_repo(tmp_path / "repo")
    (repo / relative).unlink()
    (repo / relative).mkdir()
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", relative)]


@pytest.mark.parametrize("directory", ALL_DIRS)
def test_a_file_in_place_of_a_parent_directory_is_refused(tmp_path, directory):
    repo = make_repo(tmp_path / "repo")
    stash = tmp_path / "stash"
    (repo / directory).rename(stash)
    (repo / directory).write_bytes(b"not a directory")
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", directory)]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available on this platform")
def test_a_fifo_in_place_of_an_input_file_is_refused_without_blocking(tmp_path):
    repo = make_repo(tmp_path / "repo")
    (repo / DEV).unlink()
    os.mkfifo(repo / DEV)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", DEV)]  # opening a FIFO would block forever


def _fake_reparse(monkeypatch, name: str) -> None:
    real = os.lstat

    def lstat(path, *args, **kwargs):
        info = real(path, *args, **kwargs)
        if os.fspath(path).replace("\\", "/").endswith(name):
            return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(os, "lstat", lstat)


@pytest.mark.parametrize("relative", fb.READ_PATHS)
def test_a_windows_reparse_point_on_an_input_file_is_refused(tmp_path, monkeypatch, relative):
    repo = make_repo(tmp_path / "repo")
    _fake_reparse(monkeypatch, "/" + relative.rsplit("/", 1)[-1])
    assert ("FREEZE_PATH_UNSAFE", relative) in unsafe(repo)


@pytest.mark.parametrize("directory", ALL_DIRS)
def test_a_windows_junction_in_place_of_a_directory_is_refused(tmp_path, monkeypatch, directory):
    repo = make_repo(tmp_path / "repo")
    _fake_reparse(monkeypatch, "/" + directory)
    assert unsafe(repo) == [("FREEZE_PATH_UNSAFE", directory)]


def test_file_attributes_without_the_reparse_flag_are_accepted(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo")
    real = os.lstat

    def lstat(path, *args, **kwargs):
        info = real(path, *args, **kwargs)
        return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x20)

    monkeypatch.setattr(os, "lstat", lstat)
    assert fb.load_freeze_binding(repo) == run()


def test_a_symlinked_repository_root_is_allowed(tmp_path):
    repo = make_repo(tmp_path / "real-repo")
    link(repo, tmp_path / "linked-root", directory=True)
    assert fb.load_freeze_binding(tmp_path / "linked-root") == run()


def test_the_safety_check_runs_before_any_file_is_read(tmp_path, guard):
    repo = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    (repo / MANIFEST).unlink()
    link(outside, repo / MANIFEST)
    guard.start()
    unsafe(repo)
    assert guard.opened == set()  # the three safe files were not read either


def test_unsafe_path_issue_has_a_fixed_message_and_a_schema_location(tmp_path):
    repo = make_repo(tmp_path / "repo")
    (repo / DEV).unlink()
    link(tmp_path / CANARY, repo / DEV)
    with pytest.raises(EvaluationDataError) as caught:
        fb.load_freeze_binding(repo)
    (issue,) = caught.value.issues
    assert issue.code == "FREEZE_PATH_UNSAFE" and issue.severity == "error"
    assert issue.message == fb._MESSAGES["FREEZE_PATH_UNSAFE"] and issue.location == DEV
    assert CANARY not in repr(caught.value.issues)


# ---------- I: the test split is never touched ----------

class FilesystemGuard:
    """Makes any filesystem operation that names test.jsonl fail, and records what is opened under a root.

    Installed with ``start()`` AFTER the test has built its temporary repository, so the guard only
    watches the code under test.
    """

    def __init__(self, monkeypatch, root: Path) -> None:
        self._monkeypatch = monkeypatch
        self.root = os.path.normcase(os.fspath(root.resolve())).replace("\\", "/")
        self.opened: set[str] = set()
        self.lstatted: set[str] = set()
        self.statted: set[str] = set()
        self.resolved: list[str] = []
        self.enumerated: list[str] = []

    @staticmethod
    def _text(path) -> str | None:
        if isinstance(path, int):
            return None
        raw = os.fspath(path)
        return raw.replace("\\", "/") if isinstance(raw, str) else None

    def _under_root(self, text: str) -> bool:
        return os.path.normcase(text).replace("\\", "/").startswith(self.root)

    def check(self, path, *, enumerate_: bool = False, record: bool = False, kind: str | None = None) -> None:
        text = self._text(path)
        if text is None:
            return
        if text.rstrip("/").rsplit("/", 1)[-1].casefold() == "test.jsonl":
            raise AssertionError("the test split must never be touched")
        if enumerate_ and self._under_root(text):
            self.enumerated.append(text)
        if kind == "resolve":
            self.resolved.append(text)
        if kind in ("lstat", "stat") and self._under_root(text):
            getattr(self, kind + "ted").add(os.path.relpath(text, self.root).replace("\\", "/"))
        if record and self._under_root(text):
            # Python 3.10's Path.open uses a captured io.open, so opens are also recorded at the Path
            # methods; a set removes the double record that Python 3.12 produces.
            self.opened.add(os.path.relpath(text, self.root).replace("\\", "/"))

    def start(self) -> None:
        patch = self._monkeypatch
        state = self

        def opening(original):
            def wrapper(file, *args, **kwargs):
                state.check(file, record=True)
                return original(file, *args, **kwargs)
            return wrapper

        def checking(original, *, enumerate_=False, kind=None):
            def wrapper(path=".", *args, **kwargs):
                state.check(path, enumerate_=enumerate_, kind=kind)
                return original(path, *args, **kwargs)
            return wrapper

        def method(original, *, enumerate_=False, record=False, kind=None):
            def wrapper(self, *args, **kwargs):
                state.check(self, enumerate_=enumerate_, record=record, kind=kind)
                return original(self, *args, **kwargs)
            return wrapper

        patch.setattr(builtins, "open", opening(builtins.open))
        patch.setattr(io, "open", opening(io.open))
        patch.setattr(os, "open", checking(os.open))
        patch.setattr(os, "stat", checking(os.stat, kind="stat"))
        patch.setattr(os, "lstat", checking(os.lstat, kind="lstat"))
        patch.setattr(os.path, "realpath", checking(os.path.realpath, kind="resolve"))
        patch.setattr(os, "scandir", checking(os.scandir, enumerate_=True))
        patch.setattr(os, "listdir", checking(os.listdir, enumerate_=True))
        for name in ("stat", "lstat", "exists", "is_file", "resolve", "samefile"):
            kind = name if name in ("stat", "lstat", "resolve") else None
            patch.setattr(pathlib.Path, name, method(getattr(pathlib.Path, name), kind=kind))
        for name in ("read_bytes", "read_text", "open"):
            patch.setattr(pathlib.Path, name, method(getattr(pathlib.Path, name), record=True))
        for name in ("iterdir", "glob", "rglob"):
            patch.setattr(pathlib.Path, name, method(getattr(pathlib.Path, name), enumerate_=True))


@pytest.fixture
def guard(monkeypatch, tmp_path):
    return FilesystemGuard(monkeypatch, tmp_path)


def test_the_guard_itself_trips_on_every_kind_of_access_to_the_test_file(tmp_path, guard):
    sentinel = make_repo(tmp_path, with_test_file=True) / TEST_FILE
    guard.start()
    for attempt in (
        lambda: open(sentinel, "rb"),  # noqa: SIM115 - the call is expected to raise
        lambda: io.open(sentinel, "rb"),  # noqa: SIM115, UP020 - io.open is patched separately from open
        lambda: os.open(sentinel, os.O_RDONLY),
        lambda: os.stat(sentinel),
        lambda: os.lstat(sentinel),
        lambda: sentinel.stat(),
        lambda: sentinel.lstat(),
        lambda: sentinel.exists(),
        lambda: sentinel.is_file(),
        lambda: sentinel.read_bytes(),
        lambda: sentinel.read_text(),
        lambda: sentinel.open("rb"),
        lambda: sentinel.resolve(),
        lambda: os.path.getsize(sentinel),
        lambda: os.path.exists(sentinel),
    ):
        with pytest.raises(AssertionError, match="test split"):
            attempt()


def test_the_guard_sees_enumeration_of_the_dataset_folder(tmp_path, guard):
    repo = make_repo(tmp_path, with_test_file=True)
    guard.start()
    list(os.scandir(repo / "eval" / "dataset"))
    os.listdir(repo / "eval" / "dataset")
    list((repo / "eval" / "dataset").iterdir())
    assert len(guard.enumerated) == 3


def test_loading_the_binding_never_opens_stats_lists_or_hashes_the_test_file(tmp_path, guard):
    repo = make_repo(tmp_path, with_test_file=True)  # the sentinel sits where the real file would be
    guard.start()
    binding = fb.load_freeze_binding(repo)
    assert binding == run()
    assert guard.opened == set(fb.READ_PATHS)
    assert guard.lstatted == set(fb.READ_PATHS) | {"eval", "eval/dataset", "eval/corpus"}
    assert guard.statted == set()
    assert guard.resolved == []
    assert guard.enumerated == []


def test_the_recorded_hash_of_the_test_file_is_never_compared_with_the_file(tmp_path, guard):
    # The sentinel's bytes cannot match the hash in the record. Passing proves the file was not hashed.
    repo = make_repo(tmp_path, with_test_file=True)
    guard.start()
    assert fb.load_freeze_binding(repo).dataset_version == "eval-dataset-v1.0.0"


def test_the_pure_function_touches_the_filesystem_not_at_all(tmp_path, guard):
    guard.start()
    run()
    assert guard.opened == set()
    assert guard.lstatted == set()
    assert guard.enumerated == []


# ---------- J: nothing sensitive is ever reported ----------

SCENARIOS = {
    "syntax": lambda: {"record_bytes": b"{" + CANARY.encode()},
    "unknown-field": lambda: {"record_bytes": mutated(lambda r: r.update({CANARY: CANARY}))},
    "canary-values": lambda: {"record_bytes": mutated(lambda r: r.update(
        status=CANARY, record_format=CANARY, dataset_version=CANARY, frozen_on=CANARY))},
    "canary-hash": lambda: {"record_bytes": mutated(lambda r: r["files"].update({DEV: CANARY}))},
    "canary-limitations": lambda: {"record_bytes": mutated(lambda r: r.update(limitations=[CANARY, 1]))},
    "dev-truncated": lambda: {"dev": REAL[DEV][:-40]},
    "dev-shorter": lambda: {"dev": REAL[DEV].rstrip(b"\r\n").rsplit(b"\n", 1)[0] + b"\n"},
    "dev-garbage": lambda: {"dev": CANARY.encode() + b"\n"},
    "manifest-garbage": lambda: {"manifest": CANARY.encode()},
    "label-mismatch": lambda: {"record_bytes": mutated(lambda r: r["label_binding_hashes"].update(dev=OTHER_HASH))},
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_no_issue_or_exception_text_holds_question_text_or_record_content(name):
    with pytest.raises(EvaluationDataError) as caught:
        run(**SCENARIOS[name]())
    error = caught.value
    everything = "\n".join([str(error), repr(error), *(repr(i) for i in error.issues),
                            *(f"{i.code} {i.location} {i.message}" for i in error.issues)])
    assert CANARY not in everything
    for question in DEV_QUESTIONS:
        assert question not in everything
    assert all(i.message in fb._MESSAGES.values() for i in error.issues)  # fixed messages only
    assert all(re.fullmatch(r"[A-Za-z0-9_./\-]+", i.location) for i in error.issues)
    assert str(error) == f"{len(error.issues)} evaluation data issue(s)"


def test_every_issue_code_has_a_fixed_message_and_all_issues_are_errors():
    for scenario in SCENARIOS.values():
        with pytest.raises(EvaluationDataError) as caught:
            run(**scenario())
        assert all(i.severity == "error" for i in caught.value.issues)
    assert set(fb._MESSAGES) == {
        value for key, value in vars(fb).items() if key.startswith("FREEZE_") and isinstance(value, str)
    }


# ---------- K: the module imports no database or application configuration ----------

FORBIDDEN_LOADED = (
    "sqlalchemy", "fastapi", "pydantic", "torch", "transformers", "numpy", "psycopg2", "app.main",
    "app.core", "app.models", "app.crud", "app.api", "app.rag.retrieval", "app.rag.embedder",
    "app.rag.indexing_service", "app.rag.corpus_ingest", "app.rag.evaluation.labelling",
    "app.rag.evaluation.pool_db", "app.rag.evaluation.engine", "app.rag.evaluation.report",
)


def test_importing_the_module_loads_no_database_model_or_application_config():
    code = (
        "import sys\n"
        "import app.rag.evaluation.freeze_binding\n"
        f"forbidden = {FORBIDDEN_LOADED!r}\n"
        "bad = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden))\n"
        "print(','.join(bad))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for variable in ("DATABASE_URL", "SECRET_KEY"):
        env.pop(variable, None)
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=True, env=env)
    assert done.stdout.strip() == ""


def _module_ast() -> ast.Module:
    return ast.parse(Path(fb.__file__).read_text(encoding="utf-8"))


def test_the_module_imports_only_the_standard_library_and_the_two_existing_evaluation_modules():
    allowed_stdlib = {"__future__", "datetime", "hashlib", "json", "math", "os", "re", "stat", "collections.abc",
                      "dataclasses", "pathlib", "typing"}
    allowed_app = {"app.rag.evaluation.parsing", "app.rag.evaluation.schema"}
    found = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            found.add(node.module)
    assert found <= allowed_stdlib | allowed_app
    assert allowed_app <= found


def _docstring_nodes(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def test_the_test_file_name_appears_only_in_the_recorded_only_constant():
    tree = _module_ast()
    docstrings = _docstring_nodes(tree)
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
        and "test.jsonl" in node.value.casefold()
    ]
    assert literals == [TEST_FILE]


def test_only_the_loader_reads_files_and_only_through_read_paths():
    tree = _module_ast()
    reading = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        and any(isinstance(n, ast.Attribute) and n.attr in ("read_bytes", "read_text", "open", "stat", "exists",
                                                              "iterdir", "glob", "rglob", "resolve", "lstat",
                                                              "realpath")
                for n in ast.walk(node))
    }
    assert reading == {"load_freeze_binding", "_filesystem_issues"}
    source = Path(fb.__file__).read_text(encoding="utf-8")
    assert source.count(".read_bytes(") == 1 and "for relative in READ_PATHS" in source
    called = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert called.count("lstat") == 1 and "resolve" not in called and "realpath" not in called
    assert "readlink" not in called and "samefile" not in called


def test_the_gate_leaves_the_frozen_files_unchanged(tmp_path):
    repo = make_repo(tmp_path)
    before = {p: (repo / p).read_bytes() for p in fb.READ_PATHS}
    fb.load_freeze_binding(repo)
    assert {p: (repo / p).read_bytes() for p in fb.READ_PATHS} == before
