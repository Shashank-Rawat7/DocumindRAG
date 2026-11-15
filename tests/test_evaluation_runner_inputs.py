"""E1.7.1: the development-only input and fail-closed gate of the evaluation runner.

A  dev path checks (test split paths refused, only the path is examined)
B  the pure gate: result, split of answerable and unanswerable, hashes, read-only texts
C  the test split is refused: records of another split
D  empty dev file and malformed records
E  manifest gate
F  source texts: missing, extra, altered, truncated, empty
G  stale labels: source hash, offsets, quote, unknown document
H  privacy: no quote, question or text in errors
I  warnings do not stop the gate
J  argument types and threshold
K  the loader: temporary corpus, test.jsonl never opened
L  the real repository dev split (structure only)
M  import isolation
N  the loader needs no application config or database (subprocess, no credentials)

Run focused: pytest tests/test_evaluation_runner_inputs.py --noconftest -p no:cacheprovider
"""
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import sha256_text
from app.rag.evaluation import parsing
from app.rag.evaluation import runner_inputs as r
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.6
CANARY = "SECRET-CANARY-7731"

TEXT_A = "Alpha clause one. Beta clause two. Gamma clause three."
TEXT_B = "Delta section four. Epsilon section five. Zeta section six."


def doc(key, text, fmt="txt", file_bytes=None):
    raw = text.encode("utf-8") if file_bytes is None else file_bytes
    return s.ManifestDocument(
        corpus_key=key, title="T", publisher="P", source_url="https://example.invalid/a", license="MIT",
        license_url="https://example.invalid/l", format=fmt, language="en",
        source_file_sha256=hashlib.sha256(raw).hexdigest(),
        extracted_text_sha256=sha256_text(text), extracted_text_length=len(text),
        extraction=s.Extraction(processing_ref="d" * 40, python="3.12.4"), redistributable=True,
    )


MANIFEST = s.Manifest("eval-manifest-v1", "c1", "ingested", (doc("doc-a", TEXT_A), doc("doc-b", TEXT_B)))
TEXTS = {"doc-a": TEXT_A, "doc-b": TEXT_B}


def anchored(key, text, quote):
    start = text.index(quote)
    return s.Span(key, sha256_text(text), start, start + len(quote), quote, sha256_text(quote))


def answerable(qid="d1", question="What does the first clause say about alpha?", key="doc-a", text=TEXT_A,
               quote="Alpha clause one.", split="dev", span=None):
    return s.Question(
        question_id=qid, split=split, answerable=True, question=question, query_language="en",
        author_type="human", hindi_reviewed=False, expected_answer="Clause one.",
        units=(s.EvidenceUnit("u1", (span or anchored(key, text, quote),)),),
        source_language="en", difficulty="direct", close_distractor=False,
    )


def unanswerable(qid="du1", question="How many employees work in the delivery department?"):
    return s.Question(
        question_id=qid, split="dev", answerable=False, question=question, query_language="en",
        author_type="human", hindi_reviewed=False, unanswerable_type="off_topic",
    )


def jsonl(questions):
    return "".join(json.dumps(dataclasses.asdict(q), ensure_ascii=False) + "\n" for q in questions)


def good_dev():
    return jsonl([
        answerable("d1"),
        answerable("d2", "Which party pays the refund amount in the agreement?", "doc-b", TEXT_B, "Delta section four."),
        unanswerable("du1"),
    ])


_DEFAULT = object()


def build(dev_text=_DEFAULT, manifest=MANIFEST, texts=TEXTS, threshold=THRESHOLD):
    return r.build_dev_inputs(good_dev() if dev_text is _DEFAULT else dev_text, manifest, texts,
                              near_duplicate_threshold=threshold)


def error_of(callable_, *args, **kwargs):
    with pytest.raises(s.EvaluationDataError) as info:
        callable_(*args, **kwargs)
    return info.value


def codes(error):
    return {item.code for item in error.issues}


# ---------- A: paths ----------

def test_the_dev_file_path_is_accepted(tmp_path):
    assert r.check_dev_path(tmp_path / "dev.jsonl") == (tmp_path / "dev.jsonl").resolve()
    assert r.check_dev_path(str(tmp_path / "DEV.JSONL")).name.casefold() == "dev.jsonl"


def test_the_test_file_path_is_refused_with_its_own_code(tmp_path):
    assert codes(error_of(r.check_dev_path, tmp_path / "test.jsonl")) == {r.RUN_TEST_SPLIT_PATH}
    assert codes(error_of(r.check_dev_path, str(tmp_path / "Test.JSONL"))) == {r.RUN_TEST_SPLIT_PATH}


@pytest.mark.parametrize("name", ["audit.jsonl", "dev.json", "dev.jsonl.bak", "questions.jsonl", "x"])
def test_any_other_file_name_is_refused(tmp_path, name):
    assert codes(error_of(r.check_dev_path, tmp_path / name)) == {r.RUN_DEV_PATH_NAME}


def test_a_link_named_dev_that_points_to_the_test_file_is_refused(tmp_path):
    (tmp_path / "test.jsonl").write_text("x", encoding="utf-8")
    link = tmp_path / "dev.jsonl"
    try:
        link.symlink_to(tmp_path / "test.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    assert codes(error_of(r.check_dev_path, link)) == {r.RUN_TEST_SPLIT_PATH}


def test_the_path_check_does_not_touch_the_file(tmp_path, monkeypatch):
    opened = []
    real_open = Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy)
    r.check_dev_path(tmp_path / "dev.jsonl")
    assert codes(error_of(r.check_dev_path, tmp_path / "test.jsonl")) == {r.RUN_TEST_SPLIT_PATH}
    assert opened == []


@pytest.mark.parametrize("value", [None, 3, b"dev.jsonl", ["dev.jsonl"]])
def test_a_wrongly_typed_path_is_a_type_error(value):
    with pytest.raises(TypeError):
        r.check_dev_path(value)


# ---------- B: the pure gate ----------

def test_a_valid_dev_split_passes_and_is_split_without_reordering():
    result = build()
    assert [q.question_id for q in result.questions] == ["d1", "d2", "du1"]
    assert [q.question_id for q in result.answerable] == ["d1", "d2"]
    assert [q.question_id for q in result.unanswerable] == ["du1"]
    assert all(q.split == "dev" for q in result.questions)
    assert result.warnings == ()
    assert result.manifest is MANIFEST


def test_the_hashes_are_the_parsing_module_hashes_and_stable():
    first, second = build(), build()
    assert first.manifest_sha256 == parsing.manifest_hash(MANIFEST) == second.manifest_sha256
    assert first.dev_dataset_sha256 == parsing.dataset_hash(first.questions) == second.dev_dataset_sha256
    assert len(first.manifest_sha256) == len(first.dev_dataset_sha256) == 64


def test_the_dev_hash_changes_when_a_label_changes():
    other = jsonl([answerable("d1", quote="Beta clause two."), unanswerable("du1")])
    assert build(other).dev_dataset_sha256 != build(jsonl([answerable("d1"), unanswerable("du1")])).dev_dataset_sha256


def test_the_texts_are_read_only_and_a_copy():
    source = dict(TEXTS)
    result = build(texts=source)
    with pytest.raises(TypeError):
        result.texts["doc-a"] = "x"
    source["doc-a"] = "changed afterwards"
    assert result.texts["doc-a"] == TEXT_A


def test_the_result_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        build().questions = ()


# ---------- C: the test split is refused ----------

def test_a_test_split_record_in_the_dev_file_is_refused():
    line = jsonl([answerable("t1", question="Who may terminate the agreement early?", split="test")])
    assert "PARSE_SPLIT_MISMATCH" in codes(error_of(build, line))


def test_the_runner_check_for_a_foreign_split_is_independent_of_the_parser(monkeypatch):
    leaked = answerable("t1", question="Who may terminate the agreement early?", split="test")
    monkeypatch.setattr(r, "parse_questions", lambda text, split: (answerable("d1"), leaked))
    assert codes(error_of(build)) == {r.RUN_TEST_RECORD}


# ---------- D: empty and malformed dev files ----------

def test_an_empty_dev_file_is_refused():
    assert codes(error_of(build, "")) == {r.RUN_DEV_EMPTY}


def test_a_dev_file_with_only_a_blank_line_is_refused():
    assert codes(error_of(build, "\n")) == {"PARSE_BLANK_LINE"}


def test_a_malformed_record_is_refused():
    assert "PARSE_SYNTAX" in codes(error_of(build, "{not json}\n"))


# ---------- E: manifest ----------

def test_a_draft_manifest_is_refused():
    assert codes(error_of(build, manifest=dataclasses.replace(MANIFEST, stage="draft"))) == {r.RUN_MANIFEST_NOT_INGESTED}


def test_an_invalid_manifest_is_refused():
    assert codes(error_of(build, manifest=dataclasses.replace(MANIFEST, schema_version="nope"))) == {r.RUN_MANIFEST_INVALID}


# ---------- F: source texts ----------

def test_a_missing_text_is_refused():
    assert codes(error_of(build, texts={"doc-a": TEXT_A})) == {r.RUN_SOURCE_TEXT_MISSING}


def test_an_extra_text_is_refused():
    assert codes(error_of(build, texts={**TEXTS, "doc-z": "zzz"})) == {r.RUN_SOURCE_TEXT_UNKNOWN}


def test_an_altered_text_is_refused():
    assert "SOURCE_HASH_MISMATCH" in codes(error_of(build, texts={**TEXTS, "doc-a": TEXT_A.replace("Alpha", "Alpho")}))


def test_a_truncated_text_is_refused():
    found = codes(error_of(build, texts={**TEXTS, "doc-b": TEXT_B[:-8]}))
    assert {"SOURCE_HASH_MISMATCH", "SOURCE_LENGTH_MISMATCH"} <= found


def test_an_empty_text_is_refused():
    assert "SOURCE_TEXT_EMPTY" in codes(error_of(build, texts={**TEXTS, "doc-a": ""}))


def test_every_text_finding_is_reported_together():
    found = error_of(build, texts={"doc-b": TEXT_B[:-8], "doc-z": "zzz"})
    assert {r.RUN_SOURCE_TEXT_MISSING, r.RUN_SOURCE_TEXT_UNKNOWN, "SOURCE_HASH_MISMATCH"} <= codes(found)


# ---------- G: stale labels ----------

def test_a_stale_source_hash_is_refused():
    stale = dataclasses.replace(anchored("doc-a", TEXT_A, "Alpha clause one."), source_sha256="a" * 64)
    assert "DATASET_SPAN_HASH_MISMATCH" in codes(error_of(build, jsonl([answerable(span=stale)])))


def test_shifted_offsets_are_refused():
    good = anchored("doc-a", TEXT_A, "Alpha clause one.")
    shifted = dataclasses.replace(good, start=good.start + 1, end=good.end + 1)
    assert "ANCHOR_QUOTE_MISMATCH" in codes(error_of(build, jsonl([answerable(span=shifted)])))


def test_a_changed_quote_is_refused():
    good = anchored("doc-a", TEXT_A, "Alpha clause one.")
    changed = dataclasses.replace(good, quote="Alpha clause ONE.", quote_sha256=sha256_text("Alpha clause ONE."))
    assert "ANCHOR_QUOTE_MISMATCH" in codes(error_of(build, jsonl([answerable(span=changed)])))


def test_an_end_beyond_the_text_is_refused():
    good = anchored("doc-a", TEXT_A, "Alpha clause one.")
    beyond = dataclasses.replace(good, start=len(TEXT_A) - 2, end=len(TEXT_A) + 40)
    assert "ANCHOR_END_BEYOND_TEXT" in codes(error_of(build, jsonl([answerable(span=beyond)])))


def test_an_unknown_document_is_refused():
    ghost = dataclasses.replace(anchored("doc-a", TEXT_A, "Alpha clause one."), corpus_key="doc-ghost")
    assert "DATASET_SPAN_KEY_UNKNOWN" in codes(error_of(build, jsonl([answerable(span=ghost)])))


def test_a_duplicate_question_id_is_refused():
    both = jsonl([answerable("d1"), answerable("d1", "Which party pays the refund amount in the agreement?")])
    assert "DATASET_DUPLICATE_QUESTION_ID" in codes(error_of(build, both))


def test_an_exact_duplicate_question_text_is_refused():
    both = jsonl([answerable("d1"), answerable("d2")])
    assert "DATASET_DUPLICATE_QUESTION_TEXT" in codes(error_of(build, both))


# ---------- H: privacy ----------

def test_no_error_holds_a_quote_a_question_or_document_text():
    good = anchored("doc-a", TEXT_A, "Alpha clause one.")
    bad = dataclasses.replace(good, quote=CANARY, quote_sha256=sha256_text(CANARY))
    question = answerable(span=bad, question="What " + CANARY + " says about the first clause?")
    error = error_of(build, jsonl([question]), texts={**TEXTS, "doc-b": TEXT_B + CANARY})
    blob = str(error) + " ".join(f"{i.code} {i.location} {i.message}" for i in error.issues)
    assert CANARY not in blob and TEXT_A not in blob and TEXT_B not in blob


# ---------- I: warnings ----------

def test_a_similar_question_is_a_warning_not_a_failure():
    similar = jsonl([
        answerable("d1", "What does the first clause say about the alpha item?"),
        answerable("d2", "What does the first clause say about the alpha items?", "doc-b", TEXT_B, "Delta section four."),
    ])
    result = build(similar, threshold=0.5)
    assert [i.code for i in result.warnings] == ["DATASET_SIMILAR_QUESTION"]
    assert all(i.severity == s.SEVERITY_WARNING for i in result.warnings)


# ---------- J: arguments ----------

@pytest.mark.parametrize("kwargs", [
    {"dev_text": None}, {"dev_text": b"x"}, {"manifest": None}, {"manifest": {}}, {"texts": None}, {"texts": ["a"]},
    {"threshold": None}, {"threshold": "0.6"}, {"threshold": True},
])
def test_wrong_argument_types_are_type_errors(kwargs):
    with pytest.raises(TypeError):
        build(**kwargs)


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
def test_a_threshold_outside_zero_to_one_is_rejected(threshold):
    with pytest.raises(ValueError):
        build(threshold=threshold)


def test_arguments_are_not_modified():
    texts = dict(TEXTS)
    build(texts=texts)
    assert texts == TEXTS


# ---------- K: the loader ----------

def manifest_record(manifest):
    record = dataclasses.asdict(manifest)
    for entry in record["documents"]:
        entry["extraction"] = {k: v for k, v in entry["extraction"].items() if v is not None}
    return record


def write_corpus(directory, texts=TEXTS):
    directory.mkdir()
    documents = []
    for key, text in texts.items():
        raw = text.encode("utf-8")
        (directory / f"{key}.txt").write_bytes(raw)
        documents.append(doc(key, text, file_bytes=raw))
    manifest = dataclasses.replace(MANIFEST, documents=tuple(documents))
    (directory / "manifest.json").write_text(json.dumps(manifest_record(manifest)), encoding="utf-8")
    return manifest


def make_dataset(tmp_path, dev=None):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dev.jsonl").write_bytes((good_dev() if dev is None else dev).encode("utf-8"))
    (dataset / "test.jsonl").write_bytes(b"THIS IS NOT JSON AND MUST NEVER BE READ\n")
    return dataset


class OpenSpy:
    """Records every file opened or read by name, to prove that test.jsonl is untouched."""

    def __init__(self, monkeypatch):
        self.names = []
        real_open, real_read_bytes, real_read_text = Path.open, Path.read_bytes, Path.read_text
        spy = self

        def wrap(real):
            def inner(path, *a, **k):
                spy.names.append(Path(path).name)
                return real(path, *a, **k)
            return inner

        monkeypatch.setattr(Path, "open", wrap(real_open))
        monkeypatch.setattr(Path, "read_bytes", wrap(real_read_bytes))
        monkeypatch.setattr(Path, "read_text", wrap(real_read_text))
        import builtins
        real_builtin = builtins.open

        def builtin_open(file, *a, **k):
            spy.names.append(Path(str(file)).name)
            return real_builtin(file, *a, **k)

        monkeypatch.setattr(builtins, "open", builtin_open)


def test_the_loader_reads_the_dev_split_and_never_opens_the_test_file(tmp_path, monkeypatch):
    write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    spy = OpenSpy(monkeypatch)
    result = r.load_dev_inputs(dataset / "dev.jsonl", tmp_path / "corpus", near_duplicate_threshold=THRESHOLD)
    assert [q.question_id for q in result.questions] == ["d1", "d2", "du1"]
    assert "dev.jsonl" in spy.names
    assert "test.jsonl" not in [name.casefold() for name in spy.names]


def test_the_loader_refuses_the_test_file_without_reading_it(tmp_path, monkeypatch):
    write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    spy = OpenSpy(monkeypatch)
    assert codes(error_of(r.load_dev_inputs, dataset / "test.jsonl", tmp_path / "corpus")) == {r.RUN_TEST_SPLIT_PATH}
    assert "test.jsonl" not in [name.casefold() for name in spy.names]


def test_the_loader_matches_the_pure_gate(tmp_path):
    manifest = write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    loaded = r.load_dev_inputs(dataset / "dev.jsonl", tmp_path / "corpus", near_duplicate_threshold=THRESHOLD)
    pure = r.build_dev_inputs(good_dev(), manifest, TEXTS, near_duplicate_threshold=THRESHOLD)
    assert loaded.dev_dataset_sha256 == pure.dev_dataset_sha256
    assert loaded.manifest_sha256 == pure.manifest_sha256


def test_the_loader_uses_a_default_threshold(tmp_path):
    write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    assert len(r.load_dev_inputs(dataset / "dev.jsonl", tmp_path / "corpus").questions) == 3


def test_a_missing_dev_file_is_refused(tmp_path):
    write_corpus(tmp_path / "corpus")
    assert codes(error_of(r.load_dev_inputs, tmp_path / "dev.jsonl", tmp_path / "corpus")) == {r.RUN_DEV_FILE_UNREADABLE}


def test_a_dev_file_that_is_not_utf8_is_refused(tmp_path):
    write_corpus(tmp_path / "corpus")
    (tmp_path / "dev.jsonl").write_bytes(b"\xff\xfe\x00")
    assert codes(error_of(r.load_dev_inputs, tmp_path / "dev.jsonl", tmp_path / "corpus")) == {r.RUN_DEV_FILE_UNREADABLE}


def test_a_missing_corpus_is_refused(tmp_path):
    dataset = make_dataset(tmp_path)
    assert codes(error_of(r.load_dev_inputs, dataset / "dev.jsonl", tmp_path / "nowhere")) == {r.RUN_CORPUS_UNAVAILABLE}


def test_a_corpus_file_that_no_longer_matches_the_manifest_is_refused(tmp_path):
    write_corpus(tmp_path / "corpus")
    (tmp_path / "corpus" / "doc-a.txt").write_bytes(b"tampered text")
    dataset = make_dataset(tmp_path)
    assert codes(error_of(r.load_dev_inputs, dataset / "dev.jsonl", tmp_path / "corpus")) == {r.RUN_CORPUS_UNAVAILABLE}


def test_the_loader_does_not_modify_any_file(tmp_path):
    write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    r.load_dev_inputs(dataset / "dev.jsonl", tmp_path / "corpus", near_duplicate_threshold=THRESHOLD)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


# ---------- L: the real repository dev split ----------

REAL_DEV = PROJECT_ROOT / "eval" / "dataset" / "dev.jsonl"
REAL_CORPUS = PROJECT_ROOT / "eval" / "corpus"


@pytest.mark.skipif(not (REAL_DEV.is_file() and (REAL_CORPUS / "manifest.json").is_file()), reason="eval data not present")
def test_the_real_dev_split_passes_the_gate_without_opening_the_test_file(monkeypatch):
    spy = OpenSpy(monkeypatch)
    result = r.load_dev_inputs(REAL_DEV, REAL_CORPUS)
    assert result.questions and len(result.answerable) + len(result.unanswerable) == len(result.questions)
    assert all(q.split == "dev" for q in result.questions)
    assert set(result.texts) == {d.corpus_key for d in result.manifest.documents}
    assert len(result.dev_dataset_sha256) == len(result.manifest_sha256) == 64
    assert "test.jsonl" not in [name.casefold() for name in spy.names]


# ---------- M: import isolation ----------

def test_importing_the_module_loads_no_database_retrieval_or_labelling_code():
    code = (
        "import sys\n"
        "import app.rag.evaluation.runner_inputs\n"
        "bad = [m for m in ('app.rag.evaluation.labelling', 'app.rag.retrieval', 'app.rag.embedder', "
        "'app.core.processing', 'sqlalchemy', 'torch') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""


# ---------- N: the loader needs no application config or database ----------

_APP_MODULES = (
    "app.core.database", "app.core.config", "app.core.processing", "app.core.storage",
    "app.models", "sqlalchemy",
)


def _clean_env(tmp_path):
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_clean(tmp_path, code):
    # cwd is a temporary folder, so no repository .env can supply credentials.
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=_clean_env(tmp_path), capture_output=True, text=True, check=False
    )


def _loader_probe(dev, corpus):
    return (
        "import sys\n"
        "from app.rag.evaluation.runner_inputs import load_dev_inputs\n"
        f"res = load_dev_inputs({str(dev)!r}, {str(corpus)!r}, near_duplicate_threshold=0.6)\n"
        f"bad = [m for m in {_APP_MODULES!r} if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]\n"
        "print(len(res.questions), ','.join(bad))\n"
    )


def test_the_loader_runs_without_database_credentials_and_loads_no_app_config_or_engine(tmp_path):
    write_corpus(tmp_path / "corpus")
    dataset = make_dataset(tmp_path)
    done = _run_clean(tmp_path, _loader_probe(dataset / "dev.jsonl", tmp_path / "corpus"))
    assert done.returncode == 0, done.stderr[-800:]
    count, _, bad = done.stdout.strip().partition(" ")
    assert int(count) == len(good_dev().splitlines()) and bad == ""


@pytest.mark.skipif(not (REAL_DEV.is_file() and (REAL_CORPUS / "manifest.json").is_file()), reason="eval data not present")
def test_the_real_dev_split_loads_without_credentials_or_database_modules(tmp_path):
    done = _run_clean(tmp_path, _loader_probe(REAL_DEV, REAL_CORPUS))
    assert done.returncode == 0, done.stderr[-800:]
    count, _, bad = done.stdout.strip().partition(" ")
    assert int(count) > 0 and bad == ""


def test_the_extraction_module_imports_no_app_config_and_is_shared_with_processing(tmp_path):
    done = _run_clean(
        tmp_path,
        "import sys\n"
        "import app.core.text_extraction as t\n"
        f"bad = [m for m in {_APP_MODULES!r} if m in sys.modules]\n"
        "print('bad=' + ','.join(bad))\n",
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip() == "bad="
    env = {**_clean_env(tmp_path), "DATABASE_URL": "sqlite:///:memory:", "SECRET_KEY": "x" * 40}
    shared = subprocess.run(
        [sys.executable, "-c",
         ("import app.core.processing as p, app.core.text_extraction as t\n"
          "print(p.extract_text is t.extract_text, p.ExtractionError is t.ExtractionError)")],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert shared.returncode == 0, shared.stderr[-800:]
    assert shared.stdout.strip() == "True True"


def test_the_clean_environment_really_fails_for_application_config(tmp_path):
    # Guards the probe itself: without credentials, importing the application config must fail here.
    done = _run_clean(tmp_path, "import app.core.config")
    assert done.returncode != 0
