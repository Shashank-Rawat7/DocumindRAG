"""Inputs and the fail-closed gate of the evaluation runner (E1.7.1).

This module is pure: it opens no database, runs no retrieval and computes no metric. It prepares
what a development-split validity pilot may read, and it refuses everything else.

Development split only. The runner is given the path of ``dev.jsonl`` and nothing else from the
dataset folder. ``test.jsonl`` is never opened, never parsed and never validated here:

* a path whose resolved file name is not ``dev.jsonl`` is refused, and ``test.jsonl`` (also through a
  link named ``dev.jsonl``) has its own error (``RUN_TEST_SPLIT_PATH``). Only the path is looked at;
  no file other than the one named is touched;
* a record whose ``split`` is not ``dev`` is refused by ``parsing.parse_questions`` and checked again
  here (``RUN_TEST_RECORD``).

Because the test split is not read, the dev/test evidence-collision check of ``validate_dataset`` is
not run here (it is part of the E1.6 ``check``, which sees both files and has passed). The gate
validates the dev split against an empty test split.

The gate fails closed, with one ``EvaluationDataError`` that carries every finding:

* the manifest must be valid and in the ``ingested`` stage;
* a verified text must be supplied for every manifest document and for no other key, and each text
  must pass ``anchoring.verify_source_text`` (recorded SHA-256 and length): a missing, extra,
  incomplete, altered or empty text is refused;
* the dev file must hold at least one question, and every record must pass ``validate_dataset`` run
  with the texts, so a stale source hash, a wrong offset or a quote that no longer matches the text
  (``resolve_question``) is refused. Warnings (for example a similar question) do not stop the gate;
  they are returned.

Nothing is repaired, relocated or normalised. The result carries the SHA-256 of the manifest and of
the dev questions (``parsing.manifest_hash`` and ``parsing.dataset_hash``) so a run can be bound to
exactly these labels. No issue and no message holds a quote, a question or any document text.

``build_dev_inputs`` is the pure function (strings and mappings in, a result out) and is what the
tests exercise most. ``load_dev_inputs`` only reads the dev file and the corpus through the existing
``labelling`` loaders and then calls it. Those loaders extract text through ``app.core.text_extraction``,
which imports no application settings, database engine or model, so the loader needs neither
``DATABASE_URL`` nor ``SECRET_KEY`` and creates no engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from app.rag.evaluation.anchoring import verify_source_text
from app.rag.evaluation.dataset import validate_dataset
from app.rag.evaluation.parsing import dataset_hash, manifest_hash, parse_questions
from app.rag.evaluation.schema import (
    SEVERITY_ERROR,
    EvaluationDataError,
    Issue,
    Manifest,
    Question,
)
from app.rag.evaluation.validation import validate_manifest

DEV_FILE_NAME = "dev.jsonl"
TEST_FILE_NAME = "test.jsonl"

RUN_TEST_SPLIT_PATH = "RUN_TEST_SPLIT_PATH"
RUN_DEV_PATH_NAME = "RUN_DEV_PATH_NAME"
RUN_DEV_FILE_UNREADABLE = "RUN_DEV_FILE_UNREADABLE"
RUN_DEV_EMPTY = "RUN_DEV_EMPTY"
RUN_TEST_RECORD = "RUN_TEST_RECORD"
RUN_MANIFEST_INVALID = "RUN_MANIFEST_INVALID"
RUN_MANIFEST_NOT_INGESTED = "RUN_MANIFEST_NOT_INGESTED"
RUN_SOURCE_TEXT_MISSING = "RUN_SOURCE_TEXT_MISSING"
RUN_SOURCE_TEXT_UNKNOWN = "RUN_SOURCE_TEXT_UNKNOWN"
RUN_CORPUS_UNAVAILABLE = "RUN_CORPUS_UNAVAILABLE"

_MESSAGES = {
    RUN_TEST_SPLIT_PATH: "the test split must not be read: the path names the test file",
    RUN_DEV_PATH_NAME: "the runner reads only a file named dev.jsonl",
    RUN_DEV_FILE_UNREADABLE: "the dev file is missing or cannot be read as UTF-8",
    RUN_DEV_EMPTY: "the dev split holds no question",
    RUN_TEST_RECORD: "a record in the dev file does not belong to the dev split",
    RUN_MANIFEST_INVALID: "the corpus manifest is invalid",
    RUN_MANIFEST_NOT_INGESTED: "the corpus manifest is not in the ingested stage",
    RUN_SOURCE_TEXT_MISSING: "a manifest document has no source text",
    RUN_SOURCE_TEXT_UNKNOWN: "a source text has no manifest document",
    RUN_CORPUS_UNAVAILABLE: "the corpus manifest or texts could not be loaded",
}

_THRESHOLD_MESSAGE = "near_duplicate_threshold must be a float or int"


def _issue(code: str, location: str, severity: str = SEVERITY_ERROR) -> Issue:
    return Issue(code, location, _MESSAGES[code], severity)


@dataclass(frozen=True, slots=True)
class DevInputs:
    """A development split that passed the gate, with the verified corpus it is bound to.

    ``questions`` are the dev questions in file order; ``answerable`` and ``unanswerable`` split
    them without reordering. ``texts`` is read-only. ``warnings`` are findings that did not stop
    the gate (for example similar questions). The two hashes bind a run to these exact inputs.
    """

    manifest: Manifest
    texts: Mapping[str, str]
    questions: tuple[Question, ...]
    answerable: tuple[Question, ...]
    unanswerable: tuple[Question, ...]
    manifest_sha256: str
    dev_dataset_sha256: str
    warnings: tuple[Issue, ...]


def check_dev_path(path: str | Path) -> Path:
    """The resolved dev file path, or ``EvaluationDataError`` for anything but ``dev.jsonl``.

    Only the path is examined (links are resolved, nothing is read), so ``test.jsonl`` is never
    opened by this check.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or Path")
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        raise EvaluationDataError([_issue(RUN_DEV_PATH_NAME, "dev_path")]) from None
    name = resolved.name.casefold()
    if name == TEST_FILE_NAME:
        raise EvaluationDataError([_issue(RUN_TEST_SPLIT_PATH, "dev_path")])
    if name != DEV_FILE_NAME:
        raise EvaluationDataError([_issue(RUN_DEV_PATH_NAME, "dev_path")])
    return resolved


def _text_issues(manifest: Manifest, texts: Mapping[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    keys = set()
    for index, entry in enumerate(manifest.documents):
        keys.add(entry.corpus_key)
        location = f"manifest.documents[{index}]"
        if entry.corpus_key not in texts:
            issues.append(_issue(RUN_SOURCE_TEXT_MISSING, location))
            continue
        for item in verify_source_text(entry, texts[entry.corpus_key]):
            issues.append(Issue(item.code, f"{location}.text", item.message, item.severity))
    for index, _ in enumerate(sorted(key for key in texts if key not in keys)):
        issues.append(_issue(RUN_SOURCE_TEXT_UNKNOWN, f"texts[{index}]"))
    return issues


def build_dev_inputs(
    dev_text: str,
    manifest: Manifest,
    texts: Mapping[str, str],
    *,
    near_duplicate_threshold: float,
) -> DevInputs:
    """Run the gate on the text of ``dev.jsonl``, a manifest and the manifest's source texts.

    ``near_duplicate_threshold`` is required (see ``near_duplicates``). Raises
    ``EvaluationDataError`` (all findings in ``.issues``) when anything fails; ``TypeError`` for a
    wrongly typed argument. Never reads a file and never sees the test split.
    """
    if type(dev_text) is not str:
        raise TypeError("dev_text must be a str")
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    if not isinstance(texts, Mapping):
        raise TypeError("texts must be a Mapping")
    if isinstance(near_duplicate_threshold, bool) or not isinstance(near_duplicate_threshold, (int, float)):
        raise TypeError(_THRESHOLD_MESSAGE)

    # The manifest first: nothing else is meaningful against an invalid or unfinished corpus.
    if any(item.severity == SEVERITY_ERROR for item in validate_manifest(manifest)):
        raise EvaluationDataError([_issue(RUN_MANIFEST_INVALID, "manifest")])
    if manifest.stage != "ingested":
        raise EvaluationDataError([_issue(RUN_MANIFEST_NOT_INGESTED, "manifest.stage")])

    issues = _text_issues(manifest, texts)
    if issues:
        raise EvaluationDataError(issues)

    questions = parse_questions(dev_text, split="dev")  # a record of another split is refused here
    if not questions:
        raise EvaluationDataError([_issue(RUN_DEV_EMPTY, "dev")])
    if any(question.split != "dev" for question in questions):
        raise EvaluationDataError([_issue(RUN_TEST_RECORD, "dev")])

    # The test split is deliberately empty: it is not read (see the module documentation).
    found = validate_dataset(
        {"dev": questions, "test": ()}, manifest, near_duplicate_threshold=near_duplicate_threshold, texts=texts
    )
    if any(item.severity == SEVERITY_ERROR for item in found):
        raise EvaluationDataError(found)

    return DevInputs(
        manifest=manifest,
        texts=MappingProxyType(dict(texts)),
        questions=questions,
        answerable=tuple(q for q in questions if q.answerable),
        unanswerable=tuple(q for q in questions if not q.answerable),
        manifest_sha256=manifest_hash(manifest),
        dev_dataset_sha256=dataset_hash(questions),
        warnings=tuple(found),
    )


def load_dev_inputs(
    dev_path: str | Path,
    corpus_dir: str | Path,
    *,
    near_duplicate_threshold: float | None = None,
) -> DevInputs:
    """Read ``dev.jsonl`` and the corpus, then run the gate (``build_dev_inputs``).

    ``dev_path`` must be the dev file itself (``check_dev_path``). The manifest and the source texts
    come from the existing ``labelling`` loaders, which verify every text against the manifest and
    never touch the database and load no application config. ``near_duplicate_threshold`` defaults to the labelling tool's value.
    """
    # Imported here so that importing this module stays light and the pure gate has no corpus tooling.
    from app.rag.evaluation import labelling

    path = check_dev_path(dev_path)
    try:
        dev_text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise EvaluationDataError([_issue(RUN_DEV_FILE_UNREADABLE, "dev_path")]) from None
    try:
        manifest = labelling.load_manifest(Path(corpus_dir))
        texts = labelling.load_texts(Path(corpus_dir), manifest)
    except EvaluationDataError:
        raise
    except (labelling.LabellingError, OSError):
        raise EvaluationDataError([_issue(RUN_CORPUS_UNAVAILABLE, "corpus")]) from None
    threshold = (
        labelling.PROVISIONAL_NEAR_DUPLICATE_THRESHOLD if near_duplicate_threshold is None else near_duplicate_threshold
    )
    return build_dev_inputs(dev_text, manifest, texts, near_duplicate_threshold=threshold)
