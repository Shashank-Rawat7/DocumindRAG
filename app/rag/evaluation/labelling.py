"""Quote-anchored labelling of the evaluation dataset (E1.6).

This module adds no second anchoring mechanism. A span is built only from an EXACT quote that is
found in the manifest-verified extracted text of one corpus document, and the result is the E1.2
``Span`` (corpus key, source hash from the manifest, half-open offsets, quote, quote hash). Every
write is then checked by the existing ``validate_dataset`` (which runs ``resolve_question`` against
the verified texts), so a stale hash, a wrong offset or an invalid record is refused and nothing is
written.

Fail-closed rules:

* The corpus texts come from the repository files through ``extract_text`` and must pass
  ``verify_source_text`` against the manifest. The database is never read.
* A quote that is absent is an error. A quote that occurs more than once is an error unless the
  caller names the occurrence (1-based, counting overlapping matches). No span is ever searched
  for again, adjusted or repaired.
* A batch is all-or-nothing: if any record or the combined dataset has an error, no file changes.
* Writing is recoverable: the question files are written first and the review log second. If either
  write fails, the question files are put back as they were and the log is cut back to its old
  length, so a question never exists without its recorded review. If a restore itself fails, the
  error says so and ``check`` reports the leftover as an unreviewed question.
* A reviewer identity is never taken from a spec file. The human evidence reviewer, the Hindi
  reviewer and the semantic second reader are given by the person running the tool, and each
  addition is recorded in the append-only ``audit.jsonl``. Adding a question is therefore the promotion
  gate: it needs a named person for every review it records.
* ``check_all`` is structural validation only (records, anchoring, event schemas, audit integrity). It does
  not fail a record because a human review is still pending and never marks anything reviewed.
  ``review_readiness`` reports the human-review state separately: missing or stale evidence reviews,
  Hindi questions without a native-speaker review, and unanswerable questions with an AI-only second
  reader. A ``label_review`` carries a hash of the reviewed question text, expected answer and evidence
  (``review_content_sha256``); a later change to any of them makes that review stale until a person
  records a new one, which supersedes the old one. History is never rewritten.

* A reviewer name is refused when it is blank or is an obvious placeholder or AI name (``dryrun``,
  ``tbd``, ``chatgpt``, ``Claude Sonnet`` ...). A name is an attribution made by the person running the
  tool; the software does not verify who anyone is. The tool never sets a Hindi review from the question language: a Hindi question is added only when
  the person running it names the Hindi reviewer, and that review is also written to the log.
* ``audit.jsonl`` accepts only the event types below and validates every one of them, so an unknown
  or malformed event is an error and not silently ignored. ``label_review`` and ``hindi_review`` are
  human actions by a named reviewer. ``label_correction`` records a change made after a review and
  says explicitly whether a human did it; it is never a review. ``audit_sample`` is a procedure
  record and ``audit_result`` is the blind auditor's comparison.

No error raised here and no issue reported here contains a quote or other document text.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.text_extraction import ExtractionError, extract_text
from app.rag.chunking import sha256_text
from app.rag.evaluation.anchoring import verify_source_text
from app.rag.evaluation.dataset import is_valid_iso_date, validate_dataset
from app.rag.evaluation.dataset_metadata import parse_dataset_metadata, validate_dataset_metadata
from app.rag.evaluation.intervals import overlaps
from app.rag.evaluation.parsing import parse_manifest, parse_questions
from app.rag.evaluation.schema import (
    AUTHOR_TYPES,
    DIFFICULTIES,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    EvaluationDataError,
    EvidenceUnit,
    Issue,
    Manifest,
    Question,
    RetrievalAssistance,
    SemanticReview,
    Span,
)
from app.rag.evaluation.validation import validate_manifest

DEV_FILE = "dev.jsonl"
TEST_FILE = "test.jsonl"
METADATA_FILE = "dataset.json"
AUDIT_FILE = "audit.jsonl"

# Provisional: the real value is calibrated on the actual dev questions (E1.2d). Matches are warnings.
PROVISIONAL_NEAR_DUPLICATE_THRESHOLD = 0.6

AUDIT_PERCENT = 20
AUDIT_MINIMUM = 10

EVENT_LABEL_REVIEW = "label_review"
EVENT_AUDIT_SAMPLE = "audit_sample"
EVENT_AUDIT_RESULT = "audit_result"
EVENT_HINDI_REVIEW = "hindi_review"
EVENT_LABEL_CORRECTION = "label_correction"
# An independent AI review of an unanswerable question is recorded as an ``ai_semantic_review`` event. It has
# no reviewer, model or date unless the person recording it knows them, so it can never pass for a person.
EVENT_AI_SEMANTIC_REVIEW = "ai_semantic_review"
AI_SEMANTIC_REVIEW_TYPE = "independent_ai_semantic_review"
EVENT_TYPES = (EVENT_LABEL_REVIEW, EVENT_HINDI_REVIEW, EVENT_LABEL_CORRECTION, EVENT_AI_SEMANTIC_REVIEW,
               EVENT_AUDIT_SAMPLE, EVENT_AUDIT_RESULT)

AUDIT_SAMPLE_REPEATED = "AUDIT_SAMPLE_REPEATED"
AUDIT_UNKNOWN_QUESTION = "AUDIT_UNKNOWN_QUESTION"
SEMANTIC_REVIEWER_NOT_A_PERSON = "SEMANTIC_REVIEWER_NOT_A_PERSON"
AUDIT_RESULT_MISSING = "AUDIT_RESULT_MISSING"
AUDIT_DISAGREEMENT_UNRESOLVED = "AUDIT_DISAGREEMENT_UNRESOLVED"
AUDIT_EVENT_UNKNOWN = "AUDIT_EVENT_UNKNOWN"
AUDIT_EVENT_INVALID = "AUDIT_EVENT_INVALID"
AUDIT_CORRECTION_MISMATCH = "AUDIT_CORRECTION_MISMATCH"


# Obvious placeholders and AI names. Compared case-insensitively. A reviewer name is an attribution made by
# the person running the tool; nothing here verifies identity. Single-word real names stay valid (including
# "Claude" or "Ai"): the model name is refused only together with a model word ("Claude Sonnet").
_NOT_A_PERSON = frozenset({
    "dryrun", "dry-run", "dry_run", "dry run", "placeholder", "tbd", "todo", "unknown", "n/a", "none", "null",
    "pending", "anonymous", "test", "chatgpt", "gpt", "llm", "assistant", "ai assistant", "ai-assisted", "ai reviewer",
})
_AI_TOKENS = frozenset({"chatgpt", "gpt", "llm", "anthropic", "openai", "copilot", "dryrun", "placeholder"})
_MODEL_WORDS = frozenset({"ai", "assistant", "code", "sonnet", "opus", "haiku", "fable", "mythos", "model", "llm",
                          "anthropic"})


def reviewer_problem(name: object) -> str | None:
    """Why ``name`` cannot be recorded as a reviewer, or ``None`` if it can. Attribution only: no identity check."""
    if type(name) is not str or name.strip() == "":
        return "must be a non-blank name"
    folded = " ".join(name.casefold().split())
    tokens = {t for t in re.split(r"[^a-z0-9]+", folded) if t}
    if (folded in _NOT_A_PERSON or tokens & _AI_TOKENS
            or (tokens & {"claude", "ai"} and tokens & (_MODEL_WORDS - {"ai"}))):
        return "must be the name of a person, not a placeholder or an AI name"
    return None

_CORPUS_SUFFIXES = (".md", ".txt", ".pdf", ".docx")


class LabellingError(ValueError):
    """A labelling request that cannot be carried out. Messages hold no document text."""


# ---------- corpus ----------

def load_manifest(corpus_dir: Path) -> Manifest:
    """The published manifest, which must be valid and in the ``ingested`` stage."""
    path = Path(corpus_dir) / "manifest.json"
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError:
        raise LabellingError("the corpus manifest cannot be read") from None
    except UnicodeDecodeError:
        raise LabellingError("the corpus manifest is not valid UTF-8") from None
    manifest = parse_manifest(text)
    errors = [item for item in validate_manifest(manifest) if item.severity == SEVERITY_ERROR]
    if errors:
        raise EvaluationDataError(errors)
    if manifest.stage != "ingested":
        raise LabellingError("the corpus manifest is not in the ingested stage")
    return manifest


def load_texts(corpus_dir: Path, manifest: Manifest) -> dict[str, str]:
    """Extracted text of every manifest document, each verified against the manifest.

    A corpus file is matched to its entry by ``source_file_sha256`` (the manifest has no file
    name). Raises ``LabellingError`` if a file is missing or its text fails verification.
    """
    directory = Path(corpus_dir)
    by_hash: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _CORPUS_SUFFIXES:
            by_hash.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
    texts: dict[str, str] = {}
    for entry in manifest.documents:
        path = by_hash.get(entry.source_file_sha256)
        if path is None:
            raise LabellingError(f"no corpus file matches source_file_sha256 of {entry.corpus_key}")
        try:
            text = extract_text(path)
        except ExtractionError:
            raise LabellingError(f"the text of {entry.corpus_key} cannot be extracted") from None
        problems = verify_source_text(entry, text)
        if problems:
            codes = ", ".join(sorted({item.code for item in problems}))
            raise LabellingError(f"the text of {entry.corpus_key} fails verification ({codes})")
        texts[entry.corpus_key] = text
    return texts


# ---------- quotes and spans ----------

def find_quote(text: str, quote: str, occurrence: int | None = None) -> tuple[int, int]:
    """``(start, end)`` of an exact quote in ``text``.

    Exactly one match is accepted without ``occurrence``. With several matches (overlapping ones
    count) ``occurrence`` (1-based) must choose one. Nothing is normalised.
    """
    if type(text) is not str or type(quote) is not str:
        raise TypeError("text and quote must be str")
    if quote == "":
        raise LabellingError("the quote is empty")
    starts: list[int] = []
    position = text.find(quote)
    while position != -1:
        starts.append(position)
        position = text.find(quote, position + 1)
    if not starts:
        raise LabellingError("the quote does not occur in the document")
    if occurrence is None:
        if len(starts) > 1:
            raise LabellingError(f"the quote occurs {len(starts)} times; give an occurrence from 1 to {len(starts)}")
        chosen = starts[0]
    else:
        if type(occurrence) is not int or not 1 <= occurrence <= len(starts):
            raise LabellingError(f"occurrence must be an integer from 1 to {len(starts)}")
        chosen = starts[occurrence - 1]
    return chosen, chosen + len(quote)


def make_span(manifest: Manifest, texts: Mapping[str, str], corpus_key: str, quote: str,
              occurrence: int | None = None) -> Span:
    """The anchored ``Span`` for an exact quote in one document."""
    entry = None
    if type(corpus_key) is str:
        entry = next((d for d in manifest.documents if d.corpus_key == corpus_key), None)
    if entry is None or corpus_key not in texts:
        raise LabellingError("the corpus key is not a document of the manifest")
    start, end = find_quote(texts[corpus_key], quote, occurrence)
    return Span(
        corpus_key=corpus_key,
        source_sha256=entry.extracted_text_sha256,
        start=start,
        end=end,
        quote=quote,
        quote_sha256=sha256_text(quote),
    )


# ---------- specs -> questions ----------

@dataclass(frozen=True, slots=True)
class Review:
    """Who reviewed what, given by the person running the tool and never read from a spec."""

    evidence_reviewer: str
    date: str
    hindi_reviewer: str | None = None
    semantic_reviewer: str | None = None
    # Where an independent AI review of unanswerable questions is documented. It is recorded as an
    # ``ai_semantic_review`` event and never fills ``semantic_review``, which is for a human reader only.
    ai_semantic_basis: str | None = None
    # The date of the native-speaker Hindi review, when it differs from ``date``. It is used only for the
    # ``hindi_review`` event; the evidence review and the human semantic review keep ``date``.
    hindi_review_date: str | None = None


_COMMON = ("question_id", "split", "question", "query_language", "author_type")
_ANSWERABLE = _COMMON + ("expected_answer", "source_language", "difficulty", "close_distractor", "notes", "units")
_ANSWERABLE_REQUIRED = _COMMON + ("expected_answer", "difficulty", "close_distractor", "units")
_UNANSWERABLE = _COMMON + ("unanswerable_type", "notes", "retrieval_assistance")
_UNANSWERABLE_REQUIRED = _COMMON + ("unanswerable_type",)
_ASSISTANCE = ("date", "method", "run_ref", "description")
_ASSISTANCE_REQUIRED = ("date", "description")


def read_spec(text: str) -> list[dict]:
    """One JSON object, or a JSON array of objects. Duplicate keys are refused."""
    def pairs(items: list) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise LabellingError("the spec has a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except json.JSONDecodeError:
        raise LabellingError("the spec is not valid JSON") from None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise LabellingError("the spec must be a JSON object or a non-empty array of objects")
    return value


def _check_keys(item: dict, allowed: Sequence[str], required: Sequence[str], where: str) -> None:
    unknown = [key for key in item if key not in allowed]
    if unknown:
        raise LabellingError(f"{where} has {len(unknown)} unknown field(s)")
    missing = [key for key in required if key not in item]
    if missing:
        raise LabellingError(f"{where} is missing: {', '.join(missing)}")


def _units(item: dict, manifest: Manifest, texts: Mapping[str, str], where: str) -> tuple[EvidenceUnit, ...]:
    raw_units = item["units"]
    if not isinstance(raw_units, list):
        raise LabellingError(f"{where}.units must be an array")
    units = []
    for u, raw in enumerate(raw_units):
        at = f"{where}.units[{u}]"
        if not isinstance(raw, dict):
            raise LabellingError(f"{at} must be an object")
        _check_keys(raw, ("unit_id", "spans"), ("unit_id", "spans"), at)
        if not isinstance(raw["spans"], list):
            raise LabellingError(f"{at}.spans must be an array")
        spans = []
        for s, raw_span in enumerate(raw["spans"]):
            here = f"{at}.spans[{s}]"
            if not isinstance(raw_span, dict):
                raise LabellingError(f"{here} must be an object")
            _check_keys(raw_span, ("corpus_key", "quote", "occurrence"), ("corpus_key", "quote"), here)
            try:
                spans.append(make_span(manifest, texts, raw_span["corpus_key"], raw_span["quote"],
                                       raw_span.get("occurrence")))
            except LabellingError as error:
                raise LabellingError(f"{here}: {error}") from None
        units.append(EvidenceUnit(unit_id=raw["unit_id"], spans=tuple(spans)))
    return tuple(units)


def _hindi_fields(language: object, review: Review, where: str) -> tuple[bool, str | None]:
    if language == "hi":
        if review.hindi_reviewer is None or review.hindi_reviewer.strip() == "":
            raise LabellingError(f"{where}: a Hindi query needs --hindi-reviewer")
        problem = reviewer_problem(review.hindi_reviewer)
        if problem is not None:
            raise LabellingError(f"{where}: --hindi-reviewer {problem}")
        return True, review.hindi_reviewer
    return False, None


def build_answerable(item: dict, manifest: Manifest, texts: Mapping[str, str], review: Review,
                     where: str = "spec[0]") -> Question:
    _check_keys(item, _ANSWERABLE, _ANSWERABLE_REQUIRED, where)
    units = _units(item, manifest, texts, where)
    source_language = item.get("source_language")
    if source_language is None:
        languages = {
            d.language for unit in units for span in unit.spans for d in manifest.documents
            if d.corpus_key == span.corpus_key
        }
        if len(languages) != 1:
            raise LabellingError(f"{where}: source_language cannot be derived from the spans")
        source_language = next(iter(languages))
    reviewed, reviewer = _hindi_fields(item["query_language"], review, where)
    return Question(
        question_id=item["question_id"],
        split=item["split"],
        answerable=True,
        question=item["question"],
        query_language=item["query_language"],
        author_type=item["author_type"],
        hindi_reviewed=reviewed,
        hindi_reviewer=reviewer,
        notes=item.get("notes"),
        expected_answer=item["expected_answer"],
        units=units,
        source_language=source_language,
        difficulty=item["difficulty"],
        close_distractor=item["close_distractor"],
    )


def build_unanswerable(item: dict, review: Review, where: str = "spec[0]") -> Question:
    _check_keys(item, _UNANSWERABLE, _UNANSWERABLE_REQUIRED, where)
    human_reader = review.semantic_reviewer is not None
    if human_reader:
        problem = reviewer_problem(review.semantic_reviewer)
        if problem is not None:
            raise LabellingError(f"{where}: --semantic-reviewer {problem}")
    elif review.ai_semantic_basis is None or review.ai_semantic_basis.strip() == "":
        raise LabellingError(
            f"{where}: an unanswerable question needs a human --semantic-reviewer or an --ai-semantic-basis")
    assistance = None
    raw = item.get("retrieval_assistance")
    if raw is not None:
        if not isinstance(raw, dict):
            raise LabellingError(f"{where}.retrieval_assistance must be an object")
        _check_keys(raw, _ASSISTANCE, _ASSISTANCE_REQUIRED, f"{where}.retrieval_assistance")
        assistance = RetrievalAssistance(
            date=raw["date"], method=raw.get("method"), run_ref=raw.get("run_ref"), description=raw["description"]
        )
    reviewed, reviewer = _hindi_fields(item["query_language"], review, where)
    return Question(
        question_id=item["question_id"],
        split=item["split"],
        answerable=False,
        question=item["question"],
        query_language=item["query_language"],
        author_type=item["author_type"],
        hindi_reviewed=reviewed,
        hindi_reviewer=reviewer,
        notes=item.get("notes"),
        unanswerable_type=item["unanswerable_type"],
        semantic_review=SemanticReview(
            reviewer=review.semantic_reviewer, date=review.date, result="confirmed_unanswerable"
        ) if human_reader else None,
        retrieval_assistance=assistance,
    )


# ---------- dataset files ----------

@dataclass(frozen=True, slots=True)
class Dataset:
    dev: tuple[Question, ...] = ()
    test: tuple[Question, ...] = ()

    def split(self, name: str) -> tuple[Question, ...]:
        return self.dev if name == "dev" else self.test

    def questions(self) -> tuple[Question, ...]:
        return self.dev + self.test


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError):
        raise LabellingError(f"{path.name} cannot be read as UTF-8") from None


def load_dataset(dataset_dir: Path) -> Dataset:
    """Both question files. A file that does not exist yet is an empty split."""
    directory = Path(dataset_dir)
    return Dataset(
        dev=parse_questions(_read(directory / DEV_FILE), split="dev"),
        test=parse_questions(_read(directory / TEST_FILE), split="test"),
    )


def serialise(question: Question) -> str:
    """One JSONL line (no newline) that ``parse_questions`` reads back to an equal ``Question``."""
    return json.dumps(dataclasses.asdict(question), ensure_ascii=False, separators=(",", ":"))


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_split(dataset_dir: Path, name: str, questions: Sequence[Question]) -> None:
    body = "".join(serialise(q) + "\n" for q in questions)
    _atomic_write(Path(dataset_dir) / (DEV_FILE if name == "dev" else TEST_FILE), body.encode("utf-8"))


# ---------- review binding ----------

def review_content_sha256(question: Question) -> str:
    """Hash of what an evidence review covers: the question text, the expected answer and the evidence.

    The evidence is each unit's id and, for each span, its corpus key, source hash, offsets and quote hash
    (the quote text is covered by its hash). ``difficulty``, ``close_distractor`` and ``notes`` are not part
    of it, so relabelling a record does not make an evidence review stale.
    """
    payload = {
        "v": 1, "question": question.question, "expected_answer": question.expected_answer,
        "unanswerable_type": question.unanswerable_type,
        "units": [[u.unit_id, [[s.corpus_key, s.source_sha256, s.start, s.end, s.quote_sha256] for s in u.spans]]
                  for u in question.units],
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _review_event(question: Question, reviewer: str, date: str) -> dict:
    return {"event": EVENT_LABEL_REVIEW, "question_id": question.question_id, "reviewer": reviewer, "date": date,
            "author_type": question.author_type, "content_sha256": review_content_sha256(question)}


def record_evidence_review(dataset_dir: Path, question_ids: Sequence[str], reviewer: str, date: str) -> int:
    """Append a human evidence review of the questions as they are now. It supersedes any earlier review.

    The reviewer is a name given by the person running the tool (attribution, not verified). Nothing in the
    dataset or in the earlier events changes.
    """
    problem = reviewer_problem(reviewer)
    if problem is not None:
        raise LabellingError(f"--evidence-reviewer {problem}")
    if not is_valid_iso_date(date):
        raise LabellingError("the review date must be a real YYYY-MM-DD date")
    ids = list(question_ids)
    if not ids or len(set(ids)) != len(ids):
        raise LabellingError("name at least one question, each once")
    known = {q.question_id: q for q in load_dataset(dataset_dir).questions()}
    if any(i not in known for i in ids):
        raise LabellingError("a named question is not in the dataset")
    _append_events(Path(dataset_dir), [_review_event(known[i], reviewer, date) for i in ids])
    return len(ids)


def _ai_review_events(ids: Sequence[str], basis: str, date: str | None = None, note: str | None = None) -> list[dict]:
    events = []
    for i in ids:
        event = {"event": EVENT_AI_SEMANTIC_REVIEW, "question_id": i, "review_type": AI_SEMANTIC_REVIEW_TYPE,
                 "human_review": False, "result": "confirmed_unanswerable", "basis": basis}
        if date is not None:
            event["date"] = date
        if note is not None:
            event["note"] = note
        events.append(event)
    return events


def record_ai_semantic_review(dataset_dir: Path, question_ids: Sequence[str], basis: str,
                              date: str | None = None, note: str | None = None) -> int:
    """Append an ``ai_semantic_review`` event per question: provenance of an independent AI review.

    It names no reviewer and no model, and adds a date only if one is given, because none may be invented.
    It is not a human review and does not satisfy the human semantic-reader gate.
    """
    if type(basis) is not str or basis.strip() == "":
        raise LabellingError("--basis must say where the AI review is documented")
    if date is not None and not is_valid_iso_date(date):
        raise LabellingError("the date must be a real YYYY-MM-DD date")
    ids = list(question_ids)
    if not ids or len(set(ids)) != len(ids):
        raise LabellingError("name at least one question, each once")
    known = {q.question_id: q for q in load_dataset(dataset_dir).questions()}
    if any(i not in known or known[i].answerable for i in ids):
        raise LabellingError("each named question must be an unanswerable question in the dataset")
    _append_events(Path(dataset_dir), _ai_review_events(ids, basis, date, note))
    return len(ids)


# ---------- audit log ----------

def load_events(dataset_dir: Path) -> list[dict]:
    events = []
    for number, line in enumerate(_read(Path(dataset_dir) / AUDIT_FILE).split("\n"), start=1):
        if line.strip() == "":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            raise LabellingError(f"{AUDIT_FILE} line {number} is not valid JSON") from None
        if not isinstance(event, dict) or type(event.get("event")) is not str:
            raise LabellingError(f"{AUDIT_FILE} line {number} is not an event record")
        events.append(event)
    return events


def _append_events(dataset_dir: Path, events: Sequence[dict]) -> None:
    """Append event lines. If anything fails, the file is cut back to its length before the call."""
    path = Path(dataset_dir) / AUDIT_FILE
    data = b"".join(
        (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for event in events
    )
    size = path.stat().st_size if path.exists() else None
    try:
        with open(path, "ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            if size is None:
                path.unlink(missing_ok=True)
            else:
                with open(path, "r+b") as handle:
                    handle.truncate(size)
        except OSError:
            pass  # check_all reports an invalid file; review_readiness reports the unreviewed records
        raise


def _read_bytes_or_none(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _restore(path: Path, data: bytes | None) -> bool:
    """Put a file back as it was (or remove it if it did not exist). True if it succeeded."""
    try:
        if data is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write(path, data)
    except OSError:
        return False
    return True


# ---------- validation ----------

def dataset_issues(dataset: Dataset, manifest: Manifest, texts: Mapping[str, str], threshold: float) -> list[Issue]:
    """The existing ``validate_dataset``, run with the verified texts (so every span is anchored)."""
    return validate_dataset(
        {"dev": dataset.dev, "test": dataset.test}, manifest, near_duplicate_threshold=threshold, texts=texts
    )


def _blank(value: object) -> bool:
    return type(value) is not str or value.strip() == ""


def _invalid(index: int, field: str, message: str) -> Issue:
    return Issue(AUDIT_EVENT_INVALID, f"audit.events[{index}].{field}", message)


def _correction_issues(index: int, event: dict) -> list[Issue]:
    issues: list[Issue] = []
    if "human_review" in event and type(event["human_review"]) is not bool:
        issues.append(_invalid(index, "human_review", "must be true or false: say whether a person did this"))
    elif event.get("human_review") is True:
        problem = reviewer_problem(event.get("reviewer"))
        if problem is not None:
            issues.append(_invalid(index, "reviewer", f"the reviewer {problem}"))
    elif event.get("human_review") is False and "reviewer" in event:
        issues.append(_invalid(index, "reviewer", "a correction that is not a human review has no reviewer"))
    for field in ("basis", "review_type"):
        if field in event and _blank(event[field]):
            issues.append(_invalid(index, field, "must be a non-blank string"))
    if "note" in event and _blank(event["note"]):
        issues.append(_invalid(index, "note", "must be a non-blank string when present"))
    if "changes" not in event:
        return issues
    changes = event["changes"]
    if not isinstance(changes, dict) or not changes:
        issues.append(_invalid(index, "changes", "must be a non-empty object"))
        return issues
    for key in changes:
        if key not in ("difficulty", "evidence"):
            issues.append(_invalid(index, "changes", "may only contain difficulty and evidence"))
            break
    if "difficulty" in changes:
        value = changes["difficulty"]
        ok = (isinstance(value, dict) and set(value) == {"from", "to"} and value["from"] in DIFFICULTIES
              and value["to"] in DIFFICULTIES and value["from"] != value["to"])
        if not ok:
            issues.append(_invalid(index, "changes.difficulty", "must be {from, to}: two different permitted values"))
    if "evidence" in changes and _blank(changes["evidence"]):
        issues.append(_invalid(index, "changes.evidence", "must be a non-blank description"))
    return issues


def event_issues(dataset: Dataset, events: Sequence[dict]) -> list[Issue]:
    """Every event must be of a known type; the human and correction events are validated in full.

    ``audit_sample`` and ``audit_result`` keep their earlier, looser handling (``review_issues``).
    """
    issues: list[Issue] = []
    questions = {q.question_id: q for q in dataset.questions()}
    required = {
        EVENT_LABEL_REVIEW: ("question_id", "reviewer", "date", "author_type"),
        EVENT_HINDI_REVIEW: ("question_id", "reviewer", "date"),
        EVENT_LABEL_CORRECTION: ("question_id", "date", "changes", "basis", "review_type", "human_review"),
        EVENT_AI_SEMANTIC_REVIEW: ("question_id", "review_type", "human_review", "result", "basis"),
    }
    optional = {EVENT_LABEL_CORRECTION: ("reviewer", "note"), EVENT_LABEL_REVIEW: ("content_sha256",),
                EVENT_AI_SEMANTIC_REVIEW: ("date", "note")}
    latest_difficulty: dict[str, tuple[int, str]] = {}
    for index, event in enumerate(events):
        kind = event["event"]
        if kind not in EVENT_TYPES:
            issues.append(Issue(AUDIT_EVENT_UNKNOWN, f"audit.events[{index}]", "the event type is not recognised"))
            continue
        if kind not in required:
            continue
        allowed = {"event", *required[kind], *optional.get(kind, ())}
        if set(event) - allowed:
            issues.append(_invalid(index, "event", "has a field that this event type does not allow"))
        missing = [f for f in required[kind] if f not in event]
        for field in missing:
            issues.append(_invalid(index, field, "is required"))
        known = type(event.get("question_id")) is str and event["question_id"] in questions
        if "question_id" in event:
            if type(event["question_id"]) is not str:
                issues.append(_invalid(index, "question_id", "must be a string"))
            elif not known:
                issues.append(Issue(AUDIT_UNKNOWN_QUESTION, f"audit.events[{index}].question_id",
                                    "the event names a question that is not in the dataset"))
        if "date" in event and not is_valid_iso_date(event["date"]):
            issues.append(_invalid(index, "date", "must be a real YYYY-MM-DD date"))
        if kind in (EVENT_LABEL_REVIEW, EVENT_HINDI_REVIEW) and "reviewer" not in missing:
            problem = reviewer_problem(event["reviewer"])
            if problem is not None:
                issues.append(_invalid(index, "reviewer", f"the reviewer {problem}"))
        if kind == EVENT_LABEL_REVIEW and "author_type" not in missing and event["author_type"] not in AUTHOR_TYPES:
            issues.append(_invalid(index, "author_type", "is not a permitted value"))
        if kind == EVENT_LABEL_REVIEW and "content_sha256" in event:
            digest = event["content_sha256"]
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                issues.append(_invalid(index, "content_sha256", "must be a lowercase SHA-256 hex digest"))
        if kind == EVENT_HINDI_REVIEW and not missing and known:
            q = questions[event["question_id"]]
            if q.query_language != "hi" or q.hindi_reviewed is not True or q.hindi_reviewer != event["reviewer"]:
                issues.append(_invalid(index, "question_id",
                                       "the question is not a Hindi question marked as reviewed by this reviewer"))
        if kind == EVENT_AI_SEMANTIC_REVIEW:
            if event.get("review_type") != AI_SEMANTIC_REVIEW_TYPE:
                issues.append(_invalid(index, "review_type", f"must be {AI_SEMANTIC_REVIEW_TYPE!r}"))
            if event.get("human_review") is not False:
                issues.append(_invalid(index, "human_review", "must be false: an AI review is never a human review"))
            if event.get("result") != "confirmed_unanswerable":
                issues.append(_invalid(index, "result", "must be 'confirmed_unanswerable'"))
            if "basis" in event and (type(event["basis"]) is not str or event["basis"].strip() == ""):
                issues.append(_invalid(index, "basis", "must say where the review is documented"))
            if known and questions[event["question_id"]].answerable:
                issues.append(_invalid(index, "question_id", "an AI semantic review applies to unanswerable questions"))
        if kind == EVENT_LABEL_CORRECTION:
            issues.extend(_correction_issues(index, event))
            changes = event.get("changes")
            if (isinstance(changes, dict) and isinstance(changes.get("difficulty"), dict)
                    and known and "to" in changes["difficulty"]):
                latest_difficulty[event["question_id"]] = (index, changes["difficulty"]["to"])
    for question_id, (index, value) in latest_difficulty.items():
        if questions[question_id].difficulty != value:
            issues.append(Issue(AUDIT_CORRECTION_MISMATCH, f"audit.events[{index}].changes.difficulty",
                                "the latest correction does not match the question's current difficulty"))
    return issues


def review_issues(dataset: Dataset, events: Sequence[dict]) -> list[Issue]:
    """Invalid events and the state of the blind audit. Pending human reviews are not reported here
    (see ``review_readiness``)."""
    issues: list[Issue] = event_issues(dataset, events)
    samples = [e for e in events if e["event"] == EVENT_AUDIT_SAMPLE]
    if len(samples) > 1:
        issues.append(Issue(AUDIT_SAMPLE_REPEATED, "audit", "more than one audit sample is recorded"))
    if samples:
        known = {q.question_id for q in dataset.questions()}
        latest: dict[str, dict] = {}
        for e in events:
            if e["event"] == EVENT_AUDIT_RESULT:
                latest[e.get("question_id")] = e
        for j, question_id in enumerate(samples[0].get("question_ids", [])):
            where = f"audit.sample[{j}]"
            if question_id not in known:
                issues.append(Issue(AUDIT_UNKNOWN_QUESTION, where, "a sampled question is not in the dataset"))
            elif question_id not in latest:
                issues.append(Issue(AUDIT_RESULT_MISSING, where, "a sampled question has no audit result yet",
                                    SEVERITY_WARNING))
            elif latest[question_id].get("outcome") != "agree" and not latest[question_id].get("resolution"):
                issues.append(Issue(AUDIT_DISAGREEMENT_UNRESOLVED, where,
                                    "the audit disagrees with the label and no resolution is recorded"))
    return issues


_PENDING_HINDI_CODES = ("Q_HINDI_REVIEW_REQUIRED", "Q_HINDI_REVIEWER_REQUIRED")


def _without_pending_hindi(dataset: Dataset, issues: list[Issue]) -> list[Issue]:
    """A Hindi question that is not yet reviewed (``hindi_reviewed`` false and no reviewer) is pending, not
    malformed: ``review_readiness`` reports it. Any other Hindi-review inconsistency stays an error. The add
    commands keep the strict rule, so a Hindi question cannot be promoted without a named reviewer."""
    pending = {
        f"{name}.questions[{i}]" for name in ("dev", "test") for i, q in enumerate(dataset.split(name))
        if q.query_language == "hi" and q.hindi_reviewed is False and q.hindi_reviewer is None
    }
    return [
        item for item in issues
        if not (item.code in _PENDING_HINDI_CODES and item.location.split(".hindi_review")[0] in pending)
    ]


def _semantic_reviewer_issues(dataset: Dataset) -> list[Issue]:
    """``semantic_review`` is for a human reader: a placeholder or AI name there is an AI review passed off as human."""
    return [
        Issue(SEMANTIC_REVIEWER_NOT_A_PERSON, f"{name}.questions[{i}].semantic_review.reviewer",
              "the semantic reviewer must be a person; record an AI review as an ai_semantic_review event instead")
        for name in ("dev", "test") for i, q in enumerate(dataset.split(name))
        if q.semantic_review is not None and type(q.semantic_review.reviewer) is str
        and reviewer_problem(q.semantic_review.reviewer) is not None
    ]


def check_all(
    dataset_dir: Path, corpus_dir: Path, threshold: float = PROVISIONAL_NEAR_DUPLICATE_THRESHOLD
) -> list[Issue]:
    """Everything ``check`` reports. Read-only."""
    manifest = load_manifest(corpus_dir)
    texts = load_texts(corpus_dir, manifest)
    dataset = load_dataset(dataset_dir)
    issues = _without_pending_hindi(dataset, dataset_issues(dataset, manifest, texts, threshold))
    issues.extend(_semantic_reviewer_issues(dataset))
    issues.extend(review_issues(dataset, load_events(dataset_dir)))
    metadata_path = Path(dataset_dir) / METADATA_FILE
    if metadata_path.exists():
        try:
            metadata = parse_dataset_metadata(_read(metadata_path))
        except EvaluationDataError as error:
            issues.extend(error.issues)
        else:
            issues.extend(validate_dataset_metadata(metadata, manifest))
    return issues


# ---------- review readiness ----------

@dataclass(frozen=True, slots=True)
class Readiness:
    """The human-review state of a dataset. Not a validation result."""

    questions: int
    missing_evidence_review: tuple[str, ...]
    stale_evidence_review: tuple[tuple[str, str], ...]   # (question id, reason)
    hindi_review_pending: tuple[str, ...]
    hindi_questions: int
    ai_only_semantic_review: tuple[str, ...]
    semantic_review_unverified: tuple[str, ...] = ()   # unanswerable, no human reader and no AI-review event

    @property
    def current_evidence_reviews(self) -> int:
        return self.questions - len(self.missing_evidence_review) - len(self.stale_evidence_review)

    @property
    def ready(self) -> bool:
        return not (self.missing_evidence_review or self.stale_evidence_review or self.hindi_review_pending
                    or self.semantic_review_unverified)  # AI-only semantic review is reported, not blocking


def review_readiness(dataset: Dataset, events: Sequence[dict]) -> Readiness:
    """Which human reviews are missing or no longer current. Read-only; never changes a status.

    A question's current evidence review is its latest ``label_review`` by a real reviewer (the log is
    append-only, so a later review supersedes an earlier one and nothing is lost). A review with a
    ``content_sha256`` is stale when the question text, expected answer or evidence no longer hash to it.
    An earlier review without a hash cannot be compared, so it is stale only when a later
    ``label_correction`` records an evidence change for the question.
    """
    ai_reviewed = {e["question_id"] for e in events
                   if e.get("event") == EVENT_AI_SEMANTIC_REVIEW and type(e.get("question_id")) is str}
    reviews: dict[str, tuple[int, dict]] = {}
    evidence_corrections: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        qid = event.get("question_id")
        if type(qid) is not str:
            continue
        if event["event"] == EVENT_LABEL_REVIEW and reviewer_problem(event.get("reviewer")) is None:
            reviews[qid] = (index, event)
        elif (event["event"] == EVENT_LABEL_CORRECTION and isinstance(event.get("changes"), dict)
              and "evidence" in event["changes"]):
            evidence_corrections.setdefault(qid, []).append(index)
    missing: list[str] = []
    stale: list[tuple[str, str]] = []
    hindi: list[str] = []
    ai_only: list[str] = []
    unverified: list[str] = []
    hindi_total = 0
    questions = dataset.questions()
    for q in questions:
        found = reviews.get(q.question_id)
        if found is None:
            missing.append(q.question_id)
        else:
            index, event = found
            digest = event.get("content_sha256")
            if digest is not None:
                if digest != review_content_sha256(q):
                    stale.append((q.question_id, "the question text, expected answer or evidence changed after the review"))
            elif any(i > index for i in evidence_corrections.get(q.question_id, ())):
                stale.append((q.question_id, "the evidence was corrected after the review, which records no content hash"))
        if q.query_language == "hi":
            hindi_total += 1
            if not (q.hindi_reviewed is True and reviewer_problem(q.hindi_reviewer) is None):
                hindi.append(q.question_id)
        if not q.answerable:
            human = q.semantic_review is not None and reviewer_problem(q.semantic_review.reviewer) is None
            if not human:
                (ai_only if q.question_id in ai_reviewed else unverified).append(q.question_id)
    return Readiness(len(questions), tuple(missing), tuple(stale), tuple(hindi), hindi_total, tuple(ai_only),
                     tuple(unverified))


# ---------- adding ----------

@dataclass(frozen=True, slots=True)
class AddResult:
    added: tuple[str, ...]
    issues: tuple[Issue, ...]
    written: bool


def add_questions(
    dataset_dir: Path,
    corpus_dir: Path,
    items: Sequence[dict],
    *,
    answerable: bool,
    review: Review,
    threshold: float = PROVISIONAL_NEAR_DUPLICATE_THRESHOLD,
) -> AddResult:
    """Add a batch. Nothing is written unless the whole combined dataset has no error.

    Warnings (near-duplicates, a span reused by two units) are returned and do not block.
    """
    problem = reviewer_problem(review.evidence_reviewer)
    if problem is not None:
        raise LabellingError(f"--evidence-reviewer {problem}")
    if not is_valid_iso_date(review.date):
        raise LabellingError("the review date must be a real YYYY-MM-DD date")
    if review.hindi_review_date is not None and not is_valid_iso_date(review.hindi_review_date):
        raise LabellingError("the Hindi review date must be a real YYYY-MM-DD date")
    manifest = load_manifest(corpus_dir)
    texts = load_texts(corpus_dir, manifest)
    existing = load_dataset(dataset_dir)

    built = []
    for i, item in enumerate(items):
        where = f"spec[{i}]"
        built.append(
            build_answerable(item, manifest, texts, review, where) if answerable
            else build_unanswerable(item, review, where)
        )
    if review.hindi_review_date is not None and not any(q.hindi_reviewed for q in built):
        raise LabellingError("--hindi-review-date was given but no question in the batch is a Hindi question")
    additions = {"dev": [], "test": []}
    for q in built:
        if q.split not in additions:
            raise LabellingError("split must be 'dev' or 'test'")
        additions[q.split].append(q)

    combined = Dataset(dev=existing.dev + tuple(additions["dev"]), test=existing.test + tuple(additions["test"]))
    issues = dataset_issues(combined, manifest, texts, threshold)
    if any(item.severity == SEVERITY_ERROR for item in issues):
        return AddResult(added=(), issues=tuple(issues), written=False)

    directory = Path(dataset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    previous = {name: _read_bytes_or_none(directory / f) for name, f in (("dev", DEV_FILE), ("test", TEST_FILE))}
    try:
        for name in ("dev", "test"):
            if additions[name]:
                _write_split(directory, name, combined.split(name))
        # _append_events cuts audit.jsonl back to its old length if it fails part way.
        _append_events(directory, [
            _review_event(q, review.evidence_reviewer, review.date)
            for q in built
        ] + [
            {"event": EVENT_HINDI_REVIEW, "question_id": q.question_id, "reviewer": q.hindi_reviewer,
             "date": review.hindi_review_date or review.date}
            for q in built if q.hindi_reviewed
        ] + (_ai_review_events([q.question_id for q in built], review.ai_semantic_basis)
             if review.ai_semantic_basis is not None and not answerable else []))
    except BaseException as error:
        # Roll back: the question files must not hold questions whose review was not recorded.
        failed = [
            f for name, f in (("dev", DEV_FILE), ("test", TEST_FILE))
            if additions[name] and not _restore(directory / f, previous[name])
        ]
        if not isinstance(error, Exception):
            raise
        if failed:
            raise LabellingError(
                f"writing failed and {', '.join(failed)} could not be restored; run check and readiness before anything else"
            ) from error
        raise LabellingError("writing failed; the dataset files were restored and nothing was added") from error
    return AddResult(added=tuple(q.question_id for q in built), issues=tuple(issues), written=True)


# ---------- blind audit ----------

def draw_audit_sample(dataset: Dataset, seed: int) -> list[str]:
    """20% of the answerable questions (at least 10, or all if fewer) plus every unanswerable one."""
    if type(seed) is not int or seed < 0:
        raise LabellingError("the seed must be a non-negative integer")
    answerable = sorted(q.question_id for q in dataset.questions() if q.answerable)
    unanswerable = sorted(q.question_id for q in dataset.questions() if not q.answerable)
    count = min(len(answerable), max(AUDIT_MINIMUM, math.ceil(len(answerable) * AUDIT_PERCENT / 100)))
    chosen = random.Random(seed).sample(answerable, count)
    return sorted(chosen + unanswerable)


def record_audit_sample(dataset_dir: Path, dataset: Dataset, seed: int, date: str) -> list[str]:
    if not is_valid_iso_date(date):
        raise LabellingError("the date must be a real YYYY-MM-DD date")
    events = load_events(dataset_dir)
    if any(e["event"] == EVENT_AUDIT_SAMPLE for e in events):
        raise LabellingError("an audit sample is already recorded")
    if not dataset.questions():
        raise LabellingError("the dataset has no questions")
    ids = draw_audit_sample(dataset, seed)
    _append_events(Path(dataset_dir), [{
        "event": EVENT_AUDIT_SAMPLE, "seed": seed, "date": date, "percent": AUDIT_PERCENT,
        "minimum": AUDIT_MINIMUM, "question_ids": ids,
    }])
    return ids


def judge_audit(question: Question, derived: Sequence[tuple[str, int, int]]) -> str:
    """``agree`` or ``disagree``.

    Answerable: every required unit has an original span that overlaps a derived span of the same
    document, and every derived span overlaps some original span. Unanswerable: the auditor must
    have found no evidence at all.
    """
    if not question.answerable:
        return "agree" if not derived else "disagree"
    originals = [[(s.corpus_key, s.start, s.end) for s in unit.spans] for unit in question.units]

    def meets(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
        return a[0] == b[0] and overlaps((a[1], a[2]), (b[1], b[2]))

    units_found = all(any(meets(o, d) for o in unit for d in derived) for unit in originals)
    derived_explained = all(any(meets(d, o) for unit in originals for o in unit) for d in derived)
    return "agree" if units_found and derived_explained else "disagree"


def record_audit_results(
    dataset_dir: Path, corpus_dir: Path, items: Sequence[dict], *, auditor: str, date: str,
    alias_map: Mapping[str, str] | None = None, document_map: Mapping[str, str] | None = None,
) -> list[dict]:
    """Resolve the auditor's independently found quotes, compare them, and append the results.

    With ``alias_map`` (audit id to real question id, from the private map of ``audit_bundle``) every item names a
    question by its audit id, each audit id may appear once, and the map must list exactly the recorded sample.
    With ``document_map`` (neutral document id to corpus key, from the same file) every span names its document by
    the neutral id. The two maps are used together. The events written always carry the real ids and corpus keys.
    """
    if type(auditor) is not str or auditor.strip() == "":
        raise LabellingError("--auditor must be a non-blank name")
    if not is_valid_iso_date(date):
        raise LabellingError("the date must be a real YYYY-MM-DD date")
    manifest = load_manifest(corpus_dir)
    texts = load_texts(corpus_dir, manifest)
    dataset = load_dataset(dataset_dir)
    by_id = {q.question_id: q for q in dataset.questions()}
    samples = [e for e in load_events(dataset_dir) if e["event"] == EVENT_AUDIT_SAMPLE]
    if not samples:
        raise LabellingError("no audit sample is recorded; run audit-sample first")
    sampled = set(samples[0].get("question_ids", []))
    if (alias_map is None) != (document_map is None):
        raise LabellingError("the alias map and the document map are used together")
    if alias_map is not None and set(alias_map.values()) != sampled:
        raise LabellingError("the alias map does not match the recorded audit sample")

    results = []
    seen_aliases: set[str] = set()
    for i, item in enumerate(items):
        where = f"spec[{i}]"
        _check_keys(item, ("question_id", "spans", "resolution"), ("question_id", "spans"), where)
        if alias_map is not None:
            alias = item["question_id"]
            if type(alias) is not str or alias not in alias_map:
                raise LabellingError(f"{where}: the audit id is not in the alias map")
            if alias in seen_aliases:
                raise LabellingError(f"{where}: the audit id is repeated")
            seen_aliases.add(alias)
            item = {**item, "question_id": alias_map[alias]}
        question = by_id.get(item["question_id"])
        if question is None or item["question_id"] not in sampled:
            raise LabellingError(f"{where}: the question is not in the audit sample")
        if not isinstance(item["spans"], list):
            raise LabellingError(f"{where}.spans must be an array")
        derived = []
        for s, raw in enumerate(item["spans"]):
            here = f"{where}.spans[{s}]"
            if not isinstance(raw, dict):
                raise LabellingError(f"{here} must be an object")
            _check_keys(raw, ("corpus_key", "quote", "occurrence"), ("corpus_key", "quote"), here)
            if document_map is not None:
                document = raw["corpus_key"]
                if type(document) is not str or document not in document_map:
                    raise LabellingError(f"{here}: the document id is not in the document map")
                raw = {**raw, "corpus_key": document_map[document]}
            try:
                span = make_span(manifest, texts, raw["corpus_key"], raw["quote"], raw.get("occurrence"))
            except LabellingError as error:
                raise LabellingError(f"{here}: {error}") from None
            derived.append((span.corpus_key, span.start, span.end))
        resolution = item.get("resolution")
        if resolution is not None and (type(resolution) is not str or resolution.strip() == ""):
            raise LabellingError(f"{where}.resolution must be a non-blank string")
        results.append({
            "event": EVENT_AUDIT_RESULT, "question_id": question.question_id, "auditor": auditor, "date": date,
            "outcome": judge_audit(question, derived), "derived": [list(d) for d in derived],
            "resolution": resolution,
        })
    _append_events(Path(dataset_dir), results)
    return results
