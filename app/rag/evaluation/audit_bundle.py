"""Materials for the second-pass audit of the evaluation labels (E1.6, E1.8).

The bundle is built so that a person who has seen no label would receive nothing that reveals one. When the auditor
has already seen the labels, as in the E1.8 second pass, the same bundle supports a consistency check, not a blind or
independent audit, and its instructions and declaration say so. This module builds what the auditor receives and
keeps what they should not need apart from it:

* an **opaque audit id** (``A01``, ``A02`` ...) for every sampled question, assigned in a shuffled order that
  does not follow the question ids, the split or whether the question is answerable;
* the **auditor questions**: for each audit id only the question text and the query language;
* a **private alias map** (audit id to real question id, and neutral document id ``D01`` ... to real corpus key),
  written to a file in a private folder that is never part of the bundle, never printed, and bound to the recorded
  ``audit_sample`` by a hash of the sampled ids. The folder is created by this module with owner-only access, which is
  verified (``icacls`` and ``Get-Acl`` on Windows, mode bits elsewhere); if that cannot be verified nothing is written;
* the **auditor bundle**: instructions, the questions, the manifest-bound corpus text files under neutral names
  (``D01.txt`` ...), a document list, a blank response worksheet and a declaration. It is built in a temporary folder, validated, and only then
  moved into place. A bundle that fails validation is deleted and is never delivered.

Nothing here changes the audit schema: ``audit_sample`` still records the real question ids and
``audit_result`` is still written by ``labelling.record_audit_results`` (which translates audit ids and document ids
through the private map and records real ids). The corpus text is the manifest-verified extracted text, never normalised.

Validation of a bundle is an allowlist: its file set must be exactly the expected one, every generated file must
equal what the generator produces from the validated question list, and every corpus file must equal the
verified text. On top of that, the generated files are scanned for real question ids, internal label names,
reviewer names and hashes. Corpus text is compared, not scanned (it legitimately contains the facts).
No error here contains a question, an answer, a quote or a real question id.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import os
import random
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.rag.chunking import sha256_text
from app.rag.evaluation import labelling
from app.rag.evaluation.labelling import LabellingError
from app.rag.evaluation.schema import Manifest

MAP_FORMAT = "e16-audit-alias-map-2"
MAP_FILE = "alias_map.json"
QUESTION_KEYS = ("audit_id", "question", "query_language")
LANGUAGES = ("en", "hi")
_ALIAS = re.compile(r"A[0-9]{2,3}")
_MAX_ATTEMPTS = 1000
_IS_WINDOWS = os.name == "nt"  # a variable so that tests can exercise the Windows branch anywhere

INSTRUCTIONS_FILE = "AUDITOR_INSTRUCTIONS.md"
QUESTIONS_FILE = "questions.md"
DOCUMENT_LIST_FILE = "document_list.md"
WORKSHEET_FILE = "response_worksheet.md"
DECLARATION_FILE = "declaration.md"
CORPUS_FOLDER = "corpus"
GENERATED_FILES = (INSTRUCTIONS_FILE, QUESTIONS_FILE, DOCUMENT_LIST_FILE, WORKSHEET_FILE, DECLARATION_FILE)

# Internal names that must never appear in a generated bundle file.
_FORBIDDEN_TOKENS = (
    "expected_answer", "unanswerable", "unanswerable_type", "retrieval_type", "query_language", "close_distractor",
    "hindi_reviewed", "hindi_reviewer", "semantic_review", "label_review", "hindi_review", "ai_semantic_review",
    "audit_sample", "audit_result", "label_correction", "content_sha256", "quote_sha256", "source_sha256",
    "author_type", "llm_assisted", "on_topic_not_covered", "off_topic", "en_to_en", "hi_to_en", "hi_to_hi", "en_to_hi",
)


# ---------- aliases ----------

def sample_digest(question_ids: Sequence[str]) -> str:
    """Binds a private map to one sample: the hash of the sorted real ids."""
    return sha256_text("\n".join(sorted(question_ids)))


def _runs(values: Sequence[object]) -> int:
    return 1 + sum(1 for a, b in itertools.pairwise(values) if a != b)


def _ordering_problem(order: Sequence[str], by_id: Mapping[str, object]) -> str | None:
    """Why an auditor-facing order would let someone infer a label, or None. Too few questions cannot be checked."""
    if len(order) < 3:
        return None
    if list(order) == sorted(order) or list(order) == sorted(order, reverse=True):
        return "the order follows the question ids"
    for name, key in (("answerability", lambda q: q.answerable), ("split", lambda q: q.split)):
        values = [key(by_id[i]) for i in order]
        distinct = len(set(values))
        if distinct > 1 and _runs(values) == distinct:
            return f"the order groups the questions by {name}"
    return None


def make_aliases(dataset: labelling.Dataset, question_ids: Sequence[str], alias_seed: int | None = None) -> dict[str, str]:
    """Audit id to real question id, in a shuffled order.

    The shuffle never looks at the ids or the classification; an order that follows the ids or groups the
    questions by split or answerability is drawn again. ``alias_seed`` only makes the shuffle reproducible
    (tests, rehearsals); without it the system random source is used.
    """
    ids = list(question_ids)
    by_id = {q.question_id: q for q in dataset.questions()}
    if not ids or len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
        raise LabellingError("the sample must name each question once, and every question must be in the dataset")
    if alias_seed is not None and (type(alias_seed) is not int or alias_seed < 0):
        raise LabellingError("the alias seed must be a non-negative integer")
    rng = random.Random(alias_seed) if alias_seed is not None else random.SystemRandom()
    order = sorted(ids)
    for _ in range(_MAX_ATTEMPTS):
        rng.shuffle(order)
        if _ordering_problem(order, by_id) is None:
            break
    else:
        raise LabellingError("no order of the sample hides the classification; the sample is too small to blind")
    width = max(2, len(str(len(order))))
    return {f"A{number:0{width}d}": question_id for number, question_id in enumerate(order, start=1)}


def make_document_aliases(manifest: Manifest, alias_seed: int | None = None) -> dict[str, str]:
    """Neutral document id (``D01`` ...) to real corpus key, in a shuffled order.

    The corpus keys name the documents' topics, so the auditor only ever sees the neutral ids.
    """
    keys = sorted(d.corpus_key for d in manifest.documents)
    if not keys or len(set(keys)) != len(keys):
        raise LabellingError("the manifest must name each document once")
    rng = random.Random(f"{alias_seed}:documents") if alias_seed is not None else random.SystemRandom()
    rng.shuffle(keys)
    width = max(2, len(str(len(keys))))
    return {f"D{number:0{width}d}": key for number, key in enumerate(keys, start=1)}


def auditor_questions(dataset: labelling.Dataset, aliases: Mapping[str, str]) -> list[dict]:
    """The only per-question material an auditor receives: audit id, question text, query language."""
    by_id = {q.question_id: q for q in dataset.questions()}
    return [
        {"audit_id": alias, "question": by_id[aliases[alias]].question, "query_language": by_id[aliases[alias]].query_language}
        for alias in sorted(aliases)
    ]


def validate_questions(questions: object) -> list[dict]:
    """Strict allowlist for the auditor questions: exactly the three fields, contiguous audit ids A01..An."""
    if not isinstance(questions, list) or not questions:
        raise LabellingError("the auditor questions must be a non-empty array")
    width = max(2, len(str(len(questions))))
    for number, entry in enumerate(questions, start=1):
        if not isinstance(entry, dict) or tuple(sorted(entry)) != tuple(sorted(QUESTION_KEYS)):
            raise LabellingError(f"auditor question {number} must have exactly the fields {', '.join(QUESTION_KEYS)}")
        if entry["audit_id"] != f"A{number:0{width}d}":
            raise LabellingError("the audit ids must run A01, A02, ... without gaps, in order")
        if type(entry["question"]) is not str or entry["question"].strip() == "":
            raise LabellingError(f"auditor question {number} has no question text")
        if entry["query_language"] not in LANGUAGES:
            raise LabellingError(f"auditor question {number} has an unsupported language")
    return questions


def write_questions(path: Path, questions: Sequence[dict]) -> None:
    validate_questions(list(questions))
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(list(questions), ensure_ascii=False, indent=2) + "\n")


# ---------- the private folder and map ----------

def _run(command: Sequence[str]) -> str:
    """Run a Windows tool and return its output. Any failure refuses, so a permission is never assumed."""
    try:
        done = subprocess.run(list(command), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        raise LabellingError(f"the private folder's permissions could not be set or checked ({command[0]} did not run)") from None
    if done.returncode != 0:
        raise LabellingError(f"the private folder's permissions could not be set or checked ({command[0]} failed)")
    return done.stdout


def _windows_sid() -> str:
    """The security identifier of the account running this process (``whoami`` prints it as CSV)."""
    rows = list(csv.reader(io.StringIO(_run(["whoami", "/user", "/fo", "csv", "/nh"]).strip())))
    if len(rows) != 1 or len(rows[0]) != 2 or not re.fullmatch(r"S-1-[0-9-]+", rows[0][1]):
        raise LabellingError("the account running this process could not be identified")
    return rows[0][1]


def _windows_acl(path: Path) -> tuple[bool, list[tuple[str, str]]]:
    """Whether inheritance is switched off, and every access rule as (Allow or Deny, SID). SIDs do not depend on language."""
    quoted = str(path).replace("'", "''")
    script = ("$ErrorActionPreference = 'Stop'; $a = Get-Acl -LiteralPath '" + quoted + "'; "
              "'PROTECTED=' + $a.AreAccessRulesProtected; "
              "foreach ($r in $a.Access) { $r.AccessControlType.ToString() + ';' + "
              "$r.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value }")
    lines = [ln.strip() for ln in _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]).splitlines()
             if ln.strip()]
    if not lines or lines[0] not in ("PROTECTED=True", "PROTECTED=False"):
        raise LabellingError("the private folder's permissions could not be read")
    rules = []
    for line in lines[1:]:
        kind, _, sid = line.partition(";")
        if kind not in ("Allow", "Deny") or not re.fullmatch(r"S-1-[0-9-]+", sid):
            raise LabellingError("the private folder's permissions could not be read")
        rules.append((kind, sid))
    return lines[0] == "PROTECTED=True", rules


def verify_private(path: Path, *, directory: bool) -> None:
    """Raise unless only the current account can use ``path``. Windows: every access rule is an Allow for this
    account's SID (a directory must also have inheritance switched off). Elsewhere: owned by this user, no group or
    other permission bits. Anything that cannot be checked is refused."""
    path = Path(path)
    if _IS_WINDOWS:
        protected, rules = _windows_acl(path.resolve())
        me = _windows_sid()
        if (directory and not protected) or not rules or any(kind != "Allow" or sid != me for kind, sid in rules):
            raise LabellingError("the private folder is not restricted to this account")
        return
    try:
        info = os.lstat(path)
    except OSError:
        raise LabellingError("the private folder cannot be inspected") from None
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if not wanted(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise LabellingError("the private folder is not restricted to this account")


def create_private_dir(path: Path) -> Path:
    """Create a new folder only this account can use, and verify that. The folder must not exist. If the restriction
    cannot be applied and verified, the folder (empty, created a moment ago) is removed and an error is raised."""
    path = Path(path)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        raise LabellingError("the private folder already exists") from None
    except OSError:
        raise LabellingError("the private folder cannot be created (its parent folder must exist)") from None
    try:
        if _IS_WINDOWS:
            me, folder = _windows_sid(), str(path.resolve())
            # Drop inherited access and give this account full control (inherited by the files inside).
            _run(["icacls", folder, "/inheritance:r", "/grant:r", f"*{me}:(OI)(CI)F"])
            # /inheritance:r keeps EXPLICIT entries, and os.mkdir(path, 0o700) (Python 3.12.4+) creates some: SYSTEM,
            # Administrators and OWNER RIGHTS. Remove every other principal that is really present, then verify strictly
            # below. Nothing is whitelisted: anything that is still there afterwards makes verification fail.
            extras = sorted({sid for _, sid in _windows_acl(path.resolve())[1] if sid != me})
            if extras:
                _run(["icacls", folder, "/remove", *(f"*{sid}" for sid in extras)])
        else:
            os.chmod(path, 0o700)
        verify_private(path, directory=True)
    except BaseException:
        try:
            os.rmdir(path)
        except OSError:
            pass
        raise
    return path


def write_private_map(path: Path, aliases: Mapping[str, str], documents: Mapping[str, str], date: str) -> None:
    """Create the map file; it must not exist. Created with owner-only mode bits where those apply; the caller
    verifies the folder and file (``verify_private``) because on Windows access comes from the folder."""
    payload = {"format": MAP_FORMAT, "created": date, "sample_sha256": sample_digest(list(aliases.values())),
               "aliases": dict(sorted(aliases.items())), "documents": dict(sorted(documents.items()))}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_map(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    try:
        payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise LabellingError("the private alias map cannot be read") from None
    bad = LabellingError("the private alias map is not valid")
    if not isinstance(payload, dict) or set(payload) != {"format", "created", "sample_sha256", "aliases", "documents"}:
        raise bad
    aliases, documents = payload["aliases"], payload["documents"]
    if payload["format"] != MAP_FORMAT or not isinstance(aliases, dict) or not aliases:
        raise bad
    if not isinstance(documents, dict) or not documents:
        raise bad
    for mapping in (aliases, documents):
        prefix = "A" if mapping is aliases else "D"
        width = max(2, len(str(len(mapping))))
        if sorted(mapping) != [f"{prefix}{n:0{width}d}" for n in range(1, len(mapping) + 1)]:
            raise bad
        values = list(mapping.values())
        if any(type(v) is not str or v.strip() == "" for v in values) or len(set(values)) != len(values):
            raise bad
    if payload["sample_sha256"] != sample_digest(list(aliases.values())):
        raise bad
    return dict(aliases), dict(documents)


def read_private_map(path: Path) -> dict[str, str]:
    """The audit id to real question id map, checked against its own digest. Errors never show an id."""
    return _load_map(path)[0]


def read_document_map(path: Path) -> dict[str, str]:
    """The neutral document id to real corpus key map from the same private file."""
    return _load_map(path)[1]


def _check_documents(documents: Mapping[str, str], manifest: Manifest) -> None:
    if sorted(documents.values()) != sorted(d.corpus_key for d in manifest.documents):
        raise LabellingError("the private map's documents do not match the corpus manifest")


# ---------- generated files ----------

def _instructions(count: int, documents: int) -> str:
    last = f"A{count:0{max(2, len(str(count)))}d}"
    return f"""# Evidence audit: instructions (non-independent second pass)

## Nature of this audit

This is a **non-independent second pass**. The person doing it has already seen the dataset labels, expected answers,
evidence passages and earlier decisions for these questions. Its purpose is to find mistakes and inconsistencies in the
labels. It is not an independent or blind validation, and its results must not be described as one. The question
numbers and document keys are neutral names used for recording; they do not hide anything from the auditor.

## Purpose

You are re-checking {count} questions against {documents} documents. For each question, decide from the documents alone
whether they answer it. If they do, quote the passage or passages that state the answer. Do not look up earlier labels
or evidence while you work. Find the evidence in the documents as if for the first time, and if a question reminds you
of an earlier decision, say so under "Notes on uncertainty".

## What you have

- `{QUESTIONS_FILE}`: the {count} questions, numbered A01 to {last}. Some questions may be written in Hindi.
- `{CORPUS_FOLDER}/`: the {documents} documents as plain-text files (UTF-8). Cite a document by its key, which is the file
  name without `.txt` (`D01`, `D02`, ...).
- `{DOCUMENT_LIST_FILE}`: the key and language of each document.
- `{WORKSHEET_FILE}`: where you record your findings, one block per question.
- `{DECLARATION_FILE}`: to be completed and returned with the worksheet.

## Rules

1. Use only the files in `{CORPUS_FOLDER}/`. Do not use the internet, search engines, AI assistants or any other
   document, and do not use outside knowledge.
2. Some questions may have no answer in the documents. Do not assume how many. "Not answered" is the correct
   finding when no passage answers the question.
3. Judge each question on its own. Do not infer anything from a question's number, wording or position.
4. Do not ask any other person or tool to find the evidence for you, and do not share these files.
5. Do not guess. If a passage seems related but you cannot tell whether it answers the question, record Unsure.

## For each question, record one status

- **Answered**: one or more passages state what the question asks for.
- **Not answered**: you searched and found no passage that answers it.
- **Unsure**: you cannot decide. Say why.

## How to quote

- Copy each quote exactly from the text file: the same characters, spacing, punctuation, special characters and
  line breaks. Do not retype, correct, shorten in the middle, or translate it.
- Quote the smallest passage that states the fact, usually one sentence or one table row.
- If a complete answer needs several separate facts, give a separate quote for each fact.
- If the same fact is stated in more than one place, give each place as its own quote.
- If your quote might occur more than once in a file, add a few words of the surrounding text so it is unique.
- Give the document key with every quote.

## Also record, for every question

- Your answer in one or two sentences, in your own words.
- Your confidence: High, Medium or Low.
- For "Not answered" and "Unsure": the closest passages you inspected, and why they do not settle the question.
- Anything about the question or the documents that you were unsure of.

## Returning your work

Return the completed `{WORKSHEET_FILE}` and the completed `{DECLARATION_FILE}` to the project owner. Questions about
the process are welcome, but they must not mention any answer you have found.
"""


def _questions_md(questions: Sequence[Mapping[str, str]]) -> str:
    names = {"en": "English", "hi": "Hindi"}
    parts = ["# Questions\n"]
    for entry in questions:
        parts.append(f"\n## {entry['audit_id']}\n\nLanguage: {names[entry['query_language']]}\n\n{entry['question']}\n")
    return "".join(parts)


def _document_list(manifest: Manifest, documents: Mapping[str, str]) -> str:
    """Neutral keys only: a title or a corpus key would name the document's topic."""
    language = {d.corpus_key: d.language for d in manifest.documents}
    rows = "".join(f"| `{alias}` | {language[documents[alias]]} | `{CORPUS_FOLDER}/{alias}.txt` |\n" for alias in sorted(documents))
    return f"# Documents\n\n| Key | Language | File |\n|---|---|---|\n{rows}"


def _worksheet(questions: Sequence[Mapping[str, str]]) -> str:
    blocks = "".join(
        f"\n## {entry['audit_id']}\n\n"
        "Status (keep one: Answered / Not answered / Unsure):\n\n"
        "Supporting quotes (document key, then the exact text; one block per quote):\n\n"
        "Answer in your own words (one or two sentences):\n\n"
        "Confidence (High / Medium / Low):\n\n"
        "Closest passages inspected (required for Not answered and Unsure):\n\n"
        "Notes on uncertainty:\n"
        for entry in questions
    )
    return f"# Response worksheet\n{blocks}"


def _declaration() -> str:
    return """# Declaration (non-independent second pass)

This audit is a non-independent second pass. It checks the labels for mistakes and inconsistencies. It is not an
independent or blind validation of them.

I confirm that:

- before this audit I have seen the labels, expected answers, evidence passages and earlier decisions for some or all
  of these questions;
- during this audit I used only the files in this bundle, and no internet, search engine, AI assistant or other
  document;
- I recorded evidence from the corpus text, and where a question reminded me of an earlier decision I said so in my
  notes;
- I followed the rules in the instructions.

Name:

Date:

Signature:
"""


def render_files(questions: Sequence[Mapping[str, str]], manifest: Manifest, documents: Mapping[str, str]) -> dict[str, str]:
    return {
        INSTRUCTIONS_FILE: _instructions(len(questions), len(manifest.documents)),
        QUESTIONS_FILE: _questions_md(questions),
        DOCUMENT_LIST_FILE: _document_list(manifest, documents),
        WORKSHEET_FILE: _worksheet(questions),
        DECLARATION_FILE: _declaration(),
    }


# ---------- validation and building ----------

def _check_mapping(questions: Sequence[Mapping[str, str]], aliases: Mapping[str, str], dataset: labelling.Dataset) -> None:
    by_id = {q.question_id: q for q in dataset.questions()}
    if sorted(aliases) != [e["audit_id"] for e in questions]:
        raise LabellingError("the auditor questions and the private alias map do not list the same audit ids")
    for entry in questions:
        real = by_id.get(aliases[entry["audit_id"]])
        if real is None or real.question != entry["question"] or real.query_language != entry["query_language"]:
            raise LabellingError("an auditor question does not match the dataset question it maps to")
    problem = _ordering_problem([aliases[a] for a in sorted(aliases)], by_id)
    if problem is not None:
        raise LabellingError(f"the auditor-facing order is not blind: {problem}")


def _names(dataset: labelling.Dataset, events: Sequence[dict]) -> set[str]:
    names = {e[k] for e in events for k in ("reviewer", "auditor") if type(e.get(k)) is str}
    for q in dataset.questions():
        if q.hindi_reviewer:
            names.add(q.hindi_reviewer)
        if q.semantic_review is not None:
            names.add(q.semantic_review.reviewer)
    return {n for n in names if len(n.strip()) >= 3}


def _leak_codes(
    text: str, ids: Sequence[str], names: set[str], keys: Sequence[str], *, strict: bool, dataset: labelling.Dataset,
) -> list[str]:
    found = []
    lowered = text.lower()
    if any(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(i)}(?![A-Za-z0-9_-])", text, flags=re.IGNORECASE) for i in ids):
        found.append("a real question id")
    if any(token in lowered for token in _FORBIDDEN_TOKENS):
        found.append("an internal label name")
    if any(key.lower() in lowered for key in keys):
        found.append("a corpus file name")
    if re.search(r"[0-9a-f]{64}", lowered):
        found.append("a hash")
    if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, flags=re.IGNORECASE) for n in names):
        found.append("a reviewer name")
    if strict:  # static files only: a question may legitimately reuse wording of the evidence
        texts = [q.expected_answer for q in dataset.questions() if q.expected_answer]
        texts += [s.quote for q in dataset.questions() for u in q.units for s in u.spans]
        if any(len(t) >= 25 and t in text for t in texts):
            found.append("an expected answer or an evidence quote")
    return found


def validate_bundle(
    bundle_dir: Path, *, questions: Sequence[Mapping[str, str]], aliases: Mapping[str, str],
    documents: Mapping[str, str], manifest: Manifest, texts: Mapping[str, str], dataset: labelling.Dataset,
    events: Sequence[dict],
) -> None:
    """Fail closed. Raises ``LabellingError`` naming the file and the kind of problem, never the content."""
    questions = validate_questions(list(questions))
    _check_mapping(questions, aliases, dataset)
    _check_documents(documents, manifest)
    root = Path(bundle_dir)
    expected = render_files(questions, manifest, documents)
    expected_corpus = {f"{CORPUS_FOLDER}/{alias}.txt": texts[key] for alias, key in documents.items()}
    present = {}
    for current, folders, files in os.walk(root, followlinks=False):
        for name in [*folders, *files]:
            full = Path(current) / name
            if full.is_symlink():
                raise LabellingError("the bundle contains a link")
        for name in files:
            full = Path(current) / name
            present[full.relative_to(root).as_posix()] = full
    if set(present) != set(expected) | set(expected_corpus):
        raise LabellingError("the bundle does not contain exactly the expected files")
    for name, text in {**expected, **expected_corpus}.items():
        if present[name].read_bytes() != text.encode("utf-8"):
            raise LabellingError(f"{name} is not the expected content")
    ids = [q.question_id for q in dataset.questions()]
    names = _names(dataset, events)
    for name, text in expected.items():
        codes = _leak_codes(text, ids, names, list(documents.values()), strict=name != QUESTIONS_FILE, dataset=dataset)
        if codes:
            raise LabellingError(f"{name} contains {', '.join(codes)}")


def build_bundle(dataset_dir: Path, corpus_dir: Path, questions_path: Path, map_path: Path, out_dir: Path) -> int:
    """Build, validate and then move into place the auditor bundle. Returns the number of questions."""
    out = Path(out_dir)
    if out.exists():
        raise LabellingError("the bundle folder already exists")
    if not out.parent.is_dir():
        raise LabellingError("the folder that should hold the bundle does not exist")
    resolved_map = Path(map_path).resolve()
    if resolved_map == out.resolve() or out.resolve() in resolved_map.parents:
        raise LabellingError("the private alias map must not be inside the bundle folder")
    dataset = labelling.load_dataset(dataset_dir)
    events = labelling.load_events(dataset_dir)
    samples = [e for e in events if e["event"] == labelling.EVENT_AUDIT_SAMPLE]
    if len(samples) != 1:
        raise LabellingError("exactly one recorded audit sample is needed")
    aliases, documents = _load_map(map_path)
    if set(aliases.values()) != set(samples[0].get("question_ids", [])):
        raise LabellingError("the private alias map does not match the recorded audit sample")
    try:
        raw = json.loads(Path(questions_path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise LabellingError("the auditor questions file cannot be read") from None
    questions = validate_questions(raw)
    manifest = labelling.load_manifest(corpus_dir)
    texts = labelling.load_texts(corpus_dir, manifest)
    _check_mapping(questions, aliases, dataset)
    _check_documents(documents, manifest)
    files = {**render_files(questions, manifest, documents),
             **{f"{CORPUS_FOLDER}/{alias}.txt": texts[key] for alias, key in documents.items()}}
    staging = Path(tempfile.mkdtemp(prefix=".audit-bundle-", dir=out.parent))
    try:
        for name, text in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        validate_bundle(staging, questions=questions, aliases=aliases, documents=documents, manifest=manifest,
                        texts=texts, dataset=dataset, events=events)
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(questions)
