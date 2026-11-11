"""Duplicate and near-duplicate question detection (E1.2d, step 2).

``find_duplicate_questions`` compares the question texts of a whole dataset (dev and test
together) and returns ``Issue`` objects. It reads the questions and nothing else, and does no
I/O.

Normalisation: ``" ".join(text.split()).casefold()``. Blank normalised text is ignored.

* Exact duplicate (error): two questions with the same normalised text, whatever their split
  or language, answerable or not. The later occurrence is reported; for three identical
  questions that is the second and the third.
* Similar (warning): two questions with the same ``query_language`` whose character 5-gram
  Jaccard similarity is at least ``threshold``. Each pair is reported once, at the later
  question, and a pair already reported as an exact duplicate is not reported again. If either
  normalised text has fewer than five characters its similarity is 0. A warning asks for a
  human to look at the pair; it never fails validation by itself.

Order: dev questions are processed before test questions, and each split keeps its input
order. "Later" means later in that order. The threshold is required and has no default.

A question whose text or language is not exactly ``str``, whose split is not exactly ``dev``
or ``test``, or an item that is not a ``Question``, is skipped; the record validator reports
those problems.

Privacy: issue messages are fixed text. A location holds only the split name, the fixed field
name ``question`` and a numeric position within that split (the question's position among the
input questions of the same split, counting skipped ones), never question text or
identifiers: ``test.questions[5].question`` for an exact duplicate, and
``test.questions[5].question~dev.questions[2].question`` for a similar pair (the later
question, then the earlier one).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from app.rag.evaluation.schema import SEVERITY_WARNING, Issue, Question

DATASET_DUPLICATE_QUESTION_TEXT = "DATASET_DUPLICATE_QUESTION_TEXT"
DATASET_SIMILAR_QUESTION = "DATASET_SIMILAR_QUESTION"

NGRAM_SIZE = 5

_MESSAGES = {
    DATASET_DUPLICATE_QUESTION_TEXT: "the question text repeats an earlier question",
    DATASET_SIMILAR_QUESTION: "the question text is very similar to an earlier question; review the pair",
}


def normalize_question_text(text: str) -> str:
    """Collapse whitespace runs to one space, trim the ends and case-fold."""
    return " ".join(text.split()).casefold()


def _ngrams(normalized: str) -> frozenset[str]:
    return frozenset(normalized[i : i + NGRAM_SIZE] for i in range(len(normalized) - NGRAM_SIZE + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def character_ngram_jaccard(first: str, second: str) -> float:
    """Jaccard similarity of the character 5-gram sets of two normalised texts.

    0.0 if either text has fewer than five characters (it has no 5-grams).
    """
    return _jaccard(_ngrams(first), _ngrams(second))


def _check_threshold(threshold: object) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if isinstance(threshold, float) and not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number from 0 to 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number from 0 to 1")


def find_duplicate_questions(questions: Sequence[Question], *, threshold: float) -> list[Issue]:
    """Exact-duplicate errors and similar-pair warnings for ``questions``.

    ``threshold`` is a number from 0 to 1 (``bool`` is refused): a same-language pair is
    reported when its similarity is at least ``threshold``. Raises ``TypeError`` for a
    non-number threshold or for ``questions`` that is a ``str`` or ``bytes``, and
    ``ValueError`` for a threshold that is NaN, infinite or outside [0, 1].
    """
    _check_threshold(threshold)
    if isinstance(questions, (str, bytes)):
        raise TypeError("questions must be a sequence of Question")

    entries = []  # (is test, location, language, normalised text, 5-grams) for usable questions
    seen = {"dev": 0, "test": 0}  # questions seen so far in each split, usable or not
    for q in questions:
        if not isinstance(q, Question) or type(q.split) is not str or q.split not in seen:
            continue
        position = seen[q.split]
        seen[q.split] += 1
        if type(q.question) is not str or type(q.query_language) is not str:
            continue
        normalized = normalize_question_text(q.question)
        if normalized == "":
            continue
        location = f"{q.split}.questions[{position}].question"
        entries.append((q.split == "test", location, q.query_language, normalized, _ngrams(normalized)))
    entries.sort(key=lambda entry: entry[0])  # stable: dev first, input order kept within a split

    issues: list[Issue] = []
    for position, (_, location, language, normalized, grams) in enumerate(entries):
        if any(earlier[3] == normalized for earlier in entries[:position]):
            code = DATASET_DUPLICATE_QUESTION_TEXT
            issues.append(Issue(code, location, _MESSAGES[code]))
        for _, earlier_location, earlier_language, earlier_text, earlier_grams in entries[:position]:
            if earlier_language != language or earlier_text == normalized:
                continue
            if _jaccard(grams, earlier_grams) >= threshold:
                code = DATASET_SIMILAR_QUESTION
                issues.append(Issue(code, f"{location}~{earlier_location}", _MESSAGES[code], SEVERITY_WARNING))
    return issues
