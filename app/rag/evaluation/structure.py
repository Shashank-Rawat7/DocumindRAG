"""Derived structure of an answerable evaluation question (E1.2d, step 1).

``derive_structure`` computes three facts from a question that is already known to be valid:

* ``unit_structure``: ``single_unit`` for exactly one evidence unit, ``multi_unit`` for two or
  more. Only labelled units are counted, never spans or documents. Units are required facts
  (all of them are needed); the spans inside one unit are alternatives (any one is enough).
* ``has_alternatives``: true when any unit has more than one span. The spans are only
  counted. Whether they overlap is irrelevant here, and no interval check is made.
* ``retrieval_type``: ``<query language>_to_<source language>``. The two cross-language
  values are listed in ``CROSS_LANGUAGE_RETRIEVAL_TYPES``.

These values are derived, never stored: ``Question`` has no such fields and nothing is written
to the dataset files, so they cannot disagree with the data. ``STRUCTURE_VERSION`` names the
classification rules; change it whenever those rules change, so a report can say which
definitions produced it. It is separate from the dataset and manifest schema versions.

The function assumes the question has passed the dataset validator. It makes only the checks it
needs to give a correct answer or to refuse, and does not repeat the validator's rules (the
unit cap, span contents, enumerations, identifiers). A question it cannot classify raises
``ValueError`` with a fixed message that never contains any part of the question. The function
is pure: it reads the question, does no I/O and modifies nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.rag.evaluation.schema import LANGUAGES, EvidenceUnit, Question

STRUCTURE_VERSION = "1"

UNIT_STRUCTURES = ("single_unit", "multi_unit")
RETRIEVAL_TYPES = ("en_to_en", "hi_to_hi", "en_to_hi", "hi_to_en")
CROSS_LANGUAGE_RETRIEVAL_TYPES = frozenset({"en_to_hi", "hi_to_en"})

_UNANSWERABLE_MESSAGE = "an unanswerable question has no structure"
_INVALID_MESSAGE = "the answerable question cannot be classified"


@dataclass(frozen=True, slots=True)
class QuestionStructure:
    """The derived structure of one answerable question."""

    unit_structure: str
    has_alternatives: bool
    retrieval_type: str


def _units_are_usable(units: object) -> bool:
    """A non-empty tuple of units, each with a non-empty tuple of spans (spans are not inspected)."""
    if type(units) is not tuple or not units:
        return False
    return all(
        isinstance(unit, EvidenceUnit) and type(unit.spans) is tuple and len(unit.spans) > 0
        for unit in units
    )


def _is_language(value: object) -> bool:
    return type(value) is str and value in LANGUAGES


def derive_structure(q: Question) -> QuestionStructure:
    """The structure of the answerable question ``q``.

    Raises ``TypeError`` if ``q`` is not a ``Question``. Raises ``ValueError`` for an
    unanswerable question, and for an answerable question that is not in a form that can be
    classified (``answerable`` not a bool, ``units`` not a non-empty tuple of units that each
    hold a non-empty tuple of spans, or either language not an exact ``en`` or ``hi``). The two
    ``ValueError`` messages are fixed and differ.
    """
    if not isinstance(q, Question):
        raise TypeError("q must be a Question")
    if q.answerable is False:
        raise ValueError(_UNANSWERABLE_MESSAGE)
    if (
        q.answerable is not True
        or not _units_are_usable(q.units)
        or not _is_language(q.query_language)
        or not _is_language(q.source_language)
    ):
        raise ValueError(_INVALID_MESSAGE)

    return QuestionStructure(
        unit_structure="single_unit" if len(q.units) == 1 else "multi_unit",
        has_alternatives=any(len(unit.spans) > 1 for unit in q.units),
        retrieval_type=f"{q.query_language}_to_{q.source_language}",
    )
