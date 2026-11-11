"""Core retrieval metrics for one answerable question (E1.3.1).

A ranking is a list of ``RankedChunk`` in rank order (the first item is rank 1). A question's
gold evidence is its ``units``: every unit is required (AND), and a unit is satisfied when any
one of its alternative spans is satisfied (OR). A chunk and a span are matched by
``corpus_key`` and by half-open character interval, the same convention as ``intervals``.
Source versions and hashes are not checked here (E1.7 does that), and ``text_sha256`` plays
no part in relevance or coverage.

Relevance (``RELEVANCE_RULES``): a chunk is relevant to one gold span when its overlap with
that span covers at least half of the span (``HALF_COVERAGE``, the default), any position of
it (``ANY_OVERLAP``), or all of it (``FULL_COVERAGE``). The rule is applied to one chunk and
one span at a time: overlap from several chunks is never added up to reach the threshold. It
is decided with integer arithmetic, so exactly 50% passes and 49% does not.

Metrics, all over the first ``k`` chunks (``k`` is 1 to ``MAX_RANK``; the usual cutoffs are
``CUTOFFS``):

  * ``hit_at_k``: 1 if at least one required unit is satisfied, else 0.
  * ``unit_recall_at_k``: satisfied units divided by required units.
  * ``evidence_coverage_at_k``: for a span, the length of the union of the chunks' overlaps
    with it divided by its length (overlapping chunks are never counted twice); for a unit,
    the largest value over its alternative spans; for the question, the mean over its units.
    Coverage does not depend on the relevance rule.
  * ``reciprocal_rank``: 1 / the rank of the first chunk that is relevant to any span of any
    unit, looking only at the first ``MAX_RANK`` chunks; 0 if there is none. (MRR is the mean
    of this over questions, computed in a later checkpoint.)

Distinct-passage metrics (``distinct_hit_at_k``, ``distinct_reciprocal_rank``) apply the same
definitions to the representatives of ``passages.representatives`` instead of to the raw chunks:
the ranking is collapsed to one chunk per passage (see that module for the rules, including the
consequence of building passages from all 20 chunks), and ``k`` and the reciprocal rank count
positions in that compressed ranking, not original ranks. Relevance is judged on the
representative chunk alone. These are separate from the raw metrics, which stay primary. Because
the compressed ranking can move a relevant passage up, a distinct Hit@k can exceed the raw Hit@k.

Fractions are computed exactly and converted to ``float`` once at the end.

Errors follow the E1.2 convention of ``near_duplicates`` and ``structure``: ``TypeError`` for an
argument of the wrong type (not a ``Question``, ``Span`` or ``RankedChunk``; a container that is
not a list or tuple; a string, offset or ``k`` that is not exactly ``str`` or ``int``, so a
``bool`` is a type error), ``ValueError`` for a value of the right type that is not allowed (an
unanswerable question, no units or spans, an interval that is not ``0 <= start < end``, ``k``
outside ``1..MAX_RANK``, an unknown rule name). ``intervals`` itself raises ``ValueError`` for
both; the split here is deliberate. Messages are fixed text and never contain input values.
Nothing here modifies or reorders its arguments.

``RankedChunk`` accepts any ``str`` for ``corpus_key`` and ``text_sha256``, including an empty
one: the corpus key and hash patterns of E1.2 belong to labelled spans (``validate_span``), and
a chunk key that matches no gold span is simply never relevant.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from app.rag.evaluation.schema import EvidenceUnit, Question, Span

MAX_RANK = 20
CUTOFFS = (1, 3, 5, 10, 20)

ANY_OVERLAP = "any_overlap"
HALF_COVERAGE = "half_coverage"
FULL_COVERAGE = "full_coverage"
RELEVANCE_RULES = (ANY_OVERLAP, HALF_COVERAGE, FULL_COVERAGE)
DEFAULT_RULE = HALF_COVERAGE

_OFFSET_TYPE_MESSAGE = "an interval must be two plain integers"
_OFFSET_RANGE_MESSAGE = "an interval must satisfy 0 <= start < end"
_CHUNK_TEXT_MESSAGE = "corpus_key and text_sha256 must be str"


def _check_offsets(start: object, end: object) -> None:
    if type(start) is not int or type(end) is not int:
        raise TypeError(_OFFSET_TYPE_MESSAGE)
    if not 0 <= start < end:
        raise ValueError(_OFFSET_RANGE_MESSAGE)


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """One retrieved chunk as the metrics see it: a document key and a half-open interval.

    ``text_sha256`` is carried for the baselines of a later checkpoint; it is never used in
    relevance or coverage. Construction rejects a non-``str`` key or hash and any interval
    that is not two plain ``int`` values (``bool`` is not accepted: ``TypeError``) with
    ``0 <= start < end`` (``ValueError``).
    """

    corpus_key: str
    start: int
    end: int
    text_sha256: str

    def __post_init__(self) -> None:
        if type(self.corpus_key) is not str or type(self.text_sha256) is not str:
            raise TypeError(_CHUNK_TEXT_MESSAGE)
        _check_offsets(self.start, self.end)


# ---------- interval helpers ----------

def overlap_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    """The number of positions shared by two half-open intervals (0 if they only touch)."""
    for interval in (a, b):
        if not isinstance(interval, (tuple, list)):
            raise TypeError("an interval must be a tuple or list")
        if len(interval) != 2:
            raise ValueError("an interval must have exactly two items")
        _check_offsets(interval[0], interval[1])
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _check_rule(rule: object) -> None:
    if type(rule) is not str:
        raise TypeError("rule must be a str")
    if rule not in RELEVANCE_RULES:
        raise ValueError("unknown relevance rule")


def _check_span(span: object) -> Span:
    if not isinstance(span, Span):
        raise TypeError("an evidence span must be a Span")
    if type(span.corpus_key) is not str:
        raise TypeError("an evidence span's corpus_key must be a str")
    _check_offsets(span.start, span.end)
    return span


def _check_chunk(chunk: object) -> RankedChunk:
    if not isinstance(chunk, RankedChunk):
        raise TypeError("a ranking must contain only RankedChunk items")
    return chunk


def is_relevant(chunk: RankedChunk, span: Span, rule: str = DEFAULT_RULE) -> bool:
    """True if ``chunk`` is relevant to the gold ``span`` under ``rule`` (one chunk, one span)."""
    _check_rule(rule)
    chunk = _check_chunk(chunk)
    span = _check_span(span)
    return _relevant(chunk, span, rule)


def _relevant(chunk: RankedChunk, span: Span, rule: str) -> bool:
    if chunk.corpus_key != span.corpus_key:
        return False
    shared = overlap_length((chunk.start, chunk.end), (span.start, span.end))
    if rule == ANY_OVERLAP:
        return shared > 0
    if rule == FULL_COVERAGE:
        return shared == span.end - span.start
    return 2 * shared >= span.end - span.start


def span_coverage(chunks: Sequence[RankedChunk], span: Span) -> Fraction:
    """The share of ``span`` covered by the union of ``chunks`` (exact; overlaps counted once).

    ``chunks`` must be a list or tuple of ``RankedChunk`` (possibly empty); any other container is
    a ``TypeError``, so a wrong argument is never mistaken for "no coverage".
    """
    span = _check_span(span)
    if not isinstance(chunks, (list, tuple)):
        raise TypeError("chunks must be a list or tuple of RankedChunk")
    pieces = []
    for chunk in chunks:
        chunk = _check_chunk(chunk)
        if chunk.corpus_key != span.corpus_key:
            continue
        low, high = max(chunk.start, span.start), min(chunk.end, span.end)
        if low < high:
            pieces.append((low, high))
    pieces.sort()
    covered, reach = 0, span.start
    for low, high in pieces:
        low = max(low, reach)
        if high > low:
            covered += high - low
            reach = high
    return Fraction(covered, span.end - span.start)


# ---------- argument checking ----------

def _units(question: object) -> tuple[EvidenceUnit, ...]:
    if not isinstance(question, Question):
        raise TypeError("question must be a Question")
    if type(question.answerable) is not bool:
        raise TypeError("answerable must be a bool")
    if question.answerable is False:
        raise ValueError("the question must be answerable")
    units = question.units
    if type(units) is not tuple:
        raise TypeError("units must be a tuple of EvidenceUnit")
    if not units:
        raise ValueError("the question has no evidence units")
    for unit in units:
        if not isinstance(unit, EvidenceUnit):
            raise TypeError("units must be a tuple of EvidenceUnit")
        if type(unit.spans) is not tuple:
            raise TypeError("a unit's spans must be a tuple of Span")
        if not unit.spans:
            raise ValueError("an evidence unit has no spans")
        for span in unit.spans:
            _check_span(span)
    return units


def _check_k(k: object) -> int:
    if type(k) is not int:
        raise TypeError("k must be an int")
    if not 1 <= k <= MAX_RANK:
        raise ValueError("k must be from 1 to the maximum rank")
    return k


def _top(ranking: object, k: object) -> list[RankedChunk]:
    _check_k(k)
    if not isinstance(ranking, (list, tuple)):
        raise TypeError("ranking must be a list or tuple of RankedChunk")
    return [_check_chunk(chunk) for chunk in ranking[:k]]


def _unit_satisfied(unit: EvidenceUnit, chunks: Sequence[RankedChunk], rule: str) -> bool:
    return any(_relevant(chunk, span, rule) for span in unit.spans for chunk in chunks)


# ---------- metrics ----------

def hit_at_k(question: Question, ranking: Sequence[RankedChunk], k: int, rule: str = DEFAULT_RULE) -> int:
    """1 if at least one required unit is satisfied within the first ``k`` chunks, else 0."""
    _check_rule(rule)
    units = _units(question)
    chunks = _top(ranking, k)
    return 1 if any(_unit_satisfied(unit, chunks, rule) for unit in units) else 0


def unit_recall_at_k(question: Question, ranking: Sequence[RankedChunk], k: int, rule: str = DEFAULT_RULE) -> float:
    """The share of required units satisfied within the first ``k`` chunks."""
    _check_rule(rule)
    units = _units(question)
    chunks = _top(ranking, k)
    return float(Fraction(sum(_unit_satisfied(unit, chunks, rule) for unit in units), len(units)))


def evidence_coverage_at_k(question: Question, ranking: Sequence[RankedChunk], k: int) -> float:
    """The mean over units of the best span coverage by the first ``k`` chunks."""
    units = _units(question)
    chunks = _top(ranking, k)
    total = sum((max(span_coverage(chunks, span) for span in unit.spans) for unit in units), Fraction(0))
    return float(total / len(units))


def reciprocal_rank(question: Question, ranking: Sequence[RankedChunk], rule: str = DEFAULT_RULE) -> float:
    """1 / the rank of the first relevant chunk within the first ``MAX_RANK``, else 0.0."""
    _check_rule(rule)
    units = _units(question)
    chunks = _top(ranking, MAX_RANK)
    spans = [span for unit in units for span in unit.spans]
    for rank, chunk in enumerate(chunks, start=1):
        if any(_relevant(chunk, span, rule) for span in spans):
            return float(Fraction(1, rank))
    return 0.0


# ---------- distinct-passage metrics ----------

def distinct_hit_at_k(question: Question, ranking: Sequence[RankedChunk], k: int, rule: str = DEFAULT_RULE) -> int:
    """1 if a required unit is satisfied by one of the first ``k`` representatives, else 0.

    ``k`` counts positions in the compressed representative ranking (see ``passages``).
    """
    from app.rag.evaluation.passages import representatives

    _check_rule(rule)
    units = _units(question)
    _check_k(k)
    chunks = [item.chunk for item in representatives(ranking)[:k]]
    return 1 if any(_unit_satisfied(unit, chunks, rule) for unit in units) else 0


def distinct_reciprocal_rank(question: Question, ranking: Sequence[RankedChunk], rule: str = DEFAULT_RULE) -> float:
    """1 / the compressed position of the first relevant representative, else 0.0."""
    from app.rag.evaluation.passages import representatives

    _check_rule(rule)
    units = _units(question)
    spans = [span for unit in units for span in unit.spans]
    for position, item in enumerate(representatives(ranking), start=1):
        if any(_relevant(item.chunk, span, rule) for span in spans):
            return float(Fraction(1, position))
    return 0.0
