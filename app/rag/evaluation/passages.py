"""Distinct passages of a ranking (E1.3.2).

Overlapping or touching chunks of one document are really one passage of text shown more than
once. ``representatives`` collapses a ranking to one chunk per passage:

  1. Only the first ``MAX_RANK`` (20) chunks are looked at, once, whatever cutoff ``k`` a
     metric later asks for.
  2. Chunks are grouped by ``corpus_key``; chunks of different documents never join.
  3. Within a document, chunks whose half-open intervals overlap or touch (``intervals.collides``:
     the intervals share a position, or one ends exactly where the next starts) are connected,
     and connectivity is transitive: if A meets B and B meets C, all three are one component
     even when A and C do not meet. A one-position gap separates components. Duplicate chunks
     are one component.
  4. Each component is represented by its highest-ranked chunk (lowest original rank). Later
     members add nothing: they are not relevant on the representative's behalf, and their
     intervals are never combined with it.
  5. The representatives are returned in their original relative order. Their position in that
     list (1 for the first) is the "compressed" rank used by the distinct metrics in
     ``metrics``; ``original_rank`` keeps the 1-based position in the input ranking, so nothing
     is lost for tracing. The two are different numbers: do not use one for the other.

Consequence of building the components once from all 20 chunks (not once per cutoff): a chunk at
rank 20 can connect two earlier chunks that do not meet each other, merging them into one
passage whose representative is the earlier of the two. A relevant chunk that was a
representative of its own passage at smaller cutoffs then disappears from the compressed ranking,
so a distinct metric at a small ``k`` depends on chunks beyond ``k``. Chunks after rank 20 are
never inspected, so a bridge at rank 21 changes nothing.

Errors (as in ``metrics``): ``TypeError`` for a ranking that is not a list or tuple or an item
that is not a ``RankedChunk`` among the first 20; ``ValueError`` for an ``original_rank`` outside
``1..MAX_RANK`` (representatives come only from the first ``MAX_RANK`` chunks). Messages are fixed text. Nothing is modified.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.evaluation.intervals import collides
from app.rag.evaluation.metrics import MAX_RANK, RankedChunk, _top


@dataclass(frozen=True, slots=True)
class Representative:
    """The chunk standing for one passage, and where it was in the input ranking (1-based)."""

    original_rank: int
    chunk: RankedChunk

    def __post_init__(self) -> None:
        if type(self.original_rank) is not int:
            raise TypeError("original_rank must be an int")
        if not isinstance(self.chunk, RankedChunk):
            raise TypeError("chunk must be a RankedChunk")
        if not 1 <= self.original_rank <= MAX_RANK:
            raise ValueError("original_rank must be from 1 to the maximum rank")


def representatives(ranking: Sequence[RankedChunk]) -> tuple[Representative, ...]:
    """One ``Representative`` per passage of the first ``MAX_RANK`` chunks, in original rank order."""
    chunks = _top(ranking, MAX_RANK)
    by_document: dict[str, list[tuple[int, int, int]]] = {}
    for rank, chunk in enumerate(chunks, start=1):
        by_document.setdefault(chunk.corpus_key, []).append((chunk.start, rank, chunk.end))

    best_ranks: list[int] = []
    for items in by_document.values():
        items.sort()  # by start, then rank: a fixed order
        merged_start, merged_end, best = items[0][0], items[0][2], items[0][1]
        for start, rank, end in items[1:]:
            if collides((merged_start, merged_end), (start, end)):
                merged_end = max(merged_end, end)
                best = min(best, rank)
            else:
                best_ranks.append(best)
                merged_start, merged_end, best = start, end, rank
        best_ranks.append(best)

    return tuple(Representative(rank, chunks[rank - 1]) for rank in sorted(best_ranks))
