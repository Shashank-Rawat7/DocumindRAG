"""Tokenizer content-token offsets -> weighted source units for the chunker (om-v1).

Pure function, no model, no I/O. Input is the output of
``tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)``
plus ``convert_ids_to_tokens`` for the same ids. Special tokens must not be
included.

The result is a sequence of non-overlapping half-open code-point spans of the
ORIGINAL text, each carrying the number of model tokens it represents. Pass it
to ``chunk_text(text, spans, config, token_counts=counts)``.

om-v1 rules (anything else raises OffsetMappingError; nothing is guessed):
  * Every offset is a pair of Python ints with 0 <= start < end <= len(text).
    Zero-length offsets are rejected.
  * Token starts never decrease.
  * Verified Metaspace pattern: a token "\u2581" at (s, s+1), where text[s] is
    not whitespace, immediately followed by a non-"\u2581" token starting at s
    and ending at or after s+1, becomes one unit (s, end) with token_count 2.
  * Every other token becomes its own unit with token_count 1 and must not
    overlap the previous unit.
  * Uncovered characters that are ignorable (text_boundaries.is_ignorable) are
    never absorbed: they stay uncovered. Uncovered non-ignorable characters are
    absorbed into the unit ending immediately before them only when they
    continue that unit's grapheme cluster (combining marks, variation
    selectors, keycap, skin-tone modifiers) and no ignorable character
    separates them from the unit; otherwise the mapping fails.
  * The sum of token counts equals the number of offsets.
The source text is never modified or normalised.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.text_boundaries import (
    GRAPHEME_SEGMENTATION,
    grapheme_boundaries,
    is_cluster_extender,
    is_ignorable,
)

METASPACE = "\u2581"
OFFSET_MAPPING_VERSION = "om-v1"


class OffsetMappingError(ValueError):
    """Offsets that om-v1 cannot map safely. ``reason`` is a fixed code; the
    message never contains document text."""

    def __init__(self, reason: str, position: int | None = None):
        self.reason = reason
        self.position = position
        super().__init__(reason if position is None else f"{reason} at offset {position}")


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """A half-open code-point span of the original text and the number of model
    tokens it represents. ``first_token`` indexes the content-token sequence."""

    start: int
    end: int
    token_count: int
    first_token: int


@dataclass(frozen=True, slots=True)
class OffsetMapping:
    units: tuple[SourceUnit, ...]
    total_tokens: int          # always equals the number of input offsets
    metaspace_groups: int
    expanded_characters: int


def signature_fields() -> dict:
    """Identity of the mapping rules, for documents.indexed_signature (C2.5)."""
    return {
        "offset_mapping_version": OFFSET_MAPPING_VERSION,
        "overlap_rule": "metaspace_prefix_same_start",
        "zero_length_offsets": "reject",
        "grapheme_expansion": "same_cluster_extenders_only",
        "grapheme_segmentation": GRAPHEME_SEGMENTATION,
    }


def map_token_offsets(
    text: str,
    offsets: Sequence[tuple[int, int]],
    tokens: Sequence[str],
) -> OffsetMapping:
    """Map content-token offsets to weighted, non-overlapping source units."""
    pairs = _validate(text, offsets, tokens)
    units, metaspace_groups = _group(text, pairs, tokens)
    expanded = _expand(text, units)
    _check(text, units, len(pairs))
    return OffsetMapping(
        units=tuple(SourceUnit(s, e, n, first) for s, e, n, first in units),
        total_tokens=len(pairs),
        metaspace_groups=metaspace_groups,
        expanded_characters=expanded,
    )


# ---------- step 1: validation ----------

def _validate(text, offsets, tokens) -> list[tuple[int, int]]:
    if type(text) is not str:
        raise OffsetMappingError("text_not_str")
    for name, value in (("offsets", offsets), ("tokens", tokens)):
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise OffsetMappingError(f"{name}_not_sequence")
    if len(offsets) != len(tokens):
        raise OffsetMappingError("length_mismatch")
    pairs: list[tuple[int, int]] = []
    previous_start = 0
    for i, item in enumerate(offsets):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise OffsetMappingError("malformed_offset", i)
        s, e = item
        if type(s) is not int or type(e) is not int:
            raise OffsetMappingError("non_int_offset", i)
        if type(tokens[i]) is not str:
            raise OffsetMappingError("non_str_token", i)
        if not 0 <= s <= e <= len(text):
            raise OffsetMappingError("out_of_range", s)
        if s == e:
            raise OffsetMappingError("zero_length_offset", s)
        if s < previous_start:
            raise OffsetMappingError("non_monotonic", s)
        previous_start = s
        pairs.append((s, e))
    return pairs


# ---------- step 2: grouping ----------

def _group(text, pairs, tokens) -> tuple[list[list[int]], int]:
    """Units as mutable [start, end, token_count, first_token]."""
    units: list[list[int]] = []
    metaspace_groups = 0
    previous_end = 0
    i = 0
    while i < len(pairs):
        s, e = pairs[i]
        if _is_metaspace_prefix(text, pairs, tokens, i, previous_end):
            end = pairs[i + 1][1]
            units.append([s, end, 2, i])
            metaspace_groups += 1
            previous_end = end
            i += 2
            continue
        if s < previous_end:
            raise OffsetMappingError("unverified_overlap", s)
        units.append([s, e, 1, i])
        previous_end = e
        i += 1
    return units, metaspace_groups


def _is_metaspace_prefix(text, pairs, tokens, i, previous_end) -> bool:
    s, e = pairs[i]
    if tokens[i] != METASPACE or e != s + 1 or s < previous_end or text[s].isspace():
        return False
    if i + 1 >= len(pairs) or tokens[i + 1] == METASPACE:
        return False
    s2, e2 = pairs[i + 1]
    return s2 == s and e2 >= s + 1


# ---------- step 3: grapheme expansion ----------

def _expand(text, units) -> int:
    boundaries = grapheme_boundaries(text)
    covered = bytearray(len(text))
    for s, e, _, _ in units:
        covered[s:e] = b"\x01" * (e - s)
    unit_ending_at = {u[1]: u for u in units}
    expanded = 0
    pos = 0
    while pos < len(text):
        if covered[pos]:
            pos += 1
            continue
        run_start = pos
        while pos < len(text) and not covered[pos]:
            pos += 1
        run_end = pos
        segment_start = run_start
        while segment_start < run_end:  # judge each grapheme-cluster piece separately
            segment_end = segment_start + 1
            while segment_end < run_end and not boundaries[segment_end]:
                segment_end += 1
            segment = text[segment_start:segment_end]
            if not all(is_ignorable(ch) for ch in segment):
                # Only the non-ignorable characters directly after the unit may be absorbed;
                # ignorable characters are never absorbed and stay uncovered.
                head = 0
                while head < len(segment) and not is_ignorable(segment[head]):
                    head += 1

                for k in range(head, len(segment)):
                    if not is_ignorable(segment[k]):
                        raise OffsetMappingError("uncovered_character", segment_start + k)

                unit = _absorbing_unit(
                    text, boundaries, unit_ending_at, segment_start, segment[:head]
                )
                del unit_ending_at[unit[1]]
                unit[1] = segment_start + head
                unit_ending_at[unit[1]] = unit
                expanded += head

            segment_start = segment_end
    return expanded


def _absorbing_unit(text, boundaries, unit_ending_at, start, segment):
    """The unit that may absorb ``segment`` (uncovered, not all ignorable), or raise."""
    unit = unit_ending_at.get(start)
    cluster_start = start
    while cluster_start > 0 and not boundaries[cluster_start]:
        cluster_start -= 1
    if (
        unit is None
        or boundaries[start]                      # the segment starts its own cluster
        or unit[1] <= cluster_start               # the unit is not part of that cluster
        or text[cluster_start].isspace()          # the cluster's base is whitespace
        or any(ch.isspace() for ch in segment)
        or not all(is_ignorable(ch) or is_cluster_extender(ch) for ch in segment)
    ):
        bad = start + next(k for k, ch in enumerate(segment) if not is_ignorable(ch))
        raise OffsetMappingError("uncovered_character", bad)
    return unit


# ---------- step 4: coverage and conservation ----------

def _check(text, units, total_tokens) -> None:
    covered = bytearray(len(text))
    previous_end = 0
    next_first = 0
    for s, e, count, first in units:
        if not (previous_end <= s < e <= len(text)) or count < 1 or first != next_first:
            raise OffsetMappingError("internal_unit_error", s)
        covered[s:e] = b"\x01" * (e - s)
        previous_end, next_first = e, first + count
    for pos, ch in enumerate(text):
        if not covered[pos] and not is_ignorable(ch):
            raise OffsetMappingError("internal_coverage_error", pos)
    if next_first != total_tokens:
        raise OffsetMappingError("count_mismatch")