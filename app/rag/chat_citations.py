"""Citation extraction, validation and mapping for chat answers (C9.2): pure functions.

The mapping rule
----------------
``resolve_citations`` takes the ``PromptSource`` tuple that ``build_prompt`` produced. A
marker ``[n]`` in the answer is valid only if ``n`` is one of the sources actually rendered
into the prompt, and it resolves to the very ``RetrievedChunk`` object held by that source.
Nothing is reconstructed from chunk ids, scores or the original retrieval list, so a chunk
that was retrieved but dropped for budget can never be cited.

What counts as a marker
-----------------------
Only an exact ``[n]`` with one to three ASCII digits, outside Markdown code. ``[2020]``
(four digits), ``[1, 2]``, ``[1-3]`` and ``[a]`` are ordinary text and are left untouched.
A marker is valid iff it has no leading zero and ``1 <= n <= k`` for ``k`` rendered sources.
Invalid markers (``[0]``, ``[007]``, ``[9]`` with three sources) are removed.

Markdown code
-------------
Text inside inline code spans and fenced blocks is literal: markers there are neither
counted nor removed. Code is detected CommonMark-style (a backtick run closes at the next run
of the same length). A backtick run with no closing run is treated as literal text, so an
unterminated fence does not protect the rest of the answer.

Removal never normalizes whitespace globally. Only the removed marker group (consecutive
invalid markers, optionally separated by spaces or tabs) and, where needed, the horizontal
whitespace directly before it are deleted:

* before punctuation, a closing bracket or quote, the end of a line, or a closing emphasis
  run (``*``, ``_`` or ``~`` not followed by a letter or digit), the space before the group
  goes too (``claim [9].`` -> ``claim.``, ``**bold [9]**`` -> ``**bold**``);
* before a space, the group and the space before it go (``a [9] b`` -> ``a b``);
* before anything else (a letter, digit, opening bracket or quote, or an opening emphasis
  run such as ``*italic*``) the space before the group stays (``a [9]b`` -> ``a b``,
  ``x [9][1]`` -> ``x [1]``, ``x [9]*it*`` -> ``x *it*``);
* a group at the start of a line keeps the line's indentation and drops the space after it
  (``    [9] x`` -> ``    x``).

Valid markers and every other character are kept byte-for-byte. The returned text is not
stripped: it may be empty or blank, and the caller decides what that means.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.chat_prompt import PromptSource
from app.rag.retrieval import RetrievedChunk

_MARKER = re.compile(r"\[([0-9]{1,3})\]")
_HORIZONTAL_SPACE = " \t"
_LINE_BREAKS = "\r\n"
# Characters that close or end a phrase: the space before a removed marker is dropped.
# Written as escapes: ideographic and fullwidth punctuation, closing quotes and guillemet,
# Devanagari danda and double danda, and the ellipsis character.
_CLOSERS = frozenset(
    ".,;:!?)]}\"'"
    + "\u3002\uff0c\u3001\uff1b\uff1a\uff01\uff1f\uff09\u3011\u300d\u300f\u300b\u201d\u2019\u00bb\u0964\u0965\u2026"
)


_EMPHASIS_CHARS = "*_~"


def _closes_emphasis(text: str, index: int) -> bool:
    """True if a run of ``*``, ``_`` or ``~`` starts at ``index`` and closes emphasis.

    A closing run is not followed by a letter or digit (``**bold**`` then space, punctuation
    or end of text); an opening run is (``*italic*``), and must keep the space before it.
    """
    end = index
    while end < len(text) and text[end] in _EMPHASIS_CHARS:
        end += 1
    if end == index:
        return False
    return end == len(text) or not text[end].isalnum()


class CitationInputError(Exception):
    """The answer or the sources are malformed. Messages never contain answer text."""


@dataclass(frozen=True, slots=True)
class Citation:
    """A valid marker: ``chunk`` is the object rendered as source ``marker``."""

    ordinal: int  # 1-based, order of first appearance in the answer
    marker: int  # the n in [n]
    chunk: RetrievedChunk


@dataclass(frozen=True, slots=True)
class CitationResult:
    text: str  # the answer with invalid markers removed
    citations: tuple[Citation, ...]  # distinct valid markers, by first appearance
    invalid_marker_count: int  # invalid marker occurrences removed from the text


def _check_sources(sources: object) -> dict[int, PromptSource]:
    if type(sources) not in (list, tuple):
        raise CitationInputError("sources must be a list or tuple of PromptSource")
    by_marker: dict[int, PromptSource] = {}
    for expected, source in enumerate(sources, start=1):
        if type(source) is not PromptSource:
            raise CitationInputError("every source must be a PromptSource")
        if type(source.marker) is not int or source.marker != expected:
            raise CitationInputError("source markers must be 1..k in order")
        if type(source.chunk) is not RetrievedChunk:
            raise CitationInputError("every source must hold a RetrievedChunk")
        by_marker[source.marker] = source
    return by_marker


def _code_regions(text: str) -> list[tuple[int, int]]:
    """Half-open spans of inline code and fenced code, found CommonMark-style."""
    regions: list[tuple[int, int]] = []
    length = len(text)
    index = 0
    while index < length:
        if text[index] != "`":
            index += 1
            continue
        run_end = index
        while run_end < length and text[run_end] == "`":
            run_end += 1
        run_length = run_end - index
        closing_end = -1
        cursor = run_end
        while cursor < length:
            if text[cursor] != "`":
                cursor += 1
                continue
            candidate_end = cursor
            while candidate_end < length and text[candidate_end] == "`":
                candidate_end += 1
            if candidate_end - cursor == run_length:
                closing_end = candidate_end
                break
            cursor = candidate_end
        if closing_end == -1:
            index = run_end
            continue
        regions.append((index, closing_end))
        index = closing_end
    return regions


def _inside(position: int, starts: list[int], regions: list[tuple[int, int]]) -> bool:
    slot = bisect.bisect_right(starts, position) - 1
    return slot >= 0 and position < regions[slot][1]


def _group_deletion(text: str, group_start: int, group_end: int) -> tuple[int, int]:
    """The half-open span to delete for one removed marker group."""
    pre_start = group_start
    while pre_start > 0 and text[pre_start - 1] in _HORIZONTAL_SPACE:
        pre_start -= 1
    at_line_start = pre_start == 0 or text[pre_start - 1] in _LINE_BREAKS
    has_text_before = not at_line_start
    has_pre_space = pre_start < group_start
    post = text[group_end] if group_end < len(text) else None

    # The space before the group goes only where closing text follows; anywhere else
    # (a letter, an opening bracket or quote, emphasis marks) it stays, so no word is glued
    # to its neighbour and nothing that opens a construct loses its separator.
    remove_pre = False
    extend_post = False
    if (
        post is None
        or post in _LINE_BREAKS
        or post in _CLOSERS
        or _closes_emphasis(text, group_end)
    ):
        remove_pre = has_text_before
    elif post in _HORIZONTAL_SPACE:
        remove_pre = has_text_before
        extend_post = at_line_start

    delete_start = pre_start if (remove_pre and has_pre_space) else group_start
    delete_end = group_end
    if extend_post:
        while delete_end < len(text) and text[delete_end] in _HORIZONTAL_SPACE:
            delete_end += 1
    return delete_start, delete_end


def resolve_citations(answer: str, sources: Sequence[PromptSource]) -> CitationResult:
    """Validate the answer's ``[n]`` markers against ``sources`` and resolve them."""
    if type(answer) is not str:
        raise CitationInputError("answer must be a str")
    by_marker = _check_sources(sources)

    regions = _code_regions(answer)
    starts = [start for start, _ in regions]

    citations: list[Citation] = []
    seen_markers: set[int] = set()
    # Invalid marker groups as [start, end): consecutive invalid markers, optionally
    # separated by spaces or tabs only.
    groups: list[list[int]] = []
    invalid_count = 0

    for match in _MARKER.finditer(answer):
        if _inside(match.start(), starts, regions):
            continue
        raw = match.group(1)
        number = int(raw)
        if raw == str(number) and number in by_marker:
            if number not in seen_markers:
                seen_markers.add(number)
                citations.append(
                    Citation(
                        ordinal=len(citations) + 1,
                        marker=number,
                        chunk=by_marker[number].chunk,
                    )
                )
            continue
        invalid_count += 1
        if (
            groups
            and answer[groups[-1][1] : match.start()].strip(_HORIZONTAL_SPACE) == ""
        ):
            groups[-1][1] = match.end()
        else:
            groups.append([match.start(), match.end()])

    if not groups:
        return CitationResult(
            text=answer, citations=tuple(citations), invalid_marker_count=invalid_count
        )

    pieces: list[str] = []
    cursor = 0
    for group_start, group_end in groups:
        delete_start, delete_end = _group_deletion(answer, group_start, group_end)
        delete_start = max(delete_start, cursor)
        pieces.append(answer[cursor:delete_start])
        cursor = delete_end
    pieces.append(answer[cursor:])
    return CitationResult(
        text="".join(pieces),
        citations=tuple(citations),
        invalid_marker_count=invalid_count,
    )
