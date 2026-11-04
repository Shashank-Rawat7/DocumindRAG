"""Token-budgeted, boundary-aware chunking with exact source offsets (C2.1).

The chunker is a pure function. It does not load a model and has no I/O.
The caller supplies the document text together with the character span of
every *content* token of that text, i.e. tokenizer offsets obtained with
``add_special_tokens=False`` (C2.4 provides them from the model tokenizer).

Input contract (violations raise ChunkingError):
  * ``text`` is a ``str``.
  * ``token_spans`` is a sequence of 2-item tuples/lists of Python ``int``
    (``bool``, ``float``, ``str`` and other types are rejected, never coerced).
  * Every span satisfies ``0 <= start < end <= len(text)`` (zero-length spans
    are rejected), spans are in order and do not overlap.
  * Every character outside all spans is whitespace or a control/format
    character (Unicode categories Cc, Cf, Zs, Zl, Zp); content is never dropped.

Guarantees for every returned ChunkSpan ``c`` (all verified before returning):
  * ``c.text == text[c.start:c.end]`` (Python code-point indices, half-open).
  * ``c.text`` is non-empty and has no leading or trailing whitespace.
  * ``c.token_count == c.token_end - c.token_start <= config.max_tokens``.
  * ``c.span_sha256 == sha256(c.text.encode("utf-8")).hexdigest()``.
  * Starts, ends, token_starts and token_ends strictly increase chunk to chunk.
  * Both character edges (``c.start`` and ``c.end``) are Unicode extended
    grapheme cluster boundaries as defined by UAX #29 and implemented by the
    pinned ``regex`` package (``\\X``), so no grapheme cluster is ever split:
    Hangul L+V(+T) syllables, regional-indicator flags, CR LF, emoji ZWJ
    sequences, modifiers, combining marks and Devanagari conjuncts (GB9c).
  * Every non-ignorable character of ``text`` lies inside at least one chunk.
If no grapheme-cluster-safe boundary exists within ``max_tokens``, or a chunk
edge at the very start or end of the text would split a cluster (an isolated
combining mark attached to whitespace), ChunkingError is raised and nothing
is returned. The text is never normalised, collapsed or truncated.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import regex

# Bump whenever the algorithm changes in a way that can change chunk output.
CHUNKER_VERSION = "c2.1-v2"

SENTENCE_TERMINATORS = frozenset(".!?\u2026\u0964\u0965")  # . ! ? and the ellipsis, danda, double danda
CLOSING_PUNCTUATION = frozenset("\"')]}\u00bb\u201d\u2019")  # closing quotes and brackets

# Unicode extended grapheme clusters (UAX #29), implemented by the regex package.
_GRAPHEME_CLUSTER = regex.compile(r"\X")
GRAPHEME_SEGMENTATION = f"regex-{regex.__version__}-X-uax29-extended-grapheme-clusters"

# Ranks of the cut *before* token k (between token k-1 and token k).
RANK_HARD = 0       # no whitespace at the cut: inside a word / between sub-word pieces
RANK_SPACE = 1      # whitespace at the cut
RANK_SENTENCE = 2   # whitespace at the cut, preceded by a sentence terminator
RANK_LINE = 3       # the whitespace at the cut contains a line break
RANK_PARAGRAPH = 4  # the whitespace at the cut contains two or more line breaks
RANK_END = 5        # end of the text


class ChunkingError(ValueError):
    """Invalid input, no grapheme-safe boundary, or a violated invariant."""


def sha256_text(value: str) -> str:
    """Lowercase hex SHA-256 of the UTF-8 encoding (used for span_sha256)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Token budgets, counted in content tokens only (see ChunkSpan)."""

    target_tokens: int = 300
    max_tokens: int = 400
    overlap_tokens: int = 50

    def __post_init__(self) -> None:
        for name in ("target_tokens", "max_tokens", "overlap_tokens"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an int")
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be >= 1")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < target_tokens")

    @property
    def min_preferred_tokens(self) -> int:
        """Smallest chunk size considered in the preferred boundary search."""
        return max(1, self.target_tokens // 2)

    def signature_fields(self) -> dict:
        """Everything about chunking that identifies an indexing version.

        C2.5 combines this with the tokenizer and embedding identity to build
        documents.indexed_signature.
        """
        return {
            "chunker_version": CHUNKER_VERSION,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "token_count_scope": "content_tokens_only",
            "token_assignment": "contiguous_token_range_before_edge_trim",
            "grapheme_segmentation": GRAPHEME_SEGMENTATION,
        }


DEFAULT_CHUNKER_CONFIG = ChunkerConfig()


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """One chunk: an exact slice of the source text.

    The chunk owns the contiguous token range ``[token_start, token_end)`` of
    the document-level tokenization (content tokens, add_special_tokens=False).
    ``start``/``end`` are that range's character extent with leading and
    trailing whitespace trimmed. ``token_count == token_end - token_start``:
    tokens whose span includes whitespace that was trimmed (for example a
    SentencePiece token " world") are still counted, so the count never
    under-states. It excludes the "passage: " prefix and special tokens; C2.4
    tokenizes the final model input separately and rejects anything over 512.
    """

    index: int
    start: int
    end: int
    text: str
    span_sha256: str
    token_start: int
    token_end: int
    token_count: int


def chunk_text(
    text: str,
    token_spans: Sequence[tuple[int, int]],
    config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG,
) -> list[ChunkSpan]:
    """Split ``text`` into overlapping chunks along natural boundaries."""
    if not isinstance(text, str):
        raise ChunkingError("text must be a str")
    if not isinstance(config, ChunkerConfig):
        raise ChunkingError("config must be a ChunkerConfig")
    spans = _validate_token_spans(text, token_spans)
    boundaries = _grapheme_boundaries(text)

    n = len(spans)
    chunks: list[ChunkSpan] = []
    if n == 0:
        return chunks

    i = 0                # first token of the current chunk
    previous_end = 0     # exclusive end token of the previous chunk
    while True:
        j = _choose_end(text, spans, boundaries, i, previous_end, config)
        start, end = _trim(text, spans[i][0], spans[j - 1][1])
        if start < end:
            chunks.append(
                ChunkSpan(
                    index=len(chunks),
                    start=start,
                    end=end,
                    text=text[start:end],
                    span_sha256=sha256_text(text[start:end]),
                    token_start=i,
                    token_end=j,
                    token_count=j - i,
                )
            )
        if j == n:
            break
        i = _choose_next_start(text, spans, boundaries, i, j, config)
        previous_end = j

    _check_result(text, spans, boundaries, chunks, config)
    return chunks


# ---------- input validation ----------

def _validate_token_spans(text: str, token_spans) -> list[tuple[int, int]]:
    if isinstance(token_spans, (str, bytes)) or not isinstance(token_spans, Sequence):
        raise ChunkingError("token_spans must be a sequence of (start, end) pairs")
    spans: list[tuple[int, int]] = []
    covered = bytearray(len(text))
    previous_end = 0
    for k, item in enumerate(token_spans):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ChunkingError(f"token {k} must be a (start, end) pair, got {item!r}")
        s, e = item
        if type(s) is not int or type(e) is not int:
            raise ChunkingError(f"token {k} offsets must be Python ints, got {item!r}")
        if not 0 <= s < e <= len(text):
            raise ChunkingError(
                f"token {k} span ({s}, {e}) must satisfy 0 <= start < end <= {len(text)}"
            )
        if s < previous_end:
            raise ChunkingError(f"token {k} span ({s}, {e}) overlaps or precedes token {k - 1}")
        covered[s:e] = b"\x01" * (e - s)
        previous_end = e
        spans.append((s, e))
    for pos, ch in enumerate(text):
        if not covered[pos] and not _is_ignorable(ch):
            raise ChunkingError(
                f"character U+{ord(ch):04X} at offset {pos} is not covered by any token; "
                "refusing to drop content"
            )
    return spans


def _is_ignorable(ch: str) -> bool:
    """Characters a tokenizer may legitimately leave outside every token."""
    return ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf", "Zs", "Zl", "Zp")


def _grapheme_boundaries(text: str) -> bytearray:
    """boundaries[p] == 1 iff character offset p starts an extended grapheme
    cluster (or p == len(text))."""
    boundaries = bytearray(len(text) + 1)
    for match in _GRAPHEME_CLUSTER.finditer(text):
        boundaries[match.start()] = 1
    boundaries[len(text)] = 1
    return boundaries


# ---------- cut points ----------

def _whitespace_around_cut(text: str, spans: list[tuple[int, int]], k: int) -> tuple[int, int]:
    """Maximal whitespace run [left, right) around the cut before token k.

    It may extend into the neighbouring tokens, so whitespace inside a token
    (" world", or a whitespace-only token) counts the same as whitespace
    between tokens.
    """
    left, right = spans[k - 1][1], spans[k][0]
    while left > 0 and text[left - 1].isspace():
        left -= 1
    while right < len(text) and text[right].isspace():
        right += 1
    return left, right


def _boundary_rank(text: str, spans: list[tuple[int, int]], k: int) -> int:
    """Rank of the cut before token k, for 0 < k <= len(spans)."""
    if k == len(spans):
        return RANK_END
    left, right = _whitespace_around_cut(text, spans, k)
    run = text[left:right]
    if run.count("\n") >= 2:
        return RANK_PARAGRAPH
    if "\n" in run:
        return RANK_LINE
    if any(ch.isspace() for ch in run):
        return RANK_SENTENCE if _ends_sentence(text, left) else RANK_SPACE
    return RANK_HARD  # touching tokens, or only format characters (e.g. ZWJ) between them


def _ends_sentence(text: str, pos: int) -> bool:
    """True if the non-whitespace text ending at ``pos`` ends a sentence."""
    pos -= 1
    while pos >= 0 and text[pos] in CLOSING_PUNCTUATION:
        pos -= 1
    return pos >= 0 and text[pos] in SENTENCE_TERMINATORS


def _is_safe_cut(
    text: str,
    spans: list[tuple[int, int]],
    boundaries: bytearray,
    k: int,
) -> bool:
    """True if cutting before token k splits no grapheme cluster.

    The left chunk would end at ``left`` and the right chunk would start at
    ``right`` (the edges of the whitespace run around the cut, see
    _whitespace_around_cut). Both must be extended grapheme cluster boundaries.
    """
    if k <= 0 or k >= len(spans):
        return True  # start or end of the text (edges re-checked in _check_result)
    left, right = _whitespace_around_cut(text, spans, k)
    return bool(boundaries[left] and boundaries[right])


def _pick(
    text: str,
    spans: list[tuple[int, int]],
    boundaries: bytearray,
    candidates: range,
    ideal: int,
    max_rank: int,
) -> int | None:
    """Best grapheme-safe candidate: highest rank (capped at max_rank), then
    closest to ideal, then the later position."""
    best: tuple[int, int, int] | None = None
    best_k: int | None = None
    for k in candidates:
        if not _is_safe_cut(text, spans, boundaries, k):
            continue
        rank = min(_boundary_rank(text, spans, k), max_rank)
        key = (rank, -abs(k - ideal), k)
        if best is None or key > best:
            best, best_k = key, k
    return best_k


def _choose_end(
    text: str,
    spans: list[tuple[int, int]],
    boundaries: bytearray,
    i: int,
    previous_end: int,
    config: ChunkerConfig,
) -> int:
    """Exclusive end token index for the chunk starting at token i.

    1. If the rest fits in max_tokens, take it all.
    2. Preferred search over sizes [target // 2, max]: best rank, then closest
       to target.
    3. Only if no grapheme-safe cut exists there: widened search over sizes
       [1, target // 2). This is the only way a non-final chunk can be
       shorter than target // 2 tokens.
    4. If no grapheme-safe cut exists within max_tokens at all: ChunkingError.
    """
    n = len(spans)
    if n - i <= config.max_tokens:
        return n  # the rest fits: never leave a small tail
    hard_end = i + config.max_tokens
    lowest = max(i + 1, previous_end + 1)  # every chunk must cover a new token
    ideal = i + config.target_tokens
    preferred_low = max(lowest, i + config.min_preferred_tokens)
    k = _pick(text, spans, boundaries, range(preferred_low, hard_end + 1), ideal, RANK_END)
    if k is None:
        k = _pick(text, spans, boundaries, range(lowest, preferred_low), ideal, RANK_END)
    if k is None:
        raise ChunkingError(
            f"no grapheme-safe chunk boundary within {config.max_tokens} tokens "
            f"after character offset {spans[i][0]}"
        )
    return k


def _choose_next_start(
    text: str,
    spans: list[tuple[int, int]],
    boundaries: bytearray,
    i: int,
    j: int,
    config: ChunkerConfig,
) -> int:
    """First token of the next chunk: inside the last overlap_tokens of
    [i, j), always > i. Falls back to j (no overlap), which is safe because j
    was itself chosen as a safe cut."""
    if config.overlap_tokens == 0:
        return j
    lo = max(i + 1, j - config.overlap_tokens)
    ideal = j - config.overlap_tokens
    k = _pick(text, spans, boundaries, range(lo, j), ideal, RANK_SENTENCE)
    return j if k is None else k


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# ---------- post-conditions ----------

def _check_result(
    text: str,
    spans: list[tuple[int, int]],
    boundaries: bytearray,
    chunks: list[ChunkSpan],
    config: ChunkerConfig,
) -> None:
    covered = bytearray(len(text))
    previous: ChunkSpan | None = None
    for c in chunks:
        if c.text != text[c.start:c.end] or not c.text or c.text != c.text.strip():
            raise ChunkingError(f"chunk {c.index} is not an exact trimmed slice")
        if c.span_sha256 != sha256_text(c.text):
            raise ChunkingError(f"chunk {c.index} has a wrong span_sha256")
        if not 0 <= c.token_start < c.token_end <= len(spans):
            raise ChunkingError(f"chunk {c.index} has an invalid token range")
        if c.token_count != c.token_end - c.token_start or c.token_count > config.max_tokens:
            raise ChunkingError(f"chunk {c.index} has an invalid token_count")
        if (c.start, c.end) != _trim(text, spans[c.token_start][0], spans[c.token_end - 1][1]):
            raise ChunkingError(f"chunk {c.index} does not match its token range")
        if not (boundaries[c.start] and boundaries[c.end]):
            raise ChunkingError(
                f"chunk {c.index} edge at offset {c.start if not boundaries[c.start] else c.end} "
                "would split a grapheme cluster"
            )
        if previous is not None and not (
            c.start > previous.start
            and c.end > previous.end
            and c.token_start > previous.token_start
            and c.token_end > previous.token_end
        ):
            raise ChunkingError(f"chunk {c.index} does not advance past chunk {previous.index}")
        covered[c.start:c.end] = b"\x01" * (c.end - c.start)
        previous = c
    for pos, ch in enumerate(text):
        if not covered[pos] and not _is_ignorable(ch):
            raise ChunkingError(f"character at offset {pos} is not in any chunk")