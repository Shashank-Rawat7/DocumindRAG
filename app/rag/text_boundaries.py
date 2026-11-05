"""Shared Unicode boundary rules for chunking and tokenizer-offset mapping.

Pure functions, no I/O. ``chunking`` and ``token_offsets`` both use these so
that "ignorable character" and "grapheme cluster boundary" mean exactly the
same thing in both modules.
"""
from __future__ import annotations

import unicodedata

import regex

# Unicode extended grapheme clusters (UAX #29), as implemented by the pinned regex package.
_GRAPHEME_CLUSTER = regex.compile(r"\X")
GRAPHEME_SEGMENTATION = f"regex-{regex.__version__}-X-uax29-extended-grapheme-clusters"

_IGNORABLE_CATEGORIES = frozenset(("Cc", "Cf", "Zs", "Zl", "Zp"))
_EXTENDER_CATEGORIES = frozenset(("Mn", "Mc", "Me"))
_EMOJI_MODIFIERS = range(0x1F3FB, 0x1F3FF + 1)  # Fitzpatrick skin-tone modifiers


def is_ignorable(ch: str) -> bool:
    """Characters a tokenizer may legitimately leave outside every token:
    whitespace, or Unicode category Cc, Cf, Zs, Zl or Zp."""
    return ch.isspace() or unicodedata.category(ch) in _IGNORABLE_CATEGORIES


def grapheme_boundaries(text: str) -> bytearray:
    """``b[p] == 1`` iff character offset ``p`` starts an extended grapheme
    cluster, or ``p == len(text)``. ``len(b) == len(text) + 1``."""
    boundaries = bytearray(len(text) + 1)
    for match in _GRAPHEME_CLUSTER.finditer(text):
        boundaries[match.start()] = 1
    boundaries[len(text)] = 1
    return boundaries


def is_cluster_extender(ch: str) -> bool:
    """Characters that only ever extend the preceding grapheme cluster:
    combining marks (Mn, Mc, Me; this includes variation selectors U+FE00-FE0F,
    U+E0100-E01EF and the enclosing keycap U+20E3) and emoji skin-tone
    modifiers U+1F3FB-U+1F3FF."""
    return unicodedata.category(ch) in _EXTENDER_CATEGORIES or ord(ch) in _EMOJI_MODIFIERS