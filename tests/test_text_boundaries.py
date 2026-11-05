"""Shared Unicode rules (app/rag/text_boundaries.py). Pure, no database access."""
import pytest
import regex

from app.rag import chunking, text_boundaries
from app.rag.text_boundaries import (
    GRAPHEME_SEGMENTATION,
    grapheme_boundaries,
    is_cluster_extender,
    is_ignorable,
)


def starts(text):
    b = grapheme_boundaries(text)
    return [p for p in range(len(b)) if b[p]]


def test_grapheme_segmentation_identity_is_unchanged():
    assert GRAPHEME_SEGMENTATION == f"regex-{regex.__version__}-X-uax29-extended-grapheme-clusters"
    assert chunking.GRAPHEME_SEGMENTATION == GRAPHEME_SEGMENTATION


def test_chunking_uses_the_shared_rules():
    assert chunking.grapheme_boundaries is text_boundaries.grapheme_boundaries
    assert chunking.is_ignorable is text_boundaries.is_ignorable


@pytest.mark.parametrize("ch", [
    " ", "\t", "\n", "\r", "\x00", "\x1f",       # whitespace and Cc
    "\xa0", "\u3000",                           # Zs
    "\u2028", "\u2029",                           # Zl, Zp
    "\u200b", "\u200c", "\u200d", "\ufeff", "\xad",  # Cf
])
def test_ignorable_characters(ch):
    assert is_ignorable(ch)


@pytest.mark.parametrize("ch", [
    "a", "1", ".", "\u0915", "\u0301", "\ufe0f", "\u20e3", "\U0001F600", "\U0001F3FD", "\u2640",
])
def test_non_ignorable_characters(ch):
    assert not is_ignorable(ch)


def test_empty_text_has_a_single_boundary():
    assert grapheme_boundaries("") == bytearray(b"\x01")


@pytest.mark.parametrize("text, expected", [
    ("abc", [0, 1, 2, 3]),
    ("e\u0301x", [0, 2, 3]),                               # base + combining mark
    ("e\u0301\u0308", [0, 3]),                             # two combining marks
    ("\r\n", [0, 2]),                                      # CR LF
    ("\u1100\u1161\u11a8", [0, 3]),                        # Hangul L V T
    ("\U0001F1EE\U0001F1F3\U0001F1FA", [0, 2, 3]),         # regional indicators pair from the start
    ("\U0001F468\u200d\U0001F469\u200d\U0001F467", [0, 5]),  # emoji ZWJ sequence
    ("\U0001F44D\U0001F3FD", [0, 2]),                      # skin-tone modifier
    ("1\ufe0f\u20e3", [0, 3]),                             # keycap
    ("\u0915\u094d\u0937", [0, 3]),                        # Devanagari conjunct (GB9c)
    ("a \u0301", [0, 1, 3]),                               # a mark attaches to the space before it
])
def test_grapheme_boundaries(text, expected):
    b = grapheme_boundaries(text)
    assert len(b) == len(text) + 1 and b[len(text)] == 1
    assert starts(text) == expected


@pytest.mark.parametrize("ch", [
    "\u0301", "\u0308", "\u094d",        # Mn
    "\u0903", "\u093e",                  # Mc
    "\u20e3",                            # Me (enclosing keycap)
    "\ufe0e", "\ufe0f", "\U000E0100",    # variation selectors (Mn)
    "\U0001F3FB", "\U0001F3FF",          # skin-tone modifiers
])
def test_cluster_extenders(ch):
    assert is_cluster_extender(ch)


@pytest.mark.parametrize("ch", [
    "a", " ", "\u200d", "\u200c", "\u2640", "\U0001F600", "\U0001F3FA", "\U0001F400", "\U0001F1EE",
])
def test_not_cluster_extenders(ch):
    assert not is_cluster_extender(ch)