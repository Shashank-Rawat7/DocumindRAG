"""om-v1 tokenizer-offset mapping (app/rag/token_offsets.py).

Pure rule tests on hand-written offsets, plus real-tokenizer tests that run
only when the pinned model is already in the local Hugging Face cache (they
are skipped otherwise; nothing is ever downloaded).
"""
import dataclasses
import os
import random
import unicodedata
from pathlib import Path

import pytest

from app.rag import token_offsets
from app.rag.chunking import DEFAULT_CHUNKER_CONFIG, ChunkerConfig, chunk_text
from app.rag.text_boundaries import GRAPHEME_SEGMENTATION
from app.rag.token_offsets import (
    METASPACE,
    OFFSET_MAPPING_VERSION,
    OffsetMappingError,
    SourceUnit,
    map_token_offsets,
    signature_fields,
)

M = METASPACE
SAMPLE_PDF = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def units_of(mapping):
    return [(u.start, u.end, u.token_count, u.first_token) for u in mapping.units]


def reject(text, offsets, tokens):
    with pytest.raises(OffsetMappingError) as excinfo:
        map_token_offsets(text, offsets, tokens)
    return excinfo.value


# ---------- identity ----------

def test_signature_fields():
    assert OFFSET_MAPPING_VERSION == "om-v2"
    assert signature_fields() == {
        "offset_mapping_version": "om-v2",
        "overlap_rule": "metaspace_prefix_same_start+ligature_fb01_fb02_continuation",
        "zero_length_offsets": "reject",
        "grapheme_expansion": "same_cluster_extenders_only",
        "grapheme_segmentation": GRAPHEME_SEGMENTATION,
    }


def test_results_are_immutable():
    mapping = map_token_offsets("ab", [(0, 2)], ["ab"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        mapping.units[0].end = 1
    assert isinstance(mapping.units, tuple) and type(mapping.units[0]) is SourceUnit


# ---------- acceptance ----------

@pytest.mark.parametrize("text, offsets, tokens, expected, groups", [
    ("", [], [], [], 0),
    ("abc", [(0, 3)], ["abc"], [(0, 3, 1, 0)], 0),
    # verified metaspace pattern in the middle of the text
    ("x 422", [(0, 1), (2, 3), (2, 5)], ["x", M, "422"], [(0, 1, 1, 0), (2, 5, 2, 1)], 1),
    # at the start of the text
    ("422 x", [(0, 1), (0, 3), (4, 5)], [M, "422", "x"], [(0, 3, 2, 0), (4, 5, 1, 2)], 1),
    # followed by a one-character Devanagari token
    ("\u0928\u092e", [(0, 1), (0, 1), (1, 2)], [M, "\u0928", "\u092e"], [(0, 1, 2, 0), (1, 2, 1, 2)], 1),
    # metaspace on a real space: no overlap, an ordinary unit
    ("a b", [(0, 1), (1, 2), (2, 3)], ["a", M, "b"], [(0, 1, 1, 0), (1, 2, 1, 1), (2, 3, 1, 2)], 0),
    # whitespace-only token span
    ("a  b", [(0, 1), (1, 3), (3, 4)], ["a", "  ", "b"], [(0, 1, 1, 0), (1, 3, 1, 1), (3, 4, 1, 2)], 0),
    # tokens list may be a tuple and offsets may be lists
    ("ab", ([0, 1], [1, 2]), ("a", "b"), [(0, 1, 1, 0), (1, 2, 1, 1)], 0),
        # a ligature split between two tokens: the continuation extends the unit (token_count 2)
    ("de\ufb01ne", [(0, 3), (2, 5)], [M + "def", "ine"], [(0, 5, 2, 0)], 0),
    ("de\ufb02ne", [(0, 3), (2, 5)], [M + "def", "lne"], [(0, 5, 2, 0)], 0),
    ("abc\ufb01de,", [(0, 4), (3, 6), (6, 7)], [M + "abcf", "ide", ","], [(0, 6, 2, 0), (6, 7, 1, 2)], 0),
    ("abc\ufb02de,", [(0, 4), (3, 6), (6, 7)], [M + "abcf", "lde", ","], [(0, 6, 2, 0), (6, 7, 1, 2)], 0),
    ("ab de\ufb01ne", [(0, 2), (3, 6), (5, 8)], [M + "ab", M + "def", "ine"], [(0, 2, 1, 0), (3, 8, 2, 1)], 0),
    ("de\ufb01ne x", [(0, 3), (2, 5), (6, 7)], [M + "def", "ine", "x"], [(0, 5, 2, 0), (6, 7, 1, 2)], 0),
    # two isolated ligature splits in one text (not adjacent)
    ("de\ufb01ne ab\ufb02cd", [(0, 3), (2, 5), (6, 9), (8, 11)], [M + "def", "ine", M + "abf", "lcd"],
     [(0, 5, 2, 0), (6, 11, 2, 2)], 0),
    # after an ordinary Metaspace pair
    ("de\ufb01ne", [(0, 1), (0, 3), (2, 5)], [M, "def", "ine"], [(0, 5, 3, 0)], 1),
], ids=["empty", "single", "metaspace-middle", "metaspace-start", "metaspace-devanagari",
        "metaspace-on-space", "whitespace-token", "list-and-tuple-inputs",
        "ligature-fb01", "ligature-fb02", "ligature-fb01-observed-geometry",
        "ligature-fb02-observed-geometry", "ligature-mid-text", "ligature-then-unit",
        "ligature-two-isolated", "ligature-after-metaspace-pair"])

def test_accepted_offsets(text, offsets, tokens, expected, groups):
    mapping = map_token_offsets(text, offsets, tokens)
    assert units_of(mapping) == expected
    assert mapping.total_tokens == len(offsets)
    assert mapping.metaspace_groups == groups
    assert mapping.expanded_characters == 0


# ---------- rejection ----------

@pytest.mark.parametrize("text, offsets, tokens, reason, position", [
    ("abc", [(0, 0), (0, 3)], ["", "abc"], "zero_length_offset", 0),
    ("abc", [(0, 2), (2, 2), (2, 3)], ["ab", "", "c"], "zero_length_offset", 2),
    ("abc", [(0, 3), (3, 3)], ["abc", ""], "zero_length_offset", 3),
    ("abc", [(0, 3), (0, 0)], ["abc", "</s>"], "zero_length_offset", 0),    # special token leaked in
    ("abcdef", [(5, 6), (1, 2)], ["f", "b"], "non_monotonic", 1),
    ("abcdef", [(0, 2), (2, 4), (1, 6)], ["ab", "cd", "x"], "non_monotonic", 1),
    ("abcde", [(0, 5), (1, 3)], ["abcde", "bc"], "unverified_overlap", 1),    # nested
    ("abc", [(0, 2), (1, 3)], ["ab", "bc"], "unverified_overlap", 1),        # partial
    ("ab", [(0, 2), (0, 2)], ["ab", "ab"], "unverified_overlap", 0),         # identical
    ("ab", [(0, 1), (0, 1), (0, 2)], [M, M, "ab"], "unverified_overlap", 0),  # two metaspaces
    ("ab", [(0, 1), (0, 1), (0, 1)], [M, "a", "b"], "unverified_overlap", 0),  # identical after a group
    ("de\ufb01ne", [(0, 3), (2, 3)], [M + "def", "i"], "unverified_overlap", 2),            # nested, plausible pieces
    ("de\ufb01ne", [(0, 3), (2, 3), (3, 5)], [M + "def", "i", "ne"], "unverified_overlap", 2),  # nested, then a normal token
    ("de\ufb01ne", [(0, 3), (2, 3)], [M + "def", "xyz"], "unverified_overlap", 2),          # nested, unrelated pieces
    ("\ufb01", [(0, 1), (0, 1)], [M + "f", "i"], "unverified_overlap", 0),                  # identical, pieces spell fi
    ("de\ufb01", [(0, 2), (2, 3), (2, 3)], [M + "de", "f", "i"], "unverified_overlap", 2),  # identical mid-text
    ("de\ufb01ne", [(0, 3), (2, 5)], [M + "dex", "ine"], "unverified_overlap", 2),          # unrelated first piece
    ("de\ufb01ne", [(0, 3), (2, 5)], [M + "def", "xne"], "unverified_overlap", 2),          # unrelated second piece
    ("de\ufb01ne", [(0, 3), (2, 5)], [M + "def", M + "ine"], "unverified_overlap", 2),      # marker on the continuation
    ("de\ufb01ne", [(0, 4), (2, 5)], [M + "def", "ine"], "unverified_overlap", 2),          # two-character overlap
    ("de\ufb01ne", [(0, 2), (2, 3), (2, 5)], [M + "de", "f", "ine"], "unverified_overlap", 2),  # previous token starts at s
    ("de\ufb01ne", [(0, 4), (3, 5)], [M + "defi", "ne"], "unverified_overlap", 3),          # overlap on a non-ligature
    ("de\ufb01ne", [(0, 3), (2, 3), (2, 5)], [M + "def", "i", "ne"], "unverified_overlap", 2),  # three tokens on one character
    ("de\ufb01ne", [(0, 3), (2, 5), (3, 5)], [M + "def", "ine", "ne"], "unverified_overlap", 3),  # extra overlap after a merge
    ("x\ufb01\ufb01ne", [(0, 2), (1, 3), (2, 5)], [M + "xf", "if", "ine"], "unverified_overlap", 2),  # chained: second ligature after a merge
    ("define", [(0, 3), (2, 5)], [M + "def", "ine"], "unverified_overlap", 2),              # ASCII text, same shape
    ("a\u2122b", [(0, 2), (1, 3)], [M + "aT", "Mb"], "unverified_overlap", 1),              # TM sign
    ("de\ufb00ne", [(0, 3), (2, 5)], [M + "def", "fne"], "unverified_overlap", 2),          # unsupported U+FB00
    ("de\ufb03ne", [(0, 3), (2, 5)], [M + "def", "fine"], "unverified_overlap", 2),         # unsupported U+FB03
    ("de\ufb04ne", [(0, 3), (2, 5)], [M + "def", "flne"], "unverified_overlap", 2),        # unsupported U+FB04
    ("de\ufb05ne", [(0, 3), (2, 5)], [M + "des", "tne"], "unverified_overlap", 2),         # unsupported U+FB05
    ("de\ufb06ne", [(0, 3), (2, 5)], [M + "des", "tne"], "unverified_overlap", 2),         # unsupported U+FB06
    ("\u2026", [(0, 1), (0, 1), (0, 1)], [M, "..", "."], "unverified_overlap", 0),  # no compat rule
    (" a", [(0, 1), (0, 2)], [M, "a"], "unverified_overlap", 0),             # metaspace on whitespace
    ("abc", [(0, 1), (1, 2), (0, 3)], ["a", "b", M], "non_monotonic", 0),
    ("abc", [(0, 4)], ["abcd"], "out_of_range", 0),
    ("abc", [(-1, 2)], ["ab"], "out_of_range", -1),
    ("abc", [(2, 1)], ["x"], "out_of_range", 2),
    ("abc", [(0, True)], ["a"], "non_int_offset", 0),
    ("abc", [(0.0, 1)], ["a"], "non_int_offset", 0),
    ("abc", [(0, 1, 2)], ["a"], "malformed_offset", 0),
    ("abc", [5], ["a"], "malformed_offset", 0),
    ("abc", [(0, 3)], [None], "non_str_token", 0),
    ("abc", [(0, 3)], ["a", "b"], "length_mismatch", None),
], ids=["zero-length-first", "zero-length-middle", "zero-length-last", "special-token",
        "non-monotonic", "non-monotonic-later", "nested", "partial", "identical",
        "double-metaspace", "identical-after-group",
        "ligature-nested", "ligature-nested-then-token", "ligature-nested-unrelated",
        "ligature-identical-start", "ligature-identical-mid", "ligature-bad-first-piece",
        "ligature-bad-second-piece", "ligature-continuation-marker", "ligature-two-char-overlap",
        "ligature-previous-starts-at-s", "ligature-non-ligature-overlap", "ligature-three-tokens",
        "ligature-extra-overlap", "ligature-chained-rejected", "ascii-same-shape", "trade-mark-overlap",
        "unsupported-fb00", "unsupported-fb03", "unsupported-fb04", "unsupported-fb05", "unsupported-fb06",
        "ellipsis-expansion", "metaspace-on-space-overlap",
        "trailing-metaspace-backwards", "past-end", "negative", "reversed", "bool", "float",
        "triple", "not-a-pair", "non-str-token", "length-mismatch"])
def test_rejected_offsets(text, offsets, tokens, reason, position):
    error = reject(text, offsets, tokens)
    assert error.reason == reason
    assert error.position == position


@pytest.mark.parametrize("offsets, tokens, reason", [
    ("ab", ["a", "b"], "offsets_not_sequence"),
    ([(0, 2)], "ab", "tokens_not_sequence"),
    (iter([(0, 2)]), ["ab"], "offsets_not_sequence"),
])
def test_inputs_must_be_sequences(offsets, tokens, reason):
    assert reject("ab", offsets, tokens).reason == reason


def test_text_must_be_a_str():
    assert reject(b"ab", [(0, 2)], ["ab"]).reason == "text_not_str"

def test_supported_ligatures_match_their_nfkc_expansion():
    assert set(token_offsets._LIGATURE_EXPANSION) == {"\ufb01", "\ufb02"}
    for character, expansion in token_offsets._LIGATURE_EXPANSION.items():
        assert unicodedata.normalize("NFKC", character) == expansion
        assert len(expansion) == 2


def test_ligature_continuation_needs_the_previous_token_to_end_the_unit():
    # Not reachable through map_token_offsets (_group keeps previous_end == pairs[i - 1][1]);
    # pinned here so the helper cannot silently drop that condition.
    text, tokens = "de\ufb01ne", [M + "def", "ine"]
    assert token_offsets._is_ligature_continuation(text, [(0, 3), (2, 5)], tokens, 1, 3) is True
    assert token_offsets._is_ligature_continuation(text, [(0, 4), (2, 5)], tokens, 1, 3) is False

def test_error_messages_never_contain_document_text():
    error = reject("SECRET-WORD", [(0, 6)], ["SECRET"])
    assert error.reason == "uncovered_character"
    assert "SECRET" not in str(error)


# ---------- grapheme expansion ----------

@pytest.mark.parametrize("text, offsets, tokens, expected, expanded", [
    ("cafe\u0301", [(0, 3), (3, 4)], ["caf", "e"], [(0, 3, 1, 0), (3, 5, 1, 1)], 1),
    ("e\u0301", [(0, 1), (0, 1)], [M, "e"], [(0, 2, 2, 0)], 1),
    ("e\u0301\u0308", [(0, 1)], ["e"], [(0, 3, 1, 0)], 2),                    # two marks
    ("\u2764\ufe0f", [(0, 1)], ["h"], [(0, 2, 1, 0)], 1),                    # VS16
    ("1\ufe0f\u20e3", [(0, 1)], ["1"], [(0, 3, 1, 0)], 2),                   # keycap
    ("\U0001F44D\U0001F3FD", [(0, 1)], ["t"], [(0, 2, 1, 0)], 1),            # skin tone
    ("e\u0301\u0308", [(0, 1), (2, 3)], ["e", "x"], [(0, 2, 1, 0), (2, 3, 1, 1)], 1),  # stops at a unit
    ("\U0001F468\u200d\U0001F469", [(0, 1), (2, 3)], ["a", "b"], [(0, 1, 1, 0), (2, 3, 1, 1)], 0),  # ZWJ stays
    ("a\u200bb", [(0, 1), (2, 3)], ["a", "b"], [(0, 1, 1, 0), (2, 3, 1, 1)], 0),  # ZWSP stays
    ("e\u0301 x", [(0, 1), (3, 4)], ["e", "x"], [(0, 2, 1, 0), (3, 4, 1, 1)], 1),
    ("e\u0301\u200dx", [(0, 1), (3, 4)], ["e", "x"], [(0, 2, 1, 0), (3, 4, 1, 1)], 1), # whitespace not crossed
], ids=["decomposed-accent", "accent-after-metaspace-group", "two-marks", "vs16", "keycap",
        "skin-tone", "stops-at-next-unit", "zwj-left-uncovered", "zwsp-left-uncovered",
        "stops-before-space","trailing-zwj-not-absorbed"])
def test_grapheme_expansion(text, offsets, tokens, expected, expanded):
    mapping = map_token_offsets(text, offsets, tokens)
    assert units_of(mapping) == expected
    assert mapping.expanded_characters == expanded
    for u in mapping.units:
        assert text[u.start:u.end]                   # exact, non-empty slices of the original


@pytest.mark.parametrize("text, offsets, tokens, position", [
    ("a \u0301b", [(0, 1), (3, 4)], ["a", "b"], 2),            # mark attached to a space
    ("x\U0001F600", [(0, 1)], ["x"], 1),                       # dropped character starts its own cluster
    ("ab", [(0, 1)], ["a"], 1),                                # plain dropped letter
    ("\U0001F469\u200d\u2640\ufe0f", [(0, 1)], ["w"], 2),       # visible symbol dropped from a ZWJ sequence
    ("\u0301x", [(1, 2)], ["x"], 0),                           # leading mark with no unit before it
    ("e\u0301\u0301x", [(0, 1), (3, 4)], ["e", "x"], None),
    ("e\u200d\u0301", [(0, 1)], ["e"], 2),
    ("e\u0301\u200d\u0308", [(0, 1)], ["e"], 3),   # accepted: both marks absorbed (control)
], ids=["mark-on-space", "dropped-emoji", "dropped-letter", "dropped-symbol-in-zwj",
        "leading-mark", "control-two-marks","mark-after-zwj","mark-after-mark-and-zwj",])
def test_grapheme_expansion_refusals(text, offsets, tokens, position):
    if position is None:
        assert units_of(map_token_offsets(text, offsets, tokens))[0] == (0, 3, 1, 0)
        return
    error = reject(text, offsets, tokens)
    assert error.reason == "uncovered_character"
    assert error.position == position


def test_source_text_is_never_modified():
    text = "cafe\u0301 \u2764\ufe0f"
    original = str(text)
    mapping = map_token_offsets(text, [(0, 3), (3, 4), (6, 7)], ["caf", "e", "h"])
    assert text == original
    assert "".join(text[u.start:u.end] for u in mapping.units) == "cafe\u0301\u2764\ufe0f"


# ---------- conservation ----------

@pytest.mark.parametrize("seed", range(25))
def test_random_offsets_conserve_every_token(seed):
    rng = random.Random(seed)
    words = [rng.choice(["ab", "c", "def", "\u0928\u092e", "x1"]) for _ in range(rng.randint(1, 60))]
    text = " ".join(words)
    offsets, tokens, pos = [], [], 0
    for word in words:
        if rng.random() < 0.4:                       # bare metaspace + word sharing its first char
            offsets.append((pos, pos + 1))
            tokens.append(M)
        offsets.append((pos, pos + len(word)))
        tokens.append(word)
        pos += len(word) + 1
    mapping = map_token_offsets(text, offsets, tokens)
    assert mapping.total_tokens == len(offsets)
    assert sum(u.token_count for u in mapping.units) == len(offsets)
    assert len(mapping.units) == len(offsets) - mapping.metaspace_groups
    expected_first = 0
    for u in mapping.units:
        assert u.first_token == expected_first
        expected_first += u.token_count
    assert [text[u.start:u.end] for u in mapping.units] == words


def test_mapping_feeds_the_weighted_chunker():
    text = "x 422 cafe\u0301 \u2764\ufe0f"
    mapping = map_token_offsets(
        text, [(0, 1), (2, 3), (2, 5), (6, 10), (12, 13)], ["x", M, "422", "cafe", "h"])
    chunks = chunk_text(text, [(u.start, u.end) for u in mapping.units], DEFAULT_CHUNKER_CONFIG,
                        token_counts=[u.token_count for u in mapping.units])
    assert [c.text for c in chunks] == [text]
    assert chunks[0].token_count == mapping.total_tokens == 5


# ---------- the pinned tokenizer (skipped when it is not in the local cache) ----------

@pytest.fixture(scope="module")
def pinned_tokenizer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True, trust_remote_code=False)
    except (OSError, ValueError) as exc:
        pytest.skip(f"pinned tokenizer not in the local cache ({type(exc).__name__})")


def tokenize(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]], list(tok.convert_ids_to_tokens(enc["input_ids"]))


@pytest.mark.parametrize("text", [
    "\ufb01", "office \ufb01le", "\u2026", "\xbd", "\u2122", "\uff21\uff22\uff23",
    "e\u0301", "cafe\u0301", "\u2460",
    "PostgreSQL stores table rows in heap files.",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
], ids=["fi-ligature", "office-fi-le", "ellipsis", "one-half", "trade-mark", "fullwidth",
        "e-acute-decomposed", "cafe-decomposed", "circled-one", "english", "hindi"])
def test_pinned_tokenizer_offsets_are_accepted(pinned_tokenizer, text):
    offsets, tokens = tokenize(pinned_tokenizer, text)
    mapping = map_token_offsets(text, offsets, tokens)
    assert mapping.total_tokens == len(offsets) == sum(u.token_count for u in mapping.units)


def test_sample_pdf_offsets_map_and_chunk(pinned_tokenizer):
    from app.core.processing import extract_text

    text = extract_text(SAMPLE_PDF)
    offsets, tokens = tokenize(pinned_tokenizer, text)
    mapping = map_token_offsets(text, offsets, tokens)
    assert mapping.total_tokens == len(offsets) == sum(u.token_count for u in mapping.units)
    assert len(mapping.units) == len(offsets) - mapping.metaspace_groups
    chunks = chunk_text(text, [(u.start, u.end) for u in mapping.units], DEFAULT_CHUNKER_CONFIG,
                        token_counts=[u.token_count for u in mapping.units])
    assert chunks[0].token_start == 0 and chunks[-1].token_end == mapping.total_tokens
    for c in chunks:
        model_tokens = len(pinned_tokenizer("passage: " + c.text, add_special_tokens=True)["input_ids"])
        assert model_tokens <= 512
    small = ChunkerConfig(target_tokens=40, max_tokens=60, overlap_tokens=10)
    assert len(chunk_text(text, [(u.start, u.end) for u in mapping.units], small,
                          token_counts=[u.token_count for u in mapping.units])) > 1