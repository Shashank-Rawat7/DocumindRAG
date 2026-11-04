"""Unit tests for app/rag/chunking.py (C2.1). Pure Python: no model, no network.

Token spans come from small fake tokenizers defined below, so the chunking rules
are tested independently of the real model tokenizer (verified in C2.4).
"""
import random
import re
import unicodedata

import pytest
import regex

from app.rag.chunking import (
    CHUNKER_VERSION,
    DEFAULT_CHUNKER_CONFIG,
    GRAPHEME_SEGMENTATION,
    ChunkerConfig,
    ChunkingError,
    ChunkSpan,
    chunk_text,
    sha256_text,
)

SMALL = ChunkerConfig(target_tokens=10, max_tokens=14, overlap_tokens=3)

ZWJ = "\u200d"
ZWNJ = "\u200c"
VIRAMA = "\u094d"
ANUSVARA = "\u0902"
KA = "\u0915"
SSA = "\u0937"
L_JAMO = "\u1100"      # HANGUL CHOSEONG KIYEOK (leading consonant, L)
V_JAMO = "\u1161"      # HANGUL JUNGSEONG A (vowel, V)
T_JAMO = "\u11a8"      # HANGUL JONGSEONG KIYEOK (trailing consonant, T)
RI_I = "\U0001f1ee"    # REGIONAL INDICATOR SYMBOL LETTER I
RI_N = "\U0001f1f3"    # REGIONAL INDICATOR SYMBOL LETTER N
PREPEND = "\u0600"     # ARABIC NUMBER SIGN (Grapheme_Cluster_Break=Prepend)
NO_OVERLAP = ChunkerConfig(target_tokens=9, max_tokens=10, overlap_tokens=0)


# ---------- fake tokenizers (content tokens only, like add_special_tokens=False) ----------

def word_spans(text):
    """One token per whitespace-delimited word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def piece_spans(text, size):
    """Naive sub-word pieces: every word cut into `size`-code-point pieces,
    deliberately ignoring graphemes so unsafe cut points exist."""
    spans = []
    for m in re.finditer(r"\S+", text):
        for s in range(m.start(), m.end(), size):
            spans.append((s, min(s + size, m.end())))
    return spans


def codepoint_spans(text):
    """One token per non-whitespace code point (worst case for grapheme safety)."""
    return [(i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]


def leading_space_spans(text):
    """SentencePiece-like: each word token also covers the whitespace before it."""
    spans, previous_end = [], 0
    for m in re.finditer(r"\S+", text):
        spans.append((previous_end, m.end()))
        previous_end = m.end()
    return spans


def char_spans(text):
    """One token per character, whitespace included (\\r and \\n become separate tokens)."""
    return [(i, i + 1) for i in range(len(text))]


def whitespace_token_spans(text):
    """Separate tokens for words and for whitespace runs (like a bare "\u2581" token)."""
    return [m.span() for m in re.finditer(r"\S+|\s+", text)]


# ---------- invariant checks shared by many tests ----------

def grapheme_starts(text):
    """Cross-check oracle: offsets that start an extended grapheme cluster (regex \\X).

    The UAX #29 behaviour of regex itself is pinned by hand-derived expected clusters
    in test_regex_segmentation_matches_hand_derived_uax29_clusters."""
    starts = {m.start() for m in regex.finditer(r"\X", text)}
    starts.add(len(text))
    return starts


def assert_invariants(text, spans, chunks, config):
    starts = grapheme_starts(text)
    previous = None
    covered = [False] * len(text)
    for expected_index, c in enumerate(chunks):
        assert isinstance(c, ChunkSpan)
        assert c.index == expected_index
        assert 0 <= c.start < c.end <= len(text)
        assert c.text == text[c.start:c.end]
        assert c.text == c.text.strip() and c.text
        assert c.span_sha256 == sha256_text(c.text)
        assert len(c.span_sha256) == 64 and c.span_sha256 == c.span_sha256.lower()
        # Token assignment: contiguous range [token_start, token_end) before edge trimming.
        assert 0 <= c.token_start < c.token_end <= len(spans)
        assert c.token_count == c.token_end - c.token_start
        assert 1 <= c.token_count <= config.max_tokens
        range_start, range_end = spans[c.token_start][0], spans[c.token_end - 1][1]
        assert range_start <= c.start and c.end <= range_end
        assert text[range_start:c.start].strip() == ""   # only whitespace was trimmed
        assert text[c.end:range_end].strip() == ""
        if c.token_start > 0:
            assert spans[c.token_start - 1][1] <= c.start   # earlier tokens are outside
        if c.token_end < len(spans):
            assert spans[c.token_end][0] >= c.end         # later tokens are outside
        assert c.start in starts and c.end in starts, (c.index, c.start, c.end)
        if previous is not None:
            assert c.start > previous.start and c.end > previous.end
            assert c.token_start > previous.token_start and c.token_end > previous.token_end
        for pos in range(c.start, c.end):
            covered[pos] = True
        previous = c
    for pos, ch in enumerate(text):
        if not ch.isspace():
            assert covered[pos], f"offset {pos} ({ch!r}) not in any chunk"


def words(n, prefix="w"):
    return " ".join(f"{prefix}{k}" for k in range(n))


def outcome(text, spans, config):
    """Chunks, or the ChunkingError message: used for determinism checks."""
    try:
        return chunk_text(text, spans, config)
    except ChunkingError as exc:
        return str(exc)


# ---------- configuration ----------

def test_default_config_is_the_approved_provisional_budget():
    assert (DEFAULT_CHUNKER_CONFIG.target_tokens,
            DEFAULT_CHUNKER_CONFIG.max_tokens,
            DEFAULT_CHUNKER_CONFIG.overlap_tokens) == (300, 400, 50)


def test_signature_fields_identify_the_chunking_version():
    assert DEFAULT_CHUNKER_CONFIG.signature_fields() == {
        "chunker_version": CHUNKER_VERSION,
        "target_tokens": 300,
        "max_tokens": 400,
        "overlap_tokens": 50,
        "token_count_scope": "content_tokens_only",
        "token_assignment": "contiguous_token_range_before_edge_trim",
        "grapheme_segmentation": GRAPHEME_SEGMENTATION,
    }
    assert GRAPHEME_SEGMENTATION == f"regex-{regex.__version__}-X-uax29-extended-grapheme-clusters"


@pytest.mark.parametrize("kwargs", [
    dict(target_tokens=0, max_tokens=10, overlap_tokens=0),
    dict(target_tokens=10, max_tokens=9, overlap_tokens=1),
    dict(target_tokens=10, max_tokens=10, overlap_tokens=10),
    dict(target_tokens=10, max_tokens=10, overlap_tokens=-1),
    dict(target_tokens=10.0, max_tokens=10, overlap_tokens=1),
    dict(target_tokens=10, max_tokens=True, overlap_tokens=1),
], ids=["target-zero", "max-below-target", "overlap-not-below-target", "overlap-negative",
        "float-budget", "bool-budget"])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)


# ---------- empty, short, whitespace ----------

def test_empty_text_gives_no_chunks():
    assert chunk_text("", []) == []


def test_whitespace_only_text_gives_no_chunks():
    text = "  \n\n\t  \n"
    assert chunk_text(text, word_spans(text)) == []


def test_whitespace_only_tokens_give_no_chunks():
    text = "  \n\n\t  \n"
    assert chunk_text(text, whitespace_token_spans(text)) == []


def test_short_text_is_a_single_chunk():
    text = "Hello world."
    [c] = chunk_text(text, word_spans(text), SMALL)
    assert (c.index, c.start, c.end, c.text) == (0, 0, 12, "Hello world.")
    assert (c.token_start, c.token_end, c.token_count) == (0, 2, 2)


def test_chunk_edges_are_trimmed_without_touching_the_source():
    text = "\n\n  Hello world.  \n"
    [c] = chunk_text(text, word_spans(text), SMALL)
    assert (c.start, c.end, c.text) == (4, 16, "Hello world.")


# ---------- budgets, overlap, progress ----------

def test_long_english_text_respects_the_default_budget():
    text = "\n\n".join(
        " ".join(f"Sentence {p}-{s} has a few words in it." for s in range(6))
        for p in range(120)
    )
    spans = word_spans(text)
    chunks = chunk_text(text, spans)
    assert len(chunks) > 5
    assert_invariants(text, spans, chunks, DEFAULT_CHUNKER_CONFIG)
    assert all(c.token_count >= DEFAULT_CHUNKER_CONFIG.min_preferred_tokens for c in chunks[:-1])


def test_consecutive_chunks_overlap_by_at_most_overlap_tokens():
    text = words(200)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert 1 <= prev.token_end - nxt.token_start <= SMALL.overlap_tokens


def test_zero_overlap_produces_adjacent_chunks():
    config = ChunkerConfig(target_tokens=10, max_tokens=14, overlap_tokens=0)
    text = words(100)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, config)
    assert_invariants(text, spans, chunks, config)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.token_start == prev.token_end


def test_small_remainder_is_merged_into_the_last_chunk():
    text = words(16)  # 16 tokens > max 14, so two chunks; no 1-token tail
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[-1].end == len(text)
    assert all(c.token_count >= 2 for c in chunks)


def test_extreme_overlap_still_terminates_and_advances():
    config = ChunkerConfig(target_tokens=4, max_tokens=4, overlap_tokens=3)
    text = words(60)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, config)
    assert_invariants(text, spans, chunks, config)


# ---------- natural boundaries and small chunks ----------

def test_paragraph_boundary_wins_inside_the_window():
    para1 = " ".join(["alpha"] * 7) + " end."
    text = para1 + "\n\n" + " ".join(["beta"] * 20) + "."
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert chunks[0].text == para1  # 8 tokens: paragraph beats the target of 10


def test_sentence_boundary_beats_plain_space():
    text = " ".join(["word"] * 11) + " stop. " + " ".join(["more"] * 20)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert chunks[0].text.endswith("stop.")
    assert chunks[0].token_count == 12


def test_equal_boundaries_are_resolved_towards_the_target():
    text = " ".join(["a"] * 6) + "\n\n" + " ".join(["b"] * 5) + "\n\n" + " ".join(["c"] * 20)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert chunks[0].token_count == 11  # paragraph ends at 6 and 11; 11 is closer to 10


def test_early_paragraph_below_the_preferred_minimum_is_not_used():
    text = "Title here\n\n" + words(40)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[0].token_count >= SMALL.min_preferred_tokens  # not the 2-token title
    assert chunks[0].token_count == SMALL.target_tokens


def test_small_chunk_only_when_no_safe_cut_exists_in_the_preferred_range():
    cluster = KA + ANUSVARA * 11  # one grapheme cluster, 12 code-point tokens
    text = "a b c " + cluster + " " + words(20)
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[0].text == "a b c"      # widened search: sizes 5..14 all split the cluster
    assert chunks[0].token_count == 3
    assert chunks[1].text.startswith("b c " + cluster)


def test_every_non_final_chunk_reaches_the_preferred_minimum_when_cuts_are_safe():
    for seed in range(20):
        rng = random.Random(seed)
        text = "".join(
            f"w{k}" + rng.choice([" ", " ", "\n", "\n\n", ". "]) for k in range(rng.randint(50, 300))
        )
        spans = word_spans(text)
        chunks = chunk_text(text, spans, SMALL)
        assert_invariants(text, spans, chunks, SMALL)
        assert all(c.token_count >= SMALL.min_preferred_tokens for c in chunks[:-1])


def test_hindi_danda_is_a_sentence_boundary():
    sentence = "यह एक छोटा हिंदी वाक्य है।"  # 6 tokens
    text = " ".join([sentence] * 6)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert all(c.text.endswith("।") for c in chunks)


def test_consecutive_blank_lines_never_appear_at_chunk_edges():
    text = "one two three\n\n\n\n\nfour five six\n\n\n" + words(20)
    spans = word_spans(text)
    config = ChunkerConfig(target_tokens=4, max_tokens=6, overlap_tokens=1)
    chunks = chunk_text(text, spans, config)
    assert_invariants(text, spans, chunks, config)
    assert chunks[0].text == "one two three"
    assert all(not c.text.startswith("\n") and not c.text.endswith("\n") for c in chunks)


# ---------- token offsets that include whitespace ----------

def test_leading_whitespace_in_token_offsets_is_trimmed_and_still_counted():
    text = "  " + words(40, prefix="word")
    spans = leading_space_spans(text)          # e.g. (0, 7) for "  word0", (7, 13) for " word1"
    chunks = chunk_text(text, spans, SMALL)
    assert len(chunks) > 1
    assert_invariants(text, spans, chunks, SMALL)
    for c in chunks:
        assert c.token_count == len(c.text.split())  # one token per word, leading space included
        assert text[spans[c.token_start][0]:c.start].isspace()


def test_whitespace_only_tokens_keep_natural_boundaries_and_are_counted():
    para1 = " ".join(["alpha"] * 3) + " end."
    text = para1 + "\n\n" + " ".join(["beta"] * 20)
    spans = whitespace_token_spans(text)      # words and whitespace runs are separate tokens
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    first = chunks[0]
    assert first.text == para1                 # the "\n\n" token is still a paragraph break
    # Both cuts around the "\n\n" token rank as PARAGRAPH; the tie goes to the later cut,
    # so the trimmed "\n\n" token is the last token of the range and is counted:
    # 4 words + 3 single-space tokens + 1 "\n\n" token.
    last_s, last_e = spans[first.token_end - 1]
    assert text[last_s:last_e] == "\n\n"
    assert first.token_count == 8
    for c in chunks:
        assert c.token_count >= len(c.text.split())


# ---------- Unicode safety and exact offsets ----------

def test_offsets_are_python_code_points_with_emoji_and_hindi():
    text = "Intro 🙂 नमस्ते दुनिया। Next 🙂 sentence here. " + words(30)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    # Code-point offsets differ from UTF-16 offsets once an emoji precedes the chunk.
    second = chunks[1]
    assert len(text[:second.start].encode("utf-16-le")) // 2 != second.start


def test_source_text_is_never_unicode_normalised():
    decomposed = "cafe\u0301 क\u093c " + words(20)  # NFD forms on purpose
    spans = word_spans(decomposed)
    chunks = chunk_text(decomposed, spans, SMALL)
    assert_invariants(decomposed, spans, chunks, SMALL)
    assert chunks[0].text.startswith("cafe\u0301 क\u093c")
    assert unicodedata.normalize("NFC", chunks[0].text) != chunks[0].text


def test_hindi_without_spaces_is_never_cut_inside_a_cluster():
    text = "नमस्तेदुनियाक्षत्रज्ञ" * 40
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert len(chunks) > 1
    assert_invariants(text, spans, chunks, SMALL)


def test_long_hindi_conjuncts_stay_whole():
    conjunct = (KA + VIRAMA) * 4 + SSA          # 9 code points, one conjunct cluster
    text = " ".join([conjunct] * 20)
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    for c in chunks:
        assert c.text.startswith(conjunct) and c.text.endswith(conjunct)


def test_emoji_zwj_sequences_are_not_split():
    family = "👨" + ZWJ + "👩" + ZWJ + "👧"
    text = family * 30
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    for c in chunks:
        assert c.text[0] != ZWJ and c.text[-1] != ZWJ


def test_joiners_left_between_tokens_are_never_cut_points():
    # A tokenizer may leave ZWJ (a format character) outside every token.
    family = "👨" + ZWJ + "👩" + ZWJ + "👧"
    text = family * 30
    spans = [(i, i + 1) for i, ch in enumerate(text) if ch != ZWJ]
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)


def test_zwnj_stays_with_the_preceding_half_form():
    text = (KA + VIRAMA + ZWNJ + SSA) * 60     # clusters: [KA VIRAMA ZWNJ] [SSA]
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    for c in chunks:
        assert c.text[0] in (KA, SSA)           # never starts with VIRAMA or ZWNJ
        assert c.text[-1] in (ZWNJ, SSA)         # never ends between KA and VIRAMA


def test_skin_tone_modifiers_and_variation_selectors_stay_attached():
    text = ("👍🏽" + "❤\ufe0f") * 40
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)


def test_long_url_is_split_within_budget():
    url = "https://example.com/" + "/".join(f"segment{k}" for k in range(250))
    text = "See " + url + " for details."
    spans = piece_spans(text, 4)
    chunks = chunk_text(text, spans)
    assert len(chunks) >= 2
    assert_invariants(text, spans, chunks, DEFAULT_CHUNKER_CONFIG)


def test_very_long_word_uses_hard_cuts_closest_to_the_target():
    text = "x" * 200
    spans = piece_spans(text, 1)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[0].token_count == SMALL.target_tokens


def test_mixed_script_text_without_any_whitespace():
    text = ("abcनमस्ते🙂" + KA + VIRAMA + ZWJ + SSA) * 30
    spans = piece_spans(text, 2)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)


# ---------- UAX #29 extended grapheme clusters ----------

@pytest.mark.parametrize("text, expected", [
    (L_JAMO + V_JAMO, [L_JAMO + V_JAMO]),                                # Hangul L+V (GB6)
    (L_JAMO + V_JAMO + T_JAMO, [L_JAMO + V_JAMO + T_JAMO]),              # Hangul L+V+T (GB6, GB7)
    ("\uac00" + T_JAMO, ["\uac00" + T_JAMO]),                            # LV syllable + T (GB7)
    (RI_I + RI_N + RI_I, [RI_I + RI_N, RI_I]),                           # RI pairs (GB12)
    (RI_I + RI_N + RI_I + RI_N, [RI_I + RI_N, RI_I + RI_N]),
    ("a\r\nb", ["a", "\r\n", "b"]),                                     # CR LF (GB3)
    ("a\r\n\nb", ["a", "\r\n", "\n", "b"]),
    ("👨" + ZWJ + "👩" + ZWJ + "👧", ["👨" + ZWJ + "👩" + ZWJ + "👧"]),     # emoji ZWJ (GB11)
    ("👍🏽", ["👍🏽"]),                                                    # modifier (GB9)
    ("❤\ufe0f", ["❤\ufe0f"]),                                            # variation selector
    (KA + VIRAMA + SSA, [KA + VIRAMA + SSA]),                            # conjunct (GB9c)
    ((KA + VIRAMA) * 3 + SSA, [(KA + VIRAMA) * 3 + SSA]),                # long conjunct chain
    (KA + VIRAMA + ZWNJ + SSA, [KA + VIRAMA + ZWNJ, SSA]),               # ZWNJ half form
    ("e\u0301", ["e\u0301"]),                                            # combining mark
    ("a \u0301b", ["a", " \u0301", "b"]),                                 # mark on a space
    ("x" + ZWJ + "y", ["x" + ZWJ, "y"]),                                 # ZWJ between letters (GB9)
    (PREPEND + " a", [PREPEND + " ", "a"]),                              # prepend (GB9b)
], ids=["hangul-lv", "hangul-lvt", "hangul-lv-syllable-t", "ri-3", "ri-4", "crlf", "crlf-lf",
        "emoji-zwj", "skin-tone", "vs16", "conjunct", "conjunct-chain", "zwnj-half-form",
        "e-acute", "space-mark", "zwj-between-letters", "prepend-space"])
def test_regex_segmentation_matches_hand_derived_uax29_clusters(text, expected):
    assert [m.group() for m in regex.finditer(r"\X", text)] == expected


def test_hangul_lv_syllables_are_never_split():
    text = (L_JAMO + V_JAMO) * 30               # reproduces the reported L|V split
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, NO_OVERLAP)
    assert_invariants(text, spans, chunks, NO_OVERLAP)
    assert all(c.start % 2 == 0 and c.end % 2 == 0 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_hangul_lvt_syllables_are_never_split():
    text = (L_JAMO + V_JAMO + T_JAMO) * 30
    spans = codepoint_spans(text)
    for config in (NO_OVERLAP, SMALL):
        chunks = chunk_text(text, spans, config)
        assert_invariants(text, spans, chunks, config)
        assert all(c.start % 3 == 0 and c.end % 3 == 0 for c in chunks)


def test_regional_indicators_pair_from_the_start_of_the_run():
    text = RI_I * 41                            # 20 flags + 1 lone indicator
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, NO_OVERLAP)
    assert_invariants(text, spans, chunks, NO_OVERLAP)
    for c in chunks:
        assert c.start % 2 == 0                 # a flag is never split
        assert c.end % 2 == 0 or c.end == 41


def test_regional_indicator_pairing_restarts_after_other_characters():
    group = "x" + RI_I + RI_N + RI_I            # clusters: [x] [RI RI] [RI]
    text = group * 12
    spans = codepoint_spans(text)
    chunks = chunk_text(text, spans, NO_OVERLAP)
    assert_invariants(text, spans, chunks, NO_OVERLAP)
    allowed = {g * 4 + d for g in range(12) for d in (0, 1, 3)} | {len(text)}
    assert all(c.start in allowed and c.end in allowed for c in chunks)


def test_crlf_is_never_split_and_never_left_at_chunk_edges():
    text = "\r\n".join(f"line {k} has some words" for k in range(30))
    spans = char_spans(text)                    # "\r" and "\n" are separate tokens
    config = ChunkerConfig(target_tokens=40, max_tokens=60, overlap_tokens=5)
    chunks = chunk_text(text, spans, config)
    assert_invariants(text, spans, chunks, config)
    for c in chunks:
        assert c.text[0] not in "\r\n" and c.text[-1] not in "\r\n"
        assert text[c.start - 1:c.start + 1] != "\r\n"   # start is not between CR and LF
        assert text[c.end - 1:c.end + 1] != "\r\n"       # end is not between CR and LF


def test_crlf_blank_line_is_a_paragraph_boundary():
    para1 = " ".join(["alpha"] * 7) + " end."
    text = para1 + "\r\n\r\n" + " ".join(["beta"] * 20)
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[0].text == para1


def test_cut_before_a_mark_attached_to_whitespace_is_not_used():
    para1 = " ".join(["alpha"] * 7) + " end."
    text = para1 + "\n\n \u0301" + " ".join(["beta"] * 20)  # " \u0301" is one cluster
    spans = word_spans(text)
    chunks = chunk_text(text, spans, SMALL)
    assert_invariants(text, spans, chunks, SMALL)
    assert chunks[0].end != len(para1)          # the paragraph cut would split " \u0301"
    assert all(not unicodedata.combining(c.text[0]) for c in chunks)


def test_joiner_left_between_letters_blocks_the_cut_before_it():
    # Tokenizer leaves ZWJ between the tokens; clusters are [x ZWJ] [y]: only the
    # left edge of the cut "x | ZWJ | y" would split a cluster.
    text = ("x" + ZWJ + "y") * 30
    spans = [(i, i + 1) for i, ch in enumerate(text) if ch != ZWJ]
    chunks = chunk_text(text, spans, NO_OVERLAP)
    assert_invariants(text, spans, chunks, NO_OVERLAP)
    for c in chunks:
        assert c.text[0] == "x" and c.text[-1] == "y"


def test_prepend_character_keeps_the_following_space():
    # [PREPEND " "] is one cluster, so a chunk may not end right after PREPEND:
    # only the left edge of that cut would split a cluster.
    text = " ".join(["word", PREPEND] * 30 + ["word"])
    spans = word_spans(text)
    chunks = chunk_text(text, spans, NO_OVERLAP)
    assert_invariants(text, spans, chunks, NO_OVERLAP)
    assert all(c.text[-1] != PREPEND for c in chunks)


def test_mark_attached_to_leading_whitespace_is_refused():
    text = " \u0301abc def"                     # the first cluster is " \u0301"
    with pytest.raises(ChunkingError, match="would split a grapheme cluster"):
        chunk_text(text, word_spans(text), SMALL)


# ---------- safe failure when no grapheme-safe boundary exists ----------

def test_combining_sequence_within_the_budget_is_one_chunk():
    text = KA + ANUSVARA * 10                    # 11 tokens <= max 14
    spans = codepoint_spans(text)
    [c] = chunk_text(text, spans, SMALL)
    assert c.text == text and c.token_count == 11


@pytest.mark.parametrize("text", [
    KA + ANUSVARA * 1000,                              # extremely long combining sequence
    "start " + KA + ANUSVARA * 100 + " end",           # unsafe run in the middle of text
    (KA + VIRAMA) * 20 + SSA,                          # conjunct chain longer than the budget
    "👨" + (ZWJ + "👩") * 20,                          # ZWJ sequence longer than the budget
], ids=["combining-1000", "combining-mid-text", "long-conjunct-chain", "long-zwj-sequence"])
def test_no_grapheme_safe_boundary_within_budget_raises(text):
    spans = codepoint_spans(text)
    with pytest.raises(ChunkingError, match="no grapheme-safe chunk boundary"):
        chunk_text(text, spans, SMALL)


def test_failure_returns_nothing_even_after_valid_chunks():
    text = words(40) + " " + KA + ANUSVARA * 100   # valid chunks first, then an unsafe run
    spans = codepoint_spans(text)
    result = None
    with pytest.raises(ChunkingError):
        result = chunk_text(text, spans, SMALL)
    assert result is None                       # no partial list escapes


# ---------- determinism and randomised invariants ----------

def test_chunking_is_deterministic():
    text = "\n\n".join(words(37, prefix=f"p{k}_") for k in range(10))
    spans = word_spans(text)
    assert chunk_text(text, spans, SMALL) == chunk_text(text, spans, SMALL)


ALPHABET = ["the", "data", "नमस्ते", "दुनिया", "क्षत्र", "🙂", "👍🏽", "end.", "रुको।",
            "https://x.io/a/b", "e\u0301"]
SEPARATORS = [" ", " ", " ", "\n", "\n\n", "\n\n\n", "  ", "\t", ""]


@pytest.mark.parametrize("seed", range(40))
def test_randomised_documents_are_deterministic_and_complete(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(ALPHABET) + rng.choice(SEPARATORS) for _ in range(rng.randint(0, 400)))
    spans = piece_spans(text, rng.choice([1, 2, 3, 5]))
    target = rng.randint(2, 30)
    config = ChunkerConfig(
        target_tokens=target,
        max_tokens=target + rng.randint(0, 20),
        overlap_tokens=rng.randint(0, target - 1),
    )
    first, second = outcome(text, spans, config), outcome(text, spans, config)
    assert first == second                      # same chunks, or the same error
    if isinstance(first, str):
        assert first.startswith("no grapheme-safe chunk boundary")
    else:
        assert_invariants(text, spans, first, config)


# ---------- invalid token spans ----------

@pytest.mark.parametrize("spans", [
    [(0, 6)],               # beyond the end of "hello"
    [(-1, 2)],              # negative start
    [(3, 2)],               # start after end
    [(0, 3), (2, 5)],       # overlapping
    [(3, 5), (0, 2)],       # out of order
    [(0, 0), (0, 5)],       # zero-length at the start
    [(0, 5), (5, 5)],       # zero-length at the end
    [(0, 2), (2, 2), (2, 5)],  # zero-length in the middle
], ids=["past-end", "negative", "reversed", "overlapping", "unordered",
        "zero-length-first", "zero-length-last", "zero-length-middle"])
def test_invalid_token_spans_are_rejected(spans):
    with pytest.raises(ChunkingError):
        chunk_text("hello", spans)


@pytest.mark.parametrize("spans", [
    [(0.0, 5)],
    [(0, 5.0)],
    [("0", 5)],
    [(0, "5")],
    [(False, 5)],
    [(0, 5, 6)],
    [(0,)],
    [5],
    "05",
    None,
], ids=["float-start", "float-end", "str-start", "str-end", "bool-start",
        "triple", "single", "bare-int", "string", "none"])
def test_malformed_token_spans_are_rejected_not_coerced(spans):
    with pytest.raises(ChunkingError):
        chunk_text("hello", spans)


def test_list_pairs_are_accepted_like_tuples():
    assert chunk_text("hello", [[0, 5]]) == chunk_text("hello", [(0, 5)])


def test_non_string_text_is_rejected():
    with pytest.raises(ChunkingError):
        chunk_text(b"hello", [(0, 5)])


def test_content_not_covered_by_tokens_is_refused():
    with pytest.raises(ChunkingError, match="not covered"):
        chunk_text("hello world", [(0, 5)])


def test_whitespace_and_format_characters_may_sit_between_tokens():
    text = "hello\u200bworld"  # zero-width space (Cf) between two tokens
    [c] = chunk_text(text, [(0, 5), (6, 11)], SMALL)
    assert c.text == text