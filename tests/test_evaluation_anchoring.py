"""E1.2c anchoring: source-text verification and question span anchoring.

A  verify_source_text: accepted text
B  verify_source_text: rejected text and malformed entries
C  resolve_question: anchored spans
D  resolve_question: rejected spans (exact codes and locations)
E  Unicode and offset correctness
F  grapheme-edge warnings
G  structure, ordering, no partial success, no re-anchoring
H  argument guards
I  information leakage
J  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from app.rag.evaluation import anchoring as an
from app.rag.evaluation import schema as s
from app.rag.evaluation import validation as v

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECRET = "SECRET-SOURCE-TEXT"
SECRET_KEY = "secret-corpus-key"  # a valid corpus key, so it reaches the anchoring checks

EMOJI = chr(0x1F600)
LIGATURE_FI = chr(0xFB01)
E_ACUTE_NFC = chr(0x00E9)
E_ACUTE_NFD = "e" + chr(0x0301)
KA_AA = chr(0x0915) + chr(0x093E)  # ka + vowel sign aa: one grapheme cluster of two code points
CR, LF = chr(13), chr(10)
CRLF = CR + LF
LONE_SURROGATE = chr(0xD800)
SHA_B = "b" * 64


class StrSubclass(str):
    pass


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- builders ----------

def make_entry(text: str = "hello world", **overrides) -> s.ManifestDocument:
    values = {
        "corpus_key": "doc-a",
        "title": "T",
        "publisher": "P",
        "source_url": "https://example.invalid/a",
        "license": "MIT",
        "license_url": "https://example.invalid/license",
        "format": "txt",
        "language": "en",
        "source_file_sha256": "a" * 64,
        "extracted_text_sha256": sha(text),
        "extracted_text_length": len(text),
        "extraction": None,
        "redistributable": True,
    }
    values.update(overrides)
    return s.ManifestDocument(**values)


def make_span(text: str, start: int, end: int, **overrides) -> s.Span:
    """A span of ``text`` with a consistent quote and hashes, unless ``overrides`` break it."""
    quote = text[start:end]
    values = {
        "corpus_key": "doc-a",
        "source_sha256": sha(text),
        "start": start,
        "end": end,
        "quote": quote,
        "quote_sha256": sha(quote) if quote else SHA_B,
    }
    values.update(overrides)
    return s.Span(**values)


def unit(*spans, unit_id="u1") -> s.EvidenceUnit:
    return s.EvidenceUnit(unit_id=unit_id, spans=tuple(spans))


def make_question(*units) -> s.Question:
    return s.Question(
        question_id="q1", split="test", answerable=True, question="What is it?", query_language="en",
        author_type="human", hindi_reviewed=False, units=tuple(units),
    )


P = "question.units[0].spans[0]"


def anchor(units, texts):
    q = make_question(*units) if isinstance(units, (tuple, list)) else make_question(units)
    return an.resolve_question(q, texts)


def pairs(issues):
    return [(i.code, i.location) for i in issues]


def one(span, texts):
    return pairs(an.resolve_question(make_question(unit(span)), texts))


# ---------- A: verify_source_text accepted ----------

# Explicit ids: pytest would otherwise use each string itself as its test id, and the
# 100,000-character row produces an id too long for Windows' environment-variable limit.
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("hello world", id="ascii"),
        pytest.param("x", id="single-character"),
        pytest.param("a" + EMOJI + "b", id="astral-emoji"),
        pytest.param(KA_AA + " text " + KA_AA, id="devanagari-clusters"),
        pytest.param("line1" + CRLF + "line2", id="crlf"),
        pytest.param(E_ACUTE_NFC, id="nfc-e-acute"),
        pytest.param(E_ACUTE_NFD, id="nfd-e-acute"),
        pytest.param(LIGATURE_FI + "ne", id="ligature-fi"),
        pytest.param("tab" + chr(9) + "and" + chr(0x2028) + "separator", id="tab-and-line-separator"),
        pytest.param("a" * 100000, id="long-100000-characters"),
    ],
)
def test_correct_hash_and_length_pass(text):
    assert an.verify_source_text(make_entry(text), text) == []


def test_correct_hash_without_a_recorded_length_passes():
    text = "hello world"
    assert an.verify_source_text(make_entry(text, extracted_text_length=None), text) == []


def test_length_counts_code_points_not_utf16_units():
    text = "a" + EMOJI
    assert an.verify_source_text(make_entry(text), text) == []                              # length 2
    assert pairs(an.verify_source_text(make_entry(text, extracted_text_length=3), text)) == [
        (an.SOURCE_LENGTH_MISMATCH, "source_text")]                                          # 3 would be UTF-16


# ---------- B: verify_source_text rejected ----------

H = "entry.extracted_text_sha256"
L = "entry.extracted_text_length"
TEXT = "hello world"

VERIFY_ROWS = [
    pytest.param(TEXT, {}, [], id="baseline-valid"),
    # text
    pytest.param("", {}, [(an.SOURCE_TEXT_EMPTY, "source_text")], id="empty-text"),
    pytest.param(None, {}, [(an.SOURCE_TEXT_TYPE, "source_text")], id="none-text"),
    pytest.param(b"hello world", {}, [(an.SOURCE_TEXT_TYPE, "source_text")], id="bytes-text"),
    pytest.param(5, {}, [(an.SOURCE_TEXT_TYPE, "source_text")], id="int-text"),
    pytest.param(["hello world"], {}, [(an.SOURCE_TEXT_TYPE, "source_text")], id="list-text"),
    pytest.param(StrSubclass(TEXT), {}, [(an.SOURCE_TEXT_TYPE, "source_text")], id="str-subclass-text"),
    # hash mismatch
    pytest.param("hello worle", {}, [(an.SOURCE_HASH_MISMATCH, "source_text")], id="one-character-changed"),
    pytest.param("hello world ", {}, [(an.SOURCE_HASH_MISMATCH, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")],
                 id="extra-trailing-space"),
    pytest.param(TEXT, {"extracted_text_sha256": SHA_B}, [(an.SOURCE_HASH_MISMATCH, "source_text")], id="other-hash"),
    pytest.param("hello" + CRLF + "world", {"extracted_text_sha256": sha("hello" + LF + "world"),
                                            "extracted_text_length": 11},
                 [(an.SOURCE_HASH_MISMATCH, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")],
                 id="crlf-is-not-lf"),
    # length mismatch
    pytest.param(TEXT, {"extracted_text_length": 12}, [(an.SOURCE_LENGTH_MISMATCH, "source_text")], id="length-too-long"),
    pytest.param(TEXT, {"extracted_text_length": 10}, [(an.SOURCE_LENGTH_MISMATCH, "source_text")], id="length-too-short"),
    pytest.param(TEXT, {"extracted_text_length": 0}, [(an.SOURCE_LENGTH_MISMATCH, "source_text")], id="length-zero"),
    pytest.param(TEXT, {"extracted_text_length": -11}, [(an.SOURCE_LENGTH_MISMATCH, "source_text")], id="length-negative"),
    # malformed entry metadata
    pytest.param(TEXT, {"extracted_text_sha256": None}, [(an.SOURCE_ENTRY_HASH_MISSING, H)], id="hash-missing"),
    pytest.param(TEXT, {"extracted_text_sha256": sha(TEXT).upper()}, [(an.SOURCE_ENTRY_HASH_FORMAT, H)], id="hash-uppercase"),
    pytest.param(TEXT, {"extracted_text_sha256": sha(TEXT)[:63]}, [(an.SOURCE_ENTRY_HASH_FORMAT, H)], id="hash-63"),
    pytest.param(TEXT, {"extracted_text_sha256": sha(TEXT) + LF}, [(an.SOURCE_ENTRY_HASH_FORMAT, H)], id="hash-newline"),
    pytest.param(TEXT, {"extracted_text_sha256": ""}, [(an.SOURCE_ENTRY_HASH_FORMAT, H)], id="hash-empty"),
    pytest.param(TEXT, {"extracted_text_sha256": 5}, [(an.SOURCE_ENTRY_HASH_FORMAT, H)], id="hash-int"),
    pytest.param(TEXT, {"extracted_text_length": True}, [(an.SOURCE_ENTRY_LENGTH_TYPE, L)], id="length-bool"),
    pytest.param(TEXT, {"extracted_text_length": 11.0}, [(an.SOURCE_ENTRY_LENGTH_TYPE, L)], id="length-float"),
    pytest.param(TEXT, {"extracted_text_length": "11"}, [(an.SOURCE_ENTRY_LENGTH_TYPE, L)], id="length-string"),
    pytest.param(TEXT, {"extracted_text_sha256": None, "extracted_text_length": None},
                 [(an.SOURCE_ENTRY_HASH_MISSING, H)], id="draft-entry-is-not-verifiable"),
    # several problems, fixed order
    pytest.param("", {"extracted_text_sha256": None, "extracted_text_length": "x"},
                 [(an.SOURCE_TEXT_EMPTY, "source_text"), (an.SOURCE_ENTRY_HASH_MISSING, H), (an.SOURCE_ENTRY_LENGTH_TYPE, L)],
                 id="empty-text-and-bad-entry"),
    pytest.param(None, {"extracted_text_sha256": "zz"},
                 [(an.SOURCE_TEXT_TYPE, "source_text"), (an.SOURCE_ENTRY_HASH_FORMAT, H)], id="bad-text-type-and-hash-format"),
]


@pytest.mark.parametrize("text, overrides, expected", VERIFY_ROWS)
def test_verify_source_text_rows(text, overrides, expected):
    entry_text = TEXT
    entry = make_entry(entry_text, **overrides)
    assert pairs(an.verify_source_text(entry, text)) == expected


def test_lone_surrogate_text_is_reported_not_raised():
    bad = "ab" + LONE_SURROGATE
    assert pairs(an.verify_source_text(make_entry("abc", extracted_text_length=3), bad)) == [
        (an.SOURCE_TEXT_ENCODING, "source_text")]
    assert pairs(an.verify_source_text(make_entry("abc", extracted_text_length=9), bad)) == [
        (an.SOURCE_TEXT_ENCODING, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")]


def test_nfc_and_nfd_texts_are_different_sources():
    nfc, nfd = "x" + E_ACUTE_NFC + "y", "x" + E_ACUTE_NFD + "y"
    assert an.verify_source_text(make_entry(nfc), nfc) == []
    assert pairs(an.verify_source_text(make_entry(nfc), nfd)) == [
        (an.SOURCE_HASH_MISMATCH, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")]
    assert pairs(an.verify_source_text(make_entry(nfd), nfc)) == [
        (an.SOURCE_HASH_MISMATCH, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")]


def test_a_ligature_is_not_expanded_when_verifying():
    ligature, expanded = LIGATURE_FI + "ne", "fine"
    assert pairs(an.verify_source_text(make_entry(ligature), expanded)) == [
        (an.SOURCE_HASH_MISMATCH, "source_text"), (an.SOURCE_LENGTH_MISMATCH, "source_text")]


def test_garbage_entry_metadata_never_raises():
    garbage = s.ManifestDocument(*([[1], {}, 4.5, (6,), None, b"x", object(), 7, {}, [], (), 1j, "x"]))
    got = an.verify_source_text(garbage, "text")
    assert got and all(i.severity == s.SEVERITY_ERROR for i in got)


def test_verify_source_text_requires_a_manifest_document():
    for bad in (None, "x", {}, s.Manifest("a", "b", "draft", ())):
        with pytest.raises(TypeError, match="ManifestDocument"):
            an.verify_source_text(bad, "text")


# ---------- C: resolve_question anchored ----------

def test_a_span_at_the_start_of_the_text_is_anchored():
    text = "hello world"
    assert one(make_span(text, 0, 5), {"doc-a": text}) == []


def test_a_span_ending_exactly_at_the_end_of_the_text_is_anchored():
    text = "hello world"
    assert one(make_span(text, 6, len(text)), {"doc-a": text}) == []


def test_the_whole_text_is_a_valid_span():
    text = "hello"
    assert one(make_span(text, 0, 5), {"doc-a": text}) == []


def test_several_units_with_alternative_spans_are_all_checked():
    text = "alpha beta gamma delta epsilon"
    units = [
        unit(make_span(text, 0, 5), make_span(text, 6, 10), unit_id="u1"),
        unit(make_span(text, 11, 16), unit_id="u2"),
        unit(make_span(text, 17, 22), make_span(text, 23, 30), make_span(text, 0, 30), unit_id="u3"),
    ]
    assert anchor(units, {"doc-a": text}) == []


def test_spans_in_several_documents_are_each_checked_against_their_own_text():
    a, b = "first document text", "second document text"
    units = [unit(make_span(a, 0, 5), make_span(b, 0, 6, corpus_key="doc-b"))]
    texts = {"doc-a": a, "doc-b": b, "doc-unused": "not referenced"}
    assert anchor(units, texts) == []


def test_a_question_without_units_has_nothing_to_anchor():
    assert an.resolve_question(make_question(), {}) == []


def test_other_mapping_types_work_and_are_not_modified():
    text = "hello world"
    texts = MappingProxyType({"doc-a": text})
    assert one(make_span(text, 0, 5), texts) == []
    plain = {"doc-a": text}
    snapshot = dict(plain)
    one(make_span(text, 0, 5), plain)
    assert plain == snapshot


# ---------- D: resolve_question rejected ----------

TEXT_D = "hello world"


def d_span(**overrides):
    return dataclasses.replace(make_span(TEXT_D, 0, 5), **overrides)


D_ROWS = [
    pytest.param(d_span(), {}, [(an.ANCHOR_TEXT_MISSING, f"{P}.corpus_key")], id="no-texts"),
    pytest.param(d_span(), {"doc-b": TEXT_D}, [(an.ANCHOR_TEXT_MISSING, f"{P}.corpus_key")], id="other-key-only"),
    pytest.param(d_span(), {"doc-a": ""}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")], id="empty-text"),
    pytest.param(d_span(), {"doc-a": None}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")], id="none-text"),
    pytest.param(d_span(), {"doc-a": b"hello world"}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")], id="bytes-text"),
    pytest.param(d_span(), {"doc-a": 5}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")], id="int-text"),
    pytest.param(d_span(), {"doc-a": StrSubclass(TEXT_D)}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")], id="str-subclass-text"),
    pytest.param(d_span(), {"doc-a": "hello" + LONE_SURROGATE}, [(an.ANCHOR_TEXT_INVALID, f"{P}.corpus_key")],
                 id="lone-surrogate-text"),
    pytest.param(d_span(source_sha256=SHA_B), {"doc-a": TEXT_D}, [(an.ANCHOR_SOURCE_HASH_MISMATCH, f"{P}.source_sha256")],
                 id="wrong-span-source-hash"),
    pytest.param(d_span(), {"doc-a": "hello worle"}, [(an.ANCHOR_SOURCE_HASH_MISMATCH, f"{P}.source_sha256")],
                 id="source-changed-by-one-character"),
    pytest.param(d_span(), {"doc-a": TEXT_D + " "}, [(an.ANCHOR_SOURCE_HASH_MISMATCH, f"{P}.source_sha256")],
                 id="source-with-trailing-space"),
    # offsets beyond the end: quote length is consistent with end - start, so only anchoring notices
    pytest.param(s.Span("doc-a", sha(TEXT_D), 8, 14, "rldxyz", sha("rldxyz")), {"doc-a": TEXT_D},
                 [(an.ANCHOR_END_BEYOND_TEXT, f"{P}.end")], id="end-beyond-text"),
    pytest.param(s.Span("doc-a", sha(TEXT_D), 11, 12, "x", sha("x")), {"doc-a": TEXT_D},
                 [(an.ANCHOR_END_BEYOND_TEXT, f"{P}.end")], id="start-at-text-end"),
    pytest.param(s.Span("doc-a", sha(TEXT_D), 100, 105, "hello", sha("hello")), {"doc-a": TEXT_D},
                 [(an.ANCHOR_END_BEYOND_TEXT, f"{P}.end")], id="both-offsets-beyond-text"),
    # structural problems are reported by validate_span and block anchoring
    pytest.param(d_span(start=-1, end=4), {"doc-a": TEXT_D}, [(s.SPAN_OFFSET_RANGE, f"{P}.offsets")], id="negative-start"),
    pytest.param(d_span(start=5, end=5), {"doc-a": TEXT_D}, [(s.SPAN_OFFSET_RANGE, f"{P}.offsets")], id="start-equals-end"),
    pytest.param(d_span(start=7, end=2), {"doc-a": TEXT_D}, [(s.SPAN_OFFSET_RANGE, f"{P}.offsets")], id="start-after-end"),
    pytest.param(d_span(start=True), {"doc-a": TEXT_D}, [(s.SPAN_FIELD_TYPE, f"{P}.start")], id="bool-start"),
    pytest.param(d_span(end=5.0), {"doc-a": TEXT_D}, [(s.SPAN_FIELD_TYPE, f"{P}.end")], id="float-end"),
    pytest.param(d_span(corpus_key="Bad Key"), {"Bad Key": TEXT_D}, [(s.SPAN_CORPUS_KEY, f"{P}.corpus_key")], id="bad-corpus-key"),
    pytest.param(d_span(source_sha256="X" * 64), {"doc-a": TEXT_D}, [(s.SPAN_SOURCE_SHA256_FORMAT, f"{P}.source_sha256")],
                 id="bad-source-hash-format"),
    # quote
    pytest.param(d_span(quote="jello", quote_sha256=sha("jello")), {"doc-a": TEXT_D},
                 [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")], id="right-offsets-wrong-quote"),
    pytest.param(d_span(quote="HELLO", quote_sha256=sha("HELLO")), {"doc-a": TEXT_D},
                 [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")], id="case-differs"),
    pytest.param(d_span(quote="hello ", quote_sha256=sha("hello ")), {"doc-a": TEXT_D},
                 [(s.SPAN_QUOTE_LENGTH, f"{P}.quote"), (an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")],
                 id="quote-length-differs-from-offsets"),
    pytest.param(d_span(quote_sha256=SHA_B), {"doc-a": TEXT_D}, [(s.SPAN_QUOTE_SHA256_MISMATCH, f"{P}.quote_sha256")],
                 id="wrong-quote-hash-with-correct-anchoring"),
    pytest.param(d_span(quote="jello", quote_sha256=SHA_B), {"doc-a": TEXT_D},
                 [(s.SPAN_QUOTE_SHA256_MISMATCH, f"{P}.quote_sha256"), (an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")],
                 id="wrong-quote-and-wrong-hash"),
    pytest.param(d_span(quote="", quote_sha256=SHA_B), {"doc-a": TEXT_D},
                 [(s.SPAN_QUOTE_EMPTY, f"{P}.quote"), (s.SPAN_QUOTE_SHA256_MISMATCH, f"{P}.quote_sha256"),
                  (an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")], id="empty-quote"),
]


@pytest.mark.parametrize("span, texts, expected", D_ROWS)
def test_rejected_span_rows(span, texts, expected):
    assert one(span, texts) == expected


def test_a_text_that_does_not_match_the_span_hash_blocks_every_later_check():
    # The offsets and quote would match this changed text, yet only the hash mismatch is reported.
    changed = "hello worle"
    assert one(make_span(TEXT_D, 0, 5), {"doc-a": changed}) == [(an.ANCHOR_SOURCE_HASH_MISMATCH, f"{P}.source_sha256")]


# ---------- E: Unicode and offset correctness ----------

def test_nfc_quote_does_not_match_nfd_text():
    text = "x" + E_ACUTE_NFD + "y"                       # x e U+0301 y
    span = s.Span("doc-a", sha(text), 1, 2, E_ACUTE_NFC, sha(E_ACUTE_NFC))
    assert one(span, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_nfd_quote_does_not_match_nfc_text():
    text = "x" + E_ACUTE_NFC + " y"
    span = s.Span("doc-a", sha(text), 1, 3, E_ACUTE_NFD, sha(E_ACUTE_NFD))
    assert one(span, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_nfc_and_nfd_are_each_anchored_in_their_own_text():
    nfc, nfd = "x" + E_ACUTE_NFC + "y", "x" + E_ACUTE_NFD + "y"
    assert one(make_span(nfc, 1, 2), {"doc-a": nfc}) == []
    assert one(make_span(nfd, 1, 3), {"doc-a": nfd}) == []


def test_an_nfd_offset_range_over_nfc_text_is_not_silently_repaired():
    nfc = "x" + E_ACUTE_NFC + "y"                         # length 3
    span = s.Span("doc-a", sha(nfc), 1, 3, E_ACUTE_NFD, sha(E_ACUTE_NFD))   # NFD length 2, offsets say 2
    assert one(span, {"doc-a": nfc}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_a_ligature_is_never_expanded():
    ligature_text = LIGATURE_FI + "ne"                      # 3 code points
    expanded_quote = s.Span("doc-a", sha(ligature_text), 0, 2, "fi", sha("fi"))
    assert one(expanded_quote, {"doc-a": ligature_text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]
    plain_text = "fine"
    ligature_quote = s.Span("doc-a", sha(plain_text), 0, 1, LIGATURE_FI, sha(LIGATURE_FI))
    assert one(ligature_quote, {"doc-a": plain_text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]
    assert one(make_span(ligature_text, 0, 1), {"doc-a": ligature_text}) == []   # the ligature itself anchors


def test_astral_characters_use_code_point_offsets():
    text = EMOJI + " hello world"
    assert one(make_span(text, 2, 7), {"doc-a": text}) == []          # "hello" by code points
    assert one(make_span(text, 0, 1), {"doc-a": text}) == []          # the emoji is one position
    assert one(make_span(text, 1, 1 + 1), {"doc-a": text}) == []      # the space after it


def test_utf16_offsets_for_text_with_astral_characters_are_rejected():
    text = EMOJI + " hello world"
    utf16 = s.Span("doc-a", sha(text), 3, 8, "hello", sha("hello"))   # "hello" starts at UTF-16 index 3
    assert one(utf16, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_utf16_offsets_past_the_end_are_rejected_as_beyond_the_text():
    text = EMOJI + " hello"                                         # 7 code points
    utf16 = s.Span("doc-a", sha(text), 3, 8, "hello", sha("hello"))
    assert one(utf16, {"doc-a": text}) == [(an.ANCHOR_END_BEYOND_TEXT, f"{P}.end")]


def test_crlf_in_a_quote_is_exact():
    text = "line one" + CRLF + "line two"
    assert one(make_span(text, 0, 10), {"doc-a": text}) == []          # "line one" + CRLF
    lf_quote = "line one" + LF
    wrong = s.Span("doc-a", sha(text), 0, 9, lf_quote, sha(lf_quote))
    assert one(wrong, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_lf_text_does_not_match_a_crlf_quote():
    text = "line one" + LF + "line two"
    crlf_quote = "line one" + CRLF
    span = s.Span("doc-a", sha(text), 0, 10, crlf_quote, sha(crlf_quote))
    assert one(span, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_devanagari_is_matched_exactly_without_normalisation():
    text = "ab" + KA_AA + KA_AA + "cd"
    assert one(make_span(text, 2, 4), {"doc-a": text}) == []           # one whole cluster
    assert one(make_span(text, 2, 6), {"doc-a": text}) == []           # two clusters
    swapped = KA_AA[::-1]
    wrong = s.Span("doc-a", sha(text), 2, 4, swapped, sha(swapped))
    assert one(wrong, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]


def test_a_lone_surrogate_quote_is_a_structural_error_not_a_crash():
    text = "abc" + "d"
    span = s.Span("doc-a", sha(text), 0, 1, LONE_SURROGATE, SHA_B)
    got = one(span, {"doc-a": text})
    assert (s.SPAN_QUOTE_ENCODING, f"{P}.quote") in got
    assert (an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote") in got


# ---------- F: grapheme-edge warnings ----------

def with_severity(span, texts):
    return [(i.code, i.location, i.severity) for i in an.resolve_question(make_question(unit(span)), texts)]


def test_spans_on_grapheme_boundaries_give_no_warnings():
    text = "ab" + KA_AA + "cd" + E_ACUTE_NFD + "ef" + CRLF + "gh"
    for start, end in ((0, 2), (2, 4), (4, 6), (2, 9), (0, len(text))):
        assert with_severity(make_span(text, start, end), {"doc-a": text}) == []


def test_a_span_starting_inside_a_cluster_warns():
    text = "ab" + KA_AA + "cd"
    assert with_severity(make_span(text, 3, 5), {"doc-a": text}) == [
        (an.ANCHOR_START_INSIDE_GRAPHEME, f"{P}.start", s.SEVERITY_WARNING)]


def test_a_span_ending_inside_a_cluster_warns():
    text = "ab" + KA_AA + "cd"
    assert with_severity(make_span(text, 1, 3), {"doc-a": text}) == [
        (an.ANCHOR_END_INSIDE_GRAPHEME, f"{P}.end", s.SEVERITY_WARNING)]


def test_both_edges_inside_clusters_warn_start_then_end():
    text = "ab" + KA_AA + KA_AA + "cd"           # clusters at 2-3 and 4-5
    assert with_severity(make_span(text, 3, 5), {"doc-a": text}) == [
        (an.ANCHOR_START_INSIDE_GRAPHEME, f"{P}.start", s.SEVERITY_WARNING),
        (an.ANCHOR_END_INSIDE_GRAPHEME, f"{P}.end", s.SEVERITY_WARNING)]


@pytest.mark.parametrize(
    "text, start, end, code",
    [
        ("a" + E_ACUTE_NFD + "b", 2, 3, an.ANCHOR_START_INSIDE_GRAPHEME),          # between e and the combining mark
        ("a" + E_ACUTE_NFD + "b", 0, 2, an.ANCHOR_END_INSIDE_GRAPHEME),
        ("ab" + CRLF + "cd", 3, 5, an.ANCHOR_START_INSIDE_GRAPHEME),               # between CR and LF
        ("ab" + CRLF + "cd", 0, 3, an.ANCHOR_END_INSIDE_GRAPHEME),
        (chr(0x1F44D) + chr(0x1F3FD) + "x", 1, 3, an.ANCHOR_START_INSIDE_GRAPHEME),  # emoji + skin-tone modifier
    ],
)
def test_other_cluster_kinds_warn(text, start, end, code):
    got = with_severity(make_span(text, start, end), {"doc-a": text})
    assert [(c, sev) for c, _, sev in got] == [(code, s.SEVERITY_WARNING)]


def test_warnings_do_not_make_the_result_invalid():
    text = "ab" + KA_AA + "cd"
    issues = an.resolve_question(make_question(unit(make_span(text, 3, 5))), {"doc-a": text})
    assert issues and all(i.severity == s.SEVERITY_WARNING for i in issues)
    assert v.require_valid(issues) is None


def test_the_span_is_not_altered_by_a_warning():
    text = "ab" + KA_AA + "cd"
    span = make_span(text, 3, 5)
    an.resolve_question(make_question(unit(span)), {"doc-a": text})
    assert (span.start, span.end, span.quote) == (3, 5, text[3:5])


def test_no_warning_is_added_to_a_span_that_failed_anchoring():
    text = "ab" + KA_AA + "cd"
    wrong = s.Span("doc-a", sha(text), 3, 5, "zz", sha("zz"))     # starts inside a cluster, wrong quote
    assert with_severity(wrong, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote", s.SEVERITY_ERROR)]


def test_an_error_and_a_warning_together_still_fail_require_valid():
    text = "ab" + KA_AA + "cd"
    units = [unit(make_span(text, 3, 5)), unit(make_span(text, 0, 2, quote_sha256=SHA_B), unit_id="u2")]
    issues = an.resolve_question(make_question(*units), {"doc-a": text})
    assert {i.severity for i in issues} == {s.SEVERITY_WARNING, s.SEVERITY_ERROR}
    with pytest.raises(s.EvaluationDataError) as caught:
        v.require_valid(issues)
    assert caught.value.issues == tuple(issues)


# ---------- G: structure, ordering, no partial success ----------

def test_issues_follow_unit_then_span_order_with_positional_locations():
    text = "hello world"
    good = make_span(text, 0, 5)
    bad_quote = make_span(text, 0, 5, quote="jello", quote_sha256=sha("jello"))
    units = [unit(good, bad_quote, unit_id="u1"), unit(bad_quote, good, make_span(text, 0, 5, source_sha256=SHA_B), unit_id="u2")]
    assert pairs(anchor(units, {"doc-a": text})) == [
        ("ANCHOR_QUOTE_MISMATCH", "question.units[0].spans[1].quote"),
        ("ANCHOR_QUOTE_MISMATCH", "question.units[1].spans[0].quote"),
        ("ANCHOR_SOURCE_HASH_MISMATCH", "question.units[1].spans[2].source_sha256"),
    ]


def test_structural_issues_come_before_anchoring_issues_for_the_same_span():
    text = "hello world"
    span = make_span(text, 0, 5, quote="jello", quote_sha256=SHA_B)
    assert [c for c, _ in one(span, {"doc-a": text})] == [s.SPAN_QUOTE_SHA256_MISMATCH, an.ANCHOR_QUOTE_MISMATCH]


def test_one_bad_span_makes_the_whole_question_fail():
    text = "hello world"
    units = [unit(make_span(text, 0, 5)), unit(make_span(text, 6, 11), make_span(text, 0, 5, source_sha256=SHA_B), unit_id="u2")]
    got = anchor(units, {"doc-a": text})
    assert pairs(got) == [(an.ANCHOR_SOURCE_HASH_MISMATCH, "question.units[1].spans[1].source_sha256")]


def test_every_failing_span_is_reported_not_only_the_first():
    text = "hello world"
    bad = make_span(text, 0, 5, source_sha256=SHA_B)
    got = anchor([unit(bad, bad, bad)], {"doc-a": text})
    assert len(got) == 3 and len({i.location for i in got}) == 3


def test_a_missing_document_is_reported_per_span_and_other_documents_are_still_checked():
    a, b = "alpha text", "beta text"
    units = [unit(make_span(a, 0, 5), make_span(b, 0, 4, corpus_key="doc-b", quote="XXXX", quote_sha256=sha("XXXX")))]
    got = pairs(anchor(units, {"doc-b": b}))
    assert got == [(an.ANCHOR_TEXT_MISSING, f"{P}.corpus_key"), (an.ANCHOR_QUOTE_MISMATCH, "question.units[0].spans[1].quote")]


def test_each_document_text_is_hashed_once_per_call(monkeypatch):
    calls = []
    real = an.sha256_text

    def counting(value):
        calls.append(len(value))
        return real(value)

    monkeypatch.setattr(an, "sha256_text", counting)
    text = "hello world"
    spans = [make_span(text, i, i + 2) for i in range(0, 8)]
    # the span helper itself uses the module-level sha(); only the anchoring module is counted
    assert anchor([unit(*spans)], {"doc-a": text}) == []
    assert calls.count(len(text)) == 1


def test_no_re_anchoring_the_quote_exists_elsewhere_but_the_offsets_are_wrong():
    text = "xx hello world hello"
    wrong = s.Span("doc-a", sha(text), 0, 5, "hello", sha("hello"))   # "hello" is at 3 and 15, not 0
    assert one(wrong, {"doc-a": text}) == [(an.ANCHOR_QUOTE_MISMATCH, f"{P}.quote")]
    assert (wrong.start, wrong.end) == (0, 5)


def test_a_malformed_question_structure_is_reported_without_crashing():
    text = "hello world"
    good = make_span(text, 0, 5)
    q = s.Question("q1", "test", True, "?", "en", "human", False, units=[unit(good)])
    assert pairs(an.resolve_question(q, {"doc-a": text})) == [(an.ANCHOR_UNITS_TYPE, "question.units")]
    q = make_question("not a unit", unit(good), s.EvidenceUnit("u", [good]), unit("not a span", good, unit_id="u4"))
    assert pairs(an.resolve_question(q, {"doc-a": text})) == [
        (an.ANCHOR_UNIT_TYPE, "question.units[0]"),
        (an.ANCHOR_SPANS_TYPE, "question.units[2].spans"),
        (an.ANCHOR_SPAN_TYPE, "question.units[3].spans[0]"),
    ]


def test_the_location_prefix_is_used_verbatim():
    text = "hello world"
    q = make_question(unit(make_span(text, 0, 5, source_sha256=SHA_B)))
    assert pairs(an.resolve_question(q, {"doc-a": text}, location="q12")) == [
        (an.ANCHOR_SOURCE_HASH_MISMATCH, "q12.units[0].spans[0].source_sha256")]


def test_results_are_deterministic():
    text = "hello world"
    q = make_question(unit(make_span(text, 0, 5, source_sha256=SHA_B), make_span(text, 6, 11, quote="xxxxx", quote_sha256=SHA_B)))
    assert an.resolve_question(q, {"doc-a": text}) == an.resolve_question(q, {"doc-a": text})


# ---------- H: argument guards ----------

def test_resolve_question_rejects_wrong_argument_types():
    q = make_question()
    with pytest.raises(TypeError, match="Question"):
        an.resolve_question("q", {})
    with pytest.raises(TypeError, match="Mapping"):
        an.resolve_question(q, ["text"])
    with pytest.raises(TypeError, match="Mapping"):
        an.resolve_question(q, None)
    with pytest.raises(TypeError, match="location"):
        an.resolve_question(q, {}, location=None)
    with pytest.raises(TypeError):
        an.resolve_question(q, {}, "positional-location")


# ---------- I: information leakage ----------

def every_text(issues):
    for issue in issues:
        yield issue.code
        yield issue.location
        yield issue.message
        yield repr(issue)


def test_verify_source_text_never_leaks_text_or_metadata():
    entry = make_entry("other", extracted_text_sha256=SECRET, extracted_text_length="x" + SECRET)
    for text in (SECRET, SECRET + LONE_SURROGATE, "", SECRET.encode(), None):
        issues = an.verify_source_text(entry, text)
        assert issues
        for piece in every_text(issues):
            assert SECRET not in piece
    mismatch = an.verify_source_text(make_entry("other"), SECRET)
    assert mismatch and all(SECRET not in piece for piece in every_text(mismatch))


def test_resolve_question_never_leaks_text_quotes_or_corpus_keys():
    text = "prefix " + SECRET + " suffix"
    good_quote = SECRET
    start = text.index(SECRET)
    cases = [
        # quote mismatch: the real text and the wrong quote both contain the secret
        s.Span(SECRET_KEY, sha(text), start, start + len(SECRET), SECRET.lower(), sha(SECRET.lower())),
        # wrong source hash
        s.Span(SECRET_KEY, SHA_B, start, start + len(SECRET), good_quote, sha(good_quote)),
        # beyond the end
        s.Span(SECRET_KEY, sha(text), 100, 100 + len(SECRET), good_quote, sha(good_quote)),
        # bad structure
        s.Span(SECRET_KEY, sha(text), -1, 5, good_quote, SECRET),
        # unusable text under the key
        s.Span(SECRET_KEY, sha(text), start, start + len(SECRET), good_quote, sha(good_quote)),
    ]
    texts_variants = [{SECRET_KEY: text}, {SECRET_KEY: text}, {SECRET_KEY: text}, {SECRET_KEY: text},
                      {SECRET_KEY: SECRET.encode()}]
    for span, texts in zip(cases, texts_variants):
        issues = an.resolve_question(make_question(unit(span)), texts)
        assert issues
        for piece in every_text(issues):
            assert SECRET not in piece and SECRET.lower() not in piece and SECRET_KEY not in piece
    missing = an.resolve_question(make_question(unit(cases[0])), {})
    assert missing and all(SECRET_KEY not in piece for piece in every_text(missing))


def test_exceptions_raised_from_issues_never_contain_content():
    text = "prefix " + SECRET
    span = s.Span(SECRET_KEY, SHA_B, 0, 6, SECRET[:6], sha(SECRET[:6]))
    issues = an.resolve_question(make_question(unit(span)), {SECRET_KEY: text})
    with pytest.raises(s.EvaluationDataError) as caught:
        v.require_valid(issues)
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)
    assert SECRET_KEY not in str(caught.value) and SECRET_KEY not in repr(caught.value)


def test_each_code_has_one_fixed_message():
    seen = {}
    text = "ab" + KA_AA + "cd"
    cases = [
        make_span(text, 3, 5), make_span(text, 1, 3), make_span(text, 0, 2, source_sha256=SHA_B),
        make_span(text, 0, 2, quote="zz", quote_sha256=sha("zz")), make_span(text, 0, 2, corpus_key="doc-b"),
        s.Span("doc-a", sha(text), 50, 52, "zz", sha("zz")),
    ]
    for span in cases:
        for issue in an.resolve_question(make_question(unit(span)), {"doc-a": text}):
            seen.setdefault(issue.code, set()).add(issue.message)
    for text_arg in ("", None, "other"):
        for issue in an.verify_source_text(make_entry(text), text_arg):
            seen.setdefault(issue.code, set()).add(issue.message)
    assert seen and all(len(messages) == 1 for messages in seen.values())


# ---------- J: import isolation ----------

def test_importing_anchoring_loads_no_model_database_or_retrieval_code():
    code = (
        "import sys; import app.rag.evaluation.anchoring; "
        "names = ['torch', 'transformers', 'sqlalchemy', 'fastapi', 'numpy', 'pypdf', "
        "'app.rag.embedder', 'app.rag.retrieval', 'app.rag.indexing', 'app.core.database', 'app.main']; "
        "print(sorted(n for n in names if n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
