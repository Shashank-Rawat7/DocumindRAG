"""Tests for app/rag/chat_citations.py (C9.2): pure functions, no database, no network."""

import dataclasses
import random
import re

import pytest

from app.rag.chat_citations import (
    Citation,
    CitationInputError,
    CitationResult,
    resolve_citations,
)
from app.rag.chat_prompt import PromptSource, build_prompt
from app.rag.retrieval import RetrievedChunk


def make_chunk(
    chunk_id: int, text: str = "some text", *, score: float = 0.5
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id + 1000,
        chunk_index=chunk_id,
        start_offset=0,
        end_offset=len(text),
        page_start=None,
        page_end=None,
        text=text,
        span_sha256="0" * 64,
        score=score,
        source_version=1,
        source_sha256="1" * 64,
    )


def sources_for(count: int, *, first_id: int = 101) -> tuple[PromptSource, ...]:
    return tuple(
        PromptSource(marker=i + 1, chunk=make_chunk(first_id + i)) for i in range(count)
    )


S3 = sources_for(3)


def stripped(answer: str, sources=S3) -> str:
    return resolve_citations(answer, sources).text


class TestMappingToExactChunks:
    def test_markers_resolve_to_the_very_chunk_objects_rendered(self):
        result = resolve_citations("a [2] b [1] c [3]", S3)
        assert [c.marker for c in result.citations] == [2, 1, 3]
        assert result.citations[0].chunk is S3[1].chunk
        assert result.citations[1].chunk is S3[0].chunk
        assert result.citations[2].chunk is S3[2].chunk

    def test_round_trip_through_build_prompt_with_a_budget_dropped_chunk(self):
        retrieved = [make_chunk(i, "x" * 1500) for i in range(1, 6)]
        built = build_prompt("q?", retrieved, token_budget=1500)
        kept = len(built.sources)
        assert 0 < kept < len(retrieved)  # some retrieved chunks were dropped
        dropped_marker = kept + 1  # a number a model could invent for a dropped chunk
        answer = (
            " ".join(f"c [{n}]" for n in range(1, kept + 1)) + f" d [{dropped_marker}]."
        )
        result = resolve_citations(answer, built.sources)
        assert [c.marker for c in result.citations] == list(range(1, kept + 1))
        for citation in result.citations:
            assert citation.chunk is retrieved[citation.marker - 1]
        assert result.invalid_marker_count == 1
        assert f"[{dropped_marker}]" not in result.text
        cited_ids = {c.chunk.chunk_id for c in result.citations}
        assert not cited_ids & set(built.dropped_chunk_ids)

    def test_mapping_follows_marker_position_not_chunk_ids_or_scores(self):
        a = make_chunk(900, score=0.1)
        b = make_chunk(5, score=0.99)
        sources = (PromptSource(1, a), PromptSource(2, b))
        result = resolve_citations("x [1] y [2]", sources)
        assert [c.chunk.chunk_id for c in result.citations] == [900, 5]
        assert result.citations[0].chunk is a
        assert result.citations[1].chunk is b

    def test_value_equal_chunks_keep_their_own_identity(self):
        first = make_chunk(1)
        twin = dataclasses.replace(first, chunk_id=2)
        sources = (PromptSource(1, first), PromptSource(2, twin))
        result = resolve_citations("[2] then [1]", sources)
        assert result.citations[0].chunk is twin
        assert result.citations[1].chunk is first

    def test_build_prompt_sources_feed_straight_in(self):
        chunks = [make_chunk(7, "alpha"), make_chunk(8, "beta")]
        built = build_prompt("q", chunks)
        result = resolve_citations("see [1] and [2].", built.sources)
        assert [c.chunk for c in result.citations] == chunks
        assert result.citations[0].chunk is chunks[0]


class TestSourceValidation:
    def test_no_sources_means_every_marker_is_invalid(self):
        result = resolve_citations("a [1] b [2].", ())
        assert result.text == "a b."
        assert result.citations == ()
        assert result.invalid_marker_count == 2

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "abc",
            [object()],
            [PromptSource(2, make_chunk(1))],
            [PromptSource(0, make_chunk(1))],
            [PromptSource(1, make_chunk(1)), PromptSource(3, make_chunk(2))],
            [PromptSource(2, make_chunk(2)), PromptSource(1, make_chunk(1))],
            [PromptSource(1, object())],
            [PromptSource(True, make_chunk(1))],
        ],
    )
    def test_malformed_sources_are_rejected(self, bad):
        with pytest.raises(CitationInputError):
            resolve_citations("a [1]", bad)

    @pytest.mark.parametrize("bad", [None, b"a [1]", 5, ["a"]])
    def test_answer_must_be_a_str(self, bad):
        with pytest.raises(CitationInputError):
            resolve_citations(bad, S3)

    def test_error_messages_never_contain_answer_text(self):
        secret = "SECRET-ANSWER-TEXT"
        with pytest.raises(CitationInputError) as info:
            resolve_citations(secret, [object()])
        assert secret not in str(info.value)

    def test_results_are_frozen(self):
        result = resolve_citations("a [1]", S3)
        assert isinstance(result, CitationResult)
        assert isinstance(result.citations[0], Citation)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.text = "x"  # type: ignore[misc]


class TestValidMarkers:
    def test_ordinals_follow_first_appearance_and_duplicates_collapse(self):
        result = resolve_citations("b [2] then a [1] and [2] again [1][3]", S3)
        assert [(c.ordinal, c.marker) for c in result.citations] == [
            (1, 2),
            (2, 1),
            (3, 3),
        ]
        assert result.invalid_marker_count == 0

    def test_valid_markers_are_byte_identical(self):
        answer = "x  [1]\t[2]\r\n[3] y [1]."
        assert stripped(answer) == answer

    def test_no_markers(self):
        result = resolve_citations("There is nothing to cite.", S3)
        assert result.citations == ()
        assert result.text == "There is nothing to cite."
        assert result.invalid_marker_count == 0

    def test_valid_markers_beside_unicode_text_are_untouched(self):
        answer = "नमस्ते [1] 😀 और [2] 中文[3]。"
        assert stripped(answer) == answer


class TestInvalidMarkerRemoval:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("claim [9].", "claim."),
            ("claim [9], more", "claim, more"),
            ("claim [8][9].", "claim."),
            ("claim [8] [9].", "claim."),
            ("claim [8]\t[9] more", "claim more"),
            ("claim [8][9][7] more", "claim more"),
            ("a [9] b", "a b"),
            ("a [9]\nb", "a\nb"),
            ("a [9]\r\nb", "a\r\nb"),
            ("a [9]b", "a b"),
            ("a[9]b", "ab"),
            ("a[9] b", "a b"),
            ("(see [9])", "(see)"),
            ("[9] start", "start"),
            ("    [9] indented", "    indented"),
            ("\t[9]text", "\ttext"),
            ("- [9] item", "- item"),
            ("| a [9] | b |", "| a | b |"),
            ("x [9][1]", "x [1]"),
            ("x [1] [9].", "x [1]."),
            ("x [9] [1].", "x [1]."),
            ("x [1][9]", "x [1]"),
            ("x [9]", "x"),
            ("x [9]  \nnext", "x  \nnext"),
            ("[9]", ""),
            ("  [9]  ", "  "),
            ("दावा [9]।", "दावा।"),
            ('He said "yes [9]".', 'He said "yes".'),
            ("see [9]: text", "see: text"),
            ("两个[9]。", "两个。"),
        ],
    )
    def test_cases(self, answer, expected):
        assert stripped(answer) == expected

    @pytest.mark.parametrize(
        "marker", ["[0]", "[00]", "[007]", "[01]", "[4]", "[10]", "[99]", "[999]"]
    )
    def test_invalid_markers_are_removed(self, marker):
        result = resolve_citations(f"claim {marker}.", S3)
        assert result.text == "claim."
        assert result.invalid_marker_count == 1
        assert result.citations == ()

    @pytest.mark.parametrize(
        "text",
        [
            "in [2020] we",
            "id [1000] here",
            "n [12345]",
            "see [a] and [ 1 ] and [1 ]",
            "pair [1, 2] and range [1-3]",
            "arabic [\u0663] and fullwidth [\uff11]",
            "empty [] brackets",
        ],
    )
    def test_non_markers_are_left_exactly_alone(self, text):
        result = resolve_citations(text, ())
        assert result.text == text
        assert result.invalid_marker_count == 0

    def test_four_digit_numbers_survive_next_to_real_markers(self):
        result = resolve_citations("in [2020] [1] and [9] and [1999].", S3)
        assert result.text == "in [2020] [1] and and [1999]."
        assert [c.marker for c in result.citations] == [1]
        assert result.invalid_marker_count == 1

    def test_leading_zero_form_of_a_valid_number_is_invalid(self):
        result = resolve_citations("a [01] b [1]", S3)
        assert result.text == "a b [1]"
        assert result.invalid_marker_count == 1

    def test_counts_every_invalid_occurrence(self):
        assert resolve_citations("a [8][9] [7] b [9]", S3).invalid_marker_count == 4

    def test_unrelated_whitespace_and_formatting_are_preserved(self):
        template = (
            "Line one  with  double spaces.\r\n"
            "\tTabbed line{m}.\n"
            "\n"
            "    indented   code-ish\n"
            "- item  one\n"
            "| a  | b  |\n"
            "trailing spaces   \n"
        )
        assert stripped(template.format(m=" [9]")) == template.format(m="")
        assert stripped(template.format(m=" [9] [8]")) == template.format(m="")

    def test_nothing_to_remove_returns_the_text_unchanged(self):
        answer = "  keep   all\tthis \n"
        assert stripped(answer) == answer

    def test_result_is_not_stripped_and_may_be_empty(self):
        assert stripped("[9]") == ""
        assert stripped("\n\n[9]\n") == "\n\n\n"

    def test_removal_is_idempotent(self):
        first = resolve_citations("a [9] b [1] c [8][7]. d[2]", S3)
        second = resolve_citations(first.text, S3)
        assert second.text == first.text
        assert second.invalid_marker_count == 0
        assert second.citations == first.citations


class TestMarkdownEmphasis:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("**bold [9]**", "**bold**"),
            ("*italic [9]*", "*italic*"),
            ("~~strike [9]~~", "~~strike~~"),
            ("__under [9]__", "__under__"),
            ("***both [9]***", "***both***"),
            ("**bold [9]** more", "**bold** more"),
            ("*italic [9]*.", "*italic*."),
            ("**bold [9].**", "**bold.**"),
            ("**bold [8][9]**", "**bold**"),
            ("**bold [8] [9]** end", "**bold** end"),
            ("a **bold [9]**\nnext", "a **bold**\nnext"),
            ("(*italic [9]*)", "(*italic*)"),
        ],
    )
    def test_space_before_a_closing_emphasis_run_goes_with_the_marker(
        self, answer, expected
    ):
        assert stripped(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("claim [9]*italic*", "claim *italic*"),
            ("claim [9]**bold** x", "claim **bold** x"),
            ("claim [9]~~gone~~", "claim ~~gone~~"),
            ("claim [9]_it_", "claim _it_"),
        ],
    )
    def test_space_before_an_opening_emphasis_run_stays(self, answer, expected):
        assert stripped(answer) == expected

    def test_valid_markers_inside_emphasis_are_untouched(self):
        for answer in (
            "**bold [1]**",
            "*italic [2]*",
            "~~strike [3]~~",
            "**a [1] b [9]**",
        ):
            result = resolve_citations(answer, S3)
            assert result.text == answer.replace(" [9]", "")

    def test_emphasis_at_end_of_text_and_with_unicode(self):
        assert stripped("**दावा [9]**") == "**दावा**"
        assert stripped("*x [9]*") == "*x*"


class TestMarkdownCode:
    def test_markers_inside_inline_code_are_literal(self):
        answer = "use `arr[1]` and `x[9]` here [9]."
        result = resolve_citations(answer, S3)
        assert result.text == "use `arr[1]` and `x[9]` here."
        assert result.citations == ()
        assert result.invalid_marker_count == 1

    def test_valid_looking_marker_in_code_is_not_a_citation(self):
        result = resolve_citations("see `[1]` only", S3)
        assert result.text == "see `[1]` only"
        assert result.citations == ()

    def test_fenced_blocks_are_untouched(self):
        answer = "```\ndata[9]\n[1]\n```\nsee [1] and [9]."
        result = resolve_citations(answer, S3)
        assert result.text == "```\ndata[9]\n[1]\n```\nsee [1] and."
        assert [c.marker for c in result.citations] == [1]
        assert result.invalid_marker_count == 1

    def test_longer_fence_contains_a_shorter_one(self):
        answer = "````\n```\n[9]\n```\n````\n[9]"
        assert stripped(answer) == "````\n```\n[9]\n```\n````\n"

    def test_double_backtick_span_may_contain_a_single_backtick(self):
        answer = "``a ` [9]`` then [9]"
        assert stripped(answer) == "``a ` [9]`` then"

    def test_unmatched_backtick_is_literal_and_protects_nothing(self):
        assert stripped("a ` [9] b") == "a ` b"
        assert stripped("```\n[9]\nno closing fence") == "```\n\nno closing fence"

    def test_marker_next_to_code(self):
        assert stripped("`x` [9].") == "`x`."
        assert stripped("[9] `x`") == "`x`"
        assert stripped("`a`[9]`b`") == "`a``b`"

    def test_code_after_a_removed_marker_stays_intact(self):
        assert stripped("go [9] `f[8]()` now") == "go `f[8]()` now"


class TestFuzzInvariants:
    """Random answers checked against a small independent oracle.

    Pieces that contain backticks are always surrounded by separators, so code spans are
    unambiguous and a simple regex can serve as the oracle.
    """

    PIECES = (
        "word",
        "ünï",
        "नमस्ते",
        "中文",
        "😀",
        "text.",
        "a,",
        "(x)",
        '"q"',
        "-",
        "|",
        "*",
        "`code[9]`",
        "`[1]`",
        "```\nfence[8]\n```",
    )
    SEPARATORS = (" ", "  ", "\t", "\n", "\r\n")
    MARKERS = (
        "[1]",
        "[2]",
        "[3]",
        "[4]",
        "[9]",
        "[0]",
        "[01]",
        "[10]",
        "[2020]",
        "[1, 2]",
    )
    MARKER = re.compile(r"\[([0-9]{1,3})\]")
    CODE = re.compile(r"(```.*?```|`[^`\n]*`)", flags=re.DOTALL)

    def build(self, rng: random.Random) -> str:
        parts = [rng.choice(self.SEPARATORS)]
        for _ in range(rng.randint(1, 14)):
            piece = (
                rng.choice(self.PIECES)
                if rng.random() < 0.55
                else rng.choice(self.MARKERS)
            )
            parts.append(piece)
            parts.append(rng.choice(self.SEPARATORS) if rng.random() < 0.8 else "")
            if "`" in piece and not parts[-1]:
                parts[-1] = " "
        text = "".join(parts)
        return re.sub(r"(?<=[^\s`])(?=`)", " ", text)

    def is_valid(self, raw: str) -> bool:
        return raw == str(int(raw)) and int(raw) in (1, 2, 3)

    def oracle_without_invalid_markers(self, answer: str) -> str:
        pieces = self.CODE.split(answer)  # odd indexes are code spans
        for index in range(0, len(pieces), 2):
            pieces[index] = self.MARKER.sub(
                lambda m: m.group(0) if self.is_valid(m.group(1)) else "", pieces[index]
            )
        return "".join(pieces)

    def oracle_citations(self, answer: str) -> list[int]:
        seen: list[int] = []
        for index, piece in enumerate(self.CODE.split(answer)):
            if index % 2:
                continue
            for m in self.MARKER.finditer(piece):
                if self.is_valid(m.group(1)) and int(m.group(1)) not in seen:
                    seen.append(int(m.group(1)))
        return seen

    def test_invariants_over_random_answers(self):
        rng = random.Random(12)

        def core(value: str) -> str:
            return re.sub(r"[ \t]", "", value)

        for _ in range(3000):
            answer = self.build(rng)
            result = resolve_citations(answer, S3)

            # Nothing but invalid markers and horizontal whitespace may change.
            assert core(result.text) == core(
                self.oracle_without_invalid_markers(answer)
            )
            assert result.text.count("\n") == answer.count("\n")
            assert result.text.count("\r") == answer.count("\r")

            # Code spans survive byte-for-byte and in order.
            assert self.CODE.findall(result.text) == self.CODE.findall(answer)

            # Valid markers survive in full; citations mirror first appearances.
            for token in ("[1]", "[2]", "[3]"):
                assert result.text.count(token) == answer.count(token)
            assert [c.marker for c in result.citations] == self.oracle_citations(answer)

            # Idempotent.
            again = resolve_citations(result.text, S3)
            assert again.text == result.text
            assert again.invalid_marker_count == 0
