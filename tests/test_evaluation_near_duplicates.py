"""E1.2d step 2: exact-duplicate and near-duplicate question detection.

A  normalisation
B  exact duplicates
C  similarity (Jaccard over character 5-grams)
D  same-language restriction
E  threshold: valid values, boundaries, invalid values
F  unanswerable questions
G  locations, ordering and severity
H  defensive skipping
I  privacy
J  import isolation

Non-ASCII text is built from code points (chr) so this file stays plain ASCII.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.evaluation import near_duplicates as nd
from app.rag.evaluation import schema as s

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DUP = nd.DATASET_DUPLICATE_QUESTION_TEXT
NEAR = nd.DATASET_SIMILAR_QUESTION
HINDI_A = "".join(chr(c) for c in (0x092D, 0x093E, 0x0930, 0x0924, 0x0915, 0x0940, 0x0930, 0x093E, 0x091C, 0x0927, 0x093E, 0x0928, 0x0940))
HINDI_B = HINDI_A + chr(0x0964)
CANARY = "SECRET-CANARY-7731"


def q(text, *, split="dev", language="en", qid="q1", answerable=True):
    base = dict(
        question_id=qid,
        split=split,
        answerable=answerable,
        question=text,
        query_language=language,
        author_type="human",
        hindi_reviewed=False,
    )
    if not answerable:
        base.update(unanswerable_type="off_topic")
    return s.Question(**base)


def run(questions, threshold=0.5):
    return nd.find_duplicate_questions(questions, threshold=threshold)


def pair(issues):
    return [(i.code, i.location) for i in issues]


def test_issue_codes_are_the_approved_names():
    assert nd.DATASET_DUPLICATE_QUESTION_TEXT == "DATASET_DUPLICATE_QUESTION_TEXT"
    assert nd.DATASET_SIMILAR_QUESTION == "DATASET_SIMILAR_QUESTION"


# ---------- A. normalisation ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param("  Hello   World  ", "hello world", id="whitespace-collapsed-and-trimmed"),
        pytest.param("Hello\tWorld\n", "hello world", id="tab-and-newline"),
        pytest.param("HELLO", "hello", id="case-folded"),
        pytest.param("Stra" + chr(0xDF) + "e", "strasse", id="casefold-not-lower"),
        pytest.param("a" + chr(0x3000) + "b", "a b", id="ideographic-space"),
        pytest.param("   ", "", id="blank"),
    ],
)
def test_normalisation(raw, expected):
    assert nd.normalize_question_text(raw) == expected


# ---------- B. exact duplicates ----------

def test_exact_duplicate_is_an_error_at_the_later_question():
    issues = run([q("What is the notice period?", qid="a"), q("what is the NOTICE   period?", qid="b")])
    assert pair(issues) == [(DUP, "dev.questions[1].question")]
    assert issues[0].severity == s.SEVERITY_ERROR


def test_three_identical_questions_report_the_second_and_third():
    issues = run([q("same question here")] * 3, threshold=1)
    assert pair(issues) == [(DUP, "dev.questions[1].question"), (DUP, "dev.questions[2].question")]


def test_distinct_questions_are_not_duplicates():
    assert run([q("what is the notice period?"), q("how long is the lease term?")]) == []


@pytest.mark.parametrize(
    "first, second, location",
    [
        pytest.param(("dev", "en"), ("test", "en"), "test.questions[0].question", id="across-splits"),
        pytest.param(("dev", "en"), ("dev", "hi"), "dev.questions[1].question", id="across-languages-same-split"),
        pytest.param(("dev", "en"), ("test", "hi"), "test.questions[0].question", id="across-splits-and-languages"),
    ],
)
def test_duplicates_are_found_across_splits_and_languages(first, second, location):
    issues = run(
        [q("same text for both", split=first[0], language=first[1]), q("Same Text For Both", split=second[0], language=second[1])]
    )
    assert pair(issues) == [(DUP, location)]


def test_dev_is_processed_before_test_whatever_the_input_order():
    issues = run([q("repeat me please", split="test"), q("repeat me please", split="dev")])
    assert pair(issues) == [(DUP, "test.questions[0].question")]


def test_blank_questions_are_skipped():
    assert run([q("   "), q(""), q("\t\n")], threshold=0) == []


# ---------- C. similarity ----------

def test_jaccard_over_character_five_grams():
    # "abcdefg" -> abcde bcdef cdefg ; "abcdefgh" -> the same three plus defgh
    assert nd.character_ngram_jaccard("abcdefg", "abcdefgh") == pytest.approx(3 / 4)
    assert nd.character_ngram_jaccard("abcde", "abcde") == 1.0
    assert nd.character_ngram_jaccard("abcde", "vwxyz") == 0.0


def test_texts_under_five_characters_have_similarity_zero():
    assert nd.character_ngram_jaccard("abcd", "abcd") == 0.0
    assert nd.character_ngram_jaccard("abcd", "abcdefgh") == 0.0
    assert nd.character_ngram_jaccard("", "abcdefgh") == 0.0


def test_similar_pair_is_a_warning_at_the_later_question():
    issues = run([q("abcdefg", qid="a"), q("abcdefgh", qid="b")], threshold=0.75)
    assert pair(issues) == [(NEAR, "dev.questions[1].question~dev.questions[0].question")]
    assert issues[0].severity == s.SEVERITY_WARNING


def test_a_pair_below_the_threshold_is_not_reported():
    assert run([q("abcdefg"), q("abcdefgh")], threshold=0.76) == []


def test_similarity_is_computed_on_normalised_text():
    issues = run([q("ABCDEFG  "), q(" abcdefgh")], threshold=0.75)
    assert pair(issues) == [(NEAR, "dev.questions[1].question~dev.questions[0].question")]


def test_similar_pairs_are_found_across_splits():
    issues = run([q("abcdefg", split="dev"), q("abcdefgh", split="test")], threshold=0.75)
    assert pair(issues) == [(NEAR, "test.questions[0].question~dev.questions[0].question")]


def test_hindi_text_is_compared_by_code_point():
    issues = run([q(HINDI_A, language="hi"), q(HINDI_B, language="hi")], threshold=0.5)
    assert pair(issues) == [(NEAR, "dev.questions[1].question~dev.questions[0].question")]


def test_exact_duplicates_are_not_also_reported_as_similar():
    issues = run([q("exactly the same words"), q("Exactly  the same words")], threshold=0)
    assert pair(issues) == [(DUP, "dev.questions[1].question")]


def test_each_similar_pair_is_reported_once_at_the_later_question():
    issues = run([q("abcdefg"), q("abcdefgh"), q("abcdefghi")], threshold=0.5)
    assert pair(issues) == [
        (NEAR, "dev.questions[1].question~dev.questions[0].question"),
        (NEAR, "dev.questions[2].question~dev.questions[0].question"),
        (NEAR, "dev.questions[2].question~dev.questions[1].question"),
    ]


# ---------- D. same-language restriction ----------

def test_similar_questions_in_different_languages_are_not_compared():
    assert run([q("abcdefg", language="en"), q("abcdefgh", language="hi")], threshold=0) == []


def test_an_exact_duplicate_across_languages_is_still_an_error():
    issues = run([q("abcdefg", language="en"), q("abcdefg", language="hi")])
    assert pair(issues) == [(DUP, "dev.questions[1].question")]


# ---------- E. threshold ----------

@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0, id="int-zero"),
        pytest.param(1, id="int-one"),
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param(0.5, id="float-half"),
    ],
)
def test_valid_thresholds_are_accepted(threshold):
    assert isinstance(run([q("abcdefg"), q("vwxyzzz")], threshold=threshold), list)


def test_a_pair_exactly_at_the_threshold_is_reported():
    assert len(run([q("abcdefg"), q("abcdefgh")], threshold=0.75)) == 1


def test_threshold_zero_reports_every_same_language_pair_of_nonblank_texts():
    assert len(run([q("abcdefg"), q("vwxyzzz"), q("ab")], threshold=0)) == 3


def test_threshold_one_reports_nothing_unless_the_gram_sets_are_identical():
    assert run([q("abcdefg"), q("abcdefgh")], threshold=1) == []
    # same 5-gram set, different text: "aaaaaa" and "aaaaaaa" share the one gram "aaaaa"
    assert len(run([q("aaaaaa"), q("aaaaaaa")], threshold=1)) == 1


def test_threshold_is_required_and_keyword_only():
    with pytest.raises(TypeError):
        nd.find_duplicate_questions([q("abcdefg")])
    with pytest.raises(TypeError):
        nd.find_duplicate_questions([q("abcdefg")], 0.5)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param("0.5", id="string"),
        pytest.param(None, id="none"),
        pytest.param([0.5], id="list"),
        pytest.param(0.5 + 0j, id="complex"),
    ],
)
def test_a_non_numeric_threshold_is_a_type_error(bad):
    with pytest.raises(TypeError, match="^threshold must be a number$"):
        run([], threshold=bad)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(-0.1, id="below-zero"),
        pytest.param(1.1, id="above-one"),
        pytest.param(2, id="int-above-one"),
        pytest.param(-1, id="int-below-zero"),
        pytest.param(10**400, id="huge-int"),
    ],
)
def test_an_out_of_range_threshold_is_a_value_error(bad):
    with pytest.raises(ValueError, match="^threshold must be a finite number from 0 to 1$"):
        run([], threshold=bad)


def test_questions_must_not_be_a_string():
    with pytest.raises(TypeError, match="^questions must be a sequence of Question$"):
        run("abcdefg")


# ---------- F. unanswerable questions ----------

def test_unanswerable_questions_are_checked_like_answerable_ones():
    issues = run([q("abcdefg", answerable=False), q("ABCDEFG", split="test", answerable=True)])
    assert pair(issues) == [(DUP, "test.questions[0].question")]
    issues = run([q("abcdefg"), q("abcdefgh", answerable=False)], threshold=0.75)
    assert pair(issues) == [(NEAR, "dev.questions[1].question~dev.questions[0].question")]


# ---------- G. locations, ordering, severity ----------

def test_locations_use_positions_within_each_split():
    questions = [q("abcdefgh", split="test"), q("abcdefg", split="dev")]
    # dev is processed first, so the test question is the later one
    assert pair(run(questions, threshold=0.75)) == [(NEAR, "test.questions[0].question~dev.questions[0].question")]


def test_issues_follow_processing_order_and_are_deterministic():
    questions = [
        q("abcdefgh", split="test"),
        q("zzzzzzzz", split="dev"),
        q("abcdefg", split="dev"),
        q("zzzzzzzz", split="test"),
    ]
    first = run(questions, threshold=0.5)
    assert pair(first) == [
        (NEAR, "test.questions[0].question~dev.questions[1].question"),
        (DUP, "test.questions[1].question"),
    ]
    assert run(questions, threshold=0.5) == first


def test_exact_duplicates_are_errors_and_similar_pairs_are_warnings():
    issues = run([q("abcdefg"), q("abcdefgh"), q("abcdefg")], threshold=0.75)
    severities = {i.code: i.severity for i in issues}
    assert severities == {DUP: s.SEVERITY_ERROR, NEAR: s.SEVERITY_WARNING}


def test_warnings_alone_do_not_fail_require_valid():
    from app.rag.evaluation import validation as v

    v.require_valid(run([q("abcdefg"), q("abcdefgh")], threshold=0.75))


def test_positions_count_each_split_separately_in_input_order():
    questions = [
        q("alpha question one"),
        q("beta question two", split="test"),
        q("gamma question three"),
        q("gamma question three", split="test"),
        q("beta question two"),
    ]
    assert pair(run(questions, threshold=1)) == [
        (DUP, "test.questions[0].question"),
        (DUP, "test.questions[1].question"),
    ]


# ---------- H. defensive skipping ----------

@pytest.mark.parametrize(
    "bad, position",
    [
        pytest.param(q(None), 2, id="text-none"),
        pytest.param(q(5), 2, id="text-int"),
        pytest.param(q("abcdefg", language=None), 2, id="language-none"),
        pytest.param(q("abcdefg", language=5), 2, id="language-int"),
        pytest.param(q("abcdefg", split="other"), 1, id="split-not-dev-or-test"),
        pytest.param("not a question", 1, id="not-a-question"),
    ],
)
def test_unusable_items_are_skipped_without_error(bad, position):
    # a skipped Question still takes a position in its split; an item that is not a Question does not
    assert run([q("abcdefg"), bad, q("abcdefg")], threshold=0) == [
        s.Issue(DUP, f"dev.questions[{position}].question", nd._MESSAGES[DUP])
    ]


def test_empty_input_gives_no_issues():
    assert run([]) == []
    assert run(()) == []


def test_the_questions_are_not_modified():
    questions = [q("  ABC  defg "), q("abc defgh")]
    before = list(questions)
    run(questions, threshold=0)
    assert questions == before


# ---------- I. privacy ----------

def test_issues_never_contain_question_text_or_identifiers():
    questions = [
        q(CANARY + " first", qid="id-canary-a"),
        q(CANARY + " first", qid="id-canary-b"),
        q(CANARY + " firsts", qid="id-canary-c"),
    ]
    issues = run(questions, threshold=0.5)
    assert {i.code for i in issues} == {DUP, NEAR}
    for issue in issues:
        text = repr(issue)
        assert "CANARY" not in text and "canary" not in text and "first" not in text


def test_messages_are_fixed():
    issues = run([q("abcdefg"), q("abcdefgh"), q("abcdefg")], threshold=0.75)
    assert {i.message for i in issues} == set(nd._MESSAGES.values())


# ---------- J. import isolation ----------

def test_importing_the_module_loads_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import app.rag.evaluation.near_duplicates\n"
        "heavy = ('sqlalchemy', 'torch', 'transformers', 'fastapi', 'sentence_transformers', 'numpy', 'pypdf')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
