"""Tests for app/rag/chat_idempotency.py (C8.1 step 1): pure, no database.

The two known-hash constants were computed outside the code under test, with
``printf '%s' '<canonical json>' | sha256sum`` on the hand-written canonical strings
quoted next to them.
"""

import ast
import collections
import datetime
import decimal
import enum
import hashlib
import json
import math
import pathlib
import re
import sys

import pytest

from app.rag import chat_idempotency
from app.rag.chat_idempotency import (
    IDEMPOTENCY_KEY_PATTERN,
    CanonicalRequestError,
    ChatIdempotencyError,
    IdempotencyKeyError,
    canonical_request_sha256,
    validate_idempotency_key,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Canonical JSON: {"conversation_id":null,"operation":"chat.create","options":{"top_k":5},
#                  "question":"What is MVCC in PostgreSQL?"}
ASCII_HASH = "9590508f5ad1b250da4a78c6f4596f9338311d4eaa7cf63b768872964b597d54"
# Canonical JSON (raw UTF-8, 146 bytes): {"conversation_id":42,"operation":"chat.create",
#   "options":{"language":"hi","top_k":5},"question":"PostgreSQL में MVCC क्या है?"}
HINDI_HASH = "1ba5dd90ecc2552c0a3a6b71b34b17e9e91f7cd65311363b2f596fa0bc46fc0d"

HINDI_QUESTION = "PostgreSQL में MVCC क्या है?"


class _StrSub(str):
    pass


class _IntSub(int):
    pass


class _DictSub(dict):
    pass


class _Color(enum.IntEnum):
    RED = 1


def h(operation="chat.create", conversation_id=None, question="q", options=None):
    return canonical_request_sha256(
        operation, conversation_id, question, {} if options is None else options
    )


# ---------------------------------------------------------------- idempotency key


@pytest.mark.parametrize(
    "key",
    [
        "a",
        "Z",
        "0",
        ".",
        "_",
        ":",
        "-",
        "a" * 128,
        "Abc-123_x.y:z",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-",
        "550e8400-e29b-41d4-a716-446655440000",
    ],
)
def test_valid_keys_are_returned_unchanged(key):
    assert validate_idempotency_key(key) is key


def test_shortest_and_longest_valid_keys():
    assert validate_idempotency_key("k") == "k"
    assert validate_idempotency_key("k" * 128) == "k" * 128


@pytest.mark.parametrize(
    "key",
    [
        "",
        "a" * 129,
        "a" * 1000,
        " ",
        "a b",
        " a",
        "a ",
        "\ta",
        "a\t",
        "a\n",
        "\na",
        "a\r\n",
        "a\x00",
        "a/b",
        "a@b",
        "a#b",
        "a+b",
        "a=b",
        "a,b",
        "\N{LATIN SMALL LETTER E WITH ACUTE}",
        "आ",
        "😀",
        "\N{FULLWIDTH LATIN SMALL LETTER A}",  # fullwidth a
        "key\N{ZERO WIDTH SPACE}",
        "key\N{NO-BREAK SPACE}",
    ],
)
def test_invalid_keys_are_rejected(key):
    with pytest.raises(IdempotencyKeyError):
        validate_idempotency_key(key)


@pytest.mark.parametrize(
    "key",
    [
        None,
        1,
        1.5,
        True,
        b"abc",
        bytearray(b"abc"),
        ["a"],
        ("a",),
        {"a": 1},
        object(),
        _StrSub("abc"),
    ],
)
def test_non_string_keys_are_rejected(key):
    with pytest.raises(IdempotencyKeyError):
        validate_idempotency_key(key)


def test_key_is_never_stripped_or_normalized():
    for padded in (" abc", "abc ", "abc\n"):
        with pytest.raises(IdempotencyKeyError):
            validate_idempotency_key(padded)
    assert validate_idempotency_key("ABC") == "ABC"  # case is preserved


def test_key_pattern_agrees_with_the_c7_database_check():
    expected = "^" + IDEMPOTENCY_KEY_PATTERN + "$"
    sources = [
        ROOT / "app" / "models" / "chat_request.py",
        ROOT
        / "migrations"
        / "versions"
        / "e9749572298d_add_chat_requests_and_attempts.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        found = re.findall(r"idempotency_key ~ '([^']+)'", text)
        assert found == [expected], path.name


def test_key_error_message_does_not_echo_the_key():
    secret = "SECRET KEY with spaces"
    with pytest.raises(IdempotencyKeyError) as excinfo:
        validate_idempotency_key(secret)
    assert "SECRET" not in str(excinfo.value)


# ---------------------------------------------------------------- known hashes


def test_known_hash_ascii():
    assert (
        h("chat.create", None, "What is MVCC in PostgreSQL?", {"top_k": 5})
        == ASCII_HASH
    )


def test_known_hash_hindi_question_is_hashed_as_raw_utf8():
    result = canonical_request_sha256(
        "chat.create", 42, HINDI_QUESTION, {"top_k": 5, "language": "hi"}
    )
    assert result == HINDI_HASH
    # ensure_ascii=True would escape the Hindi text and give a different digest.
    escaped = json.dumps(
        {
            "operation": "chat.create",
            "conversation_id": 42,
            "question": HINDI_QUESTION,
            "options": {"language": "hi", "top_k": 5},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert hashlib.sha256(escaped.encode("utf-8")).hexdigest() != HINDI_HASH


def test_digest_is_lowercase_64_hex_and_deterministic():
    first = h(question="x")
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert h(question="x") == first


# ---------------------------------------------------------------- Unicode and text exactness


@pytest.mark.parametrize(
    "composed, decomposed",
    [
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
            "cafe\N{COMBINING ACUTE ACCENT}",
        ),  # NFC vs NFD
        (
            "\N{DEVANAGARI LETTER QA}",
            "\N{DEVANAGARI LETTER KA}\N{DEVANAGARI SIGN NUKTA}",
        ),  # composition-excluded: QA vs KA + NUKTA
        (
            "\N{LATIN CAPITAL LETTER A WITH RING ABOVE}",
            "A\N{COMBINING RING ABOVE}",
        ),  # A with ring above
    ],
)
def test_unicode_normalization_is_not_applied(composed, decomposed):
    assert composed != decomposed
    assert h(question=composed) != h(question=decomposed)


@pytest.mark.parametrize(
    "variant",
    [
        " q",
        "q ",
        "Q",
        "q\n",
        "q\r\n",
        "q\t",
        "\N{NO-BREAK SPACE}q",
        "q\N{ZERO WIDTH SPACE}",
    ],
)
def test_question_is_not_trimmed_or_rewritten(variant):
    assert h(question=variant) != h(question="q")


def test_empty_question_is_accepted_and_distinct_from_whitespace():
    assert h(question="") != h(question=" ")
    assert re.fullmatch(r"[0-9a-f]{64}", h(question=""))


# ---------------------------------------------------------------- ordering and field sensitivity


def test_top_level_and_nested_option_order_does_not_matter():
    first = {"a": 1, "b": {"x": 1, "y": [1, 2], "z": {"p": None, "q": "w"}}, "c": "s"}
    second = {"c": "s", "b": {"z": {"q": "w", "p": None}, "y": [1, 2], "x": 1}, "a": 1}
    assert list(first) != list(second)
    assert h(options=first) == h(options=second)


def test_list_order_inside_options_does_matter():
    assert h(options={"ids": [1, 2]}) != h(options={"ids": [2, 1]})


@pytest.mark.parametrize(
    "changed",
    [
        {"operation": "chat.other"},
        {"conversation_id": 1},
        {"question": "different"},
        {"options": {"top_k": 6}},
    ],
)
def test_changing_any_top_level_field_changes_the_hash(changed):
    base = {
        "operation": "chat.create",
        "conversation_id": None,
        "question": "q",
        "options": {"top_k": 5},
    }
    assert canonical_request_sha256(**base) != canonical_request_sha256(
        **{**base, **changed}
    )


def test_changing_a_nested_option_changes_the_hash():
    assert h(options={"a": {"b": 1}}) != h(options={"a": {"b": 2}})


def test_none_and_integer_conversation_ids_differ():
    assert h(conversation_id=None) != h(conversation_id=0)
    assert h(conversation_id=None) != h(conversation_id=1)
    assert h(conversation_id=1) != h(conversation_id=2)


def test_omitted_and_explicit_defaults_hash_alike_only_when_the_caller_resolves_them():
    defaults = {"top_k": 5}

    def effective(supplied):
        return {**defaults, **supplied}

    assert h(options=effective({})) == h(options=effective({"top_k": 5}))
    # The function itself fills in nothing: an unresolved option set is a different request.
    assert h(options={}) != h(options={"top_k": 5})


def test_python_types_stay_distinct_where_json_text_differs():
    assert h(options={"v": 1}) != h(options={"v": 1.0})
    assert h(options={"v": True}) != h(options={"v": 1})
    assert h(options={"v": None}) != h(options={"v": 0})


def test_tuple_and_list_hash_alike():
    assert h(options={"ids": (1, 2)}) == h(options={"ids": [1, 2]})


def test_all_supported_value_kinds_are_accepted():
    options = {
        "n": None,
        "t": True,
        "f": False,
        "i": -3,
        "x": 2.5,
        "s": "été",
        "l": [1, [2]],
        "d": {"k": {}},
    }
    assert re.fullmatch(r"[0-9a-f]{64}", h(options=options))


def test_inputs_are_not_mutated():
    options = {"b": [1, {"z": 1, "a": 2}], "a": "x"}
    snapshot = json.dumps(options)
    h(options=options)
    assert json.dumps(options) == snapshot


# ---------------------------------------------------------------- argument rejection


@pytest.mark.parametrize(
    "operation", ["", None, 1, b"chat", ["chat"], _StrSub("chat.create")]
)
def test_invalid_operation_is_rejected(operation):
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256(operation, None, "q", {})


@pytest.mark.parametrize(
    "conversation_id", [True, False, "1", 1.0, 1.5, [], b"1", _IntSub(1), _Color.RED]
)
def test_invalid_conversation_id_is_rejected(conversation_id):
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256("chat.create", conversation_id, "q", {})


@pytest.mark.parametrize("question", [None, 1, b"q", ["q"], _StrSub("q")])
def test_invalid_question_is_rejected(question):
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256("chat.create", None, question, {})


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        (),
        "x",
        1,
        {1, 2},
        _DictSub(),
        collections.OrderedDict(),
        collections.UserDict(),
    ],
)
def test_options_must_be_an_exact_dict(options):
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256("chat.create", None, "q", options)


@pytest.mark.parametrize(
    "options",
    [
        {1: "a"},
        {True: "a"},
        {None: "a"},
        {b"k": "a"},
        {_StrSub("k"): "a"},
        {"a": {1: "b"}},
        {"a": [{2: "b"}]},
    ],
)
def test_option_keys_must_be_strings_at_every_depth(options):
    with pytest.raises(CanonicalRequestError):
        h(options=options)


@pytest.mark.parametrize(
    "value",
    [
        {1, 2},
        frozenset({1}),
        b"bytes",
        bytearray(b"b"),
        object(),
        complex(1, 2),
        decimal.Decimal("1.5"),
        datetime.datetime(2026, 1, 1),
        pathlib.Path("."),
        len,
        range(3),
        _StrSub("s"),
        _IntSub(2),
        _Color.RED,
        _DictSub(a=1),
    ],
)
def test_unsupported_option_values_are_rejected(value):
    for options in ({"v": value}, {"v": [value]}, {"v": {"k": value}}):
        with pytest.raises(CanonicalRequestError):
            h(options=options)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, float("nan")])
def test_non_finite_numbers_are_rejected_at_any_depth(value):
    for options in ({"v": value}, {"v": [1, value]}, {"v": {"k": {"j": value}}}):
        with pytest.raises(CanonicalRequestError):
            h(options=options)


@pytest.mark.parametrize("bad", [chr(0xD800), chr(0xDFFF), "a" + chr(0xD83D)])
def test_lone_surrogates_are_rejected_wherever_they_appear(bad):
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256("chat.create", None, bad, {})
    with pytest.raises(CanonicalRequestError):
        canonical_request_sha256(bad, None, "q", {})
    with pytest.raises(CanonicalRequestError):
        h(options={"v": bad})
    with pytest.raises(CanonicalRequestError):
        h(options={bad: 1})


def test_valid_surrogate_pair_as_a_real_character_is_fine():
    assert re.fullmatch(r"[0-9a-f]{64}", h(question="😀"))


def test_circular_options_are_rejected():
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    for options in (cyclic_dict, {"v": cyclic_list}):
        with pytest.raises(CanonicalRequestError):
            h(options=options)


def test_extremely_deep_options_are_rejected_not_crashed():
    deep = current = []
    for _ in range(sys.getrecursionlimit() * 3):
        nxt = []
        current.append(nxt)
        current = nxt
    with pytest.raises(CanonicalRequestError):
        h(options={"v": deep})


@pytest.mark.skipif(
    getattr(sys, "get_int_max_str_digits", lambda: 0)() == 0,
    reason="Python's int-to-str digit limit is disabled",
)
def test_integer_too_large_to_serialize_is_a_typed_error():
    with pytest.raises(CanonicalRequestError):
        h(options={"v": 10 ** (sys.get_int_max_str_digits() + 10)})


def test_error_messages_never_contain_the_request_text():
    secret = "SECRET-QUESTION-TEXT"
    attempts = [
        lambda: canonical_request_sha256("chat.create", None, secret, {"v": object()}),
        lambda: canonical_request_sha256("chat.create", None, secret, {secret: {1: 2}}),
        lambda: canonical_request_sha256(
            "chat.create", None, secret, {"v": {secret, 1}}
        ),
        lambda: canonical_request_sha256("chat.create", None, secret + chr(0xD800), {}),
        lambda: canonical_request_sha256("chat.create", True, secret, {}),
        lambda: canonical_request_sha256("", None, secret, {}),
    ]
    for attempt in attempts:
        with pytest.raises(CanonicalRequestError) as excinfo:
            attempt()
        assert secret not in str(excinfo.value)
        assert secret not in repr(excinfo.value)


# ---------------------------------------------------------------- module shape


def test_error_hierarchy():
    assert issubclass(IdempotencyKeyError, ChatIdempotencyError)
    assert issubclass(CanonicalRequestError, ChatIdempotencyError)
    assert issubclass(ChatIdempotencyError, Exception)
    assert not issubclass(IdempotencyKeyError, CanonicalRequestError)


def test_module_is_pure_standard_library_only():
    tree = ast.parse(
        pathlib.Path(chat_idempotency.__file__).read_text(encoding="utf-8")
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported == {"__future__", "hashlib", "json", "math", "re"}
