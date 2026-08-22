import pytest

from premove_itn.tokenize import tokenize
from premove_itn.types import WordToken


def test_tokenize_preserves_case_and_offsets_across_whitespace() -> None:
    text = "  Meet\tme at  Four thirty \n"

    tokens = tokenize(text)

    assert tokens == (
        WordToken("Meet", 2, 6),
        WordToken("me", 7, 9),
        WordToken("at", 10, 12),
        WordToken("Four", 14, 18),
        WordToken("thirty", 19, 25),
    )
    assert tuple(text[token.start : token.end] for token in tokens) == (
        "Meet",
        "me",
        "at",
        "Four",
        "thirty",
    )


def test_tokenize_separates_punctuation_without_losing_offsets() -> None:
    text = "Call me at 7:30, please!"

    tokens = tokenize(text)

    assert tokens == (
        WordToken("Call", 0, 4),
        WordToken("me", 5, 7),
        WordToken("at", 8, 10),
        WordToken("7", 11, 12),
        WordToken(":", 12, 13),
        WordToken("30", 13, 15),
        WordToken(",", 15, 16),
        WordToken("please", 17, 23),
        WordToken("!", 23, 24),
    )


def test_tokenize_keeps_internal_apostrophes_inside_words() -> None:
    text = "Don't change Aryaman’s ID."

    assert tokenize(text) == (
        WordToken("Don't", 0, 5),
        WordToken("change", 6, 12),
        WordToken("Aryaman’s", 13, 22),
        WordToken("ID", 23, 25),
        WordToken(".", 25, 26),
    )


def test_tokenize_handles_mixed_numeric_and_lexical_tokens() -> None:
    text = "  AB12 007 two-four  "

    tokens = tokenize(text)

    assert tuple(token.text for token in tokens) == ("AB12", "007", "two", "-", "four")
    assert all(text[token.start : token.end] == token.text for token in tokens)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", ("hello", "world")),
        ("I'll arrive", ("I'll", "arrive")),
        ("I’ll arrive", ("I’ll", "arrive")),
        ("seven thirty.", ("seven", "thirty", ".")),
        ("3 seven one 8", ("3", "seven", "one", "8")),
        ("a@gmail.com", ("a", "@", "gmail", ".", "com")),
        ("twenty-first", ("twenty", "-", "first")),
        ("7:30 pm", ("7", ":", "30", "pm")),
        ("", ()),
        ("   ", ()),
    ],
)
def test_tokenize_covers_expected_asr_token_shapes(
    text: str, expected: tuple[str, ...]
) -> None:
    tokens = tokenize(text)

    assert tuple(token.text for token in tokens) == expected
    assert all(text[token.start : token.end] == token.text for token in tokens)
