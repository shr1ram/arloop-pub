"""Token ratio conversions."""
from __future__ import annotations

import pytest

from arloop import tokens


@pytest.fixture(autouse=True)
def reset_ratio():
    tokens.set_chars_per_token(tokens.DEFAULT_CHARS_PER_TOKEN)
    yield
    tokens.set_chars_per_token(tokens.DEFAULT_CHARS_PER_TOKEN)


def test_empty_text_is_zero_tokens():
    assert tokens.approx_tokens("") == 0


def test_short_text_floors_to_at_least_one_token():
    assert tokens.approx_tokens("a") == 1


def test_tokens_are_utf8_bytes_over_the_ratio():
    assert tokens.approx_tokens("a" * 40) == 10
    # multi-byte text is counted by bytes, never undercounted by characters
    assert tokens.approx_tokens("é" * 20) == 10


def test_set_chars_per_token_changes_every_conversion():
    tokens.set_chars_per_token(3.5)
    assert tokens.chars_per_token() == 3.5
    assert tokens.approx_tokens("a" * 35) == 10
    assert tokens.tokens_to_chars(100) == 350
    assert tokens.tokens_to_words(100) == int(350 / tokens.CHARS_PER_WORD)


def test_words_route_through_chars():
    assert tokens.tokens_to_words(300) == int(
        tokens.tokens_to_chars(300) / tokens.CHARS_PER_WORD)


def test_non_positive_ratio_refused():
    with pytest.raises(ValueError):
        tokens.set_chars_per_token(0)
