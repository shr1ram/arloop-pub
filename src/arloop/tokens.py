"""Char/word <-> token conversion, the one ratio a cell uses everywhere.

Budget enforcement, the disclosed write-budget word count and the usage
fallback all route through here, so they can never disagree. The ratio is
process-global and set once per cell from the arm config.
"""
from __future__ import annotations

DEFAULT_CHARS_PER_TOKEN = 4.0
CHARS_PER_WORD = 6.0

_chars_per_token: float = DEFAULT_CHARS_PER_TOKEN


def set_chars_per_token(chars_per_token: float) -> None:
    """Set the process-global char/token ratio (run_cell, once per cell)."""
    global _chars_per_token
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    _chars_per_token = float(chars_per_token)


def chars_per_token() -> float:
    """The ratio currently in force."""
    return _chars_per_token


def approx_tokens(text: str) -> int:
    """Approximate a token count as UTF-8 bytes / the ratio.

    Bytes rather than characters so multi-byte text is never undercounted.
    """
    if not text:
        return 0
    return max(1, int(len(text.encode("utf-8")) / _chars_per_token))


def tokens_to_chars(n_tokens: int) -> int:
    """The char budget a token budget permits (the enforcement side)."""
    return int(n_tokens * _chars_per_token)


def tokens_to_words(n_tokens: int) -> int:
    """The word count a token budget permits (the disclosure side)."""
    return int(tokens_to_chars(n_tokens) / CHARS_PER_WORD)
