"""Out-of-band reasoning is read whichever field the server spells it in.

A build that uses the spelling we do not read would make the whole reasoning
axis measure zero while looking healthy, so both are covered.
"""
import types

from arloop.llm import reasoning_stopped, reasoning_text
from arloop.tokens import tokens_to_chars


def _msg(**fields):
    return types.SimpleNamespace(**fields)


def test_either_field_spelling_is_read():
    assert reasoning_text(_msg(reasoning_content="abc")) == "abc"
    assert reasoning_text(_msg(reasoning="xyz")) == "xyz"
    # both present: the explicit content field wins
    assert reasoning_text(_msg(reasoning_content="abc", reasoning="xyz")) == "abc"


def test_absent_or_empty_reasoning_reads_as_none():
    assert reasoning_text(_msg(content="hi")) == ""
    assert reasoning_text(_msg(reasoning=None, reasoning_content="")) == ""
    assert reasoning_stopped(0, "", 3000) == "none"


def test_a_span_at_the_ceiling_is_budget_forced():
    at_ceiling = int(0.95 * tokens_to_chars(3000))
    assert reasoning_stopped(at_ceiling, "x" * at_ceiling, 3000) == "budget_forced"
    assert reasoning_stopped(100, "x" * 100, 3000) == "natural"
    # no budget in force: never forced, whatever the length
    assert reasoning_stopped(at_ceiling, "x" * at_ceiling, None) == "natural"


def test_a_measured_cut_span_is_detected_as_forced():
    """Live 27b spans cut at their budget, in chars: reasoning runs denser
    than the calibrated chars_per_token, so a cut lands under the nominal
    ceiling. These are the measured lengths at each budget."""
    for budget, measured in ((500, 1743), (1000, 3449)):
        assert reasoning_stopped(measured, "x" * measured, budget) == "budget_forced"
    # the same span with a budget it never approached is a natural close
    assert reasoning_stopped(1743, "x" * 1743, 9000) == "natural"
