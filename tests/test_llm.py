"""LLM helpers, the submit tool schema and the scripted double.

Everything here is pure: no network, and the openai SDK is never imported.
"""
from __future__ import annotations

import json

import pytest

from arloop import tokens
from arloop.llm import (
    REASONING_BUDGET_MARKER, RETRY_DELAYS_S, ContextOverflowError, LLMResponse,
    ScriptedLLM, error_text, reasoning_stopped, retry_provider_errors,
    submit_tool, thinking_ignored,
)


@pytest.fixture(autouse=True)
def reset_ratio():
    tokens.set_chars_per_token(tokens.DEFAULT_CHARS_PER_TOKEN)
    yield
    tokens.set_chars_per_token(tokens.DEFAULT_CHARS_PER_TOKEN)


# ------------------------------------------------------------- submit tool

def _params(tool):
    return tool["function"]["parameters"]


def test_submit_tool_at_zero_drops_the_approach_field():
    params = _params(submit_tool(0))
    assert "approach" not in params["properties"]
    assert params["required"] == ["code"]
    assert "code" in params["properties"]


def test_submit_tool_above_zero_discloses_the_word_budget():
    params = _params(submit_tool(300))
    words = tokens.tokens_to_words(300)
    assert params["required"] == ["code", "approach"]
    assert f"about {words} words" in params["properties"]["approach"]["description"]


def test_submit_tool_word_count_follows_the_ratio():
    tokens.set_chars_per_token(3.5)
    desc = _params(submit_tool(900))["properties"]["approach"]["description"]
    assert f"about {tokens.tokens_to_words(900)} words" in desc


def test_submit_tool_returns_a_fresh_copy_each_call():
    a = submit_tool(100)
    _params(a)["properties"].pop("approach")
    assert "approach" in _params(submit_tool(100))["properties"]


# --------------------------------------------------------- reasoning spans

def test_no_reasoning_is_stopped_none():
    assert reasoning_stopped(0, "", budget=None) == "none"
    assert reasoning_stopped(0, "", budget=2048) == "none"


def test_reasoning_without_a_budget_is_natural():
    text = "thinking hard " * 100
    assert reasoning_stopped(len(text), text, budget=None) == "natural"


def test_closure_marker_under_a_budget_is_budget_forced():
    text = "some thinking. " + REASONING_BUDGET_MARKER.upper()
    assert reasoning_stopped(len(text), text, budget=2048) == "budget_forced"


def test_reasoning_at_the_ceiling_is_budget_forced_without_a_marker():
    chars = tokens.tokens_to_chars(64)
    assert reasoning_stopped(chars, "x" * chars, budget=64) == "budget_forced"


def test_short_reasoning_under_a_budget_is_natural():
    assert reasoning_stopped(10, "short think", budget=2048) == "natural"


# -------------------------------------------------------- thinking ignored

def test_thinking_ignored_only_when_thinking_was_requested_off():
    assert thinking_ignored(False, 500, "") is True
    assert thinking_ignored(True, 500, "") is False
    assert thinking_ignored(None, 500, "") is False


def test_inline_think_block_counts_as_reasoning():
    assert thinking_ignored(False, 0, "<think>hmm</think>answer") is True
    assert thinking_ignored(False, 0, "answer") is False


# ------------------------------------------------------------ error text

class _FakeStatusError(Exception):
    def __init__(self, message, body):
        super().__init__("api error")
        self.message = message
        self.body = body


def test_error_text_joins_every_channel():
    exc = _FakeStatusError("maximum context length is 32768",
                           {"error": {"message": "please reduce"}})
    text = error_text(exc)
    assert "api error" in text
    assert "maximum context length" in text
    assert "please reduce" in text


def test_error_text_tolerates_a_bare_exception():
    assert error_text(ValueError("nope")) == "nope"


# ----------------------------------------------------------- retry ladder

class _Boom(Exception):
    pass


def test_retry_retries_then_succeeds_with_jittered_delays():
    slept: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom("transient")
        return LLMResponse(text="ok", prompt_tokens=1, completion_tokens=1)

    resp = retry_provider_errors(fn, lambda e: isinstance(e, _Boom),
                                 sleep=slept.append, rng=lambda: 0.5)
    assert resp.text == "ok" and calls["n"] == 3
    assert slept == [RETRY_DELAYS_S[0], RETRY_DELAYS_S[1]]


def test_jitter_scales_each_delay_between_half_and_one_and_a_half():
    for r, factor in ((0.0, 0.5), (1.0, 1.5)):
        slept: list[float] = []
        with pytest.raises(_Boom):
            retry_provider_errors(
                _raise_boom, lambda e: isinstance(e, _Boom),
                delays_s=(10.0,), sleep=slept.append, rng=lambda: r)
        assert slept == [10.0 * factor]


def _raise_boom():
    raise _Boom("always")


def test_non_retryable_errors_propagate_immediately():
    slept: list[float] = []
    with pytest.raises(ContextOverflowError):
        retry_provider_errors(_raise_overflow, lambda e: isinstance(e, _Boom),
                              sleep=slept.append, rng=lambda: 0.5)
    assert slept == []


def _raise_overflow():
    raise ContextOverflowError("prompt too long")


def test_retry_gives_up_after_the_ladder():
    slept: list[float] = []
    with pytest.raises(_Boom):
        retry_provider_errors(_raise_boom, lambda e: isinstance(e, _Boom),
                              sleep=slept.append, rng=lambda: 0.5)
    assert len(slept) == len(RETRY_DELAYS_S)


# ---------------------------------------------------------------- response

def test_total_tokens_sums_prompt_and_completion():
    assert LLMResponse("t", 100, 25).total_tokens == 125


# -------------------------------------------------------------- scripted

def test_scripted_pops_in_order_and_records_calls():
    llm = ScriptedLLM(["first", "second"])
    assert llm.complete("sys", "a").text == "first"
    assert llm.complete_text("sys", "b").text == "second"
    assert llm.calls == [("sys", "a"), ("sys", "b")]
    assert llm.model == "scripted"


def test_scripted_token_counts_are_length_proxies():
    llm = ScriptedLLM(["hello"])
    r = llm.complete("sys", "user")
    assert r.prompt_tokens == tokens.approx_tokens("sysuser")
    assert r.completion_tokens == tokens.approx_tokens("hello")


def test_scripted_tool_args_path():
    args = {"code": "print(1)", "approach": "trivial"}
    r = ScriptedLLM([{"tool_args": args}]).complete("s", "u")
    assert r.tool_args == args and r.tool_args_raw is None
    assert r.text == ""
    assert r.completion_tokens == tokens.approx_tokens(json.dumps(args))


def test_scripted_malformed_tool_args_path():
    r = ScriptedLLM([{"tool_args_raw": "{not json"}]).complete("s", "u")
    assert r.tool_args is None and r.tool_args_raw == "{not json"


def test_scripted_runs_out_of_responses_loudly():
    llm = ScriptedLLM([])
    with pytest.raises(RuntimeError, match="ran out"):
        llm.complete("s", "u")
