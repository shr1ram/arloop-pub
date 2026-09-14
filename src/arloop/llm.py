"""The agent's LLM client: one self-served vLLM OpenAI-compatible endpoint.

Plain completions stream (a reasoning model can think longer than a gateway's
idle timeout before its first byte); the forced `submit_solution` tool call
does not. ScriptedLLM is the test double, so no test needs a network.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from arloop.tokens import approx_tokens, tokens_to_chars, tokens_to_words

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

#: delays between provider-error retries; length = retries after the first try
RETRY_DELAYS_S = (5.0, 15.0, 45.0, 90.0)

#: lower-cased substrings (matched by ANY) that mark an HTTP-400 as a
#: context-length overflow rather than some other bad request. Broad enough
#: that a server-version wording tweak cannot blind the guard, specific enough
#: that an unrelated 400 (bad schema, bad param) never matches.
CONTEXT_LENGTH_MARKERS = (
    "maximum context length",
    "context length",
    "context window",
    "reduce the length",
    "longer than the maximum",
    "please reduce",
)

#: Qwen's graceful-closure sentence, injected by the vLLM budget processor at
#: the thinking_token_budget cap. Its presence in the reasoning text is the
#: client-visible proof the span was CUT rather than closing on its own.
REASONING_BUDGET_MARKER = "give the solution based on the thinking directly now"

#: fraction of the budget the realised reasoning length must reach for the
#: length co-signal to fire — a cut span lands at the ceiling with no marker,
#: a natural close well under it. The margin is wide because the ceiling is
#: computed from chars_per_token, which is calibrated on bank cases (code and
#: prose); reasoning runs slightly denser, so a cut span measures a little
#: short of the nominal ceiling rather than exactly at it.
REASONING_BUDGET_NEAR = 0.8


class ContextOverflowError(Exception):
    """A deterministic context-length HTTP-400: the same request 400s forever,
    so it is never retried and the run stops immediately."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    tool_args: Optional[dict] = None
    #: the tool-call argument string, kept ONLY when it could not be decoded
    tool_args_raw: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_call_missing: bool = False
    reasoning_chars: int = 0
    reasoning_stopped: str = "none"       # none | natural | budget_forced
    thinking_ignored: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt + completion; reasoning is already inside completion."""
        return self.prompt_tokens + self.completion_tokens


_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_solution",
        "description": "Submit the complete solution.py and a one-sentence "
                       "summary of the approach it takes.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                         "description": "The complete, self-contained "
                                        "contents of solution.py."},
                "approach": {"type": "string",
                             "description": "One sentence summarising this "
                                            "solution's approach."},
            },
            "required": ["code", "approach"],
        },
    },
}


def submit_tool(history_write_tokens: int) -> dict:
    """The submit_solution schema specialised to the write budget.

    0 drops `approach` entirely (code only); n > 0 discloses the word budget
    in its description. A deep copy, so concurrent cells share no schema.
    """
    tool = copy.deepcopy(_SUBMIT_TOOL)
    params = tool["function"]["parameters"]
    if history_write_tokens == 0:
        params["properties"].pop("approach")
        params["required"] = [r for r in params["required"] if r != "approach"]
    else:
        words = tokens_to_words(history_write_tokens)
        params["properties"]["approach"]["description"] = (
            f"A description of this solution's approach. Use about {words} "
            "words: explain the algorithm/design, the key decisions and why "
            "you made them, and any trade-offs or caveats.")
    return tool


def error_text(exc: Exception) -> str:
    """Every string channel a provider error carries its message on, joined."""
    parts = [str(exc)]
    msg = getattr(exc, "message", None)
    if isinstance(msg, str):
        parts.append(msg)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            parts.append(err["message"])
        elif isinstance(err, str):
            parts.append(err)
    return " ".join(parts)


def is_context_length_error(exc: Exception) -> bool:
    """Is this a bad request whose message signals a context-length overflow?"""
    import openai
    if not isinstance(exc, openai.BadRequestError):
        return False
    text = error_text(exc).lower()
    return any(m in text for m in CONTEXT_LENGTH_MARKERS)


def reasoning_stopped(reasoning_chars: int, reasoning: str,
                      budget: Optional[int]) -> str:
    """How the reasoning span ended: none | natural | budget_forced.

    Only ever budget_forced when a budget was actually in force. Two signals:
    the served closure marker, and reasoning that ran to the ceiling (a bare
    </think> cut inserts no marker).
    """
    if not (reasoning_chars or reasoning):
        return "none"
    if budget is not None:
        if REASONING_BUDGET_MARKER in reasoning.lower():
            return "budget_forced"
        ceiling = tokens_to_chars(budget)
        if ceiling > 0 and reasoning_chars >= REASONING_BUDGET_NEAR * ceiling:
            return "budget_forced"
    return "natural"


def thinking_ignored(thinking: Optional[bool], reasoning_chars: int,
                     text: str) -> bool:
    """Did a thinking-OFF request reason anyway?

    Inline `<think>` counts: some chat templates emit reasoning into the
    ordinary content stream, where no token counter would reveal it.
    """
    if thinking is not False:
        return False
    return bool(reasoning_chars or "<think>" in (text or ""))


def is_transient(e: Exception) -> bool:
    """Provider weather (retry) vs a deterministic failure (raise now).

    429 and 5xx are weather; any other HTTP status fails identically forever.
    Without a status, connection and timeout errors from the SDK or its HTTP
    client are weather — matched by module name because the SDK vendors its
    HTTP client under a versioned name, and a mid-stream cut surfaces as the
    client's own exception rather than an SDK one. Anything else (a local
    bug, a test double that ran out of replies) is deterministic.
    """
    status = getattr(e, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    module = type(e).__module__.split(".")[0]
    return (isinstance(e, (OSError, TimeoutError))
            or module == "openai" or module.startswith("httpx"))


def retry_provider_errors(fn: Callable[[], LLMResponse],
                          is_retryable: Callable[[Exception], bool],
                          delays_s=RETRY_DELAYS_S,
                          sleep=time.sleep,
                          rng=random.random) -> LLMResponse:
    """Retry through transient provider errors; others propagate.

    Each delay is jittered to 0.5-1.5x its ladder value: the endpoint's quota
    is shared, so a fixed ladder makes a whole fleet retry in lockstep and
    trip the quota again on schedule.
    """
    for i, delay in enumerate([*delays_s, None]):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - filtered by is_retryable
            if delay is None or not is_retryable(e):
                raise
            delay *= 0.5 + rng()
            log.warning("provider error (%s: %s) - retry %d/%d in %.0fs",
                        type(e).__name__, str(e)[:120], i + 1, len(delays_s),
                        delay)
            sleep(delay)
    raise AssertionError("unreachable")


#: fields a server may return out-of-band reasoning on. vLLM's reasoning
#: parsers have shipped both spellings, and a build that uses the one we do
#: not read makes the whole reasoning axis silently read as zero — the exact
#: failure the axis exists to measure. Reading both costs nothing.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def reasoning_text(msg) -> str:
    """The out-of-band reasoning on a message or delta, or ""."""
    for name in REASONING_FIELDS:
        value = getattr(msg, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _tool_args_from(tool_calls) -> tuple[Optional[dict], Optional[str]]:
    """(tool_args, tool_args_raw) from a provider's tool-call list.

    The whole extraction is guarded, not just json.loads: a schema-off entry
    must degrade to the prose fallback rather than raise past the retry layer
    as a non-retryable error. The raw string is kept whenever one was read, so
    the malformed path is never empty in the trace.
    """
    raw = None
    args = None
    try:
        raw = tool_calls[0].function.arguments
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
        args = None
    if not isinstance(args, dict):
        args = None
    if args is None and raw is not None:
        return None, raw if isinstance(raw, str) else repr(raw)
    return args, None


class VllmClient:
    """OpenAI-compatible client for a self-served vLLM endpoint."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 seed: int | None = None, timeout_s: float = 600,
                 max_retries: int = 4, thinking: bool | None = None,
                 thinking_token_budget: int | None = None,
                 use_tool: bool = False, history_write_tokens: int = 100):
        from openai import OpenAI   # lazy: importing arloop.llm needs no SDK
        self.model = model
        self.temperature = temperature
        self._seed = seed
        self._thinking = thinking
        self._budget = thinking_token_budget if thinking else None
        self._use_tool = use_tool
        self._submit_tool = submit_tool(history_write_tokens)
        extra: dict = {}
        if thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": thinking}
            # the reasoning-span cap is a TOP-LEVEL request field; vLLM
            # silently ignores it inside chat_template_kwargs
            if self._budget is not None:
                extra["thinking_token_budget"] = self._budget
        self._extra_body = extra or None
        # SDK retries off: our ladder owns every retry, so each round is
        # visible in the log instead of silently spending wall clock
        self._client = OpenAI(
            base_url=base_url or os.environ.get("VLLM_BASE_URL",
                                                DEFAULT_BASE_URL),
            api_key=api_key or os.environ.get("VLLM_API_KEY", "none"),
            timeout=timeout_s, max_retries=0)
        n = max(0, max_retries)
        ladder = list(RETRY_DELAYS_S) + [RETRY_DELAYS_S[-1]] * n
        self._delays = tuple(ladder[:n])

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        return is_transient(e)

    def _create(self, **kwargs):
        """The single provider request boundary for both transport paths."""
        import openai
        try:
            return self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if is_context_length_error(e):
                raise ContextOverflowError(str(e)) from e
            raise

    def _base_kwargs(self) -> dict:
        kwargs: dict = {"model": self.model, "temperature": self.temperature}
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._extra_body is not None:
            kwargs["extra_body"] = self._extra_body
        return kwargs

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def _fallback_tokens(self, system: str, user: str, body: str,
                         reasoning_chars: int) -> tuple[int, int]:
        """Token counts when the server sent no usage block.

        Reasoning chars are included: out-of-band reasoning never enters the
        text, and omitting it under-charges exactly the arms that reason.
        """
        return (approx_tokens(system + user),
                approx_tokens(body + " " * reasoning_chars))

    def _stream(self, system: str, user: str) -> LLMResponse:
        stream = self._create(messages=self._messages(system, user),
                              stream=True,
                              stream_options={"include_usage": True},
                              **self._base_kwargs())
        parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = completion_tokens = 0
        finish_reason = None
        for chunk in stream:
            for choice in (chunk.choices or []):
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if not delta:
                    continue
                if delta.content:
                    parts.append(delta.content)
                rt = reasoning_text(delta)
                if rt:
                    reasoning_parts.append(rt)
            if chunk.usage:   # final chunk under include_usage
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
        text = "".join(parts)
        reasoning = "".join(reasoning_parts)
        if not (prompt_tokens or completion_tokens):
            prompt_tokens, completion_tokens = self._fallback_tokens(
                system, user, text, len(reasoning))
        return LLMResponse(
            text=text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, finish_reason=finish_reason,
            reasoning_chars=len(reasoning),
            reasoning_stopped=reasoning_stopped(len(reasoning), reasoning,
                                                self._budget),
            thinking_ignored=thinking_ignored(self._thinking, len(reasoning),
                                              text))

    def _tool_call(self, system: str, user: str) -> LLMResponse:
        # not streamed: forced tool_choice replies are short JSON args with no
        # reasoning preamble, so they land inside any gateway idle cutoff
        name = self._submit_tool["function"]["name"]
        resp = self._create(
            messages=self._messages(system, user), stream=False,
            tools=[self._submit_tool],
            tool_choice={"type": "function", "function": {"name": name}},
            **self._base_kwargs())
        choice = resp.choices[0]
        usage = resp.usage
        text = choice.message.content or ""
        reasoning = reasoning_text(choice.message)
        # the PRESENCE of tool_calls is the fact; finish_reason is advisory
        # (vLLM reports "stop" on a valid forced tool call) and never gates it
        tool_calls = choice.message.tool_calls
        tool_args, tool_args_raw = (_tool_args_from(tool_calls)
                                    if tool_calls else (None, None))
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        if not (prompt_tokens or completion_tokens):
            body = text or tool_args_raw or json.dumps(tool_args or {})
            prompt_tokens, completion_tokens = self._fallback_tokens(
                system, user, body, len(reasoning))
        return LLMResponse(
            text=text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, tool_args=tool_args,
            tool_args_raw=tool_args_raw,
            finish_reason=choice.finish_reason,
            tool_call_missing=not tool_calls,
            reasoning_chars=len(reasoning),
            reasoning_stopped=reasoning_stopped(len(reasoning), reasoning,
                                                self._budget),
            thinking_ignored=thinking_ignored(self._thinking, len(reasoning),
                                              text))

    def complete(self, system: str, user: str) -> LLMResponse:
        """A completion: the forced submit tool call under use_tool, else text."""
        fn = self._tool_call if self._use_tool else self._stream
        return retry_provider_errors(lambda: fn(system, user),
                                     self._is_retryable, delays_s=self._delays)

    def complete_text(self, system: str, user: str) -> LLMResponse:
        """A plain streaming completion that never carries the submit tool."""
        return retry_provider_errors(lambda: self._stream(system, user),
                                     self._is_retryable, delays_s=self._delays)


@dataclass
class ScriptedLLM:
    """Test double: pops responses in order, records every (system, user).

    A response is a plain str (the prose path), {"tool_args": {...}} (the
    tool-call path) or {"tool_args_raw": "..."} (undecodable tool arguments).
    """
    responses: list
    model: str = "scripted"
    temperature: float = 0.0
    calls: list = field(default_factory=list)

    def complete(self, system: str, user: str) -> LLMResponse:
        """Pop the next scripted response."""
        self.calls.append((system, user))
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        item = self.responses.pop(0)
        prompt_tokens = approx_tokens(system + user)
        if isinstance(item, dict) and "tool_args_raw" in item:
            raw = item["tool_args_raw"]
            return LLMResponse(text="", prompt_tokens=prompt_tokens,
                               completion_tokens=approx_tokens(raw),
                               tool_args_raw=raw)
        if isinstance(item, dict):
            args = item["tool_args"]
            return LLMResponse(text="", prompt_tokens=prompt_tokens,
                               completion_tokens=approx_tokens(
                                   json.dumps(args)),
                               tool_args=args)
        return LLMResponse(text=item, prompt_tokens=prompt_tokens,
                           completion_tokens=approx_tokens(item))

    def complete_text(self, system: str, user: str) -> LLMResponse:
        """Same script; the double draws no distinction between the paths."""
        return self.complete(system, user)
