"""Prompt assembly for the agent loop.

The template components are DATA (a PromptSet, defaults in prompts.yaml);
this module only renders them. template_hash goes into every run manifest
and into the arm's config_hash, so a prompt tweak is never a silent confound.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields
from importlib.resources import files as _pkg_files

import yaml

_AGENT = yaml.safe_load(
    _pkg_files("arloop").joinpath("prompts.yaml").read_text())["agent"]


@dataclass(frozen=True)
class PromptSet:
    """The agent-facing templates; an arm config may carry its own."""

    system: str = _AGENT["system"]
    cases_header: str = _AGENT["cases_header"]
    case_line: str = _AGENT["case_line"]
    case_label: str = _AGENT["case_label"]
    history_header: str = _AGENT["history_header"]
    draft: str = _AGENT["draft"]
    debug: str = _AGENT["debug"]
    improve: str = _AGENT["improve"]


def render_system(ps: PromptSet, exec_timeout_s: int, code_budget_chars: int,
                  cpu_cap: int, compute: str = "cpu") -> str:
    """The system prompt with its static environment facts substituted."""
    # plain replace, not str.format: the template carries literal braces
    compute_desc = "CPU only — no GPU is available" if compute == "cpu" else compute
    text = ps.system
    for key, value in (("{exec_ceiling_s}", str(exec_timeout_s)),
                       ("{code_budget_chars}", str(code_budget_chars)),
                       ("{cpu_cap}", str(cpu_cap)),
                       ("{compute}", compute_desc)):
        text = text.replace(key, value)
    return text


def render_cases(cases, ps: PromptSet) -> str:
    """The retrieved cases block, or "" when nothing was retrieved."""
    if not cases:
        return ""
    lines = [ps.cases_header]
    lines += [ps.case_line.format(label=ps.case_label, content=c.content)
              for c in cases]
    return "\n".join(lines) + "\n\n"


def render_history(entries, ps: PromptSet) -> str:
    """One mechanical line per prior attempt: (num, move, outcome, approach)."""
    if not entries:
        return ""
    lines = [ps.history_header.rstrip()]
    for num, move, outcome, approach in entries:
        lines.append(f"  {num}. {move}: {outcome} — APPROACH: "
                     f"{approach or '(none given)'}")
    return "\n".join(lines) + "\n\n"


def render_draft(ps: PromptSet, goal: str, eval: str, cases: str) -> str:
    """The first attempt's prompt."""
    return ps.draft.format(goal=goal, eval=eval, cases=cases)


def render_debug(ps: PromptSet, goal: str, eval: str, cases: str,
                 history: str, prev_code: str, excerpt: str) -> str:
    """Repair the attempt that just failed."""
    return ps.debug.format(goal=goal, eval=eval, cases=cases, history=history,
                           prev_code=prev_code, excerpt=excerpt)


def render_improve(ps: PromptSet, goal: str, eval: str, cases: str,
                   history: str, prev_code: str, prev_score: str) -> str:
    """Extend a scoring attempt."""
    return ps.improve.format(goal=goal, eval=eval, cases=cases,
                             history=history, prev_code=prev_code,
                             prev_score=prev_score)


_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_python_block(text: str) -> str:
    """The LAST fenced block — models restate, so the final one is the answer."""
    blocks = _BLOCK_RE.findall(text)
    if not blocks:
        raise ValueError("response contained no fenced python code block")
    return blocks[-1].strip() + "\n"


def template_hash(ps: PromptSet) -> str:
    """Identity of the prompt text; part of the arm's config_hash."""
    payload = "\n".join(getattr(ps, f.name) for f in fields(ps))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
