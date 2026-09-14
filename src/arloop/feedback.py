"""Turn a Submission into the feedback the next prompt shows the agent.

Two kinds only: `success` (the script exited 0; the proxy score is filled in
later by the harness public eval) and `agent_fixable` (anything else). There
is no infra class — execution faults are the agent's to fix or the run's to
fail on, never an uncharged retry.
"""
from __future__ import annotations

from dataclasses import dataclass

from arloop.runner import Submission

EXCERPT_TAIL_LINES = 30


@dataclass
class Feedback:
    """One attempt's outcome as the loop and the next prompt see it."""

    kind: str                       # success | agent_fixable
    proxy_score: float | None
    error_text: str                 # full log (empty on success)
    excerpt: str                    # the slice fed back to the LLM
    eval_note: str = ""             # the public eval's own summary line


def excerpt(log: str) -> str:
    """Last traceback block plus the last lines — structural, never char-cut.

    An arbitrary character cut amputates the lesson, so the selection is
    always whole lines.
    """
    lines = log.splitlines()
    tb_start = None
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last"):
            tb_start = i
    parts = []
    if tb_start is not None:
        parts.append("\n".join(lines[tb_start:]))
    tail = "\n".join(lines[-EXCERPT_TAIL_LINES:])
    if not parts or tail not in parts[0]:
        parts.append(tail)
    return "\n---\n".join(parts)


def classify(sub: Submission) -> Feedback:
    """Exit 0 is a success; any other exit is the agent's to fix."""
    if sub.exit_code == 0:
        return Feedback("success", None, "", "")
    return Feedback("agent_fixable", None, sub.log,
                    f"exit {sub.exit_code}\n{excerpt(sub.log)}")
