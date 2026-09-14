"""The prompt templates render, hash, and give back the code they carry."""
from __future__ import annotations

from dataclasses import replace

import pytest

from arloop.prompts import (PromptSet, extract_python_block, render_cases,
                            render_debug, render_draft, render_history,
                            render_improve, render_system, template_hash)

PS = PromptSet()


class _Case:
    def __init__(self, content):
        self.content = content


def test_system_substitutes_every_placeholder():
    out = render_system(PS, exec_timeout_s=900, code_budget_chars=30_000,
                        cpu_cap=4, compute="cpu")
    assert "killed after 900 seconds" in out
    assert "under 30000 characters" in out
    assert "CPU only — no GPU is available" in out
    for placeholder in ("{exec_ceiling_s}", "{code_budget_chars}",
                        "{cpu_cap}", "{compute}"):
        assert placeholder not in out


def test_system_keeps_the_submit_tool_contract():
    out = render_system(PS, 900, 30_000, 1)
    assert "submit_solution tool" in out
    assert "approach-field description asks for" in out
    # the grid runs on the official public eval: the agent never self-scores
    assert "Do not write result.json" in out


def test_system_is_budget_blind():
    out = render_system(PS, 900, 30_000, 1).lower()
    assert "budget" not in out and "remaining" not in out


def test_non_cpu_compute_is_passed_through_verbatim():
    out = render_system(PS, 900, 30_000, 1, compute="1x A100 40GB")
    assert "Compute available to your script: 1x A100 40GB." in out


def test_draft_carries_task_eval_and_cases():
    cases = render_cases([_Case("earlier run: anneal beat greedy")], PS)
    out = render_draft(PS, goal="GOAL-X", eval="EVAL-Y", cases=cases)
    assert "GOAL-X" in out and "EVAL-Y" in out
    assert "Relevant experience from earlier tasks" in out
    assert "[a full earlier run, summarised] earlier run: anneal beat greedy" in out
    assert "submit_solution tool" in out


def test_no_cases_renders_empty_and_leaves_no_placeholder():
    assert render_cases([], PS) == ""
    out = render_draft(PS, goal="G", eval="E", cases="")
    assert "{cases}" not in out
    assert "Relevant experience" not in out


def test_debug_shows_previous_code_and_the_error():
    out = render_debug(PS, goal="G", eval="E", cases="",
                       history=render_history([(1, "draft", "failed", "greedy")], PS),
                       prev_code="print(1)", excerpt="ZeroDivisionError")
    assert "It FAILED" in out
    assert "print(1)" in out and "ZeroDivisionError" in out
    assert "1. draft: failed — APPROACH: greedy" in out


def test_improve_shows_the_score_with_its_eval_note():
    out = render_improve(PS, goal="G", eval="E", cases="", history="",
                         prev_code="print(1)", prev_score="12.5 (8/8 cases ok)")
    assert "scored 12.5 (8/8 cases ok)" in out
    assert "Improve the score" in out


def test_history_is_one_mechanical_line_per_attempt():
    out = render_history([(1, "draft", "failed", None),
                          (2, "debug", "proxy 3.0", "fixed the index")], PS)
    assert out.startswith("Your attempts so far:")
    assert "1. draft: failed — APPROACH: (none given)" in out
    assert "2. debug: proxy 3.0 — APPROACH: fixed the index" in out
    assert render_history([], PS) == ""


def test_extract_python_block_takes_the_last_one():
    text = "```python\nold = 1\n```\nthen\n```python\nnew = 2\n```"
    assert extract_python_block(text) == "new = 2\n"
    with pytest.raises(ValueError):
        extract_python_block("no fences here")


def test_template_hash_is_stable_and_covers_every_field():
    base = template_hash(PS)
    assert base == template_hash(PromptSet())
    assert len(base) == 16
    for field in ("system", "cases_header", "case_line", "case_label",
                  "history_header", "draft", "debug", "improve"):
        assert template_hash(replace(PS, **{field: "changed"})) != base
