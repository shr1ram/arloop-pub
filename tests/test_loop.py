"""The greedy chain end to end: a scripted LLM, a callable runner, and a fake
public_eval / validate / grade. No network, no GPU, seconds to run.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from arloop.budget import Budget
from arloop.llm import ScriptedLLM
from arloop.loop import RunConfig, TaskSpec, run_task
from arloop.runner import CallableRunner, Submission, SubprocessRunner
from arloop.trace import read_events, read_manifest

TASK = TaskSpec(task_id="ahc001", goal="Place rectangles to cover requests.",
                eval="score, higher is better",
                submission_filename="submission.py")


@dataclass
class FakeScore:
    """Stands in for an arbench Score."""

    value: float | None
    valid: bool = True
    is_higher_better: bool = True
    details: dict = field(default_factory=dict)


def _tool(code: str, approach: str = "a heuristic"):
    return {"tool_args": {"code": code, "approach": approach}}


# solution.py writes submission.py; its text is what the fake eval scores
def _writes(marker: str) -> str:
    return f"open('submission.py','w').write({marker!r})\n"


BUGGY = "raise RuntimeError('deliberate bug')\n"


def _config(tmp_path, **kw):
    defaults = dict(run_id="run-1", config_hash="cfgh", seed=0,
                    traces_root=tmp_path / "traces",
                    workspace_root=tmp_path / "ws", exec_timeout_s=60,
                    command=f"{sys.executable} solution.py")
    defaults.update(kw)
    return RunConfig(**defaults)


def _eval_by_marker(scores: dict, **kw):
    """public_eval: read the submission and look its marker up."""
    def run(path: Path):
        return FakeScore(scores[path.read_text()],
                         details={"feedback": "8/8 cases ok"}, **kw)
    return run


def _ok_validate(path: Path):
    return True, None


def _runner():
    return SubprocessRunner(sandbox=False)


def _grade_by_marker(scores: dict):
    def run(path: Path):
        return FakeScore(scores[path.read_text()])
    return run


def test_draft_improve_chain_branches_from_the_best_attempt(tmp_path):
    # v2 scores worse than v1, so the third attempt must improve from v1
    llm = ScriptedLLM([_tool(_writes("v1"), "greedy"),
                       _tool(_writes("v2"), "annealing"),
                       _tool(_writes("v3"), "annealing, tuned")])
    scores = {"v1": 5.0, "v2": 2.0, "v3": 9.0}
    out = run_task(TASK, None, _runner(), Budget("attempts", 3),
                   _config(tmp_path, branch_policy="best"), llm,
                   public_eval=_eval_by_marker(scores), validate=_ok_validate,
                   grade=_grade_by_marker({"v3": 8.5}))

    assert out.status == "ok"
    assert out.n_attempts == 3
    assert out.proxy_score == 9.0
    assert out.heldout_score == 8.5          # graded on the BEST submission

    assert len(llm.calls) == 3
    assert "Improve the score" in llm.calls[1][1]
    # attempt 3 improves from v1 (5.0), not from the tip v2 (2.0)
    third = llm.calls[2][1]
    assert "scored 5.0" in third and _writes("v1") in third
    assert _writes("v2") not in third


def test_branch_policy_last_extends_the_tip(tmp_path):
    llm = ScriptedLLM([_tool(_writes("v1")), _tool(_writes("v2")),
                       _tool(_writes("v3"))])
    scores = {"v1": 5.0, "v2": 2.0, "v3": 1.0}
    run_task(TASK, None, _runner(), Budget("attempts", 3),
             _config(tmp_path, branch_policy="last"), llm,
             public_eval=_eval_by_marker(scores), validate=_ok_validate,
             grade=None)
    third = llm.calls[2][1]
    assert "scored 2.0" in third and _writes("v2") in third


def test_debug_follows_a_failure_and_shows_the_error(tmp_path):
    llm = ScriptedLLM([_tool(BUGGY), _tool(_writes("v1"))])
    out = run_task(TASK, None, _runner(), Budget("attempts", 2),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=None)

    assert out.status == "ok" and out.n_attempts == 2
    assert out.proxy_score == 4.0
    assert "It FAILED" in llm.calls[1][1]
    assert "deliberate bug" in llm.calls[1][1]

    events = list(read_events(out.run_dir))
    fail = next(e for e in events if e.type == "execution_result"
                and e.payload["classification"] == "agent_fixable")
    assert fail.attempt == 0
    assert "deliberate bug" in fail.payload["error_text"]
    assert (out.run_dir / fail.payload["log_ref"]).exists()
    assert fail.payload["error_sig_hash"]


def test_malformed_tool_call_is_charged_as_an_attempt(tmp_path):
    llm = ScriptedLLM([{"tool_args_raw": '{"code": "unterminated'},
                       _tool(_writes("v1"))])
    budget = Budget("attempts", 2)
    out = run_task(TASK, None, _runner(), budget, _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=None)

    assert budget.spent.attempts == 2        # the malformed turn cost one
    assert out.n_attempts == 2
    kinds = [e.payload["classification"] for e in read_events(out.run_dir)
             if e.type == "execution_result"]
    assert kinds[0] == "malformed_tool_call"
    # the next move debugs the malformed reply rather than re-drafting
    assert "It FAILED" in llm.calls[1][1]


def test_empty_code_is_charged_as_an_attempt(tmp_path):
    llm = ScriptedLLM([_tool("   "), _tool(_writes("v1"))])
    budget = Budget("attempts", 2)
    run_task(TASK, None, _runner(), budget, _config(tmp_path), llm,
             public_eval=_eval_by_marker({"v1": 4.0}),
             validate=_ok_validate, grade=None)
    assert budget.spent.attempts == 2


def test_prose_fallback_when_the_tool_call_is_missing(tmp_path):
    prose = f"here you go\n```python\n{_writes('v1')}```"
    llm = ScriptedLLM([prose])
    out = run_task(TASK, None, _runner(), Budget("attempts", 1),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=None)
    assert out.proxy_score == 4.0


def test_code_over_budget_is_rejected_unexecuted(tmp_path):
    llm = ScriptedLLM([_tool("x = 1\n" * 500), _tool(_writes("v1"))])
    budget = Budget("attempts", 2)
    out = run_task(TASK, None, _runner(), budget,
                   _config(tmp_path, code_budget_chars=100), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=None)

    assert budget.spent.attempts == 2
    events = list(read_events(out.run_dir))
    rejected = next(e for e in events if e.type == "execution_result"
                    and e.payload["classification"] == "code_over_budget")
    assert rejected.payload["exit_code"] is None
    assert "over the 100-character limit" in rejected.payload["excerpt"]
    # never executed: no submit event for that attempt
    assert not [e for e in events if e.type == "submit" and e.attempt == 0]


def test_token_budget_stops_the_loop_at_an_attempt_boundary(tmp_path):
    llm = ScriptedLLM([_tool(_writes("v1")), _tool(_writes("v2")),
                       _tool(_writes("v3"))])
    budget = Budget("tokens", 1)             # one attempt always overshoots
    out = run_task(TASK, None, _runner(), budget, _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0, "v2": 5.0,
                                                "v3": 6.0}),
                   validate=_ok_validate, grade=None)

    assert out.status == "ok"
    assert out.n_attempts == 1               # atomic attempts, then stop
    assert budget.spent.tokens >= 1
    assert read_manifest(out.run_dir).budget["type"] == "tokens"


def test_minimize_problems_have_their_proxy_negated(tmp_path):
    # raw 100 then raw 40: lower is better, so 40 must win
    llm = ScriptedLLM([_tool(_writes("v1")), _tool(_writes("v2"))])
    out = run_task(TASK, None, _runner(), Budget("attempts", 2),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 100.0, "v2": 40.0},
                                               is_higher_better=False),
                   validate=_ok_validate, grade=_grade_by_marker({"v2": 33.0}))

    assert out.proxy_score == -40.0
    assert out.heldout_score == 33.0         # heldout is the RAW grade value
    assert "minimise problem" in llm.calls[1][1]
    assert "scored -100.0" in llm.calls[1][1]


def test_validate_failure_comes_back_as_agent_fixable(tmp_path):
    llm = ScriptedLLM([_tool(_writes("v1")), _tool(_writes("v2"))])

    def validate(path: Path):
        return (False, "not valid Python") if path.read_text() == "v1" \
            else (True, None)

    out = run_task(TASK, None, _runner(), Budget("attempts", 2),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 1.0, "v2": 4.0}),
                   validate=validate, grade=None)

    assert out.proxy_score == 4.0            # the invalid attempt never scored
    assert "failed validation: not valid Python" in llm.calls[1][1]
    events = list(read_events(out.run_dir))
    verdicts = [e.payload for e in events if e.type == "submission_validation"]
    assert verdicts[0] == {"ok": False, "reason": "not valid Python"}
    # a rejected attempt is never public-eval'd, so it cannot become best
    assert len([e for e in events if e.type == "public_eval"]) == 1


def test_a_missing_submission_file_reads_as_a_failure(tmp_path):
    llm = ScriptedLLM([_tool("x = 1\n"), _tool(_writes("v1"))])
    out = run_task(TASK, None, _runner(), Budget("attempts", 2),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=None)
    assert "submission.py was not written" in llm.calls[1][1]
    assert out.proxy_score == 4.0


def test_public_eval_infra_fault_fails_the_run(tmp_path):
    # the first attempt scores; the scorer then faults on the second
    llm = ScriptedLLM([_tool(_writes("v1")), _tool(_writes("v2")),
                       _tool(_writes("v3"))])

    def flaky(path: Path):
        if path.read_text() == "v1":
            return FakeScore(4.0, details={"feedback": "8/8 cases ok"})
        return FakeScore(None, valid=False,
                         details={"infra": True, "reason": "scorer missing"})

    out = run_task(TASK, None, _runner(), Budget("attempts", 3),
                   _config(tmp_path), llm, public_eval=flaky,
                   validate=_ok_validate, grade=None)

    assert out.status == "failed"
    assert out.n_attempts == 1               # the faulting attempt never lands
    m = read_manifest(out.run_dir)
    assert m.status == "failed"
    assert "public_eval infra: scorer missing" in m.extra["failure_reason"]
    assert len(llm.calls) == 2               # the run stopped, did not retry
    infra = [e for e in read_events(out.run_dir)
             if e.type == "public_eval" and e.payload.get("infra")]
    assert infra[0].payload["reason"] == "scorer missing"


def test_an_eval_fault_on_the_first_attempt_discards_the_run_dir(tmp_path):
    llm = ScriptedLLM([_tool(_writes("v1"))])

    def broken(path: Path):
        return FakeScore(None, valid=False,
                         details={"infra": True, "reason": "scorer missing"})

    out = run_task(TASK, None, _runner(), Budget("attempts", 3),
                   _config(tmp_path), llm, public_eval=broken,
                   validate=_ok_validate, grade=None)

    # nothing scored, so there is no experimental signal to keep
    assert out.status == "failed" and out.n_attempts == 0
    assert not out.run_dir.exists()


def test_an_infra_failure_with_no_attempts_discards_the_run_dir(tmp_path):
    class Dead:
        model = "dead"
        temperature = 0.0

        def complete(self, system, user):
            raise RuntimeError("endpoint down")

    out = run_task(TASK, None, _runner(), Budget("attempts", 3),
                   _config(tmp_path, llm_retry_wall_s=0.0,
                           llm_retry_sleep_s=0.0), Dead(),
                   public_eval=_eval_by_marker({}), validate=_ok_validate,
                   grade=None)

    assert out.status == "failed" and out.n_attempts == 0
    assert not out.run_dir.exists()          # no signal, so no corpse


def test_manifest_and_trace_record_the_whole_run(tmp_path):
    llm = ScriptedLLM([_tool(BUGGY), _tool(_writes("v1"))])
    out = run_task(TASK, None, _runner(), Budget("attempts", 2),
                   _config(tmp_path, manifest_extra={"arm": "r0-w100-x0"}),
                   llm, public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=_grade_by_marker({"v1": 3.0}))

    m = read_manifest(out.run_dir)
    assert (m.run_id, m.task_id, m.seed) == ("run-1", "ahc001", 0)
    assert (m.status, m.config_hash, m.model) == ("ok", "cfgh", "scripted")
    assert (m.proxy_score, m.heldout_score) == (4.0, 3.0)
    assert m.budget == {"type": "attempts", "limit": 2, "spent": 2}
    assert m.cost["tokens"] > 0 and m.cost["tool_call_missing"] == 0
    assert m.extra["arm"] == "r0-w100-x0" and len(m.extra["template_hash"]) == 16

    events = list(read_events(out.run_dir))
    kinds = [e.type for e in events]
    assert kinds[0] == "task_presented" and kinds[-1] == "final_outcome"
    assert kinds.count("llm_call") == 2
    assert kinds.count("code_written") == 2
    assert kinds.count("execution_result") == 2
    assert kinds.count("score") == 2
    assert "memory_retrieval" not in kinds        # view=None IS the baseline

    call = next(e for e in events if e.type == "llm_call")
    assert call.payload["move"] == "draft"
    assert call.payload["base_attempt"] is None
    assert len(call.payload["prompt_fp"]) == 12
    assert call.payload["tokens"]["prompt"] > 0

    written = [e for e in events if e.type == "code_written"]
    assert written[0].payload["move"] == "draft"
    assert written[0].payload["parent_idx"] == -1
    assert written[1].payload["move"] == "debug:code"
    assert written[1].payload["parent_idx"] == 0
    assert written[0].payload["code_fp"]

    final = events[-1].payload
    assert (final["status"], final["best_attempt"]) == ("ok", 1)
    assert final["infra_failure"] is None
    assert final["budget_spent"]["attempts"] == 2

    # budget-blind: no prompt ever mentions the budget or what remains
    for system, user in llm.calls:
        joined = (system + user).lower()
        assert "budget" not in joined and "remaining" not in joined


def test_memory_view_is_queried_per_attempt_and_keyed_on_the_move(tmp_path):
    @dataclass
    class FakeCase:
        id: str
        content: str
        task_id: str

    class FakeView:
        def __init__(self):
            self.queries = []

        def query(self, text):
            self.queries.append(text)
            return [(FakeCase("case-1", "anneal beat greedy", "ahc002"), 0.9)]

    view = FakeView()
    llm = ScriptedLLM([_tool(BUGGY), _tool(_writes("v1")),
                       _tool(_writes("v2"))])
    out = run_task(TASK, view, _runner(), Budget("attempts", 3),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0, "v2": 5.0}),
                   validate=_ok_validate, grade=None)

    assert len(view.queries) == 3
    assert view.queries[0] == f"{TASK.goal}\n{TASK.eval}"    # draft: task only
    assert "deliberate bug" in view.queries[1]               # debug: the error
    assert _writes("v1") in view.queries[2]                  # improve: the code

    assert "anneal beat greedy" in llm.calls[0][1]
    events = list(read_events(out.run_dir))
    retrievals = [e for e in events if e.type == "memory_retrieval"]
    assert len(retrievals) == 3
    assert retrievals[0].payload["move"] == "draft"
    assert retrievals[0].payload["returned"][0]["case_id"] == "case-1"
    assert retrievals[0].payload["realised_tokens"] > 0


def test_a_bad_branch_policy_is_refused_before_any_trace_is_written(tmp_path):
    with pytest.raises(ValueError, match="branch_policy"):
        run_task(TASK, None, _runner(), Budget("attempts", 1),
                 _config(tmp_path, branch_policy="sideways"),
                 ScriptedLLM([]), public_eval=_eval_by_marker({}),
                 validate=_ok_validate, grade=None)
    assert not (tmp_path / "traces").exists()


def test_a_grader_crash_fails_the_run_without_losing_it(tmp_path):
    def explode(path: Path):
        raise RuntimeError("grader segfault")

    llm = ScriptedLLM([_tool(_writes("v1"))])
    out = run_task(TASK, None, _runner(), Budget("attempts", 1),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({"v1": 4.0}),
                   validate=_ok_validate, grade=explode)

    assert out.status == "failed"
    assert out.proxy_score == 4.0            # the attempt's work survives
    assert out.heldout_score is None
    m = read_manifest(out.run_dir)
    assert "grading_error" in m.extra["failure_reason"]


def test_a_runner_timeout_reads_as_a_fixable_failure(tmp_path):
    def slow(ws, cmd, timeout):
        return Submission(124, "TIMEOUT: killed after 60s\n", 60.0)

    llm = ScriptedLLM([_tool(_writes("v1"))])
    out = run_task(TASK, None, CallableRunner(slow), Budget("attempts", 1),
                   _config(tmp_path), llm,
                   public_eval=_eval_by_marker({}), validate=_ok_validate,
                   grade=None)

    assert out.status == "ok"                # a timeout is the agent's problem
    assert out.proxy_score is None
    assert out.heldout_score is None
