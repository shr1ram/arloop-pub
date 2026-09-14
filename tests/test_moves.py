"""The prompt-visible surface of the loop: move labels, retrieval keys, and
the submit tool's wording — pinned because a rerun must see the same text."""
from dataclasses import dataclass, field
from pathlib import Path

from arloop.budget import Budget
from arloop.llm import ScriptedLLM, submit_tool
from arloop.tokens import tokens_to_words
from arloop.loop import RunConfig, TaskSpec, run_task
from arloop.runner import CallableRunner, Submission
from arloop.trace import read_events

CODE = {"tool_args": {"code": "print(1)\n", "approach": "prints one"}}


@dataclass
class FakeScore:
    value: float
    valid: bool = True
    is_higher_better: bool = True
    details: dict = field(default_factory=lambda: {"feedback": "1/1 cases scored"})


class RecordingView:
    def __init__(self):
        self.queries = []

    def query(self, text):
        self.queries.append(text)
        return []


def _run(tmp_path, responses, exit_codes, view=None):
    exits = list(exit_codes)

    def execute(workspace: Path, command: str, timeout_s: int) -> Submission:
        (workspace / "submission.py").write_text("print(1)\n")
        code = exits.pop(0)
        return Submission(code, "Traceback (most recent call last):\nboom\n" if code else "", 0.1)

    task = TaskSpec("t1", "GOAL", "EVAL")
    config = RunConfig(run_id="r1", config_hash="abc", seed=0,
                       traces_root=tmp_path / "traces",
                       workspace_root=tmp_path / "ws")
    out = run_task(task, view, CallableRunner(execute),
                   Budget("attempts", len(responses)), config,
                   ScriptedLLM(list(responses)),
                   public_eval=lambda p: FakeScore(1.0),
                   validate=lambda p: (True, None), grade=None)
    events = list(read_events(out.run_dir))
    return out, events


def test_debug_after_a_malformed_reply_is_a_formatting_debug(tmp_path):
    view = RecordingView()
    _, events = _run(tmp_path, [{"tool_args_raw": "{not json"}, CODE, CODE],
                     [0, 0], view)
    moves = [e.payload["move"] for e in events if e.type == "llm_call"]
    assert moves == ["draft", "debug:formatting", "improve"]
    # the protocol message is not a retrieval key: task text only
    assert view.queries[1] == "GOAL\nEVAL"
    assert view.queries[2].startswith("GOAL\nEVAL\n")


def test_debug_after_a_crash_is_a_code_debug_keyed_on_the_error(tmp_path):
    view = RecordingView()
    _, events = _run(tmp_path, [CODE, CODE], [1, 0], view)
    moves = [e.payload["move"] for e in events if e.type == "llm_call"]
    assert moves == ["draft", "debug:code"]
    assert view.queries[1].startswith("exit 1\nTraceback")
    history = [e for e in events if e.type == "llm_call"][1].payload["prompt"]
    assert "1. draft: failed — APPROACH: prints one" in history


def test_submit_tool_wording_is_pinned():
    tool = submit_tool(300)["function"]
    assert tool["description"] == ("Submit the complete solution.py and a "
                                   "one-sentence summary of the approach it takes.")
    assert tool["parameters"]["properties"]["approach"]["description"] == (
        f"A description of this solution's approach. Use about "
        f"{tokens_to_words(300)} words: explain the algorithm/design, the key "
        "decisions and why you made them, and any trade-offs or caveats.")
    assert "approach" not in submit_tool(0)["function"]["parameters"]["properties"]
