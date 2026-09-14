"""EpisodeWriter over a synthetic trace store, driven by a ScriptedLLM."""
from __future__ import annotations

import pytest

from arloop.llm import ScriptedLLM
from arloop.memory.writer import (EpisodeWriter, NodeCache, WriteParams,
                                  WritePrompts, attempt_slice, best_line,
                                  run_skeleton, write_prompts_hash)
from arloop.trace import RunManifest, TraceWriter, iter_runs, run_dir_for


def write_run(root, task_id, run_id, seed, attempts):
    """One run of `attempts` (move, code, classification, excerpt, score)."""
    manifest = RunManifest(run_id=run_id, task_id=task_id, seed=seed,
                           config_hash="cfg0", model="scripted", temperature=0.0)
    run_dir = run_dir_for(root, task_id, run_id)
    best = None
    with TraceWriter(run_dir, manifest) as tw:
        tw.emit("task_presented", {"goal": f"goal of {task_id}",
                                   "eval": "higher is better"}, 0)
        for n, (move, code, cls, excerpt, score) in enumerate(attempts):
            tw.emit("llm_call", {"move": move}, n)
            tw.emit("code_written", {"path": f"attempt_{n}/solution.py",
                                     "content": code}, n)
            tw.emit("execution_result", {"classification": cls,
                                         "excerpt": excerpt}, n)
            tw.emit("score", {"proxy_score": score}, n)
            if score is not None and (best is None or score > best):
                best = score
        tw.close("ok", proxy_score=best)
    return run_dir


@pytest.fixture
def store(tmp_path):
    """Two tasks x one run each; the second task's run has two attempts."""
    root = tmp_path / "traces" / "cell-a"
    write_run(root, "taskA", "run-a", 0,
              [("draft", "print(1)", "success", "", 1.0)])
    write_run(root, "taskB", "run-b", 0,
              [("draft", "print(2)", "agent_fixable", "Traceback: boom", None),
               ("debug", "print(3)", "success", "", 2.5)])
    return root


def corpus(root):
    return list(iter_runs(root, status="ok"))


def make_writer(root, responses, tmp_path, **kw):
    llm = ScriptedLLM(responses=list(responses))
    kw.setdefault("workers", 1)          # ScriptedLLM's script order is the contract
    kw.setdefault("params", WriteParams(dedupe="none"))
    return EpisodeWriter(corpus(root), llm, nodes_dir=kw.pop("nodes_dir", None), **kw), llm


# The writer issues, per run in run_id order: one `step` per attempt, one
# `fold` per window, then one `contextualize`.
SCRIPT_A = ["step note A0", "fold A", "context A"]
SCRIPT_B = ["step note B0", "step note B1", "fold B", "context B"]


def test_mechanical_helpers(store):
    (m, run_dir), = [r for r in corpus(store) if r[0].task_id == "taskB"]
    from collections import defaultdict

    from arloop.trace import read_events
    groups = defaultdict(list)
    for e in read_events(run_dir):
        groups[e.attempt].append(e)
    groups = dict(groups)

    text, span = attempt_slice(groups[0])
    assert "move: draft" in text and "print(2)" in text
    assert "outcome: agent_fixable" in text and "Traceback: boom" in text
    assert span[0] <= span[1]

    skel = run_skeleton(m, groups)
    assert skel.splitlines()[0] == "task: taskB (benchmark ale_bench)"
    assert "attempt 1: move=debug outcome=success proxy=2.5" in skel
    assert skel.splitlines()[-1].startswith("final: proxy=")
    assert best_line(groups) == "Best: attempt 1 -> 2.5"


def test_best_line_when_nothing_scored(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskC", "run-c", 0,
              [("draft", "x", "agent_fixable", "err", None)])
    from collections import defaultdict

    from arloop.trace import read_events
    groups = defaultdict(list)
    for e in read_events(next(iter_runs(root))[1]):
        groups[e.attempt].append(e)
    assert best_line(dict(groups)) == "Best: none - no attempt scored"


def test_one_case_per_run_with_footer_and_context(store, tmp_path):
    writer, llm = make_writer(store, SCRIPT_A + SCRIPT_B, tmp_path)
    cases = writer.cases()
    assert len(cases) == 2
    by_task = {c.task_id: c for c in cases}
    assert by_task["taskA"].content == "fold A\nBest: attempt 0 -> 1.0"
    assert by_task["taskB"].content == "fold B\nBest: attempt 1 -> 2.5"
    assert by_task["taskA"].meta["retrieval_context"] == "context A"
    assert by_task["taskB"].meta["retrieval_context"] == "context B"
    assert not llm.responses                       # the script is exactly consumed

    prov = by_task["taskB"].provenance
    assert len(prov) == 1 and prov[0]["run_id"] == "run-b"
    lo, hi = prov[0]["seq_span"]
    assert lo == 0 and hi >= 8                      # spans the whole run


def test_step_prompt_carries_the_task_and_the_slice(store, tmp_path):
    writer, llm = make_writer(store, SCRIPT_A + SCRIPT_B, tmp_path)
    writer.cases()
    step_calls = [u for _, u in llm.calls if "One attempt from" in u]
    assert len(step_calls) == 3
    assert "goal of taskA" in step_calls[0]
    assert "EVALUATION: higher is better" in step_calls[0]
    assert "print(1)" in step_calls[0]


def test_fold_windows_bound_the_call(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskD", "run-d", 0,
              [("draft", f"print({i})", "success", "", float(i)) for i in range(5)])
    params = WriteParams(fold_window=2, dedupe="none")
    llm = ScriptedLLM(responses=[f"step {i}" for i in range(5)]
                      + ["fold 1", "fold 2", "fold 3", "ctx"])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    cases = writer.cases()
    assert len(cases) == 1
    assert cases[0].content == "fold 3\nBest: attempt 4 -> 4.0"
    folds = [u for _, u in llm.calls if "running summary" in u]
    assert len(folds) == 3                          # ceil(5 / 2)
    assert "(start of episode)" in folds[0]
    assert "fold 1" in folds[1]                     # the summary is carried forward
    # each fold sees exactly its window's notes, never the whole run
    assert "[attempt 0]" in folds[0] and "[attempt 2]" not in folds[0]


def test_node_cache_makes_a_second_build_free(store, tmp_path):
    nodes = tmp_path / "nodes"
    writer, llm = make_writer(store, SCRIPT_A + SCRIPT_B, tmp_path, nodes_dir=nodes)
    first = writer.cases()
    assert writer.llm_calls == 7

    writer2, llm2 = make_writer(store, [], tmp_path, nodes_dir=nodes)
    second = writer2.cases()
    assert writer2.llm_calls == 0                   # ScriptedLLM would have raised
    assert writer2.cache.hits == 7
    assert [c.content for c in second] == [c.content for c in first]


def test_over_cap_note_takes_one_shorten_call(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskE", "run-e", 0, [("draft", "x", "success", "", 1.0)])
    params = WriteParams(c_step=20, fold_window=2, dedupe="none")
    llm = ScriptedLLM(responses=["s" * 50, "short note", "fold", "ctx"])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    cases = writer.cases()
    assert len(llm.calls) == 4
    assert "Shorten the following text" in llm.calls[1][1]
    assert "short note" in [u for _, u in llm.calls if "running summary" in u][0]
    assert cases[0].content.startswith("fold")


def test_over_cap_note_still_over_is_dropped_whole(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskF", "run-f", 0, [("draft", "x", "success", "", 1.0)])
    params = WriteParams(c_step=20, fold_window=2, dedupe="none")
    llm = ScriptedLLM(responses=["s" * 50, "t" * 60, "fold", "ctx"])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    writer.cases()
    fold_prompt = [u for _, u in llm.calls if "running summary" in u][0]
    assert "[attempt 0]" not in fold_prompt         # dropped whole, never cut
    assert "attempt 0: move=draft" in fold_prompt   # the skeleton still covers it


def test_dropped_episode_note_yields_no_case(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskG", "run-g", 0, [("draft", "x", "success", "", 1.0)])
    params = WriteParams(c_step=10, fold_window=2, dedupe="none")
    llm = ScriptedLLM(responses=["note", "f" * 50, "g" * 60])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    assert writer.cases() == []


def test_retrieval_context_beyond_two_n_is_dropped(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskH", "run-h", 0, [("draft", "x", "success", "", 1.0)])
    params = WriteParams(retrieval_context_chars=10, dedupe="none")
    llm = ScriptedLLM(responses=["note", "fold", "c" * 40, "d" * 30])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    assert writer.cases()[0].meta["retrieval_context"] == ""


def test_dedupe_merge_replaces_the_same_task_neighbour(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskI", "run-i1", 0, [("draft", "x", "success", "", 1.0)])
    write_run(root, "taskI", "run-i2", 1, [("draft", "y", "success", "", 2.0)])
    params = WriteParams(dedupe="llm")
    llm = ScriptedLLM(responses=[
        "note 1", "greedy packing beat random on this grid", "ctx 1",
        "note 2", "greedy packing beat random here too", "ctx 2",
        "MERGE 1\ngreedy packing beat random on both runs of this grid",
    ])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    cases = writer.cases()
    assert len(cases) == 1
    assert cases[0].content.startswith("greedy packing beat random on both runs")
    # provenance is the union of the two runs
    assert {p["run_id"] for p in cases[0].provenance} == {"run-i1", "run-i2"}
    assert cases[0].meta["retrieval_context"] == "ctx 1"     # the anchor's


def test_dedupe_no_merge_keeps_both(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskJ", "run-j1", 0, [("draft", "x", "success", "", 1.0)])
    write_run(root, "taskJ", "run-j2", 1, [("draft", "y", "success", "", 2.0)])
    llm = ScriptedLLM(responses=[
        "note 1", "beam search overran the time limit", "ctx 1",
        "note 2", "beam search overran the time limit", "ctx 2",
        "NO_MERGE",
    ])
    writer = EpisodeWriter(corpus(root), llm, params=WriteParams(dedupe="llm"), workers=1)
    assert len(writer.cases()) == 2


def test_dedupe_never_merges_across_tasks(store, tmp_path):
    writer, llm = make_writer(store, SCRIPT_A + SCRIPT_B, tmp_path,
                              params=WriteParams(dedupe="llm"))
    cases = writer.cases()
    assert len(cases) == 2
    # different tasks -> empty neighbour pool -> no merge call is ever issued
    assert not any("about to enter" in u for _, u in llm.calls)


def test_merge_over_cap_is_rejected(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskK", "run-k1", 0, [("draft", "x", "success", "", 1.0)])
    write_run(root, "taskK", "run-k2", 1, [("draft", "y", "success", "", 2.0)])
    params = WriteParams(c_step=10, fold_window=2, dedupe="llm")   # episode cap 20
    llm = ScriptedLLM(responses=[
        "note 1", "lesson one", "ctx 1",
        "note 2", "lesson two", "ctx 2",
        "MERGE 1\n" + "m" * 50,
    ])
    writer = EpisodeWriter(corpus(root), llm, params=params, workers=1)
    assert len(writer.cases()) == 2                 # both kept, nothing truncated


def test_heldout_score_never_reaches_a_prompt(store, tmp_path):
    root = store.parent / "cell-b"
    manifest = RunManifest(run_id="run-x", task_id="taskX", seed=0,
                           config_hash="cfg0", model="scripted", temperature=0.0)
    run_dir = run_dir_for(root, "taskX", "run-x")
    with TraceWriter(run_dir, manifest) as tw:
        tw.emit("task_presented", {"goal": "g", "eval": "e"}, 0)
        tw.emit("llm_call", {"move": "draft"}, 0)
        tw.emit("execution_result", {"classification": "success"}, 0)
        tw.emit("score", {"proxy_score": 1.0}, 0)
        tw.close("ok", proxy_score=1.0, heldout_score=999.5)
    llm = ScriptedLLM(responses=["note", "fold", "ctx"])
    EpisodeWriter(corpus(root), llm, params=WriteParams(dedupe="none"),
                  workers=1).cases()
    assert not any("999.5" in u for _, u in llm.calls)


def test_retry_wall_raises_immediately_on_a_4xx(tmp_path):
    root = tmp_path / "t"
    write_run(root, "taskL", "run-l", 0, [("draft", "x", "success", "", 1.0)])

    class Boom:
        model = "boom"

        def complete(self, system, user):
            e = RuntimeError("bad request")
            e.status_code = 400
            raise e

    writer = EpisodeWriter(corpus(root), Boom(), params=WriteParams(dedupe="none"),
                           workers=1, retry_wall_s=1e6, retry_sleep_s=1e6)
    with pytest.raises(RuntimeError, match="bad request"):
        writer.cases()


def test_node_cache_key_covers_the_whole_derivation():
    base = NodeCache.key(("m", 0.0, 0), "step", "sys", "tmpl", ["in"])
    assert base != NodeCache.key(("m", 0.5, 0), "step", "sys", "tmpl", ["in"])
    assert base != NodeCache.key(("m", 0.0, 0), "step", "SYS", "tmpl", ["in"])
    assert base != NodeCache.key(("m", 0.0, 0), "fold", "sys", "tmpl", ["in"])
    assert base == NodeCache.key(("m", 0.0, 0), "step", "sys", "tmpl", ["in"])


def test_write_prompts_hash_is_field_complete():
    base = write_prompts_hash(WritePrompts())
    assert base == write_prompts_hash(WritePrompts())
    assert base != write_prompts_hash(WritePrompts(shorten="Shorten {text} {cap}"))


def test_write_prompts_refuse_taskless_text():
    with pytest.raises(ValueError, match="task"):
        WritePrompts(step="no placeholder here {slice} {cap}")


def test_fold_window_must_reduce():
    with pytest.raises(ValueError, match="fold_window"):
        WriteParams(fold_window=1)


def test_the_cache_key_is_derived_from_the_client(tmp_path):
    """A NodeCache key that disagreed with the client would serve one model's
    notes to another; it is taken from the client, not passed beside it."""
    from arloop.llm import ScriptedLLM
    from arloop.memory.writer import EpisodeWriter
    llm = ScriptedLLM([], model="some-model")
    llm.temperature = 0.7
    writer = EpisodeWriter([], llm, workers=1)
    assert writer.llm_key == ("some-model", 0.7, 0)
