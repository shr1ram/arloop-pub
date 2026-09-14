"""build_corpus over a synthetic multi-cell trace store."""
from __future__ import annotations

import pytest

from arloop.grid.build_corpus import DEFAULT_CELLS, build_corpus, collect_runs
from arloop.llm import ScriptedLLM
from arloop.memory.bank import load_bank
from arloop.memory.writer import WriteParams
from arloop.trace import RunManifest, TraceWriter, run_dir_for

PARAMS = WriteParams(dedupe="none")


def write_run(root, cell, task_id, run_id, seed, config_hash="cfg0", status="ok"):
    manifest = RunManifest(run_id=run_id, task_id=task_id, seed=seed,
                           config_hash=config_hash, model="scripted", temperature=0.0)
    run_dir = run_dir_for(root / cell, task_id, run_id)
    with TraceWriter(run_dir, manifest) as tw:
        tw.emit("task_presented", {"goal": f"goal {task_id}", "eval": "higher"}, 0)
        tw.emit("llm_call", {"move": "draft"}, 0)
        tw.emit("code_written", {"path": "solution.py", "content": "print(1)"}, 0)
        tw.emit("execution_result", {"classification": "success"}, 0)
        tw.emit("score", {"proxy_score": 1.0}, 0)
        tw.close(status, proxy_score=1.0)
    return run_dir


def script(n_runs):
    """step, fold, contextualize per single-attempt run."""
    return [f"r{i}-{name}" for i in range(n_runs) for name in ("step", "fold", "ctx")]


def test_default_cells_are_the_nine_x0_cells():
    assert len(DEFAULT_CELLS) == 9
    assert "final-grid-r0-w100-x0" in DEFAULT_CELLS
    assert "final-grid-r9000-w900-x0" in DEFAULT_CELLS
    assert all(c.endswith("-x0") for c in DEFAULT_CELLS)


def test_collect_runs_pools_cells_and_skips_non_ok(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-1", 0)
    write_run(root, "cell-b", "taskB", "run-2", 0, config_hash="cfg1")
    write_run(root, "cell-b", "taskC", "run-3", 0, config_hash="cfg1",
              status="failed")
    runs = collect_runs(root, ["cell-a", "cell-b"])
    assert [m.run_id for m, _ in runs] == ["run-1", "run-2"]


def test_collect_runs_keeps_the_first_run_id_per_cell_key(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-b", 0)
    write_run(root, "cell-a", "taskA", "run-a", 0)      # same (cfg, task, seed)
    runs = collect_runs(root, ["cell-a"])
    assert [m.run_id for m, _ in runs] == ["run-a"]


def test_build_corpus_writes_a_bank_with_derivation(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-1", 0)
    write_run(root, "cell-b", "taskB", "run-2", 0, config_hash="cfg1")
    out = tmp_path / "bank"
    llm = ScriptedLLM(responses=script(2))
    bank = build_corpus(root, ["cell-a", "cell-b"], out, llm, PARAMS, workers=1)

    assert len(bank.cases) == 2
    assert {c.task_id for c in bank.cases} == {"taskA", "taskB"}
    d = bank.manifest["derivation"]
    assert d["source_cells"] == ["cell-a", "cell-b"]
    assert len(d["corpus_hash"]) == 12
    assert d["write_params"]["c_step"] == PARAMS.c_step
    assert d["llm"] == {"model": "scripted", "temperature": 0.0, "seed": 0}
    assert load_bank(out).bank_id == bank.bank_id


def test_build_corpus_refuses_an_existing_bank_without_force(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-1", 0)
    out = tmp_path / "bank"
    build_corpus(root, ["cell-a"], out, ScriptedLLM(responses=script(1)),
                 PARAMS, workers=1)
    with pytest.raises(FileExistsError):
        build_corpus(root, ["cell-a"], out, ScriptedLLM(responses=script(1)),
                     PARAMS, workers=1)
    bank = build_corpus(root, ["cell-a"], out, ScriptedLLM(responses=script(1)),
                        PARAMS, workers=1, force=True)
    assert len(bank.cases) == 1


def test_build_corpus_refuses_an_empty_corpus(tmp_path):
    with pytest.raises(RuntimeError, match="no ok runs"):
        build_corpus(tmp_path / "traces", ["cell-a"], tmp_path / "bank",
                     ScriptedLLM(responses=[]), PARAMS, workers=1)


def test_build_corpus_embeds_when_a_retriever_is_given(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-1", 0)
    seen = []

    class FakeEmbedder:
        def embed_bank(self, bank):
            seen.append(bank.bank_id)

    bank = build_corpus(root, ["cell-a"], tmp_path / "bank",
                        ScriptedLLM(responses=script(1)), PARAMS, workers=1,
                        embed_retriever=FakeEmbedder())
    assert seen == [bank.bank_id]


def test_node_cache_resumes_a_build(tmp_path):
    root = tmp_path / "traces"
    write_run(root, "cell-a", "taskA", "run-1", 0)
    nodes = tmp_path / "nodes"
    first = build_corpus(root, ["cell-a"], tmp_path / "b1",
                         ScriptedLLM(responses=script(1)), PARAMS,
                         nodes_dir=nodes, workers=1)
    # an empty script would raise if anything had to be regenerated
    second = build_corpus(root, ["cell-a"], tmp_path / "b2",
                          ScriptedLLM(responses=[]), PARAMS,
                          nodes_dir=nodes, workers=1)
    assert second.bank_id == first.bank_id
