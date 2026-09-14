"""GridConfig: what the arm hash covers, what validate refuses, and that the
27 generated cells all load as distinct arms."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from arloop.grid.config import GridConfig, config_hash, load_config, resolved_prompts

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "experiments" / "final_grid" / "cells"


def base(**over) -> GridConfig:
    fields = dict(name="arm", tasks=("ahc001",), seeds=(0,), model="qwen27b")
    return GridConfig(**{**fields, **over})


def test_hash_excludes_name_tasks_seeds_ops():
    cfg = base()
    for over in ({"name": "other"}, {"tasks": ("ahc002", "ahc003")},
                 {"seeds": (0, 1, 2)}, {"ops": {"driver": {"max_workers": 9}}}):
        assert config_hash(replace(cfg, **over)) == config_hash(cfg)


def test_hash_covers_scientific_fields():
    cfg = base()
    for over in ({"model": "other"}, {"temperature": 0.7},
                 {"history_write_tokens": 300}, {"chars_per_token": 4.0},
                 {"branch_policy": "last"}, {"exec_timeout_s": 600},
                 {"code_budget_chars": 10_000}, {"sandbox": False},
                 {"run_budget": {"type": "tokens", "limit": 200_000}}):
        assert config_hash(replace(cfg, **over)) != config_hash(cfg)


def test_hash_includes_resolved_prompts():
    cfg = base()
    override = {"system": "a different system prompt"}
    assert config_hash(replace(cfg, prompts=override)) != config_hash(cfg)
    # the defaults are hashed as their full text, not as "no overrides": an
    # explicit override equal to the default is the SAME arm
    same = {"system": resolved_prompts(cfg).system}
    assert config_hash(replace(cfg, prompts=same)) == config_hash(cfg)


def test_validate_refuses_budget_without_thinking():
    base(thinking=True, thinking_token_budget=3000).validate()
    with pytest.raises(ValueError, match="requires thinking"):
        base(thinking_token_budget=3000).validate()
    with pytest.raises(ValueError, match="positive int"):
        base(thinking=True, thinking_token_budget=0).validate()


def test_validate_refuses_bad_fields():
    with pytest.raises(ValueError, match="history_write_tokens"):
        base(history_write_tokens=-1).validate()
    with pytest.raises(ValueError, match="unknown budget type"):
        base(run_budget={"type": "wall_seconds", "limit": 10}).validate()
    with pytest.raises(ValueError, match="branch_policy"):
        base(branch_policy="first").validate()
    with pytest.raises(ValueError, match="tasks and seeds"):
        base(seeds=()).validate()


def memory_block(**over) -> dict:
    block = {"bank": "b", "embed_model": "e", "reranker_model": "r",
             "retrieval_tokens": 4000, "loto": True}
    return {**block, **over}


def test_validate_memory_block():
    base(memory=memory_block()).validate()
    with pytest.raises(ValueError, match="retrieval_tokens must be a positive int"):
        base(memory=memory_block(retrieval_tokens=0)).validate()
    with pytest.raises(ValueError, match="retrieval_tokens must be a positive int"):
        base(memory=memory_block(retrieval_tokens=-1)).validate()
    with pytest.raises(ValueError, match="missing 'bank'"):
        base(memory={k: v for k, v in memory_block().items() if k != "bank"}).validate()
    with pytest.raises(ValueError, match="loto"):
        base(memory=memory_block(loto="yes")).validate()


def test_load_config_freezes_sequences(tmp_path):
    path = tmp_path / "arm.yaml"
    path.write_text(yaml.safe_dump(
        {"name": "arm", "tasks": ["ahc001", "ahc002"], "seeds": [0, 1],
         "model": "qwen27b", "thinking": True, "thinking_token_budget": 3000}))
    cfg = load_config(path)
    assert cfg.tasks == ("ahc001", "ahc002") and cfg.seeds == (0, 1)
    assert cfg.thinking_token_budget == 3000


def test_generated_cells_load_and_are_distinct():
    if not CELLS.exists():
        pytest.skip("cells not generated")
    paths = sorted(CELLS.glob("final-grid-r*-w*-x*.yaml"))
    assert len(paths) == 27
    hashes = {}
    for path in paths:
        cfg = load_config(path)
        assert cfg.name == path.stem
        hashes[config_hash(cfg)] = path.name
    assert len(hashes) == 27


def test_generated_cells_carry_their_axis_levels():
    if not CELLS.exists():
        pytest.skip("cells not generated")
    for path in sorted(CELLS.glob("final-grid-r*-w*-x*.yaml")):
        r, w, x = (int(p[1:]) for p in path.stem.split("-")[2:5])
        cfg = load_config(path)
        assert cfg.thinking_token_budget == (r or None)
        assert cfg.thinking is (r > 0)
        assert cfg.history_write_tokens == w
        assert (cfg.memory or {}).get("retrieval_tokens") == (x or None)


def test_smoke_cell_loads():
    load_config(REPO / "experiments" / "final_grid" / "smoke.yaml")


def test_make_cells_is_reproducible(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "experiments" / "final_grid"))
    import make_cells

    monkeypatch.setattr(make_cells, "CELLS_DIR", tmp_path)
    for path in make_cells.write_cells():
        assert path.read_text() == (CELLS / path.name).read_text()


def test_ops_embedding_server_is_not_part_of_the_arm_identity(tmp_path):
    cell = load_config(Path(__file__).parents[1] / "experiments/final_grid/cells"
                       / "final-grid-r0-w100-x4000.yaml")
    flipped = replace(cell, ops={**cell.ops, "embedding": {"server": False}})
    assert config_hash(flipped) == config_hash(cell)
