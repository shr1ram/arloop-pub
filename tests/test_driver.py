"""The grid driver over a fake cell command: the ledger decides success, the
sweep is idempotent, failures requeue, and claims arbitrate across drivers."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from arloop.grid.config import GridConfig, config_hash
from arloop.grid.driver import _claim_path, ledger_status, run_grid

WRITE_OK = (
    "import json, pathlib, sys\n"
    "root, cfg, task, seed = sys.argv[1:5]\n"
    "d = pathlib.Path(root) / task / f'{task}-s{seed}-run'\n"
    "d.mkdir(parents=True, exist_ok=True)\n"
    "(d / 'manifest.json').write_text(json.dumps("
    "{'run_id': d.name, 'task_id': task, 'seed': int(seed), "
    "'config_hash': cfg, 'model': 'fake', 'temperature': 0.0, 'status': 'ok'}))\n"
)


def make_cfg(tasks=("ahc001", "ahc002"), seeds=(0,)) -> GridConfig:
    return GridConfig(name="arm", tasks=tasks, seeds=seeds, model="qwen27b")


@pytest.fixture()
def grid(tmp_path):
    """A traces root plus the grid dir the driver derives from it."""
    traces = tmp_path / "data" / "traces" / "arm"
    traces.mkdir(parents=True)
    return traces


def ok_cmd(traces_root: Path, cfg_hash: str, script: Path):
    script.write_text(WRITE_OK)

    def build(task_id: str, seed: int) -> list[str]:
        return [sys.executable, str(script), str(traces_root), cfg_hash,
                task_id, str(seed)]
    return build


def test_sweep_writes_ok_cells_then_is_idempotent(grid, tmp_path):
    cfg = make_cfg()
    cmd = ok_cmd(grid, config_hash(cfg), tmp_path / "cell.py")
    first = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=2,
                     poll_s=0.01, cell_cmd=cmd, log=lambda *_: None)
    assert (first["cells"], first["ok"], first["incomplete"]) == (2, 2, 0)

    dispatched = []

    def spy(task_id, seed):
        dispatched.append((task_id, seed))
        return cmd(task_id, seed)

    second = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=2,
                      poll_s=0.01, cell_cmd=spy, log=lambda *_: None)
    assert dispatched == []
    assert (second["already_ok"], second["ran"], second["incomplete"]) == (2, 0, 0)

    summary = json.loads((grid.parent / "grid" / config_hash(cfg) /
                          "summary.json").read_text())
    assert summary["config_hash"] == config_hash(cfg)


def test_failing_cell_is_requeued_then_reported(grid, tmp_path):
    cfg = make_cfg(tasks=("ahc001",))
    calls = []

    def failing(task_id, seed):
        calls.append((task_id, seed))
        return [sys.executable, "-c", "raise SystemExit(3)"]

    summary = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=1,
                       poll_s=0.01, max_attempts=3, cell_cmd=failing,
                       log=lambda *_: None)
    assert len(calls) == 3
    assert (summary["ok"], summary["failed"], summary["incomplete"]) == (0, 1, 1)
    assert summary["incomplete_cells"] == ["ahc001-s0"]


def test_zero_exit_without_an_ok_manifest_is_a_failure(grid, tmp_path):
    cfg = make_cfg(tasks=("ahc001",))
    summary = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=1,
                       poll_s=0.01, max_attempts=1,
                       cell_cmd=lambda t, s: [sys.executable, "-c", "pass"],
                       log=lambda *_: None)
    assert (summary["ok"], summary["failed"]) == (0, 1)


def test_fresh_claim_by_another_driver_defers_the_cell(grid, tmp_path):
    cfg = make_cfg(tasks=("ahc001",))
    cfg_hash = config_hash(cfg)
    grid_dir = grid.parent / "grid" / cfg_hash
    grid_dir.mkdir(parents=True)
    _claim_path(grid_dir, ("ahc001", 0)).write_text("otherbox:1234")

    dispatched = []
    summary = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=1,
                       poll_s=0.01,
                       cell_cmd=lambda t, s: dispatched.append((t, s)) or [
                           sys.executable, "-c", "pass"],
                       log=lambda *_: None)
    assert dispatched == []
    assert (summary["ran"], summary["incomplete"]) == (0, 1)


def test_stale_claim_is_stolen(grid, tmp_path):
    cfg = make_cfg(tasks=("ahc001",))
    cfg_hash = config_hash(cfg)
    grid_dir = grid.parent / "grid" / cfg_hash
    grid_dir.mkdir(parents=True)
    claim = _claim_path(grid_dir, ("ahc001", 0))
    claim.write_text("deadbox:1")
    old = time.time() - 10_000
    os.utime(claim, (old, old))

    summary = run_grid(cfg, tmp_path / "arm.yaml", grid, max_workers=1,
                       poll_s=0.01, claim_stale_s=900.0,
                       cell_cmd=ok_cmd(grid, cfg_hash, tmp_path / "cell.py"),
                       log=lambda *_: None)
    assert (summary["ok"], summary["incomplete"]) == (1, 0)
    assert not claim.exists()


def test_running_manifest_without_a_claim_reads_as_failed(grid, tmp_path):
    cfg = make_cfg(tasks=("ahc001",))
    cfg_hash = config_hash(cfg)
    run_dir = grid / "ahc001" / "ahc001-s0-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": run_dir.name, "task_id": "ahc001", "seed": 0,
         "config_hash": cfg_hash, "model": "fake", "temperature": 0.0,
         "status": "running"}))
    grid_dir = grid.parent / "grid" / cfg_hash
    grid_dir.mkdir(parents=True)
    assert ledger_status(grid, cfg_hash, grid_dir) == {("ahc001", 0): "failed"}

    # a live claim makes the same manifest read as running again
    _claim_path(grid_dir, ("ahc001", 0)).write_text("box:1")
    assert ledger_status(grid, cfg_hash, grid_dir) == {("ahc001", 0): "running"}
