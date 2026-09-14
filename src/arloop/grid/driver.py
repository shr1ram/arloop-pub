"""The grid driver: run every (task, seed) cell of one arm as a subprocess.

    python -m arloop.grid.driver <arm.yaml> [--max-workers N]

The run manifests ARE the ledger — a cell is done iff an ok manifest exists
for (config_hash, task, seed) — so re-running an arm is idempotent. Claim
files on the shared filesystem make concurrent drivers on different boxes safe.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from arloop.grid.config import GridConfig, config_hash, load_config
from arloop.grid.run_cell import export_data_env, resolve_path
from arloop.sandbox import THREAD_CAP_KEYS, derive_thread_cap
from arloop.trace import iter_runs

CellKey = tuple[str, int]   # (task_id, seed)

#: what dispatch() did with a cell, and what the sweep loop must do next
STARTED = "started"          # running now: pop it from the queue
SKIPPED = "skipped"          # not ours to run (claimed elsewhere, already ok,
                             # or given up on): pop it, never retry
RETRY_LATER = "retry_later"  # transient dispatch failure: leave it queued

#: a claim not heartbeat-touched for this long belongs to a dead driver
DEFAULT_CLAIM_STALE_S = 900.0

#: how long a cell gets to die on its own after SIGTERM before the SIGKILL
TERM_GRACE_S = 10.0


def _claim_path(grid_dir: Path, cell: CellKey) -> Path:
    return grid_dir / f"{cell[0]}-s{cell[1]}.claim"


def ledger_status(traces_root: Path, cfg_hash: str, grid_dir: Path,
                  claim_stale_s: float = DEFAULT_CLAIM_STALE_S) -> dict[CellKey, str]:
    """One status per cell of this arm: ok beats running beats failed.

    A `running` manifest is trusted only while its claim is alive. A killed
    cell can never close its own manifest, so a claim-less or stale-claimed
    `running` is a DEAD run and reads as failed — re-dispatchable at once
    rather than blocking the cell forever.
    """
    best: dict[CellKey, str] = {}
    rank = {"ok": 3, "running": 2, "failed": 1}
    for manifest, _ in iter_runs(traces_root):
        if manifest.config_hash != cfg_hash:
            continue
        status = manifest.status
        key = (manifest.task_id, manifest.seed)
        if status == "running":
            try:
                if time.time() - _claim_path(grid_dir, key).stat().st_mtime > claim_stale_s:
                    status = "failed"
            except OSError:
                status = "failed"
        if rank.get(status, 0) > rank.get(best.get(key, ""), 0):
            best[key] = status
    return best


def default_cell_cmd(config_path: Path) -> Callable[[str, int], list[str]]:
    """The command that runs one cell of the arm at config_path."""
    def build(task_id: str, seed: int) -> list[str]:
        return [sys.executable, "-m", "arloop.grid.run_cell",
                "--config", str(config_path),
                "--task", task_id, "--seed", str(seed)]
    return build


def cell_env(max_workers: int) -> dict[str, str]:
    """The cell's environment with its thread caps already set.

    BLAS/OpenMP and torch size their pools at import from the environment as
    it was at exec, so only the parent can cap a cell. ARLOOP_CPU_CAP
    publishes the number the driver actually launched under, which the cell
    both enforces and discloses in its prompt.
    """
    cap = str(derive_thread_cap(max_workers))
    return dict(os.environ, ARLOOP_CPU_CAP=cap,
                **{key: cap for key in THREAD_CAP_KEYS})


def _try_claim(path: Path, stale_s: float) -> bool:
    """Win this cell across boxes, stealing a claim whose driver died.

    The running manifest alone cannot prevent double-dispatch (each run writes
    a new run_id dir, so two drivers scanning in the same window both see the
    cell missing). O_CREAT|O_EXCL on one fixed path per cell is the election;
    the holder heartbeats its mtime every poll.
    """
    for _ in range(2):          # second pass after stealing a stale claim
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{socket.gethostname()}:{os.getpid()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime <= stale_s:
                    return False        # live claim: another driver owns it
                # steal by RENAME, not unlink: two drivers judging the same
                # claim stale must not both win — rename is atomic, so the
                # second one fails here instead of deleting the first's
                # fresh claim
                stale = path.with_name(f"{path.name}.stale{os.getpid()}")
                os.rename(path, stale)
                stale.unlink()
            except OSError:
                return False            # vanished, refreshed or stolen under us
    return False


def _release_claim(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _kill_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the cell's process group, wait grace_s, then SIGKILL it.

    Group-wide because the agent subprocesses are the expensive part:
    terminating the run_cell leader alone leaves solution.py running.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=grace_s)
        return
    except (ProcessLookupError, PermissionError):
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()                     # reap, so the exit code is recorded


def run_grid(cfg: GridConfig, config_path: Path, traces_root: Path, *,
             max_workers: int = 1, poll_s: float = 1.0,
             job_timeout_s: float = 43200.0, max_attempts: int = 2,
             claim_stale_s: float = DEFAULT_CLAIM_STALE_S,
             cell_cmd: Optional[Callable[[str, int], list[str]]] = None,
             grid_root: Optional[Path] = None, log=print) -> dict:
    """Sweep one arm's cells to completion and return (and write) its summary.

    Claims, per-cell logs and summary.json live under <grid_root>/<config_hash>
    (default: a `grid` dir beside the traces root).
    """
    cfg_hash = config_hash(cfg)
    traces_root = Path(traces_root)
    grid_dir = Path(grid_root or traces_root.parent / "grid") / cfg_hash
    grid_dir.mkdir(parents=True, exist_ok=True)
    build_cmd = cell_cmd or default_cell_cmd(Path(config_path))
    env = cell_env(max_workers)

    def status() -> dict[CellKey, str]:
        return ledger_status(traces_root, cfg_hash, grid_dir, claim_stale_s)

    ledger = status()
    cells = [(t, s) for t in cfg.tasks for s in cfg.seeds]
    already_ok = [c for c in cells if ledger.get(c) == "ok"]
    elsewhere = [c for c in cells if ledger.get(c) == "running"]
    pending = [c for c in cells if c not in already_ok and c not in elsewhere]
    log(f"[grid] {cfg.name} ({cfg_hash}): {len(cells)} cells, "
        f"{len(already_ok)} already ok, {len(elsewhere)} running elsewhere, "
        f"{len(pending)} to run, {max_workers} workers")

    running: dict[CellKey, tuple[subprocess.Popen, float]] = {}
    attempts: dict[CellKey, int] = {}
    results: list[dict] = []
    shutdown: list[int] = []

    def dispatch(cell: CellKey) -> str:
        """Launch one cell, returning STARTED, SKIPPED or RETRY_LATER.

        Only RETRY_LATER leaves the cell queued, so the sweep loop pops on
        anything else. A skipped cell is not a failure: another driver holds
        it, or finished it, and the sweep simply ends incomplete.
        """
        task_id, seed = cell
        claim = _claim_path(grid_dir, cell)
        if not _try_claim(claim, claim_stale_s):
            # another driver is running this cell right now: defer it rather
            # than double-dispatch, and let the sweep end incomplete
            log(f"[grid] ~> {task_id}-s{seed} claimed by another driver — deferring")
            return SKIPPED
        if status().get(cell) == "ok":
            # finished elsewhere after our scan; the claim is held while we
            # check, so dropping it here is race-free
            _release_claim(claim)
            log(f"[grid] ~> {task_id}-s{seed} already ok elsewhere — skipping")
            return SKIPPED
        try:
            with open(grid_dir / f"{task_id}-s{seed}.log", "a") as logf:
                # start_new_session: the cell leads its own process group, so
                # one killpg reaches its agent subprocesses too
                proc = subprocess.Popen(build_cmd(task_id, seed), stdout=logf,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env)
        except Exception as e:  # noqa: BLE001 — one bad cell must not end the sweep
            _release_claim(claim)
            attempts[cell] = attempts.get(cell, 0) + 1
            log(f"[grid] xx {task_id}-s{seed}: dispatch failed ({e})")
            if attempts[cell] < max_attempts:
                return RETRY_LATER
            results.append({"cell": cell, "status": "failed",
                            "error": f"dispatch failed: {e}"})
            return SKIPPED
        attempts[cell] = attempts.get(cell, 0) + 1
        running[cell] = (proc, time.monotonic())
        log(f"[grid] -> {task_id}-s{seed} (attempt {attempts[cell]})")
        return STARTED

    def reap(cell: CellKey, proc: subprocess.Popen, killed: bool = False) -> None:
        task_id, seed = cell
        del running[cell]
        # release BEFORE the requeue decision: a requeued cell re-claims at
        # its next dispatch, and holding the claim would make the driver
        # defer to itself
        _release_claim(_claim_path(grid_dir, cell))
        ok = (not killed) and proc.returncode == 0 and status().get(cell) == "ok"
        if not ok and attempts.get(cell, 0) < max_attempts:
            log(f"[grid] ~~ {task_id}-s{seed} "
                f"{'TIMEOUT' if killed else 'FAILED'} -> requeue "
                f"({attempts.get(cell, 0)}/{max_attempts})")
            pending.append(cell)
            return
        outcome = "timeout" if killed else ("ok" if ok else "failed")
        results.append({"cell": cell, "status": outcome, "rc": proc.returncode})
        log(f"[grid] <- {task_id}-s{seed} {outcome.upper()}")

    def on_signal(signum, frame):   # noqa: ANN001 — stdlib handler shape
        shutdown.append(signum)

    previous: dict[int, object] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, on_signal)
        except ValueError:
            pass                # not the main thread; the finally still runs

    try:
        while pending or running:
            if shutdown:
                log(f"[grid] signal {shutdown[0]} — terminating "
                    f"{len(running)} live cell(s)")
                break
            now = time.monotonic()
            for cell in list(running):
                proc, started = running[cell]
                if proc.poll() is not None:
                    reap(cell, proc)
                elif now - started > job_timeout_s:
                    log(f"[grid] !! {cell[0]}-s{cell[1]} exceeded "
                        f"{job_timeout_s:.0f}s — killing")
                    _kill_group(proc, TERM_GRACE_S)
                    reap(cell, proc, killed=True)
            # peek-then-pop: a cell worth retrying stays at the head for the
            # next tick, so popping first would lose it
            while pending and len(running) < max_workers:
                if dispatch(pending[0]) == RETRY_LATER:
                    break
                pending.pop(0)
            for cell in list(running):      # heartbeat our claims
                try:
                    os.utime(_claim_path(grid_dir, cell))
                except OSError:
                    pass
            if pending or running:
                time.sleep(poll_s)
    finally:
        # claims are released only after the process is gone: an unclaimed
        # live cell is exactly the window that lets another driver take it
        for cell in list(running):
            proc, _ = running.pop(cell)
            _kill_group(proc, TERM_GRACE_S)
            _release_claim(_claim_path(grid_dir, cell))
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    final = status()
    incomplete = [c for c in cells if final.get(c) != "ok"]
    ok_n = sum(1 for r in results if r["status"] == "ok")
    summary = {"arm": cfg.name, "config_hash": cfg_hash, "cells": len(cells),
               "already_ok": len(already_ok), "ran": len(results), "ok": ok_n,
               "failed": len(results) - ok_n, "incomplete": len(incomplete),
               "incomplete_cells": [f"{t}-s{s}" for t, s in incomplete],
               "grid_dir": str(grid_dir)}
    (grid_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[grid] done: {ok_n}/{len(results)} ok, {len(already_ok)} pre-done"
        + (f", {len(incomplete)} INCOMPLETE" if incomplete else ""))
    if shutdown:
        raise SystemExit(128 + shutdown[0])
    return summary


def main() -> None:
    """Sweep one arm; exit 0 when complete, 3 when cells remain elsewhere, 1 on failures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-workers", type=int, default=None)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    ops = cfg.ops or {}
    export_data_env(ops)
    if "all" in cfg.tasks:
        from arbench import ALEBench
        cfg = dataclasses.replace(cfg, tasks=tuple(ALEBench().list_tasks()))
    driver_ops = ops.get("driver", {})
    # the driver's ledger must read the root run_cell writes to: a mismatch
    # makes every completed cell look failed
    paths = ops.get("paths") or {}
    summary = run_grid(
        cfg, config_path, resolve_path(paths.get("traces_root"), "data/traces"),
        max_workers=args.max_workers or driver_ops.get("max_workers", 1),
        poll_s=driver_ops.get("poll_s", 1.0),
        job_timeout_s=driver_ops.get("job_timeout_s", 43200.0),
        max_attempts=driver_ops.get("max_attempts", 2),
        grid_root=resolve_path(paths.get("grid_root"), "data/grid"))
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed"]:
        raise SystemExit(1)
    if summary["incomplete"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
