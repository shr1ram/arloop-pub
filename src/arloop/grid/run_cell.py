"""One grid cell = one (config, task, seed) run, dispatched by the driver.

    python -m arloop.grid.run_cell --config <arm.yaml> --task <id> --seed <n>

Wires an arbench ALE task into the loop's TaskSpec plus its eval callables,
builds the arm's budget / runner / memory view, and exits 0 iff the run closed
ok (the driver's ledger check is the real success test).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import uuid
from pathlib import Path

from arbench import ALEBench

from arloop.budget import Budget
from arloop.grid.config import GridConfig, config_hash, load_config, resolved_prompts
from arloop.llm import VllmClient
from arloop.loop import RunConfig, TaskSpec, run_task
from arloop.memory.bank import MemoryView, load_bank
from arloop.memory.retrievers import HybridRerankRetriever
from arloop.runner import SubprocessRunner
from arloop.sandbox import THREAD_CAP_KEYS, assert_bwrap_works, derive_thread_cap
from arloop.tokens import set_chars_per_token

#: where a sandboxed run sees its task data (the ro bind and render_goal agree)
CONTAINER_DATA = "/data"

#: packages the system prompt promises the agent; agent code runs under this
#: same interpreter, so a venv without them burns every attempt on ImportError
PROMISED_MODULES = ("numpy",)


def check_promised_modules() -> None:
    """Refuse to start a cell whose interpreter lacks a promised package."""
    missing = [m for m in PROMISED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"{sys.executable} cannot import {missing}, which the "
                         f"prompt promises the agent; install the `agent` extra")


def resolve_path(value: str | None, default: str) -> Path:
    """A config path, resolved against the working directory (the repo root)."""
    return Path(value or default).resolve()


def export_data_env(ops: dict) -> None:
    """Publish ops.paths.data_env so ALEBench() finds its data roots."""
    for key, value in ((ops.get("paths") or {}).get("data_env") or {}).items():
        os.environ[key] = str(resolve_path(value, value))


def build_view(cfg: GridConfig, banks_root: Path, task_id: str, embedding_ops: dict):
    """The arm's (MemoryView, bank_id), or (None, None) for the x=0 cells.

    embedding_ops is ops.embedding: `server` (default true) connects to the
    per-box model server, else the models load in-process on `device`.
    """
    mem = cfg.memory
    if not mem:
        return None, None
    bank = load_bank(banks_root / mem["bank"])
    embed_fn = ce_fn = None
    if embedding_ops.get("server", True):
        from arloop.memory.model_server import connect
        client = connect(mem["embed_model"], mem["reranker_model"])
        embed_fn, ce_fn = client.embed_fn, client.ce_fn
    retriever = HybridRerankRetriever(
        model_name=mem["embed_model"], reranker_model=mem["reranker_model"],
        embed_fn=embed_fn, ce_fn=ce_fn,
        rerank_pool=mem.get("rerank_pool", 64), rrf_k=mem.get("rrf_k", 60),
        device=embedding_ops.get("device", "cpu"))
    view = MemoryView(bank, retriever, mem["retrieval_tokens"],
                      exclude_task_id=(task_id if mem.get("loto") else None))
    return view, bank.bank_id


def main() -> None:
    """Run one cell and exit 0 iff it closed ok."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    cfg_hash = config_hash(cfg)
    # the one char<->token ratio for this cell, set before any budget packing
    # or realised-token measurement runs
    set_chars_per_token(cfg.chars_per_token)
    check_promised_modules()

    ops = cfg.ops or {}
    export_data_env(ops)
    bench = ALEBench(sandbox=cfg.sandbox)
    task = bench.load_task(args.task)

    # the driver publishes the cap it actually launched us under; it wins over
    # the config because --max-workers can differ from ops.driver.max_workers,
    # and the runtime count is what decides how many cells share the box
    published = os.environ.get("ARLOOP_CPU_CAP")
    cpu_cap = (int(published) if published
               else derive_thread_cap(ops.get("driver", {}).get("max_workers", 1)))
    for key in THREAD_CAP_KEYS:
        os.environ[key] = str(cpu_cap)

    if cfg.sandbox:
        # fail closed before any attempt: an unconfined fallback would run a
        # different experiment under this arm's config_hash
        assert_bwrap_works()
    runner = SubprocessRunner(cpu_cap=cpu_cap,
                              ro_binds=((str(task.data_dir), CONTAINER_DATA),),
                              sandbox=cfg.sandbox)

    client_ops = ops.get("client", {})
    llm = VllmClient(model=cfg.model, temperature=cfg.temperature,
                     seed=args.seed,
                     timeout_s=client_ops.get("timeout_s", 600.0),
                     max_retries=client_ops.get("max_retries", 4),
                     thinking=cfg.thinking,
                     thinking_token_budget=cfg.thinking_token_budget,
                     use_tool=True,
                     history_write_tokens=cfg.history_write_tokens)

    paths = ops.get("paths", {})
    view, bank_id = build_view(cfg, resolve_path(paths.get("banks_root"), "data/banks"),
                               args.task, ops.get("embedding", {}))
    loop_ops = ops.get("loop", {})
    run_config = RunConfig(
        run_id=f"{cfg_hash[:8]}-{args.task}-s{args.seed}-{uuid.uuid4().hex[:6]}",
        config_hash=cfg_hash, seed=args.seed,
        traces_root=resolve_path(paths.get("traces_root"), "data/traces"),
        workspace_root=resolve_path(paths.get("workspaces_root"), "data/workspaces"),
        command=f"{sys.executable} solution.py",
        exec_timeout_s=cfg.exec_timeout_s,
        code_budget_chars=cfg.code_budget_chars,
        branch_policy=cfg.branch_policy,
        history_write_tokens=cfg.history_write_tokens,
        cpu_cap=cpu_cap,
        prompts=resolved_prompts(cfg),
        llm_retry_wall_s=loop_ops.get("llm_retry_wall_s", 1800.0),
        llm_retry_sleep_s=loop_ops.get("llm_retry_sleep_s", 30.0),
        manifest_extra={"data_version": task.metadata.get("data_version", ""),
                        "arm": cfg.name, "bank_id": bank_id})

    spec = TaskSpec(task_id=task.task_id, goal=task.render_goal(CONTAINER_DATA),
                    eval=task.eval, submission_filename=task.submission_filename)
    out = run_task(
        spec, view, runner, Budget(cfg.run_budget["type"], cfg.run_budget["limit"]),
        run_config, llm,
        public_eval=lambda p: bench.public_eval(task, p),
        validate=lambda p: bench.validate_submission(task, p),
        grade=lambda p: bench.grade(task, p))
    print(f"[cell] {run_config.run_id}: {out.status} proxy={out.proxy_score} "
          f"heldout={out.heldout_score} attempts={out.n_attempts}"
          + (f" failure={out.failure_reason}" if out.failure_reason else ""),
          flush=True)
    sys.exit(0 if out.status == "ok" else 3)


if __name__ == "__main__":
    main()
