"""Offline corpus build: the x=0 cells' traces -> the frozen episode bank.

Pools every ok run of the nine memory-free cells into one bank of whole-run
episode cases. LOTO is NOT applied here — each case carries its anchor task_id
and the query-time filter keeps a running task from seeing its own cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from arloop.llm import VllmClient
from arloop.memory.bank import build_bank
from arloop.memory.retrievers import EmbeddingRetriever
from arloop.memory.writer import EpisodeWriter, WriteParams
from arloop.trace import iter_runs

#: the nine x=0 cells of the final grid — the whole memory-free phase
DEFAULT_CELLS = tuple(f"final-grid-r{r}-w{w}-x0"
                      for r in (0, 3000, 9000) for w in (100, 300, 900))


def collect_runs(traces_root: Path, cells: list[str]) -> list:
    """Every ok run under the named cells, one per (config_hash, task, seed)."""
    runs = []
    for cell in cells:
        runs.extend(iter_runs(traces_root / cell, status="ok"))
    # a cross-box race can double-run a source cell (both ok, distinct run_ids);
    # keeping the lexicographically first run_id keeps the bank id deterministic
    runs.sort(key=lambda mp: mp[0].run_id)
    seen, deduped = set(), []
    for m, path in runs:
        key = (m.config_hash, m.task_id, m.seed)
        if key not in seen:
            seen.add(key)
            deduped.append((m, path))
    return deduped


def build_corpus(traces_root: Path, cells: list[str], out_dir: Path, llm,
                 params: WriteParams, nodes_dir: Path | None = None,
                 workers: int = 8, embed_retriever: EmbeddingRetriever | None = None,
                 force: bool = False):
    """Write the episode bank for `cells`; returns the Bank."""
    out_dir = Path(out_dir)
    if (out_dir / "bank_manifest.json").exists() and not force:
        # bank ids do not move when writer behaviour changes, so a stale dir
        # must be overwritten deliberately, never resolved as a cache hit
        raise FileExistsError(f"{out_dir} already holds a bank; pass force=True to overwrite")
    runs = collect_runs(Path(traces_root), cells)
    if not runs:
        raise RuntimeError(f"no ok runs for {cells} under {traces_root}")
    writer = EpisodeWriter(runs, llm, params=params, nodes_dir=nodes_dir,
                           workers=workers)
    cases = writer.cases()
    corpus_hash = hashlib.sha256(
        json.dumps(sorted(m.run_id for m, _ in runs)).encode()).hexdigest()[:12]
    derivation = {"corpus_hash": corpus_hash, "source_cells": sorted(cells),
                  **writer.derivation()}
    bank = build_bank(out_dir, cases, derivation)
    n_tasks = len({m.task_id for m, _ in runs})
    print(f"{len(runs)} ok runs across {n_tasks} tasks -> {len(cases)} cases "
          f"({writer.llm_calls} LLM calls, {writer.cache.hits} cache hits, "
          f"{writer.build_cost_tokens} tokens)", flush=True)
    if embed_retriever is not None:
        embed_retriever.embed_bank(bank)
        print("embedding index written", flush=True)
    print(f"bank {bank.bank_id} -> {out_dir}", flush=True)
    return bank


def main() -> int:
    """CLI: python -m arloop.grid.build_corpus --traces-root ... --out ..."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-root", required=True, type=Path)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="qwen27b")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--timeout-s", type=float, default=1800.0,
                    help="per-call client timeout; a saturated server queues "
                         "requests for minutes before the first byte")
    ap.add_argument("--nodes-dir", default=Path("data/nodes"), type=Path,
                    help="content-addressed cache of every LLM call, so a "
                         "relaunch resumes instead of restarting")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--embed-model", default=None,
                    help="also write the on-disk embedding index for the bank")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--c-step", type=int, default=300)
    ap.add_argument("--fold-window", type=int, default=8)
    ap.add_argument("--retrieval-context-chars", type=int, default=240)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # thinking off: distillation needs no reasoning span, and a Qwen template
    # left to reason makes every fold ~100x slower
    llm = VllmClient(args.model, base_url=args.base_url, temperature=0.0,
                     timeout_s=args.timeout_s, thinking=False)
    params = WriteParams(c_step=args.c_step, fold_window=args.fold_window,
                         retrieval_context_chars=args.retrieval_context_chars)
    embedder = (EmbeddingRetriever(args.embed_model, device=args.device)
                if args.embed_model else None)
    build_corpus(args.traces_root, args.cells.split(","), args.out, llm, params,
                 nodes_dir=args.nodes_dir, workers=args.workers,
                 embed_retriever=embedder, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
