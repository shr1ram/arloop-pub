# arloop

The agent loop, the memory module, and the experiment grid for a study of how
an autoresearch agent should spend a fixed token budget. Tasks and grading
come from [`arbench`](../arbench-pub), a sibling checkout.

```
src/arloop/          the loop: prompts, LLM client, budget, sandbox, runner,
                     trace writer, and run_task itself
src/arloop/memory/   episode cases written from earlier runs, the retrievers
                     that read them back, and the per-box model server
src/arloop/grid/     arm configs, the per-cell entry point, and the driver
experiments/         the final 3x3x3 allocation grid
data/                gitignored outputs (see below)
```

**The loop** runs a budgeted greedy chain on one task: draft, then debug on
failure or improve on success, branching improve from the best-scoring attempt
so far. The budget is counted in tokens (prompt plus completion, reasoning
included) and checked at attempt boundaries. Agent code runs in a bwrap
sandbox with only the task's public data visible; every attempt is scored by
the official public evaluator and the held-out grade runs once at the end.

**The memory module** turns whole runs into episode cases, retrieves them per
attempt with BM25 + embeddings fused by RRF and reranked by a cross-encoder,
and packs the result under a read budget. A running task never retrieves cases
anchored on itself.

**The grid** sweeps one arm's (task, seed) cells. The run manifests are the
ledger — a cell is done iff an ok manifest exists for its (config hash, task,
seed) — so re-running an arm is idempotent and concurrent drivers on different
boxes are safe.

## Two external services

Both are started by the operator, not by the code:

- **vLLM** serves the agent model over an OpenAI-compatible endpoint
  (`scripts/serve_vllm.sh`). Cells reach it at `VLLM_BASE_URL`, default
  `http://127.0.0.1:8000/v1`, with `VLLM_API_KEY` if the endpoint wants one.
- **The model server** holds one resident copy of the embedding and reranker
  models per box and answers cells over a Unix socket
  (`scripts/serve_models.sh`). Only the cross-task memory cells use it.

## Install

```bash
uv sync --extra dev --extra agent   # loop, grid, tests; numpy for the agent's code
uv sync --extra dev --extra agent --extra models
                                    # + sentence-transformers/torch, for the
                                    #   memory cells, the model server and
                                    #   the corpus build
```

`arbench` is resolved from the sibling checkout through `[tool.uv.sources]`,
so install with `uv`, not plain `pip`. Run every command from the repo root:
config paths are relative to the working directory.

## Run the grid

See [`experiments/final_grid/README.md`](experiments/final_grid/README.md) for
the axes, what is held fixed, and the launch order. One cell:

```bash
arloop-grid experiments/final_grid/cells/final-grid-r3000-w300-x4000.yaml \
    --max-workers 14
```

## Data layout

Everything under `data/` is gitignored and reproducible from the traces:

```
data/traces/      run manifests + event logs — the ledger and the raw record
data/banks/       episode banks built from traces, with their embedding index
data/workspaces/  per-attempt scratch dirs the agent's code runs in
data/nodes/       the corpus writer's content-addressed LLM call cache
data/grid/        per-arm claim files, per-cell driver logs, summary.json
```

## Tests

```bash
uv run pytest       # no network, no GPU, no model downloads
```
