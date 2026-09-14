#!/usr/bin/env bash
# Serve the embedding + reranker models for the x>0 cells: one resident copy
# per box, reached by every cell over a Unix socket. Needed only for the
# cross-task memory plane; the x=0 cells never connect.
#
#   EMBED_MODEL=... RERANKER_MODEL=... DEVICE=... SOCKET=... bash scripts/serve_models.sh
#
# The socket path must match what the cells compute: leave SOCKET unset to use
# the default, or export ARLOOP_MODEL_SOCKET to the same value for both.
set -euo pipefail

EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}
RERANKER_MODEL=${RERANKER_MODEL:-Qwen/Qwen3-Reranker-0.6B}
DEVICE=${DEVICE:-cuda}

exec arloop-model-server \
    --embed-model "$EMBED_MODEL" \
    --reranker-model "$RERANKER_MODEL" \
    --device "$DEVICE" \
    ${SOCKET:+--socket "$SOCKET"}
