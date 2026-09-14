#!/usr/bin/env bash
# Serve the agent model for the grid. One endpoint per box; cells reach it
# through VLLM_BASE_URL (default http://127.0.0.1:8000/v1).
#
#   MODEL=... PORT=... TP=... bash scripts/serve_vllm.sh
#
# --reasoning-parser qwen3 --reasoning-config are REQUIRED for the r>0 cells:
# without them the top-level thinking_token_budget request field is accepted
# and ignored, and the reasoning plane runs unbounded while looking correct.
# NEVER add --speculative-config: speculative decoding bypasses the thinking
# budget, which makes the whole reasoning axis read as null.
set -euo pipefail

MODEL=${MODEL:-cyankiwi/Qwen3.5-27B-AWQ-4bit}
SERVED_NAME=${SERVED_NAME:-qwen27b}
PORT=${PORT:-8000}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
# a hybrid (Mamba + attention) model holds one cache block per decode
# sequence, so the sequence count is bounded by that cache, not by the KV
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'
