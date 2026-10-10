#!/usr/bin/env bash
# Serve the competition model on the H100 with vLLM (OpenAI-compatible API on :8000).
#   bash scripts/serve_vllm.sh            # foreground; Ctrl-C to stop
#   mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &
# 32K context = the Kaggle runtime's limit, so dev runs see the same budget as the leaderboard.
# Tool-calling flags (parser, chat template) come from the harness README once it is downloaded.
set -euo pipefail
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"     # int4 symmetric, group 32, compressed-tensors (~17 GB)
exec vllm serve "$MODEL" \
  --served-model-name gemma4 \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --port "${PORT:-8000}" \
  "$@"
