#!/usr/bin/env bash
# Serve the competition model with vLLM the way the Kaggle scorer does (HARNESS_README §3.1, §3.3, §3.4).
# The local `swegemma eval` CLI does NOT start a server (checked on the box 2026-10-12; the README's VllmServer
# is the Kaggle scorer's) — it expects one on 127.0.0.1:8000, so start this first and leave it running.
#   mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &    # then: bash scripts/eval_sample.sh
#   SUBMISSION=path/to/submission bash scripts/serve_vllm.sh                     # register that submission's adapters/
#   TP=1 GPU_UTIL=0.5 CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh          # one card (dev); default TP 4 = Kaggle
# Served under the registry alias that agent.yaml declares and as "gemma4" (scripts/smoke.py). Each
# <submission>/adapters/<name>/ becomes LoRA module "<name>": an agent with `adapter: <name>` requests that model id.
# The harness forwards sampling.yaml's thinking_budget as vLLM's `thinking_token_budget`, which needs the thinking
# delimiters (--reasoning-config) so vLLM can force the block closed: Gemma 4 thinks in `<|channel>thought\n … <channel|>`.
# The scorer's start string is "<|channel>" (HARNESS_README §3.1; used on Kaggle 4×L4 in the 2026-10-20 runs). W1–W3 used
# "<|channel>thought\n"; REASONING_START='<|channel>thought\n' reproduces them.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"     # int4 symmetric, group 32, compressed-tensors (~17 GB)
SUBMISSION="${SUBMISSION:-harness/sample_submission}"
LORA=()
for d in "$SUBMISSION"/adapters/*/; do
  [ -f "$d/adapter_config.json" ] && LORA+=("$(basename "$d")=$(cd "$d" && pwd)")
done
if [ ${#LORA[@]} -gt 0 ]; then
  echo "LoRA modules: ${LORA[*]}"
  LORA=(--enable-lora --max-loras 8 --max-lora-rank 128 --lora-modules "${LORA[@]}")
fi
exec vllm serve "$MODEL" \
  --served-model-name gemma-4-31b-it-qat-w4a16-ct gemma4 \
  --tensor-parallel-size "${TP:-4}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --max-model-len "${MAX_LEN:-32768}" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config "{\"reasoning_start_str\": \"${REASONING_START:-<|channel>}\", \"reasoning_end_str\": \"<channel|>\"}" \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --port "${PORT:-8000}" \
  ${LORA[@]+"${LORA[@]}"} "$@"
