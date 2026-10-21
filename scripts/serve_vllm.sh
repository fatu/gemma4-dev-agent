#!/usr/bin/env bash
# Serve the competition model with vLLM the way the Kaggle scorer does (HARNESS_README §3.1, §3.3, §3.4).
# The local `swegemma eval` CLI does NOT start a server (checked on the box 2026-10-12; the README's VllmServer
# is the Kaggle scorer's) — it expects one on 127.0.0.1:8000, so start this first and leave it running.
#   mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &    # then: bash scripts/eval_sample.sh
#   SUBMISSION=path/to/submission bash scripts/serve_vllm.sh                     # register that submission's adapters/
#   TP=1 GPU_UTIL=0.5 CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh          # one card (dev); default TP 4 = Kaggle
#   MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TP=1 GPU_UTIL=0.9 bash scripts/serve_vllm.sh   # small-model bonus (E4B/E2B)
# Served under the registry alias that agent.yaml declares and as "gemma4" (scripts/smoke.py). The alias is MODEL's
# basename in lower case (…/gemma-4-E4B-it-qat-w4a16-ct → gemma-4-e4b-it-qat-w4a16-ct); SERVED=… overrides it, e.g.
# when MODEL is a local path. Each
# <submission>/adapters/<name>/ becomes LoRA module "<name>": an agent with `adapter: <name>` requests that model id.
# The harness forwards sampling.yaml's thinking_budget as vLLM's `thinking_token_budget`, which needs the thinking
# delimiters (--reasoning-config) so vLLM can force the block closed: Gemma 4 thinks in `<|channel>thought\n … <channel|>`.
# The scorer's start string is "<|channel>" (HARNESS_README §3.1; used on Kaggle 4×L4 in the 2026-10-20 runs). W1–W3 used
# "<|channel>thought\n"; REASONING_START='<|channel>thought\n' reproduces them.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"     # int4 symmetric, group 32, compressed-tensors (~17 GB)
SERVED="${SERVED:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
case "$SERVED" in      # HARNESS_README §3 ALLOWED_MODEL_NAMES: the harness requests agent.yaml's `model:` by this name
  gemma-4-31b-it-qat-w4a16-ct|gemma-4-12b-it-qat-w4a16-ct|gemma-4-e4b-it-qat-w4a16-ct|gemma-4-e2b-it-qat-w4a16-ct) ;;
  *) echo "'$SERVED' (from MODEL=$MODEL) is not a scorer alias: set SERVED=gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct"
     exit 1 ;;
esac
EXTRA=()               # HARNESS_README §3.1 MODEL_VLLM_EXTRA_ARGS: only 12B gets extra flags
if [ "$SERVED" = gemma-4-12b-it-qat-w4a16-ct ]; then
  EXTRA=(--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')
fi
TP="${TP:-4}"; GPU_UTIL="${GPU_UTIL:-0.80}"; MAX_LEN="${MAX_LEN:-32768}"; PORT="${PORT:-8000}"
REASONING_START="${REASONING_START:-<|channel>}"
SUBMISSION="${SUBMISSION:-harness/sample_submission}"
LORA=()
for d in "$SUBMISSION"/adapters/*/; do
  [ -f "$d/adapter_config.json" ] && LORA+=("$(basename "$d")=$(cd "$d" && pwd)")
done
LORA_NAMES="${LORA[*]%%=*}"
if [ ${#LORA[@]} -gt 0 ]; then
  LORA=(--enable-lora --max-loras 8 --max-lora-rank 128 --lora-modules "${LORA[@]}")
fi
# what this server is, at the top of runs/vllm.log (📷 it next to the eval summary)
echo "serve_vllm $(date '+%F %T %Z') · commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  model $MODEL → served as $SERVED + gemma4 · TP $TP · gpu_util $GPU_UTIL · max_len $MAX_LEN · port $PORT"
echo "  reasoning_start $REASONING_START · lora: ${LORA_NAMES:-none} · extra: ${EXTRA[*]:-none}${*:+ $*} · CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /' || true
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" gemma4 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config "{\"reasoning_start_str\": \"$REASONING_START\", \"reasoning_end_str\": \"<channel|>\"}" \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --port "$PORT" \
  ${EXTRA[@]+"${EXTRA[@]}"} ${LORA[@]+"${LORA[@]}"} "$@"
