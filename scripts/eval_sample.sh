#!/usr/bin/env bash
# W1 step 4: the organiser's sample submission on 5 public tasks (2 fastapi, 2 rich, 1 requests; small gold patches).
#   bash scripts/eval_sample.sh                      # 📷 the summary at the end
#   SANDBOX=docker bash scripts/eval_sample.sh       # Docker containers instead of subprocess isolation (W2+)
#
# The harness starts ITS OWN vLLM server on 127.0.0.1:8000 (tensor-parallel 4, gpu util 0.80, tool_call_parser
# gemma4, reasoning_parser gemma4, thinking on, LoRA enabled) — so stop scripts/serve_vllm.sh first and free
# the GPUs, or the port and memory collide.  The sample submission's own eval_config.yaml caps each task at
# 1 min / 10 tool calls (the leaderboard baseline scores ~0), which is fine for this end-to-end check.
set -euo pipefail
cd "$(dirname "$0")/.."
if curl -s --max-time 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "something is already serving on :8000 (scripts/serve_vllm.sh?) — stop it first: pkill -f 'vllm serve'"; exit 1
fi
RUN="${RUN:-runs/w1_sample5}"
swegemma eval \
  --tasks harness/tasks.jsonl \
  --snapshots-dir harness/snapshots \
  --submission-dir harness/sample_submission \
  --results-dir "$RUN" \
  --sandbox "${SANDBOX:-subprocess}" \
  --task-ids fastapi_14492 fastapi_14786 rich_3278 rich_4077 requests_6589 \
  --concurrency 1 \
  --display auto
echo; echo "== $RUN/summary.json  📷"; python3 -m json.tool "$RUN/summary.json" | head -40
echo; echo "== per task"; python3 -c "
import json; [print({k: r.get(k) for k in ('instance_id','resolved','exit_code','tool_calls','turns','elapsed_seconds','error') if k in r}) for r in map(json.loads, open('$RUN/task_results.jsonl'))]"
