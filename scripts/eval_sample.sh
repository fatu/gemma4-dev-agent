#!/usr/bin/env bash
# W1 step 4: the organiser's sample submission on 5 public tasks (2 fastapi, 2 rich, 1 requests; small gold patches).
#   nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &   # first — swegemma eval does NOT start vLLM itself
#   bash scripts/eval_sample.sh                               # 📷 the summary at the end
#   SANDBOX=docker bash scripts/eval_sample.sh                # Docker containers instead of subprocess isolation (W2+)
#
# Waits for the server on 127.0.0.1:8000 and checks it serves the alias the submission declares plus one LoRA
# module per adapters/<name>/ (serve_vllm.sh does both).  The sample submission's own eval_config.yaml caps each
# task at 1 min / 10 tool calls (the leaderboard baseline scores ~0), which is fine for this end-to-end check.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"          # LiteLLM's openai/ provider insists on a key; vLLM ignores it
SUB="${SUBMISSION:-harness/sample_submission}"
RUN="${RUN:-runs/w1_sample5}"
up() { curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; }
if ! up; then
  echo -n "waiting for vLLM on :8000 (start it: nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &) "
  for _ in $(seq 1 180); do up && break; echo -n .; sleep 10; done; echo
  up || { echo "no server after 30 min — see runs/vllm.log"; exit 1; }
fi
SUB="$SUB" python3 - <<'EOF'
import json, os, pathlib, sys, urllib.request
ids = [m["id"] for m in json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"]]
print("served:", ", ".join(ids))
need = ["gemma-4-31b-it-qat-w4a16-ct"] + sorted(p.parent.name for p in pathlib.Path(os.environ["SUB"], "adapters").glob("*/adapter_config.json"))
missing = [m for m in need if m not in ids]
if missing:
    sys.exit(f"the server lacks {missing} — restart it with scripts/serve_vllm.sh (serves the alias + the submission's adapters)")
EOF
swegemma eval \
  --tasks harness/tasks.jsonl \
  --snapshots-dir harness/snapshots \
  --submission-dir "$SUB" \
  --results-dir "$RUN" \
  --sandbox "${SANDBOX:-subprocess}" \
  --task-ids fastapi_14492 fastapi_14786 rich_3278 rich_4077 requests_6589 \
  --concurrency 1 \
  --display auto
echo; echo "== $RUN/summary.json  📷"; python3 -m json.tool "$RUN/summary.json" | head -40
echo; echo "== per task"; python3 -c "
import json; [print({k: r.get(k) for k in ('instance_id','resolved','exit_code','tool_calls','turns','elapsed_seconds','error') if k in r}) for r in map(json.loads, open('$RUN/task_results.jsonl'))]"
