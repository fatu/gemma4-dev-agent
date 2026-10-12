#!/usr/bin/env bash
# W2 step 2: baseline on the fixed 33-task dev split (splits/dev33.txt), one run per "arm", then one comparison table.
#   nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &        # first, as for eval_sample.sh (all 4 cards: queue behind Bonsai)
#   nohup bash scripts/eval_dev.sh > runs/eval_dev.log 2>&1 &      # ~1 min/task + verification per arm; then:
#   python scripts/compare_runs.py runs/dev33_sample runs/dev33_out8k   # 📷 (eval_dev.sh prints it at the end too)
#
#   TAG=r2 nohup bash scripts/eval_dev.sh > runs/eval_dev_r2.log 2>&1 &  # a repeat → runs/dev33_<arm>_r2 (variance)
# Arms (ARMS="sample out8k" by default):
#   sample — the organiser's sample submission unchanged (max_output_tokens 16384)
#   out8k  — the same with max_output_tokens 8192. Finding #1 (2026-10-12): at 16384 the prompt only gets 32768-16384
#            tokens, and 42 calls died with "maximum context length"; 8192 leaves the prompt ~24K.
# Each variant is a copy of the sample under runs/subs/<arm>/ (gitignored: it holds the sample's adapters).
# vLLM's log is counted before/after every arm, so each run dir gets its own ctx_overflow count.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE="${SUBMISSION:-harness/sample_submission}"
ARMS="${ARMS:-sample out8k}"
LOG=runs/vllm.log
curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null || { echo "no vLLM on :8000 — start scripts/serve_vllm.sh first"; exit 1; }
mapfile -t IDS < splits/dev33.txt
echo "dev split: ${#IDS[@]} tasks · arms: $ARMS"

overflows() { n=$(grep -c "maximum context length" "$LOG" 2>/dev/null) || true; echo "${n:-0}"; }

for arm in $ARMS; do
  SUB="runs/subs/$arm"; RUN="runs/dev33_$arm${TAG:+_$TAG}"
  [ -e "$RUN" ] && { echo "$RUN exists (task_results.jsonl is append-only) — repeat with TAG=r2"; exit 1; }
  rm -rf "$SUB"; mkdir -p runs/subs; cp -r "$BASE" "$SUB"
  case "$arm" in
    sample) ;;
    out8k)  sed -i 's/^max_output_tokens: .*/max_output_tokens: 8192/' "$SUB/configs/sampling.yaml" ;;
    *)      echo "unknown arm $arm"; exit 1 ;;
  esac
  echo; echo "== $arm: $(grep -h '^max_output_tokens' "$SUB/configs/sampling.yaml")  →  $RUN   ($(date +%H:%M))"
  before=$(overflows)
  swegemma eval \
    --tasks harness/tasks.jsonl \
    --snapshots-dir harness/snapshots \
    --submission-dir "$SUB" \
    --results-dir "$RUN" \
    --sandbox "${SANDBOX:-subprocess}" \
    --task-ids "${IDS[@]}" \
    --concurrency "${CONC:-1}" \
    --display auto || echo "swegemma eval exited $? for $arm (summary below may be partial)"
  echo $(( $(overflows) - before )) > "$RUN/ctx_overflow.txt"
done

echo; python3 scripts/compare_runs.py $(for a in $ARMS; do echo "runs/dev33_$a${TAG:+_$TAG}"; done)
