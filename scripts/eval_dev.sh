#!/usr/bin/env bash
# W2 step 2: baseline on the fixed 33-task dev split (splits/dev33.txt), one run per "arm", then one comparison table.
#   nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &        # first, as for eval_sample.sh (all 4 cards: queue behind Bonsai)
#   nohup bash scripts/eval_dev.sh > runs/eval_dev.log 2>&1 &      # measured 38-490 s/task (avg ~230) → ~2 h/arm, 4-5 h total
#   python scripts/compare_runs.py runs/dev33_sample runs/dev33_out8k   # 📷 (eval_dev.sh prints it at the end too)
#
#   TAG=r2 nohup bash scripts/eval_dev.sh > runs/eval_dev_r2.log 2>&1 &  # a repeat → runs/dev33_<arm>_r2 (variance)
# Arms (ARMS="sample out8k" by default):
#   sample — the organiser's sample submission unchanged (max_output_tokens 16384)
#   out8k  — the same with max_output_tokens 8192. Finding #1 (2026-10-12): at 16384 the prompt only gets 32768-16384
#            tokens, and 42 calls died with "maximum context length"; 8192 leaves the prompt ~24K.
#   nothink — out8k + thinking_config.include_thoughts: false. Finding #4: every generated token, thinking included, is
#             re-sent in later prompts (growth.py b = 1.00, ~35-38% of growth). Does ADK drop thoughts from history then?
#   think1k — out8k + thinking_budget: 1024 (was 4096): keep thinking, but make each step's thinking smaller.
#   ARMS="nothink think1k" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3a.log 2>&1 &   # compares with dev33_out8k
#   readdisc — out8k + a "Context Discipline" block appended to prompts/system.md (scripts/arms/readdisc.md): ranged
#              read_file, no re-reads, short command output, brief thoughts. Finding #3: context dies by accumulation,
#              read_file ≈ 38 % and run_command ≈ 11 % of what the agent adds — the tool-output side of the growth.
#   noanalyzer — out8k with the code_analyzer AgentTool removed: the root agent explores by itself, so every read_file
#              lands in its own context. Measures what the sub-agent isolation (README §10.4) is worth.
#   ARMS="readdisc noanalyzer" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3b.log 2>&1 &
# Arms stack with "+": "nothink+readdisc" = out8k + both edits. Finding #5 (2026-10-14): nothink 12/33 vs out8k 8/33 and
# b 1.00 → 0.12, so round 2 builds on nothink; the plain nothink repeat (TAG=r2) checks that 12/33 is not noise:
#   TAG=r2 ARMS="nothink nothink+readdisc nothink+noanalyzer" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3b.log 2>&1 &
# Each variant is a copy of the sample under runs/subs/<arm>/ (gitignored: it holds the sample's adapters).
# vLLM's log (VLLM_LOG, default runs/vllm.log) is counted before/after every arm → <run>/ctx_overflow.txt. It counts
# rejected CALLS, not tasks: LiteLLM retries each failed call up to 5x, so 42 lines were only a few tasks. Compare it
# between arms; don't read it as a task count. If the log is missing it writes "?" instead of a fake 0.
# Note: the sample's eval_config (10 tool calls / 1 min) is not enforced by the local CLI (seen: 32 calls, 490 s), so
# these numbers are not comparable to the leaderboard's ~0 baseline. One run per arm at temp 0.2 is noisy: repeat with TAG.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE="${SUBMISSION:-harness/sample_submission}"
ARMS="${ARMS:-sample out8k}"
LOG="${VLLM_LOG:-runs/vllm.log}"
curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null || { echo "no vLLM on :8000 — start scripts/serve_vllm.sh first"; exit 1; }
# the server must serve the alias AND every adapter of the submission, else each task dies in ~2 s with NotFoundError
# and the run looks like a 0/33 result instead of a failure (seen in W1)
SUB="$BASE" python3 - <<'PY'
import json, os, pathlib, sys, urllib.request
ids = [m["id"] for m in json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"]]
need = ["gemma-4-31b-it-qat-w4a16-ct"] + sorted(p.parent.name for p in pathlib.Path(os.environ["SUB"], "adapters").glob("*/adapter_config.json"))
missing = [m for m in need if m not in ids]
print("served:", ", ".join(ids))
if missing:
    sys.exit(f"the server lacks {missing} — restart it with SUBMISSION=$BASE scripts/serve_vllm.sh")
PY
[ -f "$LOG" ] || echo "warning: no $LOG — ctx_overflow will be '?' (set VLLM_LOG=… to the server's log)"
mapfile -t IDS < splits/dev33.txt
echo "dev split: ${#IDS[@]} tasks · arms: $ARMS"
for arm in $ARMS; do                                  # check every arm up front, not 2 h into the first one
  for part in ${arm//+/ }; do
    case "$part" in out8k|nothink|think1k|readdisc|noanalyzer) ;; sample) [ "$arm" = sample ] || { echo "sample can't stack: $arm"; exit 1; } ;;
      *) echo "unknown arm part $part in $arm"; exit 1 ;; esac
  done
  RUN="runs/dev33_$arm${TAG:+_$TAG}"
  [ -e "$RUN" ] && { echo "$RUN exists (task_results.jsonl is append-only) — repeat with TAG=r2"; exit 1; }
done

overflows() { [ -f "$LOG" ] || { echo "?"; return; }; n=$(grep -c "maximum context length" "$LOG") || true; echo "${n:-0}"; }

has() { [[ "+$arm+" == *"+$1+"* ]]; }                # arm "nothink+readdisc" has nothink and readdisc

for arm in $ARMS; do
  SUB="runs/subs/$arm"; RUN="runs/dev33_$arm${TAG:+_$TAG}"
  rm -rf "$SUB"; mkdir -p runs/subs; cp -r "$BASE" "$SUB"
  Y="$SUB/configs/sampling.yaml"
  [ "$arm" != sample ] && sed -i 's/^max_output_tokens: .*/max_output_tokens: 8192/' "$Y"     # every W3 arm builds on out8k
  has nothink && sed -i 's/^\( *\)include_thoughts: .*/\1include_thoughts: false/' "$Y"
  has think1k && sed -i 's/^\( *\)thinking_budget: .*/\1thinking_budget: 1024/' "$Y"
  has readdisc && cat scripts/arms/readdisc.md >> "$SUB/prompts/system.md"
  if has noanalyzer; then                      # drop the 3-line agent_tool entry (code_analyzer) from the tool list
    sed -i '/^ *- agent_tool:/,/skip_summarization/d' "$SUB/agent.yaml"
    grep -q "agent_tool" "$SUB/agent.yaml" && { echo "noanalyzer: agent_tool still in agent.yaml"; exit 1; }
  fi
  echo; echo "== $arm: $(grep -hE 'max_output_tokens|thinking_budget|include_thoughts' "$SUB/configs/sampling.yaml" | tr -s ' \n' ' ')· tools $(grep -c '^  - ' "$SUB/agent.yaml") · prompt $(wc -c < "$SUB/prompts/system.md") chars → $RUN   ($(date +%H:%M))"
  before=$(overflows)
  swegemma eval \
    --tasks harness/tasks.jsonl \
    --snapshots-dir harness/snapshots \
    --submission-dir "$SUB" \
    --results-dir "$RUN" \
    --sandbox "${SANDBOX:-subprocess}" \
    --task-ids "${IDS[@]}" \
    --concurrency "${CONC:-1}" \
    --display quiet || echo "swegemma eval exited $? for $arm (summary below may be partial)"
  after=$(overflows); mkdir -p "$RUN"                   # the eval may die before creating $RUN
  if [ "$before" = "?" ] || [ "$after" = "?" ]; then echo "?"; else echo $(( after - before )); fi > "$RUN/ctx_overflow.txt"
done

DIRS=$(for a in $ARMS; do echo "runs/dev33_$a${TAG:+_$TAG}"; done)
REF=""                                                # W3 arms: show the out8k / first-nothink references too
for r in out8k nothink; do
  [[ " $ARMS " != *" $r "* || -n "${TAG:-}" ]] && [ -d "runs/dev33_$r" ] && REF="$REF runs/dev33_$r"
done
echo; python3 scripts/compare_runs.py $REF $DIRS
echo; python3 scripts/growth.py $REF $DIRS | grep -E '^==|Δprompt|→ b|growth explained' || true
