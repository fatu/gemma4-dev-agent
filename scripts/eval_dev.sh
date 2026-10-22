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
# Budgets: the local CLI ignores eval_config.yaml (seen: 32 calls, 490 s under a 10-call / 1-min config); only flags
# count. Since 2026-10-20 every arm runs under the scorer's per-task budget, passed as flags, with the time scaled
# to this box's speed:
#   MAX_CALLS=28 TIME_MIN=4 CMD_TIMEOUT=240 VERIFY_TIMEOUT=300  (the best public eval_config + README's fixed 300 s)
#   BUDGET=none reproduces the W2/W3 runs (no cap). Those results (Finding #5 included) were unbudgeted: on Kaggle's
#   4×L4 at the scorer budget, nothink scored 4/33 vs 8/33 for the LB-0.24 config (it ran out of the 8 minutes).
#   TIME_MIN defaults to 4, not the scorer's 8: this box decodes ~62 tok/s vs ~29 on 4×L4, so 8 min here buys ~2x
#   the steps; 8 x 29/62 ≈ 3.7, rounded up to 4 because the CLI flag's type is unchecked (a bit looser for
#   generation). Unvalidated: test runs and tool calls take the same wall time on both machines but are squeezed
#   into half the minutes here, so it is not an exact L4 match. TIME_MIN=8 = the literal scorer number (e.g. on Kaggle
#   L4 hardware). Token effects (overflow, schema size) need no scaling.
# One run per arm at temp 0.2 is noisy: repeat with TAG.
#
# Public-top base (W4): SUBMISSION=<dir> from a notebook that embeds its files (scripts/from_notebook.py):
#   kaggle kernels pull honghanhhh/gemma-4-baseline-lb-0-24 -p /tmp/lb024
#   python3 scripts/from_notebook.py /tmp/lb024/*.ipynb runs/base_lb024
#   SUBMISSION=runs/base_lb024 ARMS="base tools6 tools6+submit" nohup bash scripts/eval_dev.sh > runs/eval_dev_w4.log 2>&1 &
#   base   — SUBMISSION unchanged (like sample, for any base)
#   out6k / out4k — max_output_tokens 6144 / 4096 (+ thinking_budget 2048). ab1 on 4×L4: every overflow was
#            prompt + 8192 > 32768, while per-call output p99 was ~3.5K and the max ~4.4K tokens.
#   tools6 — drop get_code_neighbors / search_similar_code / get_code_subgraph from agent.yaml (the LB-0.24 prompt
#            already forbids them; their schemas are sent on every call)
#   submit — prompt rule from scripts/arms/submit.md: submit as soon as the fix is checked (ab1: 8/24 failed runs had
#            edits but a 0-byte patch). It replaces the base's "edits count without submit_patch" line if present.
# Rule arms (B-3, BOX_PLAN.md): each inserts or replaces one line of prompts/system.md via scripts/arms/apply.py
# (scripts/arms/<arm>.md names the anchor line; prints "appended (anchor not found)" on a different base prompt).
# Evidence from the ab1/ab2 traces on 4×L4 (aggregates in RESULTS.md):
#   editrecover — a malformed edit_file call (missing old_string: 377 of 377 failed edits) → switch to a heredoc
#            Python replace at once instead of retrying (one run retried 7x and overflowed)
#   issuefirst — the issue names the place or shows the fix → open it and edit first (wins: first edit at call 6
#            after 2 reads; losses: call 15 after 6)
#   tmpfix — the base's "scratch files go in /tmp" contradicts write_file, which rejects /tmp paths
#   test1  — one targeted test before submit_patch is required (only 6 of 33 runs ran a test)
# B-3 re-runs base in the same pass (runs/dev33_base exists, so TAG keeps the dirs apart):
#   TAG=w5 SUBMISSION=runs/base_lb024 ARMS="base editrecover issuefirst tmpfix+test1" \
#     nohup bash scripts/eval_dev.sh > runs/eval_dev_w5.log 2>&1 &
# B-3 result: none passed, and the two base passes (B-2 11, B-3 12) differed on 5 tasks. So before more arms: the
# end of every run now prints scripts/rulestats.py (per-arm rule-following: edit failures, first edit, tests,
# /tmp writes), and base is repeated to measure the noise (REFS lists the earlier base runs for the compare table):
#   nohup bash -c 'TAG=r2 SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh;
#     TAG=r3 REFS="base base_w5 base_r2" SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh' \
#     > runs/eval_dev_rep.log 2>&1 &
# B-3c showed 10 of the 17 flip losses were the 4-min cap under CONC=3 (per-stream decode is slower when 3 tasks
# share the server). The scorer runs one task at a time, so the Kaggle-like box config is CONC=1 with TIME_MIN scaled
# by the box's *measured* single-stream rate. Three such lanes at once (one card + one private 127.0.0.1:8000 each):
# scripts/lane.sh — it calls this script with CONC=1 inside a network namespace (BOX_PLAN.md B-3d).
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE="${SUBMISSION:-harness/sample_submission}"
ARMS="${ARMS:-sample out8k}"
if [ "${BUDGET:-scorer}" = none ]; then BUDGET_FLAGS=(); else
  BUDGET_FLAGS=(--max-tool-calls "${MAX_CALLS:-28}" --max-time-minutes "${TIME_MIN:-4}"
                --timeout-seconds "${CMD_TIMEOUT:-240}" --verification-timeout-seconds "${VERIFY_TIMEOUT:-300}"); fi
LOG="${VLLM_LOG:-runs/vllm.log}"
curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null || { echo "no vLLM on :8000 — start scripts/serve_vllm.sh first"; exit 1; }
# the server must serve the alias AND every adapter of the submission, else each task dies in ~2 s with NotFoundError
# and the run looks like a 0/33 result instead of a failure (seen in W1)
SUB="$BASE" python3 - <<'PY'
import json, os, pathlib, re, sys, urllib.request
ids = [m["id"] for m in json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"]]
# every agent YAML's `model:` (one base model per submission: 31B, or E4B/E2B for the small-model bonus)
models = sorted({m for p in pathlib.Path(os.environ["SUB"]).rglob("*.y*ml") if "adapters" not in p.parts
                 for m in re.findall(r"^\s*model:\s*['\"]?([\w.-]+)", p.read_text(), re.M)}) or ["gemma-4-31b-it-qat-w4a16-ct"]
print("submission models:", ", ".join(models))
need = models + sorted(p.parent.name for p in pathlib.Path(os.environ["SUB"], "adapters").glob("*/adapter_config.json"))
missing = [m for m in need if m not in ids]
print("served:", ", ".join(ids))
if missing:
    sys.exit(f"the server lacks {missing} — restart it with SUBMISSION=$BASE scripts/serve_vllm.sh")
PY
python3 -c 'import pytest_httpbin, inline_snapshot' 2>/dev/null || {   # without them requests/fastapi tests can't run
  echo "test deps missing (pytest-httpbin / inline-snapshot): run scripts/install_testdeps.sh first, or set NO_TESTDEPS=1"
  [ -n "${NO_TESTDEPS:-}" ] || exit 1; }
[ -f "$LOG" ] || echo "warning: no $LOG — ctx_overflow will be '?' (set VLLM_LOG=… to the server's log)"
mapfile -t IDS < splits/dev33.txt
echo "dev split: ${#IDS[@]} tasks · arms: $ARMS · budget: ${BUDGET_FLAGS[*]:-none} · concurrency ${CONC:-1}"
for arm in $ARMS; do                                  # check every arm up front, not 2 h into the first one
  for part in ${arm//+/ }; do
    case "$part" in out8k|nothink|think1k|readdisc|noanalyzer|out6k|out4k|tools6|submit|editrecover|issuefirst|tmpfix|test1) ;;
      sample|base) [ "$arm" = "$part" ] || { echo "$part can't stack: $arm"; exit 1; } ;;
      *) echo "unknown arm part $part in $arm"; exit 1 ;; esac
  done
  RUN="runs/dev33_$arm${TAG:+_$TAG}"
  [ -e "$RUN" ] && { echo "$RUN exists (task_results.jsonl is append-only) — repeat with TAG=r2"; exit 1; }
done

overflows() { [ -f "$LOG" ] || { echo "?"; return; }; n=$(grep -c "maximum context length" "$LOG") || true; echo "${n:-0}"; }

has() { [[ "+$arm+" == *"+$1+"* ]]; }                # arm "nothink+readdisc" has nothink and readdisc

for arm in $ARMS; do
  SUB="runs/subs/$arm${TAG:+_$TAG}"; RUN="runs/dev33_$arm${TAG:+_$TAG}"   # per TAG: parallel lanes must not share it
  rm -rf "$SUB"; mkdir -p runs/subs; cp -r "$BASE" "$SUB"
  Y="$SUB/configs/sampling.yaml"
  [ "$arm" != sample ] && [ "$arm" != base ] && sed -i 's/^max_output_tokens: .*/max_output_tokens: 8192/' "$Y"   # W3+ arms build on out8k
  has out6k && sed -i 's/^max_output_tokens: .*/max_output_tokens: 6144/' "$Y"
  if has out4k; then sed -i 's/^max_output_tokens: .*/max_output_tokens: 4096/' "$Y"
    sed -i 's/^\( *\)thinking_budget: .*/\1thinking_budget: 2048/' "$Y"; fi
  has nothink && sed -i 's/^\( *\)include_thoughts: .*/\1include_thoughts: false/' "$Y"
  has think1k && sed -i 's/^\( *\)thinking_budget: .*/\1thinking_budget: 1024/' "$Y"
  has readdisc && cat scripts/arms/readdisc.md >> "$SUB/prompts/system.md"
  if has noanalyzer; then                      # drop the 3-line agent_tool entry (code_analyzer) from the tool list
    sed -i '/^ *- agent_tool:/,/skip_summarization/d' "$SUB/agent.yaml"
    grep -q "agent_tool" "$SUB/agent.yaml" && { echo "noanalyzer: agent_tool still in agent.yaml"; exit 1; }
  fi
  if has tools6; then
    sed -i -E '/^ *- (get_code_neighbors|search_similar_code|get_code_subgraph) *$/d' "$SUB/agent.yaml"
    grep -qE 'get_code_neighbors|search_similar_code|get_code_subgraph' "$SUB/agent.yaml" && { echo "tools6: graph tool left in agent.yaml"; exit 1; }
  fi
  if has submit; then                          # replace the base's "edits count without submit" sentence, else append
    P="$SUB/prompts/system.md" R="scripts/arms/submit.md" python3 - <<'PY'
import os, re
p, rule = os.environ["P"], open(os.environ["R"]).read().strip()
s = open(p).read()
s2, n = re.subn(r"An edit that exists counts[^.]*\.", rule, s, count=1)
open(p, "w").write(s2 if n else s.rstrip() + "\n\n# SUBMIT\n- " + rule + "\n")
assert "{" not in rule and "}" not in rule        # ADK templates {name} from session state
print("submit:", "replaced the base's line" if n else "appended")
PY
  fi
  for rule in editrecover issuefirst tmpfix test1; do   # one-line prompt rules (scripts/arms/<rule>.md)
    has "$rule" && python3 scripts/arms/apply.py "$rule" "$SUB/prompts/system.md"
  done
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
    --display quiet ${BUDGET_FLAGS[@]+"${BUDGET_FLAGS[@]}"} || echo "swegemma eval exited $? for $arm (summary below may be partial)"
  after=$(overflows); mkdir -p "$RUN"                   # the eval may die before creating $RUN
  if [ "$before" = "?" ] || [ "$after" = "?" ]; then echo "?"; else echo $(( after - before )); fi > "$RUN/ctx_overflow.txt"
done

DIRS=$(for a in $ARMS; do echo "runs/dev33_$a${TAG:+_$TAG}"; done)
REF=""                                                # earlier runs shown as references: B-2's clean base by default
for r in ${REFS-base}; do                             # (REFS="out8k nothink" = the W3 references, pre test-deps fix)
  [[ " $ARMS " != *" $r "* || -n "${TAG:-}" ]] && [ -d "runs/dev33_$r" ] && REF="$REF runs/dev33_$r"
done
echo; python3 scripts/compare_runs.py $REF $DIRS
echo; python3 scripts/growth.py $REF $DIRS | grep -E '^==|Δprompt|→ b|growth explained' || true
echo; python3 scripts/rulestats.py $REF $DIRS || true      # did the agent follow the prompt rules? (📷)
