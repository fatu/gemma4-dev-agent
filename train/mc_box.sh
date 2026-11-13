#!/usr/bin/env bash
# B-9 part 2 (BOX_PLAN): a LoRA on MetaCurate's public DSv4Pro trajectories, steps 0–2 on the box.
#   MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh serve     # one card (31B first)
#   MODEL=… GPU=0 PORT=8011 bash train/mc_box.sh probe     # step 0: 2 Verified tasks through the request logger → 📷
#   (THINK_HIST=1, the default since option ③: the logger also renames reasoning_content → reasoning, so the server
#    renders earlier thoughts; outputs in runs/verified/probe_th; THINK_HIST=0 = the original B-9 setup, runs/verified/probe)
#   MODEL=… bash train/mc_box.sh data                      # step 1: download (HF token) + filter → runs/sft/mc/r1k   → 📷
#   MODEL=… bash train/mc_box.sh render                    # step 2: render.py --pack in RENDER_PROCS shards (gate R) → 📷
# Knobs (all printed): MODEL (31B) · GPU (0) · PORT (8011; the logger listens on PORT+100) · PROBE_SLICE (0:2) ·
# HISTORY (from step 0's probe_summary.json; override keep|drop) · N (1000) · SEED (0) · MAX_CTX (30720) · ARM (r1k) ·
# RENDER_PROCS (16) · T2T (on) · PACK (from step 0: on only if the server renders earlier thoughts). Outputs quote task text → runs/ only (gitignored).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?probe|data|render}"
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"; SERVED="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
GPU="${GPU:-0}"; PORT="${PORT:-8011}"; PROBE_SLICE="${PROBE_SLICE:-0:2}"
N="${N:-1000}"; SEED="${SEED:-0}"; MAX_CTX="${MAX_CTX:-30720}"; ARM="${ARM:-r1k}"; RENDER_PROCS="${RENDER_PROCS:-16}"
PY=.venv-eval/bin/python; MC=runs/sft/mc
THINK_HIST="${THINK_HIST:-1}"                     # option ③ (user, 2026-11-13): the server sees earlier thoughts
# the probe's sampling = the new baselines' (user, 2026-11-13): 8,192 tokens per reply, top_p 0.95, thinking budget 4,096
MAX_TOK="${MAX_TOK:-8192}"; TOP_P="${TOP_P:-0.95}"; THINK_BUDGET="${THINK_BUDGET:-4096}"
PR=runs/verified/probe$([ "$THINK_HIST" = 1 ] && echo _th); WIRE=$([ "$THINK_HIST" = 1 ] && echo mini-think || echo mini)
HISTORY="${HISTORY:-$($PY -c "import json; print(json.load(open('$PR/probe_summary.json'))['history_rule'])" 2>/dev/null || echo unset)}"
# T2T: move the text of text + tool-call replies into the thought. On by the user's decision after the 31B probe
# (2026-11-13: 782 such turns in 31B's own histories — the Gemma template bug HF #67/#115 hits the baseline).
T2T="${T2T:-on}"
# PACK: one sequence per trajectory only if the server renders earlier thoughts; vLLM 0.19.1 drops mini's
# reasoning_content (probe 2026-11-13: 0 blocks), so then every call is its own sample (render.py without --pack).
PACK="${PACK:-$($PY -c "import json; print('on' if json.load(open('$PR/probe_summary.json')).get('server renders earlier thoughts') else 'off')" 2>/dev/null || echo off)}"
tokdir() { $PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json', '*.jinja']))" 2>/dev/null | tail -1; }
echo "mc_box $STEP · model $MODEL (served $SERVED) · gpu $GPU port $PORT (logger $(( PORT + 100 ))) · probe slice $PROBE_SLICE · think hist $THINK_HIST (wire $WIRE, probe dir $PR) · max tok $MAX_TOK · top_p $TOP_P · thinking budget $THINK_BUDGET · history $HISTORY · text to thought $T2T · pack $PACK · n $N seed $SEED max ctx $MAX_CTX · arm $ARM · render procs $RENDER_PROCS · code $(git log --oneline -1 | cut -c1-40)"
T0=$(date +%s)

case "$STEP" in
probe)
  curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" || { echo "no $SERVED on port $PORT: serve it first"; exit 1; }
  PT=vf_probe$([ "$THINK_HIST" = 1 ] && echo _th)
  rm -rf "$PR" runs/verified/$PT runs/verified/$PT.*; mkdir -p "$PR"
  python3 scripts/req_log.py proxy --listen $(( PORT + 100 )) --upstream "http://127.0.0.1:$PORT" --out "$PR/requests" \
    $([ "$THINK_HIST" = 1 ] && echo --rename-reasoning) > "$PR/proxy.log" 2>&1 &
  PX=$!; sleep 2
  MODEL=$MODEL GPU=$GPU PORT=$PORT TAG=$PT SLICE=$PROBE_SLICE WORKERS=2 API_PORT_OFFSET=100 THINK_HIST=$THINK_HIST MAX_TOK=$MAX_TOK TOP_P=$TOP_P THINK_BUDGET=$THINK_BUDGET bash scripts/verified_box.sh run | grep -E "^verified_box|^== run"
  kill $PX 2>/dev/null
  echo "  $(ls "$PR"/requests/req_*.json 2>/dev/null | wc -l) chat requests logged"
  python3 scripts/req_log.py summarize --dir "$PR/requests" --upstream "http://127.0.0.1:$PORT" --model "$SERVED" --out "$PR" || exit 1
  H=$($PY -c "import json; print(json.load(open('$PR/probe_summary.json'))['history_rule'])")
  $PY train/mc_prep.py verify --traj runs/verified/$PT --probe "$PR/records" --history "$H" | tail -6
  echo "  render.py on the logged requests vs the server's own rendering (A2) and the reply spans (A3/A4):"
  $PY train/render.py "$PR/records" --tokenizer "$(tokdir)" --check "$PR/server_prompts" --wire "$WIRE" $([ "$THINK_HIST" = 1 ] && echo --pack --split-runs) --sample 20 \
    | grep -E "^render|^calls|^FAIL|^RESULT|^MALFORMED" | cut -c1-400
  echo "📷 the == probe line, the == verify line, the calls … line and RESULT ($(( $(date +%s) - T0 )) s)" ;;
data)
  [ "$HISTORY" = keep ] || [ "$HISTORY" = drop ] || { echo "run probe first (or set HISTORY=keep|drop)"; exit 1; }
  [ "$T2T" = on ] || [ "$T2T" = off ] || { echo "run probe first (or set T2T=on|off)"; exit 1; }
  $PY train/mc_prep.py download --dir "$MC/raw" || exit 1
  $PY train/mc_prep.py build --raw "$MC/raw" --tokenizer "$(tokdir)" --history "$HISTORY" --wire "$WIRE" --out "$MC/$ARM" \
    --n "$N" --seed "$SEED" --max-ctx "$MAX_CTX" --think-budget "$THINK_BUDGET" --reply-cap "$MAX_TOK" $([ "$T2T" = on ] && echo --text-to-thought) || exit 1
  du -sh "$MC/$ARM" | sed 's/^/  records on disk: /'
  echo "📷 the rows → … line, the printed-only line and the == line ($(( $(date +%s) - T0 )) s)" ;;
render)
  [ -d "$MC/$ARM" ] || { echo "no $MC/$ARM: run data first"; exit 1; }
  TOK="$(tokdir)"; pids=()
  for i in $(seq 0 $(( RENDER_PROCS - 1 ))); do
    $PY train/render.py "$MC/$ARM" --tokenizer "$TOK" --wire "$WIRE" $([ "$PACK" = on ] && echo --pack --split-runs) --sample 4 --shard "$i/$RENDER_PROCS" \
      --out "$MC/$ARM.part$i.jsonl" > "$MC/$ARM.render$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$MC/$ARM".part*.jsonl > "$MC/$ARM.rendered.jsonl"; rm -f "$MC/$ARM".part*.jsonl
  $PY - "$MC" "$ARM" "$RENDER_PROCS" <<'EOF'
import glob, json, re, sys
mc, arm, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tot = {"calls": 0, "targets": 0, "label": 0, "packed": 0, "seq": 0, "mixed": 0, "nothought": 0, "fail": 0, "malformed": 0}
for i in range(n):
    t = open(f"{mc}/{arm}.render{i}.log").read()
    m = re.search(r"calls (\d+) · targets (\d+) · label tokens (\d+)", t)
    if not m:
        print(f"  shard {i}: no summary line → tail {mc}/{arm}.render{i}.log"); tot["fail"] += 1; continue
    for k, v in zip(["calls", "targets", "label"], map(int, m.groups())):
        tot[k] += v
    p = re.search(r"packed label tokens (\d+) \(sequences (\d+), segments unpacked because of a mixed reply (\d+) / a no-thought reply after a tool response (\d+)\)", t)
    if p:
        for k, v in zip(["packed", "seq", "mixed", "nothought"], map(int, p.groups())):
            tot[k] += v
    tot["fail"] += len(re.findall(r"^FAIL", t, re.M)); tot["malformed"] += len(re.findall(r"^MALFORMED", t, re.M))
lens = [len(json.loads(l)["input_ids"]) for l in open(f"{mc}/{arm}.rendered.jsonl")]
tot["seq"] = tot["seq"] or len(lens)
lens.sort()
print(f"== render {arm}: {tot['calls']} calls · {tot['targets']} targets · label tokens {tot['label']} (packed {tot['packed']}) "
      f"· {tot['seq']} sequences, {sum(lens):,} tokens (len p50 {lens[len(lens)//2] if lens else 0}, max {lens[-1] if lens else 0}) "
      f"· run breaks: text+call {tot['mixed']}, no-thought {tot['nothought']} · malformed replies {tot['malformed']} "
      f"· FAIL lines {tot['fail']} → {'PASS' if tot['fail'] == 0 else 'FAIL: grep FAIL ' + mc + '/' + arm + '.render*.log'}")
EOF
  echo "📷 the == render line ($(( $(date +%s) - T0 )) s)" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
