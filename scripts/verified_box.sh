#!/usr/bin/env bash
# SWE-bench Verified on the box, route B of BOX_PLAN B-8: the official per-task SWE-bench images + mini-swe-agent (a small
# bash-tool agent, not our harness) against a local vLLM model, graded by the official swebench harness. Gives a number
# comparable to papers; it is NOT our prompt / tools / budget, so it does not test a LoRA trained on our format.
#   bash scripts/verified_box.sh setup                      # once: .venv-swe with pinned mini-swe-agent + swebench (PyPI)
#   GPU=7 bash scripts/verified_box.sh serve                # vLLM (our serve_vllm.sh: tool + reasoning parser gemma4) on one card
#   SLICE=0:2 WORKERS=2 TAG=vm_smoke bash scripts/verified_box.sh run     # smoke: 2 tasks
#   TAG=vm_e4b bash scripts/verified_box.sh run             # the 50 tasks of Verified-mini (≈ 1 h at 8 workers)
#   TAG=vm_e4b bash scripts/verified_box.sh eval            # official grading in Docker → report
#   TAG=vm_e4b bash scripts/verified_box.sh report          # 📷 resolved, exit statuses, steps, empty patches
#   GPU=7 bash scripts/verified_box.sh stop                 # stop this script's vLLM
# Several cards (GPUS="0 1 2 3"): serve starts one vLLM per card on ports PORT, PORT+1, …; run splits the tasks into one
# contiguous slice per card (parts run in parallel, WORKERS each, outputs $OUT/part<k>/), then merges the predictions into
# $OUT/preds.json; eval / report / stop work on the merged run. E.g. the 31B reference on 4 cards:
#   MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPUS="0 1 2 3" PORT=8011 TAG=vm_31b bash scripts/verified_box.sh serve|run|eval|report|stop
# Knobs (all printed): MODEL (E4B) · GPU (0) · PORT (8010) · TAG · SUBSET (MariusHobbhahn/swe-bench-verified-mini: 50 tasks,
# ≈ 5 GB of images; `verified` = all 500, ≈ 120 GB) · SLICE ("" = all) · WORKERS (8) · STEPS (250 = mini's swebench.yaml)
# · TEMP (0.2 = our latest) · MAX_TOK (8192 per reply, thought included) · PULL_TIMEOUT (900 s; mini's default 120 is short
# for a cold pull) · EVAL_JOBS (8) · MAX_LEN (32768, the scorer's) · GPU_UTIL (0.85) · MSA (2.4.6) · SWEB (4.1.0: the last
# swebench with --namespace = pull the prebuilt images; v5 dropped it) · GPUS (= GPU; several = one vLLM + one slice per card)
# · NTASK (tasks in SUBSET for the split: 50 for verified-mini, 500 for verified).
# Facts behind the flags (read 2026-11-11 from the mini-swe-agent repo / PyPI / HF, not run here): `mini-extra swebench`
# defaults to --subset lite --split dev (so --split test is passed); -c replaces the default config, so swebench.yaml is
# passed first and ours second; native tool calling is the default since v2 (our vLLM has --tool-call-parser gemma4);
# cost tracking must be "ignore_errors" for a model litellm does not know; there is no context guard: a run that passes
# MAX_LEN ends with ContextWindowExceeded and an empty patch (counted in `report`). Outputs: runs/verified/<TAG>/
# (<id>/<id>.traj.json, preds.json, minisweagent.log), grading under the same dir (logs/, <model>.<TAG>.json).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?setup|serve|run|eval|report|stop}"
MODEL="${MODEL:-google/gemma-4-E4B-it-qat-w4a16-ct}"; SERVED="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
GPU="${GPU:-0}"; PORT="${PORT:-8010}"; TAG="${TAG:-vm_e4b}"; SUBSET="${SUBSET:-MariusHobbhahn/swe-bench-verified-mini}"
SLICE="${SLICE:-}"; WORKERS="${WORKERS:-8}"; STEPS="${STEPS:-250}"; TEMP="${TEMP:-0.2}"; MAX_TOK="${MAX_TOK:-8192}"
PULL_TIMEOUT="${PULL_TIMEOUT:-900}"; EVAL_JOBS="${EVAL_JOBS:-8}"; MAX_LEN="${MAX_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.85}"
MSA="${MSA:-2.4.6}"; SWEB="${SWEB:-4.1.0}"; VS=.venv-swe/bin; OUT=runs/verified/$TAG
case "$SUBSET" in verified) DS=princeton-nlp/SWE-bench_Verified; NT=500 ;; *) DS=$SUBSET; NT=50 ;; esac
GPUS=(${GPUS:-$GPU}); NP=${#GPUS[@]}; NTASK="${NTASK:-$NT}"
port_of() { echo $(( PORT + $1 )); }                       # card k of GPUS (0-based) → its port
echo "verified_box $STEP · model $MODEL (served $SERVED) · gpus ${GPUS[*]} → ports $PORT..$(port_of $(( NP - 1 ))) ($NP part(s), $NTASK tasks) · tag $TAG · subset $SUBSET (grading dataset $DS) · slice ${SLICE:-all} · workers $WORKERS · step limit $STEPS · temperature $TEMP · max tokens $MAX_TOK · pull timeout $PULL_TIMEOUT s · eval jobs $EVAL_JOBS · max len $MAX_LEN · gpu util $GPU_UTIL · mini-swe-agent $MSA · swebench $SWEB · code $(git log --oneline -1 | cut -c1-40)"
T0=$(date +%s)
case "$STEP" in
setup)
  command -v uv >/dev/null || { echo "no uv"; exit 1; }
  docker run --rm hello-world >/dev/null 2>&1 || { echo "docker does not run without sudo here: route B is not possible"; exit 1; }
  [ -x $VS/python ] || uv venv -q --python 3.12 .venv-swe
  uv pip install -q --python $VS/python "mini-swe-agent==$MSA" "swebench==$SWEB" 2>&1 | tail -3
  $VS/python -c "import importlib.metadata as m; print('  installed: mini-swe-agent', m.version('mini-swe-agent'), '· swebench', m.version('swebench'), '· litellm', m.version('litellm'))"
  ls $VS/mini-extra >/dev/null && echo "  mini-extra: ok"
  echo "📷 the lines above" ;;
serve)
  mkdir -p runs/verified/noadapter/adapters runs/verified
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$g" | awk -F', ' '$2 > 2000 {print $2}')
    [ -z "$busy" ] || { echo "card $g in use ($busy MiB): pick free cards with GPUS= (nothing started)"; exit 1; }
  done
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}; p=$(port_of $k)
    CUDA_VISIBLE_DEVICES=$g PATH="$PWD/.venv-eval/bin:$PATH" MODEL=$MODEL TP=1 PORT=$p MAX_LEN=$MAX_LEN GPU_UTIL=$GPU_UTIL \
      SUBMISSION=runs/verified/noadapter nohup bash scripts/serve_vllm.sh > runs/verified/vllm_gpu$g.log 2>&1 &
    echo $! > runs/verified/vllm_gpu$g.pid
  done
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}; p=$(port_of $k)
    for i in $(seq 1 180); do
      curl -s --max-time 2 "http://127.0.0.1:$p/v1/models" | grep -q "$SERVED" && break
      kill -0 "$(cat runs/verified/vllm_gpu$g.pid)" 2>/dev/null || { echo "vLLM on card $g died: tail -30 runs/verified/vllm_gpu$g.log"; exit 1; }
      sleep 5
    done
    curl -s --max-time 2 "http://127.0.0.1:$p/v1/models" | grep -q "$SERVED" || { echo "vLLM on card $g not up after 15 min"; exit 1; }
    echo "  vLLM up on card $g port $p ($(( $(date +%s) - T0 )) s since start) · log runs/verified/vllm_gpu$g.log"
  done
  echo "📷 the 'vLLM up' line(s)" ;;
run)
  part() {  # one part: $1 port · $2 output dir · $3 slice ("" = all) · $4 log
    curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" | grep -q "$SERVED" || { echo "no $SERVED on port $1: run serve first"; return 1; }
    mkdir -p "$2"; local cfg="$2.model.yaml"
    cat > "$cfg" <<EOF
agent:
  step_limit: $STEPS
environment:
  pull_timeout: $PULL_TIMEOUT
model:
  model_name: "hosted_vllm/$SERVED"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:$1/v1"
    temperature: $TEMP
    max_tokens: $MAX_TOK
    drop_params: true
EOF
    MSWEA_COST_TRACKING=ignore_errors $VS/mini-extra swebench --subset "$SUBSET" --split test ${3:+--slice "$3"} \
      -m "hosted_vllm/$SERVED" -c swebench.yaml -c "$cfg" -w "$WORKERS" -o "$2" > "$4" 2>&1
  }
  npred() { $VS/python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }
  if [ "$NP" -eq 1 ]; then
    echo "  config: swebench.yaml + $OUT.model.yaml"
    part "$PORT" "$OUT" "$SLICE" "$OUT.run.log"; rc=$?
    n=$(npred "$OUT/preds.json")
    echo "== run $TAG: exit $rc · $n predictions in $OUT/preds.json · $(( ($(date +%s) - T0) / 60 )) min · log $OUT.run.log"
    [ "$n" -gt 0 ] || tail -15 "$OUT.run.log"
  else
    [ -z "$SLICE" ] || { echo "SLICE is for one card only (several cards split all $NTASK tasks themselves)"; exit 1; }
    pids=()
    for k in $(seq 0 $(( NP - 1 ))); do
      a=$(( k * NTASK / NP )); b=$(( (k + 1) * NTASK / NP ))
      part "$(port_of $k)" "$OUT/part$k" "$a:$b" "$OUT.part$k.run.log" & pids+=($!)
      echo "  part $k: tasks $a:$b → card ${GPUS[$k]} port $(port_of $k) · $WORKERS workers · $OUT/part$k · log $OUT.part$k.run.log"
    done
    for k in $(seq 0 $(( NP - 1 ))); do wait "${pids[$k]}"; echo "  part $k: exit $? · $(npred "$OUT/part$k/preds.json") predictions"; done
    $VS/python -c '
import json, os, sys
out, n = sys.argv[1], int(sys.argv[2])
merged = {}
for k in range(n):
    p = os.path.join(out, f"part{k}", "preds.json")
    if os.path.exists(p):
        merged.update(json.load(open(p)))
json.dump(merged, open(os.path.join(out, "preds.json"), "w"), indent=1)' "$OUT" "$NP"
    n=$(npred "$OUT/preds.json")
    echo "== run $TAG: $NP parts · $n of $NTASK predictions merged into $OUT/preds.json · $(( ($(date +%s) - T0) / 60 )) min"
    [ "$n" -gt 0 ] || tail -15 "$OUT.part0.run.log"
  fi
  echo "📷 the part lines + the == line (and the log tail if printed)" ;;
eval)
  [ -s "$OUT/preds.json" ] || { echo "no $OUT/preds.json: run first"; exit 1; }
  (cd "$OUT" && ../../../$VS/python -m swebench.harness.run_evaluation --dataset_name "$DS" --split test \
     --predictions_path preds.json --max_workers "$EVAL_JOBS" --run_id "$TAG" --namespace swebench > ../$TAG.eval.log 2>&1)
  echo "  exit $? · $(( ($(date +%s) - T0) / 60 )) min · log runs/verified/$TAG.eval.log"; tail -12 "runs/verified/$TAG.eval.log" | cut -c1-160
  echo "📷 the lines above, then run report" ;;
report)
  $VS/python - "$OUT" "$TAG" <<'PY'
import collections, glob, json, os, sys
out, tag = sys.argv[1:]
preds = json.load(open(os.path.join(out, "preds.json"))) if os.path.exists(os.path.join(out, "preds.json")) else {}
rep = sorted(glob.glob(os.path.join(out, f"*.{tag}.json")))
r = json.load(open(rep[-1])) if rep else {}
res = set(r.get("resolved_ids", []))
st, calls = collections.Counter(), []
for f in glob.glob(os.path.join(out, "**", "*.traj.json"), recursive=True):     # part<k>/<id>/ on several cards
    info = json.load(open(f)).get("info", {})
    st[info.get("exit_status") or "?"] += 1
    c = (info.get("model_stats") or {}).get("api_calls")
    if isinstance(c, int):
        calls.append(c)
calls.sort()
empty = sum(1 for p in preds.values() if not (p.get("model_patch") or "").strip())
print(f"== {tag}: resolved {len(res)} / {len(preds)} predictions"
      + (f" (report: submitted {r.get('submitted_instances')}, completed {r.get('completed_instances')}, errors {r.get('error_instances')}, empty {r.get('empty_patch_instances')})" if r else " (no grading report yet: run eval)"))
print(f"   exit statuses {dict(st.most_common())} · empty patches {empty}")
if calls:
    print(f"   model calls per task p50 {calls[len(calls) // 2]} · p90 {calls[int(len(calls) * 0.9)]} · max {calls[-1]}")
PY
  echo "📷 the == block" ;;
stop)
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}; p=$(port_of $k)
    [ -f runs/verified/vllm_gpu$g.pid ] && kill "$(cat runs/verified/vllm_gpu$g.pid)" 2>/dev/null; sleep 3
    pkill -f "vllm serve.*--port $p" 2>/dev/null; echo "  stopped the vLLM of card $g / port $p"
  done ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
