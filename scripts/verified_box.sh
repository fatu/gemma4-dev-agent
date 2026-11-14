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
# All 500 of Verified on a disk that cannot hold every image (BOX_PLAN B-9): `full` = chunks of CHUNK tasks, each one
# run (split over GPUS) → graded → its images deleted (unless KEEP_IMAGES=1) → the next; a chunk whose grading report
# exists is skipped, so a stopped `full` resumes where it was (delete runs/verified/<TAG>/c<k>/ to redo chunk k):
#   SUBSET=verified GPUS="0 1 2 3" TAG=vf_e4b nohup bash scripts/verified_box.sh full > runs/verified/vf_e4b.full.log 2>&1 &
# A LoRA adapter (B-9 part 2): the same ADAPTER=<PEFT dir> for serve and for run/full; serve mounts it as LoRA module
# ADAPTER_NAME and checks it changes the output, run asks for that module instead of the base. Then compare per task:
#   ADAPTER=adapters/mc_gemma-4-E4B_r1k_lr1e-5/epoch2 GPUS="0 1 2 3 4 5 6 7" bash scripts/verified_box.sh serve
#   ADAPTER=adapters/mc_gemma-4-E4B_r1k_lr1e-5/epoch2 SUBSET=verified GPUS="0 1 2 3 4 5 6 7" THINK_HIST=1 … TAG=vf_e4b_mc full
#   python3 scripts/paired.py runs/verified/vf_e4b_th runs/verified/vf_e4b_mc     # McNemar, exits, empty patches
# Knobs (all printed): MODEL (E4B) · GPU (0) · PORT (8010) · TAG · SUBSET (MariusHobbhahn/swe-bench-verified-mini: 50 tasks,
# ≈ 5 GB of images; `verified` = all 500, ≈ 120 GB) · SLICE ("" = all) · WORKERS (8) · STEPS (250 = mini's swebench.yaml)
# · TEMP (0.2 = our latest) · MAX_TOK (8192 per reply, thought included) · PULL_TIMEOUT (900 s; mini's default 120 is short
# for a cold pull) · EVAL_JOBS (8) · MAX_LEN (32768, the scorer's) · GPU_UTIL (0.85) · MSA (2.4.6) · SWEB (4.1.0: the last
# swebench with --namespace = pull the prebuilt images; v5 dropped it) · GPUS (= GPU; several = one vLLM + one slice per card)
# · NTASK (tasks in SUBSET for the split: 50 for verified-mini, 500 for verified) · CHUNK (100, `full` only) · KEEP_IMAGES
# (0 = `full` deletes the sweb.eval.x86_64.* images after each graded chunk; they are re-pullable) · MIN_FREE_GB (50: `full`
# stops before a chunk when the Docker root has less).
# Facts behind the flags (read 2026-11-11 from the mini-swe-agent repo / PyPI / HF, not run here): `mini-extra swebench`
# defaults to --subset lite --split dev (so --split test is passed); -c replaces the default config, so swebench.yaml is
# passed first and ours second; native tool calling is the default since v2 (our vLLM has --tool-call-parser gemma4);
# cost tracking must be "ignore_errors" for a model litellm does not know; there is no context guard: a run that passes
# MAX_LEN ends with ContextWindowExceeded and an empty patch (counted in `report`). Outputs: runs/verified/<TAG>/
# (<id>/<id>.traj.json, preds.json, minisweagent.log), grading under the same dir (logs/, <model>.<TAG>.json).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?setup|serve|run|eval|full|report|stop}"
MODEL="${MODEL:-google/gemma-4-E4B-it-qat-w4a16-ct}"; SERVED="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
GPU="${GPU:-0}"; PORT="${PORT:-8010}"; TAG="${TAG:-vm_e4b}"; SUBSET="${SUBSET:-MariusHobbhahn/swe-bench-verified-mini}"
SLICE="${SLICE:-}"; WORKERS="${WORKERS:-8}"; STEPS="${STEPS:-250}"; TEMP="${TEMP:-0.2}"; MAX_TOK="${MAX_TOK:-8192}"
PULL_TIMEOUT="${PULL_TIMEOUT:-900}"; EVAL_JOBS="${EVAL_JOBS:-8}"; MAX_LEN="${MAX_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.85}"
MSA="${MSA:-2.4.6}"; SWEB="${SWEB:-4.1.0}"; VS=.venv-swe/bin; OUT=runs/verified/$TAG
case "$SUBSET" in verified) DS=princeton-nlp/SWE-bench_Verified; NT=500 ;; *) DS=$SUBSET; NT=50 ;; esac
GPUS=(${GPUS:-$GPU}); NP=${#GPUS[@]}; NTASK="${NTASK:-$NT}"
API_PORT_OFFSET="${API_PORT_OFFSET:-0}"                     # B-9 part 2 step 0: mini talks to PORT+offset (the request logger)
# THINK_HIST=1 (B-9 part 2 option ③): each part talks to a req_log.py --rename-reasoning proxy on its port+200, so vLLM
# 0.19.1 sees `reasoning` and renders the model's earlier thoughts (0 = the original B-9 setup: they are dropped).
THINK_HIST="${THINK_HIST:-0}"
# Kaggle-like sampling (user, 2026-11-13: "mini 加上输出上限8192，TOP_P=0.95，THINK_BUDGET=4096"): TOP_P "" = not sent
# (vLLM falls back to the model's generation_config); THINK_BUDGET > 0 = vLLM's per-request thinking_token_budget
# (extra_body; serve_vllm.sh sets --reasoning-config, which it needs), the same field the Kaggle harness sends.
TOP_P="${TOP_P:-}"; THINK_BUDGET="${THINK_BUDGET:-0}"
CHUNK="${CHUNK:-100}"; KEEP_IMAGES="${KEEP_IMAGES:-0}"; MIN_FREE_GB="${MIN_FREE_GB:-50}"
# ADAPTER (B-9 part 2): a PEFT dir with adapter_config.json + adapter_model.safetensors. serve mounts it through
# serve_vllm.sh (runs/verified/adapter_<ADAPTER_NAME>/adapters/<ADAPTER_NAME>/ → --enable-lora, the scorer's limits);
# run / full request model ADAPTER_NAME (default "lora": never a base alias) instead of the base. "" = the base model.
ADAPTER="${ADAPTER:-}"; ADAPTER_NAME="${ADAPTER_NAME:-lora}"; REQ="$SERVED"; AINFO=none
if [ -n "$ADAPTER" ]; then
  [ -f "$ADAPTER/adapter_config.json" ] && [ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "ADAPTER=$ADAPTER: no adapter_config.json + adapter_model.safetensors there"; exit 1; }
  case "$ADAPTER_NAME" in "$SERVED"|gemma4|"") echo "ADAPTER_NAME '$ADAPTER_NAME' must differ from the base names"; exit 1 ;; esac
  REQ="$ADAPTER_NAME"
  AINFO="$ADAPTER as '$ADAPTER_NAME' ($(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print("r", c.get("r"), "alpha", c.get("lora_alpha"))' "$ADAPTER/adapter_config.json") · sha256 $(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-12))"
fi
has_model() { curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" | grep -Eq "\"id\": ?\"$2\""; }   # port · model id
port_of() { echo $(( PORT + $1 )); }                       # card k of GPUS (0-based) → its port
echo "verified_box $STEP · model $MODEL (served $SERVED) · gpus ${GPUS[*]} → ports $PORT..$(port_of $(( NP - 1 ))) ($NP part(s), $NTASK tasks) · tag $TAG · subset $SUBSET (grading dataset $DS) · slice ${SLICE:-all} · workers $WORKERS · step limit $STEPS · temperature $TEMP · top_p ${TOP_P:-not sent} · thinking budget $THINK_BUDGET · max tokens $MAX_TOK · pull timeout $PULL_TIMEOUT s · eval jobs $EVAL_JOBS · max len $MAX_LEN · gpu util $GPU_UTIL · api port offset $API_PORT_OFFSET · think hist $THINK_HIST · adapter $AINFO (requests go to $REQ) · chunk $CHUNK keep images $KEEP_IMAGES min free $MIN_FREE_GB GB (full) · mini-swe-agent $MSA · swebench $SWEB · code $(git log --oneline -1 | cut -c1-40)"
T0=$(date +%s)
part() {  # one part: $1 port · $2 output dir · $3 slice ("" = all) · $4 log
  has_model "$1" "$REQ" || { echo "no model $REQ on port $1: run serve first (with the same ADAPTER)"; return 1; }
  mkdir -p "$2"; local cfg="$2.model.yaml" ap=$(( $1 + API_PORT_OFFSET )) px=""
  if [ "$THINK_HIST" = 1 ] && [ "$API_PORT_OFFSET" = 0 ]; then
    ap=$(( $1 + 200 ))
    python3 scripts/req_log.py proxy --listen "$ap" --upstream "http://127.0.0.1:$1" --rename-reasoning --no-log > "$2.thinkproxy.log" 2>&1 &
    px=$!; disown "$px" 2>/dev/null; sleep 1
    kill -0 "$px" 2>/dev/null || { echo "think proxy on $ap did not start: cat $2.thinkproxy.log"; return 1; }
  fi
  cat > "$cfg" <<EOF
agent:
  step_limit: $STEPS
environment:
  pull_timeout: $PULL_TIMEOUT
model:
  model_name: "hosted_vllm/$REQ"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:$ap/v1"
    temperature: $TEMP
    max_tokens: $MAX_TOK
    drop_params: true
EOF
  [ -n "$TOP_P" ] && printf '    top_p: %s\n' "$TOP_P" >> "$cfg"
  [ "$THINK_BUDGET" -gt 0 ] 2>/dev/null && printf '    extra_body:\n      thinking_token_budget: %s\n' "$THINK_BUDGET" >> "$cfg"
  MSWEA_COST_TRACKING=ignore_errors $VS/mini-extra swebench --subset "$SUBSET" --split test ${3:+--slice "$3"} \
    -m "hosted_vllm/$REQ" -c swebench.yaml -c "$cfg" -w "$WORKERS" -o "$2" > "$4" 2>&1
  local rc=$?
  [ -n "$px" ] && kill "$px" 2>/dev/null
  return $rc
}
npred() { $VS/python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }
merge_preds() {  # $1 output file · the rest: preds.json files (missing ones skipped)
  $VS/python -c '
import json, os, sys
m = {}
for p in sys.argv[2:]:
    if os.path.exists(p):
        m.update(json.load(open(p)))
json.dump(m, open(sys.argv[1], "w"), indent=1)' "$@"
}
split_run() {  # tasks $1:$2 split over GPUS, parts in parallel → $3/part<k>/, merged into $3/preds.json
  local lo=$1 hi=$2 dir=$3 pids=() files=() k a b
  for k in $(seq 0 $(( NP - 1 ))); do
    a=$(( lo + k * (hi - lo) / NP )); b=$(( lo + (k + 1) * (hi - lo) / NP ))
    part "$(port_of $k)" "$dir/part$k" "$a:$b" "$dir.part$k.run.log" & pids+=($!)
    echo "  part $k: tasks $a:$b → card ${GPUS[$k]} port $(port_of $k) · $WORKERS workers · $dir/part$k · log $dir.part$k.run.log"
    files+=("$dir/part$k/preds.json")
  done
  for k in $(seq 0 $(( NP - 1 ))); do wait "${pids[$k]}"; echo "  part $k: exit $? · $(npred "$dir/part$k/preds.json") predictions"; done
  merge_preds "$dir/preds.json" "${files[@]}"
}
eval_dir() {  # grade $1/preds.json with run id $2 → $1/<model>.$2.json · log $1.eval.log
  (cd "$1" && "$OLDPWD_ABS/$VS/python" -m swebench.harness.run_evaluation --dataset_name "$DS" --split test \
     --predictions_path preds.json --max_workers "$EVAL_JOBS" --run_id "$2" --namespace swebench > "$OLDPWD_ABS/$1.eval.log" 2>&1)
}
OLDPWD_ABS="$PWD"
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
  mkdir -p runs/verified/noadapter/adapters runs/verified; SUB=runs/verified/noadapter
  if [ -n "$ADAPTER" ]; then                 # serve_vllm.sh mounts every <SUBMISSION>/adapters/<name>/ as LoRA module <name>
    SUB=runs/verified/adapter_$ADAPTER_NAME; rm -rf "$SUB"; mkdir -p "$SUB/adapters/$ADAPTER_NAME"
    for f in adapter_config.json adapter_model.safetensors; do ln -s "$(cd "$ADAPTER" && pwd)/$f" "$SUB/adapters/$ADAPTER_NAME/$f"; done
    echo "  adapter: $SUB/adapters/$ADAPTER_NAME → $ADAPTER"
  fi
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$g" | awk -F', ' '$2 > 2000 {print $2}')
    [ -z "$busy" ] || { echo "card $g in use ($busy MiB): pick free cards with GPUS= (nothing started)"; exit 1; }
  done
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}; p=$(port_of $k)
    CUDA_VISIBLE_DEVICES=$g PATH="$PWD/.venv-eval/bin:$PATH" MODEL=$MODEL TP=1 PORT=$p MAX_LEN=$MAX_LEN GPU_UTIL=$GPU_UTIL \
      SUBMISSION=$SUB nohup bash scripts/serve_vllm.sh > runs/verified/vllm_gpu$g.log 2>&1 &
    echo $! > runs/verified/vllm_gpu$g.pid
  done
  for k in $(seq 0 $(( NP - 1 ))); do
    g=${GPUS[$k]}; p=$(port_of $k)
    for i in $(seq 1 180); do
      has_model "$p" "$REQ" && break
      kill -0 "$(cat runs/verified/vllm_gpu$g.pid)" 2>/dev/null || { echo "vLLM on card $g died: tail -30 runs/verified/vllm_gpu$g.log"; exit 1; }
      sleep 5
    done
    has_model "$p" "$REQ" || { echo "vLLM on card $g not up (or no model $REQ) after 15 min: tail -30 runs/verified/vllm_gpu$g.log"; exit 1; }
    echo "  vLLM up on card $g port $p ($(( $(date +%s) - T0 )) s since start) · $(grep -m1 'lora:' runs/verified/vllm_gpu$g.log | sed 's/^ *//' | cut -c1-120) · log runs/verified/vllm_gpu$g.log"
  done
  if [ -n "$ADAPTER" ]; then                 # is the adapter active? the same prompt, temperature 0, base vs adapter, on every card
    for k in $(seq 0 $(( NP - 1 ))); do
      python3 - "$(port_of $k)" "$SERVED" "$REQ" <<'EOF'
import json, sys, urllib.request
port, base, lora = sys.argv[1:]
def ask(m):
    body = {"model": m, "prompt": "def parse_version(s):\n    \"\"\"Split '1.2.3' into a tuple of ints.\"\"\"\n",
            "max_tokens": 24, "temperature": 0, "logprobs": 1}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=120))["choices"][0]
    return r["text"], sum(x for x in r["logprobs"]["token_logprobs"] if x is not None)
(tb, lb), (ta, la) = ask(base), ask(lora)
same = tb == ta and abs(lb - la) < 1e-6
print(f"  adapter check port {port}: base sum logprob {lb:.4f} · {lora} {la:.4f} · text {'same' if tb == ta else 'differs'}"
      + (" → IDENTICAL: the adapter does not act (check the vLLM log for LoRA warnings)" if same else " → the adapter acts"))
EOF
    done
  fi
  echo "📷 the 'vLLM up' line(s)$([ -n "$ADAPTER" ] && echo " and the adapter check lines")" ;;
run)
  if [ "$NP" -eq 1 ]; then
    echo "  config: swebench.yaml + $OUT.model.yaml"
    part "$PORT" "$OUT" "$SLICE" "$OUT.run.log"; rc=$?
    n=$(npred "$OUT/preds.json")
    echo "== run $TAG: exit $rc · $n predictions in $OUT/preds.json · $(( ($(date +%s) - T0) / 60 )) min · log $OUT.run.log"
    [ "$n" -gt 0 ] || tail -15 "$OUT.run.log"
  else
    [ -z "$SLICE" ] || { echo "SLICE is for one card only (several cards split all $NTASK tasks themselves)"; exit 1; }
    split_run 0 "$NTASK" "$OUT"
    n=$(npred "$OUT/preds.json")
    echo "== run $TAG: $NP parts · $n of $NTASK predictions merged into $OUT/preds.json · $(( ($(date +%s) - T0) / 60 )) min"
    [ "$n" -gt 0 ] || tail -15 "$OUT.part0.run.log"
  fi
  echo "📷 the part lines + the == line (and the log tail if printed)" ;;
eval)
  [ -s "$OUT/preds.json" ] || { echo "no $OUT/preds.json: run first"; exit 1; }
  eval_dir "$OUT" "$TAG"
  echo "  exit $? · $(( ($(date +%s) - T0) / 60 )) min · log $OUT.eval.log"; tail -12 "$OUT.eval.log" | cut -c1-160
  echo "📷 the lines above, then run report" ;;
full)
  [ "$SUBSET" = verified ] || echo "  note: full is meant for SUBSET=verified (now $SUBSET, $NTASK tasks)"
  for k in $(seq 0 $(( NP - 1 ))); do
    has_model "$(port_of $k)" "$REQ" || { echo "no model $REQ on port $(port_of $k): run serve first (with the same ADAPTER)"; exit 1; }
  done
  DROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
  nc=$(( (NTASK + CHUNK - 1) / CHUNK )); mkdir -p "$OUT"
  echo "  $NTASK tasks in $nc chunks of $CHUNK · docker root $DROOT · $(docker info 2>/dev/null | grep -q Username && echo 'docker hub: logged in' || echo 'docker hub: not logged in (anonymous pull limits)')"
  for c in $(seq 0 $(( nc - 1 ))); do
    lo=$(( c * CHUNK )); hi=$(( lo + CHUNK )); [ "$hi" -le "$NTASK" ] || hi=$NTASK; CD="$OUT/c$c"; TC=$(date +%s)
    if ls "$CD"/*."$TAG-c$c".json >/dev/null 2>&1; then echo "== chunk $c ($lo:$hi): graded before — skipped"; continue; fi
    free=$(df -Pk "$DROOT" | awk 'NR == 2 {print int($4 / 1048576)}')
    [ "$free" -ge "$MIN_FREE_GB" ] || { echo "STOP before chunk $c: $free GB free on $DROOT < $MIN_FREE_GB (CHUNK smaller, or free disk)"; exit 1; }
    echo "-- chunk $c: tasks $lo:$hi · $free GB free · $(date +%H:%M)"
    split_run "$lo" "$hi" "$CD"
    eval_dir "$CD" "$TAG-c$c"; erc=$?
    $VS/python - "$CD" "$TAG-c$c" <<'PY'
import glob, json, os, sys
d, rid = sys.argv[1:]
r = sorted(glob.glob(os.path.join(d, f"*.{rid}.json")))
r = json.load(open(r[-1])) if r else {}
print(f"   graded: resolved {r.get('resolved_instances', len(r.get('resolved_ids', [])))} · completed {r.get('completed_instances')} · empty {r.get('empty_patch_instances')} · errors {r.get('error_instances')}" if r else "   graded: NO REPORT (see the eval log)")
PY
    n_img=0
    if [ "$KEEP_IMAGES" != 1 ]; then
      imgs=($(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '(^|/)sweb\.eval\.x86_64\.'))
      n_img=${#imgs[@]}; [ "$n_img" -eq 0 ] || docker rmi "${imgs[@]}" >/dev/null 2>&1; docker image prune -f >/dev/null 2>&1
    fi
    merge_preds "$OUT/preds.json" "$OUT"/c*/preds.json
    echo "== chunk $c ($lo:$hi): $(npred "$CD/preds.json") predictions · eval exit $erc · $(( ($(date +%s) - TC) / 60 )) min · images deleted $n_img (left $(docker images --format '{{.Repository}}' | grep -cE 'sweb\.eval\.x86_64\.')) · $(df -Pk "$DROOT" | awk 'NR == 2 {print int($4 / 1048576)}') GB free · total so far $(npred "$OUT/preds.json")"
  done
  echo "== full $TAG done: $(npred "$OUT/preds.json") of $NTASK predictions · $(( ($(date +%s) - T0) / 60 )) min → run report"
  echo "📷 the '== chunk' lines (grep '^== chunk\|graded\|STOP' runs/verified/$TAG.full.log) + the == full line" ;;
report)
  $VS/python - "$OUT" "$TAG" <<'PY'
import collections, glob, json, os, sys
out, tag = sys.argv[1:]
preds = json.load(open(os.path.join(out, "preds.json"))) if os.path.exists(os.path.join(out, "preds.json")) else {}
reps = sorted(glob.glob(os.path.join(out, f"*.{tag}.json")) + glob.glob(os.path.join(out, "c*", f"*.{tag}-c*.json")))
r, res = {}, set()
for f in reps:                                   # one report, or one per chunk of `full`: summed
    x = json.load(open(f))
    res |= set(x.get("resolved_ids", []))
    for k in ("submitted_instances", "completed_instances", "error_instances", "empty_patch_instances"):
        r[k] = (r.get(k) or 0) + (x.get(k) or 0)
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
print(f"   exit statuses {dict(st.most_common())} · empty patches {empty}" + (f" · {len(reps)} grading reports" if len(reps) > 1 else ""))
if len(preds) > 60:                              # the full set: resolved per repo
    tot = collections.Counter(i.rsplit("-", 1)[0] for i in preds)
    got = collections.Counter(i.rsplit("-", 1)[0] for i in res)
    print("   per repo " + " · ".join(f"{k.split('__')[1]} {got[k]}/{v}" for k, v in tot.most_common()))
    try:                                         # the same model's number on the 50 Verified-mini tasks (route B's set)
        from datasets import load_dataset
        mini = {x["instance_id"] for x in load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")}
        print(f"   on the 50 Verified-mini tasks: resolved {len(res & mini)} / {len(mini & set(preds))}")
    except Exception as e:
        print(f"   (Verified-mini ids not loaded: {str(e)[:80]})")
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
