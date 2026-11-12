#!/usr/bin/env bash
# Our own native 31B training trajectories, the dmitrii-kit way but with OUR latest setting (BOX_PLAN.md B-3n):
# 31B through the harness on public SWE-rebench tasks (the kit's 81 by default), several sampled passes, keep the
# resolved ones → call-level records with our prompt as the model really saw it (no re-wrap needed) → rendered +
# checked training file. One stage per call; every stage ends with a 📷 summary.
#   bash scripts/rebench_gen.sh build      # internet: rows (pinned parquet) + snapshots + per-task deps   (~20–40 min)
#   bash scripts/rebench_gen.sh gold       # no GPU: the gold patch must pass here → runs/rebench/ok.txt      (~10–20 min)
#   bash scripts/rebench_gen.sh gen        # 4 lanes (one card each, offline), $PASSES sampled passes of ok.txt   (~3–5 h)
#   bash scripts/rebench_gen.sh data       # after the lanes: resolved trajectories → runs/sft/rebench31b.rendered.jsonl
# Knobs (each printed): IDS_FILE (splits/rebench_kit81.txt) · OUT (runs/rebench) · ARM (t07: our latest + temperature
# 0.7, the plan's sampling temperature; the kit used 0.65) · PASSES (4) · GPUS ("0 1 2 3") · TIME_MIN (7: our latest's
# literal minutes; the box is faster than an L4, so lenient — fine for data, not for scores) · MODEL (31B) · CONC_GOLD (8)
# · KEEP (resolved | all: every trajectory, failed tool calls stay masked) · GTAG (g: pass tags g1..g4 → runs/dev33_<ARM>_g<i>) · SFT (runs/sft/rebench31b) · SRC (31b_teacher) · POLICY (the
# 31B alias) — E4B's own rollouts (BOX_PLAN B-3p): MODEL=<E4B id> LATEST=e4b ADAPTER= ARM=base GTAG=e SFT=runs/sft/rebenche4b
# SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct TOK=<E4B id>
# · TOK (tokenizer for render/parser: the 31B id; same chat template as E4B, A2-checked 2026-11-03) · MAX_TURNS (80: the
# scorer reads max_turns: 80 from our eval_config.yaml and then prints "- Max loop iterations: 80 turns" in the task
# prompt; the local CLI ignores eval_config, so without it the generated prompts would lack that line — as the kit's have it).
# More tasks (BOX_PLAN B-7): BUILD_JOBS (1; > 1 = that many build processes, tasks dealt by repo, outputs merged) · LEAN (0;
# 1 = blobless git clones deleted after use) · MIN_FREE_GB (0 = no guard) · SNAP_MAX_MB (0 = any size) · UV_CACHE ("";
# a dir under OUT = one hard-linked wheel cache) · KILL_VLLM (1; 0 = leave other lanes' servers alone) · ARM_BY_PASS
# ("" = ARM for every pass; "base base t07 t07" = pass i gets word i) · LANE_CONC (1; lane.sh's eval concurrency).
# SWE-bench Verified in our harness (BOX_PLAN B-8 route A, via scripts/verified_a.sh): SPECS ("" = none; a json of install
# configs per repo/version) · DEPS_TIMEOUT (1800 s per install command).
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?build|gold|gen|data}"
IDS_FILE="${IDS_FILE:-splits/rebench_kit81.txt}"; OUT="${OUT:-runs/rebench}"; ARM="${ARM:-t07}"; PASSES="${PASSES:-4}"
GPUS=(${GPUS:-0 1 2 3}); TIME_MIN="${TIME_MIN:-7}"; MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
TOK="${TOK:-$MODEL}"; PY=.venv-eval/bin/python; SFT="${SFT:-runs/sft/rebench31b}"; MAX_TURNS="${MAX_TURNS:-80}"
GTAG="${GTAG:-g}"; KEEP="${KEEP:-resolved}"; SRC="${SRC:-31b_teacher}"; POLICY="${POLICY:-gemma-4-31b-it-qat-w4a16-ct}"
BUILD_JOBS="${BUILD_JOBS:-1}"; LEAN="${LEAN:-0}"; MIN_FREE_GB="${MIN_FREE_GB:-0}"; SNAP_MAX_MB="${SNAP_MAX_MB:-0}"; UV_CACHE="${UV_CACHE:-}"
SPECS="${SPECS:-}"; DEPS_TIMEOUT="${DEPS_TIMEOUT:-1800}"; KILL_VLLM="${KILL_VLLM:-1}"; ARM_BY_PASS=(${ARM_BY_PASS:-}); export LANE_CONC="${LANE_CONC:-1}"
arm_of() { local w="${ARM_BY_PASS[$(( $1 - 1 ))]:-}"; echo "${w:-$ARM}"; }
export TASKS_FILE="$OUT/tasks.jsonl" SNAPSHOTS_DIR="$OUT/snapshots" REBENCH_SITE="$OUT/site"
echo "rebench_gen $STAGE · ids $IDS_FILE · out $OUT · arm $ARM · arms by pass [${ARM_BY_PASS[*]:-}] · passes $PASSES on GPUs ${GPUS[*]} · lane concurrency $LANE_CONC · kill vllm $KILL_VLLM · TIME_MIN $TIME_MIN · MAX_TURNS $MAX_TURNS · model $MODEL · tags $GTAG<i> · keep $KEEP · sft $SFT ($SRC, $POLICY) · LATEST ${LATEST:-31b} ADAPTER ${ADAPTER-unset} · build jobs $BUILD_JOBS lean $LEAN min free ${MIN_FREE_GB} GB snapshot max ${SNAP_MAX_MB} MB uv cache ${UV_CACHE:-default} specs ${SPECS:-none} deps timeout ${DEPS_TIMEOUT} s"

case "$STAGE" in
build)
  mkdir -p "$OUT"; BF=(--ids "$IDS_FILE" --out "$OUT" --min-free-gb "$MIN_FREE_GB" --max-snapshot-mb "$SNAP_MAX_MB" --deps-timeout "$DEPS_TIMEOUT")
  [ -n "$SPECS" ] && BF+=(--specs "$SPECS")
  [ "$LEAN" = 1 ] && BF+=(--lean); [ -n "$UV_CACHE" ] && BF+=(--uv-cache "$UV_CACHE")
  if [ "$BUILD_JOBS" -le 1 ]; then
    $PY scripts/rebench_build.py "${BF[@]}" 2>&1 | tee "$OUT.build.log" | grep -E "^rebench_build|NO ROW|snapshot:|FAILED|partial|dropped|^STOP|^TOTAL"
  else
    rm -f "$OUT"/tasks.shard*.jsonl "$OUT"/built.shard*.txt
    for k in $(seq 0 $(( BUILD_JOBS - 1 ))); do
      $PY scripts/rebench_build.py "${BF[@]}" --shard "$k" --nshards "$BUILD_JOBS" > "$OUT.build$k.log" 2>&1 &
    done
    echo "  $BUILD_JOBS build processes (logs $OUT.build<k>.log; progress: tail -1 $OUT.build*.log)"; wait
    cat "$OUT".build[0-9]*.log > "$OUT.build.log"
    grep -hE "NO ROW|snapshot:|FAILED|partial|dropped|^STOP|^TOTAL" "$OUT".build[0-9]*.log | cut -c1-200
    cat "$OUT"/tasks.shard*.jsonl > "$OUT/tasks.jsonl"; cat "$OUT"/built.shard*.txt > "$OUT/built.txt"
    echo "== build: $(wc -l < "$OUT/built.txt") of $(grep -vc '^#' "$IDS_FILE") tasks built → $OUT/tasks.jsonl · site $(du -sh "$OUT/site" | cut -f1) · snapshots $(du -sh "$OUT/snapshots" | cut -f1) · $(df -h "$OUT" | tail -1 | awk '{print $4}') free"
  fi
  echo "📷 the lines above (every failed/partial/dropped task + TOTAL / == build); full log $OUT.build.log" ;;
gold)
  [ -s "$OUT/built.txt" ] || { echo "run build first"; exit 1; }
  OUT_G="$OUT/gold"; rm -rf "$OUT_G"
  OUT="$OUT_G" CONC="${CONC_GOLD:-8}" $PY scripts/goldcheck.py $(cat "$OUT/built.txt") > "$OUT_G.log" 2>&1
  grep -E " gold patch → RESOLVED" "$OUT_G.log" | awk '{print $1}' | sort > "$OUT/ok.txt"
  n=$(wc -l < "$OUT/built.txt"); k=$(wc -l < "$OUT/ok.txt")
  echo "== gold: $k / $n built tasks resolve with the reference patch here → $OUT/ok.txt (the kit resolved 55 of its 81)"
  grep -E " gold patch → NOT" "$OUT_G.log" | cut -c1-140
  echo "📷 the == line + the NOT list" ;;
gen)
  [ -s "$OUT/ok.txt" ] || { echo "run gold first"; exit 1; }
  [ "$KILL_VLLM" = 1 ] && { pkill -f "vllm serve" 2>/dev/null; sleep 5; }
  for i in $(seq 1 "$PASSES"); do
    g=${GPUS[$(( (i - 1) % ${#GPUS[@]} ))]}
    if [ "$i" -gt "${#GPUS[@]}" ]; then echo "PASSES > cards: pass $i waits for nothing — run it after the first wave"; continue; fi
    A=$(arm_of $i)
    GPU=$g TAG=$GTAG$i TIME_MIN=$TIME_MIN MAX_TURNS=$MAX_TURNS MODEL=$MODEL ARMS="$A" SPLIT="$OUT/ok.txt" SUBMISSION="${SUBMISSION:-runs/base_lb024}" TRAJ=1 \
      nohup bash scripts/lane.sh run > "runs/rebench_lane$g.log" 2>&1 &
    echo "  pass $GTAG$i → GPU $g · arm $A · runs/dev33_${A}_$GTAG$i · log runs/rebench_lane$g.log"
  done
  echo "📷 these lines; progress: tail -2 runs/rebench_lane*.log · when all lanes end: bash scripts/rebench_gen.sh data" ;;
data)
  rm -rf "$SFT" "$SFT"_pass*; mkdir -p "$SFT"
  for i in $(seq 1 "$PASSES"); do
    R="runs/dev33_$(arm_of $i)_$GTAG$i"; [ -d "$R/traj" ] || { echo "  missing $R/traj"; continue; }
    python3 train/convert_traj.py --run "$R" --source "$SRC" --policy "$POLICY" --tasks "$TASKS_FILE" \
      --out "${SFT}_pass$i" | tail -1
    python3 - "$R" "${SFT}_pass$i" "$SFT" "$GTAG$i" "$KEEP" <<'PY'
import json, os, sys
run, src, dst, i, keep = sys.argv[1:]
res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(run, "task_results.jsonl"))}
kept = 0
for f in sorted(os.listdir(src)):
    rec = json.load(open(os.path.join(src, f)))
    if keep == "all" or rec.get("resolved"):
        json.dump(rec, open(os.path.join(dst, f"{rec['instance_id']}.{i}.json"), "w"), ensure_ascii=False); kept += 1
print(f"  pass {i}: resolved {sum(1 for r in res.values() if r.get('resolved'))} / {len(res)} tasks → {kept} records kept (KEEP={keep})")
PY
  done
  $PY - "$SFT" <<'PY'
import glob, json, sys, collections
recs = [json.load(open(f)) for f in glob.glob(sys.argv[1] + "/*.json")]
tasks = collections.Counter(r["instance_id"] for r in recs)
print(f"== trajectories kept {len(recs)} on {len(tasks)} tasks (kit: 103 on 55) · per task {dict(collections.Counter(tasks.values()))}")
PY
  $PY train/render.py "$SFT" --tokenizer "$TOK" --pack --sample 200 --out "$SFT.rendered.jsonl" > "$SFT.render.log" 2>&1
  grep -E "^calls|^RESULT" "$SFT.render.log" | cut -c1-260
  $PY train/parser_check.py "$SFT" --tokenizer "$TOK" --out "$SFT.parser.json" > "$SFT.parser.log" 2>&1
  grep -E "^A5|failure kinds" "$SFT.parser.log" | cut -c1-200
  $PY train/data_stats.py "$SFT" --gate 2>&1 | grep -E "records|identical|same tool|similarity|named in own|thought chars|after a tool error|gate" | cut -c1-160
  echo "📷 everything from the first 'pass g' line down" ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
