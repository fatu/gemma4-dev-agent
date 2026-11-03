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
# · TOK (tokenizer for render/parser: the 31B id; same chat template as E4B, A2-checked 2026-11-03).
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?build|gold|gen|data}"
IDS_FILE="${IDS_FILE:-splits/rebench_kit81.txt}"; OUT="${OUT:-runs/rebench}"; ARM="${ARM:-t07}"; PASSES="${PASSES:-4}"
GPUS=(${GPUS:-0 1 2 3}); TIME_MIN="${TIME_MIN:-7}"; MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
TOK="${TOK:-$MODEL}"; PY=.venv-eval/bin/python; SFT=runs/sft/rebench31b
export TASKS_FILE="$OUT/tasks.jsonl" SNAPSHOTS_DIR="$OUT/snapshots" REBENCH_SITE="$OUT/site"
echo "rebench_gen $STAGE · ids $IDS_FILE · out $OUT · arm $ARM · passes $PASSES on GPUs ${GPUS[*]} · TIME_MIN $TIME_MIN · model $MODEL"

case "$STAGE" in
build)
  $PY scripts/rebench_build.py --ids "$IDS_FILE" --out "$OUT" 2>&1 | tee "$OUT.build.log" | grep -E "^rebench_build|NO ROW|snapshot:|FAILED|partial|^TOTAL"
  echo "📷 the lines above (every failed/partial task + TOTAL); full log $OUT.build.log" ;;
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
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  for i in $(seq 1 "$PASSES"); do
    g=${GPUS[$(( (i - 1) % ${#GPUS[@]} ))]}
    if [ "$i" -gt "${#GPUS[@]}" ]; then echo "PASSES > cards: pass $i waits for nothing — run it after the first wave"; continue; fi
    GPU=$g TAG=g$i TIME_MIN=$TIME_MIN MODEL=$MODEL ARMS="$ARM" SPLIT="$OUT/ok.txt" SUBMISSION="${SUBMISSION:-runs/base_lb024}" TRAJ=1 \
      nohup bash scripts/lane.sh run > "runs/rebench_lane$g.log" 2>&1 &
    echo "  pass g$i → GPU $g · runs/dev33_${ARM}_g$i · log runs/rebench_lane$g.log"
  done
  echo "📷 these lines; progress: tail -2 runs/rebench_lane*.log · when all lanes end: bash scripts/rebench_gen.sh data" ;;
data)
  rm -rf "$SFT" "$SFT"_pass*; mkdir -p "$SFT"
  for i in $(seq 1 "$PASSES"); do
    R="runs/dev33_${ARM}_g$i"; [ -d "$R/traj" ] || { echo "  missing $R/traj"; continue; }
    python3 train/convert_traj.py --run "$R" --source 31b_teacher --policy gemma-4-31b-it-qat-w4a16-ct --tasks "$TASKS_FILE" \
      --out "${SFT}_pass$i" | tail -1
    python3 - "$R" "${SFT}_pass$i" "$SFT" "$i" <<'PY'
import json, os, sys
run, src, dst, i = sys.argv[1:]
res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(run, "task_results.jsonl"))}
kept = 0
for f in sorted(os.listdir(src)):
    rec = json.load(open(os.path.join(src, f)))
    if rec.get("resolved"):
        json.dump(rec, open(os.path.join(dst, f"{rec['instance_id']}.g{i}.json"), "w"), ensure_ascii=False); kept += 1
print(f"  pass g{i}: resolved {sum(1 for r in res.values() if r.get('resolved'))} / {len(res)} tasks → {kept} records kept")
PY
  done
  $PY - "$SFT" <<'PY'
import glob, json, sys, collections
recs = [json.load(open(f)) for f in glob.glob(sys.argv[1] + "/*.json")]
tasks = collections.Counter(r["instance_id"] for r in recs)
print(f"== resolved trajectories {len(recs)} on {len(tasks)} tasks (kit: 103 on 55) · per task {dict(collections.Counter(tasks.values()))}")
PY
  $PY train/render.py "$SFT" --tokenizer "$TOK" --pack --sample 200 --out "$SFT.rendered.jsonl" > "$SFT.render.log" 2>&1
  grep -E "^calls|^RESULT" "$SFT.render.log" | cut -c1-260
  $PY train/parser_check.py "$SFT" --tokenizer "$TOK" --out "$SFT.parser.json" > "$SFT.parser.log" 2>&1
  grep -E "^A5|failure kinds" "$SFT.parser.log" | cut -c1-200
  $PY train/data_stats.py "$SFT" --gate 2>&1 | grep -E "records|identical|same tool|similarity|named in own|thought chars|after a tool error|gate" | cut -c1-160
  echo "📷 everything from the first 'pass g' line down" ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
