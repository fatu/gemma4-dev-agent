#!/usr/bin/env bash
# Token-level on-policy distillation 31B → E4B (BOX_PLAN B-4). Steps S0 / S0b: does the 31B teacher point at E4B's own
# errors, and where would a distillation signal land? No training. One stage per call; each ends with a 📷 line.
#   bash train/opd_box.sh s0data                                           # CPU ~5–10 min: E4B rollouts → records → rows
#   nohup bash train/opd_box.sh score > runs/opd_score.log 2>&1 &          # all cards: the 31B scores every E4B reply token
#                                                                          #   (top-20), one process per card (~4 min a shard)
#   nohup bash train/opd_box.sh student > runs/opd_student.log 2>&1 &      # S0b, all cards: the same rows scored by E4B itself
#   bash train/opd_box.sh probe                                            # CPU ~5–10 min: kd_probe (S0 lines + S0b BIRTH /
#                                                                          #   SEV / SUBMIT / MAP + the verdict line)
# s0data: E4B's own rebench rollouts (B-3p step 4: runs/dev33_base_e1..e3, failed tasks included) → convert_traj with the
# failed calls KEPT as targets (--mask-errors off: the distillation has to see the wrong tokens) → render.py --pack →
# rows.jsonl (each row tagged with its pass). score: teacher_score.py, rows dealt to the cards (row i → card i mod N).
# probe: kd_probe.py (read-only). Knobs (each printed): RUNS · TASKS · OUTD · GPUS (default every card) · TOK (E4B, the
# rows' tokenizer) · T31 (the teacher) · E4BM (the student model for `student`; default TOK) · K · MBT · GPU_UTIL · BATCH · MAX_LEN · FAKE (laptop test only: same|other =
# teacher_score --fake, no engine).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?s0data|score|student|probe}"
PY=.venv-eval/bin/python
RUNS="${RUNS:-runs/dev33_base_e1 runs/dev33_base_e2 runs/dev33_base_e3}"
TASKS="${TASKS:-runs/rebench/tasks.jsonl}"; OUTD="${OUTD:-runs/opd/s0}"
ALLGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
GPUS="${GPUS:-$ALLGPUS}"; IFS=, read -ra GL <<< "$GPUS"; NG=${#GL[@]}
TOK="${TOK:-google/gemma-4-E4B-it-qat-w4a16-ct}"; T31="${T31:-google/gemma-4-31B-it-qat-w4a16-ct}"; E4BM="${E4BM:-$TOK}"
FAKE="${FAKE:-}"; K="${K:-20}"; MBT="${MBT:-4096}"; GPU_UTIL="${GPU_UTIL:-0.80}"; BATCH="${BATCH:-8}"; MAX_LEN="${MAX_LEN:-32768}"
mkdir -p "$OUTD"
echo "opd_box $STEP · runs [$RUNS] · tasks $TASKS · out $OUTD · cards $GPUS ($NG) · tokenizer $TOK · teacher $T31 · student $E4BM · top-k $K · max_num_batched_tokens $MBT · gpu_util $GPU_UTIL · batch $BATCH rows · max_len $MAX_LEN${FAKE:+ · FAKE TEACHER $FAKE} · code $(git log --oneline -1 | cut -c1-50)"
T0=$(date +%s)
case "$STEP" in
s0data)
  [ -s "$TASKS" ] || { echo "no $TASKS (the rebench tasks of B-3n): set TASKS"; exit 1; }
  rm -rf "$OUTD"/recs_* "$OUTD"/*.rendered.jsonl "$OUTD"/rows.jsonl "$OUTD"/t31_*.npz "$OUTD"/s_*.npz
  for R in $RUNS; do
    [ -d "$R/traj" ] || { echo "  missing $R/traj — set RUNS to E4B's rollout dirs"; exit 1; }
    tag=${R##*_}
    python3 train/convert_traj.py --run "$R" --source e4b_onpolicy --policy gemma-4-e4b-it-qat-w4a16-ct --tasks "$TASKS" \
      --mask-errors off --out "$OUTD/recs_$tag" > "$OUTD/convert_$tag.log" 2>&1 || { tail -3 "$OUTD/convert_$tag.log"; exit 1; }
    echo "  $tag $(grep -E '^TOTAL' "$OUTD/convert_$tag.log" | cut -c7-130)"
    echo "     $(grep -E '^tool-error masking \(' "$OUTD/convert_$tag.log" | cut -c1-170)"
    $PY train/render.py "$OUTD/recs_$tag" --tokenizer "$TOK" --pack --sample 50 --out "$OUTD/$tag.rendered.jsonl" > "$OUTD/render_$tag.log" 2>&1
    echo "     render: $(grep -E '^calls' "$OUTD/render_$tag.log" | cut -c1-120) · $(grep -E '^RESULT' "$OUTD/render_$tag.log")"
  done
  python3 - "$OUTD" $RUNS <<'PY'
import json, sys
outd, runs = sys.argv[1], sys.argv[2:]
n = tok = lab = 0
with open(f"{outd}/rows.jsonl", "w") as w:
    for r in runs:
        tag = r.rsplit("_", 1)[-1]
        for l in open(f"{outd}/{tag}.rendered.jsonl"):
            row = json.loads(l); row["tag"] = tag; w.write(json.dumps(row) + "\n")
            n += 1; tok += len(row["input_ids"]); lab += sum(1 for x in row["labels"] if x != -100)
print(f"== s0data: {n} rows · {tok:,} tokens · {lab:,} E4B reply tokens (thought + text + call, failed calls included) → {outd}/rows.jsonl")
PY
  echo "  $(( $(date +%s) - T0 )) s · 📷 every line from 'opd_box s0data' down" ;;
score|student)
  [ -s "$OUTD/rows.jsonl" ] || { echo "run s0data first"; exit 1; }
  if [ "$STEP" = score ]; then MDL=$T31; PRE=t31; LOGP=score; else MDL=$E4BM; PRE=s; LOGP=student; fi
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") && $2 > 2000 {printf "%s(%s MiB) ", $1, $2}')
  [ -z "$busy" ] || { echo "cards in use: $busy— stop the servers / lanes on them first (or set GPUS)"; exit 1; }
  rm -f "$OUTD"/${PRE}_*.npz
  for k in "${!GL[@]}"; do
    CUDA_VISIBLE_DEVICES=${GL[$k]} nohup $PY train/teacher_score.py "$OUTD/rows.jsonl" --model "$MDL" --tokenizer "$TOK" --k "$K" \
      --mbt "$MBT" --gpu-util "$GPU_UTIL" --batch "$BATCH" --max-len "$MAX_LEN" --shard "$k" --nshards "$NG" \
      ${FAKE:+--fake $FAKE} --out "$OUTD/${PRE}_$k.npz" > "$OUTD/${LOGP}_$k.log" 2>&1 &
  done
  echo "  $NG scoring processes started with $MDL (progress: tail -2 $OUTD/${LOGP}_0.log)"; wait
  for k in "${!GL[@]}"; do
    if grep -q "^== teacher_score" "$OUTD/${LOGP}_$k.log"; then grep -E "^(  per span|== teacher_score)" "$OUTD/${LOGP}_$k.log" | cut -c1-230
    else echo "  card ${GL[$k]}: no result — tail $OUTD/${LOGP}_$k.log:"; tail -3 "$OUTD/${LOGP}_$k.log"; fi
  done
  n=$(ls "$OUTD"/${PRE}_*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "== $STEP: $n of $NG shards wrote their file ($MDL) · $(( ($(date +%s) - T0) / 60 )) min · 📷 the == lines and this one" ;;
probe)
  ls "$OUTD"/t31_*.npz >/dev/null 2>&1 || { echo "run score first"; exit 1; }
  SN=""; ls "$OUTD"/s_*.npz >/dev/null 2>&1 && SN="$OUTD/s_*.npz"
  echo "  student scores: ${SN:-none (run 'student' for the push / MAP lines)}"
  $PY train/kd_probe.py "$OUTD/rows.jsonl" --npz "$OUTD/t31_*.npz" ${SN:+--student-npz "$SN"} --recs "$OUTD" --tokenizer "$TOK" > "$OUTD/probe.log" 2>&1
  grep -vE "^kd_probe|[Ww]arn" "$OUTD/probe.log" | grep -v '^$' | cut -c1-700
  echo "  $(( $(date +%s) - T0 )) s · 📷 every line from the first counts line down" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
