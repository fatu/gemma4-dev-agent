#!/usr/bin/env bash
# Rejection-sampling fine-tune on E4B's own resolved runs (BOX_PLAN B-6). Data stage only; training, the adapter build and
# the lanes use the existing scripts (commands in BOX_PLAN B-6).
#   bash train/rft_box.sh data      # CPU ~10 min: every pass → records (failed calls masked as targets) → the flip tasks'
#                                   # resolved runs (train/rft_select.py) → rendered + packed → the two files the training reads
# Knobs (each printed): RUNS (E4B's base passes) · TASKS · OUTD (runs/rft) · TOK · CAP (4 runs per task) · HELD (4 flip tasks
# kept out) · SEED · ALWAYS (off) · SETN (_rft: the files are named for `SET=_rft … train/drills_box.sh train`).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?data}"
PY=.venv-eval/bin/python
RUNS="${RUNS:-$(echo runs/dev33_base_e{1..3} runs/dev33_base_f{1..8})}"
TASKS="${TASKS:-runs/rebench/tasks.jsonl}"; OUTD="${OUTD:-runs/rft}"; TOK="${TOK:-google/gemma-4-E4B-it-qat-w4a16-ct}"
CAP="${CAP:-4}"; HELD="${HELD:-4}"; SEED="${SEED:-0}"; ALWAYS="${ALWAYS:-off}"; SETN="${SETN:-_rft}"
MIX=runs/sft/drills${SETN}_mix.rendered.jsonl; HO=runs/sft/drills${SETN}_heldout.rendered.jsonl
echo "rft_box $STEP · runs [$RUNS] · tasks $TASKS · out $OUTD · tokenizer $TOK · cap $CAP · held $HELD · seed $SEED · always $ALWAYS · train file $MIX · eval file $HO · code $(git log --oneline -1 | cut -c1-50)"
T0=$(date +%s)
case "$STEP" in
data)
  [ -s "$TASKS" ] || { echo "no $TASKS: set TASKS"; exit 1; }
  rm -rf "$OUTD"; mkdir -p "$OUTD" runs/sft; DIRS=()
  for R in $RUNS; do
    [ -d "$R/traj" ] || { echo "  missing $R/traj — set RUNS"; exit 1; }
    tag=${R##*_}
    python3 train/convert_traj.py --run "$R" --source e4b_onpolicy --policy gemma-4-e4b-it-qat-w4a16-ct --tasks "$TASKS" \
      --out "$OUTD/recs_$tag" > "$OUTD/convert_$tag.log" 2>&1 || { tail -3 "$OUTD/convert_$tag.log"; exit 1; }
    echo "  $tag $(grep -E '^TOTAL' "$OUTD/convert_$tag.log" | cut -c7-130)"
    DIRS+=("$OUTD/recs_$tag")
  done
  python3 train/rft_select.py "${DIRS[@]}" --out "$OUTD" --cap "$CAP" --held "$HELD" --seed "$SEED" --always "$ALWAYS" || exit 1
  for s in train heldout; do
    $PY train/render.py "$OUTD/$s" --tokenizer "$TOK" --pack --sample 100 --out "$OUTD/$s.rendered.jsonl" > "$OUTD/render_$s.log" 2>&1
    echo "  render $s: $(grep -E '^calls' "$OUTD/render_$s.log" | cut -c1-140) · $(grep -E '^RESULT' "$OUTD/render_$s.log")"
  done
  cp "$OUTD/train.rendered.jsonl" "$MIX"; cp "$OUTD/heldout.rendered.jsonl" "$HO"
  python3 - "$MIX" "$HO" <<'PY'
import json, sys
for f in sys.argv[1:]:
    n = tok = lab = 0
    for l in open(f):
        r = json.loads(l); n += 1; tok += len(r["input_ids"]); lab += sum(1 for x in r["labels"] if x != -100)
    print(f"== {f}: {n} rows · {tok:,} tokens · {lab:,} label tokens")
PY
  echo "  $(( $(date +%s) - T0 )) s · 📷 every line from 'rft_box data' down" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
