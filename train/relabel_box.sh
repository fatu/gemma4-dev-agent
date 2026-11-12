#!/usr/bin/env bash
# Corrected labels for E4B's own failed edits (BOX_PLAN B-7, the relabel stage). Data stage only; training, the adapter
# build and the lanes use the existing scripts (commands in BOX_PLAN B-7).
#   bash train/relabel_box.sh data     # CPU ~5–15 min: every pass replayed on its snapshot → corrected edit calls + ok edits
#                                      # (train/relabel_edits.py) → rendered + packed → the two files the training reads
# Knobs (each printed): RUNS (the 8 passes of B-7 step 3 + the 11 no-adapter passes on the first 67 tasks) · TASKS · SNAP
# (one entry per task set) · OUTD (runs/sft/e4b_relabel) · TOK · MIN_RATIO ·
# NEED_VISIBLE · PER_KEY · TASK_CAP · OK_RATIO · HELD (share of the repos held out) · SEED · SETN (_relab: the files are
# named for `SET=_relab … train/drills_box.sh train`) · ACCUM / EPOCHS (only for the optimizer-step line printed at the end).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?data}"
PY=.venv-eval/bin/python
RUNS="${RUNS:-$(echo runs/dev33_base_m{1..4} runs/dev33_t07_m{5..8} runs/dev33_base_e{1..3} runs/dev33_base_f{1..8})}"
TASKS="${TASKS:-runs/rebench2/tasks.jsonl runs/rebench/tasks.jsonl}"; SNAP="${SNAP:-runs/rebench2/snapshots runs/rebench/snapshots}"; OUTD="${OUTD:-runs/sft/e4b_relabel}"
TOK="${TOK:-google/gemma-4-E4B-it-qat-w4a16-ct}"; MIN_RATIO="${MIN_RATIO:-0.75}"; NEED_VISIBLE="${NEED_VISIBLE:-1}"
PER_KEY="${PER_KEY:-2}"; TASK_CAP="${TASK_CAP:-24}"; OK_RATIO="${OK_RATIO:-1.0}"; HELD="${HELD:-0.1}"; SEED="${SEED:-0}"
SETN="${SETN:-_relab}"; ACCUM="${ACCUM:-16}"; EPOCHS="${EPOCHS:-3}"
MIX=runs/sft/drills${SETN}_mix.rendered.jsonl; HO=runs/sft/drills${SETN}_heldout.rendered.jsonl
echo "relabel_box $STEP · runs [$RUNS] · tasks $TASKS · snapshots $SNAP · out $OUTD · tokenizer $TOK · min_ratio $MIN_RATIO · need_visible $NEED_VISIBLE · per_key $PER_KEY · task_cap $TASK_CAP · ok_ratio $OK_RATIO · held $HELD · seed $SEED · train file $MIX · eval file $HO · steps line for accum $ACCUM × $EPOCHS epochs · code $(git log --oneline -1 | cut -c1-50)"
T0=$(date +%s)
case "$STEP" in
data)
  for f in $TASKS; do [ -s "$f" ] || { echo "no $f: set TASKS"; exit 1; }; done
  for d in $SNAP; do [ -d "$d" ] || { echo "no $d: set SNAP"; exit 1; }; done
  for R in $RUNS; do [ -d "$R/traj" ] || { echo "  missing $R/traj — set RUNS"; exit 1; }; done
  rm -rf "$OUTD" "${OUTD}_heldout"; mkdir -p runs/sft
  python3 train/relabel_edits.py --runs $RUNS --tasks $TASKS --snapshots $SNAP --out "$OUTD" --min-ratio "$MIN_RATIO" \
    --need-visible "$NEED_VISIBLE" --per-key "$PER_KEY" --task-cap "$TASK_CAP" --ok-ratio "$OK_RATIO" --heldout-frac "$HELD" \
    --seed "$SEED" 2>&1 | tee "$OUTD.relabel.log" | cut -c1-400 || exit 1
  for s in "" _heldout; do
    $PY train/render.py "$OUTD$s" --tokenizer "$TOK" --pack --sample 100 --out "$OUTD$s.rendered.jsonl" > "$OUTD$s.render.log" 2>&1
    echo "  render ${s:-train}: $(grep -E '^calls' "$OUTD$s.render.log" | cut -c1-200) · $(grep -E '^RESULT' "$OUTD$s.render.log")"
    grep -E "^(A[0-9]|PACK|MALFORMED)" "$OUTD$s.render.log" | head -5 | cut -c1-200
  done
  cp "$OUTD.rendered.jsonl" "$MIX"; cp "${OUTD}_heldout.rendered.jsonl" "$HO"
  python3 - "$ACCUM" "$EPOCHS" "$MIX" "$HO" <<'PY'
import json, math, sys
accum, epochs = int(sys.argv[1]), int(sys.argv[2])
for k, f in enumerate(sys.argv[3:]):
    n = tok = lab = big = 0
    for l in open(f):
        r = json.loads(l); n += 1; tok += len(r["input_ids"]); lab += sum(1 for x in r["labels"] if x != -100)
        big += len(r["input_ids"]) > 26624
    print(f"== {f}: {n} sequences · {tok:,} tokens · {lab:,} label tokens · longer than 26,624 tokens (the training drops them) {big}")
    if k == 0:
        s = math.ceil((n - big) / accum)
        need = min(6, max(epochs, math.ceil(150 / max(s, 1))))
        print(f"== optimizer steps: {s} per epoch at accum {accum} → {s * epochs} in {epochs} epochs (B-6 had 33 in all) · "
              f"for ≥ 150 steps: EPOCHS={need} → {s * need} steps" + ("" if s * need >= 150 else " — still under 150: stop + photo"))
PY
  echo "  $(( $(date +%s) - T0 )) s · 📷 every line from 'relabel_box data' down" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
