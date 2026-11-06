#!/usr/bin/env bash
# Is each local task valid on THIS machine (box / pod)? Gold patch must pass AND the empty patch must fail — only then can
# a task tell two agents apart (scripts/disc_tasks.py; TechNaom's audit, forum 756400). No GPU, no model.
#   bash scripts/validcheck.sh                 # all tasks of harness/tasks.jsonl, ~10–20 min · 📷 the == lines + lists
# Knobs (each printed): IDS (default: every task in $TASKS_FILE) · CONC (8) · OUT (runs/validcheck → $OUT/gold, $OUT/empty)
# · SANDBOX (subprocess, as eval_dev.sh; SANDBOX=docker = BOX_PLAN B-3h) · TASKS_FILE / SNAPSHOTS_DIR (harness/…).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-eval/bin/python; OUT="${OUT:-runs/validcheck}"; CONC="${CONC:-8}"; TASKS_FILE="${TASKS_FILE:-harness/tasks.jsonl}"
IDS="${IDS:-$($PY -c 'import json, sys; print(" ".join(json.loads(l)["instance_id"] for l in open(sys.argv[1]) if l.strip()))' "$TASKS_FILE")}"
n=$(echo $IDS | wc -w | tr -d ' ')
echo "validcheck: $n tasks from $TASKS_FILE · CONC $CONC · SANDBOX ${SANDBOX:-subprocess} · out $OUT/{gold,empty}"
mkdir -p "$OUT"
for m in gold empty; do
  E=0; [ "$m" = empty ] && E=1
  EMPTY=$E OUT="$OUT/$m" CONC=$CONC TASKS_FILE="$TASKS_FILE" $PY scripts/goldcheck.py $IDS > "$OUT/$m.log" 2>&1
  echo "  $m: $(grep -c ' patch → ' "$OUT/$m.log") result lines · log $OUT/$m.log"
done
echo "== gold: $(grep -c 'gold patch → RESOLVED' "$OUT/gold.log") / $n resolve with the reference patch"
echo "   gold NOT resolved: $(grep 'gold patch → NOT' "$OUT/gold.log" | awk '{print $1}' | tr '\n' ' ')"
echo "== empty: $(grep -c 'empty patch → RESOLVED' "$OUT/empty.log") / $n pass with NO patch (should be 0)"
echo "   empty RESOLVED: $(grep 'empty patch → RESOLVED' "$OUT/empty.log" | awk '{print $1}' | tr '\n' ' ')"
echo "📷 the two == lines and the two lists under them"
