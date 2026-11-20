#!/usr/bin/env bash
# B-11: teacher trajectories inside the competition harness, one command per step (BOX_PLAN.md B-11). The 8-card box, no GPU.
#   bash scripts/teacher_gen.sh setup     # docker, versions, the V21 files, the task table → runs/teach/tasks.jsonl, models.yaml  (no token)
#   bash scripts/teacher_gen.sh check     # step 2: 6 tasks (3 per source) × 1 attempt, gold + empty patch graded            (cents)
#   bash scripts/teacher_gen.sh pilot     # step 3: the 30 tasks of B-10's pilot30 × 1 attempt                               (paid)
#   bash scripts/teacher_gen.sh gen       # step 4: every task × ATTEMPTS                                                    (paid)
#   bash scripts/teacher_gen.sh report    # counts, tokens, cost      ·     inspect [NAME]   thoughts, tools, time per call
# Environment (names are printed, values never): TEACHER = the gateway's name of the teacher model · the address in
# GATEWAY_BASE_URL or ANTHROPIC_BASE_URL · the key in GATEWAY_API_KEY, ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY.
# Knobs (each printed): API messages|chat (the request format) · TEMP 0.7 · THINK effort:high (CEIL_THINK form) · CONC 4 ·
# ATTEMPTS 3 · CHUNK 50 · BUDGET_USD (the ledger's budget; set by setup/check/pilot/gen ONLY when given) · PRICE "in,cached,out"
# $ per 1M tokens (used when the gateway reports no cost) · RESERVE_USD (per attempt) · KEEP_IMAGES 0 · MIN_FREE_GB 50 ·
# SOURCES rebench,v2 (task sources kept by setup; SWE-Gym is left out, `all` keeps every source) · OUT runs/teach · TABLE runs/b10/tasks/all.jsonl · SUBMISSION runs/subs/v21_e4b · EXCLUDE_REPOS (files, one owner/name per line;
# default: the 4 repositories of the evaluation tasks. splits/b7_repos.txt lists the repositories of the B-7 task sets: adding
# it would remove about 31 % of the tasks, so a later evaluation on B-7 tasks takes only tasks of repositories not trained on).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?setup|check|pilot|gen|report|inspect}"
PY=.venv-eval/bin/python
OUT="${OUT:-runs/teach}"; TABLE="${TABLE:-runs/b10/tasks/all.jsonl}"; TASKS_DIR="${TASKS_DIR:-runs/b10/tasks}"
SUBMISSION="${SUBMISSION:-runs/subs/v21_e4b}"; API="${API:-messages}"; TEMP="${TEMP:-0.7}"; THINK="${THINK:-effort:high}"
CONC="${CONC:-4}"; ATTEMPTS="${ATTEMPTS:-3}"; CHUNK="${CHUNK:-50}"; KEEP_IMAGES="${KEEP_IMAGES:-0}"; MIN_FREE_GB="${MIN_FREE_GB:-50}"
export REMINDER="${REMINDER:-0}" HIST_THOUGHTS="${HIST_THOUGHTS:-1}"; RESERVE_USD="${RESERVE_USD:-}"; BUDGET_USD="${BUDGET_USD:-}"; PRICE="${PRICE:-}"; EXCLUDE_REPOS="${EXCLUDE_REPOS:-splits/eval_repos.txt}"
ALIAS=gemma-4-e4b-it-qat-w4a16-ct
BASE_NAME=$([ -n "${GATEWAY_BASE_URL:-}" ] && echo GATEWAY_BASE_URL || { [ -n "${ANTHROPIC_BASE_URL:-}" ] && echo ANTHROPIC_BASE_URL || echo ""; })
KEY_NAME=$(for k in GATEWAY_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY; do [ -n "${!k:-}" ] && { echo $k; break; }; done)
echo "teacher_gen $STEP · out $OUT · table $TABLE · submission $SUBMISSION · api $API · temperature $TEMP · thinking $THINK · reminder $REMINDER (1 = think-before-every-call line in the teacher's request only) · history thoughts $HIST_THOUGHTS (1 = the teacher's earlier thoughts are re-sent, as the student's are) · conc $CONC · attempts $ATTEMPTS · chunk $CHUNK · keep images $KEEP_IMAGES · min free $MIN_FREE_GB GB · reserve ${RESERVE_USD:-unset} · budget ${BUDGET_USD:-unchanged} · price ${PRICE:-unset} · teacher $([ -n "${TEACHER:-}" ] && echo set || echo MISSING) · address from ${BASE_NAME:-MISSING} · key from ${KEY_NAME:-MISSING} · code $(git log --oneline -1 | cut -c1-60)"

need_env() {
  [ -n "${TEACHER:-}" ] && [ -n "$BASE_NAME" ] && [ -n "$KEY_NAME" ] || { echo "STOP: TEACHER, an address and a key are needed in this shell (names above; type them here, never into a file under git)"; exit 1; }
  export LITELLM_API_KEY="${!KEY_NAME}"
  mkdir -p "$OUT"
  BASE="${!BASE_NAME}"; BASE="${BASE%/}"; BASE="${BASE%/v1}"
  if [ "$API" = chat ]; then ROUTE=openai; BASE="$BASE/v1"; export CEIL_REMOTE=1; else ROUTE=anthropic; fi
  ( umask 077; printf 'models:\n  %s:\n    path: %s/%s\n    api_base: %s\n' "$ALIAS" "$ROUTE" "$TEACHER" "$BASE" > "$OUT/models.yaml" )
  echo "  models.yaml written ($OUT/models.yaml, mode 600, gitignored): alias $ALIAS → route $ROUTE/<TEACHER>"
}
if [ "${BUDGET:-}" = none ]; then echo "STOP: BUDGET=none removes every cap (no limit on tool calls or turns, 60 min per task): the harness CLI does not read eval_config.yaml. Teacher runs need the V21 budget as flags; unset BUDGET."; exit 1; fi
ledger() {   # the budget is set only when BUDGET_USD is given; otherwise the ledger is left as it is
  if [ -n "$BUDGET_USD" ]; then $PY scripts/teach_ledger.py init --dir "$OUT/ledger" --budget "$BUDGET_USD" ${PRICE:+--price "$PRICE"} | sed 's/^/  /'
  else $PY scripts/teach_ledger.py status --dir "$OUT/ledger" | sed 's/^/  /'; fi
  # the budget counts dollars only if a call can be priced: a gateway that reports no cost needs PRICE, else every call counts $0
  $PY -c "import json,sys; st=json.load(open('$OUT/ledger/state.json')); sys.exit(0 if st.get('price') and st.get('budget_usd') is not None else 1)" 2>/dev/null \
    || { echo "STOP: the ledger has no price (or no budget). Give PRICE=\"in,cached,out\" in \$ per 1M tokens (and BUDGET_USD): without a price a call the gateway does not price counts as \$0 and the budget cannot stop anything."; exit 1; }
}
gen() {      # $1 name  $2 ids file  $3 attempts  $4 chunk  $5 keep images
  [ -n "$RESERVE_USD" ] || { echo "STOP: RESERVE_USD is not set (what one attempt may cost; check uses 0.50 unless given)"; exit 1; }
  $PY scripts/teacher_gen.py gen --out "$OUT" --ids "$2" --name "$1" --table "$TABLE" --tasks-dir "$TASKS_DIR" --attempts "$3" --chunk "$4" \
      --conc "$CONC" --reserve-usd "$RESERVE_USD" --temp "$TEMP" --think "$THINK" --submission "$SUBMISSION" --model-alias "$ALIAS" \
      --models-yaml "$OUT/models.yaml" --keep-images "$5" --min-free-gb "$MIN_FREE_GB"
}

case "$STEP" in
setup)
  docker run --rm hello-world >/dev/null 2>&1 && echo "  docker: ok · $(docker info 2>/dev/null | grep -c Username) login(s) (0 = anonymous pulls: 100 per 6 h)" || { echo "  docker does not run without sudo → stop"; exit 1; }
  [ -x $PY ] || { echo "  no .venv-eval → stop"; exit 1; }
  $PY -c "import importlib.metadata as m, docker; print('  swegemma', m.version('swegemma'), '· litellm', m.version('litellm'), '· google-adk', m.version('google-adk'), '· docker-py', docker.__version__)" || exit 1
  for f in agent.yaml configs/sampling.yaml eval_config.yaml prompts/system.md; do [ -f "$SUBMISSION/$f" ] || { echo "  V21 file missing: $SUBMISSION/$f — unpack the four files of the private dataset there (BOX_PLAN.md B-11 step 2) → stop"; exit 1; }; done
  echo "  V21 files: prompt $(wc -c < "$SUBMISSION/prompts/system.md") bytes sha256 $(sha256sum "$SUBMISSION/prompts/system.md" | cut -c1-16) (expected 4814 · 794e38a822fd18c7) · tools $(grep -c '^  - ' "$SUBMISSION/agent.yaml") · $(tr '\n' ' ' < "$SUBMISSION/eval_config.yaml" | tr -s ' ') · $(grep -E 'temperature|max_output|thinking_budget' "$SUBMISSION/configs/sampling.yaml" | tr '\n' ' ' | tr -s ' ')"
  [ -f "$TABLE" ] || { echo "  no task table $TABLE: bash scripts/rebench_mini.sh setup → stop"; exit 1; }
  EX=(); for f in $EXCLUDE_REPOS; do [ -f "$f" ] || { echo "  no $f (repositories to keep out of the training data) → stop"; exit 1; }; EX+=(--exclude-repos "$f"); done
  $PY scripts/teacher_gen.py tasks --table "$TABLE" --out "$OUT" --sources "${SOURCES:-rebench,v2}" "${EX[@]}" | sed 's/^/  /'
  $PY scripts/teach_ledger.py selftest | sed 's/^/  /'; $PY scripts/docker_task_shim.py selftest | sed 's/^/  /'
  echo "  📷 every line from 'teacher_gen setup' down" ;;
check)
  need_env
  BUDGET_USD="${BUDGET_USD:-3}"; RESERVE_USD="${RESERVE_USD:-0.50}"; ledger
  $PY - "$TABLE" "$OUT" <<'PYEOF'
import json, random, sys
ok = set(open(sys.argv[2] + "/all.ids").read().split())
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pick = []
for src in ("rebench", "v2"):
    c = sorted(r["instance_id"] for r in rows if r.get("source") == src and r["instance_id"] in ok)
    random.Random(0).shuffle(c); pick += c[:3]
open(sys.argv[2] + "/check6.ids", "w").write("\n".join(pick) + "\n")
print("  check6: 3 tasks per source, seed 0 →", " ".join(pick))
PYEOF
  gen check6 "$OUT/check6.ids" 1 6 1
  cp "$OUT/check6.ids" "$TASKS_DIR/check6_empty.ids"
  $PY scripts/rebench_mini.py grade --tasks "$TASKS_DIR" --name check6_empty --preds empty --out "$OUT/grade/check6_empty" 2>&1 | tail -4 | sed 's/^/  empty patch · /'
  $PY scripts/teacher_gen.py inspect --out "$OUT" --name check6
  echo "  📷 from 'teacher_gen check' down. Gate: gold-failing tasks are dropped and counted (at least 4 of 6 go on), the empty patch resolves on none, the 6 shim lines end with 'PREP ok', no attempt over the tool-call cap, and the inspect line 'calls without thought text … GATE' says ok." ;;
probe)
  # B-11, 2026-11-20: do the teacher's earlier thoughts reach the model? 2 pilot tasks, 1 attempt each (about $0.25);
  # the adapter sends one extra request without the thoughts and prints both prompt token counts.
  need_env; BUDGET_USD="${BUDGET_USD:-1}"; RESERVE_USD="${RESERVE_USD:-0.40}"; ledger
  [ -f "$TASKS_DIR/pilot30.ids" ] || { echo "STOP: no $TASKS_DIR/pilot30.ids"; exit 1; }
  grep -Fxf "$OUT/all.ids" "$TASKS_DIR/pilot30.ids" | head -"${PROBE_N:-2}" > "$OUT/probe.ids"; echo "  probe: $(tr '\n' ' ' < "$OUT/probe.ids")(PROBE_N ${PROBE_N:-2})"
  gen probe "$OUT/probe.ids" 1 "${PROBE_N:-2}" 1
  echo "  -- probe lines (the eval's own log):"; grep -h "ceil hist probe\|ceil history:\|commands start in" "$OUT/gen.log" | tail -8 | sed 's/^/  /'
  $PY scripts/teacher_gen.py inspect --out "$OUT" --name probe | head -14
  echo "  📷 from '-- probe lines' down. Read: the probe line ends with 'REACH the model'; the 'history:' line says all and ok; the reply shapes line; 'commands start in /workspace'." ;;
pilot)
  need_env; ledger
  [ -f "$TASKS_DIR/pilot30.ids" ] || { echo "STOP: no $TASKS_DIR/pilot30.ids (B-10: NAME=pilot30 N=30 bash scripts/rebench_mini.sh subset)"; exit 1; }
  grep -Fxf "$OUT/all.ids" "$TASKS_DIR/pilot30.ids" > "$OUT/pilot30.ids"; echo "  pilot30: $(wc -l < "$OUT/pilot30.ids") of $(wc -l < "$TASKS_DIR/pilot30.ids") tasks are in all.ids"
  gen pilot30 "$OUT/pilot30.ids" 1 30 "${KEEP_IMAGES}"
  $PY scripts/teacher_gen.py report --out "$OUT"
  $PY scripts/teacher_gen.py inspect --out "$OUT" --name pilot30 | head -48
  echo "  📷 the report and inspect blocks" ;;
gen)
  need_env; ledger
  gen all "$OUT/all.ids" "$ATTEMPTS" "$CHUNK" "$KEEP_IMAGES" ;;
report)  $PY scripts/teacher_gen.py report --out "$OUT" ;;
inspect) $PY scripts/teacher_gen.py inspect --out "$OUT" --name "${2:-}" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
