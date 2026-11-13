#!/usr/bin/env bash
# B-10 steps 1–3: the MetaCurate task set in mini-swe-agent with a remote teacher, graded in the tasks' own images.
# No GPU. Needs .venv-swe (bash scripts/verified_box.sh setup) and docker without sudo.
#   bash scripts/rebench_mini.sh setup                 # checks + the task table (1,415 tasks, ~720 MB of upstream parquet)
#   bash scripts/rebench_mini.sh check                 # PER_SOURCE=5 tasks per source: gold patch, empty patch and the
#                                                      # pool's own patches through our grader (pulls ~15 images) → 📷
#   NAME=pilot30 N=30 bash scripts/rebench_mini.sh subset
#   TAG=glm NAME=pilot30 bash scripts/rebench_mini.sh run      # PAID: needs GATEWAY_BASE_URL, GATEWAY_API_KEY, TEACHER
#   TAG=glm NAME=pilot30 bash scripts/rebench_mini.sh grade
#   TAG=glm NAME=pilot30 bash scripts/rebench_mini.sh report
# Generation settings = the MetaCurate pool's, read from its rows (BOX_PLAN B-10): mini-swe-agent's swebench.yaml with the
# pool's instance prompt (scripts/b10_instance_template.txt: three edits to the stock one) · STEPS 250 · TEMP 0.7 ·
# MAX_TOK 4096 · command timeout 60 s and parallel tool calls from swebench.yaml · NET none (the prompt says the
# environment is offline; the pool's docker flags are not published; NET="" = docker's default network).
# Knobs (all printed): POOL (the MetaCurate parquet shards) · OUT (runs/b10) · NAME (task subset) · N / PER_SOURCE / SEED /
# SOURCES (subset) · TAG · ATTEMPT (1; the output is $OUT/$TAG/$NAME.a$ATTEMPT) · WORKERS (8) · STEPS · TEMP · MAX_TOK ·
# EFFORT ("" = reasoning_effort not sent) · NET · PULL_TIMEOUT (1800 s) · GRADE_WORKERS (4) · GRADE_TIMEOUT (1800 s) ·
# GRADE_NET ("" = docker default) · PRICE ("in,cached,out" $ per M tokens, report only) · MIN_FREE_GB (50).
# V2 images keep the repository at /<repo name>, not /testbed: a startup command links /testbed to it, so the prompt
# and mini's working directory stay the same for all three sources. The address of the gateway is written to
# $OUT/$TAG/*.model.yaml and into every trajectory's config (runs/ is gitignored); the key only lives in the environment.
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?setup|check|subset|run|grade|report}"
VS=.venv-swe/bin; PY=$VS/python
POOL="${POOL:-runs/sft/mc/raw}"; OUT="${OUT:-runs/b10}"; NAME="${NAME:-pilot30}"; N="${N:-30}"; PER_SOURCE="${PER_SOURCE:-0}"
SEED="${SEED:-0}"; SOURCES="${SOURCES:-}"; TAG="${TAG:-glm}"; ATTEMPT="${ATTEMPT:-1}"; WORKERS="${WORKERS:-8}"
STEPS="${STEPS:-250}"; TEMP="${TEMP:-0.7}"; MAX_TOK="${MAX_TOK:-4096}"; EFFORT="${EFFORT:-}"; NET="${NET-none}"
PULL_TIMEOUT="${PULL_TIMEOUT:-1800}"; GRADE_WORKERS="${GRADE_WORKERS:-4}"; GRADE_TIMEOUT="${GRADE_TIMEOUT:-1800}"
GRADE_NET="${GRADE_NET:-}"; PRICE="${PRICE:-}"; MIN_FREE_GB="${MIN_FREE_GB:-50}"
TASKS=$OUT/tasks; RUN=$OUT/$TAG/$NAME.a$ATTEMPT
FREE=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc 0-9); FREE="${FREE:-0}"
echo "rebench_mini $STEP · pool $POOL · out $OUT · subset $NAME (n $N, per source $PER_SOURCE, seed $SEED, sources ${SOURCES:-all}) · tag $TAG attempt $ATTEMPT → $RUN · workers $WORKERS · step limit $STEPS · temperature $TEMP · max tokens $MAX_TOK · reasoning_effort ${EFFORT:-not sent} · agent network ${NET:-docker default} · pull timeout $PULL_TIMEOUT s · grade workers $GRADE_WORKERS timeout $GRADE_TIMEOUT s network ${GRADE_NET:-docker default} · price ${PRICE:-not given} · free disk $FREE GB (min $MIN_FREE_GB) · teacher ${TEACHER:-not set} · code $(git log --oneline -1 | cut -c1-40)"
T0=$(date +%s)
[ -x $PY ] || { echo "no .venv-swe: bash scripts/verified_box.sh setup"; exit 1; }
disk() { [ "$FREE" -ge "$MIN_FREE_GB" ] || { echo "only $FREE GB free (< $MIN_FREE_GB): nothing started. Unused images: docker image prune -a (asks first)"; exit 1; }; }
grade() { $PY scripts/rebench_mini.py grade --tasks "$TASKS" --name "$1" --preds "$2" --out "$3" --pool "$POOL" \
            --workers "$GRADE_WORKERS" --timeout "$GRADE_TIMEOUT" --pull-timeout "$PULL_TIMEOUT" --net "$GRADE_NET"; }
case "$STEP" in
setup)
  docker run --rm hello-world >/dev/null 2>&1 && echo "  docker: ok · $(docker info 2>/dev/null | grep -c Username) login(s) (0 = anonymous pulls: 100 per 6 h per address)" || { echo "docker does not run without sudo here"; exit 1; }
  $PY -c "import importlib.metadata as m; print('  mini-swe-agent', m.version('mini-swe-agent'), '· litellm', m.version('litellm'), '· datasets', m.version('datasets'), '· pyarrow', m.version('pyarrow'), '· huggingface_hub', m.version('huggingface_hub'))" || exit 1
  ls "$POOL"/data/*.parquet >/dev/null 2>&1 || { echo "  no pool under $POOL: .venv-eval/bin/python train/mc_prep.py download --dir $POOL"; exit 1; }
  $PY scripts/rebench_mini.py tasks --pool "$POOL" --out "$TASKS" 2>&1 | grep -v -i warning
  echo "  $(( $(date +%s) - T0 )) s · 📷 every line from 'rebench_mini setup' down" ;;
check)
  disk; [ -s "$TASKS/all.jsonl" ] || { echo "no $TASKS/all.jsonl: run setup first"; exit 1; }
  P="${PER_SOURCE:-5}"; [ "$P" = 0 ] && P=5
  $PY scripts/rebench_mini.py subset --tasks "$TASKS" --name "check$P" --per-source "$P" --seed "$SEED" ${SOURCES:+--sources "$SOURCES"}
  for kind in gold empty pool; do grade "check$P" $kind "$OUT/check$P.$kind" 2>&1 | tee "$OUT/check$P.$kind.log" | cut -c1-260; done
  echo "-- git history inside one image per source (a count of commits not reachable from HEAD > 0 = the fix may be readable):"
  $PY - "$TASKS" "$P" <<'PY'
import json, subprocess, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/all.jsonl")]
ids = set(open(f"{sys.argv[1]}/check{sys.argv[2]}.ids").read().split())
seen = set()
for r in rows:
    if r["instance_id"] in ids and r["source"] not in seen and not r["skip"]:
        seen.add(r["source"])
        cmd = (f"cd {r['workdir']} && echo head=$(git rev-parse --short HEAD) base={r['base_commit'][:7]} all=$(git rev-list --all | wc -l) "
               "reachable=$(git rev-list HEAD | wc -l) remotes=$(git remote | wc -l) tags=$(git tag | wc -l) dirty=$(git status --porcelain | wc -l); "
               "ls -d /testbed 2>/dev/null; which python; (conda env list 2>/dev/null | grep -c testbed)")
        p = subprocess.run(["docker", "run", "--rm", "--entrypoint", "bash", r["image_name"], "-c", cmd], capture_output=True, text=True, timeout=900)
        print(f"   {r['source']:8s} {(p.stdout + p.stderr).strip().replace(chr(10), ' · ')[:300]}")
PY
  echo "  read: gold = resolved for every task, empty = 0 resolved, pool = agrees with the pool's labels. $(( $(date +%s) - T0 )) s · 📷 every line from 'rebench_mini check' down" ;;
subset)
  $PY scripts/rebench_mini.py subset --tasks "$TASKS" --name "$NAME" --n "$N" --per-source "$PER_SOURCE" --seed "$SEED" ${SOURCES:+--sources "$SOURCES"} ;;
run)
  disk
  : "${GATEWAY_BASE_URL:?export GATEWAY_BASE_URL first}" "${GATEWAY_API_KEY:?export GATEWAY_API_KEY first}" "${TEACHER:?export TEACHER first}"
  [ -d "$TASKS/$NAME/data" ] || { echo "no subset $NAME: NAME=$NAME N=… bash scripts/rebench_mini.sh subset"; exit 1; }
  mkdir -p "$OUT/$TAG"; CFG="$OUT/$TAG/$NAME.a$ATTEMPT.model.yaml"; ( umask 077; : > "$CFG" )
  { echo "agent:"; echo "  step_limit: $STEPS"; echo "  instance_template: |"; sed 's/^/    /' scripts/b10_instance_template.txt; echo
    echo "environment:"; echo "  pull_timeout: $PULL_TIMEOUT"
    if [ -n "$NET" ]; then echo "  run_args: [\"--rm\", \"--network\", \"$NET\"]"; fi
    echo "run:"
    echo "  env_startup_command: 'if [ \"{{workdir}}\" != /testbed ]; then cd / && rmdir /testbed 2>/dev/null; ln -sfn {{workdir}} /testbed; [ \"\$(readlink /testbed)\" = \"{{workdir}}\" ]; fi'"
    echo "model:"; echo "  model_name: \"openai/$TEACHER\""; echo "  cost_tracking: \"ignore_errors\""; echo "  model_kwargs:"
    echo "    api_base: \"$GATEWAY_BASE_URL\""; echo "    temperature: $TEMP"; echo "    max_tokens: $MAX_TOK"; echo "    drop_params: true"
    if [ -n "$EFFORT" ]; then echo "    reasoning_effort: \"$EFFORT\""; fi; } >> "$CFG"
  echo "  config: swebench.yaml + $CFG ($(wc -l < "$CFG") lines; holds the gateway address, not the key) · log $RUN.run.log"
  OPENAI_API_KEY="$GATEWAY_API_KEY" MSWEA_COST_TRACKING=ignore_errors $VS/mini-extra swebench --subset "$TASKS/$NAME" --split test \
    -m "openai/$TEACHER" -c swebench.yaml -c "$CFG" -w "$WORKERS" -o "$RUN" > "$RUN.run.log" 2>&1
  echo "  mini exit $? · predictions $($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$RUN/preds.json" 2>/dev/null || echo 0) · trajectories $(ls "$RUN"/*/*.traj.json 2>/dev/null | wc -l) · $(( $(date +%s) - T0 )) s"
  ls "$RUN"/exit_statuses_*.yaml >/dev/null 2>&1 && tail -n 25 "$(ls -t "$RUN"/exit_statuses_*.yaml | head -1)" | cut -c1-160
  echo "  next: TAG=$TAG NAME=$NAME ATTEMPT=$ATTEMPT bash scripts/rebench_mini.sh grade" ;;
grade)
  disk; [ -s "$RUN/preds.json" ] || { echo "no $RUN/preds.json"; exit 1; }
  grade "$NAME" "$RUN/preds.json" "$RUN.grade" 2>&1 | tee "$RUN.grade.log" | cut -c1-260
  echo "  next: TAG=$TAG NAME=$NAME ATTEMPT=$ATTEMPT bash scripts/rebench_mini.sh report" ;;
report)
  $PY scripts/rebench_mini.py report --run "$RUN" --grade "$RUN.grade" ${PRICE:+--price "$PRICE"}
  echo "  📷 every line from 'rebench_mini report' down" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
