#!/usr/bin/env bash
# B-9 part 2 (BOX_PLAN): a LoRA on MetaCurate's public DSv4Pro trajectories, steps 0–2 on the box.
#   MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh serve     # one card (31B first)
#   MODEL=… GPU=0 PORT=8011 bash train/mc_box.sh probe     # step 0: 2 Verified tasks through the request logger → 📷
#   (THINK_HIST=1, the default since option ③: the logger also renames reasoning_content → reasoning, so the server
#    renders earlier thoughts; outputs in runs/verified/probe_th; THINK_HIST=0 = the original B-9 setup, runs/verified/probe)
#   MODEL=… bash train/mc_box.sh data                      # step 1: download (HF token) + filter → runs/sft/mc/r1k   → 📷
#   MODEL=… bash train/mc_box.sh render                    # step 2: render.py --pack in RENDER_PROCS shards (gate R) → 📷
#   MODEL=… bash train/mc_box.sh env|dequant               # step 3a: .venv-train (setup_train_env.sh) · bf16 copy of MODEL
#   MODEL=… GPUS=0 bash train/mc_box.sh smoke              # step 3b: 1 optimizer step on the 16 longest sequences → peak GB
#   MODEL=… GPUS=0,1,2,3,4,5,6,7 bash train/mc_box.sh train # step 3c: one LoRA, one sft_lora.py rank per CARDS_PER_RANK cards
# Training knobs (printed): GPUS (all cards) · CARDS_PER_RANK (1; 2 = one model copy split over two cards, device map
# auto — for 31B if smoke runs out of memory) · LR (1e-5, user 2026-11-13) · EPOCHS (2) · ACCUM (16) · R / ALPHA (32 / 32)
# · WARMUP (0.05) · MAX_LEN (= MAX_CTX) · DQ (runs/dequant/<model>-dq) · OUTD (adapters/mc_<model>_<arm>_lr<LR>) ·
# DIST_PORT (29600).
# Knobs (all printed): MODEL (31B) · GPU (0) · PORT (8011; the logger listens on PORT+100) · PROBE_SLICE (0:2) ·
# HISTORY (from step 0's probe_summary.json; override keep|drop) · N (1000) · SEED (0) · MAX_CTX (30720) · ARM (r1k) ·
# RENDER_PROCS (16) · T2T (on) · PACK (from step 0: on only if the server renders earlier thoughts). Outputs quote task text → runs/ only (gitignored).
# Another model family (B-9 part 3, the control on the paper's 9B model): MODEL=Qwen/Qwen3.5-9B → FAMILY qwen (printed):
#   bash train/mc_box.sh check      # render_tmpl.py on the logged probe requests (PROBE_DIR) vs the server's prompts (A2)
#   bash train/mc_box.sh render     # the SAME records ($MC/$ARM, $MC/${ARM}_holdout: not rebuilt) through the model's own
#                                   # chat template (train/render_tmpl.py) → $MC/$ARM.<served>.rendered.jsonl
#   GPUS=0 bash train/mc_box.sh smoke · GPUS=… bash train/mc_box.sh train      # base = the HF repo (bf16 already: no
#                                   # dequant), LoRA on TARGET_LEAVES, adapters/mc_<model>_<arm>_lr<LR>
# probe / data are Gemma's steps (the records are tokenizer-free; the 30,720 fit was decided with Gemma's tokenizer).
# Any family (B-9 part 3 step 9): ARM=r1k_nt LIKE=r1k T2T=off bash train/mc_box.sh like = the SAME trajectories and held-out
#   tasks as $MC/$LIKE, rebuilt from the raw rows with this T2T ($MC/$ARM, $MC/${ARM}_holdout); then render / train with that ARM.
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?probe|data|like|check|render|env|dequant|smoke|train}"
MODEL="${MODEL:-google/gemma-4-E4B-it-qat-w4a16-ct}"; SERVED="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
GPU="${GPU:-0}"; PORT="${PORT:-8011}"; PROBE_SLICE="${PROBE_SLICE:-0:2}"
N="${N:-1000}"; SEED="${SEED:-0}"; MAX_CTX="${MAX_CTX:-30720}"; ARM="${ARM:-r1k}"; RENDER_PROCS="${RENDER_PROCS:-16}"
PY=.venv-eval/bin/python; MC=runs/sft/mc
THINK_HIST="${THINK_HIST:-1}"                     # option ③ (user, 2026-11-13): the server sees earlier thoughts
# the probe's sampling = the new baselines' (user, 2026-11-13): 8,192 tokens per reply, top_p 0.95, thinking budget 4,096
MAX_TOK="${MAX_TOK:-8192}"; TOP_P="${TOP_P:-0.95}"; THINK_BUDGET="${THINK_BUDGET:-4096}"
PR=runs/verified/probe$([ "$THINK_HIST" = 1 ] && echo _th); WIRE=$([ "$THINK_HIST" = 1 ] && echo mini-think || echo mini)
HISTORY="${HISTORY:-$($PY -c "import json; print(json.load(open('$PR/probe_summary.json'))['history_rule'])" 2>/dev/null || echo unset)}"
# T2T: move the text of text + tool-call replies into the thought. On by the user's decision after the 31B probe
# (2026-11-13: 782 such turns in 31B's own histories — the Gemma template bug HF #67/#115 hits the baseline).
T2T="${T2T:-on}"
# LABELS (B-9 part 3 step 12, non-Gemma render only): reply = loss on the whole reply; action = no loss on the thought
# and its close token (render_tmpl.py --label-scope). Use it with its own ARM so the rendered files have their own name.
LABELS="${LABELS:-reply}"
# PACK: one sequence per trajectory only if the server renders earlier thoughts; vLLM 0.19.1 drops mini's
# reasoning_content (probe 2026-11-13: 0 blocks), so then every call is its own sample (render.py without --pack).
PACK="${PACK:-$($PY -c "import json; print('on' if json.load(open('$PR/probe_summary.json')).get('server renders earlier thoughts') else 'off')" 2>/dev/null || echo off)}"
tokdir() { [ -n "${TOKDIR:-}" ] && { echo "$TOKDIR"; return; }; $PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json', '*.jinja']))" 2>/dev/null | tail -1; }
case "$(basename "$MODEL")" in [Qq]wen3.5*) FAMILY="${FAMILY:-qwen}" ;; *) FAMILY="${FAMILY:-gemma}" ;; esac
# RSUF: the rendered files of a non-Gemma model sit next to Gemma's under their own name
RSUF=$([ "$FAMILY" = gemma ] || echo ".$SERVED"); RD="$MC/$ARM$RSUF"
# TARGET_LEAVES (qwen): the language model's attention + MLP projections. Qwen3.5's 24 linear-attention layers have
# in_proj_qkv / in_proj_z / out_proj instead of q/k/v/o_proj; their two 32-wide gate projections (in_proj_b / in_proj_a,
# which vLLM serves as one packed module) are left out.
TARGET_LEAVES="${TARGET_LEAVES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj}"
PROBE_DIR="${PROBE_DIR:-runs/verified/probe_q9}"
case "$FAMILY:$STEP" in qwen:probe|qwen:data) echo "$STEP is Gemma's step (it rebuilds $MC/$ARM): for $MODEL use check | render | smoke | train"; exit 1 ;; esac
echo "mc_box $STEP · family $FAMILY (rendered files $RD.*)$([ "$FAMILY" = gemma ] || echo " · target leaves $TARGET_LEAVES · probe dir $PROBE_DIR") · model $MODEL (served $SERVED) · gpu $GPU port $PORT (logger $(( PORT + 100 ))) · probe slice $PROBE_SLICE · think hist $THINK_HIST (wire $WIRE, probe dir $PR) · max tok $MAX_TOK · top_p $TOP_P · thinking budget $THINK_BUDGET · history $HISTORY · text to thought $T2T · pack $PACK · n $N seed $SEED max ctx $MAX_CTX · arm $ARM · render procs $RENDER_PROCS · tokenizer dir ${TOKDIR:-from the HF cache of MODEL} · code $(git log --oneline -1 | cut -c1-40)"
T0=$(date +%s)
ALLGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
GPUS="${GPUS:-${ALLGPUS:-0}}"; IFS=, read -r -a GL <<< "$GPUS"; CPR="${CARDS_PER_RANK:-1}"; NR=$(( ${#GL[@]} / CPR ))
LR="${LR:-1e-5}"; EPOCHS="${EPOCHS:-2}"; ACCUM="${ACCUM:-16}"; R="${R:-32}"; ALPHA="${ALPHA:-32}"; WARMUP="${WARMUP:-0.05}"
MAX_LEN="${MAX_LEN:-$MAX_CTX}"; MSHORT=$(basename "$MODEL" | sed 's/-it-qat-w4a16-ct//'); DQ="${DQ:-runs/dequant/$MSHORT-dq}"
[ "$FAMILY" = gemma ] || DQ="${DQ_OVERRIDE:-$MODEL}"      # a bf16 HF repo is its own training base
OUTD="${OUTD:-adapters/mc_${MSHORT}_${ARM}_lr${LR}}"
EVALF="$RD.holdout.rendered.jsonl"; HOLD=$([ -s "$EVALF" ] && echo "--eval $EVALF --holdout 0" || echo "--holdout 8"); DIST_PORT="${DIST_PORT:-29600}"; TPY="${TPY:-.venv-train/bin/python}"
SFT_EXTRA="${SFT_EXTRA:-}"                        # laptop test of the launcher: TPY=.venv-eval/bin/python SFT_EXTRA=--mock
SFT="--base $DQ --ct $MODEL --alias $SERVED --max-len $MAX_LEN --epochs $EPOCHS --lr $LR --warmup $WARMUP --accum $ACCUM --r $R --alpha $ALPHA --clip 1.0 --eval-every 25 --seed 0 --device-map $([ "$CPR" -gt 1 ] && echo auto || echo cuda) $([ "$FAMILY" = gemma ] || echo "--target-leaves $TARGET_LEAVES --base-name $MODEL") $SFT_EXTRA"
case "$STEP" in env|dequant|smoke|train)
  echo "  train · cards $GPUS · $CPR per rank → $NR rank(s) · dequant $DQ · out $OUTD · held out: $HOLD · sft: $SFT" ;; esac
rank_cards() { local i=$1; local c=(); for j in $(seq 0 $(( CPR - 1 ))); do c+=("${GL[$(( i * CPR + j ))]}"); done; (IFS=,; echo "${c[*]}"); }

case "$STEP" in
probe)
  curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" || { echo "no $SERVED on port $PORT: serve it first"; exit 1; }
  PT=vf_probe$([ "$THINK_HIST" = 1 ] && echo _th)
  rm -rf "$PR" runs/verified/$PT runs/verified/$PT.*; mkdir -p "$PR"
  python3 scripts/req_log.py proxy --listen $(( PORT + 100 )) --upstream "http://127.0.0.1:$PORT" --out "$PR/requests" \
    $([ "$THINK_HIST" = 1 ] && echo --rename-reasoning) > "$PR/proxy.log" 2>&1 &
  PX=$!; sleep 2
  MODEL=$MODEL GPU=$GPU PORT=$PORT TAG=$PT SLICE=$PROBE_SLICE WORKERS=2 API_PORT_OFFSET=100 THINK_HIST=$THINK_HIST MAX_TOK=$MAX_TOK TOP_P=$TOP_P THINK_BUDGET=$THINK_BUDGET bash scripts/verified_box.sh run | grep -E "^verified_box|^== run"
  kill $PX 2>/dev/null
  echo "  $(ls "$PR"/requests/req_*.json 2>/dev/null | wc -l) chat requests logged"
  python3 scripts/req_log.py summarize --dir "$PR/requests" --upstream "http://127.0.0.1:$PORT" --model "$SERVED" --out "$PR" || exit 1
  H=$($PY -c "import json; print(json.load(open('$PR/probe_summary.json'))['history_rule'])")
  $PY train/mc_prep.py verify --traj runs/verified/$PT --probe "$PR/records" --history "$H" | tail -6
  echo "  render.py on the logged requests vs the server's own rendering (A2) and the reply spans (A3/A4):"
  $PY train/render.py "$PR/records" --tokenizer "$(tokdir)" --check "$PR/server_prompts" --wire "$WIRE" $([ "$THINK_HIST" = 1 ] && echo --pack --split-runs) --sample 20 \
    | grep -E "^render|^calls|^FAIL|^RESULT|^MALFORMED" | cut -c1-400
  echo "📷 the == probe line, the == verify line, the calls … line and RESULT ($(( $(date +%s) - T0 )) s)" ;;
data)
  [ "$HISTORY" = keep ] || [ "$HISTORY" = drop ] || { echo "run probe first (or set HISTORY=keep|drop)"; exit 1; }
  [ "$T2T" = on ] || [ "$T2T" = off ] || { echo "run probe first (or set T2T=on|off)"; exit 1; }
  $PY train/mc_prep.py download --dir "$MC/raw" || exit 1
  rm -rf "$MC/$ARM" "$MC/${ARM}_holdout"       # a rerun must not keep records of tasks that are now held out
  $PY train/mc_prep.py build --raw "$MC/raw" --tokenizer "$(tokdir)" --history "$HISTORY" --wire "$WIRE" --out "$MC/$ARM" \
    --n "$N" --seed "$SEED" --holdout-tasks "${HOLDOUT_TASKS:-8}" --max-ctx "$MAX_CTX" --think-budget "$THINK_BUDGET" --reply-cap "$MAX_TOK" $([ "$T2T" = on ] && echo --text-to-thought) || exit 1
  du -sh "$MC/$ARM" | sed 's/^/  records on disk: /'
  echo "📷 the rows → … line, the printed-only line and the == line ($(( $(date +%s) - T0 )) s)" ;;
like)       # the same trajectories as $MC/$LIKE with this T2T setting (no tokenizer, nothing selected again)
  LIKE="${LIKE:?LIKE=<existing arm, e.g. r1k>}"
  [ "$T2T" = on ] || [ "$T2T" = off ] || { echo "set T2T=on|off"; exit 1; }
  [ "$ARM" != "$LIKE" ] || { echo "ARM must differ from LIKE ($LIKE): e.g. ARM=r1k_nt"; exit 1; }
  [ -d "$MC/$LIKE" ] || { echo "no $MC/$LIKE"; exit 1; }
  echo "  like $MC/$LIKE → $MC/$ARM · T2T $T2T"
  $PY train/mc_prep.py like --raw "$MC/raw" --like "$MC/$LIKE" --out "$MC/$ARM" $([ "$T2T" = on ] && echo --text-to-thought) || exit 1
  echo "📷 the two → lines and the == line ($(( $(date +%s) - T0 )) s)" ;;
check)      # non-Gemma: our rendering of the logged requests = the prompts the server built?
  [ -d "$PROBE_DIR/records" ] || { echo "no $PROBE_DIR/records: run the probe of BOX_PLAN (req_log.py proxy + summarize) first"; exit 1; }
  $PY train/render_tmpl.py "$PROBE_DIR/records" --tokenizer "$(tokdir)" --check "$PROBE_DIR/server_prompts" --max-len "$MAX_CTX" \
    | grep -E "^render_tmpl|^calls|^FAIL|^MALFORMED|^RESULT" | cut -c1-600
  echo "📷 the calls … line and RESULT (FAIL lines too) ($(( $(date +%s) - T0 )) s)" ;;
render)
  [ -d "$MC/$ARM" ] || { echo "no $MC/$ARM: run data first"; exit 1; }
  if [ "$FAMILY" != gemma ]; then             # the model's own chat template; the same records as Gemma's run
    [ "$LABELS" = reply ] || [ "$LABELS" = action ] || { echo "LABELS=$LABELS: reply|action"; exit 1; }
    echo "  render labels: $LABELS$([ "$LABELS" = action ] && echo " (no loss on thoughts; train and held-out rows alike, so the held-out NLL is over action tokens only)")"
    TOK="$(tokdir)"; pids=()
    for i in $(seq 0 $(( RENDER_PROCS - 1 ))); do
      $PY train/render_tmpl.py "$MC/$ARM" --tokenizer "$TOK" --max-len "$MAX_LEN" --shard "$i/$RENDER_PROCS" --label-scope "$LABELS" --out "$RD.part$i.jsonl" \
        --stats "$RD.render$i.json" > "$RD.render$i.log" 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    cat "$RD".part*.jsonl > "$RD.rendered.jsonl"; rm -f "$RD".part*.jsonl
    if [ -d "$MC/${ARM}_holdout" ]; then
      $PY train/render_tmpl.py "$MC/${ARM}_holdout" --tokenizer "$TOK" --max-len "$MAX_LEN" --label-scope "$LABELS" --out "$RD.holdout.rendered.jsonl" > "$RD.holdout.render.log" 2>&1
      echo "  holdout: $(wc -l < "$RD.holdout.rendered.jsonl") sequences from $(ls "$MC/${ARM}_holdout" | grep -c json) records · $(grep -E '^calls' "$RD.holdout.render.log" | cut -c1-330) · $(grep -E '^RESULT' "$RD.holdout.render.log")"
    else
      echo "  no $MC/${ARM}_holdout: training falls back to --holdout 8"
    fi
    $PY - "$RD" "$RENDER_PROCS" "$MAX_LEN" <<'EOF'
import collections, json, os, sys
rd, n, ml = sys.argv[1], int(sys.argv[2]), sys.argv[3]
T, missing = collections.Counter(), 0
for i in range(n):
    f = f"{rd}.render{i}.json"
    if not os.path.exists(f):
        missing += 1; print(f"  shard {i}: no stats → tail {rd}.render{i}.log"); continue
    for k, v in json.load(open(f)).items():
        if k not in ("len p50", "len max"):
            T[k] += v
lens = sorted(len(json.loads(l)["input_ids"]) for l in open(f"{rd}.rendered.jsonl"))
br = " · ".join(f"{k[7:]} {v}" for k, v in sorted(T.items()) if k.startswith("break: ")) or "none"
bad = T["fail"] + missing
print(f"== render {os.path.basename(rd)}: {T['records']} records · {T['calls']} calls · label tokens {T['label tokens']} · {len(lens)} sequences, "
      f"{sum(lens):,} tokens (len p50 {lens[len(lens) // 2] if lens else 0}, max {lens[-1] if lens else 0}) · breaks: {br} · replies without a thought "
      f"{T['replies without a thought']} · over {ml}: {T['sequences over ' + ml]} sequences ({T['label tokens in sequences over ' + ml]} label tokens; "
      f"sft_lora drops them) · thought label tokens masked {T['thought label tokens masked']} (replies with no thought close "
      f"{T['replies with no thought close (all labels kept)']}, with several {T['replies with several thought closes (cut at the first)']}) "
      f"· malformed calls {T['calls skipped (malformed)']} · FAIL {bad} → {'PASS' if not bad else 'FAIL: grep FAIL ' + rd + '.render*.log'}")
EOF
    echo "📷 the holdout line and the == render line ($(( $(date +%s) - T0 )) s)"; exit 0
  fi
  TOK="$(tokdir)"; pids=()
  for i in $(seq 0 $(( RENDER_PROCS - 1 ))); do
    $PY train/render.py "$MC/$ARM" --tokenizer "$TOK" --wire "$WIRE" $([ "$PACK" = on ] && echo --pack --split-runs) --sample 4 --shard "$i/$RENDER_PROCS" \
      --out "$MC/$ARM.part$i.jsonl" > "$MC/$ARM.render$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$MC/$ARM".part*.jsonl > "$MC/$ARM.rendered.jsonl"; rm -f "$MC/$ARM".part*.jsonl
  if [ -d "$MC/${ARM}_holdout" ]; then      # the held-out tasks (mc_prep --holdout-tasks): same rendering, one process
    $PY train/render.py "$MC/${ARM}_holdout" --tokenizer "$TOK" --wire "$WIRE" $([ "$PACK" = on ] && echo --pack --split-runs) \
      --sample 8 --out "$MC/$ARM.holdout.rendered.jsonl" > "$MC/$ARM.holdout.render.log" 2>&1
    echo "  holdout: $(wc -l < "$MC/$ARM.holdout.rendered.jsonl") sequences from $(ls "$MC/${ARM}_holdout" | grep -c json) records · $(grep -E '^RESULT' "$MC/$ARM.holdout.render.log")"
  else
    echo "  no $MC/${ARM}_holdout (data from before the task holdout): training falls back to --holdout 8"
  fi
  $PY - "$MC" "$ARM" "$RENDER_PROCS" <<'EOF'
import glob, json, re, sys
mc, arm, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tot = {"calls": 0, "targets": 0, "label": 0, "packed": 0, "seq": 0, "mixed": 0, "nothought": 0, "fail": 0, "malformed": 0}
for i in range(n):
    t = open(f"{mc}/{arm}.render{i}.log").read()
    m = re.search(r"calls (\d+) · targets (\d+) · label tokens (\d+)", t)
    if not m:
        print(f"  shard {i}: no summary line → tail {mc}/{arm}.render{i}.log"); tot["fail"] += 1; continue
    for k, v in zip(["calls", "targets", "label"], map(int, m.groups())):
        tot[k] += v
    p = re.search(r"packed label tokens (\d+) \(sequences (\d+), segments unpacked because of a mixed reply (\d+) / a no-thought reply after a tool response (\d+)\)", t)
    if p:
        for k, v in zip(["packed", "seq", "mixed", "nothought"], map(int, p.groups())):
            tot[k] += v
    tot["fail"] += len(re.findall(r"^FAIL", t, re.M)); tot["malformed"] += len(re.findall(r"^MALFORMED", t, re.M))
lens = [len(json.loads(l)["input_ids"]) for l in open(f"{mc}/{arm}.rendered.jsonl")]
tot["seq"] = tot["seq"] or len(lens)
lens.sort()
print(f"== render {arm}: {tot['calls']} calls · {tot['targets']} targets · label tokens {tot['label']} (packed {tot['packed']}) "
      f"· {tot['seq']} sequences, {sum(lens):,} tokens (len p50 {lens[len(lens)//2] if lens else 0}, max {lens[-1] if lens else 0}) "
      f"· run breaks: text+call {tot['mixed']}, no-thought {tot['nothought']} · malformed replies {tot['malformed']} "
      f"· FAIL lines {tot['fail']} → {'PASS' if tot['fail'] == 0 else 'FAIL: grep FAIL ' + mc + '/' + arm + '.render*.log'}")
EOF
  echo "📷 the == render line ($(( $(date +%s) - T0 )) s)" ;;
env)
  [ -x "$TPY" ] && { echo "  $TPY exists: $($TPY -c 'import torch, transformers, peft; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)')"; exit 0; }
  VENV="${VENV:-$HOME/venv-train}" UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.cache/uv-python}" \
    bash train/setup_train_env.sh 2>&1 | tail -4
  echo "📷 the version line ($(( $(date +%s) - T0 )) s)" ;;
dequant)
  [ "$FAMILY" = gemma ] || { echo "  $MODEL is bf16 already: no dequant, the base is the HF repo"; exit 0; }
  if [ -f "$DQ/config.json" ] && ls "$DQ"/*.safetensors >/dev/null 2>&1; then echo "  $DQ exists ($(du -sh "$DQ" | cut -f1)), skipping"; exit 0; fi
  df -h runs | tail -1 | awk '{print "  free disk for runs/: " $4}'
  CUDA_VISIBLE_DEVICES=${GL[0]} $TPY train/dequant_ct.py --model "$MODEL" --out "$DQ" 2>&1 | grep -v Warning | tail -6 | cut -c1-200
  echo "  $(du -sh "$DQ" | cut -f1) in $DQ · 📷 these lines ($(( $(date +%s) - T0 )) s)" ;;
smoke)      # one rank, the 16 longest sequences, 1 optimizer step: does it fit, how fast
  [ -s "$RD.rendered.jsonl" ] || { echo "no $RD.rendered.jsonl: run render first"; exit 1; }
  $PY - "$RD.rendered.jsonl" "$RD.smoke.jsonl" "$MAX_LEN" <<'EOF'
import json, sys
rows = sorted((r for r in (json.loads(l) for l in open(sys.argv[1])) if len(r["input_ids"]) <= int(sys.argv[3])),
              key=lambda r: -len(r["input_ids"]))[:16]                      # the longest that training keeps (≤ MAX_LEN)
with open(sys.argv[2], "w") as w:
    for r in rows: w.write(json.dumps(r) + "\n")
print(f"  smoke data: the 16 longest sequences, {len(rows[0]['input_ids'])}–{len(rows[-1]['input_ids'])} tokens")
EOF
  SM=adapters/smoke_mc$(echo "$RSUF" | tr . _); rm -rf "$SM"
  CUDA_VISIBLE_DEVICES=$(rank_cards 0) $TPY -u train/sft_lora.py $SFT --data "$RD.smoke.jsonl" --holdout 0 --max-steps 1 --eval-every 0 --seq-log \
    --out "$SM" 2>&1 | grep --line-buffered -v Warning | tee runs/sft_mc_smoke.log | grep --line-buffered -E "model|base |seq |step|epoch|done|Error|error|CUDA|falling back" | cut -c1-250
  echo "  smoke adapter (1 step): $SM/epoch1"
  echo "📷 the model / step / done lines: peak GB (one rank on $(rank_cards 0)) and tok/s → full run ≈ total tokens × $EPOCHS / (tok/s × $NR ranks) ($(( $(date +%s) - T0 )) s)" ;;
train)
  [ -s "$RD.rendered.jsonl" ] || { echo "no $RD.rendered.jsonl: run render first"; exit 1; }
  [ "$FAMILY" != gemma ] || [ -f "$DQ/config.json" ] || { echo "no bf16 base in $DQ: run dequant first"; exit 1; }
  rm -rf "$OUTD"; PIDS=(); LOGB=runs/sft_$(basename "$OUTD")
  for i in $(seq 1 $(( NR - 1 ))); do       # ranks 1…: in the background; rank 0 in front, it logs and exports
    CUDA_VISIBLE_DEVICES=$(rank_cards $i) $TPY train/sft_lora.py $SFT --data "$RD.rendered.jsonl" $HOLD --out "$OUTD" \
      --world $NR --rank $i --dist-port $DIST_PORT > $LOGB.rank$i.log 2>&1 &
    PIDS+=($!)
  done
  if [ $NR -gt 1 ]; then                     # a rank that dies would leave the others waiting: stop them all
    ( while sleep 20; do for q in "${PIDS[@]}"; do
        if ! kill -0 $q 2>/dev/null; then sleep 30
          pgrep -f "sft_lora.py.*--out $OUTD --world $NR --rank 0 " >/dev/null && { echo "  a rank ended early (see $LOGB.rank*.log): stopping the others"; pkill -f "sft_lora.py.*--out $OUTD --world"; }
          exit; fi; done; done ) & WATCH=$!
  fi
  CUDA_VISIBLE_DEVICES=$(rank_cards 0) $TPY train/sft_lora.py $SFT --data "$RD.rendered.jsonl" $HOLD --out "$OUTD" \
    --world $NR --rank 0 --dist-port $DIST_PORT 2>&1 | grep -v Warning | tee $LOGB.log | cut -c1-250
  if [ $NR -gt 1 ]; then
    { kill $WATCH; wait $WATCH; } 2>/dev/null; sleep 10; pkill -f "sft_lora.py.*--out $OUTD --world" && echo "  stopped ranks still running after rank 0 ended"
    wait "${PIDS[@]}" 2>/dev/null; for f in $LOGB.rank*.log; do echo "  $f: $(tail -n 1 $f | cut -c1-160)"; done
  fi
  echo "📷 the schedule line, the eval lines, the == epoch lines and the done line ($(( ($(date +%s) - T0) / 60 )) min)" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
