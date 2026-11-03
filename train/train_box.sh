#!/usr/bin/env bash
# SFT M1 / M2 on the GPU box (one RTX PRO 6000 of g4-4), mirroring the RunPod run of 2026-10-29 (05_run_log):
# data from the private Kaggle dataset → train venv → dequantised bf16 base → 3-step smoke → M1 → M2 → submission dirs
# for eval_dev.sh. One stage per call so a shared box can be queued; `all` runs them in order. No `set -e`: every stage
# prints its own result and the script goes on so the log is complete.
#   GPU=2 bash train/train_box.sh all | data | env | dequant | smoke | m1 | m2 | subs
# The kit line (BOX_PLAN.md B-K, 2026-11-03), named stages only (not part of `all`): the dmitrii kit's 31B trajectories with
# the kit's own prompt (train/convert_kit.py --prompt kit) → E4B LoRA → an adapter for eval_dev.sh LATEST=kit:
#   bash train/train_box.sh kitdata                      # no GPU: public kit file (207 MB) → records → render → parser
#   GPU=3 bash train/train_box.sh env|dequant|kit        # train venv, bf16 base, the LoRA (one card)
#   bash train/train_box.sh kitsubs                      # the epoch with the lower held-out NLL → runs/adapters/e4b_kit
# Knobs (every one prints its effective value): GPU (CUDA_VISIBLE_DEVICES), DATA (dataset dir), VENV, DQ (dequant dir),
# KAGGLE_DS (dataset ref), EPOCHS LR ACCUM R ALPHA (sft_lora.py defaults otherwise); kit line: KIT_DS (public dataset),
# KIT_TOOLS (any run's traj/ dir: our 9 tool schemas), KIT_TOK (tokenizer for render/parser), KIT_OUT (adapter dir),
# KIT_NAME (name under runs/adapters/), KIT_EPOCH (default: the lower held-out NLL), PARSER_WHEEL (laptop only).
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:-all}"; T0=$(date +%s)
say() { echo "== $(date '+%T') [$(( $(date +%s) - T0 ))s] $*"; }
run() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
DATA="${DATA:-runs/sft_data}"; KAGGLE_DS="${KAGGLE_DS:-boaztu0/gemma4-sft-private}"
export VENV="${VENV:-$HOME/venv-train}" UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.cache/uv-python}"
DQ="${DQ:-runs/dequant/gemma-4-E4B-dq}"
M1="$DATA/d3_train.rendered.jsonl"          # 496 seq · 3.73 M tok · 74,172 labels · max 23,659 (D3 Opus, thought none)
M2="$DATA/d3_relabel_31b.rendered.jsonl"    # 308 seq · 2.72 M tok · 146,056 labels · max 27,434 (D3 + synthetic_31b thoughts)
SFT="--base $DQ --max-len 32768 --epochs ${EPOCHS:-2} --lr ${LR:-3e-5} --warmup 0.05 --accum ${ACCUM:-16} --r ${R:-32} --alpha ${ALPHA:-32} --clip 1.0 --holdout 8 --eval-every 25 --seed 0"
ALIAS=gemma-4-e4b-it-qat-w4a16-ct
echo "train_box · GPU $CUDA_VISIBLE_DEVICES · DATA $DATA (from $KAGGLE_DS) · VENV $VENV · DQ $DQ · sft: $SFT"

if run data; then
  say "data: $KAGGLE_DS → $DATA"
  mkdir -p "$DATA"; kaggle datasets download -d "$KAGGLE_DS" -p "$DATA" --unzip && ls -la "$DATA"
  for f in "$M1" "$M2"; do [ -f "$f" ] && echo "  $f: $(wc -l < "$f") sequences" || echo "  MISSING $f"; done
fi
if run env; then
  say "train venv → $VENV (uv cache $UV_CACHE_DIR)"; bash train/setup_train_env.sh 2>&1 | tail -3
fi
if run dequant; then
  if [ -f "$DQ/config.json" ] && [ "$(ls "$DQ"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then say "dequant: $DQ exists, skipping"
  else say "dequant → $DQ (17 GB, ~1 min)"; .venv-train/bin/python train/dequant_ct.py --out "$DQ" 2>&1 | grep -v Warning | tail -6 | cut -c1-200; fi
fi
if run smoke; then
  say "smoke: 3 optimizer steps on M2 (the longest sequences), export check — pod: 1.5-1.7K tok/s, peak 34.8 GB"
  rm -rf adapters/smoke
  .venv-train/bin/python train/sft_lora.py $SFT --data "$M2" --max-steps 3 --eval-every 0 --out adapters/smoke 2>&1 | grep -v Warning | tee runs/sft_smoke.log | cut -c1-250
fi
if run m1; then
  say "M1: D3 without thoughts (62 steps)"; rm -rf adapters/e4b_m1
  .venv-train/bin/python train/sft_lora.py $SFT --data "$M1" --out adapters/e4b_m1 2>&1 | grep -v Warning | tee runs/sft_m1.log | cut -c1-250
fi
if run m2; then
  say "M2: D3 + 31B thoughts (38 steps)"; rm -rf adapters/e4b_m2
  .venv-train/bin/python train/sft_lora.py $SFT --data "$M2" --out adapters/e4b_m2 2>&1 | grep -v Warning | tee runs/sft_m2.log | cut -c1-250
fi
if run subs; then
  say "submission dirs: runs/subs_sft (server: both adapters) · runs/subs_m1 / subs_m2 (eval: adapter line on the root agent)"
  [ -d runs/base_lb024 ] || { echo "runs/base_lb024 missing (scripts/from_notebook.py, see eval_dev.sh header)"; exit 1; }
  rm -rf runs/subs_sft runs/subs_m1 runs/subs_m2
  cp -r runs/base_lb024 runs/subs_sft; mkdir -p runs/subs_sft/adapters
  for m in m1 m2; do
    cp -r adapters/e4b_$m/epoch2 runs/subs_sft/adapters/e4b_$m
    cp -r runs/base_lb024 runs/subs_$m; mkdir -p runs/subs_$m/adapters
    cp -r adapters/e4b_$m/epoch2 runs/subs_$m/adapters/e4b_$m   # the harness discovers adapters in the SUBMISSION dir: the
    sed -i "s/^model: .*/model: $ALIAS\nadapter: e4b_$m/" runs/subs_$m/agent.yaml   # line alone → AdapterNotFoundError every task (pod 2026-10-29)
    echo "subs_$m:"; head -3 runs/subs_$m/agent.yaml; ls -la runs/subs_$m/adapters/e4b_$m
  done
  cat <<EOF
next (eval, one card, like the pod run):
  MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TP=1 GPU_UTIL=0.85 SUBMISSION=runs/subs_sft CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup bash scripts/serve_vllm.sh > runs/vllm_sft.log 2>&1 &
  MODEL_ALIAS=$ALIAS TIME_MIN=8 SUBMISSION=runs/base_lb024 ARMS=base TAG=s1   VLLM_LOG=runs/vllm_sft.log REFS="" CONC=1 bash scripts/eval_dev.sh > runs/eval_base_s1.log 2>&1 &
  MODEL_ALIAS=$ALIAS TIME_MIN=8 SUBMISSION=runs/subs_m1   ARMS=base TAG=m1s1 VLLM_LOG=runs/vllm_sft.log REFS="" CONC=1 bash scripts/eval_dev.sh > runs/eval_m1_s1.log 2>&1 &
  MODEL_ALIAS=$ALIAS TIME_MIN=8 SUBMISSION=runs/subs_m2   ARMS=base TAG=m2s1 VLLM_LOG=runs/vllm_sft.log REFS="" CONC=1 bash scripts/eval_dev.sh > runs/eval_m2_s1.log 2>&1 &
  (then the same with s2; python scripts/compare_runs.py runs/dev33_base_s1 runs/dev33_base_m1s1 runs/dev33_base_m2s1)
EOF
fi
only() { [ "$STEP" = "$1" ]; }       # the kit stages run only when named
KIT_DS="${KIT_DS:-dmitriigluzdov/gemma-4-serving-and-lora-verification-kit}"; KIT_RAW=runs/ext/dmitrii_kit/lora_native_resolved.jsonl
KIT_SHA=2a4771640ed867adf273945f75e7392707d62427c6c3468c8ad46db8a19fc3ca   # the file the laptop converted and checked (2026-11-03)
KIT_TOK="${KIT_TOK:-google/gemma-4-E4B-it-qat-w4a16-ct}"; KIT_OUT="${KIT_OUT:-adapters/e4b_kit}"; KIT_NAME="${KIT_NAME:-e4b_kit}"
KD=runs/sft/kitnat; PY=.venv-eval/bin/python
if only kitdata; then
  KIT_TOOLS="${KIT_TOOLS:-$(ls -d runs/*/traj 2>/dev/null | head -1)}"
  say "kit data: $KIT_DS → $KIT_RAW · tools from ${KIT_TOOLS:-NONE (set KIT_TOOLS=<run>/traj)} · tokenizer $KIT_TOK · parser ${PARSER_WHEEL:-vllm of the eval venv}"
  if [ ! -f "$KIT_RAW" ]; then
    mkdir -p "$(dirname "$KIT_RAW")"
    kaggle datasets download "$KIT_DS" -f lora_native_resolved.jsonl -p "$(dirname "$KIT_RAW")" --unzip >/dev/null 2>&1
    [ -f "$KIT_RAW.zip" ] && (cd "$(dirname "$KIT_RAW")" && unzip -oq lora_native_resolved.jsonl.zip && rm -f lora_native_resolved.jsonl.zip)
  fi
  SHA=$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$KIT_RAW" 2>/dev/null)
  if [ "$SHA" = "$KIT_SHA" ]; then echo "  $KIT_RAW sha256 ${SHA:0:16} = the laptop's file ✓"
  else echo "  $KIT_RAW sha256 ${SHA:-missing} ≠ the laptop's ${KIT_SHA:0:16} — stop: download failed or the dataset changed"; exit 1; fi
  rm -rf $KD ${KD}_val
  $PY train/convert_kit.py --prompt kit --data "$KIT_RAW" --tools-from "$KIT_TOOLS" --out $KD --val-out ${KD}_val > $KD.convert.log 2>&1
  echo "  $(grep -oE 'tools: [0-9]+ schemas from [^ ]+ \(sha256 [0-9a-f]+' $KD.convert.log))"
  grep -E "^held-out|DROPPED|^test-named|^dropped|^TOTAL" $KD.convert.log | cut -c1-260
  for d in $KD ${KD}_val; do
    $PY train/render.py $d --tokenizer "$KIT_TOK" --pack --sample 50 --out $d.rendered.jsonl > $d.render.log 2>&1
    echo "  $d: $(grep -E '^calls' $d.render.log | cut -c1-150) · $(grep -E '^RESULT' $d.render.log)"
    $PY train/parser_check.py $d --tokenizer "$KIT_TOK" ${PARSER_WHEEL:+--wheel $PARSER_WHEEL} --out $d.parser.json > $d.parser.log 2>&1
    echo "  $d: $(grep -E '^A5' $d.parser.log | cut -c1-90)"
  done
  echo "  laptop 2026-11-03: tools sha256 ee6a6459335f412d · TOTAL trajectories 101 (2 dropped, 24 renamed in 21) · train calls 1505 ·"
  echo "  targets 1461 · label tokens 416563 · sequences 130 · held out calls 159 · targets 155 · label tokens 33387 · sequences 14 · A5 0 excluded"
  echo "📷 everything from the 'kit data' line down"
fi
if only kit; then
  say "kit LoRA on GPU $CUDA_VISIBLE_DEVICES: $KD.rendered.jsonl, held out ${KD}_val.rendered.jsonl → $KIT_OUT · sft: $SFT"
  [ -s $KD.rendered.jsonl ] && [ -s ${KD}_val.rendered.jsonl ] || { echo "run kitdata first"; exit 1; }
  [ -f "$DQ/config.json" ] || { echo "no bf16 base in $DQ: run env and dequant first"; exit 1; }
  rm -rf "$KIT_OUT"
  .venv-train/bin/python train/sft_lora.py $SFT --data $KD.rendered.jsonl --eval ${KD}_val.rendered.jsonl --out "$KIT_OUT" 2>&1 \
    | grep -v Warning | tee runs/sft_kit.log | cut -c1-250
  echo "📷 the schedule line, the 'eval[...]' lines and both '== epoch' lines"
fi
if only kitsubs; then
  [ -f "$KIT_OUT/train_summary.json" ] || { echo "no $KIT_OUT/train_summary.json: run kit first"; exit 1; }
  PICK=$(python3 -c 'import json, sys
ep = [e for e in json.load(open(sys.argv[1]))["epochs"] if e.get("holdout_nll") is not None and not e.get("export_problems")]
print(min(ep, key=lambda e: e["holdout_nll"])["epoch"] if ep else 0, " ".join("epoch%d=%.4f" % (e["epoch"], e["holdout_nll"]) for e in ep) or "none")' "$KIT_OUT/train_summary.json")
  read -r BEST NLLS <<< "$PICK"
  EP="${KIT_EPOCH:-$BEST}"
  say "kitsubs: held-out NLL $NLLS → epoch ${EP} (KIT_EPOCH=${KIT_EPOCH:-auto}) → runs/adapters/$KIT_NAME"
  [ -f "$KIT_OUT/epoch$EP/adapter_model.safetensors" ] || { echo "no $KIT_OUT/epoch$EP/adapter_model.safetensors"; exit 1; }
  rm -rf "runs/adapters/$KIT_NAME"; mkdir -p runs/adapters; cp -r "$KIT_OUT/epoch$EP" "runs/adapters/$KIT_NAME"
  ls -la "runs/adapters/$KIT_NAME"; python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print('  adapter_config: r',c.get('r'),'alpha',c.get('lora_alpha'),'base',c.get('base_model_name_or_path'))" "runs/adapters/$KIT_NAME/adapter_config.json"
  BUILD_ONLY=1 LATEST=kit ADAPTER=$KIT_NAME SUBMISSION=runs/base_lb024 ARMS=base TAG=kitchk bash scripts/eval_dev.sh 2>&1 \
    | grep -E "^adapter |latest set|task prompt time line|latest e4b" | cut -c1-220
  echo "📷 these lines; eval lanes: SERVE_ADAPTER is taken from ADAPTER=$KIT_NAME (BOX_PLAN.md B-K step K5)"
fi
say "done ($STEP)"
