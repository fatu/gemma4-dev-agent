#!/usr/bin/env bash
# SFT M1 / M2 on the GPU box (one RTX PRO 6000 of g4-4), mirroring the RunPod run of 2026-10-29 (05_run_log):
# data from the private Kaggle dataset → train venv → dequantised bf16 base → 3-step smoke → M1 → M2 → submission dirs
# for eval_dev.sh. One stage per call so a shared box can be queued; `all` runs them in order. No `set -e`: every stage
# prints its own result and the script goes on so the log is complete.
#   GPU=2 bash train/train_box.sh all | data | env | dequant | smoke | m1 | m2 | subs
# Knobs (every one prints its effective value): GPU (CUDA_VISIBLE_DEVICES), DATA (dataset dir), VENV, DQ (dequant dir),
# KAGGLE_DS (dataset ref), EPOCHS LR ACCUM R ALPHA (sft_lora.py defaults otherwise).
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
say "done ($STEP)"
