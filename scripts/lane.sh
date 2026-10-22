#!/usr/bin/env bash
# One "Kaggle lane": a private network namespace with its own 127.0.0.1:8000, one vLLM on one card (TP=1), one
# `swegemma eval` at concurrency 1 — the scorer's shape (one task at a time against one server). Three lanes on
# three cards run three passes in the wall time of one, without sharing a server (B-3c: a shared server at CONC=3
# slowed each stream and the time cap killed solvable runs). The harness hardcodes api_base 127.0.0.1:8000
# (HARNESS_README §3), so ports can't differ; the namespace makes the same address private per lane.
#
#   bash scripts/lane.sh check                       # once: can this box make namespaces? (unshare, lo, GPU, port)
#   GPU=0 bash scripts/lane.sh smoke                 # decode speed of ONE card at TP=1 (scripts/smoke.py, warm tok/s)
#   GPU=0 TAG=k1 TIME_MIN=5 bash scripts/lane.sh run # one pass of ARMS (default base) in a lane; eval_dev.sh flags pass through
#   for g in 0 1 2; do GPU=$g TAG=k$((g+1)) TIME_MIN=5 nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
#
# Inside the namespace there is no network at all, so the model must already be in the HF cache (it is after any
# earlier serve) and the offline switches below stop vLLM/transformers from trying the hub. Stop any host-level vLLM
# first (it holds memory on every card): pkill -f "vllm serve"; nvidia-smi.
# Logs: runs/vllm_lane$GPU.log (the server; eval_dev.sh counts its overflow lines) and whatever you redirect this to.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-run}"; GPU="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

if [ "$MODE" = check ]; then
  command -v unshare >/dev/null || { echo "no unshare (util-linux)"; exit 1; }
  unshare -rn bash -c '
    ip link set lo up 2>/dev/null || { echo "cannot bring lo up in the namespace"; exit 1; }
    nvidia-smi -L >/dev/null 2>&1 && echo "gpu visible in namespace: yes" || echo "gpu visible in namespace: NO"
    python3 -m http.server 8000 --bind 127.0.0.1 >/dev/null 2>&1 & HP=$!; sleep 1
    curl -s -o /dev/null 127.0.0.1:8000 && echo "private 127.0.0.1:8000: ok (independent of the host port)" || echo "private port: FAILED"
    kill $HP 2>/dev/null; true' || { echo "unshare -rn refused: user namespaces are disabled on this box (fallback: patch the harness api_base)"; exit 1; }
  echo "namespace check passed"; exit 0
fi

[ "$MODE" = smoke ] || [ "$MODE" = run ] || { echo "usage: lane.sh check|smoke|run"; exit 1; }
[ "$MODE" = run ] && [ -z "${TIME_MIN:-}" ] && { echo "set TIME_MIN explicitly (8 × 29 / <lane smoke tok/s>, rounded up)"; exit 1; }
# whole lane body runs inside one namespace; env (GPU, TAG, TIME_MIN, ARMS, SUBMISSION, …) is inherited
exec unshare -rn bash -c '
  set -euo pipefail
  ip link set lo up
  export CUDA_VISIBLE_DEVICES='"$GPU"' VLLM_LOG=runs/vllm_lane'"$GPU"'.log
  mkdir -p runs
  TP=1 PORT=8000 bash scripts/serve_vllm.sh > "$VLLM_LOG" 2>&1 &
  SRV=$!
  trap "kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true" EXIT
  for i in $(seq 1 180); do                       # up to 15 min for the first load from disk
    curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "vLLM died — tail -30 $VLLM_LOG"; exit 1; }
    sleep 5
  done
  echo "lane gpu '"$GPU"': vLLM up after ~$((i*5)) s ($(date +%H:%M))"
  if [ "'"$MODE"'" = smoke ]; then
    python3 scripts/smoke.py | grep -E "^--- "      # 📷 the warm line: TIME_MIN = 8 × 29 / <warm tok/s>, rounded up
  else
    CONC=1 bash scripts/eval_dev.sh                 # TAG, TIME_MIN, ARMS, SUBMISSION, REFS come from the environment
  fi'
