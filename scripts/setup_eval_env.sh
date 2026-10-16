#!/usr/bin/env bash
# Build .venv-eval on the GPU box from requirements-eval.txt (scorer-matched pins + pytest for verification).
#   bash scripts/setup_eval_env.sh && source .venv-eval/bin/activate && bash scripts/env_report.sh   # 📷
# Needs harness/wheelhouse/ (scripts/get_harness.sh — run it wherever Kaggle is reachable, then copy harness/ to the box).
# Uses uv when present (fast); otherwise python -m venv + pip. PYTHON=3.12 by default (uv fetches it if missing).
# Two stages, like the box env was built: the pins don't resolve in one go — prometheus-fastapi-instrumentator 8.1.0
# declares starlette>=1 while google-adk 1.36.1 declares starlette<1 (uv: "No solution found", Colab 2026-10-16).
# So everything else first, then the prometheus pins with --no-deps (8.1.0 is what keeps vLLM 0.19's server from 500ing).
set -euo pipefail
cd "$(dirname "$0")/.."
ls harness/wheelhouse/swegemma-*.whl >/dev/null 2>&1 || { echo "harness/wheelhouse/ missing — run scripts/get_harness.sh first"; exit 1; }
PY="${PYTHON:-3.12}"
MAIN=$(mktemp); grep -vE '^prometheus' requirements-eval.txt > "$MAIN"
LATE=$(grep -E '^prometheus' requirements-eval.txt | tr '\n' ' ')
have=$(.venv-eval/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
if [ -d .venv-eval ] && [ "$have" != "${PY#python}" ]; then
  echo "rebuilding .venv-eval: Python ${have:-?} ≠ $PY"; rm -rf .venv-eval
fi
if command -v uv >/dev/null; then
  unset UV_SYSTEM_PYTHON                               # Colab sets it; it makes uv ignore the venv
  export UV_PYTHON_DOWNLOADS=automatic                 # fetch $PY if the system lacks it (Colab: 3.13 only?)
  [ -d .venv-eval ] || uv venv --python "$PY" .venv-eval
  uv pip install --python .venv-eval/bin/python --find-links harness/wheelhouse -r "$MAIN"
  uv pip install --python .venv-eval/bin/python --find-links harness/wheelhouse --no-deps $LATE
else
  [ -d .venv-eval ] || "python$PY" -m venv .venv-eval 2>/dev/null || "$PY" -m venv .venv-eval
  .venv-eval/bin/python -m pip install -q --upgrade pip
  .venv-eval/bin/python -m pip install --find-links harness/wheelhouse -r "$MAIN"
  .venv-eval/bin/python -m pip install --find-links harness/wheelhouse --no-deps $LATE
fi
echo; bash scripts/env_report.sh
