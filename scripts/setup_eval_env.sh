#!/usr/bin/env bash
# Build .venv-eval on the GPU box from requirements-eval.txt (scorer-matched pins + pytest for verification).
#   bash scripts/setup_eval_env.sh && source .venv-eval/bin/activate && bash scripts/env_report.sh   # 📷
# Needs harness/wheelhouse/ (scripts/get_harness.sh — run it wherever Kaggle is reachable, then copy harness/ to the box).
# Uses uv when present (fast); otherwise python -m venv + pip. PYTHON=python3.12 to choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
ls harness/wheelhouse/swegemma-*.whl >/dev/null 2>&1 || { echo "harness/wheelhouse/ missing — run scripts/get_harness.sh first"; exit 1; }
PY="${PYTHON:-python3}"
if command -v uv >/dev/null; then
  [ -d .venv-eval ] || uv venv --python "$PY" .venv-eval
  uv pip install --python .venv-eval/bin/python --find-links harness/wheelhouse -r requirements-eval.txt
else
  [ -d .venv-eval ] || "$PY" -m venv .venv-eval
  .venv-eval/bin/python -m pip install -q --upgrade pip
  .venv-eval/bin/python -m pip install --find-links harness/wheelhouse -r requirements-eval.txt
fi
echo; bash scripts/env_report.sh
