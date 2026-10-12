#!/usr/bin/env bash
# One-screen environment report for a photo (W1 env fixes → repo).
#   bash scripts/env_report.sh
cd "$(dirname "$0")/.."
PIP=.venv-eval/bin/pip; [ -x "$PIP" ] || PIP=pip
W="${COLUMNS:-160}"
echo "== PKGS (.venv-eval)"
$PIP freeze 2>/dev/null | grep -iE '^(vllm|torch|transformers|compressed.tensors|swegemma|adk|google-adk|prometheus|pytest|kaggle)' | column -c "$W"
echo "== kaggle CLI: $(kaggle --version 2>/dev/null || echo none)"
echo "== WHEELHOUSE (prometheus/pytest): $(ls harness/wheelhouse 2>/dev/null | grep -iE 'prometheus|pytest' | paste -sd' ' - || true)"
echo "== PIP CONFIG: $($PIP config list 2>/dev/null | paste -sd' ' -)"
echo "== GIT: $(git log --oneline -1)"
echo "== DIRTY: $(git status --short | paste -sd' ' -)"
