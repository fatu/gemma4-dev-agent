#!/usr/bin/env bash
# W1 step 2: the competition dataset — harness wheels (swegemma, adk-eval-core), sample_submission/,
# tasks.jsonl, HARNESS_README.md — into harness/ (gitignored: never redistribute).
#   bash scripts/get_harness.sh
# Needs: rules accepted on kaggle.com/competitions/gemma-4-developer-agent, and your Kaggle API token at
# ~/.kaggle/kaggle.json (kaggle.com → Settings → API → "Create New Token"; chmod 600 it).
set -euo pipefail
cd "$(dirname "$0")/.."
COMP=gemma-4-developer-agent
command -v kaggle >/dev/null || pip install -q kaggle
[ -f ~/.kaggle/kaggle.json ] || [ -n "${KAGGLE_KEY:-}" ] || { echo "no Kaggle token: put kaggle.json in ~/.kaggle/ first"; exit 1; }
mkdir -p harness
kaggle competitions files -c "$COMP"
kaggle competitions download -c "$COMP" -p harness/
cd harness
for z in *.zip; do [ -e "$z" ] && unzip -q -o "$z" && rm "$z"; done
echo; echo "== harness/"; ls
echo; echo "== HARNESS_README.md (head) — 📷 the usage section"
f=$(find . -iname "HARNESS_README*" | head -1); [ -n "$f" ] && sed -n 1,120p "$f" || echo "(no HARNESS_README found — list the files above)"
