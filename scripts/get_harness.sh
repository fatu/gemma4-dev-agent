#!/usr/bin/env bash
# W1 step 2: competition data + the harness CLI, into harness/ (gitignored: never redistribute).
#   bash scripts/get_harness.sh
# Needs: rules accepted on kaggle.com/competitions/gemma-4-developer-agent, your Kaggle API token at
# ~/.kaggle/kaggle.json (kaggle.com → Settings → API → "Create New Token"; chmod 600), and ~25 GB of disk.
#
# Two downloads — the harness README documents the CLI but the CLI itself ships separately:
#   1. competition data (~22 GB): HARNESS_README.md, tasks.jsonl (129 tasks), sample_submission/, snapshots/*.tgz
#      (one repo snapshot per task), embeddings/*.npz (graph tools), wheels/ (sandbox test deps), docker/
#   2. the "wheelhouse" dataset metric/gemma-4-developer-agent-wheelhouse (~870 MB): swegemma 0.2.x (the `swegemma`
#      console command) + adk-submission + adk-eval-core + google-adk — none of them are on PyPI, so the install
#      points pip at the wheelhouse with --find-links (NOT --no-index: their public deps still come from PyPI).
set -euo pipefail
cd "$(dirname "$0")/.."
COMP=gemma-4-developer-agent
command -v kaggle >/dev/null || pip install -q kaggle
[ -f ~/.kaggle/kaggle.json ] || [ -n "${KAGGLE_KEY:-}" ] || { echo "no Kaggle token: put kaggle.json in ~/.kaggle/ first"; exit 1; }
mkdir -p harness/wheelhouse

if [ ! -f harness/HARNESS_README.md ]; then
  kaggle competitions download -c "$COMP" -p harness/
  ( cd harness && for z in *.zip; do [ -e "$z" ] && unzip -q -o "$z" && rm "$z"; done )
fi
if ! ls harness/wheelhouse/swegemma-*.whl >/dev/null 2>&1; then
  kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p harness/wheelhouse --unzip
fi
pip install -q --find-links harness/wheelhouse swegemma

echo; echo "== harness/"; ls harness; echo "tasks: $(wc -l < harness/tasks.jsonl) · snapshots: $(ls harness/snapshots | wc -l)"
echo; echo "== swegemma  📷"; swegemma --version 2>/dev/null || pip show swegemma | head -2; swegemma eval --help | head -40
