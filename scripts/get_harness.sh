#!/usr/bin/env bash
# W1 step 2: competition data + the harness CLI, into harness/ (gitignored: never redistribute).
#   bash scripts/get_harness.sh
# Download only — runs anywhere Kaggle is reachable. If the GPU box can't reach Kaggle (firewalled), run this on a laptop, copy
# harness/ to the box, then build the venv there with scripts/setup_eval_env.sh.
# Needs: rules accepted on kaggle.com/competitions/gemma-4-developer-agent, a Kaggle token (KAGGLE_API_TOKEN, or
# ~/.kaggle/kaggle.json; kaggle.com → Settings → API), the kaggle CLI ≥ 2 (1.7.x ignores KAGGLE_API_TOKEN), ~25 GB of disk.
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
kaggle --version 2>/dev/null | grep -qE 'CLI [2-9]' || { echo "need the kaggle CLI >= 2:  uv tool install 'kaggle>=2'  (or pip install 'kaggle>=2')"; exit 1; }
[ -f ~/.kaggle/kaggle.json ] || [ -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ] || { echo "no Kaggle token: export KAGGLE_API_TOKEN=… or put kaggle.json in ~/.kaggle/"; exit 1; }
mkdir -p harness/wheelhouse

if [ ! -f harness/HARNESS_README.md ]; then
  kaggle competitions download -c "$COMP" -p harness/
  ( cd harness && for z in *.zip; do [ -e "$z" ] && unzip -q -o "$z" && rm "$z"; done )
fi
if ! ls harness/wheelhouse/swegemma-*.whl >/dev/null 2>&1; then
  kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p harness/wheelhouse --unzip
fi

echo; echo "== harness/"; ls harness; echo "tasks: $(wc -l < harness/tasks.jsonl) · snapshots: $(ls harness/snapshots | wc -l)"
echo "wheelhouse: $(ls harness/wheelhouse | wc -l) files"
echo "next (on the box): bash scripts/setup_eval_env.sh"   # builds .venv-eval from requirements-eval.txt
