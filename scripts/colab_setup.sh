#!/usr/bin/env bash
# Colab step 1: a fresh Colab G4 runtime (1× RTX PRO 6000, 96 GB) → the same eval env as the GPU box.
#   In a notebook cell first (the terminal can't mount Drive):
#     from google.colab import drive; drive.mount('/content/drive')
#   Then in Colab's Terminal:
#     cd /content && git clone https://github.com/fatu/gemma4-dev-agent.git && cd gemma4-dev-agent   # later: git pull
#     bash scripts/colab_setup.sh                    # 📷 the summary at the end · ~20-30 min on a fresh runtime
# Everything big stays on the runtime's local disk and is fetched again each session (Kaggle/HF → Colab is fast):
#   harness/ (~22 GB, Kaggle), model weights (~17 GB, HF cache), .venv-eval. Re-running is safe: done steps are skipped.
# On a CPU runtime it runs in check-only mode: Drive, tokens, Kaggle rules accepted, HF access to the model — nothing
# big is downloaded, because changing the runtime type wipes /content. Fix what it flags, then switch to G4 and re-run.
# Only runs/ lives on Drive (runs → $G4D/runs symlink), so a disconnect never loses finished tasks.
# Tokens: KAGGLE_API_TOKEN (kaggle.com → Settings → API) and HF_TOKEN (the Gemma licence accepted on huggingface.co),
# from the environment or from $G4D/secrets.env — two `export X=…` lines on your private Drive, never in this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
G4D="${G4D:-/content/drive/MyDrive/gemma4}"
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
ok() { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; exit 1; }

echo "== 1/6 GPU + disk"
if gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null); then
  ok "$gpu"
  avail=$(df -BG --output=avail /content | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 70 ] || bad "only ${avail} GB free on /content, need ~70 (harness 22 + weights 17 + venv)"
  ok "${avail} GB free on /content"
else
  gpu=""; echo "  ! no GPU — CPU check-only mode (no downloads: a runtime change wipes /content)"
fi

echo "== 2/6 Drive + tokens"
[ -d /content/drive/MyDrive ] || bad "Drive not mounted — run drive.mount('/content/drive') in a notebook cell"
mkdir -p "$G4D/runs"
if [ -L runs ]; then ok "runs → $(readlink runs)"
elif [ -e runs ]; then bad "runs/ exists and is not a link — move it into $G4D/runs first"
else ln -s "$G4D/runs" runs; ok "runs → $G4D/runs"; fi
if [ -f "$G4D/secrets.env" ]; then source "$G4D/secrets.env"; ok "tokens from $G4D/secrets.env"; fi
[ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f ~/.kaggle/kaggle.json ] || bad "no KAGGLE_API_TOKEN — add 'export KAGGLE_API_TOKEN=…' to $G4D/secrets.env"
[ -n "${HF_TOKEN:-}" ] || echo "  ! no HF_TOKEN — the weights download fails if the model is gated for you"

echo "== 3/6 tools (uv, kaggle CLI ≥ 2)"
command -v uv >/dev/null || pip install -q uv
kaggle --version 2>/dev/null | grep -qE 'CLI (2\.([2-9]|[1-9][0-9])|[3-9])' || pip install -q -U 'kaggle>=2.2'   # 2.0.2: upload-info 'token' bug
ok "$(uv --version) · $(kaggle --version)"

if [ -z "$gpu" ]; then
  echo "== check-only: access"
  kaggle competitions files gemma-4-developer-agent 2>&1 | grep -qE '\.(jsonl|md|tgz|npz|zip|whl)' \
    && ok "Kaggle: competition data visible (rules accepted)" \
    || bad "Kaggle can't list gemma-4-developer-agent — accept the rules on its page, check KAGGLE_API_TOKEN"
  wh=$(kaggle datasets files metric/gemma-4-developer-agent-wheelhouse 2>&1 || true)   # one page only (~20 files)
  if grep -q '\.whl' <<<"$wh"; then ok "Kaggle: wheelhouse visible ($(grep -c '\.whl' <<<"$wh") wheels on page 1)"
  else echo "$wh" | head -5; bad "Kaggle can't list the wheelhouse dataset metric/gemma-4-developer-agent-wheelhouse (output above)"; fi
  MODEL="$MODEL" python3 -c '
import os, sys
from huggingface_hub import HfApi
m = os.environ["MODEL"]
try:
    files = HfApi().list_repo_files(m)
except Exception as e:
    sys.exit(f"  ✗ HF: no access to {m}: {type(e).__name__} — accept the Gemma licence, check HF_TOKEN")
n = sum(x.endswith(".safetensors") for x in files)
print(f"  ✓ HF: {m} visible ({n} safetensors files)")'
  echo; echo "== check-only passed — Runtime → Change runtime type → G4, then the same two commands again"
  exit 0
fi

echo "== 4/6 harness (competition data + wheelhouse)"
bash scripts/get_harness.sh

echo "== 5/6 .venv-eval"
bash scripts/setup_eval_env.sh

# the subprocess sandbox builds a venv per task with this Python (`python3.12 -m venv`); Colab's Debian python3.12 has no
# ensurepip until python3.12-venv is installed (seen 2026-10-16: "ensurepip is not available", every task fails)
pyv=$(.venv-eval/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
t=$(mktemp -d)
if ! .venv-eval/bin/python -m venv "$t/v" >/dev/null 2>&1; then
  echo "  ! python$pyv can't make venvs (no ensurepip) — apt-get install python$pyv-venv"
  apt-get install -y -qq "python$pyv-venv" >/dev/null || { apt-get update -qq && apt-get install -y -qq "python$pyv-venv" >/dev/null; }
  rm -rf "$t/v"; .venv-eval/bin/python -m venv "$t/v" >/dev/null 2>&1 || bad "python$pyv still can't make venvs — the sandbox will fail"
fi
rm -rf "$t"; ok "python$pyv makes venvs (sandbox OK)"

echo "== 6/6 weights ($MODEL)"
MODEL="$MODEL" .venv-eval/bin/python -c '
import os, pathlib
from huggingface_hub import snapshot_download
p = pathlib.Path(snapshot_download(os.environ["MODEL"]))
gb = sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e9
print(f"  ✓ {p}  {gb:.1f} GB")'

cat <<EOF

== ready ($gpu) — next:
  source .venv-eval/bin/activate
  nohup env TP=1 GPU_UTIL=0.85 bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &
  python scripts/smoke.py          # 📷 once the server is up (tail -f runs/vllm.log)
EOF
