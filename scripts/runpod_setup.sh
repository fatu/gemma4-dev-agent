#!/usr/bin/env bash
# RunPod: a fresh pod from our template → the same eval env as the GPU box, with everything big on the network volume.
# Template (RunPod → Templates → New, private): image runpod/pytorch with CUDA ≥ 12.8 (Blackwell cards) · container
# disk 50 GB (the sandbox's per-task temp dirs live in /tmp) · network volume mounted at /vol, not the default
# /workspace (the harness rewrites "/workspace" in the agent's commands) · port 22 · environment variables:
#   HF_HOME=/vol/hf  UV_CACHE_DIR=/vol/uv-cache  VLLM_CACHE_ROOT=/vol/vllm-cache
#   HF_TOKEN={{ RUNPOD_SECRET_hf_token }}  KAGGLE_API_TOKEN={{ RUNPOD_SECRET_kaggle_api_token }}   (RunPod → Secrets)
# Then in the pod's terminal:
#   cd /vol && git clone https://github.com/fatu/gemma4-dev-agent.git && cd gemma4-dev-agent   # later pods: git pull
#   bash scripts/runpod_setup.sh                    # 📷 the summary · first pod ~20-30 min, later pods ~1-2 min
#   MODEL=google/gemma-4-31B-it-qat-w4a16-ct bash scripts/runpod_setup.sh   # another base model's weights
# On /vol (kept when the pod is stopped or terminated): this repo, harness/ (~22 GB), the HF weights cache, the uv
# cache, vLLM's compile cache, runs/. On the container disk (gone with the pod, so re-run this on every new pod):
# .venv-eval (a link from the repo to $LOCAL/venv-eval, rebuilt from the uv cache) and the uv Python it uses, uv, the
# kaggle CLI, and the env lines this script adds to ~/.bashrc. Why the venv is local: /vol is a network filesystem,
# and Python imports from it were 15-30x slower (bare python 411 ms vs 13 ms, import torch 23.8 s vs 1.6 s, 2026-10-21);
# that slowed the server start (13 min) and every sandbox command and verification inside the time budget.
# Python 3.13 (PYTHON=…): the scorer's sandbox is python:3.13-slim, and its ssl defaults differ from 3.12's.
# Pods are containers: no Docker inside, so runs use --sandbox subprocess (scripts/build_sandbox_image.sh is box-only).
# Tokens: HF_TOKEN (Gemma licence accepted on huggingface.co); KAGGLE_API_TOKEN only while harness/ is incomplete.
# Every step is skipped when done; the summary prints what the pod ended up with (GPU arch vs torch, versions, paths).
set -euo pipefail
cd "$(dirname "$0")/.."
T0=$(date +%s)
VOL="${VOL:-/vol}"
MODEL="${MODEL:-google/gemma-4-E4B-it-qat-w4a16-ct}"   # RunPod = small-model bonus on 32 GB cards; 31B: MODEL=…
LOCAL="${LOCAL:-/root}"                                # container disk: .venv-eval + its uv Python
export PYTHON="${PYTHON:-3.13}"                        # setup_eval_env.sh builds .venv-eval with it
ok() { echo "  ✓ $*"; }
warn() { echo "  ! $*"; }
bad() { echo "  ✗ $*"; exit 1; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
have_harness() { [ -f harness/HARNESS_README.md ] && ls harness/wheelhouse/swegemma-*.whl >/dev/null 2>&1; }

echo "== 1/7 GPU + disks"
gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1) \
  || bad "no GPU: nvidia-smi failed"
ngpu=$(nvidia-smi -L | wc -l | tr -d ' ')
drv_cuda=$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | grep -oE '[0-9.]+$' || echo '?')
ok "$ngpu × $gpu · driver CUDA $drv_cuda"
[ -d "$VOL" ] || bad "no $VOL: attach the network volume with mount path $VOL (template → Volume Mount Path)"
if mountpoint -q "$VOL"; then ok "$VOL is a mount ($(df -h --output=fstype,size "$VOL" | tail -1 | xargs))"
else warn "$VOL is not a mount: it sits on the container disk and is lost with the pod"; fi
case "$PWD/" in "$VOL"/*) ok "repo on the volume: $PWD" ;;
  *) warn "repo at $PWD, not under $VOL: harness/, .venv-eval and runs/ are lost with the pod" ;; esac
case "$PWD/" in /workspace/*) warn "repo under /workspace, which the harness rewrites in agent commands: prefer $VOL" ;; esac
need=10; have_harness || need=70
avail=$(free_gb "$PWD")
[ "$avail" -ge "$need" ] || bad "only $avail GB free for the repo, need ~$need (harness 22 + weights + venv)"
ok "$avail GB free for the repo"
TMP="${TMPDIR:-/tmp}"; tmp_free=$(free_gb "$TMP")
[ "$tmp_free" -ge 15 ] || warn "only $tmp_free GB free in $TMP: the sandbox's per-task temp dirs need room (container disk 50 GB)"
ok "$TMP: $tmp_free GB free (sandbox temp dirs)"
local_free=$(free_gb "$LOCAL")
[ "$local_free" -ge 25 ] || warn "only $local_free GB free in $LOCAL: .venv-eval needs ~20 GB there"
ok "$LOCAL: $local_free GB free (.venv-eval, container disk)"

echo "== 2/7 env + tokens"
env_src=shell      # shell = already exported · container = template's env, which SSH sessions don't inherit · default
for kv in "HF_HOME=$VOL/hf" "UV_CACHE_DIR=$VOL/uv-cache" "VLLM_CACHE_ROOT=$VOL/vllm-cache"; do
  k=${kv%%=*}; v=${kv#*=}
  if [ -z "${!k:-}" ]; then
    c=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s|^$k=||p" | head -1 || true)
    if [ -n "$c" ]; then v=$c; [ "$env_src" = default ] || env_src=container
    else env_src=default; fi
    export "$k=$v"
    grep -qxF "export $k=$v" ~/.bashrc 2>/dev/null || echo "export $k=$v" >> ~/.bashrc
    if [ -n "$c" ]; then ok "$k=$v from the template (container env; added to ~/.bashrc for SSH shells)"
    else warn "$k unset → $v (added to ~/.bashrc for this pod; set it in the template)"; fi
  else ok "$k=${!k}"; fi
  case "${!k}/" in "$VOL"/*) ;; *) warn "$k=${!k} is not on $VOL: it is lost with the pod" ;; esac
done
export UV_PYTHON_INSTALL_DIR="$LOCAL/uv-python"        # the venv's Python + stdlib: local, whatever the template says
sed -i '/^export UV_PYTHON_INSTALL_DIR=/d' ~/.bashrc 2>/dev/null || true; echo "export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR" >> ~/.bashrc
ok "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR (container disk; overrides the template's $(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's|^UV_PYTHON_INSTALL_DIR=||p' | head -1 || true))"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"            # the uv cache (/vol) and the venv (container disk) are different filesystems
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"   # venv → a uv Python (has ensurepip, unlike some images')
if [ -n "${HF_TOKEN:-}" ]; then ok "HF_TOKEN set"; else warn "no HF_TOKEN: the weights download fails if the model is gated for you"; fi
if have_harness; then ok "harness/ present: no Kaggle token needed"
elif [ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f ~/.kaggle/kaggle.json ]; then ok "Kaggle token set"
else bad "harness/ incomplete and no KAGGLE_API_TOKEN: add the RunPod secret to the template"; fi

echo "== 3/7 tools (uv, kaggle CLI ≥ 2.2)"
command -v uv >/dev/null || pipi uv
kaggle --version 2>/dev/null | grep -qE 'CLI (2\.([2-9]|[1-9][0-9])|[3-9])' || pipi -U 'kaggle>=2.2'   # 2.0.2: upload-info 'token' bug
ok "$(uv --version) · $(kaggle --version)"

echo "== 4/7 harness (competition data + wheelhouse)"
if have_harness; then ok "harness/: $(wc -l < harness/tasks.jsonl | tr -d " ") tasks · $(ls harness/snapshots | wc -l | tr -d " ") snapshots"
else bash scripts/get_harness.sh; fi

echo "== 5/7 .venv-eval (Python $PYTHON, $LOCAL/venv-eval, linked from the repo)"
VENV_DIR="$LOCAL/venv-eval"
if [ -d .venv-eval ] && [ ! -L .venv-eval ]; then     # pods before 2026-10-22 built it on the volume
  warn "removing the old .venv-eval on $VOL (Python $(.venv-eval/bin/python -c 'import platform; print(platform.python_version())' 2>/dev/null || echo ?))"
  rm -rf .venv-eval
fi
vv=$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
[ -z "$vv" ] || [ "$vv" = "$PYTHON" ] || { warn "$VENV_DIR has Python $vv ≠ $PYTHON: rebuilding"; rm -rf "$VENV_DIR"; }
[ -x "$VENV_DIR/bin/python" ] || uv venv -q --python "$PYTHON" "$VENV_DIR"
ln -sfn "$VENV_DIR" .venv-eval
ok ".venv-eval → $(readlink .venv-eval)"
bash scripts/setup_eval_env.sh
# the subprocess sandbox builds a venv per task with this Python; without ensurepip every task fails (Colab 2026-10-16)
pyv=$(.venv-eval/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
t=$(mktemp -d)
.venv-eval/bin/python -m venv "$t/v" >/dev/null 2>&1 || bad "python$pyv can't make venvs (no ensurepip): the sandbox would fail"
rm -rf "$t"; ok "python$pyv makes venvs (sandbox OK) · $(readlink -f .venv-eval/bin/python)"

echo "== 6/7 test deps (pytest-httpbin, inline-snapshot)"
if .venv-eval/bin/python -c 'import pytest_httpbin, inline_snapshot' 2>/dev/null; then ok "already in .venv-eval"
else bash scripts/install_testdeps.sh; fi

echo "== 7/7 weights ($MODEL)"
weights=$(MODEL="$MODEL" .venv-eval/bin/python -c '
import os, pathlib
from huggingface_hub import snapshot_download
p = pathlib.Path(snapshot_download(os.environ["MODEL"]))
gb = sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e9
print(f"{gb:.1f} GB · {p}")')
ok "$weights"

echo
echo "== runpod_setup $(date '+%F %T %Z') · commit $(git rev-parse --short HEAD) · $(( $(date +%s) - T0 )) s   📷"
GPU="$gpu" NGPU="$ngpu" DRV_CUDA="$drv_cuda" VOL="$VOL" TMP="$TMP" TMP_FREE="$tmp_free" ENV_SRC="$env_src" \
  MODEL="$MODEL" WEIGHTS="$weights" .venv-eval/bin/python - <<'PY'
import importlib.metadata as md, os, pathlib, platform, shutil
e = os.environ
def ver(n):
    try:
        return md.version(n)
    except md.PackageNotFoundError:
        return "-"
import torch
if torch.cuda.is_available():
    cap = "sm_%d%d" % torch.cuda.get_device_capability()
    arch = torch.cuda.get_arch_list()
    built = "yes" if cap in arch or cap.replace("sm", "compute") in arch else f"NO (torch has {' '.join(arch)})"
    gpu_arch = f"{cap} · torch built for it: {built}"
else:
    gpu_arch = f"torch sees no GPU (torch CUDA {torch.version.cuda} vs driver CUDA {e['DRV_CUDA']}: pick a newer driver)"
try:
    import pytest_httpbin, inline_snapshot  # noqa: F401
    testdeps = "ok"
except ImportError:
    testdeps = "MISSING"
h = pathlib.Path("harness")
vol_free = shutil.disk_usage(e["VOL"]).free / 1e9
rows = [
    ("gpu", f"{e['NGPU']} × {e['GPU']} · driver CUDA {e['DRV_CUDA']} · {gpu_arch}"),
    ("volume", f"{e['VOL']} · {vol_free:.0f} GB free · repo {pathlib.Path.cwd()}"),
    ("scratch", f"{e['TMP']} · {e['TMP_FREE']} GB free (sandbox temp dirs, container disk)"),
    ("env", f"HF_HOME={os.environ.get('HF_HOME')} · VLLM_CACHE_ROOT={os.environ.get('VLLM_CACHE_ROOT')} · "
            f"UV_PYTHON_INSTALL_DIR={os.environ.get('UV_PYTHON_INSTALL_DIR')} · "
            + {"shell": "from this shell", "container": "from the template (copied to ~/.bashrc for SSH shells)"}
                .get(e["ENV_SRC"], "defaults added to ~/.bashrc: set them in the template")),
    ("tokens", f"HF_TOKEN {'set' if os.environ.get('HF_TOKEN') else 'unset'} · KAGGLE_API_TOKEN {'set' if os.environ.get('KAGGLE_API_TOKEN') else 'unset'}"),
    ("venv", f"{os.path.realpath('.venv-eval')} · Python {platform.python_version()} · vllm {ver('vllm')} · torch {torch.__version__} (CUDA {torch.version.cuda}) · "
             f"transformers {ver('transformers')} · swegemma {ver('swegemma')} · test deps {testdeps}"),
    ("harness", f"{sum(1 for _ in open(h / 'tasks.jsonl'))} tasks · {len(list((h / 'snapshots').iterdir()))} snapshots · "
                f"{len(list((h / 'wheels').iterdir()))} wheels"),
    ("weights", f"{e['MODEL']} · {e['WEIGHTS']}"),
    ("sandbox", "subprocess only (no Docker inside a pod)"),
]
for k, v in rows:
    print(f"  {k:<8} {v}")
PY

cat <<EOF

== next
EOF
case "$env_src" in shell) ;; *) echo "  source ~/.bashrc                 # HF_HOME etc. for this shell (vLLM reads the weights from there)" ;; esac
cat <<EOF
  source .venv-eval/bin/activate && mkdir -p runs
  MODEL=$MODEL TP=1 GPU_UTIL=0.9 nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &
  head -5 runs/vllm.log && python scripts/smoke.py      # 📷 once the server is up (tail -f runs/vllm.log)
EOF
