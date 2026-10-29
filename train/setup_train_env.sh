#!/usr/bin/env bash
# Training venv on a RunPod pod (Phase 0 A6): torch (cu128, Blackwell) + transformers + peft + accelerate +
# compressed-tensors, separate from .venv-eval (whose vLLM pins torch). Lives on the container disk like .venv-eval
# (imports from /vol are 15-30x slower); the uv cache on /vol makes a rebuild ~1-2 min. Prints every version and writes
# train/requirements-train.lock (uv pip freeze) so the next pod installs exactly the same set.
#   bash train/setup_train_env.sh            # → /root/venv-train, linked from .venv-train
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-/root/venv-train}"
PY="${PY:-3.12}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/vol/uv-cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/uv-python}"
export UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
T0=$(date +%s)
command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python "$PY" "$VENV"
ln -sfn "$VENV" .venv-train
if [ -f train/requirements-train.lock ] && [ "${LOCK:-1}" = 1 ]; then
  echo "installing train/requirements-train.lock"
  # the lock pins torch==…+cu128, which only the PyTorch index serves
  uv pip install -q --python "$VENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 -r train/requirements-train.lock
else
  echo "installing fresh (no lock): torch cu128 + transformers 5.20.2 + peft + accelerate + compressed-tensors"
  uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$VENV/bin/python" "transformers==5.20.2" peft accelerate compressed-tensors safetensors
  uv pip freeze --python "$VENV/bin/python" > train/requirements-train.lock
  echo "wrote train/requirements-train.lock ($(wc -l < train/requirements-train.lock | tr -d ' ') lines)"
fi
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, platform, torch
v = lambda n: md.version(n)
cap = "sm_%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "no GPU"
print(f"venv-train · python {platform.python_version()} · torch {torch.__version__} (CUDA {torch.version.cuda}, {cap}, "
      f"arch ok {cap in torch.cuda.get_arch_list() if torch.cuda.is_available() else '-'}) · transformers {v('transformers')} · "
      f"peft {v('peft')} · accelerate {v('accelerate')} · compressed-tensors {v('compressed-tensors')} · safetensors {v('safetensors')}")
PY
echo "setup_train_env done in $(( $(date +%s) - T0 )) s · $VENV"
