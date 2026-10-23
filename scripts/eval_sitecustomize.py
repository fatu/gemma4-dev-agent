"""Loaded as `sitecustomize` through PYTHONPATH by eval_dev.sh (symlinked to runs/pypath/sitecustomize.py) for every
`swegemma eval` it starts: always the sandbox shim (scripts/sandbox_shim.py, B-3i); plus the remote-model request adapter
(scripts/ceil_sitecustomize.py) when MODELS_YAML is set (B-3f/B-3g ceiling runs). Both print what they do into the run log."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # resolve(): follow the symlink to scripts/
import sandbox_shim   # noqa: E402,F401

if os.environ.get("MODELS_YAML"):
    import ceil_sitecustomize   # noqa: E402,F401
