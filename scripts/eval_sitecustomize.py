"""Loaded as `sitecustomize` through PYTHONPATH by eval_dev.sh (symlinked to runs/pypath/sitecustomize.py) for every
`swegemma eval` it starts: always the sandbox shim (scripts/sandbox_shim.py, B-3i) and the scorer's events compaction
(scripts/compaction_shim.py, COMPACTION=0 turns it off); plus the remote-model request adapter
(scripts/ceil_sitecustomize.py) when MODELS_YAML is set (B-3f/B-3g ceiling runs); plus the trajectory log
(scripts/traj_log.py) when TRAJ_DIR is set. Each prints what it does into the run log."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # resolve(): follow the symlink to scripts/
import sandbox_shim   # noqa: E402,F401
import compaction_shim   # noqa: E402,F401

if os.environ.get("MODELS_YAML"):
    import ceil_sitecustomize   # noqa: E402,F401

if os.environ.get("TRAJ_DIR"):          # last: it wraps whatever litellm.acompletion is by now
    import traj_log   # noqa: E402,F401
