"""Make `--sandbox subprocess` import the task repo the way the scorer's Docker sandbox does (BOX_PLAN.md B-3i).

In Docker, `harness/sandbox/setup.py`'s fast path writes `workspace_paths.pth` into site-packages: /workspace and every
top-level directory under it (so a src-layout repo's `src/` is importable). swegemma 0.2.11 skips that step — and the
editable install — for the subprocess sandbox, and pytest runs with PYTHONSAFEPATH=1, so a src-layout repo (requests)
imported the harness venv's own copy instead of the patched workspace: no requests patch was ever really verified
(Insight 29). Flat repos (fastapi/, rich/) were fine only because pytest prepends the root conftest's directory.

Import this module before the harness creates sandboxes (goldcheck.py / reverify.py import it; eval_dev.sh loads it through
scripts/eval_sitecustomize.py on PYTHONPATH). It wraps SubprocessManager.start: after the per-task venv exists, it writes
`00_workspace.pth` (an `import` line that puts <workspace> and <workspace>/src first on sys.path) into the venv's site-packages. Prints the effective paths once per process.
No effect on the Docker sandbox. SANDBOX_SHIM=0 disables it (and says so).
"""
import os
import sys

_state = {"announced": False, "count": 0}


def _write_pth(manager, sid):
    paths = manager._sandboxes.get(sid) or {}
    ws, venv = paths.get("workspace"), paths.get("venv")
    if not ws or not venv:
        return
    paths_ = [str(ws), str(ws / "src")]
    # an `import` line runs at startup and puts the workspace FIRST: a plain path list would be appended after the
    # harness's `_host_env.pth` (host site-packages — where the venv's own requests lives), which is the whole problem
    line = f"import sys; sys.path[:0] = [p for p in {paths_!r} if p not in sys.path]\n"
    n = 0
    for sp in venv.glob("lib/python*/site-packages"):
        (sp / "00_workspace.pth").write_text(line, encoding="utf-8"); n += 1
    _state["count"] += 1
    if not _state["announced"]:
        _state["announced"] = True
        print(f"sandbox shim: per-task venv gets 00_workspace.pth → sys.path[:0] = {paths_[0]}, {paths_[1]} "
              f"({n} site-packages dir{'s' if n != 1 else ''}; subprocess sandbox; what Docker's workspace_paths.pth does)", flush=True)


def install():
    if os.environ.get("SANDBOX_SHIM", "1") == "0":
        print("sandbox shim: OFF (SANDBOX_SHIM=0) — src-layout repos import the harness venv's copy", flush=True)
        return False
    try:
        from swegemma.sandbox import subprocess as sp
    except Exception:
        return False
    cls = sp.SubprocessManager
    if getattr(cls, "_shim_installed", False):
        return True
    orig = cls.start

    def start(self, *a, **kw):
        sid = orig(self, *a, **kw)
        try:
            _write_pth(self, sid)
        except Exception as e:                     # never break a run over the shim
            print(f"sandbox shim: could not write _workspace.pth for {sid}: {e}", flush=True)
        return sid

    cls.start = start
    cls._shim_installed = True
    return True


install()
