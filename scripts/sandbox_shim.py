"""Make `--sandbox subprocess` import the task repo the way the scorer's Docker sandbox does (BOX_PLAN.md B-3i), and an
optional LENIENT mode that pins a task's venv to the dependency versions its tests were written against (B-3i-L).

Strict (default). In Docker, `harness/sandbox/setup.py`'s fast path writes `workspace_paths.pth` into site-packages:
/workspace and every top-level directory under it (so a src-layout repo's `src/` is importable). swegemma 0.2.11 skips that
step — and the editable install — for the subprocess sandbox, and pytest runs with PYTHONSAFEPATH=1, so a src-layout repo
(requests) imported the harness venv's own copy instead of the patched workspace: no requests patch was ever really verified
(Insight 29). Flat repos (fastapi/, rich/) were fine only because pytest prepends the root conftest's directory. This module
wraps SubprocessManager.start: after the per-task venv exists it writes `00_workspace.pth` (an `import` line that puts
<workspace> and <workspace>/src first on sys.path) into the venv's site-packages. Prints the effective paths once per process.

LENIENT=1 (analysis only — never the reference numbers). The shared sandbox installs the highest wheel of every package, so
5 dev tasks fail even with the gold patch (pygments / werkzeug drift, verified by hand). With LENIENT=1 the per-task venv of
those tasks gets the pins in PINS below (`pip install --no-index` from harness/wheels + runs/lenient_wheels, --no-deps; the
venv's own site-packages precedes the host entries, so the pin wins), for the agent's sandbox and the verification sandbox
alike. The task id reaches `start()` through a ContextVar set by wrappers around swegemma.evaluate.verify_task /
run_agent_sandbox (asyncio.to_thread carries the context). Prints the mode at load and one `lenient: <task> → <pins>` line
per task; a pin that fails to install is printed and the task runs strict. LENIENT_PINS="task:spec,spec;task:spec" adds or
overrides entries. Tasks whose tests are `async def` under the harness's `-p no:anyio` are not pinnable (no entry).

Import this module before the harness creates sandboxes (goldcheck.py / reverify.py import it; eval_dev.sh loads it through
scripts/eval_sitecustomize.py on PYTHONPATH). No effect on the Docker sandbox. SANDBOX_SHIM=0 disables everything (and says so).
"""
import contextvars
import functools
import os
import pathlib
import shutil
import subprocess
import sys
import time

LENIENT = os.environ.get("LENIENT") == "1"
PINS = {
    "rich_3469": ["pygments==2.18.0"],                        # test_markdown inline-code colours (task from 2024-08)
    "rich_3782": ["pygments==2.19.2"],                        # test_syntax: 2.20/2.21 colour `iter` as a builtin
    "rich_4070": ["pygments==2.19.2"],
    "requests_6592": ["werkzeug==2.3.8", "flask==2.3.3"],     # werkzeug 3.x raises on the redirect to port -1 inside
    "requests_6629": ["werkzeug==2.3.8", "flask==2.3.3"],     # the test server; flask 3 needs werkzeug 3
}
for _part in filter(None, os.environ.get("LENIENT_PINS", "").split(";")):
    _t, _s = _part.split(":", 1)
    PINS[_t.strip()] = [x.strip() for x in _s.split(",") if x.strip()]

_REPO = pathlib.Path(__file__).resolve().parent.parent
WHEEL_DIRS = [_REPO / "harness" / "wheels", _REPO / "runs" / "lenient_wheels"]
_task = contextvars.ContextVar("swegemma_task", default=None)
_state = {"announced": False, "count": 0}


def pins_text():
    return " · ".join(f"{t}: {' '.join(p)}" for t, p in sorted(PINS.items()))


def write_marker(results_dir):
    """goldcheck/reverify/eval_dev: leave `lenient.txt` in a results dir so compare_runs.py can mark it."""
    if LENIENT:
        d = pathlib.Path(results_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "lenient.txt").write_text(f"LENIENT=1 — analysis only, not the reference\npins: {pins_text()}\n")


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


def _pin(manager, sid):
    task = _task.get()
    pins = PINS.get(task or "")
    if not pins:
        return
    venv = manager._sandboxes[sid]["venv"]
    py = venv / "bin" / "python3"
    links = [a for d in WHEEL_DIRS if d.exists() for a in ("--find-links", str(d))]
    t0 = time.time()
    r = subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-index", *links, "--no-deps", *pins],
                       capture_output=True, text=True, timeout=300)
    how = "pip"
    if r.returncode != 0 and shutil.which("uv"):
        r = subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), "--no-index", *links, "--no-deps", *pins],
                           capture_output=True, text=True, timeout=300)
        how = "uv"
    if r.returncode == 0:
        print(f"lenient: {task} → {' '.join(pins)} ({how} {time.time() - t0:.1f} s)", flush=True)
    else:
        print(f"lenient: {task} → {' '.join(pins)} FAILED — task runs strict: {(r.stderr or r.stdout).strip()[-200:]}", flush=True)


def _wrap_task_context():
    """LENIENT only: make the task id visible to start() via a ContextVar."""
    import swegemma.evaluate as ev

    def wrap(fn):
        @functools.wraps(fn)
        async def w(*a, **kw):
            task = next((x for x in (*a, *kw.values()) if hasattr(x, "instance_id")), None)
            tok = _task.set(getattr(task, "instance_id", None))
            try:
                return await fn(*a, **kw)
            finally:
                _task.reset(tok)
        return w

    ev.verify_task = wrap(ev.verify_task)
    ev.run_agent_sandbox = wrap(ev.run_agent_sandbox)


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
            if LENIENT:
                _pin(self, sid)
        except Exception as e:                     # never break a run over the shim
            print(f"sandbox shim: failed for {sid}: {e}", flush=True)
        return sid

    cls.start = start
    cls._shim_installed = True
    if LENIENT:
        _wrap_task_context()
        print(f"sandbox mode: LENIENT — analysis only, not the reference; pins: {pins_text()}", flush=True)
    return True


install()
