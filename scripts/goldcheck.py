#!/usr/bin/env python3
"""Verify tasks with the dataset's own (gold) patch: does this machine's sandbox accept the reference fix?

    .venv-eval/bin/python scripts/goldcheck.py requests_6592 rich_3535 ...      # → runs/goldcheck/<ids>

A task whose gold patch does not resolve here is unwinnable on this machine (environment), not a model failure
(diag4 did this on a Kaggle Linux CPU kernel: 31/33 valid). Implementation: `swegemma eval --skip-agent-patch` with
`verify_task` wrapped so the empty agent patch becomes the task's `patch` field. Same sandbox, same timeouts.
Imports scripts/sandbox_shim.py first (B-3i): under the subprocess sandbox a src-layout repo is otherwise never imported.
LENIENT=1 → the shim's era pins (pygments/werkzeug) per task, OUT defaults to runs/goldcheck_lenient, a lenient.txt marker
is written; analysis only, never the reference.
EMPTY=1 → the other half of the validity check: score the EMPTY patch (the base commit as is; OUT defaults to
runs/emptycheck). A task whose hidden tests already pass with no change cannot tell agents apart (a free point for any
harmless patch) — scripts/disc_tasks.py drops it. Unlike an agent run's empty patch, nothing short-circuits: the tests run.
"""
import functools
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sandbox_shim   # noqa: E402,F401

sys.argv[0] = "swegemma"
ids = [a for a in sys.argv[1:] if not a.startswith("-")]
if not ids:
    sys.exit(__doc__)
import swegemma.evaluate as ev   # noqa: E402

_orig = ev.verify_task


@functools.wraps(_orig)
async def verify_gold(docker, config, task, snapshot_path, **kw):
    kw["agent_patch"] = task.patch or ""
    kw["agent_error"] = None
    return await _orig(docker, config, task, snapshot_path, **kw)


EMPTY = os.environ.get("EMPTY", "0") == "1"
if not EMPTY:
    ev.verify_task = verify_gold
from swegemma.cli import main   # noqa: E402

out = os.environ.get("OUT", "runs/emptycheck" if EMPTY else "runs/goldcheck_lenient" if sandbox_shim.LENIENT else "runs/goldcheck")
kind = "empty patch" if EMPTY else "gold patch"
sub = os.environ.get("SUBMISSION", "runs/base_lb024")
print(f"goldcheck: {len(ids)} tasks → {out} · scoring the {kind.upper()} (EMPTY={int(EMPTY)}) · sandbox mode "
      f"{'LENIENT (analysis only)' if sandbox_shim.LENIENT else 'strict'}", flush=True)
sandbox_shim.write_marker(out)
tasks_file = os.environ.get("TASKS_FILE", "harness/tasks.jsonl"); snaps = os.environ.get("SNAPSHOTS_DIR", "harness/snapshots")
print(f"goldcheck: tasks file {tasks_file} · snapshots {snaps}", flush=True)
rc = main(["eval", "--tasks", tasks_file, "--snapshots-dir", snaps, "--submission-dir", sub,
           "--results-dir", out, "--sandbox", os.environ.get("SANDBOX", "subprocess"), "--task-ids", *ids,
           "--concurrency", os.environ.get("CONC", "3"), "--display", "quiet", "--skip-agent-patch",
           "--verification-timeout-seconds", os.environ.get("VERIFY_TIMEOUT", "900")])
import json   # noqa: E402
for line in pathlib.Path(out, "task_results.jsonl").read_text().splitlines():
    r = json.loads(line)
    print(f"{r['instance_id']:<16} {kind} → {'RESOLVED' if r['resolved'] else 'NOT resolved'} · {r.get('duration_seconds', 0):.0f} s"
          f" · {(r.get('error') or '')[:80]}")
sys.exit(rc or 0)
