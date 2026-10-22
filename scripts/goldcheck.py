#!/usr/bin/env python3
"""Verify tasks with the dataset's own (gold) patch: does this machine's sandbox accept the reference fix?

    .venv-eval/bin/python scripts/goldcheck.py requests_6592 rich_3535 ...      # → runs/goldcheck/<ids>

A task whose gold patch does not resolve here is unwinnable on this machine (environment), not a model failure
(diag4 did this on a Kaggle Linux CPU kernel: 31/33 valid). Implementation: `swegemma eval --skip-agent-patch` with
`verify_task` wrapped so the empty agent patch becomes the task's `patch` field. Same sandbox, same timeouts.
"""
import functools
import os
import pathlib
import sys

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


ev.verify_task = verify_gold
from swegemma.cli import main   # noqa: E402

out = os.environ.get("OUT", "runs/goldcheck")
sub = os.environ.get("SUBMISSION", "runs/base_lb024")
rc = main(["eval", "--tasks", "harness/tasks.jsonl", "--snapshots-dir", "harness/snapshots", "--submission-dir", sub,
           "--results-dir", out, "--sandbox", os.environ.get("SANDBOX", "subprocess"), "--task-ids", *ids,
           "--concurrency", os.environ.get("CONC", "3"), "--display", "quiet", "--skip-agent-patch",
           "--verification-timeout-seconds", os.environ.get("VERIFY_TIMEOUT", "900")])
import json   # noqa: E402
for line in pathlib.Path(out, "task_results.jsonl").read_text().splitlines():
    r = json.loads(line)
    print(f"{r['instance_id']:16} gold patch → {'RESOLVED' if r['resolved'] else 'NOT resolved'} · {r.get('duration_seconds', 0):.0f} s"
          f" · {(r.get('error') or '')[:80]}")
sys.exit(rc or 0)
