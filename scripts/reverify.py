#!/usr/bin/env python3
"""Re-score a finished run's saved patches without the model: verification only (BOX_PLAN.md B-3i).

    .venv-eval/bin/python scripts/reverify.py runs/dev33_base_ceilx [runs/dev33_base_k1 ...]   # → runs/<name>_rv/

For each run: every task with a non-empty `patches/<id>.patch` goes through the harness's own `verify_task` again
(`swegemma eval --skip-agent-patch` with the empty patch replaced by the saved one; same sandbox, same timeouts), into
`<run>_rv/` with the usual task_results.jsonl / summary.json / test_outputs, so `compare_runs.py <run> <run>_rv` works.
Tasks without a patch are copied over as unresolved (they were unresolved before: nothing to re-verify). Ends with a
per-task old → new line for every change and the totals. Env: SANDBOX (subprocess), CONC (3), VERIFY_TIMEOUT (900),
SUBMISSION (runs/base_lb024). Imports scripts/sandbox_shim.py first, so src-layout repos are really tested.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sandbox_shim   # noqa: E402,F401

sys.argv[0] = "swegemma"
runs = [pathlib.Path(a) for a in sys.argv[1:] if not a.startswith("-")]
if not runs:
    sys.exit(__doc__)
import swegemma.evaluate as ev   # noqa: E402
from swegemma.cli import main    # noqa: E402

PATCHES: dict[str, str] = {}
_orig = ev.verify_task


async def verify_saved(docker, config, task, snapshot_path, **kw):
    kw["agent_patch"] = PATCHES.get(task.instance_id, "")
    kw["agent_error"] = None
    return await _orig(docker, config, task, snapshot_path, **kw)


ev.verify_task = verify_saved
sub = os.environ.get("SUBMISSION", "runs/base_lb024")
rc_all = 0
for run in runs:
    old = {}
    for line in (run / "task_results.jsonl").read_text().splitlines():
        r = json.loads(line); old[r["instance_id"]] = r
    PATCHES.clear()
    for p in sorted((run / "patches").glob("*.patch")):
        txt = p.read_text()
        if txt.strip():
            PATCHES[p.stem] = txt
    out = run.parent / (run.name + "_rv")
    ids = sorted(PATCHES)
    print(f"{run}: {len(old)} tasks, {len(ids)} with a saved patch → {out} (sandbox {os.environ.get('SANDBOX', 'subprocess')}, "
          f"CONC {os.environ.get('CONC', '3')}, verify timeout {os.environ.get('VERIFY_TIMEOUT', '900')} s)", flush=True)
    rc = main(["eval", "--tasks", "harness/tasks.jsonl", "--snapshots-dir", "harness/snapshots", "--submission-dir", sub,
               "--results-dir", str(out), "--sandbox", os.environ.get("SANDBOX", "subprocess"), "--task-ids", *ids,
               "--concurrency", os.environ.get("CONC", "3"), "--display", "quiet", "--skip-agent-patch",
               "--verification-timeout-seconds", os.environ.get("VERIFY_TIMEOUT", "900")]) or 0
    rc_all = rc_all or rc
    new = {}
    for line in (out / "task_results.jsonl").read_text().splitlines():
        r = json.loads(line); new[r["instance_id"]] = r
    with (out / "task_results.jsonl").open("a") as f:          # patch-less tasks: carried over, still unresolved
        for t, r in old.items():
            if t not in new:
                r = dict(r, resolved=False, error=(r.get("error") or "no saved patch (reverify: carried over)"))
                new[t] = r; f.write(json.dumps(r) + "\n")
    s = json.loads((out / "summary.json").read_text())
    s["total_tasks"] = len(new); s["resolved"] = sum(1 for r in new.values() if r.get("resolved"))
    s["resolution_rate"] = round(s["resolved"] / max(len(new), 1), 4)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    flips = [(t, old[t].get("resolved"), new[t].get("resolved")) for t in sorted(old) if bool(old[t].get("resolved")) != bool(new[t].get("resolved"))]
    print(f"== {run.name}: resolved {sum(1 for r in old.values() if r.get('resolved'))}/{len(old)} → "
          f"{s['resolved']}/{len(new)} after re-verification; {len(flips)} changed")
    for t, a, b in flips:
        print(f"   {t:16} {'✓' if a else '✗'} → {'✓' if b else '✗'}   {(new[t].get('error') or '')[:70]}")
sys.exit(rc_all)
