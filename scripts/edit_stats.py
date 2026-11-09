#!/usr/bin/env python3
"""edit_file outcomes of one or more runs, from their traj logs — the edit-level readout of an A/B (B-3q): the resolved
count of 51 tasks has a noise of about ± 3.5 tasks, these rates move first.
  python3 scripts/edit_stats.py runs/<run> [runs/<run> …]      # several runs: one block each, then all of them pooled
  python3 scripts/edit_stats.py --pooled runs/<run> …          # only the pooled block (an A/B arm's passes)
Per run (aggregates only, no task ids):
  edit calls            ok / not found / several matches / other, and failed edits per task
  first tries           outcome of the first edit on a file, or the first after a success on it
  after a failed edit   the next edit on the same file: sent straight away / after reading the file again / after
                        another command; how often it succeeds; how often it is the identical old_string again
  loops                 runs of ≥ 2 failed edits in a row on one file: count, tasks with one, the longest
Reads <run>/traj/<task>.jsonl (the harness-side log: `new` messages per call) and <run>/task_results.jsonl if there.
"""
import glob
import json
import os
import sys
from collections import Counter


def load(run):
    res = {}
    p = os.path.join(run, "task_results.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            res[r["instance_id"]] = bool(r.get("resolved"))
    tasks = {}
    for f in sorted(glob.glob(os.path.join(run, "traj", "*.jsonl"))):
        if f.endswith(".tools.jsonl"):
            continue
        pend, ev = {}, []
        for l in open(f):
            for m in json.loads(l).get("new") or []:
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        try:
                            a = json.loads(tc["function"]["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            a = {}
                        pend[tc["id"]] = (tc["function"]["name"], a if isinstance(a, dict) else {})
                elif m.get("role") == "tool" and m.get("tool_call_id") in pend:
                    name, a = pend.pop(m["tool_call_id"])
                    out = None
                    if name == "edit_file":
                        try:
                            r = json.loads(m["content"])
                        except (json.JSONDecodeError, TypeError):
                            r = {}
                        msg = (r.get("error_message") or "") if isinstance(r, dict) else ""
                        out = "ok" if isinstance(r, dict) and r.get("status") == "ok" else \
                            "several" if "occurs" in msg else "not_found" if "not found" in msg else "other"
                    ev.append((name, a.get("filepath"), out, a.get("old_string"), a.get("command") or ""))
        tasks[os.path.basename(f)[:-6]] = ev
    return res, tasks


def pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "–"


def report(run, res, tasks):
    edits, first, nxt, how_ok = Counter(), Counter(), Counter(), Counter()
    failed_per_task, loops, loop_tasks, longest, same = [], 0, 0, 0, 0
    for ev in tasks.values():
        E = [(i, e) for i, e in enumerate(ev) if e[0] == "edit_file" and e[2]]
        last, run_len, had_loop = {}, Counter(), False
        failed_per_task.append(sum(e[2] != "ok" for _, e in E))
        for k, (i, e) in enumerate(E):
            _, p, out, old, _ = e
            edits[out] += 1
            if last.get(p, "ok") == "ok":
                first[out] += 1
            if out != "ok":
                run_len[p] += 1
                if run_len[p] == 2:
                    loops += 1
                    had_loop = True
                longest = max(longest, run_len[p])
                j = next((x for x in E[k + 1:] if x[1][1] == p), None)
                if j is None:
                    nxt["no further edit"] += 1
                else:
                    between = ev[i + 1:j[0]]
                    how = "read again" if any(b[0] == "read_file" and b[1] == p for b in between) else \
                        "another command" if between else "straight away"
                    nxt[how] += 1
                    how_ok[how] += j[1][2] == "ok"
                    same += j[1][3] == old
            else:
                run_len[p] = 0
            last[p] = out
        loop_tasks += had_loop
    n, t = sum(edits.values()), len(tasks)
    fails = n - edits["ok"]
    print(f"== {run}: tasks {t}" + (f" · resolved {sum(res.values())}" if res else "") + f" · edit calls {n}")
    print("   edit calls: " + " · ".join(f"{k} {edits[k]} ({pct(edits[k], n)})" for k in ("ok", "not_found", "several", "other"))
          + f" · failed edits per task {fails / t:.2f}" if t else "   no tasks")
    nf = sum(first.values())
    print("   first tries on a file: " + " · ".join(f"{k} {first[k]} ({pct(first[k], nf)})" for k in ("ok", "not_found", "several", "other")))
    retried = sum(v for k, v in nxt.items() if k != "no further edit")
    print("   after a failed edit: " + " · ".join(f"{k} {nxt[k]} (next ok {pct(how_ok[k], nxt[k])})" for k in
                                                   ("straight away", "read again", "another command"))
          + f" · no further edit {nxt['no further edit']} · identical old_string again {same} ({pct(same, retried)} of retries)")
    print(f"   loops (≥ 2 failed edits in a row on a file): {loops} in {loop_tasks} tasks ({pct(loop_tasks, t)}) · longest {longest}")


if __name__ == "__main__":
    runs = [r.rstrip("/") for r in sys.argv[1:] if r != "--pooled"]
    if not runs:
        sys.exit(__doc__)
    all_res, all_tasks = {}, {}
    for r in runs:
        res, tasks = load(r)
        if "--pooled" not in sys.argv:
            report(r, res, tasks)
        all_res.update({f"{r}/{k}": v for k, v in res.items()})
        all_tasks.update({f"{r}/{k}": v for k, v in tasks.items()})
    if len(runs) > 1 or "--pooled" in sys.argv:
        report(f"pooled over {len(runs)} runs", all_res, all_tasks)
