#!/usr/bin/env python3
"""Does nothink stop the model thinking, or only drop old thoughts from the history? And what does it cost in time?

    python scripts/gen_stats.py runs/dev33_out8k runs/dev33_nothink runs/dev33_think1k     # 📷
    LIMIT=600 python scripts/gen_stats.py runs/dev33_*                                     # + share of tasks over 600 s

Table 1, over the agent's steps:
  gen tok/step        completion_tokens per model call (thinking + tool call)
  thought chars/step  length of the thought text saved in the trace (`message`)
Reading: nothink ≈ out8k gen tok/step → it still thinks, and Finding #5 (b 1.00 → 0.12) is only the history dropping it;
nothink much lower → it also thinks less. think1k ≈ out8k → the 1024 budget was not enforced.
Table 2, wall-clock per task (task_results.jsonl `elapsed_seconds`, else first → last trace step timestamp). The scorer
has a per-task time budget that the local CLI does not enforce (Finding #6: nothink generates 2.6x more per step), so
check this before submitting an arm. Local times are on our GPUs, not the scorer's 4xL4: compare arms, not absolutes.
"""
import datetime
import glob
import json
import os
import pathlib
import statistics
import sys


def name(r):
    return r.rstrip("/").split("/")[-1]


def ts(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return datetime.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def task_secs(r):
    """seconds per task: task_results field if present, else the span of the trace's step timestamps"""
    out, src = {}, set()
    tr = pathlib.Path(r, "task_results.jsonl")
    for row in map(json.loads, tr.open()) if tr.exists() else []:
        for k in ("elapsed_seconds", "duration_seconds", "wall_time_seconds", "elapsed", "duration"):
            if isinstance(row.get(k), (int, float)):
                out[row["instance_id"]] = float(row[k]); src.add(k); break
    for f in glob.glob(f"{r}/traces/trace_*.json"):
        iid = pathlib.Path(f).stem[len("trace_"):]
        if iid in out:
            continue
        t = [ts(s.get("timestamp")) for s in json.load(open(f)).get("steps", []) if s.get("timestamp") is not None]
        t = [x for x in t if x is not None]
        if len(t) > 1:
            out[iid] = max(t) - min(t); src.add("trace span")
    return out, "/".join(sorted(src)) or "-"


runs = sys.argv[1:] or sys.exit(__doc__)
print(f"{'run':28} {'steps':>6} {'gen tok/step':>13} {'thought chars/step':>19}")
for r in runs:
    S = [s for f in glob.glob(f"{r}/traces/*.json") for s in json.load(open(f)).get("steps", [])
         if s.get("source") == "agent" and (s.get("metrics") or {}).get("completion_tokens") is not None]
    if not S:
        print(f"{name(r):28} no agent steps with metrics"); continue
    gen = sum(s["metrics"]["completion_tokens"] for s in S) / len(S)
    th = sum(len(str(s.get("message") or "")) for s in S) / len(S)
    print(f"{name(r):28} {len(S):>6} {gen:>13.0f} {th:>19.0f}")

limit = float(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
print(f"\n{'run':28} {'tasks':>5} {'mean s':>7} {'median':>7} {'p90':>6} {'max':>6}" + (f" {'>'+str(int(limit))+'s':>7}" if limit else "") + "  source")
for r in runs:
    secs, src = task_secs(r)
    v = sorted(secs.values())
    if not v:
        print(f"{name(r):28} no timing found (run --probe in buckets.py and photo it)"); continue
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    over = f" {sum(x > limit for x in v):>4}/{len(v):<2}" if limit else ""
    print(f"{name(r):28} {len(v):>5} {statistics.mean(v):>7.0f} {statistics.median(v):>7.0f} {p90:>6.0f} {v[-1]:>6.0f}{over}  {src}")
