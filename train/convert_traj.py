#!/usr/bin/env python3
"""traj log (scripts/traj_log.py, one JSON line per model call) → call-level training records (train/schema.md).

    python3 train/convert_traj.py --run runs/pod_w7/dev33_base_w7 --source e4b_onpolicy \
        --policy gemma-4-e4b-it-qat-w4a16-ct --out runs/sft/e4b_w7

Replays each task's log: `new` are the messages added since the previous call (a `reset` line carries the whole
history again; a `compaction` line is the summariser's own call and does not move the history pointer), so the
input of call k is the accumulated list. Segment boundaries: a user message after call 1 (nudge), a compaction
summary, a reset. `resolved` comes from the run's task_results.jsonl when present. Prints one line per task and a
total; every number here is the record, there is no other log.
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def load_tasks(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            t = json.loads(line)
            out[t["instance_id"]] = t
    return out


def load_results(run):
    out = {}
    p = os.path.join(run, "task_results.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out[r["instance_id"]] = r
    return out


def patch_sha(run, task):
    for name in (f"{task}.patch", f"{task}.diff"):
        p = os.path.join(run, "patches", name)
        if os.path.exists(p):
            return hashlib.sha256(open(p, "rb").read()).hexdigest()
    return None


def convert(path, source, policy, tasks, results, run):
    rows = [json.loads(l) for l in open(path)]
    task = rows[0]["task"]
    setup = rows[0].get("setup") or {}
    params = setup.get("params") or {}
    extra = params.get("extra_body") or {}
    history, calls = [], []
    segment, pending = 0, []
    for r in rows:
        if r.get("compaction"):
            # the summariser's call: not an agent call; the next agent call sees a rewritten history (logged as reset)
            pending.append("compaction")
            continue
        if r.get("reset"):
            history = []
            pending.append("reset")
        new = r.get("new") or []
        if r["call"] > 1 and any(m.get("role") == "user" for m in new):
            pending.append("nudge")
        history = history + new
        if pending:
            segment += 1
        resp = r.get("resp") or {}
        target = None
        if "err" not in r and resp and "unparsed" not in resp:
            target = {"reasoning": resp.get("reasoning"), "content": resp.get("content"),
                      "tool_calls": resp.get("tool_calls") or [], "finish_reason": resp.get("finish_reason")}
        calls.append({"call": r["call"], "segment": segment, "events_before": pending,
                      "input_messages": history, "target": target, "thought_source": "native" if target else "none",
                      "usage_prompt_tokens": (r.get("usage") or {}).get("prompt"), "err": r.get("err")})
        pending = []
    t = tasks.get(task, {})
    res = results.get(task, {})
    rec = {"instance_id": task, "repo": t.get("repo") or res.get("repo"), "base_commit": t.get("base_commit"),
           "pr": None, "patch_sha": patch_sha(run, task), "source": source,
           "resolved": res.get("resolved") if res else None, "policy_model": policy or setup.get("model"),
           "sampling": {"temperature": params.get("temperature"), "top_p": params.get("top_p"),
                        "max_completion_tokens": params.get("max_completion_tokens") or params.get("max_tokens"),
                        "thinking_token_budget": extra.get("thinking_token_budget"),
                        "enable_thinking": (extra.get("chat_template_kwargs") or {}).get("enable_thinking")},
           "env_fingerprint": None, "tools": setup.get("tool_schemas") or [], "calls": calls}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with traj/<task>.jsonl (+ task_results.jsonl, patches/)")
    ap.add_argument("--source", required=True, choices=["e4b_onpolicy", "31b_teacher", "opus", "d1_qwen", "mined"])
    ap.add_argument("--policy", default=None, help="policy model alias (default: the log's model)")
    ap.add_argument("--tasks", default="harness/tasks.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks, results = load_tasks(a.tasks), load_results(a.run)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.run, "traj", "*.jsonl")))
    files = [f for f in files if not f.endswith(".tools.jsonl")]
    tot = {"tasks": 0, "calls": 0, "targets": 0, "segments": 0, "nudge": 0, "compaction": 0, "reset": 0, "resolved": 0}
    for f in files:
        rec = convert(f, a.source, a.policy, tasks, results, a.run)
        ev = [e for c in rec["calls"] for e in c["events_before"]]
        n_t = sum(1 for c in rec["calls"] if c["target"])
        tot["tasks"] += 1; tot["calls"] += len(rec["calls"]); tot["targets"] += n_t
        tot["segments"] += (rec["calls"][-1]["segment"] + 1) if rec["calls"] else 0
        for k in ("nudge", "compaction", "reset"):
            tot[k] += ev.count(k)
        tot["resolved"] += 1 if rec["resolved"] else 0
        with open(os.path.join(a.out, f"{rec['instance_id']}.json"), "w") as w:
            json.dump(rec, w, ensure_ascii=False)
        print(f"{rec['instance_id']:<16} calls {len(rec['calls']):>3} targets {n_t:>3} segments "
              f"{rec['calls'][-1]['segment'] + 1 if rec['calls'] else 0:>2} nudge {ev.count('nudge')} "
              f"compaction {ev.count('compaction')} reset {ev.count('reset')} resolved {rec['resolved']}")
    print("TOTAL " + " · ".join(f"{k} {v}" for k, v in tot.items()) + f" → {a.out}")
    return 0 if tot["tasks"] else 1


if __name__ == "__main__":
    sys.exit(main())
