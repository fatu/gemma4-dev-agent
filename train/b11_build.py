#!/usr/bin/env python3
"""B-11 step 5: the selected teacher attempts → call-level records (train/schema.md), masked, split by repository.

    python3 train/b11_build.py --sel splits/b11_train625.jsonl \
        --root teach_h2=runs/kaggle_b11_h2stop/teach_h2 --root teach_low_h100=runs/kaggle_b11_low_h100/teach_low_h100 \
        --out runs/sft/b11

--sel: one JSON line per attempt {dir, task, attempt}; dir names a --root (a teacher_gen output directory with
attempts.jsonl and runs/<batch>/traj/<task>.jsonl). Each attempt is replayed with convert_traj.convert, then
mask_errors (tool errors, hidden failures behind exit code 0). Two copies are written:
  A/{train,val}/  thoughts as recorded (in the history and in the targets)
  B/{train,val}/  every thought removed (history and targets) — the no-thinking ablation
Validation: whole repositories drawn with --seed until --val-min..--val-max distinct tasks are held out; a repository
with more than --big tasks is never drawn. No repository is on both sides (asserted). Every count is printed.
"""
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
from collections import Counter

from convert_traj import convert, load_results
from mask_errors import mask_tool_errors, mask_hidden_failures, summary, hidden_summary


def repo_of(task):
    return re.sub(r"-\d+$", "", task)


def strip_thoughts(rec):
    r = copy.deepcopy(rec)
    n = 0
    for c in r["calls"]:
        for m in c["input_messages"]:
            if m.get("role") == "assistant":
                for k in ("reasoning", "reasoning_content"):
                    if m.get(k):
                        m[k] = None; n += 1
        t = c.get("target")
        if t and t.get("reasoning"):
            t["reasoning"] = None
        c["thought_source"] = "none"
    r["sampling"]["enable_thinking"] = False
    return r, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--root", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-min", type=int, default=15)
    ap.add_argument("--val-max", type=int, default=20)
    ap.add_argument("--big", type=int, default=6, help="a repository with more tasks than this is never drawn for validation")
    ap.add_argument("--mask-hidden", choices=["on", "off"], default="on")
    a = ap.parse_args()
    roots = dict(x.split("=", 1) for x in a.root)
    sel = [json.loads(l) for l in open(a.sel)]
    print(f"b11_build · selection {a.sel}: {len(sel)} attempts · roots {roots} · seed {a.seed} · validation {a.val_min}-{a.val_max} tasks, "
          f"repositories over {a.big} tasks stay in training · tool-error masking on · hidden-failure masking {a.mask_hidden}")
    att = {}
    for name, p in roots.items():
        for l in open(os.path.join(p, "attempts.jsonl")):
            r = json.loads(l)
            if r.get("state") == "done":
                att[(name, r["task"], r["attempt"])] = r
    stats, recs, miss = Counter(), [], []
    per_dir = {n: Counter() for n in roots}
    for s in sel:
        r = att.get((s["dir"], s["task"], s["attempt"]))
        if not r or not r.get("resolved"):
            miss.append(s); continue
        run = os.path.join(roots[s["dir"]], "runs", os.path.basename(r["run"]))
        path = os.path.join(run, "traj", s["task"] + ".jsonl")
        if not os.path.exists(path):
            miss.append(s); continue
        rec = convert(path, "glm_teacher", None, {}, load_results(run), run)
        rec["resolved"] = True
        rec["repo"] = rec.get("repo") or repo_of(s["task"])
        rec["teacher"] = {"dir": s["dir"], "batch": os.path.basename(r["run"]), "attempt": s["attempt"], "think": r.get("think")}
        before = sum(1 for c in rec["calls"] if c["target"] and (c["target"].get("tool_calls")))
        mask_tool_errors(rec, stats, "on")
        mask_hidden_failures(rec, stats, a.mask_hidden)
        after = sum(1 for c in rec["calls"] if c["target"] and (c["target"].get("tool_calls")))
        ev = [e for c in rec["calls"] for e in c["events_before"]]
        d = per_dir[s["dir"]]
        d["attempts"] += 1; d["calls"] += len(rec["calls"]); d["tool-call targets before masking"] += before
        d["tool-call targets"] += after; d["compactions"] += ev.count("compaction"); d["resets"] += ev.count("reset")
        d["calls with a thought"] += sum(1 for c in rec["calls"] if c["target"] and c["target"].get("reasoning"))
        recs.append((f"{s['task']}__{s['dir']}_a{s['attempt']}.json", rec))
    if miss:
        print(f"FAIL: {len(miss)} selected attempts not found or not resolved, first: {miss[:3]}"); return 1
    for n, d in per_dir.items():
        print(f"  {n}: " + " · ".join(f"{k} {v}" for k, v in d.items()))
    print("  " + summary(stats, "on")); print("  " + hidden_summary(stats, a.mask_hidden))
    # split by repository
    by_repo = {}
    for _, rec in recs:
        by_repo.setdefault(rec["repo"], set()).add(rec["instance_id"])
    cand = sorted(r for r, t in by_repo.items() if len(t) <= a.big)
    random.Random(a.seed).shuffle(cand)
    val_repos, n_val = [], 0
    for r in cand:
        if n_val >= a.val_min:
            break
        if n_val + len(by_repo[r]) > a.val_max:
            continue
        val_repos.append(r); n_val += len(by_repo[r])
    val_repos = set(val_repos)
    for v in ("A", "B"):
        for side in ("train", "val"):
            p = os.path.join(a.out, v, side)
            shutil.rmtree(p, ignore_errors=True); os.makedirs(p)
    side_n = {"train": Counter(), "val": Counter()}
    stripped = 0
    for name, rec in recs:
        side = "val" if rec["repo"] in val_repos else "train"
        c = side_n[side]
        c["trajectories"] += 1; c["tool-call targets"] += sum(1 for x in rec["calls"] if x["target"] and x["target"].get("tool_calls"))
        json.dump(rec, open(os.path.join(a.out, "A", side, name), "w"), ensure_ascii=False)
        b, n = strip_thoughts(rec); stripped += n
        json.dump(b, open(os.path.join(a.out, "B", side, name), "w"), ensure_ascii=False)
    tr = {rec["repo"] for _, rec in recs if rec["repo"] not in val_repos}
    assert not (tr & val_repos), "a repository is on both sides"
    for side, repos in (("train", tr), ("val", val_repos)):
        tasks = {rec["instance_id"] for _, rec in recs if (rec["repo"] in val_repos) == (side == "val")}
        print(f"  {side}: repositories {len(repos)} · tasks {len(tasks)} · " + " · ".join(f"{k} {v}" for k, v in side_n[side].items()))
    print(f"  repositories on both sides 0 (asserted) · big repositories kept out of the draw {sum(1 for t in by_repo.values() if len(t) > a.big)}"
          f" · B: thoughts removed from {stripped} history messages")
    json.dump({"val_repos": sorted(val_repos), "seed": a.seed}, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"DONE → {a.out}/{{A,B}}/{{train,val}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
