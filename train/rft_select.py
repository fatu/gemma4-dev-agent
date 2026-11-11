#!/usr/bin/env python3
"""Pick E4B's own resolved runs on the tasks that flip between passes (BOX_PLAN B-6: rejection-sampling fine-tune).

    python3 train/rft_select.py runs/rft/recs_e1 runs/rft/recs_e2 … --out runs/rft        # one dir per pass (convert_traj)

A flip task is resolved in some passes and not in others. Tasks resolved in every pass teach nothing new and tasks never
resolved have no positive run, so only flip tasks give data. Per flip task at most --cap resolved runs are kept (those
with the fewest model calls; ties by pass order), so tasks E4B solves often do not crowd out the rare ones. --held flip
tasks (seeded draw) are kept out of training whole: their resolved runs go to <out>/heldout (the training's eval loss).
Writes <out>/train/*.json and <out>/heldout/*.json. Prints aggregates only (no task ids).
"""
import argparse
import collections
import glob
import json
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--held", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--always", choices=["off", "on"], default="off", help="on: one resolved run of every always-resolved task too")
    a = ap.parse_args()
    runs = collections.defaultdict(list)                       # task → [(pass, file, resolved, calls)]
    for p, d in enumerate(a.recs):
        fs = sorted(glob.glob(os.path.join(d, "*.json")))
        k = 0
        for f in fs:
            r = json.load(open(f))
            n = sum(1 for c in r.get("calls") or [] if c.get("target"))
            runs[r["instance_id"]].append((p, f, bool(r.get("resolved")), n)); k += bool(r.get("resolved"))
        print(f"  {d}: {len(fs)} runs · resolved {k}")
    flip = sorted(t for t, v in runs.items() if 0 < sum(x[2] for x in v) < len(v))
    always = sorted(t for t, v in runs.items() if all(x[2] for x in v))
    never = [t for t, v in runs.items() if not any(x[2] for x in v)]
    rng = random.Random(a.seed); held = set(rng.sample(flip, min(a.held, max(len(flip) - 1, 0))))
    print(f"rft_select · passes {len(a.recs)} · tasks {len(runs)}: always {len(always)} · never {len(never)} · flip {len(flip)} · "
          f"cap {a.cap} runs per task · held-out flip tasks {len(held)} (seed {a.seed}) · always-resolved tasks in training: {a.always}")
    for sub in ("train", "heldout"):
        shutil.rmtree(os.path.join(a.out, sub), ignore_errors=True); os.makedirs(os.path.join(a.out, sub))
    cnt = collections.Counter(); calls = collections.Counter(); per = collections.Counter(); avail = 0
    pick = [(t, a.cap) for t in flip] + ([(t, 1) for t in always] if a.always == "on" else [])
    for t, cap in pick:
        good = sorted((x for x in runs[t] if x[2]), key=lambda x: (x[3], x[0]))
        avail += len(good)
        sub = "heldout" if t in held else "train"
        for p, f, _, n in good[:cap]:
            shutil.copy(f, os.path.join(a.out, sub, f"{t}.p{p}.json")); cnt[sub] += 1; calls[sub] += n
        if sub == "train":
            per[min(len(good), cap)] += 1
    print(f"== rft_select: resolved runs available {avail} → train {cnt['train']} runs · {calls['train']} model calls on "
          f"{sum(per.values())} tasks (runs per task → tasks: {dict(sorted(per.items()))}) · heldout {cnt['heldout']} runs · "
          f"{calls['heldout']} calls on {len(held)} tasks → {a.out}/train, {a.out}/heldout")


if __name__ == "__main__":
    main()
