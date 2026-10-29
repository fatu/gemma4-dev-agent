#!/usr/bin/env python3
"""D3 training set: the Opus records (train/convert_atif.py) restricted to the non-dev tasks, env-defect tasks removed.

    python3 train/filter_d3.py --records runs/sft/opus --keep splits/rest96.txt --drop splits/dev33.txt \
        --labels runs/labels_table.txt --out runs/sft/d3_train --ids splits/d3_train.txt

Rules (user decision 2026-10-28: training data = D3 + mined tasks; dev33 = dev set): keep a record when its task is in
--keep and not in --drop, its audit category is not 4 (env_defect: the gold patch itself fails here), and it is the one
record kept per task (strict verdict before lenient, then fewer calls). Prints every count: in → dropped per reason →
kept, by repo, by audit label, lenient, calls, > 28 calls. Records are copied; the id list is public (ids only).
"""
import argparse
import glob
import json
import os
import shutil
import sys

CATS = {"1": "clean", "2": "edge_strict", "3": "assert_over", "4": "env_defect"}


def read_ids(path):
    return {l.strip() for l in open(path) if l.strip()} if path else set()


def read_labels(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 3 and parts[1] in CATS:
            out[parts[0]] = CATS[parts[1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="runs/sft/opus")
    ap.add_argument("--keep", default="splits/rest96.txt")
    ap.add_argument("--drop", default="splits/dev33.txt")
    ap.add_argument("--labels", default="runs/labels_table.txt")
    ap.add_argument("--out", default="runs/sft/d3_train")
    ap.add_argument("--ids", default="splits/d3_train.txt")
    a = ap.parse_args()
    keep, drop, labels = read_ids(a.keep), read_ids(a.drop), read_labels(a.labels)
    print(f"filter_d3 · records {a.records} · keep {a.keep} ({len(keep)}) · drop {a.drop} ({len(drop)}) · labels {a.labels} ({len(labels)})")
    recs = [json.load(open(f)) | {"_file": f} for f in sorted(glob.glob(os.path.join(a.records, "*.json")))]
    why = {"in_drop_split": 0, "not_in_keep_split": 0, "env_defect": 0, "no_label": 0, "duplicate_task": 0}
    by_task = {}
    for r in recs:
        t = r["instance_id"]
        if t in drop:
            why["in_drop_split"] += 1; continue
        if t not in keep:
            why["not_in_keep_split"] += 1; continue
        lab = labels.get(t)
        if lab is None:
            why["no_label"] += 1; continue
        if lab == "env_defect":
            why["env_defect"] += 1; continue
        by_task.setdefault(t, []).append(r)
    kept = []
    for t, rs in sorted(by_task.items()):
        rs.sort(key=lambda r: (bool(r.get("lenient")), len(r["calls"])))
        kept.append(rs[0]); why["duplicate_task"] += len(rs) - 1
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "*.json")):
        os.remove(f)
    for r in kept:
        shutil.copy(r["_file"], os.path.join(a.out, os.path.basename(r["_file"])))
    with open(a.ids, "w") as w:
        w.write("".join(r["instance_id"] + "\n" for r in kept))
    repo, lab, lenient, calls, over = {}, {}, 0, 0, 0
    for r in kept:
        repo[r["repo"]] = repo.get(r["repo"], 0) + 1
        l = labels[r["instance_id"]]; lab[l] = lab.get(l, 0) + 1
        lenient += bool(r.get("lenient")); calls += len(r["calls"]); over += len(r["calls"]) > 28
    print(f"  in {len(recs)} → dropped " + " · ".join(f"{k} {v}" for k, v in why.items()) + f" → kept {len(kept)} tasks")
    print("  by repo " + " · ".join(f"{k} {v}" for k, v in sorted(repo.items())))
    print("  by audit label " + " · ".join(f"{k} {v}" for k, v in sorted(lab.items())))
    print(f"  lenient verdicts {lenient} · calls {calls} (mean {calls / max(len(kept), 1):.1f}) · records > 28 calls {over}")
    print(f"  → {a.out}/ ({len(kept)} records) · ids {a.ids}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
