#!/usr/bin/env python3
"""The local tasks that can tell two E4B arms apart → splits/disc_e4b.txt, derived from runs we already have.

    python3 scripts/disc_tasks.py            # laptop: class table + diff against the committed split (no write)
    python3 scripts/disc_tasks.py --write    # rewrite splits/disc_e4b.txt (only on purpose: arms are comparable
                                             # only on the same version of the list — rerun the base arm after a change)
    # then on the box / a pod:  SPLIT=splits/disc_e4b.txt TAG=s1 bash scripts/eval_dev.sh <arm>   (≥ 3 seeds per arm)

A task carries signal for an A/B only if its outcome can change with the arm. Classes, in this order:
  X_gold        the gold patch fails here (runs/gc_all, laptop strict sandbox; plus BOX_GOLD_FAIL) → nobody can win it
  X_empty       the empty patch passes (runs/emptycheck*/, an agent run resolved with an empty patch and no error, plus
                BOX_EMPTY_PASS) → a free point for any harmless patch (TechNaom's audit, forum 756400, found 5 such)
  floor         valid, but no run of any model ever resolved it (Opus ceilings included)
  headroom_opus E4B never resolved it, Opus did, our one 31B run did not
  headroom_31b  E4B never resolved it, 31B did — where a better E4B arm would gain first
  e4b_some      E4B (any arm: base, LoRA, course …) resolved it in some runs but not all
  e4b_always    E4B resolved it in every run (≥ 3 runs) — kept as a regression guard
disc_e4b = e4b_some + headroom_31b + e4b_always. A task that is 0 (or 1) under every arm adds no noise to a paired comparison,
only time, so the gain is runtime: the list costs about a third of an all-129 E4B run (printed), i.e. three seeds per arm
for the price of one run. Per-task estimates come from 1–8 runs, so the classes are rough; the table prints k/n per family.
Inputs are local, gitignored run dirs (laptop only); a missing dir is printed, never silently skipped. Box-only facts arrive
by photo and go into BOX_GOLD_FAIL / BOX_EMPTY_PASS below.
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "splits/disc_e4b.txt"
GOLD = ROOT / "runs/gc_all/all.jsonl"
EMPTY_GLOB = "runs/emptycheck*/task_results.jsonl"   # goldcheck.py EMPTY=1
BOX_GOLD_FAIL: set[str] = set()                      # from the box photo of `EMPTY=0 goldcheck` (none yet)
BOX_EMPTY_PASS: set[str] = set()                     # from the box photo of `EMPTY=1 goldcheck` (none yet)
TIME_RUN = "runs/pod_all129_e4b/runs/dev33_base_all129_e4b"   # per-task durations for the time share
RUNS = [  # (result dir, family) — every local run with a task_results.jsonl that used the scorer's verification
    ("runs/dev33_base_ceilx", "opus"), ("runs/dev33_base_ceilx_rv", "opus"), ("runs/dev33_base_ceilsc", "opus"),
    ("runs/dev33_base_ceil96", "opus"),
    ("runs/pod_all129/runs/dev33_base_all129_31b", "31b"),
    ("runs/pod_all129_e4b/runs/dev33_base_all129_e4b", "e4b"),
    ("runs/pod_borisv6/runs/dev33_base_all129_borisv6", "e4b"),
    ("runs/pod_borisv6noad/runs/dev33_base_all129_borisv6noad", "e4b"),
    ("runs/pod_w7/dev33_base_w7", "e4b"), ("runs/pod_w7/dev33_course_w7", "e4b"), ("runs/pod_w7/dev33_coursefull_w7", "e4b"),
    ("runs/pod_sft/runs/dev33_base_s1", "e4b"), ("runs/pod_sft/runs/dev33_base_m2s1", "e4b"),
    ("runs/pod_all129_e2b/runs/dev33_base_all129_e2b", "e2b"),
    ("runs/pod_boris/dev33_base_bbases1", "e2b"), ("runs/pod_boris/dev33_base_bloras1", "e2b"),
]
FAMILIES = ["opus", "31b", "e4b", "e2b"]
ORDER = ["X_gold", "X_empty", "floor", "headroom_opus", "headroom_31b", "e4b_some", "e4b_always"]
KEEP = ["headroom_31b", "e4b_some", "e4b_always"]


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    gold = {r["instance_id"]: r["resolved"] for r in rows(GOLD)}
    print(f"disc_tasks: {len(gold)} tasks · gold {GOLD.relative_to(ROOT)} ({sum(gold.values())} resolve)"
          f" + box gold fails {len(BOX_GOLD_FAIL)} · box empty passes {len(BOX_EMPTY_PASS)}")
    empty_pass, empty_seen = set(BOX_EMPTY_PASS), set()
    for f in sorted(ROOT.glob(EMPTY_GLOB)):
        rs = rows(f)
        empty_seen |= {r["instance_id"] for r in rs}
        empty_pass |= {r["instance_id"] for r in rs if r["resolved"]}
        print(f"  empty check {f.parent.relative_to(ROOT)}: {len(rs)} tasks, {sum(r['resolved'] for r in rs)} pass empty")
    kn = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for d, fam in RUNS:
        p = ROOT / d / "task_results.jsonl"
        if not p.exists():
            print(f"  MISSING {d} — skipped"); continue
        rs = rows(p)
        for r in rs:
            t = r["instance_id"]; kn[t][fam][0] += bool(r["resolved"]); kn[t][fam][1] += 1
            if r["resolved"] and not r.get("agent_patch_size") and not r.get("error"):
                empty_pass.add(t)
                print(f"  {t}: resolved with an empty patch in {d}")
        print(f"  run {d} [{fam}]: {sum(r['resolved'] for r in rs)}/{len(rs)}")
    print(f"  empty patch checked on {len(empty_seen)} of {len(gold)} tasks; the rest only through agent runs")

    cls = {}
    for t in gold:
        f = kn[t]; anyres = sum(f[m][0] for m in f)
        if not gold[t] or t in BOX_GOLD_FAIL: c = "X_gold"
        elif t in empty_pass: c = "X_empty"
        elif anyres == 0: c = "floor"
        elif f["e4b"][0] == 0: c = "headroom_31b" if f["31b"][0] else "headroom_opus"
        elif f["e4b"][0] == f["e4b"][1] and f["e4b"][1] >= 3: c = "e4b_always"
        else: c = "e4b_some"
        cls[t] = c

    dur = {r["instance_id"]: r.get("duration_seconds") or 0 for r in rows(ROOT / TIME_RUN / "task_results.jsonl")}
    total = sum(dur.values()) or 1
    dev33 = set((ROOT / "splits/dev33.txt").read_text().split())
    for c in ORDER:
        ts = sorted(t for t in cls if cls[t] == c)
        share = 100 * sum(dur.get(t, 0) for t in ts) / total
        print(f"\n== {c}: {len(ts)} tasks (dev33 {len(dev33 & set(ts))}) · {share:.0f}% of an all-129 E4B run's task time"
              f"{'  → disc_e4b' if c in KEEP else ''}")
        for t in ts:
            print(f"  {t:<15} " + "  ".join(f"{m} {kn[t][m][0]}/{kn[t][m][1]}" if kn[t][m][1] else f"{m} -" for m in FAMILIES))
    keep = sorted(t for t in cls if cls[t] in KEEP)
    share = 100 * sum(dur.get(t, 0) for t in keep) / total
    print(f"\ndisc_e4b: {len(keep)} tasks ({', '.join(f'{c} {sum(cls[t] == c for t in keep)}' for c in KEEP)})"
          f" · {share:.0f}% of an all-129 E4B run's task time")
    old = OUT.read_text().split() if OUT.exists() else []
    if old:
        add, drop = sorted(set(keep) - set(old)), sorted(set(old) - set(keep))
        print(f"vs committed {OUT.relative_to(ROOT)} ({len(old)}): {'same' if not add and not drop else f'+{add} -{drop}'}")
    if "--write" in sys.argv:
        OUT.write_text("\n".join(keep) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)} — rerun the base arm on it before comparing new arms")


if __name__ == "__main__":
    main()
