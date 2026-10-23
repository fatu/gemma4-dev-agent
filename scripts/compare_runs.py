#!/usr/bin/env python3
"""One-screen comparison of swegemma result dirs (summary.json + task_results.jsonl + ctx_overflow.txt).

    python scripts/compare_runs.py runs/dev33_sample runs/dev33_out8k     # 📷

Table 1: one row per run — resolved, errors, ctx✗ = vLLM context-overflow CALLS (incl. retries, not tasks), per-repo.
Table 2: tasks whose outcome differs between runs (on 33 tasks, a 1-2 task gap is noise — look at which tasks flip).
"""
import json
import pathlib
import sys

REPOS = [("fastapi/fastapi", "fastapi"), ("Textualize/rich", "rich"), ("psf/requests", "requests")]


def load(d: pathlib.Path):
    s = json.loads((d / "summary.json").read_text())
    tr = d / "task_results.jsonl"
    tasks = {r["instance_id"]: r for r in map(json.loads, tr.open())} if tr.exists() else {}
    ov = d / "ctx_overflow.txt"
    return s, tasks, ov.read_text().strip() if ov.exists() else "?"


def main():
    runs = [pathlib.Path(p) for p in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    for r in [r for r in runs if not (r / "summary.json").exists()]:
        print(f"skip {r}: no summary.json (run unfinished or failed)")
    runs = [r for r in runs if (r / "summary.json").exists()]
    if not runs:
        sys.exit(1)
    data = {r: load(r) for r in runs}

    print(f"{'run':28} {'resolved':>9} {'rate':>5} {'err':>4} {'ctx✗':>5}  " + "  ".join(f"{n:>8}" for _, n in REPOS))
    for r, (s, _, ov) in data.items():
        by = s.get("by_repo", {})
        cells = "  ".join(f"{by[k]['resolved']}/{by[k]['total']:<2}".rjust(8) if k in by else f"{'-':>8}" for k, _ in REPOS)
        name = r.name + (" L" if (r / "lenient.txt").exists() else "")      # L = LENIENT sandbox (era pins; analysis only)
        print(f"{name:28} {s['resolved']:>4}/{s['total_tasks']:<4} {s['resolution_rate']:>5.2f} {s.get('errors', '?'):>4} {ov:>5}  {cells}")

    if len(runs) > 1:
        ids = sorted(set().union(*(t.keys() for _, t, _ in data.values())))
        mark = lambda t, i: "-" if i not in t else ("✓" if t[i].get("resolved") else ("E" if t[i].get("error") else "✗"))
        flips = [i for i in ids if len({mark(t, i) for _, t, _ in data.values()}) > 1]
        print(f"\ntasks that differ ({len(flips)}; ✓ resolved · ✗ failed · E error):")
        for i in flips:
            print(f"  {i:16} " + " ".join(f"{mark(t, i):>3}" for _, t, _ in data.values()))
        always = [i for i in ids if all(t.get(i, {}).get("resolved") for _, t, _ in data.values())]
        never = [i for i in ids if not any(t.get(i, {}).get("resolved") for _, t, _ in data.values())]
        print(f"resolved in every run: {len(always)}: {' '.join(always)}")      # ids so a ceiling run can be split against them
        print(f"resolved in no run:    {len(never)}: {' '.join(never)}")
        print(f"resolved in ≥1 run:    {len(ids) - len(never)}")


if __name__ == "__main__":
    main()
