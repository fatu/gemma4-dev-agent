#!/usr/bin/env python3
"""Failure patterns, step S4 of the failure study (notes/16): reads runs/fail_atlas/<run>.tsv (S1) and
runs/fail_judge/<run>/*.json (S3) for several runs and prints the comparison tables. No API calls.

    python3 scripts/fail_patterns.py e2b=dev33_base_all129_e2b e4b=dev33_base_w7 31b=dev33_base_all129_31b
    EXAMPLES=3 python3 scripts/fail_patterns.py …        # + up to 3 task ids per code with the judge's fix hint

Each argument is name=<run dir name> (the atlas / judge file names). Valid failures = outcome ≠ ✓ and env_defect 0.
Tables: 1 primary cause per run (count, % of judged) · 2 would_fix per run · 3 mechanics from the atlas, per valid
failure (never in a gold file, clobbered, edit failures per edit attempt, cut-off replies, compactions, tested) ·
4 the first run's causes split by whether the reference run (last argument) solved the task · 5 the first run's
primary cause by repo and by difficulty. Task ids are printed only with EXAMPLES (private notes; keep out of RESULTS.md).
"""
import csv
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODES = ["L1", "L2", "R1", "R2", "R3", "R4", "T1", "T2", "I1", "B1", "E1", "P3", "V1", "C1", "X"]


def load(run):
    rows = {r["task"]: r for r in csv.DictReader(open(ROOT / "runs" / "fail_atlas" / f"{run}.tsv"), delimiter="\t")}
    judged = {p.stem: json.loads(p.read_text()) for p in (ROOT / "runs" / "fail_judge" / run).glob("*.json")}
    valid = {i: r for i, r in rows.items() if r["outcome"] != "✓" and r["env_defect"] == "0"}
    return rows, valid, {i: j for i, j in judged.items() if i in valid}


def pct(n, d):
    return f"{n:3} {100 * n / d:3.0f}%" if d else f"{n:3}   – "


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    if not pairs:
        sys.exit(__doc__)
    data = {n: load(r) for n, r in pairs}
    names = [n for n, _ in pairs]
    print("fail_patterns: " + " · ".join(f"{n}={r} ({len(data[n][1])} valid failures, {len(data[n][2])} judged)"
                                        for n, r in pairs))

    print("\n1 · primary cause (count, % of judged valid failures)")
    print(f"{'code':5}" + "".join(f"{n:>12}" for n in names))
    cnt = {n: Counter(j.get("primary") for j in data[n][2].values()) for n in names}
    for c in CODES + sorted({k for n in names for k in cnt[n]} - set(CODES)):
        if any(cnt[n][c] for n in names):
            print(f"{c:5}" + "".join(f"{pct(cnt[n][c], len(data[n][2])):>12}" for n in names))
    sec = {n: Counter(s for j in data[n][2].values() for s in (j.get("secondary") or [])) for n in names}
    print("  secondary (any mention): " + " | ".join(
        f"{n}: " + " ".join(f"{k} {v}" for k, v in sec[n].most_common(6)) for n in names))

    print("\n2 · would_fix (cheapest lever)")
    for n in names:
        w = Counter(j.get("would_fix") for j in data[n][2].values())
        print(f"  {n:6} " + " · ".join(f"{k} {pct(v, len(data[n][2])).strip()}" for k, v in w.most_common()))

    print("\n3 · mechanics per valid failure (atlas)")
    print(f"{'':28}" + "".join(f"{n:>12}" for n in names))

    def num(x):
        try:
            return float(x)
        except ValueError:
            return 0.0
    rows = [("never touched a gold file", lambda v: sum(r["files_hit"] in ("none", "-") for r in v), True),
            ("source file clobbered", lambda v: sum(num(r.get("clobber", 0)) > 0 for r in v), True),
            ("edit failures / attempts", None, False),
            ("cut-off replies (any)", lambda v: sum(num(r["len_cuts"]) > 0 for r in v), True),
            ("compactions per task", lambda v: sum(num(r["compactions"]) for r in v) / max(len(v), 1), False),
            ("ran tests", lambda v: sum(r["tested"] == "1" for r in v), True),
            ("budget stop (T / C)", lambda v: sum(r["outcome"] in ("T", "C") for r in v), True),
            ("no patch (∅)", lambda v: sum(r["outcome"] == "∅" for r in v), True)]
    for label, f, share in rows:
        cells = []
        for n in names:
            v = list(data[n][1].values())
            if f is None:
                bad = sum(num(r["edit_mismatch"]) + num(r["edit_missing"]) for r in v)
                tot = bad + sum(num(r["edit_ok"]) for r in v)
                cells.append(f"{100 * bad / tot:5.0f}% of {int(tot)}" if tot else "–")
            elif share:
                cells.append(pct(f(v), len(v)))
            else:
                cells.append(f"{f(v):6.1f}")
        print(f"  {label:26}" + "".join(f"{c:>12}" for c in cells))

    if len(names) >= 2:
        a, ref = names[0], names[-1]
        print(f"\n4 · {a} causes, split by whether {ref} solved the task")
        rrows = data[ref][0]
        split = defaultdict(Counter)
        for i, j in data[a][2].items():
            o = rrows.get(i, {}).get("outcome")
            split["solved by " + ref if o == "✓" else (f"{ref} failed too" if o else f"{ref} not run")][j.get("primary")] += 1
        for k, c in split.items():
            print(f"  {k:22} ({sum(c.values()):3}): " + " · ".join(f"{x} {y}" for x, y in c.most_common()))
        both = [(j.get("primary"), data[ref][2][i].get("primary")) for i, j in data[a][2].items() if i in data[ref][2]]
        if both:
            print(f"  both failed and judged ({len(both)}): {a}→{ref} " +
                  " · ".join(f"{x}→{y} {v}" for (x, y), v in Counter(both).most_common(8)))

    a = names[0]
    print(f"\n5 · {a} primary cause by repo and by difficulty")
    for key in ("repo", "difficulty"):
        g = defaultdict(Counter)
        for i, j in data[a][2].items():
            g[data[a][1][i][key]][j.get("primary")] += 1
        for k in sorted(g):
            print(f"  {key} {k:10} ({sum(g[k].values()):3}): " + " · ".join(f"{x} {y}" for x, y in g[k].most_common(5)))

    ex = int(os.environ.get("EXAMPLES", "0"))
    if ex:
        print(f"\nexamples ({a}, up to {ex} per code; private)")
        by = defaultdict(list)
        for i, j in sorted(data[a][2].items()):
            by[j.get("primary")].append((i, j))
        for c in CODES:
            for i, j in by.get(c, [])[:ex]:
                print(f"  {c:3} {i:16} call {str(j.get('first_wrong_call')):4} {j.get('would_fix', ''):8} "
                      f"{(j.get('fix_hint') or '')[:110]}")


if __name__ == "__main__":
    main()
