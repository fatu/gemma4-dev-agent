#!/usr/bin/env python3
"""W2 step 1: a fixed 33-task dev split of the 129 public tasks, so every later run is scored on the same tasks.

    python scripts/dev_split.py                 # writes splits/dev33.txt + prints the one-screen table  📷
    python scripts/dev_split.py --check         # recompute and compare with the committed list (no write)

Why 33 and why stratified: 5 tasks gave 4/5 then 2/5 for the same config (one task = 20 points), so W1's sample
can't rank configs. 33 is ~1/4 of the public set, small enough to rerun per ablation. Two strata:
  1. repo, by largest remainder: fastapi 17 · rich 12 · requests 4 · httpx 0 (one task; keeping it would over-weight it 4x)
  2. within a repo, gold-patch size: sort by changed lines, cut into n equal bins, draw one task per bin (seeded),
     so the split has the same easy/hard mix as the public set instead of whatever a plain random draw gives.
The gold patch is used ONLY to measure difficulty for the split — never shown to the agent.
Task ids are not competition data, so splits/dev33.txt is committed; harness/tasks.jsonl stays gitignored.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
N, SEED = 33, 0


def changed_lines(patch: str) -> int:
    return sum(1 for l in patch.splitlines()
               if l[:1] in "+-" and not l.startswith(("+++", "---")))


def quotas(counts: dict, n: int) -> dict:
    total = sum(counts.values())
    exact = {r: n * c / total for r, c in counts.items()}
    q = {r: int(x) for r, x in exact.items()}
    for r in sorted(exact, key=lambda r: exact[r] - q[r], reverse=True)[: n - sum(q.values())]:
        q[r] += 1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "harness/tasks.jsonl", type=pathlib.Path)
    ap.add_argument("--out", default=ROOT / "splits/dev33.txt", type=pathlib.Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in a.tasks.open()]
    by_repo = collections.defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append((changed_lines(t["patch"]), t["instance_id"]))
    q = quotas({r: len(v) for r, v in by_repo.items()}, N)

    rng = random.Random(SEED)
    picked, rows = [], []
    for repo in sorted(by_repo):
        pool = sorted(by_repo[repo])                       # (size, id): deterministic order
        k = q[repo]
        chosen = []
        for i in range(k):                                 # bin i = pool[i*len/k : (i+1)*len/k]
            lo, hi = i * len(pool) // k, (i + 1) * len(pool) // k
            chosen.append(rng.choice(pool[lo:hi]))
        picked += [iid for _, iid in chosen]
        sizes = sorted(s for s, _ in chosen)
        all_sizes = sorted(s for s, _ in pool)
        rows.append((repo, len(pool), k,
                     f"{all_sizes[len(all_sizes)//2]}" if pool else "-",
                     f"{sizes[len(sizes)//2]} ({sizes[0]}-{sizes[-1]})" if sizes else "-"))

    text = "".join(f"{i}\n" for i in sorted(picked))
    sha = hashlib.sha256(text.encode()).hexdigest()[:12]

    print(f"{'repo':18} {'public':>6} {'dev':>4}  {'median Δlines':>13}  dev median (min-max)")
    for r in rows:
        print(f"{r[0]:18} {r[1]:>6} {r[2]:>4}  {r[3]:>13}  {r[4]}")
    print(f"{'total':18} {len(tasks):>6} {len(picked):>4}    seed {SEED}  sha {sha}")

    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("matches committed", a.out.relative_to(ROOT) if same else "DIFFERS from committed list")
        sys.exit(0 if same else 1)
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(text)
    print("wrote", a.out.relative_to(ROOT))


if __name__ == "__main__":
    main()
