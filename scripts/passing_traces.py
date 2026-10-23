#!/usr/bin/env python3
"""Index the trajectories that really passed verification across runs (pass@k bookkeeping, B-3j).

    python3 scripts/passing_traces.py runs/dev33_base_ceilx_rvl runs/dev33_base_lenp1 runs/dev33_base_lenp2 ...

For every task: which of the given runs resolved it, and where the trace is. A run whose results were re-verified
(`<run>_rv` / `<run>_rvl`) carries the verdict while the trace lives in the original run dir — pass the re-verified dir and
the script follows the name back. Writes `runs/passing_traces.json` ({task: [{run, trace, patch, lenient}]}) and symlinks
`runs/passing/<task>/<run>.json` → the trace, so "the passing trajectories" is one directory. Marks lenient verdicts (L).
Prints a one-screen table: task · passes/attempts · runs. Everything stays under runs/ (gitignored; task text inside).
"""
import json
import os
import pathlib
import sys


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]] or sys.exit(__doc__)
    index: dict[str, list] = {}
    attempts: dict[str, int] = {}
    for r in runs:
        res_path = r / "task_results.jsonl"
        if not res_path.exists():
            print(f"skip {r}: no task_results.jsonl"); continue
        base = r
        for suf in ("_rvl", "_rv"):
            if r.name.endswith(suf) and (r.parent / r.name[: -len(suf)]).exists():
                base = r.parent / r.name[: -len(suf)]
                break
        lenient = (r / "lenient.txt").exists()
        for line in res_path.read_text().splitlines():
            t = json.loads(line); tid = t["instance_id"]
            attempts[tid] = attempts.get(tid, 0) + 1
            if not t.get("resolved"):
                continue
            trace = base / "traces" / f"trace_{tid}.json"
            patch = base / "patches" / f"{tid}.patch"
            index.setdefault(tid, []).append({"run": str(r), "trace": str(trace) if trace.exists() else None,
                                              "patch": str(patch) if patch.exists() else None, "lenient": lenient})
    out = pathlib.Path("runs/passing_traces.json"); out.write_text(json.dumps(index, indent=1, sort_keys=True))
    root = pathlib.Path("runs/passing")
    links = 0
    for tid, hits in index.items():
        d = root / tid; d.mkdir(parents=True, exist_ok=True)
        for h in hits:
            if h["trace"]:
                dst = d / (pathlib.Path(h["run"]).name + ".json")
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(os.path.relpath(h["trace"], d)); links += 1
    print(f"{'task':16} {'pass/att':>8}  runs (L = lenient verdict)")
    for tid in sorted(attempts):
        hits = index.get(tid, [])
        tags = " ".join(pathlib.Path(h["run"]).name.replace("dev33_base_", "") + (" L" if h["lenient"] else "") for h in hits)
        print(f"{tid:16} {len(hits):>4}/{attempts[tid]:<3}  {tags or '—'}")
    print(f"\n{len(index)} tasks with ≥1 passing trajectory of {len(attempts)} · {links} trace links under {root}/ · index {out}")


if __name__ == "__main__":
    main()
