#!/usr/bin/env python3
"""Contrast A of the failure study (notes/16): on the tasks a reference run solved and a small model failed, how did
the two paths differ before the first edit? Reads traj/<task>.jsonl + .tools.jsonl of both runs, harness/tasks.jsonl
(gold files), runs/fail_atlas/*.tsv and runs/fail_judge/<small>/ (first_wrong_call). No API calls.

    python3 scripts/fail_contrast.py <small run> <ref run>                      # per-path features, both runs, + table
    python3 scripts/fail_contrast.py <small run> <ref run> --pair fastapi_11194 # the two call lists side by side
                                                                                # (📷-safe: tool + short args, no output)
Features per run and task (calls numbered as in the trajectory, compactions skipped):
  repro      a python -c / python script run (not pytest) before the first edit
  greps      grep / rg / find commands before the first edit · gdef: one of them looks for a definition (def / class)
  gold_at    first call whose arguments name a gold source file (path or basename) · gold_read: read_file of one
  reads      read_file calls before the first edit · win: share of them with a line window (start_line / end_line)
  first_edit call of the first edit attempt (edit_file / write_file / heredoc) · in_gold: it targets a gold file
  efail      failed edit_file calls before the first one that worked · tested: pytest after the first edit
Task ids appear in --pair and in the per-task table (private; the summary lines are aggregates).
"""
import csv
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTFILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
HEREDOC = re.compile(r"<<\s*['\"]?\w+|python3? - <<|cat\s*>|sed -i|>\s*\S+\.py")


def gold_files(patch):
    return [f for f in re.findall(r"^\+\+\+ b/(\S+)", patch or "", re.M) if not TESTFILE.search(f)]


def calls(run, task):
    p = run / "traj" / f"{task}.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("compaction"):
            continue
        for t in (r.get("resp") or {}).get("tool_calls") or []:
            a = t.get("args")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except ValueError:
                    a = {"raw": a}
            out.append((r["call"], t.get("name"), a or {}))
    return out


def tool_status(run, task):
    p = run / "traj" / f"{task}.tools.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def is_edit(name, a):
    return name in ("edit_file", "write_file") or (name == "run_command" and HEREDOC.search(a.get("command", "") or ""))


def features(run, task, gold):
    cs = calls(run, task)
    base = [pathlib.Path(g).name for g in gold]
    names_gold = lambda a: any(g in json.dumps(a) for g in gold + base)
    fe = next((i for i, (c, n, a) in enumerate(cs) if is_edit(n, a)), None)
    pre = cs[:fe] if fe is not None else cs
    cmds = [a.get("command", "") or "" for c, n, a in pre if n == "run_command"]
    greps = [x for x in cmds if re.search(r"\b(grep|rg|find)\b", x)]
    reads = [a for c, n, a in pre if n == "read_file"]
    st = [s for s in tool_status(run, task) if s.get("tool") == "edit_file"]
    efail = 0
    for s in st:
        if s.get("status") == "ok":
            break
        efail += 1
    post = cs[fe:] if fe is not None else []
    edit_target = json.dumps(cs[fe][2]) if fe is not None else ""
    return dict(
        calls=len(cs),
        repro=int(any(re.search(r"python3? (-c|\S+\.py)", x) and "pytest" not in x for x in cmds)),
        greps=len(greps), gdef=int(any(re.search(r"(def|class)\s", x) for x in greps)),
        gold_at=next((c for c, n, a in cs if names_gold(a)), None),
        gold_read=int(any(names_gold(a) for a in reads)),
        reads=len(reads), win=round(sum(bool(a.get("start_line") or a.get("end_line")) for a in reads) / len(reads), 2) if reads else None,
        first_edit=cs[fe][0] if fe is not None else None,
        in_gold=int(any(g in edit_target for g in gold + base)) if fe is not None else 0,
        efail=efail,
        tested=int(any(n == "run_command" and "pytest" in (a.get("command", "") or "") for c, n, a in post)))


def short(n, a):
    if n == "run_command":
        return "$ " + (a.get("command", "") or "").replace("\n", " ")[:70]
    if n in ("read_file", "edit_file", "write_file"):
        w = f" {a.get('start_line', '')}-{a.get('end_line', '')}" if a.get("start_line") or a.get("end_line") else ""
        return f"{n} {a.get('filepath') or a.get('path') or '?'}{w}"
    return n


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    small, ref = pathlib.Path(args[0]), pathlib.Path(args[1])
    tasks = {t["instance_id"]: t for t in map(json.loads, open(ROOT / "harness" / "tasks.jsonl"))}
    at = lambda run: {r["task"]: r for r in csv.DictReader(open(ROOT / "runs" / "fail_atlas" / f"{run.name}.tsv"), delimiter="\t")}
    sa, ra = at(small), at(ref)
    judge = {p.stem: json.loads(p.read_text()) for p in (ROOT / "runs" / "fail_judge" / small.name).glob("*.json")}
    if "--pair" in args:
        t = args[args.index("--pair") + 1]
        A, B = calls(small, t), calls(ref, t)
        fw = (judge.get(t) or {}).get("first_wrong_call")
        print(f"{t} · gold {gold_files(tasks[t]['patch'])} · {small.name} {sa[t]['outcome']} (judge {judge.get(t, {}).get('primary')}, "
              f"first wrong call {fw}) | {ref.name} {ra[t]['outcome']}")
        for k in range(max(len(A), len(B))):
            a = f"{A[k][0]:>3} {short(A[k][1], A[k][2])}" if k < len(A) else ""
            b = f"{B[k][0]:>3} {short(B[k][1], B[k][2])}" if k < len(B) else ""
            mark = " ◀" if k < len(A) and fw and A[k][0] == fw else ""
            print(f"{a[:78]:78}{mark:2} | {b[:78]}")
        return
    ids = sorted(i for i in sa if sa[i]["outcome"] != "✓" and sa[i]["env_defect"] == "0" and ra.get(i, {}).get("outcome") == "✓")
    rows = []
    for i in ids:
        g = gold_files(tasks[i]["patch"])
        rows.append((i, (judge.get(i) or {}).get("primary", "-"), features(small, i, g), features(ref, i, g)))
    print(f"contrast A: {len(ids)} tasks {ref.name} solved and {small.name} failed (valid)")
    print(f"{'task':16} {'cause':5} {'small: repro greps gdef gold@ read win edit@ inG efail':55} | ref")
    for i, c, a, b in rows:
        f = lambda d: f"{d['repro']:5} {d['greps']:5} {d['gdef']:4} {str(d['gold_at']):5} {d['reads']:4} {str(d['win']):4} {str(d['first_edit']):5} {d['in_gold']:3} {d['efail']:5}"
        print(f"{i:16} {c:5} {f(a):55} | {f(b)}")

    def agg(k, kind):
        out = []
        for side in (2, 3):
            v = [r[side][k] for r in rows if r[side][k] is not None]
            out.append(f"{sum(v)}/{len(rows)}" if kind == "n" else (f"p50 {statistics.median(v):g}" if v else "–"))
        return out
    print(f"\nsummary (small | ref) over {len(rows)} tasks:")
    for k, kind, label in [("repro", "n", "reproduced before editing"), ("gdef", "n", "grepped for a definition"),
                           ("greps", "m", "grep/find calls before edit"), ("gold_read", "n", "read a gold file before editing"),
                           ("gold_at", "m", "call that first names a gold file"), ("reads", "m", "reads before the first edit"),
                           ("win", "m", "share of windowed reads"), ("first_edit", "m", "call of the first edit"),
                           ("in_gold", "n", "first edit in a gold file"), ("efail", "m", "failed edits before one works"),
                           ("tested", "n", "ran pytest after editing")]:
        s, r = agg(k, kind)
        print(f"  {label:36} {s:>9} | {r}")


if __name__ == "__main__":
    main()
