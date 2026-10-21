#!/usr/bin/env python3
"""Did the agent follow the prompt rules? One column per run, from the traces (B-3 follow-up, BOX_PLAN.md).

    python3 scripts/rulestats.py runs/dev33_base runs/dev33_base_w5 runs/dev33_editrecover_w5 ...   # 📷

Each row is a behaviour one of the arms is supposed to change, so a rule that "lost" can be read as either
"the agent ignored it" (the row doesn't move) or "the agent followed it and it didn't help" (the row moves, the
score doesn't). Reads <run>/traces/trace_*.json and <run>/task_results.jsonl, also when they sit in one
sub-directory per task (the Kaggle lab layout). Aggregates only; task ids are never printed.
"""
import collections
import json
import pathlib
import re
import statistics
import sys

TEST_RE = re.compile(r"\bpytest\b|\bunittest\b|python3? -m pytest")
HEREDOC_RE = re.compile(r"python3?\s+-\s*<<|<<\s*['\"]?(EOF|PY)['\"]?")


def p50(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.0f}" if xs else "–"


def pct(n, d):
    return f"{n}/{d}" if d else "–"


def load(run: pathlib.Path):
    results = {}
    for f in run.rglob("task_results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[r["instance_id"]] = r
    traces = {}
    for f in run.rglob("trace_*.json"):
        traces[f.stem[len("trace_"):]] = json.loads(f.read_text()).get("steps", [])
    return results, traces


def analyse(steps, result):
    """per-task facts from one trace"""
    calls = []                                        # (function_name, arguments, observation-json)
    for s in steps:
        if s.get("source") != "agent":
            continue
        obs = json.dumps(s.get("observation") or "")
        for c in s.get("tool_calls") or []:
            calls.append((c["function_name"], c.get("arguments") or {}, obs))
    names = [n for n, _, _ in calls]
    f = collections.Counter()
    f["calls"] = len(calls)
    # edit_file outcomes
    missing_at = []
    for i, (n, a, o) in enumerate(calls):
        if n != "edit_file":
            continue
        if "mandatory input parameters" in o or ("old_string" not in a and "new_string" not in a):
            f["edit_missing"] += 1; missing_at.append(i)
        elif "not found" in o or "No match" in o or "multiple" in o.lower():
            f["edit_mismatch"] += 1
        else:
            f["edit_ok"] += 1
    # retry loops: 3+ consecutive missing-param edits
    run_len = best = 0
    for n, a, o in calls:
        if n == "edit_file" and ("mandatory input parameters" in o or ("old_string" not in a and "new_string" not in a)):
            run_len += 1; best = max(best, run_len)
        else:
            run_len = 0
    f["edit_loop"] = int(best >= 3)
    # heredoc edits through run_command, and whether they follow a failed edit (editrecover)
    heredoc_at = [i for i, (n, a, o) in enumerate(calls)
                  if n == "run_command" and HEREDOC_RE.search(a.get("command", "") or "")]
    f["heredoc_edits"] = len(heredoc_at)
    f["recovered"] = sum(1 for m in missing_at if any(m < h <= m + 2 for h in heredoc_at))
    # first edit (edit_file / write_file / heredoc), reads before it (issuefirst)
    edit_idx = [i for i, n in enumerate(names) if n in ("edit_file", "write_file")] + heredoc_at
    first = min(edit_idx) + 1 if edit_idx else None
    f["never_edited"] = int(first is None)
    reads = sum(1 for n in names[:(first or len(names))] if n == "read_file")
    # /tmp scratch attempts (tmpfix)
    f["tmp_writes"] = sum(1 for n, a, o in calls
                          if n in ("write_file", "edit_file") and str(a.get("filepath", a.get("path", ""))).startswith("/tmp"))
    f["traversal"] = sum(1 for n, a, o in calls if "traversal" in o.lower())
    # tests (test1): any pytest/unittest run_command, and its position
    test_idx = [i + 1 for i, (n, a, o) in enumerate(calls)
                if n == "run_command" and TEST_RE.search(a.get("command", "") or "")]
    f["tested"] = int(bool(test_idx))
    f["test_calls"] = len(test_idx)
    # submit
    sub_idx = [i + 1 for i, n in enumerate(names) if n == "submit_patch"]
    f["submitted"] = int(bool(sub_idx))
    f["test_before_submit"] = int(bool(test_idx) and (not sub_idx or test_idx[0] < sub_idx[0]))
    f["graph_tools"] = sum(1 for n in names if n in ("get_code_neighbors", "search_similar_code", "get_code_subgraph"))
    err = (result or {}).get("error") or ""
    return f, dict(first_edit=first, reads_before=reads, first_test=test_idx[0] if test_idx else None,
                   submit_at=sub_idx[0] if sub_idx else None,
                   overflow=int("context length" in err or "ContextWindow" in err),
                   timeout=int("session timeout" in err), callcap=int("tool call budget" in err),
                   resolved=int(bool((result or {}).get("resolved"))),
                   duration=(result or {}).get("duration_seconds"),
                   patch=(result or {}).get("agent_patch_size") or 0)


ROWS = [
    ("tasks (with a trace)", lambda S: f"{S['n']}"),
    ("resolved", lambda S: f"{S['resolved']}"),
    ("— edits —", None),
    ("edit_file ok / missing-param / mismatch", lambda S: f"{S['edit_ok']} / {S['edit_missing']} / {S['edit_mismatch']}"),
    ("runs with a 3+ missing-param loop", lambda S: f"{S['edit_loop']}"),
    ("heredoc edits via run_command", lambda S: f"{S['heredoc_edits']}"),
    ("missing-param edits followed by a heredoc ≤2 calls later", lambda S: pct(S['recovered'], S['edit_missing'])),
    ("— localisation —", None),
    ("first edit at call (p50)", lambda S: p50(S['first_edit'])),
    ("runs that never edited", lambda S: f"{S['never_edited']}"),
    ("read_file before the first edit (p50)", lambda S: p50(S['reads_before'])),
    ("— scratch files —", None),
    ("write/edit under /tmp · 'traversal' errors", lambda S: f"{S['tmp_writes']} · {S['traversal']}"),
    ("— tests —", None),
    ("runs that ran a test", lambda S: f"{S['tested']}"),
    ("test calls in total · first test at call (p50)", lambda S: f"{S['test_calls']} · {p50(S['first_test'])}"),
    ("runs with a test before submit_patch", lambda S: f"{S['test_before_submit']}"),
    ("— ending —", None),
    ("runs that called submit_patch · at call (p50)", lambda S: f"{S['submitted']} · {p50(S['submit_at'])}"),
    ("overflow · 8-min timeout · 28-call cap", lambda S: f"{S['overflow']} · {S['timeout']} · {S['callcap']}"),
    ("tool calls (p50) · seconds (p50)", lambda S: f"{p50(S['calls_list'])} · {p50(S['duration'])}"),
    ("graph-tool calls", lambda S: f"{S['graph_tools']}"),
    ("resolved runs: first edit (p50) · tested", lambda S: f"{p50(S['first_edit_res'])} · {S['tested_res']}"),
    ("failed runs: first edit (p50) · tested", lambda S: f"{p50(S['first_edit_fail'])} · {S['tested_fail']}"),
]


def summarise(run):
    results, traces = load(run)
    S = collections.Counter(n=len(traces))
    lists = collections.defaultdict(list)
    for tid, steps in traces.items():
        f, x = analyse(steps, results.get(tid))
        S.update(f)
        for k in ("resolved", "overflow", "timeout", "callcap"):
            S[k] += x[k]
        lists["first_edit"].append(x["first_edit"]); lists["reads_before"].append(x["reads_before"])
        lists["first_test"].append(x["first_test"]); lists["submit_at"].append(x["submit_at"])
        lists["calls_list"].append(f["calls"]); lists["duration"].append(x["duration"])
        key = "res" if x["resolved"] else "fail"
        lists[f"first_edit_{key}"].append(x["first_edit"]); S[f"tested_{key}"] += f["tested"]
    S = dict(S); S.update(lists)                 # lists must not go through Counter.update
    return S


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]] or sys.exit(__doc__)
    names = [r.name[len("dev33_"):] if r.name.startswith("dev33_") else r.name for r in runs]
    depth = 1                                          # same dir name twice: prefix parents until unique
    while len(set(names)) < len(names) and depth < 6:
        depth += 1
        names = ["/".join(r.resolve().parts[-depth:]) for r in runs]
    cols = {n: summarise(r) for n, r in zip(names, runs)}
    w = max(len(k) for k, _ in ROWS) + 1
    cw = {n: max(14, len(n) + 2) for n in names}
    print(" " * w + "".join(f"{n:>{cw[n]}}" for n in names))
    for label, fn in ROWS:
        if fn is None:
            print(label); continue
        print(f"{label:{w}}" + "".join(f"{fn(S):>{cw[n]}}" for n, S in cols.items()))


if __name__ == "__main__":
    main()
