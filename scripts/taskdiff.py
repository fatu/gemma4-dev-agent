#!/usr/bin/env python3
"""One task across several passes: how each pass ended, and whether the net was involved (BOX_PLAN.md B-3e).

    python3 scripts/taskdiff.py requests_7328 runs/dev33_base_r2 runs/dev33_base_k1 runs/dev33_base_k2   # 📷

Per pass: outcome, seconds, tool calls, the result's error (first 160 chars), the last two tool observations
(first 200 chars each), and a count of network-failure phrases anywhere in the trace or in any file under the run
directory that names the task (verify logs). Prints one task id (the dev split is public), never task text beyond
those observation snippets; keep the photo private all the same.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from flips import outcome, facts   # noqa: E402

NET_RE = re.compile(r"Name or service not known|Temporary failure in name resolution|ConnectionError|"
                    r"Network is unreachable|httpbin\.org|Max retries exceeded|Could not resolve host|"
                    r"ProxyError|SSLError|ConnectTimeout|NewConnectionError", re.I)


def snippets(steps, n=2):
    out = []
    for s in steps:
        if s.get("source") != "agent":
            continue
        o = s.get("observation")
        o = o.get("content") if isinstance(o, dict) else o
        if o:
            out.append(re.sub(r"\s+", " ", str(o))[:200])
    return out[-n:]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    task, runs = sys.argv[1], [pathlib.Path(a) for a in sys.argv[2:]]
    for run in runs:
        res = {}
        for f in run.rglob("task_results.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    res[r["instance_id"]] = r
        r = res.get(task)
        tr = next(run.rglob(f"trace_{task}.json"), None)
        steps = json.loads(tr.read_text()).get("steps", []) if tr else []
        print(f"== {run.name}: {task}")
        if not r:
            print("   no result row"); continue
        calls, peak, first = facts(steps)
        print(f"   {outcome(r)}  {r.get('duration_seconds') or 0:.0f} s · {calls} calls · first edit {first or '–'} · "
              f"peak prompt {peak} · patch {r.get('agent_patch_size') or 0} B")
        err = (r.get("error") or "").replace("\n", " ")
        print(f"   error: {err[:160] or '–'}")
        for sn in snippets(steps):
            print(f"   obs: {sn}")
        net_trace = len(NET_RE.findall(tr.read_text())) if tr else 0
        hits = {}
        for f in run.rglob("*"):
            if f.is_file() and f.suffix in (".log", ".txt", ".json", ".jsonl", ".out") and task in f.name and f != tr:
                try:
                    n = len(NET_RE.findall(f.read_text(errors="ignore")))
                except OSError:
                    continue
                if n:
                    hits[str(f.relative_to(run))] = n
        print(f"   network phrases: trace {net_trace}" + "".join(f" · {k} {v}" for k, v in hits.items()))
    print("\n(phrases counted: " + NET_RE.pattern.replace("\\", "") + ")")


if __name__ == "__main__":
    main()
