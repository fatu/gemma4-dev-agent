#!/usr/bin/env python3
"""B-11 step 7 · live check of an adapter in the competition harness: reads <run>/traj/<task>.jsonl (scripts/traj_log.py)
and <run>/task_results.jsonl of a few tasks and prints, per task and in total, what the plan asks to see.

    python3 scripts/b11_live.py --expect-thinking on  runs/dev33_base_liveA
    python3 scripts/b11_live.py --expect-thinking off runs/dev33_think0_liveB

Per task: agent turns (compaction calls left out) · turns with a parsed tool call · turns whose text or thought still
holds raw call markup (a call the server did not parse) · tools used · submit_patch called · longest run of identical
consecutive calls (same tool and arguments) and how many of those pairs also repeat the thought word for word · turns
with a thought and thought characters · failed model calls · the harness's result line (resolved, patch size, seconds,
error). Then one line per check: ok / LOOK. No task text is printed.
Checks: 1 parse = every task has ≥ 1 parsed call and no raw markup · 2 submit = submit_patch called in ≥ 1 task ·
3 copy = no run of ≥ 3 identical consecutive calls · 4 time = no failed model call and no harness error ·
5 thinking = the request's enable_thinking and the thoughts match --expect-thinking (off: 0 thought characters;
on: a thought on ≥ half of the tool-call turns)."""
import argparse
import collections
import glob
import json
import os
import re

RAW = re.compile(r"<\|tool_call>|<tool_call>|<\|tool_response>|call:[a-z_]+\{")


def results(run):
    p = os.path.join(run, "task_results.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); out[r.get("instance_id")] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--expect-thinking", choices=["on", "off"], required=True)
    a = ap.parse_args()
    print(f"b11_live · expect thinking {a.expect_thinking} · runs {' '.join(a.runs)}")
    for run in a.runs:
        res = results(run)
        files = sorted(glob.glob(os.path.join(run, "traj", "*.jsonl")))
        print(f"\n== {run}: {len(files)} trajectories · {len(res)} result lines")
        bad = collections.defaultdict(list)
        n_submit = 0
        for k, f in enumerate(files, 1):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            task = rows[0].get("task") if rows else os.path.basename(f)[:-6]
            setup = next((r["setup"] for r in rows if r.get("setup")), {})
            et = ((setup.get("params") or {}).get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking")
            turns = [r for r in rows if not r.get("compaction")]
            ncomp = len(rows) - len(turns)
            calls, raw, th_turns, th_chars, th_on_call, errs = 0, 0, 0, 0, 0, 0
            tools = collections.Counter(); seq = []
            for r in turns:
                resp = r.get("resp") or {}
                if r.get("err"):
                    errs += 1
                tc = resp.get("tool_calls") or []
                th = resp.get("reasoning") or ""
                if RAW.search((resp.get("content") or "") + th):
                    raw += 1
                if th.strip():
                    th_turns += 1; th_chars += len(th)
                if tc:
                    calls += 1; th_on_call += bool(th.strip())
                    for c in tc:
                        tools[c.get("name")] += 1
                    seq.append((json.dumps([(c.get("name"), c.get("args")) for c in tc], sort_keys=True), th.strip()))
            longest, cur, pairs, exact = 1 if seq else 0, 1, 0, 0
            for i in range(1, len(seq)):
                if seq[i][0] == seq[i - 1][0]:
                    cur += 1; pairs += 1; exact += seq[i][1] == seq[i - 1][1]
                else:
                    cur = 1
                longest = max(longest, cur)
            sub = tools.get("submit_patch", 0) > 0; n_submit += sub
            r = res.get(task, {})
            print(f"task {k}: turns {len(turns)} (+{ncomp} compaction) · with a parsed call {calls} · raw call markup in {raw} · "
                  f"tools {dict(tools.most_common())} · submit_patch {'yes' if sub else 'NO'}")
            print(f"        identical consecutive calls: pairs {pairs} (thought identical too {exact}) · longest run {longest} · "
                  f"enable_thinking sent {et} · thought on {th_turns} turns ({th_on_call} of {calls} call turns), {th_chars} chars · failed model calls {errs}")
            print(f"        result: resolved {r.get('resolved')} · patch {r.get('agent_patch_size')} bytes · {r.get('duration_seconds')} s · "
                  f"tool calls {r.get('tool_calls')} · error {str(r.get('error'))[:80]}")
            if calls == 0 or raw:
                bad["1 parse"].append(k)
            if longest >= 3:
                bad["3 copy"].append(k)
            if errs or str(r.get("error")) not in ("None", "", "nan") or not r:
                bad["4 time"].append(k)
            want = a.expect_thinking == "on"
            if et is not want or (not want and th_chars) or (want and calls and th_on_call * 2 < calls):
                bad["5 thinking"].append(k)
        if n_submit == 0:
            bad["2 submit"] = ["none of the tasks"]
        for c in ["1 parse", "2 submit", "3 copy", "4 time", "5 thinking"]:
            print(f"check {c}: {'ok' if not bad[c] else 'LOOK — task ' + ', '.join(map(str, bad[c]))}")
        print(f"→ {'all five ok' if not any(bad.values()) else 'LOOK at the lines above before step 8'}")


if __name__ == "__main__":
    main()
