#!/usr/bin/env python3
"""Distribution statistics of call-level training records (train/schema.md) — the data-side acceptance gate of a
fine-tune mix, next to render.py's format checks. Born from the 2026-10-30 recheck of M2's data (notes/14 § 训练数据与
loss mask 复查): the copy-the-previous-call loop had no literal prototype in the data, but three measurable shifts from
E4B's own behaviour pointed the same way (same-tool prior 60% vs 36%, thoughts naming only 22% of the action's
identifiers vs 47%, no long-thought tail, almost no error-recovery / compaction states).

    python3 train/data_stats.py runs/sft/d3_relabel_31b runs/sft/e4b_d3 [--gate]
    python3 train/convert_traj.py --run runs/pod_sft/runs/dev33_base_s1 --source e4b_onpolicy --out runs/sft/ref_base   # a reference lane

One column per records dir, the E4B base reference (dev33, 2026-10-29 run) as the last column. --gate prints PASS/FAIL
against the thresholds below (proposed 2026-10-30; the base reference values are what they are measured against).
Every number printed is the record; nothing else is written.
"""
import argparse
import difflib
import glob
import json
import os
import re
import sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{3,}")
# E4B base on dev33 (runs/pod_sft/runs/dev33_base_s1, 455 tool-call targets), measured 2026-10-30 with this code's formulas
REF = {"name": "E4B base dev33 (ref)", "identical": 3, "same_tool": 36, "call_sim_p50": 0.45, "thought_sim80": 0, "names_own": 0.47,
       "names_prev": 0.30, "th_p50": 692, "th_p90": 3756, "th_p99": 14540, "th_empty": 0, "after_err": 22, "after_err_diff": 89}
GATE = {"identical": ("<=", 3), "same_tool": ("<=", 45), "names_own": (">=", 0.40), "th_p90": (">=", 2000), "after_err": (">=", 10)}


def q(v, f):
    v = sorted(v)
    return v[min(int(len(v) * f), len(v) - 1)] if v else None


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else None


def args_text(tcs):
    out = []
    for tc in tcs:
        a = tc.get("args")
        out.append(a if isinstance(a, str) else json.dumps(a, sort_keys=True, ensure_ascii=False))
    return " ".join(out)


def stats(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if isinstance(r, dict) and isinstance(r.get("calls"), list):
            recs.append(r)
    s = {"name": os.path.basename(d.rstrip("/")), "records": len(recs), "resolved": sum(1 for r in recs if r.get("resolved")),
         "tasks": len({r["instance_id"] for r in recs})}
    per_traj, ident, same_tool, call_sim, th_sim, own, prev, th_len, after_err, after_err_diff = [], 0, 0, [], [], [], [], [], 0, 0
    pairs, nudge, compaction, tools, sources = 0, 0, 0, {}, {}
    for r in recs:
        last = None
        n = 0
        for c in r["calls"]:
            t = c.get("target") or {}
            tcs = t.get("tool_calls") or []
            nudge += "nudge" in (c.get("events_before") or []); compaction += "compaction" in (c.get("events_before") or [])
            if not tcs:
                continue
            n += 1
            sources[c.get("thought_source")] = sources.get(c.get("thought_source"), 0) + 1
            for tc in tcs:
                tools[tc["name"]] = tools.get(tc["name"], 0) + 1
            th = (t.get("reasoning") or "").strip()
            th_len.append(len(th))
            a_txt = args_text(tcs)
            cur = (json.dumps([(tc["name"], a_txt)], sort_keys=True), th, tcs[0]["name"], set(IDENT.findall(a_txt)))
            m = c["input_messages"][-1] if c.get("input_messages") else {}
            err = m.get("role") == "tool" and '"error"' in str(m.get("content"))[:40]
            if cur[3]:
                own.append(len(cur[3] & set(IDENT.findall(th))) / len(cur[3]))
                prev.append(len(cur[3] & (last[3] if last else set())) / len(cur[3]))
            if last:
                pairs += 1
                ident += cur[0] == last[0]
                same_tool += cur[2] == last[2]
                call_sim.append(sim(cur[0], last[0]))
                if th and last[1]:
                    th_sim.append(sim(th, last[1]))
                if err:
                    after_err += 1; after_err_diff += cur[0] != last[0]
            last = cur
        per_traj.append(n)
    s.update(calls=sum(per_traj), calls_p50=q(per_traj, .5), calls_max=max(per_traj) if per_traj else 0, pairs=pairs,
             identical=round(100 * ident / max(pairs, 1)), same_tool=round(100 * same_tool / max(pairs, 1)),
             call_sim_p50=round(q(call_sim, .5) or 0, 2), thought_sim80=round(100 * sum(x >= .8 for x in th_sim) / max(len(th_sim), 1)),
             names_own=round(sum(own) / max(len(own), 1), 2), names_prev=round(sum(prev) / max(len(prev), 1), 2),
             th_p50=q(th_len, .5), th_p90=q(th_len, .9), th_p99=q(th_len, .99), th_empty=round(100 * sum(x == 0 for x in th_len) / max(len(th_len), 1)),
             after_err=round(100 * after_err / max(pairs, 1)), after_err_diff=round(100 * after_err_diff / max(after_err, 1)),
             nudge=nudge, compaction=compaction, tools=tools, sources=sources)
    return s


ROWS = [("records (resolved) / tasks", lambda s: f"{s['records']} ({s['resolved']}) / {s['tasks']}"),
        ("tool-call targets · per traj p50 / max", lambda s: f"{s['calls']} · {s['calls_p50']} / {s['calls_max']}"),
        ("next call identical to previous %", "identical"), ("same tool name as previous %", "same_tool"),
        ("call string similarity p50", "call_sim_p50"), ("consecutive thoughts ≥0.8 similar %", "thought_sim80"),
        ("action identifiers named in own thought", "names_own"), ("action identifiers from previous call", "names_prev"),
        ("thought chars p50 / p90 / p99", lambda s: f"{s['th_p50']} / {s['th_p90']} / {s['th_p99']}"), ("empty thoughts %", "th_empty"),
        ("calls right after a tool error %", "after_err"), ("  of which a different call %", "after_err_diff"),
        ("nudge / compaction events", lambda s: f"{s['nudge']} / {s['compaction']}"),
        ("tools", lambda s: " ".join(f"{k} {v}" for k, v in sorted(s["tools"].items(), key=lambda kv: -kv[1]))),
        ("thought_source", lambda s: " ".join(f"{k} {v}" for k, v in s["sources"].items()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gate", action="store_true", help="PASS/FAIL the first dir against GATE")
    ap.add_argument("--no-ref", action="store_true")
    a = ap.parse_args()
    cols = [stats(d) for d in a.dirs] + ([] if a.no_ref else [REF])
    w = max(len(c["name"]) for c in cols) + 2
    print(f"{'':44}" + "".join(f"{c['name']:>{w}}" for c in cols))
    for label, key in ROWS:
        vals = []
        for c in cols:
            if callable(key):
                vals.append("" if c is REF else key(c))
            else:
                vals.append(str(c.get(key, "")))
        if max(len(v) for v in vals) > w:               # long rows (tool mix, sources): one line per column
            print(f"{label}:" + "".join(f"\n    {c['name']}: {v}" for c, v in zip(cols, vals) if v))
        else:
            print(f"{label:44}" + "".join(f"{v:>{w}}" for v in vals))
    if a.gate:
        s = cols[0]; fails = []
        for k, (op, thr) in GATE.items():
            ok = s[k] <= thr if op == "<=" else s[k] >= thr
            print(f"  gate {k:12} {s[k]!s:>6} {op} {thr:<6} {'ok' if ok else 'FAIL'}")
            fails += [] if ok else [k]
        print(f"data gate: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)} ({s['name']})")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
