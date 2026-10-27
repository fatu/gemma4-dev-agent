#!/usr/bin/env python3
"""Where a run's tokens and seconds go: every tool output, every command's time, every compaction (📷 after each run).

    python3 scripts/run_costs.py runs/dev33_base_n1                  # 📷 summary; writes runs/…/costs_calls.tsv
    python3 scripts/run_costs.py runs/dev33_base_n1 --task rich_3063 # 📷 every model call + tool call of one task

Inputs, from <run>/traj/ (scripts/traj_log.py, TRAJ=1, the default):
  <task>.jsonl        one line per model call: prompt / completion / cached tokens, seconds, new messages, reply
  <task>.tools.jsonl  one line per tool call: tool, arguments, wall seconds, output chars, exit code (runs from
                      2026-10-27 on; older runs: tool outputs are read from the re-sent messages, no times)
Token numbers: model tokens are the server's usage counts; a tool output's tokens are ≈ chars / 4 (the output goes back
into every later prompt of that task, so it is paid again on each later call — the `re-paid` column multiplies it out;
an upper bound once a compaction summarises it away).
Context before → after a compaction = prompt tokens of the agent call just before the summary and of the next one.
Prints the summary (photo-sized) and writes every tool call to <run>/costs_calls.tsv (task, call, tool, args, chars,
≈tok, seconds, exit). Outputs quote task data → they stay in runs/ (gitignored).
"""
import json
import pathlib
import statistics
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ONE = sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else None
if ONE in args:
    args.remove(ONE)
RUN = pathlib.Path(args[0])
TRAJ = RUN / "traj"
TOK = 4                                    # chars per token, estimate for tool outputs


def text_of(c):
    if isinstance(c, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c if isinstance(c, str) else json.dumps(c) if c else ""


def one(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def args_of(tool, a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return a
    if isinstance(a, dict):
        if "command" in a:
            return a["command"]
        if tool == "read_file":
            rng = "" if a.get("start_line") is None else f" :{a.get('start_line')}-{a.get('end_line', '')}"
            return f"{a.get('filepath', a.get('path', ''))}{rng}"
        return " ".join(f"{k}={v}" for k, v in a.items() if k not in ("content", "new_string", "old_string")) + (
            f" (+{sum(len(str(a.get(k) or '')) for k in ('content', 'new_string', 'old_string')):,} chars)"
            if any(k in a for k in ("content", "new_string", "old_string")) else "")
    return str(a)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def load(task_file):
    recs = [json.loads(x) for x in open(task_file) if x.strip()]
    tf = task_file.with_name(task_file.name.replace(".jsonl", ".tools.jsonl"))
    tools = [json.loads(x) for x in open(tf) if x.strip()] if tf.exists() else None
    return recs, tools


def tools_from_messages(recs):
    """old runs (no .tools.jsonl): tool results as re-sent in the next request, matched to the call that asked"""
    names, out = {}, []
    for r in recs:
        if r.get("compaction"):
            continue
        for m in r.get("new", []):
            if m.get("role") in ("tool", "tool_responses"):
                name, a, k = names.get(m.get("tool_call_id"), ("?", "", None))
                out.append({"tool": name, "args": a, "out_chars": len(text_of(m.get("content"))), "s": None,
                            "exit_code": None, "call": k})
        for tc in (r.get("resp") or {}).get("tool_calls") or []:
            names[tc.get("id")] = (tc.get("name"), tc.get("args"), r["call"])
    return out


def call_of(recs, t):
    """the model call whose reply asked for a tool call that started at time t"""
    k = None
    for r in recs:
        if not r.get("compaction") and r.get("t") is not None and r["t"] <= t:
            k = r["call"]
    return k


tasks = sorted(f for f in TRAJ.glob("*.jsonl") if not f.name.endswith(".tools.jsonl"))
if not tasks:
    sys.exit(f"run_costs: no {TRAJ}/<task>.jsonl (TRAJ=0 run, or not a run dir)")
if ONE:
    tasks = [f for f in tasks if f.stem == ONE] or sys.exit(f"run_costs: no traj for {ONE}")

rows, comps, per_task = [], [], []
tot = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "reasoning": 0, "think_chars": 0, "first": [],
       "nudges": 0, "nudge_chars": 0, "c_calls": 0, "c_prompt": 0, "c_completion": 0, "c_s": 0.0, "model_s": 0.0}
timed = True
for f in tasks:
    recs, tools = load(f)
    task = f.stem
    if tools is None:
        timed = False
        tools = tools_from_messages(recs)
    agent = [r for r in recs if not r.get("compaction")]
    n_agent = len(agent)
    for i, r in enumerate(recs):
        u = r.get("usage") or {}
        if r.get("compaction"):
            tot["c_calls"] += 1
            tot["c_prompt"] += u.get("prompt") or 0
            tot["c_completion"] += u.get("completion") or 0
            tot["c_s"] += r.get("s") or 0
            before = next((x for x in reversed(recs[:i]) if not x.get("compaction")), None)
            after = next((x for x in recs[i + 1:] if not x.get("compaction")), None)
            comps.append({"task": task, "call": r["call"],
                          "before": ((before or {}).get("usage") or {}).get("prompt"),
                          "after": ((after or {}).get("usage") or {}).get("prompt"),
                          "in": u.get("prompt"), "summary_chars": len(text_of((r.get("resp") or {}).get("content"))),
                          "s": r.get("s")})
            continue
        tot["calls"] += 1
        tot["prompt"] += u.get("prompt") or 0
        tot["cached"] += u.get("cached") or 0
        tot["completion"] += u.get("completion") or 0
        tot["reasoning"] += u.get("reasoning") or 0
        tot["think_chars"] += len((r.get("resp") or {}).get("reasoning") or "")
        tot["model_s"] += r.get("s") or 0
        if r["call"] == 1 or r is agent[0]:
            tot["first"].append(u.get("prompt") or 0)
        if r is not agent[0]:
            for m in r.get("new", []):
                if m.get("role") == "user":
                    tot["nudges"] += 1
                    tot["nudge_chars"] += len(text_of(m.get("content")))
    for x in tools:
        k = x["call"] if "call" in x else call_of(recs, x["t"]) if x.get("t") is not None else None
        later = sum(1 for r in agent if k is not None and r["call"] > k)       # later calls that re-send this output
        rows.append({"task": task, "call": k, "tool": x["tool"], "args": args_of(x["tool"], x.get("args")),
                     "chars": x["out_chars"], "tok": x["out_chars"] // TOK, "repaid": x["out_chars"] // TOK * later,
                     "s": x.get("s"), "exit": x.get("exit_code")})
    per_task.append({"task": task, "model_s": sum(r.get("s") or 0 for r in agent),
                     "tool_s": sum(x.get("s") or 0 for x in tools), "calls": n_agent,
                     "peak": max(((r.get("usage") or {}).get("prompt") or 0) for r in agent) if agent else 0})

# ---------- one task: every call ----------
if ONE:
    recs, _ = load(tasks[0])
    print(f"run_costs {RUN.name} · {ONE}: {per_task[0]['calls']} model calls · "
          f"{len(comps)} compaction · tools {len(rows)} · model {per_task[0]['model_s']:.0f} s · tools "
          f"{per_task[0]['tool_s']:.0f} s{'' if timed else ' (no tool times: run before 2026-10-27)'}")
    print(" #  prompt   out  think  model_s | tool           ≈tok    s   exit  args")
    by_call = {}
    for x in rows:
        by_call.setdefault(x["call"], []).append(x)
    for r in recs:
        u, rp = r.get("usage") or {}, r.get("resp") or {}
        if r.get("compaction"):
            print(f"{r['call']:2} {u.get('prompt') or 0:7,} {u.get('completion') or 0:5,}        {r.get('s', 0):7.1f}  | "
                  f"COMPACTION summary → {len(text_of(rp.get('content'))):,} chars")
            continue
        head = (f"{r['call']:2} {u.get('prompt') or 0:7,} {u.get('completion') or 0:5,} {len(rp.get('reasoning') or ''):6,} "
                f"{r.get('s', 0):7.1f}  | ")
        xs = by_call.get(r["call"]) or [None]
        for j, x in enumerate(xs):
            pre = head if j == 0 else " " * (len(head) - 2) + "| "
            if x is None:
                print(pre + (f"TEXT {one(rp.get('content'), 60)}" if rp.get("content") else "-"))
            else:
                print(pre + f"{x['tool']:14} {x['tok']:5,} {'' if x['s'] is None else format(x['s'], '5.1f'):>5} "
                      f"{'' if x['exit'] is None else x['exit']:>4}  {one(x['args'], 70)}")
    sys.exit(0)

# ---------- whole run ----------
n_tools = len(rows)
tool_tok = sum(x["tok"] for x in rows)
print(f"run_costs {RUN.name}: {len(tasks)} tasks · {tot['calls']} model calls + {tot['c_calls']} compaction · "
      f"{n_tools} tool calls{'' if timed else ' · no tool times (run before 2026-10-27: from re-sent messages)'}")
print("== where the tokens go")
fp = statistics.median(tot["first"]) if tot["first"] else 0
print(f"  prompt tokens processed  {tot['prompt']:>11,}  (cached {tot['cached']:,} = {tot['cached'] / max(1, tot['prompt']):.0%}) "
      f"· mean {tot['prompt'] // max(1, tot['calls']):,}/call")
print(f"  fixed prefix per call    {fp:>11,.0f}  (call 1 prompt, median: system + 9 tool schemas + task prompt) "
      f"× {tot['calls']} calls = {fp * tot['calls']:,.0f}")
rtok = f", reasoning tok {tot['reasoning']:,}" if tot["reasoning"] else ""
print(f"  model output             {tot['completion']:>11,}  (thinking {tot['think_chars']:,} chars{rtok}; re-sent in "
      f"later prompts when include_thoughts is on)")
print(f"  tool outputs             {tool_tok:>11,}  ≈tok once · re-paid in later prompts ≈ {sum(x['repaid'] for x in rows):,}")
print(f"  harness nudges           {tot['nudges']:>11,}  messages · {tot['nudge_chars']:,} chars")
print(f"  compaction calls         {tot['c_calls']:>11,}  · in {tot['c_prompt']:,} · out {tot['c_completion']:,} tok · "
      f"{tot['c_s']:.0f} s")
print("== tool outputs by tool (≈tok = chars/4; re-paid = ≈tok × later calls of its task, an upper bound once compaction drops it)")
stime = "  s: sum   mean   p90    max" if timed else ""
print(f"  {'tool':20} {'n':>4} {'≈tok sum':>9} {'mean':>6} {'p50':>6} {'p90':>6} {'max':>6} {'re-paid':>9}{stime}")
for name in sorted({x["tool"] for x in rows}, key=lambda n: -sum(x["tok"] for x in rows if x["tool"] == n)):
    xs = [x for x in rows if x["tool"] == name]
    t = [x["tok"] for x in xs]
    line = (f"  {name:20} {len(xs):4} {sum(t):9,} {sum(t) // len(t):6,} {pct(t, .5):6,} {pct(t, .9):6,} {max(t):6,} "
            f"{sum(x['repaid'] for x in xs):9,}")
    if timed:
        s = [x["s"] or 0 for x in xs]
        line += f"  {sum(s):6.0f} {sum(s) / len(s):6.1f} {pct(s, .9):5.1f} {max(s):6.1f}"
    print(line)
print(f"== compaction: {len(comps)} summaries in {len({c['task'] for c in comps})} tasks"
      + ("" if comps else " (none fired, or a run without compaction)"))
if comps:
    print(f"  context before → after (prompt tok): median {statistics.median([c['before'] or 0 for c in comps]):,.0f} → "
          f"{statistics.median([c['after'] or 0 for c in comps if c['after']] or [0]):,.0f}")
    for c in comps[:12]:
        print(f"  {c['task']:16} call {c['call']:2}  {c['before'] or 0:6,} → {c['after'] or 0:6,}  "
              f"(summary in {c['in'] or 0:,} tok, {c['summary_chars']:,} chars, {c['s'] or 0:.0f} s)")
    if len(comps) > 12:
        print(f"  … {len(comps) - 12} more in costs_calls.tsv")
if timed:
    cmds = sorted((x for x in rows if x["s"] is not None), key=lambda x: -x["s"])
    print("== slowest 10 tool calls")
    for x in cmds[:10]:
        print(f"  {x['s']:6.1f} s  exit {'' if x['exit'] is None else x['exit']!s:>3}  {x['tok']:5,} tok  {x['task']:16} "
              f"{x['tool']:12} {one(x['args'], 60)}")
print("== largest 10 tool outputs")
for x in sorted(rows, key=lambda x: -x["tok"])[:10]:
    print(f"  {x['tok']:6,} tok  {x['task']:16} call {x['call'] or '?':>2}  {x['tool']:12} {one(x['args'], 60)}")
ms = [p["model_s"] for p in per_task]
ts = [p["tool_s"] for p in per_task]
print(f"== time per task, median: model {statistics.median(ms):.0f} s · tools {statistics.median(ts):.0f} s"
      f"{'' if timed else ' (unknown)'} · peak prompt {statistics.median([p['peak'] for p in per_task]):,.0f} tok")

with open(RUN / "costs_calls.tsv", "w") as f:
    f.write("task\tcall\ttool\targs\tchars\tapprox_tok\trepaid_tok\tseconds\texit\n")
    for x in rows:
        f.write(f"{x['task']}\t{x['call']}\t{x['tool']}\t{one(x['args'], 200)}\t{x['chars']}\t{x['tok']}\t{x['repaid']}\t"
                f"{'' if x['s'] is None else x['s']}\t{'' if x['exit'] is None else x['exit']}\n")
    for c in comps:
        f.write(f"{c['task']}\t{c['call']}\tCOMPACTION\tbefore {c['before']} → after {c['after']}\t{c['summary_chars']}\t"
                f"\t\t{c['s']}\t\n")
print(f"every call: {RUN / 'costs_calls.tsv'} ({n_tools} tool rows + {len(comps)} compaction) · one task: "
      f"python3 scripts/run_costs.py {RUN} --task <id>")
