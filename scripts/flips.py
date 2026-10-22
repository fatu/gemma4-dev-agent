#!/usr/bin/env python3
"""Why do the same tasks flip between identical passes? Per flip task, one line per pass (B-3b follow-up, BOX_PLAN.md).

    python3 scripts/flips.py runs/dev33_base runs/dev33_base_w5 runs/dev33_base_r2 runs/dev33_base_r3   # 📷

A flip task is resolved in some passes and not in others. For every pass it prints the outcome and where the run
stood: seconds, tool calls, the call of the first edit, the peak prompt size, patch bytes. Then a summary: what
the losing passes died of, and how close the winning passes were to the time / context budget. Prints task ids
(the dev split is public) but never task text; keep the photo private all the same.
Outcomes: ✓ resolved · ✗ wrong patch (tests ran, failed) · O context overflow · T time budget · C call budget ·
∅ no patch (ended with nothing to grade) · E other error.
"""
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rulestats import load, HEREDOC_RE   # noqa: E402

CAP_S = {"4": 240, "8": 480}


def outcome(r):
    err = (r.get("error") or "")
    if r.get("resolved"):
        return "✓"
    if "ContextWindow" in err or "context length" in err:
        return "O"
    if "session timeout" in err:
        return "T"
    if "tool call budget" in err:
        return "C"
    if err:
        return "E"
    return "✗" if (r.get("agent_patch_size") or 0) > 0 else "∅"


def facts(steps):
    calls, peak, first = 0, 0, None
    for s in steps:
        if s.get("source") != "agent":
            continue
        peak = max(peak, (s.get("metrics") or {}).get("prompt_tokens") or 0)
        for c in s.get("tool_calls") or []:
            calls += 1
            n, a = c["function_name"], c.get("arguments") or {}
            if first is None and (n in ("edit_file", "write_file") or
                                  (n == "run_command" and HEREDOC_RE.search(a.get("command", "") or ""))):
                first = calls
    return calls, peak, first


def p50(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.0f}" if xs else "–"


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]] or sys.exit(__doc__)
    names = [r.name[len("dev33_"):] if r.name.startswith("dev33_") else r.name for r in runs]
    depth = 1                                        # same dir name twice: prefix parents until unique
    while len(set(names)) < len(names) and depth < 6:
        depth += 1
        names = ["/".join(r.resolve().parts[-depth:]) for r in runs]
    data = [(n, *load(r)) for n, r in zip(names, runs)]   # per pass: (name, results, traces)
    tasks = sorted({t for _, res, _ in data for t in res})
    always, never, flips = [], [], []
    for t in tasks:
        outs = [bool(res.get(t, {}).get("resolved")) for _, res, _ in data if t in res]
        (always if all(outs) else never if not any(outs) else flips).append(t)
    print(f"{len(data)} passes · {len(tasks)} tasks: {len(always)} always resolved · {len(never)} never · "
          f"{len(flips)} flip\n")
    print(f"{'task':16} {'pass':12} {'out':>3} {'sec':>5} {'calls':>5} {'1st edit':>8} {'peak tok':>8} {'patch B':>7}")
    lose = {}; win_s = []; win_peak = []; win_calls = []; t_s = []; o_peak = []
    for t in flips:
        for name, res, tr in data:
            r = res.get(t)
            if not r:
                continue
            o = outcome(r)
            calls, peak, first = facts(tr.get(t, []))
            sec = r.get("duration_seconds") or 0
            print(f"{t:16} {name:12} {o:>3} {sec:5.0f} {calls:5} {first if first else '–':>8} {peak:8} "
                  f"{r.get('agent_patch_size') or 0:7}")
            if o == "✓":
                win_s.append(sec); win_peak.append(peak); win_calls.append(calls)
            else:
                lose[o] = lose.get(o, 0) + 1
                if o == "T": t_s.append(sec)
                if o == "O": o_peak.append(peak)
        print()
    tot = sum(lose.values())
    print("== flip tasks: losing passes by outcome:",
          " · ".join(f"{k} {v}" for k, v in sorted(lose.items(), key=lambda kv: -kv[1])), f"(n {tot})")
    print(f"   winning passes: seconds p50 {p50(win_s)} (max {max(win_s) if win_s else '–':.0f}) · calls p50 {p50(win_calls)}"
          f" · peak prompt p50 {p50(win_peak)}")
    if t_s:
        print(f"   timeout passes: seconds p50 {p50(t_s)}  (the cap is {CAP_S['4']} s at TIME_MIN=4, {CAP_S['8']} at 8)")
    if o_peak:
        print(f"   overflow passes: peak prompt p50 {p50(o_peak)}  (limit 32768 − max_output_tokens)")
    # how the always-resolved tasks look, for contrast
    a_s = [res[t].get("duration_seconds") or 0 for t in always for _, res, _ in data if t in res]
    a_c = [facts(tr.get(t, []))[0] for t in always for _, _, tr in data if t in tr]
    print(f"== always-resolved tasks ({len(always)}): seconds p50 {p50(a_s)} (max {max(a_s) if a_s else 0:.0f}) · calls p50 {p50(a_c)}")
    # never-resolved: did any pass at least run the tests on a patch?
    n_patch = sum(1 for t in never for _, res, _ in data if (res.get(t, {}).get("agent_patch_size") or 0) > 0)
    n_tot = sum(1 for t in never for _, res, _ in data if t in res)
    print(f"== never-resolved tasks ({len(never)}): passes with a non-empty patch {n_patch}/{n_tot}")


if __name__ == "__main__":
    main()
