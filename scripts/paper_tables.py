#!/usr/bin/env python3
"""Every table of the paper rebuilt from the local run directories → Markdown on stdout. Aggregates only.

    python3 scripts/paper_tables.py            # all tables
    python3 scripts/paper_tables.py 1 3 6      # some of them (1 ladder · 2 Opus vs 31B · 3 E4B vs 31B · 4 M2 ·
                                               # 5 probes · 6 adapters · 7 the E4B–31B gap)

Inputs are gitignored local dirs (runs/…, harness/tasks.jsonl). A missing one is printed, never skipped silently.
Computed from traces (`traces/trace_<task>.json`, the ATIF steps every run writes):
  tool calls     a call blocked by the budget (BudgetExceeded / TimeoutExceeded) is not counted, except in self-copy
  self-copy      share of consecutive tool-call pairs with the same name and arguments, pooled over tasks, all calls
  chained ops    shell operators per run_command (&&, ||, ;, |, newline), heredoc bodies and quoted text removed
  edit failure   edit_file calls rejected (not found, several matches, other), of those executed: budget-blocked and
                 missing-parameter calls excluded
  no-test pytest a pytest command whose exit code is 2, 4 or 5 (interrupted / usage / nothing collected)
  valid tasks    the gold patch passes (runs/gc_all) and no empty patch passes (disc_tasks.py's X_gold / X_empty)
Rows marked † come from GPU-box photos or the leaderboard: no per-task logs exist on the laptop, so they are the
constants in PHOTO / LB below, with their date. Never prints task text or task ids.
"""
import collections
import json
import math
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import disc_tasks as D          # noqa: E402  same X_gold / X_empty rules as the A/B split
from rulestats import load      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = "runs/"
LADDER = {"Opus 5.5": [R + "dev33_base_ceilsc", R + "dev33_base_ceil96"],
          "Gemma 4 31B": [R + "pod_all129/runs/dev33_base_all129_31b"],
          "Gemma 4 E4B": [R + "pod_all129_e4b/runs/dev33_base_all129_e4b"],
          "Gemma 4 E2B": [R + "pod_all129_e2b/runs/dev33_base_all129_e2b"]}
PAIRS = {  # (no adapter, adapter) run dirs of the same session
    "M2 (Opus, r32 α32), dev 33": (R + "pod_sft/runs/dev33_base_s1", R + "pod_sft/runs/dev33_base_m2s1"),
    "public E4B v6 (r8 α4), all 129": (R + "pod_borisv6noad/runs/dev33_base_all129_borisv6noad",
                                       R + "pod_borisv6/runs/dev33_base_all129_borisv6"),
    "public E2B (r16 α16), dev 33": (R + "pod_boris/dev33_base_bbases1", R + "pod_boris/dev33_base_bloras1"),
}
PROBES = {"healthy": R + "pod_boris/probe_copy.out", "onset": R + "pod_boris/probe_onset.out"}
PHOTO = {  # † per pass, read from box photos (run log rows of these dates); /129 or /51 as named
    "31B-own r32 α32 ep2, all 129 (2026-11-06)": ([14, 16], [8, 13]),
    "51-split no adapter, 2 passes of the all-129 runs (2026-11-06)": [13, 14],
    "51-split 31B-own α32 ep2 (2026-11-06)": [6, 13], "51-split 31B-own α16 ep2 (2026-11-06)": [10, 15, 13],
    "51-split 31B-own α8 ep2 (2026-11-06)": [17, 15, 12], "51-split 31B-own α16 ep1 (2026-11-06)": [19, 14, 14],
    "51-split 31B-own α8 ep1 (2026-11-06)": [13, 12, 15],
    "51-split r8 α8 s0 ep1 (2026-11-07)": [15, 11, 11], "51-split r8 α8 s0 ep2 (2026-11-07)": [12, 15, 14],
    "51-split r8 α8 s1 ep1 (2026-11-07)": [12, 15, 13], "51-split r8 α8 s1 ep2 (2026-11-07)": [15, 14, 18],
    "51-split base prompt (2026-11-08)": [12, 14, 16, 14], "51-split scaffold mech (2026-11-08)": [13, 12, 16, 13],
    "51-split scaffold all (2026-11-08)": [15, 14, 13, 14],
}
PHOTO_EDIT_FAIL = {"base": (77, 267), "mech": (60, 229), "all": (85, 245)}   # † rulestats p1+p2, 2026-11-08
LB = {  # † public leaderboard, 58 tasks (1 task ≈ 0.017)
    "31B, LB-0.24 files": 0.24, "31B, latest setting (2026-10-26)": 0.18,
    "E4B, latest 31B setting, no adapter (2026-11-02)": 0.03, "E4B + M2 (2026-10-29)": 0.01,
    "E4B public v6 (2026-11-01)": 0.03, "E4B public v7 (2026-11-03)": 0.03, "E4B + ours, kit data (2026-11-05)": 0.05,
    "E2B public adapter (2026-10-30)": 0.03,
}
GAP_STUDY = {  # † not recomputed here: per-task judgements of gap_study/A (classify + revise) and E (edit audit), 11-07
    "first divergence on the gap tasks": {"right file, wrong fix": 17, "localization": 11, "edit mechanics": 3,
                                          "pacing / compaction / truncation": 0},
    "E4B edit failures by kind (of 170)": {"miscopied though in context": "36%", "several matches": "32%",
                                           "stale (file already changed)": "13%", "call syntax": "11%"},
    "not-found rate by distance to the source output (E4B / 31B)": {"last output": "13% / 0–2%",
                                                                     "2–3 back": "38% / 0–2%", "4+ back": "45% / 0–2%"},
    "perfect editing, estimated tasks on 129": "+3 to +8",
}
TESTFILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)|\.(md|rst|txt)$")
WRITE_CMD = re.compile(r"sed\s+-i|\.write\(|write_text|cat\s*>|>>?\s*\S+\.py\b|\btee\b|patch\s+-p|git\s+apply|"
                       r"open\([^)]*['\"][wa]['\"]|\.replace\(.*\)\s*\)|perl\s+-pi")
PYTEST = re.compile(r"\bpytest\b|\bunittest\b")
REPRO = re.compile(r"python3?\s+(-c|\S+\.py)")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(?=\n|$)", re.S)
QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")       # may span lines (python3 -c '…')
OPS = re.compile(r"&&|\|\||;|\||\n")
MISSING = []


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def p50(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def obs(o):
    """a step's observation → the tool's JSON reply as a dict"""
    c = o.get("content") if isinstance(o, dict) else o
    try:
        d = json.loads(c) if isinstance(c, str) else c
    except ValueError:
        return {"raw": str(c)}
    if not isinstance(d, dict):
        return {"raw": str(d)}
    det = d.get("details")
    if isinstance(det, dict):
        for k in ("stdout", "stderr", "exit_code"):
            if d.get(k) is None and det.get(k) is not None:
                d[k] = det[k]
    return d


def edit_kind(d):
    if d.get("status") == "ok":
        return "ok"
    em = str(d.get("error_message") or d.get("error") or d.get("raw") or "")
    if d.get("error_type") in ("BudgetExceeded", "TimeoutExceeded"):
        return "budget"
    if "mandatory input parameters" in em:
        return "missing_param"
    if "not found. Ensure" in em or "old_string not found" in em:
        return "not_found"
    if "occurs" in em or "flexibly matches" in em:
        return "multiple"
    return "other"


def chained(cmd):
    c = QUOTED.sub("''", HEREDOC.sub("", (cmd or "").replace("\\\n", " ")))
    return len(OPS.findall(c.strip()))


def task_features(steps, result):
    calls = []
    for s in steps:
        if s.get("source") != "agent":
            continue
        for tc in s.get("tool_calls") or []:
            d = obs(s.get("observation"))
            calls.append({"name": tc.get("function_name"), "args": tc.get("arguments") or {}, "obs": d,
                          "blocked": d.get("error_type") in ("BudgetExceeded", "TimeoutExceeded")})
    sig = [(c["name"], json.dumps(c["args"], sort_keys=True)) for c in calls]
    eff = [c for c in calls if not c["blocked"]]
    f = {"resolved": bool(result.get("resolved")), "patch": result.get("agent_patch_size") or 0,
         "secs": result.get("duration_seconds") or 0,
         "error": str(result.get("error") or ""), "calls": len(eff),
         "pairs": max(0, len(sig) - 1), "copies": sum(a == b for a, b in zip(sig, sig[1:])),
         "submits": sum(c["name"] == "submit_patch" for c in calls), "reads": 0, "edit_new": 0,
         "cmd_len": [], "chain": [], "edits": collections.Counter(), "first_edit": None, "greps": 0, "repro": 0,
         "tests": []}
    for k, c in enumerate(eff, 1):
        n, a, d = c["name"], c["args"], c["obs"]
        is_edit = n in ("edit_file", "write_file")
        if n == "read_file":
            f["reads"] += 1
        elif n == "edit_file":
            f["edits"][edit_kind(d)] += 1
            f["edit_new"] += len(a.get("new_string") or "")
        elif n == "run_command":
            cmd = a.get("command") or ""
            f["cmd_len"].append(len(cmd))
            f["chain"].append(chained(cmd))
            if PYTEST.search(cmd):
                f["tests"].append(d.get("exit_code"))
            elif WRITE_CMD.search(cmd):
                is_edit = True
            if f["first_edit"] is None and not is_edit:
                f["repro"] += bool(REPRO.search(cmd)) and not PYTEST.search(cmd)
                f["greps"] += bool(re.search(r"\b(grep|rg|find)\b", cmd))
        if is_edit and f["first_edit"] is None:
            f["first_edit"] = k
    return f


def run_features(dirs):
    """{task: features} over one or more run dirs (several dirs = disjoint task subsets, e.g. Opus ceilsc + ceil96)"""
    out = {}
    for d in dirs:
        p = ROOT / d
        if not (p / "task_results.jsonl").exists():
            MISSING.append(d)
            continue
        results, traces = load(p)
        for t, r in results.items():
            out[t] = task_features(traces.get(t, []), r)
    return out


def task_sets():
    gold = {r["instance_id"]: r["resolved"] for r in rows(D.GOLD)}
    empty = set(D.BOX_EMPTY_PASS)
    for f in ROOT.glob(D.EMPTY_GLOB):
        empty |= {r["instance_id"] for r in rows(f) if r["resolved"]}
    for d, _ in D.RUNS:
        p = ROOT / d / "task_results.jsonl"
        if p.exists():
            empty |= {r["instance_id"] for r in rows(p)
                      if r["resolved"] and not r.get("agent_patch_size") and not r.get("error")}
    valid = {t for t, ok in gold.items() if ok and t not in D.BOX_GOLD_FAIL} - empty
    diff = {p.stem: json.loads(p.read_text()).get("difficulty") for p in (ROOT / "runs/labels").glob("*.json")}
    return gold, valid, diff


def md(head, body):
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for b in body:
        print("| " + " | ".join("" if x is None else str(x) for x in b) + " |")
    print()


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "—"


def copy_rate(F):
    c, p = sum(f["copies"] for f in F.values()), sum(f["pairs"] for f in F.values())
    return f"{100 * c / p:.1f}%" if p else "—"


def mcnemar(a, b):
    """exact two-sided McNemar on paired outcomes {task: bool}"""
    x = sum(a[t] and not b[t] for t in a if t in b)
    y = sum(b[t] and not a[t] for t in a if t in b)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, round(p, 2)


def t1_ladder(S, ctx):
    gold, valid, diff = ctx
    tiers = range(1, 6)
    print(f"### Table 1 · tasks resolved on the {len(valid)} valid tasks (of {len(gold)}), one run per model\n")
    head = ["Model"] + [f"{d} ({sum(diff.get(t) == d for t in valid)})" for d in tiers] + [f"All ({len(valid)})", "all 129"]
    md(head, [[m] + [sum(F[t]["resolved"] for t in valid if t in F and diff.get(t) == d) for d in tiers]
              + [sum(F[t]["resolved"] for t in valid if t in F), sum(f["resolved"] for f in F.values())]
              for m, F in S.items()])
    ms = list(S)
    sol = {m: {t for t in valid if t in S[m] and S[m][t]["resolved"]} for m in ms}
    print("Nesting on the valid tasks (smaller ⊂ larger):")
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            print(f"- {ms[j]} ⊂ {ms[i]}: {len(sol[ms[j]] & sol[ms[i]])}/{len(sol[ms[j]])} · {ms[i]} only {len(sol[ms[i]] - sol[ms[j]])}")
    o = S["Opus 5.5"]
    print(f"- Opus per task: tool calls max {max(f['calls'] for f in o.values())} · seconds max "
          f"{max(f['secs'] for f in o.values()):.0f} (the students' budget: 28 calls, 7 min)\n")


def profile_opus(F, tasks):
    return [p50([F[t]["calls"] for t in tasks]), p50([p50(F[t]["cmd_len"]) for t in tasks]),
            p50([statistics.mean(F[t]["chain"]) if F[t]["chain"] else None for t in tasks]),
            p50([F[t]["reads"] for t in tasks]), p50([F[t]["edit_new"] for t in tasks]), p50([F[t]["patch"] for t in tasks])]


def t2_opus_31b(S, ctx):
    _, valid, diff = ctx
    o, g = S["Opus 5.5"], S["Gemma 4 31B"]
    for scope, ts in (("valid", valid), ("all 129", set(o) & set(g))):
        both = sorted(t for t in ts if t in o and t in g and o[t]["resolved"] and g[t]["resolved"])
        only = sorted(t for t in ts if t in o and t in g and o[t]["resolved"] and not g[t]["resolved"])
        print(f"### Table 2 · per-task medians, Opus vs 31B ({scope}: both solved {len(both)}, Opus only {len(only)})\n")
        names = ["tool calls", "command length (chars)", "chained operations per command", "`read_file` calls",
                 "`edit_file` new characters", "patch bytes"]
        a, b, c, d = profile_opus(o, both), profile_opus(g, both), profile_opus(o, only), profile_opus(g, only)
        md(["Per-task median", "Opus (both)", "31B (both)", "Opus (Opus only)", "31B (Opus only)"],
           [[n, a[i], b[i], c[i], d[i]] for i, n in enumerate(names)])
        print(f"Opus-only by difficulty 1–5: {[sum(diff.get(t) == k for t in only) for k in range(1, 6)]}\n")


def profile_student(F):
    ef = sum(sum(f["edits"].values()) - f["edits"]["budget"] - f["edits"]["missing_param"] for f in F.values())
    eok = sum(f["edits"]["ok"] for f in F.values())
    tests = [x for f in F.values() for x in f["tests"]]
    return [p50([f["calls"] for f in F.values()]), p50([f["first_edit"] for f in F.values()]),
            p50([f["greps"] for f in F.values()]), sum(f["repro"] > 0 for f in F.values()),
            f"{pct(ef - eok, ef)} ({ef - eok}/{ef})", f"{sum(x in (2, 4, 5) for x in tests)}/{len(tests)}",
            sum(bool(f["tests"]) and f["tests"][-1] not in (0, None) and f["submits"] > 0 for f in F.values()),
            copy_rate(F)]


def t3_e4b_31b(S, ctx):
    print("### Table 3 · all 129 tasks, E4B vs 31B (and E2B)\n")
    names = ["tool calls (median)", "first edit at call (median)", "greps before first edit (median)",
             "tasks with a reproduction before the first edit", "`edit_file` failure rate",
             "pytest calls that ran no test", "tasks submitted after a red or empty last test", "self-copy rate"]
    cols = [profile_student(S[m]) for m in ("Gemma 4 E4B", "Gemma 4 31B", "Gemma 4 E2B")]
    md(["", "E4B", "31B", "E2B"], [[n] + [c[i] for c in cols] for i, n in enumerate(names)])


def t4_m2(S, ctx):
    base, m2 = (run_features([d]) for d in PAIRS["M2 (Opus, r32 α32), dev 33"])
    print("### Table 4 · dev split (33 tasks), base E4B vs + M2\n")
    row = lambda F: [sum(f["resolved"] for f in F.values()), sum("imeout" in f["error"] for f in F.values()),
                     sum(f["submits"] for f in F.values()), copy_rate(F)]
    a, b = row(base), row(m2)
    md(["", "base E4B", "+ M2"], [[n, a[i], b[i]] for i, n in enumerate(
        ["resolved", "tasks ended by the time budget", "`submit_patch` calls", "self-copy rate"])]
       + [["public leaderboard †", LB["E4B, latest 31B setting, no adapter (2026-11-02)"], LB["E4B + M2 (2026-10-29)"]]])


def t5_probes(S, ctx):
    print("### Table 5 · replayed histories, share of next calls that copy the previous call (E2B, one server)\n")
    body = []
    for name, f in PROBES.items():
        p = ROOT / f
        if not p.exists():
            MISSING.append(f)
            continue
        head = next((l for l in p.read_text().splitlines() if l.startswith(("states", "onset states"))), "")
        n = re.search(r"(\d+)", head)
        got = {}
        for l in p.read_text().splitlines():
            m = re.match(r"(base|lora)\s+([\d.]+)\s+(kept|stripped)\s+(\d+)\s+(\d+)\s+\(\s*(\d+)%\)", l)
            if m:
                got[(m[1], m[2], m[3])] = f"{m[6]}% ({m[5]}/{m[4]})"
        for temp in ("0.2", "1.0"):
            body.append([f"{name} states ({n[1] if n else '?'}), temperature {temp}",
                         got.get(("base", temp, "kept")), got.get(("lora", temp, "kept"))])
    md(["Histories replayed", "base", "adapter"], body)


def t6_adapters(S, ctx):
    print("### Table 6 · adapters against a no-adapter control (resolved; † = box photo or leaderboard)\n")
    body = []
    for name, (d0, d1) in PAIRS.items():
        a, b = run_features([d0]), run_features([d1])
        if not a or not b:
            continue
        x, y, p = mcnemar({t: f["resolved"] for t, f in a.items()}, {t: f["resolved"] for t, f in b.items()})
        body.append([name, sum(f["resolved"] for f in a.values()), sum(f["resolved"] for f in b.values()),
                     f"discordant {x} vs {y}, exact p {p}", f"copy {copy_rate(a)} → {copy_rate(b)}"])
    for name, v in PHOTO.items():
        if isinstance(v, tuple):
            body.append([name + " †", " · ".join(map(str, v[0])), " · ".join(map(str, v[1])), "", ""])
        else:
            body.append([name + " †", "", f"{' · '.join(map(str, v))} (mean {statistics.mean(v):.1f})", "", ""])
    for name, v in LB.items():
        body.append([name + " † LB", "", v, "", ""])
    md(["Adapter / arm", "no adapter", "adapter", "paired test", "self-copy"], body)
    print("Scaffold-arm edit failures † (rulestats, 2 passes): "
          + " · ".join(f"{k} {pct(a, b)} ({a}/{b})" for k, (a, b) in PHOTO_EDIT_FAIL.items()) + "\n")


def gold_src(patch):
    files = set()
    for line in (patch or "").splitlines():
        if line.startswith("+++ "):
            f = line[4:].split("\t")[0].removeprefix("b/")
            if f != "/dev/null" and not TESTFILE.search(f):
                files.add(f)
    return files


def t7_gap(S, ctx):
    _, valid, _ = ctx
    print("### § 7 · the E4B–31B gap\n")
    tasks = {t["instance_id"]: t for t in rows(ROOT / "harness/tasks.jsonl")}
    e4b, g31 = S["Gemma 4 E4B"], S["Gemma 4 31B"]
    gap = [t for t in valid if t in e4b and t in g31 and g31[t]["resolved"] and not e4b[t]["resolved"]]
    print(f"- gap tasks (31B ✓, E4B ✗) on the valid tasks: {len(gap)}; E4B ✓ and 31B ✗: "
          f"{sum(e4b[t]['resolved'] and not g31[t]['resolved'] for t in valid if t in e4b and t in g31)}")
    for m, F, d in (("E4B", e4b, LADDER["Gemma 4 E4B"][0]), ("31B", g31, LADDER["Gemma 4 31B"][0])):
        land = []
        for t in valid:
            pp = ROOT / d / "patches" / f"{t}.patch"
            mine = gold_src(pp.read_text(errors="replace")) if pp.exists() else set()
            if mine & gold_src(tasks[t]["patch"]):
                land.append(t)
        res = sum(F[t]["resolved"] for t in land if t in F)
        print(f"- {m}: final patch touches a gold source file in {len(land)}/{len(valid)} valid tasks; "
              f"resolved given that {res}/{len(land)} = {pct(res, len(land))}")
    order = collections.Counter()
    for p in (ROOT / LADDER["Gemma 4 E4B"][0] / "traj").glob("*.jsonl"):
        if p.name.endswith(".tools.jsonl"):
            continue
        for line in p.read_text().splitlines():
            for tc in ((json.loads(line).get("resp") or {}).get("tool_calls") or []):
                raw = tc.get("args")
                if tc.get("name") == "edit_file" and isinstance(raw, str) and '"old_string"' in raw and '"new_string"' in raw:
                    order["new first" if raw.index('"new_string"') < raw.index('"old_string"') else "old first"] += 1
    print(f"- E4B edit_file argument order as generated: {dict(order)}")
    for k, v in GAP_STUDY.items():
        print(f"- {k} †: {v}")
    print()


TABLES = {"1": t1_ladder, "2": t2_opus_31b, "3": t3_e4b_31b, "4": t4_m2, "5": t5_probes, "6": t6_adapters, "7": t7_gap}


def main():
    want = [a for a in sys.argv[1:] if a in TABLES] or list(TABLES)
    S = {m: run_features(ds) for m, ds in LADDER.items()}
    ctx = task_sets()
    for k in want:
        TABLES[k](S, ctx)
    if MISSING:
        print("MISSING (tables above lack them): " + ", ".join(sorted(set(MISSING))))


if __name__ == "__main__":
    main()
