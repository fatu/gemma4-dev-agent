#!/usr/bin/env python3
"""How much room is there in E4B's own repeat passes? (BOX_PLAN B-5 step 1; CPU, no model, aggregates only.)

    python3 scripts/stab.py runs/dev33_base_e{1..3} runs/dev33_base_f{1..8}      # 📷 every line

Part 1 STABILITY: tasks resolved in every pass / in none / in some (flip), a histogram of "resolved in k of N passes",
and the size of the data a rejection-sampling fine-tune would have: resolved runs on flip tasks.
Part 2 SUBMIT: every run's first submit_patch, classed by the last run_command since the last edit —
  none          no command since the last edit
  clean         exit 0 and no failure text
  hidden fail   exit 0 but the output shows a failure (a pipe such as `| tail` hides pytest's exit code)
  exit != 0     the command failed
and per class how many of those runs were resolved, over all tasks and over flip tasks only (always / never tasks have
no room). If runs that submit on a failure are resolved about as often as the others, holding the submission back
cannot gain much. Prints no task ids unless --ids is given.
"""
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rulestats import load   # noqa: E402

FAIL_RE = re.compile(r"\b\d+ (failed|errors?)\b|Traceback \(most recent call last\)|\b(Syntax|Import|Indentation|ModuleNotFound)Error\b"
                     r"|no tests ran")
CLASSES = ["none", "clean", "hidden fail", "exit != 0"]


def submit_class(steps):
    """class of the run's first submit_patch, or None when it never submits"""
    calls = []
    for s in steps:
        if s.get("source") != "agent":
            continue
        obs = s.get("observation") or {}
        for c in s.get("tool_calls") or []:
            calls.append((c["function_name"], obs.get("content") if isinstance(obs, dict) else str(obs)))
    names = [n for n, _ in calls]
    if "submit_patch" not in names:
        return None
    k = names.index("submit_patch")
    le = max([j for j in range(k) if names[j] in ("edit_file", "write_file")], default=-1)
    cmds = [o for n, o in calls[le + 1:k] if n == "run_command"]
    if not cmds:
        return "none"
    raw = cmds[-1] if isinstance(cmds[-1], str) else json.dumps(cmds[-1])
    try:
        d = json.loads(raw)
        ec = d.get("exit_code", (d.get("details") or {}).get("exit_code"))
        text = " ".join(str(x) for x in (d.get("stdout"), d.get("stderr"), d.get("error_message"), json.dumps(d.get("details") or "")))
        if d.get("status") == "error" and ec is None:
            ec = 1
    except (ValueError, AttributeError):
        ec, text = None, raw
    if ec not in (0, None):
        return "exit != 0"
    return "hidden fail" if FAIL_RE.search(text) else "clean"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ids = "--ids" in sys.argv
    runs = [pathlib.Path(a) for a in args] or sys.exit(__doc__)
    data = [(r.name, *load(r)) for r in runs]
    print(f"stab · {len(runs)} passes [{' '.join(r.name for r in runs)}] · task ids printed: {'yes' if ids else 'no'} · "
          f"failure text = /{FAIL_RE.pattern[:60]}…/")
    for name, res, tr in data:
        print(f"  {name}: {sum(bool(x.get('resolved')) for x in res.values())} / {len(res)} resolved · {len(tr)} traces")
    tasks = sorted({t for _, res, _ in data for t in res})
    k_of = {t: sum(bool(res.get(t, {}).get("resolved")) for _, res, _ in data) for t in tasks}
    n_of = {t: sum(t in res for _, res, _ in data) for t in tasks}
    always = [t for t in tasks if k_of[t] == n_of[t]]; never = [t for t in tasks if k_of[t] == 0]
    flip = [t for t in tasks if 0 < k_of[t] < n_of[t]]
    hist = collections.Counter(k_of.values())
    print(f"== STABILITY {len(tasks)} tasks: always {len(always)} · never {len(never)} · flip {len(flip)}")
    print("   resolved in k passes → tasks: " + " · ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
    win = sum(k_of[t] for t in flip); tot = sum(n_of[t] for t in flip)
    print(f"   flip tasks: {win} resolved runs of {tot} ({100 * win / max(tot, 1):.0f} %) — the data a fine-tune on E4B's own "
          f"successes would have · flip tasks resolved in at most half the passes: {sum(1 for t in flip if 2 * k_of[t] <= n_of[t])}")
    best = len(always) + len(flip); mean = sum(k_of.values()) / max(len(data), 1)
    print(f"   mean per pass {mean:.1f} · resolved at least once {best} (the ceiling if every flip task always went right: "
          f"+{best - mean:.1f} tasks per pass)")
    if ids:
        print("   flip: " + " ".join(f"{t}({k_of[t]}/{n_of[t]})" for t in flip))
    fl = set(flip)
    for label, keep in (("all tasks", lambda t: True), ("flip tasks only", lambda t: t in fl)):
        C = collections.defaultdict(lambda: [0, 0]); nosub = [0, 0]; notrace = 0
        for _, res, tr in data:
            for t, r in res.items():
                if not keep(t):
                    continue
                if t not in tr:
                    notrace += 1; continue
                c = submit_class(tr[t]); slot = C[c] if c else nosub
                slot[0] += 1; slot[1] += bool(r.get("resolved"))
        print(f"== SUBMIT ({label}) first submit_patch by the last command since the last edit → runs · resolved")
        for c in CLASSES:
            n, k = C[c]; print(f"   {c:12} {n:5} runs · resolved {k:4} ({100 * k / max(n, 1):.0f} %)")
        print(f"   {'no submit':12} {nosub[0]:5} runs · resolved {nosub[1]:4} · runs without a trace (not counted): {notrace}")


if __name__ == "__main__":
    main()
