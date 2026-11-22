#!/usr/bin/env python3
"""B-11r: does the teacher recall the upstream fix? One block per directory of teacher trajectories, and a paired
comparison of every further directory with the first.

    python3 scripts/teach_recall.py runs/teach_h2@pilot30 runs/teach_r1 [...] --ids splits/recall20.ids [--list]     # DIR@prefix = only those runs
    python3 scripts/teach_recall.py selftest

The measure (fixed on 2026-11-22, before any arm was run; it agreed with eleven trajectories read in full, 9 with and
2 without): a trajectory HAS RECALL when the thought or the visible text of at least one tool-call turn holds
  · a memory phrase:    I recall / I remember / let me recall / from memory
  · an upstream phrase: upstream fix|commit|PR|implementation|repo|code|version|change|test · how upstream ·
                        actual upstream|commit|PR · current|modern <word> master|main
  · a PR / issue number that is not in the task text.
Per turn it finds fewer than a reader does (47 against 60), so the reading is per trajectory. It does not measure
whether the FIX was taken from memory: that needs a reader.
Also printed: trajectories that mention the hidden tests · thought characters per tool-call turn · compactions ·
resolved · cost · what was SENT (calls whose system prompt held the hidden-tests clause) against what was LOGGED
(trajectories whose own system prompt holds it — the data must keep it).
Reads only; no model, nothing written."""
import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teach_audit as TA
import teach_reminder

MEM = re.compile(r"\bI (?:\w+ )?(?:recall|remember)\b|\blet me recall\b|\bfrom memory\b", re.I)
UP = re.compile(r"\bupstream (?:fix|commit|PR|implementation|repo|code|version|change|test)|\bhow upstream\b|\bactual (?:upstream|commit|PR)\b"
                r"|\b(?:current|modern) \w+ (?:master|main)\b", re.I)
PRN = re.compile(r"(?:PR|pull request|issue) ?#(\d+)|#(\d{3,5})\b", re.I)
HID = re.compile(r"\bhidden tests?\b", re.I)


def turn_has_recall(text, task_numbers):
    return bool(MEM.search(text) or UP.search(text) or ({a or b for a, b in PRN.findall(text)} - task_numbers))


def score(path):
    """one trajectory file → facts"""
    steps, f = TA.read_attempt(path, [])
    nums = set(re.findall(r"#(\d+)", f["task_text"] or "")); rec = []; hid = 0; chars = []; summaries = 0
    for x in steps:
        summaries += x["kind"] == "summary"
        if x["kind"] != "tool":
            continue
        t = (x["th"] or "") + "\n" + (x["tx"] or ""); chars.append(len(x["th"] or ""))
        if turn_has_recall(t, nums):
            rec.append(x["n"])
        hid += bool(HID.search(t))
    return dict(recall_calls=rec, hidden=hid, turns=len(chars), chars=sum(chars), summaries=summaries,
                clause_logged=teach_reminder.HIDDEN_CLAUSE in (f["system"] or ""))


def read_dir(out, ids, name):
    rows, _ = TA.load_rows(out, ids, name); R = {}; runs = set()
    for r in rows:
        if r["state"] != "done":
            continue
        run = os.path.join(out, "runs", os.path.basename(r.get("run") or "?")); p = os.path.join(run, "traj", r["task"] + ".jsonl")
        if not os.path.exists(p):
            continue
        s = score(p)
        if not s["turns"]:
            continue
        s.update(task=r["task"], attempt=r["attempt"], resolved=bool(r.get("resolved")), usd=r.get("usd") or 0.0,
                 setting=" · ".join(str(x) for x in (r.get("teacher") or "teacher ?", "cut " + r.get("sys_cut", "off"), "think " + r.get("think", "?"),
                                                    "reminder " + r.get("reminder", "off") + "@" + r.get("reminder_place", "end"), "hist " + r.get("hist", "off"))))
        R[(r["task"], r["attempt"])] = s; runs.add(os.path.basename(r.get("run") or ""))
    sent = [0, 0, 0]          # calls · with the clause in the system prompt · rows that record it
    for rn in sorted(runs):
        for line in (open(os.path.join(out, "calls", rn + ".jsonl")) if os.path.exists(os.path.join(out, "calls", rn + ".jsonl")) else []):
            c = json.loads(line)
            if c.get("err"):
                continue
            sent[0] += 1
            if "hidden_clause" in c:
                sent[2] += 1; sent[1] += bool(c["hidden_clause"])
    return R, sent


def block(out, R, sent):
    v = list(R.values()); n = len(v)
    if not n:
        print(f"== {out}: no done trajectory"); return
    turns = sum(s["turns"] for s in v)
    print(f"== {out}")
    print("   setting(s): " + " ‖ ".join(sorted({s["setting"] for s in v})))
    print(f"   trajectories {n} · tasks {len({s['task'] for s in v})} · resolved {sum(s['resolved'] for s in v)} · cost ${sum(s['usd'] for s in v):.2f} (mean ${sum(s['usd'] for s in v) / n:.3f})")
    print(f"   WITH RECALL            {sum(bool(s['recall_calls']) for s in v):3d} of {n}   (tool-call turns with recall {sum(len(s['recall_calls']) for s in v)} of {turns})")
    print(f"   mention hidden tests   {sum(s['hidden'] > 0 for s in v):3d} of {n}   (turns {sum(s['hidden'] for s in v)} of {turns})")
    print(f"   thought chars per tool-call turn: mean {sum(s['chars'] for s in v) / max(1, turns):.0f} · median over trajectories {statistics.median(s['chars'] / s['turns'] for s in v):.0f} · compactions {sum(s['summaries'] for s in v)}")
    print(f"   sent: hidden-tests clause in the system prompt of {sent[1]} of {sent[2]} recorded calls ({sent[0]} calls in all; older rows do not record it)"
          f" · logged: clause in the trajectory's own system prompt {sum(s['clause_logged'] for s in v)} of {n} (must be all: the data keeps the prompt)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--ids", default=""); ap.add_argument("--name", default="")
    ap.add_argument("--list", action="store_true", help="one line per task in the paired comparison")
    a = ap.parse_args()
    ids = set(open(a.ids).read().split()) if a.ids else None
    print(f"teach_recall · dirs {len(a.dirs)} · ids {a.ids or 'all'} ({len(ids) if ids else '-'}) · name prefix {a.name or 'any'} · measure: memory phrase | upstream phrase | PR / issue number not in the task text, per trajectory")
    D = []
    for i, d in enumerate(a.dirs):
        path, _, pref = d.partition("@")      # DIR@prefix: only the runs of that directory whose name starts with prefix
        R, sent = read_dir(path, ids, pref or a.name); D.append((d, R)); block(d, R, sent)
    for d, R in D[1:]:
        base = D[0][1]; common = sorted(set(k[0] for k in base) & set(k[0] for k in R))
        has = lambda S, t: any(bool(s["recall_calls"]) for k, s in S.items() if k[0] == t)
        tab = [(has(base, t), has(R, t)) for t in common]
        print(f"== paired on {len(common)} common tasks: {D[0][0]} → {d}")
        print(f"   recall in both {tab.count((True, True))} · only in the first {tab.count((True, False))} · only in this {tab.count((False, True))} · in neither {tab.count((False, False))}"
              f" → {sum(x for x, _ in tab)} → {sum(y for _, y in tab)} of {len(common)}")
        if a.list:
            for t, (x, y) in zip(common, tab):
                print(f"     {t:48s} {'recall' if x else '-':7s} → {'recall' if y else '-'}")


def selftest():
    T = {"12"}
    pos = ["Let me check how upstream fixed this. The actual fix (PR #1383?) used /x.", "I recall the change was in dist.py.", "I now recall the actual implementation.",
           "Current pygfx master has scale_x.", "This is from memory, not from the code.", "I genuinely remember modern flax main doing this.", "see issue #4092 for it"]
    neg = ["The fix for issue #12 is in core.py.", "Let me apply the actual fix to the source file.", "The test passed; now check git status.", "upstream of the data flow is the parser",
           "Call 3 of 36: read lines 1-80.", "color #fff and offset 0x1234", "The hidden tests are written from the issue."]
    bad = [t for t in pos if not turn_has_recall(t, T)] + [t for t in neg if turn_has_recall(t, T)]
    assert not bad, bad
    assert HID.search(neg[-1]) and teach_reminder.cut_text("hidden") == teach_reminder.HIDDEN_CLAUSE and teach_reminder.cut_text("0") == "" and teach_reminder.cut_ident("hidden") == "hidden"
    print(f"teach_recall selftest ok: {len(pos)} planted recalls found, {len(neg)} clean texts left alone")


if __name__ == "__main__":
    main()
