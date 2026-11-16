"""Does the model copy its previous turn? Counts over the trajectories of scripts/verified_box.sh runs (B-9 part 2 / 3).
  python3 scripts/copy_stats.py runs/verified/vf_q9_th runs/verified/vf_q9_mc [--ids runs/verified/paired_….json]
Reads every <id>.traj.json under each run dir (mini-swe-agent: messages with role assistant, `reasoning_content`,
`tool_calls`). A turn = one assistant message; its command = the arguments of its tool calls. Prints per run, for 📷:
  exact copies = turn pairs (k-1, k) with the same thought AND the same command (share of all pairs; trajectories with
  one; the call of the first copy p10 / p50) · same command back-to-back, any thought (pairs; longest run per trajectory
  p50 / p90 / max) · thought loops (a sentence ≥ 5 times in one thought, or a thought over 12,000 characters) ·
  turns without a tool call · thought characters p50 · exits. No task ids or text are printed. Standard library only."""
import argparse
import collections
import glob
import json
import os
import re


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def loops(th):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", th) if len(x.strip()) > 20]
    return len(th) > 12000 or (bool(s) and collections.Counter(s).most_common(1)[0][1] >= 5)


def prose_loops(th):
    """the same rule without code: fenced blocks dropped, sentences of 30+ characters that do not look like a code line.
    (The first rule also counts repeated code lines such as `return True`; B-9 part 3, 2026-11-16.)"""
    if len(th) > 12000:
        return True
    t = re.sub(r"```.*?```", "", th, flags=re.S)
    s = [x.strip() for x in re.split(r"(?<=[.!?\n])\s+", t) if len(x.strip()) >= 30]
    s = [x for x in s if not x.startswith(("-", "`", "return", "if ", "elif", "else"))]
    return bool(s) and collections.Counter(s).most_common(1)[0][1] >= 5


def stats(d, only=None):
    C, exits, first, runs, thl = collections.Counter(), collections.Counter(), [], [], []
    for f in sorted(glob.glob(os.path.join(d, "**", "*.traj.json"), recursive=True)):
        iid = os.path.basename(f)[:-len(".traj.json")]
        if only is not None and iid not in only:
            continue
        t = json.load(open(f))
        exits[(t.get("info") or {}).get("exit_status") or "?"] += 1
        asst = [m for m in t.get("messages") or [] if m.get("role") == "assistant"]
        turns = [((m.get("reasoning_content") or "").strip(),
                  json.dumps([(tc.get("function") or {}).get("arguments") for tc in m.get("tool_calls") or []]),
                  bool(m.get("tool_calls")))
                 for m in asst]
        C["turns with text beside a tool call"] += sum(1 for m in asst if m.get("tool_calls") and isinstance(m.get("content"), str) and m["content"].strip())
        C["turns with several tool calls"] += sum(1 for m in asst if len(m.get("tool_calls") or []) > 1)
        C["trajectories"] += 1; C["turns"] += len(turns)
        fc, run, best, any_loop, any_prose = None, 1, 1, False, False
        for k, (th, cmd, has) in enumerate(turns):
            thl.append(len(th)); C["turns without a tool call"] += not has
            if loops(th):
                C["turns with a thought loop"] += 1; any_loop = True
            if prose_loops(th):
                C["turns with a prose loop"] += 1; any_prose = True
            C["thoughts over 3000 chars"] += len(th) > 3000; C["thoughts over 12000 chars"] += len(th) > 12000
            if k == 0:
                continue
            C["pairs"] += 1
            pth, pcmd, phas = turns[k - 1]
            same_cmd = has and phas and cmd == pcmd
            C["same command pairs"] += same_cmd
            run = run + 1 if same_cmd else 1; best = max(best, run)
            if same_cmd and th == pth:
                C["exact copy pairs"] += 1
                if fc is None:
                    fc = k + 1
        runs.append(best); C["trajectories with a thought loop"] += any_loop; C["trajectories with a prose loop"] += any_prose
        if fc is not None:
            first.append(fc); C["trajectories with an exact copy"] += 1
    n, p = C["trajectories"], max(C["pairs"], 1)
    print(f"== copy stats {os.path.basename(os.path.normpath(d))}: {n} trajectories · {C['turns']} turns · {C['pairs']} turn pairs")
    print(f"   exact copies of the previous turn (thought + command): {C['exact copy pairs']} pairs = {C['exact copy pairs'] / p:.1%}"
          f" · in {C['trajectories with an exact copy']} / {n} trajectories · first copy at call p10 {q(first, .1)} p50 {q(first, .5)}")
    print(f"   same command back-to-back: {C['same command pairs']} pairs = {C['same command pairs'] / p:.1%}"
          f" · longest run per trajectory p50 {q(runs, .5)} p90 {q(runs, .9)} max {max(runs) if runs else None}")
    print(f"   thought loops: {C['turns with a thought loop']} turns in {C['trajectories with a thought loop']} / {n} trajectories"
          f" · turns without a tool call {C['turns without a tool call']} · thought chars p50 {q(thl, .5)} p90 {q(thl, .9)}")
    T = max(C["turns"], 1)
    print(f"   prose loops (code left out): {C['turns with a prose loop']} turns in {C['trajectories with a prose loop']} / {n} trajectories"
          f" · thoughts over 3,000 chars {C['thoughts over 3000 chars']} = {C['thoughts over 3000 chars'] / T:.1%} · over 12,000 chars {C['thoughts over 12000 chars']}")
    print(f"   turns with text beside a tool call {C['turns with text beside a tool call']} = {C['turns with text beside a tool call'] / T:.1%}"
          f" · turns with several tool calls {C['turns with several tool calls']} = {C['turns with several tool calls'] / T:.1%}")
    print(f"   exits {dict(exits.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (runs/verified/<TAG>)")
    ap.add_argument("--ids", default=None, help="paired.py's JSON: only the tasks in both runs")
    a = ap.parse_args()
    only = None
    if a.ids:
        j = json.load(open(a.ids))
        only = set(j["both"]) | set(j["only_a"]) | set(j["only_b"]) | set(j["neither"])
        print(f"copy_stats · only the {len(only)} tasks of {a.ids}")
    else:
        print("copy_stats · all trajectories under each run dir (--ids = only the paired tasks)")
    for d in a.runs:
        stats(d, only)
    print("📷 the == blocks")


if __name__ == "__main__":
    main()
