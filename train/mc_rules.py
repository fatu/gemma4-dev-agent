#!/usr/bin/env python3
"""B-10 step 4: the seven-signal rule filter of the MetaCurate paper (arXiv 2610.23134, App. A "Rule filter", after the
Nemotron 3 Ultra report), on mini-swe-agent trajectories. No model, no execution labels.

    .venv-eval/bin/python train/mc_rules.py --raw <dir with the pool's *.parquet> [--turns msg|call] [--max-ctx 30720]

The definitions, regexes and the six thresholds are a port of `filters/initial_ranking_nometa/rules.py` of
github.com/furiosa-ai/MetaCurate (MIT licence, commit 90e8c8f, read 2026-11-13): same command canonicalizer, same
edit / test / revert patterns, same thresholds. Not published by the paper, so chosen here and printed:
  * how a bash tool-call message becomes a "turn": --turns msg = one turn per assistant message, its commands joined by a
    newline (default) · call = one turn per tool call. A message without a command is a malformed turn in both.
  * which signals remove a trajectory: the paper's text says signals 1–4 reject and 5–7 flag; the released file rejects
    on all seven. Both counts are printed.
Counts only: per signal (all / resolved / unresolved), survivors, tasks, sources; a context-fit proxy from the teacher's
own usage record (last call's prompt + completion tokens ≤ --max-ctx; teacher tokens, not Gemma tokens).
"""
import argparse
import collections
import glob
import json
import re

# --- thresholds (rules.py:31-36 of the released file) ---
EDIT_TEST_LOOP_MAX_RETURNS = 8
EDIT_TEST_LOOP_MAX_REVERTS = 3
LOST_EXPLORATION_MIN_ACTIONS = 20
LOST_EXPLORATION_EXPLORE_SHARE = 0.85
LOST_EXPLORATION_MAX_EDITS = 2
MALFORMED_RATE_MAX = 0.10

WRAPPER_HEADS = {"bash", "sh", "zsh", "env", "sudo", "time", "nohup", "xargs"}
DISALLOWED_GIT = {"push", "pull", "fetch", "clone", "cherry-pick", "reflog", "fsck", "remote", "ls-remote"}
EDIT_RE = re.compile(
    r"(sed\s+(-[a-zA-Z]+\s+)*-i"
    r"|cat\s*>{1,2}\s*\S"
    r"|tee\s+(-a\s+)?\S"
    r"|\bgit\s+apply\b"
    r"|\bpatch\s+-p\d"
    r"|apply_patch"
    r"|(echo|printf)\s+.{0,400}?>{1,2}\s*\S"
    r"|open\(\s*[^)]{0,200}['\"][wa]\+?['\"]"
    r"|\.write_text\(|\.write\("
    r")", re.DOTALL)
HEREDOC_WRITE_RE = re.compile(r"(cat|tee)\s*(>{1,2}\s*\S+\s*)?<<|>{1,2}\s*\S+\s*<<")
GIT_DISALLOWED_RE = re.compile(r"\bgit\s+(?:-[^\s]+\s+)*(push|pull|fetch|clone|cherry-pick|reflog|fsck|remote|ls-remote)\b")
GIT_REVERT_RE = re.compile(r"\bgit\s+(checkout\s+(\S+\s+)?--\s|restore\s|stash\b(?!\s+list)|reset\s+--hard)")
TEST_TOKENS = {"pytest", "py.test", "nosetests", "nose2", "nose", "tox", "unittest"}
PYRUN_TOKENS = {"python", "python_inline"}
SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\n")
DEBUG_PATTERNS = [re.compile(r"\bprint\("), re.compile(r"\bimport\s+pdb\b|\bpdb\.set_trace\b"), re.compile(r"\bbreakpoint\(\)")]

SIGNALS = ["no_valid_submission", "disallowed_git", "edit_test_loop", "lost_in_exploration",      # reject in the paper
           "tool_call_hygiene", "debug_artifacts", "no_verification"]                              # flag in the paper
HARD = set(SIGNALS[:4])


def canonical(cmd, depth=0):
    """A raw shell segment → an action token (pytest, python, git_<sub>, the head word …)."""
    if not cmd or depth > 3:
        return "shell"
    s = cmd.strip()
    if not s:
        return "shell"
    while re.match(r"^[A-Z_][A-Z_0-9]*=\S+\s+", s):
        s = s.split(None, 1)[1] if " " in s else s
    tokens = s.split()
    if not tokens:
        return "shell"
    head = tokens[0].lower().lstrip("./")
    if head in {"cd", "pushd", "popd", "chdir", "pwd", "true"} and depth < 3:
        for sep in (" && ", "&&", " ; ", ";"):
            if sep in s:
                rest = s.split(sep, 1)[1].strip()
                if rest:
                    return canonical(rest, depth + 1)
    if head in WRAPPER_HEADS and len(tokens) >= 2:
        m = re.search(r"""-c\s+(?:"([^"]+)"|'([^']+)'|(\S+))""", s)
        if m:
            return canonical(next(g for g in m.groups() if g), depth + 1)
        if head in {"env", "sudo", "time", "nohup", "xargs"}:
            return canonical(" ".join(tokens[1:]), depth + 1)
        return head
    if head in {"python", "python3", "python2", "py"}:
        if len(tokens) >= 3 and tokens[1] == "-m":
            return tokens[2].split(".")[0].lower()
        if len(tokens) >= 2 and tokens[1] == "-c":
            return "python_inline"
        if len(tokens) >= 3 and tokens[1] == "setup.py" and tokens[2] == "test":
            return "pytest"
        return "python"
    if head == "git" and len(tokens) >= 2:
        return f"git_{tokens[1].lower()}"
    m = re.match(r"([A-Za-z_][\w\-\.]*)", head)
    return m.group(1).lower() if m else "shell"


def classify(cmd):
    tokens = [canonical(seg.strip()) for seg in SEGMENT_SPLIT_RE.split(cmd) if seg.strip()]
    return {"test": any(t in TEST_TOKENS for t in tokens),
            "pyrun": any(t in PYRUN_TOKENS for t in tokens),
            "edit": bool(EDIT_RE.search(cmd) or HEREDOC_WRITE_RE.search(cmd)),
            "git_disallowed": bool(GIT_DISALLOWED_RE.search(cmd)) or any(
                t.startswith("git_") and t[4:] in DISALLOWED_GIT for t in tokens),
            "git_revert": bool(GIT_REVERT_RE.search(cmd))}


def signals(turns, submission):
    """turns = the content of every assistant turn (a command, or "" for a turn without one) → the fired signals."""
    cmds = [t for t in turns if t.strip()]
    n_assistant, n_malformed = len(turns), len(turns) - len(cmds)
    flags = [classify(c) for c in cmds]
    seq = "".join("E" if f["edit"] else "T" if f["test"] else "P" if f["pyrun"] else "R" for f in flags)
    core = [c for c in seq if c in "ET"]
    n_actions, n_edits, n_tests = len(flags), sum(f["edit"] for f in flags), sum(f["test"] for f in flags)
    returns = sum(1 for a, b in zip(core, core[1:]) if a == "T" and b == "E")
    reverts = sum(f["git_revert"] for f in flags)
    explore = seq.count("R") / n_actions if n_actions else 1.0
    malformed = n_malformed / n_assistant if n_assistant else 1.0
    sub = submission or ""
    added = [l for l in sub.splitlines() if l.startswith("+") and not l.startswith("+++")]
    out = []
    if not sub.strip().startswith("diff --git"):
        out.append("no_valid_submission")
    if any(f["git_disallowed"] for f in flags):
        out.append("disallowed_git")
    if returns >= EDIT_TEST_LOOP_MAX_RETURNS or reverts >= EDIT_TEST_LOOP_MAX_REVERTS:
        out.append("edit_test_loop")
    if n_edits == 0 or (n_actions >= LOST_EXPLORATION_MIN_ACTIONS and explore >= LOST_EXPLORATION_EXPLORE_SHARE
                        and n_edits <= LOST_EXPLORATION_MAX_EDITS):
        out.append("lost_in_exploration")
    if malformed > MALFORMED_RATE_MAX:
        out.append("tool_call_hygiene")
    if any(p.search(l) for l in added for p in DEBUG_PATTERNS):
        out.append("debug_artifacts")
    if n_edits >= 1 and n_tests == 0:
        out.append("no_verification")
    return out


def commands_of(message):
    out = []
    for tc in message.get("tool_calls") or []:
        try:
            a = json.loads(tc["function"]["arguments"])
        except Exception:
            a = {}
        c = a.get("command") if isinstance(a, dict) else None
        if isinstance(c, str) and c.strip():
            out.append(c)
    return out


def turns_of(messages, mode):
    a = [m for m in messages if m["role"] == "assistant"]
    if mode == "msg":
        return ["\n".join(commands_of(m)) for m in a]
    return [c for m in a for c in (commands_of(m) or [""])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with the pool's parquet shards (searched recursively)")
    ap.add_argument("--turns", default="msg", choices=["msg", "call"])
    ap.add_argument("--max-ctx", type=int, default=30720)
    ap.add_argument("--out", default="", help="optional: one JSON line per trajectory (id, task, resolved, fired signals)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    files = sorted(glob.glob(a.raw + "/**/*train*.parquet", recursive=True))
    print(f"mc_rules · raw {a.raw} ({len(files)} shards) · turns {a.turns} · max_ctx {a.max_ctx} (teacher tokens, proxy) · "
          f"thresholds returns ≥ {EDIT_TEST_LOOP_MAX_RETURNS} or reverts ≥ {EDIT_TEST_LOOP_MAX_REVERTS} · actions ≥ "
          f"{LOST_EXPLORATION_MIN_ACTIONS} ∧ explore ≥ {LOST_EXPLORATION_EXPLORE_SHARE} ∧ edits ≤ {LOST_EXPLORATION_MAX_EDITS} · "
          f"malformed > {MALFORMED_RATE_MAX}")
    rows = []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=40, columns=["id", "messages", "instance_id", "dataset",
                                                                        "model_patch", "resolved"]):
            for r in b.to_pylist():
                ms = r["messages"]
                asst = [m for m in ms if m["role"] == "assistant"]
                u = (((asst[-1].get("extra") or {}).get("response") or {}).get("usage") or {}) if asst else {}
                think = sum(1 for m in asst if (m.get("reasoning_content") or "").strip()) / max(len(asst), 1)
                rows.append({"id": r["id"], "task": r["instance_id"], "ds": r["dataset"].split("/")[-1],
                             "res": bool(r["resolved"]), "n": len(asst), "think": think,
                             "peak": (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0),
                             "fired": signals(turns_of(ms, a.turns), r["model_patch"])})
    n, nr = len(rows), sum(x["res"] for x in rows)
    nu = n - nr
    print(f"pool {n} trajectories · resolved {nr} · tasks {len({x['task'] for x in rows})}")
    print(f"{'signal':22s} {'all':>6s} {'%':>5s} {'resolved':>9s} {'%':>5s} {'unresolved':>10s} {'%':>5s}")
    for k in SIGNALS:
        c = sum(k in x["fired"] for x in rows)
        cr = sum(k in x["fired"] and x["res"] for x in rows)
        print(f"{k:22s} {c:6d} {100 * c / n:5.1f} {cr:9d} {100 * cr / nr:5.1f} {c - cr:10d} {100 * (c - cr) / nu:5.1f}   "
              f"{'reject' if k in HARD else 'flag in the paper text, reject in the released file'}")
    print("signals fired per trajectory", dict(sorted(collections.Counter(len(x["fired"]) for x in rows).items())))
    for name, keep in (("pass signals 1–4 (paper text)", lambda x: not (set(x["fired"]) & HARD)),
                       ("pass all seven (released file)", lambda x: not x["fired"])):
        s = [x for x in rows if keep(x)]
        sr = [x for x in s if x["res"]]
        fit = [x for x in sr if x["peak"] <= a.max_ctx]
        per = collections.Counter(x["task"] for x in fit)
        print(f"== {name}: {len(s)} of {n} ({100 * len(s) / n:.1f} %) · tasks {len({x['task'] for x in s})} · by source "
              f"{dict(collections.Counter(x['ds'] for x in s))}")
        print(f"   resolved among them {len(sr)} ({100 * len(sr) / max(len(s), 1):.1f} %; pool {100 * nr / n:.1f} %) · tasks "
              f"{len({x['task'] for x in sr})}")
        print(f"   resolved ∧ fits {a.max_ctx} (proxy) {len(fit)} · tasks {len(per)} · at most 3 per task "
              f"{sum(min(3, c) for c in per.values())} · with a thought in every turn {sum(x['think'] == 1 for x in fit)} · "
              f"in ≥ 90 % of turns {sum(x['think'] >= .9 for x in fit)}")
    base = [x for x in rows if x["res"] and x["peak"] <= a.max_ctx]
    print(f"== no rule filter: fits (proxy) {sum(x['peak'] <= a.max_ctx for x in rows)} · resolved ∧ fits {len(base)} · tasks "
          f"{len({x['task'] for x in base})}")
    if a.out:
        with open(a.out, "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
