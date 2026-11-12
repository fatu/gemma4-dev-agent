#!/usr/bin/env python3
"""E4B's own failed edit_file calls → corrected training targets (BOX_PLAN B-7, the relabel stage). No model, CPU only.

    python3 train/relabel_edits.py --runs runs/dev33_base_m1 … --tasks runs/rebench2/tasks.jsonl [more …] \
        --snapshots runs/rebench2/snapshots [more …] --out runs/sft/e4b_relabel

For every trajectory the file state is replayed from the task's snapshot: write_file → the content, an ok edit_file → its
result diff applied (or the exact replace when the diff is cut), a run_command that may write files (sed -i, a redirect,
git checkout / stash / apply, mv, rm, a formatter …) → every file unknown until it is read whole or written again. Each
read_file result is compared with the replayed lines: that share is the check of the replay (printed).

A failed edit (old_string not found) with a known file state S becomes one of (a quote full of escape junk — \\" for ",
\\\\ for \\ — is first tried without it, in old_string and new_string alike, when that brings it closer to the file):
  already applied   new_string is in S, or the closest span of S is closer to new_string than to old_string: the
                    change is there, nothing to correct
  no anchor         no span of S is close enough to old_string (ratio < --min-ratio): the quote is invented
  ambiguous         two different spans are equally close
  conflict          the merge below leaves overlapping pieces, or its result is far from the true span
  not unique        the true span occurs several times even with 3 more lines on each side
  syntax broken     a .py file that parsed before does not parse after the corrected edit
  not visible       a corrected line is in no tool output the model had in its context (--need-visible 1): the right
                    move there is to read the file, not to quote it from memory
  corrected         old_string := the true span T of S; new_string := the model's own change (old → new) carried over
                    to T line by line: a line the quote had wrong and the edit does not touch → the true line; a line
                    the edit changes → the edit's line (with the true indentation). A wrong line quoted in the thought
                    is replaced too
Not-found edits whose file state is unknown, "several matches" errors and replies with more than one tool call are
counted and left out.

Output: call-level records (train/schema.md), one per trajectory that has a target. A corrected call is a target with its
own segment (the history after it holds the wrong call, so it cannot be packed with later calls); ok edits (reply = one
edit_file call that succeeded) are native targets, sampled to --ok-ratio × the corrected ones; every other call has no
target. Calls that are neither a target nor the call after a native target are dropped (they would only be re-rendered).
De-duplication: one key = (task, file, corrected old, corrected new); at most --per-key per key (one first failure, one
after an error on that file) and --task-cap corrected per task. --heldout-frac of the repos go to <out>_heldout.
Every number printed is the record. Outputs quote task text → runs/ (gitignored).
"""
import argparse
import ast
import difflib
import glob
import json
import os
import random
import re
import sys
import tarfile
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_traj import convert, load_results, load_tasks  # noqa: E402

MUTATING = re.compile(
    r"\bsed\b[^|;&]*\s-[a-zA-Z]*i|\bperl\b[^|;&]*\s-[a-zA-Z]*i|(?<![0-9&>])>{1,2}(?!\s*&|\s*/dev/null)|\btee\b|"
    r"\bgit\s+(checkout|restore|stash|apply|reset|revert|merge|pull|rebase|cherry-pick|clean|rm|mv|am)\b|"
    r"\bpatch\b|\b(rm|mv|cp|touch|truncate|dd|ln)\s|\b(black|autopep8|isort|yapf|2to3|autoflake|pyupgrade)\b|"
    r"\bruff\s+(format|check[^|;&]*--fix)|write_text\(|\.write\(|open\([^)]*['\"][wa]|\bfileinput\b|\bshutil\.")
TOKEN = re.compile(r"\s+|\w+|[^\w\s]")
NOT_RUN = re.compile(r"SyntaxError|IndentationError|sed: -e expression|unterminated|unknown option to|command not found|No such file or directory")
WORD = re.compile(r"\w+|[^\w\s]")
UNESCAPE = (('\\" → "', lambda x: x.replace('\\"', '"')), ("\\\\ → \\", lambda x: x.replace("\\\\", "\\")),
            ('both', lambda x: x.replace('\\"', '"').replace("\\\\", "\\")))
SNAP = {}                                                    # tgz → {path: text or None}, kept over the runs


def snapshot_files(tgz, paths):
    """{path: text or None} for the wanted repo paths from snapshots/<id>.tgz (entries ./path)"""
    cache = SNAP.setdefault(tgz, {})
    out = {p: None for p in paths}
    want = {os.path.normpath(p) for p in paths if p} - set(cache)
    if not os.path.exists(tgz):
        return out
    if not want:
        return {p: cache.get(os.path.normpath(p)) for p in paths}
    out = cache
    for p in want:
        out[p] = None
    with tarfile.open(tgz) as tf:
        for m in tf:
            n = os.path.normpath(m.name)
            if n in want and m.isfile():
                try:
                    out[n] = tf.extractfile(m).read().decode("utf-8")
                except UnicodeDecodeError:
                    out[n] = None
    return {p: out.get(os.path.normpath(p)) for p in paths}


def apply_diff(text, diff):
    """the unified diff of an ok edit applied to text; None when a hunk does not fit"""
    lines = text.split("\n")
    out, pos, hunks = [], 0, 0
    dl = diff.split("\n")
    i = 0
    while i < len(dl):
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", dl[i])
        if not m:
            i += 1
            continue
        start = int(m.group(1)) - 1 if m.group(2) != "0" else int(m.group(1))
        i += 1
        old, new = [], []
        while i < len(dl) and not dl[i].startswith("@@"):
            l = dl[i]
            if l.startswith("\\"):
                pass
            elif l.startswith("-"):
                old.append(l[1:])
            elif l.startswith("+"):
                new.append(l[1:])
            elif l.startswith(" ") or l == "":
                if l == "" and i == len(dl) - 1:
                    break
                old.append(l[1:]); new.append(l[1:])
            i += 1
        if start < pos or lines[start:start + len(old)] != old:
            return None
        out += lines[pos:start] + new
        pos = start + len(old)
        hunks += 1
    if not hunks:
        return None
    return "\n".join(out + lines[pos:])


def flat_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from flat_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from flat_strings(v)


def context_lines(messages):
    """stripped lines of everything the tools returned (and the task text) in a call's input; diff lines also without ±"""
    seen = set()
    for m in messages:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        try:
            c = json.loads(c) if isinstance(c, str) and m["role"] == "tool" else c
        except ValueError:
            pass
        for s in flat_strings(c):
            for l in s.split("\n"):
                seen.add(l.strip())
                if l[:1] in "+- ":
                    seen.add(l[1:].strip())
    return seen


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def words(s):
    return WORD.findall(s)


def align(old, S, min_ratio):
    """the span of S closest to old: (start, end, ratio, second) in characters of S, or None. second = the best ratio of a
    span elsewhere in S. Windows of whole lines are ranked by their word-level ratio (cheap); only the best one is then cut
    to the characters old covers."""
    core = old.rstrip("\n")
    if not core.strip():
        return None
    sl = S.split("\n")
    offs = [0]
    for l in sl:
        offs.append(offs[-1] + len(l) + 1)
    ol = core.split("\n")
    n = len(ol)
    where = defaultdict(list)
    for i, l in enumerate(sl):
        where[l.strip()].append(i)
    votes = Counter()
    for j, l in enumerate(ol):
        k = l.strip()
        if len(k) >= 3:
            for i in where.get(k, ())[:50]:
                votes[i - j] += 1
    if len(votes) < 3:                                          # no line is exact: the closest lines of S instead
        keys = [k for k in where if len(k) >= 3]
        for j, l in enumerate(ol[:40]):
            k = l.strip()
            if len(k) >= 3:
                for c in difflib.get_close_matches(k, keys, n=4, cutoff=0.6):
                    for i in where[c][:20]:
                        votes[i - j] += 1
    cw = words(core)
    cands = []
    for s, _ in votes.most_common(12):
        s = max(0, min(s, len(sl) - 1))
        best = None
        for m in range(max(1, n - 3), n + 4):
            if s + m > len(sl):
                break
            r = difflib.SequenceMatcher(None, cw, words("\n".join(sl[s:s + m])), autojunk=False).ratio()
            if best is None or r > best[0]:
                best = (r, s, m)
        if best:
            cands.append(best)
    if not cands:
        return None
    cands.sort(reverse=True)
    r, s, m = cands[0]
    second = max((c[0] for c in cands[1:] if c[1] + c[2] <= s or c[1] >= s + m), default=0.0)
    W = "\n".join(sl[s:s + m])
    ts, te = 0, len(W)
    if len(core) * len(W) <= 4_000_000:                         # character level: where in the window old starts and ends
        blocks = [b for b in difflib.SequenceMatcher(None, core, W, autojunk=False).get_matching_blocks() if b.size]
        if blocks:
            a0, b0 = blocks[0].a, blocks[0].b
            a1, b1 = blocks[-1].a + blocks[-1].size, blocks[-1].b + blocks[-1].size
            ts, te = max(0, b0 - a0), min(len(W), b1 + (len(core) - a1))
            r = ratio(core, W[ts:te])                           # before the span is widened to the line start
            if not W[:ts].strip():                              # only indentation before the span: take the line start
                ts = 0
            if not W[te:].strip() and core.rstrip() == core:
                te = len(W.rstrip()) if W.rstrip() != W else len(W)
    a, b = offs[s] + ts, offs[s] + te
    if r < min_ratio:
        return (a, b, r, second)
    tail = old[len(core):]
    if tail and S[b:b + len(tail)] == tail:
        b += len(tail)
    elif tail and S[b:b + 1] == "\n":
        b += 1
    return (a, b, r, second)


def line_ops(a, b):
    """non-equal opcodes a → b as (i1, i2, replacement lines); a replace of as many lines is split into single lines"""
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            out += [(i1 + k, i1 + k + 1, [b[j1 + k]]) for k in range(i2 - i1)]
        else:
            out.append((i1, i2, b[j1:j2]))
    return out


def indent(l):
    return l[:len(l) - len(l.lstrip())]


def merge3(base, ours, theirs):
    """theirs' change of base carried over to ours, line by line → (text, clashes); (None, n) when it cannot be done.
    A line the quote has wrong (base ≠ ours) that the edit does not touch → the true line. A line the edit changes →
    the edit's line, also when the quote had that line wrong (a clash: there the model said what the result should be;
    only the true indentation is put back). A wrong block of the quote that the edit covers only in part, or lines
    missing from the quote at the very place where the edit inserts → no merge."""
    b, o, t = base.split("\n"), ours.split("\n"), theirs.split("\n")
    A, B = line_ops(b, o), line_ops(b, t)
    keep, clashes = [], 0
    B = list(B)
    for x in A:
        if x[0] == x[1]:                                        # lines the quote left out
            inside = [y for y in B if y[0] < x[0] < y[1]]
            if any(y[0] == y[1] == x[0] and y[2] != x[2] for y in B):
                return None, clashes
            if inside:
                clashes += 1; continue
            if not any(y == x for y in B):
                keep.append(x)
            continue
        over = [k for k, y in enumerate(B) if y[0] < x[1] and x[0] < y[1]]
        if not over:
            keep.append(x); continue
        covered = all(any(B[k][0] <= i < B[k][1] for k in over) for i in range(x[0], x[1]))
        if not covered:
            return None, clashes
        clashes += 1
        if x[1] - x[0] == 1 and len(x[2]) == 1:                 # one wrong line that the edit rewrites: keep its true indent
            wi, ti = indent(b[x[0]]), indent(x[2][0])
            if wi != ti:
                for k in over:
                    y = B[k]
                    if y[1] - y[0] == 1 and y[0] == x[0]:
                        B[k] = (y[0], y[1], [ti + l[len(wi):] if l.startswith(wi) and l.strip() else l for l in y[2]])
    ops = sorted(keep + B, key=lambda x: (x[0], x[1]))
    for u, v in zip(ops, ops[1:]):
        if v[0] < u[1]:
            return None, clashes
    out = list(b)
    for i1, i2, rep in reversed(ops):
        out[i1:i2] = rep
    return "\n".join(out), clashes


def unique_span(S, a, b):
    """grow [a, b) to line ends, then by up to 3 lines on each side, until S[a:b] occurs once; None if it never does"""
    if S.count(S[a:b]) == 1:
        return a, b
    a2 = S.rfind("\n", 0, a) + 1
    b2 = S.find("\n", b)
    b2 = len(S) if b2 < 0 else b2
    for _ in range(4):
        if S.count(S[a2:b2]) == 1:
            return a2, b2
        a2 = S.rfind("\n", 0, max(0, a2 - 1)) + 1
        nb = S.find("\n", b2 + 1)
        b2 = len(S) if nb < 0 else nb
    return None


def parses(text):
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def fix_thought(reasoning, old, old2):
    """wrong quoted lines in the thought → the true ones; returns (text, changed?)"""
    if not reasoning:
        return reasoning, False
    r = reasoning
    if old.strip() and old.strip() in r:
        r = r.replace(old.strip(), old2.strip())
    a, b = old.split("\n"), old2.split("\n")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for w, t in zip(a[i1:i2], b[j1:j2]):
                if len(w.strip()) >= 8 and w.strip() in r:
                    r = r.replace(w.strip(), t.strip())
    return r, r != reasoning


def kind_of_fix(old, old2):
    """what was wrong with the quote"""
    if re.sub(r"\s+", "", old) == re.sub(r"\s+", "", old2):
        return "whitespace / indentation only"
    a, b = old.split("\n"), old2.split("\n")
    bad = sum(i2 - i1 for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b], autojunk=False).get_opcodes() if tag != "equal")
    add = sum(j2 - j1 for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b], autojunk=False).get_opcodes() if tag == "insert")
    if bad == 0 and add:
        return "lines left out of the quote"
    return "1 line wrong" if bad <= 1 else "2–3 lines wrong" if bad <= 3 else "4+ lines wrong"


def process(run, path, tasks, results, snapdir, a, st, hist):
    """one trajectory → (record, [corrected call numbers with meta], [ok edit call numbers with key])"""
    rec = convert(path, "e4b_onpolicy", a.policy, tasks, results, run)
    iid = rec["instance_id"]
    by_call = {c["call"]: c for c in rec["calls"]}
    owner, pend = {}, {}                                         # tool_call id → call number / (name, args)
    for c in rec["calls"]:
        for tc in ((c.get("target") or {}).get("tool_calls") or []):
            owner[tc.get("id")] = c["call"]
    rows = [json.loads(l) for l in open(path)]
    msgs, seen = [], set()
    for r in rows:
        if r.get("compaction"):
            continue
        for m in r.get("new") or []:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (ValueError, TypeError):
                        args = {}
                    pend.setdefault(tc["id"], (tc["function"]["name"], args if isinstance(args, dict) else {}))
            elif m.get("role") == "tool" and m.get("tool_call_id") in pend and m["tool_call_id"] not in seen:
                seen.add(m["tool_call_id"])
                msgs.append((m["tool_call_id"],) + pend[m["tool_call_id"]] + (m.get("content"),))
    paths = sorted({x[2].get("filepath") for x in msgs if x[1] in ("edit_file", "write_file") and isinstance(x[2].get("filepath"), str)})
    tgz = next((x for x in (os.path.join(d, f"{iid}.tgz") for d in snapdir) if os.path.exists(x)), "")
    state = snapshot_files(tgz, paths)
    if not tgz:
        st["trajectories without a snapshot"] += 1
    failed_before = Counter()                                    # failed edits on a file since its last ok edit
    corrected, okedits = [], []
    for tcid, name, args, content in msgs:
        try:
            o = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            o = {}
        o = o if isinstance(o, dict) else {}
        ok = o.get("status") == "ok"
        p = args.get("filepath")
        call = owner.get(tcid)
        if name == "run_command":
            err = (o.get("error_message") or "") + ((o.get("details") or {}).get("stderr") or "") if not ok else ""
            if MUTATING.search(args.get("command") or "") and not NOT_RUN.search(err):
                st["replay: commands that may write files"] += 1
                for k in state:
                    state[k] = None
            continue
        if name == "read_file" and ok and p in state:
            got = o.get("content") or ""
            whole = o.get("start_line") == 1 and not o.get("is_truncated") and o.get("end_line") == o.get("total_lines")
            if state[p] is None:
                if whole:
                    state[p] = got; st["replay: file state taken from a whole read"] += 1
            else:
                sl = state[p].split("\n")
                want = "\n".join(sl[(o.get("start_line") or 1) - 1:o.get("end_line") or len(sl)])
                same = want.rstrip("\n") == got.rstrip("\n")
                st["replay: reads checked"] += 1; st["replay: reads equal to the replayed lines"] += same
                if not same:
                    state[p] = got if whole else None
            continue
        if name == "write_file" and ok and isinstance(p, str):
            state[p] = args.get("content") or ""
            failed_before[p] = 0
            continue
        if name != "edit_file" or not isinstance(p, str):
            continue
        old, new = args.get("old_string"), args.get("new_string")
        S = state.get(p)
        reply = (by_call.get(call) or {}).get("target") or {}
        single = len(reply.get("tool_calls") or []) == 1
        if ok:
            st["edits: ok"] += 1
            if S is not None:
                S2 = apply_diff(S, o.get("diff") or "") if not o.get("is_truncated") else None
                if S2 is None and isinstance(old, str) and isinstance(new, str) and S.count(old) == 1:
                    S2 = S.replace(old, new); st["replay: ok edits applied by exact replace"] += 1
                elif S2 is not None:
                    st["replay: ok edits applied by their diff"] += 1
                else:
                    st["replay: ok edits that did not apply (file unknown from there)"] += 1
                state[p] = S2
            if single and call and isinstance(old, str) and isinstance(new, str):
                okedits.append((call, (iid, p, old, new)))
            failed_before[p] = 0
            continue
        msg = (o.get("error_message") or o.get("error") or "")
        if "occurs" in msg:
            st["edits: several matches (left out)"] += 1; failed_before[p] += 1; continue
        if "not found" not in msg:
            st["edits: other errors (left out)"] += 1; continue
        st["edits: not found"] += 1
        after_error = failed_before[p] > 0
        failed_before[p] += 1

        def out(k):
            st[f"not found → {k}"] += 1
        if not isinstance(old, str) or not isinstance(new, str) or not call:
            out("no usable arguments"); continue
        if not single:
            out("reply with several tool calls"); continue
        if S is None:
            out("file state unknown (missing file, or after a writing command)"); continue
        if old in S:
            out("replay disagrees (old_string is in the replayed file)"); continue
        if new.strip() and new in S:
            out("already applied"); continue
        al, raw_old = align(old, S, a.min_ratio), old
        for name, fn in UNESCAPE:                                # escape junk (\" for ", \\ for \): try the quote without it
            if al and al[2] >= 0.999:
                break
            o2 = fn(old)
            if o2 == old:
                continue
            al2 = align(o2, S, a.min_ratio)
            if al2 and (not al or al2[2] > al[2] + 0.005) and o2 not in S:
                al, old, new = al2, o2, fn(new)
                st[f"unescaped before the alignment: {name}"] += 1
        hist[min(10, int((al[2] if al else 0) * 10))] += 1
        if not al or al[2] < a.min_ratio:
            out("no anchor"); continue
        s0, s1, r, second = al
        if second >= r - a.margin:
            out("ambiguous"); continue
        T = S[s0:s1]
        if ratio(words(T), words(new)) >= ratio(words(T), words(old)):
            out("already applied (the file is closer to new_string than to old_string)"); continue
        new2, clashes = merge3(old, T, new)
        if new2 is None:
            out("conflict"); continue
        if clashes and ratio(T, new2) < a.min_ratio / 2:
            out("conflict"); continue
        u = unique_span(S, s0, s1)
        if u is None:
            out("not unique"); continue
        old2, new2 = S[u[0]:u[1]], S[u[0]:s0] + new2 + S[s1:u[1]]
        if old2 == new2 or not old2:
            out("no change left"); continue
        if p.endswith(".py") and parses(S) and not parses(S[:u[0]] + new2 + S[u[1]:]):
            out("syntax broken"); continue
        ctx = context_lines(by_call[call]["input_messages"])
        ol = {l.strip() for l in old.split("\n")}
        unseen = [l for l in old2.split("\n") if l.strip() and l.strip() not in ol and l.strip() not in ctx]
        if unseen and a.need_visible:
            out("not visible"); continue
        out("corrected")
        st["corrected: a corrected line not in the context (kept, --need-visible 0)"] += bool(unseen)
        thought, changed = fix_thought(reply.get("reasoning"), raw_old, old2)
        corrected.append((call, {"key": (iid, p, old2, new2), "after_error": after_error, "ratio": round(r, 3),
                                 "kind": kind_of_fix(raw_old, old2), "old": raw_old, "new": args.get("new_string"), "old2": old2, "new2": new2,
                                 "clashes": clashes,
                                 "thought": thought, "thought_changed": changed, "path": p}))
    return rec, corrected, okedits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True, help="tasks.jsonl of every task set the runs come from")
    ap.add_argument("--snapshots", nargs="+", required=True, help="snapshot dirs (<id>.tgz); the first that has the task is used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="gemma-4-e4b-it-qat-w4a16-ct")
    ap.add_argument("--min-ratio", type=float, default=0.75, help="closeness of the true span to old_string (0–1)")
    ap.add_argument("--margin", type=float, default=0.03, help="a second span within this of the best → ambiguous")
    ap.add_argument("--need-visible", type=int, default=1, help="1: every corrected line must be in a tool output of the context")
    ap.add_argument("--per-key", type=int, default=2, help="copies of one corrected call (first failure / after an error)")
    ap.add_argument("--task-cap", type=int, default=24, help="corrected calls per task over all runs")
    ap.add_argument("--ok-ratio", type=float, default=1.0, help="ok edits kept = this × corrected (0 = none)")
    ap.add_argument("--heldout-frac", type=float, default=0.1, help="share of the repos whose records go to <out>_heldout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=40, help="corrected examples written to <out>.examples.txt")
    a = ap.parse_args()
    print(f"relabel_edits · runs {len(a.runs)} [{' '.join(os.path.basename(r.rstrip('/')) for r in a.runs)}] · tasks {' '.join(a.tasks)} · "
          f"snapshots {' '.join(a.snapshots)} · out {a.out} · min_ratio {a.min_ratio} · margin {a.margin} · need_visible {a.need_visible} · "
          f"per_key {a.per_key} · task_cap {a.task_cap} · ok_ratio {a.ok_ratio} · heldout_frac {a.heldout_frac} · seed {a.seed}")
    tasks = {}
    for t in a.tasks:
        tasks.update(load_tasks(t))
    print(f"   tasks known: {len(tasks)}")
    st, hist = Counter(), Counter()
    items = []                                                   # (run, file, rec, corrected, okedits)
    for run in a.runs:
        run = run.rstrip("/")
        results = load_results(run)
        files = [f for f in sorted(glob.glob(os.path.join(run, "traj", "*.jsonl"))) if not f.endswith(".tools.jsonl")]
        n0 = st["not found → corrected"]
        for f in files:
            st["trajectories"] += 1
            try:
                rec, cor, oke = process(run, f, tasks, results, a.snapshots, a, st, hist)
            except Exception as e:                               # one broken log must not stop the stage
                st["trajectories that raised (left out)"] += 1
                print(f"  ! {os.path.basename(run)}/{os.path.basename(f)}: {type(e).__name__}: {str(e)[:100]}")
                continue
            items.append((os.path.basename(run), rec, cor, oke))
        print(f"  {os.path.basename(run)}: {len(files)} trajectories · corrected so far +{st['not found → corrected'] - n0}")
    if not st["trajectories"]:
        sys.exit("no trajectories")

    def show(prefix, total=None):
        ks = [k for k in st if k.startswith(prefix)]
        tot = total or sum(st[k] for k in ks)
        for k in sorted(ks, key=lambda k: -st[k]):
            print(f"   {k[len(prefix):]:<68} {st[k]:>6}  ({100 * st[k] / tot:.0f} %)" if tot else f"   {k}")
    print(f"== trajectories {st['trajectories']} (raised {st['trajectories that raised (left out)']}, without a snapshot {st['trajectories without a snapshot']})")
    print(f"== edits: ok {st['edits: ok']} · not found {st['edits: not found']} · several matches {st['edits: several matches (left out)']} · other errors {st['edits: other errors (left out)']}")
    rc = st["replay: reads checked"]
    print(f"== replay check: {st['replay: reads equal to the replayed lines']} of {rc} reads equal the replayed file lines "
          f"({100 * st['replay: reads equal to the replayed lines'] / rc:.1f} %)" if rc else "== replay check: no reads to compare")
    show("replay: ", None)
    print(f"== the {st['edits: not found']} not-found edits:")
    show("not found → ", st["edits: not found"])
    for k in sorted(k for k in st if k.startswith("unescaped")):
        print(f"   {k}: {st[k]}")
    print("   closeness of the best span (aligned ones): " + " · ".join(f"{b / 10:.1f}+ {hist[b]}" for b in range(11) if hist[b]))

    # de-duplication and caps over all runs
    rnd = random.Random(a.seed)
    allc = [(run, rec["instance_id"], call, m) for run, rec, cor, _ in items for call, m in cor]
    per_key, per_task, keep = defaultdict(list), Counter(), set()
    for run, iid, call, m in allc:
        per_key[m["key"]].append((run, iid, call, m))
    order = list(per_key)
    rnd.shuffle(order)
    dropped_key = dropped_cap = 0
    for k in order:
        g = per_key[k]
        pick = [x for x in g if not x[3]["after_error"]][:1] + [x for x in g if x[3]["after_error"]][:1]
        pick = (pick + [x for x in g if x not in pick])[:a.per_key] if len(pick) < min(a.per_key, len(g)) else pick[:a.per_key]
        dropped_key += len(g) - len(pick)
        for x in pick:
            if per_task[x[1]] >= a.task_cap:
                dropped_cap += 1; continue
            per_task[x[1]] += 1; keep.add((x[0], x[1], x[2]))
    kept = [x for x in allc if (x[0], x[1], x[2]) in keep]
    print(f"== corrected {len(allc)} → different (task, file, old, new) {len(per_key)} → kept {len(kept)} "
          f"(dropped: {dropped_key} repeats beyond {a.per_key} per key, {dropped_cap} over the task cap {a.task_cap}) on {len(per_task)} tasks")
    if kept:
        tc = sorted(per_task.values())
        print(f"   per task: median {tc[len(tc) // 2]} · max {tc[-1]} · tasks with ≥ 10: {sum(v >= 10 for v in tc)}")
        print("   first failure " + str(sum(not x[3]["after_error"] for x in kept)) + " · after an error on that file " + str(sum(x[3]["after_error"] for x in kept))
              + " · thought changed " + str(sum(x[3]["thought_changed"] for x in kept))
              + " · the quote was wrong inside the part being changed (the edit's own text kept there) " + str(sum(bool(x[3]["clashes"]) for x in kept)))
        kc = Counter(x[3]["kind"] for x in kept)
        print("   what was wrong: " + " · ".join(f"{k} {v}" for k, v in kc.most_common()))
    okall = [(run, rec["instance_id"], call, key) for run, rec, _, oke in items for call, key in oke]
    okkey = {}
    for x in okall:
        okkey.setdefault(x[3], x)
    okpool = sorted(okkey.values(), key=lambda x: (x[0], x[1], x[2]))
    rnd.shuffle(okpool)
    n_ok = min(len(okpool), int(round(a.ok_ratio * len(kept))))
    okkeep = {(x[0], x[1], x[2]) for x in okpool[:n_ok]}
    print(f"== ok edits (reply = one edit call) {len(okall)} → different {len(okpool)} → kept {n_ok} (ok_ratio {a.ok_ratio})")

    held = lambda iid: (zlib.crc32(iid.rsplit("-", 1)[0].encode()) % 1000) < a.heldout_frac * 1000
    for d in (a.out, a.out + "_heldout"):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.json")):
            os.remove(f)
    meta = {(x[0], x[1], x[2]): x[3] for x in kept}
    n = Counter()
    ex = []
    for run, rec, cor, oke in items:
        iid = rec["instance_id"]
        tgt = {}
        for c in rec["calls"]:
            k = (run, iid, c["call"])
            if k in meta:
                tgt[c["call"]] = "corrected"
            elif k in okkeep:
                tgt[c["call"]] = "ok"
        if not tgt:
            continue
        calls, prev_native = [], False
        for c in rec["calls"]:
            kind = tgt.get(c["call"])
            if kind == "corrected":
                m = meta[(run, iid, c["call"])]
                t = dict(c["target"])
                tc = dict(t["tool_calls"][0])
                args = json.loads(tc["args"]) if isinstance(tc.get("args"), str) else dict(tc.get("args") or {})
                args["old_string"], args["new_string"] = m["old2"], m["new2"]
                tc["args"] = json.dumps(args, ensure_ascii=False)
                t["tool_calls"] = [tc]; t["reasoning"] = m["thought"]
                c = dict(c, target=t, segment=100000 + c["call"],
                         relabel={"kind": m["kind"], "ratio": m["ratio"], "after_error": m["after_error"], "thought_changed": m["thought_changed"], "clashes": m["clashes"]})
                calls.append(c); prev_native = False
                if len(ex) < a.examples:
                    ex.append((run, iid, c["call"], m))
            elif kind == "ok":
                calls.append(c); prev_native = True
            elif prev_native:
                calls.append(dict(c, target=None, excluded="context")); prev_native = False
        d = a.out + "_heldout" if held(iid) else a.out
        split = "heldout" if held(iid) else "train"
        rec2 = dict(rec, calls=calls, source="e4b_onpolicy")
        with open(os.path.join(d, f"{iid}.{run}.json"), "w") as w:
            json.dump(rec2, w, ensure_ascii=False)
        n[f"{split} records"] += 1
        n[f"{split} corrected"] += sum(1 for v in tgt.values() if v == "corrected")
        n[f"{split} ok"] += sum(1 for v in tgt.values() if v == "ok")
        n[f"{split} tasks " + iid] = 1
    for s in ("train", "heldout"):
        print(f"== {s}: {n[s + ' records']} records · targets: corrected {n[s + ' corrected']} + ok {n[s + ' ok']} · "
              f"tasks {sum(1 for k in n if k.startswith(s + ' tasks '))} → {a.out + ('_heldout' if s == 'heldout' else '')}")
    with open(a.out + ".examples.txt", "w") as w:
        for run, iid, call, m in ex:
            w.write(f"===== {run} {iid} call {call} {m['path']} · {m['kind']} · ratio {m['ratio']} · after_error {m['after_error']} · clashes {m['clashes']}\n")
            for name, x, y in (("old_string", m["old"], m["old2"]), ("new_string", m["new"], m["new2"])):
                w.write(f"--- {name}: model → corrected\n")
                w.write("\n".join(difflib.unified_diff(x.split("\n"), y.split("\n"), lineterm="", n=1)) + "\n")
    print(f"   {len(ex)} examples (model → corrected, as diffs) → {a.out}.examples.txt (not for a photo: task text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
