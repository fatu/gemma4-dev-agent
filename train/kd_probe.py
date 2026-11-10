#!/usr/bin/env python3
"""S0 / S0b of the token-level distillation plan (BOX_PLAN B-4): does the 31B teacher point at E4B's errors, and where
would a distillation signal land? Read-only, CPU.

    .venv-eval/bin/python train/kd_probe.py runs/opd/s0/rows.jsonl --npz 'runs/opd/s0/t31_*.npz' --recs runs/opd/s0 \
        [--student-npz 'runs/opd/s0/s_*.npz'] --tokenizer google/gemma-4-E4B-it-qat-w4a16-ct
    python3 train/kd_probe.py --selftest          # laptop: the normalisations and the locator on made-up strings
Several scored sets are pooled by giving rows / --npz / --recs / --student-npz the same number of values.

Inputs: the rows teacher_score.py scored (render.py --pack rows of E4B's own trajectories, each with `tag` = its pass and
`calls` = the record's call numbers, one contiguous label run per call), the teacher's npz shards, optionally the same
rows scored by E4B itself (teacher_score.py --model <E4B>: the student's own log-probabilities), and the call-level
records (`<recs>/recs_<tag>/<instance_id>.json`, convert_traj.py --mask-errors off) for tool results and contexts.
Every reply's text is rebuilt token by token (per-token decode; a reply whose pieces do not join to its full decode is
skipped and counted). "Shown" text = read_file content, command stdout/stderr with grep -n / cat -n prefixes removed,
tool diff/output fields, the agent's own earlier edit new_strings — all DECODED, while the model saw them inside a JSON
string (a newline as backslash-n, a quote as backslash-quote) and often writes them that way.

The locator (third version, 2026-11-10) follows what the harness's edit matcher forgives, because a forgiven difference
is not why an edit was "not found": the matcher tries the string as written (exact, then per line without leading /
trailing whitespace, then with any whitespace between its tokens) and then the string with one or two layers of JSON
escapes removed (exact, then per line without leading / trailing whitespace). Modes here: `as written` with every
whitespace run removed (one space kept between two word characters), and unicode / inline / inline x2 / full / full x2
un-escaping with each line stripped. The shown text is normalised the same way. The first wrong character of a
not-found old_string is the LATEST first-unexplained character over the modes, mapped back to the text as written. A
string that some mode explains completely was shown in full: a stale view (or the wrong file), set apart.

  BIRTH  every not-found edit's wrong text W = old_string up to and including that character (the explained prefix must
         be ≥ --min-prefix normalised chars). Resends of the same W in a task count once. W's FIRST occurrence in the
         trajectory (a thought or a call, searched in the same normalised space, any un-escaping of the reply) is where
         the error was made; measured there, with what was shown at that point. Set apart (not in the verdict): written
         before the source was shown, right as shown then (the file or the view changed later), the shown continuation is
         not unique, the prefix ends its source.
         At the token covering the wrong character (`lead` = that token's text before the character):
           flagged   = E4B's token is not the teacher's top-1
           corrected = the teacher's top-1 agrees with the lead, reaches the wrong character and continues the shown
                       text there (as shown, or written with JSON escapes)
           undetermined = the teacher's top-1 is a prefix of the lead: it stops before the wrong character, so its
                       choice there is not in the scores → left out of the verdict's n
           prefers   = the teacher's probability of the right tokens in its top-K > its probability of E4B's token
           with --student-npz: push = log p_E4B − log p_31B of E4B's token (> 0: distillation lowers it; it is > 0 on
           average for ANY teacher because the token was sampled from E4B → read it against CTRL) and lift = log p_31B −
           log p_E4B of the right tokens (a lower bound when they are outside E4B's top-K).
  CTRL   the control: the tokens of the same quotes that E4B copied RIGHT (from W's start to the wrong character).
  SEV    every several-matches edit's old_string: its first occurrence; at the first point after it where E4B's text
         leaves every shown continuation: would the teacher go on (its top-1 continues a shown copy) or stop as E4B did.
  SUBMIT at each submit_patch decision (first token of the tool name): the teacher's top-1, by the last command since the
         last edit (exit 0 · exit ≠ 0 · none).
  TAIL   the teacher's probability mass outside its top-K per span (1 − Σ exp of the stored log-probabilities): how much
         a top-K bucketed KL cannot see. Decides K for S1's scoring (BOX_PLAN B-4).
  STUDENT (needs --student-npz) per span E4B's own NLL of its tokens against the 31B's: E4B's must be the lower one,
         else the model that scored is not the model that sampled.
  MAP    (needs --student-npz) where the signal log p_E4B − log p_31B lands: thought prose · code inside ``` in a thought ·
         tool name · tool arguments · text.
S0 lines kept only so that the S0 photo's numbers reappear (same files read): b / c (the failing call and its thought,
the old as-written locator), d (closing old_string in the failing call), e (tool name swaps), f (where the teacher would
end the thought). a = per span.
"""
import argparse
import bisect
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_score import runs_of, span_kinds  # noqa: E402

GREP_PREFIX = re.compile(r"^(?:[^\n:]{1,300}:\d+[:-]|\s*\d+\t)", re.M)    # grep -n / grep -rn "path:12:" · cat -n "    12\t"
ESC_UNI = re.compile(r"(?<!\\)\\u([0-9a-fA-F]{4})")
ESC_INLINE = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\])')
ESC_FULL = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/bfnrt])')
ESC_MAP = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
ESC_RX = {"u": ESC_UNI, "i": ESC_INLINE, "f": ESC_FULL}
WS_STRIP = re.compile(r"(?m)^[ \t]+|[ \t\r]+$")
WS_RUN = re.compile(r"\s+")
DEINDENT = re.compile(r"\n[ \t]+")
CHK = "birth check: E4B's token differs from every way of writing the right text at the wrong character"
VARIANTS = (("as written", ()), ("unicode", ("u",)), ("inline", ("i",)), ("inline x2", ("i", "i")), ("full", ("f",)),
            ("full x2", ("f", "f")))
MODES = tuple((n, l, "collapse" if not l else "strip") for n, l in VARIANTS)


def _sub(s, idx, rx, repl):
    """every match of rx in s → repl(match); idx[i] = the index in the ORIGINAL text of s[i] (len(idx) == len(s) + 1)"""
    out, om, last = [], [], 0
    for m in rx.finditer(s):
        if m.end() == m.start():
            continue
        out.append(s[last:m.start()]); om.extend(idx[last:m.start()])
        r = repl(m)
        out.append(r); om.extend([idx[m.start()]] * len(r))
        last = m.end()
    out.append(s[last:]); om.extend(idx[last:])
    return "".join(out), om


def _unesc(m, layer):
    g = m.group(1)
    if layer == "u":
        return chr(int(g, 16))
    if len(g) == 5 and g[0] == "u":
        return chr(int(g[1:], 16))
    return ESC_MAP.get(g, g)


def unescape(text, layers):
    s, idx = text, list(range(len(text) + 1))
    for layer in layers:
        s, idx = _sub(s, idx, ESC_RX[layer], lambda m, layer=layer: _unesc(m, layer))
    return s, idx


def _word(c):
    return c.isalnum() or c == "_"


def norm(text, layers, ws):
    """→ (normalised text, map to the original indices): `layers` of JSON un-escaping, then the whitespace rule"""
    s, idx = unescape(text, layers)
    if ws == "strip":
        s, idx = _sub(s, idx, WS_STRIP, lambda m: "")
    elif ws == "collapse":
        t = s
        s, idx = _sub(s, idx, WS_RUN, lambda m: " " if 0 < m.start() and m.end() < len(t) and _word(t[m.start() - 1])
                      and _word(t[m.end()]) else "")
    return s, idx


def esc_inline(g):
    return g.replace("\\", "\\\\").replace('"', '\\"')


def esc_full(g):
    return esc_inline(g).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def renderings(raw):
    """the ways the model may write the shown text `raw`: as shown, or with one / two layers of JSON escapes"""
    out = []
    for g in (raw, esc_inline(raw), esc_full(raw), esc_full(esc_full(raw))):
        if g and g not in out:
            out.append(g)
    return out


OWN = {"as written": lambda g: g, "unicode": lambda g: g, "inline": esc_inline, "inline x2": lambda g: esc_inline(esc_inline(g)),
       "full": esc_full, "full x2": lambda g: esc_full(esc_full(g))}


def sources(msgs):
    """texts the model has been shown in this context: tool outputs + its own earlier edit new_strings"""
    out = []
    for m in msgs:
        if m.get("role") == "tool":
            c = m.get("content")
            try:
                j = json.loads(c) if isinstance(c, str) else c
            except ValueError:
                j = None
            if isinstance(j, dict):
                for k in ("content", "stdout", "stderr", "diff", "output"):
                    v = j.get(k)
                    if isinstance(v, str) and v:
                        out.append(GREP_PREFIX.sub("", v) if k in ("stdout", "stderr") else v)
            elif isinstance(c, str):
                out.append(c)
        elif m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or tc
                a = f.get("arguments") or f.get("args")
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                except ValueError:
                    a = None
                if isinstance(a, dict) and isinstance(a.get("new_string"), str):
                    ns = a["new_string"]
                    out.append(ns)
                    if "\\" in ns:                                   # the matcher un-escapes new_string with old_string
                        for layers in (("i",), ("f",), ("f", "f")):
                            u = unescape(ns, layers)[0]
                            if u not in out:
                                out.append(u)
    return out


def longest_prefix(s, srcs):
    lo, hi = 0, len(s)                       # invariant: s[:lo] occurs somewhere
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if any(s[:mid] in x for x in srcs):
            lo = mid
        else:
            hi = mid - 1
    return lo


def continuations(prefix, srcs, n=120):
    """the shown text after every occurrence of prefix (the S0 lines' as-written locator)"""
    out = []
    for s in srcs:
        k = s.find(prefix)
        while k >= 0:
            out.append(s[k + len(prefix):k + len(prefix) + n]); k = s.find(prefix, k + 1)
    return out


def nsources(srcs, ws):
    """the shown texts under a whitespace rule, each with its map back to the raw text"""
    return [norm(s, (), ws) + (s,) for s in srcs]


def conts_n(prefix, nsrcs):
    """(normalised source, map, raw source, end index of the prefix) for every occurrence of a normalised prefix"""
    out = []
    for nt, mp, raw in nsrcs:
        q = nt.find(prefix)
        while q >= 0:
            if q + len(prefix) < len(nt):
                out.append((nt, mp, raw, q + len(prefix)))
            q = nt.find(prefix, q + 1)
    return out


def locate(old, srcs_by_ws):
    """the first character of `old` that no forgiven reading explains. srcs_by_ws: {ws: [normalised shown texts]}.
    → ("whole", mode name) when some mode explains the whole string, else (orig index, mode, normalised old, map, Ln)"""
    best, seen = None, set()
    for order, (name, layers, ws) in enumerate(MODES):
        if layers:
            u = unescape(old, layers)[0]
            if u == old or u in seen:
                continue
            seen.add(u)
        o, om = norm(old, layers, ws)
        if not o:
            continue
        ln = longest_prefix(o, srcs_by_ws[ws])
        if ln >= len(o):
            return "whole", name
        cand = (om[ln], -order, (name, layers, ws), o, om, ln)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best and (best[0], best[2], best[3], best[4], best[5])


def outcome(text):
    if text is None:
        return "?"
    if '"status": "ok"' in text or '"status":"ok"' in text:
        return "ok"
    if "not found" in text:
        return "not_found"
    if "times" in text or "multiple" in text or "locations" in text:
        return "several"
    return "other"


def tool_results(rec):
    out = {}
    for c in rec["calls"]:
        for m in c["input_messages"]:
            if m.get("role") == "tool" and m.get("tool_call_id") and m["tool_call_id"] not in out:
                v = m.get("content")
                out[m["tool_call_id"]] = v if isinstance(v, str) else json.dumps(v)
    return out


def q(xs, p=0.5):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def load_npz(pattern, pre):
    arr, files = {}, sorted(glob.glob(pattern)) if pattern else []
    for f in files:
        z = np.load(f)
        for k in z.files:
            arr[pre + k] = z[k]
    return arr, files


def pc(xs, f):
    return f"{100 * sum(1 for x in xs if f(x)) / max(len(xs), 1):.0f} %"


def selftest():
    for layers in VARIANTS:
        s = 'a\\nb \\"c\\" \\\\n \\u0041 \\\\"x\\\\" \\t'
        t, mp = unescape(s, layers[1])
        assert len(mp) == len(t) + 1 and mp[-1] == len(s) and all(mp[i] <= mp[i + 1] for i in range(len(t)))
    assert unescape("a\\nb", ("f",))[0] == "a\nb" and unescape('a\\"b', ("i",))[0] == 'a"b' and unescape("a\\nb", ("i",))[0] == "a\\nb"
    assert unescape("a\\\\nb", ("f", "f"))[0] == "a\nb" and unescape("\\u0041", ("u",))[0] == "A"
    try:                                                             # the harness's own functions, when it is installed
        from adk_eval_core.editing import edit as he
        for s in ('x\\n  y\\"z\\\\', "\\u00e9\\t\\/", 'plain', '\\\\n\\\\"'):
            assert unescape(s, ("i",))[0] == he._unescape_inline_json_layer(s) and unescape(s, ("f",))[0] == he._unescape_full_json_layer(s)
            assert unescape(s, ("u",))[0] == he._unescape_unicode_only(s)
        hmsg = "equal to the harness's un-escaping on 4 strings"
    except ImportError:
        hmsg = "harness not installed here: its un-escaping not compared"
    t, mp = norm("  def f(a,\n        b):  \n    return a", (), "strip")
    assert t == "def f(a,\nb):\nreturn a" and mp[0] == 2 and mp[len(t)] == len("  def f(a,\n        b):  \n    return a")
    t, mp = norm("def  f( a ,\n b )", (), "collapse")
    assert t == "def f(a,b)", t
    assert [mp[i] for i in (0, 3, 4)] == [0, 3, 5]
    file_ = "class A:\n    def f(self, x):\n        if x is None):\n            return 0\n        return x + 1\n"
    S = {ws: [norm(file_, (), ws)[0]] for ws in ("strip", "collapse")}
    # (1) literal backslash-n everywhere, a real error at the end: the old locator stops at the first backslash
    old = "    def f(self, x):\\n        if x is None]:"
    assert longest_prefix(old, [file_]) == len("    def f(self, x):")
    p, mode, o, om, ln = locate(old, S)
    assert old[p] == "]" and mode[0] == "full", (p, mode)
    # (2) only escapes and indentation differ → explained completely (stale view / wrong file)
    assert locate("def f(self, x):\\n    if x is None):\\n        return 0", S)[0] == "whole"
    assert locate("def f(self, x):\n  if x is None):\n      return 0", S)[0] == "whole"
    # (3) as written, whitespace differs between tokens and the error is a word
    p, mode, o, om, ln = locate("def f( self,x ):\n if x is Nome):", S)
    assert "def f( self,x ):\n if x is Nome):"[p] == "m" and mode[0] == "as written", (p, mode)
    # (4) a dropped line is still an error
    p, mode, o, om, ln = locate("def f(self, x):\n            return 0", S)
    assert "def f(self, x):\n            return 0"[p] == "r", p
    # (5) the continuation of the explained prefix, raw and rendered
    p, mode, o, om, ln = locate(old, S)
    c = conts_n(o[:ln], nsources([file_], mode[2]))
    assert len(c) == 1 and c[0][2][c[0][1][c[0][3]]:][:2] == "):"
    assert renderings('a"\n') == ['a"\n', 'a\\"\n', 'a\\"\\n', 'a\\\\\\"\\\\n']
    print(f"kd_probe selftest PASS (un-escape maps; {hmsg}; strip / collapse maps; locator: escapes and whitespace are not "
          "the wrong character, a dropped line and a wrong word are, a fully explained string is 'whole'; renderings)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="*")
    ap.add_argument("--npz", nargs="+", default=[], help="glob of the teacher's npz shards (one per rows file)")
    ap.add_argument("--student-npz", nargs="*", default=[], help="glob of the same rows scored by E4B itself (optional)")
    ap.add_argument("--recs", nargs="+", default=[], help="dir holding recs_<tag>/ (convert_traj --mask-errors off), one per rows file")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--min-prefix", type=int, default=12, help="normalised chars: a shorter explained prefix = not locatable")
    ap.add_argument("--examples", type=int, default=0, help="print this many birth cases in full (laptop checks)")
    ap.add_argument("--dump-births", default="", help="write the measured births (row, position, lead, right text) as jsonl (laptop tests)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.rows or len(a.npz) != len(a.rows) or len(a.recs) != len(a.rows) or len(a.student_npz) not in (0, len(a.rows)):
        sys.exit("rows, --npz and --recs need the same number of values (and --student-npz none or as many)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    END, QUOTE = tok.convert_tokens_to_ids("<channel|>"), tok.convert_tokens_to_ids('<|"|>')
    T, S, rows, K = {}, {}, [], 0
    for d, rf in enumerate(a.rows):
        t_, tf = load_npz(a.npz[d], f"{d}:"); T.update(t_)
        s_, sf = load_npz(a.student_npz[d], f"{d}:") if a.student_npz else ({}, [])
        S.update(s_)
        rs = [json.loads(l) for l in open(rf)]
        rows += [(f"{d}:{i}", d, r) for i, r in enumerate(rs)]
        print(f"kd_probe set {d} · rows {len(rs)} ({rf}) · teacher npz {len(tf)} files ({a.npz[d]}) · student npz {len(sf)} files "
              f"({a.student_npz[d] if a.student_npz else 'none: no push / STUDENT / MAP lines'}) · recs {a.recs[d]}")
    print(f"kd_probe · sets {len(a.rows)} · tokenizer {a.tokenizer} · min prefix {a.min_prefix} normalised chars · locator modes "
          + ", ".join(f"{n} [{ws}]" for n, _, ws in MODES))
    cnt = Counter()
    piece_of = {}
    P = lambda x: piece_of.setdefault(int(x), tok.decode([int(x)]))
    # ---- index: one entry per scored reply, keyed (set, tag, instance, call) ----
    R, recs = {}, {}
    span = defaultdict(lambda: [0, 0.0, 0, 0, 0.0, 0.0, 0])            # kind → teacher n, nll, top1 · student n, nll, teacher nll there, top1
    for i, d, row in rows:
        if f"{i}/pos" not in T:
            cnt["rows not scored by the teacher (over max_len / missing shard)"] += 1; continue
        K = max(K, T[f"{i}/t_lp"].shape[1] if T[f"{i}/t_lp"].ndim == 2 else 0)
        pos = T[f"{i}/pos"].tolist()
        if pos != [j for j in range(1, len(row["input_ids"])) if row["labels"][j] != -100]:
            cnt["rows whose scored positions differ from the labels (skipped)"] += 1; continue
        has_s = bool(S) and f"{i}/pos" in S and S[f"{i}/pos"].tolist() == pos
        if S and not has_s:
            cnt["rows the student scores do not cover (no push / map there)"] += 1
        runs = runs_of(pos)
        if len(runs) != len(row["calls"]):
            cnt["rows whose reply runs ≠ calls (skipped)"] += 1; continue
        key = (d, row.get("tag", ""), row["instance_id"])
        if key not in recs:
            p = os.path.join(a.recs[d], f"recs_{key[1]}", f"{key[2]}.json")
            recs[key] = json.load(open(p)) if os.path.exists(p) else None
        if recs[key] is None:
            cnt["rows without their record (skipped)"] += 1; continue
        cnt["rows used"] += 1
        for run, cn in zip(runs, row["calls"]):
            ids = [row["input_ids"][pos[n]] for n in run]
            toks = tok.convert_ids_to_tokens(ids)
            for n, kd in zip(run, span_kinds(toks)):
                s_ = span[kd]; s_[0] += 1; s_[1] -= float(T[f"{i}/tok_lp"][n]); s_[2] += int(T[f"{i}/tok_rank"][n] == 1)
                if has_s:
                    s_[3] += 1; s_[4] -= float(S[f"{i}/tok_lp"][n]); s_[5] -= float(T[f"{i}/tok_lp"][n]); s_[6] += int(S[f"{i}/tok_rank"][n] == 1)
            pieces = [P(x) for x in ids]
            text = "".join(pieces)
            cnt["replies"] += 1
            if text != tok.decode(ids):
                cnt["replies whose pieces do not join (skipped)"] += 1; continue
            starts, o = [], 0
            for p_ in pieces:
                starts.append(o); o += len(p_)
            R[key + (cn,)] = {"i": i, "idx": run, "ids": ids, "pieces": pieces, "toks": toks, "text": text, "starts": starts,
                              "t_end": text.find("<channel|>"), "s": has_s, "nc": {}}

    def at(rp, ch):
        return bisect.bisect_right(rp["starts"], ch) - 1

    def judge0(rp, ch, goods):
        """S0's judgement, kept as it was for lines b / c"""
        k = at(rp, ch); n = rp["idx"][k]; i = rp["i"]
        lead = rp["text"][rp["starts"][k]:ch]
        top1 = P(T[f"{i}/t_ids"][n][0]); me = rp["pieces"][k]
        return {"flagged": int(T[f"{i}/tok_rank"][n]) > 1,
                "corrected": bool(top1) and top1 != me and any((lead + g).startswith(top1) for g in goods)}

    def fits(p, lead, gs):
        """a token text that agrees with the lead, reaches the wrong character and continues a way of writing the right text"""
        return (len(p) > len(lead) and p.startswith(lead)
                and any(g.startswith(p) or DEINDENT.sub("\n", g).startswith(DEINDENT.sub("\n", p)) for g in gs))

    def right_mass(arr, i, n, lead, etext, gs):
        ids, lps = arr[f"{i}/t_ids"][n], arr[f"{i}/t_lp"][n]
        return sum(math.exp(float(lp)) for tid, lp in zip(ids, lps)
                   if lp > -1e3 and fits(P(tid), lead, gs) and not etext.startswith(P(tid)))

    def tail(i, n):
        return max(0.0, 1.0 - float(np.exp(T[f"{i}/t_lp"][n].astype(np.float64)).sum()))

    def judge(rp, ch, raws, mode):
        """at the token covering char ch of a reply; raws = the shown (raw) continuations from ch; mode = the reading
        under which E4B's text was matched (its own way of writing: as shown, or with JSON escapes). When E4B writes
        with escapes the wrong character can sit inside an escape sequence (backslash-n where backslash-quote is
        right): ch moves on over what E4B's text shares with the right text written E4B's way. The teacher's token
        counts as right in any way of writing (the matcher forgives them all)."""
        own = [OWN[mode](raw) for raw in raws]
        here = rp["text"][ch:ch + 8]
        adv = max([len(os.path.commonprefix([here, g[:8]])) for g in own], default=0)
        if adv:
            goods = [g[adv:] for g in own if g[:adv] == here[:adv] and len(g) > adv]
            ch += adv
        else:
            goods = [g for raw in raws for g in renderings(raw)]
        k = at(rp, ch); n = rp["idx"][k]; i = rp["i"]
        lead = rp["text"][rp["starts"][k]:ch]
        etext = rp["text"][rp["starts"][k]:rp["starts"][k] + 40]                     # E4B's own text from this token on
        gs = [lead + g for g in goods]
        own_gs = [lead + g[adv:] for g in own if len(g) > adv]
        top1 = P(T[f"{i}/t_ids"][n][0]); me = rp["pieces"][k]
        e_cons = bool(top1) and etext.startswith(top1)                               # the teacher's token is also E4B's text
        r_cons = fits(top1, lead, gs)
        t_me = float(T[f"{i}/tok_lp"][n])
        rm_t = right_mass(T, i, n, lead, etext, gs)
        out = {"flagged": int(T[f"{i}/tok_rank"][n]) > 1, "corrected": r_cons and not e_cons,
               "undet": top1 != me and e_cons and (len(top1) <= len(lead) or r_cons), "prefers": rm_t > math.exp(t_me),
               "sound": bool(own_gs) and len(me) > len(lead) and not any(g.startswith(me[:len(lead) + 1]) for g in own_gs),
               "me": me, "top1": top1, "rank": int(T[f"{i}/tok_rank"][n]), "push": None, "lift": None, "tail": tail(i, n),
               "k": k, "lead": lead, "right": gs[0][len(lead):len(lead) + 40] if gs else "", "adv": adv}
        if rp["s"]:
            out["push"] = float(S[f"{i}/tok_lp"][n]) - t_me                           # > 0: distillation lowers E4B's token
            if rm_t > 0:                                                              # > 0: it raises the right tokens
                rm_s = right_mass(S, i, n, lead, etext, gs)
                out["lift"] = math.log(rm_t) - (math.log(rm_s) if rm_s > 0 else float(S[f"{i}/t_lp"][n].min()))
        return out

    def rnorm(rj, part, layers, ws):
        """a reply part (thought / call) normalised, cached: (text, map, offset of the part in the reply)"""
        ck = (part, layers, ws)
        if ck not in rj["nc"]:
            lo = 0 if part == "thought" else max(rj["t_end"], 0)
            hi = rj["t_end"] if part == "thought" else len(rj["text"])
            rj["nc"][ck] = norm(rj["text"][lo:hi], layers, ws) + (lo,) if hi > lo else ("", [0], lo)
        return rj["nc"][ck]

    births, ctrl, sev, b_old, c_old, d_old = [], [], [], [], [], defaultdict(list)
    E, E_n, F, sub = Counter(), [0, 0], [], defaultdict(list)
    moved, shown, dump = [], 0, []
    ns_cache = {}
    for key, rec in recs.items():
        if rec is None:
            continue
        calls = {c["call"]: c for c in rec["calls"]}
        order = [c["call"] for c in rec["calls"]]
        res = tool_results(rec)
        seen_w, seen_o = set(), set()

        def nsrc(cj, ws):
            ck = (key, cj, ws)
            if ck not in ns_cache:
                if len(ns_cache) > 64:
                    ns_cache.clear()
                ns_cache[ck] = nsources(sources(calls[cj]["input_messages"]), ws)
            return ns_cache[ck]

        def first_occ(wn, ws, cn):
            """the earliest reply (≤ cn) holding the normalised text wn: (call, reply, part, text, map, offset, k, unscored earlier)"""
            gap = False
            for cj in order:
                rj = R.get(key + (cj,))
                if rj:
                    for part in ("thought", "call"):
                        seen_t = set()
                        for _, layers in VARIANTS:
                            nt, mp, lo = rnorm(rj, part, layers, ws)
                            if not nt or nt in seen_t:
                                continue
                            seen_t.add(nt)
                            k = nt.find(wn)
                            if k >= 0:
                                return cj, rj, part, nt, mp, lo, k, gap
                else:
                    gap = True
                if cj == cn:
                    break
            return None

        for cn in order:
            rp = R.get(key + (cn,))
            if not rp:
                continue
            c = calls[cn]; text = rp["text"]; t_end = rp["t_end"]; i = rp["i"]
            if t_end > 0:                                                             # f: where the teacher would end
                k_end = at(rp, t_end); ix = np.array(rp["idx"][:k_end], dtype=np.int64)
                pe = (np.exp(T[f"{i}/t_lp"][ix]) * (T[f"{i}/t_ids"][ix] == END)).sum(1) if k_end else np.zeros(0)
                first = next((k for k, v in enumerate(pe.tolist()) if v > 0.5), None)
                F.append((k_end, first, float(np.exp(T[f"{i}/tok_lp"][rp["idx"][k_end]]))))
            m_ = re.search(r"<\|tool_call>call:", text)
            tc = ((c.get("target") or {}).get("tool_calls") or [])
            if m_:                                                                    # e + SUBMIT
                kn = at(rp, m_.end())
                if rp["starts"][kn] == m_.end():
                    me, top1 = rp["pieces"][kn], P(T[f"{i}/t_ids"][rp["idx"][kn]][0])
                    E_n[0] += 1; E_n[1] += me == top1
                    if me != top1:
                        E[f"{me.strip()}→{top1.strip()}"] += 1
                    if tc and tc[0].get("name") == "submit_patch":
                        hist = [m for m in c["input_messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
                        names = [((m["tool_calls"][0].get("function") or {}).get("name")) for m in hist]
                        le = max([j for j, nm in enumerate(names) if nm in ("edit_file", "write_file")], default=-1)
                        cmds = [m for m, nm in zip(hist[le + 1:], names[le + 1:]) if nm == "run_command"]
                        cls = "no command since the last edit"
                        if cmds:
                            try:
                                ec = json.loads(res.get(cmds[-1]["tool_calls"][0].get("id"), "{}")).get("exit_code")
                            except (ValueError, AttributeError):
                                ec = None
                            cls = "last command exit 0" if ec == 0 else "last command exit ≠ 0"
                        sub[cls].append(top1.strip())
            if not tc or tc[0].get("name") != "edit_file":
                continue
            try:
                args = json.loads(tc[0]["args"]) if isinstance(tc[0]["args"], str) else tc[0]["args"]
            except (ValueError, KeyError, TypeError):
                cnt["edit calls with unreadable args"] += 1; continue
            old = args.get("old_string") if isinstance(args, dict) else None
            if not isinstance(old, str) or not old:
                continue
            out = outcome(res.get(tc[0].get("id")))
            cnt[f"edit calls {out}"] += 1
            mo = text.find('old_string:<|"|>')
            if mo < 0 or text[mo + 16: mo + 16 + len(old)] != old:
                cnt["edit calls whose old_string is not found as written in the reply (skipped)"] += 1; continue
            vs = mo + 16
            kq = at(rp, vs + len(old))                                                # d (S0): closing quote in the call
            if rp["starts"][kq] == vs + len(old) and rp["ids"][kq] == QUOTE:
                n = rp["idx"][kq]
                d_old[out].append((float(np.exp(T[f"{i}/tok_lp"][n])), int(T[f"{i}/t_ids"][n][0]) != QUOTE))
            if out == "several":                                                      # SEV: the first too-short anchor
                if old in seen_o:
                    cnt["sev: resends of an anchor already counted"] += 1; continue
                seen_o.add(old)
                hit = None
                for name, layers, ws in MODES:                                        # the reading under which it was shown
                    o_, om_ = norm(old, layers, ws)
                    fo = o_ and first_occ(o_, ws, cn)
                    if fo and conts_n(o_, nsrc(fo[0], ws)):
                        hit = (o_, ws, fo, name); break
                if not hit:
                    cnt["sev: the anchor is not in what was shown at its first occurrence (set apart)"] += 1; continue
                o_, ws, (cj, rj, part, nt, mp, lo, k, gap), sname = hit
                cs = conts_n(o_, nsrc(cj, ws))
                after = nt[k + len(o_):k + len(o_) + 120]
                m = max(len(os.path.commonprefix([after, s_[0][s_[3]:s_[3] + 120]])) for s_ in cs)
                goods = [s_ for s_ in cs if s_[0][s_[3]:s_[3] + m] == after[:m] and s_[3] + m < len(s_[0])]
                if not goods or m >= 120 or k + len(o_) + m >= len(nt):
                    cnt["sev: E4B's text follows a shown copy to its end or for 120+ chars (set apart)"] += 1; continue
                raws = [s_[2][s_[1][s_[3] + m]:s_[1][s_[3] + m] + 160] for s_ in goods]
                sev.append(dict(judge(rj, lo + mp[k + len(o_) + m], raws, sname), where=part))
                continue
            if out != "not_found":
                continue
            srcs = sources(c["input_messages"])
            L0 = longest_prefix(old, srcs)                                            # S0's b / c: the as-written locator
            if a.min_prefix <= L0 < len(old):
                conts_fail = [g for g in continuations(old[:L0], srcs) if g]
                in_thought = old[:L0 + 1] in (text[:t_end] if t_end > 0 else "")
                if conts_fail:
                    b_old.append(dict(judge0(rp, vs + L0, conts_fail), in_thought=in_thought))
                    if in_thought:
                        c_old.append(judge0(rp, text.find(old[:L0 + 1]) + L0, conts_fail))
            loc = locate(old, {ws: [x[0] for x in nsrc(cn, ws)] for ws in ("strip", "collapse")})
            if loc is None:
                cnt["birth: empty after normalising (skipped)"] += 1; continue
            if loc[0] == "whole":
                cnt["birth: the whole string was shown once escapes / whitespace are forgiven (stale view or wrong file; set apart)"] += 1
                cnt[f"   … explained by the reading '{loc[1]}'"] += 1
                continue
            p0, (mname, layers, ws), o, om, ln = loc
            if ln < a.min_prefix:
                cnt["birth: not locatable (explained prefix < min chars: written from memory or elsewhere)"] += 1; continue
            wn = o[:ln + 1]
            if (mname, wn) in seen_w:
                cnt["birth: resends of a wrong text already counted"] += 1; continue
            seen_w.add((mname, wn))
            fo = first_occ(wn, ws, cn)
            if not fo:
                cnt["birth: the wrong text is not found in any scored reply (skipped)"] += 1; continue
            cj, rj, part, nt, mp, lo, k, gap = fo
            sb = nsrc(cj, ws)
            lb = longest_prefix(o, [x[0] for x in sb])
            if lb < ln:
                cnt["birth: written before the source was shown (set apart)"] += 1; continue
            if lb > ln:                                                               # right as shown then: stale later
                cnt["birth: matched what was shown at its first occurrence (the file / view changed later; set apart)"] += 1; continue
            cs = conts_n(o[:ln], sb)
            if not cs:
                cnt["birth: the prefix ends its source (set apart)"] += 1; continue
            if len({s_[0][s_[3]] for s_ in cs}) > 1:
                cnt["birth: the shown continuation is not unique (set apart)"] += 1; continue
            raws = []
            for s_ in cs:
                r_ = s_[2][s_[1][s_[3]]:s_[1][s_[3]] + 160]
                if r_ and r_ not in raws:
                    raws.append(r_)
            ch = lo + mp[k + ln]
            r_ = judge(rj, ch, raws, mname)
            first_try = not any("FileEditError" in (m.get("content") or "") for m in calls[cj]["input_messages"] if m.get("role") == "tool")
            births.append(dict(r_, where=part, first_try=first_try, same=cj == cn, mode=mname))
            cnt["birth: measured"] += 1
            cnt[CHK] += r_["sound"]
            if gap:
                cnt["birth: measured, but an earlier reply of the trajectory is not scored (the first occurrence may be earlier)"] += 1
            if L0 < len(old) and p0 != L0:
                moved.append(p0 - L0)
            k0 = at(rj, lo + mp[k])                                                   # CTRL: the tokens E4B copied right
            k0 += rj["starts"][k0] < lo + mp[k]
            for kk in range(k0, r_["k"]):
                n = rj["idx"][kk]; ii = rj["i"]
                ctrl.append((int(T[f"{ii}/tok_rank"][n]) > 1,
                             float(S[f"{ii}/tok_lp"][n]) - float(T[f"{ii}/tok_lp"][n]) if rj["s"] else None))
            if a.dump_births:
                dump.append({"row": rj["i"], "n": int(rj["idx"][r_["k"]]), "lead": r_["lead"], "me": r_["me"], "right": r_["right"],
                             "id": key[2], "call": cj, "fail": cn, "mode": mname, "p0": p0, "L0": L0, "len": len(old)})
            if shown < a.examples:
                shown += 1
                print(f"\n--- birth example {shown}: {key[2]} ({key[1]}) first written in call {cj} ({part}), failing call {cn} · "
                      f"reading '{mname}' · explained prefix {ln} normalised chars · as-written locator at char {L0}, this one at "
                      f"{p0} of {len(old)} · E4B {r_['me']!r} (lead {r_['lead']!r}) rank {r_['rank']} · teacher top-1 {r_['top1']!r} · "
                      f"right continuation as written {r_['right'][:30]!r}")
                print(f"    as written around the wrong character: {old[max(0, p0 - 50):p0]!r} | {old[p0:p0 + 12]!r}")
    if a.dump_births:
        with open(a.dump_births, "w") as w:
            for x in dump:
                w.write(json.dumps(x) + "\n")
    # ---- TAIL / STUDENT / MAP: one pass over every scored reply ----
    mp_ = defaultdict(lambda: [0, 0.0, 0.0])
    tails = defaultdict(list)
    allA = []
    for key, rp in R.items():
        i = rp["i"]; kinds = span_kinds(rp["toks"]); te = rp["t_end"]
        fence = [m.start() for m in re.finditer("```", rp["text"][:te] if te > 0 else "")]
        fk = ["thought code (inside ```)" if kd == "thought" and bisect.bisect_right(fence, rp["starts"][k]) % 2 == 1
              else "thought prose" if kd == "thought" else kd for k, kd in enumerate(kinds)]
        ix = np.array(rp["idx"], dtype=np.int64)
        tl = np.clip(1.0 - np.exp(T[f"{i}/t_lp"][ix].astype(np.float64)).sum(1), 0.0, 1.0)
        for kd, v in zip(fk, tl.tolist()):
            tails[kd].append(v)
        if rp["s"]:
            A = (S[f"{i}/tok_lp"][ix].astype(np.float64) - T[f"{i}/tok_lp"][ix].astype(np.float64)).tolist()
            for kd, v in zip(fk, A):
                m_ = mp_[kd]; m_[0] += 1; m_[1] += v; m_[2] += max(v, 0.0); allA.append((v, kd))
    # ---- print ----
    items = [f"{k} {v}" for k, v in sorted(cnt.items())]
    print()
    for k in range(0, len(items), 3):
        print("counts: " + " · ".join(items[k:k + 3]))
    print("a  per span (teacher NLL of E4B's tokens, top-1 agreement): " + " · ".join(
        f"{kd} {s_[0]:,} tok NLL {s_[1] / max(s_[0], 1):.3f} top-1 {100 * s_[2] / max(s_[0], 1):.0f} %" for kd, s_ in sorted(span.items())))
    print(f"b  (S0's as-written locator, at the failing call: kept to match the S0 photo) n {len(b_old)} · flagged "
          f"{pc(b_old, lambda x: x['flagged'])} · corrected {pc(b_old, lambda x: x['corrected'])} · wrong text already in the reply's "
          f"thought {sum(x['in_thought'] for x in b_old)} (corrected {pc([x for x in b_old if x['in_thought']], lambda x: x['corrected'])})")
    print(f"c  (S0's locator, same text in the failing reply's thought) n {len(c_old)} · flagged {pc(c_old, lambda x: x['flagged'])} · corrected "
          f"{pc(c_old, lambda x: x['corrected'])}")
    print("d  (S0, closing old_string in the call) teacher P(close) median / top-1 ≠ close: " + " · ".join(
        f"{k} n {len(v)} {q([x[0] for x in v]):.2f} / {pc(v, lambda x: x[1])}" for k, v in sorted(d_old.items())))
    print(f"e  tool name, first token: E4B = teacher top-1 {100 * E_n[1] / max(E_n[0], 1):.0f} % of {E_n[0]} · swaps (E4B→teacher) "
          + ", ".join(f"{k} {v}" for k, v in E.most_common(8)))
    ended = [x for x in F if x[1] is not None]
    print(f"f  thoughts {len(F)} (tokens p50 {q([x[0] for x in F])}) · teacher P(end) > 0.5 before E4B's end in "
          f"{pc(F, lambda x: x[1] is not None)} · at a median {q([x[1] / max(x[0], 1) for x in ended]):.2f} of the thought · "
          f"teacher P(end) at E4B's own end: median {q([x[2] for x in F]):.2f}")
    det = [x for x in births if not x["undet"]]

    def bl(xs, tag):
        return (f"{tag} n {len(xs)} · flagged {pc(xs, lambda x: x['flagged'])} · corrected {pc(xs, lambda x: x['corrected'])} · "
                f"prefers {pc(xs, lambda x: x['prefers'])}")
    print(f"LOCATOR the new first wrong character against S0's as-written one: moved in {len(moved)} of {len(births)} measured births"
          + (f" (by a median of {q([abs(m) for m in moved])} chars, later in {sum(m > 0 for m in moved)})" if moved else "")
          + " · readings used: " + ", ".join(f"{k} {v}" for k, v in Counter(x["mode"] for x in births).most_common()))
    print(f"BIRTH  where each wrong text is first written — measured {len(births)} · undetermined (the teacher's top-1 stops before "
          f"the wrong character) {len(births) - len(det)} · the verdict line: " + bl(det, "determined"))
    print("       by place: " + " · ".join(bl([x for x in det if x["where"] == w], w) for w in ("thought", "call")))
    print("       " + " · ".join(bl([x for x in det if x["first_try"] == f], "before any failed edit" if f else "after a failed edit")
                                 for f in (True, False)) + f" · first written in the failing reply (its thought or its call) {sum(x['same'] for x in det)}")
    cflag = sum(x[0] for x in ctrl) / max(len(ctrl), 1)
    bflag = sum(x["flagged"] for x in det) / max(len(det), 1)
    cpush = [x[1] for x in ctrl if x[1] is not None]
    print(f"CTRL   the tokens of the same quotes that E4B copied right: n {len(ctrl):,} · flagged {100 * cflag:.1f} % (births "
          f"{100 * bflag:.0f} % → {bflag / cflag:.1f}× the control)" if cflag > 0 else
          f"CTRL   the tokens of the same quotes that E4B copied right: n {len(ctrl):,} · flagged 0 % (births {100 * bflag:.0f} %)")
    if S:
        pushes = [x["push"] for x in det if x["push"] is not None]
        lifts = [x["lift"] for x in det if x["lift"] is not None]
        print(f"       push on E4B's token (log p_E4B − log p_31B; > 0 for any teacher on average): births n {len(pushes)} median "
              f"{q(pushes):.2f} · > 1 nat {pc(pushes, lambda v: v > 1)} — control n {len(cpush):,} median {q(cpush):.2f} · > 1 nat "
              f"{pc(cpush, lambda v: v > 1)} · lift of the right tokens (log p_31B − log p_E4B, lower bound) n {len(lifts)} median "
              f"{q(lifts):.2f} · > 1 nat {pc(lifts, lambda v: v > 1)}")
    print(f"SEV    first too-short anchor, at the first point E4B leaves every shown copy: n {len(sev)} · teacher would go on "
          f"{pc(sev, lambda x: x['corrected'])} · teacher top-1 = E4B's token {pc(sev, lambda x: not x['flagged'])} · undetermined "
          f"{sum(x['undet'] for x in sev)}" + (f" · push median {q([x['push'] for x in sev if x['push'] is not None]):.2f}" if S else ""))
    print("SUBMIT teacher's top-1 at E4B's submit_patch decisions: " + " · ".join(
        f"{k}: n {len(v)} run {pc(v, lambda t: t == 'run')} submit {pc(v, lambda t: t == 'submit')}" for k, v in sorted(sub.items())))
    worst = 0.0
    parts = []
    for kd, v in sorted(tails.items()):
        s10 = sum(1 for x in v if x > 0.10) / max(len(v), 1); worst = max(worst, s10)
        parts.append(f"{kd} {len(v):,} tok mean {100 * sum(v) / max(len(v), 1):.2f} % median {100 * q(v):.2f} % · > 1 % at "
                     f"{pc(v, lambda x: x > 0.01)} · > 10 % at {100 * s10:.1f} %")
    bt = [x["tail"] for x in births]
    print(f"TAIL   teacher mass outside its top-{K} (what a top-{K} bucketed KL cannot see): " + " · ".join(parts)
          + f" · at the births: median {100 * q(bt):.2f} % max {100 * max(bt, default=0):.0f} %")
    print(f"       → by the BOX_PLAN B-4 rule (> 10 % at under 5 % of the positions of every span): "
          + (f"top-{K} is enough for S1" if worst < 0.05 else f"score S1 with K = 100 (worst span {100 * worst:.1f} %)"))
    if S:
        bad = [kd for kd, s_ in span.items() if s_[3] and s_[4] >= s_[5]]
        print("STUDENT E4B's own NLL of its tokens / the 31B's NLL there / E4B token = E4B top-1: " + " · ".join(
            f"{kd} {s_[4] / max(s_[3], 1):.3f} / {s_[5] / max(s_[3], 1):.3f} / {100 * s_[6] / max(s_[3], 1):.0f} %"
            for kd, s_ in sorted(span.items()) if s_[3])
            + (" → PASS (E4B's is the lower one on every span)" if not bad else
               f" → FAIL on {', '.join(sorted(bad))}: the scoring model is not the sampling model — stop, send the photo"))
        tot = sum(m_[2] for m_ in mp_.values()) or 1.0
        top = sorted(allA, key=lambda x: x[0], reverse=True)[:max(1, len(allA) // 100)]
        tc_ = Counter(kd for _, kd in top)
        print("MAP    where the signal lands (log p_E4B − log p_31B per E4B token): " + " · ".join(
            f"{kd} {m_[0]:,} tok mean {m_[1] / max(m_[0], 1):.3f} positive mass {100 * m_[2] / tot:.0f} % · share of the largest 1 % "
            f"{100 * tc_[kd] / len(top):.0f} %" for kd, m_ in sorted(mp_.items())))
    nb = len(det)
    cor = sum(x["corrected"] for x in det) / max(nb, 1)
    pre = sum(x["prefers"] for x in det) / max(nb, 1)
    spec = bflag >= 5 * cflag
    if nb < 20:
        verdict = "TOO FEW (n < 20): no verdict — score more of E4B's rollouts and pool them (BOX_PLAN B-4), send the photo"
    elif cor >= 0.5:
        verdict = "GO to S1"
    elif cor < 0.3:
        verdict = "STOP the line"
    else:
        verdict = ("GO to S1 (between: prefers ≥ 60 % and flagged at births ≥ 5× the control)" if pre >= 0.6 and spec else
                   f"STOP the line (between: prefers {100 * pre:.0f} % {'≥' if pre >= 0.6 else '<'} 60 %, flagged at births "
                   f"{'≥' if spec else '<'} 5× the control)")
    print(f"== S0b verdict by the BOX_PLAN B-4 rule (determined births, n ≥ 20: corrected ≥ 50 % go · < 30 % stop · between: go only "
          f"if prefers ≥ 60 % and flagged ≥ 5× the control): n {nb} · corrected {100 * cor:.0f} % · prefers {100 * pre:.0f} % · flagged "
          f"{100 * bflag:.0f} % vs control {100 * cflag:.1f} % → {verdict} · locator check "
          f"{cnt[CHK]} / {len(births)}")
    print("== kd_probe done (📷 from the first counts line down)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
