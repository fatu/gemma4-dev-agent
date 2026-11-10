#!/usr/bin/env python3
"""S0 of the token-level distillation plan (BOX_PLAN B-4): does the 31B teacher point at E4B's errors? Read-only, CPU.

    .venv-eval/bin/python train/kd_probe.py runs/opd/s0/rows.jsonl --npz 'runs/opd/s0/t31_*.npz' --recs runs/opd/s0 \
        --tokenizer google/gemma-4-E4B-it-qat-w4a16-ct

Inputs: the rows teacher_score.py scored (render.py --pack rows of E4B's own trajectories, each with `tag` = its pass and
`calls` = the record's call numbers, one contiguous label run per call), the shards' npz files, and the call-level
records (`<recs>/recs_<tag>/<instance_id>.json`, convert_traj.py --mask-errors off) for each call's tool result and
context. Every reply's text is rebuilt token by token (per-token decode; a run whose pieces do not join to the full decode
is skipped and counted).
Lines (each with its counts):
  a  per span (thought / name / args / text): teacher NLL of the student's tokens, top-1 agreement.
  b  edit_file calls whose old_string was NOT FOUND: the longest prefix of old_string found verbatim in the context's
     tool outputs (read_file content, command stdout/stderr with grep -n / cat -n prefixes removed, earlier edit_file
     new_strings) marks the first wrong character; at the token covering it: the student's token rank under the teacher
     (flagged = rank > 1) and whether the teacher's top-1 token continues the source text (corrected). A prefix shorter
     than --min-prefix chars = not locatable (written from memory / elsewhere), counted apart; the whole string found =
     stale / escaping, counted apart. Split by whether the same wrong text is already in this reply's thought.
  c  the same wrong text inside this reply's THOUGHT (first occurrence): the same two numbers at its first wrong
     character; split into born here (not in an earlier thought of the context) and carried (already in one).
  d  edit_file calls that hit several matches vs ok edits: the teacher's probability of closing old_string (the
     <|"|> token after it) and how often its top-1 is something else (it would have written more).
  e  tool choice: at the first token of the tool name, student vs teacher top-1 (agreement, the most common swaps).
  f  thought end: where the teacher's P(<channel|>) first exceeds 0.5 inside the student's thought (share of replies,
     median position as a fraction of the thought), and the teacher's P at the student's own <channel|>.
The reading rule is in BOX_PLAN B-4.
"""
import argparse
import bisect
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_score import runs_of, span_kinds  # noqa: E402

GREP_PREFIX = re.compile(r"^(?:[^\n:]{1,300}:\d+[:-]|\s*\d+\t)", re.M)    # grep -n / grep -rn "path:12:" · cat -n "    12\t"


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
                    out.append(a["new_string"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--npz", required=True, help="glob of the shards' npz files")
    ap.add_argument("--recs", required=True, help="dir holding recs_<tag>/ (convert_traj --mask-errors off)")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--min-prefix", type=int, default=12, help="chars: a shorter verbatim prefix = not locatable")
    ap.add_argument("--examples", type=int, default=0, help="print this many b cases in full (laptop checks)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    END, QUOTE = tok.convert_tokens_to_ids("<channel|>"), tok.convert_tokens_to_ids('<|"|>')
    arr = {}
    files = sorted(glob.glob(a.npz))
    for f in files:
        z = np.load(f)
        for k in z.files:
            arr[k] = z[k]
    rows = [json.loads(l) for l in open(a.rows)]
    print(f"kd_probe · rows {len(rows)} · npz {len(files)} files ({a.npz}) · recs {a.recs} · tokenizer {a.tokenizer} · "
          f"min prefix {a.min_prefix} chars · sources: read_file content, command stdout/stderr (grep -n / cat -n prefixes removed), "
          f"tool diff/output fields, the agent's own earlier edit new_strings")
    cnt = Counter()
    span = defaultdict(lambda: [0, 0.0, 0])
    B = defaultdict(list); C = defaultdict(list); D = defaultdict(list); E = Counter(); E_n = [0, 0]; F = []
    shown = 0
    rec_cache = {}
    for i, row in enumerate(rows):
        if f"{i}/pos" not in arr:
            cnt["rows not scored (over max_len or missing shard)"] += 1; continue
        pos = arr[f"{i}/pos"].tolist()
        want = [j for j in range(1, len(row["input_ids"])) if row["labels"][j] != -100]
        if pos != want:
            cnt["rows whose scored positions differ from the labels (skipped)"] += 1; continue
        t_ids, t_lp, tok_lp, tok_rank = arr[f"{i}/t_ids"], arr[f"{i}/t_lp"], arr[f"{i}/tok_lp"], arr[f"{i}/tok_rank"]
        runs = runs_of(pos)
        if len(runs) != len(row["calls"]):
            cnt["rows whose reply runs ≠ calls (skipped)"] += 1; continue
        key = (row.get("tag", ""), row["instance_id"])
        if key not in rec_cache:
            p = os.path.join(a.recs, f"recs_{row.get('tag', '')}", f"{row['instance_id']}.json")
            rec_cache[key] = json.load(open(p)) if os.path.exists(p) else None
        rec = rec_cache[key]
        if rec is None:
            cnt["rows without their record (skipped)"] += 1; continue
        calls = {c["call"]: c for c in rec["calls"]}
        res = tool_results(rec)
        cnt["rows used"] += 1
        for run, cn in zip(runs, row["calls"]):
            c = calls[cn]
            ids = [row["input_ids"][pos[n]] for n in run]
            toks = tok.convert_ids_to_tokens(ids)
            for n, kd in zip(run, span_kinds(toks)):
                s_ = span[kd]; s_[0] += 1; s_[1] -= float(tok_lp[n]); s_[2] += int(tok_rank[n] == 1)
            cnt["replies"] += 1
            pieces = [tok.decode([x]) for x in ids]
            text = "".join(pieces)
            if text != tok.decode(ids):
                cnt["replies whose pieces do not join (skipped for b-f)"] += 1; continue
            starts, o = [], 0
            for p_ in pieces:
                starts.append(o); o += len(p_)
            at = lambda ch: bisect.bisect_right(starts, ch) - 1                         # token covering char ch
            t_end = text.find("<channel|>")
            # f: thought end
            if t_end > 0:
                k_end = at(t_end)
                ix = np.array(run[:k_end], dtype=np.int64)
                pe = (np.exp(t_lp[ix]) * (t_ids[ix] == END)).sum(1) if k_end else np.zeros(0)
                first = next((k for k, v in enumerate(pe.tolist()) if v > 0.5), None)
                F.append((k_end, first, float(np.exp(tok_lp[run[k_end]]))))
            # e: tool choice at the first token of the name
            m_ = re.search(r"<\|tool_call>call:", text)
            if m_:
                kn = at(m_.end())
                if starts[kn] == m_.end():
                    sname, tname = pieces[kn], tok.decode([int(t_ids[run[kn], 0])])
                    E_n[0] += 1; E_n[1] += sname == tname
                    if sname != tname:
                        E[f"{sname.strip()}→{tname.strip()}"] += 1
            tc = (c.get("target") or {}).get("tool_calls") or []
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
            thought = text[:t_end] if t_end > 0 else ""
            # d: closing quote after old_string
            kq = at(vs + len(old))
            if starts[kq] == vs + len(old) and ids[kq] == QUOTE:
                D[out].append((float(np.exp(tok_lp[run[kq]])), int(t_ids[run[kq], 0]) != QUOTE))
            if out != "not_found":
                continue
            srcs = sources(c["input_messages"])
            L = longest_prefix(old, srcs)
            if L >= len(old):
                cnt["b: not found but the whole string is in the context (stale / escaping)"] += 1; continue
            if L < a.min_prefix:
                cnt["b: not locatable (prefix < min chars: written from memory or elsewhere)"] += 1; continue
            src = next(x for x in srcs[::-1] if old[:L] in x)
            corr = src[src.find(old[:L]) + L:][:80]
            if not corr:
                cnt["b: the prefix ends its source (skipped)"] += 1; continue
            wrong = old[:L + 1]
            in_thought = wrong in thought
            earlier = any(wrong in (m.get("reasoning_content") or "") for m in c["input_messages"] if m.get("role") == "assistant")

            def judge(ch):                       # ch = absolute char of the first wrong character in `text`
                k = at(ch)
                good = text[starts[k]:ch] + corr  # what the source says from this token's start
                top1 = tok.decode([int(t_ids[run[k], 0])])
                o = tok.decode(tok(good, add_special_tokens=False)["input_ids"][:1])   # what an oracle teacher's top-1 would be
                cnt["b/c check: an oracle teacher would count as corrected"] += bool(o) and o != pieces[k] and good.startswith(o)
                cnt["b/c check: cases"] += 1
                return int(tok_rank[run[k]]) > 1, bool(top1) and top1 != pieces[k] and good.startswith(top1), k, top1
            fl, co, k, top1 = judge(vs + L)
            B["thought has it" if in_thought else "not in thought"].append((fl, co))
            if shown < a.examples:
                shown += 1
                print(f"\n--- b example {shown}: {row['instance_id']} ({row.get('tag')}) call {cn} · prefix {L} chars · "
                      f"student token {pieces[k]!r} rank {int(tok_rank[run[k]])} · teacher top-1 {top1!r} · source continues {corr[:30]!r}")
                print(f"    old_string[:L+20] = {old[:L + 20]!r}")
            if in_thought:
                fl2, co2, _, _ = judge(thought.find(wrong) + L)
                C["carried" if earlier else "born here"].append((fl2, co2))
    pc = lambda xs, f: f"{100 * sum(f(x) for x in xs) / max(len(xs), 1):.0f} %"
    items = [f"{k} {v}" for k, v in sorted(cnt.items())]
    print()
    for k in range(0, len(items), 4):
        print("counts: " + " · ".join(items[k:k + 4]))
    print("a  per span: " + " · ".join(f"{kd} {s_[0]:,} tok NLL {s_[1] / max(s_[0], 1):.3f} top-1 {100 * s_[2] / max(s_[0], 1):.0f} %"
                                       for kd, s_ in sorted(span.items())))
    allb = [x for v in B.values() for x in v]
    print(f"b  not-found edits, first wrong token in the CALL: locatable {len(allb)} · flagged (student token rank > 1) "
          f"{pc(allb, lambda x: x[0])} · corrected (teacher top-1 continues the source) {pc(allb, lambda x: x[1])} · "
          + " · ".join(f"{k}: n {len(v)} flagged {pc(v, lambda x: x[0])} corrected {pc(v, lambda x: x[1])}" for k, v in sorted(B.items())))
    allc = [x for v in C.values() for x in v]
    print(f"c  the same wrong text in the THOUGHT: n {len(allc)} · flagged {pc(allc, lambda x: x[0])} · corrected {pc(allc, lambda x: x[1])} · "
          + " · ".join(f"{k}: n {len(v)} flagged {pc(v, lambda x: x[0])} corrected {pc(v, lambda x: x[1])}" for k, v in sorted(C.items())))
    print("d  closing old_string, teacher P(close) median / teacher top-1 ≠ close: " + " · ".join(
        f"{k} n {len(v)} {q([x[0] for x in v]):.2f} / {pc(v, lambda x: x[1])}" for k, v in sorted(D.items())))
    print(f"e  tool name, first token: student = teacher top-1 {100 * E_n[1] / max(E_n[0], 1):.0f} % of {E_n[0]} · swaps (student→teacher) "
          + ", ".join(f"{k} {v}" for k, v in E.most_common(8)))
    ended = [x for x in F if x[1] is not None]
    print(f"f  thoughts {len(F)} (tokens p50 {q([x[0] for x in F])}) · teacher P(end) > 0.5 before the student's end in "
          f"{pc(F, lambda x: x[1] is not None)} · at a median {q([x[1] / max(x[0], 1) for x in ended]):.2f} of the thought · "
          f"teacher P(end) at the student's own end: median {q([x[2] for x in F]):.2f}")
    print("== kd_probe done (📷 from the counts line down)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
