#!/usr/bin/env python3
"""S0 / S0b of the token-level distillation plan (BOX_PLAN B-4): does the 31B teacher point at E4B's errors, and where
would a distillation signal land? Read-only, CPU.

    .venv-eval/bin/python train/kd_probe.py runs/opd/s0/rows.jsonl --npz 'runs/opd/s0/t31_*.npz' --recs runs/opd/s0 \
        [--student-npz 'runs/opd/s0/s_*.npz'] --tokenizer google/gemma-4-E4B-it-qat-w4a16-ct

Inputs: the rows teacher_score.py scored (render.py --pack rows of E4B's own trajectories, each with `tag` = its pass and
`calls` = the record's call numbers, one contiguous label run per call), the teacher's npz shards, optionally the same
rows scored by E4B itself (teacher_score.py --model <E4B>: the student's own log-probabilities), and the call-level
records (`<recs>/recs_<tag>/<instance_id>.json`, convert_traj.py --mask-errors off) for tool results and contexts.
Every reply's text is rebuilt token by token (per-token decode; a reply whose pieces do not join to its full decode is
skipped and counted). "Shown" text = read_file content, command stdout/stderr with grep -n / cat -n prefixes removed,
tool diff/output fields, the agent's own earlier edit new_strings.

S0b (2026-11-10, decided before any first-occurrence data was seen; the rule is in BOX_PLAN B-4):
  BIRTH  every not-found edit's wrong text W = old_string up to and including its first character that no shown text
         continues (the longest verbatim prefix must be ≥ --min-prefix chars). Resends of the same W in a task count
         once. W's FIRST occurrence in the trajectory (a thought or a call, at or before the failing call) is where the
         error was made; measured there, with what was shown at that point. Set apart (not in the verdict): written
         before the source was shown (the prefix found at birth is shorter), right as shown then (a longer prefix was
         shown at birth: the file or the view changed later), the shown continuation is not unique (the prefix repeats
         with different next characters), the prefix ends its source.
         At the first wrong token: flagged = E4B's token is not the teacher's top-1 · corrected = the teacher's top-1
         continues the shown text · prefers = the teacher gives the right token more probability than E4B's · with
         --student-npz: push = log p_E4B − log p_31B of E4B's token (> 0: distillation lowers it) and lift = log p_31B −
         log p_E4B of the right token (> 0: distillation raises it).
  SEV    every several-matches edit's old_string: its first occurrence; at the first point after it where E4B's text
         leaves every shown continuation: would the teacher go on (its top-1 continues a shown copy) or stop as E4B did.
  SUBMIT at each submit_patch decision (first token of the tool name): the teacher's top-1, by the last command since the
         last edit (exit 0 · exit ≠ 0 · none).
  MAP    (needs --student-npz) where the signal log p_E4B − log p_31B lands: thought prose · code inside ``` in a thought ·
         tool name · tool arguments · text.
S0 lines kept for comparison: a (per span), b / c (the failing call and its thought: mostly copies of text already in
context), d (closing old_string in the failing call), e (tool name swaps), f (where the teacher would end the thought).
"""
import argparse
import bisect
import glob
import json
import os
import re
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


def continuations(prefix, srcs, n=120):
    """the shown text after every occurrence of prefix"""
    out = []
    for s in srcs:
        k = s.find(prefix)
        while k >= 0:
            out.append(s[k + len(prefix):k + len(prefix) + n]); k = s.find(prefix, k + 1)
    return out


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


def load_npz(pattern):
    arr, files = {}, sorted(glob.glob(pattern)) if pattern else []
    for f in files:
        z = np.load(f)
        for k in z.files:
            arr[k] = z[k]
    return arr, files


def pc(xs, f):
    return f"{100 * sum(1 for x in xs if f(x)) / max(len(xs), 1):.0f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--npz", required=True, help="glob of the teacher's npz shards")
    ap.add_argument("--student-npz", default="", help="glob of the same rows scored by E4B itself (optional)")
    ap.add_argument("--recs", required=True, help="dir holding recs_<tag>/ (convert_traj --mask-errors off)")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--min-prefix", type=int, default=12, help="chars: a shorter verbatim prefix = not locatable")
    ap.add_argument("--examples", type=int, default=0, help="print this many birth cases in full (laptop checks)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    END, QUOTE = tok.convert_tokens_to_ids("<channel|>"), tok.convert_tokens_to_ids('<|"|>')
    T, tfiles = load_npz(a.npz)
    S, sfiles = load_npz(a.student_npz)
    rows = [json.loads(l) for l in open(a.rows)]
    print(f"kd_probe · rows {len(rows)} · teacher npz {len(tfiles)} files ({a.npz}) · student npz {len(sfiles)} files "
          f"({a.student_npz or 'none: no push / map lines'}) · recs {a.recs} · tokenizer {a.tokenizer} · min prefix {a.min_prefix} chars")
    cnt = Counter()
    piece_of = {}
    P = lambda x: piece_of.setdefault(int(x), tok.decode([int(x)]))
    # ---- index: one entry per scored reply, keyed (tag, instance, call) ----
    R, recs = {}, {}
    span = defaultdict(lambda: [0, 0.0, 0])
    for i, row in enumerate(rows):
        if f"{i}/pos" not in T:
            cnt["rows not scored by the teacher (over max_len / missing shard)"] += 1; continue
        pos = T[f"{i}/pos"].tolist()
        if pos != [j for j in range(1, len(row["input_ids"])) if row["labels"][j] != -100]:
            cnt["rows whose scored positions differ from the labels (skipped)"] += 1; continue
        has_s = bool(S) and f"{i}/pos" in S and S[f"{i}/pos"].tolist() == pos
        if S and not has_s:
            cnt["rows the student scores do not cover (no push / map there)"] += 1
        runs = runs_of(pos)
        if len(runs) != len(row["calls"]):
            cnt["rows whose reply runs ≠ calls (skipped)"] += 1; continue
        key = (row.get("tag", ""), row["instance_id"])
        if key not in recs:
            p = os.path.join(a.recs, f"recs_{key[0]}", f"{key[1]}.json")
            recs[key] = json.load(open(p)) if os.path.exists(p) else None
        if recs[key] is None:
            cnt["rows without their record (skipped)"] += 1; continue
        cnt["rows used"] += 1
        for run, cn in zip(runs, row["calls"]):
            ids = [row["input_ids"][pos[n]] for n in run]
            toks = tok.convert_ids_to_tokens(ids)
            for n, kd in zip(run, span_kinds(toks)):
                s_ = span[kd]; s_[0] += 1; s_[1] -= float(T[f"{i}/tok_lp"][n]); s_[2] += int(T[f"{i}/tok_rank"][n] == 1)
            pieces = [P(x) for x in ids]
            text = "".join(pieces)
            cnt["replies"] += 1
            if text != tok.decode(ids):
                cnt["replies whose pieces do not join (skipped)"] += 1; continue
            starts, o = [], 0
            for p_ in pieces:
                starts.append(o); o += len(p_)
            R[key + (cn,)] = {"i": i, "idx": run, "ids": ids, "pieces": pieces, "toks": toks, "text": text, "starts": starts,
                              "t_end": text.find("<channel|>"), "s": has_s}

    def at(rp, ch):
        return bisect.bisect_right(rp["starts"], ch) - 1

    def judge(rp, ch, goods):
        """at the token covering char ch of a reply; goods = the shown continuations from ch"""
        k = at(rp, ch); n = rp["idx"][k]; i = rp["i"]
        tids, tlp = T[f"{i}/t_ids"][n], T[f"{i}/t_lp"][n]
        lead = rp["text"][rp["starts"][k]:ch]
        gs = [lead + g for g in goods]
        top1 = P(tids[0]); me = rp["pieces"][k]
        corrected = bool(top1) and top1 != me and any(g.startswith(top1) for g in gs)
        cid = tok(gs[0], add_special_tokens=False)["input_ids"][:1]
        cid = cid[0] if cid else -1
        oracle = cid >= 0 and P(cid) != me and gs[0].startswith(P(cid))
        lpc = tlp[tids == cid]
        prefers = bool(len(lpc)) and float(lpc[0]) > float(T[f"{i}/tok_lp"][n])
        out = {"flagged": int(T[f"{i}/tok_rank"][n]) > 1, "corrected": corrected, "prefers": prefers, "oracle": oracle,
               "me": me, "top1": top1, "rank": int(T[f"{i}/tok_rank"][n]), "push": None, "lift": None}
        if rp["s"]:
            out["push"] = float(S[f"{i}/tok_lp"][n]) - float(T[f"{i}/tok_lp"][n])      # > 0: distillation lowers E4B's token
            sids, slp = S[f"{i}/t_ids"][n], S[f"{i}/t_lp"][n]
            sc = slp[sids == cid]
            if len(lpc):                                                              # > 0: it raises the right token
                out["lift"] = float(lpc[0]) - (float(sc[0]) if len(sc) else float(slp.min()))
        return out

    births, sev, b_old, c_old, d_old = [], [], [], [], defaultdict(list)
    E, E_n, F, sub = Counter(), [0, 0], [], defaultdict(list)
    shown = 0
    for key, rec in recs.items():
        if rec is None:
            continue
        calls = {c["call"]: c for c in rec["calls"]}
        order = [c["call"] for c in rec["calls"]]
        res = tool_results(rec)
        seen_w, seen_o = set(), set()
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

            def first_occ(s):                                                         # earliest reply holding s (≤ cn)
                for cj in order:
                    rj = R.get(key + (cj,))
                    if rj:
                        th = rj["text"][:rj["t_end"]] if rj["t_end"] > 0 else ""
                        k = th.find(s)
                        if k >= 0:
                            return cj, rj, k, "thought"
                        k = rj["text"].find(s, max(rj["t_end"], 0))
                        if k >= 0:
                            return cj, rj, k, "call"
                    if cj == cn:
                        break
                return cn, rp, vs, "call"
            if out == "several":                                                      # SEV: the first too-short anchor
                if old in seen_o:
                    cnt["sev: resends of an anchor already counted"] += 1; continue
                seen_o.add(old)
                cj, rj, k, where = first_occ(old)
                conts = continuations(old, sources(calls[cj]["input_messages"]))
                if not conts:
                    cnt["sev: the anchor is not in what was shown at its first occurrence (set apart)"] += 1; continue
                after = rj["text"][k + len(old):k + len(old) + 120]
                m = max(len(os.path.commonprefix([after, g])) for g in conts)
                goods = [g[m:] for g in conts if g.startswith(after[:m]) and len(g) > m]
                if not goods or k + len(old) + m >= len(rj["text"]):
                    cnt["sev: E4B's text follows a shown copy to its end (set apart)"] += 1; continue
                sev.append(dict(judge(rj, k + len(old) + m, goods), where=where))
                continue
            if out != "not_found":
                continue
            srcs = sources(c["input_messages"])
            L = longest_prefix(old, srcs)
            if L >= len(old):
                cnt["birth: the whole string was shown (stale view / escaping; set apart)"] += 1; continue
            if L < a.min_prefix:
                cnt["birth: not locatable (prefix < min chars: written from memory or elsewhere)"] += 1; continue
            conts_fail = [g for g in continuations(old[:L], srcs) if g]
            in_thought = old[:L + 1] in (text[:t_end] if t_end > 0 else "")
            if conts_fail:                                                            # S0's b / c at the failing call
                b_old.append(dict(judge(rp, vs + L, conts_fail), in_thought=in_thought))
                if in_thought:
                    c_old.append(judge(rp, text.find(old[:L + 1]) + L, conts_fail))
            W = old[:L + 1]
            if W in seen_w:
                cnt["birth: resends of a wrong text already counted"] += 1; continue
            seen_w.add(W)
            cj, rj, k, where = first_occ(W)
            sb = sources(calls[cj]["input_messages"])
            Lb = longest_prefix(old, sb)
            if Lb < L:
                cnt["birth: written before the source was shown (set apart)"] += 1; continue
            if Lb > L:                                                                # right as shown then: stale later
                cnt["birth: matched what was shown at its first occurrence (the file / view changed later; set apart)"] += 1; continue
            conts = [g for g in continuations(old[:L], sb) if g]
            if not conts:
                cnt["birth: the prefix ends its source (set apart)"] += 1; continue
            if len({g[:1] for g in conts}) > 1:
                cnt["birth: the shown continuation is not unique (set apart)"] += 1; continue
            r_ = judge(rj, k + L, conts)
            first_try = not any("FileEditError" in (m.get("content") or "") for m in calls[cj]["input_messages"] if m.get("role") == "tool")
            births.append(dict(r_, where=where, first_try=first_try, same=cj == cn))
            cnt["birth: measured"] += 1
            cnt["birth check: an oracle teacher would count as corrected"] += r_["oracle"]
            if shown < a.examples:
                shown += 1
                print(f"\n--- birth example {shown}: {key[1]} ({key[0]}) first written in call {cj} ({where}), failing call {cn} · "
                      f"prefix {L} chars · E4B {r_['me']!r} rank {r_['rank']} · teacher top-1 {r_['top1']!r} · shown continues "
                      f"{conts[0][:30]!r}")
                print(f"    wrong text tail {W[-50:]!r}")
    # ---- MAP: where log p_E4B − log p_31B lands ----
    mp = defaultdict(lambda: [0, 0.0, 0.0])
    allA = []
    if S:
        for key, rp in R.items():
            if not rp["s"]:
                continue
            i = rp["i"]; kinds = span_kinds(rp["toks"]); te = rp["t_end"]
            fence = [m.start() for m in re.finditer("```", rp["text"][:te] if te > 0 else "")]
            for k, (n, kd) in enumerate(zip(rp["idx"], kinds)):
                if kd == "thought":
                    kd = "thought code (inside ```)" if bisect.bisect_right(fence, rp["starts"][k]) % 2 == 1 else "thought prose"
                A = float(S[f"{i}/tok_lp"][n]) - float(T[f"{i}/tok_lp"][n])
                m_ = mp[kd]; m_[0] += 1; m_[1] += A; m_[2] += max(A, 0.0); allA.append((A, kd))
    # ---- print ----
    items = [f"{k} {v}" for k, v in sorted(cnt.items())]
    print()
    for k in range(0, len(items), 3):
        print("counts: " + " · ".join(items[k:k + 3]))
    print("a  per span (teacher NLL of E4B's tokens, top-1 agreement): " + " · ".join(
        f"{kd} {s_[0]:,} tok NLL {s_[1] / max(s_[0], 1):.3f} top-1 {100 * s_[2] / max(s_[0], 1):.0f} %" for kd, s_ in sorted(span.items())))
    print(f"b  (S0, at the failing call: mostly copies) n {len(b_old)} · flagged {pc(b_old, lambda x: x['flagged'])} · corrected "
          f"{pc(b_old, lambda x: x['corrected'])} · wrong text already in the reply's thought {sum(x['in_thought'] for x in b_old)} "
          f"(corrected {pc([x for x in b_old if x['in_thought']], lambda x: x['corrected'])})")
    print(f"c  (S0, same text in the failing reply's thought) n {len(c_old)} · flagged {pc(c_old, lambda x: x['flagged'])} · corrected "
          f"{pc(c_old, lambda x: x['corrected'])}")
    print("d  (S0, closing old_string in the call) teacher P(close) median / top-1 ≠ close: " + " · ".join(
        f"{k} n {len(v)} {q([x[0] for x in v]):.2f} / {pc(v, lambda x: x[1])}" for k, v in sorted(d_old.items())))
    print(f"e  tool name, first token: E4B = teacher top-1 {100 * E_n[1] / max(E_n[0], 1):.0f} % of {E_n[0]} · swaps (E4B→teacher) "
          + ", ".join(f"{k} {v}" for k, v in E.most_common(8)))
    ended = [x for x in F if x[1] is not None]
    print(f"f  thoughts {len(F)} (tokens p50 {q([x[0] for x in F])}) · teacher P(end) > 0.5 before E4B's end in "
          f"{pc(F, lambda x: x[1] is not None)} · at a median {q([x[1] / max(x[0], 1) for x in ended]):.2f} of the thought · "
          f"teacher P(end) at E4B's own end: median {q([x[2] for x in F]):.2f}")

    def bl(xs, tag):
        return (f"{tag} n {len(xs)} · flagged {pc(xs, lambda x: x['flagged'])} · corrected {pc(xs, lambda x: x['corrected'])} · "
                f"prefers {pc(xs, lambda x: x['prefers'])}")
    print("BIRTH  where each wrong text is first written (the verdict line): " + bl(births, "all"))
    print("       by place: " + " · ".join(bl([x for x in births if x["where"] == w], w) for w in ("thought", "call")))
    print("       " + " · ".join(bl([x for x in births if x["first_try"] == f], "before any failed edit" if f else "after a failed edit")
                                 for f in (True, False)) + f" · first written in the failing call itself {sum(x['same'] for x in births)}")
    if S:
        pushes = [x["push"] for x in births if x["push"] is not None]
        lifts = [x["lift"] for x in births if x["lift"] is not None]
        print(f"       push on E4B's wrong token (log p_E4B − log p_31B; > 0 = distillation lowers it): n {len(pushes)} · median "
              f"{q(pushes):.2f} · > 1 nat {pc(pushes, lambda v: v > 1)} · lift of the right token (log p_31B − log p_E4B) n {len(lifts)} "
              f"median {q(lifts):.2f} · > 1 nat {pc(lifts, lambda v: v > 1)}")
    print(f"SEV    first too-short anchor, at the first point E4B leaves every shown copy: n {len(sev)} · teacher would go on "
          f"{pc(sev, lambda x: x['corrected'])} · teacher top-1 = E4B's token {pc(sev, lambda x: not x['flagged'])}"
          + (f" · push median {q([x['push'] for x in sev if x['push'] is not None]):.2f}" if S else ""))
    print("SUBMIT teacher's top-1 at E4B's submit_patch decisions: " + " · ".join(
        f"{k}: n {len(v)} run {pc(v, lambda t: t == 'run')} submit {pc(v, lambda t: t == 'submit')}" for k, v in sorted(sub.items())))
    if S:
        tot = sum(m_[2] for m_ in mp.values()) or 1.0
        top = sorted(allA, key=lambda x: x[0], reverse=True)[:max(1, len(allA) // 100)]
        tc_ = Counter(kd for _, kd in top)
        print("MAP    where the signal lands (log p_E4B − log p_31B per E4B token): " + " · ".join(
            f"{kd} {m_[0]:,} tok mean {m_[1] / max(m_[0], 1):.3f} positive mass {100 * m_[2] / tot:.0f} % top-1 % {100 * tc_[kd] / len(top):.0f} %"
            for kd, m_ in sorted(mp.items())))
    nb = len(births)
    cor = sum(x["corrected"] for x in births) / max(nb, 1)
    verdict = ("TOO FEW (n < 20): the laptop decides" if nb < 20 else "GO to S1" if cor >= 0.5 else "STOP the line" if cor < 0.3
               else "BETWEEN: the laptop decides")
    print(f"== S0b verdict by the BOX_PLAN B-4 rule (BIRTH line, corrected ≥ 50 % go · < 30 % stop · n ≥ 20): n {nb} · corrected "
          f"{100 * cor:.0f} % → {verdict} · oracle check {cnt['birth check: an oracle teacher would count as corrected']} / {nb}")
    print("== kd_probe done (📷 from the first counts line down)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
