#!/usr/bin/env python3
"""Audit of the label mask in a rendered training file (B-11 step 1b). Independent of train/render.py.

    .venv-eval/bin/python train/label_audit.py runs/sft/X.rendered.jsonl --tokenizer <hf snapshot dir or model id> \\
        --expect tool_call [--records runs/sft/X] [--show 3] [--no-thought-tokens]
    .venv-eval/bin/python train/label_audit.py --selftest        # a hand-built sequence, no tokenizer needed

render.py decides the labels from the reply it rendered. This script does not use that: it reads only
{input_ids, labels}, walks the tokens with the Gemma 4 template's control tokens and puts every token in one class:

  bos · turn_header (`<|turn>` + role + newline, and `<turn|>` of a system / user turn) · system · tool_decl
  (`<|tool>` … `<tool|>`) · user (task prompt, nudges, compaction summaries) · tool_response (the response body and
  its closing tag; also a `<|tool_response>` that follows another response) · thought (`<|channel>` … `<channel|>`,
  tags included) · tool_call (`<|tool_call>` … `<tool_call|>`, tags, name and arguments) · stop_tool (the
  `<|tool_response>` the model emits right after its tool calls) · stop_turn (`<turn|>` that ends a model turn) ·
  text (anything else in a model turn)

and then counts, per class, the tokens and the labelled tokens. --expect names what must hold:

  reply      system, tool_decl, user, tool_response, turn_header, bos: 0 labelled.
  action     as reply, and thought: 0 labelled.
  tool_call  as action, and text: 0 labelled, stop_turn: 0 labelled.
  always     a tool call is labelled completely or not at all (never in part), and the stop token after a run of
             tool calls has the same state as those calls.

A tool call without labels is history or a turn that is no target (a failed call, a schema violation): the file alone
cannot tell which. --records DIR (the call-level records the file was rendered from) closes that: the number of
completely labelled tool calls must equal the number of tool calls in the target of every call the rows list
(targets that are null are excluded turns and must add none). --no-thought-tokens (ablation B): the file must hold
no thought token at all. Any failed assertion → exit 1. --show N prints N sequences with the labelled runs inside
⟦ ⟧. Outputs quote task text: keep them out of git.
"""
import argparse
import collections
import glob
import json
import os
import sys

CLASSES = ["bos", "turn_header", "system", "tool_decl", "user", "tool_response", "thought", "tool_call", "stop_tool",
           "stop_turn", "text"]
ZERO = {"reply": ["bos", "turn_header", "system", "tool_decl", "user", "tool_response"]}
ZERO["action"] = ZERO["reply"] + ["thought"]
ZERO["tool_call"] = ZERO["action"] + ["text", "stop_turn"]


def classify(toks):
    """→ (class per token, tool-call spans [(start, end)], stop_tool positions with the spans they close)"""
    cls, calls, stops = [], [], []
    role, i, n = None, 0, len(toks)
    run = []                                   # tool-call spans since the last tool response / text / turn start
    while i < n:
        t = toks[i]
        if t == "<bos>":
            cls.append("bos"); i += 1
        elif t == "<|turn>":
            role = toks[i + 1] if i + 1 < n else None
            role = {"model": "model", "user": "user", "system": "system"}.get(role, role)
            k = 3 if i + 2 < n and toks[i + 2].strip("\n") == "" and "\n" in toks[i + 2] else 2
            cls += ["turn_header"] * min(k, n - i); i += k; run = []
        elif t == "<turn|>":
            cls.append("stop_turn" if role == "model" else "turn_header"); i += 1; role = None; run = []
        elif t == "<|tool>":
            j = i
            while j < n and toks[j] != "<tool|>":
                j += 1
            cls += ["tool_decl"] * (min(j, n - 1) - i + 1); i = j + 1
        elif role == "model" and t == "<|channel>":
            j = i
            while j < n and toks[j] != "<channel|>":
                j += 1
            cls += ["thought"] * (min(j, n - 1) - i + 1); i = j + 1
        elif role == "model" and t == "<|tool_call>":
            j = i
            while j < n and toks[j] != "<tool_call|>":
                j += 1
            e = min(j, n - 1) + 1
            cls += ["tool_call"] * (e - i); calls.append((i, e)); run.append((i, e)); i = e
        elif t == "<|tool_response>":
            if role == "model" and run:        # emitted by the model: it ends the run of tool calls
                cls.append("stop_tool"); stops.append((i, list(run))); run = []; i += 1
            else:
                cls.append("tool_response"); i += 1
            j = i
            while j < n and toks[j] != "<tool_response|>":
                j += 1
            e = min(j, n - 1) + 1
            cls += ["tool_response"] * (e - i); i = e
        else:
            if role is None and not t.strip():                       # the newline the template puts between two turns
                cls.append("turn_header")
            else:
                cls.append({"model": "text", "user": "user", "system": "system"}.get(role, "text"))
            i += 1
    assert len(cls) == n, (len(cls), n)
    return cls, calls, stops


def audit_row(toks, labels, expect):
    """→ (Counter tokens per class, Counter labelled per class, problems, number of fully labelled tool calls)"""
    cls, calls, stops = classify(toks)
    tot, lab, prob = collections.Counter(cls), collections.Counter(), []
    for c, x in zip(cls, labels):
        lab[c] += x != -100
    full = 0
    state = {}
    for s, e in calls:
        k = sum(x != -100 for x in labels[s:e])
        if k not in (0, e - s):
            prob.append(f"tool call at {s}:{e} labelled in part ({k} of {e - s})")
        state[(s, e)] = k == e - s; full += k == e - s
    for pos, run in stops:
        st = {state[x] for x in run}
        if len(st) > 1:
            prob.append(f"tool calls before the stop token at {pos} differ in label state")
        elif (labels[pos] != -100) != st.pop():
            prob.append(f"stop token at {pos} labelled {labels[pos] != -100}, its tool calls the opposite")
    for c in ZERO[expect]:
        if lab[c]:
            prob.append(f"{lab[c]} labelled token(s) in class {c}")
    return tot, lab, prob, full


def show(toks, labels, tok, width=260):
    cls, _, _ = classify(toks)
    out, i, n = [], 0, len(toks)
    while i < n:
        j = i
        on = labels[i] != -100
        while j < n and (labels[j] != -100) == on and cls[j] == cls[i]:
            j += 1
        text = tok.convert_tokens_to_string(toks[i:j]) if tok else " ".join(toks[i:j])
        if len(text) > width:
            text = text[:width // 2] + f" …[{j - i} tokens]… " + text[-width // 4:]
        out.append(f"  {'⟦' if on else ' '}{cls[i]:13}{'⟧' if on else ' '} {text!r}")
        i = j
    return "\n".join(out)


def selftest():
    T = ("<bos> <|turn> system \n S1 S2 <|tool> D1 D2 <tool|> <turn|> <|turn> user \n U1 U2 <turn|> <|turn> model \n "
         "<|channel> thought \n H1 H2 <channel|> <|tool_call> call:a {x} <tool_call|> <|tool_response> R1 R2 <tool_response|> "
         "<|channel> thought \n H3 <channel|> X1 X2 <|tool_call> call:b {y} <tool_call|> <|tool_call> call:c {z} <tool_call|> "
         "<|tool_response> R3 <tool_response|> <|tool_response> R4 <tool_response|> X3 X4 <turn|> <|turn> user \n N1 <turn|> "
         "<|turn> model \n <|channel> thought \n H4 <channel|> <|tool_call> call:d {} <tool_call|> <|tool_response>").split(" ")
    T = [t if t != "\\n" else "\n" for t in T]
    cls, calls, stops = classify(T)
    by = collections.Counter(cls)
    want = {"bos": 1, "turn_header": 3 + 1 + 3 + 1 + 3 + 3 + 1 + 3, "system": 2, "tool_decl": 4, "user": 3, "thought": 6 + 5 + 5,
            "tool_call": 4 * 4, "stop_tool": 3, "tool_response": 3 + 2 + 3, "text": 4, "stop_turn": 1}
    assert dict(by) == want, (dict(by), want)
    assert len(calls) == 4 and [len(r) for _, r in stops] == [1, 2, 1], (calls, stops)

    def labels(scope, excluded=()):
        L = [-100] * len(T)
        for k, (s, e) in enumerate(calls):
            if k in excluded:
                continue
            L[s:e] = [1] * (e - s)
        for pos, run in stops:
            if not any(calls.index(x) in excluded for x in run):
                L[pos] = 1
        if scope in ("reply", "action"):
            for i, c in enumerate(cls):
                if c in ("text", "stop_turn") or (scope == "reply" and c == "thought" and T[i] not in ("<|channel>", "thought", "\n")):
                    L[i] = 1
        return L
    for scope in ("reply", "action", "tool_call"):
        for exp in ("reply", "action", "tool_call"):
            _, _, prob, full = audit_row(T, labels(scope), exp)
            ok = ["reply", "action", "tool_call"].index(scope) >= ["reply", "action", "tool_call"].index(exp)
            assert (not prob) == ok, (scope, exp, prob)
            assert full == 4
    _, _, prob, full = audit_row(T, labels("tool_call", excluded={0}), "tool_call")
    assert not prob and full == 3, (prob, full)                       # an excluded turn: no label, stop token included
    L = labels("tool_call"); L[calls[1][0] + 1] = -100                 # one argument token unlabelled
    assert any("in part" in p for p in audit_row(T, L, "tool_call")[2])
    L = labels("tool_call"); L[stops[0][0]] = -100                     # the stop token forgotten
    assert any("stop token" in p for p in audit_row(T, L, "tool_call")[2])
    L = labels("tool_call"); L[T.index("R1")] = 1                      # a tool response trained
    assert any("tool_response" in p for p in audit_row(T, L, "tool_call")[2])
    L = labels("tool_call"); L[T.index("N1")] = 1                      # a nudge trained
    assert any("class user" in p for p in audit_row(T, L, "tool_call")[2])
    L = labels("tool_call"); L[T.index("X1")] = 1                      # plain text trained
    assert any("class text" in p for p in audit_row(T, L, "tool_call")[2]) and not audit_row(T, L, "action")[2]
    L = labels("tool_call"); L[T.index("H3")] = 1                      # a thought token trained
    assert any("class thought" in p for p in audit_row(T, L, "action")[2]) and not audit_row(T, L, "reply")[2]
    print("label_audit selftest: classes, 3 scopes × 3 expectations, an excluded turn, 6 planted faults → all as expected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rendered", nargs="?", help="rendered jsonl ({input_ids, labels, calls, instance_id})")
    ap.add_argument("--tokenizer")
    ap.add_argument("--expect", choices=["reply", "action", "tool_call"], default="tool_call")
    ap.add_argument("--records", default=None, help="dir of the call-level records: count the target tool calls")
    ap.add_argument("--no-thought-tokens", action="store_true", help="ablation B: no thought token may be in the file")
    ap.add_argument("--show", type=int, default=0, help="print N sequences with the labelled runs marked")
    ap.add_argument("--limit", type=int, default=0, help="debug: first rows only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.rendered or not a.tokenizer:
        ap.error("rendered file and --tokenizer are needed (or --selftest)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"label_audit · file {a.rendered} · expect {a.expect} · records {a.records or '(not given: target count not checked)'}"
          f" · no-thought-tokens {a.no_thought_tokens} · show {a.show} · limit {a.limit or 'all'}")
    TOT, LAB = collections.Counter(), collections.Counter()
    problems, rows, full_calls, listed, shares = [], 0, 0, [], []
    for line in open(a.rendered):
        if not line.strip():
            continue
        r = json.loads(line); rows += 1
        toks = tok.convert_ids_to_tokens(r["input_ids"])
        tot, lab, prob, full = audit_row(toks, r["labels"], a.expect)
        TOT.update(tot); LAB.update(lab); full_calls += full
        shares.append(sum(lab.values()) / max(1, len(toks)))
        problems += [f"{r.get('instance_id')} segment {r.get('segment')}: {p}" for p in prob]
        if not sum(lab.values()):
            problems.append(f"{r.get('instance_id')} segment {r.get('segment')}: a sequence without any label")
        listed.append((r.get("record") or r.get("instance_id"), r.get("calls") or []))
        if rows <= a.show:
            print(f"-- sequence {rows}: {r.get('instance_id')} segment {r.get('segment')} calls {r.get('calls')} · "
                  f"{len(toks)} tokens, {sum(lab.values())} labelled")
            print(show(toks, r["labels"], tok))
        if a.limit and rows >= a.limit:
            break
    print(f"== label audit: {rows} sequences · {sum(TOT.values()):,} tokens · {sum(LAB.values()):,} labelled "
          f"({100 * sum(LAB.values()) / max(1, sum(TOT.values())):.1f} %) · labelled share per sequence min {min(shares):.3f} "
          f"median {sorted(shares)[len(shares) // 2]:.3f} max {max(shares):.3f}")
    print(f"   {'class':14} {'tokens':>12} {'labelled':>10}   required for --expect {a.expect}")
    for c in CLASSES:
        need = "0 labelled" if c in ZERO[a.expect] else ("all or none per call; the labelled ones are the targets" if c == "tool_call"
                                                         else "as its tool calls" if c == "stop_tool" else "free")
        bad = c in ZERO[a.expect] and LAB[c]
        print(f"   {c:14} {TOT[c]:12,} {LAB[c]:10,}   {need}{'   ← FAIL' if bad else ''}")
    print(f"   tool calls labelled completely: {full_calls}")
    if a.no_thought_tokens and TOT["thought"]:
        problems.append(f"{TOT['thought']} thought token(s) in a file that must hold none")
    if a.records:
        recs = {}
        for f in glob.glob(os.path.join(a.records, "*.json")):
            j = json.load(open(f)); calls = {c["call"]: c for c in j["calls"]}
            recs[os.path.basename(f)] = calls; recs.setdefault(j["instance_id"], calls)   # rows name the record file (older rows: the id)
        want = miss = null = textonly = 0
        for iid, calls in listed:
            for k in calls:
                c = recs.get(iid, {}).get(k)
                if c is None:
                    miss += 1; continue
                t = c.get("target")
                if not t:
                    null += 1; continue
                n = len(t.get("tool_calls") or [])
                want += n; textonly += not n
        print(f"   records: target tool calls of the listed calls {want} · listed calls with no target (excluded turns) {null} "
              f"· targets without a tool call (text only) {textonly} · listed calls not found in the records {miss}")
        if a.expect == "tool_call" and want != full_calls:
            problems.append(f"labelled tool calls {full_calls} ≠ target tool calls in the records {want}")
        if miss:
            problems.append(f"{miss} listed call(s) not found in --records")
    if problems:
        print(f"LABEL AUDIT FAIL: {len(problems)} problem(s)")
        for p in problems[:25]:
            print("   -", p)
        sys.exit(1)
    print("LABEL AUDIT PASS")


if __name__ == "__main__":
    main()
