#!/usr/bin/env python3
"""Template-driven renderer for a non-Gemma model (B-9 part 3: the control on Qwen3.5-9B). Call-level records
(train/mc_prep.py, scripts/req_log.py summarize) → {input_ids, labels} rows for train/sft_lora.py.

    .venv-eval/bin/python train/render_tmpl.py runs/sft/mc/r1k --tokenizer <snapshot dir> --out runs/sft/mc/r1k.q.jsonl
    .venv-eval/bin/python train/render_tmpl.py runs/verified/probe_q9/records --tokenizer <dir> \
        --check runs/verified/probe_q9/server_prompts          # = what vLLM built? (A2)
    .venv-eval/bin/python train/render_tmpl.py --selftest --tokenizer <any ChatML tokenizer dir> [--template file.jinja]

No model-specific text is written here: the prompt of a call is tokenizer.apply_chat_template(its input messages, tools,
add_generation_prompt) and its reply is what the same template appends when the target is added as an assistant
message (up to and including the end-of-turn token). The messages are first put in the shape vLLM 0.19.1 hands the
template (entrypoints/chat_utils.py, read 2026-11-14): tool-call arguments parsed to a mapping, the earlier thought
from `reasoning` (what scripts/req_log.py --rename-reasoning sends, THINK_HIST=1) else `reasoning_content`.

Sequences. A call's training sample is tokens(prompt) + tokens(reply), labels on the reply only. Consecutive calls
share one sequence while the next call's prompt TOKENS start with the sequence so far (then only the new part is
appended): the same prompts the server builds, each reply labelled once. A new sequence starts where that fails:
after a harness user message (the template stops rendering the thoughts before it) or where the tokenizer merges
across a reply boundary (e.g. an empty thought: "<think>\\n" + "\\n</think>" → one "\\n\\n" token in later prompts).
Printed: calls, label tokens, sequences, breaks by reason, lengths, sequences over --max-len (not removed here:
sft_lora.py drops them and prints the count). Every knob prints its value. Rows quote task text → runs/ only.
"""
import argparse
import collections
import glob
import json
import os
import sys


def text_of(c):
    if c is None:
        return ""
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def wire_message(m):
    """one request message → what the chat template receives from vLLM 0.19.1"""
    role = m.get("role")
    out = {"role": role, "content": text_of(m.get("content"))}
    if role == "assistant":
        r = m.get("reasoning") if m.get("reasoning") is not None else m.get("reasoning_content")
        if r:
            out["reasoning_content"] = r
        tcs = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or tc
            args = fn.get("arguments", fn.get("args"))
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            tcs.append({"type": "function", "function": {"name": fn["name"], "arguments": args}})
        if tcs:
            out["tool_calls"] = tcs
    return out


def target_message(t):
    return wire_message({"role": "assistant", "content": t.get("content") or "", "reasoning_content": t.get("reasoning") or "",
                         "tool_calls": [{"function": {"name": tc["name"], "arguments": tc.get("args") or "{}"}}
                                        for tc in t.get("tool_calls") or []]})


class Renderer:
    def __init__(self, tok, template=None):
        self.tok, self.template = tok, template
        self.eot = tok.eos_token if isinstance(tok.eos_token, str) else "<|im_end|>"

    def text(self, messages, tools, gen):
        kw = {"chat_template": self.template} if self.template else {}
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=gen, tokenize=False,
                                            enable_thinking=True, **kw)

    def ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def call(self, c, tools):
        """→ (prompt text, reply text): the reply ends with the end-of-turn token"""
        msgs = [wire_message(m) for m in c["input_messages"]]
        p = self.text(msgs, tools, True)
        full = self.text(msgs + [target_message(c["target"])], tools, False)
        if not full.startswith(p):
            raise ValueError("the template renders the prompt differently once the reply is added")
        reply = full[len(p):].rstrip("\n")
        if not reply.endswith(self.eot):
            raise ValueError(f"reply does not end with {self.eot}: {reply[-40:]!r}")
        return p, reply


def render_record(R, rec, stats, check=None, fails=None):
    rows, S, L, cs, prev = [], None, None, [], None

    def flush():
        if S is not None:
            rows.append({"instance_id": rec["instance_id"], "segment": cs[0][1], "calls": [k for k, _ in cs],
                         "input_ids": S, "labels": L})
    for c in rec["calls"]:
        try:
            p, reply = R.call(c, rec.get("tools"))
        except (ValueError, KeyError, TypeError) as e:
            stats["calls skipped (malformed)"] += 1
            if fails is not None and len(fails) < 12:
                fails.append(f"MALFORMED {rec['instance_id']} call {c['call']}: {type(e).__name__}: {str(e)[:160]}")
            continue
        pid, rid = R.ids(p), R.ids(reply)
        if R.tok.decode(rid) != reply:
            stats["replies that do not decode back"] += 1
        if check is not None:
            ref = check.get(c["call"])
            if ref is not None:
                stats["A2 calls"] += 1
                te, ne = ref["prompt"] == p, ref["rendered_tokens"] == len(pid)
                stats["A2 text equal"] += te; stats["A2 token count equal"] += ne
                if not (te and ne) and fails is not None and len(fails) < 12:
                    k = next((i for i, (x, y) in enumerate(zip(p, ref["prompt"])) if x != y), min(len(p), len(ref["prompt"])))
                    fails.append(f"FAIL A2 {rec['instance_id']} call {c['call']}: text_equal={te} tokens {len(pid)} vs "
                                 f"{ref['rendered_tokens']} · first diff at char {k}: ours={p[k:k + 60]!r} server={ref['prompt'][k:k + 60]!r}")
        stats["calls"] += 1; stats["label tokens"] += len(rid)
        empty = not (c["target"].get("reasoning") or "").strip()
        stats["replies without a thought"] += empty
        if S is not None and len(pid) >= len(S) and pid[:len(S)] == S:
            L = L + [-100] * (len(pid) - len(S)) + rid
            S = pid + rid
            cs.append((c["call"], c.get("segment", 0)))
        else:
            if S is not None:
                why = ("a user message" if c.get("segment", 0) != cs[-1][1] else
                       "an empty thought" if (empty or prev) else "another token boundary")
                stats[f"break: {why}"] += 1
            flush()
            S, L, cs = pid + rid, [-100] * len(pid) + rid, [(c["call"], c.get("segment", 0))]
        prev = empty
    flush()
    return rows


def run(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    R = Renderer(tok, open(a.template).read() if a.template else None)
    files = sorted(f for f in glob.glob(os.path.join(a.records, "*.json")) if not os.path.basename(f).startswith("_"))
    i, n = (int(x) for x in a.shard.split("/"))
    files = files[i::n]
    print(f"render_tmpl · records {a.records} ({len(files)} files, shard {a.shard}) · tokenizer {a.tokenizer} · template "
          f"{a.template or 'the tokenizer own'} · end of turn {R.eot!r} · max len {a.max_len} (counted only) · check {a.check or 'off'} · out {a.out or 'none'}",
          flush=True)
    stats, fails, lens = collections.Counter(), [], []
    out = open(a.out, "w") if a.out else None
    for f in files:
        rec = json.load(open(f))
        if not isinstance(rec, dict) or not isinstance(rec.get("calls"), list):
            continue
        check = None
        if a.check:
            rp = os.path.join(a.check, f"{rec['instance_id']}.json")
            check = {c["call"]: c for c in json.load(open(rp))["calls"]} if os.path.exists(rp) else {}
        rows = render_record(R, rec, stats, check, fails)
        stats["records"] += 1; stats["sequences"] += len(rows)
        for r in rows:
            lens.append(len(r["input_ids"])); stats["tokens"] += len(r["input_ids"])
            stats["labels in rows"] += sum(1 for x in r["labels"] if x != -100)
            stats[f"sequences over {a.max_len}"] += len(r["input_ids"]) > a.max_len
            stats[f"label tokens in sequences over {a.max_len}"] += sum(1 for x in r["labels"] if x != -100) if len(r["input_ids"]) > a.max_len else 0
            if out:
                out.write(json.dumps(r) + "\n")
    if out:
        out.close()
    lens.sort()
    stats["len p50"], stats["len max"] = (lens[len(lens) // 2], lens[-1]) if lens else (0, 0)
    if stats["labels in rows"] != stats["label tokens"]:
        fails.append(f"FAIL label tokens in the rows {stats['labels in rows']} != per-call sum {stats['label tokens']}")
    if a.check and (stats["A2 text equal"] != stats["A2 calls"] or stats["A2 token count equal"] != stats["A2 calls"] or not stats["A2 calls"]):
        fails.append(f"FAIL A2: text equal {stats['A2 text equal']} / token count equal {stats['A2 token count equal']} of {stats['A2 calls']} calls")
    for x in fails:
        print(x)
    br = " · ".join(f"{k[7:]} {v}" for k, v in sorted(stats.items()) if k.startswith("break: ")) or "none"
    print(f"calls {stats['calls']} · label tokens {stats['label tokens']} · records {stats['records']} → sequences {stats['sequences']} "
          f"({stats['tokens']:,} tokens, len p50 {stats['len p50']} max {stats['len max']}) · breaks: {br} · replies without a thought "
          f"{stats['replies without a thought']} · over {a.max_len}: {stats[f'sequences over {a.max_len}']} sequences "
          f"({stats[f'label tokens in sequences over {a.max_len}']} label tokens) · malformed calls {stats['calls skipped (malformed)']} "
          f"· replies not decoding back {stats['replies that do not decode back']}"
          + (f" · A2 text equal {stats['A2 text equal']}/{stats['A2 calls']}, token count equal {stats['A2 token count equal']}/{stats['A2 calls']}" if a.check else ""))
    hard = [x for x in fails if x.startswith("FAIL")]
    print(f"RESULT {'PASS' if not hard else 'FAIL'} ({len(hard)} FAIL lines, {len(fails) - len(hard)} malformed shown)")
    if a.stats:
        json.dump({**dict(stats), "fail": len(hard)}, open(a.stats, "w"))
    return 0 if not hard else 1


def selftest(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    R = Renderer(tok, open(a.template).read() if a.template else None)
    tools = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    def asst(th, cmd, text=""):
        return {"role": "assistant", "content": text, "reasoning_content": th,
                "tool_calls": [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]}
    tool = lambda s: {"role": "tool", "content": s, "tool_call_id": "x"}
    msgs = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the parser."},
            asst("Let me look at the files.", "ls"), tool("a.py\nb.py"),
            asst("Read a.py first.", "cat a.py"), tool("def f(): pass"),
            asst("", "cat b.py"), tool("x = 1"),                                   # empty thought
            asst("Now edit.", "sed -i s/1/2/ b.py"), tool(""),
            {"role": "user", "content": "Please submit."},                         # harness message → new segment
            asst("Submit.", "echo done"), tool("done"), asst("Finished.", "echo COMPLETE")]
    calls, users = [], 0
    for i, m in enumerate(msgs):
        users += m["role"] == "user"
        if m["role"] == "assistant":
            calls.append({"call": len(calls) + 1, "segment": users - 1, "input_messages": msgs[:i],
                          "target": {"reasoning": m["reasoning_content"], "content": m["content"],
                                     "tool_calls": [{"name": "bash", "args": m["tool_calls"][0]["function"]["arguments"]}]}})
    stats = collections.Counter()
    rows = render_record(R, {"instance_id": "t", "tools": tools, "calls": calls}, stats)
    n_lab = sum(1 for r in rows for x in r["labels"] if x != -100)
    assert stats["calls"] == 6 and n_lab == stats["label tokens"], (stats, n_lab)
    assert [c for r in rows for c in r["calls"]] == [1, 2, 3, 4, 5, 6]
    for r in rows:                                    # every labelled span = one whole reply ending with the end of turn
        assert len(r["input_ids"]) == len(r["labels"])
        spans, cur = [], []
        for t, l in zip(r["input_ids"], r["labels"]):
            if l != -100:
                assert l == t; cur.append(t)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        assert len(spans) == len(r["calls"]), (len(spans), r["calls"])
        for s in spans:
            assert tok.decode(s).endswith(R.eot), tok.decode(s)[-30:]
        # the sequence up to each reply = the prompt the server would build for that call
        for k in r["calls"]:
            p, reply = R.call(calls[k - 1], tools)
            pid = R.ids(p)
            assert r["input_ids"][:len(pid)] == pid, k
    first = rows[0]
    p1, r1 = R.call(calls[0], tools)
    assert first["input_ids"][:len(R.ids(p1))] == R.ids(p1) and all(x == -100 for x in first["labels"][:len(R.ids(p1))])
    assert stats["break: a user message"] == 1, stats
    print(f"  {len(rows)} sequences for 6 calls · calls per sequence {[r['calls'] for r in rows]} · breaks "
          f"{ {k: v for k, v in stats.items() if k.startswith('break')} }")
    print("  first reply:", repr(r1[:120]))
    print("render_tmpl selftest PASS (labels = whole replies ending with the end-of-turn token, each call once, prompts are prefixes, user message breaks)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="?", help="dir of call-level JSON records")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--template", default=None, help="a chat template file instead of the tokenizer's own (tests)")
    ap.add_argument("--check", default=None, help="dir of server-rendered prompts <id>.json (req_log.py summarize)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=None, help="write the counters as JSON (for the sharded launcher)")
    ap.add_argument("--max-len", type=int, default=30720, help="count the sequences longer than this (not removed)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    if not a.records:
        sys.exit("records dir required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
