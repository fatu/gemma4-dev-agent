#!/usr/bin/env python3
"""Render call-level records (train/schema.md) with the competition model's chat template → {input_ids, labels}.

    .venv-eval/bin/python train/render.py runs/sft/e4b_w7 --tokenizer <hf snapshot dir or model id> \
        [--check runs/pod_w7/w7_example/rendered_base] [--pack] [--out runs/sft/e4b_w7.rendered.jsonl]

Per call: the prompt is `apply_chat_template(input_messages, tools, add_generation_prompt=True, enable_thinking=True)`
after two normalisations that reproduce how the server saw the request (system content as one text part — the
template appends a space per part; tool-call arguments deserialised). The reply is rendered by appending the
assistant message built from `target` and taking the text after the prompt; it ends with the stop token the model
emitted: <|tool_response> (50) after tool calls, <turn|> (106) after a text-only reply. labels = −100 outside the
reply. Checks (all printed; any failure → exit 1):
  A2  --check DIR: every call's prompt text and token count equal the server-rendered prompt in DIR/<task>.json
      (vLLM /tokenize; usage_prompt is the server's count); with --pack the token-level prefix property of every
      segment is asserted and the packed label total equals the per-call total.
  A3  reply spans: thought / text / tool-call token ranges start and end on control tokens; --sample N calls are
      decoded back and compared with target.reasoning / content / tool_calls.
  A4  mask: every token before the reply is −100 (system, user, nudge, compaction summary, tool responses), every
      reply token incl. the stop token is a label; the label share per call is printed. A call with
      `label_scope: tool_call` (train/edit_drills.py) keeps labels on its tool call + stop token only.
Outputs quote task text → keep them under runs/ (gitignored).
Large sets (B-9 part 2, mini-swe-agent trajectories): `--wire mini` keeps a string system message as it is (mini sends
a plain string; the ADK harness sent parts); `--shard i/n` renders every n-th record from i (run n processes, then
concatenate the outputs); `--pack --split-runs` starts a new packed sequence at each expected prefix break (a text +
tool-call reply in the history, a no-thought reply) instead of one sample per call for the whole segment — the
"segments unpacked …" counts are then counts of such breaks; the output is written record by record and A3 keeps a reservoir of --sample calls, so memory
stays at one record's calls.
"""
import argparse
import copy
import glob
import json
import os
import random
import sys

from transformers import AutoTokenizer

STOP_TOOL, STOP_TURN = "<|tool_response>", "<turn|>"
CTRL = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<turn|>", "<|turn>"}


WIRE = {"mode": "adk"}      # --wire: "adk" (the competition harness) or "mini" (mini-swe-agent through litellm)


def normalise(messages):
    """the wire format of the traj log → what the template expects (and what vLLM rendered)"""
    out = []
    for i, m in enumerate(messages):
        m = copy.deepcopy(m)
        if WIRE["mode"] == "adk" and i == 0 and m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
        out.append(m)
    return out


def assistant_from_target(t):
    msg = {"role": "assistant", "content": t.get("content") or ""}
    if t.get("reasoning"):
        msg["reasoning_content"] = t["reasoning"]
    if t.get("tool_calls"):
        msg["tool_calls"] = [{"id": tc.get("id"), "type": "function",
                              "function": {"name": tc["name"], "arguments": tc.get("args") or "{}"}}
                             for tc in t["tool_calls"]]
    return msg


class Renderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def prompt(self, messages, tools):
        return self.tok.apply_chat_template(normalise(messages), tools=tools, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)

    def reply_synthetic(self, target, open_in_prompt):
        """the reply rendered from `target` alone (a two-message conversation through the template, so thinking
        passes the template's gate), incl. the stop token. Used for the last call of a segment; cross-checked
        against reply_delta wherever the next call's prompt exists."""
        conv = [{"role": "user", "content": "x"}, assistant_from_target(target)]
        if target.get("tool_calls"):
            txt = self.tok.apply_chat_template(normalise(conv), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)          # ends with <|tool_response>
        else:
            txt = self.tok.apply_chat_template(normalise(conv), tokenize=False, add_generation_prompt=False,
                                               enable_thinking=True).rstrip("\n")
        txt = txt.split("<|turn>model\n", 1)[1]
        if open_in_prompt:
            if txt.startswith("<|channel>thought\n"):
                txt = txt[len("<|channel>thought\n"):]
            else:                       # no reasoning: the model closed the channel it was given
                txt = "<channel|>" + txt
        return txt

    @staticmethod
    def reply_delta(prompt, next_prompt, target):
        """the reply as the server re-rendered it: the next prompt minus this prompt, cut at the stop token"""
        if not next_prompt.startswith(prompt):
            return None
        rest = next_prompt[len(prompt):]
        stop = STOP_TOOL if target.get("tool_calls") else STOP_TURN
        k = rest.find(stop)
        return None if k < 0 else rest[:k + len(stop)]

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def render_call(self, rec, c, next_prompt=None):
        p = self.prompt(c["input_messages"], rec["tools"])
        p_ids = self.ids(p)
        if not c.get("target"):
            return {"prompt": p, "prompt_ids": p_ids, "reply": None}
        open_in_prompt = p.endswith("<|channel>thought\n")
        t = c["target"]
        if t.get("tool_calls") and (t.get("content") or "").strip():
            # text AND tool calls in one reply: the model wrote the text before the tool call (that is how the
            # server's parser split it), while the chat template re-renders the text after the tool response.
            # Build the generation order explicitly; the delta (cut at <|tool_response>) would lose the text.
            tc_only = self.reply_synthetic(dict(t, content=""), open_in_prompt)
            head, sep, tail = tc_only.partition("<channel|>") if "<channel|>" in tc_only else ("", "", tc_only)
            reply, source, delta, synth = head + sep + t["content"].strip() + tail, "mixed", None, None
        else:
            synth = self.reply_synthetic(t, open_in_prompt)
            delta = self.reply_delta(p, next_prompt, t) if next_prompt else None
            reply, source = (delta, "delta") if delta is not None else (synth, "synthetic")
        if not reply.endswith((STOP_TOOL, STOP_TURN)):
            raise ValueError(f"{rec['instance_id']} call {c['call']}: reply has no stop token: {reply[-60:]!r}")
        full = p + reply
        f_ids = self.ids(full)
        if f_ids[:len(p_ids)] != p_ids:
            raise ValueError(f"{rec['instance_id']} call {c['call']}: token boundary moved at the prompt end")
        r_ids = f_ids[len(p_ids):]
        return {"prompt": p, "prompt_ids": p_ids, "reply": reply, "reply_source": source,
                "delta_equals_synthetic": (delta == synth) if delta is not None else None,
                "reply_ids": r_ids, "input_ids": f_ids, "labels": [-100] * len(p_ids) + r_ids}


def strip_thinking(text):
    """the chat template's strip_thinking macro: drop '<|channel>…' fragments, keep the rest, trim"""
    out = ""
    for part in text.split("<channel|>"):
        out += part.split("<|channel>")[0] if "<|channel>" in part else part
    return out.strip()


def spans(tok, reply_ids, thought_open_in_prompt):
    """thought / text / tool_call / stop ranges over the reply tokens (A3).
    When the prompt already ends with `<|channel>thought\\n` (the usual case with thinking on), the reply starts
    inside the thought channel and its first span runs to the first `<channel|>` inclusive."""
    toks = tok.convert_ids_to_tokens(reply_ids)
    out, i, n = [], 0, len(toks)
    if thought_open_in_prompt and n:
        j = 0
        while j < n and toks[j] != "<channel|>":
            j += 1
        out.append(("thought", 0, j + 1)); i = j + 1
    while i < n:
        t = toks[i]
        if t == "<|channel>":
            j = i + 1
            while j < n and toks[j] != "<channel|>":
                j += 1
            out.append(("thought", i, j + 1)); i = j + 1
        elif t == "<|tool_call>":
            j = i + 1
            while j < n and toks[j] != "<tool_call|>":
                j += 1
            if j == n:                       # the model emitted a tool call it never closed (parsed as text)
                raise ValueError(f"unclosed <|tool_call> at reply position {i}")
            out.append(("tool_call", i, j + 1)); i = j + 1
        elif t in (STOP_TOOL, STOP_TURN):
            out.append(("stop", i, i + 1)); i += 1
        else:
            j = i
            while j < n and toks[j] not in CTRL:
                j += 1
            if j == i:
                raise ValueError(f"unexpected control token {t!r} at reply position {i}")
            out.append(("text", i, j)); i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="dir of call-level JSON records")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--check", default=None, help="dir of server-rendered prompts <task>.json (A2)")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--sample", type=int, default=50, help="calls decoded back for A3")
    ap.add_argument("--out", default=None, help="rendered jsonl (one line per call, or per segment with --pack)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wire", choices=["adk", "mini"], default="adk", help="how the client sent the system message")
    ap.add_argument("--split-runs", action="store_true",
                    help="with --pack: where the prefix property breaks (an expected break), start a new packed sequence "
                         "instead of unpacking the whole segment into one sample per call (exact either way)")
    ap.add_argument("--shard", default="0/1", help="i/n: render records i, i+n, i+2n, … (sorted by file name)")
    a = ap.parse_args()
    WIRE["mode"] = a.wire
    sh_i, sh_n = (int(x) for x in a.shard.split("/"))
    print(f"render · wire {a.wire} · shard {sh_i}/{sh_n} · pack {a.pack} · split runs {a.split_runs} · check {a.check or 'none'} · sample {a.sample}")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    R = Renderer(tok)
    fails, n_calls, n_targets, label_tot, packed_tot = [], 0, 0, 0, 0
    shares, a3_pool, mism, malformed = [], [], [], []
    n_rendered, n_a3_seen = 0, 0
    random.seed(a.seed)
    w_out = open(a.out, "w") if a.out else None

    def emit(row):
        nonlocal n_rendered
        n_rendered += 1
        if w_out:
            w_out.write(json.dumps(row) + "\n")
    n_src = {"delta": 0, "synthetic": 0, "mixed": 0, "delta!=synthetic": 0, "pack_fallback_mixed": 0, "pack_fallback_nothought": 0,
             "content_control_tokens": 0, "scope_tool_call": 0}
    ref_checked = ref_ok = 0
    for fi, f in enumerate(sorted(glob.glob(os.path.join(a.records, "*.json")))):
        if fi % sh_n != sh_i:
            continue
        rec = json.load(open(f))
        if not isinstance(rec, dict) or not isinstance(rec.get("calls"), list):   # a summary / verdict json beside the records
            continue
        ref = None
        if a.check:
            rp = os.path.join(a.check, f"{rec['instance_id']}.json")
            ref = {c["call"]: c for c in json.load(open(rp))["calls"]} if os.path.exists(rp) else None
        rows = []
        calls = rec["calls"]
        for idx, c in enumerate(calls):
            n_calls += 1
            nxt = calls[idx + 1] if idx + 1 < len(calls) and calls[idx + 1]["segment"] == c["segment"] else None
            try:
                next_prompt = R.prompt(nxt["input_messages"], rec["tools"]) if nxt else None
                r = R.render_call(rec, c, next_prompt)
            except Exception as e:
                fails.append(f"{rec['instance_id']} call {c['call']}: {type(e).__name__}: {e}"); continue
            if r.get("reply_source"):
                n_src[r["reply_source"]] += 1
                if r["delta_equals_synthetic"] is False:
                    n_src["delta!=synthetic"] += 1
                    mism.append((rec["instance_id"], c["call"]))
            if ref and c["call"] in ref:
                rc = ref[c["call"]]; ref_checked += 1
                same_text, same_n = r["prompt"] == rc["prompt"], len(r["prompt_ids"]) == rc["rendered_tokens"]
                if same_text and same_n:
                    ref_ok += 1
                else:
                    k = 0
                    while k < min(len(r["prompt"]), len(rc["prompt"])) and r["prompt"][k] == rc["prompt"][k]:
                        k += 1
                    fails.append(f"A2 {rec['instance_id']} call {c['call']}: text_equal={same_text} tokens "
                                 f"{len(r['prompt_ids'])} vs server {rc['rendered_tokens']} (usage {rc['usage_prompt']}); "
                                 f"first diff at char {k}: ours={r['prompt'][k:k+60]!r} server={rc['prompt'][k:k+60]!r}")
            if r["reply"] is None:
                continue
            n_targets += 1
            label_tot += len(r["reply_ids"])
            shares.append(len(r["reply_ids"]) / len(r["input_ids"]))
            # A4: mask structure
            assert all(x == -100 for x in r["labels"][:len(r["prompt_ids"])]) and \
                all(x != -100 for x in r["labels"][len(r["prompt_ids"]):]), "mask layout"
            # A3: spans
            open_in_prompt = r["prompt"].endswith("<|channel>thought\n")
            try:
                sp = spans(tok, r["reply_ids"], open_in_prompt)
            except ValueError as e:          # malformed reply (data quality, not a renderer fault): no target
                malformed.append(f"{rec['instance_id']} call {c['call']}: {e}")
                n_targets -= 1; label_tot -= len(r["reply_ids"]); shares.pop()
                continue
            if not c["target"].get("tool_calls") and any(k == "tool_call" for k, _, _ in sp):
                # a tool call the server's parser did not accept (it came back as text): not a training target
                malformed.append(f"{rec['instance_id']} call {c['call']}: tool-call text the server did not parse: "
                                 f"{(c['target'].get('content') or '')[:80]!r}")
                n_targets -= 1; label_tot -= len(r["reply_ids"]); shares.pop()
                continue
            toks = tok.convert_ids_to_tokens(r["reply_ids"])
            for kind, s, e in sp:
                if kind == "thought" and not ((toks[s] == "<|channel>" or (s == 0 and open_in_prompt))
                                              and toks[e - 1] == "<channel|>"):
                    fails.append(f"A3 {rec['instance_id']} call {c['call']}: thought span not closed")
                if kind == "tool_call" and not (toks[s] == "<|tool_call>" and toks[e - 1] == "<tool_call|>"):
                    fails.append(f"A3 {rec['instance_id']} call {c['call']}: tool_call span not closed")
            if sp[-1][0] != "stop":
                fails.append(f"A3 {rec['instance_id']} call {c['call']}: reply does not end with a stop token")
            if c.get("label_scope") == "tool_call":
                # loss on the tool call (+ its stop token) only: thought and text stay context (train/edit_drills.py)
                P = len(r["prompt_ids"])
                for kind, s, e in sp:
                    if kind in ("thought", "text"):
                        r["labels"][P + s:P + e] = [-100] * (e - s)
                kept = sum(1 for x in r["labels"][P:] if x != -100)
                label_tot += kept - len(r["reply_ids"]); n_src["scope_tool_call"] += 1
            n_a3_seen += 1                       # reservoir sample of --sample calls (memory: one record at a time)
            if len(a3_pool) < a.sample:
                a3_pool.append((rec["instance_id"], c, r, sp))
            else:
                j = random.randrange(n_a3_seen)
                if j < a.sample:
                    a3_pool[j] = (rec["instance_id"], c, r, sp)
            rows.append((c, r))
        # --pack: one sequence per segment when the token prefix property holds for every call
        if a.pack and rows:
            by_seg = {}
            for c, r in rows:
                by_seg.setdefault(c["segment"], []).append((c, r))
            for seg, items in by_seg.items():
                ok, runs, cur = True, [], [items[0]]
                for (c0, r0), (c1, r1) in zip(items, items[1:]):
                    if r1["prompt_ids"][:len(r0["input_ids"])] == r0["input_ids"]:
                        cur.append((c1, r1))
                    else:
                        ok = False
                        mixed_hist = any(m.get("role") == "assistant" and m.get("tool_calls") and (m.get("content") or "").strip()
                                         for m in c0["input_messages"])
                        if r0["reply_source"] == "mixed" or mixed_hist:
                            # expected: the template re-orders text+tool replies; when such a reply sits in the history,
                            # the generation prompt closes it with "<turn|>\n" before opening the thought channel while
                            # the next prompt renders the following model turn without that boundary (pod_a5 rich_3454)
                            n_src["pack_fallback_mixed"] += 1
                        elif not c0["target"].get("reasoning") and r0["prompt"].endswith("<|channel>thought\n"):
                            # expected: a reply without thought after a tool response closes the channel the
                            # prompt opened (<channel|>…), while the template re-renders that turn with no channel
                            n_src["pack_fallback_nothought"] += 1
                        else:
                            fails.append(f"PACK {rec['instance_id']} segment {seg}: call {c1['call']} does not "
                                         f"extend call {c0['call']} (prefix property broken)")
                        if not a.split_runs:
                            break
                        runs.append(cur); cur = [(c1, r1)]      # --split-runs: a new sequence starts at the break
                runs.append(cur)
                if ok or a.split_runs:
                    for run in runs:                            # each run: the last call's tokens, every reply labelled
                        c_last, r_last = run[-1]
                        ids = r_last["input_ids"]; labels = [-100] * len(ids)
                        for c, r in run:
                            s = len(r["prompt_ids"]); e = len(r["input_ids"])
                            labels[s:e] = r["labels"][s:e]      # = ids[s:e] unless label_scope masked a span
                        packed_tot += sum(1 for x in labels if x != -100)
                        emit({"instance_id": rec["instance_id"], "segment": seg,
                              "calls": [c["call"] for c, _ in run], "input_ids": ids, "labels": labels})
                else:
                    for c, r in items:
                        packed_tot += sum(1 for x in r["labels"][len(r["prompt_ids"]):] if x != -100)
                        emit({"instance_id": rec["instance_id"], "segment": seg, "calls": [c["call"]],
                              "input_ids": r["input_ids"], "labels": r["labels"]})
        elif rows:
            for c, r in rows:
                emit({"instance_id": rec["instance_id"], "segment": c["segment"], "calls": [c["call"]],
                      "input_ids": r["input_ids"], "labels": r["labels"]})
    # A3 sample: decode spans back and compare with the target
    for task, c, r, sp in a3_pool:
        t = c["target"]
        got_thought = "".join(tok.decode(r["reply_ids"][(s if s == 0 and r["prompt"].endswith("<|channel>thought\n") else s + 1):e - 1])
                              for k, s, e in sp if k == "thought")
        got_thought = got_thought.removeprefix("thought\n") if not r["prompt"].endswith("<|channel>thought\n") else got_thought
        got_text = "".join(tok.decode(r["reply_ids"][s:e]) for k, s, e in sp if k == "text").strip()
        got_calls = ["".join(tok.decode(r["reply_ids"][s:e])) for k, s, e in sp if k == "tool_call"]
        if (t.get("reasoning") or "").strip() != got_thought.strip():
            fails.append(f"A3 {task} call {c['call']}: decoded thought differs from target.reasoning")
        want_text = (t.get("content") or "")
        if "<|channel>" in want_text or "<channel|>" in want_text:   # stray channel tags in the text: the template's
            n_src["content_control_tokens"] += 1                        # strip_thinking drops them on re-send
            want_text = strip_thinking(want_text)
        if want_text.strip() != got_text:
            fails.append(f"A3 {task} call {c['call']}: decoded text differs from target.content")
        for tc, g in zip(t.get("tool_calls") or [], got_calls):
            if f"call:{tc['name']}{{" not in g:
                fails.append(f"A3 {task} call {c['call']}: decoded tool call lacks {tc['name']}")
        if len(got_calls) != len(t.get("tool_calls") or []):
            fails.append(f"A3 {task} call {c['call']}: {len(got_calls)} tool-call spans vs {len(t.get('tool_calls') or [])}")
    if w_out:
        w_out.close()
    print(f"calls {n_calls} · targets {n_targets} · label tokens {label_tot}"
          + (f" · packed label tokens {packed_tot} (sequences {n_rendered}, segments unpacked because of a mixed reply {n_src['pack_fallback_mixed']} / a no-thought reply after a tool response {n_src['pack_fallback_nothought']})" if a.pack else "")
          + (f" · A2 server-rendered prompts equal {ref_ok}/{ref_checked}" if a.check else "")
          + f" · reply from delta {n_src['delta']} / synthetic {n_src['synthetic']} / mixed text+tool {n_src['mixed']} / delta≠synthetic {n_src['delta!=synthetic']}"
          + (f" e.g. {mism[:5]}" if mism else "")
          + f" · text replies with stray channel tags (stripped like the template) {n_src['content_control_tokens']}"
          + f" · label_scope tool_call (thought/text masked) {n_src['scope_tool_call']}"
          + f" · A3 sampled {len(a3_pool)}"
          + (f" · A4 label share per call min {min(shares):.3f} median {sorted(shares)[len(shares)//2]:.3f} max {max(shares):.3f}" if shares else ""))
    if a.pack and packed_tot != label_tot:
        fails.append(f"PACK label total {packed_tot} != per-call total {label_tot}")
    for x in malformed:
        print("MALFORMED (excluded from targets)", x)
    for x in fails:
        print("FAIL", x)
    print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
