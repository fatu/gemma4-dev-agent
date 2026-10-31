#!/usr/bin/env python3
"""Write the missing thought of a call: given the call's real context and the action the agent took, another model
writes the reasoning that leads to exactly that action (plan v2.1 decision 4; MindForge's reasoning rewrite).

    # pilot on the pod with the 31B served by scripts/serve_vllm.sh (MODEL=google/gemma-4-31B-it-qat-w4a16-ct):
    python3 train/relabel_thoughts.py runs/sft/d3_train --model gemma31b --tasks 10 --sample 30 \
        --out runs/sft/d3_relabel_31b_pilot
    python3 train/relabel_thoughts.py runs/sft/d3_train --model gemma31b --sample 30 --dry-run   # prints 2 requests
    # GLM (OpenAI-compatible endpoint; key in $GLM_API_KEY, never in the repo):
    python3 train/relabel_thoughts.py runs/sft/d3_train --model glm --sample 30 --out runs/sft/d3_relabel_glm_pilot

Request, by style: `name` (default) = the instruction ALONE — {prev} carries the whole history as text (system prompt,
task, earlier calls with full arguments, tool results, nudges; most recent last; cut from the front only over
--prev-chars) and {action} the action; no input_messages, no tool schemas. `bare` (M2) = the call's `input_messages`
verbatim + tool schemas + the instruction as a last user message. The action is rendered as `name(key="value")` lines
(plus the agent's visible text, or the text reply). Thinking is
OFF for the writer and its *content* becomes the thought: with thinking on, its reasoning would be about how to write,
not the agent's own deliberation. Checks per thought: no control tokens / tool-call markup, no leak phrases ("the
given action", "as instructed" …), ≤ --max-thought-tokens (chars/4 when no tokenizer). Output: copies of the records
with `target.reasoning` filled and `thought_source = synthetic_<model>` on the relabelled calls only (the others keep
`none`), plus per-call `relabel: {model, style, instruction_sha, prompt_tokens, completion_tokens, seconds, attempts}`,
`<out>.summary.json` (style, instruction sha + full text, retries, prev cuts, per-call rows), and a printed table. Sampling: --tasks tasks (seeded), per task the first call, the last call and random middle calls until
--sample. Every number printed; the pilot's per-call cost × 498 calls is the Phase 1 budget line.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

CTRL = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|turn>",
        "<turn|>", '<|"|>', "<start_of_turn>", "<end_of_turn>", "<think>", "</think>", "<|think|>"]
LEAK = re.compile(r"\b(the )?(given|provided|specified|predetermined|chosen|instructed|required) (next )?(action|step|call|command)"
                  r"|as (told|asked|instructed|requested)|next action:|you (told|asked) me|the user (told|asked|wants) me to (call|run|use)",
                  re.I)
# the `name` instruction speaks of "a coding agent" and "a trajectory": a thought that slips into third person or into the
# writer's own task ("The agent needs to…", "reconstructing the reasoning", "this trajectory") is not the agent's thought
META = re.compile(r"\b(the|this|a) (coding )?agent\b|\breconstruct(s|ed|ing)? (the |its |my |this )?(internal )?"
                  r"(reasoning|thinking|thoughts?)\b|\b(the|this|a) trajectory\b", re.I)
INSTRUCTIONS = {
    # "bare" = M2's writer (2026-10-29). Its "do not write the action itself" made thoughts that name only 22% of the
    # action's identifiers (E4B's native thoughts: 47%), decoupling thought from action — one measured source of the
    # copy-the-previous-call loop (notes/14 § 训练数据与 loss mask 复查). Kept for reproducing M2.
    "bare": (
        "You are the software engineer agent in this conversation. Your next action has already been decided and is shown "
        "below. Write the private reasoning that leads you to exactly this action: what the latest results tell you, what "
        "is still uncertain, and why this is the right next step now. First person, as thoughts, 2 to 8 sentences, under "
        "{words} words. Do not mention that the action was given to you, do not quote tool outputs verbatim, do not use "
        "any tool-call syntax or markup, and do not write the action itself. Output only the reasoning text.\n\n"
        "NEXT ACTION:\n{action}"),
    # "name" (default since 2026-10-30, rewritten by the user the same day): reconstruct the reasoning right before the
    # turn — {prev} = the last step and its result, {action} = what the agent then said and did; rules: no hindsight,
    # land on the actual action, do not restate the visible output, no fabrication, no meta, first-person voice, length
    # matched to the step. No word limit: the writer's --max-tokens is the only cap (a cut thought fails the checks).
    "name": (
        "You are reconstructing the internal reasoning of a coding agent working in a real repository. You will be shown one "
        "turn from a trajectory that already happened. Your job is to write what the agent was thinking immediately BEFORE "
        "it produced its visible output on that turn.\n"
        "## Context available to the agent at that moment\n"
        "<history>\n{prev}\n</history>\n"
        "## What the agent then actually said and did\n"
        "{action}\n"
        "## Rules\n"
        "1. NO HINDSIGHT. Write only from what is in the context above. You may not reference any file content, command result, "
        "error message, or conclusion that the agent had not yet observed at this point. If the outcome of the action is "
        "unknown at this moment, your reasoning must treat it as unknown.\n"
        "2. IT MUST LAND ON THE ACTUAL ACTION. The reasoning has to make the observed output the natural next step. Do not argue "
        "for a different approach, and do not critique the action even if you think it is suboptimal -- you are recovering the "
        "reasoning that led here, not improving it.\n"
        "3. DO NOT RESTATE THE OUTPUT. The thinking precedes the decision. If the agent then says \"I'll update the retry "
        "handler\", your thinking works out *why* that is the next move; it does not announce it again.\n"
        "4. NO FABRICATION. Do not invent file paths, class names, config keys, ticket numbers, or test results that do not appear "
        "in the context. If you need to refer to something unseen, refer to it the way the agent would -- as something still to be "
        "checked.\n"
        "5. NO META. Never mention training, examples, datasets, \"the user wants\", or that you are reconstructing anything. Write "
        "from inside the task.\n"
        "6. VOICE. First person, present tense, working-it-out register. Flowing prose, not headings or bullet lists. Uncertainty "
        "is expected and should be stated plainly (\"I'm not sure yet whether...\", \"that could also be...\"). Contract the reasoning "
        "where the step is obvious -- a routine file read does not need a paragraph.\n"
        "7. LENGTH. Shorter is fine and usually better; pad nothing. Match effort to the difficulty of the "
        "step.\n"
        "## OUTPUT\n"
        "Emit the reasoning text only. No <think> tags, no preamble, no trailing commentary, no quotes around it.\n"

        ),
}
# {prev} budget: the 31B serves 32,768 tokens; minus the writer's 2,048 and the rest of the instruction leaves ~29K tokens,
# and the history renders at 3.3-3.7 chars/token (D3, Gemma tokenizer) → 90K chars ≈ 27K tokens. D3's longest full
# history is 76K chars (fastapi_14371 c25, 23K tokens), so no D3 call is cut; the front cut is a guard for longer runs.
PREV_CHARS = 90_000
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{3,}")


def action_idents(target):
    """identifiers (≥ 4 chars: paths, symbols, flags, test names) in the action's arguments — not the tool names"""
    out = set()
    for tc in target.get("tool_calls") or []:
        args = tc.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        out |= set(IDENT.findall(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
    return out


def overlap(target, thought):
    """share of the action's identifiers that the thought names (E4B native thoughts 0.47; M2's bare writer 0.22)"""
    a = action_idents(target)
    return round(len(a & set(IDENT.findall(thought or ""))) / len(a), 2) if a else None


def render_action(target, neutral=False, clip=True):
    """the action as `name(key="value")` lines plus the visible text. neutral=True (the `name` style, which speaks of
    "the agent") labels the text without a pronoun: "Said alongside the call:" / "Replied to the user:"; the default keeps
    M2's second-person labels for `bare`. clip=True keeps M2's limits (each argument 400 chars, the text 1,500); the
    `name` style passes clip=False: its rules make the thought land on the actual action, so the writer sees all of it
    (D3: 99 of 498 actions exceed 400 chars, the longest 6,522)."""
    lines = []
    for tc in target.get("tool_calls") or []:
        args = tc.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        parts = []
        for k, v in (args or {}).items():
            s = json.dumps(v, ensure_ascii=False)
            if clip and len(s) > 400:
                s = s[:400] + "…" + s[-1]
            parts.append(f"{k}={s}")
        lines.append(f"{tc['name']}({', '.join(parts)})")
    if (target.get("content") or "").strip():
        # visible text next to a tool call is the agent's own note in the same turn (D3: 42 of 498 targets, e.g.
        # "`copy` is now unused; tidy import."), not a reply; only a text-only turn is a reply to the user (D3: 68)
        if neutral:
            label = "Said alongside the call: " if lines else "Replied to the user: "
        else:
            label = "Alongside the call you also write: " if lines else "Reply to the user with: "
        text = target["content"].strip()
        lines.append(label + (text[:1500] if clip else text))
    return "\n".join(lines) or "(empty reply)"


def _text(s):
    if isinstance(s, list):
        s = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in s)
    return (s or "").strip()


def _call_line(name, args):
    """a tool call as `name(key=value, …)` with the FULL arguments (render_action clips them; history never is)"""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"{name}({args})"
    return f"{name}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (args or {}).items()) + ")"


def render_prev(c, max_chars):
    """{prev} for the standalone styles (the request is the instruction alone, no input_messages): everything the agent
    had seen before this call, as text, oldest first and most recent last — its system prompt, the task, every earlier
    agent turn (thought if one is recorded, visible text, tool calls with full arguments), every tool result and every
    later user/harness message (nudge, compaction summary). Nothing is clipped inside a message. Over max_chars, the
    text is cut from the FRONT only (the oldest blocks go first; the cut snaps to the next block start), never from the
    end, so the latest steps always survive intact. Returns (text, chars_dropped)."""
    names, blocks, seen_user = {}, [], False
    for i, m in enumerate(c["input_messages"], 1):
        role = m.get("role")
        if role == "system":
            blocks.append(f"### [{i}] System prompt\n{_text(m.get('content'))}")
        elif role == "user":
            blocks.append(f"### [{i}] {'User message' if seen_user else 'Task'}\n{_text(m.get('content'))}")
            seen_user = True
        elif role == "assistant":
            lines = []
            th = _text(m.get("reasoning_content") or m.get("reasoning"))
            if th:
                lines.append("Thought: " + th)
            tx = _text(m.get("content"))
            if tx:
                lines.append("Said: " + tx)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name")
                names[tc.get("id")] = name
                lines.append("Called: " + _call_line(name, fn.get("arguments", tc.get("args"))))
            blocks.append(f"### [{i}] Agent\n" + ("\n".join(lines) or "(empty turn)"))
        elif role == "tool":
            blocks.append(f"### [{i}] Tool result ({names.get(m.get('tool_call_id')) or m.get('name') or 'tool'})\n{_text(m.get('content'))}")
        else:
            blocks.append(f"### [{i}] {role}\n{_text(m.get('content'))}")
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text, 0
    keep = text[-max(max_chars - 80, 0):]
    j = keep.find("\n\n### [")
    if j != -1:
        keep = keep[j + 2:]
    dropped = len(text) - len(keep)
    return f"[… {dropped} earlier characters omitted …]\n\n" + keep, dropped


def propagate(rec):
    """copy every relabelled call's thought into the later calls' history: the harness re-sends an assistant turn with
    its `reasoning_content`, so the thought must appear wherever that turn is part of a later call's input (else the
    next prompt's delta has no thought and the segment's prefix property breaks). Matched by tool-call ids, else by
    content. Returns the number of history messages updated."""
    n = 0
    for c in rec["calls"]:
        t = c.get("target")
        if not t or not t.get("reasoning") or c.get("thought_source") in (None, "none"):
            continue
        ids = {tc.get("id") for tc in t.get("tool_calls") or [] if tc.get("id")}
        for later in rec["calls"]:
            if later["call"] <= c["call"]:
                continue
            for m in later["input_messages"]:
                if m.get("role") != "assistant":
                    continue
                m_ids = {tc.get("id") for tc in m.get("tool_calls") or [] if tc.get("id")}
                same = (ids and m_ids == ids) or (not ids and not m_ids and (m.get("content") or "").strip() == (t.get("content") or "").strip())
                if same and m.get("reasoning_content") != t["reasoning"]:
                    m["reasoning_content"] = t["reasoning"]; n += 1
    return n


def pick(records, n_tasks, n_sample, seed):
    rnd = random.Random(seed)
    recs = [r for r in records if any(c.get("target") for c in r["calls"])]
    rnd.shuffle(recs)
    recs = recs[:n_tasks]
    picks = []
    for r in recs:                                  # first and last call of every picked task
        calls = [c for c in r["calls"] if c.get("target")]
        picks.append((r, calls[0]))
        if len(calls) > 1:
            picks.append((r, calls[-1]))
    middle = [(r, c) for r in recs for c in [c for c in r["calls"] if c.get("target")][1:-1]]
    rnd.shuffle(middle)
    picks += middle[: max(0, n_sample - len(picks))]
    return recs, picks[:n_sample]


class Writer:
    def __init__(self, model, url, served, max_tokens, api_key=None):
        self.kind, self.url, self.max_tokens = model, url.rstrip("/"), max_tokens
        self.key = api_key
        if model == "gemma31b":
            ids = [m["id"] for m in json.load(urllib.request.urlopen(self.url + "/v1/models"))["data"]]
            self.served = served or next((m for m in ids if m.startswith("gemma-4-")), ids[0])
        else:
            self.served = served or "glm-5.2"

    def post(self, body):
        hdr = {"Content-Type": "application/json"}
        if self.key:
            hdr["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(), hdr)
        return json.load(urllib.request.urlopen(req, timeout=600))

    def write(self, messages, tools):
        body = {"model": self.served, "messages": messages, "temperature": 0.7, "top_p": 0.95,
                "max_tokens": self.max_tokens}
        if self.kind == "gemma31b":
            if tools:                       # conversation mode only: the history's tool calls render against them
                body["tools"] = tools
                body["tool_choice"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        r = self.post(body)
        ch = r["choices"][0]["message"]
        u = r.get("usage") or {}
        return (ch.get("content") or "").strip(), {"prompt_tokens": u.get("prompt_tokens"),
                                                   "completion_tokens": u.get("completion_tokens"),
                                                   "seconds": round(time.time() - t0, 1),
                                                   "finish_reason": r["choices"][0].get("finish_reason")}


def check(thought, max_tokens, tok=None):
    problems = []
    if not thought:
        problems.append("empty")
    for c in CTRL:
        if c in thought:
            problems.append(f"control token {c}")
    m = LEAK.search(thought)
    if m:
        problems.append(f"leak phrase {m.group(0)!r}")
    m = META.search(thought)
    if m:
        problems.append(f"third-person / meta phrase {m.group(0)!r}")
    if re.search(r"\w+\(\w+=", thought) and "(" in thought and "=" in thought and re.search(r"\b(run_command|read_file|edit_file|write_file|submit_patch|get_status)\(", thought):
        problems.append("contains the action call")
    n = len(tok(thought, add_special_tokens=False)["input_ids"]) if tok else len(thought) // 4
    if n > max_tokens:
        problems.append(f"too long {n} tokens")
    return problems, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--model", choices=["gemma31b", "glm"], required=True)
    ap.add_argument("--url", default=None, help="server (default vLLM 127.0.0.1:8000; GLM: $GLM_API_BASE)")
    ap.add_argument("--served", default=None, help="served model id (default: the first gemma-4-* id; GLM glm-5.2)")
    ap.add_argument("--all", action="store_true", help="every call with a target (ignores --tasks/--sample)")
    ap.add_argument("--retries", type=int, default=0, help="re-ask the writer up to N times when a thought fails the checks")
    ap.add_argument("--style", choices=sorted(INSTRUCTIONS), default="name",
                    help="writer instruction: name = the thought names the next file/function/command (default); bare = M2's writer")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=2048, help="writer's max_tokens")
    ap.add_argument("--max-thought-tokens", type=int, default=3000)
    ap.add_argument("--prev-chars", type=int, default=PREV_CHARS,
                    help="standalone styles: {prev} budget in chars; over it the history is cut from the front only")
    ap.add_argument("--tokenizer", default=None, help="count thought tokens with this tokenizer (else chars/4)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print 2 requests, call nothing")
    ap.add_argument("--propagate-only", action="store_true", help="only copy existing thoughts into the later calls' history (records dir in place)")
    a = ap.parse_args()
    if a.propagate_only:
        tot = 0
        for f in sorted(glob.glob(os.path.join(a.records, "*.json"))):
            rec = json.load(open(f))
            if not isinstance(rec, dict) or not isinstance(rec.get("calls"), list):
                continue
            n = propagate(rec); tot += n
            if n:
                json.dump(rec, open(f, "w"), ensure_ascii=False)
            print(f"  {rec['instance_id']:<16} history messages updated {n}")
        print(f"propagate: {tot} history messages now carry the relabelled thought → {a.records}")
        return 0
    src = f"synthetic_{'31b' if a.model == 'gemma31b' else 'glm'}"
    records = [json.load(open(f)) | {"_file": f} for f in sorted(glob.glob(os.path.join(a.records, "*.json")))]
    if a.all:
        recs = [r for r in records if any(c.get("target") for c in r["calls"])]
        picks = [(r, c) for r in recs for c in r["calls"] if c.get("target")]
    else:
        recs, picks = pick(records, a.tasks, a.sample, a.seed)
    INSTRUCTION = INSTRUCTIONS[a.style]
    instr_sha = hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:12]   # the style name alone does not pin the text
    print(f"relabel_thoughts · {a.records}: {len(records)} records → {len(recs)} tasks, {len(picks)} calls · writer {a.model} "
          f"· style {a.style} (instruction sha {instr_sha}) · thought_source {src} · "
          f"{f'max {a.max_words} words' if '{words}' in INSTRUCTION else 'no word limit in the instruction'} · "
          f"writer max_tokens {a.max_tokens} · thought check ≤ {a.max_thought_tokens} tokens · seed {a.seed}")
    standalone = "{prev}" in INSTRUCTION
    print("  request: " + (f"the instruction alone, no input_messages / tool schemas · {{prev}} = the whole history as text, most recent "
                           f"last, cut from the front over {a.prev_chars} chars" if standalone else
                           "input_messages verbatim + tool schemas + the instruction as a last user message"))

    def build(r, c):
        """(messages, tools, prev_chars, prev_cut) for one call"""
        if standalone:
            action = render_action(c["target"], neutral=True, clip=False)
            prev, cut = render_prev(c, a.prev_chars - len(action))     # the uncut action shares the budget
            body = INSTRUCTION.format(words=a.max_words, action=action, prev=prev.replace("</history>", "</ history>"))
            return [{"role": "user", "content": body}], None, len(prev), cut
        body = INSTRUCTION.format(words=a.max_words, action=render_action(c["target"]))
        return c["input_messages"] + [{"role": "user", "content": body}], r["tools"], None, None
    print("  picked: " + (" ".join(f"{r['instance_id']}:c{c['call']}" for r, c in picks) if len(picks) <= 40 else f"all {len(picks)} calls of {len(recs)} records"))
    if a.dry_run:
        for r, c in picks[:2]:
            msgs, tools, pc, cut = build(r, c)
            print(f"\n--- {r['instance_id']} call {c['call']} · context {len(c['input_messages'])} messages · request {len(msgs)} "
                  f"message(s), {sum(len(m.get('content') or '') for m in msgs)} chars" + (f" · prev {pc} chars, cut {cut}" if standalone else ""))
            print(msgs[-1]["content"])
        return 0
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    url = a.url or (os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4") if a.model == "glm" else "http://127.0.0.1:8000")
    key = os.environ.get("GLM_API_KEY") if a.model == "glm" else None
    if a.model == "glm" and not key:
        print("GLM_API_KEY is not set"); return 1
    w = Writer(a.model, url, a.served, a.max_tokens, key)
    print(f"  server {url} · served model {w.served}")
    os.makedirs(a.out, exist_ok=True)
    by_file = {}
    rows, bad, n_retried = [], 0, 0
    for i, (r, c) in enumerate(picks, 1):
        msgs, tools, pc, cut = build(r, c)
        attempts = 0
        while True:
            attempts += 1
            try:
                thought, u = w.write(msgs, tools)
            except Exception as e:  # noqa: BLE001
                thought, u = "", {"error": f"{type(e).__name__}: {str(e)[:120]}", "seconds": None, "prompt_tokens": None, "completion_tokens": None}
            problems, n_tok = check(thought, a.max_thought_tokens, tok)
            if u.get("finish_reason") == "length":
                problems.append(f"cut at the writer's max_tokens {a.max_tokens}")
            if not problems or attempts > a.retries:
                break
        u["attempts"] = attempts
        n_retried += attempts - 1
        ok = not problems
        bad += not ok
        ov = overlap(c["target"], thought)
        rows.append({"instance_id": r["instance_id"], "call": c["call"], "ok": ok, "problems": problems, "thought_tokens": n_tok,
                     "overlap": ov, "prev_chars": pc, "prev_cut": cut, **u, "thought": thought})
        print(f"  {i:>2} {r['instance_id']:<16} c{c['call']:<3} {'ok ' if ok else 'BAD'} {n_tok:>4} tok · names {ov if ov is not None else '-':<4} · "
              f"prompt {u.get('prompt_tokens')} · out {u.get('completion_tokens')} · {u.get('seconds')} s"
              f"{f' · prev {pc} ch' + (f' (cut {cut})' if cut else '') if standalone else ''}{(' · ' + '; '.join(problems)) if problems else ''}")
        print(f"       {thought[:160]!r}")
        rec = by_file.setdefault(r["_file"], json.loads(json.dumps({k: v for k, v in r.items() if k != '_file'})))
        tgt = next(x for x in rec["calls"] if x["call"] == c["call"])
        if ok:
            tgt["target"]["reasoning"] = thought
            tgt["thought_source"] = src
        tgt["relabel"] = {"model": w.served, "style": a.style, "instruction_sha": instr_sha, **u, "ok": ok, "problems": problems}
    n_prop = 0
    for f, rec in by_file.items():
        n_prop += propagate(rec)
        json.dump(rec, open(os.path.join(a.out, os.path.basename(f)), "w"), ensure_ascii=False)
    print(f"  propagated the thoughts into {n_prop} later history messages (re-sent reasoning_content)")
    done = [x for x in rows if x["ok"]]
    pt = [x["prompt_tokens"] for x in rows if x.get("prompt_tokens")]
    ct = [x["completion_tokens"] for x in rows if x.get("completion_tokens")]
    sec = [x["seconds"] for x in rows if x.get("seconds")]
    tt = sorted(x["thought_tokens"] for x in done)
    ovs = [x["overlap"] for x in done if x.get("overlap") is not None]
    summ = {"writer": a.model, "served": w.served, "style": a.style, "instruction_sha": instr_sha, "instruction": INSTRUCTION,
            "calls": len(rows), "ok": len(done), "bad": bad, "retries": n_retried,
            "request": "standalone" if standalone else "conversation", "prev_chars_budget": a.prev_chars if standalone else None,
            "prev_cut_calls": sum(1 for x in rows if x.get("prev_cut")),
            "overlap_mean": round(sum(ovs) / len(ovs), 3) if ovs else None,
            "prompt_tokens_total": sum(pt), "completion_tokens_total": sum(ct), "seconds_total": round(sum(sec), 1),
            "thought_tokens": {"p50": tt[len(tt) // 2] if tt else None, "max": tt[-1] if tt else None},
            "per_call": {"prompt_tokens": round(sum(pt) / max(len(pt), 1)), "completion_tokens": round(sum(ct) / max(len(ct), 1)),
                         "seconds": round(sum(sec) / max(len(sec), 1), 1)},
            "rows": rows}
    json.dump(summ, open(a.out.rstrip("/") + ".summary.json", "w"), ensure_ascii=False, indent=1)   # beside the dir: the dir holds records only
    print(f"relabel ({a.style}): {len(rows)} calls · ok {len(done)} · bad {bad} · retries used {n_retried} · thought tokens p50 {summ['thought_tokens']['p50']} max {summ['thought_tokens']['max']} · "
          f"action identifiers named by the thought: mean {summ['overlap_mean']} (E4B native 0.47 · M2 bare 0.22) · "
          f"per call prompt {summ['per_call']['prompt_tokens']} + out {summ['per_call']['completion_tokens']} tok, {summ['per_call']['seconds']} s · "
          f"total {summ['seconds_total']} s → {a.out} ({len(by_file)} records)")
    print(f"  Phase 1 estimate at this rate: 498 calls ≈ {498 * summ['per_call']['seconds'] / 3600:.1f} h of writer time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
