#!/usr/bin/env python3
"""Write the missing thought of a call: given the call's real context and the action the agent took, another model
writes the reasoning that leads to exactly that action (plan v2.1 decision 4; MindForge's reasoning rewrite).

    # pilot on the pod with the 31B served by scripts/serve_vllm.sh (MODEL=google/gemma-4-31B-it-qat-w4a16-ct):
    python3 train/relabel_thoughts.py runs/sft/d3_train --model gemma31b --tasks 10 --sample 30 \
        --out runs/sft/d3_relabel_31b_pilot
    python3 train/relabel_thoughts.py runs/sft/d3_train --model gemma31b --sample 30 --dry-run   # prints 2 requests
    # GLM (OpenAI-compatible endpoint; key in $GLM_API_KEY, never in the repo):
    python3 train/relabel_thoughts.py runs/sft/d3_train --model glm --sample 30 --out runs/sft/d3_relabel_glm_pilot

Request = the call's `input_messages` verbatim (same system prompt, task, earlier calls and tool results the agent saw)
+ one user instruction carrying the action (tool calls as `name(key="value")` lines, or the text reply). Thinking is
OFF for the writer and its *content* becomes the thought: with thinking on, its reasoning would be about how to write,
not the agent's own deliberation. Checks per thought: no control tokens / tool-call markup, no leak phrases ("the
given action", "as instructed" …), ≤ --max-thought-tokens (chars/4 when no tokenizer). Output: copies of the records
with `target.reasoning` filled and `thought_source = synthetic_<model>` on the relabelled calls only (the others keep
`none`), plus per-call `relabel: {model, prompt_tokens, completion_tokens, seconds}`, `relabel_summary.json`, and a
printed table. Sampling: --tasks tasks (seeded), per task the first call, the last call and random middle calls until
--sample. Every number printed; the pilot's per-call cost × 498 calls is the Phase 1 budget line.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request

CTRL = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|turn>",
        "<turn|>", '<|"|>', "<start_of_turn>", "<end_of_turn>"]
LEAK = re.compile(r"\b(the )?(given|provided|specified|predetermined|chosen|instructed|required) (next )?(action|step|call|command)"
                  r"|as (told|asked|instructed|requested)|next action:|you (told|asked) me|the user (told|asked|wants) me to (call|run|use)",
                  re.I)
INSTRUCTION = (
    "You are the software engineer agent in this conversation. Your next action has already been decided and is shown "
    "below. Write the private reasoning that leads you to exactly this action: what the latest results tell you, what "
    "is still uncertain, and why this is the right next step now. First person, as thoughts, 2 to 8 sentences, under "
    "{words} words. Do not mention that the action was given to you, do not quote tool outputs verbatim, do not use "
    "any tool-call syntax or markup, and do not write the action itself. Output only the reasoning text.\n\n"
    "NEXT ACTION:\n{action}")


def render_action(target):
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
            if len(s) > 400:
                s = s[:400] + "…" + s[-1]
            parts.append(f"{k}={s}")
        lines.append(f"{tc['name']}({', '.join(parts)})")
    if (target.get("content") or "").strip():
        lines.append("Reply to the user with: " + target["content"].strip()[:1500])
    return "\n".join(lines) or "(empty reply)"


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
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=1024, help="writer's max_tokens")
    ap.add_argument("--max-thought-tokens", type=int, default=3000)
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
    print(f"relabel_thoughts · {a.records}: {len(records)} records → {len(recs)} tasks, {len(picks)} calls · writer {a.model} "
          f"· thought_source {src} · max {a.max_words} words / {a.max_thought_tokens} tokens · seed {a.seed}")
    print("  picked: " + (" ".join(f"{r['instance_id']}:c{c['call']}" for r, c in picks) if len(picks) <= 40 else f"all {len(picks)} calls of {len(recs)} records"))
    if a.dry_run:
        for r, c in picks[:2]:
            print(f"\n--- {r['instance_id']} call {c['call']} · context {len(c['input_messages'])} messages · last role {c['input_messages'][-1]['role']}")
            print(INSTRUCTION.format(words=a.max_words, action=render_action(c["target"])))
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
        msgs = c["input_messages"] + [{"role": "user", "content": INSTRUCTION.format(words=a.max_words, action=render_action(c["target"]))}]
        attempts = 0
        while True:
            attempts += 1
            try:
                thought, u = w.write(msgs, r["tools"])
            except Exception as e:  # noqa: BLE001
                thought, u = "", {"error": f"{type(e).__name__}: {str(e)[:120]}", "seconds": None, "prompt_tokens": None, "completion_tokens": None}
            problems, n_tok = check(thought, a.max_thought_tokens, tok)
            if not problems or attempts > a.retries:
                break
        u["attempts"] = attempts
        n_retried += attempts - 1
        ok = not problems
        bad += not ok
        rows.append({"instance_id": r["instance_id"], "call": c["call"], "ok": ok, "problems": problems, "thought_tokens": n_tok,
                     **u, "thought": thought})
        print(f"  {i:>2} {r['instance_id']:<16} c{c['call']:<3} {'ok ' if ok else 'BAD'} {n_tok:>4} tok · prompt {u.get('prompt_tokens')} · "
              f"out {u.get('completion_tokens')} · {u.get('seconds')} s{(' · ' + '; '.join(problems)) if problems else ''}")
        print(f"       {thought[:160]!r}")
        rec = by_file.setdefault(r["_file"], json.loads(json.dumps({k: v for k, v in r.items() if k != '_file'})))
        tgt = next(x for x in rec["calls"] if x["call"] == c["call"])
        if ok:
            tgt["target"]["reasoning"] = thought
            tgt["thought_source"] = src
        tgt["relabel"] = {"model": w.served, **u, "ok": ok, "problems": problems}
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
    summ = {"writer": a.model, "served": w.served, "calls": len(rows), "ok": len(done), "bad": bad,
            "prompt_tokens_total": sum(pt), "completion_tokens_total": sum(ct), "seconds_total": round(sum(sec), 1),
            "thought_tokens": {"p50": tt[len(tt) // 2] if tt else None, "max": tt[-1] if tt else None},
            "per_call": {"prompt_tokens": round(sum(pt) / max(len(pt), 1)), "completion_tokens": round(sum(ct) / max(len(ct), 1)),
                         "seconds": round(sum(sec) / max(len(sec), 1), 1)},
            "rows": rows}
    json.dump(summ, open(a.out.rstrip("/") + ".summary.json", "w"), ensure_ascii=False, indent=1)   # beside the dir: the dir holds records only
    summ["retries"] = n_retried
    print(f"relabel: {len(rows)} calls · ok {len(done)} · bad {bad} · retries used {n_retried} · thought tokens p50 {summ['thought_tokens']['p50']} max {summ['thought_tokens']['max']} · "
          f"per call prompt {summ['per_call']['prompt_tokens']} + out {summ['per_call']['completion_tokens']} tok, {summ['per_call']['seconds']} s · "
          f"total {summ['seconds_total']} s → {a.out} ({len(by_file)} records)")
    print(f"  Phase 1 estimate at this rate: 498 calls ≈ {498 * summ['per_call']['seconds'] / 3600:.1f} h of writer time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
