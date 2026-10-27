#!/usr/bin/env python3
"""Render one task's trajectory as Markdown: every model call's context, thinking, reply, tool call and tool result.

    python3 scripts/walkthrough.py runs/dev33_base/traj/fastapi_13713.jsonl              # → stdout (Markdown)
    python3 scripts/walkthrough.py runs/dev33_base/traj/fastapi_13713.jsonl -o x.md      # → file
    python3 scripts/walkthrough.py runs/dev33_base/traj/fastapi_13713.jsonl --brief      # 📷 one line per call
    python3 scripts/walkthrough.py runs/dev33_base/traces/trace_fastapi_13713.json       # harness trace (no thinking)
    python3 scripts/walkthrough.py runs/dev33_base/traj                                  # 📷 one summary line per run dir

Input 1, traj/<task>.jsonl (scripts/traj_log.py, box runs): the exact requests and replies, with the thinking.
Input 2, traces/trace_<task>.json (the harness's own trace): tool calls, results and tokens, but no thinking and no
re-sent messages. Outputs quote task data → keep them in runs/ (gitignored).
"""
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("-")]
BRIEF = "--brief" in sys.argv
OUT = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None
if OUT in args:
    args.remove(OUT)
SRC = args[0]
L = []
w = L.append


def text_of(c):
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c or ""


def tok(chars):
    return f"~{chars // 4:,} tok"


def one(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def fence(body, lang="text"):
    ticks = "````" if "```" in body else "```"
    return f"{ticks}{lang}\n{body}\n{ticks}\n"


def render_msg(m, k):
    role = m.get("role")
    if role == "system":
        b = text_of(m.get("content"))
        w(f"**[system]** ({len(b):,} chars, {tok(len(b))})\n")
        w(fence(b, "markdown"))
    elif role == "user":
        b = text_of(m.get("content"))
        w(f"**[user]** ({len(b):,} chars, {tok(len(b))})\n")
        w(fence(b))
    elif role == "assistant":
        r = m.get("reasoning_content") or m.get("reasoning")
        tb = m.get("thinking_blocks")
        w(f"**[assistant, re-sent]** content {len(text_of(m.get('content'))):,} chars · "
          f"tool_calls {len(m.get('tool_calls') or [])} · "
          f"**thinking re-sent: {'yes, reasoning_content ' + format(len(r), ',') + ' chars' if r else ('thinking_blocks' if tb else 'NO')}**\n")
    else:   # tool / tool_responses
        b = text_of(m.get("content"))
        w(f"**[{role}]** id `{m.get('tool_call_id')}` ({len(b):,} chars, {tok(len(b))})\n")
        w(fence(b))


def from_traj(path):
    recs = [json.loads(x) for x in open(path) if x.strip()]
    task = recs[0]["task"] if recs else "?"
    tot_p = sum((r.get("usage") or {}).get("prompt") or 0 for r in recs)
    tot_c = sum((r.get("usage") or {}).get("completion") or 0 for r in recs)
    think = sum(len((r.get("resp") or {}).get("reasoning") or "") for r in recs)
    if BRIEF:
        nc = sum(1 for r in recs if r.get("compaction"))
        print(f"{task}: {len(recs)} calls{f' ({nc} compaction)' if nc else ''} · prompt {tot_p:,} · completion {tot_c:,} · thinking {think:,} chars")
        print(" #  prompt  Δ      out  think   s    reply")
        prev = None
        for r in recs:
            u, rp = r.get("usage") or {}, r.get("resp") or {}
            p = u.get("prompt") or 0
            tc = rp.get("tool_calls") or []
            if r.get("compaction"):   # an events-compaction summary call (scripts/compaction_shim.py), not an agent turn
                reply = f"COMPACTION summary → {len(text_of(rp.get('content'))):,} chars"
            elif tc:   # every tool call of the reply; the first one with its arguments
                reply = f"{tc[0]['name']} {one(tc[0].get('args'), 60 if len(tc) == 1 else 40)}" + "".join(f" + {t['name']}" for t in tc[1:])
            else:
                reply = f"TEXT {one(rp.get('content'), 60)}" if rp.get("content") else rp.get("unparsed") or "-"
            fr = rp.get("finish_reason")
            print(f"{r['call']:2} {p:7,} {'' if prev is None else format(p - prev, '+,'):>6} {u.get('completion') or 0:5,} "
                  f"{len(rp.get('reasoning') or ''):6,} {r.get('s', 0):5.1f}  {reply}"
                  f"{'  [' + fr + ']' if fr and fr not in ('stop', 'tool_calls') else ''}{'  ERR ' + one(r['err'], 50) if r.get('err') else ''}")
            prev = p if not r.get("compaction") else prev
        resent = [m for r in recs for m in r.get("new", []) if m.get("role") == "assistant"]
        with_r = sum(1 for m in resent if m.get("reasoning_content") or m.get("reasoning"))
        print(f"re-sent assistant turns: {len(resent)} · with thinking attached: {with_r}")
        return
    w(f"# Trajectory · {task}\n")
    w(f"Source: `{path}` (scripts/traj_log.py) · {len(recs)} model calls · prompt {tot_p:,} tok total · completion "
      f"{tot_c:,} tok · thinking {think:,} chars\n")
    if recs and recs[0].get("setup"):
        st = recs[0]["setup"]
        w(f"**Setup (call 1):** model `{st.get('model')}` · tools {len(st.get('tools') or [])}: "
          f"{', '.join(st.get('tools') or [])}\n")
        w(fence(json.dumps(st.get("params"), indent=1), "json"))
    w("Each call shows the messages ADDED to the request since the previous call (the first call: system + task), "
      "then what the model returned. The previous reply reappears as `[assistant, re-sent]`: that line says whether "
      "its thinking was sent back.\n")
    prev = None
    for r in recs:
        u, rp = r.get("usage") or {}, r.get("resp") or {}
        p = u.get("prompt") or 0
        w(f"## Call {r['call']}{' — COMPACTION summary (not an agent turn)' if r.get('compaction') else ''}\n")
        w(f"**Request:** {r['n_msgs']} messages{' (history RESET: logged in full)' if r.get('reset') else ''} · "
          f"prompt {p:,} tok{'' if prev is None else f' (Δ {p - prev:+,})'} · cached {u.get('cached') or 0:,} · "
          f"{r.get('s', 0):.1f} s\n")
        w("### Added to the context\n")
        for k, m in enumerate(r.get("new", [])):
            render_msg(m, k)
        w("### Model output\n")
        th = rp.get("reasoning")
        w(f"💭 **Thinking** ({len(th or ''):,} chars · reasoning tokens {u.get('reasoning') or '?'}):\n")
        w(fence(th) if th else "_(none returned)_\n")
        if rp.get("content"):
            w("**Text:**\n\n> " + text_of(rp["content"]).replace("\n", "\n> ") + "\n")
        for tc in rp.get("tool_calls") or []:
            a = tc.get("args")
            try:
                a = json.loads(a) if isinstance(a, str) else a
            except ValueError:
                pass
            w(f"**Tool call:** `{tc.get('name')}` (id `{tc.get('id')}`)\n")
            w(fence(a["command"], "bash") if isinstance(a, dict) and list(a) == ["command"] else
              fence(json.dumps(a, indent=1) if not isinstance(a, str) else a, "json"))
        w(f"finish_reason `{rp.get('finish_reason')}` · completion {u.get('completion') or 0:,} tok"
          f"{' · ERROR ' + r['err'] if r.get('err') else ''}{' · ' + rp['unparsed'] if rp.get('unparsed') else ''}\n")
        prev = p if not r.get("compaction") else prev


def from_trace(path):
    d = json.load(open(path))
    steps = d["steps"]
    if BRIEF:
        print(f"{path}: {len(steps)} steps (harness trace: no thinking)")
        for s in steps[2:]:
            for tc in s.get("tool_calls") or []:
                print(f"{s['step_id']:3} {(s.get('metrics') or {}).get('prompt_tokens', ''):>7} {tc['function_name']} "
                      f"{one(json.dumps(tc['arguments']), 70)}")
            if s.get("message"):
                print(f"{s['step_id']:3} {'':>7} TEXT {one(s['message'], 70)}")
        return
    fm = d.get("final_metrics", {})
    w(f"# Trace · {path.split('trace_')[-1].removesuffix('.json')}\n")
    w(f"Source: `{path}` (harness trace: tool calls, results and tokens; no thinking, no re-sent messages) · "
      f"prompt {fm.get('total_prompt_tokens', 0):,} · completion {fm.get('total_completion_tokens', 0):,}\n")
    w(f"## System ({len(steps[0]['message']):,} chars)\n")
    w(fence(steps[0]["message"], "markdown"))
    w(f"## Task prompt ({len(steps[1]['message']):,} chars)\n")
    w(fence(steps[1]["message"]))
    for s in steps[2:]:
        m = s.get("metrics") or {}
        w(f"## Step {s['step_id']} ({s['source']}{', ' + s['extra'].get('event_type') if s['extra'].get('event_type') else ''})"
          f"{' · prompt ' + format(m['prompt_tokens'], ',') + ' · completion ' + format(m.get('completion_tokens', 0), ',') if m else ''}\n")
        if s.get("message"):
            w("> " + s["message"].replace("\n", "\n> ") + "\n")
        for tc in s.get("tool_calls") or []:
            a = tc["arguments"]
            w(f"**Tool call:** `{tc['function_name']}`\n")
            w(fence(a["command"], "bash") if list(a) == ["command"] else fence(json.dumps(a, indent=1), "json"))
        if s.get("observation"):
            raw = s["observation"].get("content", "")
            try:
                j = json.loads(raw)
                body = (j.get("stdout") or "") + (("\n[stderr]\n" + j["stderr"]) if j.get("stderr") else "")
                meta = {k: v for k, v in j.items() if k not in ("stdout", "stderr")}
            except ValueError:
                body, meta = raw, {}
            w(f"**Tool result** ({len(raw):,} chars) {json.dumps(meta)}\n")
            w(fence(body or raw))


def summary(dirs):
    """one line per run: calls, thinking volume, and whether re-sent assistant turns still carry their thinking"""
    import pathlib
    for dd in dirs:
        recs = [json.loads(x) for f in sorted(pathlib.Path(dd).glob("*.jsonl")) if not f.name.endswith(".tools.jsonl")
                for x in open(f) if x.strip()]
        if not recs:
            print(f"traj {dd}: no calls logged")
            continue
        tasks = {r["task"] for r in recs}
        th = [len((r.get("resp") or {}).get("reasoning") or "") for r in recs]
        resent = [m for r in recs for m in r.get("new", []) if m.get("role") == "assistant"]
        with_r = sum(1 for m in resent if m.get("reasoning_content") or m.get("reasoning") or m.get("thinking_blocks"))
        cut = sum(1 for r in recs if str((r.get("resp") or {}).get("finish_reason")) in ("length", "max_tokens"))
        errs = sum(1 for r in recs if r.get("err"))
        print(f"traj {dd}: {len(tasks)} tasks · {len(recs)} calls · thinking in {sum(1 for t in th if t)}/{len(recs)} replies, "
              f"mean {sum(th) // max(1, len(th)):,} chars · re-sent turns with thinking {with_r}/{len(resent)} · "
              f"cut at max tokens {cut} · call errors {errs}")


if not SRC.endswith((".json", ".jsonl")):
    summary(args)
    sys.exit(0)
(from_traj if SRC.endswith(".jsonl") else from_trace)(SRC)
if not BRIEF:
    text = "\n".join(L) + "\n"
    if OUT:
        open(OUT, "w").write(text)
        print(f"wrote {OUT}: {len(text):,} chars")
    else:
        sys.stdout.write(text)
