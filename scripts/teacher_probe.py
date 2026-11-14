#!/usr/bin/env python3
"""B-10 step 0: does the teacher behind the gateway work for mini-swe-agent data? (a few cents)

    export GATEWAY_BASE_URL=…  GATEWAY_API_KEY=…  TEACHER=<the gateway's model name>     # typed on the box, never in git
    .venv-swe/bin/python scripts/teacher_probe.py            # python3 works too (then the litellm part is skipped)

--api messages (default) = the Anthropic format, POST <base>/v1/messages with `x-api-key` (and the same key as a Bearer
token); --api chat = the OpenAI format, POST <base>/v1/chat/completions. The base may be given with or without a
trailing /v1. If GATEWAY_BASE_URL / GATEWAY_API_KEY are not set, ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN /
ANTHROPIC_API_KEY are used (the names that were used are printed, never the values); a base URL is always required, so
nothing can go to a provider's own endpoint by default.

Three turns with mini-swe-agent's system prompt and its single `bash` tool; the tool results are canned text (nothing is
executed). Per turn it prints: tool calls returned · where the reasoning comes back (messages: `thinking` blocks; chat:
`reasoning_content`, `reasoning`, `reasoning_details`, a <think> block) · usage (prompt, cached, completion, reasoning
tokens) · latency · rate-limit headers. The history is re-sent the way mini does it (the assistant turn as returned,
reasoning included); if the gateway refuses that, it is retried without the reasoning and the fact is printed. Then the
same first turn through litellm (`anthropic/$TEACHER` or `openai/$TEACHER`: the route mini uses) to see whether litellm
passes the reasoning on. The base URL and the key are never printed.
Knobs (printed): --temperature 0.7 · --max-tokens 4096 (both = the settings of the MetaCurate pool) · --thinking-budget 0
(messages: > 0 sends `thinking: {type: enabled, budget_tokens}`; 0 = not sent, the model's default) · --effort "" (chat:
sent as `reasoning_effort` only if given) · --extra "" (JSON merged into the request body) · --rounds 3 · --timeout 300.
RESULT go = every turn has a tool call and the reasoning came back in at least all but one turn; otherwise stop + photo.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SYSTEM = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
TASK = """<pr_description>
`mean([])` in /testbed/stats.py raises ZeroDivisionError. It should return 0.0 for an empty list.
</pr_description>

<instructions>
You work in /testbed by issuing bash commands with the `bash` tool, one step at a time: first look at the file, then fix
it, then check the fix. Explain your reasoning briefly before each command. When you are done, run
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
</instructions>"""
PARAMS = {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
          "required": ["command"]}
TOOLS_CHAT = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": PARAMS}}]
TOOLS_MSG = [{"name": "bash", "description": "Execute a bash command", "input_schema": PARAMS}]
CANNED = ["<returncode>0</returncode>\n<output>\ndef mean(xs):\n    return sum(xs) / len(xs)\n</output>",
          "<returncode>0</returncode>\n<output>\n</output>",
          "<returncode>0</returncode>\n<output>\n0.0\n</output>"]
EMPTY = "<returncode>0</returncode>\n<output>\n</output>"
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thinking_blocks")


def root_of(base):
    """The address without a trailing /v1, /v1/messages or /v1/chat/completions."""
    base = base.rstrip("/")
    for tail in ("/chat/completions", "/messages", "/v1"):
        if base.endswith(tail):
            base = base[: -len(tail)].rstrip("/")
    return base


def post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), dict(r.headers), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error_body": e.read().decode("utf-8", "replace")[:400]}, dict(e.headers), time.time() - t0
    except Exception as e:  # noqa: BLE001  (network errors are a result here)
        return 0, {"error_body": f"{type(e).__name__}: {str(e)[:200]}"}, {}, time.time() - t0


def reasoning_fields(msg):
    """chat format: {field: characters} for every place the reasoning could be."""
    out = {}
    for k in REASONING_KEYS:
        v = msg.get(k)
        if v:
            out[k] = len(v) if isinstance(v, str) else len(json.dumps(v))
    psf = msg.get("provider_specific_fields") or {}
    for k in REASONING_KEYS:
        if isinstance(psf, dict) and psf.get(k):
            out["provider_specific_fields." + k] = len(json.dumps(psf[k]))
    c = msg.get("content")
    if isinstance(c, str) and "<think>" in c:
        out["<think> in content"] = len(c)
    return out


class Chat:
    """OpenAI format."""
    name, path, lit = "chat", "/v1/chat/completions", "openai/"

    def headers(self, key):
        return {"Authorization": "Bearer " + key}

    def body(self, a, teacher, msgs, extra):
        b = {"model": teacher, "tools": TOOLS_CHAT, "temperature": a.temperature, "max_tokens": a.max_tokens,
             "messages": [{"role": "system", "content": SYSTEM}] + msgs, **extra}
        if a.effort:
            b["reasoning_effort"] = a.effort
        return b

    def read(self, j):
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        cmds = []
        for tc in msg.get("tool_calls") or []:
            try:
                cmds.append((tc.get("id"), json.loads(tc["function"]["arguments"]).get("command", "")))
            except Exception:  # noqa: BLE001
                cmds.append((tc.get("id"), f"(arguments not JSON: {str(tc.get('function', {}).get('arguments'))[:60]!r})"))
        u = j.get("usage") or {}
        return {"msg": msg, "cmds": cmds, "finish": ch.get("finish_reason"), "content": len(msg.get("content") or ""),
                "reasoning": reasoning_fields(msg), "keys": sorted(msg),
                "usage": (u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          u.get("completion_tokens"), (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))}

    def extend(self, msgs, r, k):
        msgs.append(r["msg"])
        for i, (tid, _) in enumerate(r["cmds"]):
            msgs.append({"role": "tool", "tool_call_id": tid, "content": CANNED[min(k, len(CANNED) - 1)] if i == 0 else EMPTY})

    def strip(self, m):
        if m.get("role") != "assistant":
            return m
        return {k: v for k, v in m.items() if k not in REASONING_KEYS and k != "provider_specific_fields"}


class Messages:
    """Anthropic format."""
    name, path, lit = "messages", "/v1/messages", "anthropic/"

    def headers(self, key):
        return {"x-api-key": key, "Authorization": "Bearer " + key, "anthropic-version": "2023-06-01"}

    def body(self, a, teacher, msgs, extra):
        b = {"model": teacher, "tools": TOOLS_MSG, "temperature": a.temperature, "max_tokens": a.max_tokens,
             "system": SYSTEM, "messages": msgs, **extra}
        if a.thinking_budget > 0:
            b["thinking"] = {"type": "enabled", "budget_tokens": a.thinking_budget}
        return b

    def read(self, j):
        blocks = j.get("content") or []
        kinds = [b.get("type") for b in blocks]
        cmds = [(b.get("id"), (b.get("input") or {}).get("command", "") if isinstance(b.get("input"), dict)
                 else f"(input not an object: {str(b.get('input'))[:60]!r})") for b in blocks if b.get("type") == "tool_use"]
        rs = {}
        for b in blocks:
            if b.get("type") in ("thinking", "redacted_thinking"):
                rs[b["type"] + " block"] = rs.get(b["type"] + " block", 0) + len(b.get("thinking") or b.get("data") or "")
            if b.get("type") == "text" and "<think>" in (b.get("text") or ""):
                rs["<think> in text"] = len(b["text"])
        u = j.get("usage") or {}
        fresh, read_, made = u.get("input_tokens") or 0, u.get("cache_read_input_tokens"), u.get("cache_creation_input_tokens")
        return {"msg": {"role": "assistant", "content": blocks}, "cmds": cmds, "finish": j.get("stop_reason"),
                "content": sum(len(b.get("text") or "") for b in blocks if b.get("type") == "text"), "reasoning": rs,
                "keys": kinds + [f"signature {'yes' if any(b.get('signature') for b in blocks) else 'no'}"],
                "usage": (fresh + (read_ or 0) + (made or 0), read_, u.get("output_tokens"), None)}

    def extend(self, msgs, r, k):
        msgs.append(r["msg"])
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": CANNED[min(k, len(CANNED) - 1)] if i == 0 else EMPTY}
            for i, (tid, _) in enumerate(r["cmds"])]})

    def strip(self, m):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            return m
        return {**m, "content": [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("GATEWAY_API", "messages"), choices=["messages", "chat"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--effort", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    env = os.environ
    base_name = next((n for n in ("GATEWAY_BASE_URL", "ANTHROPIC_BASE_URL") if env.get(n)), "")
    key_name = next((n for n in ("GATEWAY_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY") if env.get(n)), "")
    teacher = env.get("TEACHER", "")
    if not (base_name and key_name and teacher):
        sys.exit("set GATEWAY_BASE_URL (or ANTHROPIC_BASE_URL), GATEWAY_API_KEY (or ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY) "
                 f"and TEACHER first — found: base {base_name or 'none'}, key {key_name or 'none'}, teacher "
                 f"{teacher or 'none'}; nothing was sent")
    base, key = env[base_name], env[key_name]
    api = Messages() if a.api == "messages" else Chat()
    extra = json.loads(a.extra) if a.extra else {}
    root = root_of(base)
    host = root.split("//")[-1].split("/")[0]

    def red(t):                                   # an error text may echo the address or the key: never print them
        for secret in (base, root, host, host.split(":")[0], key):
            t = t.replace(secret, "<hidden>")
        return t

    print(f"teacher_probe · api {api.name} (POST <base>{api.path}) · teacher {teacher} · base url from ${base_name} ({len(base)} "
          f"chars, not printed) · key from ${key_name} ({len(key)} chars, not printed) · temperature {a.temperature} · "
          f"max_tokens {a.max_tokens} · thinking budget {a.thinking_budget or 'not sent'} · reasoning_effort "
          f"{a.effort or 'not sent'} · extra body keys {sorted(extra) or 'none'} · rounds {a.rounds} · timeout {a.timeout} s")
    url, hdrs = root + api.path, api.headers(key)
    msgs = [{"role": "user", "content": TASK}]
    n_tool = n_reason = 0
    history_ok, stripped = "not tested (no second turn)", False
    tot = [0, 0, 0, 0]
    secs = 0.0
    for k in range(a.rounds):
        status, j, hdr, dt = post(url, hdrs, api.body(a, teacher, msgs, extra), a.timeout)
        if status == 400 and k > 0 and not stripped:
            print(f"  turn {k + 1}: HTTP 400 with the reasoning in the history ({red(j.get('error_body', ''))[:160]!r}) → retried without it")
            msgs = [api.strip(m) for m in msgs]
            stripped, history_ok = True, "NO (the gateway refuses the reasoning in earlier assistant turns)"
            status, j, hdr, dt = post(url, hdrs, api.body(a, teacher, msgs, extra), a.timeout)
        if status != 200:
            hint = " · 404 / 405 here usually means the gateway does not serve this format: try --api " + \
                   ("chat" if api.name == "messages" else "messages") if status in (404, 405) else ""
            print(f"  turn {k + 1}: HTTP {status} after {dt:.1f} s · {red(j.get('error_body', ''))[:300]!r}{hint}")
            break
        r = api.read(j)
        n_tool += bool(r["cmds"])
        n_reason += bool(r["reasoning"])
        for i, v in enumerate(r["usage"]):
            tot[i] += v or 0
        secs += dt
        rl = {h: v for h, v in hdr.items() if "ratelimit" in h.lower()}
        first = r["cmds"][0][1][:90].replace("\n", " ⏎ ") if r["cmds"] else ""
        print(f"  turn {k + 1}: {dt:.1f} s · model {j.get('model')} · finish {r['finish']} · tool calls {len(r['cmds'])}"
              f"{' · first: ' + first if first else ''} · text {r['content']} chars · reasoning {r['reasoning'] or 'NOT RETURNED'} · "
              f"usage prompt {r['usage'][0]} cached {r['usage'][1]} completion {r['usage'][2]} reasoning {r['usage'][3]} · "
              f"reply parts {r['keys']}{' · rate-limit headers ' + json.dumps(rl) if rl and k == 0 else ''}")
        if k > 0 and not stripped:
            history_ok = "yes"
        if not r["cmds"]:
            print("    no tool call in this turn: stopping the conversation here")
            break
        if stripped:
            r["msg"] = api.strip(r["msg"])
        api.extend(msgs, r, k)
    print(f"== turns with a tool call {n_tool} / {a.rounds} · turns with reasoning returned {n_reason} / {a.rounds} · history with "
          f"reasoning accepted: {history_ok} · tokens prompt {tot[0]} (cached {tot[1]}) completion {tot[2]} (reasoning {tot[3]}) "
          f"· {secs:.1f} s in all")
    # --- the route mini-swe-agent uses: litellm ---
    lit = "skipped (litellm not importable: run with .venv-swe/bin/python)"
    try:
        import litellm
        from importlib.metadata import version
        t0 = time.time()
        kw = dict(extra)
        if a.effort and api.name == "chat":
            kw["reasoning_effort"] = a.effort
        if a.thinking_budget > 0 and api.name == "messages":
            kw["thinking"] = {"type": "enabled", "budget_tokens": a.thinking_budget}
        r = litellm.completion(model=api.lit + teacher, api_base=root if api.name == "messages" else root + "/v1", api_key=key,
                               messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}],
                               tools=TOOLS_CHAT, temperature=a.temperature, max_tokens=a.max_tokens, drop_params=True,
                               timeout=a.timeout, **kw)
        m = r.choices[0].message.model_dump()
        u = r.usage.model_dump() if getattr(r, "usage", None) else {}
        lit = (f"{time.time() - t0:.1f} s · tool calls {len(m.get('tool_calls') or [])} · reasoning "
               f"{reasoning_fields(m) or 'NOT RETURNED'} · message keys {sorted(k for k, v in m.items() if v)} · usage prompt "
               f"{u.get('prompt_tokens')} cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')} completion "
               f"{u.get('completion_tokens')} · litellm {version('litellm')}")
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        lit = f"FAILED {type(e).__name__}: {red(str(e))[:240]}"
    print(f"== through litellm ({api.lit}{teacher}): {lit}")
    go = n_tool == a.rounds and n_reason >= a.rounds - 1
    print(f"RESULT {'go' if go else 'stop + photo'} · 📷 every line from 'teacher_probe' down")


if __name__ == "__main__":
    main()
