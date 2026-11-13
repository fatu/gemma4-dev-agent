#!/usr/bin/env python3
"""B-10 step 0: does the teacher behind an OpenAI-compatible gateway work for mini-swe-agent data? (a few cents)

    export GATEWAY_BASE_URL=…  GATEWAY_API_KEY=…  TEACHER=<the gateway's model name>     # typed on the box, never in git
    .venv-swe/bin/python scripts/teacher_probe.py            # python3 works too (then the litellm part is skipped)

Three chat turns with mini-swe-agent's system prompt and its single `bash` tool; the tool results are canned text (nothing
is executed). Per turn it prints: tool calls returned · where the reasoning comes back (`reasoning_content`, `reasoning`,
`reasoning_details`, a <think> block in the content) · usage (prompt, cached, completion, reasoning tokens) · latency ·
rate-limit headers. The history is re-sent the way mini does it (the assistant message as returned, reasoning included);
if the gateway refuses that, it is retried without the reasoning fields and the fact is printed. Then the same first turn
through litellm (`openai/$TEACHER`, the route mini uses) to see whether litellm passes the reasoning on.
The base URL and the key are never printed. Knobs (printed): --temperature 0.7 · --max-tokens 4096 (both = the settings of
the MetaCurate pool) · --effort "" (sent as `reasoning_effort` only if given) · --extra "" (JSON merged into the request
body, for a gateway-specific thinking switch) · --rounds 3 · --timeout 300.
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
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                   "required": ["command"]}}}]
CANNED = ["<returncode>0</returncode>\n<output>\ndef mean(xs):\n    return sum(xs) / len(xs)\n</output>",
          "<returncode>0</returncode>\n<output>\n</output>",
          "<returncode>0</returncode>\n<output>\n0.0\n</output>"]
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thinking_blocks")


def endpoint(base):
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def post(url, key, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), dict(r.headers), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error_body": e.read().decode("utf-8", "replace")[:400]}, dict(e.headers), time.time() - t0
    except Exception as e:  # noqa: BLE001  (network errors are a result here)
        return 0, {"error_body": f"{type(e).__name__}: {str(e)[:200]}"}, {}, time.time() - t0


def reasoning_of(msg):
    """{field: characters} for every place the reasoning could be."""
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


def strip_reasoning(msg):
    return {k: v for k, v in msg.items() if k not in REASONING_KEYS and k != "provider_specific_fields"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--effort", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    base, key, teacher = (os.environ.get(k, "") for k in ("GATEWAY_BASE_URL", "GATEWAY_API_KEY", "TEACHER"))
    if not (base and key and teacher):
        sys.exit("set GATEWAY_BASE_URL, GATEWAY_API_KEY and TEACHER (export …) first; nothing was sent")
    extra = json.loads(a.extra) if a.extra else {}
    print(f"teacher_probe · teacher {teacher} · base url set ({len(base)} chars, not printed) · key set ({len(key)} chars, not "
          f"printed) · temperature {a.temperature} · max_tokens {a.max_tokens} · reasoning_effort {a.effort or 'not sent'} · "
          f"extra body keys {sorted(extra) or 'none'} · rounds {a.rounds} · timeout {a.timeout} s")
    url = endpoint(base)
    host = base.split("//")[-1].split("/")[0]

    def red(t):                                   # an error text may echo the address or the key: never print them
        for secret in (base, host, host.split(":")[0], key):
            t = t.replace(secret, "<hidden>")
        return t
    common = {"model": teacher, "tools": TOOLS, "temperature": a.temperature, "max_tokens": a.max_tokens, **extra}
    if a.effort:
        common["reasoning_effort"] = a.effort
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    n_tool = n_reason = 0
    history_ok, stripped = "not tested (no second turn)", False
    tot = {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0, "s": 0.0}
    for k in range(a.rounds):
        status, j, hdr, dt = post(url, key, {**common, "messages": msgs}, a.timeout)
        if status == 400 and k > 0 and not stripped:
            print(f"  turn {k + 1}: HTTP 400 with the reasoning in the history ({red(j.get('error_body', ''))[:160]!r}) → retried without it")
            msgs = [strip_reasoning(m) if m.get("role") == "assistant" else m for m in msgs]
            stripped, history_ok = True, "NO (the gateway refuses reasoning fields in earlier assistant messages)"
            status, j, hdr, dt = post(url, key, {**common, "messages": msgs}, a.timeout)
        if status != 200:
            print(f"  turn {k + 1}: HTTP {status} after {dt:.1f} s · {red(j.get('error_body', ''))[:300]!r}")
            break
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        tcs = msg.get("tool_calls") or []
        rs = reasoning_of(msg)
        u = j.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        cmd = ""
        if tcs:
            try:
                cmd = json.loads(tcs[0]["function"]["arguments"]).get("command", "")
            except Exception:  # noqa: BLE001
                cmd = f"(arguments not JSON: {str(tcs[0].get('function', {}).get('arguments'))[:60]!r})"
        n_tool += bool(tcs)
        n_reason += bool(rs)
        for name, v in (("prompt", u.get("prompt_tokens")), ("cached", cached), ("completion", u.get("completion_tokens")),
                        ("reasoning", rtok)):
            tot[name] += v or 0
        tot["s"] += dt
        rl = {h: v for h, v in hdr.items() if "ratelimit" in h.lower()}
        print(f"  turn {k + 1}: {dt:.1f} s · model {j.get('model')} · finish {ch.get('finish_reason')} · tool calls {len(tcs)}"
              f"{' · first: ' + cmd[:90].replace(chr(10), ' ⏎ ') if cmd else ''} · content {len(msg.get('content') or '')} chars · "
              f"reasoning {rs or 'NOT RETURNED'} · usage prompt {u.get('prompt_tokens')} cached {cached} completion "
              f"{u.get('completion_tokens')} reasoning {rtok} · message keys {sorted(msg)}"
              f"{' · rate-limit headers ' + json.dumps(rl) if rl and k == 0 else ''}")
        if k > 0 and not stripped:
            history_ok = "yes"
        if not tcs:
            print("    no tool call in this turn: stopping the conversation here")
            break
        msgs.append(strip_reasoning(msg) if stripped else msg)
        for i, tc in enumerate(tcs):
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": CANNED[min(k, len(CANNED) - 1)] if i == 0
                         else "<returncode>0</returncode>\n<output>\n</output>"})
    print(f"== turns with a tool call {n_tool} / {a.rounds} · turns with reasoning returned {n_reason} / {a.rounds} · history with "
          f"reasoning accepted: {history_ok} · tokens prompt {tot['prompt']} (cached {tot['cached']}) completion "
          f"{tot['completion']} (reasoning {tot['reasoning']}) · {tot['s']:.1f} s in all")
    # --- the route mini-swe-agent uses: litellm, model "openai/<name>" + api_base ---
    lit = "skipped (litellm not importable: run with .venv-swe/bin/python)"
    try:
        import litellm
        t0 = time.time()
        kw = dict(extra)
        if a.effort:
            kw["reasoning_effort"] = a.effort
        r = litellm.completion(model="openai/" + teacher, api_base=base, api_key=key, messages=msgs[:2], tools=TOOLS,
                               temperature=a.temperature, max_tokens=a.max_tokens, drop_params=True, timeout=a.timeout, **kw)
        m = r.choices[0].message.model_dump()
        lit = (f"{time.time() - t0:.1f} s · tool calls {len(m.get('tool_calls') or [])} · reasoning {reasoning_of(m) or 'NOT RETURNED'} "
               f"· message keys {sorted(k for k, v in m.items() if v)} · litellm {getattr(litellm, '__version__', '?')}")
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        lit = f"FAILED {type(e).__name__}: {red(str(e))[:240]}"
    print(f"== through litellm (openai/{teacher}): {lit}")
    go = n_tool == a.rounds and n_reason >= a.rounds - 1
    print(f"RESULT {'go' if go else 'stop + photo'} · 📷 every line from 'teacher_probe' down")


if __name__ == "__main__":
    main()
