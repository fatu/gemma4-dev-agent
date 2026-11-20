#!/usr/bin/env python3
"""A local stand-in for the teacher's gateway (B-11 tests; no key, no network, nothing leaves the machine).

    .venv-eval/bin/python scripts/teach_fake.py serve --port 8765 [--script submit|loop]    # then point api_base at it
    .venv-eval/bin/python scripts/teach_fake.py selftest                                   # adapter + ledger through litellm

`serve` answers POST /v1/messages (the Anthropic format litellm's `anthropic/<name>` route sends) and
POST /v1/chat/completions (the OpenAI format). Every request body is appended to <dir>/requests.jsonl so a test can
read back exactly what was sent. The reply always carries a thinking part, one tool call and a usage block:
  script submit  call 1: `run_command` with `git status --short`; call 2: `write_file` of a one-line new file (when
                 the tool exists); then `submit_patch`; then a text-only reply, which ends the harness's loop
  script loop    always `run_command` (an attempt that never ends by itself — for budget tests)
`selftest` starts the server in a thread, imports scripts/ceil_sitecustomize.py with the B-11 knobs set and checks, on
the bodies the server received and on the ledger: the temperature that was sent, the thinking request for each
CEIL_THINK form, that the harness's vLLM-only fields are gone, that the thought comes back as reasoning_content, that
every call is charged with its tokens, and that the gate stops the calls at the budget.
"""
import argparse
import http.server
import json
import os
import sys
import threading

USAGE = {"input_tokens": 1200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0, "output_tokens": 150}


def SIG():
    """FAKE_SIG=none: thinking blocks without a `signature` key (the shape with which ADK discards the thought)"""
    return {} if os.environ.get("FAKE_SIG") == "none" else {"signature": "fake"}


def make_handler(logdir, script):
    n = {"calls": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            n["calls"] += 1
            with open(os.path.join(logdir, "requests.jsonl"), "a") as f:
                f.write(json.dumps({"path": self.path, "body": body}) + "\n")
            tools = [t.get("name") or (t.get("function") or {}).get("name") for t in body.get("tools") or []]
            seen = sum(len(b.get("thinking", "")) for m in body.get("messages") or [] if isinstance(m.get("content"), list)
                       for b in m["content"] if isinstance(b, dict) and b.get("type") == "thinking")
            seen += sum(len(m.get("reasoning_content") or "") for m in body.get("messages") or [])
            turns = sum(1 for m in body.get("messages") or [] if m.get("role") == "assistant")
            name = next((t for t in tools if t in ("run_command", "bash")), (tools or ["run_command"])[0])
            args = {"command": "git status --short"}
            if script == "submit" and turns == 1 and "write_file" in tools:       # an edit, so that the patch is not empty
                name, args = "write_file", {"filepath": "zz_teach_fake.py", "content": "x = 1\n"}
            elif script == "submit" and turns >= 1 and "submit_patch" in tools:
                name, args = "submit_patch", {}
            thought = f"Stand-in thought {turns + 1}: look at the tree, then act."
            done = script == "submit" and "submit_patch" in json.dumps([m for m in body.get("messages") or [] if m.get("role") == "assistant"])
            if done:                      # after its submit_patch the model ends with a text-only reply; that is what ends
                # the harness's loop (measured 2026-11-18: a submit_patch call alone does not, the run went on to 80 turns)
                if self.path.rstrip("/").endswith("/messages"):
                    out = {"id": f"msg_fake_{n['calls']}", "type": "message", "role": "assistant", "model": body.get("model"),
                           "content": [dict({"type": "thinking", "thinking": thought}, **SIG()), {"type": "text", "text": "Submitted."}],
                           "stop_reason": "end_turn", "stop_sequence": None, "usage": dict(USAGE)}
                else:
                    out = {"id": f"chatcmpl-fake-{n['calls']}", "object": "chat.completion", "created": 0, "model": body.get("model"),
                           "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Submitted.", "reasoning_content": thought}}],
                           "usage": {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350,
                                     "prompt_tokens_details": {"cached_tokens": 800}, "completion_tokens_details": {"reasoning_tokens": 40}}}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                return
            if self.path.rstrip("/").endswith("/messages"):
                out = {"id": f"msg_fake_{n['calls']}", "type": "message", "role": "assistant", "model": body.get("model"),
                       "content": [dict({"type": "thinking", "thinking": thought}, **SIG()),
                                   {"type": "tool_use", "id": f"toolu_fake_{n['calls']}", "name": name, "input": args}],
                       "stop_reason": "tool_use", "stop_sequence": None, "usage": dict(USAGE, input_tokens=USAGE["input_tokens"] + seen)}
            else:
                out = {"id": f"chatcmpl-fake-{n['calls']}", "object": "chat.completion", "created": 0, "model": body.get("model"),
                       "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                           "role": "assistant", "content": None, "reasoning_content": thought,
                           "tool_calls": [{"id": f"call_fake_{n['calls']}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}]}}],
                       "usage": {"prompt_tokens": 1200 + seen, "completion_tokens": 150, "total_tokens": 1350 + seen,
                                 "prompt_tokens_details": {"cached_tokens": 800},
                                 "completion_tokens_details": {"reasoning_tokens": 40}}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    return H


def start(logdir, port=0, script="submit"):
    os.makedirs(logdir, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(logdir, script))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def bodies(logdir):
    p = os.path.join(logdir, "requests.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


TOOLS = [{"type": "function", "function": {"name": "run_command", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "submit_patch", "description": "submit", "parameters": {"type": "object", "properties": {}}}}]


def _one(think, fmt, tmp, extra_env=None):
    """a fresh interpreter per CEIL_THINK form (the adapter reads its knobs at import)"""
    import subprocess
    code = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import teach_fake, teach_ledger
srv = teach_fake.start(os.environ["FAKE_DIR"], script=os.environ.get("FAKE_SCRIPT", "submit")); port = srv.server_address[1]
import ceil_sitecustomize, litellm
model = ("anthropic/" if os.environ["FAKE_FMT"] == "anthropic" else "openai/") + "stand-in-teacher"
base = f"http://127.0.0.1:{port}" + ("" if os.environ["FAKE_FMT"] == "anthropic" else "/v1")
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
if os.environ.get("FAKE_TOOLMSG"):     # an agent request as the harness sends it after a tool ran: the last message is the tool's
    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": ("submit_patch" if os.environ["FAKE_TOOLMSG"] == "submit" else "run_command"), "arguments": "{\"command\": \"ls\"}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "{\"stdout\": \"a.py\"}"}]
before = json.dumps(msgs)
out = {"calls": 0, "reasoning": 0, "gated": False}
try:
    for i in range(8):
        r = litellm.completion(model=model, api_base=base, api_key="not-a-key", messages=msgs, tools=teach_fake.TOOLS,
                               temperature=0.2, top_p=0.95, max_tokens=512,
                               extra_body={"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096})
        m = r.choices[0].message; out["calls"] += 1
        out["reasoning"] += bool(getattr(m, "reasoning_content", None))
        if not m.tool_calls:      # the closing text-only reply
            out["tool"] = None; continue
        out["tool"] = m.tool_calls[0].function.name
        if os.environ.get("FAKE_ADK"):
            import asyncio
            from google.adk.models import lite_llm as L
            content = L._message_to_generate_content_response(m).content        # reply → ADK parts (what the session stores)
            back = asyncio.run(L._content_to_message_param(content, model=model))  # parts → the message of the next request
            back = dict(back)
            out.setdefault("resent_keys", []).append(sorted(k for k, v in back.items() if v is not None))
            msgs = msgs + [back, {"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": "{\"stdout\": \"ok\"}"}]
            if i == 3: break
except teach_ledger.BudgetReached:
    out["gated"] = True
out["caller_messages_unchanged"] = json.dumps(msgs) == before
print("RESULT " + json.dumps(out))
'''
    env = dict(os.environ, FAKE_DIR=tmp, FAKE_FMT=fmt, CEIL_TEMP="0.7", CEIL_THINK=think, CEIL_GEMMA_CTX="0",
               TEACH_LEDGER_DIR=os.path.join(tmp, "ledger"), TEACH_ATTEMPT="2", CEIL_CALL_LOG=os.path.join(tmp, "calls.jsonl"))
    env.update(extra_env or {})
    if fmt == "openai":
        env["CEIL_REMOTE"] = "1"
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise SystemExit(f"selftest child failed ({think}, {fmt}):\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}")
    banner = next((l for l in p.stdout.splitlines() if l.startswith("ceil adapter:")), "")
    return dict(json.loads(line[7:]), stdout=p.stdout), banner


def selftest():
    import shutil
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import teach_ledger
    rows = []
    for think, fmt, want in (("budget:2000", "anthropic", lambda b: b.get("thinking") == {"type": "enabled", "budget_tokens": 2000}),
                             ("off", "anthropic", lambda b: "thinking" not in b),
                             ('extra:{"reasoning_effort": "high"}', "anthropic", lambda b: b.get("reasoning_effort") == "high"),
                             ('extra:{"reasoning_effort": "high"}', "openai", lambda b: b.get("reasoning_effort") == "high"),
                             ("effort:high", "openai", lambda b: b.get("reasoning_effort") == "high")):
        tmp = tempfile.mkdtemp(prefix="teach_fake_")
        try:
            L = teach_ledger.Ledger(os.path.join(tmp, "ledger")); L.init(1.0, [1.0, 0.1, 4.0])   # one call = (1200 fresh × 1 + 800 cached × 0.1 + 150 out × 4) / 1e6 = $0.00188
            res, banner = _one(think, fmt, tmp)
            B = [x["body"] for x in bodies(tmp)]
            st = L.status(); per = L.per_key()
            assert B, "the server received nothing"
            assert all(abs(b.get("temperature", -1) - 0.7) < 1e-9 for b in B), [b.get("temperature") for b in B]
            assert all("top_p" not in b for b in B), "top_p was sent beside the temperature"
            assert all("chat_template_kwargs" not in b and "thinking_token_budget" not in b for b in B), "vLLM-only fields reached the gateway"
            assert all(want(b) for b in B), (think, fmt, {k: B[0].get(k) for k in ("thinking", "reasoning_effort")})
            assert res["reasoning"] == res["calls"] > 0, res
            assert list(per) == ["unknown#2"] and per["unknown#2"]["calls"] == len(B) == st["calls"], (per, len(B))
            want_in = 2000 if fmt == "anthropic" else 1200      # litellm reports Anthropic input as fresh + cache reads
            assert per["unknown#2"]["in"] == want_in * len(B) and per["unknown#2"]["cached"] == 800 * len(B) and per["unknown#2"]["out"] == 150 * len(B), per
            assert per["unknown#2"]["errors"] == 0 and len(B) == 8, (per, len(B))
            assert abs(st["spent_usd"] - len(B) * ((want_in - 800) * 1.0 + 800 * 0.1 + 150 * 4.0) / 1e6) < 1e-9, st["spent_usd"]
            rows.append(f"{think} [{fmt}]: {len(B)} calls · temperature 0.7 · thought returned {res['reasoning']}/{res['calls']} · charged ${st['spent_usd']:.5f}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # the reminder: at the end of the last message of the request the server receives, on both routes, after a user and
    # after a tool message; the caller's own message list (what the harness keeps and the trajectory log sees) unchanged
    for fmt in ("anthropic", "openai"):
        for toolmsg in ("", "1"):
            tmp = tempfile.mkdtemp(prefix="teach_fake_")
            try:
                teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
                res, banner_r = _one("effort:high", fmt, tmp, {"CEIL_REMINDER": "1", "FAKE_TOOLMSG": toolmsg})
                B = [x["body"] for x in bodies(tmp)]
                assert len(B) == 8 and res["caller_messages_unchanged"], res
                for b in B:
                    last = json.dumps(b["messages"][-1]); rest = json.dumps(b["messages"][:-1]) + json.dumps(b.get("system", ""))
                    assert "<reminder>Before EVERY tool call" in last and last.count("<reminder>") == 1 and "<reminder>" not in rest, last[:300]
                for b in B:      # a block of its own: never inside a tool's output, never glued to the task text
                    m = b["messages"][-1]; c = m["content"]
                    assert m["role"] == "user" and "<reminder>" not in json.dumps([x for x in b["messages"] if x.get("role") == "tool"]), m
                    if isinstance(c, list):
                        assert c[-1].get("type") == "text" and c[-1]["text"].startswith("<reminder>"), c[-1]
                        assert all("<reminder>" not in json.dumps(x) for x in c[:-1]), c
                        if toolmsg and fmt == "anthropic":
                            assert c[-2].get("type") == "tool_result", [x.get("type") for x in c]
                    else:
                        assert c.startswith("<reminder>"), c[:80]
                assert "reminder on" in banner_r, banner_r
                logged = [json.loads(l) for l in open(os.path.join(tmp, "calls.jsonl"))]
                assert all(x.get("reminder") is True for x in logged), logged[:1]
                rows.append(f"reminder [{fmt}, last message {'tool' if toolmsg else 'user'}]: a text block of its own at the end of 8/8 requests, once, not inside the tool output; the caller's messages unchanged")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    # earlier thoughts in the history, built through ADK's own conversions as the harness does: with the knob they reach
    # the server on both routes whatever the reply's shape; without it, a block without a signature key is lost (the
    # defect of 2026-11-20, reproduced); the probe reads the difference in prompt tokens
    def thoughts_received(b):
        n = sum(1 for m in b["messages"] if m.get("role") == "assistant" and (m.get("reasoning_content") or (
            isinstance(m.get("content"), list) and any(x.get("type") == "thinking" for x in m["content"] if isinstance(x, dict)))))
        return n, sum(1 for m in b["messages"] if m.get("role") == "assistant")
    for fmt, sig, hist, want in (("anthropic", "none", "1", True), ("anthropic", "fake", "1", True), ("openai", "fake", "1", True),
                                 ("anthropic", "none", "0", False)):
        tmp = tempfile.mkdtemp(prefix="teach_fake_")
        try:
            teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
            os.environ["FAKE_SIG"] = sig
            res, _ = _one("effort:high", fmt, tmp, {"FAKE_ADK": "1", "FAKE_SCRIPT": "loop", "FAKE_SIG": sig, "CEIL_HIST_THOUGHTS": hist, "CEIL_HIST_PROBE": "1", "CEIL_REMINDER": "1"})
            os.environ.pop("FAKE_SIG", None)
            B = [x["body"] for x in bodies(tmp)]
            main_calls = [b for b in B if b.get("max_tokens") != 16]; probes = [b for b in B if b.get("max_tokens") == 16]
            got = [thoughts_received(b) for b in main_calls]
            assert len(main_calls) == 4, len(B)
            if want:
                assert got == [(0, 0), (1, 1), (2, 2), (3, 3)], (fmt, sig, got, res)
                assert len(probes) == 1 and thoughts_received(probes[0]) == (0, 1), [thoughts_received(b) for b in probes]
                assert "REACH the model" in res["stdout"], res["stdout"][-600:]
                if fmt == "anthropic" and sig == "fake":      # the signature a block came with goes back with it
                    assert '"signature": "fake"' in json.dumps(main_calls[-1]["messages"]), main_calls[-1]["messages"][2]
                assert all("reasoning_content" in k and "thinking_blocks" not in k for k in res["resent_keys"]), res["resent_keys"]
            else:
                assert got == [(0, 0), (0, 1), (0, 2), (0, 3)] and not probes, (got, len(probes))
            logged = [json.loads(l) for l in open(os.path.join(tmp, "calls.jsonl"))]
            assert [x["hist"]["with_thought"] for x in logged] == [g[0] for g in got] and "blocks" in logged[0]["think_shape"], logged[:2]
            assert all("<reminder>" in json.dumps(b["messages"][-1]) for b in main_calls)
            rows.append(f"history thoughts [{fmt}, reply block {'without a signature key' if sig == 'none' else 'signed'}, knob {hist}]: "
                        f"earlier thoughts received in calls 1–4: {[g[0] for g in got]} of {[g[1] for g in got]}"
                        + (" · probe: 1 extra request without them, verdict REACH" if want else " (the defect, reproduced)"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for fmt in ("anthropic", "openai"):                # place system: once, in the system prompt, nowhere else
        for toolmsg in ("", "1"):
            tmp = tempfile.mkdtemp(prefix="teach_fake_")
            try:
                teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
                res, banner_s = _one("effort:high", fmt, tmp, {"CEIL_REMINDER": "1", "CEIL_REMINDER_PLACE": "system", "FAKE_TOOLMSG": toolmsg})
                B = [x["body"] for x in bodies(tmp)]
                assert len(B) == 8 and res["caller_messages_unchanged"] and "place system" in banner_s, (len(B), banner_s)
                for b in B:
                    if fmt == "anthropic":
                        sysb = json.dumps(b.get("system", "")); rest = json.dumps(b["messages"])
                    else:
                        sysb = json.dumps([m for m in b["messages"] if m["role"] == "system"]); rest = json.dumps([m for m in b["messages"] if m["role"] != "system"])
                    assert sysb.count("<reminder>Before EVERY tool call") == 1 and "<reminder>" not in rest and "sys" in sysb, (sysb[:300], rest[:200])
                logged = [json.loads(l) for l in open(os.path.join(tmp, "calls.jsonl"))]
                assert all(x.get("reminder") is True and x.get("reminder_place") == "system" for x in logged), logged[:1]
                rows.append(f"reminder place system [{fmt}, last message {'tool' if toolmsg else 'user'}]: once in the system prompt of 8/8 requests, in no other message; the caller's messages unchanged")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(prefix="teach_fake_")       # REMINDER=rule: the untagged rule, at the end of the system prompt
    try:
        teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
        res, banner_u = _one("effort:high", "anthropic", tmp, {"CEIL_REMINDER": "rule", "CEIL_REMINDER_PLACE": "system", "FAKE_TOOLMSG": "1"})
        B = [x["body"] for x in bodies(tmp)]
        assert len(B) == 8 and "on [rule]" in banner_u and "place system" in banner_u, banner_u
        for b in B:
            sysb = json.dumps(b.get("system", ""), ensure_ascii=False); rest = json.dumps(b["messages"], ensure_ascii=False)
            assert sysb.count("Reason Before EVERY Tool Call") == 1 and "reminder" not in sysb.lower() and "Reason Before" not in rest, sysb[:300]
        rows.append("reminder rule, place system [anthropic]: the untagged rule once in the system prompt of 8/8 requests; the word 'reminder' in 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for fmt in ("anthropic", "openai"):                # after the result of submit_patch: no reminder (nothing is called after it)
        tmp = tempfile.mkdtemp(prefix="teach_fake_")
        try:
            teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
            res, _ = _one("effort:high", fmt, tmp, {"CEIL_REMINDER": "1", "FAKE_TOOLMSG": "submit"})
            B = [x["body"] for x in bodies(tmp)]
            assert B and "<reminder>" not in json.dumps(B), "the reminder followed a submit_patch result"
            rows.append(f"reminder [{fmt}, last message = the result of submit_patch]: in 0 of {len(B)} requests")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(prefix="teach_fake_")       # off by default: no request carries it
    try:
        teach_ledger.Ledger(os.path.join(tmp, "ledger")).init(1.0, [1.0, 0.1, 4.0])
        res, b0 = _one("effort:high", "anthropic", tmp, {"FAKE_TOOLMSG": "1"})
        assert "<reminder>" not in json.dumps([x["body"] for x in bodies(tmp)]) and "reminder off" in b0
        rows.append("reminder unset: in 0 requests")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the gate: a budget of 3 calls stops the 4th
    tmp = tempfile.mkdtemp(prefix="teach_fake_")
    try:
        L = teach_ledger.Ledger(os.path.join(tmp, "ledger")); L.init(0.005, [1.0, 0.1, 4.0])          # $0.00188 a call: spent passes the budget with call 3
        res, _ = _one("off", "anthropic", tmp)
        n = len(bodies(tmp))
        assert res["gated"] and res["calls"] == n == 3 and L.status()["calls"] == 3, (res, n, L.status())
        rows.append(f"gate: budget $0.005 at $0.00188 a call → {n} requests reached the server, call 4 raised BudgetReached before any request")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("teach_fake selftest (adapter + ledger through litellm, stand-in server):")
    for r in rows:
        print("   " + r)
    print("   " + banner)
    print("teach_fake selftest PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "selftest"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default="runs/teach/fake")
    ap.add_argument("--script", choices=["submit", "loop"], default="submit")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    srv = start(a.dir, a.port, a.script)
    print(f"teach_fake: http://127.0.0.1:{srv.server_address[1]} · script {a.script} · requests → {a.dir}/requests.jsonl (Ctrl-C to stop)", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
