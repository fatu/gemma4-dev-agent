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
                           "content": [{"type": "thinking", "thinking": thought, "signature": "fake"}, {"type": "text", "text": "Submitted."}],
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
                       "content": [{"type": "thinking", "thinking": thought, "signature": "fake"},
                                   {"type": "tool_use", "id": f"toolu_fake_{n['calls']}", "name": name, "input": args}],
                       "stop_reason": "tool_use", "stop_sequence": None, "usage": dict(USAGE)}
            else:
                out = {"id": f"chatcmpl-fake-{n['calls']}", "object": "chat.completion", "created": 0, "model": body.get("model"),
                       "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                           "role": "assistant", "content": None, "reasoning_content": thought,
                           "tool_calls": [{"id": f"call_fake_{n['calls']}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}]}}],
                       "usage": {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350,
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


def _one(think, fmt, tmp):
    """a fresh interpreter per CEIL_THINK form (the adapter reads its knobs at import)"""
    import subprocess
    code = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import teach_fake, teach_ledger
srv = teach_fake.start(os.environ["FAKE_DIR"]); port = srv.server_address[1]
import ceil_sitecustomize, litellm
model = ("anthropic/" if os.environ["FAKE_FMT"] == "anthropic" else "openai/") + "stand-in-teacher"
base = f"http://127.0.0.1:{port}" + ("" if os.environ["FAKE_FMT"] == "anthropic" else "/v1")
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
out = {"calls": 0, "reasoning": 0, "gated": False}
try:
    for i in range(8):
        r = litellm.completion(model=model, api_base=base, api_key="not-a-key", messages=msgs, tools=teach_fake.TOOLS,
                               temperature=0.2, top_p=0.95, max_tokens=512,
                               extra_body={"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096})
        m = r.choices[0].message; out["calls"] += 1
        out["reasoning"] += bool(getattr(m, "reasoning_content", None))
        out["tool"] = m.tool_calls[0].function.name
except teach_ledger.BudgetReached:
    out["gated"] = True
print("RESULT " + json.dumps(out))
'''
    env = dict(os.environ, FAKE_DIR=tmp, FAKE_FMT=fmt, CEIL_TEMP="0.7", CEIL_THINK=think, CEIL_GEMMA_CTX="0",
               TEACH_LEDGER_DIR=os.path.join(tmp, "ledger"), TEACH_ATTEMPT="2", CEIL_CALL_LOG=os.path.join(tmp, "calls.jsonl"))
    if fmt == "openai":
        env["CEIL_REMOTE"] = "1"
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise SystemExit(f"selftest child failed ({think}, {fmt}):\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}")
    banner = next((l for l in p.stdout.splitlines() if l.startswith("ceil adapter:")), "")
    return json.loads(line[7:]), banner


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
