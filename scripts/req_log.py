#!/usr/bin/env python3
"""B-9 part 2 step 0: what does Gemma see at call k when mini-swe-agent talks to our vLLM? (stdlib only)

    python3 scripts/req_log.py proxy --listen 8111 --upstream http://127.0.0.1:8011 --out runs/verified/probe/requests
    python3 scripts/req_log.py summarize --dir runs/verified/probe/requests --upstream http://127.0.0.1:8011 \
        --model gemma-4-31b-it-qat-w4a16-ct --out runs/verified/probe

proxy: a pass-through HTTP server (point mini's api_base at it); every POST …/chat/completions is saved with its response
as <out>/req_<n>.json; everything else is forwarded untouched. summarize: groups the saved requests into conversations
(same first user message), sends each request's messages + tools to vLLM's /tokenize and /detokenize (the server's own
chat template and defaults: what the model was fed) and writes
  <out>/server_prompts/<conv>.json  {calls: [{call, prompt, rendered_tokens, usage_prompt}]}   (render.py --check)
  <out>/records/<conv>.json         call-level records whose input_messages are exactly what was sent, targets = replies
  <out>/probe_summary.json + one printed line: earlier reasoning sent? rendered? multi-call / text+call turns, tokens.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def cmd_proxy(a):
    os.makedirs(a.out, exist_ok=True)
    lock, counter = threading.Lock(), [len(glob.glob(os.path.join(a.out, "req_*.json")))]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _fwd(self, method):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
            hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
            req = urllib.request.Request(a.upstream + self.path, data=body, headers=hdr, method=method)
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    status, rh, data = r.status, dict(r.headers), r.read()
            except urllib.error.HTTPError as e:
                status, rh, data = e.code, dict(e.headers), e.read()
            if method == "POST" and self.path.rstrip("/").endswith("chat/completions"):
                with lock:
                    n = counter[0]; counter[0] += 1
                try:
                    rec = {"n": n, "path": self.path, "status": status, "request": json.loads(body or b"{}"),
                           "response": json.loads(data or b"{}")}
                except ValueError:
                    rec = {"n": n, "path": self.path, "status": status, "raw_request": (body or b"").decode(errors="replace")[:2000]}
                json.dump(rec, open(os.path.join(a.out, f"req_{n:05d}.json"), "w"))
            self.send_response(status)
            for k, v in rh.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._fwd("GET")

        def do_POST(self):
            self._fwd("POST")

    print(f"req_log proxy · listen 127.0.0.1:{a.listen} → {a.upstream} · saving chat requests to {a.out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.listen), H).serve_forever()


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def first_user(msgs):
    c = next((m.get("content") for m in msgs if m.get("role") == "user"), "") or ""
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def cmd_summarize(a):
    logs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "req_*.json")))]
    logs = [x for x in logs if "request" in x]
    convs = collections.defaultdict(list)
    for x in logs:
        convs[hashlib.sha256(first_user(x["request"].get("messages") or []).encode()).hexdigest()[:16]].append(x)
    os.makedirs(os.path.join(a.out, "server_prompts"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "records"), exist_ok=True)
    S = collections.Counter()
    keys = collections.defaultdict(set)
    toks, examples = [], {}
    for ci, (sha, xs) in enumerate(sorted(convs.items())):
        xs.sort(key=lambda x: len(x["request"].get("messages") or []))
        cid = f"conv{ci:02d}"
        sp, calls = [], []
        for k, x in enumerate(xs, 1):
            rq = x["request"]
            msgs, tools = rq.get("messages") or [], rq.get("tools")
            for m in msgs:
                keys[m.get("role")] |= set(m)
            prev = [m for m in msgs if m.get("role") == "assistant"]
            with_r = [m for m in prev if (m.get("reasoning_content") or m.get("reasoning") or "").strip()]
            S["assistant msgs in requests"] += len(prev); S["... carrying reasoning"] += len(with_r)
            S["multi-call turns in history"] += sum(1 for m in prev if len(m.get("tool_calls") or []) > 1)
            S["text+call turns in history"] += sum(1 for m in prev if (m.get("content") or "").strip() and m.get("tool_calls"))
            t = post(a.upstream + "/tokenize", {"model": a.model, "messages": msgs, **({"tools": tools} if tools else {}),
                                                "add_generation_prompt": True})
            ids = t.get("tokens") or []
            prompt = post(a.upstream + "/detokenize", {"model": a.model, "tokens": ids}).get("prompt", "")
            # thought blocks of EARLIER turns in the prompt (the open one at the end of the generation prompt excluded)
            body = prompt[:-len("<|channel>thought\n")] if prompt.endswith("<|channel>thought\n") else prompt
            S["earlier thought blocks rendered"] += body.count("<|channel>thought")
            usage = ((x.get("response") or {}).get("usage") or {}).get("prompt_tokens")
            S["usage == tokenize"] += usage == len(ids); S["calls"] += 1
            toks.append(len(ids))
            sp.append({"call": k, "prompt": prompt, "rendered_tokens": len(ids), "usage_prompt": usage})
            if "multi" not in examples and any(len(m.get("tool_calls") or []) > 1 for m in prev):
                examples["multi"] = prompt[-3000:]
            msg = (((x.get("response") or {}).get("choices") or [{}])[0].get("message") or {})
            tgt = {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "content": msg.get("content") or "",
                   "tool_calls": [{"name": tc["function"]["name"], "args": tc["function"].get("arguments") or "{}",
                                   "id": tc.get("id")} for tc in (msg.get("tool_calls") or [])]}
            calls.append({"call": k, "segment": max(0, sum(1 for m in msgs if m.get("role") == "user") - 1),
                          "input_messages": msgs, "target": tgt})
        json.dump({"calls": sp}, open(os.path.join(a.out, "server_prompts", f"{cid}.json"), "w"))
        json.dump({"instance_id": cid, "first_user_sha": sha, "source": "probe", "tools": xs[0]["request"].get("tools"),
                   "calls": calls}, open(os.path.join(a.out, "records", f"{cid}.json"), "w"))
    toks.sort()
    sent = "yes" if S["... carrying reasoning"] else "no"
    rend = "yes" if S["earlier thought blocks rendered"] else "no"
    rule = "keep" if sent == "yes" else "drop"           # records reproduce what was SENT (verify); the server decides what is rendered
    S["server renders earlier thoughts"] = int(rend == "yes")
    summ = {"conversations": len(convs), **dict(S), "message keys per role": {k: sorted(v) for k, v in keys.items()},
            "tokens p50": toks[len(toks) // 2] if toks else None, "tokens max": toks[-1] if toks else None,
            "history_rule": rule, "multi_call_prompt_tail": examples.get("multi")}
    json.dump(summ, open(os.path.join(a.out, "probe_summary.json"), "w"), indent=1)
    print(f"== probe: {len(convs)} conversations · {S['calls']} calls · earlier reasoning sent: {sent} "
          f"({S['... carrying reasoning']} of {S['assistant msgs in requests']} assistant msgs) · rendered by the template: {rend} "
          f"({S['earlier thought blocks rendered']} blocks) → wire rule {rule}; {'one sequence per trajectory' if rend == 'yes' else 'one training sample per call (the server drops earlier thoughts)'} · multi-call turns {S['multi-call turns in history']} "
          f"· text+call turns {S['text+call turns in history']} · tokens p50 {summ['tokens p50']} max {summ['tokens max']} "
          f"· server usage = /tokenize {S['usage == tokenize']}/{S['calls']}")
    print("   message keys: " + " · ".join(f"{k}: {','.join(v)}" for k, v in summ["message keys per role"].items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy")
    p.add_argument("--listen", type=int, required=True); p.add_argument("--upstream", required=True); p.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True); s.add_argument("--upstream", required=True)
    s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"proxy": cmd_proxy, "summarize": cmd_summarize}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
