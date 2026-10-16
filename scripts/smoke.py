#!/usr/bin/env python3
"""One chat completion against the local vLLM server — W1's smoke test (no dependencies).

    python scripts/smoke.py                      # default: http://localhost:8000, model "gemma4"
Prints the reply, token counts and decode speed. 📷 this.
Two requests: the first after a server start is cold; the second is the warm speed to compare between boxes.
With thinking on, the thought block comes first — max_tokens 1024 leaves room for the answer (200 gave content None).
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:8000") + "/v1/chat/completions"
body = {"model": os.environ.get("MODEL_NAME", "gemma4"), "temperature": 0, "max_tokens": int(os.environ.get("MAX_TOKENS", 1024)),
        "messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                                  "iteratively, then one sentence on its time complexity."}]}
for run in ("cold", "warm"):
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    dt = time.time() - t0
    u, ch = out["usage"], out["choices"][0]
    msg = ch["message"]
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if run == "warm":
        print(msg.get("content"))
    print(f"--- {run}: {out['model']} · prompt {u['prompt_tokens']} · completion {u['completion_tokens']} tokens "
          f"(thinking {len(think)} chars) · finish {ch.get('finish_reason')} · {dt:.1f} s · "
          f"{u['completion_tokens'] / dt:.1f} tok/s (incl. prefill)")
