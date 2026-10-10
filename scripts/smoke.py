#!/usr/bin/env python3
"""One chat completion against the local vLLM server — W1's smoke test (no dependencies).

    python scripts/smoke.py                      # default: http://localhost:8000, model "gemma4"
Prints the reply, token counts and decode speed. 📷 this.
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:8000") + "/v1/chat/completions"
body = {"model": os.environ.get("MODEL_NAME", "gemma4"), "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                                  "iteratively, then one sentence on its time complexity."}]}
t0 = time.time()
req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.load(r)
dt = time.time() - t0
u = out["usage"]
print(out["choices"][0]["message"]["content"])
print(f"\n--- {out['model']} · prompt {u['prompt_tokens']} · completion {u['completion_tokens']} tokens · "
      f"{dt:.1f} s · {u['completion_tokens'] / dt:.1f} tok/s (incl. prefill)")
