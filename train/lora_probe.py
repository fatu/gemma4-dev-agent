#!/usr/bin/env python3
"""A6/A7 on a running vLLM server: is the adapter loaded, does it change the forward pass, what did LoRA cost in KV?

    python3 train/lora_probe.py --adapter canary --log runs/vllm_canary.log

Lists /v1/models (base alias + adapter), sends the same greedy completion to both with logprobs, prints the top-1
logprobs side by side, the mean |Δ| and whether the texts differ (PASS when any logprob differs by > 1e-3: the adapter
is active, #749528). Then greps the vLLM log for the KV cache size and max concurrency (A7: KV ≥ 27K tokens with the
adapter) and the lora flags. Every number printed; exit 1 when the adapter is missing or inert.
"""
import argparse
import json
import re
import sys
import urllib.request

URL = "http://127.0.0.1:8000"
PROMPT = ("<bos><|turn>user\nYou are an autonomous software engineer. The repository test `test_parse_url` fails "
          "with `IndexError: list index out of range` in `urlparse.py` line 42. What is your first step?<turn|>\n"
          "<|turn>model\n")


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


_detok = {}


def tok_text(i):
    if i not in _detok:
        _detok[i] = post("/detokenize", {"model": "gemma4", "tokens": [i]})["prompt"]
    return _detok[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="canary")
    ap.add_argument("--log", default=None, help="vLLM log for the KV / lora lines")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--save", default=None, help="json with prompt ids + per-model continuation ids/logprobs (for train/lora_hf_probe.py)")
    a = ap.parse_args()
    models = [m["id"] for m in json.load(urllib.request.urlopen(URL + "/v1/models"))["data"]]
    base = [m for m in models if m.startswith("gemma-4-")]
    print(f"lora_probe · served models: {models}")
    if a.adapter not in models:
        print(f"RESULT FAIL: adapter {a.adapter!r} is not served"); return 1
    res = {}
    for m in (base[0], a.adapter):
        r = post("/v1/completions", {"model": m, "prompt": PROMPT, "max_tokens": a.max_tokens, "temperature": 0,
                                     "logprobs": 1, "skip_special_tokens": False, "return_tokens_as_token_ids": True})
        ch = r["choices"][0]
        ids = [int(t.split(":")[1]) for t in ch["logprobs"]["tokens"]]
        res[m] = {"text": ch["text"], "lp": ch["logprobs"]["token_logprobs"], "ids": ids,
                  "tok": [tok_text(i) for i in ids]}
        print(f"  {m}: {ch['text'][:100]!r}")
    b, c = res[base[0]], res[a.adapter]
    n = min(len(b["lp"]), len(c["lp"]))
    diffs = [abs((b["lp"][i] or 0) - (c["lp"][i] or 0)) for i in range(n)]
    same_text = b["text"] == c["text"]
    print("  pos  base_tok            base_lp     adapter_tok         adapter_lp   |Δ|")
    for i in range(min(n, 8)):
        print(f"  {i:>3}  {b['tok'][i]!r:<18} {b['lp'][i]:>9.4f}   {c['tok'][i]!r:<18} {c['lp'][i]:>9.4f}   {diffs[i]:.4f}")
    mean = sum(diffs) / max(n, 1)
    print(f"  compared {n} positions · mean |Δ logprob| {mean:.4f} · max {max(diffs) if diffs else 0:.4f} · text {'same' if same_text else 'differs'}")
    active = any(d > 1e-3 for d in diffs)
    if a.log:
        txt = open(a.log, errors="replace").read()
        for pat in (r"GPU KV cache size: [\d,]+ tokens", r"Maximum concurrency for [\d,]+ tokens per request: [\d.]+x",
                    r"Available KV cache memory: [\d.]+ GiB", r"lora: .*", r"max_lora_rank[^,}]*"):
            m = re.search(pat, txt)
            print(f"  log: {m.group(0) if m else pat + ' (not found)'}")
        kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        if kv:
            n_kv = int(kv.group(1).replace(",", ""))
            print(f"  A7 KV capacity with adapter: {n_kv:,} tokens → {'OK (≥ 27,000)' if n_kv >= 27000 else 'TOO SMALL'}")
    if a.save:
        prompt_ids = post("/tokenize", {"model": "gemma4", "prompt": PROMPT, "add_special_tokens": True})["tokens"]
        json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids, "base": base[0], "adapter": a.adapter, "results": res},
                  open(a.save, "w"), indent=1)
        print(f"  saved {a.save} (prompt {len(prompt_ids)} tokens)")
    print(f"RESULT {'PASS' if active else 'FAIL'}: adapter {'changes' if active else 'does not change'} the logprobs")
    return 0 if active else 1


if __name__ == "__main__":
    sys.exit(main())
