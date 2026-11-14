#!/usr/bin/env python3
"""Does vLLM serve a trained LoRA the way it was trained? (B-9 part 2, after the E4B adapter collapsed on Verified)

    CUDA_VISIBLE_DEVICES=0 .venv-eval/bin/python scripts/lora_nll_check.py runs/sft/mc/r1k.holdout.rendered.jsonl \
        --adapter ep2=adapters/mc_gemma-4-E4B_r1k_lr1e-5/epoch2 --adapter ep1=adapters/mc_gemma-4-E4B_r1k_lr1e-5/epoch1 \
        --expect base=0.7917,ep1=0.5863,ep2=0.5661
    python3 scripts/lora_nll_check.py --selftest          # laptop: alignment + verdicts with a fake engine

1. NLL: vLLM 0.19.1 offline (the eval venv; the same engine as the server, --enable-lora with the scorer's limits:
   max_loras = number of adapters, max_lora_rank = the largest r rounded up) scores every rendered sequence once with
   the base and once per adapter: max_tokens 1, prompt_logprobs 1 → the logprob of every prompt token given the ones
   before it. Mean NLL over the label positions (labels != -100: the model's replies) = exactly what sft_lora.py's
   eval[...] lines measured with the bf16 copy in transformers. Printed next to --expect (those training numbers):
   base ≈ expected and adapter ≈ expected → vLLM serves the adapter as trained (the collapse is in free generation);
   adapter not below the base → the adapter does not act / acts wrongly in vLLM.
2. Generation: the prompt of the first call of the first --gen sequences (everything before its first label token),
   greedy, --gen-tokens tokens, base and each adapter → the first 400 characters and the most repeated sentence count.
Memory: prompt logprobs materialise [chunk, 262,144] fp32 logits per scheduled chunk, which vLLM's profiling does not
reserve (train/teacher_score.py) → --mbt 4096, --gpu-util 0.80. Needs one free card (stop the servers first).
"""
import argparse
import collections
import json
import math
import random
import re
import sys
import time

SIZES = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def label_positions(row):
    return [j for j, x in enumerate(row["labels"]) if x != -100 and j > 0]


def nll(row, plp):
    """plp: list (len = len(input_ids)) of None | {token_id: logprob-like}; plp[j] = distribution of token j | < j"""
    ids, s, n = row["input_ids"], 0.0, 0
    for j in label_positions(row):
        d = plp[j]
        v = d[ids[j]]
        s -= getattr(v, "logprob", v); n += 1
    return s, n


def most_repeated(text):
    sents = [x.strip() for x in re.split(r"[.!?\n]", text) if len(x.strip()) > 20]
    if not sents:
        return 0, ""
    k, v = collections.Counter(sents).most_common(1)[0]
    return v, k


class FakeEngine:
    """selftest: logprobs depend on (model, token) so the alignment and the verdicts can be checked"""
    def __init__(self, shift):
        self.shift = shift

    def score(self, rows, model):
        out = []
        for r in rows:
            plp = [None] + [{t: -(0.8 + self.shift.get(model, 0.0)) * (1 + (t % 3) / 10)} for t in r["input_ids"][1:]]
            out.append(plp)
        return out

    def generate(self, prompts, model, n):
        return ["Let me understand the issue first. " * (12 if model != "base" else 1)] * len(prompts)


class VllmEngine:
    def __init__(self, a, adapters):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        ranks = [json.load(open(f"{p}/adapter_config.json")).get("r") or 128 for _, p in adapters]
        top = max(ranks) if ranks else 0
        kw = {}
        if adapters:
            kw = dict(enable_lora=True, max_loras=len(adapters), max_lora_rank=next(s for s in SIZES if s >= top))
        print(f"  vLLM: model {a.model} · max_len {a.max_len} · mbt {a.mbt} · gpu_util {a.gpu_util} · lora {kw or 'off'}"
              f" · adapters {[(n, p, r) for (n, p), r in zip(adapters, ranks)]}", flush=True)
        self.llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util,
                       max_num_batched_tokens=a.mbt, enable_prefix_caching=False, **kw)
        self.req = {n: LoRARequest(n, i + 1, p) for i, (n, p) in enumerate(adapters)}
        self.tok = self.llm.get_tokenizer()

    def _run(self, prompts, sp, model):
        from vllm.inputs import TokensPrompt
        lr = self.req.get(model)
        return self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                                 lora_request=[lr] * len(prompts) if lr else None, use_tqdm=False)

    def score(self, rows, model):
        from vllm import SamplingParams
        outs = self._run([r["input_ids"] for r in rows], SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1,
                                                                         detokenize=False), model)
        return [o.prompt_logprobs for o in outs]

    def generate(self, prompts, model, n):
        from vllm import SamplingParams
        outs = self._run(prompts, SamplingParams(max_tokens=n, temperature=0, skip_special_tokens=False), model)
        return [o.outputs[0].text for o in outs]


def run(rows, engine, names, expect, gen, gen_tokens):
    res = {}
    for m in names:
        t = time.time()
        plps = engine.score(rows, m)
        per = [nll(r, p) for r, p in zip(rows, plps)]
        S, N = sum(s for s, _ in per), sum(n for _, n in per)
        res[m] = {"nll": S / max(N, 1), "per": [s / max(n, 1) for s, n in per], "tokens": N}
        e = expect.get(m)
        print(f"  {m:>6}: vLLM NLL {res[m]['nll']:.4f} over {N:,} label tokens ({len(rows)} sequences, {time.time() - t:.0f} s)"
              + (f" · training (transformers bf16) {e:.4f} · Δ {res[m]['nll'] - e:+.4f}" if e is not None else ""), flush=True)
    b = res["base"]
    for m in names[1:]:
        r = res[m]
        better = sum(1 for x, y in zip(r["per"], b["per"]) if x < y)
        print(f"  {m} − base: {r['nll'] - b['nll']:+.4f} (training: "
              + (f"{expect[m] - expect['base']:+.4f}" if m in expect and "base" in expect else "n/a")
              + f") · lower than base on {better} / {len(rows)} sequences")
        if r["nll"] >= b["nll"] - 0.01:
            v = "the adapter does NOT lower the NLL in vLLM → it does not act / acts wrongly when served (serving mismatch)"
        elif m in expect and abs(r["nll"] - expect[m]) <= 0.03 and abs(b["nll"] - expect.get("base", b["nll"])) <= 0.03:
            v = "vLLM serves the adapter as trained → the collapse comes from free generation, not from serving"
        else:
            v = "the adapter acts in vLLM but the NLL differs from training by > 0.03 → partly a serving difference; compare per sequence"
        print(f"  VERDICT {m}: {v}")
    for i, r in enumerate(rows[:gen]):
        lp = label_positions(r)
        if not lp:
            continue
        prompt = r["input_ids"][:lp[0]]
        print(f"-- generation {i + 1}: {r.get('instance_id', '?')} · prompt {len(prompt):,} tokens · greedy {gen_tokens} tokens")
        for m in names:
            txt = engine.generate([prompt], m, gen_tokens)[0]
            k, s = most_repeated(txt)
            print(f"  [{m}] {len(txt):,} chars · most repeated sentence ×{k}{(': ' + s[:90]) if k > 1 else ''}")
            print("     " + txt[:400].replace("\n", "⏎ "))
    return res


def selftest():
    random.seed(0)
    rows = []
    for k in range(3):
        ids = [2] + [random.randrange(10, 1000) for _ in range(50)]
        lab = [-100] * 20 + ids[20:35] + [-100] * 5 + ids[40:]
        rows.append({"instance_id": f"t{k}", "input_ids": ids, "labels": lab})
    exp = {"base": 0.88, "good": 0.68, "bad": 0.68}
    print("selftest: good = as trained, bad = higher than base")
    res = run(rows, FakeEngine({"good": -0.2, "bad": 0.3}), ["base", "good", "bad"], exp, 1, 32)
    n_lab = sum(len(label_positions(r)) for r in rows)
    assert res["base"]["tokens"] == n_lab == 3 * 26, res["base"]["tokens"]
    assert abs(res["good"]["nll"] - 0.6 * 1.1) < 0.1 and res["bad"]["nll"] > res["base"]["nll"]
    print("lora_nll_check selftest PASS (label positions, per-model NLL, verdicts, generation lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", help="render.py rows (input_ids, labels), e.g. runs/sft/mc/r1k.holdout.rendered.jsonl")
    ap.add_argument("--adapter", action="append", default=[], help="name=PEFT dir (repeatable)")
    ap.add_argument("--expect", default="", help="training NLLs, e.g. base=0.7917,ep2=0.5661")
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--mbt", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--gen", type=int, default=2, help="sequences whose first-call prompt is also generated from")
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.data:
        sys.exit("data file required")
    rows = [json.loads(l) for l in open(a.data)]
    adapters = [tuple(x.split("=", 1)) for x in a.adapter]
    expect = {k: float(v) for k, v in (x.split("=") for x in a.expect.split(",") if x)}
    print(f"lora_nll_check · data {a.data} ({len(rows)} sequences, {sum(len(r['input_ids']) for r in rows):,} tokens, "
          f"longest {max(len(r['input_ids']) for r in rows):,}) · expect {expect or 'none'} · gen {a.gen} × {a.gen_tokens} tokens", flush=True)
    run(rows, VllmEngine(a, adapters), ["base"] + [n for n, _ in adapters], expect, a.gen, a.gen_tokens)
    print("📷 every line from 'base: vLLM NLL' down")


if __name__ == "__main__":
    main()
