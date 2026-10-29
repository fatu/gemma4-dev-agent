#!/usr/bin/env python3
"""Does the scorer's vLLM apply a LoRA the way PEFT trained it? Score the SAME tokens on both sides and compare.

    .venv-train/bin/python train/lora_hf_probe.py --base /vol/hf/dequant/gemma-4-E4B-dq --adapter adapters/canary \
        --vllm runs/canary/probe.json --zero-out adapters/canary_zero

`--vllm` is train/lora_probe.py --save: the prompt token ids and, per served model (base alias, adapter), the greedy
continuation ids and their logprobs. Here the HF model (bf16 dequantised base) teacher-forces exactly those ids,
without the adapter (base) and with the PEFT adapter, and prints per position: vLLM logprob vs HF logprob for the
base and for the adapter, the mean |Δ| of each, and the mean |Δ| the adapter itself causes on both sides. The two
sides share the int4 values but not the kernels, so the base rows calibrate the numerical noise; the adapter rows
must sit in the same band, else serving and training disagree about the adapter. `--zero-out` also writes an
untrained adapter (PEFT init: lora_B = 0) with the trained one's config: served by vLLM it must reproduce the base
exactly (a loader-semantics check). Every number printed.
"""
import argparse
import json
import os
import shutil
import sys

import torch


def score(model, prompt_ids, cont_ids):
    ids = torch.tensor([prompt_ids + cont_ids], device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, logits_to_keep=len(cont_ids) + 1)
    logits = out.logits[0, :-1].float()                 # positions predicting cont_ids
    lp = torch.log_softmax(logits, -1)
    return [lp[i, t].item() for i, t in enumerate(cont_ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--zero-out", default=None)
    a = ap.parse_args()
    import transformers
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    cls = getattr(transformers, "Gemma4ForConditionalGeneration", AutoModelForImageTextToText)
    v = json.load(open(a.vllm))
    pid = v["prompt_ids"]
    rb, ra = v["results"][v["base"]], v["results"][v["adapter"]]
    print(f"lora_hf_probe · base {a.base} · adapter {a.adapter} · vllm probe {a.vllm} (prompt {len(pid)} tokens, "
          f"continuations base {len(rb['ids'])} / adapter {len(ra['ids'])} tokens)")
    model = cls.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    hf_base_on_base = score(model, pid, rb["ids"])
    hf_base_on_adapt = score(model, pid, ra["ids"])
    # the exported adapter_config.json lists the 7 short names (what the scorer's sample does; vLLM only reads the
    # tensor keys) — in HF those names would also match the vision tower's Gemma4ClippableLinear, so rebuild the
    # target list as the full module paths present in the adapter's tensors
    import re
    from safetensors import safe_open
    with safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt") as f:
        mods = sorted({re.sub(r"^base_model\.model\.", "", k).rsplit(".lora_", 1)[0] for k in f.keys()})
    cfg = LoraConfig.from_pretrained(a.adapter)
    cfg.target_modules = mods
    print(f"  adapter modules from the tensor keys: {len(mods)} (config lists {LoraConfig.from_pretrained(a.adapter).target_modules})")
    if a.zero_out:
        zero = get_peft_model(model, cfg)
        os.makedirs(a.zero_out, exist_ok=True)
        zero.save_pretrained(a.zero_out, safe_serialization=True)
        shutil.copy(os.path.join(a.adapter, "adapter_config.json"), os.path.join(a.zero_out, "adapter_config.json"))
        from safetensors import safe_open
        with safe_open(os.path.join(a.zero_out, "adapter_model.safetensors"), "pt") as f:
            nb = sum(1 for k in f.keys() if "lora_B" in k)
            zb = sum(1 for k in f.keys() if "lora_B" in k and float(f.get_tensor(k).abs().max()) == 0.0)
        print(f"  zero adapter → {a.zero_out}: {nb} lora_B tensors, {zb} all-zero")
        model = zero.unload()
    peft = PeftModel.from_pretrained(model, a.adapter, config=cfg)
    peft.eval()
    hf_adapt_on_adapt = score(peft, pid, ra["ids"])
    hf_adapt_on_base = score(peft, pid, rb["ids"])
    print("  base continuation (vLLM base vs HF base) and adapter continuation (vLLM adapter vs HF adapter):")
    print("  pos  tok(base)        vllm_b     hf_b    |Δ|   tok(adapter)     vllm_a     hf_a    |Δ|")
    db, da = [], []
    for i in range(min(len(rb["ids"]), len(ra["ids"]))):
        d1 = abs(rb["lp"][i] - hf_base_on_base[i]); d2 = abs(ra["lp"][i] - hf_adapt_on_adapt[i])
        db.append(d1); da.append(d2)
        if i < 12:
            print(f"  {i:>3}  {rb['tok'][i]!r:<15} {rb['lp'][i]:>8.4f} {hf_base_on_base[i]:>8.4f} {d1:>6.4f}   "
                  f"{ra['tok'][i]!r:<15} {ra['lp'][i]:>8.4f} {hf_adapt_on_adapt[i]:>8.4f} {d2:>6.4f}")
    mb, ma = sum(db) / len(db), sum(da) / len(da)
    # the adapter's own effect on the adapter continuation, HF side (vLLM scored each continuation with its own model)
    eff_hf = sum(abs(x - y) for x, y in zip(hf_adapt_on_adapt, hf_base_on_adapt)) / len(da)
    print(f"  mean |Δ| vLLM vs HF: base {mb:.4f} · adapter {ma:.4f} (max base {max(db):.4f} · adapter {max(da):.4f})")
    print(f"  adapter effect on the adapter continuation (HF, adapter vs base): mean |Δ logprob| {eff_hf:.4f}")
    verdict = "PASS" if ma <= max(3 * mb, 0.05) else "FAIL"
    print(f"RESULT {verdict}: serving-vs-training disagreement with the adapter ({ma:.4f}) "
          f"{'within' if verdict == 'PASS' else 'beyond'} 3× the base kernel noise ({mb:.4f}) or 0.05")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
