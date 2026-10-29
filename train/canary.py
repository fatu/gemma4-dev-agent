#!/usr/bin/env python3
"""A6 canary: a few LoRA steps on real rendered samples → PEFT export → tensor names/shapes checked against the
competition checkpoint's modules. The vLLM load / logprob / real-task half is train/lora_probe.py + eval_dev.sh.

    .venv-train/bin/python train/canary.py --base /vol/hf/dequant/gemma-4-E4B-dq --ct google/gemma-4-E4B-it-qat-w4a16-ct \
        --data runs/sft/d3_train.rendered.jsonl --steps 20 --max-len 8192 --out adapters/canary

Training: bf16 base (train/dequant_ct.py), gradient checkpointing, LoRA r32 α32 dropout 0 on the language model's
q/k/v/o/gate/up/down projections only (targets listed by full module path, built from the modules that exist in the
checkpoint), AdamW lr 1e-4, one sequence per step, loss on `labels` (render.py's masks). Every step prints loss,
tokens, seconds, tokens/s, peak memory → trainer_log.jsonl. Export: adapter_model.safetensors + adapter_config.json in
the scorer's sample schema (target_modules as the 7 short names, base_model_name_or_path = the alias). Check: every
tensor name follows the sample adapter's pattern `base_model.model.model.language_model.layers.N.<mod>.lora_{A,B}.weight`,
count = 2 × modules, shapes (r, in) / (out, r) match the base; prints PASS/FAIL.
"""
import argparse
import json
import os
import re
import sys
import time

import torch

MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAMPLE_KEYS = ["alpha_pattern", "auto_mapping", "base_model_name_or_path", "bias", "fan_in_fan_out", "inference_mode",
               "init_lora_weights", "layer_replication", "layers_pattern", "layers_to_transform", "loftq_config",
               "lora_alpha", "lora_dropout", "megatron_config", "megatron_core", "modules_to_save", "peft_type", "r",
               "rank_pattern", "revision", "target_modules", "task_type", "use_dora", "use_rslora"]


def ct_modules(ct):
    """module paths of the quantised language-model projections in the competition checkpoint"""
    import glob
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    src = snapshot_download(ct)
    mods = set()
    for sh in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(sh, "pt") as f:
            for k in f.keys():
                m = re.match(r"(model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(\w+_proj))\.weight(_packed)?$", k)
                if m and m.group(3) in MODS:
                    mods.add(m.group(1))
    return sorted(mods, key=lambda s: (int(s.split(".")[3]), MODS.index(s.split(".")[-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="bf16 copy from train/dequant_ct.py")
    ap.add_argument("--ct", default="google/gemma-4-E4B-it-qat-w4a16-ct", help="competition checkpoint (module list)")
    ap.add_argument("--alias", default="gemma-4-e4b-it-qat-w4a16-ct")
    ap.add_argument("--data", required=True, help="render.py jsonl (input_ids, labels)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="adapters/canary")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    import transformers
    from transformers import AutoModelForImageTextToText
    cls = getattr(transformers, "Gemma4ForConditionalGeneration", AutoModelForImageTextToText)
    t0 = time.time()
    dev = "cuda"
    print(f"canary · base {a.base} · ct {a.ct} · r {a.r} α {a.alpha} lr {a.lr} · steps {a.steps} · max_len {a.max_len} · out {a.out}")
    targets = ct_modules(a.ct)
    layers = sorted({int(t.split(".")[3]) for t in targets})
    per = {}
    for t in targets:
        per[t.split(".")[-1]] = per.get(t.split(".")[-1], 0) + 1
    print(f"  target modules from the checkpoint: {len(targets)} in layers {layers[0]}..{layers[-1]} ({len(layers)}) · per name {per}")
    model = cls.from_pretrained(a.base, dtype=torch.bfloat16, device_map=dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=targets)
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  model loaded {time.time() - t0:.0f} s · trainable {n_train / 1e6:.1f} M of {n_all / 1e9:.2f} B · "
          f"mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    rows = []
    for line in open(a.data):
        r = json.loads(line)
        if len(r["input_ids"]) <= a.max_len and any(x != -100 for x in r["labels"]):
            rows.append(r)
        if len(rows) >= a.steps:
            break
    print(f"  data: {len(rows)} sequences ≤ {a.max_len} tokens from {a.data}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    model.train()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "trainer_log.jsonl"), "w")
    tot_tok, tot_s, losses = 0, 0.0, []
    for step, r in enumerate(rows, 1):
        ids = torch.tensor([r["input_ids"]], device=dev)
        labels = torch.tensor([r["labels"]], device=dev)
        torch.cuda.synchronize(); ts = time.time()
        # loss only at label positions: the full-sequence logits (262K vocab × T) alone are 4 GB bf16 at T = 8192,
        # while ~2% of the positions carry labels. Hidden state t predicts labels[t+1]; `logits_to_keep` takes an index
        # tensor and the model applies its final-logit softcapping to exactly those rows.
        tgt = labels[0, 1:]
        keep = (tgt != -100).nonzero(as_tuple=True)[0]
        out = model(input_ids=ids, logits_to_keep=keep)
        loss = torch.nn.functional.cross_entropy(out.logits[0].float(), tgt[keep])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts
        n_lab = int((labels != -100).sum())
        rec = {"step": step, "loss": round(loss.item(), 4), "seq_tokens": ids.shape[1], "label_tokens": n_lab,
               "seconds": round(dt, 2), "tokens_per_s": round(ids.shape[1] / dt), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log.write(json.dumps(rec) + "\n"); losses.append(rec["loss"]); tot_tok += ids.shape[1]; tot_s += dt
        print(f"  step {step:>2} loss {rec['loss']:.4f} · {rec['seq_tokens']} tok ({n_lab} labels) · {dt:.1f} s · "
              f"{rec['tokens_per_s']} tok/s · peak {rec['max_mem_gb']} GB")
    log.close()
    print(f"  train: {len(rows)} steps · {tot_tok} tokens · {tot_s:.0f} s · {tot_tok / max(tot_s, 1e-9):.0f} tok/s · "
          f"loss first {losses[0]:.3f} → last {losses[-1]:.3f} (mean of last 3 {sum(losses[-3:]) / len(losses[-3:]):.3f}) · "
          f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    model.save_pretrained(a.out, safe_serialization=True)
    # adapter_config.json in the scorer's sample schema
    sample = json.load(open("harness/sample_submission/adapters/main_lora/adapter_config.json"))
    mine = json.load(open(os.path.join(a.out, "adapter_config.json")))
    clean = {k: mine.get(k, sample.get(k)) for k in SAMPLE_KEYS}
    clean.update({"base_model_name_or_path": f"google/{a.alias}", "target_modules": MODS, "inference_mode": True,
                  "layers_to_transform": None, "r": a.r, "lora_alpha": a.alpha, "lora_dropout": 0.0})
    json.dump(clean, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=2)
    # tensor names / shapes vs the base
    from safetensors import safe_open
    shapes = {}
    for name, mod in model.base_model.model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod.base_layer, "in_features"):
            shapes[name] = (mod.base_layer.in_features, mod.base_layer.out_features)
    ok, bad, n = True, [], 0
    with safe_open(os.path.join(a.out, "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            n += 1
            m = re.match(r"base_model\.model\.(model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_([AB])\.weight$", k)
            if not m:
                bad.append(f"unexpected name {k}"); continue
            mod, ab = m.group(1), m.group(2)
            shp = tuple(f.get_slice(k).get_shape())
            inn, out = shapes.get(mod, (None, None))
            want = (a.r, inn) if ab == "A" else (out, a.r)
            if shp != want:
                bad.append(f"{k}: shape {shp} vs expected {want}")
    expect = 2 * len(targets)
    if n != expect:
        bad.append(f"tensor count {n} vs expected {expect}")
    size = os.path.getsize(os.path.join(a.out, "adapter_model.safetensors"))
    print(f"  export: {n} tensors · {size / 1e6:.0f} MB · first key {keys[0]}")
    print(f"  sample adapter pattern: base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight")
    for b in bad[:10]:
        print("  BAD " + b)
    print(f"canary export check: {'PASS' if not bad else 'FAIL'} · {n} tensors, {len(targets)} modules, r {a.r}, "
          f"config target_modules {MODS} · {time.time() - t0:.0f} s")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
