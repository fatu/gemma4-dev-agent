#!/usr/bin/env python3
"""Dequantise the competition checkpoint (compressed-tensors, int4 group-32 symmetric) into a bf16 copy for LoRA
training: PEFT cannot attach to the packed weights, and the scorer serves exactly these int4 values, so the bf16 base
the adapter is trained on is the served model, not the separate bf16 release.

    .venv-train/bin/python train/dequant_ct.py --model google/gemma-4-E4B-it-qat-w4a16-ct --out /vol/hf/dequant/gemma-4-E4B-dq

Two independent paths, cross-checked on sampled tensors: (1) transformers' compressed-tensors integration
(`CompressedTensorsConfig(run_compressed=False)` decompresses at load), (2) a manual unpack of `weight_packed` /
`weight_scale` / `weight_shape` with compressed_tensors' own `unpack_from_int32`. The saved copy drops the
quantization_config, keeps the tokenizer/template/processor files, and prints: tensors, params, bytes, max |Δ| of the
cross-check, and that no packed tensor is left. Everything printed is the record.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open

TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
             "processor_config.json", "special_tokens_map.json"]


def find_unpack():
    """compressed_tensors' unpack_from_int32, wherever the installed version keeps it; else our own (same layout:
    8 int4 per int32 along the last dim, element i in bits 4i..4i+3, stored unsigned with offset 8)"""
    import importlib
    import pkgutil
    import compressed_tensors
    for m in pkgutil.walk_packages(compressed_tensors.__path__, "compressed_tensors."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(mod, "unpack_from_int32"):
            return mod.unpack_from_int32, m.name

    def own(value, num_bits, shape, packed_dim=1):
        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1
        v = value.to(torch.int32)
        cols = v.shape[1] * pack_factor
        out = torch.zeros((v.shape[0], cols), dtype=torch.int32)
        for i in range(pack_factor):
            out[:, i::pack_factor] = (v >> (num_bits * i)) & mask
        out = out - (1 << (num_bits - 1))
        return out[: shape[0], : shape[1]].to(torch.int8)
    return own, "own implementation"


def manual_dequant(f, prefix):
    """prefix = '…q_proj' → bf16 weight from weight_packed/weight_scale/weight_shape (symmetric int4, group 32)"""
    unpack_from_int32, _ = find_unpack()
    packed = f.get_tensor(prefix + ".weight_packed")
    scale = f.get_tensor(prefix + ".weight_scale")
    shape = f.get_tensor(prefix + ".weight_shape").tolist()
    w = unpack_from_int32(packed, 4, torch.Size(shape))            # int8 in [-8, 7], (out, in)
    out, inn = shape
    g = inn // scale.shape[1]
    w = w.view(out, inn // g, g).to(torch.float32) * scale.view(out, inn // g, 1).to(torch.float32)
    return w.view(out, inn).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", type=int, default=6, help="quantised tensors cross-checked against the manual unpack")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    import transformers
    from transformers import AutoModelForImageTextToText, CompressedTensorsConfig
    cls = getattr(transformers, "Gemma4ForConditionalGeneration", AutoModelForImageTextToText)
    t0 = time.time()
    src = snapshot_download(a.model)
    print(f"dequant_ct · {a.model} · {src} · device {a.device}")
    model = cls.from_pretrained(
        src, dtype=torch.bfloat16, device_map=a.device,
        quantization_config=CompressedTensorsConfig(run_compressed=False))
    print(f"  loaded + decompressed in {time.time() - t0:.0f} s · class {type(model).__name__}")
    sd_all = model.state_dict()
    QKEYS = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx", "weight_global_scale")
    quant_mods = {k.rsplit(".", 1)[0] for k in sd_all if k.endswith(QKEYS)}
    missing_dense = [m for m in quant_mods if m + ".weight" not in sd_all]
    sd = {k: v for k, v in sd_all.items() if not k.endswith(QKEYS)}
    print(f"  quantised modules {len(quant_mods)} · dropped quant buffers {len(sd_all) - len(sd)} · modules without a dense weight {len(missing_dense)}")
    packed_left = [k for k in sd if k.endswith(QKEYS)] + missing_dense
    n_params = sum(v.numel() for v in sd.values())
    n_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    dtypes = {}
    for v in sd.values():
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
    print(f"  tensors {len(sd)} · params {n_params / 1e9:.3f} B · {n_bytes / 1e9:.2f} GB · dtypes {dtypes} · packed left {len(packed_left)}")
    # cross-check against the manual unpack on the first --check quantised Linear weights
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    checked, worst = 0, 0.0
    print(f"  manual unpack from {find_unpack()[1]}")
    for sh in shards:
        with safe_open(sh, "pt") as f:
            for k in f.keys():
                if not k.endswith(".weight_packed") or checked >= a.check:
                    continue
                prefix = k[: -len(".weight_packed")]
                manual = manual_dequant(f, prefix)
                key = prefix + ".weight"
                if key not in sd:
                    print(f"  cross-check: {key} not in the decompressed state dict (name mapping?)"); continue
                got = sd[key].to("cpu")
                d = (got.float() - manual.float()).abs().max().item()
                worst = max(worst, d); checked += 1
                print(f"  cross-check {key} {tuple(got.shape)} max|Δ| {d:.3e}")
    print(f"  cross-check: {checked} tensors · worst max|Δ| {worst:.3e} → {'OK' if checked and worst < 1e-2 else 'FAIL'}")
    os.makedirs(a.out, exist_ok=True)
    # write the dense state dict ourselves (save_pretrained would re-attach the quantizer): ≤ 5 GB shards + index
    from safetensors.torch import save_file
    shards, cur, cur_bytes = [], {}, 0
    for k, v in sd.items():
        n = v.numel() * v.element_size()
        if cur and cur_bytes + n > 5e9:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v.contiguous().cpu(); cur_bytes += n
    if cur:
        shards.append(cur)
    weight_map = {}
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(a.out, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in sh})
    json.dump({"metadata": {"total_size": n_bytes}, "weight_map": weight_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.loads(model.config.to_json_string())
    for k in [k for k in cfg if "quantization" in k]:
        cfg.pop(k)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"  wrote {len(shards)} shards · config without quantization_config")
    for tf in TOK_FILES:
        if os.path.exists(os.path.join(src, tf)):
            shutil.copy(os.path.join(src, tf), os.path.join(a.out, tf))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.out, "*")))
    print(f"  saved {a.out} · {size / 1e9:.2f} GB · files {sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.out, '*')))}")
    print(f"dequant_ct done in {time.time() - t0:.0f} s")
    return 0 if not packed_left and (not a.check or (checked and worst < 1e-2)) else 1


if __name__ == "__main__":
    sys.exit(main())
