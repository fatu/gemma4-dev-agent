"""PEFT adapter export in the scorer's sample schema + the tensor name/shape check (shared by canary.py and sft_lora.py)."""
import json
import os
import re

MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAMPLE_KEYS = ["alpha_pattern", "auto_mapping", "base_model_name_or_path", "bias", "fan_in_fan_out", "inference_mode",
               "init_lora_weights", "layer_replication", "layers_pattern", "layers_to_transform", "loftq_config",
               "lora_alpha", "lora_dropout", "megatron_config", "megatron_core", "modules_to_save", "peft_type", "r",
               "rank_pattern", "revision", "target_modules", "task_type", "use_dora", "use_rslora"]
SAMPLE_CFG = "harness/sample_submission/adapters/main_lora/adapter_config.json"


def ct_modules(ct, mods=MODS):
    """module paths of the quantised language-model projections present in the competition checkpoint (the 18
    KV-shared layers of E4B have no k_proj / v_proj: 258 modules, not 294)"""
    import glob
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    src = snapshot_download(ct)
    found = set()
    for sh in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(sh, "pt") as f:
            for k in f.keys():
                m = re.match(r"(model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(\w+_proj))\.weight(_packed)?$", k)
                if m and m.group(3) in mods:
                    found.add(m.group(1))
    return sorted(found, key=lambda s: (int(s.split(".")[3]), mods.index(s.split(".")[-1])))


def export_and_check(peft_model, out, alias, r, alpha, targets, keep=("trainer_log.jsonl",)):
    """save the adapter, rewrite adapter_config.json like the sample's, check every tensor name and shape.
    Returns (n_tensors, bytes, problems)."""
    from safetensors import safe_open
    os.makedirs(out, exist_ok=True)
    peft_model.save_pretrained(out, safe_serialization=True)
    for f in os.listdir(out):                       # PEFT writes a README; the scorer allows .json/.safetensors only
        if f not in ("adapter_config.json", "adapter_model.safetensors") and f not in keep:
            os.remove(os.path.join(out, f))
    sample = json.load(open(SAMPLE_CFG)) if os.path.exists(SAMPLE_CFG) else {}
    mine = json.load(open(os.path.join(out, "adapter_config.json")))
    clean = {k: mine.get(k, sample.get(k)) for k in SAMPLE_KEYS}
    clean.update({"base_model_name_or_path": f"google/{alias}", "target_modules": MODS, "inference_mode": True,
                  "layers_to_transform": None, "r": r, "lora_alpha": alpha, "lora_dropout": 0.0})
    json.dump(clean, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    shapes = {}
    for name, mod in peft_model.base_model.model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod.base_layer, "in_features"):
            shapes[name] = (mod.base_layer.in_features, mod.base_layer.out_features)
    problems, n = [], 0
    path = os.path.join(out, "adapter_model.safetensors")
    with safe_open(path, "pt") as f:
        for k in f.keys():
            n += 1
            m = re.match(r"base_model\.model\.(model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_([AB])\.weight$", k)
            if not m:
                problems.append(f"unexpected name {k}"); continue
            inn, outf = shapes.get(m.group(1), (None, None))
            want = (r, inn) if m.group(2) == "A" else (outf, r)
            shp = tuple(f.get_slice(k).get_shape())
            if shp != want:
                problems.append(f"{k}: shape {shp} vs {want}")
    if n != 2 * len(targets):
        problems.append(f"tensor count {n} vs expected {2 * len(targets)}")
    return n, os.path.getsize(path), problems
