#!/usr/bin/env python3
"""A1 · environment fingerprint: everything that decides what the model sees, as versions + sha256, in one JSON.

    python3 train/env_fingerprint.py --out runs/sft/env_fingerprint.json            # this machine / venv
    python3 train/env_fingerprint.py --compare runs/sft/env_fingerprint_pod.json    # list the differences

Two levels: `model_visible` (tokenizer.json, chat_template.jinja, generation/config, the 9 tool schemas, the latest
system prompt, sampling + eval config, the compaction values, the vLLM / swegemma / adk / transformers versions) is
what a training sample must match the scorer on; its canonical JSON is hashed into `fingerprint`, the value written
into every record's `env_fingerprint`. `environment` (every wheel in harness/wheelhouse with size + sha256, the
installed package versions, python, HARNESS_README) is the wider record for diffing two machines. Every value is
printed; a difference in `model_visible` means the data and the scorer drifted apart and must be explained.
"""
import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json",
               "special_tokens_map.json"]
PACKAGES = ["vllm", "transformers", "tokenizers", "swegemma", "adk-submission", "adk-eval-core", "google-adk", "torch",
            "litellm", "openai", "peft"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def model_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, local_files_only=True, allow_patterns=MODEL_FILES)


def compaction_values():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import compaction_shim
        return {"on": compaction_shim.ON, **compaction_shim.VALUES}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def build(a):
    mdir = model_dir(a.model)
    model = {f: sha(os.path.join(mdir, f)) for f in MODEL_FILES if os.path.exists(os.path.join(mdir, f))}
    tools = None
    if a.tools_from and os.path.exists(a.tools_from):
        rec = json.load(open(a.tools_from))
        tools = rec.get("tools") or (rec.get("setup") or {}).get("tool_schemas")
    latest = os.path.join(ROOT, "scripts", "arms", "latest")
    files = {p: sha(os.path.join(latest, p)) for p in ("prompts/system.md", "configs/sampling.yaml", "eval_config.yaml")
             if os.path.exists(os.path.join(latest, p))}
    wheels = {}
    for w in sorted(glob.glob(os.path.join(ROOT, "harness", "wheelhouse", "*.whl"))):
        name = os.path.basename(w)
        m = re.match(r"([A-Za-z0-9_.]+?)-(\d[^-]*)-", name)
        wheels[name] = {"dist": m.group(1) if m else None, "version": m.group(2) if m else None,
                        "size": os.path.getsize(w), "sha256": sha(w)}
    key_wheels = {w["dist"].lower().replace("_", "-"): w["version"] for w in wheels.values() if w["dist"]}
    installed = {p: ver(p) for p in PACKAGES}
    m = re.search(r"models--([^/]+)--([^/]+)", mdir)        # a snapshot path → the hub id, so two machines compare equal
    model_id = f"{m.group(1)}/{m.group(2)}" if m else a.model
    model_visible = {
        "model": model_id, "model_files": model,
        "tool_schemas_sha256": sha_text(json.dumps(tools, sort_keys=True)) if tools else None,
        "tool_names": [t["function"]["name"] for t in tools] if tools else None,
        "latest_setting": files,
        "compaction": compaction_values(),
        "scorer_versions": {k: key_wheels.get(k) for k in ("vllm", "swegemma", "adk-submission", "adk-eval-core",
                                                            "google-adk", "transformers", "tokenizers")},
    }
    fp = sha_text(json.dumps(model_visible, sort_keys=True))
    readme = os.path.join(ROOT, "harness", "HARNESS_README.md")
    return {"fingerprint": fp, "model_visible": model_visible,
            "environment": {"host": platform.node(), "python": platform.python_version(), "installed": installed,
                            "wheelhouse": wheels, "harness_readme_sha256": sha(readme) if os.path.exists(readme) else None,
                            "model_dir": mdir}}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-w4a16-ct", help="HF id (local cache) or a snapshot dir")
    ap.add_argument("--tools-from", default=None, help="a record or traj log carrying the tool schemas")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="another fingerprint json: print the differences")
    a = ap.parse_args()
    if not a.tools_from:
        cands = sorted(glob.glob(os.path.join(ROOT, "runs", "sft", "e4b_w7", "*.json")))
        a.tools_from = cands[0] if cands else None
    fp = build(a)
    mv = fp["model_visible"]
    print(f"env_fingerprint {fp['fingerprint'][:16]} · {fp['environment']['host']} · python {fp['environment']['python']}")
    print(f"  model {mv['model']} · files " + " ".join(f"{k} {v[:8]}" for k, v in mv["model_files"].items()))
    print(f"  tools {len(mv['tool_names'] or [])} sha {(mv['tool_schemas_sha256'] or '-')[:8]} (from {a.tools_from})")
    print("  latest " + " ".join(f"{k} {v[:8]}" for k, v in mv["latest_setting"].items()))
    print("  compaction " + " ".join(f"{k} {v}" for k, v in mv["compaction"].items()))
    print("  scorer wheels " + " ".join(f"{k} {v}" for k, v in mv["scorer_versions"].items()))
    print("  installed " + " ".join(f"{k} {v}" for k, v in fp["environment"]["installed"].items() if v))
    print(f"  wheelhouse {len(fp['environment']['wheelhouse'])} wheels · readme {(fp['environment']['harness_readme_sha256'] or '-')[:8]}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(fp, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"  → {a.out}")
    if a.compare:
        other = json.load(open(a.compare))
        same = other["fingerprint"] == fp["fingerprint"]
        print(f"compare with {a.compare}: fingerprint {'SAME' if same else 'DIFFERENT'}")
        for level in ("model_visible", "environment"):
            x, y = flat(fp[level]), flat(other[level])
            diffs = [(k, x.get(k), y.get(k)) for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)
                     and not k.startswith(("host", "model_dir"))]
            print(f"  {level}: {len(diffs)} differences")
            for k, v1, v2 in diffs[:60]:
                print(f"    {k}: here {str(v1)[:40]} · other {str(v2)[:40]}")
        return 0 if same else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
