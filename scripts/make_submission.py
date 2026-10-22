#!/usr/bin/env python3
"""Build + check one submission from an evaluated arm dir → dist/<name>/submission.zip (+ optional private-dataset upload).

    .venv-eval/bin/python scripts/make_submission.py runs/subs/nothink              # 📷 the check table
    UPLOAD=1 .venv-eval/bin/python scripts/make_submission.py runs/subs/nothink     # + new version of the private dataset
    ALIAS=gemma-4-e4b-it-qat-w4a16-ct DATASET=gemma4-submission-e4b UPLOAD=1 … runs/subs/<arm>   # small-model bonus entry
      (build the arm dir without an eval: BUILD_ONLY=1 MODEL_ALIAS=… ARMS=<arm> TAG=… bash scripts/eval_dev.sh)

Source = the arm dir eval_dev.sh evaluated (runs/subs/<arm>), so what we submit is what we measured. Two edits on top,
both submission-only (our checklist, course slides 26 + 32):
  nolora   — drop every `adapter:` line (agent.yaml + sub_agents/*.yaml) and adapters/: any adapter makes the scorer's
             vLLM reserve 8 × rank-128 LoRA slots and the KV cache falls ~46K → ~7.6K tokens (forum #744331).
             NOLORA=0 keeps them.
  budget   — eval_config.yaml evaluation = TIMEOUT 240 s / CALLS 28 / MINUTES 8 / TURNS 80 (the sample's 60/10/1/50 is why
             the organiser baseline scores ~0). Override with the env vars.
Then the course's 8 checks (slide 34) on the built dir, the zip, and again on an unzipped copy (round trip):
  1 agent.yaml at the root · 2 every YAML parses (SafeLoader + an !include constructor) · 3 every !include and
  config_path exists · 4 model = the base alias, no adapter left · 5 tools ⊆ the 9 harness tools (+ agent_tool) ·
  6 extensions whitelist, no hidden files / symlinks, ≤ 10,000 files · 7 zip < 3 GiB · 8 round trip
plus the harness's own adk_submission.discovery.validate_directory() on the unzipped copy.
Upload (UPLOAD=1): dist/<name>/ holds only submission.zip + dataset-metadata.json → kaggle datasets create (first time,
private) or version, id <KAGGLE_USER>/gemma4-submission. The Kaggle notebook (scripts/kaggle_notebook_cell.py) copies
that zip to /kaggle/working/submission.zip.
"""
import hashlib
import inspect
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALIAS = os.environ.get("ALIAS", "gemma-4-31b-it-qat-w4a16-ct")        # check 4; E4B/E2B for the small-model bonus
TOOLS = {"run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
DATASET = os.environ.get("DATASET", "gemma4-submission")                 # private dataset the notebook reads
results = []                                            # (check, ok, detail)


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    return ok


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("!include", lambda loader, node: {"__include__": loader.construct_scalar(node)})


def yaml_files(d):
    return sorted(p for p in d.rglob("*") if p.suffix in (".yaml", ".yml") and "adapters" not in p.parts)


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def run_checks(d, tag):
    """checks 1-6 on a submission dir; returns the parsed agent.yaml"""
    check(f"{tag} 1 agent.yaml at root", (d / "agent.yaml").is_file())
    parsed, bad = {}, []
    for y in yaml_files(d):
        try:
            parsed[y] = yaml.load(y.read_text(), Loader=Loader)
        except yaml.YAMLError as e:
            bad.append(f"{y.relative_to(d)}: {str(e).splitlines()[0]}")
    check(f"{tag} 2 YAML parses ({len(parsed)} files)", not bad, "; ".join(bad))
    missing = []
    for y, doc in parsed.items():
        for node in walk(doc):
            for k, v in node.items():
                ref = v["__include__"] if isinstance(v, dict) and "__include__" in v else v if k == "config_path" else None
                if isinstance(ref, str) and not ((d / ref).exists() or (y.parent / ref).exists()):
                    missing.append(f"{y.relative_to(d)} → {ref}")
    check(f"{tag} 3 includes + config_paths exist", not missing, "; ".join(missing))
    models, adapters, tools = set(), [], []
    for y, doc in parsed.items():
        if not isinstance(doc, dict) or "model" not in doc:
            continue                                     # sampling.yaml, eval_config.yaml
        models.add(doc["model"])
        if doc.get("adapter"):
            adapters.append(f"{y.relative_to(d)}: {doc['adapter']}")
        for t in doc.get("tools") or []:
            tools.append(t if isinstance(t, str) else next(iter(t)) if isinstance(t, dict) else repr(t))
    want_lora = os.environ.get("NOLORA", "1") == "0"
    check(f"{tag} 4 model = base alias" + ("" if want_lora else ", no adapter"),
          models == {ALIAS} and (want_lora or not adapters), f"models {sorted(models)} adapters {adapters}")
    extra = sorted(set(tools) - TOOLS - {"agent_tool"})
    check(f"{tag} 5 tools ⊆ the 9 (+agent_tool)", tools and not extra, f"not allowed: {extra}" if extra else f"{len(tools)} entries")
    files = [p for p in d.rglob("*") if p.is_file() or p.is_symlink()]
    odd = [str(p.relative_to(d)) for p in files
           if p.is_symlink() or p.suffix not in EXTS or any(s.startswith(".") for s in p.relative_to(d).parts)]
    check(f"{tag} 6 extensions / hidden / symlinks ({len(files)} files)", not odd and len(files) <= 10000, ", ".join(odd[:5]))
    return parsed.get(d / "agent.yaml") or {}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = pathlib.Path(sys.argv[1]).resolve()
    if not (src / "agent.yaml").is_file():
        sys.exit(f"{src} has no agent.yaml — pass an arm dir like runs/subs/nothink")
    name = os.environ.get("NAME", src.name)
    out = ROOT / "dist" / name
    shutil.rmtree(out, ignore_errors=True)
    sub = out / "build"
    shutil.copytree(src, sub, ignore=shutil.ignore_patterns(".*", "__pycache__", "*.pyc"))

    # nolora
    if os.environ.get("NOLORA", "1") != "0":
        shutil.rmtree(sub / "adapters", ignore_errors=True)
        for y in yaml_files(sub):
            t = y.read_text()
            t2 = re.sub(r"(?m)^adapter:.*\n", "", t)
            if t2 != t:
                y.write_text(t2)
    # budget
    ev = {"timeout_seconds": int(os.environ.get("TIMEOUT", 240)), "max_tool_calls": int(os.environ.get("CALLS", 28)),
          "max_time_minutes": int(os.environ.get("MINUTES", 8)), "max_turns": int(os.environ.get("TURNS", 80))}
    (sub / "eval_config.yaml").write_text(
        "# Per-task execution budgets and sandbox command timeouts (scripts/make_submission.py).\n"
        + yaml.safe_dump({"evaluation": ev}, sort_keys=False))

    run_checks(sub, "dir")
    zpath = out / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sub.rglob("*")):
            if p.is_file():                              # fixed timestamp → same content, same sha256 (card)
                zi = zipfile.ZipInfo(p.relative_to(sub).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
                z.writestr(zi, p.read_bytes())
    size = zpath.stat().st_size
    check("zip 7 size < 3 GiB", size < 3 * 2**30, f"{size / 2**20:.2f} MiB")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    check("zip agent.yaml at zip root", "agent.yaml" in names, f"first entries {names[:3]}")
    with tempfile.TemporaryDirectory() as t:
        rt = pathlib.Path(t)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(rt)
        before = len(results)
        run_checks(rt, "unzip")
        check("8 round trip: 1-6 green again", all(ok for _, ok, _ in results[before:]))
        try:
            from adk_submission.discovery import validate_directory
            sig = str(inspect.signature(validate_directory))
            r = validate_directory(rt)
            check("official validate_directory()", r is not False, f"{sig} → {str(r)[:80]}")
        except Exception as e:
            check("official validate_directory()", False, f"{type(e).__name__}: {str(e)[:160]}")
    shutil.rmtree(sub)

    s = yaml.safe_load((src / "configs" / "sampling.yaml").read_text())
    mot = s.get("max_output_tokens")
    if mot and mot > 8192:
        print(f"  ! max_output_tokens {mot} > 8192: Finding #1 context overflows — use an out8k-based arm")
    zsha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    psha = hashlib.sha256((src / "prompts" / "system.md").read_bytes()).hexdigest()
    git = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    card = (f"name {name} · repo {git} · zip sha256 {zsha[:16]} · system.md sha256 {psha[:16]}\n"
            f"model {ALIAS} · dataset {DATASET}\n"
            f"sampling {json.dumps(s, separators=(',', ':'))}\n"
            f"eval_config {ev} · lora {'kept' if os.environ.get('NOLORA', '1') == '0' else 'removed'}")
    (out / "card.txt").write_text(card + "\n")

    print(f"== submission {name}  ←  {src}")
    for n, ok, det in results:
        print(f"  {'✓' if ok else '✗'} {n:42} {det}")
    print(card)
    if not all(ok for _, ok, _ in results):
        sys.exit("✗ not submittable — fix the red lines")
    print(f"✓ {zpath}  ({size / 2**20:.2f} MiB)")

    if os.environ.get("UPLOAD") == "1":
        user = os.environ.get("KAGGLE_USER") or os.environ.get("KAGGLE_USERNAME") or next(
            (l.split(":", 1)[1].strip() for l in subprocess.run(["kaggle", "config", "view"], capture_output=True,
                                                                 text=True).stdout.splitlines() if "username" in l), "")
        if not user or user == "None":
            sys.exit("set KAGGLE_USER=<your kaggle username> for the upload (or KAGGLE_USERNAME, e.g. `source $G4D/secrets.env`)")
        dsid = f"{user}/{DATASET}"
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": dsid, "licenses": [{"name": "CC0-1.0"}]}))
        card_txt = out / "card.txt"
        card_tmp = ROOT / "dist" / f"{name}.card.txt"
        shutil.move(card_txt, card_tmp)                  # the dataset holds only the zip + metadata
        print(f"$ kaggle datasets version -p {out} …   (first time: create, private)")
        r = subprocess.run(["kaggle", "datasets", "version", "-p", str(out), "-m", f"{name} {zsha[:12]}"])
        if r.returncode != 0:                            # no dataset yet → create it (private unless --public)
            r = subprocess.run(["kaggle", "datasets", "create", "-p", str(out)])
        shutil.move(card_tmp, card_txt)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
