#!/usr/bin/env python3
"""What differs between a submission and the one latest setting (scripts/arms/latest/ + the reference prompt).

    python3 scripts/latest_diff.py dist/course_nb/submission.zip          # a built zip
    python3 scripts/latest_diff.py runs/subs/course                       # an arm dir (agent.yaml at the root)
    python3 scripts/latest_diff.py /tmp/k03/some-notebook.ipynb           # a notebook that writes the files inline
    REF=runs/base_lb024 python3 scripts/latest_diff.py …                  # the reference prompt dir (default)

Latest = configs/sampling.yaml + eval_config.yaml from scripts/arms/latest/, agent.yaml = the base model alias with the
9 harness tools and no adapter, prompt = $REF/prompts/system.md with its budget line rewritten to the latest calls /
minutes (what eval_dev.sh LATEST=1 builds for the base arm). Prints every knob as `same` or `latest → this`, then the
prompt's changed lines. Read-only.
"""
import ast
import difflib
import json
import os
import pathlib
import re
import sys
import zipfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
LDIR = ROOT / "scripts" / "arms" / "latest"
REF = pathlib.Path(os.environ.get("REF", ROOT / "runs" / "base_lb024"))
TOOLS9 = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
          "get_code_neighbors", "search_similar_code", "get_code_subgraph"]


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("!include", lambda loader, node: "!include " + loader.construct_scalar(node))


def files_of(src):
    """→ {relative path: text} for agent.yaml, configs/*.yaml, eval_config.yaml, prompts/*.md"""
    p = pathlib.Path(src)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            return {n: z.read(n).decode("utf-8", "replace") for n in z.namelist() if not n.endswith("/")}
    if p.suffix == ".ipynb":     # string constants + a FILES = {path: text|NAME} dict, no code is run
        env, out = {}, {}
        for c in json.load(open(p))["cells"]:
            if c["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(c["source"]))
            except SyntaxError:
                continue
            for node in tree.body:
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                    continue
                name, v = node.targets[0].id, node.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    env[name] = v.value
                elif isinstance(v, ast.Dict):
                    d = {}
                    for k, val in zip(v.keys, v.values):
                        if isinstance(k, ast.Constant) and isinstance(val, ast.Constant) and isinstance(val.value, str):
                            d[k.value] = val.value
                        elif isinstance(k, ast.Constant) and isinstance(val, ast.Name) and val.id in env:
                            d[k.value] = env[val.id]
                    if "agent.yaml" in d:
                        out = d
        if not out:
            sys.exit(f"{p}: no FILES dict with agent.yaml in its code cells (a notebook that copies a zip: pass the zip)")
        return out
    return {str(f.relative_to(p)): f.read_text() for f in p.rglob("*") if f.is_file() and f.suffix in (".yaml", ".yml", ".md")}


def y(text):
    return yaml.load(text, Loader=Loader) if text else None


lat_s = y((LDIR / "configs" / "sampling.yaml").read_text())
lat_e = y((LDIR / "eval_config.yaml").read_text())["evaluation"]
F = files_of(sys.argv[1])
agent = y(F.get("agent.yaml"))
inc = lambda key, default: (str(agent.get(key) or "").removeprefix("!include ").strip() or default) if agent else default
s = y(F.get(inc("generate_content_config", "configs/sampling.yaml"))) or {}
e = (y(F.get("eval_config.yaml")) or {}).get("evaluation", {})
print(f"latest_diff: {sys.argv[1]}  vs  latest ({LDIR.relative_to(ROOT)}) + prompt {REF}/prompts/system.md")
diffs = 0


def row(name, want, got):
    global diffs
    same = want == got
    diffs += not same
    print(f"  {name:36} {'same' if same else f'{want} → {got}'}")


def flat(d, pre=""):
    for k, v in (d or {}).items():
        yield from (flat(v, f"{pre}{k}.") if isinstance(v, dict) else [(pre + k, v)])


ls, ss = dict(flat(lat_s)), dict(flat(s))
for k in sorted(set(ls) | set(ss)):
    row(f"sampling {k}", ls.get(k), ss.get(k))
for k in sorted(set(lat_e) | set(e)):
    row(f"eval {k}", lat_e.get(k), e.get(k))
tools = [t if isinstance(t, str) else next(iter(t)) for t in (agent or {}).get("tools", [])]
row("agent tools", "9 harness tools", "9 harness tools" if tools == TOOLS9 else ", ".join(tools))
row("agent adapter", None, (agent or {}).get("adapter"))
row("agent sub_agents", None, (agent or {}).get("sub_agents"))
print(f"  {'agent model':36} {(agent or {}).get('model')}")

prompt = F.get(inc("instruction", "prompts/system.md"), "")
ref = (REF / "prompts" / "system.md").read_text() if (REF / "prompts" / "system.md").exists() else None
if ref is None:
    print(f"  prompt: no reference at {REF}/prompts/system.md (build it: scripts/from_notebook.py, see eval_dev.sh)")
else:
    ref = re.sub(r"\d+ tool calls and \d+ minutes in total",
                 f"{lat_e['max_tool_calls']} tool calls and {lat_e['max_time_minutes']} minutes in total", ref)
    d = [l for l in difflib.unified_diff(ref.splitlines(), prompt.splitlines(), "latest", "this", n=0, lineterm="")
         if not l.startswith(("---", "+++", "@@"))]
    diffs += bool(d)
    print(f"  {'prompt':36} {len(ref):,} → {len(prompt):,} chars · "
          f"{'same' if not d else f'{sum(l[0] == chr(45) for l in d)} lines out, {sum(l[0] == chr(43) for l in d)} in'}")
    for l in d:
        print(f"    {l[0]} {l[1:][:220]}")
print(f"→ {diffs} difference(s) from latest")
