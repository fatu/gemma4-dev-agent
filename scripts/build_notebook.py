#!/usr/bin/env python3
"""Turn a checked submission.zip into a self-contained Kaggle notebook that rewrites it (CPU, internet off).

    python3 scripts/build_notebook.py dist/course_nb/submission.zip dist/course_nb/notebook "Gemma 4 agent: course config"

Writes <out_dir>/<slug>.ipynb + kernel-metadata.json (private, competition source, no GPU, no internet; id from
KAGGLE_USERNAME). The notebook embeds every file of the zip (the public-top pattern: one FILES dict), writes them,
checks them (agent.yaml at root, 9 harness tools only, no adapter, no {…} in the prompt — ADK templates it, thinking on,
max_output ≤ 8192, eval_config budget keys, sha256 of every file = the checked zip's), and writes
/kaggle/working/submission.zip. Nothing here uploads or submits.
"""
import hashlib, json, os, pathlib, re, subprocess, sys, zipfile

if len(sys.argv) != 4:
    sys.exit(__doc__)
zpath, out, title = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
root = pathlib.Path(__file__).resolve().parent.parent
with zipfile.ZipFile(zpath) as z:
    files = {n: z.read(n).decode("utf-8") for n in sorted(z.namelist()) if not n.endswith("/")}
sha = {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()}
commit = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]

md = f"""# {title}

CPU-only, internet off: this notebook only writes `submission.zip` (agent config for the Gemma 4 Developer Agent
competition). Built from repo commit `{commit}` by `scripts/build_notebook.py` from `{zpath}`, after
`scripts/make_submission.py` passed its 8 checks + `validate_directory()`.

Files: {", ".join(f"`{n}`" for n in files)}."""

code = f'''import hashlib, json, pathlib, re, zipfile
import yaml

FILES = json.loads({json.dumps(files)!r})
EXPECTED_SHA256 = {json.dumps(sha, indent=1)}

out = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path("submission_build")
sub = out / "submission"
for rel, text in FILES.items():
    p = sub / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({{info}})" if info else ""))
    assert ok, name

class Loader(yaml.SafeLoader):
    pass
Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
agent = yaml.load(FILES["agent.yaml"], Loader=Loader)
sampling = yaml.safe_load(FILES["configs/sampling.yaml"])
ev = yaml.safe_load(FILES["eval_config.yaml"])["evaluation"]
prompt = FILES["prompts/system.md"]
NINE = {{"run_command", "read_file", "write_file", "edit_file", "submit_patch", "get_status",
        "get_code_neighbors", "search_similar_code", "get_code_subgraph"}}

check("1 agent.yaml, sampling.yaml, eval_config.yaml, prompts/system.md at the root",
      all(k in FILES for k in ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]))
check("2 tools = the 9 harness tools", set(agent["tools"]) <= NINE, f"{{len(agent['tools'])}} tools")
check("3 no LoRA adapter", "adapter" not in FILES["agent.yaml"])
check("4 thinking on", sampling["thinking_config"]["include_thoughts"] is True,
      f"thinking_budget {{sampling['thinking_config']['thinking_budget']}}")
check("5 max_output_tokens <= 8192", sampling["max_output_tokens"] <= 8192, f"{{sampling['max_output_tokens']}}")
check("6 eval_config budget", set(ev) >= {{"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}}, json.dumps(ev))
check("7 no braces in the prompt (ADK would template them)", "{{" not in prompt and "}}" not in prompt,
      f"{{len(prompt)}} chars")
got = {{n: hashlib.sha256(t.encode()).hexdigest() for n, t in FILES.items()}}
check("8 every file = the checked build (sha256)", got == EXPECTED_SHA256)
print("model", agent["model"], "· sampling", json.dumps(sampling), "· temperature", sampling.get("temperature"))

zpath = out / "submission.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in FILES:
        z.write(sub / rel, rel)
with zipfile.ZipFile(zpath) as z:
    check("9 zip round trip", sorted(z.namelist()) == sorted(FILES) and z.read("agent.yaml").decode() == FILES["agent.yaml"])
print("wrote", zpath, zpath.stat().st_size, "bytes")
'''

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
user = os.environ.get("KAGGLE_USERNAME", "<KAGGLE_USERNAME>")
meta = {"id": f"{user}/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "competition_sources": ["gemma-4-developer-agent"],
        "dataset_sources": [], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {out / (slug + '.ipynb')} + kernel-metadata.json · {len(files)} files embedded · commit {commit}")
for n in files:
    print(f"  {n:24} {sha[n][:16]}")
