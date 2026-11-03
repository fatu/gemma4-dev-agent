#!/usr/bin/env python3
"""SWE-rebench tasks → swegemma tasks for generating our own native 31B training trajectories (BOX_PLAN.md B-3n).

    .venv-eval/bin/python scripts/rebench_build.py [--ids splits/rebench_kit81.txt] [--out runs/rebench] [--only ID ...]

Reproduces the task side of dmitriigluzdov's LoRA kit (its native tier: Gemma 4 31B on 81 public SWE-rebench tasks,
none a competition task; CC BY 4.0 task metadata, credit Nebius / White Circle). The kit names its tasks, sources and
base commits (lora_sources_native.json → splits/rebench_kit81.txt) and says its labels are "the recorded local
subprocess grader results" — the same `--sandbox subprocess` we run; how it installed each repo's dependencies is not
published, so this is our own version. Per task (internet needed; the lanes later run offline):
  1. row: nebius/SWE-rebench (splits test, filtered) or whitecircle/swe-rebench-v2-clean-python-tasks (train, test) at
     the revisions the kit names (its lora_sources_native.json): the parquet files once (≈ 310 MB, HF cache), filtered
     by instance_id with pyarrow (the datasets-server /filter API answered "index is loading" / timed out, 2026-11-03)
  2. tasks.jsonl line in swegemma's Task schema: instance_id, repo, base_commit, patch (gold, never shown to the agent),
     test_patch, problem_statement, hints_text, created_at, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
  3. snapshots/<id>.tgz: the repo's tracked files at base_commit in a fresh one-commit git repo on `main` (the harness
     commits its own baseline on top; no history after base_commit exists, nor before it)
  4. site/<id>/: the task's Python dependencies for THIS interpreter (`uv pip install --target`): the project itself
     (pulls its install_requires) + install_config.pip_packages + install_config.reqs_path files. scripts/sandbox_shim.py
     puts it on the sandbox's sys.path right after the workspace when REBENCH_SITE=<out>/site. If the project does
     not install, retried without it ("deps partial"). Python is the eval venv's (3.12), not the row's install_config
     python (e.g. 3.5): tasks that do not run here fail the gold check (step `gold` of rebench_gen.sh) and are dropped.
  meta/<id>.json keeps the row's install_config, test_cmd, docker_image, license and python version.
Prints one line per task and a total (📷); every number is the record. Outputs quote task text → runs/ (gitignored).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

DATASETS = {"nebius": ("nebius/SWE-rebench", "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881",
                       ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet",
                        "data/filtered-00000-of-00001.parquet")),
            "whitecircle": ("whitecircle/swe-rebench-v2-clean-python-tasks", "17038e42a5a92252531a7fa4be271cb97c6a936d",
                            ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"))}


def load_rows(tag, ids):
    """{instance_id: (row, source)} for the ids of one dataset, from its parquet files at the pinned revision"""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    ds, rev, files = DATASETS[tag]
    out = {}
    for f in files:
        path = hf_hub_download(ds, f, repo_type="dataset", revision=rev)
        t = pq.read_table(path)
        col = t.column("instance_id").to_pylist()
        keep = [i for i, x in enumerate(col) if x in ids and x not in out]
        for r in t.take(keep).to_pylist():
            out[r["instance_id"]] = (r, f"{ds}@{rev[:7]}:{os.path.basename(f).split('-')[0]}")
        print(f"  {ds}@{rev[:7]} {f}: {t.num_rows} rows, {len(keep)} of ours", flush=True)
    return out


def as_list(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return [x] if x else []
    return list(x or [])


def task_row(row, iid):
    return {"instance_id": iid, "repo": row["repo"], "base_commit": row["base_commit"], "patch": row.get("patch") or "",
            "test_patch": row.get("test_patch") or "", "problem_statement": row.get("problem_statement") or "",
            "hints_text": row.get("hints_text") or "", "created_at": str(row.get("created_at") or ""),
            "FAIL_TO_PASS": as_list(row.get("FAIL_TO_PASS")), "PASS_TO_PASS": as_list(row.get("PASS_TO_PASS")),
            "environment_setup_commit": row.get("environment_setup_commit") or row["base_commit"]}


def sh(cmd, cwd=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def snapshot(repo, sha, git_cache, out_tgz, work):
    """tracked files at sha → fresh one-commit repo on main → tgz (entries ./…)"""
    cache = os.path.join(git_cache, repo.replace("/", "__"))
    if not os.path.isdir(os.path.join(cache, ".git")):
        r = sh(["git", "clone", "-q", "--no-tags", f"https://github.com/{repo}.git", cache])
        if r.returncode:
            return f"clone failed: {r.stderr.strip()[-120:]}"
    if sh(["git", "-C", cache, "cat-file", "-e", f"{sha}^{{commit}}"]).returncode:
        r = sh(["git", "-C", cache, "fetch", "-q", "origin", sha])
        if r.returncode:
            return f"commit {sha[:10]} not fetchable: {r.stderr.strip()[-120:]}"
    src = os.path.join(work, "src")
    shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
    r = sh(["git", "-C", cache, f"--work-tree={src}", "checkout", "-f", sha, "--", "."])
    sh(["git", "-C", cache, "reset", "-q"])                                   # leave the cache index clean
    if r.returncode:
        return f"checkout failed: {r.stderr.strip()[-120:]}"
    for c in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
              ["git", "-c", "user.name=swegemma", "-c", "user.email=swegemma@localhost", "commit", "-q", "-m", f"base {sha}"]):
        r = sh(c, cwd=src)
        if r.returncode:
            return f"{c[1]} failed: {r.stderr.strip()[-120:]}"
    r = sh(["tar", "czf", out_tgz, "-C", src, "."])
    return None if r.returncode == 0 else f"tar failed: {r.stderr.strip()[-120:]}"


def deps(src, cfg, site, py):
    """uv pip install --target site: project + pip_packages + reqs files; retry without the project"""
    pkgs = [p for p in (cfg.get("pip_packages") or []) if p]
    rps = cfg.get("reqs_path") or []
    reqs = [x for rp in (rps if isinstance(rps, list) else [rps]) if os.path.isfile(os.path.join(src, rp))
            for x in ("-r", os.path.join(src, rp))]
    shutil.rmtree(site, ignore_errors=True); os.makedirs(site)
    base = ["uv", "pip", "install", "-q", "--python", py, "--target", site]
    r = sh(base + [src] + pkgs + reqs)
    how = "full"
    if r.returncode:
        err = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
        shutil.rmtree(site, ignore_errors=True); os.makedirs(site)
        r = sh(base + (pkgs or ["pytest"]) + reqs)
        how = f"partial (project not installable: {err[0][:70]})" if r.returncode == 0 else \
            f"FAILED: {((r.stderr or r.stdout).strip().splitlines() or [''])[-1][:90]}"
    n = sum(1 for d in os.listdir(site) if d.endswith(".dist-info"))
    return how, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="splits/rebench_kit81.txt", help="lines: <nebius|whitecircle> <instance_id> [base_commit …]")
    ap.add_argument("--out", default="runs/rebench")
    ap.add_argument("--only", nargs="*", default=None, help="a subset of instance ids (smoke)")
    ap.add_argument("--python", default=sys.executable, help="interpreter the sandbox venvs are made from (the eval venv's)")
    ap.add_argument("--no-deps", action="store_true", help="skip step 4")
    a = ap.parse_args()
    for t in ("git", "tar") + (() if a.no_deps else ("uv",)):
        if not shutil.which(t):
            sys.exit(f"{t} not found")
    entries = [l.split() for l in open(a.ids) if l.strip() and not l.startswith("#")]
    if a.only:
        entries = [e for e in entries if e[1] in set(a.only)]
    for d in ("snapshots", "site", "meta", "git"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    pyv = sh([a.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]).stdout.strip()
    print(f"rebench_build: {len(entries)} tasks from {a.ids} → {a.out} · deps for python {pyv} ({a.python})"
          f"{' · deps skipped' if a.no_deps else ''}", flush=True)
    rows = {}
    for tag in sorted({e[0] for e in entries}):
        rows.update(load_rows(tag, {e[1] for e in entries if e[0] == tag}))
    tasks, tot = [], {"tasks": 0, "rows": 0, "snapshots": 0, "deps full": 0, "deps partial": 0, "deps failed": 0}
    work = tempfile.mkdtemp(prefix="rebench_")
    for tag, iid, *rest in entries:
        t0 = time.time(); tot["tasks"] += 1
        row, src_name = rows.get(iid, (None, None))
        if not row:
            print(f"{iid:<58} NO ROW in {DATASETS[tag][0]}", flush=True); continue
        tot["rows"] += 1
        t = task_row(row, iid)
        if rest and rest[0] != t["base_commit"]:
            print(f"  {iid}: base_commit {t['base_commit'][:10]} ≠ list {rest[0][:10]} (the list's is the kit's)", flush=True)
        tgz = os.path.join(a.out, "snapshots", f"{iid}.tgz")
        err = None if os.path.exists(tgz) else snapshot(t["repo"], t["base_commit"], os.path.join(a.out, "git"), tgz, work)
        if err:
            print(f"{iid:<58} snapshot: {err}", flush=True); continue
        tot["snapshots"] += 1
        cfg = row.get("install_config") or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        how, n = ("skipped", 0)
        if not a.no_deps:
            src = os.path.join(work, "src")
            if not os.path.isdir(src) or not os.listdir(src):            # snapshot existed: unpack it for the install
                shutil.rmtree(src, ignore_errors=True); os.makedirs(src); sh(["tar", "xzf", tgz, "-C", src])
            how, n = deps(src, cfg, os.path.join(a.out, "site", iid), a.python)
            tot["deps full" if how == "full" else "deps partial" if how.startswith("partial") else "deps failed"] += 1
        shutil.rmtree(os.path.join(work, "src"), ignore_errors=True)
        json.dump({"source": src_name, "install_config": cfg, "python": cfg.get("python"), "test_cmd": cfg.get("test_cmd"),
                   "docker_image": row.get("docker_image") or row.get("image_name"), "license": row.get("license_name") or row.get("license"),
                   "kit_list": rest}, open(os.path.join(a.out, "meta", f"{iid}.json"), "w"))
        tasks.append(t)
        print(f"{iid:<58} {src_name.split('/')[0]:<11} snapshot {os.path.getsize(tgz) / 1e6:6.1f} MB · F2P {len(t['FAIL_TO_PASS']):>3} "
              f"P2P {len(t['PASS_TO_PASS']):>4} · py {cfg.get('python') or '?':<4} · deps {how} ({n} dists) · {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(a.out, "tasks.jsonl"), "w") as w:
        for t in tasks:
            w.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "built.txt"), "w") as w:
        w.write("".join(t["instance_id"] + "\n" for t in tasks))
    print("TOTAL " + " · ".join(f"{k} {v}" for k, v in tot.items()) + f" → {a.out}/tasks.jsonl ({len(tasks)} lines), built.txt")
    return 0 if tasks else 1


if __name__ == "__main__":
    sys.exit(main())
