#!/usr/bin/env python3
"""Kaggle submission kernel from the CLI: the one-cell notebook (scripts/kaggle_notebook_cell.py) that copies / re-zips the
private dataset's submission.zip to /kaggle/working, pushed as a private CPU kernel with that dataset as input, polled
until it completes, then (optionally) submitted to the competition from that kernel version.

    python3 scripts/kaggle_sub_kernel.py dist/<name> --kernel g4-sub-e4b-m2 [--adapter e4b_m2] [--alias gemma-4-e4b-it-qat-w4a16-ct]
                                         [--submit "message"] [--dry-run]

dist/<name>/ comes from `UPLOAD=1 … scripts/make_submission.py` (submission.zip + dataset-metadata.json → the dataset id).
--adapter <name>: the cell checks the LoRA entry instead of the no-LoRA trap (adapters/<name>/ present, rank ≤ 128, name
≠ alias, `adapter: <name>` in agent.yaml). Prints every effective value; the kernel's log (sha256 of the zip it wrote)
is compared with dist/<name>/card.txt.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMP = "gemma-4-developer-agent"


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)
    if check and r.returncode != 0:
        sys.exit(f"$ {' '.join(cmd)}\n{out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dist")
    ap.add_argument("--kernel", required=True, help="kernel slug (new or existing, under the dataset's owner)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--alias", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--title", default=None)
    ap.add_argument("--submit", default=None, help="competition submission message (omit = push + poll only)")
    ap.add_argument("--poll", type=int, default=20, help="seconds between status polls")
    ap.add_argument("--dry-run", action="store_true", help="write the kernel dir only")
    a = ap.parse_args()
    dist = pathlib.Path(a.dist).resolve()
    meta = json.loads((dist / "dataset-metadata.json").read_text())
    dsid = meta["id"]; user = dsid.split("/")[0]
    card = (dist / "card.txt").read_text().strip() if (dist / "card.txt").exists() else ""
    zsha = re.search(r"zip sha256 (\w+)", card)
    cell = (ROOT / "scripts" / "kaggle_notebook_cell.py").read_text()
    cell = cell.replace("ADAPTER = None ", f"ADAPTER = {a.adapter!r} ", 1).replace('ALIAS = "gemma-4-31b-it-qat-w4a16-ct"', f"ALIAS = {a.alias!r}", 1)
    assert f"ADAPTER = {a.adapter!r}" in cell and f"ALIAS = {a.alias!r}" in cell, "cell substitution failed"
    kdir = dist / "kernel"; kdir.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": cell, "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (kdir / f"{a.kernel}.ipynb").write_text(json.dumps(nb, indent=1))
    kid = f"{user}/{a.kernel}"
    kmeta = {"id": kid, "title": a.title or a.kernel, "code_file": f"{a.kernel}.ipynb", "language": "python",
             "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
             "keywords": [], "dataset_sources": [dsid], "kernel_sources": [], "competition_sources": [COMP], "model_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=1))
    print(f"kernel {kid} · dataset {dsid} · adapter {a.adapter} · alias {a.alias} · competition {COMP} · internet off · CPU · "
          f"local zip sha256 {zsha.group(1) if zsha else '?'} · dir {kdir}")
    if a.dry_run:
        print("dry run: not pushed"); return 0
    out = sh("kaggle", "kernels", "push", "-p", str(kdir)); print(out)
    m = re.search(r"version (\d+)", out)
    version = m.group(1) if m else None
    t0 = time.time()
    while True:
        st = sh("kaggle", "kernels", "status", kid, check=False)
        print(f"  [{time.time() - t0:4.0f} s] {st.strip()}")
        if "complete" in st.lower() or "error" in st.lower() or "cancel" in st.lower():
            break
        time.sleep(a.poll)
    logdir = kdir / "output"; logdir.mkdir(exist_ok=True)
    sh("kaggle", "kernels", "output", kid, "-p", str(logdir), check=False)
    log = next(iter(sorted(logdir.glob("*.log"))), None)
    text = log.read_text() if log else ""
    if log:
        try:
            lines = [j.get("data", "") for j in json.loads(text) if isinstance(j, dict)]
            text = "".join(lines)
        except Exception:
            pass
    print("-- kernel log:", text[:1500] if text else "(none)")
    ksha = re.search(r"sha256 (\w+)", text)
    if "complete" not in st.lower():
        sys.exit("✗ kernel did not complete")
    if zsha and ksha:
        same = ksha.group(1) == zsha.group(1)
        print(f"zip sha256 on Kaggle {ksha.group(1)} vs local {zsha.group(1)} → {'same' if same else 'DIFFERENT (re-zipped from the unpacked dataset?)'}")
    if a.submit:
        args = ["kaggle", "competitions", "submit", "-c", COMP, "-k", kid, "-f", "submission.zip", "-m", a.submit]
        if version:
            args += ["-v", version]
        print("$", " ".join(args)); print(sh(*args))
        print(sh("kaggle", "competitions", "submissions", "-c", COMP, check=False).splitlines()[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
