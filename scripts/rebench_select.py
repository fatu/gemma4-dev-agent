#!/usr/bin/env python3
"""Pick more public SWE-rebench tasks for E4B's own rollouts (BOX_PLAN B-7) → an --ids file for rebench_build.py.

    .venv-eval/bin/python scripts/rebench_select.py [--n 450] [--per-repo 2] [--seed 0] [--out splits/rebench_more450.txt]

Source: nebius/SWE-rebench at the revision rebench_build.py pins (test + filtered, 21,336 unique rows, 3,468 repos;
the parquet files from the HF cache, ≈ 243 MB). Building a task needs CPU and network only; the GPU comes after the
gold check. The filters keep tasks that our sandbox can plausibly run (Python 3.12 eval venv, pip deps only, pytest
ids) and that look like the competition's (a bug fix in 1–3 files, a clear issue, fair tests):
  · not one of the kit's 81 (splits/rebench_kit81.txt) · repo not in EXCLUDE_REPOS (the 4 repos of the competition's
    129 public tasks + the 2 OOD-45 repos named in its README; OOD-45's tasks are from 2026-09, after this revision)
  · install_config.python 3.9–3.12 (the kit's 81: 3.5–3.10, 67 passed gold on the box; older code breaks more on 3.12)
  · log_parser parse_log_pytest, every FAIL_TO_PASS id a pytest node id ("::"), 1–20 FAIL_TO_PASS, ≤ 1,500 PASS_TO_PASS
  · no environment.yml / Pipfile / poetry install, pre_install only `apt-get update` / `apt-get install -y gcc`
  · gold patch: 1–3 files (meta.num_modified_files), ≤ 80 changed lines, no binary hunk
  · meta.llm_score: issue_text ≤ 2, test ≤ 2, difficulty 0–2 (SWE-rebench's own LLM grades; 3 = bad / hard)
  · created_at ≥ 2020
  · at most --per-repo tasks per repo; the order inside the pool is sha1(seed + instance_id), so a rerun with the same
    seed gives the same list. Output lines (sorted by repo): `nebius <instance_id> <base_commit>`.
Prints how many rows each filter removed and the picked set's mix (📷). The id list is public (task ids, no text).
"""
import argparse
import collections
import glob
import hashlib
import os
import sys

REV = "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
EXCLUDE_REPOS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx",
                 "has2k1/plotnine", "posit-dev/great-tables"}
OK_PRE = {"apt-get update", "apt-get install -y gcc"}
COLS = ["instance_id", "repo", "base_commit", "install_config", "meta", "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "created_at"]


def parquet_files():
    from huggingface_hub import hf_hub_download
    return [hf_hub_download("nebius/SWE-rebench", f, repo_type="dataset", revision=REV)
            for f in ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet",
                      "data/filtered-00000-of-00001.parquet")]


def changed_lines(patch):
    return sum(1 for l in patch.splitlines() if (l.startswith("+") or l.startswith("-"))
               and not l.startswith("+++") and not l.startswith("---"))


def why_not(r, kit):
    """the first filter a row fails (None = keep)"""
    cfg = r["install_config"] or {}
    meta = r["meta"] or {}
    sc = meta.get("llm_score") or {}
    f2p, p2p, patch = r["FAIL_TO_PASS"] or [], r["PASS_TO_PASS"] or [], r["patch"] or ""
    py = str(cfg.get("python") or "")
    checks = [
        ("kit 81", r["instance_id"] in kit),
        ("excluded repo", r["repo"] in EXCLUDE_REPOS),
        ("python not 3.9-3.12", py.split(".")[:2] not in (["3", "9"], ["3", "10"], ["3", "11"], ["3", "12"])),
        ("log parser", cfg.get("log_parser") != "parse_log_pytest"),
        ("FAIL_TO_PASS 0 or > 20", not 1 <= len(f2p) <= 20),
        ("FAIL_TO_PASS not node ids", not all("::" in x for x in f2p)),
        ("PASS_TO_PASS > 1500", len(p2p) > 1500),
        ("conda/pipenv/poetry install", bool(cfg.get("env_yml_path")) or (cfg.get("packages") or "") in
         ("environment.yml", "Pipfile", "poetry")),
        ("pre_install beyond gcc", any(x.strip() not in OK_PRE for x in (cfg.get("pre_install") or []))),
        ("patch files not 1-3", not 1 <= (meta.get("num_modified_files") or 0) <= 3),
        ("patch > 80 lines or binary", changed_lines(patch) > 80 or "GIT binary patch" in patch or "Binary files" in patch),
        ("llm score (issue/test > 2, difficulty not 0-2)", not (0 <= (sc.get("issue_text_score") if sc.get("issue_text_score") is not None else 9) <= 2
                                                               and 0 <= (sc.get("test_score") if sc.get("test_score") is not None else 9) <= 2
                                                               and 0 <= (sc.get("difficulty_score") if sc.get("difficulty_score") is not None else 9) <= 2)),
        ("created before 2020", (r["created_at"] or "0") < "2020"),
    ]
    return next((name for name, bad in checks if bad), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450, help="tasks to pick (≈ 2/3 survive build + gold → ~300)")
    ap.add_argument("--per-repo", type=int, default=2)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--kit", default="splits/rebench_kit81.txt")
    ap.add_argument("--files", nargs="*", help="parquet files (default: the pinned revision from the HF cache / hub)")
    ap.add_argument("--out", default="splits/rebench_more450.txt")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    kit = {l.split()[1] for l in open(a.kit) if l.strip() and not l.startswith("#")}
    rows = {}
    for f in a.files or parquet_files():
        for r in pq.read_table(f, columns=COLS).to_pylist():
            rows.setdefault(r["instance_id"], r)
    print(f"rebench_select: nebius/SWE-rebench@{REV[:7]} · {len(rows):,} unique rows · {len({r['repo'] for r in rows.values()}):,} repos "
          f"· n {a.n} · per repo ≤ {a.per_repo} · seed {a.seed} · kit {a.kit} ({len(kit)}) · excluded repos {len(EXCLUDE_REPOS)} → {a.out}")
    drop, pool = collections.Counter(), []
    for r in rows.values():
        w = why_not(r, kit)
        if w:
            drop[w] += 1
        else:
            pool.append(r)
    for name, k in drop.most_common():
        print(f"  dropped {k:>6,}  {name}")
    print(f"  pool    {len(pool):>6,} tasks in {len({r['repo'] for r in pool}):,} repos")
    pool.sort(key=lambda r: hashlib.sha1(f"{a.seed}:{r['instance_id']}".encode()).hexdigest())
    per, pick = collections.Counter(), []
    for r in pool:
        if per[r["repo"]] < a.per_repo:
            per[r["repo"]] += 1; pick.append(r)
            if len(pick) == a.n:
                break
    if len(pick) < a.n:
        print(f"  only {len(pick)} tasks pass with ≤ {a.per_repo} per repo")
    pick.sort(key=lambda r: (r["repo"].lower(), r["instance_id"]))
    with open(a.out, "w") as w:
        w.write(f"# rebench_select.py --n {a.n} --per-repo {a.per_repo} --seed {a.seed} · nebius/SWE-rebench@{REV[:7]} · "
                f"{len(pick)} tasks, {len(per)} repos\n")
        for r in pick:
            w.write(f"nebius {r['instance_id']} {r['base_commit']}\n")
    C = lambda f: " ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(f(r) for r in pick).items()))
    print(f"== picked {len(pick)} tasks in {len(per)} repos (per repo {dict(collections.Counter(per.values()))}) → {a.out}")
    print(f"   python {C(lambda r: (r['install_config'] or {}).get('python'))}")
    print(f"   files {C(lambda r: (r['meta'] or {}).get('num_modified_files'))} · difficulty "
          f"{C(lambda r: ((r['meta'] or {}).get('llm_score') or {}).get('difficulty_score'))} · year {C(lambda r: r['created_at'][:4])}")
    print(f"   patch lines p50 {sorted(changed_lines(r['patch']) for r in pick)[len(pick) // 2]} · FAIL_TO_PASS p50 "
          f"{sorted(len(r['FAIL_TO_PASS']) for r in pick)[len(pick) // 2]} · gcc pre_install "
          f"{sum(1 for r in pick if (r['install_config'] or {}).get('pre_install'))}")
    return 0 if pick else 1


if __name__ == "__main__":
    sys.exit(main())
