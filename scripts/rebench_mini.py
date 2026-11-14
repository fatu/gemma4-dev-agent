#!/usr/bin/env python3
"""B-10 step 1: the MetaCurate task set (1,415 tasks from SWE-rebench, SWE-rebench-V2, SWE-Gym) for mini-swe-agent, and
one grader for all three sources. Called by scripts/rebench_mini.sh (run it with .venv-swe/bin/python).

    tasks   --pool <dir with the MetaCurate parquet shards> --out runs/b10/tasks
            the pool's (instance_id, source) pairs joined with the upstream rows at the revisions the pool's release note
            pins → <out>/all.jsonl: one row per task with what mini needs (problem_statement, image_name, workdir) and what
            the grader needs (base_commit, patch = gold, test_patch, FAIL_TO_PASS, PASS_TO_PASS, test_cmd, log_parser).
    subset  --tasks <out> --name pilot30 --n 30 [--per-source 0] [--seed 0] [--sources a,b]
            a stratified sample → <out>/<name>/data/test-00000-of-00001.parquet (what `mini-extra swebench --subset <dir>
            --split test` loads) + <out>/<name>.ids
    grade   --tasks <out> --name <subset> --preds gold|empty|pool|<preds.json> --out <dir> [--workers 4] [--install auto]
            every patch applied in a fresh container of the task's image, the tests run, the log parsed → <dir>/report.json
            (`pool` = the pool's own patches for these tasks, compared with the pool's `resolved` labels)
    report  --run <mini output dir> [--grade <dir>]        the pilot line: exits, calls, thoughts, tokens, wall time

Grading, per source (ported from the three upstream harnesses, read 2026-11-13; nothing of them is imported):
  SWE-rebench (SWE-rebench/SWE-bench-fork) and SWE-Gym (SWE-Gym/SWE-Bench-Fork): in /testbed with the conda env `testbed`:
    apply the patch (git apply → git apply --reject → patch --fuzz=5) · `git checkout <base> -- <test files of the test
    patch that exist at base>` · apply the test patch · `<test_cmd> <files of the test patch, without data files>` ·
    parse the pytest -rA lines · resolved = every FAIL_TO_PASS and PASS_TO_PASS test is PASSED or XFAIL.
    SWE-rebench: test_cmd and log_parser come from the row's install_config. Its images do not always hold the project
    installed (box 2026-11-13: 2 of 5 images; the pool's own runs met the same import errors, and its labels count those
    patches as resolved), so `--install auto` gives a patch that is not resolved, and whose log parsed nothing or shows
    an import error, a second pass in the same container after the row's install command (needs the network). SWE-Gym: test_cmd per repo (GYM_TEST_CMD
    below, from the fork's constants; the fork also re-runs the repo's install command before the tests: NOT done here,
    so a task whose gold patch needs a rebuild fails the gold check and is left out).
  SWE-rebench-V2 (SWE-rebench/SWE-rebench-V2 scripts/eval.py): in /<repo name>: `git reset --hard HEAD` · `git apply
    --3way --recount --ignore-space-change` for the patch and the test patch · the row's test_cmd as is · resolved =
    every FAIL_TO_PASS and PASS_TO_PASS test is PASSED (their strict "the passed set equals the expected set" is printed
    as well). Only the pytest parsers are ported: a V2 task with another log_parser is marked unsupported in `tasks`.
A task is usable only if its gold patch grades resolved and the empty patch does not (step `check`, later on all tasks).
"""
import argparse
import collections
import concurrent.futures
import glob
import json
import os
import random
import re
import subprocess
import tempfile
import time
import uuid

UPSTREAM = {  # the pool's `dataset` value → (HF id, split, revision pinned in the pool's release note)
    "nebius/SWE-rebench": ("nebius/SWE-rebench", "test", "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"),
    "nebius/SWE-rebench-V2": ("nebius/SWE-rebench-V2", "train", "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"),
    "SWE-Gym/SWE-Gym": ("SWE-Gym/SWE-Gym", "train", "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"),
}
SHORT = {"nebius/SWE-rebench": "rebench", "nebius/SWE-rebench-V2": "v2", "SWE-Gym/SWE-Gym": "gym"}
GYM_TEST_CMD = {"getmoto/moto": "pytest -n0 -rA", "conan-io/conan": "pytest -n0 -rA", "modin-project/modin": "pytest -n0 -rA",
                "dask/dask": "pytest -n0 -rA --color=no", "Project-MONAI/MONAI": "pytest -rA", "iterative/dvc": "pytest -rA",
                "bokeh/bokeh": "pytest -rA -n0", "pandas-dev/pandas": "pytest -rA --tb=long",
                "facebookresearch/hydra": "pytest -rA --tb=long", "python/mypy": "pytest -n0 -rA -k",
                "pydantic/pydantic": "pytest -rA --tb=short -vv -o console_output_style=classic --no-header"}
NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")
STATUSES = ("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")
PARSERS = ("parse_log_pytest", "parse_log_pytest_options", "parse_log_pytest_v2", "parse_log_pytest_pydantic")
ESCAPES = str.maketrans("", "", "".join(chr(c) for c in range(1, 32)))
TIMING = [re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.I), re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.I),
          re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.I)]


def as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v]
    return list(v)


# ----------------------------------------------------------------------------------------------------------- tasks
def cmd_tasks(a):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(glob.glob(os.path.join(a.pool, "**", "*train*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet shards under {a.pool}: .venv-eval/bin/python train/mc_prep.py download --dir {a.pool}")
    want = collections.defaultdict(dict)                       # dataset → {instance_id: [attempts, resolved]}
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["instance_id", "dataset", "resolved"]):
            for r in b.to_pylist():
                c = want[r["dataset"]].setdefault(r["instance_id"], [0, 0])
                c[0] += 1
                c[1] += bool(r["resolved"])
    print(f"tasks · pool {a.pool} ({len(files)} shards) · " + " · ".join(f"{SHORT.get(d, d)} {len(v)} ids" for d, v in want.items()))
    os.makedirs(a.out, exist_ok=True)
    rows, stats = [], collections.Counter()
    parsers = collections.Counter()
    for ds, ids in want.items():
        if ds not in UPSTREAM:
            print(f"  unknown source {ds}: {len(ids)} ids skipped")
            continue
        hf, split, rev = UPSTREAM[ds]
        shards = [f for f in list_repo_files(hf, repo_type="dataset", revision=rev)
                  if f.endswith(".parquet") and os.path.basename(f).startswith(split + "-")]
        found = {}                                             # only the wanted rows are kept in memory
        for sh in shards:
            pf = pq.ParquetFile(hf_hub_download(hf, sh, repo_type="dataset", revision=rev))
            for b in pf.iter_batches(batch_size=500):
                for r in b.to_pylist():
                    if r["instance_id"] in ids:
                        found[r["instance_id"]] = r
        src = SHORT[ds]
        print(f"  {hf} @ {rev[:8]} split {split}: {len(shards)} shard(s) · {len(found)} of {len(ids)} ids found")
        for iid, (att, res) in sorted(ids.items()):
            if iid not in found:
                stats[src, "not in upstream"] += 1
                continue
            r = found[iid]
            ic = r.get("install_config") or {}
            if isinstance(ic, str):
                ic = json.loads(ic)
            row = {"instance_id": iid, "source": src, "repo": r["repo"], "base_commit": r["base_commit"],
                   "problem_statement": r["problem_statement"], "patch": r["patch"], "test_patch": r["test_patch"],
                   "FAIL_TO_PASS": as_list(r.get("FAIL_TO_PASS")), "PASS_TO_PASS": as_list(r.get("PASS_TO_PASS")),
                   "pool_attempts": att, "pool_resolved": res, "skip": ""}
            if src == "rebench":
                row.update(image_name=r.get("image_name") or r.get("docker_image") or "", workdir="/testbed",
                           test_cmd=ic.get("test_cmd") or "", log_parser=ic.get("log_parser") or "",
                           install=ic.get("install") or "")
            elif src == "v2":
                tc = ic.get("test_cmd") or ""
                row.update(image_name=r.get("image_name") or "", workdir="/" + r["repo"].split("/")[1],
                           test_cmd=tc if isinstance(tc, str) else "\n".join(tc), log_parser=ic.get("log_parser") or "",
                           language=r.get("language") or "")
            else:
                row.update(image_name=("docker.io/xingyaoww/sweb.eval.x86_64." + iid.replace("__", "_s_")).lower(),
                           workdir="/testbed", test_cmd=GYM_TEST_CMD.get(r["repo"], ""),
                           log_parser="parse_log_pytest_pydantic" if r["repo"] == "pydantic/pydantic" else "parse_log_pytest")
            if isinstance(row["test_cmd"], list):
                row["test_cmd"] = " ".join(row["test_cmd"])
            parsers[src, row["log_parser"]] += 1
            for why, bad in (("no image", not row["image_name"]), ("no test_cmd", not row["test_cmd"]),
                             ("parser not ported", row["log_parser"] not in PARSERS),
                             ("no FAIL_TO_PASS", not row["FAIL_TO_PASS"]), ("no test patch", not row["test_patch"])):
                if bad and not row["skip"]:
                    row["skip"] = why
            stats[src, row["skip"] or "usable"] += 1
            rows.append(row)
    rows.sort(key=lambda r: (("rebench", "v2", "gym").index(r["source"]), r["instance_id"]))
    seen = {}
    for r in rows:                                             # an id listed under two sources: the first source keeps it
        if r["instance_id"] in seen and not r["skip"]:
            r["skip"] = f"same id as a {seen[r['instance_id']]} task"
            stats[r["source"], "usable"] -= 1
            stats[r["source"], "duplicate id"] += 1
        seen.setdefault(r["instance_id"], r["source"])
    with open(os.path.join(a.out, "all.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for src in ("rebench", "v2", "gym"):
        print(f"  {src}: " + " · ".join(f"{k[1]} {v}" for k, v in sorted(stats.items()) if k[0] == src)
              + " · parsers " + json.dumps({k[1]: v for k, v in parsers.items() if k[0] == src}))
    ok = [r for r in rows if not r["skip"]]
    print(f"  rebench rows with an install command (used by grade --install): {sum(bool(r.get('install')) for r in rows)}")
    print(f"== {len(rows)} tasks written to {a.out}/all.jsonl · usable {len(ok)} · with ≥ 1 resolved run in the pool "
          f"{sum(r['pool_resolved'] > 0 for r in ok)} · repos {len({r['repo'] for r in ok})}")


def load_tasks(d):
    return [r for r in (json.loads(l) for l in open(os.path.join(d, "all.jsonl"))) if not r["skip"].startswith("same id")]


def cmd_subset(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [r for r in load_tasks(a.tasks) if not r["skip"]]
    srcs = [s for s in re.split(r"[,\s]+", a.sources.strip()) if s and s != "all"]
    if srcs:
        rows = [r for r in rows if r["source"] in srcs]
    if a.ids:
        keep = set(open(a.ids).read().split())
        rows = [r for r in rows if r["instance_id"] in keep]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    pick = []
    for src, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["instance_id"])
        random.Random(f"{a.seed}:{src}").shuffle(rs)       # per source: the sample of a source does not depend on --sources
        k = a.per_source or (round(a.n * len(rs) / len(rows)) if a.n else len(rs))
        pick += rs[:k]
    pick.sort(key=lambda r: r["instance_id"])
    d = os.path.join(a.tasks, a.name, "data")
    os.makedirs(d, exist_ok=True)
    cols = ("instance_id", "problem_statement", "image_name", "workdir", "source", "repo")
    pq.write_table(pa.table({c: [r[c] for r in pick] for c in cols}), os.path.join(d, "test-00000-of-00001.parquet"))
    open(os.path.join(a.tasks, a.name + ".ids"), "w").write("\n".join(r["instance_id"] for r in pick) + "\n")
    print(f"subset {a.name} · seed {a.seed} · n {a.n} · per source {a.per_source} · sources {','.join(srcs) or 'all'} → {len(pick)} "
          f"tasks ({dict(collections.Counter(r['source'] for r in pick))}) · {os.path.join(a.tasks, a.name)} (mini: --subset "
          f"this directory, --split test)")


# ----------------------------------------------------------------------------------------------------------- grade
def parse_log(log, parser):
    out = {}
    opt = re.compile(r"(.*?)\[(.*)\]")
    for line in log.split("\n"):
        if parser in ("parse_log_pytest_v2", "parse_log_pytest_pydantic"):
            line = re.sub(r"\[(\d+)m", "", line).translate(ESCAPES)
            if parser == "parse_log_pytest_pydantic":
                line = re.sub(r"FAILED\s*\[.*?\]", "FAILED", line)
        if line.startswith(STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            t = line.split()
            if len(t) <= 1:
                continue
            name = t[1]
            if parser == "parse_log_pytest_options":
                m = opt.search(name)
                if m:
                    main, option = m.groups()
                    if option.startswith("/") and not option.startswith("//") and "*" not in option:
                        option = "/" + option.split("/")[-1]
                    name = f"{main}[{option}]"
            out[name] = t[0]
        elif parser in ("parse_log_pytest_v2", "parse_log_pytest_pydantic") and line.endswith(STATUSES):
            t = line.split()
            if len(t) >= 2:
                out[t[0]] = t[1]
    return out


def norm(name):
    for p in TIMING:
        name = p.sub("", name)
    return name.strip()


def grade_script(row, has_patch, install=False):
    """install=True: the second pass in the same container (patches are already applied): the row's install command, then
    the tests again."""
    q = lambda s: "'" + s.replace("'", "'\\''") + "'"  # noqa: E731
    tp = row["test_patch"]
    if row["source"] == "v2":
        ap = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"
        return "\n".join([
            f"cd {q(row['workdir'])} || echo B10_NO_WORKDIR", "git reset --hard HEAD",
            f"{ap} /b10/patch.diff || echo B10_PATCH_FAILED" if has_patch else "echo B10_EMPTY_PATCH",
            f"{ap} /b10/test.diff || echo B10_TEST_PATCH_FAILED", "echo B10_START", row["test_cmd"], "echo B10_END rc=$?"])
    reset = [p for p in re.findall(r"^--- a/(.*)$", tp, flags=re.M)]
    files = [d for d in re.findall(r"diff --git a/.* b/(.*)", tp) if not d.endswith(NON_TEST_EXTS)]
    cmd = row["test_cmd"]
    if row["repo"] == "python/mypy":                                  # the fork runs the named data-driven cases
        cases = re.findall(r"\[case ([^\]]+)\]", tp)
        cmd, files = f"{cmd} {q(' or '.join(cases))}", []
    head = ["for f in /opt/conda/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do [ -f $f ] && . $f; done",
            "conda activate testbed 2>/dev/null || echo B10_NO_CONDA_ENV", "cd /testbed || echo B10_NO_WORKDIR"]
    test = ["echo B10_START", f"{cmd} {' '.join(q(f) for f in files)}", "echo B10_END rc=$?"]
    if install:
        return "\n".join(head + ["echo B10_INSTALL", f"( {row['install']} ) > /b10/install.log 2>&1; echo B10_INSTALL_RC=$?",
                                 "tail -n 6 /b10/install.log"] + test)
    return "\n".join(head + [
        "git config --global --add safe.directory /testbed",
        ("git apply -v /b10/patch.diff || git apply -v --reject /b10/patch.diff || patch --batch --fuzz=5 -p1 -i /b10/patch.diff "
         "|| echo B10_PATCH_FAILED") if has_patch else "echo B10_EMPTY_PATCH",
        f"git checkout {row['base_commit']} -- {' '.join(q(p) for p in reset)}" if reset else ":",
        "git apply -v /b10/test.diff || echo B10_TEST_PATCH_FAILED"] + test)


def grade_one(row, patch, a):
    name, t0 = "b10g-" + uuid.uuid4().hex[:10], time.time()
    res = {"instance_id": row["instance_id"], "source": row["source"], "resolved": False, "note": ""}
    has_patch = bool((patch or "").strip())
    with tempfile.TemporaryDirectory(prefix="b10g_") as tmp:
        for fn, txt in (("patch.diff", (patch or "").rstrip("\n") + "\n"), ("test.diff", row["test_patch"].rstrip("\n") + "\n"),
                        ("run.sh", grade_script(row, has_patch) + "\n")):
            open(os.path.join(tmp, fn), "w").write(txt)
        net = ["--network", a.net] if a.net else []
        try:
            p = subprocess.run(["docker", "run", "-d", "--rm", "--name", name, *net, "--entrypoint", "sleep", row["image_name"],
                                str(2 * a.timeout + 600)], capture_output=True, text=True, timeout=a.pull_timeout)
            if p.returncode != 0:
                res["note"] = "image: " + (p.stderr or p.stdout).strip().splitlines()[-1][:160]
                return res
            subprocess.run(["docker", "cp", tmp + "/.", f"{name}:/b10"], capture_output=True, timeout=120)
            p = subprocess.run(["docker", "exec", name, "bash", "/b10/run.sh"], capture_output=True, timeout=a.timeout)
            log = (p.stdout + p.stderr).decode("utf-8", "replace")
            first = score(row, log)
            broken = not first["parsed"] or any(m in log for m in ("ModuleNotFoundError", "ImportError"))
            if row.get("install") and (a.install == "always" or (a.install == "auto" and not first["resolved"] and broken)):
                open(os.path.join(tmp, "run2.sh"), "w").write(grade_script(row, has_patch, install=True) + "\n")
                subprocess.run(["docker", "cp", tmp + "/run2.sh", f"{name}:/b10/run2.sh"], capture_output=True, timeout=120)
                p = subprocess.run(["docker", "exec", name, "bash", "/b10/run2.sh"], capture_output=True, timeout=a.timeout)
                log += "\n=== B10_SECOND_PASS (after the install command) ===\n" + (p.stdout + p.stderr).decode("utf-8", "replace")
                res["installed"] = True
        except subprocess.TimeoutExpired:
            res["note"] = "timeout"
            return res
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    open(os.path.join(a.out, "logs", row["instance_id"] + ".log"), "w").write(log)
    res.update(score(row, log), s=round(time.time() - t0))
    if res.get("installed"):
        rc = re.findall(r"B10_INSTALL_RC=(\d+)", log)
        res["note"] = (res["note"] + f" installed(rc {rc[-1] if rc else '?'})").strip()
    return res


def score(row, log):
    """The verdict from a grading log; with a second pass in the log, the tests of the last pass count."""
    marks = [m for m in ("B10_NO_WORKDIR", "B10_NO_CONDA_ENV", "B10_PATCH_FAILED", "B10_TEST_PATCH_FAILED") if m in log]
    body = log.rsplit("B10_START", 1)[1] if "B10_START" in log else ""
    st = {norm(k): v for k, v in parse_log(body, row["log_parser"]).items()}
    good = ("PASSED",) if row["source"] == "v2" else ("PASSED", "XFAIL")
    f2p, p2p = [norm(t) for t in row["FAIL_TO_PASS"]], [norm(t) for t in row["PASS_TO_PASS"]]
    fp, pp = sum(st.get(t) in good for t in f2p), sum(st.get(t) in good for t in p2p)
    return dict(f2p=f"{fp}/{len(f2p)}", p2p=f"{pp}/{len(p2p)}", parsed=len(st),
                note=" ".join(m[4:].lower() for m in marks) + ("" if "B10_END" in body else " no_end"),
                resolved=bool(f2p) and fp == len(f2p) and pp == len(p2p) and "B10_PATCH_FAILED" not in log,
                exact=sorted(k for k, v in st.items() if v == "PASSED") == sorted(set(f2p + p2p)))


def cmd_grade(a):
    rows = {r["instance_id"]: r for r in load_tasks(a.tasks)}
    ids = open(os.path.join(a.tasks, a.name + ".ids")).read().split()
    jobs = []                                                   # (key, row, patch, pool label or None)
    if a.preds in ("gold", "empty"):
        jobs = [(i, rows[i], rows[i]["patch"] if a.preds == "gold" else "", None) for i in ids]
    elif a.preds == "pool":
        import pyarrow.parquet as pq
        keep = set(ids)
        for f in sorted(glob.glob(os.path.join(a.pool, "**", "*train*.parquet"), recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["id", "instance_id", "model_patch", "resolved"]):
                jobs += [(r["id"], rows[r["instance_id"]], r["model_patch"], bool(r["resolved"])) for r in b.to_pylist()
                         if r["instance_id"] in keep]
    else:
        preds = json.load(open(a.preds))
        jobs = [(i, rows[i], (preds.get(i) or {}).get("model_patch") or "", None) for i in ids if i in preds]
    os.makedirs(a.out, exist_ok=True)
    print(f"grade · tasks {a.tasks} · subset {a.name} ({len(ids)} ids) · preds {a.preds} → {len(jobs)} patches · workers {a.workers} "
          f"· timeout {a.timeout} s · pull timeout {a.pull_timeout} s · network {a.net or 'docker default'} · install {a.install} (auto = a second pass after "
          f"the row's install command when the first pass is not resolved and parsed nothing or shows an import error; needs the "
          f"network) · rows with an install command {sum(bool(r.get('install')) for _, r, _, _ in jobs)} · out {a.out}")
    out, t0 = {}, time.time()
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(grade_one, row, patch, a): (key, label) for key, row, patch, label in jobs}
        for k, fu in enumerate(concurrent.futures.as_completed(futs)):
            key, label = futs[fu]
            r = fu.result()
            r["label"] = label
            out[key] = r
            print(f"  {k + 1}/{len(jobs)} {r['source']:8s} {'RESOLVED' if r['resolved'] else 'no      '} f2p {r.get('f2p', '-'):>7s} p2p "
                  f"{r.get('p2p', '-'):>9s} parsed {r.get('parsed', 0):5d} {r.get('s', 0):4d} s"
                  f"{'' if label is None else ' · pool label ' + str(label)} · {key}{' · ' + r['note'].strip() if r['note'].strip() else ''}",
                  flush=True)
    json.dump(out, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    for src in ("rebench", "v2", "gym", "all"):
        rs = [r for r in out.values() if src in (r["source"], "all")]
        if not rs:
            continue
        line = (f"== {a.preds} · {src}: resolved {sum(r['resolved'] for r in rs)} / {len(rs)} · image or timeout errors "
                f"{sum(r['note'].startswith(('image', 'timeout')) for r in rs)} · patch did not apply "
                f"{sum('patch_failed' in r['note'] and 'test_patch_failed' not in r['note'] for r in rs)} · test patch did not apply "
                f"{sum('test_patch_failed' in r['note'] for r in rs)} · nothing parsed {sum(not r.get('parsed') for r in rs)} · second pass "
                f"after the install command {sum(bool(r.get('installed')) for r in rs)} (resolved "
                f"{sum(bool(r.get('installed')) and r['resolved'] for r in rs)})")
        if a.preds == "pool":
            lab = [r for r in rs if r["label"] is not None]
            line += (f" · agrees with the pool's label {sum(r['resolved'] == r['label'] for r in lab)} / {len(lab)} (label true → ours "
                     f"false {sum(r['label'] and not r['resolved'] for r in lab)}, false → true {sum(r['resolved'] and not r['label'] for r in lab)})")
        print(line)
    print(f"   {time.time() - t0:.0f} s · report {a.out}/report.json · logs {a.out}/logs/")


# ----------------------------------------------------------------------------------------------------------- report
def cmd_report(a):
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else 0  # noqa: E731
    graded = json.load(open(os.path.join(a.grade, "report.json"))) if a.grade and os.path.exists(os.path.join(a.grade, "report.json")) else {}
    exits, steps, think, peak, ptok, ctok, cached, rtok, wall, fits, usable = collections.Counter(), [], [], [], [], [], [], [], [], 0, 0
    n = 0
    for f in sorted(glob.glob(os.path.join(a.run, "*", "*.traj.json"))):
        t = json.load(open(f))
        n += 1
        iid = t.get("instance_id") or os.path.basename(os.path.dirname(f))
        ms = t.get("messages") or []
        asst = [m for m in ms if m.get("role") == "assistant"]
        exits[(t.get("info") or {}).get("exit_status")] += 1
        us = [((m.get("extra") or {}).get("response") or {}).get("usage") or {} for m in asst]
        ts = [(m.get("extra") or {}).get("timestamp") for m in asst if (m.get("extra") or {}).get("timestamp")]
        th = sum(1 for m in asst if (m.get("reasoning_content") or "").strip()) / max(len(asst), 1)
        pk = max([(u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0) for u in us] or [0])
        steps.append(len(asst)); think.append(th); peak.append(pk)
        ptok.append(sum(u.get("prompt_tokens") or 0 for u in us)); ctok.append(sum(u.get("completion_tokens") or 0 for u in us))
        cached.append(sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in us))
        rtok.append(sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in us))
        if len(ts) > 1:
            wall.append(max(ts) - min(ts))
        res = (graded.get(iid) or {}).get("resolved", False)
        fits += res and pk <= a.max_ctx
        usable += res and pk <= a.max_ctx and th >= 1.0
    if not n:
        raise SystemExit(f"no trajectories under {a.run}")
    nres = sum(r["resolved"] for r in graded.values())
    print(f"report · run {a.run} · grade {a.grade or '-'} · max_ctx {a.max_ctx} (teacher tokens, a proxy for Gemma tokens)")
    print(f"== trajectories {n} · exits {dict(exits)} · resolved {nres} / {len(graded)} graded · resolved ∧ fits {fits} · resolved ∧ "
          f"fits ∧ a thought in every turn {usable} ({usable / n:.2f} per attempt; rule ≥ 0.15)")
    print(f"   calls p50 {q(steps, .5)} p90 {q(steps, .9)} max {max(steps)} · turns with a thought: mean {100 * sum(think) / n:.0f} % "
          f"(rule ≥ 90 %), trajectories with one in every turn {sum(x >= 1 for x in think)} · peak context p50 {q(peak, .5):,} p90 "
          f"{q(peak, .9):,} · fits {sum(x <= a.max_ctx for x in peak)}")
    print(f"   tokens per run (median): prompt {q(ptok, .5):,} (cached {q(cached, .5):,}) · completion {q(ctok, .5):,} (reasoning "
          f"{q(rtok, .5):,}) · totals: prompt {sum(ptok):,} cached {sum(cached):,} completion {sum(ctok):,} · wall s p50 "
          f"{q(wall, .5):.0f} p90 {q(wall, .9):.0f}")
    if a.price:
        pi, pc, po = (float(x) for x in a.price.split(","))
        cost = ((sum(ptok) - sum(cached)) * pi + sum(cached) * pc + sum(ctok) * po) / 1e6
        print(f"   cost at {pi} / {pc} / {po} $ per M (input / cached / output): {cost:.2f} $ in all · {cost / n:.3f} $ per run"
              f"{f' · {cost / usable:.2f} $ per usable run' if usable else ''}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tasks"); s.add_argument("--pool", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("subset"); s.add_argument("--tasks", required=True); s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=0); s.add_argument("--per-source", type=int, default=0)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--sources", default=""); s.add_argument("--ids", default="")
    s = sub.add_parser("grade"); s.add_argument("--tasks", required=True); s.add_argument("--name", required=True)
    s.add_argument("--preds", required=True); s.add_argument("--out", required=True); s.add_argument("--pool", default="")
    s.add_argument("--workers", type=int, default=4); s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--pull-timeout", type=int, default=1800); s.add_argument("--net", default="")
    s.add_argument("--install", default="auto", choices=("auto", "always", "never"))
    s = sub.add_parser("report"); s.add_argument("--run", required=True); s.add_argument("--grade", default="")
    s.add_argument("--max-ctx", type=int, default=30720); s.add_argument("--price", default="")
    a = ap.parse_args()
    {"tasks": cmd_tasks, "subset": cmd_subset, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
