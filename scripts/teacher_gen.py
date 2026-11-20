#!/usr/bin/env python3
"""B-11: teacher trajectories by MetaCurate's method, generated inside the competition harness. The driver.

    python3 scripts/teacher_gen.py tasks  --table runs/b10/tasks/all.jsonl --out runs/teach [--exclude-repos FILE …]
    python3 scripts/teacher_gen.py gen    --out runs/teach --ids runs/teach/all.ids --name pilot30 [knobs below]
    python3 scripts/teacher_gen.py report --out runs/teach
    python3 scripts/teacher_gen.py inspect --out runs/teach [--name check6]   # thoughts, tools, seconds per call, shim lines
    python3 scripts/teacher_gen.py selftest                      # the bookkeeping, no harness, no Docker, no network

`tasks` turns the B-10 task table into what the harness reads: a tasks.jsonl with instance_id, repo, base_commit and
problem_statement ONLY (no test fields: the harness then runs the agent, keeps the patch and does not verify — the
grader does), and all.ids. Rows the table marks `skip`, and tasks of the repositories in --exclude-repos (one
`owner/name` per line: the evaluation tasks' repositories), are left out and counted.

`gen` works through the ids in chunks; per chunk:
  1. gold check    `rebench_mini.py grade --preds gold` (it pulls the chunk's images); only tasks whose gold patch
                   resolves go on. Done once per task (state `gold`).
  2. attempts      for attempt 1 … ATTEMPTS: the tasks that have no finished attempt of that number. For each one a
                   reservation of RESERVE_USD is taken from the shared ledger (scripts/teach_ledger.py); a task whose
                   reservation is refused is not started (state stays open, the run stops after this batch). The
                   batch runs through scripts/eval_dev.sh (the V21 files, the model replaced by the teacher through
                   MODELS_YAML, TEACH_ATTEMPT set, one call log line per model call in the ledger), its patches are
                   written as a predictions file and graded by `rebench_mini.py grade`.
  3. states        one line per (task, attempt) in <out>/attempts.jsonl, the last line wins:
                     done       the agent ran: resolved true / false, calls, patch size, cost from the ledger
                     infra      nothing usable came back for a reason that is not the model's: no model call at all,
                                a gateway / rate-limit / connection error, the budget gate, a container or image
                                error. Retried, at most RETRIES times, then `failed`.
                     failed     not retried any more (kept apart from `done` in every count)
  4. images        the chunk's images are removed (`docker rmi`) unless KEEP_IMAGES=1.
A second `gen` with the same --out continues: finished attempts and gold checks are skipped.

Knobs (flags or environment, every effective value is printed): CHUNK 50 · ATTEMPTS 3 · RETRIES 2 · CONC 4 ·
RESERVE_USD (per attempt; the pilot's p95) · TEMP 0.7 · THINK (CEIL_THINK form, e.g. effort:high) · SUBMISSION
runs/subs/v21_e4b · MODEL_ALIAS gemma-4-e4b-it-qat-w4a16-ct · MODELS_YAML runs/teach/models.yaml · MAX_CALLS 36 ·
TIME_MIN 7.5 · MAX_TURNS 80 · COMPACT_INTERVAL 5 · SANDBOX docker · KEEP_IMAGES 0 · MIN_FREE_GB 50 · TASKS_DIR
runs/b10/tasks (the grader's task directory). The budget itself lives in the ledger
(`teach_ledger.py init --dir <out>/ledger --budget … --price …`): this script never raises it.
Nothing here holds a gateway address, a key or a model name; MODELS_YAML and the key come from the box's environment.
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teach_ledger  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA_RE = re.compile(r"RateLimit|APIConnection|APIError|InternalServerError|ServiceUnavailable|Timeout(?!Exceeded)|BadGateway|"
                      r"AuthenticationError|BudgetReached|budget reached|ContainerSetupError|docker_task_shim|ImageNotFound|"
                      r"NotFound.*image|pull access denied|No such image|Snapshot file not found|Connection (refused|reset)", re.I)


def env_knob(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


# ---------- state ----------
class State:
    """attempts.jsonl (append only; the last line of a key wins) + gold.json"""
    def __init__(self, out):
        self.out = out; os.makedirs(out, exist_ok=True)
        self.path = os.path.join(out, "attempts.jsonl")
        self.att = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                if line.strip():
                    r = json.loads(line); self.att[(r["task"], r["attempt"])] = r
        g = os.path.join(out, "gold.json")
        self.gold = json.load(open(g)) if os.path.exists(g) else {}

    def put(self, rec):
        rec = dict(rec, t=round(time.time(), 1))
        self.att[(rec["task"], rec["attempt"])] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def set_gold(self, d):
        self.gold.update(d)
        tmp = os.path.join(self.out, "gold.json.tmp")
        json.dump(self.gold, open(tmp, "w")); os.replace(tmp, os.path.join(self.out, "gold.json"))

    def missing_traj(self):
        """done rows whose trajectory file is not on disk (2026-11-19: a second batch with the same name replaced the
        first one's directory — the rows stayed, the trajectories did not)"""
        return [r for r in self.att.values() if r["state"] == "done" and r.get("run")
                and not os.path.exists(os.path.join(r["run"], "traj", r["task"] + ".jsonl"))]

    def open_tasks(self, ids, attempt, retries):
        """tasks of this attempt number that still have to run: no line yet, or `infra` with tries ≤ retries"""
        out = []
        for i in ids:
            r = self.att.get((i, attempt))
            if r is None or r["state"] in ("lost", "reopened") or (r["state"] == "infra" and r.get("tries", 1) <= retries):
                out.append(i)
        return out


def classify(result, patch, per_key):
    """one harness result → (state, fields). `result` = the task's line of task_results.jsonl or None."""
    if result is None:
        return "infra", {"why": "no result line from the harness"}
    err = result.get("error") or ""
    if err.startswith("Missing test specification"):      # expected: the tasks file has no tests, the harness does not verify
        err = ""
    calls = result.get("total_llm_calls") or 0
    ledger_errs = (per_key or {}).get("errors", 0)
    if calls == 0:
        return "infra", {"why": ("no model call: " + err)[:200]}
    if INFRA_RE.search(err) and not patch:
        return "infra", {"why": err[:200]}
    return "done", {"llm_calls": calls, "tool_calls": result.get("tool_calls"), "patch_chars": len(patch or ""),
                    "harness_error": err[:200] or None, "ledger_errors": ledger_errs, "seconds": result.get("duration_seconds")}


# ---------- steps ----------
def cmd_tasks(a):
    excl = set()
    for f in a.exclude_repos or []:
        excl |= {x.strip().lower() for x in open(f) if x.strip() and not x.startswith("#")}
    n = collections.Counter(); keep = []
    src = {x for x in (getattr(a, "sources", None) or "").replace(",", " ").split() if x and x != "all"}
    for line in open(a.table):
        if not line.strip():
            continue
        r = json.loads(line); n["rows"] += 1
        if r.get("skip"):
            n["skip: " + str(r["skip"])[:40]] += 1; continue
        if (r.get("repo") or "").lower() in excl:
            n["excluded repository"] += 1; continue
        if src and str(r.get("source")) not in src:
            n["left out · source " + str(r.get("source"))] += 1; continue
        keep.append(r); n["kept · " + str(r.get("source"))] += 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tasks.jsonl"), "w") as f:
        for r in keep:
            f.write(json.dumps({"instance_id": r["instance_id"], "repo": r["repo"], "base_commit": r.get("base_commit") or "",
                                "problem_statement": r.get("problem_statement") or "", "hints_text": ""}) + "\n")
    open(os.path.join(a.out, "all.ids"), "w").write("\n".join(r["instance_id"] for r in keep) + "\n")
    print(f"teacher_gen tasks · sources {' '.join(sorted(src)) if src else 'all'} · table {a.table} · excluded repositories {len(excl)} from {a.exclude_repos or 'none'} → "
          f"{len(keep)} tasks in {a.out}/tasks.jsonl (4 fields, no tests) and all.ids")
    for k, v in sorted(n.items()):
        print(f"   {k:40} {v}")
    print(f"   distinct repositories kept {len({r['repo'] for r in keep})}")


def run(cmd, env=None, log=None):
    with open(log, "a") if log else open(os.devnull, "w") as f:
        f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def free_gb(path="."):
    return shutil.disk_usage(path).free / 1e9


def grade(a, name, ids, preds, outdir):
    """→ {id: resolved bool} through the grader (or its stand-in a.grader for tests)"""
    os.makedirs(a.tasks_dir, exist_ok=True)
    open(os.path.join(a.tasks_dir, name + ".ids"), "w").write("\n".join(ids) + "\n")
    os.makedirs(outdir, exist_ok=True)
    if a.grader:
        cmd = a.grader.format(tasks=a.tasks_dir, name=name, preds=preds, out=outdir).split()
    else:
        cmd = [sys.executable, os.path.join(HERE, "rebench_mini.py"), "grade", "--tasks", a.tasks_dir, "--name", name,
               "--preds", preds, "--out", outdir, "--workers", str(a.grade_workers), "--install", a.grade_install]
    rc = run(cmd, log=os.path.join(a.out, "gen.log"))
    rp = os.path.join(outdir, "report.json")
    rep = json.load(open(rp)) if os.path.exists(rp) else {}
    return {i: bool((rep.get(i) or {}).get("resolved")) for i in ids}, rep, rc


def harness(a, name, ids, attempt):
    """one eval_dev.sh batch → (results by id, patches by id, run dir)"""
    idf = os.path.join(a.out, "batches", name + ".ids"); os.makedirs(os.path.dirname(idf), exist_ok=True)
    open(idf, "w").write("\n".join(ids) + "\n")
    rundir = os.path.join("runs", f"dev33_base_{name}")
    shutil.rmtree(rundir, ignore_errors=True)
    env = dict(os.environ, LATEST="0", ARMS="base", SUBMISSION=a.submission, MODEL_ALIAS=a.model_alias, MODELS_YAML=a.models_yaml,
               MAX_CALLS=str(a.max_calls), TIME_MIN=str(a.time_min), MAX_TURNS=str(a.max_turns), COMPACT_INTERVAL=str(a.compact_interval),
               NO_TESTDEPS="1", SPLIT=idf, TAG=name, CONC=str(a.conc), TRAJ="1", SANDBOX=a.sandbox,
               CEIL_TEMP=str(a.temp), CEIL_THINK=a.think, CEIL_REMINDER=str(a.reminder), CEIL_HIST_THOUGHTS=str(a.hist_thoughts), CEIL_HIST_PROBE=str(a.hist_thoughts), TEACH_LEDGER_DIR=os.path.abspath(os.path.join(a.out, "ledger")),
               TEACH_ATTEMPT=str(attempt), CEIL_CALL_LOG=os.path.abspath(os.path.join(a.out, "calls", name + ".jsonl")))
    os.makedirs(os.path.join(a.out, "calls"), exist_ok=True)
    if a.sandbox == "docker":
        env.update(TASKS_FILE=os.path.join(a.out, "tasks.jsonl"), SNAPSHOTS_DIR=os.path.join(a.out, "no_snapshots"), DOCKER_TASKS=a.table)
        os.makedirs(os.path.join(a.out, "no_snapshots"), exist_ok=True)
    else:                                       # tests on the laptop: the harness's own tasks and snapshots
        for k in ("TASKS_FILE", "SNAPSHOTS_DIR"):
            if getattr(a, k.lower(), None):
                env[k] = getattr(a, k.lower())
    rc = run(["bash", os.path.join(HERE, "eval_dev.sh")], env=env, log=os.path.join(a.out, "gen.log"))
    res = {}
    rp = os.path.join(rundir, "task_results.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            if line.strip():
                r = json.loads(line); res[r["instance_id"]] = r
    patches = {}
    for f in glob.glob(os.path.join(rundir, "patches", "*.patch")):
        patches[os.path.basename(f)[:-6]] = open(f, errors="replace").read()
    patches = {i: patches.get(i) or patches.get(i.replace("/", "__")) or "" for i in ids}
    keep = os.path.join(a.out, "runs", name)          # trajectories and patches stay with the run's own state
    if os.path.exists(keep):                          # batch_name() makes this impossible; never delete a kept run
        raise RuntimeError(f"{keep} exists — a kept run is never replaced")
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    if os.path.isdir(rundir):
        shutil.move(rundir, keep)
    return res, patches, keep, rc


def batch_name(out, base):
    """a name no earlier batch of this output directory has used: base, base_n2, base_n3, …"""
    n = 1
    while True:
        name = base if n == 1 else f"{base}_n{n}"
        if not any(os.path.exists(p) for p in (os.path.join(out, "runs", name), os.path.join(out, "batches", name + ".ids"),
                                               os.path.join(out, "preds", name + ".json"))):
            return name
        n += 1


def cmd_gen(a):
    st = State(a.out); L = teach_ledger.Ledger(os.path.join(a.out, "ledger"))
    ids = [x for x in open(a.ids).read().split() if x]
    knobs = {k: getattr(a, k) for k in ("chunk", "attempts", "retries", "conc", "reserve_usd", "temp", "think", "submission", "model_alias",
                                        "models_yaml", "max_calls", "time_min", "max_turns", "compact_interval", "sandbox", "keep_images",
                                        "min_free_gb", "tasks_dir", "table", "grader", "reopen_lost", "reopen_failed", "pull_waits", "pull_wait_min", "reminder", "hist_thoughts")}
    print(f"teacher_gen gen · name {a.name} · {len(ids)} tasks from {a.ids} · out {a.out}")
    print("   knobs: " + " · ".join(f"{k} {v}" for k, v in knobs.items()))
    print("   " + teach_ledger.fmt_status(L.status()))
    if L.status()["budget_usd"] is None:
        sys.exit("no budget in the ledger: python3 scripts/teach_ledger.py init --dir <out>/ledger --budget <usd> [--price in,cached,out]")
    if a.reserve_usd <= 0:
        sys.exit("RESERVE_USD must be > 0 (the cost one attempt may reach; the pilot's p95)")
    # one output directory = one way of generating: attempts made with and without the reminder are never mixed
    rem = "on" if str(a.reminder) not in ("", "0") else "off"
    other = [r for r in st.att.values() if r["state"] == "done" and r.get("reminder", "off") != rem]
    if other:
        sys.exit(f"STOP: {len(other)} done attempts in {a.out} were made with the reminder {other[0].get('reminder', 'off')}, this run has it {rem}. "
                 f"Use another OUT for the other setting.")
    hist = "on" if str(a.hist_thoughts) not in ("", "0") else "off"
    other = [r for r in st.att.values() if r["state"] == "done" and r.get("hist", "off") != hist]
    if other:
        sys.exit(f"STOP: {len(other)} done attempts in {a.out} were made with the teacher's earlier thoughts in its history {other[0].get('hist', 'off')}, "
                 f"this run has them {hist}. Use another OUT for the other setting.")
    lost = st.missing_traj()
    print(f"   done rows whose trajectory file is missing: {len(lost)}" + (f" → reopened (--reopen-lost): they run again and are paid again" if lost and a.reopen_lost
          else " (they stay 'done' and have no data; --reopen-lost / REOPEN_LOST=1 runs them again)" if lost else ""))
    if a.reopen_lost:
        for r in lost:
            st.put(dict(r, state="lost", resolved=None, why="trajectory file missing"))
    failed = [r for r in st.att.values() if r["state"] == "failed" and r["task"] in set(ids)]
    print(f"   failed rows among these tasks (not retried): {len(failed)}" + (" → reopened (--reopen-failed)" if failed and a.reopen_failed
          else " (REOPEN_FAILED=1 runs them again, e.g. rows left by a run whose environment was broken)" if failed else ""))
    if a.reopen_failed:
        for r in failed:
            st.put(dict(r, state="reopened", tries=0, resolved=None))
    stop = False
    for c0 in range(0, len(ids), a.chunk):
        chunk = ids[c0:c0 + a.chunk]; cname = f"{a.name}_c{c0 // a.chunk:03d}"
        if free_gb() < a.min_free_gb:
            print(f"STOP: {free_gb():.0f} GB free < MIN_FREE_GB {a.min_free_gb}"); break
        todo = [i for i in chunk if i not in st.gold]
        if todo:
            gname = batch_name(a.out, cname + "_gold")
            open(os.path.join(a.out, "batches", gname + ".ids"), "w").write("\n".join(todo) + "\n")
            ok, rep, rc = grade(a, gname, todo, "gold", os.path.join(a.out, "grade", gname))
            # an image that could not be pulled (Docker Hub's pull limit, a network error) says nothing about the task:
            # such a task gets no gold result; wait and try again, and what is still missing is tried by the next run
            noimg = lambda rep_, ids_: [i for i in ids_ if str((rep_.get(i) or {}).get("note", "no result")).startswith(("image", "no result"))]
            bad = noimg(rep, todo); waits = 0
            while bad and waits < a.pull_waits:
                waits += 1
                print(f"-- chunk {cname}: {len(bad)} images not pulled ({str((rep.get(bad[0]) or {}).get('note'))[:110]}) → waiting {a.pull_wait_min} min, "
                      f"try {waits} of {a.pull_waits} (PULL_WAIT_MIN, PULL_WAITS)", flush=True)
                time.sleep(a.pull_wait_min * 60)
                gname = batch_name(a.out, cname + "_gold")
                open(os.path.join(a.out, "batches", gname + ".ids"), "w").write("\n".join(bad) + "\n")
                ok2, rep2, rc = grade(a, gname, bad, "gold", os.path.join(a.out, "grade", gname))
                ok.update(ok2); rep.update(rep2); bad = noimg(rep2, bad)
            st.set_gold({i: v for i, v in ok.items() if i not in set(bad)})
            why = collections.Counter(str((rep.get(i) or {}).get("note") or "tests fail")[:50] for i in todo if not ok.get(i) and i not in set(bad))
            print(f"-- chunk {cname}: gold check on {len(todo)} tasks → {sum(ok.values())} resolve (grader exit {rc}) · no image, no result recorded {len(bad)} · "
                  f"dropped: {dict(why.most_common(5)) or 'none'}", flush=True)
        good = [i for i in chunk if st.gold.get(i)]
        for att in range(1, a.attempts + 1):
            while True:                      # ends: nothing open (infra rows close after `retries`), or nothing can be reserved
                want = st.open_tasks(good, att, a.retries)
                if not want:
                    break
                batch = [i for i in want if L.reserve(f"{i}#{att}", a.reserve_usd)]
                if not batch:
                    stop = True; break
                if len(batch) < len(want):
                    print(f"   {cname} attempt {att}: {len(want) - len(batch)} of {len(want)} wait — their reservation does not fit beside this batch's; "
                          f"they start when it ends and its unspent reservations are back", flush=True)
                name = batch_name(a.out, f"{cname}_a{att}")
                t0 = time.time()
                try:
                    res, patches, rundir, rc = harness(a, name, batch, att)
                    preds = os.path.join(a.out, "preds", name + ".json"); os.makedirs(os.path.dirname(preds), exist_ok=True)
                    json.dump({i: {"model_patch": patches[i]} for i in batch}, open(preds, "w"))
                    per = L.per_key(); done = []
                    states = {}
                    for i in batch:
                        s, f = classify(res.get(i), patches[i], per.get(f"{i}#{att}"))
                        states[i] = (s, f); done += [i] if s == "done" else []
                    resolved = {}
                    if done:
                        resolved, rep, _ = grade(a, name, done, preds, os.path.join(a.out, "grade", name))
                    for i in batch:
                        s, f = states[i]; k = per.get(f"{i}#{att}", {})
                        prev = st.att.get((i, att)); tries = (prev.get("tries", 1) + 1) if prev and prev["state"] == "infra" else 1
                        if s == "infra" and tries > a.retries:
                            s = "failed"
                        st.put(dict(f, task=i, attempt=att, state=s, tries=tries, resolved=resolved.get(i) if s == "done" else None,
                                    usd=round(k.get("usd", 0.0), 6), tok_in=k.get("in", 0), tok_cached=k.get("cached", 0),
                                    tok_out=k.get("out", 0), tok_reasoning=k.get("reasoning", 0), model_calls=k.get("calls", 0),
                                    est_calls=k.get("est_calls", 0), run=rundir, reminder=rem, hist=hist))
                finally:
                    for i in batch:
                        L.release(f"{i}#{att}")
                c = collections.Counter(st.att[(i, att)]["state"] for i in batch)
                print(f"   {name}: {len(batch)} started of {len(want)} wanted · done {c['done']} (resolved {sum(bool(st.att[(i, att)].get('resolved')) for i in batch)}) · "
                      f"infra {c['infra']} · failed {c['failed']} · {time.time() - t0:.0f} s · harness exit {rc} · {teach_ledger.fmt_status(L.status())}", flush=True)
            if stop:
                break
        if not a.keep_images and a.sandbox == "docker":
            imgs = [json.loads(l)["image_name"] for l in open(a.table) if l.strip() and json.loads(l)["instance_id"] in set(chunk) and json.loads(l).get("image_name")]
            if imgs:
                run(["docker", "rmi", "-f"] + imgs, log=os.path.join(a.out, "gen.log"))
                print(f"   {cname}: {len(imgs)} images removed (KEEP_IMAGES=1 keeps them) · {free_gb():.0f} GB free")
        if stop:
            print("STOP: the ledger refused a reservation — spent + reserved would pass the budget. No new attempt was started. "
                  "Raise the budget only on the user's word (teach_ledger.py init), then run the same command again: it continues.")
            break
    cmd_report(a)


def cmd_report(a):
    st = State(a.out); L = teach_ledger.Ledger(os.path.join(a.out, "ledger"))
    rows = list(st.att.values())
    by = collections.Counter(r["state"] for r in rows)
    done = [r for r in rows if r["state"] == "done"]; res = [r for r in done if r.get("resolved")]
    q = lambda xs, p: (sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else None)
    usd = [r["usd"] for r in done]
    print(f"== teacher_gen report · {a.out}")
    print(f"   gold check: {len(st.gold)} tasks checked · {sum(st.gold.values())} resolve · resolved trajectories on disk "
          f"{sum(1 for r in rows if r['state'] == 'done' and r.get('resolved')) - sum(1 for r in st.missing_traj() if r.get('resolved'))}")
    print(f"   attempts: {len(rows)} · done {by['done']} · infra (to retry) {by['infra']} · failed (not retried) {by['failed']} · lost (to run again) {by['lost']} · "
          f"done rows whose trajectory file is missing {len(st.missing_traj())}")
    print(f"   done: resolved {len(res)} = {100 * len(res) / max(1, len(done)):.1f} % · tasks with ≥ 1 resolved {len({r['task'] for r in res})} of {len({r['task'] for r in done})}")
    for att in sorted({r["attempt"] for r in rows}):
        d = [r for r in done if r["attempt"] == att]
        print(f"      attempt {att}: done {len(d)} · resolved {sum(bool(r.get('resolved')) for r in d)}")
    if done:
        print(f"   per done attempt: model calls p50 {q([r.get('llm_calls') or 0 for r in done], .5)} p90 {q([r.get('llm_calls') or 0 for r in done], .9)} · "
              f"tool calls p50 {q([r.get('tool_calls') or 0 for r in done], .5)} p90 {q([r.get('tool_calls') or 0 for r in done], .9)} · "
              f"seconds p50 {q([r.get('seconds') or 0 for r in done], .5):.0f} p90 {q([r.get('seconds') or 0 for r in done], .9):.0f} · "
              f"empty patch {sum(not r.get('patch_chars') for r in done)} · ended by a harness limit {sum(bool(r.get('harness_error')) for r in done)}")
        print(f"   cost per done attempt: mean ${sum(usd) / len(usd):.4f} · p50 ${q(usd, .5):.4f} · p95 ${q(usd, .95):.4f} · max ${max(usd):.4f} "
              f"(→ RESERVE_USD for the scale) · per resolved trajectory ${sum(r['usd'] for r in rows) / max(1, len(res)):.4f}")
        tot = {k: sum(r.get(k, 0) for r in rows) for k in ("tok_in", "tok_cached", "tok_out", "tok_reasoning", "model_calls", "est_calls")}
        print(f"   tokens (all attempts): input {tot['tok_in']:,} of which cached {tot['tok_cached']:,} · output {tot['tok_out']:,} of which reasoning {tot['tok_reasoning']:,} · "
              f"model calls {tot['model_calls']:,} · priced by estimate {tot['est_calls']:,}")
    why = collections.Counter((r.get("why") or "")[:60] for r in rows if r["state"] in ("infra", "failed"))
    if why:
        print("   infra / failed reasons: " + " · ".join(f"{v}× {k!r}" for k, v in why.most_common(6)))
    print("   " + teach_ledger.fmt_status(L.status()))


def cmd_inspect(a):
    """what the trajectory logs of the kept runs show (step 2 and the pilot): thoughts, tools, time per call, compaction"""
    st = State(a.out)
    runs = sorted({r["run"] for r in st.att.values() if r.get("run") and (not a.name or os.path.basename(r["run"]).startswith(a.name))})
    calls = thought = 0; tools = collections.Counter(); fin = collections.Counter(); secs = []; per_task = []; resets = collections.Counter()
    thought_chars = []; multi = 0; flags = {}; rem_lines = 0; back = [0, 0]; by_depth = collections.defaultdict(lambda: [0, 0])
    for run_ in runs:
        for f in sorted(glob.glob(os.path.join(run_, "traj", "*.jsonl"))):
            if f.endswith(".tools.jsonl"):
                continue
            n = t = 0; seq = []
            for line in open(f):
                rem_lines += "<reminder>" in line
                c = json.loads(line); r = c.get("resp") or {}
                # the previous reply as it comes back in this request: does its thought come with it?
                if seq and seq[-1] and not c.get("reset"):
                    am = [m for m in c.get("new") or [] if m.get("role") == "assistant"]
                    if am:
                        back[1] += 1
                        back[0] += bool(am[0].get("reasoning_content") or am[0].get("reasoning") or am[0].get("thinking_blocks"))
                seq.append(bool(r.get("reasoning")))
                d = by_depth["01-03" if n < 3 else "04-08" if n < 8 else "09-25" if n < 25 else "26+"]; d[0] += 1; d[1] += bool(r.get("reasoning"))
                n += 1; t += bool(r.get("reasoning")); thought_chars.append(len(r.get("reasoning") or ""))
                tc = r.get("tool_calls") or []; multi += len(tc) > 1
                for x in tc:
                    tools[x.get("name")] += 1
                fin[r.get("finish_reason")] += 1; secs.append(c.get("s") or 0)
                if c.get("reset"):
                    resets[str(c["reset"])[:30]] += 1
            calls += n; thought += t; per_task.append((os.path.basename(f)[:-6], n, t))
            flags[os.path.basename(f)[:-6]] = seq
    q = lambda xs, p: (sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else 0)
    print(f"== teacher_gen inspect · {a.out} · runs {len(runs)} ({a.name or 'all'}) · trajectories {len(per_task)} · model calls {calls}")
    print(f"   thoughts: {thought} of {calls} calls = {100 * thought / max(1, calls):.1f} %  (no gate on this share since 2026-11-18; the gate is the 'reasoning tokens reported' line below) · chars per thought p50 {q(thought_chars, .5)} p90 {q(thought_chars, .9)} max {max(thought_chars or [0])}")
    print(f"   tools called: {dict(tools.most_common())} · calls with several tool calls {multi} · finish reasons {dict(fin)}")
    print(f"   seconds per model call: p50 {q(secs, .5):.1f} · p90 {q(secs, .9):.1f} · max {max(secs or [0]):.1f} · history resets / compactions {dict(resets) or 'none'}")
    # where the missing thoughts are lost: the ledger's line of the same call (its order within the task) has the output
    # and reasoning token counts. No thought text but many output / reasoning tokens = produced and dropped on the way;
    # few output tokens = the teacher did not think on that call.
    led = collections.defaultdict(list); lf = os.path.join(a.out, "ledger", "calls.jsonl")
    for line in (open(lf) if os.path.exists(lf) else []):
        c = json.loads(line)
        if not c.get("err"):
            led[c["key"].split("#")[0]].append(c)
    cell = {True: [], False: []}; unmatched = []
    for name, seq in flags.items():
        if len(led.get(name, [])) != len(seq):
            unmatched.append(f"{name} {len(seq)} vs {len(led.get(name, []))}"); continue
        for has, c in zip(seq, led[name]):
            cell[has].append(c)
    for has in (True, False):
        cs = cell[has]; outs = [c.get("out") or 0 for c in cs]; rs = [c.get("reasoning") or 0 for c in cs]
        print(f"   calls {'with   ' if has else 'without'} thought text: {len(cs):4} · output tokens p50 {q(outs, .5)} p90 {q(outs, .9)} · "
              f"reasoning tokens reported > 0 on {sum(x > 0 for x in rs)} (p50 {q(rs, .5)})"
              + (f"  ← GATE: must be 0 (a thought that was produced and is not in the log) → {'ok' if not sum(x > 0 for x in rs) else 'NOT MET'}" if not has else ""))
    print("   thoughts by call number: " + " · ".join(f"{k} {100 * v[1] / max(1, v[0]):.0f} % of {v[0]}" for k, v in sorted(by_depth.items())))
    made_h = collections.Counter(r.get("hist", "off") for r in st.att.values() if r["state"] == "done")
    want_all = set(made_h) == {"on"}
    print(f"   history: done attempts made with the earlier thoughts kept {dict(made_h)} · replies with a thought that come back WITH it in the next request: "
          f"{back[0]} of {back[1]}" + (f"  ← GATE: all → {'ok' if back[0] == back[1] and back[1] else 'NOT MET'}" if want_all else "  (no gate: the setting is off)"))
    shapes = collections.Counter(); sent = [0, 0]
    for run_ in runs:
        cf = os.path.join(a.out, "calls", os.path.basename(run_) + ".jsonl")
        for line in (open(cf) if os.path.exists(cf) else []):
            c = json.loads(line)
            if c.get("think_shape"):
                shapes[c["think_shape"].split(" · reasoning_content")[0]] += 1
            h = c.get("hist") or {}; sent[0] += h.get("with_thought", 0); sent[1] += h.get("assistant_msgs", 0)
    print(f"   as sent to the gateway (call log): earlier assistant messages with a thought {sent[0]} of {sent[1]} · reply shapes: {dict(shapes.most_common(4)) or 'not logged'}")
    made = collections.Counter(r.get("reminder", "off") for r in st.att.values() if r["state"] == "done")
    print(f"   reminder: done attempts made with it {dict(made)} · trajectory log lines that contain '<reminder>': {rem_lines}  ← GATE: must be 0 "
          f"(the line is for the teacher's request only; no training prompt may hold it) → {'ok' if not rem_lines else 'NOT MET'}")
    print(f"   (matched to the ledger by call order; trajectories whose call count differs from the ledger's, left out: {unmatched or 'none'})")
    for name, n, t in per_task[:40]:
        r = next((x for x in st.att.values() if x["task"] == name), {})
        print(f"      {name:44} calls {n:3} · with thought {t:3} · {r.get('state')} · resolved {r.get('resolved')} · tool calls {r.get('tool_calls')} · "
              f"patch {r.get('patch_chars')} chars · {r.get('seconds') or 0:.0f} s · ${r.get('usd', 0):.4f}{' (estimate)' if r.get('est_calls') else ''}")
    log = os.path.join(a.out, "gen.log")
    if os.path.exists(log):
        L = open(log, errors="replace").read().splitlines()
        for pat in ("docker_task_shim:", "ceil adapter:", "compaction: on", "context cap:"):
            hits = [x for x in L if pat in x]
            for x in (hits[:1] if pat != "docker_task_shim:" else hits[:14]):
                print("   log · " + x.strip()[:330])


def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="teacher_gen_")
    try:
        # tasks: skip rows and excluded repositories
        tab = os.path.join(d, "all.jsonl")
        rows = [{"instance_id": "a__b-1", "repo": "a/b", "source": "rebench", "base_commit": "c1", "problem_statement": "p", "FAIL_TO_PASS": ["t"], "image_name": "i1"},
                {"instance_id": "a__b-2", "repo": "a/b", "source": "rebench", "skip": "no test_cmd"},
                {"instance_id": "x__y-3", "repo": "X/Y", "source": "v2", "base_commit": "c3", "problem_statement": "q", "image_name": "i3"},
                {"instance_id": "g__h-4", "repo": "g/h", "source": "gym", "base_commit": "c4", "problem_statement": "r", "image_name": "i4"}]
        open(tab, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        ex = os.path.join(d, "ex.txt"); open(ex, "w").write("x/y\n")
        cmd_tasks(argparse.Namespace(table=tab, out=os.path.join(d, "o"), exclude_repos=[ex], sources="all"))
        assert open(os.path.join(d, "o", "all.ids")).read().split() == ["a__b-1", "g__h-4"]
        cmd_tasks(argparse.Namespace(table=tab, out=os.path.join(d, "o"), exclude_repos=[ex], sources="rebench,v2"))
        T = [json.loads(l) for l in open(os.path.join(d, "o", "tasks.jsonl"))]
        assert [t["instance_id"] for t in T] == ["a__b-1"] and set(T[0]) == {"instance_id", "repo", "base_commit", "problem_statement", "hints_text"}, T
        # classify
        assert classify(None, "", None)[0] == "infra"
        assert classify({"total_llm_calls": 0, "error": "x"}, "", None)[0] == "infra"
        assert classify({"total_llm_calls": 3, "error": "litellm.RateLimitError: slow down"}, "", None)[0] == "infra"
        assert classify({"total_llm_calls": 3, "error": "teach_ledger: budget reached (spent …)"}, "", None)[0] == "infra"
        assert classify({"total_llm_calls": 9, "error": "Agent exceeded turns budget (80 turns)", "tool_calls": 36}, "diff", None)[0] == "done"
        assert classify({"total_llm_calls": 9, "error": None, "tool_calls": 7}, "", None)[0] == "done"       # an empty patch is the model's result
        assert classify({"total_llm_calls": 9, "error": "TimeoutExceeded: Command timed out"}, "", None)[0] == "done"
        # state: last line wins, retries, resume
        S = State(os.path.join(d, "s"))
        S.put({"task": "t1", "attempt": 1, "state": "infra", "tries": 1}); S.put({"task": "t2", "attempt": 1, "state": "done", "resolved": True})
        assert S.open_tasks(["t1", "t2", "t3"], 1, 2) == ["t1", "t3"]
        S.put({"task": "t1", "attempt": 1, "state": "infra", "tries": 3})
        assert S.open_tasks(["t1", "t2", "t3"], 1, 2) == ["t3"]                                               # tries > retries: left alone
        S2 = State(os.path.join(d, "s"))
        assert S2.att[("t1", 1)]["tries"] == 3 and S2.open_tasks(["t2"], 1, 2) == [] and S2.open_tasks(["t2"], 2, 2) == ["t2"]
        print("teacher_gen selftest: task conversion (skip rows, excluded repositories, sources kept, no test fields), 7 result classes, "
              "state file (last line wins, retry limit, resume) → all as expected")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tasks"); s.add_argument("--table", required=True); s.add_argument("--out", required=True)
    s.add_argument("--exclude-repos", action="append")
    s.add_argument("--sources", default="rebench,v2", help="task sources kept (the table's `source` field); `all` keeps every source")
    for name in ("gen", "report", "inspect"):
        s = sub.add_parser(name); s.add_argument("--out", required=True)
        s.add_argument("--ids"); s.add_argument("--name", default="gen" if name == "gen" else "")
        s.add_argument("--table", default=env_knob("DOCKER_TASKS_TABLE", "runs/b10/tasks/all.jsonl"))
        s.add_argument("--tasks-dir", default=env_knob("TASKS_DIR", "runs/b10/tasks"))
        s.add_argument("--chunk", type=int, default=env_knob("CHUNK", 50, int))
        s.add_argument("--attempts", type=int, default=env_knob("ATTEMPTS", 3, int))
        s.add_argument("--retries", type=int, default=env_knob("RETRIES", 2, int))
        s.add_argument("--conc", type=int, default=env_knob("CONC", 4, int))
        s.add_argument("--reserve-usd", type=float, default=env_knob("RESERVE_USD", 0.0, float))
        s.add_argument("--temp", type=float, default=env_knob("TEMP", 0.7, float))
        s.add_argument("--think", default=env_knob("THINK", "effort:high"))
        s.add_argument("--submission", default=env_knob("SUBMISSION", "runs/subs/v21_e4b"))
        s.add_argument("--model-alias", default=env_knob("MODEL_ALIAS", "gemma-4-e4b-it-qat-w4a16-ct"))
        s.add_argument("--models-yaml", default=env_knob("MODELS_YAML", "runs/teach/models.yaml"))
        s.add_argument("--max-calls", type=int, default=env_knob("MAX_CALLS", 36, int))
        s.add_argument("--time-min", type=float, default=env_knob("TIME_MIN", 7.5, float))
        s.add_argument("--max-turns", type=int, default=env_knob("MAX_TURNS", 80, int))
        s.add_argument("--compact-interval", type=int, default=env_knob("COMPACT_INTERVAL", 5, int))
        s.add_argument("--sandbox", default=env_knob("SANDBOX", "docker"), choices=["docker", "subprocess"])
        s.add_argument("--keep-images", type=int, default=env_knob("KEEP_IMAGES", 0, int))
        s.add_argument("--hist-thoughts", default=env_knob("HIST_THOUGHTS", "1", str), help="CEIL_HIST_THOUGHTS: 1 the teacher's earlier thoughts stay in its history, as the student's do · 0 the old behaviour")
        s.add_argument("--reminder", default=env_knob("REMINDER", "0", str), help="CEIL_REMINDER: 0 off · 1 the default think-before-every-call line · other = that text")
        s.add_argument("--reopen-failed", type=int, default=env_knob("REOPEN_FAILED", 0, int), help="1 = failed rows run again")
        s.add_argument("--pull-waits", type=int, default=env_knob("PULL_WAITS", 16, int)); s.add_argument("--pull-wait-min", type=float, default=env_knob("PULL_WAIT_MIN", 30, float))
        s.add_argument("--reopen-lost", type=int, default=env_knob("REOPEN_LOST", 0, int), help="1 = done rows without a trajectory file run again")
        s.add_argument("--min-free-gb", type=float, default=env_knob("MIN_FREE_GB", 50.0, float))
        s.add_argument("--grade-workers", type=int, default=env_knob("GRADE_WORKERS", 4, int))
        s.add_argument("--grade-install", default=env_knob("GRADE_INSTALL", "auto"))
        s.add_argument("--grader", default=env_knob("GRADER", ""), help="tests only: a stand-in command with {tasks} {name} {preds} {out}")
        s.add_argument("--tasks-file", default=env_knob("TASKS_FILE", "")); s.add_argument("--snapshots-dir", default=env_knob("SNAPSHOTS_DIR", ""))
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "gen" and not a.ids:
        ap.error("gen needs --ids")
    {"tasks": cmd_tasks, "gen": cmd_gen, "report": cmd_report, "inspect": cmd_inspect}[a.cmd](a)


if __name__ == "__main__":
    main()
