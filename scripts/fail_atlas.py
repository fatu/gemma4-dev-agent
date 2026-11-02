#!/usr/bin/env python3
"""Failure atlas, step S1 of the failure study (notes/16): one row per task of a run, every objective fact about how it
failed. No API calls. Rows go to runs/fail_atlas/<run name>.tsv (gitignored: task ids + paths, no task text).

    python3 scripts/fail_atlas.py runs/pod_all129_e2b/runs/dev33_base_all129_e2b [more runs …]
    REFS="31b=runs/pod_all129/runs/dev33_base_all129_31b opus=runs/dev33_base_ceilsc" python3 scripts/fail_atlas.py <run>

Columns:
  task repo outcome (✓ ✗ O T C E ∅, flips.outcome) · label difficulty env_defect (runs/labels + runs/gc_all) ·
  rulestats.analyse(): calls first_edit reads_before edit_ok edit_mismatch edit_missing edit_loop heredoc_edits tested
    test_calls first_test submit_at test_before_submit duration patch ·
  traj/<task>.jsonl (if present): llm_calls len_cuts (finish_reason=length) empty_replies (no tool call and no text)
    compactions peak_prompt ·
  patch vs gold (harness/tasks.jsonl patch, <run>/patches/<task>.patch): gold_files files_hit (none/some/all)
    extra_files hunk_near (model hunks within ±10 lines of a gold hunk in a gold file: none/some/all) test_files_edited
    clobber (hunks that drop ≥ 100 lines and keep < 1/4: a source file overwritten by a fragment) ·
  refs: one column per REFS name, ✓/✗/– (resolved there, failed there, not run).
Ends with a one-screen summary: outcome split, env_defect excluded, gold-file hit rates by outcome.
"""
import json
import os
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from flips import outcome                    # noqa: E402
from rulestats import load, analyse          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "fail_atlas"
GC = ROOT / os.environ.get("GC", "runs/gc_all/all.jsonl")
NEAR = 10
TESTFILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$|pytest\.ini$)")


def hunks(patch):
    """{file: [(old_start, old_end), …]} from a unified diff (gold patches have only ---/+++ headers, no diff --git)"""
    out, cur, old = {}, None, None
    for line in (patch or "").splitlines():
        if line.startswith("--- "):
            old = line[4:].split("\t")[0].removeprefix("a/")
        elif line.startswith("+++ "):
            new = line[4:].split("\t")[0].removeprefix("b/")
            cur = old if new == "/dev/null" else new          # a deleted file keeps its old name
            out.setdefault(cur, [])
        elif line.startswith("@@") and cur:
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            if m:
                a, n = int(m.group(1)), int(m.group(2) or 1)
                out[cur].append((a, a + max(n, 1) - 1))
    return out


def degree(hit, total):
    return "none" if hit == 0 else ("all" if hit == total else "some")


def vs_gold(gold, mine):
    g, m = hunks(gold), hunks(mine)
    gsrc = {f: h for f, h in g.items() if not TESTFILE.search(f)}
    msrc = {f: h for f, h in m.items() if not TESTFILE.search(f)}
    hit = [f for f in gsrc if f in msrc]
    near = 0
    for f in hit:
        if any(ga - NEAR <= ma <= gb + NEAR or ga - NEAR <= mb <= gb + NEAR or (ma <= ga and mb >= gb)
               for ma, mb in msrc[f] for ga, gb in gsrc[f]):
            near += 1
    clob = 0                                   # file clobber: a hunk that drops ≥ 100 lines and keeps < 1/4 of them
    for mm in re.finditer(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", mine or "", re.M):
        o, n = int(mm.group(1) or 1), int(mm.group(2) or 1)
        clob += o >= 100 and n < o / 4
    return dict(gold_files=len(gsrc), files_hit=degree(len(hit), len(gsrc)) if m else "-",
                extra_files=len([f for f in msrc if f not in gsrc]),
                hunk_near=degree(near, len(gsrc)) if m else "-",
                test_files_edited=len([f for f in m if TESTFILE.search(f)]), clobber=clob)


def traj_facts(path):
    if not path.exists():
        return dict(llm_calls="-", len_cuts="-", empty_replies="-", compactions="-", peak_prompt="-")
    calls = cuts = empty = comp = peak = 0
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("compaction"):
            comp += 1
            continue
        calls += 1
        resp, use = r.get("resp") or {}, r.get("usage") or {}
        cuts += resp.get("finish_reason") == "length"
        empty += not resp.get("tool_calls") and not (resp.get("content") or "").strip() and not r.get("err")
        peak = max(peak, use.get("prompt") or 0)
    return dict(llm_calls=calls, len_cuts=cuts, empty_replies=empty, compactions=comp, peak_prompt=peak)


def resolved_set(run):
    res, _ = load(pathlib.Path(run))
    return {i: bool(r.get("resolved")) for i, r in res.items()}


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    tasks = {t["instance_id"]: t for t in map(json.loads, open(ROOT / "harness" / "tasks.jsonl"))}
    gc = {r["instance_id"]: r for r in map(json.loads, open(GC))} if GC.exists() else {}
    refs = dict(x.split("=", 1) for x in os.environ.get("REFS", "").split())
    ref_res = {n: resolved_set(ROOT / p if not pathlib.Path(p).is_absolute() else p) for n, p in refs.items()}
    print(f"fail_atlas: goldcheck {GC.relative_to(ROOT) if gc else 'MISSING'} ({len(gc)}) · labels runs/labels · "
          f"near = ±{NEAR} lines · refs {', '.join(f'{n}={p}' for n, p in refs.items()) or 'none'}")
    OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        results, traces = load(run)
        rows = []
        for i in sorted(results):
            r = results[i]
            lab_p = ROOT / "runs" / "labels" / f"{i}.json"
            lab = json.loads(lab_p.read_text()) if lab_p.exists() else {}
            env = bool(gc.get(i)) and not gc[i]["resolved"]
            f, d = analyse(traces.get(i, []), r)
            p = run / "patches" / f"{i}.patch"
            row = dict(task=i, repo=i.rsplit("_", 1)[0], outcome=outcome(r), label=lab.get("primary", "-"),
                       difficulty=lab.get("difficulty", "-"), env_defect=int(env), calls=f["calls"],
                       first_edit=d["first_edit"] or "-", reads_before=d["reads_before"], edit_ok=f["edit_ok"],
                       edit_mismatch=f["edit_mismatch"], edit_missing=f["edit_missing"], edit_loop=f["edit_loop"],
                       heredoc_edits=f["heredoc_edits"], tested=f["tested"], test_calls=f["test_calls"],
                       first_test=d["first_test"] or "-", submit_at=d["submit_at"] or "-",
                       test_before_submit=f["test_before_submit"], duration=round(float(d["duration"] or 0)),
                       patch=d["patch"])
            row.update(traj_facts(run / "traj" / f"{i}.jsonl"))
            row.update(vs_gold(tasks.get(i, {}).get("patch", ""), p.read_text() if p.exists() else ""))
            for n, rs in ref_res.items():
                row[n] = "–" if i not in rs else ("✓" if rs[i] else "✗")
            rows.append(row)
        name = run.name
        out = OUT / f"{name}.tsv"
        cols = list(rows[0])
        out.write_text("\t".join(cols) + "\n" + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in rows))
        valid = [r for r in rows if not r["env_defect"]]
        oc = Counter(r["outcome"] for r in rows)
        print(f"\n== {run}  → {out.relative_to(ROOT)}  ({len(rows)} tasks, traces {len(traces)})")
        print("outcomes: " + " · ".join(f"{k} {v}" for k, v in oc.most_common()))
        print(f"env_defect excluded: {len(rows) - len(valid)} · valid {len(valid)} · valid failures "
              f"{sum(r['outcome'] != '✓' for r in valid)}")
        for o in ("✓", "✗"):
            sub = [r for r in valid if r["outcome"] == o]
            if sub:
                print(f"  {o} {len(sub):3}: gold files hit all/some/none " +
                      "/".join(str(sum(r["files_hit"] == k for r in sub)) for k in ("all", "some", "none")) +
                      " · hunk near all/some/none " +
                      "/".join(str(sum(r["hunk_near"] == k for r in sub)) for k in ("all", "some", "none")))
        fails = [r for r in valid if r["outcome"] != "✓"]
        if fails:
            em = sum(r["edit_mismatch"] + r["edit_missing"] for r in fails)
            print(f"  failures: never edited {sum(r['first_edit'] == '-' for r in fails)} · edit failures {em} "
                  f"(vs ok {sum(r['edit_ok'] for r in fails)}) · edit loop {sum(r['edit_loop'] for r in fails)} · "
                  f"clobbered {sum(r['clobber'] > 0 for r in fails)} · tested {sum(r['tested'] for r in fails)} · test before submit {sum(r['test_before_submit'] for r in fails)} · "
                  f"len cuts {sum(r['len_cuts'] for r in fails if r['len_cuts'] != '-')} · "
                  f"compactions {sum(r['compactions'] for r in fails if r['compactions'] != '-')}")
            for n in ref_res:
                print(f"  failures solved by {n}: {sum(r[n] == '✓' for r in fails)} (run there: {sum(r[n] != '–' for r in fails)})")


if __name__ == "__main__":
    main()
