"""Paired comparison of two scripts/verified_box.sh runs on the same tasks (B-9 part 2: base vs base + LoRA).
  python3 scripts/paired.py runs/verified/vf_e4b_th runs/verified/vf_e4b_mc [--out runs/verified/paired_th_mc.json]
Reads each run dir like `verified_box.sh report` does: preds.json (merged), the grading report(s) <model>.<TAG>.json or
c<k>/<model>.<TAG>-c<k>.json (TAG = the dir name), and every <id>.traj.json (exit status, model calls). Only the tasks
predicted in BOTH runs are compared (the counts of the others are printed). Prints, for 📷:
  resolved A / B with Wilson 95% CIs · both / only A / only B / neither · difference B − A with a paired 95% CI ·
  exact McNemar p (two-sided binomial on the discordant tasks) · exit statuses and empty patches per run, and how the
  ContextWindowExceeded / empty-patch tasks move between runs · model calls p50/p90 · resolved per repo where they differ.
The task ids of each cell go to --out (JSON), not to the screen. Standard library only."""
import argparse
import collections
import glob
import json
import math
import os


def load(d):
    tag = os.path.basename(os.path.normpath(d))
    pf = os.path.join(d, "preds.json")
    preds = json.load(open(pf)) if os.path.exists(pf) else {}
    reps = sorted(glob.glob(os.path.join(d, f"*.{tag}.json")) + glob.glob(os.path.join(d, "c*", f"*.{tag}-c*.json")))
    res = set()
    for f in reps:
        res |= set(json.load(open(f)).get("resolved_ids", []))
    exit_st, calls = {}, {}
    for f in glob.glob(os.path.join(d, "**", "*.traj.json"), recursive=True):
        iid = os.path.basename(f)[:-len(".traj.json")]
        info = json.load(open(f)).get("info", {})
        exit_st[iid] = info.get("exit_status") or "?"
        c = (info.get("model_stats") or {}).get("api_calls")
        if isinstance(c, int):
            calls[iid] = c
    empty = {i for i, p in preds.items() if not (p.get("model_patch") or "").strip()}
    return {"tag": tag, "preds": preds, "reports": len(reps), "res": res, "exit": exit_st, "calls": calls, "empty": empty}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="run dir of the reference (e.g. the base)")
    ap.add_argument("b", help="run dir to compare (e.g. base + adapter)")
    ap.add_argument("--out", default=None, help="JSON with the task ids per cell (default runs/verified/paired_<A>_<B>.json)")
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    for R in (A, B):
        if not R["reports"]:
            print(f"!! {R['tag']}: no grading report: run `TAG={R['tag']} bash scripts/verified_box.sh eval` (or full) first")
    ids = sorted(set(A["preds"]) & set(B["preds"]))
    n = len(ids)
    print(f"== paired {A['tag']} (A) vs {B['tag']} (B): {n} tasks in both · only in A {len(set(A['preds']) - set(B['preds']))}"
          f" · only in B {len(set(B['preds']) - set(A['preds']))} · grading reports A {A['reports']} B {B['reports']}")
    if not n:
        return
    ra = {i for i in ids if i in A["res"]}; rb = {i for i in ids if i in B["res"]}
    both, oa, ob = ra & rb, ra - rb, rb - ra
    neither = set(ids) - ra - rb
    for R, r in ((A, ra), (B, rb)):
        lo, hi = wilson(len(r), n)
        print(f"   {R['tag']}: resolved {len(r)} / {n} = {len(r) / n:.1%} (95% CI {lo:.1%}–{hi:.1%})")
    b, c = len(oa), len(ob)
    diff = (c - b) / n
    se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0)) / n
    print(f"   both {len(both)} · only A {b} · only B {c} · neither {len(neither)}")
    print(f"   B − A = {c - b:+d} tasks = {diff:+.1%} (paired 95% CI {diff - 1.96 * se:+.1%} … {diff + 1.96 * se:+.1%})"
          f" · exact McNemar p = {mcnemar_exact(b, c):.3g} ({b + c} discordant)")
    for R in (A, B):
        st = collections.Counter(R["exit"].get(i, "no traj") for i in ids)
        cs = [R["calls"][i] for i in ids if i in R["calls"]]
        print(f"   {R['tag']}: exits {dict(st.most_common())} · empty patches {len(R['empty'] & set(ids))}"
              f" · model calls p50 {pct(cs, 0.5)} p90 {pct(cs, 0.9)}")
    cwe = lambda R: {i for i in ids if R["exit"].get(i) == "ContextWindowExceeded"}
    ca, cb = cwe(A), cwe(B)
    ea, eb = A["empty"] & set(ids), B["empty"] & set(ids)
    print(f"   ContextWindowExceeded: A {len(ca)} · B {len(cb)} · in both {len(ca & cb)} · A only {len(ca - cb)} · B only {len(cb - ca)}"
          f" · B-resolved among A's CWE {len(ca & rb)}")
    print(f"   empty patch: A {len(ea)} · B {len(eb)} · in both {len(ea & eb)} · A only {len(ea - eb)} · B only {len(eb - ea)}")
    repo = lambda i: i.rsplit("-", 1)[0].split("__")[-1]
    tot = collections.Counter(repo(i) for i in ids)
    ga, gb = collections.Counter(repo(i) for i in ra), collections.Counter(repo(i) for i in rb)
    moved = [f"{k} {ga[k]}→{gb[k]}/{v}" for k, v in tot.most_common() if ga[k] != gb[k]]
    print("   per repo where A ≠ B: " + (" · ".join(moved) if moved else "none"))
    out = x.out or os.path.join(os.path.dirname(os.path.normpath(x.a)), f"paired_{A['tag']}_{B['tag']}.json")
    json.dump({"a": A["tag"], "b": B["tag"], "n": n, "both": sorted(both), "only_a": sorted(oa), "only_b": sorted(ob),
               "neither": sorted(neither), "cwe_a": sorted(ca), "cwe_b": sorted(cb), "empty_a": sorted(ea), "empty_b": sorted(eb)},
              open(out, "w"), indent=1)
    print(f"   task ids per cell → {out}")


if __name__ == "__main__":
    main()
