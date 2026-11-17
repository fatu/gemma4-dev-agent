#!/usr/bin/env python3
"""mini-swe-agent's SWE-bench runner with a submission fallback (B-9 part 3 step 18).

  .venv-swe/bin/python scripts/mini_fb.py --subset … -m … -c swebench.yaml -c cfg.yaml -w 8 -o OUT   # = mini-extra swebench …
  python3 scripts/mini_fb.py --stats runs/verified/<TAG>      # after grading: what the fallback changed

mini takes as the patch whatever the submit command printed after its marker line. The fallback acts only when a run
ends as Submitted and that text holds no diff hunk (a `--- ` / `+++ ` / `@@ ` triple): then `git diff HEAD`, run in
the task's container before it is removed, becomes the prediction. A submission with a hunk is never touched, and
no other exit status is touched. The submission as the model printed it stays in the trajectory's last message, and
every instance gets one line in OUT/fallback.jsonl (MSWEA_FB_LOG). A replaced submission could not have been graded
as resolved (no hunk = nothing to apply), so: resolved as submitted = resolved − resolved among `used`.
Read from mini-swe-agent v2.4.6 (run/benchmarks/swebench.py, environments/docker.py), not run on the laptop.
"""
import collections, glob, json, os, re, sys, threading

HUNK = re.compile(r"^--- .*\n\+\+\+ .*\n@@ ", re.M)
TREE_CMD = "git diff HEAD"


def has_hunk(s):
    return bool(HUNK.search(s or ""))


def decide(exit_status, submission, tree):
    """→ (use the tree diff?, why)"""
    if exit_status != "Submitted":
        return False, "not submitted"
    if has_hunk(submission):
        return False, "submission has a hunk"
    if not has_hunk(tree):
        return False, "no hunk in the tree diff either"
    return True, "empty submission" if not (submission or "").strip() else "submission without a hunk"


def stats(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "c*", "part*", "fallback.jsonl")) + glob.glob(os.path.join(run, "part*", "fallback.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    res = set()
    for f in glob.glob(os.path.join(run, "c*", "*.json")) + glob.glob(os.path.join(run, "*.json")):
        if os.path.basename(f) != "preds.json":
            try: res |= set(json.load(open(f)).get("resolved_ids", []))
            except Exception: pass
    last = {r["instance_id"]: r for r in rows}; R = list(last.values()); used = [r for r in R if r["used"]]
    why = collections.Counter(r["why"] for r in R)
    print(f"== fallback {os.path.basename(run.rstrip('/'))}: {len(R)} instances logged · used {len(used)} "
          f"(empty submission {sum(r['why'] == 'empty submission' for r in used)}, without a hunk {sum(r['why'] == 'submission without a hunk' for r in used)}) "
          f"· resolved {len(res)} · resolved through the fallback {sum(r['instance_id'] in res for r in used)} "
          f"→ resolved as submitted {len(res) - sum(r['instance_id'] in res for r in used)} · reasons {dict(why)} "
          f"· tree diff with a hunk at a non-submitted exit {sum(r['tree_has_hunk'] and r['exit_status'] != 'Submitted' for r in R)} (not used) "
          f"· git diff failed {sum(r.get('tree_rc') != 0 for r in R)}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--stats":
        for run in sys.argv[2:]:
            stats(run)
        return
    import minisweagent.run.benchmarks.swebench as sb
    log, lock = os.environ.get("MSWEA_FB_LOG"), threading.Lock()
    orig = sb.ProgressTrackingAgent.run

    def run(self, *a, **k):
        info = orig(self, *a, **k)
        row = {"instance_id": getattr(self, "instance_id", "?")}
        try:
            st, sub = info.get("exit_status"), info.get("submission") or ""
            try:
                out = self.env.execute({"command": TREE_CMD}, timeout=120)
            except Exception as e:                       # incl. mini's Submitted, should a diff start with the marker
                out = {"output": "", "returncode": -2, "exception_info": f"{type(e).__name__}: {e}"[:200]}
            tree = out.get("output", "") if out.get("returncode") == 0 else ""
            use, why = decide(st, sub, tree)
            row.update(exit_status=st, sub_chars=len(sub), sub_has_hunk=has_hunk(sub), tree_chars=len(tree),
                       tree_has_hunk=has_hunk(tree), tree_rc=out.get("returncode"), used=use, why=why, sub_head=sub[:160])
            if use:
                info = {**info, "submission": tree}      # a copy: the exit message keeps what the model printed
        except Exception as e:
            row.update(used=False, why=f"fallback error {type(e).__name__}: {e}"[:200])
        if log:
            with lock, open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
        return info

    sb.ProgressTrackingAgent.run = run
    print(f"mini_fb · submission fallback ON · used only at exit Submitted with no diff hunk in the submission · tree command `{TREE_CMD}` · log {log}", flush=True)
    sb.app()


if __name__ == "__main__":
    main()
