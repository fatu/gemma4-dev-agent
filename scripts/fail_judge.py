#!/usr/bin/env python3
"""Failure judge, step S3 of the failure study (notes/16): Opus reads one failed task — issue, gold patch, the agent's
patch, the test output, the agent's trajectory and its fail_atlas row — and records the root cause with a fixed codebook.

    python3 scripts/fail_judge.py runs/pod_all129_e2b/runs/dev33_base_all129_e2b                 # every valid failure
    python3 scripts/fail_judge.py <run> fastapi_11194 rich_3506 …                                 # a subset (calibration)
    python3 scripts/fail_judge.py <run> --table                                                   # the table only, no API
    python3 scripts/fail_judge.py <run> --show fastapi_11194                                      # print the prompt, no API

Needs runs/fail_atlas/<run name>.tsv (scripts/fail_atlas.py) — valid failures = outcome ≠ ✓ and env_defect 0.
Output: runs/fail_judge/<run name>/<task>.json (gitignored; it quotes task data). Analysis only — these labels are never
training data. Key: ANTHROPIC_API_KEY or ANTHROPIC_APIKEY (never printed). Prints its token use and cost at the end.

Codebook (code names follow christopherjkouzios' E4B taxonomy where they mean the same; T1, T2, R3, R4, V1, C1 are ours):
  B1 budget stop: the time / call budget ran out before a real fix existed (exploring, waiting on long commands)
  T1 edit-tool trouble: repeated edit_file failures (old_string not found / missing args) cost the run
  T2 file clobber: write_file / a shell redirect overwrote a source file with a fragment (added 2026-11-01, calibration)
  I1 malformed calls: garbled or empty tool calls, cut-off replies (finish length), text instead of calls
  E1 environment / runtime: the sandbox, a command or the harness broke the run (incl. context overflow)
  L1 wrong files: never edited the file(s) the fix belongs in
  L2 right file, wrong place: edited the right file but not the code that needed to change
  R1 wrong fix: changed the right code with the wrong logic (misread the issue or the code)
  R2 partial fix: the right idea, but missed a case, a sibling implementation (sync/async…), an export or a second file
  R3 regression: the fix works for the issue but breaks existing tests in the test file
  R4 hidden spec: a reasonable issue-following fix that fails only on names / messages / edge cases only the gold reveals
  P3 no patch: nothing useful submitted (no edit kept, or the patch was reverted / empty)
  V1 wrong verification: tested the wrong thing or misread a test result, then submitted believing it worked
  C1 lost after compaction: the run was on track, then lost the task or its progress after a context compaction
  X  other
"""
import concurrent.futures as cf
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")
CONC = int(os.environ.get("JUDGE_CONC", "6"))
PRICE_IN, PRICE_OUT = 5.0, 25.0          # $ / 1M tokens, Opus list price (printed; used for the cost line only)
CODES = ["B1", "T1", "T2", "I1", "E1", "L1", "L2", "R1", "R2", "R3", "R4", "P3", "V1", "C1", "X"]
CAP_PATCH, CAP_TEST, CAP_TRAJ, CAP_STEP = 12000, 3000, 45000, 500

PROMPT = """You are diagnosing why a small open model (Gemma 4, {model_note}) failed a SWE task as a coding agent.
The agent saw only the issue and the repository; it had tools (read_file, run_command, edit_file, write_file,
submit_patch, …), a budget of {budget}, and context compaction (old turns summarised) when the prompt grew large.
Grading: every test in the hidden test files must pass. The gold patch below is the reference fix (hidden from the agent).

Pick the single PRIMARY root cause — the earliest decisive reason this run did not produce a passing patch (if it never
worked in the right file, that is L1 even when later trouble piled up) — and any
SECONDARY causes, from this codebook:
B1 budget stop: the time / call budget ran out before a real fix existed (exploring too long, waiting on long commands)
T1 edit-tool trouble: repeated edit_file failures (old_string not found / missing arguments) cost the run
T2 file clobber: write_file (or a shell redirect) overwrote a source file with a fragment, destroying it
I1 malformed calls: garbled or empty tool calls, cut-off replies (finish_reason length), prose instead of tool calls
E1 environment / runtime: the sandbox, a command or the harness broke the run (including context overflow)
L1 wrong files: never edited the file(s) where the gold fix lives (or an equivalent place)
L2 right file, wrong place: edited the right file but not the code that needed to change
R1 wrong fix: changed the right code with the wrong logic (misread the issue or the code)
R2 partial fix: the right idea, but missed a case, a sibling implementation (sync/async, other class), an export or a second file
R3 regression: the change handles the issue but breaks existing tests in the hidden test files
R4 hidden spec: a reasonable issue-following fix that fails only on exact names / messages / edge cases that only the gold reveals
P3 no patch: nothing useful submitted (no edit kept, patch empty or reverted)
V1 wrong verification: tested the wrong thing or misread a test result, then submitted believing it worked
C1 lost after compaction: the run was on track, then lost the task or its progress after a compaction
X other
Judge from the evidence: compare the agent's patch with the gold patch and the failing tests, and read the trajectory.
first_wrong_call = the call number (as numbered in the trajectory) where the run first went decisively wrong, or null.
would_fix = the cheapest lever that would most likely have fixed THIS failure: prompt (an instruction the agent ignored
or lacked), setting (budget / sampling / output length), data (the model lacks the skill: needs training), harness
(tool or environment), none (not fixable: hidden spec). evidence = at most 2 short items, specific (call numbers, what
the patch does vs what the tests need). Call record_failure once.

OBJECTIVE FACTS (scripts/fail_atlas.py):
{facts}

ISSUE:
{issue}

GOLD PATCH (hidden from the agent):
{gold}

AGENT PATCH (what was graded; empty = no patch):
{patch}

TEST OUTPUT (tail):
{tests}

TRAJECTORY (call n: reasoning excerpt / reply / tool call, then ← the tool result excerpt; COMPACTION = history summarised):
{traj}
"""

TOOL = {"name": "record_failure", "description": "Record the root cause of this failed run.", "input_schema": {
    "type": "object", "required": ["primary", "secondary", "first_wrong_call", "evidence", "would_fix", "fix_hint", "confidence"],
    "properties": {
        "primary": {"type": "string", "enum": CODES},
        "secondary": {"type": "array", "items": {"type": "string", "enum": CODES}, "maxItems": 3},
        "first_wrong_call": {"type": ["integer", "null"]},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "would_fix": {"type": "string", "enum": ["prompt", "setting", "data", "harness", "none"]},
        "fix_hint": {"type": "string", "description": "one short sentence: what would have fixed it"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}}


def cap(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n // 2] + f"\n… [{len(s) - n} chars cut] …\n" + s[-n // 2:]


def render_traj(path):
    if not path.exists():
        return "(no trajectory file)"
    out = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("compaction"):
            out.append(f"{r['call']:>3} COMPACTION (history summarised)")
            continue
        for m in r.get("new") or []:
            if m.get("role") == "tool":
                out.append("    ← " + cap(str(m.get("content") or "").replace("\n", " "), CAP_STEP))
        resp = r.get("resp") or {}
        bits = []
        if resp.get("reasoning"):
            bits.append("think: " + cap(resp["reasoning"].replace("\n", " "), 300))
        if (resp.get("content") or "").strip():
            bits.append("text: " + cap(resp["content"].replace("\n", " "), 300))
        for t in resp.get("tool_calls") or []:
            bits.append(f"CALL {t.get('name')} {cap(str(t.get('args')), CAP_STEP)}")
        if resp.get("finish_reason") == "length":
            bits.append("[cut off: finish_reason length]")
        if r.get("err"):
            bits.append("ERROR " + r["err"][:200])
        out.append(f"{r['call']:>3} " + " | ".join(bits or ["(empty reply)"]))
    return cap("\n".join(out), CAP_TRAJ)


def atlas(run):
    p = ROOT / "runs" / "fail_atlas" / f"{run.name}.tsv"
    if not p.exists():
        sys.exit(f"{p} missing — run scripts/fail_atlas.py {run} first")
    return {r["task"]: r for r in csv.DictReader(open(p), delimiter="\t")}


def build(run, row, task):
    i = row["task"]
    model_note = os.environ.get("JUDGE_MODEL_NOTE", "a small edge model (E2B / E4B)")
    budget = os.environ.get("JUDGE_BUDGET", "28 tool calls and 7 minutes")
    facts = ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("task",))
    patch_p, test_p = run / "patches" / f"{i}.patch", run / "test_outputs" / f"{i}.log"
    return (PROMPT.replace("{model_note}", model_note).replace("{budget}", budget).replace("{facts}", facts)
            .replace("{issue}", cap(task["problem_statement"], 8000))
            .replace("{gold}", cap(task["patch"], CAP_PATCH))
            .replace("{patch}", cap(patch_p.read_text() if patch_p.exists() else "", CAP_PATCH))
            .replace("{tests}", cap(test_p.read_text()[-CAP_TEST:] if test_p.exists() else "(no test output)", CAP_TEST))
            .replace("{traj}", render_traj(run / "traj" / f"{i}.jsonl")))


def judge(client, run, out_dir, row, task):
    p = out_dir / f"{row['task']}.json"
    if p.exists():
        return json.loads(p.read_text())
    r = client.messages.create(model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": build(run, row, task)}],
                               tools=[TOOL], tool_choice={"type": "auto"})   # opus_label: forced tool_choice is rejected
    out = next((dict(b.input) for b in r.content if getattr(b, "type", "") == "tool_use"), None)
    if out is None:
        text = "".join(getattr(b, "text", "") for b in r.content)
        out = {"primary": "unparsed", "raw": text[:500]}
    if isinstance(out.get("secondary"), str):      # the model sometimes sends "R3, T1" instead of a list
        out["secondary"] = [x.strip() for x in out["secondary"].replace(";", ",").split(",") if x.strip()]
    out.update(task=row["task"], outcome=row["outcome"], model=MODEL, in_tokens=r.usage.input_tokens,
               out_tokens=r.usage.output_tokens)
    p.write_text(json.dumps(out, indent=1))
    return out


def table(out_dir, rows):
    from collections import Counter
    got = {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json")}
    print(f"{'task':16} out prim sec        call fix      conf  hint")
    for i in sorted(rows):
        g = got.get(i)
        if g:
            print(f"{i:16} {rows[i]['outcome']:3} {g.get('primary', '?'):4} {','.join(g.get('secondary') or []):10} "
                  f"{str(g.get('first_wrong_call')):4} {g.get('would_fix', '?'):8} {g.get('confidence', '?'):5} "
                  f"{(g.get('fix_hint') or '')[:90]}")
    c = Counter(g.get("primary") for g in got.values() if g["task"] in rows)
    w = Counter(g.get("would_fix") for g in got.values() if g["task"] in rows)
    print(f"judged {sum(c.values())}/{len(rows)} · primary: " + " · ".join(f"{k} {v}" for k, v in c.most_common()))
    print("would_fix: " + " · ".join(f"{k} {v}" for k, v in w.most_common()))
    tin = sum(g.get("in_tokens", 0) for g in got.values())
    tout = sum(g.get("out_tokens", 0) for g in got.values())
    print(f"tokens: in {tin:,} · out {tout:,} · ≈ ${tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.2f} "
          f"(list ${PRICE_IN}/${PRICE_OUT} per 1M, {MODEL})")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    run = pathlib.Path(args[0])
    rows = {i: r for i, r in atlas(run).items() if r["outcome"] != "✓" and r["env_defect"] == "0"}
    ids = [a for a in args[1:] if not a.startswith("-")]
    if ids:
        rows = {i: rows[i] for i in ids if i in rows}
    tasks = {t["instance_id"]: t for t in map(json.loads, open(ROOT / "harness" / "tasks.jsonl"))}
    out_dir = ROOT / "runs" / "fail_judge" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if "--show" in args:
        for i in rows:
            print(build(run, rows[i], tasks[i]))
        return
    if "--table" not in args:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_APIKEY")
        client = anthropic.Anthropic(api_key=key, max_retries=4)
        todo = [i for i in rows if not (out_dir / f"{i}.json").exists()]
        print(f"fail_judge: {run} · {len(rows)} valid failures · {len(todo)} to judge · model {MODEL} · conc {CONC} · "
              f"budget note '{os.environ.get('JUDGE_BUDGET', '28 tool calls and 7 minutes')}' · out {out_dir.relative_to(ROOT)}",
              flush=True)
        with cf.ThreadPoolExecutor(CONC) as ex:
            futs = {ex.submit(judge, client, run, out_dir, rows[i], tasks[i]): i for i in todo}
            for f in cf.as_completed(futs):
                try:
                    r = f.result()
                    print(f"  {futs[f]:16} {r.get('primary')}", flush=True)
                except Exception as e:      # one bad task must not stop the rest
                    print(f"  {futs[f]:16} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
    table(out_dir, rows)


if __name__ == "__main__":
    main()
