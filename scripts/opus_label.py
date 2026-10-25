#!/usr/bin/env python3
"""Label every task by why it is hard: Opus reads the issue, the gold patch and the hidden tests; goldcheck decides env.

    python3 scripts/opus_label.py                      # all tasks in harness/tasks.jsonl → runs/labels/
    python3 scripts/opus_label.py fastapi_15588 ...    # a subset
    GC=runs/gc_all/all.jsonl python3 scripts/opus_label.py --table   # 📷 the table only (no API calls)

Categories (the final one, `cat`):
  1 clean        the tests check what the issue says; names/messages they need are in the issue or already in the repo
  2 edge_strict  the tests exercise inputs / edge cases the issue does not mention, so a fix that follows the issue fails
  3 assert_over  the tests require exact names, messages, signatures or formats that only the gold patch reveals
  4 env_defect   the gold patch itself does not pass here (goldcheck NOT resolved, or the test run times out)
Opus gives 1-3 (plus both 2/3 flags, and env_suspect from reading only); goldcheck overrides with 4.
Needs ANTHROPIC_API_KEY (or ANTHROPIC_APIKEY). Outputs stay in runs/ (gitignored): they quote task data.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import sys

MODEL = os.environ.get("LABEL_MODEL", "claude-opus-5-5")
CAP = int(os.environ.get("LABEL_CAP", "40000"))       # chars per field sent (patch, test_patch)
CONC = int(os.environ.get("LABEL_CONC", "6"))
OUT = pathlib.Path(os.environ.get("LABEL_OUT", "runs/labels"))
GC = os.environ.get("GC", "runs/gc_all/all.jsonl")

PROMPT = """You are auditing a SWE benchmark task. An agent sees ONLY the issue text and the repository at base_commit.
It never sees the gold patch or the hidden tests. Hidden tests = every test in the test patch's files must pass.

Classify how hard it is to pass the hidden tests from the issue alone, using these definitions:
- clean: the hidden tests check behaviour the issue states, or what a competent maintainer would obviously infer.
  Every name, message, signature or format the tests rely on is in the issue or already exists in the repository.
- edge_strict: the tests exercise inputs, edge cases or variants the issue does not mention (other types, empty/None,
  sibling entry points, async twins...), so a fix that does exactly what the issue says would likely fail some test.
- assert_over: the tests require exact identifiers, error messages, parameter names, return structures, output text
  or file paths that are NOT in the issue and NOT already in the repository, i.e. only the gold patch reveals them.
Also flag env_suspect if, from reading alone, the tests look slow, network/timing/platform dependent, or depend on
package versions (only a flag; the environment verdict comes from actually running the gold patch).

Answer by calling the record_label tool once. primary = the single biggest reason a good issue-following fix would
fail (clean if none); evidence = at most 3 short items, each the specific test expectation and why the issue does or
does not give it.

ISSUE:
{issue}

GOLD PATCH (hidden from the agent):
{patch}

TEST PATCH (hidden from the agent):
{tests}
"""

CATS = {"clean": 1, "edge_strict": 2, "assert_over": 3}
TOOL = {"name": "record_label", "description": "Record the task's label.", "input_schema": {
    "type": "object", "required": ["primary", "edge_strict", "assert_over", "env_suspect", "evidence", "difficulty", "confidence"],
    "properties": {
        "primary": {"type": "string", "enum": ["clean", "edge_strict", "assert_over"]},
        "edge_strict": {"type": "boolean"}, "assert_over": {"type": "boolean"}, "env_suspect": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}}


def cap(s):
    return s if len(s) <= CAP else s[:CAP] + f"\n... [truncated, {len(s) - CAP} more chars]"


def label(client, t):
    p = OUT / f"{t['instance_id']}.json"
    if p.exists():
        return json.loads(p.read_text())
    msg = PROMPT.replace("{issue}", cap(t["problem_statement"] + ("\n\nHINTS:\n" + t["hints_text"] if t.get("hints_text") else "")))
    msg = msg.replace("{patch}", cap(t["patch"])).replace("{tests}", cap(t["test_patch"]))
    r = client.messages.create(model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": msg}],
                               tools=[TOOL], tool_choice={"type": "auto"})   # this model rejects a forced tool_choice
    out = next((dict(b.input) for b in r.content if getattr(b, "type", "") == "tool_use"), None)
    if out is None:   # answered in text: take the first JSON object that parses
        text = "".join(getattr(b, "text", "") for b in r.content)
        out = {"primary": "unparsed", "raw": text[:500]}
        for k in (k for k, ch in enumerate(text) if ch == "{"):
            try:
                out = json.JSONDecoder().raw_decode(text[k:])[0]
                break
            except ValueError:
                continue
    out.update(instance_id=t["instance_id"], model=MODEL, in_tokens=r.usage.input_tokens, out_tokens=r.usage.output_tokens)
    p.write_text(json.dumps(out, indent=1))
    return out


def table(tasks):
    gc = {}
    if pathlib.Path(GC).exists():
        gc = {r["instance_id"]: r for r in map(json.loads, open(GC))}
    rows, counts = [], {}
    for t in tasks:
        i = t["instance_id"]
        lab = json.loads((OUT / f"{i}.json").read_text()) if (OUT / f"{i}.json").exists() else {}
        g = gc.get(i)
        env = None
        if g and not g["resolved"]:
            err = (g.get("error") or "").lower()
            env = "timeout" if "timeout" in err or "timed out" in err else "gold_fails"
        cat = 4 if env else CATS.get(lab.get("primary"), 0)
        counts[cat] = counts.get(cat, 0) + 1
        rows.append((i, cat, lab.get("primary", "-"), "E" if lab.get("edge_strict") else ".",
                     "A" if lab.get("assert_over") else ".", "s" if lab.get("env_suspect") else ".",
                     "-" if not g else ("ok" if g["resolved"] else env), lab.get("difficulty", "-"), lab.get("confidence", "-")))
    print(f"labels: {OUT} · model {MODEL} · goldcheck {GC if gc else 'MISSING (cat 4 not decided)'} · {len(gc)} gold results")
    print(f"{'task':16} cat  primary      E A s  gold        diff conf")
    for r in rows:
        print(f"{r[0]:16} {r[1]:3}  {r[2]:12} {r[3]} {r[4]} {r[5]}  {r[6]:10}  {r[7]!s:4} {r[8]}")
    names = {0: "unlabelled", 1: "clean", 2: "edge_strict", 3: "assert_over", 4: "env_defect"}
    print("totals: " + " · ".join(f"{c} {names[c]} {counts[c]}" for c in sorted(counts)))
    return rows


def main():
    args = sys.argv[1:]
    tasks = [json.loads(l) for l in open("harness/tasks.jsonl")]
    ids = [a for a in args if not a.startswith("-")]
    if ids:
        tasks = [t for t in tasks if t["instance_id"] in ids]
    OUT.mkdir(parents=True, exist_ok=True)
    if "--table" not in args:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_APIKEY")
        client = anthropic.Anthropic(api_key=key, max_retries=4)
        print(f"opus_label: {len(tasks)} tasks · model {MODEL} · cap {CAP} chars/field · conc {CONC} · out {OUT}", flush=True)
        with cf.ThreadPoolExecutor(CONC) as ex:
            futs = {ex.submit(label, client, t): t["instance_id"] for t in tasks}
            for f in cf.as_completed(futs):
                try:
                    r = f.result()
                    print(f"  {futs[f]:16} {r.get('primary')}", flush=True)
                except Exception as e:   # one bad task must not stop the rest
                    print(f"  {futs[f]:16} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        tot_in = sum(json.loads(p.read_text()).get("in_tokens", 0) for p in OUT.glob("*.json"))
        tot_out = sum(json.loads(p.read_text()).get("out_tokens", 0) for p in OUT.glob("*.json"))
        print(f"tokens so far: in {tot_in:,} · out {tot_out:,}")
    table(tasks)


if __name__ == "__main__":
    main()
