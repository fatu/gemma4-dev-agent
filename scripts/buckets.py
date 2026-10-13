#!/usr/bin/env python3
"""W2 step 3: why tasks fail, and what fills the 32K context — from a swegemma results dir.

    python scripts/buckets.py runs/dev33_out8k                 # 📷 the one-screen report
    python scripts/buckets.py runs/dev33_sample runs/dev33_out8k
    python scripts/buckets.py --probe runs/dev33_out8k         # 📷 if the report says "no steps": prints one trace's shape

Reads <run>/task_results.jsonl, patches/<id>.patch and traces/trace_<id>.json (ATIF trajectory: steps with message,
reasoning, tool_calls, observation, metrics.prompt_tokens). Field names are looked up defensively; --probe shows the
real ones if they differ.

Buckets (first match wins):
  resolved       hidden tests pass
  ctx_overflow   error mentions the context window (Finding #1)
  other_error    any other harness/model error
  no_patch       empty patch (gave up, or ended without editing)
  bad_apply      patch did not apply in the verifier
  wrong_patch    patch applied, tests fail
Context: per task the peak prompt tokens, steps, and the size of each tool's outputs (≈ chars/3.5 tokens). A drop in
prompt tokens between two consecutive steps is counted as a compaction event, so 0 means the history only ever grew.
"""
import collections
import json
import pathlib
import sys

CTX = ("context length", "contextwindowexceeded", "maximum context")
TOK = 3.5                                                     # chars per token, rough for code + English


def g(d, *keys, default=None):
    """first present key (dotted paths allowed)"""
    for k in keys:
        cur = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return default


def text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(text(i) for i in x)
    if isinstance(x, dict):
        return text(g(x, "content", "text", "output", "result", default=json.dumps(x)))
    return str(x)


def bucket(r, patch: str) -> str:
    err = text(r.get("error")).lower()
    if r.get("resolved"):
        return "resolved"
    if any(c in err for c in CTX):
        return "ctx_overflow"
    if "failed to apply" in err:
        return "bad_apply"
    if err:
        return "other_error"
    if not patch.strip():
        return "no_patch"
    return "wrong_patch"


def steps_of(trace):
    s = g(trace, "steps", "trajectory.steps", "events", default=[])
    return s if isinstance(s, list) else []


def analyse(run: pathlib.Path):
    rows = [json.loads(l) for l in (run / "task_results.jsonl").open()]
    out = []
    for r in rows:
        iid = r["instance_id"]
        pf = run / "patches" / f"{iid}.patch"
        patch = pf.read_text(errors="replace") if pf.exists() else ""
        tf = run / "traces" / f"trace_{iid}.json"
        trace = json.loads(tf.read_text()) if tf.exists() else {}
        steps = steps_of(trace)
        prompts, tool_tok, think_tok, calls = [], collections.Counter(), 0, collections.Counter()
        biggest = (0, "")
        for st in steps:
            p = g(st, "metrics.prompt_tokens", "metrics.input_tokens", "usage.prompt_tokens", "prompt_tokens")
            if isinstance(p, (int, float)):
                prompts.append(int(p))
            think_tok += len(text(g(st, "reasoning_content", "reasoning", "thought", "thoughts"))) / TOK
            tcs = g(st, "tool_calls", default=[]) or []
            names = {}
            for tc in tcs:
                n = g(tc, "function_name", "name", "function.name", default="?")
                calls[n] += 1
                names[g(tc, "tool_call_id", "id", default=len(names))] = n
            results = g(st, "observation.results", "observations", "tool_results", default=[]) or []
            for res in results:
                n = names.get(g(res, "source_call_id", "tool_call_id", "id"), None) or \
                    (next(iter(names.values())) if len(names) == 1 else "?")
                t = len(text(res)) / TOK
                tool_tok[n] += t
                if t > biggest[0]:
                    biggest = (t, n)
        drops = sum(1 for a, b in zip(prompts, prompts[1:]) if b < a * 0.8)
        out.append(dict(id=iid, bucket=bucket(r, patch), steps=len(steps), peak=max(prompts, default=0),
                        drops=drops, tool_tok=tool_tok, think=think_tok, calls=calls, biggest=biggest,
                        patch_lines=sum(1 for l in patch.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))))
    return out


def report(run, res):
    n = len(res)
    print(f"== {run.name}  ({n} tasks)")
    order = ["resolved", "ctx_overflow", "other_error", "no_patch", "bad_apply", "wrong_patch"]
    cnt = collections.Counter(x["bucket"] for x in res)
    print("  " + "  ".join(f"{b} {cnt[b]}" for b in order if cnt[b]))
    if not any(x["steps"] for x in res):
        print("  no steps found in traces/ — run with --probe and photo it"); return
    print(f"  {'bucket':13} {'n':>3} {'steps':>6} {'peak prompt':>11} {'compactions':>11} {'patch Δ':>8}")
    for b in order:
        xs = [x for x in res if x["bucket"] == b]
        if xs:
            m = lambda k: sum(x[k] for x in xs) / len(xs)
            print(f"  {b:13} {len(xs):>3} {m('steps'):>6.1f} {m('peak'):>11.0f} {m('drops'):>11.1f} {m('patch_lines'):>8.0f}")
    tot = collections.Counter()
    calls = collections.Counter()
    for x in res:
        tot.update(x["tool_tok"]); calls.update(x["calls"])
    think = sum(x["think"] for x in res)
    allt = sum(tot.values()) + think
    if allt == 0:
        print("  no tool-output/thinking text found under the guessed field names — one real step with a tool call:")
        return dump_step(run)
    print(f"  what the agent added to context (≈tokens, all tasks; system prompt not included):")
    for name, t in tot.most_common(6):
        print(f"    {name:22} {t/1000:7.0f}K  {100*t/allt:4.0f}%   calls {calls[name]:>4}  avg {t/max(calls[name],1):6.0f}/call")
    print(f"    {'thinking':22} {think/1000:7.0f}K  {100*think/allt:4.0f}%")
    big = sorted(res, key=lambda x: -x["biggest"][0])[:3]
    print("  biggest single tool outputs: " + " · ".join(f"{x['id']} {x['biggest'][1]} ≈{x['biggest'][0]/1000:.1f}K" for x in big))


def dump_step(run):
    """print every field of the first step that has tool calls, plus the step after it (outputs may live there)"""
    for tf in sorted((run / "traces").glob("trace_*.json")):
        st = steps_of(json.loads(tf.read_text()))
        for i, s in enumerate(st):
            if g(s, "tool_calls", "tool_call", "function_calls"):
                for j, x in enumerate(st[i:i + 2]):
                    print(f"  {tf.name} step {i + j}:")
                    for k, v in x.items():
                        print(f"    {k:18} {type(v).__name__:5} {json.dumps(v, ensure_ascii=False)[:110]}")
                return
    print("  (no step with tool_calls either) — run --probe")


def probe(run):
    tf = sorted((run / "traces").glob("trace_*.json"))
    if not tf:
        sys.exit(f"no traces in {run}/traces")
    t = json.loads(tf[0].read_text())
    print(tf[0].name, "top keys:", list(t)[:20])
    st = steps_of(t)
    print("steps:", len(st))
    for s in st[:3]:
        print(" step keys:", {k: type(v).__name__ for k, v in s.items()})
        for k in ("metrics", "observation", "tool_calls"):
            if k in s:
                print(f"   {k}:", json.dumps(s[k])[:160])
    r = json.loads((run / "task_results.jsonl").open().readline())
    print("task_results keys:", {k: (str(v)[:40]) for k, v in r.items()})


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--probe":
        return probe(pathlib.Path(args[1]))
    for a in args:
        report(pathlib.Path(a), analyse(pathlib.Path(a)))
        print()


if __name__ == "__main__":
    main()
