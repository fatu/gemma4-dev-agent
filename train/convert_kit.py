#!/usr/bin/env python3
"""dmitriigluzdov's LoRA kit, native tier (`lora_native_resolved.jsonl`) → call-level records (train/schema.md),
source `kit_31b`.

    python3 train/convert_kit.py --data runs/ext/dmitrii_kit/lora_native_resolved.jsonl --out runs/sft/kit31b \
        --orig runs/sft/kit31b_orig --ref runs/sft/kit31b_ref

The tier: 103 resolved trajectories of Gemma 4 31B (Budget-Fit configuration, official serving stack) on 55 public
SWE-rebench tasks, none of them a competition task (kit README); CC BY 4.0, credit Dmitrii Gluzdov / Nebius /
White Circle. Each saved example keeps the request as served (messages incl. earlier reasoning, tool calls and tool
results; tools; sampling) and the reply (`response_message`), so one example maps onto one record call:
input_messages = request.messages, target = response_message. A summariser call (one user message, ADK's "The
following is a conversation history" prompt) is not an agent call: it is skipped and marks `compaction` on the next
call; a history that does not extend the previous call's marks `reset` (the summary replaces the task prompt).

Prompt re-wrap (plan decision 3, as convert_atif.py): system = scripts/arms/latest/prompts/system.md + ADK's agent
suffix; the task prompt's budget lines → the latest values (7.0 minutes, 28 calls); tools = the 9 schemas of our traj
logs (the kit's differ from ours only in docstring indentation — counted and printed). The 31B's own summaries in
later histories stay as they were.

Write targets: a write_file into the workspace whose file name looks like a scratch / repro / test script is not a
target (`target: null`, `excluded: "repro_write"`): the harness's `git add -N .` puts every untracked file into the
graded patch, and our prompt forbids creating test files. The call stays in the history of the calls after it.

--orig / --ref: the same calls without the re-wrap, plus the kit's server-rendered prompt per call in render.py's
--check format, so `render.py <orig> --check <ref>` tests our renderer on these data (A2). Both use
`<task>.<id8>` as instance_id (several trajectories per task). Every number here is the record.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

ADK_SUFFIX = '\n\n\nYou are an agent. Your internal name is "swe_agent".'
SUMMARY_PROMPT = "The following is a conversation history"
LATEST_BUDGET = {"minutes": "7.0", "calls": "28"}
RE_TIME = re.compile(r"^- Time allowance: [0-9.]+ minutes$", re.M)
RE_CALLS = re.compile(r"^- Tool calls allowance: \d+ calls$", re.M)
RE_SCRATCH = re.compile(r"^(test_|tests?_|repro|reproduce|debug|check_|verify|scratch|tmp|temp)", re.I)


def text_of(content):
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return content or ""


def is_summary_call(msgs):
    return len(msgs) == 1 and msgs[0].get("role") == "user" and text_of(msgs[0].get("content")).startswith(SUMMARY_PROMPT)


def rewrap(msgs, system, stats):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "system":
            assert text_of(m["content"]).endswith(ADK_SUFFIX), "kit system message without ADK's agent suffix"
            m["content"] = system
        elif m.get("role") == "user" and isinstance(m.get("content"), str):
            c, k1 = RE_TIME.subn(f"- Time allowance: {LATEST_BUDGET['minutes']} minutes", m["content"])
            c, k2 = RE_CALLS.subn(f"- Tool calls allowance: {LATEST_BUDGET['calls']} calls", c)
            stats["budget_lines_rewritten"] += k1 + k2
            m["content"] = c
        out.append(m)
    return out


def target_of(step):
    rm = step.get("response_message") or {}
    tcs = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"),
            "args": (tc.get("function") or {}).get("arguments") or "{}"} for tc in rm.get("tool_calls") or []]
    return {"reasoning": rm.get("reasoning") or rm.get("reasoning_content"), "content": rm.get("content"),
            "tool_calls": tcs, "finish_reason": step.get("finish_reason")}


def norm_ws(x):
    """collapse whitespace in every string (the kit's tool docstrings keep their source indentation, ours do not)"""
    if isinstance(x, dict):
        return {k: norm_ws(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm_ws(v) for v in x]
    return re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x


def schema_violation(target, params):
    """a tool call naming an unknown tool, missing a required argument or passing an unknown one (the tool returned
    an error; train/parser_check.py's schema check, applied here so the records carry no such target)"""
    for tc in target["tool_calls"]:
        if tc["name"] not in params:
            return f"unknown tool {tc['name']}"
        try:
            args = json.loads(tc["args"])
        except Exception:
            return f"{tc['name']}: arguments are not JSON"
        missing = [p for p in params[tc["name"]]["required"] if p not in args]
        if missing:
            return f"{tc['name']}: missing {missing}"
        unknown = [k for k in args if k not in params[tc["name"]]["properties"]]
        if unknown:
            return f"{tc['name']}: unknown keys"
    return None


def scratch_write(target):
    for tc in target["tool_calls"]:
        if tc["name"] == "write_file":
            try:
                path = json.loads(tc["args"]).get("filepath", "")
            except Exception:
                path = ""
            if not path.startswith("/tmp") and (RE_SCRATCH.match(os.path.basename(path)) or
                                                re.match(r"^(/workspace/)?tests?/", path)):
                return path
    return None


def convert(row, system, tools, stats, rewrite=True):
    params = {t["function"]["name"]: {"required": t["function"]["parameters"].get("required", []),
                                      "properties": t["function"]["parameters"].get("properties", {})} for t in tools}
    calls, prev, pending, segment = [], None, [], 0
    for step in row["sft_examples"]:
        msgs = step["request"]["messages"]
        if is_summary_call(msgs):
            pending.append("compaction")
            stats["summariser_calls"] += 1
            continue
        if prev is not None and (len(msgs) < len(prev) or msgs[:len(prev)] != prev):
            pending.append("reset")
        if pending:
            segment += 1
        prev = msgs
        target, excluded = target_of(step), None
        bad = schema_violation(target, params)
        path = None if bad else scratch_write(target)
        if bad:
            excluded = "schema_violation"
            stats["schema_violation_excluded"] += 1
            stats["violations"][bad] += 1
        elif path:
            excluded = "repro_write"
            stats["repro_write_excluded"] += 1
            stats["excluded_paths"][os.path.basename(path)] += 1
        else:
            for tc in target["tool_calls"]:
                if tc["name"] == "write_file":
                    stats["write_file_kept"] += 1
                    stats["kept_write_paths"].append(json.loads(tc["args"]).get("filepath", "?"))
        calls.append({"call": len(calls) + 1, "segment": segment, "events_before": pending,
                      "input_messages": rewrap(msgs, system, stats) if rewrite else msgs,
                      "target": None if excluded else target, "excluded": excluded, "thought_source": "native",
                      "usage_prompt_tokens": None if rewrite else step.get("prompt_tokens"), "err": None,
                      "kit_turn": step.get("turn"), "kit_canonical_target": step.get("canonical_target_matches")})
        pending = []
    req = row["sft_examples"][0]["request"]
    task = row["task_id"] if rewrite else f"{row['task_id']}.{row['id'][:8]}"
    return {"instance_id": task, "repo": row.get("repository"), "base_commit": None, "pr": None, "patch_sha": None,
            "source": "kit_31b", "resolved": bool(row.get("resolved")), "policy_model": req.get("model"),
            "sampling": {"temperature": req.get("temperature"), "top_p": req.get("top_p"),
                         "max_completion_tokens": req.get("max_completion_tokens"),
                         "thinking_token_budget": req.get("thinking_token_budget"),
                         "enable_thinking": (req.get("chat_template_kwargs") or {}).get("enable_thinking"),
                         "max_tool_calls": 28, "max_time_minutes": 8, "configuration": row.get("configuration")},
            "env_fingerprint": None, "tools": tools if rewrite else req["tools"], "calls": calls,
            "kit_id": row["id"], "kit_outcome_sha256": row.get("outcome_sha256"), "kit_seed": row.get("seed")}


def reference(row):
    """the kit's server-rendered prompt per agent call, in render.py's --check format"""
    calls = []
    for step in row["sft_examples"]:
        if is_summary_call(step["request"]["messages"]):
            continue
        calls.append({"call": len(calls) + 1, "prompt": step["prefix"], "rendered_tokens": len(step["prompt_token_ids"]),
                      "usage_prompt": step.get("prompt_tokens")})
    return {"calls": calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/ext/dmitrii_kit/lora_native_resolved.jsonl")
    ap.add_argument("--system", default="scripts/arms/latest/prompts/system.md")
    ap.add_argument("--tools-from", default="runs/sft/e4b_w7", help="any traj-log record dir: our 9 tool schemas")
    ap.add_argument("--out", default="runs/sft/kit31b")
    ap.add_argument("--orig", default=None, help="also write the calls without the re-wrap here (A2 input)")
    ap.add_argument("--ref", default=None, help="also write the kit's rendered prompts here (render.py --check)")
    a = ap.parse_args()
    system = open(a.system).read().rstrip() + ADK_SUFFIX
    tools = json.load(open(sorted(glob.glob(os.path.join(a.tools_from, "*.json")))[0]))["tools"]
    ours = {t["function"]["name"]: t for t in tools}
    print(f"system prompt: latest {a.system} + ADK suffix = {len(system)} chars · tools: {len(tools)} schemas from {a.tools_from}"
          f" · budget lines → {LATEST_BUDGET['minutes']} minutes / {LATEST_BUDGET['calls']} calls")
    for d in (a.out, a.orig, a.ref):
        if d:
            os.makedirs(d, exist_ok=True)
    stats = Counter()
    stats["excluded_paths"], stats["kept_write_paths"], stats["violations"] = Counter(), [], Counter()
    schema_diff = Counter()
    tasks, repos = set(), Counter()
    for line in open(a.data):
        row = json.loads(line)
        kit_tools = {t["function"]["name"]: t for t in row["sft_examples"][0]["request"]["tools"]}
        assert set(kit_tools) == set(ours), f"{row['task_id']}: tool set differs: {sorted(kit_tools)}"
        for n, t in kit_tools.items():
            if t != ours[n]:
                schema_diff["whitespace only" if norm_ws(t) == norm_ws(ours[n]) else f"{n}: content"] += 1
        rec = convert(row, system, tools, stats)
        tasks.add(row["task_id"]); repos[row.get("repository")] += 1
        n_t = sum(1 for c in rec["calls"] if c["target"])
        ev = [e for c in rec["calls"] for e in c["events_before"]]
        stats["trajectories"] += 1; stats["calls"] += len(rec["calls"]); stats["targets"] += n_t
        stats["compaction"] += ev.count("compaction"); stats["reset"] += ev.count("reset")
        stats["segments"] += rec["calls"][-1]["segment"] + 1 if rec["calls"] else 0
        stats["non_canonical_targets"] += sum(1 for c in rec["calls"] if c["kit_canonical_target"] is False)
        name = f"{row['task_id']}.{row['id'][:8]}"
        with open(os.path.join(a.out, f"{name}.json"), "w") as w:
            json.dump(rec, w, ensure_ascii=False)
        if a.orig:
            with open(os.path.join(a.orig, f"{name}.json"), "w") as w:
                json.dump(convert(row, system, tools, Counter(excluded_paths=Counter(), kept_write_paths=[], violations=Counter()),
                                  rewrite=False), w, ensure_ascii=False)
        if a.ref:
            with open(os.path.join(a.ref, f"{name}.json"), "w") as w:
                json.dump(reference(row), w, ensure_ascii=False)
        print(f"{name:<44} calls {len(rec['calls']):>3} targets {n_t:>3} segments {rec['calls'][-1]['segment'] + 1:>2} "
              f"compaction {ev.count('compaction')} reset {ev.count('reset')}")
    print(f"tool schemas vs ours: {dict(schema_diff)}")
    print(f"schema violations excluded {stats['schema_violation_excluded']}: {dict(stats['violations'])}")
    print(f"write_file targets: excluded as scratch/repro/test {stats['repro_write_excluded']} "
          f"(names: {dict(stats['excluded_paths'].most_common(12))}) · kept {stats['write_file_kept']}: {stats['kept_write_paths']}")
    keys = ("trajectories", "calls", "targets", "segments", "compaction", "reset", "summariser_calls",
            "non_canonical_targets", "budget_lines_rewritten")
    print("TOTAL " + " · ".join(f"{k} {stats[k]}" for k in keys) + f" · tasks {len(tasks)} · repos {len(repos)} → {a.out}"
          + (f" (+ {a.orig}, {a.ref})" if a.orig else ""))
    return 0 if stats["trajectories"] else 1


if __name__ == "__main__":
    sys.exit(main())
