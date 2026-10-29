#!/usr/bin/env python3
"""Opus ATIF traces (runs/passing_traces.json → trace_<task>.json, ATIF-v1.7) → call-level records (train/schema.md).

    python3 train/convert_atif.py --index runs/passing_traces.json --out runs/sft/opus

One record per passing (task, run) pair: `runs/sft/opus/<task>.<run>.json`. The Opus runs had no thinking, so every
call is `thought_source: none`, `target.reasoning: null`. The harness wire format is mirrored from the traj-log
records (system/user content strings; assistant {content?, tool_calls[{type, id, function{name, arguments: JSON
string}}]}; one `role: tool` message per tool call, in call order).

ATIF step grammar (counted over the 96 passing traces, 2026-10-28): system_instruction · task_prompt · agent
tool-call steps (1 or 2 tool_calls; `observation` = the first call's result) · a `system` step without event_type
right after a 2-call step = the second call's observation · agent `text` step followed by a tool-call step with the
same timestamp and no usage = ONE reply (content + tool_calls) · agent `final` = a text-only reply (ends with <turn|>);
when it is not the last step the harness answered with a `system` `continuation_nudge`, which the model saw as a user
message (same as the traj logs) → new segment.

Prompt re-wrap (plan decision 3): the system message is scripts/arms/latest/prompts/system.md plus ADK's agent-name
suffix exactly as the server received it in the wave-7 records; the task prompt is the trace's, with the budget lines
rewritten to the latest values (7.0 minutes, 28 calls). Everything derived is printed.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

TOOLS9 = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
          "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
ADK_SUFFIX = '\n\n\nYou are an agent. Your internal name is "swe_agent".'
LATEST_BUDGET = {"minutes": "7.0", "calls": "28"}
RE_TIME = re.compile(r"^- Time allowance: [0-9.]+ minutes$", re.M)
RE_CALLS = re.compile(r"^- Tool calls allowance: \d+ calls$", re.M)


def latest_system(path, ref_record):
    """latest system.md + ADK suffix; the suffix rule is checked against a wave-7 record (base prompt + suffix)."""
    txt = open(path).read().rstrip() + ADK_SUFFIX
    if ref_record:
        ref = json.load(open(ref_record))["calls"][0]["input_messages"][0]["content"]
        assert ref.endswith(ADK_SUFFIX), "ADK suffix rule no longer matches the wave-7 system content"
    return txt


def rewrap_prompt(msg):
    n = 0
    msg, k = RE_TIME.subn(f"- Time allowance: {LATEST_BUDGET['minutes']} minutes", msg); n += k
    msg, k = RE_CALLS.subn(f"- Tool calls allowance: {LATEST_BUDGET['calls']} calls", msg); n += k
    return msg, n


def steps_to_calls(steps, stats):
    """ATIF steps → list of (events_before, reply{content, tool_calls, finish_reason}, tool_results[(id, name, content)])
    plus the task prompt and the nudge texts."""
    calls = []
    i = 0
    pending_events = []
    pending_text = None
    while i < len(steps):
        s = steps[i]
        src, et = s["source"], (s.get("extra") or {}).get("event_type")
        if src == "system" and et == "system_instruction":
            stats["system_instruction"] += 1
        elif src == "user" and et == "task_prompt":
            stats["task_prompt"] += 1
        elif src == "system" and et == "continuation_nudge":
            pending_events.append(("nudge", s["message"]))
            stats["nudge"] += 1
        elif src == "agent" and et == "text":
            # text + the next tool-call step with the same timestamp and no usage = one reply
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            assert nxt and nxt.get("tool_calls") and nxt["tool_calls"][0]["extra"].get("usage") is None \
                and nxt["tool_calls"][0]["extra"]["timestamp"] == s["extra"]["timestamp"], "text step not paired"
            pending_text = s["message"]
            stats["mixed"] += 1
        elif src == "agent" and s.get("tool_calls"):
            tcs = s["tool_calls"]
            assert all(tc["function_name"] in TOOLS9 for tc in tcs), [tc["function_name"] for tc in tcs]
            results = [(tcs[0]["tool_call_id"], tcs[0]["function_name"], s["observation"]["content"])]
            if len(tcs) > 1:
                stats["multi"] += 1
                for j, tc in enumerate(tcs[1:], 1):
                    obs = steps[i + j]
                    assert obs["source"] == "system" and not (obs.get("extra") or {}).get("event_type") \
                        and isinstance(obs.get("observation"), dict), "second observation missing"
                    assert obs["observation"]["extra"]["tool_name"] == tc["function_name"]
                    results.append((tc["tool_call_id"], tc["function_name"], obs["observation"]["content"]))
                i += len(tcs) - 1
            reply = {"content": pending_text, "tool_calls": [
                {"id": tc["tool_call_id"], "name": tc["function_name"], "args": json.dumps(tc["arguments"], ensure_ascii=False)}
                for tc in tcs], "finish_reason": "tool_calls"}
            calls.append((pending_events, reply, results))
            pending_events, pending_text = [], None
        elif src == "agent" and et == "final":
            reply = {"content": s["message"], "tool_calls": [], "finish_reason": "stop"}
            calls.append((pending_events, reply, []))
            pending_events = []
            stats["text_only"] += 1
        else:
            raise AssertionError(f"unexpected step {src} {et} #{s['step_id']}")
        i += 1
    assert not pending_text
    return calls


def build_record(task, hit, system, tools, stats):
    trace = json.load(open(hit["trace"]))
    steps = trace["steps"]
    prompt, n = rewrap_prompt(steps[1]["message"])
    stats["prompts_rewritten"] += 1 if n else 0
    stats["budget_lines_rewritten"] += n
    calls_raw = steps_to_calls(steps, stats)
    history = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, segment = [], 0
    for k, (events, reply, results) in enumerate(calls_raw, 1):
        ev = []
        for kind, text in events:
            history = history + [{"role": "user", "content": text}]
            ev.append(kind)
        if ev:
            segment += 1
        target = {"reasoning": None, "content": reply["content"], "tool_calls": reply["tool_calls"],
                  "finish_reason": reply["finish_reason"]}
        calls.append({"call": k, "segment": segment, "events_before": ev, "input_messages": history,
                      "target": target, "thought_source": "none", "usage_prompt_tokens": None, "err": None})
        a = {"role": "assistant"}
        if reply["content"] is not None:
            a["content"] = reply["content"]
        if reply["tool_calls"]:
            a["tool_calls"] = [{"type": "function", "id": tc["id"], "function": {"name": tc["name"], "arguments": tc["args"]}}
                               for tc in reply["tool_calls"]]
        history = history + [a] + [{"role": "tool", "tool_call_id": cid, "content": content} for cid, _, content in results]
    run = os.path.basename(hit["run"])
    patch_sha = hashlib.sha256(open(hit["patch"], "rb").read()).hexdigest() if hit.get("patch") else None
    repo = re.search(r"repository (\S+)\.", prompt)
    return {"instance_id": task, "run": run, "repo": repo.group(1) if repo else None, "base_commit": None, "pr": None,
            "patch_sha": patch_sha, "source": "opus", "resolved": True, "lenient": bool(hit["lenient"]),
            "policy_model": "claude-opus-5-5",
            "sampling": {"temperature": None, "top_p": None, "max_completion_tokens": None, "thinking_token_budget": None,
                         "enable_thinking": False, "max_tool_calls": 80, "max_time_minutes": 30},
            "env_fingerprint": None, "tools": tools, "over_28_calls": len(calls) > 28, "calls": calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="runs/passing_traces.json")
    ap.add_argument("--system", default="scripts/arms/latest/prompts/system.md")
    ap.add_argument("--ref", default="runs/sft/e4b_w7/rich_3506.json", help="a traj-log record: tools list + ADK suffix check")
    ap.add_argument("--out", default="runs/sft/opus")
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    tools = ref["tools"]
    assert {t["function"]["name"] for t in tools} == TOOLS9
    system = latest_system(a.system, a.ref)
    base_len = len(json.load(open(a.ref))["calls"][0]["input_messages"][0]["content"])
    print(f"system prompt: latest {a.system} {len(system)} chars (file {len(open(a.system).read())} + ADK suffix "
          f"{len(ADK_SUFFIX)}) · the traces' / wave-7 prompt as served {base_len} chars · budget lines → "
          f"{LATEST_BUDGET['minutes']} minutes / {LATEST_BUDGET['calls']} calls")
    index = json.load(open(a.index))
    os.makedirs(a.out, exist_ok=True)
    tot = {"tasks": len(index), "records": 0, "calls": 0, "targets": 0, "segments": 0, "nudge": 0, "mixed": 0,
           "multi": 0, "text_only": 0, "over_28": 0, "lenient": 0, "prompts_rewritten": 0, "budget_lines_rewritten": 0,
           "system_instruction": 0, "task_prompt": 0}
    for task in sorted(index):
        for hit in index[task]:
            stats = {k: 0 for k in tot}
            rec = build_record(task, hit, system, tools, stats)
            for k in ("nudge", "mixed", "multi", "text_only", "prompts_rewritten", "budget_lines_rewritten",
                      "system_instruction", "task_prompt"):
                tot[k] += stats[k]
            tot["records"] += 1; tot["calls"] += len(rec["calls"]); tot["targets"] += len(rec["calls"])
            tot["segments"] += rec["calls"][-1]["segment"] + 1
            tot["over_28"] += rec["over_28_calls"]; tot["lenient"] += rec["lenient"]
            with open(os.path.join(a.out, f"{task}.{rec['run']}.json"), "w") as w:
                json.dump(rec, w, ensure_ascii=False)
            print(f"{task:<16} {rec['run']:<24} calls {len(rec['calls']):>3} segments {rec['calls'][-1]['segment'] + 1} "
                  f"nudge {stats['nudge']} mixed {stats['mixed']} multi {stats['multi']} text_only {stats['text_only']} "
                  f"{'>28 ' if rec['over_28_calls'] else ''}{'L' if rec['lenient'] else ''}")
    assert tot["system_instruction"] == tot["records"] == tot["task_prompt"]
    print("TOTAL " + " · ".join(f"{k} {v}" for k, v in tot.items()) + f" → {a.out}")
    return 0 if tot["records"] else 1


if __name__ == "__main__":
    sys.exit(main())
