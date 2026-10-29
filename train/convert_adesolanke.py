#!/usr/bin/env python3
"""adesolanke/gemma4-swe-agent-trajectories (Qwen3-Coder on SWE-rebench, harness tool format) → call-level records
(train/schema.md), source `d1_qwen`.

    python3 train/convert_adesolanke.py --data <dir>/conv_curated_v2.jsonl --audit runs/sft/d1_audit.json \
        --out runs/sft/d1_qwen

The dataset rows carry only assistant/tool messages (no system, no user prompt), tool-call arguments as dicts and the
tool results already in the harness's result format. This module rebuilds the prompt the harness would have sent:
system = scripts/arms/latest/prompts/system.md + ADK's agent-name suffix (as observed in the wave 7 traj logs), user =
build_agent_prompt wording (swegemma/harness/agent_runner.py) with the latest budget (7 min · 28 calls), the harness
limits (240 s · 5,000 chars · 150 lines · 10,000 chars), no Code Intelligence section (these repos have no graph data)
and a Workspace Layout rebuilt from the trajectory's own opening `find . -maxdepth 2` listing (depth 2, not the
harness's 3; the egg-info / __pycache__ / pruned top dirs filtered like build_workspace_tree_command). Wire format
mirrors runs/sft/e4b_w7 (arguments as JSON strings, reasoning sent back as reasoning_content + reasoning).
`--audit` restricts the output to rows the offline audit kept (train/audit_d1.py). Every number is printed.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SYSTEM_MD = os.path.join(REPO, "scripts", "arms", "latest", "prompts", "system.md")
AGENT_SUFFIX = '\n\nYou are an agent. Your internal name is "swe_agent".'   # ADK appends this (wave 7 traj logs)
TOOLS_FROM = os.path.join(REPO, "runs", "sft", "e4b_w7")                     # any record there carries the 9 schemas
POLICY = "Qwen3-Coder-480B-A35B-Instruct"
PRUNED_TOP_DIRS = ("docs", "docs_src", "tests", "test_data", "examples", "benchmarks")   # agent_runner._PRUNED_TOP_DIRS


def load_system():
    return open(SYSTEM_MD).read() + AGENT_SUFFIX


def load_tools():
    for f in sorted(os.listdir(TOOLS_FROM)):
        if f.endswith(".json"):
            return json.load(open(os.path.join(TOOLS_FROM, f)))["tools"]
    raise SystemExit(f"no record with tools under {TOOLS_FROM}")


def workspace_tree(listing, max_lines=200):
    """The dataset's opening `find . -maxdepth 2 -not -path '*/.*'` stdout → the harness tree format."""
    paths = []
    for line in listing.splitlines():
        p = line.strip()
        if not p.startswith("/workspace/"):
            continue
        p = p[len("/workspace/"):].rstrip("/")
        if not p:
            continue
        parts = p.split("/")
        if any(x.startswith(".") for x in parts) or parts[0] in PRUNED_TOP_DIRS and len(parts) > 1:
            continue
        if ".egg-info" in parts[0] or "__pycache__" in parts or p.endswith(".pyc"):
            continue
        paths.append("./" + p)
    # breadth first (depth, name), head -N, then plain sort — like build_workspace_tree_command
    paths = sorted(set(paths), key=lambda s: (s.count("/"), s))[:max_lines]
    return "\n".join(sorted(paths))


def build_user_prompt(repo, issue, tree):
    parts = [f"You are evaluating a software engineering task for repository {repo}.\n\nProblem Statement:\n{issue}\n"]
    parts.append("## Task Budget (Session terminates when any budget is exhausted)\n"
                 "- Time allowance: 7.0 minutes\n- Tool calls allowance: 28 calls\n")
    parts.append("## Execution Environment Rules\n"
                 "- Single command timeout: 240 seconds (commands exceeding this fail without ending the session)\n"
                 "- Command output limit: 5000 characters\n"
                 "- File view limit: 150 lines per read_file call\n"
                 "- File character limit: 10000 characters per read_file call\n"
                 "- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY "
                 "pre-installed. Do NOT attempt to run pip install or download packages.\n")
    parts.append("\n".join([
        "## Instructions:",
        "0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system "
        "site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.",
        "1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI "
        "subcommands, or Python modules.",
        "2. Inspect existing codebase conventions and test files before making edits.",
        "3. Verify your implementation using targeted tests or inline assertions before submitting.",
        "4. Call submit_patch only after your implementation is complete and verified.",
        "5. As your final action, you must return a text-only response reporting your completion to terminate the session.",
    ]))
    if tree:
        parts.append("## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up to 3 "
                     "levels):\n```\n" + tree + "\n```\n")
    return "\n".join(parts)


def opening_listing(messages):
    """stdout of the first call when it is the normalised `find` listing, else None."""
    if len(messages) < 2 or messages[0]["role"] != "assistant" or messages[1]["role"] != "tool":
        return None
    tcs = messages[0].get("tool_calls") or []
    if len(tcs) != 1 or tcs[0]["function"]["name"] != "run_command":
        return None
    if "find ." not in (tcs[0]["function"]["arguments"].get("command") or ""):
        return None
    try:
        res = json.loads(messages[1]["content"])
    except Exception:
        return None
    return res.get("stdout") if isinstance(res, dict) else None


def wire_assistant(m):
    """Dataset assistant turn → the harness's re-sent wire message."""
    out = {"role": "assistant"}
    if m.get("content"):
        out["content"] = m["content"]
    tcs = m.get("tool_calls") or []
    if tcs:
        out["tool_calls"] = [{"type": "function", "id": tc["id"],
                              "function": {"name": tc["function"]["name"],
                                           "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                             for tc in tcs]
    if m.get("reasoning"):
        out["reasoning_content"] = m["reasoning"]
        out["reasoning"] = m["reasoning"]
    return out


def build_record(row, system, tools, resolved=True, nebius=None):
    ms = row["messages"]
    tree = workspace_tree(opening_listing(ms) or "")
    history = [{"role": "system", "content": system},
               {"role": "user", "content": build_user_prompt(row["repo"], row["issue"], tree)}]
    calls = []
    i = 0
    while i < len(ms):
        m = ms[i]
        if m["role"] != "assistant":
            raise ValueError(f"{row['instance_id']}: message {i} is {m['role']}, expected assistant")
        tcs = m.get("tool_calls") or []
        target = {"reasoning": m.get("reasoning") or None, "content": m.get("content") or None,
                  "tool_calls": [{"name": tc["function"]["name"],
                                  "args": json.dumps(tc["function"]["arguments"], ensure_ascii=False),
                                  "id": tc["id"]} for tc in tcs],
                  "finish_reason": "tool_calls" if tcs else "stop"}
        calls.append({"call": len(calls) + 1, "segment": 0, "events_before": [], "input_messages": list(history),
                      "target": target, "thought_source": "qwen" if m.get("reasoning") else "none",
                      "usage_prompt_tokens": None, "err": None})
        history = history + [wire_assistant(m)]
        i += 1
        # the tool results that answer this turn's calls, in order
        for tc in tcs:
            if i >= len(ms) or ms[i]["role"] != "tool":
                raise ValueError(f"{row['instance_id']}: missing tool result for {tc['id']}")
            t = ms[i]
            history = history + [{"role": "tool", "tool_call_id": t["tool_call_id"], "content": t["content"]}]
            i += 1
    return {"instance_id": row["instance_id"], "repo": row["repo"], "base_commit": None, "pr": None,
            "patch_sha": (nebius or {}).get("patch_sha"), "source": "d1_qwen", "resolved": resolved,
            "policy_model": POLICY, "sampling": {"temperature": None, "top_p": None, "max_completion_tokens": None,
                                                 "thinking_token_budget": None, "enable_thinking": None},
            "env_fingerprint": None, "nebius_trajectory_id": (nebius or {}).get("trajectory_id"),
            "prompt_rebuilt": {"system": "scripts/arms/latest/prompts/system.md + agent suffix",
                               "tree_depth": 2 if tree else None}, "tools": tools, "calls": calls}


def iter_rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="conv_curated_v2.jsonl (untrusted download; read only)")
    ap.add_argument("--audit", default=None, help="runs/sft/d1_audit.json: write only rows it kept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N written records (0 = all)")
    a = ap.parse_args()
    system, tools = load_system(), load_tools()
    print(f"system prompt: {SYSTEM_MD} {len(system) - len(AGENT_SUFFIX)} chars + agent suffix {len(AGENT_SUFFIX)} "
          f"= {len(system)} chars · dataset rows carry no system/user message (rebuilt) · tools {len(tools)}")
    keep = None
    if a.audit:
        au = json.load(open(a.audit))
        keep = {(r["instance_id"], r["n"]): r for r in au["rows"] if r["keep"]}
        print(f"audit {a.audit}: keep {len(keep)} of {len(au['rows'])}")
    os.makedirs(a.out, exist_ok=True)
    seen, n_written, n_calls, n_tree, n_qwen = {}, 0, 0, 0, 0
    for row in iter_rows(a.data):
        n = seen.get(row["instance_id"], 0); seen[row["instance_id"]] = n + 1
        if keep is not None and (row["instance_id"], n) not in keep:
            continue
        info = keep[(row["instance_id"], n)] if keep else {}
        rec = build_record(row, system, tools, resolved=info.get("resolved", None), nebius=info)
        n_written += 1; n_calls += len(rec["calls"]); n_tree += rec["prompt_rebuilt"]["tree_depth"] is not None
        n_qwen += sum(1 for c in rec["calls"] if c["thought_source"] == "qwen")
        name = row["instance_id"] + (f".{n}" if n else "")
        with open(os.path.join(a.out, name + ".json"), "w") as w:
            json.dump(rec, w, ensure_ascii=False)
        if a.limit and n_written >= a.limit:
            break
    print(f"TOTAL records {n_written} · calls {n_calls} · calls with qwen reasoning {n_qwen} · workspace tree rebuilt "
          f"{n_tree}/{n_written} → {a.out}")
    return 0 if n_written else 1


if __name__ == "__main__":
    sys.exit(main())
