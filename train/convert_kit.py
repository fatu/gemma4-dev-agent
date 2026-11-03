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

Scratch files (--scratch tmp, the default; user decision 2026-11-03 "不删，改写到 /tmp"): 67 of the 103 trajectories
write repro / scratch / test scripts into the workspace (Budget-Fit's prompt says nothing about /tmp), while our
prompt — now in front of every call — says "Scratch files go in /tmp only". Under our prompt our own 31B never does it
(0/129 tasks) and runs /tmp scripts fine (43/44). So each such file moves to /tmp/<name>, consistently in every
message of the trajectory: the file tools are sandboxed to the workspace ("Path traversal detected … escapes workspace
root", 27/27 of our 31B's write_file to /tmp failed), so write_file → run_command `cat > /tmp/<name> << 'EOF'` and a
scratch edit_file → run_command `python3 - << 'EOF'` (an in-place replace; not a target, `excluded: "scratch_edit"`),
their results → run_command's {"status": "ok", "stdout": "", …}; paths in commands, thoughts and text → /tmp/<name>;
in tool output `/workspace/<path>` and traceback `File "<path>"` → /tmp/<name> and `?? <path>` git-status lines
dropped. Bare names left in tool output (e.g. an `ls` listing) are counted, not changed. --scratch keep leaves the
trajectories as they are.

--orig / --ref: the same calls without the re-wrap, plus the kit's server-rendered prompt per call in render.py's
--check format, so `render.py <orig> --check <ref>` tests our renderer on these data (A2). Both use
`<task>.<id8>` as instance_id (several trajectories per task). Every number here is the record.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

from mask_errors import mask_tool_errors, summary

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


RE_SHELL_OUT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*([\w./-]+)")
RUN_OK = json.dumps({"status": "ok", "stdout": "", "stderr": "", "exit_code": 0})


def rel(p):
    p = p or ""
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def is_scratch(p):
    return bool(p) and not p.startswith(("/tmp", "/dev", "/proc")) and not (p.startswith("/") and not p.startswith("/workspace/")) \
        and bool(RE_SCRATCH.match(os.path.basename(p)) or re.match(r"^tests?/", rel(p)))


def scratch_map(row):
    """workspace scratch files of one trajectory (write_file targets and shell redirections with a scratch name) →
    /tmp/<name>; two files with one name keep their directory in the new name"""
    paths = set()
    for s in row["sft_examples"]:
        for tc in (s.get("response_message") or {}).get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                continue
            if fn.get("name") == "write_file" and is_scratch(a.get("filepath", "")):
                paths.add(rel(a["filepath"]))
            elif fn.get("name") == "run_command":
                paths.update(rel(m.group(1)) for m in RE_SHELL_OUT.finditer(a.get("command", "")) if is_scratch(m.group(1)))
    base = Counter(os.path.basename(p) for p in paths)
    return {p: "/tmp/" + (os.path.basename(p) if base[os.path.basename(p)] == 1 else p.replace("/", "_")) for p in paths}


def sub_paths(s, mp, bare=True):
    """every mention of a scratch path → its /tmp name (bare=False: only `/workspace/<p>` and traceback `File "<p>"`)"""
    for p in sorted(mp, key=len, reverse=True):
        q = re.escape(p)
        if bare:
            s = re.sub(r"(?<![\w/.-])(?:\./|/workspace/)?" + q + r"(?![\w.-])", mp[p], s)
        else:
            s = re.sub(r"/workspace/" + q + r"(?![\w.-])", mp[p], s)
            s = re.sub(r'(File \\?")(?:\./)?' + q + r'(\\?")', lambda m: m.group(1) + mp[p] + m.group(2), s)
    return s


def heredoc(path, body, cmd):
    delim = next(d for d in ("EOF", "PYEOF", "SCRATCH_EOF", "EOF_1") if d not in body.splitlines())
    return f"{cmd} << '{delim}'\n{body if body.endswith(chr(10)) else body + chr(10)}{delim}"


def py_edit(path, old, new):
    body = (f"import pathlib\np = pathlib.Path({path!r})\ns = p.read_text()\nold = {old!r}\nnew = {new!r}\n"
            "assert s.count(old) == 1, s.count(old)\np.write_text(s.replace(old, new))\n")
    return heredoc(path, body, "python3 -")


class Scratch:
    """moves one trajectory's workspace scratch files to /tmp in every message (module docstring)"""

    def __init__(self, mp, stats):
        self.mp, self.stats, self.conv, self.cache, self.ops = mp, stats, {}, {}, {}

    def call(self, tc, count):
        fn = tc.get("function") or {}
        name = fn.get("name")
        try:
            a = json.loads(fn.get("arguments") or "{}")
        except Exception:
            return tc
        p = rel(a.get("filepath", "")) if name in ("write_file", "edit_file", "read_file") else None
        if p in self.mp:
            new = self.mp[p]
            cmd = (heredoc(new, a.get("content", ""), f"cat > {new}") if name == "write_file" else
                   py_edit(new, a.get("old_string", ""), a.get("new_string", "")) if name == "edit_file" else f"cat {new}")
            self.conv[tc.get("id")] = name
            self.ops[tc.get("id")] = ("write", p, a.get("content", "")) if name == "write_file" else \
                ("edit", p, a.get("old_string", ""), a.get("new_string", "")) if name == "edit_file" else ("read", p)
            if count:
                self.stats[f"scratch {name} → run_command"] += 1
            return {**tc, "function": {"name": "run_command", "arguments": json.dumps({"command": cmd}, ensure_ascii=False)}}
        if name == "run_command":
            c2 = sub_paths(a.get("command", ""), self.mp)
            if c2 != a.get("command", ""):
                if count:
                    self.stats["scratch paths rewritten in commands"] += 1
                return {**tc, "function": {**fn, "arguments": json.dumps({"command": c2}, ensure_ascii=False)}}
        return tc

    def text(self, s, count, what):
        s2 = sub_paths(s, self.mp)
        if count and s2 != s:
            self.stats[f"scratch paths rewritten in {what}"] += 1
        return s2

    def result(self, content, kind):
        if kind in ("write_file", "edit_file", "read_file"):
            ok = '"status": "ok"' in content[:40]
            self.stats[f"{kind} results → run_command {'ok' if ok else 'error'}"] += 1
            if ok:
                return RUN_OK
            m = re.search(r"(\d+) (?:occurrences|matches|times)", content)
            tb = f"Traceback (most recent call last):\n  File \"<stdin>\", line 6, in <module>\nAssertionError: {m.group(1) if m else 0}\n"
            return json.dumps({"status": "error", "error_type": "CommandError", "error_message": tb,
                               "details": {"stdout": "", "stderr": tb, "exit_code": 1}})
        c2 = sub_paths(content, self.mp, bare=False)
        for p in self.mp:                     # git status lines of the moved files (short and long form)
            q = r"(?:\./)?" + re.escape(p)
            c2 = re.sub(r"(^|(?<=\\n)|(?<=\": \"))[ ?AMD]{2} " + q + r"(\\n|$)", r"\1", c2)
            c2 = re.sub(r"\\t(?:new file|deleted|modified):\s+" + q + r"\\n", "", c2)
            # listings of the workspace (grep -r / find / git diff --name-only) would not show a /tmp file
            c2 = re.sub(r"(^|(?<=\\n)|(?<=\": \"))\./" + re.escape(p) + r"(?::.*?)?\\n", r"\1", c2)
            c2 = re.sub(r"(^|(?<=\\n)|(?<=\": \"))" + re.escape(p) + r"\\n", r"\1", c2)
            # `git diff` after the harness's `git add -N .` shows the new file as a hunk: gone with the file
            c2 = re.sub(r"diff --git a/" + re.escape(p) + r" b/" + re.escape(p) + r"\\n.*?(?=diff --git |\", \"|\"\}|$)", "",
                        c2, flags=re.S)
        c2 = sub_paths(c2, self.mp)           # what is left names the file itself (linter output, error echo)
        if c2 != content:
            self.stats["tool outputs rewritten"] += 1
        left = [p for p in self.mp if re.search(r"(?<![\w/.-])(?:\./)?" + re.escape(p) + r"(?![\w.-])", c2)]
        if left:
            self.stats["tool outputs still naming a moved file (left as is)"] += 1
        return c2

    def message(self, m, count=False):
        key = json.dumps(m, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        m2 = dict(m)
        if m.get("role") == "assistant":
            for k in ("reasoning", "reasoning_content", "content"):
                if isinstance(m.get(k), str):
                    m2[k] = self.text(m[k], count and k != "reasoning_content", "thoughts/text")
            if m.get("tool_calls"):
                m2["tool_calls"] = [self.call(tc, count) for tc in m["tool_calls"]]
        elif m.get("role") == "tool":
            m2["content"] = self.result(m.get("content") or "", self.conv.get(m.get("tool_call_id")))
        elif m.get("role") == "user":           # compaction summaries name the files the 31B wrote
            c = m.get("content")
            if isinstance(c, list):
                m2["content"] = [{**x, "text": self.text(x["text"], True, "summaries")} if isinstance(x, dict) and
                                 isinstance(x.get("text"), str) else x for x in c]
            elif isinstance(c, str):
                m2["content"] = self.text(c, True, "summaries")
        self.cache[key] = m2
        return m2


CLEANUP_CMD = re.compile(r"^\s*(?:(?:rm|git|ls|cat|echo|find|head|tail|wc|true)\b[^;&|]*?(?:\s*(?:&&|;|\|\|?)\s*|$))+$")
PATCH_CLAIM = re.compile(r"working tree|patch|untracked|files_changed|git status|git diff|included|clean ?up|before submit", re.I)


def reply_actions(rec):
    """(call index, tool-call id, name, args) of every reply: the target, or for an excluded call the next call's history"""
    calls, out = rec["calls"], []
    for i, c in enumerate(calls):
        if c["target"]:
            tcs = [(tc.get("id"), tc["name"], tc["args"]) for tc in c["target"]["tool_calls"]]
        else:
            nxt = next((c2 for c2 in calls[i + 1:] if c2["segment"] == c["segment"]), None)
            m = nxt["input_messages"][len(c["input_messages"])] if nxt and len(nxt["input_messages"]) > len(c["input_messages"]) else {}
            tcs = [(tc.get("id"), tc["function"]["name"], tc["function"]["arguments"]) for tc in m.get("tool_calls") or []]
        for tid, name, args in tcs:
            try:
                a = json.loads(args or "{}")
            except Exception:
                a = {}
            out.append((i, tid, name, a if isinstance(a, dict) else {}))
    return out


RE_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")


def heredoc_writes(cmd, moved):
    """{/tmp path: content} written by shell heredocs in one command (`cat > P << 'D'` / `cat <<D > P`)"""
    out = {}
    first, _, rest = cmd.partition("\n")
    m, t = RE_HEREDOC.search(first), re.search(r">\s*(\S+)", first)
    if m and t and t.group(1) in moved:
        lines = rest.split("\n")
        if m.group(1) in lines:
            out[t.group(1)] = "\n".join(lines[:lines.index(m.group(1))]) + "\n"
    return out


def py_edit_apply(cmd, content):
    """our py_edit heredoc → the edited content (None when it would not apply: count ≠ 1)"""
    try:
        old = ast.literal_eval(re.search(r"^old = (.*)$", cmd, re.M).group(1))
        new = ast.literal_eval(re.search(r"^new = (.*)$", cmd, re.M).group(1))
    except Exception:
        return None
    return content.replace(old, new) if content is not None and content.count(old) == 1 else None


def newfile_diff(path, content):
    """the `git diff` text of an untracked file after `git add -N .` (its length matched the kit's patch_size delta)"""
    data = content.encode()
    h = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()[:7]
    lines = content.split("\n")
    end_nl = content.endswith("\n")
    if end_nl:
        lines = lines[:-1]
    body = "".join("+" + ln + "\n" for ln in lines) + ("" if end_nl or not lines else "\\ No newline at end of file\n")
    rng = "" if len(lines) == 1 else f",{len(lines)}"
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..{h}\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1{rng} @@\n" + body


def tool_result(rec, tid):
    return next((m.get("content") or "" for c in rec["calls"] for m in c["input_messages"]
                 if m.get("role") == "tool" and m.get("tool_call_id") == tid), None)


def cut(rec, cut_calls, acts):
    """drop calls (by index) from the trajectory and their reply + tool results from every later history; renumber"""
    gone = {tid for i, tid, _, _ in acts if i in cut_calls}
    keep = []
    for ci, c in enumerate(rec["calls"]):
        if ci in cut_calls:
            continue
        c["input_messages"] = [m for m in c["input_messages"]
                               if not (m.get("role") == "tool" and m.get("tool_call_id") in gone)
                               and not (m.get("role") == "assistant" and m.get("tool_calls")
                                        and all(tc.get("id") in gone for tc in m["tool_calls"]))]
        keep.append(c)
    rec["calls"] = keep
    for n, c in enumerate(rec["calls"], 1):
        c["call"] = n


def cut_repeats(rec, stats):
    """A call identical to the previous one (same tool, byte-identical arguments) after that one failed is cut, with
    its error, from the trajectory and every later history: the first failure and its error stay, the next different
    call follows it. Our prompt: "Never repeat an identical failed edit" (user decision 2026-11-03 on python-markdown
    #713: one malformed edit_file — new_string closed with a backtick, old_string swallowed — sent 9 times)."""
    acts = reply_actions(rec)
    by_call = {}
    for i, tid, name, a in acts:
        by_call.setdefault(i, []).append((tid, name, json.dumps(a, sort_keys=True)))
    drop, run = set(), 0
    for i in range(1, len(rec["calls"])):
        prev, cur = by_call.get(i - 1), by_call.get(i)
        same = bool(prev) and bool(cur) and [x[1:] for x in prev] == [x[1:] for x in cur] \
            and rec["calls"][i]["segment"] == rec["calls"][i - 1]["segment"] and not rec["calls"][i]["events_before"]
        failed = same and all('"error"' in (tool_result(rec, tid) or "")[:60] for tid, _, _ in prev)
        if failed:
            drop.add(i)
            by_call[i] = prev                    # compare the next one with the kept failure
            run += 1
        elif run:
            stats["repeats: runs"] += 1
            run = 0
    if run:
        stats["repeats: runs"] += 1
    if drop:
        kinds = Counter(rec["calls"][i]["excluded"] or "target" for i in drop)
        for k, v in kinds.items():
            stats[f"repeats: cut calls that were {k}"] += v
        stats["repeats: trajectories"] += 1
        cut(rec, drop, acts)
    return rec


def fix_patch_view(rec, stats, ops=None):
    """After the /tmp move, what the 31B saw of its patch no longer fits: a submit_patch made while a scratch file was
    still in the workspace counted it (files_changed, patch_size), and the 31B then reasoned about it, deleted it and
    submitted again. Here such a submit returns what the later clean submit returned, the clean-up calls in between
    (rm / git / ls / get_status / the re-submit; no compaction inside) are cut from the trajectory and from every later
    history, and any remaining thought that says a /tmp file belongs to the working tree or the patch is not a target
    (`excluded: "scratch_patch_reasoning"`). A submit with no clean submit after it keeps its numbers minus the files
    (files_changed only; counted as approximate)."""
    moved = set((rec.get("scratch_moved") or {}).values())
    if not moved:
        return rec
    acts = reply_actions(rec)
    rel = {v: k for k, v in (rec.get("scratch_moved") or {}).items()}
    ops = ops or {}
    present, at, content, diff_at = set(), [], {}, []
    for i, tid, name, a in acts:
        if name == "run_command":
            cmd = a.get("command", "")
            op = ops.get(tid)
            if op and op[0] == "write":
                content[rec["scratch_moved"][op[1]]] = op[2]
            elif op and op[0] == "edit":
                tp = rec["scratch_moved"][op[1]]
                res = tool_result(rec, tid) or ""
                if '"status": "ok"' in res[:40] and content.get(tp) is not None and content[tp].count(op[2]) == 1:
                    content[tp] = content[tp].replace(op[2], op[3])
            else:
                content.update(heredoc_writes(cmd, moved))
            present |= {m.group(1) for m in RE_SHELL_OUT.finditer(cmd) if m.group(1) in moved}
            if re.search(r"\brm\b", cmd):
                gone_now = {p for p in moved if p in cmd}
                present -= gone_now
                for p in gone_now:
                    content.pop(p, None)
        at.append(set(present))
        # what the scratch files added to a patch at this moment (None when a content is unknown)
        sizes = [len(newfile_diff(rel[p], content[p])) if content.get(p) is not None else None for p in present]
        diff_at.append(None if None in sizes else sum(sizes))
    cut_calls, fixed = set(), {}
    for k, (i, tid, name, a) in enumerate(acts):
        if name != "submit_patch" or not at[k] or tid in fixed or i in cut_calls:
            continue
        k2 = next((j for j in range(k + 1, len(acts)) if acts[j][2] == "submit_patch" and not at[j]), None)
        window = set(range(acts[k + 1][0], acts[k2][0] + 1)) if k2 is not None and k2 > k + 0 and k + 1 <= k2 else set()
        ok = bool(window) and all(
            acts[j][2] in ("submit_patch", "get_status") or (acts[j][2] == "run_command" and CLEANUP_CMD.match(acts[j][3].get("command", "")))
            for j in range(k + 1, k2 + 1)) and \
            all(any(x[0] == ci for x in acts[k + 1:k2 + 1]) and not rec["calls"][ci]["events_before"] for ci in window) and \
            acts[k][0] not in window
        clean = tool_result(rec, acts[k2][1]) if ok else None
        r = tool_result(rec, tid) or ""
        ps = re.search(r'"patch_size": (\d+)', r)
        if ok and clean and '"status": "ok"' in clean:
            fixed[tid] = clean
            cut_calls |= window
            stats["patch view: submits given the clean result"] += 1
            cs = re.search(r'"patch_size": (\d+)', clean)
            if ps and cs and diff_at[k] is not None:     # check of the size formula where no source edit came between
                d = int(ps.group(1)) - diff_at[k] - int(cs.group(1))
                stats["patch view: size formula = clean result" if d == 0 else "patch view: size formula ≠ clean result"] += 1
        else:
            m = re.search(r'"files_changed": (\d+)', r)
            if m:
                r = r.replace(m.group(0), f'"files_changed": {max(1, int(m.group(1)) - len(at[k]))}')
            if ps and diff_at[k] is not None:
                r = r.replace(ps.group(0), f'"patch_size": {int(ps.group(1)) - diff_at[k]}')
                stats["patch view: submits recomputed (files_changed and patch_size)"] += 1
            else:
                stats["patch view: submits with files_changed lowered only (content unknown)"] += 1
            fixed[tid] = r
    for c in rec["calls"]:
        for m in c["input_messages"]:
            if m.get("role") == "tool" and m.get("tool_call_id") in fixed:
                m["content"] = fixed[m["tool_call_id"]]
    if cut_calls:
        cut(rec, cut_calls, acts)
        stats["patch view: clean-up calls cut"] += len(cut_calls)
        stats["patch view: trajectories cut"] += 1
    for c in rec["calls"]:
        t = c["target"]
        if not t:
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n+", t.get("reasoning") or ""):
            if any(p in sent for p in moved) and PATCH_CLAIM.search(sent):
                c["target"], c["excluded"] = None, "scratch_patch_reasoning"
                stats["scratch_patch_reasoning_excluded"] += 1
                if len(stats["patch claim examples"]) < 6:
                    stats["patch claim examples"].append(f"{rec['instance_id']} c{c['call']}: {sent.strip()[:150]}")
                break
    return rec


def scratch_to_tmp(row, stats):
    mp = scratch_map(row)
    if not mp:
        return row
    stats["trajectories with scratch files"] += 1
    stats["scratch files moved"] += len(mp)
    X = Scratch(mp, stats)
    steps = []
    for s in row["sft_examples"]:
        s2 = dict(s)
        rm = s.get("response_message") or {}
        rm2 = X.message({**rm, "role": "assistant"}, count=True)
        if any(X.conv.get(tc.get("id")) == "edit_file" for tc in rm.get("tool_calls") or []):
            s2["_excluded"] = "scratch_edit"
        s2["response_message"] = rm2
        req = dict(s["request"])
        req["messages"] = [X.message(m) for m in s["request"]["messages"]]
        s2["request"] = req
        steps.append(s2)
    out = dict(row)
    out["sft_examples"] = steps
    out["scratch_moved"] = mp
    out["scratch_ops"] = X.ops          # the original write/edit of each converted call (exact patch sizes)
    return out


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
        if bad:
            excluded = "schema_violation"
            stats["schema_violation_excluded"] += 1
            stats["violations"][bad] += 1
        elif step.get("_excluded"):
            excluded = step["_excluded"]
            stats[f"{excluded}_excluded"] += 1
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
    ap.add_argument("--scratch", choices=["tmp", "keep"], default="tmp", help="workspace scratch files: move to /tmp or keep")
    ap.add_argument("--mask-errors", choices=["on", "off"], default="on",
                    help="replies whose tool call failed as an action → not targets (train/mask_errors.py); off = count only")
    a = ap.parse_args()
    system = open(a.system).read().rstrip() + ADK_SUFFIX
    tools = json.load(open(sorted(glob.glob(os.path.join(a.tools_from, "*.json")))[0]))["tools"]
    ours = {t["function"]["name"]: t for t in tools}
    print(f"system prompt: latest {a.system} + ADK suffix = {len(system)} chars · tools: {len(tools)} schemas from {a.tools_from}"
          f" · budget lines → {LATEST_BUDGET['minutes']} minutes / {LATEST_BUDGET['calls']} calls · scratch files: {a.scratch}"
          f" · tool-error masking: {a.mask_errors}")
    for d in (a.out, a.orig, a.ref):
        if d:
            os.makedirs(d, exist_ok=True)
    stats = Counter()
    stats["kept_write_paths"], stats["violations"], moved_names = [], Counter(), Counter()
    stats["patch claim examples"] = []
    schema_diff = Counter()
    tasks, repos = set(), Counter()
    for line in open(a.data):
        row = json.loads(line)
        kit_tools = {t["function"]["name"]: t for t in row["sft_examples"][0]["request"]["tools"]}
        assert set(kit_tools) == set(ours), f"{row['task_id']}: tool set differs: {sorted(kit_tools)}"
        for n, t in kit_tools.items():
            if t != ours[n]:
                schema_diff["whitespace only" if norm_ws(t) == norm_ws(ours[n]) else f"{n}: content"] += 1
        moved = scratch_to_tmp(row, stats) if a.scratch == "tmp" else row
        moved_names.update(os.path.basename(p) for p in moved.get("scratch_moved", {}))
        rec = convert(moved, system, tools, stats)
        rec["scratch_moved"] = moved.get("scratch_moved", {})
        fix_patch_view(rec, stats, moved.get("scratch_ops", {}))
        cut_repeats(rec, stats)
        mask_tool_errors(rec, stats, a.mask_errors)
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
                json.dump(convert(row, system, tools, Counter(kept_write_paths=[], violations=Counter()), rewrite=False), w,
                          ensure_ascii=False)
        if a.ref:
            with open(os.path.join(a.ref, f"{name}.json"), "w") as w:
                json.dump(reference(row), w, ensure_ascii=False)
        print(f"{name:<44} calls {len(rec['calls']):>3} targets {n_t:>3} segments {rec['calls'][-1]['segment'] + 1:>2} "
              f"compaction {ev.count('compaction')} reset {ev.count('reset')}")
    print(f"tool schemas vs ours: {dict(schema_diff)}")
    print(f"schema violations excluded {stats['schema_violation_excluded']}: {dict(stats['violations'])}")
    print(f"scratch → /tmp: {stats['scratch files moved']} files in {stats['trajectories with scratch files']} trajectories "
          f"(names: {dict(moved_names.most_common(10))}) · "
          + " · ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k.startswith(("scratch ", "write_file results", "edit_file results",
                                                                                       "read_file results", "tool outputs")))
          + f" · scratch edits not targets {stats['scratch_edit_excluded']}")
    print("patch view after the move: " + " · ".join(f"{k[len('patch view: '):]} {v}" for k, v in sorted(stats.items())
                                                       if k.startswith("patch view: "))
          + f" · thoughts saying a /tmp file is in the tree/patch, not targets {stats['scratch_patch_reasoning_excluded']}")
    for e in stats["patch claim examples"]:
        print("   e.g.", e)
    print("repeated failed calls: " + " · ".join(f"{k[len('repeats: '):]} {v}" for k, v in sorted(stats.items())
                                                  if k.startswith("repeats: ")))
    print(summary(stats, a.mask_errors))
    print(f"write_file targets left (workspace, not scratch) {stats['write_file_kept']}: {stats['kept_write_paths']}")
    keys = ("trajectories", "calls", "targets", "segments", "compaction", "reset", "summariser_calls",
            "non_canonical_targets", "budget_lines_rewritten")
    print("TOTAL " + " · ".join(f"{k} {stats[k]}" for k in keys) + f" · tasks {len(tasks)} · repos {len(repos)} → {a.out}"
          + (f" (+ {a.orig}, {a.ref})" if a.orig else ""))
    return 0 if stats["trajectories"] else 1


if __name__ == "__main__":
    sys.exit(main())
