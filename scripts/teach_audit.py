#!/usr/bin/env python3
"""B-11: the checks on a directory of teacher trajectories, and a local page to read them (2026-11-21).

    python3 scripts/teach_audit.py --out runs/teach_h2 [--ids runs/teach_h2/first_200.ids] [--name all] --page runs/audit/h2

One way of counting for `teacher_gen.py inspect` and for the page. Everything is keyed by (task, attempt): with three
attempts per task a count keyed by the task alone mixes them (the ledger match of the old inspect matched nothing).
Every model call of an attempt is one of four kinds, and the shares are taken on the kind they belong to:
    summary   the harness's compaction call (no tools, not a turn of the agent)
    tool      a reply with at least one tool call — the turns a training set takes its targets from
    text      a reply without a tool call before the submit (mostly a thought cut by the output limit)
    closing   a reply after `submit_patch`
Each check has a status — pass / FAIL (a gate), note (a count above its usual size, to be read), info (a count) — a
value, and the list of offenders as (task, attempt, call).

The page is written as plain files (index.html + audit.js + traj/NN.js) and opens from the disk: no server, no
network, nothing uploaded. The trajectory text is data from a model and from repositories: the page shows it with
textContent only. Stdlib only. Knobs (all printed): --ids --name --attempts --min-thought --max-patch --eval-repos --page.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teach_reminder  # noqa: E402

try:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
    import mc_rules  # noqa: E402
except Exception as e:  # the rule preview is then left out, and said so
    mc_rules = None; MC_ERR = repr(e)

BUDGET_END_RE = re.compile(r"Agent exceeded (session timeout|turns budget|maximum allowed)", re.I)
CTX_END_RE = re.compile(r"ContextWindowExceeded", re.I)

# does a reply repeat or refer to the teacher-only reminder? (2026-11-20) The gate on '<reminder>' finds only the
# literal tag; these find the sentence's own wording and talk about having been told to reason. A count, not a gate:
# every match is listed with its context and read by a person before the turn goes into a training set.
ECHO_PATS = (("the tag", r"</?reminder>"), ("the word remind*", r"\bremind"),
             ("before every tool call", r"before (every|each|any|the|a|my|making|calling)[^.\n]{0,20}tool call"),
             ("reason briefly / think before", r"reason(ing)? briefly|think(ing)? before|brief(ly)? reason"),
             ("told to reason", r"(instruct|told|asked|required|supposed)[^.\n]{0,30}(reason|think)"),
             ("why this step is needed / expect to see", r"why this step is needed|what (i|you) expect to see"),
             ("the rule's wording", r"without (thinking|reasoning) first|(think|reason)(ing)? (first )?before (each|every|any)|"
                                    r"(rule|prompt|instruction)s?\b[^.\n]{0,40}\b(think|reason)(ing)? (first|before|in 1-3)"))


def echo_hits(resp):
    """→ [(pattern name, where, context)] for one logged reply (its thought, visible text and tool-call arguments)"""
    c = resp.get("content")
    parts = (("thought", resp.get("reasoning") or ""), ("text", c if isinstance(c, str) else json.dumps(c or "")),
             ("tool args", " ".join(str(t.get("args") or "") for t in resp.get("tool_calls") or [])))
    out = []
    for where, txt in parts:
        for name, pat in ECHO_PATS:
            for m in re.finditer(pat, txt, flags=re.I):
                out.append((name, where, txt[max(0, m.start() - 90):m.end() + 90].replace("\n", " ")))
    return out


def q(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def pct(a, b):
    return f"{a:,} / {b:,} = {100 * a / b:.1f} %" if b else f"{a:,} / 0"


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return "" if c is None else json.dumps(c)


# the V21 prompt has every command start with a clock snippet (date > /dev/shm/.t0 … echo CLOCK_MIN=…): it is not an
# edit of the repository and is taken off before a command is classified
CLOCK_RE = re.compile(r"^\s*\[ -s /dev/shm/\.t0 \].*?echo CLOCK_MIN=0\s*;?\s*", re.S)


NOISE_RE = re.compile(r"\b\d?>&\d|\d?>>?\s*/dev/(null|shm/\S+)|\btimeout\s+(-\S+\s+)*\d+[smh]?\s+")


def plain_cmd(cmd):
    """a harness command as the rules should see it: without the clock snippet, without `timeout N` in front of a
    program (the rules do not look through it: no test run was found in 283 of 283 resolved trajectories) and without
    stream redirections (`echo x; pytest 2>&1` read as a file written by echo)"""
    return NOISE_RE.sub(" ", CLOCK_RE.sub("", cmd or ""))


def harness_signals(seq_cmds, patch):
    """MetaCurate's tool-independent rules on a harness trajectory — a preview of step 5, not the filter itself.
    seq_cmds: the tool calls in order as (tool name, command or '')."""
    letters = []; reverts = 0; bad_git = False
    for name, cmd in seq_cmds:
        if name in ("edit_file", "write_file"):
            letters.append("E")
        elif name == "run_command":
            f = mc_rules.classify(plain_cmd(cmd))
            letters.append("E" if f["edit"] else "T" if f["test"] else "P" if f["pyrun"] else "R")
            reverts += f["git_revert"]; bad_git = bad_git or f["git_disallowed"]
        elif name == "read_file":
            letters.append("R")
    core = [c for c in letters if c in "ET"]
    returns = sum(1 for a, b in zip(core, core[1:]) if a == "T" and b == "E")
    n = len(letters); edits = letters.count("E"); out = []
    if not (patch or "").lstrip().startswith("diff --git"):
        out.append("no_valid_submission")
    if bad_git:
        out.append("disallowed_git")
    if returns >= mc_rules.EDIT_TEST_LOOP_MAX_RETURNS or reverts >= mc_rules.EDIT_TEST_LOOP_MAX_REVERTS:
        out.append("edit_test_loop")
    if edits == 0 or (n >= mc_rules.LOST_EXPLORATION_MIN_ACTIONS and letters.count("R") / n >= mc_rules.LOST_EXPLORATION_EXPLORE_SHARE
                      and edits <= mc_rules.LOST_EXPLORATION_MAX_EDITS):
        out.append("lost_in_exploration")
    if edits >= 1 and "T" not in letters:
        out.append("no_verification")
    return out


def read_attempt(path, leak):
    """one trajectory file → (steps, facts). A step is one model call with what the page shows of it."""
    lines = open(path, errors="replace").read().splitlines()
    L = []; leak_lines = 0
    for line in lines:
        if line.strip():
            leak_lines += any(x in line for x in leak); L.append(json.loads(line))
    results = {}; steps = []; submitted = False; prev_thought = None
    f = dict(calls=len(L), leak=leak_lines, back=[0, 0], back_miss=[], task_text="", system="", schemas={})
    for i, c in enumerate(L):
        r = c.get("resp") or {}; new = c.get("new") or []; u = c.get("usage") or {}
        for m in new:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                results.setdefault(m["tool_call_id"], _text(m.get("content")))
        if i == 0:
            f["system"] = next((_text(m.get("content")) for m in new if m.get("role") == "system"), "")
            f["task_text"] = next((_text(m.get("content")) for m in new if m.get("role") == "user"), "")
            for s in (c.get("setup") or {}).get("tool_schemas") or []:
                fn = s.get("function") or {}; p = fn.get("parameters") or {}
                f["schemas"][fn.get("name")] = (set(p.get("required") or []), set(p.get("properties") or {}))
        th = (r.get("reasoning") or "").strip(); tx = _text(r.get("content")).strip(); tcs = r.get("tool_calls") or []
        comp = bool(c.get("compaction"))
        kind = "summary" if comp else "tool" if tcs else "closing" if submitted else "text"
        st = dict(n=c.get("call", i + 1), kind=kind, th=th, tx=tx, fin=r.get("finish_reason"), s=c.get("s") or 0,
                  tin=u.get("prompt") or 0, tout=u.get("completion") or 0, treason=u.get("reasoning") or 0,
                  reset=bool(c.get("reset")), calls=[dict(name=t.get("name") or "", args=t.get("args") if isinstance(t.get("args"), str) else json.dumps(t.get("args")), id=t.get("id")) for t in tcs])
        if not comp:
            said = [_text(m.get("content")) for m in new if m.get("role") == "user"]
            if c.get("reset"):
                st["ctx"] = said            # the history as rebuilt after a compaction: its user messages (the summary is one)
            elif i and said:
                st["nudge"] = said          # a message of the harness between two replies
            # the previous reply as it comes back in this request: does its thought come with it?
            if prev_thought and not c.get("reset"):
                am = [m for m in new if m.get("role") == "assistant"]
                if am:
                    f["back"][1] += 1
                    ok = bool(am[0].get("reasoning_content") or am[0].get("reasoning") or am[0].get("thinking_blocks"))
                    f["back"][0] += ok
                    if not ok:
                        f["back_miss"].append(st["n"])
            prev_thought = bool(th)
        else:
            st["ask"] = "\n".join(_text(m.get("content")) for m in new if m.get("role") == "user")[:300]
            prev_thought = None
        st["hits"] = [list(h) for h in echo_hits(r)]
        st["after_submit"] = submitted
        submitted = submitted or any(t.get("name") == "submit_patch" for t in tcs)
        steps.append(st)
    for st in steps:
        for t in st["calls"]:
            res = results.get(t["id"])
            t["res"] = res
            try:
                t["status"] = json.loads(res).get("status") if res else None
            except Exception:
                t["status"] = "unparsed" if res else None
            t.pop("id", None)
    f["submitted"] = submitted
    return steps, f


def load_rows(out, ids=None, name=""):
    rows = {}
    p = os.path.join(out, "attempts.jsonl")
    for line in (open(p) if os.path.exists(p) else []):
        if line.strip():
            r = json.loads(line); rows[(r["task"], r["attempt"])] = r
    keep = [r for r in rows.values() if (ids is None or r["task"] in ids) and (not name or os.path.basename(r.get("run") or "").startswith(name))]
    return sorted(keep, key=lambda r: (r["task"].lower(), r["attempt"])), len(rows)


def audit(out, ids=None, name="", attempts=3, min_thought=90.0, eval_repos=(), bodies=False, max_patch=20000):
    """→ {meta, checks, charts, attempts, [bodies]} for the done attempts of `out` (optionally: only the tasks in
    `ids`, only the runs whose name starts with `name`)."""
    rows, n_all = load_rows(out, ids, name)
    done = [r for r in rows if r["state"] == "done"]
    gold_p = os.path.join(out, "gold.json"); gold = json.load(open(gold_p)) if os.path.exists(gold_p) else {}
    repo = {}
    tp = os.path.join(out, "tasks.jsonl")
    for line in (open(tp, errors="replace") if os.path.exists(tp) else []):
        try:
            t = json.loads(line); repo[t["instance_id"]] = t.get("repo") or ""
        except Exception:
            pass
    led = collections.defaultdict(list); lf = os.path.join(out, "ledger", "calls.jsonl")
    for line in (open(lf) if os.path.exists(lf) else []):
        c = json.loads(line)
        if not c.get("err"):
            led[c["key"]].append(c)
    LEAK = teach_reminder.needles()
    TOOLS = None
    C = collections.Counter
    A = []; B = []; checks = []; off = collections.defaultdict(list)       # off[check id] = [[attempt index, call, note]]
    k = C(); by_depth = collections.defaultdict(lambda: [0, 0]); by_tool = collections.defaultdict(lambda: [0, 0])
    th_len = []; multi = C(); tools = C(); res_stat = collections.defaultdict(C); prompts = set(); sig = C(); sig_res = C()
    patches = collections.defaultdict(dict); echo_rows = []; gemma_max = []; shapes = C(); ctx_req = []; ctx_pre = []; ctx_sum = []
    for r in done:
        i = len(A); run = os.path.join(out, "runs", os.path.basename(r.get("run") or "?"))
        path = os.path.join(run, "traj", r["task"] + ".jsonl")
        a = dict(i=i, task=r["task"], attempt=r["attempt"], repo=repo.get(r["task"], ""), resolved=bool(r.get("resolved")),
                 usd=round(r.get("usd") or 0, 4), seconds=round(r.get("seconds") or 0), run=os.path.basename(r.get("run") or ""),
                 patch_chars=r.get("patch_chars") or 0, end=(r.get("harness_error") or "")[:80], flags=[],
                 calls=0, tool_turns=0, thought_turns=0, tool_calls=r.get("tool_calls") or 0, summaries=0, submitted=False)
        A.append(a)
        if not os.path.exists(path):
            off["files"].append([i, 0, "no trajectory file"]); B.append(None); continue
        steps, f = read_attempt(path, LEAK)
        B.append(dict(task=f["task_text"], steps=steps) if bodies else None)
        TOOLS = TOOLS or set(f["schemas"]) or None
        prompts.add(hashlib.sha256(f["system"].encode()).hexdigest()[:12]); a["prompt"] = hashlib.sha256(f["system"].encode()).hexdigest()[:12]
        a["calls"] = f["calls"]; a["submitted"] = f["submitted"]
        if f["leak"]:
            off["leak"].append([i, 0, f"{f['leak']} log lines hold the reminder text"])
        k["leak"] += f["leak"]; k["back_ok"] += f["back"][0]; k["back_n"] += f["back"][1]
        off["back"] += [[i, n, "the previous reply's thought is not in this request"] for n in f["back_miss"]]
        n_led = len(led.get(f"{r['task']}#{r['attempt']}", []))
        if lf and os.path.exists(lf) and n_led != f["calls"]:
            off["ledger"].append([i, 0, f"trajectory {f['calls']} calls, ledger {n_led}"])
        seq = []; n_agent = 0; last_tin = 0
        for st in steps:
            kd = st["kind"]; k[kd] += 1; th = bool(st["th"])
            if a["resolved"]:               # the context sizes of what would be trained on: every request of a resolved attempt
                if kd == "summary":
                    ctx_sum.append(st["tin"]); ctx_pre.append(last_tin)
                else:
                    last_tin = st["tin"]
                    if kd == "tool":
                        ctx_req.append(st["tin"])
            if st["treason"] > 0 and not th:
                off["reasoning"].append([i, st["n"], f"{st['treason']} reasoning tokens, no thought text"])
            if kd == "summary":
                a["summaries"] += 1; k["summary_thought"] += th
                for h in st["hits"]:
                    echo_rows.append((i, st["n"], "summary", h))
                continue
            n_agent += 1
            for h in st["hits"]:
                echo_rows.append((i, st["n"], "closing" if kd == "closing" else "text" if kd == "text" else "tool", h))
            if kd == "text":
                k["text_length"] += st["fin"] == "length"
                off["text_only"].append([i, st["n"], "finish " + str(st["fin"]) + (" · no text" if not st["tx"] else "")])
                if st["fin"] == "length":
                    off["cut"].append([i, st["n"], f"thought of {len(st['th']):,} chars cut by the output limit"])
            if kd != "tool":
                continue
            a["tool_turns"] += 1; a["thought_turns"] += th; k["tool_thought"] += th
            if a["resolved"]:
                k["res_tool"] += 1; k["res_tool_thought"] += th
            d = by_depth["01" if n_agent == 1 else "02-03" if n_agent <= 3 else "04-08" if n_agent <= 8 else "09-25" if n_agent <= 25 else "26+"]
            d[0] += 1; d[1] += th
            shapes[("thought" if th else "no thought") + (" + text" if st["tx"] else "")] += 1
            if th:
                th_len.append(len(st["th"]))
            else:
                off["no_thought"].append([i, st["n"], ("first call" if n_agent == 1 else f"call {n_agent}") + " · " + st["calls"][0]["name"][:20]])
                k["no_thought_first"] += n_agent == 1
            if not th and not st["tx"]:
                off["bare"].append([i, st["n"], st["calls"][0]["name"][:20]])
            if st["tx"]:
                k["text_beside"] += 1
            multi[min(len(st["calls"]), 5)] += 1
            if len(st["calls"]) > 1:
                off["multi"].append([i, st["n"], f"{len(st['calls'])} tool calls in one reply"])
            bt = by_tool[st["calls"][0]["name"] if st["calls"][0]["name"] in (TOOLS or ()) else "other"]; bt[0] += 1; bt[1] += th
            for t in st["calls"]:
                nm = t["name"]; known = TOOLS is None or nm in TOOLS
                tools[nm if known else "(not one of the tools)"] += 1
                if not known:
                    off["foreign"].append([i, st["n"], "name: " + nm[:60].replace("\n", " ")]); continue
                try:
                    args = json.loads(t["args"] or "{}")
                except Exception:
                    args = None
                req, props = f["schemas"].get(nm, (set(), set()))
                if not isinstance(args, dict):
                    off["schema"].append([i, st["n"], f"{nm}: arguments are not a JSON object"])
                elif (req - set(args)) or (props and set(args) - props):
                    off["schema"].append([i, st["n"], f"{nm}: missing {sorted(req - set(args))} · unknown {sorted(set(args) - props)}"])
                res_stat[nm][t["status"] or "no result logged"] += 1
                if t["status"] not in ("ok", None):
                    k["tool_err"] += 1
                    if nm == "edit_file":
                        off["edit_fail"].append([i, st["n"], (t["res"] or "")[:120].replace("\n", " ")])
                k["tool_res"] += t["status"] is not None
                seq.append((nm, (args or {}).get("command", "") if isinstance(args, dict) else ""))
        gemma_max.append(max([st["tin"] for st in steps] or [0])); a["gemma_max"] = gemma_max[-1]
        pf = os.path.join(run, "patches", r["task"] + ".patch")
        patch = open(pf, errors="replace").read() if os.path.exists(pf) else ""
        if a["resolved"]:
            patches[r["task"]][r["attempt"]] = hashlib.sha256(patch.encode()).hexdigest()
        if mc_rules:
            a["signals"] = harness_signals(seq, patch)
            for s in a["signals"]:
                sig[s] += 1; sig_res[s] += a["resolved"]
                if a["resolved"] and s != "no_verification":
                    off["rule_" + s].append([i, 0, s])
        if a["resolved"] and not f["submitted"]:
            off["no_submit"].append([i, 0, "resolved, but submit_patch was never called (" + (a["end"] or "no harness error") + ")"])
        if not a["patch_chars"]:
            off["empty_patch"].append([i, 0, a["end"] or "empty patch"])
        if BUDGET_END_RE.search(a["end"]) or CTX_END_RE.search(a["end"]):
            off["limit"].append([i, 0, a["end"]])

    def add(cid, group, en, zh, status, value, note="", note_zh=""):
        checks.append(dict(id=cid, group=group, en=en, zh=zh, status=status, value=value, note=note, note_zh=note_zh, off=off.get(cid, [])))
    G = ("integrity", "gates", "reminder", "thoughts", "shape", "outcome", "fitness", "cost")
    gate = lambda bad: "FAIL" if bad else "pass"
    # --- integrity
    states = C(r["state"] for r in rows)
    add("states", G[0], "attempt rows that are not `done`", "不是 done 的尝试行", gate(len(rows) - len(done)), f"{len(rows) - len(done)} of {len(rows)}", str(dict(states)), str(dict(states)))
    add("files", G[0], "done rows without a trajectory file", "done 行缺轨迹文件", gate(off["files"]), f"{len(off['files'])} of {len(done)}")
    per_task = collections.defaultdict(list)
    for a in A:
        per_task[a["task"]].append(a)
    wrong_n = [t for t, v in per_task.items() if len(v) != attempts]
    for t in wrong_n:
        off["n_attempts"].append([per_task[t][0]["i"], 0, f"{len(per_task[t])} done attempts, {attempts} wanted"])
    add("n_attempts", G[0], f"tasks whose number of done attempts is not {attempts}", f"done 尝试数不是 {attempts} 的任务", gate(wrong_n), f"{len(wrong_n)} of {len(per_task)} tasks")
    nogold = sorted({a["task"] for a in A if gold and not gold.get(a["task"])})
    for t in nogold:
        off["gold"].append([per_task[t][0]["i"], 0, "attempted although its gold patch did not resolve / was not checked"])
    add("gold", G[0], "attempted tasks whose gold patch does not resolve", "gold 补丁不通过却被尝试的任务", gate(nogold), f"{len(nogold)} of {len(per_task)} tasks",
        f"gold checked on {len(gold)} tasks of the directory, {sum(map(bool, gold.values()))} resolve", f"目录里 gold 检查过 {len(gold)} 题，{sum(map(bool, gold.values()))} 题通过")
    add("ledger", G[0], "attempts whose ledger call count differs from the trajectory's", "账本调用数与轨迹调用数不等的尝试", "note" if off["ledger"] else "pass",
        f"{len(off['ledger'])} of {len(done)}", "ledger rows with an error are left out; a retried attempt leaves extra rows", "带 err 的账本行不计；重试过的尝试会多出行")
    # rows written before the id and the place were recorded are the tagged line at the end (the only form there was)
    sets = C((r.get("reminder", "off"), r.get("reminder_id", "1") if r.get("reminder") == "on" else "-", r.get("reminder_place", "end") if r.get("reminder") == "on" else "-", r.get("hist", "off")) for r in done)
    add("settings", G[0], "settings mixed in one directory (reminder, its id and place, history thoughts)", "同一目录里混用的设置（提醒、编号、位置、历史思考）", gate(len(sets) > 1),
        f"{len(sets)} setting(s)", " · ".join(f"{'/'.join(map(str, s))} × {n}" for s, n in sets.most_common()), " · ".join(f"{'/'.join(map(str, s))} × {n}" for s, n in sets.most_common()))
    add("prompt", G[0], "distinct system prompts", "不同的系统提示词", gate(len(prompts) > 1), f"{len(prompts)}", " ".join(sorted(prompts)), " ".join(sorted(prompts)))
    # --- gates
    add("reasoning", G[1], "calls with reasoning tokens but no thought text (a thought produced and lost)", "有 reasoning token 却没有思考文字的调用", gate(off["reasoning"]),
        f"{len(off['reasoning'])} of {k['summary'] + k['tool'] + k['text'] + k['closing']:,} calls", "read from each call's own usage, no ledger match needed", "直接读每次调用自己的 usage，不需要对账本")
    hist_on = bool(done) and all(r.get("hist", "off") == "on" for r in done)
    add("back", G[1], "replies with a thought that come back WITH it in the next request", "带思考的回复在下一次请求里带回思考", (gate(k["back_ok"] != k["back_n"] or not k["back_n"]) if hist_on else "info"),
        pct(k["back_ok"], k["back_n"]), "" if hist_on else "no gate: history thoughts are off in this directory", "" if hist_on else "不设门槛：这个目录没开历史思考")
    add("leak", G[1], f"trajectory log lines that hold the reminder text ({len(LEAK)} needles)", f"轨迹日志里含提醒原文的行（{len(LEAK)} 个特征串）", gate(k["leak"]), f"{k['leak']}")
    # --- reminder
    n_no_rem = n_calls_ok = 0; runs_used = sorted({a["run"] for a in A})
    for rn in runs_used:
        cf = os.path.join(out, "calls", rn + ".jsonl")
        for line in (open(cf) if os.path.exists(cf) else []):
            c = json.loads(line)
            if not c.get("err"):
                n_calls_ok += 1; n_no_rem += not c.get("reminder")
    rem_on = bool(done) and all(r.get("reminder", "off") == "on" for r in done)
    place_end = all(r.get("reminder_place", "end") == "end" for r in done)
    want = k["summary"] + k["closing"]
    if rem_on and place_end and ids is None and n_calls_ok:
        diff = abs(n_no_rem - want)
        add("placement", G[2], "requests sent without the reminder = summaries + closing replies", "不带提醒的请求数 = 摘要数 + 收尾回复数", "pass" if diff <= max(3, 0.005 * want) else "note",
            f"{n_no_rem:,} vs {want:,}", "call log of the runs; the reminder is not for a summary and not after the submit", "按运行的调用日志；摘要请求和提交之后都不该带提醒")
    else:
        add("placement", G[2], "requests sent without the reminder = summaries + closing replies", "不带提醒的请求数 = 摘要数 + 收尾回复数", "info", f"{n_no_rem:,} of {n_calls_ok:,} requests in {len(runs_used)} runs · summaries + closing here {want:,}",
            "not comparable: " + ("--ids takes part of the runs" if ids is not None else "reminder off, or not at the end, or no call log"), "不可比：" + ("--ids 只取了运行的一部分" if ids is not None else "提醒未开、不在末尾，或没有调用日志"))
    seen = {}
    for i, n, where, h in echo_rows:
        seen.setdefault((i, n, where), []).append(h)
    for (i, n, where), hs in sorted(seen.items()):
        cid = {"tool": "echo_tool", "text": "echo_text", "closing": "echo_close", "summary": "echo_summary"}[where]
        off[cid].append([i, n, " ‖ ".join(sorted({f"[{h[0]} · {h[1]}] …{h[2][:170]}…" for h in hs}))[:520]])
        A[i]["flags"].append(cid)
    add("echo_tool", G[2], "tool-call turns before the submit that match a reminder pattern (training turns; read each)", "提交前的工具调用回合里匹配提醒短语的（训练回合，逐条读）", "note" if off["echo_tool"] else "pass",
        f"{len(off['echo_tool'])} of {k['tool']:,}", f"{len(ECHO_PATS)} fixed patterns; a paraphrase is not found, a match may be something else", f"{len(ECHO_PATS)} 个固定短语；转述查不到，命中也可能是别的意思")
    add("echo_text", G[2], "text-only replies before the submit that match", "提交前的纯文本回复里匹配的", "note" if off["echo_text"] else "pass", f"{len(off['echo_text'])} of {k['text']:,}")
    add("echo_close", G[2], "closing replies (after the submit) that match", "提交后的收尾回复里匹配的", "note" if off["echo_close"] else "pass", f"{len(off['echo_close'])} of {k['closing']:,}", "not training targets", "不是训练目标")
    add("echo_summary", G[2], "compaction summaries that match (a summary is context of later turns)", "压缩摘要里匹配的（摘要是后面回合的上下文）", "note" if off["echo_summary"] else "pass", f"{len(off['echo_summary'])} of {k['summary']:,}")
    # --- thoughts
    share = 100 * k["tool_thought"] / k["tool"] if k["tool"] else 0
    add("no_thought", G[3], f"tool-call turns with a thought (gate ≥ {min_thought:g} %)", f"工具调用回合带思考（门槛 ≥ {min_thought:g} %）", gate(share < min_thought), pct(k["tool_thought"], k["tool"]),
        f"resolved trajectories {pct(k['res_tool_thought'], k['res_tool'])} · without: {k['no_thought_first']} on the first call, {k['tool'] - k['tool_thought'] - k['no_thought_first']} later · offenders = the turns without",
        f"解出的轨迹里 {pct(k['res_tool_thought'], k['res_tool'])} · 没有思考的：第一次调用 {k['no_thought_first']} 个，之后 {k['tool'] - k['tool_thought'] - k['no_thought_first']} 个 · 违例 = 没有思考的回合")
    add("cut", G[3], "thoughts cut by the output limit (a reply with no tool call)", "被输出上限截断的思考（回复里没有工具调用）", "note" if len(off["cut"]) > 0.02 * max(1, k["tool"]) else "info", f"{len(off['cut'])}")
    add("th_len", G[3], "thought length on tool-call turns, characters", "工具调用回合的思考长度（字符）", "info", f"p50 {q(th_len, .5):,} · p90 {q(th_len, .9):,} · max {max(th_len or [0]):,}",
        f"over 2,000: {sum(x > 2000 for x in th_len):,} · over 8,000: {sum(x > 8000 for x in th_len):,}", f"超过 2,000 的 {sum(x > 2000 for x in th_len):,} 个 · 超过 8,000 的 {sum(x > 8000 for x in th_len):,} 个")
    # --- shape
    add("foreign", G[4], "tool calls whose name is not one of the tools", "工具名不在工具表里的调用", "note" if off["foreign"] else "pass", f"{len(off['foreign'])} of {sum(tools.values()):,}", "tools: " + (", ".join(sorted(TOOLS)) if TOOLS else "not logged"), "工具表：" + (", ".join(sorted(TOOLS)) if TOOLS else "未记录"))
    add("schema", G[4], "tool calls whose arguments do not fit the tool's schema", "参数不符合工具 schema 的调用", "note" if off["schema"] else "pass", f"{len(off['schema'])} of {sum(tools.values()):,}")
    add("multi", G[4], "replies with several tool calls", "一次回复里有多个工具调用", "info", pct(len(off["multi"]), k["tool"]), " · ".join(f"{n if n < 5 else '5+'} calls: {v:,}" for n, v in sorted(multi.items())), " · ".join(f"{n if n < 5 else '5+'} 个调用：{v:,}" for n, v in sorted(multi.items())))
    add("text_beside", G[4], "tool-call turns with visible text beside the call", "工具调用旁带可见文字的回合", "info", pct(k["text_beside"], k["tool"]), str(dict(shapes.most_common())), str(dict(shapes.most_common())))
    add("bare", G[4], "tool-call turns with neither a thought nor text", "既没有思考也没有文字的工具调用回合", "info", pct(len(off["bare"]), k["tool"]))
    add("text_only", G[4], "replies without a tool call before the submit", "提交前没有工具调用的回复", "info", f"{k['text']:,}", f"{k['text_length']} ended by the output limit", f"其中 {k['text_length']} 个因输出上限结束")
    # --- outcome
    res = [a for a in A if a["resolved"]]
    add("resolved", G[5], "resolved attempts", "解出的尝试", "info", pct(len(res), len(A)),
        " · ".join(f"attempt {n}: {sum(a['resolved'] for a in A if a['attempt'] == n)} / {sum(1 for a in A if a['attempt'] == n)}" for n in sorted({a["attempt"] for a in A})),
        " · ".join(f"第 {n} 次：{sum(a['resolved'] for a in A if a['attempt'] == n)} / {sum(1 for a in A if a['attempt'] == n)}" for n in sorted({a["attempt"] for a in A})))
    dist = C(sum(a["resolved"] for a in v) for v in per_task.values())
    add("per_task", G[5], "tasks resolved at least once", "至少解出一次的任务", "info", pct(sum(v for n, v in dist.items() if n), len(per_task)),
        " · ".join(f"{n}×: {v}" for n, v in sorted(dist.items())), " · ".join(f"解出 {n} 次：{v} 题" for n, v in sorted(dist.items())))
    add("no_submit", G[5], "resolved attempts that never called submit_patch", "解出却没有调用 submit_patch 的尝试", "note" if off["no_submit"] else "pass", f"{len(off['no_submit'])} of {len(res)}", "the patch was taken from the working tree when a limit ended the session", "会话被限制终止时补丁取自工作区")
    add("empty_patch", G[5], "attempts with an empty patch", "空补丁的尝试", "info", f"{len(off['empty_patch'])} of {len(A)}")
    add("limit", G[5], "attempts ended by a harness limit (time, turns, context window)", "被 harness 限制终止的尝试（时间、轮数、上下文）", "info", f"{len(off['limit'])} of {len(A)}", str(dict(C(a["end"][:44] for a in A if a["end"]).most_common(4))), str(dict(C(a["end"][:44] for a in A if a["end"]).most_common(4))))
    ef = res_stat["edit_file"]; ne = sum(ef.values()) - ef["no result logged"]
    add("edit_fail", G[5], "edit_file calls that failed", "失败的 edit_file 调用", "info", pct(len(off["edit_fail"]), ne), f"all tools: {pct(k['tool_err'], k['tool_res'])} results not ok", f"全部工具：{pct(k['tool_err'], k['tool_res'])} 的返回不是 ok")
    # --- fitness (a preview of step 5)
    ev = {x.lower() for x in eval_repos}
    hit = sorted({a["task"] for a in A if a["repo"].lower() in ev})
    for t in hit:
        off["eval_repo"].append([per_task[t][0]["i"], 0, "repository of an evaluation task: " + per_task[t][0]["repo"]])
    add("eval_repo", G[6], f"tasks from a repository of the evaluation tasks ({len(ev)} repositories listed)", f"来自评测任务所在仓库的任务（清单 {len(ev)} 个仓库）", gate(hit) if ev else "info", f"{len(hit)} of {len(per_task)} tasks",
        "" if ev else "no list given (--eval-repos)", "" if ev else "没有给清单（--eval-repos）")
    if mc_rules:
        hard = [s for s in ("no_valid_submission", "disallowed_git", "edit_test_loop", "lost_in_exploration")]
        rej = sum(1 for a in res if set(a.get("signals", [])) & set(hard))
        off["rules"] = [x for s in hard for x in off.get("rule_" + s, [])]
        add("rules", G[6], "resolved trajectories a reject rule would drop (preview; step 5 holds the filter)", "解出的轨迹里会被拒绝规则剔掉的（预估，正式过滤在步骤 5）", "info", pct(rej, len(res)),
            " · ".join(f"{s} {sig_res[s]}" for s in hard) + f" · counted only: no_verification {sig_res['no_verification']}", " · ".join(f"{s} {sig_res[s]}" for s in hard) + f" · 只计数：no_verification {sig_res['no_verification']}")
    else:
        add("rules", G[6], "reject rules (preview)", "拒绝规则（预估）", "info", "not available", "train/mc_rules.py could not be imported: " + MC_ERR, "train/mc_rules.py 导入失败")
    dup = 0
    for t, d in patches.items():
        hs = [d[n] for n in sorted(d)]
        for j, h in enumerate(hs):
            if h in hs[:j]:
                dup += 1; a0 = next(a for a in per_task[t] if a["attempt"] == sorted(d)[j]); off["dup_patch"].append([a0["i"], 0, "the same patch as an earlier resolved attempt of this task"])
    add("dup_patch", G[6], "resolved attempts whose patch equals an earlier attempt's of the same task", "补丁与同题更早一次完全相同的解出尝试", "info", pct(dup, len(res)))
    for a in res:
        if (a.get("patch_chars") or 0) > max_patch:
            off["big_patch"].append([a["i"], 0, f"patch of {a['patch_chars']:,} characters (limit {max_patch:,})"])
    add("big_patch", G[6], f"resolved attempts whose patch is larger than {max_patch:,} characters (files left in the working tree)", f"补丁超过 {max_patch:,} 字符的解出尝试（工作区里留下了文件）", "info",
        f"{len(off['big_patch'])} of {len(res)}", f"largest patch under the limit: {max([a.get('patch_chars') or 0 for a in res if (a.get('patch_chars') or 0) <= max_patch] or [0]):,}", f"上限以内最大的补丁：{max([a.get('patch_chars') or 0 for a in res if (a.get('patch_chars') or 0) <= max_patch] or [0]):,}")
    # what is left for training after the removals decided so far (a preview: the filter itself is step 5)
    why = collections.defaultdict(list)
    for cid, lab in (("dup_patch", "same patch as an earlier attempt"), ("rules", "reject rule"), ("no_submit", "no submit_patch"), ("big_patch", "patch over the limit")):
        for x in off.get(cid, []):
            if A[x[0]]["resolved"] and lab not in why[x[0]]:
                why[x[0]].append(lab)
    off["trainable"] = [[i, 0, "left out: " + " + ".join(v)] for i, v in sorted(why.items())]
    keep = [a for a in res if a["i"] not in why]
    n_by = {lab: sum(1 for v in why.values() if lab in v) for lab in ("same patch as an earlier attempt", "reject rule", "no submit_patch", "patch over the limit")}
    add("trainable", G[6], "resolved trajectories left for training (preview; the list holds the ones left out)", "预计可训练的轨迹（预估；清单里是被去掉的）", "info",
        f"{len(keep)} of {len(res)} · {len({a['task'] for a in keep})} tasks · {sum(a['tool_turns'] for a in keep):,} tool-call turns ({sum(a['thought_turns'] for a in keep):,} with a thought)",
        f"left out {len(why)}: same patch {n_by['same patch as an earlier attempt']} · reject rule {n_by['reject rule']} · no submit_patch {n_by['no submit_patch']} · patch over the limit {n_by['patch over the limit']} (one attempt may have several reasons)",
        f"去掉 {len(why)} 条：补丁重复 {n_by['same patch as an earlier attempt']} · 规则拒绝 {n_by['reject rule']} · 没有 submit_patch {n_by['no submit_patch']} · 补丁超过上限 {n_by['patch over the limit']}（一条可能有多个原因）")
    rc = C(a["repo"] or a["task"].rsplit("-", 1)[0] for a in res)
    add("repos", G[6], "repositories among the resolved trajectories", "解出的轨迹涉及的仓库", "info", f"{len(rc)} repositories · top 5 hold {pct(sum(v for _, v in rc.most_common(5)), len(res))}",
        " · ".join(f"{r} {v}" for r, v in rc.most_common(5)), " · ".join(f"{r} {v}" for r, v in rc.most_common(5)))
    add("summaries", G[6], "resolved trajectories with at least one compaction", "有过压缩的解出轨迹", "info", pct(sum(1 for a in res if a["summaries"]), len(res)), f"{k['summary']:,} summary calls in all, {k['summary_thought']:,} with a thought", f"共 {k['summary']:,} 次摘要调用，其中 {k['summary_thought']:,} 次带思考")
    add("gemma", G[6], "largest request of an attempt, Gemma tokens", "每次尝试最大的一次请求（Gemma token）", "info", f"p50 {q(gemma_max, .5):,} · p90 {q(gemma_max, .9):,} · max {max(gemma_max or [0]):,}", "all done attempts", "全部 done 的尝试")
    add("ctx", G[6], "context size of a tool-call turn in a resolved trajectory, Gemma tokens of the request", "解出轨迹里每个工具调用回合的上下文大小（请求的 Gemma token）", "info",
        f"p10 {q(ctx_req, .1):,} · p50 {q(ctx_req, .5):,} · p90 {q(ctx_req, .9):,} · max {max(ctx_req or [0]):,}  ({len(ctx_req):,} turns)",
        f"the request before a compaction: p10 {q(ctx_pre, .1):,} · p50 {q(ctx_pre, .5):,} · p90 {q(ctx_pre, .9):,} · max {max(ctx_pre or [0]):,} ({len(ctx_pre):,} compactions) — where the harness compacts in this setting; "
        f"the summary request itself: p50 {q(ctx_sum, .5):,}",
        f"压缩前那次请求：p10 {q(ctx_pre, .1):,} · p50 {q(ctx_pre, .5):,} · p90 {q(ctx_pre, .9):,} · max {max(ctx_pre or [0]):,}（{len(ctx_pre):,} 次压缩）——这个设置下 harness 在这里压缩；"
        f"摘要请求本身：p50 {q(ctx_sum, .5):,}")
    # --- cost
    usd = [a["usd"] for a in A]; secs = [a["seconds"] for a in A]
    add("usd", G[7], "cost per attempt, $ (the ledger's price estimate)", "每次尝试的花费（美元，账本按单价估算）", "info", f"mean {sum(usd) / max(1, len(usd)):.4f} · p50 {q(usd, .5):.4f} · p95 {q(usd, .95):.4f} · max {max(usd or [0]):.4f}",
        f"total {sum(usd):,.2f} · per resolved trajectory {sum(usd) / max(1, len(res)):.4f}", f"合计 {sum(usd):,.2f} · 每条解出的轨迹 {sum(usd) / max(1, len(res)):.4f}")
    add("secs", G[7], "seconds per attempt", "每次尝试的秒数", "info", f"p50 {q(secs, .5)} · p90 {q(secs, .9)} · max {max(secs or [0])}")
    for c in checks:
        for o in c["off"]:
            if c["id"] not in A[o[0]]["flags"]:
                A[o[0]]["flags"].append(c["id"])
    hist = lambda xs, edges: [[f"{lo}–{hi - 1}" if hi < 10 ** 9 else f"{lo}+", sum(lo <= x < hi for x in xs)] for lo, hi in zip(edges, edges[1:] + [10 ** 9])]
    charts = [
        dict(en="thought on a tool-call turn, by call number", zh="工具调用回合带思考的占比（按调用序号）", unit="%", bars=[[b, round(100 * v[1] / v[0], 1), f"{v[1]:,}/{v[0]:,}"] for b, v in sorted(by_depth.items())]),
        dict(en="thought on a tool-call turn, by the tool called first", zh="带思考的占比（按回合里第一个工具）", unit="%", bars=[[b, round(100 * v[1] / v[0], 1), f"{v[1]:,}/{v[0]:,}"] for b, v in sorted(by_tool.items(), key=lambda x: -x[1][0])]),
        dict(en="tasks by number of resolved attempts", zh="任务按解出次数", unit="", bars=[[f"{n}×", v, ""] for n, v in sorted(dist.items())]),
        dict(en="tool calls per resolved trajectory", zh="每条解出轨迹的工具调用数", unit="", bars=[[a_, b_, ""] for a_, b_ in hist([a["tool_calls"] for a in res], [0, 6, 11, 16, 21, 26, 31, 36])]),
        dict(en="context size of a tool-call turn, resolved trajectories (Gemma tokens)", zh="工具调用回合的上下文大小（解出的轨迹，Gemma token）", unit="",
             bars=[[a_, b_, ""] for a_, b_ in hist(ctx_req, [0, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 24000])]),
        dict(en="context size right before a compaction (Gemma tokens)", zh="压缩前那次请求的上下文大小（Gemma token）", unit="",
             bars=[[a_, b_, ""] for a_, b_ in hist(ctx_pre, [0, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 24000])]),
        dict(en="tool calls made", zh="各工具的调用数", unit="", bars=[[n, v, ""] for n, v in tools.most_common()]),
        dict(en="cost per attempt, $", zh="每次尝试的花费（美元）", unit="", bars=[[a_, b_, ""] for a_, b_ in hist([int(100 * u) for u in usd], [0, 5, 10, 20, 30, 50, 100])], label_div=100),
    ]
    meta = dict(out=os.path.abspath(out), ids=None if ids is None else len(ids), name=name, attempts_wanted=attempts, min_thought=min_thought, max_patch=max_patch, rows_in_file=n_all,
                rows=len(rows), done=len(done), tasks=len(per_task), resolved=len(res), calls=dict(summary=k["summary"], tool=k["tool"], text=k["text"], closing=k["closing"]),
                all_calls_thought_note="", made=time.strftime("%Y-%m-%d %H:%M"), echo_pats=[[n, p] for n, p in ECHO_PATS], eval_repos=sorted(ev),
                mc_rules=bool(mc_rules), groups=list(G))
    return dict(meta=meta, checks=checks, charts=charts, attempts=A, bodies=B if bodies else None)


def print_table(R, echo_all=False):
    m = R["meta"]; C = collections.Counter(c["status"] for c in R["checks"])
    print(f"   attempts: {m['done']} done of {m['rows']} rows read ({m['rows_in_file']} in the file) · tasks {m['tasks']} · resolved {m['resolved']} · model calls by kind {m['calls']}")
    grp = None
    for c in R["checks"]:
        if c["group"] != grp:
            grp = c["group"]; print(f"   -- {grp}")
        print(f"   {c['status']:4} · {c['en']}: {c['value']}" + (f"  ({c['note']})" if c["note"] else ""))
    print(f"   CHECKS: {C.get('FAIL', 0)} FAIL · {C.get('note', 0)} note · {C.get('pass', 0)} pass · {C.get('info', 0)} info"
          + ("  ← gates not met: " + ", ".join(c["id"] for c in R["checks"] if c["status"] == "FAIL") if C.get("FAIL") else ""))
    A = R["attempts"]
    for c in R["checks"]:
        if c["id"].startswith("echo_") and c["off"]:
            print(f"   {c['en']}:")
            for i, n, note in (c["off"] if echo_all else c["off"][:8]):
                print(f"      {A[i]['task'][:40]} a{A[i]['attempt']} call {n} · {'resolved' if A[i]['resolved'] else 'unresolved'} · {note[:230]}")
            if len(c["off"]) > 8 and not echo_all:
                print(f"      … {len(c['off']) - 8} more (--echo-all)")


def _js(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_page(R, page, chunk_mb=5.0):
    os.makedirs(os.path.join(page, "traj"), exist_ok=True)
    for f in os.listdir(os.path.join(page, "traj")):
        if re.fullmatch(r"\d+\.js", f):
            os.remove(os.path.join(page, "traj", f))
    cur = {}; size = 0; n = 0; total = 0
    def flush():
        nonlocal cur, size, n
        if cur:
            open(os.path.join(page, "traj", f"{n:03d}.js"), "w").write(f"window.TRAJ_LOADED({n},{_js(cur)});\n"); n += 1; cur = {}; size = 0
    for a, b in zip(R["attempts"], R["bodies"]):
        if b is None:
            a["chunk"] = -1; continue
        s = _js(b); a["chunk"] = n; cur[str(a["i"])] = b; size += len(s); total += len(s)
        if size > chunk_mb * 1e6:
            flush()
    flush()
    head = {k: v for k, v in R.items() if k != "bodies"}
    open(os.path.join(page, "audit.js"), "w").write("window.AUDIT=" + _js(head) + ";\n")
    tpl = open(os.path.join(HERE, "teach_audit_page.html")).read()
    open(os.path.join(page, "index.html"), "w").write(tpl)
    return n, total


def _fixture(d, break_=""):
    """a directory of 2 tasks × 2 attempts as the driver writes it, clean — or with one planted defect"""
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    for sub in ("ledger", "calls", "runs/r_a1/traj", "runs/r_a1/patches", "runs/r_a2/traj", "runs/r_a2/patches"):
        os.makedirs(os.path.join(d, sub))
    schemas = [{"type": "function", "function": {"name": n, "parameters": {"properties": {k: {} for k in props}, "required": req}}}
               for n, props, req in (("run_command", ["command"], ["command"]), ("edit_file", ["filepath", "old_string", "new_string"], ["filepath", "old_string", "new_string"]), ("submit_patch", [], []))]
    rem = teach_reminder.text("1")
    att = open(os.path.join(d, "attempts.jsonl"), "w"); led = open(os.path.join(d, "ledger", "calls.jsonl"), "w")
    json.dump({"o__a-1": True, "o__b-2": True}, open(os.path.join(d, "gold.json"), "w"))
    open(os.path.join(d, "tasks.jsonl"), "w").write("".join(json.dumps({"instance_id": t, "repo": ("encode/httpx" if break_ == "eval_repo" and t == "o__b-2" else "o/" + t[3])}) + "\n" for t in ("o__a-1", "o__b-2")))
    for task in ("o__a-1", "o__b-2"):
        for n in (1, 2):
            mark = (task, n) == ("o__b-2", 2)          # the attempt every defect is planted in
            B = break_ if mark else ""
            plan = [("run_command", {"command": "timeout 60 python -m pytest -q"}, "Let me run the tests first."),
                    ("edit_file", {"filepath": "a.py", "old_string": "x", "new_string": "</script><img src=x onerror=1>"}, "The fix is one line."),
                    ("run_command", {"command": "git diff"}, "Check the diff before the submit."),
                    ("submit_patch", {}, "The diff is clean, submit.")]
            if B == "foreign":
                plan[2] = ("bash", {"command": "ls"}, "List the files.")
            if B == "echo":
                plan[2] = (plan[2][0], plan[2][1], "The reminder says reason before every tool call. Check the diff.")
            if B == "no_thought":
                plan = [(a_, b_, "") for a_, b_, _ in plan]
            lines = []; prev = None
            for i, (name, args, th) in enumerate(plan + [(None, None, "")]):
                new = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "the task"}] if i == 0 else []
                if prev:
                    am = {"role": "assistant", "content": "", "tool_calls": [{"id": prev[0], "function": {"name": prev[1], "arguments": "{}"}}]}
                    if prev[2] and not (B == "back" and i == 2):
                        am["reasoning_content"] = prev[2]
                    new += [am, {"role": "tool", "tool_call_id": prev[0], "content": json.dumps({"status": "ok", "stdout": (rem if B == "leak" and i == 2 else "ok")})}]
                resp = {"reasoning": th, "content": "Done." if name is None else "", "finish_reason": "stop" if name is None else "tool_calls",
                        "tool_calls": [] if name is None else [{"name": name, "args": json.dumps(args), "id": f"c{i}"}]}
                usage = {"prompt": 900 + 100 * i, "completion": 50, "cached": 0, "reasoning": 20 if th else 0}
                if B == "reasoning" and i == 1:
                    resp["reasoning"] = ""; usage["reasoning"] = 300
                line = {"task": task, "call": i + 1, "t": 1.0 + i, "s": 1.5, "n_msgs": 2 * i + 2, "reset": None, "new": new, "resp": resp, "usage": usage}
                if i == 0:
                    line["setup"] = {"tool_schemas": schemas}
                lines.append(line); prev = (f"c{i}", name, resp["reasoning"]) if name else None
                if not (B == "ledger" and i == 3):
                    led.write(json.dumps({"key": f"{task}#{n}", "in": 1, "out": 1, "reasoning": 0, "usd": 0.01, "err": None}) + "\n")
            if B != "files":
                open(os.path.join(d, f"runs/r_a{n}/traj/{task}.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in lines))
            open(os.path.join(d, f"runs/r_a{n}/patches/{task}.patch"), "w").write(f"diff --git a/a.py b/a.py\n+{task}{n}\n")
            open(os.path.join(d, f"calls/r_a{n}.jsonl"), "a").write("".join(json.dumps({"reminder": i < 4, "err": None}) + "\n" for i in range(5)))
            att.write(json.dumps({"task": task, "attempt": n, "state": "done", "resolved": task == "o__b-2", "usd": 0.05, "seconds": 9, "tool_calls": 4, "patch_chars": (30000 if break_ == "big_patch" and (task, n) == ("o__b-2", 2) else 30), "harness_error": None,
                                  "run": f"runs/x/runs/r_a{n}", "reminder": "on", "reminder_id": "1", "reminder_place": "system" if B == "settings" else "end", "hist": "on"}) + "\n")
    att.close(); led.close()


def selftest():
    """a clean directory passes every gate; each planted defect turns its own check, and the offender is the planted one"""
    import tempfile, shutil
    d = tempfile.mkdtemp(prefix="teach_audit_")
    try:
        ev = ["encode/httpx"]
        _fixture(d); R = audit(d, attempts=2, eval_repos=ev, bodies=True); st = {c["id"]: c for c in R["checks"]}
        bad = [c["id"] for c in R["checks"] if c["status"] in ("FAIL", "note")]
        assert not bad, bad
        assert st["no_thought"]["value"].startswith("16 / 16") and st["back"]["value"].startswith("16 / 16") and st["placement"]["value"] == "4 vs 4", (st["no_thought"]["value"], st["back"]["value"], st["placement"]["value"])
        assert R["meta"]["calls"] == dict(summary=0, tool=16, text=0, closing=4) and st["rules"]["value"].startswith("0 / 2"), (R["meta"]["calls"], st["rules"])
        assert int(st["trainable"]["value"].split()[0]) + len(st["trainable"]["off"]) == R["meta"]["resolved"] == 2, st["trainable"]
        n, _ = write_page(R, os.path.join(d, "page")); js = open(os.path.join(d, "page", "traj", "000.js")).read()
        assert n == 1 and "</script>" not in js and "<\\/script>" in js and os.path.exists(os.path.join(d, "page", "index.html")), n
        want = {"leak": ("leak", "FAIL", 0), "back": ("back", "FAIL", 3), "foreign": ("foreign", "note", 3), "files": ("files", "FAIL", 0), "ledger": ("ledger", "note", 0),
                "reasoning": ("reasoning", "FAIL", 2), "settings": ("settings", "FAIL", None), "eval_repo": ("eval_repo", "FAIL", None), "no_thought": ("no_thought", "FAIL", 1), "echo": ("echo_tool", "note", 3), "big_patch": ("big_patch", "info", 0)}
        for brk, (cid, status, call) in want.items():
            _fixture(d, brk); R = audit(d, attempts=2, eval_repos=ev); c = next(x for x in R["checks"] if x["id"] == cid)
            assert c["status"] == status, (brk, c["status"], c["value"])
            if call is not None:
                o = c["off"][0]; a = R["attempts"][o[0]]
                assert (a["task"], a["attempt"], o[1]) == ("o__b-2", 2, call) and all(R["attempts"][x[0]]["task"] == "o__b-2" for x in c["off"]), (brk, a["task"], a["attempt"], o)
            others = [x["id"] for x in R["checks"] if x["status"] == "FAIL" and x["id"] != cid]
            assert not others or brk in ("files", "no_thought", "reasoning"), (brk, others)
        print(f"teach_audit selftest: a clean directory passes every gate (16 / 16 thoughts, 16 / 16 re-sent, placement 4 vs 4); {len(want)} planted defects "
              f"({', '.join(want)}) each turn their own check and name the planted attempt and call; the page files are written and '</script>' in the text is escaped → all as expected")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the generation directory (attempts.jsonl, runs/, ledger/, calls/)")
    ap.add_argument("--ids", default="", help="a file of task ids: only these tasks")
    ap.add_argument("--name", default="", help="only the runs whose name starts with this")
    ap.add_argument("--attempts", type=int, default=3, help="attempts wanted per task")
    ap.add_argument("--min-thought", type=float, default=90.0, help="gate: share of tool-call turns with a thought, %%")
    ap.add_argument("--max-patch", type=int, default=20000, help="a resolved attempt with a larger patch (characters) is left out of the trainable count")
    ap.add_argument("--eval-repos", default=os.path.join(os.path.dirname(HERE), "splits", "eval_repos.txt"))
    ap.add_argument("--page", default="", help="write the local page into this directory")
    ap.add_argument("--echo-all", action="store_true")
    a = ap.parse_args()
    ids = set(open(a.ids).read().split()) if a.ids else None
    ev = [l.strip() for l in open(a.eval_repos) if l.strip() and not l.startswith("#")] if a.eval_repos and os.path.exists(a.eval_repos) else []
    print(f"== teach_audit · out {a.out} · ids {a.ids or 'all tasks'}{'' if ids is None else f' ({len(ids)})'} · name {a.name or 'all runs'} · attempts wanted {a.attempts} · "
          f"min thought {a.min_thought:g} % · max patch {a.max_patch:,} chars · eval repos {a.eval_repos if ev else 'none'} ({len(ev)}) · page {a.page or 'not written'} · rule preview {'on (train/mc_rules.py)' if mc_rules else 'OFF: ' + MC_ERR}")
    R = audit(a.out, ids, a.name, a.attempts, a.min_thought, ev, bodies=bool(a.page), max_patch=a.max_patch)
    print_table(R, a.echo_all)
    if a.page:
        n, total = write_page(R, a.page)
        print(f"   page: {os.path.abspath(os.path.join(a.page, 'index.html'))} · {n} trajectory files, {total / 1e6:.1f} MB of text · opens from the disk, no server")


if __name__ == "__main__":
    main()
