#!/usr/bin/env python3
"""B-9 part 2: MetaCurate's public DSv4Pro trajectories (mini-swe-agent) → call-level records for train/render.py.

    .venv-eval/bin/python train/mc_prep.py download --dir runs/sft/mc/raw
    .venv-eval/bin/python train/mc_prep.py build --raw runs/sft/mc/raw --tokenizer <-ct snapshot dir> \
        --history keep|drop --out runs/sft/mc/r1k [--n 1000 --seed 0 --max-ctx 30720]
    .venv-eval/bin/python train/mc_prep.py verify --traj runs/verified/vf_probe --probe runs/verified/probe/records \
        --history keep|drop                    # our reconstruction of the requests = what mini actually sent?

Data: huggingface.co/datasets/furiosa-ai/MetaCurate-DeepSeekV4Pro @ REVISION (4 parquet shards, 4,286 trajectories,
1,760 resolved; CC BY 4.0 compilation, upstream licences per row). The paper's comparable arm is "Resolved" (resolved
only, 1,000 per arm); the only filter added here is the fit: peak context ≤ --max-ctx Gemma tokens (the paper trained at
128K; B-9 and Kaggle run at 32K). Printed only, nothing removed: rows from Verified repos, thoughts over 8,192 tokens,
licences, rows per task, multi-tool-call turns, text + tool-call turns, harness (user) messages after the first.

History rule (--history, decided by B-9 part 2 step 0): `keep` = earlier assistant turns are sent with their
`reasoning_content` (then the Gemma template renders them and one trajectory packs into one sequence); `drop` = they are
sent without it. A record = {instance_id (the row id), task_id, repo, dataset, license, source, tools, history, calls:
[{call, segment, input_messages, target: {reasoning, content, tool_calls: [{name, args, id}]}}]}; segment = the number of
user messages after the first one so far. The terminal `exit` event is never part of a sequence.
Outputs quote task text → runs/ only (gitignored). Every knob prints its value.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

REPO_ID = "furiosa-ai/MetaCurate-DeepSeekV4Pro"
REVISION = "6bf4bb1761bd9748b010a7a7750c15b5f1c9930b"          # 2026-11-09, read 2026-11-13
SHARDS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
VERIFIED_REPOS = {"django/django", "sympy/sympy", "sphinx-doc/sphinx", "matplotlib/matplotlib", "scikit-learn/scikit-learn",
                  "astropy/astropy", "pydata/xarray", "pytest-dev/pytest", "pylint-dev/pylint", "psf/requests",
                  "mwaskom/seaborn", "pallets/flask"}



def wire(m, history):
    """one stored mini message → the fields the chat request carries (role, content, tool calls, ids, reasoning)"""
    role = m.get("role")
    if role == "assistant":
        out = {"role": "assistant", "content": m.get("content") or ""}
        tcs = [{"id": tc.get("id"), "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
               for tc in (m.get("tool_calls") or [])]
        if tcs:
            out["tool_calls"] = tcs
        if history == "keep" and (m.get("reasoning_content") or "").strip():
            out["reasoning_content"] = m["reasoning_content"]
        return out
    if role == "tool":
        return {"role": "tool", "content": m.get("content") or "", "tool_call_id": m.get("tool_call_id")}
    return {"role": role, "content": m.get("content") or ""}


def target_of(m):
    return {"reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
            "tool_calls": [{"name": tc["function"]["name"], "args": tc["function"]["arguments"], "id": tc.get("id")}
                           for tc in (m.get("tool_calls") or [])]}


def text_to_thought(m):
    """--text-to-thought: a reply with text AND tool calls → the text appended to the thought, no text. The Gemma
    template re-orders such a reply in later prompts, which breaks packing (render.py --split-runs splits there)."""
    if m.get("role") == "assistant" and m.get("tool_calls") and (m.get("content") or "").strip():
        m = dict(m)
        m["reasoning_content"] = ((m.get("reasoning_content") or "").rstrip() + "\n\n" + m["content"].strip()).strip()
        m["content"] = ""
    return m


def to_calls(messages, history, t2t=False):
    msgs = [m for m in messages if m.get("role") in ("system", "user", "assistant", "tool")]   # drops `exit`
    if t2t:
        msgs = [text_to_thought(m) for m in msgs]
    calls, users = [], 0
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            users += 1
        if m["role"] != "assistant":
            continue
        calls.append({"call": len(calls) + 1, "segment": max(0, users - 1),
                      "input_messages": [wire(x, history) for x in msgs[:i]], "target": target_of(m)})
    return calls


def load_rows(raw):
    import pyarrow.parquet as pq
    rows, have = [], [s for s in SHARDS if os.path.exists(os.path.join(raw, s))]
    print(f"  shards present {len(have)} / {len(SHARDS)}" + ("" if len(have) == len(SHARDS) else "  ← NOT the full dataset"))
    for s in have:
        rows += pq.read_table(os.path.join(raw, s)).to_pylist()
    return rows


def cmd_download(a):
    from huggingface_hub import hf_hub_download
    print(f"download {REPO_ID} @ {REVISION} → {a.dir} (HF token from the environment / cache)")
    for f in SHARDS + ["tools.json", "README.md"]:
        p = hf_hub_download(REPO_ID, f, repo_type="dataset", revision=REVISION, local_dir=a.dir)
        print(f"  {f}: {os.path.getsize(p):,} bytes")


def cmd_build(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    import render
    render.WIRE["mode"] = a.wire
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    R = render.Renderer(tok)
    tools = json.load(open(os.path.join(a.raw, "tools.json")))
    print(f"mc_prep build · {REPO_ID}@{REVISION[:12]} · holdout tasks {a.holdout_tasks} · history {a.history} · wire {a.wire} · think budget {a.think_budget} · reply cap {a.reply_cap} · text to thought {a.text_to_thought} · max ctx {a.max_ctx} · n {a.n} · seed {a.seed}"
          f" · tokenizer {a.tokenizer} · limit {a.limit or 'all'} · out {a.out}")
    rows = load_rows(a.raw)
    if a.limit:
        rows = rows[:a.limit]
    n0 = len(rows)
    res = [r for r in rows if str(r.get("resolved")) == "True"]
    fit, peaks, stats = [], [], collections.Counter()
    over = collections.Counter()
    for r in res:
        calls = to_calls(r["messages"], a.history, a.text_to_thought)
        if not calls:
            stats["no model turn"] += 1
            continue
        last = calls[-1]
        # peak context = the last call's prompt + its reply (the history only grows, so the last call is the largest)
        p = R.prompt(last["input_messages"], tools)
        t = last["target"]
        reply = (t["reasoning"] + t["content"] + "".join(tc["args"] for tc in t["tool_calls"]))
        peak = len(R.ids(p)) + len(R.ids(reply)) + 16 * (1 + len(t["tool_calls"]))

        peaks.append(peak)
        if peak <= a.max_ctx:
            fit.append((r, calls, peak))
    random.seed(a.seed)
    pick = fit if len(fit) <= a.n else random.sample(fit, a.n)
    pick.sort(key=lambda x: x[0]["id"])
    # held-out TASKS (user 2026-11-13): tasks with exactly one picked trajectory, so nothing of the task is trained on;
    # their records go to <out>_holdout/ (render it separately, train with sft_lora --eval)
    n_traj = collections.Counter(r["instance_id"] for r, _, _ in pick)
    single = sorted(t for t, c in n_traj.items() if c == 1)
    hold_tasks = set(random.Random(a.seed + 1).sample(single, min(a.holdout_tasks, len(single))))
    hold = [x for x in pick if x[0]["instance_id"] in hold_tasks]
    pick = [x for x in pick if x[0]["instance_id"] not in hold_tasks]
    hold_dir = a.out.rstrip("/") + "_holdout"
    os.makedirs(hold_dir, exist_ok=True)
    for r, calls, peak in hold:
        json.dump({"instance_id": r["id"], "task_id": r["instance_id"], "repo": r["repo"], "dataset": r["dataset"],
                   "license": r["license"], "source": "metacurate_dsv4pro", "revision": REVISION, "history": a.history,
                   "text_to_thought": a.text_to_thought, "peak_ctx": peak, "tools": tools, "calls": calls, "holdout": True},
                  open(os.path.join(hold_dir, f"{r['id']}.json"), "w"))
    print(f"  held-out tasks: {len(hold_tasks)} of the {len(single)} tasks with one trajectory (seed {a.seed + 1}) → "
          f"{len(hold)} records in {hold_dir} (no other trajectory of these tasks is trained): {sorted(hold_tasks)}")
    os.makedirs(a.out, exist_ok=True)
    info = collections.Counter()
    per_task = collections.Counter(r["instance_id"] for r, _, _ in pick)
    for r, calls, peak in pick:
        # turns the Kaggle-like sampling could not have produced (printed only; the user decides on removal)
        th = [len(R.ids(c["target"]["reasoning"])) for c in calls]
        rp = [t + len(R.ids(c["target"]["content"] + "".join(tc["args"] for tc in c["target"]["tool_calls"])))
              + 16 * len(c["target"]["tool_calls"]) for t, c in zip(th, calls)]
        n_th, n_rp = sum(x > a.think_budget for x in th), sum(x > a.reply_cap for x in rp)
        over[f"turns with a thought > {a.think_budget}"] += n_th; over[f"trajectories with such a turn"] += n_th > 0
        over[f"turns with a reply > {a.reply_cap}"] += n_rp; over[f"trajectories with such a reply"] += n_rp > 0
        asst = [m for m in r["messages"] if m.get("role") == "assistant"]
        info["calls"] += len(calls)
        info["multi-tool-call turns"] += sum(1 for m in asst if len(m.get("tool_calls") or []) > 1)
        info["text + tool-call turns"] += sum(1 for m in asst if (m.get("content") or "").strip() and m.get("tool_calls"))
        info["turns without a thought"] += sum(1 for m in asst if not (m.get("reasoning_content") or "").strip())
        info["harness user messages"] += sum(1 for m in r["messages"] if m.get("role") == "user") - 1
        info["rows from Verified repos"] += r["repo"] in VERIFIED_REPOS
        rec = {"instance_id": r["id"], "task_id": r["instance_id"], "repo": r["repo"], "dataset": r["dataset"],
               "license": r["license"], "source": "metacurate_dsv4pro", "revision": REVISION, "history": a.history,
               "text_to_thought": a.text_to_thought, "peak_ctx": peak, "tools": tools, "calls": calls}
        json.dump(rec, open(os.path.join(a.out, f"{r['id']}.json"), "w"))
    sp = sorted(peaks)
    q = (lambda v, x: v[min(len(v) - 1, int(x * len(v)))]) if sp else (lambda v, x: None)
    summ = {"rows": n0, "resolved": len(res), "fit": len(fit), "picked": len(pick), "tasks": len(per_task),
            "peak p50": q(sp, .5), "peak p90": q(sp, .9), **dict(over),
            "licences": dict(collections.Counter(r["license"] for r, _, _ in pick)),
            "rows per task": dict(collections.Counter(per_task.values())), **dict(info), **dict(stats)}
    json.dump(summ, open(os.path.join(a.out, "_prep_summary.json"), "w"), indent=1)
    print(f"  rows {n0} → resolved {len(res)} → peak ≤ {a.max_ctx} {len(fit)} → picked {len(pick)} ({len(per_task)} tasks)"
          f" · peak tokens p50 {q(sp, .5)} p90 {q(sp, .9)} (resolved rows)")
    print("  printed only (nothing removed): " + " · ".join(f"{k} {v}" for k, v in summ.items()
                                                          if k not in ("rows", "resolved", "fit", "picked", "tasks", "peak p50", "peak p90")))
    print(f"== mc_prep: {len(pick)} records in {a.out}")


def norm_req(m):
    """compare what matters to the template: role, content, tool calls (name, parsed args, id), tool_call_id, reasoning"""
    tcs = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            pass
        tcs.append((fn.get("name"), json.dumps(args, sort_keys=True), tc.get("id")))
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return (m.get("role"), (c or "").strip(), tuple(tcs), m.get("tool_call_id"),
            (m.get("reasoning_content") or m.get("reasoning") or "").strip())


def cmd_verify(a):
    import hashlib
    print(f"mc_prep verify · traj {a.traj} · probe records {a.probe} · history {a.history}")
    trajs = {}
    for f in glob.glob(os.path.join(a.traj, "**", "*.traj.json"), recursive=True):
        msgs = json.load(open(f)).get("messages") or []
        first_user = next((m.get("content") for m in msgs if m.get("role") == "user"), "") or ""
        trajs[hashlib.sha256(first_user.encode()).hexdigest()[:16]] = msgs
    n = ok = 0
    diffs = []
    for f in sorted(glob.glob(os.path.join(a.probe, "*.json"))):
        rec = json.load(open(f))
        msgs = trajs.get(rec.get("first_user_sha"))
        if msgs is None:
            diffs.append(f"{os.path.basename(f)}: no traj with the same first user message")
            continue
        ours = {c["call"]: c["input_messages"] for c in to_calls(msgs, a.history)}
        for c in rec["calls"]:
            n += 1
            want = [norm_req(m) for m in c["input_messages"]]
            got = [norm_req(m) for m in ours.get(c["call"], [])]
            if want == got:
                ok += 1
            elif len(diffs) < 8:
                k = next((i for i, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
                diffs.append(f"{rec['instance_id']} call {c['call']}: {len(got)} vs {len(want)} messages, first diff at "
                             f"{k}: ours={str(got[k] if k < len(got) else None)[:160]} sent={str(want[k] if k < len(want) else None)[:160]}")
    for d in diffs:
        print("  DIFF", d)
    print(f"== verify history={a.history}: {ok} / {n} requests reconstructed exactly from the traj files")
    return 0 if n and ok == n else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--dir", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True); b.add_argument("--tokenizer", required=True)
    b.add_argument("--history", choices=["keep", "drop"], required=True); b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=1000); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--holdout-tasks", type=int, default=8, help="whole tasks kept out of training (one-trajectory tasks)")
    b.add_argument("--wire", choices=["mini", "mini-think"], default="mini", help="how the server renders (render.py --wire)")
    b.add_argument("--think-budget", type=int, default=4096, help="count (not drop) turns whose thought is longer")
    b.add_argument("--reply-cap", type=int, default=8192, help="count (not drop) turns whose whole reply is longer")
    b.add_argument("--text-to-thought", action="store_true", help="move the text of text + tool-call replies into the thought")
    b.add_argument("--max-ctx", type=int, default=30720); b.add_argument("--limit", type=int, default=0, help="debug: first rows only")
    v = sub.add_parser("verify")
    v.add_argument("--traj", required=True); v.add_argument("--probe", required=True)
    v.add_argument("--history", choices=["keep", "drop"], required=True)
    a = ap.parse_args()
    return {"download": cmd_download, "build": cmd_build, "verify": cmd_verify}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
