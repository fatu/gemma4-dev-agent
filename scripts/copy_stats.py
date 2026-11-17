"""Does the model copy its previous turn? Counts over the trajectories of scripts/verified_box.sh runs (B-9 part 2 / 3).
  python3 scripts/copy_stats.py runs/verified/vf_q9_th runs/verified/vf_q9_mc [--ids runs/verified/paired_….json]
Reads every <id>.traj.json under each run dir (mini-swe-agent: messages with role assistant, `reasoning_content`,
`tool_calls`). A turn = one assistant message; its command = the arguments of its tool calls. Prints per run, for 📷:
  exact copies = turn pairs (k-1, k) with the same thought AND the same command (share of all pairs; trajectories with
  one; the call of the first copy p10 / p50) · same command back-to-back, any thought (pairs; longest run per trajectory
  p50 / p90 / max) · thought loops (a sentence ≥ 5 times in one thought, or a thought over 12,000 characters) ·
  turns without a tool call · thought characters p50 · exits. No task ids or text are printed.
  n-gram loops (B-9 part 3, 2026-11-17) = the rule of arXiv 2512.12895: a thought holds some 30-gram at least 20 times
  (also printed at 5 times), plus thought tokens per turn and per trajectory. The paper does not name its unit; tokens
  are assumed. Unit here: the model's tokenizer when `tokenizers` can be imported and a tokenizer.json is found
  (--tokenizer <file or dir>, else the HF cache of --model / $MODEL, default Qwen/Qwen3.5-9B) — run it with
  .venv-eval/bin/python for that; otherwise a regex split (words and single symbols), and the line says which."""
import argparse
import collections
import glob
import json
import os
import re


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def loops(th):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", th) if len(x.strip()) > 20]
    return len(th) > 12000 or (bool(s) and collections.Counter(s).most_common(1)[0][1] >= 5)


def prose_loops(th):
    """the same rule without code: fenced blocks dropped, sentences of 30+ characters that do not look like a code line.
    (The first rule also counts repeated code lines such as `return True`; B-9 part 3, 2026-11-16.)"""
    if len(th) > 12000:
        return True
    t = re.sub(r"```.*?```", "", th, flags=re.S)
    s = [x.strip() for x in re.split(r"(?<=[.!?\n])\s+", t) if len(x.strip()) >= 30]
    s = [x for x in s if not x.startswith(("-", "`", "return", "if ", "elif", "else"))]
    return bool(s) and collections.Counter(s).most_common(1)[0][1] >= 5


def load_unit(path, model):
    """→ (function text → list of units, description printed in every block)"""
    rx = re.compile(r"\w+|[^\w\s]")
    fallback = lambda why: (rx.findall, f"regex words/symbols ({why}; token units need .venv-eval/bin/python)")
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        return fallback(f"no `tokenizers` module: {type(e).__name__}")
    cands = []
    if path:
        cands = [path] if os.path.isfile(path) else [os.path.join(path, "tokenizer.json")]
    else:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
        cands = sorted(glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*", "tokenizer.json")))
    for c in cands:
        if os.path.isfile(c):
            tk = Tokenizer.from_file(c)
            return (lambda t: tk.encode(t, add_special_tokens=False).ids), f"tokens of {c.split('models--')[-1].split('/snapshots')[0] if 'models--' in c else c}"
    return fallback(f"no tokenizer.json for {path or model}")


def ngram_max(u, n):
    if len(u) < n:
        return 0
    return max(collections.Counter(tuple(u[i:i + n]) for i in range(len(u) - n + 1)).values())


def stats(d, only=None, unit=None):
    C, exits, first, runs, thl = collections.Counter(), collections.Counter(), [], [], []
    ntok, ntraj = [], []
    for f in sorted(glob.glob(os.path.join(d, "**", "*.traj.json"), recursive=True)):
        iid = os.path.basename(f)[:-len(".traj.json")]
        if only is not None and iid not in only:
            continue
        t = json.load(open(f))
        exits[(t.get("info") or {}).get("exit_status") or "?"] += 1
        asst = [m for m in t.get("messages") or [] if m.get("role") == "assistant"]
        turns = [((m.get("reasoning_content") or "").strip(),
                  json.dumps([(tc.get("function") or {}).get("arguments") for tc in m.get("tool_calls") or []]),
                  bool(m.get("tool_calls")))
                 for m in asst]
        C["turns with text beside a tool call"] += sum(1 for m in asst if m.get("tool_calls") and isinstance(m.get("content"), str) and m["content"].strip())
        C["turns with several tool calls"] += sum(1 for m in asst if len(m.get("tool_calls") or []) > 1)
        C["trajectories"] += 1; C["turns"] += len(turns)
        fc, run, best, any_loop, any_prose = None, 1, 1, False, False
        ng20, ng5, tt = False, False, 0
        for k, (th, cmd, has) in enumerate(turns):
            thl.append(len(th)); C["turns without a tool call"] += not has
            if unit:
                u = unit[0](th) if th else []
                ntok.append(len(u)); tt += len(u)
                m = ngram_max(u, 30) if len(u) >= 34 else 0
                C["ngram20 turns"] += m >= 20; C["ngram5 turns"] += m >= 5; ng20 |= m >= 20; ng5 |= m >= 5
            if loops(th):
                C["turns with a thought loop"] += 1; any_loop = True
            if prose_loops(th):
                C["turns with a prose loop"] += 1; any_prose = True
            C["thoughts over 3000 chars"] += len(th) > 3000; C["thoughts over 12000 chars"] += len(th) > 12000
            if k == 0:
                continue
            C["pairs"] += 1
            pth, pcmd, phas = turns[k - 1]
            same_cmd = has and phas and cmd == pcmd
            C["same command pairs"] += same_cmd
            run = run + 1 if same_cmd else 1; best = max(best, run)
            if same_cmd and th == pth:
                C["exact copy pairs"] += 1
                if fc is None:
                    fc = k + 1
        runs.append(best); C["trajectories with a thought loop"] += any_loop; C["trajectories with a prose loop"] += any_prose
        C["ngram20 traj"] += ng20; C["ngram5 traj"] += ng5; ntraj.append(tt)
        if fc is not None:
            first.append(fc); C["trajectories with an exact copy"] += 1
    n, p = C["trajectories"], max(C["pairs"], 1)
    print(f"== copy stats {os.path.basename(os.path.normpath(d))}: {n} trajectories · {C['turns']} turns · {C['pairs']} turn pairs")
    print(f"   exact copies of the previous turn (thought + command): {C['exact copy pairs']} pairs = {C['exact copy pairs'] / p:.1%}"
          f" · in {C['trajectories with an exact copy']} / {n} trajectories · first copy at call p10 {q(first, .1)} p50 {q(first, .5)}")
    print(f"   same command back-to-back: {C['same command pairs']} pairs = {C['same command pairs'] / p:.1%}"
          f" · longest run per trajectory p50 {q(runs, .5)} p90 {q(runs, .9)} max {max(runs) if runs else None}")
    print(f"   thought loops: {C['turns with a thought loop']} turns in {C['trajectories with a thought loop']} / {n} trajectories"
          f" · turns without a tool call {C['turns without a tool call']} · thought chars p50 {q(thl, .5)} p90 {q(thl, .9)}")
    T = max(C["turns"], 1)
    print(f"   prose loops (code left out): {C['turns with a prose loop']} turns in {C['trajectories with a prose loop']} / {n} trajectories"
          f" · thoughts over 3,000 chars {C['thoughts over 3000 chars']} = {C['thoughts over 3000 chars'] / T:.1%} · over 12,000 chars {C['thoughts over 12000 chars']}")
    print(f"   turns with text beside a tool call {C['turns with text beside a tool call']} = {C['turns with text beside a tool call'] / T:.1%}"
          f" · turns with several tool calls {C['turns with several tool calls']} = {C['turns with several tool calls'] / T:.1%}")
    if unit:
        print(f"   n-gram loops (a 30-gram ≥ 20 times in one thought; unit: {unit[1]}): {C['ngram20 turns']} turns in {C['ngram20 traj']} / {n} trajectories"
              f" · at ≥ 5 times: {C['ngram5 turns']} turns in {C['ngram5 traj']} / {n}"
              f" · thought units per turn mean {sum(ntok) / T:.0f} p50 {q(ntok, .5)} p99 {q(ntok, .99)} · per trajectory mean {sum(ntraj) / max(n, 1):,.0f} p50 {q(ntraj, .5)}"
              f" · turns with ≥ 4,000 units {sum(1 for x in ntok if x >= 4000)}")
    print(f"   exits {dict(exits.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (runs/verified/<TAG>)")
    ap.add_argument("--ids", default=None, help="paired.py's JSON: only the tasks in both runs")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json or its dir for the n-gram line (default: HF cache of --model)")
    ap.add_argument("--model", default=os.environ.get("MODEL") or "Qwen/Qwen3.5-9B", help="model whose cached tokenizer counts the units")
    a = ap.parse_args()
    unit = load_unit(a.tokenizer, a.model)
    print(f"copy_stats · n-gram unit: {unit[1]}")
    only = None
    if a.ids:
        j = json.load(open(a.ids))
        only = set(j["both"]) | set(j["only_a"]) | set(j["only_b"]) | set(j["neither"])
        print(f"copy_stats · only the {len(only)} tasks of {a.ids}")
    else:
        print("copy_stats · all trajectories under each run dir (--ids = only the paired tasks)")
    for d in a.runs:
        stats(d, only, unit)
    print("📷 the == blocks")


if __name__ == "__main__":
    main()
