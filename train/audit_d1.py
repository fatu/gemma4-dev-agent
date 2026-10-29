#!/usr/bin/env python3
"""Offline audit of D1 = adesolanke/gemma4-swe-agent-trajectories conv_curated_v2 (plan 13, Phase 1 D1 checks).

    .venv-eval/bin/python train/audit_d1.py --data <dir>/conv_curated_v2.jsonl --ood <dir>/tasks.jsonl \
        --tokenizer <hf snapshot of the -ct model> --out runs/sft/d1_audit.json

Checks (each prints its count; a row is kept only when every check passes):
  1. resolved      joined to nebius/SWE-rebench-openhands-trajectories (runs/sft/nebius_meta.parquet, the small columns
                   fetched once with --fetch-nebius) by instance_id, disambiguated by the submit_patch result's
                   patch_size == len(model_patch) and the turn count; keep resolved == 1 only
  2. behaviour     from the matched model_patch: edits test files · new file at the repo root · build/config file
                   touched · repro/debug script left in the patch; from the commands: pip install / network; trajectory
                   shape: last tool call is submit_patch + final text reply · model calls ≤ --max-calls · tool names in
                   the 9 harness tools · arguments match the tool schemas (required keys, types, no extra keys)
  3. reasoning     share of turns with reasoning, length (chars, -ct tokens), turns over --max-thought tokens
  4. dedup         repo ∈ OOD-45 repos or ∈ our 4 public repos; patch sha ∈ our 129 gold patches
  5. length        rendered prompt tokens of the last call (whole conversation, our prompt) > --max-tokens
NOT offline (skipped, needs the docker images): tool-result fidelity replay (plan check 2).
Writes --out with the per-row verdicts (instance_id, n, keep, reasons, nebius ids) for convert_adesolanke.py --audit.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_adesolanke as conv  # noqa: E402

REPO = conv.REPO
NEBIUS = os.path.join(REPO, "runs", "sft", "nebius_meta.parquet")
OUR_REPOS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"}
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$|(^|/)tests\.unit(/|$)")
BUILD_RE = re.compile(r"^(setup\.py|setup\.cfg|pyproject\.toml|requirements[^/]*\.txt|tox\.ini|Makefile|MANIFEST\.in|"
                      r"Pipfile[^/]*|poetry\.lock|noxfile\.py|Dockerfile|\.github/.*|\.pre-commit-config\.yaml|"
                      r"[^/]*\.ya?ml|[^/]*\.cfg|[^/]*\.ini)$")
REPRO_RE = re.compile(r"(^|/)(repro|reproduce|debug|scratch|tmp|temp|test_issue|test_fix|verify|check)[^/]*\.(py|sh)$", re.I)
NET_RE = re.compile(r"\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|curl\s|wget\s|git\s+clone|apt(-get)?\s+install|"
                    r"conda\s+install|npm\s+install|uv\s+pip\s+install)")


def fetch_nebius():
    from huggingface_hub import HfFileSystem
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["trajectory_id", "instance_id", "repo", "model_patch", "exit_status", "resolved", "gen_tests_correct",
            "pred_passes_gen_tests", "trajectory.list.element.role",
            "trajectory.list.element.tool_calls.list.element.function.name"]
    fs, t0, parts = HfFileSystem(), time.time(), []
    with fs.open("datasets/nebius/SWE-rebench-openhands-trajectories/trajectories.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            out = []
            for r in pf.read_row_group(g, columns=cols).to_pylist():
                names = [",".join(tc["function"]["name"] for tc in (m.get("tool_calls") or [])) or "-"
                         for m in (r["trajectory"] or []) if m["role"] == "assistant"]
                out.append({k: r[k] for k in cols[:8]} | {"n_msgs": len(r["trajectory"] or []), "calls": names})
            parts.append(pa.Table.from_pylist(out))
            print(f"  nebius row group {g}: {len(out)} rows · {time.time() - t0:.0f} s", flush=True)
    tb = pa.concat_tables(parts)
    pq.write_table(tb, NEBIUS)
    print(f"nebius meta: {tb.num_rows} rows → {NEBIUS} ({os.path.getsize(NEBIUS) / 1e6:.0f} MB, {time.time() - t0:.0f} s)")


def load_nebius():
    import pyarrow.parquet as pq
    by = collections.defaultdict(list)
    for r in pq.read_table(NEBIUS).to_pylist():
        by[r["instance_id"]].append(r)
    return by


def patch_files(patch):
    """[(path, new_file?)] from a unified diff."""
    out, new = [], False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            new = False
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            out.append([m.group(2) if m else line, False])
        elif line.startswith("new file mode") and out:
            out[-1][1] = True
    return [tuple(x) for x in out]


def check_args(tc, schemas):
    f = tc["function"]; name, args = f["name"], f["arguments"]
    if name not in schemas:
        return "tool_name"
    p = schemas[name]
    if not isinstance(args, dict):
        return "args_not_dict"
    props = p.get("properties") or {}
    for k in p.get("required") or []:
        if k not in args:
            return f"missing:{k}"
    types = {"string": str, "integer": int, "boolean": bool, "array": list, "number": (int, float)}
    for k, v in args.items():
        if k not in props:
            return f"extra:{k}"
        t = props[k].get("type")
        if v is None and props[k].get("nullable"):
            continue
        if t in types and not isinstance(v, types[t]) or (t == "integer" and isinstance(v, bool)):
            return f"type:{k}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ood", required=True, help="OOD-45 tasks.jsonl (repo set)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "sft", "d1_audit.json"))
    ap.add_argument("--fetch-nebius", action="store_true", help="(re)download the nebius metadata columns")
    ap.add_argument("--max-calls", type=int, default=28)
    ap.add_argument("--max-thought", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=26624)
    a = ap.parse_args()
    t0 = time.time()
    if a.fetch_nebius or not os.path.exists(NEBIUS):
        fetch_nebius()
    by = load_nebius()
    print(f"nebius meta loaded: {sum(len(v) for v in by.values())} rows · {len(by)} instances")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    system, tools = conv.load_system(), conv.load_tools()
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    ood_repos = collections.Counter(json.loads(l)["repo"] for l in open(a.ood) if l.strip())
    gold_sha = {hashlib.sha256(json.loads(l)["patch"].encode()).hexdigest() for l in
                open(os.path.join(REPO, "harness", "tasks.jsonl")) if l.strip()}
    print(f"OOD-45 repos {len(ood_repos)} · our repos {len(OUR_REPOS)} · our gold patches {len(gold_sha)}")

    rows, seen = [], collections.Counter()
    join = collections.Counter(); exit_status = collections.Counter(); resolved_c = collections.Counter()
    fails = collections.Counter(); build_names = collections.Counter(); root_names = collections.Counter()
    repro_names = collections.Counter(); overlap_repos = collections.Counter(); call_counts = []
    reas_turns = 0; turns = 0; reas_chars = []; reas_tokens = []; over_thought = 0; rows_over_thought = 0
    last_tokens = []; arg_fail = collections.Counter(); net_cmds = collections.Counter()
    for row in conv.iter_rows(a.data):
        iid = row["instance_id"]; n = seen[iid]; seen[iid] += 1
        ms = row["messages"]
        reasons = []
        # ---- 1. nebius join + resolved
        try:
            ps = json.loads(ms[-2]["content"]).get("patch_size") if ms[-2]["role"] == "tool" else None
        except Exception:
            ps = None
        n_turns = sum(1 for m in ms if m["role"] == "assistant")
        cands = by.get(iid, [])
        match, how = None, "unmatched"
        if cands:
            c1 = [c for c in cands if len(c["model_patch"] or "") == ps]
            if len(c1) == 1:
                match, how = c1[0], "unique" if len(cands) == 1 else "patch_size"
            elif len(c1) > 1:
                c2 = [c for c in c1 if sum(1 for x in c["calls"] if x not in ("think", "-")) in (n_turns, n_turns + 1)]
                if len(c2) == 1:
                    match, how = c2[0], "patch_size+turns"
                else:
                    pool = c2 or c1
                    if len({c["resolved"] for c in pool}) == 1:
                        match, how = pool[0], "ambiguous_same_verdict"
                    else:
                        how = "ambiguous_verdicts_differ"
            else:
                how = "patch_size_no_match"
        join[how] += 1
        if match is None:
            reasons.append("unmatched")
            resolved = None
        else:
            exit_status[match["exit_status"][:40]] += 1
            resolved = int(match["resolved"]) == 1
            resolved_c[resolved] += 1
            if not resolved:
                reasons.append("unresolved")
        # ---- 2. behaviour (patch-based when matched)
        if match is not None:
            files = patch_files(match["model_patch"] or "")
            if not files:
                reasons.append("empty_patch")
            if any(TEST_RE.search(p) for p, new in files if not new):
                reasons.append("edits_existing_tests")
            if any(TEST_RE.search(p) for p, new in files if new):
                reasons.append("adds_test_file")
            if any(p == "issue.md" for p, _ in files):
                fails["issue_md_in_patch"] += 1      # the OpenHands setup writes /workspace/issue.md; not the agent's
            roots = [p for p, new in files if new and "/" not in p and p != "issue.md"]
            if roots:
                reasons.append("root_new_file"); root_names.update(os.path.basename(p) for p in roots)
            builds = [p for p, _ in files if BUILD_RE.match(p)]
            if builds:
                reasons.append("build_file"); build_names.update(builds)
            repro = [p for p, new in files if new and REPRO_RE.search(p) and p != "issue.md"]
            if repro:
                reasons.append("repro_left"); repro_names.update(os.path.basename(p) for p in repro)
        cmds = [tc["function"]["arguments"].get("command", "") for m in ms for tc in (m.get("tool_calls") or [])
                if tc["function"]["name"] == "run_command"]
        nets = [c for c in cmds if NET_RE.search(c)]
        if nets:
            reasons.append("network_cmd"); net_cmds.update(NET_RE.search(c).group(0).split()[0] for c in nets)
        tcs_all = [tc for m in ms for tc in (m.get("tool_calls") or [])]
        if not tcs_all or tcs_all[-1]["function"]["name"] != "submit_patch" or ms[-1]["role"] != "assistant" \
                or ms[-1].get("tool_calls"):
            reasons.append("bad_ending")
        call_counts.append(n_turns)
        if n_turns > a.max_calls:
            reasons.append("too_many_calls")
        bad = [check_args(tc, schemas) for tc in tcs_all]
        bad = [b for b in bad if b]
        if bad:
            reasons.append("bad_args"); arg_fail.update(bad)
        # ---- 3. reasoning
        row_over = 0
        for m in ms:
            if m["role"] != "assistant":
                continue
            turns += 1
            r = m.get("reasoning") or ""
            if r:
                reas_turns += 1; reas_chars.append(len(r)); nt = len(tok.encode(r, add_special_tokens=False))
                reas_tokens.append(nt)
                if nt > a.max_thought:
                    over_thought += 1; row_over += 1
        if row_over:
            rows_over_thought += 1; reasons.append("thought_over")
        # ---- 4. dedup
        if row["repo"] in ood_repos or row["repo"] in OUR_REPOS:
            reasons.append("repo_overlap"); overlap_repos[row["repo"]] += 1
        psha = hashlib.sha256((match["model_patch"] or "").encode()).hexdigest() if match else None
        if psha in gold_sha:
            reasons.append("gold_patch_dup")
        # ---- 5. length: the whole conversation under our prompt, rendered with the chat template
        rec = conv.build_record(row, system, tools, resolved=resolved, nebius=None)
        last = rec["calls"][-1]
        from render import normalise  # noqa: E402  (train/render.py, same dir)
        msgs = normalise(last["input_messages"] + [conv.wire_assistant(ms[-1])])
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        ntok = len(tok.encode(text, add_special_tokens=False))   # the template already carries <bos>
        last_tokens.append(ntok)
        if ntok > a.max_tokens:
            reasons.append("too_long")
        for r_ in set(reasons):
            fails[r_] += 1
        rows.append({"instance_id": iid, "n": n, "repo": row["repo"], "keep": not reasons, "reasons": reasons,
                     "resolved": resolved, "join": how, "trajectory_id": match["trajectory_id"] if match else None,
                     "patch_sha": psha, "calls": n_turns, "tokens_last": ntok, "thought_over": row_over})
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} rows · {time.time() - t0:.0f} s", flush=True)

    def pct(xs, q):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    kept = [r for r in rows if r["keep"]]
    calls_ok = {k: sum(1 for c in call_counts if c <= k) for k in (28, 40, 50)}
    print(f"\n== D1 audit · {len(rows)} rows · {len(seen)} instances · {time.time() - t0:.0f} s")
    print(f"1. join: " + " · ".join(f"{k} {v}" for k, v in join.most_common()))
    print(f"   resolved {resolved_c[True]} · unresolved {resolved_c[False]} · unmatched {join['unmatched'] + join['patch_size_no_match'] + join['ambiguous_verdicts_differ']}")
    print(f"   exit_status: " + " · ".join(f"{k} {v}" for k, v in exit_status.most_common(5)))
    print(f"2. behaviour: " + " · ".join(f"{k} {fails[k]}" for k in ("empty_patch", "edits_existing_tests", "adds_test_file", "root_new_file", "build_file", "repro_left", "network_cmd", "bad_ending", "too_many_calls", "bad_args")))
    print(f"   root new files (issue.md excluded; in patch {fails['issue_md_in_patch']} rows): {root_names.most_common(10)}")
    print(f"   build files: {build_names.most_common(10)}")
    print(f"   repro left: {repro_names.most_common(10)}")
    print(f"   network cmds: {net_cmds.most_common()} · bad args: {arg_fail.most_common(10)}")
    print(f"   calls per trajectory: min {min(call_counts)} p50 {pct(call_counts, .5)} p90 {pct(call_counts, .9)} max {max(call_counts)} · ≤28 {calls_ok[28]} · ≤40 {calls_ok[40]} · ≤50 {calls_ok[50]}")
    print(f"3. reasoning: turns {turns} · with reasoning {reas_turns} ({reas_turns / turns:.1%}) · chars p50 {pct(reas_chars, .5)} p90 {pct(reas_chars, .9)} max {max(reas_chars)} · tokens p50 {pct(reas_tokens, .5)} p90 {pct(reas_tokens, .9)} max {max(reas_tokens)} · turns > {a.max_thought} tokens {over_thought} (rows {rows_over_thought})")
    print(f"4. dedup: repo overlap {fails['repo_overlap']} rows {dict(overlap_repos)} · gold patch dup {fails['gold_patch_dup']}")
    print(f"5. length: last-call tokens p50 {pct(last_tokens, .5)} p90 {pct(last_tokens, .9)} max {max(last_tokens)} · > {a.max_tokens}: {fails['too_long']}")
    print(f"   NOT offline: tool-result fidelity replay (docker) — skipped")
    kept_nc = [r for r in rows if not [x for x in r["reasons"] if x != "too_many_calls"]]
    print(f"   without the calls limit: kept {len(kept_nc)} ({len({r['instance_id'] for r in kept_nc})} instances) · of them calls ≤40 {sum(1 for r in kept_nc if r['calls'] <= 40)} · ≤50 {sum(1 for r in kept_nc if r['calls'] <= 50)}")
    print(f"FUNNEL {len(rows)} → kept {len(kept)} ({len({r['instance_id'] for r in kept})} instances, {len({r['repo'] for r in kept})} repos) · fail reasons: " + " · ".join(f"{k} {v}" for k, v in fails.most_common()))
    summary = {"rows": rows, "params": vars(a), "join": dict(join), "fails": dict(fails), "kept": len(kept),
               "calls_le": calls_ok, "kept_without_calls_limit": len(kept_nc), "reasoning": {"turns": turns, "with_reasoning": reas_turns,
                                                   "tokens_p50": pct(reas_tokens, .5), "tokens_p90": pct(reas_tokens, .9)},
               "tokens_last_p50": pct(last_tokens, .5), "tokens_last_max": max(last_tokens), "skipped": ["tool_result_fidelity_replay"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
