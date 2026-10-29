#!/usr/bin/env python3
"""A5 · parser round-trip: feed every training target back through the scorer's vLLM 0.19.1 gemma4 parsers.

    # laptop: the real parser modules pulled out of the competition wheel + stubs (train/vllm_shim/README.md)
    .venv-eval/bin/python train/parser_check.py runs/sft/e4b_w7 --tokenizer <-ct snapshot> \
        --wheel harness/wheelhouse/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl --out runs/sft/e4b_w7.parser.json
    # pod: the installed vLLM (authoritative)
    python train/parser_check.py runs/sft/e4b_w7 --tokenizer google/gemma-4-E4B-it-qat-w4a16-ct --out …

Per call: the reply text we train on (render.py's rendering of the target, minus the stop token, which vLLM never
returns) → Gemma4ReasoningParser.extract_reasoning → Gemma4ToolParser.extract_tool_calls on the content, the order
of vLLM's non-streaming chat path. Checks: (1) reasoning parsed == target.reasoning, (2) content parsed ==
target.content, (3) tool calls: same count, names, and arguments (JSON-equal), (4) every tool call is legal
against the request's tool schemas (name ∈ tools, required keys present, no unknown keys, JSON types), (5) no
control token leaks into reasoning / content text. A call failing any check is listed and excluded (verdict file
--out); the exclusion rate is printed. Exit 0 when the script ran; A5 is judged by the printed numbers.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import zipfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import CTRL, STOP_TOOL, STOP_TURN, Renderer  # noqa: E402

REAL_FILES = ["vllm/reasoning/abs_reasoning_parsers.py", "vllm/reasoning/basic_parsers.py",
              "vllm/reasoning/gemma4_reasoning_parser.py", "vllm/reasoning/gemma4_utils.py",
              "vllm/tool_parsers/abstract_tool_parser.py", "vllm/tool_parsers/gemma4_tool_parser.py",
              "vllm/tool_parsers/gemma4_utils.py"]
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def load_parsers(wheel):
    if wheel:
        tmp = tempfile.mkdtemp(prefix="vllm_parsers_")
        with zipfile.ZipFile(wheel) as z:
            for f in REAL_FILES:
                z.extract(f, tmp)
        shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_shim", "vllm"),
                        os.path.join(tmp, "vllm"), dirs_exist_ok=True)
        sys.path.insert(0, tmp)
        src = f"wheel {os.path.basename(wheel)} + stubs ({tmp})"
    else:
        src = "installed vllm"
    from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
    from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
    import vllm.tool_parsers.gemma4_tool_parser as m
    try:
        import importlib.metadata as md
        ver = md.version("vllm")
    except Exception:
        ver = "0.19.1 (wheel)"
    return Gemma4ReasoningParser, Gemma4ToolParser, src, ver, m.__file__


def reply_text(r, rec, c):
    """the reply render.py trains on (same branch logic as Renderer.render_call, text only)"""
    p = r.prompt(c["input_messages"], rec["tools"])
    open_in_prompt = p.endswith("<|channel>thought\n")
    t = c["target"]
    if t.get("tool_calls") and (t.get("content") or "").strip():
        tc_only = r.reply_synthetic(dict(t, content=""), open_in_prompt)
        head, sep, tail = tc_only.partition("<channel|>") if "<channel|>" in tc_only else ("", "", tc_only)
        return head + sep + t["content"].strip() + tail, open_in_prompt
    return r.reply_synthetic(t, open_in_prompt), open_in_prompt


def check_schema(name, args, schemas):
    s = schemas.get(name)
    if s is None:
        return f"unknown tool {name!r}"
    props, req = s.get("properties") or {}, s.get("required") or []
    missing = [k for k in req if k not in args]
    unknown = [k for k in args if k not in props]
    if missing:
        return f"{name}: missing required {missing}"
    if unknown and not s.get("additionalProperties", True) or unknown:
        return f"{name}: unknown keys {unknown}"
    for k, v in args.items():
        ty = (props[k].get("type") if isinstance(props[k], dict) else None)
        if ty is None and isinstance(props[k], dict) and "anyOf" in props[k]:
            opts = [o.get("type") for o in props[k]["anyOf"]]
            if "null" in opts and v is None:
                continue
            if not any(isinstance(v, TYPES[o]) and not (o in ("integer", "number") and isinstance(v, bool))
                       for o in opts if o in TYPES):
                return f"{name}.{k}: type {type(v).__name__} not in {opts}"
            continue
        if ty in TYPES:
            if ty in ("integer", "number") and isinstance(v, bool) or not isinstance(v, TYPES[ty]):
                return f"{name}.{k}: expected {ty}, got {type(v).__name__}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wheel", default=None, help="vLLM wheel to pull the parser modules from (laptop); omit on the pod")
    ap.add_argument("--out", default=None, help="per-call verdicts json")
    ap.add_argument("--show", type=int, default=20, help="failures printed in full")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    RP, TP, src, ver, path = load_parsers(a.wheel)
    rp, tp = RP(tok), TP(tok)
    r = Renderer(tok)
    req = SimpleNamespace(tools=None, tool_choice="auto", skip_special_tokens=False)
    print(f"parser_check · records {a.records} · tokenizer {a.tokenizer} · parsers from {src} · vllm {ver}\n  {path}")
    files = sorted(glob.glob(os.path.join(a.records, "*.json")))
    n = {"calls": 0, "targets": 0, "ok": 0, "fail": 0, "reasoning_ws_only": 0, "content_ws_only": 0}
    kinds, fails, verdicts = {}, [], {}
    for f in files:
        rec = json.load(open(f))
        if not isinstance(rec.get("calls"), list) or "tools" not in rec:   # not a record
            continue
        schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in rec["tools"]}
        if not req.tools:
            req.tools = rec["tools"]
            print(f"  tools in the request: {len(schemas)} · {' '.join(schemas)}")
        for c in rec["calls"]:
            n["calls"] += 1
            t = c.get("target")
            if not t:
                continue
            n["targets"] += 1
            key = f"{rec['instance_id']} c{c['call']}"
            reply, open_in_prompt = reply_text(r, rec, c)
            stop = STOP_TOOL if t.get("tool_calls") else STOP_TURN
            assert reply.endswith(stop), key
            out = reply[:-len(stop)]
            problems = []
            reasoning, content = rp.extract_reasoning(out, req)
            info = tp.extract_tool_calls(content if content is not None else "", req)
            # (1) reasoning
            want_r = t.get("reasoning") or None
            if (reasoning or None) != want_r:
                if (reasoning or "").strip() == (want_r or "").strip():
                    n["reasoning_ws_only"] += 1
                else:
                    problems.append(f"reasoning: parsed {len(reasoning or '')} chars vs target {len(want_r or '')}")
            # (2) content
            want_c, got_c = (t.get("content") or "").strip(), (info.content or "").strip()
            if got_c != want_c:
                if got_c.replace("<channel|>", "").strip() == want_c.replace("<channel|>", "").strip():
                    n["content_ws_only"] += 1
                else:
                    problems.append(f"content: parsed {got_c[:60]!r} vs target {want_c[:60]!r}")
            # (3) tool calls
            want_tc = t.get("tool_calls") or []
            got_tc = info.tool_calls
            if len(got_tc) != len(want_tc):
                problems.append(f"tool calls: parsed {len(got_tc)} vs target {len(want_tc)}")
            else:
                for g, w in zip(got_tc, want_tc):
                    wargs = w["args"] if isinstance(w.get("args"), dict) else json.loads(w.get("args") or "{}")
                    try:
                        gargs = json.loads(g.function.arguments)
                    except json.JSONDecodeError as e:
                        problems.append(f"tool call {g.function.name}: arguments not JSON ({e})"); continue
                    if g.function.name != w["name"]:
                        problems.append(f"tool call name {g.function.name!r} vs {w['name']!r}")
                    elif gargs != wargs:
                        diff = [k for k in set(gargs) | set(wargs) if gargs.get(k) != wargs.get(k)]
                        problems.append(f"tool call {w['name']}: arguments differ in {diff}")
                    # (4) schema
                    err = check_schema(g.function.name, gargs, schemas)
                    if err:
                        problems.append(f"schema: {err}")
            # (5) control tokens leaking into text
            for label, txt in (("reasoning", reasoning or ""), ("content", info.content or "")):
                leaks = [x for x in CTRL | {'<|"|>'} if x in txt]
                if leaks:
                    problems.append(f"{label} contains control tokens {leaks}")
            if info.tools_called and not want_tc:
                problems.append("parser found tool calls in a text-only target")
            verdicts[key] = {"ok": not problems, "problems": problems, "open_in_prompt": open_in_prompt,
                             "tool_calls": len(want_tc)}
            if problems:
                n["fail"] += 1
                for pr in problems:
                    kinds[pr.split(":")[0]] = kinds.get(pr.split(":")[0], 0) + 1
                fails.append((key, problems))
            else:
                n["ok"] += 1
    for key, problems in fails[:a.show]:
        print(f"  FAIL {key}: " + " | ".join(problems))
    if len(fails) > a.show:
        print(f"  … {len(fails) - a.show} more")
    print(f"A5 · calls {n['calls']} · targets {n['targets']} · parse OK {n['ok']} · excluded {n['fail']} "
          f"({100 * n['fail'] / max(n['targets'], 1):.1f}%) · whitespace-only diffs (not failures): reasoning "
          f"{n['reasoning_ws_only']} (the template closes a thought with '\\n<channel|>'; the parser keeps that '\\n', "
          f"the server's stored reasoning has none) · content {n['content_ws_only']}")
    print("  failure kinds: " + (" · ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"))
    if a.out:
        json.dump({"records": a.records, "parsers": src, "vllm": ver, "summary": n, "kinds": kinds, "calls": verdicts},
                  open(a.out, "w"), indent=1)
        print(f"  verdicts → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
