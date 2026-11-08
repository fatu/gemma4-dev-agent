#!/usr/bin/env python3
"""Edit-skill drills for E4B (step 2 of the gap plan, BOX_PLAN.md B-3p): short edit_file exercises whose answer is
checked with the harness's own tools, built from library files that are not eval repos, in the call-level schema
(train/schema.md) so render.py and sft_lora.py take them as they are.

Why: E4B's edit_file fails 47 % of the time (31B 10 %). The failed old_string is mostly a bad copy of lines the model
saw several tool outputs back (13 % failures when the lines are in the last output, 45 % at 4+ outputs back) or an
anchor that occurs several times in a repetitive file. Whole 31B trajectories also teach 31B's pacing (B-3n); these
drills teach only where old_string comes from.

  PY=.venv-eval/bin/python
  $PY train/edit_drills.py build --libs click,jinja2,attr,anyio,h11,werkzeug,packaging,markdown_it,yaml,dateutil \
      --n 1500 --tools-from <any traj log> --out runs/sft/drills                                     # training drills
  $PY train/edit_drills.py build --libs sqlalchemy,tqdm,tenacity,starlette --n 300 --seed 1 \
      --tools-from <same> --out runs/sft/drills_heldout                                               # the gate set
  $PY train/edit_drills.py decode runs/sft/drills_heldout --model gemma-4-e4b-it-qat-w4a16-ct --temperature 0 \
      --out runs/drills/base_t0.jsonl                                                                 # server on :8000
  $PY train/edit_drills.py score runs/sft/drills_heldout runs/drills/base_t0.jsonl                    # 📷 gate line
  $PY train/edit_drills.py think runs/sft/drills runs/drills/think_t07.jsonl --out runs/sft/drills_native

Families (record field `family`):
  copy     a read_file view of the site, then 0-5 read_file outputs of other files → the edit with the shortest exact
           old_string that occurs once (distance = tool outputs between the view and the edit)
  anchor   the changed line occurs more than once in the file → the answer must carry the distinguishing lines
  recover  copy + one failed edit_file whose old_string has an E4B-style corruption (the study's mix: token change,
           quote wrap, dropped line, re-wrapped block, JSON wrapper, trailing backslash, too-short anchor) and the
           harness's real error → target 1 reads the lines again, target 2 is the correct edit
Task text (record field `mutation`): the instruction names the site by its def/class scope and the first occurrence of
a token in that scope, never by quoting a line, so old_string has to be copied from the read_file output:
  one line   rename / constant / number / string: "in `Cls.meth`, change the first `True` to `False`" · insert_after:
             "add the line `logger.debug(...)` right after the line that holds the first `name`"
  multi-line (--multi of copy/recover, the two kinds 50/50) rename_local: "in `func`, rename the local variable `v` to
             `v_new` everywhere in that function" (every use an ast.Name, 2-6 lines, not in a docstring/comment/string) ·
             call_arg: "in the first call to `f(…)`, change the argument `k=1` to `k=2`" (the answer is the whole 2-6-line
             call)
Rejected and counted by reason: no_site · leak (a whole old line ≥ 6 chars in the task text) · syntax (the intended file
must ast.parse; no `return_new x`) · window (the view must show the old lines and the scope's def/class line, ≤ 150
lines, not truncated) · corrupt · apply. Scope names must be unique in the file.
Checks: every answer is applied with the harness's edit_file on a copy and must give the intended file (strategy
exact, one occurrence); every corruption must fail under all matcher tiers, so the error in context is the one the
harness returns. Source files: <site-packages>/<lib>/**.py of the eval venv; eval repos (fastapi, rich, requests, httpx)
and pydantic are refused. Each record keeps the file's site-packages path + sha256; score re-reads and checks it.
Context = what the harness serves: --system (the arm's system.md; {problem_description} filled like ADK does),
swegemma's build_agent_prompt (scorer budget 7 min / 28 calls / 80 turns, 240 s commands) with the real workspace-tree
command run on the drill workspace, tool messages from swegemma's own read_file / edit_file, tools from a traj log.
Labels: every call has `label_scope: tool_call` (render.py): loss on the tool call and its stop token only. The thought
is context — E4B's own after `think`, else empty — so the drills do not teach when to edit or how long to think.
Every knob prints its effective value.
"""
import argparse
import ast
import asyncio
import hashlib
import io
import json
import keyword
import os
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tokenize
import types
from collections import Counter
from pathlib import Path

REFUSED_LIBS = {"fastapi", "rich", "requests", "httpx", "pydantic", "pydantic_core"}   # eval repos + fastapi's core dep
DEFAULT_SAMPLING = {"temperature": 0.2, "top_p": 0.95, "max_completion_tokens": 6144,
                    "enable_thinking": True, "thinking_token_budget": 4096}             # latest_e4b sampling.yaml
FAMILIES = (["copy", "anchor", "recover"], [0.5, 0.2, 0.3])
DISTANCES = ([0, 1, 2, 3, 4, 5], [0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
CORRUPTIONS = ["token_change"] * 55 + ["quote_wrap"] * 12 + ["dropped_line"] * 12 + ["reformat"] * 8 + \
              ["trailing_backslash"] * 3 + ["json_wrapper"] * 4 + ["short_nonunique"] * 6   # E4B's own mix (study E)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def site_packages():
    return Path(sysconfig.get_paths()["purelib"])


# ------------------------------------------------- the harness's tool functions on a local drill workspace
class _Res:
    def __init__(self, code=0, out="", err=""):
        self.exit_code, self.stdout, self.stderr, self.timed_out = code, out, err, False


class _LocalDocker:
    """the subset of swegemma's container API that read_file / edit_file use, on a local directory"""
    def __init__(self, root):
        self.root = Path(root)

    def _local(self, p):
        p = str(p)
        return self.root / (p[len("/workspace/"):] if p.startswith("/workspace/") else p.lstrip("/"))

    def copy_from(self, cid, src, dest):
        s = self._local(src)
        if s.exists():
            shutil.copyfile(s, dest)

    def copy_to(self, cid, local, destdir):
        d = self._local(destdir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, d / Path(local).name)

    def exec(self, cid, cmd, timeout=None):
        return _Res()


class _Ctx:
    def __init__(self, root, used=5):
        from swegemma.budget import EvaluationBudget, HarnessLimits
        self.docker, self.container_id = _LocalDocker(root), "drill"
        self.harness, self.budget = HarnessLimits(), EvaluationBudget(time_minutes=7, tool_calls=28)
        self.tool_calls_used, self.tool_lock = used, None

    def check_budget(self, count_tool_call=True):
        self.tool_calls_used += bool(count_tool_call)
        return None


def tool_msg(raw):
    """harness tool message content = json.dumps(dict, ensure_ascii=False) (checked against the traj logs)"""
    return json.dumps(json.loads(raw), ensure_ascii=False)


def n_matches(content, old):
    """occurrences under the harness matcher (any tier); 0 = no match"""
    from swegemma.edit import apply_replacement
    try:
        return apply_replacement(content, old, old + "\u0000", allow_multiple=True).occurrences
    except ValueError:
        return 0


def minimal_unique(lines, l0, l1, content, cap=8):
    """the smallest window [a, b) around the changed lines [l0, l1) that matches exactly once under every matcher tier"""
    a, b = l0, l1
    score = lambda s: -1 if s is None else len(re.findall(r"[A-Za-z_]\w{2,}", s))
    for _ in range(cap):
        old = "\n".join(lines[a:b])
        if old.strip() and n_matches(content, old) == 1 and content.count(old) == 1:
            return a, b
        up = lines[a - 1] if a > 0 else None
        dn = lines[b] if b < len(lines) else None
        if up is not None and score(up) >= score(dn):
            a -= 1
        elif dn is not None:
            b += 1
        elif up is not None:
            a -= 1
    old = "\n".join(lines[a:b])
    return (a, b) if n_matches(content, old) == 1 and content.count(old) == 1 else None


def code_lines(content):
    """0-based lines that hold code: not blank, not a comment, not inside a multi-line string (docstrings)"""
    skip = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(content).readline):
            if t.type == tokenize.STRING and t.end[0] > t.start[0]:
                skip.update(range(t.start[0] - 1, t.end[0]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return [i for i, l in enumerate(content.split("\n"))
            if i not in skip and l.strip() and not l.strip().startswith("#") and len(l) < 160]


IDENT = re.compile(r"[a-z_][a-z0-9_]*")
CONST_SWAP = {"True": "False", "False": "True", "None": "0"}
INSERT = 'logger.debug("checkpoint")'


def plain_name(s, min_len):
    return len(s) >= min_len and IDENT.fullmatch(s) and not keyword.iskeyword(s) and not keyword.issoftkeyword(s)


def bcol(line, col):
    """ast's UTF-8 byte column → str index"""
    return len(line.encode()[:col].decode(errors="ignore"))


class Src:
    """one library file: tokens, def/class scopes, simple one-line statements and the multi-line sites. The instruction
    names a site by its scope and by the first occurrence of a token in that scope, never by quoting a whole line, so
    the old_string has to come from the read_file output"""
    def __init__(self, content):
        self.content, self.lines = content, content.split("\n")
        self.tree = ast.parse(content)
        self.toks = [t for t in tokenize.generate_tokens(io.StringIO(content).readline)
                     if t.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING)]
        self.scopes = []                        # (qualname, first line, last line, node), 1-based, outer before inner

        def walk(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = f"{prefix}.{ch.name}" if prefix else ch.name
                    self.scopes.append((q, ch.lineno, ch.end_lineno, ch))
                    walk(ch, q)
                else:
                    walk(ch, prefix)
        walk(self.tree, "")
        self.qcount = Counter(s[0] for s in self.scopes)
        self.simple = {n.lineno for n in ast.walk(self.tree)
                       if isinstance(n, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and n.lineno == n.end_lineno}
        self.multi = [x for x in (self._rename_local(), self._call_arg()) if x]   # one list per kind, picked 50/50

    def scope(self, ln):
        """(qualname, first line, last line) of the innermost def/class holding line ln; module = (None, 1, n)"""
        best = None
        for s in self.scopes:
            if s[1] <= ln <= s[2] and (best is None or s[1] >= best[1]):
                best = s
        return (best[0], best[1], best[2]) if best else (None, 1, len(self.lines))

    def first(self, t, lo, hi):
        """is token t the first token with its type and text in lines lo..hi?"""
        return next(x for x in self.toks if lo <= x.start[0] <= hi and x.type == t.type and x.string == t.string) is t

    def where(self, q):
        return f"in `{q}`" if q else "at module level"

    def line_site(self, i, rng):
        """a one-line change at 0-based line i, described by scope + first token"""
        li, line = i + 1, self.lines[i]
        q, lo, hi = self.scope(li)
        if q and self.qcount[q] != 1:
            return None
        toks = [t for t in self.toks if t.start[0] == li == t.end[0] and self.first(t, lo, hi)]
        opts = []
        for t in toks:
            if t.type == tokenize.NAME and t.string in CONST_SWAP:
                opts.append(("constant", t, CONST_SWAP[t.string]))
            elif t.type == tokenize.NAME and plain_name(t.string, 4):
                opts.append(("rename", t, t.string + "_new"))
            elif t.type == tokenize.NUMBER and t.string.isdigit():
                opts.append(("number", t, str(int(t.string) + 1)))
            elif t.type == tokenize.STRING and re.fullmatch(r"(['\"])[A-Za-z][^'\"\\]{2,40}\1", t.string):
                opts.append(("string", t, t.string[:-1] + "_v2" + t.string[-1]))
        anchors = [t for t in toks if t.type == tokenize.NAME and plain_name(t.string, 4)] if li in self.simple else []
        if anchors:
            opts.append(("insert_after", rng.choice(anchors), None))
        if not opts:
            return None
        kind, t, nt = rng.choice(opts)
        site = {"kind": kind, "l0": i, "l1": i + 1, "scope": q, "scope_line": lo if q else None}
        if kind == "insert_after":
            ind = line[:len(line) - len(line.lstrip())]
            return dict(site, new_lines=[line, ind + INSERT],
                        desc=f"{self.where(q)}, add the line `{INSERT}` right after the line that holds the first `{t.string}`")
        verb = "rename" if kind == "rename" else "change"
        return dict(site, new_lines=[line[:t.start[1]] + nt + line[t.end[1]:]],
                    desc=f"{self.where(q)}, {verb} the first `{t.string}` to `{nt}`")

    def _rename_local(self):
        """a local variable of a function whose every use (2-6 lines apart) is an ast.Name → rename all of them"""
        out = []
        for q, s, e, node in self.scopes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or self.qcount[q] != 1:
                continue
            args = {x.arg for x in ast.walk(node.args) if isinstance(x, ast.arg)}
            stores = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            for v in sorted(stores - args):
                if not plain_name(v, 3):
                    continue
                names = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == v]
                toks = [t for t in self.toks if t.type == tokenize.NAME and t.string == v and s <= t.start[0] <= e]
                if len(toks) < 2 or len(toks) != len(names):        # an attribute / keyword / global of the same name
                    continue
                body = "\n".join(self.lines[s - 1:e])
                if len(re.findall(rf"\b{v}\b", body)) != len(toks):  # also in a docstring / comment / string:
                    continue                                         # "everywhere" would be ambiguous
                l0, l1 = min(t.start[0] for t in toks) - 1, max(t.start[0] for t in toks)
                if not 2 <= l1 - l0 <= 6:
                    continue
                new = self.lines[l0:l1]
                for t in sorted(toks, key=lambda t: t.start, reverse=True):
                    k = t.start[0] - 1 - l0
                    new[k] = new[k][:t.start[1]] + v + "_new" + new[k][t.end[1]:]
                out.append({"kind": "rename_local", "l0": l0, "l1": l1, "new_lines": new, "scope": q, "scope_line": s,
                            "desc": f"in `{q}`, rename the local variable `{v}` to `{v}_new` everywhere in that function"})
        return out

    def _call_arg(self):
        """the first call to a callee in its scope, spanning 2-6 lines, with a constant keyword argument → change it;
        the answer covers the whole call"""
        out, calls = [], [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)]
        seg = lambda n: ast.get_source_segment(self.content, n)
        for c in calls:
            if not 1 <= c.end_lineno - c.lineno <= 5:
                continue
            callee = seg(c.func)
            if not callee or "\n" in callee or len(callee) > 40:
                continue
            q, lo, hi = self.scope(c.lineno)
            if q and self.qcount[q] != 1:
                continue
            same = [x for x in calls if lo <= x.lineno <= hi and seg(x.func) == callee]
            if min(same, key=lambda x: (x.lineno, x.col_offset)) is not c:
                continue
            for kw in c.keywords:
                val = kw.value
                if not kw.arg or not isinstance(val, ast.Constant) or val.lineno != val.end_lineno:
                    continue
                old = seg(val)
                if isinstance(val.value, bool) or val.value is None:
                    new = CONST_SWAP[old]
                elif isinstance(val.value, int) and old.isdigit():
                    new = str(val.value + 1)
                elif isinstance(val.value, str) and re.fullmatch(r"(['\"])[A-Za-z][^'\"\\]{2,40}\1", old):
                    new = old[:-1] + "_v2" + old[-1]
                else:
                    continue
                l0, l1 = c.lineno - 1, c.end_lineno
                new_lines = self.lines[l0:l1]
                k, line = val.lineno - 1 - l0, self.lines[val.lineno - 1]
                new_lines[k] = line[:bcol(line, val.col_offset)] + new + line[bcol(line, val.end_col_offset):]
                out.append({"kind": "call_arg", "l0": l0, "l1": l1, "new_lines": new_lines, "scope": q,
                            "scope_line": lo if q else None,
                            "desc": f"{self.where(q)}, in the first call to `{callee}(…)`, change the argument "
                                    f"`{kw.arg}={old}` to `{kw.arg}={new}`"})
        return out


def corrupt(old, content, lines, a, b, rng):
    """one E4B-style corruption → (bad, kind); kept only if the full matcher really fails on it (or, for
    short_nonunique, really reports several matches). The corrupted text spans 1-5 lines around the site, like E4B's
    own attempts (median 4 lines), while the answer stays the shortest unique span."""
    a, b = max(0, a - rng.randint(0, 2)), min(len(lines), b + rng.randint(0, 2))
    old = "\n".join(lines[a:b])
    for _ in range(12):
        k, ol, bad = rng.choice(CORRUPTIONS), old.split("\n"), None
        if k == "token_change":
            j = rng.randrange(len(ol))
            toks = list(re.finditer(r"[A-Za-z_]\w+|\d+|[\[\]():]", ol[j]))
            if toks:
                t = rng.choice(toks)
                g = t.group(0)
                rep = {"(": "[", ")": "]", "[": "(", "]": ")", ":": ""}.get(g, g[:-1] + ("s" if not g.endswith("s") else ""))
                ol[j] = ol[j][:t.start()] + rep + ol[j][t.end():]
                bad = "\n".join(ol)
        elif k == "quote_wrap":
            bad = '"' + old + '"'
        elif k == "dropped_line" and len(ol) >= 3:
            j = rng.randrange(1, len(ol) - 1)
            bad = "\n".join(ol[:j] + ol[j + 1:])
        elif k == "reformat" and len(ol) >= 2:
            bad = ol[0].rstrip() + " " + " ".join(x.strip() for x in ol[1:])
        elif k == "trailing_backslash":
            bad = old.rstrip("\n") + "\\"
        elif k == "json_wrapper":
            bad = old + '", "start_line": %d, "end_line": %d, "total_lines": %d, "is_truncated": false}' % (a + 1, b, len(lines))
        elif k == "short_nonunique":
            cand = [x for x in ol if x.strip() and content.count(x) > 1]
            if cand and n_matches(content, cand[0]) > 1:
                return cand[0], k
            continue
        if bad and bad != old and n_matches(content, bad) == 0:
            return bad, k
    return None, None


# ------------------------------------------------- building
class Builder:
    def __init__(self, a):
        self.a, self.rng, self.n = a, random.Random(a.seed), 0
        self.src, self.why = {}, None          # per-file analysis cache · why the last exercise was rejected
        setup = json.loads(open(a.tools_from).readline())["setup"]
        drop = set(filter(None, a.drop_tools.split(",")))
        self.tools = [t for t in setup["tool_schemas"] if t["function"]["name"] not in drop]
        self.tool_names = [t["function"]["name"] for t in self.tools]
        p = setup.get("params") or {}
        eb = p.get("extra_body") or {}
        self.sampling = {"temperature": p.get("temperature", DEFAULT_SAMPLING["temperature"]),
                         "top_p": p.get("top_p", DEFAULT_SAMPLING["top_p"]),
                         "max_completion_tokens": p.get("max_completion_tokens", DEFAULT_SAMPLING["max_completion_tokens"]),
                         "enable_thinking": (eb.get("chat_template_kwargs") or {}).get("enable_thinking", True),
                         "thinking_token_budget": eb.get("thinking_token_budget", DEFAULT_SAMPLING["thinking_token_budget"])}
        self.system = Path(a.system).read_text()
        self.model = (setup.get("model") or "").split("/")[-1] or None

    def user(self, lib, instruction, root):
        from swegemma.budget import EvaluationBudget, HarnessLimits
        from swegemma.harness.agent_runner import build_agent_prompt, build_workspace_tree_command
        cmd = build_workspace_tree_command(workspace_dir=str(root))
        tree = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
        task = types.SimpleNamespace(repo=lib, problem_statement=instruction, hints_text="", base_commit=None)
        cfg = types.SimpleNamespace(budget=EvaluationBudget(time_minutes=7, tool_calls=28, turns=80),
                                    harness=HarnessLimits(command_timeout_seconds=240), graph_dir=None,
                                    embeddings_dir=None, enable_sandbox_testing=None)
        return build_agent_prompt(task, cfg, tree, declared_tools=self.tool_names)

    def call(self, name, args):
        self.n += 1
        cid = f"chatcmpl-tool-drill{self.n:08x}"
        return cid, {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "id": cid, "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}

    def exercise(self, lib, rel, content, root, others, family):
        from swegemma.edit import apply_replacement
        from swegemma.tools import workspace as ws
        rng, lines = self.rng, content.split("\n")
        self.why = "no_site"
        if rel not in self.src:
            try:
                self.src[rel] = Src(content)
            except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
                self.src[rel] = None
        src, cand = self.src[rel], code_lines(content)
        if not src or not cand:
            return None
        multi = family != "anchor" and src.multi and rng.random() < self.a.multi
        for _ in range(30):
            site = rng.choice(rng.choice(src.multi)) if multi else src.line_site(rng.choice(cand), rng)
            if not site:
                continue
            core = "\n".join(lines[site["l0"]:site["l1"]])
            single = n_matches(content, core) == 1 and content.count(core) == 1
            if (family == "anchor" and single) or (family != "anchor" and not single and rng.random() < 0.7):
                continue
            w = minimal_unique(lines, site["l0"], site["l1"], content)
            if not w:
                continue
            a, b = w
            old = "\n".join(lines[a:b])
            new_lines = lines[a:site["l0"]] + site["new_lines"] + lines[site["l1"]:b]
            new = "\n".join(new_lines)
            intended = "\n".join(lines[:a] + new_lines + lines[b:])
            r = apply_replacement(content, old, new)
            if r.strategy == "exact" and r.occurrences == 1 and r.new_content == intended:
                break
        else:
            return None
        i = site["l0"]
        try:                                      # the answer must leave valid Python (no `return_new x`)
            ast.parse(intended)
        except SyntaxError:
            self.why = "syntax"
            return None
        instr = f"In `{rel}`, {site['desc']}."
        if any(len(x.strip()) >= 6 and x.strip() in instr for x in old.split("\n")):
            self.why = "leak"                     # a whole old line quoted in the task: copyable without reading
            return None
        ctx = _Ctx(root)
        s0 = max(1, i + 1 - rng.randint(5, 40))
        e0 = min(len(lines), s0 + rng.randint(30, 120))
        s0 = min(s0, a + 1, site["scope_line"] or s0)     # the view shows the def/class line that names the scope
        e0 = max(e0, b)
        if e0 - s0 + 1 > 150:                     # harness view limit
            e0 = b + min(10, len(lines) - b)
            if e0 - s0 + 1 > 150:
                self.why = "window"
                return None
        view = json.loads(ws.read_file(ctx, rel, s0, e0))
        shown = view.get("content") or ""
        if view.get("is_truncated") or old not in shown or (site["scope_line"] and lines[site["scope_line"] - 1] not in shown):
            self.why = "window"
            return None
        system = self.system.replace("{problem_description}", instr)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": self.user(lib, instr, root)}]
        cid, am = self.call("read_file", {"filepath": rel, "start_line": s0, "end_line": e0})
        msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, rel, s0, e0))}]
        for orel, oc in others:
            ol = oc.split("\n")
            ds = rng.randint(1, max(1, len(ol) - 60))
            de = ds + rng.randint(30, 100)
            cid, am = self.call("read_file", {"filepath": orel, "start_line": ds, "end_line": de})
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, orel, ds, de))}]
        edit = {"id": None, "name": "edit_file",
                "args": json.dumps({"filepath": rel, "new_string": new, "old_string": old}, ensure_ascii=False)}
        tgt = lambda tc: {"reasoning": None, "content": "", "tool_calls": [tc], "finish_reason": "tool_calls"}
        one = lambda k, m, tc: {"call": k, "segment": 0, "events_before": [], "input_messages": m, "target": tgt(tc),
                                "thought_source": "none", "label_scope": "tool_call", "usage_prompt_tokens": None}
        rec = {"instance_id": f"drill_{lib}_{self.a.seed}_{self.n:06d}", "repo": lib, "base_commit": None, "pr": None,
               "patch_sha": None, "source": "edit_drill", "resolved": None, "policy_model": self.model,
               "sampling": dict(self.sampling, max_tool_calls=28, max_time_minutes=7), "env_fingerprint": None,
               "family": family, "distance": len(others), "site_line": i + 1, "old_lines": b - a, "mutation": site["kind"], "scope": site["scope"],
               "verify": {"lib": lib, "path": rel, "file_sha256": sha(content), "intended_sha256": sha(intended)},
               "tools": self.tools}
        if family == "recover":
            bad, kind = corrupt(old, content, lines, a, b, rng)
            if not bad:
                self.why = "corrupt"
                return None
            cid, am = self.call("edit_file", {"filepath": rel, "new_string": new, "old_string": bad})
            err = ws.edit_file(ctx, rel, bad, new)
            if json.loads(err).get("status") != "error":
                self.why = "corrupt"
                return None
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(err)}]
            rs, re_ = max(1, a + 1 - 4), min(len(lines), b + 4)
            reread = {"id": None, "name": "read_file", "args": json.dumps({"filepath": rel, "start_line": rs, "end_line": re_})}
            cid, am = self.call("read_file", {"filepath": rel, "start_line": rs, "end_line": re_})
            msgs2 = msgs + [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, rel, rs, re_))}]
            rec["corruption"] = kind
            rec["calls"] = [one(1, msgs, reread), one(2, msgs2, edit)]
        else:
            rec["calls"] = [one(1, msgs, edit)]
        # the answer through the REAL edit_file on the workspace copy must give the intended file
        out = json.loads(ws.edit_file(_Ctx(root), rel, old, new))
        ok = out.get("status") == "ok" and out.get("strategy", "exact") == "exact" and (Path(root) / rel).read_text() == intended
        (Path(root) / rel).write_text(content)
        self.why = None if ok else "apply"
        return rec if ok else None


def lib_files(lib, min_lines):
    base = site_packages() / lib
    if lib in REFUSED_LIBS:
        sys.exit(f"refused lib {lib}: an eval repo (or its core dependency)")
    if not base.is_dir():
        sys.exit(f"no package dir {base}")
    out = []
    for p in sorted(base.rglob("*.py")):
        if "__pycache__" in p.parts or re.search(r"(^|/)tests?/|/test_", str(p.relative_to(site_packages()))):
            continue
        try:
            c = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(c.splitlines()) >= min_lines and max(map(len, c.splitlines())) < 400:
            out.append((str(p.relative_to(site_packages())), c))
    return out


def build(a):
    libs = [x for x in a.libs.split(",") if x]
    sp = site_packages()
    print(f"edit_drills build · libs {libs} from {sp} · n {a.n} · seed {a.seed} · families {dict(zip(*FAMILIES))} · "
          f"distances {dict(zip(*DISTANCES))} · libraries equally often, files ∝ √size, code lines only · system {a.system} ({len(Path(a.system).read_text())} chars) · "
          f"tools from {a.tools_from} minus [{a.drop_tools}] · min file lines {a.min_lines} · multi-line share (copy/recover) {a.multi} · "
          f"instructions name scope + first token, never a whole line · out {a.out}")
    b = Builder(a)
    print(f"  tools {b.tool_names} · sampling {b.sampling} · policy model {b.model}")
    files = {lib: lib_files(lib, a.min_lines) for lib in libs}
    for lib, fs in files.items():
        print(f"  {lib}: {len(fs)} files ≥ {a.min_lines} lines")
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    st, span = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="drills_") as tmp:
        roots = {}
        for lib, fs in files.items():
            roots[lib] = Path(tmp) / lib
            for rel, c in fs:
                (roots[lib] / rel).parent.mkdir(parents=True, exist_ok=True)
                (roots[lib] / rel).write_text(c)
        libs_ok = [lib for lib in libs if files[lib]]
        weights = {lib: [len(c) ** 0.5 for _, c in files[lib]] for lib in libs_ok}   # bigger files more often
        while sum(v for k, v in st.items() if k.startswith("ok_")) < a.n and st["tried"] < a.n * 6:
            lib = b.rng.choice(libs_ok)                           # libraries equally often
            rel, c = b.rng.choices(files[lib], weights[lib])[0]
            fam = b.rng.choices(*FAMILIES)[0]
            dist = b.rng.choices(*DISTANCES)[0]
            others = [(r2, c2) for r2, c2 in files[lib] if r2 != rel]
            dis = [b.rng.choice(others) for _ in range(dist)] if others else []
            st["tried"] += 1
            rec = b.exercise(lib, rel, c, roots[lib], dis, fam)
            if rec is None:
                st[f"rejected_{fam}:{b.why}"] += 1
                continue
            st[f"ok_{fam}"] += 1
            st[f"kind_{rec['mutation']}"] += 1
            st[f"lib_{lib}"] += 1
            span[min(rec["old_lines"], 6)] += 1
            if fam == "recover":
                st[f"corrupt_{rec['corruption']}"] += 1
            json.dump(rec, open(out / f"{rec['instance_id']}.json", "w"), ensure_ascii=False)
    print(json.dumps({**dict(sorted(st.items())), "old_lines (shortest unique span; 6 = 6+)": dict(sorted(span.items()))}))
    ok_n = sum(v for k, v in st.items() if k.startswith("ok_")) or 1
    print(f"  multi-line old_string {sum(v for k, v in span.items() if k >= 2)} / {ok_n} · rejected: "
          + " · ".join(f"{k} {v}" for k, v in sorted(Counter(k.split(':')[1] for k in st.elements() if k.startswith('rejected_')).items())))
    print(f"== drills: {sum(v for k, v in st.items() if k.startswith('ok_'))} records in {out} "
          f"({st['tried']} tried; every answer applied by edit_file, every corruption failing every matcher tier)")


# ------------------------------------------------- decode / think / score
def load_records(d):
    recs = {}
    for f in sorted(Path(d).glob("*.json")):
        r = json.load(open(f))
        if isinstance(r, dict) and isinstance(r.get("calls"), list):
            recs[r["instance_id"]] = r
    return recs


async def _decode(a):
    import litellm
    recs = load_records(a.records)
    jobs = [(r, c) for r in recs.values() for c in r["calls"] if c.get("target")]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"edit_drills decode · {len(jobs)} target calls from {a.records} · model openai/{a.model} at {a.url} · "
          f"temperature {a.temperature} · samples {a.samples} · concurrency {a.conc} · out {a.out}")
    sem, done, errs = asyncio.Semaphore(a.conc), [0], Counter()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fout = open(a.out, "w")

    async def one(rec, c, k):
        s = rec.get("sampling") or DEFAULT_SAMPLING
        extra = {"chat_template_kwargs": {"enable_thinking": s.get("enable_thinking", True)}}
        if s.get("thinking_token_budget"):
            extra["thinking_token_budget"] = s["thinking_token_budget"]
        async with sem:
            try:
                r = await litellm.acompletion(model=f"openai/{a.model}", api_base=a.url, api_key="EMPTY",
                                              messages=c["input_messages"], tools=rec["tools"],
                                              temperature=a.temperature, top_p=s.get("top_p"),
                                              max_completion_tokens=s.get("max_completion_tokens"), extra_body=extra)
            except Exception as e:                       # one failed request never stops the pass
                errs[type(e).__name__] += 1
                return
        ch = r.choices[0]
        md = ch.message.model_dump() if hasattr(ch.message, "model_dump") else dict(ch.message)
        reasoning = md.get("reasoning_content") or md.get("reasoning") or \
            (md.get("provider_specific_fields") or {}).get("reasoning")
        tcs = [{"name": (t.get("function") or {}).get("name"), "args": (t.get("function") or {}).get("arguments")}
               for t in (md.get("tool_calls") or [])]
        fout.write(json.dumps({"instance_id": rec["instance_id"], "call": c["call"], "sample": k, "model": a.model,
                               "temperature": a.temperature, "finish_reason": ch.finish_reason, "reasoning": reasoning,
                               "content": md.get("content"), "tool_calls": tcs}, ensure_ascii=False) + "\n")
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]} / {len(jobs) * a.samples}", flush=True)

    await asyncio.gather(*(one(r, c, k) for r, c in jobs for k in range(a.samples)))
    fout.close()
    print(f"== decoded {done[0]} / {len(jobs) * a.samples} → {a.out} · request errors {dict(errs) or 0}")


def decode(a):
    asyncio.run(_decode(a))


def score(a):
    """per decoded edit call: parsed, edit_file on the right file, applied exactly once, the intended file — by family
    and distance. The gate reads not-found at distance ≥ 2 and several-matches."""
    from swegemma.edit import apply_replacement
    recs = load_records(a.records)
    sp, files, rows = site_packages(), {}, []
    for l in open(a.decoded):
        m = json.loads(l)
        rec = recs.get(m["instance_id"])
        if not rec:
            continue
        c = rec["calls"][m["call"] - 1]
        tgt = c["target"]["tool_calls"][0]
        v = rec["verify"]
        if v["path"] not in files:
            content = (sp / v["path"]).read_text(encoding="utf-8")
            if sha(content) != v["file_sha256"]:
                sys.exit(f"{v['path']}: file differs from the one the drills were built from (sha256) — another venv?")
            files[v["path"]] = content
        content = files[v["path"]]
        row = {"family": rec["family"], "distance": rec["distance"], "call": m["call"], "target": tgt["name"],
               "outcome": None}
        got = (m.get("tool_calls") or [None])[0]
        if not got:
            row["outcome"] = "no_call" if m.get("finish_reason") != "length" else "cut_off"
        elif got["name"] != tgt["name"]:
            row["outcome"] = f"other_tool:{got['name']}"
        elif tgt["name"] != "edit_file":
            row["outcome"] = "same_tool"
        else:
            try:
                ga = json.loads(got["args"]) if isinstance(got["args"], str) else got["args"]
            except (json.JSONDecodeError, TypeError):
                ga = None
            if not isinstance(ga, dict) or "old_string" not in ga or "new_string" not in ga:
                row["outcome"] = "bad_args"
            elif ga.get("filepath", "").removeprefix("/workspace/") != v["path"]:
                row["outcome"] = "wrong_file"
            else:
                try:
                    r = apply_replacement(content, ga["old_string"], ga["new_string"])
                    row["strategy"] = r.strategy
                    row["outcome"] = "intended" if sha(r.new_content) == v["intended_sha256"] else "applied_other_change"
                except ValueError as e:
                    msg = str(e)
                    row["outcome"] = "several_matches" if ("times" in msg or "locations" in msg or "multiple" in msg) \
                        else "not_found"
        rows.append(row)
    edits = [r for r in rows if r["target"] == "edit_file"]
    tot = Counter(r["outcome"] for r in edits)
    n = len(edits) or 1
    print(f"edit_drills score · {a.decoded} on {a.records} · {len(rows)} decoded calls, {len(edits)} edit targets")
    print("  edit outcomes:", {k: f"{v} ({100 * v / n:.0f}%)" for k, v in tot.most_common()})
    for key in ("family", "distance"):
        groups = sorted({r[key] for r in edits})
        for g in groups:
            sub = [r for r in edits if r[key] == g]
            k = Counter(r["outcome"] for r in sub)
            print(f"  {key} {g}: n {len(sub)} · intended {100 * k['intended'] / len(sub):.0f}% · not_found "
                  f"{100 * k['not_found'] / len(sub):.0f}% · several {100 * k['several_matches'] / len(sub):.0f}%")
    rr = [r for r in rows if r["target"] == "read_file"]
    if rr:
        print(f"  recover call 1 (target read_file): same tool {sum(r['outcome'] == 'same_tool' for r in rr)} / {len(rr)}")
    far = [r for r in edits if r["distance"] >= 2]
    nf = sum(r["outcome"] == "not_found" for r in far) / (len(far) or 1)
    sev = tot["several_matches"] / n
    parsed = 1 - (tot["no_call"] + tot["bad_args"] + tot["cut_off"]) / n
    intended = tot["intended"] / n
    ok = intended >= 0.80 and nf <= 0.10 and sev <= 0.03 and parsed >= 0.99
    print(f"== gate: intended {100 * intended:.0f}% (≥ 80) · not_found at distance ≥ 2 {100 * nf:.0f}% (≤ 10) · "
          f"several matches {100 * sev:.0f}% (≤ 3) · parsed {100 * parsed:.0f}% (≥ 99) → {'PASS' if ok else 'not yet'}")


def think(a):
    """E4B's own thought into each target whose sampled action matches it (same tool; for edit_file the same file);
    the answer stays the checked one, the loss stays on the tool call (label_scope)"""
    recs = load_records(a.records)
    best = {}
    for l in open(a.decoded):
        m = json.loads(l)
        rec = recs.get(m["instance_id"])
        if not rec or not m.get("reasoning") or len(m["reasoning"]) > a.max_thought_chars:
            continue
        c = rec["calls"][m["call"] - 1]
        tgt, got = c["target"]["tool_calls"][0], (m.get("tool_calls") or [None])[0]
        if not got or got["name"] != tgt["name"]:
            continue
        if tgt["name"] == "edit_file":
            try:
                if json.loads(got["args"]).get("filepath", "").removeprefix("/workspace/") != rec["verify"]["path"]:
                    continue
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        best.setdefault((m["instance_id"], m["call"]), m["reasoning"])
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    st = Counter()
    for iid, rec in recs.items():
        for k, c in enumerate(rec["calls"]):
            th = best.get((iid, c["call"]))
            st["calls"] += 1
            if th:
                c["target"]["reasoning"], c["thought_source"] = th, "native"
                st["native"] += 1
                if k + 1 < len(rec["calls"]):     # the next call's history re-sends this reply with its thought,
                    tc = c["target"]["tool_calls"][0]   # as the harness does (keeps the segment's prefix property)
                    hist = [m for m in rec["calls"][k + 1]["input_messages"] if m.get("role") == "assistant" and
                            (m.get("tool_calls") or [{}])[0].get("function", {}).get("name") == tc["name"] and
                            (m.get("tool_calls") or [{}])[0].get("function", {}).get("arguments") == tc["args"]]
                    if not hist:
                        sys.exit(f"{iid}: call {c['call']}'s reply not found in call {c['call'] + 1}'s history")
                    hist[-1]["reasoning_content"] = th
            c["label_scope"] = "tool_call"
        json.dump(rec, open(out / f"{iid}.json", "w"), ensure_ascii=False)
    print(f"== think: {st['native']} / {st['calls']} target calls got E4B's own thought ({a.decoded}, ≤ "
          f"{a.max_thought_chars} chars, matching action); the rest keep an empty thought · {len(recs)} records → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--libs", required=True)
    b.add_argument("--n", type=int, default=1500)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tools-from", required=True, help="a traj log: setup.tool_schemas + setup.params")
    b.add_argument("--drop-tools", default="", help="comma list, e.g. the 3 graph tools for a tools6 arm")
    b.add_argument("--system", default="scripts/arms/latest_e4b/prompts/system.md")
    b.add_argument("--min-lines", type=int, default=60)
    b.add_argument("--multi", type=float, default=0.45, help="share of copy/recover drills drawn from multi-line sites")
    b.add_argument("--out", required=True)
    d = sub.add_parser("decode")
    d.add_argument("records")
    d.add_argument("--model", required=True, help="served model or LoRA name")
    d.add_argument("--url", default="http://127.0.0.1:8000/v1")
    d.add_argument("--temperature", type=float, default=0.0)
    d.add_argument("--samples", type=int, default=1)
    d.add_argument("--conc", type=int, default=16)
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("records")
    s.add_argument("decoded")
    t = sub.add_parser("think")
    t.add_argument("records")
    t.add_argument("decoded")
    t.add_argument("--max-thought-chars", type=int, default=12000)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": build, "decode": decode, "score": score, "think": think}[a.cmd](a)


if __name__ == "__main__":
    main()
