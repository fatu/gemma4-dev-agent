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
      --block-libs sqlmodel,huggingface_hub,pandas --tools-from <same> --out runs/sft/drills_heldout   # the gate set
  $PY train/edit_drills.py build --libs sqlalchemy --block-libs sqlmodel,huggingface_hub,pandas --families block=1 \
      --n 80 --seed 2 --tools-from <same> --out runs/sft/drills_heldout_block                        # block slice
  $PY train/edit_drills.py decode runs/sft/drills_heldout --model gemma-4-e4b-it-qat-w4a16-ct --temperature 0 \
      --out runs/drills/base_t0.jsonl                                                                 # server on :8000
  $PY train/edit_drills.py score runs/sft/drills_heldout runs/drills/base_t0.jsonl                    # 📷 gate line
  $PY train/edit_drills.py think runs/sft/drills runs/drills/think_t07.jsonl --out runs/sft/drills_native

Families (record field `family`, weights FAMILIES):
  copy     a read_file view of the site, then 0-9 other tool outputs (distance; DISTANCES) → the edit with the shortest
           exact old_string that occurs once. The outputs between are `grep -rn <name>` results through the harness's
           own run_command on the drill workspace (GREP_FILLER) or read_file windows of other files (field `fillers`)
  anchor   the changed line occurs more than once in the file → the answer must carry the distinguishing lines
  recover  copy + one failed edit_file and the harness's real error → target 1 reads the lines again, target 2 is the
           correct edit. Corruptions by CORRUPTIONS weight, after E4B's own failed edits (study E): too short an
           anchor (short_nonunique: the changed lines alone, which occur several times → "occurs N times") ≈ 40 %,
           copied wrong (token change, dropped line, re-wrapped block) ≈ 45 %, JSON wrapper / quotes / backslash ≈ 15 %
  block    a site inside a 3-line block that repeats in the file (sibling classes / functions copy-pasted, like
           fastapi's), from --block-libs; the shortest unique span is 3-25 lines (BLOCK_LINES, BLOCK_CAP). BLOCK_FAIL of
           them first copy the longest span that still repeats ("occurs N times", corruption block_nonunique), then
           target 1 reads again and target 2 is the edit
  retry    E4B's real retry loops (all-129: after a failed edit E4B retried straight away 103 of 147 times, ok 14 %;
           it resent the identical old_string 48 times, 0 ok; 31B re-reads first and is ok 81 %). Context: copy + a
           failed edit_file (FAILED_LINES long, lines misremembered at MISQUOTE strength, both from E4B's real failures)
           whose error is drawn by RETRY_ERRORS from E4B's real not-found / several-matches causes
           (too short an anchor · a misremembered line copied into old AND new · a line altered in old only · invented
           lines · text from before an earlier edit of the file, stale_text · a skipped line; ESCAPED of the not-found
           ones also written with literal \n escapes, as E4B does) and a loop state by
           RETRY_STATES: once (one failure) · resent (the identical call failed twice) · reread_resent (failed, read the
           lines again, resent the identical call, failed again). Targets: once / resent → target 1 reads the lines
           again, target 2 is the correct edit; reread_resent → the correct edit (the fresh view is in context). score
           counts edits that resend a failed old_string from the context (`resend`)
  stale    the task asks for two changes; the first is already done in the context (edit_file ok, with its diff) and
           changed a line within 3 lines of the second site; the answer window covers that line, so copying the first
           view fails (checked) and the old_string has to come from the edit's own output (verify.pre_edit: score
           applies it before the answer)
Task text (record field `mutation`): the instruction names the site by its def/class scope and the first occurrence of
a token (or a parameter / callee / variable name) in that scope, never by quoting a line, so old_string has to be
copied from the read_file output. Every change is a small, bug-fix-like edit that keeps the file valid Python:
  one line (drawn by LINE_KINDS weight) operator `<`→`<=`, `==`→`!=` … · logic `and`↔`or`, delete `not`, `is`→`is not` ·
             constant True/False/None · number +1 · string first-letter case · default ("change the default value of the
             parameter `p` from `1` to `2`") · swap_args (the first two positional arguments of the first call to `f`) ·
             insert_guard (after the first assignment to `x` in a function: `if x is None: return None`, `if not x: raise
             ValueError(...)` or `assert x is not None`, in the file's indentation)
  multi-line (--multi of copy/recover, the two kinds 50/50) rename_local: every use of a local (all ast.Name, 2-6 lines,
             not in a docstring/comment/string) renamed to an unused name from VOCAB, so the code keeps working ·
             call_arg: "in the first call to `f(…)`, change the argument `k=1` to `k=2`" (the answer is the whole
             2-6-line call)
Rejected and counted by reason: dup (same file, line, kind) · overlap (answer window overlaps an earlier drill of the
file; ≤ --file-cap drills per file, libraries kept level) · no_site · leak (a whole old line ≥ 6 chars in the task text) · syntax (the intended file
must ast.parse; no `return_new x`) · window (the view must show the old lines and the scope's def/class line, ≤ 150
lines, not truncated) · corrupt · apply. Scope names must be unique in the file.
Duplicates seen (record field `dup_seen`): when the changed lines occur elsewhere in the file, another copy is always
in the context: already in the view (in_view), the view widened to the nearest copy (view, DUP_VIEW), or a
`grep -nF '<first changed line>' <file>` through run_command right after the view (grep: a uniqueness check, counted in
`distance`); none = the lines are unique.
"Elsewhere" also counts the first changed line stripped inside another line (an anchor copied without its indentation
matches there too).
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
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tokenize
import types
from collections import Counter
from pathlib import Path

REFUSED_LIBS = {"fastapi", "rich", "requests", "httpx", "httpx2", "pydantic", "pydantic_core"}   # eval repos (+ an httpx fork) + fastapi's core dep
DEFAULT_SAMPLING = {"temperature": 0.2, "top_p": 0.95, "max_completion_tokens": 6144,
                    "enable_thinking": True, "thinking_token_budget": 4096}             # latest_e4b sampling.yaml
FAMILIES = (["copy", "anchor", "recover", "stale", "block", "retry"], [0.37, 0.1, 0.2, 0.1, 0.08, 0.15])   # block ≈ 8 % of drills ≈ 15 % of
# label tokens (its answers are long: at 14 % of drills it held 26 % of the label tokens, at 11 % 20 % → a pull toward
# long anchors everywhere)
# tool outputs between the view and the edit: E4B's not-found rate is 13 % at 0, 38 % at 2-3, 45 % at 4+ (study E)
DISTANCES = (list(range(12)), [0.05, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.1, 0.09, 0.08, 0.07, 0.07])   # tokens p50 ≈ 8K
GREP_FILLER = 0.5          # share of those outputs that are `grep -rn` results (the rest: read_file of another file)
EMPTY_GREP = 0.18          # share of the greps that find nothing (a one-letter typo; harness CommandError, exit 1), as E4B's
# answer old_string length in lines: the 31B's successful edits (all-129: 1 line 33 %, 3-5 lines 33 %, 9+ 22 % folded
# into 6-8 here). The answer is the shortest unique span grown to whole statements, then to this length
ANCHOR_LINES = (list(range(1, 9)), [60, 1, 6, 10, 9, 9, 9, 12])   # 1 = keep the shortest span (only ~half are 1 line)
RARE_NAME = 6              # grep a name the library uses at most this often (E4B's greps: median 130 chars of output) …
RARE_SHARE = 0.7           # … this share of the time; else any name (E4B's long tail: p90 3.2K chars, 8 % at the cap)
# the changed lines occur elsewhere in the file but the view shows no other copy (139 of 1,500 drills at 8507112): then
# whether a short anchor is unique cannot be learned from the context. Widen the view to the nearest other copy (this
# share, when the view stays ≤ 150 lines), else a `grep -nF` of the first changed line right after the view lists every copy
DUP_VIEW = 0.5
# E4B's several-matches failures (all-129): 52 of 67 in fastapi, 36 of 55 failed old_strings 3+ lines — whole blocks
# copy-pasted across sibling classes, where the line that tells the copies apart is far away. Code lines inside a
# repeated 3-line block: fastapi 46 %, the 10 training libraries ~ 1-20 %. block drills come from --block-libs (the
# most repetitive non-eval libraries), sites inside a repeated BLOCK_LINES-line block whose shortest unique span is
# ≥ BLOCK_LINES lines (up to BLOCK_CAP); BLOCK_FAIL of them first copy the longest span that still repeats (the
# harness's "occurs N times"), then read again and edit
BLOCK_LINES = 3
BLOCK_CAP = 25
BLOCK_FAIL = 0.5
# E4B's failed edits (study E, 170): copied wrong 36 % · several matches 32 % · stale 13 % (own family) · call
# syntax / JSON wrapper 11 % → the recover corruptions, drawn by weight before the site is chosen
CORRUPTIONS = (["token_change", "dropped_line", "reformat", "short_nonunique", "json_wrapper", "quote_wrap",
                "trailing_backslash"], [30, 8, 7, 40, 6, 6, 3])
# retry family, from E4B's real failed edits (all-129 base; 126 not-found + 67 several-matches errors, but only 63 + 40
# distinct (task, old_string): the rest are identical resends). The failed edit's cause by distinct count (literal \n
# escapes undone first: the harness's matcher undoes them too, so they are never the cause): several matches 40 · a
# misremembered line in old and new 23 · a line altered in old only 11 · invented 11 · stale text 10 · a skipped line 8.
# The loop state before the target: after a failed edit E4B retried straight away 103 of 147 times, re-read first 44;
# identical resends 48 (0 ok)
RETRY_ERRORS = (["short_nonunique", "misquote_both", "token_change", "invented", "stale_text", "dropped_line"],
                [40, 23, 11, 11, 10, 8])
RETRY_STATES = (["once", "resent", "reread_resent"], [0.4, 0.35, 0.25])
# the failed old_string's length in lines (the 63 distinct real ones, escaped lines counted: p50 6 · p90 26; buckets 7-8 /
# 9-12 / 13-20 / 21-40 drawn uniformly inside); how far a misremembered line is from the real one (distinct, lines with
# escapes left out: best similarity to any line of the file p10 0.39 · p25 0.80 · p50 0.91 · p75 0.95 → small / medium /
# large: 1 / 2 / 3-4 of token_change's LLM-like misrememberings, MISREMEMBER_OPS); ESCAPED of the not-found ones written with literal \n / \"
# escapes like 16 of the 63 real ones (5 the whole span on one line, 11 a few lines joined): a look, not the cause
FAILED_LINES = ([1, 2, 3, 4, 5, 6, 8, 12, 20, 40], [5, 6, 9, 5, 3, 5, 4, 8, 7, 11])
MISQUOTE = (["small", "medium", "large"], [50, 35, 15])
ESCAPED = (0.25, 5 / 16)                  # share of not-found failed edits escaped · of those, the whole span on one line


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def site_packages():
    return Path(sysconfig.get_paths()["purelib"])


# ------------------------------------------------- the harness's tool functions on a local drill workspace
class _Res:
    def __init__(self, code=0, out="", err=""):
        self.exit_code, self.stdout, self.stderr, self.timed_out = code, out, err, False


class _LocalDocker:
    """the subset of swegemma's container API that read_file / edit_file / run_command use, on a local directory.
    exec runs a command only when run=True (the grep fillers); otherwise it is a no-op as before"""
    def __init__(self, root, run=False):
        self.root, self.run = Path(root), run

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
        if not self.run:
            return _Res()
        p = subprocess.run(["bash", "-c", cmd], cwd=self.root, capture_output=True, text=True, timeout=timeout or 60)
        return _Res(p.returncode, p.stdout, p.stderr)


class _Ctx:
    def __init__(self, root, used=5, run=False):
        from swegemma.budget import EvaluationBudget, HarnessLimits
        self.docker, self.container_id = _LocalDocker(root, run), "drill"
        self.harness = HarnessLimits(command_timeout_seconds=240)
        self.budget = EvaluationBudget(time_minutes=7, tool_calls=28)
        self.tool_calls_used, self.tool_lock, self.remaining_time_seconds = used, None, 300

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


def block_lines(lines):
    """0-based lines inside a BLOCK_LINES-line block (stripped, no near-empty line) that occurs more than once"""
    st = [x.strip() for x in lines]
    tri = lambda i: tuple(st[i:i + BLOCK_LINES])
    full = [i for i in range(len(st) - BLOCK_LINES + 1) if all(len(x) > 3 for x in tri(i))]
    seen = Counter(tri(i) for i in full)
    return sorted({j for i in full if seen[tri(i)] > 1 for j in range(i, i + BLOCK_LINES)})


def repeated_span(lines, l0, l1, content, cap=BLOCK_CAP):
    """the longest span around the changed lines [l0, l1) that still occurs more than once, grown along the same path
    as minimal_unique (so one more step makes it unique): E4B's whole-block anchor"""
    a, b, best = l0, l1, None
    score = lambda s: -1 if s is None else len(re.findall(r"[A-Za-z_]\w{2,}", s))
    for _ in range(cap):
        old = "\n".join(lines[a:b])
        if not old.strip() or n_matches(content, old) < 2:
            break
        best = (a, b)
        up = lines[a - 1] if a > 0 else None
        dn = lines[b] if b < len(lines) else None
        if up is not None and score(up) >= score(dn):
            a -= 1
        elif dn is not None:
            b += 1
        elif up is not None:
            a -= 1
    return best


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
OP_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
LOGIC = {"and": "or", "or": "and"}
# one-line kinds, drawn by weight before a line is chosen (rename_local / call_arg come from --multi)
LINE_KINDS = (["operator", "logic", "constant", "number", "string", "default", "swap_args", "insert_guard"],
              [18, 12, 10, 6, 3, 8, 6, 12])
# realistic replacement names for rename_local (a name already in the file as a word is never used)
VOCAB = ["item", "entry", "record", "payload", "current", "target", "source", "options", "config", "handler", "buffer",
         "token", "node", "parts", "spec", "info", "meta", "pending", "output", "chunk_data", "index", "count", "total",
         "label", "field", "element", "candidate", "selected", "matched", "resolved", "parsed", "raw_value", "normalized",
         "prepared", "remaining", "extra", "fallback", "position", "offset", "segment", "fragment", "context", "state",
         "outcome", "response", "request_obj", "message", "pieces", "values", "mapping", "lookup", "cursor", "limit",
         "window", "batch", "group_key", "attempt", "step", "marker", "prefix_text", "suffix_text", "base_value"]
# insert_guard templates on the first assignment to x inside a function: (lines, how the task names them)
GUARDS = [(["if {x} is None:", "{u}return None"], "two lines: `if {x} is None:` and, one level deeper, `return None`"),
          (["if not {x}:", '{u}raise ValueError("{x} must not be empty")'],
           'two lines: `if not {x}:` and, one level deeper, `raise ValueError("{x} must not be empty")`'),
          (["assert {x} is not None"], "the line `assert {x} is not None`")]


def plain_name(s, min_len):
    return len(s) >= min_len and IDENT.fullmatch(s) and not keyword.iskeyword(s) and not keyword.issoftkeyword(s)


def bcol(line, col):
    """ast's UTF-8 byte column → str index"""
    return len(line.encode()[:col].decode(errors="ignore"))


def flip_case(lit):
    """a short string literal with its first letter's case swapped ('Environment' → 'environment')"""
    q, body = lit[0], lit[1:-1]
    return q + body[0].swapcase() + body[1:] + q


def const_swap(node, seg):
    """the replacement text for a constant (bool / None / int / short word string), or None"""
    v = node.value
    if isinstance(v, bool) or v is None:
        return CONST_SWAP[seg]
    if isinstance(v, int) and seg.isdigit():
        return str(v + 1)
    if isinstance(v, str) and re.fullmatch(r"(['\"])[A-Za-z][^'\"\\]{2,40}\1", seg):
        return flip_case(seg)
    return None


class Src:
    """one library file: tokens, def/class scopes and the candidate sites. The task names a site by its scope and by the
    first occurrence of a token (or a parameter / callee / variable name) in that scope, never by quoting a whole line,
    so the old_string has to come from the read_file output. Every change keeps the file valid Python, and the renames
    keep it working (all uses of a local, to a name the file does not use)"""
    def __init__(self, content):
        self.content, self.lines = content, content.split("\n")
        self.tree = ast.parse(content)
        self.toks = [t for t in tokenize.generate_tokens(io.StringIO(content).readline)
                     if t.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING)
                     or (t.type == tokenize.OP and t.string in OP_SWAP)]
        self.words = set(re.findall(r"[A-Za-z_]\w*", content))
        self.unit = "\t" if any(l.startswith("\t") for l in self.lines) else "    "
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
        # def/class header lines (signature, annotations): value kinds never edit them (`-> None` → `-> 0`)
        self.header = {ln for _, _, _, n in self.scopes for ln in range(n.lineno, n.body[0].lineno)}
        self.stmts = []                         # (first, last) line of each statement, compound ones by header only
        for n in ast.walk(self.tree):
            if isinstance(n, ast.stmt):
                body = getattr(n, "body", None)
                hi = body[0].lineno - 1 if isinstance(body, list) and body and body[0].lineno > n.lineno else n.end_lineno
                if hi - n.lineno <= 7:
                    self.stmts.append((n.lineno, hi))
        self.assign1 = {n.lineno: n.targets[0].id for n in ast.walk(self.tree)
                        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                        and n.lineno == n.end_lineno}
        self.calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)]
        self.lists = {"default": self._default(), "swap_args": self._swap_args()}
        # 0-based lines that hold a candidate token per one-line kind, so a kind is not starved by random lines
        has = lambda f: sorted({t.start[0] - 1 for t in self.toks if t.start[0] == t.end[0] and f(t)})
        self.kind_lines = {
            "operator": has(lambda t: t.type == tokenize.OP),
            "logic": has(lambda t: t.type == tokenize.NAME and t.string in ("and", "or", "not", "is")),
            "constant": has(lambda t: t.type == tokenize.NAME and t.string in CONST_SWAP),
            "number": has(lambda t: t.type == tokenize.NUMBER and t.string.isdigit()),
            "string": has(lambda t: t.type == tokenize.STRING),
            "insert_guard": sorted(ln - 1 for ln in self.assign1)}
        self.multi = [x for x in (self._rename_local(), self._call_arg()) if x]   # one list per kind, picked 50/50
        self.block_lines = block_lines(self.lines)

    def stmt_span(self, ln):
        """(first, last) 1-based lines of the smallest statement (or compound header) holding line ln"""
        best = None
        for lo, hi in self.stmts:
            if lo <= ln <= hi and (best is None or hi - lo < best[1] - best[0]):
                best = (lo, hi)
        return best or (ln, ln)

    def seg(self, n):
        return ast.get_source_segment(self.content, n)

    def scope(self, ln):
        """(qualname, first line, last line, node) of the innermost def/class holding line ln; module = (None, 1, n, None)"""
        best = None
        for s in self.scopes:
            if s[1] <= ln <= s[2] and (best is None or s[1] >= best[1]):
                best = s
        return best if best else (None, 1, len(self.lines), None)

    def first(self, t, lo, hi):
        """is token t the first token with its type and text in lines lo..hi?"""
        return next(x for x in self.toks if lo <= x.start[0] <= hi and x.type == t.type and x.string == t.string) is t

    def text_first(self, t, lo):
        """does t's text not occur in the scope's text before t (docstrings, comments and strings included)? A reader
        looking for "the first `or`" must not meet an earlier "or" in prose, nor `<` inside an earlier `<=`"""
        li = t.start[0]
        before = "\n".join(self.lines[lo - 1:li - 1] + [self.lines[li - 1][:t.start[1]]])
        if t.type == tokenize.NAME:
            pat = rf"\b{re.escape(t.string)}\b"
        else:                                     # numbers too: "the first `1`" must not meet an earlier `10`
            pat = re.escape(t.string)
        return not re.search(pat, before)

    def where(self, q):
        return f"in `{q}`" if q else "at module level"

    def site(self, kind, l0, l1, new_lines, q, lo, desc):
        return {"kind": kind, "l0": l0, "l1": l1, "new_lines": new_lines, "scope": q, "scope_line": lo if q else None,
                "desc": desc}

    def first_call(self, c, lo, hi):
        """is call c the first call to its callee in lines lo..hi? → the callee text, or None"""
        callee = self.seg(c.func)
        if not callee or "\n" in callee or len(callee) > 40:
            return None
        same = [x for x in self.calls if lo <= x.lineno <= hi and self.seg(x.func) == callee]
        return callee if min(same, key=lambda x: (x.lineno, x.col_offset)) is c else None

    def line_site(self, i, rng, kind):
        """a one-line change of the given kind at 0-based line i"""
        li, line = i + 1, self.lines[i]
        q, lo, hi, node = self.scope(li)
        if q and self.qcount[q] != 1:
            return None
        toks = [t for t in self.toks if t.start[0] == li == t.end[0] and self.first(t, lo, hi) and self.text_first(t, lo)]
        if kind != "insert_guard" and li in self.header:
            return None
        w = self.where(q)
        if kind == "insert_guard":
            x = self.assign1.get(li)
            if not x or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not plain_name(x, 3):
                return None
            if not any(t.type == tokenize.NAME and t.string == x and t.start[1] < line.find("=") for t in toks):
                return None                       # not the first assignment to x in this scope
            lines_t, said = rng.choice(GUARDS)
            ind = line[:len(line) - len(line.lstrip())]
            new = [ind + l.format(x=x, u=self.unit) for l in lines_t]
            return self.site(kind, i, i + 1, [line] + new, q, lo,
                             f"{w}, right after the first assignment to `{x}`, add {said.format(x=x)}")
        if kind == "operator":
            opts = [(t, OP_SWAP[t.string]) for t in toks if t.type == tokenize.OP]
        elif kind == "logic":
            opts = [(t, LOGIC[t.string]) for t in toks if t.type == tokenize.NAME and t.string in LOGIC]
            opts += [(t, "") for t in toks if t.type == tokenize.NAME and t.string == "not"]
            nxt = lambda t: line[t.end[1]:].lstrip()
            opts += [(t, "is not") for t in toks if t.type == tokenize.NAME and t.string == "is" and not nxt(t).startswith("not")]
        elif kind == "constant":
            opts = [(t, CONST_SWAP[t.string]) for t in toks if t.type == tokenize.NAME and t.string in CONST_SWAP]
        elif kind == "number":
            opts = [(t, str(int(t.string) + 1)) for t in toks if t.type == tokenize.NUMBER and t.string.isdigit()]
        elif kind == "string":
            opts = [(t, flip_case(t.string)) for t in toks
                    if t.type == tokenize.STRING and re.fullmatch(r"(['\"])[A-Za-z][^'\"\\`]{2,40}\1", t.string)]   # no backticks: the task quotes in them
        else:
            return None
        if not opts:
            return None
        t, nt = rng.choice(opts)
        if nt == "":                              # delete the first `not` (and the space after it)
            new = line[:t.start[1]] + line[t.end[1]:].lstrip(" ")
            desc = f"{w}, delete the first `not`"
        else:
            new = line[:t.start[1]] + nt + line[t.end[1]:]
            desc = f"{w}, change the first `{t.string}` to `{nt}`"
        return self.site(kind, i, i + 1, [new], q, lo, desc)

    def _default(self):
        """a parameter default that is a constant → another constant ("change the default of `p` from 1 to 2")"""
        out = []
        for q, s, e, node in self.scopes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or self.qcount[q] != 1:
                continue
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, d in pairs:
                if not isinstance(d, ast.Constant) or d.lineno != d.end_lineno:
                    continue
                old = self.seg(d)
                new = const_swap(d, old)
                if not new:
                    continue
                line = self.lines[d.lineno - 1]
                nl = line[:bcol(line, d.col_offset)] + new + line[bcol(line, d.end_col_offset):]
                out.append(self.site("default", d.lineno - 1, d.lineno, [nl], q, s,
                                     f"in `{q}`, change the default value of the parameter `{arg.arg}` from `{old}` to `{new}`"))
        return out

    def _swap_args(self):
        """the first call to a callee in its scope, on one line, with two different simple leading positional arguments
        → swap them"""
        out = []
        for c in self.calls:
            if c.lineno != c.end_lineno or len(c.args) < 2 or any(isinstance(x, ast.Starred) for x in c.args[:2]):
                continue
            a0, a1 = c.args[:2]
            if not all(isinstance(x, (ast.Name, ast.Attribute, ast.Constant)) for x in (a0, a1)):
                continue
            s0, s1 = self.seg(a0), self.seg(a1)
            if not s0 or not s1 or s0 == s1:
                continue
            q, lo, hi, _ = self.scope(c.lineno)
            if q and self.qcount[q] != 1:
                continue
            callee = self.first_call(c, lo, hi)
            if not callee:
                continue
            line = self.lines[c.lineno - 1]
            p0, e0 = bcol(line, a0.col_offset), bcol(line, a0.end_col_offset)
            p1, e1 = bcol(line, a1.col_offset), bcol(line, a1.end_col_offset)
            nl = line[:p0] + s1 + line[e0:p1] + s0 + line[e1:]
            out.append(self.site("swap_args", c.lineno - 1, c.lineno, [nl], q, lo,
                                 f"{self.where(q)}, in the first call to `{callee}(…)`, swap the first two positional "
                                 f"arguments (`{s0}` and `{s1}`)"))
        return out

    def _rename_local(self):
        """a local variable of a function whose every use (2-6 lines apart) is an ast.Name → rename all of them to a
        name the file does not use, so the code keeps working"""
        out = []
        for q, s, e, node in self.scopes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or self.qcount[q] != 1:
                continue
            args = {x.arg for x in ast.walk(node.args) if isinstance(x, ast.arg)}
            stores = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            free = [v for v in VOCAB if v not in self.words]
            for v in sorted(stores - args):
                if not plain_name(v, 3) or not free:
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
                w = free[int(hashlib.md5((q + v).encode()).hexdigest(), 16) % len(free)]   # stable per build
                new = self.lines[l0:l1]
                for t in sorted(toks, key=lambda t: t.start, reverse=True):
                    k = t.start[0] - 1 - l0
                    new[k] = new[k][:t.start[1]] + w + new[k][t.end[1]:]
                out.append(self.site("rename_local", l0, l1, new, q, s,
                                     f"in `{q}`, rename the local variable `{v}` to `{w}` everywhere in that function"))
        return out

    def _call_arg(self):
        """the first call to a callee in its scope, spanning 2-6 lines, with a constant keyword argument → change it;
        the answer covers the whole call"""
        out = []
        for c in self.calls:
            if not 1 <= c.end_lineno - c.lineno <= 5:
                continue
            q, lo, hi, _ = self.scope(c.lineno)
            if q and self.qcount[q] != 1:
                continue
            callee = self.first_call(c, lo, hi)
            if not callee:
                continue
            for kw in c.keywords:
                val = kw.value
                if not kw.arg or not isinstance(val, ast.Constant) or val.lineno != val.end_lineno:
                    continue
                old = self.seg(val)
                new = const_swap(val, old)
                if not new:
                    continue
                l0, l1 = c.lineno - 1, c.end_lineno
                new_lines = self.lines[l0:l1]
                k, line = val.lineno - 1 - l0, self.lines[val.lineno - 1]
                new_lines[k] = line[:bcol(line, val.col_offset)] + new + line[bcol(line, val.end_col_offset):]
                out.append(self.site("call_arg", l0, l1, new_lines, q, lo,
                                     f"{self.where(q)}, in the first call to `{callee}(…)`, change the argument "
                                     f"`{kw.arg}={old}` to `{kw.arg}={new}`"))
        return out


MISREMEMBER_OPS = (["name_variant", "punct", "word_drop", "arg_drop", "truncate", "other_name"], [30, 25, 12, 13, 5, 15])


def token_change(line, rng, words=()):
    """one way an LLM misremembers a line it read (E4B's real misquotes, all-129): a name variant (max_redirects →
    max_redirect_count, current_x → x) · punctuation ([:2] → [0:2], a comma or closing bracket more or less, quotes) · a
    few words dropped from a comment / string · an argument dropped from a call · the line cut short · another name of
    the file (words) → the line or None (never a character-level typo: LLMs do not make those)"""
    w = line.split()
    text = len(w) >= 4 and not re.search(r"[=()\[\]{}]", line) and not line.rstrip().endswith(":") \
        and w[0] not in keyword.kwlist                       # docstring / prose line: words, not names
    for _ in range(6):
        op = rng.choices(*MISREMEMBER_OPS)[0]
        if text and op in ("name_variant", "arg_drop", "other_name"):
            op = "word_drop"
        out = None
        prose = [m.span() for m in re.finditer(r"#.*$|\"[^\"]*\"|'[^']*'", line)]   # strings and comments: no names there
        ids = [m for m in re.finditer(r"\b[A-Za-z_]\w{2,}\b", line) if not keyword.iskeyword(m.group(0))
               and not m.group(0).startswith("__") and not any(a0 <= m.start() < b0 for a0, b0 in prose)]
        if op == "name_variant" and ids:
            m = rng.choice(ids)
            g = m.group(0)
            parts = g.split("_")
            if len([x for x in parts if x]) >= 2 and rng.random() < 0.5:
                keep = [x for x in parts if x]
                v = "_".join(keep[1:] if rng.random() < 0.5 else keep[:-1])
                if not v.isidentifier() or keyword.iskeyword(v):
                    continue
            else:
                v = g + rng.choice(["_count", "_name", "_value", "s", "_list", "_id", "_type"])
            out = line[:m.start()] + v + line[m.end():]
        elif op == "punct":
            st = line.rstrip()
            cands = []
            if re.search(r"\[:\d", line):
                cands.append(re.sub(r"\[:(\d)", r"[0:\1", line, count=1))
            if st.endswith(","):
                cands.append(st[:-1])
            elif st and st[-1] not in ":,([{\\":
                cands.append(st + ",")
            if st.endswith((")", "]", "}")) and len(st) > 1:
                cands.append(st[:-1])
            if "'" in line and '"' not in line:
                cands.append(line.replace("'", '"'))
            elif '"' in line and "'" not in line:
                cands.append(line.replace('"', "'"))
            out = rng.choice(cands) if cands else None
        elif op == "word_drop":
            m = re.search(r"(^\s*|#\s*|\"|')([^\"'#]*\s[^\"'#]*\s[^\"'#]*\s[^\"'#]+)", line) if text else \
                re.search(r"(#\s*|\"|')([^\"'#]*\s[^\"'#]*\s[^\"'#]*\s[^\"'#]+)", line)
            if m:
                ws = m.group(2).split(" ")
                k = rng.randrange(len(ws))
                n = rng.randint(1, min(3, len(ws) - 2))
                out = line[:m.start(2)] + " ".join(ws[:k] + ws[k + n:]) + line[m.end(2):]
        elif op == "arg_drop":
            m = re.search(r"\(([^()]*,[^()]*)\)", line)
            if m:
                args = m.group(1).split(",")
                k = rng.randrange(len(args))
                out = line[:m.start(1)] + ",".join(args[:k] + args[k + 1:]).strip() + line[m.end(1):]
        elif op == "truncate" and len(line.strip()) >= 40:
            out = line[:int(len(line) * rng.uniform(0.5, 0.8))].rstrip()
        elif op == "other_name" and ids and words:
            m = rng.choice(ids)
            out = line[:m.start()] + rng.choice(words) + line[m.end():]
        if out is not None and out != line and out.strip():
            return out
    return None


def corrupt(kind, old, new, core, core_new, content, lines, a, b, rng):
    """one E4B-style failed edit of the given kind → (bad old_string, its new_string) or None. Kept only if the full
    matcher really fails on it (not found), or for short_nonunique really reports several matches. short_nonunique
    is the changed lines alone (`core`, which then must occur more than once) with their own new text, the way E4B
    anchors too short; the others corrupt a 1-5-line span around the site (E4B's own attempts: median 4 lines) and keep
    the full new_string."""
    if kind == "short_nonunique":
        return (core, core_new) if core.strip() and n_matches(content, core) > 1 else None
    a, b = max(0, a - rng.randint(0, 2)), min(len(lines), b + rng.randint(0, 2))
    old = "\n".join(lines[a:b])
    for _ in range(12):
        ol, bad, bad_new = old.split("\n"), None, new
        if kind == "token_change":
            j = rng.randrange(len(ol))
            alt = token_change(ol[j], rng)
            if alt:
                ol[j] = alt
                bad = "\n".join(ol)
        elif kind == "quote_wrap":
            bad = '"' + old + '"'
        elif kind == "dropped_line" and len(ol) >= 3:
            j = rng.randrange(1, len(ol) - 1)
            bad = "\n".join(ol[:j] + ol[j + 1:])
        elif kind == "reformat" and len(ol) >= 2:
            bad = ol[0].rstrip() + " " + " ".join(x.strip() for x in ol[1:])
        elif kind == "trailing_backslash":
            bad = old.rstrip("\n") + "\\"
        elif kind == "json_wrapper":
            bad = old + '", "start_line": %d, "end_line": %d, "total_lines": %d, "is_truncated": false}' % (a + 1, b, len(lines))
        if bad and bad != old and n_matches(content, bad) == 0:
            return bad, bad_new
    return None


def misquote(line, rng, words):
    """the line as E4B misremembers it, at a strength drawn from MISQUOTE → the altered line or None"""
    lvl = rng.choices(*MISQUOTE)[0]

    out = line
    for _ in range({"small": 1, "medium": 2, "large": rng.randint(3, 4)}[lvl]):
        out = token_change(out, rng, words) or out
    return out if out != line and out.strip() else None


def retry_fail(kind, content, lines, site, a, b, vs, ve, rng):
    """a retry drill's failed edit, E4B-style → (old, new) or None: a span of FAILED_LINES lines around the answer
    window (inside the view [vs, ve), 0-based) with the change applied in new, then misquote_both (an unchanged line
    misremembered, the same in old and new) · token_change (a line misremembered in old only) · invented (every line
    misremembered) · dropped_line (a line skipped in both); ESCAPED of them then written with literal \\n / \\" escapes (the
    whole span or a few lines). Kept only if the harness's matcher cannot find it"""
    want = rng.choices(*FAILED_LINES)[0]
    want = {8: (7, 8), 12: (9, 12), 20: (13, 20), 40: (21, 40)}.get(want, (want, want))
    want = rng.randint(*want)
    a2, b2, down = a, b, rng.random() < 0.5
    while b2 - a2 < want and (a2 > vs or b2 < ve):
        if (down and b2 < ve) or a2 <= vs:
            b2 += 1
        else:
            a2 -= 1
        down = not down
    old_l = lines[a2:b2]
    new_l = lines[a2:site["l0"]] + site["new_lines"] + lines[site["l1"]:b2]
    c0, c1, off = site["l0"] - a2, site["l1"] - a2, len(site["new_lines"]) - (site["l1"] - site["l0"])
    nidx = lambda j: j if j < c0 else j + off             # an unchanged line's index in new
    unchanged = [j for j in range(len(old_l)) if old_l[j].strip() and not c0 <= j < c1]
    words = sorted({w for w in re.findall(r"\b[A-Za-z_]\w{3,}\b", content) if not keyword.iskeyword(w)})
    for _ in range(12):
        ol, nl = list(old_l), list(new_l)
        if kind in ("misquote_both", "token_change"):
            js = unchanged if kind == "misquote_both" else [j for j in range(len(ol)) if ol[j].strip()]
            if not js:
                return None
            j = rng.choice(js)
            alt = misquote(ol[j], rng, words)
            if not alt:
                continue
            ol[j] = alt
            if kind == "misquote_both":
                nl[nidx(j)] = alt
        elif kind == "invented":
            for j in range(len(ol)):
                if ol[j].strip():
                    alt = misquote(ol[j], rng, words) or token_change(ol[j], rng, words)
                    if alt:
                        ol[j] = alt
                        if not c0 <= j < c1:
                            nl[nidx(j)] = alt
        elif kind == "dropped_line":
            if len(unchanged) < 1 or len(ol) < 3:
                return None
            inner = [j for j in unchanged if 0 < j < len(ol) - 1]
            if not inner:
                return None
            j = rng.choice(inner)
            del nl[nidx(j)]
            del ol[j]
        if rng.random() < ESCAPED[0] and len(ol) >= 2 and len(nl) >= 2:   # the look: literal \n / \" escapes
            esc = lambda xs: [json.dumps("\n".join(xs))[1:-1]]
            if rng.random() < ESCAPED[1]:
                ol, nl = esc(ol), esc(nl)
            else:                                 # a few lines joined; in new the same lines where they are unchanged
                i = rng.randrange(len(ol) - 1)
                k = min(len(ol), i + rng.randint(2, 4))
                ni, nk = nidx(i), nidx(k - 1) + 1
                ol = ol[:i] + esc(ol[i:k]) + ol[k:]
                if 0 <= ni < nk <= len(nl):
                    nl = nl[:ni] + esc(nl[ni:nk]) + nl[nk:]
        bad, bad_new = "\n".join(ol), "\n".join(nl)
        if bad != "\n".join(old_l) and bad.strip() and n_matches(content, bad) == 0:
            return bad, bad_new
    return None


# ------------------------------------------------- building
class Builder:
    def __init__(self, a):
        self.a, self.rng, self.n = a, random.Random(a.seed), 0
        self.src, self.why = {}, None          # per-file analysis cache · why the last exercise was rejected
        self.lib_words = {}                    # lib → Counter of identifiers in its files (set by build)
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

    def analyse(self, rel, content):
        if rel not in self.src:
            try:
                self.src[rel] = Src(content)
            except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
                self.src[rel] = None
        return self.src[rel]

    def pick(self, src, content, lines, cand, multi, kind, repeat, near=None, cap=8, min_span=1):
        """a site and its answer window → (site, a, b, old, new, intended) or None. repeat: the changed lines must
        occur more than once (anchor drills, short_nonunique recover drills); near: a 0-based line the answer window
        must also cover, sites only within ±3 lines of it (stale drills: the line the earlier edit changed); cap / min_span:
        the shortest unique span may grow cap steps and must be ≥ min_span lines (block drills: BLOCK_CAP, BLOCK_LINES),
        and list kinds then only take sites on a cand line"""
        from swegemma.edit import apply_replacement
        rng = self.rng
        for _ in range(30):
            if repeat or near is not None:        # few sites fit for any one kind: redraw the kind per try
                kind = rng.choices(*LINE_KINDS)[0]
            if multi:
                site = rng.choice(rng.choice(src.multi))
            elif kind in src.lists:
                pool = [x for x in src.lists[kind] if (near is None or 0 < abs(x["l0"] - near) <= 3)
                        and (min_span == 1 or x["l0"] in cand)]
                site = rng.choice(pool) if pool else None
            else:
                ok = set(cand) & set(src.kind_lines.get(kind, cand))
                pool = sorted(i for i in ok if near is None or 0 < abs(i - near) <= 3)
                site = src.line_site(rng.choice(pool), rng, kind) if pool else None
            if not site:
                continue
            core = "\n".join(lines[site["l0"]:site["l1"]])
            single = n_matches(content, core) == 1 and content.count(core) == 1
            if repeat and single:
                continue
            if not repeat and near is None and not single and rng.random() < 0.7:
                continue
            l0, l1 = site["l0"], site["l1"]
            if near is not None:
                l0, l1 = min(l0, near), max(l1, near + 1)
            w = minimal_unique(lines, l0, l1, content, cap)
            if not w or w[1] - w[0] < min_span:
                continue
            a, b = self.grow(src, lines, w, l0, l1)
            new_lines = lines[a:site["l0"]] + site["new_lines"] + lines[site["l1"]:b]
            old, new = "\n".join(lines[a:b]), "\n".join(new_lines)
            intended = "\n".join(lines[:a] + new_lines + lines[b:])
            r = apply_replacement(content, old, new)
            if r.strategy == "exact" and r.occurrences == 1 and r.new_content == intended:
                return site, a, b, old, new, intended
        return None

    def grow(self, src, lines, w, l0, l1):
        """the unique span w grown to whole statements, then up and down to a length drawn from ANCHOR_LINES (any
        span containing a unique span is unique); blank lines at the grown ends are dropped"""
        a0, b0 = w
        want = self.rng.choices(*ANCHOR_LINES)[0]
        if want == 1:                             # the shortest unique span as it is
            return a0, b0
        s0, s1 = src.stmt_span(l0 + 1), src.stmt_span(l1)
        a, b = min(a0, s0[0] - 1), max(b0, s1[1])
        down = self.rng.random() < 0.5
        while b - a < want and (a > 0 or b < len(lines)):
            if (down and b < len(lines)) or a == 0:
                b += 1
            else:
                a -= 1
            down = not down
        while b - 1 >= b0 and b - a > 1 and not lines[b - 1].strip():
            b -= 1
        while a < a0 and b - a > 1 and not lines[a].strip():
            a += 1
        return a, b

    def fillers(self, msgs, root, others, lib):
        """the tool outputs between the view and the edit: `grep -rn <name>` through the harness's run_command on the
        drill workspace (GREP_FILLER of them) or a read_file window of another file → counts by kind"""
        from swegemma.tools import workspace as ws
        from swegemma.tools.execution import run_command
        rng, got = self.rng, Counter()
        for orel, oc in others:
            if rng.random() < GREP_FILLER:
                words = self.lib_words.get(lib, Counter())
                names = sorted({t for t in re.findall(r"\b[a-z_][a-z0-9_]{4,}\b", oc) if not keyword.iskeyword(t)})
                if names:
                    empty = rng.random() < EMPTY_GREP
                    if empty:                     # a one-letter typo of a real name: grep finds nothing
                        nm = rng.choice(names)
                        j = rng.randrange(1, len(nm))
                        nm = nm[:j] + ("x" if nm[j] != "x" else "q") + nm[j + 1:]
                    else:
                        rare = [n for n in names if words[n] <= RARE_NAME] if rng.random() < RARE_SHARE else []
                        nm = rng.choice(rare or names)
                    cmd = f'grep -rn "{nm}" {orel.split("/")[0]}/ --include="*.py"'
                    res = run_command(_Ctx(root, run=True), cmd)
                    r = json.loads(res)
                    none = r.get("status") == "error" and (r.get("details") or {}).get("exit_code") == 1 \
                        and not (r.get("details") or {}).get("stdout")
                    if (r.get("status") == "ok" and not empty) or (empty and none):
                        cid, am = self.call("run_command", {"command": cmd})
                        msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(res)}]
                        got["grep_empty" if empty else "grep"] += 1
                        continue
            ol = oc.split("\n")
            ds = rng.randint(1, max(1, len(ol) - 60))
            de = ds + rng.randint(30, 100)
            cid, am = self.call("read_file", {"filepath": orel, "start_line": ds, "end_line": de})
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(_Ctx(root), orel, ds, de))}]
            got["read"] += 1
        return got

    def exercise(self, lib, rel, content, root, others, family):
        from swegemma.tools import workspace as ws
        rng, lines = self.rng, content.split("\n")
        self.why = "no_site"
        src, cand = self.analyse(rel, content), code_lines(content)
        if not src or not cand:
            return None
        ckind = rng.choices(*CORRUPTIONS)[0] if family == "recover" else \
            rng.choices(*RETRY_ERRORS)[0] if family == "retry" else None
        if family == "block":
            ckind = "block_nonunique" if rng.random() < BLOCK_FAIL else None
            cand = sorted(set(cand) & set(src.block_lines))
            if not cand:
                self.why = "no_block"
                return None
        repeat = family in ("anchor", "block") or ckind == "short_nonunique"
        pre = None                                # stale: (old, new, site) of the earlier edit, applied first
        if family == "stale" or ckind == "stale_text":   # retry stale_text: the stale set-up, then the failed copy
            k1 = rng.choice(["operator", "logic", "constant", "number", "string"])
            p1 = self.pick(src, content, lines, cand, False, k1, False)
            if not p1 or p1[0]["l1"] - p1[0]["l0"] != 1 or len(p1[0]["new_lines"]) != 1:
                return None
            s1, _, _, old1, new1, content1 = p1
            try:
                src1 = Src(content1)
            except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
                return None
            lines1 = content1.split("\n")
            p2 = self.pick(src1, content1, lines1, code_lines(content1), False, None, False, near=s1["l0"])
            if not p2:
                return None
            site, a, b, old, new, intended = p2
            if n_matches(content1, "\n".join(lines[a:b])):
                self.why = "stale_match"          # copying the first view's text must fail
                return None
            pre = (old1, new1, s1)
            instr = f"In `{rel}`, make two changes: first, {s1['desc']}; then, {site['desc']}."
        else:
            multi = family != "anchor" and not repeat and src.multi and rng.random() < self.a.multi
            kind = None if multi else rng.choices(*LINE_KINDS)[0]
            blk = dict(cap=BLOCK_CAP, min_span=BLOCK_LINES) if family == "block" else {}
            p = self.pick(src, content, lines, cand, multi, kind, repeat, **blk)
            if not p:
                return None
            site, a, b, old, new, intended = p
            instr = f"In `{rel}`, {site['desc']}."
        i = site["l0"]
        try:                                      # the answer must leave valid Python (no `return_new x`)
            ast.parse(intended)
        except SyntaxError:
            self.why = "syntax"
            return None
        olds = [old] + ([pre[0]] if pre else [])
        if any(len(x.strip()) >= 6 and x.strip() in instr for o in olds for x in o.split("\n")):
            self.why = "leak"                     # a whole old line quoted in the task: copyable without reading
            return None
        ctx = _Ctx(root)
        need_a, need_b = (min(a, pre[2]["l0"]), max(b, pre[2]["l1"])) if pre else (a, b)
        scope_lines = [x for x in (site["scope_line"], pre[2]["scope_line"] if pre else None) if x]
        s0 = max(1, i + 1 - rng.randint(5, 40))
        e0 = min(len(lines), s0 + rng.randint(30, 120))
        s0 = min([s0, need_a + 1] + scope_lines)  # the view shows the def/class lines that name the scopes
        e0 = max(e0, need_b)
        if e0 - s0 + 1 > 150:                     # harness view limit
            e0 = need_b + min(10, len(lines) - need_b)
            if e0 - s0 + 1 > 150:
                self.why = "window"
                return None
        core = "\n".join(lines[site["l0"]:site["l1"]])
        newset = {x.strip() for x in site["new_lines"]}   # the first line the change touches (call_arg: inside the call)
        probe = next((x.strip() for x in core.split("\n") if x.strip() and x.strip() not in newset),
                     next((x.strip() for x in core.split("\n") if x.strip()), ""))
        dup_at = [k for k, x in enumerate(lines) if probe and probe in x and not site["l0"] <= k < site["l1"]]
        single = n_matches(content, core) == 1 and content.count(core) == 1
        if single and not dup_at:
            dup_seen = "none"                     # the changed lines are unique, even stripped: nothing to show
        elif not dup_at:
            self.why = "dup_probe"                # repeats only as a block across lines: no one-line probe shows it
            return None
        elif any(s0 <= k + 1 <= e0 for k in dup_at):
            dup_seen = "in_view"
        else:                                     # make another copy visible (widen the view, else grep)
            near = min(dup_at, key=lambda k: abs(k - i))
            ws0, we0 = min(s0, near + 1), max(e0, near + 1)
            dup_seen = "view" if rng.random() < DUP_VIEW and we0 - ws0 + 1 <= 150 else "grep"
            if dup_seen == "view":
                s0, e0 = ws0, we0
        view = json.loads(ws.read_file(ctx, rel, s0, e0))
        shown = view.get("content") or ""
        if view.get("is_truncated") or "\n".join(lines[a:b]) not in shown or any(lines[x - 1] not in shown for x in scope_lines):
            self.why = "window"
            return None
        system = self.system.replace("{problem_description}", instr)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": self.user(lib, instr, root)}]
        cid, am = self.call("read_file", {"filepath": rel, "start_line": s0, "end_line": e0})
        msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, rel, s0, e0))}]
        if dup_seen == "grep":                    # after the view: every copy of the first changed line (a uniqueness check)
            from swegemma.tools.execution import run_command
            cmd = f"grep -nF -e {shlex.quote(probe)} {rel}"
            res = run_command(_Ctx(root, run=True), cmd)
            r = json.loads(res)
            if r.get("status") != "ok" or len((r.get("stdout") or "").strip().split("\n")) < 2:
                self.why = "dup_grep"
                return None
            cid, am = self.call("run_command", {"command": cmd})
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(res)}]
        if pre:                                   # the first change, done: its diff is the only view of the new text
            cid, am = self.call("edit_file", {"filepath": rel, "new_string": pre[1], "old_string": pre[0]})
            res = ws.edit_file(ctx, rel, pre[0], pre[1])
            if json.loads(res).get("status") != "ok":
                (Path(root) / rel).write_text(content)
                self.why = "apply"
                return None
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(res)}]
        got = self.fillers(msgs, root, others, lib)
        edit = {"id": None, "name": "edit_file",
                "args": json.dumps({"filepath": rel, "new_string": new, "old_string": old}, ensure_ascii=False)}
        tgt = lambda tc: {"reasoning": None, "content": "", "tool_calls": [tc], "finish_reason": "tool_calls"}
        one = lambda k, m, tc: {"call": k, "segment": 0, "events_before": [], "input_messages": m, "target": tgt(tc),
                                "thought_source": "none", "label_scope": "tool_call", "usage_prompt_tokens": None}
        verify = {"lib": lib, "path": rel, "file_sha256": sha(content), "intended_sha256": sha(intended)}
        if pre:
            verify["pre_edit"] = {"old": pre[0], "new": pre[1]}
        rec = {"instance_id": f"drill_{lib}_{self.a.seed}_{self.n:06d}", "repo": lib, "base_commit": None, "pr": None,
               "patch_sha": None, "source": "edit_drill", "resolved": None, "policy_model": self.model,
               "sampling": dict(self.sampling, max_tool_calls=28, max_time_minutes=7), "env_fingerprint": None,
               "family": family, "distance": len(others) + (dup_seen == "grep"), "fillers": dict(got), "site_line": i + 1, "old_lines": b - a,
               "window": [a + 1, b], "dup_seen": dup_seen,
               "mutation": site["kind"], "scope": site["scope"], "verify": verify, "tools": self.tools}
        if pre:
            rec["pre_kind"] = pre[2]["kind"]
        if ckind:
            core = "\n".join(lines[site["l0"]:site["l1"]])
            if ckind == "block_nonunique":        # the longest span that still repeats, with the change applied
                sp_ = repeated_span(lines, site["l0"], site["l1"], content)
                c = None
                if sp_ and sp_[1] - sp_[0] >= BLOCK_LINES and sp_[0] <= site["l0"] and site["l1"] <= sp_[1]:
                    c = ("\n".join(lines[sp_[0]:sp_[1]]),
                         "\n".join(lines[sp_[0]:site["l0"]] + site["new_lines"] + lines[site["l1"]:sp_[1]]))
            elif ckind == "stale_text":            # the first view's text (before the earlier edit), with the change
                c = ("\n".join(lines[a:b]), "\n".join(lines[a:site["l0"]] + site["new_lines"] + lines[site["l1"]:b]))
            elif family == "retry" and ckind != "short_nonunique":
                c = retry_fail(ckind, content, lines, site, a, b, s0 - 1, e0, rng)
            else:
                c = corrupt(ckind, old, new, core, "\n".join(site["new_lines"]), content, lines, a, b, rng)
            if not c:
                self.why = "corrupt"
                return None
            bad, bad_new = c
            cid, am = self.call("edit_file", {"filepath": rel, "new_string": bad_new, "old_string": bad})
            err = ws.edit_file(ctx, rel, bad, bad_new)
            if json.loads(err).get("status") != "error":
                (Path(root) / rel).write_text(content)
                self.why = "corrupt"
                return None
            msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(err)}]
            rs, re_ = max(1, a + 1 - 4), min(len(lines), b + 4)
            reread = {"id": None, "name": "read_file", "args": json.dumps({"filepath": rel, "start_line": rs, "end_line": re_})}
            state = rng.choices(*RETRY_STATES)[0] if family == "retry" else None
            if state in ("resent", "reread_resent"):  # the loop: (read again,) the identical call, the same error
                if state == "reread_resent":
                    cid, am = self.call("read_file", {"filepath": rel, "start_line": rs, "end_line": re_})
                    msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, rel, rs, re_))}]
                cid, am = self.call("edit_file", {"filepath": rel, "new_string": bad_new, "old_string": bad})
                msgs += [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.edit_file(ctx, rel, bad, bad_new))}]
            rec["corruption"] = ckind
            rec["failed_old"] = bad
            if state:
                rec["retry_state"] = state
            if state == "reread_resent":          # the fresh view is already in context: the correct edit
                rec["calls"] = [one(1, msgs, edit)]
            else:
                cid, am = self.call("read_file", {"filepath": rel, "start_line": rs, "end_line": re_})
                msgs2 = msgs + [am, {"role": "tool", "tool_call_id": cid, "content": tool_msg(ws.read_file(ctx, rel, rs, re_))}]
                rec["calls"] = [one(1, msgs, reread), one(2, msgs2, edit)]
        else:
            rec["calls"] = [one(1, msgs, edit)]
        # the answer through the REAL edit_file on the workspace copy (stale: after the first change) must give the
        # intended file
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
    block_libs = [x for x in a.block_libs.split(",") if x and x not in libs]
    sp = site_packages()
    print(f"edit_drills build · libs {libs} from {sp} · n {a.n} · seed {a.seed} · families {dict(zip(*FAMILIES))} · "
          f"distances {dict(zip(*DISTANCES))} · libraries equally often, files ∝ √size, code lines only · system {a.system} ({len(Path(a.system).read_text())} chars) · "
          f"tools from {a.tools_from} minus [{a.drop_tools}] · min file lines {a.min_lines} · multi-line share (copy/recover) {a.multi} · "
          f"instructions name scope + first token, never a whole line · one-line kinds by weight {dict(zip(*LINE_KINDS))} · "
          f"renames to an unused name from a {len(VOCAB)}-word list · recover corruptions by weight {dict(zip(*CORRUPTIONS))} · "
          f"fillers: grep -rn share {GREP_FILLER}, else read_file · stale = an earlier edit in context changes a line the answer must cover · "
          f"answer = shortest unique span grown to whole statements and to {dict(zip(*ANCHOR_LINES))} lines (31B's edits) · "
          f"duplicate sites (file, line, kind) and overlapping answer windows rejected · ≤ {a.file_cap} drills per file · "
          f"libraries kept level (one with 60 misses in a row counts as used up) · greps: {RARE_SHARE} a name used ≤ {RARE_NAME}× "
          f"in the library, else any, {EMPTY_GREP} of them a typo that finds "
          f"nothing · repeated changed lines: another copy always in context (view widened to it {DUP_VIEW}, else "
          f"grep -nF right after the view) · block family from {block_libs or 'nothing (no --block-libs): off'}: sites in a "
          f"repeated {BLOCK_LINES}-line block, shortest unique span {BLOCK_LINES}-{BLOCK_CAP} lines, {BLOCK_FAIL} first fail "
          f"on the longest repeating span · retry errors by weight {dict(zip(*RETRY_ERRORS))} · retry loop states "
          f"{dict(zip(*RETRY_STATES))} · retry failed old_string lines by weight {dict(zip(*FAILED_LINES))} (7-8 / 9-12 / 13-20 / "
          f"21-40 inside the last four) · misquote strength {dict(zip(*MISQUOTE))} · escaped look {ESCAPED[0]} (whole span "
          f"{ESCAPED[1]:.2f}) · out {a.out}")
    b = Builder(a)
    print(f"  tools {b.tool_names} · sampling {b.sampling} · policy model {b.model} · family weights "
          f"{a.families or dict(zip(*FAMILIES))}")
    files = {lib: lib_files(lib, a.min_lines) for lib in libs + block_libs}
    for lib in block_libs:                       # block libraries: only files with a repeated block
        files[lib] = [(r, c) for r, c in files[lib] if block_lines(c.split("\n"))]
    for lib, fs in files.items():
        print(f"  {lib}: {len(fs)} files ≥ {a.min_lines} lines")
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    st, span, seen = Counter(), Counter(), set()
    with tempfile.TemporaryDirectory(prefix="drills_") as tmp:
        roots = {}
        for lib, fs in files.items():
            roots[lib] = Path(tmp) / lib
            for rel, c in fs:
                (roots[lib] / rel).parent.mkdir(parents=True, exist_ok=True)
                (roots[lib] / rel).write_text(c)
        libs_ok = [lib for lib in libs if files[lib]]
        block_ok = [lib for lib in block_libs if files[lib]]
        weights = {lib: [len(c) ** 0.5 for _, c in files[lib]] for lib in libs_ok}   # bigger files more often
        weights.update({lib: [len(block_lines(c.split("\n"))) ** 0.5 for _, c in files[lib]] for lib in block_ok})
        b.lib_words = {lib: Counter(re.findall(r"\b[a-z_][a-z0-9_]{4,}\b", "\n".join(c for _, c in files[lib])))
                       for lib in libs_ok + block_ok}
        per_file, per_lib, wins, misses = Counter(), Counter(), {}, Counter()
        fams = dict(zip(*FAMILIES))
        if a.families:                            # e.g. block=1: a held-out slice of one family
            fams = {k: float(v) for k, v in (x.split("=") for x in a.families.split(","))}
            assert set(fams) <= set(FAMILIES[0]), fams
        if not block_ok and "block" in fams:
            fams.pop("block")
        while sum(v for k, v in st.items() if k.startswith("ok_")) < a.n and st["tried"] < a.n * 12:
            fam = b.rng.choices(list(fams), list(fams.values()))[0]
            group = block_ok if fam == "block" else libs_ok        # block drills: their own libraries
            avail = {lib: [k for k, (r2, _) in enumerate(files[lib]) if per_file[r2] < a.file_cap] for lib in group}
            libs_av = [lib for lib in group if avail[lib] and misses[lib] < 60]   # 60 misses in a row: used up
            if not libs_av:
                if fam == "block":
                    print(f"  block libraries used up after {st['ok_block']} block drills; the rest without them")
                    fams.pop("block")
                    if not fams:
                        print("  no family left; stopping early")
                        break
                    continue
                print(f"  every library is used up (--file-cap {a.file_cap}, 60 misses in a row); stopping early")
                break
            low = min(per_lib[lib] for lib in libs_av)            # the library with the fewest drills so far
            lib = b.rng.choice([x for x in libs_av if per_lib[x] == low])
            k = b.rng.choices(avail[lib], [weights[lib][j] for j in avail[lib]])[0]
            rel, c = files[lib][k]
            dist = b.rng.choices(*DISTANCES)[0]
            others = [(r2, c2) for r2, c2 in files[lib] if r2 != rel]
            dis = [b.rng.choice(others) for _ in range(dist)] if others else []
            st["tried"] += 1
            rec = b.exercise(lib, rel, c, roots[lib], dis, fam)
            misses[lib] += 1
            if rec is None:
                st[f"rejected_{fam}:{b.why}"] += 1
                continue
            key = (rec["verify"]["path"], rec["site_line"], rec["mutation"])
            if key in seen:
                st[f"rejected_{fam}:dup"] += 1
                continue
            w0, w1 = rec["window"]
            if any(w0 <= y and x <= w1 for x, y in wins.get(rel, [])):
                st[f"rejected_{fam}:overlap"] += 1            # the same lines as an earlier drill of this file
                continue
            seen.add(key)
            wins.setdefault(rel, []).append((w0, w1))
            misses[lib] = 0
            per_file[rel] += 1
            per_lib[lib] += 1
            for k, v in rec["fillers"].items():
                st[f"filler_{k}"] += v
            st[f"ok_{fam}"] += 1
            st[f"kind_{rec['mutation']}"] += 1
            st[f"dup_seen_{rec['dup_seen']}"] += 1
            st[f"lib_{lib}"] += 1
            span[min(rec["old_lines"], 6)] += 1
            if rec.get("corruption"):
                st[f"corrupt_{rec['corruption']}"] += 1
            if rec.get("retry_state"):
                st[f"retry_{rec['retry_state']}"] += 1
            json.dump(rec, open(out / f"{rec['instance_id']}.json", "w"), ensure_ascii=False)
    print(json.dumps({**dict(sorted(st.items())), "old_lines (answer lines; 6 = 6+)": dict(sorted(span.items()))}))
    ok_all = sum(span.values()) or 1
    print(f"  answer old_string lines: 1 line {100 * span[1] / ok_all:.0f} % · 2 {100 * span[2] / ok_all:.0f} % · 3-5 "
          f"{100 * (span[3] + span[4] + span[5]) / ok_all:.0f} % · 6+ {100 * span[6] / ok_all:.0f} %   (31B's successful edits: "
          f"1 line 33 % · 2 4 % · 3-5 32 % · 6+ 31 %)")
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


def drill_file(rec, cache):
    """the file the answer applies to (stale drills: after the earlier edit in the context), checked against the sha"""
    from swegemma.edit import apply_replacement
    v = rec["verify"]
    if v["path"] not in cache:
        content = (site_packages() / v["path"]).read_text(encoding="utf-8")
        if sha(content) != v["file_sha256"]:
            sys.exit(f"{v['path']}: file differs from the one the drills were built from (sha256) — another venv?")
        cache[v["path"]] = content
    content = cache[v["path"]]
    if v.get("pre_edit"):
        content = apply_replacement(content, v["pre_edit"]["old"], v["pre_edit"]["new"]).new_content
    return content


def call_outcome(rec, call, m, cache):
    """one decoded reply against its target → (outcome, the model's edit args or None). Outcomes: intended,
    applied_other_change, several_matches, not_found, wrong_file, bad_args, other_tool:<name>, same_tool (a read_file
    target answered by read_file of the right file), no_call, cut_off"""
    from swegemma.edit import apply_replacement
    tgt = rec["calls"][call - 1]["target"]["tool_calls"][0]
    v = rec["verify"]
    got = (m.get("tool_calls") or [None])[0]
    if not got:
        return ("no_call" if m.get("finish_reason") != "length" else "cut_off"), None
    if got["name"] != tgt["name"]:
        return f"other_tool:{got['name']}", None
    try:
        ga = json.loads(got["args"]) if isinstance(got["args"], str) else got["args"]
    except (json.JSONDecodeError, TypeError):
        ga = None
    if not isinstance(ga, dict):
        return "bad_args", None
    if ga.get("filepath", "").removeprefix("/workspace/") != v["path"]:
        return "wrong_file", None
    if tgt["name"] != "edit_file":
        return "same_tool", None
    if "old_string" not in ga or "new_string" not in ga:
        return "bad_args", None
    try:
        r = apply_replacement(drill_file(rec, cache), ga["old_string"], ga["new_string"])
        return ("intended" if sha(r.new_content) == v["intended_sha256"] else "applied_other_change"), ga
    except ValueError as e:
        msg = str(e)
        return ("several_matches" if ("times" in msg or "locations" in msg or "multiple" in msg) else "not_found"), ga


def score(a):
    """per decoded edit call: parsed, edit_file on the right file, applied exactly once, the intended file — by family
    and distance. The gate reads not-found at distance ≥ 2 and several-matches."""
    recs = load_records(a.records)
    files, rows = {}, []
    for l in open(a.decoded):
        m = json.loads(l)
        rec = recs.get(m["instance_id"])
        if not rec:
            continue
        tgt = rec["calls"][m["call"] - 1]["target"]["tool_calls"][0]
        row = {"family": rec["family"], "distance": rec["distance"], "call": m["call"], "target": tgt["name"],
               "outcome": None, "repeated": rec.get("dup_seen", "?")}
        row["outcome"], ga = call_outcome(rec, m["call"], m, files)
        if rec.get("failed_old") is not None and tgt["name"] == "edit_file":   # a failed old_string sent again
            row["resend"] = bool(ga) and ga.get("old_string") == rec["failed_old"]
        if ga and isinstance(ga.get("old_string"), str):        # answer length: the model's old_string vs the target's
            row["old_lines"] = ga["old_string"].count("\n") + 1
            row["tgt_lines"] = json.loads(tgt["args"])["old_string"].count("\n") + 1
        rows.append(row)
    edits = [r for r in rows if r["target"] == "edit_file"]
    tot = Counter(r["outcome"] for r in edits)
    n = len(edits) or 1
    print(f"edit_drills score · {a.decoded} on {a.records} · {len(rows)} decoded calls, {len(edits)} edit targets")
    print("  edit outcomes:", {k: f"{v} ({100 * v / n:.0f}%)" for k, v in tot.most_common()})
    for key in ("family", "distance", "repeated"):
        groups = sorted({r[key] for r in edits})
        for g in groups:
            sub = [r for r in edits if r[key] == g]
            k = Counter(r["outcome"] for r in sub)
            print(f"  {key} {g}: n {len(sub)} · intended {100 * k['intended'] / len(sub):.0f}% · not_found "
                  f"{100 * k['not_found'] / len(sub):.0f}% · several {100 * k['several_matches'] / len(sub):.0f}%")
    rr = [r for r in rows if r["target"] == "read_file"]
    if rr:
        print(f"  recover call 1 (target read_file): same tool {sum(r['outcome'] == 'same_tool' for r in rr)} / {len(rr)}")
    rs_ = [r for r in edits if "resend" in r]
    if rs_:
        print(f"   resend (edit targets after a failed edit in context: the model's old_string is the failed one): "
              f"{sum(r['resend'] for r in rs_)} / {len(rs_)} · " + " · ".join(
                  f"{f} {sum(r['resend'] for r in rs_ if r['family'] == f)}/{sum(r['family'] == f for r in rs_)}"
                  for f in sorted({r['family'] for r in rs_})))
    far = [r for r in edits if r["distance"] >= 2]
    nf = sum(r["outcome"] == "not_found" for r in far) / (len(far) or 1)
    sev = tot["several_matches"] / n
    parsed = 1 - (tot["no_call"] + tot["bad_args"] + tot["cut_off"]) / n
    intended = tot["intended"] / n
    ok = intended >= 0.80 and nf <= 0.10 and sev <= 0.03 and parsed >= 0.99
    print(f"== gate: intended {100 * intended:.0f}% (≥ 80) · not_found at distance ≥ 2 {100 * nf:.0f}% (≤ 10) · "
          f"several matches {100 * sev:.0f}% (≤ 3) · parsed {100 * parsed:.0f}% (≥ 99) → {'PASS' if ok else 'not yet'}")
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    for lab_, sub in (("non-block", [r for r in edits if r["family"] != "block" and "old_lines" in r]),
                      ("block", [r for r in edits if r["family"] == "block" and "old_lines" in r])):
        if sub:
            print(f"   answer length ({lab_}, edits with an old_string): model median {med([r['old_lines'] for r in sub])} "
                  f"lines · mean {sum(r['old_lines'] for r in sub) / len(sub):.1f} · target median "
                  f"{med([r['tgt_lines'] for r in sub])} · mean {sum(r['tgt_lines'] for r in sub) / len(sub):.1f} (n {len(sub)})")
    blk = [r for r in edits if r["family"] == "block"]
    if blk:
        kb = Counter(r["outcome"] for r in blk)
        print(f"   block slice (repeated blocks, fastapi-like; read next to the gate): n {len(blk)} · intended "
              f"{100 * kb['intended'] / len(blk):.0f}% · several matches {100 * kb['several_matches'] / len(blk):.0f}%")


def think(a):
    """E4B's own thought into each target, taken only from a sample (decode --samples k) whose own call already was
    the checked answer (edit_file → the intended file; read_file → the right file). A thought that planned another
    anchor is never kept before the answer (it would teach "think X, do Y"); targets with no such sample keep an empty
    thought. The answer stays the checked one, the loss stays on the tool call (label_scope)"""
    recs = load_records(a.records)
    best, why, cache = {}, Counter(), {}
    for l in open(a.decoded):
        m = json.loads(l)
        rec = recs.get(m["instance_id"])
        if not rec:
            continue
        key = (m["instance_id"], m["call"])
        if not m.get("reasoning") or len(m["reasoning"]) > a.max_thought_chars:
            why["no_or_long_thought"] += 1
            continue
        out_, _ = call_outcome(rec, m["call"], m, cache)
        if out_ in ("intended", "same_tool"):     # the thought led to the checked answer itself (read: same file)
            best.setdefault(key, m["reasoning"])
            why["kept:" + out_] += 1
        else:                                     # a thought planning another anchor must not sit before the answer
            why["dropped:" + out_.split(":")[0]] += 1
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
    fam = Counter(rec["family"] for iid, rec in recs.items() for c in rec["calls"] if (iid, c["call"]) in best)
    tot = Counter(rec["family"] for rec in recs.values() for c in rec["calls"])
    print(f"   samples: {dict(sorted(why.items()))}")
    print(f"   calls with a thought by family: " + " · ".join(f"{k} {fam[k]}/{tot[k]}" for k in sorted(tot)))
    print(f"== think: {st['native']} / {st['calls']} target calls got E4B's own thought ({a.decoded}, ≤ "
          f"{a.max_thought_chars} chars, only from a sample whose own call was the checked answer: edit → the intended "
          f"file, read → the right file); the rest keep an empty thought · {len(recs)} records → {out}")


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
    b.add_argument("--file-cap", type=int, default=15, help="at most this many drills per source file")
    b.add_argument("--multi", type=float, default=0.3, help="share of copy/recover drills drawn from multi-line sites")
    b.add_argument("--families", default="", help="override the family weights, e.g. block=1 (the held-out block slice)")
    b.add_argument("--block-libs", default="typer,openai,networkx",
                   help="libraries for the block family ('' = no block drills); held-out: sqlmodel,huggingface_hub")
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
