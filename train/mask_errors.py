"""Step-level error masking for call-level records (train/schema.md): a reply whose tool call failed as an action stays
in the trajectory (later calls see it and its error as context) but is not a target (`target: null`,
`excluded: "tool_error"`), so the student learns the recovery after the error, not the error itself (SWE-Lego
2601.01426 step-level error masking; ROME 2512.24873 masks turns that trigger a tool error).

Which errors count (the harness returns {"status": "error", "error_type": …}, or {"error": …} for a missing parameter):
  FileEditError (old_string not found …), FileReadError, FileWriteError, ValueError (unknown tool), BudgetExceeded,
  missing_param ({"error": "Invoking … failed"}) — the action itself was wrong.
  CommandError (run_command exit code ≠ 0) only when the command itself was broken: BROKEN_CMD in its message
  (command not found, a SyntaxError / IndentationError in the agent's own python -c or heredoc, a path that does not
  exist); for a pytest command only pytest's own usage errors (PYTEST_USAGE), since a test's traceback may name
  SyntaxError / missing files of the code under test. Any other non-zero exit is kept as a target: a failing test or
  an empty grep is the information the agent asked for (a repro that fails is the right move).
A reply with several tool calls is excluded if any of them failed this way. Calls already excluded keep their reason.

Hidden failures (2026-11-25, `mask_hidden_failures`, reason `hidden_failure`): a run_command whose result says
{"status": "ok", "exit_code": 0} although the command failed, because a pipe (`| tail`, `| head`, `| grep`) or a
following `; cmd` replaced the exit code. Same idea as above — only a command that was itself broken is masked, a
failing test or a repro that shows the bug stays a target — read from stdout + stderr:
  pytest_usage      a pytest command: file or directory not found, unrecognized arguments, `ERROR: not found:`,
                    no tests ran, every test deselected by -k, a summary line with no test in it
  pytest_collect    a pytest command: errors during collection / a conftest that does not import, and no test ran
  shell             (only when the command printed nothing else) a line of the shell or a tool itself: `bash: …: command not found`, `grep|cat|ls|sed|…: X: No such
                    file or directory`, `python: can't open file`, a shell syntax error, git's `fatal: …`
  own_script        a Python traceback whose LAST frame is the agent's own inline code (File "<string>" / "<stdin>" /
                    a script under /tmp or /dev/shm), or a SyntaxError / IndentationError there, or a
                    ModuleNotFoundError / ImportError raised by an inline script. AssertionError is left out (an
                    assert that fails is what the script was written to report). A traceback that ends inside the
                    repository's code is the bug or the test speaking and stays a target.
  killed            the output shows the process was killed (`Killed`, an echoed exit status 137 / 124)
"""
import json
import re

ACTION_ERRORS = ("FileEditError", "FileReadError", "FileWriteError", "ValueError", "BudgetExceeded")
BROKEN_CMD = re.compile(r"command not found|: not found\b|SyntaxError|IndentationError|No such file or directory|"
                        r"can't open file|ERROR: file or directory not found|unrecognized arguments", re.I)
PYTEST_USAGE = re.compile(r"ERROR: file or directory not found|unrecognized arguments|ERROR: not found:|no tests ran", re.I)


SHELL_LINE = re.compile(
    r"^(?:/bin/)?(?:ba)?sh: (?:line \d+: )?.*(?:command not found|No such file or directory|[Ss]yntax error|unexpected EOF)"
    r"|^(?:grep|cat|ls|sed|head|tail|find|wc|cd|awk|diff|cp|mv|rm|sort|xargs|rg|tree|stat|file|touch): .*"
    r"(?:No such file or directory|cannot access|[Nn]ot a directory)"
    r"|^python3?(?:\.\d+)?: can't open file|^fatal: |^error: unknown (?:option|switch)|^usage: git ", re.M)
PYTEST_COLLECT = re.compile(r"errors? during collection|ImportError while loading conftest|"
                            r"ERROR collecting|INTERNALERROR", re.I)
PYTEST_RAN = re.compile(r"\b\d+ (?:passed|failed|xfailed|xpassed)\b")
NOTHING_RAN = re.compile(r"^(?:=+ )?(?:no tests ran|\d+ (?:deselected|warnings?))(?:, \d+ (?:deselected|warnings?))* in \d+\.\d+s", re.M)
SEPARATOR = re.compile(r"^[-=#*_ ]*$")
OWN_FRAME = re.compile(r'File "(<string>|<stdin>|/tmp/[^"]*|/dev/shm/[^"]*)", line \d+')
ANY_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+', re.M)
ERR_LINE = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Failure|Exit|Interrupt))\b", re.M)
KILLED = re.compile(r"^.*\bKilled\b\s+.*$|\b(?:RC|EXIT|exit|rc|PYEXIT|PYTEST_RC)\w*[=: ]+(?:137|124)\b", re.M)
INLINE = re.compile(r"python3?(?:\.\d+)?\s+(?:-\w+\s+)*(?:-c\b|-\s|<<)|<<\s*['\"]?\w+|python3?(?:\.\d+)?\s+/(?:tmp|dev/shm)/")


def hidden_kind(text, cmd=""):
    """why a run_command that reports status ok / exit code 0 in fact failed as an action, or None"""
    try:
        r = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(r, dict) or r.get("status") != "ok" or r.get("exit_code") not in (0, None):
        return None
    out = (r.get("stdout") or "") + "\n" + (r.get("stderr") or "")
    if not out.strip():
        return None
    if "pytest" in cmd or "py.test" in cmd:
        ran = PYTEST_RAN.search(out)
        if PYTEST_USAGE.search(out):
            return "pytest_usage"
        if NOTHING_RAN.search(out) and not ran and not re.search(r"\b\d+ errors?\b", out):
            return "pytest_usage"
        if PYTEST_COLLECT.search(out) and not ran:
            return "pytest_collect"
    if SHELL_LINE.search(out) and not [l for l in out.splitlines() if not SEPARATOR.match(l) and not SHELL_LINE.match(l)]:
        return "shell"
    if KILLED.search(out):
        return "killed"
    if INLINE.search(cmd):
        if re.search(r"^(?:SyntaxError|IndentationError)\b", out, re.M) and OWN_FRAME.search(out):
            return "own_script"
        tb = out.rfind("Traceback (most recent call last)")
        if tb >= 0:
            tail = out[tb:]
            frames = ANY_FRAME.findall(tail)
            errs = [e for e in ERR_LINE.findall(tail)]
            last = errs[-1].split(".")[-1] if errs else ""
            if last == "AssertionError":
                return None
            if last in ("ModuleNotFoundError", "ImportError"):
                return "own_script"
            if frames and (frames[-1] in ("<string>", "<stdin>") or frames[-1].startswith(("/tmp/", "/dev/shm/"))):
                return "own_script"
        elif re.search(r"^(?:ModuleNotFoundError|ImportError): ", out, re.M):
            return "own_script"
    return None


def mask_hidden_failures(rec, stats, mode="on"):
    """set target None + excluded "hidden_failure" on replies with a run_command that failed behind exit code 0;
    counts go to stats (keys "hidden: …"). mode "off" only counts."""
    res = results(rec)
    hit = 0
    for c in rec["calls"]:
        t = c.get("target")
        if not t or c.get("excluded"):
            continue
        kinds = [k for tc in t.get("tool_calls") or [] if tc.get("name") == "run_command"
                 for k in [hidden_kind(res.get(tc.get("id")) or "", command(tc))] if k]
        if not kinds:
            continue
        for k in kinds:
            stats[f"hidden: {k}"] += 1
        if mode == "on":
            c["target"], c["excluded"] = None, "hidden_failure"
            hit += 1
    stats["hidden: targets excluded"] += hit
    stats["hidden: trajectories"] += bool(hit)
    return rec


def hidden_summary(stats, mode):
    return (f"hidden-failure masking ({mode}): " + " · ".join(f"{k[len('hidden: '):]} {v}" for k, v in sorted(stats.items())
                                                           if k.startswith("hidden: ")))


def results(rec):
    """{tool_call_id: tool result text} from every call's history"""
    out = {}
    for c in rec["calls"]:
        for m in c["input_messages"]:
            if m.get("role") == "tool" and m.get("tool_call_id") and m["tool_call_id"] not in out:
                v = m.get("content")
                out[m["tool_call_id"]] = v if isinstance(v, str) else json.dumps(v)
    return out


def command(tc):
    a = tc.get("args") or "{}"
    try:
        a = json.loads(a) if isinstance(a, str) else a
    except ValueError:
        return ""
    return (a.get("command") or "") if isinstance(a, dict) else ""


def kind(text, cmd=""):
    """the error kind that makes a call an excluded target, or None (ok, no result, or an informative non-zero exit)"""
    try:
        r = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(r, dict):
        return None
    if r.get("status") != "error":
        return "missing_param" if "error" in r and "status" not in r else None
    et = r.get("error_type")
    if et in ACTION_ERRORS:
        return et
    if et == "CommandError":
        msg = (r.get("error_message") or "") + ((r.get("details") or {}).get("stderr") or "")
        pat = PYTEST_USAGE if "pytest" in (cmd or "") else BROKEN_CMD
        return "CommandError:broken" if pat.search(msg) else None
    return f"other:{et}"


def mask_tool_errors(rec, stats, mode="on"):
    """set target None + excluded "tool_error" on replies whose tool call failed as an action; counts go to stats
    (keys "tool_error: …"). mode "off" only counts."""
    res = results(rec)
    hit = 0
    for c in rec["calls"]:
        t = c.get("target")
        if not t or c.get("excluded"):
            continue
        kinds = []
        for tc in t.get("tool_calls") or []:
            txt = res.get(tc.get("id")) or ""
            k = kind(txt, command(tc))
            if k:
                kinds.append(k)
            elif '"CommandError"' in txt[:80]:
                stats["tool_error: CommandError kept (informative exit)"] += 1
        if not kinds:
            continue
        for k in kinds:
            stats[f"tool_error: {k}"] += 1
        if mode == "on":
            c["target"], c["excluded"] = None, "tool_error"
            hit += 1
    stats["tool_error: targets excluded"] += hit
    stats["tool_error: trajectories"] += bool(hit)
    return rec


def summary(stats, mode):
    return (f"tool-error masking ({mode}): " + " · ".join(f"{k[len('tool_error: '):]} {v}" for k, v in sorted(stats.items())
                                                       if k.startswith("tool_error: ")))
