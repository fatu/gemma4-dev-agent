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
"""
import json
import re

ACTION_ERRORS = ("FileEditError", "FileReadError", "FileWriteError", "ValueError", "BudgetExceeded")
BROKEN_CMD = re.compile(r"command not found|: not found\b|SyntaxError|IndentationError|No such file or directory|"
                        r"can't open file|ERROR: file or directory not found|unrecognized arguments", re.I)
PYTEST_USAGE = re.compile(r"ERROR: file or directory not found|unrecognized arguments|ERROR: not found:|no tests ran", re.I)


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
