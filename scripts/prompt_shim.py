"""Show the submission's literal minutes in the task prompt's time line (PROMPT_TIME_MIN; eval_dev.sh sets it for
LATEST=kit to the setting's eval_config max_time_minutes).

The harness writes "- Time allowance: {budget.time_minutes} minutes" (agent_runner.build_agent_prompt). The scorer takes
the minutes from the submission's eval_config.yaml, an int there ("8 minutes" in all of the dmitrii kit's served
prompts), while `swegemma eval --max-time-minutes` is a float that eval_dev.sh scales to the box's decode speed
(TIME_MIN x minutes / 8): local prompts said "7.0 minutes", box prompts e.g. "4.375 minutes". With PROMPT_TIME_MIN set,
the line shows that value (no ".0" for a whole number); the enforced budget stays the run flag. Prints one line the first
time it rewrites a prompt. Loaded by scripts/eval_sitecustomize.py.
"""
import functools
import os
import re

VAL = os.environ.get("PROMPT_TIME_MIN")
_state = {"said": False}


def shown(v):
    f = float(v)
    return str(int(f)) if f == int(f) else str(f)


def install():
    try:
        from swegemma.harness import agent_runner
    except Exception:                        # not the eval venv: do nothing
        return
    orig = agent_runner.build_agent_prompt

    @functools.wraps(orig)
    def build(*a, **kw):                     # printed here, not at import: only a real eval builds a task prompt
        p = orig(*a, **kw)
        p2, n = re.subn(r"^- Time allowance: [0-9.]+ minutes$", f"- Time allowance: {shown(VAL)} minutes", p, count=1, flags=re.M)
        if not _state["said"]:
            _state["said"] = True
            old = re.search(r"^- Time allowance: ([0-9.]+) minutes$", p, re.M)
            print(f"prompt_shim: task prompt time line {old.group(1) if old else '(none)'} → {shown(VAL)} minutes "
                  f"(PROMPT_TIME_MIN; the enforced budget is unchanged){'' if n else ' — NO time line found, prompt unchanged'}",
                  flush=True)
        return p2

    agent_runner.build_agent_prompt = build


if VAL:
    install()
