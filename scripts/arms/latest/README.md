# The latest 31B setting — `LATEST=31b` (the default)

One of two latest settings since 2026-10-30: this one is the 31B entry; `../latest_e4b/` is the small-model entry
(E4B + LoRA, `LATEST=e4b`). `model.env` here names the model and no adapter.

Every eval_dev.sh arm (LATEST=1, the default; `sample` keeps the organiser's prompt) and every make_submission.py build
starts from this directory. Arms are explicit deltas on top (t10, out4k, course rule parts, …). Change these files only
when a newer setting is adopted, and add a line to the history below in the same commit.

| knob | value | file |
|---|---|---|
| temperature / top_p | 0.2 / 0.95 | configs/sampling.yaml |
| max_output_tokens | 6144 (thinking + text + tool call) | configs/sampling.yaml |
| thinking_budget / include_thoughts | 4096 / true | configs/sampling.yaml |
| command timeout | 240 s | eval_config.yaml |
| tool calls / time / turns | 28 / 7 min / 80 | eval_config.yaml |
| prompt | 7,144 chars, budget line "28 tool calls and 7 minutes" | prompts/system.md |
| model | gemma-4-31b-it-qat-w4a16-ct (the E4B entry is ../latest_e4b/) | model.env |
| tools | the 9 harness tools, no adapter, no sub-agents | agent.yaml of the base |

**Check any submission against it** (dir, zip or a notebook that writes the files inline):

    python3 scripts/latest_diff.py dist/<name>/submission.zip      # every knob: same | latest → this, then the prompt diff

The submission card (make_submission.py) also says `= latest prompt` / `≠ latest prompt`.

## History
- 2026-10-25: v5 safe from the course's configuration slide (0.2 · 6144 · think 4096 · 240 s · 28 calls · 7 min · 80
  turns) replaces the base's own 8192 / 8 min.
- 2026-10-26: prompts/system.md joins the setting = the base prompt + the course lecture-3 notebook's 3 edits:
  finish with `git status --short && git diff` (untracked files are in the graded patch: the harness runs
  `git add -N .` before its diff), "keep the current behaviour for every input the issue does not mention", and pytest
  `2>&1 | tail -40` (no pipefail: the exit code is tail's; the `tail40` arm part adds `set -o pipefail`).
- 2026-10-30: two latest settings (user: "我们维护两个最新配置，31B和小模型LoRA"): this dir = 31B, ../latest_e4b/ = E4B +
  the PLE-fix LoRA. `LATEST=31b|e4b` picks one in eval_dev.sh, make_submission.py, latest_diff.py.
