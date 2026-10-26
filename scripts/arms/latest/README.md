# The latest setting (one copy)

Every eval_dev.sh arm (LATEST=1, the default) and every make_submission.py build starts from this directory. Arms are
explicit deltas on top (t10, out4k, course rule parts, …). Change these files only when a newer setting is adopted, and
add a line to the history below in the same commit.

| knob | value | file |
|---|---|---|
| temperature / top_p | 0.2 / 0.95 | configs/sampling.yaml |
| max_output_tokens | 6144 (thinking + text + tool call) | configs/sampling.yaml |
| thinking_budget / include_thoughts | 4096 / true | configs/sampling.yaml |
| command timeout | 240 s | eval_config.yaml |
| tool calls / time / turns | 28 / 7 min / 80 | eval_config.yaml |
| model | gemma-4-31b-it-qat-w4a16-ct (E4B entry: gemma-4-e4b-it-qat-w4a16-ct) | agent.yaml of the base |
| tools | the 9 harness tools, no adapter, no sub-agents | agent.yaml of the base |
| prompt | the base submission's prompts/system.md, budget line rewritten to "28 tool calls and 7 minutes" | eval_dev.sh |

The base submission dir is built locally with `scripts/from_notebook.py` (see eval_dev.sh) and is not in the repo.

**Check any submission against it** (dir, zip or a notebook that writes the files inline):

    python3 scripts/latest_diff.py dist/<name>/submission.zip      # every knob: same | latest → this, then the prompt diff

## History
- 2026-10-25: v5 safe from the course's configuration slide (0.2 · 6144 · think 4096 · 240 s · 28 calls · 7 min · 80
  turns) replaces the base's own 8192 / 8 min.
