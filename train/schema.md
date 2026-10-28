# Training data schema (call level)

One JSON file per trajectory under `runs/sft/<source>/<instance_id>[.<run>].json` (gitignored: it quotes task
text). The unit of training is **one model call**: every call carries the complete input the model saw and the
reply it produced. A `segment` is only a boundary marker: calls in the same segment satisfy the *prefix property*
(the input of call k+1 = the input of call k + the reply of call k + the tool response). A nudge (harness user
message, which also drops earlier thoughts), a compaction summary or a history reset starts a new segment.

```
{
  "instance_id": "rich_3506",            # task id (public splits may quote ids; never the task text)
  "repo": "Textualize/rich",
  "base_commit": "…",                    # from harness/tasks.jsonl when known, else null
  "pr": null,                            # PR number when known (mined tasks)
  "patch_sha": "…",                      # sha256 of the final agent patch, for dedup against eval tasks
  "source": "e4b_onpolicy | 31b_teacher | opus | d1_qwen | mined",
  "resolved": true,                      # strict verification result of this run (null = unknown)
  "policy_model": "gemma-4-e4b-it-qat-w4a16-ct",
  "sampling": {"temperature": 0.2, "top_p": 0.95, "max_completion_tokens": 6144, "thinking_token_budget": 4096,
               "max_tool_calls": 28, "max_time_minutes": 7},
  "env_fingerprint": null,               # sha256 from train/env_fingerprint.py (Phase 0), null for old runs
  "tools": [ … the tool list exactly as sent (OpenAI function schemas) … ],
  "calls": [
    {
      "call": 1,                         # 1-based, as in the traj log
      "segment": 0,                      # increments at nudge / compaction / reset
      "events_before": [],               # subset of ["nudge", "compaction", "reset"] that happened before this call
      "input_messages": [ … the full message list sent to the model, verbatim from the traj log … ],
      "target": {"reasoning": "…", "content": "", "tool_calls": [{"id": "…", "name": "run_command", "args": "{…}"}],
                 "finish_reason": "tool_calls"},
      "thought_source": "native | qwen | synthetic_glm | synthetic_31b | none",
      "usage_prompt_tokens": 4267,       # the server's prompt_tokens for this call (null when unknown)
      "err": null                        # the call's exception when it failed (no target then)
    }
  ]
}
```

Conventions
- `input_messages` keep the harness's wire format: system/user `content` strings, assistant messages re-sent with
  `reasoning_content`, `tool_calls[].function.arguments` as a JSON **string**, tool results as `role: tool` with
  `tool_call_id`. `render.py` normalises for the chat template (system content → one text part, arguments → dict),
  which reproduces vLLM's rendering byte for byte.
- `target` is the model's reply as returned by the server (thinking in `reasoning`). A call with `err` or without a
  reply has `target: null` and is never a training target; its input still documents the trajectory.
- `thought_source` is per call: `native` = produced by the policy itself; `qwen` = folded `think` calls from the
  SWE-rebench conversion; `synthetic_*` = written afterwards by another model given the context and the action;
  `none` = no thought (Opus traces).

Rendered form (`render.py`): per call `{input_ids, labels, meta}` where labels are −100 everywhere except the reply
tokens, which end with the stop token the model emitted: `<|tool_response>` (id 50) after tool calls, `<turn|>`
(id 106) after a text-only reply. `--pack` joins the calls of one segment into one sequence with several label
spans; it is allowed only when the token-level prefix property holds for every call of the segment, and the total
number of label tokens must equal the per-call total.
