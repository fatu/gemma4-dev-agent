# The kit line (E4B) — `LATEST=kit`

The setting the dmitrii kit's native 31B trajectories were generated with (Budget-Fit 0.18), for the kit line (user
decision 2026-11-03: "一条用kit的数据用kit里面的提示词来做训练和之后推理"): training data from
`train/convert_kit.py --prompt kit` (the kit's prompt as served) and inference with this directory, so the model sees the
same prompt in both. A third setting next to `../latest/` (31B) and `../latest_e4b/` (E4B); it is not the latest
setting and does not replace either.

| knob | value | file |
|---|---|---|
| model / adapter | gemma-4-e4b-it-qat-w4a16-ct + `ple_fix` (as ../latest_e4b/; `MODEL_ALIAS=… ADAPTER=` for the 31B reference) | model.env |
| temperature / top_p | 0.65 / 0.95 | configs/sampling.yaml |
| max_output_tokens / thinking | 8192 / 4096, include_thoughts | configs/sampling.yaml |
| command timeout · calls · time · turns | 240 s · 28 · 8 min · 80 | eval_config.yaml |
| prompt | the kit's served system prompt without ADK's agent suffix (4,676 chars) | prompts/system.md |
| task prompt time line | the literal 8 minutes (PROMPT_TIME_MIN; the run flag is scaled to the box) | scripts/prompt_shim.py |

Source of every value: the saved requests of the kit's 103 native resolved trajectories (`lora_native_resolved.jsonl`
of dmitriigluzdov's "gemma-4-serving-and-lora-verification-kit", CC BY 4.0): one system prompt (sha1 dd2039aca7) in all
103, its task prompts' budget lines, and the request sampling (seeds 42/43 not reproduced). The system prompt is
Budget-Fit 0.18 = flexonafft's budget-aware prompt + verracodeguacas's Budget-Fit core (Apache-2.0 notebooks; kit
`lora_build_notes.md`). It is not the kit's `budget_aware_system.md` file, which differs.

    LATEST=kit SUBMISSION=runs/base_lb024 ARMS=base TAG=kit1 TIME_MIN=5 MAX_TURNS=80 bash scripts/eval_dev.sh   # E4B
    LATEST=kit MODEL_ALIAS=gemma-4-31b-it-qat-w4a16-ct ADAPTER= … bash scripts/eval_dev.sh                       # 31B reference
    BUILD_ONLY=1 LATEST=kit SUBMISSION=runs/base_lb024 ARMS=base TAG=kitsub bash scripts/eval_dev.sh             # build only

## History
- 2026-11-03: created (kit line step 2).
