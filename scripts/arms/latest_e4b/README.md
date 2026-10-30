# The latest small-model setting (E4B + LoRA) — `LATEST=e4b`

One of the two latest settings since 2026-10-30: `../latest/` is the 31B entry (`LATEST=31b`, the default), this
directory the small-model entry (E4B/E2B bonus). Same files as `../latest/` (copied that day; they may diverge) plus
`model.env`: the model alias and the adapter every E4B arm and build carries. Change these files only when a newer
small-model setting is adopted, and add a line to the history below in the same commit.

| knob | value | file |
|---|---|---|
| model | gemma-4-e4b-it-qat-w4a16-ct | model.env |
| adapter | `ple_fix` = the training-free PLE-projection LoRA (r128 α128 on `per_layer_model_projection`; fixes the QAT copy defect of forum #753277) from the public Kaggle dataset `michi883/ple-proj-svd-r128`, sha256 pinned | model.env |
| temperature / top_p | 0.2 / 0.95 | configs/sampling.yaml |
| max_output_tokens / thinking | 6144 / 4096, include_thoughts | configs/sampling.yaml |
| command timeout · calls · time · turns | 240 s · 28 · 7 min · 80 | eval_config.yaml |
| prompt | = ../latest/prompts/system.md (7,144 chars) | prompts/system.md |

The adapter files are not in the repo: `runs/adapters/ple_fix/` (gitignored). `eval_dev.sh` fetches them with the
kaggle CLI when missing and refuses a file whose sha256 differs from `model.env`.

    LATEST=e4b SUBMISSION=runs/base_lb024 ARMS=base TAG=e4bL bash scripts/eval_dev.sh        # box/pod eval
    MODEL=google/gemma-4-E4B-it-qat-w4a16-ct SUBMISSION=runs bash scripts/serve_vllm.sh     # server with runs/adapters/*
    BUILD_ONLY=1 LATEST=e4b SUBMISSION=runs/base_lb024 ARMS=base TAG=e4bsub bash scripts/eval_dev.sh
    LATEST=e4b .venv-eval/bin/python scripts/make_submission.py runs/subs/base_e4bsub      # NOLORA=0 implied, check 4b
    LATEST=e4b python3 scripts/latest_diff.py dist/<name>/submission.zip

Known cost: any adapter makes the scorer's vLLM reserve LoRA slots, which shrinks its KV cache (forum #744331; see
make_submission.py `nolora`).

## History
- 2026-10-30: created = ../latest/ (0.2 · 6144 · think 4096 · 240 s / 28 / 7 min / 80 · the 7,144-char prompt) +
  E4B + `ple_fix`.
