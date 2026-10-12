# gemma4-dev-agent

Agent config, prompts, tools and training scripts for the Kaggle
"Gemma 4 Developer Agent" competition (+ Paper Track).

Competition harness and task data are **not** included — download them from Kaggle
after accepting the rules into `harness/` (gitignored).

## Setup on the GPU box (W1)

```bash
git clone https://github.com/fatu/gemma4-dev-agent.git && cd gemma4-dev-agent
bash scripts/get_harness.sh            # competition data (~22 GB) + swegemma wheelhouse → harness/ (download only)
#   needs the kaggle CLI >= 2 + KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json). If the box can't reach Kaggle
#   (firewalled), run this on a laptop and copy harness/ to the box.
bash scripts/setup_eval_env.sh         # .venv-eval from requirements-eval.txt — scorer-matched pins + pytest  📷 env report
source .venv-eval/bin/activate
mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &   # stays up
python scripts/smoke.py                # 📷 one chat completion + tokens/s
bash scripts/eval_sample.sh            # 📷 sample submission on 5 tasks → runs/w1_sample5/summary.json (4/5 on 2026-10-12)
```

What `.venv-eval` must have (all in `requirements-eval.txt`; check any time with `bash scripts/env_report.sh`):
- **pytest + pytest-timeout** — the subprocess sandbox verifies each patch with this venv's python; without them every
  task fails verification (the sample went 0/5 → 4/5 once they were installed).
- **prometheus-fastapi-instrumentator 8.1.0** (from the wheelhouse) — with 7.1.0 vLLM 0.19's server returns 500 on every
  request: `'_IncludedRouter' object has no attribute 'path'`.
- vLLM 0.19.1 / torch 2.10.0 / transformers 5.13.1 / compressed-tensors 0.15.0.1 / swegemma 0.2.11 / adk-submission 0.2.13
  — the scorer's versions. A newer vLLM in another venv is not a substitute.

Box gotchas:
- No outbound HTTPS to Hugging Face (TLS interception → `CERTIFICATE_VERIFY_FAILED`): once the model is cached, serve with
  `HF_HUB_OFFLINE=1 nohup bash scripts/serve_vllm.sh …`.
- Restarting the server over SSH: `pkill -f '[v]llm serve'` — the plain `pkill -f 'vllm serve'` also matches the SSH
  command line that contains it and kills your own session's command.

`swegemma eval` does **not** start an inference server (the `VllmServer` in HARNESS_README is the Kaggle scorer's):
`serve_vllm.sh` serves the model with the scorer's flags — TP 4, 0.80 utilisation, 32K, `gemma4` tool-call/reasoning
parsers, thinking on, LoRA enabled — under the alias `gemma-4-31b-it-qat-w4a16-ct`, with every `adapters/<name>/` of the
submission (`SUBMISSION=…`, default the sample) registered as LoRA module `<name>`. `eval_sample.sh` waits for :8000 and
checks that list before running. One card for dev: `TP=1 GPU_UTIL=0.5 CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh`.

Facts from HARNESS_README.md that shape everything later: the Kaggle runtime is 4× L4 (24 GB each), vLLM TP 4 at
0.80 utilisation, **32,768-token context**, thinking enabled, `gemma4` tool-call/reasoning parsers; the agent gets 9
tools (run_command, read_file, edit_file, write_file, get_status, submit_patch, get_code_neighbors,
search_similar_code, get_code_subgraph); patches are `git diff` of /workspace, verified by hidden pytest in a second
container; budgets per task come from the submission's `eval_config.yaml` (sample: 1 min, 10 tool calls, 50 turns);
the public set is 129 tasks over fastapi (67), rich (48), requests (13), httpx (1).
