# gemma4-dev-agent

Agent config, prompts, tools and training scripts for the Kaggle
"Gemma 4 Developer Agent" competition (+ Paper Track).

Competition harness and task data are **not** included — download them from Kaggle
after accepting the rules into `harness/` (gitignored).

## On the H100 (W1)

```bash
git clone https://github.com/fatu/gemma4-dev-agent.git && cd gemma4-dev-agent
bash scripts/get_harness.sh                        # competition data (~22 GB) + the swegemma wheelhouse → harness/, installs the CLI
#   (needs ~/.kaggle/kaggle.json; the CLI is a separate Kaggle dataset, not on PyPI)
pip install vllm                                   # or uv venv + uv pip install vllm
mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &   # first start downloads ~17 GB
python scripts/smoke.py                            # 📷 one chat completion + tokens/s
pkill -f 'vllm serve'                              # the harness runs its own vLLM (TP 4, :8000) — free the GPUs first
bash scripts/eval_sample.sh                        # 📷 sample submission on 5 tasks → runs/w1_sample5/summary.json
```

Facts from HARNESS_README.md that shape everything later: the Kaggle runtime is 4× L4 (24 GB each), vLLM TP 4 at
0.80 utilisation, **32,768-token context**, thinking enabled, `gemma4` tool-call/reasoning parsers; the agent gets 9
tools (run_command, read_file, edit_file, write_file, get_status, submit_patch, get_code_neighbors,
search_similar_code, get_code_subgraph); patches are `git diff` of /workspace, verified by hidden pytest in a second
container; budgets per task come from the submission's `eval_config.yaml` (sample: 1 min, 10 tool calls, 50 turns);
the public set is 129 tasks over fastapi (67), rich (48), requests (13), httpx (1).
