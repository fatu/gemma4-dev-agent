# gemma4-dev-agent

Agent config, prompts, tools and training scripts for the Kaggle
"Gemma 4 Developer Agent" competition (+ Paper Track).

Competition harness and task data are **not** included — download them from Kaggle
after accepting the rules into `harness/` (gitignored).

## On the H100 (W1)

```bash
git clone https://github.com/fatu/gemma4-dev-agent.git && cd gemma4-dev-agent
# harness, sample submission, public tasks: download from the competition's Data tab (needs your Kaggle login)
#   into harness/ (gitignored) and follow its README for swegemma.
pip install vllm                                   # or uv venv + uv pip install vllm
mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &   # first start downloads ~17 GB
python scripts/smoke.py                            # 📷 one chat completion + tokens/s
```
