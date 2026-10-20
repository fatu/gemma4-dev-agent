# Box plan: what to run on the RTX PRO 6000 box next (written 2026-10-20)

**Why the box now.** The Kaggle GPU quota is spent until Sat Oct 24 00:00 UTC (4×L4 burns ~2 quota h per hour). The
box is free to use, ~2× faster at decoding than 4×L4, and since `install_testdeps.sh` it verifies all dev33 tasks
correctly. L4 is kept for one confirmation run of the final candidate.

**What the Kaggle lab already settled** (`RESULTS.md`):
- The public top config ("Baseline | LB 0.24" notebook) is reproducible: 8 and then 10 of 33, and the core 8 tasks repeat.
- `nothink`, `out6k` and `out4k` (budget-envelope tweaks) tie or lose. Only 10 distinct tasks were ever solved.
  **The next gain has to come from solving new tasks**, so the arms below change *how the agent works*, not its
  budget.

**How the box runs.** The box can `git clone` / `git pull` the public repo but can't push, and there's no network to
the laptop. So the laptop writes code → you `git pull` on the box and run → you photograph the result → the laptop
logs it in `RESULTS.md` and `05_run_log.md`. Each step below = one run + its photos.

**Decision rule** (dev33 noise is about ±2 tasks): an arm counts as better only if it solves **≥ 2 more tasks than
`base` in the same run, with ≥ 2 arm-only tasks** in the pairs. Otherwise drop it. An arm that loses ≥ 2 tasks is
dropped too.

---

## B-1 · One-time setup (~20 min) ▶ next

```bash
cd gemma4-dev-agent && git pull && source .venv-eval/bin/activate
CHECK=<a requests task id from splits/dev33.txt> bash scripts/install_testdeps.sh   # needs PyPI
kaggle kernels pull honghanhhh/gemma-4-baseline-lb-0-24 -p /tmp/lb024   # needs Kaggle
python3 scripts/from_notebook.py /tmp/lb024/*.ipynb runs/base_lb024
mkdir -p runs && nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &   # all 4 cards; wait if Bonsai runs
curl -s http://127.0.0.1:8000/v1/models | head -c 200
```
📷 the last 2 lines of `install_testdeps.sh` (`testdeps ok …` and `… recursive-fixture errors: 0`), plus the `curl` output.
If it breaks: Kaggle unreachable → photo the error (another way to get the prompt). vLLM down → `tail -30 runs/vllm.log`.

## B-2 · Clean baseline + ab3 (~3–4 h)

```bash
SUBMISSION=runs/base_lb024 ARMS="base tools6 tools6+submit" nohup bash scripts/eval_dev.sh > runs/eval_dev_w4.log 2>&1 &
```
- `base`: the public top, unchanged. **The first dev33 number after the env fix**, the new reference for everything.
- `tools6`: the 3 graph tools the prompt forbids are removed from `agent.yaml` (their schemas go out on every call).
- `tools6+submit`: plus "submit as soon as the fix is in place and one targeted check has run".
- Budget: 28 calls · `TIME_MIN=4` (≈ L4's 8 min at the box's speed) · 240 s cmd · 300 s verify.

📷 the first lines (`dev split: … budget: --max-tool-calls 28 --max-time-minutes 4 …`) and the final compare
table (`tail -60 runs/eval_dev_w4.log`).
**Decide:** the B-3 base = `tools6` (or `tools6+submit`) if it passes the decision rule, else `base`.

## B-3 · Arms that target new solves (~4–5 h; the laptop writes them first, you say "go")

Each comes from a measured failure in the Kaggle traces:

| arm | change | evidence |
|---|---|---|
| `editrecover` | prompt rule: if `edit_file` reports a missing parameter, switch at once to `run_command` with a quoted-heredoc Python replace (assert the old text occurs once) | 377/377 failed edits lacked `old_string`. One overflow run retried 7× and died |
| `issuefirst` | prompt rule: if the issue names a file / function or shows the fix, open that place first and apply it, then verify | wins edit at call 6 vs 15 for losses, after 2 reads vs 6 |
| `tmpfix` | fix the prompt's "/tmp for scratch files": `write_file` can't write outside /workspace → use `python3 -c` or a heredoc | `write_file('/tmp/…')` is rejected (path traversal) and costs a call |
| `test1` | one targeted pytest before `submit_patch` becomes required (a failing run → fix once) | only 6/33 runs ran any test |

```bash
SUBMISSION=runs/base_lb024 ARMS="<B2 base> <B2 base>+editrecover <B2 base>+issuefirst <B2 base>+tmpfix+test1" \
  nohup bash scripts/eval_dev.sh > runs/eval_dev_w5.log 2>&1 &
```
📷 the compare table. **Decide:** stack every arm that passes the rule into one config.

## B-4 · Repeat the best 2 configs (~2–3 h)

```bash
TAG=r2 SUBMISSION=runs/base_lb024 ARMS="<B2 base> <best stack>" nohup bash scripts/eval_dev.sh > runs/eval_dev_r2.log 2>&1 &
```
📷 the compare table. A second run checks the gain is not noise (same-direction pairs in both runs).

## B-5 · L4 confirmation (after Oct 24, Kaggle, laptop side)

The laptop pushes the best stack + `base` as one Kaggle 4×L4 kernel (~11 quota h for 2 arms). Two goals: confirm the
gain on the scorer's hardware, and **calibrate the box**: does box-with-TIME_MIN=4 rank configs the same way as L4?

## B-6 · Candidate submission (the laptop asks before submitting)

Build the zip (`scripts/make_submission.py`), card it (zip sha + prompt sha + commit), submit **only after your OK**,
then log the public score in `RESULTS.md`. Compare with the public top's 0.22–0.24 (±2 tasks = ±0.035).

---

## Not in this plan
- **Small-model bonus (E4B/E2B):** planned separately. Running it on the box
  first needs a `SERVED=` override in `serve_vllm.sh`.
- **Budget arms** (`nothink`, output caps, thinking budget): settled by ab1/ab2, so not repeated.

## Time budget
~1–1.5 h per arm at TIME_MIN=4 (33 tasks). B-1…B-4 ≈ 10–12 box hours over 2–3 sessions. Each step needs one 📷 set.
