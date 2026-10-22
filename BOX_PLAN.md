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

## B-1 · One-time setup (~20 min) ✅ 2026-10-21

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

## B-2 · Clean baseline + ab3 (~3–4 h) ✅ 2026-10-21: base 11 · tools6 10 · tools6+submit 11 → B-3 base = `base`

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

## B-3 · Arms that target new solves ✅ 2026-10-21: base 12 · editrecover 9 · issuefirst 11 · tmpfix+test1 8, none passes — (~4–5 h; the laptop writes them first, you say "go")

Each comes from a measured failure in the Kaggle traces:

Each arm = the LB-0.24 base + one prompt line (`scripts/arms/<arm>.md`, applied by `scripts/arms/apply.py`); `base` runs again in the same pass.

| arm | change | evidence |
|---|---|---|
| `editrecover` | prompt rule: if `edit_file` reports a missing parameter, switch at once to `run_command` with a quoted-heredoc Python replace (assert the old text occurs once) | 377/377 failed edits lacked `old_string`. One overflow run retried 7× and died |
| `issuefirst` | prompt rule: if the issue names a file / function or shows the fix, open that place first and apply it, then verify | wins edit at call 6 vs 15 for losses, after 2 reads vs 6 |
| `tmpfix` | fix the prompt's "/tmp for scratch files": `write_file` can't write outside /workspace → use `python3 -c` or a heredoc | `write_file('/tmp/…')` is rejected (path traversal) and costs a call |
| `test1` | one targeted pytest before `submit_patch` becomes required (a failing run → fix once) | only 6/33 runs ran any test |

```bash
TAG=w5 SUBMISSION=runs/base_lb024 ARMS="base editrecover issuefirst tmpfix+test1" \
  nohup bash scripts/eval_dev.sh > runs/eval_dev_w5.log 2>&1 &
```
**After the run (📷 both):**
```bash
# 1. each rule must say "inserted …" or "replaced the anchor", never "appended (anchor not found)"
grep -E "^(editrecover|issuefirst|tmpfix|test1):|^== " runs/eval_dev_w5.log
# 2. the compare table (B-2's dev33_base is shown as a reference) + context growth per arm
tail -60 runs/eval_dev_w5.log
```
If a rule says "appended", the base prompt differs from the LB-0.24 one: photo `head -40 runs/subs/base/prompts/system.md`.
**Decide:** stack every arm that passes the rule into one config.

## B-3b · Rule stats + noise (~2.5 h) ✅ 2026-10-21: base 11 · 12 · 11 · 9; 7 always / 15 ever / 8 flip; rules followed, no effect

Two questions before any new arm: *did the agent follow the B-3 rules* (ignored vs followed-and-useless), and *how
noisy is one pass* (the two base passes differed on 5 tasks).

```bash
cd gemma4-dev-agent && git pull
# 1. rule-following per arm, from the w4/w5 traces (no GPU, ~1 min) 📷
python3 scripts/rulestats.py runs/dev33_base runs/dev33_base_w5 runs/dev33_editrecover_w5 runs/dev33_issuefirst_w5 runs/dev33_tmpfix+test1_w5
# 2. two more base passes (~1.2 h each; vLLM must be up). The second compare table lists all 4 base runs 📷
nohup bash -c 'TAG=r2 SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh;
  TAG=r3 REFS="base base_w5 base_r2" SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh' \
  > runs/eval_dev_rep.log 2>&1 &
```
**After the run (📷 three):**
```bash
# 1. the rule-following table (5 columns, ~25 rows; two photos if the font is small)
python3 scripts/rulestats.py runs/dev33_base runs/dev33_base_w5 runs/dev33_editrecover_w5 runs/dev33_issuefirst_w5 runs/dev33_tmpfix+test1_w5
# 2. the compare table with all 4 base runs + "tasks that differ" + "resolved in every run"
tail -45 runs/eval_dev_rep.log
# 3. same concurrency in B-2 / B-3 / the repeats? and did r2, r3 finish all 33 tasks?
grep -m1 "dev split" runs/eval_dev_w4.log runs/eval_dev_w5.log runs/eval_dev_rep.log; grep -c resolved runs/dev33_base_r2/task_results.jsonl runs/dev33_base_r3/task_results.jsonl
# if the tail shows fewer than 4 base rows (REFS not applied):
python3 scripts/compare_runs.py runs/dev33_base runs/dev33_base_w5 runs/dev33_base_r2 runs/dev33_base_r3
```
To see these commands on the box: `sed -n '/## B-3b/,/## B-4/p' BOX_PLAN.md`.
**Read:** a rule whose row did not move was ignored (→ rewrite it, or it needs a tool change, not a prompt line); a row
that moved with no score gain is a dead end. With 4 base passes: the per-task flip count sets how many passes an arm
needs before the decision rule can be applied (if base-vs-base swings 5, one pass per arm decides nothing).

## B-3c · Why the flip tasks flip (no GPU, ~1 min) ✅ 2026-10-21: 10 of 17 flip losses are the 4-min cap; wins on the same tasks at 244–246 s → calibrate TIME_MIN first (B-3d, to be written)

The 8 flip tasks are solvable and die of budget. Before touching time or context, see *which* budget and how close
the winning passes were to it:

```bash
cd gemma4-dev-agent && git pull
python3 scripts/flips.py runs/dev33_base runs/dev33_base_w5 runs/dev33_base_r2 runs/dev33_base_r3
```
📷 the whole output (≈40 lines: one line per flip task × pass, then the summary). **Read:** losing passes mostly `T`
with winning passes near 240 s → the box's 4-min cap is the flipper (a time lever, or trust L4 at 8 min); mostly `O`
with winning peak prompts near the limit → a context lever (bounded reads, smaller output reservation on this base);
mostly `✗` → the fix itself is unreliable (temperature / sampling), not the budget.

## B-3d · Kaggle-like lanes: 3 cards = 3 independent single-task passes ▶ next

The scorer runs one task at a time against one server (our L4 lab did too; 58 × 8 min fits the 12 h notebook cap only
sequentially). B-3b/B-3c shared one server between 3 tasks (CONC=3), each stream slowed, and the 4-min cap killed
solvable runs. CONC=1 on one server would be faithful but 3 passes take ~5–6 h. **`scripts/lane.sh`** runs a lane =
private network namespace (its own `127.0.0.1:8000`, which the harness hardcodes) + one vLLM on one card (TP=1) + one
`eval_dev.sh` at CONC=1. Three lanes on cards 0–2 = three passes in the wall time of one; card 3 stays free.

```bash
cd gemma4-dev-agent && git pull
pkill -f "vllm serve"; sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv   # the host server holds all 4 cards
bash scripts/lane.sh check                   # 📷 must end with "namespace check passed"
GPU=0 bash scripts/lane.sh smoke             # 📷 the "--- warm:" line = one card's decode speed (~5 min incl. model load)
```
**Set the cap:** `TIME_MIN = 8 × 29 / <warm tok/s>`, rounded **up** to a whole minute (29 = L4's smoke speed, probe v3;
same instrument). E.g. 58 tok/s → 4; 45 → 5.2 → 6.
```bash
for g in 0 1 2; do GPU=$g TAG=k$((g+1)) TIME_MIN=<N> REFS="base base_w5 base_r2 base_r3" SUBMISSION=runs/base_lb024 ARMS=base \
  nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done            # ~1.5–2.5 h, all three at once
```
**After the run (📷):**
```bash
grep -h "concurrency\|vLLM up" runs/lane[012].log                 # 1. each lane: "concurrency 1" + its server came up
python3 scripts/compare_runs.py runs/dev33_base runs/dev33_base_w5 runs/dev33_base_r2 runs/dev33_base_r3 runs/dev33_base_k1 runs/dev33_base_k2 runs/dev33_base_k3
python3 scripts/rulestats.py runs/dev33_base_r2 runs/dev33_base_k1 runs/dev33_base_k2 runs/dev33_base_k3 | grep -E "^ |resolved$|^resolved|timeout|decode tok/s"
```
**Read:** k1–k3 timeouts should fall to L4's 1–2 per pass (r2 had 6). k1–k3 are the reference every later arm is measured
against (an arm = the same 3 lanes, pooled: 99 pairs vs 99). If `check` fails (user namespaces disabled), the fallback
is patching the harness's api_base: photo `grep -rn "127.0.0.1:8000" .venv-eval/lib/python3*/site-packages/ | head`.

## B-4 · Repeat the best 2 configs (~2–3 h) ✖ dropped: nothing to repeat. New protocol: an arm = 3 passes pooled vs the 4 base passes

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
- **Small-model bonus (E4B/E2B):** planned separately. On the box (or a rented
  32 GB card): `MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TP=1 GPU_UTIL=0.9 bash scripts/serve_vllm.sh` serves it
  as `gemma-4-e4b-it-qat-w4a16-ct`, and the eval scripts check the alias that the submission's agent.yaml declares.
- **Budget arms** (`nothink`, output caps, thinking budget): settled by ab1/ab2, so not repeated.

## Time budget
~1–1.5 h per arm at TIME_MIN=4 (33 tasks). B-1…B-4 ≈ 10–12 box hours over 2–3 sessions. Each step needs one 📷 set.
