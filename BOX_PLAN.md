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

## B-3c · Why the flip tasks flip (no GPU, ~1 min) ✅ 2026-10-21: 10 of 17 flip losses are the 4-min cap; wins on the same tasks at 244–246 s → calibrate TIME_MIN first (B-3d)

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

## B-3d · Kaggle-like lanes: 3 cards = 3 independent single-task passes ✅ 2026-10-21: k1 12 · k2 10 · k3 9 (7 base passes: 11 12 11 9 12 10 9); each lane 51 tok/s, 0.1 s/call; timeouts 6 → 3–5 but overflows 5 → 6–8 at the still-uncalibrated TIME_MIN=4; requests 1/4 → 0/4 in every lane (offline?) → B-3e

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
Measured so far by the trace fit (L4 = 34): Oct-12 CONC=1 TP=4 run 28 tok/s, this week's CONC=3 run 46 per stream, so the
old 62-tok/s assumption (idle smoke) was wrong and TIME_MIN=4 was ~30 % tighter than L4. After the first lane pass, check
the fit's "decode tok/s" row for the lanes and re-derive: `TIME_MIN = 8 × 34 / <lane fit tok/s>`.
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

## B-3e · The reference at the calibrated cap, and the offline check ▶ next

What the lanes told us (📷): every lane decodes **51 tok/s** with 0.1 s/call overhead (the shared server gave 46 and
0.9), so the three cards are independent and the shape is right. But the budget line still read `--max-time-minutes 4`
(the smoke figure, like every smoke figure, overstated the real rate), so the cap stays ~25 % tighter than L4: 240 s ×
51 ≈ 12K generated tokens vs L4's 480 s × 34 ≈ 16K. Timeouts fell (6 → 3 · 5 · 3), overflows rose (5 → 7 · 6 · 8): the
budget deaths just moved from one wall to the other, ~11 per pass either way. **Calibrated cap: `TIME_MIN = 8 × 34 / 51
= 5.3 → 5`** (300 s × 51 ≈ 15K, 94 % of L4; 6 would be 113 %).

Second thing: **requests went 1/4 → 0/4 in all three lanes** (one task that resolved in every one of the 5 host passes).
The namespace has no network at all; the scorer's Kaggle kernel is offline too, so if that task's tests need the net
it is unwinnable on Kaggle as well and the lanes are the *more* faithful setup — but confirm it is the network and not
something else about the namespace before trusting the lanes:

```bash
cd gemma4-dev-agent && git pull
T=<the requests task every host pass solved, from the compare table>   # private copy of this plan has the id
python3 scripts/taskdiff.py $T runs/dev33_base_r2 runs/dev33_base_k1 runs/dev33_base_k2 runs/dev33_base_k3   # 📷 (~25 lines)
grep -ohE "https?://[^/ \"']+" runs/dev33_base_k1/test_outputs/$T.log | sort | uniq -c | sort -rn | head   # 📷 which hosts failed
```
**Result 2026-10-21 (📷):** r2 ✓ with a 925 B patch and 0 network phrases; k1 ✗ / k2 T / k3 ✗ with near-identical patches
(536 / 924 / 930 B) and **104 network-failure phrases in each lane's `test_outputs/<task>.log`**, 0 in the trace. So the
agent's fix was the same and the *verification* died offline: the task's tests need the net, it is unwinnable in any
offline scorer, and the lanes are the faithful instrument. One line left to read: the host grep above. External hosts
(httpbin.org, example.com, …) → done, note it. `127.0.0.1` / `localhost` → the namespace broke the local httpbin
fixture (lo was up, so unexpected) — tell the laptop; it would bias all 4 requests tasks, not just this one.
**Host grep result (📷):** only documentation links (github.com ×4, urllib3.readthedocs.io, docs.pytest.org) — the failing
tests name no host, so the 104 phrases are tracebacks without a URL. One more line, no GPU, run it *while k4–k6 run*:
```bash
L=runs/dev33_base_k1/test_outputs/$T.log
grep -oE "Name or service not known|Temporary failure in name resolution|NewConnectionError|Max retries exceeded|Network is unreachable|ConnectionError|ConnectTimeout" $L | sort | uniq -c
grep -E "^(FAILED|ERROR) " $L | cut -c1-110 | head -12; grep -cE "^(FAILED|ERROR) " $L runs/dev33_base_r2/test_outputs/$T.log
grep -m2 -B3 "Name or service not known\|NewConnectionError" $L | cut -c1-160                                   # 📷 all of it
```
**Read:** `Name or service not known` / `Temporary failure in name resolution` with a hostname in the -B3 lines → DNS, an
external host, lanes are right. Failing tests from `test_lowlevel.py` or the `httpbin` fixture, or `127.0.0.1` in the
-B3 lines → the namespace broke local servers; the fix goes into lane.sh (likely `/etc/hosts` for the box hostname) and
only the requests tasks of k4–k6 need a re-read, not the whole pass.

Then the reference at the right cap, three lanes again (card 3 stays free; can start right after the photo above):
```bash
pkill -f "vllm serve"; sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv
for g in 0 1 2; do GPU=$g TAG=k$((g+4)) TIME_MIN=5 REFS="base_k1 base_k2 base_k3" SUBMISSION=runs/base_lb024 ARMS=base \
  nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done            # ~2–3 h (longer cap)
```
**After the run (📷):**
```bash
grep -h "concurrency\|vLLM up" runs/lane[012].log                                      # must say --max-time-minutes 5
python3 scripts/compare_runs.py runs/dev33_base_k1 runs/dev33_base_k2 runs/dev33_base_k3 runs/dev33_base_k4 runs/dev33_base_k5 runs/dev33_base_k6
python3 scripts/rulestats.py runs/dev33_base_k1 runs/dev33_base_k4 runs/dev33_base_k5 runs/dev33_base_k6 | grep -E "^ |resolved$|^resolved|timeout|decode tok/s|seconds"
```
**Read:** timeouts should be L4's 1–2 per pass, overflows ~9–11 (L4 pub had 11). k4–k6 are the reference every later
arm is measured against: an arm = the same 3 lanes at TIME_MIN=5, pooled, 99 pairs vs 99 (McNemar).
**Also photo (new lines at the end of compare_runs.py after `git pull`):** `resolved in every run: 6: <ids>` and `resolved in no run: … <ids>`. The laptop's ceiling run (B-3g) needs those ids to confirm its inferred split (Opus solved exactly the 6 tasks every Gemma pass solved, and none of the 6 no-pass ever did).

## B-3f · Ceiling run: Claude Opus 5.5 through the same harness, same budget, dev33 ✖ not needed (2026-10-21): B-3g's uncapped pass never touched the Kaggle budget (max 21 calls · 250 s · 24.8K prompt), so its 20/33 is also the capped answer. Nothing to run here

**Question:** which dev33 tasks can *any* agent solve under the scorer's budget (28 calls · 8 min · 240 s per command ·
300 s verify · the 6 tools · 32K context) and the LB-0.24 prompt? Seven base passes solve 15 distinct tasks (6 in every
pass). A strong model under the identical harness splits the other 18 into *reachable* (Gemma's prompt/tool arms should
aim there) and *harness-bound* (budget, tools or the task itself — stop tuning for them). diag4 already settled the
test side (31/33 valid; one requests task needs the net, one rich task's verify exceeds 300 s). This is a measurement,
one pass, not an arm. Forum thread 747168 claims the ceiling is the harness; this checks it on our split.

**Setup — built by the Claude Code session on the box, not by the laptop:** the harness hardcodes `api_base
http://127.0.0.1:8000/v1`; the three lanes own that address only inside their namespaces, so the host's port 8000 is
free while k4–k6 run. Put an OpenAI-compatible proxy there (litellm proxy is the obvious choice) that maps the model name
the harness sends to `anthropic/claude-opus-5-5`, answers `/v1/models`, passes tool calls both ways, and drops the
Gemma-only request fields (thinking / `chat_template_kwargs`, `drop_params`). Then the normal eval on the host:
```bash
CONC=1 MAX_CALLS=28 TIME_MIN=8 CMD_TIMEOUT=240 VERIFY_TIMEOUT=300 TAG=ceil ARMS=base \
  REFS="base_k1 base_k2 base_k3" SUBMISSION=runs/base_lb024 VLLM_LOG=runs/ceil/proxy.log \
  nohup bash scripts/eval_dev.sh > runs/eval_dev_ceil.log 2>&1 &          # ~2–3.5 h; verify is CPU, it coexists with the lanes
```
Rules for the box session:
- The key stays in the shell environment of the proxy process (`os.environ/ANTHROPIC_API_KEY` in the config). Never in
  the config file literally, never in `runs/*.log`, never in anything under git. Config and proxy log live in `runs/ceil/`.
- Spend guard on the proxy (`max_budget` ≈ 80 USD) and per-request logging, so the total is readable afterwards.
- Decide and **print one line** into `runs/eval_dev_ceil.log` before the first task: where the 32K context cap is
  enforced. If the harness enforces it (its own config / compaction threshold), the ceiling run has Gemma's context
  budget — the comparison we want. If only the vLLM server enforced it (`max_model_len`), the proxy run is effectively
  uncapped at 32K: still useful, but then the line must say so and the read below changes (a task Opus solves with a
  60K prompt is not "reachable" for Gemma). Same for `max_output_tokens`.
- TIME_MIN=8 literally (API speed is not the box's decode speed; a generous cap is right for a ceiling).
- Traces contain task text: they stay on the box; and every task's text goes to the Anthropic API during the run — the
  user has accepted that for a private measurement; nothing of it is published.

**Simpler route, verified on the laptop 2026-10-21 (no proxy):** the harness's `--models-yaml` entries take a litellm
`path` and `api_base`, and the registry silently overrides the submission's alias, so `scripts/eval_dev.sh` now has a
`MODELS_YAML` switch: it skips the local-server checks, passes `--models-yaml`, loads `scripts/ceil_sitecustomize.py`
(drops the Gemma-only `extra_body`, sets Anthropic `thinking: adaptive`, adds prompt-cache markers, logs every call's
tokens to `runs/ceil/calls_<TAG>.jsonl`) and prints the effective values. The key comes from `LITELLM_API_KEY` (shell
env only). `runs/ceil/models.yaml` (gitignored) is three lines:
```yaml
models:
  gemma-4-31b-it-qat-w4a16-ct:          # the alias in the submission's agent.yaml
    path: anthropic/claude-opus-5-5
    api_base: https://api.anthropic.com
```
```bash
export LITELLM_API_KEY="$ANTHROPIC_API_KEY"     # never echo it, never put it in a file under git
source .venv-eval/bin/activate
MODELS_YAML=runs/ceil/models.yaml CONC=1 MAX_CALLS=28 TIME_MIN=8 CMD_TIMEOUT=240 VERIFY_TIMEOUT=300 TAG=ceil ARMS=base \
  REFS="base_k1 base_k2 base_k3" SUBMISSION=runs/base_lb024 VLLM_LOG=runs/ceil/none.log \
  nohup bash scripts/eval_dev.sh > runs/eval_dev_ceil.log 2>&1 &
```
**Context cap, settled:** swegemma itself enforces no context limit — the 32K cap is the vLLM server's `max_model_len`
(the ContextWindowExceeded errors come from the server). So a remote-model run has the model's own window; B-3f reads as
the *calls + time* ceiling with context as a known difference (an approximate 32K guard in the adapter is possible but
not built). Two probe tasks on the laptop: resolved in 25 s / 4–5 calls each, prompt cache read ≈ 80 % of input.

**Pre-flight (📷 one photo, ~6 lines):**
```bash
git check-ignore -v runs/ceil/proxy.yaml && ! grep -qE "sk-ant|ANTHROPIC_API_KEY=" runs/ceil/proxy.yaml && echo "config ignored, no literal key"
curl -s 127.0.0.1:8000/v1/models | head -c 200; echo                       # the model name the harness will send
python3 scripts/smoke.py | grep -E "^--- "                                 # tool-call request parsed through the proxy
grep -m1 -i "context cap" runs/eval_dev_ceil.log                           # the enforcement line, once the eval has started
sleep 600; wc -l runs/dev33_base_ceil/task_results.jsonl; tail -c 300 runs/dev33_base_ceil/task_results.jsonl   # first task ended, no parameter error
```
If the first task's `error` mentions an unsupported parameter, a 4xx from the proxy, or tool calls arriving as plain
text, stop the eval, fix the proxy, delete `runs/dev33_base_ceil`, restart.

**After the run (📷):**
```bash
grep -h "concurrency\|context cap" runs/eval_dev_ceil.log                                        # 1. budget line says 8 min, CONC 1
python3 scripts/compare_runs.py runs/dev33_base_k1 runs/dev33_base_k2 runs/dev33_base_k3 runs/dev33_base_k4 runs/dev33_base_k5 runs/dev33_base_k6 runs/dev33_base_ceil   # 2. the differ list
python3 scripts/rulestats.py runs/dev33_base_k1 runs/dev33_base_ceil | grep -E "^ |resolved$|^resolved|timeout|tool calls|first edit|ran a test|decode"   # 3.
<total spend from the proxy: litellm /global/spend or a sum over its log>                            # 4. one number, USD
```
**Read:** (a) `ceil` resolved vs the 6-pass union: tasks Opus solves that no base pass ever did = the *reachable* set →
the next arms are judged on those tasks specifically (do they move?). (b) Tasks Opus also fails: which wall (overflow /
time / wrong patch / no patch, from the differ marks and rulestats) → harness-bound; name them in the run log and stop
designing for them. (c) Opus's calls / first-edit / tested rows vs Gemma's tell how a stronger agent spends the same 28
calls — the shape of a trajectory worth imitating in the prompt. (d) If `ceil` is below the union of base passes,
the proxy or the prompt mismatch is the problem, not the ceiling; check the error column before reading anything else.

## B-3g · Uncapped ceiling: Opus 5.5, budget lifted ✅ 2026-10-21 (laptop, 11 min, $6–19): **20/33 = 20 of the 24 tasks whose gold patch passes here** · 4 genuine wrong patches · 0 budget deaths (max 21 calls · 250 s · 24.8K prompt) · calls p50 6 · first edit at call 3 · 30 s/task p50. Gemma's 7 passes: 16 distinct, all inside Opus's 20; 4 reachable tasks (2 fastapi · 2 rich; ids in the private log). Per-task table: `python3 scripts/tasktable.py runs/dev33_base_ceilx`. Gold check → B-3h

B-3f says *who* is reachable under the scorer's budget. This pass says *how much* budget each task really needs and what a
solution looks like, including for the tasks B-3f still fails. Same proxy, same prompt, same tools; only the caps move:
```bash
CONC=3 MAX_CALLS=80 TIME_MIN=30 CMD_TIMEOUT=240 VERIFY_TIMEOUT=900 TAG=ceilx ARMS=base \
  REFS="base_k1 base_ceil" SUBMISSION=runs/base_lb024 VLLM_LOG=runs/ceil/proxy.log \
  nohup bash scripts/eval_dev.sh > runs/eval_dev_ceilx.log 2>&1 &         # CONC=3 is fine here: no GPU, the API scales
```
Box session: if the 32K context cap / compaction threshold / max output are harness-side (the B-3f line), lift them
to the model's real window (≥ 128K, output 16K) for this pass and print the effective values the same way; if they
were server-side only, nothing to lift and the line says so. Keep CMD_TIMEOUT at 240 s (a single command is a task-level
fact, not an agent budget); VERIFY_TIMEOUT 900 tells whether the rich task that fails verification at 300 s is solvable
at all. Spend: longer trajectories, expect 1.5–2× B-3f; raise the proxy budget accordingly before starting. ~1.5–2 h wall.

**After the run (📷):**
```bash
grep -h "concurrency\|context cap" runs/eval_dev_ceilx.log
python3 scripts/compare_runs.py runs/dev33_base_k1 runs/dev33_base_ceil runs/dev33_base_ceilx          # capped vs uncapped Opus, Gemma for scale
python3 scripts/rulestats.py runs/dev33_base_k1 runs/dev33_base_ceil runs/dev33_base_ceilx | grep -E "^ |resolved$|^resolved|timeout|tool calls|first edit|ran a test|never edited"
<total spend>
```
**Read:** three bands. (1) Solved in B-3f already: the Kaggle budget suffices; Gemma's loss on them is the agent. (2) Solved
only uncapped: read their calls / seconds / peak prompt in ceilx — if they end at 30–40 calls or 40K prompt the budget is
the wall and no prompt arm will reach them on Gemma; if they end just past 28 calls or 8 min, a tighter prompt might.
(3) Unsolved even uncapped: the task or the tool set, not the budget — list them as dead for this competition. Per-task
"how was it solved" (files read, file edited, tests run, call of the first edit, Opus vs Gemma side by side) needs a
per-task table script (`tasktable.py`, laptop writes it after "go"); until then `flips.py` on `k1 ceil ceilx` gives the
per-pass line for every task that differs. The ceilx traces are also the only teacher trajectories we own, should SFT
ever come back on the table (forum: no gain at the top; not planned).

## B-3h · Validity audit: does the gold patch pass on the box? (no GPU, ~5 min) ▶ still wanted (expect the same nine) — laptop dissection below

On the laptop the dataset's own patch fails verification for **9 of 33** dev tasks under the subprocess sandbox
(`scripts/goldcheck.py`): one needs `orjson` (in no wheel set), two new fastapi tasks need starlette 1.x while google-adk
caps the harness venv at <1 (the sandbox venv inherits it), three requests tasks and three rich tasks fail the same single
test with and without any patch (dependency/Python-version behaviour; pinning urllib3 to the scorer's 2.7.0 did not fix
the requests ones). Gemma solved none of those nine in seven box passes, so the box is probably the same: dev33 has been
dev24. Every rate so far should be read over 24, and 9 of the "never" tasks are not agent failures.
```bash
cd gemma4-dev-agent && git pull
.venv-eval/bin/python scripts/goldcheck.py $(cat splits/dev33.txt)       # 📷 the 33 result lines (CONC=3 default)
```
**Read:** `NOT resolved` = invalid on the box. If the set matches the laptop's nine (3 fastapi · 3 requests · 3 rich), the
fix is the scorer's own sandbox: `bash scripts/build_sandbox_image.sh` (x86_64 wheels match the box; needs docker without
sudo), then `SANDBOX=docker OUT=runs/goldcheck_docker .venv-eval/bin/python scripts/goldcheck.py $(cat splits/dev33.txt)` 📷.
31–33 resolved there → all later runs use `SANDBOX=docker` (lane.sh passes env through) and k4–k6 are re-run under it as
the reference; the subprocess numbers stay as history. If docker is not available on the box, say so: the laptop can
build the image under emulation and we decide then.

**Laptop result 2026-10-22 (Insight 29 — the nine dissected by hand, one dependency swapped at a time):** the reference patches
are correct; the failures are ours or the harness's. (1) 3 requests: the subprocess sandbox skips both the editable install
and the `workspace_paths.pth` the Docker sandbox writes (workspace + `src/`), and `PYTHONSAFEPATH=1` drops the cwd, so a
src-layout repo's tests import the harness venv's own requests — **no requests patch, gold or agent, has ever been tested
under this sandbox, on the laptop or on the box** (flat repos are fine: pytest prepends the root conftest's directory).
`PYTHONPATH=src` → pass. (2) 3 rich: pygments 2.19.2 passes two, ≤2.18 the third; 2.21.0 (our venv and the highest public
wheel) fails all three — the public Docker image most likely fails them too; the Docker goldcheck decides. (3) 1 fastapi:
needs orjson + ujson (then 9/9), in no wheel set. (4) 2 fastapi: the failing tests are `async def` + `@pytest.mark.anyio`
and the harness runs `-p no:anyio` → "async def functions are not natively supported" in every environment; unwinnable.
So dev33 ≤ 31 valid anywhere, 28 if the scorer's pygments is 2.21. B-3h's box goldcheck is still wanted (same nine expected).

## B-3i · Make the subprocess sandbox faithful, re-verify saved patches — laptop ✅ 2026-10-22 · box ▶ next (no GPU, ~10 min)

**Laptop result:** shim + orjson/ujson in; goldcheck on the nine → 2 flip to RESOLVED (the orjson task, the src-layout requests task); 2 requests tasks need werkzeug 2.3.8 (the older public wheel; 3.1.x raises on port −1 inside the test server — drift, like pygments), 3 rich + 2 async fastapi unchanged. `reverify.py` on the Opus ceiling run: **20 → 21/33** — 2 gains, and **1 loss: the one requests task every Gemma run has ever "solved" was the harness venv's own newer requests passing the hidden tests; Opus's patch is wrong there** (off-by-one), so expect the box's requests column to drop to 0/4 on re-verification. Real Opus ceiling: 21 of the 26 tasks valid here; 5 genuine misses.
```bash
cd gemma4-dev-agent && git pull && bash scripts/install_testdeps.sh                       # prints orjson/ujson versions
.venv-eval/bin/python scripts/goldcheck.py $(cat splits/dev33.txt)                          # 📷 33 lines + the "sandbox shim:" line; expect 26 RESOLVED
for k in 1 2 3 4 5 6; do .venv-eval/bin/python scripts/reverify.py runs/dev33_base_k$k; done  # 📷 the "== … changed" blocks
python3 scripts/compare_runs.py runs/dev33_base_k1 runs/dev33_base_k1_rv runs/dev33_base_k2 runs/dev33_base_k2_rv runs/dev33_base_k3 runs/dev33_base_k3_rv   # 📷
```
**Read:** only requests rows may change (expect the lone requests win to disappear; everything else identical). From here every run
has the shim (eval_dev.sh prints `sandbox shim: … on`), so new runs need no re-verification. B-3h (Docker goldcheck) still decides
pygments/werkzeug: the denominator is 26 here, 31 at best in Docker, 24–26 if the public image shares the drift.

**B-3i-L · LENIENT sandbox mode ✅ 2026-10-22 (laptop) — analysis only, never the reference.** `LENIENT=1` makes
`scripts/sandbox_shim.py` pin the per-task venv of the 5 dependency-drift tasks to the versions their tests were written
against (3 rich → pygments 2.18.0 / 2.19.2, 2 requests → werkzeug 2.3.8 + flask 2.3.3; an explicit verified table — a
date rule does not work), installed offline from `harness/wheels/` + `runs/lenient_wheels/` (fetched by
`install_testdeps.sh`). Prints `sandbox mode: LENIENT …` and one `lenient: <task> → <pins>` line per task; goldcheck writes
to `runs/goldcheck_lenient`, reverify to `<run>_rvl`, eval_dev.sh needs a TAG containing `len`; every such dir gets
`lenient.txt` and compare_runs.py marks it `L`. Result: gold patch RESOLVED on all 5 (the 2 async fastapi tasks stay
unfixable: `-p no:anyio` is a harness flag, no pin). Use it to ask "is this patch right?" (ceiling runs, trajectories);
reference numbers stay strict because the scorer is strict.
```bash
LENIENT=1 .venv-eval/bin/python scripts/goldcheck.py $(cat splits/dev33.txt)      # expect 31 RESOLVED (26 strict + 5)
LENIENT=1 .venv-eval/bin/python scripts/reverify.py runs/dev33_base_ceilx          # → _rvl; compare with _rv
```

*Original step list:*

1. `scripts/sandbox_shim.py` (loaded like the ceiling adapter, via the PYTHONPATH `sitecustomize`): after the per-task venv
   is created, write `_workspace.pth` = workspace + `workspace/src` into its site-packages — what `harness/sandbox/setup.py`'s
   fast path does in Docker — and print one line `sandbox shim: .pth → <workspace>, <workspace>/src` into the run log;
   `eval_dev.sh`, `lane.sh`, `goldcheck.py` all load it. Verification-only entry point `scripts/reverify.py <run>`: re-runs
   `verify_task` on a run's saved `patches/` (no model calls) → `runs/<run>_rv/` with the same result files.
2. `install_testdeps.sh` adds `orjson ujson` (box has pip access; print their versions).
3. Laptop: `goldcheck.py` on the nine → expect 3 requests + the orjson task to flip to RESOLVED; 3 rich + 2 async stay
   NOT (📷 not needed, it's local). Then `reverify.py runs/dev33_base_ceilx` → the Opus ceiling over the 4 requests tasks for
   real; log both.
4. Box, after `git pull`: `goldcheck.py` on the nine (📷 9 lines), then `reverify.py` on `runs/dev33_base_k1..k6` (📷 the
   compare table, k vs k_rv) — the requests column is the only thing that may move; everything else must be identical.
5. Pygments / the two async tasks: no local fix. The box's Docker goldcheck (B-3h) says what the scorer does; the
   accounting denominator becomes 31 or 28 accordingly, and `tasktable.py`'s band split excludes the invalid set.

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
