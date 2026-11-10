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

**Latest setting (since 2026-10-26).** Every arm except `sample` starts from `scripts/arms/latest/`: sampling, eval_config
**and prompts/system.md** (= the old base prompt + 3 edits: `git status --short && git diff` before submitting, "keep the
current behaviour for every input the issue does not mention", pytest `2>&1 | tail -40`). Arm parts apply on top. So
`base` from 2026-10-26 on is not the earlier `base`: compare arms only with the `base` of the same run. The run prints
`latest prompt: … (7144 chars)`; `python3 scripts/latest_diff.py runs/subs/<arm>` shows any arm's delta.

**Compaction (since 2026-10-27).** swegemma's CLI never set `events_compaction_config`, so every box run before this
date had no compaction (the history only grew to the 32K window); the scorer compacts. `scripts/compaction_shim.py`
(loaded with the sandbox shim) now gives every `swegemma eval` the scorer's config: interval 15 · overlap 2 · threshold
14,336 · retention 5; the summary is a call to the same model (ADK's default prompt, no temperature). The log prints
`compaction: on …` at the start and one `compaction: #n summarised k events …` line per summary; `traj/` marks them
`compaction: true` and `walkthrough.py --brief` shows them as `COMPACTION`. `COMPACTION=0` = the old runs. Runs before
and after this date are not comparable either: compare only within one run.

**Cost report (since 2026-10-27, every run).** At the end of each eval `scripts/run_costs.py <run>` prints 📷: where
the tokens go (prompt tokens processed, the fixed prefix per call, model output and thinking, tool outputs once and
re-paid in later prompts, harness nudges, compaction calls) · tool outputs by tool (n, ≈tok sum/mean/p50/p90/max,
wall seconds) · every compaction with context before → after · the 10 slowest tool calls · the 10 largest outputs · time
per task (model vs tools). Tool times come from `traj/<task>.tools.jsonl` (traj_log.py now times each tool call); older
runs show sizes only. Every single call: `<run>/costs_calls.tsv`; one task call by call: `run_costs.py <run> --task <id>`.

**Decision rule** (dev33 noise is about ±2 tasks): an arm counts as better only if it solves **≥ 2 more tasks than
`base` in the same run, with ≥ 2 arm-only tasks** in the pairs. Otherwise drop it. An arm that loses ≥ 2 tasks is
dropped too.

---

## B-4 · Token-level distillation 31B → E4B, step S0: does the 31B point at E4B's own errors? (2026-11-10) — ✓ S0 done 2026-11-10 · ⏸ next: S0b (code not pushed yet)

**Status (2026-11-10).** S0 ran: of 88 not-found edits 56 were locatable; the 31B's top-1 continued the source at the first wrong
token in 2 % (call) and 12 % (thought) → "stop" by the rule below. But most measured positions repeat a wrong text that is
already in the context (48 of 56 calls; 39 of 48 thoughts), where any model copies; only 9 were where the text is first
written. **Next, S0b** (laptop, not pushed yet): measure where each wrong text is first written in the trajectory (resends once),
set apart texts written before the source was shown and sources whose continuation is not unique, add "the teacher prefers the
right token", score E4B too (where the distillation signal lands), and re-measure the several-matches and run-before-submit lines;
thresholds unchanged. On the box it will be `git pull` → `bash train/opd_box.sh student` → `bash train/opd_box.sh probe` (the 31B
scores are reused). **Do not rerun S0 and do not start S1.** Keep `runs/opd/s0/` (rows, recs, t31_*.npz).

### S0 as it ran (kept as the record)

**Idea.** E4B runs tasks itself; the 31B gives its top-20 next-token distribution at every token E4B wrote (thought, text,
tool call); a LoRA is then trained toward it (reverse KL on E4B's own tokens: no format gap, the thought gets a signal
too). Before any training code: does the 31B, at the first wrong token of E4B's failed `old_string` (in the call and in
the thought where E4B first wrote it), rank E4B's token low and the right text first? No → the line stops. Inputs: E4B's
own rebench rollouts from B-3p step 4 (`runs/dev33_base_e1..e3`, failed tasks included) and `runs/rebench/tasks.jsonl`.
Nothing else may use the cards during `score` (it takes every card; run B-3q-T before or after, not at the same time).

```bash
# 0. code, cards, disk, inputs (📷): the 31B QAT (~22 GB) must be in the HF cache or downloadable
cd gemma4-dev-agent && git pull && git log --oneline -1 && ls train/opd_box.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; df -h . | tail -1
ls ~/.cache/huggingface/hub | grep -i 31b; ls -d runs/dev33_base_e[123]/traj runs/rebench/tasks.jsonl
# 1. records + rows (CPU, ~5–10 min) 📷 every line from 'opd_box s0data' down
bash train/opd_box.sh s0data
# 2. the 31B scores every E4B reply token, one process per card (~20–40 min; progress: tail -2 runs/opd/s0/score_0.log)
nohup bash train/opd_box.sh score > runs/opd_score.log 2>&1 &
tail -25 runs/opd_score.log            # when it ends: 📷 the '== teacher_score' lines and the '== score' line
# 3. the readout (CPU, ~5 min) 📷 every line from the first 'counts:' line down
bash train/opd_box.sh probe
```

**Stop and send the photo** on: a missing input in step 0, less than ~30 GB free while the 31B is not cached (the user
decides what to delete), `render` not PASS, a card in use, a shard with no result, or a `b/c check` line where the oracle
count differs from the cases (the measurement itself is broken).

**Reading rule (written before the data).** `b` = the first wrong token in the CALL of a not-found edit, `c` = the same
wrong text where it first appears in the THOUGHT (`born here` = not in an earlier thought). "corrected" = the 31B's top-1
token continues the text the tool output showed.
- **go to S1** (training code, then two adapters + an A/B): corrected ≥ 50 % in `b` overall AND ≥ 50 % in `c born here`
  (if `c born here` has n < 20, `b` alone decides);
- **stop the line**: corrected < 30 % in both `b` and `c` (the 31B follows E4B's wrong text instead of the source);
- in between: the laptop decides from the photo.
- `d` (closing a too-short old_string), `e` (tool choice: how often the 31B would pick another tool) and `f` (where the
  31B would end the thought) are not go/stop lines: they set S1's choices (which tokens get the loss).
Nothing is pushed from the box. Append each step's lines to `runs/B4_RUN_LOG.md`.

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

## B-3g-SC · Opus 5.5 under Gemma's harness limits ✅ 2026-10-28 (laptop, 8.5 min): **22/33** · 0 overflows · 0 call-cap hits · 1 compaction
Same harness limits as a Gemma run, time lifted (user: calls, compaction, 32K the same; no time limit): latest setting (0 diffs), 28 calls, the scorer's events compaction (compaction_shim.py), and Gemma's 32K window emulated in `scripts/ceil_sitecustomize.py` (`CEIL_GEMMA_CTX`): every request is counted with Gemma 4's own tokenizer + chat template (fetched once into `runs/gemma_tok`; equals vLLM's prompt_tokens exactly on 823 E4B calls), prompt + max_output_tokens > 32,768 raises vLLM's own ContextWindowExceededError, and the reply's prompt_tokens are relabelled to the Gemma count so the 14,336 compaction threshold fires where it would for Gemma (Opus's own count runs ≈ 1.5× higher). Result: the limits never bind — max prompt 14.4K Gemma tokens, calls p50 6, first edit at call 3 — and the score matches the uncapped B-3g (20/33; 2 flips, both gains). The Gemma gap on dev33 is the model, not the harness limits. Not matched: speed (2–4 s/call) and thinking (adaptive, not a 4,096 budget).
```bash
eval "$(sed -n 141p ~/.zshrc | grep -E '^export ANTHROPIC_APIKEY=')"
PATH="$PWD/.venv-eval/bin:$PATH" LITELLM_API_KEY="$ANTHROPIC_APIKEY" SUBMISSION=runs/base_lb024 MODELS_YAML=runs/ceil/models.yaml \
  CONC=3 MAX_CALLS=28 TIME_MIN=40 CMD_TIMEOUT=240 VERIFY_TIMEOUT=900 TAG=ceilsc ARMS=base REFS="base_ceilx" \
  bash scripts/eval_dev.sh > runs/eval_dev_ceilsc.log 2>&1      # CEIL_GEMMA_CTX=32768 is the default; 0 = the old uncapped window
```

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

### B-3h-2 · All 129 tasks, gold AND empty patch → the A/B split `splits/disc_e4b.txt` (no GPU, ~15 min) ▶ next, any time no lane runs
A task can separate two arms only if the gold patch passes **and** the empty patch fails (forum 756400 found 5 of 129 that
pass with no patch). The laptop has the gold half for all 129 (112 pass) and the empty half for dev33 only (0 of 32 pass;
two rest96 tasks are already known to pass empty from agent runs). `scripts/disc_tasks.py` (laptop) combines that with every
local run and keeps the 51 tasks some arm can still move: E4B solved some of the time (19), 31B solved but E4B never (24),
E4B always (8, regression guard). They take ~1/3 of an all-129 E4B run, so **an E4B A/B from now on = `SPLIT=splits/disc_e4b.txt`,
3 seeds per arm** (the time of one all-129 run per arm); never-solved and invalid tasks only cost time.
```bash
cd gemma4-dev-agent && git pull
bash scripts/validcheck.sh            # 📷 the two == lines + the two lists (CONC 8; lanes running → CONC=4, slower tests)
```
**Read:** the laptop puts the box's `gold NOT` and `empty RESOLVED` ids into `disc_tasks.py` (BOX_GOLD_FAIL / BOX_EMPTY_PASS)
and rewrites the split if anything moves; expected: the gold list ≈ the laptop's 17 (a task that fails only here is dropped here too), the empty list = the 2
known + 0–3 new.

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

## B-3j · Opus pass@3 on the genuine misses, passing-trajectory index ✅ 2026-10-22 (laptop, LENIENT, ~9 min, ≈$3–10)

`SPLIT=<file>` (new in eval_dev.sh, prints `tasks: N from <file>`) runs a subset; three passes with TAG lenp1–3 at the B-3g
budget. Per pass **1 · 2 · 3 / 6**: the requests task 3/3 (the ceiling run's miss was one bad sample), one rich task 2/3,
one 1/3, one fastapi + two rich tasks 0/3 — all finished in 40–220 s, no budget deaths, so these three are not budget-bound
either; they are the hard core (k=4 and still nothing). With the ceiling run, **28 of 33 tasks have a verified-passing
Opus trajectory** (= 28 of the 31 pin-valid tasks). `scripts/passing_traces.py <run>…` indexes them: `runs/passing/<task>/
<run>.json` symlinks + `runs/passing_traces.json`, re-verified dirs (`_rv`, `_rvl`) resolve back to the original traces,
lenient verdicts marked `L`. Local only (task text inside). Pass@k is cheap (≈$1–3 a pass on 6 tasks); the next lever for
the 3 hard tasks is a different sampling (Fable 5.1, or a prompt variant), not more of the same.
```bash
python3 scripts/passing_traces.py runs/dev33_base_ceilx_rvl runs/dev33_base_lenp1 runs/dev33_base_lenp2 runs/dev33_base_lenp3
```

**The 3 hard-core tasks dissected (2026-10-22, gold vs the 4 Opus attempts):** all three issues are a bare title + an offline
link. One is unreachable by anyone: the gold adds a new 200-line script whose exported names and subcommands exist only in the
hidden test (every attempt dies at import). The other two are strategy misses, not missing information: on one, Opus wrote the
exact gold line once, saw no change in a probe that did not exercise the edge case (empty cells), reverted it, and later
submitted with 4 previously-passing tests red; on the other it fixed the caller instead of the layer that drops the data,
although it had already found the test calling that layer. Attempts took 40 s–3.5 min, far under budget. So 30 of the 31
pin-valid dev tasks are reachable; Opus has 28. Candidate prompt rules → arms: **R1** never submit while a previously-passing
test fails; **R2** fix at the layer where the data is lost and run every existing test of that layer; **R3** bare-title issue →
enumerate edge inputs (empty / trailing / single element) before choosing a hypothesis. Private detail: notes/08.

## B-3l · The course slides as one test case, `course` ▶ today (~2–3 h, 3 lanes)

`course` = every executable item from the course slides (lecture 3), one arm part; it grows as more slides arrive and
the run uses whatever `git pull` brings. Now: `t00+out6k+time7+tail40+outline+win100+grepbatch+twins5+callchain+
dryhelper+noregress+ask3+editctx+editguard+tmprepro`:
- config: `t00` temperature 0.0 · `out6k` max_output_tokens 6144 · `time7` eval_config 8 → 7 min, run flag TIME_MIN ×
  7/8 (5 → 4.375, printed as a `time7:` line);
- prompt: `grepbatch` (3-call locate SOP: batched `git grep -n -E 'A|B|C' -- '*.py' | head -40` + find tests → def/class
  outline → ranged reads), `outline` (ast skeleton for > ~300-line files, brace-free), `win100` (60–100-line windows),
  `twins5` (the 5 twin patterns as a checklist), `callchain` (follow a new/changed parameter through callers, callees and
  registration tables), `dryhelper` (one shared helper for all twin entry points), `noregress` (unmentioned inputs keep
  behaviour, `__all__` complete), `ask3` (3 caller/callee questions before editing: who calls me, what do I call,
  did unmentioned inputs change), `editctx` (old_string with 2–3 lines of context, right after reading it),
  `editguard` (fallback script: assert count == 1, `compile()` before writing, then `git diff`),
  `tmprepro` (repro scripts in /tmp/repro.py via a run_command heredoc; write_file rejects /tmp), `tail40` (pytest `| tail -40` with `set -o pipefail;` so the exit code stays pytest's).
Thinking 4096, include_thoughts, 28 calls, 240 s, graph-tool ban, edit recovery, /tmp and `git status && git diff` are
base's already. Prior evidence for the config half predicts no gain (6144 alone tied/lost; temperature 0.0 was the forum
sweep's lowest row; the notebook author's v5 with 7 min + 6144 + tail -40 scored 0.13); the prompt half is mostly new,
`callchain` aims at the largest measured gap (no dev task with a multi-file gold patch solved). Same protocol as every arm
since B-3e: 3 lanes at TIME_MIN=5, pooled, 99 pairs vs the k4–k6 base reference.

```bash
cd gemma4-dev-agent && git pull
ls -d runs/dev33_base_k4 runs/dev33_base_k5 runs/dev33_base_k6      # 📷 — all three must exist (B-3e); if not, run B-3e first
BUILD_ONLY=1 SUBMISSION=runs/base_lb024 ARMS=course TAG=chk bash scripts/eval_dev.sh | grep -E "^course|time7:|: (replaced|inserted|appended)|^=="   # 📷
pkill -f "vllm serve"; sleep 5
for g in 0 1 2; do GPU=$g TAG=l$((g+1)) TIME_MIN=5 REFS="base_k4 base_k5 base_k6" SUBMISSION=runs/base_lb024 \
  ARMS=course nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done                              # ~2–3 h
```
**Check before reading results (📷):** `course = …` lists the 15 parts; the `==` line shows `temperature: 0.0 …
max_output_tokens: 6144 thinking_budget: 4096 include_thoughts: true` and `prompt ~11470 chars`; `time7:` shows
`max_time_minutes 7 · run flag --max-time-minutes 4.375`; 12 rule lines say `replaced` / `inserted`. Any "appended
(anchor not found)" → stop and photo.
**After (📷):**
```bash
python3 scripts/compare_runs.py runs/dev33_base_k4 runs/dev33_base_k5 runs/dev33_base_k6 runs/dev33_course_l1 runs/dev33_course_l2 runs/dev33_course_l3
python3 scripts/rulestats.py runs/dev33_base_k4 runs/dev33_course_l1 runs/dev33_course_l2 runs/dev33_course_l3 | grep -E "^ |resolved|timeout|seconds"
python3 scripts/locstats.py runs/dev33_base_k4 runs/dev33_course_l1 runs/dev33_course_l2 runs/dev33_course_l3 | grep -E "^==|read|edited|unresolved that"
python3 - <<'X'                                     # runs whose tool calls used the ast outline (the prompt text itself is not counted)
import glob, json
for d in ["runs/dev33_course_l1", "runs/dev33_course_l2", "runs/dev33_course_l3"]:
    fs = glob.glob(d + "/traces/*.json")
    n = sum(any("import ast" in str(s.get("tool_calls", "")) for s in json.load(open(f))["steps"]) for f in fs)
    print(d, "outline used in", n, "of", len(fs), "runs")
X
```
**Trajectories (on by default since 2026-10-26; `TRAJ=0` turns it off):** every run dir now gets `traj/<task>.jsonl`
from `scripts/traj_log.py`: each Gemma call's new messages, its thinking, reply, tool calls, finish reason and tokens.
The harness trace drops the thinking. The lane log ends with one `traj …` line per run (📷): thinking in how many
replies, and **re-sent turns with thinking x/y**, which answers whether old thoughts go back to the model. One task, call
by call (📷), pick any task id from `ls runs/dev33_course_l1/traj`:
```bash
python3 scripts/walkthrough.py runs/dev33_course_l1/traj/<task>.jsonl --brief
```
**Read:** pooled resolved over 3 passes vs k4–k6. Overflows should fall (6144), timeouts may rise (7/8 of the minutes,
longer prompt); locstats' "first edit" call should move earlier if the SOP helps (base: call 14). Adopt only if the pooled
difference is ≥ +3 tasks with more winning than losing pairs. Round 2 (next free 3 lanes, `ARMS=callchain`): the one
rule aimed at the multi-file gap, alone on base, to see whether it carries any course effect.
**Optional comparison (user 2026-10-23: one version, `course`): `ARMS=coursefull`** (dirs `runs/dev33_coursefull_l1..3`) — the course's
own "final deliverable": prompts/system.md replaced by its ~20-line prompt verbatim (1,037 chars,
`scripts/arms/course_system.md`) + `t00+out6k+time7`. Build check: `coursefull = scripts/arms/course_system.md … +
t00+out6k+time7`, `==` line `prompt 1037 chars`, temperature 0.0, 6144, `time7:` 4.375. The submission candidate is `course`
only (it contains every slide rule; coursefull is the final slide's condensed prompt).

## B-3m · Failure-study prompt fixes on E4B, all 129 tasks → folded into B-3o (the same four rules on the 51-task split, 4 passes)

The failure study (2026-11-01: an Opus judge on every failed run of E2B, E4B and 31B, plus contrast A on the tasks 31B
solved and a small model failed) found that the small models:
- **never reproduce before editing:** 0 of 45 / 36 such tasks, while 31B reproduced in about 2/3;
- **edit at call 5 vs 31B's 12, and rarely look up the definition;**
- **misread their own failing tests:** E4B in 90% of its failures.

Four prompt rules came out of it. Each is one arm part (`scripts/arms/<name>.md`), they stack, and every one prints
`replaced` / `inserted` in the log:
- `reprofirst` (f): the first call reproduces the issue with `python3 -c`, the same command runs again after the edit,
  never submit without it. It replaces the "you may run it once" line.
- `failstop` (d): pytest exit ≠ 0, exit 5 (no tests ran), or an Import-/Indentation-/SyntaxError from an edited file
  = a broken patch, never "the environment". Fix or revert it, never submit it.
- `grepdef` (g): `git grep -n -E 'def NAME|class NAME'`, read ~60 lines around the definition, never a whole file.
- `latedit` (h): calls 1–10 reproduce / locate / read, edit at 11–16 (deadline 12 → 16, no "best-guess edit now"),
  verify at 17–22.

**Model and split:**
- E4B through the small-model latest setting with **no adapter** (`LATEST=e4b ADAPTER=`), so the prompt is the only
  difference; the server needs no LoRA.
- **All 129 tasks,** because dev33 is too small for E4B's ~4–6 solves.
- `TIME_MIN=8`, so the run flag is the scorer's 7 min (E4B decodes faster here than on an L4: lenient on time, the
  same for every lane).

**Four lanes, one pass each:**
| lane | arm |
|---|---|
| GPU 0 `e1` | `base` (control) |
| GPU 1 `e2` | `reprofirst+failstop` (the verification pair) |
| GPU 2 `e3` | `reprofirst+failstop+grepdef+latedit` (all four) |
| GPU 3 `e4` | `base` again (noise: how far two identical passes differ) |

```bash
cd gemma4-dev-agent && git pull
cat splits/dev33.txt splits/rest96.txt > runs/all129.txt && wc -l runs/all129.txt                 # 📷 129
.venv-eval/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('google/gemma-4-E4B-it-qat-w4a16-ct'))"   # once, on the host (lanes are offline)
BUILD_ONLY=1 LATEST=e4b ADAPTER= SUBMISSION=runs/base_lb024 ARMS="base reprofirst+failstop reprofirst+failstop+grepdef+latedit" \
  TAG=chk bash scripts/eval_dev.sh | grep -E "latest set|: (replaced|inserted|appended)|^=="           # 📷 6 rule lines, 3 "==" lines
pkill -f "vllm serve"; sleep 5
A=(base reprofirst+failstop reprofirst+failstop+grepdef+latedit base)
for g in 0 1 2 3; do GPU=$g TAG=e$((g+1)) TIME_MIN=8 LATEST=e4b ADAPTER= MODEL=google/gemma-4-E4B-it-qat-w4a16-ct \
  SPLIT=runs/all129.txt SUBMISSION=runs/base_lb024 ARMS="${A[$g]}" nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
```
After it finishes:
```bash
python3 scripts/compare_runs.py runs/dev33_base_e1 runs/dev33_base_e4 runs/dev33_reprofirst+failstop_e2 runs/dev33_reprofirst+failstop+grepdef+latedit_e3   # 📷
python3 scripts/fail_atlas.py runs/dev33_base_e1 runs/dev33_base_e4 runs/dev33_reprofirst+failstop_e2 runs/dev33_reprofirst+failstop+grepdef+latedit_e3 | grep -E "^==|outcomes|failures:"   # 📷
for r in runs/dev33_base_e1 runs/dev33_reprofirst+failstop_e2 runs/dev33_reprofirst+failstop+grepdef+latedit_e3; do echo "== $r"; python3 scripts/run_costs.py $r | head -12; done   # 📷
```
**Read it as:**
- an arm counts only if it beats both base passes by more than they differ from each other;
- the atlas line shows whether the rules were followed (tests run, edit position).

Reference points:
- E4B under boris v6's prompt, same 129 tasks: 15.
- E2B under our prompt: 6.

## B-3n · Our own native 31B training data on public SWE-rebench tasks ▶ next (~1 h setup + 3–5 h, 4 lanes)

**Why.** The public LoRA kit (dmitriigluzdov, CC BY 4.0) trains on 103 trajectories of Gemma 4 31B on 81 public
SWE-rebench tasks (none of them competition tasks), run under the Budget-Fit prompt. Converted to our format
(`train/convert_kit.py`), 2/3 of them break a rule of our prompt (scratch files in the repo), so we had to rewrite
them. Running the same tasks through **our latest setting** gives trajectories whose prompt, tools and budget are
exactly what our agent sees: no re-wrap, no rewriting. It is the C4a data ("a closer teacher") with no overlap with
our 129 tasks.

**What is the kit's and what is ours:**
| | the kit (2026-10-20) | B-3n |
|---|---|---|
| tasks | 81 = 61 nebius/SWE-rebench + 20 whitecircle/swe-rebench-v2 (`splits/rebench_kit81.txt`) | the same 81, rows from the same pinned revisions |
| model / serving | 31B INT4, official vLLM | the same |
| agent | Budget-Fit 0.18 prompt, 9 tools, 28 calls / 8 min | **our latest 31B setting** (`scripts/arms/latest/`), 9 tools, 28 calls / 7 min, scorer compaction |
| sampling | temp 0.65, seeds 42 + 43 (2 passes) | arm `t07` (temp 0.7, the plan's), **4 passes** (one lane each) |
| sandbox / grading | "local subprocess grader" | `--sandbox subprocess` + `REBENCH_SITE` per-task dependencies, gold-checked |
| result | 103 resolved on 55 tasks | ? |

How the kit installed each repo's dependencies is not published, so `scripts/rebench_build.py` does its own: the
project + its `install_config.pip_packages` + `reqs_path` files via `uv pip install --target runs/rebench/site/<id>`
for the eval venv's Python (3.12), put on the sandbox's `sys.path` right after the workspace (`REBENCH_SITE`, in
`scripts/sandbox_shim.py`). Tasks written for old Pythons may not run on 3.12; the gold check drops them.

```bash
cd gemma4-dev-agent && git pull
bash scripts/rebench_gen.sh build     # internet; rows (≈ 310 MB parquet, HF cache) + 81 snapshots + deps   📷 failures + TOTAL
bash scripts/rebench_gen.sh gold      # no GPU; gold patch per task → runs/rebench/ok.txt                     📷 "== gold" + the NOT list
bash scripts/rebench_gen.sh gen       # stops any host vLLM, 4 lanes (GPUs 0–3) × 1 pass each, offline        📷 the 4 "pass" lines
tail -2 runs/rebench_lane*.log        # progress
bash scripts/rebench_gen.sh data      # after all 4 lanes: resolved → records → render → parser → data_stats  📷 all of it
```
Knobs: `IDS_FILE` `OUT` `ARM` `PASSES` `GPUS` `TIME_MIN` `MAX_TURNS` `MODEL` `CONC_GOLD` `TOK` (the script prints them).
`MAX_TURNS=80` (default) puts "- Max loop iterations: 80 turns" into the task prompt, as the scorer does from our
eval_config.yaml (found 2026-11-03 reviewing the kit data: our local latest-setting runs never passed it, so their
prompts lacked that line).
`TIME_MIN=7` is our latest's literal number; the box decodes faster than an L4 (B-3d: 51 vs 34 tok/s), so this is more
lenient than the scorer — fine for collecting solved trajectories, not for reading a score.

**Read it as:**
- `build`: rows found 81/81; snapshot or deps failures name the task. `deps partial` = the project itself would not
  install (its imports still come from the workspace).
- `gold`: how many of the 81 our sandbox can grade at all (the kit's subprocess grader resolved 55 with the model).
- `data`: resolved trajectories and tasks (kit: 103 on 55), render `RESULT PASS`, A5 excluded ≈ 0, the data_stats
  lines (identical-call rate, same-tool rate, thought length; the kit's numbers are in the private run log 2026-11-03).

**Next (laptop decides after the photos):** train E4B on `runs/sft/rebench31b.rendered.jsonl` on the box
(a `train_box.sh` stage) and test it against no adapter on all 129 tasks.

## B-3o · Step 1 of the E4B gap plan: scaffold arms, no training, the 51-task split ✓ done 2026-11-08: no arm counts

**Result (2026-11-08, 4 passes each):** base 14.0 (12 · 14 · 16 · 14) · mech 13.5 (13 · 12 · 16 · 13) · all 14.0 (15 · 14 · 13 · 14) / 51 →
neither reaches base + 3; B-3p uses the base prompt. RESULTS.md has the row.

Why (5-agent laptop study, 2026-11-07, private notes): of the 31 tasks 31B solves and E4B misses, E4B finds the right
file and writes the wrong fix in 17, picks the wrong file or place in 11, and loses 3 to failed edits; the budget, the
compaction and cut-off replies decide none. Unlike 31B it edits at call 5 (31B 12), never reproduces the issue first
(0/129 vs 56/129), runs pytest commands that collect no test in about half its calls, submits after a red test, and
leaves ~17 of its 28 calls unused. Its edit_file fails 47 % (31B 10 %): old_string is copied badly from lines shown
several outputs back (13 % failures when the lines are in the last output, 45 % at 4+ outputs back) or is not unique.
No LoRA so far beat no adapter, so this step changes only the prompt and the tool list.

Three arms, E4B with **no adapter**, each **4 passes** (one per lane), all in this one session:
| arm | what changes |
|---|---|
| `base` | nothing (control) |
| `tools6+spec+edit2` ("mech") | drop the 3 graph tools (E4B called them 2× in 129 tasks; −700 tokens per call) · the issue text again at the end of the system prompt, so it survives compaction (`{problem_description}`, filled per task by ADK) · re-read the ~20 lines right before each edit and copy old_string from that output, 1–3 lines with a unique name; on "occurs N times" add the nearest def/class line |
| `tools6+spec+edit2+reprofirst+failstop+grepdef+latedit` ("all") | mech + the four B-3m rules from 31B's routine: reproduce first and again after the edit · a red/empty test = a broken patch · grep the definition · edit at calls 11–16 |

Each lane runs all three arms in a rotated order (time of day hits every arm alike); `MAX_TURNS=80` = the scorer's
prompt line, `TIME_MIN=8` as in B-3m.
```bash
cd gemma4-dev-agent && git pull
M=tools6+spec+edit2; A=$M+reprofirst+failstop+grepdef+latedit
BUILD_ONLY=1 LATEST=e4b ADAPTER= SUBMISSION=runs/base_lb024 SPLIT=splits/disc_e4b.txt ARMS="base $M $A" TAG=chk \
  bash scripts/eval_dev.sh | grep -E "latest set|tasks:|: (replaced|inserted|appended)|^spec:|^=="   # 📷 51 tasks · edit2/edit2b/spec lines · tools 9/6/6
pkill -f "vllm serve"; sleep 5
O=("base $M $A" "$M $A base" "$A base $M" "base $A $M")
for g in 0 1 2 3; do GPU=$g TAG=p$((g+1)) TIME_MIN=8 MAX_TURNS=80 LATEST=e4b ADAPTER= MODEL=google/gemma-4-E4B-it-qat-w4a16-ct \
  SPLIT=splits/disc_e4b.txt SUBMISSION=runs/base_lb024 ARMS="${O[$g]}" nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
```
After all four lanes finish:
```bash
for a in base $M $A; do python3 -c "import json,sys; c=[sum(json.loads(l)['resolved'] for l in open(f'runs/dev33_{sys.argv[1]}_p{i}/task_results.jsonl')) for i in (1,2,3,4)]; print(f'{sys.argv[1][:40]:<40}', c, 'mean', sum(c)/4)" "$a"; done   # 📷
python3 scripts/rulestats.py runs/dev33_base_p1 runs/dev33_${M}_p1 runs/dev33_${A}_p1   # 📷 did E4B follow the rules (edits, tests, first edit)
python3 scripts/compare_runs.py runs/dev33_base_p{1,2,3,4} | head -8                     # 📷 ctx✗ / errors of the control
```
**Read it as:** an arm counts only if its 4-pass mean beats base's by ≥ 3 tasks and its worst pass ≥ base's best − 1
(passes of one arm differ by up to 5–6 tasks on these 51). Next on the laptop either way: step 2 (an edit-skill LoRA
on short, checked edit exercises, no whole 31B trajectories) and, later, step 3 (31B relabels E4B's own states).

## B-3q · The edit-skill LoRA again: drills rebuilt from E4B's real failures, gate in tiers ▶ steps 1–6 done 2026-11-10: gate 1 not passed (epoch 1: 6 of 8 lines) → rebuilt as **B-3r** (first subsection below: start there)

B-3p's two runs did not pass gate 1 (B-3p below: kept as the record, **do not run its steps**). What changed for B-3q, all in the code:
- **Drills:** a `retry` family from E4B's real failure loops (a failed edit by E4B's real causes, the identical call resent,
  a re-read followed by the same call), LLM-like misremembered lines, re-reads whose range no longer gives the answer away.
- **Training mix:** 3,600 drills with more anchor / block / retry (`TRAIN_FAMILIES`, ≤ 30 per file, ≤ 5 per identical line of a library): 1,489 of them make the
  model choose a unique anchor while a repeat is in view (478 of 1,500 before; ~1,500 is the estimate for several matches
  ≤ 10 %). The held-out sets keep the standard family weights.
- **Gate:** read by tier, paired against gate0 on the same drills (exact McNemar). The 31B-level gate (hit ≥ 80 %, several
  ≤ 3 %) is printed beside it but no longer decides.
- **Training:** one learning rate, 1e-5, four epochs (in B-3p 3e-5 made the block slice worse with every epoch; 1e-5 was
  still improving at epoch 2). Every epoch's adapter goes through gate1.
- **E4B's rollouts are reused** (`runs/sft/rebenche4b.rendered.jsonl` from B-3p's step 4): no new rollouts.

### B-3q-T · Is the adapter's gain on the drills tied to histories without thoughts? (2026-11-10) — START HERE: one command, about 15 min, no training

**Why.** In a real run every earlier model turn reaches the model as a thought followed by its tool call (E4B's run on
the 129 tasks: 1,899 of 1,906 earlier thoughts are in the rendered prompt at its edit calls). In the drills every earlier
turn is the bare call: 27,932 earlier turns, no thought. In B-3q's training mix 88 % of the sequences were such drills and
the rest E4B's own rollouts with thoughts, so the history format tells the two apart. If the adapter learned "drill
behaviour when the history has no thoughts", its gate-1 gain cannot carry over to real tasks — and the gate, which
decodes in the drill format, cannot see it.

**The test.** B-3q's three held-out sets, as they are on the box, in three history formats: `orig` (as built),
`template` (one or two plain sentences before each earlier call, made from the call itself), `written` (the served E4B,
no adapter, writes one to four sentences before each earlier call). Targets, answers and checks do not change. Each
format is decoded at temperature 0 with no adapter and with `e4b_drill_q` epoch 1, and scored as in gate 1.

Needs code ≥ `ebe850e`, all cards free, B-3q's files (nothing is rebuilt, nothing is deleted; new files:
`runs/sft/drills_heldout*_ht`, `*_hw`, `runs/drills/hist_*`, about 100 MB).

```bash
cd gemma4-dev-agent && git status --short train/ && git pull && git log --oneline -1
ls -d runs/sft/drills_heldout runs/sft/drills_heldout_block runs/sft/drills_heldout_retry runs/sft_out/e4b_drill_q/epoch1/adapter_model.safetensors && df -h . | tail -1
GPUS=0,1,2,3,4,5,6,7 NAME=e4b_drill_q EPOCH=1 bash train/drills_box.sh histtest 2>&1 | tee runs/drills_histtest.log
grep -E "^== gate|^   (intended|several_matches):" runs/drills/hist_main_orig_ad.txt        # the sanity line below
```
(4 cards: `GPUS=0,1,2,3`. Do not put `SET=` in front: this test runs on B-3q's sets.)

**📷** the `mounted:` line, the six `histthink` lines, the 18 `decoded` lines, the whole `edit_drills histcmp` block, and
the sanity lines of the last command.

**Three checks before the result is read** (stop and 📷 if one fails):
1. `mounted:` names `e4b_drill_q_epoch1`.
2. In the `decoded` lines, `prompt tokens mean` is higher for `template` and `written` than for `orig` of the same set
   (by roughly 150–600 tokens). Equal numbers mean the server did not put the thoughts into the prompt: the test says
   nothing then.
3. The sanity lines repeat gate 1 for epoch 1 on the main set: hit 60 → 67 %, several matches 21 → 14 % (same drills,
   same temperature; a point or two of difference is fine).
Also read the `written` histthink lines: `written` should be most of the turns. If most fell back to `template`
(`written_empty` or `written_error` large), say so in the photo's caption; the `template` format still counts.

**How it is read** (fixed before the run; printed by `histcmp` as the last word of each `orig → …` line):
- **KEPT** — with thoughts in the history the adapter keeps at least two thirds of its original gain, on hit and on
  several matches. The history format is not what stops a transfer; the drill rebuild goes on without it.
- **LOST** — at most one third is kept on both. The gain belongs to the drill format: drills need thoughts on the
  earlier turns before any further training, and B-3q's epoch-1 A/B is not worth its two hours.
- **unclear** — in between, or the two measures disagree. The pooled 460 drills give each gain a noise of about ± 2
  points and the original gain is about 5–7, so this can happen; then `written` counts more than `template` (it is
  closer to a real history), and the answer-length column is read as well (the adapter moved the median from 1 line
  to 3 in the original format: does it still?).

**Then stop.** Append the lines to `runs/B3Q_RUN_LOG.md`. B-3r below stays on hold; the epoch-1 A/B (B-3q after gate 1,
step 3) still only on the user's word.

### B-3r · The drills rebuilt so that every answer can be decided from the context (2026-11-10) — ⏸ HOLD: do not start step 1 yet

**⏸ Hold (2026-11-10, laptop audit of the built drills):** the edit after the check can be right without reading the
check's output. Taking one more line above and one below the failed old_string gives a unique match in 100 % of the
block check drills (three more lines each side: 94–99 % in every family). In E4B's real runs the same move — more lines
after "occurs N times" — worked 4 of 31 times. The check drills are far easier than the real case, so neither the
training nor the gate could tell whether the grep was used. The builder is to be changed before the box builds; this
line is removed when the fix is pushed, and nothing below runs until then.

**Why.** B-3q's readouts (all on this box, 2026-11-10):
- In 82 % of the block drills and 41 % of the anchor drills the lines shown did not let the model know how long the
  old_string had to be: the context held one copy of the repeated line and at most the bare line of the other copy,
  while the answer's length depends on the lines **around** the other copy. The 31B did not edit there either (49 % of
  its samples grepped or read other lines first); E4B guessed (block: 4 % right).
- In E4B's real runs (all 129 tasks) 148 of 205 failed edits are further tries after a first failure. After "occurs N
  times" it edited again at once 49 of 67 times (38 several again) and grepped once.
- Thoughts and trained calls did not agree: a thought was kept when the sample's own call gave the right file, but the
  trained call stayed the builder's answer.

**What changed (code ≥ `cccbce0`; nothing to patch on the box):**
- **Builder:** around every other copy of the changed line, the lines that sit where the answer sits are in the context
  before an edit target: in the view, in a view widened to them, or in a `grep -nF -C 5` of the line. Sites that cannot
  be shown that way are rejected (`check_span`, `check_many`, `check_quote` in the build's `rejected:` line).
- **New two-step drill:** after a several-matches failure with the copies out of sight, target 1 is that grep (the
  check) and target 2 the edit, with the grep's output in its context. Before, target 1 read the same lines again.
- **Gate:** the block slice's line is now the check — after a several-matches failure, a check (or the right edit at
  once) ≥ +20 points over gate 0 — and its several-matches rate may not rise. The other lines are unchanged.
- **think:** an edit target takes the call of the sample its thought comes from (`THINK_LABEL=own`, printed).
- **Families:** `copy=0.22,anchor=0.08,recover=0.19,stale=0.05,block=0.27,retry=0.19` (fewer first-try anchor drills, more
  drills that start after a failure).
- **Checked on the laptop:** on the four sets built with these commands, the shortest span that is unique among the
  lines shown is unique in the file in 98 % of the block drills (16 % in B-3q), 100 % of the anchor drills (55 %) and
  99.9–100 % of the others; scoring, the tier, `think` and the render pass on synthetic decodes; gate0 and think ran end
  to end on fake servers with `SET`. Not run on a GPU: everything from step 2 on.
- **`SET=_r` on every command:** this build's files go to `runs/sft/drills_r*` and `runs/drills_r/`; B-3q's files and
  its adapter `e4b_drill_q` stay where they are.

```bash
ALL=0,1,2,3,4,5,6,7                      # 8 cards.  4 cards: ALL=0,1,2,3
```

**Step 0 · pull and check**
```bash
cd gemma4-dev-agent && git status --short train/ && git pull && git log --oneline -1
grep -c "CHECK_CONTEXT" train/edit_drills.py && ls runs/sft/rebenche4b.rendered.jsonl runs/dequant/gemma-4-E4B-dq/config.json && df -h . | tail -1
```
Go if: the pull is clean, the grep prints a number ≥ 1, both files exist, ≥ 8 GB free.

**Step 1 · build** (CPU, one process, about 1–1.5 h)
```bash
SET=_r nohup bash train/drills_box.sh build > runs/drills_r_build.log 2>&1 &
grep -cE "^== drills" runs/drills_r_build.log            # progress: 4 when done
grep -E "families built|^== drills|rejected:" runs/drills_r_build.log | cut -c1-400    # 📷 these
grep -oE '"(check_chain|dup_seen_[a-z_]+)": [0-9]+' runs/drills_r_build.log | paste -sd' ' -   # 📷 and this
```
Go if: four `== drills` lines with 3600 · 300 · 80 · 80 records. The laptop's build of the same commands gave, for
comparison (the box's library files differ a little): training anchor 231 · block 383 · copy 1,270 · recover 791 ·
retry 773 · stale 152 with 419 two-step check drills (5,023 targets: 3,600 edits · 1,004 re-reads · 419 checks); held-out
300 with 23 check drills, block slice 80 with 35, retry slice 80 with 10. The training build's line "block libraries used
up after ~380 block drills; the rest without them" is expected. Stop and 📷 only if a `== drills` line shows fewer records
than asked, or `check_chain` is 0 in the training set or the block slice.

**Step 2 · gate 0** (all cards, a few minutes)
```bash
SET=_r GPUS=$ALL bash train/drills_box.sh gate0 2>&1 | tee runs/drills_r_gate0.log
```
📷 from `edit_drills score` down, with the `check after a several-matches failure` lines (main set and block slice).

**Step 3 · E4B's thoughts** (all cards, about 25 min)
```bash
SET=_r GPUS=$ALL nohup bash train/drills_box.sh think > runs/drills_r_think.log 2>&1 &
tail -8 runs/drills_r_think.log
```
📷 the decode line, `samples:`, `calls with a thought by family`, the `== think` line (it now says how many edit targets
carry the sample's own call) and the two score lines.

**Step 4 · mix** (CPU, 2 min)
```bash
SET=_r bash train/drills_box.sh mix
```
📷 the render lines, `drills kept by family`, `== mix`. Stop if a render is not PASS.

**Step 5 · train** (all cards; 2 epochs — in B-3q epoch 1 was the best and later ones worse)
```bash
SET=_r GPUS=$ALL LR=1e-5 EPOCHS=2 NAME=e4b_drill_r nohup bash train/drills_box.sh train > runs/drills_r_train.log 2>&1 &
sleep 600; grep -hE "ranks joined|schedule|step +[0-9]+/" runs/drills_r_train.log | tail -5
```
10-minute check, 📷: `N ranks joined` with N = the number of cards, and the step lines' speed-up. 8 ranks have not run on
real cards yet (4 did: ×3.8–3.9). Speed-up below ×5 on 8 cards, a Traceback, or a checksum WARNING → stop it
(`pkill -f sft_lora.py`) and start it again with `GPUS=0,1,2,3`. Estimate: 1.5–2.5 h on 8 cards, 3–3.5 h on 4.
At the end 📷 the summary lines (time, held-out NLL per epoch, checksum line, export check).

**Step 6 · gate 1** (all cards, about 10 min)
```bash
SET=_r GPUS=$ALL NAMES=e4b_drill_r bash train/drills_box.sh gate1 2>&1 | tee runs/drills_r_gate1.log
grep -E "^== tier" runs/drills_r_gate1.log
```
📷 each epoch's block, the `block slice` lines with `check after a several-matches failure`, and the whole tier block.

**Step 7 · stop, 📷.** The A/B (which adapters, how many passes) is decided on the laptop from gate 1. Do not run `subs`,
an A/B or anything on B-3q's files without a new line here. Append every step to `runs/B3R_RUN_LOG.md` as before.

### B-3q after gate 1 (2026-11-10) — steps 1–2 done 2026-11-10; step 3 (the epoch-1 A/B) still only on the user's word; B-3r above comes first

Where it stands (steps 0–6 ran 2026-11-09 → 11-10, 8 cards):
- **gate 1, adapter A:** no epoch passed the A/B tier. Epoch 1 passed 6 of the 8 lines (hit 60 → 67 %, several matches
  21 → 14 %, both significant); the two missed lines are the 80-drill slices (block several −8 against −12, retry hit +3
  against +10). Later epochs are worse (several back to 20 %, answers back from 3 lines to 2).
- **adapter B:** not trained. The 31B gave a usable thought for 9.7 % of the targets E4B left empty; of its samples with a
  usable thought, 62 % called another tool instead of the edit.
- **several-card training** worked: 4 cards, ×3.8–3.9 of one card, identical weights on every rank, 6.3 h.

**1. Pull.** The box's local patch (`serve ""` mounted the sample submission, whose adapter is the 31B's) is in the repo
now: `serve ""` mounts the empty `runs/noadapter` and prints a `mounted:` line. Drop the local copy first:
```bash
cd gemma4-dev-agent && git checkout train/drills_box.sh && git pull && git log --oneline -1
```

**2. What the 31B (and E4B) did instead of the edit** — no GPU, reads the decodes that are already there, writes nothing
but the two text files. A minute or two each:
```bash
.venv-eval/bin/python train/edit_drills.py why runs/sft/drills_native runs/drills/train_think31_t07.jsonl 2>&1 | grep -v LiteLLM | tee runs/drills/why_think31.txt
.venv-eval/bin/python train/edit_drills.py why runs/sft/drills runs/drills/train_think_t07.jsonl --max-thought-chars 12000 2>&1 | grep -v LiteLLM | tee runs/drills/why_think.txt
```
📷 both outputs whole (from `edit_drills why` down to `== why done`). They show: thoughts missing / over the cap / usable,
the outcome by thought state, how many targets a higher cap would fill, which tool was called instead of the edit (a
read of the same lines again, of other lines, the whole file, a grep …), and all of it by family.

**3. Only when the user at the box says "run the A/B": the exploratory A/B with epoch 1** (8 cards, ~2 h). No epoch
passed the tier, so this is outside the gate rule and its result is reported as exploratory; it answers whether the
drill gain carries over to real tasks at all.
- **Disk first.** About 20 GB are needed (the box's estimate), 12 GB were free. Show the user
  `df -h . | tail -1; du -sh runs/* ~/.cache/huggingface/hub/* 2>/dev/null | sort -h | tail -15` and delete only what
  the user names. Keep `runs/dequant` (every retraining needs it), `runs/sft*`, `runs/drills` and `runs/adapters`. The 31B
  download of step 3b is the natural candidate (not used by this A/B, can be downloaded again).
- **The adapter** (`EPOCH=1` is required: without a passing tier `subs` stops):
  ```bash
  EPOCH=1 NAME=e4b_drill_q bash train/drills_box.sh subs
  ```
  📷 the subs lines. Go if the build check passes.
- **The A/B:** the step-8 command block further down, unchanged (`NAME=e4b_drill_q`, `LANES=(0 1 2 3 4 5 6 7); PASSES="1"`),
  then its readout lines. 📷 the per-arm resolved counts, both `edit_stats` blocks, `rulestats`, the drill-template count.

**4. Stop.** Not without a new plan from the laptop: retraining, more thought sampling, a `think31` rerun with another
cap, another teacher model, any other deletion. Append what ran to `runs/B3Q_RUN_LOG.md` as before.

### B-3q run order on the box (2026-11-09, code ≥ `614dc93` for steps 3–6: several cards) — one step at a time, 📷 after each

**Cards (2026-11-09):** `drills_box.sh` takes `GPUS=a,b,…` (default: every card). think / think31 / gate1 start one
server per card and spread the calls over them; `train` trains ONE adapter on all of its `GPUS` (the 16 sequences of a
step dealt to the cards, gradients summed: the single-card update); gate1 takes `NAMES="a b"` and decodes every epoch of
every name at once. The card sets `ALL` / `CA` / `CB` are set in the runbook's step 0.
Steps 1–2 are done on the box (training 3,600 · held-out 300 · 80 · 80; gate0 hit 60 % · several 21 % · not found 3 % ·
parsed 100 %): **do not run them again** — a rebuild would change the held-out sets and gate0 with them. The times below
are estimates from the one-card times ÷ cards (training: ÷ ~0.85 × cards); every stage prints its real rate
(`calls/s` and `eta` in decode, `×… of one card` in each training step line).

### B-3q runbook from step 3 (2026-11-09) — copy one block, wait for its end line, 📷, then the next

The table under this runbook is the same plan in one view; where they differ, the runbook is right. Every stage ends
with a `== … done (<stage>)` line. Nothing is pushed from the box. A `Traceback`, a `NOT UP`, a `FAILED` or a number
outside a block's "go if" → stop and send the photo; do not repair and rerun on the box.

**Step 0 · pull, check, set the cards** (CPU, 1 min)
```bash
cd gemma4-dev-agent && git pull && git log --oneline -1          # 614dc93 or later
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # every card free
ls runs/sft/drills | wc -l; ls runs/sft/rebenche4b.rendered.jsonl runs/drills/gate0.out.jsonl runs/drills/gate0_block.out.jsonl runs/drills/gate0_retry.out.jsonl runs/dequant/gemma-4-E4B-dq/config.json
```
Then ONE of these two lines, in every shell that runs a step (the commands below read them):
```bash
ALL=0,1,2,3,4,5,6,7; CA=0,1,2,3; CB=4,5,6,7      # the box has 8 cards: adapter A on CA, adapter B on CB
```
```bash
ALL=0,1,2,3; CA=0,1; CB=2,3                      # the box has 4 cards
```
📷 the commit line, the cards, the file lines. **Go if:** 3600 drills, all five files there (a missing file → photo;
never run `build` or `gate0` again, they are done).

**Step 3 · E4B's own thoughts into the training drills** (all cards · 8: ~40–75 min · 4: ~1.3–2.5 h)
```bash
GPUS=$ALL nohup bash train/drills_box.sh think > runs/drills_think.log 2>&1 &
grep -E "servers up|NOT UP|^  [0-9]+ /" runs/drills_think.log | tail -3     # progress: calls done, calls/s, eta
tail -8 runs/drills_think.log                                               # at the end: '== … done (think)'
```
📷 the `drills_box think` header (cards), `servers up: N of N`, the `edit_drills decode` line (N servers), `== decoded`
(per server, request errors), `== think`, `samples:`, `calls with a thought by family`, the two score lines.
**Go if:** `== decoded` reaches (nearly) all calls. Fewer servers up than cards → it goes on with those; send the photo.
There is no stop rule on how many thoughts are kept (block / anchor / retry will have few; adapter B fills them).

**Step 4 · mix for adapter A** (CPU, minutes)
```bash
bash train/drills_box.sh mix 2>&1 | tail -8
```
📷 the two render lines, `drills kept by family`, `== mix`. **Go if:** both renders say PASS. If the kept line says
drills were dropped, run `DRILL_SHARE=0.75 bash train/drills_box.sh mix 2>&1 | tail -8` and send both.

**Step 5 · train adapter A on CA** (8 cards: 4 cards ~2.5–4 h · 4 cards: 2 cards ~4.5–7 h)
```bash
GPUS=$CA LR=1e-5 EPOCHS=4 NAME=e4b_drill_q nohup bash train/drills_box.sh train > runs/drills_train_q.log 2>&1 &
# after ~10 min — the first check of the several-card trainer on the real model:
grep -hE "^drills_box|schedule|ranks joined|step +[0-9]+/|Traceback|FAILED" runs/drills_train_q.log | cut -c1-260 | head -9
# when it ends:
grep -hE "schedule|eval\[|== epoch|checksum|FAILED" runs/drills_train_q.log | cut -c1-200; tail -2 runs/drills_train_q.log
```
📷 after 10 min: the header (cards · lr 1e-5 · epochs 4 · r 16 · alpha 16 · eval-every 100), the schedule line (`cards
for this adapter N`), `N ranks joined`, two step lines (`×… of one card`, `gradient sync … s`). 📷 at the end: the
`eval[...]` lines, four `== epoch` lines (export check PASS), the `parameter checksum` line.
**Go if:** the step lines show a speed-up of at least ×1.3 and no Traceback (also in `runs/sft_e4b_drill_q.rank*.log`).
Otherwise send the photo, then fall back to one card: `pkill -f sft_lora.py; GPUS=${CA%%,*} LR=1e-5 EPOCHS=4
NAME=e4b_drill_q nohup bash train/drills_box.sh train > runs/drills_train_q.log 2>&1 &` (~8–13 h).

**Step 3b · the 31B's thoughts for the targets E4B left empty** (CB — start it as soon as step 5's 10-minute check is
good; 4 cards ~15–45 min · 2 cards ~0.5–1.5 h)
```bash
GPUS=$CB nohup bash train/drills_box.sh think31 > runs/drills_think31.log 2>&1 &
grep -E "servers up|NOT UP|^  [0-9]+ /" runs/drills_think31.log | tail -3
tail -6 runs/drills_think31.log                                             # at the end: '== … done (think31)'
```
It stops every `vllm serve` when it starts: none runs now, and step 5's training is not a vllm process.
📷 `servers up`, the decode line, `== decoded`, `== think`, `samples:`, `thought source by family`.
**Go if:** at least a third of the empty targets were filled (the `== think` line); fewer → photo before step 4b.

**Step 4b · mix for adapter B** (CPU, minutes)
```bash
NATIVE=runs/sft/drills_native31 MIX=runs/sft/drills_mix31.rendered.jsonl bash train/drills_box.sh mix 2>&1 | tail -8
```
📷 and go rule as step 4 (the same `DRILL_SHARE` as step 4 ended with).

**Step 5b · train adapter B on CB, beside step 5** (as step 5)
```bash
GPUS=$CB LR=1e-5 EPOCHS=4 NAME=e4b_drill_q31 MIX=runs/sft/drills_mix31.rendered.jsonl nohup bash train/drills_box.sh train > runs/drills_train_q31.log 2>&1 &
grep -hE "^drills_box|schedule|ranks joined|step +[0-9]+/|Traceback|FAILED" runs/drills_train_q31.log | cut -c1-260 | head -9
grep -hE "schedule|eval\[|== epoch|checksum|FAILED" runs/drills_train_q31.log | cut -c1-200; tail -2 runs/drills_train_q31.log
```
📷 and go rule as step 5 (fallback card: `${CB%%,*}`).

**Step 6 · gate1: every epoch of A and B against gate0** (all cards, after steps 5 AND 5b have ended · 8: ~15 min · 4: ~25 min)
```bash
ls runs/sft_out/e4b_drill_q/epoch*/adapter_model.safetensors runs/sft_out/e4b_drill_q31/epoch*/adapter_model.safetensors   # 8 files
GPUS=$ALL NAMES="e4b_drill_q e4b_drill_q31" bash train/drills_box.sh gate1 2>&1 | tee runs/drills_gate1.log | tail -120
grep -E "^== tier" runs/drills_gate1.log                                    # the eight verdicts in one view
```
(Only A is trained: `NAMES=e4b_drill_q`. Never start it while think31 runs: it stops every vllm serve.)
📷 `N adapters on N servers`, each adapter's `-- <name>` block (edit outcomes, gate, answer length) and its whole
`edit_drills tier` block (✓/✗ lines, `== tier`), the eight `== tier` lines together.
**Go if:** at least one `== tier` line says **A/B tier PASS**. None → stop, send the photos: the laptop decides.

**Step 7 · the adapter for the A/B** (CPU, minutes) — which NAME: the A-or-B rule under the table; the epoch is chosen
by `subs` (the passing epoch with the highest hit; `EPOCH=<n>` names one)
```bash
NAME=e4b_drill_q bash train/drills_box.sh subs          # or NAME=e4b_drill_q31
```
📷 the subs lines (epoch, adapter_config r / alpha, build check). **Go if:** the build check passes.

**Step 8 · the A/B on the 51 tasks** (8 cards ~2 h · 4 cards ~4 h): the command block under the table — 4 passes
without and 4 with the adapter. 📷 the per-arm resolved counts, both `edit_stats` blocks, `rulestats`, the
drill-template line count. Read it with the rule under that block.

**After step 8** (decided on the laptop from the photos, nothing more runs on the box by itself):
- the adapter counts by the rule → it becomes the E4B setting's adapter candidate; the loser of A / B can be put
  through steps 7–8 as a second A/B if it also passed the tier;
- gate1 passed but the A/B shows no change in `edit_stats` → the drills' skill does not carry over to real tasks:
  the first suspect is the drills' instruction wording (about a dozen sentence patterns, a third of them one pattern);
- gate1 not passed → the per-family lines of the photos say which family did not move; no rerun before that is read.


| # | step | card(s) · time | 📷 | stop / go |
|---|---|---|---|---|
| 0 | `cd gemma4-dev-agent && git pull && git log --oneline -1 && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` | — | the commit line and the cards | commit `614dc93` or later; every card of `$ALL` free |
| 1 ✔ done | `bash train/drills_box.sh build` | CPU · ~1 h | four `== drills` lines (3600 · 300 · 80 block slice · 80 retry slice) and the `families built:` line above each (the training one shows the B-3q weights) | any `Traceback`, or a set below its count → stop |
| 2 ✔ done | `GPU=0 bash train/drills_box.sh gate0` | 1 card · ~15 min | everything from `edit_drills score` down, the `-- block slice` and `-- retry slice` lines | none: this is the baseline every adapter is paired with (new held-out sets: not comparable with B-3p's 62 / 20) |
| 3 | `GPUS=$ALL nohup bash train/drills_box.sh think > runs/drills_think.log 2>&1 &` then `tail -8 runs/drills_think.log` when it ends (progress: `grep -E "servers up|/ " runs/drills_think.log \| tail -3`) | all cards · 8: ~40–75 min · 4: ~1.3–2.5 h (`THINK_SAMPLES=2` halves it) | the header (cards), `servers up: N of N`, the decode line (N servers) and `== decoded` (per server, errors), `== think`, `samples:`, `calls with a thought by family`, the two score lines | `servers up` below the card count → send the photo (it goes on with the ones that came up). **No stop rule** on the thoughts: block / anchor / retry will have few (39 % / 28 % in B-3p); they train with an empty thought (decided 2026-11-09), adapter B fills them |
| 4 | `bash train/drills_box.sh mix` | CPU · minutes | the render lines, `drills kept by family`, `== mix` | render not PASS → stop; if the kept line says drills were dropped, run it again with `DRILL_SHARE=0.75` and send both |
| 5 | `GPUS=$CA LR=1e-5 EPOCHS=4 NAME=e4b_drill_q nohup bash train/drills_box.sh train > runs/drills_train_q.log 2>&1 &`; after ~10 min `grep -hE "^drills_box|schedule|ranks joined|step +[0-9]+/" runs/drills_train_q.log \| cut -c1-260 \| head -8` 📷; when it ends `grep -hE "schedule|eval\[|== epoch|checksum|FAILED" runs/drills_train_q.log \| cut -c1-200` | CA · 4 cards: ~2.5–4 h · 2 cards: ~4.5–7 h (one card: ~8–13 h) | the header (cards · lr 1e-5 · epochs 4 · r 16 · alpha 16 · eval-every 100), the schedule line (`cards for this adapter N`), `N ranks joined`, a step line (`×… of one card`, `gradient sync … s`), the `eval[...]` lines, four `== epoch` lines, the `parameter checksum` line | a `Traceback` in `runs/drills_train_q.log` or `runs/sft_e4b_drill_q.rank*.log`, `TRAIN FAILED`, or a speed-up below ×1.3 → send the photo and run it on one card (`GPUS=${CA%%,*}`); a checksum WARNING → send the photo |
| 6 | after steps 5 and 5b have both ended: `GPUS=$ALL NAMES="e4b_drill_q e4b_drill_q31" bash train/drills_box.sh gate1` (adapter A alone, earlier: `GPUS=$CA NAMES=e4b_drill_q …`; it stops every vllm serve, so not while think31 runs) | all cards · 8 adapters at once · 8 cards: ~15 min · 4: ~25 min | `N adapters on N servers`, each adapter's `-- <name>` block and its whole `edit_drills tier` block (✓/✗ lines, `== tier`) | go only for an adapter whose `== tier` line says **A/B tier PASS**; several epochs of one name pass → the one with the highest hit; none → stop, send the photos |
| 7 | `NAME=e4b_drill_q bash train/drills_box.sh subs` (it takes the epoch that passed the A/B tier with the highest hit; `EPOCH=<n>` names one) | CPU · minutes | the subs lines (epoch, adapter_config r/alpha, build check) | build check fails → stop |
| 8 | the A/B (commands below): no adapter vs the adapter, 4 passes per arm on the 51 tasks (8 cards: 8 lanes × 1 pass; 4 cards: 4 lanes × 2 passes) | 8 cards · ~2 h · 4 cards · ~4 h | the per-arm resolved counts, both `edit_stats` blocks, `rulestats`, the drill-template line count | read it with the rule under the commands |

**Adapter B — the 31B's thoughts on the hard targets (2026-11-09, optional but wanted; same drills, one more variable):**
E4B's own thought exists only where E4B already answers right, so the hardest calls (block / anchor / retry) train after
an empty thought. Adapter B fills those from the 31B (open weights, so its outputs may be trained on): the 31B is
sampled on exactly the targets still empty, a thought is kept by the same rule (only from a sample whose own call was the
checked answer, ≤ 3,000 chars), and on those calls the thought is trained too (label scope full) — E4B cannot write it
yet, so as context alone it would not be there at inference. Everything else is adapter A's. Steps, after step 3:

| # | step | card(s) · time | 📷 | stop / go |
|---|---|---|---|---|
| 3b | right after step 5 has started (step 3 has ended, no gate1 running): `GPUS=$CB nohup bash train/drills_box.sh think31 > runs/drills_think31.log 2>&1 &` then `tail -6 runs/drills_think31.log` — it stops every vllm serve when it starts (none is running; step 5's training is not touched) | CB · 4 cards: ~15–45 min · 2 cards: ~0.5–1.5 h (only the empty targets, 2 samples) | `servers up`, the decode line, `== decoded`, `== think`, `samples:`, `thought source by family` | fewer than a third of the empty targets filled → send the photo before 4b (adapter B would differ little from A) |
| 4b | `NATIVE=runs/sft/drills_native31 MIX=runs/sft/drills_mix31.rendered.jsonl bash train/drills_box.sh mix` | CPU · minutes | the render lines, `drills kept by family`, `== mix` | as step 4 |
| 5b | `GPUS=$CB LR=1e-5 EPOCHS=4 NAME=e4b_drill_q31 MIX=runs/sft/drills_mix31.rendered.jsonl nohup bash train/drills_box.sh train > runs/drills_train_q31.log 2>&1 &` (beside step 5; its own ranks port, 29500 + its first card) | CB · as step 5 | as step 5 | as step 5 |
| 6b | — (step 6 gates both names in one call) | | | |

**A or B for the A/B:** only adapters whose `== tier` says A/B tier PASS. Both pass → the one with the lower block-slice
several-matches rate, and with it the retry slice's hit; within 3 points on both → B only if its `answer length` and
`not found` lines are no worse than A's (the 31B's thoughts must not make E4B slower or wordier: in the A/B, also compare
`rulestats`' tool calls and first edit against no adapter). `subs` and step 8 then run with that NAME.

The A/B tier (step 6): against gate0 — hit ≥ +5 pts · several matches ≤ −5 pts with p < 0.05 · not found at distance ≥ 2 no
more than +2 · parsed ≥ 99 % · non-block median answer ≤ the target's · block slice several matches ≤ −12 pts · retry slice hit
≥ +10 pts and resends no more. The block drills are now 25 % of the drills' label tokens (12 % before): if answers get longer
than the target everywhere, the answer-length line fails — that is the check for it.

```bash
# step 8, after subs: 4 passes without and 4 with the adapter (even cards: no adapter, odd cards: the adapter)
pkill -f "vllm serve"; sleep 5; E4B=google/gemma-4-E4B-it-qat-w4a16-ct; AR=base; NAME=e4b_drill_q     # NAME: the one subs built
LANES=(0 1 2 3 4 5 6 7); PASSES="1"       # 8 cards: 8 lanes × 1 pass (~2 h).   4 cards: LANES=(0 1 2 3); PASSES="1 2" (~4 h)
for g in "${LANES[@]}"; do AD=$([ $((g % 2)) = 1 ] && echo $NAME); nohup bash -c "for w in $PASSES; do GPU=$g TAG=$([ -n "$AD" ] && echo qd || echo qn)$g\$w TIME_MIN=8 MAX_TURNS=80 LATEST=e4b ADAPTER=$AD MODEL=$E4B SPLIT=splits/disc_e4b.txt SUBMISSION=runs/base_lb024 ARMS=$AR bash scripts/lane.sh run; done" > runs/lane$g.log 2>&1 & done
tail -n 2 runs/lane*.log                  # progress; done when every lane's last pass has printed its summary
# after every lane has ended (4 runs per arm in both layouts):
for k in qn qd; do python3 -c "import glob,json,sys; c=[sum(json.loads(l)['resolved'] for l in open(f)) for f in sorted(glob.glob(f'runs/dev33_{sys.argv[2]}_{sys.argv[1]}[0-7][12]/task_results.jsonl'))]; print(sys.argv[1], c, 'mean', sum(c)/max(1,len(c)))" $k $AR; done   # 📷
python3 scripts/edit_stats.py --pooled runs/dev33_${AR}_qn[0-7][12]   # 📷 no adapter, its 4 passes pooled: edit outcomes, retries, identical resends, loops
python3 scripts/edit_stats.py --pooled runs/dev33_${AR}_qd[0-7][12]   # 📷 the adapter
python3 scripts/rulestats.py runs/dev33_${AR}_qn01 runs/dev33_${AR}_qd11   # 📷
for k in qn qd; do echo "$k drill-template lines added in patches: $(cat runs/dev33_${AR}_${k}[0-7][12]/patches/*.patch 2>/dev/null | grep -cE '^\+(.*must not be empty|.*assert [A-Za-z_]+ is not None|\s*if [A-Za-z_]+ is None:$)')"; done   # 📷
```
**Read the A/B as:** the resolved mean of 51 tasks has a noise of about ± 3.5 tasks, and removing E4B's edit loops can win
back 2–3 tasks at most, so the resolved counts alone will not decide. The adapter counts if (1) its 4-pass mean is not
below no adapter's by more than the noise, (2) `edit_stats` shows the edit skill moved in real runs: failed edits per task
down, identical resends down, loops (tasks with ≥ 2 failed edits in a row on a file) down, "read again → next ok" up, and
(3) the first edit does not move earlier and the drill-template line count is not clearly above no adapter's (a rise = the
adapter copies the drills' content, not their skill). A ≥ 3-task gain with (2) and (3) is the clear case.

After every GPU step: the run-cost lines (`scripts/run_costs.py`, where the step prints them) go in the same photo.
Nothing here is pushed from the box; the laptop logs each photo in RESULTS.md and the private run log.

## B-3p · Step 2 of the E4B gap plan: an edit-skill LoRA from checked drills ✖ gate 1 not passed twice (2026-11-08, 2026-11-09) → B-3q above. Kept as the record: do not run these steps

### B-3p run order on the box (2026-11-08, code ≥ `f00ab93`) — one step at a time, 📷 after each

| # | step | card(s) · time | 📷 | stop / go |
|---|---|---|---|---|
| 0 | `cd gemma4-dev-agent && git pull && git log --oneline -1` | — | the commit line | the commit must be `f00ab93` or later (B-3q, with the retry family: a newer commit than the one this table was first written for — rebuild; the old held-out sets are not comparable) |
| 1 | `bash train/drills_box.sh build` | CPU · ~20–25 min | four `== drills` lines (1500 · 300 · 80 block slice · 80 retry slice) + the tools/sampling line | any `Traceback` or a set below its count → stop |
| 2 | `GPU=0 bash train/drills_box.sh gate0` (before step 4: nothing else may run, it stops every vllm serve) | 1 card · ~12 min | `edit outcomes`, `== gate`, `block slice`, both `answer length` lines, the `-- block slice` block and the `-- retry slice` block (`resend`, `recover call 1`) | the block slice's several matches with no adapter should be high (fastapi-like); if it is low (< 10 %), the block drills are not hard for E4B → send the photo before going on |
| 3 | `think` on GPU 0 **in parallel with** step 4 on GPUs 1–3 — start step 4 FIRST (it stops every vllm serve when it starts), then `GPU=0 KILL_ALL=0 DPORT=8010 nohup bash train/drills_box.sh think > runs/drills_think.log 2>&1 &` (KILL_ALL=0: think stops only its own server on port 8010, never the lanes'; without it, think kills the rollouts) | 1 card · ~2–4 h (`THINK_SAMPLES=2` halves it) | `== think`, `samples: {kept… dropped…}`, `calls with a thought by family`, the two score lines | block or anchor with a thought in < 50 % of its calls → send the photo and wait: the laptop adds hint-written thoughts (STaR-style rationalization) before training |
| 4 | E4B's own SWE-rebench rollouts on GPUs 1–3, three passes: the rebench task data must be on this box first (`runs/rebench/`: tasks.jsonl, ok.txt, snapshots, site). `GPUS="1 2 3" PASSES=3 LATEST=e4b ADAPTER= MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TOK=google/gemma-4-E4B-it-qat-w4a16-ct ARM=base GTAG=e KEEP=all SFT=runs/sft/rebenche4b SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct bash scripts/rebench_gen.sh gen`; when the 3 lanes end, the same line with `data` | 3 cards · ~3–5 h | the gen lines; later the data lines | optional — without it `mix` trains on drills only (it warns); `data` needs the same PASSES=3 GTAG=e |
| 5 | `bash train/drills_box.sh mix` | CPU · minutes | the render lines + `== mix` | render not PASS → stop |
| 6 | two learning rates at once: `GPU=0 LR=1e-5 NAME=e4b_drill_lr1 …train` and `GPU=1 LR=3e-5 NAME=e4b_drill_lr3 …train` (full lines below) | 2 cards · ~1–2 h | the `grep -hE …` line below: r/alpha/lr, schedules, eval lines, four `== epoch` lines | — |
| 7 | `for n in lr1 lr3; do GPU=0 NAME=e4b_drill_$n bash train/drills_box.sh gate1; done` | 1 card · ~30 min | every `--` block (outcomes, distance lines, gate, block slice, answer length, retry slice with `resend`) and the gate0 lines | go only if the adapter's `== tier` line says **A/B tier PASS** (since 2026-11-09, replacing (a)–(c)): against gate0 on the same drills, paired — hit ≥ +5 pts, several matches ≤ −5 pts with McNemar p < 0.05, not found at distance ≥ 2 no more than +2, parsed ≥ 99 %, non-block median answer ≤ the target's, block slice several matches ≤ −12 pts, retry slice hit ≥ +10 pts and resends no more; the stretch tier (hit ≥ 72 %, several ≤ 10 %, block slice several ≤ 50 %, retry slice hit ≥ 60 %) and the old 31B-level gate are printed beside it; otherwise stop and send the photos |
| 8 | `NAME=<the passing one> bash train/drills_box.sh subs` | CPU · minutes | the subs lines (epoch, adapter_config r/alpha, build check) | build check fails → stop |
| 9 | the A/B: no adapter vs the adapter, 4 lanes × 2 passes on the 51 tasks (commands below) | 4 cards · ~4 h | the per-arm resolved counts, `rulestats` and the drill-template line count | read it with the rule under the A/B commands |

After every GPU step: the run-cost lines (`scripts/run_costs.py`, where the step prints them) go in the same photo.
Nothing here is pushed from the box; the laptop logs each photo in RESULTS.md and the private run log.

Why: E4B's edit_file fails 47 % (31B 10 %), mostly bad copies of lines read several outputs back and anchors that are
not unique; a prompt rule got it to ~30 % at best (wave 5 `edit1`), and whole 31B trajectories also teach 31B's slow
pacing (B-3n). `train/edit_drills.py` builds short edit exercises from library files that are not eval repos (click,
jinja2, attrs, anyio, h11, werkzeug, packaging, markdown-it, yaml, dateutil; held out: sqlalchemy, tqdm, tenacity,
starlette), each answer checked with the harness's own edit_file, and puts the loss on the tool call only
(`label_scope: tool_call`; the thought is E4B's own, as context). Before any agent eval, the adapter must pass an
offline gate on the held-out drills (temperature 0): intended file ≥ 80 %, not-found at distance ≥ 2 ≤ 10 %, several
matches ≤ 3 %, every call parsed. `train/drills_box.sh` runs it in stages; every stage ends with a 📷 line.

**Since the first B-3p run (gate 1 not passed: several matches 14–19 %):** E4B's real several-matches failures are mostly
3+ line blocks copy-pasted across sibling classes (fastapi), which the first drills never had. The drills now (1) always
show another copy of a repeated changed line, and (2) have a `block` family (~15 %) from the most repetitive non-eval
libraries (typer, openai, networkx; held out: sqlmodel, huggingface_hub, pandas): a change inside a repeated block whose
shortest unique span is 3–25 lines, half of them first failing on the whole repeated block. gate0 / gate1 also score a
held-out **block slice** (80 drills, `runs/sft/drills_heldout_block`), printed as `block slice` beside the gate.

**Which prompt:** the drills carry the system prompt and tool list of the arm the A/B will use. Default = `base`
(latest_e4b). If B-3o picks `mech` or `all`, first build that arm (`BUILD_ONLY=1 … TAG=chk`, as in B-3o) and add
`SYSTEM=runs/subs/<arm>_chk/prompts/system.md DROP_TOOLS=get_code_neighbors,search_similar_code,get_code_subgraph` to
**every** drills_box.sh call below, and `ARMS=<arm>` in the A/B.

```bash
cd gemma4-dev-agent && git pull
bash train/drills_box.sh build                 # 📷 four '== drills' lines (1500 + 300 + 80 block slice + 80 retry slice) + the 'multi-line … rejected' lines + the tools/sampling line (CPU, ~15–20 min)
GPU=0 bash train/drills_box.sh gate0           # 📷 E4B without adapter on the held-out drills + the block and retry slices (~15 min; stops every vllm serve)
GPU=0 bash train/drills_box.sh think           # 📷 E4B's own thoughts for the training drills (temperature 0.7, 4 samples each, ~2–4 h; THINK_SAMPLES=2 halves it)
```
Optional, all 4 cards (~3–5 h), also the data for step 3: E4B's own rollouts on the SWE-rebench tasks of B-3n, every
trajectory kept (failed tool calls masked), mixed in as an anchor so the drills do not shift when E4B edits:
```bash
LATEST=e4b ADAPTER= MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TOK=google/gemma-4-E4B-it-qat-w4a16-ct ARM=base GTAG=e KEEP=all \
  SFT=runs/sft/rebenche4b SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct bash scripts/rebench_gen.sh gen    # 📷
# when the 4 lanes end: the same line with `data` instead of `gen` → runs/sft/rebenche4b.rendered.jsonl   📷
```
Then (without the rollouts `mix` warns and trains on drills only):
```bash
bash train/drills_box.sh mix                   # 📷 render lines + '== mix' (drills ≤ 40 % of label tokens)
# two learning rates at the same time, one card each; r16 α16 (scale 1) are the defaults since 2026-11-08, 2 epochs each
GPU=0 LR=1e-5 NAME=e4b_drill_lr1 nohup bash train/drills_box.sh train > runs/drills_train_lr1.log 2>&1 &
GPU=1 LR=3e-5 NAME=e4b_drill_lr3 nohup bash train/drills_box.sh train > runs/drills_train_lr3.log 2>&1 &
# when both have ended:
grep -hE "^drills_box|schedule|eval\[|== epoch" runs/drills_train_lr1.log runs/drills_train_lr3.log | cut -c1-200   # 📷 r 16 · alpha 16 · lr in both sft lines, schedules, eval lines, four '== epoch' lines
for n in lr1 lr3; do GPU=0 NAME=e4b_drill_$n bash train/drills_box.sh gate1; done   # 📷 both blocks: each epoch's gate beside gate0 (one after the other: gate1 stops every vllm serve)
NAME=e4b_drill_lr3 bash train/drills_box.sh subs   # 📷 NAME = the learning rate whose best epoch passes the gate (both pass: the higher 'intended'; none: stop here)
```
Why two learning rates and r16 α16 (Kaggle survey, 2026-11-08: 315 forum topics, 490 notebooks): nobody's LoRA beat
no adapter in a matched comparison at any rank (4–64) or lr (8e-6–2e-4); α/r = 2 looped (a 31B LB 0.0); our 3e-5 at
r32 α32 changed B-3n's behaviour strongly (loops) and fitted the first drills to a validation NLL of 0.035; our 1e-4 canary lost
thinking in 20 steps; the one E4B paired point (a public kit) used lr 2e-5 at scale 0.5. α = r keeps the scale at 1 so
the lr is the only strength knob, and a smaller rank leaves the scorer more KV cache (one r16 adapter: 27.7K tokens).
**Box changes since the first B-3p run (2026-11-08; all in the code, nothing to edit by hand):**
- `build` makes three sets (training 1500 with ~9 % block drills, held-out 300, block slice 80) and takes ~15–20 min.
- `think` samples E4B 4× per target and keeps a thought only from a sample whose own call was the checked answer
  (edit → the intended file, read → the right file); a thought that planned another anchor is never put before the
  answer. Its new lines `samples: {kept:… dropped:…}` and `calls with a thought by family` go in the photo: a family
  with few thoughts is one E4B rarely gets right (expected: block, anchor).
- `gate0` / `gate1` print `answer length (non-block …)` and `(block …)`: the model's old_string lines vs the target's.
**Reading gate1 (since 2026-11-09):** read the `edit_drills tier` block — the A/B tier decides (paired against gate0's per-drill outcomes, McNemar exact; the drills oversample repeated lines, so the 31B-level absolute gate is far off for E4B: E4B's real first-try several-matches rate is 5 %, the drills' 20 %). Earlier reading, kept for reference: the gate is unchanged; the non-block median answer length must not exceed the target's
median printed on the same line (3 lines at gate0, where E4B wrote 1; more = the block drills pulled every anchor longer
than needed; the earlier rule "< gate0's + 2" would have rejected an adapter that reaches the target); beside it, the block slice's several-matches rate should fall clearly below
gate0's block slice (the fastapi-like case). If the gate passes but the block slice does not move, the adapter learned
the easy anchors only — say so in the run log before the A/B.
**A/B only if gate1 passes** — no adapter vs the adapter `subs` built (`$NAME`), two lanes each, two passes per lane (4 vs 4), 51 tasks:
```bash
pkill -f "vllm serve"; sleep 5; E4B=google/gemma-4-E4B-it-qat-w4a16-ct; AR=base; NAME=e4b_drill_lr3   # NAME = the adapter subs built
for g in 0 1 2 3; do AD=$([ $((g % 2)) = 1 ] && echo $NAME); nohup bash -c "for w in 1 2; do GPU=$g TAG=$([ -n "$AD" ] && echo d || echo n)$g\$w TIME_MIN=8 MAX_TURNS=80 LATEST=e4b ADAPTER=$AD MODEL=$E4B SPLIT=splits/disc_e4b.txt SUBMISSION=runs/base_lb024 ARMS=$AR bash scripts/lane.sh run; done" > runs/lane$g.log 2>&1 & done
# after both waves:
for k in n d; do python3 -c "import glob,json,sys; c=[sum(json.loads(l)['resolved'] for l in open(f)) for f in sorted(glob.glob(f'runs/dev33_{sys.argv[2]}_{sys.argv[1]}[0-3][12]/task_results.jsonl'))]; print(sys.argv[1], c, 'mean', sum(c)/max(1,len(c)))" $k $AR; done   # 📷
python3 scripts/rulestats.py runs/dev33_${AR}_n01 runs/dev33_${AR}_d11   # 📷 edit_ok / edit_mismatch / first edit, without vs with the adapter
for k in n d; do echo "$k drill-template lines added in patches: $(cat runs/dev33_${AR}_${k}[0-3][12]/patches/*.patch 2>/dev/null | grep -cE '^\+(.*must not be empty|.*assert [A-Za-z_]+ is not None|\s*if [A-Za-z_]+ is None:$)')"; done   # 📷
```
**Read it as:** the adapter counts only if its 4-pass mean beats no adapter's by ≥ 3 tasks, and rulestats shows the
edit mismatches down without the first edit moving earlier, and the drill-template line count (the drills' insert_guard
lines) is not clearly above no adapter's — a rise means the adapter copies the drills' content, not their skill.

## B-K · The kit line: the kit's 31B data under the kit's own prompt → an E4B LoRA, against no adapter ▶ data any time; GPU after B-3n's `gen` (~4 h wave 1 + training, ~4 h wave 2)

**Why.** User decision (2026-11-03): train on the kit's data with the prompt it was generated under, and run the model
with that same prompt. Two lines run side by side: this one, and B-3n (our own 31B data under our own prompt).

**Done on the laptop (2026-11-03):**
- **Data** (`train/convert_kit.py --prompt kit`): every message as the kit served it — the Budget-Fit 0.18 system prompt,
  and task prompts saying 8 minutes / 28 calls / 80 turns. The tool schemas are ours (only whitespace differs). The data
  is cleaned by the kit prompt's own rule against touching test files:
  - 2 trajectories that edit the repository's tests are dropped;
  - 24 test-named scratch files in 21 trajectories are renamed everywhere (`test_x.py` → `repro_x.py`), with the
    patch numbers updated;
  - 2 replies that write or delete `setup.cfg` are not targets.

  Result: 101 trajectories on 54 tasks, held-out set 6 tasks (whole repositories).
- **Setting `LATEST=kit`** (`scripts/arms/kit_e4b/`): our harness builds all 103 of the kit's task prompts character
  for character.
- No repository overlap with our 129 tasks (they are in 4 repositories; the kit's tasks are in 51 others).

**Left out (user decision 2026-11-03):**
- **No `ple_fix` merge.** vLLM serves one adapter per request, so the LoRA lanes run without the PLE fix. That is why the
  comparison is LoRA vs **no adapter**; a `ple_fix` lane shows what the PLE fix alone gives under this prompt.
- **No hyperparameter search.** The training uses train_box.sh's defaults: 2 epochs · lr 3e-5 (warmup 5 %) · 16
  sequences per step · r 32 α 32 · clip 1.0. With 130 sequences that is 9 optimizer steps per epoch, 18 in all.
  `EPOCHS LR ACCUM R ALPHA` override them (printed).

### K-a · Data on the box (no GPU, internet; any time, also while B-3n runs, ~1 min + the 207 MB download)
```bash
cd gemma4-dev-agent && git pull
bash train/train_box.sh kitdata     # 📷 from the 'kit data' line down
```
The stage downloads `lora_native_resolved.jsonl` from the public kit dataset, checks its sha256 against the laptop's
file, takes our tool schemas from the first `runs/*/traj` (set `KIT_TOOLS=<run>/traj` otherwise), converts, renders
with the E4B tokenizer and runs the parser check. Expected (printed under it):
- tools sha256 `ee6a6459335f412d`;
- TOTAL 101 trajectories;
- train: 1,461 targets · 416,563 label tokens · 130 sequences;
- held out: 155 targets · 33,387 label tokens · 14 sequences;
- `RESULT PASS` twice and A5 0 excluded.

### K-b · Wave 1, after B-3n's `gen` has ended: three E4B base lanes on GPUs 0–2, training on GPU 3
```bash
pkill -f "vllm serve"; sleep 5
.venv-eval/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('google/gemma-4-E4B-it-qat-w4a16-ct'))"   # once (lanes are offline)
cat splits/dev33.txt splits/rest96.txt > runs/all129.txt && wc -l runs/all129.txt                     # 📷 129
BUILD_ONLY=1 LATEST=kit SUBMISSION=runs/base_lb024 ARMS=base TAG=kchk bash scripts/eval_dev.sh \
  | grep -E "latest set|time line|latest setting"                                                      # 📷 (also fetches ple_fix once)
A=("" "" ple_fix)
for g in 0 1 2; do GPU=$g TAG=kb$((g+1)) TIME_MIN=8 MAX_TURNS=80 LATEST=kit ADAPTER=${A[$g]} MODEL=google/gemma-4-E4B-it-qat-w4a16-ct \
  SPLIT=runs/all129.txt SUBMISSION=runs/base_lb024 ARMS=base nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
head -3 runs/lane2.log                                                                                 # 📷 "server adapters … ple_fix"
GPU=3 bash train/train_box.sh env && GPU=3 bash train/train_box.sh dequant                             # 📷 last lines (skips what exists)
GPU=3 nohup bash train/train_box.sh kit > runs/train_kit.log 2>&1 &                                    # ~30–40 min
```
| lane | what | run dir |
|---|---|---|
| GPU 0 `kb1` | E4B, kit prompt, **no adapter** | `runs/dev33_base_kb1` |
| GPU 1 `kb2` | the same again (noise: how far two identical passes differ) | `runs/dev33_base_kb2` |
| GPU 2 `kb3` | E4B, kit prompt, `ple_fix` (the setting as written) | `runs/dev33_base_kb3` |
| GPU 3 | training: `adapters/e4b_kit/epoch{1,2}` + `trainer_log.jsonl` + `train_summary.json` | `runs/train_kit.log` |

- `TIME_MIN=8` → the run flag is 8 min, as B-3m: E4B decodes faster here than on an L4, so this is lenient, and equally
  for every lane. The task prompt shows "8 minutes" either way (`prompt_shim`, printed as `task prompt time line`).
- `lane.sh` mounts the lane's `ADAPTER` on its own server (`SERVE_ADAPTER`, printed as `server adapters`).

When training ends:
```bash
grep -E "schedule|eval\[|== epoch" runs/train_kit.log | cut -c1-200                                 # 📷 held-out NLL at start, epoch 1, epoch 2
bash train/train_box.sh kitsubs     # 📷 the epoch with the lower held-out NLL → runs/adapters/e4b_kit + the BUILD_ONLY check
```

### K-c · Wave 2, after wave 1's lanes have ended: the LoRA, two passes (+ an optional 31B reference)
```bash
for g in 0 1; do GPU=$g TAG=kl$((g+1)) TIME_MIN=8 MAX_TURNS=80 LATEST=kit ADAPTER=e4b_kit MODEL=google/gemma-4-E4B-it-qat-w4a16-ct \
  SPLIT=runs/all129.txt SUBMISSION=runs/base_lb024 ARMS=base nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
head -3 runs/lane0.log                                                                                 # 📷 "server adapters … e4b_kit"
# optional: 31B under the kit prompt, no adapter (TIME_MIN=5 = the L4-calibrated budget for 31B lanes, B-3d)
GPU=2 TAG=k31 TIME_MIN=5 MAX_TURNS=80 LATEST=kit ADAPTER= MODEL_ALIAS=gemma-4-31b-it-qat-w4a16-ct MODEL=google/gemma-4-31B-it-qat-w4a16-ct \
  SPLIT=runs/all129.txt SUBMISSION=runs/base_lb024 ARMS=base nohup bash scripts/lane.sh run > runs/lane2.log 2>&1 &
```

### K-d · Read it
```bash
R="runs/dev33_base_kb1 runs/dev33_base_kb2 runs/dev33_base_kb3 runs/dev33_base_kl1 runs/dev33_base_kl2"
python3 scripts/compare_runs.py $R                                                                     # 📷
python3 scripts/fail_atlas.py $R | grep -E "^==|outcomes|failures:"                                    # 📷
for r in runs/dev33_base_kb1 runs/dev33_base_kl1; do echo "== $r"; python3 scripts/run_costs.py $r | head -12; done   # 📷
```
**Read it as:**
- The LoRA counts only if **both** LoRA passes (`kl1`, `kl2`) beat **both** no-adapter passes (`kb1`, `kb2`) by more
  than those two differ from each other (the B-3m rule).
- The held-out NLL (start → epoch 1 → epoch 2) says whether the adapter learned the kit data at all. If it barely moves
  and the scores tie, the adapter is close to a no-op (as boristown's v7: 0.3830 → 0.3829).
- `kb3` vs `kb1`/`kb2` = what `ple_fix` alone gives under this prompt (context for the LoRA, which runs without it).
- From the atlas and the costs: where the first edit comes, whether tests run, whether E4B now writes `repro_*.py`
  scripts the way the 31B data does, and malformed or repeated calls (the kit warns that too large a scale gives loops).

**Next (the laptop decides after the photos):** if the LoRA wins, the questions left out above (`ple_fix` merge,
scale/epochs), then the same test with B-3n's data; if it ties or loses, the held-out NLL and the atlas say whether to
change the training or to stop this line.

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


**Course candidate (2026-10-23, laptop):** `course` only (user: one complete version), packaged and waiting for the
box result — user decision: submit only after B-3l. Rebuild + push the winner (laptop; the laptop asks before the submit):
```bash
BUILD_ONLY=1 SUBMISSION=runs/base_lb024 ARMS=course TAG=nb bash scripts/eval_dev.sh
MINUTES=7 NAME=course_nb .venv-eval/bin/python scripts/make_submission.py runs/subs/course_nb        # 8 checks + validate_directory
python3 scripts/build_notebook.py dist/course_nb/submission.zip dist/course_nb/notebook "Gemma 4 agent course slides config"     # CPU notebook, 9 checks
kaggle kernels push -p dist/course_nb/notebook       # private kernel; run completes → submit that version (asks first)
```

## Not in this plan
- **Small-model bonus (E4B/E2B):** planned separately. On the box (or a rented
  32 GB card): `MODEL=google/gemma-4-E4B-it-qat-w4a16-ct TP=1 GPU_UTIL=0.9 bash scripts/serve_vllm.sh` serves it
  as `gemma-4-e4b-it-qat-w4a16-ct`, and the eval scripts check the alias that the submission's agent.yaml declares.
- **Budget arms** (`nothink`, output caps, thinking budget): settled by ab1/ab2, so not repeated.

## Time budget
~1–1.5 h per arm at TIME_MIN=4 (33 tasks). B-1…B-4 ≈ 10–12 box hours over 2–3 sessions. Each step needs one 📷 set.
