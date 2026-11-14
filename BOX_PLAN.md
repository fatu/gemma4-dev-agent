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

## B-10 · Our own MetaCurate-style data for Gemma: teacher trajectories from GLM-5.3 through an OpenAI-compatible gateway (plan 2026-11-13, revised the same day: generation settings as the paper, GLM-5.3 only, the official rule logic) — step 1 `setup` ✓ · `check` ✓ for SWE-rebench-V2 only (2026-11-13) · ▶ **now on the box: the three commands under "On the box now" below**, 📷

**On the box now (2026-11-13, after the first `check`; no tokens except a few cents for the probe).**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
bash scripts/rebench_mini.sh diag                    # 1 why two SWE-rebench logs have no result lines · 2 the patches whose
                                                     #   verdict differs from the pool's label · 3 later commits in an image → 📷
.venv-swe/bin/python scripts/teacher_probe.py        # step 0, the gateway in the Anthropic format (POST <base>/v1/messages) → 📷
```
`diag` reads the logs the first `check` left in `runs/b10/check5.*` and pulls one SWE-rebench image again (~1.3 GB);
`DIAG_SOURCES="rebench gym" bash scripts/rebench_mini.sh diag` adds one SWE-Gym image (10 GB or more: only with the disk
free). The probe needs the teacher's name and the gateway's address and key in the environment of that shell:
`TEACHER=<the gateway's name for GLM-5.3>` plus either `GATEWAY_BASE_URL` / `GATEWAY_API_KEY` or the Anthropic-style
names `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY`; it prints which names it used, never the
values. If they live in a shell profile, start the probe from a shell that has loaded it; do not copy the values into a
command line, a script or a file under git. A 404 in turn 1 = the gateway does not serve this format (then
`--api chat`). Reasoning NOT RETURNED → once more with `--thinking-budget 2000`.
**Open after the first `check` (2026-11-13):** SWE-rebench: 2 of 5 tasks give no parsable test log (the grading steps
were re-read against the harness: the same; the logs decide) · SWE-Gym: images of 10–17 GB each, commits outside HEAD's
history in the image, 2 of 15 labels differ → proposed: leave SWE-Gym out (194 tasks; 1,221 remain), the user decides ·
SWE-rebench images also hold commits outside HEAD's history (4,948 vs 3,065): `diag` shows whether any is later than the
task's commit; if so they are removed in the container before the agent starts.

**Why.** B-9 part 2 tests the public DeepSeek V4 Pro pool. B-10 makes the same kind of data with teachers we can call,
shaped for Gemma from the start, and asks whether the teacher matters for Gemma (DSv4Pro vs GLM-5.3 on the
same tasks; user 2026-11-13: no teacher comparison, GLM-5.3 first). **Method = MetaCurate App. A**: tasks from SWE-rebench, SWE-rebench-V2 and SWE-Gym (prebuilt per-task
images) · mini-swe-agent · temperature 0.7 · 3 attempts per task · resolved = fail-to-pass and pass-to-pass tests pass in
the task's container. **Generation settings = the paper's** (user 2026-11-13: "照论文"). The generation code is not
published (repo = the filter search only; the dataset's release note gives the harness revision as unknown), so the
settings are read from the dataset itself, all 4,286 rows: `swebench.yaml` of mini-swe-agent with three edits to the
instance prompt (an added "the environment is offline: do not install, clone or download" line · "create new local
scripts" · the "you can also install it" sentence removed) · step limit 250 (the only limit-exceeded rows end at 249–250)
· replies capped at 4,096 tokens (the largest reply is exactly 4,096) · command timeout 60 s · parallel tool calls on
(15 % of turns carry 2–11 calls) · observations cut at 10,000 characters. Not recoverable, built here: the images for
the three sources (SWE-Gym rows carry no image name), the grader (the dataset gives the expected F2P / P2P test names
per row: used to check ours), the exact mini-swe-agent revision (2.4.6 is used), the reasoning request parameter.
**Added for Gemma at curation, not at generation** (each printed): kept only if resolved, every model turn has a thought
and the peak context is ≤ 30,720 Gemma tokens · first on the **same 1,415 tasks as MetaCurate**, so the teacher is the
only difference from B-9 part 2's data.
**Where.** The 8-card box, no GPU: the teacher is remote, the box runs the containers and the grading, so B-10 runs beside
B-9 part 2's training (shared disk: the chunked image deletion of `verified_box.sh full` is reused).
**Teacher.** GLM-5.3 only. Background (web, 2026-11-13): no public comparison of the candidate teachers as teachers; a
controlled study on terminal tasks found the weakest of its teachers gave the best students, so benchmark rank is not a
selection criterion; the licence text of GLM-5.3 has no clause on outputs (the gateway's own terms: the user checks).
**Secrets and protocol.** The gateway is used in the Anthropic format (user 2026-11-13: "改成兼容 /v1/messages"): POST
`<base>/v1/messages`, mini's model config `anthropic/$TEACHER` with `api_base` = the address without `/v1` (litellm adds
`/v1/messages` itself; checked on the laptop against a stand-in server). `API=chat` keeps the OpenAI format
(`openai/$TEACHER`). Address, key and `TEACHER` come from the environment on the box; never echoed, never written to a
file under git.

**Step 0 · Gateway check (box, 5 min, a few cents).** New `scripts/teacher_probe.py`: one chat call with mini's `bash`
tool for the teacher. Prints: tool call returned (yes / no, how many) · reasoning returned and under which field
(`reasoning_content` / `reasoning` / thinking blocks) · usage (prompt, cached, completion tokens) · latency. If the
reasoning does not come back: stop + photo (thought-less data is the canary failure mode).
Rewritten for the Anthropic format and tested against local stand-in servers in both formats, the litellm route
included (2026-11-13); the command is under "On the box now" above. Three turns with canned tool results (nothing is
executed), then one call through litellm (the route mini uses). 📷 every line. `RESULT go` = a tool call in every turn
and reasoning (thinking blocks) returned in at least two of three. If the reasoning is NOT RETURNED, try once with
`--thinking-budget 2000`, then with the gateway's own switch via `--extra '{"…": …}'`; still nothing → stop + photo. If "history with reasoning accepted" prints NO, say so with the photo: step 1's config then has to
drop the reasoning from the history.

**Step 1 · The task set in mini-swe-agent, and the grader (box, no tokens, ~30–60 min).** Written 2026-11-13:
`scripts/rebench_mini.sh` (setup | check | subset | run | grade | report) + `scripts/rebench_mini.py` +
`scripts/b10_instance_template.txt` (the pool's prompt). Tested on the laptop without docker: the task table (all 1,417
source–id pairs of the pool found upstream at the pinned revisions; every one has an image name and a pytest log parser;
1,415 distinct ids; SWE-rebench 690, V2 531 all Python, SWE-Gym 194), the subsets, the generated config, the log parser.
Not tested anywhere yet: everything that touches docker.
```
cd gemma4-dev-agent && git pull && git log --oneline -1
bash scripts/rebench_mini.sh setup        # docker, versions, the task table (downloads ~720 MB of upstream parquet) → 📷
bash scripts/rebench_mini.sh check        # 5 tasks per source: gold, empty and the pool's own patches graded → 📷
```
What differs per source (read from the upstream harnesses and image configs, 2026-11-13): SWE-rebench images
`swerebench/sweb.eval.x86_64.<id>` (in the row) · V2 images in the row, repository at `/<repo name>` and no conda (a
startup command links `/testbed` to it, so prompt and working directory are the same for all) · SWE-Gym rows have no
image: `xingyaoww/sweb.eval.x86_64.<id with __ → _s_>`. One grader for all three (a port of each harness's steps; in the
header of `rebench_mini.py`); for SWE-Gym the harness also re-runs the repository's install command before the tests,
which ours does not. **Read `check`:** gold = resolved on every task · empty = 0 resolved · pool = our verdict equals
the pool's label on (nearly) every patch · the git line shows `all` = `reachable` (else later commits are readable in
the image: SWE-Gym's README warns of this). A source that fails these is fixed or left out before any token is spent;
then the gold check runs on all 1,415 (it also pulls every image: Docker Hub allows 100 anonymous pulls per 6 h per
address, 200 with a login) and only tasks whose gold patch passes are generated on.

**Step 2 · Pilot (box, ~1–2 h, paid).** 30 of the 1,415 tasks (stratified by source, seed 0) × 1 attempt, GLM-5.3.
`NAME=pilot30 N=30 bash scripts/rebench_mini.sh subset` → `TAG=glm NAME=pilot30 bash scripts/rebench_mini.sh run` (needs the
three exports of step 0) → `… grade` → `PRICE=<in,cached,out $ per M> … report`. Not before step 0 and `check` have passed.
📷: resolved / 30 · runs that fit (≤ 30,720 Gemma tokens) · turns with a thought · thought tokens p50 / max ·
calls p50 / p90 · prompt / cached / completion tokens per run · cost per run (gateway price × usage) · wall s per run ·
errors; plus DSv4Pro's public runs on the same 30 tasks as the reference. **Rule set before the data:** go on if
thoughts are in ≥ 90 % of turns and usable runs (resolved ∧ fits ∧ thoughts) are ≥ 0.15 per attempt; otherwise stop +
photo. The scale step's cost estimate comes from these numbers and needs the user's go.
For scale: DSv4Pro shard 0 used 0.65 M prompt tokens per run (median; 98 % cached), 10.6 K completion tokens, 200 s.

**Step 3 · Scale (box, paid).** GLM-5.3 on the 1,415 tasks × 3 attempts (1 attempt first if the estimate is
high), chunked and resumable. Target ≥ 1,000 usable runs; the public pool itself gives fewer after the rule filter (table
below), so a fourth attempt or more tasks may be needed: decided from the pilot's yield. Data stays on the box (`runs/`,
gitignored).

**Step 4 · Curate (box CPU).** `train/mc_rules.py` = the paper's rule filter, a port of the released seed rules
(`filters/initial_ranking_nometa/rules.py`, MIT): the same seven signals, patterns and six thresholds, no model and no
execution labels. Reject: (1) the submission is not a git diff · (2) a disallowed git command (push, pull, fetch, clone,
cherry-pick, reflog, fsck, remote, ls-remote) · (3) ≥ 8 returns from a test run to editing, or ≥ 3 self-reverts
(`git checkout --`, restore, stash, `reset --hard`) · (4) no edit at all, or ≥ 20 commands with ≥ 85 % exploration and ≤ 2
edits. Flag, counted and printed only (the paper's text; the released file rejects on these too): (5) > 10 % of turns
without a command · (6) `print(`, `pdb` or `breakpoint()` on an added line of the patch · (7) edits but no test runner.
Not published, chosen here: one turn per assistant message (`--turns msg`); 1,000 drawn at random (seed 0) from the
survivors that are also resolved, fit and have thoughts → arm **G1k**. MetaCurate's own filter search only later: it
needs a paid judge model (their code has an OpenAI-compatible client, so the gateway could be the judge).

The rule filter on the public DSv4Pro pool (laptop, 2026-11-13, all 4,286 trajectories; fit = the teacher's own token
count of the last call ≤ 30,720, a proxy for Gemma tokens):

| | trajectories | resolved | resolved ∧ fits |
|---|---|---|---|
| pool | 4,286 | 1,760 | 1,027 |
| passes signals 1–4 (the paper's text) | 3,725 (87 %) | 1,549 | 949 |
| passes all seven (the released file) | 2,296 (54 %) | 921 | 507 |

Per signal, share of the pool: not a diff 0.8 % · disallowed git 0.7 % · loops 3.8 % · exploration 8.2 % · turns without a
command 0 % · debug lines 1.5 % · edits without a test runner 40.6 %. The filter does not separate resolved from
unresolved runs: resolved share 41.6 % among the survivors of signals 1–4, 41.1 % in the pool. With one turn per tool call
(`--turns call`) the exploration signal fires on 12.6 % and 3,540 pass signals 1–4.

**Step 5 · Train and evaluate exactly as B-9 part 2 steps 2–4** (same renderer, recipe, ADAPTER eval, paired.py), 31B
first, any free box. **Read it (set before the data):** G1k vs base by B-9 part 2's criterion (improved or not); G1k vs R1k on the same 500 tasks: G1k ≥ R1k + 3 pts →
our own generation becomes the data source for the Kaggle-harness stage (plan 24 P2); otherwise the public pool stays.
**Optional:** Gemma 31B as the teacher for E4B (free, GPU) on the same tasks, if step 2 shows teacher thoughts far from
Gemma's style (native thoughts; distillation inside the family).

**Timeline.** 11-14 steps 0–1 · 11-15 pilot · 11-16 → 11-18 scale (beside B-9 part 2) · 11-19 → 11-21 train + eval E4B.
Final submission 12-02. **Risks:** token cost; gateway rate limits; reasoning not returned; ~1,400 image pulls (Docker Hub
limits, disk); teacher licences.

## B-9 · All 500 of SWE-bench Verified: 31B on the 8-card box, E4B on the 4-card box (2026-11-11) — 31B ✓ 272 / 500 (54.4 %) · E4B ▶ running (another session) · part 2 (a LoRA on MetaCurate's data, 31B first) planned

**31B result (photo 2026-11-13).** 272 / 500 resolved = **54.4 %** (Wilson 95 % ≈ 50.0–58.7 %), one pass, all 500 predictions (no duplicates or gaps; merged from three runs over task ranges 0–249, 250–274 and 275–499, 16 grading reports). Exit statuses: Submitted 385, ContextWindowExceededError 115. The 115 empty patches are exactly the context-window exits; no other failure type. Among the 385 submitted: 272 resolved (70.6 %). Model calls per task: median 32, p90 56, max 158. Per repo (resolved / tasks): django 128/231, sympy 40/75, sphinx 20/44, matplotlib 18/34, scikit-learn 26/32, astropy 12/22, xarray 14/22, pytest 12/19, pylint 1/10, requests 0/8, seaborn 0/2, flask 1/1.
Read by the rule set before the data: ≥ 40 % → our setup is near the public level (Vals AI 53.92 % on its subset, a different setup: close, not equivalent). The 32K window is the one visible cost: 23 % of tasks (115) ended on it. Missing: the 50 Verified-mini line (the box could not load the HF dataset: a self-signed certificate; set `SSL_CERT_FILE` and run `report` again, no rerun). The box copy of `verified_box.sh` has two changes not in this repo (MIRROR, START).

### B-9 part 2 · Does a LoRA on MetaCurate's public data improve on the B-9 baseline? (plan 2026-11-13, revised the same day by the user: 31B first, our own LoRA recipe, no pipeline check, no QLoRA, any free box, the criterion is "improved or not") — step 0 ✓ (server drops earlier thoughts) → option ③ THINK_HIST=1 · 0b ✓ (thought newline → stripped) → data ✓ → render ✓ → **switched to E4B first (31B training too slow)** ▶ E4B: data+render (task holdout) → dequant → smoke → train → 1b E4B baseline → eval; 31B later

**Question.** Train a LoRA on the public `furiosa-ai/MetaCurate-DeepSeekV4Pro` trajectories (arXiv 2610.23134) and evaluate it
under exactly the B-9 setup (mini-swe-agent 2.4.6, thinking on, 32K, temperature 0.2, 250 steps, 8,192 tokens per reply,
Verified 500, official grading). **31B first, then E4B.**
**Why this data** (checked on the laptop, shard 0 of 4 = 1,072 rows, 2026-11-13): same agent and the same first prompt as
B-9 (100 / 100 sampled trajectories match the 2.4.6 `swebench.yaml`); 97 % of model turns carry `reasoning_content`
(B-9 runs with thinking on); 0 trajectories from the 12 Verified repos; 417 resolved, 275 of them fit 32K in Gemma tokens
(≈ 1,100 in all 4 shards). It is the only public multi-run LoRA gain on Verified (Qwen3.5-27B, LoRA r64: 72.3 ± 0.6 →
74.3 ± 0.6). **Their training code is not public** (GitHub history, Hugging Face and the web searched 2026-11-13; the
authors' mirror README: "Consumers should apply their own chat-template and loss-masking policy"), so we render and train
with our own tools and our own LoRA recipe.
**Where.** Any free box (8-card or 4-card) for training and evaluation; a model is trained and evaluated on the same box.
Costs: box time only.

**▶ Steps 0–2 are built (2026-11-13): `scripts/req_log.py`, `train/mc_prep.py`, `train/mc_box.sh`, `render.py --wire mini
--split-runs --shard`, `verified_box.sh` knob `API_PORT_OFFSET`.** Laptop tests: a fake vLLM + a replayed trajectory
through the logger (4/4 requests rebuilt from the traj file with history keep, 1/4 with drop; A2 4/4; one packed
sequence); 20 real DSv4Pro rows → render PASS; `render.py` on the old w7 set gives byte-identical output (A2 427/427).
Not tested: real mini-swe-agent, real vLLM, the HF download. On the free box (31B first; any free card):
```
cd gemma4-dev-agent && git pull && git log --oneline -1
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh serve     # one card, "vLLM up" 📷
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash train/mc_box.sh probe             # step 0, ~20 min 📷
MODEL=google/gemma-4-31B-it-qat-w4a16-ct bash train/mc_box.sh data                              # step 1, ~15 min 📷
MODEL=google/gemma-4-31B-it-qat-w4a16-ct bash train/mc_box.sh render                            # step 2, ~10–40 min 📷
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh stop      # free the card
```
Read the probe 📷 (4 lines: `== probe …`, `== verify …`, `calls … A2 …`, `RESULT`):
- `earlier reasoning sent: yes|no` → HISTORY keep|drop for step 1 (the records reproduce what mini sent; the template
  decides what is rendered; `rendered by the template: yes` → a trajectory packs into one sequence).
- `text+call turns N` = Gemma 31B's own replies with text AND tool calls. Such a reply hits a **known Gemma 4 chat-template
  bug**: the text is re-rendered after the tool results and the next prompt gets no `<|turn>model` header (HF
  google/gemma-4-31B-it discussions #67, #115 — Google: "working on this"; community fix #154 and vLLM PR #66269, both
  open on 2026-11-13; Kaggle forum #744844: not fixable from a submission; Gemma itself does write text before a tool
  call). T2T (`--text-to-thought`) moves that text to the end of the thought (content → ""), so the training data never
  contains such a turn and history = generation order (on 20 rows also 172 → 36 sequences, 2.9 M → 0.6 M tokens).
  Rule now: N = 0 → T2T=on automatically; N > 0 → off unless set (`T2T=on|off`) — **open: the user decides after the
  photo** (N > 0 means the bug already affects the baseline, an argument for on).
- `== verify …: n / n` (our traj → request rebuild = what mini sent) and `RESULT PASS` (A2: our renderer = the server,
  token for token) must both hold, else stop and photo `grep -E "DIFF|FAIL" -m5` of the output.
Step 1 needs the HF token; if the download fails on the certificate, set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` to the
box's CA file first. Step 2 must end `→ PASS`. Steps 3–4 (training, the ADAPTER eval, paired.py) are built after these photos.

**Step 0 result (31B, 2026-11-13, photo):** 2 tasks · 82 calls · 9 min. mini sent earlier thoughts (`reasoning_content` on
1,615 of 2,040 history assistant messages) but the server rendered **0** of them; 782 text + tool-call turns in 31B's own
histories; prompts p50 13,194 / max 23,701 tokens; usage = /tokenize 82/82; verify 82/82; **A2 0/82 → FAIL**. Cause, read in
vLLM 0.19.1 `entrypoints/chat_utils.py` `_parse_chat_message_content`: (1) string content becomes text parts, and the Gemma
template appends a space after system parts (1 token); (2) only the `reasoning` field is passed to the template —
`reasoning_content` (what mini / litellm send) is dropped, so **31B never sees its earlier thoughts in B-9** (Google's
guide says thoughts between tool calls must not be removed). Fixed on the laptop: `render.py --wire mini` now does what
vLLM does (`vllm0191_message`); replies are always rendered from the target in that mode; w7 output byte-identical; 20
DSv4Pro rows PASS. Consequence: no packing — **one training sample per call** (20 rows: 650 samples, 9.2 M tokens, 149 K
label tokens ≈ 0.46 M tokens per trajectory, ~20× the packed size). T2T = on (user, after the photo). Re-check on the box
without a GPU (the probe's saved requests and server prompts):
```
git pull && .venv-eval/bin/python train/render.py runs/verified/probe/records --check runs/verified/probe/server_prompts \
  --wire mini --sample 20 --tokenizer "$(.venv-eval/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-31B-it-qat-w4a16-ct', allow_patterns=['*.json','*.jinja']))")" | grep -E "^calls|^FAIL|^RESULT" | cut -c1-300
```
📷 must show `A2 server-rendered prompts equal 82/82` and `RESULT PASS` (this checks the fix on the original setup).

**Decision (user, 2026-11-13): option ③ — let the server see the model's earlier thoughts (`THINK_HIST=1`).** Each mini
part talks to `req_log.py --rename-reasoning` (port+200), which copies `reasoning_content` to `reasoning` before vLLM;
then the template renders every earlier thought (Google's guide: no stripping between tool calls), a trajectory packs into
one sequence again (20 rows: 36 sequences, 0.62 M tokens, vs 650 / 9.2 M per call), and **B-9's baselines are re-run in
this setup** (the old 272 / 500 stays as the reference for "does seeing its own thoughts help"). Built 2026-11-13:
`THINK_HIST` in `verified_box.sh` (default 0 = old setup) and `mc_box.sh` (default 1), `render.py --wire mini-think`,
`mc_prep build --wire`, the probe's "generated thoughts ending with a newline" count. Laptop: a fake server that drops
`reasoning_content` like 0.19.1 → without the rename 0 thoughts rendered, with it 6 / 6, A2 4/4, one packed sequence PASS;
20 DSv4Pro rows with mini-think PASS; w7 byte-identical. Risk: contexts grow faster (31B already had 115 / 500
ContextWindowExceeded at 32K), so the new baselines may be lower; with thoughts kept, 24 of 50 resolved rows fit 30,720
(31 without), so expect ≈ 850 trajectories, not 1,000.

**Step 0b · re-probe with THINK_HIST=1 (31B, 1 card, ~10 min).**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh serve
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash train/mc_box.sh probe        # THINK_HIST=1 by default → runs/verified/probe_th
```
Sampling of the probe and of the new baselines (user, 2026-11-13): **8,192 tokens per reply · top_p 0.95 · thinking budget
4,096** (vLLM's per-request `thinking_token_budget`, sent through mini's `extra_body`, the field the Kaggle harness sends;
our Kaggle latest has 6,144 / 4,096 / 0.95; temperature 0.2 everywhere). mc_box's probe passes MAX_TOK / TOP_P /
THINK_BUDGET (defaults 8192 / 0.95 / 4096) to verified_box.sh, which writes them into mini's model config and prints them;
the probe prints a `sampling sent by mini (as the server got it)` line — check it shows top_p 0.95 and
thinking_token_budget 4096 (laptop: config and proxy tested with a stub mini; whether litellm forwards extra_body is only
visible on the box).
📷 the `== probe` line + the `sampling sent by mini …` line + the `generated thoughts …` line, `== verify`, `calls … A2 …`, `RESULT`. Go on only if: rendered by
the template **yes**, verify n / n, A2 n / n, RESULT PASS. If many generated thoughts end with a newline, stop: the
template would re-render them with two (photo the line; the reply targets would need a rule for that).

**Step 0b result (31B, 2026-11-13, photo):** 2 tasks · 57 calls · 3 min · prompts p50 11,020 / max 24,289. Thoughts
rendered by the template **yes** (856 / 856 history assistant messages); verify 57/57; A2 57/57; packed (57 calls → 5
sequences); RESULT PASS. Sampling the server got, 57/57: max_tokens 8192 · temperature 0.2 · top_p 0.95 ·
thinking_token_budget 4096 (litellm forwards `extra_body`). **Stop rule hit:** 47 of 57 generated thoughts end with a
newline — Gemma writes `…\n<channel|>`, vLLM keeps the `\n` in the parsed reasoning, and the template adds its own `\n`
before `<channel|>`, so the history would show two. User (on the box): "用方案1，去掉换行再训练". Done 2026-11-13: the
THINK_HIST proxy sends `reasoning = reasoning_content.rstrip("\n")` and `render.py --wire mini-think` strips the same in
the history and in every target, so target = generation = history: `<thought>\n<channel|>` (DSv4Pro thoughts never end
with a newline: 0 / 46,211 in shard 0). Laptop: fake server with newline-ended thoughts → A2 4/4, delta = synthetic, one
sequence, PASS; w7 byte-identical. **The proxy change alters what the server sees, so re-run the probe once (~3 min) and
then start 1b with this commit:**
```
git pull && MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh serve
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash train/mc_box.sh probe    # 📷 == probe · == verify · calls … A2 · RESULT
MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPU=0 PORT=8011 bash scripts/verified_box.sh stop
```
Go on to 1b if: rendered yes, verify n/n, A2 n/n, **delta≠synthetic 0**, RESULT PASS (the `generated thoughts … ending
with a newline` count stays high — that is the model's output, now handled).

**Steps 1–2 tested on the laptop (2026-11-13, shard 0 only, E4B tokenizer; the box uses all 4 shards and 31B's):**
`mc_prep build` as mc_box runs it (keep · mini-think · T2T · max ctx 30,720 · budgets 4,096 / 8,192): 1,072 rows → 417
resolved → 223 fit → 223 records (139 tasks), 24 s, 251 MB; over-budget turns 0 / 0; Verified repos 0. `mc_box.sh render`
(8 procs): 5,491 calls → 337 packed sequences, 5.85 M tokens, 1.20 M label tokens (20.6 %), len p50 17,516 / max 30,464,
run breaks only at no-thought turns (113), malformed 0, FAIL 0 → PASS, 25 s. `sft_lora.py --dry-run --lr 1e-5 --max-len
30720` reads it: 329 train + 8 holdout sequences, 21 steps per epoch. Full data ≈ 4×: ~890 records, ~1,350 sequences,
~23 M tokens, ~85 optimizer steps per epoch.

**E4B first (user, 2026-11-13: "盒子上训练有点慢，把整个plan先改成E4B的训练").** 31B training on the box was too
slow; the whole sequence now runs on E4B, 31B comes after. What carries over unchanged:
- **Data and render.** The 31B-tokenizer render of the shard-0 trajectories is token-identical to the E4B-tokenizer
  render (337 / 337 sequences, laptop check 2026-11-13), and the peak-context filter uses the same tokenizer, so
  `r1k.rendered.jsonl` / `r1k.holdout.rendered.jsonl` are E4B's training files as they are. (If data + render were not
  re-run since `1802230`, re-run them once for the task holdout; MODEL does not matter for these two steps.)
- **Setup.** THINK_HIST=1, wire mini-think, T2T on, thought newlines stripped, 8,192 per reply · top_p 0.95 · thinking budget
  4,096, lr 1e-5, 2 epochs, 16 per step, r32 / α32, max len 30,720 (170 steps on the full set).
- `mc_box.sh` now defaults to MODEL=E4B (`google/gemma-4-E4B-it-qat-w4a16-ct`).
Not checked for E4B: its own generation habits under THINK_HIST=1 (the probes ran on 31B). Optional, 3 min:
`MODEL=google/gemma-4-E4B-it-qat-w4a16-ct GPU=0 PORT=8010 bash scripts/verified_box.sh serve` then
`GPU=0 PORT=8010 bash train/mc_box.sh probe` (📷 the same lines; the rendering rules are the same template, so A2 should pass).
E4B sequence on the free box (stop any 31B training first: `pkill -f "sft_lora.py.*mc_gemma-4-31B"`):
```
git pull
bash train/mc_box.sh dequant                                    # 📷 bf16 E4B → runs/dequant/gemma-4-E4B-dq (~16 GB)
GPUS=0 bash train/mc_box.sh smoke                               # 📷 peak GB + tok/s
GPUS=0,1,2,3,4,5,6,7 nohup bash train/mc_box.sh train > runs/mc_train_e4b.log 2>&1 &    # or the 4 cards of the 4-card box
# then the E4B baseline in the same setup (1b), on whichever box is free:
export MODEL=google/gemma-4-E4B-it-qat-w4a16-ct GPUS="0 1 2 3" PORT=8010 TAG=vf_e4b_th SUBSET=verified WORKERS=8 EVAL_JOBS=16 THINK_HIST=1 MAX_TOK=8192 TOP_P=0.95 THINK_BUDGET=4096
bash scripts/verified_box.sh serve && nohup bash scripts/verified_box.sh full > runs/verified/vf_e4b_th.full.log 2>&1 &
```
The adapter's eval (ADAPTER knob + paired.py) is built next; it runs against `vf_e4b_th` on the same box.

**Order changed (user, 2026-11-13): train first, the new baselines (1b) after — training shows data errors early.**
Built 2026-11-13: `mc_box.sh env | dequant | smoke | train` (knobs printed: GPUS · CARDS_PER_RANK · LR 1e-5 · EPOCHS 2 ·
ACCUM 16 · R/ALPHA 32/32 · WARMUP 0.05 · MAX_LEN 30,720 · DQ · OUTD · DIST_PORT) and `sft_lora.py --device-map auto` (one
model copy split over the cards a rank sees — the fallback for 31B instead of QLoRA). Laptop: the launcher ran 2 ranks with
the mock model on the shard-0 data (21 steps, identical parameter checksum on both ranks, export check PASS) and the smoke
step; the real model / GPU path is untested here. On the free 8-card box, after the quick re-probe above:
```
git pull
export MODEL=google/gemma-4-31B-it-qat-w4a16-ct
bash train/mc_box.sh data      # 📷 rows → … line, printed-only line, == line   (HF token; all 4 shards)
bash train/mc_box.sh render    # 📷 == render … → PASS
bash train/mc_box.sh env       # 📷 versions (skips if .venv-train exists)
bash train/mc_box.sh dequant   # 📷 bf16 31B → runs/dequant/gemma-4-31B-dq (~62 GB disk)
GPUS=0 bash train/mc_box.sh smoke                          # 📷 peak GB + tok/s on the 16 longest sequences
GPUS=0,1,2,3,4,5,6,7 nohup bash train/mc_box.sh train > runs/mc_train_31b.log 2>&1 &   # 8 ranks
```
**Held-out tasks (user, 2026-11-13: "改一下"):** the random 8-sequence holdout leaked (4 of the 8 shared a trajectory with
training sequences; a task has up to 3 trajectories). `mc_prep build --holdout-tasks 8` now keeps 8 whole tasks out — chosen
among tasks with exactly one picked trajectory, so nothing of them is trained — in `runs/sft/mc/r1k_holdout/`; `render`
renders them to `r1k.holdout.rendered.jsonl`; `train` uses `--eval` on that file (falls back to `--holdout 8` if it is
missing). Costs ≈ 3 sequences of training data. Laptop (shard 0): 8 of 57 one-trajectory tasks → 8 records → 13 held-out
sequences PASS; 215 training records → 324 sequences PASS; mock 2-rank train reads `holdout 13 (file)`. **Re-run on the
box (~3 min): `bash train/mc_box.sh data && bash train/mc_box.sh render`** (data clears the old r1k dirs first), then
upload the new files if you keep the Kaggle copy. If smoke runs out of memory on one card: `CARDS_PER_RANK=2 GPUS=0,1 bash train/mc_box.sh smoke`, then train with
`CARDS_PER_RANK=2 GPUS=0,1,2,3,4,5,6,7` (4 ranks). Stop and photo if: render is not PASS, the schedule line drops many
sequences (`dropped: > 30720 tokens`), the holdout NLL rises from `eval[start]`, or a rank ends early. Read the time from
smoke: total tokens × 2 / (tok/s × ranks). Then 1b, then the adapter eval (ADAPTER knob + paired.py, built next).

**Step 1b · the new baselines, all 500, THINK_HIST=1** (31B on the 8-card box; E4B on whichever box is free; same
commands as part 1 plus the knob and new tags; ~ the part-1 times):
```
export MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPUS="0 1 2 3 4 5 6 7" PORT=8011 TAG=vf_31b_th SUBSET=verified WORKERS=6 EVAL_JOBS=16 THINK_HIST=1 MAX_TOK=8192 TOP_P=0.95 THINK_BUDGET=4096
bash scripts/verified_box.sh serve && nohup bash scripts/verified_box.sh full > runs/verified/vf_31b_th.full.log 2>&1 &
# E4B: MODEL=google/gemma-4-E4B-it-qat-w4a16-ct PORT=8010 TAG=vf_e4b_th (GPUS = that box's cards), the rest the same
```
📷 the `report` block of each. Read: resolved vs 272 / 500 (31B, old setup) is the effect of seeing its own thoughts;
count ContextWindowExceeded. Step 1's filter prints (does not drop) the turns a 4,096 thinking budget or an 8,192
reply cap could not have produced (shard 0 on the laptop: 0 and 0 among 223 picked rows). Steps 1–2 below (`mc_box.sh data`, `render`) then run with THINK_HIST=1 (default), and the
adapters are evaluated with THINK_HIST=1 against these baselines.

**Step 0 · What does Gemma see at call k? (free box, 1 card, ~20 min)** — decides the renderer's history rule.
New `scripts/req_log.py`: a logging pass-through proxy (port PORT+100 → vLLM) that saves each chat request. Two Verified
tasks with 31B through it (`TAG=vf_probe SLICE=0:2`); for 3 calls per task the saved messages go to vLLM's `/tokenize`.
Prints: earlier `reasoning_content` present in the request (yes / no) · rendered by the Gemma template (yes / no) · how a
turn with several tool calls and a turn with text + a tool call render · Gemma tokens per call. 📷 the summary line.

**Step 1 · Data (box CPU, ~15 min).** Download the 4 parquet shards (~0.5 GB, HF token typed by the user; the box needs
`SSL_CERT_FILE` or its mirror for HF). New `train/mc_prep.py`. The paper's comparable arm is **Resolved** ("retains only
trajectories that pass the execution-based evaluation", then 1,000 per arm); its best arm (MetaCurate-selected) cannot be
rebuilt because the selected subsets are not published. Filters, each count printed:
1. resolved only (as the paper's Resolved arm);
2. peak context ≤ 30,720 Gemma tokens under step 0's history rule — **not in the paper** (it trained at 128K and evaluated
   with a 262K window); needed because B-9 and Kaggle run at 32K;
3. arm **R1k** = 1,000 at random (seed 0) from what is left (all of it if fewer, printed).
Printed only, nothing removed: trajectories from Verified repos (expected 0), thoughts over 8,192 Gemma tokens (B-9's reply
cap), licence values, trajectories per task, multi-tool-call turns. The terminal `exit` event is not a filter: the renderer
leaves it out of the sequence (dataset card: "account for the terminal `exit` event when applying a chat template").

**Step 2 · Render (box CPU).** New `train/mc_render.py` (label logic from `render.py`): the `-ct` tokenizer's Gemma chat
template, tools = mini's one `bash` tool; one sequence per trajectory with several label spans if step 0 says earlier
thoughts stay in the prompt, else one sample per call; labels = each model turn's thought + text + tool calls + end of turn;
system, user, tool output masked. Gate R (all must pass): for every probe call, our rendered prefix = vLLM `/tokenize`
token for token; 50 decoded targets = the source text; every trajectory's label share printed.

**Step 3 · Train 31B, then E4B (free box, all its cards).** `train/sft_lora.py` (exists; `--world <cards>`, one model copy
per card, data-parallel; bf16 base from `dequant_ct.py`; no QLoRA) with **our own recipe = its defaults**: r32 · α32 ·
dropout 0 · every language-model projection present in the checkpoint · **lr 1e-5** (user 2026-11-13; the script default 3e-5
was our own pick after the 10-29 canary collapse at 1e-4, no paper behind it) · 5 % warmup then cosine to 10 % · 16
sequences per optimizer step · 2 epochs · clip 1.0; `--max-len 30720` (matches the filter; the default is 26,624). 31B:
`--ct google/gemma-4-31B-it-qat-w4a16-ct --alias gemma-4-31b-it-qat-w4a16-ct`. 31B in bf16 is ~62 GB per card before
activations; the first step prints the peak memory. If one copy per card does not fit, the fix is to split one copy over
two cards (`--device-map`, a new knob) — not QLoRA. Time: a guess, 31B 6–10 h, E4B 2–4 h.

**Step 4 · Evaluate (the same box).** New knob `ADAPTER=<dir>` in `verified_box.sh` (serve mounts it; mini asks for the
adapter's name; printed) → `full` on Verified 500 with every other B-9 setting unchanged. New `scripts/paired.py`: adapter
vs the B-9 baseline of the same model. **Criterion (set before the data): improved or not = adapter resolved > baseline
resolved** (31B: > 272 / 500). Printed with it: the difference with its 95 % interval; exact McNemar p and the per-task
win / loss counts when the baseline's report is on the same box (else the unpaired interval only); ContextWindowExceeded
and empty-patch counts. Expect small differences to sit inside the noise (one 500-task pass: ± 4.4 pts at 54 %; the
paper's 27B LoRA gain was +2.0 pts), so the report says whether the gain is significant, not only its sign.

**Timeline.** 11-13 laptop: req_log, mc_prep, mc_render, the ADAPTER knob, `--device-map`, paired.py (tested on fakes).
11-14 box: steps 0–2, 31B training starts. 11-15 → 11-16 31B eval; E4B train + eval on whichever box is free.

### B-9 part 1 · the baselines

**Why.** The user wants the full Verified number (all 500 tasks, the usual way to report it) for both models. Same agent
and grader as B-8 route B: `mini-swe-agent 2.4.6` (bash tool only, step limit 250) against our vLLM (thinking on, tool
parser gemma4, 32K window, temperature 0.2, 8,192 tokens per reply, one pass), official `swebench 4.1.0` grading in the
per-task images. The label to quote: "Gemma 4 <31B | E4B> (w4a16 QAT) + mini-swe-agent, 1 pass". Not our harness; nothing
here tests a LoRA.
How (new step `full` in `verified_box.sh`): the 500 tasks in chunks of CHUNK (100). Each chunk is split over the box's
cards (one vLLM + one slice per card, run in parallel) → graded → its images deleted → next chunk. Only one chunk's images
are on the disk at a time (all 500 ≈ 120 GB would not fit). A chunk that has a grading report is skipped, so after any
crash the same two commands (serve, full) resume. Tested on the laptop with a fake agent, grader and docker (chunks,
slices, ports, resume, the disk guard, deletion of only `sweb.eval.x86_64.*` images, the summed report); not with real
vLLM / Docker. Time (a guess, not measured): E4B on 4 cards ~4–6 h; 31B on 8 cards ~5–8 h. Both boxes run at the same time.
Supersedes "the 31B reference on 4 cards" in B-8 (the 500 include the 50 Verified-mini tasks; `report` prints those 50 too).

**Step 0 · on EACH box (read only, 1 min).**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
ls .venv-eval/bin/vllm .venv-swe/bin/mini-extra 2>&1 | cut -c1-80
docker run --rm hello-world 2>&1 | grep -m1 Hello; docker info 2>/dev/null | grep -E "Username|Docker Root Dir"
df -h "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" | tail -1; nproc
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ls ~/.cache/huggingface/hub | grep -i gemma-4
```
📷 both boxes. Read it:
- `.venv-eval/bin/vllm` missing → this box was never set up for our harness → stop, photo (the laptop writes its setup).
- `.venv-swe/bin/mini-extra` missing → `bash scripts/verified_box.sh setup` (📷 its versions line).
- No "Hello from Docker" → stop (Docker without sudo is required).
- Free on the Docker root ≥ 60 GB → CHUNK=100 (default) · 30–59 GB → `CHUNK=50 MIN_FREE_GB=25` · < 30 GB → stop.
- No "Username" → Docker Hub pulls are anonymous and rate-limited; 500 images may hit "toomanyrequests". Optional:
  `docker login` with a free Docker account (the user types it in; nothing else needs it).
- `nproc` ≥ 32 → EVAL_JOBS=16, else 8. A card busy → stop that job first (e.g. `GPU=3 bash scripts/verified_box.sh stop`).
- The model not in the HF cache → serve downloads it (an HF token with the Gemma licence accepted; 31B ≈ 20 GB).

**Step 1 · 8-card box: 31B.**
```
export MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPUS="0 1 2 3 4 5 6 7" PORT=8011 TAG=vf_31b SUBSET=verified WORKERS=6 EVAL_JOBS=16
bash scripts/verified_box.sh serve                                        # 8 × "vLLM up"     📷
nohup bash scripts/verified_box.sh full > runs/verified/vf_31b.full.log 2>&1 &
```
**Step 1 · 4-card box: E4B.**
```
export MODEL=google/gemma-4-E4B-it-qat-w4a16-ct GPUS="0 1 2 3" PORT=8010 TAG=vf_e4b SUBSET=verified WORKERS=8 EVAL_JOBS=16
bash scripts/verified_box.sh serve                                        # 4 × "vLLM up"     📷
nohup bash scripts/verified_box.sh full > runs/verified/vf_e4b.full.log 2>&1 &
```
Use the CHUNK / MIN_FREE_GB / EVAL_JOBS values step 0 picked (add them to the export line). Progress, any time:
`grep -E '^-- chunk|^== chunk|graded|STOP' runs/verified/$TAG.full.log` — one `== chunk` line per graded chunk.

**Step 2 · check after chunk 0 (📷 the grep lines).** Stop (`pkill -f "verified_box.sh full"; pkill -f mini-extra`) and
photo `tail -20 runs/verified/$TAG/c0.part0.run.log` if chunk 0 has < 90 predictions (< 45 with CHUNK=50), or `graded`
shows errors > 10 or NO REPORT, or a part line says 0 predictions. Otherwise let it run.

**Step 3 · when `== full … done` is printed (📷 the grep lines + the report).**
```
bash scripts/verified_box.sh report                                       # 📷 the == block (per repo + the 50 mini tasks)
bash scripts/verified_box.sh stop                                         # frees the cards (same exports)
```
If it stopped early (STOP line, crash, reboot): free disk or lower CHUNK, `serve` again if the vLLMs are gone, run the same
`full` line: graded chunks are skipped. Delete `runs/verified/$TAG/c<k>/` only to redo chunk k.

**Read it (set before the data).**
- Each number is pass@1 on 500 tasks: the 95 % interval is about ± 2 pts at 6 %, ± 4 pts at 50 %.
- E4B: B-8 had 3 / 50 on Verified-mini. The report's "on the 50 Verified-mini tasks" line is the same model again
  (sampling noise only): 0–7 of 50 is consistent; more → say so.
- 31B against the only public figure (Vals AI: 53.92 % on its Verified subset, full precision via Google, its own harness;
  not this setup): ≥ 40 % → our setup is near that level; 25–39 % → the setup costs some (count ContextWindowExceeded in
  the exit statuses: the 32K window); < 25 % → our setup, not the model, dominates — read the exit statuses before quoting it.
- E4B vs 31B on the same 500 tasks is the size gap within one setup; the per-repo line shows where it comes from.
- Costs: box time only (free); no token totals (mini's cost tracking is off for local models).

## B-8 · SWE-bench Verified on the box (2026-11-11) — step 0 ✓ (Docker ok) · route B ✓ **E4B 3 / 50 on Verified-mini** · ▶ 31B reference on 4 cards

**Why.** The user wants E4B's SWE-bench Verified number (no public one: the Gemma 4 model card has LiveCodeBench, Codeforces,
Tau2 only). Two routes, picked after this photo:
- **B (comparable to papers):** the official per-task SWE-bench images + a small agent (mini-swe-agent) against vLLM E4B, graded
  by the `swebench` harness, on Verified-mini (50 tasks) first. Needs Docker without sudo and ~60 GB for images.
- **A (our harness, the LoRA test bench):** our tasks format + subprocess sandbox, like `rebench_build`. Our verify step runs
  pytest, so django (231 of 500) and sympy (75) cannot be graded; compiled repos (matplotlib, scikit-learn, astropy) will
  mostly fail on Python 3.12; requests is a competition repo → expect a biased 50–100-task subset. No Docker needed.
Nothing is installed, pulled or deleted in this step except the tiny `hello-world` image.
```
cd gemma4-dev-agent && git pull && git log --oneline -1
docker info 2>&1 | grep -E "Server Version|Storage Driver|Docker Root Dir|ERROR|permission" ; docker run --rm hello-world 2>&1 | tail -2
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | head -8      # e.g. swebench-sandbox (B-3h never ran)
df -h . "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" 2>&1 | tail -2; nproc; free -g | head -2
```
📷 all of it. Read: `Hello from Docker!` printed → Docker works without sudo → route B possible if the Docker root's disk has
≥ 60 GB free. `permission denied` / `command not found` → route A only (or the user adds docker rights). The laptop then
writes the chosen route's steps.

**Step 0 result (user, 2026-11-11): "检查通过"** → route B. (The photo's numbers were not sent: Docker root free GB, nproc.)

### B-8 route B · E4B on Verified-mini (50 tasks) with mini-swe-agent + the official grader (one card, ~1.5–2 h)

What it is: `scripts/verified_box.sh` = vLLM (our `serve_vllm.sh`, no adapter, thinking on, tool parser gemma4) on ONE
card + `mini-extra swebench` (mini-swe-agent 2.4.6: one `bash` tool, step limit 250, its own prompt) in the official
per-task images (`swebench/sweb.eval.x86_64.*`, pulled on first use) + `swebench==4.1.0` grading. Dataset
`MariusHobbhahn/swe-bench-verified-mini` (50 of the 500 Verified tasks, ≈ 5 GB of images; full Verified ≈ 120 GB). This is
NOT our harness: the number is comparable to papers, not to our dev splits. Not tested on the laptop (would need the pip
packages + Docker images here); the report parser was tested on fake files. Any free card works (GPU=).
```
cd gemma4-dev-agent && git pull && git log --oneline -1
bash scripts/verified_box.sh setup                         # 📷 installed versions (PyPI: mini-swe-agent, swebench, litellm)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader        # pick a free card, e.g. 7
GPU=7 bash scripts/verified_box.sh serve                   # 📷 "vLLM up …"
GPU=7 SLICE=0:2 WORKERS=2 TAG=vm_smoke bash scripts/verified_box.sh run       # smoke, ~5–15 min   📷 the == line
TAG=vm_smoke bash scripts/verified_box.sh report           # 📷
```
**Smoke stop rules:** 0 predictions, or both exit statuses are errors (format / API / docker pull) → stop + photo of
`tail -30 runs/verified/vm_smoke.run.log`. A `ContextWindowExceeded` or an empty patch on 2 tasks alone is not a stop.
```
GPU=7 TAG=vm_e4b bash scripts/verified_box.sh run          # 50 tasks, 8 workers, ~1 h          📷 the == line
TAG=vm_e4b bash scripts/verified_box.sh eval               # Docker grading, ~20–40 min          📷 the tail
TAG=vm_e4b bash scripts/verified_box.sh report             # 📷 the == block
GPU=7 bash scripts/verified_box.sh stop
```
**Read it (set before the data).** No published number exists for E4B (the model card has LiveCodeBench 52 %, Tau2 42 %).
For scale only: SWE-Gym's 32B Qwen base was ≈ 7 % on full Verified before fine-tuning, a LoRA study found 7B models near 0.
- The report's `resolved n / 50` is E4B's Verified-mini pass@1 in mini-swe-agent; with n = 50 the 95 % interval is about
  ± 10 pts, so it is a rough level, not an A/B measure.
- Exit statuses: many `ContextWindowExceeded` → the 32K window, not the model, limits it (mini has no context guard);
  many format errors → E4B's tool calls do not fit mini's bash tool; say which dominates.
- ≥ 5 resolved → route A (our harness subset) is worth building next as the LoRA test bench; 0–2 → Verified is too hard
  for E4B to show a LoRA effect in resolved counts; the laptop then proposes per-edit metrics instead.
Optional reference: the 31B on the same 50 tasks — on 4 cards, see "the 31B reference on 4 cards" below.

**Result (photo, 2026-11-11): resolved 3 / 50 (6 %; Wilson 95 % ≈ 2–16 %).** Grading: submitted 50, completed 39, empty patch
11, errors 0. Exit statuses: 47 `Submitted`, 3 `ContextWindowExceededError` (so ≥ 8 tasks submitted an empty diff). Model
calls per task p50 16, p90 39. The eval image pulls took the disk from ~100 to ~80 GB free. Reading: 3 falls in the gap
between the pre-set bands (0–2 / ≥ 5), on the low side; at this level a 50-task resolved count cannot show a LoRA effect
(+2 tasks is inside the noise), so route A is NOT built now, and the LoRA test stays on our harness (rebench split, per-edit
metrics + resolved). Not yet read: why the 11 patches are empty (format, give-up, or edits outside the repo).

### B-8 route B · the 31B reference on 4 cards (user: "31B 对照 … 帮我写个使用4张GPU加速的script", 2026-11-11) — superseded by B-9 (31B on all 500 on the 8-card box); run it only if B-9 is dropped

Same 50 tasks, same agent, same grader as the E4B row, model = 31B. `verified_box.sh` now takes GPUS: one vLLM per card
(ports PORT, PORT+1, …), the 50 tasks split into 4 contiguous slices (0:12, 12:25, 25:37, 37:50), one mini-extra per slice
in parallel (WORKERS = 8 each, 32 agents in all), predictions merged into one preds.json, then one eval and one report.
The images are already on the disk from the E4B run. Tested on the laptop with a fake mini-extra and fake servers (slices,
ports in each config, merge, report over the parts); not with real vLLM / Docker.
```
cd gemma4-dev-agent && git pull && git log --oneline -1
GPU=3 bash scripts/verified_box.sh stop                    # the E4B vLLM of route B still holds card 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader        # 4 free cards; else another GPUS=
export MODEL=google/gemma-4-31B-it-qat-w4a16-ct GPUS="0 1 2 3" PORT=8011 TAG=vm_31b
bash scripts/verified_box.sh serve                         # 4 × "vLLM up", ~3–10 min        📷
bash scripts/verified_box.sh run                           # ~1 h (guess)                   📷 part lines + == line
bash scripts/verified_box.sh eval                          # ~20 min                        📷 the tail
bash scripts/verified_box.sh report                        # 📷 the == block
bash scripts/verified_box.sh stop                          # frees the 4 cards
```
**Stop rules.** A vLLM does not come up → photo of `tail -30 runs/verified/vllm_gpu<card>.log`. A part with 0 predictions
→ photo of `tail -20 runs/verified/vm_31b.part<k>.run.log` (the other parts' results still count; say how many merged).
**Read it.** Next to E4B's 3 / 50 (same tasks, so a paired comparison): 31B ≥ 11 → the model size, not the agent/setup,
limits E4B here; 31B ≤ 5 → mini-swe-agent + this setup is hard for Gemma 4 in general (look at the exit statuses and empty
patches before blaming the model). 6–10 → in between, say so. One pass each, n = 50: a level, not a precise gap.

### B-8 route A · Verified in OUR harness on 4 cards — written on a misread request (the user meant the 31B reference below); do NOT run unless the user asks

What it is: `scripts/verified_a.sh` = the SWE-rebench pipeline (rebench_build → gold → lane.sh) on SWE-bench Verified
(princeton-nlp/SWE-bench_Verified @c104f84). Our prompt, our 9 tools, the subprocess sandbox, pytest verify, the E4B
latest setting with NO adapter (ADAPTER=) — the bench a LoRA trained on our format can be tested on (route B cannot).
Skipped repos: django + sympy (not pytest), requests (competition repo), and by default astropy / matplotlib /
scikit-learn (C extensions are not built in the workspace the sandbox imports first → cannot pass gold; `SKIP_REPOS=""`
tries them at hours of failed compiles). Expected kept ≈ 98 tasks (sphinx 44 · xarray 22 · pytest 19 · pylint 10 ·
seaborn 2 · flask 1; `prep` prints the real counts). Install specs come from swebench 4.1.0 itself (`.venv-swe` of route B):
pinned first, then unpinned (numpy < 2) because the sandbox Python is 3.12, not the specs' 3.9–3.11. "4 cards" = one
full pass per card in parallel: 4 seeds in the wall time of one (a bench needs the mean; one pass split 4 ways is not built).
Tested on the laptop: syntax, the specs conversion on swebench 4.1.0's constants source, `prep` on a fake parquet, the
install fallback chain offline on a tiny project, `report` on fake runs. Not tested: a real build / gold / lane.
```
cd gemma4-dev-agent && git pull && git log --oneline -1
GPU=3 bash scripts/verified_box.sh stop                    # route B's vLLM still holds card 3
df -h . | tail -1                                          # ≥ 40 GB free wanted (build stops itself at 20)
bash scripts/verified_a.sh prep                            # ~1 min, internet               📷 all
bash scripts/verified_a.sh build                           # ~30–60 min, CPU + internet     📷 == build + deps lines
bash scripts/verified_a.sh gold                            # ~10–20 min, CPU                📷 == gold + per repo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader     # 4 free cards; else GPUS="4 5 6 7"
GPUS="0 1 2 3" bash scripts/verified_a.sh run              # 4 passes at once, ~1–2 h       📷 the 4 pass lines
bash scripts/verified_a.sh report                          # after all 4 lanes end          📷 == block + edit stats
```
Progress while the lanes run: `tail -2 runs/verified_a_lane*.log`. Faster but not the scorer's shape: `LANE_CONC=2`.
**Stop rules (set before the data).** prep: kept far from ≈ 98 or many "without a spec" → photo. build: `== build` < 80
built → photo of the deps lines. gold: ok < 30 → stop (bench too small to show a LoRA effect; the laptop then decides
between SKIP_REPOS="" and dropping route A); 30–49 → go and say so; ≥ 50 → go. run: any pass with fewer results than
ok.txt has lines → photo of `tail -20` of that lane log.
**Read it.** A useful LoRA bench needs room both ways: E4B's mean resolved between ~15 % and ~70 % of ok, and a pass-to-pass
spread small next to the effect we hope for (the report prints min–max). The pooled edit stats (not-found rate, loops) are
the primary readout of the edit-copy LoRA (line 5); resolved is the guard. Repo overlap: B-7's 450 tasks include 2 seaborn,
1 xarray, 2 sphinx tasks (prep prints the overlap) → a LoRA tested here must be trained without those repos' tasks.
A LoRA A/B later: `ADAPTER=<name> GTAG=vb bash scripts/verified_a.sh run` on the same ok.txt, then report with GTAG=vb.

---

## B-7 · More tasks for E4B's own rollouts: 450 more public SWE-rebench tasks → build → gold → 8 E4B passes (2026-11-11) — steps 1–3 ✓ (422 built, 161 pass gold, 8 passes: 1,277 failed edits ≥ 400) · ▶ step 4: the relabel stage (CPU, ~10 min) → step 5 train → step 6 A/B

**Why.** Line 5 (user, 2026-11-11: "这个会话做5，数据肯定是要增强的" → "box用4个GPU，扩题到300题"): fix E4B's edit-copy errors with
labels made from its OWN rollouts: every failed `edit_file` gets a corrected call (the true span of the file at that moment, the
wrong quote in the thought fixed), every ok edit is a positive. The 67 kit tasks gave ~29 not-found edits per pass; that is too
little, so first more tasks. This step only makes the rollouts; the relabel stage is written on the laptop after the photos.
Building a task needs CPU + network only (rows from the pinned parquet, a git snapshot, `uv pip install --target`); the GPU is
needed only for the rollouts.

**The task list** (laptop, pushed): `splits/rebench_more450.txt` = `scripts/rebench_select.py` (seed 0, deterministic): 450 tasks
in 349 repos, ≤ 2 per repo, none of the kit's 81, not fastapi / rich / requests / httpx / the two named OOD-45 repos; Python
3.9–3.12, pip-only installs (pre_install at most `apt-get install -y gcc`), gold patch 1–3 files and ≤ 80 lines, 1–20 pytest
FAIL_TO_PASS ids, SWE-rebench's own LLM grades issue ≤ 2 / tests ≤ 2 / difficulty 0–2, created ≥ 2020. Expect about 2/3 to
survive build + gold → ~300.

New knobs (all printed in the header line): `BUILD_JOBS` (build processes, tasks dealt by repo) · `LEAN=1` (blobless clones,
deleted after use) · `MIN_FREE_GB` (the build stops, keeping what it built, below this) · `SNAP_MAX_MB` (bigger snapshots are
dropped) · `UV_CACHE` (one hard-linked wheel cache on the same disk: on the laptop 3 tasks took 1.0 GB instead of 1.6) ·
`KILL_VLLM=0` (leave other servers alone) · `ARM_BY_PASS` (pass i gets word i: `base` = temp 0.2, `t07` = temp 0.7) ·
`LANE_CONC` (tasks at once inside a lane; 2 here: data, not a score). Laptop test: 6 tasks, 3 build processes, merge OK, one
147 MB snapshot dropped by the cap.

**Step 0 · checks (no GPU, 1 min)**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
df -h . | tail -1; nproc; which uv; grep -vc '^#' splits/rebench_more450.txt     # 450
du -sh runs/rebench/site runs/rebench/snapshots runs/rebench/git 2>/dev/null
```
📷. Stop + photo if less than **60 GB** free: then `du -sh runs/* ~/.cache/huggingface/hub/* 2>/dev/null | sort -h | tail -15`
(📷) and the laptop says what can go. Nothing is deleted in this step.

**Step 1 · build (CPU + network, ~1–1.5 h; may run while B-6-D's lanes run)**
```
OUT=runs/rebench2 IDS_FILE=splits/rebench_more450.txt BUILD_JOBS=6 LEAN=1 MIN_FREE_GB=20 SNAP_MAX_MB=60 UV_CACHE=runs/rebench2/uvcache \
  nohup bash scripts/rebench_gen.sh build > runs/rebench2_build.out 2>&1 &
tail -3 runs/rebench2.build*.log          # progress
cat runs/rebench2_build.out               # 📷 at the end: the header, the failed / partial / dropped lines, 6 TOTAL lines, the == build line
```
`runs/rebench` (the kit's 67) is not touched. Read: built ≥ 350 → go on. A `STOP: … GB free` line → stop + photo.
BUILD_JOBS=6 if `nproc` ≥ 32, else 3.

**Step 2 · gold (CPU, ~1–2 h; after B-6-D's lanes have ended: the test runs would slow its lanes)**
```
OUT=runs/rebench2 CONC_GOLD=16 bash scripts/rebench_gen.sh gold      # 📷 the == gold line + the NOT list (long: the first 40 lines are enough)
```
CONC_GOLD=16 if `nproc` ≥ 32, else 8. Read: ok ≥ 250 → step 3. 150–249 → step 3 anyway, say so. < 150 → stop + photo of
`grep -E "NOT" runs/rebench2/gold.log | head -40`.

**Step 3 · rollouts: 8 E4B passes, one per card, no adapter (~3 h with the 161 tasks that passed gold; all 8 cards)**

Changed 2026-11-11 (user: "你改一下，跑8张卡"): gold left 161 tasks, not ~300, so 4 passes would give about 300 failed edits
(under the 400 line). Eight passes on eight cards take the same wall time and double the material. Nothing else may run
on the cards meanwhile: the first line below must show all 8 free. If B-8 holds a card (it uses GPU 7), either wait for its
`stop`, or leave that card out: `GPUS="0 1 2 3 4 5 6" PASSES=7 ARM_BY_PASS="base base base base t07 t07 t07"` (then m5–m7 are the
temp-0.7 passes and the lines below take `m{5..7}`).
```
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader          # 📷 cards 0–7 must be free (< 2,000 MiB)
E4B=google/gemma-4-E4B-it-qat-w4a16-ct
OUT=runs/rebench2 GPUS="0 1 2 3 4 5 6 7" PASSES=8 ARM_BY_PASS="base base base base t07 t07 t07 t07" LANE_CONC=2 KILL_VLLM=0 LATEST=e4b ADAPTER= MODEL=$E4B TOK=$E4B \
  GTAG=m KEEP=all SFT=runs/sft/rebenche4b_more SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct bash scripts/rebench_gen.sh gen   # 📷 the 8 pass lines
```
Same settings as the e/f passes otherwise (TIME_MIN 7, MAX_TURNS 80, NO_TESTDEPS as then). Runs: `runs/dev33_base_m1` …
`runs/dev33_base_m4` (temp 0.2), `runs/dev33_t07_m5` … `runs/dev33_t07_m8` (temp 0.7). The lane logs are `runs/rebench_lane0..7.log`
(the same names B-6-D used: read B-6-D's lora line before this step). When all 8 lanes have ended:
```
for r in runs/dev33_base_m{1..4} runs/dev33_t07_m{5..8}; do echo "$r results $(wc -l < $r/task_results.jsonl) resolved $(grep -c '"resolved": true' $r/task_results.jsonl)"; done
grep -hE "eval concurrency|^== (base|t07):" runs/rebench_lane0.log runs/rebench_lane4.log | cut -c1-220     # concurrency 2 · temperature 0.2 / 0.7
python3 scripts/edit_stats.py runs/dev33_base_m{1..4} runs/dev33_t07_m{5..8}
```
📷 all of it. Do NOT run the `data` stage (the relabel stage replaces it; written on the laptop after this photo).

**Read it (set before the data).**
- Every pass has as many results as `runs/rebench2/ok.txt` has lines; otherwise stop + photo of `tail -20` of that lane log.
- The pooled block of edit_stats: not found + several matches = the raw material. ≥ 400 → the relabel stage is worth writing.
  200–399 → it is, but say the number. < 200 → stop: more passes would be needed first (the laptop decides).
- Resolved per pass: about a third (E4B resolved 24–29 of the kit's 67). Far below → check the lane log for a missing site or a
  test that never ran.
- Temp 0.7 vs 0.2 (m5–m8 vs m1–m4): the not-found rate per edit and resolved; just a record.

Nothing goes to Kaggle. Cost: own box only, ≈ 8 cards × ~3 h (161 tasks) + ~3 h CPU.

**Step 3 result (box photo, 2026-11-11 18:34; 8 cards, ~58 min, not the ~3 h planned).**
- Every pass has 161 / 161 results.
- Edits, pooled: 3,468 = ok 2,055 (59 %) · not found 1,189 (34 %) · several matches 88 (3 %) · other 136 (4 %). Not found +
  several = **1,277 ≥ 400 → the relabel stage is worth writing.**
- The first edit of a file fails with not-found in 16 %. After a failed edit, the next edit repeats the same `old_string` 608
  times (49 %). Runs of ≥ 2 failed edits in a row: 310, in 288 task runs (22 %), longest 12.
- Resolved 25–31 of 161 per pass, 231 / 1,288 (18 %), under the "about a third" of the read rule, so the plan's check was
  made: 12 of 1,288 runs ended in a run error (4 session timeouts, 4 without a submission, 2 context limit, 1 patch that did
  not apply, 1 call cap); no missing site, no test that never ran. Per task: 111 never solved, 50 at least once, 11 in all 8.
- Temp 0.2 vs 0.7: resolved 115 vs 116; not-found share about 34 % in both.
- Laptop note for the relabel stage (not measured on the box): 608 of the failed edits repeat the previous `old_string`, so
  the number of different failed quotes is well under 1,277. Count them after de-duplication before sizing the training.

**The relabel stage (laptop, 2026-11-11; this session took B-7 over).** `train/relabel_edits.py` (no model): every trajectory
is replayed on its snapshot (write_file, the diff of each ok edit; a command that may write files → the files are unknown
until read whole), so the true text of the file at the moment of each failed edit is known. A not-found edit then gets
`old_string` := the true span closest to the model's quote, and `new_string` := the model's own change carried over to it,
line by line; the wrong line in the thought is replaced too. It is kept only if the corrected call would apply exactly once,
a `.py` file still parses, and every corrected line was in a tool output of the model's context. Left out: the change is
already in the file · no span close enough (an invented quote) · two equally close spans · file state unknown.
A corrected call is a training target on its own sequence (the history is E4B's own, up to that call); ok edits are added
1 : 1 as native targets; no other call is a target. 10 % of the repos are held out (the eval loss).

Laptop run on the 11 no-adapter passes of the first 67 tasks (the trajectories from the private dataset, snapshots rebuilt
here; 20 s):
- Replay check: 1,644 of 1,644 `read_file` results equal the replayed file lines.
- 345 not-found edits (edit_stats says 401: it counts a tool result again each time the history is re-sent after a
  compaction, so the 1,277 of step 3 is too high by about the same share) → corrected 123 (36 %) · no anchor 52 · already
  applied 66 · file state unknown 44 · conflict 18 · not visible 15 · syntax broken 15 · ambiguous 12.
- 123 corrected = 76 different calls → 108 kept (at most 2 per call: one first failure, one after an error) on 28 tasks.
  What was wrong in the quote: 1 line 49 · 2–3 lines 33 · 4+ lines 18 · lines left out 6 · whitespace only 2. 50 of the 345
  quotes had escape junk (`\"` for `"`, `\\` for `\`). In 42 of the 108 the wrong line was the one being changed (mostly a file E4B had
  already edited, quoted as it was before). The thought quoted the wrong line in 103 of 108.
- Rendered: 199 sequences, 2.19 M tokens, 270 K label tokens, `RESULT PASS`; longest 20,205 tokens.
- Expected on the box (8 new passes + these 11): roughly 450–550 corrected, ~800 sequences, ~50 optimizer steps per epoch.

**Step 4 · data (CPU, ~10 min; nothing else needed)**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
ls -d runs/dev33_base_m{1..4} runs/dev33_t07_m{5..8} runs/dev33_base_e{1..3} runs/dev33_base_f{1..8} | wc -l      # 19
ls runs/rebench2/snapshots | wc -l; ls runs/rebench/snapshots | wc -l                                              # 422 · the first task set (≥ 67)
bash train/relabel_box.sh data        # 📷 every line from 'relabel_box data' down
```
If the 11 e/f passes or `runs/rebench/snapshots` are gone: `RUNS="$(echo runs/dev33_base_m{1..4} runs/dev33_t07_m{5..8})"
TASKS=runs/rebench2/tasks.jsonl SNAP=runs/rebench2/snapshots bash train/relabel_box.sh data` (say so).
**Read it (set before the data).**
- `replay check` ≥ 99 % of the reads equal. Lower → stop + photo: the replay is not the file the model saw.
- `corrected … kept` ≥ 300 → go on. 150–299 → go on, say the number. < 150 → stop + photo (too little to train on).
- Both render lines `RESULT PASS`; `trajectories … raised 0`. Else stop + photo of `tail -5 runs/sft/e4b_relabel.render.log`.
- The last `== optimizer steps` line names the EPOCHS for ≥ 150 steps. "still under 150" → stop + photo.

**Step 5 · train (all cards; the drill line's trainer on these files)** — EPOCHS = the number the steps line named.
```
ALL=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
SET=_relab NAME=e4b_relab EPOCHS=<from the steps line> LR=3e-5 EVAL_EVERY=25 GPUS=$ALL nohup bash train/drills_box.sh train > runs/relab_train.log 2>&1 &
tail -5 runs/relab_train.log        # 📷 at the end: the schedule line, the eval[...] lines, every '== epoch' line, the checksum line
```
r16 α16, as the drill adapters. Epoch to use = the one with the lowest held-out loss (call it N). Stop + photo on
`TRAIN FAILED`, or if the schedule line shows fewer than 150 steps (B-6's adapter had 33 and did not move).
```
SET=_relab NAME=e4b_relab EPOCH=N bash train/drills_box.sh subs        # 📷 → runs/adapters/e4b_relab + the build check
```
**Step 6 · A/B on the 51-task split (tasks of neither task set): 4 passes without vs 4 with, one wave (8 cards, ~3–4 h)**
```
ls -d runs/dev33_base_[xy][0-7] 2>/dev/null        # must list nothing (else stop + photo: the tags are taken)
pkill -f "vllm serve"; sleep 5; E4B=google/gemma-4-E4B-it-qat-w4a16-ct
for g in 0 1 2 3 4 5 6 7; do AD=$([ $((g % 2)) = 1 ] && echo e4b_relab); GPU=$g TAG=$([ -n "$AD" ] && echo y || echo x)$g TIME_MIN=8 MAX_TURNS=80 LATEST=e4b ADAPTER=$AD MODEL=$E4B SPLIT=splits/disc_e4b.txt SUBMISSION=runs/base_lb024 ARMS=base TRAJ=1 nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
# when all 8 lanes have ended:
for k in x y; do python3 -c "import glob,json,sys; c=[sum(json.loads(l)['resolved'] for l in open(f)) for f in sorted(glob.glob(f'runs/dev33_base_{sys.argv[1]}[0-7]/task_results.jsonl'))]; print(sys.argv[1], c, 'mean', sum(c)/max(1,len(c)))" $k; done   # 📷  x = no adapter, y = adapter
python3 scripts/edit_stats.py --pooled runs/dev33_base_x{0,2,4,6}; python3 scripts/edit_stats.py --pooled runs/dev33_base_y{1,3,5,7}   # 📷 both blocks
grep -m1 -o " lora: [^·]*" runs/lane1.log
```
If B-8 holds a card, run 3 vs 3 on six cards (drop g = 6 7) and say so.
**Read it (set before the data).**
- Edit level (moves first): the adapter's `not_found` share of the edit calls is lower than no adapter's by ≥ 8 points
  (pooled, 4 passes each) → the adapter fixes the quoting. Within ± 4 → it does not. In between → no verdict.
- Tasks: the rule of B-3o. The adapter counts only if its 4-pass mean beats no adapter's by ≥ 3 tasks AND its worst pass ≥
  no adapter's best − 1. +1 to +2 → no verdict. ≤ 0 with the edit level moved → the quoting was not what costs tasks.
- Stop + photo if a lane log shows no adapter mounted on an odd card, or a pass has fewer than 51 results.
Nothing goes to Kaggle from this step. Cost: own box only, ≈ 10 min CPU + training (8 cards, about 1 h) + 8 cards × 3–4 h.

---

## B-6 · A LoRA from E4B's own resolved runs on the flip tasks (rejection-sampling fine-tune) (2026-11-10) — ✓ ran 2026-11-11: ■ STOP (51-task split, 4 vs 4 passes: no adapter 14.5, adapter 12.25)

**Result (2026-11-11).** No adapter 12 · 16 · 15 · 15 (mean 14.5); `e4b_rft` 14 · 11 · 11 · 13 (mean 12.25): −2.25, rule needs ≥ +3 → stop.
The user agreed to the diagnostic (2026-11-11): it is the block below.

### B-6-D · Diagnostic: does `e4b_rft` reproduce its own successes on the 67 training tasks? (2026-11-11) — ✓ ran 2026-11-11: trained flip tasks 48 % → 47 % (within ± 10 pts): imitation of the resolved runs does not remove the variation

**Result.** 8 adapter passes 27 · 28 · 26 · 25 · 24 · 24 · 23 · 25 (mean 25.25; no adapter 26.1 over 11). always 100 % → 99 % · flip in training
48 % → 47 % · flip held out 39 % → 22 % (32 runs only) · never 0 % → 1 % · all 39 % → 38 %. No damage, no gain. B-6 is closed.

It explains the stop, it does not reopen it. Eight adapter passes (one per card; user 2026-11-11: all 8 cards may be used) on the tasks the adapter was trained on, read per task
class against the 11 no-adapter passes.
```
cd gemma4-dev-agent && git pull && git log --oneline -1
ls runs/adapters/e4b_rft/adapter_model.safetensors runs/rft/heldout | head -3; ls -d runs/dev33_base_e{1..3} runs/dev33_base_f{1..8} | wc -l    # 11
E4B=google/gemma-4-E4B-it-qat-w4a16-ct
GPUS="0 1 2 3 4 5 6 7" PASSES=8 LATEST=e4b ADAPTER=e4b_rft MODEL=$E4B TOK=$E4B ARM=base GTAG=rft KEEP=all SFT=runs/sft/rebenche4b_rft SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct bash scripts/rebench_gen.sh gen     # 📷; same settings as the f passes (incl. NO_TESTDEPS as then)
# when the 8 lanes have ended (do NOT run the `data` stage):
grep -m1 -o " lora: [^·]*" runs/rebench_lane0.log; python3 scripts/stab.py runs/dev33_base_e{1..3} runs/dev33_base_f{1..8} --vs runs/dev33_base_rft{1..8} --held runs/rft/heldout     # 📷 every line from '== VS' down
```
**Read it (set before the data; 26 trained flip tasks × 8 passes ≈ 208 runs, so about ± 7 pts).**
- `flip, in training` difference ≥ +15 pts → the adapter did learn its own successes and they do not carry over to other
  tasks: the data is too narrow (30 tasks), not the method.
- within ± 10 pts → it did not even make its own tasks more reliable: imitating the resolved runs does not remove the
  variation; more data of this kind would not help either.
- ≤ −10 pts, or `always` down by ≥ 10 pts → the adapter damages the model (training strength / epoch).
- In between → say so. Stop + photo if the lane log shows no adapter mounted, or a pass has fewer than 67 results.
Nothing is started after it.

**Why.** B-5: of the 67 tasks 12 are always resolved, 25 never, 30 flip; the flip tasks have 155 resolved runs of 330 (47 %),
16 of them are resolved in at most half the passes. Mean 26.1 per pass; if every flip task always went right it would be
42 (+15.9). That is the ceiling, not the expected gain: no adapter has beaten no adapter so far. What is new here: the
states are E4B's own, the format is its own, every kept run passed the tests, and failed calls are not targets.

**Step 0 · checks (no GPU)**
```
cd gemma4-dev-agent && git pull && git log --oneline -1
ls -d runs/dev33_base_e{1..3} runs/dev33_base_f{1..8} | wc -l                       # 11
ls .venv-train/bin/python runs/dequant/gemma-4-E4B-dq/config.json                   # missing → GPU=0 bash train/train_box.sh env dequant  (📷 its end)
ls harness/tasks.jsonl splits/disc_e4b.txt && ls harness/snapshots | wc -l          # the 51-task split of step 4; missing → stop + photo
df -h . | tail -1
```
**Step 1 · data (CPU, ~10 min)**
```
bash train/rft_box.sh data        # 📷 every line from 'rft_box data' down
```
Per flip task at most 4 resolved runs (fewest model calls first); 4 flip tasks are kept out whole (the eval loss).
Stop + photo if: `rft_select` does not say flip 30 · train < 60 runs · a render line is not `RESULT PASS`.

**Step 2 · train (all cards; the drill line's trainer on these files)**
```
ALL=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
SET=_rft NAME=e4b_rft EPOCHS=3 LR=3e-5 EVAL_EVERY=25 GPUS=$ALL nohup bash train/drills_box.sh train > runs/rft_train.log 2>&1 &
tail -5 runs/rft_train.log        # 📷 at the end: the schedule line, the eval[...] lines, every '== epoch' line, the checksum line
```
r16 α16, as the drill adapters. Epoch to use = the one with the lowest held-out loss (call it N). Stop + photo on `TRAIN FAILED`.
```
SET=_rft NAME=e4b_rft EPOCH=N bash train/drills_box.sh subs        # 📷 → runs/adapters/e4b_rft + the build check
```
**Step 3 · A/B on the 51-task split, tasks the adapter never saw: 4 passes without vs 4 with, one wave (8 cards, ~3–4 h)**
```
pkill -f "vllm serve"; sleep 5; E4B=google/gemma-4-E4B-it-qat-w4a16-ct
for g in 0 1 2 3 4 5 6 7; do AD=$([ $((g % 2)) = 1 ] && echo e4b_rft); GPU=$g TAG=$([ -n "$AD" ] && echo t || echo u)$g TIME_MIN=8 MAX_TURNS=80 LATEST=e4b ADAPTER=$AD MODEL=$E4B SPLIT=splits/disc_e4b.txt SUBMISSION=runs/base_lb024 ARMS=base TRAJ=1 nohup bash scripts/lane.sh run > runs/lane$g.log 2>&1 & done
# when all 8 lanes have ended:
for k in u t; do python3 -c "import glob,json,sys; c=[sum(json.loads(l)['resolved'] for l in open(f)) for f in sorted(glob.glob(f'runs/dev33_base_{sys.argv[1]}[0-7]/task_results.jsonl'))]; print(sys.argv[1], c, 'mean', sum(c)/max(1,len(c)))" $k; done   # 📷  u = no adapter, t = adapter
python3 scripts/rulestats.py runs/dev33_base_u0 runs/dev33_base_t1        # 📷
```
**Read it (set before the data).** The adapter counts only if its 4-pass mean beats no adapter's by ≥ 3 tasks AND its worst pass
≥ no adapter's best − 1 (the rule of B-3o; no adapter was 12–16, mean 14.0 there). +1 to +2 → no verdict, say so. ≤ 0 → stop:
one diagnostic may follow (4 adapter passes on the 67 training tasks: does it even reproduce its own successes?), on the user's word.
Nothing goes to Kaggle from this step.

## B-5 · After B-4: is there room in E4B's own repeat passes, and does submitting on a failure cost anything? (2026-11-10) — ✓ ran 2026-11-10: candidate 1 goes on (30 flip tasks, ceiling +15.9 per pass) → B-6 · candidate 2 dropped (ratio 0.80)

**Why.** B-4 stopped. Two candidates are left that use what the box already has, and both need a count before any training:
1. **Fine-tune on E4B's own resolved runs** (rejection sampling): the 11 base passes on the same tasks (3 of S0, 8 of S0c)
   are E4B's own states with labels checked by the tests. It only has data if tasks flip between passes.
2. **The submit decision**: where E4B submits after a failed command the 31B would mostly run another command. That only
   matters if those runs are resolved less often than the others. (The prompt rule against it, `failstop`, was already in
   the `all` arm of B-3o: no gain, mostly not followed. A hard guard in the harness is not built.)

```
cd gemma4-dev-agent && git pull && git log --oneline -1
python3 scripts/stab.py runs/dev33_base_e{1..3} runs/dev33_base_f{1..8}      # 📷 every line
```

**Read it (rules set before the data).**
- STABILITY → candidate 1 goes on only if flip tasks ≥ 15 AND the ceiling line says ≥ +6 tasks per pass; else it is dropped.
- SUBMIT (flip tasks only) → pool `hidden fail` + `exit != 0` (n ≥ 30 needed). Resolved rate ≤ half of `clean`'s → candidate 2
  goes on (a hard guard first, then the decision arm). ≥ 0.8 × `clean`'s → dropped: holding the submission back cannot gain.
  In between → no decision; say so.
- Stop + photo: a pass with 0 traces, or fewer than 11 passes found (then list `ls -d runs/dev33_base_[ef]*`).
- Nothing is started after this step; the next step is written on the laptop from the photo.

**Laptop reading of the same script (one pass, the 129 evaluation tasks, not these tasks):** first submit after `exit != 0`
51 runs, 16 % resolved; after a clean command 62 runs, 16 %; hidden fail 5 runs, 0 %; no command since the last edit 9 runs, 11 %.

## B-4 · Token-level distillation 31B → E4B, step S0: does the 31B point at E4B's own errors? (2026-11-10) — ✓ S0, S0b, S0c done 2026-11-10 · ■ STOPPED by the pre-set rule (pooled n 60: corrected 38 %, prefers 43 % < 60 %); S1 is not built

**Status (2026-11-10).** S0 ran: of 88 not-found edits 56 were locatable; the 31B's top-1 continued the source at the first wrong
token in 2 % (call) and 12 % (thought) → "stop" by S0's rule. But most measured positions repeat a wrong text that is already
in the context (48 of 56 calls; 39 of 48 thoughts), where any model copies; only 9 were where the text is first written.

### S0c · a second scored set, pooled with S0's (2026-11-10) — ✓ ran 2026-11-10: STOP

**Result.** Pooled determined first occurrences n 60: flagged 52 % · corrected 38 % · prefers 43 % · flagged 27× the control.
Corrected is in the 30–50 % band, which needs prefers ≥ 60 % → STOP. Nothing further to run in B-4; do not write S1.
Kept for later: where E4B submits after a failed last command (n 308) the 31B's top-1 is run in 71 %, submit in 16 %.

**Why.** S0b ran (2026-11-10): every check passed, but only 17 determined births (19 measured) in 201 rollouts — under
the rule's 20, so no verdict. Point values: corrected 47 %, prefers 47 %, flagged 47 % against 1.5 % on the 1,032
control tokens. More of E4B's own rollouts on the same tasks, scored the same way and pooled, give the rule its n
(about 6 births a pass → 8 passes ≈ 50 more). The rule and its thresholds do not change.

```bash
# 0. code, inputs, cards (📷). Needs the rebench task environment on THIS box; if a path is missing → stop, send the photo
cd gemma4-dev-agent && git pull && git log --oneline -1
ls runs/rebench/tasks.jsonl runs/rebench/ok.txt && ls runs/rebench/site | wc -l && ls runs/opd/s0/t31_*.npz runs/opd/s0/s_*.npz | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; df -h . | tail -1
# 1. eight more passes of E4B (no adapter) on the rebench tasks, one per card (~3–4 h; progress: tail -2 runs/rebench_lane*.log)
E4B=google/gemma-4-E4B-it-qat-w4a16-ct
GPUS="0 1 2 3 4 5 6 7" PASSES=8 LATEST=e4b ADAPTER= MODEL=$E4B TOK=$E4B ARM=base GTAG=f KEEP=all SFT=runs/sft/rebenche4b_f \
  SRC=e4b_onpolicy POLICY=gemma-4-e4b-it-qat-w4a16-ct bash scripts/rebench_gen.sh gen          # 📷 the 8 pass lines
# 2. when all 8 lanes have ended: rows, the 31B's scores, E4B's scores for the new set (its own directory: S0's files stay)
export OUTD=runs/opd/s0x RUNS="$(echo runs/dev33_base_f{1..8})" TASKS=runs/rebench/tasks.jsonl
bash train/opd_box.sh s0data                                             # 📷 every line
nohup bash train/opd_box.sh score > runs/opd_score_x.log 2>&1 &          # ~10 min; 📷 the == lines when it ends
nohup bash train/opd_box.sh student > runs/opd_student_x.log 2>&1 &      # AFTER score has ended (~3 min); 📷 the == lines
# 3. the pooled readout: 📷 every line from the first 'counts:' line down
POOL="runs/opd/s0 runs/opd/s0x" bash train/opd_box.sh probe
unset OUTD RUNS TASKS
```

**Reading.** The same `== S0b verdict` line, now on both sets. Lines b / c will no longer equal the S0 photo (two sets).
Stop and send the photo on: a missing path in step 0, a lane that ends without its results, `render` not PASS, a shard
with no result, `STUDENT … FAIL`, a `locator check` below the measured births. Nothing is deleted; `runs/opd/s0` is not
touched (`s0data` only clears its own `OUTD`). The new rollouts are also S1's training data if the verdict is go.

### S0b · the same check where each wrong text is FIRST written, plus E4B's own scores (2026-11-10; locator rewritten the same day) — ✓ ran 2026-11-10: no verdict (n 17 < 20), see S0c

Needs the code whose `kd_probe.py` has `--selftest`, and S0's files on the box (`runs/opd/s0/`: rows.jsonl, recs_*,
t31_*.npz — the 31B scores are reused, nothing is rescored by the 31B). **Do not run `s0data` again with the default
`OUTD`: it deletes the 31B scores.** All cards free for step 1.

```bash
# 0. code, S0's files, cards (📷)
cd gemma4-dev-agent && git pull && git log --oneline -1 && .venv-eval/bin/python train/kd_probe.py --selftest
ls runs/opd/s0/rows.jsonl && ls runs/opd/s0/t31_*.npz | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 1. E4B scores the same rows, one process per card (~3–5 min)
nohup bash train/opd_box.sh student > runs/opd_student.log 2>&1 &
tail -20 runs/opd_student.log          # when it ends: 📷 the '== teacher_score' lines and the '== student' line
# 2. the readout (CPU, ~1–5 min): 📷 every line from the first 'counts:' line down
bash train/opd_box.sh probe
```

**What the readout measures.** Each not-found edit's wrong text is measured where it FIRST appears in the trajectory (a
thought or a call; resends count once), with what had been shown at that point. The first wrong character now follows
what the harness's edit matcher forgives (found in a review before the box ran S0b: the model is shown file content
inside a JSON string, a newline as backslash-n, and often writes it that way; the matcher un-escapes such strings and
ignores indentation, so neither is why an edit fails — on the laptop's test data the old locator sat on such a
character in 15 of 33 cases, in one 1,888 characters before the real error). Readings tried: the text as written with
any whitespace between tokens, and five un-escapings with each line stripped; the latest unexplained character wins; a
string that some reading explains completely is a stale view and set apart. Other classes set apart as before: written
before the source was shown · right as shown then · the shown continuation is not unique · the prefix ends its source.
- `LOCATOR`: how many births moved against S0's locator, and the readings used.
- `BIRTH` (the verdict line): flagged / corrected / prefers over the **determined** births. corrected = the teacher's
  top-1 reaches the wrong character and continues the shown text there (in any way of writing it). A top-1 that stops
  before the wrong character is **undetermined** and left out of n. With E4B's scores: push and lift.
- `CTRL`: the same numbers on the tokens of the same quotes that E4B copied right. push is positive on average for any
  teacher (the token was sampled from E4B), so it only means something against this line.
- `SEV`, `SUBMIT`, `MAP`: as before (SEV in the same normalised reading).
- `TAIL`: the 31B's probability mass outside its top-20 per span — what a top-20 bucketed KL cannot see.
- `STUDENT`: E4B's own NLL of its tokens against the 31B's, per span.
- Lines a, d, e, f and b / c are S0's (b / c with S0's old locator): they must repeat the S0 photo (b n 56 · corrected
  2 %, c n 48 · 12 %) — the proof that the same files were read.

**Reading rule (S0b; written before any data from this locator was seen).** The probe's last `== S0b verdict` line
applies it, on the determined births:
- n ≥ 20 and corrected ≥ 50 % → **go to S1** (training code, then the A/B); corrected < 30 % → **stop the line**;
- 30–50 %: **go** only if prefers ≥ 60 % **and** flagged at the births ≥ 5 × flagged on the `CTRL` tokens (the 31B
  disagrees at E4B's errors, not everywhere); otherwise **stop**;
- n < 20: **no verdict** — send the photo; the laptop gives the line for a second scored set (E4B's evaluation passes
  that kept trajectories: `OUTD=runs/opd/s0x RUNS="…" TASKS=harness/tasks.jsonl` with `s0data`, `score`, `student`,
  then `POOL="runs/opd/s0 runs/opd/s0x" bash train/opd_box.sh probe`).
- `TAIL` decides the K of S1's scoring (user, 2026-11-10: measure first, never store the whole vocabulary — 1.2 M reply
  tokens × 262,144 ≈ 630 GB): mass > 10 % at under 5 % of the positions of every span → K stays 20; else K = 100.
- `prefers`, push / lift, `SEV`, `SUBMIT` and `MAP` do not decide: they set S1's loss scope (code quotes, weighting
  near first occurrences, a tool-choice arm).

**Stop and send the photo** on: the selftest not printing PASS (or saying the harness is not installed), a missing S0
file, a card in use, a `student` shard with no result (the probe still runs without it, only the push / STUDENT / MAP
lines are missing — send both photos), `STUDENT … FAIL` (the model that scored is not the model that sampled), lines
b / c differing from the S0 photo, or a `locator check` count below the measured births (the measurement itself is
broken).

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
