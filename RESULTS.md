# Results log: every run, one row (newest first in each table)

Every experiment run of this repo gets a row here when it finishes: Kaggle leaderboard submissions, private Kaggle
kernels on the scorer's 4×L4 hardware, and runs on our own GPUs. Aggregate numbers only: no task text, task ids, run
logs or patches (Kaggle data must not be redistributed). dev33 = our fixed 33-task split of the 129 public tasks
(`splits/dev33.txt`, `scripts/dev_split.py`).

**Caveat for every row before 2026-10-20 21:00 UTC:** subprocess-sandbox verifications lacked `pytest-httpbin` and
`inline_snapshot`, so up to 10 dev33 tasks were unwinnable (all 4 requests tasks + 6 fastapi tasks; fixed by `scripts/install_testdeps.sh`). Paired comparisons within a
run stand, absolute rates are biased low. Rows after the fix are marked 🧪✓.

## Leaderboard submissions (public LB = 58 hidden tasks; 1 task ≈ 0.017)

| date | what | public | notes |
|---|---|---|---|
| 2026-10-19 | a copy of the public "Budget-Aware Coding Agent v2" notebook (budget-aware, public 0.22) | **0.15** | 9/58 vs v2's ~13/58: 4 tasks below the public copy, more than the ±2 noise. Diff vs v2 not yet checked |
| 2026-10-16 | `sub_001_nothink_nolora` (sample agent, nothink, no LoRA, 240/28/8/80) | — | ❌ Notebook Timeout (12 h). CLI shows COMPLETE with an empty score |

## Kaggle 4×L4 lab (scorer hardware, kaggle CLI, private kernels; dev33 unless noted)

Budget = scorer: 28 calls · 8 min · 240 s cmd · 300 s verify. 4×L4 burns ~2 GPU-quota h per wall h.

| date | kernel | arms | result (resolved / 33) | wall | quota h | notes |
|---|---|---|---|---|---|---|
| 2026-10-20 | g4-lab-ab2 v1 | pub · out6k · out4k | **pub 10 · out6k 8 · out4k 9** | 7.8 h | ~15 | out-caps halve overflow (11 → 5), rescued runs become wrong patches. out6k/out4k ⊂ pub. Insights 15–17 |
| 2026-10-20 | g4-lab-ab1 v1 | pub · nothink | **pub 8 · nothink 4** | 5.6 h | ~11 | nothink 8 timeouts (pub 2). McNemar p 0.22. Insights 5–14 |
| 2026-10-20 | g4-lab-probe v4 | pub, 1 task | 0/1 (26 calls, 221 s, 2,118 B patch) | ~0.5 h | ~1 | first working agent loop. vLLM load 830 s |
| 2026-10-20 | g4-lab-probe v3 | pub | 0 (all died at the 1st call) | — | ~0.5 | litellm 1.106 ↔ mcp ImportError → pin 1.85.7. 28.9 tok/s |
| 2026-10-20 | g4-lab-probe v2 | — | — | — | ~0.3 | vLLM OOM at util 0.90 → 0.85 + max-num-seqs 32 |

### Kaggle CPU kernels (free; no agent, no GPU)

| date | kernel | purpose | result |
|---|---|---|---|
| 2026-10-20 | g4-lab-diag5 | box layout simulation of `scripts/install_testdeps.sh` (pip-less venv) | 🧪✓ a requests task: 0 fixture errors; a fastapi task: tests run (no import error) |
| 2026-10-20 | g4-lab-diag4 | dev33 audit: base commit + hidden tests, after the fix | 🧪✓ 31/33 valid; 1 task passes at baseline, 1 times out at 300 s verification |
| 2026-10-20 | g4-lab-diag3 | verify the test-deps fix on 5 tasks | a requests task: 185 setup errors → 0; fastapi still needed inline_snapshot |
| 2026-10-20 | g4-lab-diag2 / `diag1` | find the cause (CPU, + internet for diag2) | `install_test_dependencies` returns early for the subprocess sandbox; no pytest-httpbin in wheels/ |
| 2026-10-20 | g4-lab-testdeps v2 | download test-only wheels (internet) | 28 wheels: pytest-httpbin 2.1.0, httpbin 0.10.4, inline-snapshot 0.36.1, dirty-equals 0.11 + deps |
| 2026-10-20 | g4-lab-drytest v3 | Linux dry-run of the box's `eval_dev.sh` arms (stubs) | all arm diffs + budget flags ✓ (v2 hung on a pipe, no GPU cost) |
| 2026-10-20 | g4-lab-wheels313 v2 | cp313 wheel set (Kaggle retired Py 3.12) | 254 wheels, litellm pinned 1.85.7 |

## Own GPU box (4× RTX PRO 6000) and Colab (1× RTX PRO 6000), dev33 unless noted. No time cap before 2026-10-20

| date | where | commit | arms | result (resolved / 33) | notes |
|---|---|---|---|---|---|
| 2026-10-21 | laptop (CPU, Opus 5.5 via API) | `wip` | ⏳ B-3g uncapped ceiling: the LB-0.24 submission unchanged, model alias → `anthropic/claude-opus-5-5` through `--models-yaml` + `scripts/ceil_sitecustomize.py`; 80 calls · 30 min · 240 s cmd · 900 s verify · CONC 3 · subprocess sandbox on macOS | ⏳ | probes: two tasks resolved in 25 s / 4–5 calls each; prompt cache reads ≈ 80 % of input |
| 2026-10-21 | box | `871b130` | `scripts/taskdiff.py` on the requests task the lanes lost (BOX_PLAN.md B-3e step 1, no GPU) | online pass ✓ · three lane passes ✗ / timeout / ✗ | Near-identical patches in all four passes (0.5–0.9 KB); the verify log carries 104 network-failure phrases in every lane and none online → the task's tests need the network, so it is unwinnable in any offline scorer and the lanes are the faithful setup. Host grep: the log names only documentation links, so the failing tests print no URL; the failure phrases and test names are the next read, k4–k6 start regardless |
| 2026-10-21 | box | `30e9c8f` | base ×3 in Kaggle-like lanes (`scripts/lane.sh`: one card TP=1 + CONC=1 per network namespace; TAG k1–k3; BOX_PLAN.md B-3d) | **k1 12 · k2 10 · k3 9** → 7 identical base passes: 11 · 12 · 11 · 9 · 12 · 10 · 9 | 🧪✓ Each lane decodes 51 tok/s with 0.1 s/call overhead (shared server: 46, 0.9), all three servers up in ~75 s, so the cards are independent and the shape matches the scorer. The cap was still TIME_MIN=4 (the smoke figure overstated the rate again): timeouts 6 → 3 · 5 · 3 but overflows 5 → 7 · 6 · 8, ~11 budget deaths per pass either way. Calibrated cap from the fit: 8 × 34 / 51 → **TIME_MIN=5** (B-3e re-runs the reference at 5). 6 tasks resolved in all 7 passes (was 7): the lanes are offline, and requests fell 1/4 → 0/4 in every lane — to be checked (`scripts/taskdiff.py`); the Kaggle scorer is offline too |
| 2026-10-21 | box | `8c8fe22` | base (public "Baseline \| LB 0.24" notebook) · tools6 · tools6+submit (BOX_PLAN.md B-2) | **base 11 · tools6 10 · tools6+submit 11** | 🧪✓ first clean baseline · 28 calls · TIME_MIN=4. Pairs vs base: tools6 +2/−3, tools6+submit +2/−2 → neither passes the decision rule, so B-3 builds on `base`. Overflow lines in the vLLM log 41 → 12 / 12, errors 12 → 10 / 8: dropping the 3 graph schemas removes most overflows but, as with ab2's output caps, those runs don't become wins. requests 1/4 in every arm. Prompt growth: tool output ~49% · generated tokens (thinking kept) ~45% |
| 2026-10-21 | box | — | `scripts/flips.py` on the 4 base passes (BOX_PLAN.md B-3c, no GPU) | 17 losing passes of the 8 flip tasks: **10 time cap** · 4 wrong patch · 2 overflow · 1 no patch | Timeouts end at 242 s; the winning passes of the same tasks run up to 246 s. So on this box TIME_MIN=4 with CONC=3 is the flipper, not the agent: the box under-measures every config (the union of 4 passes, 15/33, is the better estimate). Next: calibrate the cap from the traces' tok/s, then base ×3 |
| 2026-10-21 | box | `08106a4` | base ×2 (TAG r2, r3; BOX_PLAN.md B-3b noise floor) + `scripts/rulestats.py` on the B-2/B-3 runs | **r2 11 · r3 9** → 4 identical base passes: 11 · 12 · 11 · 9 | 🧪✓ 7 tasks solved in every pass, 15 in at least one, 8 flip, mostly budget deaths (overflow / time cap) rather than wrong patches → every B-2/B-3 arm lies inside base's own range; one pass per arm decides nothing. rulestats: the B-3 rules were followed (missing-param edits 55→10 and heredoc edits 1→35 for editrecover; first edit call 14→10 for issuefirst; /tmp writes 3→0 for tmpfix; test1 unchanged, base already tests in 8–12 runs) with no score effect. Box time cap (TIME_MIN=4, CONC=3) kills 5–8 runs per pass vs 1–2 on L4 at 8 min |
| 2026-10-21 | box | `25570e3` | base · editrecover · issuefirst · tmpfix+test1 (BOX_PLAN.md B-3, one prompt line each) | **base 12 · editrecover 9 · issuefirst 11 · tmpfix+test1 8** | 🧪✓ Pairs vs the same-pass base: editrecover +1/−4, issuefirst +1/−2, tmpfix+test1 +0/−4 → none passes the decision rule; added rules on top of the LB-0.24 prompt cost more than they fix. **The two base passes (B-2 11, B-3 12) disagree on 5 tasks**: one pass per arm can't resolve ±2. A forced test run (test1) fills context and time (overhead 14% vs ~6%) |
| 2026-10-16 | Colab | `eb705a1` | nothink (stopped at 13/33) | 2/13 | 6/13 ContextWindowExceeded; median 424 s/task |
| 2026-10-14 | box | `ec41085` | out8k · nothink · think1k | out8k 8 · **nothink 12** · think1k 9 | Finding #5 (uncapped: does not hold on L4, see ab1) |
| 2026-10-12 | box | `7c7f580` | sample (out 16,384) · out8k | sample 5 · out8k 8 | Finding #1: 16K output reservation → 400s |
| 2026-10-12 | box | `ed3feac` | sample ×2, 5 tasks | 2/5 · 2/5 | 5 tasks too noisy (was 4/5 the first time) |
| 2026-10-12 | box | `14cfacb` | sample, 5 tasks | 4/5 | first box run after pytest in the eval venv |

## RunPod pods (rented, 1 GPU per pod, network volume; `scripts/runpod_setup.sh`)

| date | GPU | commit | what | result | notes |
|---|---|---|---|---|---|
| 2026-10-21 | 1× RTX PRO 4500 Blackwell 32 GB | `e532cb2` | `runpod_setup.sh` (Py 3.12) + small-model smoke: `serve_vllm.sh` (TP 1, util 0.9, no LoRA) → `smoke.py` + one tool-calling request, E4B then E2B | ✅ **both serve** · E4B warm **152 tok/s**, E2B **254 tok/s** · tool calls parsed, thought block split | setup 30 min · server start 13 / 9 min, mostly Python imports from the network volume (import torch 23.8 s vs 1.6 s on local disk) · E4B 9.8 GiB weights, KV 188,720 tokens (28.8× at 32K) · E2B 7.3 GiB, KV 692,912 tokens (91.3×) |

## Quota (Kaggle GPU, weekly 30 h, refresh Sat 00:00 UTC)
| week ending | used | runs |
|---|---|---|
| 2026-10-24 | 28.7 h+ (≈ 30) | probes v2–v4, ab1, ab2 |
