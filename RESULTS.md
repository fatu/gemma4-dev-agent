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
| 2026-10-21 | box | `8c8fe22` | base (public "Baseline \| LB 0.24" notebook) · tools6 · tools6+submit (BOX_PLAN.md B-2) | **base 11 · tools6 10 · tools6+submit 11** | 🧪✓ first clean baseline · 28 calls · TIME_MIN=4. Pairs vs base: tools6 +2/−3, tools6+submit +2/−2 → neither passes the decision rule, so B-3 builds on `base`. Overflow lines in the vLLM log 41 → 12 / 12, errors 12 → 10 / 8: dropping the 3 graph schemas removes most overflows but, as with ab2's output caps, those runs don't become wins. requests 1/4 in every arm. Prompt growth: tool output ~49% · generated tokens (thinking kept) ~45% |
| 2026-10-21 | box | `25570e3` | base · editrecover · issuefirst · tmpfix+test1 (BOX_PLAN.md B-3, one prompt line each) | **base 12 · editrecover 9 · issuefirst 11 · tmpfix+test1 8** | 🧪✓ Pairs vs the same-pass base: editrecover +1/−4, issuefirst +1/−2, tmpfix+test1 +0/−4 → none passes the decision rule; added rules on top of the LB-0.24 prompt cost more than they fix. **The two base passes (B-2 11, B-3 12) disagree on 5 tasks**: one pass per arm can't resolve ±2. A forced test run (test1) fills context and time (overhead 14% vs ~6%) |
| 2026-10-16 | Colab | `eb705a1` | nothink (stopped at 13/33) | 2/13 | 6/13 ContextWindowExceeded; median 424 s/task |
| 2026-10-14 | box | `ec41085` | out8k · nothink · think1k | out8k 8 · **nothink 12** · think1k 9 | Finding #5 (uncapped: does not hold on L4, see ab1) |
| 2026-10-12 | box | `7c7f580` | sample (out 16,384) · out8k | sample 5 · out8k 8 | Finding #1: 16K output reservation → 400s |
| 2026-10-12 | box | `ed3feac` | sample ×2, 5 tasks | 2/5 · 2/5 | 5 tasks too noisy (was 4/5 the first time) |
| 2026-10-12 | box | `14cfacb` | sample, 5 tasks | 4/5 | first box run after pytest in the eval venv |

## Quota (Kaggle GPU, weekly 30 h, refresh Sat 00:00 UTC)
| week ending | used | runs |
|---|---|---|
| 2026-10-24 | 28.7 h+ (≈ 30) | probes v2–v4, ab1, ab2 |
