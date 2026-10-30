#!/usr/bin/env bash
# W2 step 2: baseline on the fixed 33-task dev split (splits/dev33.txt), one run per "arm", then one comparison table.
#   nohup bash scripts/serve_vllm.sh > runs/vllm.log 2>&1 &        # first, as for eval_sample.sh (all 4 cards: queue behind Bonsai)
#   nohup bash scripts/eval_dev.sh > runs/eval_dev.log 2>&1 &      # measured 38-490 s/task (avg ~230) → ~2 h/arm, 4-5 h total
#   python scripts/compare_runs.py runs/dev33_sample runs/dev33_out8k   # 📷 (eval_dev.sh prints it at the end too)
#
#   TAG=r2 nohup bash scripts/eval_dev.sh > runs/eval_dev_r2.log 2>&1 &  # a repeat → runs/dev33_<arm>_r2 (variance)
# Arms (ARMS="sample out8k" by default):
#   sample — the organiser's sample submission unchanged (max_output_tokens 16384)
#   out8k  — the same with max_output_tokens 8192. Finding #1 (2026-10-12): at 16384 the prompt only gets 32768-16384
#            tokens, and 42 calls died with "maximum context length"; 8192 leaves the prompt ~24K.
#   nothink — out8k + thinking_config.include_thoughts: false. Finding #4: every generated token, thinking included, is
#             re-sent in later prompts (growth.py b = 1.00, ~35-38% of growth). Does ADK drop thoughts from history then?
#   think1k — out8k + thinking_budget: 1024 (was 4096): keep thinking, but make each step's thinking smaller.
#   ARMS="nothink think1k" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3a.log 2>&1 &   # compares with dev33_out8k
#   readdisc — out8k + a "Context Discipline" block appended to prompts/system.md (scripts/arms/readdisc.md): ranged
#              read_file, no re-reads, short command output, brief thoughts. Finding #3: context dies by accumulation,
#              read_file ≈ 38 % and run_command ≈ 11 % of what the agent adds — the tool-output side of the growth.
#   noanalyzer — out8k with the code_analyzer AgentTool removed: the root agent explores by itself, so every read_file
#              lands in its own context. Measures what the sub-agent isolation (README §10.4) is worth.
#   ARMS="readdisc noanalyzer" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3b.log 2>&1 &
# Arms stack with "+": "nothink+readdisc" = out8k + both edits. Finding #5 (2026-10-14): nothink 12/33 vs out8k 8/33 and
# b 1.00 → 0.12, so round 2 builds on nothink; the plain nothink repeat (TAG=r2) checks that 12/33 is not noise:
#   TAG=r2 ARMS="nothink nothink+readdisc nothink+noanalyzer" nohup bash scripts/eval_dev.sh > runs/eval_dev_w3b.log 2>&1 &
# Each variant is a copy of the sample under runs/subs/<arm>/ (gitignored: it holds the sample's adapters).
# vLLM's log (VLLM_LOG, default runs/vllm.log) is counted before/after every arm → <run>/ctx_overflow.txt. It counts
# rejected CALLS, not tasks: LiteLLM retries each failed call up to 5x, so 42 lines were only a few tasks. Compare it
# between arms; don't read it as a task count. If the log is missing it writes "?" instead of a fake 0.
# Budgets: the local CLI ignores eval_config.yaml (seen: 32 calls, 490 s under a 10-call / 1-min config); only flags
# count. Since 2026-10-20 every arm runs under the scorer's per-task budget, passed as flags, with the time scaled
# to this box's speed:
#   MAX_CALLS=28 TIME_MIN=4 CMD_TIMEOUT=240 VERIFY_TIMEOUT=300  (the best public eval_config + README's fixed 300 s)
#   BUDGET=none reproduces the W2/W3 runs (no cap). Those results (Finding #5 included) were unbudgeted: on Kaggle's
#   4×L4 at the scorer budget, nothink scored 4/33 vs 8/33 for the LB-0.24 config (it ran out of the 8 minutes).
#   TIME_MIN defaults to 4, not the scorer's 8. Measured by the trace fit (rulestats.py): one lane TP=1 = 51 tok/s, a shared
#   TP=4 server at CONC=3 = 46 per stream, L4 = 34; the 62 figure was an idle smoke. 8 × 34 / 51 → TIME_MIN=5 for lanes.
#   the steps; 8 x 29/62 ≈ 3.7, rounded up to 4 because the CLI flag's type is unchecked (a bit looser for
#   generation). Unvalidated: test runs and tool calls take the same wall time on both machines but are squeezed
#   into half the minutes here, so it is not an exact L4 match. TIME_MIN=8 = the literal scorer number (e.g. on Kaggle
#   L4 hardware). Token effects (overflow, schema size) need no scaling.
# One run per arm at temp 0.2 is noisy: repeat with TAG.
#
# Public-top base (W4): SUBMISSION=<dir> from a notebook that embeds its files (scripts/from_notebook.py):
#   kaggle kernels pull honghanhhh/gemma-4-baseline-lb-0-24 -p /tmp/lb024
#   python3 scripts/from_notebook.py /tmp/lb024/*.ipynb runs/base_lb024
#   SUBMISSION=runs/base_lb024 ARMS="base tools6 tools6+submit" nohup bash scripts/eval_dev.sh > runs/eval_dev_w4.log 2>&1 &
#   base   — SUBMISSION unchanged (like sample, for any base)
#   out6k / out4k — max_output_tokens 6144 / 4096 (+ thinking_budget 2048). ab1 on 4×L4: every overflow was
#            prompt + 8192 > 32768, while per-call output p99 was ~3.5K and the max ~4.4K tokens.
#   tools6 — drop get_code_neighbors / search_similar_code / get_code_subgraph from agent.yaml (the LB-0.24 prompt
#            already forbids them; their schemas are sent on every call)
#   submit — prompt rule from scripts/arms/submit.md: submit as soon as the fix is checked (ab1: 8/24 failed runs had
#            edits but a 0-byte patch). It replaces the base's "edits count without submit_patch" line if present.
# Rule arms (B-3, BOX_PLAN.md): each inserts or replaces one line of prompts/system.md via scripts/arms/apply.py
# (scripts/arms/<arm>.md names the anchor line; prints "appended (anchor not found)" on a different base prompt).
# Evidence from the ab1/ab2 traces on 4×L4 (aggregates in RESULTS.md):
#   editrecover — a malformed edit_file call (missing old_string: 377 of 377 failed edits) → switch to a heredoc
#            Python replace at once instead of retrying (one run retried 7x and overflowed)
#   issuefirst — the issue names the place or shows the fix → open it and edit first (wins: first edit at call 6
#            after 2 reads; losses: call 15 after 6)
#   tmpfix — the base's "scratch files go in /tmp" contradicts write_file, which rejects /tmp paths
#   test1  — one targeted test before submit_patch is required (only 6 of 33 runs ran a test)
#   t10    — temperature 1.0 + top_k 64 (Gemma's recommended sampling; forum try-list #1: 0.2 → 1.0 raised two
#            independent runs, #747696 / #746250). ADK passes top_k through LiteLLM (google/adk/models/lite_llm.py).
#   repro  — "## 4. Verify" becomes: first a `python -c` run of the issue's own example, compared with what the issue
#            asks for; then one test file (E4B wave 1: failed runs said "the tests passed, so the fix is verified" after
#            existing tests that don't cover the issue; runs end at ~10 of 28 calls, so the extra checks fit).
#   lean   — prompts/system.md replaced by scripts/arms/lean_system.md (~2.5K chars vs 7K): the same hard limits, five
#            steps, the repro check built in, edit_file with 2-5 whole lines (E4B wave 1: 32 of 70 edits mismatched).
#   t00    — temperature 0.0 (an Oct-23 course lecture 3 slide's sampling.yaml; its later config-upgrade slide = the latest
#            setting says 0.2; forum #746250 had 0.0: 47 vs 0.2-1.0: 48-51)
#   time7  — eval_config max_time_minutes 8 → 7 in the submission (the base prompt's '28 tool calls and 8 minutes'
#            line too), and the run's --max-time-minutes scaled the same way:
#            TIME_MIN × 7/8 (the box's TIME_MIN is already the L4-equivalent of 8, see Budgets above). Printed per arm.
#   tail40 — the base prompt's targeted pytest command piped through `2>&1 | tail -40`, with `set -o pipefail;` so the
#            exit code stays pytest's (run_command is bash -c; the harness keeps the FIRST 5000 chars of stdout)
#   Course lecture 3 config = t00+out6k+time7 (thinking 4096 / include_thoughts true / 28 calls / 240 s are base's);
#   the whole lecture-3 roadmap = t00+out6k+time7+tail40. (honghanhhh's v5 shipped 7 min + 6144 + tail -40 at temp 0.2
#   and scored 0.13 by its author's note; v6 reverted it — notes Insight 31.)
#   outline — after locating a large file (> ~300 lines): one `python3 -c` ast skeleton (classes, functions incl. async, line
#            ranges; brace-free so ADK doesn't template it) before ranged reads (course: funnel L2 / AST slide; base never does it)
#   win100 — read windows "about 40 lines" → "60-100 lines (never more than 120)" (course reading-discipline slide)
#   callchain — after the twins rule: follow a new/changed parameter, option, attribute or return value up to its callers
#            and down to every callee/table that must accept or list it (notes Insight 36: Gemma 0/9 on multi-file gold
#            patches, 8 edited only the first layer; the misses are call-chain plumbing + registration, the course's
#            "vertical caller/callee check" stage). Also inside `course`; alone it is the single-rule ablation.
#   grepbatch — the course's 3-call locate SOP: batched `git grep -n -E 'A|B|C' -- '*.py' | head -40` + find tests, then a
#            def/class outline, then ranged reads of target + twins + test asserts
#   twins5 — the course's 5 twin patterns as a checklist (sync/async, container methods, function vs class entry,
#            serialize/deserialize round trip, __all__)
#   dryhelper — put a shared fix in one internal helper all twin entry points call (course 4-step, step 3)
#   noregress — unmentioned inputs keep their behaviour; __init__/__all__ export every public name (course step 4)
#   ask3   — the course's 3 caller/callee questions before editing: who calls me (git grep 'f('; defaults, compatible
#            signature), what do I call (helpers, dict key format), did unmentioned inputs change (python -c before/after
#            for None, '', int, unknown key) + 'Keep current behaviour for inputs the issue does not mention'
#   editctx — edit_file old_string with 2-3 real lines of context, only right after reading that range (course 3-step
#            edit recovery, stage 1)
#   editguard — the guarded fallback script: assert count == 1, compile(updated, path, 'exec') before writing, then
#            git diff; never rewrite a whole file or touch pytest.ini/conftest.py (course fallback-weapon slide)
#   tmprepro — repro scripts always in /tmp/repro.py, written via run_command heredoc (write_file rejects /tmp; notes
#            Insight 13) — course upgrade-4 slide
#   coursefull — the course's "final deliverable" slide: prompts/system.md replaced by its ~20-line "LB 0.24 v5 Safe" prompt
#            verbatim (scripts/arms/course_system.md) on the latest setting (LATEST=0: + t00+out6k+time7, the pre-10-25
#            build). `course` = base prompt + every slide rule; `coursefull` = the course's own prompt.
#   course — every executable item from the course slides the user sends, as one part (= $COURSE below; grows as slides
#            arrive): tail40 outline win100 grepbatch twins5 callchain dryhelper noregress ask3 editctx editguard tmprepro
#            on the latest setting (LATEST=0: t00 out6k time7 first). Graph-tool ban, edit recovery, /tmp, git status are
#            base's already. Printed as "course = …" per arm.
# LATEST=1 (default since 2026-10-25, user: "maintain one latest setting, use this one"): every arm copy, base and sample
# included, gets scripts/arms/latest/configs/sampling.yaml + eval_config.yaml in place of the base's before any arm part,
# and every arm but sample gets scripts/arms/latest/prompts/system.md (since 2026-10-26: the base prompt + the course
# lecture-3 notebook's 3 edits — git status --short before git diff, keep behaviour the issue does not mention, | tail -40)
# (course lecture 3's config-upgrade slide: temp 0.2 · top_p 0.95 · 6144 · thinking 4096 · 240 s · 28 calls · 7 min ·
# 80 turns). Arms are deltas on top of it. The run flags take calls / timeout from it and minutes as
# TIME_MIN × max_time_minutes / 8; the prompt's "N tool calls and M minutes in total" line follows it. Printed as
# "latest setting: …" at the top and "latest: …" per arm. LATEST=0 = the old behaviour (the base's own config, W3+ arms
# on out8k, course/coursefull with t00+out6k+time7) for reruns of pre-2026-10-25 rows.
#   edit1  — on top of lean: old_string = ONE whole line copied from the latest read_file output, new_string = that
#            line's full replacement (E4B waves 1-3: ~75% of the mismatched old_strings had lines never shown by a read).
# BUILD_ONLY=1: build runs/subs/<arm>[_TAG] for each arm and stop (no server, no test deps, no eval) — the source dir
# for scripts/make_submission.py, so the submission is built by the same code as the measured arm (works on macOS too).
# MODEL_ALIAS=gemma-4-e4b-it-qat-w4a16-ct: every agent YAML of each arm copy gets this `model:` (small-model bonus on
# the same files; the server must serve the alias: MODEL=google/gemma-4-E4B-it-qat-w4a16-ct scripts/serve_vllm.sh).
#   MODEL_ALIAS=gemma-4-e4b-it-qat-w4a16-ct TIME_MIN=8 SUBMISSION=runs/base_lb024 ARMS=base TAG=e4b bash scripts/eval_dev.sh
# B-3 re-runs base in the same pass (runs/dev33_base exists, so TAG keeps the dirs apart):
#   TAG=w5 SUBMISSION=runs/base_lb024 ARMS="base editrecover issuefirst tmpfix+test1" \
#     nohup bash scripts/eval_dev.sh > runs/eval_dev_w5.log 2>&1 &
# B-3 result: none passed, and the two base passes (B-2 11, B-3 12) differed on 5 tasks. So before more arms: the
# end of every run now prints scripts/rulestats.py (per-arm rule-following: edit failures, first edit, tests,
# /tmp writes), and base is repeated to measure the noise (REFS lists the earlier base runs for the compare table):
#   nohup bash -c 'TAG=r2 SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh;
#     TAG=r3 REFS="base base_w5 base_r2" SUBMISSION=runs/base_lb024 ARMS=base bash scripts/eval_dev.sh' \
#     > runs/eval_dev_rep.log 2>&1 &
# B-3c showed 10 of the 17 flip losses were the 4-min cap under CONC=3 (per-stream decode is slower when 3 tasks
# share the server). The scorer runs one task at a time, so the Kaggle-like box config is CONC=1 with TIME_MIN scaled
# by the box's *measured* single-stream rate. Three such lanes at once (one card + one private 127.0.0.1:8000 each):
# scripts/lane.sh — it calls this script with CONC=1 inside a network namespace (BOX_PLAN.md B-3d).
set -euo pipefail
cd "$(dirname "$0")/.."
if sed --version >/dev/null 2>&1; then sedi() { sed -i "$@"; }; else sedi() { sed -i '' "$@"; }; fi   # GNU (pod, box) · BSD (macOS)
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE="${SUBMISSION:-harness/sample_submission}"
ARMS="${ARMS:-sample out8k}"
if [ "${BUDGET:-scorer}" = none ]; then BUDGET_FLAGS=(); else
  BUDGET_FLAGS=(--max-tool-calls "${MAX_CALLS:-28}" --max-time-minutes "${TIME_MIN:-4}"
                --timeout-seconds "${CMD_TIMEOUT:-240}" --verification-timeout-seconds "${VERIFY_TIMEOUT:-300}"); fi
# Two latest settings since 2026-10-30 (user: "我们维护两个最新配置，31B和小模型LoRA"), chosen with LATEST:
#   LATEST=31b (default; 1 = the same) → scripts/arms/latest      the 31B entry, no adapter
#   LATEST=e4b                         → scripts/arms/latest_e4b  the small-model entry: MODEL_ALIAS + ADAPTER from its
#     model.env (E4B + the PLE-fix LoRA); the adapter files come from runs/adapters/<ADAPTER>/ (gitignored, fetched once
#     from the public Kaggle dataset in model.env) into every arm copy, with `adapter: <ADAPTER>` on the root agent.
#     Serve it with: MODEL=google/gemma-4-E4B-it-qat-w4a16-ct SUBMISSION=runs bash scripts/serve_vllm.sh
#   LATEST=0                           → off (the base's own config)
LSET="${LATEST:-31b}"
case "$LSET" in 1|31b) LSET=31b; LATEST=1; LDIR=scripts/arms/latest ;; e4b) LATEST=1; LDIR=scripts/arms/latest_e4b ;;
  0) LDIR=scripts/arms/latest ;; *) echo "LATEST=$LSET: use 31b (default), e4b or 0"; exit 1 ;; esac
L_ADAPTER=""
if [ "$LSET" = e4b ]; then
  menv() { sed -n "s/^$1=//p" "$LDIR/model.env"; }
  MODEL_ALIAS="${MODEL_ALIAS:-$(menv MODEL_ALIAS)}"; L_ADAPTER="${ADAPTER-$(menv ADAPTER)}"; ADIR="runs/adapters/$L_ADAPTER"
  if [ -n "$L_ADAPTER" ] && [ ! -f "$ADIR/adapter_model.safetensors" ]; then
    mkdir -p "$ADIR"; for f in adapter_config.json adapter_model.safetensors; do
      kaggle datasets download "$(menv ADAPTER_DATASET)" -f "$f" -p "$ADIR" >/dev/null 2>&1 || true; done
    [ -f "$ADIR/adapter_model.safetensors" ] || { echo "LATEST=e4b: no $ADIR/adapter_model.safetensors and the kaggle download failed — fetch $(menv ADAPTER_DATASET) into $ADIR (or ADAPTER= for none)"; exit 1; }
  fi
  if [ -n "$L_ADAPTER" ]; then
    A_SHA=$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$ADIR/adapter_model.safetensors")
    [ "$A_SHA" = "$(menv ADAPTER_SHA256)" ] || { echo "LATEST=e4b: $ADIR/adapter_model.safetensors sha256 $A_SHA ≠ model.env's $(menv ADAPTER_SHA256)"; exit 1; }
  fi
fi
echo "latest set: $LSET ($LDIR)$([ "$LSET" = e4b ] && echo " · model $MODEL_ALIAS · adapter ${L_ADAPTER:-none (ADAPTER=)}${L_ADAPTER:+ ← $ADIR, sha256 ${A_SHA:0:16} ✓}")"
if [ "$LATEST" = 1 ]; then
  lval() { grep -oE "^ *$1: *[0-9.]+" "$LDIR/eval_config.yaml" | sed 's/.*: *//'; }
  L_MIN=$(lval max_time_minutes); L_CALLS=$(lval max_tool_calls); L_TO=$(lval timeout_seconds); L_FLAG_MIN=""
  if [ ${#BUDGET_FLAGS[@]} -gt 0 ]; then
    L_FLAG_MIN=$(python3 -c "print(round(${TIME_MIN:-4} * $L_MIN / 8, 3))")
    BUDGET_FLAGS=(--max-tool-calls "${MAX_CALLS:-$L_CALLS}" --max-time-minutes "$L_FLAG_MIN"
                  --timeout-seconds "${CMD_TIMEOUT:-$L_TO}" --verification-timeout-seconds "${VERIFY_TIMEOUT:-300}")
  fi
  echo "latest setting: $LDIR → $(grep -hE '^(temperature|top_p|top_k):|max_output_tokens|thinking_budget|include_thoughts' "$LDIR/configs/sampling.yaml" | tr -s ' \n' ' ')· eval_config $L_TO s / $L_CALLS calls / $L_MIN min · run flag --max-time-minutes ${L_FLAG_MIN:-none (BUDGET=none)} (= TIME_MIN ${TIME_MIN:-4} × $L_MIN/8)"
else
  echo "latest setting: OFF (LATEST=0) — the base's own sampling/eval_config, W3+ arms on out8k (pre-2026-10-25 behaviour)"
fi
LOG="${VLLM_LOG:-runs/vllm.log}"
# MODELS_YAML=<file>: no local server. The harness's --models-yaml re-registers the submission's alias with a litellm
# `path` (e.g. anthropic/claude-opus-5-5) and `api_base`; the key comes from LITELLM_API_KEY (BOX_PLAN.md B-3f/B-3g
# ceiling runs). Nothing in swegemma caps the context: the 32K limit is the vLLM server's max_model_len, so since
# 2026-10-28 the request adapter emulates it (CEIL_GEMMA_CTX below) — printed here so the log says what was measured.
# Every eval loads scripts/eval_sitecustomize.py (PYTHONPATH sitecustomize): the sandbox shim (B-3i — the subprocess sandbox
# otherwise never imports a src-layout repo such as requests, so its patches were never tested), the scorer's events compaction
# (scripts/compaction_shim.py — swegemma's CLI sets none; COMPACTION=0 = the old no-compaction runs) and, with MODELS_YAML, the
# remote-model request adapter. All print their effective settings into this log.
mkdir -p runs/pypath && ln -sf "$PWD/scripts/eval_sitecustomize.py" runs/pypath/sitecustomize.py
export PYTHONPATH="$PWD/runs/pypath${PYTHONPATH:+:$PYTHONPATH}"
echo "trajectory log: $([ "${TRAJ:-1}" = 1 ] && echo "on (TRAJ=1) → <run>/traj/<task>.jsonl: every model call's new messages, thinking, reply, tokens · render: scripts/walkthrough.py" || echo "off (TRAJ=0)")"
echo "sandbox shim: scripts/sandbox_shim.py $([ "${SANDBOX_SHIM:-1}" = 0 ] && echo OFF || echo "on — per-task venv gets 00_workspace.pth → workspace + workspace/src first on sys.path (subprocess sandbox only)")"
echo "compaction: $([ "${COMPACTION:-1}" = 0 ] && echo "OFF (COMPACTION=0) — history only grows, unlike the scorer" || echo "on (scripts/compaction_shim.py) — the scorer's EventsCompactionConfig: interval ${COMPACT_INTERVAL:-15} · overlap ${COMPACT_OVERLAP:-2} · threshold ${COMPACT_THRESHOLD:-14336} · retention ${COMPACT_RETAIN:-5}; before 2026-10-27 box runs had none")"
if [ "${LENIENT:-0}" = 1 ]; then                      # B-3i-L: era pins for the drift tasks — analysis only, never the reference
  case "${TAG:-}" in *len*) ;; *) echo "LENIENT=1 needs a TAG containing 'len' (e.g. TAG=lenA) so the run dir says so"; exit 1;; esac
  echo "sandbox mode: LENIENT — analysis only, not the reference (pins per task from scripts/sandbox_shim.py PINS${LENIENT_PINS:+ + LENIENT_PINS=$LENIENT_PINS}); runs get lenient.txt"
else
  echo "sandbox mode: strict (the scorer's: highest wheel of every package)"
fi
if [ -n "${MODELS_YAML:-}" ]; then
  [ -f "$MODELS_YAML" ] || { echo "MODELS_YAML=$MODELS_YAML not found"; exit 1; }
  [ -n "${LITELLM_API_KEY:-}" ] || { echo "MODELS_YAML set but LITELLM_API_KEY is empty"; exit 1; }
  echo "models: --models-yaml $MODELS_YAML → $(grep -E '^\s*path:' "$MODELS_YAML" | sed 's/.*path: *//' | tr '\n' ' ')(remote; the submission alias is overridden)"
  # CEIL_GEMMA_CTX (default 32768, 0 = off): the vLLM server's window, emulated in scripts/ceil_sitecustomize.py with
  # Gemma 4's tokenizer + chat template (runs/gemma_tok, fetched once from the ungated E4B repo); the reply's prompt
  # tokens are relabelled to that count so the compaction threshold fires as for Gemma. Before 2026-10-28: no cap.
  export CEIL_GEMMA_CTX="${CEIL_GEMMA_CTX:-32768}" CEIL_GEMMA_TOK="${CEIL_GEMMA_TOK:-$PWD/runs/gemma_tok}"
  if [ "$CEIL_GEMMA_CTX" != 0 ] && [ ! -f "$CEIL_GEMMA_TOK/chat_template.jinja" ]; then
    .venv-eval/bin/python -c "
from huggingface_hub import hf_hub_download
for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    hf_hub_download('google/gemma-4-E4B-it-qat-w4a16-ct', f, local_dir='$CEIL_GEMMA_TOK')" >/dev/null 2>&1 \
      || { echo "CEIL_GEMMA_CTX: could not fetch Gemma's tokenizer into $CEIL_GEMMA_TOK (CEIL_GEMMA_CTX=0 to run without the cap)"; exit 1; }
  fi
  if [ "$CEIL_GEMMA_CTX" != 0 ]; then
    echo "context cap: Gemma's $CEIL_GEMMA_CTX emulated (CEIL_GEMMA_CTX) — requests counted with Gemma 4's tokenizer + chat template ($CEIL_GEMMA_TOK), prompt + max_output_tokens > $CEIL_GEMMA_CTX → vLLM's ContextWindowExceededError; reply prompt_tokens relabelled to the Gemma count (compaction threshold in Gemma tokens)"
  else
    echo "context cap: OFF (CEIL_GEMMA_CTX=0) — the remote model's own window, its own token counts for compaction · max_output_tokens from the submission"
  fi
  MODEL_FLAGS=(--models-yaml "$MODELS_YAML")
  export CEIL_CALL_LOG="${CEIL_CALL_LOG:-$PWD/runs/ceil/calls${TAG:+_$TAG}.jsonl}"; mkdir -p runs/ceil
  echo "request adapter: scripts/ceil_sitecustomize.py (extra_body dropped; anthropic thinking $([ "${CEIL_NOTHINK:-0}" = 1 ] && echo off || echo adaptive), temperature left to the provider) · call log $CEIL_CALL_LOG"
elif [ "${BUILD_ONLY:-0}" = 1 ]; then
  MODEL_FLAGS=(); echo "BUILD_ONLY=1: arm dirs only (runs/subs/<arm>${TAG:+_$TAG}) — no server check, no eval"
else
  MODEL_FLAGS=()
curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null || { echo "no vLLM on :8000 — start scripts/serve_vllm.sh first"; exit 1; }
# the server must serve the alias AND every adapter of the submission, else each task dies in ~2 s with NotFoundError
# and the run looks like a 0/33 result instead of a failure (seen in W1)
SUB="$BASE" ADAPTER="$L_ADAPTER" python3 - <<'PY'
import json, os, pathlib, re, sys, urllib.request
ids = [m["id"] for m in json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"]]
# every agent YAML's `model:` (one base model per submission: 31B, or E4B/E2B for the small-model bonus)
models = sorted({m for p in pathlib.Path(os.environ["SUB"]).rglob("*.y*ml") if "adapters" not in p.parts
                 for m in re.findall(r"^\s*model:\s*['\"]?([\w.-]+)", p.read_text(), re.M)}) or ["gemma-4-31b-it-qat-w4a16-ct"]
if os.environ.get("MODEL_ALIAS"):
    print(f"submission models: {', '.join(models)} → MODEL_ALIAS {os.environ['MODEL_ALIAS']} in every arm copy")
    models = [os.environ["MODEL_ALIAS"]]
else:
    print("submission models:", ", ".join(models))
need = models + sorted(p.parent.name for p in pathlib.Path(os.environ["SUB"], "adapters").glob("*/adapter_config.json"))
need += [os.environ["ADAPTER"]] if os.environ.get("ADAPTER") else []   # LATEST=e4b adds it to every arm copy
missing = [m for m in need if m not in ids]
print("served:", ", ".join(ids))
if missing:
    sys.exit(f"the server lacks {missing} — restart it with SUBMISSION=$BASE scripts/serve_vllm.sh")
PY
fi
TDPY=python3; [ -x .venv-eval/bin/python ] && TDPY=.venv-eval/bin/python   # the venv the sandboxes see (laptop python3 ≠ it)
[ "${BUILD_ONLY:-0}" = 1 ] || "$TDPY" -c 'import pytest_httpbin, inline_snapshot' 2>/dev/null || {   # without them requests/fastapi tests can't run
  echo "test deps missing (pytest-httpbin / inline-snapshot): run scripts/install_testdeps.sh first, or set NO_TESTDEPS=1"
  [ -n "${NO_TESTDEPS:-}" ] || exit 1; }
[ -f "$LOG" ] || echo "warning: no $LOG — ctx_overflow will be '?' (set VLLM_LOG=… to the server's log)"
SPLIT="${SPLIT:-splits/dev33.txt}"                    # SPLIT=<file>: a task subset (one id per line), e.g. pass@k on the misses
IDS=($(cat "$SPLIT"))                                 # no mapfile: macOS bash 3.2
echo "tasks: ${#IDS[@]} from $SPLIT"
echo "dev split: ${#IDS[@]} tasks · arms: $ARMS · budget: ${BUDGET_FLAGS[*]:-none} · concurrency ${CONC:-1}"
for arm in $ARMS; do                                  # check every arm up front, not 2 h into the first one
  for part in ${arm//+/ }; do
    case "$part" in out8k|nothink|think1k|readdisc|noanalyzer|out6k|out4k|tools6|submit|editrecover|issuefirst|tmpfix|test1|t10|repro|lean|edit1|r1|r2|r3|v4|t00|time7|tail40|outline|win100|course|callchain|grepbatch|twins5|dryhelper|noregress|ask3|editctx|editguard|tmprepro|coursefull) ;;
      sample|base) [ "$arm" = "$part" ] || { echo "$part can't stack: $arm"; exit 1; } ;;
      *) echo "unknown arm part $part in $arm"; exit 1 ;; esac
  done
  RUN="runs/dev33_$arm${TAG:+_$TAG}"
  [ "${BUILD_ONLY:-0}" != 1 ] && [ -e "$RUN" ] && { echo "$RUN exists (task_results.jsonl is append-only) — repeat with TAG=r2"; exit 1; }
done

overflows() { [ -f "$LOG" ] || { echo "?"; return; }; n=$(grep -c "maximum context length" "$LOG") || true; echo "${n:-0}"; }

[ "$LATEST" = 1 ] && CCFG="" || CCFG="t00 out6k time7 "   # the slides' config: the latest setting, or the pre-10-25 parts
COURSE="${CCFG}tail40 outline win100 grepbatch twins5 callchain dryhelper noregress ask3 editctx editguard tmprepro"   # the course slides' test case (part "course")
COURSEFULL="${CCFG% }"                                # part "coursefull": the slides' final prompt (+ the config parts if LATEST=0)
has() { [[ "+$arm+" == *"+$1+"* ]] || { [[ "+$arm+" == *"+course+"* ]] && [[ " $COURSE " == *" $1 "* ]]; } \
        || { [[ "+$arm+" == *"+coursefull+"* ]] && [[ " $COURSEFULL " == *" $1 "* ]]; }; }   # "nothink+readdisc" has both

for arm in $ARMS; do
  SUB="runs/subs/$arm${TAG:+_$TAG}"; RUN="runs/dev33_$arm${TAG:+_$TAG}"   # per TAG: parallel lanes must not share it
  rm -rf "$SUB"; mkdir -p runs/subs; cp -r "$BASE" "$SUB"
  Y="$SUB/configs/sampling.yaml"
  [ "$LATEST" = 1 ] && { cp "$LDIR/configs/sampling.yaml" "$Y"; cp "$LDIR/eval_config.yaml" "$SUB/eval_config.yaml"; }
  if [ "$LATEST" = 1 ] && [ "$arm" != sample ]; then   # the latest prompt too (2026-10-26); sample keeps the organiser's
    cp "$LDIR/prompts/system.md" "$SUB/prompts/system.md"
    echo "latest prompt: $LDIR/prompts/system.md ($(wc -c < "$LDIR/prompts/system.md" | tr -d ' ') chars) in place of $BASE's — arm parts apply on top"
  fi
  has course && echo "course = ${COURSE// /+} (course slides test case)$([ "$LATEST" = 1 ] && echo ' on the latest setting')"
  has coursefull && echo "coursefull = scripts/arms/course_system.md (the slides' final ~20-line prompt, verbatim) + ${COURSEFULL:-the latest setting}"
  [ "$LATEST" != 1 ] && [ "$arm" != sample ] && [ "$arm" != base ] && sedi 's/^max_output_tokens: .*/max_output_tokens: 8192/' "$Y"   # W3+ arms built on out8k before LATEST
  has out6k && sedi 's/^max_output_tokens: .*/max_output_tokens: 6144/' "$Y"
  if has out4k; then sedi 's/^max_output_tokens: .*/max_output_tokens: 4096/' "$Y"
    sedi 's/^\( *\)thinking_budget: .*/\1thinking_budget: 2048/' "$Y"; fi
  has nothink && sedi 's/^\( *\)include_thoughts: .*/\1include_thoughts: false/' "$Y"
  has think1k && sedi 's/^\( *\)thinking_budget: .*/\1thinking_budget: 1024/' "$Y"
  if has t10; then sedi 's/^temperature: .*/temperature: 1.0/' "$Y"
    if grep -q '^top_k:' "$Y"; then sedi 's/^top_k: .*/top_k: 64/' "$Y"; else sedi '/^top_p:/a\
top_k: 64' "$Y"; fi; fi
  has t00 && sedi 's/^temperature: .*/temperature: 0.0/' "$Y"
  T7=""; ARM_BUDGET=(${BUDGET_FLAGS[@]+"${BUDGET_FLAGS[@]}"})
  if has time7; then
    E="$SUB/eval_config.yaml"; [ -f "$E" ] && sedi 's/^\( *\)max_time_minutes: .*/\1max_time_minutes: 7/' "$E"
    if [ ${#ARM_BUDGET[@]} -gt 0 ]; then
      T7=$(python3 -c "print(round(${TIME_MIN:-4} * 7 / 8, 3))")
      ARM_BUDGET=(--max-tool-calls "${MAX_CALLS:-28}" --max-time-minutes "$T7"
                  --timeout-seconds "${CMD_TIMEOUT:-240}" --verification-timeout-seconds "${VERIFY_TIMEOUT:-300}")
    fi
    P7="$SUB/prompts/system.md"; if has coursefull; then echo "time7: prompt = the course's own (already says 7 minutes)"; else sedi 's/28 tool calls and 8 minutes in total/28 tool calls and 7 minutes in total/' "$P7"
    echo "time7: prompt budget line -> $(grep -c '28 tool calls and 7 minutes' "$P7") x '28 tool calls and 7 minutes'"; fi
    echo "time7: submission eval_config max_time_minutes $(grep -hoE 'max_time_minutes: *[0-9.]+' "$E" 2>/dev/null | sed 's/.*: *//') · run flag --max-time-minutes ${T7:-none (BUDGET=none)} (= TIME_MIN ${TIME_MIN:-4} × 7/8)"
  fi
  if [ -n "${MODEL_ALIAS:-}" ]; then           # every agent YAML (sub-agents too), not adapter configs
    find "$SUB" -name '*.y*ml' -not -path '*/adapters/*' | while IFS= read -r f; do
      sedi -E "s/^( *model: *)['\"]?[A-Za-z0-9_.-]+['\"]?/\1$MODEL_ALIAS/" "$f"; done
  fi
  if [ -n "$L_ADAPTER" ]; then                 # LATEST=e4b: the adapter files in the submission dir + the root agent's line
    mkdir -p "$SUB/adapters"; rm -rf "$SUB/adapters/$L_ADAPTER"; cp -r "$ADIR" "$SUB/adapters/$L_ADAPTER"
    awk -v a="$L_ADAPTER" '!/^adapter:/ {print} /^model:/ {print "adapter: " a}' "$SUB/agent.yaml" > "$SUB/agent.yaml.new"
    mv "$SUB/agent.yaml.new" "$SUB/agent.yaml"
    echo "latest e4b: adapter $L_ADAPTER → $SUB/adapters/$L_ADAPTER · agent.yaml: $(grep -E '^(model|adapter):' "$SUB/agent.yaml" | tr '\n' ' ')"
  fi
  has coursefull && cp scripts/arms/course_system.md "$SUB/prompts/system.md"   # replaces the whole prompt
  has lean && cp scripts/arms/lean_system.md "$SUB/prompts/system.md"   # first: rule arms then append to it
  if [ "$LATEST" = 1 ]; then                   # after the prompt swaps: the budget line says the latest calls / minutes
    P="$SUB/prompts/system.md"; sedi "s/[0-9][0-9]* tool calls and [0-9][0-9]* minutes in total/$L_CALLS tool calls and $L_MIN minutes in total/" "$P"
    echo "latest: $Y + eval_config.yaml from $LDIR · prompt budget line -> $(grep -c "$L_CALLS tool calls and $L_MIN minutes in total" "$P") x '$L_CALLS tool calls and $L_MIN minutes' (0 = the prompt words it differently)"
  fi
  has readdisc && cat scripts/arms/readdisc.md >> "$SUB/prompts/system.md"
  if has noanalyzer; then                      # drop the 3-line agent_tool entry (code_analyzer) from the tool list
    sedi '/^ *- agent_tool:/,/skip_summarization/d' "$SUB/agent.yaml"
    grep -q "agent_tool" "$SUB/agent.yaml" && { echo "noanalyzer: agent_tool still in agent.yaml"; exit 1; }
  fi
  if has tools6; then
    sedi -E '/^ *- (get_code_neighbors|search_similar_code|get_code_subgraph) *$/d' "$SUB/agent.yaml"
    grep -qE 'get_code_neighbors|search_similar_code|get_code_subgraph' "$SUB/agent.yaml" && { echo "tools6: graph tool left in agent.yaml"; exit 1; }
  fi
  if has submit; then                          # replace the base's "edits count without submit" sentence, else append
    P="$SUB/prompts/system.md" R="scripts/arms/submit.md" python3 - <<'PY'
import os, re
p, rule = os.environ["P"], open(os.environ["R"]).read().strip()
s = open(p).read()
s2, n = re.subn(r"An edit that exists counts[^.]*\.", rule, s, count=1)
open(p, "w").write(s2 if n else s.rstrip() + "\n\n# SUBMIT\n- " + rule + "\n")
assert "{" not in rule and "}" not in rule        # ADK templates {name} from session state
print("submit:", "replaced the base's line" if n else "appended")
PY
  fi
  for rule in editrecover issuefirst tmpfix test1 repro edit1 r1 r2 r3 v4 tail40 outline win100 grepbatch twins5 callchain dryhelper noregress ask3 editctx editguard tmprepro; do   # one-line prompt rules (scripts/arms/<rule>.md)
    has "$rule" && python3 scripts/arms/apply.py "$rule" "$SUB/prompts/system.md"
  done
  echo; echo "== $arm: model $(grep -rhoE '^ *model: *[A-Za-z0-9_.-]+' --include='*.y*ml' --exclude-dir=adapters "$SUB" | sed 's/.*model: *//' | sort -u | tr '\n' ' ')· $(grep -hE '^(temperature|top_p|top_k):|max_output_tokens|thinking_budget|include_thoughts' "$SUB/configs/sampling.yaml" | tr -s ' \n' ' ')· tools $(grep -c '^  - ' "$SUB/agent.yaml") · prompt $(wc -c < "$SUB/prompts/system.md") chars → $RUN   ($(date +%H:%M))"
  [ "${BUILD_ONLY:-0}" = 1 ] && { echo "   built $SUB (BUILD_ONLY: no eval)"; continue; }
  before=$(overflows)
  TRAJ_DIR=$([ "${TRAJ:-1}" = 1 ] && echo "$PWD/$RUN/traj") swegemma eval \
    --tasks harness/tasks.jsonl \
    --snapshots-dir harness/snapshots \
    --submission-dir "$SUB" \
    --results-dir "$RUN" \
    --sandbox "${SANDBOX:-subprocess}" \
    --task-ids "${IDS[@]}" \
    --concurrency "${CONC:-1}" \
    --display quiet ${ARM_BUDGET[@]+"${ARM_BUDGET[@]}"} ${MODEL_FLAGS[@]+"${MODEL_FLAGS[@]}"} || echo "swegemma eval exited $? for $arm (summary below may be partial)"
  after=$(overflows); mkdir -p "$RUN"                   # the eval may die before creating $RUN
  if [ "$before" = "?" ] || [ "$after" = "?" ]; then echo "?"; else echo $(( after - before )); fi > "$RUN/ctx_overflow.txt"
  [ "${LENIENT:-0}" = 1 ] && printf 'LENIENT=1 — analysis only, not the reference\npins: scripts/sandbox_shim.py PINS%s\n' "${LENIENT_PINS:+ + LENIENT_PINS=$LENIENT_PINS}" > "$RUN/lenient.txt"
done

[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
DIRS=$(for a in $ARMS; do echo "runs/dev33_$a${TAG:+_$TAG}"; done)
REF=""                                                # earlier runs shown as references: B-2's clean base by default
for r in ${REFS-base}; do                             # (REFS="out8k nothink" = the W3 references, pre test-deps fix)
  [[ " $ARMS " != *" $r "* || -n "${TAG:-}" ]] && [ -d "runs/dev33_$r" ] && REF="$REF runs/dev33_$r"
done
echo; python3 scripts/compare_runs.py $REF $DIRS
echo; python3 scripts/growth.py $REF $DIRS | grep -E '^==|Δprompt|→ b|growth explained' || true
echo; python3 scripts/rulestats.py $REF $DIRS || true      # did the agent follow the prompt rules? (📷)
[ "${TRAJ:-1}" = 1 ] && { echo; for d in $DIRS; do python3 scripts/walkthrough.py "$d/traj"; done; } || true   # 📷 thinking re-sent? (scripts/traj_log.py)
[ "${TRAJ:-1}" = 1 ] && { for d in $DIRS; do echo; python3 scripts/run_costs.py "$d"; done; } || true   # 📷 tokens per tool output, command times, compactions (scripts/run_costs.py)
