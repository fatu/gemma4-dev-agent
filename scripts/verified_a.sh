#!/usr/bin/env bash
# SWE-bench Verified in OUR harness, route A of BOX_PLAN B-8: the test bench for a LoRA trained on our format (our prompt,
# our 9 tools, the subprocess sandbox, pytest verify), unlike route B (mini-swe-agent's bash tool in the official images).
# Reuses the SWE-rebench pipeline: rebench_build.py (dataset tag `verified`) → rebench_gen.sh build / gold → lane.sh.
#   bash scripts/verified_a.sh prep     # internet, no GPU, ~1 min: Verified rows (pinned parquet) → ids.txt; swebench's
#                                       #   install specs (.venv-swe from route B setup) → specs.json; overlap with our training splits
#   bash scripts/verified_a.sh build    # internet, CPU: snapshots + per-task deps (4 build processes)
#   bash scripts/verified_a.sh gold     # CPU: the reference patch must pass here → ok.txt (the bench = these tasks)
#   bash scripts/verified_a.sh run      # 4 cards: one full pass of ok.txt per card, in parallel (4 seeds in the time of 1)
#   bash scripts/verified_a.sh report   # after the lanes: resolved per pass, mean, per repo, pooled edit stats
# Knobs (all printed): OUT (runs/verified_a) · SKIP_REPOS (django/django sympy/sympy psf/requests always: django and sympy
# are not pytest, requests is a competition repo; + by default the compiled repos astropy/astropy matplotlib/matplotlib
# scikit-learn/scikit-learn: their C extensions are not built in the workspace the sandbox imports first, so they cannot
# pass gold; SKIP_REPOS="" tries them, at hours of failed compiles) · BUILD_JOBS (4) · MIN_FREE_GB (20) · DEPS_TIMEOUT
# (600 s) · CONC_GOLD (8) · GPUS ("0 1 2 3") · GTAG (va → runs/dev33_<ARM>_va1..va4) · ARM (base) · LATEST (e4b) ·
# ADAPTER ("" = none; a LoRA name under runs/adapters/ for the A/B) · MODEL (E4B) · TIME_MIN (7) · MAX_TURNS (80) ·
# LANE_CONC (1 = the scorer's shape; 2 halves the wall time but shares a server between two tasks).
# Install specs: swebench's MAP_REPO_VERSION_TO_SPECS per (repo, version): pip_packages + the `packages` names (not the
# requirements.txt / environment.yml markers; conda envs are not reproduced), requirements files only where swebench uses
# them (packages = requirements.txt), extras of `pip install -e .[x]`, the `sed -i` lines of pre_install (apt/wget lines
# dropped). Python stays the eval venv's 3.12 (specs say 3.9–3.11), so old pins rarely install: the build then retries
# with the names unpinned, numpy capped < 2. Tasks that still do not run fail gold and are not in the bench: it is a
# biased subset of Verified (pure-Python repos, mostly sphinx / xarray / pytest / pylint), comparable between our own
# arms only — not to Verified scores.
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?prep|build|gold|run|report}"
OUT="${OUT:-runs/verified_a}"; SKIP_REPOS="${SKIP_REPOS-astropy/astropy matplotlib/matplotlib scikit-learn/scikit-learn}"
BUILD_JOBS="${BUILD_JOBS:-4}"; MIN_FREE_GB="${MIN_FREE_GB:-20}"; DEPS_TIMEOUT="${DEPS_TIMEOUT:-600}"; CONC_GOLD="${CONC_GOLD:-8}"
GPUS=(${GPUS:-0 1 2 3}); GTAG="${GTAG:-va}"; ARM="${ARM:-base}"; LATEST="${LATEST:-e4b}"; ADAPTER="${ADAPTER-}"
MODEL="${MODEL:-google/gemma-4-E4B-it-qat-w4a16-ct}"; TIME_MIN="${TIME_MIN:-7}"; MAX_TURNS="${MAX_TURNS:-80}"; LANE_CONC="${LANE_CONC:-1}"
PY=.venv-eval/bin/python; ALWAYS="django/django sympy/sympy psf/requests"
export TASKS_FILE="$OUT/tasks.jsonl" SNAPSHOTS_DIR="$OUT/snapshots" REBENCH_SITE="$OUT/site"
echo "verified_a $STEP · out $OUT · skip repos [$ALWAYS${SKIP_REPOS:+ $SKIP_REPOS}] · build jobs $BUILD_JOBS · min free $MIN_FREE_GB GB · deps timeout $DEPS_TIMEOUT s · gold conc $CONC_GOLD · gpus ${GPUS[*]} · tags $GTAG<i> · arm $ARM · LATEST $LATEST · adapter ${ADAPTER:-none} · model $MODEL · TIME_MIN $TIME_MIN · MAX_TURNS $MAX_TURNS · lane concurrency $LANE_CONC · code $(git log --oneline -1 | cut -c1-40)"
case "$STEP" in
prep)
  [ -x .venv-swe/bin/python ] || { echo "no .venv-swe: run 'bash scripts/verified_box.sh setup' (B-8 route B) first"; exit 1; }
  mkdir -p "$OUT"
  .venv-swe/bin/python - "$OUT/specs.json" <<'PY' || exit 1
import importlib.metadata as md, json, re, shlex, sys
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
try:
    from swebench.harness.constants.python import MAP_REPO_TO_REQS_PATHS
except ImportError:
    from swebench.harness.constants import MAP_REPO_TO_REQS_PATHS
MARKERS = {"requirements.txt", "environment.yml"}
def name(p):
    return re.split(r"[=<>!~;\[ ]", p.strip().strip("'\""), maxsplit=1)[0]
def dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out
out, n = {}, 0
for repo, vers in MAP_REPO_VERSION_TO_SPECS.items():
    if not isinstance(vers, dict) or not all(isinstance(v, dict) for v in vers.values()):
        continue
    for ver, sp in vers.items():
        pk = sp.get("packages") or ""
        pins = dedup(list(sp.get("pip_packages") or []) + [p for p in shlex.split(pk) if p not in MARKERS])
        loose = dedup(["numpy<2" if name(p).lower() == "numpy" else name(p) for p in pins])
        ex = re.search(r"\.\[([\w,-]+)\]", sp.get("install") or "")
        sed = [c for c in sp.get("pre_install") or [] if c.lstrip().startswith(("sed ", "grep -q"))]
        out.setdefault(repo, {})[ver] = {
            "python": sp.get("python"), "pip_packages": pins, "pip_packages_unpinned": loose,
            "reqs_path": MAP_REPO_TO_REQS_PATHS.get(repo, []) if pk == "requirements.txt" else [],
            "extras": ex.group(1) if ex else "", "pre_install_sed": sed, "test_cmd": sp.get("test_cmd"),
            "conda_env_yml": pk == "environment.yml", "from": f"swebench {md.version('swebench')} specs"}
        n += 1
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"  specs: {n} repo versions of {len(out)} repos (swebench {md.version('swebench')}) → {sys.argv[1]}")
PY
  $PY -I - "$OUT" "$ALWAYS $SKIP_REPOS" <<'PY' || exit 1
import collections, glob, json, os, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
out, skip = sys.argv[1], set(sys.argv[2].split())
ds, rev, f = "princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "data/test-00000-of-00001.parquet"
src = os.environ.get("VA_PARQUET") or hf_hub_download(ds, f, repo_type="dataset", revision=rev)   # VA_PARQUET: a local copy (tests)
rows = pq.read_table(src, columns=["instance_id", "repo", "base_commit", "version"]).to_pylist()
specs = json.load(open(os.path.join(out, "specs.json")))
keep = sorted((r for r in rows if r["repo"] not in skip), key=lambda r: r["instance_id"])
by = collections.Counter(r["repo"] for r in rows)
print(f"  {ds}@{rev[:7]}: {len(rows)} tasks, {len(by)} repos · kept {len(keep)}, skipped {len(rows) - len(keep)}")
for repo, c in by.most_common():
    nospec = sum(1 for r in rows if r["repo"] == repo and str(r["version"]) not in specs.get(repo, {}))
    print(f"    {repo:<28} {c:>4}  {'skip' if repo in skip else 'keep'}{f' · {nospec} without a spec' if nospec else ''}")
with open(os.path.join(out, "ids.txt"), "w") as w:
    w.write("".join(f"verified {r['instance_id']} {r['base_commit']}\n" for r in keep))
kept_repos = {r["repo"].replace("/", "__") for r in keep}
ids = {r["instance_id"] for r in rows}
for sp in sorted(glob.glob("splits/*.txt")):
    toks = [l.split() for l in open(sp) if l.strip() and not l.startswith("#")]
    tids = [t[1] if len(t) > 1 and t[0] in ("nebius", "whitecircle", "verified") else t[0] for t in toks]
    same = [t for t in tids if t.rsplit("-", 1)[0] in kept_repos]
    exact = [t for t in tids if t in ids]
    if same or exact:
        print(f"  overlap {sp}: {len(same)} tasks from a kept Verified repo"
              f" ({dict(collections.Counter(t.rsplit('-', 1)[0] for t in same))}) · {len(exact)} identical ids")
print(f"  → {out}/ids.txt ({len(keep)} lines). A LoRA tested here must not train on tasks of the kept repos (overlap lines above).")
PY
  echo "📷 everything above" ;;
build)
  [ -s "$OUT/ids.txt" ] && [ -s "$OUT/specs.json" ] || { echo "run prep first"; exit 1; }
  IDS_FILE="$OUT/ids.txt" OUT="$OUT" SPECS="$OUT/specs.json" DEPS_TIMEOUT="$DEPS_TIMEOUT" BUILD_JOBS="$BUILD_JOBS" LEAN=1 \
    MIN_FREE_GB="$MIN_FREE_GB" UV_CACHE="$OUT/uvcache" bash scripts/rebench_gen.sh build
  $PY -I - "$OUT.build.log" <<'PY'
import collections, re, sys
how = collections.Counter()
for l in open(sys.argv[1]):
    m = re.search(r"^(\S+)-\d+ .* deps (FAILED|full \(no extras\)|full \(unpinned\)|full|partial \(unpinned, no reqs files\)|partial \(unpinned\)|partial)", l)
    if m:
        how[(m.group(1), m.group(2))] += 1
for repo in sorted({r for r, _ in how}):
    print(f"  deps {repo:<28} " + " · ".join(f"{h} {c}" for (r, h), c in sorted(how.items()) if r == repo))
PY
  echo "📷 the == build line + the deps lines (per repo: which install attempt worked)" ;;
gold)
  [ -s "$OUT/built.txt" ] || { echo "run build first"; exit 1; }
  OUT="$OUT" CONC_GOLD="$CONC_GOLD" bash scripts/rebench_gen.sh gold | grep -vE " gold patch → NOT" | sed 's/ (the kit resolved 55 of its 81)//'
  echo "  per repo (gold ok / built):"
  for r in $(sed 's/-[0-9]*$//' "$OUT/built.txt" | sort -u); do
    echo "    $r $(grep -c "^$r-" "$OUT/ok.txt") / $(grep -c "^$r-" "$OUT/built.txt")"
  done
  echo "📷 the == gold line + the per-repo lines (the NOT list is in $OUT/gold.log)" ;;
run)
  [ -s "$OUT/ok.txt" ] || { echo "run gold first"; exit 1; }
  for g in "${GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    [ "$used" -lt 2000 ] || { echo "card $g in use ($used MiB): free it (e.g. GPU=$g bash scripts/verified_box.sh stop) or set GPUS="; exit 1; }
  done
  i=0
  for g in "${GPUS[@]}"; do
    i=$((i + 1))
    GPU=$g TAG=$GTAG$i TIME_MIN=$TIME_MIN MAX_TURNS=$MAX_TURNS MODEL=$MODEL ARMS="$ARM" SPLIT="$OUT/ok.txt" LATEST=$LATEST \
      ADAPTER="$ADAPTER" LANE_CONC=$LANE_CONC SUBMISSION="${SUBMISSION:-runs/base_lb024}" TRAJ=1 \
      nohup bash scripts/lane.sh run > "runs/verified_a_lane$g.log" 2>&1 &
    echo "  pass $GTAG$i → GPU $g · $(wc -l < "$OUT/ok.txt") tasks · runs/dev33_${ARM}_$GTAG$i · log runs/verified_a_lane$g.log"
  done
  echo "📷 these lines; progress: tail -2 runs/verified_a_lane*.log · when all lanes end: bash scripts/verified_a.sh report" ;;
report)
  DIRS=(); i=0
  for g in "${GPUS[@]}"; do i=$((i + 1)); DIRS+=("runs/dev33_${ARM}_$GTAG$i"); done
  $PY -I - "$OUT/ok.txt" "${DIRS[@]}" <<'PY'
import collections, json, os, sys
ok = [l.strip() for l in open(sys.argv[1]) if l.strip()]
per, pooled, n = [], collections.Counter(), collections.Counter(t.rsplit("-", 1)[0] for t in ok)
for d in sys.argv[2:]:
    p = os.path.join(d, "task_results.jsonl")
    if not os.path.exists(p):
        print(f"  {d}: no task_results.jsonl"); continue
    rs = [json.loads(l) for l in open(p)]
    res = [r["instance_id"] for r in rs if r.get("resolved")]
    per.append(len(res)); pooled.update(t.rsplit("-", 1)[0] for t in res)
    print(f"  {d}: resolved {len(res)} / {len(rs)} results (bench {len(ok)})")
if per:
    print(f"== mean resolved {sum(per) / len(per):.2f} / {len(ok)} over {len(per)} passes ({min(per)}–{max(per)})")
    for repo, c in n.most_common():
        print(f"    {repo:<28} {pooled[repo] / len(per):5.2f} / {c} per pass")
PY
  python3 scripts/edit_stats.py --pooled "${DIRS[@]}" 2>&1 | head -30
  echo "📷 the == block + the edit stats" ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
