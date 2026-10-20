#!/usr/bin/env bash
# Scorer-like sandbox: build the competition's Docker image (swebench-sandbox:latest), then check it with no-agent
# verifications side by side with the subprocess sandbox. Needs docker usable without sudo; no vLLM.
#   bash scripts/build_sandbox_image.sh            # build + image facts + no-agent check (docker vs subprocess) 📷 the table
#   CHECK="requests_6592 fastapi_13713" MODES="docker subprocess" bash scripts/build_sandbox_image.sh
#   SANDBOX=docker nohup bash scripts/eval_dev.sh > runs/eval_dev_docker.log 2>&1 &     # afterwards: agent runs in Docker
# Why (HARNESS_README § 4): the scorer runs Phase 1 (agent) and Phase 2 (verification) each in a fresh container from
# harness/docker/ (python:3.13-slim, network none, 4 GiB, 2 vCPUs), and installs test deps into it. Our runs used
# --sandbox subprocess: no network cut, no limits, the host venv's Python (3.12 on this box, 3.13 on the scorer), and
# install_test_dependencies() returns early for it. With --sandbox docker the harness applies all of that itself; this
# script only builds the image it expects.
# The Dockerfiles and wheels are competition data: they stay in harness/ (gitignored, never committed). The build
# context is a throwaway dir under runs/ with just imp.py, telnetlib.py and wheels/ (harness/ is ~22 GB of snapshots).
# Reading the table: a no-agent run has an empty patch, so resolved is always False; look at the pytest summary.
# httpbin-rec > 0 = requests' tests can't find the pytest-httpbin plugin; no-module = imports the sandbox lacks.
# The docker run may copy test deps from the host venv (README § 4.2 step 5), so "host venv has pytest_httpbin" is
# printed too: docker clean while the host has it doesn't prove the real scorer has it.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-swebench-sandbox:latest}"
DOCKERFILE="${DOCKERFILE:-harness/docker/Dockerfile.public}"   # = Dockerfile.sandbox + the public wheels in /wheels
VENV="${VENV:-.venv-eval}"; PY="$VENV/bin/python"
CHECK="${CHECK:-requests_6592 fastapi_13713}"
MODES="${MODES:-docker subprocess}"
OUT=runs/sandbox_check

command -v docker >/dev/null && docker info >/dev/null 2>&1 \
  || { echo "docker is not usable by $(whoami): sudo usermod -aG docker $(whoami), then log in again"; exit 1; }
[ -f "$DOCKERFILE" ] || { echo "no $DOCKERFILE — run scripts/get_harness.sh"; exit 1; }
[ -x "$PY" ] || { echo "no $PY — run scripts/setup_eval_env.sh first"; exit 1; }

# ---- 1. build
CTX=runs/.sandbox_ctx; rm -rf "$CTX"; mkdir -p "$CTX"; trap 'rm -rf "$CTX"' EXIT
cp harness/docker/imp.py harness/docker/telnetlib.py "$CTX"/
cp -r harness/wheels "$CTX"/wheels
t0=$(date +%s)
docker build -q -f "$DOCKERFILE" -t "$IMAGE" "$CTX"
BUILD_S=$(( $(date +%s) - t0 ))
FACTS=$(docker run --rm --network none "$IMAGE" sh -c \
  'echo "$(python3 --version | cut -d" " -f2)|$(pytest --version 2>&1 | head -1 | grep -oE "[0-9][0-9.]+")|$(ls /wheels | wc -l)|$(python3 -c "import pytest_httpbin" 2>/dev/null && echo yes || echo no)"')
HOST_HTTPBIN=$("$PY" -c 'import pytest_httpbin' 2>/dev/null && echo yes || echo no)
HOST_PY=$("$PY" -c 'import platform; print(platform.python_version())')

# ---- 2. the Docker backend's Python client (only if swegemma imports one and it is missing)
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if grep -rqsE '^\s*(import docker|from docker )' "$SITE/adk_eval_core" "$SITE/swegemma" && ! "$PY" -c 'import docker' 2>/dev/null; then
  echo "swegemma's Docker backend imports the docker SDK, missing in $PY → installing it (--no-deps)"
  if command -v uv >/dev/null; then uv pip install --python "$PY" --no-deps docker; else "$PY" -m pip install --no-deps docker; fi
fi

# ---- 3. no-agent verifications, same tasks in each sandbox
rm -rf "$OUT"
for mode in $MODES; do
  mkdir -p "$OUT/$mode"
  for tid in $CHECK; do
    echo "== $mode $tid ($(date +%H:%M))"
    t0=$(date +%s)
    "$VENV/bin/swegemma" eval --tasks harness/tasks.jsonl --snapshots-dir harness/snapshots \
      --submission-dir harness/sample_submission --results-dir "$OUT/$mode/$tid" --sandbox "$mode" \
      --task-ids "$tid" --skip-agent-patch --display quiet --verification-timeout-seconds 300 \
      > "$OUT/$mode/$tid.log" 2>&1 || echo "   swegemma eval exited $? (see $OUT/$mode/$tid.log)"
    echo $(( $(date +%s) - t0 )) > "$OUT/$mode/$tid.secs"
  done
done

# ---- 4. one-screen summary 📷
OUT="$OUT" MODES="$MODES" CHECK="$CHECK" FACTS="$FACTS" IMAGE="$IMAGE" BUILD_S="$BUILD_S" HOST_HTTPBIN="$HOST_HTTPBIN" \
  HOST_PY="$HOST_PY" python3 - <<'PY'
import collections, json, os, pathlib, re
e = os.environ
py, pyt, nwheels, httpbin = (e["FACTS"].split("|") + ["?"] * 4)[:4]
print(f"\n== image {e['IMAGE']} (built in {e['BUILD_S']} s): Python {py} · pytest {pyt} · /wheels {nwheels} · "
      f"pytest_httpbin in image: {httpbin}")
print(f"   host venv: Python {e['HOST_PY']} · pytest_httpbin: {e['HOST_HTTPBIN']}")
print(f"\n{'mode':<11}{'task':<17}{'secs':>5}  {'pytest summary':<48}{'httpbin-rec':>11}{'no-module':>10}  missing / error")
for mode in e["MODES"].split():
    for tid in e["CHECK"].split():
        run = pathlib.Path(e["OUT"], mode, tid)
        secs = (run.parent / f"{tid}.secs").read_text().strip() if (run.parent / f"{tid}.secs").exists() else "?"
        txt = "".join(f.read_text(errors="replace") for f in run.glob("test_outputs/*.log"))
        summ = ([l.strip(" =") for l in txt.splitlines() if re.search(r"\d+ (passed|failed|errors?)", l)] or ["(no pytest summary)"])[-1]
        summ = re.sub(r" in [\d.]+s.*", "", summ)                     # drop the timing: it is in the secs column
        mods = collections.Counter(re.findall(r"No module named '([\w.]+)'", txt))
        row = {}
        if (run / "task_results.jsonl").exists():
            row = json.loads((run / "task_results.jsonl").read_text().splitlines()[-1])
        note = ", ".join(f"{m}×{n}" for m, n in mods.most_common(3)) or (str(row.get("error") or "")[:40] or "-")
        print(f"{mode:<11}{tid:<17}{secs:>5}  {summ[:47]:<48}{txt.count('recursive dependency'):>11}{sum(mods.values()):>10}  {note}")
PY
