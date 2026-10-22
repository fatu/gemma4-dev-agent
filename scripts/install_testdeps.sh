#!/usr/bin/env bash
# Fill the test-dependency gaps in .venv-eval so `--sandbox subprocess` verifications can run every repo's tests.
#   bash scripts/install_testdeps.sh          # once, after setup_eval_env.sh (needs PyPI for the 4 extra packages)
#   CHECK=requests_6592 bash scripts/install_testdeps.sh   # + a no-agent verification: expect 0 "recursive dependency"
# See scripts/fill_testdeps.py for why. Extra packages = what the competition wheels/ lacks but the tests import:
# pytest-httpbin 2.1.0 + httpbin 0.10 (requests' requirements-dev pins) and inline-snapshot + dirty-equals (fastapi).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv-eval}"; PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no $PY — run scripts/setup_eval_env.sh first"; exit 1; }
WHEELS="${WHEELS:-harness/wheels}"; [ -d "$WHEELS" ] || { echo "no $WHEELS — run scripts/get_harness.sh"; exit 1; }
DL=runs/testdeps; mkdir -p "$DL"
# a throwaway pip for the venv's Python version (uv-built venvs have no pip)
BASE=$("$PY" -c 'import sys; print(sys._base_executable)')
if "$PY" -m pip --version >/dev/null 2>&1; then PIP=("$PY" -m pip)
elif "$BASE" -m pip --version >/dev/null 2>&1; then PIP=("$BASE" -m pip)
else [ -x runs/dlvenv/bin/pip ] || "$BASE" -m venv runs/dlvenv; PIP=(runs/dlvenv/bin/pip); fi
"${PIP[@]}" download -q --dest "$DL" --only-binary=:all: \
  "pytest-httpbin==2.1.0" "httpbin~=0.10.0" inline-snapshot dirty-equals
PICK=$("$PY" scripts/fill_testdeps.py "$WHEELS" "$DL")
FILES=(); while IFS= read -r f; do [ -n "$f" ] && FILES+=("$f"); done \
  < <(echo "$PICK" | "$PY" -c 'import json,sys; [print(p) for p in json.load(sys.stdin).values()]')   # no mapfile: macOS bash 3.2
echo "adding ${#FILES[@]}: $(echo "$PICK" | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))')"
if [ ${#FILES[@]} -gt 0 ]; then
  if command -v uv >/dev/null; then uv pip install --python "$PY" --no-deps "${FILES[@]}"
  else "${PIP[@]}" install --python "$PY" --no-deps "${FILES[@]}" 2>/dev/null || "$PY" -m pip install --no-deps "${FILES[@]}"; fi
fi
"$PY" -c 'import pytest_httpbin, httpbin, inline_snapshot, dirty_equals, swegemma, google.adk; print("testdeps ok · harness imports ok")'
if [ -n "${CHECK:-}" ]; then                          # no model needed: verifies the unpatched base commit
  SUB="${SUBMISSION:-harness/sample_submission}"; RUN="runs/testdeps_check_$CHECK"; rm -rf "$RUN"
  "$VENV/bin/swegemma" eval --tasks harness/tasks.jsonl --snapshots-dir harness/snapshots --submission-dir "$SUB" \
    --results-dir "$RUN" --sandbox subprocess --task-ids "$CHECK" --skip-agent-patch --display quiet >/dev/null 2>&1 || true
  LOGF=$(ls "$RUN"/test_outputs/*.log 2>/dev/null | head -1)
  [ -n "$LOGF" ] || { echo "no test log under $RUN"; exit 1; }
  echo "$CHECK: $(grep -oE '[0-9]+ (passed|failed|errors?)[^=]*' "$LOGF" | tail -1) · recursive-fixture errors: $(grep -c 'recursive dependency' "$LOGF" || true) · inline_snapshot missing: $(grep -c "No module named 'inline_snapshot'" "$LOGF" || true)"
fi
