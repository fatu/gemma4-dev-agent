#!/usr/bin/env bash
# One-screen environment report for a photo (W1 env fixes → repo).
#   bash scripts/env_report.sh
cd "$(dirname "$0")/.."
PY=.venv-eval/bin/python; [ -x "$PY" ] || { echo "no .venv-eval/bin/python here"; PY=python3; }
W="${COLUMNS:-160}"
# importlib.metadata, not pip: uv-built venvs have no pip, and a bare `pip` silently reports ANOTHER env.
echo "== PKGS  ($("$PY" -c 'import sys; print(sys.prefix)'))"
"$PY" - <<'PYEOF' | column -c "$W"
import importlib.metadata as md, re
want = re.compile(r'^(vllm|torch|transformers|compressed.tensors|swegemma|adk.*|google.adk|prometheus.*|pytest.*|kaggle)$', re.I)
for n, v in sorted({(d.metadata['Name'], d.version) for d in md.distributions() if d.metadata['Name'] and want.match(d.metadata['Name'])}, key=lambda x: x[0].lower()):
    print(f"{n}=={v}")
PYEOF
echo "== kaggle CLI: $(kaggle --version 2>/dev/null || echo none)"
echo "== WHEELHOUSE (prometheus/pytest): $(ls harness/wheelhouse 2>/dev/null | grep -iE 'prometheus|pytest' | paste -sd' ' - || true)"
echo "== PIP CONFIG: $(pip config list 2>/dev/null | paste -sd' ' -)  · swegemma bin: $(command -v swegemma || echo none)"
echo "== GIT: $(git log --oneline -1)"
echo "== DIRTY: $(git status --short | paste -sd' ' -)"
