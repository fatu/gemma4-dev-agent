mode: replace
anchor: then run `timeout 150 python -m pytest tests/path/test_x.py -x -q -k keyword 2>&1 | tail -40`.
anchor_alt: then run `timeout 150 python -m pytest tests/path/test_x.py -x -q -k keyword`.
---
then run `set -o pipefail; timeout 150 python -m pytest tests/path/test_x.py -x -q -k keyword 2>&1 | tail -40` (only the last 40 lines: the failing assertion and the summary; pipefail keeps pytest's exit code).
