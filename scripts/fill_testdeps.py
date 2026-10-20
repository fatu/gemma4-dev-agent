"""Pick the test wheels a venv lacks: print a JSON {name: wheel path} for `pip install --no-deps`.

    .venv-eval/bin/python scripts/fill_testdeps.py harness/wheels runs/testdeps

Why: with `--sandbox subprocess` the harness skips its test-dependency step (swegemma 0.2.11
`install_test_dependencies` returns early for SubprocessManager); the sandbox venv only inherits this venv.
The scorer's Docker sandbox injects the competition wheels/ instead. Without them, on Kaggle 4×L4 (2026-10-20):
requests verifications had 185 setup errors ("recursive dependency involving fixture 'httpbin'": the pytest-httpbin
plugin is missing) and fastapi test modules failed to import inline_snapshot.
Rules: never replace an installed package (the harness stack stays pinned), skip wheels for another Python/platform,
skip the scorer's own skip list and the task repos themselves (their workspace is installed editable).
"""
import importlib.metadata as md
import json
import pathlib
import sys

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

SKIP = {"pytest", "pytest-timeout", "pytest-asyncio", "pytest-xdist", "pluggy", "pip", "setuptools", "wheel",
        "fastapi", "rich", "requests", "httpx"}


def main(dirs):
    have = {canonicalize_name(d.metadata["Name"]) for d in md.distributions() if d.metadata["Name"]}
    ok_tags = set(sys_tags())
    best = {}
    for d in dirs:
        for w in pathlib.Path(d).glob("*.whl"):
            try:
                name, ver, _, tags = parse_wheel_filename(w.name)
            except Exception:
                continue
            if name in SKIP or name in have or not (tags & ok_tags):
                continue
            if name not in best or ver > best[name][0]:
                best[name] = (ver, str(w))
    print(json.dumps({n: p for n, (_, p) in sorted(best.items())}))


if __name__ == "__main__":
    main(sys.argv[1:])
