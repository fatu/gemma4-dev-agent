#!/usr/bin/env python3
"""One line per task: how a run solved (or lost) it — and, with a second run, the same task side by side (B-3g).

    python3 scripts/tasktable.py runs/dev33_base_ceilx                              # one run, 33 lines
    python3 scripts/tasktable.py runs/dev33_base_ceilx runs/dev33_base_k1            # two runs: A | B per task
    GEMMA="always:a b c never:x y z" python3 scripts/tasktable.py runs/dev33_base_ceilx   # tag tasks with another
                                                                                          # machine's always/never sets

Columns per run: outcome (✓ ✗ O T C E ∅ as in flips.py) · seconds · tool calls · call of the first edit · reads before
it · tests run · peak prompt tokens · patch bytes · files edited (basenames; a heredoc edit shows as "heredoc").
Ends with the band split: tasks resolved here that the other side never resolved (reachable), resolved by both,
failed here too (harness-bound). Task ids are printed (dev split is public); no task text. Keep photos private.
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rulestats import load, HEREDOC_RE   # noqa: E402
from flips import outcome                # noqa: E402

TEST_RE = re.compile(r"\bpytest\b|\bunittest\b|python3? -m pytest")


def facts(steps):
    calls = reads = tests = 0
    first = None
    peak = 0
    files = []
    for s in steps:
        if s.get("source") != "agent":
            continue
        peak = max(peak, (s.get("metrics") or {}).get("prompt_tokens") or 0)
        for c in s.get("tool_calls") or []:
            calls += 1
            n, a = c["function_name"], c.get("arguments") or {}
            cmd = a.get("command", "") or ""
            if n == "read_file" and first is None:
                reads += 1
            if n == "run_command" and TEST_RE.search(cmd):
                tests += 1
            edited = None
            if n in ("edit_file", "write_file"):
                edited = pathlib.Path(str(a.get("filepath") or a.get("path") or "?")).name
            elif n == "run_command" and HEREDOC_RE.search(cmd):
                # the file a heredoc script rewrites: the first source path it names (open()/Path()/sed target)
                paths = [p for p in re.findall(r"[\w./-]+\.(?:py|md|toml|cfg|txt|yaml|yml|json)", cmd)
                         if not p.startswith(("python", "-"))]
                edited = (pathlib.Path(paths[0]).name + "~") if paths else "heredoc"   # ~ = via heredoc
            if edited:
                if first is None:
                    first = calls
                if edited not in files:
                    files.append(edited)
    return dict(calls=calls, reads=reads, tests=tests, first=first, peak=peak, files=files)


def cell(r, f):
    if r is None:
        return f"{'–':>3} {'':>5} {'':>5} {'':>4} {'':>3} {'':>3} {'':>6} {'':>6}  "
    return (f"{outcome(r):>3} {r.get('duration_seconds') or 0:5.0f} {f['calls']:5} {f['first'] or '–':>4} "
            f"{f['reads']:3} {f['tests']:3} {f['peak']:6} {r.get('agent_patch_size') or 0:6}  "
            f"{','.join(f['files'][:3])}{'…' if len(f['files']) > 3 else ''}")


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]] or sys.exit(__doc__)
    data = [load(r) for r in runs]
    names = [r.name[len("dev33_"):] if r.name.startswith("dev33_") else r.name for r in runs]
    tasks = sorted({t for res, _ in data for t in res})
    gem = {}
    for part in (os.environ.get("GEMMA") or "").split():
        if part.endswith(":"):
            key = part[:-1]
        elif ":" in part:
            key, first = part.split(":", 1); gem[first] = key
        else:
            gem[part] = key
    hdr = f"{'out':>3} {'sec':>5} {'calls':>5} {'1st':>4} {'rd':>3} {'tst':>3} {'peak':>6} {'patch':>6}  files"
    print(f"{'task':16} " + "   | ".join(f"{n:<60}" for n in names))
    print(f"{'':16} " + "   | ".join(f"{hdr:<60}" for _ in names))
    for t in tasks:
        cells = []
        for res, tr in data:
            r = res.get(t)
            cells.append(cell(r, facts(tr.get(t, [])) if r else None))
        tag = f" [{gem[t]}]" if t in gem else ""
        print(f"{t:16} " + "   | ".join(f"{c:<60}" for c in cells) + tag)
    # band split against the first run
    res0 = data[0][0]
    won = {t for t, r in res0.items() if r.get("resolved")}
    print(f"\n== {names[0]}: resolved {len(won)}/{len(res0)}")
    if len(data) > 1:
        other = {t for res, _ in data[1:] for t, r in res.items() if r.get("resolved")}
        print(f"   resolved here, never by {' / '.join(names[1:])}: {len(won - other)}: {' '.join(sorted(won - other))}")
        print(f"   resolved by both: {len(won & other)}: {' '.join(sorted(won & other))}")
        print(f"   lost here, resolved there: {len(other - won)}: {' '.join(sorted(other - won))}")
        print(f"   lost everywhere: {len(set(res0) - won - other)}: {' '.join(sorted(set(res0) - won - other))}")
    if gem:
        alw = {t for t, k in gem.items() if k == "always"}; nev = {t for t, k in gem.items() if k == "never"}
        print(f"   vs GEMMA sets — reachable (won here, Gemma never): {len(won & nev)}: {' '.join(sorted(won & nev))}")
        print(f"   lost here although Gemma always: {len(alw - won)}: {' '.join(sorted(alw - won))}")
        print(f"   lost here and Gemma never (harness-bound candidates): {len(nev - won)}: {' '.join(sorted(nev - won))}")
    by_out = {}
    for t, r in res0.items():
        by_out[outcome(r)] = by_out.get(outcome(r), 0) + 1
    print("   outcomes: " + " · ".join(f"{k} {v}" for k, v in sorted(by_out.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
