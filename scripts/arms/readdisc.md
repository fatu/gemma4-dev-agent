

## Context Discipline (your context window is 32K tokens and every tool output stays in it)
- `read_file` only with `start_line` / `end_line`, at most ~80 lines per call. Locate first (`search_similar_code`,
  `get_code_neighbors`, the problem statement's line numbers), then read exactly that range. Never read a whole file.
- Never read the same range twice. If you need a file again, read only the lines you have not seen.
- Keep `run_command` output short: `pytest … -q -x --no-header -p no:cacheprovider 2>&1 | tail -40`, `grep -n … | head -20`.
  Never print whole files with `cat`; use `read_file` ranges.
- Think briefly: a few sentences per step, then act. Do not restate file contents or earlier reasoning in your thoughts.
