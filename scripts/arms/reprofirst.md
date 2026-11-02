mode: replace
anchor: - If the issue includes a code snippet that reproduces the problem, you may run it once with python -c (one call, never create a file) to see the real error.
---
- Reproduce first (failure study 2026-11-01: 31B reproduced before editing in 2/3 of the tasks it solved, E2B/E4B never). Your first call runs the issue's example, or the smallest call that shows the bug, with `python3 -c` (never create a file) and you see it fail. After your edit, run the same command again and see the expected result. Never submit without that second run.
