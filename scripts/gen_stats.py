#!/usr/bin/env python3
"""Does nothink stop the model thinking, or only drop old thoughts from the history? Per run, over the agent's steps:

    python scripts/gen_stats.py runs/dev33_out8k runs/dev33_nothink runs/dev33_think1k     # 📷

  gen tok/step        completion_tokens per model call (thinking + tool call)
  thought chars/step  length of the thought text saved in the trace (`message`)
Reading: nothink ≈ out8k gen tok/step → it still thinks, and Finding #5 (b 1.00 → 0.12) is only the history dropping it;
nothink much lower → it also thinks less. think1k ≈ out8k → the 1024 budget was not enforced.
"""
import glob
import json
import sys

print(f"{'run':28} {'steps':>6} {'gen tok/step':>13} {'thought chars/step':>19}")
for r in sys.argv[1:] or sys.exit(__doc__):
    S = [s for f in glob.glob(f"{r}/traces/*.json") for s in json.load(open(f)).get("steps", [])
         if s.get("source") == "agent" and (s.get("metrics") or {}).get("completion_tokens") is not None]
    if not S:
        print(f"{r.rstrip('/').split('/')[-1]:28} no agent steps with metrics"); continue
    gen = sum(s["metrics"]["completion_tokens"] for s in S) / len(S)
    th = sum(len(str(s.get("message") or "")) for s in S) / len(S)
    print(f"{r.rstrip('/').split('/')[-1]:28} {len(S):>6} {gen:>13.0f} {th:>19.0f}")
