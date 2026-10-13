#!/usr/bin/env python3
"""W2 follow-up: does the agent's old thinking stay in the prompt? Fit per-step prompt growth from the traces.

    python scripts/growth.py runs/dev33_sample runs/dev33_out8k      # 📷

For consecutive steps i → i+1 of one task (no compaction in between, one model call each):
    Δprompt = prompt_tokens[i+1] - prompt_tokens[i]
            ≈ a · tool_output_chars[i]  +  b · completion_tokens[i]  +  c
completion_tokens counts everything the model generated at step i: thinking + tool call. So
    b ≈ 1   → thinking is re-sent in every later prompt (stripping old thoughts would free that space)
    b ≈ 0.1 → only the tool call is kept; thinking is already dropped (then read_file/run_command are the levers)
a should come out ≈ 1/3.5 tokens per char (sanity check of the fit). c = per-turn overhead (role markers etc.).
Also prints what the unattributed "?" observations of buckets.py are (steps with an observation but no tool call).
"""
import collections
import json
import pathlib
import sys


def solve3(rows):
    """least squares y = a·x1 + b·x2 + c via the 3x3 normal equations (no numpy needed)"""
    S = [[0.0] * 4 for _ in range(3)]
    for x1, x2, y in rows:
        v = (x1, x2, 1.0)
        for i in range(3):
            for j in range(3):
                S[i][j] += v[i] * v[j]
            S[i][3] += v[i] * y
    for i in range(3):                                   # Gauss-Jordan
        p = max(range(i, 3), key=lambda r: abs(S[r][i]))
        S[i], S[p] = S[p], S[i]
        for r in range(3):
            if r != i and S[i][i]:
                f = S[r][i] / S[i][i]
                S[r] = [a - f * b for a, b in zip(S[r], S[i])]
    return [S[i][3] / S[i][i] if S[i][i] else float("nan") for i in range(3)]


def obs_chars(st):
    o = st.get("observation")
    if not isinstance(o, dict):
        return 0
    body = o.get("content", "")
    if isinstance(body, str) and body[:1] == "{":
        try:
            j = json.loads(body)
            body = j.get("content") or j.get("output") or j.get("stdout") or body if isinstance(j, dict) else body
        except ValueError:
            pass
    return len(body if isinstance(body, str) else json.dumps(body))


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]] or sys.exit(__doc__)
    for run in runs:
        rows, orphans = [], collections.Counter()
        orphan_ex = []
        for tf in sorted((run / "traces").glob("trace_*.json")):
            steps = json.loads(tf.read_text()).get("steps", [])
            for s in steps:
                if s.get("observation") and not s.get("tool_calls"):
                    orphans[s.get("source", "?")] += 1
                    if len(orphan_ex) < 2:
                        orphan_ex.append(f"{tf.stem[6:]} step {s.get('step_id')}: model={s.get('model_name')} "
                                         f"{json.dumps(s.get('observation'))[:90]}")
            for a, b in zip(steps, steps[1:]):
                ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
                pa, pb, ca = ma.get("prompt_tokens"), mb.get("prompt_tokens"), ma.get("completion_tokens")
                if None in (pa, pb, ca) or a.get("llm_call_count", 1) != 1:
                    continue
                if a.get("model_name") != b.get("model_name"):   # sub-agent steps have their own history
                    continue
                d = pb - pa
                if d < 0:                                         # compaction or a new context: skip
                    continue
                rows.append((obs_chars(a), ca, d))
        print(f"== {run.name}: {len(rows)} step pairs")
        if len(rows) < 10:
            print("  too few pairs"); continue
        a, b, c = solve3(rows)
        print(f"  Δprompt ≈ {a:.3f}·tool_chars + {b:.2f}·completion_tokens + {c:.0f}")
        print(f"  → b = {b:.2f}: " + ("thinking STAYS in the prompt" if b > 0.6 else
                                       "thinking is DROPPED (only tool calls kept)" if b < 0.3 else "partly kept / unclear"))
        print(f"    (a sanity: expect ≈ {1/3.5:.2f} tokens/char)")
        tot_c = sum(r[1] for r in rows); tot_o = sum(r[0] for r in rows) * a; tot_d = sum(r[2] for r in rows)
        print(f"  growth explained: tool output {100*tot_o/tot_d:.0f}% · generated tokens {100*b*tot_c/tot_d:.0f}%"
              f" · overhead {100*c*len(rows)/tot_d:.0f}%   (mean Δ {tot_d/len(rows):.0f} tok/step)")
        if orphans:
            print(f"  observation without tool call: {dict(orphans)}")
            for e in orphan_ex:
                print("   ", e)
        print()


if __name__ == "__main__":
    main()
