#!/usr/bin/env python3
"""Does a (synthetic) thought make the action more likely under the policy? Score call targets with the E4B base.

    .venv-train/bin/python train/nll_score.py runs/sft/d3_relabel_31b_pilot --base /vol/hf/dequant/gemma-4-E4B-dq \
        --tokenizer google/gemma-4-E4B-it-qat-w4a16-ct --only-relabelled

For every call with a target (or only the relabelled ones): render the reply twice with train/render.py — without the
thought (reasoning = null) and with it — and compute, under the bf16 base, the NLL of the ACTION tokens (text +
tool call + stop, the same tokens in both variants) and the NLL of the thought tokens themselves. Prints per call and
the means: action NLL without vs with thought (a lower "with" means the thought helps the policy predict its own
action), thought NLL per token (how far the thought sits from the model's own distribution). This is the plan's
relabel acceptance signal next to the human check; every number printed.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Renderer, spans  # noqa: E402


def nll_of(model, ids, start):
    """per-token logprobs of ids[start:] given the prefix (one forward, logits only where needed)"""
    x = torch.tensor([ids], device="cuda")
    keep = torch.arange(start - 1, len(ids) - 1, device="cuda")
    with torch.no_grad():
        out = model(input_ids=x, logits_to_keep=keep)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    tgt = x[0, start:]
    return -lp[torch.arange(len(tgt)), tgt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--only-relabelled", action="store_true")
    ap.add_argument("--max-len", type=int, default=16384)
    a = ap.parse_args()
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    cls = getattr(transformers, "Gemma4ForConditionalGeneration", AutoModelForImageTextToText)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    R = Renderer(tok)
    model = cls.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = []
    print(f"nll_score · {a.records} · base {a.base} · {'relabelled calls only' if a.only_relabelled else 'all targets'}")
    print("  task             call  src            action_tok  nll/tok no-thought  with-thought   Δ      thought_tok  thought nll/tok")
    for f in sorted(glob.glob(os.path.join(a.records, "*.json"))):
        rec = json.load(open(f))
        if not isinstance(rec, dict) or not isinstance(rec.get("calls"), list):      # a summary or verdict file next to the records
            continue
        for c in rec["calls"]:
            t = c.get("target")
            if not t or (a.only_relabelled and not t.get("reasoning")):
                continue
            p = R.prompt(c["input_messages"], rec["tools"])
            p_ids = R.ids(p)
            if len(p_ids) > a.max_len:
                continue
            open_in_prompt = p.endswith("<|channel>thought\n")
            res = {}
            for variant, tgt in (("no_thought", dict(t, reasoning=None)), ("with_thought", t)):
                if variant == "with_thought" and not t.get("reasoning"):
                    continue
                reply = R.reply_synthetic(tgt, open_in_prompt)
                ids = R.ids(p + reply)
                r_ids = ids[len(p_ids):]
                sp = spans(tok, r_ids, open_in_prompt)
                nll = nll_of(model, ids, len(p_ids)).tolist()
                act = [nll[i] for k, s, e in sp if k in ("text", "tool_call", "stop") for i in range(s, e)]
                th = [nll[i] for k, s, e in sp if k == "thought" for i in range(s, e)]
                res[variant] = {"action_tok": len(act), "action_nll": sum(act) / max(len(act), 1),
                                "thought_tok": len(th), "thought_nll": sum(th) / max(len(th), 1)}
            if "with_thought" not in res:
                continue
            n0, n1 = res["no_thought"], res["with_thought"]
            row = {"instance_id": rec["instance_id"], "call": c["call"], "source": c.get("thought_source"),
                   "action_tok": n1["action_tok"], "nll_no_thought": n0["action_nll"], "nll_with_thought": n1["action_nll"],
                   "thought_tok": n1["thought_tok"], "thought_nll": n1["thought_nll"]}
            rows.append(row)
            print(f"  {row['instance_id']:<16} {row['call']:>4}  {row['source'] or '-':<14} {row['action_tok']:>10}  "
                  f"{row['nll_no_thought']:>18.3f}  {row['nll_with_thought']:>12.3f}  {row['nll_with_thought'] - row['nll_no_thought']:>+6.3f}  "
                  f"{row['thought_tok']:>11}  {row['thought_nll']:>15.3f}")
    if not rows:
        print("no scorable calls"); return 1
    m = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    better = sum(1 for r in rows if r["nll_with_thought"] < r["nll_no_thought"])
    print(f"nll_score: {len(rows)} calls · action nll/tok without thought {m('nll_no_thought'):.3f} · with thought {m('nll_with_thought'):.3f} "
          f"(Δ {m('nll_with_thought') - m('nll_no_thought'):+.3f}; lower with thought in {better}/{len(rows)}) · thought nll/tok {m('thought_nll'):.3f}")
    json.dump(rows, open(a.records.rstrip("/") + ".nll.json", "w"), indent=1)   # beside the dir: it holds records only
    return 0


if __name__ == "__main__":
    sys.exit(main())
