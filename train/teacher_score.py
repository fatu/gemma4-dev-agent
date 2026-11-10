#!/usr/bin/env python3
"""31B teacher scoring of the STUDENT's own sampled replies (token-level on-policy distillation, BOX_PLAN B-4).

    .venv-eval/bin/python train/teacher_score.py runs/opd/s0/rows.jsonl --shard 0 --nshards 8 --out runs/opd/s0/t31_0.npz \
        [--model google/gemma-4-31B-it-qat-w4a16-ct] [--k 20] [--mbt 4096] [--gpu-util 0.80] [--batch 8]
    python3 train/teacher_score.py --selftest                       # laptop: alignment check with a fake engine
    python3 train/teacher_score.py rows.jsonl --fake same|other --out x.npz   # laptop: arrays without vLLM (tests kd_probe)

Input: render.py --pack rows of the student's trajectories ({input_ids, labels}; labels = the student's reply tokens:
thought, text, tool call, stop token). Engine: vLLM offline `LLM` (the eval venv's 0.19.1), prompt_token_ids = the whole
row, max_tokens 1, SamplingParams(prompt_logprobs=K, detokenize=False). vLLM returns raw (temperature-free, soft-capped)
log-softmax values for every prompt position; prompt_logprobs[j] is the distribution of token j given tokens < j, and it
always holds the actual token j with its rank. Requests with prompt_logprobs skip prefix-cache reads, so each row is one
full prefill. Memory: the prompt-logprob path materialises [chunk, 262,144] fp32 logits per scheduled chunk, which
vLLM's profiling does not reserve (8,192 tokens → 8.6 GB) → --mbt 4096 and --gpu-util 0.80. Rows go to vLLM --batch at a
time and only the label positions are kept, so the per-position Python objects of a batch are freed before the next.
--shard i --nshards n: this process takes rows i, i+n, i+2n, … (one process per card, CUDA_VISIBLE_DEVICES set by the
caller); the npz keys carry the GLOBAL row index, so the shards' files are read together.
Output per row r: r/pos (label positions j), r/t_ids [n,K] int32, r/t_lp [n,K] float32 (the teacher's top-K at j),
r/tok_lp [n] (the teacher's logprob of the student's token j), r/tok_rank [n]. Printed: knobs, rows / tokens, prefill
tok/s, the teacher's NLL of the student's tokens, top-1 / in-top-K agreement, per span (thought / name / args / text).
"""
import argparse
import json
import sys
import time

import numpy as np


def align(row, prompt_logprobs, k):
    """→ dict of arrays over the label positions of one row. prompt_logprobs: list (len = len(input_ids)) of
    None | {token_id: obj with .logprob and .rank}; entry j is the teacher's distribution of input_ids[j]."""
    ids, labels = row["input_ids"], row["labels"]
    pos = [j for j in range(1, len(ids)) if labels[j] != -100]
    t_ids = np.zeros((len(pos), k), np.int32); t_lp = np.full((len(pos), k), -1e4, np.float32)
    tok_lp = np.zeros(len(pos), np.float32); tok_rank = np.zeros(len(pos), np.int32)
    for n, j in enumerate(pos):
        d = prompt_logprobs[j]
        tok = ids[j]
        top = sorted(((tid, lp) for tid, lp in d.items() if lp.rank is not None and lp.rank <= k), key=lambda x: x[1].rank)[:k]
        for m, (tid, lp) in enumerate(top):
            t_ids[n, m] = tid; t_lp[n, m] = lp.logprob
        if len(top) < k:                                         # fewer than k (ties / tiny vocab in tests): pad with the
            t_ids[n, len(top):] = t_ids[n, 0]; t_lp[n, len(top):] = -1e4   # first id at ~zero probability
        tok_lp[n] = d[tok].logprob; tok_rank[n] = d[tok].rank or 0
    return {"pos": np.array(pos, np.int32), "t_ids": t_ids, "t_lp": t_lp, "tok_lp": tok_lp, "tok_rank": tok_rank}


def fake(row, k, mode):
    """arrays as align() returns them, without an engine: `same` = the teacher's top-1 is the student's token
    (rank 1 everywhere), `other` = the teacher prefers id+1 and puts the student's token second"""
    ids, labels = row["input_ids"], row["labels"]
    pos = [j for j in range(1, len(ids)) if labels[j] != -100]
    lp = np.log(np.exp(-np.arange(k, dtype=np.float64)) / np.exp(-np.arange(k, dtype=np.float64)).sum() * 0.9).astype(np.float32)
    t_ids = np.zeros((len(pos), k), np.int32); t_lp = np.tile(lp, (len(pos), 1))
    for n, j in enumerate(pos):
        others = [ids[j] + 1 + m for m in range(k)]
        t_ids[n] = ([ids[j]] + others[:k - 1]) if mode == "same" else ([ids[j] + 1, ids[j]] + others[1:k - 1])
    rank = np.ones(len(pos), np.int32) * (1 if mode == "same" else 2)
    return {"pos": np.array(pos, np.int32), "t_ids": t_ids, "t_lp": t_lp, "tok_lp": t_lp[:, rank[0] - 1].copy() if len(pos) else
            np.zeros(0, np.float32), "tok_rank": rank}


def span_kinds(toks):
    """thought / name / args / text per token of one reply run (Gemma 4 format, see render.spans)"""
    tc = toks.index("<|tool_call>") if "<|tool_call>" in toks else len(toks)
    state = "thought" if "<channel|>" in toks[:tc] else "text"
    out = []
    for t in toks:
        if state == "thought":
            out.append("thought"); state = "text" if t == "<channel|>" else state; continue
        if t == "<|tool_call>":
            state = "name"
        out.append(state)
        if state == "name" and t == "{":
            state = "args"
        elif state == "args" and t == "<tool_call|>":
            state = "text"
    return out


def runs_of(pos):
    """contiguous runs of label positions = one reply each (a tool response always sits between two replies)"""
    out, cur = [], []
    for n, p in enumerate(pos):
        if cur and p != pos[cur[-1]] + 1:
            out.append(cur); cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def selftest():
    class LP:
        def __init__(self, lp, rank):
            self.logprob, self.rank = lp, rank
    rng = np.random.default_rng(0)
    V, L = 30, 12
    ids = rng.integers(0, V, L).tolist()
    labels = [-100] * 6 + ids[6:]
    plp = [None]
    for j in range(1, L):
        lp = np.log(rng.dirichlet(np.ones(V)))
        order = np.argsort(-lp)
        rank = {int(t): r + 1 for r, t in enumerate(order)}
        d = {int(t): LP(float(lp[t]), rank[int(t)]) for t in order[:5]}
        d[ids[j]] = LP(float(lp[ids[j]]), rank[ids[j]])
        plp.append((d, lp))
    out = align({"input_ids": ids, "labels": labels}, [None] + [p[0] for p in plp[1:]], 5)
    assert out["pos"].tolist() == list(range(6, L))
    for n, j in enumerate(out["pos"]):
        lp = plp[j][1]
        assert abs(out["tok_lp"][n] - lp[ids[j]]) < 1e-6
        assert np.allclose(np.sort(out["t_lp"][n])[::-1], np.sort(lp)[::-1][:5], atol=1e-6)
    toks = ['<|channel>', 'thought', '\n', 'ok', '<channel|>', '<|tool_call>', 'call', ':', 'read', '{', 'a', '}', '<tool_call|>', '<|tool_response>']
    assert span_kinds(toks[1:]) == ['thought'] * 4 + ['name'] * 5 + ['args'] * 3 + ['text']
    assert runs_of([3, 4, 5, 9, 10, 20]) == [[0, 1, 2], [3, 4], [5]]
    f = fake({"input_ids": ids, "labels": labels}, 5, "other")
    assert (f["t_ids"][:, 1] == np.array(ids[6:])).all() and (f["tok_rank"] == 2).all()
    print("teacher_score selftest PASS (label position j ↔ prompt_logprobs[j]; top-k sorted by rank; sampled-token logprob; "
          "span kinds; reply runs; fake arrays)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--model", default="google/gemma-4-31B-it-qat-w4a16-ct")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--mbt", type=int, default=4096, help="max_num_batched_tokens (prompt-logprob logits memory)")
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=8, help="rows per generate call")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tokenizer", default=None, help="for the per-span table (default: the model's)")
    ap.add_argument("--fake", choices=["same", "other"], default=None, help="no engine: test arrays (laptop)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rows = [(i, json.loads(l)) for i, l in enumerate(open(a.rows)) if i % a.nshards == a.shard]
    long_ = [i for i, r in rows if len(r["input_ids"]) > a.max_len]
    rows = [(i, r) for i, r in rows if len(r["input_ids"]) <= a.max_len]
    n_tok = sum(len(r["input_ids"]) for _, r in rows); n_lab = sum(1 for _, r in rows for x in r["labels"] if x != -100)
    print(f"teacher_score · {a.rows} shard {a.shard}/{a.nshards} · rows {len(rows)} ({n_tok:,} tokens, {n_lab:,} student reply tokens) "
          f"· over max_len {a.max_len}: {len(long_)} left out · model {a.fake and f'FAKE ({a.fake})' or a.model} · top-k {a.k} · "
          f"max_num_batched_tokens {a.mbt} · gpu_util {a.gpu_util} · batch {a.batch} rows · prefix-cache reads off (prompt_logprobs)",
          flush=True)
    t0 = time.time()
    arrays = {}
    if a.fake:
        for i, r in rows:
            for key, v in fake(r, a.k, a.fake).items():
                arrays[f"{i}/{key}"] = v
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, max_num_batched_tokens=a.mbt,
                  max_logprobs=a.k, enable_prefix_caching=False)
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=a.k, detokenize=False)
        t0 = time.time(); done = 0
        for b in range(0, len(rows), a.batch):
            chunk = rows[b:b + a.batch]
            outs = llm.generate([{"prompt_token_ids": r["input_ids"]} for _, r in chunk], sp, use_tqdm=False)
            for (i, r), o in zip(chunk, outs):
                for key, v in align(r, o.prompt_logprobs, a.k).items():
                    arrays[f"{i}/{key}"] = v
            del outs
            done += sum(len(r["input_ids"]) for _, r in chunk)
            dt = time.time() - t0
            print(f"  rows {min(b + a.batch, len(rows))}/{len(rows)} · {done:,} tokens · {done / max(dt, 1e-9):,.0f} tok/s · "
                  f"eta {(n_tok - done) / max(done / max(dt, 1e-9), 1e-9) / 60:.0f} min", flush=True)
    dt = time.time() - t0
    from collections import defaultdict
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    agree1 = agreek = 0; nll = 0.0
    by = defaultdict(lambda: [0, 0.0, 0])                           # kind → [n, nll, top1]
    for i, r in rows:
        pos, tl, tr = arrays[f"{i}/pos"].tolist(), arrays[f"{i}/tok_lp"], arrays[f"{i}/tok_rank"]
        agree1 += int((tr == 1).sum()); agreek += int((tr <= a.k).sum()); nll -= float(tl.sum())
        for run in runs_of(pos):
            kinds = span_kinds(tok.convert_ids_to_tokens([r["input_ids"][pos[n]] for n in run]))
            for n, kd in zip(run, kinds):
                bb = by[kd]; bb[0] += 1; bb[1] -= float(tl[n]); bb[2] += int(tr[n] == 1)
    np.savez_compressed(a.out, **arrays)
    print("  per span: " + " · ".join(f"{kd} {bb[0]:,} tok NLL {bb[1] / max(bb[0], 1):.3f} top-1 {100 * bb[2] / max(bb[0], 1):.0f}%"
                                      for kd, bb in sorted(by.items())))
    print(f"== teacher_score shard {a.shard}/{a.nshards}: {len(rows)} rows · {n_tok / max(dt, 1e-9):,.0f} tok/s ({dt:.0f} s) · teacher NLL of "
          f"student tokens {nll / max(n_lab, 1):.3f}/tok · student token = teacher top-1 {100 * agree1 / max(n_lab, 1):.1f}% · "
          f"in top-{a.k} {100 * agreek / max(n_lab, 1):.1f}% → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
