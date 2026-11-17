#!/usr/bin/env python3
"""Held-out NLL of teacher trajectories, split by what the token is (B-9 part 3 step 19).

  .venv-train/bin/python train/nll_split.py --base Qwen/Qwen3.5-9B --data runs/sft/mc/r1k_nt.qwen3.5-9b.holdout.rendered.jsonl \\
      [--adapter adapters/mc_Qwen3.5-9B_r1k_nt_lr1e-5/epoch2] --name full --out runs/nll_split/full.jsonl

The training log gives one held-out number over all label tokens. This splits it: a reply (a run of label tokens in a
rendered sequence; the file must be rendered with label scope `reply`) is cut at its first thought-close token into
  thought first 16 tokens · thought rest · the close token · action (text, tool calls, end of turn),
and a reply without a close token is its own bucket. Per bucket: tokens, NLL per token, the model's mean probability of
the right token, and how often the right token is the model's first choice. One forward pass per sequence, no
training; the same hidden-state → lm_head path as sft_lora.py, so `all label tokens` must reproduce the trainer's
held-out NLL of that model (printed for the check). --out: one line per reply (lengths and summed NLL per part).
Adapters: the leaf-name export of sft_lora.py --target-leaves (Qwen). --mock: random logits on the CPU, tests the split.
"""
import argparse, collections, json, math, os, re, sys, time

BUCKETS = ["thought first 16", "thought rest", "thought close", "action", "reply without a thought close"]


def replies(labels):
    """→ [(start, end)] runs of label positions"""
    out, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] == -100:
            i += 1; continue
        j = i
        while j < n and labels[j] != -100:
            j += 1
        out.append((i, j)); i = j
    return out


def bucket_of(ids, runs, close_id):
    """position → bucket index, and per reply (start, end, close position or None)"""
    b, info = {}, []
    for s, e in runs:
        c = next((p for p in range(s, e) if ids[p] == close_id), None)
        info.append((s, e, c))
        for p in range(s, e):
            if c is None: b[p] = 4
            elif p < c: b[p] = 0 if p - s < 16 else 1
            elif p == c: b[p] = 2
            else: b[p] = 3
    return b, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B"); ap.add_argument("--adapter", default="")
    ap.add_argument("--data", required=True); ap.add_argument("--tokenizer", default="", help="default: --base")
    ap.add_argument("--think-close", default="</think>"); ap.add_argument("--name", default="")
    ap.add_argument("--out", default=""); ap.add_argument("--max-len", type=int, default=30720)
    ap.add_argument("--limit", type=int, default=0, help="first N sequences (0 = all)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    name = a.name or (os.path.basename(os.path.dirname(a.adapter.rstrip("/"))) + "/" + os.path.basename(a.adapter.rstrip("/")) if a.adapter else "base")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.base)
    cid = tok.encode(a.think_close, add_special_tokens=False)
    if len(cid) != 1:
        sys.exit(f"--think-close {a.think_close!r} is {len(cid)} tokens, not 1")
    cid = cid[0]
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if len(r["input_ids"]) <= a.max_len and any(x != -100 for x in r["labels"])]
    rows = rows[:a.limit] if a.limit else rows
    print(f"nll_split · model {name} · base {a.base} · adapter {a.adapter or 'none'} · data {a.data} · {len(rows)} sequences (max len {a.max_len}"
          f"{', first ' + str(a.limit) if a.limit else ''}) · thought close {a.think_close!r} = id {cid} · mock {a.mock}", flush=True)
    import torch
    t0 = time.time()
    if a.mock:
        V = 997; g = torch.Generator().manual_seed(0); emb = torch.randn(V, 16, generator=g); head = torch.randn(V, 16, generator=g)
        hidden = lambda ids: emb[torch.tensor(ids) % V]
        logits = lambda h: h @ head.T
        fold = lambda t: t % V
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        if a.adapter:
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file
            cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
            leaves = set(cfg["target_modules"]) if isinstance(cfg["target_modules"], list) else sys.exit("adapter_config target_modules is not a list of leaf names")
            targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model" in n.split(".") and n.split(".")[-1] in leaves]
            model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
            sd = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
            own = dict(model.named_parameters()); hit = 0; norm = 0.0
            with torch.no_grad():
                for k, v in sd.items():
                    kk = re.sub(r"\.lora_([AB])\.weight$", r".lora_\1.default.weight", k)
                    if kk not in own or own[kk].shape != v.shape:
                        sys.exit(f"adapter tensor {k} has no place in the model ({kk})")
                    own[kk].copy_(v.to(own[kk].device, own[kk].dtype)); hit += 1
                    if ".lora_B." in k: norm += float(v.float().pow(2).sum())
            if hit != 2 * len(targets):
                sys.exit(f"adapter has {hit} tensors, the model has {len(targets)} target modules (expected {2 * len(targets)})")
            print(f"  adapter loaded: r {cfg['r']} alpha {cfg['lora_alpha']} · {len(targets)} modules · {hit} tensors placed · lora_B norm {math.sqrt(norm):.3f} (0 = an untrained adapter)", flush=True)
            base = model.get_base_model()
        else:
            base = model
        model.eval()
        cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
        hidden = lambda ids: base.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0]

        def logits(h):
            lg = base.lm_head(h)
            return (torch.tanh(lg / cap) * cap if cap is not None else lg).float()
        fold = lambda t: t
        print(f"  loaded {time.time() - t0:.0f} s · mem {torch.cuda.memory_allocated() / 1e9:.1f} GB · final logit softcapping {cap}", flush=True)
    S = [[0, 0.0, 0.0, 0] for _ in BUCKETS]      # tokens, nll sum, prob sum, top-1 hits
    if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = open(a.out, "w") if a.out else None
    n_rep = 0; tl = []
    with torch.no_grad():
        for qi, r in enumerate(rows):
            ids, lab = r["input_ids"], r["labels"]
            runs = replies(lab); b, info = bucket_of(ids, runs, cid)
            pos = sorted(p for p in b if p > 0)
            hs = hidden(ids)
            nll = {}
            for i in range(0, len(pos), 1024):
                pp = pos[i:i + 1024]
                lg = logits(hs[torch.tensor([p - 1 for p in pp], device=hs.device)])
                tg = fold(torch.tensor([ids[p] for p in pp], device=lg.device))
                lp = torch.log_softmax(lg, -1); x = -lp.gather(1, tg[:, None])[:, 0]; top = lg.argmax(-1) == tg
                for p, v, t in zip(pp, x.tolist(), top.tolist()):
                    nll[p] = v; k = b[p]; S[k][0] += 1; S[k][1] += v; S[k][2] += math.exp(-v); S[k][3] += t
            for s, e, c in info:
                n_rep += 1
                th = [nll[p] for p in range(s, c)] if c is not None else []
                ac = [nll[p] for p in range((c + 1) if c is not None else s, e) if p in nll]
                tl.append(len(th))
                if out:
                    out.write(json.dumps({"model": name, "seq": qi, "id": r.get("instance_id"), "start": s, "thought_tokens": len(th), "thought_nll": round(sum(th), 4),
                                          "thought_first16_nll": round(sum(th[:16]), 4), "close_nll": round(nll[c], 4) if c is not None else None,
                                          "action_tokens": len(ac), "action_nll": round(sum(ac), 4)}) + "\n")
            if (qi + 1) % 4 == 0 or qi + 1 == len(rows):
                print(f"  {qi + 1}/{len(rows)} sequences · {time.time() - t0:.0f} s", flush=True)
    if out: out.close()
    tot = sum(x[0] for x in S); tnll = sum(x[1] for x in S); tl.sort()
    print(f"== nll split {name}: {len(rows)} sequences · {n_rep} replies · thought tokens per reply p50 {tl[len(tl) // 2] if tl else 0} max {tl[-1] if tl else 0}")
    print(f"   {'bucket':30} {'tokens':>9} {'share':>7} {'nll/tok':>8} {'mean p':>7} {'top-1':>7}")
    for nm, (n, s, p, t) in zip(BUCKETS, S):
        print(f"   {nm:30} {n:>9} {n / max(tot, 1):>7.1%} {s / max(n, 1):>8.4f} {p / max(n, 1):>7.3f} {t / max(n, 1):>7.1%}")
    th = [S[0][i] + S[1][i] for i in range(4)]
    print(f"   {'thought (first 16 + rest)':30} {th[0]:>9} {th[0] / max(tot, 1):>7.1%} {th[1] / max(th[0], 1):>8.4f} {th[2] / max(th[0], 1):>7.3f} {th[3] / max(th[0], 1):>7.1%}")
    print(f"   {'all label tokens':30} {tot:>9} {'100.0%':>7} {tnll / max(tot, 1):>8.4f}   ← must equal this model's held-out NLL in the training log")
    print("📷 the header, the adapter line and the == block")


if __name__ == "__main__":
    main()
