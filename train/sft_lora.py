#!/usr/bin/env python3
"""LoRA SFT on rendered call-level samples (train/render.py jsonl), bf16 base from train/dequant_ct.py.

    .venv-train/bin/python train/sft_lora.py --base /root/dequant/gemma-4-E4B-dq --data runs/sft/d3_relabel_31b.rendered.jsonl \
        --holdout 8 --epochs 2 --lr 3e-5 --accum 16 --out adapters/e4b_m2
    .venv-train/bin/python train/sft_lora.py --data … --dry-run        # data + schedule only, no model (laptop)

What the canary taught (05_run_log 2026-10-29): lr 1e-4 · batch 1 · no warmup on no-thought targets collapsed the
policy in 20 steps. Defaults here: lr 3e-5, 5% linear warmup then cosine to 10%, 16 sequences per optimizer step
(token-mean loss over the whole window), 2 epochs, grad clip 1.0, r32 α32 dropout 0 on the 258 language-model
projections present in the checkpoint, loss only at label positions (`logits_to_keep` index tensor: the 262K-vocab
logits of a whole sequence do not fit). Every optimizer step → trainer_log.jsonl {step, epoch, loss, lr, tokens,
label_tokens, seconds, tokens_per_s, max_mem_gb}; holdout NLL/token every --eval-every steps and per epoch; one
adapter per epoch (`<out>/epoch<k>/`, scorer schema, tensor names/shapes checked) + `<out>/train_summary.json`.
Every knob prints its effective value.

Several cards for ONE adapter (--world K --rank i, one process per card, started by train/drills_box.sh train with
GPUS=a,b,…): every rank holds the model and the same schedule; the 16 sequences of an optimizer step are dealt to the
ranks by length (longest first to the least loaded rank), each rank backpropagates its own with the same
normalisation (the step's label tokens), the LoRA gradients are summed over the ranks (torch.distributed, gloo over
localhost) and rank 0's sum is sent to all, so every rank clips and steps on identical numbers. The update is the one a
single card computes, up to float summation order. The held-out eval is dealt out the same way. Rank 0 logs and
exports. --mock: a tiny random model on the CPU (no checkpoint, no export) to test the schedule and the multi-rank
path on a laptop: `--mock --world 1` and `--mock --world 4` must print the same losses and parameter checksum.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_export import ct_modules, export_and_check  # noqa: E402


def load_rows(paths, max_len):
    rows, dropped_len, dropped_empty = [], 0, 0
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if not any(x != -100 for x in r["labels"]):
                dropped_empty += 1; continue
            if len(r["input_ids"]) > max_len:
                dropped_len += 1; continue
            rows.append({"input_ids": r["input_ids"], "labels": r["labels"], "id": r.get("instance_id"), "calls": r.get("calls")})
    return rows, dropped_len, dropped_empty


def lr_at(step, total, warmup, lr, floor=0.1):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--ct", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--alias", default="gemma-4-e4b-it-qat-w4a16-ct")
    ap.add_argument("--data", nargs="+", required=True, help="render.py jsonl file(s)")
    ap.add_argument("--eval", nargs="*", default=[], help="held-out jsonl (else --holdout sequences are split off)")
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=26624)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of optimizer steps")
    ap.add_argument("--accum", type=int, default=16, help="sequences per optimizer step")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--world", type=int, default=1, help="processes (cards) training this one adapter")
    ap.add_argument("--rank", type=int, default=0, help="this process's index, 0 … world-1 (rank 0 logs and exports)")
    ap.add_argument("--dist-port", type=int, default=29500, help="localhost port the ranks meet on")
    ap.add_argument("--mock", action="store_true", help="tiny random model on the CPU: tests the schedule / multi-rank path")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    rows, d_len, d_empty = load_rows(a.data, a.max_len)
    if a.eval:
        ev, _, _ = load_rows(a.eval, a.max_len)
    else:
        rnd.shuffle(rows)
        ev, rows = rows[: a.holdout], rows[a.holdout:]
    n_tok = sum(len(r["input_ids"]) for r in rows); n_lab = sum(1 for r in rows for x in r["labels"] if x != -100)
    steps_per_epoch = math.ceil(len(rows) / a.accum)
    n_total = steps_per_epoch * a.epochs if not a.max_steps else min(a.max_steps, steps_per_epoch * a.epochs)
    warm = max(1, round(a.warmup * n_total))
    main_rank = a.rank == 0
    say = print if main_rank else (lambda *x, **k: None)
    print(f"sft_lora rank {a.rank} of {a.world} · dist port {a.dist_port} · mock {a.mock} · "
          f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    say(f"sft_lora · data {a.data} → train {len(rows)} sequences ({n_tok:,} tokens, {n_lab:,} label tokens = {100 * n_lab / max(n_tok, 1):.1f}%) · "
          f"holdout {len(ev)} ({'file' if a.eval else 'split off'}) · dropped: > {a.max_len} tokens {d_len}, no labels {d_empty}")
    say(f"  schedule: {a.epochs} epochs × {steps_per_epoch} optimizer steps (accum {a.accum}) = {n_total} steps · lr {a.lr} warmup {warm} steps "
          f"then cosine → {a.lr * 0.1:.1e} · clip {a.clip} · r {a.r} α {a.alpha} dropout 0 · seed {a.seed} · eval every {a.eval_every} · out {a.out} · "
          f"cards for this adapter {a.world} (each step's {a.accum} sequences dealt to them by length; gradients summed)")
    if a.dry_run:
        print("  dry run: no model loaded"); return 0
    import torch
    torch.manual_seed(a.seed)
    t0 = time.time()
    dev = "cpu" if a.mock else "cuda"
    sync = (lambda: None) if a.mock else torch.cuda.synchronize
    mem = (lambda: 0.0) if a.mock else (lambda: torch.cuda.max_memory_allocated() / 1e9)
    V = 997                                                  # --mock: token ids are folded into this vocabulary
    if a.mock:
        class Mock(torch.nn.Module):                         # frozen embedding + head, one low-rank trainable pair
            def __init__(self):
                super().__init__()
                g = torch.Generator().manual_seed(a.seed)
                self.emb = torch.nn.Parameter(torch.randn(V, 32, generator=g), requires_grad=False)
                self.head = torch.nn.Parameter(torch.randn(32, V, generator=g) * 0.1, requires_grad=False)
                self.A = torch.nn.Parameter(torch.randn(32, 8, generator=g) * 0.1)
                self.B = torch.nn.Parameter(torch.zeros(8, V))

            def forward(self, input_ids, logits_to_keep):
                h = self.emb[input_ids[0] % V][logits_to_keep]
                return type("O", (), {"logits": (h @ self.head + h @ self.A @ self.B)[None]})
        model, targets = Mock(), []
    else:
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText
        cls = getattr(transformers, "Gemma4ForConditionalGeneration", AutoModelForImageTextToText)
        targets = ct_modules(a.ct)
        model = cls.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=targets))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"  model {'mock' if a.mock else a.base} · {len(targets)} LoRA modules · trainable {n_train / 1e6:.1f} M · loaded {time.time() - t0:.0f} s · "
        f"mem {0.0 if a.mock else torch.cuda.memory_allocated() / 1e9:.1f} GB")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    td = None
    if a.world > 1:                                          # the ranks meet once every model is loaded
        import datetime
        import torch.distributed as td
        td.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.dist_port}", rank=a.rank, world_size=a.world,
                              timeout=datetime.timedelta(minutes=120))
        chk = torch.tensor([float(len(rows)), float(n_lab), float(len(ev)), float(n_train)], dtype=torch.float64)
        ref = chk.clone(); td.broadcast(ref, src=0)
        if not torch.equal(chk, ref):
            sys.exit(f"rank {a.rank}: data or model differs from rank 0's ({chk.tolist()} vs {ref.tolist()})")
        say(f"  {a.world} ranks joined (gloo, 127.0.0.1:{a.dist_port}) · same data and trainable size on every rank")

    def deal(idx):                                           # a step's sequences → this rank's share, balanced by length
        if a.world == 1:
            return list(idx)
        load, mine = [0] * a.world, []
        for i in sorted(idx, key=lambda i: (-len(rows[i]["input_ids"]), i)):
            k = min(range(a.world), key=lambda k: (load[k], k))
            load[k] += len(rows[i]["input_ids"])
            if k == a.rank:
                mine.append(i)
        return mine

    def share_grads():                                       # sum the ranks' gradients; every rank ends with rank 0's sum
        if td is None:
            return 0.0
        ts = time.time()
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu() for p in params])
        td.all_reduce(flat, op=td.ReduceOp.SUM); td.broadcast(flat, src=0)
        o = 0
        for p in params:
            n = p.numel()
            p.grad = flat[o:o + n].view_as(p).to(device=p.device, dtype=p.dtype); o += n
        return time.time() - ts

    def total(*v):                                           # sums over the ranks (float64)
        t = torch.tensor(v, dtype=torch.float64)
        if td is not None:
            td.all_reduce(t, op=td.ReduceOp.SUM)
        return t.tolist()

    if main_rank:
        os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "trainer_log.jsonl"), "w") if main_rank else open(os.devnull, "w")

    def loss_sum(r):
        ids = torch.tensor([r["input_ids"]], device=dev)
        labels = torch.tensor([r["labels"]], device=dev)
        if a.mock:
            labels = torch.where(labels == -100, labels, labels % V)
        tgt = labels[0, 1:]
        keep = (tgt != -100).nonzero(as_tuple=True)[0]
        out = model(input_ids=ids, logits_to_keep=keep)
        return torch.nn.functional.cross_entropy(out.logits[0].float(), tgt[keep], reduction="sum"), len(keep), ids.shape[1]

    def evaluate(tag):
        if not ev:
            return None
        model.eval(); s = n = 0
        with torch.no_grad():
            for r in ev[a.rank::a.world]:                    # the held-out sequences are dealt to the ranks too
                l, k, _ = loss_sum(r); s += l.item(); n += k
        model.train()
        s, n = total(s, n); n = int(n)
        v = s / max(n, 1)
        say(f"  eval[{tag}] holdout nll/tok {v:.4f} over {n} label tokens ({len(ev)} sequences)")
        log.write(json.dumps({"eval": tag, "holdout_nll": v, "label_tokens": n}) + "\n"); log.flush()
        return v

    model.train()
    step, hist, summary = 0, [], {"epochs": []}
    evaluate("start")
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(rows))); rnd.shuffle(order)
        windows = [order[i:i + a.accum] for i in range(0, len(order), a.accum)]
        e_t0, e_tok, e_loss = time.time(), 0, []
        for w in windows:
            if a.max_steps and step >= a.max_steps:
                break
            lr = lr_at(step, n_total, warm, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            w_lab = sum(1 for i in w for x in rows[i]["labels"] if x != -100)
            sync(); ts = time.time(); w_tok = 0; w_loss = 0.0
            for i in deal(w):
                l, k, n = loss_sum(rows[i])
                (l / w_lab).backward()
                w_loss += l.item(); w_tok += n
            sync(); t_own = time.time() - ts
            t_sync = share_grads()
            w_loss, w_tok, t_all = total(w_loss, w_tok, t_own); w_tok = int(w_tok)
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip).item()
            opt.step(); opt.zero_grad(set_to_none=True)
            sync(); dt = time.time() - ts
            step += 1; e_tok += w_tok; e_loss.append(w_loss / w_lab)
            rec = {"step": step, "epoch": epoch, "loss": round(w_loss / w_lab, 4), "lr": lr, "grad_norm": round(gn, 3),
                   "tokens": w_tok, "label_tokens": w_lab, "seconds": round(dt, 1), "tokens_per_s": round(w_tok / dt),
                   "max_mem_gb": round(mem(), 2), "ranks": a.world, "sync_s": round(t_sync, 2),
                   "speedup": round(t_all / dt, 2)}
            log.write(json.dumps(rec) + "\n"); log.flush(); hist.append(rec["loss"])
            if step <= 3 or step % 5 == 0:
                eta = (n_total - step) * dt / 60
                say(f"  step {step:>4}/{n_total} ep {epoch} loss {rec['loss']:.4f} lr {lr:.2e} gn {gn:.2f} · {w_tok} tok ({w_lab} labels) · "
                    f"{dt:.0f} s · {rec['tokens_per_s']} tok/s · mem {rec['max_mem_gb']} GB · eta {eta:.0f} min"
                    + (f" · {a.world} cards: ×{rec['speedup']} of one card, gradient sync {t_sync:.1f} s" if a.world > 1 else ""))
            if a.eval_every and step % a.eval_every == 0:
                evaluate(f"step{step}")
        ev_v = evaluate(f"epoch{epoch}")
        out_dir = os.path.join(a.out, f"epoch{epoch}")
        n_t, size, problems = export_and_check(model, out_dir, a.alias, a.r, a.alpha, targets) if main_rank and not a.mock else (0, 0, [])
        ep = {"epoch": epoch, "steps": step, "train_loss_mean": sum(e_loss) / max(len(e_loss), 1),
              "train_loss_last5": sum(e_loss[-5:]) / max(len(e_loss[-5:]), 1), "holdout_nll": ev_v,
              "tokens": e_tok, "seconds": round(time.time() - e_t0), "adapter": out_dir, "tensors": n_t,
              "adapter_mb": round(size / 1e6), "export_problems": problems}
        summary["epochs"].append(ep)
        say(f"== epoch {epoch}: {len(windows)} steps · loss mean {ep['train_loss_mean']:.4f} (last 5 {ep['train_loss_last5']:.4f}) · "
            f"holdout {ev_v if ev_v is None else round(ev_v, 4)} · {e_tok:,} tokens in {ep['seconds']} s ({e_tok / max(ep['seconds'], 1):.0f} tok/s) · "
            f"adapter {out_dir} {n_t} tensors {ep['adapter_mb']} MB · export check {'PASS' if not problems else 'FAIL ' + problems[0]}")
        if a.max_steps and step >= a.max_steps:
            break
    checksum = sum(p.detach().double().abs().sum().item() for p in params)      # equal on every rank and for any --world
    summary.update({"args": vars(a), "targets": len(targets), "trainable": n_train, "total_steps": step,
                    "wall_s": round(time.time() - t0), "peak_mem_gb": round(mem(), 2), "ranks": a.world,
                    "param_checksum": checksum})
    if td is not None:                                       # every rank stepped on the same gradients: same parameters
        lo, hi = torch.tensor([checksum], dtype=torch.float64), torch.tensor([checksum], dtype=torch.float64)
        td.all_reduce(lo, op=td.ReduceOp.MIN); td.all_reduce(hi, op=td.ReduceOp.MAX)
        spread = (hi.item() - lo.item()) / max(1e-12, abs(checksum))
        say(f"  parameter checksum over the {a.world} ranks: relative spread {spread:.1e}"
            + ("" if spread < 1e-6 else "   ← WARNING: the ranks' parameters drifted apart; rank 0's are the exported ones"))
        td.destroy_process_group()
    if not main_rank:
        print(f"sft_lora rank {a.rank} done: {step} steps · {round(time.time() - t0)} s · parameter checksum {checksum:.6f}")
        return 0
    json.dump(summary, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print(f"sft_lora done: {step} steps · {summary['wall_s']} s · peak {summary['peak_mem_gb']} GB · loss first {hist[0]:.3f} → last {hist[-1]:.3f} · "
          f"{a.world} card(s) · parameter checksum {checksum:.6f} (the same on every rank) → {a.out}/train_summary.json")
    return 0 if all(not e["export_problems"] for e in summary["epochs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
