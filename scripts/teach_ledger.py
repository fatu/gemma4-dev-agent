#!/usr/bin/env python3
"""One spend ledger for every worker of a teacher run (B-11 step 1): a budget that holds under concurrency.

    python3 scripts/teach_ledger.py status  --dir runs/teach/ledger
    python3 scripts/teach_ledger.py init    --dir runs/teach/ledger --budget 50 --price 0.6,0.11,2.2
    python3 scripts/teach_ledger.py selftest

A directory holds three files, all written under one exclusive lock (`lock`, fcntl.flock), so any number of processes
and threads can use it at once:
  state.json    {budget_usd, price [in, cached_in, out $ per 1M tokens] or null, spent_usd, reserved_usd,
                 reservations {key: usd}, calls, estimated_calls}
  calls.jsonl   one line per model call: key (task + attempt), tokens in / cached in / out / reasoning, usd, whether
                usd is the provider's own figure or an estimate from `price`, seconds, error
  events.jsonl  reserve / release / gate lines (who was refused and why)

Rules (the three the plan names):
  reserve(key, usd)  before an attempt starts. Refused when spent + reserved + usd > budget: no new attempt is started.
  gate()             before every model call. Raises BudgetReached when spent ≥ budget — the reservation was too low,
                     the running attempts stop at their next call.
  charge(key, …)     after every call (failed calls too: their tokens are billed all the same when the provider
                     reports them). release(key) when the attempt ends: the reservation is dropped, what was spent stays.

Cost of a call: the provider's own figure when the response carries one (`usage.cost`, or litellm's
`_hidden_params.response_cost`); otherwise price × tokens, counted in `estimated_calls` and marked `est: true`.
With no figure and no price the call costs 0 and is counted in `unpriced_calls` — status prints that loudly.
Nothing here knows a gateway address, a key or a model name.
"""
import argparse
import contextlib
import fcntl
import json
import os
import sys
import time


class BudgetReached(RuntimeError):
    pass


class Ledger:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        with open(os.path.join(self.d, "lock"), "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                p = os.path.join(self.d, "state.json")
                st = json.load(open(p)) if os.path.exists(p) else {
                    "budget_usd": None, "price": None, "spent_usd": 0.0, "reserved_usd": 0.0, "reservations": {},
                    "calls": 0, "estimated_calls": 0, "unpriced_calls": 0}
                yield st
                tmp = p + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(st, f)
                os.replace(tmp, p)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _append(self, name, rec):
        with open(os.path.join(self.d, name), "a") as f:      # called inside _locked
            f.write(json.dumps(rec) + "\n")

    def init(self, budget_usd, price=None):
        with self._locked() as st:
            st["budget_usd"] = float(budget_usd)
            if price is not None:
                st["price"] = [float(x) for x in price]
            self._append("events.jsonl", {"t": round(time.time(), 1), "ev": "init", "budget_usd": st["budget_usd"], "price": st["price"]})
            return dict(st)

    def status(self):
        with self._locked() as st:
            return json.loads(json.dumps(st))

    def reserve(self, key, usd):
        """→ True and the amount is held, or False (budget would be passed; nothing changes)"""
        with self._locked() as st:
            b = st["budget_usd"]
            held = st["reservations"].get(key, 0.0)
            ok = b is None or st["spent_usd"] + st["reserved_usd"] - held + usd <= b + 1e-9
            if ok:
                st["reserved_usd"] += usd - held; st["reservations"][key] = usd
            self._append("events.jsonl", {"t": round(time.time(), 1), "ev": "reserve" if ok else "reserve_refused", "key": key,
                                          "usd": usd, "spent": round(st["spent_usd"], 6), "reserved": round(st["reserved_usd"], 6), "budget": b})
            return ok

    def release(self, key):
        with self._locked() as st:
            held = st["reservations"].pop(key, None)
            if held is not None:
                st["reserved_usd"] = max(0.0, st["reserved_usd"] - held)
                self._append("events.jsonl", {"t": round(time.time(), 1), "ev": "release", "key": key, "usd": held})

    def gate(self, key=None):
        with self._locked() as st:
            b = st["budget_usd"]
            if b is not None and st["spent_usd"] >= b - 1e-9:
                self._append("events.jsonl", {"t": round(time.time(), 1), "ev": "gate_closed", "key": key, "spent": round(st["spent_usd"], 6), "budget": b})
                raise BudgetReached(f"teach_ledger: budget reached (spent ${st['spent_usd']:.4f} of ${b:.2f}) — no further model call")

    def charge(self, key, tin=0, cached=0, tout=0, reasoning=0, usd=None, seconds=None, err=None, extra=None):
        """record one call. tin = all input tokens (cached ones included), cached = the part read from the cache.
        Spending draws down the key's reservation first, so a reserved attempt does not count twice."""
        with self._locked() as st:
            est = False
            if usd is None:
                if st["price"]:
                    pi, pc, po = st["price"]
                    usd = ((max(0, tin - cached)) * pi + cached * pc + tout * po) / 1e6; est = True
                else:
                    usd = 0.0; st["unpriced_calls"] += 1
            st["spent_usd"] += usd; st["calls"] += 1; st["estimated_calls"] += est
            held = st["reservations"].get(key)
            if held is not None:                       # the reservation shrinks by what was spent
                left = max(0.0, held - usd)
                st["reserved_usd"] = max(0.0, st["reserved_usd"] - (held - left)); st["reservations"][key] = left
            rec = {"t": round(time.time(), 1), "key": key, "in": tin, "cached": cached, "out": tout, "reasoning": reasoning,
                   "usd": round(usd, 8), "est": est, "s": seconds, "err": (str(err)[:160] if err else None)}
            if extra:
                rec.update(extra)
            self._append("calls.jsonl", rec)
            return usd

    def per_key(self):
        """→ {key: {calls, in, cached, out, reasoning, usd, est_calls, errors}} from calls.jsonl"""
        out = {}
        p = os.path.join(self.d, "calls.jsonl")
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                a = out.setdefault(r["key"], {"calls": 0, "in": 0, "cached": 0, "out": 0, "reasoning": 0, "usd": 0.0, "est_calls": 0, "errors": 0})
                a["calls"] += 1; a["usd"] += r["usd"]; a["est_calls"] += bool(r.get("est")); a["errors"] += bool(r.get("err"))
                for k in ("in", "cached", "out", "reasoning"):
                    a[k] += r.get(k) or 0
        return out


def usage_of(resp):
    """(tin, cached, tout, reasoning, provider usd or None) from a litellm response (or None)"""
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, 0, 0, None
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    ptd = d.get("prompt_tokens_details") if isinstance(d.get("prompt_tokens_details"), dict) else {}
    ctd = d.get("completion_tokens_details") if isinstance(d.get("completion_tokens_details"), dict) else {}
    cached = ptd.get("cached_tokens") or d.get("cache_read_input_tokens") or 0
    usd = d.get("cost")
    if usd is None:
        hp = getattr(resp, "_hidden_params", None) or {}
        usd = hp.get("response_cost") if isinstance(hp, dict) else None
    return (d.get("prompt_tokens") or 0, cached, d.get("completion_tokens") or 0, ctd.get("reasoning_tokens") or 0,
            float(usd) if isinstance(usd, (int, float)) and usd > 0 else None)


def fmt_status(st):
    b = st["budget_usd"]
    s = (f"teach_ledger · budget {'none' if b is None else f'${b:.2f}'} · spent ${st['spent_usd']:.4f} · reserved ${st['reserved_usd']:.4f} "
         f"({len(st['reservations'])} open) · calls {st['calls']} · priced by estimate {st['estimated_calls']} · "
         f"price $/M in,cached,out {st['price']}")
    if st["unpriced_calls"]:
        s += f" · ⚠ {st['unpriced_calls']} call(s) with NO cost figure and no price: spent is too low — set --price"
    return s


def _worker(args):
    d, wid, n, usd = args
    L = Ledger(d); done = refused = gated = 0
    for i in range(n):
        key = f"w{wid}-a{i}"
        if not L.reserve(key, usd * 3):          # an attempt = 3 calls of `usd` each
            refused += 1; continue
        try:
            for _ in range(3):
                L.gate(key); L.charge(key, tin=1000, cached=500, tout=100, usd=usd)
            done += 1
        except BudgetReached:
            gated += 1
        finally:
            L.release(key)
    return done, refused, gated


def selftest():
    import multiprocessing as mp
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="teach_ledger_")
    try:
        L = Ledger(d); L.init(3.0)
        with mp.Pool(8) as p:                     # 8 workers × 20 attempts × 3 calls × $0.05 = $24 wanted, $3 allowed
            res = p.map(_worker, [(d, w, 20, 0.05) for w in range(8)])
        st = L.status(); done = sum(r[0] for r in res); refused = sum(r[1] for r in res); gated = sum(r[2] for r in res)
        lines = sum(1 for _ in open(os.path.join(d, "calls.jsonl")))
        assert abs(st["spent_usd"] - lines * 0.05) < 1e-6, (st["spent_usd"], lines)       # no lost update
        assert st["spent_usd"] <= 3.0 + 1e-6, st["spent_usd"]                              # the budget held
        assert st["reserved_usd"] < 1e-6 and not st["reservations"], st                    # every reservation released
        assert done == 20 and refused == 140 and gated == 0, (done, refused, gated)        # exactly budget / attempt cost
        assert sum(v["calls"] for v in L.per_key().values()) == lines == st["calls"]
        # a reservation that is too low: the hard gate stops the calls
        d2 = tempfile.mkdtemp(prefix="teach_ledger_"); L2 = Ledger(d2); L2.init(0.10)
        assert L2.reserve("a", 0.01) and L2.reserve("b", 0.01)
        n = 0
        try:
            for _ in range(50):
                L2.gate("a"); L2.charge("a", usd=0.03); n += 1
        except BudgetReached:
            pass
        assert n == 4 and L2.status()["spent_usd"] >= 0.10 - 1e-9 and not L2.reserve("c", 0.001), (n, L2.status())
        # price-based estimate, and the unpriced warning
        d3 = tempfile.mkdtemp(prefix="teach_ledger_"); L3 = Ledger(d3); L3.init(10, [1.0, 0.1, 4.0])
        usd = L3.charge("k", tin=1_000_000, cached=400_000, tout=250_000)
        assert abs(usd - (0.6 + 0.04 + 1.0)) < 1e-9 and L3.status()["estimated_calls"] == 1, usd
        d4 = tempfile.mkdtemp(prefix="teach_ledger_"); L4 = Ledger(d4); L4.init(10); L4.charge("k", tin=10, tout=10)
        assert L4.status()["unpriced_calls"] == 1 and "NO cost figure" in fmt_status(L4.status())
        for x in (d2, d3, d4):
            shutil.rmtree(x)
        print(f"teach_ledger selftest: 8 processes, {lines} calls recorded, spent ${st['spent_usd']:.2f} of $3.00, "
              f"attempts started {done} / refused {refused} · hard gate stops at the budget · estimate and unpriced paths → all as expected")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "init", "selftest"])
    ap.add_argument("--dir", default="runs/teach/ledger")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--price", help="in,cached_in,out in $ per 1M tokens (used when the provider gives no cost)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    L = Ledger(a.dir)
    if a.cmd == "init":
        if a.budget is None:
            ap.error("init needs --budget")
        L.init(a.budget, [float(x) for x in a.price.split(",")] if a.price else None)
    print(fmt_status(L.status()))


if __name__ == "__main__":
    main()
