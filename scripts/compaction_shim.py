"""Give `swegemma eval` the scorer's events compaction (box/laptop runs; COMPACTION=0 turns it off).

swegemma 0.2.11's CLI builds EvalConfig without `events_compaction_config` (default None), so every local / box run so
far had NO compaction: the history only grew until the 32K window. The scorer's inference.py sets it (HARNESS_README
§7.2, Kaggle copy of 2026-10-13): compaction_interval 15 · overlap_size 2 · token_threshold 14,336 · event_retention_size 5.
This module wraps EvalConfig.__init__ and fills that config in when the caller left it None; agent_runner.py:694 then
passes it to the ADK App. ADK builds the summarizer from the root agent's own model (same Gemma, same vLLM server, the
default "summarize the conversation" prompt; the request carries the thinking settings but no temperature / top_p / top_k,
forum #753225 — same as on the scorer).

Knobs (printed when the eval builds its EvalConfig): COMPACTION=0 off · COMPACT_INTERVAL · COMPACT_OVERLAP · COMPACT_THRESHOLD · COMPACT_RETAIN.
Every summary call prints one line: `compaction: #n summarised k events → c chars (s s)`.
Loaded by scripts/eval_sitecustomize.py.
"""
import functools
import os
import time

ON = os.environ.get("COMPACTION", "1") != "0"
VALUES = {
    "compaction_interval": int(os.environ.get("COMPACT_INTERVAL", 15)),
    "overlap_size": int(os.environ.get("COMPACT_OVERLAP", 2)),
    "token_threshold": int(os.environ.get("COMPACT_THRESHOLD", 14336)),
    "event_retention_size": int(os.environ.get("COMPACT_RETAIN", 5)),
}
_state = {"n": 0}


def describe():
    if not ON:
        return "compaction: OFF (COMPACTION=0) — history only grows, unlike the scorer"
    return ("compaction: on, the scorer's EventsCompactionConfig — " +
            " · ".join(f"{k} {v:,}" for k, v in VALUES.items()) +
            " (summarizer = the root agent's model, ADK default prompt)")


def install():
    try:
        from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
        from swegemma import config as sc
        try:
            from google.adk.apps._configs import EventsCompactionConfig
        except ImportError:
            from google.adk.apps.app import EventsCompactionConfig
    except Exception:                        # not the eval venv (system python3 in eval_dev.sh): do nothing
        return

    orig_init = sc.EvalConfig.__init__

    @functools.wraps(orig_init)
    def init(self, *a, **kw):                # printed here, not at import: only a real eval builds an EvalConfig
        orig_init(self, *a, **kw)
        if ON and self.events_compaction_config is None:
            self.events_compaction_config = EventsCompactionConfig(**VALUES)
        print(describe() if ON or self.events_compaction_config is None else
              f"compaction: caller's own config kept ({self.events_compaction_config})", flush=True)

    sc.EvalConfig.__init__ = init

    orig_sum = LlmEventSummarizer.maybe_summarize_events

    @functools.wraps(orig_sum)
    async def summarize(self, *, events):
        t0 = time.time()
        ev = await orig_sum(self, events=events)
        _state["n"] += 1
        parts = (ev.actions.compaction.compacted_content.parts or []) if ev and ev.actions and ev.actions.compaction else []
        chars = sum(len(p.text or "") for p in parts if not getattr(p, "thought", False))
        print(f"compaction: #{_state['n']} summarised {len(events)} events → {chars:,} chars "
              f"({time.time() - t0:.1f} s){'' if ev else ' — no summary returned'}", flush=True)
        return ev

    LlmEventSummarizer.maybe_summarize_events = summarize


install()
