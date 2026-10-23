"""Ceiling runs (BOX_PLAN.md B-3f/B-3g): adapt the harness's vLLM-shaped requests to a remote Anthropic model.

Imported by scripts/eval_sitecustomize.py (the PYTHONPATH `sitecustomize` eval_dev.sh installs) when MODELS_YAML is set,
so it runs before the harness imports litellm. The harness's generation resolver (adk_submission) puts Gemma/vLLM-only fields into
`extra_body` (chat_template_kwargs.enable_thinking, thinking_token_budget); the Anthropic API rejects unknown fields,
and Claude Opus 5.5 takes `thinking: {type: adaptive}` instead of a token budget (temperature must then be 1; litellm
drops it). So for anthropic/* models this wrapper:
  - removes extra_body, sets thinking adaptive unless CEIL_NOTHINK=1 (the submission's include_thoughts/budget are
    Gemma knobs; the log line in eval_dev.sh says what was used)
  - appends one JSON line per call to CEIL_CALL_LOG (tokens in/out, cache reads, seconds) for the spend total
Everything else (model alias → path via --models-yaml, tools, messages, max_tokens) passes through unchanged.
"""
import json
import os
import time

try:
    import litellm
except Exception:                      # not the eval venv: do nothing
    litellm = None

if litellm is not None:
    litellm.drop_params = True
    _LOG = os.environ.get("CEIL_CALL_LOG")
    _NOTHINK = os.environ.get("CEIL_NOTHINK") == "1"
    _orig_acompletion = litellm.acompletion
    _orig_completion = litellm.completion

    _CC = {"type": "ephemeral"}

    def _mark(msg):
        """put cache_control on a message's last content block (string content becomes one text block)"""
        c = msg.get("content")
        if isinstance(c, str):
            if c.strip():
                msg["content"] = [{"type": "text", "text": c, "cache_control": dict(_CC)}]
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            c[-1] = dict(c[-1], cache_control=dict(_CC))

    def _cache(messages):
        """prompt caching (CEIL_NOCACHE=1 turns it off): the static system prompt + tool schemas, and the growing
        conversation prefix (its last message) — the usual agent pattern; the previous call's prefix is read back"""
        if os.environ.get("CEIL_NOCACHE") == "1" or not messages:
            return messages
        messages = [dict(m) if isinstance(m, dict) else m for m in messages]
        if isinstance(messages[0], dict) and messages[0].get("role") == "system":
            _mark(messages[0])
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") in ("user", "tool"):
            _mark(last)
        return messages

    def _adapt(kwargs):
        model = str(kwargs.get("model", ""))
        extra = kwargs.pop("extra_body", None)
        if "anthropic/" in model or model.startswith("claude"):
            if not _NOTHINK:
                kwargs.setdefault("thinking", {"type": "adaptive"})
            for k in ("top_k",):
                kwargs.pop(k, None)
            if isinstance(kwargs.get("messages"), list):
                kwargs["messages"] = _cache(kwargs["messages"])
        elif extra is not None:
            kwargs["extra_body"] = extra      # other providers keep their fields
        return kwargs

    def _log(kwargs, resp, t0, err=None):
        if not _LOG:
            return
        u = getattr(resp, "usage", None)
        d = (u.model_dump() if hasattr(u, "model_dump") else dict(u or {})) if u is not None else {}
        rec = {"t": round(time.time(), 1), "s": round(time.time() - t0, 2), "model": kwargs.get("model"),
               "in": d.get("prompt_tokens"), "out": d.get("completion_tokens"),
               "cache_read": (d.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(d.get("prompt_tokens_details"), dict) else d.get("cache_read_input_tokens"),
               "cache_write": d.get("cache_creation_input_tokens"), "err": (str(err)[:160] if err else None)}
        try:
            with open(_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    async def acompletion(*args, **kwargs):
        kwargs = _adapt(kwargs); t0 = time.time()
        try:
            r = await _orig_acompletion(*args, **kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e); raise
        _log(kwargs, r, t0)
        return r

    def completion(*args, **kwargs):
        kwargs = _adapt(kwargs); t0 = time.time()
        try:
            r = _orig_completion(*args, **kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e); raise
        _log(kwargs, r, t0)
        return r

    litellm.acompletion = acompletion
    litellm.completion = completion
