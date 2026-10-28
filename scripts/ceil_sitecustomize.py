"""Ceiling runs (BOX_PLAN.md B-3f/B-3g): adapt the harness's vLLM-shaped requests to a remote Anthropic model.

Imported by scripts/eval_sitecustomize.py (the PYTHONPATH `sitecustomize` eval_dev.sh installs) when MODELS_YAML is set,
so it runs before the harness imports litellm. The harness's generation resolver (adk_submission) puts Gemma/vLLM-only fields into
`extra_body` (chat_template_kwargs.enable_thinking, thinking_token_budget); the Anthropic API rejects unknown fields,
and Claude Opus 5.5 takes `thinking: {type: adaptive}` instead of a token budget (temperature must then be 1; litellm
drops it). So for anthropic/* models this wrapper:
  - removes extra_body, sets thinking adaptive unless CEIL_NOTHINK=1 (the submission's include_thoughts/budget are
    Gemma knobs; the log line in eval_dev.sh says what was used)
  - appends one JSON line per call to CEIL_CALL_LOG (tokens in/out, cache reads, seconds) for the spend total
  - CEIL_GEMMA_CTX=32768 (eval_dev.sh's default for MODELS_YAML runs; 0 = off): Gemma's context window and token units.
    Every request is counted the way the vLLM server counts it — the messages + tool schemas rendered with Gemma 4's own
    chat template (enable_thinking) and tokenizer from CEIL_GEMMA_TOK (runs/gemma_tok: tokenizer.json +
    chat_template.jinja, fetched by eval_dev.sh) + 2, which matched vLLM's prompt_tokens on all 823 E4B calls of wave 7
    (2026-10-28, ±0). If prompt + max_tokens exceeds the window, the call raises the error vLLM raises (same text, so
    the harness and buckets.py see a ctx_overflow); otherwise usage.prompt_tokens in the reply is replaced by the Gemma
    count, so ADK's events compaction (token_threshold 14,336) fires at the same point as for Gemma. The call log keeps
    the provider's own count ("in") next to it ("gemma_in").
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
    _CTX = int(os.environ.get("CEIL_GEMMA_CTX", "0") or 0)
    _TOKDIR = os.environ.get("CEIL_GEMMA_TOK", "runs/gemma_tok")
    _OFFSET = 2                       # vLLM prompt_tokens − rendered tokens, measured (see the docstring)
    _tok = {}
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

    def _gemma_count(kwargs):
        """the prompt tokens Gemma's vLLM server would count for this request"""
        if "t" not in _tok:
            from transformers import PreTrainedTokenizerFast
            t = PreTrainedTokenizerFast(tokenizer_file=os.path.join(_TOKDIR, "tokenizer.json"))
            t.chat_template = open(os.path.join(_TOKDIR, "chat_template.jinja")).read()
            _tok["t"] = t
        msgs = []
        for m in kwargs.get("messages") or []:
            m = dict(m) if isinstance(m, dict) else {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
            m.pop("cache_control", None)
            if isinstance(m.get("content"), list):
                m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
            calls = []
            for c in m.get("tool_calls") or []:
                c = dict(c) if isinstance(c, dict) else c.model_dump()
                f = dict(c.get("function") or {})
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"])
                    except ValueError:
                        pass
                calls.append(dict(c, function=f))
            if calls:
                m["tool_calls"] = calls
            msgs.append(m)
        text = _tok["t"].apply_chat_template(msgs, tools=kwargs.get("tools"), add_generation_prompt=True,
                                             tokenize=False, enable_thinking=True)
        return len(_tok["t"](text, add_special_tokens=False)["input_ids"]) + _OFFSET

    def _window(kwargs):
        """→ the Gemma count, or raise vLLM's context error (None when the emulation is off / not an anthropic call)"""
        model = str(kwargs.get("model", ""))
        if not _CTX or not ("anthropic/" in model or model.startswith("claude")):
            return None
        n = _gemma_count(kwargs)
        out = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0)
        if n + out > _CTX:
            print(f"ceil ctx: Gemma prompt {n:,} + {out:,} output > {_CTX:,} → ContextWindowExceededError (as vLLM)", flush=True)
            raise litellm.ContextWindowExceededError(
                message=(f"OpenAIException - This model's maximum context length is {_CTX} tokens. However, you requested "
                         f"{out} output tokens and your prompt contains at least {n} input tokens, for a total of at least "
                         f"{n + out} tokens. Please reduce the length of the input prompt or the number of requested output "
                         f"tokens. (parameter=input_tokens, value={n})"),
                model=model, llm_provider="openai")
        return n

    def _relabel(resp, n):
        """usage.prompt_tokens → the Gemma count (ADK's compaction threshold reads it); the provider's count is kept"""
        u = getattr(resp, "usage", None)
        if n is None or u is None:
            return None
        own = getattr(u, "prompt_tokens", None)
        try:
            u.prompt_tokens = n
        except Exception:
            pass
        return own

    def _log(kwargs, resp, t0, err=None, gemma=None, own=None):
        if not _LOG:
            return
        u = getattr(resp, "usage", None)
        d = (u.model_dump() if hasattr(u, "model_dump") else dict(u or {})) if u is not None else {}
        rec = {"t": round(time.time(), 1), "s": round(time.time() - t0, 2), "model": kwargs.get("model"),
               "in": own if own is not None else d.get("prompt_tokens"), "gemma_in": gemma, "out": d.get("completion_tokens"),
               "cache_read": (d.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(d.get("prompt_tokens_details"), dict) else d.get("cache_read_input_tokens"),
               "cache_write": d.get("cache_creation_input_tokens"), "err": (str(err)[:160] if err else None)}
        try:
            with open(_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    async def acompletion(*args, **kwargs):
        t0 = time.time()
        try:
            n = _window(kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e); raise
        kwargs = _adapt(kwargs)
        try:
            r = await _orig_acompletion(*args, **kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e, n); raise
        _log(kwargs, r, t0, None, n, _relabel(r, n))
        return r

    def completion(*args, **kwargs):
        t0 = time.time()
        try:
            n = _window(kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e); raise
        kwargs = _adapt(kwargs)
        try:
            r = _orig_completion(*args, **kwargs)
        except Exception as e:
            _log(kwargs, None, t0, e, n); raise
        _log(kwargs, r, t0, None, n, _relabel(r, n))
        return r

    litellm.acompletion = acompletion
    litellm.completion = completion
