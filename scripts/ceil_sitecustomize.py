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

Teacher runs (BOX_PLAN.md B-11) — every knob prints its effective value once, at import:
  CEIL_TEMP=<float>     the temperature sent to the remote model (the submission's own value is the student's, e.g. 0.2);
                        unset = the old behaviour (Claude: dropped by litellm under adaptive thinking; others: passed on).
  CEIL_THINK=…          the thinking request for the remote model:
                          adaptive      `thinking: {type: adaptive}` (the default; Claude)
                          off           no thinking field is sent (set explicitly, it also removes the harness's own)
                          harness       what the harness sends is left as it is: ADK turns the submission's
                                        thinking_config into `thinking: {type: enabled, budget_tokens: <its budget>}`
                          budget:<n>    `thinking: {type: enabled, budget_tokens: n}` (written into the request body)
                          effort:<lvl>  `reasoning_effort: <lvl>` (written into the request body)
                          extra:<json>  the JSON object is merged into the request body as it is (a gateway's own switch)
  CEIL_REMOTE=1         treat every model of this run as remote (window emulation, cache marks, the knobs above) — for
                        a gateway reached in the OpenAI format (`openai/<name>`); anthropic/* is remote without it.
  TEACH_LEDGER_DIR      the shared spend ledger (scripts/teach_ledger.py): the gate is checked before every call
                        (budget reached → the call raises, the attempt ends) and every call is charged to
                        `<task id>#<TEACH_ATTEMPT>` with its input / cached / output / reasoning tokens and the
                        provider's cost when the response carries one.
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
    _TEMP = os.environ.get("CEIL_TEMP")
    _TEMP = float(_TEMP) if _TEMP not in (None, "") else None
    _THINK_SET = bool(os.environ.get("CEIL_THINK"))
    _THINK = os.environ.get("CEIL_THINK") or ("off" if _NOTHINK else "adaptive")
    _REMOTE_ALL = os.environ.get("CEIL_REMOTE") == "1"
    _ATTEMPT = os.environ.get("TEACH_ATTEMPT", "0")
    _LEDGER = None
    if os.environ.get("TEACH_LEDGER_DIR"):
        import teach_ledger
        _LEDGER = teach_ledger.Ledger(os.environ["TEACH_LEDGER_DIR"])

    def _think_kwargs():
        """CEIL_THINK → (keyword arguments, fields for the request body)"""
        k, _, v = _THINK.partition(":")
        if k in ("off", "harness"):
            return {}, {}
        if k == "adaptive":
            return {"thinking": {"type": "adaptive"}}, {}
        if k == "budget":      # in the body, not as litellm's keyword: litellm drops `thinking` for a model name it does not know
            return {}, {"thinking": {"type": "enabled", "budget_tokens": int(v)}}
        if k == "effort":      # also a body field: litellm drops the keyword for a model name it does not know
            return {}, {"reasoning_effort": v}
        if k == "extra":
            return {}, json.loads(v)
        raise ValueError(f"CEIL_THINK={_THINK!r}: adaptive | off | harness | budget:<n> | effort:<level> | extra:<json>")

    _TK, _TB = _think_kwargs()
    print(f"ceil adapter: temperature {'as sent by the harness / dropped by the provider rule' if _TEMP is None else _TEMP} (CEIL_TEMP) · "
          f"thinking {_THINK} (CEIL_THINK) · every model remote {_REMOTE_ALL} (CEIL_REMOTE) · "
          f"ledger {os.environ.get('TEACH_LEDGER_DIR') or 'off'} · attempt {_ATTEMPT} (TEACH_ATTEMPT)", flush=True)

    def _is_remote(model):
        return _REMOTE_ALL or "anthropic/" in model or model.startswith("claude")

    def _key():
        try:
            import sandbox_shim
            t = sandbox_shim.current_task()
        except Exception:
            t = None
        return f"{t or 'unknown'}#{_ATTEMPT}"
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
        if _is_remote(model):
            if _THINK_SET and _THINK != "harness":      # an explicit CEIL_THINK replaces what the harness derived from the
                for k in ("thinking", "reasoning_effort"):   # submission's thinking_config (ADK sends thinking enabled + its budget)
                    kwargs.pop(k, None)
            for k, v in _TK.items():
                kwargs.setdefault(k, v)
            if _TB:      # only the gateway's own fields; the harness's vLLM fields stay dropped
                # how a field reaches the body differs by litellm route (measured with scripts/teach_fake.py, litellm of
                # the eval venv): openai/* merges extra_body into the body; anthropic/* sends extra_body as a literal
                # key, drops known keywords of an unknown model name, and passes top-level keywords it was told to allow
                if model.startswith("anthropic/"):
                    kwargs.update(_TB); kwargs["allowed_openai_params"] = sorted(set(kwargs.get("allowed_openai_params") or []) | set(_TB))
                else:
                    kwargs["extra_body"] = dict(_TB)
            if _TEMP is not None:
                kwargs["temperature"] = _TEMP; kwargs.pop("top_p", None)     # one of the two, as the provider advises
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
        if not _CTX or not _is_remote(model):
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
        if _LEDGER is not None:
            try:
                tin, cached, tout, reasoning, usd = teach_ledger.usage_of(resp)
                if own is not None:                   # usage.prompt_tokens was replaced by the Gemma count: bill the provider's
                    tin = own
                if resp is not None or err is None:
                    _LEDGER.charge(_key(), tin=tin, cached=cached, tout=tout, reasoning=reasoning, usd=usd,
                                   seconds=round(time.time() - t0, 2), err=err, extra={"gemma_in": gemma})
                else:                                 # a failed call: no usage to bill, kept as a line for the failure count
                    _LEDGER.charge(_key(), usd=0.0, seconds=round(time.time() - t0, 2), err=err, extra={"gemma_in": gemma})
            except Exception as e:                    # the ledger must never break a run silently: say so
                print(f"ceil adapter: ledger charge failed: {type(e).__name__}: {e}", flush=True)
        if not _LOG:
            return
        u = getattr(resp, "usage", None)
        d = (u.model_dump() if hasattr(u, "model_dump") else dict(u or {})) if u is not None else {}
        rec = {"t": round(time.time(), 1), "s": round(time.time() - t0, 2), "model": kwargs.get("model"),
               "in": own if own is not None else d.get("prompt_tokens"), "gemma_in": gemma, "out": d.get("completion_tokens"),
               "reasoning": (d.get("completion_tokens_details") or {}).get("reasoning_tokens") if isinstance(d.get("completion_tokens_details"), dict) else None,
               "temp": kwargs.get("temperature"),
               "cache_read": (d.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(d.get("prompt_tokens_details"), dict) else d.get("cache_read_input_tokens"),
               "cache_write": d.get("cache_creation_input_tokens"), "err": (str(err)[:160] if err else None)}
        try:
            with open(_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    async def acompletion(*args, **kwargs):
        if _LEDGER is not None:
            _LEDGER.gate(_key())       # budget reached → raises here: no request, no ledger line
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
        if _LEDGER is not None:
            _LEDGER.gate(_key())       # budget reached → raises here: no request, no ledger line
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
