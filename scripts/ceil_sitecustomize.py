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
  CEIL_HIST_THOUGHTS=1  the remote model's thought stays in the harness's history and is re-sent with every later
                        request, as the harness does for Gemma (0 / unset = the layer below decides, which on the
                        Anthropic route loses it: ADK takes the reply's `thinking_blocks` before `reasoning_content`
                        and discards a block that has no `signature` key — found 2026-11-20, 0 of 391 thoughts re-sent).
                        Reply: the thought is left as `reasoning_content` only (signatures are remembered here).
                        Request: on anthropic/* each earlier `reasoning_content` is sent as a thinking block (LiteLLM
                        drops the plain field there); on the OpenAI route the field passes as it is.
  CEIL_HIST_PROBE=1     once per process, at the first request that carries an earlier thought: the same request is
                        sent again without the thoughts (16 output tokens) and both prompt token counts are printed —
                        the only way to see from outside whether the gateway hands the thoughts to the model.
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
    # CEIL_REMINDER (B-11, the user's decision of 2026-11-20): a line added as a text block of its own at the END of
    # every agent request (one that carries tools; after a tool's output: a user message that follows it), so the teacher reasons before every tool call. It exists only in the
    # request that leaves this adapter: the harness's history is not touched (a copy is changed), the trajectory log
    # (traj_log.py wraps this adapter and sees the request before it) and the Gemma token count do not contain it,
    # and so no training prompt does. unset / 0 = off · 1 = the default text · anything else = that text.
    import teach_reminder
    _REM = teach_reminder.text(os.environ.get("CEIL_REMINDER", ""))
    _REM_ID = teach_reminder.ident(os.environ.get("CEIL_REMINDER", ""))
    # CEIL_REMINDER_PLACE (2026-11-20, the user's test): end = the block at the end of the request (above) ·
    # system = the same line appended to the system prompt of the outgoing request, nothing at the end. Either way the
    # harness's own messages and the trajectory log keep the submission's prompt unchanged.
    _REM_PLACE = os.environ.get("CEIL_REMINDER_PLACE", "end") or "end"
    if _REM_PLACE not in ("end", "system"):
        raise ValueError(f"CEIL_REMINDER_PLACE={_REM_PLACE!r}: end | system")
    _REM_N = [0, 0]                   # requests with the reminder · agent requests without (last message not user / tool)

    def _answers(msgs, tool):
        """is the last message the result of a call to `tool`?"""
        cid = msgs[-1].get("tool_call_id")
        for m in reversed(msgs[:-1]):
            if isinstance(m, dict) and m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    c = c if isinstance(c, dict) else c.model_dump()
                    if (c.get("function") or {}).get("name") == tool and (cid is None or c.get("id") == cid):
                        return True
                return False
        return False

    def _remind(kwargs):
        msgs = kwargs.get("messages")
        if not _REM or not kwargs.get("tools") or not isinstance(msgs, list) or not msgs:
            return
        if _REM_PLACE == "system":
            first = msgs[0]
            if not isinstance(first, dict) or first.get("role") != "system":
                _REM_N[1] += 1; return
            first = dict(first); c = first.get("content")
            if isinstance(c, list):      # the cache mark sits on the last block: the line goes in front of it, inside the cached part
                blocks = list(c); blocks.insert(max(0, len(blocks) - 1), {"type": "text", "text": _REM}); first["content"] = blocks
                if len(c) == 1 and isinstance(c[0], dict) and c[0].get("type") == "text":
                    first["content"] = [dict(c[0], text=c[0].get("text", "") + "\n\n" + _REM)]
            else:
                first["content"] = (c or "") + "\n\n" + _REM
            kwargs["messages"] = [first] + msgs[1:]; _REM_N[0] += 1
            return
        last = msgs[-1]
        if not isinstance(last, dict) or last.get("role") not in ("user", "tool"):
            _REM_N[1] += 1; return
        # always a text block of its own, never inside a tool's output (2026-11-20: the box's replay gave thoughts on
        # 42 % of calls with the line inside the tool output string and 96 % with it as a separate block)
        if last.get("role") == "tool" and _answers(msgs, "submit_patch"):
            # nothing is called after the submit: the line there only made the teacher talk about it (2026-11-20, third
            # pilot: 8 of the 27 closing replies mention "the reminder", 0 of the 610 replies before the submit)
            _REM_N[1] += 1; return
        if last.get("role") == "tool":      # a user message after the tool's: the Anthropic format merges the two into
            kwargs["messages"] = msgs + [{"role": "user", "content": _REM}]      # one turn [tool_result, text]
        else:
            last = dict(last); c = last.get("content")
            blocks = list(c) if isinstance(c, list) else ([{"type": "text", "text": c}] if c else [])
            last["content"] = blocks + [{"type": "text", "text": _REM}]
            kwargs["messages"] = msgs[:-1] + [last]
        _REM_N[0] += 1
    _HIST = os.environ.get("CEIL_HIST_THOUGHTS", "0") not in ("", "0")
    _PROBE = [os.environ.get("CEIL_HIST_PROBE", "0") not in ("", "0")]
    _SIG = {}                         # thought text → the signature its block came with (sent back with it)
    _HIST_N = [0, 0]                  # replies whose thought was kept · thinking blocks without a signature key among them

    def _thought_of(m):
        """(thought text, shape of the reply's thought fields) of a reply message"""
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        blocks = get("thinking_blocks"); rc = get("reasoning_content") or get("reasoning")
        bl = [b for b in blocks or [] if isinstance(b, dict) and b.get("thinking")]
        text = rc if isinstance(rc, str) and rc.strip() else "\n".join(b["thinking"] for b in bl)
        shape = (f"blocks {'none' if blocks is None else len(blocks)} · with a signature key {sum('signature' in b for b in bl)} · "
                 f"signed {sum(bool(b.get('signature')) for b in bl)} · reasoning_content {len(rc) if isinstance(rc, str) else 'none'}")
        return (text if isinstance(text, str) and text.strip() else None), shape, bl

    def _keep_thought(resp):
        """reply → its thought in the one form ADK keeps (reasoning_content); returns the shape for the call log"""
        shapes = []
        for ch in getattr(resp, "choices", None) or []:
            m = getattr(ch, "message", None)
            if m is None:
                continue
            text, shape, bl = _thought_of(m); shapes.append(shape)
            if not _HIST or not text:
                continue
            for b in bl:
                if b.get("signature"):
                    _SIG[b["thinking"]] = b["signature"]
            if len(bl) == 1 and bl[0].get("signature"):
                _SIG[text] = bl[0]["signature"]
            _HIST_N[0] += 1; _HIST_N[1] += any("signature" not in b for b in bl)
            try:
                m.reasoning_content = text
                m.thinking_blocks = None
                psf = getattr(m, "provider_specific_fields", None)
                if isinstance(psf, dict) and "thinking_blocks" in psf:
                    m.provider_specific_fields = {k: v for k, v in psf.items() if k != "thinking_blocks"}
            except Exception as e:
                print(f"ceil adapter: could not keep the thought ({type(e).__name__}: {e})", flush=True)
        return "; ".join(shapes) or None

    def _hist_stats(messages):
        """(assistant messages in the request, those carrying a thought, characters of those thoughts)"""
        n = k = c = 0
        for m in messages or []:
            if isinstance(m, dict) and m.get("role") == "assistant":
                n += 1
                t = m.get("reasoning_content") or "".join(b.get("thinking", "") for b in m.get("thinking_blocks") or [] if isinstance(b, dict))
                k += bool(t); c += len(t or "")
        return n, k, c

    def _hist_out(messages, model):
        """earlier thoughts in the form the route sends (a copy; the harness's messages are not touched)"""
        if not _HIST or not model.startswith("anthropic/"):
            return messages
        out = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content") and not m.get("thinking_blocks"):
                t = m["reasoning_content"]
                m = dict(m, thinking_blocks=[{"type": "thinking", "thinking": t, "signature": _SIG.get(t, "")}])
            out.append(m)
        return out

    def _no_thoughts(messages):
        out = []
        for m in messages or []:
            if isinstance(m, dict) and m.get("role") == "assistant":
                m = {k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning", "thinking_blocks")}
            out.append(m)
        return out

    def _probe_kwargs(kwargs):
        """→ the request for the probe (no earlier thoughts, 16 output tokens), or None when it is not this call's turn"""
        if not _PROBE[0] or not _hist_stats(kwargs.get("messages"))[1]:
            return None
        _PROBE[0] = False
        k = dict(kwargs, messages=_no_thoughts(kwargs["messages"]))
        for name in ("max_tokens", "max_completion_tokens"):
            k.pop(name, None)
        k["max_tokens"] = 16
        return k

    def _probe_report(kwargs, own, resp2, err=None):
        n, k, c = _hist_stats(kwargs.get("messages"))
        if err is not None or resp2 is None:
            print(f"ceil hist probe: the request without the thoughts failed ({type(err).__name__ if err else 'no reply'}: {str(err)[:160]}) → no reading", flush=True)
            return
        without = teach_ledger_usage(resp2)[0]
        if _LEDGER is not None:
            try:
                tin, cached, tout, reasoning, usd = teach_ledger_usage(resp2)
                _LEDGER.charge("hist_probe#0", tin=tin, cached=cached, tout=tout, reasoning=reasoning, usd=usd, seconds=0.0, err=None, extra={"probe": True})
            except Exception as e:
                print(f"ceil adapter: ledger charge of the probe failed: {type(e).__name__}: {e}", flush=True)
        d = (own or 0) - without
        verdict = ("REACH the model" if d >= c / 8 else "DO NOT reach the model (the gateway or the route drops them)") if own else "no reading (no prompt token count)"
        print(f"ceil hist probe: provider prompt tokens {own} with the {k} earlier thoughts ({c} chars) · {without} without them · "
              f"difference {d} (≥ {c // 8} expected) → the thoughts {verdict}", flush=True)

    def teach_ledger_usage(resp):
        import teach_ledger as tl
        return tl.usage_of(resp)

    print(f"ceil history: thoughts {'kept and re-sent' if _HIST else 'as the layer below decides'} (CEIL_HIST_THOUGHTS) · "
          f"probe {'on' if _PROBE[0] else 'off'} (CEIL_HIST_PROBE)", flush=True)
    print(f"ceil adapter: temperature {'as sent by the harness / dropped by the provider rule' if _TEMP is None else _TEMP} (CEIL_TEMP) · "
          f"thinking {_THINK} (CEIL_THINK) · every model remote {_REMOTE_ALL} (CEIL_REMOTE) · "
          f"ledger {os.environ.get('TEACH_LEDGER_DIR') or 'off'} · attempt {_ATTEMPT} (TEACH_ATTEMPT) · "
          f"reminder {('on [' + _REM_ID + '], ' + str(len(_REM)) + ' chars, sha ' + __import__('hashlib').sha256(_REM.encode()).hexdigest()[:8] + ': ' + _REM[:60] + '…') if _REM else 'off'} (CEIL_REMINDER) · place {_REM_PLACE} (CEIL_REMINDER_PLACE; "
          f"in the outgoing request only, never in the trajectory log)", flush=True)

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
                kwargs["messages"] = _cache(_hist_out(kwargs["messages"], model))
            _remind(kwargs)      # after the cache mark: the mark stays on the conversation itself, the line follows it
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

    def _log(kwargs, resp, t0, err=None, gemma=None, own=None, shape=None):
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
               "temp": kwargs.get("temperature"), "reminder": bool(_REM) and _REM in json.dumps((kwargs.get("messages")[:1] if _REM_PLACE == "system" else kwargs.get("messages")[-1:]) if isinstance(kwargs.get("messages"), list) else "", default=str),
               "reminder_place": _REM_PLACE if _REM else None, "reminder_id": _REM_ID,
               "cache_read": (d.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(d.get("prompt_tokens_details"), dict) else d.get("cache_read_input_tokens"),
               "cache_write": d.get("cache_creation_input_tokens"), "err": (str(err)[:160] if err else None),
               "think_shape": shape, "hist": dict(zip(("assistant_msgs", "with_thought", "thought_chars"), _hist_stats(kwargs.get("messages"))))}
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
        shape = _keep_thought(r) if _is_remote(str(kwargs.get("model", ""))) else None
        own = _relabel(r, n)
        _log(kwargs, r, t0, None, n, own, shape)
        pk = _probe_kwargs(kwargs)
        if pk is not None:
            try:
                _probe_report(kwargs, own if own is not None else teach_ledger_usage(r)[0], await _orig_acompletion(*args, **pk))
            except Exception as e:
                _probe_report(kwargs, None, None, e)
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
        shape = _keep_thought(r) if _is_remote(str(kwargs.get("model", ""))) else None
        own = _relabel(r, n)
        _log(kwargs, r, t0, None, n, own, shape)
        pk = _probe_kwargs(kwargs)
        if pk is not None:
            try:
                _probe_report(kwargs, own if own is not None else teach_ledger_usage(r)[0], _orig_completion(*args, **pk))
            except Exception as e:
                _probe_report(kwargs, None, None, e)
        return r

    litellm.acompletion = acompletion
    litellm.completion = completion
