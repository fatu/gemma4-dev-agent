"""Trajectory log: every model call of a run, with the thinking the harness trace drops (box runs; TRAJ=0 turns it off).

Imported by scripts/eval_sitecustomize.py when TRAJ_DIR is set; eval_dev.sh sets it on the `swegemma eval` command only
(TRAJ_DIR=<run>/traj), so the other python calls in eval_dev.sh never import litellm. It wraps litellm.acompletion — the
function ADK's LiteLlm calls (it reads the module attribute lazily, so this patch is in place before the first call) —
and appends one JSON line per call to <TRAJ_DIR>/<task>.jsonl:
  call      1, 2, … per task
  new       the request messages added since that task's previous call (the first call: system + task prompt), so the
            file grows linearly; the previous reply comes back here too, which shows how it was RE-SENT (with or
            without reasoning_content, role tool_responses, …)
  setup     first call only: model, sampling params, extra_body (enable_thinking / thinking budget), tool names, and
            tool_schemas = the full tool list as sent (needed to re-render the exact prompt with the chat template)
  resp      what the model returned: reasoning (the thinking), content (text), tool_calls, finish_reason
  usage     prompt / completion / cached tokens · s = seconds for the call · err = the exception if the call failed
The task id comes from the ContextVar scripts/sandbox_shim.py sets around run_agent_sandbox (task_context()).
Render one task: python3 scripts/walkthrough.py <run>/traj/<task>.jsonl [--brief]   (📷 --brief).
Outputs quote task data → they stay in runs/ (gitignored).
"""
import json
import os
import pathlib
import time

DIR = os.environ.get("TRAJ_DIR")
try:
    import litellm
except Exception:                      # not the eval venv: do nothing
    litellm = None


def _plain(x):
    """messages / responses → JSON-safe plain data (litellm objects are pydantic or dict-like)"""
    if hasattr(x, "model_dump"):
        try:
            return _plain(x.model_dump())
        except Exception:
            pass
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


if litellm is not None and DIR:
    import sandbox_shim

    sandbox_shim.task_context()
    _out = pathlib.Path(DIR)
    _out.mkdir(parents=True, exist_ok=True)
    _seen = {}                          # task → (calls so far, messages already logged)
    _orig = litellm.acompletion

    def _resp(r):
        try:
            ch = r.choices[0]
            m = ch.message
            md = _plain(m)
            reasoning = md.get("reasoning_content") or md.get("reasoning") or (
                (md.get("provider_specific_fields") or {}).get("reasoning")) or "\n".join(
                b.get("thinking", "") for b in (md.get("thinking_blocks") or []) if isinstance(b, dict)) or None
            tcs = [{"name": (t.get("function") or {}).get("name"), "args": (t.get("function") or {}).get("arguments"),
                    "id": t.get("id")} for t in (md.get("tool_calls") or [])]
            return {"reasoning": reasoning, "content": md.get("content"), "tool_calls": tcs,
                    "finish_reason": getattr(ch, "finish_reason", None)}
        except Exception as e:
            return {"unparsed": f"{type(e).__name__}: {e}"}

    def _usage(r):
        u = _plain(getattr(r, "usage", None) or {})
        det = u.get("prompt_tokens_details") or {}
        return {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                "cached": det.get("cached_tokens") if isinstance(det, dict) else None,
                "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}

    def _write(kwargs, r, t0, err=None):
        try:
            task = sandbox_shim.current_task() or "unknown_task"
            msgs = kwargs.get("messages") or []
            n_calls, n_prev = _seen.get(task, (0, 0))
            reset = len(msgs) < n_prev          # history shrank (compaction / restart): log it all again
            start = 0 if reset else n_prev
            rec = {"task": task, "call": n_calls + 1, "t": round(t0, 2), "s": round(time.time() - t0, 2),
                   "n_msgs": len(msgs), "reset": reset or None, "new": _plain(msgs[start:])}
            if n_calls == 0:
                tools = kwargs.get("tools") or []
                rec["setup"] = {"model": kwargs.get("model"),
                                "params": _plain({k: kwargs.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens",
                                                                             "max_completion_tokens", "extra_body", "stream")}),
                                "tools": [(t.get("function") or {}).get("name") for t in tools if isinstance(t, dict)],
                                "tool_schemas": _plain(tools)}
            if kwargs.get("stream"):
                rec["resp"] = {"unparsed": "stream=True — reply not captured"}
            elif r is not None:
                rec["resp"], rec["usage"] = _resp(r), _usage(r)
            if err is not None:
                rec["err"] = f"{type(err).__name__}: {str(err)[:300]}"
            _seen[task] = (n_calls + 1, len(msgs))
            with open(_out / f"{task}.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:                  # never break a run over the log
            print(f"traj log: failed: {type(e).__name__}: {e}", flush=True)

    async def acompletion(*args, **kwargs):
        t0 = time.time()
        try:
            r = await _orig(*args, **kwargs)
        except Exception as e:
            _write(kwargs, None, t0, e)
            raise
        _write(kwargs, r, t0)
        return r

    litellm.acompletion = acompletion
    print(f"traj log: on → {DIR}/<task>.jsonl (per call: new messages, thinking, reply, tokens)", flush=True)
