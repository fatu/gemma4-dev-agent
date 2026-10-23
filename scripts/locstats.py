"""How fast does the agent find, read and edit the file the gold patch changes? (course claim: "localize in <= 3 calls")
Usage: python3 scripts/locstats.py <run dir>…   (any dir with traces/trace_<id>.json and task_results.jsonl below it)
Per run: gold source file named in the issue · first call whose args/output contain a gold path · first read of it
(read_file, or sed/cat/head/… in run_command) · first edit of it (edit_file/write_file, or a run_command that
writes: sed -i, .write(, write_text, cat >, > x.py, patch, git apply) · prompt growth until it was seen ·
the same split by resolved / unresolved. Gold files = the patch's non-test paths (harness/tasks.jsonl).
"""
import json, glob, re, ast, os, statistics as st, sys
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open("harness/tasks.jsonl")}
def gold(t):
    fs = re.findall(r"^\+\+\+ b/(\S+)", tasks[t]["patch"], re.M)
    return [f for f in fs if not re.search(r"(^|/)tests?/|test_", f)]
def parse(x):
    if isinstance(x,(list,dict)): return x
    try: return ast.literal_eval(x)
    except Exception: return []
for arm in sys.argv[1:]:
    rows=[]
    resolved = {}
    for rf in glob.glob(f"{arm}/**/task_results.jsonl", recursive=True):
        for l in open(rf):
            if l.strip(): r = json.loads(l); resolved[r["instance_id"]] = r["resolved"]
    for tf in sorted(glob.glob(f"{arm}/**/traces/trace_*.json", recursive=True)):
        t = re.search(r"trace_(.+)\.json", tf).group(1)
        if t not in tasks or t not in resolved: continue
        res = resolved[t]
        g = gold(t); 
        if not g: continue
        steps = json.load(open(tf))["steps"]
        in_issue = any(f in tasks[t]["problem_statement"] or f.split("/")[-1] in tasks[t]["problem_statement"] for f in g)
        n=0; seen=read=edit=None; p0=None; tok_seen=None
        for s in steps:
            if s.get("source")!="agent": continue
            m = parse(s.get("metrics","{}")) or {}
            if p0 is None and m: p0 = m.get("prompt_tokens")
            for c in parse(s.get("tool_calls","[]")) or []:
                n+=1
                a = json.dumps(c.get("arguments",{})); fn=c.get("function_name")
                obs = str(s.get("observation",""))
                hit = lambda txt: any(f in txt for f in g)
                if seen is None and (hit(obs) or hit(a)):
                    seen=n; tok_seen=(m.get("prompt_tokens",0)-(p0 or 0))
                if read is None and ((fn=="read_file" and hit(a)) or (fn=="run_command" and hit(a) and re.search(r"\b(sed|cat|head|tail|nl|awk)\b",a))):
                    read=n
                wrote = fn in ("edit_file", "write_file") or (fn == "run_command" and re.search(
                    r"sed -i|\.write\(|write_text|cat >|>> ?\S+\.py|> ?\S+\.py|patch -p|git apply", a))
                if edit is None and wrote and hit(a): edit = n
        rows.append(dict(t=t,res=res,in_issue=in_issue,seen=seen,read=read,edit=edit,calls=n,tok=tok_seen))
    N=len(rows)
    def q(k,sub=rows):
        v=[r[k] for r in sub if r[k] is not None]; return f"{len(v)}/{len(sub)} · median {st.median(v) if v else '-'}"
    print(f"== {os.path.basename(arm.rstrip('/'))}: {N} traces with a gold source file")
    print(f"  gold file named in the issue      {sum(r['in_issue'] for r in rows)}/{N}")
    print(f"  gold path appears (call #)        {q('seen')} · ≤3 calls: {sum(1 for r in rows if r['seen'] and r['seen']<=3)}/{N}")
    print(f"  gold file read (call #)           {q('read')} · ≤3 calls: {sum(1 for r in rows if r['read'] and r['read']<=3)}/{N}")
    print(f"  gold file edited (call #)         {q('edit')}")
    tk=[r['tok'] for r in rows if r['tok'] is not None]; print(f"  prompt growth until gold seen     median {st.median(tk):.0f} tok, max {max(tk)}")
    W=[r for r in rows if r['res']]; L=[r for r in rows if not r['res']]
    print(f"  resolved {len(W)}: seen {q('seen',W)} · read {q('read',W)} · edit {q('edit',W)}")
    print(f"  unresolved {len(L)}: seen {q('seen',L)} · read {q('read',L)} · edit {q('edit',L)}")
    print(f"  unresolved that edited a gold file: {sum(1 for r in L if r['edit'])}/{len(L)}")
