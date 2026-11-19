"""Run the competition harness's agent loop inside each task's own Docker image (BOX_PLAN.md B-11).

Loaded by scripts/eval_sitecustomize.py when DOCKER_TASKS=<the task table, runs/b10/tasks/all.jsonl> is set and the
run uses `--sandbox docker`. The harness as shipped starts ONE image for every task, unpacks a snapshot into
/workspace and installs the project there. The public tasks used here (SWE-rebench, SWE-rebench-V2) come the other
way round: one prebuilt image per task with the repository already checked out and installed at `workdir`
(/testbed, or /<repo name> for V2). So, per task:

  start            the task's `image_name` instead of the global image (entrypoint `sleep infinity`, the harness's
                   own network mode — none —, memory and CPU limits), found through the task id that
                   scripts/sandbox_shim.py keeps in a ContextVar. A warm idle container is never reused.
  prepare          inside the container, before the harness touches it:
                     mv <workdir> /workspace, then ln -s /workspace <workdir>   (the harness's file tools refuse a
                       /workspace that is a symlink out of the sandbox; the reverse link keeps editable installs and
                       absolute paths of the image working)
                     git: safe.directory, an identity for the harness's baseline commit, and — DOCKER_STRIP_REFS=1,
                       the default — every ref that is not an ancestor of HEAD deleted, reflog expired, unreachable
                       objects pruned (the same lines as scripts/b10_startup.sh: later commits must not be readable)
                     /etc/swegemma_env.sh, named by BASH_ENV: every `bash -c` the harness runs first activates the
                       image's conda env `testbed` when there is one (the harness does not use a login shell)
  switched off     the snapshot copy, the wheel setup, the editable install, the test-dependency install and the
                   harness's own pytest.ini / conftest.py: the image is the environment, as the task's grader uses it.
  kept             the harness's git exclude and its baseline commit + `_swegemma_baseline` tag, so `submit_patch`
                   returns the diff against the prepared tree.

The tasks file given to the harness carries no test fields, so its verification returns at once and keeps the patch;
grading is scripts/rebench_mini.py on the patch files. Every step prints one line per task (photos are the record).
Not testable without Docker: the unit test (`python3 scripts/docker_task_shim.py selftest`) runs the start and the
prepare script against a stand-in client and checks the calls; the script itself is also run in a temp dir with plain
bash and git.
"""
import json
import os
import shlex
import sys
import threading
import time

TABLE = os.environ.get("DOCKER_TASKS")
STRIP = os.environ.get("DOCKER_STRIP_REFS", "1")
SCRATCH_SNAPSHOT = os.environ.get("DOCKER_DUMMY_SNAPSHOT", "/tmp/swegemma_docker_task_shim.tgz")
_rows, _lock = {}, threading.Lock()


def rows():
    if not _rows and TABLE:
        for line in open(TABLE):
            if line.strip():
                r = json.loads(line)
                _rows[r["instance_id"]] = r
    return _rows


def prepare_script(workdir, strip=STRIP):
    w = shlex.quote(workdir)
    return f"""set -e
[ -d {w} ] || {{ echo "no repository at {workdir}"; exit 3; }}
if [ {w} != /workspace ]; then
  rm -rf /workspace
  mv {w} /workspace
  ln -s /workspace {w}
fi
cd /workspace
git config --global --add safe.directory '*'
git config --global user.email harness@localhost
git config --global user.name harness
if [ "{strip}" = 1 ] && git rev-parse HEAD >/dev/null 2>&1; then
  git for-each-ref --no-merged HEAD --format='%(symref)|%(refname)' | while IFS='|' read -r s r; do [ -z "$s" ] || git symbolic-ref --delete "$r"; done
  git for-each-ref --no-merged HEAD --format='delete %(refname)' | git update-ref --no-deref --stdin
  rm -f "$(git rev-parse --git-dir)/FETCH_HEAD" "$(git rev-parse --git-dir)/ORIG_HEAD"
  git reflog expire --expire=now --all >/dev/null 2>&1 || true
  timeout 40 git gc -q --prune=now >/dev/null 2>&1 || true
  [ "$(git for-each-ref --no-merged HEAD | wc -l)" -eq 0 ] || {{ echo "refs later than HEAD are still there"; exit 4; }}
fi
ACT=""
for a in /opt/miniconda3/bin/activate /opt/conda/bin/activate /root/miniconda3/bin/activate; do
  [ -f "$a" ] && [ -d "$(dirname "$(dirname "$a")")/envs/testbed" ] && ACT="$a" && break
done
if [ -n "$ACT" ]; then
  printf 'unset BASH_ENV\\n. %s testbed >/dev/null 2>&1 || true\\n' "$ACT" > /etc/swegemma_env.sh
else
  printf 'unset BASH_ENV\\n' > /etc/swegemma_env.sh
fi
echo "PREP ok · conda ${{ACT:-none}} · python $(. /etc/swegemma_env.sh; command -v python3 || command -v python || echo none) · refs $(git for-each-ref | wc -l) · head $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"""


def start_task_container(manager, iid, out=print):
    """→ container id; raises with a message that names the task when something is missing"""
    r = rows().get(iid)
    if r is None:
        raise RuntimeError(f"docker_task_shim: task {iid!r} is not in {TABLE}")
    if r.get("skip") or not r.get("image_name") or not r.get("workdir"):
        raise RuntimeError(f"docker_task_shim: task {iid} cannot run: skip={r.get('skip')!r} image={r.get('image_name')!r} workdir={r.get('workdir')!r}")
    cfg = manager.config
    env = dict(getattr(cfg, "environment", None) or {}, BASH_ENV="/etc/swegemma_env.sh")
    t0 = time.time()
    c = manager.client.containers.run(
        r["image_name"], entrypoint=["sleep"], command=["infinity"], detach=True,
        network_mode=cfg.network_mode, mem_limit=cfg.mem_limit, cpu_period=cfg.cpu_period, cpu_quota=cfg.cpu_quota,
        working_dir="/", environment=env)
    cid = str(c.id or "")
    with manager._lock:
        manager._active_containers.add(cid)
        if manager._default_container_id is None:
            manager._default_container_id = cid
    res = c.exec_run(cmd=["/bin/bash", "-c", prepare_script(r["workdir"])], stdout=True, stderr=True)
    text = (res.output or b"").decode("utf-8", "replace").strip()
    last = text.splitlines()[-1] if text else ""
    out(f"docker_task_shim: {iid} · image {r['image_name']} · workdir {r['workdir']} → /workspace · {time.time() - t0:.1f} s · "
        f"exit {res.exit_code} · {last[:200]}", flush=True)
    if res.exit_code != 0:
        raise RuntimeError(f"docker_task_shim: preparing {iid} failed (exit {res.exit_code}): {text[-400:]}")
    return cid


def install():
    if not TABLE:
        return False
    try:
        from swegemma.sandbox import docker as sd
        import swegemma.evaluate as ev
        import swegemma.harness.agent_runner as ar
        import sandbox_shim
    except Exception as e:
        print(f"docker_task_shim: not installed ({type(e).__name__}: {e})", flush=True)
        return False
    cls = sd.ContainerManager
    if getattr(cls, "_task_shim", False):
        return True
    sandbox_shim.task_context()

    def start(self):
        iid = sandbox_shim.current_task()
        if iid is None:
            raise RuntimeError("docker_task_shim: no task id in the context (start() called outside a task)")
        return start_task_container(self, iid)

    cls.start = start
    cls._task_shim = True
    off = []
    for name in ("setup_container_wheels", "extract_snapshot", "install_editable_package", "install_test_dependencies",
                 "setup_workspace_test_config"):
        if hasattr(ar, name):
            setattr(ar, name, lambda *a, **k: None); off.append(name)
    if not os.path.exists(SCRATCH_SNAPSHOT):
        open(SCRATCH_SNAPSHOT, "wb").close()
    import pathlib
    ev.resolve_task_snapshot_paths = lambda *a, **k: (pathlib.Path(SCRATCH_SNAPSHOT), None, None)
    print(f"docker_task_shim: on · task table {TABLE} ({len(rows())} rows, {sum(1 for r in rows().values() if not r.get('skip'))} runnable) · "
          f"strip later refs {STRIP} (DOCKER_STRIP_REFS) · harness steps switched off: {', '.join(off)} · snapshot check answered with {SCRATCH_SNAPSHOT}", flush=True)
    return True


def selftest():
    import subprocess
    import tempfile
    import types

    class Exec:
        def __init__(self, code, out):
            self.exit_code, self.output = code, out

    class C:
        id = "cid123"

        def __init__(self, code=0):
            self.code, self.cmds = code, []

        def exec_run(self, cmd, **kw):
            self.cmds.append(cmd)
            return Exec(self.code, b"line 1\nPREP ok \xc2\xb7 conda none" if not self.code else b"no repository at /testbed")

    class Containers:
        def __init__(self, c):
            self.c, self.calls = c, []

        def run(self, image, **kw):
            self.calls.append((image, kw)); return self.c

    def manager(code=0):
        c = C(code)
        m = types.SimpleNamespace(client=types.SimpleNamespace(containers=Containers(c)), _lock=threading.Lock(), _active_containers=set(),
                                  _default_container_id=None,
                                  config=types.SimpleNamespace(network_mode="none", mem_limit="4g", cpu_period=100000, cpu_quota=200000,
                                                               environment={"TEST_TMPDIR": "/tmp"}))
        return m, c

    _rows.clear()
    _rows.update({"a__b-1": {"instance_id": "a__b-1", "image_name": "swerebench/sweb.eval.x86_64.a_1776_b-1", "workdir": "/testbed"},
                  "c__d-2": {"instance_id": "c__d-2", "image_name": "img2", "workdir": "/d"},
                  "e__f-3": {"instance_id": "e__f-3", "image_name": "img3", "workdir": "/testbed", "skip": "no test_cmd"}})
    log = []
    m, c = manager()
    assert start_task_container(m, "a__b-1", out=lambda *a, **k: log.append(a[0])) == "cid123"
    image, kw = m.client.containers.calls[0]
    assert image.startswith("swerebench/") and kw["entrypoint"] == ["sleep"] and kw["command"] == ["infinity"] and kw["network_mode"] == "none"
    assert kw["environment"] == {"TEST_TMPDIR": "/tmp", "BASH_ENV": "/etc/swegemma_env.sh"} and kw["working_dir"] == "/" and kw["mem_limit"] == "4g"
    assert m._active_containers == {"cid123"} and m._default_container_id == "cid123"
    assert "mv /testbed /workspace" in c.cmds[0][2] and "PREP ok" in log[0] and "exit 0" in log[0]
    m, c = manager()
    start_task_container(m, "c__d-2", out=lambda *a, **k: None)
    assert "mv /d /workspace" in c.cmds[0][2] and "ln -s /workspace /d" in c.cmds[0][2]
    for iid, word in (("e__f-3", "cannot run"), ("zz", "not in")):
        try:
            start_task_container(manager()[0], iid, out=lambda *a, **k: None); raise AssertionError("no error")
        except RuntimeError as e:
            assert word in str(e), e
    try:
        start_task_container(manager(3)[0], "a__b-1", out=lambda *a, **k: None); raise AssertionError("no error")
    except RuntimeError as e:
        assert "preparing a__b-1 failed" in str(e)
    # the git part of the prepare script, with real bash + git in a temp dir (paths rewritten: no root needed)
    d = tempfile.mkdtemp(prefix="dts_")
    sh = lambda s: subprocess.run(["bash", "-c", s], cwd=d, capture_output=True, text=True)
    r = sh("mkdir repo && cd repo && git init -q . && git config user.email t@t && git config user.name t && echo 1 > f && git add f && git commit -qm base "
           "&& git checkout -q -b later && echo 2 > f && git commit -qam later && git tag v-later && git checkout -q - "
           # a remote branch later than HEAD with origin/HEAD pointing at it: the SWE-rebench images (box, 2026-11-18)
           "&& git update-ref refs/remotes/origin/master later && git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/master")
    assert r.returncode == 0, r.stderr
    script = prepare_script(d + "/repo").replace("/workspace", d + "/workspace").replace("/etc/swegemma_env.sh", d + "/env.sh").replace("--global", "--file " + d + "/gitconfig")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=dict(os.environ, GIT_CONFIG_GLOBAL=d + "/gitconfig"))
    assert r.returncode == 0 and "PREP ok" in r.stdout, (r.stdout, r.stderr)
    assert os.path.islink(d + "/repo") and os.path.isdir(d + "/workspace/.git") and open(d + "/env.sh").read().startswith("unset BASH_ENV")
    refs = subprocess.run(["git", "for-each-ref", "--format=%(refname)"], cwd=d + "/workspace", capture_output=True, text=True).stdout.split()
    assert all("later" not in x for x in refs) and len(refs) == 1, refs           # the later branch and its tag are gone
    gone = subprocess.run(["git", "log", "--all", "--oneline"], cwd=d + "/workspace", capture_output=True, text=True).stdout
    assert "later" not in gone and "base" in gone, gone
    import shutil
    shutil.rmtree(d)
    print("docker_task_shim selftest: per-task image and limits, /workspace move with the reverse link, BASH_ENV file, "
          "a skipped / unknown task and a failed prepare raise with the task named, later refs removed by the real script → all as expected")


install()

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        print(__doc__)
