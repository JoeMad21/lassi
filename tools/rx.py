#!/usr/bin/env python3
"""rx: run repository work on the remote build host through the project gate.

Usage (from the repository root):
    uv run tools/rx.py doctor                     local and remote health, free space, jobs
    uv run tools/rx.py run [opts] -- CMD...       sync HEAD, run CMD in a worktree slot, print tail
    uv run tools/rx.py job start [opts] -- CMD... sync HEAD, start CMD detached; prints the job id
    uv run tools/rx.py job status [ID]            one job, or recent jobs
    uv run tools/rx.py job tail ID [-n N]         last lines of a job or run log
    uv run tools/rx.py job wait ID [--timeout S]  poll until done (exit 3 if still running)
    uv run tools/rx.py job kill ID                stop a job
    uv run tools/rx.py exec -- CMD...             inspection command in the scratch root, no checkout
    uv run tools/rx.py pull ID [--into DIR]       fetch a run record; --into copies provenance.json
    uv run tools/rx.py pull --path REL --into DIR fetch lassi-runs/... or lassi-wt/<slot>/... (small)
    uv run tools/rx.py devcheck                   read-only accelerator inventory
    uv run tools/rx.py slot-rm SLOT               delete a worktree slot to free space
    uv run tools/rx.py bootstrap [--update-gate]  owner only: install or update the gate

Run options: --slot NAME (default <machine>-<branch>), --timeout SECONDS, --big (large build or
run; stricter free-space floor, one at a time), --live (stream output), --tail N (lines shown).

Settings come from environment variables or an optional untracked .rx.json at the repo root:
RX_HOST (ssh alias, default ionx), RX_SCRATCH (default /mnt/nvme10/joseph_ufl),
RX_REMOTE_PYTHON (default python3), RX_MACHINE (default this host's name),
RX_TRANSPORT (ssh | local; local runs the gate directly, for tests).

A dirty working tree is sent as a snapshot commit that never touches your branch; its runs are
recorded as dirty and are exploratory only. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOCAL_CACHE = REPO / ".rx"


def settings() -> Dict[str, str]:
    cfg = {
        "host": "ionx",
        "scratch": "/mnt/nvme10/joseph_ufl",
        "remote_python": "python3",
        "machine": re.sub(r"[^A-Za-z0-9._-]", "-", platform.node().split(".")[0].lower()) or "local",
        "transport": "ssh",
    }
    override = REPO / ".rx.json"
    if override.is_file():
        cfg.update(json.loads(override.read_text()))
    for key in list(cfg):
        env = os.environ.get("RX_" + key.upper())
        if env:
            cfg[key] = env
    return cfg


CFG = settings()


def die(msg: str, code: int = 2) -> None:
    print(f"rx: {msg}", file=sys.stderr)
    sys.exit(code)


def ssh_bin() -> str:
    found = shutil.which("ssh")
    if not found:
        die("ssh not found on PATH")
    return found


def gate_argv(request: dict) -> List[str]:
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    gate = f"{CFG['scratch']}/lassi-gate/gate.py"
    if CFG["transport"] == "local":
        return [sys.executable, gate, payload]
    return [ssh_bin(), "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            CFG["host"], f"{CFG['remote_python']} {gate} {payload}"]


def call_gate(request: dict) -> Tuple[int, dict, str]:
    """Call the gate, capture output, return (rc, result, text before the result line)."""
    request.setdefault("machine", CFG["machine"])
    p = subprocess.run(gate_argv(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = p.stdout.decode("utf-8", errors="replace")
    result: dict = {}
    body = text
    idx = text.rfind("RX-RESULT ")
    if idx >= 0:
        body = text[:idx]
        try:
            result = json.loads(text[idx + len("RX-RESULT "):].strip().splitlines()[0])
        except (ValueError, IndexError):
            result = {}
    err = p.stderr.decode("utf-8", errors="replace").strip()
    if p.returncode == 255 and CFG["transport"] == "ssh":
        die(f"ssh to '{CFG['host']}' failed: {err or 'no message'}")
    if not result.get("ok", True) or (not result and p.returncode != 0):
        die(f"gate refused: {result.get('error') or err or body.strip()[-500:]}", 3)
    return p.returncode, result, body


def stream_gate(request: dict, live: bool, tail: int) -> Tuple[int, dict]:
    """Call the gate for run/exec; print live or the last lines; return (rc, result)."""
    request.setdefault("machine", CFG["machine"])
    proc = subprocess.Popen(gate_argv(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    last = collections.deque(maxlen=max(tail, 1))
    total = 0
    result: dict = {}
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if line.startswith("RX-RESULT "):
            try:
                result = json.loads(line[len("RX-RESULT "):])
            except ValueError:
                pass
            continue
        total += 1
        if live:
            print(line, flush=True)
        else:
            last.append(line)
    rc = proc.wait()
    err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
    if rc == 255 and CFG["transport"] == "ssh":
        die(f"ssh to '{CFG['host']}' failed: {err or 'no message'}")
    if result and not result.get("ok", True):
        die(f"gate refused: {result.get('error')}", 3)
    if not live:
        if total > len(last):
            print(f"[rx] showing last {len(last)} of {total} lines; full log: rx job tail {result.get('id')} "
                  f"or rx pull {result.get('id')}")
        for line in last:
            print(line)
    if err:
        print(err, file=sys.stderr)
    print(f"[rx] id={result.get('id')} rc={result.get('rc', rc)} state={result.get('state')}")
    return int(result.get("rc", rc) if result else rc), result


# ---------------------------------------------------------------- git sync


def git(*args: str, env: Optional[dict] = None, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env)
    if check and p.returncode != 0:
        die(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout.decode("utf-8", errors="replace").strip()


def slot_name(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", f"{CFG['machine']}-{branch}")[:80]


def push_url() -> str:
    bare = f"{CFG['scratch']}/lassi.git"
    return bare if CFG["transport"] == "local" else f"{CFG['host']}:{bare}"


def snapshot_commit(head: str) -> str:
    """Commit the working tree (tracked and untracked, ignores honored) without touching refs."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        git("read-tree", head, env=env)
        git("add", "-A", env=env)
        tree = git("write-tree", env=env)
        return git("commit-tree", tree, "-p", head, "-m", f"rx snapshot of {head[:12]}", env=env)


def sync(slot: Optional[str]) -> dict:
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD", check=False) or f"detached-{head[:8]}"
    dirty = bool(git("status", "--porcelain"))
    commit = snapshot_commit(head) if dirty else head
    slot = slot or slot_name(branch)
    env = dict(os.environ)
    if CFG["transport"] == "ssh":
        env["GIT_SSH_COMMAND"] = f'"{ssh_bin()}" -o BatchMode=yes'
    ref = f"refs/wip/{CFG['machine']}/{slot}"
    p = subprocess.run(["git", "-C", str(REPO), "push", "--quiet", "--force", push_url(), f"{commit}:{ref}"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if p.returncode != 0:
        die(f"push to the remote bare repository failed: {p.stderr.decode(errors='replace').strip()}")
    return {"commit": commit, "branch": branch, "dirty": dirty, "snapshot_of": head if dirty else None,
            "slot": slot}


# ---------------------------------------------------------------- commands


def cmd_text(parts: List[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        die("missing command after --")
    return " ".join(parts) if len(parts) > 1 else parts[0]


def do_run(a: argparse.Namespace) -> int:
    s = sync(a.slot)
    req = dict(s, verb="run", cmd=cmd_text(a.cmd), timeout=a.timeout, big=a.big)
    if s["dirty"]:
        print(f"[rx] dirty tree sent as snapshot {s['commit'][:12]} (exploratory; not reportable)")
    rc, _ = stream_gate(req, a.live, a.tail)
    return rc


def do_exec(a: argparse.Namespace) -> int:
    rc, _ = stream_gate({"verb": "exec", "cmd": cmd_text(a.cmd), "timeout": a.timeout}, a.live, a.tail)
    return rc


def do_job(a: argparse.Namespace) -> int:
    if a.jobcmd == "start":
        s = sync(a.slot)
        req = dict(s, verb="job_start", cmd=cmd_text(a.cmd), timeout=a.timeout, big=a.big, name=a.name)
        _, res, _ = call_gate(req)
        print(json.dumps(res))
        return 0
    if a.jobcmd == "status":
        _, res, _ = call_gate({"verb": "job_status", "id": a.id})
        for j in res.get("jobs", []):
            print(f"{j['id']}  {j.get('state')}  rc={j.get('rc')}  slot={j.get('slot')}  "
                  f"start={j.get('start')}  end={j.get('end')}  cmd={(j.get('cmd') or '')[:80]}")
        return 0
    if a.jobcmd == "tail":
        _, res, _ = call_gate({"verb": "job_tail", "id": a.id, "n": a.n})
        print(res.get("tail", ""))
        print(f"[rx] id={res.get('id')} state={res.get('state')} rc={res.get('rc')}")
        return 0
    if a.jobcmd == "kill":
        _, res, _ = call_gate({"verb": "job_kill", "id": a.id})
        print(json.dumps(res))
        return 0
    if a.jobcmd == "wait":
        end = time.time() + a.timeout
        while True:
            _, res, _ = call_gate({"verb": "job_status", "id": a.id})
            job = (res.get("jobs") or [{}])[0]
            if job.get("state") not in ("starting", "running"):
                _, t, _ = call_gate({"verb": "job_tail", "id": a.id, "n": a.n})
                print(t.get("tail", ""))
                print(f"[rx] id={a.id} state={job.get('state')} rc={job.get('rc')}")
                return int(job.get("rc") or (0 if job.get("state") == "done" else 1))
            if time.time() >= end:
                print(f"[rx] id={a.id} still {job.get('state')} after {a.timeout}s; poll again later")
                return 3
            time.sleep(min(a.interval, max(1.0, end - time.time())))
    die(f"unknown job command {a.jobcmd}")
    return 2


def safe_extract(data: bytes, dest: Path) -> List[str]:
    names = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                die(f"unsafe path in archive: {m.name}")
            names.append(m.name)
        tar.extractall(dest)
    return names


def do_pull(a: argparse.Namespace) -> int:
    if not a.id and not a.path:
        die("pull needs a run or job ID, or --path")
    req = {"verb": "tar", "id": a.id} if a.id else {"verb": "tar", "path": a.path}
    req["machine"] = CFG["machine"]
    p = subprocess.run(gate_argv(req), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not p.stdout.startswith(b"\x1f\x8b"):
        msg = p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace")
        die(f"pull failed: {msg.strip()[-500:]}")
    dest = LOCAL_CACHE / "pulls"
    dest.mkdir(parents=True, exist_ok=True)
    names = safe_extract(p.stdout, dest)
    root = dest / names[0].split("/")[0]
    print(f"[rx] pulled {len(names)} entries into {root}")
    if a.into:
        into = (REPO / a.into).resolve()
        into.mkdir(parents=True, exist_ok=True)
        if a.id:
            shutil.copy2(root / "meta.json", into / "provenance.json")
            print(f"[rx] wrote {into / 'provenance.json'}")
        else:
            for f in root.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(root)
                    (into / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, into / rel)
            print(f"[rx] copied {a.path} into {into}")
    return 0


def do_doctor(a: argparse.Namespace) -> int:
    ok = True
    print(f"[rx] settings: {json.dumps(CFG)}")
    if CFG["transport"] == "ssh":
        p = subprocess.run([ssh_bin(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", CFG["host"], "true"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[rx] ssh {CFG['host']}: {'PASS' if p.returncode == 0 else 'FAIL ' + p.stderr.decode().strip()}")
        if p.returncode != 0:
            return 1
    _, res, _ = call_gate({"verb": "doctor"})
    print(json.dumps(res, indent=2))
    if res.get("stop"):
        print("[rx] WARN owner STOP is set on the gate")
        ok = False
    if (res.get("scratch_free_gb") or 0) < 60:
        print(f"[rx] WARN scratch free {res.get('scratch_free_gb')} GB is below the big-job floor")
    print("[rx] Ready" if ok else "[rx] Not ready")
    return 0 if ok else 1


def do_simple(verb: str, extra: Optional[dict] = None) -> int:
    _, res, _ = call_gate(dict({"verb": verb}, **(extra or {})))
    print(json.dumps(res, indent=2))
    return 0


def upload(local: Path, remote: str, mode: str) -> None:
    if CFG["transport"] == "local":
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        os.chmod(remote, int(mode, 8))
        return
    with open(local, "rb") as fh:
        p = subprocess.run([ssh_bin(), "-o", "BatchMode=yes", CFG["host"],
                            f"mkdir -p $(dirname {remote}) && cat > {remote} && chmod {mode} {remote}"],
                           stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"upload of {local.name} failed: {p.stderr.decode(errors='replace').strip()}")


def remote_sh(script: str) -> str:
    argv = ["bash", "-c", script] if CFG["transport"] == "local" else \
        [ssh_bin(), "-o", "BatchMode=yes", CFG["host"], "bash -s"]
    p = subprocess.run(argv, input=None if CFG["transport"] == "local" else script.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode(errors="replace")
    if p.returncode != 0:
        die(f"remote setup failed:\n{out}")
    return out


def do_bootstrap(a: argparse.Namespace) -> int:
    s = CFG["scratch"]
    server = REPO / "tools" / "server"
    remote_sh(f"set -e; mkdir -p {s}/lassi-gate/runs {s}/lassi-wt {s}/lassi-runs {s}/toolchains {s}/tmp "
              f"{s}/.cache {s}/bin; test -d {s}/lassi.git || git init --quiet --bare {s}/lassi.git")
    upload(server / "gate.py", f"{s}/lassi-gate/gate.py", "755")
    upload(server / "config.default.json", f"{s}/lassi-gate/config.default.json", "644")
    upload(server / "pre-receive", f"{s}/lassi.git/hooks/pre-receive", "755")
    out = remote_sh(
        f"set -e; cd {s}/lassi-gate; test -f config.json || cp config.default.json config.json; "
        f"git --git-dir={s}/lassi.git config receive.denyNonFastForwards false; "
        f"git --git-dir={s}/lassi.git config receive.denyDeletes false; "
        f"echo python: $({CFG['remote_python']} --version 2>&1); echo git: $(git --version); "
        f"echo uv: $(PATH={s}/bin:$PATH command -v uv || echo MISSING); df -h {s} / | sed 's/^/df: /'")
    print(out.strip())
    print("[rx] gate installed; config.json kept if it existed (edit it by hand; agents never do)")
    return do_simple("ping")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rx", description="Remote execution through the project gate")
    sub = p.add_subparsers(dest="verb", required=True)

    def run_opts(sp: argparse.ArgumentParser, slot: bool = True) -> None:
        if slot:
            sp.add_argument("--slot")
            sp.add_argument("--big", action="store_true")
        sp.add_argument("--timeout", type=int, default=None)
        sp.add_argument("--live", action="store_true")
        sp.add_argument("--tail", type=int, default=120)
        sp.add_argument("cmd", nargs=argparse.REMAINDER)

    sub.add_parser("doctor")
    sub.add_parser("devcheck")
    run_opts(sub.add_parser("run"))
    run_opts(sub.add_parser("exec"), slot=False)
    j = sub.add_parser("job")
    jsub = j.add_subparsers(dest="jobcmd", required=True)
    js = jsub.add_parser("start")
    js.add_argument("--slot")
    js.add_argument("--big", action="store_true")
    js.add_argument("--name")
    js.add_argument("--timeout", type=int, default=86400)
    js.add_argument("cmd", nargs=argparse.REMAINDER)
    jst = jsub.add_parser("status")
    jst.add_argument("id", nargs="?")
    for name in ("tail", "kill"):
        x = jsub.add_parser(name)
        x.add_argument("id")
        x.add_argument("-n", type=int, default=80)
    jw = jsub.add_parser("wait")
    jw.add_argument("id")
    jw.add_argument("--timeout", type=float, default=600)
    jw.add_argument("--interval", type=float, default=30)
    jw.add_argument("-n", type=int, default=80)
    pl = sub.add_parser("pull")
    pl.add_argument("id", nargs="?")
    pl.add_argument("--path")
    pl.add_argument("--into")
    sr = sub.add_parser("slot-rm")
    sr.add_argument("slot")
    b = sub.add_parser("bootstrap")
    b.add_argument("--update-gate", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.verb == "doctor":
        return do_doctor(a)
    if a.verb == "devcheck":
        return do_simple("devcheck")
    if a.verb == "run":
        return do_run(a)
    if a.verb == "exec":
        return do_exec(a)
    if a.verb == "job":
        return do_job(a)
    if a.verb == "pull":
        return do_pull(a)
    if a.verb == "slot-rm":
        return do_simple("slot_rm", {"slot": a.slot})
    if a.verb == "bootstrap":
        return do_bootstrap(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
