#!/usr/bin/env python3
"""Project gate on the remote build host.

Every remote action from tools/rx.py arrives here as one base64-encoded JSON request:

    python3 <scratch>/lassi-gate/gate.py <b64-json>

The gate checks out pushed commits into disposable worktree slots, runs commands with a
scratch-only environment, enforces disk, concurrency, device, and command policy, records
provenance for every run, and appends every call to an audit log. The owner can halt all
mutating verbs by creating <scratch>/lassi-gate/STOP.

Layout under the scratch root (default: parent of this file's directory):
    lassi-gate/            gate.py, config.json (owner-edited), audit.log, runs/<id>/, STOP
    lassi.git/             bare repository; accepts refs/wip/* only (hooks/pre-receive)
    lassi-wt/<slot>/       worktree slots; never edit by hand
    lassi-runs/            raw run trees written by the pipeline
    toolchains/            installed toolchains, <name>@<pin>/
    tmp/, .cache/          scratch temp and caches

Standard library only; compatible with Python 3.6+.
"""

import base64
import datetime
import glob
import io
import json
import os
import platform
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import time

GATE_VERSION = 1
GATE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("LASSI_SCRATCH") or os.path.dirname(GATE_DIR)
BARE = os.path.join(SCRATCH, "lassi.git")
WT = os.path.join(SCRATCH, "lassi-wt")
RUNS = os.path.join(GATE_DIR, "runs")
RUNS_ROOT = os.path.join(SCRATCH, "lassi-runs")
TOOLCHAINS = os.path.join(SCRATCH, "toolchains")
TMP = os.path.join(SCRATCH, "tmp")
STOP_FILE = os.path.join(GATE_DIR, "STOP")
AUDIT = os.path.join(GATE_DIR, "audit.log")
BUSY = os.path.join(WT, ".busy")
SLOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9._-]{1,40}-[0-9a-f]{4}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_TAR_BYTES = 200 * 1024 * 1024

DEFAULT_CONFIG = {
    "min_free_gb": 5,
    "min_free_gb_big": 60,
    "max_jobs": 3,
    "max_big_jobs": 1,
    "run_timeout_default_s": 1800,
    "run_timeout_max_s": 7200,
    "job_timeout_max_s": 172800,
    "max_build_jobs": 32,
    "nice": 10,
    "deny_patterns": [],
    "devices": {},
    "tt_guard": True,
}


class GateError(Exception):
    """A refused or failed request; the message goes back to the client."""


# ---------------------------------------------------------------- utilities


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    path = os.path.join(GATE_DIR, "config.json")
    if not os.path.isfile(path):
        path = os.path.join(GATE_DIR, "config.default.json")
    if os.path.isfile(path):
        with open(path) as fh:
            cfg.update(json.load(fh))
    return cfg


def audit(entry):
    entry = dict(entry)
    entry["ts"] = now_iso()
    try:
        with open(AUDIT, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass


def free_gb(path):
    try:
        st = os.statvfs(path)
    except OSError:
        return -1.0
    return round(st.f_bavail * st.f_frsize / 1e9, 1)


def sh(args, cwd=None, env=None, timeout=60):
    """Run a short command and return (rc, combined output)."""
    try:
        p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, universal_newlines=True)
        return p.returncode, p.stdout
    except FileNotFoundError:
        return 127, "not found: %s" % args[0]
    except subprocess.TimeoutExpired:
        return 124, "timeout after %ss" % timeout


def git(*args, **kw):
    rc, out = sh(["git"] + list(args), **kw)
    if rc != 0:
        raise GateError("git %s failed: %s" % (" ".join(args), out.strip()[-800:]))
    return out


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def new_id(slot):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = re.sub(r"[^A-Za-z0-9._-]", "-", slot)[:40] or "x"
    return "%s-%s-%04x" % (stamp, short, random.randint(0, 0xFFFF))


def read_meta(run_id):
    path = os.path.join(RUNS, run_id, "meta.json")
    if not os.path.isfile(path):
        raise GateError("unknown run or job id %s" % run_id)
    with open(path) as fh:
        return json.load(fh)


def write_meta(run_dir, meta):
    tmp = os.path.join(run_dir, "meta.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(run_dir, "meta.json"))


def host_info():
    info = {"hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
            "nproc": os.cpu_count()}
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    info["os"] = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["mem_gb"] = round(int(line.split()[1]) / 1e6, 1)
                if line.startswith("MemAvailable:"):
                    info["mem_avail_gb"] = round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    try:
        info["load"] = list(os.getloadavg())
    except OSError:
        pass
    return info


# ---------------------------------------------------------------- policy


def check_stop():
    if os.path.exists(STOP_FILE):
        raise GateError("owner STOP is set (%s); mutating verbs are refused" % STOP_FILE)


def check_disk(cfg, big):
    need = cfg["min_free_gb_big"] if big else cfg["min_free_gb"]
    have = free_gb(SCRATCH)
    if have < need:
        raise GateError("scratch free space %.1f GB is below the %s GB floor%s" %
                        (have, need, " for --big" if big else ""))
    return have


def check_command(cfg, cmd):
    for pat in cfg.get("deny_patterns", []):
        if re.search(pat, cmd):
            raise GateError("command refused by policy pattern /%s/" % pat)
    for name, dev in sorted(cfg.get("devices", {}).items()):
        if dev.get("enabled"):
            continue
        for pat in dev.get("patterns", []):
            if re.search(pat, cmd):
                raise GateError("device class '%s' is disabled by the owner; command matches /%s/. "
                                "Record the need in plans/OWNER-QUEUE.md" % (name, pat))
    tt_dev = cfg.get("devices", {}).get("tt_silicon", {})
    if cfg.get("tt_guard", True) and not tt_dev.get("enabled") and glob.glob("/dev/tenstorrent*"):
        if re.search(r"tt-metal|tt_metal|ttnn|metal_example|TT_METAL", cmd) and "TT_METAL_SIMULATOR" not in cmd:
            raise GateError("Tenstorrent silicon is present but disabled; set TT_METAL_SIMULATOR in the "
                            "command to run on ttsim")


def running_jobs():
    jobs = []
    if not os.path.isdir(RUNS):
        return jobs
    for d in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, d, "meta.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if meta.get("kind") == "job" and meta.get("state") in ("starting", "running"):
            if meta.get("runner_pid") and not pid_alive(meta.get("runner_pid")):
                meta["state"] = "lost"
                write_meta(os.path.join(RUNS, d), meta)
                continue
            jobs.append(meta)
    return jobs


def check_concurrency(cfg, big):
    jobs = running_jobs()
    if len(jobs) >= cfg["max_jobs"]:
        raise GateError("%d jobs running (max %d): %s" % (len(jobs), cfg["max_jobs"],
                                                         ", ".join(j["id"] for j in jobs)))
    bigs = [j for j in jobs if j.get("big")]
    if big and len(bigs) >= cfg["max_big_jobs"]:
        raise GateError("big job already running: %s" % ", ".join(j["id"] for j in bigs))


def clamp_timeout(cfg, value, kind):
    limit = cfg["job_timeout_max_s"] if kind == "job" else cfg["run_timeout_max_s"]
    t = int(value or cfg["run_timeout_default_s"])
    return max(10, min(t, limit))


# ---------------------------------------------------------------- slots


def slot_busy(slot):
    path = os.path.join(BUSY, slot)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    if pid_alive(data.get("pid")):
        return data
    os.remove(path)
    return None


def mark_busy(slot, owner_id, pid):
    os.makedirs(BUSY, exist_ok=True)
    with open(os.path.join(BUSY, slot), "w") as fh:
        json.dump({"id": owner_id, "pid": pid}, fh)


def clear_busy(slot, owner_id):
    path = os.path.join(BUSY, slot)
    try:
        with open(path) as fh:
            if json.load(fh).get("id") == owner_id:
                os.remove(path)
    except (OSError, ValueError):
        pass


def prepare_slot(slot, commit):
    """Check out commit (detached) into the slot worktree; keep ignored build outputs."""
    if not SLOT_RE.match(slot or ""):
        raise GateError("bad slot name %r" % slot)
    if not SHA_RE.match(commit or ""):
        raise GateError("bad commit %r" % commit)
    busy = slot_busy(slot)
    if busy:
        raise GateError("slot %s is busy with %s; use another --slot or wait" % (slot, busy["id"]))
    git("--git-dir=" + BARE, "cat-file", "-e", commit + "^{commit}")
    path = os.path.join(WT, slot)
    if not os.path.isdir(path):
        os.makedirs(WT, exist_ok=True)
        git("--git-dir=" + BARE, "worktree", "prune")
        git("--git-dir=" + BARE, "worktree", "add", "--detach", "--force", path, commit, timeout=600)
    else:
        git("-C", path, "checkout", "--detach", "--force", commit, timeout=600)
        git("-C", path, "clean", "-fd", "-q", timeout=600)
    return path


def read_pins(path):
    pins = {}
    for pin in sorted(glob.glob(os.path.join(path, "toolchains", "*.pin"))):
        try:
            with open(pin) as fh:
                pins[os.path.basename(pin)] = fh.read()[:4000]
        except OSError:
            pass
    return pins


def build_env(cfg, extra):
    env = dict(os.environ)
    jobs = max(1, min(int(cfg["max_build_jobs"]), (os.cpu_count() or 2) // 2))
    for sub in ("tmp", ".cache", "bin"):
        os.makedirs(os.path.join(SCRATCH, sub), exist_ok=True)
    env.update({
        "LASSI_SCRATCH": SCRATCH,
        "LASSI_RUNS_ROOT": RUNS_ROOT,
        "LASSI_TOOLCHAINS": TOOLCHAINS,
        "LASSI_JOBS": str(jobs),
        "TMPDIR": TMP, "TMP": TMP, "TEMP": TMP,
        "XDG_CACHE_HOME": os.path.join(SCRATCH, ".cache"),
        "UV_CACHE_DIR": os.path.join(SCRATCH, ".cache", "uv"),
        "PIP_CACHE_DIR": os.path.join(SCRATCH, ".cache", "pip"),
        "HF_HOME": os.path.join(SCRATCH, "hf"),
        "CCACHE_DIR": os.path.join(SCRATCH, ".cache", "ccache"),
        "CARGO_HOME": os.path.join(SCRATCH, ".cargo"),
        "RUSTUP_HOME": os.path.join(SCRATCH, ".rustup"),
        "DOTNET_CLI_HOME": os.path.join(SCRATCH, ".dotnet-home"),
        "NUGET_PACKAGES": os.path.join(SCRATCH, ".nuget", "packages"),
        "npm_config_cache": os.path.join(SCRATCH, ".cache", "npm"),
        "PATH": os.path.join(SCRATCH, "bin") + os.pathsep + env.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
    })
    env.update(extra)
    return env


def wrap(cfg, cmd):
    prefix = ["nice", "-n", str(cfg["nice"])]
    if shutil.which("ionice"):
        prefix = ["ionice", "-c2", "-n7"] + prefix
    return prefix + ["bash", "-c", cmd]


def base_meta(req, kind, run_id, cfg):
    return {
        "id": run_id, "kind": kind, "gate_version": GATE_VERSION, "cmd": req.get("cmd", ""),
        "slot": req.get("slot"), "commit": req.get("commit"), "branch": req.get("branch"),
        "dirty": bool(req.get("dirty")), "snapshot_of": req.get("snapshot_of"),
        "machine": req.get("machine"), "big": bool(req.get("big")),
        "timeout_s": clamp_timeout(cfg, req.get("timeout"), kind), "host": host_info(),
        "scratch_free_gb_before": free_gb(SCRATCH), "root_free_gb": free_gb("/"),
        "start": None, "end": None, "rc": None, "state": "starting",
    }


# ---------------------------------------------------------------- execution


def stream_process(argv, cwd, env, log_path, timeout, echo):
    """Run argv, tee output to log (and stdout when echo), enforce timeout. Returns rc."""
    out = sys.stdout.buffer if echo else None
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        fd = proc.stdout.fileno()
        timed_out = False
        while True:
            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                if out is not None:
                    try:
                        out.write(chunk)
                        out.flush()
                    except (BrokenPipeError, OSError):
                        out = None
                continue
            if proc.poll() is not None:
                break
            if time.time() > deadline:
                timed_out = True
                kill_group(proc.pid)
                break
        rc = proc.wait()
        if timed_out:
            msg = "\n[gate] timeout after %ss; process group killed\n" % timeout
            log.write(msg.encode())
            if out is not None:
                try:
                    out.write(msg.encode())
                    out.flush()
                except OSError:
                    pass
            return 124
        return rc


def kill_group(pid, grace=10):
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return
    end = time.time() + grace
    while time.time() < end:
        try:
            os.killpg(pid, 0)
        except OSError:
            return
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def finish_meta(run_dir, meta, rc):
    meta.update({"rc": rc, "end": now_iso(), "state": "done" if rc == 0 else "failed",
                 "scratch_free_gb_after": free_gb(SCRATCH)})
    write_meta(run_dir, meta)


def verb_run(req, cfg):
    check_stop()
    big = bool(req.get("big"))
    check_disk(cfg, big)
    cmd = req.get("cmd") or ""
    check_command(cfg, cmd)
    slot = req.get("slot") or ""
    path = prepare_slot(slot, req.get("commit"))
    run_id = new_id(slot)
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir)
    meta = base_meta(req, "run", run_id, cfg)
    meta["pins"] = read_pins(path)
    meta["cwd"] = path
    meta["start"] = now_iso()
    meta["state"] = "running"
    write_meta(run_dir, meta)
    mark_busy(slot, run_id, os.getpid())
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    env = build_env(cfg, {"LASSI_SLOT": slot, "LASSI_COMMIT": req.get("commit"), "LASSI_RX_RUN_ID": run_id})
    try:
        rc = stream_process(wrap(cfg, cmd), path, env, os.path.join(run_dir, "output.log"),
                            meta["timeout_s"], echo=True)
    finally:
        clear_busy(slot, run_id)
    finish_meta(run_dir, meta, rc)
    return {"id": run_id, "rc": rc, "state": meta["state"]}


def verb_exec(req, cfg):
    check_stop()
    check_disk(cfg, False)
    cmd = req.get("cmd") or ""
    check_command(cfg, cmd)
    run_id = new_id("exec")
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir)
    meta = base_meta(req, "exec", run_id, cfg)
    meta.update({"cwd": SCRATCH, "start": now_iso(), "state": "running"})
    write_meta(run_dir, meta)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    rc = stream_process(wrap(cfg, cmd), SCRATCH, build_env(cfg, {"LASSI_RX_RUN_ID": run_id}),
                        os.path.join(run_dir, "output.log"), meta["timeout_s"], echo=True)
    finish_meta(run_dir, meta, rc)
    return {"id": run_id, "rc": rc, "state": meta["state"]}


def verb_job_start(req, cfg):
    check_stop()
    big = bool(req.get("big"))
    check_disk(cfg, big)
    check_concurrency(cfg, big)
    cmd = req.get("cmd") or ""
    check_command(cfg, cmd)
    slot = req.get("slot") or ""
    path = prepare_slot(slot, req.get("commit"))
    run_id = new_id(req.get("name") or slot)
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir)
    meta = base_meta(req, "job", run_id, cfg)
    meta.update({"pins": read_pins(path), "cwd": path, "name": req.get("name")})
    write_meta(run_dir, meta)
    runner = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--jobrun", run_dir],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True, close_fds=True)
    mark_busy(slot, run_id, runner.pid)
    meta["runner_pid"] = runner.pid
    write_meta(run_dir, meta)
    return {"id": run_id, "state": "starting", "slot": slot}


def jobrun(run_dir):
    """Detached job runner; owns the slot until the command exits."""
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    cfg = load_config()
    meta.update({"runner_pid": os.getpid(), "start": now_iso(), "state": "running"})
    write_meta(run_dir, meta)
    env = build_env(cfg, {"LASSI_SLOT": meta["slot"], "LASSI_COMMIT": meta["commit"],
                          "LASSI_RX_RUN_ID": meta["id"]})
    rc = 1
    try:
        rc = stream_process(wrap(cfg, meta["cmd"]), meta["cwd"], env, os.path.join(run_dir, "output.log"),
                            meta["timeout_s"], echo=False)
    finally:
        clear_busy(meta["slot"], meta["id"])
        with open(os.path.join(run_dir, "meta.json")) as fh:
            meta = json.load(fh)
        if meta.get("state") != "killed":
            finish_meta(run_dir, meta, rc)
    return rc


def tail_file(path, n):
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 256 * 1024))
        data = fh.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-n:])


def verb_job_status(req, cfg):
    running_jobs()
    ids = [req["id"]] if req.get("id") else sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []
    out = []
    for run_id in ids[-50:]:
        try:
            meta = read_meta(run_id)
        except GateError:
            if req.get("id"):
                raise
            continue
        if meta.get("kind") != "job" and not req.get("id"):
            continue
        out.append({k: meta.get(k) for k in ("id", "kind", "name", "state", "rc", "start", "end", "slot",
                                            "commit", "big", "cmd")})
    return {"jobs": out}


def verb_job_tail(req, cfg):
    meta = read_meta(req["id"])
    text = tail_file(os.path.join(RUNS, meta["id"], "output.log"), int(req.get("n") or 80))
    return {"id": meta["id"], "state": meta.get("state"), "rc": meta.get("rc"), "tail": text}


def verb_job_kill(req, cfg):
    meta = read_meta(req["id"])
    if meta.get("state") not in ("starting", "running"):
        return {"id": meta["id"], "state": meta.get("state"), "note": "not running"}
    pid = meta.get("runner_pid")
    if pid and pid_alive(pid):
        kill_group(int(pid))
    meta.update({"state": "killed", "end": now_iso()})
    write_meta(os.path.join(RUNS, meta["id"]), meta)
    clear_busy(meta.get("slot"), meta["id"])
    return {"id": meta["id"], "state": "killed"}


def verb_doctor(req, cfg):
    checks = {
        "gate_version": GATE_VERSION, "scratch": SCRATCH, "stop": os.path.exists(STOP_FILE),
        "scratch_free_gb": free_gb(SCRATCH), "root_free_gb": free_gb("/"), "host": host_info(),
        "bare_repo": os.path.isdir(BARE), "git": sh(["git", "--version"])[1].strip(),
        "uv": shutil.which("uv", path=os.path.join(SCRATCH, "bin") + os.pathsep + os.environ.get("PATH", "")),
        "config": {k: cfg[k] for k in ("min_free_gb", "min_free_gb_big", "max_jobs", "max_big_jobs",
                                       "max_build_jobs")},
        "devices_enabled": {k: bool(v.get("enabled")) for k, v in cfg.get("devices", {}).items()},
        "running_jobs": [j["id"] for j in running_jobs()],
        "slots": sorted(d for d in os.listdir(WT) if not d.startswith(".")) if os.path.isdir(WT) else [],
    }
    return checks


DEVCHECK = [
    ("tenstorrent_pci", ["lspci", "-d", "1e52:"]),
    ("furiosa_pci", ["lspci", "-d", "1ed2:"]),
    ("tenstorrent_dev", ["bash", "-c", "ls -l /dev/tenstorrent* 2>&1 || true"]),
    ("furiosa_smi_info", ["bash", "-c", "command -v furiosa-smi >/dev/null && furiosa-smi info 2>&1 || "
                                         "echo 'furiosa-smi not on PATH'"]),
    ("groups", ["id", "-nG"]),
    ("kfd", ["bash", "-c", "ls -l /dev/kfd 2>&1; python3 -c \"open('/dev/kfd')\" 2>&1 && echo kfd-open-ok"]),
    ("nvidia", ["bash", "-c", "command -v nvidia-smi >/dev/null && nvidia-smi -L 2>&1 || echo 'no nvidia-smi'"]),
]


def verb_devcheck(req, cfg):
    out = {}
    for name, argv in DEVCHECK:
        rc, text = sh(argv, timeout=30)
        out[name] = {"rc": rc, "out": text.strip()[-2000:]}
    out["checked_at"] = now_iso()
    out["host"] = platform.node()
    return out


def allowed_path(rel):
    """Resolve a pull path; only run records, raw run trees, and slot results are readable."""
    full = os.path.realpath(os.path.join(SCRATCH, rel))
    roots = [os.path.realpath(p) for p in (RUNS, RUNS_ROOT, WT)]
    if not any(full == r or full.startswith(r + os.sep) for r in roots):
        raise GateError("path %s is outside lassi-gate/runs, lassi-runs, and lassi-wt" % rel)
    if not os.path.exists(full):
        raise GateError("path %s does not exist" % rel)
    return full


def verb_tar(req, cfg):
    """Write a gzip tar of a run record or an allowed path to stdout."""
    if req.get("id"):
        meta = read_meta(req["id"])
        src = os.path.join(RUNS, meta["id"])
        arc = meta["id"]
    else:
        src = allowed_path(req.get("path") or "")
        arc = os.path.basename(src.rstrip(os.sep))
    total = 0
    for root, _, files in os.walk(src):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    if total > MAX_TAR_BYTES:
        raise GateError("refusing to pull %.1f MB; pull summaries, not raw trees" % (total / 1e6))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src, arcname=arc)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    return None


def verb_slot_rm(req, cfg):
    check_stop()
    slot = req.get("slot") or ""
    if not SLOT_RE.match(slot):
        raise GateError("bad slot name %r" % slot)
    if slot_busy(slot):
        raise GateError("slot %s is busy" % slot)
    path = os.path.join(WT, slot)
    if os.path.isdir(path):
        sh(["git", "--git-dir=" + BARE, "worktree", "remove", "--force", path], timeout=600)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        sh(["git", "--git-dir=" + BARE, "worktree", "prune"])
    return {"removed": slot, "scratch_free_gb": free_gb(SCRATCH)}


VERBS = {
    "ping": lambda req, cfg: {"gate": "lassi-gate", "version": GATE_VERSION, "stop": os.path.exists(STOP_FILE)},
    "doctor": verb_doctor,
    "devcheck": verb_devcheck,
    "run": verb_run,
    "exec": verb_exec,
    "job_start": verb_job_start,
    "job_status": verb_job_status,
    "job_tail": verb_job_tail,
    "job_kill": verb_job_kill,
    "tar": verb_tar,
    "slot_rm": verb_slot_rm,
}


def main(argv):
    if len(argv) == 3 and argv[1] == "--jobrun":
        return jobrun(argv[2])
    if len(argv) != 2:
        sys.stderr.write("usage: gate.py <base64-json-request>\n")
        return 2
    for d in (RUNS, WT, RUNS_ROOT, TOOLCHAINS, TMP):
        os.makedirs(d, exist_ok=True)
    try:
        req = json.loads(base64.b64decode(argv[1]).decode("utf-8"))
    except (ValueError, TypeError) as exc:
        sys.stderr.write("gate: bad request: %s\n" % exc)
        return 2
    verb = req.get("verb")
    cfg = load_config()
    entry = {"verb": verb, "machine": req.get("machine"), "slot": req.get("slot"),
             "commit": req.get("commit"), "cmd": (req.get("cmd") or "")[:500], "id": req.get("id")}
    if verb not in VERBS:
        audit(dict(entry, result="unknown verb"))
        sys.stderr.write("gate: unknown verb %r\n" % verb)
        return 2
    if verb in ("run", "exec", "job_start", "job_kill", "slot_rm"):
        audit(dict(entry, result="start"))
    try:
        result = VERBS[verb](req, cfg)
    except GateError as exc:
        audit(dict(entry, result="refused", reason=str(exc)))
        sys.stdout.write("RX-RESULT " + json.dumps({"ok": False, "error": str(exc)}) + "\n")
        sys.stdout.flush()
        return 3
    audit(dict(entry, result="ok", rc=(result or {}).get("rc"), run_id=(result or {}).get("id")))
    if result is not None:
        sys.stdout.write("\nRX-RESULT " + json.dumps(dict(result, ok=True)) + "\n")
        sys.stdout.flush()
        return int(result.get("rc") or 0) if verb in ("run", "exec") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
