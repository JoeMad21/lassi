#!/usr/bin/env python3
"""Write provenance.json for a local run on the workstation (task P5.15; OQ-055, decision 6 (b)).

    python3 tools/local/provenance.py --out <path>/provenance.json --device cpu|ttsim
        (--job <job.json> | --cmd <text> --rc <n> --start <iso> --end <iso>) [--repo <clone>]

It stands in for the provenance.json that `rx pull` copies from the build host's gate (tools/rx.py, pull;
tools/server/gate.py, base_meta, host_info, and read_pins) when a run happens in the LASSI distro (AGENTS.md,
Results, local runs; Agent Rules 1 and 2). It runs under the distro's system python3.10, standard library only,
after the host profile (tools/local/host-profile.sh) is sourced.

The record keeps the gate's key names where the meaning is the same: cmd, cwd, commit, dirty, branch, start,
end, rc, host, and pins, and in host: hostname, os, kernel, nproc, mem_gb (gate.py's unit and rounding), and
python. It adds kind "local"; in host, kind "workstation", distro (WSL_DISTRO_NAME), os_version (VERSION_ID),
util_linux, and cpu_model; device; job (the job record and its path, or null); and date, the writer's own run
date in UTC. commit, branch, and dirty come from the clone's own git as tools/rx.py reads them (untracked files
count as dirty). The run comes from one of two routes:

- --job: a job record that tools/local/longjob.py wrote (command, exit_status, start, end, cwd, commit, dirty,
  host). The record's cwd is the job's; its command words are joined with shell quoting. A record whose commit
  is not a 40-character lowercase hex string, or whose dirty is not true or false, is refused: the job's tree
  state was not recorded (longjob.py writes both null when the job's cwd is not a git checkout, and dirty null
  when git status fails), so the job runs again from inside the clone. A job that ran at another commit than
  the clone's HEAD is refused, and one that ran dirty makes the record dirty.
- --cmd, --rc, --start, and --end, all four; cwd is then the clone. This route cannot check that the run happened
  at the clone's HEAD and tree state when the writer runs, which the record takes as the run's; "job": null marks
  it, so a [MEASURED] result names the job route or says how the run held that commit.

Differences from the gate's record: pins hold the full text of every toolchains/*.pin of the clone, where the
gate's read_pins truncates each pin at 4000 characters and skips one it cannot read (this writer refuses one);
and every host fact must be readable, or the writer refuses rather than leave a key out.

Refusals (a message on stderr, exit status 2, nothing written): LASSI_HOST is not "workstation"; LASSI_SCRATCH
is unset, empty, or relative; the clone, --out, --job, or the job's cwd lies outside LASSI_SCRATCH (resolved
paths, links followed); a host fact, the clone's git, or a pin cannot be read; and a record that would name
alpha01 anywhere. That last check covers every value the writer puts in the record except the verbatim pin
texts: committed pins name alpha01 in their comments as where a pin value was measured (cuda.pin, gcc.pin,
hipcc.pin, tt-metal.pin), and the full text is kept as committed. argparse's usage errors also exit 2.

The ttsim device is a simulator on the host CPU, never silicon, and its timing is never performance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Union

HOST_KIND = "workstation"
DEVICES = ("cpu", "ttsim")
FORBIDDEN = "alpha01"
PROFILE_HINT = "source tools/local/host-profile.sh in the LASSI distro first"
UTIL_LINUX_COMMANDS = (("findmnt", "--version"), ("mount", "--version"))
UTIL_LINUX_VERSION = re.compile(r"util-linux\s+([0-9][0-9A-Za-z.~+-]*)")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
NOT_RECORDED = (
    "the job's tree state was not recorded (tools/local/longjob.py writes commit and dirty null when the job's cwd "
    "is not a git checkout, and dirty null when git status fails): run the job from inside the clone"
)
TTSIM_NOTE = (
    "ttsim is a simulator running on the host CPU: its results are simulator results, never silicon, "
    "and its timing is never performance (Agent Rule 2)."
)

PathText = Union[str, "os.PathLike[str]"]
GitRunner = Callable[[str, Sequence[str]], str]


class Refusal(Exception):
    """A reason the writer will not write the record; main prints it and returns 2."""


def run_version_command(argv: Sequence[str]) -> str:
    """Return the stdout of a version command such as `findmnt --version`; raise OSError when it cannot run."""
    words = [str(word) for word in argv]
    try:
        done = subprocess.run(words, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.SubprocessError as error:
        raise OSError(f"{' '.join(words)} did not finish: {error}") from error
    if done.returncode != 0:
        raise OSError(f"{' '.join(words)} exited {done.returncode}: {done.stderr.strip()}")
    return done.stdout


def run_git(repo: str, args: Sequence[str]) -> str:
    """Run git in `repo` without optional locks and return its stdout, stripped; raise OSError when it fails."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    argv = ["git", "-C", repo, *args]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60, check=False)
    except subprocess.SubprocessError as error:
        raise OSError(f"git {' '.join(args)} did not finish: {error}") from error
    if done.returncode != 0:
        raise OSError(f"git {' '.join(args)} in {repo} exited {done.returncode}: {done.stderr.strip()}")
    return done.stdout.strip()


@dataclass
class HostSources:
    """Where the writer reads the host: file paths and commands, each replaceable by a fake in tests.

    version_output(argv) returns a version command's stdout and raises OSError when it cannot run;
    hostname() returns the host's name; git(repo, args) runs git in the clone and returns its stdout.
    """

    os_release: PathText = "/etc/os-release"
    cpuinfo: PathText = "/proc/cpuinfo"
    meminfo: PathText = "/proc/meminfo"
    kernel_release: PathText = "/proc/sys/kernel/osrelease"
    version_output: Callable[[Sequence[str]], str] = run_version_command
    hostname: Callable[[], str] = platform.node
    git: GitRunner = run_git


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# The environment, the paths, and the clone


def check_host(environ: Mapping[str, str]) -> None:
    """Refuse unless the host profile named this host the workstation."""
    value = environ.get("LASSI_HOST")
    if value != HOST_KIND:
        shown = "unset" if value is None else repr(value)
        raise Refusal(f"LASSI_HOST is {shown}, not {HOST_KIND!r}: this writer runs only on the workstation; "
                      f"{PROFILE_HINT}")


def scratch_root(environ: Mapping[str, str]) -> str:
    """The resolved scratch root from LASSI_SCRATCH, which has no default."""
    value = environ.get("LASSI_SCRATCH", "")
    if not value:
        raise Refusal(f"LASSI_SCRATCH is unset or empty: {PROFILE_HINT}")
    if not os.path.isabs(value):
        raise Refusal(f"LASSI_SCRATCH ({value}) is not an absolute path: {PROFILE_HINT}")
    return os.path.realpath(value)


def inside(root: str, path: str) -> bool:
    """True when the resolved `path` is the resolved `root` or lies inside it."""
    root_key, path_key = os.path.normcase(root), os.path.normcase(path)
    try:
        return os.path.commonpath([root_key, path_key]) == root_key
    except ValueError:  # different drives on Windows
        return False


def require_inside(root: str, path: str, what: str) -> str:
    """Resolve `path` (links followed) and refuse it unless it lies under the scratch root `root`."""
    resolved = os.path.realpath(path)
    if not inside(root, resolved):
        raise Refusal(f"{what} ({path}, resolved to {resolved}) lies outside LASSI_SCRATCH ({root}): a local "
                      "run's tree and its record stay under the scratch root")
    return resolved


def clone_top(git: GitRunner, repo: str | None) -> str:
    """The resolved git top level of --repo, or of the cwd when --repo is not given."""
    start = repo if repo is not None else os.getcwd()
    try:
        top = git(start, ["rev-parse", "--show-toplevel"])
    except OSError as error:
        where = f"--repo ({start})" if repo is not None else f"the cwd ({start})"
        raise Refusal(f"{where} is not inside a git clone; pass --repo with the clone: {error}") from error
    return os.path.realpath(top)


def clone_state(git: GitRunner, clone: str) -> dict[str, Any]:
    """The clone's commit, branch, and dirty state from its own git, read as tools/rx.py reads them."""
    try:
        commit = git(clone, ["rev-parse", "--verify", "HEAD"])
        status = git(clone, ["status", "--porcelain", "--untracked-files=normal"])
    except OSError as error:
        raise Refusal(f"cannot read the clone's git state: {error}") from error
    try:
        branch = git(clone, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    except OSError:
        branch = ""
    return {"commit": commit, "branch": branch or f"detached-{commit[:8]}", "dirty": bool(status)}


# ---------------------------------------------------------------------------
# The host


def read_text(path: PathText, what: str) -> str:
    """The text of a host file; refuse when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as error:
        raise Refusal(f"cannot read {what} ({os.fspath(path)}): {error}") from error


def fields(text: str, separator: str) -> dict[str, str]:
    """The first value of each `key<separator>value` line of `text`, both stripped, outer quotes removed."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(separator)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if sep and key.strip() not in found:
            found[key.strip()] = value
    return found


def required(found: Mapping[str, str], key: str, path: PathText) -> str:
    """found[key], refusing when it is missing or empty."""
    value = found.get(key, "")
    if not value:
        raise Refusal(f"{os.fspath(path)} has no {key}")
    return value


def util_linux_version(version_output: Callable[[Sequence[str]], str]) -> str:
    """util-linux's version from `findmnt --version`, else from `mount --version`."""
    problems = []
    for argv in UTIL_LINUX_COMMANDS:
        shown = " ".join(argv)
        try:
            text = version_output(list(argv))
        except OSError as error:
            problems.append(f"{shown}: {error}")
            continue
        match = UTIL_LINUX_VERSION.search(text)
        if match:
            return match.group(1)
        problems.append(f"{shown}: no util-linux version in {text.strip()!r}")
    raise Refusal("cannot read util-linux's version: " + "; ".join(problems))


def mem_gb(meminfo: Mapping[str, str], path: PathText) -> float:
    """MemTotal in gate.py host_info's unit and rounding: kB / 1e6, one decimal."""
    text = required(meminfo, "MemTotal", path)
    try:
        return round(int(text.split()[0]) / 1e6, 1)
    except (IndexError, ValueError) as error:
        raise Refusal(f"{os.fspath(path)} has an unreadable MemTotal ({text!r})") from error


def host_block(sources: HostSources, environ: Mapping[str, str]) -> dict[str, Any]:
    """The host: the workstation's name, the distro, the OS, the kernel, util-linux, the CPU, memory, python."""
    os_release = fields(read_text(sources.os_release, "the OS release file"), "=")
    cpuinfo = fields(read_text(sources.cpuinfo, "the CPU information"), ":")
    meminfo = fields(read_text(sources.meminfo, "the memory information"), ":")
    kernel = read_text(sources.kernel_release, "the kernel release").strip()
    if not kernel:
        raise Refusal(f"{os.fspath(sources.kernel_release)} is empty")
    distro = environ.get("WSL_DISTRO_NAME", "")
    if not distro:
        raise Refusal("WSL_DISTRO_NAME is unset or empty: run the writer inside the LASSI distro through wsl.exe")
    return {
        "kind": HOST_KIND,
        "hostname": sources.hostname(),
        "distro": distro,
        "os": required(os_release, "PRETTY_NAME", sources.os_release),
        "os_version": required(os_release, "VERSION_ID", sources.os_release),
        "kernel": kernel,
        "util_linux": util_linux_version(sources.version_output),
        "nproc": os.cpu_count(),
        "cpu_model": required(cpuinfo, "model name", sources.cpuinfo),
        "mem_gb": mem_gb(meminfo, sources.meminfo),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Pins and the device


def read_pins(clone: str) -> dict[str, str]:
    """The full text of every toolchains/*.pin of the clone, keyed by file name, as committed.

    tools/server/gate.py's read_pins truncates each pin at 4000 characters and skips a pin it cannot read;
    this writer keeps every character (newlines as they are) and refuses an unreadable pin.
    """
    folder = os.path.join(clone, "toolchains")
    try:
        names = sorted(name for name in os.listdir(folder) if name.endswith(".pin") and not name.startswith("."))
    except OSError as error:
        raise Refusal(f"cannot list the clone's pins ({folder}): {error}") from error
    pins: dict[str, str] = {}
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                pins[name] = handle.read()
        except (OSError, UnicodeDecodeError) as error:
            raise Refusal(f"cannot read the pin {path}: {error}") from error
    if not pins:
        raise Refusal(f"the clone has no toolchains/*.pin ({folder}); is it a LASSI clone?")
    return pins


def pin_value(pins: Mapping[str, str], name: str, key: str) -> str:
    """The last `key=value` assignment of pin `name`, as the shell would source it, without a trailing comment."""
    if name not in pins:
        raise Refusal(f"--device ttsim needs toolchains/{name} in the clone")
    value = None
    for line in pins[name].splitlines():
        left, sep, right = line.partition("=")
        if sep and left == key:
            value = right
    value = "" if value is None else re.split(r"\s#", value, maxsplit=1)[0].strip().strip("\"'")
    if not value:
        raise Refusal(f"toolchains/{name} sets no {key}")
    return value


def device_block(device: str, pins: Mapping[str, str], cpu_model: str) -> dict[str, Any]:
    """The device: the host CPU with its model, or ttsim named from the clone's ttsim and tt-metal pins."""
    if device == "cpu":
        return {"kind": "cpu", "model": cpu_model, "simulator": False}
    version = pin_value(pins, "ttsim.pin", "VERSION")
    commit = pin_value(pins, "tt-metal.pin", "COMMIT")
    if not COMMIT_RE.fullmatch(commit):
        raise Refusal(f"toolchains/tt-metal.pin's COMMIT ({commit}) is not a 40-character commit")
    return {"kind": "ttsim", "name": f"ttsim {version} on tt-metal {commit[:8]}", "simulator": True,
            "host_cpu": cpu_model, "note": TTSIM_NOTE}


# ---------------------------------------------------------------------------
# The run: a job record or the explicit values


def checked_date(value: Any, what: str) -> str:
    """`value` unchanged when it is an ISO 8601 date with a UTC offset (Z allowed); refuse it otherwise."""
    if not isinstance(value, str) or not value:
        raise Refusal(f"{what} is missing")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise Refusal(f"{what} ({value}) is not an ISO 8601 date: {error}") from error
    if parsed.utcoffset() is None:
        raise Refusal(f"{what} ({value}) carries no UTC offset")
    return value


def explicit_run(args: argparse.Namespace, clone: str) -> dict[str, Any]:
    """The run from --cmd, --rc, --start, and --end; its cwd is the clone."""
    if not args.cmd.strip():
        raise Refusal("--cmd is empty")
    return {"cmd": args.cmd, "cwd": clone, "rc": args.rc, "start": checked_date(args.start, "--start"),
            "end": checked_date(args.end, "--end"), "dirty": False, "job": None}


def read_job(path: str) -> dict[str, Any]:
    """The parsed job record at `path`."""
    try:
        with open(path, encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, ValueError) as error:
        raise Refusal(f"cannot read the job record --job ({path}): {error}") from error
    if not isinstance(job, dict):
        raise Refusal(f"the job record --job ({path}) is not a JSON object")
    return job


def job_command(job: Mapping[str, Any]) -> str:
    """The job's command as one string: its words joined with shell quoting."""
    command = job.get("command")
    if isinstance(command, list) and command and all(isinstance(word, str) for word in command):
        return shlex.join(command)
    if isinstance(command, str) and command.strip():
        return command
    raise Refusal("the job record has no command (a list of words, as tools/local/longjob.py writes it)")


def job_run(job_path: str, scratch: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """The run from a tools/local/longjob.py record; refuse one that does not belong to this clone and host."""
    job = read_job(job_path)
    if job.get("host") != HOST_KIND:
        raise Refusal(f"the job record's host is {job.get('host')!r}, not {HOST_KIND!r}")
    status = job.get("exit_status")
    if not isinstance(status, int) or isinstance(status, bool):
        raise Refusal(f"the job record's exit_status ({status!r}) is not an integer")
    cwd = job.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise Refusal("the job record has no cwd")
    return {"cmd": job_command(job), "cwd": require_inside(scratch, cwd, "the job's cwd"), "rc": status,
            "start": checked_date(job.get("start"), "the job's start"),
            "end": checked_date(job.get("end"), "the job's end"),
            "dirty": job_tree_state(job, state["commit"]), "job": {"path": job_path, "record": job}}


def job_tree_state(job: Mapping[str, Any], clone_commit: str) -> bool:
    """The job's dirty state; refuse a record whose tree state was not read, or one at another commit.

    longjob.py's git_state writes commit and dirty null when the job's cwd is not a git checkout, and dirty null
    when git status fails; neither may read as the clone's commit or as clean (Agent Rule 1).
    """
    commit, dirty = job.get("commit"), job.get("dirty")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise Refusal(f"the job record's commit ({json.dumps(commit)}) is not a 40-character lowercase hex commit: "
                      f"{NOT_RECORDED}")
    if not isinstance(dirty, bool):
        raise Refusal(f"the job record's dirty ({json.dumps(dirty)}) is not true or false: {NOT_RECORDED}")
    if commit != clone_commit:
        raise Refusal(f"the job ran at commit {commit}, but the clone is at {clone_commit}: write the record "
                      "from the clone at the job's commit")
    return dirty


# ---------------------------------------------------------------------------
# The record


def alpha01_places(value: Any, where: str = "") -> list[str]:
    """The key paths in `value` whose key or string names alpha01, in any case."""
    if isinstance(value, Mapping):
        found = []
        for key, item in value.items():
            place = f"{where}.{key}" if where else str(key)
            if FORBIDDEN in str(key).lower():
                found.append(place)
            found += alpha01_places(item, place)
        return found
    if isinstance(value, (list, tuple)):
        return [place for index, item in enumerate(value) for place in alpha01_places(item, f"{where}[{index}]")]
    return [where] if isinstance(value, str) and FORBIDDEN in value.lower() else []


def refuse_alpha01(record: Mapping[str, Any]) -> None:
    """Refuse a record that names alpha01 anywhere outside the verbatim pin texts (their names are checked)."""
    scanned = {key: value for key, value in record.items() if key != "pins"}
    scanned["pins"] = sorted(record["pins"])
    places = alpha01_places(scanned)
    if places:
        raise Refusal(f"the record would name alpha01 (at {', '.join(places)}): a local record names the "
                      "workstation and never alpha01 (AGENTS.md, Results)")


def build_record(args: argparse.Namespace, environ: Mapping[str, str], sources: HostSources) -> tuple[str, dict]:
    """Check the host and the paths, then gather the record; return the resolved --out and the record."""
    check_host(environ)
    scratch = scratch_root(environ)
    clone = require_inside(scratch, clone_top(sources.git, args.repo),
                           "the clone (--repo, by default the cwd's git top level)")
    out = require_inside(scratch, args.out, "--out")
    state = clone_state(sources.git, clone)
    if args.job is not None:
        run = job_run(require_inside(scratch, args.job, "--job"), scratch, state)
    else:
        run = explicit_run(args, clone)
    host = host_block(sources, environ)
    pins = read_pins(clone)
    record = {
        "kind": "local", "cmd": run["cmd"], "cwd": run["cwd"], "commit": state["commit"],
        "branch": state["branch"], "dirty": state["dirty"] or run["dirty"], "start": run["start"],
        "end": run["end"], "rc": run["rc"], "host": host, "pins": pins,
        "device": device_block(args.device, pins, host["cpu_model"]), "job": run["job"],
        "date": utc_now().isoformat(timespec="seconds"),
    }
    refuse_alpha01(record)
    return out, record


def write_record(out: str, record: Mapping[str, Any]) -> None:
    """Write the record as plain ASCII JSON, through a temporary file beside it, as the gate writes meta.json."""
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = out + ".tmp"
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(temporary, "w", encoding="ascii", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, out)
    except OSError as error:
        raise Refusal(f"cannot write --out ({out}): {error}") from error


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line; the job route and the explicit route exclude each other."""
    parser = argparse.ArgumentParser(prog="provenance.py", description="Write provenance.json for a local run.")
    parser.add_argument("--out", required=True, help="the provenance.json to write, under LASSI_SCRATCH")
    parser.add_argument("--device", required=True, choices=DEVICES, help="the host CPU, or ttsim on it")
    parser.add_argument("--job", help="a job.json that tools/local/longjob.py wrote")
    parser.add_argument("--cmd", help="the command that ran (explicit route)")
    parser.add_argument("--rc", type=int, help="its exit status (explicit route)")
    parser.add_argument("--start", help="its start, ISO 8601 with a UTC offset (explicit route)")
    parser.add_argument("--end", help="its end, ISO 8601 with a UTC offset (explicit route)")
    parser.add_argument("--repo", help="the clone, under LASSI_SCRATCH (default: the cwd's git top level)")
    args = parser.parse_args(argv)
    explicit = (args.cmd, args.rc, args.start, args.end)
    if args.job is not None and any(value is not None for value in explicit):
        parser.error("--job excludes --cmd, --rc, --start, and --end")
    if args.job is None and any(value is None for value in explicit):
        parser.error("give --job, or all four of --cmd, --rc, --start, and --end")
    return args


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         sources: HostSources | None = None) -> int:
    """Write --out for one local run; return 0, or 2 after a refusal printed on stderr."""
    args = parse_args(argv)
    try:
        out, record = build_record(args, os.environ if environ is None else environ, sources or HostSources())
        write_record(out, record)
    except Refusal as refusal:
        print(f"provenance.py: refused: {refusal}", file=sys.stderr)
        return 2
    print(f"provenance.py: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
