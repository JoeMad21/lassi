"""Helpers for the P4.9 spike batch, plans/spikes/p4-ttsim-runtime/run.sh (report: plans/spikes/p4-ttsim-runtime.md).

Standard library only, run by the build host's python3. Subcommands:

- measure: run one command in a new session under a wall limit, sample the tasks and resident memory of its
  process group once a second, stop the group at the limit (SIGTERM, then SIGKILL after 10 s), and write
  result.json with the exit status in the shell's form, the start and end times, the wall time, the peaks, and
  the fields of the /usr/bin/time -v report the command wrote to time.txt, when there is one.
- seed: copy a reference kernel into a run directory and insert one seeded line after its kernel_main line.
  The copy never lands in a protected tree (the pinned tt-metal, another project's checkout).
- strace: summarize an strace log of file, network, and exec calls by path class.
- getenv: summarize a getenv_log.c log (which variables each executable asked for, set or unset).
- summarize: build report/ and fixtures/ from the step directories, write the step table to report/summary.txt,
  and print the findings (the batch's stdout).
- quiet: succeed when a step failed fast without a ttsim or JIT error line (run.sh then reruns it line-buffered).

Every text file bound for report/ or fixtures/ (results/ and tests/ once pulled) is written as ASCII, with
any other character written as a backslash escape (write_ascii), and summarize ends by rewriting every file
there the same way (asciify_tree), which also covers the files run.sh writes. Every record that holds a wall
time names the device and says it is simulator wall time, exploratory, sizing only (Agent Rule 2).
Nothing here runs generated code, and nothing writes outside the directories its arguments name.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

# Seconds between two samples of the process group, and the grace between SIGTERM and SIGKILL at the limit.
SAMPLE_S = 1.0
KILL_GRACE_S = 10.0
# The seeded lines, inserted as their own line right after `void kernel_main() {` of a copy of the gate
# example's kernel (add_2_integers_in_riscv), so a compiler diagnostic quotes only the seeded line. Each
# targets a message found in libttsim_wh.so v1.3.4's strings (plans/spikes/p4-ttsim-runtime.md); which
# class ttsim gives each is what the run measures. Runtime argument 3 of that kernel is an L1 buffer address
# (get_arg_val<uint32_t>(3) in the kernel; the host program sets it, see local/source-excerpts.txt). Several
# gap candidates are offered so one job captures a gap class whichever way ttsim classes each instruction.
SEEDS = {
    "ub-unaligned": "    { volatile uint32_t* lassi_seed_p = reinterpret_cast<volatile uint32_t*>("
    "get_arg_val<uint32_t>(3) + 2); uint32_t lassi_seed_v = *lassi_seed_p; (void)lassi_seed_v; }"
    "  // LASSI P4.9 seed: a 4-byte load from an address 2 mod 4",
    "ub-fence": '    asm volatile(".word 0x0ff0000f" ::: "memory");  // LASSI P4.9 seed: fence iorw, iorw',
    "ub-fencei": '    asm volatile(".word 0x0000100f" ::: "memory");  // LASSI P4.9 seed: fence.i',
    "gap-zicsr": '    asm volatile(".word 0xb0002573" ::: "a0", "memory");  // LASSI P4.9 seed: csrrs a0, mcycle, zero',
    "gap-decode": '    asm volatile(".word 0xffffffff" ::: "memory");  // LASSI P4.9 seed: an undecodable word',
    "gap-ecall": '    asm volatile(".word 0x00000073" ::: "memory");  // LASSI P4.9 seed: ecall',
    "gap-wfi": '    asm volatile(".word 0x10500073" ::: "memory");  // LASSI P4.9 seed: wfi',
    "jit-error": "    lassi_seeded_jit_error();  // LASSI P4.9 seed: a call to an undeclared function",
    "hang": "    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3)), "
    "0x5a5a5a5au);  // LASSI P4.9 seed: wait for a value nothing writes",
}
# ttsim's error line is "[<n>] ERROR: <class>: <message>" (the format string in libttsim_wh.so v1.3.4).
SIM_ERROR = re.compile(r"ERROR: (UndefinedBehavior|UnimplementedFunctionality|UnsupportedFunctionality):?\s*(.*)")
# ttsim prefixes its own lines with a cycle count in brackets, "[<n>] ..."; these are run-time output.
SIM_LINE = re.compile(r"^\[\d+\]\s")
# ttsim's own rate line ("%.1f seconds (%.1f MHz)" in the library's strings): the simulated clock rate.
SIM_RATE = re.compile(r"\d+\.\d seconds \(\d+\.\d [MK]Hz\)")
# A generated unpack-descriptor value that means unpack_to_dest is on: true, a nonzero digit, or an Fp32 mode.
UNPACK_ENABLED = re.compile(r"\btrue\b|[1-9]|Fp32|FP32|UnpackToDestFp32", re.IGNORECASE)
# Compiler and backtrace lines that carry upstream text; write_fixture replaces them with a marker.
UPSTREAM_LINE = re.compile(r"TT_FATAL|TT_ASSERT|TT_THROW|backtrace|riscv-tt-elf-g\+\+")
JIT_ERROR = re.compile(r"build failed|: error: |fatal error", re.IGNORECASE)
PCC = re.compile(r"PCC = ([-+0-9.eEnaif]+)")
DEVICE = ("ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf; "
          "simulator, not silicon")
TIMING_NOTE = ("wall_s, program_s, and sim_rate_line are simulator wall time and the simulator's own rate: "
               "exploratory, sizing only, never performance")
# A compiler context line quoting a source line ("  6 |     text"); in a fixture only the seeded line may appear.
SOURCE_LINE = re.compile(r"^\s*\d+\s+\|\s?(.*)$")
SEED_MARK = "LASSI P4.9 seed"
TIME_FIELDS = {
    "Maximum resident set size (kbytes)": "maxrss_kib",
    "User time (seconds)": "user_s",
    "System time (seconds)": "system_s",
    "Exit status": "time_exit_status",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "time_elapsed",
}
OPEN_CALLS = {"open", "openat", "openat2", "creat"}
WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")
WRITE_CALLS = {
    "mkdir", "mkdirat", "unlink", "unlinkat", "rmdir", "rename", "renameat", "renameat2", "link", "linkat",
    "symlink", "symlinkat", "truncate", "chmod", "fchmodat", "chown", "fchownat", "lchown", "utimensat",
    "utime", "utimes", "mknod", "mknodat",
}
NET_CALLS = {"socket", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "socketpair"}
PRIVATE_DIRS = ("/tmp", "/var/tmp", "/dev/shm")
EXPECTED_READS = {"run", "tt-metal", "ttsim", "private", "/usr", "/lib", "/lib64", "/proc", "/sys", "/etc",
                  "/dev", "/bin", "/sbin"}
FIXTURE_CAP = 64 << 10
LOCAL_CAP = 1 << 20


def shell_status(returncode: int | None) -> int | None:
    """Return a Popen status in the shell's form: a death by signal N is 128 + N."""
    if returncode is None or returncode >= 0:
        return returncode
    return 128 - returncode


def group_usage(pgid: int) -> tuple[int, int, int]:
    """Return (tasks, the most threads of one process, resident KiB) summed over process group `pgid`.

    Reads /proc/<pid>/stat: after the command name, field 3 is the group, 18 the thread count, and 22 the
    resident pages. A process that ends while being read is skipped.
    """
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    tasks = top = rss = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            text = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        fields = text[text.rfind(")") + 2 :].split()
        if len(fields) < 22 or int(fields[2]) != pgid:
            continue
        threads = int(fields[17])
        tasks += threads
        top = max(top, threads)
        rss += int(fields[21]) * page_kib
    return tasks, top, rss


def stop_group(proc: subprocess.Popen) -> None:
    """Stop the process group led by `proc`: SIGTERM, then SIGKILL after KILL_GRACE_S; wait for the leader."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


def watch(proc: subprocess.Popen, limit_s: float, out: Path, stop_files: list[Path],
          max_bytes: int) -> tuple[dict, str]:
    """Sample `proc`'s process group until it exits, the limit passes, a stop file appears, or the output cap.

    Returns (peaks, reason), reason one of "": exited on its own, "limit", "stop-file", or "output-cap". On any
    of the last three the process group is stopped (SIGTERM, then SIGKILL after the grace).
    """
    deadline = time.monotonic() + limit_s
    peaks = {"peak_tasks": 0, "peak_process_threads": 0, "peak_group_rss_kib": 0, "samples": 0}
    while proc.poll() is None:
        reason = ""
        if time.monotonic() >= deadline:
            reason = "limit"
        elif any(path.exists() for path in stop_files):
            reason = "stop-file"
        elif sum((out / name).stat().st_size for name in ("stdout.txt", "stderr.txt")
                 if (out / name).exists()) > max_bytes:
            reason = "output-cap"
        if reason:
            stop_group(proc)
            return peaks, reason
        tasks, top, rss = group_usage(proc.pid)
        peaks["peak_tasks"] = max(peaks["peak_tasks"], tasks)
        peaks["peak_process_threads"] = max(peaks["peak_process_threads"], top)
        peaks["peak_group_rss_kib"] = max(peaks["peak_group_rss_kib"], rss)
        peaks["samples"] += 1
        try:
            proc.wait(timeout=SAMPLE_S)
        except subprocess.TimeoutExpired:
            pass
    return peaks, ""


def parse_time(path: Path) -> dict:
    """Return the /usr/bin/time -v fields of TIME_FIELDS (and a terminating signal) from `path`, or {}."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    found: dict = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Command terminated by signal"):
            found["time_signal"] = int(line.split()[-1])
        key, _, value = line.rpartition(": ")
        if key in TIME_FIELDS:
            name = TIME_FIELDS[key]
            found[name] = value if name == "time_elapsed" else float(value)
    return found


def step_line(result: dict) -> str:
    """Return the one stdout line a finished step prints; its wall time is labelled simulator wall time."""
    rss = result.get("maxrss_kib")
    rss_text = f"{rss / 1024:.0f}MiB" if isinstance(rss, (int, float)) else "?"
    reason = result.get("stopped_reason") or ""
    flag = f" [{reason}]" if reason else ""
    return (
        f"{result.get('name', '?'):<30} rc={result.get('rc')}{flag} sim_wall={result.get('wall_s')}s "
        f"maxrss={rss_text} tasks={result.get('peak_tasks')}"
    )


def now_iso() -> str:
    """Return the local time with its UTC offset, to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cmd_measure(args: argparse.Namespace) -> int:
    """Run args.argv in args.cwd under args.timeout seconds; write result.json, stdout.txt, stderr.txt.

    A SIGTERM or SIGHUP to this helper (the gate's job timeout signals the session group) is turned into a
    SIGKILL of the step's own process group, so no step outlives the helper. A stop file appearing or the output
    cap being passed stops the step too.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    stop_paths = [Path(p) for p in args.stop_file]
    proc_ref: dict[str, subprocess.Popen] = {}

    def relay(signum, _frame):
        popen = proc_ref.get("proc")
        if popen is not None:
            try:
                os.killpg(popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, relay)
    started_at, started = now_iso(), time.monotonic()
    with open(out / "stdout.txt", "wb") as stdout, open(out / "stderr.txt", "wb") as stderr:
        proc = subprocess.Popen(argv, cwd=args.cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        proc_ref["proc"] = proc
        peaks, reason = watch(proc, args.timeout, out, stop_paths, args.max_bytes)
    wall = time.monotonic() - started
    leftover, _, _ = group_usage(proc.pid)
    if leftover:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    result = {"name": args.name, "rc": shell_status(proc.returncode), "timed_out": reason == "limit",
              "stopped_reason": reason, "wall_s": round(wall, 2), "limit_s": args.timeout,
              "leftover_tasks_killed": leftover, "wrapper_reached_program": (out / "private-tmp.txt.ready").exists(),
              "started_at": started_at, "ended_at": now_iso(), "device": DEVICE, "timing_note": TIMING_NOTE,
              **peaks, **parse_time(out / "time.txt")}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(step_line(result), flush=True)
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Copy args.src to args.dst with SEEDS[args.seed] inserted after the kernel_main line; 2 on a refusal."""
    dst = Path(args.dst).resolve()
    for protected in args.protect:
        root = Path(protected).resolve()
        if dst == root or root in dst.parents:
            print(f"seed: refusing to write {dst} inside the protected tree {root}")
            return 2
    lines = Path(args.src).read_text(encoding="utf-8").split("\n")
    index = next((i for i, text in enumerate(lines)
                  if text.strip().startswith("void kernel_main()") and text.rstrip().endswith("{")), None)
    if index is None:
        print(f"seed: no line 'void kernel_main() {{' in {args.src}")
        return 2
    lines.insert(index + 1, SEEDS[args.seed])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")
    return 0


def join_resumed(lines: Iterable[str]) -> Iterator[str]:
    """Yield complete strace lines, joining each "<unfinished ...>" call with its "<... resumed>" rest."""
    pending: dict[str, str] = {}
    unfinished = re.compile(r"^(\d+)\s+(.*?)\s*<unfinished \.\.\.>\s*$")
    resumed = re.compile(r"^(\d+)\s+<\.\.\. (\w+) resumed>(.*)$")
    for line in lines:
        line = line.rstrip("\n")
        match = unfinished.match(line)
        if match:
            pending[match[1]] = match[2]
            continue
        match = resumed.match(line)
        if match:
            head = pending.pop(match[1], match[2] + "(")
            yield f"{match[1]} {head}{match[3]}"
            continue
        yield line


def classify_call(name: str, args: str) -> tuple[str, list[str]] | None:
    """Return (kind, paths) for a traced call: exec, read, write, net, or look (a stat-like call); or None."""
    paths = re.findall(r'"((?:[^"\\]|\\.)*)"', args)
    if name == "execve":
        return "exec", paths[:1]
    if name in OPEN_CALLS:
        return ("write" if any(flag in args for flag in WRITE_FLAGS) else "read"), paths[:1]
    if name in WRITE_CALLS:
        return "write", paths[:2]
    if name in NET_CALLS:
        family = re.search(r"(AF_\w+|sa_family=\w+)", args)
        port = re.search(r"sin6?_port=htons\((\d+)\)", args)
        detail = [family[1] if family else "?"] + paths + ([f"port {port[1]}"] if port else [])
        return "net", detail
    if paths:
        return "look", paths[:1]
    return None


def path_class(path: str, roots: list[tuple[str, str]]) -> str:
    """Return the class of `path`: a named root, private (/tmp, /var/tmp, /dev/shm), relative, or /<top>."""
    if not path.startswith("/"):
        return "relative"
    for name, root in roots:
        if path == root or path.startswith(root + "/"):
            return name
    if any(path == d or path.startswith(d + "/") for d in PRIVATE_DIRS):
        return "private"
    return "/" + path.split("/")[1]


def scan_strace(log: Path, roots: list[tuple[str, str]]) -> dict:
    """Read an strace log (gzip-compressed when its name ends in .gz) and return the tallies scan_report prints."""
    call = re.compile(r"^(\d+)\s+(\w+)\((.*)\)\s+=\s+(\S+)(.*)$")
    tally = {"counts": Counter(), "writes": {}, "outside": {}, "dev": {}, "etc": set(), "pin": Counter(),
             "execs": Counter(), "net": Counter(), "lines": 0}
    opener = gzip.open if log.suffix == ".gz" else open
    try:
        with opener(log, "rt", errors="replace") as handle:
            for line in join_resumed(handle):
                tally["lines"] += 1
                match = call.match(line)
                found = classify_call(match[2], match[3]) if match else None
                if found is None:
                    continue
                kind, paths = found
                ok = not match[4].startswith("-1")
                if kind == "net":
                    tally["net"][(match[2], " ".join(paths), ok)] += 1
                    continue
                for path in paths:
                    note_path(tally, kind, path, ok, match[2], roots)
    except (EOFError, OSError, zlib.error) as exc:
        tally["lines"] = f"{tally['lines']} (the log ends early: {exc}; a probe stopped at its limit)"
    return tally


def note_path(tally: dict, kind: str, path: str, ok: bool, call: str, roots: list[tuple[str, str]]) -> None:
    """Add one traced path to the tallies."""
    where = path_class(path, roots)
    tally["counts"][(where, kind, "ok" if ok else "failed")] += 1
    if kind == "exec" and ok:
        tally["execs"][path] += 1
    if kind == "write" and where not in ("run", "private"):
        tally["writes"][path] = f"{call} {'ok' if ok else 'failed'}"
    if kind != "write" and where not in EXPECTED_READS:
        tally["outside"][path] = f"{kind} {'ok' if ok else 'failed'}"
    if where == "/dev":
        tally["dev"][path] = f"{kind} {'ok' if ok else 'failed'}"
    if where == "/etc" and ok:
        tally["etc"].add(path)
    if where == "tt-metal" and ok:
        root = dict(roots)["tt-metal"]
        tally["pin"]["/".join(path[len(root) + 1 :].split("/")[:3])] += 1


def scan_report(tally: dict) -> list[str]:
    """Return the text lines of the strace summary."""
    out = [f"strace lines: {tally['lines']}", "== calls by path class, kind, result"]
    out += [f"{n:>8}  {w} {k} {r}" for (w, k, r), n in sorted(tally["counts"].items())]
    out.append("== writes outside the run directory and the private /tmp, /var/tmp, /dev/shm (should be /dev only)")
    out += [f"  {p}  ({v})" for p, v in sorted(tally["writes"].items())[:60]]
    out.append(f"== reads and lookups outside the pins, the run, and system directories ({len(tally['outside'])})")
    out += [f"  {p}  ({v})" for p, v in sorted(tally["outside"].items())[:60]]
    out.append("== /dev entries")
    out += [f"  {p}  ({v})" for p, v in sorted(tally["dev"].items())[:40]]
    out.append("== /etc files read")
    out += [f"  {p}" for p in sorted(tally["etc"])[:40]]
    out.append("== tt-metal pin, successful reads and lookups by directory (top 25)")
    out += [f"{n:>8}  {p}" for p, n in tally["pin"].most_common(25)]
    out.append("== executables started")
    out += [f"{n:>8}  {p}" for p, n in tally["execs"].most_common(40)]
    out.append("== network calls (call, family and address, result)")
    out += [f"{n:>8}  {c} {d} {'ok' if ok else 'failed'}" for (c, d, ok), n in sorted(tally["net"].items())]
    return out


def cmd_strace(args: argparse.Namespace) -> int:
    """Summarize args.log into args.out."""
    roots = [tuple(item.split("=", 1)) for item in args.root]
    if not Path(args.log).is_file():
        write_ascii(Path(args.out), "no strace log\n")
        return 0
    lines = scan_report(scan_strace(Path(args.log), roots))
    write_ascii(Path(args.out), "\n".join(lines) + "\n")
    return 0


def cmd_getenv(args: argparse.Namespace) -> int:
    """Summarize a getenv_log.c log into args.out: per executable, each name asked for and whether it was set."""
    rows: dict[str, Counter] = defaultdict(Counter)
    try:
        with open(args.log, errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 4:
                    rows[os.path.basename(parts[0]) or "?"][(parts[2], parts[3])] += 1
    except OSError:
        write_ascii(Path(args.out), "no getenv log\n")
        return 0
    out = []
    for exe in sorted(rows, key=lambda name: (not name.startswith("metal_example_"), name)):
        names = sorted(rows[exe].items())
        out.append(f"== {exe}: {len(names)} distinct lookups (name state count)")
        out += [f"  {name} {state} {count}" for (name, state), count in names]
    write_ascii(Path(args.out), "\n".join(out) + "\n")
    return 0


def capped(data: bytes, cap: int) -> bytes:
    """Return `data`, or its first and last cap/2 bytes around a marker line when it is longer than `cap`."""
    if len(data) <= cap:
        return data
    half = cap // 2
    cut = len(data) - 2 * half
    return data[:half] + f"\n[... {cut} bytes cut by the P4.9 batch ...]\n".encode() + data[-half:]


def read_text(path: Path) -> str:
    """Return the file's text (undecodable bytes replaced), or '' when it is missing."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def ascii_bytes(data: bytes) -> bytes:
    """Return `data` as ASCII: read as UTF-8, with every other character or byte as a backslash escape."""
    return data.decode("utf-8", errors="backslashreplace").encode("ascii", errors="backslashreplace")


def write_ascii(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII, any other character as a backslash escape (report/ and fixtures/)."""
    Path(path).write_bytes(text.encode("ascii", errors="backslashreplace"))


def asciify_tree(root: Path) -> int:
    """Rewrite every regular file under `root` that holds a non-ASCII byte as ASCII (ascii_bytes); return how many."""
    changed = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath, name)
            data = path.read_bytes()
            if any(byte > 127 for byte in data):
                path.write_bytes(ascii_bytes(data))
                changed += 1
    return changed


def tree_usage(root: Path) -> tuple[int, int]:
    """Return (KiB allocated, file count) under `root`, counting files and directories as du -sk does.

    Where the platform gives no block count (not on the build host), the size rounded up to 4 KiB stands in.
    """
    blocks = files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            blocks += getattr(info, "st_blocks", -(-info.st_size // 4096) * 8)
            files += name in filenames
    return blocks // 2, files


def listing(root: Path, limit: int = 30) -> list[str]:
    """Return 'size relative-path' for the regular files under `root`, at most `limit` of them."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = Path(dirpath, name)
            try:
                found.append(f"{path.lstat().st_size} {path.relative_to(root).as_posix()}")
            except OSError:
                continue
    return sorted(found)[:limit] + ([f"... {len(found) - limit} more"] if len(found) > limit else [])


def find_watcher_log(logs_dir: str) -> Path | None:
    """Return the first file named watcher*.log anywhere under `logs_dir`, or None. Watcher's exact path is a
    reading of watcher_server.cpp; searching avoids missing it if the pin writes elsewhere under the logs dir."""
    if logs_dir in ("", "-"):
        return None
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        for name in sorted(filenames):
            if name.startswith("watcher") and name.endswith(".log"):
                return Path(dirpath, name)
    return None


def output_facts(text: str) -> dict:
    """Return what a step's combined stdout and stderr show: check, PCC, every ttsim line, rate line, JIT error.

    ttsim_lines keeps every cycle-prefixed "[<n>] ..." line (capped), which is run-time output, so the acceptance
    "ttsim messages" for each example sits in report, not only in local. seeded_copy_compiled is the weak
    log-based guess; step_record adds the direct cache-based check.
    """
    errors = SIM_ERROR.findall(text)
    rate = SIM_RATE.search(text)
    jit = next((line.strip() for line in text.split("\n") if JIT_ERROR.search(line)), "")
    sim_lines = [line.strip() for line in text.split("\n") if SIM_LINE.match(line.strip())]
    pcc = PCC.search(text)
    if "Test Passed" in text:
        check = "passed"
    elif "Test Failed" in text or "Test failed with exception" in text:
        check = "failed"
    else:
        check = "none printed"
    return {"check": check, "pcc": pcc[1] if pcc else None, "sim_errors": len(errors),
            "sim_first": f"{errors[0][0]}: {errors[0][1].strip()[:200]}" if errors else "",
            "sim_classes": sorted({kind for kind, _ in errors}), "sim_rate_line": rate[0] if rate else "",
            "sim_lines": [line[:200] for line in sim_lines[:40]], "sim_line_count": len(sim_lines),
            "jit_first": jit[:240], "seeded_copy_compiled": "/cwd/tt_metal/programming_examples/" in text}


def cmd_quiet(args: argparse.Namespace) -> int:
    """Exit 0 when the step in args.dir failed without a ttsim error or JIT error line, so a retry may help.

    A failure counts whether it exited fast or was stopped at its limit (a step that printed a finding through
    buffered stdio and then hung would lose the line to SIGKILL; a retry may still catch it). run.sh uses this
    to pick the loopback fallback, the HOME retry, and the sandbox HOME retry.
    """
    directory = Path(args.dir)
    try:
        result = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError):
        return 1
    facts = output_facts(read_text(directory / "stdout.txt") + "\n" + read_text(directory / "stderr.txt"))
    failed = result.get("rc") not in (0, None) or result.get("timed_out")
    return 0 if failed and not facts["sim_errors"] and not facts["jit_first"] else 1


def unpack_lines(cache: Path) -> list[str]:
    """Return the distinct unpack_to_dest and dest-accumulation lines of the JIT's generated headers, counted."""
    pattern = re.compile(r"unpack_to_dest|UNPACK_TO_DEST|DST_ACCUM_MODE|fp32_dest_acc|unpack_src_format|"
                         r"unpack_dst_format")
    seen: Counter = Counter()
    for dirpath, _dirnames, filenames in os.walk(cache):
        for name in filenames:
            if name.endswith((".h", ".hpp")):
                for line in read_text(Path(dirpath, name)).split("\n"):
                    if pattern.search(line):
                        seen[" ".join(line.split())[:200]] += 1
    return [f"{count:>4}  {line}" for line, count in sorted(seen.items())]


def unpack_facts(lines: list[str]) -> dict:
    """Return a value-level answer derived from unpack_lines' output, quoting none of it (the lines go to local/).

    A line is "enabled" when its value is true, holds a nonzero digit, or names an Fp32 mode (UNPACK_ENABLED),
    which also catches a 0/1 array or an enum, the case the earlier "true"-only count missed. any_enabled is the
    tracked answer to "does this kernel use unpack_to_dest"; the descriptor lines behind it stay in local/.
    """
    naming = [line for line in lines if re.search(r"unpack_to_dest|UNPACK_TO_DEST", line)]
    accumulation = [line for line in lines if re.search(r"DST_ACCUM_MODE|fp32_dest_acc", line)]
    naming_on = [line for line in naming if UNPACK_ENABLED.search(line)]
    accumulation_on = [line for line in accumulation if UNPACK_ENABLED.search(line)]
    return {"distinct_descriptor_lines": len(lines), "unpack_to_dest_lines": len(naming),
            "unpack_to_dest_lines_enabled": len(naming_on),
            "dest_accumulation_lines": len(accumulation),
            "dest_accumulation_lines_enabled": len(accumulation_on),
            "any_unpack_to_dest_enabled": bool(naming_on or accumulation_on)}


def seeded_copy_in_cache(cache: Path) -> bool:
    """Return True when the JIT cache holds the seeded kernel: a file naming SEED_MARK or the copy's cwd path.

    This is direct proof the seeded copy was compiled, unlike the log-string guess seeded_copy_compiled, so a
    seed that ttsim tolerates silently can still be told apart from a copy that was never used.
    """
    for dirpath, _dirnames, filenames in os.walk(cache):
        for name in filenames:
            if name.endswith((".d", ".h", ".hpp", ".cpp", ".txt", ".json")):
                text = read_text(Path(dirpath, name))
                if SEED_MARK in text or "/cwd/tt_metal/programming_examples/" in text:
                    return True
    return False


def step_record(raw: Path, row: list[str]) -> dict:
    """Return one step's record: result.json plus what its output, cache, logs, HOME, and private /tmp show.

    `row` is a line of raw/steps.tsv: name, kind, example, cache dir, logs dir, seed ("-" when none), loopback.
    """
    name, kind, example, cache, logs, seed, loopback = (row + ["-"] * 7)[:7]
    directory = raw / name
    try:
        record = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError):
        record = {"name": name, "rc": None, "skipped_or_unfinished": True}
    text = read_text(directory / "stdout.txt") + "\n" + read_text(directory / "stderr.txt")
    record.update(kind=kind, example=example, seed=seed, cache_dir=cache, logs_dir=logs, loopback=loopback,
                  **output_facts(text))
    record.setdefault("device", DEVICE)
    record.setdefault("timing_note", TIMING_NOTE)
    if cache != "-" and kind in ("clean", "clean-lo-up", "sfplm", "sandbox"):
        record["cache_kib"], record["cache_files"] = tree_usage(Path(cache))
        record["unpack_lines"] = unpack_lines(Path(cache))
        record["unpack_to_dest_facts"] = unpack_facts(record["unpack_lines"])
    if kind in ("seed", "watcher") and seed not in ("-", "hang") and cache != "-":
        record["seeded_copy_in_cache"] = seeded_copy_in_cache(Path(cache))
    watcher = find_watcher_log(logs)
    record["watcher_log"] = str(watcher) if watcher else ""
    record["logs"] = listing(Path(logs)) if logs != "-" else []
    record["home"] = listing(directory / "home")
    record["cwd_files"] = listing(directory / "cwd")
    private = [line for line in read_text(directory / "private-tmp.txt").split("\n") if line.strip()]
    record["private_tmp"] = private[:30] + ([f"... {len(private) - 30} more"] if len(private) > 30 else [])
    return record


def copy_logs(raw: Path, local: Path, record: dict) -> None:
    """Copy a step's stdout, stderr, watcher log, and helper.txt, each capped at LOCAL_CAP, into local/.

    local/ is never committed, so its content (which may quote upstream text) stays off the tracked path. The
    sandbox helper's traceback (helper.txt) is copied so a crash need not be re-read with another rx exec.
    """
    name = record["name"]
    sources = {f"{name}.{s}.txt": raw / name / f"{s}.txt" for s in ("stdout", "stderr")}
    sources[f"{name}.helper.txt"] = raw / name / "helper.txt"
    sources[f"{name}.unavailable.txt"] = raw / name / "unavailable.txt"
    if record.get("watcher_log"):
        sources[f"{name}.watcher.log"] = Path(record["watcher_log"])
    for target, source in sources.items():
        if source.is_file():
            (local / target).write_bytes(capped(source.read_bytes(), LOCAL_CAP))


def sanitize_capture(data: bytes) -> tuple[bytes, int]:
    """Return (sanitized bytes, replacements): every upstream-derived line becomes one fixed marker line.

    A line is upstream-derived when it is a compiler source-context line other than the seeded one (SOURCE_LINE
    without SEED_MARK), a compile-command or backtrace or TT_FATAL/TT_ASSERT/TT_THROW line (UPSTREAM_LINE). The
    ttsim messages, the example's own prints, and the seeded line are kept. Replacement is by code, so the
    tracked fixture is deterministic and never hand-edited.
    """
    marker = "[P4.9: upstream-derived line removed]"
    kept, replaced = [], 0
    for line in data.decode("utf-8", errors="backslashreplace").split("\n"):
        source = SOURCE_LINE.match(line)
        if (source and SEED_MARK not in line) or UPSTREAM_LINE.search(line):
            kept.append(marker)
            replaced += 1
        else:
            kept.append(line)
    return "\n".join(kept).encode("ascii", errors="backslashreplace"), replaced


def write_fixture(fixtures: Path, raw: Path, record: dict, provenance: dict) -> None:
    """Write a seeded or Watcher capture, mechanically sanitized of upstream source and ASCII: stdout, stderr,
    status.json, and watcher.log if any. `provenance` holds the commit, the rx run id, dirty, and snapshot_of.
    """
    target = fixtures / record["name"]
    target.mkdir(parents=True, exist_ok=True)
    replaced = 0
    for stream in ("stdout", "stderr"):
        source = raw / record["name"] / f"{stream}.txt"
        data = source.read_bytes() if source.is_file() else b""
        clean, count = sanitize_capture(capped(data, FIXTURE_CAP))
        replaced += count
        (target / f"{stream}.txt").write_bytes(clean)
    keys = ("rc", "timed_out", "stopped_reason", "wall_s", "limit_s", "check", "sim_classes", "sim_first",
            "sim_lines", "jit_first", "seeded_copy_compiled", "seeded_copy_in_cache", "example", "seed",
            "started_at", "ended_at")
    status = {key: record.get(key) for key in keys}
    status.update(provenance, upstream_lines_removed=replaced, seed_line=SEEDS.get(record["seed"]),
                  device=DEVICE, timing_note=TIMING_NOTE + "; times inside stdout.txt, stderr.txt, and "
                  "watcher.log are simulator wall time too")
    write_ascii(target / "status.json", json.dumps(status, indent=1, sort_keys=True) + "\n")
    if record.get("watcher_log") and Path(record["watcher_log"]).is_file():
        clean, _ = sanitize_capture(capped(Path(record["watcher_log"]).read_bytes(), FIXTURE_CAP))
        (target / "watcher.log").write_bytes(clean)


def fixture_check(fixtures: Path, local: Path) -> list[str]:
    """Return report/fixture-check.txt lines: per capture, counts and line numbers only. Any quoted upstream
    source goes to local/fixture-context.txt, which is never committed. The captures are already sanitized by
    write_fixture; this re-checks the tracked files hold no source-context line and reports where any is."""
    out = ["== per capture (already sanitized): source-context lines still present (should be 0), compile-command"
           " lines, and Watcher lines before the first dump; any quoted line is in local/fixture-context.txt"]
    context: list[str] = []
    for directory in sorted(path for path in fixtures.iterdir() if path.is_dir()):
        lines = (read_text(directory / "stdout.txt") + "\n" + read_text(directory / "stderr.txt")).split("\n")
        hits = [(i, m[0]) for i, m in enumerate(SOURCE_LINE.match(x) for x in lines) if m and SEED_MARK not in m[0]]
        commands = sum("riscv-tt-elf-g++" in line for line in lines)
        watcher_lines = read_text(directory / "watcher.log").split("\n")
        legend = next((i for i, line in enumerate(watcher_lines) if "Dump" in line), len(watcher_lines))
        out.append(f"{directory.name}: {len(hits)} source-context lines at {[i for i, _ in hits][:10]}; "
                   f"{commands} compile-command lines; "
                   f"{legend if (directory / 'watcher.log').is_file() else 0} watcher lines before the first dump")
        context += [f"{directory.name}:{i}: {line.strip()[:160]}" for i, line in hits]
    (local.parent / "fixture-context.txt").write_text("\n".join(context) + "\n")
    return out


def table_lines(records: list[dict]) -> list[str]:
    """Return the per-step table: one line per step."""
    header = (f"{'step':<30} {'rc':>4} {'check':<12} {'sim_wall_s':>10} {'rss_MiB':>7} {'tasks':>5} "
              f"{'cache_KiB/files':>15}")
    out = [f"== steps; device: {DEVICE}", f"   {TIMING_NOTE}", header + " sim"]
    for r in records:
        rss = r.get("maxrss_kib")
        cache = f"{r['cache_kib']}/{r['cache_files']}" if "cache_kib" in r else "-"
        rc = "T/O" if r.get("timed_out") else r.get("rc")
        sim = ",".join(r.get("sim_classes", [])) or ("JIT error" if r.get("jit_first") else "-")
        out.append(f"{r['name']:<30} {str(rc):>4} {r.get('check', '?'):<12} {str(r.get('wall_s')):>10} "
                   f"{(rss or 0) / 1024:>7.0f} {str(r.get('peak_tasks', '-')):>5} {cache:>15} {sim}")
    return out


def finding_lines(records: list[dict]) -> list[str]:
    """Return the findings lines (the batch's stdout): errors, rate lines, unpack_to_dest, writes, sandbox, Watcher."""
    ran = [r for r in records if not r.get("skipped_or_unfinished")]
    passed = [r["name"] for r in ran if r["kind"].startswith("clean") and r.get("rc") == 0]
    out = [f"== {len(ran)} of {len(records)} steps ran; clean runs exiting 0: {' '.join(passed) or 'none'}",
           f"   device: {DEVICE}; step table and wall times: report/summary.txt ({TIMING_NOTE})",
           "== ttsim errors (class: first message) and JIT errors"]
    for r in records:
        if r.get("sim_first") or r.get("jit_first"):
            seeded = " (seeded copy compiled)" if r.get("seeded_copy_compiled") else ""
            out.append(f"{r['name']}: {r.get('sim_first') or r.get('jit_first')}"[:200] + seeded)
    rates = {r["name"]: r["sim_rate_line"] for r in records if r.get("sim_rate_line")}
    out.append(f"== ttsim's own rate lines (simulated clock rate, not performance): {len(rates)} steps; "
               f"first: {next(iter(rates.items()), '-')}")
    facts = {r["name"]: r["unpack_to_dest_facts"] for r in records if "unpack_to_dest_facts" in r}
    enabled = [name for name, f in facts.items() if f["any_unpack_to_dest_enabled"]]
    out.append(f"== unpack_to_dest: {len(facts)} runs; enabled in: {' '.join(enabled) or 'none'} "
               "(per-run any_unpack_to_dest_enabled in report/unpack_to_dest-facts.txt; lines in local/)")
    out.append("== files outside the JIT cache, steps that wrote any (details: report/files-written.txt)")
    for key, label in (("logs", "logs dir"), ("home", "HOME"), ("cwd_files", "cwd (a seeded step holds 1 copy)"),
                       ("private_tmp", "private /tmp, /var/tmp, /dev/shm")):
        steps = [f"{r['name']}:{len(r[key])}" for r in records if r.get(key)]
        out.append(f"{label}: {' '.join(steps) or 'none'}"[:300])
    out.append("== sandbox (P0.16 sandbox, toolchains root read-only; sim_program_s is simulator wall time)")
    for r in records:
        if r["kind"] == "sandbox":
            out.append(f"{r['name']}: rc={r.get('rc')} hang={r.get('timed_out')} killed={r.get('killed')} "
                       f"sim_program_s={r.get('program_s')} incomplete={r.get('workdir_incomplete')} "
                       f"unavailable_len={r.get('unavailable_len')}")
    out += [f"allowlist: {r.get('allowlist', '')[:200]}" for r in records if r["kind"] == "sandbox"][:1]
    out.append("== Watcher")
    for r in records:
        if r["kind"] == "watcher":
            path = r.get("watcher_log") or ""
            size = Path(path).stat().st_size if path and Path(path).is_file() else None
            lines = read_text(Path(path)).count("\n") if size else 0
            out.append(f"{r['name']}: rc={r.get('rc')} timed_out={r.get('timed_out')} watcher.log "
                       f"{'missing' if size is None else f'{size} bytes, {lines} lines'}")
    return out


def write_reports(report: Path, local: Path, records: list[dict]) -> None:
    """Write runs.tsv, steps/<name>.json, unpack_to_dest-facts.txt, and files-written.txt under report/, as ASCII.

    The JIT-generated descriptor lines behind the unpack_to_dest facts go to local/unpack_to_dest.txt only, and
    no record in report/ holds them. runs.tsv starts with a comment line naming the device and the timing note.
    """
    keys = ("name", "kind", "example", "loopback", "rc", "timed_out", "stopped_reason", "check", "pcc", "wall_s",
            "program_s", "maxrss_kib", "peak_group_rss_kib", "peak_tasks", "peak_process_threads", "cache_kib",
            "cache_files", "sim_classes", "sim_first", "jit_first", "sim_rate_line", "seeded_copy_in_cache",
            "started_at", "ended_at")
    rows = [f"# device: {DEVICE}; {TIMING_NOTE}", "\t".join(keys)]
    rows += ["\t".join(" ".join(str(r.get(k, "")).split()) for k in keys) for r in records]
    write_ascii(report / "runs.tsv", "\n".join(rows) + "\n")
    (report / "steps").mkdir(exist_ok=True)
    unpack, facts, written = [], [], []
    for r in records:
        lines = r.pop("unpack_lines", None)
        write_ascii(report / "steps" / f"{r['name']}.json", json.dumps(r, indent=1, sort_keys=True) + "\n")
        if lines is not None:
            unpack += [f"== {r['name']} ({r['example']})"] + lines
            facts.append(f"{r['name']} ({r['example']}): {json.dumps(r['unpack_to_dest_facts'], sort_keys=True)}")
        written += [f"== {r['name']}", "logs dir:"] + r.get("logs", []) + ["HOME:"] + r.get("home", [])
        written += ["cwd (a seeded step holds its kernel copy here):"] + r.get("cwd_files", [])
        written += ["private /tmp, /var/tmp, /dev/shm (type size path):"] + r.get("private_tmp", [])
    (local.parent / "unpack_to_dest.txt").write_text("\n".join(unpack) + "\n")
    write_ascii(report / "unpack_to_dest-facts.txt", "\n".join(facts) + "\n")
    write_ascii(report / "files-written.txt", "\n".join(written) + "\n")


def cmd_summarize(args: argparse.Namespace) -> int:
    """Build report/, fixtures/, and local/ for the run directory args.run; print the findings.

    The step table goes to report/summary.txt only. Last, every file under report/ and fixtures/, those run.sh
    wrote included, is rewritten as ASCII where it is not already (asciify_tree).
    """
    run = Path(args.run)
    raw, report, fixtures, local = run / "raw", run / "report", run / "fixtures", run / "local" / "logs"
    for directory in (report, fixtures, local):
        directory.mkdir(parents=True, exist_ok=True)
    rows = [line.split("\t") for line in read_text(raw / "steps.tsv").split("\n") if line.strip()]
    records = [step_record(raw, row) for row in rows]
    provenance = {"commit": args.commit, "rx_run_id": args.rx_id, "dirty": args.dirty == "true", "snapshot_of": None}
    for record in records:
        copy_logs(raw, local, record)
        if record["kind"] in ("seed", "watcher") and not record.get("skipped_or_unfinished"):
            write_fixture(fixtures, raw, record, provenance)
    write_reports(report, local, records)
    write_ascii(report / "fixture-check.txt", "\n".join(fixture_check(fixtures, local)) + "\n")
    findings = finding_lines(records)
    write_ascii(report / "summary.txt", "\n".join(table_lines(records) + findings) + "\n")
    rewritten = asciify_tree(report) + asciify_tree(fixtures)
    print("\n".join(findings + [f"== files rewritten as ASCII: {rewritten}"]), flush=True)
    return 0


def cmd_asciify(args: argparse.Namespace) -> int:
    """Rewrite any non-ASCII file under report/ and fixtures/ (run.sh calls this after checks.txt is written)."""
    run = Path(args.run)
    rewritten = asciify_tree(run / "report") + asciify_tree(run / "fixtures")
    print(f"p49: ascii pass rewrote {rewritten} file(s)", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    """Return the command-line parser with one subcommand per helper."""
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--name", required=True)
    measure.add_argument("--out", required=True)
    measure.add_argument("--timeout", type=float, required=True)
    measure.add_argument("--cwd", required=True)
    measure.add_argument("--stop-file", action="append", default=[])
    measure.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    measure.add_argument("argv", nargs=argparse.REMAINDER)
    seed = sub.add_parser("seed")
    seed.add_argument("--seed", required=True, choices=sorted(SEEDS))
    seed.add_argument("--src", required=True)
    seed.add_argument("--dst", required=True)
    seed.add_argument("--protect", action="append", default=[])
    trace = sub.add_parser("strace")
    trace.add_argument("--log", required=True)
    trace.add_argument("--out", required=True)
    trace.add_argument("--root", action="append", default=[])
    getenv = sub.add_parser("getenv")
    getenv.add_argument("--log", required=True)
    getenv.add_argument("--out", required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    summarize.add_argument("--commit", required=True)
    summarize.add_argument("--rx-id", required=True)
    summarize.add_argument("--dirty", default="false")
    quiet = sub.add_parser("quiet")
    quiet.add_argument("--dir", required=True)
    asciify = sub.add_parser("asciify")
    asciify.add_argument("--run", required=True)
    return top


def main(argv: list[str]) -> int:
    """Dispatch to the subcommand."""
    args = parser().parse_args(argv)
    commands = {"measure": cmd_measure, "seed": cmd_seed, "strace": cmd_strace, "getenv": cmd_getenv,
                "summarize": cmd_summarize, "quiet": cmd_quiet, "asciify": cmd_asciify}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
