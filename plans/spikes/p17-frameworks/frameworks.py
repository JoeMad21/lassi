"""P17.1 spike helper: the sandboxed steps and the summaries of plans/spikes/p17-frameworks/run.sh.

Run by run.sh as `uv run --frozen python <this file> <command> ...` from the repository root on the build host
(report: plans/spikes/p17-frameworks.md). Every program this helper starts runs inside the P0.16 sandbox
(lassi.executors.sandbox). Its private /dev holds only null, zero, full, random, urandom, tty, a devpts, and a
shm tmpfs, and /sys/class, /sys/bus, and every /sys/devices entry but system are empty, so no step can open a
GPU device node or read a GPU's attribute files, even by accident inside a library (OQ-002). $LASSI_SCRATCH,
$HOME, and $LASSI_RUNS_ROOT are hidden roots; the step's workdir is the one writable host directory; the one
exposed tree (the scratch venv, or the ROCm root) is bind-mounted read-only, like the rest of the host; and the
sandbox has no network.

Commands:
  imports  imports the framework set from the scratch venv and prints each version, torch.version.cuda and
           torch.version.hip, the CPU capability, and one small CPU matmul; nothing queries a device;
  hip      runs hipcc --version, the compile-only gfx942 probe (env -i gives hipcc exactly HIP_ENV, so HIP_PATH,
           ROCM_PATH, HIP_PLATFORM, and every loader variable are unset), the same compile under strace when
           the host has strace, and the object's inspection with the ROCm LLVM tools;
  freeze   compares `uv pip freeze` with the expected pins and counts the resolution's index annotations;
  ascii    replaces each non-ASCII byte of every file under a directory with "?".
Each command writes its records under --report (and full outputs under --raw) and prints at most --lines lines.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import (
    COMPILE_CPUS,
    COMPILE_DISK_MB,
    COMPILE_MEMORY_MB,
    SANDBOX_PATH,
    Sandbox,
    SandboxSpec,
    SandboxUnavailableError,
)

# The import step's limits: wide margin over a torch import (a few hundred MiB); the workdir cap holds the
# library caches the import may create under HOME, which the sandbox sets to the workdir.
IMPORT_LIMITS = Limits(wall_s=600, memory_mb=8192, cpus=4)
IMPORT_DISK_MB = 1024
# Every ROCm step uses the limits the stage runner gives a compile (SandboxedCompileRunner): COMPILE_MEMORY_MB,
# COMPILE_DISK_MB, and COMPILE_CPUS, with these wall limits in seconds.
HIP_WALL_S = 300
TOOL_WALL_S = 60
# The program environment of every ROCm step, exactly (the sandbox runs `env -i -- NAME=value... argv`). TMPDIR
# is the sandbox's private /tmp, so nothing is written outside the workdir and nothing needs removing.
HIP_ENV = {"PATH": SANDBOX_PATH, "LANG": "C", "LC_ALL": "C", "TMPDIR": "/tmp"}
TARGET = "gfx942"
COMPILE_ARGS = [f"--offload-arch={TARGET}", "-O2", "-c", "probe.hip", "-o", "probe.o"]
# strace's first child is a shell that always exits normally, so strace never re-raises a fatal signal of the
# compiler with its own core limit lowered (plans/LESSONS.md, alpha01); the sandbox's seccomp filter also
# refuses any change of RLIMIT_CORE.
STRACE_ARGS = ["-f", "-qq", "-s", "256", "-e", "trace=%file", "-e", "signal=none", "-o", "trace.txt", "--",
               "/bin/sh", "-c", '"$@"; exit "$?"', "sh"]
# Patterns the trace summary looks for in the trace log (a log file, never a command line): device-query
# programs, and device paths that would mean an attempt to reach a GPU.
PROBE_PROGRAMS = {"amdgpu-arch", "offload-arch", "rocm_agent_enumerator", "rocminfo", "rocm-smi", "amd-smi",
                  "nvidia-smi", "nvptx-arch", "hipInfo"}
GPU_PATH = re.compile(
    r"kfd|/dri(/|$)|renderD|amdgpu|/sys/class/drm|/sys/bus/pci|/sys/devices/pci|/proc/bus/pci|/proc/driver"
)
# The paths the trace summary records: /dev and /sys, plus the /proc entries that stay readable in the sandbox.
TRACED_PREFIXES = ("/dev/", "/sys/", "/proc/bus/pci", "/proc/driver")
# The import program, run as `<venv>/bin/python -I -c IMPORT_PROGRAM <module>...`. It sets HOME, the library
# caches, the offline switches, and the thread counts before any import. It calls no torch.cuda function and
# no device query: torch.version.* and torch.backends.cpu.get_cpu_capability() are build and CPU facts.
IMPORT_PROGRAM = r"""
import json, os, sys, time
home = os.getcwd()
os.environ.update(HOME=home, HF_HOME=os.path.join(home, "hf"), XDG_CACHE_HOME=os.path.join(home, "cache"),
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
out = {"python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix, "modules": {}}
for name in sys.argv[1:]:
    start = time.monotonic()
    try:
        module = __import__(name)
        out["modules"][name] = {"version": str(getattr(module, "__version__", None)),
                                "import_s": round(time.monotonic() - start, 2)}
    except Exception as exc:
        out["modules"][name] = {"error": ("%s: %s" % (type(exc).__name__, exc))[:300]}
if "torch" in sys.modules:
    import torch
    facts = {"cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
             "git": torch.version.git_version, "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    torch.set_num_threads(4)
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    facts["matmul_sum"] = float((a @ a.T).sum())
    out["torch"] = facts
print(json.dumps(out, sort_keys=True))
"""
# arange(6) as 2x3 times its transpose is [[5, 14], [14, 50]], which sums to 83.
MATMUL_SUM = 83.0


class Printer:
    """Prints at most `limit` lines, each cut to 160 characters, and keeps every line for the report."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            if len(self.lines) <= self.limit:
                print(line[:160], flush=True)


def wrap(prefix: str, items: list[str], width: int = 150) -> list[str]:
    """Return `items` joined with ", " in lines of about `width` characters, the first starting with `prefix`."""
    lines, current = [], prefix
    for item in items:
        piece = item if current in (prefix, "  ") else ", " + item
        if len(current) + len(piece) > width and current not in (prefix, "  "):
            lines.append(current + ",")
            current, piece = "  ", item
        current += piece
    lines.append(current)
    return lines


def hidden_roots() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH, $HOME, and $LASSI_RUNS_ROOT, resolved, the ones that are set, without repeats."""
    roots: list[Path] = []
    for name in ("LASSI_SCRATCH", "HOME", "LASSI_RUNS_ROOT"):
        value = os.environ.get(name)
        if value and Path(os.path.realpath(value)) not in roots:
            roots.append(Path(os.path.realpath(value)))
    return tuple(roots)


def under_hidden_root(path: Path) -> bool:
    """Return True when `path`, resolved, is or lies under a hidden root (so the sandbox would not show it)."""
    real = Path(os.path.realpath(path))
    return any(real == root or root in real.parents for root in hidden_roots())


def first_lines(text: str, count: int) -> list[str]:
    """Return the first `count` non-empty lines of `text`, with each run of white space made one space."""
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()][:count]


def sandboxed(name: str, workdir: Path, exposed: Path, argv: list[str], limits: Limits, disk_mb: int,
              environment: dict[str, str] | None, raw: Path) -> dict:
    """Run argv in the sandbox with `exposed` read-only; write raw/<name>.{stdout,stderr}.txt; return the record."""
    workdir = Path(os.path.realpath(workdir))
    record: dict = {"name": name, "argv": argv, "workdir": str(workdir), "exposed": os.path.realpath(exposed),
                    "limits": {"wall_s": limits.wall_s, "memory_mb": limits.memory_mb, "cpus": limits.cpus,
                               "disk_mb": disk_mb},
                    "environment": environment if environment is not None else "sandbox defaults"}
    try:
        spec = SandboxSpec(workdir=workdir, hidden_roots=hidden_roots(), toolchains=Path(os.path.realpath(exposed)),
                           disk_mb=disk_mb, environment=environment)
        result = Sandbox().run(spec, argv, limits)
    except (SandboxUnavailableError, ValueError) as exc:
        text = str(exc)
        (raw / f"{name}.unavailable.txt").write_text(text + "\n")
        record.update(rc=None, stdout="", stderr="", unavailable=(text.strip().splitlines() or [""])[-1][:300])
        return record
    (raw / f"{name}.stdout.txt").write_text(result.stdout)
    (raw / f"{name}.stderr.txt").write_text(result.stderr)
    record.update(rc=result.returncode, hang=result.hang, killed=result.killed, wall_s=round(result.wall_s, 2),
                  program_s=result.program_s, stdout_truncated=result.stdout_truncated,
                  stderr_truncated=result.stderr_truncated, workdir_incomplete=result.workdir_incomplete,
                  unavailable=None, stdout=result.stdout, stderr=result.stderr)
    return record


def status(record: dict) -> str:
    """Return a short status text for a sandbox record."""
    if record.get("rc") is None:
        return f"sandbox unavailable ({record.get('unavailable')})"
    extra = " HANG" if record.get("hang") else (" KILLED" if record.get("killed") else "")
    return f"rc={record['rc']}{extra} in {record.get('wall_s')} s"


def slim(record: dict) -> dict:
    """Return the record without its stdout and stderr texts (they are in raw/), plus their first lines."""
    out = {key: value for key, value in record.items() if key not in ("stdout", "stderr")}
    out["stdout_head"] = first_lines(record.get("stdout") or "", 8)
    out["stderr_head"] = first_lines(record.get("stderr") or "", 12)
    return out


# ---------------------------------------------------------------- imports


def cmd_imports(args: argparse.Namespace) -> int:
    """Import the module list from the venv in the sandbox; write report/imports.json; print a summary."""
    say = Printer(args.lines)
    raw, report = Path(args.raw), Path(args.report)
    work = raw / "imports-work"
    work.mkdir(parents=True, exist_ok=True)
    python = str(Path(args.venv) / "bin" / "python")
    record = sandboxed("imports", work, Path(args.venv), [python, "-I", "-c", IMPORT_PROGRAM, *args.modules],
                       IMPORT_LIMITS, IMPORT_DISK_MB, None, raw)
    data: dict = {}
    for line in reversed((record.get("stdout") or "").splitlines()):
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                data = {}
            break
    modules = data.get("modules", {})
    torch_facts = data.get("torch")
    errors = [f"{name}: {info['error']}" for name, info in modules.items() if "error" in info]
    ok = record.get("rc") == 0 and len(modules) == len(args.modules) and not errors
    if torch_facts is not None and torch_facts.get("matmul_sum") != MATMUL_SUM:
        ok = False
    summary = {"sandbox": slim(record), "result": data, "all_imported": ok, "expected_matmul_sum": MATMUL_SUM}
    (report / "imports.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    say(f"imports (sandbox, venv read-only, no network): {status(record)}; python {data.get('python')}; "
        f"all imported: {'yes' if ok else 'NO'}")
    versions = [f"{name} {info.get('version', 'ERROR')}" for name, info in modules.items()]
    if versions:
        for line in wrap("  versions: ", versions)[:3]:
            say(line)
    if torch_facts is not None:
        say(f"  torch.version.cuda={torch_facts.get('cuda')} torch.version.hip={torch_facts.get('hip')} "
            f"cpu_capability={torch_facts.get('cpu_capability')} matmul_sum={torch_facts.get('matmul_sum')} "
            f"(expected {MATMUL_SUM})")
    for line in errors[:2]:
        say(f"  import error: {line}")
    if record.get("rc") not in (0, None) and not data:
        for line in first_lines(record.get("stderr") or "", 3):
            say(f"  stderr: {line}")
    (report / "imports-stdout.txt").write_text("\n".join(say.lines) + "\n")
    return 0


# ---------------------------------------------------------------- hip


def parse_trace(path: Path) -> dict:
    """Summarize an strace -f log: programs started, device-query programs, and the paths TRACED_PREFIXES names."""
    started: dict[str, int] = {}
    failed: dict[str, int] = {}
    pending: dict[str, str] = {}
    touched: dict[str, str] = {}
    head = re.compile(r'^(\d+)\s+(\w+)\((?:[A-Z_0-9]+, |-?\d+, )?"([^"]*)"')
    resumed = re.compile(r"^(\d+)\s+<\.\.\. (execve|execveat) resumed>.*\) = (-?\d+)")
    result = re.compile(r"\) = (-?\d+|\?)(?: (E[A-Z0-9]+))?")
    lines = 0
    with open(path, errors="replace") as handle:
        for line in handle:
            lines += 1
            match = resumed.match(line)
            if match:
                program = pending.pop(match.group(1), None)
                if program is not None:
                    bucket = started if match.group(3) == "0" else failed
                    bucket[program] = bucket.get(program, 0) + 1
                continue
            match = head.match(line)
            if not match:
                continue
            pid, call, target = match.groups()
            ends = list(result.finditer(line))
            end = ends[-1] if ends else None
            if call in ("execve", "execveat"):
                if "<unfinished ...>" in line or end is None:
                    pending[pid] = target
                else:
                    bucket = started if end.group(1) == "0" else failed
                    bucket[target] = bucket.get(target, 0) + 1
            if target.startswith(TRACED_PREFIXES) or target in ("/dev", "/sys"):
                code = "?" if end is None else (end.group(2) or "ok")
                touched.setdefault(target, f"{call} {code}")
    probes = sorted({os.path.basename(p) for p in (*started, *failed) if os.path.basename(p) in PROBE_PROGRAMS})
    gpu = sorted(path for path in touched if GPU_PATH.search(path))
    return {"lines": lines, "started": started, "failed_exec": failed, "probe_programs": probes,
            "dev_sys_paths": dict(sorted(touched.items())[:60]), "dev_sys_count": len(touched),
            "gpu_paths": {path: touched[path] for path in gpu}}


def inspect_object(rocm: Path, compile_work: Path, raw: Path, records: list[dict]) -> dict:
    """Inspect probe.o in the sandbox with the ROCm LLVM tools; return what shows gfx942 device code."""
    llvm = rocm / "lib" / "llvm" / "bin"
    work = raw / "hip-inspect-work"
    work.mkdir(parents=True, exist_ok=True)
    found: dict = {"tools_missing": [], "sections": [], "bundle_targets": [], "code_object": [], "notes": [],
                   "offloading": []}
    if not (compile_work / "probe.o").is_file():
        found["skipped"] = "no probe.o"
        return found
    shutil.copy2(compile_work / "probe.o", work / "probe.o")

    def tool(step: str, name: str, *argv: str) -> dict | None:
        program = llvm / name
        if not program.is_file():
            found["tools_missing"].append(name)
            return None
        record = sandboxed(step, work, rocm, [str(program), *argv], Limits(TOOL_WALL_S, COMPILE_MEMORY_MB,
                           COMPILE_CPUS), COMPILE_DISK_MB, HIP_ENV, raw)
        records.append(record)
        return record

    record = tool("inspect-sections", "llvm-readelf", "-S", "-W", "probe.o")
    sections = re.findall(r"\]\s+(\S+)", (record or {}).get("stdout") or "")
    found["sections"] = [name for name in sections if re.search(r"hip|offload|CLANG_OFFLOAD", name)]
    if ".hip_fatbin" in sections:
        tool("inspect-dump-fatbin", "llvm-objcopy", "--dump-section=.hip_fatbin=probe.hipfb", "probe.o",
             "probe-copy.o")
    if (work / "probe.hipfb").is_file():
        found["fatbin_bytes"] = (work / "probe.hipfb").stat().st_size
        record = tool("inspect-bundle-list", "clang-offload-bundler", "--list", "--type=o", "--input=probe.hipfb")
        found["bundle_targets"] = first_lines((record or {}).get("stdout") or "", 8)
        device = [target for target in found["bundle_targets"] if TARGET in target]
        if device:
            tool("inspect-unbundle", "clang-offload-bundler", "--unbundle", "--type=o", f"--targets={device[0]}",
                 "--input=probe.hipfb", f"--output=probe-{TARGET}.co")
    code_object = work / f"probe-{TARGET}.co"
    if code_object.is_file():
        found["code_object_bytes"] = code_object.stat().st_size
        record = tool("inspect-co-header", "llvm-readelf", "-h", code_object.name)
        found["code_object"] = [line for line in first_lines((record or {}).get("stdout") or "", 40)
                                if line.startswith(("Class:", "Machine:", "Flags:", "OS/ABI:", "Type:"))]
        record = tool("inspect-co-notes", "llvm-readelf", "--notes", code_object.name)
        found["notes"] = [line for line in first_lines((record or {}).get("stdout") or "", 400)
                          if re.search(r"amdhsa\.target|\.name:|\.symbol:", line)][:8]
    record = tool("inspect-offloading", "llvm-objdump", "--offloading", "probe.o")
    found["offloading"] = first_lines((record or {}).get("stdout") or "", 10)
    flags = " ".join(found["code_object"])
    found["gfx942_in_bundle"] = any(TARGET in target for target in found["bundle_targets"])
    found["gfx942_in_code_object"] = "GFX942" in flags.upper() or any(TARGET in note for note in found["notes"])
    # An offloading image counts only as machine code: its kind line must say elf (bitcode is not ISA).
    found["offloading_kind_elf"] = any(line.split()[:2] == ["kind", "elf"] for line in found["offloading"])
    found["gfx942_in_offloading"] = found["offloading_kind_elf"] and any(TARGET in line for line in found["offloading"])
    return found


def cmd_hip(args: argparse.Namespace) -> int:
    """hipcc --version, the gfx942 compile-only probe, the traced compile, and the inspection, all sandboxed."""
    say = Printer(args.lines)
    raw, report = Path(args.raw), Path(args.report)
    raw.mkdir(parents=True, exist_ok=True)
    rocm = Path(os.path.realpath(args.rocm))
    hipcc = rocm / "bin" / "hipcc"
    limits = Limits(HIP_WALL_S, COMPILE_MEMORY_MB, COMPILE_CPUS)
    records: list[dict] = []
    summary: dict = {"rocm": args.rocm, "rocm_realpath": str(rocm), "hipcc": str(hipcc), "target": TARGET,
                     "environment": HIP_ENV, "compile_argv": [str(hipcc), *COMPILE_ARGS]}
    say(f"hipcc: {hipcc} (ROCm root {args.rocm} -> {rocm}); env -i {' '.join(sorted(HIP_ENV))} only")

    work = raw / "hip-version-work"
    work.mkdir(parents=True, exist_ok=True)
    record = sandboxed("hip-version", work, rocm, [str(hipcc), "--version"], Limits(TOOL_WALL_S, COMPILE_MEMORY_MB,
                       COMPILE_CPUS), COMPILE_DISK_MB, HIP_ENV, raw)
    records.append(record)
    say(f"  hipcc --version: {status(record)}")
    for line in first_lines(record.get("stdout") or "", 3):
        say(f"    {line}")

    work = raw / "hip-compile-work"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, work / "probe.hip")
    record = sandboxed("hip-compile", work, rocm, [str(hipcc), *COMPILE_ARGS], limits, COMPILE_DISK_MB, HIP_ENV, raw)
    records.append(record)
    obj = work / "probe.o"
    summary["object_bytes"] = obj.stat().st_size if obj.is_file() else None
    say(f"  compile ({' '.join(COMPILE_ARGS)}): {status(record)}; probe.o {summary['object_bytes']} bytes")
    for line in first_lines(record.get("stderr") or "", 3):
        say(f"    stderr: {line}")

    found = inspect_object(rocm, work, raw, records)
    summary["inspection"] = found
    say(f"  sections: {', '.join(found['sections']) or 'none named hip/offload'}; fatbin "
        f"{found.get('fatbin_bytes')} bytes; tools missing: {', '.join(found['tools_missing']) or 'none'}")
    for line in wrap("  bundle targets: ", found["bundle_targets"] or ["none"])[:2]:
        say(line)
    say(f"  {TARGET} code object: {found.get('code_object_bytes')} bytes; "
        f"{'; '.join(line for line in found['code_object'] if line.startswith(('Machine:', 'Flags:'))) or 'n/a'}")
    for line in found["notes"][:2]:
        say(f"    note: {line}")
    for line in found["offloading"][:1]:
        say(f"    objdump --offloading: {line}")

    strace = args.strace
    if not strace:
        summary["trace"] = {"skipped": "strace is not on the host's PATH"}
        say("  trace: skipped, strace is not on the host's PATH")
    elif under_hidden_root(Path(strace)):
        summary["trace"] = {"skipped": f"strace ({strace}) lies under a hidden root, so the sandbox does not show it"}
        say(f"  trace: skipped, {strace} lies under a hidden root")
    else:
        work = raw / "hip-trace-work"
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, work / "probe.hip")
        argv = [os.path.realpath(strace), *STRACE_ARGS, str(hipcc), *COMPILE_ARGS[:-1], "probe-trace.o"]
        record = sandboxed("hip-trace", work, rocm, argv, limits, COMPILE_DISK_MB, HIP_ENV, raw)
        records.append(record)
        trace = work / "trace.txt"
        parsed = parse_trace(trace) if trace.is_file() else {}
        summary["trace"] = {"strace": strace, "record": record["name"], **parsed}
        say(f"  trace (strace -f -e trace=%file in the sandbox): {status(record)}; log "
            f"{parsed.get('lines', 'missing')} lines")
        if parsed:
            programs = [f"{p.replace(str(rocm), '<rocm>')} x{n}" for p, n in parsed["started"].items()]
            for line in wrap("    programs started: ", programs or ["none"])[:4]:
                say(line)
            say(f"    failed exec attempts: {sum(parsed['failed_exec'].values())}; device-query programs: "
                f"{', '.join(parsed['probe_programs']) or 'none'}; /dev and /sys paths: {parsed['dev_sys_count']},"
                f" GPU-related: {', '.join(f'{p} ({c})' for p, c in parsed['gpu_paths'].items()) or 'none'}")
        elif record.get("rc") is not None:
            for line in first_lines(record.get("stderr") or "", 2):
                say(f"    stderr: {line}")

    compiled = next(r for r in records if r["name"] == "hip-compile")
    yes = compiled.get("rc") == 0 and (found.get("gfx942_in_code_object") or found.get("gfx942_in_offloading"))
    summary["verdict"] = bool(yes)
    summary["steps"] = [slim(r) for r in records]
    (report / "hip.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    say(f"hip verdict: hipcc built {TARGET} device code with no GPU device in view: {'yes' if yes else 'NO'}"
        f" (bundle {found.get('gfx942_in_bundle')}, code object {found.get('gfx942_in_code_object')},"
        f" offloading {found.get('gfx942_in_offloading')})")
    (report / "hip-stdout.txt").write_text("\n".join(say.lines) + "\n")
    return 0


# ---------------------------------------------------------------- freeze


def normalize(name: str) -> str:
    """Return a package name in its normalized form (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return {name: version} and {name: index} from a freeze or an annotated `uv pip compile` output."""
    pins: dict[str, str] = {}
    index: dict[str, str] = {}
    last = None
    if not path.is_file():
        return pins, index
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line)
        if match:
            last = normalize(match.group(1))
            pins[last] = match.group(2)
            continue
        match = re.match(r"^\s+# from (\S+)", line)
        if match and last is not None:
            index[last] = match.group(1)
    return pins, index


def cmd_freeze(args: argparse.Namespace) -> int:
    """Compare the freeze with the expected pins and the resolution; write report/freeze-check.json."""
    say = Printer(args.lines)
    frozen, _ = read_pins(Path(args.freeze))
    resolved, index = read_pins(Path(args.compiled))
    expected = dict(item.split("=", 1) for item in args.expect)
    mismatches = [f"{name} {frozen.get(normalize(name))} (expected {want})" for name, want in expected.items()
                  if frozen.get(normalize(name)) != want]
    by_index: dict[str, list[str]] = {}
    for name, url in sorted(index.items()):
        by_index.setdefault(url, []).append(name)
    differ = sorted(name for name in set(frozen) | set(resolved) if frozen.get(name) != resolved.get(name))
    summary = {"frozen": frozen, "resolved": resolved, "index": index, "expected": expected,
               "mismatches": mismatches, "freeze_vs_resolution": differ}
    Path(args.report, "freeze-check.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    say(f"freeze: {len(frozen)} packages; expected pins matching: {len(expected) - len(mismatches)} of "
        f"{len(expected)}" + (f"; differ: {'; '.join(mismatches)}" if mismatches else ""))
    parts = [f"{url}: {len(names)}" + (f" ({', '.join(names)})" if len(names) <= 3 else "")
             for url, names in sorted(by_index.items(), key=lambda kv: len(kv[1]))]
    for line in wrap("  indexes (uv pip compile annotations): ", parts or ["none recorded"])[:2]:
        say(line)
    say(f"  freeze vs resolution: {'identical' if not differ else 'differ in ' + ', '.join(differ[:8])}")
    return 0


# ---------------------------------------------------------------- ascii


def cmd_ascii(args: argparse.Namespace) -> int:
    """Replace non-ASCII bytes in every file under --dir with '?'; print the count of files changed."""
    changed = 0
    for path in sorted(Path(args.dir).rglob("*")):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            if any(byte > 127 for byte in data):
                path.write_bytes(bytes(byte if byte < 128 else 63 for byte in data))
                changed += 1
    print(f"ascii: {changed} file(s) had non-ASCII bytes replaced")
    return 0


def main(argv: list[str]) -> int:
    """Parse the command line and run one command."""
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="command", required=True)
    imports = sub.add_parser("imports")
    imports.add_argument("--venv", required=True)
    imports.add_argument("--raw", required=True)
    imports.add_argument("--report", required=True)
    imports.add_argument("--lines", type=int, default=8)
    imports.add_argument("modules", nargs="+")
    hip = sub.add_parser("hip")
    hip.add_argument("--rocm", required=True)
    hip.add_argument("--source", required=True)
    hip.add_argument("--strace", default="")
    hip.add_argument("--raw", required=True)
    hip.add_argument("--report", required=True)
    hip.add_argument("--lines", type=int, default=23)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--freeze", required=True)
    freeze.add_argument("--compiled", required=True)
    freeze.add_argument("--expect", action="append", default=[])
    freeze.add_argument("--report", required=True)
    freeze.add_argument("--lines", type=int, default=4)
    ascii_ = sub.add_parser("ascii")
    ascii_.add_argument("--dir", required=True)
    args = top.parse_args(argv)
    return {"imports": cmd_imports, "hip": cmd_hip, "freeze": cmd_freeze, "ascii": cmd_ascii}[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
