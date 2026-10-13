"""Remote tests of the HIP toolchain `hipcc-gfx942` on the build host (task P17.6).

Bible: Toolchain Pins (a host compiler pinned by version and path; the
--version check against EXPECT_VERSION in the compile sandbox before the
first build), Execution Backends (none row: compile only), Sandbox
(compiles: a private /dev with no GPU node), Host Facts (ROCm), Agent
Rules 6, 7, and 10; OQ-002 (nothing opens an AMD device). The contract is in
test_hipcc.py's docstring.

The tests run in this file's order, and the rx command runs pytest with -x,
so a failure stops the run before the fixture capture that follows it:

1. build_toolchain("hipcc-gfx942", $LASSI_TOOLCHAINS), with the compile
   runner it builds made to run each command under strace -f, runs exactly
   one command, the --version check `<EXECUTABLE> --offload-arch=gfx942
   --version` (the class's VERSION_ARGS), which prints the pin's
   EXPECT_VERSION, starts no device-query program, and touches no GPU path,
   read as in 3. hipcc given no target runs rocm_agent_enumerator before it
   prints its version (ROCm/llvm-project tag therock-7.12,
   amd/hipcc/src/hipBin_amd.h:475, :568-581, :779-792; read from source).
   This test is first, so -x stops the run before the untraced checks below
   repeat the same command.
2. build_toolchain("hipcc-gfx942", $LASSI_TOOLCHAINS) ran the pinned hipcc
   (toolchains/hipcc.pin EXECUTABLE) through the compile sandbox, which
   printed the pin's EXPECT_VERSION, with exactly PATH, LANG, and LC_ALL
   set (no ROCm variable); a copy of the pin with another EXPECT_VERSION is
   refused there.
3. The kept program (fixtures/hipcc/sources/hip_clean/main.hip) builds once
   with the toolchain's own command under strace -f inside the toolchain's
   own compile runner. The build starts no device-query program (each
   execve is read by the program path it starts, never by its argv, which
   carries --offload-arch=gfx942) and touches no GPU path (each traced path
   by prefix). This is the first measurement of a hipcc link: P17.1 traced
   a compile with -c only (plans/spikes/p17-frameworks.md, Results 5).
4. The kept program builds through build(): an artifact and no error.
5. That artifact holds gfx942 device code and links the HIP runtime: its
   .hip_fatbin section is a clang offload bundle with a host entry and
   exactly one device entry, for gfx942, an ELF whose e_machine is
   EM_AMDGPU (224) and whose e_flags low byte is
   EF_AMDGPU_MACH_AMDGCN_GFX942 (0x4c) (llvm/llvm-project llvmorg-21.1.0
   llvm/include/llvm/BinaryFormat/ELF.h:321, :759, :841, as the P17.1 spike
   cites them), read with a standard-library reader copied from
   plans/spikes/p17-frameworks/inspect_bundle.py; the host binutils'
   `readelf -d -W` lists a NEEDED libamdhip64 entry. RUNPATH is not
   asserted.

Nothing here runs a built program, and no command names a device; every
hipcc command names the target, and the only programs started are hipcc
and what it starts, strace, and readelf (Agent Rules 6, 7, and 10; OQ-002).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools, readelf, and strace on PATH, strace outside every hidden
root, a reachable user systemd manager, $LASSI_SCRATCH, $LASSI_RUNS_ROOT,
and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH, and the pin's
EXECUTABLE present); with LASSI_REQUIRE_SANDBOX=1 they fail instead, so a
silent skip never passes for evidence. Run them, in the same rx run as the
fixture capture, with `uv run tools/rx.py run -- '<command>'`, the command
holding `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -x -p no:cacheprovider -m
remote tests/toolchains/test_hipcc_remote.py`. Every file they write lies
in a temp directory under $LASSI_RUNS_ROOT (the build root of every compile
here), removed afterwards, or in pytest's temp directory under TMPDIR on
the scratch disk, of which pytest keeps the last three (Agent Rule 7). No
value here is a measurement.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import pytest

from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.runner import BuiltToolchain, RunError, build_toolchain
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS_DIR = REPO / "toolchains"
KEPT_PROGRAM = Path(__file__).resolve().parent / "fixtures" / "hipcc" / "sources" / "hip_clean" / "main.hip"
NAME = "hipcc-gfx942"
PIN_NAME = "hipcc"
TARGET = "gfx942"
# The --version check's arguments after the executable (HipccGfx942.VERSION_ARGS): the target first.
VERSION_ARGS = [f"--offload-arch={TARGET}", "--version"]
SOURCE = "main.hip"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "readelf", "strace")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# The roots every compile hides (lassi.core.runner); a tool under one of them is not in a compile's view.
HIDDEN_VARIABLES = ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT")
# A traced build's wall limit in seconds; the traced -c compile took 2.3 s in P17.1 (Results 5).
TRACE_TIMEOUT_S = 600.0
# Copied from plans/spikes/p17-frameworks/frameworks.py (STRACE_ARGS): strace's first child is a shell that always
# exits normally, so strace never re-raises a fatal signal of the compiler with its own core limit lowered
# (plans/LESSONS.md, alpha01); the log goes to the build dir.
TRACE_LOG = "trace.txt"
STRACE_ARGS = [
    "-f", "-qq", "-s", "256", "-e", "trace=%file", "-e", "signal=none", "-o", TRACE_LOG, "--",
    "/bin/sh", "-c", '"$@"; exit "$?"', "sh",
]  # fmt: skip
# Programs that ask the host which GPUs it has (OQ-002), matched by the basename of the program an execve starts.
DEVICE_QUERY_PROGRAMS = frozenset(
    {"amdgpu-arch", "offload-arch", "rocm_agent_enumerator", "rocminfo", "rocm-smi", "amd-smi"}
)
# Paths that would mean an attempt to reach an AMD GPU or read a PCI device, matched by prefix (not by a word such
# as "amdgpu", which a link may read in a library name like libdrm_amdgpu).
GPU_PATH_PREFIXES = (
    "/dev/kfd",
    "/dev/dri",
    "/sys/class/kfd",
    "/sys/class/drm",
    "/sys/devices/virtual/kfd",
    "/sys/devices/pci",
    "/sys/bus/pci",
    "/sys/module/amdgpu",
    "/proc/bus/pci",
    "/proc/driver",
)
# One strace line: an optional pid, the call, and its arguments; and every quoted string in it (strace escapes a
# quote inside a string with a backslash).
TRACE_CALL = re.compile(r"^(?:\d+\s+)?(?P<call>\w+)\((?P<rest>.*)$")
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
# One readelf -d entry that matters here, for example "(NEEDED)  Shared library: [libamdhip64.so.7]".
DYNAMIC_ENTRY = re.compile(r"\((NEEDED|RUNPATH|RPATH)\)\s.*\[(.*)\]")
# The clang offload bundle and the ELF facts the spike's reader checks (inspect_bundle.py).
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
EM_AMDGPU = 224
MACH_GFX942 = 0x4C


def strace_path() -> str:
    """Return the strace on PATH, resolved, or "" when there is none."""
    found = shutil.which("strace")
    return os.path.realpath(found) if found else ""


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    strace = Path(strace_path())
    for variable in HIDDEN_VARIABLES:
        hidden = Path(os.environ.get(variable) or "/nonexistent-hidden-root").resolve()
        if hidden == strace or hidden in strace.parents:
            return f"strace ({strace}) lies under ${variable}, which a compile does not see"
    try:
        executable = pins_module.read_pin(PIN_NAME).get("EXECUTABLE", "")
    except (OSError, ValueError) as error:
        return f"toolchains/hipcc.pin cannot be read: {error}"
    if not executable or not Path(executable).is_file():
        return f"the pinned hipcc {executable!r} (EXECUTABLE in toolchains/hipcc.pin) is not on this host"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can build HIP programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def build_root() -> Iterator[Path]:
    """Create the build root of every compile here, a temp directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    root = Path(tempfile.mkdtemp(prefix="lassi-hipcc.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.fixture(scope="module")
def built(build_root: Path) -> BuiltToolchain:
    """Build hipcc-gfx942 as a run builds it: the pinned hipcc, checked by --version in the compile sandbox."""
    return build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]), build_root=build_root)


def attempt_dir(build_root: Path, case: str) -> Path:
    """Create and return a fresh attempt build directory for `case` under the build root."""
    build = build_root / case / "attempt00" / "build"
    build.mkdir(parents=True)
    return build


def kept_program() -> str:
    """Return the kept HIP program, the hip_clean fixture source, as text."""
    return KEPT_PROGRAM.read_bytes().decode("ascii")


@pytest.fixture(scope="module")
def kept_build(built: BuiltToolchain, build_root: Path) -> tuple[BuildResult, Path]:
    """Build the kept program once through the toolchain's own build(); return the result and its build dir."""
    workdir = attempt_dir(build_root, "kept")
    return built.toolchain.build({SOURCE: kept_program()}, workdir), workdir


def copy_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy every file of toolchains/ into a temp pins directory and point the pin reader at it."""
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    for path in TOOLCHAINS_DIR.iterdir():
        if path.is_file():
            shutil.copyfile(path, pins_dir / path.name)
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    return pins_dir


def traced_calls(log: str) -> tuple[list[str], list[str]]:
    """Return the programs every execve in an strace -f log tried to start, and every other path it names.

    An execve or execveat line is read by its program path only (the
    first quoted string), whether the call succeeded or failed, never by
    its argv; any other line contributes each quoted string that is an
    absolute path. A line strace split ("<... resumed>") names its program
    on the first half. A path relative to a directory descriptor is not
    resolved, and an execveat of a descriptor with an empty path (fexecve)
    records an empty program name.
    """
    programs: list[str] = []
    paths: list[str] = []
    for line in log.splitlines():
        match = TRACE_CALL.match(line)
        if match is None:
            continue
        strings = QUOTED.findall(match["rest"])
        if match["call"] in ("execve", "execveat"):
            programs.extend(strings[:1])
        else:
            paths.extend(text for text in strings if text.startswith("/"))
    return programs, paths


def check_trace(log: str, what: str) -> None:
    """Assert the strace -f `log` of `what` covers hipcc and a clang, and no device-query program or GPU path.

    The pinned hipcc and a clang it starts must appear among the programs,
    which traced_calls reads by program path; no program's basename may be
    in DEVICE_QUERY_PROGRAMS, and no path may start with a GPU_PATH_PREFIXES
    entry.
    """
    programs, paths = traced_calls(log)
    assert pins_module.read_pin(PIN_NAME)["EXECUTABLE"] in programs, f"the trace of {what} covers the pinned hipcc"
    assert any(PurePosixPath(program).name.startswith("clang") for program in programs), "and the clang it starts"
    assert paths, f"the trace records the paths {what} named"
    queries = sorted({program for program in programs if PurePosixPath(program).name in DEVICE_QUERY_PROGRAMS})
    assert not queries, f"{what} started a device-query program (OQ-002): {queries}"
    touched = sorted({path for path in paths if path.startswith(GPU_PATH_PREFIXES)})
    assert not touched, f"{what} reached for a GPU path (OQ-002): {touched}"


def tracing_compile_runner(seen: list[tuple[list[str], str]]) -> type:
    """Return a SandboxedCompileRunner that runs each command under strace -f and appends (argv, trace) to `seen`."""

    class TracingCompileRunner(SandboxedCompileRunner):
        """The compile runner as lassi.core.runner builds it; each command runs under strace -f in its build dir."""

        def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
            """Run `argv` under strace -f in the sandbox, and keep its trace before the caller removes `cwd`."""
            result = super().__call__([strace_path(), *STRACE_ARGS, *argv], cwd, timeout_s)
            log = cwd / TRACE_LOG
            seen.append((list(argv), log.read_bytes().decode("utf-8", errors="replace") if log.is_file() else ""))
            return result

    return TracingCompileRunner


def elf_sections(data: bytes) -> dict[str, tuple[int, int]]:
    """Return {name: (offset, size)} from a little-endian ELF64 file's section headers (inspect_bundle.sections)."""
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + index * shentsize) for index in range(shnum)]
    strtab = headers[shstrndx][4]
    found: dict[str, tuple[int, int]] = {}
    for header in headers:
        end = data.index(b"\0", strtab + header[0])
        found[data[strtab + header[0] : end].decode("ascii")] = (header[4], header[5])
    return found


def bundle_entries(bundle: bytes) -> list[tuple[str, bytes]]:
    """Return (id, contents) for each entry of a clang offload bundle (inspect_bundle.entries)."""
    assert bundle.startswith(BUNDLE_MAGIC), "no clang offload bundle magic at the start of .hip_fatbin"
    (count,) = struct.unpack_from("<Q", bundle, len(BUNDLE_MAGIC))
    at = len(BUNDLE_MAGIC) + 8
    entries: list[tuple[str, bytes]] = []
    for _ in range(count):
        offset, size, id_size = struct.unpack_from("<QQQ", bundle, at)
        at += 24
        entries.append((bundle[at : at + id_size].decode("ascii"), bundle[offset : offset + size]))
        at += id_size
    return entries


def dynamic(path: Path) -> dict[str, list[str]]:
    """Return the NEEDED, RUNPATH, and RPATH entries the host's readelf -d -W reads from an ELF file; it never runs."""
    environment = {"PATH": os.environ["PATH"], "LANG": "C", "LC_ALL": "C"}
    done = subprocess.run(
        ["readelf", "-d", "-W", str(path)], capture_output=True, text=True, env=environment, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    entries: dict[str, list[str]] = {"NEEDED": [], "RUNPATH": [], "RPATH": []}
    for line in done.stdout.splitlines():
        match = DYNAMIC_ENTRY.search(line)
        if match:
            entries[match[1]].append(match[2])
    return entries


def test_the_version_check_names_the_target_and_starts_no_device_query_program(
    build_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], str]] = []
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", tracing_compile_runner(seen))
    refusal = ""
    try:
        traced: BuiltToolchain | None = build_toolchain(
            NAME, Path(os.environ["LASSI_TOOLCHAINS"]), build_root=build_root
        )
    except RunError as error:
        traced, refusal = None, str(error)
    pin = pins_module.read_pin(PIN_NAME)
    assert [argv for argv, _ in seen] == [[pin["EXECUTABLE"], *VERSION_ARGS]], "one command: the check, target named"
    check_trace(seen[0][1], "the --version check")
    assert traced is not None and traced.version_status == 0, f"the traced check was refused: {refusal}"
    assert pin["EXPECT_VERSION"] in "\n".join(traced.version), "hipcc prints the pinned version after a target"


def test_the_version_check_ran_the_pinned_hipcc_in_the_compile_sandbox(built: BuiltToolchain) -> None:
    pin = pins_module.read_pin(PIN_NAME)
    assert built.executable == pin["EXECUTABLE"]
    assert built.version_status == 0
    assert pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert built.pins == {PIN_NAME: pin}
    assert isinstance(built.toolchain.runner, SandboxedCompileRunner), "every compile runs in the compile sandbox"
    assert built.environment is not None and set(built.environment) == {"PATH", "LANG", "LC_ALL"}


def test_a_hipcc_whose_version_is_not_the_pin_is_refused_in_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins_dir = copy_pins(tmp_path, monkeypatch)
    text = (pins_dir / "hipcc.pin").read_text(encoding="ascii")
    expected = pins_module.read_pin(PIN_NAME)["EXPECT_VERSION"]
    changed = text.replace(expected, "PLACEHOLDER version text that no hipcc prints")
    assert changed != text
    (pins_dir / "hipcc.pin").write_text(changed, encoding="ascii", newline="\n")
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))


def test_the_build_starts_no_device_query_program_and_touches_no_gpu_path(
    built: BuiltToolchain, build_root: Path
) -> None:
    workdir = attempt_dir(build_root, "traced")
    (workdir / SOURCE).write_bytes(kept_program().encode("ascii"))
    command = built.toolchain.command([SOURCE])
    assert [word for word in command if "offload-arch" in word] == [f"--offload-arch={TARGET}"]
    result = built.toolchain.runner([strace_path(), *STRACE_ARGS, *command], workdir, TRACE_TIMEOUT_S)
    assert result.returncode == 0, f"the traced build failed:\n{result.stderr}"
    assert (workdir / OUTPUT).is_file(), "the traced build compiled and linked the program"
    check_trace((workdir / TRACE_LOG).read_bytes().decode("utf-8", errors="replace"), "the build")


def test_the_kept_program_builds_through_build(kept_build: tuple[BuildResult, Path]) -> None:
    result, workdir = kept_build
    stderr = (workdir / ATTACHMENT).read_bytes().decode("utf-8", errors="replace")
    assert result.artifact == workdir / OUTPUT, f"the kept program did not build:\n{stderr}"
    errors = [item for item in result.diagnostics if item.severity == "error"]
    assert errors == [], f"a clean build has no error:\n{stderr}"


def test_the_artifact_holds_gfx942_code_and_links_the_hip_runtime(kept_build: tuple[BuildResult, Path]) -> None:
    result, _ = kept_build
    assert result.artifact is not None
    data = result.artifact.read_bytes()
    found = elf_sections(data)
    assert ".hip_fatbin" in found, f"no .hip_fatbin section among {sorted(found)}"
    offset, size = found[".hip_fatbin"]
    entries = bundle_entries(data[offset : offset + size])
    hosts = [ident for ident, _ in entries if ident.startswith("host-")]
    devices = [(ident, body) for ident, body in entries if not ident.startswith("host-")]
    assert len(hosts) == 1, [ident for ident, _ in entries]
    assert len(devices) == 1 and devices[0][0].endswith(f"--{TARGET}"), [ident for ident, _ in entries]
    body = devices[0][1]
    assert body[:4] == b"\x7fELF", "the device entry is an ELF code object"
    (machine,) = struct.unpack_from("<H", body, 0x12)
    (flags,) = struct.unpack_from("<I", body, 0x30)
    assert (machine, flags & 0xFF) == (EM_AMDGPU, MACH_GFX942), f"e_machine {machine}, e_flags {flags:#x}"
    needed = dynamic(result.artifact)["NEEDED"]
    assert any(name.startswith("libamdhip64.so") for name in needed), f"NEEDED: {needed}"
