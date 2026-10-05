"""The ttsim executor, registered as Executor "ttsim" (bible Execution Backends, the ttsim row; ttsim Facts).

It runs a TT host program (built by the ttmetal-host toolchain) with its
inputs as arguments on ttsim, the pinned simulator of a Wormhole chip,
only through lassi.executors.sandbox (Agent Rule 6), as the native executor
runs a CPU program: the artifact's directory, resolved, is the run's workdir
and must lie inside the runs root; the hidden roots are $LASSI_SCRATCH and
$HOME unless configured, plus the runs root; the harness (config key
`harness`) and the toolchains root ($LASSI_TOOLCHAINS unless configured)
are mounted read-only. No device is ever opened: the program's kernels run
on ttsim's simulated cores inside the host process (Agent Rules 2 and 9).
Its capabilities are runs_code, sandboxed, and simulator
(lassi.core.capabilities SIMULATOR), so the stages add the Harness
Contract's hang diagnostic to a hung attempt and label its wall times as
simulator wall time.

Configuration (a recipe's executor section, such as `executor: {kind:
ttsim, arch: wormhole_b0, dispatch: slow}`): `arch` (only "wormhole_b0", the chip the
pinned library simulates), `dispatch` (only "slow", as the ttsim row sets),
`chips` (only the int 1, a single chip; a bool is not an int), and
`harness`. Any other value is a ValueError at construction naming the key
and the one supported value.

The pinned install, from toolchains/ttsim.pin and toolchains/tt-metal.pin,
under the resolved toolchains root: the library
<root>/<ttsim PREFIX_NAME>/<LIBRARY> with the SoC descriptor
<SOC_DESCRIPTOR> beside it, and the tt-metal tree
<root>/<tt-metal PREFIX_NAME>. Before anything runs or is created in the
workdir, run() refuses with SandboxUnavailableError naming the path: no
toolchains root, a missing library or descriptor, a library whose sha256 is
not the pin's SHA256, a descriptor whose sha256 is not
SOC_DESCRIPTOR_SHA256, and a tree whose lassi-install.txt does not name the
pin's NAME and COMMIT on its first line or whose lassi-cpm-sources.txt
differs from toolchains/tt-metal-cpm-sources.txt (the ttmetal-host
toolchain's own checks, lassi.toolchains.ttmetal_build).

The program's environment is exactly the ttsim row's (program_environment):
PATH=SANDBOX_PATH, TMPDIR=/tmp, LANG=C and LC_ALL=C (so the kernel
compiler's messages stay ASCII), TT_METAL_SIMULATOR on the library,
TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_DISABLE_SFPLOADMACRO=1,
TT_METAL_RUNTIME_ROOT on the tree, TT_METAL_CACHE and TT_METAL_LOGS_PATH in
<workdir>/@ttsim/ (RESERVED_DIR; no model file path may start with "@",
lassi.core.files, so none can take its place), created before the run,
TT_METAL_INSPECTOR_RPC=0, and TT_METAL_THREADCOUNT=<Limits.cpus>, which
sizes the host's thread pool to the run's CPUs. There is no HOME and nothing
from the caller's environment. The kernel JIT's cache stays in the workdir,
so no run touches a shared cache (PHASE-NOTES P4, the collision rule). The
working directory is the workdir, where the JIT finds a kernel at the path
the program names. output_files are the regular files new after the run,
as native gives them, but none under @ttsim.

What the executor reads from a run (task P4.6's RunResult fields):

- Findings (read_findings; OQ-036 option (a)). ttsim prints each on stdout
  as `[<cycle>] ERROR: <class>: <function>: <message>`; every such line
  counts, in order. UB_CLASSES are undefined behavior (sim_ub True); every
  other class, GAP_CLASSES and any class not listed, is a gap, and sim_gap
  is the first gap's class in stdout order. sim_ub is False when no
  undefined behavior appeared, since the executor always checks. Each
  finding is also one Diagnostic(stage "run", severity "error", code the
  class, message "<class>: <function>: <message>").
- Kernel JIT errors (read_jit_diagnostics). A failed kernel build prints
  "<processor> build failed. Log:" and the kernel compiler's output; each
  `<path>:<line>:<column>: <severity>: <message>` line of that log, up to
  its first blank line, becomes a Diagnostic(stage "jit") with the severity
  as printed (fatal error as error), the file relative to the workdir when
  it lies inside it and as printed otherwise, the line, the column, and a
  trailing ` [-W...]` option as the code. Both streams are read, and the
  same diagnostic in both counts once. A line of that form outside such a
  log is not read. The host aborts (status 134); the stages read the
  diagnostics, not the status.
- Hang and Watcher (OQ-037 option (b); Harness Contract). When the sandbox
  reports a hang, the program runs once more with TT_METAL_WATCHER=1 added
  and everything else the same, under the same limits, in
  <workdir>/@ttsim/watcher/, a fresh copy of the workdir's pre-run regular
  files (outside @ttsim) with its own @ttsim cache and logs, so the scored
  run's streams and output files are never touched. The sha256 of every
  pre-run file is taken before the scored run; if any of them changed or
  vanished by the rerun (or the artifact is not one of them), there is no
  rerun. From the last WATCHER_READ_BYTES of the copy's
  logs/generated/watcher/watcher.log, watcher_note condenses the last
  complete dump into one Diagnostic(stage "run", severity "note", code
  lassi.core.capabilities WATCHER_CODE), at most WATCHER_BYTES_CAP bytes.
  No log or no complete dump: no note. The rerun's own findings, streams,
  and status are never read. Every hang is rerun, a reference run's too;
  the stages decide what to say (lassi.core.stages run_findings).

Every message keeps printable ASCII only; any other character becomes "?".
Nothing the model wrote is muted: the finding lines, the kernel paths, and
the Watcher lines are run output. The host program runs in the same process
as the simulator, so it can print a line that reads as a finding or as a
kernel build failure; the executor cannot tell such a line from ttsim's or
the JIT's own, and does not try.

device() names ttsim and its pin, and the pinned tt-metal, as a simulator,
from the pins alone. The executor declares `pins` (ttsim and tt-metal, as
read from the pin files at construction), which the runner records in each
trial's toolchain_pins and in provenance.json (Agent Rule 10).
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from lassi.core.capabilities import SIMULATOR, WATCHER_CODE
from lassi.core.interfaces import Limits, RunResult
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.executors.native import _absolute_variable, _checked_workdir, _environment_roots, _regular_files, _resolved
from lassi.executors.sandbox import SANDBOX_PATH, Sandbox, SandboxResult, SandboxSpec, SandboxUnavailableError
from lassi.executors.workdir import runs_root
from lassi.toolchains import pins as pin_files
from lassi.toolchains._stderr import GCC, NO_FILES, parse_stderr
from lassi.toolchains.ttmetal_build import check_cpm_sources, check_install_record

# The one supported value of each config key (the bible's ttsim row): the chip the pinned library simulates, the
# dispatch mode, and the chip count.
ARCH = "wormhole_b0"
DISPATCH = "slow"
CHIPS = 1
# The pin files the executor reads, and the keys it needs from each.
TTSIM_PIN = "ttsim"
TT_METAL_PIN = "tt-metal"
_TTSIM_KEYS = ("VERSION", "PREFIX_NAME", "LIBRARY", "SHA256", "SOC_DESCRIPTOR", "SOC_DESCRIPTOR_SHA256")
_TT_METAL_KEYS = ("NAME", "VERSION", "COMMIT", "PREFIX_NAME")
# The environment variable that names the default toolchains root.
_TOOLCHAINS_VARIABLE = "LASSI_TOOLCHAINS"
# The executor's own directory in a workdir, and the JIT cache, logs, and Watcher copy under it. No model file path
# may start with "@" (lassi.core.files), as for the compile sandbox's private TMPDIR, "@lassi-tmp".
RESERVED_DIR = "@ttsim"
CACHE_DIR = "cache"
LOGS_DIR = "logs"
WATCHER_DIR = "watcher"
# Where tt-metal writes Watcher's log under TT_METAL_LOGS_PATH (the P4.9 captures), and the variable that turns
# Watcher on for the rerun of a hang.
WATCHER_LOG = ("generated", "watcher", "watcher.log")
WATCHER_VARIABLE = "TT_METAL_WATCHER"
# The Watcher note's cap in bytes (all ASCII): enough for the busy cores and kernel paths of a small grid, well under
# the correction prompt's DIAGNOSTIC_BYTES_CAP (lassi.core.stages). A longer note is cut on a line boundary.
WATCHER_BYTES_CAP = 4096
# How many bytes at the end of the Watcher log are read. One dump of every core takes about 13 KB (the P4.9 capture:
# 1637305 bytes for 128 dumps), so the last 1 MiB holds the last complete dumps, and a log a program swelled to the
# disk cap is never read whole.
WATCHER_READ_BYTES = 1 << 20
# ttsim's finding classes (OQ-036 option (a); the P4.9 captures). Any class not in UB_CLASSES is a gap.
UB_CLASSES = frozenset({"UndefinedBehavior", "NonContractualBehavior", "UnpredictableValueUsed"})
GAP_CLASSES = frozenset(
    {
        "UnimplementedFunctionality",
        "UnsupportedFunctionality",
        "UntestedFunctionality",
        "SystemError",
        "ConfigurationError",
        "AssertionFailure",
    }
)
# The head of a finding line, `[<cycle>] ERROR: <class>: `, then `<function>: <message>` (the fixtures:
# "[4500] ERROR: NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4"). The bounded counts keep a
# search linear in the line's length.
_FINDING = re.compile(r"\[[0-9]{1,20}\] ERROR: (?P<cls>[A-Za-z_][A-Za-z0-9_]{0,127}): ")
# The JIT's failed kernel build, "<processor> build failed. Log: <first log line>" (fixture jit-error: "brisc build
# failed. Log: In file included from ../kernel_includes.hpp:1,").
_BUILD_FAILED = re.compile(r"[A-Za-z0-9_]{1,64} build failed\. Log:(?P<rest>.*)")
# A Watcher dump's first and last lines: "Dump #128 at 130.156s" and "Dump #128 completed at 130.168s".
_DUMP_START = re.compile(r"Dump #(?P<number>[0-9]{1,20}) at ")
_DUMP_END = re.compile(r"Dump #(?P<number>[0-9]{1,20}) completed at ")
# A core line ends "k_ids:  1|  0|  0|  0|  0" (one id per processor); a kernel line reads "k_id[  1]: <path>".
_CORE_MARK = " core("
_CORE_IDS = "k_ids:"
_ID_LIST = re.compile(r"[0-9 |]+")
_KERNEL_ID = re.compile(r"k_id\[ *(?P<id>[0-9]{1,20})\]: ")


def printable(text: str) -> str:
    """Return `text` with every character that is not printable ASCII written as "?"."""
    return "".join(character if character.isascii() and character.isprintable() else "?" for character in text)


def _pin(name: str, keys: Sequence[str]) -> dict[str, str]:
    """Return toolchains/<name>.pin's pairs; ValueError naming the file when one of `keys` is missing or empty."""
    pin = pin_files.read_pin(name)
    missing = [key for key in keys if not pin.get(key)]
    if missing:
        raise ValueError(f"toolchains/{name}.pin has no {', '.join(missing)}, which the ttsim executor needs")
    return pin


@dataclass(frozen=True)
class TtsimInstall:
    """The checked install a run uses: the resolved toolchains root, the library, the descriptor, and the tree."""

    root: Path
    library: Path
    descriptor: Path
    tree: Path


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file's bytes, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(path: Path, pin: Mapping[str, str], key: str) -> Path:
    """Return `path` when it is a regular file whose sha256 is ttsim.pin's `key`; SandboxUnavailableError otherwise.

    The message names the path.
    """
    expected = pin[key]
    if not path.is_file():
        raise SandboxUnavailableError(
            f"the pinned ttsim file {path} is missing; install it on the build host with toolchains/ttsim.sh; "
            "nothing ran"
        )
    found = _sha256(path)
    if found != expected:
        raise SandboxUnavailableError(
            f"{path} is not the pinned file: its sha256 is {found}, and toolchains/{TTSIM_PIN}.pin's {key} is "
            f"{expected}; nothing ran"
        )
    return path


def check_install(root: Path, pins: Mapping[str, Mapping[str, str]]) -> TtsimInstall:
    """Return the install under the toolchains root `root` after checking it against `pins`; nothing is created.

    `root` is used as given (the executor resolves it first). The library
    and the SoC descriptor must be regular files with the pin's SHA256 and
    SOC_DESCRIPTOR_SHA256, and the tree must be the pinned tt-metal install
    (lassi.toolchains.ttmetal_build check_install_record and
    check_cpm_sources). Raise SandboxUnavailableError naming the path of
    the first problem.
    """
    ttsim, metal = pins[TTSIM_PIN], pins[TT_METAL_PIN]
    prefix = root / ttsim["PREFIX_NAME"]
    library = _checked_file(prefix / ttsim["LIBRARY"], ttsim, "SHA256")
    descriptor = _checked_file(prefix / ttsim["SOC_DESCRIPTOR"], ttsim, "SOC_DESCRIPTOR_SHA256")
    tree = root / metal["PREFIX_NAME"]
    try:
        check_install_record(tree, metal)
        check_cpm_sources(tree)
    except ValueError as error:
        message = f"the tt-metal tree {tree} is not the pinned install: {error}; nothing ran"
        raise SandboxUnavailableError(message) from None
    return TtsimInstall(root=root, library=library, descriptor=descriptor, tree=tree)


def program_environment(install: TtsimInstall, workdir: Path, limits: Limits) -> dict[str, str]:
    """Return a program's environment in `workdir`: the ttsim row's settings from `install`, nothing else.

    The JIT cache and the logs are <workdir>/@ttsim/cache and
    <workdir>/@ttsim/logs, and TT_METAL_THREADCOUNT is limits.cpus (see the
    module docstring). No HOME; the caller's environment is never read.
    """
    reserved = workdir / RESERVED_DIR
    return {
        "PATH": SANDBOX_PATH,
        "TMPDIR": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "TT_METAL_SIMULATOR": str(install.library),
        "TT_METAL_SLOW_DISPATCH_MODE": "1",
        "TT_METAL_DISABLE_SFPLOADMACRO": "1",
        "TT_METAL_RUNTIME_ROOT": str(install.tree),
        "TT_METAL_CACHE": str(reserved / CACHE_DIR),
        "TT_METAL_LOGS_PATH": str(reserved / LOGS_DIR),
        "TT_METAL_INSPECTOR_RPC": "0",
        "TT_METAL_THREADCOUNT": str(limits.cpus),
    }


def read_findings(stdout: str) -> tuple[list[Diagnostic], bool, str | None]:
    """Return ttsim's findings in `stdout`: their Diagnostics in order, sim_ub, and sim_gap.

    A finding is a line holding `[<cycle>] ERROR: <class>: <function>:
    <message>` (the first such head in the line; a line whose rest has no
    ": " is not one). It becomes Diagnostic(stage "run", severity "error",
    code the class, message "<class>: <function>: <message>" in printable
    ASCII). sim_ub is True when any class is in UB_CLASSES, else False;
    sim_gap is the first other class, None when there is none.
    """
    diagnostics: list[Diagnostic] = []
    sim_ub, sim_gap = False, None
    for line in stdout.split("\n"):
        text = line.rstrip("\r")
        match = _FINDING.search(text)
        if match is None:
            continue
        function, colon, message = text[match.end() :].partition(": ")
        if not colon:
            continue
        cls = match["cls"]
        rendered = printable(f"{cls}: {function}: {message}")
        diagnostics.append(Diagnostic(stage="run", severity="error", code=cls, message=rendered))
        if cls in UB_CLASSES:
            sim_ub = True
        elif sim_gap is None:
            sim_gap = cls
    return diagnostics, sim_ub, sim_gap


def _build_logs(stream: str) -> list[str]:
    """Return each kernel build log in `stream`: from a "<processor> build failed. Log:" line to its first blank line.

    The text after "Log:" on the opening line is the log's first line.
    """
    logs: list[str] = []
    current: list[str] | None = None
    for line in stream.split("\n"):
        header = _BUILD_FAILED.search(line)
        if header is not None:
            if current is not None:
                logs.append("\n".join(current))
            current = [header["rest"].strip()]
        elif current is not None and not line.strip():
            logs.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        logs.append("\n".join(current))
    return logs


def _in_workdir(file: str, workdir: Path) -> str:
    """Return `file` relative to `workdir` when it names a path inside it, else `file` as printed."""
    base = workdir.as_posix().rstrip("/") + "/"
    normal = posixpath.normpath(file)
    return normal[len(base) :] if normal.startswith(base) and len(normal) > len(base) else file


def read_jit_diagnostics(stdout: str, stderr: str, workdir: Path) -> list[Diagnostic]:
    """Return the kernel JIT's Diagnostics (stage "jit") from the build logs in both streams, each once, in order.

    Each `<path>:<line>:<column>: <severity>: <message>` line of a build log
    (_build_logs; GCC source echo and caret lines skipped) is read as the
    shared GCC style pattern reads it (lassi.toolchains._stderr GCC): the
    severity as printed with fatal error as error, a trailing ` [-W...]`
    option as the code, and column 0 as none. The file is made relative to
    the resolved `workdir` when it lies inside it (_in_workdir). Code,
    file, and message keep printable ASCII only. stdout's logs come first; a
    Diagnostic equal to one already found is left out.
    """
    found: list[Diagnostic] = []
    for stream in (stdout, stderr):
        for log in _build_logs(stream):
            for item in parse_stderr(log, NO_FILES, (GCC,)):
                file = None if item.file is None else printable(_in_workdir(item.file, workdir))
                code = None if item.code is None else printable(item.code)
                diagnostic = replace(item, stage="jit", code=code, file=file, message=printable(item.message))
                if diagnostic not in found:
                    found.append(diagnostic)
    return found


def _last_complete_dump(lines: Sequence[str]) -> tuple[str, list[str]] | None:
    """Return the number and the body lines of the last dump that has both its start and its completion line."""
    last: tuple[str, list[str]] | None = None
    opened: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        end = _DUMP_END.match(line)
        if end is not None:
            if opened is not None and opened[0] == end["number"]:
                last = (end["number"], list(lines[opened[1] + 1 : index]))
            opened = None
            continue
        start = _DUMP_START.match(line)
        if start is not None:
            opened = (start["number"], index)
    return last


def _busy_ids(line: str) -> set[str]:
    """Return the nonzero kernel ids a core line's k_ids list holds (leading zeros dropped); none for another line."""
    if _CORE_MARK not in line:
        return set()
    _, found, ids = line.rpartition(_CORE_IDS)
    if not found or not _ID_LIST.fullmatch(ids):
        return set()
    return {value.strip().lstrip("0") for value in ids.split("|")} - {""}


def _kept_lines(body: Sequence[str]) -> list[str]:
    """Return a dump's core lines that hold a nonzero kernel id, then the k_id lines of those ids, in log order."""
    cores: list[str] = []
    ids: set[str] = set()
    for line in body:
        busy = _busy_ids(line)
        if busy:
            cores.append(line)
            ids |= busy
    kernels = []
    for line in body:
        match = _KERNEL_ID.match(line)
        if match is not None and (match["id"].lstrip("0") or "0") in ids:
            kernels.append(line)
    return [printable(line.rstrip()) for line in (*cores, *kernels)]


def watcher_note(log: str) -> Diagnostic | None:
    """Return the Watcher note for a Watcher log, or None when the log holds no complete dump.

    The note is Diagnostic(stage "run", severity "note", code WATCHER_CODE)
    whose message is "Watcher dump #<n>: " and the last complete dump's kept
    lines (_kept_lines) joined by "; ", or "no core held a kernel" when none
    is kept. When that passes WATCHER_BYTES_CAP bytes, only the first kept
    lines that fit are shown, followed by "; ... <count> more lines left
    out", so no line is cut in the middle.
    """
    dump = _last_complete_dump([line.rstrip("\r") for line in log.split("\n")])
    if dump is None:
        return None
    number, body = dump
    lines = _kept_lines(body)
    head = f"Watcher dump #{number}: "
    message = head + ("; ".join(lines) if lines else "no core held a kernel")
    if len(message) > WATCHER_BYTES_CAP:
        message = _capped(head, lines)
    return Diagnostic(stage="run", severity="note", code=WATCHER_CODE, message=message)


def _left_out(count: int) -> str:
    """Return the part of a capped Watcher note that says how many kept lines it leaves out."""
    return f"... {count} more lines left out"


def _capped(head: str, lines: Sequence[str]) -> str:
    """Return `head` and the first `lines` that fit in WATCHER_BYTES_CAP with the left-out count, joined by "; "."""
    used, shown = len(head), 0
    while shown < len(lines):
        more = used + len(lines[shown]) + 2
        if more + len(_left_out(len(lines) - shown - 1)) > WATCHER_BYTES_CAP:
            break
        used, shown = more, shown + 1
    return head + "; ".join([*lines[:shown], _left_out(len(lines) - shown)])


def _read_tail(path: Path, size: int) -> str | None:
    """Return the last `size` bytes of a regular file as text (UTF-8, errors replaced); None for no such file."""
    if path.is_symlink() or not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, handle.seek(0, os.SEEK_END) - size))
        return handle.read(size).decode("utf-8", errors="replace")


def _is_reserved(relative: str) -> bool:
    """Return True for a workdir path (relative, POSIX) at or under RESERVED_DIR."""
    return relative == RESERVED_DIR or relative.startswith(RESERVED_DIR + "/")


def _pre_run_hashes(workdir: Path) -> dict[str, str]:
    """Return the sha256 of every regular file in `workdir` outside RESERVED_DIR, by relative POSIX path."""
    files = _regular_files(workdir)
    return {relative: _sha256(path) for relative, path in sorted(files.items()) if not _is_reserved(relative)}


def _remove(path: Path) -> None:
    """Remove a file, a symbolic link, or a directory tree at `path`; nothing when there is none."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _fresh_copy(workdir: Path, hashes: Mapping[str, str]) -> Path | None:
    """Return <workdir>/@ttsim/watcher, made anew with each pre-run file whose bytes still have its hash.

    Each file is copied with its permission bits and its copy hashed; when a
    pre-run file is gone, is no longer a regular file, or its copy's sha256
    differs from the one taken before the scored run, the copy is removed
    and None is returned.
    """
    copy = workdir / RESERVED_DIR / WATCHER_DIR
    _remove(copy)
    copy.mkdir(parents=True)
    for relative, digest in sorted(hashes.items()):
        source, target = workdir / relative, copy / relative
        if source.is_symlink() or not source.is_file():
            _remove(copy)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copymode(source, target)
        if _sha256(target) != digest:
            _remove(copy)
            return None
    return copy


@dataclass(frozen=True)
class _Mounts:
    """What every sandboxed run of one run() call sees: the hidden roots, the harness, and the checked install."""

    hidden_roots: tuple[Path, ...]
    harness: Path | None
    install: TtsimInstall


@register("Executor", "ttsim")
class TtsimExecutor:
    """Runs a TT host program on ttsim, the pinned Wormhole simulator, only inside the sandbox.

    The config keys are `arch`, `dispatch`, `chips`, and `harness` (see the
    module docstring); the keyword settings `hidden_roots`, `toolchains`,
    and `sandbox` are for the runner and tests, as NativeExecutor has them
    (None means $LASSI_SCRATCH and $HOME, $LASSI_TOOLCHAINS, and a Sandbox
    with the default runner). `pins` holds toolchains/ttsim.pin's and
    toolchains/tt-metal.pin's pairs, read at construction.
    """

    name = "ttsim"
    capabilities = frozenset({"runs_code", "sandboxed", SIMULATOR})
    config_keys = frozenset({"arch", "dispatch", "chips", "harness"})

    def __init__(
        self,
        *,
        arch: str = ARCH,
        dispatch: str = DISPATCH,
        chips: int = CHIPS,
        harness: str | None = None,
        hidden_roots: Sequence[str] | None = None,
        toolchains: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        """Check the configuration and keep the settings; ValueError names a refused key and its one supported value.

        factory() with no argument works on any host: nothing is checked on
        disk until run(). The pin files must hold the keys the executor
        reads (ValueError naming the file otherwise).
        """
        if arch != ARCH:
            raise ValueError(f"arch must be {ARCH!r}, the chip the pinned ttsim library simulates; got {arch!r}")
        if dispatch != DISPATCH:
            raise ValueError(f"dispatch must be {DISPATCH!r}, as the ttsim row runs it; got {dispatch!r}")
        if isinstance(chips, bool) or not isinstance(chips, int) or chips != CHIPS:
            raise ValueError(f"chips must be the integer {CHIPS}: ttsim simulates a single chip; got {chips!r}")
        if isinstance(hidden_roots, (str, os.PathLike)):
            raise ValueError(f"hidden_roots must be a sequence of paths, got {hidden_roots!r}")
        self.harness: Path | None = None if harness is None else Path(harness)
        self.hidden_roots: tuple[Path, ...] | None = (
            None if hidden_roots is None else tuple(Path(root) for root in hidden_roots)
        )
        self.toolchains: Path | None = None if toolchains is None else Path(toolchains)
        self.sandbox: Sandbox = Sandbox() if sandbox is None else sandbox
        self.pins: dict[str, dict[str, str]] = {
            TTSIM_PIN: _pin(TTSIM_PIN, _TTSIM_KEYS),
            TT_METAL_PIN: _pin(TT_METAL_PIN, _TT_METAL_KEYS),
        }

    def device(self) -> str:
        """Return the device, one line of printable ASCII from the pins alone; no process, no sandbox.

        For example "ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on
        the host CPU, with tt-metal 5280a9cf: a simulator, not silicon"
        (ttsim.pin VERSION and LIBRARY, tt-metal.pin VERSION; Agent Rule 2).
        """
        ttsim, metal = self.pins[TTSIM_PIN], self.pins[TT_METAL_PIN]
        text = (
            f"ttsim {ttsim['VERSION']} ({ttsim['LIBRARY']}, a virtual Wormhole) on the host CPU, "
            f"with tt-metal {metal['VERSION']}: a simulator, not silicon"
        )
        return printable(" ".join(text.split()))

    def install(self) -> TtsimInstall:
        """Return the checked install under the resolved toolchains root (check_install); nothing is created.

        The root is the configured `toolchains`, else $LASSI_TOOLCHAINS.
        Raise SandboxUnavailableError when neither is set, when
        $LASSI_TOOLCHAINS is relative, or when check_install refuses.
        """
        root = self.toolchains
        if root is None:
            root = _absolute_variable(_TOOLCHAINS_VARIABLE, "the root of the pinned ttsim and tt-metal installs")
        if root is None:
            raise SandboxUnavailableError(
                f"no toolchains root: set ${_TOOLCHAINS_VARIABLE} or pass toolchains; the ttsim executor runs "
                "programs against the pinned installs there; nothing ran"
            )
        return check_install(_resolved(root), self.pins)

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` on ttsim in the sandbox under `limits`; return its RunResult and findings.

        A relative artifact, a workdir outside the runs root, or a string of
        inputs raises ValueError; no hidden root, or an install check that
        fails (install), raises SandboxUnavailableError; all before anything
        runs or is created in the workdir. Then @ttsim/cache and
        @ttsim/logs are created, the program runs with program_environment's
        variables and argv [artifact, *inputs], and the RunResult carries the
        sandbox's status, streams, flags, and wall time, the output files
        (none under @ttsim), sim_ub and sim_gap (read_findings), and the
        diagnostics: the findings, then the kernel JIT's
        (read_jit_diagnostics), then, after a hang, the Watcher note of the
        rerun (_watcher_rerun), when there is one. SandboxUnavailableError
        from either sandboxed run propagates.
        """
        artifact = Path(artifact)
        if not artifact.is_absolute():
            raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
        if isinstance(inputs, str):
            raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
        workdir = _checked_workdir(artifact)
        harness = None if self.harness is None else _resolved(self.harness)
        mounts = _Mounts(self._hidden_roots(), harness, self.install())
        hashes = _pre_run_hashes(workdir)
        before = _regular_files(workdir)
        result = self._sandboxed(mounts, workdir, [str(artifact), *inputs], limits, watcher=False)
        after = _regular_files(workdir)
        outputs = {path: after[path] for path in sorted(after) if path not in before and not _is_reserved(path)}
        findings, sim_ub, sim_gap = read_findings(result.stdout)
        diagnostics = [*findings, *read_jit_diagnostics(result.stdout, result.stderr, workdir)]
        if result.hang:
            note = self._watcher_rerun(mounts, workdir, artifact.name, list(inputs), limits, hashes)
            diagnostics.extend([] if note is None else [note])
        return RunResult(
            exit_code=result.returncode,
            hang=result.hang,
            stdout=result.stdout,
            stderr=result.stderr,
            output_files=outputs,
            wall_s=result.wall_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=result.workdir_incomplete,
            sim_ub=sim_ub,
            sim_gap=sim_gap,
            diagnostics=diagnostics,
        )

    def _hidden_roots(self) -> tuple[Path, ...]:
        """Return the configured or environment hidden roots and the runs root, resolved.

        SandboxUnavailableError when neither the configuration nor the
        environment gives a hidden root.
        """
        roots = _environment_roots() if self.hidden_roots is None else self.hidden_roots
        if not roots:
            raise SandboxUnavailableError(
                "no hidden root: set $LASSI_SCRATCH or $HOME, or pass hidden_roots; nothing ran"
            )
        return (*(_resolved(root) for root in roots), runs_root().resolve())

    def _sandboxed(
        self, mounts: _Mounts, workdir: Path, argv: list[str], limits: Limits, *, watcher: bool
    ) -> SandboxResult:
        """Create the workdir's @ttsim cache and logs, then run `argv` there in the sandbox; Watcher on when asked."""
        for name in (CACHE_DIR, LOGS_DIR):
            (workdir / RESERVED_DIR / name).mkdir(parents=True, exist_ok=True)
        environment = program_environment(mounts.install, workdir, limits)
        if watcher:
            environment[WATCHER_VARIABLE] = "1"
        spec = SandboxSpec(
            workdir=workdir,
            hidden_roots=mounts.hidden_roots,
            harness=mounts.harness,
            toolchains=mounts.install.root,
            environment=environment,
        )
        return self.sandbox.run(spec, argv, limits)

    def _watcher_rerun(
        self,
        mounts: _Mounts,
        workdir: Path,
        name: str,
        inputs: list[str],
        limits: Limits,
        hashes: Mapping[str, str],
    ) -> Diagnostic | None:
        """Rerun a hung program once with Watcher in a fresh copy of its pre-run files; return the Watcher note.

        `name` is the artifact's file name in the workdir. There is no rerun,
        and None is returned, when the artifact is not a pre-run regular file
        or any pre-run file changed or vanished (_fresh_copy). The rerun's
        result is never read; only its Watcher log is (watcher_note, from its
        last WATCHER_READ_BYTES), and no log gives None.
        """
        if name not in hashes:
            return None
        copy = _fresh_copy(workdir, hashes)
        if copy is None:
            return None
        self._sandboxed(mounts, copy, [str(copy / name), *inputs], limits, watcher=True)
        log = _read_tail(copy.joinpath(RESERVED_DIR, LOGS_DIR, *WATCHER_LOG), WATCHER_READ_BYTES)
        return None if log is None else watcher_note(log)
