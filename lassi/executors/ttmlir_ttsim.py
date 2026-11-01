"""The tt-mlir flatbuffer executor on ttsim, registered as Executor "ttmlir-ttsim" (task P17.48).

It runs one tt-mlir flatbuffer, TT-Metal ("TTM0") or, since the owner's
answer to OQ-112 (route A), TT-NN ("TTNN"), through the pinned tt-mlir
runtime on ttsim, with the held-out inputs as tensors and the outputs as
lassi_io files. ttrt cannot load given inputs (plans/spikes/ttir-d2m.md,
section 4), so the run goes through lassi-ttmlir-run, LASSI's own driver
(toolchains/tt-mlir-driver/lassi_ttmlir_run.cpp, built by
toolchains/tt-mlir-driver.sh; plans/spikes/ttmlir-runtime.md, "The LASSI
driver"). It is the ttsim executor (lassi.executors.ttsim) with four
changes, and everything else is that executor's: the sandbox (Agent Rule 6),
the workdir and its mounts, the reserved @ttsim directory, the ttsim row's
environment otherwise, the findings, the kernel JIT diagnostics, the Watcher
rerun of a hang, and the config keys and their one supported values.

- HOME. TT-NN's static configuration in _ttnncpp.so, which the runtime
  library always links, reads getenv("HOME") with no check while the
  libraries load, so a process without HOME dies by SIGSEGV before main
  (plans/spikes/ttmlir-runtime.md, 2026-10-30). The environment the executor
  passes to the sandbox is the ttsim row's plus HOME=<workdir>/@ttsim/home
  (HOME_DIR, made before the run like the cache and logs), a reserved
  directory inside the sandbox, so nothing written there is an output file
  and no host HOME reaches the program (the owner's answer to OQ-115; Agent
  Rule 12). The driver sets no environment of its own.

- The install. Under the toolchains root: the pinned ttsim library and SoC
  descriptor, checked as the ttsim executor checks them; the tt-mlir runtime
  of toolchains/tt-mlir-runtime.pin, whose lassi-install.txt must start with
  "<NAME> <COMMIT> <URL> <VERSION> <BUILD>" (toolchains/tt-mlir-runtime.sh
  writes it so), with its runtime library and its own tt-metal tree
  (METAL_DIR), which is the program's TT_METAL_RUNTIME_ROOT, since the
  runtime's libraries and kernels were built from it; and the driver of
  toolchains/tt-mlir-driver.pin, whose lassi-install.txt must start with
  "<NAME> <VERSION> <SOURCE_SHA256> <HEADER_SHA256> <runtime PREFIX_NAME>",
  with its program BINARY and the recorded ttsim system descriptor
  SYSTEM_DESC, whose sha256 must be the pin's SYSTEM_DESC_SHA256. A
  PLACEHOLDER in the driver pin refuses every run: no flatbuffer runs before
  the descriptor it must be compiled for is recorded. Each refusal is a
  SandboxUnavailableError naming the path, before anything runs or is
  created in the workdir.
- The argv: [driver, "run", flatbuffer, descriptor, *inputs], where the
  inputs are the item's input files then its output files (lassi.bench
  program_args). The driver refuses a flatbuffer whose embedded system
  descriptor is not the recorded one (a flatbuffer for the mock descriptor
  places L1 buffers the runtime never checks; plans/spikes/ttir-d2m.md,
  section 1), a dtype or element count that differs from the program's, and
  a file count that is not the program's inputs plus outputs, all before it
  opens the one device it opens.
- The diagnostics. After the ttsim executor's, each driver line on stderr
  "lassi-ttmlir-run: refused: <reason>" becomes Diagnostic(stage "run",
  severity "error", code DRIVER_REFUSED) and each "lassi-ttmlir-run: error:
  <what>" (the runtime threw) one with code DRIVER_ERROR; a run that ended by
  a signal (exit status 128 + N, N from 1 to 64) and did not hang gets one
  with code SIGNAL_CODE naming the signal: a simulator abort, a crash, or a
  limit's kill is a failed run, and the exit status says which.

write_system_desc runs the driver's describe mode once in the sandbox, in a
directory under the runs root, with the same environment, and returns the
run and the path of the descriptor it wrote (tools/ttmlir_system_desc.py
installs it in the driver prefix). Nothing chooses a device but
TT_METAL_SIMULATOR: the sandbox shows no /dev/tenstorrent node, and each
process opens one device and closes it (Agent Rule 9).

The executor declares `pins` ttsim (toolchains/ttsim.pin) and tt-mlir-runtime
(the runtime pin, toolchains/tt-mlir-runtime.pin, its own pin name), which
the runner records (Agent Rule 10) under the record's tt_mlir field, since it
is a build variant of tt-mlir's source (lassi.core.runner pin_field; its
recorded version is 0.9.0.dev20260221-runtime). The driver is this
repository's source at the run's commit, and its install record ties the
binary to that source; it is named in device(). A trial records one tt_mlir
version, so the runner refuses this executor beside a toolchain on the
compiler pin (the compile-only toolchains of lassi.toolchains.ttir_compile),
whose flatbuffers target the mock descriptor the driver refuses anyway; a
recipe binds it with a toolchain built from the runtime pin's tools for the
recorded descriptor (ttir-ttmetal-ttsim, d2m-ttmetal-ttsim), which declares
the same pin.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lassi.core.capabilities import SIMULATOR
from lassi.core.interfaces import Limits, RunResult
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.executors.native import _regular_files
from lassi.executors.sandbox import Sandbox, SandboxUnavailableError
from lassi.executors.ttsim import (
    ARCH,
    CHIPS,
    DISPATCH,
    RESERVED_DIR,
    RESERVED_SUBDIRS,
    TTSIM_PIN,
    TtsimExecutor,
    TtsimInstall,
    _checked_file,
    _Mounts,
    _pin,
    _resolved,
    _sha256,
    printable,
)
from lassi.executors.workdir import runs_root
from lassi.toolchains import pins as pin_files

# The pin files the executor reads beyond ttsim's; it declares the runtime pin under its own name.
DRIVER_PIN = "tt-mlir-driver"
RUNTIME_PIN = "tt-mlir-runtime"
_DRIVER_KEYS = (
    "NAME",
    "VERSION",
    "PREFIX_NAME",
    "SOURCE_SHA256",
    "HEADER_SHA256",
    "BINARY",
    "SYSTEM_DESC",
    "SYSTEM_DESC_SHA256",
)
_RUNTIME_KEYS = (
    "NAME",
    "VERSION",
    "BUILD",
    "COMMIT",
    "URL",
    "PREFIX_NAME",
    "RUNTIME_LIBRARY",
    "METAL_DIR",
    "METAL_COMMIT",
)
# HOME's directory under the workdir's RESERVED_DIR (the module docstring, HOME), and the variable.
HOME_DIR = "home"
HOME_VARIABLE = "HOME"
# The driver's two modes (lassi_ttmlir_run.cpp).
RUN_MODE = "run"
DESCRIBE_MODE = "describe"
# A value not measured yet (AGENTS.md, Results).
PLACEHOLDER = "PLACEHOLDER"
# Each install's record file.
INSTALL_RECORD = "lassi-install.txt"
# The driver's own stderr lines and the codes of their diagnostics.
DRIVER_PREFIX = "lassi-ttmlir-run: "
DRIVER_REFUSED = "driver-refused"
DRIVER_ERROR = "runtime-error"
_DRIVER_CODES = {
    "refused: ": (DRIVER_REFUSED, "the driver refused the run: "),
    "error: ": (DRIVER_ERROR, "the tt-mlir runtime failed: "),
}
# A death by signal N, in the shell's form the sandbox gives (lassi.executors.sandbox SandboxResult.returncode).
SIGNAL_CODE = "signal"
SIGNAL_BASE = 128
MAX_SIGNAL = 64
# Linux's signal numbers, by name, for the messages; the host running the tests may number them otherwise.
_SIGNAL_NAMES = {
    1: "SIGHUP",
    2: "SIGINT",
    4: "SIGILL",
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    15: "SIGTERM",
    24: "SIGXCPU",
    25: "SIGXFSZ",
}


@dataclass(frozen=True)
class TtmlirInstall(TtsimInstall):
    """The checked install of a run: ttsim's (with the runtime's tt-metal tree as `tree`), the driver, and the
    recorded ttsim system descriptor."""

    driver: Path
    system_desc: Path


def _first_line(path: Path) -> str:
    """Return the first line of a file, or "" when it cannot be read."""
    try:
        lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return lines[0] if lines else ""


def _unavailable(message: str) -> SandboxUnavailableError:
    """Return the refusal of an install check; nothing ran."""
    return SandboxUnavailableError(f"{message}; nothing ran")


def check_runtime(root: Path, pin: Mapping[str, str]) -> Path:
    """Return the runtime's tt-metal tree after checking the tt-mlir runtime install under `root` against `pin`.

    The record's first line must be "<NAME> <COMMIT> <URL> <VERSION>
    <BUILD>", and the runtime library and the tree must exist; otherwise
    SandboxUnavailableError names the path.
    """
    prefix = root / pin["PREFIX_NAME"]
    head = " ".join(pin[key] for key in ("NAME", "COMMIT", "URL", "VERSION", "BUILD"))
    record = prefix / INSTALL_RECORD
    if _first_line(record) != head:
        raise _unavailable(
            f"{prefix} is not the pinned tt-mlir runtime: {record} does not start with {head!r}; install it with "
            f"toolchains/{RUNTIME_PIN}.sh"
        )
    library = prefix / pin["RUNTIME_LIBRARY"]
    if not library.is_file():
        raise _unavailable(f"the tt-mlir runtime library {library} is missing")
    tree = prefix / pin["METAL_DIR"]
    if not tree.is_dir():
        raise _unavailable(f"the tt-mlir runtime's tt-metal tree {tree} is missing")
    return tree


def check_driver(
    root: Path, pin: Mapping[str, str], runtime_pin: Mapping[str, str], *, need_desc: bool
) -> tuple[Path, Path]:
    """Return the driver program and the recorded descriptor's path after checking the driver install under `root`.

    The pin must hold no PLACEHOLDER in the record's values, the record's
    first line must name them and the runtime's PREFIX_NAME, and the program
    must be a regular file. With `need_desc` the descriptor must be a
    regular file whose sha256 is SYSTEM_DESC_SHA256, which must not be a
    PLACEHOLDER. Otherwise SandboxUnavailableError names the path or key.
    """
    values = [pin[key] for key in ("NAME", "VERSION", "SOURCE_SHA256", "HEADER_SHA256")]
    if any(PLACEHOLDER in value for value in [*values, pin["PREFIX_NAME"]]):
        raise _unavailable(f"toolchains/{DRIVER_PIN}.pin still holds a PLACEHOLDER; fill it from the driver's source")
    prefix = root / pin["PREFIX_NAME"]
    head = " ".join([*values, runtime_pin["PREFIX_NAME"]])
    record = prefix / INSTALL_RECORD
    if _first_line(record) != head:
        raise _unavailable(
            f"{prefix} is not the pinned driver: {record} does not start with {head!r}; build it with "
            f"toolchains/{DRIVER_PIN}.sh"
        )
    driver = prefix / pin["BINARY"]
    if driver.is_symlink() or not driver.is_file():
        raise _unavailable(f"the driver program {driver} is missing")
    desc = prefix / pin["SYSTEM_DESC"]
    if need_desc:
        expected = pin["SYSTEM_DESC_SHA256"]
        if expected == PLACEHOLDER:
            raise _unavailable(
                f"no ttsim system descriptor is recorded (toolchains/{DRIVER_PIN}.pin SYSTEM_DESC_SHA256 is a "
                f"PLACEHOLDER): write {desc} with tools/ttmlir_system_desc.py and record its sha256"
            )
        if desc.is_symlink() or not desc.is_file():
            raise _unavailable(f"the recorded ttsim system descriptor {desc} is missing")
        found = _sha256(desc)
        if found != expected:
            raise _unavailable(f"{desc} is not the recorded descriptor: its sha256 is {found}, not {expected}")
    return driver, desc


def read_driver_diagnostics(stderr: str) -> list[Diagnostic]:
    """Return a run-stage error for each driver line in `stderr`, in order (see the module docstring)."""
    found: list[Diagnostic] = []
    for line in stderr.split("\n"):
        text = line.rstrip("\r")
        if not text.startswith(DRIVER_PREFIX):
            continue
        rest = text[len(DRIVER_PREFIX) :]
        for lead, (code, words) in _DRIVER_CODES.items():
            if rest.startswith(lead):
                message = printable(words + rest[len(lead) :])
                found.append(Diagnostic(stage="run", severity="error", code=code, message=message))
    return found


def signal_diagnostic(exit_code: int | None, hang: bool) -> Diagnostic | None:
    """Return the run-stage error of a run that ended by a signal and did not hang; None for any other run."""
    if hang or exit_code is None or not SIGNAL_BASE < exit_code <= SIGNAL_BASE + MAX_SIGNAL:
        return None
    number = exit_code - SIGNAL_BASE
    name = _SIGNAL_NAMES.get(number, f"signal {number}")
    message = (
        f"the run ended by signal {number} ({name}, exit status {exit_code}): a simulator or runtime abort, a "
        "crash, or a kill by a memory or CPU-time limit; the outputs were not written"
    )
    return Diagnostic(stage="run", severity="error", code=SIGNAL_CODE, message=message)


@register("Executor", "ttmlir-ttsim")
class TtmlirTtsimExecutor(TtsimExecutor):
    """Runs a tt-mlir flatbuffer through the pinned runtime and LASSI's driver on ttsim, only inside the sandbox.

    The config keys and keyword settings are the ttsim executor's. `pins`
    holds ttsim's pairs and the runtime pin's (under tt-mlir-runtime), and
    `driver_pin` the driver pin's, read at construction.
    """

    name = "ttmlir-ttsim"
    capabilities = frozenset({"runs_code", "sandboxed", SIMULATOR})
    config_keys = frozenset({"arch", "dispatch", "chips", "harness"})
    reserved_dirs: tuple[str, ...] = (*RESERVED_SUBDIRS, HOME_DIR)

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
        """Check the configuration as the ttsim executor does, then read the driver and runtime pins.

        ValueError names a refused key, or a pin file without a key the
        executor needs. Nothing on disk is checked until run().
        """
        super().__init__(
            arch=arch,
            dispatch=dispatch,
            chips=chips,
            harness=harness,
            hidden_roots=hidden_roots,
            toolchains=toolchains,
            sandbox=sandbox,
        )
        self.driver_pin: dict[str, str] = _pin(DRIVER_PIN, _DRIVER_KEYS)
        self.runtime_pin: dict[str, str] = _pin(RUNTIME_PIN, _RUNTIME_KEYS)
        self.pins = {TTSIM_PIN: self.pins[TTSIM_PIN], RUNTIME_PIN: self.runtime_pin}

    def device(self) -> str:
        """Return the device, one line of printable ASCII from the pins alone; no process, no sandbox.

        It names ttsim and its pin, the runtime's version and its tt-mlir and
        tt-metal commits, and the driver's version, as a simulator (Agent
        Rule 2).
        """
        ttsim, runtime, driver = self.pins[TTSIM_PIN], self.runtime_pin, self.driver_pin
        text = (
            f"ttsim {ttsim['VERSION']} ({ttsim['LIBRARY']}, a virtual Wormhole) on the host CPU, with the tt-mlir "
            f"runtime {pin_files.recorded_version(runtime)} (tt-mlir {runtime['COMMIT'][:8]}, tt-metal "
            f"{runtime['METAL_COMMIT'][:8]}) and lassi-ttmlir-run {driver['VERSION']}: a simulator, not silicon"
        )
        return printable(" ".join(text.split()))

    def install(self) -> TtmlirInstall:
        """Return the checked install (see the module docstring), the recorded descriptor included."""
        return self._install(need_desc=True)

    def recorded_system_desc(self) -> Path:
        """Return where the recorded ttsim descriptor lies: <toolchains root>/<driver PREFIX_NAME>/<SYSTEM_DESC>.

        Nothing is checked or created; install() checks it.
        """
        return self._toolchains_root() / self.driver_pin["PREFIX_NAME"] / self.driver_pin["SYSTEM_DESC"]

    def _install(self, *, need_desc: bool) -> TtmlirInstall:
        """Check ttsim's files, the runtime, and the driver under the toolchains root; nothing is created."""
        root = self._toolchains_root()
        ttsim = self.pins[TTSIM_PIN]
        prefix = root / ttsim["PREFIX_NAME"]
        library = _checked_file(prefix / ttsim["LIBRARY"], ttsim, "SHA256")
        descriptor = _checked_file(prefix / ttsim["SOC_DESCRIPTOR"], ttsim, "SOC_DESCRIPTOR_SHA256")
        tree = check_runtime(root, self.runtime_pin)
        driver, system_desc = check_driver(root, self.driver_pin, self.runtime_pin, need_desc=need_desc)
        return TtmlirInstall(
            root=root, library=library, descriptor=descriptor, tree=tree, driver=driver, system_desc=system_desc
        )

    def _environment(self, install: TtsimInstall, workdir: Path, limits: Limits) -> dict[str, str]:
        """Return the ttsim row's environment plus HOME=<workdir>/@ttsim/home, reserved (module docstring)."""
        environment = super()._environment(install, workdir, limits)
        environment[HOME_VARIABLE] = str(workdir / RESERVED_DIR / HOME_DIR)
        return environment

    def _argv(self, mounts: _Mounts, artifact: Path, inputs: Sequence[str]) -> list[str]:
        """Return [driver, "run", flatbuffer, recorded descriptor, *inputs]."""
        install = mounts.install
        if not isinstance(install, TtmlirInstall):
            raise TypeError(f"the ttmlir-ttsim executor runs with a TtmlirInstall, got {type(install).__name__}")
        return [str(install.driver), RUN_MODE, str(artifact), str(install.system_desc), *inputs]

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run the flatbuffer `artifact` as the ttsim executor runs a program, through the driver; add its diagnostics.

        The RunResult is the ttsim executor's, with the driver's refusal and
        error lines (read_driver_diagnostics) and a signal's error
        (signal_diagnostic) after its diagnostics.
        """
        result = super().run(artifact, inputs, limits)
        extra = read_driver_diagnostics(result.stderr)
        signal = signal_diagnostic(result.exit_code, result.hang)
        extra.extend([] if signal is None else [signal])
        if not extra:
            return result
        return dataclasses.replace(result, diagnostics=[*result.diagnostics, *extra])

    def write_system_desc(self, workdir: Path, limits: Limits) -> tuple[RunResult, Path]:
        """Run the driver's describe mode once in the sandbox in `workdir`; return the run and the descriptor's path.

        `workdir` must be an existing directory inside the runs root
        (ValueError otherwise). The install is checked without a recorded
        descriptor. The descriptor is <workdir>/<SYSTEM_DESC>; the caller
        reads the run's exit status before using it.
        """
        workdir = _resolved(Path(workdir))
        if not workdir.is_dir() or runs_root().resolve() not in workdir.parents:
            raise ValueError(f"the workdir {str(workdir)!r} must be a directory inside the runs root")
        harness = None if self.harness is None else _resolved(self.harness)
        install = self._install(need_desc=False)
        mounts = _Mounts(self._hidden_roots(), harness, install)
        name = self.driver_pin["SYSTEM_DESC"]
        before = _regular_files(workdir)
        result = self._sandboxed(mounts, workdir, [str(install.driver), DESCRIBE_MODE, name], limits, watcher=False)
        after = _regular_files(workdir)
        outputs = {path: after[path] for path in sorted(after) if path not in before and path == name}
        signal = signal_diagnostic(result.returncode, result.hang)
        diagnostics = [*read_driver_diagnostics(result.stderr), *([] if signal is None else [signal])]
        run = RunResult(
            exit_code=result.returncode,
            hang=result.hang,
            stdout=result.stdout,
            stderr=result.stderr,
            output_files=outputs,
            wall_s=result.wall_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=result.workdir_incomplete,
            diagnostics=diagnostics,
        )
        return run, workdir / name
