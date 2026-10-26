"""The ttkernel toolchain, registered as Toolchain "ttkernel-ttmetal" (bible IR Strategy, the ttkernel row).

The language ttkernel holds an item's Tenstorrent kernels as ttkernel MLIR
modules, one file per kernel, which the model writes; the item's host
program is a trusted harness file, never model output (bible Level Rules).
TtKernelBuild lowers each module to TT-Metalium kernel C++ with the pinned
tt-mlir tools and then builds the host program as ttmetal-host does
(lassi.toolchains.ttmetal_build), so the kernel JIT finds each emitted
kernel at the path the host program launches.

Per kernel module `<k>.mlir` (a model file ending in .mlir with a directory
named kernels in its path), in sorted order, three commands run in the
build directory. Steps 2 and 3 are the two commands tt-mlir's own lit
tests run at the pin, 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c
(test/ttmlir/Translate/TTKernel/ttkernel_noc.mlir:1-2 and
ttkernel_compute.mlir:1-2); step 1 runs ttmlir-opt with no pass, so a
parse or verifier error is told apart from a lowering error:

1. verify (stage "verify"): `ttmlir-opt <k>.mlir -o
   _ttkernel_emitc/<k>.verified.mlir`; MlirOptMain parses and verifies the
   module and prints it back (tools/ttmlir-opt/ttmlir-opt.cpp:13-22).
2. lower (stage "lower"): `ttmlir-opt --convert-ttkernel-to-emitc
   --mlir-print-debuginfo -o _ttkernel_emitc/<k>.emitc.mlir <k>.mlir`. The
   pass is ConvertTTKernelToEmitC (include/ttmlir/Conversion/Passes.td:261);
   it converts each func.func that carries ttkernel.thread, its ttkernel,
   arith, scf, and memref ops included, to EmitC
   (lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp:940-974). The one
   word the lit tests lack, --mlir-print-debuginfo (an MLIR printer
   option), keeps each op's location in the model's file in the output, so
   an error of step 3 names a line of <k>.mlir, not of the intermediate.
3. translate (stage "lower"): `ttmlir-translate --ttkernel-to-cpp -o
   <k>.cpp _ttkernel_emitc/<k>.emitc.mlir`. The translation
   (lib/Target/TTKernel/TTKernelToCppRegistration.cpp:22-33) writes, for
   each func.func with a ttkernel.thread attribute, the thread's includes
   and `void kernel_main()` (lib/Target/TTKernel/TTKernelToCpp.cpp:38-173,
   :379-381, :418-432), and nothing for a module without one, so the
   emitted text must hold exactly one kernel_main (KERNEL_ENTRY), or the
   module gets a kernel-count error.

The intermediate files lie under WORK_DIR, a directory no model or harness
file may use; its name starts with neither "-" nor "@", which a tool's
command line would read as an option or a response file. A model file that
is not a kernel module (any other suffix, a .mlir file outside a kernels
directory, or the host program itself) is refused with a bad-path or
not-a-kernel error before anything is written or run. Each kernel is
lowered even after another failed, so one correction sees every kernel's
errors; the host program is built only when every kernel lowered. Its
command is ttmetal-host's (TtMetalHost.command) over the harness's host
sources: the C++ sources (.cpp, .cc, .cxx) of the harness files outside any
kernels directory, in sorted order.

Diagnostics. Each tool's stderr is read with parse_mlir_diagnostics, at the
step's stage: `<file>:<line>:<column>: <severity>: <message>`, the form
MLIR's source manager handler prints, with a leading "./" dropped from the
file; `<file>:<line>: <severity>: <message>`, where `<unknown>:0` (an op
with no location) gives no place; `<severity>: <message>` with no place;
and `LLVM ERROR: <message>` as an error. Severities are error, warning, and
note; remark reads as note. A line followed by a caret line is a source
echo and never a diagnostic. A step that fails with no parsed error, or
times out, gets one exit-status error naming the tool, the kernel, and the
status. The host
build's stderr is clang's, read as ttmetal-host reads it (stage compile).
The raw stderr of every command, each after a line naming the command and
its status, is the attachment STDERR_ATTACHMENT.

Pins. The class declares PIN "tt-metal" with no PIN_BIN and check_tree, as
ttmetal-host does: the host compiler is that pin's EXECUTABLE, checked by
--version, and the tree its install. TOOL_PINS names the tt-mlir pin and
its two tool keys, TTMLIR_OPT and TTMLIR_TRANSLATE, each a path under
<resolved toolchains root>/<tt-mlir PREFIX_NAME>; the runner checks that
install with check_tool_pin, runs each tool's --version through the compile
sandbox against that pin's EXPECT_VERSION, records the pin, and passes the
paths as `tools` (lassi.core.runner, _tool_pins). The toolchain declares
`diagnostics` and `emits_warnings`, and no host_compute_guard: the host
program is the item's, so there is no model host code to read.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from lassi.core.files import _check_path
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import OUTPUT, STDERR_ATTACHMENT, CommandResult, CommandRunner
from lassi.toolchains._stderr import compile_diagnostic
from lassi.toolchains.ttmetal_build import KERNEL_DIR, TtMetalHost

# The tt-mlir pin and the keys of its two tool paths (toolchains/tt-mlir.pin), and those tools' default names.
MLIR_PIN = "tt-mlir"
OPT_KEY = "TTMLIR_OPT"
TRANSLATE_KEY = "TTMLIR_TRANSLATE"
DEFAULT_TOOLS = {OPT_KEY: "ttmlir-opt", TRANSLATE_KEY: "ttmlir-translate"}
# The pass and the translation the lit tests at the pin run, and the printer option this toolchain adds.
LOWER_PASS = "--convert-ttkernel-to-emitc"
DEBUG_INFO = "--mlir-print-debuginfo"
TRANSLATION = "--ttkernel-to-cpp"
# The suffix of a kernel module, the suffix of its emitted kernel, and the intermediate files' directory.
MODULE_SUFFIX = ".mlir"
KERNEL_SUFFIX = ".cpp"
WORK_DIR = "_ttkernel_emitc"
VERIFIED_SUFFIX = ".verified.mlir"
EMITC_SUFFIX = ".emitc.mlir"
# What the translation writes for each kernel function (TTKernelToCpp.cpp:379-381, printed by the C++ emitter).
KERNEL_ENTRY = re.compile(r"\bvoid\s+kernel_main\s*\(")
# The install record toolchains/tt-mlir.sh writes once every check passed; its first line is "<NAME> <COMMIT> ...".
INSTALL_RECORD = "lassi-install.txt"
# A harness file the host build compiles: a C++ source outside any kernels directory.
HOST_SUFFIXES = (".cpp", ".cc", ".cxx")
# The diagnostic stages of the three steps and of the host build.
VERIFY_STAGE = "verify"
LOWER_STAGE = "lower"

_SEVERITY = r"(?P<severity>error|warning|note|remark)"
_PLACE = re.compile(r"(?P<file>[^\s:][^:]*?):(?P<line>[0-9]{1,10}):(?P<column>[0-9]{1,10}): " + _SEVERITY
                    + r": (?P<message>.*)")
_LINE_ONLY = re.compile(r"(?P<file>[^\s:][^:]*?):(?P<line>[0-9]{1,10}): " + _SEVERITY + r": (?P<message>.*)")
_BARE = re.compile(_SEVERITY + r": (?P<message>.*)")
_FATAL = re.compile(r"LLVM ERROR: (?P<message>.*)")
_CARET = re.compile(r"[ \t]*\^[ \t~]*")
_UNKNOWN = "<unknown>"


def verify_command(opt: str, kernel: str, out: str) -> list[str]:
    """Return the verify step's command: ttmlir-opt parses and verifies `kernel` and prints it to `out`."""
    return [opt, kernel, "-o", out]


def lower_command(opt: str, kernel: str, out: str) -> list[str]:
    """Return the lowering step's command: `kernel` through convert-ttkernel-to-emitc, with locations, to `out`."""
    return [opt, LOWER_PASS, DEBUG_INFO, "-o", out, kernel]


def translate_command(translate: str, emitc: str, out: str) -> list[str]:
    """Return the translation step's command: the EmitC module `emitc` to kernel C++ at `out`."""
    return [translate, TRANSLATION, "-o", out, emitc]


def kernel_cpp_path(module: str) -> str:
    """Return the kernel path a module's C++ is written at: the module's path with .cpp in place of .mlir."""
    return str(PurePosixPath(module).with_suffix(KERNEL_SUFFIX))


def is_kernel_module(path: str) -> bool:
    """Return True for a path ending in .mlir with a directory named kernels in it."""
    pure = PurePosixPath(path)
    return pure.suffix == MODULE_SUFFIX and KERNEL_DIR in pure.parts[:-1]


def host_sources(harness: Mapping[str, str]) -> list[str]:
    """Return the harness files the host build compiles, sorted: C++ sources outside any kernels directory."""
    return sorted(
        path for path in harness
        if PurePosixPath(path).suffix in HOST_SUFFIXES and KERNEL_DIR not in PurePosixPath(path).parts[:-1]
    )


def parse_mlir_diagnostics(stderr: str, stage: str) -> list[Diagnostic]:
    """Return the Diagnostics in an MLIR tool's `stderr` at `stage`, in order (see the module docstring)."""
    lines = stderr.split("\n")
    found: list[Diagnostic] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        index += 1
        diagnostic = _mlir_line(line, stage)
        if diagnostic is None:
            continue
        if index + 1 < len(lines) and _CARET.fullmatch(lines[index + 1].rstrip()):
            index += 2  # the source echo and its caret line
        found.append(diagnostic)
    return found


def _mlir_line(line: str, stage: str) -> Diagnostic | None:
    """Return the Diagnostic one stderr line holds, or None when it holds none."""
    fatal = _FATAL.fullmatch(line)
    if fatal:
        return Diagnostic(stage=stage, severity="error", message=fatal.group("message"))
    for pattern in (_PLACE, _LINE_ONLY, _BARE):
        match = pattern.fullmatch(line)
        if match is None:
            continue
        groups = match.groupdict()
        severity = "note" if groups["severity"] == "remark" else groups["severity"]
        file = groups.get("file")
        if file is None or file == _UNKNOWN:
            return Diagnostic(stage=stage, severity=severity, message=groups["message"])
        file = file[2:] if file.startswith("./") else file
        column = int(groups["column"]) if groups.get("column") else None
        return Diagnostic(
            stage=stage, severity=severity, file=file, line=int(groups["line"]), column=column,
            message=groups["message"],
        )
    return None


def check_tool_install(home: Path, pin: Mapping[str, str]) -> None:
    """Raise ValueError unless `home` holds lassi-install.txt whose first line names the pin's NAME and COMMIT."""
    name, commit = pin.get("NAME", ""), pin.get("COMMIT", "")
    if not name or not commit:
        raise ValueError(f"toolchains/{MLIR_PIN}.pin has no NAME or no COMMIT, so no install can be checked")
    record = Path(home) / INSTALL_RECORD
    try:
        lines = record.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError as error:
        raise ValueError(
            f"{record} cannot be read ({error.strerror or error}), so {home} is not an install of the pinned "
            f"{name} (an unfinished build leaves lassi-install.unfinished); install it with toolchains/{MLIR_PIN}.sh"
        ) from None
    words = lines[0].split()[:2] if lines else []
    if words != [name, commit]:
        raise ValueError(
            f"the first line of {record} names {' '.join(words) or 'nothing'!r}, not {name} {commit}; the tree is "
            f"not the pinned install, so install the pin with toolchains/{MLIR_PIN}.sh"
        )


def _has_error(diagnostics: Sequence[Diagnostic]) -> bool:
    """Return True when any diagnostic is an error."""
    return any(item.severity == "error" for item in diagnostics)


def _section(argv: Sequence[str], result: CommandResult) -> str:
    """Return one command's part of the stderr attachment: a line naming it and its status, then its stderr."""
    text = result.stderr if result.stderr.endswith("\n") or not result.stderr else result.stderr + "\n"
    return f"$ {' '.join(argv)}  (exit {result.returncode})\n{text}"


@register("Toolchain", "ttkernel-ttmetal")
class TtKernelBuild(TtMetalHost):
    """Lowers ttkernel kernel modules to kernel C++ with the pinned tt-mlir and builds the item's host program."""

    name = "ttkernel-ttmetal"
    capabilities = frozenset({"diagnostics", "emits_warnings"})
    SOURCE_SUFFIXES = (MODULE_SUFFIX,)
    TOOL_PINS = {MLIR_PIN: (OPT_KEY, TRANSLATE_KEY)}

    def __init__(
        self,
        *,
        tree: Path,
        tools: Mapping[str, str] | None = None,
        executable: str = "clang++-20",
        runner: CommandRunner | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        """Keep the tt-metal tree and host compiler (TtMetalHost) and the tt-mlir tools' paths by pin key.

        `tools` maps TTMLIR_OPT and TTMLIR_TRANSLATE to the pinned tools,
        as the runner passes them; None (tests and local checks) names
        ttmlir-opt and ttmlir-translate on PATH.
        """
        super().__init__(tree=tree, executable=executable, runner=runner, timeout_s=timeout_s)
        given = dict(DEFAULT_TOOLS if tools is None else tools)
        self.opt = str(given[OPT_KEY])
        self.translate = str(given[TRANSLATE_KEY])

    @classmethod
    def check_tool_pin(cls, pin_name: str, home: Path, pin: Mapping[str, str]) -> None:
        """Raise ValueError unless `home` is the install of the TOOL_PINS pin `pin_name` (check_tool_install)."""
        if pin_name != MLIR_PIN:
            raise ValueError(f"{cls.name} names no tool pin {pin_name!r}; its tool pin is {MLIR_PIN}")
        check_tool_install(home, pin)

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write the files, lower each kernel module, and build the harness host program (module docstring).

        Every path is checked and every refusal made before anything is
        written; a build with no kernel module, or with a refused file,
        runs nothing. The artifact is workdir/OUTPUT when every kernel
        lowered and the host build wrote it.
        """
        workdir = Path(workdir)
        harness = dict(harness or {})
        for path in [*files, *harness]:
            _check_path(path)
        payloads = {path: text.encode("utf-8") for path, text in [*files.items(), *harness.items()]}
        refused = self._refused(files, harness)
        if refused:
            return BuildResult(artifact=None, diagnostics=refused, stderr_ref="")
        for path, data in payloads.items():
            target = workdir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if not files:
            return BuildResult(artifact=None, diagnostics=[self._no_sources()], stderr_ref="")
        hosts = host_sources(harness)
        if not hosts:
            message = f"the item gives {self.name} no host program to build (a harness C++ source outside kernels/)"
            return BuildResult(artifact=None, diagnostics=[compile_diagnostic("error", message, code="no-host")])
        output = workdir / OUTPUT
        if output.is_file() or output.is_symlink():
            output.unlink()
        log: list[str] = []
        diagnostics: list[Diagnostic] = []
        for kernel in sorted(files):
            diagnostics.extend(self._lowered(kernel, workdir, log))
        if _has_error(diagnostics):
            (workdir / STDERR_ATTACHMENT).write_bytes("".join(log).encode("utf-8"))
            return BuildResult(artifact=None, diagnostics=diagnostics, stderr_ref=STDERR_ATTACHMENT)
        command = self.command(hosts)
        result = self.runner(command, workdir, self.timeout_s)
        log.append(_section(command, result))
        (workdir / STDERR_ATTACHMENT).write_bytes("".join(log).encode("utf-8"))
        host = self.parse(result.stderr, {path: harness[path] for path in hosts})
        artifact = output if result.returncode == 0 and output.is_file() else None
        failure = self._failure(result.returncode, artifact, host)
        diagnostics.extend([*host, *([failure] if failure is not None else [])])
        return BuildResult(artifact=artifact, diagnostics=diagnostics, stderr_ref=STDERR_ATTACHMENT)

    def _lowered(self, kernel: str, workdir: Path, log: list[str]) -> list[Diagnostic]:
        """Verify, lower, and translate one kernel module; return its diagnostics (an error stops its steps)."""
        stem = str(PurePosixPath(WORK_DIR) / PurePosixPath(kernel).with_suffix(""))
        emitc = stem + EMITC_SUFFIX
        cpp = kernel_cpp_path(kernel)
        (workdir / emitc).parent.mkdir(parents=True, exist_ok=True)
        steps = (
            (VERIFY_STAGE, verify_command(self.opt, kernel, stem + VERIFIED_SUFFIX)),
            (LOWER_STAGE, lower_command(self.opt, kernel, emitc)),
            (LOWER_STAGE, translate_command(self.translate, emitc, cpp)),
        )
        found: list[Diagnostic] = []
        for stage, argv in steps:
            result = self.runner(argv, workdir, self.timeout_s)
            log.append(_section(argv, result))
            parsed = parse_mlir_diagnostics(result.stderr, stage)
            found.extend(parsed)
            if result.returncode == -1 or (result.returncode != 0 and not _has_error(parsed)):
                found.append(self._tool_failure(stage, argv[0], kernel, result.returncode))
            if result.returncode != 0 or _has_error(parsed):
                return found
        return found + self._kernel_count(workdir, kernel, cpp)

    def _kernel_count(self, workdir: Path, kernel: str, cpp: str) -> list[Diagnostic]:
        """Return a kernel-count error unless the emitted C++ holds exactly one kernel_main; [] when it does."""
        try:
            count = len(KERNEL_ENTRY.findall((workdir / cpp).read_bytes().decode("utf-8", errors="replace")))
        except OSError:
            count = 0
        if count == 1:
            return []
        message = (
            f"the module emitted {count} kernel functions; a kernel module holds exactly one func.func with the "
            "attribute ttkernel.thread = #ttkernel.thread<noc> (a data-movement kernel) or "
            f"#ttkernel.thread<compute> (a compute kernel), which becomes the kernel's void kernel_main() in {cpp}"
        )
        return [Diagnostic(stage=LOWER_STAGE, severity="error", code="kernel-count", file=kernel, message=message)]

    def _tool_failure(self, stage: str, tool: str, kernel: str, status: int) -> Diagnostic:
        """Return the error of a tool that failed and printed no error the parser reads."""
        if status == -1:
            text = f"{PurePosixPath(tool).name} timed out after {self.timeout_s:g} s on {kernel} and was stopped"
        else:
            text = f"{PurePosixPath(tool).name} exited with status {status} on {kernel} and printed no error " \
                   "that could be parsed"
        return Diagnostic(stage=stage, severity="error", code="exit-status", file=kernel, message=text)

    def _refused(self, files: Mapping[str, str], harness: Mapping[str, str]) -> list[Diagnostic]:
        """Return the errors of every path refused before a build: the base checks, then the kernel-module rule."""
        refused = self._unwritable(files, harness)
        named = {item.file for item in refused}
        hosts = ", ".join(host_sources(harness)) or "none"
        for path in sorted({*files, *harness}):
            if path in named:
                continue
            if path.split("/")[0] == WORK_DIR:
                reason = f"the build keeps the directory {WORK_DIR!r} for the lowered modules"
                refused.append(self._bad_path(path, reason))
            elif path in files and not is_kernel_module(path):
                refused.append(self._not_a_kernel(path, hosts))
            elif path in files and kernel_cpp_path(path) in harness:
                refused.append(self._bad_path(path, f"its kernel {kernel_cpp_path(path)!r} is a harness file"))
        return refused

    @staticmethod
    def _bad_path(path: str, reason: str) -> Diagnostic:
        """Return a bad-path error for `path`."""
        message = f"the file {path!r} cannot be written: {reason}. Rename it and return the files again"
        return compile_diagnostic("error", message, code="bad-path", file=path)

    @staticmethod
    def _not_a_kernel(path: str, hosts: str) -> Diagnostic:
        """Return the error for a model file that is not a kernel module."""
        message = (
            f"the file {path!r} is not a kernel module: return only ttkernel MLIR modules, files ending in .mlir "
            "under a kernels/ directory, each at the path the host program launches with .mlir in place of .cpp; "
            f"the harness provides the host program ({hosts}) and lassi_io.h, so never return a host file"
        )
        return compile_diagnostic("error", message, code="not-a-kernel", file=path)

    def _no_sources(self) -> Diagnostic:
        """Return the error for a build with no kernel module."""
        message = (
            f"no kernel module to lower: {self.name} lowers ttkernel MLIR modules, files ending in .mlir under a "
            "kernels/ directory; return one per kernel, each in a fenced block whose first line is // FILE: <path>"
        )
        return compile_diagnostic("error", message, code="no-sources")
