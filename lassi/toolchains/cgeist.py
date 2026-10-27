"""The pinned cgeist as a lowering toolchain, registered as Toolchain "cgeist-affine" (task P17.41).

The owner's process (2026-10-26): lower the C++ program to affine, linalg,
or a similar level, and have the model write the Tenstorrent kernels in
ttkernel from it. plans/spikes/mlir-issues.md, Half 2, found that the pinned
cgeist (Polygeist 77c04bb2, LLVM 18.0.0git at 26eb4285) aborts on the
items' whole sources (an assertion in tools/cgeist/Lib/ValueCategory.cc:154,
most likely on the I/O harness, inferred), so each item has a compute-only C
unit (lassi.bench Suite.compute_source): its computation as one function
with C linkage and no I/O. The recipe's source_mlir section binds this
toolchain, and the generate stage shows the module it gives
(lassi.core.stages lowered_source).

The level is affine: cgeist raises its scf loops to affine with
--raise-scf-to-affine (tools/cgeist/driver.cc:175-176), and the pinned
polygeist-opt has no --raise-affine-to-linalg pass (the spike's probe:
"Unknown command line argument"), so no linalg is available at this pin;
linalg needs another raiser or another pin (OQ-065).

lower(file, text, function, workdir) writes the unit as `file` (a C source,
a plain name ending in .c) under `workdir` and runs one command there, the
`runner`'s (the runner passes the compile sandbox's,
lassi.executors.sandbox SandboxedCompileRunner):

    cgeist <file> -S --raise-scf-to-affine --function=<function> -o _cgeist/<stem>.affine.mlir

-S stops at MLIR (EmitAssembly, driver.cc:146-147; without it cgeist writes
LLVM IR to a /tmp file, driver.cc:1200-1202), and --function names the one
function to emit by its mangled name, which for a function with C linkage
is its C name (cfunction, driver.cc:211-213; the emitIfFound check of
tools/cgeist/Lib/clang-mlir.cc:5312-5315 and :5387-5390); its callees come with it. A
function name that is not a C identifier, and a file name the build
directory would not hold, are refused before anything runs. The result is
a lassi.core.interfaces Lowering: the module's text, or None with the
tool's parsed diagnostics (clang's and MLIR's `<file>:<line>:<column>:
<severity>: <message>` lines, read as lassi.toolchains.ttkernel_build
reads MLIR's) and, when the tool failed with no parsed error or wrote no
module, one exit-status error naming the status and the last line it
printed (an assertion's, say). It never raises for the tool's failure.

Pins. PIN "polygeist" with PIN_BIN "{CGEIST}": the runner builds it as every
pinned toolchain (lassi.core.runner build_toolchain), the executable
<toolchains root>/<PREFIX_NAME>/<CGEIST> run through the compile sandbox,
whose `--version` must print the pin's EXPECT_VERSION; the pin joins every
trial's toolchain_pins as polygeist. It declares only lowers_source, so no
stage binds it to build a language, and build() refuses.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from lassi.core.capabilities import LOWERS_SOURCE
from lassi.core.files import _check_path
from lassi.core.interfaces import BuildResult, Lowering
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import compile_diagnostic
from lassi.toolchains.ttkernel_build import LOWER_STAGE, parse_mlir_diagnostics

# The flags of the lowering: stop at MLIR, and raise scf loops to affine (tools/cgeist/driver.cc:146-147, :175-176).
EMIT_MLIR = "-S"
RAISE_TO_AFFINE = "--raise-scf-to-affine"
# The directory under the workdir the module is written to, and its suffix; no unit may use the directory.
WORK_DIR = "_cgeist"
MODULE_SUFFIX = ".affine.mlir"
# A unit's suffix: C, so cgeist's clang reads it as C and every function in it has C linkage.
UNIT_SUFFIX = ".c"
# A function name cgeist's --function takes for a function with C linkage: a C identifier.
_C_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
# How much of the tool's last stderr line an exit-status error quotes.
QUOTE_CHARS = 300


@register("Toolchain", "cgeist-affine")
class CgeistAffine(CompilerToolchain):
    """Lowers a compute-only C unit's one function to affine MLIR with the pinned cgeist (module docstring).

    It is a CompilerToolchain for the runner contract alone (the executable,
    the command runner, None meaning subprocess_runner, and the timeout);
    lower() is its one entry, and build() refuses.
    """

    name = "cgeist-affine"
    capabilities = frozenset({LOWERS_SOURCE})
    SOURCE_SUFFIXES = (UNIT_SUFFIX,)
    PIN = "polygeist"
    PIN_BIN = "{CGEIST}"
    LEVEL = "affine"

    def __init__(
        self, *, executable: str = "cgeist", runner: CommandRunner | None = None, timeout_s: float = 300.0
    ) -> None:
        """Keep the pinned cgeist's path, the command runner (the compile sandbox's), and the timeout."""
        super().__init__(executable=str(executable), runner=runner, timeout_s=timeout_s)

    def lower_command(self, file: str, function: str, out: str) -> list[str]:
        """Return the lowering command of unit `file`'s function `function`, writing the module to `out`."""
        return [self.executable, file, EMIT_MLIR, RAISE_TO_AFFINE, f"--function={function}", "-o", out]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in cgeist's `stderr` at the lower stage (parse_mlir_diagnostics)."""
        del files
        return parse_mlir_diagnostics(stderr, LOWER_STAGE)

    def lower(self, file: str, text: str, function: str, workdir: Path) -> Lowering:
        """Write the unit under `workdir`, lower `function`, and return the module or the errors (module docstring)."""
        refusal = _refusal(file, function)
        if refusal is not None:
            return Lowering(module=None, level=self.LEVEL, diagnostics=[refusal])
        workdir = Path(workdir)
        (workdir / file).write_bytes(text.encode("utf-8"))
        out = str(PurePosixPath(WORK_DIR) / (PurePosixPath(file).stem + MODULE_SUFFIX))
        (workdir / WORK_DIR).mkdir(parents=True, exist_ok=True)
        result = self.runner(self.lower_command(file, function, out), workdir, self.timeout_s)
        found = self.parse(result.stderr, {file: text})
        module = workdir / out
        if result.returncode == 0 and module.is_file() and not any(item.severity == "error" for item in found):
            return Lowering(module=module.read_bytes().decode("utf-8", "replace"), level=self.LEVEL,
                            diagnostics=found)
        if not any(item.severity == "error" for item in found):
            found.append(self._failure(file, result.returncode, result.stderr, module.is_file()))
        return Lowering(module=None, level=self.LEVEL, diagnostics=found)

    def _failure(self, file: str, status: int, stderr: str, wrote: bool) -> Diagnostic:
        """Return the exit-status error of a lowering that failed with no parsed error, or that wrote no module."""
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        quoted = "".join(char if " " <= char <= "~" else "?" for char in (lines[-1] if lines else ""))[:QUOTE_CHARS]
        head = next((line for line in lines if "Assertion" in line), quoted)
        said = "".join(char if " " <= char <= "~" else "?" for char in head)[:QUOTE_CHARS]
        if status == -1:
            message = f"cgeist timed out after {self.timeout_s:g} s on {file} and was stopped"
        elif status == 0 and not wrote:
            message = f"cgeist exited with status 0 on {file} but wrote no module"
        else:
            message = f"cgeist exited with status {status} on {file}" + (f": {said}" if said else "")
        return Diagnostic(stage=LOWER_STAGE, severity="error", code="exit-status", file=file, message=message)

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Refuse: this toolchain lowers a compute unit for a prompt (source_mlir) and builds no program."""
        message = (
            f"{self.name} lowers an item's compute unit for the generation prompt (the recipe's source_mlir section) "
            "and builds no program; bind a compiler toolchain for the language instead"
        )
        return BuildResult(artifact=None, diagnostics=[compile_diagnostic("error", message, code="not-a-builder")])


def _refusal(file: str, function: str) -> Diagnostic | None:
    """Return the error of a unit name or function name the lowering refuses before anything runs, else None."""
    try:
        _check_path(file)
    except ValueError as error:
        return _bad(file, str(error))
    if "/" in file or not file.endswith(UNIT_SUFFIX) or file.startswith(WORK_DIR):
        return _bad(file, f"a compute unit is a plain file name ending in {UNIT_SUFFIX}, not under {WORK_DIR}")
    if not _C_NAME.fullmatch(function):
        return Diagnostic(stage=LOWER_STAGE, severity="error", code="bad-function", file=file,
                          message=f"the function name {function!r} is not a C identifier, which --function takes")
    return None


def _bad(file: str, reason: str) -> Diagnostic:
    """Return the bad-path error of a unit name the lowering refuses."""
    return Diagnostic(stage=LOWER_STAGE, severity="error", code="bad-path", file=None,
                      message=f"the compute unit {file!r} cannot be written: {reason}")
