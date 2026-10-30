"""The compile-only TTIR and D2M toolchains, "ttir-ttmetal-compile" and "d2m-ttmetal-compile" (task P17.48).

The languages ttir and d2m hold an item as one MLIR module, at tensor level
(TTIR) or at the level of D2M generic ops with their data movement (D2M),
which the model writes. The pinned tt-mlir (toolchains/tt-mlir.pin, commit
5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c) lowers either through D2M and
TTKernel to TT-Metal and writes a TT-Metal flatbuffer; plans/spikes/
ttir-d2m.md (task P17.46) records the route. Three commands run in the build
directory for the module `<m>.mlir`:

1. verify (stage "verify"): `ttmlir-opt <m>.mlir -o
   _ttmetal/<m>.verified.mlir`; ttmlir-opt parses and verifies the module
   and prints it back, so a parse or verifier error is told apart from a
   lowering error. The printed module is then read for its level (below).
2. lower (stage "lower"): `ttmlir-opt
   --ttir-to-ttmetal-pipeline=enable-multicast-inference=false <m>.mlir -o
   _ttmetal/<m>.ttmetal.mlir`, the full pipeline (TTMetalPipelines.cpp:
   256-303 at the pin), which the spike found is the one entry whose output
   the translation accepts, for both levels. Multicast inference is off as
   the spike's plan for ttsim has it (no NOC multicast in the programs). No
   system-desc-path is given, so the compiler targets its mock Wormhole
   descriptor (spike, section 1): the flatbuffer is compile evidence only.
3. translate (stage "lower"): `ttmlir-translate --ttmetal-to-flatbuffer
   _ttmetal/<m>.ttmetal.mlir -o _ttmetal/<m>.ttm`.

A build passes when all three succeed, the lowered module holds at least one
ttmetal.enqueue_program (TTMetalOps.td:23 at the pin; a module that only
returns its inputs lowers to no device program, spike section 2), and the
flatbuffer carries the file identifier TTM0 (spike, section 1). The
artifact is that flatbuffer, _ttmetal/<m>.ttm. Nothing runs it: the
pinned install has no runtime (spike, section 4), and these toolchains
declare only `diagnostics` and `emits_warnings`, nothing an executor reads
to run a program; recipes bind the executor kind none.

The level rule, read on the printed module of step 1 (comments gone): a
ttir module holds at least one ttir op and no d2m op; a d2m module holds at
least one d2m.generic and no ttir op. So each arm is written at its own
level and never at the other.

Model files: exactly one module, a file ending in .mlir outside the
directory WORK_DIR; any other file, or a second module, is refused with a
bad-path, not-a-module, or module-count error before anything is written or
run.

Diagnostics. Each tool's stderr is read with
lassi.toolchains.ttkernel_build parse_mlir_diagnostics at the step's stage
(`<file>:<line>:<col>: error: <message>` and its other forms, and `LLVM
ERROR:`). A diagnostic that points into the compiler's own intermediate
files under WORK_DIR loses its place, which no model file has, and its
message says where it was. A tool that dies by SIGABRT (exit status 134 in
the sandbox's shell form, -6 from Popen), as this build's assertions do
(spike, section 2), gets one error whose message is the assertion text the
C library printed (`Assertion ... failed.`). A step that fails with no
parsed error, or times out, gets one exit-status error. The raw stderr of
every command, each after a line naming the command and its status, is the
attachment STDERR_ATTACHMENT.

Pins. PIN is tt-mlir with PIN_BIN its TTMLIR_OPT, so the runner builds the
class with the pinned ttmlir-opt as `executable` and checks its --version
against EXPECT_VERSION; TOOL_PINS names the same pin's TTMLIR_TRANSLATE,
which the runner checks with check_tool_pin (the install record) and
--version and passes as `tools` (lassi.core.runner, _tool_pins).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from lassi.core.files import _check_path
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import STDERR_ATTACHMENT, CommandResult, CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import compile_diagnostic
from lassi.toolchains.ttkernel_build import (
    MLIR_PIN,
    OPT_KEY,
    TRANSLATE_KEY,
    check_tool_install,
    parse_mlir_diagnostics,
)

# The full pipeline, its one option, and the translation (plans/spikes/ttir-d2m.md, Consequences for the plan).
PIPELINE = "--ttir-to-ttmetal-pipeline"
PIPELINE_OPTIONS = "enable-multicast-inference=false"
TRANSLATION = "--ttmetal-to-flatbuffer"
# The suffix of a module, the intermediate files' directory, and the suffixes of the files written there.
MODULE_SUFFIX = ".mlir"
WORK_DIR = "_ttmetal"
VERIFIED_SUFFIX = ".verified.mlir"
TTMETAL_SUFFIX = ".ttmetal.mlir"
FLATBUFFER_SUFFIX = ".ttm"
# A flatbuffer's file identifier sits at bytes 4 to 8; tt-mlir's TT-Metal binary schema names TTM0 (spike, section 1).
FILE_IDENTIFIER = b"TTM0"
IDENTIFIER_SPAN = slice(4, 8)
# The op that enqueues one device program in the lowered TTMetal module (TTMetalOps.td:23 at the pin).
PROGRAM_OP = re.compile(r"\bttmetal\.enqueue_program\b")
# The exit statuses of a death by SIGABRT: the sandbox's shell form (128 + 6) and Popen's (-6).
ABORT_STATUSES = (134, -6)
# The C library's assertion message: `<prog>: <file>:<line>: <function>: Assertion `<expr>' failed.`
_ASSERTION = re.compile(r"(?:[^\s:]+: )?(?P<text>.*\bAssertion\b.*\bfailed\.)\s*")
# The diagnostic stages of the steps.
VERIFY_STAGE = "verify"
LOWER_STAGE = "lower"
DEFAULT_OPT = "ttmlir-opt"
DEFAULT_TRANSLATE = "ttmlir-translate"


def verify_command(opt: str, module: str, out: str) -> list[str]:
    """Return the verify step's command: ttmlir-opt parses and verifies `module` and prints it to `out`."""
    return [opt, module, "-o", out]


def lower_command(opt: str, module: str, out: str) -> list[str]:
    """Return the lowering step's command: `module` through the full TTIR-to-TTMetal pipeline to `out`."""
    return [opt, f"{PIPELINE}={PIPELINE_OPTIONS}", module, "-o", out]


def translate_command(translate: str, ttmetal: str, out: str) -> list[str]:
    """Return the translation step's command: the TTMetal module `ttmetal` to a flatbuffer at `out`."""
    return [translate, TRANSLATION, ttmetal, "-o", out]


def work_path(module: str, suffix: str) -> str:
    """Return the path under WORK_DIR of one of `module`'s intermediate files, its suffix replaced by `suffix`."""
    return str(PurePosixPath(WORK_DIR) / PurePosixPath(module).with_suffix("")) + suffix


def assertion_text(stderr: str) -> str | None:
    """Return the text of the first failed assertion in `stderr`, without the program name; None when none."""
    for line in stderr.split("\n"):
        match = _ASSERTION.fullmatch(line.rstrip())
        if match:
            return match.group("text")
    return None


def _has_error(diagnostics: Sequence[Diagnostic]) -> bool:
    """Return True when any diagnostic is an error."""
    return any(item.severity == "error" for item in diagnostics)


def _section(argv: Sequence[str], result: CommandResult) -> str:
    """Return one command's part of the stderr attachment: a line naming it and its status, then its stderr."""
    text = result.stderr if result.stderr.endswith("\n") or not result.stderr else result.stderr + "\n"
    return f"$ {' '.join(argv)}  (exit {result.returncode})\n{text}"


def _placed(diagnostic: Diagnostic) -> Diagnostic:
    """Return `diagnostic` without its place when it points into WORK_DIR, its message saying where it was."""
    if diagnostic.file is None or diagnostic.file.split("/")[0] != WORK_DIR:
        return diagnostic
    where = f"{diagnostic.file}:{diagnostic.line}" + (f":{diagnostic.column}" if diagnostic.column else "")
    message = f"{diagnostic.message} (at {where}, in the compiler's own lowered module, not in your file)"
    return Diagnostic(stage=diagnostic.stage, severity=diagnostic.severity, code=diagnostic.code, message=message)


class TtMlirCompile(CompilerToolchain):
    """Compiles one TTIR or D2M module to a TT-Metal flatbuffer with the pinned tt-mlir; nothing runs it.

    Subclasses set `name`, LANGUAGE (the language's name in messages),
    and level_errors (the level rule; see the module docstring).
    """

    name: str
    LANGUAGE: str = ""
    capabilities = frozenset({"diagnostics", "emits_warnings"})
    SOURCE_SUFFIXES = (MODULE_SUFFIX,)
    PIN = MLIR_PIN
    PIN_BIN = "{" + OPT_KEY + "}"
    TOOL_PINS = {MLIR_PIN: (TRANSLATE_KEY,)}

    def __init__(
        self,
        *,
        executable: str = DEFAULT_OPT,
        tools: Mapping[str, str] | None = None,
        runner: CommandRunner | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        """Keep ttmlir-opt (`executable`), ttmlir-translate (`tools`, by pin key), the runner, and the timeout.

        The runner passes the pinned ttmlir-opt as `executable` and the
        pinned ttmlir-translate under TTMLIR_TRANSLATE in `tools`; the
        defaults (tests and local checks) name both tools on PATH.
        """
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)
        self.translate = str(dict(tools or {}).get(TRANSLATE_KEY, DEFAULT_TRANSLATE))

    @classmethod
    def check_tool_pin(cls, pin_name: str, home: Path, pin: Mapping[str, str]) -> None:
        """Raise ValueError unless `home` is the install of the tt-mlir pin (its lassi-install.txt names it)."""
        if pin_name != MLIR_PIN:
            raise ValueError(f"{cls.name} names no tool pin {pin_name!r}; its tool pin is {MLIR_PIN}")
        check_tool_install(home, pin)

    def level_errors(self, module: str, printed: str) -> list[Diagnostic]:
        """Return the level rule's errors for `module`, read on its printed form `printed`; [] when it holds."""
        raise NotImplementedError

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write the files, then verify, lower, and translate the one module (see the module docstring).

        Every path is checked and every refusal made before anything is
        written; a refused file set runs nothing. The artifact is the
        flatbuffer when every step passed and every check held.
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
        modules = self._sources(files)
        if not modules:
            return BuildResult(artifact=None, diagnostics=[self._no_sources()], stderr_ref="")
        module = modules[0]
        (workdir / WORK_DIR).mkdir(parents=True, exist_ok=True)
        flatbuffer = workdir / work_path(module, FLATBUFFER_SUFFIX)
        if flatbuffer.is_file() or flatbuffer.is_symlink():
            flatbuffer.unlink()
        log: list[str] = []
        diagnostics = self._compiled(module, workdir, log)
        (workdir / STDERR_ATTACHMENT).write_bytes("".join(log).encode("utf-8"))
        artifact = flatbuffer if not _has_error(diagnostics) and flatbuffer.is_file() else None
        return BuildResult(artifact=artifact, diagnostics=diagnostics, stderr_ref=STDERR_ATTACHMENT)

    def _compiled(self, module: str, workdir: Path, log: list[str]) -> list[Diagnostic]:
        """Run the three steps on `module` and check their outputs; return the diagnostics (an error stops them)."""
        verified, ttmetal = work_path(module, VERIFIED_SUFFIX), work_path(module, TTMETAL_SUFFIX)
        steps = (
            (VERIFY_STAGE, verify_command(self.executable, module, verified)),
            (LOWER_STAGE, lower_command(self.executable, module, ttmetal)),
            (LOWER_STAGE, translate_command(self.translate, ttmetal, work_path(module, FLATBUFFER_SUFFIX))),
        )
        found: list[Diagnostic] = []
        for index, (stage, argv) in enumerate(steps):
            result = self.runner(argv, workdir, self.timeout_s)
            log.append(_section(argv, result))
            found.extend(self._read(result, stage, argv[0], module))
            if result.returncode != 0 or _has_error(found):
                return found
            if index == 0:
                found.extend(self.level_errors(module, _text(workdir / verified)))
                if _has_error(found):
                    return found
        return found + self._outputs(module, workdir)

    def _read(self, result: CommandResult, stage: str, tool: str, module: str) -> list[Diagnostic]:
        """Return one step's diagnostics: its parsed stderr, then an abort or exit-status error when one is due."""
        parsed = [_placed(item) for item in parse_mlir_diagnostics(result.stderr, stage)]
        name = PurePosixPath(tool).name
        if result.returncode in ABORT_STATUSES:
            text = assertion_text(result.stderr)
            said = f"on an internal assertion: {text}" if text else "and printed no assertion text"
            message = (
                f"{name} aborted (exit status {result.returncode}) while compiling {module} {said}. The compiler "
                "stopped on an input it does not support; change the construct the assertion names"
            )
            return [*parsed, Diagnostic(stage=stage, severity="error", code="abort", file=module, message=message)]
        if result.returncode == -1:
            message = f"{name} timed out after {self.timeout_s:g} s on {module} and was stopped"
        elif result.returncode != 0 and not _has_error(parsed):
            message = (f"{name} exited with status {result.returncode} on {module} and printed no error that could "
                       "be parsed")
        else:
            return parsed
        return [*parsed, Diagnostic(stage=stage, severity="error", code="exit-status", file=module, message=message)]

    def _outputs(self, module: str, workdir: Path) -> list[Diagnostic]:
        """Return the errors of the lowered module and the flatbuffer: no device program, or no TT-Metal flatbuffer."""
        found: list[Diagnostic] = []
        if not PROGRAM_OP.search(_text(workdir / work_path(module, TTMETAL_SUFFIX))):
            message = (
                f"{module} lowered to no device program (the lowered module has no ttmetal.enqueue_program): "
                f"every output must be computed on the device by the module's {self.LANGUAGE} ops, never by "
                "returning an input or a constant"
            )
            found.append(Diagnostic(stage=LOWER_STAGE, severity="error", code="no-device-program", file=module,
                                    message=message))
        try:
            head = (workdir / work_path(module, FLATBUFFER_SUFFIX)).read_bytes()[:8]
        except OSError:
            head = b""
        if head[IDENTIFIER_SPAN] != FILE_IDENTIFIER:
            message = (
                f"ttmlir-translate exited with status 0 but wrote no TT-Metal flatbuffer for {module} (no file "
                f"identifier {FILE_IDENTIFIER.decode('ascii')})"
            )
            found.append(Diagnostic(stage=LOWER_STAGE, severity="error", code="no-artifact", file=module,
                                    message=message))
        return found

    def _refused(self, files: Mapping[str, str], harness: Mapping[str, str]) -> list[Diagnostic]:
        """Return the errors of every path refused before a build: the base checks, then the module rules."""
        refused = self._unwritable(files, harness)
        named = {item.file for item in refused}
        modules = [path for path in sorted(files) if PurePosixPath(path).suffix == MODULE_SUFFIX]
        for path in sorted({*files, *harness}):
            if path in named:
                continue
            if path.split("/")[0] == WORK_DIR:
                message = (f"the file {path!r} cannot be written: the build keeps the directory {WORK_DIR!r} for "
                           "the lowered module. Rename it and return the files again")
                refused.append(compile_diagnostic("error", message, code="bad-path", file=path))
            elif path in files and path not in modules:
                message = (f"the file {path!r} is not a {self.LANGUAGE} module: return only the one MLIR module the "
                           "task names, a file ending in .mlir, and no other file")
                refused.append(compile_diagnostic("error", message, code="not-a-module", file=path))
        if len(modules) > 1:
            message = (f"{len(modules)} modules were returned ({', '.join(modules)}); the item is one {self.LANGUAGE} "
                       "module in one file, the file the task names")
            refused.append(compile_diagnostic("error", message, code="module-count"))
        return refused

    def _no_sources(self) -> Diagnostic:
        """Return the error for a build with no module."""
        message = (
            f"no module to compile: {self.name} compiles one {self.LANGUAGE} MLIR module, a file ending in .mlir; "
            "return it in a fenced block whose first line is // FILE: <path>"
        )
        return compile_diagnostic("error", message, code="no-sources")


def _text(path: Path) -> str:
    """Return a file's text as UTF-8 with replacement, or "" when it cannot be read."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _level_error(module: str, message: str) -> Diagnostic:
    """Return a level-rule error for `module` at stage verify."""
    return Diagnostic(stage=VERIFY_STAGE, severity="error", code="wrong-level", file=module, message=message)


# An op of each dialect as ttmlir-opt prints it (custom form `ttir.add` or generic form `"ttir.add"`).
_TTIR_OP = re.compile(r"\bttir\.[a-z_][a-z0-9_]*")
_D2M_OP = re.compile(r"\bd2m\.[a-z_][a-z0-9_]*")
_D2M_GENERIC = re.compile(r"\bd2m\.generic\b")


@register("Toolchain", "ttir-ttmetal-compile")
class TtirCompile(TtMlirCompile):
    """Compiles one TTIR module to a TT-Metal flatbuffer, compile only (the language ttir)."""

    name = "ttir-ttmetal-compile"
    LANGUAGE = "TTIR"

    def level_errors(self, module: str, printed: str) -> list[Diagnostic]:
        """Return an error unless the printed module holds a ttir op and no d2m op."""
        d2m = sorted(set(_D2M_OP.findall(printed)))
        if d2m:
            return [_level_error(module, f"the module holds D2M ops ({', '.join(d2m[:5])}); a TTIR module is "
                                         "written in TTIR ops on tensors, and the compiler lowers it to D2M itself")]
        if not _TTIR_OP.search(printed):
            return [_level_error(module, "the module holds no TTIR op; write the computation in TTIR ops on "
                                         "tensors, such as ttir.add")]
        return []


@register("Toolchain", "d2m-ttmetal-compile")
class D2mCompile(TtMlirCompile):
    """Compiles one D2M module to a TT-Metal flatbuffer, compile only (the language d2m)."""

    name = "d2m-ttmetal-compile"
    LANGUAGE = "D2M"

    def level_errors(self, module: str, printed: str) -> list[Diagnostic]:
        """Return an error unless the printed module holds a d2m.generic and no ttir op."""
        ttir = sorted(set(_TTIR_OP.findall(printed)))
        if ttir:
            return [_level_error(module, f"the module holds TTIR ops ({', '.join(ttir[:5])}); a D2M module writes "
                                         "its computation and data movement in d2m ops, d2m.generic among them")]
        if not _D2M_GENERIC.search(printed):
            return [_level_error(module, "the module holds no d2m.generic; a D2M module computes and moves data in "
                                         "d2m.generic ops on the device grid")]
        return []
