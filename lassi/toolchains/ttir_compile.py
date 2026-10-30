"""The compile-only tt-mlir toolchains: ttir-ttmetal-compile, d2m-ttmetal-compile (P17.48), ttir-ttnn-compile (P17.57).

The languages ttir and d2m hold an item as one MLIR module, at tensor level
(TTIR) or at the level of D2M generic ops with their data movement (D2M),
which the model writes. The pinned tt-mlir (toolchains/tt-mlir.pin, commit
5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c) lowers either through D2M and
TTKernel to TT-Metal and writes a TT-Metal flatbuffer; plans/spikes/
ttir-d2m.md (task P17.46) records the route. The language ttnn (route A of
plans/spikes/scatter-gather-workarounds.md, the owner's answer to OQ-112)
holds a TTIR module too, which the pinned tt-mlir lowers through its TT-NN
backend to TT-NN ops and writes as a TT-NN flatbuffer; the spike's section
"Compile probe at the pin (2026-10-29)" records the route. Each toolchain
names its route as a Backend (TTMETAL or TTNN). Three commands run in the
build directory for the module `<m>.mlir`, with <w> the route's work
directory (_ttmetal or _ttnn):

1. verify (stage "verify"): `ttmlir-opt <m>.mlir -o <w>/<m>.verified.mlir`;
   ttmlir-opt parses and verifies the module and prints it back, so a parse
   or verifier error is told apart from a lowering error. The printed module
   is then read for its level (below).
2. lower (stage "lower"): TT-Metal: `ttmlir-opt
   --ttir-to-ttmetal-pipeline=enable-multicast-inference=false <m>.mlir -o
   _ttmetal/<m>.ttmetal.mlir`, the full pipeline (TTMetalPipelines.cpp:
   256-303 at the pin), which the spike found is the one entry whose output
   the translation accepts, for both levels; multicast inference is off as
   the spike's plan for ttsim has it. TT-NN: `ttmlir-opt
   --ttir-to-ttnn-backend-pipeline <m>.mlir -o _ttnn/<m>.ttnn.mlir`
   (TTNNPipelines.cpp:527-533, :599-603), with its defaults. Neither gives a
   system-desc-path, so the compiler targets its mock Wormhole descriptor
   (mock-system-desc-arch defaults to wormhole_b0): the flatbuffer is
   compile evidence only.
3. translate (stage "lower"): `ttmlir-translate --ttmetal-to-flatbuffer
   _ttmetal/<m>.ttmetal.mlir -o _ttmetal/<m>.ttm`, or `ttmlir-translate
   --ttnn-to-flatbuffer _ttnn/<m>.ttnn.mlir -o _ttnn/<m>.ttnn`.

A build passes when all three succeed, the lowered module holds device work,
and the flatbuffer carries the route's file identifier: TTM0 or TTNN. Both
translations write size-prefixed flatbuffers (TTMetalToFlatbuffer.cpp:1068
and TTNNToFlatbuffer.cpp:4205 call the generated FinishSizePrefixed...
functions), so the identifier sits at bytes 8 to 12, after the size and the
root offset (IDENTIFIER_SPAN; the probes' outputs read with od). Device
work: on TT-Metal at least one ttmetal.enqueue_program (TTMetalOps.td:23;
a module that only returns its inputs lowers to no device program,
ttir-d2m.md section 2); on TT-NN at least one ttnn op outside
TTNN_GLUE, the ops that only fetch the device, free, place, move, or create
a tensor (a module that returns its input lowers to no ttnn op at all, the
probe's identity case). The artifact is the flatbuffer. Nothing runs it:
these toolchains declare only `diagnostics` and `emits_warnings`, nothing
an executor reads to run a program; recipes bind the executor kind none.

The level rule, read on the printed module of step 1 (comments gone): a
ttir or ttnn module holds at least one ttir op and no op of a dialect the
compiler writes below TTIR on its route (d2m; and ttnn on the TT-NN route);
a d2m module holds at least one d2m.generic and no ttir op. So each arm is
written at its own level and never at another.

Model files: exactly one module, a file ending in .mlir outside the route's
work directory; any other file, or a second module, is refused with a
bad-path, not-a-module, or module-count error before anything is written or
run.

Diagnostics. Each tool's stderr is read with
lassi.toolchains.ttkernel_build parse_mlir_diagnostics at the step's stage
(`<file>:<line>:<col>: error: <message>` and its other forms, and `LLVM
ERROR:`). A diagnostic that points into the compiler's own intermediate
files under the work directory loses its place, which no model file has,
and its message says where it was. A tool that dies by SIGABRT (exit status
134 in the sandbox's shell form, -6 from Popen), as this build's assertions
do (ttir-d2m.md section 2), gets one error whose message is the assertion
text the C library printed (`Assertion ... failed.`). A step that fails with
no parsed error, or times out, gets one exit-status error. The raw stderr of
every command, each after a line naming the command and its status, is the
attachment STDERR_ATTACHMENT. The TT-NN pipeline prints one warning on every
module ("Empty argument type map provided. Skipping argument type
population.", at the module's first op; the probe), which is kept as
printed, a warning like any other.

Pins. PIN is tt-mlir with PIN_BIN its TTMLIR_OPT, so the runner builds the
class with the pinned ttmlir-opt as `executable` and checks its --version
against EXPECT_VERSION; TOOL_PINS names the same pin's TTMLIR_TRANSLATE,
which the runner checks with check_tool_pin (the install record) and
--version and passes as `tools` (lassi.core.runner, _tool_pins).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Backend:
    """One route of a compile-only tt-mlir toolchain: its lowering, its translation, and where its files go.

    `name` names the backend in messages; `pipeline` is ttmlir-opt's
    lowering argument; `translation` is ttmlir-translate's; the
    intermediate files go under `work_dir`, the lowered module with
    `lowered_suffix` and the flatbuffer with `flatbuffer_suffix`; a
    flatbuffer of this route carries `identifier` at IDENTIFIER_SPAN.
    """

    name: str
    pipeline: str
    translation: str
    work_dir: str
    lowered_suffix: str
    flatbuffer_suffix: str
    identifier: bytes


# The TT-Metal route: the full pipeline, its one option, and the translation (plans/spikes/ttir-d2m.md, Consequences).
PIPELINE = "--ttir-to-ttmetal-pipeline"
PIPELINE_OPTIONS = "enable-multicast-inference=false"
TRANSLATION = "--ttmetal-to-flatbuffer"
# The suffix of a module, the intermediate files' directory, and the suffixes of the files written there.
MODULE_SUFFIX = ".mlir"
WORK_DIR = "_ttmetal"
VERIFIED_SUFFIX = ".verified.mlir"
TTMETAL_SUFFIX = ".ttmetal.mlir"
FLATBUFFER_SUFFIX = ".ttm"
# tt-mlir's TT-Metal binary schema names TTM0 (ttir-d2m.md, section 1).
FILE_IDENTIFIER = b"TTM0"
# Both translations write size-prefixed flatbuffers: bytes 0 to 4 hold the size, 4 to 8 the root offset, and 8 to
# 12 the file identifier (TTMetalToFlatbuffer.cpp:1068, TTNNToFlatbuffer.cpp:4205 at the pin; od of the probes'
# .ttm and .ttnn files, rx 20261029-205004-exec-e50a and 20261029-204857-exec-9cd1; sweep13's first launch, which
# read bytes 4 to 8, scored every flatbuffer as missing).
IDENTIFIER_SPAN = slice(8, 12)
TTMETAL = Backend(
    name="TT-Metal", pipeline=f"{PIPELINE}={PIPELINE_OPTIONS}", translation=TRANSLATION, work_dir=WORK_DIR,
    lowered_suffix=TTMETAL_SUFFIX, flatbuffer_suffix=FLATBUFFER_SUFFIX, identifier=FILE_IDENTIFIER,
)
# The TT-NN route (task P17.57): the backend pipeline with its defaults and the translation; tt-mlir's TT-NN binary
# schema names the identifier TTNN and the extension ttnn (include/ttmlir/Target/TTNN/binary.fbs:18-19 at the pin).
TTNN_PIPELINE = "--ttir-to-ttnn-backend-pipeline"
TTNN_TRANSLATION = "--ttnn-to-flatbuffer"
TTNN = Backend(
    name="TT-NN", pipeline=TTNN_PIPELINE, translation=TTNN_TRANSLATION, work_dir="_ttnn", lowered_suffix=".ttnn.mlir",
    flatbuffer_suffix=".ttnn", identifier=b"TTNN",
)
# The op that enqueues one device program in the lowered TTMetal module (TTMetalOps.td:23 at the pin).
PROGRAM_OP = re.compile(r"\bttmetal\.enqueue_program\b")
# A ttnn op as ttmlir-opt prints it, not an attribute (#ttnn.), a type (!ttnn.), or an attribute name (`ttnn.x =`).
_TTNN_OP = re.compile(r'(?<![#!\w.])"?(ttnn\.[a-z_][a-z0-9_]*)\b(?!"?\s*=)')
# The ttnn ops that fetch the device, free, place, move, or create a tensor, and compute nothing from the inputs; the
# probe's lowered modules hold get_device, deallocate, and to_layout beside the ops that compute, and zeros and ones
# from ttir.zeros and ttir.ones (spike, Compile probe at the pin).
TTNN_GLUE = frozenset({
    "arange", "constant", "deallocate", "empty", "from_device", "full", "get_device", "ones", "to_device", "to_layout",
    "to_memory_config", "zeros",
})
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


def lower_command(opt: str, module: str, out: str, pipeline: str = TTMETAL.pipeline) -> list[str]:
    """Return the lowering step's command: `module` through `pipeline` (default: TT-Metal's) to `out`."""
    return [opt, pipeline, module, "-o", out]


def translate_command(translate: str, lowered: str, out: str, translation: str = TRANSLATION) -> list[str]:
    """Return the translation step's command: the lowered module to a flatbuffer at `out` (default: TT-Metal's)."""
    return [translate, translation, lowered, "-o", out]


def work_path(module: str, suffix: str, work_dir: str = WORK_DIR) -> str:
    """Return the path under `work_dir` of one of `module`'s intermediate files, its suffix replaced by `suffix`."""
    return str(PurePosixPath(work_dir) / PurePosixPath(module).with_suffix("")) + suffix


def ttnn_ops(lowered: str) -> list[str]:
    """Return the names of the ttnn ops in a lowered module's text, without the dialect prefix, in order."""
    return [name.split(".", 1)[1] for name in _TTNN_OP.findall(lowered)]


def has_file_identifier(data: bytes, identifier: bytes = FILE_IDENTIFIER) -> bool:
    """Return True when `data` is a size-prefixed flatbuffer, as the pinned translations write, with `identifier`."""
    return data[IDENTIFIER_SPAN] == identifier


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


def _placed(diagnostic: Diagnostic, work_dir: str = WORK_DIR) -> Diagnostic:
    """Return `diagnostic` without its place when it points into `work_dir`, its message saying where it was."""
    if diagnostic.file is None or diagnostic.file.split("/")[0] != work_dir:
        return diagnostic
    where = f"{diagnostic.file}:{diagnostic.line}" + (f":{diagnostic.column}" if diagnostic.column else "")
    message = f"{diagnostic.message} (at {where}, in the compiler's own lowered module, not in your file)"
    return Diagnostic(stage=diagnostic.stage, severity=diagnostic.severity, code=diagnostic.code, message=message)


def _no_device_work(module: str, missing: str, language: str) -> Diagnostic:
    """Return the error of a module whose lowered form holds no device work; `missing` says what it lacks."""
    message = (
        f"{module} lowered to no device program ({missing}): every output must be computed on the device by the "
        f"module's {language} ops, never by returning an input or a constant"
    )
    return Diagnostic(stage=LOWER_STAGE, severity="error", code="no-device-program", file=module, message=message)


class TtMlirCompile(CompilerToolchain):
    """Compiles one TTIR or D2M module to a flatbuffer with the pinned tt-mlir; nothing runs it.

    Subclasses set `name`, LANGUAGE (the language's name in messages),
    BACKEND (the route; TTMETAL by default), and level_errors (the level
    rule; see the module docstring); a route other than TT-Metal also sets
    device_work_errors.
    """

    name: str
    LANGUAGE: str = ""
    BACKEND: Backend = TTMETAL
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

    def device_work_errors(self, module: str, lowered: str) -> list[Diagnostic]:
        """Return an error unless the lowered TTMetal module `lowered` enqueues at least one device program."""
        if PROGRAM_OP.search(lowered):
            return []
        return [_no_device_work(module, "the lowered module has no ttmetal.enqueue_program", self.LANGUAGE)]

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write the files, then verify, lower, and translate the one module (see the module docstring).

        Every path is checked and every refusal made before anything is
        written; a refused file set runs nothing. The artifact is the
        flatbuffer when every step passed and every check held.
        """
        workdir, backend = Path(workdir), self.BACKEND
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
        (workdir / backend.work_dir).mkdir(parents=True, exist_ok=True)
        flatbuffer = workdir / work_path(module, backend.flatbuffer_suffix, backend.work_dir)
        if flatbuffer.is_file() or flatbuffer.is_symlink():
            flatbuffer.unlink()
        log: list[str] = []
        diagnostics = self._compiled(module, workdir, log)
        (workdir / STDERR_ATTACHMENT).write_bytes("".join(log).encode("utf-8"))
        artifact = flatbuffer if not _has_error(diagnostics) and flatbuffer.is_file() else None
        return BuildResult(artifact=artifact, diagnostics=diagnostics, stderr_ref=STDERR_ATTACHMENT)

    def _compiled(self, module: str, workdir: Path, log: list[str]) -> list[Diagnostic]:
        """Run the three steps on `module` and check their outputs; return the diagnostics (an error stops them)."""
        backend = self.BACKEND
        verified = work_path(module, VERIFIED_SUFFIX, backend.work_dir)
        lowered = work_path(module, backend.lowered_suffix, backend.work_dir)
        flatbuffer = work_path(module, backend.flatbuffer_suffix, backend.work_dir)
        steps = (
            (VERIFY_STAGE, verify_command(self.executable, module, verified)),
            (LOWER_STAGE, lower_command(self.executable, module, lowered, backend.pipeline)),
            (LOWER_STAGE, translate_command(self.translate, lowered, flatbuffer, backend.translation)),
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
        parsed = [_placed(item, self.BACKEND.work_dir) for item in parse_mlir_diagnostics(result.stderr, stage)]
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
        """Return the errors of the lowered module and the flatbuffer: no device work, or no flatbuffer of the route."""
        backend = self.BACKEND
        found = self.device_work_errors(module, _text(workdir / work_path(module, backend.lowered_suffix,
                                                                          backend.work_dir)))
        try:
            data = (workdir / work_path(module, backend.flatbuffer_suffix, backend.work_dir)).read_bytes()
        except OSError:
            data = b""
        if not has_file_identifier(data, backend.identifier):
            message = (
                f"ttmlir-translate exited with status 0 but wrote no {backend.name} flatbuffer for {module} (no file "
                f"identifier {backend.identifier.decode('ascii')} after the size prefix)"
            )
            found.append(Diagnostic(stage=LOWER_STAGE, severity="error", code="no-artifact", file=module,
                                    message=message))
        return found

    def _refused(self, files: Mapping[str, str], harness: Mapping[str, str]) -> list[Diagnostic]:
        """Return the errors of every path refused before a build: the base checks, then the module rules."""
        refused = self._unwritable(files, harness)
        named = {item.file for item in refused}
        modules = [path for path in sorted(files) if PurePosixPath(path).suffix == MODULE_SUFFIX]
        work_dir = self.BACKEND.work_dir
        for path in sorted({*files, *harness}):
            if path in named:
                continue
            if path.split("/")[0] == work_dir:
                message = (f"the file {path!r} cannot be written: the build keeps the directory {work_dir!r} for "
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
    """Compiles one TTIR module to a TT-Metal flatbuffer, compile only (the language ttir).

    BELOW names the dialects the compiler writes below TTIR on the route,
    each with its op pattern, which a model's module may not hold;
    LOWERS_TO is the one the level message names.
    """

    name = "ttir-ttmetal-compile"
    LANGUAGE = "TTIR"
    BELOW: tuple[tuple[str, re.Pattern[str]], ...] = (("D2M", _D2M_OP),)
    LOWERS_TO = "D2M"

    def level_errors(self, module: str, printed: str) -> list[Diagnostic]:
        """Return an error unless the printed module holds a ttir op and no op of a dialect in BELOW."""
        for dialect, pattern in self.BELOW:
            below = sorted(set(pattern.findall(printed)))
            if below:
                message = (f"the module holds {dialect} ops ({', '.join(below[:5])}); a TTIR module is written in "
                           f"TTIR ops on tensors, and the compiler lowers it to {self.LOWERS_TO} itself")
                return [_level_error(module, message)]
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


@register("Toolchain", "ttir-ttnn-compile")
class TtirTtnnCompile(TtirCompile):
    """Compiles one TTIR module through the TT-NN backend to a TT-NN flatbuffer, compile only (the language ttnn)."""

    name = "ttir-ttnn-compile"
    LANGUAGE = "TTIR"
    BACKEND = TTNN
    BELOW = (("D2M", _D2M_OP), ("TTNN", _TTNN_OP))
    LOWERS_TO = "TTNN"

    def device_work_errors(self, module: str, lowered: str) -> list[Diagnostic]:
        """Return an error unless the lowered TT-NN module holds a ttnn op outside TTNN_GLUE."""
        ops = ttnn_ops(lowered)
        if any(op not in TTNN_GLUE for op in ops):
            return []
        held = ", ".join(f"ttnn.{op}" for op in sorted(set(ops))) or "no ttnn op at all"
        missing = f"the lowered module has no ttnn op that computes; it holds {held}"
        return [_no_device_work(module, missing, self.LANGUAGE)]
