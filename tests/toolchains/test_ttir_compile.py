"""Tests for the compile-only TTIR and D2M toolchains (task P17.48; plans/spikes/ttir-d2m.md).

Bible: IR Strategy (TTIR and D2M above ttkernel), Toolchain Pins (tt-mlir
0.9.0.dev20260221 at 5f396fd6), Component Interfaces (Toolchain: files ->
artifact + parsed diagnostics), Agent Rules 2, 6, and 10.

The design these tests fix. The languages ttir and d2m are bound to the
Toolchains "ttir-ttmetal-compile" and "d2m-ttmetal-compile"
(lassi.toolchains.ttir_compile). The model writes one module, <m>.mlir; the
build runs, in the build directory:

    ttmlir-opt <m>.mlir -o _ttmetal/<m>.verified.mlir
    ttmlir-opt --ttir-to-ttmetal-pipeline=enable-multicast-inference=false <m>.mlir
        -o _ttmetal/<m>.ttmetal.mlir
    ttmlir-translate --ttmetal-to-flatbuffer _ttmetal/<m>.ttmetal.mlir -o _ttmetal/<m>.ttm

The artifact is the flatbuffer when every step passed, the lowered module
holds a ttmetal.enqueue_program, and the flatbuffer's identifier is TTM0. A
verify error, a lowering error, or an assertion abort (exit status 134, the
assertion text as the message) is an error diagnostic and stops the build;
a module at the other level is refused after verify. Nothing here declares
that an executor may run the output.

No tool runs here: a fake command runner answers each command with
SYNTHETIC text (MLIR's documented diagnostic format and glibc's assertion
format, not captured from the pinned tools) and writes the files a real run
would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.toolchains import CommandResult, ttir_compile
from lassi.toolchains import pins as pins_module

OPT = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-opt"
TRANSLATE = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-translate"
WORK = "_ttmetal"
ATTACHMENT = "compile.stderr"
MODULE = "probe.mlir"
NAMES = {"ttir": "ttir-ttmetal-compile", "d2m": "d2m-ttmetal-compile"}
# SYNTHETIC modules written for these tests; neither is any bench item's.
TTIR_TEXT = (
    "// SYNTHETIC TTIR module written for this test\n"
    "func.func @probe(%x: tensor<64x128xbf16>, %y: tensor<64x128xbf16>) -> tensor<64x128xbf16> {\n"
    '  %r = "ttir.maximum"(%x, %y) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>\n'
    "  return %r : tensor<64x128xbf16>\n"
    "}\n"
)
D2M_TEXT = "// SYNTHETIC D2M module written for this test\nmodule {\n  // d2m.generic ...\n}\n"
LOWERED = "module {\n  ttmetal.enqueue_program\n}\n"
# The first 12 bytes of a real TT-Metal flatbuffer, size-prefixed (the size 52284, the root offset 32, then TTM0): od
# of the P17.46 probe's eltwise_binary.ttm (rx 20261029-205004-exec-e50a); the rest is zeros here.
FLATBUFFER = b"\x3c\xcc\x00\x00\x20\x00\x00\x00TTM0" + bytes(16)


@dataclass
class FakeTools:
    """A CommandRunner standing in for ttmlir-opt and ttmlir-translate; it runs nothing.

    `answers` maps a step (verify, lower, translate) to (status, stderr); a
    step without one succeeds and writes its -o file: `printed` for verify,
    `lowered` for lower, and `flatbuffer` for translate.
    """

    printed: str = TTIR_TEXT
    lowered: str = LOWERED
    flatbuffer: bytes = FLATBUFFER
    answers: dict[str, tuple[int, str]] = field(default_factory=dict)
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and return the scripted result, writing the tool's output on success."""
        step = self._step(list(argv))
        self.calls.append((step, list(argv)))
        status, stderr = self.answers.get(step, (0, ""))
        if status == 0:
            out = Path(cwd) / argv[argv.index("-o") + 1]
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {"verify": self.printed.encode(), "lower": self.lowered.encode(), "translate": self.flatbuffer}
            out.write_bytes(data[step])
        return CommandResult(status, "", stderr)

    @staticmethod
    def _step(argv: list[str]) -> str:
        """Name the step a command is."""
        if argv[0] == TRANSLATE:
            return "translate"
        return "lower" if any(arg.startswith("--ttir-to-ttmetal-pipeline") for arg in argv) else "verify"


def toolchain(language: str, runner: FakeTools) -> Any:
    """Return the language's toolchain built as the runner builds it: the pinned ttmlir-opt and ttmlir-translate."""
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES[language]).factory
    return cls(executable=OPT, runner=runner, tools={"TTMLIR_TRANSLATE": TRANSLATE})


def build(tmp_path: Path, runner: FakeTools, language: str = "ttir", files: dict[str, str] | None = None) -> Any:
    """Build `files` (default: the one SYNTHETIC module) in a fresh directory; return the BuildResult."""
    workdir = tmp_path / "build"
    workdir.mkdir(parents=True)
    return toolchain(language, runner).build(files or {MODULE: TTIR_TEXT}, workdir)


def errors(result: Any) -> list[Any]:
    """Return the error diagnostics of a build."""
    return [item for item in result.diagnostics if item.severity == "error"]


# ---------------------------------------------------------------------------
# Registration and pins


@pytest.mark.parametrize("language", sorted(NAMES))
def test_each_toolchain_is_registered_compile_only(language: str) -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES[language]).factory
    assert cls.__module__ == "lassi.toolchains.ttir_compile" and cls.name == NAMES[language]
    assert cls.capabilities == frozenset({"diagnostics", "emits_warnings"}), "nothing an executor reads to run it"
    assert not hasattr(cls, "check_tree") and "host_compute_guard" not in cls.capabilities


@pytest.mark.parametrize("language", sorted(NAMES))
def test_each_runs_the_pinned_tt_mlir_tools(language: str) -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES[language]).factory
    pin = pins_module.read_pin("tt-mlir")
    assert cls.PIN == "tt-mlir" and cls.PIN_BIN.format_map(pin) == pin["TTMLIR_OPT"] == "build/tt-mlir/bin/ttmlir-opt"
    assert dict(cls.TOOL_PINS) == {"tt-mlir": ("TTMLIR_TRANSLATE",)}
    assert pin["TTMLIR_TRANSLATE"] == "build/tt-mlir/bin/ttmlir-translate"
    assert callable(cls.check_tool_pin)
    with pytest.raises(ValueError, match="tool pin"):
        cls.check_tool_pin("tt-metal", Path("."), pin)


def test_the_commands_are_the_spikes() -> None:
    assert ttir_compile.verify_command(OPT, MODULE, f"{WORK}/probe.verified.mlir") == [
        OPT, MODULE, "-o", f"{WORK}/probe.verified.mlir"]
    assert ttir_compile.lower_command(OPT, MODULE, f"{WORK}/probe.ttmetal.mlir") == [
        OPT, "--ttir-to-ttmetal-pipeline=enable-multicast-inference=false", MODULE, "-o", f"{WORK}/probe.ttmetal.mlir"]
    assert ttir_compile.translate_command(TRANSLATE, f"{WORK}/probe.ttmetal.mlir", f"{WORK}/probe.ttm") == [
        TRANSLATE, "--ttmetal-to-flatbuffer", f"{WORK}/probe.ttmetal.mlir", "-o", f"{WORK}/probe.ttm"]


# ---------------------------------------------------------------------------
# Builds


def test_a_module_that_compiles_gives_the_flatbuffer(tmp_path: Path) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify", "lower", "translate"]
    workdir = tmp_path / "build"
    assert result.artifact == workdir / WORK / "probe.ttm"
    assert errors(result) == [] and result.stderr_ref == ATTACHMENT
    log = (workdir / ATTACHMENT).read_text(encoding="utf-8")
    assert log.count("$ ") == 3 and "--ttmetal-to-flatbuffer" in log
    assert (workdir / MODULE).read_text() == TTIR_TEXT


def test_a_verify_error_is_parsed_and_stops_the_build(tmp_path: Path) -> None:
    stderr = (
        "probe.mlir:3:8: error: 'ttir.maximum' op expected 2 operands, but found 3\n"
        '  %r = "ttir.maximum"(%x, %y, %y) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>\n'
        "       ^\n"
    )
    runner = FakeTools(answers={"verify": (1, stderr)})
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify"]
    assert result.artifact is None
    (found,) = errors(result)
    assert (found.stage, found.file, found.line, found.column) == ("verify", MODULE, 3, 8)
    assert "expected 2 operands" in found.message


def test_an_assertion_abort_is_an_error_with_the_assertion_text(tmp_path: Path) -> None:
    stderr = (
        "ttmlir-opt: /src/include/ttmlir/AffineMapUtils.h:268: mlir::AffineMap calculateReblockMapForGrid(): "
        "Assertion `ok && \"New grid shape must evenly divide tensor shape\"' failed.\n"
        "PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.\n"
        "Stack dump:\n"
        "0.\tProgram arguments: ttmlir-opt probe.mlir\n"
    )
    runner = FakeTools(answers={"lower": (134, stderr)})
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify", "lower"]
    assert result.artifact is None
    (found,) = errors(result)
    assert (found.stage, found.code, found.file) == ("lower", "abort", MODULE)
    assert "exit status 134" in found.message
    assert "New grid shape must evenly divide tensor shape" in found.message
    assert "calculateReblockMapForGrid" in found.message and not found.message.startswith("ttmlir-opt: /src")


def test_a_popen_sigabrt_reads_as_an_abort_too(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(answers={"lower": (-6, "")}))
    (found,) = errors(result)
    assert found.code == "abort" and "printed no assertion text" in found.message


def test_a_lowering_error_names_the_module(tmp_path: Path) -> None:
    stderr = "probe.mlir:3:8: error: failed to legalize operation 'ttir.maximum' that was explicitly marked illegal\n"
    result = build(tmp_path, FakeTools(answers={"lower": (1, stderr)}))
    (found,) = errors(result)
    assert (found.stage, found.file, found.line) == ("lower", MODULE, 3)


def test_a_translate_error_in_the_lowered_module_loses_its_place(tmp_path: Path) -> None:
    stderr = (f"{WORK}/probe.ttmetal.mlir:12:3: error: function is missing required 'tt.function_type' attribute "
              "during flatbuffer translation\n")
    result = build(tmp_path, FakeTools(answers={"translate": (1, stderr)}))
    (found,) = errors(result)
    assert found.file is None and found.line is None
    assert f"at {WORK}/probe.ttmetal.mlir:12:3" in found.message and "not in your file" in found.message


def test_a_failure_with_no_parsed_error_gets_an_exit_status_error(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(answers={"translate": (3, "something odd\n")}))
    (found,) = errors(result)
    assert found.code == "exit-status" and "status 3" in found.message


def test_a_module_with_no_device_program_fails(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(lowered="module {\n  func.func @probe() { return }\n}\n"))
    assert result.artifact is None
    (found,) = errors(result)
    assert found.code == "no-device-program" and "ttmetal.enqueue_program" in found.message


def size_prefixed(body: bytes) -> bytes:
    """A size-prefixed flatbuffer as the pinned ttmlir-translate writes it: the size, the root offset, then TTM0."""
    data = b"\x20\x00\x00\x00TTM0" + body
    return len(data).to_bytes(4, "little") + data


def test_a_size_prefixed_flatbuffer_as_the_pin_writes_it_passes(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(flatbuffer=size_prefixed(bytes(64))))
    assert errors(result) == [] and result.artifact is not None


def test_a_file_that_is_no_flatbuffer_fails(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(flatbuffer=b"not a flatbuffer"))
    assert result.artifact is None
    (found,) = errors(result)
    assert found.code == "no-artifact" and "TTM0" in found.message


def test_the_identifier_is_read_after_the_size_prefix(tmp_path: Path) -> None:
    # The pinned translation writes size-prefixed flatbuffers (TTMetalToFlatbuffer.cpp:1068), so the identifier sits at
    # bytes 8 to 12; TTM0 at bytes 4 to 8 is no TT-Metal flatbuffer of this pin (sweep13's first launch found the old
    # offset; commit 7d3f3e8, task P17.49).
    assert ttir_compile.IDENTIFIER_SPAN == slice(8, 12)
    result = build(tmp_path, FakeTools(flatbuffer=b"\x10\x00\x00\x00TTM0" + bytes(16)))
    assert result.artifact is None and errors(result)[0].code == "no-artifact"


def test_the_ttir_level_refuses_d2m_ops_and_needs_a_ttir_op(tmp_path: Path) -> None:
    printed = TTIR_TEXT + "%z = d2m.empty() : tensor<1x1xf32>\n"
    result = build(tmp_path, FakeTools(printed=printed))
    (found,) = errors(result)
    assert (found.code, found.stage) == ("wrong-level", "verify") and "d2m.empty" in found.message
    result = build(tmp_path / "again", FakeTools(printed="module {}\n"))
    assert errors(result)[0].code == "wrong-level"


def test_the_d2m_level_needs_a_generic_and_refuses_ttir_ops(tmp_path: Path) -> None:
    good = FakeTools(printed="module {\n  %0 = d2m.generic {grid = #ttcore.grid<1x1>}\n}\n")
    result = build(tmp_path, good, "d2m", {MODULE: D2M_TEXT})
    assert errors(result) == [] and result.artifact is not None
    mixed = FakeTools(printed="module {\n  %0 = d2m.generic {}\n  %1 = \"ttir.maximum\"(%a, %b)\n}\n")
    result = build(tmp_path / "mixed", mixed, "d2m", {MODULE: D2M_TEXT})
    (found,) = errors(result)
    assert found.code == "wrong-level" and "ttir.maximum" in found.message
    plain = FakeTools(printed="module {\n  %0 = d2m.to_layout %a, %b\n}\n")
    result = build(tmp_path / "plain", plain, "d2m", {MODULE: D2M_TEXT})
    assert errors(result)[0].code == "wrong-level" and "d2m.generic" in errors(result)[0].message


def test_other_files_and_a_second_module_are_refused_before_anything_runs(tmp_path: Path) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner, files={MODULE: TTIR_TEXT, "notes.txt": "x\n", "other.mlir": TTIR_TEXT})
    assert runner.calls == [] and result.artifact is None
    assert sorted(item.code for item in errors(result)) == ["module-count", "not-a-module"]
    result = build(tmp_path / "work", runner, files={f"{WORK}/probe.mlir": TTIR_TEXT})
    assert runner.calls == [] and errors(result)[0].code == "bad-path"


def test_no_module_is_a_no_sources_error(tmp_path: Path) -> None:
    runner = FakeTools()
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = toolchain("ttir", runner).build({}, workdir)
    assert runner.calls == [] and errors(result)[0].code == "no-sources"


def test_the_assertion_reader() -> None:
    text = "x\nprog: a.h:1: f(): Assertion `a' failed.\n"
    assert ttir_compile.assertion_text(text) == "a.h:1: f(): Assertion `a' failed."
    assert ttir_compile.assertion_text("no assertion here\n") is None
